//! The board's own thread, which runs the queued command-FIFO stores.
//!
//! The board decodes and draws behind the processor, so the model does its
//! decoding and drawing on a thread of its own rather than inside the
//! processor's store (interface.rs explains the two locks, fifo.rs the
//! queue). The thread sleeps until a store joins an empty queue, then runs
//! the queue in slices: each slice holds the board lock for at most `SLICE`
//! of host time and hands it fairly to a waiting processor access; the
//! thread pauses `GAP` before each slice so a burst of stores lands first.
//!
//! A slice never lets go of the board lock part-way: whatever a processor
//! may touch during one lives under the front lock.

use super::*;

use std::sync::atomic::Ordering::SeqCst;
use std::time::{Duration, Instant};

use parking_lot::MutexGuard;

/// The board thread's name.
pub const ODYSSEY_WORK_THREAD: &str = "odyssey-work";

/// One slice of host time. The guest's scheduler clock ticks every 10 ms,
/// so a slice stays well below that.
pub const SLICE: Duration = Duration::from_millis(1);

/// The pause before each slice, with both locks free. A thread that lets go
/// of a lock and takes it straight back usually wins it again, so without
/// the pause a processor waiting for the board could wait out many slices.
pub const GAP: Duration = Duration::from_micros(100);

/// How much one slice may do: host time, and for tests a count of stores.
/// A slice checks the budget between stores, so it always runs at least one
/// and stops at the first store boundary past the deadline.
pub struct Budget {
    deadline: Option<Instant>,
    words: Option<u32>,
    count: u32,
}

impl Budget {
    /// A slice on the board's thread: `SLICE` of host time, or `words`
    /// stores when not zero.
    pub fn slice(words: u32) -> Self {
        Budget { deadline: Some(Instant::now() + SLICE), words: (words != 0).then_some(words), count: 0 }
    }

    /// Exactly `stores` stores, or fewer if the queue runs dry.
    pub fn stores(stores: u32) -> Self {
        Budget { deadline: None, words: Some(stores), count: 0 }
    }

    /// Everything that waits.
    pub fn unlimited() -> Self {
        Budget { deadline: None, words: None, count: 0 }
    }

    /// Count one store against the budget; true when the slice must stop.
    pub fn spent(&mut self) -> bool {
        if let Some(w) = &mut self.words {
            *w = w.saturating_sub(1);
            if *w == 0 {
                return true;
            }
        }
        // The clock is read every 16 stores; a store costs far less than a
        // slice.
        self.count = self.count.wrapping_add(1);
        self.count % 16 == 0 && self.deadline.is_some_and(|d| Instant::now() >= d)
    }
}

impl BoardLocks {
    /// Start the board's thread, the front end, and the raster unit's
    /// behind it.
    pub fn work_start(&self) {
        let mut thread = self.fifo.thread.lock();
        if thread.is_some() {
            return;
        }
        self.raster.start();
        self.fifo.start();
        self.fifo.running.store(true, SeqCst);
        let handles = self.clone();
        *thread = Some(
            std::thread::Builder::new()
                .name(ODYSSEY_WORK_THREAD.into())
                .spawn(move || handles.work_run())
                .expect("odyssey work thread"),
        );
    }

    /// The machine stops: stop the board's thread, then finish the waiting
    /// work, so no work is left behind a stopped machine.
    pub fn work_stop(&self) {
        let thread = self.fifo.thread.lock().take();
        if let Some(t) = thread {
            self.fifo.stop();
            let _ = t.join();
        }
        self.fifo.running.store(false, SeqCst);
        let mut s = self.board.lock();
        s.work_step(&mut Budget::unlimited());
        // Then the raster work those stores handed on.
        self.raster.stop();
    }

    fn work_run(&self) {
        while self.fifo.wait_for_work() {
            // Let a burst of stores land before the slice runs, unless the
            // processor already waits for it.
            if !self.fifo.take_hurry() {
                std::thread::sleep(GAP);
            }
            if self.fifo.exiting() {
                break;
            }
            #[cfg(test)]
            let words = self.fifo.test_words.load(SeqCst);
            #[cfg(not(test))]
            let words = 0;
            self.fifo.slice.store(true, SeqCst);
            {
                let mut s = self.board.lock();
                crate::pf_value!(self.fifo.pf_level, crate::perfetto::events::ODSY_CFIFO_LEVEL, self.fifo.level());
                crate::pf_begin!(crate::perfetto::events::ODSY_SLICE, self.fifo.level());
                s.work_step(&mut Budget::slice(words));
                crate::pf_end!(crate::perfetto::events::ODSY_SLICE);
                crate::pf_value!(self.fifo.pf_level, crate::perfetto::events::ODSY_CFIFO_LEVEL, self.fifo.level());
                // A processor access waiting for the board gets it next.
                MutexGuard::unlock_fair(s);
            }
            self.fifo.slice.store(false, SeqCst);
        }
    }
}

impl Board {
    pub fn work_pending(&mut self) -> bool {
        self.fifo.pending()
    }

    /// Run the oldest waiting stores until the budget is spent. The caller
    /// holds the board lock and not the front lock: the board's thread for a
    /// slice, or an access that drains.
    pub fn work_step(&mut self, b: &mut Budget) {
        #[cfg(test)]
        {
            // A slice that holds the board as a long draw would.
            let sleep_ms = std::mem::take(&mut self.test_slice_sleep_ms);
            if sleep_ms != 0 {
                std::thread::sleep(std::time::Duration::from_millis(sleep_ms as u64));
            }
        }

        let fifo = self.fifo.clone();
        while let Some((q, after)) = fifo.pop(&self.front) {
            self.write_dispatch(q.addr as u64, q.value, q.size as u32);
            // Falling below the low-water mark as the store retires latches
            // the low-water flag, whatever the enable mask says.
            if after + q.words() >= ODSY_CFIFO_LOW_WATER && fifo.level() < ODSY_CFIFO_LOW_WATER {
                let mut f = self.front.lock();
                f.intr_flags |= ODSY_INTR_FIFO_LOW_WATER;
                crate::pf_instant!(crate::perfetto::events::ODSY_CFIFO_LOW_WATER, fifo.level());
                f.update_irq();
            }
            if b.spent() {
                break;
            }
        }
        if fifo.pending() {
            crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: worker behind: {} stores wait", fifo.stores());
        }
    }

    /// Finish the waiting work now, for an access at addr that must see it. The
    /// caller holds the board lock, so no slice runs. The stores queued when
    /// the drain starts run here; any a processor adds meanwhile are left to
    /// the board's thread.
    pub fn work_drain(&mut self, addr: u64) {
        let waiting = self.fifo.stores();
        if waiting == 0 {
            return;
        }
        let start = std::time::Instant::now();
        crate::pf_scope!(crate::perfetto::events::ODSY_DRAIN, waiting);
        self.work_step(&mut Budget::stores(waiting));
        crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: worker: drained offset {:#x} in {:?}", addr, start.elapsed());
    }

    /// Reset clears the FIFO and abandons the waiting stores. The caller holds
    /// the board lock, so no slice runs.
    pub fn work_reset(&mut self) {
        self.fifo.clear(&self.front);
    }
}
