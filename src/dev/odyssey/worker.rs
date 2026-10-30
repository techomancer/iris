//! The board's own thread, which runs the queued command-FIFO stores.
//!
//! The board decodes and draws behind the processor, so the model does its
//! decoding and drawing on a thread of its own rather than inside the
//! processor's store (interface.rs explains the front registers, fifo.rs the
//! queue). The thread sleeps until a store joins an empty queue, pauses
//! `GAP` so a burst of stores lands first, then runs the queue until it is
//! empty. It owns the board's state while it runs: a processor access that
//! must see the stores is answered busy until the queue and the raster queue
//! are empty (`BoardLocks::bus_read`), and whatever a processor may touch
//! meanwhile lives in the host interface.


use super::board::Board;
use super::fifo::Run;
use super::mmio::BoardLocks;
use std::sync::atomic::Ordering::SeqCst;
use std::time::Duration;

/// The board thread's name.
pub(super) const ODYSSEY_WORK_THREAD: &str = "odyssey-work";

/// The pause before the board's thread runs a burst, so the stores of the
/// burst land first and one run takes them together. A processor that
/// waits on the thread cuts it short (`CommandFifo::hurry`): an access the
/// bus refused.
pub(super) const GAP: Duration = Duration::from_micros(100);

/// How much one run may do: for tests a count of stores. The budget is
/// checked between stores, so a run always runs at least one.
pub(super) struct Budget {
    words: Option<u32>,
}

impl Budget {
    /// A run on the board's thread: everything that waits, or `words`
    /// stores when not zero.
    pub fn slice(words: u32) -> Self {
        Budget { words: (words != 0).then_some(words) }
    }

    /// Exactly `stores` stores, or fewer if the queue runs dry.
    pub fn stores(stores: u32) -> Self {
        Budget { words: Some(stores) }
    }

    /// Everything that waits.
    pub fn unlimited() -> Self {
        Budget { words: None }
    }

    /// Count one store against the budget; true when the run must stop.
    pub fn spent(&mut self) -> bool {
        if let Some(w) = &mut self.words {
            *w = w.saturating_sub(1);
            return *w == 0;
        }
        false
    }
}

impl BoardLocks {
    /// Start the board's thread, the front end, and the raster unit's
    /// behind it.
    ///
    /// Devices start before the processors, so no processor store is in
    /// flight on the unthreaded path when `running` is set below.
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
        // SAFETY: the board's thread is joined; the caller stops the
        // processors' threads before the devices.
        let s = unsafe { &mut *self.board.get() };
        s.work_step(&mut Budget::unlimited());
        // Then the raster work those stores handed on.
        self.raster.stop();
    }

    fn work_run(&self) {
        while self.fifo.wait_for_work() {
            // Let a burst of stores land before the run, unless the
            // processor waits for it.
            self.fifo.gap();
            if self.fifo.exiting() {
                break;
            }
            #[cfg(test)]
            let words = self.fifo.test_words.load(SeqCst);
            #[cfg(not(test))]
            let words = 0;
            self.fifo.slice.store(true, SeqCst);
            {
                // SAFETY: this is the board's thread, which owns the state
                // while it runs stores (BoardCell).
                let s = unsafe { &mut *self.board.get() };
                crate::pf_value!(self.fifo.pf_level, crate::perfetto::events::ODSY_CFIFO_LEVEL, self.fifo.level());
                crate::pf_begin!(crate::perfetto::events::ODSY_SLICE, self.fifo.level());
                s.work_step(&mut Budget::slice(words));
                crate::pf_end!(crate::perfetto::events::ODSY_SLICE);
                crate::pf_value!(self.fifo.pf_level, crate::perfetto::events::ODSY_CFIFO_LEVEL, self.fifo.level());
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
    /// is the board's thread, or an access that drains with that thread
    /// stopped; it does not hold the front lock.
    pub fn work_step(&mut self, b: &mut Budget) {
        #[cfg(test)]
        {
            // A run that takes as long as a long draw would.
            let sleep_ms = std::mem::take(&mut self.test_slice_sleep_ms);
            if sleep_ms != 0 {
                std::thread::sleep(std::time::Duration::from_millis(sleep_ms as u64));
            }
        }

        let fifo = self.fifo.clone();
        let mut run = Run::new();
        'slice: loop {
            fifo.take_run(&mut run);
            if run.stores().is_empty() {
                break;
            }
            let mut ran = 0;
            let mut spent = false;
            for q in run.stores() {
                self.write_dispatch(q.addr as u64, q.value, q.size as u32);
                ran += 1;
                if b.spent() {
                    spent = true;
                    break;
                }
            }
            // The low-water flag latches as the stores retire below the
            // mark (CommandFifo::retire).
            fifo.retire(&self.front, &run, ran);
            if spent {
                break 'slice;
            }
        }
        if fifo.pending() {
            crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: worker behind: {} stores wait", fifo.stores());
        }
    }

    /// Finish the waiting work now, for an access at addr that must see it,
    /// with the board's thread stopped. The stores queued when the drain
    /// starts run here; any a processor adds meanwhile are left to the
    /// board's thread.
    pub fn work_drain(&mut self, addr: u64) {
        let waiting = self.fifo.stores();
        // With the board's thread running the stores are its to run: a
        // second runner would be two threads in the state. The bus answers a
        // waiting access busy until it has (BoardLocks::bus_read).
        if waiting == 0 || self.fifo.running.load(SeqCst) {
            return;
        }
        let start = std::time::Instant::now();
        crate::pf_scope!(crate::perfetto::events::ODSY_DRAIN, waiting);
        self.work_step(&mut Budget::stores(waiting));
        crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: worker: drained offset {:#x} in {:?}", addr, start.elapsed());
    }

    /// Reset clears the FIFO and abandons the waiting stores.
    pub fn work_reset(&mut self) {
        self.fifo.clear();
    }
}
