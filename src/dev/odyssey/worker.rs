//! The board's own thread, which runs the queued command-FIFO stores.
//!
//! The board decodes and draws behind the processor, so the model does its
//! decoding and drawing on a thread of its own rather than inside the
//! processor's store (interface.rs explains the two locks). The thread sleeps
//! until a store joins an empty queue, then runs the queue in slices: each
//! slice holds the board lock for at most `SLICE` of host time, and the
//! thread leaves both locks free for `GAP` before the next, so that a
//! processor access waiting for the board gets it.
//!
//! A slice never lets go of the board lock part-way: whatever a processor
//! may touch during one lives under the front lock.

use super::*;

use std::sync::Arc;
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
}

impl Budget {
    /// A slice on the board's thread: `SLICE` of host time, or `words`
    /// stores when not zero.
    pub fn slice(words: u32) -> Self {
        Budget { deadline: Some(Instant::now() + SLICE), words: (words != 0).then_some(words) }
    }

    /// Exactly `words` stores, or fewer if the queue runs dry.
    pub fn words(words: u32) -> Self {
        Budget { deadline: None, words: Some(words) }
    }

    /// Everything that waits.
    pub fn unlimited() -> Self {
        Budget { deadline: None, words: None }
    }

    /// Count one store against the budget; true when the slice must stop.
    pub fn spent(&mut self) -> bool {
        if let Some(w) = &mut self.words {
            *w = w.saturating_sub(1);
            if *w == 0 {
                return true;
            }
        }
        self.deadline.is_some_and(|d| Instant::now() >= d)
    }
}

impl BoardLocks {
    /// Start the board's thread.
    pub fn work_start(&self) {
        let mut f = self.front.lock();
        if f.worker.is_some() {
            return;
        }
        f.worker_exit = false;
        let handles = self.clone();
        f.worker = Some(
            std::thread::Builder::new()
                .name(ODYSSEY_WORK_THREAD.into())
                .spawn(move || handles.work_run())
                .expect("odyssey work thread"),
        );
    }

    /// The machine stops: finish the waiting work, then stop the board's
    /// thread, so no work is left behind a stopped machine.
    pub fn work_stop(&self) {
        {
            let mut s = self.board.lock();
            let waiting = s.front.lock().work_queue.len() as u32;
            s.work_step(&mut Budget::words(waiting));
        }
        let thread = {
            let mut f = self.front.lock();
            f.worker_exit = true;
            f.kick.notify_one();
            f.worker.take()
        };
        if let Some(t) = thread {
            let _ = t.join();
        }
    }

    fn work_run(&self) {
        let mut f = self.front.lock();
        let kick: Arc<parking_lot::Condvar> = f.kick.clone();
        loop {
            while f.work_queue.is_empty() && !f.worker_exit {
                f.worker_waiting = true;
                kick.wait(&mut f);
                f.worker_waiting = false;
            }
            if f.worker_exit {
                break;
            }
            MutexGuard::unlocked(&mut f, || std::thread::sleep(GAP));
            if f.worker_exit {
                break;
            }
            #[cfg(test)]
            let words = f.test_words;
            #[cfg(not(test))]
            let words = 0;
            f.slice = true;
            MutexGuard::unlocked(&mut f, || {
                let mut s = self.board.lock();
                s.work_step(&mut Budget::slice(words));
            });
            f.slice = false;
        }
    }
}

impl Board {
    pub fn work_pending(&mut self) -> bool {
        self.front.lock().pending()
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

        loop {
            let Some(q) = self.front.lock().work_take() else {
                break;
            };
            self.write_dispatch(q.addr as u64, q.value, q.size as u32);
            {
                let mut f = self.front.lock();
                let level = f.work_queue.len() as u32 + 1;
                f.fifo_level_changed(level);
            }
            if b.spent() {
                break;
            }
        }
        let f = self.front.lock();
        if !f.work_queue.is_empty() {
            crate::dlog_dev!(LogModule::Odyssey, "sgi_odyssey_work_behind: {} stores wait, the oldest for {:?}", f.work_queue.len(), f.work_queue[0].queued_at.elapsed());
        }
    }

    /// Finish the waiting work now, for an access at addr that must see it. The
    /// caller holds the board lock, so no slice runs. The stores queued when
    /// the drain starts run here; any a processor adds meanwhile are left to
    /// the board's thread.
    pub fn work_drain(&mut self, addr: u64) {
        let waiting = self.front.lock().work_queue.len() as u32;
        if waiting == 0 {
            return;
        }
        let start = std::time::Instant::now();
        self.work_step(&mut Budget::words(waiting));
        crate::dlog_dev!(LogModule::Odyssey, "sgi_odyssey_work: drained offset {:#x} in {:?}", addr, start.elapsed());
    }

    /// Reset clears the FIFO and abandons the waiting stores. The caller holds
    /// the board lock, so no slice runs.
    pub fn work_reset(&mut self) {
        let mut f = self.front.lock();
        let f = &mut *f;
        for q in f.work_queue.iter() {
            if q.owed {
                if let Some(pi) = &f.pi {
                    pi_gfx_owe_credits(pi.as_ref(), f.slot, -1);
                }
            }
        }
        f.work_queue.clear();
        f.work_in_fifo = 0;
    }
}

