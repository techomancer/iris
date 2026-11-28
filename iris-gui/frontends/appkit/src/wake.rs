//! The shell's `Notify`: a worker event, a job or a frame queues one tick on
//! the main queue. Coalesced, and always queued, never run inline: a wake can
//! come from the main thread inside a borrow (a job cancelled from a handler).

use std::sync::atomic::{AtomicBool, Ordering};

use iris_gui_core::controller::Notify;

#[derive(Default)]
pub struct MacWake {
    pending: AtomicBool,
}

impl MacWake {
    /// The tick ran: the next wake queues again.
    pub fn ticked(&self) {
        self.pending.store(false, Ordering::Release);
    }
}

impl Notify for MacWake {
    fn wake(&self) {
        if self.pending.swap(true, Ordering::AcqRel) {
            return;
        }
        dispatch2::DispatchQueue::main().exec_async(crate::app::tick);
    }
}
