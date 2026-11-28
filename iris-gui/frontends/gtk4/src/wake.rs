//! The shell's `Notify`: a worker event, a job or a frame queues one tick on
//! the GLib main context. Coalesced, and always queued, never run inline: a
//! wake can come from the main thread inside a borrow (a job cancelled from a
//! handler), and from the worker and the graphics threads at any time.

use std::sync::atomic::{AtomicBool, Ordering};

use iris_gui_core::controller::Notify;

#[derive(Default)]
pub struct GlibWake {
    pending: AtomicBool,
}

impl GlibWake {
    /// The tick ran: the next wake queues again.
    pub fn ticked(&self) {
        self.pending.store(false, Ordering::Release);
    }
}

impl Notify for GlibWake {
    fn wake(&self) {
        if self.pending.swap(true, Ordering::AcqRel) {
            return;
        }
        // `idle_add_once` is thread-safe and attaches to the default main
        // context, which is the one the application runs; the local variants
        // would panic off the main thread.
        gtk4::glib::source::idle_add_once(crate::app::tick);
    }
}
