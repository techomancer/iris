//! What a back end's own threads may do to the front end.
//!
//! `BackPort` reaches the front end only inside `KonaBackend::job`, on the Host
//! Interface Processor's thread. The display runs on a thread of its own, at
//! the frame rate, and has to complete retrace requests and raise the video
//! interrupt when a frame starts. The handle is how it does that. It holds the
//! register file, which owns nothing of the back end, so there is no cycle.
//!
//! A handle call takes the epoch its ticket was made in (`epoch`). A reset
//! begins a new epoch, so a thread that was in the middle of completing a
//! retrace when the reset happened cannot raise a cause the reset cleared.

use super::front::Host;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

/// The front end as a back end's display thread may use it. Cheap to clone.
#[derive(Clone)]
pub struct DisplayHandle {
    host: Arc<Host>,
}

impl DisplayHandle {
    pub(super) fn new(host: &Arc<Host>) -> Self {
        DisplayHandle { host: host.clone() }
    }

    /// The epoch a ticket made now belongs to.
    pub fn epoch(&self) -> u64 {
        self.host.front.lock().epoch()
    }

    /// A retrace request made in `epoch` was honoured: the tag register reads
    /// `tag` and the video interrupt rises.
    pub fn retrace_done(&self, epoch: u64, tag: u32) {
        self.host.with_front(|f| ((), f.retrace_in(epoch, tag)));
    }

    /// A frame started in `epoch`: the video interrupt rises, with no tag.
    pub fn video_frame(&self, epoch: u64) {
        self.host.with_front(|f| ((), f.video_frame(epoch)));
    }

    /// The back end is working: the driver's watchdog sees activity. For a
    /// drain loop or the raster thread while it works. Not for the display
    /// thread, which runs whether or not the pipe does: if it set the latch
    /// the watchdog could never fire.
    pub fn activity(&self) {
        self.host.activity.store(true, Relaxed);
    }
}
