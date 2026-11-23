//! The display's clock: one frame at a time, at the video format's rate.
//!
//! At each frame boundary the display completes the retrace requests that are
//! waiting and raises the video interrupt. The interrupt rises once a frame
//! even when no request is pending. The back end drains the raster jobs before
//! it queues a retrace request, so a request is satisfied when it is made (no
//! tickets: approximated), and the display completes it at the next boundary
//! (the sequence register changes at the retrace; when a real pipe changes it
//! is unknown, approximated here as the retrace). In a frame the requests are
//! completed first and the frame composed after, so a swap applied before a
//! request shows in the frame that completes it.
//!
//! The frames come from the machine's display refresh thread
//! (`crate::display`), which runs whether or not the pipe works, so they never
//! mark activity for the driver's watchdog. The head holds only the handle the
//! front end gave it and what it must complete, nothing of the pipe's state.
//!
//! Grades. The frame rate is 60 Hz until a video format's refresh rate can be
//! parsed from its load (approximated).

use super::super::super::handle::DisplayHandle;
use crate::display::{Composed, Head, HeadHandle, VideoTiming, Want};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::{Acquire, Relaxed, Release}};
use std::sync::Arc;
use std::time::Duration;

/// The default frame period: 60 Hz.
pub const FRAME_PERIOD: Duration = Duration::from_micros(16_667);

/// What the board and its display thread share.
#[derive(Default)]
pub struct Clock {
    /// Retrace requests waiting for a frame boundary: the epoch each was made in
    /// and its tag.
    pending: Mutex<Vec<(u64, u32)>>,
    /// Frames the display has run, and requests it has completed.
    frames: AtomicU64,
    completed: AtomicU64,
    /// The display is to hold still (a snapshot), and the lock every frame
    /// callback runs under, so holding still can wait one out.
    paused: AtomicBool,
    busy: Mutex<()>,
}

impl Clock {
    /// A retrace request made in `epoch`, to complete at the next frame.
    pub fn request(&self, epoch: u64, tag: u32) {
        self.pending.lock().push((epoch, tag));
    }

    /// Forget the waiting requests: a reset cancels them.
    pub fn cancel(&self) {
        self.pending.lock().clear();
    }

    pub fn frames(&self) -> u64 {
        self.frames.load(Relaxed)
    }

    pub fn completed(&self) -> u64 {
        self.completed.load(Relaxed)
    }

    /// One frame boundary: complete what waits, then raise the video
    /// interrupt if nothing did (completing a request raises it).
    pub fn frame(&self, handle: &DisplayHandle) {
        let waiting = std::mem::take(&mut *self.pending.lock());
        if waiting.is_empty() {
            handle.video_frame(handle.epoch());
        }
        for (epoch, tag) in &waiting {
            handle.retrace_done(*epoch, *tag);
        }
        self.completed.fetch_add(waiting.len() as u64, Relaxed);
        self.frames.fetch_add(1, Release);
    }

    /// Hold the display still, and wait out a frame callback in flight. A
    /// board with no display running has nothing to wait for.
    pub fn quiesce(&self, running: bool) {
        self.paused.store(true, Release);
        if running {
            drop(self.busy.lock());
        }
    }

    pub fn resume(&self) {
        self.paused.store(false, Release);
    }
}

/// The board on the refresh thread: the clock, the handle the retraces go
/// through, and the compositor.
struct Scan {
    clock: Arc<Clock>,
    handle: DisplayHandle,
    sink: Mutex<super::Compositor>,
}

impl Head for Scan {
    fn vblank_start(&self, _field: u32, _t_ns: u64) {
        let _busy = self.clock.busy.lock();
        if !self.clock.paused.load(Acquire) {
            self.clock.frame(&self.handle);
        }
    }

    fn changed(&self) -> bool {
        let _busy = self.clock.busy.lock();
        !self.clock.paused.load(Acquire) && self.sink.lock().drawn()
    }

    fn compose(&self, screen: &mut crate::disp::Rex3Screen, _want: Want) -> Composed {
        let _busy = self.clock.busy.lock();
        self.sink.lock().compose(screen);
        Composed::Frame
    }
}

/// The running display: its head on the refresh thread, and the board's
/// side of it.
pub struct DisplayThread {
    head: Option<Arc<HeadHandle>>,
    /// The head holds this weakly.
    _scan: Arc<Scan>,
}

impl DisplayThread {
    /// Start frames every `period` on the refresh thread of `sink`'s display.
    pub fn spawn(clock: Arc<Clock>, handle: DisplayHandle, period: Duration, sink: super::Compositor) -> DisplayThread {
        let head = sink.display().head().clone();
        let scan = Arc::new(Scan { clock, handle, sink: Mutex::new(sink) });
        let weak: std::sync::Weak<dyn Head> = Arc::downgrade(&(scan.clone() as Arc<dyn Head>));
        head.attach(weak);
        if period != FRAME_PERIOD {
            // A frame of a million clocks: one clock is a millionth of it.
            let clock_hz = 1_000_000u64 * 1_000_000_000 / (period.as_nanos() as u64).max(1);
            head.set_timing(VideoTiming { h_total: 1000, v_total: 1000, pixel_clock_hz: clock_hz, vblank_lines: 40, fields: 1 });
        } else {
            head.set_timing(VideoTiming::HZ60);
        }
        head.start();
        DisplayThread { head: Some(head), _scan: scan }
    }

    /// End the frames; returns once none is running. Called with the front
    /// end's pipe slot taken and no register-file lock held: a frame takes
    /// only that lock, in a handle call that sends an interrupt packet after
    /// dropping it, and never the slot, so this cannot wait on a lock its
    /// caller holds.
    pub fn stop(&mut self) {
        if let Some(head) = self.head.take() {
            head.stop();
        }
    }
}

impl Drop for DisplayThread {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::kona::tests::{test_cfg, TestHost};
    use crate::dev::kona::{Kona, Reply};
    use crate::traits::{Device, BUS_OK};

    const INT_ENABLE: u64 = 0x800230;
    const INT_CAUSE: u64 = 0x800238;
    const RETRACE_TAG: u64 = 0x800248;

    /// A compositor over a tiny store, for a thread that has nothing to show.
    fn quiet() -> crate::dev::kona::back::display::Compositor {
        use crate::dev::kona::back::display::{Compositor, screen::Display, tables::Tables};
        use crate::dev::kona::back::store::PixelStore;
        Compositor::new(Arc::new(PixelStore::new(4, 4)), 1, Arc::default(), Arc::new(Mutex::new(Tables::default())), Arc::new(Display::detached()))
    }

    /// A started board whose back end holds the handle; the board's host to read
    /// registers and see the packets.
    fn board() -> (Arc<crate::dev::kona::Kona>, Arc<TestHost>, DisplayHandle) {
        let kept = Arc::new(Mutex::new(None));
        let host = TestHost::new();
        let kona = Kona::new(test_cfg(), host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
        let keep = kept.clone();
        kona.shared.pipe.lock().backend.hook = Some(crate::dev::kona::back::TestHook { job: Box::new(|_, _| Some(Reply::Done)), events: Arc::default(), on_attach: Some(Box::new(move |h| *keep.lock() = Some(h))) });
        for (off, v) in [(0x34u64, 0x98000u32), (0x3c, 0x180_0090), (0x5c, 0x3e_2210), (INT_ENABLE, 0xf7c)] {
            assert_eq!(kona.test_write(off, 4, v as u64), BUS_OK);
        }
        kona.start();
        let handle = kept.lock().clone().expect("attach gave a handle");
        (kona, host, handle)
    }

    fn read(kona: &crate::dev::kona::Kona, off: u64) -> u32 {
        kona.bus_read(off, 4).1 as u32
    }

    /// A frame completes a waiting request (the tag register changes, one video
    /// interrupt goes out) and, with nothing waiting, still raises the video
    /// interrupt once the driver has acknowledged the last.
    #[test]
    fn a_frame_completes_requests_and_raises_video() {
        let (kona, host, h) = board();
        let clock = Clock::default();
        clock.request(h.epoch(), 0x42);
        clock.frame(&h);
        assert_eq!(read(&kona, RETRACE_TAG), 0x42);
        assert_eq!(host.sent().len(), 1);
        assert_eq!((clock.frames(), clock.completed()), (1, 1));
        clock.frame(&h);
        assert_eq!(host.sent().len(), 1, "not again before the acknowledge");
        kona.bus_write(INT_CAUSE, 4, 8);
        clock.frame(&h);
        assert_eq!(host.sent().len(), 2);
        assert_eq!(read(&kona, RETRACE_TAG), 0x42, "a frame with no request leaves the tag");
        kona.stop();
    }

    /// A reset cancels the waiting requests, and one made in an earlier epoch is
    /// dropped by the front end.
    #[test]
    fn a_reset_cancels_waiting_requests() {
        let (kona, _host, h) = board();
        let clock = Clock::default();
        clock.request(h.epoch(), 0x7);
        clock.cancel();
        clock.frame(&h);
        assert_eq!(read(&kona, RETRACE_TAG), 0);
        let old = h.epoch();
        kona.stop();
        crate::traits::Resettable::power_on(&*kona);
        clock.request(old, 0x9);
        clock.frame(&h);
        assert_eq!(read(&kona, RETRACE_TAG), 0, "a stale epoch is dropped");
    }

    /// The thread runs frames by itself and completes a request; stopping joins it.
    #[test]
    fn the_thread_completes_a_request_and_joins() {
        let (kona, _host, h) = board();
        let clock = Arc::new(Clock::default());
        let mut t = DisplayThread::spawn(clock.clone(), h.clone(), Duration::from_millis(1), quiet());
        clock.request(h.epoch(), 0x55);
        assert!(crate::test_board::until(|| read(&kona, RETRACE_TAG) == 0x55));
        assert_eq!(read(&kona, RETRACE_TAG), 0x55);
        // The counters follow the completion within the same frame.
        assert!(crate::test_board::until(|| clock.frames() >= 1));
        t.stop();
        let f = clock.frames();
        std::thread::yield_now();
        assert_eq!(clock.frames(), f, "no frames after the join");
        kona.stop();
    }

    /// A quiesced thread holds still between frames and goes on when resumed.
    #[test]
    fn a_quiesced_thread_holds_still() {
        let (kona, _host, h) = board();
        let clock = Arc::new(Clock::default());
        let mut t = DisplayThread::spawn(clock.clone(), h, Duration::from_millis(1), quiet());
        clock.quiesce(true);
        let f = clock.frames();
        for _ in 0..10_000 {
            std::thread::yield_now();
        }
        assert_eq!(clock.frames(), f);
        clock.resume();
        assert!(crate::test_board::until(|| clock.frames() > f));
        t.stop();
        kona.stop();
    }
}
