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
//! The thread, `kona-display`, runs whether or not the pipe works, so it never
//! marks activity for the driver's watchdog. It only holds the handle the front
//! end gave it and what it must complete, nothing of the pipe's state.
//!
//! Grades. The frame rate is 60 Hz until a video format's refresh rate can be
//! parsed from its load (approximated).

use super::super::super::handle::DisplayHandle;
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::{Acquire, Relaxed, Release}};
use std::sync::Arc;
use std::thread::JoinHandle;
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
    /// The thread is to end, and whether it is to hold still (a snapshot).
    stop: AtomicBool,
    pause: Mutex<Pause>,
    changed: Condvar,
}

#[derive(Default, PartialEq, Eq)]
struct Pause {
    /// Asked to hold still, and whether it has.
    wanted: bool,
    held: bool,
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

    /// Hold the thread still at a frame boundary, and wait until it has. A
    /// board with no thread has nothing to hold.
    pub fn quiesce(&self, running: bool) {
        let mut p = self.pause.lock();
        p.wanted = true;
        self.changed.notify_all();
        while running && !p.held && !self.stop.load(Acquire) {
            self.changed.wait(&mut p);
        }
    }

    pub fn resume(&self) {
        let mut p = self.pause.lock();
        p.wanted = false;
        self.changed.notify_all();
    }

    fn hold_if_asked(&self) {
        let mut p = self.pause.lock();
        while p.wanted && !self.stop.load(Acquire) {
            p.held = true;
            self.changed.notify_all();
            self.changed.wait(&mut p);
        }
        p.held = false;
    }
}

/// The display thread and what it joins on.
pub struct DisplayThread {
    clock: Arc<Clock>,
    join: Option<JoinHandle<()>>,
}

impl DisplayThread {
    /// Start `kona-display`, running a frame every `period`.
    pub fn spawn(clock: Arc<Clock>, handle: DisplayHandle, period: Duration, mut sink: super::Compositor) -> DisplayThread {
        clock.stop.store(false, Release);
        let c = clock.clone();
        let join = std::thread::Builder::new()
            .name("kona-display".into())
            .spawn(move || {
                // Frames are paced from a fixed start so a slow one does not
                // stretch the rate.
                let mut next = std::time::Instant::now() + period;
                while !c.stop.load(Acquire) {
                    c.hold_if_asked();
                    let now = std::time::Instant::now();
                    // A pause or a stalled host leaves `next` far behind: go on
                    // from now rather than run the missed frames back to back.
                    if now.saturating_duration_since(next) > period {
                        next = now;
                    }
                    if now < next {
                        // Parked, not slept, so `stop` wakes it at once.
                        std::thread::park_timeout(next - now);
                        if c.stop.load(Acquire) {
                            break;
                        }
                        continue;
                    }
                    next += period;
                    c.frame(&handle);
                    sink.frame();
                }
                sink.finish();
            })
            .expect("kona display thread");
        DisplayThread { clock, join: Some(join) }
    }

    /// End the thread and wait for it. Called with the front end's pipe slot
    /// taken and no register-file lock held: the thread takes only that lock, in
    /// a handle call that sends an interrupt packet after dropping it, and never
    /// the slot, so the join cannot wait on a lock its caller holds.
    pub fn stop(&mut self) {
        let Some(join) = self.join.take() else { return };
        self.clock.stop.store(true, Release);
        {
            let _p = self.clock.pause.lock();
            self.clock.changed.notify_all();
        }
        join.thread().unpark();
        let _ = join.join();
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

    /// Wait for `cond`, up to a deadline, not a count of tries.
    fn wait_for(mut cond: impl FnMut() -> bool) -> bool {
        let end = std::time::Instant::now() + Duration::from_secs(30);
        while std::time::Instant::now() < end {
            if cond() {
                return true;
            }
            std::thread::yield_now();
        }
        cond()
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
        assert!(wait_for(|| read(&kona, RETRACE_TAG) == 0x55));
        assert_eq!(read(&kona, RETRACE_TAG), 0x55);
        // The counters follow the completion within the same frame.
        assert!(wait_for(|| clock.frames() >= 1));
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
        assert!(wait_for(|| clock.frames() > f));
        t.stop();
        kona.stop();
    }
}
