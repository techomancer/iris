//! What the UI, CI socket and GUI front-end need from whichever graphics board
//! drives the screen (Newport/REX3 or GR2).
//!
//! Each board owns its display thread and composition (the status bar's
//! inputs come from the machine's shared `StatusSource`); this trait only
//! exposes the hand-off points: the renderer slot the front-end installs into, the
//! last-presented screen (screenshots, VNC), the hardware cursor, and the CPU
//! cycle counter.

use std::sync::atomic::{fence, AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::disp::Rex3Screen;
use crate::cpu::mips_core::CyclesPtr;
use crate::dev::ng1::rex3::Renderer;

pub trait GfxDisplay: Send + Sync {
    /// The board's [`DisplayStop`], which its stop and its display thread use.
    fn display_stop(&self) -> &DisplayStop;

    /// Slot the front-end installs its renderer into. The board's display
    /// thread presents every frame through whatever is installed here.
    fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>>;

    /// Last presented frame state. Screenshots read `rgba` (stride 2048).
    fn screen(&self) -> &Mutex<Rex3Screen>;

    /// Ask the display thread to read the next frame back into `screen().rgba`.
    fn request_screenshot(&self);

    /// True while a [`request_screenshot`](Self::request_screenshot) readback is
    /// still pending. Callers wait for this to clear (bounded) so they copy the
    /// requested frame rather than the one that happened to be current. The
    /// default is `false`: a board that does not consume the request has
    /// nothing to wait for.
    fn screenshot_pending(&self) -> bool { false }

    /// Frames the display thread composed and skipped, for `screen frames`,
    /// on a board that counts them.
    fn frame_counts(&self) -> Option<&FrameCounts> { None }

    /// CPU cycle counter the board was wired to (GUI's live MIPS estimate).
    fn cycles(&self) -> CyclesPtr;

    /// Ask for the next composed frame in `screen().rgba`, without the
    /// screenshot file [`request_screenshot`](Self::request_screenshot) also
    /// writes. Boards that compose into `rgba` every frame ignore it.
    fn request_frame(&self) {}

    /// True while a [`request_frame`](Self::request_frame) is still pending.
    fn frame_pending(&self) -> bool { false }

    /// Whether a completed composed frame is available to consumers. This
    /// may be called with `screen()` locked and must not lock it again.
    fn frame_available(&self) -> bool { true }

    /// Keep `screen().rgba` current with every composed frame while at least
    /// one consumer (VNC) has turned capture on. Calls nest: each `true` is
    /// matched by a `false`. Newport's GL capture composes in software; its
    /// wgpu window and gated resident tests read back composed GPU output.
    /// Boards that always compose into `rgba` ignore it.
    fn set_capture(&self, _on: bool) {}

    /// The hardware cursor as the guest has set it, read live from the
    /// board's registers (not the last composed frame).
    fn hw_cursor(&self) -> Option<HwCursor> { None }

    /// The host GPU the board draws or composes on, when it uses one: a wgpu
    /// window renderer presents on the same device, so a frame composed there
    /// need not come back to the CPU.
    #[cfg(feature = "wgpu")]
    fn gpu(&self) -> Option<std::sync::Arc<crate::wgpu::Gpu>> { None }
}

/// Set when anything the scan-out reads may have changed; taken by the
/// display before it reads anything. One store per processed command, job or
/// display-register write, never one per pixel or per memory store.
///
/// The writer marks after its effect is visible and the display takes the flag
/// before it reads, so a frame composed for a mark sees what the mark follows.
/// Every mark stores, even with the flag already up: a frame that takes it
/// between a writer's effect and its mark would otherwise lose the effect. The
/// flag has a cache line to itself, away from the state the writers touch.
///
/// A display thread that has nothing to do until something changes can park on
/// it ([`FrameWait::park_until_marked`]); a mark then wakes it.
#[repr(align(64))]
#[derive(Default)]
pub struct ScreenChanged {
    flag: AtomicBool,
    /// The display thread is parked, or on its way to be.
    parked: AtomicBool,
    /// Someone asked the thread to stop waiting ([`wake`](Self::wake)).
    wake: AtomicBool,
    /// The display thread, set by [`FrameWait::new`].
    thread: Mutex<Option<std::thread::Thread>>,
}

impl ScreenChanged {
    /// A flag that starts up, for a display whose first frame must be composed.
    pub fn marked() -> Self {
        let c = Self::default();
        c.flag.store(true, Ordering::Relaxed);
        c
    }

    /// One release store, and a relaxed load of whether the display thread
    /// sleeps, which a thread that is not parked leaves false: no call, lock or
    /// read-modify-write on the writer's path.
    #[inline(always)]
    pub fn mark(&self) {
        self.flag.store(true, Ordering::Release);
        if self.parked.load(Ordering::Relaxed) {
            self.unpark();
        }
    }

    /// Whether a mark is waiting, leaving it up.
    #[inline]
    pub fn is_marked(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Whether a mark came since the last call, clearing it. The display calls
    /// it before it reads anything the mark covers.
    #[inline]
    pub fn take(&self) -> bool {
        self.flag.swap(false, Ordering::AcqRel)
    }

    /// End the display thread's current or next wait at once: a screenshot
    /// request, or a stop.
    pub fn wake(&self) {
        self.wake.store(true, Ordering::Release);
        self.parked.store(false, Ordering::Release);
        if let Some(t) = self.thread.lock().as_ref() {
            t.unpark();
        }
    }

    fn unpark(&self) {
        // The swap makes one writer the waker.
        if self.parked.swap(false, Ordering::AcqRel) {
            if let Some(t) = self.thread.lock().as_ref() {
                t.unpark();
            }
        }
    }
}

/// What the display thread did, counted by that thread alone with relaxed
/// stores and read by the monitor's `screen frames`.
#[derive(Default)]
pub struct FrameCounts {
    frames: AtomicU64,
    composed: AtomicU64,
    skipped: AtomicU64,
}

impl FrameCounts {
    /// One frame, composed or skipped.
    pub fn count(&self, composed: bool) {
        // One writer, so a load and a store do.
        let add = |a: &AtomicU64| a.store(a.load(Ordering::Relaxed).wrapping_add(1), Ordering::Relaxed);
        add(&self.frames);
        add(if composed { &self.composed } else { &self.skipped });
    }

    /// Frames, composed, skipped.
    pub fn get(&self) -> (u64, u64, u64) {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        (g(&self.frames), g(&self.composed), g(&self.skipped))
    }

    pub fn reset(&self) {
        for a in [&self.frames, &self.composed, &self.skipped] {
            a.store(0, Ordering::Relaxed);
        }
    }
}

/// Requests for a fresh frame, counted so that one made while a frame is being
/// composed is still pending after it: asked and answered are counts, not a
/// flag. A screenshot, `request_frame` and a capturing client (VNC) are each
/// one of these on a board.
///
/// The display thread takes [`note`](Self::note) before it composes and calls
/// [`answer`](Self::answer) with it once the frame is in the screen; a caller
/// waits on [`pending`](Self::pending). A board that parks its display thread
/// wakes it from the request ([`ScreenChanged::wake`]).
#[derive(Default)]
pub struct FrameRequests {
    asked: AtomicU64,
    done: AtomicU64,
    capture: std::sync::atomic::AtomicU32,
}

impl FrameRequests {
    pub fn request(&self) {
        self.asked.fetch_add(1, Ordering::AcqRel);
    }

    /// Whether a request is waiting for its frame.
    pub fn pending(&self) -> bool {
        self.asked.load(Ordering::Acquire) != self.done.load(Ordering::Acquire)
    }

    /// The requests a frame composed from now on will answer. Take it before
    /// composing.
    pub fn note(&self) -> u64 {
        self.asked.load(Ordering::Acquire)
    }

    /// Whether the frame that `noted` was taken for answers a request.
    pub fn answers(&self, noted: u64) -> bool {
        noted != self.done.load(Ordering::Acquire)
    }

    /// The frame is in the screen: everything up to `noted` is answered.
    pub fn answer(&self, noted: u64) {
        self.done.store(noted, Ordering::Release);
    }

    /// One more consumer wants every frame (`on`), or one fewer. Calls nest.
    pub fn capture(&self, on: bool) {
        if on {
            self.capture.fetch_add(1, Ordering::AcqRel);
        } else {
            let _ = self.capture.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1));
        }
    }

    pub fn capturing(&self) -> bool {
        self.capture.load(Ordering::Acquire) != 0
    }
}

/// The frame period of a 60 Hz mode.
pub const FRAME_PERIOD: Duration = Duration::from_micros(16_667);

/// Frames between presents of an unchanged screen, about ten a second at
/// 60 Hz: the status bar moves, and a window that missed a frame catches up.
pub const IDLE_REFRESH: u32 = 6;

/// The longest a parked display thread goes without looking for a mark. A
/// writer that marks while the thread is on its way to sleep may not see that
/// it sleeps, since the mark is a store with no barrier behind it. The thread's
/// own look finds the flag, so the race costs this much latency, not a frame.
const PARK_RECHECK: Duration = Duration::from_millis(100);

/// Why [`FrameWait::park_until_marked`] returned.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Woke {
    /// The display was marked; the mark is still up for the frame to take.
    Marked,
    /// [`ScreenChanged::wake`] asked for the thread's attention.
    Requested,
    /// The board's timer came due.
    Timer,
}

/// The display thread's wait between frames, for a loop that already has its
/// frames. Frames are paced from a fixed start, so a slow one does not stretch
/// the rate, and a stall resyncs to now rather than running the missed frames
/// back to back. Call it last in the loop, after the frame.
pub struct FrameWait<'a> {
    changed: &'a ScreenChanged,
    period: Duration,
    /// Where the frame now running started.
    next: Instant,
}

impl<'a> FrameWait<'a> {
    /// For the calling thread, which `changed` records to unpark.
    pub fn new(changed: &'a ScreenChanged, period: Duration) -> Self {
        *changed.thread.lock() = Some(std::thread::current());
        Self { changed, period, next: Instant::now() }
    }

    /// Wait out the frame's period; ends early if [`ScreenChanged::wake`] asks.
    pub fn wait(&mut self) {
        let mut now = Instant::now();
        let end = self.next + self.period;
        if now >= end {
            self.next = now;
            return;
        }
        loop {
            if self.changed.wake.swap(false, Ordering::AcqRel) {
                self.next = now;
                return;
            }
            std::thread::park_timeout(end - now);
            now = Instant::now();
            if now >= end {
                self.next = end;
                return;
            }
        }
    }

    /// With nothing marked, park until the display is marked,
    /// [`ScreenChanged::wake`] is called, or `timer` comes due, instead of
    /// waking every period to find nothing. A board whose retrace or other
    /// hardware duty has a time of its own passes it as `timer`.
    ///
    /// A mark ends the wait at the next vsync, not at once, so marks between
    /// two vsyncs make one frame; after a long park that vsync is long past and
    /// the frame starts at once.
    pub fn park_until_marked(&mut self, timer: Option<Instant>) -> Woke {
        let c = self.changed;
        let end = self.next + self.period;
        let woke = loop {
            if c.is_marked() {
                break Woke::Marked;
            }
            if c.wake.swap(false, Ordering::AcqRel) {
                break Woke::Requested;
            }
            let now = Instant::now();
            let wait = match timer {
                Some(t) if t <= now => break Woke::Timer,
                Some(t) => (t - now).min(PARK_RECHECK),
                None => PARK_RECHECK,
            };
            // Say so, then look again: a writer that marked before it could
            // see this is found by the look.
            c.parked.store(true, Ordering::Relaxed);
            fence(Ordering::SeqCst);
            if !c.is_marked() && !c.wake.load(Ordering::Acquire) {
                std::thread::park_timeout(wait);
            }
            c.parked.store(false, Ordering::Relaxed);
        };
        let now = Instant::now();
        if woke == Woke::Timer {
            // The board's own time: leave the slot as it is unless it ran out.
            if now >= end {
                self.next = now;
            }
        } else if end > now {
            std::thread::sleep(end - now);
            self.next = end;
        } else {
            self.next = now;
        }
        woke
    }
}

/// Whether the display thread tears its renderer down when it ends.
///
/// Releasing a renderer's GL context has to happen on the thread that made
/// it current, so the display thread does it as its last act. On macOS
/// glutin runs that release on the main thread, and the main thread is the
/// one stopping the machine when the window closes, waiting in the join.
/// Neither goes on, and the disks are never flushed. So a board calls
/// [`request`](Self::request) first thing in its stop, and its display
/// thread ends with [`finish`](Self::finish), which leaves the renderer
/// alone when the stop came from the main thread. That stop is final: the
/// process is ending, and the renderer cannot be set up again.
#[derive(Default)]
pub struct DisplayStop(AtomicBool);

impl DisplayStop {
    pub fn request(&self) {
        // std has no is-main-thread test. The platform ones exist where the
        // hang does (pthread_main_np on macOS), but Rust names the main
        // thread, and the name lets a test on Linux stand in for it.
        self.0.store(std::thread::current().name() == Some("main"), Ordering::Release);
    }

    pub fn finish(&self, renderer: &Mutex<Option<Box<dyn Renderer>>>) {
        if !self.0.load(Ordering::Acquire) {
            if let Some(r) = renderer.lock().as_mut() {
                r.stop();
            }
        }
    }
}

/// The hardware cursor: where the board draws its glyph and which of the
/// glyph's pixels are opaque. The glyph's top-left is the pointer position
/// minus the hotspot, which the hardware does not know.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HwCursor {
    /// Glyph top-left in display coordinates; negative near the top/left edge.
    pub x: i32,
    pub y: i32,
    /// Glyph side, 32 or 64.
    pub size: usize,
    /// One word per glyph row, bit `c` set where column `c` is opaque.
    pub mask: Vec<u64>,
}

impl HwCursor {
    /// The hotspot when nothing better is known: the first opaque pixel of
    /// the topmost opaque row, the tip of an arrow. (0, 0) for an empty glyph.
    pub fn tip(&self) -> (i32, i32) {
        self.mask.iter().enumerate()
            .find(|(_, &m)| m != 0)
            .map(|(y, &m)| (m.trailing_zeros() as i32, y as i32))
            .unwrap_or((0, 0))
    }

    /// A key that tells glyphs apart, to remember a learned hotspot per shape.
    pub fn shape_key(&self) -> u64 {
        // FNV-1a over the mask words.
        self.mask.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &m| (h ^ m).wrapping_mul(0x100_0000_01b3))
            ^ self.size as u64
    }
}

/// The visible part of the last composed frame, tight (`width` pixels a row,
/// `0xAABBGGRR`, R in the low byte), with its sequence number. `None` until
/// the display has a video mode. Holds the screen lock only for the copy.
pub fn copy_frame(display: &dyn GfxDisplay, out: &mut Vec<u32>) -> Option<(usize, usize, u64)> {
    if !display.frame_available() {
        return None;
    }
    let screen = display.screen().lock();
    if !display.frame_available() {
        return None;
    }
    let (w, h) = (screen.width, screen.height);
    if w == 0 || h == 0 {
        return None;
    }
    out.clear();
    out.reserve(w * h);
    for y in 0..h {
        let row = y * crate::disp::FB_STRIDE;
        out.extend_from_slice(&screen.rgba[row..row + w]);
    }
    Some((w, h, screen.frame_seq))
}

/// Request a fresh frame and wait (at most `timeout`) for the board to
/// compose it. Returns false when the wait ran out with the request still
/// pending, so the screen still holds an older frame.
pub fn wait_fresh_frame(display: &dyn GfxDisplay, timeout: std::time::Duration) -> bool {
    display.request_frame();
    let deadline = std::time::Instant::now() + timeout;
    while display.frame_pending() {
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    true
}

/// [`wait_fresh_frame`], then copy the frame as [`copy_frame`] does, whether
/// or not the wait succeeded. For callers that can use a slightly old frame;
/// a caller that cannot calls the two itself.
pub fn fresh_frame(display: &dyn GfxDisplay, out: &mut Vec<u32>, timeout: std::time::Duration) -> Option<(usize, usize, u64)> {
    wait_fresh_frame(display, timeout);
    copy_frame(display, out)
}

/// A display for tests: a screen of a given size whose requested frames are
/// composed at once, or never.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    pub(crate) struct FakeDisplay {
        stop: DisplayStop,
        renderer: Mutex<Option<Box<dyn Renderer>>>,
        screen: Mutex<Rex3Screen>,
        pub(crate) counts: FrameCounts,
        stuck: bool,
    }

    impl FakeDisplay {
        /// A `w` by `h` screen; `stuck` never composes a requested frame.
        pub(crate) fn new(w: usize, h: usize, stuck: bool) -> Self {
            let mut screen = Rex3Screen::new();
            (screen.width, screen.height) = (w, h);
            Self { stop: DisplayStop::default(), renderer: Mutex::new(None), screen: Mutex::new(screen), counts: FrameCounts::default(), stuck }
        }
    }

    impl GfxDisplay for FakeDisplay {
        fn display_stop(&self) -> &DisplayStop { &self.stop }
        fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>> { &self.renderer }
        fn screen(&self) -> &Mutex<Rex3Screen> { &self.screen }
        fn request_screenshot(&self) {}
        fn cycles(&self) -> CyclesPtr { CyclesPtr::dangling() }
        fn frame_pending(&self) -> bool { self.stuck }
        fn frame_counts(&self) -> Option<&FrameCounts> { Some(&self.counts) }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::FakeDisplay;
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;
    use std::time::Duration;

    struct ChangingDisplay {
        stop: DisplayStop,
        renderer: Mutex<Option<Box<dyn Renderer>>>,
        screen: Mutex<Rex3Screen>,
        checks: AtomicUsize,
        available_checks: AtomicUsize,
    }

    impl GfxDisplay for ChangingDisplay {
        fn display_stop(&self) -> &DisplayStop { &self.stop }
        fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>> { &self.renderer }
        fn screen(&self) -> &Mutex<Rex3Screen> { &self.screen }
        fn request_screenshot(&self) {}
        fn cycles(&self) -> CyclesPtr { CyclesPtr::dangling() }
        fn frame_available(&self) -> bool {
            self.checks.fetch_add(1, Ordering::Relaxed) < self.available_checks.load(Ordering::Relaxed)
        }
    }

    /// A board that never composes the requested frame is reported, so a
    /// caller does not take the old screen for a new one.
    #[test]
    fn waiting_for_a_frame_that_never_comes_says_so() {
        assert!(!wait_fresh_frame(&FakeDisplay::new(0, 0, true), Duration::from_millis(20)));
        assert!(wait_fresh_frame(&FakeDisplay::new(0, 0, false), Duration::from_millis(20)));
    }

    #[test]
    fn copy_frame_rechecks_availability_under_the_screen_lock() {
        std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(|| {
            let mut screen = Rex3Screen::new();
            screen.width = 1;
            screen.height = 1;
            screen.rgba[0] = 0xff12_3456;
            screen.frame_seq = 7;
            let display = ChangingDisplay {
                stop: DisplayStop::default(), renderer: Mutex::new(None), screen: Mutex::new(screen),
                checks: AtomicUsize::new(0), available_checks: AtomicUsize::new(usize::MAX),
            };
            let mut pixels = Vec::new();
            assert_eq!(copy_frame(&display, &mut pixels), Some((1, 1, 7)));
            assert_eq!(pixels, [0xff12_3456]);
            display.checks.store(0, Ordering::Relaxed);
            display.available_checks.store(1, Ordering::Relaxed);
            pixels[0] = 0xfeed_beef;
            assert_eq!(copy_frame(&display, &mut pixels), None);
            assert_eq!(pixels, [0xfeed_beef]);
            assert_eq!(display.checks.load(Ordering::Relaxed), 2);
        }).unwrap().join().unwrap();
    }

    #[test]
    fn a_mark_is_taken_once_and_every_mark_stores() {
        let c = ScreenChanged::default();
        assert!(!c.take());
        c.mark();
        c.mark();
        assert!(c.is_marked());
        assert!(c.take());
        assert!(!c.take());
        assert!(ScreenChanged::marked().take());
    }

    #[test]
    fn a_request_made_while_a_frame_is_composed_is_still_pending_after_it() {
        let r = FrameRequests::default();
        assert!(!r.pending());
        r.request();
        let noted = r.note();
        assert!(r.pending() && r.answers(noted));
        // Asked again while that frame is composed.
        r.request();
        r.answer(noted);
        assert!(r.pending(), "the second request needs a frame composed after it");
        assert!(!r.answers(noted));
        r.answer(r.note());
        assert!(!r.pending());
    }

    #[test]
    fn capture_calls_nest() {
        let r = FrameRequests::default();
        r.capture(true);
        r.capture(true);
        r.capture(false);
        assert!(r.capturing());
        r.capture(false);
        r.capture(false);
        assert!(!r.capturing(), "an extra off does not go below none");
    }

    #[test]
    fn the_counts_say_what_the_frames_did_and_reset() {
        let n = FrameCounts::default();
        for composed in [true, false, false, false] {
            n.count(composed);
        }
        assert_eq!(n.get(), (4, 1, 3));
        n.reset();
        assert_eq!(n.get(), (0, 0, 0));
    }

    #[test]
    fn waiting_out_the_frame_takes_the_period_and_does_not_drift() {
        let c = ScreenChanged::default();
        let start = Instant::now();
        let mut w = FrameWait::new(&c, Duration::from_millis(20));
        for _ in 0..4 {
            // A frame that takes a quarter of the period.
            std::thread::sleep(Duration::from_millis(5));
            w.wait();
        }
        assert!(start.elapsed() >= Duration::from_millis(80), "four periods, not four frames' work");
    }

    #[test]
    fn a_late_frame_resyncs_instead_of_running_the_missed_ones_back_to_back() {
        let c = ScreenChanged::default();
        let mut w = FrameWait::new(&c, Duration::from_millis(20));
        std::thread::sleep(Duration::from_millis(70));
        w.wait();
        let t = Instant::now();
        w.wait();
        assert!(t.elapsed() >= Duration::from_millis(19), "the next slot is a period away, not the missed ones");
    }

    /// Run `f` on a thread with a wait, and return once it is parked (or done).
    fn parked_thread<T: Send + 'static>(
        c: &Arc<ScreenChanged>,
        period: Duration,
        f: impl FnOnce(&mut FrameWait<'_>) -> T + Send + 'static,
    ) -> std::thread::JoinHandle<T> {
        let q = c.clone();
        let t = std::thread::spawn(move || f(&mut FrameWait::new(&q, period)));
        while !c.parked.load(Ordering::Relaxed) && !t.is_finished() {
            std::thread::yield_now();
        }
        t
    }

    #[test]
    fn a_wake_ends_a_wait_at_once() {
        let c = Arc::new(ScreenChanged::default());
        let q = c.clone();
        let t = std::thread::spawn(move || {
            let mut w = FrameWait::new(&q, Duration::from_secs(30));
            let t = Instant::now();
            w.wait();
            t.elapsed()
        });
        while !t.is_finished() {
            c.wake();
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(t.join().unwrap() < Duration::from_secs(10));
    }

    #[test]
    fn a_parked_display_wakes_on_a_mark_and_not_before() {
        let c = Arc::new(ScreenChanged::default());
        let t = parked_thread(&c, Duration::from_millis(1), |w| w.park_until_marked(None));
        std::thread::sleep(Duration::from_millis(30));
        assert!(!t.is_finished(), "nothing marked, so it stays parked");
        c.mark();
        assert_eq!(t.join().unwrap(), Woke::Marked);
        assert!(c.take(), "the mark is still up for the frame to take");
    }

    #[test]
    fn a_wake_ends_a_park() {
        let c = Arc::new(ScreenChanged::default());
        let t = parked_thread(&c, Duration::from_millis(1), |w| w.park_until_marked(None));
        c.wake();
        assert_eq!(t.join().unwrap(), Woke::Requested);
    }

    #[test]
    fn marks_between_two_vsyncs_make_one_frame() {
        let c = ScreenChanged::default();
        let start = Instant::now();
        let mut w = FrameWait::new(&c, Duration::from_millis(20));
        // Frame one runs at the start; four marks follow within its period.
        for _ in 0..4 {
            c.mark();
        }
        assert_eq!(w.park_until_marked(None), Woke::Marked);
        assert!(start.elapsed() >= Duration::from_millis(20), "the frame starts at the vsync, not at the mark");
        assert!(c.take());
        c.mark();
        assert_eq!(w.park_until_marked(None), Woke::Marked);
        assert!(start.elapsed() >= Duration::from_millis(40), "and the next one a period after that");
    }

    /// After a long park the vsync is long past: the frame starts at once and
    /// the next slot is a period from now, not from where the clock had got to.
    #[test]
    fn a_mark_after_a_long_park_starts_the_frame_at_once() {
        let c = ScreenChanged::default();
        let mut w = FrameWait::new(&c, Duration::from_millis(20));
        w.next -= Duration::from_secs(1);
        c.mark();
        let t = Instant::now();
        assert_eq!(w.park_until_marked(None), Woke::Marked);
        assert!(w.next >= t, "the slot starts from now");
    }

    #[test]
    fn a_board_timer_wakes_a_parked_display_when_it_is_due() {
        let c = ScreenChanged::default();
        let mut w = FrameWait::new(&c, Duration::from_millis(20));
        let due = Instant::now() + Duration::from_millis(40);
        assert_eq!(w.park_until_marked(Some(due)), Woke::Timer);
        assert!(Instant::now() >= due);
        assert!(!c.take());
    }
}
