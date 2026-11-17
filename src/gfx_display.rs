//! What the UI, CI socket and GUI front-end need from whichever graphics board
//! drives the screen (Newport/REX3 or GR2).
//!
//! Each board owns its display thread and composition (the status bar's
//! inputs come from the machine's shared `StatusSource`); this trait only
//! exposes the hand-off points: the renderer slot the front-end installs into, the
//! last-presented screen (screenshots, VNC), the hardware cursor, and the CPU
//! cycle counter.

use std::sync::atomic::{AtomicBool, Ordering};

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

    /// CPU cycle counter the board was wired to (GUI's live MIPS estimate).
    fn cycles(&self) -> CyclesPtr;

    /// Ask for the next composed frame in `screen().rgba`, without the
    /// screenshot file [`request_screenshot`](Self::request_screenshot) also
    /// writes. Boards that compose into `rgba` every frame ignore it.
    fn request_frame(&self) {}

    /// True while a [`request_frame`](Self::request_frame) is still pending.
    fn frame_pending(&self) -> bool { false }

    /// Minimum time a frontend should allow for an explicit fresh-frame
    /// request. Slow compatibility paths can raise the caller's timeout.
    fn minimum_frame_request_timeout(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }

    /// Whether a completed composed frame is available to consumers. This
    /// may be called with `screen()` locked and must not lock it again.
    fn frame_available(&self) -> bool { true }

    /// Keep `screen().rgba` current with every composed frame while at least
    /// one consumer (VNC) has turned capture on. Calls nest: each `true` is
    /// matched by a `false`. Newport's GL capture composes in software; its
    /// wgpu window and resident mode read back composed GPU output.
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

    /// Whether a visible host window must use a presenter on [`Self::gpu`].
    /// Frontends must fail startup instead of selecting another renderer.
    #[cfg(feature = "wgpu")]
    fn requires_same_device_wgpu_presenter(&self) -> bool { false }
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
    let deadline = std::time::Instant::now() + timeout.max(display.minimum_frame_request_timeout());
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
        stuck: bool,
    }

    impl FakeDisplay {
        /// A `w` by `h` screen; `stuck` never composes a requested frame.
        pub(crate) fn new(w: usize, h: usize, stuck: bool) -> Self {
            let mut screen = Rex3Screen::new();
            (screen.width, screen.height) = (w, h);
            Self { stop: DisplayStop::default(), renderer: Mutex::new(None), screen: Mutex::new(screen), stuck }
        }
    }

    impl GfxDisplay for FakeDisplay {
        fn display_stop(&self) -> &DisplayStop { &self.stop }
        fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>> { &self.renderer }
        fn screen(&self) -> &Mutex<Rex3Screen> { &self.screen }
        fn request_screenshot(&self) {}
        fn cycles(&self) -> CyclesPtr { CyclesPtr::dangling() }
        fn frame_pending(&self) -> bool { self.stuck }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::FakeDisplay;
    use super::*;
    use std::sync::atomic::AtomicUsize;
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
}
