//! The display refresh thread shared by every graphics board.
//!
//! A board registers a [`Head`] and gets a [`HeadHandle`]. One thread per
//! machine keeps each head's video timing on the host's monotonic clock and
//! runs the same frame for every board:
//!
//! 1. vblank start: [`Head::vblank_start`]. The board latches swaps and
//!    raises its retrace interrupt.
//! 2. the blank period, waited out on the deadline clock;
//! 3. vblank end: [`Head::vblank_end`];
//! 4. capture: [`Head::capture`], when the picture changed. The board
//!    takes what the frame needs from its live state (nothing, where the
//!    frame is composed straight from board memory).
//! 5. compose: [`Head::compose`];
//! 6. present through the head's renderer;
//! 7. screenshots and frame requests, served from the head's screen.
//!
//! Capture and compose come in display time, after the blank, because of
//! how the IRIX kernel swaps buffers. A swap request only schedules the
//! swap and takes the client's access away; the retrace handler writes the
//! new display mode and gives the access back. So once the retrace has
//! been serviced the visible buffer is finished, and the client draws its
//! next frame into the other one.
//!
//! The thread never blocks on a board: a head that is not ready to compose
//! says so and is asked again at the next retrace.
//!
//! The rate is the guest's timing or what the thread manages, whichever is
//! slower. A retrace the thread reaches late (the last frame took longer
//! than a period to compose, or the host held the thread) happens then,
//! and the schedule carries on from it: no retrace is made up afterwards,
//! so the guest never sees two closer together than a period.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::debug_overlay::DebugOverlay;
use crate::disp::{Rex3Screen, StatusBar, StatusBarTexture, StatusSource, FB_STRIDE};
use crate::gfx_display::DisplayStop;

mod marks;
mod renderer;
pub mod verify;
#[cfg(test)]
mod tests;

pub use marks::{Mark, Marks, Rect};
pub use renderer::Renderer;

/// Whether presents wait for the host display's vertical sync (`vsync`).
static HOST_VSYNC: AtomicBool = AtomicBool::new(false);

/// Set before the window is made.
pub fn set_host_vsync(on: bool) {
    HOST_VSYNC.store(on, Ordering::Relaxed);
}

/// Whether a renderer should block on the host's vertical sync. Off by
/// default: the refresh thread keeps the guest's timing, and a present that
/// waited for the host would hold every head's retrace back.
pub fn host_vsync() -> bool {
    HOST_VSYNC.load(Ordering::Relaxed)
}

/// How late a retrace may be, as a fraction of the field period, and keep
/// the schedule (the next one comes on time, so the rate holds through the
/// thread's wake-up jitter). Past it the schedule slips to the retrace.
const LATE_TOLERANCE_DIV: u64 = 16;

/// An unchanged frame is presented again this often, so the status bar
/// keeps moving.
pub const HEARTBEAT_NS: u64 = 100_000_000;

/// The shortest field period accepted: 1 kHz, well above any SGI display.
const MIN_PERIOD_NS: u64 = 1_000_000;

const NS_PER_SECOND: u128 = 1_000_000_000;

/// What a timing generator is programmed to: the totals, blanking included,
/// and the clock that counts them. The refresh rate follows from these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoTiming {
    /// Pixel clocks per line.
    pub h_total: u32,
    /// Lines per frame, over all of its fields.
    pub v_total: u32,
    pub pixel_clock_hz: u64,
    /// Lines of vertical blanking per field.
    pub vblank_lines: u32,
    /// Fields per frame: 1, or 2 for field-sequential stereo and the
    /// two-channel formats. Each field has its own vertical blank.
    pub fields: u32,
}

impl VideoTiming {
    /// 1280x1024 at 60 Hz (1688x1066 at 107.964 MHz), what every board ran at
    /// before it decoded its own timing.
    pub const HZ60: VideoTiming =
        VideoTiming { h_total: 1688, v_total: 1066, pixel_clock_hz: 107_964_480, vblank_lines: 42, fields: 1 };

    fn frame_clocks(&self) -> u128 {
        self.h_total as u128 * self.v_total as u128
    }

    /// Nanoseconds from field 0 to field `k`.
    fn field_offset_ns(&self, k: u64) -> u64 {
        (k as u128 * self.frame_clocks() * NS_PER_SECOND / (self.pixel_clock_hz as u128 * self.fields as u128)) as u64
    }

    pub fn field_period_ns(&self) -> u64 {
        self.field_offset_ns(1)
    }

    pub fn blank_ns(&self) -> u64 {
        (self.vblank_lines as u128 * self.h_total as u128 * NS_PER_SECOND / self.pixel_clock_hz as u128) as u64
    }

    /// Frames per second, in millihertz.
    pub fn refresh_mhz(&self) -> u64 {
        (self.pixel_clock_hz as u128 * 1000 / self.frame_clocks().max(1)) as u64
    }

    /// Whether the thread can run this timing. The tables come from the
    /// guest, so a timing is checked before anything is armed from it.
    pub fn valid(&self) -> bool {
        self.h_total != 0
            && self.v_total != 0
            && self.pixel_clock_hz != 0
            && (1..=2).contains(&self.fields)
            && self.field_period_ns() >= MIN_PERIOD_NS
            && self.field_period_ns() <= NS_PER_SECOND as u64
            && self.blank_ns() < self.field_period_ns()
    }
}

/// Where the beam is: what a board's status bits and line counters read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Beam {
    /// Line within the field, counted from the start of the vertical blank.
    pub line: u32,
    pub in_vblank: bool,
    pub field: u32,
}

/// What the thread wants from a compose.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Want {
    /// A consumer (screenshot, frame request, capture) reads this frame from
    /// memory, so it must be composed where the screen can hold it.
    pub readback: bool,
    /// The renderer presents frames that are on the board's GPU; the head
    /// may answer [`Composed::Gpu`].
    pub gpu: bool,
}

/// A head's answer to [`Head::compose`].
pub enum Composed {
    /// The screen holds a new frame: finished pixels in `fb_rgb` with
    /// `prebuilt` set, or the display state for the renderer's compositor.
    Frame,
    /// A new frame, composed on the board's GPU: words of 0x00RRGGBB in rows
    /// of `width`. Only when [`Want::gpu`] was set.
    #[cfg(feature = "wgpu")]
    Gpu { frame: wgpu::Buffer, width: u32, height: u32 },
    /// The last frame still stands; the thread shows it again.
    Unchanged,
    /// The board cannot show the frame after all; as a false from
    /// [`Head::capture`].
    NotReady,
}

/// A graphics board's side of one display channel. Every method is called
/// on the refresh thread.
pub trait Head: Send + Sync {
    /// Take what this frame needs from the live board (the planes, or a
    /// whole composed frame where there is nothing cheaper to take), into
    /// the board or into `screen`. Called after the blank, right before
    /// [`compose`](Self::compose). Returns false when the board
    /// cannot show a consistent frame yet (drawing queued before a
    /// display-state change has not drained): the last frame stays up and
    /// the head is asked again at the next retrace. The board bounds how
    /// long it answers false.
    fn capture(&self, _screen: &mut Rex3Screen, _want: Want) -> bool { true }

    /// The vertical blank of `field` begins (scheduled for `t_ns`).
    fn vblank_start(&self, _field: u32, _t_ns: u64) {}

    /// The vertical blank of `field` ends.
    fn vblank_end(&self, _field: u32, _t_ns: u64) {}

    /// The beam reached `line` of `field`, one of the lines the board asked
    /// for with [`HeadHandle::set_line_events`]. Lines come in the order the
    /// beam meets them, timed from when the field's blank really began, and
    /// all of them before the next one begins. Unlike the rest of the
    /// trait this is called on the machine's timer thread when the refresh
    /// has one ([`Refresh::set_timers`]), so a compose under way does not
    /// hold a line up: keep it short (raise the interrupt) and take no lock
    /// a compose holds.
    fn line(&self, _line: u32, _field: u32, _t_ns: u64) {}

    /// Whether the picture changed since the last compose, for a board that
    /// finds out by looking. A board that knows when it draws calls
    /// [`HeadHandle::damage`] and leaves this alone.
    fn changed(&self) -> bool { false }

    /// Compose the captured frame into `screen`, whose lock the thread
    /// holds. Sets `screen.width` and `screen.height`.
    fn compose(&self, screen: &mut Rex3Screen, want: Want) -> Composed;

    /// The live colour and auxiliary planes, when the renderer's compositor
    /// reads them in place (Newport).
    fn live_planes(&self) -> Option<(&[u32], &[u32])> { None }

    /// Depth of the board's command queue, for the status bar.
    fn gfifo_pending(&self) -> usize { 0 }

    /// Set the board's debug views in the overlay drawn over the frame.
    fn overlay(&self, _overlay: &mut DebugOverlay) {}
}

/// The line events of the field that last began.
#[derive(Default)]
struct Chain {
    /// Moves when the lines are replaced: a timer of an older field stops.
    generation: u64,
    field: u32,
    /// When the field's blank began.
    start_ns: u64,
    /// Nanoseconds a line, as a fraction: clocks a line over the clock.
    h_total: u32,
    pixel_clock_hz: u64,
    lines: Vec<u32>,
    /// How many of them have been delivered.
    done: usize,
    /// The timer that delivers the rest, when the refresh has timers.
    timer: Option<crate::hptimer::TimerId>,
}

impl Chain {
    /// Deliver the lines due by `upto`; returns when the next is due.
    fn run(&mut self, head: &dyn Head, upto: u64) -> Option<u64> {
        while let Some(&line) = self.lines.get(self.done) {
            let at = self.start_ns + (line as u128 * self.h_total as u128 * NS_PER_SECOND / self.pixel_clock_hz.max(1) as u128) as u64;
            if at > upto {
                return Some(at);
            }
            self.done += 1;
            head.line(line, self.field, at);
        }
        None
    }
}

/// The epoch of a head's schedule, published for [`HeadHandle::beam`].
struct Phase {
    timing: VideoTiming,
    epoch_ns: u64,
}

/// What the rest of the emulator holds of a head: the screen, the renderer
/// slot, and the requests. The refresh thread serves all of them.
pub struct HeadHandle {
    refresh: Weak<Shared>,
    /// The board, once it exists (see [`attach`](Self::attach)).
    head: Mutex<Option<Weak<dyn Head>>>,
    screen: Mutex<Rex3Screen>,
    renderer: Mutex<Option<Box<dyn Renderer>>>,
    stop: DisplayStop,
    status: Mutex<Arc<StatusSource>>,
    /// The timing the board asked for, and a count of the requests.
    timing: Mutex<(VideoTiming, u64)>,
    /// Lines the board wants [`Head::line`] called at.
    lines: Mutex<Vec<u32>>,
    chain: Mutex<Chain>,
    phase: Mutex<Phase>,
    active: AtomicBool,
    /// Set by `stop`, cleared when the thread has let go of the renderer.
    stopping: Mutex<bool>,
    stopped: Condvar,
    damage: AtomicBool,
    screenshot: AtomicBool,
    frame_request: AtomicBool,
    capture: AtomicU32,
    /// Retraces delivered, for tests and the monitor.
    retraces: AtomicU64,
}

impl HeadHandle {
    /// Name the board this handle is for. A board is given its handle when
    /// it is built and attaches itself once it is in its `Arc`; nothing runs
    /// before that. A board that makes a new head at each start attaches
    /// it again, with the handle stopped.
    pub fn attach(&self, head: Weak<dyn Head>) {
        *self.head.lock() = Some(head);
    }

    fn board(&self) -> Option<Arc<dyn Head>> {
        self.head.lock().as_ref().and_then(Weak::upgrade)
    }

    /// Show another machine's status in the bar. The handle stays with the
    /// refresh thread it was made on.
    pub fn set_status(&self, status: Arc<StatusSource>) {
        *self.status.lock() = status;
    }

    /// The last frame. Screenshots and VNC read `rgba`.
    pub fn screen(&self) -> &Mutex<Rex3Screen> { &self.screen }

    /// Where a front-end installs its renderer.
    pub fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>> { &self.renderer }

    pub fn display_stop(&self) -> &DisplayStop { &self.stop }

    /// The picture changed. From any thread.
    pub fn damage(&self) {
        // Drawing paths call this per primitive: the load keeps the line
        // shared until the thread clears it, once a frame.
        if !self.damage.load(Ordering::Relaxed) {
            self.damage.store(true, Ordering::Release);
        }
    }

    /// What a compose would be asked for now.
    pub fn want(&self) -> Want {
        let readback = Slot::readback_wanted(self);
        #[cfg(feature = "wgpu")]
        let gpu = !readback && self.renderer.lock().as_ref().is_some_and(|r| r.takes_gpu_frames());
        #[cfg(not(feature = "wgpu"))]
        let gpu = false;
        Want { readback, gpu }
    }

    /// The guest reprogrammed the timing generator. The phase restarts at
    /// the next retrace. Returns false, and keeps the old timing, when the
    /// thread cannot run the new one.
    pub fn set_timing(&self, timing: VideoTiming) -> bool {
        if !timing.valid() {
            return false;
        }
        let mut t = self.timing.lock();
        if t.0 != timing {
            *t = (timing, t.1 + 1);
            drop(t);
            self.kick();
        }
        true
    }

    pub fn timing(&self) -> VideoTiming {
        self.timing.lock().0
    }

    /// The lines of a field, counted from the start of its vertical blank,
    /// at which the board wants [`Head::line`] called: a board with line
    /// compare interrupts. Taken up at the next retrace. A line past the
    /// end of the field is never reached.
    pub fn set_line_events(&self, lines: &[u32]) {
        let mut l = self.lines.lock();
        if l.as_slice() != lines {
            l.clear();
            l.extend_from_slice(lines);
            l.sort_unstable();
            l.dedup();
        }
    }

    /// Whether a consumer has capture on.
    pub fn capturing(&self) -> bool {
        self.capture.load(Ordering::Acquire) > 0
    }

    /// Whether the head is started.
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    pub fn retraces(&self) -> u64 {
        self.retraces.load(Ordering::Relaxed)
    }

    /// The beam's position at `now_ns` on the refresh clock.
    pub fn beam(&self, now_ns: u64) -> Beam {
        let p = self.phase.lock();
        let t = &p.timing;
        let period = t.field_period_ns().max(1);
        let since = now_ns.saturating_sub(p.epoch_ns);
        let k = since / period;
        // Field k starts at an exact multiple, not at k * period rounded.
        let into = since.saturating_sub(t.field_offset_ns(k));
        let line_ns = (t.h_total as u128 * NS_PER_SECOND / t.pixel_clock_hz.max(1) as u128).max(1) as u64;
        Beam {
            line: (into / line_ns) as u32,
            in_vblank: into < t.blank_ns(),
            field: (k % t.fields.max(1) as u64) as u32,
        }
    }

    /// Now, on the clock [`beam`](Self::beam) takes.
    pub fn now_ns(&self) -> u64 {
        self.refresh.upgrade().map_or(0, |s| (s.clock)())
    }

    /// Ask for the next frame in `screen().rgba` and a PNG of it.
    pub fn request_screenshot(&self) {
        self.screenshot.store(true, Ordering::Release);
        self.kick();
    }

    pub fn screenshot_pending(&self) -> bool {
        self.screenshot.load(Ordering::Acquire)
    }

    /// Ask for the next frame in `screen().rgba`.
    pub fn request_frame(&self) {
        self.frame_request.store(true, Ordering::Release);
        self.kick();
    }

    pub fn frame_pending(&self) -> bool {
        self.frame_request.load(Ordering::Acquire)
    }

    /// Keep `screen().rgba` current with every frame while a consumer has
    /// capture on. Calls nest.
    pub fn set_capture(&self, on: bool) {
        if on {
            self.capture.fetch_add(1, Ordering::AcqRel);
        } else {
            let _ = self.capture.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1));
        }
    }

    /// Begin retraces and frames for this head. The refresh thread starts
    /// with the first head.
    pub fn start(&self) {
        *self.stopping.lock() = false;
        self.active.store(true, Ordering::Release);
        if let Some(s) = self.refresh.upgrade() {
            s.ensure_thread();
        }
        self.kick();
    }

    /// End retraces and frames, and have the thread let go of the renderer
    /// (see [`DisplayStop`]). Returns once no callback of this head is
    /// running or will run.
    pub fn stop(&self) {
        self.stop.request();
        if !self.active.swap(false, Ordering::AcqRel) {
            return;
        }
        let Some(shared) = self.refresh.upgrade() else { return };
        let mut stopping = self.stopping.lock();
        *stopping = true;
        if shared.threaded() {
            shared.wake();
            while *stopping {
                self.stopped.wait(&mut stopping);
            }
        } else {
            drop(stopping);
            shared.step();
        }
    }

    fn kick(&self) {
        if let Some(s) = self.refresh.upgrade() {
            s.wake();
        }
    }
}

/// The thread's own state of a head.
struct Slot {
    handle: Weak<HeadHandle>,
    timing: VideoTiming,
    timing_seq: u64,
    /// Whether the schedule below is running.
    armed: bool,
    epoch_ns: u64,
    /// The next field to begin, or the one in its blank.
    field: u64,
    in_blank: bool,
    /// When the blank in progress ends.
    blank_end_ns: u64,
    status_bar: StatusBar,
    overlay: DebugOverlay,
    sbtex: StatusBarTexture,
    size: (usize, usize),
    shown: bool,
    last_present_ns: u64,
    /// The screen holds the current frame (not only the GPU).
    cpu_frame: bool,
    /// `screen.rgba` holds the current frame.
    rgba_current: bool,
    /// A capture was refused: a change is waiting behind the frame on
    /// screen, and requests wait for it.
    waiting: bool,
    /// Composes display state into `screen.rgba` for a reader when no
    /// renderer is installed (`--ci`); made on first use.
    reader: Option<crate::compositor::SwCompositor>,
    /// The check of the board's unchanged verdicts (`verify`).
    verify: verify::Check,
    /// While capture forces the renderer onto the software compositor:
    /// whether it used GL before, to switch back when capture ends.
    restore_gl: Option<bool>,
}

struct Shared {
    clock: Box<dyn Fn() -> u64 + Send + Sync>,
    /// The machine's timers, for line events ([`Refresh::set_timers`]).
    timers: std::sync::OnceLock<Arc<crate::hptimer::TimerManager>>,
    slots: Mutex<Vec<Slot>>,
    /// Heads registered since the last step.
    incoming: Mutex<Vec<Slot>>,
    thread: Mutex<Option<std::thread::Thread>>,
    spawn: Option<fn(Arc<Shared>)>,
    shutdown: AtomicBool,
    screenshots: AtomicU32,
}

/// The refresh thread of one machine. Dropping it ends the thread.
pub struct Refresh {
    shared: Arc<Shared>,
}

/// The host's monotonic clock in nanoseconds since the first call.
pub fn monotonic_ns() -> u64 {
    static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

impl Refresh {
    pub fn new() -> Arc<Self> {
        Arc::new(Refresh { shared: Shared::new(Box::new(monotonic_ns), Some(Shared::spawn)) })
    }

    /// Deliver line events ([`Head::line`]) from the machine's timer
    /// thread, each one's timer scheduling the next, so they keep their
    /// times while this thread composes. Without it the thread delivers
    /// them itself between its other deadlines.
    pub fn set_timers(&self, timers: Arc<crate::hptimer::TimerManager>) {
        let _ = self.shared.timers.set(timers);
    }

    /// A refresh with no thread, on a clock the caller moves: each
    /// [`step`](Self::step) does what is due.
    pub fn manual(clock: impl Fn() -> u64 + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Refresh { shared: Shared::new(Box::new(clock), None) })
    }

    /// A handle for a new head. Nothing runs until the board is
    /// [attached](HeadHandle::attach) and [started](HeadHandle::start). The
    /// head leaves the schedule when the handle or the board is dropped.
    pub fn head(&self, status: Arc<StatusSource>) -> Arc<HeadHandle> {
        let timing = VideoTiming::HZ60;
        let handle = Arc::new(HeadHandle {
            refresh: Arc::downgrade(&self.shared),
            head: Mutex::new(None),
            screen: Mutex::new(Rex3Screen::new()),
            renderer: Mutex::new(None),
            stop: DisplayStop::default(),
            status: Mutex::new(status),
            timing: Mutex::new((timing, 0)),
            lines: Mutex::new(Vec::new()),
            chain: Mutex::new(Chain::default()),
            phase: Mutex::new(Phase { timing, epoch_ns: 0 }),
            active: AtomicBool::new(false),
            stopping: Mutex::new(false),
            stopped: Condvar::new(),
            damage: AtomicBool::new(true),
            screenshot: AtomicBool::new(false),
            frame_request: AtomicBool::new(false),
            capture: AtomicU32::new(0),
            retraces: AtomicU64::new(0),
        });
        self.shared.incoming.lock().push(Slot {
            handle: Arc::downgrade(&handle),
            timing,
            timing_seq: 0,
            armed: false,
            epoch_ns: 0,
            field: 0,
            in_blank: false,
            blank_end_ns: 0,
            status_bar: StatusBar::new(),
            overlay: DebugOverlay::new(),
            sbtex: StatusBarTexture::new(),
            size: (0, 0),
            shown: false,
            last_present_ns: 0,
            cpu_frame: false,
            rgba_current: false,
            waiting: false,
            reader: None,
            verify: verify::Check::default(),
            restore_gl: None,
        });
        handle
    }

    /// [`head`](Self::head) for a board that already exists.
    pub fn add_head(&self, head: Weak<dyn Head>, status: Arc<StatusSource>) -> Arc<HeadHandle> {
        let handle = self.head(status);
        handle.attach(head);
        handle
    }

    /// Do everything that is due now; returns when the next thing is due,
    /// or `None` with no head running. The thread's loop body.
    pub fn step(&self) -> Option<u64> {
        self.shared.step()
    }

    /// Hold the thread between steps: no callback runs until the guard is
    /// dropped. For snapshots.
    pub fn quiesce(&self) -> impl Drop + '_ {
        self.shared.slots.lock()
    }
}

impl Drop for Refresh {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.wake();
    }
}

impl Shared {
    fn new(clock: Box<dyn Fn() -> u64 + Send + Sync>, spawn: Option<fn(Arc<Shared>)>) -> Arc<Self> {
        Arc::new(Shared {
            clock,
            timers: std::sync::OnceLock::new(),
            slots: Mutex::new(Vec::new()),
            incoming: Mutex::new(Vec::new()),
            thread: Mutex::new(None),
            spawn,
            shutdown: AtomicBool::new(false),
            screenshots: AtomicU32::new(0),
        })
    }

    fn threaded(&self) -> bool {
        self.spawn.is_some()
    }

    fn wake(&self) {
        if let Some(t) = self.thread.lock().as_ref() {
            t.unpark();
        }
    }

    fn ensure_thread(self: &Arc<Self>) {
        let Some(spawn) = self.spawn else { return };
        let mut thread = self.thread.lock();
        if thread.is_none() {
            let shared = self.clone();
            let handle = std::thread::Builder::new()
                .name("Display-Refresh".into())
                .spawn(move || spawn(shared))
                .expect("spawn Display-Refresh");
            *thread = Some(handle.thread().clone());
        }
    }

    fn spawn(shared: Arc<Shared>) {
        crate::thread_affinity::pin_current(crate::thread_affinity::PerfRole::Rex3Refresh);
        while !shared.shutdown.load(Ordering::Acquire) {
            match shared.step() {
                Some(next) => {
                    let now = (shared.clock)();
                    if next > now {
                        std::thread::park_timeout(Duration::from_nanos(next - now));
                    }
                }
                None => std::thread::park(),
            }
        }
        // The machine is gone: release what the heads still hold.
        for slot in shared.slots.lock().iter_mut() {
            if let Some(handle) = slot.handle.upgrade() {
                slot.release(&handle);
            }
        }
    }

    fn step(&self) -> Option<u64> {
        let mut slots = self.slots.lock();
        slots.append(&mut self.incoming.lock());
        slots.retain(|s| s.handle.strong_count() > 0);
        let mut next = None;
        for slot in slots.iter_mut() {
            let Some(handle) = slot.handle.upgrade() else { continue };
            if let Some(t) = slot.step(self, &handle) {
                next = Some(next.map_or(t, |n: u64| n.min(t)));
            }
        }
        next
    }
}

impl Slot {
    fn field_time(&self, k: u64) -> u64 {
        self.epoch_ns + self.timing.field_offset_ns(k)
    }

    /// Deliver the line events due by `now`, where the thread does that
    /// itself; returns when the next is due. With timers they come on the
    /// timer thread and the thread has nothing to wake for.
    fn run_lines(shared: &Shared, handle: &HeadHandle, head: &dyn Head, now: u64) -> Option<u64> {
        if shared.timers.get().is_some() {
            return None;
        }
        handle.chain.lock().run(head, now)
    }

    /// End the field's line events: deliver the ones it still owes (or
    /// drop them), and stop its timer.
    fn end_lines(shared: &Shared, handle: &HeadHandle, deliver: Option<&dyn Head>) {
        let mut chain = handle.chain.lock();
        if let Some(head) = deliver {
            chain.run(head, u64::MAX);
        }
        chain.generation += 1;
        chain.lines.clear();
        chain.done = 0;
        if let (Some(id), Some(timers)) = (chain.timer.take(), shared.timers.get()) {
            timers.remove(id);
        }
    }

    /// A field's blank began at `start_ns`: its line events run from here.
    /// The first is delivered at once if it is due, and with timers each
    /// one's timer schedules the next, to the end of the field.
    fn begin_lines(&self, shared: &Shared, handle: &Arc<HeadHandle>, head: &dyn Head, field: u32, start_ns: u64) {
        let mut chain = handle.chain.lock();
        let per_field = self.timing.v_total / self.timing.fields.max(1);
        chain.lines.extend(handle.lines.lock().iter().copied().filter(|&l| l < per_field));
        (chain.field, chain.start_ns, chain.done) = (field, start_ns, 0);
        (chain.h_total, chain.pixel_clock_hz) = (self.timing.h_total, self.timing.pixel_clock_hz);
        let Some(timers) = shared.timers.get() else { return };
        let now = (shared.clock)();
        let Some(at) = chain.run(head, now) else { return };
        let context = (Arc::downgrade(handle), chain.generation);
        chain.timer = Some(timers.add_one_shot(Duration::from_nanos(at - now), context, |(handle, generation)| {
            use crate::hptimer::TimerReturn;
            let Some(handle) = handle.upgrade() else { return TimerReturn::Delete };
            let (Some(shared), Some(head)) = (handle.refresh.upgrade(), handle.board()) else { return TimerReturn::Delete };
            let mut chain = handle.chain.lock();
            if chain.generation != *generation {
                return TimerReturn::Delete;
            }
            let now = (shared.clock)();
            match chain.run(&*head, now) {
                Some(at) => TimerReturn::RescheduleOneShot(Duration::from_nanos(at - now)),
                None => {
                    chain.timer = None;
                    TimerReturn::Delete
                }
            }
        }));
    }

    fn rephase(&mut self, handle: &HeadHandle, epoch_ns: u64) {
        self.epoch_ns = epoch_ns;
        self.field = 0;
        // The lines of a field that is not finished are not reached.
        if let Some(shared) = handle.refresh.upgrade() {
            Self::end_lines(&shared, handle, None);
        }
        *handle.phase.lock() = Phase { timing: self.timing, epoch_ns };
    }

    /// The head stopped: end its blank, hand the renderer back, tell `stop`.
    fn release(&mut self, handle: &HeadHandle) {
        if !self.armed {
            // Stopped before its first step: the renderer is let go all
            // the same, so the next start begins clean.
            let stopping = *handle.stopping.lock();
            if stopping {
                handle.stop.finish(&handle.renderer);
                *handle.stopping.lock() = false;
            }
            handle.stopped.notify_all();
            return;
        }
        self.armed = false;
        if let Some(shared) = handle.refresh.upgrade() {
            Self::end_lines(&shared, handle, None);
        }
        if self.in_blank {
            self.in_blank = false;
            if let Some(head) = handle.board() {
                head.vblank_end((self.field % self.timing.fields as u64) as u32, self.field_time(self.field));
            }
        }
        handle.stop.finish(&handle.renderer);
        self.shown = false;
        self.size = (0, 0);
        *handle.stopping.lock() = false;
        handle.stopped.notify_all();
    }

    /// Run this head's due events; returns the time of its next one.
    fn step(&mut self, shared: &Shared, handle: &Arc<HeadHandle>) -> Option<u64> {
        if !handle.active.load(Ordering::Acquire) {
            self.release(handle);
            return None;
        }
        let Some(head) = handle.board() else {
            self.release(handle);
            return None;
        };
        let now = (shared.clock)();
        if !self.armed {
            self.armed = true;
            self.in_blank = false;
            (self.timing, self.timing_seq) = *handle.timing.lock();
            // The first retrace is one frame away.
            self.rephase(handle, now);
            self.field = self.timing.fields as u64;
        }

        loop {
            let now = (shared.clock)();
            let fields = self.timing.fields as u64;
            let field = (self.field % fields) as u32;
            if !self.in_blank {
                let (timing, seq) = *handle.timing.lock();
                if seq != self.timing_seq {
                    (self.timing, self.timing_seq) = (timing, seq);
                    self.rephase(handle, now);
                    self.field = self.timing.fields as u64;
                    continue;
                }
                let start = self.field_time(self.field);
                if now < start {
                    let line = Self::run_lines(shared, handle, &*head, now);
                    return Some(line.map_or(start, |t| t.min(start)));
                }
                let late = now - start;
                let start = if late > self.timing.field_period_ns() / LATE_TOLERANCE_DIV {
                    // The last frame overran, or the host held the thread:
                    // this retrace is now and the later ones follow from
                    // it. None is made up.
                    self.epoch_ns += late;
                    *handle.phase.lock() = Phase { timing: self.timing, epoch_ns: self.epoch_ns };
                    now
                } else {
                    start
                };
                // The lines the last field still owes come before this
                // retrace.
                Self::end_lines(shared, handle, Some(&*head));
                crate::pf_begin!(crate::perfetto::events::DISPLAY_VBLANK, field);
                head.vblank_start(field, start);
                // A field that begins late (the last frame overran) has
                // its lines as late: they keep their place in it.
                self.begin_lines(shared, handle, &*head, field, (shared.clock)().max(start));
                handle.retraces.fetch_add(1, Ordering::Relaxed);
                self.in_blank = true;
                // The blank lasts its length from when it really began: a
                // retrace delivered late (the last frame overran) still
                // gives the guest the whole blank to service it before the
                // frame is captured.
                self.blank_end_ns = (shared.clock)().max(start) + self.timing.blank_ns();
            }
            let end = self.blank_end_ns;
            if now < end {
                let line = Self::run_lines(shared, handle, &*head, now);
                return Some(line.map_or(end, |t| t.min(end)));
            }
            Self::run_lines(shared, handle, &*head, end);
            head.vblank_end(field, end);
            crate::pf_end!(crate::perfetto::events::DISPLAY_VBLANK);
            self.in_blank = false;
            self.field += 1;
            // Compose once a frame, after its last field.
            if self.field % fields == 0 {
                self.frame(shared, handle, &*head);
                // Back to the thread after a frame, so a head that is slow
                // to compose does not hold up the others' retraces.
                return Some(self.field_time(self.field).min((shared.clock)()));
            }
        }
    }

    fn readback_wanted(h: &HeadHandle) -> bool {
        h.screenshot.load(Ordering::Acquire) || h.frame_request.load(Ordering::Acquire) || h.capture.load(Ordering::Acquire) > 0
    }

    fn frame(&mut self, shared: &Shared, h: &HeadHandle, head: &dyn Head) {
        let now = (shared.clock)();
        let shot = h.screenshot.load(Ordering::Acquire);
        let readback = Self::readback_wanted(h);
        let said = h.damage.swap(false, Ordering::AcqRel) | head.changed();
        // Checking the verdicts: compose a frame the board would leave.
        let verify = verify::on() && !said && self.shown && self.cpu_frame;
        let changed = said || verify;
        // A new frame is due when the picture changed, or a reader needs
        // one the screen does not hold.
        let due = changed || !self.shown || (readback && !self.rgba_current && !self.cpu_frame);
        let stale = readback && !self.rgba_current;
        let heartbeat = now.saturating_sub(self.last_present_ns) >= HEARTBEAT_NS;
        if !(due || stale || heartbeat) {
            return;
        }

        let want = h.want();
        let mut screen = h.screen.lock();
        let mut renderer = h.renderer.lock();

        let composed = if !due {
            Composed::Unchanged
        } else {
            crate::pf_begin!(crate::perfetto::events::DISPLAY_CAPTURE);
            let captured = head.capture(&mut screen, want);
            crate::pf_end!(crate::perfetto::events::DISPLAY_CAPTURE);
            if captured {
                self.waiting = false;
                crate::pf_scope!(crate::perfetto::events::DISPLAY_COMPOSE);
                if verify::on() && screen.prebuilt {
                    self.verify.before(&screen);
                    let composed = head.compose(&mut screen, want);
                    if matches!(composed, Composed::Frame) && screen.prebuilt {
                        self.verify.after("head", &screen, said);
                    }
                    composed
                } else {
                    head.compose(&mut screen, want)
                }
            } else {
                crate::pf_instant!(crate::perfetto::events::DISPLAY_NOT_READY);
                Composed::NotReady
            }
        };
        let new_frame = match &composed {
            Composed::Frame => {
                self.cpu_frame = true;
                self.rgba_current = false;
                true
            }
            #[cfg(feature = "wgpu")]
            Composed::Gpu { .. } => {
                self.cpu_frame = false;
                self.rgba_current = false;
                true
            }
            Composed::Unchanged => false,
            Composed::NotReady => {
                h.damage.store(true, Ordering::Release);
                self.waiting = true;
                false
            }
        };
        let waiting = self.waiting;
        if !new_frame && !self.shown && !self.cpu_frame {
            // Nothing to show yet.
            return;
        }

        let screen = &mut *screen;
        let size = (screen.width, screen.height);
        if size.0 == 0 || size.1 == 0 {
            // No video mode: nothing to show, and nothing for a reader.
            h.frame_request.store(false, Ordering::Release);
            return;
        }
        if size != self.size {
            self.size = size;
            if let Some(r) = renderer.as_mut() {
                r.resize(size.0, size.1);
            }
        }

        // A finished frame is copied for its readers here; display state
        // (Newport) is read back by the renderer that composes it.
        let fill = readback && !self.rgba_current && self.cpu_frame;
        if fill {
            screen.ensure_rgba_rows(size.1);
        }
        if !(new_frame || fill || heartbeat) {
            // A head that asks every frame and had nothing new.
            return;
        }
        // Taken once per presented frame: it clears the activity lights.
        let stats = h.status.lock().clone().snapshot(Instant::now(), head.gfifo_pending());
        if fill && screen.prebuilt {
            for y in 0..size.1 {
                let row = y * FB_STRIDE;
                screen.rgba[row..row + size.0].copy_from_slice(&screen.fb_rgb[row..row + size.0]);
            }
            screen.frame_seq = screen.frame_seq.wrapping_add(1);
            self.rgba_current = true;
        }
        let renderer_reads_back = fill && !screen.prebuilt && renderer.is_some();
        if fill && !screen.prebuilt && renderer.is_none() {
            // No front-end: compose the guest's screen here, with no status
            // bar or overlay.
            let comp = self.reader.get_or_insert_with(crate::compositor::SwCompositor::new);
            comp.compose_pixels(&screen.compositor_source());
            let px = comp.pixels();
            for y in 0..size.1 {
                let row = y * FB_STRIDE;
                screen.rgba[row..row + size.0].copy_from_slice(&px[row..row + size.0]);
            }
            screen.frame_seq = screen.frame_seq.wrapping_add(1);
            self.rgba_current = true;
        }
        screen.status_bar_only = !new_frame && self.shown;
        head.overlay(&mut self.overlay);

        if let Some(r) = renderer.as_mut() {
            crate::pf_scope!(crate::perfetto::events::DISPLAY_PRESENT);
            // Capture reads every frame back: keep that a slice copy out of
            // the software compositor, never a GL readback.
            if h.capturing() {
                self.restore_gl.get_or_insert_with(|| r.gl_compositor());
                r.switch_compositor(false);
            } else if self.restore_gl.take() == Some(true) {
                r.switch_compositor(true);
            }
            match composed {
                #[cfg(feature = "wgpu")]
                Composed::Gpu { frame, width, height } => r.present_gpu_frame(
                    &frame, width, height, screen, &mut self.overlay, &mut self.status_bar, &mut self.sbtex, &stats,
                ),
                _ => {
                    let live = head.live_planes();
                    r.present(
                        screen, &mut self.overlay, &mut self.status_bar, &mut self.sbtex, &stats,
                        renderer_reads_back, live.map(|l| l.0), live.map(|l| l.1),
                    );
                }
            }
        }
        if renderer_reads_back {
            screen.frame_seq = screen.frame_seq.wrapping_add(1);
            self.rgba_current = true;
        }
        self.shown = true;
        self.last_present_ns = now;

        if self.rgba_current && !waiting {
            if shot && h.screenshot.swap(false, Ordering::AcqRel) {
                let pixels = screen.rgba.clone();
                let (w, hgt) = size;
                let n = shared.screenshots.fetch_add(1, Ordering::Relaxed);
                let path = format!("screenshot_{:04}.png", n);
                std::thread::spawn(move || match crate::disp::save_screenshot(&path, &pixels, w, hgt) {
                    Ok(()) => println!("iris: screenshot saved to {}", path),
                    Err(e) => println!("iris: screenshot failed: {}", e),
                });
            }
            h.frame_request.store(false, Ordering::Release);
        }
    }
}
