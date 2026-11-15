//! The refresh schedule on a clock the test moves.

use super::*;
use crate::disp::BarStats;
use std::sync::atomic::AtomicUsize;

type Log = Arc<Mutex<Vec<String>>>;

/// A head that logs its callbacks and answers `compose` from a script.
struct TestHead {
    name: &'static str,
    log: Log,
    /// Answers still to give "not ready".
    not_ready: AtomicUsize,
    composes: AtomicUsize,
    size: (usize, usize),
    pixel: AtomicU32,
}

impl TestHead {
    fn new(name: &'static str, log: &Log) -> Arc<Self> {
        Arc::new(TestHead {
            name,
            log: log.clone(),
            not_ready: AtomicUsize::new(0),
            composes: AtomicUsize::new(0),
            size: (4, 2),
            pixel: AtomicU32::new(0xff00_0001),
        })
    }

    fn say(&self, what: &str) {
        self.log.lock().push(format!("{} {}", self.name, what));
    }
}

impl Head for TestHead {
    fn vblank_start(&self, field: u32, _t: u64) {
        self.say(&format!("start{field}"));
    }
    fn vblank_end(&self, field: u32, _t: u64) {
        self.say(&format!("end{field}"));
    }
    fn capture(&self, _screen: &mut Rex3Screen, _want: Want) -> bool {
        if self.not_ready.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1)).is_ok() {
            self.say("notready");
            return false;
        }
        self.say("capture");
        true
    }
    fn compose(&self, screen: &mut Rex3Screen, _want: Want) -> Composed {
        self.say("compose");
        self.composes.fetch_add(1, Ordering::SeqCst);
        (screen.width, screen.height) = self.size;
        screen.prebuilt = true;
        screen.fb_rgb[..self.size.0].fill(self.pixel.load(Ordering::SeqCst));
        Composed::Frame
    }
}

/// A renderer that logs presents and its teardown.
struct TestRenderer {
    name: &'static str,
    log: Log,
}

impl Renderer for TestRenderer {
    fn present(
        &mut self, screen: &mut Rex3Screen, _: &mut DebugOverlay, _: &mut StatusBar, _: &mut StatusBarTexture,
        _: &BarStats, _: bool, _: Option<&[u32]>, _: Option<&[u32]>,
    ) {
        let what = if screen.status_bar_only { "bar" } else { "present" };
        self.log.lock().push(format!("{} {}", self.name, what));
    }
    fn resize(&mut self, w: usize, h: usize) {
        self.log.lock().push(format!("{} resize {w}x{h}", self.name));
    }
    fn stop(&mut self) {
        self.log.lock().push(format!("{} rstop", self.name));
    }
}

struct Rig {
    clock: Arc<AtomicU64>,
    refresh: Arc<Refresh>,
    log: Log,
}

impl Rig {
    fn new() -> Self {
        let clock = Arc::new(AtomicU64::new(1_000));
        let c = clock.clone();
        Rig { clock, refresh: Refresh::manual(move || c.load(Ordering::SeqCst)), log: Log::default() }
    }

    fn head(&self, name: &'static str) -> (Arc<TestHead>, Arc<HeadHandle>) {
        let head = TestHead::new(name, &self.log);
        let weak: Weak<dyn Head> = Arc::downgrade(&(head.clone() as Arc<dyn Head>));
        let handle = self.refresh.add_head(weak, StatusSource::detached());
        *handle.renderer_slot().lock() = Some(Box::new(TestRenderer { name, log: self.log.clone() }));
        (head, handle)
    }

    /// Run every deadline up to `ns` after the present time, as the thread
    /// would: wake at each, never late.
    fn run(&self, ns: u64) {
        let until = self.clock.load(Ordering::SeqCst) + ns;
        loop {
            match self.refresh.step() {
                Some(next) if next <= until => { self.clock.fetch_max(next, Ordering::SeqCst); }
                _ => break,
            }
        }
        self.clock.fetch_max(until, Ordering::SeqCst);
        self.refresh.step();
    }

    /// Move the clock without waking: the thread was held up.
    fn stall(&self, ns: u64) {
        self.clock.fetch_add(ns, Ordering::SeqCst);
        self.refresh.step();
    }

    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.log.lock())
    }

    fn count(&self, what: &str) -> usize {
        self.log.lock().iter().filter(|l| l.as_str() == what).count()
    }
}

const MS: u64 = 1_000_000;
const HZ60_NS: u64 = 16_666_666;

#[test]
fn a_frame_runs_in_one_order() {
    let rig = Rig::new();
    let (_head, handle) = rig.head("a");
    handle.start();
    // Nothing before the first retrace, one period after the start.
    rig.run(HZ60_NS - MS);
    assert!(rig.take().is_empty());
    rig.run(2 * MS);
    assert_eq!(rig.take(), ["a start0", "a end0", "a capture", "a compose", "a resize 4x2", "a present"]);
}

#[test]
fn the_blank_lasts_its_lines() {
    let rig = Rig::new();
    let (_head, handle) = rig.head("a");
    handle.start();
    let blank = VideoTiming::HZ60.blank_ns();
    assert!((600_000..700_000).contains(&blank), "{blank}");
    rig.run(HZ60_NS + blank / 2);
    assert_eq!(rig.take(), ["a start0"]);
    let now = handle.now_ns();
    assert!(handle.beam(now).in_vblank);
    assert_eq!(handle.beam(now).line, 21);
    rig.run(blank);
    assert_eq!(rig.take()[0], "a end0");
    assert!(!handle.beam(handle.now_ns()).in_vblank);
}

#[test]
fn the_rate_is_the_timings() {
    let rig = Rig::new();
    let (_head, handle) = rig.head("a");
    // IMPACT's 1280x1024_76.
    let t = VideoTiming { h_total: 1712, v_total: 1079, pixel_clock_hz: 140_250_000, vblank_lines: 55, fields: 1 };
    assert_eq!(t.refresh_mhz(), 75_923);
    assert!(handle.set_timing(t));
    handle.start();
    rig.run(10_000 * MS);
    assert_eq!(handle.retraces(), 759);
    assert_eq!(rig.count("a start0"), 759);
}

#[test]
fn a_timing_the_thread_cannot_run_is_refused() {
    let rig = Rig::new();
    let (_head, handle) = rig.head("a");
    let bad = VideoTiming { pixel_clock_hz: 0, ..VideoTiming::HZ60 };
    assert!(!handle.set_timing(bad));
    // 10 kHz.
    let fast = VideoTiming { h_total: 100, v_total: 100, pixel_clock_hz: 100_000_000, vblank_lines: 1, fields: 1 };
    assert!(!handle.set_timing(fast));
    assert_eq!(handle.timing(), VideoTiming::HZ60);
}

#[test]
fn an_unchanged_picture_is_not_composed_again() {
    let rig = Rig::new();
    let (head, handle) = rig.head("a");
    handle.start();
    rig.run(1_000 * MS);
    assert_eq!(head.composes.load(Ordering::SeqCst), 1);
    // The status bar still moves, ten times a second.
    let bars = rig.count("a bar");
    assert!((9..=10).contains(&bars), "{bars}");
    assert_eq!(rig.count("a present"), 1);
    rig.take();

    handle.damage();
    rig.run(20 * MS);
    assert_eq!(head.composes.load(Ordering::SeqCst), 2);
    assert_eq!(rig.count("a present"), 1);
}

/// Newport's fence: a head that is not ready keeps the old frame on screen
/// and is asked again every retrace until it is.
#[test]
fn a_head_that_is_not_ready_is_asked_again() {
    let rig = Rig::new();
    let (head, handle) = rig.head("a");
    handle.start();
    rig.run(20 * MS);
    rig.take();

    head.not_ready.store(3, Ordering::SeqCst);
    handle.damage();
    rig.run(3 * HZ60_NS);
    let log = rig.take();
    assert_eq!(log.iter().filter(|l| *l == "a notready").count(), 3);
    assert!(!log.iter().any(|l| l == "a present" || l == "a compose" || l == "a capture"), "{log:?}");
    rig.run(HZ60_NS);
    assert_eq!(rig.take(), ["a start0", "a end0", "a capture", "a compose", "a present"]);
}

#[test]
fn missed_retraces_are_delivered_without_composing() {
    let rig = Rig::new();
    let (head, handle) = rig.head("a");
    handle.start();
    rig.run(20 * MS);
    rig.take();
    handle.damage();
    // Held up for three periods: three retraces are owed.
    rig.stall(3 * HZ60_NS);
    assert_eq!(
        rig.take(),
["a start0", "a end0", "a start0", "a end0", "a start0", "a end0", "a capture", "a compose", "a present"]
    );
    assert_eq!(head.composes.load(Ordering::SeqCst), 2);
}

/// A host too slow to capture and compose within a period still shows
/// frames: fewer of them, with every retrace delivered.
#[test]
fn a_slow_capture_does_not_starve_the_screen() {
    struct Slow {
        clock: Arc<AtomicU64>,
        frames: AtomicUsize,
    }
    impl Head for Slow {
        fn capture(&self, _: &mut Rex3Screen, _: Want) -> bool {
            self.clock.fetch_add(40 * MS, Ordering::SeqCst);
            true
        }
        fn changed(&self) -> bool { true }
        fn compose(&self, screen: &mut Rex3Screen, _: Want) -> Composed {
            (screen.width, screen.height) = (4, 2);
            screen.prebuilt = true;
            self.frames.fetch_add(1, Ordering::SeqCst);
            Composed::Frame
        }
    }
    let rig = Rig::new();
    let head = Arc::new(Slow { clock: rig.clock.clone(), frames: AtomicUsize::new(0) });
    let weak: Weak<dyn Head> = Arc::downgrade(&(head.clone() as Arc<dyn Head>));
    let handle = rig.refresh.add_head(weak, StatusSource::detached());
    handle.start();
    let began = rig.clock.load(Ordering::SeqCst);
    rig.run(1_000 * MS);
    // The captures push the clock past the end of the run.
    let periods = (rig.clock.load(Ordering::SeqCst) - began) / HZ60_NS;
    let frames = head.frames.load(Ordering::SeqCst) as u64;
    assert!(frames >= periods / 4 && frames <= periods / 2, "{frames} frames in {periods} periods");
    assert!(handle.retraces() + 4 >= periods, "{} retraces in {periods} periods", handle.retraces());
}

#[test]
fn a_long_stall_restarts_the_phase() {
    let rig = Rig::new();
    let (_head, handle) = rig.head("a");
    handle.start();
    rig.run(20 * MS);
    let before = handle.retraces();
    rig.stall(5_000 * MS);
    assert_eq!(handle.retraces(), before + 1);
    // And the schedule carries on from there.
    rig.run(HZ60_NS + MS);
    assert_eq!(handle.retraces(), before + 2);
}

#[test]
fn a_new_timing_restarts_the_phase() {
    let rig = Rig::new();
    let (_head, handle) = rig.head("a");
    handle.start();
    rig.run(HZ60_NS + 5 * MS);
    rig.take();
    // 72 Hz: 13.9 ms. The next retrace is one new period from now, not at
    // the old schedule's 16.7 ms.
    let t = VideoTiming { h_total: 1712, v_total: 1063, pixel_clock_hz: 131_000_000, vblank_lines: 39, fields: 1 };
    assert!(handle.set_timing(t));
    rig.run(t.field_period_ns() - MS);
    assert_eq!(rig.count("a start0"), 0);
    rig.run(2 * MS);
    assert_eq!(rig.count("a start0"), 1);
}

#[test]
fn two_heads_keep_their_own_rates() {
    let rig = Rig::new();
    let (_a, ha) = rig.head("a");
    let (_b, hb) = rig.head("b");
    let t = VideoTiming { h_total: 1712, v_total: 1079, pixel_clock_hz: 140_250_000, vblank_lines: 55, fields: 1 };
    hb.set_timing(t);
    ha.start();
    hb.start();
    rig.run(995 * MS);
    assert_eq!(ha.retraces(), 59);
    assert_eq!(hb.retraces(), 75);
}

/// Field-sequential stereo: a blank per field, a compose per frame.
#[test]
fn a_stereo_frame_has_two_blanks_and_one_compose() {
    let rig = Rig::new();
    let (_head, handle) = rig.head("a");
    // 1280x492_120s.
    let t = VideoTiming { h_total: 1680, v_total: 1064, pixel_clock_hz: 107_251_200, vblank_lines: 40, fields: 2 };
    assert_eq!(t.refresh_mhz(), 60_000);
    handle.set_timing(t);
    handle.start();
    // The first retrace is a whole frame after the start, on field 0.
    rig.run(HZ60_NS + HZ60_NS / 2 + 2 * MS);
    assert_eq!(
        rig.take(),
["a start0", "a end0", "a start1", "a end1", "a capture", "a compose", "a resize 4x2", "a present"]
    );
}

#[test]
fn a_requested_frame_is_read_back_once() {
    let rig = Rig::new();
    let (head, handle) = rig.head("a");
    handle.start();
    rig.run(20 * MS);
    assert_eq!(handle.screen().lock().frame_seq, 0);

    // The frame on screen is served without composing it again.
    handle.request_frame();
    assert!(handle.frame_pending());
    rig.run(20 * MS);
    assert!(!handle.frame_pending());
    assert_eq!(head.composes.load(Ordering::SeqCst), 1);
    {
        let screen = handle.screen().lock();
        assert_eq!(screen.frame_seq, 1);
        assert_eq!(screen.rgba[0], 0xff00_0001);
    }

    // A request while the head is not ready waits for the new frame.
    head.pixel.store(0xff00_0002, Ordering::SeqCst);
    head.not_ready.store(2, Ordering::SeqCst);
    handle.damage();
    rig.run(20 * MS);
    handle.request_frame();
    rig.run(HZ60_NS);
    assert_eq!(handle.screen().lock().rgba[0], 0xff00_0001);
    rig.run(3 * HZ60_NS);
    assert!(!handle.frame_pending());
    assert_eq!(handle.screen().lock().rgba[0], 0xff00_0002);
}

#[test]
fn capture_keeps_the_frame_in_memory() {
    let rig = Rig::new();
    let (head, handle) = rig.head("a");
    handle.set_capture(true);
    handle.start();
    rig.run(20 * MS);
    assert_eq!(handle.screen().lock().frame_seq, 1);
    head.pixel.store(0xff00_0002, Ordering::SeqCst);
    handle.damage();
    rig.run(20 * MS);
    assert_eq!(handle.screen().lock().frame_seq, 2);
    assert_eq!(handle.screen().lock().rgba[0], 0xff00_0002);
    handle.set_capture(false);
    handle.damage();
    rig.run(20 * MS);
    assert_eq!(handle.screen().lock().frame_seq, 2);
}

#[test]
fn stop_ends_the_blank_and_releases_the_renderer() {
    let rig = Rig::new();
    let (_head, handle) = rig.head("a");
    handle.start();
    rig.run(HZ60_NS + 100_000);
    assert_eq!(rig.take(), ["a start0"]);
    handle.stop();
    assert_eq!(rig.take(), ["a end0", "a rstop"]);
    rig.run(100 * MS);
    assert!(rig.take().is_empty());

    // Started again, it sizes and shows the frame anew.
    handle.start();
    rig.run(20 * MS);
    assert_eq!(rig.take(), ["a start0", "a end0", "a capture", "a compose", "a resize 4x2", "a present"]);
}

/// The main thread is the one a macOS renderer needs to be torn down, so a
/// stop from it leaves the renderer alone (see `DisplayStop`).
#[test]
fn a_stop_from_the_main_thread_leaves_the_renderer() {
    let rig = Rig::new();
    let (_head, handle) = rig.head("a");
    handle.start();
    rig.run(20 * MS);
    rig.take();
    let h = handle.clone();
    std::thread::Builder::new().name("main".into()).spawn(move || h.stop()).unwrap().join().unwrap();
    assert!(rig.take().is_empty());
}

#[test]
fn a_dropped_head_leaves_the_schedule() {
    let rig = Rig::new();
    let (head, handle) = rig.head("a");
    handle.start();
    rig.run(20 * MS);
    drop(head);
    assert_eq!(rig.refresh.step(), None);
}

/// The real thread on the real clock.
#[test]
fn the_thread_runs_and_stops() {
    let refresh = Refresh::new();
    let log = Log::default();
    let head = TestHead::new("a", &log);
    let weak: Weak<dyn Head> = Arc::downgrade(&(head.clone() as Arc<dyn Head>));
    let handle = refresh.add_head(weak, StatusSource::detached());
    *handle.renderer_slot().lock() = Some(Box::new(TestRenderer { name: "a", log: log.clone() }));
    handle.start();
    let deadline = Instant::now() + Duration::from_secs(5);
    while handle.retraces() < 5 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(handle.retraces() >= 5);
    handle.stop();
    let n = handle.retraces();
    assert_eq!(log.lock().last().map(String::as_str), Some("a rstop"));
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(handle.retraces(), n);

    {
        let _held = refresh.quiesce();
        handle.start();
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(handle.retraces(), n);
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while handle.retraces() == n && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(handle.retraces() > n);
    handle.stop();
}
