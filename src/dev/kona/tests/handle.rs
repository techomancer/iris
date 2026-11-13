//! The display handle a back end's threads call, and the lifecycle hooks the
//! board gives a back end: attach, stop, quiesce and resume.

use super::*;
use crate::traits::Device;

const ACTIVITY: u64 = 0x8003a8;
const INT_ENABLE: u64 = 0x800230;
const INT_CAUSE: u64 = 0x800238;
const RETRACE_TAG: u64 = 0x800248;
const INT_DEST_UPPER: u64 = 0x34;
const INT_DEST_LOWER: u64 = 0x3c;
const INT_VECTORS: u64 = 0x5c;
const CAUSE_VIDEO: u32 = 8;
const GENERAL_VECTOR: u8 = 0x10;

/// What a test says of the back end it stands in for: it notes the lifecycle calls and keeps the handle it is given.
#[derive(Default)]
struct Probe {
    events: Arc<Mutex<Vec<&'static str>>>,
    handle: Arc<Mutex<Option<DisplayHandle>>>,
    jobs: Arc<Mutex<u32>>,
    /// Panic on the first job, to kill the board.
    panics: bool,
}

impl Probe {
    /// The hook that notes the lifecycle, keeps the handle and counts jobs (or panics on one).
    fn hook(self) -> TestHook {
        let Probe { events, handle, jobs, panics } = self;
        TestHook {
            job: Box::new(move |_, _| {
                assert!(!panics, "the back end panics on a job");
                *jobs.lock() += 1;
                Some(Reply::Done)
            }),
            events,
            on_attach: Some(Box::new(move |h| *handle.lock() = Some(h))),
        }
    }
}

struct Rig {
    f: Fixture,
    events: Arc<Mutex<Vec<&'static str>>>,
    handle: Arc<Mutex<Option<DisplayHandle>>>,
    jobs: Arc<Mutex<u32>>,
}

impl Rig {
    fn new() -> Rig {
        Rig::with(Probe::default())
    }

    fn with(probe: Probe) -> Rig {
        let (events, handle, jobs) = (probe.events.clone(), probe.handle.clone(), probe.jobs.clone());
        let f = Fixture::idle(test_cfg(), probe.hook());
        Rig { f, events, handle, jobs }
    }

    fn handle(&self) -> DisplayHandle {
        self.handle.lock().clone().expect("attach gave a handle")
    }

    fn events(&self) -> Vec<&'static str> {
        self.events.lock().clone()
    }

    /// The driver's interrupt programming, and the video cause enabled.
    fn program_video(&self) {
        self.f.w(INT_DEST_UPPER, 0x98000);
        self.f.w(INT_DEST_LOWER, 0x180_0090);
        self.f.w(INT_VECTORS, 0x3e_2210);
        self.f.w(INT_ENABLE, 0xf7c);
    }
}

/// Starting the board attaches the back end before the thread runs; stopping
/// it stops the back end once, after the stores the thread had not taken ran.
#[test]
fn start_attaches_and_stop_stops_after_the_stream_ran() {
    let r = Rig::new();
    assert!(r.events().is_empty(), "nothing before the board starts");
    let board = TestBoard::new(r.f.kona.clone());
    board.start();
    r.f.settle();
    assert_eq!(r.events(), ["attach"]);
    for w in [1, 2, 7] {
        r.f.w(0xc10000, w);
    }
    board.stop();
    assert_eq!(r.events(), ["attach", "stop"]);
    assert_eq!(*r.jobs.lock(), 1, "the command ran before the back end stopped");
    // A stop with no thread running is not a stop of the back end.
    r.f.kona.stop();
    assert_eq!(r.events(), ["attach", "stop"]);
}

/// A reset while running stops the back end, begins a new epoch and attaches
/// again: the display thread of the old run is gone before the new one starts.
#[test]
fn a_reset_cycles_the_back_end() {
    let r = Rig::new();
    let board = TestBoard::new(r.f.kona.clone());
    board.start();
    r.f.settle();
    r.f.kona.power_on();
    assert_eq!(r.events(), ["attach", "stop", "reset", "attach"]);
    board.stop();
}

/// A ticket made before a reset cannot complete after it: the tag register
/// and the video cause stay as the reset left them. A ticket of the new epoch
/// completes. The reset joins the back end first, so this is the epoch check
/// alone, not a race with a display thread mid-call.
#[test]
fn a_ticket_from_before_a_reset_is_dropped() {
    let r = Rig::new();
    r.program_video();
    let board = TestBoard::new(r.f.kona.clone());
    board.start();
    r.f.settle();
    let h = r.handle();
    let old = h.epoch();
    r.f.kona.power_on();
    r.program_video();
    h.retrace_done(old, 0x55);
    h.video_frame(old);
    assert_eq!(r.f.r(RETRACE_TAG), 0);
    assert_eq!(r.f.r(INT_CAUSE) & CAUSE_VIDEO, 0);
    assert!(r.f.host.sent().is_empty());
    let now = r.handle().epoch();
    assert_ne!(now, old);
    h.retrace_done(now, 0x56);
    assert_eq!(r.f.r(RETRACE_TAG), 0x56);
    assert_eq!(r.f.r(INT_CAUSE) & CAUSE_VIDEO, CAUSE_VIDEO);
    assert_eq!(r.f.host.sent().len(), 1);
    board.stop();
}

/// A frame raises the video cause with no tag and sends one interrupt until
/// the driver acknowledges it.
#[test]
fn a_frame_raises_video_once_without_a_tag() {
    let r = Rig::new();
    r.program_video();
    r.f.kona.start();
    r.f.settle();
    let h = r.handle();
    let e = h.epoch();
    h.video_frame(e);
    h.video_frame(e);
    assert_eq!(r.f.r(INT_CAUSE) & CAUSE_VIDEO, CAUSE_VIDEO);
    assert_eq!(r.f.r(RETRACE_TAG), 0);
    assert_eq!(r.f.host.sent().len(), 1);
    assert_eq!(r.f.host.sent()[0].vector, GENERAL_VECTOR);
    r.f.w(INT_CAUSE, CAUSE_VIDEO as u32);
    h.video_frame(e);
    assert_eq!(r.f.host.sent().len(), 2);
    r.f.kona.stop();
}

/// With no destination programmed a frame sets the cause and sends nothing.
#[test]
fn a_frame_with_no_destination_sends_no_packet() {
    let r = Rig::new();
    r.f.w(INT_ENABLE, 0xf7c);
    r.f.kona.start();
    r.f.settle();
    let h = r.handle();
    for _ in 0..3 {
        h.video_frame(h.epoch());
    }
    assert_eq!(r.f.r(INT_CAUSE) & CAUSE_VIDEO, CAUSE_VIDEO);
    assert!(r.f.host.sent().is_empty());
    r.f.kona.stop();
}

/// The back end's activity reaches the driver's watchdog latch.
#[test]
fn activity_reaches_the_watchdog_latch() {
    let r = Rig::new();
    r.f.kona.start();
    assert_eq!(r.f.r(ACTIVITY), 0);
    r.handle().activity();
    assert_eq!(r.f.r(ACTIVITY), 4);
    r.f.w(ACTIVITY, 4);
    assert_eq!(r.f.r(ACTIVITY), 0);
    r.f.kona.stop();
}

/// The handle holds the register file weakly: with the board gone every call
/// is a no-op, and the handle does not keep the register file alive.
#[test]
fn a_handle_outlives_the_board_harmlessly() {
    let r = Rig::new();
    r.f.kona.start();
    r.f.settle();
    let h = r.handle();
    let weak = Arc::downgrade(&r.f.kona);
    drop(r);
    assert!(weak.upgrade().is_none(), "the board is freed with the handle outstanding");
    assert_eq!(h.epoch(), 0);
    h.retrace_done(0, 1);
    h.video_frame(0);
    h.activity();
}

/// Quiesce lets the stores in the ring run, then quiesces the back end;
/// resume undoes it.
#[test]
fn quiesce_runs_the_ring_then_the_back_end() {
    let r = Rig::new();
    let board = TestBoard::new(r.f.kona.clone());
    board.start();
    r.f.settle();
    for _ in 0..100 {
        for w in [1, 2, 7] {
            r.f.w(0xc10000, w);
        }
    }
    r.f.kona.quiesce();
    assert_eq!(*r.jobs.lock(), 100);
    assert_eq!(r.events(), ["attach", "quiesce"]);
    r.f.kona.resume();
    assert_eq!(r.events(), ["attach", "quiesce", "resume"]);
    board.stop();
}

/// A board whose back end panicked is dead, but stopping it still reaches the
/// back end: the next start attaches again and the first run's threads must
/// be gone by then.
#[test]
fn a_dead_board_still_stops_the_back_end() {
    let r = Rig::with(Probe { panics: true, ..Default::default() });
    let board = TestBoard::new(r.f.kona.clone());
    board.start();
    r.f.settle();
    for w in [1, 2, 7] {
        r.f.kona.bus_write(0xc10000, 4, w as u64);
    }
    retry(|| r.f.kona.bus_read(INT_CAUSE, 4).0 == crate::traits::BUS_ERR);
    assert!(r.f.kona.failure().unwrap().contains("panicked"));
    board.stop();
    assert_eq!(r.events(), ["attach", "stop"]);
}

/// Packets from several threads at once all reach the controller, each as one
/// set of its vector's pending bit: the order across threads does not change
/// the pending bits, and a second packet for a set bit is another call.
#[test]
fn concurrent_packets_all_reach_the_controller() {
    use super::super::front::InterruptPacket;
    let f = Fixture::new();
    let host = f.kona.shared.host.clone();
    let packet = |vector: u8| InterruptPacket { widget: 9, address: 0x180_0090, vector };
    std::thread::scope(|s| {
        for t in 0..4u8 {
            let host = &host;
            s.spawn(move || {
                for _ in 0..100 {
                    // Two threads share vector 0x10; the others have their own.
                    host.deliver(packet(if t < 2 { 0x10 } else { 0x20 + t }));
                }
            });
        }
    });
    let sent = f.host.sent();
    assert_eq!(sent.len(), 400);
    assert!(sent.iter().all(|p| p.level), "the board never sends a clear");
    assert_eq!(sent.iter().filter(|p| p.vector == 0x10).count(), 200);
    assert_eq!(sent.iter().filter(|p| p.vector == 0x22).count(), 100);
    assert_eq!(sent.iter().filter(|p| p.vector == 0x23).count(), 100);
    assert!(f.host.pending(0x10) && f.host.pending(0x22) && f.host.pending(0x23) && !f.host.pending(0x11));
    // The processor clearing a bit and a later packet setting it again are two interrupts.
    f.host.clear(0x10);
    assert!(!f.host.pending(0x10));
    host.deliver(packet(0x10));
    assert!(f.host.pending(0x10));
    assert_eq!(f.host.sent().len(), 401);
}

/// A packet for a widget other than the host's is counted and not delivered.
#[test]
fn unrouted_packets_are_counted_and_not_delivered() {
    use super::super::front::InterruptPacket;
    use std::sync::atomic::Ordering;
    let f = Fixture::new();
    let host = f.kona.shared.host.clone();
    host.deliver(InterruptPacket { widget: 3, address: 0x80, vector: 0x11 });
    host.deliver(InterruptPacket { widget: 4, address: 0x80, vector: 0x11 });
    assert!(f.host.sent().is_empty());
    assert_eq!(host.unrouted.load(Ordering::Relaxed), 2);
    host.deliver(InterruptPacket { widget: 9, address: 0x80, vector: 0x11 });
    assert_eq!(f.host.sent().len(), 1);
}
