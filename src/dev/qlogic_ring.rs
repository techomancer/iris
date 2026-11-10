//! The ring thread's control plumbing, shared by the QLogic adapters.
//!
//! Each adapter walks its request ring on a thread of its own, because a
//! command can wait on a disk and a register access must not. What the thread
//! and the register side tell each other does not depend on the adapter's
//! ring format: a generation that a reset moves on, a fence so that no store
//! of cancelled work reaches guest memory after the reset returns, a hold for
//! saving state, a pause for a stopped machine with the time it stood still,
//! and the thread's start and stop. The ring walks stay with the adapters.
//!
//! The interrupt line's driving, the notes list `scsi notes` prints and the
//! guest-memory transfers are the same for every adapter too, and live here.
//!
//! A command's data length and how its buffer segments are placed and read
//! are the same on every adapter, and live here as plain functions.
//!
//! The adapters' timed waits use the clock here, host time less the time the
//! machine was stopped, because iris has no emulated-time source a device can
//! read. [`crate::dev::heart::Clock`] does not fit: it has no stopped-time
//! accounting, and the thread must know whether the clock it waits on is the
//! host's or one a test moves.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use parking_lot::{Condvar, Mutex};

use crate::irq::IrqLine;
use crate::pci::PciFunctionState;
use crate::scsi::scsi_cmd;

/// What the thread and the register side tell each other. An adapter's own
/// control state holds one of these and lends it through [`RingState`].
#[derive(Default)]
pub struct RingCtl {
    /// Bumped by every reset: work that finishes under another generation
    /// was cancelled, and posts nothing.
    pub(crate) gen: u64,
    /// The ring may have work the thread has not looked at.
    pub(crate) work: bool,
    /// The thread is walking the ring.
    pub(crate) walking: bool,
    /// Holders that need the thread not to start more (saving and loading
    /// state).
    pub(crate) hold: u32,
    /// The machine is stopped: the thread takes no further work.
    pub(crate) paused: bool,
    pub(crate) shutdown: bool,
    /// When the adapter's clock last stopped with the machine, and how long
    /// it has stood still in all.
    pub(crate) stopped_at: Option<Instant>,
    pub(crate) stopped_ns: i64,
}

/// An adapter's state, which carries a [`RingCtl`].
pub trait RingState {
    fn ring_ctl(&self) -> &RingCtl;
    fn ring_ctl_mut(&mut self) -> &mut RingCtl;
}

/// The state lock, its signal, the fence and the thread.
pub struct RingThread<S> {
    pub(crate) st: Mutex<S>,
    /// `ring_ctl().gen`, readable without `st`: what a store into guest
    /// memory checks under `fence`.
    pub(crate) gen: AtomicU64,
    /// Held across each store the ring thread makes into guest memory. A
    /// reset moves the generation on and then passes through it, so once the
    /// reset returns no store of a cancelled command is under way or can
    /// start; the reset waits for one store, never for a disk.
    pub(crate) fence: Mutex<()>,
    /// Signals [`RingCtl`] changes both ways.
    pub(crate) wake: Condvar,
    pub(crate) worker: Mutex<Option<JoinHandle<()>>>,
    /// Where the adapter's clock starts.
    pub(crate) start: Instant,
    /// When not `i64::MIN`, the clock in nanoseconds, for tests.
    pub(crate) fake_ns: AtomicI64,
}

impl<S: RingState> RingThread<S> {
    pub fn new(state: S) -> Self {
        let gen = state.ring_ctl().gen;
        Self {
            st: Mutex::new(state),
            gen: AtomicU64::new(gen),
            fence: Mutex::new(()),
            wake: Condvar::new(),
            worker: Mutex::new(None),
            start: Instant::now(),
            fake_ns: AtomicI64::new(i64::MIN),
        }
    }

    /// Tell the ring thread the ring may have moved.
    pub fn kick(&self, st: &mut S) {
        st.ring_ctl_mut().work = true;
        self.wake.notify_all();
    }

    pub fn publish_gen(&self, st: &S) {
        self.gen.store(st.ring_ctl().gen, Ordering::Release);
    }

    /// Wait out a store into guest memory that a cancelled command began
    /// before the generation moved on; see `fence`.
    pub fn retire_stores(&self) {
        drop(self.fence.lock());
    }

    /// Run `store` for work of generation `gen`, unless a reset has cancelled
    /// that work.
    pub fn fenced<R>(&self, gen: u64, store: impl FnOnce() -> R) -> Option<R> {
        let _fence = self.fence.lock();
        (self.gen.load(Ordering::Acquire) == gen).then(store)
    }

    /// The adapter's clock in nanoseconds: host time less the time the
    /// machine was stopped.
    pub fn now_ns(&self, st: &S) -> i64 {
        match self.fake_ns.load(Ordering::Relaxed) {
            i64::MIN => {
                let ctl = st.ring_ctl();
                let at = ctl.stopped_at.unwrap_or_else(Instant::now);
                at.duration_since(self.start).as_nanos() as i64 - ctl.stopped_ns
            }
            t => t,
        }
    }

    /// Start the ring thread (idempotent), running `run` on `core`, and let
    /// it take whatever was published while the adapter was stopped.
    pub fn start_thread<C: Send + Sync + 'static>(&self, core: &Arc<C>, name: String, run: fn(&C)) {
        let mut worker = self.worker.lock();
        if worker.is_some() {
            return;
        }
        self.st.lock().ring_ctl_mut().shutdown = false;
        let core = core.clone();
        *worker = Some(
            std::thread::Builder::new().name(name).spawn(move || run(&core)).expect("spawn ring thread"),
        );
        drop(worker);
        self.resume();
    }

    /// Park the ring between entries, then end and join the thread. A command
    /// in flight finishes and posts first; afterwards nothing touches guest
    /// memory or the interrupt line.
    pub fn stop_thread(&self) {
        self.pause();
        let Some(handle) = self.worker.lock().take() else { return };
        self.st.lock().ring_ctl_mut().shutdown = true;
        self.wake.notify_all();
        let _ = handle.join();
    }

    /// Stop the ring thread between entries while the machine is stopped:
    /// once this returns the adapter touches neither guest memory nor its
    /// interrupt until [`RingThread::resume`]. A command in flight finishes and
    /// posts first.
    pub fn pause(&self) {
        let mut st = self.st.lock();
        let ctl = st.ring_ctl_mut();
        ctl.paused = true;
        ctl.stopped_at.get_or_insert_with(Instant::now);
        // Parked mid-wait, the thread lets go of its entry.
        self.wake.notify_all();
        while st.ring_ctl().walking {
            self.wake.wait(&mut st);
        }
    }

    /// Let the ring thread run again, taking up whatever was published while
    /// it was paused.
    pub fn resume(&self) {
        let mut st = self.st.lock();
        let ctl = st.ring_ctl_mut();
        ctl.paused = false;
        if let Some(t) = ctl.stopped_at.take() {
            ctl.stopped_ns += t.elapsed().as_nanos() as i64;
        }
        self.kick(&mut st);
    }

    /// Run `f` on the state with the ring thread stopped between entries and
    /// none allowed to start: the quiesce point for saving and loading. A
    /// command in flight posts its completion first.
    pub fn quiesced<R>(&self, f: impl FnOnce(&mut S) -> R) -> R {
        let mut st = self.st.lock();
        st.ring_ctl_mut().hold += 1;
        self.wake.notify_all();
        while st.ring_ctl().walking {
            self.wake.wait(&mut st);
        }
        let r = f(&mut st);
        st.ring_ctl_mut().hold -= 1;
        drop(st);
        self.wake.notify_all();
        r
    }
}

/// How many notes an adapter keeps; the first ones are the ones a person
/// debugging a guest needs.
const NOTES_MAX: usize = 256;

/// An adapter's running commentary on the odd things a driver did (accesses
/// of a width the part does not decode, DMA that is not memory, commands not
/// modelled), as `scsi notes` lists it. The lock is a leaf: nothing is taken
/// while it is held.
#[derive(Default)]
pub struct Notes {
    list: Mutex<Vec<String>>,
}

impl Notes {
    pub fn list(&self) -> Vec<String> {
        self.list.lock().clone()
    }

    pub fn push(&self, what: String) {
        self.push_with(|| what);
    }

    /// As [`Notes::push`], building the text only if there is room for it.
    pub fn push_with(&self, what: impl FnOnce() -> String) {
        let mut list = self.list.lock();
        if list.len() < NOTES_MAX {
            list.push(what());
        }
    }

    /// As [`Notes::push_with`] for something a guest does on every register
    /// access: only the first `max` of the events `seen` counts are noted, and
    /// after that the call takes no lock, so it costs the register path an
    /// atomic add and leaves room for the notes that follow.
    pub fn push_first(&self, seen: &AtomicU64, max: u64, what: impl FnOnce() -> String) {
        if seen.fetch_add(1, Ordering::Relaxed) < max {
            self.push_with(what);
        }
    }
}

/// What an adapter last drove its interrupt line to. It lives in the adapter's
/// state, so it is only reachable with the state lock held: the adapter
/// drives the line in the same hold of the lock that changed the state, and
/// the lock orders the edges. The line goes to its parent only when the level
/// changes, as the hardware's output pin does.
///
/// The parent must never call back into the adapter while it holds a lock the
/// line's delivery takes. The Bridge, HEART and the Hub's processor interface
/// all deliver and return without calling down. The one place that reaches an
/// adapter from the parent's side is `Bridge::reg_write`, which resets a PCI
/// function only after `irq_locked` has returned; keep it that way.
#[derive(Default)]
pub struct IrqLevel {
    /// The level last written to the line, or `None` when the controller's
    /// input may not be what was last written, as after power-on.
    last: Option<bool>,
}

impl IrqLevel {
    /// The level last written, for a test to check against the state.
    #[cfg(test)]
    pub fn last(&self) -> Option<bool> {
        self.last
    }

    /// The controller's input may differ from what was last written: the
    /// device was wired to a new line, or its state was loaded over the run
    /// the controller knew. The next drive writes the line even when its level
    /// has not changed, as the first one after power-on does.
    pub fn forget(&mut self) {
        self.last = None;
    }

    /// Drive `line` to `level` if it is not already there. Call it with the
    /// state lock held, with the level that state gives. A function the bus
    /// has not wired to a line yet has nothing to drive, and keeps what it
    /// remembers.
    pub fn drive(&mut self, level: bool, line: Option<&IrqLine>) {
        let Some(line) = line else { return };
        if self.last != Some(level) {
            self.last = Some(level);
            line.set(level);
        }
    }
}

/// `buf.len()` bytes from bus address `addr`, when `allowed` (the adapter may
/// master the bus). False, and `buf` zeroed, if it is not or the fabric
/// refuses any of them.
pub fn dma_read(pci: &PciFunctionState, allowed: bool, addr: u64, buf: &mut [u8]) -> bool {
    let ok = allowed && addr.checked_add(buf.len() as u64).is_some() && pci.dma().is_some_and(|dma| dma.dma_read(addr, buf).is_ok());
    if !ok {
        buf.iter_mut().for_each(|byte| *byte = 0);
    }
    ok
}

/// `data` to bus address `addr`, as [`dma_read`].
pub fn dma_write(pci: &PciFunctionState, allowed: bool, addr: u64, data: &[u8]) -> bool {
    allowed && addr.checked_add(data.len() as u64).is_some() && pci.dma().is_some_and(|dma| dma.dma_write(addr, data).is_ok())
}

/// How many bytes a command is declared to move, and whether they go to
/// the device. `block` is the target's logical block size. An opcode
/// with no rule here moves nothing; the target answers it as SCSI says.
pub fn cdb_transfer(cdb: &[u8], block: usize) -> (usize, bool) {
    let at = |i: usize| cdb.get(i).copied().unwrap_or(0) as usize;
    let be16 = |i: usize| (at(i) << 8) | at(i + 1);
    let be24 = |i: usize| (at(i) << 16) | (at(i + 1) << 8) | at(i + 2);
    let six = || if at(4) == 0 { 256 } else { at(4) };
    match cdb[0] {
        scsi_cmd::READ_6 => (six() * block, false),
        scsi_cmd::WRITE_6 => (six() * block, true),
        scsi_cmd::READ_10 => (be16(7) * block, false),
        scsi_cmd::WRITE_10 => (be16(7) * block, true),
        scsi_cmd::INQUIRY => (be16(3), false),
        scsi_cmd::REQUEST_SENSE | scsi_cmd::MODE_SENSE_6 => (at(4), false),
        scsi_cmd::MODE_SELECT_6 => (at(4), true),
        scsi_cmd::MODE_SENSE_10
        | scsi_cmd::READ_TOC_PMA_ATIP
        | scsi_cmd::GET_CONFIGURATION
        | scsi_cmd::READ_SUB_CHANNEL
        | scsi_cmd::READ_DISC_INFORMATION => (be16(7), false),
        scsi_cmd::MODE_SELECT_10 => (be16(7), true),
        scsi_cmd::READ_CAPACITY_10 => (8, false),
        scsi_cmd::READ_BUFFER => (be24(6), false),
        scsi_cmd::WRITE_BUFFER => (be24(6), true),
        scsi_cmd::SEND_DIAGNOSTIC => (be16(3), true),
        scsi_cmd::RECEIVE_DIAGNOSTIC_RESULTS => (be16(3), false),
        _ => (0, false),
    }
}

/// One descriptor into the transfer. The descriptors name the host's
/// buffer and the command names how much of it the target has to fill,
/// so only that much is offered and a longer buffer keeps its tail. The
/// whole named length is still counted: the residual is measured against
/// it.
pub fn add_segment(segs: &mut Vec<(u64, u32)>, described: &mut u64, max: u64, base: u64, len: u32) {
    let room = max.saturating_sub(*described);
    let placed = (len as u64).min(room);
    *described += len as u64;
    if placed > 0 {
        segs.push((base, placed as u32));
    }
}

/// Bytes of one 32-bit segment descriptor (address, length) in a command entry
/// or a continuation.
pub const SEG32_SIZE: usize = 8;
/// Bytes of one 64-bit segment descriptor (low address, high address, length).
pub const SEG64_SIZE: usize = 12;
/// Where the segments start in a continuation entry, and how many it holds, in
/// each format.
pub const CONT_OFF_SEGS32: usize = 8;
pub const CONT_SEGS32: usize = 7;
pub const CONT_OFF_SEGS64: usize = 4;
pub const CONT_SEGS64: usize = 5;

/// Reads `n` segment descriptors from `p`: 32-bit (address, length) pairs, or
/// 64-bit address and 32-bit length triples when `a64`. `p` must hold them.
pub fn parse_segments(segs: &mut Vec<(u64, u32)>, p: &[u8], n: usize, a64: bool) {
    let le32 = |at: usize| u32::from_le_bytes(p[at..at + 4].try_into().unwrap());
    for i in 0..n {
        segs.push(if a64 {
            let at = i * SEG64_SIZE;
            (u64::from(le32(at)) | (u64::from(le32(at + 4)) << 32), le32(at + 8))
        } else {
            let at = i * SEG32_SIZE;
            (u64::from(le32(at)), le32(at + 4))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::irq::{LevelProbe, StallingProbe};

    /// Stands for an adapter's state: the level it gives and the line's
    /// memory of it, under one lock.
    struct Adapter {
        state: Mutex<(bool, IrqLevel)>,
        line: IrqLine,
    }

    impl Adapter {
        fn new(line: IrqLine) -> Arc<Self> {
            Arc::new(Adapter { state: Mutex::new((false, IrqLevel::default())), line })
        }

        /// Change the level and drive the line in one critical section.
        fn set(&self, level: bool) {
            let mut st = self.state.lock();
            st.0 = level;
            let (level, irq) = (st.0, &mut st.1);
            irq.drive(level, Some(&self.line));
        }
    }

    /// Thread B raises the level and stalls in the parent's delivery. C then
    /// tries to drop it. C must wait for B, so the parent sees a rise and then
    /// a fall in that order, and the line ends low. This pins the primitive
    /// with a stand-in for an adapter; the adapters' own tests, which drive
    /// their real register paths, are in `isp1040/tests.rs` and
    /// `isp2200/tests.rs`.
    #[test]
    fn a_stalled_delivery_is_not_overtaken() {
        let (stall, line) = StallingProbe::line();
        let adapter = Adapter::new(line);
        let (started, go) = stall.arm();
        std::thread::scope(|scope| {
            let b = adapter.clone();
            scope.spawn(move || b.set(true));
            started.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
            let c = adapter.clone();
            let done = scope.spawn(move || c.set(false));
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(!done.is_finished(), "C waits for B's delivery");
            go.send(()).unwrap();
        });
        assert_eq!(stall.probe.writes(), [true, false], "no spurious edge, and in order");
        assert!(!stall.probe.level());
        assert!(!adapter.state.lock().0);
    }

    /// Many threads raise and drop the level. Whatever the interleaving, each
    /// edge the parent sees is a change of the level, in the order the levels
    /// were set, and the line ends where the last one left it.
    #[test]
    fn racing_adapters_drive_only_real_edges() {
        for round in 0..200u32 {
            let (probe, line) = LevelProbe::line();
            let adapter = Adapter::new(line);
            // The levels in the order they were set, recorded under the same
            // lock that drives.
            let set = Arc::new(Mutex::new(Vec::new()));
            let workers: Vec<_> = (0..6u32)
                .map(|t| {
                    let (adapter, set) = (adapter.clone(), set.clone());
                    std::thread::spawn(move || {
                        for i in 0..200u32 {
                            let level = (i + t + round) % 3 != 0;
                            let mut st = adapter.state.lock();
                            st.0 = level;
                            set.lock().push(level);
                            let irq = &mut st.1;
                            irq.drive(level, Some(&adapter.line));
                        }
                    })
                })
                .collect();
            workers.into_iter().for_each(|w| w.join().unwrap());
            let mut want = set.lock().clone();
            want.dedup();
            assert_eq!(probe.writes(), want, "round {round}: one write per change, in order");
            assert_eq!(probe.level(), adapter.state.lock().0, "round {round}");
        }
    }

    #[test]
    fn a_forgotten_line_is_written_again() {
        let (probe, line) = LevelProbe::line();
        let mut irq = IrqLevel::default();
        irq.drive(true, Some(&line));
        irq.drive(true, Some(&line));
        assert_eq!(probe.writes(), [true]);
        irq.forget();
        irq.drive(true, Some(&line));
        irq.drive(true, Some(&line));
        assert_eq!(probe.writes(), [true, true], "one write for the forgetting, none after");
    }

    #[test]
    fn a_function_without_a_line_keeps_what_it_remembers() {
        let (probe, line) = LevelProbe::line();
        let mut irq = IrqLevel::default();
        irq.drive(true, None);
        irq.drive(true, Some(&line));
        assert_eq!(probe.writes(), [true], "the first level reaches the line once it is wired");
    }

    #[test]
    fn segments_read_in_both_formats() {
        let mut p = Vec::new();
        p.extend_from_slice(&0x1122_3344u32.to_le_bytes());
        p.extend_from_slice(&0x200u32.to_le_bytes());
        p.extend_from_slice(&0x5566_7788u32.to_le_bytes());
        p.extend_from_slice(&0x300u32.to_le_bytes());
        let mut segs = Vec::new();
        parse_segments(&mut segs, &p, 2, false);
        assert_eq!(segs, [(0x1122_3344, 0x200), (0x5566_7788, 0x300)]);

        let mut p = Vec::new();
        p.extend_from_slice(&0x1122_3344u32.to_le_bytes());
        p.extend_from_slice(&0x7u32.to_le_bytes());
        p.extend_from_slice(&0x400u32.to_le_bytes());
        let mut segs = Vec::new();
        parse_segments(&mut segs, &p, 1, true);
        assert_eq!(segs, [(0x7_1122_3344, 0x400)]);
    }

    #[test]
    fn a_long_buffer_is_cut_to_the_declared_length_and_counted_whole() {
        let (mut segs, mut described) = (Vec::new(), 0);
        add_segment(&mut segs, &mut described, 0x300, 0x1000, 0x200);
        add_segment(&mut segs, &mut described, 0x300, 0x2000, 0x200);
        add_segment(&mut segs, &mut described, 0x300, 0x3000, 0x200);
        assert_eq!(segs, [(0x1000, 0x200), (0x2000, 0x100)]);
        assert_eq!(described, 0x600);
    }

    #[test]
    fn transfer_lengths_follow_the_cdb() {
        assert_eq!(cdb_transfer(&[scsi_cmd::READ_10, 0, 0, 0, 0, 0, 0, 0, 4, 0], 512), (2048, false));
        assert_eq!(cdb_transfer(&[scsi_cmd::WRITE_6, 0, 0, 0, 0, 0], 512), (256 * 512, true));
        assert_eq!(cdb_transfer(&[scsi_cmd::TEST_UNIT_READY, 0, 0, 0, 0, 0], 512), (0, false));
    }
}
