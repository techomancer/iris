//! What several CPU threads share: which CPU this thread is, and how one
//! thread holds another CPU still.
//!
//! A held CPU stops at its next batch boundary, outside compiled code and
//! with its executor unlocked. Bit [`HOST_PARK_BIT`] of the CPU's interrupt
//! word is the request: every preamble, compiled ones included, bails on any
//! set bit, so a CPU in a tight compiled loop falls out at once and reaches
//! the boundary within a batch. Holding is cold; the running CPU pays one load
//! per batch.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

/// Most CPUs HEART has slots for. IRIX's Octane kernel and PROM use two.
pub const MAX_CPUS: usize = 4;

/// Bit of `MipsCore::hot.interrupts` asking the CPU to stop at its next batch
/// boundary. Bit 63 is the soft-reset request; Cause only takes bits 8..15.
pub const HOST_PARK_BIT: u64 = 1 << 62;

/// HEART's MODE register disables this processor.
pub const HOLD_HEART: u32 = 1 << 0;
/// Another thread needs the machine still ([`World::stop_others`]).
pub const HOLD_WORLD: u32 = 1 << 1;

const NOT_A_CPU: u8 = u8::MAX;

thread_local! {
    static CURRENT_CPU: Cell<u8> = const { Cell::new(NOT_A_CPU) };
}

/// This thread runs CPU `id`.
pub fn set_current_cpu(id: u8) {
    CURRENT_CPU.with(|c| c.set(id));
}

/// The CPU this thread runs, if it runs one.
pub fn on_cpu_thread() -> Option<usize> {
    match CURRENT_CPU.with(Cell::get) {
        NOT_A_CPU => None,
        id => Some(id as usize),
    }
}

/// The CPU a device sees on the bus. Device and monitor threads count as CPU 0,
/// which is what the single-CPU machine always answered.
pub fn current_cpu() -> usize {
    on_cpu_thread().unwrap_or(0)
}

#[derive(Default)]
struct SlotState {
    /// `HOLD_*` bits. The CPU may run when this is zero.
    holds: u32,
    /// The CPU is not touching guest state, and will not until released.
    parked: bool,
}

/// One CPU's place to wait.
pub struct ParkSlot {
    word: *const AtomicU64,
    /// A CPU thread exists for this CPU. A stopped CPU counts as parked.
    active: AtomicBool,
    state: Mutex<SlotState>,
    changed: Condvar,
}

// SAFETY: `word` points into the CPU's `MipsCore`, which outlives the slot
// (`MipsCpu::interrupts_ptr`'s contract), and is only used atomically.
unsafe impl Send for ParkSlot {}
unsafe impl Sync for ParkSlot {}

impl ParkSlot {
    /// `word` must outlive the slot.
    pub fn new(word: *const AtomicU64) -> Self {
        Self { word, active: AtomicBool::new(false), state: Mutex::new(SlotState::default()), changed: Condvar::new() }
    }

    fn word(&self) -> &AtomicU64 {
        // SAFETY: see `new`.
        unsafe { &*self.word }
    }

    /// Ask the CPU to stop. Returns at once; see [`Self::wait_parked`].
    pub fn hold(&self, why: u32) {
        let mut s = self.state.lock();
        s.holds |= why;
        self.word().fetch_or(HOST_PARK_BIT, Ordering::SeqCst);
        #[cfg(feature = "idle-pause")]
        crate::cpu::idle_park::wake();
    }

    /// Drop one reason to stop; the CPU continues when none is left.
    pub fn release(&self, why: u32) {
        let mut s = self.state.lock();
        s.holds &= !why;
        if s.holds == 0 {
            self.word().fetch_and(!HOST_PARK_BIT, Ordering::SeqCst);
            self.changed.notify_all();
        }
    }

    /// Why the CPU is held, for the monitor.
    pub fn holds(&self) -> u32 {
        self.state.lock().holds
    }

    pub fn is_parked(&self) -> bool {
        self.state.lock().parked || !self.active.load(Ordering::SeqCst)
    }

    /// Wait until the CPU has stopped touching guest state.
    pub fn wait_parked(&self) {
        let mut s = self.state.lock();
        while !s.parked && self.active.load(Ordering::SeqCst) {
            self.changed.wait_for(&mut s, Duration::from_millis(1));
        }
    }

    /// A CPU thread is starting or has ended.
    pub fn set_active(&self, on: bool) {
        let _s = self.state.lock();
        self.active.store(on, Ordering::SeqCst);
        self.changed.notify_all();
    }

    /// The CPU thread, with its executor unlocked: wait out the holds, or
    /// until the CPU is stopped.
    pub fn park(&self, running: &AtomicBool) {
        let mut s = self.state.lock();
        s.parked = true;
        self.changed.notify_all();
        while s.holds != 0 && running.load(Ordering::Relaxed) {
            self.changed.wait_for(&mut s, Duration::from_millis(1));
        }
        s.parked = false;
    }

    /// The caller is blocked outside guest state, as a parked CPU is.
    fn mark_parked(&self, on: bool) {
        let mut s = self.state.lock();
        s.parked = on;
        self.changed.notify_all();
    }
}

/// Every CPU's slot: how one thread gets the machine to itself.
#[derive(Default)]
pub struct World {
    slots: OnceLock<Vec<Arc<ParkSlot>>>,
    leader: Mutex<()>,
}

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    /// The machine's CPUs, once they exist.
    pub fn set_slots(&self, slots: Vec<Arc<ParkSlot>>) {
        let _ = self.slots.set(slots);
    }

    /// Run `f` with every CPU but the caller's parked. Called with no executor
    /// or device lock another CPU could be waiting on. Two CPUs asking at once
    /// take turns: the one waiting its turn counts as parked for the other.
    pub fn stop_others<R>(&self, f: impl FnOnce() -> R) -> R {
        let me = on_cpu_thread();
        let mine = me.and_then(|m| self.slots().get(m));
        if let Some(s) = mine {
            s.mark_parked(true);
        }
        let turn = self.leader.lock();
        if let Some(s) = mine {
            s.mark_parked(false);
        }
        let _hold = self.hold_others(turn);
        f()
    }

    /// For a thread that must not block on another leader, because that leader
    /// may be waiting for this thread (a compile worker, which a CPU leading a
    /// flush joins): every CPU but the caller's parked until the result drops,
    /// or `None` if someone else leads.
    pub fn try_lead(&self) -> Option<WorldHold<'_>> {
        self.leader.try_lock().map(|turn| self.hold_others(turn))
    }

    fn slots(&self) -> &[Arc<ParkSlot>] {
        self.slots.get().map_or(&[], |s| s.as_slice())
    }

    fn hold_others<'a>(&'a self, turn: parking_lot::MutexGuard<'a, ()>) -> WorldHold<'a> {
        let hold = WorldHold { world: self, me: on_cpu_thread(), _turn: turn };
        hold.others().for_each(|s| s.hold(HOLD_WORLD));
        hold.others().for_each(|s| s.wait_parked());
        hold
    }
}

/// The other CPUs stay parked while this lives.
pub struct WorldHold<'a> {
    world: &'a World,
    me: Option<usize>,
    _turn: parking_lot::MutexGuard<'a, ()>,
}

impl WorldHold<'_> {
    fn others(&self) -> impl Iterator<Item = &Arc<ParkSlot>> {
        let me = self.me;
        self.world.slots().iter().enumerate().filter(move |(i, _)| Some(*i) != me).map(|(_, s)| s)
    }
}

impl Drop for WorldHold<'_> {
    fn drop(&mut self) {
        self.others().for_each(|s| s.release(HOLD_WORLD));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A CPU thread as far as the slot can tell: it parks when asked.
    fn cpu_thread(slot: Arc<ParkSlot>, word: Arc<AtomicU64>, running: Arc<AtomicBool>, id: u8, work: Arc<AtomicUsize>) -> std::thread::JoinHandle<()> {
        slot.set_active(true);
        std::thread::spawn(move || {
            set_current_cpu(id);
            while running.load(Ordering::Relaxed) {
                if word.load(Ordering::Relaxed) & HOST_PARK_BIT != 0 {
                    slot.park(&running);
                    continue;
                }
                work.fetch_add(1, Ordering::SeqCst);
                std::thread::yield_now();
            }
            slot.set_active(false);
        })
    }

    fn slot() -> (Arc<ParkSlot>, Arc<AtomicU64>) {
        let word = Arc::new(AtomicU64::new(0));
        (Arc::new(ParkSlot::new(Arc::as_ptr(&word))), word)
    }

    #[test]
    fn a_thread_that_is_not_a_cpu_reads_as_cpu_zero_but_says_so() {
        std::thread::spawn(|| {
            assert_eq!(on_cpu_thread(), None);
            assert_eq!(current_cpu(), 0);
            set_current_cpu(1);
            assert_eq!(on_cpu_thread(), Some(1));
            assert_eq!(current_cpu(), 1);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn the_cpu_runs_again_only_when_every_hold_is_gone() {
        let (s, word) = slot();
        s.hold(HOLD_HEART);
        s.hold(HOLD_WORLD);
        s.release(HOLD_WORLD);
        assert_ne!(word.load(Ordering::SeqCst) & HOST_PARK_BIT, 0);
        s.release(HOLD_HEART);
        assert_eq!(word.load(Ordering::SeqCst) & HOST_PARK_BIT, 0);
    }

    #[test]
    fn holding_leaves_the_interrupt_lines_alone() {
        let (s, word) = slot();
        word.store(0x400, Ordering::SeqCst);
        s.hold(HOLD_HEART);
        s.release(HOLD_HEART);
        assert_eq!(word.load(Ordering::SeqCst), 0x400);
    }

    #[test]
    fn a_stopped_cpu_counts_as_parked() {
        let (s, _word) = slot();
        s.hold(HOLD_WORLD);
        s.wait_parked();
        assert!(s.is_parked());
    }

    #[test]
    fn stop_others_runs_with_the_other_cpu_still_and_then_lets_it_go() {
        let (s, word) = slot();
        let running = Arc::new(AtomicBool::new(true));
        let work = Arc::new(AtomicUsize::new(0));
        let t = cpu_thread(s.clone(), word, running.clone(), 1, work.clone());
        let world = World::new();
        world.set_slots(vec![s]);
        world.stop_others(|| {
            let before = work.load(Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(20));
            assert_eq!(work.load(Ordering::SeqCst), before, "held CPU kept working");
        });
        let before = work.load(Ordering::SeqCst);
        while work.load(Ordering::SeqCst) == before {
            std::thread::yield_now();
        }
        running.store(false, Ordering::SeqCst);
        t.join().unwrap();
    }

    #[test]
    fn try_lead_refuses_while_another_leads_and_works_after() {
        let world = World::new();
        world.set_slots(vec![]);
        let first = world.try_lead();
        assert!(first.is_some());
        assert!(world.try_lead().is_none());
        drop(first);
        assert!(world.try_lead().is_some());
    }

    /// Both CPUs want the machine at once, as two CPUs running out of JIT pages
    /// together would. Each waits for the other unless a CPU waiting its turn
    /// counts as parked.
    #[test]
    fn two_cpus_stopping_each_other_take_turns() {
        let (a, wa) = slot();
        let (b, wb) = slot();
        let world = Arc::new(World::new());
        world.set_slots(vec![a.clone(), b.clone()]);
        let turns = Arc::new(AtomicUsize::new(0));
        let spawn = |id: u8, s: Arc<ParkSlot>, w: Arc<AtomicU64>| {
            let (world, turns) = (world.clone(), turns.clone());
            s.set_active(true);
            std::thread::spawn(move || {
                set_current_cpu(id);
                let running = AtomicBool::new(true);
                for _ in 0..200 {
                    if w.load(Ordering::Relaxed) & HOST_PARK_BIT != 0 {
                        s.park(&running);
                    }
                    world.stop_others(|| turns.fetch_add(1, Ordering::SeqCst));
                }
                // Keep answering until the other is done with us.
                s.set_active(false);
            })
        };
        // Both words outlive both threads: a slot's word must outlive anyone who
        // can hold it, and a thread that finished first is still held by the other.
        let (t0, t1) = (spawn(0, a, wa.clone()), spawn(1, b, wb.clone()));
        t0.join().unwrap();
        t1.join().unwrap();
        drop((wa, wb));
        assert_eq!(turns.load(Ordering::SeqCst), 400);
    }
}
