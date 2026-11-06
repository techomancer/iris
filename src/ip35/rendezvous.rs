//! The IP35 node's boot rendezvous: where the PROM assumes its processors
//! run in lockstep, the model holds the one that arrived early.
//!
//! The IP35 PROM brings a node's processors up on the assumption that they
//! left reset on the same clock edge and have run continuously since. Four
//! of its early handshakes order one processor's stores before another's
//! reads by nothing more than both running at hardware speed. Here each
//! processor is a host thread that can fall far behind the others, so at
//! each of those points the processor that arrived early is held
//! ([`HOLD_RENDEZVOUS`]) until the ones it must follow catch up, or until
//! they have run [`HOLD_CYCLES`] without doing so. The four, in boot order:
//!
//! - Reset barrier. Every processor clears LB scratch 1 on its reset path,
//!   and the boot master then posts its release there, which the others
//!   poll for. A processor's second or later store there is withheld, and
//!   the processor held, until every other running processor has made its
//!   first; the withheld value lands when the hold ends, before the
//!   processor runs again.
//! - Pair liveness. Each processor of a pair stores a status value to its
//!   own real-time compare register and polls its partner's. The first of
//!   a pair to post is held until its partner posts.
//! - Presence publisher. The lowest running processor of each interface
//!   publishes its interface's processors in II scratch 0 under the hub
//!   lock. A publisher is held at its next release of the lock (a store
//!   leaving LB scratch 3 clear) until every other populated interface has
//!   published.
//! - Partner wait. The other processor of an interface is held at its next
//!   real-time counter read, once both of the pair have posted, until its
//!   publisher has released the lock after publishing.
//!
//! The bound counts the cycles the processors being waited for have
//! retired, not host time. A host thread the scheduler leaves off a core
//! for seconds retires nothing, so it cannot run a hold out: the processor
//! that arrived early waits however long the host takes, and the PROM
//! counts every processor. Host time only backs the bound as a deadlock
//! safety net ([`SAFETY_NS`]), logged when it fires. QEMU bounds the same
//! holds at 2 s of its virtual clock, which under multi-threaded TCG
//! follows the host clock, so a starved vCPU thread can run its holds out
//! there too.
//!
//! Each is an approximation with no hardware counterpart: a hold changes
//! when a processor next runs, never a value it reads. A held processor
//! stops at its next batch boundary rather than at once, so it may run a
//! few more instructions than QEMU's, which stops at the end of its
//! translation block. A publisher's or partner's hold that runs out ends
//! the join until the next reset; a liveness or reset-barrier hold that
//! runs out ends only itself, and the PROM reports the processor that never
//! came. Nothing holds with fewer than two processors running. QEMU hooks
//! the publisher's hold on LB scratch 3, as here: the PROM publishes in II
//! scratch 0 and takes scratch 3 as its console lock, and the hub lock being
//! scratch 3's read-set twin is inferred.
//!
//! Ported from hw/misc/sgi-bedrock.c in jamesbraid/qemu-sgi-origami.

use crate::cpu::mips_exec::CpuDevice;
use crate::cpu::smp::{on_cpu_thread, ParkSlot, HOLD_RENDEZVOUS};
use crate::dev::bedrock_lb::{BedrockLb, ScratchHooks};
use crate::dev::bedrock_pi::Population;
use crate::dev::heart::Clock;
use crate::devlog::LogModule;
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

/// How far the processors a hold waits for may run, in their retired
/// cycles, before the hold gives up: 2 s of a 1 GHz processor, the order of
/// QEMU's 2 s bound.
pub(crate) const HOLD_CYCLES: u64 = 2_000_000_000;
/// The longest any hold lasts in board nanoseconds whatever the processors
/// do: a deadlock safety net, far past any host stall.
pub(crate) const SAFETY_NS: u64 = 120_000_000_000;
/// How often the holds' bounds are checked, in host time.
const POLL: Duration = Duration::from_millis(50);
const SLOTS: usize = 4;
/// Processor slots per interface.
const PER_PI: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Scratch1,
    Liveness,
    Presence,
    Partner,
}

#[derive(Default)]
struct State {
    scratch1_seen: [bool; SLOTS],
    /// A processor's withheld scratch-1 store and its order among them.
    withheld: [Option<(u64, u64)>; SLOTS],
    seq: u64,
    scratch1_released: bool,
    joined: [bool; SLOTS],
    posted: [bool; SLOTS],
    presence_seen: [bool; 2],
    presence_released: [bool; 2],
    abandoned: bool,
    park: [Option<Hold>; SLOTS],
}

/// One held slot: what it waits at, when it started in board nanoseconds,
/// and how many cycles the processors it waits for had retired by then.
#[derive(Clone, Copy)]
struct Hold {
    kind: Kind,
    since: u64,
    progress: u64,
}

/// One processor the rendezvous may hold, and whether it runs.
pub(crate) struct Cpu {
    pub slot: Arc<ParkSlot>,
    /// Cycles this processor has retired; frozen while the host does not run it.
    pub progress: Box<dyn Fn() -> u64 + Send + Sync>,
    pub running: Box<dyn Fn() -> bool + Send + Sync>,
}

/// The node's rendezvous state. Slots are numbered `interface * 2 + slot`:
/// A, B on PI0, C, D on PI1.
pub(crate) struct Rendezvous {
    running: Population,
    clock: Clock,
    lb: Weak<BedrockLb>,
    cpus: OnceLock<Vec<Cpu>>,
    timers: OnceLock<Arc<TimerManager>>,
    timer: Mutex<Option<TimerId>>,
    state: Mutex<State>,
    /// Every populated interface has released its hub lock after publishing,
    /// or the join was abandoned, and nothing is held: the joins are over
    /// until the next reset. Checked before any lock, so the PROM's and the
    /// kernel's later register traffic costs one load.
    done: AtomicBool,
    weak: Weak<Self>,
}

impl Rendezvous {
    pub fn new(running: Population, clock: Clock, lb: &Arc<BedrockLb>) -> Arc<Self> {
        Arc::new_cyclic(|weak| Self {
            running,
            clock,
            lb: Arc::downgrade(lb),
            cpus: OnceLock::new(),
            timers: OnceLock::new(),
            timer: Mutex::new(None),
            state: Mutex::new(State::default()),
            done: AtomicBool::new(false),
            weak: weak.clone(),
        })
    }

    /// The processors, CPU 0 first, in the order the board's population
    /// numbers them. Once.
    pub fn set_cpus(&self, cpus: &[Arc<dyn CpuDevice>]) {
        self.set_processors(cpus.iter().map(|c| {
            let device: Weak<dyn CpuDevice> = Arc::downgrade(c);
            let cycles = c.cycles_ptr();
            Cpu {
                slot: c.park_slot(),
                progress: Box::new(move || cycles.retired()),
                running: Box::new(move || device.upgrade().is_some_and(|c| c.as_device().is_running())),
            }
        }).collect());
    }

    pub(crate) fn set_processors(&self, cpus: Vec<Cpu>) {
        assert!(self.cpus.set(cpus).is_ok(), "rendezvous processors set twice");
    }

    pub fn set_timer_manager(&self, timers: Arc<TimerManager>) {
        let _ = self.timers.set(timers);
    }

    fn cpu(&self, slot: usize) -> Option<&Cpu> {
        let cpu = self.running.cpu_in((slot / PER_PI) as u8, slot % PER_PI)?;
        self.cpus.get()?.get(cpu)
    }

    /// The slot the calling thread's processor sits in, if it is one here.
    fn current_slot(&self) -> Option<usize> {
        let (interface, slot) = self.running.slot_of(on_cpu_thread()?)?;
        Some(usize::from(interface) * PER_PI + slot)
    }

    /// Fitted, running and built; a processor this holds still counts.
    fn slot_running(&self, slot: usize) -> bool {
        self.cpu(slot).is_some_and(|c| (c.running)())
    }

    fn applicable(&self) -> bool {
        (0..SLOTS).filter(|&s| self.slot_running(s)).count() >= 2
    }

    fn interface_populated(&self, interface: usize) -> bool {
        (0..PER_PI).any(|p| self.slot_running(interface * PER_PI + p))
    }

    /// The interface's publisher: its lowest running processor.
    fn publisher(&self, interface: usize) -> Option<usize> {
        (0..PER_PI).map(|p| interface * PER_PI + p).find(|&s| self.slot_running(s))
    }

    fn other_pending(&self, s: &State, slot: usize) -> bool {
        let own = slot / PER_PI;
        !s.abandoned && (0..2).any(|i| i != own && self.interface_populated(i) && !s.presence_seen[i])
    }

    /// The running slots a hold at `kind` waits for.
    fn awaited(&self, s: &State, slot: usize, kind: Kind) -> Vec<usize> {
        let own = slot / PER_PI;
        (0..SLOTS)
            .filter(|&p| p != slot && self.slot_running(p))
            .filter(|&p| match kind {
                Kind::Scratch1 => !s.scratch1_seen[p],
                Kind::Liveness => p == slot ^ 1,
                Kind::Presence => p / PER_PI != own,
                Kind::Partner => self.publisher(own) == Some(p),
            })
            .collect()
    }

    /// Cycles the slots a hold waits for have retired between them.
    fn progress(&self, s: &State, slot: usize, kind: Kind) -> u64 {
        self.awaited(s, slot, kind)
            .into_iter()
            .filter_map(|p| self.cpu(p))
            .fold(0u64, |sum, c| sum.saturating_add((c.progress)()))
    }

    fn scratch1_ready(&self, s: &State) -> bool {
        (0..SLOTS).all(|p| !self.slot_running(p) || s.scratch1_seen[p])
    }

    fn ready(&self, s: &State, slot: usize, kind: Kind) -> bool {
        match kind {
            Kind::Presence => !self.other_pending(s, slot),
            Kind::Partner => s.abandoned || s.presence_released[slot / PER_PI],
            Kind::Liveness => s.posted[slot ^ 1] || !self.slot_running(slot ^ 1),
            Kind::Scratch1 => self.scratch1_ready(s),
        }
    }

    /// Land `slot`'s withheld scratch-1 store, if one waits.
    fn flush(&self, s: &mut State, slot: usize) {
        if let Some((value, _)) = s.withheld[slot].take() {
            if let Some(lb) = self.lb.upgrade() {
                lb.set_scratch1(value);
            }
            s.scratch1_released = true;
        }
    }

    /// An access to scratch 1 from outside the node's processors: every
    /// withheld store lands first, in the order its processor made it.
    fn external(&self, s: &mut State) {
        while let Some((oldest, _)) = (0..SLOTS).filter_map(|p| s.withheld[p].map(|(_, seq)| (p, seq))).min_by_key(|&(_, seq)| seq) {
            self.flush(s, oldest);
        }
    }

    /// The calling processor's slot, if it may be held now.
    fn park_slot(&self, s: &State) -> Option<usize> {
        let slot = self.current_slot()?;
        (self.applicable() && s.park[slot].is_none()).then_some(slot)
    }

    fn start(&self, s: &mut State, slot: usize, kind: Kind) {
        s.park[slot] = Some(Hold { kind, since: (self.clock)(), progress: self.progress(s, slot, kind) });
        dlog!(LogModule::Bedrock, "rendezvous: holding slot {} at the {kind:?} join", slot_name(slot));
        if let Some(cpu) = self.cpu(slot) {
            cpu.slot.hold(HOLD_RENDEZVOUS);
        }
        self.arm(s);
    }

    fn release(&self, s: &mut State, slot: usize, outcome: &str) {
        let kind = s.park[slot].take().map(|h| h.kind);
        // A withheld store lands before the processor runs again.
        self.flush(s, slot);
        dlog!(LogModule::Bedrock, "rendezvous: slot {} {kind:?} hold {outcome}", slot_name(slot));
        if let Some(cpu) = self.cpu(slot) {
            cpu.slot.release(HOLD_RENDEZVOUS);
        }
    }

    fn recheck(&self, s: &mut State) {
        for slot in 0..SLOTS {
            if let Some(Hold { kind, .. }) = s.park[slot] {
                if self.ready(s, slot, kind) {
                    self.release(s, slot, "satisfied");
                }
            }
        }
        self.arm(s);
        if self.finished(s) {
            self.done.store(true, Ordering::Relaxed);
            dlog!(LogModule::Bedrock, "rendezvous: the joins are over");
        }
    }

    /// Every populated interface released its hub lock after publishing, or
    /// the join was abandoned, with nothing held or withheld.
    fn finished(&self, s: &State) -> bool {
        let interfaces = (0..2).filter(|&i| self.interface_populated(i)).all(|i| s.abandoned || s.presence_released[i]);
        interfaces && s.park.iter().all(Option::is_none) && s.withheld.iter().all(Option::is_none)
    }

    /// Whether the joins are over; see [`Self::done`].
    fn over(&self) -> bool {
        self.done.load(Ordering::Relaxed)
    }

    /// While any slot is held, a timer checks the bounds every [`POLL`].
    fn arm(&self, s: &State) {
        let Some(timers) = self.timers.get() else { return };
        let mut timer = self.timer.lock();
        if let Some(id) = timer.take() {
            timers.remove(id);
        }
        if s.park.iter().all(Option::is_none) {
            return;
        }
        *timer = Some(timers.add_one_shot(POLL, self.weak.clone(), |weak| {
            if let Some(r) = weak.upgrade() {
                r.expire();
            }
            TimerReturn::Delete
        }));
    }

    /// Every hold past its bound ends: the processors it waits for have run
    /// [`HOLD_CYCLES`] without coming, or [`SAFETY_NS`] of host time has
    /// passed. A publisher's or partner's ends the join for this reset.
    pub(crate) fn expire(&self) {
        let now = (self.clock)();
        let mut s = self.state.lock();
        let mut abandon = false;
        for slot in 0..SLOTS {
            if let Some(hold) = s.park[slot] {
                let ran = self.progress(&s, slot, hold.kind).saturating_sub(hold.progress);
                let safety = now.saturating_sub(hold.since) >= SAFETY_NS;
                if ran >= HOLD_CYCLES || safety {
                    abandon |= matches!(hold.kind, Kind::Presence | Kind::Partner);
                    let why = if safety { "timed out at the safety net" } else { "timed out" };
                    dlog!(LogModule::Bedrock, "rendezvous: slot {} waited for {ran} cycles of {:?}", slot_name(slot), self.awaited(&s, slot, hold.kind).iter().map(|&p| slot_name(p)).collect::<Vec<_>>());
                    self.release(&mut s, slot, why);
                }
            }
        }
        if abandon {
            s.abandoned = true;
        }
        self.recheck(&mut s);
    }

    /// The PROM's write of II scratch 0, its interface's processors.
    pub fn presence_published(&self) {
        if self.over() {
            return;
        }
        let Some(slot) = self.current_slot() else { return };
        dlog!(LogModule::Bedrock, "rendezvous: slot {} published its interface's processors", slot_name(slot));
        let mut s = self.state.lock();
        s.presence_seen[slot / PER_PI] = true;
        self.recheck(&mut s);
    }

    /// A store to `slot` of `interface`'s real-time compare register.
    pub fn rt_compare_written(&self, interface: u8, slot: usize) {
        if self.over() {
            return;
        }
        let Some(cslot) = self.current_slot() else { return };
        let mut s = self.state.lock();
        // Only a processor's own register counts as its post, and only its
        // first store after the reset barrier: earlier ones are progress codes.
        if cslot != usize::from(interface) * PER_PI + slot || !s.joined[cslot] || s.posted[cslot] {
            return;
        }
        s.posted[cslot] = true;
        dlog!(LogModule::Bedrock, "rendezvous: slot {} posted its liveness", slot_name(cslot));
        self.recheck(&mut s);
        if self.park_slot(&s) == Some(cslot) && !self.ready(&s, cslot, Kind::Liveness) {
            self.start(&mut s, cslot, Kind::Liveness);
        }
    }

    /// A read of the real-time counter.
    pub fn rt_counter_read(&self) {
        if self.over() {
            return;
        }
        let mut s = self.state.lock();
        let Some(cslot) = self.park_slot(&s) else { return };
        let interface = cslot / PER_PI;
        let Some(publisher) = self.publisher(interface) else { return };
        if publisher == cslot || s.abandoned || s.presence_released[interface] || !s.posted[cslot] || !s.posted[publisher] {
            return;
        }
        self.start(&mut s, cslot, Kind::Partner);
    }

    /// Every hold released and every first store forgotten: a fresh start.
    pub fn reset(&self) {
        dlog!(LogModule::Bedrock, "rendezvous: reset");
        let mut s = self.state.lock();
        for slot in 0..SLOTS {
            if s.park[slot].take().is_some() {
                if let Some(cpu) = self.cpu(slot) {
                    cpu.slot.release(HOLD_RENDEZVOUS);
                }
            }
        }
        *s = State::default();
        self.done.store(false, Ordering::Relaxed);
        if let (Some(timers), Some(id)) = (self.timers.get(), self.timer.lock().take()) {
            timers.remove(id);
        }
    }

    /// Whether the joins are over, for tests.
    #[cfg(test)]
    pub(crate) fn is_over(&self) -> bool {
        self.over()
    }

    /// The slots held now, with their kinds, for tests.
    #[cfg(test)]
    pub(crate) fn held(&self) -> Vec<(usize, Kind)> {
        let s = self.state.lock();
        (0..SLOTS).filter_map(|slot| s.park[slot].map(|h| (slot, h.kind))).collect()
    }
}

impl ScratchHooks for Rendezvous {
    fn scratch1_write(&self, value: u64) {
        if self.over() {
            if let Some(lb) = self.lb.upgrade() {
                lb.set_scratch1(value);
            }
            return;
        }
        let mut s = self.state.lock();
        let Some(slot) = self.current_slot() else {
            // A test, a peer or a monitor: plain storage, after every
            // withheld local store.
            self.external(&mut s);
            if let Some(lb) = self.lb.upgrade() {
                lb.set_scratch1(value);
            }
            return;
        };
        self.flush(&mut s, slot);
        let store = |s: &mut State| {
            if let Some(lb) = self.lb.upgrade() {
                lb.set_scratch1(value);
            }
            s.scratch1_released = true;
        };
        if !s.scratch1_seen[slot] {
            if let Some(lb) = self.lb.upgrade() {
                lb.set_scratch1(value);
            }
            s.scratch1_seen[slot] = true;
            dlog!(LogModule::Bedrock, "rendezvous: slot {} made its first scratch 1 store", slot_name(slot));
            self.recheck(&mut s);
            return;
        }
        // A second store before the barrier is over is the master's release
        // of the others; every processor's join is its store after it.
        if s.scratch1_released {
            s.joined[slot] = true;
        }
        if self.park_slot(&s) != Some(slot) || self.scratch1_ready(&s) {
            store(&mut s);
            return;
        }
        s.seq += 1;
        s.withheld[slot] = Some((value, s.seq));
        self.start(&mut s, slot, Kind::Scratch1);
    }

    fn scratch1_read(&self) {
        if !self.over() && self.current_slot().is_none() {
            let mut s = self.state.lock();
            self.external(&mut s);
        }
    }

    fn scratch3_released(&self) {
        if self.over() {
            return;
        }
        let mut s = self.state.lock();
        // A publisher releasing the lock after its interface published: its
        // partner, if held, may now read the bits.
        if let Some(cslot) = self.current_slot() {
            let interface = cslot / PER_PI;
            if self.publisher(interface) == Some(cslot) && s.presence_seen[interface] && !s.presence_released[interface] {
                s.presence_released[interface] = true;
                dlog!(LogModule::Bedrock, "rendezvous: slot {} released the hub lock after publishing", slot_name(cslot));
                self.recheck(&mut s);
            }
        }
        // The publisher waits here, with the lock free, for the other
        // interface's publication.
        let Some(cslot) = self.park_slot(&s) else { return };
        if self.publisher(cslot / PER_PI) != Some(cslot) || !s.presence_seen[cslot / PER_PI] || !self.other_pending(&s, cslot) {
            return;
        }
        self.start(&mut s, cslot, Kind::Presence);
    }
}

fn slot_name(slot: usize) -> char {
    (b'A' + slot as u8) as char
}

#[cfg(test)]
mod tests;
