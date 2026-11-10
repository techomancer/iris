//! The holds on the IP35 PROM's pair liveness check and its presence
//! publisher, outside the chip.
//!
//! The PROM brings a node's processors up on the assumption that they left
//! reset on one clock edge and have run in step since. Two handshakes
//! depend on it beyond what Bedrock can order by itself.
//!
//! - Pair liveness. After its presence join on LB scratch 1, each processor
//!   of an interface with both slots fitted stores a status value to its own
//!   PI real-time compare register and reads its partner's until the
//!   partner's value appears, at most 200,000 reads (PROM 6.210 at
//!   `0x1fc69234`). If the reads run out, it clears the partner's CPU enable
//!   and prints `peer_pled failed: Other CPU on this node died.` The bound
//!   is the poller's own progress. On hardware the pair reach their posts
//!   microseconds apart; here each processor is a host thread whose
//!   progress over the seconds since the reset barrier drifts from its
//!   partner's, by more than the reads last once the poll runs compiled.
//!   So the first of a pair to post is held until its partner posts.
//! - Presence publisher. The lowest processor of each interface takes the
//!   hub lock (LB scratch 3, through its test-and-set twin), sets its
//!   interface's slot bits in II scratch 0 (bit 4 + n for slot n: A and B on
//!   PI0, C and D on PI1), releases the lock, waits a fixed 2000 units of
//!   the PI's real-time counter, which runs on host time, and then tests II
//!   scratch 0 for the other interface's bits: it waits for that interface
//!   only if its bits are already set, and otherwise goes on alone (the
//!   later polls have no timer and no count; read from PROM 6.210 and
//!   6.211). So a publisher whose partner interface is descheduled for that
//!   wait finds the other interface absent and the PROM loses its
//!   processors. A publisher that has published,
//!   with the hub lock free, while another populated interface has not, is
//!   held until that interface publishes.
//!
//! A held processor ([`HOLD_RENDEZVOUS`]) stops at its batch boundary. The
//! check runs on every processor's batch boundary and reads atomics Bedrock
//! exposes ([`Bedrock::rt_posted`], [`Bedrock::io_scratch0`],
//! [`Bedrock::hub_lock_free`]); nothing calls back from the chip. A hold is
//! bounded by the cycles the processors it waits for have retired
//! ([`HOLD_CYCLES`]), not host time, so a host thread the scheduler leaves
//! off a core cannot run it out; host time backs it only as a deadlock
//! safety net ([`SAFETY_NS`]). A liveness hold that runs out ends only
//! itself, and the PROM then finds the partner dead as it would a dead one;
//! a publisher's hold that runs out ends the join until the next reset.
//! While nothing is held and the processor has no hold to start, the check
//! takes no lock; once every populated interface has published, it is one
//! atomic load.
//!
//! These are approximations with no hardware counterpart: they change when
//! a processor next runs, never a value it reads. A held processor stops at
//! its next batch boundary, not at once. Nothing holds with fewer than two
//! processors, and the liveness hold needs both slots of an interface
//! fitted. QEMU holds the same processors, the first poster at its post and
//! the publisher at its release of the hub lock, and bounds both at 2 s of
//! its virtual clock.
//!
//! Ported from hw/misc/sgi-bedrock.c in jamesbraid/qemu-sgi-origami.

use crate::cpu::mips_core::CyclesPtr;
use crate::cpu::mips_exec::CpuDevice;
use crate::cpu::smp::{ParkSlot, HOLD_RENDEZVOUS};
use crate::dev::bedrock::{Bedrock, Population};
use crate::dev::heart::Clock;
use crate::devlog::LogModule;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

/// How far the processors a hold waits for may run, in their retired
/// cycles, before the hold gives up: 2 s of a 1 GHz processor, the order of
/// QEMU's 2 s bound.
pub(crate) const HOLD_CYCLES: u64 = 2_000_000_000;
/// The longest a hold lasts in board nanoseconds whatever the processors
/// do: a deadlock safety net, far past any host stall.
pub(crate) const SAFETY_NS: u64 = 120_000_000_000;

/// One processor the hold may stop: its park slot and its retired cycles.
pub(crate) struct Cpu {
    pub slot: Arc<ParkSlot>,
    pub cycles: Box<dyn Fn() -> u64 + Send + Sync>,
}

/// What a held processor waits for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Its pair partner's liveness post.
    Partner,
    /// The other interface's publication.
    Publication,
}

/// One held processor: what for, when the hold started, in board
/// nanoseconds, and how many cycles the processors it waits for had retired
/// by then.
#[derive(Clone, Copy)]
struct Held {
    kind: Kind,
    since: u64,
    progress: u64,
}

#[derive(Default)]
struct State {
    held: [Option<Held>; 4],
    /// A hold ran out: no more holds until the next reset.
    abandoned: bool,
}

pub(crate) struct PresenceHold {
    processors: Population,
    clock: Clock,
    bedrock: Arc<Bedrock>,
    cpus: OnceLock<Vec<Cpu>>,
    state: Mutex<State>,
    /// Every populated interface has published, or the join was abandoned,
    /// and nothing is held: the check is over until the next reset.
    done: AtomicBool,
    /// Whether any processor is held, so a boundary with nothing to release
    /// and no hold to start takes no lock.
    holding: AtomicBool,
    /// Each processor's pair partner: the other slot of its interface, if
    /// fitted.
    partners: [Option<usize>; 4],
    /// Processors, one bit per CPU number, whose liveness hold ran out: not
    /// held for their partner again until the next reset.
    gave_up: AtomicU64,
}

impl PresenceHold {
    pub fn new(processors: Population, clock: Clock, bedrock: &Arc<Bedrock>) -> Arc<Self> {
        Arc::new(Self {
            processors,
            clock,
            bedrock: bedrock.clone(),
            cpus: OnceLock::new(),
            state: Mutex::new(State::default()),
            done: AtomicBool::new(processors.count() < 2),
            holding: AtomicBool::new(false),
            partners: std::array::from_fn(|cpu| {
                let (interface, slot) = processors.slot_of(cpu)?;
                (0..processors.count()).find(|&other| processors.slot_of(other) == Some((interface, 1 - slot)))
            }),
            gave_up: AtomicU64::new(0),
        })
    }

    /// The processors, CPU 0 first, in the order the population numbers
    /// them. Once.
    pub fn set_cpus(&self, cpus: &[Arc<dyn CpuDevice>]) {
        self.set_processors(cpus.iter().map(|c| {
            let cycles: CyclesPtr = c.cycles_ptr();
            Cpu { slot: c.park_slot(), cycles: Box::new(move || cycles.retired()) }
        }).collect());
    }

    pub(crate) fn set_processors(&self, cpus: Vec<Cpu>) {
        assert!(self.cpus.set(cpus).is_ok(), "presence hold processors set twice");
    }

    fn interface_of(&self, cpu: usize) -> Option<u8> {
        self.processors.slot_of(cpu).map(|(interface, _)| interface)
    }

    fn populated(&self, interface: u8) -> bool {
        (0..2).any(|slot| self.processors.fitted(interface, slot))
    }

    /// Whether `interface` has published its processors in `presence`.
    fn published(presence: u64, interface: u8) -> bool {
        presence & (0x30 << (2 * interface)) != 0
    }

    /// The interface's publisher: its lowest fitted processor.
    fn publisher(&self, interface: u8) -> Option<usize> {
        (0..self.processors.count()).find(|&cpu| self.interface_of(cpu) == Some(interface))
    }

    /// The retired cycles of the processors on interfaces other than `own`.
    fn progress(&self, own: u8) -> u64 {
        let Some(cpus) = self.cpus.get() else { return 0 };
        cpus.iter().enumerate()
            .filter(|&(cpu, _)| self.interface_of(cpu) != Some(own))
            .fold(0u64, |sum, (_, c)| sum.saturating_add((c.cycles)()))
    }

    /// The retired cycles of processor `cpu`.
    fn cycles(&self, cpu: usize) -> u64 {
        self.cpus.get().and_then(|cpus| cpus.get(cpu)).map_or(0, |c| (c.cycles)())
    }

    /// What a hold of `cpu` started now waits for has retired so far.
    fn waited_progress(&self, cpu: usize, kind: Kind) -> u64 {
        match kind {
            Kind::Partner => self.partners[cpu].map_or(0, |partner| self.cycles(partner)),
            Kind::Publication => self.interface_of(cpu).map_or(0, |own| self.progress(own)),
        }
    }

    /// Whether `cpu` has made its liveness post and waits for its partner's:
    /// the partner has not posted and `cpu`'s hold has not run out since the
    /// last reset.
    fn awaits_partner(&self, posted: u64, cpu: usize) -> bool {
        let bit = |cpu: usize| 1u64 << cpu;
        self.partners[cpu].is_some_and(|partner| posted & bit(partner) == 0)
            && posted & bit(cpu) != 0
            && self.gave_up.load(Ordering::Relaxed) & bit(cpu) == 0
    }

    fn release(&self, s: &mut State, cpu: usize, outcome: &str) {
        if s.held[cpu].take().is_some() {
            dlog!(LogModule::Bedrock, "presence: CPU {cpu} hold {outcome}");
            if let Some(c) = self.cpus.get().and_then(|cpus| cpus.get(cpu)) {
                c.slot.release(HOLD_RENDEZVOUS);
            }
        }
    }

    /// CPU `cpu`'s batch boundary: end the holds that are satisfied or past
    /// their bound, and hold `cpu` if it has posted and waits for its
    /// partner's post, or is a publisher waiting for the other interface.
    pub fn at_batch(&self, cpu: usize) {
        if self.done.load(Ordering::Relaxed) {
            return;
        }
        let presence = self.bedrock.io_scratch0();
        let posted = self.bedrock.rt_posted();
        // Nothing held and nothing for this processor to start: no lock.
        if !self.holding.load(Ordering::Acquire) && !self.may_hold(presence, cpu) && !self.awaits_partner(posted, cpu) {
            let all = (0..2).filter(|&i| self.populated(i)).all(|i| Self::published(presence, i));
            if all && !self.done.swap(true, Ordering::Relaxed) {
                dlog!(LogModule::Bedrock, "presence: the join is over");
            }
            return;
        }
        let now = (self.clock)();
        let mut s = self.state.lock();
        for held in 0..s.held.len() {
            let Some(hold) = s.held[held] else { continue };
            let satisfied = match hold.kind {
                Kind::Partner => !self.awaits_partner(posted, held),
                Kind::Publication => self.interface_of(held).is_none_or(|own| Self::published(presence, 1 - own)),
            };
            let ran_out = self.waited_progress(held, hold.kind).saturating_sub(hold.progress) >= HOLD_CYCLES
                || now.saturating_sub(hold.since) >= SAFETY_NS;
            if satisfied {
                self.release(&mut s, held, "satisfied");
            } else if ran_out && hold.kind == Kind::Partner {
                self.gave_up.fetch_or(1 << held, Ordering::Relaxed);
                self.release(&mut s, held, "ran out; its partner is left to the PROM's check");
            } else if ran_out {
                s.abandoned = true;
                self.release(&mut s, held, "ran out; no more holds until a reset");
            }
        }
        let all = (0..2).filter(|&i| self.populated(i)).all(|i| Self::published(presence, i));
        if (all || s.abandoned) && s.held.iter().all(Option::is_none) {
            if !self.done.swap(true, Ordering::Relaxed) {
                dlog!(LogModule::Bedrock, "presence: the join is over");
            }
            return;
        }
        let kind = if s.held[cpu].is_some() {
            None
        } else if self.awaits_partner(posted, cpu) {
            Some(Kind::Partner)
        } else if !s.abandoned && self.may_hold(presence, cpu) {
            Some(Kind::Publication)
        } else {
            None
        };
        let Some(kind) = kind else {
            self.holding.store(s.held.iter().any(Option::is_some), Ordering::Release);
            return;
        };
        s.held[cpu] = Some(Held { kind, since: now, progress: self.waited_progress(cpu, kind) });
        self.holding.store(true, Ordering::Release);
        match kind {
            Kind::Partner => dlog!(LogModule::Bedrock, "presence: holding CPU {cpu} until its partner posts"),
            Kind::Publication => dlog!(LogModule::Bedrock, "presence: holding CPU {cpu} until the other interface publishes"),
        }
        if let Some(c) = self.cpus.get().and_then(|cpus| cpus.get(cpu)) {
            c.slot.hold(HOLD_RENDEZVOUS);
        }
    }

    /// Whether `cpu` is a publisher that should wait now: its interface has
    /// published and another populated one has not, with the hub lock free.
    fn may_hold(&self, presence: u64, cpu: usize) -> bool {
        let Some(own) = self.interface_of(cpu) else { return false };
        let other = 1 - own;
        self.publisher(own) == Some(cpu)
            && Self::published(presence, own)
            && self.populated(other)
            && !Self::published(presence, other)
            && self.bedrock.hub_lock_free()
    }

    /// Every hold released and the join started over.
    pub fn reset(&self) {
        let mut s = self.state.lock();
        for cpu in 0..s.held.len() {
            self.release(&mut s, cpu, "ended by a reset");
        }
        *s = State::default();
        self.holding.store(false, Ordering::Release);
        self.gave_up.store(0, Ordering::Relaxed);
        self.done.store(self.processors.count() < 2, Ordering::Relaxed);
    }

    /// The processors held now, for tests.
    #[cfg(test)]
    pub(crate) fn held(&self) -> Vec<usize> {
        let s = self.state.lock();
        (0..s.held.len()).filter(|&cpu| s.held[cpu].is_some()).collect()
    }

    /// Whether the check is over, for tests.
    #[cfg(test)]
    pub(crate) fn is_done(&self) -> bool {
        self.done.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests;
