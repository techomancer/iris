//! Bedrock's local block (LB) provides the node and region registers, atomic
//! scratch aliases, error and vector status, and the MicroLAN master used by
//! board identity parts. The guest polls the MicroLAN completion bit after
//! issuing a slot.
//!
//! # What is approximated here
//!
//! The slot is decoded immediately and completes after one microsecond of
//! machine time; no 1-wire part answers, so a reset pulse and a short write
//! slot read back as presence-free and as a one. Permission registers retain
//! guest writes, but the register path carries no requester identity except
//! the calling CPU thread's, which only the scratch-1 rule uses, so it does
//! not enforce those permissions.
//!
//! The plain register words sit in a sparse map keyed by offset, since the
//! 0xc040-byte block holds only a few dozen of them.
//!
//! Scratch 1 orders the PROM's reset barrier ([`Barrier`]): a processor's
//! first store after the block's reset that arrives once another
//! processor's later store has landed does not land. That is a model rule
//! with no hardware counterpart, standing in for processors that leave
//! reset on one clock edge (inferred: the processors share the node's
//! reset; no timing specification is retained).
//!
//! Ported from hw/misc/sgi-bedrock-lb.c in jamesbraid/qemu-sgi-origami.

use super::pi::Levels;
use super::reg::IIO_PROTECT_OVRRD_KEY;
use super::{Effect, NodeReset};
use crate::dev::w1;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

/// Length of the block-relative register aperture.
pub(super) const SIZE: u64 = 0xc040;

pub mod reg {
    pub const NODE_REGION: u64 = 0x0;
    pub const PIO_PERMISSION: u64 = 0x4000;
    pub const PIO_PERMISSION_KEY: u64 = 0x4008;
    pub const INBOUND_PERMISSION: u64 = 0x4010;
    pub const SOFT_RESET: u64 = 0x4018;
    pub const MICRO_LAN: u64 = 0x4030;
    pub const ERROR_STATUS: u64 = 0x4040;
    pub const ERROR_CLEAR: u64 = 0x4048;
    pub const PIO_REGION_MASK: u64 = 0x4028;
    pub const NODE_REGION_CONFIG: u64 = 0x4050;
    pub const NODE_REGION_MASK: u64 = 0x4058;
    pub const PIO_CONFIG: u64 = 0x4100;
    pub const INBOUND_CONFIG: u64 = 0x4200;
    pub const INBOUND_MASK: u64 = 0x4208;
    pub const VECTOR_REQUEST: u64 = 0xc000;
    pub const VECTOR_STATUS: u64 = 0xc020;
    pub const VECTOR_STATUS_ALIAS: u64 = 0xc038;
    pub const PIO_WINDOW: u64 = 0x8000;
    pub const PIO_WINDOW_END: u64 = 0x8020;
    pub const ATOMIC_SCRATCH: u64 = 0x8040;
    pub const ATOMIC_SCRATCH_END: u64 = 0x8050;
    pub const ATOMIC_READ0: u64 = 0x8058;
    pub const ATOMIC_READ1: u64 = 0x8060;
    pub const REGISTER_4020: u64 = 0x4020;
    pub const REGISTER_4060: u64 = 0x4060;
    pub const REGISTER_C008: u64 = 0xc008;
    pub const REGISTER_C010: u64 = 0xc010;
    pub const REGISTER_C028: u64 = 0xc028;
    pub const REGISTER_C030: u64 = 0xc030;
}

const ID: u64 = (0x36 << 1) | (0x3012 << 12) | (2 << 28);
/// Scratch 1 in the plain scratch window.
const SCRATCH1: u64 = reg::PIO_WINDOW + 8;
const VECTOR_MASK: u64 = 3 | (0x1fffff << 3) | (0xff << 32) | (0x7ff << 40);
struct Regs {
    words: BTreeMap<u64, u64>,
    scratch: [u64; 5],
    node: u8,
    region: u8,
    deadline: Option<u64>,
    sample: bool,
    barrier: Barrier,
}
impl Default for Regs {
    fn default() -> Self {
        let mut words = BTreeMap::new();
        words.insert(reg::PIO_PERMISSION, !0);
        words.insert(reg::INBOUND_PERMISSION, !0);
        Self { words, scratch: [0; 5], node: 0, region: 0, deadline: None, sample: true, barrier: Barrier::default() }
    }
}

/// The order of the processors' stores to scratch 1 since the block's last
/// reset, for the PROM's reset barrier.
///
/// Every processor of a node clears scratch 1 on its reset path, then the
/// boot master stores its release there and the others poll for it. On
/// hardware every processor leaves reset on the same clock edge, so every
/// clear lands before the release (inferred: the processors share the
/// node's reset; no timing specification is retained). Here each processor is a host thread
/// that can start or stall late, and a late clear would erase the release:
/// the PROM then counts that processor as absent. So a processor's first
/// store of the epoch that arrives after another processor's second or
/// later store (the release) has landed does not land: it is ordered below
/// the release, as it would have been on hardware. Until that processor
/// stores again it reads the release, whatever has landed since: on
/// hardware it would have polled the release long before the others'
/// next stores (the four-processor PROM's come millions of instructions
/// later).
///
/// Model rule, approximated: no hardware rule drops a store or answers one
/// processor differently from another. It is a function of the store
/// sequence alone (who stored, in what order), never of values or host
/// time. An access no processor makes (a monitor, a test, a peer node)
/// is plain storage and changes nothing here.
///
/// Limitations: an L1 soft reset restarts the processors without an LB
/// reset, so a second PROM run in the same epoch gets no ordering. A
/// processor that cleared on time but stalls until after the others' joins
/// reads those, not the release. A peer node's store is plain storage, and
/// a late local clear after it is dropped, keeping the peer's value. The
/// requester is the calling thread's CPU number (`smp::on_cpu_thread`),
/// not a field of the access.
#[derive(Default)]
struct Barrier {
    /// Processors, by CPU number, that have stored this epoch.
    stored: u64,
    /// The first processor's second or later store this epoch: the release.
    release: Option<u64>,
    /// Processors whose only store this epoch was ordered below the release.
    behind: u64,
}

/// What [`Barrier::store`] makes of one store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scratch1Store {
    /// It lands.
    Lands,
    /// It lands, and is its processor's first after the release: the PROM's
    /// presence join. Holds the processor's bit.
    Joins(u64),
    /// A first store after the release: it does not land, and its processor
    /// reads the release until it stores again.
    BelowRelease,
}

fn cpu_bit(cpu: Option<usize>) -> Option<u64> {
    cpu.and_then(|c| 1u64.checked_shl(c as u32))
}

impl Barrier {
    /// Order a store of `value` from `cpu`, or from no processor.
    fn store(&mut self, cpu: Option<usize>, value: u64) -> Scratch1Store {
        let Some(bit) = cpu_bit(cpu) else { return Scratch1Store::Lands };
        if self.stored & bit == 0 {
            self.stored |= bit;
            if self.release.is_none() {
                return Scratch1Store::Lands;
            }
            self.behind |= bit;
            return Scratch1Store::BelowRelease;
        }
        self.behind &= !bit;
        if self.release.is_some() {
            return Scratch1Store::Joins(bit);
        }
        self.release = Some(value);
        Scratch1Store::Lands
    }

    /// What a read from `cpu` sees, if not the register: the release, for a
    /// processor still behind it.
    fn read(&self, cpu: Option<usize>) -> Option<u64> {
        cpu_bit(cpu).filter(|bit| self.behind & bit != 0).and(self.release)
    }
}
impl Regs {
    fn error_level(&self) -> bool {
        self.words.get(&reg::ERROR_STATUS).copied().unwrap_or(0) != 0
    }
}

pub(super) struct Lb {
    regs: Mutex<Regs>,
    /// Processors, by CPU number, whose presence join on scratch 1 has
    /// landed since the block's last reset ([`Scratch1Store::Joins`]).
    /// Written under the block's lock, read without it.
    joined: AtomicU64,
    /// Scratch 3's bit 0, which the IP35 PROM takes as its hub lock through
    /// the test-and-set twin at 0x8058: kept with the register under the
    /// block's lock, read without it.
    scratch3: AtomicU64,
    /// Processors, by CPU number, that have stored to their own PI
    /// real-time compare register since their presence join: the IP35
    /// PROM's pair liveness post ([`Lb::note_rt_compare`]). Cleared with
    /// the join mask.
    posted: AtomicU64,
}

impl Lb {
    pub(super) fn new() -> Self {
        Self { regs: Mutex::new(Regs::default()), joined: AtomicU64::new(0), scratch3: AtomicU64::new(0), posted: AtomicU64::new(0) }
    }
    /// Processor `cpu` stored to its own real-time compare register. The
    /// first such store after its presence join is the PROM's liveness post
    /// to its pair partner, whatever the value: the PROM's earlier stores
    /// there are progress codes. A store before the join changes nothing.
    pub(super) fn note_rt_compare(&self, cpu: usize) {
        if let Some(bit) = cpu_bit(Some(cpu)).filter(|bit| self.joined.load(Ordering::Acquire) & bit != 0) {
            self.posted.fetch_or(bit, Ordering::Release);
        }
    }
    /// The processors, one bit per CPU number, that have posted since their
    /// presence join. One atomic load.
    pub(super) fn rt_posted(&self) -> u64 {
        self.posted.load(Ordering::Acquire)
    }
    /// Whether scratch 3's bit 0, the PROM's hub lock, is clear. One atomic
    /// load.
    pub(super) fn hub_lock_free(&self) -> bool {
        self.scratch3.load(Ordering::Acquire) & 1 == 0
    }
    /// Whether processor `cpu`'s presence join on scratch 1 has landed since
    /// the block's last reset. One atomic load.
    pub(super) fn scratch1_joined(&self, cpu: usize) -> bool {
        cpu_bit(Some(cpu)).is_some_and(|bit| self.joined.load(Ordering::Acquire) & bit != 0)
    }
    pub(super) fn locked(&self) -> bool {
        self.regs.is_locked()
    }
    fn complete(r: &mut Regs, now: u64) {
        if r.deadline.is_some_and(|d| now >= d) {
            *r.words.entry(reg::MICRO_LAN).or_default() |= 2 | u64::from(r.sample);
            r.deadline = None;
        }
    }
    fn mask(a: u64) -> Option<u64> {
        Some(match a {
            reg::PIO_PERMISSION | reg::INBOUND_PERMISSION | reg::REGISTER_4020 | reg::REGISTER_4060 | reg::REGISTER_C008 | reg::REGISTER_C010 | reg::REGISTER_C028 | reg::REGISTER_C030 => !0,
            reg::PIO_REGION_MASK => 0x80ff80ff80ff80ff,
            reg::ERROR_STATUS => 0x7ff,
            reg::NODE_REGION_CONFIG => 0x7f | (0x7ff << 8) | (0x7ff << 20) | (0xf << 32) | (1 << 40) | (1 << 44),
            reg::NODE_REGION_MASK => (1 << 38) - 1,
            reg::PIO_CONFIG => 0x3f,
            reg::INBOUND_CONFIG => 1 | (0x3ff << 4) | (0x3ff << 16) | (1 << 28) | (1 << 32),
            reg::INBOUND_MASK => {
                0x1f | (0xfff << 8) | (1 << 20) | (0x3ff << 24) | (0x3ff << 36) | (0x3ff << 48)
            }
            reg::VECTOR_REQUEST => VECTOR_MASK,
            _ => return None,
        })
    }
    /// The block's reset, under its lock: the scratch words 1 and 2 and the
    /// error status survive it. The node number returns to 0, and the reset
    /// barrier starts a new epoch.
    fn reset(&self, r: &mut Regs, node: &AtomicU8) {
        let survivors = [r.scratch[1], r.scratch[2], r.words.get(&reg::ERROR_STATUS).copied().unwrap_or(0)];
        *r = Regs::default();
        self.joined.store(0, Ordering::Release);
        self.posted.store(0, Ordering::Release);
        self.scratch3.store(0, Ordering::Release);
        r.scratch[1] = survivors[0];
        r.scratch[2] = survivors[1];
        r.words.insert(reg::ERROR_STATUS, survivors[2]);
        node.store(0, Ordering::Release);
    }
    pub(super) fn power_on(&self, node: &AtomicU8) {
        self.reset(&mut self.regs.lock(), node);
    }
    /// A doubleword read at `a` by processor `requester`, or by none.
    pub(super) fn read(&self, a: u64, now: u64, requester: Option<usize>) -> u64 {
        let mut r = self.regs.lock();
        Self::complete(&mut r, now);
        match a {
            reg::NODE_REGION => ID | ((r.node as u64) << 32) | ((r.region as u64) << 46),
            SCRATCH1 => r.barrier.read(requester).unwrap_or(r.scratch[1]),
            reg::PIO_WINDOW..=reg::PIO_WINDOW_END => r.scratch[(a - reg::PIO_WINDOW) as usize / 8],
            reg::ATOMIC_READ0 | reg::ATOMIC_READ1 => {
                let n = 3 + (a - reg::ATOMIC_READ0) as usize / 8;
                let v = r.scratch[n];
                r.scratch[n] = 1;
                self.scratch3.store(r.scratch[3], Ordering::Release);
                v
            }
            reg::VECTOR_STATUS_ALIAS => r.words.remove(&reg::VECTOR_STATUS).unwrap_or(0),
            reg::VECTOR_STATUS | reg::MICRO_LAN => r.words.get(&a).copied().unwrap_or(0),
            _ if Self::mask(a).is_some() => r.words.get(&a).copied().unwrap_or(0),
            _ => {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB unsupported read {a:#x}");
                0
            }
        }
    }
    /// A doubleword store at `a` by processor `requester`, or by none. The
    /// error level reaches `levels` and the node number `node` under the
    /// block's lock, so concurrent stores publish in register order. A soft
    /// reset resets the block and asks for the node's.
    pub(super) fn write(&self, a: u64, v: u64, now: u64, requester: Option<usize>, levels: &Levels, node: &AtomicU8) -> Effect {
        let mut r = self.regs.lock();
        Self::complete(&mut r, now);
        match a {
            reg::NODE_REGION => {
                let region = ((v >> 46) & 3) as u8;
                if region == 3 {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB invalid region size");
                } else {
                    r.node = (v >> 32) as u8;
                    r.region = region;
                    node.store(r.node, Ordering::Release);
                }
            }
            SCRATCH1 => {
                let order = r.barrier.store(requester, v);
                crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB scratch 1 store {v:#x} from CPU {requester:?}: {order:?}");
                match order {
                    Scratch1Store::BelowRelease => {}
                    Scratch1Store::Lands => r.scratch[1] = v,
                    Scratch1Store::Joins(bit) => {
                        r.scratch[1] = v;
                        self.joined.fetch_or(bit, Ordering::Release);
                    }
                }
            }
            reg::PIO_WINDOW..=reg::PIO_WINDOW_END => {
                let n = (a - reg::PIO_WINDOW) as usize / 8;
                r.scratch[n] = if n < 3 { v } else { v & 1 };
            }
            reg::ATOMIC_SCRATCH..=reg::ATOMIC_SCRATCH_END => {
                let n = (a - reg::ATOMIC_SCRATCH) as usize / 8;
                if r.scratch[n] == 0 {
                    r.scratch[n] = v;
                }
            }
            reg::PIO_PERMISSION_KEY => {
                if v == IIO_PROTECT_OVRRD_KEY {
                    *r.words.entry(reg::PIO_PERMISSION).or_default() |= 1;
                }
            }
            reg::SOFT_RESET => {
                if v & 1 != 0 {
                    self.reset(&mut r, node);
                    return Effect::Reset(NodeReset::LocalBlock);
                }
            }
            reg::MICRO_LAN => {
                if r.deadline.is_some() {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB MicroLAN write while slot pending discarded");
                } else {
                    r.words.insert(a, v & ((1 << 34) - 4));
                    let (pulse, _) = w1::slot::fields(v);
                    r.sample = pulse >= w1::slot::RESET_PULSE || pulse < w1::slot::ONE_PULSE;
                    r.deadline = Some(now.saturating_add(1000));
                }
            }
            reg::ERROR_STATUS => {
                *r.words.entry(a).or_default() |= v & 0x7ff;
            }
            reg::ERROR_CLEAR => {
                *r.words.entry(reg::ERROR_STATUS).or_default() &= !(v & 0x7ff);
            }
            reg::VECTOR_REQUEST => {
                r.words.insert(a, v & VECTOR_MASK);
                if r.words.get(&reg::VECTOR_STATUS).copied().unwrap_or(0) & (1 << 63) != 0 {
                    *r.words.entry(reg::VECTOR_STATUS).or_default() |= 1 << 62;
                } else {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB vector launch reaches no peer; no reply latched");
                }
            }
            reg::VECTOR_STATUS | reg::VECTOR_STATUS_ALIAS => {
                if v == 0 {
                    r.words.insert(reg::VECTOR_STATUS, 0);
                } else {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB nonzero vector status write ignored");
                }
            }
            _ => {
                if let Some(mask) = Self::mask(a) {
                    r.words.insert(a, v & mask);
                    if (a == reg::PIO_PERMISSION || a == reg::INBOUND_PERMISSION) && v != !0 {
                        crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB permission enforcement requires requester context");
                    }
                } else {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB unsupported write {a:#x} = {v:#x}");
                }
            }
        }
        self.scratch3.store(r.scratch[3], Ordering::Release);
        if levels.set(Levels::LB_ERROR, r.error_level()) { Effect::Reflect } else { Effect::None }
    }
}

#[cfg(test)]
mod tests {
    // Cases follow tests/qtest/sn1-bedrock-lb-test.c in
    // jamesbraid/qemu-sgi-origami: identity-permissions, scratch-reset and
    // microlan-slot.
    use super::super::reg::{pi, LB_BASE, PI1_BASE};
    use super::super::Rig;
    use super::*;
    use crate::traits::{BusDevice, BUS_ERR};

    fn error_reflected(rig: &Rig) -> [bool; 2] {
        [0, PI1_BASE].map(|base| rig.r(base + pi::PENDING_STATUS_B) & (1 << 55) != 0)
    }

    /// The guest configures node permissions and expects selected scratch words to survive reset.
    #[test]
    fn permission_writes_latch_errors_and_reset_retains_scratch() {
        {
            let rig = Rig::new();
            let node = || rig.chip.node.load(Ordering::SeqCst);
            rig.w(LB_BASE + reg::NODE_REGION, (0x5a << 32) | (2 << 46));
            assert_eq!(node(), 0x5a);
            rig.w(LB_BASE + reg::NODE_REGION, 3 << 46);
            assert_eq!(node(), 0x5a);
            rig.w(LB_BASE + reg::PIO_PERMISSION, 0);
            rig.w(LB_BASE + reg::PIO_PERMISSION_KEY, 0);
            assert_eq!(rig.r(LB_BASE + reg::PIO_PERMISSION), 0);
            rig.w(LB_BASE + reg::PIO_PERMISSION_KEY, IIO_PROTECT_OVRRD_KEY);
            assert_eq!(rig.r(LB_BASE + reg::PIO_PERMISSION), 1);
            rig.w(LB_BASE + reg::PIO_WINDOW + 8, 42);
            rig.w(LB_BASE + reg::ERROR_STATUS, !0);
            assert_eq!(rig.r(LB_BASE + reg::ERROR_STATUS), 0x7ff);
            assert_eq!(error_reflected(&rig), [true; 2], "bank 1 bit 55 on both interfaces");
            rig.chip.lb.power_on(&rig.chip.node);
            assert_eq!(rig.r(LB_BASE + reg::PIO_WINDOW + 8), 42);
            assert_eq!(error_reflected(&rig), [true; 2], "the error status survives reset");
            rig.w(LB_BASE + reg::ERROR_CLEAR, 0x7ff);
            assert_eq!(error_reflected(&rig), [false; 2]);
            assert_eq!(node(), 0);
        }
        {
            let rig = Rig::new();
            rig.w(LB_BASE + reg::ATOMIC_SCRATCH, 42);
            rig.w(LB_BASE + reg::ATOMIC_SCRATCH, 43);
            assert_eq!(rig.r(LB_BASE + reg::PIO_WINDOW), 42);
            rig.w(LB_BASE + reg::REGISTER_C030, 0xdead);
            rig.w(LB_BASE + reg::VECTOR_REQUEST, !0);
            assert_eq!(rig.r(LB_BASE + reg::VECTOR_STATUS), 0);
            assert_eq!(rig.r(LB_BASE + reg::REGISTER_C030), 0xdead);
            assert_eq!(rig.regs.read32(LB_BASE + reg::PIO_WINDOW).status, BUS_ERR);
            assert_eq!(rig.regs.read64(LB_BASE + 1).status, BUS_ERR);
        }
    }
    /// The guest polls MicroLAN completion and cannot replace a slot that is still pending.
    #[test]
    fn microlan_slots_finish_after_one_microsecond_and_reject_overlap() {
        let rig = Rig::new();
        let word = (0x66 << 27) | (60 << 10) | (0x41 << 2) | 3;
        rig.w(LB_BASE + reg::MICRO_LAN, word);
        assert_eq!(rig.r(LB_BASE + reg::MICRO_LAN), word & !3);
        rig.w(LB_BASE + reg::MICRO_LAN, 8 << 10);
        assert_eq!(rig.r(LB_BASE + reg::MICRO_LAN), word & !3);
        rig.set_time(1000);
        assert_eq!(rig.r(LB_BASE + reg::MICRO_LAN), (word & !3) | 2);
        rig.w(LB_BASE + reg::MICRO_LAN, 8 << 10);
        rig.set_time(2000);
        assert_eq!(rig.r(LB_BASE + reg::MICRO_LAN), (8 << 10) | 3);
        rig.w(LB_BASE + reg::MICRO_LAN, 60 << 10);
        rig.chip.lb.power_on(&rig.chip.node);
        rig.set_time(3000);
        assert_eq!(rig.r(LB_BASE + reg::MICRO_LAN), 0);
    }
    const RELEASE: u64 = 0x07f8_0000;
    /// The PROM's flags word that follows the joins.
    const FLAGS: u64 = 0x6000_0000_0000_0000;
    const SCRATCH1_AT: u64 = LB_BASE + SCRATCH1;

    /// Run `f` on a thread that is processor `cpu`.
    fn as_cpu<T: Send + 'static>(cpu: u8, f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::spawn(move || {
            crate::cpu::smp::set_current_cpu(cpu);
            f()
        })
        .join()
        .unwrap()
    }
    fn store_as(rig: &Rig, cpu: u8, value: u64) {
        let regs = rig.regs.clone();
        assert_eq!(as_cpu(cpu, move || regs.write64(SCRATCH1_AT, value)), crate::traits::BUS_OK);
    }
    fn read_as(rig: &Rig, cpu: u8) -> u64 {
        let regs = rig.regs.clone();
        as_cpu(cpu, move || regs.read64(SCRATCH1_AT).data)
    }
    /// Scratch 1 as a reader that is no processor sees it.
    fn scratch1(rig: &Rig) -> u64 {
        rig.r(SCRATCH1_AT)
    }
    fn joined(rig: &Rig, cpu: usize) -> bool {
        rig.chip.scratch1_joined(cpu)
    }

    /// A processor's reset-path clear that arrives after the master's
    /// release does not erase it, and that processor's next store joins.
    #[test]
    fn a_late_clear_lands_below_the_release() {
        let rig = Rig::new();
        store_as(&rig, 0, 0);
        store_as(&rig, 0, RELEASE);
        store_as(&rig, 1, 0);
        assert_eq!(scratch1(&rig), RELEASE, "the late clear did not land");
        assert_eq!(read_as(&rig, 1), RELEASE);
        assert!(!joined(&rig, 1));
        store_as(&rig, 1, 0x101);
        assert_eq!(scratch1(&rig), 0x101);
        assert!(joined(&rig, 1) && !joined(&rig, 0));
    }

    /// A processor whose clear came after the release reads the release
    /// until it stores again, even once another processor's next store has
    /// replaced it; everyone else reads the register.
    #[test]
    fn a_late_processor_reads_the_release_until_it_stores_again() {
        let rig = Rig::new();
        store_as(&rig, 0, 0);
        store_as(&rig, 1, 0);
        store_as(&rig, 0, RELEASE);
        store_as(&rig, 2, 0);
        store_as(&rig, 1, 0);
        assert_eq!(read_as(&rig, 2), RELEASE, "CPU 2 is still behind the release");
        assert_eq!((read_as(&rig, 1), read_as(&rig, 0), scratch1(&rig)), (0, 0, 0));
        store_as(&rig, 2, 0x102);
        assert_eq!(read_as(&rig, 2), 0x102);
        assert!(joined(&rig, 2));
    }

    /// Clears that come before the release land, the release lands over
    /// them, and the release itself is no join.
    #[test]
    fn a_release_after_every_clear_lands() {
        let rig = Rig::new();
        for cpu in 0..4 {
            store_as(&rig, cpu, 0x55);
            assert_eq!(scratch1(&rig), 0x55);
        }
        store_as(&rig, 0, RELEASE);
        assert_eq!(scratch1(&rig), RELEASE);
        assert!((0..4).all(|c| !joined(&rig, c)));
        for cpu in 0..4u8 {
            store_as(&rig, cpu, 0x100 + u64::from(cpu));
            assert_eq!(scratch1(&rig), 0x100 + u64::from(cpu));
            assert!(joined(&rig, usize::from(cpu)));
        }
    }

    /// The block's reset, at power-on or from its soft-reset bit, starts a
    /// new epoch: every processor's next store is a first store again, and
    /// the joins are forgotten.
    #[test]
    fn each_block_reset_starts_a_new_barrier() {
        let rig = Rig::new();
        let epoch = |late: u8| {
            store_as(&rig, 1 - late, 0);
            assert_eq!(scratch1(&rig), 0, "a first store");
            store_as(&rig, 1 - late, RELEASE);
            store_as(&rig, late, 0);
            assert_eq!(scratch1(&rig), RELEASE);
            store_as(&rig, late, 0x100);
            assert!(joined(&rig, usize::from(late)));
        };
        epoch(1);
        rig.chip.lb.power_on(&rig.chip.node);
        assert!(!joined(&rig, 1));
        assert_eq!(scratch1(&rig), 0x100, "scratch 1 survives the reset");
        epoch(0);
        rig.w(LB_BASE + reg::SOFT_RESET, 1);
        assert!(!joined(&rig, 0));
        epoch(1);
    }

    /// Stores no processor makes (a monitor, a test, a peer node's vector
    /// write) always land and move nothing.
    #[test]
    fn a_store_from_no_processor_always_lands() {
        let rig = Rig::new();
        for value in [RELEASE, 0, RELEASE, 0] {
            rig.w(SCRATCH1_AT, value);
            assert_eq!(scratch1(&rig), value);
        }
        store_as(&rig, 1, 0);
        assert_eq!(scratch1(&rig), 0, "no release has landed from a processor");
    }

    /// (CPU, value stored, scratch 1 after) for the four-processor PROM's
    /// stores to scratch 1 up to its flags word, in the order a Tezro boot
    /// with every processor on time logged them reaching the LB: four
    /// clears, CPU 0's release, each processor's join (a store of 0), then
    /// each processor's flags word.
    const ON_TIME_PROM_SEQUENCE: &[(u8, u64, u64)] = &[
        (0, 0, 0), (1, 0, 0), (3, 0, 0), (2, 0, 0),
        (0, RELEASE, RELEASE),
        (1, 0, 0), (0, 0, 0), (2, 0, 0), (3, 0, 0),
        (3, FLAGS, FLAGS), (2, FLAGS, FLAGS), (1, FLAGS, FLAGS), (0, FLAGS, FLAGS),
    ];

    /// With every processor on time the rule changes nothing.
    #[test]
    fn the_on_time_prom_sequence_is_plain_storage() {
        let rig = Rig::new();
        for &(cpu, value, after) in ON_TIME_PROM_SEQUENCE {
            store_as(&rig, cpu, value);
            assert_eq!(scratch1(&rig), after, "CPU {cpu} storing {value:#x}");
        }
        assert!((0..4).all(|c| joined(&rig, c)));
    }

    /// The same, from a four-processor Tezro boot in which CPUs 2 and 3
    /// cleared after CPU 0's release (and CPU 1 joined before either had
    /// stored again).
    const LATE_PROM_SEQUENCE: &[(u8, u64, u64)] = &[
        (1, 0, 0), (0, 0, 0),
        (0, RELEASE, RELEASE),
        (2, 0, RELEASE), (3, 0, RELEASE),
        (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 0, 0),
        (2, FLAGS, FLAGS), (0, FLAGS, FLAGS), (3, FLAGS, FLAGS), (1, FLAGS, FLAGS),
    ];

    /// Replayed, the late clears do not erase the release, and the late
    /// processors read the release until their joins even after CPU 1's
    /// join has replaced it.
    #[test]
    fn the_logged_late_prom_sequence_keeps_the_release() {
        let rig = Rig::new();
        for (i, &(cpu, value, after)) in LATE_PROM_SEQUENCE.iter().enumerate() {
            store_as(&rig, cpu, value);
            assert_eq!(scratch1(&rig), after, "store {i}: CPU {cpu} storing {value:#x}");
            if i == 5 {
                assert_eq!((read_as(&rig, 2), read_as(&rig, 3)), (RELEASE, RELEASE));
            }
        }
        assert!((0..4).all(|c| joined(&rig, c)));
    }

    /// The soft-reset bit resets the block, as a power-on does, and asks the
    /// board for a node reset once; the other bits do neither.
    #[test]
    fn the_soft_reset_bit_resets_the_block_and_asks_for_a_node_reset() {
        let rig = Rig::new();
        rig.w(LB_BASE + reg::NODE_REGION, 5 << 32);
        rig.w(LB_BASE + reg::PIO_WINDOW + 8, 42);
        rig.w(LB_BASE + reg::REGISTER_C030, 0xdead);
        rig.w(LB_BASE + reg::SOFT_RESET, 2);
        assert!(rig.events.lock().try_recv().is_err());
        assert_eq!(rig.r(LB_BASE + reg::REGISTER_C030), 0xdead);
        rig.w(LB_BASE + reg::SOFT_RESET, 1);
        assert!(matches!(rig.events.lock().try_recv(), Ok(crate::traits::MachineEvent::HardReset)));
        assert!(rig.events.lock().try_recv().is_err(), "one request");
        assert_eq!(rig.r(LB_BASE + reg::REGISTER_C030), 0);
        assert_eq!(rig.r(LB_BASE + reg::PIO_WINDOW + 8), 42);
        assert_eq!(rig.chip.node.load(Ordering::SeqCst), 0);
    }

    impl Lb {
        /// Hold the block's lock, for tests of the paths that must not take it.
        pub(in crate::dev::bedrock) fn hold(&self) -> impl Sized + '_ {
            self.regs.lock()
        }
    }
}
