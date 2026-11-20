//! Bedrock's local block (LB): node and region registers, atomic scratch aliases, error and vector
//! status, and the MicroLAN master for board identity parts.
//!
//! # What is approximated here
//!
//! A vector request is one register access at the far end of the port, latched at once; the reply's
//! return route stays zero (encoding not established). The MicroLAN slot completes after 1 us; no 1-wire
//! part answers. Permission registers retain writes but are not enforced: the register path carries no
//! requester identity beyond the calling CPU thread's.
//!
//! Scratch 1 orders the PROM's reset barrier ([`Barrier`]): a model rule with no hardware counterpart,
//! standing in for processors that leave reset on one clock edge (inferred).
//!
//! Ported from hw/misc/sgi-bedrock-lb.c in jamesbraid/qemu-sgi-origami.

use super::pi::Levels;
use super::regs::IIO_PROTECT_OVRRD_KEY;
use super::{Effect, NodeReset};
use crate::dev::w1;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Length of the block-relative register aperture.
pub(super) const SIZE: u64 = reg::VECTOR_STATUS_CLEAR::OFFSET + reg::VECTOR_STATUS_CLEAR::WIDTH as u64 / 8;

use super::regs::lb as reg;

const ID: u64 = (0x36 << 1) | (0x3012 << 12) | (2 << 28);
/// Scratch 1 in the plain scratch window.
const SCRATCH1: u64 = reg::SCRATCH1::OFFSET;
/// BUSY is status, not stored.
const VECTOR_MASK: u64 = field_masks!(reg::VECTOR_PARMS; TYPE, ADDRESS, WRITE_ID, PIO_ID);
/// The register offset a vector request names, in the target's window.
const VECTOR_OFFSET: u64 = reg::VECTOR_PARMS::ADDRESS::MASK;
const VECTOR_TYPE: u64 = reg::VECTOR_PARMS::TYPE::MASK;
/// The fields a vector reply echoes: address, write id and request id.
const VECTOR_ECHO: u64 = VECTOR_MASK & !VECTOR_TYPE;
/// Request type 1, a write; type 0 reads. The other two are refused.
const VECTOR_WRITE: u64 = reg::VECTOR_PARMS::TYPE::WRITE;
const VECTOR_VALID: u64 = reg::VECTOR_STATUS::VALID::MASK;
const VECTOR_OVERRUN: u64 = reg::VECTOR_STATUS::OVERRUN::MASK;
const ERROR_BITS_MASK: u64 = field_masks!(
    reg::ERROR_BITS; GCLK_DROP, RQ_BAD_INVAL, RQ_BAD_ADDR, RP_BAD_DATA, RQ_BAD_DATA, RP_LONG, RQ_LONG, RP_SHORT,
    RQ_SHORT, RP_BAD_CMD, RQ_BAD_CMD
);

/// A vector request for the chip to perform at the far end of the port.
pub(super) struct Launch {
    params: u64,
    route: u64,
    data: u64,
}
impl Launch {
    /// The route, four bits per hop from the least significant.
    pub(super) fn route(&self) -> u64 {
        self.route
    }
    /// The register offset in the target's window.
    pub(super) fn offset(&self) -> u64 {
        self.params & VECTOR_OFFSET
    }
    /// The word to write, for a write.
    pub(super) fn write(&self) -> Option<u64> {
        (self.params & VECTOR_WRITE != 0).then_some(self.data)
    }
}
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
        words.insert(reg::CPU_PERMISSION::OFFSET, !0);
        words.insert(reg::IO_PERMISSION::OFFSET, !0);
        Self { words, scratch: [0; 5], node: 0, region: 0, deadline: None, sample: true, barrier: Barrier::default() }
    }
}

/// The order of processors' stores to scratch 1 since the block's last reset, for the PROM's reset
/// barrier.
///
/// Hardware processors leave reset on one edge, so every clear lands before the boot master's
/// release (inferred). Host threads start late, and a late clear would erase the release, so a
/// first store arriving after another's second or later store is dropped and that processor reads
/// the release until it stores again.
///
/// Model rule, approximated: a function of the store sequence alone, never of values or host time;
/// accesses by no processor are plain storage. Limits: an L1 soft reset restarts processors without
/// an LB reset, so a second PROM run in one epoch gets no ordering; a peer's store is plain
/// storage; the requester is `smp::on_cpu_thread`.
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
        self.words.get(&reg::ERROR_BITS::OFFSET).copied().unwrap_or(0) != 0
    }
}

pub(crate) struct Lb {
    regs: Mutex<Regs>,
    /// Processors, by CPU number, whose presence join on scratch 1 has landed since reset; written
    /// under the lock.
    joined: AtomicU64,
    /// Scratch 3's bit 0, the IP35 PROM's hub lock (test-and-set twin at 0x8058); read without the
    /// lock.
    scratch3: AtomicU64,
    /// Processors that stored to their PI real-time compare since their join: the PROM's pair
    /// liveness post.
    posted: AtomicU64,
}

impl Lb {
    pub(super) fn new() -> Self {
        Self { regs: Mutex::new(Regs::default()), joined: AtomicU64::new(0), scratch3: AtomicU64::new(0), posted: AtomicU64::new(0) }
    }
    /// Processor `cpu` stored to its real-time compare register; the first such store after its
    /// join is the liveness post.
    pub(super) fn note_rt_compare(&self, cpu: usize) {
        if let Some(bit) = cpu_bit(Some(cpu)).filter(|bit| self.joined.load(Ordering::Acquire) & bit != 0) {
            self.posted.fetch_or(bit, Ordering::Release);
        }
    }
    /// The processors, one bit per CPU number, that have posted since their
    /// presence join. One atomic load.
    pub(crate) fn rt_posted(&self) -> u64 {
        self.posted.load(Ordering::Acquire)
    }
    /// Whether scratch 3's bit 0, the PROM's hub lock, is clear. One atomic
    /// load.
    pub(crate) fn hub_lock_free(&self) -> bool {
        self.scratch3.load(Ordering::Acquire) & 1 == 0
    }
    /// Whether processor `cpu`'s presence join on scratch 1 has landed since
    /// the block's last reset. One atomic load.
    #[cfg(test)]
    pub(crate) fn scratch1_joined(&self, cpu: usize) -> bool {
        cpu_bit(Some(cpu)).is_some_and(|bit| self.joined.load(Ordering::Acquire) & bit != 0)
    }
    pub(super) fn locked(&self) -> bool {
        self.regs.is_locked()
    }
    fn complete(r: &mut Regs, now: u64) {
        if r.deadline.is_some_and(|d| now >= d) {
            *r.words.entry(reg::MICROLAN_CTL::OFFSET).or_default() |= reg::MICROLAN_CTL::DONE::MASK | u64::from(r.sample);
            r.deadline = None;
        }
    }
    fn mask(a: u64) -> Option<u64> {
        Some(match a {
            reg::CPU_PERMISSION::OFFSET | reg::IO_PERMISSION::OFFSET | reg::REGION_PRESENT::OFFSET | reg::ERROR_DATA::OFFSET | reg::VECTOR_ROUTE::OFFSET | reg::VECTOR_DATA::OFFSET | reg::VECTOR_RETURN::OFFSET | reg::VECTOR_READ_DATA::OFFSET => !0,
            reg::NODES_ABSENT::OFFSET => field_masks!(
                reg::NODES_ABSENT; ENTRY3_VALID, ENTRY3_NODE, ENTRY2_VALID, ENTRY2_NODE, ENTRY1_VALID, ENTRY1_NODE,
                ENTRY0_VALID, ENTRY0_NODE
            ),
            reg::ERROR_BITS::OFFSET => ERROR_BITS_MASK,
            reg::ERROR_HDR1::OFFSET => field_masks!(reg::ERROR_HDR1; VALID, OVERRUN, ERR_TYPE, SOURCE, SUPPL, COMMAND),
            reg::ERROR_HDR2::OFFSET => reg::ERROR_HDR2::ADDRESS::MASK,
            reg::DEBUG_SELECT::OFFSET => reg::DEBUG_SELECT::SELECT::MASK,
            reg::RT_LOCAL_CTRL::OFFSET => {
                field_masks!(reg::RT_LOCAL_CTRL; USE_INTERNAL, GCLK, GCLK_COUNTER, MAX_COUNT, GCLK_ENABLE)
            }
            reg::RT_FILTER_CTRL::OFFSET => field_masks!(
                reg::RT_FILTER_CTRL; ERROR_COUNTER, DROPOUT_THRESHOLD, DROPOUT_COUNTER, MASK_ENABLE, MASK_COUNTER, OFFSET
            ),
            reg::VECTOR_PARMS::OFFSET => VECTOR_MASK,
            _ => return None,
        })
    }
    /// The block's reset: scratch 1, 2 and the error status survive; the node number returns to 0;
    /// a new barrier epoch starts.
    fn reset(&self, r: &mut Regs) {
        let survivors = [r.scratch[1], r.scratch[2], r.words.get(&reg::ERROR_BITS::OFFSET).copied().unwrap_or(0)];
        *r = Regs::default();
        self.joined.store(0, Ordering::Release);
        self.posted.store(0, Ordering::Release);
        self.scratch3.store(0, Ordering::Release);
        r.scratch[1] = survivors[0];
        r.scratch[2] = survivors[1];
        r.words.insert(reg::ERROR_BITS::OFFSET, survivors[2]);
    }
    pub(super) fn power_on(&self) {
        self.reset(&mut self.regs.lock());
    }
    /// A doubleword read at `a` by processor `requester`, or by none.
    pub(super) fn read(&self, a: u64, now: u64, requester: Option<usize>) -> u64 {
        let mut r = self.regs.lock();
        Self::complete(&mut r, now);
        match a {
            reg::REV_ID::OFFSET => {
                ID | ((r.node as u64) << reg::REV_ID::NODE_ID::SHIFT) | ((r.region as u64) << reg::REV_ID::REGION_SIZE::SHIFT)
            }
            SCRATCH1 => r.barrier.read(requester).unwrap_or(r.scratch[1]),
            reg::SCRATCH0::OFFSET..=reg::SCRATCH4::OFFSET => r.scratch[(a - reg::SCRATCH0::OFFSET) as usize / 8],
            reg::SCRATCH3_RZ::OFFSET | reg::SCRATCH4_RZ::OFFSET => {
                let n = 3 + (a - reg::SCRATCH3_RZ::OFFSET) as usize / 8;
                let v = r.scratch[n];
                r.scratch[n] = 1;
                self.scratch3.store(r.scratch[3], Ordering::Release);
                v
            }
            reg::VECTOR_STATUS_CLEAR::OFFSET => r.words.remove(&reg::VECTOR_STATUS::OFFSET).unwrap_or(0),
            reg::VECTOR_STATUS::OFFSET | reg::MICROLAN_CTL::OFFSET => r.words.get(&a).copied().unwrap_or(0),
            _ if Self::mask(a).is_some() => r.words.get(&a).copied().unwrap_or(0),
            _ => {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB unsupported read {a:#x}");
                0
            }
        }
    }
    /// A doubleword store by processor `requester`, or none. A region store returns the new node
    /// number; a soft reset asks for the node's.
    pub(super) fn write(&self, a: u64, v: u64, now: u64, requester: Option<usize>, levels: &Levels) -> Effect {
        let mut r = self.regs.lock();
        Self::complete(&mut r, now);
        let mut launch = None;
        let mut renumbered = false;
        match a {
            reg::REV_ID::OFFSET => {
                let region = ((v & reg::REV_ID::REGION_SIZE::MASK) >> reg::REV_ID::REGION_SIZE::SHIFT) as u8;
                if reg::REV_ID::REGION_SIZE::Value::from_bits(v) == Some(reg::REV_ID::REGION_SIZE::Value::INVALID) {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB invalid region size");
                } else {
                    let node_id = ((v & reg::REV_ID::NODE_ID::MASK) >> reg::REV_ID::NODE_ID::SHIFT) as u8;
                    renumbered = r.node != node_id;
                    r.node = node_id;
                    r.region = region;
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
            reg::SCRATCH0::OFFSET..=reg::SCRATCH4::OFFSET => {
                let n = (a - reg::SCRATCH0::OFFSET) as usize / 8;
                r.scratch[n] = if n < 3 { v } else { v & reg::SCRATCH3::BIT::MASK };
            }
            reg::SCRATCH0_WZ::OFFSET..=reg::SCRATCH2_WZ::OFFSET => {
                let n = (a - reg::SCRATCH0_WZ::OFFSET) as usize / 8;
                if r.scratch[n] == 0 {
                    r.scratch[n] = v;
                }
            }
            reg::CPU_PERMISSION_OVERRIDE::OFFSET => {
                if v == IIO_PROTECT_OVRRD_KEY {
                    *r.words.entry(reg::CPU_PERMISSION::OFFSET).or_default() |= 1;
                }
            }
            reg::SOFT_RESET::OFFSET => {
                if v & 1 != 0 {
                    self.reset(&mut r);
                    return Effect::Reset(NodeReset::LocalBlock);
                }
            }
            reg::MICROLAN_CTL::OFFSET => {
                if r.deadline.is_some() {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB MicroLAN write while slot pending discarded");
                } else {
                    r.words.insert(a, v & field_masks!(reg::MICROLAN_CTL; CLKDIV_PHI1, CLKDIV_PHI0, LOW_PULSE, SAMPLE_OFFSET));
                    let (pulse, _) = w1::slot::fields(v);
                    r.sample = pulse >= w1::slot::RESET_PULSE || pulse < w1::slot::ONE_PULSE;
                    r.deadline = Some(now.saturating_add(1000));
                }
            }
            reg::ERROR_BITS::OFFSET => {
                *r.words.entry(a).or_default() |= v & ERROR_BITS_MASK;
            }
            reg::ERROR_MASK_CLR::OFFSET => {
                *r.words.entry(reg::ERROR_BITS::OFFSET).or_default() &= !(v & reg::ERROR_MASK_CLR::CLEAR::MASK);
            }
            reg::VECTOR_PARMS::OFFSET => {
                let params = v & VECTOR_MASK;
                r.words.insert(a, params);
                if r.words.get(&reg::VECTOR_STATUS::OFFSET).copied().unwrap_or(0) & VECTOR_VALID != 0 {
                    // A launch over an unread reply only marks the overrun.
                    *r.words.entry(reg::VECTOR_STATUS::OFFSET).or_default() |= VECTOR_OVERRUN;
                } else if params & VECTOR_TYPE & !VECTOR_WRITE != 0 {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB vector request type {} refused", params & VECTOR_TYPE);
                } else {
                    let word = |a| r.words.get(&a).copied().unwrap_or(0);
                    launch = Some(Launch { params, route: word(reg::VECTOR_ROUTE::OFFSET), data: word(reg::VECTOR_DATA::OFFSET) });
                }
            }
            reg::VECTOR_STATUS::OFFSET | reg::VECTOR_STATUS_CLEAR::OFFSET => {
                if v == 0 {
                    r.words.insert(reg::VECTOR_STATUS::OFFSET, 0);
                } else {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB nonzero vector status write ignored");
                }
            }
            _ => {
                if let Some(mask) = Self::mask(a) {
                    r.words.insert(a, v & mask);
                    if (a == reg::CPU_PERMISSION::OFFSET || a == reg::IO_PERMISSION::OFFSET) && v != !0 {
                        crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB permission enforcement requires requester context");
                    }
                } else {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB unsupported write {a:#x} = {v:#x}");
                }
            }
        }
        self.scratch3.store(r.scratch[3], Ordering::Release);
        let reflect = levels.set(Levels::LB_ERROR, r.error_level());
        match launch {
            // Neither a vector request nor a region write changes a word the
            // error level reads.
            Some(launch) => {
                debug_assert!(!reflect);
                Effect::Vector(launch)
            }
            None if renumbered => {
                debug_assert!(!reflect);
                Effect::NodeNumber(crate::sn::node_id(u64::from(r.node), super::NODE_SHIFT))
            }
            None if reflect => Effect::Reflect,
            None => Effect::None,
        }
    }
    /// Latch the reply to `launch` (request echoed, request type in bits 2:0, a read's data), with
    /// no lock held.
    pub(super) fn vector_reply(&self, launch: &Launch, read: Option<u64>) {
        let mut r = self.regs.lock();
        if let Some(word) = read {
            r.words.insert(reg::VECTOR_READ_DATA::OFFSET, word);
        }
        r.words.insert(reg::VECTOR_RETURN::OFFSET, 0);
        r.words.insert(reg::VECTOR_STATUS::OFFSET, (launch.params & VECTOR_ECHO) | VECTOR_VALID | (launch.params & VECTOR_WRITE));
    }
}

#[cfg(test)]
mod tests {
    // Cases follow tests/qtest/sn1-bedrock-lb-test.c in jamesbraid/qemu-sgi-origami:
    // identity-permissions, scratch-reset and microlan-slot.
    use super::super::{connected_pair, Rig};
    use super::*;
    use crate::traits::{BusDevice, BUS_ERR};

    /// The guest configures node permissions and expects selected scratch words to survive reset.
    #[test]
    fn permission_writes_latch_errors_and_reset_retains_scratch() {
        {
            let rig = Rig::new();
            let node = || rig.chip.decode.number();
            rig.w(reg::BASE + reg::REV_ID::OFFSET, (0x5a << 32) | (2 << 46));
            assert_eq!(node(), 0x5a);
            rig.w(reg::BASE + reg::REV_ID::OFFSET, 3 << 46);
            assert_eq!(node(), 0x5a);
            rig.w(reg::BASE + reg::CPU_PERMISSION::OFFSET, 0);
            rig.w(reg::BASE + reg::CPU_PERMISSION_OVERRIDE::OFFSET, 0);
            assert_eq!(rig.r(reg::BASE + reg::CPU_PERMISSION::OFFSET), 0);
            rig.w(reg::BASE + reg::CPU_PERMISSION_OVERRIDE::OFFSET, IIO_PROTECT_OVRRD_KEY);
            assert_eq!(rig.r(reg::BASE + reg::CPU_PERMISSION::OFFSET), 1);
            rig.w(reg::BASE + reg::SCRATCH0::OFFSET + 8, 42);
            rig.w(reg::BASE + reg::ERROR_BITS::OFFSET, !0);
            assert_eq!(rig.r(reg::BASE + reg::ERROR_BITS::OFFSET), 0x7ff);
            assert!(rig.reflected(55), "bank 1 bit 55 on both interfaces");
            rig.chip.lb.power_on();
            assert_eq!(rig.r(reg::BASE + reg::SCRATCH0::OFFSET + 8), 42);
            assert!(rig.reflected(55), "the error status survives reset");
            rig.w(reg::BASE + reg::ERROR_MASK_CLR::OFFSET, 0x7ff);
            assert!(!rig.reflected(55));
            assert_eq!(node(), 0x5a, "the chip, not the block, publishes the number");
            crate::traits::Resettable::power_on(&*rig.chip);
            assert_eq!(node(), 0);
        }
        {
            let rig = Rig::new();
            rig.w(reg::BASE + reg::SCRATCH0_WZ::OFFSET, 42);
            rig.w(reg::BASE + reg::SCRATCH0_WZ::OFFSET, 43);
            assert_eq!(rig.r(reg::BASE + reg::SCRATCH0::OFFSET), 42);
            rig.w(reg::BASE + reg::VECTOR_READ_DATA::OFFSET, 0xdead);
            rig.w(reg::BASE + reg::VECTOR_PARMS::OFFSET, !0);
            assert_eq!(rig.r(reg::BASE + reg::VECTOR_STATUS::OFFSET), 0);
            assert_eq!(rig.r(reg::BASE + reg::VECTOR_READ_DATA::OFFSET), 0xdead);
            assert_eq!(rig.regs.read32(reg::BASE + reg::SCRATCH0::OFFSET).status, BUS_ERR);
            assert_eq!(rig.regs.read64(reg::BASE + 1).status, BUS_ERR);
        }
    }
    /// The guest polls MicroLAN completion and cannot replace a slot that is still pending.
    #[test]
    fn microlan_slots_finish_after_one_microsecond_and_reject_overlap() {
        let rig = Rig::new();
        let word = (0x66 << 27) | (60 << 10) | (0x41 << 2) | 3;
        rig.w(reg::BASE + reg::MICROLAN_CTL::OFFSET, word);
        assert_eq!(rig.r(reg::BASE + reg::MICROLAN_CTL::OFFSET), word & !3);
        rig.w(reg::BASE + reg::MICROLAN_CTL::OFFSET, 8 << 10);
        assert_eq!(rig.r(reg::BASE + reg::MICROLAN_CTL::OFFSET), word & !3);
        rig.set_time(1000);
        assert_eq!(rig.r(reg::BASE + reg::MICROLAN_CTL::OFFSET), (word & !3) | 2);
        rig.w(reg::BASE + reg::MICROLAN_CTL::OFFSET, 8 << 10);
        rig.set_time(2000);
        assert_eq!(rig.r(reg::BASE + reg::MICROLAN_CTL::OFFSET), (8 << 10) | 3);
        rig.w(reg::BASE + reg::MICROLAN_CTL::OFFSET, 60 << 10);
        rig.chip.lb.power_on();
        rig.set_time(3000);
        assert_eq!(rig.r(reg::BASE + reg::MICROLAN_CTL::OFFSET), 0);
    }
    const RELEASE: u64 = 0x07f8_0000;
    /// The PROM's flags word that follows the joins.
    const FLAGS: u64 = 0x6000_0000_0000_0000;
    const SCRATCH1_AT: u64 = reg::BASE + SCRATCH1;

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
        rig.chip.lb.scratch1_joined(cpu)
    }

    /// A clear arriving after the master's release does not erase it, and that processor's next
    /// store joins.
    #[test]
    fn a_late_clear_lands_below_the_release() {
        let rig = Rig::quad();
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

    /// A processor whose clear came after the release reads it until it stores again; others read
    /// the register.
    #[test]
    fn a_late_processor_reads_the_release_until_it_stores_again() {
        let rig = Rig::quad();
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

    /// Clears before the release land, the release lands over them, and is no join.
    #[test]
    fn a_release_after_every_clear_lands() {
        let rig = Rig::quad();
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

    /// The block's reset starts a new epoch: every processor's next store is a first store, and
    /// joins are forgotten.
    #[test]
    fn each_block_reset_starts_a_new_barrier() {
        let rig = Rig::quad();
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
        rig.chip.lb.power_on();
        assert!(!joined(&rig, 1));
        assert_eq!(scratch1(&rig), 0x100, "scratch 1 survives the reset");
        epoch(0);
        rig.w(reg::BASE + reg::SOFT_RESET::OFFSET, 1);
        assert!(!joined(&rig, 0));
        epoch(1);
    }

    /// Stores no processor makes (a monitor, a test, a peer's vector write) always land and move
    /// nothing.
    #[test]
    fn a_store_from_no_processor_always_lands() {
        let rig = Rig::quad();
        for value in [RELEASE, 0, RELEASE, 0] {
            rig.w(SCRATCH1_AT, value);
            assert_eq!(scratch1(&rig), value);
        }
        store_as(&rig, 1, 0);
        assert_eq!(scratch1(&rig), 0, "no release has landed from a processor");
    }

    /// (CPU, value stored, scratch 1 after) for the PROM's stores up to its flags word, from a
    /// Tezro boot with every processor on time.
    const ON_TIME_PROM_SEQUENCE: &[(u8, u64, u64)] = &[
        (0, 0, 0), (1, 0, 0), (3, 0, 0), (2, 0, 0),
        (0, RELEASE, RELEASE),
        (1, 0, 0), (0, 0, 0), (2, 0, 0), (3, 0, 0),
        (3, FLAGS, FLAGS), (2, FLAGS, FLAGS), (1, FLAGS, FLAGS), (0, FLAGS, FLAGS),
    ];

    /// The same from a boot in which CPUs 2 and 3 cleared after CPU 0's release.
    const LATE_PROM_SEQUENCE: &[(u8, u64, u64)] = &[
        (1, 0, 0), (0, 0, 0),
        (0, RELEASE, RELEASE),
        (2, 0, RELEASE), (3, 0, RELEASE),
        (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 0, 0),
        (2, FLAGS, FLAGS), (0, FLAGS, FLAGS), (3, FLAGS, FLAGS), (1, FLAGS, FLAGS),
    ];

    /// Replayed, on-time stores are plain storage; late clears do not erase the release, and late
    /// processors read it until their joins.
    #[test]
    fn the_logged_prom_sequences_keep_the_release() {
        for (late, sequence) in [(false, ON_TIME_PROM_SEQUENCE), (true, LATE_PROM_SEQUENCE)] {
            let rig = Rig::quad();
            for (i, &(cpu, value, after)) in sequence.iter().enumerate() {
                store_as(&rig, cpu, value);
                assert_eq!(scratch1(&rig), after, "store {i}: CPU {cpu} storing {value:#x}");
                if late && i == 5 {
                    assert_eq!((read_as(&rig, 2), read_as(&rig, 3)), (RELEASE, RELEASE));
                }
            }
            assert!((0..4).all(|c| joined(&rig, c)));
        }
    }

    /// The soft-reset bit resets the block as a power-on does and asks for a node reset once; other
    /// bits do neither.
    #[test]
    fn the_soft_reset_bit_resets_the_block_and_asks_for_a_node_reset() {
        let rig = Rig::quad();
        rig.w(reg::BASE + reg::REV_ID::OFFSET, 5 << 32);
        rig.w(reg::BASE + reg::SCRATCH0::OFFSET + 8, 42);
        rig.w(reg::BASE + reg::VECTOR_READ_DATA::OFFSET, 0xdead);
        rig.w(reg::BASE + reg::SOFT_RESET::OFFSET, 2);
        assert!(rig.events.lock().try_recv().is_err());
        assert_eq!(rig.r(reg::BASE + reg::VECTOR_READ_DATA::OFFSET), 0xdead);
        rig.w(reg::BASE + reg::SOFT_RESET::OFFSET, 1);
        assert!(matches!(rig.events.lock().try_recv(), Ok(crate::traits::MachineEvent::HardReset)));
        assert!(rig.events.lock().try_recv().is_err(), "one request");
        assert_eq!(rig.r(reg::BASE + reg::VECTOR_READ_DATA::OFFSET), 0);
        assert_eq!(rig.r(reg::BASE + reg::SCRATCH0::OFFSET + 8), 42);
        assert_eq!(rig.chip.decode.number(), 0);
    }

    // The vector engine cases follow tests/qtest/sn1-bedrock-link-test.c in
    // jamesbraid/qemu-sgi-origami.
    const SENTINEL: u64 = 0xdead_beef_dead_beef;
    const ECHO: u64 = (0x1fffff << 3) | (0xff << 32) | (0x7ff << 40);
    /// A register of the peer's local block, by its offset in the window.
    const LB_WINDOW: u64 = 0x60_0000;
    /// The firmware's own launch: read status clear, seed the read data and
    /// return route, write route and data, then the parameters last.
    fn launch(rig: &Rig, route: u64, params: u64, data: u64) -> u64 {
        rig.r(reg::BASE + reg::VECTOR_STATUS_CLEAR::OFFSET);
        rig.w(reg::BASE + reg::VECTOR_READ_DATA::OFFSET, SENTINEL);
        rig.w(reg::BASE + reg::VECTOR_RETURN::OFFSET, SENTINEL);
        rig.w(reg::BASE + reg::VECTOR_ROUTE::OFFSET, route);
        rig.w(reg::BASE + reg::VECTOR_DATA::OFFSET, data);
        rig.w(reg::BASE + reg::VECTOR_PARMS::OFFSET, params);
        rig.r(reg::BASE + reg::VECTOR_STATUS::OFFSET)
    }
    fn good_reply(status: u64, params: u64) {
        assert_eq!(status & VECTOR_VALID, VECTOR_VALID);
        assert_eq!(status & VECTOR_OVERRUN, 0);
        assert_eq!(status & 7, params & 1, "bits 2:0 echo the request type");
        assert_eq!(status & ECHO, params & ECHO);
    }

    /// Route 0 reads the peer's register, and a launch over an unread reply is an overrun and
    /// nothing else.
    #[test]
    fn route_zero_reads_the_peer() {
        let (a, b) = connected_pair();
        a.w(reg::BASE + reg::SCRATCH0::OFFSET, 0xaaaa_0000_aaaa_0000);
        b.w(reg::BASE + reg::SCRATCH0::OFFSET, 0xbbbb_0000_bbbb_0000);
        let params = (LB_WINDOW + reg::SCRATCH0::OFFSET) | (0x1c8 << 40);
        good_reply(launch(&a, 0, params, 0), params);
        assert_eq!(a.r(reg::BASE + reg::VECTOR_RETURN::OFFSET), 0);
        assert_eq!(a.r(reg::BASE + reg::VECTOR_READ_DATA::OFFSET), 0xbbbb_0000_bbbb_0000);
        good_reply(launch(&b, 0, params | (0x5a << 32), 0), params | (0x5a << 32));
        assert_eq!(b.r(reg::BASE + reg::VECTOR_READ_DATA::OFFSET), 0xaaaa_0000_aaaa_0000);
        // Reading status clear returns the reply and clears it.
        assert_eq!(a.r(reg::BASE + reg::VECTOR_STATUS_CLEAR::OFFSET) & VECTOR_VALID, VECTOR_VALID);
        assert_eq!(a.r(reg::BASE + reg::VECTOR_STATUS::OFFSET), 0);
        b.w(reg::BASE + reg::SCRATCH0::OFFSET, 0x1234);
        a.w(reg::BASE + reg::VECTOR_PARMS::OFFSET, params);
        a.w(reg::BASE + reg::VECTOR_PARMS::OFFSET, params);
        let status = a.r(reg::BASE + reg::VECTOR_STATUS::OFFSET);
        assert_eq!(status & (VECTOR_VALID | VECTOR_OVERRUN), VECTOR_VALID | VECTOR_OVERRUN);
        // A nonzero write to status leaves the reply, zero clears it.
        a.w(reg::BASE + reg::VECTOR_STATUS::OFFSET, !0);
        assert_eq!(a.r(reg::BASE + reg::VECTOR_STATUS::OFFSET), status);
        a.w(reg::BASE + reg::VECTOR_STATUS::OFFSET, 0);
        assert_eq!(a.r(reg::BASE + reg::VECTOR_STATUS::OFFSET), 0);
    }

    /// A vector write of scratch 1 and a read-modify-write of the revision word number the peer,
    /// not the launcher.
    #[test]
    fn route_zero_writes_the_peer() {
        let (a, b) = connected_pair();
        let scratch1 = LB_WINDOW + reg::SCRATCH0::OFFSET + 8;
        a.w(reg::BASE + reg::SCRATCH0::OFFSET + 8, 0x40_0000_0fff);
        b.w(reg::BASE + reg::SCRATCH0::OFFSET + 8, 0x40_0000_0fff);
        let params = scratch1 | (0x7b << 40) | (0x11 << 32) | 1;
        good_reply(launch(&a, 0, params, 0x40_0000_0001), params);
        assert_eq!(b.r(reg::BASE + reg::SCRATCH0::OFFSET + 8), 0x40_0000_0001);
        assert_eq!(a.r(reg::BASE + reg::SCRATCH0::OFFSET + 8), 0x40_0000_0fff);
        assert_eq!(a.r(reg::BASE + reg::VECTOR_READ_DATA::OFFSET), SENTINEL, "a write leaves the read data");
        let read = LB_WINDOW | (0x1c8 << 40);
        good_reply(launch(&a, 0, read, 0), read);
        let revision = a.r(reg::BASE + reg::VECTOR_READ_DATA::OFFSET);
        assert_eq!(revision, ID);
        let write = LB_WINDOW | (0x7b << 40) | 1;
        good_reply(launch(&a, 0, write, revision | (1 << 32)), write);
        assert_eq!(b.chip.decode.number(), 1);
        assert_eq!(a.chip.decode.number(), 0);
        assert_eq!(b.r(reg::BASE + reg::REV_ID::OFFSET), ID | (1 << 32));
    }

    /// A store the peer's vector engine makes is no processor's: the peer's barrier counts none of
    /// them.
    #[test]
    fn a_vector_write_is_no_processors_store_to_the_barrier() {
        let (a, b) = connected_pair();
        let scratch1 = LB_WINDOW + reg::SCRATCH0::OFFSET + 8;
        let params = scratch1 | 1;
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    crate::cpu::smp::set_current_cpu(0);
                    for value in [RELEASE, 0, RELEASE, 0] {
                        good_reply(launch(&a, 0, params, value), params);
                        assert_eq!(b.r(reg::BASE + reg::SCRATCH0::OFFSET + 8), value);
                    }
                })
                .join()
                .unwrap();
        });
        assert!(!joined(&b, 0), "the peer took the vector write for its own processor 0");
    }

    /// A remote request applies no permission check: a vector read of the peer's closed window
    /// succeeds.
    #[test]
    fn a_vector_request_ignores_the_peers_cell() {
        let (a, b) = connected_pair();
        b.w(reg::BASE + reg::SCRATCH0::OFFSET + 16, 0x77);
        b.w(super::super::regs::md::BASE + super::super::regs::md::CURRENT_CELL::OFFSET, 0);
        assert_eq!(b.regs.read64(reg::BASE + reg::SCRATCH0::OFFSET + 16).status, crate::traits::BUS_ERR);
        let read = (LB_WINDOW + reg::SCRATCH0::OFFSET + 16) | (0x1c8 << 40);
        good_reply(launch(&a, 0, read, 0), read);
        assert_eq!(a.r(reg::BASE + reg::VECTOR_READ_DATA::OFFSET), 0x77);
    }

    /// A nonzero route, a refused offset, a non-read/write type, no peer or a downed link latch no
    /// reply.
    #[test]
    fn unroutable_launches_latch_nothing() {
        let (a, b) = connected_pair();
        b.w(reg::BASE + reg::SCRATCH0::OFFSET + 8, 0x0fff);
        let write = (LB_WINDOW + reg::SCRATCH0::OFFSET + 8) | (0x7b << 40) | 1;
        assert_eq!(launch(&a, 1, write, 1), 0);
        assert_eq!(b.r(reg::BASE + reg::SCRATCH0::OFFSET + 8), 0x0fff);
        let nothing_there = 0xff_fff8 | (0x1c8 << 40);
        assert_eq!(launch(&a, 0, nothing_there, 0), 0);
        assert_eq!(a.r(reg::BASE + reg::VECTOR_RETURN::OFFSET), SENTINEL);
        assert_eq!(a.r(reg::BASE + reg::VECTOR_READ_DATA::OFFSET), SENTINEL);
        let read = (LB_WINDOW + reg::SCRATCH0::OFFSET + 8) | (0x1c8 << 40);
        assert_eq!(launch(&a, 0, read | 2, 0), 0, "type 2 is refused");
        let lone = Rig::new();
        assert_eq!(launch(&lone, 0, read, 0), 0);
        good_reply(launch(&a, 0, read, 0), read);
    }

    impl Lb {
        /// Hold the block's lock, for tests of the paths that must not take it.
        pub(in crate::dev::bedrock) fn hold(&self) -> impl Sized + '_ {
            self.regs.lock()
        }
    }
}
