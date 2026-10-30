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
//! guest writes, but the register path carries no requester identity, so it
//! cannot enforce those permissions.
//!
//! The plain register words sit in a sparse map keyed by offset, since the
//! 0xc040-byte block holds only a few dozen of them.
//!
//! Ported from hw/misc/sgi-bedrock-lb.c in jamesbraid/qemu-sgi-origami.

use super::pi::Levels;
use super::reg::IIO_PROTECT_OVRRD_KEY;
use super::{Effect, NodeReset};
use crate::dev::w1;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};

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
const VECTOR_MASK: u64 = 3 | (0x1fffff << 3) | (0xff << 32) | (0x7ff << 40);
struct Regs {
    words: BTreeMap<u64, u64>,
    scratch: [u64; 5],
    node: u8,
    region: u8,
    deadline: Option<u64>,
    sample: bool,
}
impl Default for Regs {
    fn default() -> Self {
        let mut words = BTreeMap::new();
        words.insert(reg::PIO_PERMISSION, !0);
        words.insert(reg::INBOUND_PERMISSION, !0);
        Self { words, scratch: [0; 5], node: 0, region: 0, deadline: None, sample: true }
    }
}
impl Regs {
    fn error_level(&self) -> bool {
        self.words.get(&reg::ERROR_STATUS).copied().unwrap_or(0) != 0
    }
}

pub(super) struct Lb {
    regs: Mutex<Regs>,
}

impl Lb {
    pub(super) fn new() -> Self {
        Self { regs: Mutex::new(Regs::default()) }
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
    /// error status survive it. The node number returns to 0.
    fn reset(r: &mut Regs, node: &AtomicU8) {
        let survivors = [r.scratch[1], r.scratch[2], r.words.get(&reg::ERROR_STATUS).copied().unwrap_or(0)];
        *r = Regs::default();
        r.scratch[1] = survivors[0];
        r.scratch[2] = survivors[1];
        r.words.insert(reg::ERROR_STATUS, survivors[2]);
        node.store(0, Ordering::Release);
    }
    pub(super) fn power_on(&self, node: &AtomicU8) {
        Self::reset(&mut self.regs.lock(), node);
    }
    pub(super) fn read(&self, a: u64, now: u64) -> u64 {
        let mut r = self.regs.lock();
        Self::complete(&mut r, now);
        match a {
            reg::NODE_REGION => ID | ((r.node as u64) << 32) | ((r.region as u64) << 46),
            reg::PIO_WINDOW..=reg::PIO_WINDOW_END => r.scratch[(a - reg::PIO_WINDOW) as usize / 8],
            reg::ATOMIC_READ0 | reg::ATOMIC_READ1 => {
                let n = 3 + (a - reg::ATOMIC_READ0) as usize / 8;
                let v = r.scratch[n];
                r.scratch[n] = 1;
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
    /// A doubleword store at `a`. The error level reaches `levels` and the
    /// node number `node` under the block's lock, so concurrent stores
    /// publish in register order. A soft reset resets the block and asks
    /// for the node's.
    pub(super) fn write(&self, a: u64, v: u64, now: u64, levels: &Levels, node: &AtomicU8) -> Effect {
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
                    Self::reset(&mut r, node);
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
