//! Bedrock's crossbar block (XB): fabric error queues, serviced one at a time; its error level reaches both PIs.
//!
//! # What is approximated here
//!
//! Arbitration timing and diagnostic outputs are not modeled. Unsupported queue reads return zero;
//! writes are logged and ignored.
//!
//! Ported from hw/misc/sgi-bedrock-xb.c in jamesbraid/qemu-sgi-origami.

use super::pi::Levels;
use super::Effect;
use parking_lot::Mutex;

/// Length of the block-relative register aperture.
pub(super) const SIZE: u64 = reg::XB_FIRST_ERROR_CLEAR::OFFSET + reg::XB_FIRST_ERROR_CLEAR::WIDTH as u64 / 8;

use super::regs::xb as reg;

/// The stored bits of each queue's error register, in address order. Zero
/// for the queues the model does not support.
const MASKS: [u64; 14] = {
    let mut m = [0; 14];
    m[0] = field_masks!(reg::XB_POQ0_ERROR; TAIL_TIMEOUT, UNDERFLOW, OVERFLOW, INVALID_XSEL);
    m[1] = field_masks!(reg::XB_PIQ0_ERROR; DEADLOCK_TIMEOUT, OVERFLOW);
    m[4] = field_masks!(reg::XB_MP0_ERROR; TAIL_TIMEOUT, UNDERFLOW, OVERFLOW);
    m[7] = field_masks!(reg::XB_MIQ_ERROR; DEADLOCK_TIMEOUT);
    m[8] = field_masks!(reg::XB_NOQ_ERROR; TAIL_TIMEOUT, UNDERFLOW, OVERFLOW);
    m[12] = field_masks!(reg::XB_LOQ_ERROR; TAIL_TIMEOUT, UNDERFLOW, INVALID_XSEL);
    m[13] = field_masks!(reg::XB_LIQ_ERROR; DEADLOCK_TIMEOUT, OVERFLOW);
    m
};
/// Stored bits of XB_PARMS, XB_SLOW_GNT, XB_SPEW_CONTROL and XB_IOQ_ARB_TRIGGER.
const CONTROL_MASKS: [u64; 4] = [
    field_masks!(reg::XB_PARMS; TAIL_WRAP, DEADLOCK_WRAP, AGE_WRAP, BYP_EN),
    field_masks!(reg::XB_SLOW_GNT; PI0, PI1, MP0, MP1, MMQ, NI, II, LB),
    field_masks!(
        reg::XB_SPEW_CONTROL; SNOOP_POQ0, SNOOP_MP0, SNOOP_MMQ, SNOOP_NOQ, SNOOP_IOQ, SNOOP_LOQ, SNOOP_PIQ0, SNOOP_MIQ,
        SNOOP_NIQ, SNOOP_IIQ, SNOOP_LIQ
    ),
    reg::XB_IOQ_ARB_TRIGGER::CLOCKS::MASK,
];
/// Stored bits of XB_DEBUG_DATA_CTL and XB_DEBUG_ARB_CTL.
const DEBUG_MASKS: [u64; 2] = [
    field_masks!(
        reg::XB_DEBUG_DATA_CTL; OBSERVE_LODATA, OBSERVE_SOURCE_FIELD, OBSERVE_POQ0, OBSERVE_POQ1, OBSERVE_MMQ,
        OBSERVE_MP0, OBSERVE_MP1, OBSERVE_NOQ, OBSERVE_IOQ, OBSERVE_LOQ, OBSERVE_PIQ0, OBSERVE_PIQ1, OBSERVE_MIQ,
        OBSERVE_NIQ, OBSERVE_IIQ, OBSERVE_LIQ
    ),
    reg::XB_DEBUG_ARB_CTL::SELECT::MASK,
];
const FIRST_VALID: u64 = reg::XB_FIRST_ERROR::VALID::MASK;
/// The first-error source code of queue `n`: the codes count down from the
/// first queue's.
fn first_source(n: usize) -> u64 {
    (reg::XB_FIRST_ERROR::SOURCE::POQ0 - n as u64) << reg::XB_FIRST_ERROR::SOURCE::SHIFT
}
struct Regs {
    control: [u64; 4],
    queues: [u64; 14],
    first: u64,
    debug: [u64; 2],
    unserviced: u16,
}
impl Default for Regs {
    fn default() -> Self {
        Self {
            control: [reg::XB_PARMS::RESET, reg::XB_SLOW_GNT::RESET, reg::XB_SPEW_CONTROL::RESET, 0],
            queues: [0; 14],
            first: 0,
            debug: [0; 2],
            unserviced: 0,
        }
    }
}
impl Regs {
    fn error_level(&self) -> bool {
        self.first & FIRST_VALID != 0
    }
}

pub(super) struct Xb {
    regs: Mutex<Regs>,
}

impl Xb {
    pub(super) fn new() -> Self {
        Self { regs: Mutex::new(Regs::default()) }
    }
    pub(super) fn locked(&self) -> bool {
        self.regs.is_locked()
    }
    fn reflect(r: &Regs, levels: &Levels) -> Effect {
        if levels.set(Levels::XB_ERROR, r.error_level()) { Effect::Reflect } else { Effect::None }
    }
    pub(super) fn power_on(&self, levels: &Levels) -> Effect {
        let mut r = self.regs.lock();
        *r = Regs::default();
        Self::reflect(&r, levels)
    }
    pub(super) fn read(&self, a: u64) -> u64 {
        let r = self.regs.lock();
        match a {
            reg::XB_PARMS::OFFSET..=reg::XB_IOQ_ARB_TRIGGER::OFFSET => r.control[a as usize / 8],
            reg::XB_POQ0_ERROR::OFFSET..=reg::XB_LIQ_ERROR::OFFSET => {
                let n = (a - reg::XB_POQ0_ERROR::OFFSET) as usize / 8;
                if MASKS[n] == 0 {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB unsupported queue {n}");
                }
                r.queues[n]
            }
            reg::XB_FIRST_ERROR::OFFSET => r.first,
            reg::XB_DEBUG_DATA_CTL::OFFSET | reg::XB_DEBUG_ARB_CTL::OFFSET => r.debug[(a - reg::XB_DEBUG_DATA_CTL::OFFSET) as usize / 8],
            _ => {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB unsupported read {a:#x}");
                0
            }
        }
    }
    /// A doubleword store. The error level reaches `levels` under the
    /// block's lock.
    pub(super) fn write(&self, a: u64, v: u64, levels: &Levels) -> Effect {
        let mut r = self.regs.lock();
        match a {
            reg::XB_PARMS::OFFSET..=reg::XB_IOQ_ARB_TRIGGER::OFFSET => {
                let n = a as usize / 8;
                r.control[n] = v & CONTROL_MASKS[n];
                if a == reg::XB_SPEW_CONTROL::OFFSET && r.control[n] != 0 {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB traffic observation unsupported");
                }
            }
            reg::XB_POQ0_ERROR::OFFSET..=reg::XB_LIQ_ERROR::OFFSET => {
                let n = (a - reg::XB_POQ0_ERROR::OFFSET) as usize / 8;
                let v = v & MASKS[n];
                if MASKS[n] == 0 {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB unsupported queue {n}");
                }
                if v != 0 {
                    r.queues[n] |= v;
                    r.unserviced |= 1 << n;
                    if r.first & FIRST_VALID == 0 {
                        r.first = FIRST_VALID | first_source(n);
                    }
                }
            }
            reg::XB_POQ0_ERROR_CLEAR::OFFSET..=reg::XB_LIQ_ERROR_CLEAR::OFFSET => {
                let n = (a - reg::XB_POQ0_ERROR_CLEAR::OFFSET) as usize / 8;
                r.queues[n] = 0;
                r.unserviced &= !(1 << n);
            }
            reg::XB_FIRST_ERROR_CLEAR::OFFSET => {
                if r.first & FIRST_VALID != 0 {
                    let source = (r.first & reg::XB_FIRST_ERROR::SOURCE::MASK) >> reg::XB_FIRST_ERROR::SOURCE::SHIFT;
                    let n = (reg::XB_FIRST_ERROR::SOURCE::POQ0 - source) as usize;
                    if r.unserviced & (1 << n) != 0 {
                        crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB first error cleared before queue");
                    } else {
                        r.first = 0;
                        for n in 0..14 {
                            if r.queues[n] != 0 {
                                r.first = FIRST_VALID | first_source(n);
                                break;
                            }
                        }
                    }
                }
            }
            reg::XB_DEBUG_DATA_CTL::OFFSET | reg::XB_DEBUG_ARB_CTL::OFFSET => {
                let n = (a - reg::XB_DEBUG_DATA_CTL::OFFSET) as usize / 8;
                r.debug[n] = v & DEBUG_MASKS[n];
                if r.debug[n] != 0 {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB debug output unsupported");
                }
            }
            _ => crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB unsupported or read-only write {a:#x} = {v:#x}"),
        }
        Self::reflect(&r, levels)
    }
}

#[cfg(test)]
mod tests {
    // Cases follow tests/qtest/sn1-bedrock-xb-test.c in jamesbraid/qemu-sgi-origami:
    // control-storage, service-order and cold-reset.
    use super::super::Rig;
    use super::*;
    use crate::traits::{BusDevice, BUS_ERR};

    /// The guest sees masked crossbar control words and cleared errors after reset.
    #[test]
    fn crossbar_control_masks_apply_and_reset_clears_errors() {
        let rig = Rig::new();
        for (a, mask) in [
            (reg::XB_PARMS::OFFSET, 0x000ffffffffffff1),
            (reg::XB_PARMS::OFFSET + 8, 0xffffffff),
            (reg::XB_PARMS::OFFSET + 0x10, 0x7ff),
            (reg::XB_IOQ_ARB_TRIGGER::OFFSET, 15),
            (reg::XB_DEBUG_DATA_CTL::OFFSET, 0xffff),
            (reg::XB_DEBUG_ARB_CTL::OFFSET, 7),
            // Expected values stay independent of the decoder's mask table.
            (reg::XB_POQ0_ERROR::OFFSET, 0x3333),
            (reg::XB_POQ0_ERROR::OFFSET + 8, 0x0033),
            (reg::XB_POQ0_ERROR::OFFSET + 0x10, 0),
            (reg::XB_POQ0_ERROR::OFFSET + 0x18, 0),
            (reg::XB_POQ0_ERROR::OFFSET + 0x20, 0x3330),
            (reg::XB_POQ0_ERROR::OFFSET + 0x28, 0),
            (reg::XB_POQ0_ERROR::OFFSET + 0x30, 0),
            (reg::XB_POQ0_ERROR::OFFSET + 0x38, 0x00f0),
            (reg::XB_POQ0_ERROR::OFFSET + 0x40, 0xfff0),
            (reg::XB_POQ0_ERROR::OFFSET + 0x48, 0),
            (reg::XB_POQ0_ERROR::OFFSET + 0x50, 0),
            (reg::XB_POQ0_ERROR::OFFSET + 0x58, 0),
            (reg::XB_POQ0_ERROR::OFFSET + 0x60, 0x3303),
            (reg::XB_LIQ_ERROR::OFFSET, 0x0011),
        ] {
            rig.w(reg::BASE + a, !0);
            assert_eq!(rig.r(reg::BASE + a), mask, "register {a:#x}");
        }
        assert!(rig.reflected(54));
        let _ = rig.chip.xb.power_on(&rig.chip.levels);
        assert_eq!(rig.r(reg::BASE + reg::XB_PARMS::OFFSET), 0x0008000080000021);
        assert!(!rig.reflected(54));
        assert_eq!(rig.regs.read32(reg::BASE + reg::XB_PARMS::OFFSET).status, BUS_ERR);
    }
    /// The guest services queued crossbar faults in order, one error at a time.
    #[test]
    fn servicing_one_crossbar_error_exposes_the_next_pending_error() {
        {
            let rig = Rig::new();
            rig.w(reg::BASE + reg::XB_POQ0_ERROR::OFFSET, 1);
            assert!(rig.reflected(54));
            assert_eq!(rig.r(reg::BASE + reg::XB_FIRST_ERROR::OFFSET), 0x1f00);
            rig.w(reg::BASE + reg::XB_FIRST_ERROR_CLEAR::OFFSET, !0);
            assert_eq!(rig.r(reg::BASE + reg::XB_FIRST_ERROR::OFFSET), 0x1f00);
            rig.w(reg::BASE + reg::XB_LIQ_ERROR_CLEAR::OFFSET, !0);
            rig.w(reg::BASE + reg::XB_POQ0_ERROR_CLEAR::OFFSET, 0);
            rig.w(reg::BASE + reg::XB_FIRST_ERROR_CLEAR::OFFSET, 0);
            assert!(!rig.reflected(54));
            let _ = rig.chip.xb.power_on(&rig.chip.levels);
            assert!(!rig.reflected(54));
        }
        {
            let rig = Rig::new();
            rig.w(reg::BASE + reg::XB_LIQ_ERROR::OFFSET, 0x10);
            rig.w(reg::BASE + reg::XB_POQ0_ERROR::OFFSET, 1);
            assert_eq!(rig.r(reg::BASE + reg::XB_FIRST_ERROR::OFFSET), 0x1200);
            rig.w(reg::BASE + reg::XB_LIQ_ERROR_CLEAR::OFFSET, 0);
            rig.w(reg::BASE + reg::XB_FIRST_ERROR_CLEAR::OFFSET, 0);
            assert_eq!(rig.r(reg::BASE + reg::XB_FIRST_ERROR::OFFSET), 0x1f00);
            rig.w(reg::BASE + reg::XB_POQ0_ERROR_CLEAR::OFFSET, 0);
            rig.w(reg::BASE + reg::XB_POQ0_ERROR::OFFSET, 0x10);
            rig.w(reg::BASE + reg::XB_FIRST_ERROR_CLEAR::OFFSET, 0);
            assert_eq!(rig.r(reg::BASE + reg::XB_FIRST_ERROR::OFFSET), 0x1f00);
            rig.w(reg::BASE + reg::XB_POQ0_ERROR_CLEAR::OFFSET, 0);
            rig.w(reg::BASE + reg::XB_FIRST_ERROR_CLEAR::OFFSET, 0);
            assert!(!rig.reflected(54));
        }
    }

    impl Xb {
        /// Hold the block's lock, for tests of the paths that must not take it.
        pub(in crate::dev::bedrock) fn hold(&self) -> impl Sized + '_ {
            self.regs.lock()
        }
    }
}
