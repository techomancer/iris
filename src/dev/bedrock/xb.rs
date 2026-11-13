//! Bedrock's crossbar block (XB) records the fabric error queues and lets
//! software service queued errors one at a time; its error level reaches both
//! processor interfaces.
//!
//! # What is approximated here
//!
//! Arbitration timing and diagnostic outputs are not modeled. Unsupported
//! queue reads return zero, while writes are logged and ignored.
//!
//! Ported from hw/misc/sgi-bedrock-xb.c in jamesbraid/qemu-sgi-origami.

use super::pi::Levels;
use super::Effect;
use parking_lot::Mutex;

/// Length of the block-relative register aperture.
pub(super) const SIZE: u64 = 0x198;

pub mod reg {
    pub const CONTROL_BASE: u64 = 0x0;
    pub const CONTROL_END: u64 = 0x18;
    pub const QUEUE_BASE: u64 = 0x20;
    pub const QUEUE_END: u64 = 0x88;
    pub const FIRST_ERROR: u64 = 0x90;
    pub const DEBUG_BASE: u64 = 0x98;
    pub const DEBUG_END: u64 = 0xa0;
    pub const CLEAR_QUEUE_BASE: u64 = 0x120;
    pub const CLEAR_QUEUE_END: u64 = 0x188;
    pub const CLEAR_FIRST_ERROR: u64 = 0x190;
}

const MASKS: [u64; 14] = [
    0x3333, 0x33, 0, 0, 0x3330, 0, 0, 0xf0, 0xfff0, 0, 0, 0, 0x3303, 0x11,
];
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
            control: [0x0008000080000021, 0, 0, 0],
            queues: [0; 14],
            first: 0,
            debug: [0; 2],
            unserviced: 0,
        }
    }
}
impl Regs {
    fn error_level(&self) -> bool {
        self.first & 0x1000 != 0
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
            reg::CONTROL_BASE..=reg::CONTROL_END => r.control[a as usize / 8],
            reg::QUEUE_BASE..=reg::QUEUE_END => {
                let n = (a - reg::QUEUE_BASE) as usize / 8;
                if MASKS[n] == 0 {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB unsupported queue {n}");
                }
                r.queues[n]
            }
            reg::FIRST_ERROR => r.first,
            reg::DEBUG_BASE | reg::DEBUG_END => r.debug[(a - reg::DEBUG_BASE) as usize / 8],
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
            reg::CONTROL_BASE..=reg::CONTROL_END => {
                let n = a as usize / 8;
                r.control[n] = v & [0x000ffffffffffff1, 0xffffffff, 0x7ff, 0xf][n];
                if a == 0x10 && r.control[n] != 0 {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB traffic observation unsupported");
                }
            }
            reg::QUEUE_BASE..=reg::QUEUE_END => {
                let n = (a - reg::QUEUE_BASE) as usize / 8;
                let v = v & MASKS[n];
                if MASKS[n] == 0 {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB unsupported queue {n}");
                }
                if v != 0 {
                    r.queues[n] |= v;
                    r.unserviced |= 1 << n;
                    if r.first & 0x1000 == 0 {
                        r.first = 0x1000 | ((15 - n) as u64) << 8;
                    }
                }
            }
            reg::CLEAR_QUEUE_BASE..=reg::CLEAR_QUEUE_END => {
                let n = (a - reg::CLEAR_QUEUE_BASE) as usize / 8;
                r.queues[n] = 0;
                r.unserviced &= !(1 << n);
            }
            reg::CLEAR_FIRST_ERROR => {
                if r.first & 0x1000 != 0 {
                    let n = 15 - ((r.first >> 8) & 15) as usize;
                    if r.unserviced & (1 << n) != 0 {
                        crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB first error cleared before queue");
                    } else {
                        r.first = 0;
                        for n in 0..14 {
                            if r.queues[n] != 0 {
                                r.first = 0x1000 | ((15 - n) as u64) << 8;
                                break;
                            }
                        }
                    }
                }
            }
            reg::DEBUG_BASE | reg::DEBUG_END => {
                let n = (a - reg::DEBUG_BASE) as usize / 8;
                r.debug[n] = v & [0xffff, 7][n];
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
    // Cases follow tests/qtest/sn1-bedrock-xb-test.c in
    // jamesbraid/qemu-sgi-origami: control-storage, service-order and
    // cold-reset.
    use super::super::reg::{pi, PI1_BASE, XB_BASE};
    use super::super::Rig;
    use super::*;
    use crate::traits::{BusDevice, BUS_ERR};

    fn error_level(rig: &Rig) -> bool {
        let levels = [0, PI1_BASE].map(|base| rig.r(base + pi::PENDING_STATUS_B) & (1 << 54) != 0);
        assert_eq!(levels[0], levels[1], "both interfaces reflect bit 54");
        levels[0]
    }

    /// The guest sees masked crossbar control words and cleared errors after reset.
    #[test]
    fn crossbar_control_masks_apply_and_reset_clears_errors() {
        let rig = Rig::new();
        for (a, mask) in [
            (reg::CONTROL_BASE, 0x000ffffffffffff1),
            (reg::CONTROL_BASE + 8, 0xffffffff),
            (reg::CONTROL_BASE + 0x10, 0x7ff),
            (reg::CONTROL_END, 15),
            (reg::DEBUG_BASE, 0xffff),
            (reg::DEBUG_END, 7),
        ] {
            rig.w(XB_BASE + a, !0);
            assert_eq!(rig.r(XB_BASE + a), mask);
        }
        // Keep expected register values independent of the decoder's mask table.
        for (address, expected) in [
            (reg::QUEUE_BASE, 0x3333),
            (reg::QUEUE_BASE + 8, 0x0033),
            (reg::QUEUE_BASE + 0x10, 0),
            (reg::QUEUE_BASE + 0x18, 0),
            (reg::QUEUE_BASE + 0x20, 0x3330),
            (reg::QUEUE_BASE + 0x28, 0),
            (reg::QUEUE_BASE + 0x30, 0),
            (reg::QUEUE_BASE + 0x38, 0x00f0),
            (reg::QUEUE_BASE + 0x40, 0xfff0),
            (reg::QUEUE_BASE + 0x48, 0),
            (reg::QUEUE_BASE + 0x50, 0),
            (reg::QUEUE_BASE + 0x58, 0),
            (reg::QUEUE_BASE + 0x60, 0x3303),
            (reg::QUEUE_END, 0x0011),
        ] {
            rig.w(XB_BASE + address, !0);
            assert_eq!(rig.r(XB_BASE + address), expected, "register {address:#x}");
        }
        assert!(error_level(&rig));
        let _ = rig.chip.xb.power_on(&rig.chip.levels);
        assert_eq!(rig.r(XB_BASE + reg::CONTROL_BASE), 0x0008000080000021);
        assert!(!error_level(&rig));
        assert_eq!(rig.regs.read32(XB_BASE + reg::CONTROL_BASE).status, BUS_ERR);
    }
    /// The guest services queued crossbar faults in order, one error at a time.
    #[test]
    fn servicing_one_crossbar_error_exposes_the_next_pending_error() {
        {
            let rig = Rig::new();
            rig.w(XB_BASE + reg::QUEUE_BASE, 1);
            assert!(error_level(&rig));
            assert_eq!(rig.r(XB_BASE + reg::FIRST_ERROR), 0x1f00);
            rig.w(XB_BASE + reg::CLEAR_FIRST_ERROR, !0);
            assert_eq!(rig.r(XB_BASE + reg::FIRST_ERROR), 0x1f00);
            rig.w(XB_BASE + reg::CLEAR_QUEUE_END, !0);
            rig.w(XB_BASE + reg::CLEAR_QUEUE_BASE, 0);
            rig.w(XB_BASE + reg::CLEAR_FIRST_ERROR, 0);
            assert!(!error_level(&rig));
            let _ = rig.chip.xb.power_on(&rig.chip.levels);
            assert!(!error_level(&rig));
        }
        {
            let rig = Rig::new();
            rig.w(XB_BASE + reg::QUEUE_END, 0x10);
            rig.w(XB_BASE + reg::QUEUE_BASE, 1);
            assert_eq!(rig.r(XB_BASE + reg::FIRST_ERROR), 0x1200);
            rig.w(XB_BASE + reg::CLEAR_QUEUE_END, 0);
            rig.w(XB_BASE + reg::CLEAR_FIRST_ERROR, 0);
            assert_eq!(rig.r(XB_BASE + reg::FIRST_ERROR), 0x1f00);
            rig.w(XB_BASE + reg::CLEAR_QUEUE_BASE, 0);
            rig.w(XB_BASE + reg::QUEUE_BASE, 0x10);
            rig.w(XB_BASE + reg::CLEAR_FIRST_ERROR, 0);
            assert_eq!(rig.r(XB_BASE + reg::FIRST_ERROR), 0x1f00);
            rig.w(XB_BASE + reg::CLEAR_QUEUE_BASE, 0);
            rig.w(XB_BASE + reg::CLEAR_FIRST_ERROR, 0);
            assert!(!error_level(&rig));
        }
    }

    impl Xb {
        /// Hold the block's lock, for tests of the paths that must not take it.
        pub(in crate::dev::bedrock) fn hold(&self) -> impl Sized + '_ {
            self.regs.lock()
        }
    }
}
