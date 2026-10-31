//! Bedrock XB is the node's internal crossbar status block. It records the
//! fabric error queues and lets software service queued errors; the Fuel board
//! connects its outputs to the Bedrock processor interface.
//!
//! # What is approximated here
//!
//! Arbitration timing and diagnostic outputs are not modeled. Unsupported
//! queue reads return zero, while writes are logged and ignored.
//!
//! Ported from hw/misc/sgi-bedrock-xb.c in jamesbraid/qemu-sgi-origami.

/// Length of the block-relative register aperture.
pub const SIZE: u64 = 0x198;

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

use crate::traits::{BusDevice, BusRead64, Resettable, BUS_ERR, BUS_OK};
use parking_lot::{Mutex, ReentrantMutex};
use std::sync::Arc;
type Level = Arc<dyn Fn(bool) + Send + Sync>;
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
pub struct BedrockXb {
    regs: Arc<Mutex<Regs>>,
    error: Mutex<Option<Level>>,
    delivery: ReentrantMutex<()>,
    driven: Mutex<Option<bool>>,
}
impl Default for BedrockXb {
    fn default() -> Self {
        Self::new()
    }
}
impl BedrockXb {
    pub fn new() -> Self {
        let regs = Arc::new(Mutex::new(Regs::default()));
        crate::locks::register_mutex("bedrock-xb", &regs);
        Self {
            regs,
            error: Mutex::new(None),
            delivery: ReentrantMutex::new(()),
            driven: Mutex::new(None),
        }
    }
    pub fn connect_error(&self, cb: Level) {
        let _delivery = self.delivery.lock();
        *self.error.lock() = Some(cb);
        *self.driven.lock() = None;
        self.update_irq();
    }
    pub fn error_level(&self) -> bool {
        self.regs.lock().first & 0x1000 != 0
    }
    fn update_irq(&self) {
        // Serialize the sample and callback, while leaving registers available
        // to the callback. Suppress unchanged levels so callbacks can reenter
        // register reads that recompute the output.
        let _delivery = self.delivery.lock();
        let cb = self.error.lock().clone();
        if let Some(cb) = cb {
            let level = self.error_level();
            let changed = {
                let mut driven = self.driven.lock();
                let changed = *driven != Some(level);
                *driven = Some(level);
                changed
            };
            if changed {
                cb(level);
            }
        }
    }
}
impl Resettable for BedrockXb {
    fn power_on(&self) {
        *self.regs.lock() = Regs::default();
        self.update_irq();
    }
}
impl BusDevice for BedrockXb {
    fn read64(&self, a: u64) -> BusRead64 {
        if a & 7 != 0 {
            return BusRead64::err();
        }
        let r = self.regs.lock();
        let v = match a {
            reg::CONTROL_BASE..=reg::CONTROL_END => r.control[a as usize / 8],
            reg::QUEUE_BASE..=reg::QUEUE_END => {
                let n = (a as usize - 0x20) / 8;
                if MASKS[n] == 0 {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB unsupported queue {n}");
                }
                r.queues[n]
            }
            reg::FIRST_ERROR => r.first,
            reg::DEBUG_BASE | reg::DEBUG_END => r.debug[(a as usize - 0x98) / 8],
            _ => {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB unsupported read {a:#x}");
                0
            }
        };
        BusRead64::ok(v)
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        if a & 7 != 0 {
            return BUS_ERR;
        }
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
                let n = (a as usize - 0x20) / 8;
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
                let n = (a as usize - 0x120) / 8;
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
                let n = (a as usize - 0x98) / 8;
                r.debug[n] = v & [0xffff, 7][n];
                if r.debug[n] != 0 {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB debug output unsupported");
                }
            }
            _ => crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock XB unsupported or read-only write {a:#x} = {v:#x}"),
        }
        drop(r);
        self.update_irq();
        BUS_OK
    }
}

#[cfg(test)]
mod tests {
    // Cases follow tests/qtest/sn1-bedrock-xb-test.c in
    // jamesbraid/qemu-sgi-origami: control-storage, service-order and
    // cold-reset.
    use super::*;

    /// The guest sees masked crossbar control words and cleared errors after reset.
    #[test]
    fn crossbar_control_masks_apply_and_reset_clears_errors() {
        let d = BedrockXb::new();
        for (a, mask) in [
            (reg::CONTROL_BASE, 0x000ffffffffffff1),
            (reg::CONTROL_BASE + 8, 0xffffffff),
            (reg::CONTROL_BASE + 0x10, 0x7ff),
            (reg::CONTROL_END, 15),
            (reg::DEBUG_BASE, 0xffff),
            (reg::DEBUG_END, 7),
        ] {
            d.write64(a, !0);
            assert_eq!(d.read64(a).data, mask);
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
            d.write64(address, !0);
            assert_eq!(d.read64(address).data, expected, "register {address:#x}");
        }
        d.power_on();
        assert_eq!(d.read64(reg::CONTROL_BASE).data, 0x0008000080000021);
        assert!(!d.error_level());
        assert_eq!(d.read32(reg::CONTROL_BASE).status, BUS_ERR);
    }
    /// The guest services queued crossbar faults in order, one error at a time.
    #[test]
    fn servicing_one_crossbar_error_exposes_the_next_pending_error() {
        {
            let d = BedrockXb::new();
            d.write64(reg::QUEUE_BASE, 1);
            assert!(d.error_level());
            assert_eq!(d.read64(reg::FIRST_ERROR).data, 0x1f00);
            d.write64(reg::CLEAR_FIRST_ERROR, !0);
            assert_eq!(d.read64(reg::FIRST_ERROR).data, 0x1f00);
            d.write64(reg::CLEAR_QUEUE_END, !0);
            d.write64(reg::CLEAR_QUEUE_BASE, 0);
            d.write64(reg::CLEAR_FIRST_ERROR, 0);
            assert!(!d.error_level());
            d.power_on();
            assert!(!d.error_level());
        }
        {
            let d = BedrockXb::new();
            d.write64(reg::QUEUE_END, 0x10);
            d.write64(reg::QUEUE_BASE, 1);
            assert_eq!(d.read64(reg::FIRST_ERROR).data, 0x1200);
            d.write64(reg::CLEAR_QUEUE_END, 0);
            d.write64(reg::CLEAR_FIRST_ERROR, 0);
            assert_eq!(d.read64(reg::FIRST_ERROR).data, 0x1f00);
            d.write64(reg::CLEAR_QUEUE_BASE, 0);
            d.write64(reg::QUEUE_BASE, 0x10);
            d.write64(reg::CLEAR_FIRST_ERROR, 0);
            assert_eq!(d.read64(reg::FIRST_ERROR).data, 0x1f00);
            d.write64(reg::CLEAR_QUEUE_BASE, 0);
            d.write64(reg::CLEAR_FIRST_ERROR, 0);
            assert!(!d.error_level());
        }
    }
}
