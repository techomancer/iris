//! Bedrock NI exposes node routing, sideband and error registers. Fuel runs as
//! one node without a NUMAlink peer, but the guest still reads and programs
//! these local control words during machine setup.
//!
//! # What is approximated here
//!
//! Routing and sideband writes are retained; the network fabric and packet
//! delivery are not simulated. Unspecified power-on fields start at zero, and
//! the sideband word survives reset as the current compatibility behavior.
//!
//! Ported from hw/misc/sgi-bedrock-ni.c and hw/misc/sgi-bedrock-ni-regs.c
//! in jamesbraid/qemu-sgi-origami.

/// Length of the block-relative register aperture.
pub const SIZE: u64 = 0x2008;

pub mod reg {
    pub const NODE_ID: u64 = 0x0;
    pub const LOCAL_RESET: u64 = 0x8;
    pub const CONTROL_0: u64 = 0x10;
    pub const CONTROL_1: u64 = 0x18;
    pub const CONTROL_2: u64 = 0x20;
    pub const CONTROL_3: u64 = 0x30;
    pub const CONTROL_4: u64 = 0x38;
    pub const ERROR_STATUS: u64 = 0x100;
    pub const ERROR_STATUS_UPPER: u64 = 0x108;
    pub const ERROR_STATUS_LOWER: u64 = 0x110;
    pub const SIDE_BAND: u64 = 0x118;
    pub const ERROR_CLEAR: u64 = 0x120;
    pub const ROUTES: u64 = 0x1000;
    pub const ROUTES_END: u64 = 0x1400;
    pub const GLOBAL_CONTROL: u64 = 0x2000;
}

use crate::traits::{BusDevice, BusRead64, Resettable, BUS_ERR, BUS_OK};
use parking_lot::{Mutex, ReentrantMutex};
use std::sync::Arc;
type Level = Arc<dyn Fn(bool) + Send + Sync>;
#[derive(Clone)]
struct Regs {
    control: [u64; 8],
    routes: [u64; 128],
    global: u64,
    errors: u64,
    sideband: u64,
}
impl Default for Regs {
    fn default() -> Self {
        Self {
            control: [0; 8],
            routes: [0; 128],
            global: 0,
            errors: 0,
            sideband: 0,
        }
    }
}
pub struct BedrockNi {
    regs: Arc<Mutex<Regs>>,
    error: Mutex<Option<Level>>,
    delivery: ReentrantMutex<()>,
    driven: Mutex<Option<bool>>,
    reset: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}
impl Default for BedrockNi {
    fn default() -> Self {
        Self::new()
    }
}
impl BedrockNi {
    pub fn new() -> Self {
        let regs = Arc::new(Mutex::new(Regs::default()));
        crate::locks::register_mutex("bedrock-ni", &regs);
        Self {
            regs,
            error: Mutex::new(None),
            delivery: ReentrantMutex::new(()),
            driven: Mutex::new(None),
            reset: Mutex::new(None),
        }
    }
    pub fn connect_error(&self, cb: Level) {
        let _delivery = self.delivery.lock();
        *self.error.lock() = Some(cb);
        *self.driven.lock() = None;
        self.update_irq();
    }
    pub fn connect_local_reset(&self, cb: Arc<dyn Fn() + Send + Sync>) {
        *self.reset.lock() = Some(cb);
    }
    pub fn error_level(&self) -> bool {
        self.regs.lock().errors & 0x1fffffff000000 != 0
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
    fn mask(a: u64) -> u64 {
        match a {
            reg::CONTROL_0 => 1,
            reg::CONTROL_1 => 7,
            reg::CONTROL_2 => 0x1ff,
            reg::CONTROL_3 => 0x7efffffff,
            reg::CONTROL_4 => 0xffff,
            _ => 0,
        }
    }
}
impl Resettable for BedrockNi {
    fn power_on(&self) {
        let mut r = self.regs.lock();
        let sideband = r.sideband;
        *r = Regs::default();
        r.sideband = sideband;
        drop(r);
        self.update_irq();
    }
}
impl BusDevice for BedrockNi {
    fn read64(&self, a: u64) -> BusRead64 {
        if a & 7 != 0 {
            return BusRead64::err();
        }
        let mut r = self.regs.lock();
        let v = if Self::mask(a) != 0 {
            r.control[a as usize / 8]
        } else if (reg::ROUTES..reg::ROUTES_END).contains(&a) {
            r.routes[(a as usize - 0x1000) / 8]
        } else {
            match a {
                reg::NODE_ID => 0,
                reg::ERROR_STATUS => r.errors,
                reg::ERROR_STATUS_UPPER | reg::ERROR_STATUS_LOWER => 0,
                reg::SIDE_BAND => r.sideband,
                reg::ERROR_CLEAR => {
                    let v = r.errors;
                    r.errors = 0;
                    v
                }
                reg::GLOBAL_CONTROL => r.global,
                _ => {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock NI unsupported read {a:#x}");
                    0
                }
            }
        };
        drop(r);
        self.update_irq();
        BusRead64::ok(v)
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        if a & 7 != 0 {
            return BUS_ERR;
        }
        let mut r = self.regs.lock();
        if Self::mask(a) != 0 {
            r.control[a as usize / 8] = v & Self::mask(a);
        } else if (reg::ROUTES..reg::ROUTES_END).contains(&a) {
            r.routes[(a as usize - 0x1000) / 8] = v & 31;
        } else {
            match a {
                reg::LOCAL_RESET => {}
                reg::ERROR_STATUS => r.errors = v & ((1 << 53) - 1),
                reg::SIDE_BAND => r.sideband = v & 0xfffff,
                reg::GLOBAL_CONTROL => r.global = v & 31,
                _ => crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock NI unsupported or read-only write {a:#x} = {v:#x}"),
            }
        }
        drop(r);
        self.update_irq();
        if a == reg::LOCAL_RESET && v & 4 != 0 {
            let cb = self.reset.lock().clone();
            if let Some(cb) = cb {
                cb();
            }
        }
        BUS_OK
    }
}

#[cfg(test)]
mod tests {
    // Cases ported from tests/qtest/sn1-bedrock-test.c in jamesbraid/qemu-sgi-origami.
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// The guest programs route fields and reads the error register to acknowledge faults.
    #[test]
    fn route_registers_store_fields_and_error_reads_clear_pending_errors() {
        let d = Arc::new(BedrockNi::new());
        for (a, mask) in [
            (reg::CONTROL_0, 1),
            (reg::CONTROL_1, 7),
            (reg::CONTROL_2, 0x1ff),
            (reg::CONTROL_3, 0x7efffffff),
            (reg::CONTROL_4, 0xffff),
        ] {
            d.write64(a, !0);
            assert_eq!(d.read64(a).data, mask);
        }
        for a in [reg::ROUTES, reg::ROUTES_END - 8] {
            d.write64(a, !0);
            assert_eq!(d.read64(a).data, 31);
        }
        d.write64(reg::GLOBAL_CONTROL, !0);
        assert_eq!(d.read64(reg::GLOBAL_CONTROL).data, 31);
        let weak = Arc::downgrade(&d);
        d.connect_error(Arc::new(move |_| {
            let d = weak.upgrade().unwrap();
            let _ = d.read64(reg::SIDE_BAND);
        }));
        let resets = Arc::new(AtomicU64::new(0));
        let count = resets.clone();
        d.connect_local_reset(Arc::new(move || {
            count.fetch_add(1, Ordering::SeqCst);
        }));
        d.write64(reg::CONTROL_0, 0);
        d.write64(reg::LOCAL_RESET, 4);
        assert_eq!(resets.load(Ordering::SeqCst), 1);
        d.write64(reg::ERROR_STATUS, 0xffffff);
        assert!(!d.error_level());
        d.write64(reg::ERROR_STATUS, 1 << 24);
        assert!(d.error_level());
        d.read64(reg::ERROR_CLEAR);
        assert!(!d.error_level());
        d.write64(reg::SIDE_BAND, !0);
        d.power_on();
        assert_eq!(d.read64(reg::SIDE_BAND).data, 0xfffff);
        assert_eq!(d.read64(reg::CONTROL_3).data, 0);
    }

    /// The PROM's one power-on pass writes a sign-extended port-parameter constant and zero
    /// to the error and sideband words; each must read back through its defined bits only.
    #[test]
    fn power_on_programming_pass_reads_back_through_the_defined_bits() {
        let d = BedrockNi::new();
        d.write64(reg::CONTROL_0, 1);
        d.write64(reg::CONTROL_1, 0);
        d.write64(reg::CONTROL_2, 0x166);
        d.write64(reg::CONTROL_3, 0xffff_ffff_a3ff_1990);
        d.write64(reg::CONTROL_4, 0xc800);
        d.write64(reg::ERROR_STATUS, 0);
        d.read64(reg::ERROR_CLEAR);
        d.write64(reg::SIDE_BAND, 0);
        for (a, want) in [
            (reg::CONTROL_0, 1),
            (reg::CONTROL_1, 0),
            (reg::CONTROL_2, 0x166),
            (reg::CONTROL_3, 0x7_a3ff_1990),
            (reg::CONTROL_4, 0xc800),
            (reg::ERROR_STATUS, 0),
            (reg::SIDE_BAND, 0),
            (reg::ERROR_CLEAR, 0),
        ] {
            assert_eq!(d.read64(a).data, want, "register {a:#x}");
        }
    }

    /// Only the local-reset bit of the reset word reaches the node; the other bits must
    /// not restart it, and the interface's own registers are left to the board's reset.
    #[test]
    fn only_the_local_reset_bit_requests_a_node_reset() {
        let d = BedrockNi::new();
        let resets = Arc::new(AtomicU64::new(0));
        let count = resets.clone();
        d.connect_local_reset(Arc::new(move || {
            count.fetch_add(1, Ordering::SeqCst);
        }));
        let mut expected = 0;
        for enable in 0..2 {
            d.write64(reg::CONTROL_0, enable);
            for command in 0..8u64 {
                d.write64(reg::LOCAL_RESET, command);
                expected += u64::from(command & 4 != 0);
                assert_eq!(resets.load(Ordering::SeqCst), expected, "command {command:#x}");
            }
        }
        // With nothing wired to the callback the request does not clear this block.
        assert_eq!(d.read64(reg::CONTROL_0).data, 1);
    }

    /// A latched port error raises the interrupt line through its interrupting bits only;
    /// drivers rely on the line dropping when the status is rewritten or cleared.
    #[test]
    fn stored_error_status_drives_the_error_line_through_the_interrupting_bits() {
        let d = BedrockNi::new();
        let level = Arc::new(AtomicU64::new(0));
        let seen = level.clone();
        d.connect_error(Arc::new(move |high| seen.store(u64::from(high), Ordering::SeqCst)));
        for (value, high) in [(1u64 << 30, 1), (0xff, 0), (1 << 52, 1), (0, 0)] {
            d.write64(reg::ERROR_STATUS, value);
            assert_eq!(level.load(Ordering::SeqCst), high, "status {value:#x}");
        }
        // Bit 53 is beyond the register and must not be retained.
        d.write64(reg::ERROR_STATUS, 1 << 53);
        assert_eq!(d.read64(reg::ERROR_STATUS).data, 0);
    }

    /// Reset values of the programmable words are unknown on hardware, so the model clears them;
    /// a guest that re-runs its setup after a reset must see zeroes, not stale routes.
    #[test]
    fn reset_clears_every_programmable_register_and_misaligned_accesses_fault() {
        let d = BedrockNi::new();
        let offsets = [
            reg::CONTROL_0, reg::CONTROL_1, reg::CONTROL_2, reg::CONTROL_3, reg::CONTROL_4,
            reg::ROUTES, reg::ROUTES_END - 8, reg::GLOBAL_CONTROL,
        ];
        for a in offsets {
            d.write64(a, !0);
        }
        // A misaligned write must not land in the neighbouring register.
        assert_eq!(d.write64(reg::CONTROL_4 + 1, 0), BUS_ERR);
        assert_eq!(d.read64(reg::CONTROL_4 + 1).status, BUS_ERR);
        assert_eq!(d.read64(reg::CONTROL_4).data, 0xffff);
        d.power_on();
        for a in offsets {
            assert_eq!(d.read64(a).data, 0, "register {a:#x}");
        }
    }
}
