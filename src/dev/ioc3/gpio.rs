//! The IOC3's general-purpose I/O pins: a control word (set and clear
//! aliases), a data word and one register per pin that reads and writes that
//! pin's bit.
//!
//! The chip has the pins; what hangs off them is the board's. A pin the
//! board straps to a level reads that level whatever is written. A pin the
//! board wires to a front-panel LED is mirrored to a status-bar bit, which
//! is a host feature and runs only when pin data is written.
//!
//! Layout follows Linux's `struct ioc3` (`gpcr_s`, `gpcr_c`, `gpdr`,
//! `gppr`) in `arch/mips/include/asm/sn/ioc3.h`.
//!
//! Ported from hw/misc/sgi-ioc3.c in jamesbraid/qemu-sgi-origami.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use super::reg;

/// Whether an LED is lit by its pin going high. Linux's `ip30-leds` writes 1
/// to light one; the board's polarity is inferred from that. Set false if
/// the pins turn out to be active low.
const LIT_WHEN_HIGH: bool = true;

/// What the board wires to one GPIO pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinWire {
    pub pin: u8,
    pub to: PinUse,
}

impl PinWire {
    pub const fn strap(pin: u8, high: bool) -> PinWire {
        PinWire { pin, to: PinUse::Strap(high) }
    }

    pub const fn status_bit(pin: u8, bit: u64) -> PinWire {
        PinWire { pin, to: PinUse::StatusBit(bit) }
    }
}

/// How a board uses a pin. A pin a board does not list is unwired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinUse {
    /// An input the board holds at a fixed level.
    Strap(bool),
    /// An output the host mirrors to this status-bar bit.
    StatusBit(u64),
}

pub(super) struct Gpio {
    control: AtomicU32,
    data: AtomicU32,
    /// Pins the board holds, and the levels it holds them at.
    strap_mask: u32,
    strap_values: u32,
    /// Output pin and the status-bar bit it lights.
    lit: Vec<(u8, u64)>,
    /// The status bar's activity word, shared with the graphics board.
    heartbeat: OnceLock<Arc<AtomicU64>>,
}

impl Gpio {
    /// The chip with the pins `wiring` lists wired, from the board's row.
    pub(super) fn new(wiring: &[PinWire]) -> Gpio {
        let (mut strap_mask, mut strap_values) = (0, 0);
        let mut lit = Vec::new();
        for w in wiring {
            match w.to {
                PinUse::Strap(high) => {
                    strap_mask |= 1 << w.pin;
                    strap_values |= (high as u32) << w.pin;
                }
                PinUse::StatusBit(bit) => lit.push((w.pin, bit)),
            }
        }
        Gpio {
            control: AtomicU32::new(0),
            data: AtomicU32::new(0),
            strap_mask,
            strap_values,
            lit,
            heartbeat: OnceLock::new(),
        }
    }

    pub(super) fn set_heartbeat(&self, heartbeat: Arc<AtomicU64>) {
        let _ = self.heartbeat.set(heartbeat);
    }

    /// Whether `a` is one of the GPIO registers.
    pub(super) fn owns(a: u32) -> bool {
        // Control set, control clear and data sit in the words just below the
        // first per-pin register, so one range covers all of them.
        (reg::PIN_CONTROL_SET..=reg::PIN_REG_LAST).contains(&a) && a & 3 == 0
    }

    /// The control word, which also routes the MicroLAN engine to its pin.
    pub(super) fn control(&self) -> u32 {
        self.control.load(Ordering::SeqCst)
    }

    fn data_now(&self) -> u32 {
        (self.data.load(Ordering::SeqCst) & !self.strap_mask) | self.strap_values
    }

    pub(super) fn read(&self, a: u32) -> u32 {
        match a {
            // Both write aliases read the control word.
            reg::PIN_CONTROL_SET | reg::PIN_CONTROL_CLR => self.control(),
            reg::PIN_DATA => self.data_now(),
            _ => (self.data_now() >> ((a - reg::PIN_REG_FIRST) / 4)) & 1,
        }
    }

    pub(super) fn write(&self, a: u32, val: u32) {
        match a {
            reg::PIN_CONTROL_SET => {
                self.control.fetch_or(val, Ordering::SeqCst);
            }
            reg::PIN_CONTROL_CLR => {
                self.control.fetch_and(!val, Ordering::SeqCst);
            }
            reg::PIN_DATA => {
                let old = self.data.swap(val, Ordering::SeqCst);
                self.mirror(old, val);
            }
            _ => {
                let bit = 1 << ((a - reg::PIN_REG_FIRST) / 4);
                if val & 1 != 0 {
                    let old = self.data.fetch_or(bit, Ordering::SeqCst);
                    self.mirror(old, old | bit);
                } else {
                    let old = self.data.fetch_and(!bit, Ordering::SeqCst);
                    self.mirror(old, old & !bit);
                }
            }
        }
    }

    /// Show on the status bar what a write moved from `before` to `after`.
    /// This is the only place the host feature runs, so a board with no
    /// wired LED pays a check of an empty list per pin write and nothing on
    /// any other access.
    fn mirror(&self, before: u32, after: u32) {
        if self.lit.is_empty() || before == after {
            return;
        }
        let Some(hb) = self.heartbeat.get() else { return };
        for &(pin, bit) in &self.lit {
            let (old, new) = ((before >> pin) & 1, (after >> pin) & 1);
            if old == new {
                continue;
            }
            if (new == 1) == LIT_WHEN_HIGH {
                hb.fetch_or(bit, Ordering::Relaxed);
            } else {
                hb.fetch_and(!bit, Ordering::Relaxed);
            }
        }
    }

    pub(super) fn reset(&self) {
        self.control.store(0, Ordering::SeqCst);
        self.data.store(0, Ordering::SeqCst);
    }

    /// The raw words, for a snapshot: control, then data.
    pub(super) fn words(&self) -> (u32, u32) {
        (self.control.load(Ordering::SeqCst), self.data.load(Ordering::SeqCst))
    }

    pub(super) fn restore(&self, control: u32, data: u32) {
        self.control.store(control, Ordering::SeqCst);
        self.data.store(data, Ordering::SeqCst);
    }
}
