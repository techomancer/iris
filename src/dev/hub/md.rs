//! The Hub's memory and directory interface: the memory configuration, the
//! error and DIMM registers, the one-wire master that reads the node's board
//! record, the slot identity and the diagnostic indicators.
//!
//! Offsets are from the base of the Hub window. The node's I2C controller is
//! mapped over this block by the shell, and is not here. The bring-up UART
//! after it is not modelled.
//!
//! # What is approximated here
//!
//! - MD_MEMORY_CONFIG is masked storage. No write resizes memory or changes
//!   error reporting. Its reset value is Linux's `MMC_RESET_DEFAULTS`.
//! - The eight directory, protocol, memory and miscellaneous error registers
//!   read zero: the model never latches a memory error. The two DIMM
//!   mode-initialisation registers are write-only and logged.
//! - MD_PERF_CNT0 stores 20 bits and never counts, and its event selector is
//!   not decoded.
//! - MD_SLOTID_USTAT reads the board signals the machine supplies. Bits 0 and
//!   2 on the Origin 200 and bit 7 on the Origin 2000 are approximations: no
//!   public source gives their values on those boards.
//! - The indicator registers are write-only. Writes are logged and nothing is
//!   stored; the Origin 200 pair at 0x22_0080 and 0x22_00a0 is treated like
//!   them, which is an approximation (Linux names those offsets
//!   microcontroller and UART registers). The stock PROM's progress codes are
//!   not decoded.
//! - A one-wire slot completes one microsecond of clock time after it
//!   starts. How long a slot takes on hardware is unknown; the model needs
//!   only that it does not complete inside the write that starts it. A write
//!   while a slot is under way starts another.
//! - The node's one-wire bus is the only net: the midplane net that an Origin
//!   2000 node's line also reaches is not wired.
//!
//! Ported from hw/misc/sgi-hub.c in jamesbraid/qemu-sgi-origami.

use crate::dev::heart::Clock;
use crate::dev::w1::{self, Bus};
use crate::devlog::LogModule;
use crate::traits::{BusDevice, BusRead64, Resettable, BUS_OK};
use parking_lot::Mutex;

pub mod reg {
    pub const CONFIG: u64 = 0x20_0018;
    pub const ERROR_FIRST: u64 = 0x20_0050;
    pub const ERROR_LAST: u64 = 0x20_0088;
    pub const MEM_DIMM_INIT: u64 = 0x20_0090;
    pub const DIR_DIMM_INIT: u64 = 0x20_0098;
    pub const MLAN_CTL: u64 = 0x20_00a8;
    pub const PERF_CNT0: u64 = 0x21_0010;
    /// The I2C controller's S0 and S1, mapped over the Hub window by the
    /// shell. The bring-up UART's registers follow them.
    pub const I2C_BASE: u64 = 0x22_0000;
    pub const SLOTID_USTAT: u64 = 0x22_0048;
    pub const LED0: u64 = 0x22_0050;
    pub const LED1: u64 = 0x22_0058;
    pub const OLD_LED0: u64 = 0x22_0080;
    pub const OLD_LED1: u64 = 0x22_00a0;
    pub const UART_BASE: u64 = 0x22_00c0;
    pub const UART_END: u64 = 0x22_0100;
}

/// First offset that is not the memory interface's.
pub const END: u64 = 0x40_0000;
pub const START: u64 = 0x20_0000;

const CONFIG_MASK: u64 = 0x003f_ffff_ffff_ffff;
const CONFIG_RESET: u64 = 0x001e_7fbc_3fff_ffff;
const PERF_MASK: u64 = 0xf_ffff;

const MLAN_RX: u64 = 1 << 0;
const MLAN_DONE: u64 = 1 << 1;
/// Sample time 9:2, pulse width 19:10, and the two timing phases 26:20 and
/// 33:27. The register is 34 bits wide; truncating it to 32 would rewrite the
/// upper phase.
const MLAN_WRITABLE: u64 = 0x3_ffff_fffc;
/// Both phases reset to 0x31, Linux's `MLAN_RESET_DEFAULTS`.
const MLAN_RESET: u64 = (0x31 << 20) | (0x31 << 27);
const MLAN_SLOT_NS: u64 = 1000;

const SLOT_FLASH_READY_LOW: u64 = 1 << 0;
const SLOT_XJMP_ID: u64 = 1 << 2;
const SLOT_FLASH_READY_HIGH: u64 = 1 << 4;
const SLOT_MASTER_MODULE: u64 = 1 << 7;
const SLOT_ID_MASK: u64 = 7;

/// What the board drives on the signals MD_SLOTID_USTAT samples.
#[derive(Clone, Copy, Debug, Default)]
pub struct SlotSignals {
    /// The midplane's slot identity, the low three bits, or `None` where the
    /// node sits in no numbered slot (the Origin 200).
    pub slot: Option<u8>,
    /// This processor module is the Origin 200 slave chassis.
    pub slave_module: bool,
}

struct State {
    config: u64,
    perf_cnt0: u64,
    mlan: u64,
    /// The level the slot sampled, reported once the slot's deadline passes.
    level: bool,
    /// Clock reading at which the slot started last completes.
    deadline: Option<u64>,
    bus: Option<Bus>,
}
impl State {
    fn new(bus: Option<Bus>) -> Self {
        Self { config: CONFIG_RESET, perf_cnt0: 0, mlan: MLAN_RESET, level: true, deadline: None, bus }
    }
}

pub struct Md {
    clock: Clock,
    signals: SlotSignals,
    state: Mutex<State>,
}

fn unimp(what: &str, off: u64, value: Option<u64>) {
    match value {
        Some(v) => crate::dlog!(LogModule::Hub, "Hub MD: write of {v:#x} to {what} {off:#x}"),
        None => crate::dlog!(LogModule::Hub, "Hub MD: read of {what} {off:#x}"),
    }
}

impl Md {
    pub fn new(clock: Clock, signals: SlotSignals) -> Self {
        Self { clock, signals, state: Mutex::new(State::new(None)) }
    }

    /// Attach the node's one-wire bus to the master.
    pub fn set_onewire(&self, bus: Bus) {
        self.state.lock().bus = Some(bus);
    }

    fn mlan_value(&self, s: &State) -> u64 {
        match s.deadline {
            Some(at) if (self.clock)() >= at => s.mlan | MLAN_DONE | if s.level { MLAN_RX } else { 0 },
            _ => s.mlan,
        }
    }

    fn slot_id(&self) -> u64 {
        let mut value = SLOT_FLASH_READY_LOW | SLOT_XJMP_ID | SLOT_FLASH_READY_HIGH
            | if self.signals.slave_module { 0 } else { SLOT_MASTER_MODULE };
        if let Some(bits) = self.signals.slot {
            value = (value & !SLOT_ID_MASK) | (u64::from(bits) & SLOT_ID_MASK);
        }
        value
    }

    /// A register's value without side effects, for the registers that store.
    pub fn peek(&self, off: u64) -> Option<u64> {
        let s = self.state.lock();
        Some(match off {
            reg::CONFIG => s.config,
            reg::MLAN_CTL => self.mlan_value(&s),
            reg::PERF_CNT0 => s.perf_cnt0,
            _ => return None,
        })
    }

    pub fn read(&self, off: u64) -> u64 {
        match off {
            reg::ERROR_FIRST..=reg::ERROR_LAST if off % 8 == 0 => 0,
            reg::SLOTID_USTAT => self.slot_id(),
            _ => match self.peek(off) {
                Some(v) => v,
                None => {
                    unimp("unimplemented or write-only register", off, None);
                    0
                }
            },
        }
    }

    pub fn write(&self, off: u64, value: u64) {
        match off {
            reg::CONFIG => self.state.lock().config = value & CONFIG_MASK,
            reg::ERROR_FIRST..=reg::ERROR_LAST if off % 8 == 0 => {
                unimp("error register (nothing latches an error to clear)", off, Some(value))
            }
            reg::MEM_DIMM_INIT | reg::DIR_DIMM_INIT => {
                unimp("DIMM mode initialisation register (write-only, no functional signature)", off, Some(value))
            }
            reg::MLAN_CTL => {
                // Programs the timing and starts a slot. The bus answers
                // inside the write, but the answer is reported only from the
                // slot's own deadline. A write cannot set done or the data.
                let mut s = self.state.lock();
                s.mlan = value & MLAN_WRITABLE;
                let (pulse, sample) = w1::slot::fields(value);
                s.level = match s.bus.as_mut() {
                    Some(bus) => w1::slot::slot(bus, pulse, sample),
                    None => pulse >= w1::slot::RESET_PULSE || pulse < w1::slot::ONE_PULSE,
                };
                s.deadline = Some((self.clock)().saturating_add(MLAN_SLOT_NS));
            }
            reg::PERF_CNT0 => self.state.lock().perf_cnt0 = value & PERF_MASK,
            reg::LED0 | reg::LED1 | reg::OLD_LED0 | reg::OLD_LED1 => {
                let which = u8::from(matches!(off, reg::LED1 | reg::OLD_LED1));
                crate::dlog!(LogModule::Hub, "Hub MD: indicator {which} shows {:#04x}", value & 0xff);
            }
            _ => unimp("unimplemented register", off, Some(value)),
        }
    }
}

impl Md {
    /// Whether the register state is locked now, for the `locks` monitor command.
    pub fn is_locked(&self) -> bool {
        self.state.is_locked()
    }
}

impl Resettable for Md {
    /// A slot started before reset must not complete after it.
    fn power_on(&self) {
        let mut s = self.state.lock();
        let bus = s.bus.take().map(|mut bus| {
            // The part resets its own protocol state.
            bus.reset();
            bus
        });
        *s = State::new(bus);
    }
}

impl BusDevice for Md {
    fn read64(&self, addr: u64) -> BusRead64 {
        BusRead64::ok(self.read(addr))
    }
    fn write64(&self, addr: u64, value: u64) -> u32 {
        self.write(addr, value);
        BUS_OK
    }
}

#[cfg(test)]
mod tests;
