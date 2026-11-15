//! The Hub's memory and directory interface: memory configuration, error and
//! DIMM registers, the one-wire master that reads the node's board record,
//! slot identity and diagnostic indicators. The node's I2C controller is
//! mapped over this block by the shell; the bring-up UART is not modelled.
//!
//! # What is approximated here
//!
//! - MD_MEMORY_CONFIG is masked storage (reset: Linux's `MMC_RESET_DEFAULTS`);
//!   no write resizes memory or changes error reporting.
//! - The eight directory, protocol, memory and miscellaneous error registers
//!   read zero: no memory error is ever latched. The two DIMM
//!   mode-initialisation registers are write-only and logged.
//! - MD_PERF_CNT0 stores 20 bits, never counts and decodes no event selector.
//! - MD_SLOTID_USTAT reads the machine's board signals. Bits 0 and 2 on the
//!   Origin 200 and bit 7 on the Origin 2000 are approximations; no public source gives them.
//! - The indicator registers are write-only: writes are logged, nothing is
//!   stored. The Origin 200 pair at 0x22_0080 and 0x22_00a0 is treated alike
//!   (Linux names them microcontroller and UART registers). PROM progress codes are not decoded.
//! - A one-wire slot completes one microsecond after it starts (the hardware
//!   time is unknown; it only must not finish inside the starting write). A
//!   write mid-slot starts another. The midplane net an Origin 2000 node's
//!   line also reaches is not wired.
//! - The ECC and directory back doors keep the low two bytes of each
//!   eight-byte entry; an unfitted bank answers a fixed pattern and drops writes.
//!
//! Ported from hw/misc/sgi-hub.c in jamesbraid/qemu-sgi-origami.

use crate::dev::heart::Clock;
use crate::dev::w1::{self, Bus};
use crate::devlog::LogModule;
use crate::physical::{BankMapping, Physical, RamBank};
use crate::sn_nodes::{node_mask, UnfittedMemory, SN0_NODE_SHIFT};
use crate::dev::widths::access_widths;
use crate::traits::{BusDevice, BusRead64, Resettable};
use parking_lot::Mutex;
use std::sync::Arc;

/// Banks an IP27 node board wires, each answering at its 512 MiB stride.
pub const BANKS: usize = 4;
pub const BANK_STRIDE: u64 = 512 << 20;
pub const SLOT_STRIDE: u64 = 128 << 20;
const MIB: u64 = 1 << 20;

/// Where bank `bank` of `mb` MiB lies, as `(node offset, length)` runs in
/// storage order: one run at its stride up to 128 MiB, else four at 128 MiB.
pub fn bank_mappings(bank: usize, mb: u32) -> Vec<(u64, u64)> {
    let size = u64::from(mb) * MIB;
    if size == 0 {
        return Vec::new();
    }
    let base = bank as u64 * BANK_STRIDE;
    if size <= SLOT_STRIDE {
        vec![(base, size)]
    } else {
        let run = size / 4;
        (0..4).map(|i| (base + i * SLOT_STRIDE, run)).collect()
    }
}

/// Every fitted run of `banks_mb` as a bank mapping at node offsets, in storage
/// order: the one place that skips empty banks and numbers the RAM.
pub fn fitted_mappings(banks_mb: [u32; BANKS]) -> Arc<[BankMapping]> {
    let mut runs = Vec::new();
    let mut index = 0;
    for (bank, &mb) in banks_mb.iter().enumerate() {
        if mb == 0 {
            continue;
        }
        let mut src = 0;
        for (at, len) in bank_mappings(bank, mb) {
            runs.push(BankMapping { base: at, end: at + len, bank: index, src });
            src += len;
        }
        index += 1;
    }
    runs.into()
}

/// The node's memory: fitted banks on a decode table the Hub's remap maps, plus
/// a ppmem window for `nodes` nodes that only speeds cached references.
pub fn memory(banks_mb: [u32; BANKS], nodes: usize) -> Result<Arc<Physical>, String> {
    let mut banks = Vec::new();
    for (i, &mb) in banks_mb.iter().enumerate() {
        if mb != 0 {
            banks.push(RamBank::try_new(mb as usize).map_err(|e| format!("bank {i}: {e}"))?);
        }
    }
    let phys = if nodes > 1 {
        let numbers = nodes.next_power_of_two().min(usize::from(node_mask(SN0_NODE_SHIFT)) + 1) as u64;
        Physical::with_window_regions(banks, numbers << SN0_NODE_SHIFT, 27)
    } else {
        Physical::with_window_regions(banks, crate::ppmem::WINDOW_SIZE, crate::ppmem::BITMAP_SHIFT)
    };
    phys.set_default(Arc::new(UnfittedMemory(LogModule::Hub)), "unfitted memory");
    Ok(phys)
}

use super::regs as reg;

/// First offset that is not the memory interface's.
pub const END: u64 = 0x40_0000;
pub const START: u64 = 0x20_0000;

const CONFIG_MASK: u64 = field_masks!(
    reg::MD_MEMORY_CONFIG; FPROM_CYC, FPROM_WR, UCTLR_CYC, UCTLR_WR, DIMM0_SEL, IO_PROT_EN, ARB_MLSS, IGNORE_ECC,
    DIR_PREMIUM, REPLY_GUAR, BANK7, BANK6, BANK5, BANK4, BANK3, BANK2, BANK1, BANK0
);
const CONFIG_RESET: u64 = reg::MD_MEMORY_CONFIG::RESET;
const PERF_MASK: u64 = reg::MD_PERF_CNT0::PERF_CNT::MASK;

const MLAN_RX: u64 = reg::MD_MLAN_CTL::RD_DATA::MASK;
const MLAN_DONE: u64 = reg::MD_MLAN_CTL::DONE::MASK;
/// Sample time 9:2, pulse width 19:10, timing phases 26:20 and 33:27. The
/// register is 34 bits wide; truncating to 32 would rewrite the upper phase.
const MLAN_WRITABLE: u64 = field_masks!(reg::MD_MLAN_CTL; SAMPLE, PULSE, PHI0, PHI1);
/// Both phases reset to 0x31, Linux's `MLAN_RESET_DEFAULTS`.
const MLAN_RESET: u64 = (0x31 << 20) | (0x31 << 27);
const MLAN_SLOT_NS: u64 = 1000;

const SLOT_FLASH_READY_LOW: u64 = 1 << 0;
const SLOT_XJMP_ID: u64 = 1 << 2;
const SLOT_FLASH_READY_HIGH: u64 = reg::MD_SLOTID_USTAT::FPROM_READY::MASK;
const SLOT_MASTER_MODULE: u64 = 1 << 7;
const SLOT_ID_MASK: u64 = reg::MD_SLOTID_USTAT::HUB_SLOT_ID::MASK;

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

const BACKDOOR_ENTRY: usize = 8;
const BACKDOOR_LIVE: usize = 2;
const BACKDOOR_BANK_STRIDE: u64 = 128 * MIB;
/// What an unfitted bank's back door reads: firmware's presence probe calls a
/// bank absent when neither a pattern nor zero survives a write.
const BACKDOOR_ABSENT: u64 = 0xa5a5_a5a5_a5a5_a5a5;

/// One back door window (1 GiB, eight banks of 128 MiB): per fitted bank, a
/// quarter of the bank's size of storage folded across the bank's span.
struct Backdoor {
    spans: [u64; 8],
    offsets: [u64; 8],
    data: Mutex<Vec<u8>>,
}

impl Backdoor {
    fn new(banks_mb: [u32; BANKS]) -> Self {
        let mut spans = [0u64; 8];
        let mut offsets = [0u64; 8];
        let mut total = 0;
        for (i, &mb) in banks_mb.iter().enumerate() {
            spans[i] = u64::from(mb) * MIB / 4;
            offsets[i] = total;
            total += spans[i];
        }
        Self { spans, offsets, data: Mutex::new(vec![0; total as usize]) }
    }

    /// The storage index of byte `a`, or None for a bank that is not fitted.
    fn index(&self, a: u64) -> Option<usize> {
        let bank = (a / BACKDOOR_BANK_STRIDE) as usize;
        let span = *self.spans.get(bank)?;
        (span != 0).then(|| (self.offsets[bank] + (a % BACKDOOR_BANK_STRIDE) % span) as usize)
    }

    fn read(&self, a: u64, n: usize) -> u64 {
        let Some(at) = self.index(a) else {
            return BACKDOOR_ABSENT & (u64::MAX >> (64 - 8 * n as u32));
        };
        let data = self.data.lock();
        (0..n).fold(0u64, |v, i| (v << 8) | u64::from(data[at + i]))
    }

    fn write(&self, a: u64, n: usize, v: u64) {
        let Some(at) = self.index(a) else { return };
        let mut data = self.data.lock();
        for i in 0..n {
            // Only the low two bytes of an eight-byte entry are stored.
            if (a as usize + i) % BACKDOOR_ENTRY >= BACKDOOR_ENTRY - BACKDOOR_LIVE {
                data[at + i] = (v >> (8 * (n - 1 - i))) as u8;
            }
        }
    }

    #[inline(always)]
    fn access(&self, a: u64, n: usize, value: Option<u64>) -> BusRead64 {
        match value {
            Some(v) => self.write(a, n, v),
            None => return BusRead64::ok(self.read(a, n)),
        }
        BusRead64::ok(0)
    }
}
impl BusDevice for Backdoor {
    access_widths!();
}

pub struct Md {
    clock: Clock,
    signals: SlotSignals,
    state: Mutex<State>,
    ecc: Arc<Backdoor>,
    directory: Arc<Backdoor>,
}

fn unimp(what: &str, off: u64, value: Option<u64>) {
    match value {
        Some(v) => crate::dlog!(LogModule::Hub, "Hub MD: write of {v:#x} to {what} {off:#x}"),
        None => crate::dlog!(LogModule::Hub, "Hub MD: read of {what} {off:#x}"),
    }
}

impl Md {
    pub fn new(clock: Clock, signals: SlotSignals, banks_mb: [u32; BANKS]) -> Self {
        Self {
            clock,
            signals,
            state: Mutex::new(State::new(None)),
            ecc: Arc::new(Backdoor::new(banks_mb)),
            directory: Arc::new(Backdoor::new(banks_mb)),
        }
    }

    /// The back door onto the node's ECC bytes.
    pub fn ecc_bus(&self) -> Arc<dyn BusDevice> {
        self.ecc.clone()
    }

    /// The back door onto the node's directory entries.
    pub fn directory_bus(&self) -> Arc<dyn BusDevice> {
        self.directory.clone()
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
            reg::MD_MEMORY_CONFIG::OFFSET => s.config,
            reg::MD_MLAN_CTL::OFFSET => self.mlan_value(&s),
            reg::MD_PERF_CNT0::OFFSET => s.perf_cnt0,
            _ => return None,
        })
    }

    pub fn read(&self, off: u64) -> u64 {
        match off {
            reg::MD_DIR_ERROR::OFFSET..=reg::MD_MISC_ERROR_CLR::OFFSET if off % 8 == 0 => 0,
            reg::MD_SLOTID_USTAT::OFFSET => self.slot_id(),
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
            reg::MD_MEMORY_CONFIG::OFFSET => self.state.lock().config = value & CONFIG_MASK,
            reg::MD_DIR_ERROR::OFFSET..=reg::MD_MISC_ERROR_CLR::OFFSET if off % 8 == 0 => {
                unimp("error register (nothing latches an error to clear)", off, Some(value))
            }
            reg::MD_MEM_DIMM_INIT::OFFSET | reg::MD_DIR_DIMM_INIT::OFFSET => {
                unimp("DIMM mode initialisation register (write-only, no functional signature)", off, Some(value))
            }
            reg::MD_MLAN_CTL::OFFSET => {
                // Programs the timing and starts a slot; the answer is reported only
                // from the slot's deadline. A write cannot set done or the data.
                let mut s = self.state.lock();
                s.mlan = value & MLAN_WRITABLE;
                let (pulse, sample) = w1::slot::fields(value);
                s.level = match s.bus.as_mut() {
                    Some(bus) => w1::slot::slot(bus, pulse, sample),
                    None => pulse >= w1::slot::RESET_PULSE || pulse < w1::slot::ONE_PULSE,
                };
                s.deadline = Some((self.clock)().saturating_add(MLAN_SLOT_NS));
            }
            reg::MD_PERF_CNT0::OFFSET => self.state.lock().perf_cnt0 = value & PERF_MASK,
            reg::MD_LED0::OFFSET | reg::MD_LED1::OFFSET | reg::MD_UREG1_0::OFFSET | reg::MD_UREG1_4::OFFSET => {
                let which = u8::from(matches!(off, reg::MD_LED1::OFFSET | reg::MD_UREG1_4::OFFSET));
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

#[cfg(test)]
mod tests;
