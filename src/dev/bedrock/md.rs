//! Bedrock's memory and directory controller: bank configuration, memory self-test, and the directory and ECC views.
//!
//! Memory accesses reach the banks through the node's Physical without this controller's lock; a configuration store
//! that moves or resizes a bank asks the chip for a remap. Register words, directory, protection and ECC check bytes
//! live in sparse maps. CPU coherence and protection enforcement are absent; unsupported writes are counted and ignored.
//!
//! # What is approximated here
//!
//! Memory never has an error. The ECC view keeps the check byte software writes and reads it back while the data is
//! unchanged, then reads the byte the data computes; directory entries are stored with computed check bits unless the
//! directory configuration turns that off. Nothing compares check bits with data, so no access corrects a word or latches
//! an error, the error registers read zero and the scrub window answers reads as the ECC view and refuses writes. PROM
//! 6.210's memory test and IRIX 6.5.30 on the Fuel never read memory whose check bits disagree with its data.
//!
//! Ported from hw/misc/sgi-bedrock-md.c in jamesbraid/qemu-sgi-origami.

use super::mem::{bank_bytes, bank_of, layout_differs, Memory, RamExtent, BANK_SPAN};
use super::{Bedrock, Effect};
use crate::traits::*;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use super::regs::md as reg;
/// Reset sizes: 12 entries in each header field and 24 in each data field.
const REPLY_QUEUE_SIZES: u64 = reg::OUTGOING_REPLY_QUEUE_SIZE::RESET;
/// Reset sizes: four requests for each of the three queues.
const REQUEST_QUEUE_SIZES: u64 = reg::OUTGOING_REQUEST_QUEUE_SIZE::RESET;

use crate::dev::widths::{self, access_widths};
const CONFIG_MASK: u64 = field_masks!(
    reg::MEMORY_CONFIG; RCD_RP_CONFIG, TIMING_HIGH, TIMING_LOW, IGNORE_CPU_PROTECTION, IGNORE_IO_PROTECTION,
    IO_PROTECTION_ENABLE, CELL_PROTECTION_ENABLE, BANK_SELECT, POS3_SIZE, POS3_WIDTH, POS3_ENABLES, POS2_SIZE,
    POS2_WIDTH, POS2_ENABLES, POS1_SIZE, POS1_WIDTH, POS1_ENABLES, POS0_SIZE, POS0_WIDTH, POS0_ENABLES
);
const ECC_COLUMNS: [u8; 72] = [
    0x13, 0x23, 0x43, 0x83, 0x2f, 0xf1, 0x0d, 0x07, 0xd0, 0x70, 0x4f, 0xf8, 0x61, 0x62, 0x64, 0x68,
    0x1c, 0x2c, 0x4c, 0x8c, 0x15, 0x25, 0x45, 0x85, 0x19, 0x29, 0x49, 0x89, 0x1a, 0x2a, 0x4a, 0x8a,
    0x51, 0x52, 0x54, 0x58, 0x91, 0x92, 0x94, 0x98, 0xa1, 0xa2, 0xa4, 0xa8, 0x31, 0x32, 0x34, 0x38,
    0x16, 0x26, 0x46, 0x86, 0x1f, 0xf2, 0x0b, 0x0e, 0xb0, 0xe0, 0x8f, 0xf4, 0xc1, 0xc2, 0xc4, 0xc8,
    1, 2, 4, 8, 16, 32, 64, 128,
];
pub fn ecc_compute(mut value: u64) -> u8 {
    let mut code = 0;
    while value != 0 {
        code ^= ECC_COLUMNS[value.trailing_zeros() as usize];
        value &= value - 1;
    }
    code
}

struct State {
    cell: u64,
    config: u64,
    refresh: u64,
    directory_config: u64,
    ecc_config: u64,
    storage: HashMap<u64, u64>,
    directory: HashMap<u64, u64>,
    counters: HashMap<u64, u64>,
    protection: HashMap<u64, u64>,
    compat: HashMap<u64, u64>,
    /// Check bytes software wrote, by doubleword of the dense walk, each
    /// with the data the doubleword held when it was written.
    ecc: HashMap<u64, (u8, u64)>,
    admitted: [u16; 2],
    clock_ns: u64,
    clock_anchor: Option<u64>,
    selftest_deadline: Option<u64>,
    selftest_result: bool,
    selftest_status: u64,
    selftest_error: u64,
}
impl Default for State {
    fn default() -> Self {
        Self {
            cell: reg::CURRENT_CELL::RESET,
            config: 0,
            refresh: 0,
            directory_config: 0,
            ecc_config: 0,
            storage: HashMap::new(),
            directory: HashMap::new(),
            counters: HashMap::new(),
            protection: HashMap::new(),
            compat: HashMap::new(),
            ecc: HashMap::new(),
            admitted: [0; 2],
            clock_ns: 0,
            clock_anchor: None,
            selftest_deadline: None,
            selftest_result: false,
            selftest_status: 0,
            selftest_error: 0,
        }
    }
}

pub(super) struct Md {
    memory: Arc<Memory>,
    clock: crate::dev::heart::Clock,
    // RAM uses the banks directly. The registers and the diagnostic views
    // take the state lock.
    state: Mutex<State>,
    /// The cell register, published for the checks every register access
    /// and the board's Hub-special windows make. Written under `state`.
    cell: AtomicU64,
    unsupported: AtomicU64,
}
#[derive(Clone, Copy)]
pub(super) enum View {
    Registers,
    Scrub,
    Directory,
    Ecc,
}
/// One of MD's diagnostic views as a bus device. Weak: Physical, which the
/// chip owns, may hold it.
pub(super) struct MdBus {
    chip: Weak<Bedrock>,
    view: View,
}

impl Md {
    /// A controller over `memory`, with `clock` timing its self-test.
    pub(super) fn new(memory: Memory, clock: crate::dev::heart::Clock) -> Self {
        Self {
            memory: Arc::new(memory),
            clock,
            state: Mutex::new(State::default()),
            cell: AtomicU64::new(State::default().cell),
            unsupported: AtomicU64::new(0),
        }
    }
    /// Where fitted RAM can answer, for the processor bus's cached view.
    pub(super) fn ram_extent(&self) -> RamExtent {
        self.memory.extent()
    }
    pub(super) fn state_locked(&self) -> bool {
        self.state.is_locked()
    }
    pub(super) fn physical(&self) -> &Arc<crate::physical::Physical> {
        self.memory.physical()
    }
    pub(super) fn memory(&self) -> &Memory {
        &self.memory
    }
    /// The mappings the configuration places now, for the remap.
    pub(super) fn mappings(&self) -> Arc<[crate::physical::BankMapping]> {
        self.memory.mappings(self.state.lock().config)
    }
    pub(super) fn view(&self, chip: Weak<Bedrock>, view: View) -> Arc<dyn BusDevice> {
        Arc::new(MdBus { chip, view })
    }
    fn advance(s: &mut State, elapsed: u64) {
        s.clock_ns = s.clock_ns.saturating_add(elapsed);
        if s.selftest_deadline.is_some_and(|end| s.clock_ns >= end) {
            s.selftest_deadline = None;
            let done = reg::SELF_TEST_STATUS::DONE::MASK;
            s.selftest_status = if s.selftest_result { done | reg::SELF_TEST_STATUS::PASSED::MASK } else { done };
            s.selftest_error = if s.selftest_result { 0 } else { reg::SELF_TEST_ERROR_ADDR::MEMORY_ERROR::MASK };
        }
    }
    fn catch_up(s: &mut State, now: u64) {
        if let Some(anchor) = s.clock_anchor {
            s.clock_anchor = Some(anchor.max(now));
            Self::advance(s, now.saturating_sub(anchor));
        }
    }
    pub(super) fn start_clock(&self) {
        let now = (self.clock)();
        self.state.lock().clock_anchor.get_or_insert(now);
    }
    pub(super) fn stop_clock(&self) {
        let now = (self.clock)();
        let mut s = self.state.lock();
        Self::catch_up(&mut s, now);
        s.clock_anchor = None;
    }
    /// Cell register bit 0, which gates the register window and the board's
    /// Hub-special windows. One atomic load.
    #[inline]
    pub(super) fn cell_permits(&self) -> bool {
        self.cell.load(Ordering::Relaxed) & (1 << reg::CURRENT_CELL::REGION_BITS::SHIFT) != 0
    }
    /// Count and log an access the model does not implement; the caller
    /// answers a bus error.
    fn refuse<T>(&self, what: &str, at: u64) -> Option<T> {
        self.unsupported.fetch_add(1, Ordering::Relaxed);
        crate::dlog!(crate::devlog::LogModule::Bedrock,
            "Bedrock MD refused {what} access at {at:#x}");
        None
    }
    /// An access through `view`, and what it changed beyond MD: a bank move asks for a
    /// remap, self-test for its zero fill.
    pub(super) fn access(&self, view: View, addr: u64, bytes: u64, write: Option<u64>) -> (Option<u64>, Effect) {
        if ![1, 2, 4, 8].contains(&bytes) || addr % bytes != 0 {
            return (self.refuse("access", addr), Effect::None);
        }
        // Sample outside state: even an injected clock must not call out under
        // the lock. Only register observations need the self-test timer.
        let now = matches!(view, View::Registers).then(|| (self.clock)());
        let mut effect = Effect::None;
        let mut s = self.state.lock();
        if let Some(now) = now {
            Self::catch_up(&mut s, now);
        }
        let result = match view {
            View::Registers => self.register(&mut s, addr, bytes, write, &mut effect),
            View::Directory => self.directory(&mut s, addr, bytes, write),
            View::Ecc => self.ecc(&mut s, addr, bytes, write),
            View::Scrub if write.is_none() => self.ecc(&mut s, addr, bytes, None),
            View::Scrub => self.refuse("scrub", addr),
        };
        self.cell.store(s.cell, Ordering::Relaxed);
        (result, effect)
    }
    fn register(&self, s: &mut State, at: u64, bytes: u64, write: Option<u64>, effect: &mut Effect) -> Option<u64> {
        if bytes != 8 {
            return self.refuse("register", at);
        }
        let old = match at {
            reg::CURRENT_CELL::OFFSET => s.cell,
            reg::MEMORY_CONFIG::OFFSET => s.config,
            reg::OUTGOING_REPLY_QUEUE_SIZE::OFFSET => REPLY_QUEUE_SIZES,
            reg::REFRESH_CONTROL::OFFSET => s.refresh,
            reg::SELF_TEST_ERROR_ADDR::OFFSET => s.selftest_error,
            reg::SELF_TEST_STATUS::OFFSET => s.selftest_status,
            reg::DIR_CONFIG::OFFSET => s.directory_config,
            reg::MB_ECC_CONFIG::OFFSET => s.ecc_config,
            reg::OUTGOING_REQUEST_QUEUE_SIZE::OFFSET => REQUEST_QUEUE_SIZES,
            // No error is ever latched: see the module notes.
            reg::MEM_ERROR::OFFSET | reg::MEM_ERROR_CLR::OFFSET | reg::DIR_ERROR::OFFSET | reg::DIR_ERROR_CLR::OFFSET => 0,
            _ if storage_mask(at).is_some() => *s.storage.get(&at).unwrap_or(&0),
            _ if idle(at) => 0,
            _ => return self.refuse("register", at),
        };
        if let Some(value) = write {
            match at {
                reg::SELF_TEST_CONTROL::OFFSET => return self.selftest(s, value, effect),
                reg::CURRENT_CELL::OFFSET => s.cell = value,
                reg::MEMORY_CONFIG::OFFSET => {
                    let old = std::mem::replace(&mut s.config, value & CONFIG_MASK);
                    if layout_differs(old, s.config) {
                        *effect = Effect::Remap;
                    }
                }
                reg::REFRESH_CONTROL::OFFSET => {
                    s.refresh = value & (reg::REFRESH_CONTROL::ENABLE::MASK | reg::REFRESH_CONTROL::THRESHOLD::MASK)
                }
                reg::DIR_CONFIG::OFFSET => {
                    s.directory_config = value & field_masks!(reg::DIR_CONFIG; IGNORE_DIR_ECC, PREMIUM)
                }
                reg::MB_ECC_CONFIG::OFFSET => s.ecc_config = value & reg::MB_ECC_CONFIG::IGNORE_DATA_ECC::MASK,
                _ if storage_mask(at).is_some() => {
                    s.storage.insert(at, value & storage_mask(at).unwrap());
                }
                _ => {
                    self.unsupported.fetch_add(1, Ordering::Relaxed);
                    crate::dlog!(crate::devlog::LogModule::Bedrock,
                        "Bedrock MD ignored register write at {at:#x}: {value:#x}");
                }
            }
        }
        Some(old)
    }
    fn directory(&self, s: &mut State, at: u64, bytes: u64, write: Option<u64>) -> Option<u64> {
        if bytes != 8 || at >= 1 << 30 {
            return self.refuse("directory", at);
        }
        let software = at / (BANK_SPAN / 8);
        let bank = bank_of(s.config, software);
        if at & 0x7c8 == reg::PROTECTION_ENTRY_OFFSET {
            let fitted = self.memory.fitted(bank);
            if fitted == 0 {
                return Some(0);
            }
            let within =
                ((((at % (BANK_SPAN / 8)) >> 11) << 14) | (((at >> 4) & 3) << 7)) % fitted;
            let key = ((self.memory.dense_base(bank) + within) >> 14) * 4 + ((within >> 7) & 3);
            let old = *s.protection.get(&key).unwrap_or(&0);
            if let Some(v) = write {
                s.protection.insert(key, v);
            }
            return Some(old);
        }
        let within = ((at % (BANK_SPAN / 8)) & !15) * 8;
        if within >= bank_bytes(s.config, bank) {
            return self.refuse("directory", at);
        }
        let mask = if at & 8 != 0 || s.directory_config & 1 != 0 {
            u64::MAX
        } else {
            u32::MAX as u64
        };
        let table;
        let key;
        let fitted = self.memory.fitted(bank);
        if within >= fitted {
            if !s.compat.contains_key(&at) && s.compat.len() >= 74 {
                return self.refuse("directory", at);
            }
            s.compat.entry(at).or_insert(0);
            table = &mut s.compat;
            key = at;
        } else {
            key = (self.memory.dense_base(bank) + within) / 128;
            table = if at & 8 != 0 {
                &mut s.counters
            } else {
                &mut s.directory
            };
        }
        let old = *table.get(&key).unwrap_or(&0) & mask;
        if let Some(value) = write {
            let value = if at & 8 == 0 && s.directory_config & 2 == 0 {
                directory_encode(value, s.directory_config & 1 != 0)
            } else {
                value
            };
            table.insert(key, value & mask);
        }
        Some(old)
    }
    // Approximation: report whether the selected fitting covers the programmed bank, after 1000 guest ns.
    // Only the zero write phase mutates memory.
    fn selftest(&self, s: &mut State, command: u64, effect: &mut Effect) -> Option<u64> {
        use reg::SELF_TEST_CONTROL as ctl;
        if command == ctl::STOP::MASK || command == ctl::RESET::MASK {
            s.selftest_deadline = None;
            s.selftest_status = 0;
            s.selftest_error = 0;
            return Some(0);
        }
        let selection = field_masks!(reg::SELF_TEST_CONTROL; BANK, DIMM);
        if command & field_masks!(reg::SELF_TEST_CONTROL; START, STOP, RESET) != ctl::START::MASK
            || command & !(ctl::START::MASK | selection) != 0
        {
            self.unsupported.fetch_add(1, Ordering::Relaxed);
            crate::dlog!(crate::devlog::LogModule::Bedrock,
                "Bedrock MD ignored self-test command {command:#x}");
            return Some(0);
        }
        let selected = (command & selection) >> ctl::BANK::SHIFT;
        let size = bank_bytes(s.config, bank_of(s.config, selected));
        let fits = size != 0 && size <= self.memory.fitted(selected);
        s.selftest_result = fits;
        s.selftest_status = 0;
        s.selftest_error = 0;
        s.selftest_deadline = Some(s.clock_ns.saturating_add(1000));
        if fits && s.storage.get(&reg::SELF_TEST_DATA::OFFSET).copied().unwrap_or(0) == reg::SELF_TEST_DATA::WRITE_CONTROL::MASK {
            *effect = Effect::Zero(selected as usize, size);
            // The zero fill writes every doubleword's data and check bits.
            let start = self.memory.dense_base(selected) / 8;
            s.ecc.retain(|&key, _| key < start || key >= start + size / 8);
        }
        Some(0)
    }
    fn ecc(&self, s: &mut State, at: u64, bytes: u64, write: Option<u64>) -> Option<u64> {
        if at >= 1 << 30 || (bytes != 1 && bytes != 2) {
            return self.refuse("ecc", at);
        }
        let software = at / (BANK_SPAN / 8);
        let within = (at % (BANK_SPAN / 8)) ^ if bytes == 1 { 7 } else { 6 };
        let address = software * BANK_SPAN + within * 8;
        let Some((bank, offset)) = self.memory.decode(s.config, address, bytes * 8) else {
            if bytes == 2 && [0x04000006, 0x08000006].contains(&at) {
                let i = usize::from(at == 0x08000006);
                let old = s.admitted[i];
                if let Some(value) = write {
                    s.admitted[i] = value as u16;
                }
                return Some(old as u64);
            }
            return self.refuse("ecc", at);
        };
        let mut old = 0;
        for i in 0..bytes {
            let raw = offset + i * 8;
            let key = (self.memory.dense_base(bank as u64) + raw) / 8;
            let data = self.memory.raw_read(bank, raw);
            let computed = ecc_compute(data);
            let stored = s.ecc.get(&key).filter(|&&(_, then)| then == data).map_or(computed, |&(code, _)| code);
            old |= (stored as u64) << (i * 8);
            if let Some(value) = write {
                let code = (value >> (i * 8)) as u8;
                if code == computed {
                    s.ecc.remove(&key);
                } else {
                    s.ecc.insert(key, (code, data));
                }
            }
        }
        Some(old)
    }
}
impl Resettable for Md {
    /// The chip's reset remaps once every unit is back.
    fn power_on(&self) {
        let mut s = self.state.lock();
        *s = State::default();
        self.cell.store(s.cell, Ordering::Relaxed);
    }
}

impl MdBus {
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        match self.chip.upgrade().and_then(|chip| chip.md.access(self.view, a, size as u64, value).0) {
            Some(data) => BusRead64::ok(data),
            None => BusRead64::err(),
        }
    }
}
impl BusDevice for MdBus {
    /// Bedrock's ECC view alone sees the whole source register of an uncached SH; other views take the low bits.
    fn write_narrow_from(&self, addr: u64, size: usize, gpr: u64) -> u32 {
        if size == 2 && matches!(self.view, View::Ecc) {
            // PROM 6.210's memory test stores each check word with `sh` from a register whose low 16 bits are zero; the
            // shift below matches its four observed lanes, not a known hardware wiring (unknown for the R14000).
            let memory_address = (addr ^ 6).wrapping_mul(8);
            return self.write16(addr, (gpr >> (memory_address.wrapping_neg() & 63)) as u16);
        }
        if size == 1 { self.write8(addr, gpr as u8) } else { self.write16(addr, gpr as u16) }
    }

    access_widths!();
}
fn storage_mask(at: u64) -> Option<u64> {
    Some(match at {
        reg::ARBITRATION_CONTROL::OFFSET => field_masks!(reg::ARBITRATION_CONTROL; WRITE_GUARANTEE, REPLY_GUARANTEE),
        reg::MIGRATION_CONFIG::OFFSET => field_masks!(reg::MIGRATION_CONFIG; ENABLE, NODE_MASK, INTERVAL),
        reg::PERF_SELECT0::OFFSET => field_masks!(reg::PERF_SELECT0; INTERRUPT, COMMAND, CHANNEL, SOURCE, ACTIVITY, MODE),
        reg::PERF_SELECT1::OFFSET => field_masks!(reg::PERF_SELECT1; INTERRUPT, COMMAND, CHANNEL, SOURCE, ACTIVITY, MODE),
        reg::PERF_COUNTER0::OFFSET => reg::PERF_COUNTER0::COUNT::MASK,
        reg::PERF_COUNTER1::OFFSET => reg::PERF_COUNTER1::COUNT::MASK,
        reg::JUNK_BUS_TIMING::OFFSET => field_masks!(
            reg::JUNK_BUS_TIMING; SYNERGY_ENABLE, SYNERGY_SETUP_HOLD, UART_ENABLE, UART_SETUP_HOLD, FLASH_ENABLE,
            FLASH_SETUP_HOLD
        ),
        reg::SELF_TEST_DATA::OFFSET => field_masks!(
            reg::SELF_TEST_DATA; WRITE_CONTROL, CYCLE_CONTROL, BYTE_CONTROL, NIBBLE_CONTROL, INITIAL_PATTERN
        ),
        reg::ISSUE_DEBUG_SELECT::OFFSET => reg::ISSUE_DEBUG_SELECT::SELECTION::MASK,
        reg::MIG_DIFF_THRESH::OFFSET => {
            field_masks!(reg::MIG_DIFF_THRESH; VALID, SATURATION_ACTION, THRESHOLD_ACTION, THRESHOLD)
        }
        reg::MIG_VALUE_THRESH::OFFSET => field_masks!(
            reg::MIG_VALUE_THRESH; VALID, LOCAL_INTERRUPT_ENABLE, SATURATION_ACTION, THRESHOLD_ACTION, THRESHOLD
        ),
        reg::DIR_DEBUG_SELECT::OFFSET => reg::DIR_DEBUG_SELECT::SELECTION::MASK,
        reg::MEM_DEBUG_SELECT::OFFSET => reg::MEM_DEBUG_SELECT::SELECTION::MASK,
        _ => return None,
    })
}
fn idle(at: u64) -> bool {
    matches!(
        at,
        reg::FETCHOP_CACHE_STATUS0::OFFSET
            | reg::FETCHOP_CACHE_STATUS1::OFFSET
            | reg::MISC_ERROR0::OFFSET
            | reg::MISC_ERROR1::OFFSET
            | reg::MISC_ERROR1_CLR::OFFSET
            | reg::LED0::OFFSET
            | reg::LED1::OFFSET
            | reg::LED2::OFFSET
            | reg::LED3::OFFSET
            | reg::SELF_TEST_CONTROL::OFFSET
            | reg::SELF_TEST_ERROR_ADDR::OFFSET
            | reg::SELF_TEST_STATUS::OFFSET
            | reg::PROTOCOL_ERROR::OFFSET
            | reg::PROTOCOL_ERROR_CLR::OFFSET
            | reg::MIG_CANDIDATE::OFFSET
            | reg::MIG_CANDIDATE_CLR::OFFSET
            | reg::SELFTEST_DIR_ERROR_DATA::OFFSET
            | reg::SELFTEST_MEM_ERROR_DATA0::OFFSET
            | reg::SELFTEST_MEM_ERROR_DATA1::OFFSET
            | reg::SELFTEST_MEM_ERROR_DATA2::OFFSET
            | reg::SELFTEST_MEM_ERROR_DATA3::OFFSET
    )
}

#[cfg(test)]
mod tests {
    // Cases follow tests/qtest/sn1-bedrock-md-test.c in jamesbraid/qemu-sgi-origami: memory-banks, reset, queue-sizes,
    // writable-masks, selftest-phases, fetchop-operations, ecc-carrier and directory-storage.
    use super::*;
    use super::super::BedrockConfig;

    /// The controller inside a chip of its own, with the views the node hands
    /// out.
    struct TestMd(Arc<Bedrock>);
    impl std::ops::Deref for TestMd {
        type Target = Md;
        fn deref(&self) -> &Md {
            &self.0.md
        }
    }
    impl TestMd {
        fn new(banks: &[u32]) -> Result<Arc<Self>, String> {
            Self::with_clock(banks, Arc::new(|| 0))
        }
        fn with_clock(banks: &[u32], clock: crate::dev::heart::Clock) -> Result<Arc<Self>, String> {
            Self::with_world(banks, clock, Arc::new(crate::cpu::smp::World::new()))
        }
        /// A chip whose remap parks the processors of `world`.
        fn with_world(banks: &[u32], clock: crate::dev::heart::Clock, world: Arc<crate::cpu::smp::World>) -> Result<Arc<Self>, String> {
            let chip = Bedrock::new(BedrockConfig { banks_mib: banks, ..super::super::test_config(clock) })?;
            chip.decode.set_world(world);
            Ok(Arc::new(Self(chip)))
        }
        /// MD's registers, through the chip's register window, which acts on
        /// what a store changes.
        fn md_bus(&self) -> Arc<dyn BusDevice> {
            struct Md(Arc<dyn BusDevice>);
            impl BusDevice for Md {
                fn read64(&self, a: u64) -> BusRead64 {
                    self.0.read64(super::super::regs::md::BASE + a)
                }
                fn write64(&self, a: u64, v: u64) -> u32 {
                    self.0.write64(super::super::regs::md::BASE + a, v)
                }
            }
            Arc::new(Md(self.0.registers()))
        }
        /// Uncached memory: the node's Physical.
        fn uncached_view(&self) -> Arc<dyn BusDevice> {
            self.0.decode.physical().clone()
        }
        fn fetchop_bus(&self) -> Arc<dyn BusDevice> {
            Arc::new(crate::sn_nodes::FetchOp(self.0.decode.physical().clone()))
        }
        /// Memory by number, as inbound DMA reaches it.
        fn dma_view(&self) -> Arc<dyn BusDevice> {
            self.0.decode.clone()
        }
        /// Whether each PI has bank 1 bit 61 (a correctable memory error)
        /// or error-summary bit 24 (an uncorrectable one) set.
        fn reported(&self) -> [(bool, bool); 2] {
            let regs = self.0.registers();
            std::array::from_fn(|n| {
                let base = n as u64 * super::super::regs::PI1_BASE;
                let pending = regs.read64(base + super::super::regs::pi::INT_PEND1::OFFSET).data;
                let summary = regs.read64(base + super::super::regs::pi::ERR_INT_PEND::OFFSET).data;
                (pending & (1 << 61) != 0, summary & (1 << 24) != 0)
            })
        }
    }
    impl Md {
        /// Hold the lock, for tests of the paths that must not take it.
        pub(in crate::dev::bedrock) fn hold(&self) -> impl Sized + '_ {
            self.state.lock()
        }
        /// Explicit elapsed time, for tests that drive the self-test by hand.
        fn advance_ns(&self, elapsed: u64) {
            Self::advance(&mut self.state.lock(), elapsed);
        }
        fn unsupported(&self) -> u64 {
            self.unsupported.load(Ordering::Relaxed)
        }
    }
    const FUEL_BANKS: [u32; 2] = [256, 256];
    const FITTED_BANK: u64 = 256 << 20;
    fn fuel() -> Arc<TestMd> {
        TestMd::new(&FUEL_BANKS).unwrap()
    }
    fn configured() -> Arc<TestMd> {
        let md = fuel();
        assert_eq!(md.md_bus().write64(8, 0x33), BUS_OK);
        md
    }
    /// The Tezro's default 4 GiB: two 1 GiB banks and four of 512 MiB.
    const TEZRO_BANKS: [u32; 6] = [1024, 1024, 512, 512, 512, 512];
    /// Banks 0 and 1 at size code 5 (1 GiB), the other four at code 4 (512 MiB).
    const TEZRO_CONFIG: u64 = 0x53 | (0x43 << 8) | (0x43 << 16);
    fn tezro() -> Arc<TestMd> {
        let md = TestMd::new(&TEZRO_BANKS).unwrap();
        assert_eq!(md.md_bus().write64(reg::MEMORY_CONFIG::OFFSET, TEZRO_CONFIG), BUS_OK);
        md
    }

    /// Cached, uncached and DMA memory take no MD lock; an address no enabled bank holds is a bus error.
    #[test]
    fn cached_uncached_and_dma_memory_do_not_take_the_md_locks() {
        let md = configured();
        let dma = md.dma_view();
        let uncached = md.uncached_view();
        let _held = md.hold();
        assert_eq!(md.physical().write64(0x1000, 0x1122334455667788), BUS_OK);
        assert_eq!(md.physical().read64(0x1000).data, 0x1122334455667788);
        assert_eq!(dma.read64(0x1000).data, 0x1122334455667788);
        assert_eq!(dma.write32(0x1004, 0xabcdef01), BUS_OK);
        assert_eq!(md.physical().read64(0x1000).data, 0x11223344abcdef01);
        assert_eq!(uncached.read32(0x1004).data, 0xabcdef01);
        assert_eq!(uncached.write8(0x1000, 0x99), BUS_OK);
        assert_eq!(uncached.write16(0x1002, 0x7766), BUS_OK);
        assert_eq!(md.physical().read64(0x1000).data, 0x99227766abcdef01);
        assert_eq!(uncached.read64(BANK_SPAN + 0x1000).data, 0);
        for view in [&dma, &uncached] {
            assert_eq!(view.read64(FITTED_BANK).status, BUS_ERR);
            assert_eq!(view.write64(2 * BANK_SPAN, 1), BUS_ERR);
        }
    }

    /// A configuration store that moves a bank flushes every processor's compiled code;
    /// one that moves nothing does not. Self-test's zero fill counts in the bank's page counters.
    #[cfg(feature = "jitv2")]
    #[test]
    fn a_store_that_moves_banks_asks_for_a_flush() {
        use crate::cpu::smp::{ParkSlot, World, REQUEST_JIT_FLUSH};
        let word = AtomicU64::new(0);
        let processor = Arc::new(ParkSlot::new(0, &word));
        let world = Arc::new(World::new());
        world.set_slots(vec![processor.clone()]);
        let md = TestMd::with_world(&FUEL_BANKS, Arc::new(|| 0), world).unwrap();
        md.md_bus().write64(reg::MEMORY_CONFIG::OFFSET, 0x33);
        assert!(processor.release(REQUEST_JIT_FLUSH), "mapping the banks asks for a flush");
        md.md_bus().write64(reg::MEMORY_CONFIG::OFFSET, 0x33 | (1 << 32));
        assert!(processor.release(REQUEST_JIT_FLUSH), "the swap asks for a flush");
        md.md_bus().write64(reg::MEMORY_CONFIG::OFFSET, 0x33 | (1 << 32));
        assert!(!processor.release(REQUEST_JIT_FLUSH), "a write that moves nothing does not");
        let counter = || unsafe { (*md.physical().banks()[0].gen_ptr(0)).load(Ordering::Relaxed) };
        md.md_bus().write64(reg::MEMORY_CONFIG::OFFSET, 0x13);
        md.md_bus().write64(0x20038, 0x80);
        let before = counter();
        md.md_bus().write64(reg::SELF_TEST_CONTROL::OFFSET, 1);
        assert!(counter() > before, "the zero fill invalidates the bank's code pages");
        let _ = processor.release(REQUEST_JIT_FLUSH);
        md.0.power_on();
        assert!(processor.release(REQUEST_JIT_FLUSH), "a reset that moves the banks asks for a flush");
    }

    /// Directory and ECC state is keyed per bank, so banks of different sizes never share an entry.
    #[test]
    fn directory_entries_of_different_sized_banks_stay_apart() {
        let md = TestMd::new(&[256, 256, 64, 64]).unwrap();
        md.md_bus().write64(reg::MEMORY_CONFIG::OFFSET, 0x33 | (0x13 << 8));
        md.md_bus().write64(reg::DIR_CONFIG::OFFSET, 2);
        let dir = md.0.directory_bus();
        // Software bank n's carriers start at n * 128 MiB.
        let at = |bank: u64, offset: u64| bank * (BANK_SPAN / 8) + offset;
        for bank in 0..4 {
            assert_eq!(dir.write64(at(bank, 0x20), 0x1000 + bank), BUS_OK, "bank {bank}");
        }
        for bank in 0..4 {
            assert_eq!(dir.read64(at(bank, 0x20)).data, 0x1000 + bank, "bank {bank}");
        }
        // A 64 MiB bank's directory ends at its size code's reach.
        assert_eq!(dir.read64(at(2, 0x20) + (64 << 20) / 8).status, BUS_ERR);
        assert_eq!(dir.read64(at(4, 0x20)).status, BUS_ERR, "no bank 4");
    }

    /// The board names what it fits; a bank is a whole number of direct-window regions up to the largest size code.
    #[test]
    fn a_board_may_only_fit_banks_the_controller_can_decode() {
        assert!(TestMd::new(&FUEL_BANKS).is_ok());
        assert!(TestMd::new(&TEZRO_BANKS).is_ok());
        assert!(TestMd::new(&[1024; 8]).is_ok());
        for banks in [&[][..], &[0, 256], &[2048], &[100], &[64; 9]] {
            assert!(TestMd::new(banks).is_err(), "{banks:?}");
        }
        // A byte lane holds one size code for both its banks.
        for banks in [&[1024, 512][..], &[512, 512, 1024, 512], &[1024, 1024, 512, 512, 512, 1024]] {
            assert!(TestMd::new(banks).err().is_some_and(|e| e.contains("share one size")), "{banks:?}");
        }
        for banks in [&[512][..], &[1024, 1024, 512], &[512, 512, 1024, 1024]] {
            assert!(TestMd::new(banks).is_ok(), "{banks:?}");
        }
    }

    /// The swizzle permutes slots within groups of four, so a slot reaches as far as its group's largest bank.
    #[test]
    fn the_ram_extent_reaches_as_far_as_the_largest_bank_of_a_slot_group() {
        let extent = RamExtent::new(&FUEL_BANKS);
        for slot in 0..4 {
            assert!(extent.contains(slot * BANK_SPAN + FITTED_BANK - 8), "slot {slot}");
            assert!(!extent.contains(slot * BANK_SPAN + FITTED_BANK), "slot {slot}");
        }
        assert!(!extent.contains(4 * BANK_SPAN));
        let mixed = RamExtent::new(&[64, 512]);
        assert!(mixed.contains(3 * BANK_SPAN + (512 << 20) - 8), "slot 3 may hold the 512 MiB bank");
        assert!(!mixed.contains(512 << 20));
        assert_eq!(extent, fuel().ram_extent());
        let tezro = RamExtent::new(&TEZRO_BANKS);
        assert!(tezro.contains(3 * BANK_SPAN + BANK_SPAN - 8), "slot 3 may hold a 1 GiB bank");
        assert!(tezro.contains(5 * BANK_SPAN + (512 << 20) - 8));
        assert!(!tezro.contains(5 * BANK_SPAN + (512 << 20)));
        assert!(!tezro.contains(8 * BANK_SPAN));
        assert_eq!(tezro, TestMd::new(&TEZRO_BANKS).unwrap().ram_extent());
    }

    /// Each Tezro bank answers in its own 1 GiB slot at its own size, banks 4 and 5 above 4 GiB inside the 16 GiB window.
    #[test]
    fn every_tezro_bank_answers_at_its_own_size_including_above_4_gib() {
        use crate::ppmem::MappedMemory;
        let md = tezro();
        let bus = md.physical();
        let sizes = [1024u64, 1024, 512, 512, 512, 512];
        for (slot, mib) in sizes.iter().enumerate() {
            let base = slot as u64 * BANK_SPAN;
            let size = mib << 20;
            assert_eq!(bus.write64(base, 0x1111 * (slot as u64 + 1)), BUS_OK, "slot {slot}");
            assert_eq!(bus.write64(base + size - 8, 0xe0 + slot as u64), BUS_OK, "slot {slot} top");
            if size < BANK_SPAN {
                assert_eq!(bus.read64(base + size).status, BUS_ERR, "slot {slot} past its bank");
            }
        }
        for slot in 0..6u64 {
            assert_eq!(bus.read64(slot * BANK_SPAN).data, 0x1111 * (slot + 1), "slot {slot}");
            let top = (sizes[slot as usize] << 20) - 8;
            assert_eq!(bus.read64(slot * BANK_SPAN + top).data, 0xe0 + slot, "slot {slot} top");
        }
        for slot in [6, 7] {
            assert_eq!(bus.read64(slot * BANK_SPAN).status, BUS_ERR, "slot {slot} holds no bank");
        }
        assert_eq!(bus.banks()[4].read64(0).data, 0x1111 * 5, "bank 4 holds what slot 4 wrote");
        assert_eq!(bus.banks()[5].read64(0).data, 0x1111 * 6);
        // The 16 GiB window maps every bank, in 256 MiB regions.
        let space = bus.ppmem_space().expect("a window");
        assert_eq!(space.window_size(), crate::ppmem::LARGE_WINDOW_SIZE);
        assert_eq!(space.mapped_bitmap(), 0xf | (0xf << 4) | (0x3 << 8) | (0x3 << 12) | (0x3 << 16) | (0x3 << 20));
        assert_eq!(md.dma_view().read64(5 * BANK_SPAN).data, 0x1111 * 6, "DMA reaches it too");
        assert_eq!(md.uncached_view().read64(4 * BANK_SPAN).data, 0x1111 * 5, "and uncached");
        // Selector 1 swaps the two banks of each pair, so the 512 MiB banks 4 and 5 trade places above 4 GiB.
        assert_eq!(bus.write64(4 * BANK_SPAN + 0x28, 0xfedc_ba98_7654_3210), BUS_OK);
        assert_eq!(bus.read64(5 * BANK_SPAN + 0x28).data, 0);
        assert_eq!(md.md_bus().write64(reg::MEMORY_CONFIG::OFFSET, TEZRO_CONFIG | (1 << 32)), BUS_OK);
        assert_eq!(bus.read64(5 * BANK_SPAN + 0x28).data, 0xfedc_ba98_7654_3210);
        assert_eq!(bus.read64(4 * BANK_SPAN + 0x28).data, 0);
        // A node whose banks all lie below 4 GiB, as the Fuel's do, keeps the 4 GiB window and 64 MiB regions.
        let fuel = configured();
        let space = fuel.physical().ppmem_space().expect("a window");
        assert_eq!(space.window_size(), crate::ppmem::WINDOW_SIZE);
        assert_eq!(space.mapped_bitmap(), 0xf | (0xf << 16));
    }

    /// A DMA block moves what its words would one at a time, crosses into the next bank, and stops at the first hole.
    #[test]
    fn a_dma_block_matches_word_accesses_and_stops_where_memory_ends() {
        let md = tezro();
        let dma = md.dma_view();
        let words: Vec<u64> = (0..16).map(|i| 0x0123_4567_89ab_cdef ^ (i * 0x1111_0000_0001)).collect();
        // Inside one run: the same words, read either way.
        assert_eq!(dma.write_block(0x2000, &words), BUS_OK);
        for (i, &w) in words.iter().enumerate() {
            assert_eq!(dma.read64(0x2000 + 8 * i as u64).data, w);
        }
        let mut back = vec![0u64; words.len()];
        assert_eq!(dma.read_block(0x2000, &mut back), BUS_OK);
        assert_eq!(back, words);
        // Across the end of bank 0's slot into bank 1.
        let edge = BANK_SPAN - 32;
        assert_eq!(dma.write_block(edge, &words), BUS_OK);
        assert_eq!(md.physical().banks()[0].read64(BANK_SPAN - 8).data, words[3]);
        assert_eq!(md.physical().banks()[1].read64(0).data, words[4]);
        assert_eq!(dma.read_block(edge, &mut back), BUS_OK);
        assert_eq!(back, words);
        // Past the top of bank 5 (512 MiB): the words below it landed.
        let end = 5 * BANK_SPAN + (512 << 20) - 16;
        assert_eq!(dma.write_block(end, &words[..4]), BUS_ERR);
        assert_eq!(dma.read64(end).data, words[0]);
        assert_eq!(dma.read64(end + 8).data, words[1]);
        let mut short = [0xaa; 4];
        assert_eq!(dma.read_block(end, &mut short), BUS_ERR);
        assert_eq!(short[..2], words[..2], "what came before the failure was read");
        assert_eq!(dma.read_block(6 * BANK_SPAN, &mut short), BUS_ERR);
        assert_eq!(dma.write_block(6 * BANK_SPAN, &short), BUS_ERR);
        // A bank the configuration disables answers nothing, as for a word.
        let fuel = configured();
        assert_eq!(fuel.dma_view().write_block(FITTED_BANK - 8, &words[..2]), BUS_ERR);
        assert_eq!(fuel.dma_view().read64(FITTED_BANK - 8).data, words[0]);
        assert_eq!(fuel.dma_view().write_block(2 * BANK_SPAN, &words[..1]), BUS_ERR);
    }

    /// Every page a block touches is invalidated for compiled code, once.
    #[cfg(feature = "jitv2")]
    #[test]
    fn a_dma_block_invalidates_each_page_it_touches() {
        let md = configured();
        let dma = md.physical().clone();
        let counter = |page: u64| unsafe { (*dma.gen_ptr(page * 4096)).load(Ordering::Relaxed) };
        let before: Vec<u64> = (1..6).map(counter).collect();
        // 2 pages and a bit, from the middle of page 1 to the start of page 3.
        assert_eq!(dma.write_block(4096 + 2048, &[0x55; 1032]), BUS_OK);
        let after: Vec<u64> = (1..6).map(counter).collect();
        assert_eq!(after.iter().zip(&before).map(|(a, b)| a - b).collect::<Vec<_>>(), [1, 1, 1, 0, 0]);
    }

    /// A word store through DMA or uncached invalidates its page for compiled code once, at every width; a hole invalidates nothing.
    #[cfg(feature = "jitv2")]
    #[test]
    fn a_word_store_invalidates_its_page_and_a_miss_does_not() {
        let md = configured();
        let (dma, uncached) = (md.dma_view(), md.uncached_view());
        let page = md.physical().gen_ptr(0x3000);
        let counter = || unsafe { (*page).load(Ordering::Relaxed) };
        for view in [&dma, &uncached] {
            let before = counter();
            assert_eq!([view.write8(0x3000, 1), view.write16(0x3002, 1), view.write32(0x3004, 1), view.write64(0x3008, 1)], [BUS_OK; 4]);
            assert_eq!(counter() - before, 4);
            let _ = view.read64(0x3008);
            assert_eq!(counter() - before, 4, "a read invalidates nothing");
        }
        let past = 2 * BANK_SPAN;
        assert_eq!((dma.write32(past, 1), uncached.write32(past, 1)), (BUS_ERR, BUS_ERR));
        assert_eq!((dma.read8(past).status, uncached.read8(past).status), (BUS_ERR, BUS_ERR));
    }

    /// A processor's SC above 4 GiB swaps through the DMA view in one step, leaving a failed compare alone.
    #[test]
    fn a_compare_and_swap_above_4_gib_is_one_step() {
        let md = tezro();
        let view = md.physical().clone();
        let at = 5 * BANK_SPAN + 0x40;
        assert_eq!(view.write64(at, 7), BUS_OK);
        #[cfg(feature = "jitv2")]
        let before = unsafe { (*view.gen_ptr(at)).load(Ordering::Relaxed) };
        assert_eq!(view.cas64(at, 6, 9), Some(false));
        assert_eq!(view.read64(at).data, 7);
        assert_eq!(view.cas64(at, 7, 9), Some(true));
        assert_eq!(view.read64(at).data, 9);
        assert_eq!(view.cas32(at + 4, 9, 3), Some(true));
        assert_eq!(view.read64(at).data, 3);
        #[cfg(feature = "jitv2")]
        assert!(unsafe { (*view.gen_ptr(at)).load(Ordering::Relaxed) } > before);
        assert_eq!(view.cas64(7 * BANK_SPAN, 0, 1), None, "no bank there");
    }

    /// A fitting that covers the bank zero-fills it whichever slot it is in; an unfitted bank fails.
    #[test]
    fn selftest_fills_a_tezro_bank_above_4_gib_and_fails_a_missing_bank() {
        let md = TestMd::new(&TEZRO_BANKS).unwrap();
        // Every pair at size code 1 (64 MiB): small enough to fill in a test.
        let regs = md.md_bus();
        regs.write64(reg::MEMORY_CONFIG::OFFSET, 0x13 | (0x13 << 8) | (0x13 << 16) | (0x13 << 24));
        let bus = md.physical();
        bus.write64(4 * BANK_SPAN + 0x40, 0x5555);
        regs.write64(reg::SELF_TEST_DATA::OFFSET, 0x80);
        regs.write64(reg::SELF_TEST_CONTROL::OFFSET, (4 << 4) | 1);
        md.advance_ns(1000);
        assert_eq!(regs.read64(reg::SELF_TEST_STATUS::OFFSET).data, 3, "bank 4 is fitted");
        assert_eq!(bus.read64(4 * BANK_SPAN + 0x40).data, 0, "and zeroed");
        regs.write64(reg::SELF_TEST_CONTROL::OFFSET, (6 << 4) | 1);
        md.advance_ns(1000);
        assert_eq!(regs.read64(reg::SELF_TEST_STATUS::OFFSET).data, 2, "bank 6 is not fitted");
    }

    /// A bank configuration write parks the other processors, another write does not; remapping unparked panicked a Tezro.
    #[test]
    fn a_bank_configuration_write_parks_the_other_processors() {
        use crate::cpu::smp::{set_current_cpu, ParkSlot, World, HOST_PARK_BIT, REQUEST_JIT_FLUSH};
        use std::sync::atomic::AtomicBool;
        let words: Vec<Arc<AtomicU64>> = (0..2).map(|_| Arc::new(AtomicU64::new(0))).collect();
        let slots: Vec<Arc<ParkSlot>> = words.iter().enumerate().map(|(cpu, w)| Arc::new(ParkSlot::new(cpu as u8, Arc::as_ptr(w)))).collect();
        let world = Arc::new(World::new());
        world.set_slots(slots.clone());
        let md = TestMd::with_world(&FUEL_BANKS, Arc::new(|| 0), world).unwrap();
        assert_eq!(md.md_bus().write64(reg::MEMORY_CONFIG::OFFSET, 0x33), BUS_OK);
        let space = md.physical().ppmem_space().expect("a window");
        let _mine = space.add_cached_view(&[]).unwrap();
        let theirs = space.add_cached_view(&[]).unwrap();
        let running = Arc::new(AtomicBool::new(true));
        let parks = Arc::new(AtomicU64::new(0));
        slots[1].set_active(true);
        let other = {
            let (md, slot, word, running, parks) = (md.clone(), slots[1].clone(), words[1].clone(), running.clone(), parks.clone());
            std::thread::spawn(move || {
                set_current_cpu(1);
                let space = md.physical().ppmem_space().unwrap();
                let page = crate::ppmem::map::granularity() as u64;
                while running.load(Ordering::Relaxed) {
                    if slot.release(REQUEST_JIT_FLUSH) {
                        continue;
                    }
                    if word.load(Ordering::Relaxed) & HOST_PARK_BIT != 0 {
                        slot.park(&running);
                        parks.fetch_add(1, Ordering::SeqCst);
                        continue;
                    }
                    for n in 0..32 {
                        space.hold_line_page(theirs, 0x10_0000 + n * page);
                    }
                    for n in 0..32 {
                        space.release_line_page(theirs, 0x10_0000 + n * page);
                    }
                }
                slot.set_active(false);
            })
        };
        let regs = md.md_bus();
        std::thread::scope(|s| {
            s.spawn(|| {
                set_current_cpu(0);
                for swap in [1u64, 0, 1, 0] {
                    let before = parks.load(Ordering::SeqCst);
                    assert_eq!(regs.write64(reg::MEMORY_CONFIG::OFFSET, 0x33 | swap << 32), BUS_OK);
                    while parks.load(Ordering::SeqCst) == before {
                        std::thread::yield_now();
                    }
                }
                // Once it runs again, other registers leave it running.
                let before = parks.load(Ordering::SeqCst);
                assert_eq!(regs.write64(reg::REFRESH_CONTROL::OFFSET, 0), BUS_OK);
                std::thread::sleep(std::time::Duration::from_millis(10));
                assert_eq!(parks.load(Ordering::SeqCst), before, "a refresh write moves no memory");
            });
        });
        running.store(false, Ordering::Relaxed);
        other.join().unwrap();
        assert_eq!(md.physical().read64(BANK_SPAN).status, BUS_OK);
    }

    /// Two nodes share one `World`, so a configuration write holds the other node's processor whichever thread makes it.
    #[test]
    fn a_bank_configuration_write_parks_the_other_nodes_processor() {
        use crate::cpu::smp::{set_current_cpu, ParkSlot, World, HOST_PARK_BIT, REQUEST_JIT_FLUSH};
        use crate::dev::bedrock::Population;
        use std::sync::atomic::AtomicBool;
        let chips: Vec<_> = (0..2)
            .map(|n| {
                let config = super::super::test_config(Arc::new(|| 0));
                let config = BedrockConfig { processors: Population::SINGLE.with_first_cpu(n), ..config };
                Bedrock::new(config).unwrap()
            })
            .collect();
        chips[0].connect_peer(&chips[1]);
        chips[1].connect_peer(&chips[0]);
        let words: Vec<Arc<AtomicU64>> = (0..2).map(|_| Arc::new(AtomicU64::new(0))).collect();
        let slots: Vec<Arc<ParkSlot>> = words.iter().enumerate().map(|(cpu, w)| Arc::new(ParkSlot::new(cpu as u8, Arc::as_ptr(w)))).collect();
        let world = Arc::new(World::new());
        world.set_slots(slots.clone());
        for chip in &chips {
            chip.decode.set_world(world.clone());
        }
        let running = Arc::new(AtomicBool::new(true));
        // How many times each processor stopped for a hold.
        let parks: Vec<Arc<AtomicU64>> = (0..2).map(|_| Arc::new(AtomicU64::new(0))).collect();
        let spinners: Vec<_> = (0..2usize)
            .map(|n| {
                slots[n].set_active(true);
                let (slot, word, running, parks) = (slots[n].clone(), words[n].clone(), running.clone(), parks[n].clone());
                std::thread::spawn(move || {
                    while running.load(Ordering::Relaxed) {
                        // A layout change's flush request, served as the run
                        // loop serves it, is not a hold.
                        if slot.release(REQUEST_JIT_FLUSH) {
                            continue;
                        }
                        if word.load(Ordering::Relaxed) & HOST_PARK_BIT != 0 {
                            slot.park(&running);
                            parks.fetch_add(1, Ordering::SeqCst);
                        }
                        std::thread::yield_now();
                    }
                    slot.set_active(false);
                })
            })
            .collect();
        for writer in 0..2usize {
            let (target, other) = (&chips[1 - writer], 1 - writer);
            let before: Vec<u64> = parks.iter().map(|p| p.load(Ordering::SeqCst)).collect();
            std::thread::scope(|s| {
                s.spawn(|| {
                    set_current_cpu(writer as u8);
                    let regs = target.registers();
                    assert_eq!(regs.write64(super::super::regs::md::BASE + reg::MEMORY_CONFIG::OFFSET, 0x33 | (writer as u64) << 32), BUS_OK);
                });
            });
            // The write returns once the world is released; the other
            // processor has parked and resumes.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while parks[other].load(Ordering::SeqCst) == before[other] && std::time::Instant::now() < deadline {
                std::thread::yield_now();
            }
            assert_eq!(parks[other].load(Ordering::SeqCst), before[other] + 1, "CPU {other} was not held for CPU {writer}'s write to the other node");
            assert_eq!(parks[writer].load(Ordering::SeqCst), before[writer], "the writer, CPU {writer}, held itself");
        }
        running.store(false, Ordering::Relaxed);
        for t in spinners {
            t.join().unwrap();
        }
        for chip in &chips {
            chip.decode.physical().release_devices();
        }
    }

    /// Guest density settings remap fitted RAM, and reset restores the default map.
    #[test]
    fn bank_configuration_remaps_memory_and_reset_restores_defaults() {
        let md = configured();
        let bus = md.physical();
        assert_eq!(bus.write64(0, 0x1122334455667788), BUS_OK);
        assert_eq!(bus.write64(BANK_SPAN, 0xabcdef), BUS_OK);
        assert_eq!(bus.read8(0).data, 0x11);
        assert_eq!(bus.read32(4).data, 0x55667788);
        assert_eq!(bus.read64(FITTED_BANK).status, BUS_ERR);
        assert_eq!(bus.read64(2 * BANK_SPAN).status, BUS_ERR);
        assert_eq!(md.md_bus().write64(8, 0x33 | 1 << 32), BUS_OK);
        assert_eq!(bus.read64(0).data, 0xabcdef);
        assert_eq!(bus.read64(BANK_SPAN).data, 0x1122334455667788);
        assert!(bus.mem_ptr(0).is_some());
        md.0.power_on();
        assert_eq!(bus.read64(0).status, BUS_ERR);
        bus.map_unmapped(0x1fc00000, 0x1fc01000);
        for (code, size) in [(1, 64 << 20), (2, 128 << 20), (3, FITTED_BANK),
            (4, FITTED_BANK), (5, FITTED_BANK)] {
            md.md_bus().write64(reg::MEMORY_CONFIG::OFFSET, 1 | (code << 4));
            assert_eq!(bus.read64(size - 8).status, BUS_OK);
            assert_eq!(bus.read64(size).status, BUS_ERR);
            assert_eq!(bus.read64(0x1fc00000).status, BUS_OK, "a non-RAM mapping survives");
        }
        md.md_bus().write64(reg::MEMORY_CONFIG::OFFSET, 0x11);
        assert_eq!(bus.read64(64 << 20).status, BUS_ERR, "the old prefix is removed on shrink");
        md.md_bus().write64(reg::MEMORY_CONFIG::OFFSET, 0x61);
        assert_eq!(bus.read64(0).status, BUS_ERR, "unsupported density does not map RAM");
        #[cfg(feature = "jitv2")]
        {
            let md = configured();
            let ram = md.physical();
            let first = ram.gen_ptr(0);
            let second = ram.gen_ptr(BANK_SPAN);
            assert!(!first.is_null() && !second.is_null());
            assert_ne!(first, second);
            let before = unsafe { (*first).load(Ordering::Relaxed) };
            ram.write64(0, 0x55);
            assert!(unsafe { (*first).load(Ordering::Relaxed) } > before);
            md.md_bus().write64(8, 0x33 | (1 << 32));
            // The window address stays fixed while its backing bank changes.
            assert_eq!(ram.gen_ptr(0), first);
            assert_eq!(ram.gen_ptr(BANK_SPAN), second);
            let bank0 = ram.banks()[0].gen_ptr(0);
            let bank1 = ram.banks()[1].gen_ptr(0);
            let before0 = unsafe { (*bank0).load(Ordering::Relaxed) };
            let before1 = unsafe { (*bank1).load(Ordering::Relaxed) };
            assert_eq!(ram.write64(0, 0xaa), BUS_OK);
            assert_eq!(unsafe { (*bank0).load(Ordering::Relaxed) }, before0);
            assert_eq!(unsafe { (*bank1).load(Ordering::Relaxed) }, before1 + 1);
            assert_eq!(unsafe { (*first).load(Ordering::Relaxed) }, before1 + 1);
        }
    }
    /// The error registers never latch an error: an injected check byte read past raises none.
    #[test]
    fn guest_registers_expose_ecc_configuration_and_latch_no_error() {
        {
            let md = configured();
            let r = md.md_bus();
            // These carrier offsets are idle; they do not select storage.
            for at in [0x20, 0x20010, 0x40020, 0x60020] {
                let before = md.unsupported();
                assert_eq!(r.write64(at, u64::MAX), BUS_OK, "{at:#x}");
                assert_eq!(r.read64(at).data, 0, "{at:#x}");
                assert_eq!(md.unsupported(), before + 1);
            }
            for (at, value) in [(0x60, 0x001818180c0c0c00), (0x40050, 0x04040400)] {
                let before = md.unsupported();
                assert_eq!(r.write64(at, 0), BUS_OK);
                assert_eq!(r.read64(at).data, value);
                assert_eq!(md.unsupported(), before + 1);
            }
            md.0.ecc_bus().write8(7, 3);
            assert_eq!(md.uncached_view().read64(0).status, BUS_OK);
            r.write64(0x40000, 2);
            md.0.directory_bus().write64(0, 3 << 16);
            r.write64(0x40000, 0);
            assert_eq!(md.physical().read64(0).status, BUS_OK);
            assert_eq!(md.dma_view().read64(0).status, BUS_OK);
            for at in [0x40010, 0x40018, 0x60010, 0x60018] {
                let before = md.unsupported();
                assert_eq!(r.write64(at, u64::MAX), BUS_OK);
                assert_eq!(r.read64(at).data, 0, "{at:#x}");
                assert_eq!(md.unsupported(), before + 1);
            }
            assert_eq!(md.reported(), [(false, false); 2]);
            // An unsupported self-test command leaves an in-flight command alone.
            r.write64(0x20030, 1);
            let before = md.unsupported();
            assert_eq!(r.write64(0x20030, 3), BUS_OK);
            assert_eq!(md.unsupported(), before + 1);
            md.advance_ns(1000);
            assert_eq!(r.read64(0x20048).data, 3);
            assert_eq!(r.write64(0x70000, 1), BUS_ERR);
            assert_eq!(r.read64(0x70000).status, BUS_ERR);
            assert_eq!(r.write32(0x60, 0), BUS_ERR);
            assert_eq!(r.write64(0x61, 0), BUS_ERR);
        }
        {
            let md = configured();
            let regs = md.md_bus();
            assert_eq!(regs.read64(0x60000).data, 0);
            assert_eq!(regs.write64(0x60000, u64::MAX), BUS_OK);
            assert_eq!(regs.read64(0x60000).data, 1);
            assert_eq!(regs.write64(0x60000, !1), BUS_OK);
            assert_eq!(regs.read64(0x60000).data, 0);
            assert_eq!(regs.write8(0x60000, 1), BUS_ERR);
            assert_eq!(regs.write64(0x60001, 1), BUS_ERR);
            regs.write64(0x60000, 1);
            md.0.power_on();
            assert_eq!(regs.read64(0x60000).data, 0);
            assert_eq!(regs.read64(8).data, 0);
        }
    }
    /// Self-test completion is visible only after its configured clock interval.
    #[test]
    fn selftest_waits_for_fitted_clock_cycles_and_zero_fills_ram() {
        {
            let now = Arc::new(AtomicU64::new(100));
            let time = now.clone();
            let md = TestMd::with_clock(&FUEL_BANKS, Arc::new(move || time.load(Ordering::Relaxed))).unwrap();
            let regs = md.md_bus();
            regs.write64(8, 0x33);
            md.start_clock();
            regs.write64(0x20030, 1);
            now.store(1099, Ordering::Relaxed);
            assert_eq!(regs.read64(0x20048).data, 0);
            now.store(1100, Ordering::Relaxed);
            assert_eq!(regs.read64(0x20048).data, 3);
            now.store(5000, Ordering::Relaxed);
            regs.write64(0x20030, 1);
            now.store(5500, Ordering::Relaxed);
            md.stop_clock();
            now.store(15000, Ordering::Relaxed);
            assert_eq!(regs.read64(0x20048).data, 0);
            md.start_clock();
            now.store(15499, Ordering::Relaxed);
            assert_eq!(regs.read64(0x20048).data, 0);
            now.store(15500, Ordering::Relaxed);
            assert_eq!(regs.read64(0x20048).data, 3);
            regs.write64(0x20030, 1);
            now.store(15400, Ordering::Relaxed);
            assert_eq!(regs.read64(0x20048).data, 0);
            now.store(16499, Ordering::Relaxed);
            assert_eq!(regs.read64(0x20048).data, 0);
            now.store(16500, Ordering::Relaxed);
            assert_eq!(regs.read64(0x20048).data, 3);
            regs.write64(0x20030, 1);
            regs.write64(0x20030, 2);
            now.store(20000, Ordering::Relaxed);
            assert_eq!(regs.read64(0x20048).data, 0);
            md.0.power_on();
            regs.write64(8, 0x33);
            md.start_clock();
            regs.write64(0x20030, 1);
            now.store(20999, Ordering::Relaxed);
            assert_eq!(regs.read64(0x20048).data, 0);
            now.store(21000, Ordering::Relaxed);
            assert_eq!(regs.read64(0x20048).data, 3);
        }
        {
            let md = configured();
            let r = md.md_bus();
            r.write64(0x20030, 0x21);
            md.advance_ns(1000);
            assert_eq!(r.read64(0x20048).data, 2);
            assert_eq!(r.read64(0x20040).data, 1 << 48);
            r.write64(0x20030, 1);
            r.write64(0x20030, 2);
            md.advance_ns(1000);
            assert_eq!(r.read64(0x20048).data, 0);
            assert_eq!(r.write64(0x20030, 3), BUS_OK);
        }
        {
            // The zero fill writes the check bits too, even where the data
            // was already zero.
            let md = configured();
            let r = md.md_bus();
            r.write64(8, 0x13);
            md.physical().write64(0, 0x1234);
            md.0.ecc_bus().write8(7, 1);
            md.0.ecc_bus().write8(15, 1);
            r.write64(0x20038, 0x80);
            r.write64(0x20030, 1);
            md.advance_ns(1000);
            assert_eq!(r.read64(0x20048).data, 3);
            assert_eq!(md.physical().read64(0).data, 0);
            assert_eq!(md.0.ecc_bus().read8(7).data, 0);
            assert_eq!(md.0.ecc_bus().read8(15).data, 0);
        }
    }
    /// Fetch-and-op registers preserve the big-endian lanes used by the processor.
    #[test]
    fn fetchop_preserves_big_endian_word_lanes() {
        let md = configured();
        let ram = md.physical();
        let f = md.fetchop_bus();
        ram.write64(0, 0xffffffff00000009);
        assert_eq!(f.read32(8).data, u32::MAX);
        assert_eq!(ram.read64(0).data, 9);
        assert_eq!(f.read32(12).data, 9);
        assert_eq!(ram.read64(0).data, 10);
        assert_eq!(f.write32(28, 3), BUS_OK);
        assert_eq!(ram.read64(0).data, 2);
        assert_eq!(f.write64(32, 8), BUS_OK);
        assert_eq!(ram.read64(0).data, 10);
        assert_eq!(f.read64(24).data, 10);
        assert_eq!(ram.read64(0).data, 0);
        assert_eq!(f.read64(32).status, BUS_ERR);
        assert_eq!(f.write64(40, 1), BUS_ERR);
        assert_eq!(f.read16(0).status, BUS_ERR);
    }
    /// The guest reads back check bytes it writes until the data changes; the scrub window reads as the ECC view.
    #[test]
    fn ecc_carriers_keep_written_check_bytes_until_the_data_changes() {
        {
            let md = configured();
            let ecc = md.0.ecc_bus();
            for carrier in 0..8 {
                let source = 0x11223344556677a5u64;
                assert_eq!(ecc.write8(carrier, source as u8), BUS_OK);
                assert_eq!(ecc.read8(carrier).data, source as u8, "carrier {carrier}");
            }
            assert_eq!(ecc.write16(4, 0x1234), BUS_OK);
            assert_eq!(ecc.read16(4).data, 0x1234);
            assert_eq!(ecc.write16(6, 0x2313), BUS_OK);
            assert_eq!(ecc.read16(6).data, 0x2313);
            assert_eq!(ecc.write16(0x04000006, 0xface), BUS_OK);
            assert_eq!(ecc.read16(0x04000006).data, 0xface);
        }
        {
            let md = configured();
            let (ecc, ram) = (md.0.ecc_bus(), md.uncached_view());
            assert_eq!(ecc.read8(7).data, ecc_compute(0), "nothing written: the data's own");
            ecc.write8(7, 3);
            assert_eq!(ecc.read8(7).data, 3);
            assert_eq!(md.0.ecc_scrub_bus().read8(7).data, 3);
            assert_eq!(ram.read64(0).data, 0, "memory reads raise no error");
            assert_eq!(ram.write8(7, 0x5a), BUS_OK);
            assert_eq!(ecc.read8(7).data, 0x82, "the new data's check byte");
            assert_eq!(ecc_compute(0x5a), 0x82);
            ecc.write8(7, 0x82);
            assert!(md.state.lock().ecc.is_empty(), "a computed byte is not kept");
            assert_eq!(md.0.ecc_scrub_bus().write64(0, 0), BUS_ERR);
            assert_eq!(md.reported(), [(false, false); 2]);
        }
    }
    /// The ECC view takes an uncached SH's check word from wherever the PROM put it in the source register.
    #[test]
    fn ecc_view_reads_the_check_word_from_the_whole_source_register() {
        let md = configured();
        let ecc = md.0.ecc_bus();
        // (carrier offset, register bits holding the check word)
        for (carrier, shift) in [(6u64, 0u32), (4, 48), (2, 32), (0, 16)] {
            let source = 0x1111_2222_3333_4444u64 ^ (0xa55a_u64 << shift);
            let word = (source >> shift) as u16;
            assert_eq!(ecc.write_narrow_from(carrier, 2, source), BUS_OK);
            assert_eq!(ecc.read16(carrier).data, word, "carrier {carrier}");
        }
        assert_eq!(ecc.write_narrow_from(5, 1, 0xfeed_0000_0000_0023), BUS_OK);
        assert_eq!(ecc.read8(5).data, 0x23, "a byte keeps the architectural low bits");
        // The memory view is not the ECC view: its halfword is the low bits.
        let view = md.uncached_view();
        assert_eq!(view.write_narrow_from(0x40, 2, 0xdead_beef_1234_5678), BUS_OK);
        assert_eq!(md.dma_view().read64(0x40).data >> 48, 0x5678);
    }
    /// Directory entries are stored with computed check bits unless the configuration turns that off; protection keeps its words.
    #[test]
    fn directory_entries_and_protection_words_are_stored() {
        {
            let md = configured();
            let r = md.md_bus();
            md.0.directory_bus().write64(0, 1);
            assert_eq!(md.0.directory_bus().read64(0).data, 0x001f0001);
            r.write64(0x40000, 2);
            md.0.directory_bus().write64(0, 0x001e0001);
            r.write64(0x40000, 0);
            assert_eq!(md.0.directory_bus().read64(0).data, 0x001e0001);
            assert_eq!(md.uncached_view().read64(0).status, BUS_OK);
            assert_eq!(r.read64(0x40010).data, 0);
        }
        {
            let md = configured();
            let d = md.0.directory_bus();
            assert_eq!(d.write64(0x408, 0x1234), BUS_OK);
            assert_eq!(d.read64(0x2000408).data, 0x1234);
            assert_eq!(d.write64(0x418, 0x5678), BUS_OK);
            assert_eq!(d.read64(0x408).data, 0x1234);
            assert_eq!(d.read64(0x418).data, 0x5678);
            assert_eq!(d.write64(8, u64::MAX), BUS_OK);
            assert_eq!(d.read64(8).data, u64::MAX);
            assert_eq!(d.read32(0).status, BUS_ERR);
            assert_eq!(d.read64(1 << 30).status, BUS_ERR);
        }
    }
}

const DIRECTORY_PREMIUM: [u8; 64] = [
    0x1f, 0x2f, 0x37, 0x3b, 0x3d, 0x3e, 0x07, 0x0b, 0x0d, 0x0e, 0x13, 0x15, 0x16, 0x19, 0x1a, 0x1c,
    0x7f, 0x4f, 0x57, 0x5b, 0x5d, 0x5e, 0x67, 0x6b, 0x6d, 0x6e, 0x73, 0x75, 0x76, 0x79, 0x7a, 0x7c,
    1, 2, 4, 8, 16, 32, 0x23, 0x25, 0x26, 0x29, 0x2a, 0x2c, 0x31, 0x32, 0x34, 0x38, 0x40, 0x43,
    0x45, 0x46, 0x49, 0x4a, 0x4c, 0x51, 0x52, 0x54, 0x58, 0x61, 0x62, 0x64, 0x68, 0x70,
];
const DIRECTORY_STANDARD: [u8; 32] = [
    0x1f, 0x2f, 0x37, 0x3b, 0x3d, 0x3e, 0x07, 0x0b, 0x0d, 0x0e, 0x13, 0x15, 0x16, 0x19, 0x1a, 0x1c,
    1, 2, 4, 8, 16, 32, 0x23, 0x25, 0x26, 0x29, 0x2a, 0x2c, 0x31, 0x32, 0x34, 0x38,
];
fn directory_columns(premium: bool) -> &'static [u8] {
    if premium {
        &DIRECTORY_PREMIUM
    } else {
        &DIRECTORY_STANDARD
    }
}
fn directory_syndrome(mut word: u64, premium: bool) -> u8 {
    if !premium {
        word &= u32::MAX as u64;
    }
    let mut check = 0;
    while word != 0 {
        check ^= directory_columns(premium)[word.trailing_zeros() as usize];
        word &= word - 1;
    }
    check
}
pub fn directory_encode(word: u64, premium: bool) -> u64 {
    let word = word
        & !if premium {
            (63 << 32) | (1 << 48)
        } else {
            63 << 16
        };
    let check = directory_syndrome(word, premium) as u64;
    word | if premium {
        (check & 63) << 32 | (check >> 6) << 48
    } else {
        check << 16
    }
}
