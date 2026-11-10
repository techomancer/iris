//! Bedrock's memory and directory controller: bank configuration, memory
//! self-test, and the directory, ECC and fetch-operation views.
//!
//! Memory is held by [`Memory`]. Cached, uncached and DMA accesses reach the
//! banks without this controller or its lock; the controller decodes the
//! banks from its configuration register, which [`Memory`] reads atomically.
//! CPU coherence requests and hardware protection enforcement are absent;
//! protection registers retain their words. Known unsupported writes are
//! counted and ignored, and unmapped accesses return a bus error.
//!
//! Plain register words, directory entries and reference counters,
//! protection entries and written ECC check bytes live in sparse maps. The
//! guest touches a few of the millions of possible entries, and an entry it
//! never wrote reads as its default, so a dense table would only cost memory.
//!
//! # What is approximated here
//!
//! Memory never has an error. The ECC view keeps the check byte software
//! writes for a doubleword and reads it back while the doubleword holds the
//! data it held then; once the data changes it reads the check byte the data
//! computes, as the controller's own write would have stored. Directory
//! entries are stored with the check bits the controller computes, unless
//! the directory configuration turns that off. Nothing compares stored
//! check bits with their data, so no access corrects a word, latches a
//! memory or directory error or interrupts a processor: the error registers
//! read zero and the ECC scrub window answers reads as the ECC view does and
//! refuses writes. PROM 6.210's memory test writes and reads back ECC check
//! bytes and directory entries this way, and in a boot of it and IRIX
//! 6.5.30 on the Fuel nothing reads memory whose stored check bits disagree
//! with its data.
//!
//! Ported from hw/misc/sgi-bedrock-md.c in jamesbraid/qemu-sgi-origami.

use super::mem::{Memory, RamExtent, BANK_SPAN};
use super::Bedrock;
use crate::cpu::smp::World;
use crate::traits::*;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

pub mod reg {
    pub const CELL: u64 = 0x0;
    pub const CONFIG: u64 = 0x8;
    pub const ARBITRATION: u64 = 0x10;
    pub const MIGRATION_CONFIG: u64 = 0x18;
    /// Outgoing reply queue sizes: read-only, so a write changes nothing.
    pub const REPLY_QUEUE_SIZE: u64 = 0x60;
    pub const PERF_SELECT0: u64 = 0x10000;
    pub const PERF_SELECT1: u64 = 0x10008;
    pub const PERF_COUNTER0: u64 = 0x10010;
    pub const PERF_COUNTER1: u64 = 0x10018;
    pub const REFRESH: u64 = 0x20000;
    /// Flash, UART and Synergy timing on the local "junk" bus.
    pub const JUNK_BUS_TIMING: u64 = 0x20008;
    pub const SELFTEST_COMMAND: u64 = 0x20030;
    pub const SELFTEST_DATA: u64 = 0x20038;
    pub const SELFTEST_ERROR: u64 = 0x20040;
    pub const SELFTEST_STATUS: u64 = 0x20048;
    pub const ISSUE_DEBUG: u64 = 0x20060;
    pub const DIRECTORY_CONFIG: u64 = 0x40000;
    pub const DIRECTORY_ERROR: u64 = 0x40010;
    /// Reads the directory error record and clears it.
    pub const DIRECTORY_ERROR_ALIAS: u64 = 0x40018;
    pub const MIGRATION_DIFFERENCE: u64 = 0x40040;
    pub const MIGRATION_VALUE: u64 = 0x40048;
    /// Outgoing request queue sizes; the model keeps them fixed.
    pub const REQUEST_QUEUE_SIZE: u64 = 0x40050;
    pub const DIRECTORY_DEBUG: u64 = 0x40060;
    pub const ECC_CONFIG: u64 = 0x60000;
    pub const MEMORY_ERROR: u64 = 0x60010;
    /// Reads the memory error record and clears it.
    pub const MEMORY_ERROR_ALIAS: u64 = 0x60018;
    pub const MEMORY_DEBUG: u64 = 0x60040;

    /// Reset sizes: 12 entries in each header field and 24 in each data field.
    pub const REPLY_QUEUE_SIZES: u64 = 0x0018_1818_0c0c_0c00;
    /// Reset sizes: four requests for each of the three queues.
    pub const REQUEST_QUEUE_SIZES: u64 = 0x0404_0400;
}

const CONFIG_MASK: u64 = 0xfbfbfbfb | (3 << 32) | (15 << 44) | (63 << 48) | (7 << 56) | (3 << 60);
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
            cell: u64::MAX,
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
    // RAM uses the banks directly. The registers, the diagnostic views and
    // fetch operations take the state lock.
    state: Mutex<State>,
    // A bank configuration write and a reset publish their memory maps in
    // the order they changed the register.
    remapping: Mutex<()>,
    /// The cell register, published for the checks every register access
    /// and the board's Hub-special windows make. Written under `state`.
    cell: AtomicU64,
    unsupported: AtomicU64,
    /// The node's processors. A bank configuration write moves memory in
    /// every processor's cached view, which only a remap with the others
    /// parked may do (`PpMemSpace`'s contract).
    world: OnceLock<Arc<World>>,
}
#[derive(Clone, Copy)]
pub(super) enum View {
    Registers,
    Scrub,
    Directory,
    Fetchop,
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
            remapping: Mutex::new(()),
            cell: AtomicU64::new(State::default().cell),
            unsupported: AtomicU64::new(0),
            world: OnceLock::new(),
        }
    }
    /// The processors a bank configuration write parks while it remaps.
    pub(super) fn set_world(&self, world: Arc<World>) {
        let _ = self.world.set(world);
    }
    /// Where fitted RAM can answer, for the processor bus's cached view.
    pub(super) fn ram_extent(&self) -> RamExtent {
        self.memory.extent()
    }
    pub(super) fn state_locked(&self) -> bool {
        self.state.is_locked()
    }
    pub(super) fn remapping_locked(&self) -> bool {
        self.remapping.is_locked()
    }
    pub(super) fn physical(&self) -> &Arc<crate::physical::Physical> {
        self.memory.physical()
    }
    pub(super) fn view(&self, chip: Weak<Bedrock>, view: View) -> Arc<dyn BusDevice> {
        Arc::new(MdBus { chip, view })
    }
    /// DMA uses the same Physical entries as the CPU, without accessing
    /// ppmem's CPU-owned bitmap. Hardware protection remains unsupported.
    pub(super) fn dma_view(&self) -> Arc<dyn BusDevice> {
        Arc::new(DmaView(self.memory.clone()))
    }
    /// Uncached memory: the bank the configuration register places at an
    /// address, with no lock and no allocation.
    pub(super) fn uncached_view(&self) -> Arc<dyn BusDevice> {
        Arc::new(UncachedView(self.memory.clone()))
    }

    /// Reinstall RAM after the board sets Physical's default. Call on the
    /// CPU thread or with the CPU stopped, before installing board overlays.
    /// A configuration write rebuilds its own changed mapping.
    pub(super) fn rebuild_memory(&self) {
        let _remapping = self.remapping.lock();
        let config = self.state.lock().config;
        self.memory.remap(config, true);
    }
    /// The stable logical-physical counters used by both compiled stores
    /// and SysAd page claims. Their storage outlives the CPU via its bus.
    #[cfg(feature = "jitv2")]
    pub(super) fn generation_base(&self) -> *mut AtomicU64 {
        self.memory.generations().base()
    }
    fn advance(s: &mut State, elapsed: u64) {
        s.clock_ns = s.clock_ns.saturating_add(elapsed);
        if s.selftest_deadline.is_some_and(|end| s.clock_ns >= end) {
            s.selftest_deadline = None;
            s.selftest_status = if s.selftest_result { 3 } else { 2 };
            s.selftest_error = if s.selftest_result { 0 } else { 1 << 48 };
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
        self.cell.load(Ordering::Relaxed) & 1 != 0
    }
    /// Count and log an access the model does not implement; the caller
    /// answers a bus error.
    fn refuse<T>(&self, what: &str, at: u64) -> Option<T> {
        self.unsupported.fetch_add(1, Ordering::Relaxed);
        crate::dlog!(crate::devlog::LogModule::Bedrock,
            "Bedrock MD refused {what} access at {at:#x}");
        None
    }
    /// An access through `view`.
    pub(super) fn access(&self, view: View, addr: u64, bytes: u64, write: Option<u64>) -> Option<u64> {
        if ![1, 2, 4, 8].contains(&bytes) || addr % bytes != 0 {
            return self.refuse("access", addr);
        }
        // A bank configuration write remaps memory, on the CPU thread as
        // HEART's MEMCFG does, with the other processors parked: another
        // processor's cache instruction changes the pages its view holds,
        // and the remap walks every view's held pages. Parked before taking
        // `remapping`, which a processor waiting to park could be waiting for.
        let reconfigure = write.is_some() && matches!(view, View::Registers) && addr == reg::CONFIG;
        match self.world.get() {
            Some(world) if reconfigure => world.stop_others(|| self.access_inner(view, addr, bytes, write, true)),
            _ => self.access_inner(view, addr, bytes, write, reconfigure),
        }
    }
    fn access_inner(&self, view: View, addr: u64, bytes: u64, write: Option<u64>, reconfigure: bool) -> Option<u64> {
        let remapping = reconfigure.then(|| self.remapping.lock());
        // Sample outside state: even an injected clock must not call out under
        // the lock. Only register observations need the self-test timer.
        let now = matches!(view, View::Registers).then(|| (self.clock)());
        let (result, config) = {
            let mut s = self.state.lock();
            if let Some(now) = now {
                Self::catch_up(&mut s, now);
            }
            let result = match view {
                View::Registers => self.register(&mut s, addr, bytes, write),
                View::Directory => self.directory(&mut s, addr, bytes, write),
                View::Ecc => self.ecc(&mut s, addr, bytes, write),
                View::Scrub if write.is_none() => self.ecc(&mut s, addr, bytes, None),
                View::Scrub => self.refuse("scrub", addr),
                View::Fetchop => self.fetchop(&mut s, addr, bytes, write),
            };
            self.cell.store(s.cell, Ordering::Relaxed);
            (result, s.config)
        };
        if remapping.is_some() {
            self.memory.remap(config, false);
        }
        result
    }
    fn register(&self, s: &mut State, at: u64, bytes: u64, write: Option<u64>) -> Option<u64> {
        if bytes != 8 {
            return self.refuse("register", at);
        }
        let old = match at {
            reg::CELL => s.cell,
            reg::CONFIG => s.config,
            reg::REPLY_QUEUE_SIZE => reg::REPLY_QUEUE_SIZES,
            reg::REFRESH => s.refresh,
            reg::SELFTEST_ERROR => s.selftest_error,
            reg::SELFTEST_STATUS => s.selftest_status,
            reg::DIRECTORY_CONFIG => s.directory_config,
            reg::ECC_CONFIG => s.ecc_config,
            reg::REQUEST_QUEUE_SIZE => reg::REQUEST_QUEUE_SIZES,
            // No error is ever latched: see the module notes.
            reg::MEMORY_ERROR | reg::MEMORY_ERROR_ALIAS | reg::DIRECTORY_ERROR | reg::DIRECTORY_ERROR_ALIAS => 0,
            _ if storage_mask(at).is_some() => *s.storage.get(&at).unwrap_or(&0),
            _ if idle(at) => 0,
            _ => return self.refuse("register", at),
        };
        if let Some(value) = write {
            match at {
                reg::SELFTEST_COMMAND => return self.selftest(s, value),
                reg::CELL => s.cell = value,
                reg::CONFIG => s.config = value & CONFIG_MASK,
                reg::REFRESH => s.refresh = value & ((1 << 63) | 0xfff),
                reg::DIRECTORY_CONFIG => s.directory_config = value & 3,
                reg::ECC_CONFIG => s.ecc_config = value & 1,
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
        let bank = software ^ ((s.config >> 32) & 3);
        if at & 0x7c8 == 0x408 {
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
        let field = s.config >> (bank / 2 * 8);
        let code = (field >> 4) & 15;
        if field & (1 << (bank & 1)) == 0
            || !(1..=5).contains(&code)
            || within >= (32 << 20) << code
        {
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
    /// A read or write of `bytes` at node address `address` in the bank the
    /// configuration places there, for the fetch operations.
    fn ram(&self, s: &State, address: u64, bytes: u64, write: Option<u64>) -> Option<u64> {
        let Some((bank, offset)) = self.memory.decode(s.config, address, bytes) else {
            return self.refuse("ram", address);
        };
        let at = offset & !7;
        let shift = (8 - (offset & 7) - bytes) * 8;
        let mask = u64::MAX >> ((8 - bytes) * 8);
        let old = self.memory.raw_read(bank, at);
        if let Some(value) = write {
            self.memory.raw_write(bank, at, (old & !(mask << shift)) | ((value & mask) << shift));
            Some(0)
        } else {
            Some((old >> shift) & mask)
        }
    }
    fn fetchop(&self, s: &mut State, at: u64, bytes: u64, write: Option<u64>) -> Option<u64> {
        if bytes != 4 && bytes != 8 {
            return self.refuse("fetchop", at);
        }
        let op = at & 0x38;
        if op > if write.is_some() { 32 } else { 24 } {
            return self.refuse("fetchop", at);
        }
        let address = (at & !63) | (at & 4);
        let old = self.ram(s, address, bytes, None)?;
        let next = match (write, op) {
            (None, 0) => return Some(old),
            (Some(v), 0) => v,
            (_, 8) => old.wrapping_add(1),
            (_, 16) => old.wrapping_sub(1),
            (None, 24) => 0,
            (Some(v), 24) => old & v,
            (Some(v), 32) => old | v,
            _ => return self.refuse("fetchop", at),
        };
        self.ram(s, address, bytes, Some(next))?;
        Some(old)
    }
    // Approximation: report whether the selected fitting covers the
    // programmed bank, after 1000 guest ns. Only the zero write phase mutates
    // memory; other patterns have no published transformation here.
    fn selftest(&self, s: &mut State, command: u64) -> Option<u64> {
        if command == 2 || command == 4 {
            s.selftest_deadline = None;
            s.selftest_status = 0;
            s.selftest_error = 0;
            return Some(0);
        }
        if command & 7 != 1 || command & !0x71 != 0 {
            self.unsupported.fetch_add(1, Ordering::Relaxed);
            crate::dlog!(crate::devlog::LogModule::Bedrock,
                "Bedrock MD ignored self-test command {command:#x}");
            return Some(0);
        }
        let selected = (command >> 4) & 7;
        let configured = selected ^ ((s.config >> 32) & 3);
        let field = s.config >> ((configured / 2) * 8);
        let code = (field >> 4) & 15;
        let size = if (1..=5).contains(&code) {
            (32 << 20) << code
        } else {
            0
        };
        let fits = field & (1 << (configured & 1)) != 0
            && size != 0
            && size <= self.memory.fitted(selected);
        s.selftest_result = fits;
        s.selftest_status = 0;
        s.selftest_error = 0;
        s.selftest_deadline = Some(s.clock_ns.saturating_add(1000));
        if fits && s.storage.get(&reg::SELFTEST_DATA).copied().unwrap_or(0) == 0x80 {
            self.memory.zero(selected as usize, size);
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
    fn power_on(&self) {
        // Reset is called on the CPU thread or with it stopped.
        let _remapping = self.remapping.lock();
        let mut s = self.state.lock();
        *s = State::default();
        self.cell.store(s.cell, Ordering::Relaxed);
        drop(s);
        self.memory.remap(0, false);
    }
}
/// Resolve each DMA access through the atomic Physical table, as HEART
/// does. Do not ask Physical for mem_ptr or gen_ptr: those read the inline
/// CPU bitmap. Default block methods split accesses at 4 KiB pages.
struct DmaView(Arc<Memory>);
impl DmaView {
    fn at(&self, address: u64, bytes: u64) -> Option<&dyn BusDevice> {
        self.0.decode(self.0.config(), address, bytes)?;
        let phys = self.0.physical();
        let entry = phys.entry_at(address);
        if crate::ppmem::devmap::entry_eq(entry, phys.default_entry()) {
            return None;
        }
        // SAFETY: the memory this view holds retains Physical, which keeps
        // every entry alive across remaps.
        Some(unsafe { &*entry })
    }
}
macro_rules! dma_rw {
    ($read:ident, $write:ident, $result:ident, $ty:ty) => {
        fn $read(&self, address: u64) -> $result {
            self.at(address, std::mem::size_of::<$ty>() as u64)
                .map_or($result::err(), |device| device.$read(address))
        }
        fn $write(&self, address: u64, value: $ty) -> u32 {
            self.at(address, std::mem::size_of::<$ty>() as u64)
                .map_or(BUS_ERR, |device| {
                    let result = device.$write(address, value);
                    if result == BUS_OK { self.0.written(address); }
                    result
                })
        }
    };
}
impl BusDevice for DmaView {
    dma_rw!(read8, write8, BusRead8, u8);
    dma_rw!(read16, write16, BusRead16, u16);
    dma_rw!(read32, write32, BusRead32, u32);
    dma_rw!(read64, write64, BusRead64, u64);
    fn cas32(&self, address: u64, old: u32, new: u32) -> Option<bool> {
        let swapped = self.at(address, 4)?.cas32(address, old, new)?;
        if swapped { self.0.written(address); }
        Some(swapped)
    }
    fn cas64(&self, address: u64, old: u64, new: u64) -> Option<bool> {
        let swapped = self.at(address, 8)?.cas64(address, old, new)?;
        if swapped { self.0.written(address); }
        Some(swapped)
    }
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, address: u64) -> *const AtomicU64 {
        self.0.gen_ptr(address)
    }
}

/// Uncached memory, as the processor's uncached memory space and the low
/// 256 MiB of its Hub-special space reach it: the bank the configuration
/// register places at the address, read and written directly. An address
/// no enabled bank holds answers a bus error. No lock and no allocation.
struct UncachedView(Arc<Memory>);
macro_rules! uncached_rw {
    ($read:ident, $write:ident, $result:ident, $ty:ty) => {
        fn $read(&self, address: u64) -> $result {
            match self.0.decode(self.0.config(), address, std::mem::size_of::<$ty>() as u64) {
                Some((bank, offset)) => self.0.physical().banks()[bank].$read(offset),
                None => $result::err(),
            }
        }
        fn $write(&self, address: u64, value: $ty) -> u32 {
            match self.0.decode(self.0.config(), address, std::mem::size_of::<$ty>() as u64) {
                Some((bank, offset)) => {
                    let result = self.0.physical().banks()[bank].$write(offset, value);
                    if result == BUS_OK { self.0.written(address); }
                    result
                }
                None => BUS_ERR,
            }
        }
    };
}
impl BusDevice for UncachedView {
    uncached_rw!(read8, write8, BusRead8, u8);
    uncached_rw!(read16, write16, BusRead16, u16);
    uncached_rw!(read32, write32, BusRead32, u32);
    uncached_rw!(read64, write64, BusRead64, u64);
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, address: u64) -> *const AtomicU64 {
        self.0.gen_ptr(address)
    }
}

impl BusDevice for MdBus {
    /// Bedrock's ECC view is the one device that sees the whole source register
    /// of an uncached SH (`BusDevice::write_narrow_from`); every other view
    /// takes the architectural low bits.
    fn write_narrow_from(&self, addr: u64, size: usize, gpr: u64) -> u32 {
        if size == 2 && matches!(self.view, View::Ecc) {
            // PROM 6.210's memory test stores each check word with `sh` from a
            // register whose low 16 bits are zero: the word sits elsewhere in
            // the register, by the address of the memory it checks. Observed
            // (memory address mod 0x40, carrier offset mod 8, register bits):
            //   0x00  6  15:0
            //   0x10  4  63:48
            //   0x20  2  47:32
            //   0x30  0  31:16
            // The shift below gives exactly these four. Which lanes the R14000
            // drives for a narrow store, and how Bedrock wires them to the
            // check bits, is unknown; this matches the PROM, not a hardware
            // description.
            let memory_address = (addr ^ 6).wrapping_mul(8);
            return self.write16(addr, (gpr >> (memory_address.wrapping_neg() & 63)) as u16);
        }
        if size == 1 { self.write8(addr, gpr as u8) } else { self.write16(addr, gpr as u16) }
    }

    fn read8(&self, a: u64) -> BusRead8 {
        self.chip
            .upgrade()
            .and_then(|chip| chip.md.access(self.view, a, 1, None))
            .map_or(BusRead8::err(), |v| BusRead8::ok(v as u8))
    }
    fn read16(&self, a: u64) -> BusRead16 {
        self.chip
            .upgrade()
            .and_then(|chip| chip.md.access(self.view, a, 2, None))
            .map_or(BusRead16::err(), |v| BusRead16::ok(v as u16))
    }
    fn read32(&self, a: u64) -> BusRead32 {
        self.chip
            .upgrade()
            .and_then(|chip| chip.md.access(self.view, a, 4, None))
            .map_or(BusRead32::err(), |v| BusRead32::ok(v as u32))
    }
    fn read64(&self, a: u64) -> BusRead64 {
        self.chip
            .upgrade()
            .and_then(|chip| chip.md.access(self.view, a, 8, None))
            .map_or(BusRead64::err(), BusRead64::ok)
    }
    fn write8(&self, a: u64, v: u8) -> u32 {
        self.chip
            .upgrade()
            .and_then(|chip| chip.md.access(self.view, a, 1, Some(v as u64)))
            .map_or(BUS_ERR, |_| BUS_OK)
    }
    fn write16(&self, a: u64, v: u16) -> u32 {
        self.chip
            .upgrade()
            .and_then(|chip| chip.md.access(self.view, a, 2, Some(v as u64)))
            .map_or(BUS_ERR, |_| BUS_OK)
    }
    fn write32(&self, a: u64, v: u32) -> u32 {
        self.chip
            .upgrade()
            .and_then(|chip| chip.md.access(self.view, a, 4, Some(v as u64)))
            .map_or(BUS_ERR, |_| BUS_OK)
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        self.chip
            .upgrade()
            .and_then(|chip| chip.md.access(self.view, a, 8, Some(v)))
            .map_or(BUS_ERR, |_| BUS_OK)
    }
}
fn storage_mask(at: u64) -> Option<u64> {
    Some(match at {
        reg::ARBITRATION => 0xff,
        reg::MIGRATION_CONFIG => 0x3ff | (0xff << 16) | (1 << 32),
        reg::PERF_SELECT0 | reg::PERF_SELECT1 => {
            3 | (15 << 4) | (127 << 8) | (15 << 16) | (0xffffffffff << 20) | (1 << 63)
        }
        reg::PERF_COUNTER0 | reg::PERF_COUNTER1 => (1 << 41) - 1,
        reg::JUNK_BUS_TIMING => 0xffffffffffff,
        reg::SELFTEST_DATA => 0xff,
        reg::ISSUE_DEBUG => 3,
        reg::MIGRATION_DIFFERENCE => 0x7fff | (7 << 32) | (7 << 35) | (1 << 63),
        reg::MIGRATION_VALUE => 0x7fff | (7 << 32) | (7 << 35) | (3 << 62),
        reg::DIRECTORY_DEBUG => 3,
        reg::MEMORY_DEBUG => 1,
        _ => return None,
    })
}
fn idle(at: u64) -> bool {
    matches!(
        at,
        0x20 | 0x28
            | 0x40
            | 0x48
            | 0x58
            | 0x20010
            | 0x20018
            | 0x20020
            | 0x20028
            | 0x20030
            | 0x20040
            | 0x20048
            | 0x40020
            | 0x40028
            | 0x40030
            | 0x40038
            | 0x40058
            | 0x60020
            | 0x60028
            | 0x60030
            | 0x60038
    )
}

#[cfg(test)]
mod tests {
    // Cases follow tests/qtest/sn1-bedrock-md-test.c in
    // jamesbraid/qemu-sgi-origami: memory-banks, reset, queue-sizes,
    // writable-masks, selftest-phases, fetchop-operations, ecc-carrier and
    // directory-storage.
    use super::*;
    use super::super::BedrockConfig;

    /// The controller inside a chip of its own, with the views MD hands out.
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
            let chip = Bedrock::new(BedrockConfig { banks_mib: banks, ..super::super::test_config(clock) })?;
            Ok(Arc::new(Self(chip)))
        }
        fn md_bus(&self) -> Arc<dyn BusDevice> {
            self.0.md.view(Arc::downgrade(&self.0), View::Registers)
        }
        fn uncached_view(&self) -> Arc<dyn BusDevice> {
            self.0.uncached_view()
        }
        fn directory_bus(&self) -> Arc<dyn BusDevice> {
            self.0.directory_bus()
        }
        fn ecc_bus(&self) -> Arc<dyn BusDevice> {
            self.0.ecc_bus()
        }
        fn ecc_scrub_bus(&self) -> Arc<dyn BusDevice> {
            self.0.ecc_scrub_bus()
        }
        fn fetchop_bus(&self) -> Arc<dyn BusDevice> {
            self.0.fetchop_bus()
        }
        fn dma_view(&self) -> Arc<dyn BusDevice> {
            self.0.dma_view()
        }
        /// Whether each PI has bank 1 bit 61 (a correctable memory error)
        /// or error-summary bit 24 (an uncorrectable one) set.
        fn reported(&self) -> [(bool, bool); 2] {
            let regs = self.0.registers();
            std::array::from_fn(|n| {
                let base = n as u64 * super::super::reg::PI1_BASE;
                let pending = regs.read64(base + super::super::reg::pi::PENDING_STATUS_B).data;
                let summary = regs.read64(base + super::super::reg::pi::ERROR_SUMMARY).data;
                (pending & (1 << 61) != 0, summary & (1 << 24) != 0)
            })
        }
    }
    impl Md {
        /// Hold both locks, for tests of the paths that must not take them.
        pub(in crate::dev::bedrock) fn hold(&self) -> impl Sized + '_ {
            (self.remapping.lock(), self.state.lock())
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
        assert_eq!(md.md_bus().write64(reg::CONFIG, TEZRO_CONFIG), BUS_OK);
        md
    }

    /// Cached, uncached and DMA memory take no MD lock: each works while
    /// both are held. An address no enabled bank holds answers a bus error
    /// on each.
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

    /// Guest code must not keep a compiled page after its backing bank changes.
    /// page_for deliberately retains its generation pointer. A remap must
    /// invalidate that page and subsequent writes to its new bank must keep
    /// invalidating the same pointer, including after swapping back.
    #[cfg(feature = "jitv2")]
    #[test]
    fn a_claimed_jit_page_tracks_physical_ownership_across_bank_swaps() {
        use crate::cpu::jitv2::jitv2::Jitv2;
        let md = configured();
        let dma = md.dma_view();
        let mut jit = Jitv2::new(4);
        let slot = jit.page_for(0, 0, dma.as_ref(), false).unwrap();
        let page = jit.page_ptr(slot);
        let generation = || unsafe { (*page).current_gen() };
        let pointer = dma.gen_ptr(0);
        let mut entries = [0u64; crate::cpu::jitv2::jitv2::BITMAP_WORDS];
        entries[0] = 1;
        // Exercise the JIT publication/invalidation protocol without
        // executing a synthetic function body.
        let code = std::ptr::NonNull::<()>::dangling().as_ptr() as *const ();
        assert!(unsafe { (*page).publish(&entries, code, generation(), 1, 0, false) });
        assert!(unsafe { (*page).is_runnable(0) });
        let first = generation();
        md.md_bus().write64(reg::CONFIG, 0x33 | (1 << 32));
        assert_eq!(jit.page_for(0, 0, dma.as_ref(), false), Some(slot));
        assert_eq!(dma.gen_ptr(0), pointer);
        assert!(generation() > first);
        assert!(!unsafe { (*page).is_runnable(0) });
        let swapped = generation();
        assert!(unsafe { (*page).publish(&entries, code, swapped, 1, 0, false) });
        assert!(unsafe { (*page).is_runnable(0) });
        assert_eq!(dma.write64(0, 0x11223344), BUS_OK);
        assert!(generation() > swapped);
        assert!(!unsafe { (*page).is_runnable(0) }, "code published after a swap sees new-bank writes");
        assert_eq!(md.physical().banks()[1].read64(0).data, 0x11223344);
        let written = generation();
        md.md_bus().write64(reg::CONFIG, 0x33);
        assert!(generation() > written, "swapping back cannot lower the logical generation");
        let restored = generation();
        md.uncached_view().write64(0, 0x55667788);
        assert!(generation() > restored);
        let before_fetchop = generation();
        md.fetchop_bus().write64(0, 0);
        assert!(generation() > before_fetchop);
        md.md_bus().write64(reg::CONFIG, 0x13);
        md.md_bus().write64(0x20038, 0x80);
        let before_zero = generation();
        md.md_bus().write64(reg::SELFTEST_COMMAND, 1);
        assert!(generation() > before_zero, "zero self-test invalidates logical code pages");
        let before_disable = generation();
        md.power_on();
        assert!(generation() > before_disable);
        assert_eq!(dma.gen_ptr(0), pointer, "a disabled bank keeps its logical counter alive");
    }

    /// Directory and ECC state is keyed per bank, so banks of different sizes
    /// never share an entry. Follows directory-bank-translation.
    #[test]
    fn directory_entries_of_different_sized_banks_stay_apart() {
        let md = TestMd::new(&[256, 256, 64, 64]).unwrap();
        md.md_bus().write64(reg::CONFIG, 0x33 | (0x13 << 8));
        md.md_bus().write64(reg::DIRECTORY_CONFIG, 2);
        let dir = md.directory_bus();
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

    /// The board names what it fits; nothing else is accepted. A bank is a
    /// whole number of direct-window regions and at most the largest size
    /// code names, and the controller decodes eight.
    #[test]
    fn a_board_may_only_fit_banks_the_controller_can_decode() {
        assert!(TestMd::new(&FUEL_BANKS).is_ok());
        assert!(TestMd::new(&TEZRO_BANKS).is_ok());
        assert!(TestMd::new(&[1024; 8]).is_ok());
        for banks in [&[][..], &[0, 256], &[2048], &[100], &[64; 9]] {
            assert!(TestMd::new(banks).is_err(), "{banks:?}");
        }
    }

    /// A byte lane of the configuration word holds one size code for both
    /// of its banks, so a board cannot fit them at different sizes; a lane
    /// with one bank fitted is still a board's to choose.
    #[test]
    fn the_two_banks_of_a_lane_share_one_size() {
        for banks in [&[1024, 512][..], &[512, 512, 1024, 512], &[1024, 1024, 512, 512, 512, 1024]] {
            assert!(TestMd::new(banks).err().is_some_and(|e| e.contains("share one size")), "{banks:?}");
        }
        for banks in [&[512][..], &[1024, 1024, 512], &[512, 512, 1024, 1024]] {
            assert!(TestMd::new(banks).is_ok(), "{banks:?}");
        }
    }

    /// Where RAM can be, for a board's cached view: the swizzle permutes slots
    /// within groups of four, so a slot reaches as far as the largest bank of
    /// its group.
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

    /// Each Tezro bank answers in its own 1 GiB slot at its own size: a
    /// 1 GiB bank fills its slot, a 512 MiB bank stops halfway, and the
    /// slots above bank 5 answer nothing. Banks 4 and 5 sit above 4 GiB,
    /// inside the node's 16 GiB direct window. Follows the qtest memory-banks and
    /// memory-size-boundaries cases in tests/qtest/sn1-bedrock-md-test.c.
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
    }

    /// A node whose banks all lie below 4 GiB, as the Fuel's do, keeps the
    /// 4 GiB window and its 64 MiB regions.
    #[test]
    fn a_node_below_4_gib_keeps_the_4_gib_window() {
        use crate::ppmem::MappedMemory;
        let md = configured();
        let space = md.physical().ppmem_space().expect("a window");
        assert_eq!(space.window_size(), crate::ppmem::WINDOW_SIZE);
        assert_eq!(space.mapped_bitmap(), 0xf | (0xf << 16));
    }

    /// A processor's SC to a bank above 4 GiB swaps through the DMA view in one step and leaves a failed compare
    /// alone; a successful one advances the page's logical counter.
    #[test]
    fn a_compare_and_swap_above_4_gib_is_one_step() {
        let md = tezro();
        let view = md.dma_view();
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

    /// Selector 1 swaps the two banks of each pair, so the 512 MiB banks 4 and 5
    /// trade places above 4 GiB. Follows memory-selector-high-bank.
    #[test]
    fn the_bank_select_swaps_banks_above_4_gib() {
        let md = tezro();
        let bus = md.physical();
        assert_eq!(bus.write64(4 * BANK_SPAN + 0x28, 0xfedc_ba98_7654_3210), BUS_OK);
        assert_eq!(bus.read64(5 * BANK_SPAN + 0x28).data, 0);
        assert_eq!(md.md_bus().write64(reg::CONFIG, TEZRO_CONFIG | (1 << 32)), BUS_OK);
        assert_eq!(bus.read64(5 * BANK_SPAN + 0x28).data, 0xfedc_ba98_7654_3210);
        assert_eq!(bus.read64(4 * BANK_SPAN + 0x28).data, 0);
    }

    /// A fitting that covers the bank under test zero-fills it, whichever slot
    /// the bank is in; a bank that is not fitted fails. Follows the
    /// selftest-bank-remap and origin300-selftest-fill cases.
    #[test]
    fn selftest_fills_a_tezro_bank_above_4_gib_and_fails_a_missing_bank() {
        let md = TestMd::new(&TEZRO_BANKS).unwrap();
        // Every pair at size code 1 (64 MiB): small enough to fill in a test.
        let regs = md.md_bus();
        regs.write64(reg::CONFIG, 0x13 | (0x13 << 8) | (0x13 << 16) | (0x13 << 24));
        let bus = md.physical();
        bus.write64(4 * BANK_SPAN + 0x40, 0x5555);
        regs.write64(reg::SELFTEST_DATA, 0x80);
        regs.write64(reg::SELFTEST_COMMAND, (4 << 4) | 1);
        md.advance_ns(1000);
        assert_eq!(regs.read64(reg::SELFTEST_STATUS).data, 3, "bank 4 is fitted");
        assert_eq!(bus.read64(4 * BANK_SPAN + 0x40).data, 0, "and zeroed");
        regs.write64(reg::SELFTEST_COMMAND, (6 << 4) | 1);
        md.advance_ns(1000);
        assert_eq!(regs.read64(reg::SELFTEST_STATUS).data, 2, "bank 6 is not fitted");
    }

    /// Code pages above 4 GiB have stable logical generation counters like
    /// the rest, and a write bumps them.
    #[cfg(feature = "jitv2")]
    #[test]
    fn code_pages_above_4_gib_keep_a_logical_generation() {
        let md = tezro();
        let above = 4 * BANK_SPAN + 0x2000;
        let pointer = md.dma_view().gen_ptr(above);
        assert!(!pointer.is_null());
        assert_ne!(pointer, md.dma_view().gen_ptr(5 * BANK_SPAN + 0x2000));
        assert!(md.dma_view().gen_ptr(8 * BANK_SPAN).is_null(), "past the node's 8 GiB");
        assert!(md.dma_view().gen_ptr(4 * BANK_SPAN + (512 << 20)).is_null(), "past the largest bank there");
        let before = unsafe { (*pointer).load(Ordering::Relaxed) };
        assert_eq!(md.dma_view().write64(above, 1), BUS_OK);
        assert!(unsafe { (*pointer).load(Ordering::Relaxed) } > before);
        md.md_bus().write64(reg::CONFIG, TEZRO_CONFIG | (1 << 32));
        assert_eq!(md.dma_view().gen_ptr(above), pointer, "a bank swap keeps the counter");
    }

    /// A bank configuration write moves memory under every processor's
    /// cached view, so the others park for it; another register write
    /// leaves them running. The other processor holds and drops line
    /// pages in its view while it runs, as its cache instructions do: the
    /// remap walks those pages, and doing so unparked panicked a Tezro.
    /// MD reads its clock inside the write, which is where the test looks.
    #[test]
    fn a_bank_configuration_write_parks_the_other_processors() {
        use crate::cpu::smp::{on_cpu_thread, set_current_cpu, ParkSlot, World, HOST_PARK_BIT};
        use std::sync::atomic::AtomicBool;
        let watched: Arc<OnceLock<Arc<ParkSlot>>> = Arc::new(OnceLock::new());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let clock: crate::dev::heart::Clock = {
            let (watched, seen) = (watched.clone(), seen.clone());
            Arc::new(move || {
                if let Some(slot) = watched.get().filter(|_| on_cpu_thread() == Some(0)) {
                    seen.lock().push(slot.is_parked());
                }
                0
            })
        };
        let md = TestMd::with_clock(&FUEL_BANKS, clock).unwrap();
        assert_eq!(md.md_bus().write64(reg::CONFIG, 0x33), BUS_OK);
        let space = md.physical().ppmem_space().expect("a window");
        let _mine = space.add_cached_view(&[]).unwrap();
        let theirs = space.add_cached_view(&[]).unwrap();
        let words: Vec<Arc<AtomicU64>> = (0..2).map(|_| Arc::new(AtomicU64::new(0))).collect();
        let slots: Vec<Arc<ParkSlot>> = words.iter().map(|w| Arc::new(ParkSlot::new(Arc::as_ptr(w)))).collect();
        let world = Arc::new(World::new());
        world.set_slots(slots.clone());
        md.0.set_world(world);
        let running = Arc::new(AtomicBool::new(true));
        slots[1].set_active(true);
        let other = {
            let (md, slot, word, running) = (md.clone(), slots[1].clone(), words[1].clone(), running.clone());
            std::thread::spawn(move || {
                set_current_cpu(1);
                let space = md.physical().ppmem_space().unwrap();
                let page = crate::ppmem::map::granularity() as u64;
                while running.load(Ordering::Relaxed) {
                    if word.load(Ordering::Relaxed) & HOST_PARK_BIT != 0 {
                        slot.park(&running);
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
        let _ = watched.set(slots[1].clone());
        let regs = md.md_bus();
        std::thread::scope(|s| {
            s.spawn(|| {
                set_current_cpu(0);
                for swap in [1u64, 0, 1, 0] {
                    assert_eq!(regs.write64(reg::CONFIG, 0x33 | swap << 32), BUS_OK);
                    assert_eq!(std::mem::take(&mut *seen.lock()), [true], "the other processor parked for the remap");
                }
                // Once it runs again, other registers leave it running.
                while watched.get().unwrap().is_parked() {
                    std::thread::yield_now();
                }
                assert_eq!(regs.write64(reg::REFRESH, 0), BUS_OK);
                assert_eq!(std::mem::take(&mut *seen.lock()), [false], "a refresh write moves no memory");
            });
        });
        running.store(false, Ordering::Relaxed);
        other.join().unwrap();
        assert_eq!(md.physical().read64(BANK_SPAN).status, BUS_OK);
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
        md.power_on();
        assert_eq!(bus.read64(0).status, BUS_ERR);
        bus.map_unmapped(0x1fc00000, 0x1fc01000);
        for (code, size) in [(1, 64 << 20), (2, 128 << 20), (3, FITTED_BANK),
            (4, FITTED_BANK), (5, FITTED_BANK)] {
            md.md_bus().write64(reg::CONFIG, 1 | (code << 4));
            assert_eq!(bus.read64(size - 8).status, BUS_OK);
            assert_eq!(bus.read64(size).status, BUS_ERR);
            assert_eq!(bus.read64(0x1fc00000).status, BUS_OK, "a non-RAM mapping survives");
        }
        md.md_bus().write64(reg::CONFIG, 0x11);
        assert_eq!(bus.read64(64 << 20).status, BUS_ERR, "the old prefix is removed on shrink");
        md.md_bus().write64(reg::CONFIG, 0x61);
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
            let view = md.uncached_view();
            let logical = view.gen_ptr(0);
            assert_eq!(logical, md.memory.gen_ptr(0));
            let logical_before = unsafe { (*logical).load(Ordering::Relaxed) };
            drop(md);
            assert_eq!(unsafe { (*logical).load(Ordering::Relaxed) }, logical_before,
                "an external endpoint keeps its logical generation pointers alive");
            // The endpoint keeps the backing memory alive after MD disappears.
            assert!(unsafe { (*bank1).load(Ordering::Relaxed) } > 0);
        }
    }
    /// The guest reads ECC configuration and the error registers, which never
    /// latch an error: an injected check byte read past with memory raises
    /// none, and reaches no processor.
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
            md.ecc_bus().write8(7, 3);
            assert_eq!(md.uncached_view().read64(0).status, BUS_OK);
            r.write64(0x40000, 2);
            md.directory_bus().write64(0, 3 << 16);
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
            md.power_on();
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
            md.power_on();
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
            md.ecc_bus().write8(7, 1);
            md.ecc_bus().write8(15, 1);
            r.write64(0x20038, 0x80);
            r.write64(0x20030, 1);
            md.advance_ns(1000);
            assert_eq!(r.read64(0x20048).data, 3);
            assert_eq!(md.physical().read64(0).data, 0);
            assert_eq!(md.ecc_bus().read8(7).data, 0);
            assert_eq!(md.ecc_bus().read8(15).data, 0);
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
    /// The guest reads back the check bytes it writes, until a doubleword's
    /// data changes and it reads the check byte the new data computes. The
    /// scrub window reads as the ECC view and refuses writes.
    #[test]
    fn ecc_carriers_keep_written_check_bytes_until_the_data_changes() {
        {
            let md = configured();
            let ecc = md.ecc_bus();
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
            let (ecc, ram) = (md.ecc_bus(), md.uncached_view());
            assert_eq!(ecc.read8(7).data, ecc_compute(0), "nothing written: the data's own");
            ecc.write8(7, 3);
            assert_eq!(ecc.read8(7).data, 3);
            assert_eq!(md.ecc_scrub_bus().read8(7).data, 3);
            assert_eq!(ram.read64(0).data, 0, "memory reads raise no error");
            assert_eq!(ram.write8(7, 0x5a), BUS_OK);
            assert_eq!(ecc.read8(7).data, 0x82, "the new data's check byte");
            assert_eq!(ecc_compute(0x5a), 0x82);
            ecc.write8(7, 0x82);
            assert!(md.state.lock().ecc.is_empty(), "a computed byte is not kept");
            assert_eq!(md.ecc_scrub_bus().write64(0, 0), BUS_ERR);
            assert_eq!(md.reported(), [(false, false); 2]);
        }
    }
    /// The ECC view takes an uncached SH's check word from wherever the PROM
    /// put it in the source register; other views, and bytes, take the low bits.
    #[test]
    fn ecc_view_reads_the_check_word_from_the_whole_source_register() {
        let md = configured();
        let ecc = md.ecc_bus();
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
    /// Directory entries are stored with the check bits the controller
    /// computes unless the configuration turns that off, and read back as
    /// stored; protection entries and counters keep their words.
    #[test]
    fn directory_entries_and_protection_words_are_stored() {
        {
            let md = configured();
            let r = md.md_bus();
            md.directory_bus().write64(0, 1);
            assert_eq!(md.directory_bus().read64(0).data, 0x001f0001);
            r.write64(0x40000, 2);
            md.directory_bus().write64(0, 0x001e0001);
            r.write64(0x40000, 0);
            assert_eq!(md.directory_bus().read64(0).data, 0x001e0001);
            assert_eq!(md.uncached_view().read64(0).status, BUS_OK);
            assert_eq!(r.read64(0x40010).data, 0);
        }
        {
            let md = configured();
            let d = md.directory_bus();
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
