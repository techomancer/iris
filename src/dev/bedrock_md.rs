//! Bedrock's memory and directory controller: bank configuration, memory
//! self-test, ECC, scrubbing and the directory and fetch-operation views.
//!
//! Memory is held by `bedrock_mem::BedrockMemory`. CPU coherence requests
//! and hardware protection enforcement are absent; protection registers
//! retain their words. Known unsupported writes are counted and ignored,
//! and unmapped accesses return a bus error.
//!
//! Plain register words, directory entries and reference counters,
//! protection entries and injected ECC check bytes live in sparse maps. The
//! guest touches a few of the millions of possible entries, and an entry it
//! never wrote reads as its default, so a dense table would only cost memory.
//!
//! Ported from hw/misc/sgi-bedrock-md.c in jamesbraid/qemu-sgi-origami.

use super::bedrock_mem::{BedrockMemory, RamExtent, BANK_SPAN};
use crate::traits::*;
use parking_lot::Mutex;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};


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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryError {
    Correctable,
    Uncorrectable,
}
type ErrorCallback = Arc<dyn Fn(MemoryError) + Send + Sync>;

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
    ecc: HashMap<u64, u8>,
    admitted: [u16; 2],
    memory_error: u64,
    directory_error: u64,
    marked_directory: HashSet<u64>,
    /// Marks in each 4 KiB page of the dense walk: injected check bytes and
    /// marked directory lines. A page is checked while it has any.
    marked_pages: HashMap<u64, u32>,
    /// Pages that gained their first mark since the CPU side last placed
    /// them, and how many have, so an access can tell whether it added one.
    newly_marked: Vec<u64>,
    marks_added: u64,
    /// A self-test cleared whole banks of marks.
    retire_now: bool,
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
            memory_error: 0,
            directory_error: 0,
            marked_directory: HashSet::new(),
            marked_pages: HashMap::new(),
            newly_marked: Vec::new(),
            marks_added: 0,
            retire_now: false,
            clock_ns: 0,
            clock_anchor: None,
            selftest_deadline: None,
            selftest_result: false,
            selftest_status: 0,
            selftest_error: 0,
        }
    }
}

impl State {
    fn page_marked(&mut self, page: u64) {
        let marks = self.marked_pages.entry(page).or_insert(0);
        *marks += 1;
        if *marks == 1 {
            self.newly_marked.push(page);
            self.marks_added += 1;
        }
    }
    fn page_cleared(&mut self, page: u64) {
        if let Some(marks) = self.marked_pages.get_mut(&page) {
            *marks -= 1;
            if *marks == 0 {
                self.marked_pages.remove(&page);
            }
        }
    }
    /// Inject check byte `code` over the doubleword `key` of the dense walk.
    fn ecc_mark(&mut self, key: u64, code: u8) {
        if self.ecc.insert(key, code).is_none() {
            self.page_marked(key / (4096 / 8));
        }
    }
    fn ecc_lift(&mut self, key: u64) {
        if self.ecc.remove(&key).is_some() {
            self.page_cleared(key / (4096 / 8));
        }
    }
    /// Mark directory line `line` of the dense walk as holding a bad entry.
    fn directory_mark(&mut self, line: u64) {
        if self.marked_directory.insert(line) {
            self.page_marked(line / (4096 / 128));
        }
    }
    fn directory_lift(&mut self, line: u64) {
        if self.marked_directory.remove(&line) {
            self.page_cleared(line / (4096 / 128));
        }
    }
    /// Drop every mark, as a reset does.
    fn lift_all(&mut self) {
        self.ecc.clear();
        self.marked_directory.clear();
        self.marked_pages.clear();
        self.newly_marked.clear();
    }
}

pub struct BedrockMd {
    memory: BedrockMemory,
    clock: Option<crate::dev::heart::Clock>,
    // Ordinary RAM uses Physical's bank views. Diagnostic metadata, repair
    // and fetch operations retain the state lock; direct bank traffic has
    // the same concurrency contract as the other machines' RAM.
    state: Mutex<State>,
    // CPU-side metadata writes and resets serialize snapshot/publication.
    // RAM/DMA reads and writes may remove metadata without taking this lock.
    remapping: Mutex<()>,
    callback: Mutex<Option<ErrorCallback>>,
    unsupported: AtomicU64,
}
#[derive(Clone, Copy)]
enum View {
    Registers,
    Memory,
    Uncached,
    Scrub,
    Directory,
    Fetchop,
    Ecc,
}
struct MdBus {
    md: Weak<BedrockMd>,
    view: View,
    // External endpoints may outlive MD. Keep their generation pointers valid.
    // The endpoint owned by Physical itself needs no extra reference.
    _memory: Option<Arc<crate::physical::Physical>>,
    #[cfg(feature = "jitv2")]
    _generations: Arc<super::bedrock_mem::Generations>,
}

impl BedrockMd {
    /// A controller with `banks` fitted, in MiB from locational bank 0; see
    /// [`BedrockMemory::new`] for what a board may fit.
    pub fn new(banks: &[u32]) -> Result<Arc<Self>, String> {
        Self::build(banks, None)
    }
    pub fn with_clock(banks: &[u32], clock: crate::dev::heart::Clock) -> Result<Arc<Self>, String> {
        Self::build(banks, Some(clock))
    }
    fn build(banks: &[u32], clock: Option<crate::dev::heart::Clock>) -> Result<Arc<Self>, String> {
        let md = Arc::new(Self {
            memory: BedrockMemory::new(banks)?,
            clock,
            state: Mutex::new(State::default()),
            remapping: Mutex::new(()),
            callback: Mutex::new(None),
            unsupported: AtomicU64::new(0),
        });
        md.memory.set_slow_view(md.view(View::Memory));
        md.rebuild_memory();
        Ok(md)
    }
    /// Where fitted RAM can answer, for the processor bus's cached view.
    pub fn ram_extent(&self) -> RamExtent {
        self.memory.extent()
    }
    pub fn register_locks(self: &Arc<Self>) {
        let weak = Arc::downgrade(self);
        crate::locks::register_lock_fn("bedrock-md::state", move || {
            weak.upgrade().is_some_and(|md| md.state.is_locked())
        });
        let weak = Arc::downgrade(self);
        crate::locks::register_lock_fn("bedrock-md::remapping", move || {
            weak.upgrade().is_some_and(|md| md.remapping.is_locked())
        });
        let weak = Arc::downgrade(self);
        crate::locks::register_lock_fn("bedrock-md::callback", move || {
            weak.upgrade().is_some_and(|md| md.callback.is_locked())
        });
    }
    pub fn physical(&self) -> &Arc<crate::physical::Physical> {
        self.memory.physical()
    }
    fn view(self: &Arc<Self>, view: View) -> Arc<dyn BusDevice> {
        Arc::new(MdBus {
            md: Arc::downgrade(self),
            view,
            #[cfg(feature = "jitv2")]
            _generations: self.memory.generations(),
            _memory: if matches!(view, View::Uncached) {
                Some(self.physical().clone())
            } else {
                None
            },
        })
    }
    pub fn md_bus(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        self.view(View::Registers)
    }
    pub fn directory_bus(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        self.view(View::Directory)
    }
    pub fn fetchop_bus(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        self.view(View::Fetchop)
    }
    pub fn ecc_bus(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        self.view(View::Ecc)
    }
    /// DMA uses the same Physical entries as the CPU, without accessing
    /// ppmem's CPU-owned bitmap. Hardware protection remains unsupported.
    pub fn dma_view(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(DmaView(self.clone()))
    }

    fn mapping(s: &State) -> (u64, BTreeSet<u64>) {
        (s.config, s.marked_pages.keys().copied().collect())
    }

    /// Reinstall RAM after the board sets Physical's default. Call on the
    /// CPU thread or with the CPU stopped, before installing board overlays.
    /// Ordinary CPU config/diagnostic writes rebuild their changed mappings
    /// automatically. This also releases pages whose metadata DMA consumed.
    pub fn rebuild_memory(&self) {
        let _remapping = self.remapping.lock();
        let (config, pages) = Self::mapping(&self.state.lock());
        self.memory.remap(config, pages, true);
    }
    /// The guest stored `tag` in a primary data cache tag: the PROM building
    /// its stack in cache lines, or ending that. While they stand the window
    /// answers from a page of its own on every path and stays out of the
    /// direct window. See [`BedrockMemory::startup_tag`]. Called on the CPU
    /// thread, outside the cache's locks.
    pub fn startup_tag_store(&self, tag: u32) {
        let Some(live) = self.memory.startup_tag(tag) else { return };
        let _remapping = self.remapping.lock();
        if self.memory.set_startup(live) {
            let (config, pages) = Self::mapping(&self.state.lock());
            self.memory.remap(config, pages, true);
        }
    }
    /// Zero every bank, as a power-on does. Physical's own `reset_memory`
    /// advances only the banks' counters; code pages follow the logical ones,
    /// so a board resets the banks through here.
    pub fn reset_memory(&self) {
        self.memory.physical().reset_memory();
        self.memory.changed_decode();
    }
    /// Load bank `bank` from a file of `Physical::save_bank`'s format, and
    /// advance the logical counters the bytes changed under.
    pub fn load_bank(&self, bank: usize, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let loaded = self.memory.physical().load_bank(bank, path);
        self.memory.changed_decode();
        loaded
    }
    /// Restore bank `bank` from words `Physical::snapshot_bank_inmem` took,
    /// and advance the logical counters the bytes changed under.
    pub fn restore_bank(&self, bank: usize, words: &[u32]) {
        self.memory.physical().restore_bank_inmem(bank, words);
        self.memory.changed_decode();
    }
    /// The stable logical-physical counters used by both compiled stores
    /// and SysAd page claims. Their storage outlives the CPU via its bus.
    #[cfg(feature = "jitv2")]
    pub(crate) fn generation_base(&self) -> *mut AtomicU64 {
        self.memory.generations().base()
    }
    pub fn on_error(&self, callback: ErrorCallback) {
        *self.callback.lock() = Some(callback);
    }
    /// Explicit elapsed time for a manually clocked controller.
    pub fn advance_ns(&self, elapsed: u64) {
        let mut s = self.state.lock();
        Self::advance(&mut s, elapsed);
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
    pub fn start_clock(&self) {
        if let Some(clock) = &self.clock {
            let now = clock();
            self.state.lock().clock_anchor.get_or_insert(now);
        }
    }
    pub fn stop_clock(&self) {
        if let Some(clock) = &self.clock {
            let now = clock();
            let mut s = self.state.lock();
            Self::catch_up(&mut s, now);
            s.clock_anchor = None;
        }
    }
    /// Normal MD memory semantics without re-entering the cached Physical map.
    pub fn memory_view(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        self.view(View::Memory)
    }
    pub fn uncached_view(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        self.view(View::Uncached)
    }
    pub fn ecc_scrub_bus(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        self.view(View::Scrub)
    }
    pub fn current_cell(&self) -> u64 {
        self.state.lock().cell
    }
    pub fn unsupported(&self) -> u64 {
        self.unsupported.load(Ordering::Relaxed)
    }
    /// Count and log an access the model does not implement; the caller
    /// answers a bus error.
    fn refuse<T>(&self, what: &str, at: u64) -> Option<T> {
        self.unsupported.fetch_add(1, Ordering::Relaxed);
        crate::dlog!(crate::devlog::LogModule::Bedrock,
            "Bedrock MD refused {what} access at {at:#x}");
        None
    }
    /// Called on the CPU thread or with the CPU stopped.
    pub fn soft_reset(&self) {
        let _remapping = self.remapping.lock();
        let mut s = self.state.lock();
        s.cell = u64::MAX;
        s.storage.clear();
        s.ecc_config = 0;
        s.memory_error = 0;
        s.directory_error = 0;
        s.lift_all();
        s.selftest_deadline = None;
        s.selftest_status = 0;
        s.selftest_error = 0;
        s.refresh &= (1 << 63) | 0xfff;
        let (config, pages) = Self::mapping(&s);
        drop(s);
        self.memory.remap(config, pages, false);
    }
    fn access(&self, view: View, addr: u64, bytes: u64, write: Option<u64>) -> Option<u64> {
        if ![1, 2, 4, 8].contains(&bytes) || addr % bytes != 0 {
            return self.refuse("access", addr);
        }
        // Only CPU register/configuration and diagnostic writes can add
        // metadata. Keep their remap off DMA threads: ordinary RAM, scrub
        // and fetch operations only remove entries, leaving safe slow pages
        // behind until a CPU-side retirement. These write views, like HEART's
        // MEMCFG, are CPU-thread-only while the machine runs.
        let reconfigure = write.is_some() && matches!(view, View::Registers) && addr == reg::CONFIG;
        let marking = write.is_some() && matches!(view, View::Directory | View::Ecc | View::Registers);
        let remapping = reconfigure.then(|| self.remapping.lock());
        let mut event = Vec::new();
        // Sample outside state: even an injected clock must not call out under
        // the memory lock. Only register observations need the self-test timer.
        let now = if matches!(view, View::Registers) {
            self.clock.as_ref().map(|clock| clock())
        } else {
            None
        };
        let (result, snapshot, placing) = {
            let mut s = self.state.lock();
            if let Some(now) = now {
                Self::catch_up(&mut s, now);
            }
            let marks_added = s.marks_added;
            let result = match view {
                View::Registers => self.register(&mut s, addr, bytes, write),
                View::Directory => self.directory(&mut s, addr, bytes, write),
                View::Ecc => self.ecc(&mut s, addr, bytes, write),
                View::Scrub => self.scrub(&mut s, addr, bytes, write, &mut event),
                View::Uncached => {
                    let result = self.ram(&mut s, addr, bytes, write, &mut event);
                    if write.is_none() && event.contains(&MemoryError::Uncorrectable) {
                        None
                    } else {
                        result
                    }
                }
                View::Fetchop => self.fetchop(&mut s, addr, bytes, write, &mut event),
                View::Memory => self.ram(&mut s, addr, bytes, write, &mut event),
            };
            let placing = marking && (s.marks_added != marks_added || s.retire_now);
            (result, reconfigure.then(|| Self::mapping(&s)), placing)
        };
        if let Some((config, pages)) = snapshot {
            self.memory.reconfigure(config, || pages);
        }
        drop(remapping);
        if placing {
            self.place_marks();
        }
        if !event.is_empty() {
            let callback = self.callback.lock().clone();
            if let Some(callback) = callback {
                for kind in event {
                    callback(kind);
                }
            }
        }
        result
    }
    /// Route the pages that gained a mark through the MD endpoint, and
    /// retire the pages whose marks were lifted when enough have piled up or
    /// have idled long enough. Only a mark can make a page checked, so this
    /// runs after the writes that add one, on the CPU thread.
    fn place_marks(&self) {
        let _remapping = self.remapping.lock();
        let now = self.clock.as_ref().map(|clock| clock());
        let (added, active, retire_now) = {
            let mut s = self.state.lock();
            let added = std::mem::take(&mut s.newly_marked);
            (added, s.marked_pages.len(), std::mem::take(&mut s.retire_now))
        };
        self.memory.place(&added);
        if retire_now || self.memory.retirement_due(active, now) {
            self.retire_lifted();
        }
    }
    /// Return the pages whose marks are lifted to their banks, in one change.
    /// Call on the CPU thread, holding `remapping`.
    fn retire_lifted(&self) {
        let lifted = {
            let s = self.state.lock();
            self.memory.lifted(|page| s.marked_pages.contains_key(&page))
        };
        self.memory.retire(&lifted);
    }
    /// Return the pages whose ECC or directory marks were lifted to the
    /// direct window now, without waiting for the batch to fill or idle. For
    /// the board to call from the CPU thread, or with the CPU stopped, when
    /// it has a quiet moment.
    pub fn retire_lifted_marks(&self) {
        let _remapping = self.remapping.lock();
        self.retire_lifted();
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
            reg::MEMORY_ERROR | reg::MEMORY_ERROR_ALIAS => s.memory_error,
            reg::DIRECTORY_ERROR | reg::DIRECTORY_ERROR_ALIAS => s.directory_error,
            _ if storage_mask(at).is_some() => *s.storage.get(&at).unwrap_or(&0),
            _ if idle(at) => 0,
            _ => return self.refuse("register", at),
        };
        if let Some(value) = write {
            match at {
                reg::SELFTEST_COMMAND => return self.selftest(s, value),
                reg::CELL => s.cell = value,
                reg::CONFIG => {
                    s.config = value & CONFIG_MASK;
                }
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
        } else if at == reg::MEMORY_ERROR_ALIAS {
            s.memory_error = 0;
        } else if at == reg::DIRECTORY_ERROR_ALIAS {
            s.directory_error = 0;
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
            if within < fitted && at & 8 == 0 {
                if directory_syndrome(value, s.directory_config & 1 != 0) != 0 {
                    s.directory_mark(key);
                } else {
                    s.directory_lift(key);
                }
            }
        }
        Some(old)
    }
    fn checked_word(
        &self,
        s: &mut State,
        bank: usize,
        at: u64,
        address: u64,
        event: &mut Vec<MemoryError>,
    ) -> u64 {
        let mut data = self.memory.raw_read(bank, at);
        if s.ecc_config & 1 != 0 || s.ecc.is_empty() {
            return data;
        }
        let key = (self.memory.dense_base(bank as u64) + at) / 8;
        if let Some(&check) = s.ecc.get(&key) {
            let syndrome = check ^ ecc_compute(data);
            if syndrome != 0 {
                let bit = ECC_COLUMNS.iter().position(|&v| v == syndrome);
                let valid = if bit.is_some() { 1 << 52 } else { 1 << 60 };
                if bit.is_none()
                    && s.memory_error & (1 << 60) != 0
                    && (s.memory_error ^ address) & (((1u64 << 30) - 1) << 3) & !127 != 0
                {
                    s.memory_error |= 1 << 63;
                }
                if s.memory_error & (valid | (1 << 60)) != 0 {
                    s.memory_error |= valid << 1;
                } else {
                    s.memory_error = (s.memory_error
                        & !((0xff << 40) | (((1u64 << 30) - 1) << 3)))
                        | ((syndrome as u64) << 40)
                        | (address & (((1u64 << 30) - 1) << 3));
                }
                s.memory_error |= valid;
                if let Some(bit) = bit {
                    if bit < 64 {
                        data ^= 1 << bit;
                        self.memory.raw_write(bank, at, data);
                    }
                    s.ecc_lift(key);
                    event.push(MemoryError::Correctable);
                } else {
                    event.push(MemoryError::Uncorrectable);
                }
            }
        }
        data
    }
    fn ram(
        &self,
        s: &mut State,
        address: u64,
        bytes: u64,
        write: Option<u64>,
        event: &mut Vec<MemoryError>,
    ) -> Option<u64> {
        if let Some(page) = self.memory.startup_device(address, bytes) {
            return Some(startup_access(page, address, bytes, write));
        }
        let Some((bank, offset)) = self.memory.decode(s.config, address, bytes) else {
            return self.refuse("ram", address);
        };
        if !s.marked_directory.is_empty() {
            let line = (self.memory.dense_base(bank as u64) + offset) / 128;
            if s.marked_directory.contains(&line) {
                let premium = s.directory_config & 1 != 0;
                let word = *s.directory.get(&line).unwrap_or(&0);
                if write.is_some() {
                    s.directory.insert(line, directory_encode(word, premium));
                    s.directory_lift(line);
                } else if s.directory_config & 2 == 0 {
                    let syndrome = directory_syndrome(word, premium);
                    if syndrome != 0 {
                        let bit = directory_columns(premium)
                            .iter()
                            .position(|&v| v == syndrome);
                        let valid = if bit.is_some() { 1 << 61 } else { 1 << 63 };
                        if s.directory_error & ((1 << 63) | valid) != 0 {
                            s.directory_error |= if bit.is_some() { 1 << 57 } else { 1 << 59 };
                        } else {
                            let carrier = 0x1c0000000
                                + (address / BANK_SPAN) * (BANK_SPAN / 8)
                                + ((offset & !127) / 8);
                            s.directory_error = (s.directory_error
                                & !((0x7f << 40) | (((1u64 << 30) - 1) << 3)))
                                | ((syndrome as u64) << 40)
                                | (carrier & (((1u64 << 30) - 1) << 3));
                        }
                        s.directory_error |= valid;
                        if let Some(bit) = bit {
                            s.directory.insert(line, word ^ (1 << bit));
                            s.directory_lift(line);
                            event.push(MemoryError::Correctable);
                        } else {
                            event.push(MemoryError::Uncorrectable);
                            return None;
                        }
                    }
                }
            }
        }
        let at = offset & !7;
        let shift = (8 - (offset & 7) - bytes) * 8;
        let mask = u64::MAX >> ((8 - bytes) * 8);
        if let Some(value) = write {
            let old = self.memory.raw_read(bank, at);
            self.memory.raw_write(
                bank,
                at,
                (old & !(mask << shift)) | ((value & mask) << shift),
            );
            if !s.ecc.is_empty() {
                s.ecc_lift((self.memory.dense_base(bank as u64) + at) / 8);
            }
            Some(0)
        } else {
            Some((self.checked_word(s, bank, at, address, event) >> shift) & mask)
        }
    }
    fn fetchop(
        &self,
        s: &mut State,
        at: u64,
        bytes: u64,
        write: Option<u64>,
        event: &mut Vec<MemoryError>,
    ) -> Option<u64> {
        if bytes != 4 && bytes != 8 {
            return self.refuse("fetchop", at);
        }
        let op = at & 0x38;
        if op > if write.is_some() { 32 } else { 24 } {
            return self.refuse("fetchop", at);
        }
        let address = (at & !63) | (at & 4);
        let old = self.ram(s, address, bytes, None, event)?;
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
        self.ram(s, address, bytes, Some(next), event)?;
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
        if fits && s.storage.get(&0x20038).copied().unwrap_or(0) == 0x80 {
            self.memory.zero(selected as usize, size);
            let start = self.memory.dense_base(selected);
            let words: Vec<u64> = s
                .ecc
                .keys()
                .copied()
                .filter(|&key| key >= start / 8 && key < (start + size) / 8)
                .collect();
            for key in words {
                s.ecc_lift(key);
            }
            // The pages those marks held can go back to the window now.
            s.retire_now = true;
            let lines: Vec<u64> = s
                .marked_directory
                .iter()
                .copied()
                .filter(|&line| line >= start / 128 && line < (start + size) / 128)
                .collect();
            for line in lines {
                let word = *s.directory.get(&line).unwrap_or(&0);
                s.directory
                    .insert(line, directory_encode(word, s.directory_config & 1 != 0));
                s.directory_lift(line);
            }
        }
        Some(0)
    }
    fn scrub(
        &self,
        s: &mut State,
        at: u64,
        bytes: u64,
        write: Option<u64>,
        event: &mut Vec<MemoryError>,
    ) -> Option<u64> {
        if write.is_none() {
            return self.ecc(s, at, bytes, None);
        }
        if bytes != 8 || at >= 1 << 30 {
            return self.refuse("scrub", at);
        }
        let software = at / (BANK_SPAN / 8);
        let within = at % (BANK_SPAN / 8);
        let address = software * BANK_SPAN + within * 8;
        let Some((bank, offset)) = self.memory.decode(s.config, address, 8) else {
            return self.refuse("scrub", at);
        };
        for i in 0..8 {
            if self.memory.decode(s.config, address + i * 8, 8).is_some() {
                self.checked_word(s, bank, offset + i * 8, address + i * 8, event);
            }
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
            let computed = ecc_compute(self.memory.raw_read(bank, raw));
            old |= (s.ecc.get(&key).copied().unwrap_or(computed) as u64) << (i * 8);
            if let Some(value) = write {
                let code = (value >> (i * 8)) as u8;
                if code == computed {
                    s.ecc_lift(key);
                } else {
                    s.ecc_mark(key, code);
                }
            }
        }
        Some(old)
    }
}
impl Resettable for BedrockMd {
    fn power_on(&self) {
        // Reset is called on the CPU thread or with it stopped.
        let _remapping = self.remapping.lock();
        *self.state.lock() = State::default();
        let was_live = self.memory.startup_live();
        self.memory.clear_startup();
        self.memory.remap(0, BTreeSet::new(), was_live);
    }
}
/// Resolve each DMA access through the atomic Physical table, as HEART
/// does. Do not ask Physical for mem_ptr or gen_ptr: those read the inline
/// CPU bitmap. Default block methods split accesses and respect slow pages
/// even if a caller supplies a block crossing a 4 KiB page boundary.
struct DmaView(Arc<BedrockMd>);
impl DmaView {
    fn at(&self, address: u64, bytes: u64) -> Option<&dyn BusDevice> {
        if let Some(page) = self.0.memory.startup_device(address, bytes) {
            return Some(page);
        }
        self.0.memory.decode(self.0.memory.config(), address, bytes)?;
        let phys = self.0.physical();
        let entry = phys.entry_at(address);
        if crate::ppmem::devmap::entry_eq(entry, phys.default_entry()) {
            return None;
        }
        // SAFETY: the owning MD retains Physical, which keeps every entry
        // alive across remaps, including the weak MD slow endpoint.
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
                    if result == BUS_OK { self.0.memory.written(address); }
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
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, address: u64) -> *const AtomicU64 {
        self.0.memory.gen_ptr(address)
    }
}

/// A read or write of `bytes` at `address` in the startup page. The page is
/// plain memory: no ECC, no directory, no fetch-and-op.
fn startup_access(page: &dyn BusDevice, address: u64, bytes: u64, write: Option<u64>) -> u64 {
    match (bytes, write) {
        (1, None) => page.read8(address).data as u64,
        (2, None) => page.read16(address).data as u64,
        (4, None) => page.read32(address).data as u64,
        (_, None) => page.read64(address).data,
        (1, Some(v)) => { page.write8(address, v as u8); 0 }
        (2, Some(v)) => { page.write16(address, v as u16); 0 }
        (4, Some(v)) => { page.write32(address, v as u32); 0 }
        (_, Some(v)) => { page.write64(address, v); 0 }
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

    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, address: u64) -> *const std::sync::atomic::AtomicU64 {
        let Some(md) = self.md.upgrade() else {
            return std::ptr::null();
        };
        if !matches!(self.view, View::Memory | View::Uncached) {
            return std::ptr::null();
        }
        md.memory.gen_ptr(address)
    }

    fn read8(&self, a: u64) -> BusRead8 {
        self.md
            .upgrade()
            .and_then(|m| m.access(self.view, a, 1, None))
            .map_or(BusRead8::err(), |v| BusRead8::ok(v as u8))
    }
    fn read16(&self, a: u64) -> BusRead16 {
        self.md
            .upgrade()
            .and_then(|m| m.access(self.view, a, 2, None))
            .map_or(BusRead16::err(), |v| BusRead16::ok(v as u16))
    }
    fn read32(&self, a: u64) -> BusRead32 {
        self.md
            .upgrade()
            .and_then(|m| m.access(self.view, a, 4, None))
            .map_or(BusRead32::err(), |v| BusRead32::ok(v as u32))
    }
    fn read64(&self, a: u64) -> BusRead64 {
        self.md
            .upgrade()
            .and_then(|m| m.access(self.view, a, 8, None))
            .map_or(BusRead64::err(), BusRead64::ok)
    }
    fn write8(&self, a: u64, v: u8) -> u32 {
        self.md
            .upgrade()
            .and_then(|m| m.access(self.view, a, 1, Some(v as u64)))
            .map_or(BUS_ERR, |_| BUS_OK)
    }
    fn write16(&self, a: u64, v: u16) -> u32 {
        self.md
            .upgrade()
            .and_then(|m| m.access(self.view, a, 2, Some(v as u64)))
            .map_or(BUS_ERR, |_| BUS_OK)
    }
    fn write32(&self, a: u64, v: u32) -> u32 {
        self.md
            .upgrade()
            .and_then(|m| m.access(self.view, a, 4, Some(v as u64)))
            .map_or(BUS_ERR, |_| BUS_OK)
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        self.md
            .upgrade()
            .and_then(|m| m.access(self.view, a, 8, Some(v)))
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
    use super::super::bedrock_mem::{RETIRED_IDLE_NS, RETIRED_LIMIT};
    const FUEL_BANKS: [u32; 2] = [256, 256];
    const FITTED_BANK: u64 = 256 << 20;
    fn fuel() -> Arc<BedrockMd> {
        BedrockMd::new(&FUEL_BANKS).unwrap()
    }
    fn configured() -> Arc<BedrockMd> {
        let md = fuel();
        assert_eq!(md.md_bus().write64(8, 0x33), BUS_OK);
        md
    }

    /// What PROM 6.210 stores as primary data cache tags (TagLo), measured:
    /// test patterns, then an invalidate, then a valid tag per page of the
    /// window, then the invalidate sweep it leaves.
    const WINDOW_TAGS: [u32; 8] = [0x100c0, 0x101c1, 0x102c0, 0x103c1, 0x104c0, 0x105c1, 0x106c0, 0x107c1];

    /// A valid tag for a page of the window starts a private page that every
    /// path reaches, whether or not DRAM is configured; an invalid tag ends it
    /// and leaves memory as it was.
    #[test]
    fn a_valid_tag_for_the_window_starts_a_page_every_path_answers() {
        let md = fuel();
        let phys = md.physical();
        let (dma, uncached) = (md.dma_view(), md.uncached_view());
        let at = 0x10_0008 + 0x7000;
        assert_eq!(phys.read64(at).status, BUS_ERR, "no memory behind the window yet");
        md.startup_tag_store(WINDOW_TAGS[0]);
        assert_eq!(phys.write64(at, 0x1122_3344_5566_7788), BUS_OK);
        assert_eq!(phys.read64(at).data, 0x1122_3344_5566_7788);
        assert_eq!(dma.read64(at).data, 0x1122_3344_5566_7788, "the cached path and DMA");
        assert_eq!(uncached.read32(at + 4).data, 0x5566_7788, "the uncached path");
        assert_eq!(uncached.write8(at, 0xaa), BUS_OK);
        assert_eq!(phys.read64(at).data, 0xaa22_3344_5566_7788);
        assert_eq!(phys.read64(0x10_8000).status, BUS_ERR, "only the window's 32 KiB");
        // The page wraps in itself, and the bank behind it is untouched.
        assert_eq!(md.md_bus().write64(reg::CONFIG, 0x33), BUS_OK);
        assert_eq!(phys.read64(at).data, 0xaa22_3344_5566_7788, "configuring DRAM keeps the page");
        assert_eq!(phys.banks()[0].read64(at).data, 0, "while it stands, memory is not the page");
        md.startup_tag_store(0);
        assert_eq!(phys.read64(at).data, 0, "the window is memory again");
        assert_eq!(phys.banks()[0].read64(at).data, 0);
    }

    /// Test patterns, a tag outside the window, and a kernel's invalid tags
    /// leave the decode alone; only the window's valid tags start the page.
    #[test]
    fn other_tags_leave_the_window_to_memory() {
        let md = fuel();
        for tag in [0xffff_ffcf, 0xaaaa_aa8a, 0x5555_5545, 0, 0x108c0, 0xff00c0, 0x1000c0 >> 4] {
            md.startup_tag_store(tag);
            assert!(!md.memory.startup_live(), "{tag:#x}");
        }
        assert_eq!(md.physical().read64(0x10_0000).status, BUS_ERR);
        // The tag's node bits above the node's 8 GiB do not move the window.
        md.startup_tag_store(0x2001_00c0);
        assert!(md.memory.startup_live());
    }

    /// The whole PROM pattern: the page starts at the first install, stays
    /// through the rest and across a configuration write, and ends at the
    /// first invalid tag of the final sweep.
    #[test]
    fn the_proms_tag_sequence_starts_and_ends_the_page_once() {
        let md = fuel();
        let phys = md.physical();
        for tag in [0xffff_ffcf, 0xaaaa_aa8a, 0x5555_5545, 0] {
            md.startup_tag_store(tag);
        }
        assert!(!md.memory.startup_live());
        for tag in WINDOW_TAGS {
            md.startup_tag_store(tag);
            assert!(md.memory.startup_live());
        }
        assert_eq!(phys.write64(0x10_0000, 7), BUS_OK);
        md.md_bus().write64(reg::CONFIG, 0x33);
        md.soft_reset();
        assert_eq!(phys.read64(0x10_0000).data, 7, "a controller soft reset is not a power-on");
        md.startup_tag_store(0);
        assert!(!md.memory.startup_live());
        md.startup_tag_store(0);
        assert!(!md.memory.startup_live());
        // A reboot through the PROM tags again, and the cache still holds
        // what the page held.
        md.startup_tag_store(WINDOW_TAGS[7]);
        assert!(md.memory.startup_live());
        assert_eq!(phys.read64(0x10_0000).data, 7);
    }

    /// The final sweep stores 2050 invalid tags; the page ends at the first,
    /// and the PROM's later stores to the window reach memory, not the page.
    #[test]
    fn the_page_ends_at_the_first_invalid_tag_of_the_final_sweep() {
        let md = configured();
        let phys = md.physical();
        for tag in WINDOW_TAGS {
            md.startup_tag_store(tag);
        }
        assert_eq!(phys.write64(0x10_0000, 5), BUS_OK);
        md.startup_tag_store(0);
        assert!(!md.memory.startup_live(), "the first invalid store ends it");
        assert_eq!(phys.read64(0x10_0000).data, 0);
        assert_eq!(phys.write64(0x10_0000, 6), BUS_OK);
        for _ in 1..2050 {
            md.startup_tag_store(0);
        }
        assert!(!md.memory.startup_live());
        assert_eq!(phys.read64(0x10_0000).data, 6, "memory, untouched by the rest of the sweep");
    }

    /// While the page stands its 64 MiB region is out of the direct window,
    /// so cached loads and the JIT reach it through the decode; the rest of
    /// the region comes back with the page's end.
    #[test]
    fn the_page_keeps_its_region_out_of_the_direct_window() {
        use crate::ppmem::MappedMemory;
        let md = configured();
        let space = md.physical().ppmem_space().unwrap();
        assert_ne!(space.mapped_bitmap() & 1, 0, "configured memory is in the window");
        md.startup_tag_store(WINDOW_TAGS[0]);
        assert_eq!(space.mapped_bitmap() & 1, 0, "while the page stands");
        assert_ne!(space.mapped_bitmap() & 2, 0, "the next region is not affected");
        md.startup_tag_store(0);
        assert_ne!(space.mapped_bitmap() & 1, 0, "and it returns");
    }

    /// Power-on drops the page, zeroed, and its decode.
    #[test]
    fn power_on_clears_the_page() {
        use crate::traits::Resettable;
        let md = configured();
        md.startup_tag_store(WINDOW_TAGS[0]);
        assert_eq!(md.physical().write64(0x10_0000, 9), BUS_OK);
        md.power_on();
        assert!(!md.memory.startup_live());
        assert_eq!(md.physical().read64(0x10_0000).status, BUS_ERR, "no memory until configured");
        md.startup_tag_store(WINDOW_TAGS[0]);
        assert_eq!(md.physical().read64(0x10_0000).data, 0, "the page starts zero");
    }

    /// Ordinary CPU and DMA RAM accesses remain available while MD state is locked.
    #[test]
    fn ordinary_cpu_and_dma_memory_do_not_take_the_md_lock() {
        let md = configured();
        let dma = md.dma_view();
        let _state = md.state.lock();
        assert_eq!(md.physical().write64(0x1000, 0x1122334455667788), BUS_OK);
        assert_eq!(md.physical().read64(0x1000).data, 0x1122334455667788);
        assert_eq!(dma.read64(0x1000).data, 0x1122334455667788);
        assert_eq!(dma.write32(0x1004, 0xabcdef01), BUS_OK);
        assert_eq!(md.physical().read64(0x1000).data, 0x11223344abcdef01);
        assert_eq!(dma.read64(FITTED_BANK).status, BUS_ERR);
    }

    /// An ECC injection removes its direct mapping; DMA repair keeps the page slow.
    #[test]
    fn injected_checks_disable_the_window_and_dma_repair_keeps_the_slow_page() {
        use crate::ppmem::MappedMemory;
        let md = configured();
        let phys = md.physical();
        let space = phys.ppmem_space().expect("a window");
        assert_eq!(space.mapped_bitmap() & 3, 3);
        assert!(phys.mem_ptr(0).is_some());
        assert_eq!(md.ecc_bus().write8(7, 0x13), BUS_OK);
        assert_eq!(space.mapped_bitmap() & 3, 2, "the full 64 MiB region loses its direct bit");
        assert!(phys.mem_ptr(0).is_none());
        let slow = phys.entry_at(0);
        assert!(!crate::ppmem::devmap::entry_eq(slow, phys.entry_at(0x1000)));
        let dma = md.dma_view();
        let repaired = std::thread::spawn(move || dma.read64(0)).join().unwrap();
        assert_eq!(repaired.status, BUS_OK);
        assert_eq!(repaired.data, 1);
        assert!(md.state.lock().ecc.is_empty());
        assert_eq!(space.mapped_bitmap() & 3, 2, "a DMA thread never republishes the CPU bitmap");
        assert!(crate::ppmem::devmap::entry_eq(slow, phys.entry_at(0)));
        // Adjacent pages still use the ordinary bank entry, even though
        // the region no longer has a direct window mapping.
        {
            let _state = md.state.lock();
            assert_eq!(phys.write64(0x1000, 9), BUS_OK);
            assert_eq!(phys.read64(0x1000).data, 9);
        }
        md.soft_reset();
        assert_eq!(space.mapped_bitmap() & 3, 3);
        assert!(phys.mem_ptr(0).is_some());
        assert_eq!(phys.read64(0).data, 1);
    }

    /// Directory-marked pages follow their physical bank after a configuration swap.
    #[test]
    fn marked_directory_pages_follow_their_bank_when_swizzled() {
        use crate::ppmem::MappedMemory;
        let md = configured();
        let phys = md.physical();
        let space = phys.ppmem_space().expect("a window");
        md.md_bus().write64(reg::DIRECTORY_CONFIG, 2);
        md.directory_bus().write64(0, 0x001e0001);
        assert_eq!(space.mapped_bitmap() & 1, 0);
        assert!(phys.mem_ptr(0).is_none());
        md.md_bus().write64(reg::CONFIG, 0x33 | (1 << 32));
        assert_ne!(space.mapped_bitmap() & 1, 0, "the other bank now answers at zero");
        assert_eq!(space.mapped_bitmap() & (1 << 16), 0);
        assert!(phys.mem_ptr(BANK_SPAN).is_none());
        md.md_bus().write64(reg::DIRECTORY_CONFIG, 0);
        assert_eq!(phys.read64(BANK_SPAN).status, BUS_OK);
        assert_ne!(md.md_bus().read64(reg::DIRECTORY_ERROR).data & (1 << 61), 0);
        md.rebuild_memory();
        assert!(phys.mem_ptr(BANK_SPAN).is_some());
        md.power_on();
        assert_eq!(space.mapped_bitmap(), 0);
        assert_eq!(phys.read64(BANK_SPAN).status, BUS_ERR);
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
        md.ecc_bus().write8(7, 0x13);
        let before_repair = generation();
        assert_eq!(dma.read64(0).data, 1);
        assert!(generation() > before_repair, "ECC repair invalidates logical code pages");
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

    /// The Tezro's banks, in MiB from locational bank 0.
    const TEZRO_BANKS: [u32; 6] = [1024, 1024, 512, 512, 512, 512];
    /// Banks 0 and 1 at size code 5 (1 GiB), the other four at code 4 (512 MiB).
    const TEZRO_CONFIG: u64 = 0x53 | (0x43 << 8) | (0x43 << 16);
    fn tezro() -> Arc<BedrockMd> {
        let md = BedrockMd::new(&TEZRO_BANKS).unwrap();
        assert_eq!(md.md_bus().write64(reg::CONFIG, TEZRO_CONFIG), BUS_OK);
        md
    }

    /// Each Tezro bank answers in its own 1 GiB slot at its own size: a
    /// 1 GiB bank fills its slot, a 512 MiB bank stops halfway, and the
    /// slots above bank 5 answer nothing. Banks 4 and 5 sit above 4 GiB,
    /// where the CPU's direct window does not reach, and still answer
    /// through the bus. Follows the qtest memory-banks and
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

        // The window covers the first 4 GiB: 1 GiB, 1 GiB, 512 MiB, 512 MiB.
        let space = bus.ppmem_space().expect("a window");
        assert_eq!(space.mapped_bitmap(), 0xffff | (0xffff << 16) | (0xff << 32) | (0xff << 48));
        assert!(!crate::ppmem::bitmap_maps(space.mapped_bitmap(), 4 * BANK_SPAN));
        assert!(!crate::ppmem::bitmap_maps(space.mapped_bitmap(), 5 * BANK_SPAN));
        assert_eq!(md.dma_view().read64(5 * BANK_SPAN).data, 0x1111 * 6, "DMA reaches it too");
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
        // Banks 0 and 1 swap too: the 1 GiB banks have different sizes only
        // if the board says so, so give bank 1 a distinct value.
        assert_eq!(bus.write64(0, 0xaa), BUS_OK);
        assert_eq!(bus.write64(BANK_SPAN, 0xbb), BUS_OK);
        assert_eq!(md.md_bus().write64(reg::CONFIG, TEZRO_CONFIG), BUS_OK);
        assert_eq!(bus.read64(0).data, 0xbb);
        assert_eq!(bus.read64(BANK_SPAN).data, 0xaa);
    }

    /// A 1 GiB population is wholly inside ppmem's direct window: all 16 of its
    /// 64 MiB regions are mapped, with none left to the decode table. The 4 GiB
    /// default maps only what is below 4 GiB, and its banks 4 and 5 answer
    /// through the table.
    #[test]
    fn a_1_gib_population_is_directly_mapped_and_the_default_4_gib_one_is_not_wholly() {
        use super::super::bedrock_mem::MemoryPopulation;
        use crate::ppmem::MappedMemory;
        let md = BedrockMd::new(&MemoryPopulation::one_gib().banks_mib()).unwrap();
        md.md_bus().write64(reg::CONFIG, 0x43 | (0x43 << 8));
        let bus = md.physical();
        let space = bus.ppmem_space().expect("a window");
        assert_eq!(space.mapped_bitmap(), 0xff | (0xff << 16));
        assert_eq!(bus.write64(BANK_SPAN + (512 << 20) - 8, 7), BUS_OK);
        assert_eq!(bus.read64(BANK_SPAN + (512 << 20) - 8).data, 7);
        assert_eq!(bus.read64(512 << 20).status, BUS_ERR, "bank 0 ends at 512 MiB");

        let md = BedrockMd::new(&MemoryPopulation::default().banks_mib()).unwrap();
        assert_eq!(MemoryPopulation::default(), MemoryPopulation::four_gib());
        md.md_bus().write64(reg::CONFIG, TEZRO_CONFIG);
        let space = md.physical().ppmem_space().expect("a window");
        assert!(!crate::ppmem::bitmap_maps(space.mapped_bitmap(), 4 * BANK_SPAN));
        assert_eq!(md.physical().write64(4 * BANK_SPAN, 9), BUS_OK, "still reachable");
    }

    /// Directory and ECC state is keyed per bank, so banks of different sizes
    /// never share an entry. Follows directory-bank-translation.
    #[test]
    fn directory_entries_of_different_sized_banks_stay_apart() {
        let md = tezro();
        md.md_bus().write64(reg::DIRECTORY_CONFIG, 2);
        let dir = md.directory_bus();
        // Software bank n's carriers start at n * 128 MiB.
        let at = |bank: u64, offset: u64| bank * (BANK_SPAN / 8) + offset;
        for bank in 0..6 {
            assert_eq!(dir.write64(at(bank, 0x20), 0x1000 + bank), BUS_OK, "bank {bank}");
        }
        for bank in 0..6 {
            assert_eq!(dir.read64(at(bank, 0x20)).data, 0x1000 + bank, "bank {bank}");
        }
        // A 512 MiB bank's directory ends at its size code's reach.
        assert_eq!(dir.read64(at(2, 0x20) + (BANK_SPAN / 16)).status, BUS_ERR);
        assert_eq!(dir.read64(at(6, 0x20)).status, BUS_ERR, "no bank 6");
    }

    /// A fitting that covers the bank under test zero-fills it, whichever slot
    /// the bank is in; a bank that is not fitted fails. Follows the
    /// selftest-bank-remap and origin300-selftest-fill cases.
    #[test]
    fn selftest_fills_a_tezro_bank_above_4_gib_and_fails_a_missing_bank() {
        let md = BedrockMd::new(&TEZRO_BANKS).unwrap();
        // Every pair at size code 1 (64 MiB): small enough to fill in a test.
        let regs = md.md_bus();
        regs.write64(reg::CONFIG, 0x13 | (0x13 << 8) | (0x13 << 16) | (0x13 << 24));
        let bus = md.physical();
        bus.write64(4 * BANK_SPAN + 0x40, 0x5555);
        regs.write64(0x20038, 0x80);
        regs.write64(0x20030, (4 << 4) | 1);
        md.advance_ns(1000);
        assert_eq!(regs.read64(0x20048).data, 3, "bank 4 is fitted");
        assert_eq!(bus.read64(4 * BANK_SPAN + 0x40).data, 0, "and zeroed");
        regs.write64(0x20030, (6 << 4) | 1);
        md.advance_ns(1000);
        assert_eq!(regs.read64(0x20048).data, 2, "bank 6 is not fitted");
    }

    /// The board names what it fits; nothing else is accepted. A bank is a
    /// whole number of direct-window regions and at most the largest size
    /// code names.
    #[test]
    fn a_board_may_only_fit_banks_the_controller_can_decode() {
        assert!(BedrockMd::new(&FUEL_BANKS).is_ok());
        assert!(BedrockMd::new(&TEZRO_BANKS).is_ok());
        for banks in [&[][..], &[0, 256], &[2048], &[100], &[64; 9]] {
            assert!(BedrockMd::new(banks).is_err(), "{banks:?}");
        }
    }

    /// Where RAM can be, for a board's cached view: the swizzle permutes slots
    /// within groups of four, so a slot reaches as far as the largest bank of
    /// its group.
    #[test]
    fn the_ram_extent_reaches_as_far_as_the_largest_bank_of_a_slot_group() {
        let fuel = RamExtent::new(&FUEL_BANKS);
        for slot in 0..4 {
            assert!(fuel.contains(slot * BANK_SPAN + FITTED_BANK - 8), "slot {slot}");
            assert!(!fuel.contains(slot * BANK_SPAN + FITTED_BANK), "slot {slot}");
        }
        assert!(!fuel.contains(4 * BANK_SPAN));
        let tezro = RamExtent::new(&TEZRO_BANKS);
        assert!(tezro.contains(3 * BANK_SPAN + BANK_SPAN - 8), "slot 3 may hold a 1 GiB bank");
        assert!(tezro.contains(5 * BANK_SPAN + (512 << 20) - 8));
        assert!(!tezro.contains(5 * BANK_SPAN + (512 << 20)));
        assert!(!tezro.contains(8 * BANK_SPAN));
        assert_eq!(tezro, BedrockMd::new(&TEZRO_BANKS).unwrap().ram_extent());
    }

    /// Code pages above 4 GiB have stable logical generation counters like
    /// the rest, and a write bumps them.
    #[cfg(feature = "jitv2")]
    #[test]
    fn code_pages_above_4_gib_keep_a_logical_generation() {
        let md = tezro();
        let bus = md.physical();
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
        let _ = bus;
    }

    /// The ECC carrier byte for the doubleword at `address`, as the guest
    /// addresses it.
    fn ecc_carrier(address: u64) -> u64 {
        address / BANK_SPAN * (BANK_SPAN / 8) + ((address % BANK_SPAN) / 8 ^ 7)
    }

    /// The cost counters now: host mappings, decode-table rewrites and
    /// generation counters.
    fn counted(md: &Arc<BedrockMd>) -> [u64; 3] {
        md.memory.cost.snapshot()
    }
    /// The counters one bumped 4 KiB page, and one whole 64 MiB region, cost:
    /// the logical counter and the bank's, which only a jitv2 build keeps.
    const PAGE_BUMPS: u64 = if cfg!(feature = "jitv2") { 2 } else { 0 };
    const REGION_BUMPS: u64 = PAGE_BUMPS * (64 << 20) / 4096;
    fn mark(md: &Arc<BedrockMd>, address: u64) {
        assert_eq!(md.ecc_bus().write8(ecc_carrier(address), 0x13), BUS_OK);
    }
    /// A doubleword of zero has a check byte of zero, so writing that lifts
    /// a mark, and writing 0x13 puts one there.
    fn lift(md: &Arc<BedrockMd>, address: u64) {
        assert_eq!(md.ecc_bus().write8(ecc_carrier(address), 0), BUS_OK);
    }
    fn spent(md: &Arc<BedrockMd>, work: impl FnOnce()) -> [u64; 3] {
        let before = counted(md);
        work();
        let after = counted(md);
        [after[0] - before[0], after[1] - before[1], after[2] - before[2]]
    }

    /// What marking and lifting checked pages costs in host mappings,
    /// decode-table rewrites and generation counters. The PROM's memory test
    /// marks and lifts the same few hundred doublewords thousands of times
    /// (the QEMU model counted 6630 of each for 310 doublewords, and
    /// rebuilding the map each time took the test from ten seconds to
    /// minutes), so what one costs is what the test costs. Counts, not
    /// seconds, so the test can pin them. Before this was incremental each of
    /// these was a full rebuild: 15 mappings, 3 or 2 table rewrites and
    /// 393,216 counters on the Fuel; 95, 7 or 6 and 2,621,440 on the Tezro.
    ///
    /// Follows the qtest ecc-lifted-mark-keeps-overlay case in
    /// tests/qtest/sn1-bedrock-md-test.c (QEMU commit 4f0f6072dc0b), which
    /// pins the same thing in the flat view: a lifted mark's overlay stays,
    /// a second mark there is not a map change, and the idle ones go
    /// together.
    #[test]
    fn a_checked_page_costs_a_page_not_a_rebuild() {
        let (fuel, tezro) = (configured(), tezro());
        for (md, base, windowed) in [(&fuel, 0, true), (&tezro, 0, true), (&tezro, 4 * BANK_SPAN, false)] {
            let (first, second) = (base + 0x5000, base + 0x6000);
            // The first page in a region takes the region off the direct
            // window and drops code compiled from any of it; the next page
            // in that region is one more table entry. Above the window there
            // is no region to take.
            let cost = spent(md, || mark(md, first));
            assert_eq!(cost, if windowed { [1, 1, REGION_BUMPS] } else { [0, 1, PAGE_BUMPS] }, "first page at {base:#x}");
            assert_eq!(spent(md, || mark(md, second)), [0, 1, PAGE_BUMPS], "second page at {base:#x}");
            // Lifting costs nothing: both pages stay on the MD endpoint.
            assert_eq!(spent(md, || { lift(md, first); lift(md, second) }), [0, 0, 0], "lifts at {base:#x}");
            // Marking a lifted page again is not a map change either.
            assert_eq!(spent(md, || mark(md, first)), [0, 0, 0], "re-mark at {base:#x}");
            // The retirement returns what is lifted, in one batch, and
            // leaves the marked page where it is.
            assert_eq!(spent(md, || md.retire_lifted_marks()), [0, 1, PAGE_BUMPS], "retire at {base:#x}");
            lift(md, first);
            let window = u64::from(windowed);
            assert_eq!(spent(md, || md.retire_lifted_marks()), [window, 1, PAGE_BUMPS], "last retire at {base:#x}");
        }
    }

    fn counter(md: &Arc<BedrockMd>, address: u64) -> u64 {
        #[cfg(feature = "jitv2")]
        return unsafe { (*md.memory.gen_ptr(address)).load(Ordering::Relaxed) };
        #[cfg(not(feature = "jitv2"))]
        { let _ = (md, address); 0 }
    }

    /// With the controller's clock under the test's hand.
    fn clocked() -> (Arc<BedrockMd>, Arc<AtomicU64>) {
        let now = Arc::new(AtomicU64::new(1));
        let hand = now.clone();
        let md = BedrockMd::with_clock(&FUEL_BANKS, Arc::new(move || hand.load(Ordering::SeqCst))).unwrap();
        assert_eq!(md.md_bus().write64(reg::CONFIG, 0x33), BUS_OK);
        (md, now)
    }

    /// Lifting a mark keeps its page on the MD endpoint, which answers as
    /// the bank would: the lifted doubleword reads and writes plainly, a
    /// second mark there is checked again, and the page goes back to the bank
    /// together with the other idle ones, a second after the first was seen
    /// idle, or at once on reset. Follows the qtest
    /// ecc-lifted-mark-keeps-overlay case in
    /// tests/qtest/sn1-bedrock-md-test.c (QEMU commit 4f0f6072dc0b).
    #[test]
    fn a_lifted_mark_keeps_its_page_on_the_md_endpoint_until_it_idles() {
        use crate::ppmem::MappedMemory;
        let (md, now) = clocked();
        let (phys, bus) = (md.physical(), md.ecc_bus());
        let space = phys.ppmem_space().expect("a window");
        let (address, data) = (0x5000, 0x0123_4567_89ab_cdef);
        let on_bank = |md: &Arc<BedrockMd>, address| {
            crate::ppmem::devmap::entry_eq(md.physical().entry_at(address), md.physical().entry_at(0x20_0000))
        };
        assert_eq!(phys.write64(address, data), BUS_OK);
        assert!(on_bank(&md, address));
        assert_eq!(bus.write8(ecc_carrier(address), ecc_compute(data) ^ ECC_COLUMNS[5]), BUS_OK);
        assert!(!on_bank(&md, address) && space.mapped_bitmap() & 1 == 0);

        // A write lifts the mark; the page stays on the endpoint and reads as memory.
        assert_eq!(phys.write64(address, data), BUS_OK);
        assert!(!on_bank(&md, address) && space.mapped_bitmap() & 1 == 0);
        assert_eq!(phys.read64(address).data, data);
        assert_eq!(phys.read32(address + 4).data, data as u32);
        assert_eq!(phys.write32(address, 0x7654_3210), BUS_OK);
        assert_eq!(phys.read64(address).data, 0x7654_3210_89ab_cdef);
        assert_eq!(phys.write64(address, data), BUS_OK);
        assert_eq!(bus.read8(ecc_carrier(address)).data, ecc_compute(data));
        assert_eq!(md.md_bus().read64(reg::MEMORY_ERROR).data, 0);

        // Marked again, the same doubleword is checked again.
        assert_eq!(bus.write8(ecc_carrier(address), ecc_compute(data) ^ ECC_COLUMNS[6]), BUS_OK);
        assert_eq!(phys.read64(address).data, data ^ (1 << 6));
        assert_ne!(md.md_bus().read64(reg::MEMORY_ERROR).data & (1 << 52), 0);
        assert!(md.state.lock().ecc.is_empty(), "the correcting read lifted it");

        // Idling a second retires it, on the next publication after that.
        let other = |n: u64| 0x1_0000 + n * 0x1000;
        assert_eq!(phys.write64(other(0), 0), BUS_OK);
        mark(&md, other(0));
        now.store(1 + RETIRED_IDLE_NS - 1, Ordering::SeqCst);
        mark(&md, other(1));
        assert!(!on_bank(&md, address), "not yet a second");
        now.store(1 + RETIRED_IDLE_NS, Ordering::SeqCst);
        mark(&md, other(2));
        assert!(on_bank(&md, address), "a second");
        assert_eq!(phys.read64(address).data, data ^ (1 << 6));
        assert!(!on_bank(&md, other(0)) && !on_bank(&md, other(2)), "the marked pages stay checked");

        // Reset takes a kept page back without waiting.
        lift(&md, other(0));
        assert_eq!(phys.write64(other(0), 0), BUS_OK);
        assert!(!on_bank(&md, other(0)));
        md.soft_reset();
        assert!(on_bank(&md, other(0)) && on_bank(&md, other(2)));
        assert_eq!(space.mapped_bitmap() & 1, 1);
    }

    /// A page kept after its mark is lifted still checks and rewrites a
    /// marked directory line in it, as the line's own placement would.
    /// Follows the qtest ecc-lifted-mark-over-marked-line case in
    /// tests/qtest/sn1-bedrock-md-test.c (QEMU commit 4f0f6072dc0b).
    #[test]
    fn a_kept_page_checks_a_directory_line_marked_after_the_lift() {
        let md = configured();
        let phys = md.physical();
        mark(&md, 0x48);
        lift(&md, 0x48);
        md.md_bus().write64(reg::DIRECTORY_CONFIG, 2);
        let cost = spent(&md, || { md.directory_bus().write64(0, 0x001e0001); });
        assert_eq!(cost, [0, 0, 0], "the page is already on the endpoint");
        md.md_bus().write64(reg::DIRECTORY_CONFIG, 0);
        assert_eq!(phys.read64(0x48).status, BUS_OK);
        assert_ne!(md.md_bus().read64(reg::DIRECTORY_ERROR).data & (1 << 61), 0, "the read checked the line");
        // Marked again, a write through the kept page rewrites the entry.
        md.md_bus().write64(reg::DIRECTORY_CONFIG, 2);
        md.directory_bus().write64(0, 0x001e0001);
        assert!(!md.state.lock().marked_directory.is_empty());
        assert_eq!(phys.write64(0x48, 0x22), BUS_OK);
        assert!(md.state.lock().marked_directory.is_empty());
        assert_eq!(phys.read64(0x48).data, 0x22);
    }

    /// The PROM's memory test marks and lifts the same few hundred
    /// doublewords over and over; each page is placed once however often.
    /// The QEMU model counted 6630 placements for 310 doublewords before it
    /// kept lifted overlays.
    #[test]
    fn marking_the_same_pages_over_and_over_places_each_once() {
        let md = configured();
        let pages: Vec<u64> = (0..300).map(|i| i * 0x9_0000 + 0x1000).collect();
        let regions: HashSet<u64> = pages.iter().map(|page| page >> 26).collect();
        let cost = spent(&md, || {
            for _ in 0..20 {
                pages.iter().for_each(|&page| mark(&md, page));
                pages.iter().for_each(|&page| lift(&md, page));
            }
        });
        let regions = regions.len() as u64;
        assert_eq!(cost, [regions, 300, regions * REGION_BUMPS + (300 - regions) * PAGE_BUMPS]);
    }

    /// Lifted pages pile up only so far: the first mark after
    /// `RETIRED_LIMIT` have been lifted returns them all in one batch.
    #[test]
    fn many_lifted_pages_are_retired_in_one_batch() {
        let md = configured();
        let pages: Vec<u64> = (0..RETIRED_LIMIT as u64).map(|i| 0x1000 * i).collect();
        pages.iter().for_each(|&page| mark(&md, page));
        pages.iter().for_each(|&page| lift(&md, page));
        // One more page, in another region, makes the 1024 lifted pages due.
        let cost = spent(&md, || mark(&md, 0x400_0000 + 0x1000));
        let retired = RETIRED_LIMIT as u64;
        assert_eq!(cost, [1 + 1, 1 + retired, REGION_BUMPS + PAGE_BUMPS * retired]);
        assert!(crate::ppmem::devmap::entry_eq(md.physical().entry_at(0x5000), md.physical().entry_at(0x20_0000)));
    }

    /// A zero self-test of a bank clears its marks, and so returns the pages
    /// they held.
    #[test]
    fn a_zero_self_test_returns_the_banks_lifted_pages() {
        use crate::ppmem::MappedMemory;
        let md = configured();
        let space = md.physical().ppmem_space().expect("a window");
        mark(&md, 0x5000);
        assert_eq!(space.mapped_bitmap() & 1, 0);
        md.md_bus().write64(0x20038, 0x80);
        md.md_bus().write64(reg::SELFTEST_COMMAND, 1);
        assert_eq!(space.mapped_bitmap() & 1, 1);
        assert!(crate::ppmem::devmap::entry_eq(md.physical().entry_at(0x5000), md.physical().entry_at(0x20_0000)));
    }

    /// A page that is marked, or its region's direct path, drops the code
    /// compiled from it; a page elsewhere keeps its counter, and lifting
    /// changes nothing until the page is returned.
    #[test]
    fn marking_a_page_advances_the_counters_of_what_left_the_direct_path() {
        let md = configured();
        let (page, neighbour, elsewhere) = (0x5000, 0x9000, 0x400_0000 + 0x5000);
        let seen = |md: &Arc<BedrockMd>| [page, neighbour, elsewhere].map(|a| counter(md, a));
        let start = seen(&md);
        mark(&md, page);
        let marked = seen(&md);
        let step = |a: [u64; 3], b: [u64; 3]| [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let jit = u64::from(cfg!(feature = "jitv2"));
        assert_eq!(step(start, marked), [jit, jit, 0], "the page and its region");
        mark(&md, neighbour);
        assert_eq!(step(marked, seen(&md)), [0, jit, 0], "only the second page");
        lift(&md, page);
        lift(&md, neighbour);
        assert_eq!(step(marked, seen(&md)), [0, jit, 0], "lifting changes nothing");
        md.retire_lifted_marks();
        assert_eq!(step(marked, seen(&md)), [jit, 2 * jit, 0], "returning a page advances it");
    }

    /// A read-only 4 MiB window standing for the reset PROM, whose first
    /// word reads 0x0f1e2d3c4b5a6978.
    struct Prom;
    impl BusDevice for Prom {
        fn read64(&self, a: u64) -> BusRead64 {
            if a == 0 { BusRead64::ok(0x0f1e_2d3c_4b5a_6978) } else { BusRead64::ok(0) }
        }
        fn write64(&self, _: u64, _: u64) -> u32 { BUS_ERR }
    }

    /// The Tezro's memory wired as the board wires it: the processor bus with
    /// uncached RAM views, the cached view, and the reset PROM mapped over the
    /// cached plane where the bank leaves room.
    fn board(md: &Arc<BedrockMd>) -> (crate::ip35::Ip35Bus, Arc<dyn BusDevice>) {
        let mut bus = crate::ip35::Ip35Bus::new(md.physical().clone());
        #[cfg(feature = "jitv2")]
        bus.set_ram_generations(md.dma_view());
        md.rebuild_memory();
        bus.map(3, 0, 1 << 33, md.uncached_view());
        bus.map(0, 0, 0x1000_0000, md.uncached_view());
        bus.map(0, 0x1fc0_0000, 0x40_0000, Arc::new(Prom));
        crate::ip35::map_device(md.physical(), 0x1fc0_0000, 0x40_0000, Arc::new(Prom), "test reset PROM");
        let cached = bus.cached_view(md.dma_view(), md.ram_extent());
        (bus, cached)
    }

    /// The Tezro's default population, wired and configured as the board does.
    fn default_tezro() -> (Arc<BedrockMd>, crate::ip35::Ip35Bus, Arc<dyn BusDevice>) {
        let md = BedrockMd::new(&super::super::bedrock_mem::MemoryPopulation::default().banks_mib()).unwrap();
        assert_eq!(md.md_bus().write64(reg::CONFIG, TEZRO_CONFIG), BUS_OK);
        let (bus, cached) = board(&md);
        (md, bus, cached)
    }
    /// Asserts a code page's counter moved, in builds that keep counters.
    fn assert_advanced(what: &str, before: u64, after: u64) {
        if cfg!(feature = "jitv2") {
            assert!(after > before, "{what}: {before} -> {after}");
        }
    }

    /// A processor store to banks 4 and 5, above the 4 GiB the direct window
    /// reaches, goes through the board's cached view or an uncached one into
    /// the bank behind the slot and advances that slot's logical counter,
    /// and no other slot's. The JIT page lookup and the cache's own bump read
    /// the same counter.
    #[test]
    fn processor_stores_to_banks_4_and_5_advance_their_slot_counters() {
        let (md, bus, cached) = default_tezro();
        let uncached = |address: u64| (3u64 << 57) | address;
        for (slot, other) in [(4u64, 5u64), (5, 4)] {
            let at = slot * BANK_SPAN + 0x2000;
            let elsewhere = counter(&md, other * BANK_SPAN + 0x2000);
            #[cfg(feature = "jitv2")]
            {
                assert_eq!(bus.gen_ptr(at), md.memory.gen_ptr(at));
                assert_eq!(cached.gen_ptr(at), md.memory.gen_ptr(at));
                assert_eq!(bus.gen_ptr(uncached(at)), md.memory.gen_ptr(at));
            }
            let mut last = counter(&md, at);
            let mut stored = |what: &str, status: u32| {
                assert_eq!(status, BUS_OK, "{what}");
                let now = counter(&md, at);
                assert_advanced(what, last, now);
                last = now;
            };
            stored("cached store", cached.write64(at, 0x1111));
            stored("cached block store", cached.write_block(at + 8, &[0x2222, 0x3333]));
            stored("cached masked store", cached.write64_masked(at + 0x20, 0x4444, 0xff00));
            stored("cached halfword store", cached.write16(at + 0x30, 0x5555));
            stored("uncached store", bus.write64(uncached(at + 0x40), 0x6666));
            stored("uncached byte store", bus.write8(uncached(at + 0x48), 0x77));
            let bank = &md.physical().banks()[slot as usize];
            let word = |offset: u64| bank.read64(0x2000 + offset).data;
            assert_eq!([word(0), word(8), word(0x10), word(0x20), word(0x40)],
                [0x1111, 0x2222, 0x3333, 0x4400, 0x6666], "slot {slot}");
            assert_eq!(counter(&md, other * BANK_SPAN + 0x2000), elsewhere, "slot {other} untouched");
        }
    }

    /// A checked page above 4 GiB keeps its logical counter honest: marking
    /// it advances the counter, a store through the page's slow route does
    /// too, and so does the store once the page is back on its bank.
    #[test]
    fn a_checked_page_above_4_gib_advances_its_counter() {
        let (md, _bus, cached) = default_tezro();
        let (page, address) = (4 * BANK_SPAN + 0x6000, 4 * BANK_SPAN + 0x6008);
        let untouched = counter(&md, 4 * BANK_SPAN + 0x8000);
        let start = counter(&md, page);
        mark(&md, page);
        let marked = counter(&md, page);
        assert_advanced("mark", start, marked);
        assert!(crate::ppmem::devmap::entry_eq(md.physical().entry_at(address), md.physical().entry_at(page)));
        assert!(!crate::ppmem::devmap::entry_eq(md.physical().entry_at(page), md.physical().entry_at(page + 0x2000)));
        assert_eq!(cached.write64(address, 0xaa), BUS_OK);
        let stored = counter(&md, page);
        assert_advanced("store to the checked page", marked, stored);
        lift(&md, page);
        md.retire_lifted_marks();
        let retired = counter(&md, page);
        assert_advanced("retirement", stored, retired);
        assert_eq!(cached.write64(address, 0xbb), BUS_OK);
        assert_advanced("store after retirement", retired, counter(&md, page));
        assert_eq!(md.physical().banks()[4].read64(0x6008).data, 0xbb);
        assert_eq!(counter(&md, 4 * BANK_SPAN + 0x8000), untouched, "a neighbouring page keeps its counter");
    }

    /// After a bank swap the slot's counter, which does not move, is the one
    /// a store advances, whichever bank now sits behind the slot, and the
    /// slot the bank left is not touched.
    #[test]
    fn a_store_after_a_bank_swap_advances_the_counter_of_the_slot_it_reached() {
        let (md, bus, cached) = default_tezro();
        let (four, five) = (4 * BANK_SPAN + 0x2000, 5 * BANK_SPAN + 0x2000);
        #[cfg(feature = "jitv2")]
        let pointers = (md.memory.gen_ptr(four), md.memory.gen_ptr(five));
        // Bit 32 swaps the banks of a pair: slot 4 now reaches bank 5.
        assert_eq!(md.md_bus().write64(reg::CONFIG, TEZRO_CONFIG | (1 << 32)), BUS_OK);
        #[cfg(feature = "jitv2")]
        assert_eq!((md.memory.gen_ptr(four), md.memory.gen_ptr(five)), pointers);
        let (before_four, before_five) = (counter(&md, four), counter(&md, five));
        assert_eq!(cached.write64(four, 0x44), BUS_OK);
        assert_eq!(md.physical().banks()[5].read64(0x2000).data, 0x44, "slot 4 reaches bank 5");
        assert_eq!(md.physical().banks()[4].read64(0x2000).data, 0);
        let after_four = counter(&md, four);
        assert_advanced("cached store to slot 4", before_four, after_four);
        assert_eq!(counter(&md, five), before_five);
        assert_eq!(bus.write64((3u64 << 57) | five, 0x55), BUS_OK);
        assert_eq!(md.physical().banks()[4].read64(0x2000).data, 0x55, "slot 5 reaches bank 4");
        assert_advanced("uncached store to slot 5", before_five, counter(&md, five));
        assert_eq!(counter(&md, four), after_four);
    }

    /// With bank 0 at 1 GiB the RAM extent covers the cached PROM window at
    /// 0x1fc0_0000, which a Fuel's 256 MiB banks leave to the PROM. Once
    /// bank 0 is enabled the cached plane, the uncached RAM view and DMA all
    /// see RAM there, and a cached store lands in it; the PROM answers
    /// through Hub-special space only. Before bank 0 is enabled nothing is
    /// RAM, the cached plane shows the PROM and a store is refused.
    ///
    /// The QEMU model agrees for those views and differs for one: its reset
    /// window is routed by cache attribute, and a noncoherent cached access
    /// (what the stock PROM runs with) keeps seeing the PROM after RAM is
    /// enabled (hw/mips/sgi/sn1-node.c, sn1_reset_attrs_to_index and
    /// sn1_node_map_reset, "Compatibility policy"). The cache here passes
    /// no attribute to the bus, so a noncoherent cached access sees RAM.
    #[test]
    fn a_1_gib_bank_0_covers_the_cached_prom_window_with_ram() {
        let md = BedrockMd::new(&TEZRO_BANKS).unwrap();
        let (bus, cached) = board(&md);
        let at = 0x1fc0_0000;
        let prom = 0x0f1e_2d3c_4b5a_6978;
        let uncached = (3u64 << 57) | at;
        assert_eq!(cached.read64(at).data, prom, "before RAM is enabled");
        assert_eq!(cached.write64(at, 0x55), BUS_ERR);
        assert_eq!(md.md_bus().write64(reg::CONFIG, TEZRO_CONFIG), BUS_OK);
        assert_eq!(bus.read64(at).data, prom, "Hub-special space still shows the PROM");
        for (what, value) in [("cached", cached.read64(at).data), ("uncached RAM", bus.read64(uncached).data),
            ("DMA", md.dma_view().read64(at).data)] {
            assert_eq!(value, 0, "{what} sees RAM");
        }
        assert!(md.physical().mem_ptr(at).is_some(), "the cache fills from RAM too");
        let before = counter(&md, at);
        assert_eq!(cached.write64(at, 0x55), BUS_OK);
        assert_eq!(md.physical().banks()[0].read64(at).data, 0x55);
        assert_advanced("cached store", before, counter(&md, at));
        assert_eq!(bus.read64(at).data, prom, "the PROM is not written");
        // The Fuel's 256 MiB banks never reach it.
        let fuel = BedrockMd::new(&FUEL_BANKS).unwrap();
        let (_bus, cached) = board(&fuel);
        assert_eq!(fuel.md_bus().write64(reg::CONFIG, 0x33), BUS_OK);
        assert_eq!(cached.read64(at).data, prom);
        assert_eq!(cached.write64(at, 0x55), BUS_ERR);
    }

    /// Physical's bulk writers, on a bank directly, advance only that bank's
    /// counters, which nothing that follows code pages reads on this board.
    /// The controller's own routes advance the logical ones as well.
    #[test]
    fn bulk_memory_writes_through_the_controller_advance_the_logical_counters() {
        let md = BedrockMd::new(&[64]).unwrap();
        assert_eq!(md.md_bus().write64(reg::CONFIG, 0x11), BUS_OK);
        let phys = md.physical();
        let at = 0x5000;
        let logical = counter(&md, at);
        phys.reset_memory();
        phys.restore_bank_inmem(0, &[1, 2]);
        assert_eq!(counter(&md, at), logical, "Physical alone leaves the logical counter");
        md.reset_memory();
        let reset = counter(&md, at);
        assert_advanced("reset", logical, reset);
        md.restore_bank(0, &[0x0102_0304, 0x0506_0708]);
        let restored = counter(&md, at);
        assert_advanced("restore", reset, restored);
        assert_eq!(phys.read64(0).data, 0x0102_0304_0506_0708);
        let path = std::env::temp_dir().join(format!("iris-bedrock-load-{}", std::process::id()));
        std::fs::write(&path, [0, 0, 0, 9, 0, 0, 0, 8]).unwrap();
        let loaded = md.load_bank(0, &path);
        std::fs::remove_file(&path).unwrap();
        loaded.unwrap();
        assert_advanced("load", restored, counter(&md, at));
        assert_eq!(phys.read64(0).data, 0x0000_0009_0000_0008);
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
        md.soft_reset();
        assert_eq!(bus.read64(0).data, 0xabcdef);
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
            assert_eq!(view.read64(0).status, BUS_ERR);
        }

    }
    /// The guest reads ECC configuration and memory or directory errors through status registers.
    #[test]
    fn guest_registers_expose_ecc_configuration_and_error_latches() {
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
            // Error-clear writes are ignored too: only a read clears the record.
            md.ecc_bus().write8(7, 3);
            md.uncached_view().read64(0);
            r.write64(0x40000, 2);
            md.directory_bus().write64(0, 3 << 16);
            r.write64(0x40000, 0);
            md.physical().read64(0);
            let memory_error = r.read64(0x60010).data;
            let directory_error = r.read64(0x40010).data;
            assert_ne!(memory_error & (1 << 60), 0);
            assert_ne!(directory_error & (1 << 63), 0);
            for at in [0x40010, 0x40018, 0x60010, 0x60018] {
                let before = md.unsupported();
                assert_eq!(r.write64(at, u64::MAX), BUS_OK);
                assert_eq!(r.read64(0x60010).data, memory_error);
                assert_eq!(r.read64(0x40010).data, directory_error);
                assert_eq!(md.unsupported(), before + 1);
            }
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
            md.soft_reset();
            assert_eq!(regs.read64(0x60000).data, 0);
            assert_eq!(regs.read64(8).data, 0x33);
            regs.write64(0x60000, 1);
            md.power_on();
            assert_eq!(regs.read64(0x60000).data, 0);
            assert_eq!(regs.read64(8).data, 0);
        }
        {
            for (code, kind, status, valid) in [
                (1, MemoryError::Correctable, BUS_OK, 1 << 52),
                (3, MemoryError::Uncorrectable, BUS_ERR, 1 << 60),
            ] {
                let md = configured();
                let regs = md.md_bus();
                let ecc = md.ecc_bus();
                let ram = md.uncached_view();
                let events = Arc::new(Mutex::new(Vec::new()));
                let seen = events.clone();
                md.on_error(Arc::new(move |event| seen.lock().push(event)));
                assert_eq!(ram.read64(0).status, BUS_OK);
                ecc.write8(7, code);
                regs.write64(0x60000, u64::MAX);
                assert_eq!(ram.read64(0).status, BUS_OK);
                assert_eq!(ecc.read8(7).data, code);
                assert_eq!(regs.read64(0x60010).data, 0);
                assert!(events.lock().is_empty());
                regs.write64(0x60000, !1);
                assert_eq!(regs.read64(0x60000).data, 0);
                assert_eq!(ram.read64(0).status, status);
                assert_ne!(regs.read64(0x60010).data & valid, 0);
                assert_eq!(*events.lock(), vec![kind]);
                if kind == MemoryError::Correctable {
                    assert_eq!(ecc.read8(7).data, 0);
                    assert_eq!(ram.read64(0).status, BUS_OK);
                    assert_eq!(*events.lock(), vec![kind]);
                }
            }
        }
        {
            let md = configured();
            let regs = md.md_bus();
            let ecc = md.ecc_bus();
            let ram = md.uncached_view();
            ecc.write8(7, 3);
            regs.write64(0x60000, 1);
            assert_eq!(ram.write8(7, 0x5a), BUS_OK);
            assert_eq!(ecc.read8(7).data, 0x82);
            regs.write64(0x60000, 0);
            assert_eq!(ram.read64(0).status, BUS_OK);
            assert_eq!(ram.read64(0).data, 0x5a);
            assert_eq!(regs.read64(0x60010).data, 0);
        }
    }
    /// Self-test completion is visible only after its configured clock interval.
    #[test]
    fn selftest_waits_for_fitted_clock_cycles_and_zero_fills_ram() {
        {
            let now = Arc::new(AtomicU64::new(100));
            let time = now.clone();
            let md = BedrockMd::with_clock(&FUEL_BANKS, Arc::new(move || time.load(Ordering::Relaxed))).unwrap();
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
            let md = configured();
            let r = md.md_bus();
            r.write64(8, 0x13);
            md.physical().write64(0, 0x1234);
            md.ecc_bus().write8(7, 1);
            r.write64(0x20038, 0x80);
            r.write64(0x20030, 1);
            md.advance_ns(1000);
            assert_eq!(r.read64(0x20048).data, 3);
            assert_eq!(md.physical().read64(0).data, 0);
            assert_eq!(md.ecc_bus().read8(7).data, 0);
            assert_eq!(r.read64(0x60010).data, 0);
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
    /// The guest reads ECC bytes and scrubbing repairs correctable memory.
    #[test]
    fn ecc_carriers_expose_check_bytes_and_scrub_corrections() {
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
            let events = Arc::new(Mutex::new(Vec::new()));
            let seen = events.clone();
            let weak = Arc::downgrade(&md);
            md.on_error(Arc::new(move |kind| {
                assert_eq!(weak.upgrade().unwrap().current_cell(), u64::MAX);
                seen.lock().push(kind);
            }));
            assert_eq!(md.ecc_bus().write8(7, 3), BUS_OK);
            assert_eq!(md.ecc_bus().write8(6, 0x13), BUS_OK);
            assert_eq!(md.ecc_bus().write8(5, 0x23), BUS_OK);
            assert_eq!(md.ecc_scrub_bus().write64(0, 0), BUS_OK);
            assert_eq!(
                *events.lock(),
                vec![
                    MemoryError::Uncorrectable,
                    MemoryError::Correctable,
                    MemoryError::Correctable,
                ]
            );
            let record = md.md_bus().read64(0x60010).data;
            assert_ne!(record & (1 << 60), 0);
            assert_ne!(record & (1 << 52), 0);
            assert_ne!(record & (1 << 53), 0);
            assert_eq!((record >> 40) & 0xff, 3);
            assert_eq!(record & (((1u64 << 30) - 1) << 3), 0);
            assert_eq!(md.ecc_bus().read8(6).data, 0x13);
            assert_eq!(md.ecc_bus().read8(5).data, 0x23);
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
    /// Uncached CPU and DMA views report uncorrectable errors with bus status.
    #[test]
    fn uncached_and_dma_views_report_errors_to_the_guest() {
        let md = configured();
        let r = md.md_bus();
        let e = md.ecc_bus();
        e.write8(7, 3);
        assert_eq!(md.uncached_view().read64(0).status, BUS_ERR);
        assert_eq!(md.dma_view().read64(0).status, BUS_OK);
        assert_ne!(r.read64(0x60018).data & (1 << 60), 0);
        e.write8(7, 0x13);
        assert_eq!(md.ecc_scrub_bus().write64(0, 0), BUS_OK);
        assert_eq!(md.physical().read64(0).data, 1);
        assert_eq!(e.read8(7).data, 0x13);
        assert_eq!(md.ecc_scrub_bus().write32(0, 0), BUS_ERR);
    }
    /// Directory reads report invalid metadata and protection faults to the guest.
    #[test]
    fn directory_reads_check_metadata_and_report_protection_errors() {
        {
            let md = configured();
            let events = Arc::new(Mutex::new(Vec::new()));
            let seen = events.clone();
            let weak = Arc::downgrade(&md);
            md.on_error(Arc::new(move |kind| {
                assert_eq!(weak.upgrade().unwrap().current_cell(), u64::MAX);
                seen.lock().push(kind);
            }));
            let r = md.md_bus();
            md.directory_bus().write64(0, 1);
            assert_eq!(md.directory_bus().read64(0).data, 0x001f0001);
            r.write64(0x40000, 2);
            md.directory_bus().write64(0, 0x001e0001);
            r.write64(0x40000, 0);
            md.ecc_bus().write8(7, 3);
            assert_eq!(md.uncached_view().read64(0).status, BUS_ERR);
            assert_eq!(md.directory_bus().read64(0).data, 0x001f0001);
            assert_eq!(
                *events.lock(),
                vec![MemoryError::Correctable, MemoryError::Uncorrectable]
            );
            assert_ne!(r.read64(0x40010).data & (1 << 61), 0);
            assert_ne!(r.read64(0x60010).data & (1 << 60), 0);
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
