//! The Fuel's fitted memory: two 256 MiB banks owned by `Physical`, with
//! the bank decode selected by Bedrock's memory configuration register.
//!
//! This population follows the QEMU model; it is not a measured Fuel
//! topology. Ordinary pages answer through Physical's bank views, without
//! the MD state lock. Pages with injected ECC or marked directory lines use
//! the MD endpoint. Their entire 64 MiB region is absent from the direct
//! window, so cache fills cannot bypass those checks.
//!
//! Remapping runs on the CPU thread, or with the CPU stopped, as HEART's
//! memory descriptors do. DMA may remove diagnostic metadata but retains
//! the slow mapping until the next CPU-side rebuild. Physical publishes
//! complete decode-table pages atomically; a concurrent bus master sees an
//! old or new entry. This does not promise a transaction across a remap.
//!
//! Ported from hw/misc/sgi-bedrock-md.c in jamesbraid/qemu-sgi-origami.

use crate::physical::{Physical, RamBank};
use crate::traits::*;
use crate::ppmem::devmap::SLOT_SHIFT;
use crate::ppmem::{MappedMemory, BITMAP_SHIFT};
use parking_lot::Mutex;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

pub const BANK_SPAN: u64 = 1 << 30;
pub const FITTED_BANK: u64 = 256 << 20;

/// The PROM builds its stack and data in primary data cache lines it tags for
/// this node-local window, with no memory behind it: DRAM is not configured
/// yet. See [`BedrockMemory::startup_tag`].
pub(crate) const STARTUP_BASE: u64 = 0x10_0000;
pub(crate) const STARTUP_SIZE: u64 = 0x8000;
/// A primary data cache tag: the state in bits 7:6, the physical address
/// bits 35:12 above bit 8. Bits above the node's 8 GiB (page bit 21) select
/// the node.
const TAG_STATE: u32 = 0xc0;
const TAG_PAGE_SHIFT: u32 = 8;
const TAG_NODE_PAGES: u32 = 1 << 21;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Run {
    bank: usize,
    base: u64,
    size: u64,
}

#[derive(Default)]
struct Mapping {
    runs: Vec<Run>,
    /// Whether the decode table's slots over the startup window hold the page.
    startup_placed: bool,
    // Bank-relative 4 KiB page numbers, bank 1 immediately after bank 0.
    slow_pages: BTreeSet<u64>,
    slow: Option<Arc<dyn BusDevice>>,
}

struct NoMemory;
impl BusDevice for NoMemory {}

/// JIT pages belong to physical addresses, not to the bank currently
/// answering there. Keep these counters separate from the banks' counters:
/// a compiled page pins its pointer across later bank reconfiguration.
#[cfg(feature = "jitv2")]
pub(crate) struct Generations(Box<[AtomicU64]>);
#[cfg(feature = "jitv2")]
impl Generations {
    fn new() -> Self {
        Self((0..1 << 20).map(|_| AtomicU64::new(0)).collect::<Vec<_>>().into_boxed_slice())
    }
    pub(crate) fn base(&self) -> *mut AtomicU64 {
        self.0.as_ptr() as *mut AtomicU64
    }
    fn bump(&self, address: u64) {
        if let Some(counter) = self.0.get((address >> 12) as usize) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn bump_all(&self) {
        for slot in 0..4 {
            let first = slot * (BANK_SPAN >> 12) as usize;
            for counter in &self.0[first..first + (FITTED_BANK >> 12) as usize] {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

pub struct BedrockMemory {
    physical: Arc<Physical>,
    config: AtomicU64,
    mapping: Mutex<Mapping>,
    /// The window's bytes while the PROM's cache lines stand. Zero at power-on:
    /// the PROM makes the lines valid without a fill.
    startup_page: Box<RamBank>,
    startup_live: AtomicBool,
    #[cfg(feature = "jitv2")]
    generations: Arc<Generations>,
}
impl BedrockMemory {
    pub fn new() -> Self {
        let physical = Physical::new(vec![RamBank::new(256), RamBank::new(256)]);
        physical.set_default(Arc::new(NoMemory), "Bedrock unconfigured memory");
        let startup_page = Box::new(RamBank::new(1));
        startup_page.set_addr_mask(STARTUP_SIZE as u32 - 1);
        Self { physical, config: AtomicU64::new(0), mapping: Mutex::new(Mapping::default()),
            startup_page, startup_live: AtomicBool::new(false),
            #[cfg(feature = "jitv2")]
            generations: Arc::new(Generations::new()),
        }
    }
    pub fn physical(&self) -> &Arc<Physical> {
        &self.physical
    }
    pub(crate) fn config(&self) -> u64 {
        self.config.load(Ordering::Acquire)
    }

    /// Whether a primary data cache tag store starts or ends the startup
    /// window, given its TagLo. PROM 6.210 stores test patterns (state bits
    /// set, pages far from the window), an invalidate sweep, then one valid
    /// tag for each of the window's eight pages, and a final invalidate sweep
    /// once it has memory. So a valid tag for a page of the window starts it
    /// and the next invalid one ends it; a kernel stores invalid tags only.
    ///
    /// The end is the sweep's first store, not its last. Counting accesses
    /// that reach the page in an interpreted PROM 6.210 boot, none follow the
    /// first of the final sweep's 2050 invalid stores (the count is the same
    /// at the first and at the last), and the page is read and written
    /// heavily up to that first store. Holding it to the sweep's end would
    /// change nothing the PROM sees.
    /// `Some(live)` is the state to move to, `None` leaves it alone.
    pub(crate) fn startup_tag(&self, tag: u32) -> Option<bool> {
        let live = self.startup_live.load(Ordering::Relaxed);
        if tag & TAG_STATE == 0 {
            return live.then_some(false);
        }
        let page = (tag >> TAG_PAGE_SHIFT) & (TAG_NODE_PAGES - 1);
        let in_window = (STARTUP_BASE >> 12..(STARTUP_BASE + STARTUP_SIZE) >> 12).contains(&(page as u64));
        (in_window && !live).then_some(true)
    }
    /// Move to `live`; whether that changed anything, and so needs a remap.
    pub(crate) fn set_startup(&self, live: bool) -> bool {
        self.startup_live.swap(live, Ordering::Relaxed) != live
    }
    pub(crate) fn startup_live(&self) -> bool {
        self.startup_live.load(Ordering::Relaxed)
    }
    /// Power-on: no cache lines stand, and the page holds zero.
    pub(crate) fn clear_startup(&self) {
        self.startup_live.store(false, Ordering::Relaxed);
        for at in (0..STARTUP_SIZE).step_by(8) {
            self.startup_page.write64(at, 0);
        }
    }
    /// The page, for an access it answers: the window, while live. The bus
    /// decode, the CPU's cached path and DMA all reach it through here or
    /// through Physical's table, since nothing else is behind the window.
    pub(crate) fn startup_device(&self, address: u64, bytes: u64) -> Option<&dyn BusDevice> {
        let offset = address.checked_sub(STARTUP_BASE)?;
        (self.startup_live() && offset + bytes <= STARTUP_SIZE)
            .then(|| &*self.startup_page as &dyn BusDevice)
    }
    #[cfg(feature = "jitv2")]
    pub(crate) fn generations(&self) -> Arc<Generations> { self.generations.clone() }
    #[cfg(feature = "jitv2")]
    pub(crate) fn gen_ptr(&self, address: u64) -> *const AtomicU64 {
        // Disabled RAM must still have a counter: a JIT page may already
        // have claimed it before the guest enables or moves a bank.
        if address / BANK_SPAN < 4 && address % BANK_SPAN < FITTED_BANK {
            unsafe { self.generations.base().add((address >> 12) as usize) }
        } else { std::ptr::null() }
    }
    pub(crate) fn written(&self, _address: u64) {
        #[cfg(feature = "jitv2")]
        self.generations.bump(_address);
    }

    /// The endpoint is weak back to MD: Physical owns it without a cycle.
    pub(crate) fn set_slow_view(&self, slow: Arc<dyn BusDevice>) {
        self.mapping.lock().slow = Some(slow);
    }

    fn runs(config: u64) -> Vec<Run> {
        (0..8).filter_map(|slot| {
            let base = slot * BANK_SPAN;
            let (bank, _) = Self::decode(config, base, 1)?;
            let code = (config >> (bank / 2 * 8 + 4)) & 15;
            Some(Run { bank, base, size: FITTED_BANK.min((32 << 20) << code) })
        }).collect()
    }

    /// Publish a state snapshot after releasing MD's lock. Only the CPU
    /// thread, or a caller with the CPU stopped, may change ppmem's bitmap.
    /// A forced rebuild follows the machine setting Physical's default.
    pub(crate) fn remap(&self, config: u64, slow_pages: BTreeSet<u64>, force: bool) {
        let runs = Self::runs(config);
        let mut old = self.mapping.lock();
        if !force && old.runs == runs && old.slow_pages == slow_pages {
            self.config.store(config, Ordering::Release);
            return;
        }
        let phys = &self.physical;
        let space = phys.ppmem_space();
        // Withdraw the direct window before publishing any checked page.
        if let Some(space) = space {
            space.clear_mappings();
        }
        // Restore only ranges previously owned by RAM. In particular, the
        // unused part of a 1 GiB slot may contain flash or other devices.
        if old.startup_placed {
            phys.fill_slots(STARTUP_BASE >> SLOT_SHIFT, STARTUP_SIZE >> SLOT_SHIFT, phys.default_entry());
        }
        for run in &old.runs {
            let retained = runs.iter().find(|next| next.base == run.base)
                .map_or(0, |next| next.size.min(run.size));
            if retained < run.size {
                phys.fill_slots((run.base + retained) >> SLOT_SHIFT,
                    (run.size - retained) >> SLOT_SHIFT, phys.default_entry());
            }
        }
        for run in &runs {
            phys.fill_slots(run.base >> SLOT_SHIFT, run.size >> SLOT_SHIFT, phys.bank_view(run.bank, run.base));
            let bank_page = run.bank as u64 * (FITTED_BANK >> SLOT_SHIFT);
            let pages = run.size >> SLOT_SHIFT;
            let mut checked_regions = BTreeSet::new();
            for &page in slow_pages.range(bank_page..bank_page + pages) {
                let offset = (page - bank_page) << SLOT_SHIFT;
                let address = run.base + offset;
                checked_regions.insert(offset >> BITMAP_SHIFT);
                phys.map(address, address + (1 << SLOT_SHIFT),
                    old.slow.as_ref().expect("Bedrock MD slow endpoint").clone(), "Bedrock checked RAM");
            }
            if let Some(space) = space {
                let region_size = 1 << BITMAP_SHIFT;
                for offset in (0..run.size).step_by(region_size as usize) {
                    // The startup page answers through the decode only, so the
                    // region holding it stays out of the direct window.
                    let startup = self.startup_live() && run.base + offset == 0;
                    if !startup && !checked_regions.contains(&(offset >> BITMAP_SHIFT)) {
                        if let Err(error) = space.map_range(run.bank, offset, run.base + offset, region_size) {
                            crate::dlog!(crate::devlog::LogModule::Bedrock,
                                "Bedrock RAM window at {:#x}: {error}", run.base + offset);
                        }
                    }
                }
            }
        }
        let live = self.startup_live();
        if live {
            let page: *const RamBank = &*self.startup_page;
            phys.fill_slots(STARTUP_BASE >> SLOT_SHIFT, STARTUP_SIZE >> SLOT_SHIFT,
                page as crate::ppmem::devmap::Entry);
        }
        old.startup_placed = live;
        self.config.store(config, Ordering::Release);
        // Publish changed ownership before advancing the stable counters,
        // so an in-flight compile cannot accept bytes from the old decode.
        self.changed_decode();
        old.runs = runs;
        old.slow_pages = slow_pages;
    }

    pub(crate) fn decode(config: u64, addr: u64, bytes: u64) -> Option<(usize, u64)> {
        let software = addr / BANK_SPAN;
        if software >= 8 {
            return None;
        }
        let bank = software ^ ((config >> 32) & 3);
        let field = config >> (bank / 2 * 8);
        let code = (field >> 4) & 15;
        if bank >= 2 || field & (1 << (bank & 1)) == 0 || !(1..=5).contains(&code) {
            return None;
        }
        let limit = FITTED_BANK.min((32 << 20) << code);
        let offset = addr % BANK_SPAN;
        (offset < limit && bytes <= limit - offset).then_some((bank as usize, offset))
    }
    pub(crate) fn raw_read(&self, bank: usize, at: u64) -> u64 {
        self.physical.banks()[bank].read64(at).data
    }
    pub(crate) fn raw_write(&self, bank: usize, at: u64, value: u64) -> u32 {
        let result = self.physical.banks()[bank].write64(at, value);
        if result == BUS_OK {
            let slot = bank as u64 ^ ((self.config() >> 32) & 3);
            self.written(slot * BANK_SPAN + at);
        }
        result
    }
    pub(crate) fn zero(&self, bank: usize, size: u64) {
        assert!(size <= FITTED_BANK);
        // The bank owns this contiguous mapping; zero has the same representation
        // in host word order. Self-test does not synchronize concurrent bank accesses.
        unsafe {
            std::ptr::write_bytes(self.physical.banks()[bank].host_ptr(), 0, size as usize);
        }
        #[cfg(feature = "jitv2")]
        self.physical.banks()[bank].bump_gen_all();
        #[cfg(feature = "jitv2")]
        self.generations.bump_all();
    }
    pub(crate) fn changed_decode(&self) {
        #[cfg(feature = "jitv2")]
        self.generations.bump_all();
        #[cfg(feature = "jitv2")]
        for bank in self.physical.banks() {
            bank.bump_gen_all();
        }
    }
}
