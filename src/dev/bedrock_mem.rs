//! A Bedrock node's fitted memory: banks owned by `Physical`, with the bank
//! decode selected by Bedrock's memory configuration register.
//!
//! The board names the fitted banks, locational bank 0 upward, one size in
//! MiB each: two 256 MiB banks on the Fuel, six on the Tezro. Each bank
//! answers in its own 1 GiB software slot, so a Tezro's banks 4 and 5 sit
//! at 4 to 6 GiB, above the 4 GiB window ppmem maps for the CPU's direct
//! access. Those two answer through Physical's decode table only, and
//! their code pages keep the logical generation counters like the rest.
//!
//! These populations follow the QEMU models; they are not measured
//! topologies. Ordinary pages answer through Physical's bank views, without
//! the MD state lock. Pages with injected ECC or marked directory lines use
//! the MD endpoint. Their entire 64 MiB region is absent from the direct
//! window, so cache fills cannot bypass those checks.
//!
//! Remapping runs on the CPU thread, or with the CPU stopped, as HEART's
//! memory descriptors do. A bank configuration change rebuilds the whole
//! map. Marking a page adds only that page: it takes its entry in the decode
//! table, and the first in a 64 MiB region takes the region out of the direct
//! window. Lifting a mark removes nothing. The page stays on the MD endpoint,
//! which answers as the bank would, until a batch retires every lifted page
//! at once: the PROM's memory test marks and lifts the same few hundred pages
//! thousands of times, and a map change for each made it take minutes in the
//! QEMU model. DMA may remove diagnostic metadata but retains the slow
//! mapping until a later retirement. Physical publishes complete
//! decode-table pages atomically; a concurrent bus master sees an old or new
//! entry. This does not promise a transaction across a remap.
//!
//! Ported from hw/misc/sgi-bedrock-md.c in jamesbraid/qemu-sgi-origami.

use crate::physical::{Physical, RamBank};
use crate::traits::*;
use crate::ppmem::devmap::SLOT_SHIFT;
use crate::ppmem::{MappedMemory, BITMAP_SHIFT, WINDOW_SIZE};
use parking_lot::Mutex;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

pub const BANK_SPAN: u64 = 1 << 30;
/// Software slots of [`BANK_SPAN`] in the node's 8 GiB.
pub const SLOTS: usize = 8;
/// The most memory a bank can hold: what the largest size code names.
const MAX_BANK_MIB: u32 = 1024;
/// Banks are mapped into the direct window in whole regions of this size.
const REGION_MIB: u32 = 1 << (BITMAP_SHIFT - 20);

/// A DIMM size the Tezro's IP53 board takes. One DIMM makes one bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankSize {
    Mib512,
    Mib1024,
}
impl BankSize {
    pub const fn mib(self) -> u32 {
        match self {
            Self::Mib512 => 512,
            Self::Mib1024 => 1024,
        }
    }
    fn from_mib(mib: u32) -> Result<Self, String> {
        match mib {
            512 => Ok(Self::Mib512),
            1024 => Ok(Self::Mib1024),
            _ => Err(format!("a Tezro bank is 512 or 1024 MiB, not {mib} MiB")),
        }
    }
}

/// The banks a Tezro fits, locational bank 0 upward: up to eight DIMMs of 512
/// or 1024 MiB with none left out between them. What the board needs in total
/// depends on its processors; see [`Self::check_for_cpus`].
///
/// The bank sizes and the minimums come from the Tezro manuals as the QEMU
/// assembly layer records them. The default is the 4 GiB layout of 1024, 1024
/// and four 512 MiB banks, a population a real Tezro has been observed to hold.
/// Its banks 4 and 5 lie above 4 GiB, outside ppmem's direct window, so they
/// answer through Physical's decode table, which is slower.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPopulation(Vec<BankSize>);
impl Default for MemoryPopulation {
    fn default() -> Self {
        Self::four_gib()
    }
}
impl MemoryPopulation {
    pub fn new(banks: &[BankSize]) -> Result<Self, String> {
        if banks.is_empty() || banks.len() > SLOTS {
            return Err(format!("memory: {} banks; a Tezro fits 1 to {SLOTS}", banks.len()));
        }
        Ok(Self(banks.to_vec()))
    }
    /// From sizes in MiB, rejecting any a Tezro does not take.
    pub fn from_mib(banks: &[u32]) -> Result<Self, String> {
        let sizes = banks.iter().map(|&mib| BankSize::from_mib(mib)).collect::<Result<Vec<_>, _>>()?;
        Self::new(&sizes)
    }
    /// 1 GiB as two banks of 512 MiB. The manual's minimum of 1 GB for two or
    /// four processors permits it, but no real Tezro has been observed with
    /// less than 4 GiB, and nothing here claims the PROM or IRIX boot at it. All
    /// of it lies below 4 GiB, so it is wholly inside ppmem's direct window.
    pub fn one_gib() -> Self {
        Self(vec![BankSize::Mib512; 2])
    }
    /// 4 GiB: 1024, 1024 and four 512 MiB banks, the default. Banks 4 and 5
    /// sit above 4 GiB, outside the direct window.
    pub fn four_gib() -> Self {
        let mut banks = vec![BankSize::Mib1024; 2];
        banks.extend([BankSize::Mib512; 4]);
        Self(banks)
    }
    pub fn banks(&self) -> &[BankSize] {
        &self.0
    }
    /// The sizes in MiB, as [`BedrockMd::new`](super::bedrock_md::BedrockMd::new) takes them.
    pub fn banks_mib(&self) -> Vec<u32> {
        self.0.iter().map(|bank| bank.mib()).collect()
    }
    pub fn total_mib(&self) -> u32 {
        self.0.iter().map(|bank| bank.mib()).sum()
    }
    /// Whether every bank lies below 4 GiB, so the CPU reaches all of it
    /// through the direct window.
    pub fn fits_direct_window(&self) -> bool {
        self.0.len() <= 4
    }
    /// One processor needs at least 512 MB; two or four need at least 1 GB.
    pub fn check_for_cpus(&self, cpus: usize) -> Result<(), String> {
        let needed = match cpus {
            1 => 512,
            2..=4 => 1024,
            _ => return Err(format!("{cpus} processors; a Tezro fits 1 to 4")),
        };
        if self.total_mib() < needed {
            return Err(format!(
                "{} MiB of memory; {cpus} processor{} need at least {needed} MiB",
                self.total_mib(), if cpus == 1 { "" } else { "s" }));
        }
        Ok(())
    }
}

/// Where fitted RAM can answer: for each software slot, the most any bank the
/// bank-select swizzle can place there holds. The swizzle permutes slots
/// within groups of four, so a slot may hold any bank of its group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamExtent([u64; SLOTS]);
impl RamExtent {
    /// `banks` are the fitted sizes in MiB from locational bank 0.
    pub fn new(banks: &[u32]) -> Self {
        let fitted = |bank: usize| banks.get(bank).map_or(0, |&mib| u64::from(mib) << 20);
        Self(std::array::from_fn(|slot| {
            (0..4).map(|group_slot| fitted(slot / 4 * 4 + group_slot)).max().unwrap_or(0)
        }))
    }
    /// Whether `address` can be fitted RAM under some bank-select setting.
    pub fn contains(&self, address: u64) -> bool {
        let slot = address / BANK_SPAN;
        slot < SLOTS as u64 && address % BANK_SPAN < self.0[slot as usize]
    }
    /// Slots from 0 through the last that can hold RAM.
    #[cfg(feature = "jitv2")]
    fn slots(&self) -> usize {
        self.0.iter().rposition(|&size| size != 0).map_or(0, |last| last + 1)
    }
    /// The most RAM software slot `slot` can hold, in bytes.
    pub fn prefix(&self, slot: usize) -> u64 { self.0[slot] }
}

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

/// Lifted pages kept on the MD endpoint before a batch retires them.
pub(crate) const RETIRED_LIMIT: usize = 1024;
/// How long a lifted page may idle before the next publication retires it:
/// a second of the controller's clock.
pub(crate) const RETIRED_IDLE_NS: u64 = 1_000_000_000;

#[derive(Default)]
struct Mapping {
    runs: Vec<Run>,
    /// Whether the decode table's slots over the startup window hold the page.
    startup_placed: bool,
    /// The pages that answer through the MD endpoint: those with a mark and
    /// those whose mark was lifted since the last retirement. Bank-relative
    /// 4 KiB page numbers, bank 1 immediately after bank 0.
    placed: BTreeSet<u64>,
    /// How many placed pages each 64 MiB region of the node's address space
    /// holds, for the regions inside a run.
    held: HashMap<u64, u32>,
    /// The regions mapped into the direct window now.
    windowed: HashSet<u64>,
    /// When the first lifted page was seen idle, on the controller's clock.
    idle_since: Option<u64>,
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
    /// One counter per 4 KiB page, up to the last slot that can hold RAM.
    fn new(extent: &RamExtent) -> Self {
        let pages = extent.slots() << (BANK_SPAN.trailing_zeros() - 12);
        Self((0..pages).map(|_| AtomicU64::new(0)).collect::<Vec<_>>().into_boxed_slice())
    }
    pub(crate) fn base(&self) -> *mut AtomicU64 {
        self.0.as_ptr() as *mut AtomicU64
    }
    fn bump(&self, address: u64) {
        if let Some(counter) = self.0.get((address >> 12) as usize) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn bump_all(&self, extent: &RamExtent) {
        for slot in 0..extent.slots() {
            let first = slot * (BANK_SPAN >> 12) as usize;
            let pages = (extent.prefix(slot) >> 12) as usize;
            for counter in &self.0[first..first + pages] {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// What a remap costs, counted where the work happens so a test can state
/// it without timing anything.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct Cost {
    /// Host mappings made or scrubbed in ppmem's direct window.
    pub window_ops: AtomicU64,
    /// Calls that rewrite decode-table slots.
    pub table_ops: AtomicU64,
    /// Generation counters incremented, logical and bank.
    pub bumps: AtomicU64,
}
#[cfg(test)]
impl Cost {
    pub fn add(counter: &AtomicU64, n: usize) {
        counter.fetch_add(n as u64, Ordering::Relaxed);
    }
    pub fn snapshot(&self) -> [u64; 3] {
        [&self.window_ops, &self.table_ops, &self.bumps].map(|c| c.load(Ordering::Relaxed))
    }
}

pub struct BedrockMemory {
    physical: Arc<Physical>,
    /// Fitted bytes in each locational bank, bank 0 first. Physical's banks
    /// are these, in the same order.
    fitted: Vec<u64>,
    config: AtomicU64,
    mapping: Mutex<Mapping>,
    /// The window's bytes while the PROM's cache lines stand. Zero at power-on:
    /// the PROM makes the lines valid without a fill.
    startup_page: Box<RamBank>,
    startup_live: AtomicBool,
    extent: RamExtent,
    #[cfg(feature = "jitv2")]
    generations: Arc<Generations>,
    #[cfg(test)]
    pub(crate) cost: Cost,
}
impl BedrockMemory {
    /// Memory with `banks` fitted, locational bank 0 upward, each a size in
    /// MiB: at most [`SLOTS`] banks, none empty or over 1 GiB (the largest
    /// size code), each a whole number of direct-window regions.
    pub fn new(banks: &[u32]) -> Result<Self, String> {
        if banks.is_empty() || banks.len() > SLOTS {
            return Err(format!("memory: {} banks; fit 1 to {SLOTS}", banks.len()));
        }
        if let Some((bank, mib)) = banks.iter().enumerate()
            .find(|(_, &mib)| mib == 0 || mib > MAX_BANK_MIB || mib % REGION_MIB != 0)
        {
            return Err(format!(
                "memory: bank {bank} is {mib} MiB; a bank is a multiple of {REGION_MIB} MiB up to {MAX_BANK_MIB} MiB"));
        }
        let physical = Physical::new(banks.iter().map(|&mib| RamBank::new(mib as usize)).collect());
        physical.set_default(Arc::new(NoMemory), "Bedrock unconfigured memory");
        let startup_page = Box::new(RamBank::new(1));
        startup_page.set_addr_mask(STARTUP_SIZE as u32 - 1);
        Ok(Self { physical, fitted: banks.iter().map(|&mib| u64::from(mib) << 20).collect(),
            config: AtomicU64::new(0), mapping: Mutex::new(Mapping::default()),
            startup_page, startup_live: AtomicBool::new(false),
            #[cfg(feature = "jitv2")]
            generations: Arc::new(Generations::new(&RamExtent::new(banks))),
            extent: RamExtent::new(banks),
            #[cfg(test)]
            cost: Cost::default(),
        })
    }
    /// Fitted bytes in locational bank `bank`; zero where nothing is fitted.
    pub(crate) fn fitted(&self, bank: u64) -> u64 {
        self.fitted.get(bank as usize).copied().unwrap_or(0)
    }
    /// Where the fitted banks start in a dense walk of all of them, bank 0
    /// first. Sparse tables key ECC, directory and protection state by it, so
    /// two banks never share a key.
    pub(crate) fn dense_base(&self, bank: u64) -> u64 {
        self.fitted.iter().take(bank as usize).sum()
    }
    pub fn extent(&self) -> RamExtent { self.extent }
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
        if self.extent.contains(address) {
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

    fn runs(&self, config: u64) -> Vec<Run> {
        (0..SLOTS as u64).filter_map(|slot| {
            let base = slot * BANK_SPAN;
            let (bank, _) = self.decode(config, base, 1)?;
            let code = (config >> (bank / 2 * 8 + 4)) & 15;
            Some(Run { bank, base, size: self.fitted(bank as u64).min((32 << 20) << code) })
        }).collect()
    }

    /// Publish a state snapshot after releasing MD's lock. Only the CPU
    /// thread, or a caller with the CPU stopped, may change ppmem's bitmap.
    /// A forced rebuild follows the machine setting Physical's default.
    /// This is the whole-map rebuild: `pages` replaces every placed page, so
    /// it retires the lifted ones too. Without `force` it does nothing when
    /// neither the runs nor the placed pages would change.
    pub(crate) fn remap(&self, config: u64, pages: BTreeSet<u64>, force: bool) {
        let runs = self.runs(config);
        let mut old = self.mapping.lock();
        if !force && old.runs == runs && old.placed == pages {
            self.config.store(config, Ordering::Release);
            return;
        }
        self.rebuild(&mut old, config, runs, pages);
    }

    /// The bank configuration register changed to `config`. Rebuild, with
    /// the marked `pages`, only if it changes which banks answer where.
    pub(crate) fn reconfigure(&self, config: u64, pages: impl FnOnce() -> BTreeSet<u64>) {
        let runs = self.runs(config);
        let mut old = self.mapping.lock();
        if old.runs == runs {
            self.config.store(config, Ordering::Release);
            return;
        }
        self.rebuild(&mut old, config, runs, pages());
    }

    fn rebuild(&self, old: &mut Mapping, config: u64, runs: Vec<Run>, pages: BTreeSet<u64>) {
        let phys = &self.physical;
        let space = phys.ppmem_space();
        // Withdraw the direct window before publishing any checked page.
        if let Some(space) = space {
            #[cfg(test)]
            Cost::add(&self.cost.window_ops, space.mapping_count());
            space.clear_mappings();
        }
        // Restore only ranges previously owned by RAM. In particular, the
        // unused part of a 1 GiB slot may contain flash or other devices.
        if old.startup_placed {
            #[cfg(test)]
            Cost::add(&self.cost.table_ops, 1);
            phys.fill_slots(STARTUP_BASE >> SLOT_SHIFT, STARTUP_SIZE >> SLOT_SHIFT, phys.default_entry());
        }
        for run in &old.runs {
            let retained = runs.iter().find(|next| next.base == run.base)
                .map_or(0, |next| next.size.min(run.size));
            if retained < run.size {
                #[cfg(test)]
                Cost::add(&self.cost.table_ops, 1);
                phys.fill_slots((run.base + retained) >> SLOT_SHIFT,
                    (run.size - retained) >> SLOT_SHIFT, phys.default_entry());
            }
        }
        old.held.clear();
        old.windowed.clear();
        old.idle_since = None;
        for run in &runs {
            #[cfg(test)]
            Cost::add(&self.cost.table_ops, 1);
            phys.fill_slots(run.base >> SLOT_SHIFT, run.size >> SLOT_SHIFT, phys.bank_view(run.bank, run.base));
            let bank_page = self.dense_base(run.bank as u64) >> SLOT_SHIFT;
            let run_pages = run.size >> SLOT_SHIFT;
            for &page in pages.range(bank_page..bank_page + run_pages) {
                let address = run.base + ((page - bank_page) << SLOT_SHIFT);
                *old.held.entry(address >> BITMAP_SHIFT).or_default() += 1;
                #[cfg(test)]
                Cost::add(&self.cost.table_ops, 1);
                phys.map(address, address + (1 << SLOT_SHIFT),
                    old.slow.as_ref().expect("Bedrock MD slow endpoint").clone(), "Bedrock checked RAM");
            }
            if let Some(space) = space {
                let region_size = 1 << BITMAP_SHIFT;
                for offset in (0..run.size).step_by(region_size as usize) {
                    // The startup page answers through the decode only, so the
                    // region holding it stays out of the direct window.
                    let startup = self.startup_live() && run.base + offset == 0;
                    let region = (run.base + offset) >> BITMAP_SHIFT;
                    // Above the window the bank answers through the table alone.
                    let in_window = run.base + offset + region_size <= WINDOW_SIZE;
                    if in_window && !startup && !old.held.contains_key(&region) {
                        #[cfg(test)]
                        Cost::add(&self.cost.window_ops, 1);
                        match space.map_range(run.bank, offset, run.base + offset, region_size) {
                            Ok(()) => { old.windowed.insert(region); }
                            Err(error) => crate::dlog!(crate::devlog::LogModule::Bedrock,
                                "Bedrock RAM window at {:#x}: {error}", run.base + offset),
                        }
                    }
                }
            }
        }
        let live = self.startup_live();
        if live {
            #[cfg(test)]
            Cost::add(&self.cost.table_ops, 1);
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
        old.placed = pages;
    }

    /// Where bank-relative `page` lies in the run holding it: the run, and the
    /// page's byte offset in the bank.
    fn locate(&self, runs: &[Run], page: u64) -> Option<(Run, u64)> {
        runs.iter().find_map(|run| {
            let first = self.dense_base(run.bank as u64) >> SLOT_SHIFT;
            (first..first + (run.size >> SLOT_SHIFT)).contains(&page)
                .then(|| (*run, (page - first) << SLOT_SHIFT))
        })
    }

    /// Whether `address` lies in the startup window while its page stands.
    fn in_startup(&self, address: u64) -> bool {
        self.startup_live() && (STARTUP_BASE..STARTUP_BASE + STARTUP_SIZE).contains(&address)
    }

    /// Advance the counters of `pages` 4 KiB pages starting at `address`, the
    /// bank's offset `offset` for the same bytes.
    fn bump_pages(&self, _address: u64, _bank: usize, _offset: u64, _pages: u64) {
        #[cfg(feature = "jitv2")]
        {
            for page in 0.._pages {
                self.generations.bump(_address + (page << SLOT_SHIFT));
                self.physical.banks()[_bank].bump_gen_off((_offset + (page << SLOT_SHIFT)) as usize);
            }
            #[cfg(test)]
            Cost::add(&self.cost.bumps, 2 * _pages as usize);
        }
    }

    /// Route the pages in `marked` through the MD endpoint, adding to what is
    /// placed and changing nothing else. A page already placed, as a lifted
    /// one marked again is, costs no map change. The first page in a region
    /// takes the region out of the direct window, and a page whose mapping
    /// changes advances its counters, so code compiled from it, or from the
    /// region's direct path, is dropped. CPU thread only, as `remap`.
    pub(crate) fn place(&self, marked: &[u64]) {
        let mut m = self.mapping.lock();
        let phys = &self.physical;
        let space = phys.ppmem_space();
        let region_size = 1u64 << BITMAP_SHIFT;
        for &page in marked {
            if !m.placed.insert(page) {
                continue;
            }
            let Some((run, offset)) = self.locate(&m.runs, page) else { continue };
            let address = run.base + offset;
            let region = address >> BITMAP_SHIFT;
            *m.held.entry(region).or_default() += 1;
            // Withdraw the direct window before publishing the checked page.
            if m.windowed.remove(&region) {
                if let Some(space) = space {
                    #[cfg(test)]
                    Cost::add(&self.cost.window_ops, 1);
                    space.unmap_range(region << BITMAP_SHIFT, region_size);
                }
                self.bump_pages(region << BITMAP_SHIFT, run.bank,
                    offset & !(region_size - 1), region_size >> SLOT_SHIFT);
            } else {
                self.bump_pages(address, run.bank, offset, 1);
            }
            if !self.in_startup(address) {
                #[cfg(test)]
                Cost::add(&self.cost.table_ops, 1);
                phys.map(address, address + (1 << SLOT_SHIFT),
                    m.slow.as_ref().expect("Bedrock MD slow endpoint").clone(), "Bedrock checked RAM");
            }
        }
    }

    /// Whether the lifted pages are due to be retired: `active` pages of the
    /// placed ones still hold a mark, and `now` is the controller's clock if
    /// it has one. Due when too many lifted pages have piled up, or when the
    /// first has idled for a second. Notes the time it first sees one.
    pub(crate) fn retirement_due(&self, active: usize, now: Option<u64>) -> bool {
        let mut m = self.mapping.lock();
        if m.placed.len() <= active {
            m.idle_since = None;
            return false;
        }
        if m.placed.len() - active >= RETIRED_LIMIT {
            return true;
        }
        match (now, m.idle_since) {
            (Some(now), Some(since)) => now.saturating_sub(since) >= RETIRED_IDLE_NS,
            (Some(now), None) => { m.idle_since = Some(now); false }
            _ => false,
        }
    }

    /// The placed pages for which `marked` is false, in order.
    pub(crate) fn lifted(&self, marked: impl Fn(u64) -> bool) -> Vec<u64> {
        self.mapping.lock().placed.iter().copied().filter(|&page| !marked(page)).collect()
    }

    /// Return every page of `lifted` to its bank in one change: its table
    /// entry, then the direct window for each region left with no placed
    /// page. The pages' counters advance as for any change of mapping. CPU
    /// thread only, as `remap`.
    pub(crate) fn retire(&self, lifted: &[u64]) {
        let mut m = self.mapping.lock();
        let phys = &self.physical;
        let space = phys.ppmem_space();
        let region_size = 1u64 << BITMAP_SHIFT;
        let mut emptied = Vec::new();
        for &page in lifted {
            if !m.placed.remove(&page) {
                continue;
            }
            let Some((run, offset)) = self.locate(&m.runs, page) else { continue };
            let address = run.base + offset;
            let region = address >> BITMAP_SHIFT;
            if !self.in_startup(address) {
                #[cfg(test)]
                Cost::add(&self.cost.table_ops, 1);
                phys.fill_slots(address >> SLOT_SHIFT, 1, phys.bank_view(run.bank, run.base));
            }
            self.bump_pages(address, run.bank, offset, 1);
            let held = m.held.get_mut(&region).expect("a placed page's region is held");
            *held -= 1;
            if *held == 0 {
                m.held.remove(&region);
                emptied.push((region, run, offset & !(region_size - 1)));
            }
        }
        // Return a region to the window only once none of its pages is placed.
        for (region, run, offset) in emptied {
            let startup = self.startup_live() && region == 0;
            let in_window = (region + 1) << BITMAP_SHIFT <= WINDOW_SIZE;
            if let (Some(space), true, false) = (space, in_window, startup) {
                #[cfg(test)]
                Cost::add(&self.cost.window_ops, 1);
                match space.map_range(run.bank, offset, region << BITMAP_SHIFT, region_size) {
                    Ok(()) => { m.windowed.insert(region); }
                    Err(error) => crate::dlog!(crate::devlog::LogModule::Bedrock,
                        "Bedrock RAM window at {:#x}: {error}", region << BITMAP_SHIFT),
                }
            }
        }
        m.idle_since = None;
    }

    pub(crate) fn decode(&self, config: u64, addr: u64, bytes: u64) -> Option<(usize, u64)> {
        let software = addr / BANK_SPAN;
        if software >= SLOTS as u64 {
            return None;
        }
        let bank = software ^ ((config >> 32) & 3);
        let field = config >> (bank / 2 * 8);
        let code = (field >> 4) & 15;
        if self.fitted(bank) == 0 || field & (1 << (bank & 1)) == 0 || !(1..=5).contains(&code) {
            return None;
        }
        let limit = self.fitted(bank).min((32 << 20) << code);
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
        assert!(size <= self.fitted(bank as u64));
        // The bank owns this contiguous mapping; zero has the same representation
        // in host word order. Self-test does not synchronize concurrent bank accesses.
        unsafe {
            std::ptr::write_bytes(self.physical.banks()[bank].host_ptr(), 0, size as usize);
        }
        #[cfg(feature = "jitv2")]
        self.physical.banks()[bank].bump_gen_all();
        #[cfg(feature = "jitv2")]
        self.generations.bump_all(&self.extent);
    }
    pub(crate) fn changed_decode(&self) {
        #[cfg(feature = "jitv2")]
        self.generations.bump_all(&self.extent);
        #[cfg(feature = "jitv2")]
        for bank in self.physical.banks() {
            bank.bump_gen_all();
        }
        #[cfg(all(test, feature = "jitv2"))]
        Cost::add(&self.cost.bumps, (0..self.extent.slots()).map(|slot| (self.extent.prefix(slot) >> 12) as usize).sum::<usize>()
            + self.physical.banks().iter().map(|bank| bank.size() >> 12).sum::<usize>());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Tezro takes 512 and 1024 MiB DIMMs, one per bank, none left out
    /// between, eight at most. The 4 GiB layout is the default and the 1 GiB
    /// one is still constructible.
    #[test]
    fn only_512_and_1024_mib_banks_are_accepted() {
        assert_eq!(MemoryPopulation::default(), MemoryPopulation::four_gib());
        assert_eq!(MemoryPopulation::one_gib().banks_mib(), [512, 512]);
        assert_eq!(MemoryPopulation::four_gib().banks_mib(), [1024, 1024, 512, 512, 512, 512]);
        assert_eq!(MemoryPopulation::four_gib(), MemoryPopulation::from_mib(&[1024, 1024, 512, 512, 512, 512]).unwrap());
        assert_eq!((MemoryPopulation::one_gib().total_mib(), MemoryPopulation::four_gib().total_mib()), (1024, 4096));
        for bad in [&[][..], &[256], &[2048], &[512, 0], &[768, 512], &[512; 9]] {
            assert!(MemoryPopulation::from_mib(bad).is_err(), "{bad:?}");
        }
        assert!(MemoryPopulation::from_mib(&[1024; 8]).is_ok());
        assert!(MemoryPopulation::new(&[]).is_err());
    }

    /// The 1 GiB population lies wholly inside ppmem's 4 GiB window; the 4 GiB
    /// default does not, since its banks 4 and 5 are above it. The memory model
    /// accepts both.
    #[test]
    fn the_default_fits_the_direct_window_and_the_4_gib_layout_does_not() {
        for (population, direct) in [(MemoryPopulation::one_gib(), true), (MemoryPopulation::four_gib(), false)] {
            assert_eq!(population.fits_direct_window(), direct);
            let end = |bank: usize| bank as u64 * BANK_SPAN + u64::from(population.banks()[bank].mib()) * (1 << 20);
            let last = population.banks().len() - 1;
            assert_eq!(end(last) <= WINDOW_SIZE, direct, "the last bank ends inside the window");
            assert!(super::super::bedrock_md::BedrockMd::new(&population.banks_mib()).is_ok());
        }
    }

    /// One processor needs 512 MB; two or four need 1 GB. Fewer than one or
    /// more than four processors is not a Tezro.
    #[test]
    fn the_memory_must_meet_the_minimum_for_the_processors() {
        let half = MemoryPopulation::from_mib(&[512]).unwrap();
        assert!(half.check_for_cpus(1).is_ok());
        for cpus in [2, 3, 4] {
            let error = half.check_for_cpus(cpus).unwrap_err();
            assert!(error.contains("1024"), "{error}");
        }
        for population in [MemoryPopulation::one_gib(), MemoryPopulation::four_gib()] {
            for cpus in [1, 2, 4] {
                assert!(population.check_for_cpus(cpus).is_ok());
            }
        }
        assert!(MemoryPopulation::one_gib().check_for_cpus(0).is_err());
        assert!(MemoryPopulation::one_gib().check_for_cpus(5).is_err());
        // A single 1024 MiB bank meets the two-processor minimum too.
        assert!(MemoryPopulation::from_mib(&[1024]).unwrap().check_for_cpus(4).is_ok());
    }
}
