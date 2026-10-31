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
//! memory descriptors do. DMA may remove diagnostic metadata but retains
//! the slow mapping until the next CPU-side rebuild. Physical publishes
//! complete decode-table pages atomically; a concurrent bus master sees an
//! old or new entry. This does not promise a transaction across a remap.
//!
//! Ported from hw/misc/sgi-bedrock-md.c in jamesbraid/qemu-sgi-origami.

use crate::physical::{Physical, RamBank};
use crate::traits::*;
use crate::ppmem::devmap::SLOT_SHIFT;
use crate::ppmem::{MappedMemory, BITMAP_SHIFT, WINDOW_SIZE};
use parking_lot::Mutex;
use std::collections::BTreeSet;
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
/// assembly layer records them. The default layout is inferred: the manuals'
/// rules allow other 1 GiB layouts (a single 1024 MiB bank, for one), and the
/// two banks of 512 MiB are the pair the 4 GiB kit also uses at those places.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPopulation(Vec<BankSize>);
impl Default for MemoryPopulation {
    fn default() -> Self {
        Self::gib1()
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
    /// 1 GiB: banks 0 and 1 of 512 MiB. Everything lies below 4 GiB, inside
    /// ppmem's direct window, and it meets the minimum for any processor count.
    pub fn gib1() -> Self {
        Self(vec![BankSize::Mib512; 2])
    }
    /// 4 GiB: 1024, 1024 and four of 512 MiB. Banks 4 and 5 sit above 4 GiB,
    /// outside the direct window.
    pub fn gib4() -> Self {
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
    pub(crate) fn remap(&self, config: u64, slow_pages: BTreeSet<u64>, force: bool) {
        let runs = self.runs(config);
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
            let bank_page = self.dense_base(run.bank as u64) >> SLOT_SHIFT;
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
                    // Above the window the bank answers through the table alone.
                    let in_window = run.base + offset + region_size <= WINDOW_SIZE;
                    if in_window && !startup && !checked_regions.contains(&(offset >> BITMAP_SHIFT)) {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Tezro takes 512 and 1024 MiB DIMMs, one per bank, none left out
    /// between, eight at most. The 4 GiB layout and the 1 GiB default are
    /// both there, and the default's banks all lie below 4 GiB.
    #[test]
    fn only_512_and_1024_mib_banks_are_accepted() {
        assert_eq!(MemoryPopulation::default(), MemoryPopulation::gib1());
        assert_eq!(MemoryPopulation::gib1().banks_mib(), [512, 512]);
        assert_eq!(MemoryPopulation::gib4().banks_mib(), [1024, 1024, 512, 512, 512, 512]);
        assert_eq!(MemoryPopulation::gib4(), MemoryPopulation::from_mib(&[1024, 1024, 512, 512, 512, 512]).unwrap());
        assert_eq!((MemoryPopulation::gib1().total_mib(), MemoryPopulation::gib4().total_mib()), (1024, 4096));
        for bad in [&[][..], &[256], &[2048], &[512, 0], &[768, 512], &[512; 9]] {
            assert!(MemoryPopulation::from_mib(bad).is_err(), "{bad:?}");
        }
        assert!(MemoryPopulation::from_mib(&[1024; 8]).is_ok());
        assert!(MemoryPopulation::new(&[]).is_err());
    }

    /// The default keeps the whole population inside ppmem's 4 GiB window, so
    /// the default boot never uses the decode-table path; the 4 GiB layout does
    /// not, and both are accepted by the memory model.
    #[test]
    fn the_default_fits_the_direct_window_and_the_4_gib_layout_does_not() {
        for (population, direct) in [(MemoryPopulation::gib1(), true), (MemoryPopulation::gib4(), false)] {
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
        for population in [MemoryPopulation::gib1(), MemoryPopulation::gib4()] {
            for cpus in [1, 2, 4] {
                assert!(population.check_for_cpus(cpus).is_ok());
            }
        }
        assert!(MemoryPopulation::gib1().check_for_cpus(0).is_err());
        assert!(MemoryPopulation::gib1().check_for_cpus(5).is_err());
        // A single 1024 MiB bank meets the two-processor minimum too.
        assert!(MemoryPopulation::from_mib(&[1024]).unwrap().check_for_cpus(4).is_ok());
    }
}
