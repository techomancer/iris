//! A Bedrock node's fitted memory: banks owned by `Physical`, with the bank
//! decode selected by Bedrock's memory configuration register.
//!
//! The board names the fitted banks, locational bank 0 upward, one size in
//! MiB each: two 256 MiB banks on the Fuel, up to eight on the Tezro. Each
//! bank answers in its own 1 GiB software slot, so banks 4 to 7 sit at 4 to
//! 8 GiB. A node with banks there asks ppmem for its 16 GiB window, as the
//! Octane's HEART does past 4 GiB, so every bank is in the CPU's direct
//! window.
//!
//! Every access reaches a bank through Physical's bank views or the bank
//! itself, without a lock. Remapping runs on the CPU thread, or with the CPU
//! stopped, as HEART's memory descriptors do: a bank configuration change
//! rebuilds the whole map. Physical publishes complete decode-table pages
//! atomically; a concurrent bus master sees an old or new entry. This does
//! not promise a transaction across a remap.
//!
//! Ported from hw/misc/sgi-bedrock-md.c in jamesbraid/qemu-sgi-origami.

use crate::physical::{Physical, RamBank};
use crate::traits::*;
use crate::ppmem::devmap::SLOT_SHIFT;
use crate::ppmem::{MappedMemory, BITMAP_SHIFT, LARGE_WINDOW_SIZE, WINDOW_SIZE};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const BANK_SPAN: u64 = 1 << 30;
/// Software slots of [`BANK_SPAN`] in the node's 8 GiB.
pub const SLOTS: usize = 8;
/// The most memory a bank can hold: what the largest size code names.
const MAX_BANK_MIB: u32 = 1024;
/// Banks are mapped into the direct window in whole regions of this size.
const REGION_MIB: u32 = 1 << (BITMAP_SHIFT - 20);

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

#[derive(Clone, Copy, PartialEq, Eq)]
struct Run {
    bank: usize,
    base: u64,
    size: u64,
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
    pub(crate) fn pages(&self) -> usize {
        self.0.len()
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

pub struct Memory {
    physical: Arc<Physical>,
    /// Fitted bytes in each locational bank, bank 0 first. Physical's banks
    /// are these, in the same order.
    fitted: Vec<u64>,
    config: AtomicU64,
    /// The runs the last remap mapped.
    runs: Mutex<Vec<Run>>,
    /// Each processor's window while its PROM cache lines stand.
    extent: RamExtent,
    #[cfg(feature = "jitv2")]
    generations: Arc<Generations>,
}
impl Memory {
    /// Memory with `banks` fitted, locational bank 0 upward, each a size in
    /// MiB: at most [`SLOTS`] banks, none empty or over 1 GiB (the largest
    /// size code), each a whole number of direct-window regions, and banks
    /// 2k and 2k+1 the same size where both are fitted. The configuration
    /// word is four byte lanes, each enabling its two banks separately and
    /// holding one size code for both, so the hardware cannot size a lane's
    /// banks apart (established from the stock IP35 PROM's decode of that
    /// word).
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
        // The configuration word has one size code per byte lane, shared by
        // the lane's two banks, so the two cannot differ.
        if let Some(lane) = banks.chunks(2).position(|pair| pair.len() == 2 && pair[0] != pair[1]) {
            return Err(format!(
                "memory: banks {} and {} are {} and {} MiB; the two banks of a lane share one size",
                2 * lane, 2 * lane + 1, banks[2 * lane], banks[2 * lane + 1]));
        }
        // Bank n starts at n GiB; past bank 3 the node runs over 4 GiB.
        let window = if banks.len() as u64 * BANK_SPAN > WINDOW_SIZE { LARGE_WINDOW_SIZE } else { WINDOW_SIZE };
        let physical = Physical::with_window(banks.iter().map(|&mib| RamBank::new(mib as usize)).collect(), window);
        physical.set_default(Arc::new(NoMemory), "Bedrock unconfigured memory");
        Ok(Self { physical, fitted: banks.iter().map(|&mib| u64::from(mib) << 20).collect(),
            config: AtomicU64::new(0), runs: Mutex::new(Vec::new()),
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

    fn runs(&self, config: u64) -> Vec<Run> {
        (0..SLOTS as u64).filter_map(|slot| {
            let base = slot * BANK_SPAN;
            let (bank, _) = self.decode(config, base, 1)?;
            let code = (config >> (bank / 2 * 8 + 4)) & 15;
            Some(Run { bank, base, size: self.fitted(bank as u64).min((32 << 20) << code) })
        }).collect()
    }

    /// Map the banks as `config` places them. Only the CPU thread, or a
    /// caller with the CPU stopped, may change ppmem's bitmap. Without
    /// `force` it does nothing when the banks would answer where they do; a
    /// forced rebuild follows the machine setting Physical's default.
    pub(crate) fn remap(&self, config: u64, force: bool) {
        let runs = self.runs(config);
        let mut old = self.runs.lock();
        if !force && *old == runs {
            self.config.store(config, Ordering::Release);
            return;
        }
        let phys = &self.physical;
        let space = phys.ppmem_space();
        if let Some(space) = space {
            space.clear_mappings();
        }
        // Restore only ranges previously owned by RAM. In particular, the
        // unused part of a 1 GiB slot may contain flash or other devices.
        for run in old.iter() {
            let retained = runs.iter().find(|next| next.base == run.base)
                .map_or(0, |next| next.size.min(run.size));
            if retained < run.size {
                phys.fill_slots((run.base + retained) >> SLOT_SHIFT,
                    (run.size - retained) >> SLOT_SHIFT, phys.default_entry());
            }
        }
        for run in &runs {
            phys.fill_slots(run.base >> SLOT_SHIFT, run.size >> SLOT_SHIFT, phys.bank_view(run.bank, run.base));
            if let Some(space) = space {
                if let Err(error) = space.map_range(run.bank, 0, run.base, run.size) {
                    crate::dlog!(crate::devlog::LogModule::Bedrock,
                        "Bedrock RAM window at {:#x}: {error}", run.base);
                }
            }
        }
        self.config.store(config, Ordering::Release);
        // Publish changed ownership before advancing the stable counters,
        // so an in-flight compile cannot accept bytes from the old decode.
        self.changed_decode();
        *old = runs;
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
