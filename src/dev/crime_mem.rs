//! O2 main memory: the DIMMs that are installed, and where CRIME's bank
//! controls put them.
//!
//! CRIME has eight memory banks, each 32 MB or 128 MB, and a control register
//! per bank that says where in the 1 GB memory window (from
//! [`RAM_BASE`](crate::ip32::RAM_BASE)) that bank answers and how large the
//! controller should take it to be: bits 0-4 are the base in 32 MB units, bit
//! 8 selects 128 MB. Nothing tells the firmware what is fitted; it finds out by
//! programming a bank and looking at what comes back, so the decode has to
//! behave the way the hardware does in the three cases that matter to it:
//!
//! - When more than one control matches an address, the lowest-numbered bank
//!   is the one that answers.
//! - A matching bank with nothing in it does not raise an error. It hands back
//!   whatever was last on the memory bus, which the firmware sets to zero
//!   before it looks.
//! - A bank told it is 128 MB when it is 32 MB ignores the address lines it
//!   does not have, so its 32 MB repeat four times across the range. That
//!   repetition is how the firmware tells the two sizes apart.
//!
//! An address no control matches is a bus error.
//!
//! The DIMMs go in pairs, four pairs (banks A to D in SGI's own manuals, which
//! are not CRIME's banks). A pair of single-sided DIMMs is one CRIME bank, a
//! pair of double-sided ones is two, so pair A is CRIME banks 0 and 1, pair B 2
//! and 3, and so on. SGI sold four kits:
//!
//! | kit    | DIMMs        | sides  | CRIME banks  |
//! |--------|--------------|--------|--------------|
//! | 32 MB  | 2 x 16 MB    | single | one of 32 MB |
//! | 64 MB  | 2 x 32 MB    | double | two of 32 MB |
//! | 128 MB | 2 x 64 MB    | single | one of 128 MB|
//! | 256 MB | 2 x 128 MB   | double | two of 128 MB|
//!
//! and the rules for fitting them: pair A is always populated, both DIMMs of a
//! pair match, pairs are filled in order without gaps, and the largest kit goes
//! in pair A (so no pair holds more than the one before it). That makes 32 MB
//! to 1 GB in 32 MB steps, though not every step: 992 MB would take five kits.
//!
//! The low 256 MB of the window also answers from physical address 0, which is
//! where the firmware and a 32-bit kernel do most of their work.
//!
//! Each fitted bank is a RAM bank of the O2's [`Physical`] bus. A change to
//! the bank controls remaps it as MEMCFG does on the Indy: every 32 MB slot
//! of the window (and of the low alias) points the decode table at the bank
//! that answers there, counted from that bank's base, and ppmem's window maps
//! the same bank pages, which is what the CPU's cache and the JIT read and
//! write directly and what the rendering engine draws through
//! ([`RamWindow`]). Every other master reaches memory through the bus.
//!
//! POST keeps its stack in cache lines tagged for physical page 0 while it
//! tests the same page uncached, so on the machine the two are different bytes.
//! The CPU's cached path and the window always mean the bank, so that is where
//! the stack lives here too, and for as long as those tags stand the decode
//! table points that page at a page of its own for the uncached path. See
//! [`O2Memory::stack_tag_store`].
//!
//! `O2Memory` is CRIME's bank logic: the sizes fitted, the controls, the
//! decode they make, and the remap. Its decode is atomics: the firmware
//! programs the bank controls once at power-on, and a guest that moved a
//! bank while a master was using it would be undefined on the real machine
//! too, so readers never wait. Remaps run on the CPU thread (a CRIME store)
//! or with the machine stopped (a snapshot); a lock keeps two from
//! interleaving.

use crate::ip32::{LOW_ALIAS_SIZE, RAM_BASE, RAM_WINDOW_SIZE};
use crate::physical::{Physical, RamBank};
use crate::ppmem::MappedMemory;
use crate::ppmem::devmap::SLOT_SHIFT;
use crate::traits::{BusDevice, BusRead8, BusRead16, BusRead32, BusRead64, BUS_ERR};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// The unit the bank controls count in, and the size of a small bank.
const SLOT: u32 = 32 << 20;
/// The 128 MB bank size; bit 8 of a bank control selects it.
const BIG: u32 = 128 << 20;
const CTRL_128M: u64 = 0x100;
/// The page jitv2 tracks code by.
const PAGE: usize = 4096;
const CTRL_BASE: u64 = 0x1f;
/// The window in 32 MB slots.
const SLOTS: usize = (RAM_WINDOW_SIZE / SLOT) as usize;
/// A 32 MB slot in the decode table's 4 KB entries.
const TABLE_PER_SLOT: u64 = (SLOT >> SLOT_SHIFT) as u64;

/// The one page of physical memory POST keeps its stack in.
const STACK_PAGE: u32 = 4096;
/// TagLo as R10000 and R5000 share it: the primary state, 0 for invalid.
const TAG_STATE: u32 = 0xc0;
/// TagLo's physical page number.
const TAG_PAGE_SHIFT: u32 = 8;

/// The kits SGI sold, in MB per pair.
pub const KITS_MB: [u32; 4] = [32, 64, 128, 256];

/// What a 32 MB slot of the window decodes to.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Slot {
    /// No bank control matches: a bus error.
    None,
    /// The first match is a bank with nothing fitted.
    Empty,
    /// Memory: bank `bank`, beginning `chunk` 32 MB slots into it before
    /// wrapping at its real size.
    Ram { bank: u8, chunk: u8 },
}

impl Slot {
    /// Packed for the decode table: 0 none, 1 empty, else
    /// `0x100 | bank << 4 | chunk`.
    fn pack(self) -> u32 {
        match self {
            Slot::None => 0,
            Slot::Empty => 1,
            Slot::Ram { bank, chunk } => 0x100 | (bank as u32) << 4 | chunk as u32,
        }
    }

    fn unpack(v: u32) -> Self {
        match v {
            0 => Slot::None,
            1 => Slot::Empty,
            _ => Slot::Ram { bank: ((v >> 4) & 7) as u8, chunk: (v & 0xf) as u8 },
        }
    }
}

/// Where an address lands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decode {
    /// Memory, here: the address in the window from `RAM_BASE`, which the
    /// bus reaches it at (the low alias resolved).
    Ram(usize),
    /// A bank control matches but no DIMM is there.
    Empty,
    /// Nothing answers.
    None,
}

/// What a bus made by [`O2Memory`] alone answers where no memory is: reads
/// as zero, writes as a bus error, as the O2's bus does. A machine replaces
/// it with its own, which also logs.
struct NoMemory;

impl BusDevice for NoMemory {
    fn read8(&self, _: u64) -> BusRead8 { BusRead8::ok(0) }
    fn write8(&self, _: u64, _: u8) -> u32 { BUS_ERR }
    fn read16(&self, _: u64) -> BusRead16 { BusRead16::ok(0) }
    fn write16(&self, _: u64, _: u16) -> u32 { BUS_ERR }
    fn read32(&self, _: u64) -> BusRead32 { BusRead32::ok(0) }
    fn write32(&self, _: u64, _: u32) -> u32 { BUS_ERR }
    fn read64(&self, _: u64) -> BusRead64 { BusRead64::ok(0) }
    fn write64(&self, _: u64, _: u64) -> u32 { BUS_ERR }
}

/// CRIME's memory: the banks fitted, their controls and the decode they
/// make, over the machine's physical bus.
pub struct O2Memory {
    /// The bus the banks are on; its RAM banks are the fitted CRIME banks,
    /// in bank order.
    phys: Arc<Physical>,
    /// Each CRIME bank's real size in bytes (0 for nothing fitted). Fixed
    /// when the memory is built.
    size: [u32; 8],
    /// Each fitted CRIME bank's index among the bus's RAM banks.
    index: [Option<usize>; 8],
    /// Bytes fitted.
    total: usize,
    /// The bank controls as last written.
    ctrl: [AtomicU64; 8],
    /// Each 32 MB slot's decode, [`Slot::pack`]ed.
    slots: [AtomicU32; SLOTS],
    /// Bumped whenever the decode changes, so a device can cache an offset.
    decode_gen: AtomicU64,
    /// Held while the decode is rebuilt, so two rebuilds cannot interleave.
    /// Readers never take it.
    rebuilding: std::sync::Mutex<()>,
    /// What [`RamWindow`]s need to know of the decode, published as it
    /// changes.
    shared: Arc<WindowState>,
    /// Page 0 as the uncached path sees it while POST's stack is in the bank.
    /// It wraps at 4 KB, so the same device answers at every alias. Boxed: the
    /// decode table points at it and `O2Memory` is built by value.
    stack_page: Box<RamBank>,
    /// Whether the decode table is pointing at [`Self::stack_page`].
    stack_live: AtomicBool,
}

impl O2Memory {
    /// Memory fitted as `kits`, the size in MB of the kit in each DIMM pair
    /// from pair A.
    pub fn from_kits(kits: &[u32]) -> Result<Self, String> {
        check_kits(kits)?;
        let mut banks = [0u32; 8];
        for (pair, &kit) in kits.iter().enumerate() {
            let (a, b) = match kit {
                32 => (SLOT, 0),
                64 => (SLOT, SLOT),
                128 => (BIG, 0),
                _ => (BIG, BIG),
            };
            banks[2 * pair] = a;
            banks[2 * pair + 1] = b;
        }
        Ok(Self::from_banks(banks))
    }

    /// The usual way to fit `mb`: the fewest kits, largest first.
    pub fn with_total(mb: u32) -> Result<Self, String> {
        Self::from_kits(&kits_for(mb)?)
    }

    /// One bank 0 of `bytes` and nothing else. Not a configuration an O2 can
    /// have unless `bytes` is 32 or 128 MB: it is for device tests that want a
    /// little memory at the bottom of the window. `bytes` must be a power of
    /// two no larger than 128 MB.
    pub fn single(bytes: usize) -> Self {
        let mut banks = [0u32; 8];
        banks[0] = bytes as u32;
        Self::from_banks(banks)
    }

    /// The banks of `size` on a bus of their own, with [`NoMemory`] where
    /// nothing is mapped, decoded as at reset.
    fn from_banks(size: [u32; 8]) -> Self {
        let mut index = [None; 8];
        let mut banks = Vec::new();
        let mut total = 0usize;
        for i in 0..8 {
            assert!(size[i] == 0 || (size[i].is_power_of_two() && size[i] <= BIG), "bank {i}: {}", size[i]);
            if size[i] == 0 {
                continue;
            }
            let bank = RamBank::new((size[i] as usize).div_ceil(1 << 20));
            // A test's bank smaller than ppmem's 1 MB unit wraps at its size.
            bank.set_addr_mask(size[i] - 1);
            index[i] = Some(banks.len());
            banks.push(bank);
            total += size[i] as usize;
        }
        let phys = Physical::new(banks);
        phys.set_default(Arc::new(NoMemory), "no memory");
        let stack_page = Box::new(RamBank::new(1));
        stack_page.set_addr_mask(STACK_PAGE - 1);
        let m = Self {
            phys,
            size,
            index,
            total,
            ctrl: Default::default(),
            slots: std::array::from_fn(|_| AtomicU32::new(0)),
            decode_gen: AtomicU64::new(0),
            rebuilding: std::sync::Mutex::new(()),
            shared: Default::default(),
            stack_page,
            stack_live: AtomicBool::new(false),
        };
        // At reset every control is zero: all eight banks at the bottom of
        // the window, 32 MB each, so bank 0 answers there.
        m.rebuild();
        m
    }

    /// The bus the memory is on.
    pub fn phys(&self) -> &Arc<Physical> {
        &self.phys
    }

    /// Bank `i`'s control register was written.
    pub fn set_ctrl(&self, i: usize, val: u64) {
        let _one = self.rebuilding.lock().unwrap();
        if self.ctrl[i].swap(val, Ordering::AcqRel) != val {
            self.rebuild_locked();
        }
    }

    /// Lay the table and the window out again as the controls stand: after
    /// the machine replaced the bus's default device, which covers every
    /// slot it has not mapped since.
    pub fn rebuild(&self) {
        let _one = self.rebuilding.lock().unwrap();
        self.rebuild_locked();
    }

    /// Recompute every slot from the controls. A reader racing this sees
    /// some slots old and some new, each still inside fitted memory.
    fn rebuild_locked(&self) {
        self.decode_gen.fetch_add(1, Ordering::AcqRel);
        let mut slots = [Slot::None; SLOTS];
        // Highest bank first, so the lowest-numbered match is the one left.
        for i in (0..8).rev() {
            let ctrl = self.ctrl[i].load(Ordering::Acquire);
            let base = (ctrl & CTRL_BASE) as usize;
            let span = if ctrl & CTRL_128M != 0 { (BIG / SLOT) as usize } else { 1 };
            for j in 0..span {
                let Some(slot) = slots.get_mut(base + j) else { break };
                *slot = if self.size[i] == 0 { Slot::Empty } else { Slot::Ram { bank: i as u8, chunk: j as u8 } };
            }
        }
        for (dst, s) in self.slots.iter().zip(slots) {
            dst.store(s.pack(), Ordering::Release);
        }
        self.remap(&slots);
    }

    /// Point the bus at the decode as it now stands: every slot of the
    /// window, and of the low alias, at the bank that answers there counted
    /// from that bank's base (so a 32 MB bank under a 128 MB setting repeats,
    /// as the bank wraps at its size), at memory that reads zero for an empty
    /// bank, and at the bus's default where no control matches. ppmem's
    /// window maps the same pages; empty and unmatched slots stay unmapped
    /// there, so the CPU reaches them through the bus.
    fn remap(&self, slots: &[Slot; SLOTS]) {
        use std::sync::atomic::Ordering::Release;
        let phys = &*self.phys;
        let sp = phys.ppmem_space();
        self.shared.mapped.store(0, Release);
        if let Some(sp) = sp {
            sp.clear_mappings();
        }
        // A bank now answering where another did must look changed to the
        // JIT even if the two pages' counters happen to be equal.
        #[cfg(feature = "jitv2")]
        for b in phys.banks() {
            b.bump_gen_all();
        }
        let mut mapped = 0u32;
        for (s, slot) in slots.iter().enumerate() {
            let at = s as u64 * SLOT as u64;
            let alias = at < LOW_ALIAS_SIZE as u64;
            let (entry, alias_entry) = match *slot {
                Slot::Ram { bank, chunk } => {
                    let i = self.index[bank as usize].expect("a fitted bank");
                    let base = at - chunk as u64 * SLOT as u64;
                    (phys.bank_view(i, RAM_BASE as u64 + base), phys.bank_view(i, base))
                }
                Slot::Empty => (phys.unmapped_entry(), phys.unmapped_entry()),
                Slot::None => (phys.default_entry(), phys.default_entry()),
            };
            phys.fill_slots((RAM_BASE as u64 + at) >> SLOT_SHIFT, TABLE_PER_SLOT, entry);
            if alias {
                phys.fill_slots(at >> SLOT_SHIFT, TABLE_PER_SLOT, alias_entry);
            }
            let (Some(sp), Slot::Ram { bank, chunk }) = (sp, *slot) else { continue };
            let size = self.size[bank as usize];
            if size < SLOT {
                continue; // a test's bank smaller than a slot: bus only
            }
            let i = self.index[bank as usize].unwrap();
            let src = (chunk as u64 * SLOT as u64) & (size as u64 - 1);
            if sp.map_range(i, src, RAM_BASE as u64 + at, SLOT as u64).is_ok() {
                mapped |= 1 << s;
            }
            if alias {
                let _ = sp.map_range(i, src, at, SLOT as u64);
            }
        }
        if self.stack_live.load(Ordering::Relaxed) {
            self.place_stack_page(slots, mapped);
        }
        self.shared.decode_gen.store(self.decode_gen.load(Ordering::Acquire), Release);
        self.shared.mapped.store(mapped, Release);
    }

    /// The decode table's entry for bank 0's offset 0 goes to the stack page,
    /// at every place that offset answers: a 32 MB bank under a 128 MB setting
    /// repeats, and sizing relies on it.
    fn place_stack_page(&self, slots: &[Slot; SLOTS], mapped: u32) {
        let phys = &*self.phys;
        let page: *const RamBank = &*self.stack_page;
        let page = page as crate::ppmem::devmap::Entry;
        let size = self.size[0] as u64;
        for (s, slot) in slots.iter().enumerate() {
            let Slot::Ram { bank: 0, chunk } = *slot else { continue };
            if (chunk as u64 * SLOT as u64) & (size - 1) != 0 {
                continue;
            }
            let at = s as u64 * SLOT as u64;
            phys.fill_slots((RAM_BASE as u64 + at) >> SLOT_SHIFT, 1, page);
            if at < LOW_ALIAS_SIZE as u64 {
                phys.fill_slots(at >> SLOT_SHIFT, 1, page);
            }
        }
        // ppmem's bitmap wants a whole 64 MB region mapped. At reset only the
        // first 32 MB slot is, so the cached path would take the bus and find
        // the stack page, then find the bank once sizing filled the region.
        // Show bank 0 again in the next slot, for the CPU only.
        if mapped & 2 == 0 {
            if let (Some(sp), Some(i)) = (phys.ppmem_space(), self.index[0]) {
                let _ = sp.map_range(i, 0, SLOT as u64, SLOT as u64);
            }
        }
    }

    /// The guest stored `tag` in a primary data cache tag.
    ///
    /// POST sweeps a page's worth of tags to dirty for physical page 0 and
    /// fills them, with nothing read from memory, then writes the same page
    /// uncached to test it, and sweeps every tag invalid without a writeback
    /// once it has copied what it needs and moved its stack to KSEG1. So a
    /// valid tag for page 0 starts the stack and the next invalid tag ends it.
    /// IRIX reboots by jumping into the PROM, so this catches a reboot as it
    /// does a power-on. Kernels and `sash` store invalid tags only, which is
    /// nothing to do here.
    pub fn stack_tag_store(&self, tag: u32) {
        if tag & TAG_STATE == 0 {
            if self.stack_live.load(Ordering::Relaxed) {
                self.set_stack_page(false);
            }
        } else if tag >> TAG_PAGE_SHIFT == 0 && !self.stack_live.load(Ordering::Relaxed) {
            self.set_stack_page(true);
        }
    }

    /// Start the stack (the uncached path gets a copy of the page, the bank
    /// goes on holding what the cached path sees) or end it (what was written
    /// uncached is what memory holds, as it is when the cache lines go).
    pub fn set_stack_page(&self, on: bool) {
        let _one = self.rebuilding.lock().unwrap();
        if self.stack_live.load(Ordering::Relaxed) == on {
            return;
        }
        let bank = &self.phys.banks()[self.index[0].expect("bank 0 is fitted")];
        for o in (0..STACK_PAGE as u64).step_by(8) {
            if on {
                self.stack_page.write64(o, bank.read64(o).data);
            } else {
                bank.write64(o, self.stack_page.read64(o).data);
            }
        }
        self.stack_live.store(on, Ordering::Relaxed);
        self.rebuild_locked();
    }

    /// The stack page's dwords while the stack is live, for a snapshot: the
    /// bank is saved with memory and this is what the uncached path sees.
    pub fn stack_page_words(&self) -> Option<Vec<u64>> {
        self.stack_live.load(Ordering::Relaxed).then(|| {
            (0..STACK_PAGE as u64).step_by(8).map(|o| self.stack_page.read64(o).data).collect()
        })
    }

    /// Put back what [`Self::stack_page_words`] gave. The stack is not live
    /// beforehand (a machine being restored has been powered on).
    pub fn restore_stack_page(&self, words: &[u64]) -> Result<(), String> {
        if words.len() * 8 != STACK_PAGE as usize {
            return Err(format!("saved stack page is {} dwords", words.len()));
        }
        let _one = self.rebuilding.lock().unwrap();
        for (k, &w) in words.iter().enumerate() {
            self.stack_page.write64(k as u64 * 8, w);
        }
        self.stack_live.store(true, Ordering::Relaxed);
        self.rebuild_locked();
        Ok(())
    }

    /// A view of memory through the window, for a master on its own thread
    /// (the rendering engine).
    pub fn window(&self) -> RamWindow {
        let sp = self.phys.ppmem_space().expect("ppmem: the O2 needs its memory window");
        RamWindow {
            base: sp.window_base(),
            #[cfg(feature = "jitv2")]
            gens: sp.gen_window_base(),
            shared: self.shared.clone(),
            mapped: 0,
            decode_gen: 0,
            dirty: Vec::new(),
            tile: std::cell::Cell::new((u64::MAX, 0)),
        }
    }

    /// Where a physical address lands: in the window from `RAM_BASE`, or in
    /// its low 256 MB from 0.
    pub fn decode(&self, addr: u32) -> Decode {
        let off = if addr < LOW_ALIAS_SIZE {
            addr
        } else if addr >= RAM_BASE && addr - RAM_BASE < RAM_WINDOW_SIZE {
            addr - RAM_BASE
        } else {
            return Decode::None;
        };
        match Slot::unpack(self.slots[(off / SLOT) as usize].load(Ordering::Acquire)) {
            Slot::None => Decode::None,
            Slot::Empty => Decode::Empty,
            Slot::Ram { .. } => Decode::Ram((RAM_BASE + off) as usize),
        }
    }

    /// The address memory answers at on the bus, if memory is there.
    pub fn offset(&self, addr: u32) -> Option<usize> {
        match self.decode(addr) {
            Decode::Ram(i) => Some(i),
            _ => None,
        }
    }

    /// The address a DMA master (a PCI device, the Ethernet, MACE's engines)
    /// reaches on the bus with bus address `addr`. Below 1 GB a bus address is
    /// an offset into the whole memory window, not just its first 256 MB: the
    /// CPU's low alias stops there because CRIME and MACE sit at 0x14000000 and
    /// up, but nothing on the bus does. OpenBSD programs buffers above 256 MB
    /// this way (0x3fffb000 for the top of a 1 GB machine); from `RAM_BASE` up
    /// an address is read as the CPU reads it, which is how Linux does it.
    pub fn dma_addr(addr: u32) -> u64 {
        if addr < RAM_WINDOW_SIZE { RAM_BASE.wrapping_add(addr) as u64 } else { addr as u64 }
    }

    /// [`Self::offset`] of [`Self::dma_addr`]: where a DMA master's bus
    /// address reaches memory, if it does.
    pub fn dma_offset(&self, addr: u32) -> Option<usize> {
        self.offset(Self::dma_addr(addr) as u32)
    }

    /// Changes whenever an address may decode differently.
    pub fn decode_gen(&self) -> u64 {
        self.decode_gen.load(Ordering::Acquire)
    }

    /// Bytes fitted.
    pub fn len(&self) -> usize {
        self.total
    }

    /// Total memory fitted, in bytes.
    pub fn total(&self) -> u32 {
        self.total as u32
    }

    /// Each CRIME bank's real size in bytes.
    pub fn bank_sizes(&self) -> [u32; 8] {
        self.size
    }

    /// The bank controls as last written.
    pub fn ctrls(&self) -> [u64; 8] {
        std::array::from_fn(|i| self.ctrl[i].load(Ordering::Acquire))
    }

    /// Each fitted bank's bytes as host words, in bank order: the memory as
    /// a snapshot keeps it. A machine is restored only on a host of the same
    /// architecture.
    pub fn bank_words(&self) -> Vec<&[u32]> {
        (0..8)
            .filter_map(|i| self.index[i].map(|k| (k, self.size[i] as usize)))
            .map(|(k, n)| {
                let b = &self.phys.banks()[k];
                // SAFETY: the bank's mapping is at least `n` bytes, page-aligned,
                // and lives as long as the bus.
                unsafe { std::slice::from_raw_parts(b.host_ptr() as *const u32, n / 4) }
            })
            .collect()
    }

    /// Put back words [`Self::bank_words`] gave, concatenated, with every
    /// page's generation bumped so no compiled code outlives them.
    pub fn restore_words(&self, words: &[u32]) -> Result<(), String> {
        if words.len() * 4 != self.total {
            return Err(format!("saved memory is {} bytes, this machine has {}", words.len() * 4, self.total));
        }
        let mut at = 0;
        for i in 0..8 {
            let Some(k) = self.index[i] else { continue };
            let n = self.size[i] as usize / 4;
            let b = &self.phys.banks()[k];
            // SAFETY: as in `bank_words`; the machine is stopped.
            unsafe { std::ptr::copy_nonoverlapping(words[at..at + n].as_ptr(), b.host_ptr() as *mut u32, n) };
            #[cfg(feature = "jitv2")]
            b.bump_gen_all();
            at += n;
        }
        Ok(())
    }

    /// Every fitted byte to `path`, bank after bank, as the host keeps them.
    pub fn save_raw(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut data = Vec::with_capacity(self.total);
        for w in self.bank_words() {
            // SAFETY: a byte view of the same words.
            data.extend_from_slice(unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) });
        }
        std::fs::write(path, data)
    }

    /// The bytes [`Self::save_raw`] wrote, back.
    pub fn load_raw(&self, path: &std::path::Path) -> std::io::Result<()> {
        let data = std::fs::read(path)?;
        if data.len() % 4 != 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "saved memory is not whole words"));
        }
        let words: Vec<u32> = data.chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
        self.restore_words(&words).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Device tests set memory up and look at it by hand, by an offset from
/// the bottom of the window (`o` below `RAM_BASE`) or an address `offset`
/// gave. Production code reaches memory through the bus.
#[cfg(test)]
impl O2Memory {
    /// The host byte for test offset or address `o`.
    fn host_of(&self, o: usize) -> *mut u8 {
        let addr = if o < RAM_BASE as usize { RAM_BASE as usize + o } else { o };
        let off = addr - RAM_BASE as usize;
        let Slot::Ram { bank, chunk } = Slot::unpack(self.slots[off / SLOT as usize].load(Ordering::Acquire)) else {
            panic!("O2 memory: no memory at {addr:#x}");
        };
        let size = self.size[bank as usize] as usize;
        let in_bank = (chunk as usize * SLOT as usize + off % SLOT as usize) & (size - 1);
        let b = &self.phys.banks()[self.index[bank as usize].unwrap()];
        unsafe { b.host_ptr().add(in_bank ^ 3) }
    }

    pub fn write(&self, o: usize, data: &[u8]) {
        for (k, &b) in data.iter().enumerate() {
            unsafe { *self.host_of(o + k) = b };
        }
    }

    pub fn read_into(&self, o: usize, buf: &mut [u8]) {
        for (k, b) in buf.iter_mut().enumerate() {
            *b = unsafe { *self.host_of(o + k) };
        }
    }

    pub fn bytes(&self, o: usize, n: usize) -> Vec<u8> {
        (o..o + n).map(|k| unsafe { *self.host_of(k) }).collect()
    }

    pub fn get_be(&self, o: usize, n: u32) -> u32 {
        self.bytes(o, n as usize).iter().fold(0, |v, &b| v << 8 | b as u32)
    }

    pub fn put_be(&self, o: usize, n: u32, v: u32) {
        self.write(o, &v.to_be_bytes()[4 - n as usize..]);
    }

    /// `n` bytes from `o` as the host keeps them (each word's bytes reversed).
    pub fn raw(&self, o: usize, n: usize) -> Vec<u8> {
        (o..o + n).map(|k| unsafe { *self.host_of(k ^ 3) }).collect()
    }
}

#[cfg(test)]
impl std::ops::Index<usize> for O2Memory {
    type Output = u8;
    fn index(&self, o: usize) -> &u8 {
        unsafe { &*self.host_of(o) }
    }
}

#[cfg(test)]
impl std::ops::IndexMut<usize> for O2Memory {
    fn index_mut(&mut self, o: usize) -> &mut u8 {
        unsafe { &mut *self.host_of(o) }
    }
}

/// Which 32 MB slots of the RAM window are mapped (bit n: `RAM_BASE + n *
/// 32 MB`) and the decode generation, as [`O2Memory`] last laid the window
/// out.
#[derive(Default)]
pub struct WindowState {
    mapped: std::sync::atomic::AtomicU32,
    decode_gen: std::sync::atomic::AtomicU64,
}

/// Memory as a master on its own thread sees it: guest physical addresses
/// in the ppmem window the CPU reads and writes, with no lock, as the
/// hardware's masters share memory. Offsets are guest physical addresses,
/// byte `p` at `base + (p ^ 3)`. Only the slots mapped when [`Self::begin`]
/// last ran are reachable: an address in any other faults in the window, so
/// every offset comes from [`Self::dma_offset`] and is checked against them.
/// Page generations are bumped once per page at [`Self::end`], through the
/// generation window, so compiled code never outlives the bytes it came
/// from.
pub struct RamWindow {
    base: *mut u8,
    #[cfg(feature = "jitv2")]
    gens: *mut std::sync::atomic::AtomicU64,
    shared: std::sync::Arc<WindowState>,
    mapped: u32,
    decode_gen: u64,
    dirty: Vec<usize>,
    /// A one-entry cache for the user: (key, offset), keys built with
    /// [`Self::decode_gen`] so a remap misses.
    pub tile: std::cell::Cell<(u64, usize)>,
}

// The pointers are into the window and its generation counters, which live
// as long as the memory; one thread draws through a window at a time.
unsafe impl Send for RamWindow {}

impl RamWindow {
    /// Start a run of accesses: take the decode as it stands now.
    pub fn begin(&mut self) {
        use std::sync::atomic::Ordering::Acquire;
        self.mapped = self.shared.mapped.load(Acquire);
        self.decode_gen = self.shared.decode_gen.load(Acquire);
    }

    /// End it: bump the generation of every page written since `begin`.
    pub fn end(&mut self) {
        #[cfg(feature = "jitv2")]
        for &page in &self.dirty {
            unsafe { (*self.gens.add(page)).fetch_add(1, std::sync::atomic::Ordering::Relaxed) };
        }
        self.dirty.clear();
    }

    pub fn decode_gen(&self) -> u64 {
        self.decode_gen
    }

    /// Whether `n` bytes from guest physical `p` are all in mapped slots.
    #[inline]
    pub fn holds(&self, p: usize, n: usize) -> bool {
        let slot = |a: usize| a.wrapping_sub(RAM_BASE as usize) / SLOT as usize;
        let (a, b) = (slot(p), slot(p + n.max(1) - 1));
        a < 32 && b < 32 && self.mapped & (1 << a) != 0 && self.mapped & (1 << b) != 0
    }

    /// Whether guest physical `p` is in a mapped slot: one item of up to 4
    /// bytes aligned to its size never leaves the slot it starts in.
    #[inline(always)]
    fn holds_at(&self, p: usize) -> bool {
        let s = p.wrapping_sub(RAM_BASE as usize) >> 25;
        s < 32 && (self.mapped >> s) & 1 != 0
    }

    /// The guest physical address a bus address reaches, as
    /// [`O2Memory::dma_offset`] decodes it, if it is in mapped memory.
    #[inline]
    pub fn dma_offset(&self, addr: u32) -> Option<usize> {
        let p = if addr < RAM_WINDOW_SIZE { RAM_BASE.wrapping_add(addr) } else { addr } as usize;
        self.holds(p, 1).then_some(p)
    }

    #[inline]
    fn touch(&mut self, page: usize) {
        if self.dirty.last() != Some(&page) {
            self.dirty.push(page);
        }
    }

    /// The big-endian value of the `n`-byte item (1, 2 or 4) at `p`,
    /// aligned to `n`, in one host access: each word's bytes sit reversed, so
    /// an aligned word is a host u32 as it stands and an aligned halfword a
    /// host u16 at `p ^ 2`.
    #[inline]
    pub fn get_be(&self, p: usize, n: u32) -> u32 {
        assert!(self.holds_at(p) && p % n as usize == 0, "O2 memory read {p:#x}/{n}");
        unsafe {
            match n {
                1 => *self.base.add(p ^ 3) as u32,
                2 => (self.base.add(p ^ 2) as *const u16).read_unaligned() as u32,
                _ => (self.base.add(p) as *const u32).read_unaligned(),
            }
        }
    }

    /// Store the low `n` bytes of `v` (1, 2 or 4) at `p`, aligned to `n`,
    /// big-endian.
    #[inline]
    pub fn put_be(&mut self, p: usize, n: u32, v: u32) {
        assert!(self.holds_at(p) && p % n as usize == 0, "O2 memory write {p:#x}/{n}");
        unsafe {
            match n {
                1 => *self.base.add(p ^ 3) = v as u8,
                2 => (self.base.add(p ^ 2) as *mut u16).write_unaligned(v as u16),
                _ => (self.base.add(p) as *mut u32).write_unaligned(v),
            }
        }
        self.touch(p / PAGE);
    }

    /// `data` from `p`, in guest order.
    pub fn write(&mut self, p: usize, data: &[u8]) {
        assert!(self.holds(p, data.len()), "O2 memory write {p:#x}+{:#x}", data.len());
        for (k, &b) in data.iter().enumerate() {
            unsafe { *self.base.add((p + k) ^ 3) = b };
        }
        for page in p / PAGE..=(p + data.len().max(1) - 1) / PAGE {
            self.touch(page);
        }
    }

    /// Write the bytes of `p..p + len` that `byte(k)` (k from 0) gives a
    /// value for, leaving the rest, each page bumped once: the memory engine
    /// fills and copies spans a tile row at a time.
    #[inline]
    pub fn write_span(&mut self, p: usize, len: usize, mut byte: impl FnMut(usize) -> Option<u8>) {
        if len == 0 {
            return;
        }
        assert!(self.holds(p, len), "O2 memory span {p:#x}+{len:#x}");
        for k in 0..len {
            if let Some(b) = byte(k) {
                unsafe { *self.base.add((p + k) ^ 3) = b };
            }
        }
        for page in p / PAGE..=(p + len - 1) / PAGE {
            self.touch(page);
        }
    }

    /// Store `val` through `mask` into `words` aligned 32-bit words from
    /// `p`, each page bumped once: the memory engine's clears and fills.
    pub fn fill_words_masked(&mut self, p: usize, words: usize, val: u32, mask: u32) {
        if words == 0 {
            return;
        }
        assert!(p % 4 == 0 && self.holds(p, 4 * words), "O2 memory fill {p:#x}+{words} words");
        for i in 0..words {
            unsafe {
                let q = self.base.add(p + 4 * i) as *mut u32;
                q.write_unaligned((q.read_unaligned() & !mask) | (val & mask));
            }
        }
        for page in p / PAGE..=(p + 4 * words - 1) / PAGE {
            self.touch(page);
        }
    }
}

impl RamWindow {
    /// `n` bytes from guest physical `p`, in guest order.
    pub fn bytes(&self, p: usize, n: usize) -> Vec<u8> {
        (p..p + n).map(|o| self[o]).collect()
    }
}

/// A byte, by guest physical address.
impl std::ops::Index<usize> for RamWindow {
    type Output = u8;
    fn index(&self, p: usize) -> &u8 {
        assert!(self.holds_at(p), "O2 memory read {p:#x}");
        unsafe { &*self.base.add(p ^ 3) }
    }
}

/// A [`RamWindow`] for one primitive: the decode taken when it starts, the
/// pages it wrote bumped when it ends.
pub struct Draw<'a>(std::sync::MutexGuard<'a, RamWindow>);

impl<'a> Draw<'a> {
    pub fn new(mut w: std::sync::MutexGuard<'a, RamWindow>) -> Self {
        w.begin();
        Draw(w)
    }
}

impl std::ops::Deref for Draw<'_> {
    type Target = RamWindow;
    fn deref(&self) -> &RamWindow {
        &self.0
    }
}

impl std::ops::DerefMut for Draw<'_> {
    fn deref_mut(&mut self) -> &mut RamWindow {
        &mut self.0
    }
}

impl Drop for Draw<'_> {
    fn drop(&mut self) {
        self.0.end();
    }
}


/// Whether `kits` is a fitting SGI's rules allow.
pub fn check_kits(kits: &[u32]) -> Result<(), String> {
    if kits.is_empty() {
        return Err("pair A must be populated".into());
    }
    if kits.len() > 4 {
        return Err(format!("{} kits, but an O2 has four DIMM pairs", kits.len()));
    }
    for (pair, &k) in kits.iter().enumerate() {
        let name = (b'A' + pair as u8) as char;
        if !KITS_MB.contains(&k) {
            return Err(format!("pair {name}: no {k} MB kit (SGI sold 32, 64, 128 and 256 MB)"));
        }
        if pair > 0 && k > kits[pair - 1] {
            return Err(format!("pair {name} ({k} MB) is larger than the pair before it; the largest kit goes in pair A"));
        }
    }
    Ok(())
}

/// The fewest kits that make `mb`, largest first.
pub fn kits_for(mb: u32) -> Result<Vec<u32>, String> {
    if mb == 0 || mb % 32 != 0 || mb > 1024 {
        return Err(format!("{mb} MB: an O2 holds 32 MB to 1 GB in steps of 32 MB"));
    }
    let mut kits = Vec::new();
    let mut left = mb;
    for &k in KITS_MB.iter().rev() {
        while left >= k {
            kits.push(k);
            left -= k;
        }
    }
    if kits.len() > 4 {
        return Err(format!("{mb} MB would take {} kits, but an O2 has four DIMM pairs", kits.len()));
    }
    Ok(kits)
}

/// Every fitting SGI's rules allow, as kit lists from pair A.
pub fn all_fittings() -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    fn walk(prefix: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
        if !prefix.is_empty() {
            out.push(prefix.clone());
        }
        if prefix.len() == 4 {
            return;
        }
        for &k in KITS_MB.iter() {
            if prefix.last().is_none_or(|&l| k <= l) {
                prefix.push(k);
                walk(prefix, out);
                prefix.pop();
            }
        }
    }
    walk(&mut Vec::new(), &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::BUS_OK;

    /// The CPU's window shows each byte where the decode puts it: both banks
    /// at RAM_BASE, the first 256 MB again from 0, and the mapped bitmap the
    /// JIT tests covers exactly those regions.
    #[test]
    fn the_window_shows_ram_where_the_decode_puts_it() {
        let m = O2Memory::from_kits(&[256]).unwrap();
        m.set_ctrl(0, 0x100);
        m.set_ctrl(1, 0x100 | 4);
        let (lo, hi) = (RAM_BASE + 0x0012_3450, RAM_BASE + 0x0900_0010);
        let bus = m.phys();
        assert_eq!(bus.write32(lo as u64, 0x1122_3344), BUS_OK);
        assert_eq!(bus.write16(hi as u64, 0xaabb), BUS_OK);
        let base = bus.ppmem_space().unwrap().window_base();
        let at = |p: u32| unsafe { *base.add(p as usize ^ 3) };
        assert_eq!([at(lo), at(lo + 1), at(lo + 2), at(lo + 3)], [0x11, 0x22, 0x33, 0x44]);
        assert_eq!([at(hi), at(hi + 1)], [0xaa, 0xbb], "bank 1 at its own base");
        assert_eq!(at(lo - RAM_BASE), 0x11, "the low alias");
        assert_eq!(bus.read32((lo - RAM_BASE) as u64).data, 0x1122_3344, "the low alias on the bus");
        let bits = bus.ppmem_space().unwrap().mapped_bitmap();
        let region = |p: u32| 1u64 << (p >> crate::ppmem::BITMAP_SHIFT);
        assert_ne!(bits & region(RAM_BASE), 0);
        assert_ne!(bits & region(RAM_BASE + 0x0fff_0000), 0, "both banks: 256 MB from RAM_BASE");
        assert_ne!(bits & region(0x0800_0000), 0, "the low alias");
        assert_eq!(bits & region(RAM_BASE + 0x1000_0000), 0, "nothing past the fitted memory");
    }

    fn put(m: &O2Memory, addr: u32, v: u64) {
        assert_eq!(m.phys().write64(addr as u64, v), BUS_OK, "{addr:#x}");
    }

    fn get(m: &O2Memory, addr: u32) -> Option<u64> {
        match m.decode(addr) {
            Decode::Ram(_) => Some(m.phys().read64(addr as u64).data),
            Decode::Empty => Some(0),
            Decode::None => None,
        }
    }

    /// The firmware's own sizing, as SGI's POST does it: every bank from the
    /// one being sized upwards pointed at the probe address as 128 MB, four
    /// patterns written (base, base + 32 MB - 8, base + 32 MB, base + 128 MB -
    /// 8), the bus cleared with zeros, and the four read back. Then the
    /// 128 MB banks are placed from the bottom of the window, the 32 MB ones
    /// after them, and the empty banks given bank 0's control.
    fn size_like_post(m: &O2Memory) -> ([u32; 8], u32) {
        let pat = [(0, 0xd0u64), (SLOT - 8, 0xd1), (SLOT, 0xd2), (BIG - 8, 0xd3)];
        let mut found = [0u32; 8];
        for bank in 0..8 {
            let at = bank as u32 * BIG;
            for b in bank..8 {
                m.set_ctrl(b, ((at >> 25) & 0x1f) as u64 | CTRL_128M);
            }
            for &(o, v) in &pat {
                if m.offset(RAM_BASE + at + o).is_some() {
                    put(m, RAM_BASE + at + o, v);
                }
            }
            let r: Vec<_> = pat.iter().map(|&(o, _)| get(m, RAM_BASE + at + o)).collect();
            found[bank] = if r == [Some(0xd2), Some(0xd3), Some(0xd2), Some(0xd3)] {
                32
            } else if r == [Some(0xd0), Some(0xd1), Some(0xd2), Some(0xd3)] {
                128
            } else {
                0
            };
        }
        let mut next = 0u32;
        for size in [128, 32] {
            for bank in 0..8 {
                if found[bank] == size {
                    let flag = if size == 128 { CTRL_128M } else { 0 };
                    m.set_ctrl(bank, ((next >> 25) & 0x1f) as u64 | flag);
                    next += size << 20;
                }
            }
        }
        let c0 = m.ctrl[0].load(Ordering::Relaxed);
        for bank in 0..8 {
            if found[bank] == 0 {
                m.set_ctrl(bank, c0);
            }
        }
        (found, next)
    }

    #[test]
    fn there_are_sixty_nine_fittings_from_32_mb_to_1_gb() {
        let all = all_fittings();
        assert_eq!(all.len(), 69, "multisets of 1 to 4 kits from 4 sizes");
        let totals: std::collections::BTreeSet<u32> = all.iter().map(|k| k.iter().sum()).collect();
        assert_eq!(totals.first(), Some(&32));
        assert_eq!(totals.last(), Some(&1024));
        assert!(!totals.contains(&992), "992 MB needs five kits");
        for k in &all {
            assert!(check_kits(k).is_ok(), "{k:?}");
        }
    }

    #[test]
    fn the_fitting_rules_are_sgis() {
        assert!(check_kits(&[]).is_err(), "pair A is never empty");
        assert!(check_kits(&[64, 128]).is_err(), "largest kit first");
        assert!(check_kits(&[48]).is_err(), "no such kit");
        assert!(check_kits(&[32; 5]).is_err(), "four pairs");
        assert_eq!(kits_for(1024).unwrap(), vec![256; 4]);
        assert_eq!(kits_for(224).unwrap(), vec![128, 64, 32]);
        assert!(kits_for(992).is_err());
        assert!(kits_for(100).is_err());
    }

    /// Every fitting an O2 can have sizes to what is fitted, bank by bank,
    /// and once placed, the whole of it is distinct memory from the bottom of
    /// the window up.
    #[test]
    fn every_fitting_sizes_the_way_post_sizes_it() {
        for kits in all_fittings() {
            let m = O2Memory::from_kits(&kits).unwrap();
            let (found, total) = size_like_post(&m);
            let want: Vec<u32> = m.bank_sizes().iter().map(|s| s >> 20).collect();
            assert_eq!(found.to_vec(), want, "{kits:?}");
            assert_eq!(total, m.total(), "{kits:?}");
            assert_eq!(total >> 20, kits.iter().sum::<u32>(), "{kits:?}");

            // Each 32 MB of the placed window is a different 32 MB of memory.
            for s in 0..total / SLOT {
                put(&m, RAM_BASE + s * SLOT + 0x40, 0x5100 + s as u64);
            }
            for s in 0..total / SLOT {
                assert_eq!(get(&m, RAM_BASE + s * SLOT + 0x40), Some(0x5100 + s as u64), "{kits:?}: slot {s} overlaps another");
            }
            assert_eq!(m.decode(RAM_BASE + total), Decode::None, "{kits:?}: nothing above the top");
            // The last byte is there, and the low alias is the window's start.
            assert!(m.offset(RAM_BASE + total - 1).is_some(), "{kits:?}");
            assert_eq!(get(&m, 0x40), get(&m, RAM_BASE + 0x40));
        }
    }

    #[test]
    fn the_low_alias_stops_at_256_mb() {
        let m = O2Memory::with_total(1024).unwrap();
        size_like_post(&m);
        put(&m, LOW_ALIAS_SIZE - 8, 0x0123_4567_89ab_cdef);
        assert_eq!(get(&m, RAM_BASE + LOW_ALIAS_SIZE - 8), Some(0x0123_4567_89ab_cdef));
        assert_eq!(m.decode(LOW_ALIAS_SIZE), Decode::None, "CRIME and MACE live above it");
        assert!(m.offset(RAM_BASE + (1 << 30) - 8).is_some(), "the top of a full machine");
    }

    #[test]
    fn dma_reaches_all_of_memory_below_1_gb() {
        let m = O2Memory::with_total(1024).unwrap();
        size_like_post(&m);
        let top = (1u32 << 30) - 0x5000;
        assert_eq!(m.dma_offset(top), m.offset(RAM_BASE + top), "OpenBSD's bus address for high memory");
        assert_eq!(m.dma_offset(0x1000), m.offset(0x1000), "and the low 256 MB as before");
        assert_eq!(m.dma_offset(RAM_BASE + 0x2000_0000), m.offset(RAM_BASE + 0x2000_0000), "and the CPU's own addresses");
        assert!(m.offset(0x3fff_b000).is_none(), "which the CPU cannot use: that is not its alias");
    }

    #[test]
    fn a_small_bank_repeats_across_a_large_setting() {
        let m = O2Memory::with_total(32).unwrap();
        m.set_ctrl(0, CTRL_128M);
        put(&m, RAM_BASE + 8, 0xfeed);
        assert_eq!(get(&m, RAM_BASE + 3 * SLOT + 8), Some(0xfeed));
    }

    /// A 128 MB bank based where its size does not divide counts its bytes
    /// from that base, as the decode does, on the bus and in the window.
    #[test]
    fn a_bank_counts_from_its_base() {
        let m = O2Memory::with_total(128).unwrap();
        m.set_ctrl(0, CTRL_128M | 1);
        put(&m, RAM_BASE + SLOT, 0xbeef);
        assert_eq!(m.phys().banks()[0].read64(0).data, 0xbeef, "offset 0 of the bank");
        let base = m.phys().ppmem_space().unwrap().window_base();
        let w = unsafe { *(base.add((RAM_BASE + SLOT) as usize) as *const u64) }.rotate_left(32);
        assert_eq!(w, 0xbeef, "the window agrees");
    }

    const STACK_TAG: u32 = 0xc8; // dirty, page 0: what POST stores on an R10000

    /// What each path sees of page 0 once the stack is live, and that memory
    /// ends up holding what was written uncached.
    #[test]
    fn the_stack_and_the_uncached_page_are_different_bytes() {
        let m = O2Memory::with_total(64).unwrap();
        m.set_ctrl(0, 0);
        let bus = m.phys();
        let base = bus.ppmem_space().unwrap().window_base();
        let cached = |o: usize, base: *mut u8| unsafe { (*(base.add(o) as *const u64)).rotate_left(32) };
        let store = |o: usize, v: u64, base: *mut u8| unsafe { *(base.add(o) as *mut u64) = v.rotate_left(32) };
        assert_eq!(bus.write64(0xff0, 0x1111), BUS_OK);
        m.stack_tag_store(0x0);
        m.stack_tag_store(STACK_TAG);
        assert_eq!(bus.read64(0xff0).data, 0x1111, "the uncached page starts as a copy");
        store(0xff0, 0x2222, base);
        assert_eq!(bus.write64(0xff8, 0x3333), BUS_OK);
        assert_eq!(cached(0xff0, base), 0x2222, "the stack, cached");
        assert_eq!(cached(0xff8, base), 0, "the test's write does not reach the stack");
        assert_eq!(bus.read64(0xff0).data, 0x1111, "and the stack does not reach the test");
        assert_eq!(bus.read64(0xff8).data, 0x3333);
        assert_eq!(bus.read64(RAM_BASE as u64 + 0xff8).data, 0x3333, "through the window's address too");
        assert_eq!(bus.write64(0x1000, 0x4444), BUS_OK);
        assert_eq!(cached(0x1000, base), 0x4444, "the next page is memory");
        m.stack_tag_store(0x0);
        assert_eq!(cached(0xff8, base), 0x3333, "memory holds what was written uncached");
        assert_eq!(bus.read64(0xff0).data, 0x1111);
        store(0xff8, 0x5555, base);
        assert_eq!(bus.read64(0xff8).data, 0x5555, "one memory again");
    }

    /// Sizing reads back through a 32 MB bank repeating under a 128 MB
    /// setting, so page 0 has to answer the same way at every repeat.
    #[test]
    fn the_uncached_page_follows_a_small_bank_repeating() {
        let m = O2Memory::with_total(32).unwrap();
        m.stack_tag_store(STACK_TAG);
        m.set_ctrl(0, CTRL_128M);
        put(&m, RAM_BASE + 8, 0xfeed);
        assert_eq!(get(&m, RAM_BASE + 3 * SLOT + 8), Some(0xfeed));
        assert_eq!(get(&m, 3 * SLOT + 8), Some(0xfeed), "the low alias");
        m.stack_tag_store(0);
        assert_eq!(m.phys().banks()[0].read64(8).data, 0xfeed);
    }

    /// At reset one 32 MB slot is mapped, which ppmem's bitmap does not count;
    /// the cached path must still reach the bank, not the bus, or the stack
    /// moves out from under POST when sizing completes the region.
    #[test]
    fn the_cached_path_is_on_the_bank_from_reset() {
        let m = O2Memory::with_total(32).unwrap();
        let sp = m.phys().ppmem_space().unwrap();
        let region = 1u64 << 0;
        assert_eq!(sp.mapped_bitmap() & region, 0, "a hole at reset");
        m.stack_tag_store(STACK_TAG);
        assert_ne!(sp.mapped_bitmap() & region, 0, "filled while the stack is live");
        m.stack_tag_store(0);
        assert_eq!(sp.mapped_bitmap() & region, 0, "and only then");
    }

    /// Only a valid tag for page 0 starts the stack; other firmware's tags
    /// and a kernel's invalid ones leave the decode alone.
    #[test]
    fn other_tags_leave_the_decode_alone() {
        let m = O2Memory::with_total(32).unwrap();
        for tag in [0u32, 0x1_00c8, 0x08] {
            m.stack_tag_store(tag);
            assert!(m.stack_page_words().is_none(), "{tag:#x}");
        }
    }

    /// A snapshot taken during POST comes back with both views.
    #[test]
    fn a_snapshot_carries_the_uncached_page() {
        let m = O2Memory::with_total(32).unwrap();
        m.stack_tag_store(STACK_TAG);
        put(&m, 0x20, 0xabcd);
        let words = m.stack_page_words().unwrap();
        let n = O2Memory::with_total(32).unwrap();
        n.restore_stack_page(&words).unwrap();
        assert_eq!(get(&n, 0x20), Some(0xabcd));
        n.stack_tag_store(0);
        assert_eq!(n.phys().banks()[0].read64(0x20).data, 0xabcd);
    }

    /// An empty bank's slots read zero; with no control matching, the bus's
    /// default answers.
    #[test]
    fn empty_and_unmatched_slots() {
        let m = O2Memory::from_kits(&[32]).unwrap();
        m.set_ctrl(0, 0);
        m.set_ctrl(1, 2); // bank 1 is not fitted
        for b in 2..8 {
            m.set_ctrl(b, 0);
        }
        assert_eq!(m.decode(RAM_BASE + 2 * SLOT), Decode::Empty);
        assert_eq!(get(&m, RAM_BASE + 2 * SLOT), Some(0));
        assert_eq!(m.decode(RAM_BASE + 5 * SLOT), Decode::None);
        assert_ne!(m.phys().write32((RAM_BASE + 5 * SLOT) as u64, 1), BUS_OK, "nothing answers");
    }
}
