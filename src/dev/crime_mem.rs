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
//! The banks live packed, in bank order, in one ppmem object, so a device
//! addresses memory by byte offset into it (byte `o` at `host + (o ^ 3)`, the
//! layout every iris RAM keeps). The same object is mapped into a ppmem
//! window as the decode stands, one 32 MB slot at a time, which is what the
//! CPU's cache and the JIT read and write directly, with no lock and no bus,
//! and what the rendering engine draws through ([`RamWindow`]).
//!
//! `O2Memory` takes no lock of its own and is shared as a plain `Arc`. Its
//! bytes are a host mapping every master already reaches directly, racing
//! the way the hardware's masters do. The decode is atomics: the firmware
//! programs the bank controls once at power-on, and a guest that moved a
//! bank while a master was using it would be undefined on the real machine
//! too, so readers never wait. Only two rebuilds of the decode are kept
//! from interleaving.

use crate::ip32::{LOW_ALIAS_SIZE, RAM_BASE, RAM_WINDOW_SIZE};
use crate::ppmem::{MappedMemory, PpMemSpace, PpMemory};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

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
    /// A byte of `bytes`.
    Ram(usize),
    /// A bank control matches but no DIMM is there.
    Empty,
    /// Nothing answers.
    None,
}

pub struct O2Memory {
    /// The fitted banks back to back, in bank order: a ppmem object, so the
    /// window below can map it.
    store: PpMemory,
    /// `store`'s own mapping; byte `o` is at `host + (o ^ 3)`.
    host: *mut u8,
    /// Bytes fitted (the store may be rounded up to ppmem's 1 MB unit).
    total: usize,
    /// The CPU's view: the guest physical space with each RAM slot mapped,
    /// rebuilt whenever a bank control changes.
    space: PpMemSpace,
    /// Each CRIME bank's real size in bytes (0 for nothing fitted) and where
    /// it starts in `bytes`. Fixed when the memory is built.
    size: [u32; 8],
    start: [u32; 8],
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
    shared: std::sync::Arc<WindowState>,
}

// The raw pointer is into `store`, which lives as long as this does. Every
// master writes through it, or through the window onto the same bytes, with
// no lock: they race only the way the hardware's own masters do.
unsafe impl Send for O2Memory {}
unsafe impl Sync for O2Memory {}

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

    fn from_banks(size: [u32; 8]) -> Self {
        let mut start = [0u32; 8];
        let mut total = 0u32;
        for i in 0..8 {
            assert!(size[i] == 0 || (size[i].is_power_of_two() && size[i] <= BIG), "bank {i}: {}", size[i]);
            start[i] = total;
            total += size[i];
        }
        let store = PpMemory::new((total as usize).div_ceil(1 << 20).max(1));
        let space = PpMemSpace::over(std::slice::from_ref(&store)).expect("ppmem: O2 memory window");
        let host = store.host_ptr();
        let m = Self {
            store,
            host,
            total: total as usize,
            space,
            size,
            start,
            ctrl: Default::default(),
            slots: std::array::from_fn(|_| AtomicU32::new(0)),
            decode_gen: AtomicU64::new(0),
            rebuilding: std::sync::Mutex::new(()),
            shared: Default::default(),
        };
        // At reset every control is zero: all eight banks at the bottom of
        // the window, 32 MB each, so bank 0 answers there.
        m.rebuild();
        m
    }

    /// Write `data` at byte offset `off` (from [`Self::offset`] or
    /// [`Self::dma_offset`]) and bump the generation of every page it touches.
    /// CPU stores and DMA both come here, so compiled code never outlives the
    /// bytes it was compiled from.
    pub fn write(&self, off: usize, data: &[u8]) {
        assert!(off + data.len() <= self.total, "O2 memory write past the end");
        for (k, &b) in data.iter().enumerate() {
            unsafe { *self.host.add((off + k) ^ 3) = b };
        }
        let last = (off + data.len().max(1) - 1) / PAGE;
        for page in off / PAGE..=last {
            self.touch(page);
        }
    }

    /// Read `buf.len()` bytes from offset `off`, in guest order.
    pub fn read_into(&self, off: usize, buf: &mut [u8]) {
        assert!(off + buf.len() <= self.total, "O2 memory read past the end");
        for (k, b) in buf.iter_mut().enumerate() {
            *b = unsafe { *self.host.add((off + k) ^ 3) };
        }
    }

    /// Write the bytes of `off..off + len` that `byte(k)` (k from 0) gives
    /// a value for, leaving the rest, and bump each page once: the memory
    /// engine fills and copies spans a tile row at a time, and a call per
    /// byte through [`Self::write`] was most of its time.
    #[inline]
    pub fn write_span(&self, off: usize, len: usize, mut byte: impl FnMut(usize) -> Option<u8>) {
        if len == 0 {
            return;
        }
        assert!(off + len <= self.total, "O2 memory span {off:#x}+{len:#x} past the end");
        for k in 0..len {
            if let Some(b) = byte(k) {
                unsafe { *self.host.add((off + k) ^ 3) = b };
            }
        }
        for page in off / PAGE..=(off + len - 1) / PAGE {
            self.touch(page);
        }
    }

    /// Store `val` through `mask` into `words` aligned 32-bit words from
    /// `off` (big-endian, as [`Self::put_be`]), each page bumped once: the
    /// memory engine's clears and fills.
    pub fn fill_words_masked(&self, off: usize, words: usize, val: u32, mask: u32) {
        if words == 0 {
            return;
        }
        assert!(off % 4 == 0 && off + 4 * words <= self.total, "O2 memory fill {off:#x}+{words} words");
        for i in 0..words {
            unsafe {
                let p = self.host.add(off + 4 * i) as *mut u32;
                p.write_unaligned((p.read_unaligned() & !mask) | (val & mask));
            }
        }
        for page in off / PAGE..=(off + 4 * words - 1) / PAGE {
            self.touch(page);
        }
    }

    /// Every fitted byte to `path`, as the host keeps them: a snapshot is
    /// restored only on a host of the same architecture.
    pub fn save_raw(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.raw(0, self.total))
    }

    /// The bytes [`Self::save_raw`] wrote, back, with every page's
    /// generation bumped so no compiled code outlives them.
    pub fn load_raw(&self, path: &std::path::Path) -> std::io::Result<()> {
        let data = std::fs::read(path)?;
        self.restore_raw(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Every fitted byte as host words, for a snapshot's chunk store.
    pub fn words(&self) -> &[u32] {
        // SAFETY: the store is page-aligned and `total` a multiple of 4.
        unsafe { std::slice::from_raw_parts(self.host as *const u32, self.total / 4) }
    }

    /// [`Self::restore_raw`] for words [`Self::words`] gave.
    pub fn restore_words(&self, words: &[u32]) -> Result<(), String> {
        // SAFETY: a view of the same bytes; u8 has no alignment to keep.
        self.restore_raw(unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) })
    }

    /// Put back bytes [`Self::raw`] or [`Self::words`] gave, with every
    /// page's generation bumped so no compiled code outlives them.
    pub fn restore_raw(&self, data: &[u8]) -> Result<(), String> {
        if data.len() != self.total {
            return Err(format!("saved memory is {} bytes, this machine has {}", data.len(), self.total));
        }
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.host, self.total) };
        for page in 0..self.total.div_ceil(PAGE) {
            self.touch(page);
        }
        Ok(())
    }

    /// Page `page` was written: bump its generation.
    #[inline]
    fn touch(&self, page: usize) {
        #[cfg(feature = "jitv2")]
        self.store.bump_gen_off(page * PAGE);
        #[cfg(not(feature = "jitv2"))]
        let _ = page;
    }

    /// The big-endian value of the `n`-byte item (1, 2 or 4) at `off`,
    /// which must be aligned to `n`, in one host access: each word's bytes
    /// sit reversed, so an aligned word is a host u32 as it stands and an
    /// aligned halfword a host u16 at `off ^ 2`. For the rendering engine,
    /// which moved pixels a byte at a time.
    #[inline]
    pub fn get_be(&self, off: usize, n: u32) -> u32 {
        assert!(off + n as usize <= self.total && off % n as usize == 0, "O2 memory read {off:#x}/{n}");
        unsafe {
            match n {
                1 => *self.host.add(off ^ 3) as u32,
                2 => (self.host.add(off ^ 2) as *const u16).read_unaligned() as u32,
                _ => (self.host.add(off) as *const u32).read_unaligned(),
            }
        }
    }

    /// Store the low `n` bytes of `v` (1, 2 or 4) at `off`, aligned to `n`,
    /// big-endian, as [`Self::write`] would.
    #[inline]
    pub fn put_be(&self, off: usize, n: u32, v: u32) {
        assert!(off + n as usize <= self.total && off % n as usize == 0, "O2 memory write {off:#x}/{n}");
        unsafe {
            match n {
                1 => *self.host.add(off ^ 3) = v as u8,
                2 => (self.host.add(off ^ 2) as *mut u16).write_unaligned(v as u16),
                _ => (self.host.add(off) as *mut u32).write_unaligned(v),
            }
        }
        self.touch(off / PAGE);
    }

    /// The generation counter for the page holding byte offset `off`.
    #[cfg(feature = "jitv2")]
    pub fn gen_ptr(&self, off: usize) -> *const std::sync::atomic::AtomicU64 {
        self.store.gen_ptr(off as u64)
    }

    /// `n` bytes from offset `off`, in guest order.
    pub fn bytes(&self, off: usize, n: usize) -> Vec<u8> {
        (off..off + n).map(|o| self[o]).collect()
    }

    /// `n` bytes from `off` as the host keeps them (each 32-bit word's bytes
    /// reversed): for copying whole words to something that reads them back
    /// the same way. `off` and `n` must be multiples of 4.
    pub fn raw(&self, off: usize, n: usize) -> &[u8] {
        assert!(off % 4 == 0 && n % 4 == 0 && off + n <= self.total);
        unsafe { std::slice::from_raw_parts(self.host.add(off), n) }
    }

    /// Changes whenever an address may decode differently.
    pub fn decode_gen(&self) -> u64 {
        self.decode_gen.load(Ordering::Acquire)
    }

    /// Bytes fitted.
    pub fn len(&self) -> usize {
        self.total
    }

    /// The CPU's window: guest physical `p` at `window_base() + p` wherever
    /// the mapped bitmap says RAM is.
    pub fn space(&self) -> &PpMemSpace {
        &self.space
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

    /// Bank `i`'s control register was written.
    pub fn set_ctrl(&self, i: usize, val: u64) {
        let _one = self.rebuilding.lock().unwrap();
        if self.ctrl[i].swap(val, Ordering::AcqRel) != val {
            self.rebuild_locked();
        }
    }

    fn rebuild(&self) {
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

    /// For a slot of bank `bank` beginning `chunk` slots into it: the
    /// bank's first byte in the store, the slot's offset into the bank
    /// before wrapping, and the bank's size less one.
    fn ram_slot(&self, bank: u8, chunk: u8) -> (u32, u32, u32) {
        let b = bank as usize;
        (self.start[b], chunk as u32 * SLOT, self.size[b] - 1)
    }

    /// Lay the window out as the decode now stands: every slot that decodes
    /// to RAM mapped from its bank's bytes in the store, at its place in the
    /// window and again in the low alias. Empty and unmatched slots stay
    /// unmapped, so the CPU reaches them through the bus as before.
    fn remap(&self, slots: &[Slot; SLOTS]) {
        use std::sync::atomic::Ordering::Release;
        self.shared.mapped.store(0, Release);
        self.space.clear_mappings();
        let mut mapped = 0u32;
        for (s, slot) in slots.iter().enumerate() {
            let Slot::Ram { bank, chunk } = *slot else { continue };
            let (start, chunk, mask) = self.ram_slot(bank, chunk);
            if mask < SLOT - 1 {
                continue; // a test's bank smaller than a slot: bus only
            }
            let src = (start + (chunk & mask)) as u64;
            let at = s as u64 * SLOT as u64;
            if self.space.map_range(0, src, RAM_BASE as u64 + at, SLOT as u64).is_ok() {
                mapped |= 1 << s;
            }
            if at < LOW_ALIAS_SIZE as u64 {
                let _ = self.space.map_range(0, src, at, SLOT as u64);
            }
        }
        self.shared.decode_gen.store(self.decode_gen.load(Ordering::Acquire), Release);
        self.shared.mapped.store(mapped, Release);
    }

    /// A view of memory through the window, for a master on its own thread
    /// (the rendering engine).
    pub fn window(&self) -> RamWindow {
        RamWindow {
            base: self.space.window_base(),
            #[cfg(feature = "jitv2")]
            gens: self.space.gen_window_base(),
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
            Slot::Ram { bank, chunk } => {
                let (start, chunk, mask) = self.ram_slot(bank, chunk);
                Decode::Ram((start + ((chunk + off % SLOT) & mask)) as usize)
            }
        }
    }

    /// The byte offset of a physical address, if memory is there.
    pub fn offset(&self, addr: u32) -> Option<usize> {
        match self.decode(addr) {
            Decode::Ram(i) => Some(i),
            _ => None,
        }
    }

    /// The same for an address a DMA master (a PCI device, the Ethernet)
    /// hands the memory controller. Below 1 GB a bus address is an offset
    /// into the whole memory window, not just its first 256 MB: the CPU's
    /// low alias stops there because CRIME and MACE sit at 0x14000000 and up,
    /// but nothing on the bus does. OpenBSD programs buffers above 256 MB this
    /// way (0x3fffb000 for the top of a 1 GB machine); from `RAM_BASE` up an
    /// address is read as the CPU reads it, which is how Linux does it.
    pub fn dma_offset(&self, addr: u32) -> Option<usize> {
        if addr < RAM_WINDOW_SIZE {
            return self.offset(RAM_BASE + addr);
        }
        self.offset(addr)
    }
}

/// A device indexes memory by byte offset once it has one.
impl std::ops::Index<usize> for O2Memory {
    type Output = u8;
    fn index(&self, o: usize) -> &u8 {
        assert!(o < self.total, "O2 memory offset {o:#x} past the end");
        unsafe { &*self.host.add(o ^ 3) }
    }
}

/// Device tests set memory up by hand. Everything else writes through
/// [`O2Memory::write`], which keeps the page generations honest; the lib
/// built without `cfg(test)` has no other way in.
#[cfg(test)]
impl std::ops::IndexMut<usize> for O2Memory {
    fn index_mut(&mut self, o: usize) -> &mut u8 {
        assert!(o < self.total, "O2 memory offset {o:#x} past the end");
        unsafe { &mut *self.host.add(o ^ 3) }
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

    /// As [`O2Memory::get_be`].
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

    /// As [`O2Memory::put_be`].
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

    /// As [`O2Memory::write`].
    pub fn write(&mut self, p: usize, data: &[u8]) {
        assert!(self.holds(p, data.len()), "O2 memory write {p:#x}+{:#x}", data.len());
        for (k, &b) in data.iter().enumerate() {
            unsafe { *self.base.add((p + k) ^ 3) = b };
        }
        for page in p / PAGE..=(p + data.len().max(1) - 1) / PAGE {
            self.touch(page);
        }
    }

    /// As [`O2Memory::write_span`].
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

    /// As [`O2Memory::fill_words_masked`].
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

    #[test]
    fn word_access_matches_the_byte_view() {
        let m = O2Memory::single(1 << 20);
        m.write(0x100, &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
        assert_eq!(m.get_be(0x100, 4), 0x1122_3344);
        assert_eq!(m.get_be(0x104, 2), 0x5566);
        assert_eq!(m.get_be(0x106, 2), 0x7788);
        assert_eq!(m.get_be(0x105, 1), 0x66);
        m.put_be(0x100, 4, 0xaabb_ccdd);
        m.put_be(0x106, 2, 0x1234);
        m.put_be(0x104, 1, 0xee);
        assert_eq!(m.bytes(0x100, 8), [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0x66, 0x12, 0x34]);
    }


    /// The CPU's window shows each byte where the decode puts it: both banks
    /// at RAM_BASE, the first 256 MB again from 0, and the mapped bitmap the
    /// JIT tests covers exactly those regions.
    #[test]
    fn the_window_shows_ram_where_the_decode_puts_it() {
        let m = O2Memory::from_kits(&[256]).unwrap();
        m.set_ctrl(0, 0x100);
        m.set_ctrl(1, 0x100 | 4);
        let (lo, hi) = (RAM_BASE + 0x0012_3450, RAM_BASE + 0x0900_0010);
        let (a, b) = (m.offset(lo).unwrap(), m.offset(hi).unwrap());
        m.write(a, &[0x11, 0x22, 0x33, 0x44]);
        m.write(b, &[0xaa, 0xbb]);
        let base = m.space().window_base();
        let at = |p: u32| unsafe { *base.add(p as usize ^ 3) };
        assert_eq!([at(lo), at(lo + 1), at(lo + 2), at(lo + 3)], [0x11, 0x22, 0x33, 0x44]);
        assert_eq!([at(hi), at(hi + 1)], [0xaa, 0xbb], "bank 1 at its own base");
        assert_eq!(at(lo - RAM_BASE), 0x11, "the low alias");
        let bits = m.space().mapped_bitmap();
        let region = |p: u32| 1u64 << (p >> crate::ppmem::BITMAP_SHIFT);
        assert_ne!(bits & region(RAM_BASE), 0);
        assert_ne!(bits & region(RAM_BASE + 0x0fff_0000), 0, "both banks: 256 MB from RAM_BASE");
        assert_ne!(bits & region(0x0800_0000), 0, "the low alias");
        assert_eq!(bits & region(RAM_BASE + 0x1000_0000), 0, "nothing past the fitted memory");
    }

    fn put(m: &mut O2Memory, addr: u32, v: u64) {
        let i = m.offset(addr).unwrap();
        m.write(i, &v.to_be_bytes());
    }

    fn get(m: &O2Memory, addr: u32) -> Option<u64> {
        match m.decode(addr) {
            Decode::Ram(i) => Some(u64::from_be_bytes(m.bytes(i, (i + 8) - (i)).try_into().unwrap())),
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
    fn size_like_post(m: &mut O2Memory) -> ([u32; 8], u32) {
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
            let mut m = O2Memory::from_kits(&kits).unwrap();
            let (found, total) = size_like_post(&mut m);
            let want: Vec<u32> = m.bank_sizes().iter().map(|s| s >> 20).collect();
            assert_eq!(found.to_vec(), want, "{kits:?}");
            assert_eq!(total, m.total(), "{kits:?}");
            assert_eq!(total >> 20, kits.iter().sum::<u32>(), "{kits:?}");

            // Each 32 MB of the placed window is a different 32 MB of memory.
            let firsts: std::collections::BTreeSet<usize> =
                (0..total / SLOT).map(|s| m.offset(RAM_BASE + s * SLOT).unwrap()).collect();
            assert_eq!(firsts.len() as u32, total / SLOT, "{kits:?}: placed banks overlap");
            assert_eq!(m.decode(RAM_BASE + total), Decode::None, "{kits:?}: nothing above the top");
            // The last byte is there, and the low alias is the window's start.
            assert!(m.offset(RAM_BASE + total - 1).is_some(), "{kits:?}");
            assert_eq!(m.offset(0x1000), m.offset(RAM_BASE + 0x1000));
        }
    }

    #[test]
    fn the_low_alias_stops_at_256_mb() {
        let mut m = O2Memory::with_total(1024).unwrap();
        size_like_post(&mut m);
        assert_eq!(m.offset(LOW_ALIAS_SIZE - 8), m.offset(RAM_BASE + LOW_ALIAS_SIZE - 8));
        assert_eq!(m.decode(LOW_ALIAS_SIZE), Decode::None, "CRIME and MACE live above it");
        assert!(m.offset(RAM_BASE + (1 << 30) - 8).is_some(), "the top of a full machine");
    }

    #[test]
    fn dma_reaches_all_of_memory_below_1_gb() {
        let mut m = O2Memory::with_total(1024).unwrap();
        size_like_post(&mut m);
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
        assert_eq!(m.offset(RAM_BASE + 8), m.offset(RAM_BASE + 3 * SLOT + 8));
    }
}
