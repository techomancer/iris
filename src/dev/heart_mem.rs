//! Octane main memory: the DIMMs that are fitted, and where HEART's MEMCFG
//! puts them.
//!
//! HEART has four banks (two DIMM slots each) and a 32-bit MEMCFG word per
//! bank: bit 31 valid, bits 21:16 the size less one in 32 MB units, bits
//! 8:0 the base in 32 MB units from 0x20000000 (`dev::heart`). The PROM
//! sizes a bank by programming it large and finding where its contents
//! repeat, so a bank told it is bigger than its DIMMs ignores the address
//! lines it does not have and repeats through its window. When two banks
//! claim an address the lower-numbered one answers. The first 16 KB of
//! memory also answers at physical 0 (the exception vectors live there).
//!
//! The fitted banks live packed, in bank order, in one ppmem object (byte
//! `o` at `host + (o ^ 3)`, the layout every iris RAM keeps), and the
//! object is mapped into a ppmem window as MEMCFG stands, 32 MB at a time,
//! which is what the CPU's cache and the JIT read and write directly. The
//! window is 4 GB: a bank the PROM places beyond it is reachable only
//! through the bus, and then only below 4 GB, until the bus takes 40-bit
//! addresses.

use crate::dev::heart::{MEMCFG_ADDR_MASK, MEMCFG_SIZE_MASK, MEMCFG_SIZE_SHIFT, MEMCFG_UNIT_SHIFT, MEMCFG_VALID};
use crate::ppmem::{MappedMemory, PpMemSpace, PpMemory, WINDOW_SIZE};

/// Where memory starts.
pub const RAM_BASE: u64 = 0x2000_0000;
/// How much of it also answers at physical 0.
pub const LOW_ALIAS_SIZE: u64 = 0x4000;
/// MEMCFG's unit, and the mapping granule.
const UNIT: u64 = 32 << 20;
/// The page jitv2 tracks code by.
const PAGE: usize = 4096;

/// A bank as MEMCFG places it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Bank {
    /// Physical base and the window MEMCFG gives it.
    base: u64,
    span: u64,
    /// Its bytes in the store, and its real size less one.
    start: usize,
    mask: u64,
    /// Valid in MEMCFG but no DIMMs fitted: the PROM probing for memory.
    empty: bool,
}

/// Where a physical address lands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decode {
    /// A byte offset into the store.
    Ram(usize),
    /// A valid bank with nothing in it: writes go nowhere, reads see what
    /// the memory bus last carried (zero, as the O2's CRIME is modelled).
    Empty,
    /// No bank answers.
    None,
}

pub struct HeartMemory {
    store: PpMemory,
    host: *mut u8,
    total: usize,
    space: PpMemSpace,
    /// Each bank's real size in bytes (0 for nothing fitted) and where it
    /// starts in the store.
    size: [u64; 4],
    start: [usize; 4],
    memcfg: [u32; 4],
    banks: [Option<Bank>; 4],
}

// The raw pointer is into `store`, which lives as long as this does; writers
// through it hold the memory's lock, and the CPU reaches the same bytes
// through the window the way the hardware's own masters do.
unsafe impl Send for HeartMemory {}
unsafe impl Sync for HeartMemory {}

impl HeartMemory {
    /// `banks_mb` is the DIMM pair in each bank, in MB (a multiple of 32,
    /// 0 for an empty bank). `memcfg` is HEART's view of them at reset.
    pub fn new(banks_mb: [u32; 4], memcfg: [u32; 4]) -> Self {
        let mut size = [0u64; 4];
        let mut start = [0usize; 4];
        let mut total = 0usize;
        for i in 0..4 {
            size[i] = (banks_mb[i] as u64) << 20;
            start[i] = total;
            total += size[i] as usize;
        }
        let store = PpMemory::new(total.div_ceil(1 << 20).max(1));
        let space = PpMemSpace::over(std::slice::from_ref(&store)).expect("ppmem: Octane memory window");
        let host = store.host_ptr();
        let mut m = Self { store, host, total, space, size, start, memcfg: [0; 4], banks: [None; 4] };
        m.set_memcfg(memcfg);
        m
    }

    /// HEART's MEMCFG words were written: place the banks again.
    pub fn set_memcfg(&mut self, words: [u32; 4]) {
        self.memcfg = words;
        for i in 0..4 {
            let w = words[i];
            self.banks[i] = if w & MEMCFG_VALID == 0 {
                None
            } else {
                let span = ((((w & MEMCFG_SIZE_MASK) >> MEMCFG_SIZE_SHIFT) as u64) + 1) << MEMCFG_UNIT_SHIFT;
                let base = RAM_BASE + (((w & MEMCFG_ADDR_MASK) as u64) << MEMCFG_UNIT_SHIFT);
                let empty = self.size[i] == 0;
                Some(Bank { base, span, start: self.start[i], mask: self.size[i].saturating_sub(1), empty })
            };
        }
        self.remap();
    }

    /// Lay the window out as the banks now stand: every 32 MB of a bank's
    /// span that falls inside the window mapped from the bank's bytes,
    /// repeating through its real size. Later banks never shadow earlier
    /// ones.
    pub fn remap(&mut self) {
        self.space.clear_mappings();
        let mut taken: Vec<(u64, u64)> = Vec::new();
        for bank in self.banks.iter().flatten() {
            if bank.empty || bank.mask < UNIT - 1 {
                continue; // nothing to map, or smaller than a granule: bus only
            }
            let mut at = bank.base;
            while at < bank.base + bank.span && at + UNIT <= WINDOW_SIZE {
                if !taken.iter().any(|&(s, e)| at >= s && at < e) {
                    let src = bank.start as u64 + ((at - bank.base) & bank.mask);
                    let _ = self.space.map_range(0, src, at, UNIT);
                    taken.push((at, at + UNIT));
                }
                at += UNIT;
            }
        }
    }

    /// Where `addr` lands, the low alias included.
    pub fn decode(&self, addr: u64) -> Decode {
        let addr = if addr < LOW_ALIAS_SIZE { RAM_BASE + addr } else { addr };
        for bank in self.banks.iter().flatten() {
            if addr >= bank.base && addr < bank.base + bank.span {
                if bank.empty {
                    return Decode::Empty;
                }
                return Decode::Ram(bank.start + ((addr - bank.base) & bank.mask) as usize);
            }
        }
        Decode::None
    }

    /// The big-endian value of the `n`-byte item (1, 2, 4 or 8) at `off`,
    /// aligned to `n`, in one or two host accesses: each word's bytes sit
    /// reversed, so an aligned word is a host u32 as it stands.
    #[inline]
    pub fn get_be(&self, off: usize, n: u32) -> u64 {
        debug_assert!(off + n as usize <= self.total && off % n as usize == 0);
        unsafe {
            match n {
                1 => *self.host.add(off ^ 3) as u64,
                2 => (self.host.add(off ^ 2) as *const u16).read_unaligned() as u64,
                4 => (self.host.add(off) as *const u32).read_unaligned() as u64,
                _ => {
                    let hi = (self.host.add(off) as *const u32).read_unaligned() as u64;
                    let lo = (self.host.add(off + 4) as *const u32).read_unaligned() as u64;
                    hi << 32 | lo
                }
            }
        }
    }

    /// Store the low `n` bytes of `v` at `off`, aligned to `n`, big-endian.
    #[inline]
    pub fn put_be(&mut self, off: usize, n: u32, v: u64) {
        debug_assert!(off + n as usize <= self.total && off % n as usize == 0);
        unsafe {
            match n {
                1 => *self.host.add(off ^ 3) = v as u8,
                2 => (self.host.add(off ^ 2) as *mut u16).write_unaligned(v as u16),
                4 => (self.host.add(off) as *mut u32).write_unaligned(v as u32),
                _ => {
                    (self.host.add(off) as *mut u32).write_unaligned((v >> 32) as u32);
                    (self.host.add(off + 4) as *mut u32).write_unaligned(v as u32);
                }
            }
        }
        self.touch(off / PAGE);
    }

    /// Page `page` was written: bump its generation.
    #[inline]
    fn touch(&mut self, page: usize) {
        #[cfg(feature = "jitv2")]
        self.store.bump_gen_off(page * PAGE);
        #[cfg(not(feature = "jitv2"))]
        let _ = page;
    }

    /// The generation counter for the page holding byte offset `off`.
    #[cfg(feature = "jitv2")]
    pub fn gen_ptr(&self, off: usize) -> *const std::sync::atomic::AtomicU64 {
        self.store.gen_ptr(off as u64)
    }

    /// Bytes fitted.
    pub fn len(&self) -> usize {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// The CPU's window: guest physical `p` at `window_base() + p` wherever
    /// the mapped bitmap says RAM is.
    pub fn space(&self) -> &PpMemSpace {
        &self.space
    }

    /// The MEMCFG words last placed.
    pub fn memcfg(&self) -> [u32; 4] {
        self.memcfg
    }

    /// The banks as placed: (physical base, span, real size) for each;
    /// a real size of 0 is a valid bank with no DIMMs.
    pub fn layout(&self) -> Vec<(u64, u64, u64)> {
        self.banks.iter().flatten().map(|b| (b.base, b.span, if b.empty { 0 } else { b.mask + 1 })).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::heart::encode_memcfg;

    #[test]
    fn banks_follow_memcfg_and_the_alias_is_the_first_16k() {
        let cfg = [encode_memcfg(0, 256), encode_memcfg(256 << 20, 256), 0, 0];
        let mut m = HeartMemory::new([256, 256, 0, 0], cfg);
        assert_eq!(m.decode(RAM_BASE), Decode::Ram(0));
        assert_eq!(m.decode(RAM_BASE + (256 << 20)), Decode::Ram(256 << 20));
        assert_eq!(m.decode(0x1000), Decode::Ram(0x1000));
        assert_eq!(m.decode(RAM_BASE + (512 << 20)), Decode::None);
        m.put_be(0x1000, 4, 0x1234_5678);
        assert_eq!(m.get_be(0x1000, 4), 0x1234_5678);
        assert_eq!(m.get_be(0x1000, 1), 0x12);
        assert_eq!(m.get_be(0x1002, 2), 0x5678);
        // The PROM's sizing probe: 1 GB at 2 GB on a 256 MB bank repeats
        // every 256 MB.
        m.set_memcfg([0x819f_0040, 0, 0, 0]);
        assert_eq!(m.decode(RAM_BASE + (2 << 30)), Decode::Ram(0));
        assert_eq!(m.decode(RAM_BASE + (2 << 30) + (256 << 20) + 8), Decode::Ram(8));
        assert_eq!(m.decode(RAM_BASE), Decode::None);
        // The PROM probing bank 2, which has no DIMMs: a valid bank that
        // takes writes and reads back nothing.
        m.set_memcfg([0x019f_01ff, 0, 0x81bf_0040, 0]);
        assert_eq!(m.decode(RAM_BASE + (2 << 30)), Decode::Empty);
        assert_eq!(m.layout(), vec![(RAM_BASE + (2 << 30), 2 << 30, 0)]);
    }
}
