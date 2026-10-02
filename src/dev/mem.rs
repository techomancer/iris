use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "jitv2")]
use std::sync::atomic::AtomicU64;
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BusDevice, BUS_OK, BUS_ERR, Resettable};

#[cfg(feature = "jitv2")]
const JITV2_PAGE_SIZE: usize = crate::jitv2::jitv2::PAGE_SIZE as usize;

/// Memory Module with direct word access
///
/// Stores memory as u32 words in native (host) byte order for maximum performance.
/// Big Endian conversion happens only on sub-word (8/16-bit) accesses.
///
/// Optimizations:
/// - No alignment checks (caller guarantees alignment)
/// - No bounds checks (address decoder guarantees valid range)
/// - Word-based storage for direct 32-bit access
/// - Minimal endian conversion overhead
///
/// # Safety
/// Uses UnsafeCell for zero-overhead memory access. Safe because:
/// - Memory access is serialized through the bus interface
/// - Alignment and bounds guaranteed by caller/decoder
pub struct Memory {
    data: UnsafeCell<Vec<u32>>,
    size_words: usize,
    /// Byte mask applied to addr for indexing. Initialized to mem_size_bytes-1.
    /// Always kept masked to mem_size_bytes-1 so it can never exceed the buffer.
    addr_mask: u32,
    /// JIT v2: one generation counter per 4KB physical page (rules/jitv2/jit-v2-design.md
    /// §2.4, §7). Bumped atomically on every write to that page. A compiled artifact for
    /// a page is valid iff its recorded generation still matches this counter (§4.1, §6.5).
    #[cfg(feature = "jitv2")]
    gen: Vec<AtomicU64>,
}

// Safety: Memory is safe to share across threads because access is controlled
// through the BusDevice interface which serializes access through the bus.
unsafe impl Sync for Memory {}

impl Memory {
    /// Create a new memory block.
    ///
    /// # Arguments
    /// * `size_mb` - Size in Megabytes
    pub fn new(size_mb: usize) -> Self {
        let size_bytes = size_mb * 1024 * 1024;
        let size_words = size_bytes / 4;

        Self {
            data: UnsafeCell::new(vec![0u32; size_words]),
            size_words,
            addr_mask: (size_bytes - 1) as u32,
            #[cfg(feature = "jitv2")]
            gen: (0..size_bytes.div_ceil(JITV2_PAGE_SIZE)).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// JIT v2: bump the generation counter for the page containing `addr` (post-mask).
    /// Must be called on every mutating access so a compiled artifact can detect
    /// staleness (rules/jitv2/jit-v2-design.md §7 — "gen bumps ride on the RAM write").
    #[cfg(feature = "jitv2")]
    #[inline(always)]
    fn bump_gen(&self, addr: u32) {
        let page = ((addr & self.addr_mask) as usize) / JITV2_PAGE_SIZE;
        // Relaxed: the publish-side gen re-check (§6.5) re-reads this counter after its
        // own release fetch_or on entry_bits, so ordering is provided there, not here.
        self.gen[page].fetch_add(1, Ordering::Relaxed);
    }

    /// JIT v2: raw pointer to the generation counter for the page containing `addr`.
    /// Used to populate `PhysicalCodePage::gen` (see src/jitv2/jitv2.rs). The pointer
    /// is valid for the lifetime of `&self` (the `gen` vec is never resized after `new`).
    #[cfg(feature = "jitv2")]
    #[inline]
    pub fn gen_ptr(&self, addr: u32) -> *const AtomicU64 {
        let page = ((addr & self.addr_mask) as usize) / JITV2_PAGE_SIZE;
        &self.gen[page] as *const AtomicU64
    }

    /// Set a new address mask. The provided mask is AND-ed with `mem_size_bytes-1`
    /// so it can never address outside the buffer.
    pub fn set_addr_mask(&mut self, mask: u32) {
        let size_mask = (self.size_words * 4 - 1) as u32;
        self.addr_mask = mask & size_mask;
    }

    #[inline(always)]
    unsafe fn data(&self) -> &mut [u32] {
        &mut *self.data.get()
    }

    /// Save memory contents to a raw binary file.
    /// Words are stored as big-endian bytes matching the MIPS memory layout.
    pub fn save_bin(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let data = unsafe { self.data() };
        // Convert each u32 word to big-endian bytes
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &word in data.iter() {
            bytes.extend_from_slice(&word.to_be_bytes());
        }
        std::fs::write(path, &bytes)
    }

    /// Load memory contents from a raw binary file saved by save_bin().
    pub fn load_bin(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let bytes = std::fs::read(path)?;
        let data = unsafe { self.data() };
        let words = bytes.len() / 4;
        let count = words.min(data.len());
        for i in 0..count {
            let b = &bytes[i * 4..(i + 1) * 4];
            data[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        Ok(())
    }

    /// Clone the bank's word buffer in native endian. Used by the in-memory
    /// rollback checkpoint to capture state without touching disk. Caller
    /// should ensure the CPU/peripheral threads are stopped to avoid a
    /// torn read.
    pub fn snapshot_words(&self) -> Vec<u32> {
        let data = unsafe { self.data() };
        data.to_vec()
    }

    /// Overwrite the bank's word buffer from `src`. Length is clamped to the
    /// bank's word count; extra source words are dropped, missing tail words
    /// are left untouched. Pair with `snapshot_words` for rollback.
    pub fn restore_words(&self, src: &[u32]) {
        let data = unsafe { self.data() };
        let n = src.len().min(data.len());
        data[..n].copy_from_slice(&src[..n]);
        // JIT v2 (§7.1 channel 4, §7.6): snapshot restore/rollback mutates RAM
        // out from under any compiled artifact regardless of content equality.
        #[cfg(feature = "jitv2")]
        self.bump_gen_all();
    }

    /// JIT v2: bump every page's generation counter. Used by whole-buffer mutations
    /// (restore, power-on) where per-page cursoring isn't worth the bookkeeping.
    #[cfg(feature = "jitv2")]
    fn bump_gen_all(&self) {
        for g in &self.gen {
            g.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Resettable for Memory {
    fn power_on(&self) {
        let data = unsafe { self.data() };
        data.fill(0);
        #[cfg(feature = "jitv2")]
        self.bump_gen_all();
    }
}

impl BusDevice for Memory {
    #[inline(always)]
    fn read8(&self, addr: u32) -> BusRead8 {
        unsafe {
            let data = self.data();
            let byte_ptr = data.as_ptr() as *const u8;
            let offset = (addr & self.addr_mask) as usize;
            let byte = *byte_ptr.add(offset ^ 3);
            BusRead8::ok(byte)
        }
    }

    #[inline(always)]
    fn write8(&self, addr: u32, val: u8) -> u32 {
        unsafe {
            let data = self.data();
            let byte_ptr = data.as_mut_ptr() as *mut u8;
            let offset = (addr & self.addr_mask) as usize;
            *byte_ptr.add(offset ^ 3) = val;
        }
        #[cfg(feature = "jitv2")]
        self.bump_gen(addr);
        BUS_OK
    }

    #[inline(always)]
    fn read16(&self, addr: u32) -> BusRead16 {
        unsafe {
            let data = self.data();
            let half_ptr = data.as_ptr() as *const u16;
            let offset = ((addr & self.addr_mask) >> 1) as usize;
            let halfword = *half_ptr.add(offset ^ 1);
            BusRead16::ok(halfword)
        }
    }

    #[inline(always)]
    fn write16(&self, addr: u32, val: u16) -> u32 {
        unsafe {
            let data = self.data();
            let half_ptr = data.as_mut_ptr() as *mut u16;
            let offset = ((addr & self.addr_mask) >> 1) as usize;
            *half_ptr.add(offset ^ 1) = val;
        }
        #[cfg(feature = "jitv2")]
        self.bump_gen(addr);
        BUS_OK
    }

    #[inline(always)]
    fn read32(&self, addr: u32) -> BusRead32 {
        unsafe {
            let data = self.data();
            let idx = ((addr & self.addr_mask) >> 2) as usize;
            BusRead32::ok(*data.get_unchecked(idx))
        }
    }

    #[inline(always)]
    fn write32(&self, addr: u32, val: u32) -> u32 {
        unsafe {
            let data = self.data();
            let idx = ((addr & self.addr_mask) >> 2) as usize;
            *data.get_unchecked_mut(idx) = val;
        }
        #[cfg(feature = "jitv2")]
        self.bump_gen(addr);
        BUS_OK
    }

    #[inline(always)]
    fn read64(&self, addr: u32) -> BusRead64 {
        unsafe {
            let data = self.data();
            let qword_ptr = data.as_ptr() as *const u64;
            let offset = ((addr & self.addr_mask) >> 3) as usize;
            let qword = (*qword_ptr.add(offset)).rotate_left(32);
            BusRead64::ok(qword)
        }
    }

    #[inline(always)]
    fn write64(&self, addr: u32, val: u64) -> u32 {
        unsafe {
            let data = self.data();
            let qword_ptr = data.as_mut_ptr() as *mut u64;
            let offset = ((addr & self.addr_mask) >> 3) as usize;
            *qword_ptr.add(offset) = val.rotate_left(32);
        }
        #[cfg(feature = "jitv2")]
        self.bump_gen(addr);
        BUS_OK
    }

    #[inline]
    fn mem_ptr(&self, addr: u32) -> Option<*const u64> {
        unsafe {
            let data = self.data();
            let qword_ptr = data.as_ptr() as *const u64;
            Some(qword_ptr.add(((addr & self.addr_mask) >> 3) as usize))
        }
    }

    #[inline]
    fn read_block(&self, addr: u32, buf: &mut [u64]) -> u32 {
        unsafe {
            let data = self.data();
            let qword_ptr = data.as_ptr() as *const u64;
            let offset = ((addr & self.addr_mask) >> 3) as usize;
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (*qword_ptr.add(offset + i)).rotate_left(32);
            }
        }
        BUS_OK
    }

    #[inline]
    fn write_block(&self, addr: u32, buf: &[u64]) -> u32 {
        unsafe {
            let data = self.data();
            let qword_ptr = data.as_mut_ptr() as *mut u64;
            let offset = ((addr & self.addr_mask) >> 3) as usize;
            for (i, &val) in buf.iter().enumerate() {
                *qword_ptr.add(offset + i) = val.rotate_left(32);
            }
        }
        // JIT v2: per-page write cursor (§7.2) — bump gen once per page touched by
        // this DMA-style block write, not once per qword.
        #[cfg(feature = "jitv2")]
        {
            let start_page = ((addr & self.addr_mask) as usize) / JITV2_PAGE_SIZE;
            let last_byte = addr.wrapping_add(((buf.len().max(1) - 1) as u32) * 8);
            let end_page = ((last_byte & self.addr_mask) as usize) / JITV2_PAGE_SIZE;
            if end_page >= start_page {
                for page in start_page..=end_page {
                    self.gen[page].fetch_add(1, Ordering::Relaxed);
                }
            } else {
                // Wrapped around addr_mask: bump the two remaining spans.
                for page in start_page..self.gen.len() {
                    self.gen[page].fetch_add(1, Ordering::Relaxed);
                }
                for page in 0..=end_page {
                    self.gen[page].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        BUS_OK
    }

    #[inline(always)]
    fn write64_masked(&self, addr: u32, val: u64, mask: u64) -> u32 {
        unsafe {
            let data = self.data();
            let qword_ptr = data.as_mut_ptr() as *mut u64;
            let offset = ((addr & self.addr_mask) >> 3) as usize;
            let stored = qword_ptr.add(offset);
            // Memory stores qwords as rotate_left(32); apply same rotation to val/mask
            let old = *stored;
            let new_val = val.rotate_left(32);
            let new_mask = mask.rotate_left(32);
            *stored = (old & !new_mask) | (new_val & new_mask);
        }
        #[cfg(feature = "jitv2")]
        self.bump_gen(addr);
        BUS_OK
    }

    #[cfg(feature = "jitv2")]
    #[inline]
    fn gen_ptr(&self, addr: u32) -> *const AtomicU64 {
        Memory::gen_ptr(self, addr)
    }
}

// Implement BusDevice for Arc<Memory> to allow using Arc-wrapped memory directly
impl BusDevice for Arc<Memory> {
    fn read8(&self, addr: u32) -> BusRead8 {
        (**self).read8(addr)
    }

    fn write8(&self, addr: u32, val: u8) -> u32 {
        (**self).write8(addr, val)
    }

    fn read16(&self, addr: u32) -> BusRead16 {
        (**self).read16(addr)
    }

    fn write16(&self, addr: u32, val: u16) -> u32 {
        (**self).write16(addr, val)
    }

    fn read32(&self, addr: u32) -> BusRead32 {
        (**self).read32(addr)
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        (**self).write32(addr, val)
    }

    fn read64(&self, addr: u32) -> BusRead64 {
        (**self).read64(addr)
    }

    fn write64(&self, addr: u32, val: u64) -> u32 {
        (**self).write64(addr, val)
    }

    fn write64_masked(&self, addr: u32, val: u64, mask: u64) -> u32 {
        (**self).write64_masked(addr, val, mask)
    }

    fn read_block(&self, addr: u32, buf: &mut [u64]) -> u32 {
        (**self).read_block(addr, buf)
    }

    fn write_block(&self, addr: u32, buf: &[u64]) -> u32 {
        (**self).write_block(addr, buf)
    }

    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, addr: u32) -> *const AtomicU64 {
        (**self).gen_ptr(addr)
    }
}

/// Black Hole Region
///
/// Swallows writes and returns all 1s on reads.
/// That is typical for regular machines like PC.
pub struct BlackHoleRegion {
    debug: AtomicBool,
}

impl BlackHoleRegion {
    pub fn new() -> Self {
        Self {
            debug: AtomicBool::new(false),
        }
    }

    pub fn set_debug(&self, val: bool) {
        self.debug.store(val, Ordering::Relaxed);
    }
}

impl BusDevice for BlackHoleRegion {
    fn read8(&self, addr: u32) -> BusRead8 {
        if self.debug.load(Ordering::Relaxed) {
            println!("BlackHole: Read8 {:08x}", addr);
        }
        BusRead8::ok(0xFF)
    }
    fn write8(&self, addr: u32, val: u8) -> u32 {
        if self.debug.load(Ordering::Relaxed) {
            println!("BlackHole: Write8 {:08x} val {:02x}", addr, val);
        }
        BUS_OK
    }

    fn read16(&self, addr: u32) -> BusRead16 {
        if self.debug.load(Ordering::Relaxed) {
            println!("BlackHole: Read16 {:08x}", addr);
        }
        BusRead16::ok(0xFFFF)
    }
    fn write16(&self, addr: u32, val: u16) -> u32 {
        if self.debug.load(Ordering::Relaxed) {
            println!("BlackHole: Write16 {:08x} val {:04x}", addr, val);
        }
        BUS_OK
    }

    fn read32(&self, addr: u32) -> BusRead32 {
        if self.debug.load(Ordering::Relaxed) {
            println!("BlackHole: Read32 {:08x}", addr);
        }
        BusRead32::ok(0xFFFFFFFF)
    }
    fn write32(&self, addr: u32, val: u32) -> u32 {
        if self.debug.load(Ordering::Relaxed) {
            println!("BlackHole: Write32 {:08x} val {:08x}", addr, val);
        }
        BUS_OK
    }

    fn read64(&self, addr: u32) -> BusRead64 {
        if self.debug.load(Ordering::Relaxed) {
            println!("BlackHole: Read64 {:08x}", addr);
        }
        BusRead64::ok(0xFFFFFFFFFFFFFFFF)
    }
    fn write64(&self, addr: u32, val: u64) -> u32 {
        if self.debug.load(Ordering::Relaxed) {
            println!("BlackHole: Write64 {:08x} val {:016x}", addr, val);
        }
        BUS_OK
    }
}

/// Unmapped RAM region: silently returns 0 for reads, ignores writes.
/// Used for lomem/himem slots when no bank is mapped there.
pub struct UnmappedRam;

impl BusDevice for UnmappedRam {
    fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::ok(0) }
    fn write8(&self, _addr: u32, _v: u8) -> u32 { BUS_OK }
    fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::ok(0) }
    fn write16(&self, _addr: u32, _v: u16) -> u32 { BUS_OK }
    fn read32(&self, _addr: u32) -> BusRead32 { BusRead32::ok(0) }
    fn write32(&self, _addr: u32, _v: u32) -> u32 { BUS_OK }
    fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::ok(0) }
    fn write64(&self, _addr: u32, _v: u64) -> u32 { BUS_OK }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::BusDevice;

    #[test]
    fn snapshot_and_restore_words_roundtrip() {
        let m = Memory::new(1); // 1 MB = 256 K words
        // Seed unique data via the bus interface so storage layout matches
        // production access patterns.
        for i in 0..32 {
            let addr = (i * 4) as u32;
            m.write32(addr, 0xDEAD0000 + i as u32);
        }
        let snap = m.snapshot_words();
        // Mutate in place.
        for i in 0..32 {
            let addr = (i * 4) as u32;
            m.write32(addr, 0xCAFEBABE);
        }
        // Restore from snapshot, verify original data returns.
        m.restore_words(&snap);
        for i in 0..32 {
            let addr = (i * 4) as u32;
            let v = m.read32(addr).data;
            assert_eq!(v, 0xDEAD0000 + i as u32, "word {} mismatch after restore", i);
        }
    }

    #[test]
    fn restore_words_clamps_to_bank_size() {
        let m = Memory::new(1);
        let words = m.snapshot_words();
        let bank_words = words.len();
        // Source buffer larger than bank — should not panic, should not write
        // past end.
        let mut larger = vec![0xAAAAAAAAu32; bank_words + 100];
        for (i, w) in larger.iter_mut().enumerate().take(bank_words + 100) {
            *w = (i as u32).wrapping_mul(7);
        }
        m.restore_words(&larger);
        // Spot-check a word inside the bank.
        let v = m.read32(0).data;
        assert_eq!(v, 0);
    }

    #[cfg(feature = "jitv2")]
    #[test]
    fn write_bumps_page_generation_only_for_touched_page() {
        let m = Memory::new(1); // 1 MB = 256 pages of 4KB
        let gen0_before = unsafe { (*m.gen_ptr(0)).load(Ordering::Relaxed) };
        let gen1_before = unsafe { (*m.gen_ptr(JITV2_PAGE_SIZE as u32)).load(Ordering::Relaxed) };

        m.write32(0, 0x1234); // page 0 only

        let gen0_after = unsafe { (*m.gen_ptr(0)).load(Ordering::Relaxed) };
        let gen1_after = unsafe { (*m.gen_ptr(JITV2_PAGE_SIZE as u32)).load(Ordering::Relaxed) };
        assert_eq!(gen0_after, gen0_before + 1, "written page must bump gen");
        assert_eq!(gen1_after, gen1_before, "untouched page must not bump gen");
    }

    #[cfg(feature = "jitv2")]
    #[test]
    fn read_does_not_bump_generation() {
        let m = Memory::new(1);
        let before = unsafe { (*m.gen_ptr(0)).load(Ordering::Relaxed) };
        let _ = m.read32(0);
        let after = unsafe { (*m.gen_ptr(0)).load(Ordering::Relaxed) };
        assert_eq!(before, after);
    }

    #[cfg(feature = "jitv2")]
    #[test]
    fn write_block_bumps_every_page_it_spans() {
        let m = Memory::new(1);
        // Straddle the page-0/page-1 boundary: last qword lands in page 1.
        let start = (JITV2_PAGE_SIZE - 8) as u32;
        let gen0_before = unsafe { (*m.gen_ptr(0)).load(Ordering::Relaxed) };
        let gen1_before = unsafe { (*m.gen_ptr(JITV2_PAGE_SIZE as u32)).load(Ordering::Relaxed) };

        m.write_block(start, &[0u64, 0u64]);

        let gen0_after = unsafe { (*m.gen_ptr(0)).load(Ordering::Relaxed) };
        let gen1_after = unsafe { (*m.gen_ptr(JITV2_PAGE_SIZE as u32)).load(Ordering::Relaxed) };
        assert_eq!(gen0_after, gen0_before + 1);
        assert_eq!(gen1_after, gen1_before + 1);
    }
}
