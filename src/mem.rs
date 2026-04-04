use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BusDevice, BUS_OK, BUS_ERR, Resettable};

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
        }
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
}

impl Resettable for Memory {
    fn power_on(&self) {
        let data = unsafe { self.data() };
        data.fill(0);
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
            BUS_OK
        }
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
            BUS_OK
        }
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
            BUS_OK
        }
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
            BUS_OK
        }
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
