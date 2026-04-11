// MIPS R4000 Cache Implementation - Version 2
//
// This is a complete rewrite to properly support R4000 cache semantics:
// - Unified cache object containing L1-I, L1-D, and L2
// - Proper VIPT (Virtually Indexed, Physically Tagged) support
// - R4000-compliant tag format with PState bits
// - L2 can signal back to L1 for evictions

use crate::traits::{BusRead64, BusDevice, Resettable, BUS_OK, BUS_BUSY, BUS_ERR, BUS_VCE};
use crate::snapshot::{u32_slice_to_toml, u64_slice_to_toml, load_u32_slice, load_u64_slice, get_field, toml_bool, toml_u32, hex_u32};
use crate::mips_exec::{DecodedInstr, ExecStatus, EXEC_COMPLETE, EXEC_RETRY, exec_exception_const, EXC_VCEI, EXC_VCED, EXC_IBE};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::cell::UnsafeCell;
use bitfield::bitfield;

/// Result of a cache instruction fetch, shared by `MipsCache::fetch` and `fetch_instr_impl`.
/// `status == EXEC_COMPLETE` means hit; `instr` points to the DecodedInstr slot (valid for
/// the lifetime of the cache). Any other status is an exception/retry; `instr` is null.
pub struct FetchInstrResult {
    pub status: ExecStatus,
    pub instr: *const DecodedInstr,
}

unsafe impl Send for FetchInstrResult {}

impl FetchInstrResult {
    #[inline(always)]
    pub fn hit(instr: *const DecodedInstr) -> Self {
        Self { status: EXEC_COMPLETE, instr }
    }
    #[inline(always)]
    pub fn exception(status: ExecStatus) -> Self {
        Self { status, instr: std::ptr::null() }
    }
}


// =============================================================================
// R4400 Architecture Cache Constants
// =============================================================================

/// Compile-time count-trailing-zeros for usize (stable Rust lacks const trailing_zeros).
const fn ctz(n: usize) -> u32 {
    let mut i = 0u32;
    let mut v = n;
    while v & 1 == 0 { v >>= 1; i += 1; }
    i
}

/// Cache kind discriminant — used as a const generic `u8` parameter on `Cache`.
#[repr(u8)]
enum CacheKind { Insn = 0, Data = 1, L2 = 2 }

// R4400 cache sizes and line sizes — fixed for this CPU.
// Exported so that mips_exec can build CP0 Config without duplicating these values.
pub const IC_SIZE: usize = 16 * 1024;   // 16 KB L1 instruction cache
pub const IC_LINE: usize = 16;          // 16-byte lines
pub const DC_SIZE: usize = 16 * 1024;   // 16 KB L1 data cache
pub const DC_LINE: usize = 16;          // 16-byte lines
pub const L2_SIZE: usize = 1024 * 1024; // 1 MB unified L2
pub const L2_LINE: usize = 128;         // 128-byte lines

// Re-export cache operation constants for convenience
pub use crate::mips_isa::{
    CACH_PI, CACH_PD, CACH_SI, CACH_SD,
    C_IINV, C_IWBINV, C_ILT, C_IST, C_CDX,
    C_HINV, C_HWBINV, C_FILL, C_HWB, C_HSV,
};

// =============================================================================
// R4000 Cache Tag Format (per MIPS R4000 book)
// =============================================================================

// L1 Instruction Cache Tag
//   [25]    P  - Parity bit (ignored)
//   [24]    V  - Valid bit
//   [23:0]  PTag - Physical address bits [35:12]
bitfield! {
    #[derive(Clone, Copy, PartialEq, Eq, Default)]
    pub struct L1ITag(u32);
    impl Debug;
    pub u32, ptag, set_ptag: 23, 0;   // Physical tag bits [35:12]
    pub valid, set_valid: 24;           // Valid bit
}

// L1 Data Cache Tag
//   [28]    WP - Even parity for write-back bit (ignored)
//   [27]    W  - Write-back (dirty) bit
//   [26]    P  - Even parity for PTag+CS (ignored)
//   [25:24] CS - Cache State (0=Invalid, 1=Shared, 2=CleanExclusive, 3=DirtyExclusive)
//   [23:0]  PTag - Physical address bits [35:12]
bitfield! {
    #[derive(Clone, Copy, PartialEq, Eq, Default)]
    pub struct L1DTag(u32);
    impl Debug;
    pub u32, ptag, set_ptag: 23, 0;   // Physical tag bits [35:12]
    pub u32, cs, set_cs: 25, 24;       // Cache State: 0=Invalid, 1=Shared, 2=CleanExcl, 3=DirtyExcl
    pub dirty, set_dirty: 27;          // Write-back (dirty) bit
}

// L2 Cache Tag
//   [31:25] ECC  - Error correction code (ignored)
//   [24:22] CS   - Cache State (0=Invalid, 4=CleanExcl, 5=DirtyExcl, 6=Shared, 7=DirtyShared)
//   [21:19] PIdx - Virtual address bits [14:12] (primary cache aliasing)
//   [18:0]  PTag - Physical address bits [35:17]
bitfield! {
    #[derive(Clone, Copy, PartialEq, Eq, Default)]
    pub struct L2Tag(u32);
    impl Debug;
    pub u32, ptag, set_ptag: 18, 0;   // Physical tag bits [35:17]
    pub u32, pidx, set_pidx: 21, 19;  // Virtual index bits [14:12] for VIPT aliasing
    pub u32, cs, set_cs: 24, 22;      // Cache State (3-bit)
}

// Address reconstruction constants
/// PTag for L1 covers phys addr bits [35:12]; index supplies bits [11:0]
pub const L1_PTAG_SHIFT: u32 = 12;
pub const L1_PTAG_MASK: u32 = 0x00FF_FFFF; // 24-bit field
pub const L1_INDEX_MASK: u64 = 0xFFF;

/// PTag for L2 covers phys addr bits [35:17]; index supplies bits [16:0]
pub const L2_PTAG_SHIFT: u32 = 17;
pub const L2_PTAG_MASK: u32 = 0x0007_FFFF; // 19-bit field
pub const L2_INDEX_MASK: u64 = 0x1FFFF;

/// PIdx comes from virtual address bits [14:12]
pub const L2_PIDX_VADDR_SHIFT: u32 = 12;
pub const L2_PIDX_VADDR_MASK: u32 = 0x7; // 3-bit field

// L1 D-Cache CS (Cache State) values
pub const L1D_CS_INVALID: u32 = 0;
pub const L1D_CS_SHARED: u32 = 1;
pub const L1D_CS_CLEAN_EXCLUSIVE: u32 = 2;
pub const L1D_CS_DIRTY_EXCLUSIVE: u32 = 3;

// L2 CS (Cache State) values
pub const L2_CS_INVALID: u32 = 0;
pub const L2_CS_CLEAN_EXCLUSIVE: u32 = 4;
pub const L2_CS_DIRTY_EXCLUSIVE: u32 = 5;
pub const L2_CS_SHARED: u32 = 6;
pub const L2_CS_DIRTY_SHARED: u32 = 7;

/// Reconstruct the physical base address from an L1 cache tag and the address used to index
/// the cache line.  `index_addr` contributes the low 12 bits.
#[inline]
pub fn l1_tag_to_phys(tag: L1ITag, index_addr: u64) -> u64 {
    (tag.ptag() as u64) << L1_PTAG_SHIFT | (index_addr & L1_INDEX_MASK)
}

/// Reconstruct the physical base address from an L1D cache tag and the address used to index
/// the cache line.  `index_addr` contributes the low 12 bits.
#[inline]
pub fn l1d_tag_to_phys(tag: L1DTag, index_addr: u64) -> u64 {
    (tag.ptag() as u64) << L1_PTAG_SHIFT | (index_addr & L1_INDEX_MASK)
}

/// Reconstruct the physical base address from an L2 cache tag and the address used to index
/// the cache line.  `index_addr` contributes the low 17 bits.
#[inline]
pub fn l2_tag_to_phys(tag: L2Tag, index_addr: u64) -> u64 {
    (tag.ptag() as u64) << L2_PTAG_SHIFT | (index_addr & L2_INDEX_MASK)
}

impl From<u32> for L1ITag { fn from(v: u32) -> Self { L1ITag(v) } }
impl From<L1ITag> for u32  { fn from(t: L1ITag) -> Self { t.0 } }

impl From<u32> for L1DTag { fn from(v: u32) -> Self { L1DTag(v) } }
impl From<L1DTag> for u32  { fn from(t: L1DTag) -> Self { t.0 } }

impl From<u32> for L2Tag  { fn from(v: u32) -> Self { L2Tag(v) } }
impl From<L2Tag> for u32  { fn from(t: L2Tag) -> Self { t.0 } }

// =============================================================================
// Cache Operations Interface (for CACHE instruction)
// =============================================================================

// Cache operation is encoded in bits [20:16] of CACHE instruction (rt field)
// Format: [20:18] = operation, [17:16] = cache target
// We decode this u32 internally to determine what to do

// =============================================================================
// Main Cache Interface
// =============================================================================

/// Main cache interface - supports both memory access and cache operations
///
/// This trait combines:
/// - Instruction fetch from L1-I cache
/// - Data read/write through L1-D cache (VIPT)
/// - Cache operations for CACHE instruction support
/// - Load-Linked / Store-Conditional support
pub trait MipsCache: Send + Sync {
    /// Fetch instruction from L1 instruction cache.
    /// Returns `FetchInstrResult::hit(ptr)` on success, `FetchInstrResult::exception(status)` on error.
    /// The caller must call `decode_into` on the slot before use.
    fn fetch(&self, virt_addr: u64, phys_addr: u64) -> FetchInstrResult;

    /// Read data using virtual and physical addresses.
    /// Uses virtual address for index, physical address for tag (VIPT).
    /// SIZE must be 1, 2, 4, or 8 bytes (const generic, zero runtime branch).
    /// Returns BusRead64 with data zero-extended to u64 on success.
    /// status may be BUS_OK, BUS_BUSY, BUS_ERR, or BUS_VCE (cache only).
    fn read<const SIZE: usize>(&self, virt_addr: u64, phys_addr: u64) -> BusRead64;

    /// Write 64-bit data using virtual and physical addresses with byte mask.
    /// Uses virtual address for index, physical address for tag (VIPT).
    /// Mask indicates which bytes to write (bit mask for u64).
    /// Returns BUS_OK, BUS_BUSY, BUS_ERR, or BUS_VCE (cache only).
    /// This is the core write operation — use helper methods for smaller writes.
    fn write64_masked(&self, virt_addr: u64, phys_addr: u64, val: u64, mask: u64) -> u32;

    /// Write 8-bit value
    fn write8(&self, virt_addr: u64, phys_addr: u64, val: u8) -> u32 {
        let aligned_addr = phys_addr & !7;
        let offset = (phys_addr & 7) as usize;
        let shift = (7 - offset) * 8;
        let mask = 0xFF_u64 << shift;
        let val64 = (val as u64) << shift;
        self.write64_masked(virt_addr, aligned_addr, val64, mask)
    }

    /// Write 16-bit value
    fn write16(&self, virt_addr: u64, phys_addr: u64, val: u16) -> u32 {
        let aligned_addr = phys_addr & !7;
        let offset = (phys_addr & 7) as usize;
        let shift = (6 - offset) * 8;
        let mask = 0xFFFF_u64 << shift;
        let val64 = (val as u64) << shift;
        self.write64_masked(virt_addr, aligned_addr, val64, mask)
    }

    /// Write 32-bit value
    fn write32(&self, virt_addr: u64, phys_addr: u64, val: u32) -> u32 {
        let aligned_addr = phys_addr & !7;
        let offset = (phys_addr & 7) as usize;
        let shift = if offset == 0 { 32 } else { 0 };
        let mask = 0xFFFFFFFF_u64 << shift;
        let val64 = (val as u64) << shift;
        self.write64_masked(virt_addr, aligned_addr, val64, mask)
    }

    /// Write 64-bit value
    fn write64(&self, virt_addr: u64, phys_addr: u64, val: u64) -> u32 {
        self.write64_masked(virt_addr, phys_addr, val, !0)
    }

    /// Execute a cache operation (CACHE instruction)
    ///
    /// cache_op: Combined operation and cache target from bits [20:16] of CACHE instruction
    ///   - Bits [20:18]: Operation (C_IINV, C_ILT, C_IST, C_CDX, C_HINV, C_HWBINV/C_FILL, C_HWB, C_HSV)
    ///   - Bits [17:16]: Cache target (CACH_PI, CACH_PD, CACH_SI, CACH_SD)
    /// virt_addr: Virtual address from instruction (used for index operations)
    /// phys_addr: Physical address (used for hit operations and tags)
    ///
    /// For Index_Load_Tag operations, returns the tag value in TagLo CP0 register format
    /// For other operations, returns 0
    fn cache_op(&self, cache_op: u32, virt_addr: u64, phys_addr: u64) -> u32;

    /// Get cache configuration for a specific cache target
    /// cache_target: CACH_PI (0), CACH_PD (1), CACH_SI (2), or CACH_SD (3)
    /// Returns (size in bytes, line size in bytes)
    fn get_config(&self, cache_target: u32) -> (usize, usize);

    /// Get physical memory bus device for direct access
    fn downstream(&self) -> Arc<dyn BusDevice>;

    /// Check and clear Load-Linked bit if address matches
    fn check_and_clear_llbit(&self, phys_addr: u64);

    /// Get Load-Linked bit state
    fn get_llbit(&self) -> bool;

    /// Set Load-Linked bit state
    fn set_llbit(&self, val: bool);

    /// Get Load-Linked address
    fn get_lladdr(&self) -> u32;

    /// Set Load-Linked address
    fn set_lladdr(&self, addr: u32);

    /// Debug probe a virtual+physical address (optional, for debugging)
    fn debug_probe(&self, _cache_name: &str, _virt_addr: u64, _phys_addr: u64) -> String {
        "Debug not implemented for this cache type".to_string()
    }

    /// Debug dump a cache line by index (optional, for debugging)
    fn debug_dump_line(&self, _cache_name: &str, _idx: usize) -> String {
        "Debug not implemented for this cache type".to_string()
    }

    /// Restore power-on state — invalidate all cache lines (tags → 0).
    fn power_on(&self) {}

    /// Serialize full cache state (tags, data, LL/SC) to a TOML value.
    fn save_cache_state(&self) -> toml::Value {
        toml::Value::Table(Default::default())
    }

    /// Restore full cache state from a TOML value.
    fn load_cache_state(&self, _v: &toml::Value) -> Result<(), String> {
        Ok(())
    }
}

// =============================================================================
// Passthrough Cache - No caching, for testing
// =============================================================================

/// Passthrough cache that performs no caching - all accesses go directly to memory
/// Useful for testing and debugging
pub struct PassthroughCache {
    downstream: Arc<dyn BusDevice>,
    llbit: UnsafeCell<bool>,
    lladdr: UnsafeCell<u32>,
    /// Scratch slot for fetch() — no actual caching, just a place to decode into.
    fetch_scratch: UnsafeCell<DecodedInstr>,
}

// Safety: Single-threaded access only (CPU thread)
unsafe impl Send for PassthroughCache {}
unsafe impl Sync for PassthroughCache {}

impl PassthroughCache {
    pub fn new(downstream: Arc<dyn BusDevice>) -> Self {
        Self {
            downstream,
            llbit: UnsafeCell::new(false),
            lladdr: UnsafeCell::new(0),
            fetch_scratch: UnsafeCell::new(DecodedInstr::default()),
        }
    }
}

impl From<Arc<dyn BusDevice>> for PassthroughCache {
    fn from(downstream: Arc<dyn BusDevice>) -> Self {
        Self::new(downstream)
    }
}

impl MipsCache for PassthroughCache {
    fn fetch(&self, _virt_addr: u64, phys_addr: u64) -> FetchInstrResult {
        let r = self.downstream.read32(phys_addr as u32);
        if r.is_ok() {
            let slot = unsafe { &mut *self.fetch_scratch.get() };
            if slot.raw != r.data { slot.decoded = false; }
            slot.raw = r.data;
            FetchInstrResult::hit(slot as *const DecodedInstr)
        } else {
            // BUS_BUSY == EXEC_RETRY (compile-time asserted in traits.rs); pass status through.
            FetchInstrResult::exception(r.status)
        }
    }

    fn read<const SIZE: usize>(&self, _virt_addr: u64, phys_addr: u64) -> BusRead64 {
        const { assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE") };
        if SIZE == 1 { let r = self.downstream.read8(phys_addr as u32);  BusRead64 { status: r.status, data: r.data as u64 } }
        else if SIZE == 2 { let r = self.downstream.read16(phys_addr as u32); BusRead64 { status: r.status, data: r.data as u64 } }
        else if SIZE == 4 { let r = self.downstream.read32(phys_addr as u32); BusRead64 { status: r.status, data: r.data as u64 } }
        else               { self.downstream.read64(phys_addr as u32) }
    }

    fn write64_masked(&self, _virt_addr: u64, phys_addr: u64, val: u64, mask: u64) -> u32 {
        // For passthrough cache, just do a read-modify-write on the downstream device
        let aligned_addr = (phys_addr & !7) as u32;
        let r = self.downstream.read64(aligned_addr);
        if !r.is_ok() { return r.status; }
        let new_val = (r.data & !mask) | (val & mask);
        self.downstream.write64(aligned_addr, new_val)
    }

    fn cache_op(&self, _cache_op: u32, _virt_addr: u64, _phys_addr: u64) -> u32 {
        // No-op for passthrough cache - just return 0
        0
    }

    fn get_config(&self, _cache_target: u32) -> (usize, usize) {
        (0, 16) // Report minimal cache
    }

    fn downstream(&self) -> Arc<dyn BusDevice> {
        self.downstream.clone()
    }

    fn check_and_clear_llbit(&self, _phys_addr: u64) {
        // Simplified: just clear it
        unsafe { *self.llbit.get() = false; }
    }

    fn get_llbit(&self) -> bool {
        unsafe { *self.llbit.get() }
    }

    fn set_llbit(&self, val: bool) {
        unsafe { *self.llbit.get() = val; }
    }

    fn get_lladdr(&self) -> u32 {
        unsafe { *self.lladdr.get() }
    }

    fn set_lladdr(&self, addr: u32) {
        unsafe { *self.lladdr.get() = addr; }
    }
}

// =============================================================================
// Cache Structure - Used for L1-I, L1-D, and L2
// =============================================================================

/// Wrapper around UnsafeCell<Vec<T>> that is Send+Sync
struct CacheVec<T>(UnsafeCell<Vec<T>>);

unsafe impl<T> Send for CacheVec<T> {}
unsafe impl<T> Sync for CacheVec<T> {}

impl<T> CacheVec<T> {
    fn new(v: Vec<T>) -> Self { Self(UnsafeCell::new(v)) }

    #[inline(always)]
    fn get(&self) -> &Vec<T> { unsafe { &*self.0.get() } }

    #[inline(always)]
    fn get_mut(&self) -> &mut Vec<T> { unsafe { &mut *self.0.get() } }
}

/// A single cache level parameterised by size, line size, and kind (Insn/Data/L2).
///
/// All geometry constants are computed at compile time from `SIZE` and `LINE`.
/// `KIND` (a `CacheKind` discriminant cast to `u8`) controls whether the L2
/// decoded-instruction array is allocated and which methods are meaningful.
///
/// - `tags`: `TAGS` u32 tags inline in the struct (no heap indirection; TAGS = SIZE/LINE)
/// - `data`: `DATA` u64 chunks inline in the struct (no heap indirection; DATA = SIZE/8)
/// - `instrs`: L2 only — heap Vec of SIZE/4 DecodedInstr slots (6MB, contains fn ptrs)
///
/// `TAGS` and `DATA` are redundant with `SIZE`/`LINE` but required as explicit const generics
/// because stable Rust cannot use arithmetic on generic params in array length positions.
struct Cache<const SIZE: usize, const LINE: usize, const KIND: u8,
             const TAGS: usize, const DATA: usize> {
    /// Heap-allocated tag array — one u32 per cache line. Use `get_tag`/`set_tag` for typed access.
    tags:   UnsafeCell<Box<[u32; TAGS]>>,
    /// Heap-allocated data array — entire cache contents as u64 chunks.
    data:   UnsafeCell<Box<[u64; DATA]>>,
    /// L2 decoded-instruction slots (SIZE/4 entries). Empty Vec for L1-I and L1-D.
    instrs: CacheVec<DecodedInstr>,
    /// Signals the decode thread to stop (kept for Drop compatibility).
    stop:   Arc<AtomicBool>,
}

unsafe impl<const SIZE: usize, const LINE: usize, const KIND: u8,
            const TAGS: usize, const DATA: usize> Send for Cache<SIZE, LINE, KIND, TAGS, DATA> {}
unsafe impl<const SIZE: usize, const LINE: usize, const KIND: u8,
            const TAGS: usize, const DATA: usize> Sync for Cache<SIZE, LINE, KIND, TAGS, DATA> {}

impl<const SIZE: usize, const LINE: usize, const KIND: u8,
     const TAGS: usize, const DATA: usize> Cache<SIZE, LINE, KIND, TAGS, DATA> {
    // ---- Compile-time geometry constants ----
    const NUM_LINES:             usize = SIZE / LINE;
    const LINE_SHIFT:            u32   = ctz(LINE);
    const LINE_MASK:             usize = LINE - 1;
    const NUM_LINES_MASK:        usize = Self::NUM_LINES - 1;
    const CACHE_SIZE_SHIFT:      u32   = ctz(SIZE);
    const CHUNKS_PER_LINE:       usize = LINE / 8;
    const CHUNKS_PER_LINE_SHIFT: u32   = Self::LINE_SHIFT - 3;
    /// Instructions per cache line (LINE/4). Valid for Insn and L2 kinds.
    const INSTRS_PER_LINE:       usize = LINE / 4;
    /// Shift for instruction index within a line. Valid for Insn and L2 kinds.
    const INSTR_SHIFT:           u32   = Self::LINE_SHIFT - 2;
    const INSTR_MASK:            usize = Self::INSTRS_PER_LINE - 1;

    fn new() -> Self {
        // Allocate L2 decoded-instruction slots only for the L2 cache.
        let instrs = if KIND == CacheKind::L2 as u8 {
            (0..SIZE / 4).map(|_| DecodedInstr::default()).collect()
        } else {
            Vec::new()
        };
        Self {
            // SAFETY: u32/u64 are valid at all-zero bit patterns. Box::new_zeroed avoids
            // constructing the array on the stack before moving to the heap.
            tags:   UnsafeCell::new(unsafe { Box::new_zeroed().assume_init() }),
            data:   UnsafeCell::new(unsafe { Box::new_zeroed().assume_init() }),
            instrs: CacheVec::new(instrs),
            stop:   Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get cache line index from address.
    #[inline(always)]
    fn get_index(&self, addr: u64) -> usize {
        ((addr >> Self::LINE_SHIFT) as usize) & Self::NUM_LINES_MASK
    }

    /// Get byte offset within a cache line.
    #[inline(always)]
    fn get_line_offset(&self, addr: u64) -> usize {
        (addr as usize) & Self::LINE_MASK
    }

    /// Get the u64-chunk index for a given address.
    #[inline(always)]
    fn get_data_index(&self, addr: u64) -> usize {
        let line_idx = self.get_index(addr);
        let chunk_offset = self.get_line_offset(addr) >> 3;
        (line_idx << Self::CHUNKS_PER_LINE_SHIFT) + chunk_offset
    }

    #[inline(always)]
    fn tags(&self) -> &[u32; TAGS] { unsafe { &**self.tags.get() } }
    #[inline(always)]
    fn tags_mut(&self) -> &mut [u32; TAGS] { unsafe { &mut **self.tags.get() } }
    #[inline(always)]
    fn data(&self) -> &[u64; DATA] { unsafe { &**self.data.get() } }
    #[inline(always)]
    fn data_mut(&self) -> &mut [u64; DATA] { unsafe { &mut **self.data.get() } }

    /// Read the tag at `idx` as a typed bitfield struct.
    #[inline(always)]
    fn get_tag<T: From<u32>>(&self, idx: usize) -> T {
        T::from(unsafe { *self.tags().get_unchecked(idx) })
    }

    /// Write a typed bitfield tag to `idx`.
    #[inline(always)]
    fn set_tag<T: Into<u32>>(&self, idx: usize, tag: T) {
        unsafe { *self.tags_mut().get_unchecked_mut(idx) = tag.into(); }
    }

    /// View cache data as a flat &[u32] (two per u64, big-endian word order).
    /// XOR word index with 1 to address naturally on a little-endian host.
    /// Used by the I-cache to store l2.instrs slot indices.
    #[inline(always)]
    fn data_as_words(&self) -> &[u32] {
        let arr = self.data();
        unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const u32, SIZE / 4) }
    }

    /// View cache data as a flat &[u16] (big-endian halfword order within each u64).
    /// XOR halfword index with 3 to convert MIPS big-endian address to host offset.
    #[inline(always)]
    fn data_as_halves(&self) -> &[u16] {
        let arr = self.data();
        unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const u16, SIZE / 2) }
    }

    /// View cache data as a flat &[u8] (big-endian byte order within each u64).
    /// XOR byte index with 7 to convert MIPS big-endian address to host offset.
    #[inline(always)]
    fn data_as_bytes(&self) -> &[u8] {
        let arr = self.data();
        unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const u8, SIZE) }
    }
}

// =============================================================================
// R4000 Cache Implementation - Full 2-level hierarchy
// =============================================================================


// Debug configuration - set to Some(phys_addr) to enable cache line tracking
#[cfg(feature = "debug_cache")]
const DEBUG_TRACK_ADDR: Option<u64> = Some(0x17fa5ee4);
#[cfg(not(feature = "debug_cache"))]
const DEBUG_TRACK_ADDR: Option<u64> = None;

/// R4000 cache with proper 2-level hierarchy
///
/// This implementation keeps L1-I, L1-D, and L2 in a single object
/// so that L2 evictions can invalidate L1 lines as needed.
pub struct R4000Cache {
    downstream: Arc<dyn BusDevice>,

    // L1 Instruction Cache (16 KB, 16-byte lines)
    ic: ICache,

    // L1 Data Cache (16 KB, 16-byte lines)
    dc: DCache,

    // L2 Unified Cache (1 MB, 128-byte lines)
    l2: L2Cache,

    // Load-Linked / Store-Conditional support
    llbit: UnsafeCell<bool>,
    lladdr: UnsafeCell<u32>,

    /// L1-I hit counter — incremented on every fetch that finds a valid line (no fill needed).
    pub l1i_hit_count: Arc<AtomicU64>,
    /// L1-I fetch counter — incremented on every fetch attempt (hit or miss).
    pub l1i_fetch_count: Arc<AtomicU64>,

    // Debug tracking - cache line boundaries and indices for tracked address
    #[cfg(feature = "debug_cache")]
    debug_l1d_line: u64,
    #[cfg(feature = "debug_cache")]
    debug_l2_line: u64,
    #[cfg(feature = "debug_cache")]
    debug_companion_l1d_line: u64,
    #[cfg(feature = "debug_cache")]
    debug_companion_l2_line: u64,
    #[cfg(feature = "debug_cache")]
    debug_l1d_idx: usize,
    #[cfg(feature = "debug_cache")]
    debug_l2_idx: usize,
    #[cfg(feature = "debug_cache")]
    debug_companion_l2_idx: usize,
}

unsafe impl Send for R4000Cache {}
unsafe impl Sync for R4000Cache {}

// Type aliases for the concrete cache instances, for brevity in R4000Cache impls.
// TAGS = SIZE/LINE (one tag per cache line), DATA = SIZE/8 (one u64 per 8 bytes).
type ICache  = Cache<IC_SIZE, IC_LINE, { CacheKind::Insn as u8 }, { IC_SIZE / IC_LINE }, { IC_SIZE / 8 }>;
type DCache  = Cache<DC_SIZE, DC_LINE, { CacheKind::Data as u8 }, { DC_SIZE / DC_LINE }, { DC_SIZE / 8 }>;
type L2Cache = Cache<L2_SIZE, L2_LINE, { CacheKind::L2   as u8 }, { L2_SIZE / L2_LINE }, { L2_SIZE / 8 }>;

impl R4000Cache {
    pub fn new(downstream: Arc<dyn BusDevice>) -> Self {
        let ic = ICache::new();
        let dc = DCache::new();
        let l2 = L2Cache::new();

        #[cfg(feature = "debug_cache")]
        let (debug_l1d_line, debug_l2_line, debug_companion_l1d_line, debug_companion_l2_line,
             debug_l1d_idx, debug_l2_idx, debug_companion_l2_idx) = {
            if let Some(addr) = DEBUG_TRACK_ADDR {
                let l1_line_mask = DCache::LINE_MASK as u64;
                let l2_line_mask = L2Cache::LINE_MASK as u64;
                let companion_addr = addr ^ 0x00400000; // XOR with COMPANION_BIT

                let target_l1d_line = addr & !l1_line_mask;
                let target_l2_line = addr & !l2_line_mask;
                let companion_l1d_line = companion_addr & !l1_line_mask;
                let companion_l2_line = companion_addr & !l2_line_mask;

                let target_l1d_idx = dc.get_index(addr);
                let target_l2_idx = l2.get_index(addr);
                let companion_l2_idx = l2.get_index(companion_addr);

                println!("[CACHE DEBUG] Tracking setup:");
                println!("  Target addr: 0x{:08x}, L1D line: 0x{:08x}, L1D idx: {}, L2 line: 0x{:08x}, L2 idx: {}",
                         addr, target_l1d_line, target_l1d_idx, target_l2_line, target_l2_idx);
                println!("  Companion addr: 0x{:08x}, L1D line: 0x{:08x}, L2 line: 0x{:08x}, L2 idx: {}",
                         companion_addr, companion_l1d_line, companion_l2_line, companion_l2_idx);
                println!("  L2 index collision: {}", target_l2_idx == companion_l2_idx);

                (target_l1d_line, target_l2_line, companion_l1d_line, companion_l2_line,
                 target_l1d_idx, target_l2_idx, companion_l2_idx)
            } else {
                (0, 0, 0, 0, 0, 0, 0)
            }
        };

        Self {
            downstream,
            ic,
            dc,
            l2,
            llbit: UnsafeCell::new(false),
            lladdr: UnsafeCell::new(0),
            l1i_hit_count: Arc::new(AtomicU64::new(0)),
            l1i_fetch_count: Arc::new(AtomicU64::new(0)),
            #[cfg(feature = "debug_cache")]
            debug_l1d_line,
            #[cfg(feature = "debug_cache")]
            debug_l2_line,
            #[cfg(feature = "debug_cache")]
            debug_companion_l1d_line,
            #[cfg(feature = "debug_cache")]
            debug_companion_l2_line,
            #[cfg(feature = "debug_cache")]
            debug_l1d_idx,
            #[cfg(feature = "debug_cache")]
            debug_l2_idx,
            #[cfg(feature = "debug_cache")]
            debug_companion_l2_idx,
        }
    }
}


impl From<Arc<dyn BusDevice>> for R4000Cache {
    fn from(downstream: Arc<dyn BusDevice>) -> Self {
        Self::new(downstream)
    }
}

impl R4000Cache {
    /// Check if we're tracking this physical address (for debug purposes)
    #[cfg(feature = "debug_cache")]
    #[inline]
    fn is_tracking_l1d(&self, phys_addr: u64) -> bool {
        DEBUG_TRACK_ADDR.is_some() && {
            let line = phys_addr & !(DCache::LINE_MASK as u64);
            line == self.debug_l1d_line || line == self.debug_companion_l1d_line
        }
    }

    #[cfg(feature = "debug_cache")]
    #[inline]
    fn is_tracking_l2(&self, phys_addr: u64) -> bool {
        DEBUG_TRACK_ADDR.is_some() && {
            let line = phys_addr & !(L2Cache::LINE_MASK as u64);
            line == self.debug_l2_line || line == self.debug_companion_l2_line
        }
    }

    #[cfg(feature = "debug_cache")]
    #[inline]
    fn is_tracking_l1d_idx(&self, idx: usize) -> bool {
        DEBUG_TRACK_ADDR.is_some() && idx == self.debug_l1d_idx
    }

    #[cfg(feature = "debug_cache")]
    #[inline]
    fn is_tracking_l2_idx(&self, idx: usize) -> bool {
        DEBUG_TRACK_ADDR.is_some() && (idx == self.debug_l2_idx || idx == self.debug_companion_l2_idx)
    }

    #[cfg(feature = "debug_cache")]
    #[inline]
    fn is_tracking_addr(&self, virt_addr: u64, phys_addr: u64) -> bool {
        DEBUG_TRACK_ADDR.is_some() && {
            // Check if the physical line matches (most reliable)
            let line = phys_addr & !(DCache::LINE_MASK as u64);
            if line == self.debug_l1d_line || line == self.debug_companion_l1d_line {
                return true;
            }
            // Also check virtual address (for KSEG0 cached accesses)
            if let Some(target) = DEBUG_TRACK_ADDR {
                let companion = target ^ 0x00400000;
                // Check both 32-bit and 64-bit sign-extended forms
                virt_addr == (target | 0xffffffff80000000) ||
                virt_addr == (companion | 0xffffffff80000000)
            } else {
                false
            }
        }
    }

    #[cfg(feature = "debug_cache")]
    #[inline]
    fn tracking_label(&self, phys_addr: u64) -> &'static str {
        let line = phys_addr & !(DCache::LINE_MASK as u64);
        if line == self.debug_l1d_line {
            "TARGET"
        } else if line == self.debug_companion_l1d_line {
            "COMPANION"
        } else {
            "UNKNOWN"
        }
    }

    #[cfg(feature = "debug_cache")]
    #[inline]
    fn tracking_label_l2_idx(&self, idx: usize) -> &'static str {
        if idx == self.debug_l2_idx {
            "TARGET"
        } else if idx == self.debug_companion_l2_idx {
            "COMPANION"
        } else {
            "UNKNOWN"
        }
    }

    /// Extract physical tag bits [35:12] from physical address for L1 cache
    #[inline]
    fn l1_ptag(&self, phys_addr: u64) -> u32 {
        ((phys_addr >> L1_PTAG_SHIFT) & L1_PTAG_MASK as u64) as u32
    }

    /// Extract physical tag bits [35:17] from physical address for L2 cache
    #[inline]
    fn l2_ptag(&self, phys_addr: u64) -> u32 {
        ((phys_addr >> L2_PTAG_SHIFT) & L2_PTAG_MASK as u64) as u32
    }

    /// Extract virtual index bits [14:12] for L2 PIdx field
    #[inline]
    fn pidx(&self, virt_addr: u64) -> u32 {
        ((virt_addr >> L2_PIDX_VADDR_SHIFT) & L2_PIDX_VADDR_MASK as u64) as u32
    }

    /// Reconstruct the base L1 virtual index from an L2 cache index and stored PIdx.
    ///
    /// L1-D/I caches are VIPT: the index comes from the virtual address.  When an L2
    /// line is evicted we need to know which L1 lines it covers.  The L2 tag stores
    /// PIdx = virt[14:12] so we can reconstruct the virtual index bits that were used
    /// to fill the L1 line:
    ///
    ///   virt[14:12] = pidx            (from L2 tag)
    ///   virt[11:line_shift] = phys[11:line_shift]   (below page boundary, PA == VA)
    ///
    /// Returns the base L1 index corresponding to the first L1-sized sub-block of the
    /// L2 line.  The caller iterates over `l1_lines_per_l2` indices starting here,
    /// stepping by 1 (indices wrap naturally via the cache mask).
    #[inline]
    fn l2_idx_to_l1_base_idx<const L1_SIZE: usize, const L1_LINE: usize, const L1_KIND: u8, const L1_TAGS: usize, const L1_DATA: usize>(
        &self, l2_idx: usize, pidx: u32, _l1: &Cache<L1_SIZE, L1_LINE, L1_KIND, L1_TAGS, L1_DATA>
    ) -> usize {
        // Physical bits of the L2 line start address that are below bit 12 (page boundary)
        // These bits are the same in VA and PA, so we can derive them from the L2 index.
        let phys_sub_bits = (l2_idx << L2Cache::LINE_SHIFT as usize) & 0xFFF;
        // Reconstruct the virtual address bits used for L1 indexing
        let virt_index_bits = ((pidx as usize) << L2_PIDX_VADDR_SHIFT as usize) | phys_sub_bits;
        (virt_index_bits >> Cache::<L1_SIZE, L1_LINE, L1_KIND, L1_TAGS, L1_DATA>::LINE_SHIFT as usize)
            & Cache::<L1_SIZE, L1_LINE, L1_KIND, L1_TAGS, L1_DATA>::NUM_LINES_MASK
    }

    /// Check if the given physical address overlaps with the Load Linked address.
    /// If so, clear llbit (the link is broken).
    /// The lladdr stores bits [35:4] of the physical address.
    #[inline]
    fn check_and_clear_llbit(&self, phys_addr: u64, line_mask: usize) {
        unsafe {
            if !*self.llbit.get() {
                return;
            }
            let ll_addr = (*self.lladdr.get() as u64) << 4;
            let line_mask = line_mask as u64;
            let addr_line = phys_addr & !line_mask;
            let ll_line = ll_addr & !line_mask;
            if addr_line == ll_line {
            }
        }
    }

    /// Invalidate L1 instruction cache line by index
    fn invalidate_l1i_line(&self, idx: usize) {
        #[cfg(feature = "debug_cache")]
        {
            let tag: L1ITag = self.ic.get_tag(idx);
            if tag.valid() {
                let phys_addr = l1_tag_to_phys(tag, (idx << ICache::LINE_SHIFT) as u64);
                if self.is_tracking_l1d(phys_addr) {
                    println!("[CACHE DEBUG] invalidate_l1i_line: {} idx={}, phys_addr=0x{:08x}, ptag=0x{:06x}",
                             self.tracking_label(phys_addr), idx, phys_addr, tag.ptag());
                }
            }
        }

        self.ic.set_tag(idx, L1ITag::default());
    }

    /// Invalidate L1 data cache line by index.
    /// `coherent` = true for software-initiated CACHE ops (may clear llbit);
    /// false for hardware-induced evictions (fills, L2 cascades) which must not clear llbit.
    fn invalidate_l1d_line(&self, idx: usize, coherent: bool) {
        let tag: L1DTag = self.dc.get_tag(idx);

        #[cfg(feature = "debug_cache")]
        if self.is_tracking_l1d_idx(idx) {
            if tag.cs() != L1D_CS_INVALID {
                let phys_addr = l1d_tag_to_phys(tag, (idx << DCache::LINE_SHIFT) as u64);
                println!("[CACHE DEBUG] invalidate_l1d_line: {} idx={}, phys_addr=0x{:08x}, ptag=0x{:06x}, cs={}, coherent={}",
                         self.tracking_label(phys_addr), idx, phys_addr, tag.ptag(), tag.cs(), coherent);
            } else {
                println!("[CACHE DEBUG] invalidate_l1d_line: idx={} (already invalid)", idx);
            }
        }

        // Only clear llbit for software-initiated coherency invalidations, not hardware fills.
        // On a uniprocessor R4000 there are no external snoops; llbit survives capacity evictions.
        if coherent && tag.cs() != L1D_CS_INVALID {
            let phys_addr = l1d_tag_to_phys(tag, (idx << DCache::LINE_SHIFT) as u64);
            self.check_and_clear_llbit(phys_addr, DCache::LINE_MASK);
        }

        self.dc.set_tag(idx, L1DTag::default());
    }

    /// Invalidate L2 cache line by index
    /// This also invalidates any matching L1 lines (inclusive cache property)
    fn invalidate_l2_line(&self, idx: usize) {
        let l2_tag: L2Tag = self.l2.get_tag(idx);

        #[cfg(feature = "debug_cache")]
        if self.is_tracking_l2_idx(idx) {
            if l2_tag.cs() != L2_CS_INVALID {
                let phys_base = l2_tag_to_phys(l2_tag, (idx << L2Cache::LINE_SHIFT) as u64);
                println!("[CACHE DEBUG] invalidate_l2_line: {} idx={}, phys_base=0x{:08x}, ptag=0x{:05x}, cs={}",
                         self.tracking_label_l2_idx(idx), idx, phys_base, l2_tag.ptag(), l2_tag.cs());
            } else {
                println!("[CACHE DEBUG] invalidate_l2_line: {} idx={} (already invalid)",
                         self.tracking_label_l2_idx(idx), idx);
            }
        }

        // If L2 line is already invalid, nothing to do
        if l2_tag.cs() == L2_CS_INVALID {
            self.l2.set_tag(idx, L2Tag::default());
            return;
        }

        // Reconstruct physical address range covered by this L2 line
        let phys_base = l2_tag_to_phys(l2_tag, (idx << L2Cache::LINE_SHIFT) as u64);

        // NOTE: do NOT clear llbit here. On R4000, llbit tracks L1-D state only.
        // An L2 eviction is not a coherency action and must not break LL/SC.

        // Check L1-I for any lines from this L2 line.
        // L1-I is VIPT so we must reconstruct the virtual index from pidx + physical sub-bits.
        let pidx = l2_tag.pidx();
        let l1i_lines_per_l2 = 1 << (L2Cache::LINE_SHIFT - ICache::LINE_SHIFT);
        let ic_base_idx = self.l2_idx_to_l1_base_idx(idx, pidx, &self.ic);
        for i in 0..l1i_lines_per_l2 {
            let ic_idx = (ic_base_idx + i) & ICache::NUM_LINES_MASK;
            let phys_addr = phys_base + ((i as u64) << ICache::LINE_SHIFT);
            let ic_tag: L1ITag = self.ic.get_tag(ic_idx);
            if ic_tag.valid() && ic_tag.ptag() == self.l1_ptag(phys_addr) {
                self.invalidate_l1i_line(ic_idx);
            }
        }

        // Check L1-D for any lines from this L2 line.
        // L1-D is VIPT so we must reconstruct the virtual index from pidx + physical sub-bits.
        let l1d_lines_per_l2 = 1 << (L2Cache::LINE_SHIFT - DCache::LINE_SHIFT);
        let dc_base_idx = self.l2_idx_to_l1_base_idx(idx, pidx, &self.dc);
        for i in 0..l1d_lines_per_l2 {
            let dc_idx = (dc_base_idx + i) & DCache::NUM_LINES_MASK;
            let phys_addr = phys_base + ((i as u64) << DCache::LINE_SHIFT);
            let dc_tag: L1DTag = self.dc.get_tag(dc_idx);

            if dc_tag.cs() != L1D_CS_INVALID && dc_tag.ptag() == self.l1_ptag(phys_addr) {
                self.invalidate_l1d_line(dc_idx, false); // hardware cascade, not coherent
            }
        }

        // Finally invalidate the L2 line itself
        self.l2.set_tag(idx, L2Tag::default());
    }

    /// Write back a dirty L1 data cache line to L2
    /// Since the cache is inclusive, the line must exist in L2
    /// Returns true if writeback was successful
    fn writeback_l1d_line(&self, l1_idx: usize) -> bool {
        let tag: L1DTag = self.dc.get_tag(l1_idx);

        // Check if line is dirty
        if !tag.dirty() {
            return true; // Nothing to write back
        }

        // Reconstruct physical address from tag
        let phys_addr = l1d_tag_to_phys(tag, (l1_idx << DCache::LINE_SHIFT) as u64);

        #[cfg(feature = "debug_cache")]
        if self.is_tracking_l1d(phys_addr) {
            println!("[CACHE DEBUG] writeback_l1d_line: {} l1_idx={}, phys_addr=0x{:08x}, ptag=0x{:06x}, DIRTY",
                     self.tracking_label(phys_addr), l1_idx, phys_addr, tag.ptag());
        }

        // Find the line in L2 using physical address
        let l2_idx = self.l2.get_index(phys_addr);
        let mut l2_tag: L2Tag = self.l2.get_tag(l2_idx);
        let l2_ptag = self.l2_ptag(phys_addr);

        // Verify the line is in L2 (should always be true for inclusive cache)
        if l2_tag.ptag() != l2_ptag {
            // This shouldn't happen in an inclusive cache, but handle it gracefully
            return false;
        }

        // Write data from L1-D to L2
        let dc_data = self.dc.data();
        let l2_data = self.l2.data_mut();

        let l1_start_chunk = l1_idx << DCache::CHUNKS_PER_LINE_SHIFT;

        // Calculate where in the L2 line this data goes
        // L2 lines are typically larger than L1 lines
        let l2_line_base = l2_idx << L2Cache::CHUNKS_PER_LINE_SHIFT;
        let offset_in_l2_line = ((phys_addr & L2Cache::LINE_MASK as u64) >> 3) as usize;

        for i in 0..DCache::CHUNKS_PER_LINE {
            l2_data[l2_line_base + offset_in_l2_line + i] = dc_data[l1_start_chunk + i];
        }

        // Sync l2.instrs for the updated region — read from dc_data (same values just written)
        let l2_instrs = self.l2.instrs.get_mut();
        let instrs_start = (l2_idx << L2Cache::INSTR_SHIFT) + offset_in_l2_line * 2;
        for i in 0..DCache::CHUNKS_PER_LINE {
            let chunk = dc_data[l1_start_chunk + i];
            let r0 = (chunk >> 32) as u32;
            let r1 = chunk as u32;
            let s0 = &mut l2_instrs[instrs_start + i * 2];
            if s0.raw != r0 { s0.decoded = false; }
            s0.raw = r0;
            let s1 = &mut l2_instrs[instrs_start + i * 2 + 1];
            if s1.raw != r1 { s1.decoded = false; }
            s1.raw = r1;
        }

        // Mark L2 line as dirty
        let new_cs = match l2_tag.cs() {
            L2_CS_CLEAN_EXCLUSIVE => L2_CS_DIRTY_EXCLUSIVE,
            L2_CS_SHARED => L2_CS_DIRTY_SHARED,
            cs => cs, // Already dirty or invalid
        };
        l2_tag.set_cs(new_cs);
        self.l2.set_tag(l2_idx, l2_tag);

        // Clear L1 dirty bit after successful writeback
        let mut dc_tag: L1DTag = self.dc.get_tag(l1_idx);
        dc_tag.set_dirty(false);
        self.dc.set_tag(l1_idx, dc_tag);

        true
    }

    /// Write back a dirty L2 cache line to memory
    /// Also writes back any dirty L1-D lines that are part of this L2 line
    /// Returns true if writeback was successful
    fn writeback_l2_line(&self, idx: usize) -> bool {
        let tag: L2Tag = self.l2.get_tag(idx);

        // Reconstruct physical address from tag
        let phys_addr = l2_tag_to_phys(tag, (idx << L2Cache::LINE_SHIFT) as u64);

        // First, writeback any dirty L1-D lines that are part of this L2 line.
        // L1-D is VIPT: reconstruct the virtual index using pidx from the L2 tag.
        let l1d_lines_per_l2 = 1 << (L2Cache::LINE_SHIFT - DCache::LINE_SHIFT);
        let dc_base_idx = self.l2_idx_to_l1_base_idx(idx, tag.pidx(), &self.dc);
        for i in 0..l1d_lines_per_l2 {
            let dc_idx = (dc_base_idx + i) & DCache::NUM_LINES_MASK;
            let phys_addr_l1 = phys_addr + ((i as u64) << DCache::LINE_SHIFT);
            let dc_tag: L1DTag = self.dc.get_tag(dc_idx);

            // Check if this L1-D line matches and is dirty
            if dc_tag.cs() != L1D_CS_INVALID && dc_tag.ptag() == self.l1_ptag(phys_addr_l1) {
                self.writeback_l1d_line(dc_idx);
            }
        }

        // Now check if L2 line is dirty (may have become dirty from L1-D writeback)
        let mut tag: L2Tag = self.l2.get_tag(idx);
        let cs = tag.cs();
        if cs != L2_CS_DIRTY_EXCLUSIVE && cs != L2_CS_DIRTY_SHARED {
            return true; // Nothing to write back
        }

        #[cfg(feature = "debug_cache")]
        if self.is_tracking_l2_idx(idx) {
            println!("[CACHE DEBUG] writeback_l2_line: {} idx={}, phys_addr=0x{:08x}, ptag=0x{:05x}, cs={}, WRITING TO MEMORY",
                     self.tracking_label_l2_idx(idx), idx, phys_addr, tag.ptag(), cs);
            // Dump the L2 line data being written
            let l2_data = self.l2.data();
            let start_chunk = idx << L2Cache::CHUNKS_PER_LINE_SHIFT;
            println!("  L2 line data being written (16 x u64):");
            for i in 0..L2Cache::CHUNKS_PER_LINE {
                let val = l2_data[start_chunk + i];
                println!("    [{}] addr=0x{:08x} val=0x{:016x}", i, phys_addr + ((i as u64) << 3), val);
            }
        }

        // NOTE: do NOT clear llbit here. On R4000, llbit tracks L1-D state only.
        // An L2 writeback/eviction is not a coherency action and must not break LL/SC.

        // Now write L2 data to memory
        let l2_data = self.l2.data();
        let start_chunk = idx << L2Cache::CHUNKS_PER_LINE_SHIFT;

        for i in 0..L2Cache::CHUNKS_PER_LINE {
            let chunk_addr = phys_addr + ((i as u64) << 3);
            let val = l2_data[start_chunk + i];

            if self.downstream.write64(chunk_addr as u32, val) != BUS_OK {
                return false; // Writeback failed
            }
        }

        // Change state to clean after successful writeback
        let new_cs = if cs == L2_CS_DIRTY_EXCLUSIVE { L2_CS_CLEAN_EXCLUSIVE } else { L2_CS_SHARED };
        tag.set_cs(new_cs);
        self.l2.set_tag(idx, tag);
        true
    }

    /// Fill L2 cache line from memory
    /// Evicts current line if needed (with writeback and L1 invalidation)
    /// Returns true if fill was successful
    fn fill_l2_line(&self, phys_addr: u64, virt_addr: u64) -> bool {
        let l2_idx = self.l2.get_index(phys_addr);

        // Writeback and invalidate the victim line (if any)
        // This will also writeback any dirty L1-D lines and invalidate L1-I/L1-D lines
        self.writeback_l2_line(l2_idx);
        self.invalidate_l2_line(l2_idx);

        // Calculate line-aligned address
        let line_base = phys_addr & !(L2Cache::LINE_MASK as u64);

        // Fill line from memory
        let l2_data = self.l2.data_mut();
        let start_chunk = l2_idx << L2Cache::CHUNKS_PER_LINE_SHIFT;

        let instrs_start = l2_idx << L2Cache::INSTR_SHIFT;
        for i in 0..L2Cache::CHUNKS_PER_LINE {
            let fetch_addr = line_base + ((i as u64) << 3);
            let r = self.downstream.read64(fetch_addr as u32);
            if !r.is_ok() { return false; }
            let val = r.data;
            l2_data[start_chunk + i] = val;
            let r0 = (val >> 32) as u32;
            let r1 = val as u32;
            let l2_instrs = self.l2.instrs.get_mut();
            let s0 = &mut l2_instrs[instrs_start + i * 2];
            if s0.raw != r0 { s0.decoded = false; }
            s0.raw = r0;
            let s1 = &mut l2_instrs[instrs_start + i * 2 + 1];
            if s1.raw != r1 { s1.decoded = false; }
            s1.raw = r1;
        }

        // Set tag with CleanExclusive state
        let ptag = self.l2_ptag(phys_addr);
        let pidx = self.pidx(virt_addr);
        let mut new_tag = L2Tag::default();
        new_tag.set_ptag(ptag);
        new_tag.set_cs(L2_CS_CLEAN_EXCLUSIVE);
        new_tag.set_pidx(pidx);
        self.l2.set_tag(l2_idx, new_tag);

        // println!("[CACHE DEBUG] fill_l2_line: idx={}, phys_addr=0x{:08x}, ptag=0x{:05x}, pidx={}, state=CleanExclusive",
        //          l2_idx, phys_addr, ptag, pidx);

        #[cfg(feature = "debug_cache")]
        if self.is_tracking_l2_idx(l2_idx) {
            println!("[CACHE DEBUG] fill_l2_line: {} line 0x{:08x}, idx={}, phys_addr=0x{:08x}, ptag=0x{:05x}, pidx={}",
                     self.tracking_label_l2_idx(l2_idx), line_base, l2_idx, phys_addr, ptag, pidx);
            println!("  L2 line data (16 x u64):");
            for i in 0..L2Cache::CHUNKS_PER_LINE {
                let val = l2_data[start_chunk + i];
                println!("    [{}] 0x{:016x}", i, val);
            }
        }

        true
    }

    /// Fill L1 instruction cache line
    /// Ensures data is in L2 first, then copies to L1-I
    /// For C_FILL operation, phys_addr is used for indexing
    /// Returns `EXEC_COMPLETE` on success, `exec_exception_const(EXC_VCEI)` on VCE,
    /// or `exec_exception_const(EXC_IBE)` on bus error (instruction fetch → IBE, not DBE).
    fn fill_l1i_line(&self, index_addr: u64, phys_addr: u64) -> ExecStatus {
        let ic_idx = self.ic.get_index(index_addr);

        // No writeback needed for I-cache (it's read-only)
        // Just invalidate the victim
        self.invalidate_l1i_line(ic_idx);

        // Check if data is in L2
        let l2_idx = self.l2.get_index(phys_addr);
        let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
        let l2_ptag = self.l2_ptag(phys_addr);

        // Check for L2 hit or miss
        let l2_hit = l2_tag.cs() != L2_CS_INVALID && l2_tag.ptag() == l2_ptag;

        if l2_hit {
            // L2 hit - check for Virtual Coherency Exception (VCEI)
            // VCE occurs when virtual address bits [14:12] don't match L2's stored PIdx
            let virt_pidx = self.pidx(index_addr);
            let stored_pidx = l2_tag.pidx();

            if virt_pidx != stored_pidx {
                return exec_exception_const(EXC_VCEI);
            }
        } else {
            // L2 miss - fill from memory
            if !self.fill_l2_line(phys_addr, index_addr) {
                return exec_exception_const(EXC_IBE);
            }
        }

        // Store l2.instrs slot indices into ic.data (two u32 per u64, high=even, low=odd)
        // Align to L1I line boundary before computing L2 word offset — phys_addr may
        // point to any word within the line, not necessarily the first.
        let ic_line_base = phys_addr & !(ICache::LINE_MASK as u64);
        let l2_word_offset = ((ic_line_base as usize) & L2Cache::LINE_MASK) >> 2;
        let l2_instrs_base = (l2_idx << L2Cache::INSTR_SHIFT) + l2_word_offset;
        let ic_data = self.ic.data_mut();
        let ic_data_base = ic_idx * ICache::CHUNKS_PER_LINE;
        for i in 0..ICache::CHUNKS_PER_LINE {
            let idx0 = (l2_instrs_base + i * 2    ) as u32;
            let idx1 = (l2_instrs_base + i * 2 + 1) as u32;
            ic_data[ic_data_base + i] = ((idx0 as u64) << 32) | (idx1 as u64);
        }

        // Set tag with Valid bit
        let mut ic_tag = L1ITag::default();
        ic_tag.set_ptag(self.l1_ptag(phys_addr));
        ic_tag.set_valid(true);
        self.ic.set_tag(ic_idx, ic_tag);

        EXEC_COMPLETE
    }

    /// Fill L1 data cache line. Ensures data is in L2 first, then copies to L1-D.
    /// Returns `BUS_OK` on success, `BUS_VCE` on VCE, or `BUS_ERR` on bus error.
    fn fill_l1d_line(&self, virt_addr: u64, phys_addr: u64) -> u32 {
        let dc_idx = self.dc.get_index(virt_addr);

        // Writeback and invalidate the victim line (hardware fill — not a coherency action)
        self.writeback_l1d_line(dc_idx);
        self.invalidate_l1d_line(dc_idx, false);

        // Check if data is in L2
        let l2_idx = self.l2.get_index(phys_addr);
        let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
        let l2_ptag = self.l2_ptag(phys_addr);

        // Check for L2 hit or miss
        let l2_hit = l2_tag.cs() != L2_CS_INVALID && l2_tag.ptag() == l2_ptag;

        if l2_hit {
            // L2 hit - check for Virtual Coherency Exception (VCED)
            // VCE occurs when virtual address bits [14:12] don't match L2's stored PIdx
            let virt_pidx = self.pidx(virt_addr);
            let stored_pidx = l2_tag.pidx();

            if virt_pidx != stored_pidx {
                return BUS_VCE;
            }
        } else {
            // L2 miss - fill from memory
            if !self.fill_l2_line(phys_addr, virt_addr) {
                return BUS_ERR;
            }
        }

        // Now copy from L2 to L1-D
        // Align phys_addr to L1-D cache line boundary first
        let dc_line_base = phys_addr & !(DCache::LINE_MASK as u64);
        let l2_line_base = l2_idx << L2Cache::CHUNKS_PER_LINE_SHIFT;
        let offset_in_l2_line = ((dc_line_base & (L2Cache::LINE_MASK as u64)) >> 3) as usize;

        let l2_data = self.l2.data();
        let dc_data = self.dc.data_mut();
        let dc_start_chunk = dc_idx << DCache::CHUNKS_PER_LINE_SHIFT;

        for i in 0..DCache::CHUNKS_PER_LINE {
            dc_data[dc_start_chunk + i] = l2_data[l2_line_base + offset_in_l2_line + i];
        }

        // Set tag with CleanExclusive state
        let mut dc_tag = L1DTag::default();
        dc_tag.set_ptag(self.l1_ptag(phys_addr));
        dc_tag.set_cs(L1D_CS_CLEAN_EXCLUSIVE);
        self.dc.set_tag(dc_idx, dc_tag);

        // println!("[CACHE DEBUG] fill_l1d_line: idx={}, virt_addr=0x{:016x}, phys_addr=0x{:08x}, ptag=0x{:06x}, state=CleanExclusive (BUS_OK)",
        //          dc_idx, virt_addr, phys_addr, ptag);

        #[cfg(feature = "debug_cache")]
        {
            let line_base_phys = phys_addr & !(DCache::LINE_MASK as u64);
            if self.is_tracking_l1d(line_base_phys) {
                let line_base_virt = virt_addr & !(DCache::LINE_MASK as u64);
                println!("[CACHE DEBUG] fill_l1d_line: {} line virt 0x{:016x}, phys 0x{:08x}, idx={}, ptag=0x{:06x}",
                         self.tracking_label(line_base_phys), line_base_virt, line_base_phys, dc_idx, ptag);
                println!("  L1-D line data (4 x u64):");
                for i in 0..DCache::CHUNKS_PER_LINE {
                    let val = dc_data[dc_start_chunk + i];
                    println!("    [{}] 0x{:016x}", i, val);
                }
            }
        }

        BUS_OK
    }
}

impl MipsCache for R4000Cache {
    fn fetch(&self, virt_addr: u64, phys_addr: u64) -> FetchInstrResult {
        #[cfg(feature = "debug_cache")]
        {
            if self.is_tracking_addr(virt_addr, phys_addr) {
                println!("[CACHE DEBUG] fetch: {} virt_addr 0x{:016x}, phys_addr 0x{:016x}",
                         self.tracking_label(phys_addr), virt_addr, phys_addr);
            } else {
                // Also track fetches that will hit the same L2 index (cache line aliasing)
                let l2_idx = self.l2.get_index(phys_addr);
                if self.is_tracking_l2_idx(l2_idx) {
                    let line_base = phys_addr & !(L2Cache::LINE_MASK as u64);
                    println!("[CACHE DEBUG] fetch (L2 alias): idx={}, line 0x{:08x}, virt 0x{:016x}, phys 0x{:016x}",
                             l2_idx, line_base, virt_addr, phys_addr);
                }
            }
        }

        // Ensure line is in L1-I cache
        let ic_idx = self.ic.get_index(virt_addr);
        let ic_tag: L1ITag = self.ic.get_tag(ic_idx);
        let ptag = self.l1_ptag(phys_addr);

        // Fill if not valid or tag mismatch
        #[cfg(feature = "developer")]
        self.l1i_fetch_count.fetch_add(1, Ordering::Relaxed);
        if !ic_tag.valid() || ic_tag.ptag() != ptag {
            let s = self.fill_l1i_line(virt_addr, phys_addr);
            if s != EXEC_COMPLETE { return FetchInstrResult::exception(s); }
        } else {
            #[cfg(feature = "developer")]
            self.l1i_hit_count.fetch_add(1, Ordering::Relaxed);
        }

        // Return pointer to the DecodedInstr slot in l2.instrs.
        // ic.data_as_words() gives a flat &[u32] view; XOR word index with 1 accounts
        // for the big-endian packing ((idx0<<32)|idx1) on a little-endian host.
        let word_offset = ((phys_addr as usize) & ICache::LINE_MASK) >> 2;
        let l2_slot_idx = self.ic.data_as_words()
            [((ic_idx << ICache::INSTR_SHIFT) + word_offset) ^ 1] as usize;
        let slot = &self.l2.instrs.get()[l2_slot_idx] as *const DecodedInstr;
        FetchInstrResult::hit(slot)
    }

    fn read<const SIZE: usize>(&self, virt_addr: u64, phys_addr: u64) -> BusRead64 {
        const { assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE") };
        #[cfg(feature = "debug_cache")]
        {
            if self.is_tracking_addr(virt_addr, phys_addr) {
                println!("[CACHE DEBUG] read: {} virt_addr 0x{:016x}, phys_addr 0x{:016x}, size {}",
                         self.tracking_label(phys_addr), virt_addr, phys_addr, SIZE);
            } else {
                // Also track reads that will hit the same L2 index (cache line aliasing)
                let l2_idx = self.l2.get_index(phys_addr);
                if self.is_tracking_l2_idx(l2_idx) {
                    let line_base = phys_addr & !(L2Cache::LINE_MASK as u64);
                    println!("[CACHE DEBUG] read (L2 alias): idx={}, line 0x{:08x}, virt 0x{:016x}, phys 0x{:016x}, size {}",
                             l2_idx, line_base, virt_addr, phys_addr, SIZE);
                }
            }
        }

        // Ensure line is in L1-D cache
        let dc_idx = self.dc.get_index(virt_addr);
        let dc_tag: L1DTag = self.dc.get_tag(dc_idx);
        let ptag = self.l1_ptag(phys_addr);

        // Fill if not valid or tag mismatch
        if dc_tag.cs() == L1D_CS_INVALID || dc_tag.ptag() != ptag {
            let s = self.fill_l1d_line(virt_addr, phys_addr);
            if s != BUS_OK { return BusRead64 { status: s, data: 0 }; }
        }

        // Read from L1-D cache
        let data_idx = self.dc.get_data_index(virt_addr);
        let data = if SIZE == 1 {
            self.dc.data_as_bytes()[data_idx * 8 + ((phys_addr as usize & 7) ^ 7)] as u64
        } else if SIZE == 2 {
            self.dc.data_as_halves()[data_idx * 4 + ((phys_addr as usize & 7) >> 1 ^ 3)] as u64
        } else if SIZE == 4 {
            self.dc.data_as_words()[data_idx * 2 + ((phys_addr as usize & 7) >> 2 ^ 1)] as u64
        } else {
            self.dc.data()[data_idx]
        };
        BusRead64::ok(data)
    }

    fn write64_masked(&self, virt_addr: u64, phys_addr: u64, val: u64, mask: u64) -> u32 {
        #[cfg(feature = "debug_cache")]
        {
            if self.is_tracking_addr(virt_addr, phys_addr) {
                println!("[CACHE DEBUG] write64_masked: {} virt_addr 0x{:016x}, phys_addr 0x{:016x}, val 0x{:016x}, mask 0x{:016x}",
                         self.tracking_label(phys_addr), virt_addr, phys_addr, val, mask);
            } else {
                // Also track writes that will hit the same L2 index (cache line aliasing)
                let l2_idx = self.l2.get_index(phys_addr);
                if self.is_tracking_l2_idx(l2_idx) {
                    let line_base = phys_addr & !(L2Cache::LINE_MASK as u64);
                    println!("[CACHE DEBUG] write64_masked (L2 alias): idx={}, line 0x{:08x}, virt 0x{:016x}, phys 0x{:016x}, val 0x{:016x}",
                             l2_idx, line_base, virt_addr, phys_addr, val);
                }
            }
        }

        // Ensure line is in L1-D cache (write-allocate)
        let dc_idx = self.dc.get_index(virt_addr);
        let dc_tag: L1DTag = self.dc.get_tag(dc_idx);
        let ptag = self.l1_ptag(phys_addr);

        // Fill if not valid or tag mismatch
        if dc_tag.cs() == L1D_CS_INVALID || dc_tag.ptag() != ptag {
            let s = self.fill_l1d_line(virt_addr, phys_addr);
            if s != BUS_OK { return s; }
        }

        // Write to L1-D cache
        let data_idx = self.dc.get_data_index(virt_addr);
        let dc_data = self.dc.data_mut();
        let current = dc_data[data_idx];
        dc_data[data_idx] = (current & !mask) | (val & mask);

        // Mark line as dirty and update state
        let mut dc_tag: L1DTag = self.dc.get_tag(dc_idx);
        dc_tag.set_dirty(true);
        if dc_tag.cs() == L1D_CS_CLEAN_EXCLUSIVE {
            dc_tag.set_cs(L1D_CS_DIRTY_EXCLUSIVE);
        }
        self.dc.set_tag(dc_idx, dc_tag);

        BUS_OK
    }

    fn cache_op(&self, cache_op: u32, virt_addr: u64, phys_addr: u64) -> u32 {
        // Decode cache operation
        let cache_target = cache_op & 0x3;   // bits [17:16]
        let operation = cache_op & 0x1C;     // bits [20:18] (shifted by 2)

        #[allow(unreachable_patterns)]
        let (is_icache, is_l2) = match cache_target {
            CACH_PI => (true, false),
            CACH_PD => (false, false),
            CACH_SI | CACH_SD => (false, true),
            _ => return 0,
        };

        // L1I and L1D are virtually indexed; L2 is physically indexed
        let idx = if is_l2 {
            self.l2.get_index(phys_addr)
        } else if is_icache {
            self.ic.get_index(virt_addr)
        } else {
            self.dc.get_index(virt_addr)
        };

        match operation {
            // Index Invalidate (I, SI) or Index Writeback Invalidate (D, SD)
            C_IINV => { // same as C_IWBINV
                if is_icache {
                    self.invalidate_l1i_line(idx);
                } else if !is_l2 {
                    self.writeback_l1d_line(idx);
                    self.invalidate_l1d_line(idx, true); // software CACHE op
                } else {
                    self.writeback_l2_line(idx);
                    self.invalidate_l2_line(idx);
                }
                0
            }

            // Index Load Tag — read internal tag, format as CP0 TagLo
            C_ILT => {
                if is_l2 {
                    // L2 TagLo format:
                    //   [31:13] physical tag   [12:10] state   [9:7] PIdx
                    let tag: L2Tag = self.l2.get_tag(idx);
                    let state = match tag.cs() {
                        L2_CS_INVALID => 0,
                        L2_CS_CLEAN_EXCLUSIVE => 4,
                        L2_CS_DIRTY_EXCLUSIVE => 5,
                        L2_CS_SHARED => 6,
                        L2_CS_DIRTY_SHARED => 7,
                        _ => 0,
                    };
                    (tag.ptag() << 13) | (state << 10) | (tag.pidx() << 7)
                } else if is_icache {
                    // L1-I TagLo format:  [31:8] ptag   [7:6] pstate (2=valid, 0=invalid)
                    let tag: L1ITag = self.ic.get_tag(idx);
                    let pstate = if tag.valid() { 2 } else { 0 };
                    (tag.ptag() << 8) | (pstate << 6)
                } else {
                    // L1-D TagLo format:  [31:8] ptag   [7:6] pstate
                    let tag: L1DTag = self.dc.get_tag(idx);
                    let pstate = match tag.cs() {
                        L1D_CS_INVALID => 0,
                        L1D_CS_SHARED => 1,
                        L1D_CS_CLEAN_EXCLUSIVE => 2,
                        L1D_CS_DIRTY_EXCLUSIVE => 3,
                        _ => 0,
                    };
                    (tag.ptag() << 8) | (pstate << 6)
                }
            }

            // Index Store Tag — write CP0 TagLo into internal tag
            C_IST => {
                let tag_lo = phys_addr as u32;

                if is_l2 {
                    // L2 TagLo format:  [31:13] ptag   [12:10] state   [9:7] PIdx
                    let ptag = (tag_lo >> 13) & L2_PTAG_MASK;
                    let state = (tag_lo >> 10) & 0x7;
                    let pidx = (tag_lo >> 7) & L2_PIDX_VADDR_MASK;
                    let cs = match state {
                        0 => L2_CS_INVALID,
                        4 => L2_CS_CLEAN_EXCLUSIVE,
                        5 => L2_CS_DIRTY_EXCLUSIVE,
                        6 => L2_CS_SHARED,
                        7 => L2_CS_DIRTY_SHARED,
                        _ => L2_CS_INVALID,
                    };
                    // Evict the existing L2 occupant first to maintain L1 inclusivity.
                    // C_IST does not writeback (it's used for cache init/invalidation).
                    self.invalidate_l2_line(idx);
                    let mut t = L2Tag::default();
                    t.set_ptag(ptag);
                    t.set_cs(cs);
                    t.set_pidx(pidx);
                    self.l2.set_tag(idx, t);
                } else {
                    // L1 TagLo format:  [31:8] ptag   [7:6] pstate
                    let ptag = (tag_lo >> 8) & L1_PTAG_MASK;
                    let pstate = (tag_lo >> 6) & 0x3;

                    if is_icache {
                        // Evict existing line first to maintain L1I data pointer integrity.
                        self.invalidate_l1i_line(idx);
                        let mut t = L1ITag::default();
                        if pstate != 0 {
                            t.set_ptag(ptag);
                            t.set_valid(true);
                        }
                        self.ic.set_tag(idx, t);
                    } else {
                        let cs = match pstate {
                            0 => L1D_CS_INVALID,
                            1 => L1D_CS_SHARED,
                            2 => L1D_CS_CLEAN_EXCLUSIVE,
                            3 => L1D_CS_DIRTY_EXCLUSIVE,
                            _ => L1D_CS_INVALID,
                        };
                        // Writeback dirty data before overwriting the tag.
                        self.writeback_l1d_line(idx);
                        self.invalidate_l1d_line(idx, true);
                        let mut t = L1DTag::default();
                        t.set_ptag(ptag);
                        t.set_cs(cs);
                        self.dc.set_tag(idx, t);
                    }
                }
                0
            }

            // Create Dirty Exclusive
            C_CDX => {
                if is_icache {
                    return 0; // Not valid for I-cache
                }

                if is_l2 {
                    // Evict the existing L2 occupant first to maintain L1 inclusivity.
                    // C_CDX creates a new dirty line; old occupant's L1I lines must be swept.
                    self.invalidate_l2_line(idx);
                    let mut t = L2Tag::default();
                    t.set_ptag(self.l2_ptag(phys_addr));
                    t.set_cs(L2_CS_DIRTY_EXCLUSIVE);
                    t.set_pidx(self.pidx(virt_addr));
                    self.l2.set_tag(idx, t);
                } else {
                    let mut t = L1DTag::default();
                    t.set_ptag(self.l1_ptag(phys_addr));
                    t.set_cs(L1D_CS_DIRTY_EXCLUSIVE);
                    t.set_dirty(true);
                    self.dc.set_tag(idx, t);
                }
                0
            }

            // Hit Invalidate
            C_HINV => {
                let hit = if is_l2 {
                    let tag: L2Tag = self.l2.get_tag(idx);
                    tag.ptag() == self.l2_ptag(phys_addr)
                } else if is_icache {
                    let tag: L1ITag = self.ic.get_tag(idx);
                    tag.ptag() == self.l1_ptag(phys_addr)
                } else {
                    let tag: L1DTag = self.dc.get_tag(idx);
                    tag.ptag() == self.l1_ptag(phys_addr)
                };

                if hit {
                    if is_icache {
                        self.invalidate_l1i_line(idx);
                    } else if !is_l2 {
                        self.invalidate_l1d_line(idx, true); // software CACHE op
                    } else {
                        self.invalidate_l2_line(idx);
                    }
                }
                0
            }

            // Hit Writeback Invalidate (D, SD) or Fill (I)
            C_HWBINV => { // same as C_FILL
                if is_icache {
                    // Fill operation: L1I is virtually indexed, use virt_addr for index
                    self.fill_l1i_line(virt_addr, phys_addr);
                } else {
                    let hit = if is_l2 {
                        let tag: L2Tag = self.l2.get_tag(idx);
                        tag.ptag() == self.l2_ptag(phys_addr)
                    } else {
                        let tag: L1DTag = self.dc.get_tag(idx);
                        tag.ptag() == self.l1_ptag(phys_addr)
                    };

                    if hit {
                        if !is_l2 {
                            self.writeback_l1d_line(idx);
                            self.invalidate_l1d_line(idx, true); // software CACHE op
                        } else {
                            self.writeback_l2_line(idx);
                            self.invalidate_l2_line(idx);
                        }
                    }
                }
                0
            }

            // Hit Writeback
            C_HWB => {
                if !is_icache {
                    let hit = if is_l2 {
                        let tag: L2Tag = self.l2.get_tag(idx);
                        tag.ptag() == self.l2_ptag(phys_addr)
                    } else {
                        let tag: L1DTag = self.dc.get_tag(idx);
                        tag.ptag() == self.l1_ptag(phys_addr)
                    };

                    if hit {
                        if !is_l2 {
                            self.writeback_l1d_line(idx);
                        } else {
                            self.writeback_l2_line(idx);
                        }
                    }
                }
                0
            }

            // Hit Set Virtual (SI, SD)
            C_HSV => {
                if is_l2 {
                    let mut tag: L2Tag = self.l2.get_tag(idx);
                    if tag.ptag() == self.l2_ptag(phys_addr) {
                        tag.set_pidx(self.pidx(virt_addr));
                        self.l2.set_tag(idx, tag);
                    }
                }
                0
            }

            _ => 0,
        }
    }

    fn get_config(&self, cache_target: u32) -> (usize, usize) {
        match cache_target {
            CACH_PI => (IC_SIZE, IC_LINE),
            CACH_PD => (DC_SIZE, DC_LINE),
            CACH_SI | CACH_SD => (L2_SIZE, L2_LINE),
            _ => (0, 16),
        }
    }

    fn downstream(&self) -> Arc<dyn BusDevice> {
        self.downstream.clone()
    }

    fn check_and_clear_llbit(&self, phys_addr: u64) {
        if !self.get_llbit() {
            return;
        }
        let ll_addr = (self.get_lladdr() as u64) << 4;
        let addr_line = phys_addr & !(DCache::LINE_MASK as u64);
        let ll_line = ll_addr & !(DCache::LINE_MASK as u64);
        if addr_line == ll_line {
            self.set_llbit(false);
        }
    }

    fn get_llbit(&self) -> bool {
        unsafe { *self.llbit.get() }
    }

    fn set_llbit(&self, val: bool) {
        unsafe { *self.llbit.get() = val; }
    }

    fn get_lladdr(&self) -> u32 {
        unsafe { *self.lladdr.get() }
    }

    fn set_lladdr(&self, addr: u32) {
        unsafe { *self.lladdr.get() = addr; }
    }

    fn debug_probe(&self, cache_name: &str, virt_addr: u64, phys_addr: u64) -> String {
        match cache_name {
            "l1i" => {
                // L1I is virtually indexed (use virt_addr for index)
                let idx = self.ic.get_index(virt_addr);
                let tag: L1ITag = self.ic.get_tag(idx);
                let wanted_tag = self.l1_ptag(phys_addr);
                let status = if tag.valid() && tag.ptag() == wanted_tag { "HIT" } else { "MISS" };

                format!("{} at index 0x{:x} (virt 0x{:016x})\n  Tag: 0x{:06x} (Wanted: 0x{:06x})\n  Valid: {}",
                    status, idx, virt_addr, tag.ptag(), wanted_tag, tag.valid())
            }
            "l1d" => {
                // L1D is virtually indexed (use virt_addr for index)
                let idx = self.dc.get_index(virt_addr);
                let tag: L1DTag = self.dc.get_tag(idx);
                let wanted_tag = self.l1_ptag(phys_addr);
                let status = if tag.cs() != L1D_CS_INVALID && tag.ptag() == wanted_tag { "HIT" } else { "MISS" };

                let cs_str = match tag.cs() {
                    L1D_CS_INVALID => "Invalid",
                    L1D_CS_SHARED => "Shared",
                    L1D_CS_CLEAN_EXCLUSIVE => "CleanExclusive",
                    L1D_CS_DIRTY_EXCLUSIVE => "DirtyExclusive",
                    _ => "Unknown",
                };

                format!("{} at index 0x{:x} (virt 0x{:016x})\n  Tag: 0x{:06x} (Wanted: 0x{:06x})\n  CS: {} ({})\n  Dirty: {}",
                    status, idx, virt_addr, tag.ptag(), wanted_tag, tag.cs(), cs_str, tag.dirty())
            }
            "l2" => {
                // L2 is physically indexed
                let idx = self.l2.get_index(phys_addr);
                let tag: L2Tag = self.l2.get_tag(idx);
                let wanted_tag = self.l2_ptag(phys_addr);
                let virt_pidx = self.pidx(virt_addr);
                let status = if tag.cs() != L2_CS_INVALID && tag.ptag() == wanted_tag { "HIT" } else { "MISS" };
                let pidx_ok = tag.pidx() == virt_pidx;

                let cs_str = match tag.cs() {
                    L2_CS_INVALID => "Invalid",
                    L2_CS_CLEAN_EXCLUSIVE => "CleanExclusive",
                    L2_CS_DIRTY_EXCLUSIVE => "DirtyExclusive",
                    L2_CS_SHARED => "Shared",
                    L2_CS_DIRTY_SHARED => "DirtyShared",
                    _ => "Reserved",
                };

                let vce_warn = if status == "HIT" && !pidx_ok { " *** VCE would fire!" } else { "" };
                format!("{} at index 0x{:x} (phys 0x{:016x})\n  Tag: 0x{:05x} (Wanted: 0x{:05x})\n  CS: {} ({})\n  PIdx: stored={} virt={}{}",
                    status, idx, phys_addr, tag.ptag(), wanted_tag, tag.cs(), cs_str, tag.pidx(), virt_pidx, vce_warn)
            }
            _ => format!("Unknown cache: {}", cache_name),
        }
    }

    fn debug_dump_line(&self, cache_name: &str, idx: usize) -> String {
        match cache_name {
            "l1i" => {
                if idx >= ICache::NUM_LINES {
                    return format!("Index 0x{:x} out of bounds (max 0x{:x})", idx, ICache::NUM_LINES - 1);
                }
                let tag: L1ITag = self.ic.get_tag(idx);

                // Decoded slots live in L2 — look up via ic.data indices.
                let instrs_per_ic_line = ICache::INSTRS_PER_LINE;
                let ic_instrs = self.l2.instrs.get();

                let mut s = format!("L1-I Line 0x{:x}: Tag=0x{:06x} V={}\n  Instrs:", idx, tag.ptag(), tag.valid());
                let ic_data_words = self.ic.data_as_words();
                let ic_instrs_base = idx << ICache::INSTR_SHIFT;
                for i in 0..instrs_per_ic_line {
                    if i % 4 == 0 { s.push_str("\n    "); }
                    let l2_slot_idx = ic_data_words[(ic_instrs_base + i) ^ 1] as usize;
                    if l2_slot_idx < ic_instrs.len() {
                        s.push_str(&format!("{:08x} ", ic_instrs[l2_slot_idx].raw));
                    }
                }
                s
            }
            "l1d" => {
                if idx >= DCache::NUM_LINES {
                    return format!("Index 0x{:x} out of bounds (max 0x{:x})", idx, DCache::NUM_LINES - 1);
                }
                let tag: L1DTag = self.dc.get_tag(idx);
                let cs_str = match tag.cs() {
                    L1D_CS_INVALID => "Invalid",
                    L1D_CS_SHARED => "Shared",
                    L1D_CS_CLEAN_EXCLUSIVE => "CleanExclusive",
                    L1D_CS_DIRTY_EXCLUSIVE => "DirtyExclusive",
                    _ => "Unknown",
                };

                let dc_data = self.dc.data();
                let start = idx << DCache::CHUNKS_PER_LINE_SHIFT;

                let mut s = format!("L1-D Line 0x{:x}: Tag=0x{:06x} CS={} ({}) D={}\n  Data:",
                    idx, tag.ptag(), tag.cs(), cs_str, tag.dirty());
                for i in 0..DCache::CHUNKS_PER_LINE {
                    if i % 4 == 0 { s.push_str("\n    "); }
                    if start + i < dc_data.len() {
                        s.push_str(&format!("{:016x} ", dc_data[start + i]));
                    }
                }
                s
            }
            "l2" => {
                if idx >= L2Cache::NUM_LINES {
                    return format!("Index 0x{:x} out of bounds (max 0x{:x})", idx, L2Cache::NUM_LINES - 1);
                }
                let tag: L2Tag = self.l2.get_tag(idx);
                let cs_str = match tag.cs() {
                    L2_CS_INVALID => "Invalid",
                    L2_CS_CLEAN_EXCLUSIVE => "CleanExclusive",
                    L2_CS_DIRTY_EXCLUSIVE => "DirtyExclusive",
                    L2_CS_SHARED => "Shared",
                    L2_CS_DIRTY_SHARED => "DirtyShared",
                    _ => "Reserved",
                };

                let l2_data = self.l2.data();
                let start = idx << L2Cache::CHUNKS_PER_LINE_SHIFT;

                let mut s = format!("L2 Line 0x{:x}: Tag=0x{:05x} CS={} ({})\n  Data:",
                    idx, tag.ptag(), tag.cs(), cs_str);
                for i in 0..L2Cache::CHUNKS_PER_LINE {
                    if i % 4 == 0 { s.push_str("\n    "); }
                    if start + i < l2_data.len() {
                        s.push_str(&format!("{:016x} ", l2_data[start + i]));
                    }
                }
                s
            }
            _ => format!("Unknown cache: {}", cache_name),
        }
    }

    fn power_on(&self) {
        self.ic.tags_mut().fill(0);
        self.ic.data_mut().fill(0);
        self.dc.tags_mut().fill(0);
        self.dc.data_mut().fill(0);
        self.l2.tags_mut().fill(0);
        self.l2.data_mut().fill(0);
        for s in self.l2.instrs.get_mut().iter_mut() {
            s.decoded = false;
            s.raw = 0;
        }
        unsafe {
            *self.llbit.get() = false;
            *self.lladdr.get() = 0;
        }
    }

    fn save_cache_state(&self) -> toml::Value {
        R4000Cache::save_cache_state(self)
    }

    fn load_cache_state(&self, v: &toml::Value) -> Result<(), String> {
        R4000Cache::load_cache_state(self, v)
    }
}

// ---- Drop: stop and join decode thread ----

impl Drop for R4000Cache {
    fn drop(&mut self) {
        self.ic.stop.store(true, Ordering::Relaxed);
    }
}

// ---- Resettable ----

impl Resettable for R4000Cache {
    fn power_on(&self) {
        self.ic.tags_mut().fill(0);
        self.ic.data_mut().fill(0);
        self.dc.tags_mut().fill(0);
        self.dc.data_mut().fill(0);
        self.l2.tags_mut().fill(0);
        self.l2.data_mut().fill(0);
        for s in self.l2.instrs.get_mut().iter_mut() {
            s.decoded = false;
            s.raw = 0;
        }
        unsafe {
            *self.llbit.get() = false;
            *self.lladdr.get() = 0;
        }
    }
}

// ---- snapshot helpers + MipsCache save/load override ----

impl R4000Cache {
    fn save_cache_inner<const S: usize, const L: usize, const K: u8, const TG: usize, const DA: usize>(c: &Cache<S, L, K, TG, DA>) -> (Vec<u32>, Vec<u64>) {
        (c.tags().to_vec(), c.data().to_vec())
    }

    fn load_cache_inner<const S: usize, const L: usize, const K: u8, const TG: usize, const DA: usize>(c: &Cache<S, L, K, TG, DA>, tags: &[u32], data: &[u64]) {
        let tl = tags.len().min(TG);
        c.tags_mut()[..tl].copy_from_slice(&tags[..tl]);
        let dl = data.len().min(DA);
        c.data_mut()[..dl].copy_from_slice(&data[..dl]);
    }

    pub fn save_cache_state(&self) -> toml::Value {
        let (ic_tags, ic_data) = Self::save_cache_inner(&self.ic);
        let (dc_tags, dc_data) = Self::save_cache_inner(&self.dc);
        let (l2_tags, l2_data) = Self::save_cache_inner(&self.l2);
        let llbit = unsafe { *self.llbit.get() };
        let lladdr = unsafe { *self.lladdr.get() };

        let mut t = toml::value::Table::new();
        t.insert("ic_tags".into(),  u32_slice_to_toml(&ic_tags));
        t.insert("ic_data".into(),  u64_slice_to_toml(&ic_data));
        t.insert("dc_tags".into(),  u32_slice_to_toml(&dc_tags));
        t.insert("dc_data".into(),  u64_slice_to_toml(&dc_data));
        t.insert("l2_tags".into(),  u32_slice_to_toml(&l2_tags));
        t.insert("l2_data".into(),  u64_slice_to_toml(&l2_data));
        t.insert("llbit".into(),    toml::Value::Boolean(llbit));
        t.insert("lladdr".into(),   hex_u32(lladdr));
        toml::Value::Table(t)
    }

    pub fn load_cache_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut ic_tags = vec![0u32; ICache::NUM_LINES];
        let mut ic_data = vec![0u64; IC_SIZE / 8];
        let mut dc_tags = vec![0u32; DCache::NUM_LINES];
        let mut dc_data = vec![0u64; DC_SIZE / 8];
        let mut l2_tags = vec![0u32; L2Cache::NUM_LINES];
        let mut l2_data = vec![0u64; L2_SIZE / 8];

        if let Some(f) = get_field(v, "ic_tags") { load_u32_slice(f, &mut ic_tags); }
        if let Some(f) = get_field(v, "ic_data") { load_u64_slice(f, &mut ic_data); }
        if let Some(f) = get_field(v, "dc_tags") { load_u32_slice(f, &mut dc_tags); }
        if let Some(f) = get_field(v, "dc_data") { load_u64_slice(f, &mut dc_data); }
        if let Some(f) = get_field(v, "l2_tags") { load_u32_slice(f, &mut l2_tags); }
        if let Some(f) = get_field(v, "l2_data") { load_u64_slice(f, &mut l2_data); }

        // Load ic tags and ic data (ic.data stores l2.instrs slot indices)
        Self::load_cache_inner(&self.ic, &ic_tags, &ic_data);
        Self::load_cache_inner(&self.dc, &dc_tags, &dc_data);
        Self::load_cache_inner(&self.l2, &l2_tags, &l2_data);

        // Rebuild l2.instrs from restored l2.data
        {
            let l2_data_slice = self.l2.data();
            let l2_instrs = self.l2.instrs.get_mut();
            for line in 0..L2Cache::NUM_LINES {
                let chunks_start = line << L2Cache::CHUNKS_PER_LINE_SHIFT;
                let instrs_start = line << L2Cache::INSTR_SHIFT;
                for i in 0..L2Cache::CHUNKS_PER_LINE {
                    let chunk = l2_data_slice[chunks_start + i];
                    l2_instrs[instrs_start + i * 2].raw = (chunk >> 32) as u32;
                    l2_instrs[instrs_start + i * 2].decoded = false;
                    l2_instrs[instrs_start + i * 2 + 1].raw = chunk as u32;
                    l2_instrs[instrs_start + i * 2 + 1].decoded = false;
                }
            }
        }
        // ic.data (slot indices) is already restored by load_cache_inner above;
        // l2.instrs is rebuilt from l2.data, so indices remain valid.

        if let Some(f) = get_field(v, "llbit") {
            if let Some(b) = toml_bool(f) { unsafe { *self.llbit.get() = b; } }
        }
        if let Some(f) = get_field(v, "lladdr") {
            if let Some(a) = toml_u32(f) { unsafe { *self.lladdr.get() = a; } }
        }
        Ok(())
    }
}
