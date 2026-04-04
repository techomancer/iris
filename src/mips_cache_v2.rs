// MIPS R4000 Cache Implementation - Version 2
//
// This is a complete rewrite to properly support R4000 cache semantics:
// - Unified cache object containing L1-I, L1-D, and L2
// - Proper VIPT (Virtually Indexed, Physically Tagged) support
// - R4000-compliant tag format with PState bits
// - L2 can signal back to L1 for evictions

use crate::traits::{BusRead64, BusDevice, Resettable, BUS_OK, BUS_BUSY, BUS_ERR, BUS_VCE};
use crate::snapshot::{u32_slice_to_toml, u64_slice_to_toml, load_u32_slice, load_u64_slice, get_field, toml_bool, toml_u32, hex_u32};
use crate::mips_exec::DecodedInstr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::cell::UnsafeCell;
use bitfield::bitfield;

/// Result of a fetch from the L1 instruction cache.
/// On hit, returns a raw pointer to the DecodedInstr slot in the cache (or scratch buffer).
/// The caller is responsible for calling decode_into on the slot before use.
pub enum FetchResult {
    /// Pointer to the DecodedInstr slot. Valid for the lifetime of the cache.
    Hit(*const DecodedInstr),
    VirtualCoherencyException,
    Error,
    Busy,
}

/// Result of cache line fill operations
#[derive(Debug, Clone, Copy, PartialEq)]
enum FillResult {
    Ok,
    Error,
    VirtualCoherencyException,
}

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
    /// Returns a pointer to the DecodedInstr slot (in cache or scratch).
    /// The caller must call decode_into on the slot before use.
    fn fetch(&self, virt_addr: u64, phys_addr: u64) -> FetchResult;

    /// Read data using virtual and physical addresses.
    /// Uses virtual address for index, physical address for tag (VIPT).
    /// Size must be 1, 2, 4, or 8 bytes.
    /// Returns BusRead64 with data zero-extended to u64 on success.
    /// status may be BUS_OK, BUS_BUSY, BUS_ERR, or BUS_VCE (cache only).
    fn read(&self, virt_addr: u64, phys_addr: u64, size: usize) -> BusRead64;

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

impl From<(Arc<dyn BusDevice>, R4000CacheConfig)> for PassthroughCache {
    fn from((downstream, _config): (Arc<dyn BusDevice>, R4000CacheConfig)) -> Self {
        Self::new(downstream)
    }
}

impl MipsCache for PassthroughCache {
    fn fetch(&self, _virt_addr: u64, phys_addr: u64) -> FetchResult {
        let r = self.downstream.read32(phys_addr as u32);
        if r.is_ok() {
            let slot = unsafe { &mut *self.fetch_scratch.get() };
            if slot.raw != r.data { slot.decoded = false; }
            slot.raw = r.data;
            FetchResult::Hit(slot as *const DecodedInstr)
        } else if r.status == BUS_BUSY {
            FetchResult::Busy
        } else {
            FetchResult::Error
        }
    }

    fn read(&self, _virt_addr: u64, phys_addr: u64, size: usize) -> BusRead64 {
        match size {
            1 => { let r = self.downstream.read8(phys_addr as u32);  BusRead64 { status: r.status, data: r.data as u64 } }
            2 => { let r = self.downstream.read16(phys_addr as u32); BusRead64 { status: r.status, data: r.data as u64 } }
            4 => { let r = self.downstream.read32(phys_addr as u32); BusRead64 { status: r.status, data: r.data as u64 } }
            8 =>   self.downstream.read64(phys_addr as u32),
            _ => BusRead64::err(),
        }
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

/// A single cache level - used for L1-I, L1-D, or L2
///
/// Structure:
/// - `tags`: Array of size `num_lines` u32 tags (raw; use `get_tag`/`set_tag` for typed access)
/// - `data`: Array of size `cache_size / 8` u64 values (entire cache data)
/// - `instrs`: L1-I only — one DecodedInstr per 4-byte instruction slot. Empty for L1-D/L2.
/// - Various shifts and masks for efficient indexing
unsafe impl Send for Cache {}
unsafe impl Sync for Cache {}

struct Cache {
    /// Cache tags - one u32 per cache line (use get_tag/set_tag for typed access)
    tags: CacheVec<u32>,

    /// Cache data - stored as u64 chunks
    data: CacheVec<u64>,

    /// L1-I decoded instruction slots — one per 4-byte word (cache_size / 4 entries).
    /// Empty (zero-capacity) for L1-D and L2.
    instrs: CacheVec<DecodedInstr>,

    /// Number of 4-byte instruction slots per cache line. Zero for non-I-cache.
    instrs_per_line: usize,

    /// log2(instrs_per_line): shift to get instr index from byte offset within line.
    instr_shift: u32,

    /// instrs_per_line - 1: mask for wrapping instr index within a line.
    instr_mask: usize,

    /// Signals the decode thread to stop (kept for Drop compatibility).
    stop: Arc<AtomicBool>,

    /// Total cache size in bytes
    cache_size: usize,

    /// Cache line size in bytes
    line_size: usize,

    /// Number of cache lines
    num_lines: usize,

    /// Shift to get line index from address (line_size.trailing_zeros())
    line_shift: u32,

    /// Mask for line size (line_size - 1)
    line_mask: usize,

    /// Mask for number of lines (num_lines - 1)
    num_lines_mask: usize,

    /// Shift to get cache size alignment (cache_size.trailing_zeros())
    cache_size_shift: u32,

    /// Number of u64 chunks per cache line (line_size / 8)
    chunks_per_line: usize,

    /// Shift to get chunk offset within line (line_shift - 3)
    chunks_per_line_shift: u32,
}

impl Cache {
    fn new(cache_size: usize, line_size: usize) -> Self {
        assert!(cache_size.is_power_of_two());
        assert!(line_size.is_power_of_two());
        assert!(line_size >= 8);
        assert!(cache_size >= line_size);

        let num_lines = cache_size / line_size;
        let line_shift = line_size.trailing_zeros();
        let chunks_per_line = line_size / 8;

        Self {
            tags: CacheVec::new(vec![0; num_lines]),
            data: CacheVec::new(vec![0; cache_size / 8]),
            instrs: CacheVec::new(Vec::new()),
            instrs_per_line: 0,
            instr_shift: 0,
            instr_mask: 0,
            stop: Arc::new(AtomicBool::new(false)),
            cache_size,
            line_size,
            num_lines,
            line_shift,
            line_mask: line_size - 1,
            num_lines_mask: num_lines - 1,
            cache_size_shift: cache_size.trailing_zeros(),
            chunks_per_line,
            chunks_per_line_shift: line_shift - 3,
        }
    }

    /// Initialise the L1-I cache. Call once after construction.
    /// ic.data is repurposed: stores l2.instrs slot indices (two u32 per u64).
    /// ic.instrs not used — decoded slots live in l2.instrs.
    fn init_icache(&mut self) {
        self.instrs_per_line = self.line_size / 4;
        self.instr_shift = self.line_shift - 2;
        self.instr_mask = self.instrs_per_line - 1;
    }

    /// Initialise L2 decoded-instruction arrays. Call once after construction.
    fn init_l2cache(&mut self) {
        let num_slots = self.cache_size / 4;
        *self.instrs.get_mut() = (0..num_slots).map(|_| DecodedInstr::default()).collect();
        self.instrs_per_line = self.line_size / 4;
        self.instr_shift = self.line_shift - 2;
        self.instr_mask  = self.instrs_per_line - 1;
    }

    /// Get cache line index from address
    #[inline]
    fn get_index(&self, addr: u64) -> usize {
        ((addr >> self.line_shift) as usize) & self.num_lines_mask
    }

    /// Get offset within cache line
    #[inline]
    fn get_line_offset(&self, addr: u64) -> usize {
        (addr as usize) & self.line_mask
    }

    /// Get data chunk index for a given address
    #[inline]
    fn get_data_index(&self, addr: u64) -> usize {
        let line_idx = self.get_index(addr);
        let chunk_offset = self.get_line_offset(addr) >> 3; // divide by 8
        (line_idx << self.chunks_per_line_shift) + chunk_offset
    }

    /// Read the tag at `idx` as a typed bitfield struct
    #[inline]
    fn get_tag<T: From<u32>>(&self, idx: usize) -> T {
        T::from(self.tags.get()[idx])
    }

    /// Write a typed bitfield tag to `idx`
    #[inline]
    fn set_tag<T: Into<u32>>(&self, idx: usize, tag: T) {
        self.tags.get_mut()[idx] = tag.into();
    }

    /// View ic.data as a flat &[u32] slice (two slot indices per u64, big-endian word order).
    /// On a little-endian host the packing `(idx0 << 32) | idx1` means idx1 is at even
    /// u32 offset and idx0 is at odd — so XOR the word index with 1 to address naturally.
    #[inline]
    fn data_as_words(&self) -> &[u32] {
        let slice = self.data.get();
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u32, slice.len() * 2) }
    }

    /// View cache data as flat &[u16] (big-endian halfword order within each u64).
    /// XOR halfword index with 3 to convert MIPS big-endian address to host offset.
    #[inline]
    fn data_as_halves(&self) -> &[u16] {
        let slice = self.data.get();
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u16, slice.len() * 4) }
    }

    /// View cache data as flat &[u8] (big-endian byte order within each u64).
    /// XOR byte index with 7 to convert MIPS big-endian address to host offset.
    #[inline]
    fn data_as_bytes(&self) -> &[u8] {
        let slice = self.data.get();
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 8) }
    }
}

// =============================================================================
// R4000 Cache Implementation - Full 2-level hierarchy
// =============================================================================

/// Configuration for R4000 cache
#[derive(Clone, Copy, Debug)]
pub struct R4000CacheConfig {
    // L1 Instruction Cache
    pub ic_size: usize,
    pub ic_line_size: usize,

    // L1 Data Cache
    pub dc_size: usize,
    pub dc_line_size: usize,

    // L2 Unified Cache
    pub l2_size: usize,
    pub l2_line_size: usize,
}

impl Default for R4000CacheConfig {
    fn default() -> Self {
        Self {
            ic_size: 16 * 1024,      // 16KB
            ic_line_size: 16,
            dc_size: 16 * 1024,      // 16KB
            dc_line_size: 16,
            l2_size: 1024 * 1024,    // 1MB
            l2_line_size: 128,
        }
    }
}

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

    // L1 Instruction Cache
    ic: Cache,

    // L1 Data Cache
    dc: Cache,

    // L2 Unified Cache
    l2: Cache,

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

impl R4000Cache {
    pub fn new(downstream: Arc<dyn BusDevice>, config: R4000CacheConfig) -> Self {
        let mut ic = Cache::new(config.ic_size, config.ic_line_size);
        ic.init_icache();
        let dc = Cache::new(config.dc_size, config.dc_line_size);
        let mut l2 = Cache::new(config.l2_size, config.l2_line_size);
        l2.init_l2cache();

        #[cfg(feature = "debug_cache")]
        let (debug_l1d_line, debug_l2_line, debug_companion_l1d_line, debug_companion_l2_line,
             debug_l1d_idx, debug_l2_idx, debug_companion_l2_idx) = {
            if let Some(addr) = DEBUG_TRACK_ADDR {
                let l1_line_mask = dc.line_mask as u64;
                let l2_line_mask = l2.line_mask as u64;
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


impl From<(Arc<dyn BusDevice>, R4000CacheConfig)> for R4000Cache {
    fn from((downstream, config): (Arc<dyn BusDevice>, R4000CacheConfig)) -> Self {
        Self::new(downstream, config)
    }
}

impl R4000Cache {
    /// Check if we're tracking this physical address (for debug purposes)
    #[cfg(feature = "debug_cache")]
    #[inline]
    fn is_tracking_l1d(&self, phys_addr: u64) -> bool {
        DEBUG_TRACK_ADDR.is_some() && {
            let line = phys_addr & !(self.dc.line_mask as u64);
            line == self.debug_l1d_line || line == self.debug_companion_l1d_line
        }
    }

    #[cfg(feature = "debug_cache")]
    #[inline]
    fn is_tracking_l2(&self, phys_addr: u64) -> bool {
        DEBUG_TRACK_ADDR.is_some() && {
            let line = phys_addr & !(self.l2.line_mask as u64);
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
            let line = phys_addr & !(self.dc.line_mask as u64);
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
        let line = phys_addr & !(self.dc.line_mask as u64);
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
    fn l2_idx_to_l1_base_idx(&self, l2_idx: usize, pidx: u32, l1: &Cache) -> usize {
        // Physical bits of the L2 line start address that are below bit 12 (page boundary)
        // These bits are the same in VA and PA, so we can derive them from the L2 index.
        let l2_line_shift = self.l2.line_shift;
        let phys_sub_bits = (l2_idx << l2_line_shift as usize) & 0xFFF; // bits [11:0] of L2 line start
        // Reconstruct the virtual address bits used for L1 indexing
        let virt_index_bits = ((pidx as usize) << L2_PIDX_VADDR_SHIFT as usize) | phys_sub_bits;
        (virt_index_bits >> l1.line_shift as usize) & l1.num_lines_mask
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
                let phys_addr = l1_tag_to_phys(tag, (idx << self.ic.line_shift) as u64);
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
                let phys_addr = l1d_tag_to_phys(tag, (idx << self.dc.line_shift) as u64);
                println!("[CACHE DEBUG] invalidate_l1d_line: {} idx={}, phys_addr=0x{:08x}, ptag=0x{:06x}, cs={}, coherent={}",
                         self.tracking_label(phys_addr), idx, phys_addr, tag.ptag(), tag.cs(), coherent);
            } else {
                println!("[CACHE DEBUG] invalidate_l1d_line: idx={} (already invalid)", idx);
            }
        }

        // Only clear llbit for software-initiated coherency invalidations, not hardware fills.
        // On a uniprocessor R4000 there are no external snoops; llbit survives capacity evictions.
        if coherent && tag.cs() != L1D_CS_INVALID {
            let phys_addr = l1d_tag_to_phys(tag, (idx << self.dc.line_shift) as u64);
            self.check_and_clear_llbit(phys_addr, self.dc.line_mask);
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
                let phys_base = l2_tag_to_phys(l2_tag, (idx << self.l2.line_shift) as u64);
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
        let phys_base = l2_tag_to_phys(l2_tag, (idx << self.l2.line_shift) as u64);

        // NOTE: do NOT clear llbit here. On R4000, llbit tracks L1-D state only.
        // An L2 eviction is not a coherency action and must not break LL/SC.

        // Check L1-I for any lines from this L2 line.
        // L1-I is VIPT so we must reconstruct the virtual index from pidx + physical sub-bits.
        let pidx = l2_tag.pidx();
        let l1i_lines_per_l2 = 1 << (self.l2.line_shift - self.ic.line_shift);
        let ic_base_idx = self.l2_idx_to_l1_base_idx(idx, pidx, &self.ic);
        for i in 0..l1i_lines_per_l2 {
            let ic_idx = (ic_base_idx + i) & self.ic.num_lines_mask;
            let phys_addr = phys_base + ((i as u64) << self.ic.line_shift);
            let ic_tag: L1ITag = self.ic.get_tag(ic_idx);
            if ic_tag.valid() && ic_tag.ptag() == self.l1_ptag(phys_addr) {
                self.invalidate_l1i_line(ic_idx);
            }
        }

        // Check L1-D for any lines from this L2 line.
        // L1-D is VIPT so we must reconstruct the virtual index from pidx + physical sub-bits.
        let l1d_lines_per_l2 = 1 << (self.l2.line_shift - self.dc.line_shift);
        let dc_base_idx = self.l2_idx_to_l1_base_idx(idx, pidx, &self.dc);
        for i in 0..l1d_lines_per_l2 {
            let dc_idx = (dc_base_idx + i) & self.dc.num_lines_mask;
            let phys_addr = phys_base + ((i as u64) << self.dc.line_shift);
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
        let phys_addr = l1d_tag_to_phys(tag, (l1_idx << self.dc.line_shift) as u64);

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
        let dc_data = self.dc.data.get();
        let l2_data = self.l2.data.get_mut();

        let l1_start_chunk = l1_idx << self.dc.chunks_per_line_shift;

        // Calculate where in the L2 line this data goes
        // L2 lines are typically larger than L1 lines
        let l2_line_base = l2_idx << self.l2.chunks_per_line_shift;
        let offset_in_l2_line = ((phys_addr & self.l2.line_mask as u64) >> 3) as usize;

        for i in 0..self.dc.chunks_per_line {
            l2_data[l2_line_base + offset_in_l2_line + i] = dc_data[l1_start_chunk + i];
        }

        // Sync l2.instrs for the updated region — read from dc_data (same values just written)
        let l2_instrs = self.l2.instrs.get_mut();
        let instrs_start = (l2_idx << self.l2.instr_shift) + offset_in_l2_line * 2;
        for i in 0..self.dc.chunks_per_line {
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
        let phys_addr = l2_tag_to_phys(tag, (idx << self.l2.line_shift) as u64);

        // First, writeback any dirty L1-D lines that are part of this L2 line.
        // L1-D is VIPT: reconstruct the virtual index using pidx from the L2 tag.
        let l1d_lines_per_l2 = 1 << (self.l2.line_shift - self.dc.line_shift);
        let dc_base_idx = self.l2_idx_to_l1_base_idx(idx, tag.pidx(), &self.dc);
        for i in 0..l1d_lines_per_l2 {
            let dc_idx = (dc_base_idx + i) & self.dc.num_lines_mask;
            let phys_addr_l1 = phys_addr + ((i as u64) << self.dc.line_shift);
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
            let l2_data = self.l2.data.get();
            let start_chunk = idx << self.l2.chunks_per_line_shift;
            println!("  L2 line data being written (16 x u64):");
            for i in 0..self.l2.chunks_per_line {
                let val = l2_data[start_chunk + i];
                println!("    [{}] addr=0x{:08x} val=0x{:016x}", i, phys_addr + ((i as u64) << 3), val);
            }
        }

        // NOTE: do NOT clear llbit here. On R4000, llbit tracks L1-D state only.
        // An L2 writeback/eviction is not a coherency action and must not break LL/SC.

        // Now write L2 data to memory
        let l2_data = self.l2.data.get();
        let start_chunk = idx << self.l2.chunks_per_line_shift;

        for i in 0..self.l2.chunks_per_line {
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
        let line_base = phys_addr & !(self.l2.line_mask as u64);

        // Fill line from memory
        let l2_data = self.l2.data.get_mut();
        let start_chunk = l2_idx << self.l2.chunks_per_line_shift;

        let instrs_start = l2_idx << self.l2.instr_shift;
        for i in 0..self.l2.chunks_per_line {
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
            for i in 0..self.l2.chunks_per_line {
                let val = l2_data[start_chunk + i];
                println!("    [{}] 0x{:016x}", i, val);
            }
        }

        true
    }

    /// Fill L1 instruction cache line
    /// Ensures data is in L2 first, then copies to L1-I
    /// For C_FILL operation, phys_addr is used for indexing
    fn fill_l1i_line(&self, index_addr: u64, phys_addr: u64) -> FillResult {
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
                return FillResult::VirtualCoherencyException;
            }
        } else {
            // L2 miss - fill from memory
            if !self.fill_l2_line(phys_addr, index_addr) {
                return FillResult::Error;
            }
        }

        // Store l2.instrs slot indices into ic.data (two u32 per u64, high=even, low=odd)
        // Align to L1I line boundary before computing L2 word offset — phys_addr may
        // point to any word within the line, not necessarily the first.
        let ic_line_base = phys_addr & !(self.ic.line_mask as u64);
        let l2_word_offset = ((ic_line_base as usize) & self.l2.line_mask) >> 2;
        let l2_instrs_base = (l2_idx << self.l2.instr_shift) + l2_word_offset;
        let ic_data = self.ic.data.get_mut();
        let ic_data_base = ic_idx * self.ic.chunks_per_line;
        for i in 0..self.ic.chunks_per_line {
            let idx0 = (l2_instrs_base + i * 2    ) as u32;
            let idx1 = (l2_instrs_base + i * 2 + 1) as u32;
            ic_data[ic_data_base + i] = ((idx0 as u64) << 32) | (idx1 as u64);
        }

        // Set tag with Valid bit
        let mut ic_tag = L1ITag::default();
        ic_tag.set_ptag(self.l1_ptag(phys_addr));
        ic_tag.set_valid(true);
        self.ic.set_tag(ic_idx, ic_tag);

        FillResult::Ok
    }

    /// Fill L1 data cache line
    /// Ensures data is in L2 first, then copies to L1-D
    fn fill_l1d_line(&self, virt_addr: u64, phys_addr: u64) -> FillResult {
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
                return FillResult::VirtualCoherencyException;
            }
        } else {
            // L2 miss - fill from memory
            if !self.fill_l2_line(phys_addr, virt_addr) {
                return FillResult::Error;
            }
        }

        // Now copy from L2 to L1-D
        // Align phys_addr to L1-D cache line boundary first
        let dc_line_base = phys_addr & !(self.dc.line_mask as u64);
        let l2_line_base = l2_idx << self.l2.chunks_per_line_shift;
        let offset_in_l2_line = ((dc_line_base & (self.l2.line_mask as u64)) >> 3) as usize;

        let l2_data = self.l2.data.get();
        let dc_data = self.dc.data.get_mut();
        let dc_start_chunk = dc_idx << self.dc.chunks_per_line_shift;

        for i in 0..self.dc.chunks_per_line {
            dc_data[dc_start_chunk + i] = l2_data[l2_line_base + offset_in_l2_line + i];
        }

        // Set tag with CleanExclusive state
        let mut dc_tag = L1DTag::default();
        dc_tag.set_ptag(self.l1_ptag(phys_addr));
        dc_tag.set_cs(L1D_CS_CLEAN_EXCLUSIVE);
        self.dc.set_tag(dc_idx, dc_tag);

        // println!("[CACHE DEBUG] fill_l1d_line: idx={}, virt_addr=0x{:016x}, phys_addr=0x{:08x}, ptag=0x{:06x}, state=CleanExclusive",
        //          dc_idx, virt_addr, phys_addr, ptag);

        #[cfg(feature = "debug_cache")]
        {
            let line_base_phys = phys_addr & !(self.dc.line_mask as u64);
            if self.is_tracking_l1d(line_base_phys) {
                let line_base_virt = virt_addr & !(self.dc.line_mask as u64);
                println!("[CACHE DEBUG] fill_l1d_line: {} line virt 0x{:016x}, phys 0x{:08x}, idx={}, ptag=0x{:06x}",
                         self.tracking_label(line_base_phys), line_base_virt, line_base_phys, dc_idx, ptag);
                println!("  L1-D line data (4 x u64):");
                for i in 0..self.dc.chunks_per_line {
                    let val = dc_data[dc_start_chunk + i];
                    println!("    [{}] 0x{:016x}", i, val);
                }
            }
        }

        FillResult::Ok
    }
}

impl MipsCache for R4000Cache {
    fn fetch(&self, virt_addr: u64, phys_addr: u64) -> FetchResult {
        #[cfg(feature = "debug_cache")]
        {
            if self.is_tracking_addr(virt_addr, phys_addr) {
                println!("[CACHE DEBUG] fetch: {} virt_addr 0x{:016x}, phys_addr 0x{:016x}",
                         self.tracking_label(phys_addr), virt_addr, phys_addr);
            } else {
                // Also track fetches that will hit the same L2 index (cache line aliasing)
                let l2_idx = self.l2.get_index(phys_addr);
                if self.is_tracking_l2_idx(l2_idx) {
                    let line_base = phys_addr & !(self.l2.line_mask as u64);
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
            match self.fill_l1i_line(virt_addr, phys_addr) {
                FillResult::Ok => {},
                FillResult::Error => return FetchResult::Error,
                FillResult::VirtualCoherencyException => return FetchResult::VirtualCoherencyException,
            }
        } else {
            #[cfg(feature = "developer")]
            self.l1i_hit_count.fetch_add(1, Ordering::Relaxed);
        }

        // Return pointer to the DecodedInstr slot in l2.instrs.
        // ic.data_as_words() gives a flat &[u32] view; XOR word index with 1 accounts
        // for the big-endian packing ((idx0<<32)|idx1) on a little-endian host.
        let word_offset = ((phys_addr as usize) & self.ic.line_mask) >> 2;
        let l2_slot_idx = self.ic.data_as_words()
            [((ic_idx << self.ic.instr_shift) + word_offset) ^ 1] as usize;
        let slot = &self.l2.instrs.get()[l2_slot_idx] as *const DecodedInstr;
        FetchResult::Hit(slot)
    }

    fn read(&self, virt_addr: u64, phys_addr: u64, size: usize) -> BusRead64 {
        #[cfg(feature = "debug_cache")]
        {
            if self.is_tracking_addr(virt_addr, phys_addr) {
                println!("[CACHE DEBUG] read: {} virt_addr 0x{:016x}, phys_addr 0x{:016x}, size {}",
                         self.tracking_label(phys_addr), virt_addr, phys_addr, size);
            } else {
                // Also track reads that will hit the same L2 index (cache line aliasing)
                let l2_idx = self.l2.get_index(phys_addr);
                if self.is_tracking_l2_idx(l2_idx) {
                    let line_base = phys_addr & !(self.l2.line_mask as u64);
                    println!("[CACHE DEBUG] read (L2 alias): idx={}, line 0x{:08x}, virt 0x{:016x}, phys 0x{:016x}, size {}",
                             l2_idx, line_base, virt_addr, phys_addr, size);
                }
            }
        }

        // Ensure line is in L1-D cache
        let dc_idx = self.dc.get_index(virt_addr);
        let dc_tag: L1DTag = self.dc.get_tag(dc_idx);
        let ptag = self.l1_ptag(phys_addr);

        // Fill if not valid or tag mismatch
        if dc_tag.cs() == L1D_CS_INVALID || dc_tag.ptag() != ptag {
            match self.fill_l1d_line(virt_addr, phys_addr) {
                FillResult::Ok => {},
                FillResult::Error => return BusRead64::err(),
                FillResult::VirtualCoherencyException => return BusRead64::vce(),
            }
        }

        // Read from L1-D cache
        let data_idx = self.dc.get_data_index(virt_addr);
        let data = match size {
            1 => self.dc.data_as_bytes()[data_idx * 8 + ((phys_addr as usize & 7) ^ 7)] as u64,
            2 => self.dc.data_as_halves()[data_idx * 4 + ((phys_addr as usize & 7) >> 1 ^ 3)] as u64,
            4 => self.dc.data_as_words()[data_idx * 2 + ((phys_addr as usize & 7) >> 2 ^ 1)] as u64,
            8 => self.dc.data.get()[data_idx],
            _ => return BusRead64::err(),
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
                    let line_base = phys_addr & !(self.l2.line_mask as u64);
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
            match self.fill_l1d_line(virt_addr, phys_addr) {
                FillResult::Ok => {},
                FillResult::Error => return BUS_ERR,
                FillResult::VirtualCoherencyException => return BUS_VCE,
            }
        }

        // Write to L1-D cache
        let data_idx = self.dc.get_data_index(virt_addr);
        let dc_data = self.dc.data.get_mut();
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
        let (cache, is_icache, is_l2) = match cache_target {
            CACH_PI => (&self.ic, true, false),
            CACH_PD => (&self.dc, false, false),
            CACH_SI | CACH_SD => (&self.l2, false, true),
            _ => return 0,
        };

        // L1I and L1D are virtually indexed; L2 is physically indexed
        let idx = if is_l2 { cache.get_index(phys_addr) } else { cache.get_index(virt_addr) };

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
            CACH_PI => (self.ic.cache_size, self.ic.line_size),
            CACH_PD => (self.dc.cache_size, self.dc.line_size),
            CACH_SI | CACH_SD => (self.l2.cache_size, self.l2.line_size),
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
        let addr_line = phys_addr & !(self.dc.line_mask as u64);
        let ll_line = ll_addr & !(self.dc.line_mask as u64);
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
                if idx >= self.ic.num_lines {
                    return format!("Index 0x{:x} out of bounds (max 0x{:x})", idx, self.ic.num_lines - 1);
                }
                let tag: L1ITag = self.ic.get_tag(idx);

                // Decoded slots live in L2 — look up via ic.data indices.
                let instrs_per_ic_line = self.ic.instrs_per_line;
                let ic_instrs = self.l2.instrs.get();

                let mut s = format!("L1-I Line 0x{:x}: Tag=0x{:06x} V={}\n  Instrs:", idx, tag.ptag(), tag.valid());
                let ic_data_words = self.ic.data_as_words();
                let ic_instrs_base = idx << self.ic.instr_shift;
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
                if idx >= self.dc.num_lines {
                    return format!("Index 0x{:x} out of bounds (max 0x{:x})", idx, self.dc.num_lines - 1);
                }
                let tag: L1DTag = self.dc.get_tag(idx);
                let cs_str = match tag.cs() {
                    L1D_CS_INVALID => "Invalid",
                    L1D_CS_SHARED => "Shared",
                    L1D_CS_CLEAN_EXCLUSIVE => "CleanExclusive",
                    L1D_CS_DIRTY_EXCLUSIVE => "DirtyExclusive",
                    _ => "Unknown",
                };

                let dc_data = self.dc.data.get();
                let start = idx << self.dc.chunks_per_line_shift;

                let mut s = format!("L1-D Line 0x{:x}: Tag=0x{:06x} CS={} ({}) D={}\n  Data:",
                    idx, tag.ptag(), tag.cs(), cs_str, tag.dirty());
                for i in 0..self.dc.chunks_per_line {
                    if i % 4 == 0 { s.push_str("\n    "); }
                    if start + i < dc_data.len() {
                        s.push_str(&format!("{:016x} ", dc_data[start + i]));
                    }
                }
                s
            }
            "l2" => {
                if idx >= self.l2.num_lines {
                    return format!("Index 0x{:x} out of bounds (max 0x{:x})", idx, self.l2.num_lines - 1);
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

                let l2_data = self.l2.data.get();
                let start = idx << self.l2.chunks_per_line_shift;

                let mut s = format!("L2 Line 0x{:x}: Tag=0x{:05x} CS={} ({})\n  Data:",
                    idx, tag.ptag(), tag.cs(), cs_str);
                for i in 0..self.l2.chunks_per_line {
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
        self.ic.tags.get_mut().fill(0);
        self.ic.data.get_mut().fill(0);
        self.dc.tags.get_mut().fill(0);
        self.dc.data.get_mut().fill(0);
        self.l2.tags.get_mut().fill(0);
        self.l2.data.get_mut().fill(0);
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
        self.ic.tags.get_mut().fill(0);
        self.ic.data.get_mut().fill(0);
        self.dc.tags.get_mut().fill(0);
        self.dc.data.get_mut().fill(0);
        self.l2.tags.get_mut().fill(0);
        self.l2.data.get_mut().fill(0);
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
    fn save_cache_inner(c: &Cache) -> (Vec<u32>, Vec<u64>) {
        (c.tags.get().clone(), c.data.get().clone())
    }

    fn load_cache_inner(c: &Cache, tags: &[u32], data: &[u64]) {
        let t = c.tags.get_mut();
        let tl = tags.len().min(t.len());
        t[..tl].copy_from_slice(&tags[..tl]);
        let d = c.data.get_mut();
        let dl = data.len().min(d.len());
        d[..dl].copy_from_slice(&data[..dl]);
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
        let mut ic_tags = vec![0u32; self.ic.num_lines];
        let mut ic_data = vec![0u64; self.ic.cache_size / 8];
        let mut dc_tags = vec![0u32; self.dc.num_lines];
        let mut dc_data = vec![0u64; self.dc.cache_size / 8];
        let mut l2_tags = vec![0u32; self.l2.num_lines];
        let mut l2_data = vec![0u64; self.l2.cache_size / 8];

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
            let l2_data_slice: Vec<u64> = self.l2.data.get().clone();
            let l2_instrs = self.l2.instrs.get_mut();
            for line in 0..self.l2.num_lines {
                let chunks_start = line << self.l2.chunks_per_line_shift;
                let instrs_start = line << self.l2.instr_shift;
                for i in 0..self.l2.chunks_per_line {
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
