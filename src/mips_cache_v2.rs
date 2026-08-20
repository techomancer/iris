// MIPS R4000 Cache Implementation - Version 2
//
// This is a complete rewrite to properly support R4000 cache semantics:
// - Unified cache object containing L1-I, L1-D, and L2
// - Proper VIPT (Virtually Indexed, Physically Tagged) support
// - R4000-compliant tag format with PState bits
// - L2 can signal back to L1 for evictions

use crate::traits::{BusRead64, BusDevice, Resettable, BUS_OK, BUS_BUSY, BUS_ERR, BUS_VCE};
use crate::snapshot::{u32_slice_to_toml, u64_slice_to_toml, load_u32_slice, load_u64_slice, get_field, toml_bool, toml_u32, hex_u32};
use crate::mips_exec::{DecodedInstr, ExecStatus, EXEC_COMPLETE, EXEC_RETRY, exec_exception_const, EXC_VCEI, EXC_VCED, EXC_IBE, FLAG_NOT_DECODED, FLAG_IMM_IS_NEXT};
use crate::devlog::{LogModule, CACHE_LOG_HIT, CACHE_LOG_MISS, CACHE_LOG_OP, devlog_is_active, devlog_mask};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::cell::{Cell, UnsafeCell};
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

// Cache sizes and line sizes — exported so mips_exec can build CP0 Config.
// R5K uses larger 2-way L1 caches; L2 geometry is the same for both CPUs.
// For R5K, IC_SIZE/DC_SIZE are the TOTAL (both ways combined) cache sizes.
// IC_WAYS/DC_WAYS control 2-way behaviour; all downstream constants are derived.

// R4400 (default) configuration
#[cfg(not(feature = "r5k"))]
pub const IC_SIZE: usize = 16 * 1024;   // 16 KB L1 instruction cache
#[cfg(not(feature = "r5k"))]
pub const IC_LINE: usize = 16;          // 16-byte lines
#[cfg(not(feature = "r5k"))]
pub const DC_SIZE: usize = 16 * 1024;   // 16 KB L1 data cache
#[cfg(not(feature = "r5k"))]
pub const DC_LINE: usize = 16;          // 16-byte lines
#[cfg(not(feature = "r5k"))]
pub const IC_WAYS: usize = 1;
#[cfg(not(feature = "r5k"))]
pub const DC_WAYS: usize = 1;

// R5000 configuration (2-way, larger L1, same L2)
#[cfg(feature = "r5k")]
pub const IC_SIZE: usize = 32 * 1024;   // 32 KB L1 instruction cache (both ways)
#[cfg(feature = "r5k")]
pub const IC_LINE: usize = 32;          // 32-byte lines
#[cfg(feature = "r5k")]
pub const DC_SIZE: usize = 32 * 1024;   // 32 KB L1 data cache (both ways)
#[cfg(feature = "r5k")]
pub const DC_LINE: usize = 32;
#[cfg(feature = "r5k")]
pub const IC_WAYS: usize = 2;
#[cfg(feature = "r5k")]
pub const DC_WAYS: usize = 2;

// Number of sets per way (= NUM_LINES / WAYS)
pub const IC_NUM_SETS: usize = IC_SIZE / IC_LINE / IC_WAYS;
pub const DC_NUM_SETS: usize = DC_SIZE / DC_LINE / DC_WAYS;

// L2 present only when r5ksc (or r4k which always has L2).
// r5ksc without r5ksc_triton = external R4600SC-style cache (same geometry, no SE/SS).
// r5ksc_triton = Triton on-die L2 (SE enable, SS size bits, SC=0 in config).
//
// When r5k is on but r5ksc is off, L2_SIZE=0 meaning no L2. L2Cache is still instantiated
// with 1 dummy line so const generics don't underflow; l2_active() returns false so it is
// never actually accessed.
#[cfg(not(all(feature = "r5k", not(feature = "r5ksc"))))]
pub const L2_SIZE: usize = 1024 * 1024; // R4K or r5k+r5ksc: 1 MB unified L2
#[cfg(all(feature = "r5k", not(feature = "r5ksc")))]
pub const L2_SIZE: usize = 0;           // R5K without secondary cache: logically absent

#[cfg(not(feature = "r5k"))]
pub const L2_LINE: usize = 128;         // R4K: 128-byte lines
#[cfg(all(feature = "r5k", not(feature = "r5ksc")))]
pub const L2_LINE: usize = 128;         // dummy (L2_SIZE=0, never accessed)
#[cfg(all(feature = "r5k", feature = "r5ksc"))]
pub const L2_LINE: usize = 32;          // R5K/Triton: 32-byte lines

// Effective L2 size for the Cache<> generic: at least LINE so NUM_LINES >= 1.
// When L2_SIZE=0 (r5k without r5ksc) this gives a 1-line dummy; l2_active() guards all access.
pub const L2_CACHE_SIZE: usize = if L2_SIZE == 0 { L2_LINE } else { L2_SIZE };

// Re-export cache operation constants for convenience
pub use crate::mips_isa::{
    CACH_PI, CACH_PD, CACH_SI, CACH_SD,
    C_IINV, C_IWBINV, C_ILT, C_IST, C_CDX,
    C_HINV, C_HWBINV, C_FILL, C_HWB, C_HSV,
};

/// Decode a raw cache_op field (5-bit: op[4:2] | target[1:0]) to a human-readable name.
/// Matches the disassembler mnemonic convention used by gas/objdump.
pub fn cache_op_name(op: u32) -> &'static str {
    let target = op & 0x3;
    let operation = op & 0x1C;
    match (operation, target) {
        (C_IINV,   CACH_PI) => "Index_Invalidate(PI)",
        (C_IWBINV, CACH_PD) => "Index_WBInvalidate(PD)",
        (C_IINV,   CACH_SI) => "Index_Invalidate(SI)",
        (C_IWBINV, CACH_SD) => "Index_WBInvalidate(SD)",
        (C_ILT,    CACH_PI) => "Index_Load_Tag(PI)",
        (C_ILT,    CACH_PD) => "Index_Load_Tag(PD)",
        (C_ILT,    CACH_SI) => "Index_Load_Tag(SI)",
        (C_ILT,    CACH_SD) => "Index_Load_Tag(SD)",
        (C_IST,    CACH_PI) => "Index_Store_Tag(PI)",
        (C_IST,    CACH_PD) => "Index_Store_Tag(PD)",
        (C_IST,    CACH_SI) => "Index_Store_Tag(SI)",
        (C_IST,    CACH_SD) => "Index_Store_Tag(SD)",
        (C_CDX,    CACH_PD) => "Create_Dirty_Excl(PD)",
        (C_CDX,    CACH_SD) => "Create_Dirty_Excl(SD)",
        (C_HINV,   CACH_PI) => "Hit_Invalidate(PI)",
        (C_HINV,   CACH_PD) => "Hit_Invalidate(PD)",
        (C_HINV,   CACH_SI) => "Hit_Invalidate(SI)",
        (C_HINV,   CACH_SD) => "Hit_Invalidate(SD)",
        (C_FILL,   CACH_PI) => "Fill(PI)",
        (C_HWBINV, CACH_PD) => "Hit_WBInvalidate(PD)",
        (C_HWBINV, CACH_SI) => "Hit_WBInvalidate(SI)",
        (C_HWBINV, CACH_SD) => "Hit_WBInvalidate(SD)",
        (C_HWB,    CACH_PD) => "Hit_WB(PD)",
        (C_HWB,    CACH_SD) => "Hit_WB(SD)",
        (C_HSV,    CACH_SI) => "Hit_Set_Virtual(SI)",
        (C_HSV,    CACH_SD) => "Hit_Set_Virtual(SD)",
        _ => "Unknown",
    }
}

// =============================================================================
// R4000 Cache Tag Format (per MIPS R4000 book)
// =============================================================================

// L1 Instruction Cache Tag — single u64 encodes both address and valid state.
//
// Encoding:
//   ptag = 0                          → invalid (Default)
//   ptag = (phys_addr & !0xFFF) | 1  → valid; line base address in bits [35:1], valid flag in bit 0
//
// Bit 0 of a page-aligned physical address is always 0, so it is free to use as a valid sentinel.
// This lets matches_phys() be a single branchless compare with no separate bool load:
//   self.ptag == (phys_addr & !0xFFF) | 1
//
// On-wire (CP0 TagLo) format:  [31:8] raw_ptag   [7:6] pstate
// Conversion: From<u32>/Into<u32> for snapshot save/load only (shifts happen there, not on hot path).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct L1ITag {
    /// Encoded tag: 0 = invalid; `(phys_line_base & !0xFFF) | 1` = valid.
    pub ptag: u64,
}

impl L1ITag {
    /// Construct a valid tag for the given physical address.
    #[inline(always)]
    pub fn valid(phys_addr: u64) -> Self { Self { ptag: (phys_addr & !0xFFF) | 1 } }

    /// True iff this tag is valid and covers the same physical line as `phys_addr`.
    /// Branchless: one AND+OR on phys_addr, one 64-bit compare. No separate bool load or branch.
    #[inline(always)]
    pub fn matches_phys(&self, phys_addr: u64) -> bool {
        self.ptag == (phys_addr & !0xFFF) | 1
    }

    /// True iff this tag is valid (for non-hot-path use only).
    #[inline(always)]
    pub fn is_valid(&self) -> bool { self.ptag & 1 != 0 }

    /// Physical line base address (bits [11:0] are zero). Only meaningful if is_valid().
    #[inline(always)]
    pub fn line_addr(&self) -> u64 { self.ptag & !0xFFF }
}

impl From<u32> for L1ITag {
    // Deserialize from CP0 TagLo wire format: raw_ptag in bits [31:8], pstate in [7:6].
    fn from(v: u32) -> Self {
        let raw_ptag = (v >> 8) & L1_PTAG_MASK;
        let valid = (v >> 6) & 3 != 0;
        let line = (raw_ptag as u64) << L1_PTAG_SHIFT;
        Self { ptag: if valid { line | 1 } else { 0 } }
    }
}
impl From<L1ITag> for u32 {
    fn from(t: L1ITag) -> Self {
        let raw_ptag = (t.line_addr() >> L1_PTAG_SHIFT) as u32 & L1_PTAG_MASK;
        (raw_ptag << 8) | (if t.is_valid() { 2 << 6 } else { 0 })
    }
}

// L1 Data Cache Tag — ptag encodes both address and validity, cs/dirty are cold-path fields.
//
// Encoding:
//   ptag = 0                          → invalid (Default)
//   ptag = (phys_addr & !0xFFF) | 1  → valid; line base in bits [35:1], valid sentinel in bit 0
//
// Bit 0 of a page-aligned address is always 0, so it is free as a valid sentinel.
// matches_phys() is a single branchless compare — no cs load, no branch.
// cs is only read on cold paths (writeback decisions, CACHE instruction, debug).
// cs and ptag validity are kept in sync: both set on fill, both cleared on invalidate.
//
//   cs    = Cache State byte: 0=Invalid, 1=Shared, 2=CleanExclusive, 3=DirtyExclusive
//   dirty = write-back bit — separate byte for branch-free set on every write
//
// On-wire format is the physical D-cache line tag, R4400 manual Figure 11-4:
//   [27] W (write-back)  [25:24] CS  [23:0] PTag
// W' and P are parity over fields we do not model, so they stay zero.
// Conversion: From<u32>/Into<u32> for snapshot save/load only.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct L1DTag {
    /// Encoded tag: 0 = invalid; `(phys_line_base & !0xFFF) | 1` = valid.
    pub ptag: u64,
    pub cs:   u8,
    pub dirty: bool,
}

impl L1DTag {
    /// Construct a valid tag for the given physical address and cache state.
    #[inline(always)]
    pub fn valid(phys_addr: u64, cs: u8, dirty: bool) -> Self {
        Self { ptag: (phys_addr & !0xFFF) | 1, cs, dirty }
    }

    /// True iff this tag is valid and covers the same physical line as `phys_addr`.
    /// Branchless: one AND+OR on phys_addr, one 64-bit compare. No cs load, no branch.
    #[inline(always)]
    pub fn matches_phys(&self, phys_addr: u64) -> bool {
        self.ptag == (phys_addr & !0xFFF) | 1
    }

    /// True iff this tag is valid (for non-hot-path use only).
    #[inline(always)]
    pub fn is_valid(&self) -> bool { self.ptag & 1 != 0 }

    /// Physical line base address (bits [11:0] are zero). Only meaningful if is_valid().
    #[inline(always)]
    pub fn line_addr(&self) -> u64 { self.ptag & !0xFFF }
}

impl From<u32> for L1DTag {
    fn from(v: u32) -> Self {
        let raw_ptag = v & L1_PTAG_MASK;
        let cs = ((v >> L1D_TAG_CS_SHIFT) & 0x3) as u8;
        let line = (raw_ptag as u64) << L1_PTAG_SHIFT;
        Self {
            ptag:  if cs != 0 { line | 1 } else { 0 },
            cs,
            dirty: (v >> L1D_TAG_W_SHIFT) & 1 != 0,
        }
    }
}
impl From<L1DTag> for u32 {
    fn from(t: L1DTag) -> Self {
        let raw_ptag = (t.line_addr() >> L1_PTAG_SHIFT) as u32 & L1_PTAG_MASK;
        raw_ptag | ((t.cs as u32 & 0x3) << L1D_TAG_CS_SHIFT)
            | ((t.dirty as u32) << L1D_TAG_W_SHIFT)
    }
}

/// Convert a tag word from before `dc_tag_format`, which put `PTag` at `[31:8]`
/// so that `W` overlapped its bit 19, physical address bit 31. A v0 word is
/// ambiguous there, `phys_bit31 | dirty`, and this resolves it in favour of
/// dirty: no cacheable mapping exists at or above `0x8000_0000`, since the
/// highest range in `device_map` is `HIMEM_END` at `0x3000_0000`.
pub fn migrate_l1d_tag_word_v0(word: u32) -> u32 {
    let raw_ptag = (word >> 8) & L1_PTAG_MASK & !(1 << 19);
    let cs = (word >> 6) & 0x3;
    let w = (word >> 27) & 1;
    raw_ptag | (cs << L1D_TAG_CS_SHIFT) | (w << L1D_TAG_W_SHIFT)
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

/// Serialized L1D tag field positions, R4400 manual Figure 11-4. PTag occupies
/// [23:0], so CS and W sit above it and nothing overlaps.
pub const L1D_TAG_CS_SHIFT: u32 = 24;
pub const L1D_TAG_W_SHIFT: u32 = 27;

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

/// Reconstruct the physical base address from an L1I cache tag and an address in the same line.
/// Uses `line_addr()` to strip the valid sentinel bit before OR-ing in the offset.
#[inline]
pub fn l1_tag_to_phys(tag: L1ITag, index_addr: u64) -> u64 {
    tag.line_addr() | (index_addr & L1_INDEX_MASK)
}

/// Same as `l1_tag_to_phys` for the L1D tag.
#[inline]
pub fn l1d_tag_to_phys(tag: L1DTag, index_addr: u64) -> u64 {
    tag.line_addr() | (index_addr & L1_INDEX_MASK)
}

/// Reconstruct the physical base address from an L2 cache tag and the address used to index
/// the cache line.  `index_addr` contributes the low 17 bits.
#[inline]
pub fn l2_tag_to_phys(tag: L2Tag, index_addr: u64) -> u64 {
    (tag.ptag() as u64) << L2_PTAG_SHIFT | (index_addr & L2_INDEX_MASK)
}

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

    /// Write SIZE bytes directly — no RMW, no mask computation.
    /// SIZE must be 1, 2, 4, or 8 (const generic).
    /// val is zero-extended; only the low SIZE*8 bits are used.
    /// phys_addr must be SIZE-aligned. Returns BUS_OK, BUS_BUSY, BUS_ERR, or BUS_VCE.
    fn write<const SIZE: usize>(&self, virt_addr: u64, phys_addr: u64, val: u64) -> u32;

    /// Arbitrary-mask doubleword write — escape hatch for SDL/SDR partial stores.
    /// phys_addr must be 8-byte aligned. val/mask are in MIPS big-endian doubleword space.
    /// Returns BUS_OK, BUS_BUSY, BUS_ERR, or BUS_VCE.
    fn write64_masked(&self, virt_addr: u64, phys_addr: u64, val: u64, mask: u64) -> u32;

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

    /// R5K/Triton: set L2 enable state from CONFIG_SE bit. No-op on R4K.
    /// When transitioning disabled→enabled, all L2 lines are invalidated.
    fn set_l2_enabled(&mut self, _enabled: bool) {}

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
            slot.flags = FLAG_NOT_DECODED;
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

    fn write<const SIZE: usize>(&self, _virt_addr: u64, phys_addr: u64, val: u64) -> u32 {
        const { assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE") };
        let addr = phys_addr as u32;
        if SIZE == 1      { self.downstream.write8(addr, val as u8) }
        else if SIZE == 2 { self.downstream.write16(addr, val as u16) }
        else if SIZE == 4 { self.downstream.write32(addr, val as u32) }
        else              { self.downstream.write64(addr, val) }
    }

    fn write64_masked(&self, _virt_addr: u64, phys_addr: u64, val: u64, mask: u64) -> u32 {
        // SDL/SDR only — read-modify-write on the downstream device
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

/// A single cache level parameterised by tag type, size, line size, and kind (Insn/Data/L2).
///
/// All geometry constants are computed at compile time from `SIZE` and `LINE`.
/// `KIND` (a `CacheKind` discriminant cast to `u8`) controls whether the L2
/// decoded-instruction array is allocated and which methods are meaningful.
///
/// - `tags`: heap `Box<[TAG]>` with TAGS entries — one typed tag per cache line
/// - `data`: heap `Box<[u64; DATA]>` — entire cache contents as u64 chunks (DATA = SIZE/8)
/// - `instrs`: L2 only — heap Vec of SIZE/4 DecodedInstr slots (6MB, contains fn ptrs)
///
/// `TAGS` and `DATA` are redundant with `SIZE`/`LINE` but required as explicit const generics
/// because stable Rust cannot use arithmetic on generic params in array length positions.
/// A single cache level parameterised by tag type, size, line size, ways, and kind.
///
/// `WAYS` = number of ways (1 for direct-mapped R4K L1/L2, 2 for R5K L1).
/// `NUM_LINES` = SIZE / LINE / WAYS = number of **sets**.
/// `get_index()` returns a set index in [0, NUM_LINES).
///
/// Tag and data arrays span all ways linearly:
///   way0 at [0..NUM_LINES), way1 at [NUM_LINES..2*NUM_LINES), etc.
/// `TAGS` must equal `NUM_LINES * WAYS`; `DATA` = `SIZE / 8` (all ways).
struct Cache<TAG, const SIZE: usize, const LINE: usize, const WAYS: usize, const KIND: u8,
             const TAGS: usize, const DATA: usize> {
    /// Heap-allocated typed tag array — TAGS entries (NUM_LINES * WAYS).
    tags:   UnsafeCell<Box<[TAG]>>,
    /// Heap-allocated data array — entire cache contents as u64 chunks (all ways).
    data:   UnsafeCell<Box<[u64; DATA]>>,
    /// L2 decoded-instruction slots (SIZE/4 entries). Empty Vec for L1-I and L1-D.
    instrs: CacheVec<DecodedInstr>,
    /// Signals the decode thread to stop (kept for Drop compatibility).
    stop:   Arc<AtomicBool>,
}

unsafe impl<TAG, const SIZE: usize, const LINE: usize, const WAYS: usize, const KIND: u8,
            const TAGS: usize, const DATA: usize> Send for Cache<TAG, SIZE, LINE, WAYS, KIND, TAGS, DATA> {}
unsafe impl<TAG, const SIZE: usize, const LINE: usize, const WAYS: usize, const KIND: u8,
            const TAGS: usize, const DATA: usize> Sync for Cache<TAG, SIZE, LINE, WAYS, KIND, TAGS, DATA> {}

impl<TAG: Default + Copy, const SIZE: usize, const LINE: usize, const WAYS: usize, const KIND: u8,
     const TAGS: usize, const DATA: usize> Cache<TAG, SIZE, LINE, WAYS, KIND, TAGS, DATA> {
    // ---- Compile-time geometry constants ----
    /// Number of sets = SIZE / LINE / WAYS.  get_index() returns values in [0, NUM_LINES).
    const NUM_LINES:             usize = SIZE / LINE / WAYS;
    const NUM_LINES_SHIFT:       u32   = ctz(Self::NUM_LINES);
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
        // Allocate L2 decoded-instruction slots for L2 caches only.
        // R5K always gets an empty vec (L1I fills its own ic_instrs from l2.data at fill time).
        let instrs: Vec<DecodedInstr> = {
            #[cfg(not(feature = "r5k"))]
            { if KIND == CacheKind::L2 as u8 { (0..SIZE / 4).map(|_| DecodedInstr::default()).collect() } else { Vec::new() } }
            #[cfg(feature = "r5k")]
            { Vec::new() }
        };
        Self {
            tags:   UnsafeCell::new(vec![TAG::default(); TAGS].into_boxed_slice()),
            // SAFETY: u64 is valid at all-zero bit patterns. Box::new_zeroed avoids
            // constructing the array on the stack before moving to the heap.
            data:   UnsafeCell::new(unsafe { Box::new_zeroed().assume_init() }),
            instrs: CacheVec::new(instrs),
            stop:   Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get set index from address.  Returns values in [0, NUM_LINES) regardless of WAYS.
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
    fn tags(&self) -> &[TAG] { unsafe { &*self.tags.get() } }
    #[inline(always)]
    fn tags_mut(&self) -> &mut [TAG] { unsafe { &mut *self.tags.get() } }
    #[inline(always)]
    fn data(&self) -> &[u64; DATA] { unsafe { &**self.data.get() } }
    #[inline(always)]
    fn data_mut(&self) -> &mut [u64; DATA] { unsafe { &mut **self.data.get() } }

    /// Read the tag at `idx`.
    #[inline(always)]
    fn get_tag(&self, idx: usize) -> TAG {
        unsafe { *self.tags().get_unchecked(idx) }
    }

    /// Write a tag to `idx`.
    #[inline(always)]
    fn set_tag(&self, idx: usize, tag: TAG) {
        unsafe { *self.tags_mut().get_unchecked_mut(idx) = tag; }
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

    #[inline(always)]
    fn data_as_words_mut(&self) -> &mut [u32] {
        let arr = self.data_mut();
        unsafe { std::slice::from_raw_parts_mut(arr.as_mut_ptr() as *mut u32, SIZE / 4) }
    }

    #[inline(always)]
    fn data_as_halves_mut(&self) -> &mut [u16] {
        let arr = self.data_mut();
        unsafe { std::slice::from_raw_parts_mut(arr.as_mut_ptr() as *mut u16, SIZE / 2) }
    }

    #[inline(always)]
    fn data_as_bytes_mut(&self) -> &mut [u8] {
        let arr = self.data_mut();
        unsafe { std::slice::from_raw_parts_mut(arr.as_mut_ptr() as *mut u8, SIZE) }
    }

    /// Read ACC bytes from the cache data array using a virtually-indexed address.
    /// ACC must be 1, 2, 4, or 8. The full cache index is derived from
    /// `virt_addr & (SIZE-1)`; the XOR corrects for big-endian packing within each u64.
    #[inline(always)]
    fn dc_read<const ACC: usize>(&self, virt_addr: u64) -> u64 {
        let masked = (virt_addr as usize) & (SIZE - 1);
        if ACC == 8 {
            self.data()[masked >> 3]
        } else if ACC == 4 {
            self.data_as_words()[(masked >> 2) ^ 1] as u64
        } else if ACC == 2 {
            self.data_as_halves()[(masked >> 1) ^ 3] as u64
        } else {
            self.data_as_bytes()[masked ^ 7] as u64
        }
    }

    /// Write ACC bytes into the cache data array using a virtually-indexed address.
    /// ACC must be 1, 2, 4, or 8. Only the low ACC*8 bits of `val` are written.
    #[inline(always)]
    fn dc_write<const ACC: usize>(&self, virt_addr: u64, val: u64) {
        let masked = (virt_addr as usize) & (SIZE - 1);
        if ACC == 8 {
            self.data_mut()[masked >> 3] = val;
        } else if ACC == 4 {
            self.data_as_words_mut()[(masked >> 2) ^ 1] = val as u32;
        } else if ACC == 2 {
            self.data_as_halves_mut()[(masked >> 1) ^ 3] = val as u16;
        } else {
            self.data_as_bytes_mut()[masked ^ 7] = val as u8;
        }
    }
}

// =============================================================================
// R4000 Cache Implementation - Full 2-level hierarchy
// =============================================================================


// Debug configuration - set to Some(phys_addr) to enable cache line tracking
#[cfg(feature = "debug_cache")]
const DEBUG_TRACK_ADDR: Option<u64> = Some(0x080165d4);
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

    // Triton only: L2 enable bit (mirrors CONFIG_SE bit 12). When false, L1 fills go
    // directly to memory and L1D writebacks bypass L2.
    #[cfg(feature = "r5ksc_triton")]
    l2_enabled: bool,

    // R5K: L1I owns its own decoded-instruction slots (L2 non-inclusive).
    // Indexed as: ic_instrs[way * IC_NUM_SETS * INSTRS_PER_LINE + set_idx * INSTRS_PER_LINE + word]
    // R5K has 2 ways; total: 2 × IC_NUM_SETS sets × (IC_LINE/4) instrs/line.
    #[cfg(feature = "r5k")]
    ic_instrs: CacheVec<DecodedInstr>,

    // R5K only: LRU bit per set for L1I and L1D, packed as a bitmap.
    // Bit N = 0: way0 is LRU (fill way0 next); bit N = 1: way1 is LRU.
    // 512 sets → 8 × u64, fits in one cache line.
    #[cfg(feature = "r5k")]
    ic_lru: UnsafeCell<Box<[u64]>>,
    #[cfg(feature = "r5k")]
    dc_lru: UnsafeCell<Box<[u64]>>,

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
// TAGS = (SIZE/LINE) = NUM_SETS * WAYS (tag array spans all ways).
// DATA = SIZE/8 (all ways combined). ICache DATA=0: fetch() indexes l2.instrs or ic_instrs.
// get_index() returns a set index in [0, NUM_LINES); way1 at set_idx + NUM_LINES.
type ICache  = Cache<L1ITag, IC_SIZE, IC_LINE, IC_WAYS, { CacheKind::Insn as u8 }, { IC_SIZE / IC_LINE }, 0>;
type DCache  = Cache<L1DTag, DC_SIZE, DC_LINE, DC_WAYS, { CacheKind::Data as u8 }, { DC_SIZE / DC_LINE }, { DC_SIZE / 8 }>;
type L2Cache = Cache<L2Tag,  L2_CACHE_SIZE, L2_LINE, 1, { CacheKind::L2   as u8 }, { L2_CACHE_SIZE / L2_LINE }, { L2_CACHE_SIZE / 8 }>;

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
            #[cfg(feature = "r5ksc_triton")]
            l2_enabled: false, // starts disabled; PROM enables via CONFIG_SE
            // R5K: 2-way ic_instrs (2 × IC_NUM_SETS × IC_LINE/4 words).
            #[cfg(feature = "r5k")]
            ic_instrs: CacheVec::new(
                (0..IC_WAYS * IC_NUM_SETS * (IC_LINE / 4))
                    .map(|_| DecodedInstr::default()).collect()
            ),
            #[cfg(feature = "r5k")]
            ic_lru: UnsafeCell::new(vec![0u64; IC_NUM_SETS.div_ceil(64)].into_boxed_slice()),
            #[cfg(feature = "r5k")]
            dc_lru: UnsafeCell::new(vec![0u64; DC_NUM_SETS.div_ceil(64)].into_boxed_slice()),
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

    /// Returns whether L2 is currently usable.
    /// - R4K / r5ksc (external): always true when L2_SIZE > 0.
    /// - r5ksc_triton: gated by CONFIG_SE (l2_enabled field).
    /// - r5k without r5ksc: always false (no L2).
    #[inline]
    fn l2_active(&self) -> bool {
        if L2_SIZE == 0 { return false; }
        #[cfg(feature = "r5ksc_triton")]
        { self.l2_enabled }
        #[cfg(not(feature = "r5ksc_triton"))]
        { true }
    }

    /// Triton only: set L2 enable state from CONFIG_SE. On off→on transition, invalidate
    /// all L2 lines so stale data from before the disable window isn't used.
    #[cfg(feature = "r5ksc_triton")]
    pub fn set_l2_enabled(&mut self, enabled: bool) {
        let was = self.l2_enabled;
        self.l2_enabled = enabled;
        if enabled && !was {
            for idx in 0..L2Cache::NUM_LINES {
                self.l2.set_tag(idx, L2Tag::default());
            }
        }
    }

    /// Extract physical tag bits [35:17] from physical address for L2 cache
    #[inline]
    fn l2_ptag(&self, phys_addr: u64) -> u32 {
        ((phys_addr >> L2_PTAG_SHIFT) & L2_PTAG_MASK as u64) as u32
    }

    /// Extract virtual index bits [14:12] for L2 PIdx field
    #[cfg(feature = "r5k")]
    #[inline(always)]
    unsafe fn lru_get(bm: *const Box<[u64]>, set: usize) -> bool {
        let slice: &[u64] = &*bm;
        (*slice.get_unchecked(set >> 6) >> (set & 63)) & 1 != 0
    }

    /// Set or clear LRU bit for `set` in a packed u64 bitmap.
    #[cfg(feature = "r5k")]
    #[inline(always)]
    unsafe fn lru_set(bm: *mut Box<[u64]>, set: usize, val: usize) {
        let slice: &mut [u64] = &mut *bm;
        let word = slice.get_unchecked_mut(set >> 6);
        let mask = 1u64 << (set & 63);
        *word = (*word & !mask) | ((val as u64) << (set & 63));
    }

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
    fn l2_idx_to_l1_base_idx<TAG: Default + Copy, const L1_SIZE: usize, const L1_LINE: usize, const L1_WAYS: usize, const L1_KIND: u8, const L1_TAGS: usize, const L1_DATA: usize>(
        &self, l2_idx: usize, pidx: u32, _l1: &Cache<TAG, L1_SIZE, L1_LINE, L1_WAYS, L1_KIND, L1_TAGS, L1_DATA>
    ) -> usize {
        // Physical bits of the L2 line start address that are below bit 12 (page boundary)
        // These bits are the same in VA and PA, so we can derive them from the L2 index.
        let phys_sub_bits = (l2_idx << L2Cache::LINE_SHIFT as usize) & 0xFFF;
        // Reconstruct the virtual address bits used for L1 indexing
        let virt_index_bits = ((pidx as usize) << L2_PIDX_VADDR_SHIFT as usize) | phys_sub_bits;
        (virt_index_bits >> Cache::<TAG, L1_SIZE, L1_LINE, L1_WAYS, L1_KIND, L1_TAGS, L1_DATA>::LINE_SHIFT as usize)
            & Cache::<TAG, L1_SIZE, L1_LINE, L1_WAYS, L1_KIND, L1_TAGS, L1_DATA>::NUM_LINES_MASK
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
    fn invalidate_l1i_line(&self, idx: usize, cascade: bool) {
        let tag: L1ITag = self.ic.get_tag(idx);

        #[cfg(feature = "debug_cache")]
        if tag.is_valid() {
            let phys_addr = l1_tag_to_phys(tag, (idx << ICache::LINE_SHIFT) as u64);
            if self.is_tracking_l1d(phys_addr) {
                println!("[CACHE DEBUG] invalidate_l1i_line: {} idx=0x{:x}, phys_addr=0x{:08x}, ptag=0x{:010x}",
                         self.tracking_label(phys_addr), idx, phys_addr, tag.line_addr());
            }
        }

        if cascade && tag.is_valid() {
            let phys_addr = l1_tag_to_phys(tag, (idx << ICache::LINE_SHIFT) as u64);
            self.invalidate_l2_line_phys(phys_addr);
        }

        self.ic.set_tag(idx, L1ITag::default());
    }

    /// Invalidate L1 data cache line by index.
    /// `coherent` = true for software-initiated CACHE ops (may clear llbit);
    /// false for hardware-induced evictions (fills, L2 cascades) which must not clear llbit.
    fn invalidate_l1d_line(&self, idx: usize, coherent: bool, cascade: bool) {
        let tag: L1DTag = self.dc.get_tag(idx);

        #[cfg(feature = "debug_cache")]
        if self.is_tracking_l1d_idx(idx) {
            if tag.cs != L1D_CS_INVALID as u8 {
                let phys_addr = l1d_tag_to_phys(tag, (idx << DCache::LINE_SHIFT) as u64);
                println!("[CACHE DEBUG] invalidate_l1d_line: {} idx=0x{:x}, phys_addr=0x{:08x}, ptag=0x{:010x}, cs={}, coherent={}",
                         self.tracking_label(phys_addr), idx, phys_addr, tag.line_addr(), tag.cs, coherent);
            } else {
                println!("[CACHE DEBUG] invalidate_l1d_line: idx=0x{:x} (already invalid)", idx);
            }
        }

        // Only clear llbit for software-initiated coherency invalidations, not hardware fills.
        // On a uniprocessor R4000 there are no external snoops; llbit survives capacity evictions.
        if coherent && tag.cs != L1D_CS_INVALID as u8 {
            let phys_addr = l1d_tag_to_phys(tag, (idx << DCache::LINE_SHIFT) as u64);
            self.check_and_clear_llbit(phys_addr, DCache::LINE_MASK);
        }

        if cascade && tag.cs != L1D_CS_INVALID as u8 {
            let phys_addr = l1d_tag_to_phys(tag, (idx << DCache::LINE_SHIFT) as u64);
            self.invalidate_l2_line_phys(phys_addr);
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
                println!("[CACHE DEBUG] invalidate_l2_line: {} idx=0x{:x}, phys_base=0x{:08x}, ptag=0x{:05x}, cs={}",
                         self.tracking_label_l2_idx(idx), idx, phys_base, l2_tag.ptag(), l2_tag.cs());
            } else {
                println!("[CACHE DEBUG] invalidate_l2_line: {} idx=0x{:x} (already invalid)",
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

        // R4K inclusive policy: cascade L2 eviction to L1.
        // R5K caches are non-inclusive — L2 evictions do not affect L1.
        #[cfg(not(feature = "r5k"))]
        {
            // Check L1-I for any lines from this L2 line.
            // L1-I is VIPT so we must reconstruct the virtual index from pidx + physical sub-bits.
            let pidx = l2_tag.pidx();
            let l1i_lines_per_l2 = 1 << (L2Cache::LINE_SHIFT - ICache::LINE_SHIFT);
            let ic_base_idx = self.l2_idx_to_l1_base_idx(idx, pidx, &self.ic);
            for i in 0..l1i_lines_per_l2 {
                let ic_idx = (ic_base_idx + i) & ICache::NUM_LINES_MASK;
                let phys_addr = phys_base + ((i as u64) << ICache::LINE_SHIFT);
                let ic_tag: L1ITag = self.ic.get_tag(ic_idx);
                if ic_tag.matches_phys(phys_addr) {
                    self.invalidate_l1i_line(ic_idx, false);
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

                if dc_tag.matches_phys(phys_addr) {
                    self.invalidate_l1d_line(dc_idx, false, false); // hardware cascade, not coherent
                }
            }
        }

        // Finally invalidate the L2 line itself
        self.l2.set_tag(idx, L2Tag::default());
    }

    /// Invalidate L2 line by physical address, if present and tag matches.
    fn invalidate_l2_line_phys(&self, phys_addr: u64) {
        let l2_idx = self.l2.get_index(phys_addr);
        let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
        let l2_ptag = self.l2_ptag(phys_addr);
        if l2_tag.cs() != L2_CS_INVALID && l2_tag.ptag() == l2_ptag {
            self.invalidate_l2_line(l2_idx);
        }
    }

    /// Writeback L2 line to memory by physical address, if present, dirty, and tag matches.
    fn writeback_l2_line_phys(&self, phys_addr: u64) {
        let l2_idx = self.l2.get_index(phys_addr);
        let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
        let l2_ptag = self.l2_ptag(phys_addr);
        if l2_tag.cs() != L2_CS_INVALID && l2_tag.ptag() == l2_ptag {
            self.writeback_l2_line(l2_idx);
        }
    }

    /// Triton C_INVALL: invalidate every line in the L2 cache.
    /// The address operand is ignored. Used after enabling L2 via CONFIG_SE,
    /// and by the OS to ensure L2 coherency before DMA or cache mode changes.
    #[cfg(feature = "r5ksc_triton")]
    fn invall_l2(&self) {
        for i in 0..L2Cache::NUM_LINES {
            self.invalidate_l2_line(i);
        }
    }

    /// Triton C_INVPAGE: invalidate all L2 lines within the 4KB-aligned page
    /// containing phys_addr. Used for TLB shootdown and page migration.
    #[cfg(feature = "r5ksc_triton")]
    fn invpage_l2(&self, phys_addr: u64) {
        const PAGE_SIZE: u64 = 4096;
        let page_base = phys_addr & !(PAGE_SIZE - 1);
        let page_end  = page_base + PAGE_SIZE;
        let mut addr = page_base;
        while addr < page_end {
            self.invalidate_l2_line_phys(addr);
            addr += L2_LINE as u64;
        }
    }

    /// Write back a dirty L1 data cache line to L2
    /// Since the cache is inclusive, the line must exist in L2
    /// Returns true if writeback was successful
    fn writeback_l1d_line(&self, l1_idx: usize, cascade: bool) -> bool {
        let tag: L1DTag = self.dc.get_tag(l1_idx);

        // Check if line is dirty
        if !tag.dirty {
            return true; // Nothing to write back
        }

        // Reconstruct physical address from tag
        let phys_addr = l1d_tag_to_phys(tag, (l1_idx << DCache::LINE_SHIFT) as u64);

        #[cfg(feature = "debug_cache")]
        {
            let l2_idx_check = self.l2.get_index(phys_addr);
            if self.is_tracking_l1d(phys_addr) || self.is_tracking_l2_idx(l2_idx_check) {
                println!("[CACHE DEBUG] writeback_l1d_line: {} l1_idx=0x{:x}, phys_addr=0x{:08x}, ptag=0x{:010x}, DIRTY → L2",
                         self.tracking_label(phys_addr), l1_idx, phys_addr, tag.line_addr());
            }
        }

        // Find the line in L2 using physical address
        let l2_idx = self.l2.get_index(phys_addr);
        let mut l2_tag: L2Tag = self.l2.get_tag(l2_idx);
        let l2_ptag = self.l2_ptag(phys_addr);

        // R5K: non-inclusive L2 or no L2 — line may be absent or L2 disabled.
        // In all these cases write dirty data directly to memory.
        // R4K always holds the line in its inclusive L2, so this branch is r5k-only.
        #[cfg(feature = "r5k")]
        if !self.l2_active() || l2_tag.cs() == L2_CS_INVALID || l2_tag.ptag() != l2_ptag {
            let dc_data = self.dc.data();
            let l1_start_chunk = l1_idx << DCache::CHUNKS_PER_LINE_SHIFT;
            let line_base = phys_addr & !(DCache::LINE_MASK as u64);
            let src = &dc_data[l1_start_chunk..l1_start_chunk + DCache::CHUNKS_PER_LINE];
            self.downstream.write_block(line_base as u32, src);
            let mut dc_tag: L1DTag = self.dc.get_tag(l1_idx);
            dc_tag.dirty = false;
            if dc_tag.cs == L1D_CS_DIRTY_EXCLUSIVE as u8 { dc_tag.cs = L1D_CS_CLEAN_EXCLUSIVE as u8; }
            self.dc.set_tag(l1_idx, dc_tag);
            return true;
        }

        // R4K (inclusive): L2 must always hold the line — fail if not.
        #[cfg(not(feature = "r5k"))]
        if l2_tag.ptag() != l2_ptag {
            return false; // shouldn't happen on inclusive R4K
        }

        // L2 has the line: write data from L1-D into L2.
        let dc_data = self.dc.data();
        let l2_data = self.l2.data_mut();

        let l1_start_chunk = l1_idx << DCache::CHUNKS_PER_LINE_SHIFT;

        let l2_line_base = l2_idx << L2Cache::CHUNKS_PER_LINE_SHIFT;
        let offset_in_l2_line = ((phys_addr & L2Cache::LINE_MASK as u64) >> 3) as usize;

        for i in 0..DCache::CHUNKS_PER_LINE {
            l2_data[l2_line_base + offset_in_l2_line + i] = dc_data[l1_start_chunk + i];
        }

        #[cfg(feature = "debug_cache")]
        {
            if self.is_tracking_l1d(phys_addr) || self.is_tracking_l2_idx(l2_idx) {
                println!("[CACHE DEBUG] writeback_l1d_line: wrote {} chunks to L2 idx=0x{:x} offset=0x{:x}",
                         DCache::CHUNKS_PER_LINE, l2_idx, offset_in_l2_line);
                for i in 0..DCache::CHUNKS_PER_LINE {
                    println!("    [{}] addr=0x{:08x} val=0x{:016x}",
                             i, phys_addr + ((i as u64) << 3), dc_data[l1_start_chunk + i]);
                }
            }
        }

        // R4K: sync l2.instrs for the updated region so fetch() sees fresh instruction words.
        // R5K: l2.instrs is empty; ic_instrs will be re-filled from l2.data on next L1I miss.
        // Also recomputes delay-slot fusion lookahead (FLAG_IMM_IS_NEXT) inline,
        // where raw values are already hot — see fill_l2_line for why this must
        // not happen in fetch()'s hot path instead. r0/r1 (one chunk) are known
        // together, so r0's neighbor (r1) resolves immediately; r1's neighbor is
        // next iteration's r0, so it's finished one iteration late via `prev_s1`.
        #[cfg(not(feature = "r5k"))]
        {
            let l2_instrs = self.l2.instrs.get_mut();
            let instrs_start = (l2_idx << L2Cache::INSTR_SHIFT) + offset_in_l2_line * 2;
            #[cfg(feature = "opcodefusion")]
            let dline_base = phys_addr & !(DCache::LINE_MASK as u64);
            #[cfg(feature = "opcodefusion")]
            let mut prev_s1: Option<usize> = None;
            for i in 0..DCache::CHUNKS_PER_LINE {
                let chunk = dc_data[l1_start_chunk + i];
                let r0 = (chunk >> 32) as u32;
                let r1 = chunk as u32;
                let idx0 = instrs_start + i * 2;
                let idx1 = idx0 + 1;
                #[cfg(feature = "opcodefusion")]
                if let Some(prev_idx) = prev_s1.take() {
                    let prev_phys = (dline_base as usize) + (i * 2 - 1) * 4;
                    if prev_phys & ICache::LINE_MASK != ICache::LINE_MASK - 3 {
                        let prev = &mut l2_instrs[prev_idx];
                        prev.imm = r0;
                        prev.flags |= FLAG_IMM_IS_NEXT;
                    }
                }
                let s0 = &mut l2_instrs[idx0];
                s0.flags = FLAG_NOT_DECODED;
                s0.raw = r0;
                #[cfg(feature = "opcodefusion")]
                {
                    let word0_phys = (dline_base as usize) + (i * 2) * 4;
                    if word0_phys & ICache::LINE_MASK != ICache::LINE_MASK - 3 {
                        s0.imm = r1;
                        s0.flags |= FLAG_IMM_IS_NEXT;
                    }
                }
                let s1 = &mut l2_instrs[idx1];
                s1.flags = FLAG_NOT_DECODED;
                s1.raw = r1;
                #[cfg(feature = "opcodefusion")]
                { prev_s1 = Some(idx1); }
            }
        }

        // Mark L2 line as dirty
        let new_cs = match l2_tag.cs() {
            L2_CS_CLEAN_EXCLUSIVE => L2_CS_DIRTY_EXCLUSIVE,
            L2_CS_SHARED => L2_CS_DIRTY_SHARED,
            cs => cs, // Already dirty or invalid
        };
        l2_tag.set_cs(new_cs);
        self.l2.set_tag(l2_idx, l2_tag);

        // Clear dirty bit and demote cs to CleanExclusive after successful writeback.
        // DirtyExclusive→CleanExclusive; Shared stays Shared (no promotion occurred).
        let mut dc_tag: L1DTag = self.dc.get_tag(l1_idx);
        dc_tag.dirty = false;
        if dc_tag.cs == L1D_CS_DIRTY_EXCLUSIVE as u8 { dc_tag.cs = L1D_CS_CLEAN_EXCLUSIVE as u8; }
        self.dc.set_tag(l1_idx, dc_tag);

        if cascade {
            let phys_addr = l1d_tag_to_phys(tag, (l1_idx << DCache::LINE_SHIFT) as u64);
            self.writeback_l2_line_phys(phys_addr);
        }

        true
    }

    /// Write back a dirty L2 cache line to memory
    /// Also writes back any dirty L1-D lines that are part of this L2 line
    /// Returns true if writeback was successful
    fn writeback_l2_line(&self, idx: usize) -> bool {
        let tag: L2Tag = self.l2.get_tag(idx);

        // Reconstruct physical address from tag
        let phys_addr = l2_tag_to_phys(tag, (idx << L2Cache::LINE_SHIFT) as u64);

        // R4K: first flush dirty L1-D sub-lines into L2, so L2 has the authoritative data.
        // R5K: L1 and L2 are non-inclusive; dirty L1D lines hold the latest data and will
        // write back independently. We skip the pre-flush here to avoid complexity; if a
        // dirty L1D line exists when we write back L2 to memory we may lose data, but the
        // CACHE-op sequence IRIX uses (C_IWBINV L1D then C_IWBINV L2) ensures L1D is clean
        // before L2 is touched. For safety we scan and flush anyway on R5K too.
        #[cfg(not(feature = "r5k"))]
        {
            let l1d_lines_per_l2 = 1 << (L2Cache::LINE_SHIFT - DCache::LINE_SHIFT);
            let dc_base_idx = self.l2_idx_to_l1_base_idx(idx, tag.pidx(), &self.dc);
            for i in 0..l1d_lines_per_l2 {
                let dc_idx = (dc_base_idx + i) & DCache::NUM_LINES_MASK;
                let phys_addr_l1 = phys_addr + ((i as u64) << DCache::LINE_SHIFT);
                let dc_tag: L1DTag = self.dc.get_tag(dc_idx);
                if dc_tag.matches_phys(phys_addr_l1) {
                    self.writeback_l1d_line(dc_idx, false);
                }
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
            println!("[CACHE DEBUG] writeback_l2_line: {} idx=0x{:x}, phys_addr=0x{:08x}, ptag=0x{:05x}, cs={}, WRITING TO MEMORY",
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
        let src = &l2_data[start_chunk..start_chunk + L2Cache::CHUNKS_PER_LINE];
        if self.downstream.write_block(phys_addr as u32, src) != BUS_OK {
            return false;
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

        // INVARIANT: l2.data is always accessed as u64 chunks (never as u32 words).
        // R4K: l2.instrs[n] mirrors the n-th instruction word; fetch() indexes it directly.
        // R5K: l2.instrs is empty — fill_l1i_line reads raw words from l2.data instead.
        // Do not add data_as_words() accessors on L2 or fetch indexing will silently break.
        #[cfg(not(feature = "r5k"))]
        let instrs_start = l2_idx << L2Cache::INSTR_SHIFT;
        // Delay-slot fusion lookahead (FLAG_IMM_IS_NEXT) is computed inline below,
        // where raw values are already hot, instead of in fetch()'s hot path —
        // fetch() must stay a plain shared-borrow read for the common case (cache
        // hit on an already-decoded instruction); taking get_mut() unconditionally
        // on every fetch() call regressed whetstone/dhrystone measurably even with
        // the actual write gated behind a branch, almost certainly by defeating
        // aliasing assumptions on the hottest path in the interpreter. r0/r1 (one
        // chunk) are known together, so r0's neighbor (r1) resolves immediately;
        // r1's neighbor is next iteration's r0, so it's finished one iteration
        // late via `prev_s1`. An L2 line always contains a whole number of L1I
        // lines, so the neighbor is only out-of-bounds at each L1I sub-line's
        // last word (checked via physical address, via `prev_s1`/`fuse_pair!`).
        #[cfg(all(not(feature = "r5k"), feature = "opcodefusion"))]
        macro_rules! fuse_pair {
            ($l2_instrs:expr, $prev_s1:expr, $i:expr, $idx0:expr, $idx1:expr, $r0:expr, $r1:expr) => {
                if let Some(prev_idx) = $prev_s1.take() {
                    let prev_phys = (line_base as usize) + ($i * 2 - 1) * 4;
                    if prev_phys & ICache::LINE_MASK != ICache::LINE_MASK - 3 {
                        let prev = &mut $l2_instrs[prev_idx];
                        prev.imm = $r0;
                        prev.flags |= FLAG_IMM_IS_NEXT;
                    }
                }
                let word0_phys = (line_base as usize) + ($i * 2) * 4;
                if word0_phys & ICache::LINE_MASK != ICache::LINE_MASK - 3 {
                    let s0 = &mut $l2_instrs[$idx0];
                    s0.imm = $r1;
                    s0.flags |= FLAG_IMM_IS_NEXT;
                }
                $prev_s1 = Some($idx1);
            };
        }
        #[cfg(all(not(feature = "r5k"), not(feature = "opcodefusion")))]
        macro_rules! fuse_pair {
            ($l2_instrs:expr, $prev_s1:expr, $i:expr, $idx0:expr, $idx1:expr, $r0:expr, $r1:expr) => {};
        }
        if let Some(src) = self.downstream.mem_ptr(line_base as u32) {
            // Fast path: single pass over source — rotate into l2.data and fill l2.instrs.
            #[cfg(not(feature = "r5k"))]
            let l2_instrs = self.l2.instrs.get_mut();
            // Only fuse_pair! mutates prev_s1, and it is a no-op without opcodefusion.
            #[cfg(not(feature = "r5k"))]
            #[cfg_attr(not(feature = "opcodefusion"), allow(unused_mut))]
            let mut prev_s1: Option<usize> = None;
            for i in 0..L2Cache::CHUNKS_PER_LINE {
                let val = unsafe { (*src.add(i)).rotate_left(32) };
                l2_data[start_chunk + i] = val;
                #[cfg(not(feature = "r5k"))]
                {
                    let r0 = (val >> 32) as u32;
                    let r1 = val as u32;
                    let idx0 = instrs_start + i * 2;
                    let idx1 = idx0 + 1;
                    let s0 = &mut l2_instrs[idx0];
                    s0.flags = FLAG_NOT_DECODED;
                    s0.raw = r0;
                    let s1 = &mut l2_instrs[idx1];
                    s1.flags = FLAG_NOT_DECODED;
                    s1.raw = r1;
                    fuse_pair!(l2_instrs, prev_s1, i, idx0, idx1, r0, r1);
                }
            }
        } else {
            let dest = &mut l2_data[start_chunk..start_chunk + L2Cache::CHUNKS_PER_LINE];
            let s = self.downstream.read_block(line_base as u32, dest);
            if s != crate::traits::BUS_OK { return false; }
            #[cfg(not(feature = "r5k"))]
            {
                let l2_instrs = self.l2.instrs.get_mut();
                #[cfg_attr(not(feature = "opcodefusion"), allow(unused_mut))]
                let mut prev_s1: Option<usize> = None;
                for i in 0..L2Cache::CHUNKS_PER_LINE {
                    let val = dest[i];
                    let r0 = (val >> 32) as u32;
                    let r1 = val as u32;
                    let idx0 = instrs_start + i * 2;
                    let idx1 = idx0 + 1;
                    let s0 = &mut l2_instrs[idx0];
                    s0.flags = FLAG_NOT_DECODED;
                    s0.raw = r0;
                    let s1 = &mut l2_instrs[idx1];
                    s1.flags = FLAG_NOT_DECODED;
                    s1.raw = r1;
                    fuse_pair!(l2_instrs, prev_s1, i, idx0, idx1, r0, r1);
                }
            }
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
            println!("[CACHE DEBUG] fill_l2_line: {} line 0x{:08x}, idx=0x{:x}, phys_addr=0x{:08x}, ptag=0x{:05x}, pidx={}",
                     self.tracking_label_l2_idx(l2_idx), line_base, l2_idx, phys_addr, ptag, pidx);
            println!("  L2 line data (16 x u64):");
            for i in 0..L2Cache::CHUNKS_PER_LINE {
                let val = l2_data[start_chunk + i];
                println!("    [{}] 0x{:016x}", i, val);
            }
        }

        true
    }

    /// Fill L1 instruction cache line.
    /// Ensures data is in L2 first, then populates L1-I.
    /// For C_FILL operation, phys_addr is used for indexing.
    /// Returns 0 = way0 ok, 1 = way1 ok (R5K), >1 = exception status (EXC_VCEI or EXC_IBE).
    fn fill_l1i_line(&self, index_addr: u64, phys_addr: u64) -> u32 {
        let set = self.ic.get_index(index_addr);

        // R5K: 2-way — pick LRU way; R4K: single way, eidx == set.
        #[cfg(not(feature = "r5k"))]
        let ic_eidx = set;
        #[cfg(feature = "r5k")]
        let ic_eidx = set | (unsafe { Self::lru_get(self.ic_lru.get(), set) } as usize) << ICache::NUM_LINES_SHIFT;

        // Invalidate victim slot unconditionally — clears tag before any early return.
        self.invalidate_l1i_line(ic_eidx, false);

        // Ensure L2 has the line (skipped when L2 is disabled — read directly from memory).
        if self.l2_active() {
            let l2_idx = self.l2.get_index(phys_addr);
            let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
            let l2_ptag = self.l2_ptag(phys_addr);
            let l2_hit = l2_tag.cs() != L2_CS_INVALID && l2_tag.ptag() == l2_ptag;

            if l2_hit {
                // R4K only: check for Virtual Coherency Exception (VCEI).
                // R5K dropped VCE — no pidx tracking needed.
                #[cfg(not(feature = "r5k"))]
                if self.pidx(index_addr) != l2_tag.pidx() {
                    return exec_exception_const(EXC_VCEI);
                }
            } else {
                // L2 miss — fill from memory into L2 first.
                #[cfg(not(feature = "lightning"))]
                if devlog_is_active(LogModule::L2c) && devlog_mask(LogModule::L2c) & CACHE_LOG_MISS != 0 {
                    crate::dlog!(LogModule::L2c, "fill virt={:#x} phys={:#x} idx={}", index_addr, phys_addr, self.l2.get_index(phys_addr));
                }
                if !self.fill_l2_line(phys_addr, index_addr) {
                    return exec_exception_const(EXC_IBE);
                }
            }
        }

        #[cfg(not(feature = "lightning"))]
        if devlog_is_active(LogModule::L1i) && devlog_mask(LogModule::L1i) & CACHE_LOG_MISS != 0 {
            crate::dlog!(LogModule::L1i, "fill virt={:#x} phys={:#x} eidx={}", index_addr, phys_addr, ic_eidx);
        }

        // R5K: populate ic_instrs for this way's slot.
        // Source is l2.data when L2 is active, or memory directly when L2 is disabled.
        #[cfg(feature = "r5k")]
        {
            let ic_slot_base = ic_eidx << ICache::INSTR_SHIFT;
            let ic_instrs = self.ic_instrs.get_mut();
            if self.l2_active() {
                let l2_sub_offset = ((phys_addr as usize) & (L2Cache::LINE_MASK & !ICache::LINE_MASK)) >> 3;
                let l2_chunk_base = (self.l2.get_index(phys_addr) << L2Cache::CHUNKS_PER_LINE_SHIFT)
                    + l2_sub_offset;
                let l2_data = self.l2.data();
                let src = unsafe { l2_data.as_ptr().add(l2_chunk_base) };
                for i in 0..ICache::INSTRS_PER_LINE / 2 {
                    let chunk = unsafe { *src.add(i) };
                    let w0 = (chunk >> 32) as u32;
                    let w1 = chunk as u32;
                    let d0 = &mut ic_instrs[ic_slot_base + i * 2];
                    d0.flags = FLAG_NOT_DECODED;
                    d0.raw = w0;
                    let d1 = &mut ic_instrs[ic_slot_base + i * 2 + 1];
                    d1.flags = FLAG_NOT_DECODED;
                    d1.raw = w1;
                }
            } else {
                // L2 disabled: read directly from memory.
                let line_base = phys_addr & !(ICache::LINE_MASK as u64);
                if let Some(src) = self.downstream.mem_ptr(line_base as u32) {
                    // Fast path: read word pairs directly from backing store.
                    // mem_ptr's raw u64s are host-native pairs of little-endian-loaded
                    // words; rotate_left(32) puts them in MIPS big-endian word order
                    // (high word = first instr), matching read_block's behavior.
                    for i in 0..ICache::INSTRS_PER_LINE / 2 {
                        let chunk = unsafe { (*src.add(i)).rotate_left(32) };
                        let w0 = (chunk >> 32) as u32;
                        let w1 = chunk as u32;
                        let d0 = &mut ic_instrs[ic_slot_base + i * 2];
                        d0.flags = FLAG_NOT_DECODED;
                        d0.raw = w0;
                        let d1 = &mut ic_instrs[ic_slot_base + i * 2 + 1];
                        d1.flags = FLAG_NOT_DECODED;
                        d1.raw = w1;
                    }
                } else {
                    for i in 0..ICache::INSTRS_PER_LINE {
                        let word_addr = (line_base + (i as u64) * 4) as u32;
                        let r = self.downstream.read32(word_addr);
                        let w = if r.is_ok() { r.data } else { 0 };
                        let d = &mut ic_instrs[ic_slot_base + i];
                        d.flags = FLAG_NOT_DECODED;
                        d.raw = w;
                    }
                }
            }
            // Flip LRU: just-filled way becomes MRU.
            let way = ic_eidx >> ICache::NUM_LINES_SHIFT;
            unsafe { Self::lru_set(self.ic_lru.get(), set, way ^ 1); }
        }

        self.ic.set_tag(ic_eidx, L1ITag::valid(phys_addr));

        #[cfg(feature = "debug_cache")]
        if self.is_tracking_addr(index_addr, phys_addr) || self.is_tracking_l2_idx(self.l2.get_index(phys_addr)) {
            let way = ic_eidx >> ICache::NUM_LINES_SHIFT;
            let set = ic_eidx & ICache::NUM_LINES_MASK;
            println!("[CACHE DEBUG] fill_l1i_line: {} virt 0x{:016x} phys 0x{:016x} → L1I eidx=0x{:x} way={} set=0x{:x}",
                self.tracking_label(phys_addr), index_addr, phys_addr, ic_eidx, way, set);
            #[cfg(feature = "r5k")]
            {
                let ic_instrs = self.ic_instrs.get();
                let slot_base = ic_eidx << ICache::INSTR_SHIFT;
                print!("  ic_instrs:");
                for i in 0..ICache::INSTRS_PER_LINE {
                    if i % 4 == 0 { print!("\n    "); }
                    print!("{:08x} ", ic_instrs[slot_base + i].raw);
                }
                println!();
            }
        }

        (ic_eidx >> ICache::NUM_LINES_SHIFT) as u32
    }

    /// Fill L1 data cache line. Ensures data is in L2 first, then copies to L1-D.
    ///
    /// Returns a `u32` with the same encoding as `ensure_l1d_line`:
    ///   0  = filled into way0 (BUS_OK)
    ///   1  = filled into way1 (BUS_OK, R5K only)
    ///   >1 = BUS_VCE or BUS_ERR
    fn fill_l1d_line(&self, virt_addr: u64, phys_addr: u64) -> u32 {
        // For R5K: pick victim way via LRU and encode into dc_idx via shift.
        // dc_ext_idx = set | (way << DCache::NUM_LINES_SHIFT)
        #[cfg(not(feature = "r5k"))]
        let (victim_way, dc_idx) = (0usize, self.dc.get_index(virt_addr));
        #[cfg(feature = "r5k")]
        let (victim_way, dc_idx) = {
            let set = self.dc.get_index(virt_addr);
            let way = unsafe { Self::lru_get(self.dc_lru.get(), set) } as usize;
            (way, set | (way << DCache::NUM_LINES_SHIFT))
        };

        // Writeback and invalidate the victim line (hardware fill — not a coherency action)
        self.writeback_l1d_line(dc_idx, false);
        self.invalidate_l1d_line(dc_idx, false, false);

        // Check if data is in L2 (skipped when L2 disabled — use memory directly).
        let l2_hit = if self.l2_active() {
            let l2_idx = self.l2.get_index(phys_addr);
            let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
            let l2_ptag = self.l2_ptag(phys_addr);
            if l2_tag.cs() != L2_CS_INVALID && l2_tag.ptag() == l2_ptag {
                // Check for Virtual Coherency Exception (R4K only; R5K dropped VCE)
                #[cfg(not(feature = "r5k"))]
                if self.pidx(virt_addr) != l2_tag.pidx() { return BUS_VCE; }
                true
            } else {
                #[cfg(not(feature = "lightning"))]
                if devlog_is_active(LogModule::L2c) && devlog_mask(LogModule::L2c) & CACHE_LOG_MISS != 0 {
                    crate::dlog!(LogModule::L2c, "fill virt={:#x} phys={:#x} idx={}", virt_addr, phys_addr, l2_idx);
                }
                if !self.fill_l2_line(phys_addr, virt_addr) { return BUS_ERR; }
                true
            }
        } else {
            false // L2 disabled: copy direct from memory below
        };
        #[cfg(not(feature = "lightning"))]
        if devlog_is_active(LogModule::L1d) && devlog_mask(LogModule::L1d) & CACHE_LOG_MISS != 0 {
            crate::dlog!(LogModule::L1d, "fill virt={:#x} phys={:#x} eidx={}", virt_addr, phys_addr, dc_idx);
        }

        let dc_data = self.dc.data_mut();
        let dc_start_chunk = dc_idx << DCache::CHUNKS_PER_LINE_SHIFT;

        if l2_hit {
            // Copy from L2 to L1-D
            let dc_line_base = phys_addr & !(DCache::LINE_MASK as u64);
            let l2_idx = self.l2.get_index(phys_addr);
            let l2_line_base = l2_idx << L2Cache::CHUNKS_PER_LINE_SHIFT;
            let offset_in_l2_line = ((dc_line_base & (L2Cache::LINE_MASK as u64)) >> 3) as usize;
            let l2_data = self.l2.data();
            for i in 0..DCache::CHUNKS_PER_LINE {
                dc_data[dc_start_chunk + i] = l2_data[l2_line_base + offset_in_l2_line + i];
            }
        } else {
            // L2 disabled: copy directly from memory
            let line_base = phys_addr & !(DCache::LINE_MASK as u64);
            let dest = &mut dc_data[dc_start_chunk..dc_start_chunk + DCache::CHUNKS_PER_LINE];
            self.downstream.read_block(line_base as u32, dest);
        }

        self.dc.set_tag(dc_idx, L1DTag::valid(phys_addr, L1D_CS_CLEAN_EXCLUSIVE as u8, false));

        // R5K: flip LRU — filled way is MRU, other way is now victim
        #[cfg(feature = "r5k")]
        unsafe { Self::lru_set(self.dc_lru.get(), dc_idx & DCache::NUM_LINES_MASK, victim_way ^ 1); }

        #[cfg(feature = "debug_cache")]
        {
            let line_base_phys = phys_addr & !(DCache::LINE_MASK as u64);
            let l2_idx_check = self.l2.get_index(phys_addr);
            if self.is_tracking_l1d(line_base_phys) || self.is_tracking_l2_idx(l2_idx_check) {
                let line_base_virt = virt_addr & !(DCache::LINE_MASK as u64);
                let way = dc_idx >> DCache::NUM_LINES_SHIFT;
                let set = dc_idx & DCache::NUM_LINES_MASK;
                println!("[CACHE DEBUG] fill_l1d_line: {} virt 0x{:016x} phys 0x{:016x} → L1D eidx=0x{:x} way={} set=0x{:x}",
                         self.tracking_label(line_base_phys), line_base_virt, line_base_phys, dc_idx, way, set);
                for i in 0..DCache::CHUNKS_PER_LINE {
                    println!("    [{}] 0x{:016x}", i, dc_data[dc_start_chunk + i]);
                }
            }
        }

        victim_way as u32
    }

    /// Ensure the L1-D line covering `virt_addr`/`phys_addr` is valid and tag-matched.
    ///
    /// Returns a single `u32`:
    ///   0   = hit/filled way0 (BUS_OK)
    ///   1   = hit/filled way1 (BUS_OK, R5K only)
    ///   >1  = BUS_VCE or BUS_ERR — propagate as error status
    ///
    /// Callers check `way <= 1` for success.
    /// `dc_ext_idx` for tag/data access = `set | (way << DCache::NUM_LINES_SHIFT)`.
    #[inline(always)]
    fn ensure_l1d_line(&self, virt_addr: u64, phys_addr: u64) -> u32 {
        #[cfg(not(feature = "r5k"))]
        {
            let dc_idx = self.dc.get_index(virt_addr);
            let dc_tag: L1DTag = self.dc.get_tag(dc_idx);
            if dc_tag.matches_phys(phys_addr) { 0 }
            else { self.fill_l1d_line(virt_addr, phys_addr) }
        }
        #[cfg(feature = "r5k")]
        {
            let set = self.dc.get_index(virt_addr);
            if self.dc.get_tag(set).matches_phys(phys_addr) {
                // way0 hit → way0 is MRU, way1 is LRU next
                unsafe { Self::lru_set(self.dc_lru.get(), set, 1); }
                return 0;
            }
            if self.dc.get_tag(set | (1 << DCache::NUM_LINES_SHIFT)).matches_phys(phys_addr) {
                // way1 hit → way1 is MRU, way0 is LRU next
                unsafe { Self::lru_set(self.dc_lru.get(), set, 0); }
                return 1;
            }
            self.fill_l1d_line(virt_addr, phys_addr)
        }
    }

    /// Compute the data-array address for `dc.dc_read/dc_write` from the extended tag index
    /// and the original virtual address.
    ///   dc_ext_idx = set | (way << DCache::NUM_LINES_SHIFT)
    ///   → data address = (dc_ext_idx << LINE_SHIFT) | (virt_addr & LINE_MASK)
    /// Way1 data lives in [DC_SIZE/2, DC_SIZE); dc_read/dc_write mask to (DC_SIZE-1) so this
    /// routes both ways into the correct half of the allocated data array.
    /// For R4K (1-way), dc_ext_idx = dc_idx so this equals the original virt_addr.
    #[inline(always)]
    fn dc_data_addr(dc_ext_idx: usize, virt_addr: u64) -> u64 {
        ((dc_ext_idx << DCache::LINE_SHIFT as usize) as u64)
            | (virt_addr & DCache::LINE_MASK as u64)
    }

    /// Mark the L1-D line as dirty.
    /// `dc_ext_idx` = `set | (way << DCache::NUM_LINES_SHIFT)` (returned by ensure_l1d_line).
    /// For R4K it is just `get_index(virt_addr)`.
    #[inline(always)]
    fn mark_l1d_dirty(&self, dc_ext_idx: usize) {
        let mut dc_tag: L1DTag = self.dc.get_tag(dc_ext_idx);
        dc_tag.dirty = true;
        self.dc.set_tag(dc_ext_idx, dc_tag);
    }

    /// Return the eidx of the L1-I way that holds `phys_addr`, or `None`.
    /// On R4K (1-way) only way 0 is checked. On R5K (2-way) both ways are probed.
    #[inline]
    fn hit_l1i(&self, virt_addr: u64, phys_addr: u64) -> Option<usize> {
        let set = self.ic.get_index(virt_addr);
        for way in 0..IC_WAYS {
            let eidx = set | (way << ICache::NUM_LINES_SHIFT);
            let tag: L1ITag = self.ic.get_tag(eidx);
            if tag.matches_phys(phys_addr) { return Some(eidx); }
        }
        None
    }

    /// Return the eidx of the L1-D way that holds `phys_addr`, or `None`.
    #[inline]
    fn hit_l1d(&self, virt_addr: u64, phys_addr: u64) -> Option<usize> {
        let set = self.dc.get_index(virt_addr);
        for way in 0..DC_WAYS {
            let eidx = set | (way << DCache::NUM_LINES_SHIFT);
            let tag: L1DTag = self.dc.get_tag(eidx);
            if tag.matches_phys(phys_addr) { return Some(eidx); }
        }
        None
    }

    /// Return the L2 index for `phys_addr` if the L2 is active and that line is valid, or `None`.
    /// L2 is always direct-mapped (1-way).
    #[inline]
    fn hit_l2(&self, phys_addr: u64) -> Option<usize> {
        if !self.l2_active() { return None; }
        let idx = self.l2.get_index(phys_addr);
        let tag: L2Tag = self.l2.get_tag(idx);
        if tag.ptag() == self.l2_ptag(phys_addr) && tag.cs() != L2_CS_INVALID {
            Some(idx)
        } else {
            None
        }
    }
}

impl MipsCache for R4000Cache {
    #[cfg(not(feature = "r5k"))]
    fn fetch(&self, virt_addr: u64, phys_addr: u64) -> FetchInstrResult {
        #[cfg(feature = "debug_cache")]
        let tracked = {
            if self.is_tracking_addr(virt_addr, phys_addr) {
                println!("[CACHE DEBUG] fetch: {} virt_addr 0x{:016x}, phys_addr 0x{:016x}",
                         self.tracking_label(phys_addr), virt_addr, phys_addr);
                true
            } else {
                let l2_idx = self.l2.get_index(phys_addr);
                if self.is_tracking_l2_idx(l2_idx) {
                    let line_base = phys_addr & !(L2Cache::LINE_MASK as u64);
                    println!("[CACHE DEBUG] fetch (L2 alias): idx=0x{:x}, line 0x{:08x}, virt 0x{:016x}, phys 0x{:016x}",
                             l2_idx, line_base, virt_addr, phys_addr);
                    true
                } else {
                    false
                }
            }
        };

        let ic_idx = self.ic.get_index(virt_addr);
        let ic_tag: L1ITag = self.ic.get_tag(ic_idx);

        #[cfg(feature = "developer")]
        self.l1i_fetch_count.fetch_add(1, Ordering::Relaxed);
        if !ic_tag.matches_phys(phys_addr) {
            let s = self.fill_l1i_line(virt_addr, phys_addr);
            if s != 0 { return FetchInstrResult::exception(s); }
        } else {
            #[cfg(feature = "developer")]
            self.l1i_hit_count.fetch_add(1, Ordering::Relaxed);
        }

        {
            // Plain shared-borrow read, matching the pre-fusion hot path exactly.
            // FLAG_IMM_IS_NEXT is precomputed at fill time (see fill_l2_line), not
            // here — see the comment there for why fetch() must not take get_mut().
            let l2_slot_idx = ((phys_addr as usize) & (L2_CACHE_SIZE - 1)) >> 2;
            let slot = &self.l2.instrs.get()[l2_slot_idx] as *const DecodedInstr;
            #[cfg(feature = "debug_cache")]
            if tracked {
                let raw = unsafe { (*slot).raw };
                println!("[CACHE DEBUG] fetch: virt 0x{:016x}, phys 0x{:016x} -> raw=0x{:08x}",
                         virt_addr, phys_addr, raw);
            }
            FetchInstrResult::hit(slot)
        }
    }

    #[cfg(feature = "r5k")]
    fn fetch(&self, virt_addr: u64, phys_addr: u64) -> FetchInstrResult {
        #[cfg(feature = "debug_cache")]
        {
            if self.is_tracking_addr(virt_addr, phys_addr) {
                println!("[CACHE DEBUG] fetch: {} virt_addr 0x{:016x}, phys_addr 0x{:016x}",
                         self.tracking_label(phys_addr), virt_addr, phys_addr);
            }
        }

        let set = self.ic.get_index(virt_addr);
        let way1_base = 1 << ICache::NUM_LINES_SHIFT;

        #[cfg(feature = "developer")]
        self.l1i_fetch_count.fetch_add(1, Ordering::Relaxed);

        let ic_eidx = if self.ic.get_tag(set).matches_phys(phys_addr) {
            #[cfg(feature = "developer")]
            self.l1i_hit_count.fetch_add(1, Ordering::Relaxed);
            #[cfg(not(feature = "lightning"))]
            if devlog_is_active(LogModule::L1i) && devlog_mask(LogModule::L1i) & CACHE_LOG_HIT != 0 {
                crate::dlog!(LogModule::L1i, "hit virt={:#x} phys={:#x} set={} way=0", virt_addr, phys_addr, set);
            }
            unsafe { Self::lru_set(self.ic_lru.get(), set, 1); } // way0 MRU → way1 is LRU
            set
        } else if self.ic.get_tag(set | way1_base).matches_phys(phys_addr) {
            #[cfg(feature = "developer")]
            self.l1i_hit_count.fetch_add(1, Ordering::Relaxed);
            #[cfg(not(feature = "lightning"))]
            if devlog_is_active(LogModule::L1i) && devlog_mask(LogModule::L1i) & CACHE_LOG_HIT != 0 {
                crate::dlog!(LogModule::L1i, "hit virt={:#x} phys={:#x} set={} way=1", virt_addr, phys_addr, set);
            }
            unsafe { Self::lru_set(self.ic_lru.get(), set, 0); } // way1 MRU → way0 is LRU
            set | way1_base
        } else {
            let way = self.fill_l1i_line(virt_addr, phys_addr);
            if way > 1 { return FetchInstrResult::exception(way); }
            set | (way as usize) << ICache::NUM_LINES_SHIFT
        };

        let instr_idx = (ic_eidx << ICache::INSTR_SHIFT)
            | ((virt_addr as usize >> 2) & ICache::INSTR_MASK);
        {
            let slot = &self.ic_instrs.get()[instr_idx] as *const DecodedInstr;
            FetchInstrResult::hit(slot)
        }
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
                    println!("[CACHE DEBUG] read (L2 alias): idx=0x{:x}, line 0x{:08x}, virt 0x{:016x}, phys 0x{:016x}, size {}",
                             l2_idx, line_base, virt_addr, phys_addr, SIZE);
                }
            }
        }

        // R4K (1-way): way is always 0, dc_eidx == get_index(virt_addr), da == virt_addr.
        // Skip the generic ensure_l1d_line/dc_data_addr indirection entirely.
        #[cfg(not(feature = "r5k"))]
        {
            let dc_idx = self.dc.get_index(virt_addr);
            if !self.dc.get_tag(dc_idx).matches_phys(phys_addr) {
                let r = self.fill_l1d_line(virt_addr, phys_addr);
                if r > 1 { return BusRead64 { status: r, data: 0 }; }
            }
            #[cfg(not(feature = "lightning"))]
            if devlog_is_active(LogModule::L1d) && devlog_mask(LogModule::L1d) & CACHE_LOG_HIT != 0 {
                crate::dlog!(LogModule::L1d, "read{} hit virt={:#x} phys={:#x} eidx={}", SIZE, virt_addr, phys_addr, dc_idx);
            }
            let result = self.dc.dc_read::<SIZE>(virt_addr);
            #[cfg(feature = "debug_cache")]
            if self.is_tracking_addr(virt_addr, phys_addr) {
                println!("[CACHE DEBUG] read{} result: {} virt 0x{:016x} phys 0x{:016x} val=0x{:016x}",
                         SIZE * 8, self.tracking_label(phys_addr), virt_addr, phys_addr, result);
            }
            return BusRead64::ok(result);
        }
        // R5K (2-way): use generic path with way encoding.
        #[cfg(feature = "r5k")]
        {
            let way = self.ensure_l1d_line(virt_addr, phys_addr);
            if way > 1 { return BusRead64 { status: way, data: 0 }; }
            let dc_eidx = self.dc.get_index(virt_addr) | (way as usize) << DCache::NUM_LINES_SHIFT;
            #[cfg(not(feature = "lightning"))]
            if devlog_is_active(LogModule::L1d) && devlog_mask(LogModule::L1d) & CACHE_LOG_HIT != 0 {
                crate::dlog!(LogModule::L1d, "read{} hit virt={:#x} phys={:#x} eidx={}", SIZE, virt_addr, phys_addr, dc_eidx);
            }
            let da = Self::dc_data_addr(dc_eidx, virt_addr);
            let result = self.dc.dc_read::<SIZE>(da);
            #[cfg(feature = "debug_cache")]
            if self.is_tracking_addr(virt_addr, phys_addr) {
                println!("[CACHE DEBUG] read{} result: {} virt 0x{:016x} phys 0x{:016x} val=0x{:016x}",
                         SIZE * 8, self.tracking_label(phys_addr), virt_addr, phys_addr, result);
            }
            BusRead64::ok(result)
        }
    }

    fn write<const SIZE: usize>(&self, virt_addr: u64, phys_addr: u64, val: u64) -> u32 {
        const { assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE") };
        #[cfg(feature = "debug_cache")]
        {
            if self.is_tracking_addr(virt_addr, phys_addr) {
                println!("[CACHE DEBUG] write{}: {} virt_addr 0x{:016x}, phys_addr 0x{:016x}, val 0x{:016x}",
                         SIZE * 8, self.tracking_label(phys_addr), virt_addr, phys_addr, val);
            }
        }

        // R4K (1-way): way always 0, dc_eidx == get_index(virt_addr), da == virt_addr.
        #[cfg(not(feature = "r5k"))]
        {
            let dc_idx = self.dc.get_index(virt_addr);
            if !self.dc.get_tag(dc_idx).matches_phys(phys_addr) {
                let r = self.fill_l1d_line(virt_addr, phys_addr);
                if r > 1 { return r; }
            }
            #[cfg(not(feature = "lightning"))]
            if devlog_is_active(LogModule::L1d) && devlog_mask(LogModule::L1d) & CACHE_LOG_HIT != 0 {
                crate::dlog!(LogModule::L1d, "write{} hit virt={:#x} phys={:#x} eidx={} val={:#x}", SIZE, virt_addr, phys_addr, dc_idx, val);
            }
            self.dc.dc_write::<SIZE>(virt_addr, val);
            self.mark_l1d_dirty(dc_idx);
            return BUS_OK;
        }
        // R5K (2-way): generic path.
        #[cfg(feature = "r5k")]
        {
            let way = self.ensure_l1d_line(virt_addr, phys_addr);
            if way > 1 { return way; }
            let dc_eidx = self.dc.get_index(virt_addr) | (way as usize) << DCache::NUM_LINES_SHIFT;
            #[cfg(not(feature = "lightning"))]
            if devlog_is_active(LogModule::L1d) && devlog_mask(LogModule::L1d) & CACHE_LOG_HIT != 0 {
                crate::dlog!(LogModule::L1d, "write{} hit virt={:#x} phys={:#x} eidx={} val={:#x}", SIZE, virt_addr, phys_addr, dc_eidx, val);
            }
            let da = Self::dc_data_addr(dc_eidx, virt_addr);
            self.dc.dc_write::<SIZE>(da, val);
            self.mark_l1d_dirty(dc_eidx);
            BUS_OK
        }
    }

    fn write64_masked(&self, virt_addr: u64, phys_addr: u64, val: u64, mask: u64) -> u32 {
        // SDL/SDR only — arbitrary sub-doubleword mask, always a RMW on the full u64 slot
        #[cfg(feature = "debug_cache")]
        {
            if self.is_tracking_addr(virt_addr, phys_addr) {
                println!("[CACHE DEBUG] write64_masked: {} virt_addr 0x{:016x}, phys_addr 0x{:016x}, val 0x{:016x}, mask 0x{:016x}",
                         self.tracking_label(phys_addr), virt_addr, phys_addr, val, mask);
            }
        }

        // R4K (1-way): da == virt_addr directly.
        #[cfg(not(feature = "r5k"))]
        {
            let dc_idx = self.dc.get_index(virt_addr);
            if !self.dc.get_tag(dc_idx).matches_phys(phys_addr) {
                let r = self.fill_l1d_line(virt_addr, phys_addr);
                if r > 1 { return r; }
            }
            let current = self.dc.dc_read::<8>(virt_addr);
            self.dc.dc_write::<8>(virt_addr, (current & !mask) | (val & mask));
            self.mark_l1d_dirty(dc_idx);
            return BUS_OK;
        }
        // R5K (2-way): generic path.
        #[cfg(feature = "r5k")]
        {
            let way = self.ensure_l1d_line(virt_addr, phys_addr);
            if way > 1 { return way; }
            let dc_eidx = self.dc.get_index(virt_addr) | (way as usize) << DCache::NUM_LINES_SHIFT;
            let da = Self::dc_data_addr(dc_eidx, virt_addr);
            let current = self.dc.dc_read::<8>(da);
            self.dc.dc_write::<8>(da, (current & !mask) | (val & mask));
            self.mark_l1d_dirty(dc_eidx);
            BUS_OK
        }
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

        // Drop L2 ops silently when there is no active L2 (r5k without r5ksc, or
        // r5ksc_triton with CONFIG_SE=0). hit_l2() also checks this, but index ops
        // (C_IINV, C_ILT, C_IST) go straight to self.l2 without hitting that helper.
        if is_l2 && !self.l2_active() { return 0; }

        #[cfg(not(feature = "lightning"))]
        {
            let log_mod = if is_icache { LogModule::L1i } else if is_l2 { LogModule::L2c } else { LogModule::L1d };
            if devlog_is_active(log_mod) && devlog_mask(log_mod) & CACHE_LOG_OP != 0 {
                crate::dlog!(log_mod, "{} virt={:#x} phys={:#x}", cache_op_name(cache_op), virt_addr, phys_addr);
            }
        }

        // L1I and L1D are virtually indexed; L2 is physically indexed.
        // R5K: for L1 index ops, virt_addr bit 14 selects the way — fold into eidx.
        let idx = if is_l2 {
            self.l2.get_index(phys_addr)
        } else if is_icache {
            #[cfg(not(feature = "r5k"))]
            { self.ic.get_index(virt_addr) }
            #[cfg(feature = "r5k")]
            { self.ic.get_index(virt_addr) | (((virt_addr >> 14) as usize & 1) << ICache::NUM_LINES_SHIFT) }
        } else {
            #[cfg(not(feature = "r5k"))]
            { self.dc.get_index(virt_addr) }
            #[cfg(feature = "r5k")]
            { self.dc.get_index(virt_addr) | (((virt_addr >> 14) as usize & 1) << DCache::NUM_LINES_SHIFT) }
        };

        #[cfg(feature = "debug_cache")]
        {
            let tracked = if is_l2 {
                // hit ops: phys_addr is the real address; index ops: phys_addr == virt_addr (index)
                self.is_tracking_l2_idx(idx)
                    || self.is_tracking_addr(phys_addr, phys_addr)
            } else {
                // For both L1I and L1D: fire on phys address match OR set index match
                self.is_tracking_addr(phys_addr, phys_addr)
                    || self.is_tracking_l2_idx(self.l2.get_index(phys_addr))
                    || self.is_tracking_l1d_idx(idx & DCache::NUM_LINES_MASK)
            };
            if tracked {
                let way = if !is_l2 { idx >> DCache::NUM_LINES_SHIFT } else { 0 };
                println!("[CACHE DEBUG] cache_op: {} virt={:#x} phys={:#x} idx=0x{:x} way={}",
                         cache_op_name(cache_op), virt_addr, phys_addr, idx, way);
            }
        }

        // cascade: on R5K, L1 cache ops must propagate to L2 (and L2 to memory) because
        // the PROM only flushes L1 when SC=1, relying on hardware to keep L2 coherent.
        //let cascade = !is_l2;
        #[cfg(feature = "r5k")]
        let cascade = false;
        #[cfg(not(feature = "r5k"))]
        let cascade = false;

        match operation {
            // Index Invalidate (I, SI) or Index Writeback Invalidate (D, SD).
            // R5K/Triton SI/SD: reinterpreted as C_INVALL — invalidate the entire L2
            // regardless of address. PROM issues this after enabling L2 via CONFIG_SE.
            C_IINV => { // same as C_IWBINV
                if is_icache {
                    self.invalidate_l1i_line(idx, cascade);
                } else if !is_l2 {
                    self.writeback_l1d_line(idx, cascade);
                    self.invalidate_l1d_line(idx, true, cascade);
                } else {
                    // Triton C_INVALL (SI/SD, index op): invalidate entire L2.
                    // R4K / external SC: standard index writeback+invalidate.
                    #[cfg(feature = "r5ksc_triton")]
                    self.invall_l2();
                    #[cfg(not(feature = "r5ksc_triton"))]
                    { self.writeback_l2_line(idx); self.invalidate_l2_line(idx); }
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
                    // L1-I TagLo format:  [31:8] raw_ptag   [7:6] pstate (2=valid, 0=invalid)
                    let tag: L1ITag = self.ic.get_tag(idx);
                    let raw_ptag = (tag.ptag >> L1_PTAG_SHIFT) as u32 & L1_PTAG_MASK;
                    let pstate = if tag.is_valid() { 2u32 } else { 0u32 };
                    (raw_ptag << 8) | (pstate << 6)
                } else {
                    // L1-D TagLo format:  [31:8] raw_ptag   [7:6] pstate
                    let tag: L1DTag = self.dc.get_tag(idx);
                    let raw_ptag = (tag.ptag >> L1_PTAG_SHIFT) as u32 & L1_PTAG_MASK;
                    // dirty=true promotes CleanExclusive→DirtyExclusive in TagLo output
                    let pstate = match tag.cs as u32 {
                        L1D_CS_INVALID => 0u32,
                        L1D_CS_SHARED => 1u32,
                        L1D_CS_CLEAN_EXCLUSIVE => if tag.dirty { 3u32 } else { 2u32 },
                        L1D_CS_DIRTY_EXCLUSIVE => 3u32,
                        _ => 0u32,
                    };
                    (raw_ptag << 8) | (pstate << 6)
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
                    // L1 TagLo format:  [31:8] raw_ptag   [7:6] pstate
                    let raw_ptag = (tag_lo >> 8) & L1_PTAG_MASK;
                    let ptag_line = (raw_ptag as u64) << L1_PTAG_SHIFT; // convert to line-base form
                    let pstate = (tag_lo >> 6) & 0x3;

                    if is_icache {
                        // Evict existing line first to maintain L1I data pointer integrity.
                        self.invalidate_l1i_line(idx, cascade);
                        self.ic.set_tag(idx, if pstate != 0 { L1ITag::valid(ptag_line) } else { L1ITag::default() });
                    } else {
                        let cs = match pstate {
                            0 => L1D_CS_INVALID as u8,
                            1 => L1D_CS_SHARED as u8,
                            2 => L1D_CS_CLEAN_EXCLUSIVE as u8,
                            3 => L1D_CS_DIRTY_EXCLUSIVE as u8,
                            _ => L1D_CS_INVALID as u8,
                        };
                        // Writeback dirty data before overwriting the tag.
                        self.writeback_l1d_line(idx, cascade);
                        self.invalidate_l1d_line(idx, true, cascade);
                        self.dc.set_tag(idx, if cs != 0 { L1DTag::valid(ptag_line, cs, cs == L1D_CS_DIRTY_EXCLUSIVE as u8) } else { L1DTag::default() });
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
                    // Writeback (and invalidate) the existing L2 occupant before claiming
                    // the line as dirty exclusive — otherwise dirty data is silently lost.
                    self.writeback_l2_line(idx);
                    self.invalidate_l2_line(idx);
                    let mut t = L2Tag::default();
                    t.set_ptag(self.l2_ptag(phys_addr));
                    t.set_cs(L2_CS_DIRTY_EXCLUSIVE);
                    t.set_pidx(self.pidx(virt_addr));
                    self.l2.set_tag(idx, t);
                } else {
                    // Writeback the old L1D occupant before overwriting its tag.
                    self.writeback_l1d_line(idx, false); // CDX: claiming line, no cascade needed
                    self.dc.set_tag(idx, L1DTag::valid(phys_addr, L1D_CS_DIRTY_EXCLUSIVE as u8, true));
                }
                0
            }

            // Hit Invalidate
            C_HINV => {
                if is_l2 {
                    if let Some(idx) = self.hit_l2(phys_addr) {
                        self.invalidate_l2_line(idx);
                    }
                } else if is_icache {
                    if let Some(eidx) = self.hit_l1i(virt_addr, phys_addr) {
                        self.invalidate_l1i_line(eidx, cascade);
                    }
                } else {
                    if let Some(eidx) = self.hit_l1d(virt_addr, phys_addr) {
                        self.invalidate_l1d_line(eidx, true, cascade);
                    }
                }
                0
            }

            // Hit Writeback Invalidate (D, SD) or Fill (I)
            // R5K/Triton SD: reinterpreted as C_INVPAGE — invalidate all L2 lines
            // in the 4KB-aligned page containing phys_addr (address IS significant).
            C_HWBINV => { // same as C_FILL
                if is_icache {
                    // Fill operation: L1I is virtually indexed, use virt_addr for index.
                    let _ = self.fill_l1i_line(virt_addr, phys_addr);
                } else if is_l2 {
                    if let Some(idx) = self.hit_l2(phys_addr) {
                        // Triton C_INVPAGE (SD, hit op): invalidate all L2 lines in the page.
                        // R4K / external SC: standard hit writeback+invalidate.
                        #[cfg(feature = "r5ksc_triton")]
                        self.invpage_l2(phys_addr);
                        #[cfg(not(feature = "r5ksc_triton"))]
                        { self.writeback_l2_line(idx); self.invalidate_l2_line(idx); }
                    }
                } else {
                    if let Some(eidx) = self.hit_l1d(virt_addr, phys_addr) {
                        self.writeback_l1d_line(eidx, cascade);
                        self.invalidate_l1d_line(eidx, true, cascade);
                    }
                }
                0
            }

            // Hit Writeback
            C_HWB => {
                if !is_icache {
                    if is_l2 {
                        if let Some(idx) = self.hit_l2(phys_addr) {
                            self.writeback_l2_line(idx);
                        }
                    } else {
                        if let Some(eidx) = self.hit_l1d(virt_addr, phys_addr) {
                            self.writeback_l1d_line(eidx, cascade);
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
                let set = self.ic.get_index(virt_addr);
                let num_ways = ICache::NUM_LINES / (IC_SIZE / IC_LINE / IC_WAYS).max(1);
                let sets_per_way = IC_SIZE / IC_LINE / num_ways.max(1);
                // Compute the overall verdict up front (any way hitting is a
                // HIT) so the very first line states it plainly — a per-way
                // "<-- HIT"/nothing marker buried after several unrelated
                // Way0/Way1 lines reads as "this line's data is what's here"
                // even on a clean MISS, which is exactly backwards (found
                // live: a MISS on an unrelated line was mistaken for a
                // stale/corrupt line belonging to the probed address — the
                // display never said MISS anywhere, just omitted the arrow).
                let any_hit = (0..num_ways).any(|way| {
                    let eidx = set + way * sets_per_way;
                    self.ic.get_tag(eidx).matches_phys(phys_addr)
                });
                let mut s = format!("L1-I probe virt 0x{:016x} phys 0x{:016x} set=0x{:x}: {}\n",
                    virt_addr, phys_addr, set, if any_hit { "HIT" } else { "MISS" });
                for way in 0..num_ways {
                    let eidx = set + way * sets_per_way;
                    let tag: L1ITag = self.ic.get_tag(eidx);
                    let hit = tag.matches_phys(phys_addr);
                    s.push_str(&format!("  Way{}: eidx=0x{:x} tag=0x{:010x} valid={} {}",
                        way, eidx, tag.line_addr(), tag.is_valid(),
                        if hit { "<-- HIT" } else { "" }));
                    s.push('\n');
                }
                s
            }
            "l1d" => {
                let set = self.dc.get_index(virt_addr);
                let num_ways = DCache::NUM_LINES / (DC_SIZE / DC_LINE / DC_WAYS).max(1);
                let sets_per_way = DC_SIZE / DC_LINE / num_ways.max(1);
                // See l1i's own comment above — same "state the verdict up
                // front" fix.
                let any_hit = (0..num_ways).any(|way| {
                    let eidx = set + way * sets_per_way;
                    self.dc.get_tag(eidx).matches_phys(phys_addr)
                });
                let mut s = format!("L1-D probe virt 0x{:016x} phys 0x{:016x} set=0x{:x}: {}\n",
                    virt_addr, phys_addr, set, if any_hit { "HIT" } else { "MISS" });
                for way in 0..num_ways {
                    let eidx = set + way * sets_per_way;
                    let tag: L1DTag = self.dc.get_tag(eidx);
                    let hit = tag.matches_phys(phys_addr);
                    let cs_str = match tag.cs as u32 {
                        L1D_CS_INVALID => "Invalid",
                        L1D_CS_SHARED => "Shared",
                        L1D_CS_CLEAN_EXCLUSIVE => "CleanExclusive",
                        L1D_CS_DIRTY_EXCLUSIVE => "DirtyExclusive",
                        _ => "Unknown",
                    };
                    s.push_str(&format!("  Way{}: eidx=0x{:x} tag=0x{:010x} cs={} dirty={} {}",
                        way, eidx, tag.line_addr(), cs_str, tag.dirty,
                        if hit { "<-- HIT" } else { "" }));
                    s.push('\n');
                }
                s
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
                // For R5K: valid indices are 0..NUM_LINES*WAYS-1 (both ways in flat array).
                // For R4K: valid indices are 0..NUM_LINES-1.
                let max_idx = IC_SIZE / IC_LINE; // total tags across all ways
                if idx >= max_idx {
                    return format!("Index 0x{:x} out of bounds (max 0x{:x})", idx, max_idx - 1);
                }
                let tag: L1ITag = self.ic.get_tag(idx);
                let instrs_per_ic_line = ICache::INSTRS_PER_LINE;

                // R5K: instruction words live in ic_instrs (owned by L1I, indexed by eidx).
                // R4K: instruction words live in l2.instrs (indexed by physical word address).
                #[cfg(feature = "r5k")]
                let mut s = {
                    let ic_instrs = self.ic_instrs.get();
                    let slot_base = idx << ICache::INSTR_SHIFT;
                    let way = idx / (IC_SIZE / IC_LINE / IC_WAYS);
                    let set = idx % (IC_SIZE / IC_LINE / IC_WAYS);
                    let mut s = format!("L1-I Line 0x{:x} (way={} set=0x{:x}): Tag=0x{:010x} V={}\n  Instrs:",
                        idx, way, set, tag.line_addr(), tag.is_valid());
                    for i in 0..instrs_per_ic_line {
                        if i % 4 == 0 { s.push_str("\n    "); }
                        if slot_base + i < ic_instrs.len() {
                            s.push_str(&format!("{:08x} ", ic_instrs[slot_base + i].raw));
                        }
                    }
                    s
                };
                #[cfg(not(feature = "r5k"))]
                let mut s = {
                    let l2_data = self.l2.data();
                    let phys_base = tag.line_addr() as usize;
                    let l2_slot_base = (phys_base & (L2_SIZE - 1)) >> 2;
                    let mut s = format!("L1-I Line 0x{:x}: Tag=0x{:010x} V={}\n  Instrs:", idx, tag.line_addr(), tag.is_valid());
                    for i in 0..instrs_per_ic_line {
                        if i % 4 == 0 { s.push_str("\n    "); }
                        let l2_slot_idx = l2_slot_base + i;
                        let chunk = l2_data[l2_slot_idx >> 1];
                        let from_data = if l2_slot_idx & 1 == 0 { (chunk >> 32) as u32 } else { chunk as u32 };
                        {
                            let ic_instrs = self.l2.instrs.get();
                            if l2_slot_idx < ic_instrs.len() {
                                let from_instrs = ic_instrs[l2_slot_idx].raw;
                                if from_instrs != from_data {
                                    s.push_str(&format!("{:08x}[DATA={:08x}!] ", from_instrs, from_data));
                                } else {
                                    s.push_str(&format!("{:08x} ", from_data));
                                }
                            }
                        }
                    }
                    s
                };
                // Append L2 data for comparison
                if tag.is_valid() {
                    let l2_data = self.l2.data();
                    let l2_idx = self.l2.get_index(tag.line_addr());
                    let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
                    let l2_base = l2_idx << L2Cache::CHUNKS_PER_LINE_SHIFT;
                    let sub = ((tag.line_addr() as usize) & L2Cache::LINE_MASK) >> 3;
                    s.push_str(&format!("\n  L2[0x{:x}] cs={}: ", l2_idx, l2_tag.cs()));
                    for i in 0..ICache::CHUNKS_PER_LINE {
                        if l2_base + sub + i < l2_data.len() {
                            s.push_str(&format!("{:016x} ", l2_data[l2_base + sub + i]));
                        }
                    }
                }
                s
            }
            "l1d" => {
                let max_idx = DC_SIZE / DC_LINE; // total tags across all ways
                if idx >= max_idx {
                    return format!("Index 0x{:x} out of bounds (max 0x{:x})", idx, max_idx - 1);
                }
                let tag: L1DTag = self.dc.get_tag(idx);
                let cs_str = match tag.cs as u32 {
                    L1D_CS_INVALID => "Invalid",
                    L1D_CS_SHARED => "Shared",
                    L1D_CS_CLEAN_EXCLUSIVE => "CleanExclusive",
                    L1D_CS_DIRTY_EXCLUSIVE => "DirtyExclusive",
                    _ => "Unknown",
                };

                let dc_data = self.dc.data();
                let start = idx << DCache::CHUNKS_PER_LINE_SHIFT;

                let mut s = format!("L1-D Line 0x{:x}: Tag=0x{:010x} CS={} ({}) D={}\n  Data:",
                    idx, tag.ptag, tag.cs, cs_str, tag.dirty);
                for i in 0..DCache::CHUNKS_PER_LINE {
                    if i % 4 == 0 { s.push_str("\n    "); }
                    if start + i < dc_data.len() {
                        s.push_str(&format!("{:016x} ", dc_data[start + i]));
                    }
                }
                s
            }
            "l2" => {
                if L2Cache::NUM_LINES == 0 || idx >= L2Cache::NUM_LINES {
                    return format!("Index 0x{:x} out of bounds (max 0x{:x})", idx, L2Cache::NUM_LINES.saturating_sub(1));
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
        self.ic.tags_mut().fill(L1ITag::default());
        self.dc.tags_mut().fill(L1DTag::default());
        self.dc.data_mut().fill(0);
        self.l2.tags_mut().fill(L2Tag::default());
        self.l2.data_mut().fill(0);
        #[cfg(not(feature = "r5k"))]
        for s in self.l2.instrs.get_mut().iter_mut() { s.flags = FLAG_NOT_DECODED; s.raw = 0; }
        #[cfg(feature = "r5k")]
        for s in self.ic_instrs.get_mut().iter_mut() { s.flags = FLAG_NOT_DECODED; s.raw = 0; }
        #[cfg(feature = "r5k")]
        unsafe {
            (*self.ic_lru.get()).fill(0u64);
            (*self.dc_lru.get()).fill(0u64);
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
        self.ic.tags_mut().fill(L1ITag::default());
        self.dc.tags_mut().fill(L1DTag::default());
        self.dc.data_mut().fill(0);
        self.l2.tags_mut().fill(L2Tag::default());
        self.l2.data_mut().fill(0);
        #[cfg(not(feature = "r5k"))]
        for s in self.l2.instrs.get_mut().iter_mut() { s.flags = FLAG_NOT_DECODED; s.raw = 0; }
        #[cfg(feature = "r5k")]
        for s in self.ic_instrs.get_mut().iter_mut() { s.flags = FLAG_NOT_DECODED; s.raw = 0; }
        #[cfg(feature = "r5k")]
        unsafe {
            (*self.ic_lru.get()).fill(0u64);
            (*self.dc_lru.get()).fill(0u64);
        }
        unsafe {
            *self.llbit.get() = false;
            *self.lladdr.get() = 0;
        }
    }
}

// ---- snapshot helpers + MipsCache save/load override ----

impl R4000Cache {
    fn save_tags_as_u32<TAG: Copy + Into<u32>>(tags: &[TAG]) -> Vec<u32> {
        tags.iter().map(|&t| t.into()).collect()
    }

    fn load_tags_from_u32<TAG: Default + Copy + From<u32>>(dst: &mut [TAG], src: &[u32]) {
        let tl = src.len().min(dst.len());
        for i in 0..tl { dst[i] = TAG::from(src[i]); }
    }

    pub fn save_cache_state(&self) -> toml::Value {
        let ic_tags = Self::save_tags_as_u32(self.ic.tags());
        let dc_tags = Self::save_tags_as_u32(self.dc.tags());
        let l2_tags = Self::save_tags_as_u32(self.l2.tags());
        let dc_data = self.dc.data().to_vec();
        let l2_data = self.l2.data().to_vec();
        let llbit = unsafe { *self.llbit.get() };
        let lladdr = unsafe { *self.lladdr.get() };

        let mut t = toml::value::Table::new();
        t.insert("ic_tags".into(),  u32_slice_to_toml(&ic_tags));
        t.insert("dc_tags".into(),  u32_slice_to_toml(&dc_tags));
        // Absence of this key means dc_tags carries dirty in bit 27. See migrate_l1d_tag_word_v0.
        t.insert("dc_tag_format".into(), toml::Value::Integer(1));
        t.insert("dc_data".into(),  u64_slice_to_toml(&dc_data));
        t.insert("l2_tags".into(),  u32_slice_to_toml(&l2_tags));
        t.insert("l2_data".into(),  u64_slice_to_toml(&l2_data));
        t.insert("llbit".into(),    toml::Value::Boolean(llbit));
        t.insert("lladdr".into(),   hex_u32(lladdr));
        // R5K: save LRU state as packed u32 words (1 bit per set, same on-disk format as before).
        // ic_instrs not saved — rebuilt from l2.data on first fetch miss after restore.
        #[cfg(feature = "r5k")]
        {
            let pack = |lru: &[u64], num_sets: usize| -> Vec<u32> {
                (0..num_sets.div_ceil(32)).map(|i| {
                    let base = i * 32;
                    (0..32usize).filter(|&b| base + b < num_sets)
                        .fold(0u32, |acc, b| {
                            let set = base + b;
                            acc | (((lru[set >> 6] >> (set & 63)) & 1) as u32) << b
                        })
                }).collect()
            };
            let ic_lru = unsafe { &*self.ic_lru.get() };
            let dc_lru = unsafe { &*self.dc_lru.get() };
            t.insert("ic_lru".into(), u32_slice_to_toml(&pack(ic_lru, IC_NUM_SETS)));
            t.insert("dc_lru".into(), u32_slice_to_toml(&pack(dc_lru, DC_NUM_SETS)));
        }
        toml::Value::Table(t)
    }

    pub fn load_cache_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut ic_tags = vec![0u32; ICache::NUM_LINES];
        let mut dc_tags = vec![0u32; DCache::NUM_LINES];
        let mut dc_data = vec![0u64; DC_SIZE / 8];
        let mut l2_tags = vec![0u32; L2Cache::NUM_LINES];
        let mut l2_data = vec![0u64; L2_SIZE / 8];

        if let Some(f) = get_field(v, "ic_tags") { load_u32_slice(f, &mut ic_tags); }
        if let Some(f) = get_field(v, "dc_tags") { load_u32_slice(f, &mut dc_tags); }
        if let Some(f) = get_field(v, "dc_data") { load_u64_slice(f, &mut dc_data); }
        if let Some(f) = get_field(v, "l2_tags") { load_u32_slice(f, &mut l2_tags); }
        if let Some(f) = get_field(v, "l2_data") { load_u64_slice(f, &mut l2_data); }

        match get_field(v, "dc_tag_format").and_then(|f| f.as_integer()) {
            None | Some(0) => {
                for w in dc_tags.iter_mut() { *w = migrate_l1d_tag_word_v0(*w); }
            }
            Some(1) => {}
            Some(n) => return Err(format!("unknown dc_tag_format {}", n)),
        }

        Self::load_tags_from_u32(self.ic.tags_mut(), &ic_tags);
        Self::load_tags_from_u32(self.dc.tags_mut(), &dc_tags);
        Self::load_tags_from_u32(self.l2.tags_mut(), &l2_tags);
        let dl = dc_data.len().min(DC_SIZE / 8);
        self.dc.data_mut()[..dl].copy_from_slice(&dc_data[..dl]);
        let dl = l2_data.len().min(L2_SIZE / 8);
        self.l2.data_mut()[..dl].copy_from_slice(&l2_data[..dl]);

        // R4K: rebuild l2.instrs from restored l2.data; fetch() indexes it directly.
        // R5K: l2.instrs is empty; ic_instrs will be repopulated on next L1I miss.
        #[cfg(not(feature = "r5k"))]
        {
            let l2_data_slice = self.l2.data();
            let l2_instrs = self.l2.instrs.get_mut();
            for line in 0..L2Cache::NUM_LINES {
                let chunks_start = line << L2Cache::CHUNKS_PER_LINE_SHIFT;
                let instrs_start = line << L2Cache::INSTR_SHIFT;
                for i in 0..L2Cache::CHUNKS_PER_LINE {
                    let chunk = l2_data_slice[chunks_start + i];
                    l2_instrs[instrs_start + i * 2].raw = (chunk >> 32) as u32;
                    l2_instrs[instrs_start + i * 2].flags = FLAG_NOT_DECODED;
                    l2_instrs[instrs_start + i * 2 + 1].raw = chunk as u32;
                    l2_instrs[instrs_start + i * 2 + 1].flags = FLAG_NOT_DECODED;
                }
            }
        }

        // R5K: restore LRU bits; ic_instrs will be repopulated on first fetch miss.
        #[cfg(feature = "r5k")]
        {
            let unpack = |packed: &[u32], dst: &mut [u64], num_sets: usize| {
                dst.fill(0);
                for set in 0..num_sets {
                    if (packed[set / 32] >> (set % 32)) & 1 != 0 {
                        dst[set >> 6] |= 1u64 << (set & 63);
                    }
                }
            };
            let mut ic_lru_packed = vec![0u32; IC_NUM_SETS.div_ceil(32)];
            let mut dc_lru_packed = vec![0u32; DC_NUM_SETS.div_ceil(32)];
            if let Some(f) = get_field(v, "ic_lru") { load_u32_slice(f, &mut ic_lru_packed); }
            if let Some(f) = get_field(v, "dc_lru") { load_u32_slice(f, &mut dc_lru_packed); }
            unpack(&ic_lru_packed, unsafe { &mut *self.ic_lru.get() }, IC_NUM_SETS);
            unpack(&dc_lru_packed, unsafe { &mut *self.dc_lru.get() }, DC_NUM_SETS);
        }

        if let Some(f) = get_field(v, "llbit") {
            if let Some(b) = toml_bool(f) { unsafe { *self.llbit.get() = b; } }
        }
        if let Some(f) = get_field(v, "lladdr") {
            if let Some(a) = toml_u32(f) { unsafe { *self.lladdr.get() = a; } }
        }
        Ok(())
    }
}

// =============================================================================
// Cache correctness tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::mem::Memory;
    use crate::traits::{BUS_OK, Resettable};

    // 4MB — enough tag diversity to exercise eviction; power-of-two for easy masking.
    const MEM_MB: usize = 4;
    const MEM_BYTES: usize = MEM_MB * 1024 * 1024;
    const ADDR_MASK: u32 = (MEM_BYTES as u32 - 1) & !3; // word-aligned, in range

    // Virtual address in kseg0; pidx bits[14:12] == 0 so R4K never fires VCE.
    fn kseg0(phys: u32) -> u64 { 0x8000_0000u64 | (phys as u64 & 0x0FFF_FFFF) }

    fn make_cache(mem: Arc<Memory>) -> R4000Cache {
        R4000Cache::new(mem as Arc<dyn BusDevice>)
    }

    // xorshift64 — no external crate.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self { Self(seed | 1) }
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
            self.0 as u32
        }
    }

    /// L1D random read/write: 1M word operations against a shadow copy.
    #[test]
    fn l1d_random_stress() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());
        let mut rng = Rng::new(0xdeadbeef_cafebabe);
        let mut shadow = vec![0u32; MEM_BYTES / 4];

        for _ in 0..1_000_000 {
            let phys = rng.next_u32() & ADDR_MASK;
            let virt = kseg0(phys);
            if rng.next_u32() & 1 == 0 {
                let val = rng.next_u32();
                let st = cache.write::<4>(virt, phys as u64, val as u64);
                assert_eq!(st, BUS_OK, "L1D write error phys={:#010x}", phys);
                shadow[(phys / 4) as usize] = val;
            } else {
                let r = cache.read::<4>(virt, phys as u64);
                assert_eq!(r.status, BUS_OK, "L1D read error phys={:#010x}", phys);
                let want = shadow[(phys / 4) as usize];
                assert_eq!(r.data as u32, want,
                    "L1D mismatch phys={:#010x}: got={:#010x} want={:#010x}", phys, r.data as u32, want);
            }
        }

        // Index_WBInvalidate all L1D sets (both ways for R5K — way selected by addr bit 14),
        // then Index_WBInvalidate all L2 sets, so backing memory is fully up to date.
        let dc_sets = DC_SIZE / DC_LINE / DC_WAYS;
        for way in 0..DC_WAYS {
            for set in 0..dc_sets {
                // Bit 14 selects way for R5K index ops; for R4K WAYS==1 so way is always 0.
                let virt = kseg0(((way << 14) | (set * DC_LINE)) as u32);
                cache.cache_op(C_IINV | CACH_PD, virt, virt & 0x1FFF_FFFF);
            }
        }
        for i in 0..(L2_SIZE / L2_LINE) {
            let phys = (i * L2_LINE) as u64;
            cache.cache_op(C_IWBINV | CACH_SD, phys, phys);
        }
        for (i, &want) in shadow.iter().enumerate() {
            if want == 0 { continue; }
            let phys = (i * 4) as u32;
            let got = mem.read32(phys).data;
            assert_eq!(got, want, "post-flush mismatch phys={:#010x}: got={:#010x} want={:#010x}", phys, got, want);
        }
    }

    /// L1I fetch stress: 1M random fetches against memory pre-filled with known words.
    #[test]
    fn l1i_fetch_stress() {
        let mem = Arc::new(Memory::new(MEM_MB));
        // Pre-fill with deterministic pattern directly through the bus.
        let mut rng = Rng::new(0x1234_5678_9abc_def0);
        let mut words = vec![0u32; MEM_BYTES / 4];
        for (i, w) in words.iter_mut().enumerate() {
            *w = rng.next_u32();
            mem.write32((i * 4) as u32, *w);
        }

        let cache = make_cache(mem.clone());
        let mut rng2 = Rng::new(0xfeed_face_dead_beef);

        for op in 0..1_000_000 {
            let phys = rng2.next_u32() & ADDR_MASK;
            let virt = kseg0(phys);
            let r = cache.fetch(virt, phys as u64);
            assert_eq!(r.status, EXEC_COMPLETE, "L1I exception phys={:#010x} op={}", phys, op);
            let got = unsafe { (*r.instr).raw };
            let want = words[(phys / 4) as usize];
            assert_eq!(got, want,
                "L1I mismatch phys={:#010x} op={}: got={:#010x} want={:#010x}", phys, op, got, want);
        }
    }

    // -----------------------------------------------------------------------
    // Cache-op unit tests — exercise every CACHE instruction variant against
    // a known-state cache, for both R4K and R5K geometries.
    // -----------------------------------------------------------------------

    // Flush the entire cache hierarchy to backing memory and invalidate
    // everything, so tests start from a clean slate.
    fn full_flush(cache: &R4000Cache) {
        // Index_WBInvalidate all L1D sets (both ways for R5K).
        for way in 0..DC_WAYS {
            for set in 0..DC_SIZE / DC_LINE / DC_WAYS {
                let nls = DCache::NUM_LINES_SHIFT as usize;
                let ls  = DCache::LINE_SHIFT as usize;
                let idx_addr = (way << (nls + ls)) | (set << ls);
                let v = kseg0(idx_addr as u32);
                cache.cache_op(C_IWBINV | CACH_PD, v, v & 0x1FFF_FFFF);
            }
        }
        // Index_WBInvalidate all L2 sets (single-way, physically indexed).
        for set in 0..L2_SIZE / L2_LINE {
            let p = (set * L2_LINE) as u64;
            cache.cache_op(C_IWBINV | CACH_SD, p, p);
        }
        // Index_Invalidate all L1I sets (both ways for R5K).
        for way in 0..IC_WAYS {
            for set in 0..IC_SIZE / IC_LINE / IC_WAYS {
                let nls = ICache::NUM_LINES_SHIFT as usize;
                let ls  = ICache::LINE_SHIFT as usize;
                let idx_addr = (way << (nls + ls)) | (set << ls);
                let v = kseg0(idx_addr as u32);
                cache.cache_op(C_IINV | CACH_PI, v, v & 0x1FFF_FFFF);
            }
        }
    }

    // Write a word into backing memory, bypassing the cache entirely.
    fn mem_write(mem: &Memory, phys: u32, val: u32) { mem.write32(phys, val); }
    fn mem_read(mem: &Memory, phys: u32) -> u32 { mem.read32(phys).data }

    // Flush a single L2 set to memory.  `phys` is any address within the L2 line.
    // Only valid when L2 is present (r4k or r5k+r5ksc).
    #[cfg(not(all(feature = "r5k", not(feature = "r5ksc"))))]
    fn flush_l2_to_mem(cache: &R4000Cache, phys: u32) {
        let l2_set = (phys as usize >> L2Cache::LINE_SHIFT as usize) & (L2_SIZE / L2_LINE - 1);
        let lp = (l2_set * L2_LINE) as u64;
        cache.cache_op(C_IWBINV | CACH_SD, lp, lp);
    }

    /// Index_WBInvalidate L1D: dirty line goes to L2; L1D tag becomes invalid.
    /// Verify: data reaches L2 (by also flushing L2 to memory), and tag is invalidated
    /// (subsequent read after memory update picks up new value).
    #[test]
    #[cfg(not(all(feature = "r5k", not(feature = "r5ksc"))))]
    fn cache_op_index_wbinv_l1d() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let phys: u32 = 0x1000;
        let virt = kseg0(phys);
        let _ = cache.write::<4>(virt, phys as u64, 0xABCD_1234u64);

        // Index_WBInvalidate the L1D set (way 0 — bit14 of index address = 0).
        let set = (phys as usize >> DCache::LINE_SHIFT as usize) & DCache::NUM_LINES_MASK;
        let v0 = kseg0((set << DCache::LINE_SHIFT as usize) as u32);
        cache.cache_op(C_IWBINV | CACH_PD, v0, v0 & 0x1FFF_FFFF);

        // Data should be in L2 now.  Flush L2 → memory and verify.
        flush_l2_to_mem(&cache, phys);
        assert_eq!(mem_read(&mem, phys), 0xABCD_1234,
            "Index_WBInv(PD) did not propagate dirty L1D data to L2/memory");

        // L1D tag should be invalid — write to memory and refill should see new value.
        mem_write(&mem, phys, 0xDEAD_BEEF);
        let r = cache.read::<4>(virt, phys as u64);
        assert_eq!(r.data as u32, 0xDEAD_BEEF,
            "Index_WBInv(PD) left stale L1D tag after invalidate");
    }

    /// Index_Invalidate L1I: after invalidate, L1I refills from current memory.
    /// L1I fill path goes through L2, so we must also update L2 to see new memory
    /// content; OR use addresses where L2 is also cold.
    #[test]
    #[cfg(not(all(feature = "r5k", not(feature = "r5ksc"))))]
    fn cache_op_index_inv_l1i() {
        let mem = Arc::new(Memory::new(MEM_MB));
        // Pre-fill memory with a pattern.
        let phys: u32 = 0x2000;
        mem_write(&mem, phys, 0x1111_1111);
        let cache = make_cache(mem.clone());
        let virt = kseg0(phys);

        // Warm up L1I (also fills L2).
        let r0 = cache.fetch(virt, phys as u64);
        assert_eq!(r0.status, EXEC_COMPLETE);
        let got0 = unsafe { (*r0.instr).raw };
        assert_eq!(got0, 0x1111_1111);

        // Directly update memory behind the cache.
        mem_write(&mem, phys, 0x2222_2222);

        // L1I should still see old value (stale).
        let r1 = cache.fetch(virt, phys as u64);
        let stale = unsafe { (*r1.instr).raw };
        assert_eq!(stale, 0x1111_1111, "expected stale L1I hit before invalidate");

        // Index_Invalidate L1I (way 0 — bit14=0 in index address).
        let set = (phys as usize >> ICache::LINE_SHIFT as usize) & ICache::NUM_LINES_MASK;
        let iv = kseg0((set << ICache::LINE_SHIFT as usize) as u32);
        cache.cache_op(C_IINV | CACH_PI, iv, iv & 0x1FFF_FFFF);

        // L1I tag is now invalid.  But L2 still caches the old value — also invalidate L2
        // so the next fill reads from updated memory.
        flush_l2_to_mem(&cache, phys);
        // Write the new value to memory after L2 flush (L2 is now invalid for this line).
        mem_write(&mem, phys, 0x2222_2222);

        // Fetch should refill L2 from memory then L1I from L2 — sees updated value.
        let r2 = cache.fetch(virt, phys as u64);
        assert_eq!(r2.status, EXEC_COMPLETE);
        let fresh = unsafe { (*r2.instr).raw };
        assert_eq!(fresh, 0x2222_2222,
            "Index_Inv(PI) did not invalidate L1I — stale fetch after invalidate+L2 flush");
    }

    /// Hit_WBInvalidate L1D: flushes dirty line to L2 and invalidates tag.
    #[test]
    #[cfg(not(all(feature = "r5k", not(feature = "r5ksc"))))]
    fn cache_op_hit_wbinv_l1d() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let phys: u32 = 0x3000;
        let virt = kseg0(phys);
        let _ = cache.write::<4>(virt, phys as u64, 0xCAFE_0001u64);

        // Hit_WBInvalidate flushes L1D dirty data to L2 and invalidates the tag.
        cache.cache_op(C_HWBINV | CACH_PD, virt, phys as u64);
        // Flush L2 → memory to verify data made it out of L1D.
        flush_l2_to_mem(&cache, phys);
        assert_eq!(mem_read(&mem, phys), 0xCAFE_0001, "Hit_WBInv(PD) did not flush dirty data to L2");

        // L1D tag was invalidated — overwrite memory and a subsequent read refills.
        mem_write(&mem, phys, 0xFACE_0002);
        let r = cache.read::<4>(virt, phys as u64);
        assert_eq!(r.data as u32, 0xFACE_0002, "Hit_WBInv(PD) left L1D line valid after invalidate");

        // Hit_WBInvalidate on a non-cached address — should be a no-op (no panic).
        let phys2: u32 = 0x4000;
        let virt2 = kseg0(phys2);
        mem_write(&mem, phys2, 0x1234_ABCD);
        cache.cache_op(C_HWBINV | CACH_PD, virt2, phys2 as u64);
        assert_eq!(mem_read(&mem, phys2), 0x1234_ABCD, "Hit_WBInv(PD) miss should be no-op");
    }

    /// Hit_Invalidate L1D: invalidates L1D without writeback.
    /// The clean line is simply dropped; L2 still holds the original value.
    #[test]
    #[cfg(not(all(feature = "r5k", not(feature = "r5ksc"))))]
    fn cache_op_hit_inv_l1d() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let phys: u32 = 0x5000;
        let virt = kseg0(phys);
        mem_write(&mem, phys, 0xAAAA_BBBB);
        // Read to populate L1D (and L2) cleanly.
        let _ = cache.read::<4>(virt, phys as u64);

        // Hit_Invalidate drops the L1D line (no writeback since it is clean).
        cache.cache_op(C_HINV | CACH_PD, virt, phys as u64);

        // L1D is invalid — next read refills from L2 which still holds 0xAAAA_BBBB.
        let r = cache.read::<4>(virt, phys as u64);
        assert_eq!(r.data as u32, 0xAAAA_BBBB, "Hit_Inv(PD) did not invalidate L1D line");
    }

    /// Index_WBInvalidate L2: flushes and invalidates an L2 line.
    #[test]
    #[cfg(not(all(feature = "r5k", not(feature = "r5ksc"))))]
    fn cache_op_index_wbinv_l2() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let phys: u32 = 0x6000;
        let virt = kseg0(phys);
        // Write into cache — lands in both L1D and L2.
        let _ = cache.write::<4>(virt, phys as u64, 0x5555_AAAA_u64);
        // Flush L1D first (Index_WBInv), so data propagates to L2.
        let dc_set = (phys as usize >> DCache::LINE_SHIFT as usize) & DCache::NUM_LINES_MASK;
        for way in 0..DC_WAYS {
            let nls = DCache::NUM_LINES_SHIFT as usize;
            let ls  = DCache::LINE_SHIFT as usize;
            let idx_addr = (way << (nls + ls)) | (dc_set << ls);
            let v = kseg0(idx_addr as u32);
            cache.cache_op(C_IWBINV | CACH_PD, v, v & 0x1FFF_FFFF);
        }

        // Now Index_WBInvalidate the L2 line — flushes L2 dirty data to memory.
        let l2_set = (phys as usize >> L2Cache::LINE_SHIFT as usize) & (L2_SIZE / L2_LINE - 1);
        let lp = (l2_set * L2_LINE) as u64;
        cache.cache_op(C_IWBINV | CACH_SD, lp, lp);

        // Memory should now have the value.
        assert_eq!(mem_read(&mem, phys), 0x5555_AAAA, "Index_WBInv(SD) did not flush L2 to memory");

        // L2 invalidated — read should refill from memory.
        mem_write(&mem, phys, 0xBEEF_CAFE);
        let r = cache.read::<4>(virt, phys as u64);
        assert_eq!(r.data as u32, 0xBEEF_CAFE, "Index_WBInv(SD) did not invalidate L2 line");
    }

    /// Hit_Invalidate L2: invalidates L2 line.
    /// On R4K (inclusive): also cascades to L1D.
    /// On R5K (non-inclusive): L1D is unaffected.
    #[test]
    #[cfg(not(all(feature = "r5k", not(feature = "r5ksc"))))]
    fn cache_op_hit_inv_l2() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let phys: u32 = 0x7000;
        let virt = kseg0(phys);
        mem_write(&mem, phys, 0xDECA_FBAD);
        // Read into L1D and L2.
        let _ = cache.read::<4>(virt, phys as u64);

        // Hit_Invalidate on the L2 line.
        cache.cache_op(C_HINV | CACH_SD, phys as u64, phys as u64);

        // Overwrite memory.
        mem_write(&mem, phys, 0x1234_5678);

        // On R4K, L2 invalidation cascades to L1D → next read refills from updated memory.
        // On R5K, L1D is unaffected → still holds old value 0xDECA_FBAD (from the read above).
        let r = cache.read::<4>(virt, phys as u64);
        #[cfg(not(feature = "r5k"))]
        assert_eq!(r.data as u32, 0x1234_5678,
            "Hit_Inv(SD) did not cascade invalidation to L1D (R4K inclusive)");
        #[cfg(feature = "r5k")]
        assert_eq!(r.data as u32, 0xDECA_FBAD,
            "Hit_Inv(SD) evicted L1D on R5K (non-inclusive — L1 should be unaffected)");
    }

    /// Hit_WBInvalidate L2: writes back dirty L2 line to memory.
    #[test]
    #[cfg(not(all(feature = "r5k", not(feature = "r5ksc"))))]
    fn cache_op_hit_wbinv_l2() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let phys: u32 = 0x8000;
        let virt = kseg0(phys);
        // Write into cache (dirty in L1D; L2 gets dirty during L1D writeback).
        let _ = cache.write::<4>(virt, phys as u64, 0x1122_3344_u64);
        // Flush L1D to make L2 dirty.
        let dc_set = (phys as usize >> DCache::LINE_SHIFT as usize) & DCache::NUM_LINES_MASK;
        for way in 0..DC_WAYS {
            let nls = DCache::NUM_LINES_SHIFT as usize;
            let ls  = DCache::LINE_SHIFT as usize;
            let idx_addr = (way << (nls + ls)) | (dc_set << ls);
            let v = kseg0(idx_addr as u32);
            cache.cache_op(C_IWBINV | CACH_PD, v, v & 0x1FFF_FFFF);
        }

        // Hit_WBInv L2 — writes dirty L2 to memory and invalidates.
        cache.cache_op(C_HWBINV | CACH_SD, phys as u64, phys as u64);

        assert_eq!(mem_read(&mem, phys), 0x1122_3344, "Hit_WBInv(SD) did not flush dirty L2 to memory");

        // L2 should be invalidated — overwrite memory and next read refills.
        mem_write(&mem, phys, 0x5566_7788);
        let r = cache.read::<4>(virt, phys as u64);
        assert_eq!(r.data as u32, 0x5566_7788, "Hit_WBInv(SD) did not invalidate L2 line");
    }

    /// Index_LoadTag / Index_StoreTag round-trip: stored tag must read back identically.
    #[test]
    #[cfg(not(all(feature = "r5k", not(feature = "r5ksc"))))]
    fn cache_op_ilt_ist_l2() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let phys: u32 = 0x9000;
        let virt = kseg0(phys);
        // Populate L2 by doing a read.
        let _ = cache.read::<4>(virt, phys as u64);

        let l2_set = (phys as usize >> L2Cache::LINE_SHIFT as usize) & (L2_SIZE / L2_LINE - 1);
        let lp = (l2_set * L2_LINE) as u64;

        // Index_LoadTag — read current tag.
        let tag_lo_read = cache.cache_op(C_ILT | CACH_SD, lp, lp);

        // Index_StoreTag — write it back unchanged, then read again.
        // Use phys_addr slot (second u64 arg) as the TagLo value per our calling convention.
        cache.cache_op(C_IST | CACH_SD, lp, tag_lo_read as u64);
        let tag_lo_read2 = cache.cache_op(C_ILT | CACH_SD, lp, lp);

        assert_eq!(tag_lo_read, tag_lo_read2, "ILT/IST(SD) round-trip mismatch");
    }

    /// Two-way independence: Index_WBInv on way 0 should not disturb way 1.
    /// Way selection for index ops: bit 14 of the index address selects the way.
    #[test]
    fn cache_op_two_way_independent() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        // Two addresses that map to the same L1D set but different physical tags.
        // bit14=0 → index address selects way0; bit14=1 → selects way1.
        // phys0=0x1000: set=(0x1000>>5)&511=0x80, bit14=(0x1000>>14)&1=0.
        // phys1=0x5000: set=(0x5000>>5)&511=0x80, bit14=(0x5000>>14)&1=1.
        // For R4K (1-way) both addresses alias to the same set — the test degrades gracefully.
        let phys0: u32 = 0x0000_1000; // set 0x80, bit14=0
        let phys1: u32 = 0x0000_5000; // set 0x80, bit14=1
        // Verify same set and different bit14.
        assert_eq!(
            (phys0 as usize >> DCache::LINE_SHIFT as usize) & DCache::NUM_LINES_MASK,
            (phys1 as usize >> DCache::LINE_SHIFT as usize) & DCache::NUM_LINES_MASK,
            "phys0 and phys1 must map to the same L1D set"
        );
        assert_ne!(phys1 & (1 << 14), 0, "phys1 must have bit14=1");

        let virt0 = kseg0(phys0);
        let virt1 = kseg0(phys1);

        mem_write(&mem, phys0, 0xAAAA_0001);
        mem_write(&mem, phys1, 0xBBBB_0002);

        // Load both into L1D.
        let _ = cache.read::<4>(virt0, phys0 as u64);
        let _ = cache.read::<4>(virt1, phys1 as u64);

        // Index_WBInvalidate using index address with bit14=0 — should evict way0.
        let set = (phys0 as usize >> DCache::LINE_SHIFT as usize) & DCache::NUM_LINES_MASK;
        let inv0 = kseg0((set << DCache::LINE_SHIFT as usize) as u32); // bit14=0
        assert_eq!(inv0 & (1 << 14), 0, "inv0 must have bit14=0");
        cache.cache_op(C_IWBINV | CACH_PD, inv0, inv0 & 0x1FFF_FFFF);

        // Way0 was invalidated — refill brings 0xAAAA_0001 from L2.
        // (L2 still has old value since L1D write was clean → no dirty L2).
        let r0 = cache.read::<4>(virt0, phys0 as u64);
        assert_eq!(r0.data as u32, 0xAAAA_0001, "Way0 not invalidated by Index_WBInv with bit14=0");

        // Way1 should be unaffected — still holds 0xBBBB_0002.
        #[cfg(feature = "r5k")]
        {
            let r1 = cache.read::<4>(virt1, phys1 as u64);
            assert_eq!(r1.data as u32, 0xBBBB_0002,
                "Way1 was incorrectly evicted by Index_WBInv targeting way0");
        }
        #[cfg(not(feature = "r5k"))]
        {
            // R4K single-way: both addresses alias, result is implementation-defined.
            let _ = cache.read::<4>(virt1, phys1 as u64);
        }

        // Index_Invalidate L1I using bit14=1 address — should not affect way0 L1I line.
        let ic_set = (phys1 as usize >> ICache::LINE_SHIFT as usize) & ICache::NUM_LINES_MASK;
        let inv1 = kseg0(((ic_set << ICache::LINE_SHIFT as usize) | (1 << 14)) as u32); // bit14=1
        cache.cache_op(C_IINV | CACH_PI, inv1, inv1 & 0x1FFF_FFFF);
        // No panic = test passes.
    }

    /// Full flush via Index_WBInv then verify memory coherence: same as existing
    /// l1d_random_stress but using the full_flush helper to exercise all ops.
    #[test]
    fn cache_op_full_flush_coherence() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());
        let mut rng = Rng::new(0xfedc_ba98_7654_3210);
        let mut shadow = vec![0u32; MEM_BYTES / 4];

        for _ in 0..200_000 {
            let phys = rng.next_u32() & ADDR_MASK;
            let virt = kseg0(phys);
            let val = rng.next_u32();
            let _ = cache.write::<4>(virt, phys as u64, val as u64);
            shadow[(phys / 4) as usize] = val;
        }

        // Full flush — all dirty data should land in memory.
        full_flush(&cache);

        // Verify every written address is correct in memory.
        for (i, &want) in shadow.iter().enumerate() {
            if want != 0 {
                let got = mem_read(&mem, (i * 4) as u32);
                assert_eq!(got, want,
                    "full_flush coherence failure at phys={:#010x}: got={:#010x} want={:#010x}",
                    i * 4, got, want);
            }
        }
    }

    /// Mixed L1I+L1D coherence: write via L1D, flush that line, fetch via L1I,
    /// confirm L1I sees the updated value. 1M operations.
    #[test]
    fn l1i_l1d_coherence() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());
        let mut rng = Rng::new(0xc0ffee_0b00c);
        let mut shadow = vec![0u32; MEM_BYTES / 4];

        for op in 0..1_000_000 {
            let phys = rng.next_u32() & ADDR_MASK;
            let virt = kseg0(phys);
            match rng.next_u32() % 3 {
                0 => {
                    // L1D write + invalidate L1I so next fetch re-fills.
                    let val = rng.next_u32();
                    let st = cache.write::<4>(virt, phys as u64, val as u64);
                    assert_eq!(st, BUS_OK, "L1D write error phys={:#010x} op={}", phys, op);
                    shadow[(phys / 4) as usize] = val;
                    // Index_WBInv L1D both ways: set index = (phys >> LINE_SHIFT) & NUM_LINES_MASK,
                    // way selected by bit (LINE_SHIFT + log2(NUM_SETS)) = bit 14 for both L1D and L1I.
                    let dc_set = ((phys as usize) >> DCache::LINE_SHIFT) & DCache::NUM_LINES_MASK;
                    let dc_v0 = kseg0((dc_set << DCache::LINE_SHIFT as usize) as u32);
                    let dc_v1 = kseg0(((dc_set << DCache::LINE_SHIFT as usize) | (1 << 14)) as u32);
                    cache.cache_op(C_IINV | CACH_PD, dc_v0, dc_v0 & 0x1FFF_FFFF);
                    cache.cache_op(C_IINV | CACH_PD, dc_v1, dc_v1 & 0x1FFF_FFFF);
                    // L2 hit-writeback (single-way, hit op fine).
                    cache.cache_op(C_HWBINV | CACH_SD, phys as u64, phys as u64);
                    // Index_Inv L1I both ways.
                    let ic_set = ((phys as usize) >> ICache::LINE_SHIFT) & ICache::NUM_LINES_MASK;
                    let ic_v0 = kseg0((ic_set << ICache::LINE_SHIFT as usize) as u32);
                    let ic_v1 = kseg0(((ic_set << ICache::LINE_SHIFT as usize) | (1 << 14)) as u32);
                    cache.cache_op(C_IINV | CACH_PI, ic_v0, ic_v0 & 0x1FFF_FFFF);
                    cache.cache_op(C_IINV | CACH_PI, ic_v1, ic_v1 & 0x1FFF_FFFF);
                    // Verify flush landed in memory.
                    let mem_now = mem.read32(phys).data;
                    assert_eq!(mem_now, val,
                        "post-flush mem wrong at phys={:#010x} op={}: mem={:#010x} want={:#010x}",
                        phys, op, mem_now, val);
                }
                1 => {
                    // L1D read.
                    let r = cache.read::<4>(virt, phys as u64);
                    assert_eq!(r.status, BUS_OK, "L1D read error phys={:#010x} op={}", phys, op);
                    assert_eq!(r.data as u32, shadow[(phys / 4) as usize],
                        "L1D coherence mismatch phys={:#010x} op={}", phys, op);
                }
                _ => {
                    // L1I fetch — backing mem is up to date after the writeback above.
                    let r = cache.fetch(virt, phys as u64);
                    assert_eq!(r.status, EXEC_COMPLETE, "L1I exception phys={:#010x} op={}", phys, op);
                    let got = unsafe { (*r.instr).raw };
                    let want = shadow[(phys / 4) as usize];
                    assert_eq!(got, want,
                        "L1I coherence mismatch phys={:#010x} op={}: got={:#010x} want={:#010x}",
                        phys, op, got, want);
                }
            }
        }
    }

    /// A dirty L1D tag must round-trip to its own address. Dirty used to be packed
    /// into bit 27, which is `raw_ptag` bit 19, i.e. physical address bit 31, so
    /// every dirty line came back 2 GB up. Assert on the XOR so a failure names the
    /// bit that moved.
    #[test]
    fn l1d_tag_dirty_round_trips_address() {
        // 0x2000_0000 is himem; 0x8000_0000 is the address that owns the aliased bit.
        for &addr in &[0x0000_1000u64, 0x0400_0000, 0x2000_0000, 0x8000_0000] {
            // cs 2 with dirty set is the dominant real state: mark_l1d_dirty sets
            // dirty and leaves cs alone.
            for cs in 1u8..=3 {
                let word: u32 = L1DTag::valid(addr, cs, true).into();
                let back = L1DTag::from(word);
                assert_eq!(back.line_addr() ^ addr, 0,
                    "cs={} addr={:#x} round-tripped to {:#x}", cs, addr, back.line_addr());
                assert!(back.dirty, "cs={} addr={:#x} lost the dirty flag", cs, addr);
            }
        }
    }

    #[test]
    fn l1d_tag_invalid_stays_invalid() {
        let word: u32 = L1DTag::default().into();
        assert_eq!(word, 0);
        let back = L1DTag::from(word);
        assert!(!back.is_valid());
        assert!(!back.dirty);
    }

    #[test]
    fn l1d_tag_v0_word_migrates() {
        let addr = 0x2000_1000u64;
        let raw_ptag = (addr >> L1_PTAG_SHIFT) as u32 & L1_PTAG_MASK;

        // The v0 word carried CS and W independently, so migration keeps both.
        let dirty_v0 = (raw_ptag << 8) | (L1D_CS_CLEAN_EXCLUSIVE << 6) | (1 << 27);
        let t = L1DTag::from(migrate_l1d_tag_word_v0(dirty_v0));
        assert_eq!(t.line_addr(), addr);
        assert!(t.dirty);
        assert_eq!(t.cs as u32, L1D_CS_CLEAN_EXCLUSIVE);

        let clean_v0 = (raw_ptag << 8) | (L1D_CS_CLEAN_EXCLUSIVE << 6);
        let t = L1DTag::from(migrate_l1d_tag_word_v0(clean_v0));
        assert_eq!(t.line_addr(), addr);
        assert!(!t.dirty);
        assert_eq!(t.cs as u32, L1D_CS_CLEAN_EXCLUSIVE);
    }
}
