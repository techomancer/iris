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

// Cache geometry is const-generic, not a cargo feature: each CPU model is its own
// monomorphisation, so every index/mask below still folds to a literal.

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
        Self {
            ptag: (phys_addr & !0xFFF) | 1,
            cs,
            dirty,
        }
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
    /// tcache: this L2 line's `l2.instrs` slots hold valid decoded
    /// instructions for this physical line.
    ///
    /// Under tcache, L2 *data* is bypassed entirely — nothing is copied in or
    /// written back — but on R4400 the decode slots live in `l2.instrs`
    /// indexed by physical address, so L1I fills still need L2 to hold them.
    /// A line filled on behalf of L1D therefore has valid tags but *no* valid
    /// instructions, and an L1I fill must not mistake it for a usable hit.
    /// Set only by an instruction-origin fill; see `fill_l2_line`'s
    /// `fill_instructions` argument.
    pub bool, has_code, set_has_code: 25;
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
/// Per-model settings that are not cache behaviour: ISA level, CP0/CP1 identity, TLB size.
/// Const, so every use folds at monomorphisation instead of costing a runtime check.
/// tcache instrumentation: how often the transparency gate was consulted and
/// how often it said yes. Developer builds only; printed by `tc_stats()`.
#[cfg(all(feature = "tcache", feature = "developer"))]
pub static TC_PROBES: AtomicU64 = AtomicU64::new(0);
#[cfg(all(feature = "tcache", feature = "developer"))]
pub static TC_HITS: AtomicU64 = AtomicU64::new(0);
/// Cacheable accesses that fell back to the bus because the bitmap did not
/// claim the address. Should be ~0 in a healthy run.
#[cfg(all(feature = "tcache", feature = "developer"))]
pub static TC_BUS_READS: AtomicU64 = AtomicU64::new(0);
#[cfg(all(feature = "tcache", feature = "developer"))]
pub static TC_BUS_WRITES: AtomicU64 = AtomicU64::new(0);

/// tcache: (probes, hits) since process start.
#[cfg(all(feature = "tcache", feature = "developer"))]
pub fn tc_stats() -> (u64, u64) {
    (TC_PROBES.load(Ordering::Relaxed), TC_HITS.load(Ordering::Relaxed))
}

pub trait CpuModel: MipsCache {
    /// MIPS IV opcodes decode rather than raising Reserved Instruction.
    const MIPS4: bool;
    /// CP0 PRId reset value.
    const PRID: u32;
    /// CP1 FIR reset value.
    const FIR: u32;
    /// JTLB entries.
    const TLB_ENTRIES: usize;
    /// Name as the guest and the benchmark report see it.
    const NAME: &'static str;
}

pub trait MipsCache: Send + Sync {
    /// Share the L1-I hit/fetch counters with the status display. No-op where absent.
    fn set_l1i_counters(&mut self, _hit: Arc<AtomicU64>, _fetch: Arc<AtomicU64>) {}

    /// tcache: point the cache at ppmem's window and mapped-region bitmap, so
    /// cacheable RAM accesses read/write RAM directly instead of copying line
    /// data (docs/tcache-design.md). Default no-op — a cache with no
    /// transparent path simply ignores it.
    ///
    /// # Safety
    /// `base` must be ppmem's window base and `bitmap` a live `u64` that
    /// outlives this cache.
    #[cfg(feature = "tcache")]
    unsafe fn set_tcache_window(&self, _base: *mut u8) {}

    /// tcache: pointer to this cache's inline mapped-region bitmap, for
    /// `MappedMemory::set_bitmap_sink2`. Null when the cache has no such field.
    #[cfg(feature = "tcache")]
    fn tcache_bitmap_ptr(&self) -> *mut u64 { std::ptr::null_mut() }

    /// tcache + jitv2: hand the cache ppmem's generation-window base, so window
    /// writes bump the per-page counter jitv2 validates compiled code against.
    /// Default no-op.
    ///
    /// # Safety
    /// `gen_base` must be ppmem's gen window base and outlive this cache.
    #[cfg(all(feature = "tcache", feature = "jitv2"))]
    unsafe fn set_tcache_gen_window(&self, _gen_base: *mut AtomicU64) {}

    /// L1/L2 geometry, so CP0 Config reports this model rather than a build-time constant.
    const IC_SIZE: usize;
    const IC_LINE: usize;
    const IC_WAYS: usize;
    const DC_SIZE: usize;
    const DC_LINE: usize;
    const DC_WAYS: usize;
    const L2_SIZE: usize;
    const L2_LINE: usize;

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

    /// Write back dirty L1-D (and, if present, L2) lines covering
    /// `[phys_addr, phys_addr + size)` to memory, without invalidating them.
    ///
    /// `virt_addr`, if given, is the virtual address corresponding to
    /// `phys_addr` (same page offset) — L1-D is VIPT, so a real virtual
    /// address lets each line be found directly, the same way a normal
    /// access would. When `virt_addr` is `None` (phys-only), every possible
    /// virtual alias of each physical line is checked instead (L1-D indexes
    /// on more bits than the page offset guarantees are VA==PA, so a purely
    /// physical address cannot rule any alias out) — this may write back
    /// lines that happen to tag-match but do not actually belong to the
    /// running program's mapping, which is fine for a diagnostic/flush op.
    ///
    /// Returns the number of dirty lines actually written back (L1-D + L2).
    fn writeback(&self, _virt_addr: Option<u64>, _phys_addr: u64, _size: u64) -> usize { 0 }

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
pub struct PassthroughCacheOf<const MIPS4: bool> {
    downstream: Arc<dyn BusDevice>,
    llbit: UnsafeCell<bool>,
    lladdr: UnsafeCell<u32>,
    /// Scratch slot for fetch() — no actual caching, just a place to decode into.
    fetch_scratch: UnsafeCell<DecodedInstr>,
}

// Safety: Single-threaded access only (CPU thread)
unsafe impl<const MIPS4: bool> Send for PassthroughCacheOf<MIPS4> {}
unsafe impl<const MIPS4: bool> Sync for PassthroughCacheOf<MIPS4> {}

/// MIPS III passthrough — the default for tests that do not care about ISA level.
pub type PassthroughCache = PassthroughCacheOf<false>;
/// MIPS IV passthrough, for tests that exercise MIPS IV opcodes.
pub type PassthroughCacheM4 = PassthroughCacheOf<true>;

impl<const MIPS4: bool> PassthroughCacheOf<MIPS4> {
    pub fn new(downstream: Arc<dyn BusDevice>) -> Self {
        Self {
            downstream,
            llbit: UnsafeCell::new(false),
            lladdr: UnsafeCell::new(0),
            fetch_scratch: UnsafeCell::new(DecodedInstr::default()),
        }
    }
}

impl<const MIPS4: bool> From<Arc<dyn BusDevice>> for PassthroughCacheOf<MIPS4> {
    fn from(downstream: Arc<dyn BusDevice>) -> Self {
        Self::new(downstream)
    }
}

impl<const MIPS4: bool> CpuModel for PassthroughCacheOf<MIPS4> {
    const MIPS4: bool = MIPS4;
    const PRID: u32 = if MIPS4 { 0x0000_2321 } else { 0x0000_0440 };
    const FIR: u32 = if MIPS4 { 0x0000_2300 } else { 0x0000_0500 };
    const TLB_ENTRIES: usize = 48;
    const NAME: &'static str = "passthrough";
}

impl<const MIPS4: bool> MipsCache for PassthroughCacheOf<MIPS4> {
    const IC_SIZE: usize = 0;
    const IC_LINE: usize = 0;
    const IC_WAYS: usize = 1;
    const DC_SIZE: usize = 0;
    const DC_LINE: usize = 0;
    const DC_WAYS: usize = 1;
    const L2_SIZE: usize = 0;
    const L2_LINE: usize = 0;

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
             const TAGS: usize, const DATA: usize, const NINSTRS: usize> {
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
            const TAGS: usize, const DATA: usize, const NINSTRS: usize> Send for Cache<TAG, SIZE, LINE, WAYS, KIND, TAGS, DATA, NINSTRS> {}
unsafe impl<TAG, const SIZE: usize, const LINE: usize, const WAYS: usize, const KIND: u8,
            const TAGS: usize, const DATA: usize, const NINSTRS: usize> Sync for Cache<TAG, SIZE, LINE, WAYS, KIND, TAGS, DATA, NINSTRS> {}

impl<TAG: Default + Copy, const SIZE: usize, const LINE: usize, const WAYS: usize, const KIND: u8,
     const TAGS: usize, const DATA: usize, const NINSTRS: usize> Cache<TAG, SIZE, LINE, WAYS, KIND, TAGS, DATA, NINSTRS> {
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
        // NINSTRS = decoded-instruction slots this level owns (0 when it owns none).
        let instrs: Vec<DecodedInstr> = (0..NINSTRS).map(|_| DecodedInstr::default()).collect();
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
pub struct CpuCache<
    const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize,
> {
    downstream: Arc<dyn BusDevice>,

    // L1 Instruction Cache (16 KB, 16-byte lines)
    ic: ICacheT<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS>,

    // L1 Data Cache (16 KB, 16-byte lines)
    dc: DCacheT<DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA>,

    // L2 Unified Cache (1 MB, 128-byte lines)
    l2: L2CacheT<L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS>,

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

    // 2-way L1I owns its decode slots (non-inclusive L2); empty on a direct-mapped model.
    ic_instrs: CacheVec<DecodedInstr>,
    // Per-set LRU bitmaps for 2-way L1I/L1D; empty on a direct-mapped model.
    ic_lru: UnsafeCell<Box<[u64]>>,
    dc_lru: UnsafeCell<Box<[u64]>>,

    /// tcache: base of ppmem's 4GB window, or null when unavailable.
    /// A transparent access reads/writes `tc_base + phys` directly instead of
    /// copying the line into `dc.data`. See docs/tcache-design.md §3.
    #[cfg(feature = "tcache")]
    tc_base: UnsafeCell<*mut u8>,
    /// tcache: the live mapped-region bitmap, held **inline**.
    ///
    /// `PpMemSpace` writes through a pointer to this field on every remap (see
    /// `MappedMemory::set_bitmap_sink2`), so the hot path is a single load from
    /// the cache object it is already touching — not a pointer chase into
    /// `PpMemSpace`. Zero until a window is attached, which reads as "nothing
    /// is directly mapped" and sends every access to the bus.
    #[cfg(feature = "tcache")]
    tc_bitmap: UnsafeCell<u64>,
    /// tcache + jitv2: base of ppmem's generation window, one `AtomicU64` per
    /// 4KB page (`gen_base + (phys >> 12)`).
    ///
    /// A window write stores straight into RAM, bypassing `BusDevice` — and so
    /// bypassing the gen bump that `PpMemory`'s write methods perform. Without
    /// bumping it here, jitv2 would go on executing compiled code for a page
    /// the guest just modified.
    #[cfg(all(feature = "tcache", feature = "jitv2"))]
    tc_gen: UnsafeCell<*mut AtomicU64>,

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

unsafe impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize> Send for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES> {}
unsafe impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize> Sync for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES> {}

// Per-level cache types, parameterised so each CPU model monomorphises its own.
type ICacheT<const S: usize, const L: usize, const W: usize, const T: usize> =
    Cache<L1ITag, S, L, W, { CacheKind::Insn as u8 }, T, 0, 0>;
type DCacheT<const S: usize, const L: usize, const W: usize, const T: usize, const D: usize> =
    Cache<L1DTag, S, L, W, { CacheKind::Data as u8 }, T, D, 0>;
type L2CacheT<const S: usize, const L: usize, const T: usize, const D: usize, const N: usize> =
    Cache<L2Tag, S, L, 1, { CacheKind::L2 as u8 }, T, D, N>;

/// SGI Indy R4400: direct-mapped 16K L1s, 1 MB unified L2 owning the decode slots.
pub type R4400Cache = CpuCache<16384, 16, 1, 1024,
                               16384, 16, 1, 1024, 2048,
                               1048576, 128, 8192, 131072, 262144, true,
                               false, 0x0000_0440, 0x0000_0500, 48>;
/// SGI Indy R5000: 2-way 32K L1s, no secondary cache; L1I owns its decode slots.
pub type R5000Cache = CpuCache<32768, 32, 2, 1024,
                               32768, 32, 2, 1024, 4096,
                               128, 128, 1, 16, 0, false,
                               true, 0x0000_2321, 0x0000_2300, 48>;

impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize> CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES> {
    // Model discriminator: folds to a literal, so it replaces #[cfg(feature = "r5k")].
    const IS_R5K: bool = IC_WAYS == 2;
    // Logical L2 size; 0 means the model has no secondary cache.
    pub const L2_SIZE: usize = if HAS_L2 { L2_CACHE_SIZE } else { 0 };

    const IC_NUM_SETS: usize = IC_SIZE / IC_LINE / IC_WAYS;
    const DC_NUM_SETS: usize = DC_SIZE / DC_LINE / DC_WAYS;

    const IC_LINE_SHIFT: u32 = IC_LINE.trailing_zeros();
    const IC_LINE_MASK: usize = IC_LINE - 1;
    const IC_NUM_LINES: usize = Self::IC_NUM_SETS;
    const IC_NUM_LINES_SHIFT: u32 = Self::IC_NUM_SETS.trailing_zeros();
    const IC_NUM_LINES_MASK: usize = Self::IC_NUM_SETS - 1;
    const IC_INSTRS_PER_LINE: usize = IC_LINE / 4;
    const IC_INSTR_SHIFT: u32 = Self::IC_LINE_SHIFT - 2;
    const IC_INSTR_MASK: usize = Self::IC_INSTRS_PER_LINE - 1;
    const IC_CHUNKS_PER_LINE: usize = IC_LINE / 8;
    const IC_CHUNKS_PER_LINE_SHIFT: u32 = Self::IC_LINE_SHIFT - 3;

    const DC_LINE_SHIFT: u32 = DC_LINE.trailing_zeros();
    const DC_LINE_MASK: usize = DC_LINE - 1;
    const DC_NUM_LINES: usize = Self::DC_NUM_SETS;
    const DC_NUM_LINES_SHIFT: u32 = Self::DC_NUM_SETS.trailing_zeros();
    const DC_NUM_LINES_MASK: usize = Self::DC_NUM_SETS - 1;
    const DC_INSTRS_PER_LINE: usize = DC_LINE / 4;
    const DC_INSTR_SHIFT: u32 = Self::DC_LINE_SHIFT - 2;
    const DC_INSTR_MASK: usize = Self::DC_INSTRS_PER_LINE - 1;
    const DC_CHUNKS_PER_LINE: usize = DC_LINE / 8;
    const DC_CHUNKS_PER_LINE_SHIFT: u32 = Self::DC_LINE_SHIFT - 3;

    const L2_LINE_SHIFT: u32 = L2_LINE.trailing_zeros();
    const L2_LINE_MASK: usize = L2_LINE - 1;
    const L2_NUM_LINES: usize = L2_CACHE_SIZE / L2_LINE;
    const L2_NUM_LINES_SHIFT: u32 = (L2_CACHE_SIZE / L2_LINE).trailing_zeros();
    const L2_NUM_LINES_MASK: usize = (L2_CACHE_SIZE / L2_LINE) - 1;
    const L2_INSTRS_PER_LINE: usize = L2_LINE / 4;
    const L2_INSTR_SHIFT: u32 = Self::L2_LINE_SHIFT - 2;
    const L2_INSTR_MASK: usize = Self::L2_INSTRS_PER_LINE - 1;
    const L2_CHUNKS_PER_LINE: usize = L2_LINE / 8;
    const L2_CHUNKS_PER_LINE_SHIFT: u32 = Self::L2_LINE_SHIFT - 3;

    pub fn new(downstream: Arc<dyn BusDevice>) -> Self {
        let ic = ICacheT::<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS>::new();
        let dc = DCacheT::<DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA>::new();
        let l2 = L2CacheT::<L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS>::new();

        #[cfg(feature = "debug_cache")]
        let (debug_l1d_line, debug_l2_line, debug_companion_l1d_line, debug_companion_l2_line,
             debug_l1d_idx, debug_l2_idx, debug_companion_l2_idx) = {
            if let Some(addr) = DEBUG_TRACK_ADDR {
                let l1_line_mask = Self::DC_LINE_MASK as u64;
                let l2_line_mask = Self::L2_LINE_MASK as u64;
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
            #[cfg(feature = "tcache")]
            tc_base: UnsafeCell::new(std::ptr::null_mut()),
            #[cfg(feature = "tcache")]
            tc_bitmap: UnsafeCell::new(0),
            #[cfg(all(feature = "tcache", feature = "jitv2"))]
            tc_gen: UnsafeCell::new(std::ptr::null_mut()),
            #[cfg(feature = "r5ksc_triton")]
            l2_enabled: false, // starts disabled; PROM enables via CONFIG_SE
            // Const-guarded: a direct-mapped model allocates none of these.
            ic_instrs: CacheVec::new(if Self::IS_R5K {
                (0..IC_WAYS * Self::IC_NUM_SETS * (IC_LINE / 4))
                    .map(|_| DecodedInstr::default()).collect()
            } else { Vec::new() }),
            ic_lru: UnsafeCell::new(
                if Self::IS_R5K { vec![0u64; Self::IC_NUM_SETS.div_ceil(64)] } else { Vec::new() }.into_boxed_slice()),
            dc_lru: UnsafeCell::new(
                if Self::IS_R5K { vec![0u64; Self::DC_NUM_SETS.div_ceil(64)] } else { Vec::new() }.into_boxed_slice()),
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


impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize> From<Arc<dyn BusDevice>> for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES> {
    fn from(downstream: Arc<dyn BusDevice>) -> Self {
        Self::new(downstream)
    }
}

impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize> CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES> {
    /// Check if we're tracking this physical address (for debug purposes)
    #[cfg(feature = "debug_cache")]
    #[inline]
    fn is_tracking_l1d(&self, phys_addr: u64) -> bool {
        DEBUG_TRACK_ADDR.is_some() && {
            let line = phys_addr & !(Self::DC_LINE_MASK as u64);
            line == self.debug_l1d_line || line == self.debug_companion_l1d_line
        }
    }

    #[cfg(feature = "debug_cache")]
    #[inline]
    fn is_tracking_l2(&self, phys_addr: u64) -> bool {
        DEBUG_TRACK_ADDR.is_some() && {
            let line = phys_addr & !(Self::L2_LINE_MASK as u64);
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
            let line = phys_addr & !(Self::DC_LINE_MASK as u64);
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
        let line = phys_addr & !(Self::DC_LINE_MASK as u64);
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
    /// - R4K / r5ksc (external): always true when Self::L2_SIZE > 0.
    /// - r5ksc_triton: gated by CONFIG_SE (l2_enabled field).
    /// - r5k without r5ksc: always false (no L2).
    #[inline]
    fn l2_active(&self) -> bool {
        HAS_L2
    }

    /// tcache: point the cache at ppmem's window and mapped-region bitmap.
    ///
    /// Call once `Physical` is at its final address; both pointers are stable
    /// for the process lifetime thereafter. Until this is called `tc_base` is
    /// null and every access takes the ordinary real-cache path, so tcache is
    /// inert rather than broken on a machine without ppmem.
    ///
    /// # Safety
    /// `base` must be ppmem's window base and `bitmap` a live `u64` that
    /// outlives this cache.
    #[cfg(feature = "tcache")]
    pub unsafe fn set_tcache_window_impl(&self, base: *mut u8) {
        unsafe { *self.tc_base.get() = base };
    }

    /// tcache: pointer to this cache's inline bitmap field, for
    /// `MappedMemory::set_bitmap_sink2` to publish through on every remap.
    #[cfg(feature = "tcache")]
    pub fn tc_bitmap_ptr(&self) -> *mut u64 {
        self.tc_bitmap.get()
    }

    /// tcache + jitv2: hand the cache ppmem's generation-window base so window
    /// writes can bump the per-page counter the JIT validates against.
    ///
    /// # Safety
    /// `gen_base` must be ppmem's gen window base, valid for the process
    /// lifetime.
    #[cfg(all(feature = "tcache", feature = "jitv2"))]
    pub unsafe fn set_tcache_gen_window_impl(&self, gen_base: *mut AtomicU64) {
        unsafe { *self.tc_gen.get() = gen_base };
    }

    /// tcache + jitv2: bump the generation counter for `phys_addr`'s 4KB page.
    ///
    /// Mirrors what `PpMemory::write*` does internally. Only needed on the
    /// window path — the bus path goes through `BusDevice` and bumps there.
    #[cfg(feature = "tcache")]
    #[inline(always)]
    fn tc_bump_gen(&self, _phys_addr: u64) {
        #[cfg(feature = "jitv2")]
        {
            let g = unsafe { *self.tc_gen.get() };
            if !g.is_null() {
                let page = (_phys_addr >> 12) as usize;
                unsafe { (*g.add(page)).fetch_add(1, Ordering::Relaxed) };
            }
        }
    }

    /// tcache: is `phys_addr` served directly out of ppmem's window?
    ///
    /// True only when ppmem is present *and* the address sits in a fully-mapped
    /// 64MB region — the same one-shift test the CPU's own fast path uses. The
    /// bitmap never claims a region containing MMIO, so a true result also
    /// means "this is RAM". Everything else falls back to the real-cache path.
    #[cfg(feature = "tcache")]
    #[inline(always)]
    fn tc_transparent(&self, phys_addr: u64) -> bool {
        #[cfg(feature = "developer")]
        {
            TC_PROBES.fetch_add(1, Ordering::Relaxed);
        }
        // `phys_addr` is a 32-bit physical address by construction — anything
        // larger is a bug upstream, and the shift below will panic in debug
        // rather than silently pretending the address is un-mapped.
        let bits = unsafe { *self.tc_bitmap.get() };
        let hit = bits & (1u64 << (phys_addr >> crate::ppmem::BITMAP_SHIFT)) != 0;
        #[cfg(feature = "developer")]
        if hit {
            TC_HITS.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    /// tcache: host pointer for a transparent physical address.
    #[cfg(feature = "tcache")]
    #[inline(always)]
    fn tc_ptr(&self, phys_addr: u64) -> *mut u8 {
        unsafe { (*self.tc_base.get()).add(phys_addr as usize) }
    }

    /// tcache: read `ACC` bytes straight out of ppmem's window.
    ///
    /// **ppmem's layout is not the cache's layout.** ppmem stores u32 words in
    /// native host order — MIPS byte `i` at host offset `i ^ 3`, halfword at
    /// `(i>>1) ^ 1`, word at `i>>2` unswizzled, u64 as `rotate_left(32)` — an
    /// effectively 4-byte swizzle. The cache's `dc_read`/`dc_write` use an
    /// 8-byte one (`^7`/`^3`/`^1`) because they store u64s natively. Mixing the
    /// two silently corrupts sub-word accesses, so these accessors deliberately
    /// mirror `PpMemory`'s `BusDevice` impl rather than `Cache::dc_read`.
    /// (`ppmem_byte_layout_is_4byte_swizzled` pins the layout down.)
    #[cfg(feature = "tcache")]
    #[inline(always)]
    fn tc_read<const ACC: usize>(&self, phys_addr: u64) -> u64 {
        // The bitmap answers *how* to reach RAM, not whether this line is
        // cached: either way the data lives in RAM. Direct pointer when the
        // region is mapped, bus call when it is not — mapped is the common
        // case by a wide margin, so it is the fall-through arm.
        if self.tc_transparent(phys_addr) {
            let base = unsafe { *self.tc_base.get() };
            let off = phys_addr as usize;
            unsafe {
                if ACC == 8 {
                    (*(base.add(off) as *const u64)).rotate_left(32)
                } else if ACC == 4 {
                    *(base.add(off) as *const u32) as u64
                } else if ACC == 2 {
                    *((base as *const u16).add((off >> 1) ^ 1)) as u64
                } else {
                    *base.add(off ^ 3) as u64
                }
            }
        } else {
            #[cfg(feature = "developer")]
            {
                let n = TC_BUS_READS.fetch_add(1, Ordering::Relaxed);
                if n < 20 {
                    eprintln!("[tcache] BUS READ{} phys={:#010x}", ACC * 8, phys_addr);
                }
            }
            let a = phys_addr as u32;
            if ACC == 8 {
                self.downstream.read64(a).data
            } else if ACC == 4 {
                self.downstream.read32(a).data as u64
            } else if ACC == 2 {
                self.downstream.read16(a).data as u64
            } else {
                self.downstream.read8(a).data as u64
            }
        }
    }

    /// tcache: write `ACC` bytes straight into ppmem's window.
    /// Same layout contract as [`Self::tc_read`].
    #[cfg(feature = "tcache")]
    #[inline(always)]
    fn tc_write<const ACC: usize>(&self, phys_addr: u64, val: u64) {
        // See `tc_read`: window when mapped, bus otherwise — always RAM.
        if self.tc_transparent(phys_addr) {
            let base = unsafe { *self.tc_base.get() };
            let off = phys_addr as usize;
            unsafe {
                if ACC == 8 {
                    *(base.add(off) as *mut u64) = val.rotate_left(32);
                } else if ACC == 4 {
                    *(base.add(off) as *mut u32) = val as u32;
                } else if ACC == 2 {
                    *((base as *mut u16).add((off >> 1) ^ 1)) = val as u16;
                } else {
                    *base.add(off ^ 3) = val as u8;
                }
            }
            // The store above went straight to RAM, skipping `BusDevice` — so
            // the gen bump it would have performed has to happen here.
            self.tc_bump_gen(phys_addr);
        } else {
            #[cfg(feature = "developer")]
            {
                let n = TC_BUS_WRITES.fetch_add(1, Ordering::Relaxed);
                if n < 20 {
                    eprintln!("[tcache] BUS WRITE{} phys={:#010x}", ACC * 8, phys_addr);
                }
            }
            let a = phys_addr as u32;
            if ACC == 8 {
                self.downstream.write64(a, val);
            } else if ACC == 4 {
                self.downstream.write32(a, val as u32);
            } else if ACC == 2 {
                self.downstream.write16(a, val as u16);
            } else {
                self.downstream.write8(a, val as u8);
            }
        }
    }

    /// Triton only: set L2 enable state from CONFIG_SE. On off→on transition, invalidate
    /// all L2 lines so stale data from before the disable window isn't used.
    #[cfg(feature = "r5ksc_triton")]
    pub fn set_l2_enabled(&mut self, enabled: bool) {
        let was = self.l2_enabled;
        self.l2_enabled = enabled;
        if enabled && !was {
            for idx in 0..Self::L2_NUM_LINES {
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
    #[inline(always)]
    unsafe fn lru_get(bm: *const Box<[u64]>, set: usize) -> bool {
        let slice: &[u64] = &*bm;
        (*slice.get_unchecked(set >> 6) >> (set & 63)) & 1 != 0
    }

    /// Set or clear LRU bit for `set` in a packed u64 bitmap.
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
    fn l2_idx_to_l1_base_idx<TAG: Default + Copy, const L1_SIZE: usize, const L1_LINE: usize, const L1_WAYS: usize, const L1_KIND: u8, const L1_TAGS: usize, const L1_DATA: usize, const L1_NINSTRS: usize>(
        &self, l2_idx: usize, pidx: u32, _l1: &Cache<TAG, L1_SIZE, L1_LINE, L1_WAYS, L1_KIND, L1_TAGS, L1_DATA, L1_NINSTRS>
    ) -> usize {
        // Physical bits of the L2 line start address that are below bit 12 (page boundary)
        // These bits are the same in VA and PA, so we can derive them from the L2 index.
        let phys_sub_bits = (l2_idx << Self::L2_LINE_SHIFT as usize) & 0xFFF;
        // Reconstruct the virtual address bits used for L1 indexing
        let virt_index_bits = ((pidx as usize) << L2_PIDX_VADDR_SHIFT as usize) | phys_sub_bits;
        (virt_index_bits >> Cache::<TAG, L1_SIZE, L1_LINE, L1_WAYS, L1_KIND, L1_TAGS, L1_DATA, L1_NINSTRS>::LINE_SHIFT as usize)
            & Cache::<TAG, L1_SIZE, L1_LINE, L1_WAYS, L1_KIND, L1_TAGS, L1_DATA, L1_NINSTRS>::NUM_LINES_MASK
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
            let phys_addr = l1_tag_to_phys(tag, (idx << Self::IC_LINE_SHIFT) as u64);
            if self.is_tracking_l1d(phys_addr) {
                println!("[CACHE DEBUG] invalidate_l1i_line: {} idx=0x{:x}, phys_addr=0x{:08x}, ptag=0x{:010x}",
                         self.tracking_label(phys_addr), idx, phys_addr, tag.line_addr());
            }
        }

        if cascade && tag.is_valid() {
            let phys_addr = l1_tag_to_phys(tag, (idx << Self::IC_LINE_SHIFT) as u64);
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
                let phys_addr = l1d_tag_to_phys(tag, (idx << Self::DC_LINE_SHIFT) as u64);
                println!("[CACHE DEBUG] invalidate_l1d_line: {} idx=0x{:x}, phys_addr=0x{:08x}, ptag=0x{:010x}, cs={}, coherent={}",
                         self.tracking_label(phys_addr), idx, phys_addr, tag.line_addr(), tag.cs, coherent);
            } else {
                println!("[CACHE DEBUG] invalidate_l1d_line: idx=0x{:x} (already invalid)", idx);
            }
        }

        // Only clear llbit for software-initiated coherency invalidations, not hardware fills.
        // On a uniprocessor R4000 there are no external snoops; llbit survives capacity evictions.
        if coherent && tag.cs != L1D_CS_INVALID as u8 {
            let phys_addr = l1d_tag_to_phys(tag, (idx << Self::DC_LINE_SHIFT) as u64);
            self.check_and_clear_llbit(phys_addr, Self::DC_LINE_MASK);
        }

        if cascade && tag.cs != L1D_CS_INVALID as u8 {
            let phys_addr = l1d_tag_to_phys(tag, (idx << Self::DC_LINE_SHIFT) as u64);
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
                let phys_base = l2_tag_to_phys(l2_tag, (idx << Self::L2_LINE_SHIFT) as u64);
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
        let phys_base = l2_tag_to_phys(l2_tag, (idx << Self::L2_LINE_SHIFT) as u64);

        // NOTE: do NOT clear llbit here. On R4000, llbit tracks L1-D state only.
        // An L2 eviction is not a coherency action and must not break LL/SC.

        // R4K inclusive policy: cascade L2 eviction to L1.
        // R5K caches are non-inclusive — L2 evictions do not affect L1.
        if !Self::IS_R5K {
        {
            // Check L1-I for any lines from this L2 line.
            // L1-I is VIPT so we must reconstruct the virtual index from pidx + physical sub-bits.
            let pidx = l2_tag.pidx();
            let l1i_lines_per_l2 = 1 << (Self::L2_LINE_SHIFT - Self::IC_LINE_SHIFT);
            let ic_base_idx = self.l2_idx_to_l1_base_idx(idx, pidx, &self.ic);
            for i in 0..l1i_lines_per_l2 {
                let ic_idx = (ic_base_idx + i) & Self::IC_NUM_LINES_MASK;
                let phys_addr = phys_base + ((i as u64) << Self::IC_LINE_SHIFT);
                let ic_tag: L1ITag = self.ic.get_tag(ic_idx);
                if ic_tag.matches_phys(phys_addr) {
                    self.invalidate_l1i_line(ic_idx, false);
                }
            }

            // Check L1-D for any lines from this L2 line.
            // L1-D is VIPT so we must reconstruct the virtual index from pidx + physical sub-bits.
            let l1d_lines_per_l2 = 1 << (Self::L2_LINE_SHIFT - Self::DC_LINE_SHIFT);
            let dc_base_idx = self.l2_idx_to_l1_base_idx(idx, pidx, &self.dc);
            for i in 0..l1d_lines_per_l2 {
                let dc_idx = (dc_base_idx + i) & Self::DC_NUM_LINES_MASK;
                let phys_addr = phys_base + ((i as u64) << Self::DC_LINE_SHIFT);
                let dc_tag: L1DTag = self.dc.get_tag(dc_idx);

                if dc_tag.matches_phys(phys_addr) {
                    self.invalidate_l1d_line(dc_idx, false, false); // hardware cascade, not coherent
                }
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
        for i in 0..Self::L2_NUM_LINES {
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

        // tcache: every cacheable RAM line is transparent, so `dc.data` and
        // `l2.data` do not exist as storage — RAM is the store. The cache state
        // machine below is untouched: tags, CLEAN->DIRTY, the `l2.instrs`
        // re-sync, the L1 cascade and the `cascade` branch all run exactly as
        // they do without tcache. Only the *data transfers* are elided, since
        // there is nothing to transfer.
        // Reconstruct physical address from tag
        let phys_addr = l1d_tag_to_phys(tag, (l1_idx << Self::DC_LINE_SHIFT) as u64);

        // tcache: the cache arrays are not storage — RAM is, so there is
        // nothing to copy out on a writeback. Both models.
        #[cfg(feature = "tcache")]
        let transparent = true;
        #[cfg(not(feature = "tcache"))]
        let transparent = false;

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
        if Self::IS_R5K {
        if !self.l2_active() || l2_tag.cs() == L2_CS_INVALID || l2_tag.ptag() != l2_ptag {
            // tcache: nothing to write — a transparent line has no `dc.data`,
            // its storage is RAM and RAM already holds the current bytes. The
            // tag demotion below still runs.
            if !transparent {
                let dc_data = self.dc.data();
                let l1_start_chunk = l1_idx << Self::DC_CHUNKS_PER_LINE_SHIFT;
                let line_base = phys_addr & !(Self::DC_LINE_MASK as u64);
                let src = &dc_data[l1_start_chunk..l1_start_chunk + Self::DC_CHUNKS_PER_LINE];
                self.downstream.write_block(line_base as u32, src);
            }
            let mut dc_tag: L1DTag = self.dc.get_tag(l1_idx);
            dc_tag.dirty = false;
            if dc_tag.cs == L1D_CS_DIRTY_EXCLUSIVE as u8 { dc_tag.cs = L1D_CS_CLEAN_EXCLUSIVE as u8; }
            self.dc.set_tag(l1_idx, dc_tag);
            return true;
        }
        }

        // R4K (inclusive): L2 must always hold the line — fail if not.
        if !Self::IS_R5K {
        if l2_tag.ptag() != l2_ptag {
            return false; // shouldn't happen on inclusive R4K
        }
        }

        // L2 has the line: write data from L1-D into L2.
        let dc_data = self.dc.data();
        let l2_data = self.l2.data_mut();

        let l1_start_chunk = l1_idx << Self::DC_CHUNKS_PER_LINE_SHIFT;

        let l2_line_base = l2_idx << Self::L2_CHUNKS_PER_LINE_SHIFT;
        let offset_in_l2_line = ((phys_addr & Self::L2_LINE_MASK as u64) >> 3) as usize;

        // tcache: no data moves here at all. `dc.data` holds nothing (writes
        // went to RAM) and `l2.data` is not storage, so there is nothing to
        // copy. The tag/state work below still runs unchanged.
        if !transparent {
            for i in 0..Self::DC_CHUNKS_PER_LINE {
                l2_data[l2_line_base + offset_in_l2_line + i] = dc_data[l1_start_chunk + i];
            }
        }

        #[cfg(feature = "debug_cache")]
        {
            if self.is_tracking_l1d(phys_addr) || self.is_tracking_l2_idx(l2_idx) {
                println!("[CACHE DEBUG] writeback_l1d_line: wrote {} chunks to L2 idx=0x{:x} offset=0x{:x}",
                         Self::DC_CHUNKS_PER_LINE, l2_idx, offset_in_l2_line);
                for i in 0..Self::DC_CHUNKS_PER_LINE {
                    println!("    [{}] addr=0x{:08x} val=0x{:016x}",
                             i, phys_addr + ((i as u64) << 3), dc_data[l1_start_chunk + i]);
                }
            }
        }

        // R4K: sync l2.instrs for the updated region so fetch() sees fresh instruction words.
        // R5K: l2.instrs is empty; ic_instrs will be re-filled from l2.data on next L1I miss.
        //
        // tcache: skipped. `dc.data` has nothing to sync from, and the single
        // sanctioned divergence is that `l2.instrs` is repopulated **only when
        // L1I fills**, reading RAM. `tc_invalidate_l2_code` on the write side
        // clears `has_code` so that refill actually happens.
        // Also recomputes delay-slot fusion lookahead (FLAG_IMM_IS_NEXT) inline,
        // where raw values are already hot — see fill_l2_line for why this must
        // not happen in fetch()'s hot path instead. r0/r1 (one chunk) are known
        // together, so r0's neighbor (r1) resolves immediately; r1's neighbor is
        // next iteration's r0, so it's finished one iteration late via `prev_s1`.
        if !Self::IS_R5K && !transparent {
        {
            let l2_instrs = self.l2.instrs.get_mut();
            let instrs_start = (l2_idx << Self::L2_INSTR_SHIFT) + offset_in_l2_line * 2;
            #[cfg(feature = "opcodefusion")]
            let dline_base = phys_addr & !(Self::DC_LINE_MASK as u64);
            #[cfg(feature = "opcodefusion")]
            let mut prev_s1: Option<usize> = None;
            for i in 0..Self::DC_CHUNKS_PER_LINE {
                let chunk = dc_data[l1_start_chunk + i];
                let r0 = (chunk >> 32) as u32;
                let r1 = chunk as u32;
                let idx0 = instrs_start + i * 2;
                let idx1 = idx0 + 1;
                #[cfg(feature = "opcodefusion")]
                if let Some(prev_idx) = prev_s1.take() {
                    let prev_phys = (dline_base as usize) + (i * 2 - 1) * 4;
                    if prev_phys & Self::IC_LINE_MASK != Self::IC_LINE_MASK - 3 {
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
                    if word0_phys & Self::IC_LINE_MASK != Self::IC_LINE_MASK - 3 {
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
        }

        // Mark L2 line as dirty. Unconditional: tcache changes *where the data
        // lives*, never the cache state machine. A dirty transparent L2 line is
        // harmless — flushing it reads RAM and writes RAM.
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
            let phys_addr = l1d_tag_to_phys(tag, (l1_idx << Self::DC_LINE_SHIFT) as u64);
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
        let phys_addr = l2_tag_to_phys(tag, (idx << Self::L2_LINE_SHIFT) as u64);

        // R4K: first flush dirty L1-D sub-lines into L2, so L2 has the authoritative data.
        // R5K: L1 and L2 are non-inclusive; dirty L1D lines hold the latest data and will
        // write back independently. We skip the pre-flush here to avoid complexity; if a
        // dirty L1D line exists when we write back L2 to memory we may lose data, but the
        // CACHE-op sequence IRIX uses (C_IWBINV L1D then C_IWBINV L2) ensures L1D is clean
        // before L2 is touched. For safety we scan and flush anyway on R5K too.
        if !Self::IS_R5K {
        {
            let l1d_lines_per_l2 = 1 << (Self::L2_LINE_SHIFT - Self::DC_LINE_SHIFT);
            let dc_base_idx = self.l2_idx_to_l1_base_idx(idx, tag.pidx(), &self.dc);
            for i in 0..l1d_lines_per_l2 {
                let dc_idx = (dc_base_idx + i) & Self::DC_NUM_LINES_MASK;
                let phys_addr_l1 = phys_addr + ((i as u64) << Self::DC_LINE_SHIFT);
                let dc_tag: L1DTag = self.dc.get_tag(dc_idx);
                if dc_tag.matches_phys(phys_addr_l1) {
                    self.writeback_l1d_line(dc_idx, false);
                }
            }
        }
        }

        // Now check if L2 line is dirty (may have become dirty from L1-D writeback)
        let mut tag: L2Tag = self.l2.get_tag(idx);
        let cs = tag.cs();
        if cs != L2_CS_DIRTY_EXCLUSIVE && cs != L2_CS_DIRTY_SHARED {
            return true; // Nothing to write back
        }
        // tcache: `l2.data` is not storage — every write already went to RAM,
        // so there is nothing to flush. Unconditional: the bitmap decides *how*
        // to reach RAM (window vs bus), never whether L2 holds data. The state
        // transition below still runs exactly as it always does.
        #[cfg(feature = "tcache")]
        let tc_line = true;
        #[cfg(not(feature = "tcache"))]
        let tc_line = false;

        #[cfg(feature = "debug_cache")]
        if self.is_tracking_l2_idx(idx) {
            println!("[CACHE DEBUG] writeback_l2_line: {} idx=0x{:x}, phys_addr=0x{:08x}, ptag=0x{:05x}, cs={}, WRITING TO MEMORY",
                     self.tracking_label_l2_idx(idx), idx, phys_addr, tag.ptag(), cs);
            // Dump the L2 line data being written
            let l2_data = self.l2.data();
            let start_chunk = idx << Self::L2_CHUNKS_PER_LINE_SHIFT;
            println!("  L2 line data being written (16 x u64):");
            for i in 0..Self::L2_CHUNKS_PER_LINE {
                let val = l2_data[start_chunk + i];
                println!("    [{}] addr=0x{:08x} val=0x{:016x}", i, phys_addr + ((i as u64) << 3), val);
            }
        }

        // NOTE: do NOT clear llbit here. On R4000, llbit tracks L1-D state only.
        // An L2 writeback/eviction is not a coherency action and must not break LL/SC.

        // Now write L2 data to memory.
        //
        // tcache: nothing to write. A transparent line has no `l2.data` — its
        // storage is RAM, and RAM already holds the current bytes. The state
        // transition below still runs, exactly as for a backed line.
        if !tc_line {
            let l2_data = self.l2.data();
            let start_chunk = idx << Self::L2_CHUNKS_PER_LINE_SHIFT;
            let src = &l2_data[start_chunk..start_chunk + Self::L2_CHUNKS_PER_LINE];
            if self.downstream.write_block(phys_addr as u32, src) != BUS_OK {
                return false;
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
        // Default origin is data. Instruction fills call the _for variant.
        self.fill_l2_line_for(phys_addr, virt_addr, false)
    }

    /// Fill an L2 line, recording whether it was filled to serve instructions.
    ///
    /// `fill_instructions` is what lets tcache bypass L2 data while keeping
    /// R4400's decode slots working:
    ///
    /// * `true`  — an L1I fill needs `l2.instrs` populated for this line, so
    ///   the words are read and decoded slots written, and the tag gets
    ///   `has_code`.
    /// * `false` — an L1D fill. Under tcache the data is reachable through
    ///   ppmem's window, so **nothing is read and no data is stored**; only
    ///   tags/state are installed, and `has_code` stays clear so a later L1I
    ///   fill knows it must do a real instruction fill.
    ///
    /// Without tcache both cases behave identically to the original: fill the
    /// data array and the decode slots.
    fn fill_l2_line_for(&self, phys_addr: u64, virt_addr: u64, fill_instructions: bool) -> bool {
        let _ = fill_instructions;
        let l2_idx = self.l2.get_index(phys_addr);

        // Writeback and invalidate the victim line (if any)
        // This will also writeback any dirty L1-D lines and invalidate L1-I/L1-D lines
        self.writeback_l2_line(l2_idx);
        self.invalidate_l2_line(l2_idx);

        // Calculate line-aligned address
        let line_base = phys_addr & !(Self::L2_LINE_MASK as u64);

        // Fill line from memory
        // tcache: for a transparent line `l2.data` is **never used** — not
        // filled, not read, not written back. RAM is the only store, so the
        // 128-byte fill is pure waste.
        //
        // The safety of skipping it rests on nothing ever reading `l2.data` for
        // such a line: `fill_l1d_line` copies nothing
        // of copying from L2, `writeback_l1d_line` neither copies into L2 nor
        // marks it dirty, and `writeback_l2_line` only flushes DIRTY lines —
        // which a transparent line can never become. An instruction-origin fill
        // still reads, because R4400's decode slots live in `l2.instrs` and
        // cannot be reconstructed from a pointer.
        #[cfg(feature = "tcache")]
        let skip_data = !fill_instructions;
        #[cfg(not(feature = "tcache"))]
        let skip_data = false;

        let l2_data = self.l2.data_mut();
        let start_chunk = l2_idx << Self::L2_CHUNKS_PER_LINE_SHIFT;

        // INVARIANT: l2.data is always accessed as u64 chunks (never as u32 words).
        // Direct-mapped: l2.instrs[n] mirrors word n and fetch() indexes it directly.
        // 2-way: l2.instrs is empty — fill_l1i_line reads raw words from l2.data.
        // Do not add data_as_words() accessors on L2 or fetch indexing will silently break.
        let instrs_start = l2_idx << Self::L2_INSTR_SHIFT;
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
        #[cfg(feature = "opcodefusion")]
        macro_rules! fuse_pair {
            ($l2_instrs:expr, $prev_s1:expr, $i:expr, $idx0:expr, $idx1:expr, $r0:expr, $r1:expr) => {
                if let Some(prev_idx) = $prev_s1.take() {
                    let prev_phys = (line_base as usize) + ($i * 2 - 1) * 4;
                    if prev_phys & Self::IC_LINE_MASK != Self::IC_LINE_MASK - 3 {
                        let prev = &mut $l2_instrs[prev_idx];
                        prev.imm = $r0;
                        prev.flags |= FLAG_IMM_IS_NEXT;
                    }
                }
                let word0_phys = (line_base as usize) + ($i * 2) * 4;
                if word0_phys & Self::IC_LINE_MASK != Self::IC_LINE_MASK - 3 {
                    let s0 = &mut $l2_instrs[$idx0];
                    s0.imm = $r1;
                    s0.flags |= FLAG_IMM_IS_NEXT;
                }
                $prev_s1 = Some($idx1);
            };
        }
        #[cfg(not(feature = "opcodefusion"))]
        macro_rules! fuse_pair {
            ($l2_instrs:expr, $prev_s1:expr, $i:expr, $idx0:expr, $idx1:expr, $r0:expr, $r1:expr) => {};
        }
        if skip_data {
            // Tags only; see `skip_data` above.
        } else if let Some(src) = self.downstream.mem_ptr(line_base as u32) {
            // Fast path: single pass over source — rotate into l2.data and fill l2.instrs.
            let l2_instrs = self.l2.instrs.get_mut();
            // Only fuse_pair! mutates prev_s1, and it is a no-op without opcodefusion.
            #[cfg_attr(not(feature = "opcodefusion"), allow(unused_mut))]
            let mut prev_s1: Option<usize> = None;
            for i in 0..Self::L2_CHUNKS_PER_LINE {
                let val = unsafe { (*src.add(i)).rotate_left(32) };
                l2_data[start_chunk + i] = val;
                if !Self::IS_R5K {
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
            }
        } else {
            let dest = &mut l2_data[start_chunk..start_chunk + Self::L2_CHUNKS_PER_LINE];
            let s = self.downstream.read_block(line_base as u32, dest);
            if s != crate::traits::BUS_OK { return false; }
            if !Self::IS_R5K {
            {
                let l2_instrs = self.l2.instrs.get_mut();
                #[cfg_attr(not(feature = "opcodefusion"), allow(unused_mut))]
                let mut prev_s1: Option<usize> = None;
                for i in 0..Self::L2_CHUNKS_PER_LINE {
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
        }

        // Set tag with CleanExclusive state
        let ptag = self.l2_ptag(phys_addr);
        let pidx = self.pidx(virt_addr);
        let mut new_tag = L2Tag::default();
        new_tag.set_ptag(ptag);
        new_tag.set_cs(L2_CS_CLEAN_EXCLUSIVE);
        new_tag.set_pidx(pidx);
        // `l2.instrs` was populated iff we actually read the line's words.
        new_tag.set_has_code(!skip_data);
        self.l2.set_tag(l2_idx, new_tag);

        // println!("[CACHE DEBUG] fill_l2_line: idx={}, phys_addr=0x{:08x}, ptag=0x{:05x}, pidx={}, state=CleanExclusive",
        //          l2_idx, phys_addr, ptag, pidx);

        #[cfg(feature = "debug_cache")]
        if self.is_tracking_l2_idx(l2_idx) {
            println!("[CACHE DEBUG] fill_l2_line: {} line 0x{:08x}, idx=0x{:x}, phys_addr=0x{:08x}, ptag=0x{:05x}, pidx={}",
                     self.tracking_label_l2_idx(l2_idx), line_base, l2_idx, phys_addr, ptag, pidx);
            println!("  L2 line data (16 x u64):");
            for i in 0..Self::L2_CHUNKS_PER_LINE {
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

        // 2-way picks the LRU way; direct-mapped has one way, so eidx == set.
        let ic_eidx = if Self::IS_R5K {
            set | (unsafe { Self::lru_get(self.ic_lru.get(), set) } as usize) << Self::IC_NUM_LINES_SHIFT
        } else { set };

        // Invalidate victim slot unconditionally — clears tag before any early return.
        self.invalidate_l1i_line(ic_eidx, false);

        // Ensure L2 has the line (skipped when L2 is disabled — read directly from memory).
        if self.l2_active() {
            let l2_idx = self.l2.get_index(phys_addr);
            let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
            let l2_ptag = self.l2_ptag(phys_addr);
            // tcache: a tag match is not enough. Under tcache an L1D-origin
            // fill installs a valid tag but populates no data and no decode
            // slots, so `l2.instrs` for this line is stale. Only a line filled
            // *for instructions* (has_code) can satisfy an L1I fill; otherwise
            // fall through and do a real instruction fill below.
            let l2_hit = l2_tag.cs() != L2_CS_INVALID
                && l2_tag.ptag() == l2_ptag
                && (cfg!(not(feature = "tcache")) || l2_tag.has_code());

            if l2_hit {
                // R4K only: check for Virtual Coherency Exception (VCEI).
                // R5K dropped VCE — no pidx tracking needed.
                if !Self::IS_R5K {
                if self.pidx(index_addr) != l2_tag.pidx() {
                    return exec_exception_const(EXC_VCEI);
                }
                }
            } else {
                // L2 miss — fill from memory into L2 first.
                #[cfg(not(feature = "lightning"))]
                if devlog_is_active(LogModule::L2c) && devlog_mask(LogModule::L2c) & CACHE_LOG_MISS != 0 {
                    crate::dlog!(LogModule::L2c, "fill virt={:#x} phys={:#x} idx={}", index_addr, phys_addr, self.l2.get_index(phys_addr));
                }
                if !self.fill_l2_line_for(phys_addr, index_addr, true) {
                    return exec_exception_const(EXC_IBE);
                }
            }
        }

        #[cfg(not(feature = "lightning"))]
        if devlog_is_active(LogModule::L1i) && devlog_mask(LogModule::L1i) & CACHE_LOG_MISS != 0 {
            crate::dlog!(LogModule::L1i, "fill virt={:#x} phys={:#x} eidx={}", index_addr, phys_addr, ic_eidx);
        }

        // 2-way only: populate this way's ic_instrs slot from L2, or memory if L2 is off.
        if Self::IS_R5K {
            let ic_slot_base = ic_eidx << Self::IC_INSTR_SHIFT;
            let ic_instrs = self.ic_instrs.get_mut();
            // tcache: `l2.data` holds nothing — instruction words come from RAM,
            // the same place the L2-disabled branch below reads them from.
            #[cfg(feature = "tcache")]
            let l2_has_data = false;
            #[cfg(not(feature = "tcache"))]
            let l2_has_data = self.l2_active();
            if l2_has_data {
                let l2_sub_offset = ((phys_addr as usize) & (Self::L2_LINE_MASK & !Self::IC_LINE_MASK)) >> 3;
                let l2_chunk_base = (self.l2.get_index(phys_addr) << Self::L2_CHUNKS_PER_LINE_SHIFT)
                    + l2_sub_offset;
                let l2_data = self.l2.data();
                let src = unsafe { l2_data.as_ptr().add(l2_chunk_base) };
                for i in 0..Self::IC_INSTRS_PER_LINE / 2 {
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
                let line_base = phys_addr & !(Self::IC_LINE_MASK as u64);
                if let Some(src) = self.downstream.mem_ptr(line_base as u32) {
                    // Fast path: read word pairs directly from backing store.
                    // mem_ptr's raw u64s are host-native pairs of little-endian-loaded
                    // words; rotate_left(32) puts them in MIPS big-endian word order
                    // (high word = first instr), matching read_block's behavior.
                    for i in 0..Self::IC_INSTRS_PER_LINE / 2 {
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
                    for i in 0..Self::IC_INSTRS_PER_LINE {
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
            let way = ic_eidx >> Self::IC_NUM_LINES_SHIFT;
            unsafe { Self::lru_set(self.ic_lru.get(), set, way ^ 1); }
        }

        self.ic.set_tag(ic_eidx, L1ITag::valid(phys_addr));

        #[cfg(feature = "debug_cache")]
        if self.is_tracking_addr(index_addr, phys_addr) || self.is_tracking_l2_idx(self.l2.get_index(phys_addr)) {
            let way = ic_eidx >> Self::IC_NUM_LINES_SHIFT;
            let set = ic_eidx & Self::IC_NUM_LINES_MASK;
            println!("[CACHE DEBUG] fill_l1i_line: {} virt 0x{:016x} phys 0x{:016x} → L1I eidx=0x{:x} way={} set=0x{:x}",
                self.tracking_label(phys_addr), index_addr, phys_addr, ic_eidx, way, set);
            if Self::IS_R5K {
                let ic_instrs = self.ic_instrs.get();
                let slot_base = ic_eidx << Self::IC_INSTR_SHIFT;
                print!("  ic_instrs:");
                for i in 0..Self::IC_INSTRS_PER_LINE {
                    if i % 4 == 0 { print!("\n    "); }
                    print!("{:08x} ", ic_instrs[slot_base + i].raw);
                }
                println!();
            }
        }

        (ic_eidx >> Self::IC_NUM_LINES_SHIFT) as u32
    }

    /// Fill L1 data cache line. Ensures data is in L2 first, then copies to L1-D.
    ///
    /// Returns a `u32` with the same encoding as `ensure_l1d_line`:
    ///   0  = filled into way0 (BUS_OK)
    ///   1  = filled into way1 (BUS_OK, R5K only)
    ///   >1 = BUS_VCE or BUS_ERR

    /// tcache debug: assert the foundational invariant — for a transparent
    /// address, ppmem's window is authoritative and no cache tier holds a
    /// divergent copy.
    ///
    /// This is the property a JIT emitting inline loads/stores against
    /// `tc_base + phys` depends on, so it is worth checking directly rather
    /// than inferring from behaviour. Compiled only under
    /// `tcache_verify`; it re-reads through the bus and compares, which is far
    /// too slow for normal runs but pinpoints the exact access that diverges
    /// on a real boot instead of surfacing as a panic a million instructions
    /// later.
    #[cfg(all(feature = "tcache", feature = "tcache_verify"))]
    #[inline]
    fn tc_verify(&self, virt_addr: u64, phys_addr: u64, what: &str) {
        if !self.tc_transparent(phys_addr) {
            return;
        }
        let dc_idx = self.dc.get_index(virt_addr);
        let tag: L1DTag = self.dc.get_tag(dc_idx);
        if !tag.matches_phys(phys_addr) {
            return;
        }
        // If L2 also holds this line, its data must agree with RAM, or an L2
        // eviction will clobber the window.
        if HAS_L2 {
            let l2_idx = self.l2.get_index(phys_addr);
            let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
            if l2_tag.cs() != L2_CS_INVALID && l2_tag.ptag() == self.l2_ptag(phys_addr) {
                let aligned = phys_addr & !7;
                let l2_chunk = (l2_idx << Self::L2_CHUNKS_PER_LINE_SHIFT)
                    + ((aligned & Self::L2_LINE_MASK as u64) >> 3) as usize;
                let in_l2 = self.l2.data()[l2_chunk];
                let in_ram = self.tc_read::<8>(aligned);
                let dirty = l2_tag.cs() == L2_CS_DIRTY_EXCLUSIVE
                    || l2_tag.cs() == L2_CS_DIRTY_SHARED;
                if dirty && in_l2 != in_ram {
                    panic!(
                        "tcache invariant [{what}]: DIRTY L2 line disagrees with RAM at \
                         {aligned:#x} — l2.data={in_l2:#018x} ram={in_ram:#018x}; \
                         evicting L2 will clobber the transparent write"
                    );
                }
            }
        }
    }

    /// Render an L2 tag's `has_code` state for the cache dumps. Empty string
    /// without tcache, where the flag does not exist.
    fn has_code_note(_tag: L2Tag) -> &'static str {
        #[cfg(feature = "tcache")]
        {
            if _tag.has_code() { " has_code=true" } else { " has_code=false" }
        }
        #[cfg(not(feature = "tcache"))]
        { "" }
    }

    /// tcache: drop `has_code` for the L2 line covering `phys_addr`, if it has
    /// one, because a transparent write just changed the underlying bytes
    /// without going through L2.
    ///
    /// Cheap: one tag load, and a store only when the flag was actually set —
    /// which is rare, since most lines are data-origin and never had it.
    #[cfg(feature = "tcache")]
    #[inline]
    fn tc_invalidate_l2_code(&self, phys_addr: u64) {
        if !HAS_L2 {
            return;
        }
        let l2_idx = self.l2.get_index(phys_addr);
        let mut l2_tag: L2Tag = self.l2.get_tag(l2_idx);
        if l2_tag.has_code() && l2_tag.ptag() == self.l2_ptag(phys_addr) {
            l2_tag.set_has_code(false);
            self.l2.set_tag(l2_idx, l2_tag);
        }
    }

    /// Ensure L2 holds this line for a *data* access, filling it on a miss.
    ///
    /// `Ok(true)` — L2 now has the line. `Err(status)` — VCE or bus error, to
    /// be returned from the caller. Split out of `fill_l1d_line` so the
    /// tcache path can skip it wholesale without duplicating the body.
    #[cfg(feature = "tcache")]
    fn l2_probe_for_data(&self, virt_addr: u64, phys_addr: u64) -> Result<bool, u32> {
        let l2_idx = self.l2.get_index(phys_addr);
        let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
        let l2_ptag = self.l2_ptag(phys_addr);
        if l2_tag.cs() != L2_CS_INVALID && l2_tag.ptag() == l2_ptag {
            if !Self::IS_R5K && self.pidx(virt_addr) != l2_tag.pidx() {
                return Err(BUS_VCE);
            }
            Ok(true)
        } else {
            #[cfg(not(feature = "lightning"))]
            if devlog_is_active(LogModule::L2c) && devlog_mask(LogModule::L2c) & CACHE_LOG_MISS != 0 {
                crate::dlog!(LogModule::L2c, "fill virt={:#x} phys={:#x} idx={}", virt_addr, phys_addr, l2_idx);
            }
            if !self.fill_l2_line(phys_addr, virt_addr) {
                return Err(BUS_ERR);
            }
            Ok(true)
        }
    }

    /// Fill an L1D line.
    ///
    /// tcache: the cache holds no data, so this installs tags, updates LRU and
    /// runs the VCE check but copies nothing — the bytes are read from RAM at
    /// access time instead.
    fn fill_l1d_line(&self, virt_addr: u64, phys_addr: u64) -> u32 {
        #[cfg(feature = "tcache")]
        let transparent = true;
        #[cfg(not(feature = "tcache"))]
        let transparent = false;
        // For R5K: pick victim way via LRU and encode into dc_idx via shift.
        // dc_ext_idx = set | (way << Self::DC_NUM_LINES_SHIFT)
        let (victim_way, dc_idx) = if Self::IS_R5K {
            let set = self.dc.get_index(virt_addr);
            let way = unsafe { Self::lru_get(self.dc_lru.get(), set) } as usize;
            (way, set | (way << Self::DC_NUM_LINES_SHIFT))
        } else { (0usize, self.dc.get_index(virt_addr)) };

        // Writeback and invalidate the victim line (hardware fill — not a coherency action)
        self.writeback_l1d_line(dc_idx, false);
        self.invalidate_l1d_line(dc_idx, false, false);

        // tcache: no special-casing here — the L2 probe, VCE check, tag
        // install and LRU update run exactly as they always did, for
        // transparent and backed lines alike. There are **no early returns on
        // a tcache path**; every shortcut of that kind so far turned into a
        // coherency bug. The only difference is inside `fill_l2_line_for`,
        // which skips populating `l2.data` for a transparent line (`skip_data`)
        // because nothing will ever read it back out.
        #[cfg(feature = "tcache")]
        let l2_hit = if self.l2_active() {
            match self.l2_probe_for_data(virt_addr, phys_addr) {
                Ok(hit) => hit,
                Err(status) => return status,
            }
        } else {
            false
        };
        #[cfg(not(feature = "tcache"))]
        let l2_hit = if self.l2_active() {
            let l2_idx = self.l2.get_index(phys_addr);
            let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
            let l2_ptag = self.l2_ptag(phys_addr);
            if l2_tag.cs() != L2_CS_INVALID && l2_tag.ptag() == l2_ptag {
                // Check for Virtual Coherency Exception (R4K only; R5K dropped VCE)
                if !Self::IS_R5K {
                if self.pidx(virt_addr) != l2_tag.pidx() { return BUS_VCE; }
                }
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
        let dc_start_chunk = dc_idx << Self::DC_CHUNKS_PER_LINE_SHIFT;

        // tcache: a transparent line's data stays in RAM. `l2.data` was never
        // populated for it (see `skip_data`), so copying from L2 here would
        // move garbage into `dc.data` — and `dc.data` is itself unused for such
        // a line. Skip both.
        if !transparent {
            if l2_hit {
                // Copy from L2 to L1-D
                let dc_line_base = phys_addr & !(Self::DC_LINE_MASK as u64);
                let l2_idx = self.l2.get_index(phys_addr);
                let l2_line_base = l2_idx << Self::L2_CHUNKS_PER_LINE_SHIFT;
                let offset_in_l2_line = ((dc_line_base & (Self::L2_LINE_MASK as u64)) >> 3) as usize;
                let l2_data = self.l2.data();
                for i in 0..Self::DC_CHUNKS_PER_LINE {
                    dc_data[dc_start_chunk + i] = l2_data[l2_line_base + offset_in_l2_line + i];
                }
            } else {
                // L2 disabled: copy directly from memory
                let line_base = phys_addr & !(Self::DC_LINE_MASK as u64);
                let dest = &mut dc_data[dc_start_chunk..dc_start_chunk + Self::DC_CHUNKS_PER_LINE];
                self.downstream.read_block(line_base as u32, dest);
            }
        }

        self.dc.set_tag(dc_idx, L1DTag::valid(phys_addr, L1D_CS_CLEAN_EXCLUSIVE as u8, false));

        // R5K: flip LRU — filled way is MRU, other way is now victim
        if Self::IS_R5K {
        unsafe { Self::lru_set(self.dc_lru.get(), dc_idx & Self::DC_NUM_LINES_MASK, victim_way ^ 1); }
        }

        #[cfg(feature = "debug_cache")]
        {
            let line_base_phys = phys_addr & !(Self::DC_LINE_MASK as u64);
            let l2_idx_check = self.l2.get_index(phys_addr);
            if self.is_tracking_l1d(line_base_phys) || self.is_tracking_l2_idx(l2_idx_check) {
                let line_base_virt = virt_addr & !(Self::DC_LINE_MASK as u64);
                let way = dc_idx >> Self::DC_NUM_LINES_SHIFT;
                let set = dc_idx & Self::DC_NUM_LINES_MASK;
                println!("[CACHE DEBUG] fill_l1d_line: {} virt 0x{:016x} phys 0x{:016x} → L1D eidx=0x{:x} way={} set=0x{:x}",
                         self.tracking_label(line_base_phys), line_base_virt, line_base_phys, dc_idx, way, set);
                for i in 0..Self::DC_CHUNKS_PER_LINE {
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
    /// `dc_ext_idx` for tag/data access = `set | (way << Self::DC_NUM_LINES_SHIFT)`.
    #[inline(always)]
    fn ensure_l1d_line(&self, virt_addr: u64, phys_addr: u64) -> u32 {
        if Self::IS_R5K {
        {
            let set = self.dc.get_index(virt_addr);
            if self.dc.get_tag(set).matches_phys(phys_addr) {
                // way0 hit → way0 is MRU, way1 is LRU next
                unsafe { Self::lru_set(self.dc_lru.get(), set, 1); }
                return 0;
            }
            if self.dc.get_tag(set | (1 << Self::DC_NUM_LINES_SHIFT)).matches_phys(phys_addr) {
                // way1 hit → way1 is MRU, way0 is LRU next
                unsafe { Self::lru_set(self.dc_lru.get(), set, 0); }
                return 1;
            }
            self.fill_l1d_line(virt_addr, phys_addr)
        }
        } else {
        {
            let dc_idx = self.dc.get_index(virt_addr);
            let dc_tag: L1DTag = self.dc.get_tag(dc_idx);
            if dc_tag.matches_phys(phys_addr) { 0 }
            else { self.fill_l1d_line(virt_addr, phys_addr) }
        }
        }
    }

    /// Compute the data-array address for `dc.dc_read/dc_write` from the extended tag index
    /// and the original virtual address.
    ///   dc_ext_idx = set | (way << Self::DC_NUM_LINES_SHIFT)
    ///   → data address = (dc_ext_idx << LINE_SHIFT) | (virt_addr & LINE_MASK)
    /// Way1 data lives in [DC_SIZE/2, DC_SIZE); dc_read/dc_write mask to (DC_SIZE-1) so this
    /// routes both ways into the correct half of the allocated data array.
    /// For R4K (1-way), dc_ext_idx = dc_idx so this equals the original virt_addr.
    #[inline(always)]
    fn dc_data_addr(dc_ext_idx: usize, virt_addr: u64) -> u64 {
        ((dc_ext_idx << Self::DC_LINE_SHIFT as usize) as u64)
            | (virt_addr & Self::DC_LINE_MASK as u64)
    }

    /// Mark the L1-D line as dirty.
    /// `dc_ext_idx` = `set | (way << Self::DC_NUM_LINES_SHIFT)` (returned by ensure_l1d_line).
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
            let eidx = set | (way << Self::IC_NUM_LINES_SHIFT);
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
            let eidx = set | (way << Self::DC_NUM_LINES_SHIFT);
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

    /// Write back the L1-D line(s) covering physical line `phys_line` (already
    /// line-aligned to `DC_LINE`), then cascade to L2 if written. Used by
    /// `writeback_range` for both the known-virt-addr case (a single eidx from
    /// `hit_l1d`) and the phys-only case (every possible VIPT alias below).
    /// Returns the number of L1-D lines actually written back (0 or 1).
    #[inline]
    fn writeback_dc_eidx(&self, eidx: usize) -> usize {
        let tag: L1DTag = self.dc.get_tag(eidx);
        if !tag.dirty { return 0; }
        self.writeback_l1d_line(eidx, true);
        1
    }

    /// Phys-only L1-D writeback for one line: since only the low
    /// `12 - DC_LINE_SHIFT` index bits are guaranteed to equal the physical
    /// address's (the rest are virtual page-color bits VIPT can't recover
    /// from a phys addr alone), probe every alias of those upper index bits,
    /// across every way, and write back whichever ones tag-match and are
    /// dirty. Returns the number of lines written back (0, 1, or more if
    /// multiple aliases happen to independently hold dirty copies).
    fn writeback_dc_phys_line(&self, phys_line: u64) -> usize {
        let base_idx = self.dc.get_index(phys_line);
        let alias_bits = Self::DC_NUM_LINES_SHIFT.saturating_sub(12 - Self::DC_LINE_SHIFT);
        let num_aliases = 1usize << alias_bits;
        let alias_mask = (num_aliases - 1) << (12 - Self::DC_LINE_SHIFT);
        let mut n = 0;
        for alias in 0..num_aliases {
            let set = (base_idx & !alias_mask) | (alias << (12 - Self::DC_LINE_SHIFT));
            for way in 0..DC_WAYS {
                let eidx = set | (way << Self::DC_NUM_LINES_SHIFT);
                let tag: L1DTag = self.dc.get_tag(eidx);
                if tag.dirty && tag.matches_phys(phys_line) {
                    self.writeback_l1d_line(eidx, true);
                    n += 1;
                }
            }
        }
        n
    }

    /// Shared implementation of the public `writeback` trait method — see its
    /// doc comment for the virt/phys-only contract.
    fn writeback_range(&self, virt_addr: Option<u64>, phys_addr: u64, size: u64) -> usize {
        if size == 0 { return 0; }
        let mut count = 0usize;

        // --- L1-D ---
        let dc_line = DC_LINE as u64;
        let phys_start = phys_addr & !(dc_line - 1);
        let phys_end = phys_addr + size;
        match virt_addr {
            Some(va) => {
                let virt_start = va & !(dc_line - 1);
                let mut off = 0u64;
                while phys_start + off < phys_end {
                    if let Some(eidx) = self.hit_l1d(virt_start + off, phys_start + off) {
                        count += self.writeback_dc_eidx(eidx);
                    }
                    off += dc_line;
                }
            }
            None => {
                let mut line = phys_start;
                while line < phys_end {
                    count += self.writeback_dc_phys_line(line);
                    line += dc_line;
                }
            }
        }

        // --- L2 (if present) ---
        if self.l2_active() {
            let l2_line = L2_LINE as u64;
            let mut line = phys_addr & !(l2_line - 1);
            while line < phys_end {
                if let Some(idx) = self.hit_l2(line) {
                    let tag: L2Tag = self.l2.get_tag(idx);
                    if tag.cs() == L2_CS_DIRTY_EXCLUSIVE || tag.cs() == L2_CS_DIRTY_SHARED {
                        self.writeback_l2_line(idx);
                        count += 1;
                    }
                }
                line += l2_line;
            }
        }

        count
    }
}

impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize> CpuModel for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES> {
    const MIPS4: bool = MIPS4;
    const PRID: u32 = PRID;
    const FIR: u32 = FIR;
    const TLB_ENTRIES: usize = TLB_ENTRIES;
    const NAME: &'static str = if IC_WAYS == 2 { "R5000" } else { "R4400" };
}

impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize> MipsCache for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES> {
    fn set_l1i_counters(&mut self, hit: Arc<AtomicU64>, fetch: Arc<AtomicU64>) {
        self.l1i_hit_count = hit;
        self.l1i_fetch_count = fetch;
    }

    #[cfg(feature = "tcache")]
    unsafe fn set_tcache_window(&self, base: *mut u8) {
        unsafe { self.set_tcache_window_impl(base) }
    }

    #[cfg(feature = "tcache")]
    fn tcache_bitmap_ptr(&self) -> *mut u64 { self.tc_bitmap_ptr() }

    #[cfg(all(feature = "tcache", feature = "jitv2"))]
    unsafe fn set_tcache_gen_window(&self, gen_base: *mut AtomicU64) {
        unsafe { self.set_tcache_gen_window_impl(gen_base) }
    }

    const IC_SIZE: usize = IC_SIZE;
    const IC_LINE: usize = IC_LINE;
    const IC_WAYS: usize = IC_WAYS;
    const DC_SIZE: usize = DC_SIZE;
    const DC_LINE: usize = DC_LINE;
    const DC_WAYS: usize = DC_WAYS;
    const L2_SIZE: usize = if HAS_L2 { L2_CACHE_SIZE } else { 0 };
    const L2_LINE: usize = L2_LINE;

    fn fetch(&self, virt_addr: u64, phys_addr: u64) -> FetchInstrResult {
        if Self::IS_R5K {
            #[cfg(feature = "debug_cache")]
            {
                if self.is_tracking_addr(virt_addr, phys_addr) {
                    println!("[CACHE DEBUG] fetch: {} virt_addr 0x{:016x}, phys_addr 0x{:016x}",
                             self.tracking_label(phys_addr), virt_addr, phys_addr);
                }
            }

            let set = self.ic.get_index(virt_addr);
            let way1_base = 1 << Self::IC_NUM_LINES_SHIFT;

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
                set | (way as usize) << Self::IC_NUM_LINES_SHIFT
            };

            let instr_idx = (ic_eidx << Self::IC_INSTR_SHIFT)
                | ((virt_addr as usize >> 2) & Self::IC_INSTR_MASK);
            {
                let slot = &self.ic_instrs.get()[instr_idx] as *const DecodedInstr;
                FetchInstrResult::hit(slot)
            }
        } else {
            #[cfg(feature = "debug_cache")]
            let tracked = {
                if self.is_tracking_addr(virt_addr, phys_addr) {
                    println!("[CACHE DEBUG] fetch: {} virt_addr 0x{:016x}, phys_addr 0x{:016x}",
                             self.tracking_label(phys_addr), virt_addr, phys_addr);
                    true
                } else {
                    let l2_idx = self.l2.get_index(phys_addr);
                    if self.is_tracking_l2_idx(l2_idx) {
                        let line_base = phys_addr & !(Self::L2_LINE_MASK as u64);
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
                    let line_base = phys_addr & !(Self::L2_LINE_MASK as u64);
                    println!("[CACHE DEBUG] read (L2 alias): idx=0x{:x}, line 0x{:08x}, virt 0x{:016x}, phys 0x{:016x}, size {}",
                             l2_idx, line_base, virt_addr, phys_addr, SIZE);
                }
            }
        }

        // R4K (1-way): way is always 0, dc_eidx == get_index(virt_addr), da == virt_addr.
        // Skip the generic ensure_l1d_line/dc_data_addr indirection entirely.
        if !Self::IS_R5K {
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
            // tcache: the cache stores no data — read RAM (window or bus).
            #[cfg(feature = "tcache")]
            let result = self.tc_read::<SIZE>(phys_addr);
            #[cfg(not(feature = "tcache"))]
            let result = self.dc.dc_read::<SIZE>(virt_addr);
            #[cfg(feature = "debug_cache")]
            if self.is_tracking_addr(virt_addr, phys_addr) {
                println!("[CACHE DEBUG] read{} result: {} virt 0x{:016x} phys 0x{:016x} val=0x{:016x}",
                         SIZE * 8, self.tracking_label(phys_addr), virt_addr, phys_addr, result);
            }
            return BusRead64::ok(result);
        }
        }
        // R5K (2-way): use generic path with way encoding.
        {
            let way = self.ensure_l1d_line(virt_addr, phys_addr);
            if way > 1 { return BusRead64 { status: way, data: 0 }; }
            let dc_eidx = self.dc.get_index(virt_addr) | (way as usize) << Self::DC_NUM_LINES_SHIFT;
            #[cfg(not(feature = "lightning"))]
            if devlog_is_active(LogModule::L1d) && devlog_mask(LogModule::L1d) & CACHE_LOG_HIT != 0 {
                crate::dlog!(LogModule::L1d, "read{} hit virt={:#x} phys={:#x} eidx={}", SIZE, virt_addr, phys_addr, dc_eidx);
            }
            // tcache: the cache stores no data — read RAM (window or bus).
            #[cfg(feature = "tcache")]
            let result = self.tc_read::<SIZE>(phys_addr);
            #[cfg(not(feature = "tcache"))]
            let result = {
                let da = Self::dc_data_addr(dc_eidx, virt_addr);
                self.dc.dc_read::<SIZE>(da)
            };
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
        if !Self::IS_R5K {
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
            #[cfg(feature = "tcache")]
            {
                // The cache stores no data — the write goes to RAM.
                self.tc_write::<SIZE>(phys_addr, val);
                // Writing the line makes L2's decoded slots for it stale.
                self.tc_invalidate_l2_code(phys_addr);
            }
            #[cfg(not(feature = "tcache"))]
            self.dc.dc_write::<SIZE>(virt_addr, val);
            self.mark_l1d_dirty(dc_idx);
            return BUS_OK;
        }
        }
        // R5K (2-way): generic path.
        //
        // tcache is **not implemented here** — the 2-way paths always take the
        // real-cache route, so no transparent lines are ever created on R5000
        // and the feature is silently a no-op for that model. Safe, but if
        // tcache is ever extended to R5000, the `has_code` invalidation below
        // must be added here too (see the R4K branch above for why it must not
        // be gated on `transparent`).
        {
            let way = self.ensure_l1d_line(virt_addr, phys_addr);
            if way > 1 { return way; }
            let dc_eidx = self.dc.get_index(virt_addr) | (way as usize) << Self::DC_NUM_LINES_SHIFT;
            #[cfg(not(feature = "lightning"))]
            if devlog_is_active(LogModule::L1d) && devlog_mask(LogModule::L1d) & CACHE_LOG_HIT != 0 {
                crate::dlog!(LogModule::L1d, "write{} hit virt={:#x} phys={:#x} eidx={} val={:#x}", SIZE, virt_addr, phys_addr, dc_eidx, val);
            }
            #[cfg(feature = "tcache")]
            {
                self.tc_write::<SIZE>(phys_addr, val);
                self.tc_invalidate_l2_code(phys_addr);
            }
            #[cfg(not(feature = "tcache"))]
            {
                let da = Self::dc_data_addr(dc_eidx, virt_addr);
                self.dc.dc_write::<SIZE>(da, val);
            }
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
        if !Self::IS_R5K {
        {
            let dc_idx = self.dc.get_index(virt_addr);
            if !self.dc.get_tag(dc_idx).matches_phys(phys_addr) {
                let r = self.fill_l1d_line(virt_addr, phys_addr);
                if r > 1 { return r; }
            }
            #[cfg(feature = "tcache")]
            {
                let current = self.tc_read::<8>(phys_addr);
                self.tc_write::<8>(phys_addr, (current & !mask) | (val & mask));
                self.tc_invalidate_l2_code(phys_addr);
            }
            #[cfg(not(feature = "tcache"))]
            {
                let current = self.dc.dc_read::<8>(virt_addr);
                self.dc.dc_write::<8>(virt_addr, (current & !mask) | (val & mask));
            }
            self.mark_l1d_dirty(dc_idx);
            return BUS_OK;
        }
        }
        // R5K (2-way): generic path.
        {
            let way = self.ensure_l1d_line(virt_addr, phys_addr);
            if way > 1 { return way; }
            let dc_eidx = self.dc.get_index(virt_addr) | (way as usize) << Self::DC_NUM_LINES_SHIFT;
            #[cfg(feature = "tcache")]
            {
                let current = self.tc_read::<8>(phys_addr);
                self.tc_write::<8>(phys_addr, (current & !mask) | (val & mask));
                self.tc_invalidate_l2_code(phys_addr);
            }
            #[cfg(not(feature = "tcache"))]
            {
                let da = Self::dc_data_addr(dc_eidx, virt_addr);
                let current = self.dc.dc_read::<8>(da);
                self.dc.dc_write::<8>(da, (current & !mask) | (val & mask));
            }
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
        // 2-way encodes the way into the index; direct-mapped is the plain set.
        let idx = if is_l2 {
            self.l2.get_index(phys_addr)
        } else if is_icache {
            let set = self.ic.get_index(virt_addr);
            if Self::IS_R5K { set | (((virt_addr >> 14) as usize & 1) << Self::IC_NUM_LINES_SHIFT) } else { set }
        } else {
            let set = self.dc.get_index(virt_addr);
            if Self::IS_R5K { set | (((virt_addr >> 14) as usize & 1) << Self::DC_NUM_LINES_SHIFT) } else { set }
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
                    || self.is_tracking_l1d_idx(idx & Self::DC_NUM_LINES_MASK)
            };
            if tracked {
                let way = if !is_l2 { idx >> Self::DC_NUM_LINES_SHIFT } else { 0 };
                println!("[CACHE DEBUG] cache_op: {} virt={:#x} phys={:#x} idx=0x{:x} way={}",
                         cache_op_name(cache_op), virt_addr, phys_addr, idx, way);
            }
        }

        // cascade: on R5K, L1 cache ops must propagate to L2 (and L2 to memory) because
        // the PROM only flushes L1 when SC=1, relying on hardware to keep L2 coherent.
        //let cascade = !is_l2;
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

    fn writeback(&self, virt_addr: Option<u64>, phys_addr: u64, size: u64) -> usize {
        self.writeback_range(virt_addr, phys_addr, size)
    }

    fn get_config(&self, cache_target: u32) -> (usize, usize) {
        match cache_target {
            CACH_PI => (IC_SIZE, IC_LINE),
            CACH_PD => (DC_SIZE, DC_LINE),
            CACH_SI | CACH_SD => (Self::L2_SIZE, L2_LINE),
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
        let addr_line = phys_addr & !(Self::DC_LINE_MASK as u64);
        let ll_line = ll_addr & !(Self::DC_LINE_MASK as u64);
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
                let num_ways = Self::IC_NUM_LINES / (IC_SIZE / IC_LINE / IC_WAYS).max(1);
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
                let num_ways = Self::DC_NUM_LINES / (DC_SIZE / DC_LINE / DC_WAYS).max(1);
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
                // Under tcache a tag HIT is not sufficient for an instruction
                // fill: only a line filled *for instructions* carries valid
                // decoded slots. Without this, a line that would send L1I down
                // the miss path looks identical to one that would not.
                let code_note = Self::has_code_note(tag);
                #[cfg(feature = "tcache")]
                let code_warn = if status == "HIT" && !tag.has_code() {
                    "\n  NOTE: has_code=false — an L1I fill would MISS here and re-read from memory"
                } else {
                    ""
                };
                #[cfg(not(feature = "tcache"))]
                let code_warn = "";
                format!("{} at index 0x{:x} (phys 0x{:016x})\n  Tag: 0x{:05x} (Wanted: 0x{:05x})\n  CS: {} ({}){}\n  PIdx: stored={} virt={}{}{}",
                    status, idx, phys_addr, tag.ptag(), wanted_tag, tag.cs(), cs_str,
                    code_note, tag.pidx(), virt_pidx, vce_warn, code_warn)
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
                let instrs_per_ic_line = Self::IC_INSTRS_PER_LINE;

                // R5K: instruction words live in ic_instrs (owned by L1I, indexed by eidx).
                // R4K: instruction words live in l2.instrs (indexed by physical word address).
                let mut s = if Self::IS_R5K {
                    let ic_instrs = self.ic_instrs.get();
                    let slot_base = idx << Self::IC_INSTR_SHIFT;
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
                } else {
                    let l2_data = self.l2.data();
                    // `tag.line_addr()` is only bits [35:12] — the *page* base.
                    // The line's own index bits have to come from the L1I index
                    // or every derived L2 lookup lands on the wrong line (this
                    // is why the dump used to print an unrelated, invalid L2
                    // line alongside a perfectly valid one).
                    let phys_line = l1_tag_to_phys(tag, (idx << Self::IC_LINE_SHIFT) as u64);
                    let phys_base = phys_line as usize;
                    let l2_slot_base = (phys_base & (Self::L2_SIZE - 1)) >> 2;
                    let mut s = format!(
                        "L1-I Line 0x{:x}: Tag=0x{:010x} V={} phys=0x{:08x}\n  Instrs (decoded slots in l2.instrs):",
                        idx, tag.line_addr(), tag.is_valid(), phys_line);
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
                    // Index L2 by the *line* address, not the page base.
                    let phys_line = l1_tag_to_phys(tag, (idx << Self::IC_LINE_SHIFT) as u64);
                    let l2_idx = self.l2.get_index(phys_line);
                    let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
                    let l2_base = l2_idx << Self::L2_CHUNKS_PER_LINE_SHIFT;
                    let sub = ((phys_line as usize) & Self::L2_LINE_MASK) >> 3;
                    let l2_matches = l2_tag.cs() != L2_CS_INVALID
                        && l2_tag.ptag() == self.l2_ptag(phys_line);
                    s.push_str(&format!(
                        "\n  L2[0x{:x}] cs={} tag_match={}{}: ",
                        l2_idx,
                        l2_tag.cs(),
                        l2_matches,
                        Self::has_code_note(l2_tag),
                    ));
                    for i in 0..Self::IC_CHUNKS_PER_LINE {
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
                let start = idx << Self::DC_CHUNKS_PER_LINE_SHIFT;

                let mut s = format!("L1-D Line 0x{:x}: Tag=0x{:010x} CS={} ({}) D={}\n  Data:",
                    idx, tag.ptag, tag.cs, cs_str, tag.dirty);
                for i in 0..Self::DC_CHUNKS_PER_LINE {
                    if i % 4 == 0 { s.push_str("\n    "); }
                    if start + i < dc_data.len() {
                        s.push_str(&format!("{:016x} ", dc_data[start + i]));
                    }
                }
                s
            }
            "l2" => {
                if Self::L2_NUM_LINES == 0 || idx >= Self::L2_NUM_LINES {
                    return format!("Index 0x{:x} out of bounds (max 0x{:x})", idx, Self::L2_NUM_LINES.saturating_sub(1));
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
                let start = idx << Self::L2_CHUNKS_PER_LINE_SHIFT;
                let phys_line = l2_tag_to_phys(tag, (idx << Self::L2_LINE_SHIFT) as u64);

                let mut s = format!(
                    "L2 Line 0x{:x}: Tag=0x{:05x} CS={} ({}) phys=0x{:08x}{}\n  Data (l2.data):",
                    idx, tag.ptag(), tag.cs(), cs_str, phys_line,
                    Self::has_code_note(tag),
                );
                for i in 0..Self::L2_CHUNKS_PER_LINE {
                    if i % 4 == 0 { s.push_str("\n    "); }
                    if start + i < l2_data.len() {
                        s.push_str(&format!("{:016x} ", l2_data[start + i]));
                    }
                }

                // R4K: `l2.instrs` is a *separate* array from `l2.data` — the
                // decoded-instruction slots the interpreter actually dispatches
                // from. They can disagree (that is precisely the tcache
                // coherency hazard), so print both and flag mismatches rather
                // than leaving the reader to guess which one `Data:` was.
                if !Self::IS_R5K {
                    let instrs = self.l2.instrs.get();
                    let islot = idx << Self::L2_INSTR_SHIFT;
                    s.push_str("\n  Instrs (l2.instrs, [!] = disagrees with l2.data):");
                    for i in 0..Self::L2_INSTRS_PER_LINE {
                        if i % 8 == 0 { s.push_str("\n    "); }
                        if islot + i >= instrs.len() { break; }
                        let from_instrs = instrs[islot + i].raw;
                        let chunk = l2_data[start + (i >> 1)];
                        let from_data = if i & 1 == 0 { (chunk >> 32) as u32 } else { chunk as u32 };
                        if from_instrs != from_data {
                            s.push_str(&format!("{:08x}[!{:08x}] ", from_instrs, from_data));
                        } else {
                            s.push_str(&format!("{:08x} ", from_instrs));
                        }
                    }
                }

                // Which L1I lines does this L2 line cover, and are any resident?
                // On R4K, L1I holding a line implies L2 must be able to serve
                // its instructions — so this is the observable cross-check for
                // whether `has_code` is telling the truth.
                {
                    let l1i_per_l2 = Self::L2_LINE / IC_LINE;
                    let mut resident = 0usize;
                    let mut list = String::new();
                    for i in 0..l1i_per_l2 {
                        let sub_phys = phys_line + (i as u64) * IC_LINE as u64;
                        let ic_idx = ((sub_phys as usize) >> Self::IC_LINE_SHIFT)
                            & Self::IC_NUM_LINES_MASK;
                        let ic_tag: L1ITag = self.ic.get_tag(ic_idx);
                        if ic_tag.matches_phys(sub_phys) {
                            resident += 1;
                            list.push_str(&format!("0x{:x} ", ic_idx));
                        }
                    }
                    s.push_str(&format!(
                        "\n  L1I: {}/{} sub-lines resident{}",
                        resident,
                        l1i_per_l2,
                        if list.is_empty() { String::new() } else { format!(" (sets {})", list.trim_end()) },
                    ));
                    #[cfg(feature = "tcache")]
                    if resident > 0 && !tag.has_code() {
                        s.push_str(
                            "\n  *** INCONSISTENT: L1I holds sub-lines of this L2 line but \
                             has_code=false — an L1I refill would take the miss path ***",
                        );
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
        if Self::IS_R5K {
        for s in self.ic_instrs.get_mut().iter_mut() { s.flags = FLAG_NOT_DECODED; s.raw = 0; }
        } else {
        for s in self.l2.instrs.get_mut().iter_mut() { s.flags = FLAG_NOT_DECODED; s.raw = 0; }
        }
        if Self::IS_R5K {
        unsafe {
            (*self.ic_lru.get()).fill(0u64);
            (*self.dc_lru.get()).fill(0u64);
        }
        }
        unsafe {
            *self.llbit.get() = false;
            *self.lladdr.get() = 0;
        }
    }

    fn save_cache_state(&self) -> toml::Value {
        Self::save_cache_state(self)
    }

    fn load_cache_state(&self, v: &toml::Value) -> Result<(), String> {
        Self::load_cache_state(self, v)
    }

}

// ---- Drop: stop and join decode thread ----

impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize> Drop for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES> {
    fn drop(&mut self) {
        self.ic.stop.store(true, Ordering::Relaxed);
    }
}

// ---- Resettable ----

impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize> Resettable for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES> {
    fn power_on(&self) {
        self.ic.tags_mut().fill(L1ITag::default());
        self.dc.tags_mut().fill(L1DTag::default());
        self.dc.data_mut().fill(0);
        self.l2.tags_mut().fill(L2Tag::default());
        self.l2.data_mut().fill(0);
        if Self::IS_R5K {
        for s in self.ic_instrs.get_mut().iter_mut() { s.flags = FLAG_NOT_DECODED; s.raw = 0; }
        } else {
        for s in self.l2.instrs.get_mut().iter_mut() { s.flags = FLAG_NOT_DECODED; s.raw = 0; }
        }
        if Self::IS_R5K {
        unsafe {
            (*self.ic_lru.get()).fill(0u64);
            (*self.dc_lru.get()).fill(0u64);
        }
        }
        unsafe {
            *self.llbit.get() = false;
            *self.lladdr.get() = 0;
        }
    }
}

// ---- snapshot helpers + MipsCache save/load override ----

impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize> CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES> {
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
        // 2-way only: LRU as packed u32 words, 1 bit per set (unchanged on-disk format).
        // ic_instrs not saved — rebuilt from l2.data on first fetch miss after restore.
        if Self::IS_R5K {
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
            t.insert("ic_lru".into(), u32_slice_to_toml(&pack(ic_lru, Self::IC_NUM_SETS)));
            t.insert("dc_lru".into(), u32_slice_to_toml(&pack(dc_lru, Self::DC_NUM_SETS)));
        }
        toml::Value::Table(t)
    }

    pub fn load_cache_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut ic_tags = vec![0u32; Self::IC_NUM_LINES];
        let mut dc_tags = vec![0u32; Self::DC_NUM_LINES];
        let mut dc_data = vec![0u64; DC_SIZE / 8];
        let mut l2_tags = vec![0u32; Self::L2_NUM_LINES];
        let mut l2_data = vec![0u64; Self::L2_SIZE / 8];

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
        let dl = l2_data.len().min(Self::L2_SIZE / 8);
        self.l2.data_mut()[..dl].copy_from_slice(&l2_data[..dl]);

        // R4K: rebuild l2.instrs from restored l2.data; fetch() indexes it directly.
        // R5K: l2.instrs is empty; ic_instrs will be repopulated on next L1I miss.
        if !Self::IS_R5K {
        {
            let l2_data_slice = self.l2.data();
            let l2_instrs = self.l2.instrs.get_mut();
            for line in 0..Self::L2_NUM_LINES {
                let chunks_start = line << Self::L2_CHUNKS_PER_LINE_SHIFT;
                let instrs_start = line << Self::L2_INSTR_SHIFT;
                for i in 0..Self::L2_CHUNKS_PER_LINE {
                    let chunk = l2_data_slice[chunks_start + i];
                    l2_instrs[instrs_start + i * 2].raw = (chunk >> 32) as u32;
                    l2_instrs[instrs_start + i * 2].flags = FLAG_NOT_DECODED;
                    l2_instrs[instrs_start + i * 2 + 1].raw = chunk as u32;
                    l2_instrs[instrs_start + i * 2 + 1].flags = FLAG_NOT_DECODED;
                }
            }
        }
        }

        // 2-way only: restore LRU bits; ic_instrs repopulates on the first fetch miss.
        if Self::IS_R5K {
            let unpack = |packed: &[u32], dst: &mut [u64], num_sets: usize| {
                dst.fill(0);
                for set in 0..num_sets {
                    if (packed[set / 32] >> (set % 32)) & 1 != 0 {
                        dst[set >> 6] |= 1u64 << (set & 63);
                    }
                }
            };
            let mut ic_lru_packed = vec![0u32; Self::IC_NUM_SETS.div_ceil(32)];
            let mut dc_lru_packed = vec![0u32; Self::DC_NUM_SETS.div_ceil(32)];
            if let Some(f) = get_field(v, "ic_lru") { load_u32_slice(f, &mut ic_lru_packed); }
            if let Some(f) = get_field(v, "dc_lru") { load_u32_slice(f, &mut dc_lru_packed); }
            unpack(&ic_lru_packed, unsafe { &mut *self.ic_lru.get() }, Self::IC_NUM_SETS);
            unpack(&dc_lru_packed, unsafe { &mut *self.dc_lru.get() }, Self::DC_NUM_SETS);
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

    fn make_cache(mem: Arc<Memory>) -> R4400Cache {
        R4400Cache::new(mem as Arc<dyn BusDevice>)
    }

    // Same helper for whichever CPU model a test wants to exercise.
    fn make_cache_of<C: MipsCache + From<Arc<dyn BusDevice>>>(mem: Arc<Memory>) -> C {
        C::from(mem as Arc<dyn BusDevice>)
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
    fn l1d_random_stress_r4400() { l1d_random_stress_for::<R4400Cache>() }
    #[test]
    fn l1d_random_stress_r5000() { l1d_random_stress_for::<R5000Cache>() }

    fn l1d_random_stress_for<C: MipsCache + From<Arc<dyn BusDevice>>>() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache: C = make_cache_of(mem.clone());
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
        let dc_sets = C::DC_SIZE / C::DC_LINE / C::DC_WAYS;
        for way in 0..C::DC_WAYS {
            for set in 0..dc_sets {
                // Bit 14 selects way for R5K index ops; for R4K WAYS==1 so way is always 0.
                let virt = kseg0(((way << 14) | (set * C::DC_LINE)) as u32);
                cache.cache_op(C_IINV | CACH_PD, virt, virt & 0x1FFF_FFFF);
            }
        }
        for i in 0..(C::L2_SIZE / C::L2_LINE) {
            let phys = (i * C::L2_LINE) as u64;
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
    fn l1i_fetch_stress_r4400() { l1i_fetch_stress_for::<R4400Cache>() }
    #[test]
    fn l1i_fetch_stress_r5000() { l1i_fetch_stress_for::<R5000Cache>() }

    fn l1i_fetch_stress_for<C: MipsCache + From<Arc<dyn BusDevice>>>() {
        let mem = Arc::new(Memory::new(MEM_MB));
        // Pre-fill with deterministic pattern directly through the bus.
        let mut rng = Rng::new(0x1234_5678_9abc_def0);
        let mut words = vec![0u32; MEM_BYTES / 4];
        for (i, w) in words.iter_mut().enumerate() {
            *w = rng.next_u32();
            mem.write32((i * 4) as u32, *w);
        }

        let cache: C = make_cache_of(mem.clone());
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
    fn full_flush(cache: &R4400Cache) {
        // Index_WBInvalidate all L1D sets (both ways for R5K).
        for way in 0..R4400Cache::DC_WAYS {
            for set in 0..R4400Cache::DC_SIZE / R4400Cache::DC_LINE / R4400Cache::DC_WAYS {
                let nls = R4400Cache::DC_NUM_LINES_SHIFT as usize;
                let ls  = R4400Cache::DC_LINE_SHIFT as usize;
                let idx_addr = (way << (nls + ls)) | (set << ls);
                let v = kseg0(idx_addr as u32);
                cache.cache_op(C_IWBINV | CACH_PD, v, v & 0x1FFF_FFFF);
            }
        }
        // Index_WBInvalidate all L2 sets (single-way, physically indexed).
        for set in 0..R4400Cache::L2_SIZE / R4400Cache::L2_LINE {
            let p = (set * R4400Cache::L2_LINE) as u64;
            cache.cache_op(C_IWBINV | CACH_SD, p, p);
        }
        // Index_Invalidate all L1I sets (both ways for R5K).
        for way in 0..R4400Cache::IC_WAYS {
            for set in 0..R4400Cache::IC_SIZE / R4400Cache::IC_LINE / R4400Cache::IC_WAYS {
                let nls = R4400Cache::IC_NUM_LINES_SHIFT as usize;
                let ls  = R4400Cache::IC_LINE_SHIFT as usize;
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
    fn flush_l2_to_mem(cache: &R4400Cache, phys: u32) {
        let l2_set = (phys as usize >> R4400Cache::L2_LINE_SHIFT as usize) & (R4400Cache::L2_SIZE / R4400Cache::L2_LINE - 1);
        let lp = (l2_set * R4400Cache::L2_LINE) as u64;
        cache.cache_op(C_IWBINV | CACH_SD, lp, lp);
    }

    /// Index_WBInvalidate L1D: dirty line goes to L2; L1D tag becomes invalid.
    /// Verify: data reaches L2 (by also flushing L2 to memory), and tag is invalidated
    /// (subsequent read after memory update picks up new value).
    #[test]
    fn cache_op_index_wbinv_l1d() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let phys: u32 = 0x1000;
        let virt = kseg0(phys);
        let _ = cache.write::<4>(virt, phys as u64, 0xABCD_1234u64);

        // Index_WBInvalidate the L1D set (way 0 — bit14 of index address = 0).
        let set = (phys as usize >> R4400Cache::DC_LINE_SHIFT as usize) & R4400Cache::DC_NUM_LINES_MASK;
        let v0 = kseg0((set << R4400Cache::DC_LINE_SHIFT as usize) as u32);
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
        let set = (phys as usize >> R4400Cache::IC_LINE_SHIFT as usize) & R4400Cache::IC_NUM_LINES_MASK;
        let iv = kseg0((set << R4400Cache::IC_LINE_SHIFT as usize) as u32);
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
    fn cache_op_index_wbinv_l2() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let phys: u32 = 0x6000;
        let virt = kseg0(phys);
        // Write into cache — lands in both L1D and L2.
        let _ = cache.write::<4>(virt, phys as u64, 0x5555_AAAA_u64);
        // Flush L1D first (Index_WBInv), so data propagates to L2.
        let dc_set = (phys as usize >> R4400Cache::DC_LINE_SHIFT as usize) & R4400Cache::DC_NUM_LINES_MASK;
        for way in 0..R4400Cache::DC_WAYS {
            let nls = R4400Cache::DC_NUM_LINES_SHIFT as usize;
            let ls  = R4400Cache::DC_LINE_SHIFT as usize;
            let idx_addr = (way << (nls + ls)) | (dc_set << ls);
            let v = kseg0(idx_addr as u32);
            cache.cache_op(C_IWBINV | CACH_PD, v, v & 0x1FFF_FFFF);
        }

        // Now Index_WBInvalidate the L2 line — flushes L2 dirty data to memory.
        let l2_set = (phys as usize >> R4400Cache::L2_LINE_SHIFT as usize) & (R4400Cache::L2_SIZE / R4400Cache::L2_LINE - 1);
        let lp = (l2_set * R4400Cache::L2_LINE) as u64;
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

        // R4400 L2 is inclusive: invalidation cascades to L1D, so the read refills.
        let r = cache.read::<4>(virt, phys as u64);
        assert_eq!(r.data as u32, 0x1234_5678,
            "Hit_Inv(SD) did not cascade invalidation to L1D (inclusive L2)");
    }

    /// Hit_WBInvalidate L2: writes back dirty L2 line to memory.
    #[test]
    fn cache_op_hit_wbinv_l2() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let phys: u32 = 0x8000;
        let virt = kseg0(phys);
        // Write into cache (dirty in L1D; L2 gets dirty during L1D writeback).
        let _ = cache.write::<4>(virt, phys as u64, 0x1122_3344_u64);
        // Flush L1D to make L2 dirty.
        let dc_set = (phys as usize >> R4400Cache::DC_LINE_SHIFT as usize) & R4400Cache::DC_NUM_LINES_MASK;
        for way in 0..R4400Cache::DC_WAYS {
            let nls = R4400Cache::DC_NUM_LINES_SHIFT as usize;
            let ls  = R4400Cache::DC_LINE_SHIFT as usize;
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
    fn cache_op_ilt_ist_l2() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let phys: u32 = 0x9000;
        let virt = kseg0(phys);
        // Populate L2 by doing a read.
        let _ = cache.read::<4>(virt, phys as u64);

        let l2_set = (phys as usize >> R4400Cache::L2_LINE_SHIFT as usize) & (R4400Cache::L2_SIZE / R4400Cache::L2_LINE - 1);
        let lp = (l2_set * R4400Cache::L2_LINE) as u64;

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
            (phys0 as usize >> R4400Cache::DC_LINE_SHIFT as usize) & R4400Cache::DC_NUM_LINES_MASK,
            (phys1 as usize >> R4400Cache::DC_LINE_SHIFT as usize) & R4400Cache::DC_NUM_LINES_MASK,
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
        let set = (phys0 as usize >> R4400Cache::DC_LINE_SHIFT as usize) & R4400Cache::DC_NUM_LINES_MASK;
        let inv0 = kseg0((set << R4400Cache::DC_LINE_SHIFT as usize) as u32); // bit14=0
        assert_eq!(inv0 & (1 << 14), 0, "inv0 must have bit14=0");
        cache.cache_op(C_IWBINV | CACH_PD, inv0, inv0 & 0x1FFF_FFFF);

        // Way0 was invalidated — refill brings 0xAAAA_0001 from L2.
        // (L2 still has old value since L1D write was clean → no dirty L2).
        let r0 = cache.read::<4>(virt0, phys0 as u64);
        assert_eq!(r0.data as u32, 0xAAAA_0001, "Way0 not invalidated by Index_WBInv with bit14=0");

        // Way1 should be unaffected — still holds 0xBBBB_0002.
        if R4400Cache::IS_R5K {
        {
            let r1 = cache.read::<4>(virt1, phys1 as u64);
            assert_eq!(r1.data as u32, 0xBBBB_0002,
                "Way1 was incorrectly evicted by Index_WBInv targeting way0");
        }
        } else {
        {
            // R4K single-way: both addresses alias, result is implementation-defined.
            let _ = cache.read::<4>(virt1, phys1 as u64);
        }
        }

        // Index_Invalidate L1I using bit14=1 address — should not affect way0 L1I line.
        let ic_set = (phys1 as usize >> R4400Cache::IC_LINE_SHIFT as usize) & R4400Cache::IC_NUM_LINES_MASK;
        let inv1 = kseg0(((ic_set << R4400Cache::IC_LINE_SHIFT as usize) | (1 << 14)) as u32); // bit14=1
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
                    let dc_set = ((phys as usize) >> R4400Cache::DC_LINE_SHIFT) & R4400Cache::DC_NUM_LINES_MASK;
                    let dc_v0 = kseg0((dc_set << R4400Cache::DC_LINE_SHIFT as usize) as u32);
                    let dc_v1 = kseg0(((dc_set << R4400Cache::DC_LINE_SHIFT as usize) | (1 << 14)) as u32);
                    cache.cache_op(C_IINV | CACH_PD, dc_v0, dc_v0 & 0x1FFF_FFFF);
                    cache.cache_op(C_IINV | CACH_PD, dc_v1, dc_v1 & 0x1FFF_FFFF);
                    // L2 hit-writeback (single-way, hit op fine).
                    cache.cache_op(C_HWBINV | CACH_SD, phys as u64, phys as u64);
                    // Index_Inv L1I both ways.
                    let ic_set = ((phys as usize) >> R4400Cache::IC_LINE_SHIFT) & R4400Cache::IC_NUM_LINES_MASK;
                    let ic_v0 = kseg0((ic_set << R4400Cache::IC_LINE_SHIFT as usize) as u32);
                    let ic_v1 = kseg0(((ic_set << R4400Cache::IC_LINE_SHIFT as usize) | (1 << 14)) as u32);
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
    /// A CPU model with no secondary cache must be distinguishable from one that
    /// has it, because the PROM decides whether to trust the EEPROM's CACHSZ word
    /// on exactly that. Getting this wrong does not fail any unit test or either
    /// bare-metal suite — it fails when IRIX boots and starts managing a cache
    /// that is not there. See Machine::new's CACHSZ handling.
    #[test]
    fn only_the_model_with_a_secondary_cache_reports_one() {
        assert!(<R4400Cache as MipsCache>::L2_SIZE > 0, "R4400 has a 1 MB L2");
        assert_eq!(<R5000Cache as MipsCache>::L2_SIZE, 0, "R5000 (Indy) has no secondary cache");
    }

    /// `writeback(Some(virt), phys, size)`: dirty lines in range land in memory
    /// and are no longer reported dirty; nothing outside the range is touched.
    #[test]
    // tcache writes through to RAM, so "memory is still stale until an explicit
    // writeback" — the property this test asserts — is false by construction.
    // The test still describes the real cache correctly; it simply does not
    // apply when the cache holds no data.
    #[cfg_attr(feature = "tcache", ignore = "write-through: memory is never stale under tcache")]
    fn writeback_virt_range_flushes_dirty_lines() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let base: u32 = 0x1000;
        // Two full L1D lines (16 bytes each) inside the range, one word in a
        // different L2 line entirely (writeback's cascade=true flushes the
        // whole containing L2 line, so an "outside" address must sit outside
        // that 128-byte L2 line too, not just outside the 32-byte L1D range,
        // for isolation to be a meaningful thing to assert).
        for i in 0..8u32 {
            let phys = base + i * 4;
            let _ = cache.write::<4>(kseg0(phys), phys as u64, (0x1000_0000 + i) as u64);
        }
        let outside_phys = base + R4400Cache::L2_LINE as u32;
        let _ = cache.write::<4>(kseg0(outside_phys), outside_phys as u64, 0xdead_beef);

        assert_eq!(mem_read(&mem, base), 0, "memory must still be stale before writeback");

        // Both 16-byte lines fall in the same 128-byte L2 line, so writing
        // back the first one (cascade=true) also flushes the L2 line — which
        // sweeps in the second L1D line as a side effect (writeback_l2_line's
        // own "flush dirty L1-D sub-lines into L2 first" pre-pass). So the
        // *count* of L1D lines this call itself finds still-dirty can be 1
        // even though both end up correctly in memory; assert on the actual
        // effect (data landed, dirty cleared) rather than the exact count.
        let n = cache.writeback(Some(kseg0(base) as u64), base as u64, 32);
        assert!(n >= 1, "expected at least one dirty line written back, got {}", n);

        for i in 0..8u32 {
            let phys = base + i * 4;
            assert_eq!(mem_read(&mem, phys), 0x1000_0000 + i,
                "phys={:#010x} not flushed to memory by writeback", phys);
        }
        // The word outside the requested range must remain stale in memory.
        assert_eq!(mem_read(&mem, outside_phys), 0,
            "writeback touched a line outside the requested range");

        // Re-probing the same range should now report nothing dirty.
        let n2 = cache.writeback(Some(kseg0(base) as u64), base as u64, 32);
        assert_eq!(n2, 0, "second writeback of the same range should find nothing dirty");
    }

    /// `writeback(None, phys, size)`: without a virtual address, the phys-only
    /// path must still find the dirty line via VIPT alias scanning and flush it.
    #[test]
    // tcache writes through to RAM, so "memory is still stale until an explicit
    // writeback" — the property this test asserts — is false by construction.
    // The test still describes the real cache correctly; it simply does not
    // apply when the cache holds no data.
    #[cfg_attr(feature = "tcache", ignore = "write-through: memory is never stale under tcache")]
    fn writeback_phys_only_finds_line_via_alias_scan() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let phys: u32 = 0x2040;
        let virt = kseg0(phys);
        let _ = cache.write::<4>(virt, phys as u64, 0xcafe_1234);
        assert_eq!(mem_read(&mem, phys), 0, "memory must still be stale before writeback");

        let n = cache.writeback(None, phys as u64, R4400Cache::DC_LINE as u64);
        assert_eq!(n, 1, "phys-only writeback should find and flush the one dirty line");
        assert_eq!(mem_read(&mem, phys), 0xcafe_1234,
            "phys-only writeback did not reach memory");
    }

    /// `writeback` must also flush a dirty L2 line that has no matching L1D
    /// line anymore (L1D already wrote through to L2 and was invalidated).
    #[test]
    // tcache writes through to RAM, so "memory is still stale until an explicit
    // writeback" — the property this test asserts — is false by construction.
    // The test still describes the real cache correctly; it simply does not
    // apply when the cache holds no data.
    #[cfg_attr(feature = "tcache", ignore = "write-through: memory is never stale under tcache")]
    fn writeback_flushes_dirty_l2_line() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let phys: u32 = 0x3000;
        let virt = kseg0(phys);
        let _ = cache.write::<4>(virt, phys as u64, 0x5555_aaaa);
        // Push L1D's dirty line into L2 and drop it from L1D, leaving only L2 dirty.
        cache.cache_op(C_IWBINV | CACH_PD, virt, phys as u64 & 0x1FFF_FFFF);
        assert_eq!(mem_read(&mem, phys), 0, "L2 writeback should not have happened yet");

        let n = cache.writeback(Some(virt), phys as u64, R4400Cache::L2_LINE as u64);
        assert!(n >= 1, "expected at least the L2 line to be written back");
        assert_eq!(mem_read(&mem, phys), 0x5555_aaaa,
            "writeback did not flush the dirty L2 line to memory");
    }

}

#[cfg(all(test, feature = "tcache"))]
mod tcache_tests {
    //! tcache must be *invisible*: the same access sequence has to produce the
    //! same results with and without transparent lines, and RAM has to end up
    //! byte-identical. These tests run the real cache and a tcache-enabled
    //! cache side by side over the same sequence and diff both.
    use super::*;
    use crate::ppmem::{MappedMemory, PpMemSpace, PpMemory};
    use crate::traits::BusDevice;
    use std::sync::Arc;

    // The bitmap marks a 64MB region only when the *whole* region is mapped
    // (docs/ppmem-design.md §3.1), so a bank smaller than 64MB would leave the
    // bitmap empty and silently make every test below a no-tcache test.
    const BANK_MB: usize = 64;
    const REGION: u64 = 64 * 1024 * 1024;
    const BASE: u64 = 0x0800_0000; // LOMEM — 64MB-aligned, so exactly one bit

    fn kseg0(phys: u64) -> u64 {
        0x8000_0000u64 | (phys & 0x0FFF_FFFF)
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    /// Build a bank + window, and a cache wired to that bank as downstream.
    /// `transparent` decides whether the cache is told about the window at all.
    fn setup(transparent: bool) -> (Arc<PpMemory>, PpMemSpace, R4400Cache) {
        let bank = Arc::new(PpMemory::new(BANK_MB));
        let space = PpMemSpace::over(std::slice::from_ref(&*bank)).expect("window");
        space.map_bank(0, BASE, REGION, REGION).expect("map");
        let cache = R4400Cache::new(bank.clone() as Arc<dyn BusDevice>);
        if transparent {
            unsafe {
                cache.set_tcache_window_impl(space.window_base());
                space.set_bitmap_sink2(cache.tc_bitmap_ptr());
            }
            // Guard against the whole suite silently degrading to a non-tcache
            // run if the bitmap ever stops covering this region.
            assert!(
                cache.tc_transparent(BASE),
                "test setup: BASE must be transparent or these tests prove nothing"
            );
        }
        (bank, space, cache)
    }

    /// The headline property: identical results and identical RAM, with and
    /// without tcache, over a mixed-width random access sequence that forces
    /// evictions.
    #[test]
    fn transparent_and_real_cache_agree() {
        let (bank_a, _sa, real) = setup(false);
        let (bank_b, _sb, tc) = setup(true);

        // The bank's own addressing is offset-based; the cache sees physical
        // addresses at BASE. Seed both banks identically.
        let mut seed = Rng(0x1234_5678_9ABC_DEF0);
        for i in 0..4096u32 {
            let v = seed.next() as u32;
            bank_a.write32(i * 4, v);
            bank_b.write32(i * 4, v);
        }

        let mut rng = Rng(0xDEAD_BEEF_CAFE_0001);
        for step in 0..20000 {
            let off = (rng.next() as u32) & 0x000F_FFF8; // 8-aligned, within 1MB
            let phys = BASE + off as u64;
            let va = kseg0(phys);
            match step % 7 {
                0 => {
                    let a = real.read::<8>(va, phys);
                    let b = tc.read::<8>(va, phys);
                    assert_eq!(a.status, b.status, "read64 status @{phys:#x}");
                    assert_eq!(a.data, b.data, "read64 data @{phys:#x} step {step}");
                }
                1 => {
                    let a = real.read::<4>(va, phys);
                    let b = tc.read::<4>(va, phys);
                    assert_eq!(a.data, b.data, "read32 @{phys:#x} step {step}");
                }
                2 => {
                    let a = real.read::<2>(va + 2, phys + 2);
                    let b = tc.read::<2>(va + 2, phys + 2);
                    assert_eq!(a.data, b.data, "read16 @{:#x} step {step}", phys + 2);
                }
                3 => {
                    let a = real.read::<1>(va + 5, phys + 5);
                    let b = tc.read::<1>(va + 5, phys + 5);
                    assert_eq!(a.data, b.data, "read8 @{:#x} step {step}", phys + 5);
                }
                4 => {
                    let v = rng.next();
                    assert_eq!(real.write::<8>(va, phys, v), tc.write::<8>(va, phys, v));
                }
                5 => {
                    let v = rng.next() as u32 as u64;
                    assert_eq!(real.write::<4>(va, phys, v), tc.write::<4>(va, phys, v));
                }
                _ => {
                    let v = rng.next() as u8 as u64;
                    assert_eq!(
                        real.write::<1>(va + 3, phys + 3, v),
                        tc.write::<1>(va + 3, phys + 3, v)
                    );
                }
            }
        }

        // Flush both so every dirty line lands in RAM, then diff the banks.
        real.writeback(None, BASE, 1024 * 1024);
        tc.writeback(None, BASE, 1024 * 1024);
        for i in 0..(256 * 1024u32) {
            let a = bank_a.read32(i * 4).data;
            let b = bank_b.read32(i * 4).data;
            assert_eq!(a, b, "RAM diverged at bank offset {:#x}", i * 4);
        }
    }

    /// R5000's 2-way paths are converted too, and were previously untested —
    /// every tcache test above builds an `R4400Cache`.
    #[test]
    fn r5000_transparent_and_real_cache_agree() {
        fn setup_r5k(transparent: bool) -> (Arc<PpMemory>, PpMemSpace, R5000Cache) {
            let bank = Arc::new(PpMemory::new(BANK_MB));
            let space = PpMemSpace::over(std::slice::from_ref(&*bank)).expect("window");
            space.map_bank(0, BASE, REGION, REGION).expect("map");
            let cache = R5000Cache::new(bank.clone() as Arc<dyn BusDevice>);
            if transparent {
                unsafe {
                    cache.set_tcache_window_impl(space.window_base());
                    space.set_bitmap_sink2(cache.tc_bitmap_ptr());
                }
            }
            (bank, space, cache)
        }

        let (bank_a, _sa, real) = setup_r5k(false);
        let (bank_b, _sb, tc) = setup_r5k(true);

        let mut seed = Rng(0x0F1E_2D3C_4B5A_6978);
        for i in 0..4096u32 {
            let v = seed.next() as u32;
            bank_a.write32(i * 4, v);
            bank_b.write32(i * 4, v);
        }

        let mut rng = Rng(0xFEED_FACE_1234_5678);
        for step in 0..20000 {
            let off = (rng.next() as u32) & 0x000F_FFF8;
            let phys = BASE + off as u64;
            let va = kseg0(phys);
            match step % 5 {
                0 => {
                    let a = real.read::<8>(va, phys);
                    let b = tc.read::<8>(va, phys);
                    assert_eq!(a.data, b.data, "r5k read64 @{phys:#x} step {step}");
                }
                1 => {
                    let a = real.read::<4>(va, phys);
                    let b = tc.read::<4>(va, phys);
                    assert_eq!(a.data, b.data, "r5k read32 @{phys:#x} step {step}");
                }
                2 => {
                    let a = real.read::<1>(va + 5, phys + 5);
                    let b = tc.read::<1>(va + 5, phys + 5);
                    assert_eq!(a.data, b.data, "r5k read8 step {step}");
                }
                3 => {
                    let v = rng.next();
                    assert_eq!(real.write::<8>(va, phys, v), tc.write::<8>(va, phys, v));
                }
                _ => {
                    let v = rng.next() as u32 as u64;
                    assert_eq!(real.write::<4>(va, phys, v), tc.write::<4>(va, phys, v));
                }
            }
        }

        real.writeback(None, BASE, 1024 * 1024);
        tc.writeback(None, BASE, 1024 * 1024);
        for i in 0..(256 * 1024u32) {
            assert_eq!(
                bank_a.read32(i * 4).data,
                bank_b.read32(i * 4).data,
                "r5k RAM diverged at bank offset {:#x}",
                i * 4
            );
        }
    }

    /// R5000 keeps its decoded slots in `ic_instrs`, filled from L2 data that
    /// tcache no longer populates — they must come from RAM instead.
    #[test]
    fn r5000_fetch_decodes_real_instructions() {
        let bank = Arc::new(PpMemory::new(BANK_MB));
        let space = PpMemSpace::over(std::slice::from_ref(&*bank)).unwrap();
        space.map_bank(0, BASE, REGION, REGION).unwrap();
        let tc = R5000Cache::new(bank.clone() as Arc<dyn BusDevice>);
        unsafe {
            tc.set_tcache_window_impl(space.window_base());
            space.set_bitmap_sink2(tc.tc_bitmap_ptr());
        }

        bank.write32(0x7000, 0x2402_0055);
        let phys = BASE + 0x7000;
        let r = tc.fetch(kseg0(phys), phys);
        assert_eq!(r.status, EXEC_COMPLETE, "r5k fetch failed");
        assert_eq!(
            unsafe { (*r.instr).raw },
            0x2402_0055,
            "r5k L1I fill decoded zeros — it read l2.data instead of RAM"
        );
    }

    /// A window write stores straight into RAM, bypassing `BusDevice` — so the
    /// gen bump `PpMemory::write*` would have done has to happen in the cache.
    /// Without it jitv2 keeps executing compiled code for a modified page.
    #[cfg(feature = "jitv2")]
    #[test]
    fn window_write_bumps_the_jitv2_gen_counter() {
        use std::sync::atomic::Ordering;
        let bank = Arc::new(PpMemory::new(BANK_MB));
        let space = PpMemSpace::over(std::slice::from_ref(&*bank)).unwrap();
        space.map_bank(0, BASE, REGION, REGION).unwrap();
        let tc = R4400Cache::new(bank.clone() as Arc<dyn BusDevice>);
        unsafe {
            tc.set_tcache_window_impl(space.window_base());
            space.set_bitmap_sink2(tc.tc_bitmap_ptr());
            tc.set_tcache_gen_window_impl(space.gen_window_base());
        }

        let phys = BASE + 0x3000;
        let va = kseg0(phys);
        let gen = |p: u64| unsafe {
            (*space.gen_window_base().add((p >> 12) as usize)).load(Ordering::Relaxed)
        };

        let before = gen(phys);
        tc.write::<4>(va, phys, 0xABCD_1234);
        assert!(
            gen(phys) > before,
            "window write did not bump the gen counter — jitv2 would run stale code"
        );

        // A read must not bump it.
        let after_write = gen(phys);
        let _ = tc.read::<4>(va, phys);
        assert_eq!(gen(phys), after_write, "read must not bump the gen counter");

        // A write to a different page must not touch this one's counter.
        let other = BASE + 0x9000;
        tc.write::<4>(kseg0(other), other, 0x1);
        assert_eq!(
            gen(phys),
            after_write,
            "write to another page bumped the wrong counter"
        );
        assert!(gen(other) > 0, "the written page's own counter did not move");
    }

    /// The cache holds the bitmap inline, so a remap must publish *into the
    /// cache*, not merely into `PpMemSpace`. If that second sink is ever
    /// dropped the cache silently keeps a stale bitmap and routes accesses to
    /// the wrong place — with no error anywhere.
    #[test]
    fn remaps_publish_into_the_caches_inline_bitmap() {
        let bank = Arc::new(PpMemory::new(BANK_MB));
        let space = PpMemSpace::over(std::slice::from_ref(&*bank)).unwrap();
        let cache = R4400Cache::new(bank.clone() as Arc<dyn BusDevice>);
        unsafe {
            cache.set_tcache_window_impl(space.window_base());
            space.set_bitmap_sink2(cache.tc_bitmap_ptr());
        }

        // Nothing mapped yet: the cache must see an empty bitmap.
        assert_eq!(unsafe { *cache.tc_bitmap_ptr() }, 0);
        assert!(!cache.tc_transparent(BASE));

        // Map a bank — the cache's inline copy must update.
        space.map_bank(0, BASE, REGION, REGION).unwrap();
        assert_eq!(
            unsafe { *cache.tc_bitmap_ptr() },
            space.mapped_bitmap(),
            "remap did not reach the cache's inline bitmap"
        );
        assert!(cache.tc_transparent(BASE), "cache still thinks BASE is unmapped");

        // ...and clearing must too.
        space.clear_mappings();
        assert_eq!(unsafe { *cache.tc_bitmap_ptr() }, 0, "clear did not reach the cache");
        assert!(!cache.tc_transparent(BASE));
    }

    /// Sub-word accesses are where a layout mistake shows up: ppmem swizzles on
    /// a 4-byte boundary, the cache's own data array on an 8-byte one.
    #[test]
    fn subword_accesses_match_exactly() {
        let (_ba, _sa, real) = setup(false);
        let (_bb, _sb, tc) = setup(true);

        let phys = BASE + 0x2000;
        let va = kseg0(phys);
        // Establish a known doubleword through both.
        real.write::<8>(va, phys, 0x0102_0304_0506_0708);
        tc.write::<8>(va, phys, 0x0102_0304_0506_0708);

        for k in 0..8u64 {
            let a = real.read::<1>(va + k, phys + k).data;
            let b = tc.read::<1>(va + k, phys + k).data;
            assert_eq!(a, b, "byte {k} mismatch");
            // MIPS big-endian: byte 0 of 0x0102030405060708 is 0x01.
            assert_eq!(a, (k + 1) as u64, "byte {k} is not MIPS big-endian order");
        }
        for k in [0u64, 2, 4, 6] {
            let a = real.read::<2>(va + k, phys + k).data;
            let b = tc.read::<2>(va + k, phys + k).data;
            assert_eq!(a, b, "half {k} mismatch");
        }
        for k in [0u64, 4] {
            let a = real.read::<4>(va + k, phys + k).data;
            let b = tc.read::<4>(va + k, phys + k).data;
            assert_eq!(a, b, "word {k} mismatch");
        }
    }

    /// A transparent line must never be written back — but the dirty bit still
    /// has to clear, so the guest's CACHE ops behave.
    #[test]
    fn transparent_writeback_clears_dirty_without_copying() {
        let (bank, _s, tc) = setup(true);
        let phys = BASE + 0x3000;
        let va = kseg0(phys);

        tc.write::<4>(va, phys, 0xAABB_CCDD);
        // The write went straight to RAM, before any writeback.
        assert_eq!(
            bank.read32(0x3000).data,
            0xAABB_CCDD,
            "transparent write did not reach RAM immediately"
        );
        tc.writeback(None, phys, 64);
        assert_eq!(bank.read32(0x3000).data, 0xAABB_CCDD, "writeback corrupted RAM");
    }

    /// An L1D-origin fill must NOT leave L2 claiming to hold valid decoded
    /// instructions — otherwise a later L1I fetch would dispatch whatever
    /// stale words happened to be in `l2.instrs`.
    #[test]
    fn data_fill_does_not_mark_l2_as_holding_code() {
        let (_b, _s, tc) = setup(true);
        let phys = BASE + 0x5000;
        let va = kseg0(phys);

        // Touch it as data first.
        let _ = tc.read::<8>(va, phys);
        let l2_idx = tc.l2.get_index(phys);
        let tag: L2Tag = tc.l2.get_tag(l2_idx);
        assert!(
            !tag.has_code(),
            "a transparent data fill must not claim to hold decoded instructions"
        );
    }

    /// Fetching must still work after a data access poisoned the L2 line —
    /// the has_code gate has to force a real instruction fill.
    #[test]
    fn fetch_after_data_access_still_decodes_correctly() {
        let (bank, _s, tc) = setup(true);
        let phys = BASE + 0x6000;
        let va = kseg0(phys);

        // Plant a recognisable instruction word in RAM.
        bank.write32(0x6000, 0x2402_1234); // addiu v0, zero, 0x1234
        // Touch the line as data — under tcache this installs an L2 tag with
        // no decoded slots.
        let _ = tc.read::<8>(va, phys);
        // Now fetch it. The has_code gate must force a real instruction fill.
        let r = tc.fetch(va, phys);
        assert_eq!(r.status, EXEC_COMPLETE, "fetch should succeed");
        let raw = unsafe { (*r.instr).raw };
        assert_eq!(raw, 0x2402_1234, "fetch returned stale/garbage instruction");
        let tag: L2Tag = tc.l2.get_tag(tc.l2.get_index(phys));
        assert!(tag.has_code(), "instruction fill must set has_code");
    }

    /// A transparent write over a line L2 believes holds code must clear the
    /// flag, so the next fetch re-reads rather than dispatching stale words.
    #[test]
    fn transparent_write_clears_l2_code_flag() {
        let (bank, _s, tc) = setup(true);
        let phys = BASE + 0x7000;
        let va = kseg0(phys);

        bank.write32(0x7000, 0x2402_1111);
        let r = tc.fetch(va, phys);
        assert_eq!(unsafe { (*r.instr).raw }, 0x2402_1111);
        assert!(tc.l2.get_tag(tc.l2.get_index(phys)).has_code());

        // Self-modifying write through the data path.
        tc.write::<4>(va, phys, 0x2402_2222);
        assert!(
            !tc.l2.get_tag(tc.l2.get_index(phys)).has_code(),
            "write must drop has_code so the next fetch re-decodes"
        );
        // A re-fetch *without* a CACHE flush still sees the old word, because
        // L1I's own tag is still valid — exactly as on real hardware, where
        // self-modifying code must flush the I-cache. tcache does not change
        // that contract; `has_code` only stops L2 from serving stale *decoded*
        // data once L1I does miss.
        tc.invalidate_l1i_line(tc.ic.get_index(va), false);
        let r2 = tc.fetch(va, phys);
        assert_eq!(
            unsafe { (*r2.instr).raw },
            0x2402_2222,
            "after an I-cache flush the fetch must see the new word"
        );
    }

    /// The ELF-relocation / module-load shape: fetch a line (L2 gets has_code),
    /// then write new code through the *data* path, then fetch again.
    ///
    /// Pre-tcache this was safe because `writeback_l1d_line` re-synced
    /// `l2.instrs` on every flush. A transparent line never writes back to L2,
    /// so that re-sync is gone and `has_code` has to carry the invalidation.
    #[test]
    fn code_written_through_data_path_is_not_served_stale() {
        let (bank, _s, tc) = setup(true);
        let phys = BASE + 0x9000;
        let va = kseg0(phys);

        // 1. Original instruction, fetched -> L2 caches decoded slots.
        bank.write32(0x9000, 0x2402_0001);
        let r1 = tc.fetch(va, phys);
        assert_eq!(unsafe { (*r1.instr).raw }, 0x2402_0001);

        // 2. Kernel rewrites that word through the data path (relocation).
        tc.write::<4>(va, phys, 0x2402_0002);

        // 3. Flush L1I only — as a guest would after writing code. L2 must not
        //    hand back the pre-write decoded slot.
        tc.invalidate_l1i_line(tc.ic.get_index(va), false);
        let r2 = tc.fetch(va, phys);
        assert_eq!(
            unsafe { (*r2.instr).raw },
            0x2402_0002,
            "L2 served decoded instructions from before the write"
        );
    }

    /// Whatever tag state a CACHE op installs, the *data path* is decided by
    /// the address alone: a transparent address reads and writes RAM. Tag bits
    /// are guest-visible state and must never steer storage.
    #[test]
    fn cache_ops_do_not_divert_transparent_data_away_from_ram() {
        let (bank, _s, tc) = setup(true);
        let phys = BASE + 0xA000;
        let va = kseg0(phys);
        assert!(tc.tc_transparent(phys), "test setup: address must be transparent");

        // Create Dirty Exclusive, then write: must still land in RAM.
        tc.cache_op((C_CDX << 2) | CACH_PD, va, phys);
        tc.write::<4>(va, phys, 0x1234_5678);
        assert_eq!(
            bank.read32(0xA000).data,
            0x1234_5678,
            "write after C_CDX did not reach RAM"
        );

        // Index_Store_Tag, then write: likewise.
        tc.cache_op((C_IST << 2) | CACH_PD, va, phys);
        tc.write::<4>(va, phys, 0x8765_4321);
        assert_eq!(
            bank.read32(0xA000).data,
            0x8765_4321,
            "write after C_IST did not reach RAM"
        );

        // And reads come from RAM, not from any cache array.
        bank.write32(0xA000, 0x0BAD_C0DE);
        assert_eq!(
            tc.read::<4>(va, phys).data as u32,
            0x0BAD_C0DE,
            "read did not observe RAM"
        );
    }

    /// A write to a transparent address reaches RAM immediately — there is no
    /// `dc.data` to buffer it in — and must drop L2's claim to hold decoded
    /// instructions for that line.
    #[test]
    fn write_to_transparent_address_reaches_ram_and_invalidates_code() {
        let (bank, _s, tc) = setup(true);
        let phys = BASE + 0xA000;
        let va = kseg0(phys);

        bank.write32(0xA000, 0x2402_0011);
        let r1 = tc.fetch(va, phys);
        assert_eq!(unsafe { (*r1.instr).raw }, 0x2402_0011);
        assert!(tc.l2.get_tag(tc.l2.get_index(phys)).has_code());

        tc.write::<4>(va, phys, 0x2402_0022);
        assert_eq!(
            bank.read32(0xA000).data,
            0x2402_0022,
            "a transparent write must land in RAM immediately, unbuffered"
        );
        assert!(
            !tc.l2.get_tag(tc.l2.get_index(phys)).has_code(),
            "write did not invalidate L2's decoded slots"
        );

        tc.invalidate_l1i_line(tc.ic.get_index(va), false);
        let r2 = tc.fetch(va, phys);
        assert_eq!(
            unsafe { (*r2.instr).raw },
            0x2402_0022,
            "fetch after write returned stale decoded instructions"
        );
    }

    /// A DIRTY L2 line over a transparent line is a *normal, harmless* state:
    /// tcache changes where data lives, not the state machine. Flushing such a
    /// line must move no data (there is none) and must not disturb RAM.
    #[test]
    fn dirty_l2_over_transparent_line_does_not_touch_ram() {
        let (bank, _s, tc) = setup(true);
        let phys = BASE + 0xC000;
        let va = kseg0(phys);

        bank.write32(0xC000, 0x1111_1111);
        let _ = tc.fetch(va, phys);
        let l2_idx = tc.l2.get_index(phys);

        // Force L2 dirty, as an L1D writeback does.
        let mut l2_tag: L2Tag = tc.l2.get_tag(l2_idx);
        l2_tag.set_cs(L2_CS_DIRTY_EXCLUSIVE);
        tc.l2.set_tag(l2_idx, l2_tag);

        // Write through the transparent line: lands in RAM.
        tc.write::<4>(va, phys, 0x2222_2222);
        assert_eq!(bank.read32(0xC000).data, 0x2222_2222);

        // Evicting the dirty L2 line must be a no-op for memory.
        tc.writeback_l2_line(l2_idx);
        assert_eq!(
            bank.read32(0xC000).data,
            0x2222_2222,
            "L2 flush moved data it should not have — a transparent line has none"
        );
        // ...and the state transition still happened.
        let after: L2Tag = tc.l2.get_tag(l2_idx);
        assert!(
            after.cs() == L2_CS_CLEAN_EXCLUSIVE || after.cs() == L2_CS_SHARED,
            "L2 should have been demoted to clean after writeback, cs={}",
            after.cs()
        );
    }


    /// The corruption itself: an L2 line that is DIRTY while the L1D line over
    /// it is transparent. L2's `l2.data` was never updated by the transparent
    /// writes, so evicting it writes stale bytes over correct RAM.
    #[test]
    fn dirty_l2_over_transparent_line_corrupts_ram() {
        let (bank, _s, tc) = setup(true);
        let phys = BASE + 0xC000;
        let va = kseg0(phys);

        // Get L2 holding the line with real data (instruction-origin fill).
        bank.write32(0xC000, 0x1111_1111);
        let _ = tc.fetch(va, phys);
        let l2_idx = tc.l2.get_index(phys);

        // Force L2 dirty, as an L1D writeback of a *backed* line would.
        let mut l2_tag: L2Tag = tc.l2.get_tag(l2_idx);
        l2_tag.set_cs(L2_CS_DIRTY_EXCLUSIVE);
        tc.l2.set_tag(l2_idx, l2_tag);

        // Now write through a transparent L1D line: goes to RAM, not l2.data.
        tc.write::<4>(va, phys, 0x2222_2222);
        assert_eq!(bank.read32(0xC000).data, 0x2222_2222);

        // Evict L2. It believes it owns dirty data and flushes stale bytes.
        tc.writeback_l2_line(l2_idx);
        assert_eq!(
            bank.read32(0xC000).data,
            0x2222_2222,
            "dirty L2 eviction clobbered the transparent write with stale l2.data"
        );
    }

    /// An address outside any fully-mapped region must fall back to the real
    /// cache path, tags marked `backed`.
    #[test]
    fn unmapped_regions_fall_back_to_real_cache() {
        let (_b, space, tc) = setup(true);
        // PROM space is never claimed by the bitmap.
        assert!(!tc.tc_transparent(0x1FC0_0000), "PROM must not be transparent");
        assert!(!tc.tc_transparent(0x1FA0_0000), "MC must not be transparent");
        assert!(tc.tc_transparent(BASE), "LOMEM should be transparent");
        // Sanity: the bitmap really is populated.
        assert_ne!(space.mapped_bitmap(), 0);
    }

    /// With no window set, tcache is inert and every access is a real one.
    #[test]
    fn inert_without_a_window() {
        let (_b, _s, tc) = setup(false);
        assert!(!tc.tc_transparent(BASE), "must be inert before set_tcache_window");
    }
}

#[cfg(all(test, feature = "tcache", feature = "developer"))]
mod tcache_hitrate {
    //! Is the transparent path actually taken? A pure-overhead benchmark
    //! result would look identical whether the gate never fires or the idea
    //! simply does not pay, so measure the gate directly.
    use super::*;
    use crate::ppmem::{MappedMemory, PpMemSpace, PpMemory};
    use crate::traits::BusDevice;
    use std::sync::Arc;

    /// How many accesses are hits (gate cost, no fill) versus misses (where the
    /// skipped memcpy would pay)? This ratio decides whether tcache can ever win.
    #[test]
    fn hit_to_miss_ratio_on_sequential_traffic() {
        let bank = Arc::new(PpMemory::new(64));
        let space = PpMemSpace::over(std::slice::from_ref(&*bank)).unwrap();
        let region = 64 * 1024 * 1024u64;
        space.map_bank(0, 0x0800_0000, region, region).unwrap();
        let cache = R4400Cache::new(bank.clone() as Arc<dyn BusDevice>);
        unsafe { cache.set_tcache_window_impl(space.window_base(), space.bitmap_ptr()) };

        // Walk 1MB sequentially by doubleword: a 16-byte line means one fill
        // per 2 accesses at worst, and far fewer once lines are resident.
        let n = 128 * 1024u64;
        let mut fills = 0u64;
        for i in 0..n {
            let phys = 0x0800_0000 + i * 8;
            let va = 0x8000_0000u64 | (phys & 0x0FFF_FFFF);
            let idx = cache.dc.get_index(va);
            let before = cache.dc.get_tag(idx).matches_phys(phys);
            let _ = cache.read::<8>(va, phys);
            if !before { fills += 1; }
        }
        println!(
            "sequential 8-byte reads: {n} accesses, {fills} fills ({:.1}% miss)",
            100.0 * fills as f64 / n as f64
        );
    }

    #[test]
    fn gate_fires_for_lomem_traffic() {
        let bank = Arc::new(PpMemory::new(64));
        let space = PpMemSpace::over(std::slice::from_ref(&*bank)).unwrap();
        let region = 64 * 1024 * 1024u64;
        space.map_bank(0, 0x0800_0000, region, region).unwrap();
        let cache = R4400Cache::new(bank.clone() as Arc<dyn BusDevice>);
        unsafe { cache.set_tcache_window_impl(space.window_base(), space.bitmap_ptr()) };

        let (p0, h0) = tc_stats();
        for i in 0..1000u64 {
            let phys = 0x0800_0000 + i * 8;
            let va = 0x8000_0000u64 | (phys & 0x0FFF_FFFF);
            cache.write::<8>(va, phys, i);
            let _ = cache.read::<8>(va, phys);
        }
        let (p1, h1) = tc_stats();
        let probes = p1 - p0;
        let hits = h1 - h0;
        println!("tcache gate: {probes} probes, {hits} hits ({:.1}%)",
                 100.0 * hits as f64 / probes.max(1) as f64);
        assert!(probes > 0, "gate never consulted");
        assert_eq!(hits, probes, "every LOMEM access should be transparent");
    }
}
