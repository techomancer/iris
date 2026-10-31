// MIPS R4000 Cache Implementation - Version 2
//
// This is a complete rewrite to properly support R4000 cache semantics:
// - Unified cache object containing L1-I, L1-D, and L2
// - Proper VIPT (Virtually Indexed, Physically Tagged) support
// - R4000-compliant tag format with PState bits
// - L2 can signal back to L1 for evictions

use crate::traits::{BusRead64, BusDevice, Resettable, BUS_OK, BUS_BUSY, BUS_ERR, BUS_VCE};
use crate::snapshot::{u32_slice_to_toml, u64_slice_to_toml, load_u32_slice, load_u64_slice, get_field, toml_bool, toml_u32, toml_u64, hex_u32, hex_u64};
use crate::cpu::mips_exec::{DecodedInstr, ExecStatus, EXEC_COMPLETE, EXEC_RETRY, exec_exception_const, EXC_VCEI, EXC_VCED, EXC_IBE, FLAG_NOT_DECODED, FLAG_IMM_IS_NEXT, EXEC_CACHE_ERR};
use crate::devlog::{LogModule, CACHE_LOG_HIT, CACHE_LOG_MISS, CACHE_LOG_OP, devlog_is_active, devlog_mask};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::cell::{Cell, UnsafeCell};
use bitfield::bitfield;

/// tcache, switchable at run time: `$tc` while the cache is transparent,
/// `$real` (the cache holds the data) otherwise. Without the `tcache` feature
/// this is `$real` alone, so such a build compiles exactly as before.
macro_rules! tc_or {
    ($s:expr, $tc:expr, $real:expr) => {{
        #[cfg(feature = "tcache")]
        { if $s.tc_live() { $tc } else { $real } }
        #[cfg(not(feature = "tcache"))]
        { $real }
    }};
}

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
pub use crate::cpu::mips_isa::{
    CACH_PI, CACH_PD, CACH_SI, CACH_SD,
    C_IINV, C_IWBINV, C_ILT, C_IST, C_CDX,
    C_HINV, C_HWBINV, C_FILL, C_HWB, C_HSV,
};

// R10000 reassigns cache operations 5, 6 and 7. Where an R4000 has
// Hit_Invalidate, Hit_Writeback_Invalidate and Hit_Writeback, an R10000 has a
// cache barrier and index-addressed load/store of the cache *data* array.
// Confirmed against NetBSD's mips/include/cache_r10k.h, and against the IP28
// PROM, whose secondary-cache SRAM test issues C_R10K_ISD(SD) — which this
// emulator was executing as a hit-writeback, so nothing was ever stored.
pub const C_R10K_CBARRIER: u32 = 5 << 2;
pub const C_R10K_ILD: u32 = 6 << 2;
pub const C_R10K_ISD: u32 = 7 << 2;

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
/// Geometry the jitv2 inline load/store path needs as codegen immediates.
/// `supported == false` means the cache cannot serve the inline path at all
/// (no cache, or a shape the first implementation does not cover, e.g. R5000's
/// 2-way L1-D — see docs/jit-inline-memory.md §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitDcGeometry {
    pub supported: bool,
    /// `addr >> line_shift & num_lines_mask` = tag index.
    pub line_shift: u32,
    pub num_lines_mask: u64,
    /// `virt_addr & data_mask` = byte offset into the data array.
    pub data_mask: u64,
    /// L2 present at all (`HAS_L2`). When false the inline tcache store skips
    /// the decoded-instruction invalidation entirely, matching
    /// `tc_invalidate_l2_code`'s own early return.
    pub has_l2: bool,
    /// `phys >> l2_line_shift & l2_num_lines_mask` = L2 tag index.
    pub l2_line_shift: u32,
    pub l2_num_lines_mask: u64,
    /// L1-D associativity: 1 (R4400) or 2 (R5000). With `ways == 2` the
    /// emitted guard probes both ways and, on a hit, updates the per-set LRU
    /// bit exactly as `ensure_l1d_line` does.
    pub ways: u32,
    /// `set | (way << num_lines_shift)` = extended tag index. Only meaningful
    /// when `ways > 1`.
    pub num_lines_shift: u32,
    /// No L1-D model at all: loads and stores go straight to memory, so under
    /// tcache the inline path is the ppmem-window half alone, with no tag
    /// probe, no LRU and no dirty bit (the R10000's shadow cache). The other
    /// fields are then unused.
    pub tagless: bool,
    /// Inline stores bump the page generation with an atomic add: other writers
    /// are active, and a plain add landing after theirs would move the counter
    /// backwards and let stale code publish.
    pub atomic_gen: bool,
}

impl JitDcGeometry {
    pub const fn unsupported() -> Self {
        Self {
            supported: false, line_shift: 0, num_lines_mask: 0, data_mask: 0,
            has_l2: false, l2_line_shift: 0, l2_num_lines_mask: 0,
            ways: 1, num_lines_shift: 0, tagless: false, atomic_gen: false,
        }
    }
}

/// `#[repr(C)]`: jitv2's inline load/store fast path addresses `ptag` and
/// `dirty` directly from compiled code at fixed byte offsets within the tag
/// array (docs/jit-inline-memory.md §3.1), the same requirement `MipsCore`
/// documents for itself. Offsets are taken with `offset_of!` at codegen time,
/// never hardcoded — but the layout must at least be stable and declared.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
#[repr(C)]
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
    ///
    /// A line carrying [`Self::SLOW`] never matches here, by design: that is
    /// how it reaches the miss path. Anything that must find the line
    /// regardless (CACHE hit ops, write-back and invalidation scans) uses
    /// [`Self::matches_line`].
    #[inline(always)]
    pub fn matches_phys(&self, phys_addr: u64) -> bool {
        self.ptag == (phys_addr & !0xFFF) | 1
    }

    /// Bit 1 of `ptag`: send every access to this line through the miss path.
    ///
    /// Parity forcing (Status.CE) and forced-parity lines need a check on
    /// each access, and the hit path cannot afford one. A marked line fails
    /// the exact compare in `matches_phys`, in the interpreter and in jitv2's
    /// inline guard alike, so the check lives in the miss path at no cost to
    /// anything else. Bit 1 of a page-aligned address is free, like bit 0.
    pub const SLOW: u64 = 2;

    /// [`Self::matches_phys`], ignoring the [`Self::SLOW`] mark.
    #[inline]
    pub fn matches_line(&self, phys_addr: u64) -> bool {
        self.ptag & !Self::SLOW == (phys_addr & !0xFFF) | 1
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

/// Even parity over `v`: the bit that makes the total count of ones even.
#[inline]
pub fn even_parity(v: u32) -> u32 { v.count_ones() & 1 }

/// R4000 secondary-cache tag check matrix, UM Figure 16-4. Entry `i` selects
/// the tag bits (CS:PIdx:STag, `[24:0]` as in Figure 11-6) that check bit `i`
/// covers. Every tag bit has a distinct weight-3 syndrome; see
/// `l2_tag_ecc_matrix_is_the_manuals`.
const L2_TAG_ECC_MASKS: [u32; 7] = [
    0x1ff1111, 0x045222f, 0x10a40ff, 0x09184f0, 0x0620f42, 0x114ff04, 0x0a8f888,
];

/// The 7 check bits the R4400 generates for a secondary tag.
///
/// Not stored: the hardware computes them whenever a tag is written and
/// ignores the ECC field of TagLo on `Index_Store_Tag` (UM p. 93), so
/// computing them when the tag is read back gives the same value with nothing
/// to keep in sync.
pub fn l2_tag_ecc(tag: u32) -> u32 {
    L2_TAG_ECC_MASKS.iter().enumerate()
        .fold(0, |ecc, (i, &m)| ecc | even_parity(tag & m) << i)
}

/// R4000 secondary-cache data check matrix, UM Figure 16-1: entry `i` selects
/// the data bits check bit `i` covers. Pinned by
/// `l2_data_ecc_matrix_is_the_manuals`.
const L2_DATA_ECC_MASKS: [u64; 8] = [
    0x145011110ff014ff, 0x24ff2222f000249f, 0x4c9f444400ff44d0, 0x84d08888ff0f8c50,
    0x0931f0ff11110b21, 0x0b22ff002222fa32, 0xfa24000f4444ff24, 0xff280ff088880928,
];

/// The 8 check bits the R4400 computes for a secondary-cache doubleword.
pub fn l2_data_ecc(v: u64) -> u8 {
    L2_DATA_ECC_MASKS.iter().enumerate()
        .fold(0, |ecc, (i, &m)| ecc | (((v & m).count_ones() & 1) as u8) << i)
}

/// R4400/R5000 Index_Store_Tag: the TagLo a later Index_Load_Tag returns,
/// without its check bits. A tag is only as wide as its fields, so this is
/// what a cache keeping raw TagLo words must store to read back what the
/// full model does (see `CpuCache::cache_op`, which keeps them as fields):
///
/// - primary: PTag[31:8] and PState[7:6]. An invalid line keeps nothing. Any
///   valid I-cache state reads back as 2; on the R5000, which has no Shared
///   state, a stored 1 reads back as 2.
/// - secondary (R4400): STag[31:13], SState[12:10] (reserved 1-3 read back as
///   0), PIdx[9:7].
pub fn r4k_store_taglo(r5k: bool, sel: u32, taglo: u32) -> u32 {
    if matches!(sel, CACH_SI | CACH_SD) {
        let state = match (taglo >> 10) & 7 { s @ 4..=7 => s, _ => 0 };
        return (taglo & 0xFFFF_E000) | state << 10 | (taglo & 0x380);
    }
    let pstate = match (taglo >> 6) & 3 {
        0 => return 0,
        _ if sel == CACH_PI => 2,
        1 if r5k => 2,
        p => p,
    };
    (taglo & 0xFFFF_FF00) | pstate << 6
}

/// [`r4k_store_taglo`]'s word as Index_Load_Tag returns it: with the check
/// bits the hardware generates (TagLo[0] P on a primary, TagLo[6:0] ECC on
/// the R4400 secondary).
pub fn r4k_load_taglo(r5k: bool, sel: u32, stored: u32) -> u32 {
    if matches!(sel, CACH_SI | CACH_SD) {
        let fields = (stored >> 10 & 7) << 22 | (stored >> 7 & 7) << 19 | stored >> 13;
        return stored | l2_tag_ecc(fields);
    }
    let pstate = (stored >> 6) & 3;
    let p = even_parity(stored >> 8)
        ^ if r5k { 0 } else if sel == CACH_PI { (pstate != 0) as u32 } else { even_parity(pstate) };
    stored | p
}

// L2 Cache Tag
//   [31:25] ECC  - Error correction code; not stored, see `l2_tag_ecc`
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
    /// Implemented virtual address bits.
    ///
    /// 40 on the R4x00/R5000, **44 on the R10000**. This is not cosmetic: it
    /// sets the width of XContext's BadVPN2 field and therefore where its
    /// Region and PTEBase fields sit, which is how a 64-bit kernel finds the
    /// page-table entry for a faulting address.
    const VA_BITS: u32 = 40;
    /// Implemented physical address bits: 36 on the R4x00/R5000, 40 on the
    /// R10000. LLAddr holds `PA[PA_BITS-1:4]`.
    const PA_BITS: u32 = 36;
    /// The bits of LLAddr that exist.
    const LLADDR_MASK: u64 = (1 << (Self::PA_BITS - 4)) - 1;
    /// Name as the guest and the benchmark report see it.
    const NAME: &'static str;
    /// "full" for the model whose arrays loads, stores and fetches consult;
    /// "shadow" for one that keeps only what CACHE ops observe. Recorded in
    /// snapshots: a full cache's state does not load into a shadow one.
    const CACHE_KIND: &'static str = "full";
    /// The part is in the R10000 family (R10000, R12000, R14000, R16000), which
    /// shares one programming model. Cache ops 5/6/7 carry their R10000
    /// meanings rather than their R4000 ones: `Index_Store_Data` takes its
    /// value from TagLo and `Index_Load_Data` returns into it, neither of which
    /// is true of the R4000 hit operations that share those encodings. The
    /// family also keeps every implemented VPN2 bit in EntryHi and the TLB
    /// (`VA_BITS`), carries the uncached attribute from EntryLo bits 63:62,
    /// keeps primary-cache diagnostic data, and leaves bits 63:32 alone on a
    /// partial LWR.
    const R10K_CACHE_OPS: bool = false;
    /// Bits `Index_Store_Tag` keeps for the secondary cache. All of them unless
    /// a part sets a mask. A 36-bit mask once inferred from the IP28 PROM's
    /// walking-0s phase broke its diagnostics: the truncation it reads back
    /// comes from the tag being assembled as `(TagHi << 32) | TagLo[31:0]`, not
    /// from the array, and the mask only discarded the MRU bit the PROM writes
    /// as TagHi[31]. Extending a mask to another part needs IP28 and Octane
    /// diagnostic comparisons.
    const SECONDARY_TAG_MASK: u64 = u64::MAX;
}

pub trait MipsCache: Send + Sync {
    /// Share the L1-I hit/fetch counters with the status display. No-op where absent.
    fn set_l1i_counters(&mut self, _hit: Arc<AtomicU64>, _fetch: Arc<AtomicU64>) {}

    /// Have `hook` called with TagLo on every Index Store Tag to the primary
    /// data cache: a machine whose firmware builds memory out of cache tags
    /// (the O2's POST stack, see `O2Memory::stack_tag_store`). No-op where
    /// absent. Set once, before the CPU runs.
    fn set_pd_tag_store_hook(&self, _hook: Box<dyn Fn(u32) + Send + Sync>) {}

    /// tcache: point the cache at ppmem's window and mapped-region bitmap, so
    /// cacheable RAM accesses read/write RAM directly instead of copying line
    /// data (docs/tcache-design.md). Default no-op — a cache with no
    /// transparent path simply ignores it.
    ///
    /// # Safety
    /// `base` must be ppmem's window base and `bitmap` a live `u64` that
    /// outlives this cache.
    unsafe fn set_tcache_window(&self, _base: *mut u8) {}

    /// Whether `fetch` of cacheable RAM only reads: no line fill, no LRU or
    /// tag change, no exception. Then a caller that does not need the
    /// instruction (jitv2 entering compiled code) may leave the fetch out.
    /// Default `false`: a model that keeps instruction-cache state does.
    fn fetch_is_pure(&self) -> bool { false }

    /// Write every dirty line back to memory, so RAM holds everything.
    /// Default no-op: a cache that holds no data has nothing to write.
    fn writeback_all(&self) {}

    /// tcache: pointer to this cache's inline mapped-region bitmap, for
    /// `MappedMemory::set_bitmap_sink2`. Null when the cache has no such field.
    fn tcache_bitmap_ptr(&self) -> *mut u64 { std::ptr::null_mut() }

    /// tcache: base of the ppmem window this cache reads/writes through
    /// (`tc_base + phys`). Null until `set_transparent_base` runs.
    fn tcache_base_ptr(&self) -> *mut u8 { std::ptr::null_mut() }

    /// tcache: base of the jitv2 per-page generation array (`[AtomicU64]`,
    /// indexed by `phys >> 12`). An inline store through the ppmem window
    /// must bump this or self-modifying code stops invalidating compiled
    /// regions — see `tc_bump_gen`. Null when jitv2/ppmem is not wired up.
    #[cfg(feature = "jitv2")]
    fn tcache_gen_ptr(&self) -> *mut u8 { std::ptr::null_mut() }

    // ---- jitv2 inline load/store fast path (docs/jit-inline-memory.md) ----
    //
    // Raw bases the JIT's compiled code indexes directly, mirrored into
    // `MipsCore` by `install_jit_mem_ptrs` so compiled code can reach them at
    // a fixed offset off `core_ptr`. Backing storage is a boxed slice with a
    // stable address, so caching the pointer is sound — but any path that can
    // reallocate it (snapshot restore, reconfigure) MUST re-install.
    //
    // Default null: a null `jit_dc_tags` is the signal that this cache cannot
    // support the inline path, and codegen must not emit it.

    /// L1-D-resident fast read for the jitv2 memory hooks: on a tag hit
    /// return the data, else `None` so the caller runs the full path.
    ///
    /// Deliberately does no fill, no LRU update, no breakpoint check and no
    /// bus access — each of those is a reason to decline instead. See
    /// docs/jit-inline-memory.md §7: the guard lives here, in one Rust
    /// function, rather than as per-access Cranelift IR.
    fn jit_fast_read<const SIZE: usize>(&self, _virt: u64, _phys: u64) -> Option<u64> { None }

    /// L1-D-resident fast write. Declines unless the line is present *and*
    /// already dirty: the clean->dirty transition is a tag read-modify-write
    /// the slow path owns.
    fn jit_fast_write<const SIZE: usize>(&self, _virt: u64, _phys: u64, _val: u64) -> bool { false }

    /// Base of the L2 tag array (`[L2Tag]`, one `u32` each). The inline
    /// tcache store path uses it to clear `has_code` for the written line,
    /// mirroring `tc_invalidate_l2_code`. Null when there is no L2.
    fn jit_l2_tags_ptr(&self) -> *mut u8 { std::ptr::null_mut() }

    /// Base of the L1-D tag array (`[L1DTag]`). Null if unsupported.
    fn jit_dc_tags_ptr(&self) -> *mut u8 { std::ptr::null_mut() }
    /// Number of L1-D tag entries, and the line shift used to index them.
    /// Codegen needs these as immediates; they are const-generic on the cache
    /// type and differ between R4400 and R5000.
    fn jit_dc_geometry(&self) -> JitDcGeometry { JitDcGeometry::unsupported() }

    /// Base of the L1-D per-set LRU bitmap (`[u64]`, one bit per set), for the
    /// 2-way inline path's hit-side LRU update. Null on a direct-mapped model,
    /// where there is no LRU state at all.
    fn jit_dc_lru_ptr(&self) -> *mut u8 { std::ptr::null_mut() }

    /// tcache + jitv2: hand the cache ppmem's generation-window base, so window
    /// writes bump the per-page counter jitv2 validates compiled code against.
    /// Default no-op.
    ///
    /// # Safety
    /// `gen_base` must be ppmem's gen window base and outlive this cache.
    #[cfg(feature = "jitv2")]
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

    /// Would the JIT's inline load/store fast path have handled this access
    /// without calling out to Rust? Side-effect-free predicate, `jitstats`
    /// only — it exists to size and tune the inline emit before (and while)
    /// building it, and doubles as the executable spec of what that emit must
    /// check.
    ///
    /// `STORE` selects the store predicate, which is strictly narrower: a
    /// store may only proceed inline on an already-dirty line, because the
    /// clean->dirty transition is a read-modify-write on the tag that is not
    /// worth emitting (the first store to a line calls out; the rest inline).
    ///
    /// The stages, in the order compiled code would test them:
    ///   1. L1D tag matches `phys_addr`      (one 64-bit compare)
    ///   2. store only: that line is dirty
    ///   3. tcache only: `phys_addr` is in a transparent (mapped RAM) region
    ///
    /// The nutlb hit that must precede all of this is counted separately, in
    /// the executor — the cache layer never sees translation outcomes.
    #[cfg(feature = "jitstats")]
    fn jit_probe<const STORE: bool>(&self, _virt_addr: u64, _phys_addr: u64) -> bool { false }

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
    /// Perform a CACHE operation. For `Index_Load_Tag` the return is the
    /// tag as software sees it, full width — an R10000 secondary tag does not
    /// fit in 32 bits.
    fn cache_op(&self, cache_op: u32, virt_addr: u64, phys_addr: u64) -> u64;

    /// CP0 `ECC` ($26) on the way in to a cache-data store.
    ///
    /// On an R10000 the check bits ride with the data: `Index_Store_Data`
    /// takes them from this register and `Index_Load_Data` returns them
    /// there. Models that do not keep them ignore both of these.
    fn set_cache_ecc(&self, _v: u32) {}
    /// The Status bits that govern cache error checking: CE (store the ECC
    /// register as check bits instead of computing them) and whether errors
    /// are detected at all (neither DE nor ERL set). Pushed by the executor on
    /// every Status change. Models that keep no check bits ignore it.
    fn set_error_control(&self, _ce: bool, _detect: bool) {}
    /// CacheErr for the Cache Error exception the last access reported
    /// (`EXEC_CACHE_ERR`).
    fn take_cache_err(&self) -> u32 { 0 }
    /// A Cache Error raised by the CACHE instruction just executed (an I-cache
    /// write-back of a line with bad parity), as an exception status, or 0.
    /// Reading it clears it.
    fn take_op_status(&self) -> u32 { 0 }
    /// CP0 `ECC` after the last cache operation.
    fn cache_op_ecc(&self) -> u32 { 0 }

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

    /// Atomically store `new` to the RAM word at `phys` if it still holds
    /// `old`. `None`: this model cannot, or `phys` is not RAM it can reach
    /// directly, and the caller stores the ordinary way. Only models that keep
    /// no data of their own can serve several CPUs.
    fn cas_ram32(&self, _phys: u64, _old: u32, _new: u32) -> Option<bool> { None }
    /// The doubleword form of [`Self::cas_ram32`].
    fn cas_ram64(&self, _phys: u64, _old: u64, _new: u64) -> Option<bool> { None }

    /// Get Load-Linked address
    fn get_lladdr(&self) -> u64;

    /// Set Load-Linked address
    fn set_lladdr(&self, addr: u64);

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
    lladdr: UnsafeCell<u64>,
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

    /// A fetch is a bus read: no cache state to keep.
    fn fetch_is_pure(&self) -> bool { true }

    fn fetch(&self, _virt_addr: u64, phys_addr: u64) -> FetchInstrResult {
        let r = self.downstream.read32(phys_addr);
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
        if SIZE == 1 { let r = self.downstream.read8(phys_addr);  BusRead64 { status: r.status, data: r.data as u64 } }
        else if SIZE == 2 { let r = self.downstream.read16(phys_addr); BusRead64 { status: r.status, data: r.data as u64 } }
        else if SIZE == 4 { let r = self.downstream.read32(phys_addr); BusRead64 { status: r.status, data: r.data as u64 } }
        else               { self.downstream.read64(phys_addr) }
    }

    fn write<const SIZE: usize>(&self, _virt_addr: u64, phys_addr: u64, val: u64) -> u32 {
        const { assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE") };
        let addr = phys_addr;
        if SIZE == 1      { self.downstream.write8(addr, val as u8) }
        else if SIZE == 2 { self.downstream.write16(addr, val as u16) }
        else if SIZE == 4 { self.downstream.write32(addr, val as u32) }
        else              { self.downstream.write64(addr, val) }
    }

    fn write64_masked(&self, _virt_addr: u64, phys_addr: u64, val: u64, mask: u64) -> u32 {
        // SDL/SDR only — read-modify-write on the downstream device
        let aligned_addr = phys_addr & !7;
        let r = self.downstream.read64(aligned_addr);
        if !r.is_ok() { return r.status; }
        let new_val = (r.data & !mask) | (val & mask);
        self.downstream.write64(aligned_addr, new_val)
    }

    fn cache_op(&self, _cache_op: u32, _virt_addr: u64, _phys_addr: u64) -> u64 {
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

    fn get_lladdr(&self) -> u64 {
        unsafe { *self.lladdr.get() }
    }

    fn set_lladdr(&self, addr: u64) {
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

/// Primary data parity, as far as software can observe it: what Status.CE
/// and the ECC register do to the D-cache's check bits, and the Cache Error
/// exception a mismatch raises (R4000 UM ch. 5, VR5000 UM 7.2.9).
///
/// Check bits are not stored for ordinary data: unless forced, they always
/// match the data, so they are computed when needed. Only doublewords whose
/// check bits were forced are recorded, in `forced`, and their lines carry
/// [`L1DTag::SLOW`] so every access reaches the check. While CE is set every
/// valid line is marked, since any store may force bits.
#[derive(Default)]
struct ParityState {
    /// Status.CE: stores take the ECC register as check bits.
    ce: bool,
    /// Errors raise a Cache Error exception: Status.DE and Status.ERL clear.
    detect: bool,
    /// CP0 ECC, low 8 bits: one even-parity bit per byte, byte 0 the least
    /// significant.
    ecc: u8,
    /// (L1-D extended index, doubleword within the line, its check bits).
    forced: Vec<(usize, u8, u8)>,
    /// Secondary data: per doubleword (physical address within the cache,
    /// `>> 3`), the ECC register value XOR'd into its check bits by a CE
    /// write-back. Empty until first used; `l2_xor_live` counts non-zero
    /// entries so everything else can skip it.
    l2_xor: Vec<u8>,
    l2_xor_live: usize,
    /// CP0 ECC after the last Index_Load_Tag.
    ecc_out: u32,
    /// I-cache lines a CE Fill gave forced parity: (extended index, the bits,
    /// the same for every doubleword of the line).
    ic_forced: Vec<(usize, u8)>,
    /// See `MipsCache::take_op_status`.
    op_status: u32,
    /// The way `l1d_slow_resolve` found, for the access it is serving.
    way: u32,
    /// CacheErr for the error last reported.
    cacheerr: u32,
}

/// `fill_l1d_line` result: the line is resolved (way in `ParityState::way`)
/// and the access must take the parity-checking path. Not a status: real
/// statuses are exceptions (bit 27) or retry (0x100), and a way is 0 or 1.
const L1D_SLOW: u32 = 2;

/// Parity/ECC injection and checking (`cache-ecc`). Off, every hook below
/// folds away: nothing on a miss, an invalidate or an exception looks at
/// `ParityState`. Only diagnostics ever observe it.
const ECC: bool = cfg!(feature = "cache-ecc");

/// `EXEC_IS_CACHE_ERR`, for telling a Cache Error apart from other fill
/// statuses.
const EXEC_IS_CACHE_ERR_BIT: u32 = EXEC_CACHE_ERR & !crate::cpu::mips_exec::EXEC_IS_EXCEPTION;

/// CacheErr: data reference (ER), data field error (ED). R4000 UM Table 5-10;
/// the VR5000 has both at the same positions.
const CACHEERR_ER: u32 = 1 << 31;
const CACHEERR_EC: u32 = 1 << 30;
const CACHEERR_ED: u32 = 1 << 29;

/// Even parity per byte of `v`: bit `i` covers byte `i`, least significant
/// first, as the ECC register lays them out.
fn byte_parity(v: u64) -> u8 {
    (0..8).fold(0, |p, i| p | ((((v >> (8 * i)) as u8).count_ones() & 1) as u8) << i)
}

/// The byte lanes (as in [`byte_parity`]) an access of `size` bytes at `addr`
/// touches. Big-endian: the byte at offset 0 is the most significant.
fn lane_mask(addr: u64, size: usize) -> u8 {
    let off = (addr & 7) as usize;
    (off..off + size).fold(0, |m, k| m | 1 << (7 - k))
}

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
    const MODEL: u8,
    // tcache: whether this cache is transparent (RAM is the store) in a
    // `tcache` build. Off for a cache that must hold data with nothing behind
    // it: the O2 PROM keeps its stack in the data cache while it tests RAM.
    const TC: bool = true,
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
    lladdr: UnsafeCell<u64>,

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
    /// The words each L1-I line held when it was filled, for Hit_Writeback(I).
    /// Direct-mapped models only. Their fetch path serves the L2's decode
    /// slots, which follow L2 data after the fill, but the hardware I-cache
    /// keeps its own copy until refilled. The 2-way model's `ic_instrs` already
    /// is that copy. Empty on a 2-way model.
    ic_raw: UnsafeCell<Box<[u32]>>,
    /// Status.CE / ECC parity forcing and Cache Error reporting. Cold.
    perr: UnsafeCell<ParityState>,
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
    #[cfg(feature = "jitv2")]
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
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8, const TC: bool> Send for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL, TC> {}
unsafe impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8, const TC: bool> Sync for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL, TC> {}

// Per-level cache types, parameterised so each CPU model monomorphises its own.
type ICacheT<const S: usize, const L: usize, const W: usize, const T: usize> =
    Cache<L1ITag, S, L, W, { CacheKind::Insn as u8 }, T, 0, 0>;
type DCacheT<const S: usize, const L: usize, const W: usize, const T: usize, const D: usize> =
    Cache<L1DTag, S, L, W, { CacheKind::Data as u8 }, T, D, 0>;
type L2CacheT<const S: usize, const L: usize, const T: usize, const D: usize, const N: usize> =
    Cache<L2Tag, S, L, 1, { CacheKind::L2 as u8 }, T, D, N>;

/// Which processor a monomorphisation models, as the `MODEL` const parameter.
///
/// This exists because the thing that distinguishes these parts *in this file*
/// is not any one of the shape parameters. It was originally inferred from
/// `IC_WAYS == 2`, which worked only while "2-way" and "R5000" named the same
/// processor. They do not: the R10000 is also 2-way, and it shares neither the
/// R5000's TagLo layout nor its cache-op semantics. Inferring identity from
/// shape would have quietly given an R10000 every R5000 behaviour in the 40-odd
/// places that branch on it, each one individually plausible.
pub mod model {
    pub const R4400: u8 = 0;
    pub const R5000: u8 = 1;
    pub const R10000: u8 = 2;
    pub const R12000: u8 = 3;
    pub const R14000: u8 = 4;
    pub const R16000: u8 = 5;

    /// The R10000 family: R10000 and the parts that follow it, which share one
    /// programming model.
    pub const fn r10k_family(model: u8) -> bool {
        model >= R10000
    }
}

/// SGI Indy R4400: direct-mapped 16K L1s, 1 MB unified L2 owning the decode slots.
pub type R4400Cache = CpuCache<16384, 16, 1, 1024,
                               16384, 16, 1, 1024, 2048,
                               1048576, 128, 8192, 131072, 262144, true,
                               false, 0x0000_0440, 0x0000_0500, 48, { model::R4400 }>;
/// SGI Indy R5000: 2-way 32K L1s, no secondary cache; L1I owns its decode slots.
pub type R5000Cache = CpuCache<32768, 32, 2, 1024,
                               32768, 32, 2, 1024, 4096,
                               128, 128, 1, 16, 0, false,
                               true, 0x0000_2321, 0x0000_2300, 48, { model::R5000 }>;
/// SGI Indigo2 IMPACT R10000 (IP28), modelled for speed rather than fidelity.
///
/// The real part has two-way 32 KB L1s and a two-way secondary cache. This
/// models all three **direct-mapped**, keeping the real total sizes and line
/// sizes (64-byte L1I lines, 32-byte L1D, 128-byte L2).
///
/// That is deliberate. Associativity reaches software only through the
/// way-select bits of `CACHE Index_*`, and the only thing an operating system
/// does by index is flush the whole cache — which comes out the same for any
/// geometry holding the same lines. Emulating ways costs a victim-selection
/// and LRU update on every access and buys nothing IRIX can observe.
///
/// It also buys correctness here, not just speed. A two-way L1 *with* an L2 is
/// a combination this file has never had: `fetch()` selects the two-way tag
/// probe and the L1I-resident decode slots under one condition, and such a
/// part needs the first with the second's alternative. Direct-mapped plus L2
/// is exactly the R4400's shape, so every associativity and decode-slot branch
/// already does the right thing for this model, unchanged.
///
/// What is *not* faked is anything software reads back: the PRId, the TLB
/// size, MIPS IV decoding, and the cache tag layout the PROM's diagnostics
/// inspect directly.
/// Reports revision 2.5; the IP28 PROM skips its boot tune on pre-2.0 CPUs.
pub type R10000Cache = CpuCache<32768, 64, 1, 512,
                                32768, 32, 1, 1024, 4096,
                                1048576, 128, 8192, 131072, 262144, true,
                                true, 0x0000_0925, 0x0000_0900, 64, { model::R10000 }>;

impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8, const TC: bool> CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL, TC> {
    // Model discriminator: folds to a literal, so it replaces #[cfg(feature = "r5k")].
    // Keyed on MODEL, not on associativity — see `mod model`.
    const IS_R5K: bool = MODEL == model::R5000;
    /// Virtual coherency exceptions (VCEI/VCED) are an R4000/R4400 feature. The
    /// R5000 dropped the secondary-index check altogether; the R10000 keeps the
    /// index in its secondary tags but resolves a mismatch in hardware.
    const HAS_VCE: bool = MODEL == model::R4400;

    /// IP28 bring-up watch: physical window to trace, from IRIS_IP28_WATCH.
    /// Folds to a constant `None` on every model but the R10000, so the
    /// non-IP28 hot path keeps no trace of this.
    #[inline(always)]
    fn ip28_watch() -> Option<u64> {
        if !Self::R10K_CACHE_OPS { return None; }
        static W: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
        *W.get_or_init(|| {
            std::env::var("IRIS_IP28_WATCH").ok().and_then(|v| {
                u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()
            })
        })
    }

    #[inline(always)]
    fn ip28_trace(&self, what: &str, addr: u64, val: u64) {
        if let Some(w) = Self::ip28_watch() {
            if (addr & !0x7f) == (w & !0x7f) {
                eprintln!("ip28watch: {what:<18} addr={addr:#018x} val={val:#018x}");
            }
        }
    }

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
            #[cfg(feature = "jitv2")]
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
            ic_raw: UnsafeCell::new(
                if Self::IS_R5K { Vec::new() } else { vec![0u32; IC_SIZE / 4] }.into_boxed_slice()),
            perr: UnsafeCell::new(ParityState::default()),
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
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8, const TC: bool> From<Arc<dyn BusDevice>> for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL, TC> {
    fn from(downstream: Arc<dyn BusDevice>) -> Self {
        Self::new(downstream)
    }
}

impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8, const TC: bool> CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL, TC> {
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
    /// `base` must be ppmem's window base, and must outlive this cache.
    ///
    /// The bitmap is *not* passed here — ppmem publishes into the cache's own
    /// inline field via `MappedMemory::set_bitmap_sink2(tc_bitmap_ptr())`,
    /// which callers must wire up separately (see `machine.rs`).
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
    #[cfg(feature = "jitv2")]
    pub unsafe fn set_tcache_gen_window_impl(&self, gen_base: *mut AtomicU64) {
        unsafe { *self.tc_gen.get() = gen_base };
    }

    /// tcache: is this cache transparent? A constant (the feature and `TC`),
    /// so every `tc_live()` branch folds away.
    #[inline(always)]
    fn tc_live(&self) -> bool {
        cfg!(feature = "tcache") && TC
    }

    /// Write every dirty line back to memory: primary data into the
    /// secondary, then the secondary to memory. Tags stay. For handing a
    /// guest over from a cache that holds data to one that does not.
    pub fn writeback_all_impl(&self) {
        for eidx in 0..DC_WAYS * Self::DC_NUM_LINES {
            if self.dc.get_tag(eidx).is_valid() {
                self.writeback_l1d_line(eidx, false);
            }
        }
        if HAS_L2 {
            for idx in 0..Self::L2_NUM_LINES {
                self.writeback_l2_line(idx);
            }
        }
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
        let hit = crate::ppmem::bitmap_maps(unsafe { *self.tc_bitmap.get() }, phys_addr);
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
            if ACC == 8 {
                self.downstream.read64(phys_addr).data
            } else if ACC == 4 {
                self.downstream.read32(phys_addr).data as u64
            } else if ACC == 2 {
                self.downstream.read16(phys_addr).data as u64
            } else {
                self.downstream.read8(phys_addr).data as u64
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
            if ACC == 8 {
                self.downstream.write64(phys_addr, val);
            } else if ACC == 4 {
                self.downstream.write32(phys_addr, val as u32);
            } else if ACC == 2 {
                self.downstream.write16(phys_addr, val as u16);
            } else {
                self.downstream.write8(phys_addr, val as u8);
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

    /// The line an R10000 index op names in this direct-mapped model: the
    /// index within one way (half the cache), with bit 0 of the address as
    /// the way, which becomes the top index bit.
    #[inline]
    fn r10k_way_index(addr: u64, line_shift: u32, lines_shift: u32) -> usize {
        let half = 1usize << (lines_shift - 1);
        (((addr >> line_shift) as usize) & (half - 1)) | ((addr as usize & 1) << (lines_shift - 1))
    }

    /// The same for an R10000 Index_Load/Store_Data: a doubleword slot in a
    /// data array of `slots` doublewords.
    #[inline]
    fn r10k_way_slot(addr: u64, slots: usize) -> usize {
        let half = slots / 2;
        ((addr as usize >> 3) & (half - 1)) | ((addr as usize & 1) * half)
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
    /// The lladdr stores PA[PA_BITS-1:4].
    #[inline]
    fn check_and_clear_llbit(&self, phys_addr: u64, line_mask: usize) {
        unsafe {
            if !*self.llbit.get() {
                return;
            }
            let ll_addr = *self.lladdr.get() << 4;
            let line_mask = line_mask as u64;
            let addr_line = phys_addr & !line_mask;
            let ll_line = ll_addr & !line_mask;
            if addr_line == ll_line {
            }
        }
    }

    /// Invalidate L1 instruction cache line by index
    fn invalidate_l1i_line(&self, idx: usize, cascade: bool) {
        if ECC {
            let e = self.perr();
            if !e.ic_forced.is_empty() {
                e.ic_forced.retain(|f| f.0 != idx);
            }
        }
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
        self.l1d_unforce(idx);
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
        self.clear_l2_xor((idx << Self::L2_LINE_SHIFT) as u64, L2_LINE);
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

                if dc_tag.matches_line(phys_addr) {
                    self.invalidate_l1d_line(dc_idx, false, false); // hardware cascade, not coherent
                }
            }
        }
        }

        // Finally invalidate the L2 line itself
        self.l2.set_tag(idx, L2Tag::default());
    }

    /// R10000: a secondary hit under a different virtual index than the line
    /// was last used under. Where an R4400 raises a virtual coherency
    /// exception, the R10000 writes back and drops the primary copies made
    /// under the old index, records the new one, and carries on. Without
    /// this, an R10000 kernel that never expects VCE (OpenBSD) dies on the
    /// first page it maps at two colours.
    fn realias_l2_line(&self, l2_idx: usize, new_pidx: u32) {
        let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
        let phys_base = l2_tag_to_phys(l2_tag, (l2_idx << Self::L2_LINE_SHIFT) as u64);
        let old = l2_tag.pidx();

        let l1i_lines_per_l2 = 1 << (Self::L2_LINE_SHIFT - Self::IC_LINE_SHIFT);
        let ic_base_idx = self.l2_idx_to_l1_base_idx(l2_idx, old, &self.ic);
        for i in 0..l1i_lines_per_l2 {
            let ic_idx = (ic_base_idx + i) & Self::IC_NUM_LINES_MASK;
            let phys_addr = phys_base + ((i as u64) << Self::IC_LINE_SHIFT);
            let ic_tag: L1ITag = self.ic.get_tag(ic_idx);
            if ic_tag.matches_phys(phys_addr) {
                self.invalidate_l1i_line(ic_idx, false);
            }
        }

        let l1d_lines_per_l2 = 1 << (Self::L2_LINE_SHIFT - Self::DC_LINE_SHIFT);
        let dc_base_idx = self.l2_idx_to_l1_base_idx(l2_idx, old, &self.dc);
        for i in 0..l1d_lines_per_l2 {
            let dc_idx = (dc_base_idx + i) & Self::DC_NUM_LINES_MASK;
            let phys_addr = phys_base + ((i as u64) << Self::DC_LINE_SHIFT);
            let dc_tag: L1DTag = self.dc.get_tag(dc_idx);
            if dc_tag.matches_line(phys_addr) {
                self.writeback_l1d_line(dc_idx, false);
                self.invalidate_l1d_line(dc_idx, false, false);
            }
        }

        // Re-read: the writebacks above may have changed the line's state.
        let mut t: L2Tag = self.l2.get_tag(l2_idx);
        t.set_pidx(new_pidx);
        self.l2.set_tag(l2_idx, t);
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
        let transparent = self.tc_live();

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
                self.downstream.write_block(line_base, src);
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

        // L2 recomputes the check bits of what it was just written.
        self.clear_l2_xor(phys_addr, Self::DC_LINE);

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
                if dc_tag.matches_line(phys_addr_l1) {
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
        let tc_line = self.tc_live();

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
            if self.downstream.write_block(phys_addr, src) != BUS_OK {
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
        let skip_data = self.tc_live() && !fill_instructions;

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
        } else if let Some(src) = self.downstream.mem_ptr(line_base) {
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
            let s = self.downstream.read_block(line_base, dest);
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
    ///
    /// The miss path: kept out of line and cold so its size never decides
    /// whether `fetch` inlines into the interpreter loop. When the `ic_raw`
    /// snapshot grew this function, `fetch` stopped inlining and every
    /// instruction paid a call — about x0.84 on the whole interpreter suite.
    #[inline(never)]
    #[cold]
    fn fill_l1i_line(&self, index_addr: u64, phys_addr: u64) -> u32 {
        let set = self.ic.get_index(index_addr);

        // 2-way picks the LRU way, unless a way already holds this line: a
        // CACHE Fill then refills that way. IDE's icache test relies on it on
        // the R4600/R5000, placing a line with Index_Store_Tag into the way
        // VA[14] selects and then filling it. A fetch only gets here on a
        // miss, so for it this is always the LRU way. Direct-mapped has one
        // way, so eidx == set.
        let ic_eidx = if Self::IS_R5K {
            (0..IC_WAYS)
                .map(|w| set | w << Self::IC_NUM_LINES_SHIFT)
                .find(|&e| self.ic.get_tag(e).matches_phys(phys_addr))
                .unwrap_or_else(|| set | (unsafe { Self::lru_get(self.ic_lru.get(), set) } as usize) << Self::IC_NUM_LINES_SHIFT)
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
                && (!self.tc_live() || l2_tag.has_code());

            if l2_hit {
                // R4K only: check for Virtual Coherency Exception (VCEI).
                // R5K dropped VCE — no pidx tracking needed; the R10000
                // resolves the mismatch itself.
                if !Self::IS_R5K && self.pidx(index_addr) != l2_tag.pidx() {
                    if Self::HAS_VCE {
                        return exec_exception_const(EXC_VCEI);
                    }
                    self.realias_l2_line(l2_idx, self.pidx(index_addr));
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
            let l2_has_data = !self.tc_live() && self.l2_active();
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
                if let Some(src) = self.downstream.mem_ptr(line_base) {
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
                        let word_addr = line_base + (i as u64) * 4;
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

        // The refill from L2 checks the ECC of the line it transfers.
        if let Some(status) = self.l2_refill_check(index_addr, phys_addr, IC_LINE, false) {
            return status;
        }

        // Direct-mapped: keep the words this fill captured — see `ic_raw`.
        // L2 holds the line by now, and its decode slots carry its words.
        if !Self::IS_R5K && self.l2_active() {
            self.snapshot_l1i_line(ic_eidx, phys_addr);
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
        if !tag.matches_line(phys_addr) {
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
                if Self::HAS_VCE {
                    return Err(BUS_VCE);
                }
                self.realias_l2_line(l2_idx, self.pidx(virt_addr));
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
        if ECC {
            let e = self.perr();
            if e.ce || !e.forced.is_empty() || e.l2_xor_live != 0 {
                return self.l1d_slow_resolve(virt_addr, phys_addr);
            }
        }
        self.fill_l1d_line_raw(virt_addr, phys_addr)
    }

    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    fn perr(&self) -> &mut ParityState {
        unsafe { &mut *self.perr.get() }
    }

    /// A miss while parity state is live: either the line is resident but
    /// marked [`L1DTag::SLOW`], or it is not resident and is filled (and
    /// marked, if CE is set). Returns [`L1D_SLOW`] with the way in
    /// `ParityState::way`, or a fill's exception status.
    #[cold]
    fn l1d_slow_resolve(&self, virt_addr: u64, phys_addr: u64) -> u32 {
        let set = self.dc.get_index(virt_addr);
        let found = (0..DC_WAYS)
            .find(|&w| self.dc.get_tag(set | w << Self::DC_NUM_LINES_SHIFT).matches_line(phys_addr));
        let way = match found {
            Some(w) => w as u32,
            None => {
                if let Some(status) = self.l2_refill_check(virt_addr, phys_addr, Self::DC_LINE, true) {
                    return status;
                }
                let r = self.fill_l1d_line_raw(virt_addr, phys_addr);
                if r > 1 { return r; }
                r
            }
        };
        let eidx = set | (way as usize) << Self::DC_NUM_LINES_SHIFT;
        let mut tag: L1DTag = self.dc.get_tag(eidx);
        let marked = self.perr().ce || self.l1d_forced(eidx);
        tag.ptag = if marked { tag.ptag | L1DTag::SLOW } else { tag.ptag & !L1DTag::SLOW };
        self.dc.set_tag(eidx, tag);
        self.perr().way = way;
        L1D_SLOW
    }

    fn l1d_forced(&self, eidx: usize) -> bool {
        self.perr().forced.iter().any(|f| f.0 == eidx)
    }

    /// Drop forced check bits for line `eidx`: its contents are gone.
    #[inline]
    fn l1d_unforce(&self, eidx: usize) {
        if !ECC { return; }
        let e = self.perr();
        if !e.forced.is_empty() {
            e.forced.retain(|f| f.0 != eidx);
        }
    }

    /// Set or clear the [`L1DTag::SLOW`] mark on every valid L1-D line. A line
    /// with forced check bits stays marked either way.
    fn mark_l1d_lines(&self, on: bool) {
        for eidx in 0..DC_WAYS * Self::DC_NUM_LINES {
            let mut tag: L1DTag = self.dc.get_tag(eidx);
            if !tag.is_valid() { continue; }
            tag.ptag = if on || self.l1d_forced(eidx) { tag.ptag | L1DTag::SLOW } else { tag.ptag & !L1DTag::SLOW };
            self.dc.set_tag(eidx, tag);
        }
    }

    /// The whole doubleword at `virt_addr` in resolved line `eidx`.
    fn l1d_dword(&self, eidx: usize, virt_addr: u64, phys_addr: u64) -> u64 {
        let _ = (eidx, virt_addr);
        let _ = phys_addr;
        tc_or!(self, self.tc_read::<8>(phys_addr & !7),
            self.dc.dc_read::<8>(Self::dc_data_addr(eidx, virt_addr & !7)))
    }

    /// Check a load of the `lanes` of the doubleword at `virt_addr` against its
    /// forced check bits. Returns the Cache Error status if they disagree and
    /// errors are being detected.
    fn l1d_parity_check(&self, eidx: usize, virt_addr: u64, phys_addr: u64, lanes: u8) -> Option<u32> {
        let e = self.perr();
        if !e.detect { return None; }
        let dw = ((virt_addr as usize & Self::DC_LINE_MASK) >> 3) as u8;
        let forced = e.forced.iter().find(|f| f.0 == eidx && f.1 == dw)?.2;
        if (byte_parity(self.l1d_dword(eidx, virt_addr, phys_addr)) ^ forced) & lanes == 0 {
            return None;
        }
        // SIdx: pAddr[21:3]; PIdx: vAddr[14:12].
        self.perr().cacheerr = CACHEERR_ER | CACHEERR_ED
            | (phys_addr as u32 & 0x003F_FFF8) | ((virt_addr >> 12) as u32 & 7);
        Some(EXEC_CACHE_ERR)
    }

    /// After a store to the `lanes` of the doubleword at `virt_addr` (`before`
    /// is the doubleword it replaced): with CE set those lanes take the ECC
    /// register's bits, otherwise freshly computed ones. A doubleword whose
    /// bits agree with its data again is forgotten.
    fn l1d_parity_store(&self, eidx: usize, virt_addr: u64, phys_addr: u64, before: u64, lanes: u8) {
        let after = byte_parity(self.l1d_dword(eidx, virt_addr, phys_addr));
        let dw = ((virt_addr as usize & Self::DC_LINE_MASK) >> 3) as u8;
        let e = self.perr();
        let pos = e.forced.iter().position(|f| f.0 == eidx && f.1 == dw);
        let prev = pos.map_or(byte_parity(before), |i| e.forced[i].2);
        let src = if e.ce { e.ecc } else { after };
        let bits = (prev & !lanes) | (src & lanes);
        match (pos, bits == after) {
            (Some(i), true) => { e.forced.swap_remove(i); }
            (Some(i), false) => e.forced[i].2 = bits,
            (None, false) => e.forced.push((eidx, dw, bits)),
            (None, true) => {}
        }
        if !e.ce && !self.l1d_forced(eidx) {
            let mut tag: L1DTag = self.dc.get_tag(eidx);
            tag.ptag &= !L1DTag::SLOW;
            self.dc.set_tag(eidx, tag);
        }
    }

    #[cold]
    fn read_slow<const SIZE: usize>(&self, virt_addr: u64, phys_addr: u64) -> BusRead64 {
        let eidx = self.dc.get_index(virt_addr) | (self.perr().way as usize) << Self::DC_NUM_LINES_SHIFT;
        if let Some(status) = self.l1d_parity_check(eidx, virt_addr, phys_addr, lane_mask(virt_addr, SIZE)) {
            return BusRead64 { status, data: 0 };
        }
        let result = tc_or!(self, self.tc_read::<SIZE>(phys_addr),
            self.dc.dc_read::<SIZE>(Self::dc_data_addr(eidx, virt_addr)));
        BusRead64::ok(result)
    }

    #[cold]
    fn write_slow(&self, virt_addr: u64, phys_addr: u64, val: u64, mask: u64) -> u32 {
        let eidx = self.dc.get_index(virt_addr) | (self.perr().way as usize) << Self::DC_NUM_LINES_SHIFT;
        let before = self.l1d_dword(eidx, virt_addr, phys_addr);
        let merged = (before & !mask) | (val & mask);
        tc_or!(self, {
            self.tc_write::<8>(phys_addr & !7, merged);
            self.tc_invalidate_l2_code(phys_addr);
        }, self.dc.dc_write::<8>(Self::dc_data_addr(eidx, virt_addr & !7), merged));
        self.mark_l1d_dirty(eidx);
        let lanes = (0..8).fold(0u8, |l, i| if (mask >> (8 * i)) & 0xff != 0 { l | 1 << i } else { l });
        self.l1d_parity_store(eidx, virt_addr, phys_addr, before, lanes);
        BUS_OK
    }

    /// The doubleword mask for a `SIZE`-byte store at `virt_addr`, and the
    /// value shifted into place.
    fn store_mask<const SIZE: usize>(virt_addr: u64, val: u64) -> (u64, u64) {
        if SIZE == 8 { return (u64::MAX, val); }
        let shift = (8 - SIZE - (virt_addr & 7) as usize) * 8;
        let bits = (1u64 << (SIZE * 8)) - 1;
        (bits << shift, (val & bits) << shift)
    }

    /// The check bits Index_Load_Tag loads into CP0 ECC: those of the
    /// doubleword vAddr[LINEBITS:3] selects in the indexed line (R4000 UM
    /// A-42). Secondary: data ECC, with any CE-forced bits XOR'd in. Primary
    /// data: byte parity, or the forced bits. Primary instruction: byte parity
    /// on the R4400, one bit per word on the R5000 (VR5000 UM Table 7-8).
    fn ilt_ecc(&self, is_icache: bool, is_l2: bool, idx: usize, virt_addr: u64) -> u32 {
        if is_l2 {
            let tag: L2Tag = self.l2.get_tag(idx);
            let off = virt_addr as usize & Self::L2_LINE_MASK & !7;
            let phys = l2_tag_to_phys(tag, ((idx << Self::L2_LINE_SHIFT) | off) as u64);
            let data = tc_or!(self, self.downstream.read64(phys).data,
                self.l2.data()[(idx << Self::L2_CHUNKS_PER_LINE_SHIFT) + (off >> 3)]);
            return (l2_data_ecc(data) ^ self.l2_xor(phys)) as u32;
        }
        if is_icache {
            if let Some(&(_, forced)) = self.perr().ic_forced.iter().find(|f| f.0 == idx) {
                return forced as u32;
            }
            let k = (virt_addr as usize & Self::IC_LINE_MASK & !7) >> 2;
            let (hi, lo) = (self.l1i_word(idx, k), self.l1i_word(idx, k + 1));
            return if Self::IS_R5K {
                (lo.count_ones() & 1) | (hi.count_ones() & 1) << 1
            } else {
                byte_parity((hi as u64) << 32 | lo as u64) as u32
            };
        }
        let tag: L1DTag = self.dc.get_tag(idx);
        let phys = l1d_tag_to_phys(tag, ((idx & Self::DC_NUM_LINES_MASK) << Self::DC_LINE_SHIFT) as u64)
            | (virt_addr & Self::DC_LINE_MASK as u64 & !7);
        let dw = ((virt_addr as usize & Self::DC_LINE_MASK) >> 3) as u8;
        let forced = self.perr().forced.iter().find(|f| f.0 == idx && f.1 == dw).map(|f| f.2);
        forced.unwrap_or_else(|| byte_parity(self.l1d_dword(idx, virt_addr, phys))) as u32
    }

    /// Slot of the secondary doubleword holding `phys_addr`.
    fn l2_slot(phys_addr: u64) -> usize {
        (phys_addr as usize & (L2_CACHE_SIZE - 1)) >> 3
    }

    /// The bits XOR'd into the check bits of the doubleword at `phys_addr`.
    fn l2_xor(&self, phys_addr: u64) -> u8 {
        let e = self.perr();
        if e.l2_xor_live == 0 { return 0; }
        e.l2_xor[Self::l2_slot(phys_addr)]
    }

    fn set_l2_xor(&self, phys_addr: u64, v: u8) {
        let e = self.perr();
        if e.l2_xor.is_empty() {
            if v == 0 { return; }
            e.l2_xor = vec![0; L2_CACHE_SIZE / 8];
        }
        let slot = &mut e.l2_xor[Self::l2_slot(phys_addr)];
        e.l2_xor_live = e.l2_xor_live + (v != 0) as usize - (*slot != 0) as usize;
        *slot = v;
    }

    /// Recomputed check bits for `len` bytes of secondary data at `phys_addr`:
    /// a write into L2, a fill or an invalidate.
    #[inline]
    fn clear_l2_xor(&self, phys_addr: u64, len: usize) {
        if !ECC || self.perr().l2_xor_live == 0 { return; }
        let base = phys_addr & !7;
        for i in 0..len / 8 {
            self.set_l2_xor(base + 8 * i as u64, 0);
        }
    }

    /// A primary refill from L2 checks the ECC of the `len`-byte line it
    /// transfers. Returns the Cache Error if any doubleword carries forced bits
    /// and errors are being detected.
    fn l2_refill_check(&self, virt_addr: u64, phys_addr: u64, len: usize, data: bool) -> Option<u32> {
        if !ECC { return None; }
        let e = self.perr();
        if e.l2_xor_live == 0 || !e.detect || !HAS_L2 || self.hit_l2(phys_addr).is_none() {
            return None;
        }
        let base = phys_addr & !(len as u64 - 1);
        (0..len / 8).find(|&i| self.l2_xor(base + 8 * i as u64) != 0)?;
        self.perr().cacheerr = if data { CACHEERR_ER } else { 0 } | CACHEERR_EC | CACHEERR_ED
            | (phys_addr as u32 & 0x003F_FFF8) | ((virt_addr >> 12) as u32 & 7);
        Some(EXEC_CACHE_ERR)
    }

    /// A D-cache write-back done by a CACHE op. With CE set, the ECC register
    /// is XOR'd into the check bits of the doubleword the op addresses as the
    /// line is written to L2 (R4000 UM A-45).
    fn l1d_writeback_op(&self, eidx: usize, virt_addr: u64, cascade: bool) {
        if !ECC {
            self.writeback_l1d_line(eidx, cascade);
            return;
        }
        let tag: L1DTag = self.dc.get_tag(eidx);
        let phys = l1d_tag_to_phys(tag, ((eidx & Self::DC_NUM_LINES_MASK) << Self::DC_LINE_SHIFT) as u64)
            | (virt_addr & Self::DC_LINE_MASK as u64 & !7);
        let written = tag.is_valid() && tag.dirty;
        self.writeback_l1d_line(eidx, cascade);
        let e = self.perr();
        if written && e.ce && HAS_L2 && self.hit_l2(phys).is_some() {
            let ecc = e.ecc;
            self.set_l2_xor(phys, ecc);
        }
    }

    fn fill_l1d_line_raw(&self, virt_addr: u64, phys_addr: u64) -> u32 {
        let transparent = self.tc_live();
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
        let l2_hit = tc_or!(self, if self.l2_active() {
            match self.l2_probe_for_data(virt_addr, phys_addr) {
                Ok(hit) => hit,
                Err(status) => return status,
            }
        } else {
            false
        }, if self.l2_active() {
            let l2_idx = self.l2.get_index(phys_addr);
            let l2_tag: L2Tag = self.l2.get_tag(l2_idx);
            let l2_ptag = self.l2_ptag(phys_addr);
            if l2_tag.cs() != L2_CS_INVALID && l2_tag.ptag() == l2_ptag {
                // Check for Virtual Coherency Exception (R4K only; R5K dropped
                // VCE; the R10000 resolves the mismatch itself)
                if !Self::IS_R5K && self.pidx(virt_addr) != l2_tag.pidx() {
                    if Self::HAS_VCE { return BUS_VCE; }
                    self.realias_l2_line(l2_idx, self.pidx(virt_addr));
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
        });
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
                self.downstream.read_block(line_base, dest);
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

    /// Direct-mapped: copy L1-I line `eidx`'s words into `ic_raw` from the L2
    /// decode slots for `phys_addr`, which the caller guarantees hold the line.
    fn snapshot_l1i_line(&self, eidx: usize, phys_addr: u64) {
        let n = Self::IC_INSTRS_PER_LINE;
        let l2_idx = self.l2.get_index(phys_addr);
        let word = ((phys_addr as usize) & Self::L2_LINE_MASK & !Self::IC_LINE_MASK) >> 2;
        let src = &self.l2.instrs.get()[(l2_idx << Self::L2_INSTR_SHIFT) + word..][..n];
        let dst = &mut unsafe { &mut *self.ic_raw.get() }[eidx * n..][..n];
        for (d, s) in dst.iter_mut().zip(src) {
            *d = s.raw;
        }
    }

    /// Word `i` of L1-I line `eidx`, as the I-cache holds it.
    fn l1i_word(&self, eidx: usize, i: usize) -> u32 {
        if Self::IS_R5K {
            self.ic_instrs.get()[(eidx << Self::IC_INSTR_SHIFT) + i].raw
        } else {
            unsafe { (&*self.ic_raw.get())[eidx * Self::IC_INSTRS_PER_LINE + i] }
        }
    }

    /// Hit_Writeback(I): write L1-I line `eidx` back to L2 if L2 holds the
    /// line, otherwise to memory.
    ///
    /// Both manuals define it (op 6 on the I-cache: "data is written back
    /// unconditionally"). IDE's `icache2` depends on it: it fills the I-cache,
    /// changes memory under it through the D-cache, and expects the write-back
    /// to put the I-cache's words back.
    fn writeback_l1i_line(&self, eidx: usize, virt_addr: u64, line_base: u64) {
        let n = Self::IC_INSTRS_PER_LINE;
        // A line a CE Fill gave forced parity fails its check here, which is
        // how IDE `icache1` observes it (R4000 UM Table 16-1: primary cache
        // to secondary, parity checked, trap on error).
        let e = self.perr();
        if ECC && e.detect {
            if let Some(&(_, forced)) = e.ic_forced.iter().find(|f| f.0 == eidx) {
                let bad = (0..n / 2).any(|c| {
                    let dw = (self.l1i_word(eidx, 2 * c) as u64) << 32 | self.l1i_word(eidx, 2 * c + 1) as u64;
                    byte_parity(dw) != forced
                });
                if bad {
                    e.cacheerr = CACHEERR_ED | (line_base as u32 & 0x003F_FFF8) | ((virt_addr >> 12) as u32 & 7);
                    e.op_status = EXEC_CACHE_ERR;
                    return;
                }
            }
        }
        let mut chunks = [0u64; 16];
        let chunks = &mut chunks[..n / 2];
        for (c, chunk) in chunks.iter_mut().enumerate() {
            *chunk = (self.l1i_word(eidx, 2 * c) as u64) << 32 | self.l1i_word(eidx, 2 * c + 1) as u64;
        }

        let Some(l2_idx) = self.hit_l2(line_base) else {
            // write_block bumps the jitv2 page generations it spans.
            self.downstream.write_block(line_base, chunks);
            return;
        };
        // tcache: RAM is the store, so the words go there; dropping
        // `has_code` makes the next L1-I fill re-read them.
        tc_or!(self, {
            self.downstream.write_block(line_base, chunks);
            self.tc_invalidate_l2_code(line_base);
        }, {
            let off = ((line_base as usize) & Self::L2_LINE_MASK) >> 3;
            let base = (l2_idx << Self::L2_CHUNKS_PER_LINE_SHIFT) + off;
            self.l2.data_mut()[base..base + chunks.len()].copy_from_slice(chunks);
            // Keep the decode slots in step with L2 data, as
            // `writeback_l1d_line` does. A whole I-cache line is rewritten and
            // fusion never crosses one, so clearing the flags drops every
            // fused pair this line took part in.
            if !Self::IS_R5K {
                let instrs = self.l2.instrs.get_mut();
                let ibase = (l2_idx << Self::L2_INSTR_SHIFT) + off * 2;
                for i in 0..n {
                    let slot = &mut instrs[ibase + i];
                    slot.raw = self.l1i_word(eidx, i);
                    slot.flags = FLAG_NOT_DECODED;
                }
            }
        });
        self.clear_l2_xor(line_base, n * 4);
        let mut tag: L2Tag = self.l2.get_tag(l2_idx);
        let cs = match tag.cs() {
            L2_CS_CLEAN_EXCLUSIVE => L2_CS_DIRTY_EXCLUSIVE,
            L2_CS_SHARED => L2_CS_DIRTY_SHARED,
            cs => cs,
        };
        tag.set_cs(cs);
        self.l2.set_tag(l2_idx, tag);
    }

    /// After a snapshot restore: refill the words of every valid L1-I line,
    /// which snapshots do not carry. Without this a restored valid tag served
    /// whatever the empty slots held, zeros (NOPs) on a fresh cache, until the
    /// line was evicted.
    fn rebuild_l1i_words(&self) {
        let n = Self::IC_INSTRS_PER_LINE;
        for eidx in 0..IC_WAYS * Self::IC_NUM_LINES {
            let tag: L1ITag = self.ic.get_tag(eidx);
            if !tag.is_valid() {
                continue;
            }
            let set = eidx & Self::IC_NUM_LINES_MASK;
            let line_base = tag.line_addr() | (((set << Self::IC_LINE_SHIFT) as u64) & L1_INDEX_MASK);
            // The L2 decode slots hold the line if L2 does, except under
            // tcache, where only an instruction-origin fill (`has_code`)
            // populates them.
            let l2_has_words = self.hit_l2(line_base).is_some_and(|l2_idx| {
                !self.tc_live() || self.l2.get_tag(l2_idx).has_code()
            });
            if !Self::IS_R5K && l2_has_words {
                self.snapshot_l1i_line(eidx, line_base);
                continue;
            }
            for i in 0..n {
                let r = self.downstream.read32(line_base + 4 * i as u64);
                let w = if r.is_ok() { r.data } else { 0 };
                if Self::IS_R5K {
                    let slot = &mut self.ic_instrs.get_mut()[(eidx << Self::IC_INSTR_SHIFT) + i];
                    slot.raw = w;
                    slot.flags = FLAG_NOT_DECODED;
                } else {
                    unsafe { (&mut *self.ic_raw.get())[eidx * n + i] = w };
                }
            }
        }
    }

    /// Return the eidx of the L1-D way that holds `phys_addr`, or `None`.
    #[inline]
    fn hit_l1d(&self, virt_addr: u64, phys_addr: u64) -> Option<usize> {
        let set = self.dc.get_index(virt_addr);
        for way in 0..DC_WAYS {
            let eidx = set | (way << Self::DC_NUM_LINES_SHIFT);
            let tag: L1DTag = self.dc.get_tag(eidx);
            if tag.matches_line(phys_addr) { return Some(eidx); }
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
                if tag.dirty && tag.matches_line(phys_line) {
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
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8, const TC: bool> CpuModel for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL, TC> {
    const MIPS4: bool = MIPS4;
    const PRID: u32 = PRID;
    const FIR: u32 = FIR;
    const TLB_ENTRIES: usize = TLB_ENTRIES;
    const R10K_CACHE_OPS: bool = model::r10k_family(MODEL);
    // 44 implemented virtual address bits on the R10000 family, as the shadow
    // model already says; XContext's layout follows them.
    const VA_BITS: u32 = if model::r10k_family(MODEL) { 44 } else { 40 };
    const PA_BITS: u32 = if model::r10k_family(MODEL) { 40 } else { 36 };
    const NAME: &'static str = match MODEL {
        model::R5000 => "R5000",
        model::R10000 => "R10000",
        model::R12000 => "R12000",
        model::R14000 => "R14000",
        model::R16000 => "R16000",
        _ => "R4400",
    };
}

impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8, const TC: bool> MipsCache for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL, TC> {
    fn set_l1i_counters(&mut self, hit: Arc<AtomicU64>, fetch: Arc<AtomicU64>) {
        self.l1i_hit_count = hit;
        self.l1i_fetch_count = fetch;
    }

    fn writeback_all(&self) {
        self.writeback_all_impl()
    }

    #[cfg(feature = "tcache")]
    unsafe fn set_tcache_window(&self, base: *mut u8) {
        unsafe { self.set_tcache_window_impl(base) }
    }

    #[cfg(feature = "tcache")]
    fn tcache_bitmap_ptr(&self) -> *mut u64 { self.tc_bitmap_ptr() }

    #[cfg(feature = "tcache")]
    fn tcache_base_ptr(&self) -> *mut u8 { unsafe { *self.tc_base.get() } }

    #[cfg(feature = "jitv2")]
    fn tcache_gen_ptr(&self) -> *mut u8 { unsafe { *self.tc_gen.get() as *mut u8 } }

    fn jit_l2_tags_ptr(&self) -> *mut u8 {
        if !HAS_L2 {
            return std::ptr::null_mut();
        }
        unsafe { (*self.l2.tags.get()).as_mut_ptr() as *mut u8 }
    }

    fn jit_dc_tags_ptr(&self) -> *mut u8 {
        unsafe { (*self.dc.tags.get()).as_mut_ptr() as *mut u8 }
    }

    fn jit_dc_geometry(&self) -> JitDcGeometry {
        // Both associativities are supported. R5000's 2-way L1-D needs the
        // emitted guard to probe *both* ways and, on a hit, fold the winning
        // way into the extended tag index — see `emit_inline_mem_guard`, which
        // mirrors `ensure_l1d_line` (including its hit-side LRU update).
        //
        // Without tcache the way also folds into the data address
        // (`dc_data_addr`); with tcache the data comes from `tc_base + phys`,
        // so the way affects only the tag probe and the LRU bit.
        // Under tcache the inline path reads the ppmem window and bumps the
        // jitv2 generation array directly, so both pointers must be live
        // before any region compiles. They are published by machine.rs before
        // the CPU thread starts; declining here (rather than null-checking in
        // every emitted access) is what lets the emitted code assume them.
        #[cfg(feature = "jitv2")]
        {
            // The inline path is the transparent one; a cache that holds data
            // (`TC = false`) must be reached through its own accesses.
            if !self.tc_live() {
                return JitDcGeometry::unsupported();
            }
            if unsafe { *self.tc_base.get() }.is_null()
                || unsafe { *self.tc_gen.get() }.is_null()
            {
                return JitDcGeometry::unsupported();
            }
            // The inline store clears L2's has_code bit directly, so the tag
            // base must be live too. (It is a boxed slice allocated in
            // `new`, so this only guards against a future refactor.)
            if HAS_L2 && self.jit_l2_tags_ptr().is_null() {
                return JitDcGeometry::unsupported();
            }
        }
        JitDcGeometry {
            supported: true,
            line_shift: Self::DC_LINE_SHIFT,
            // **Sets, not total lines.** `DC_NUM_LINES_MASK` is
            // `DC_NUM_SETS - 1` = `DC_SIZE / DC_LINE / DC_WAYS - 1`, which is
            // what `dc.get_index()` masks with. Writing `DC_SIZE / DC_LINE - 1`
            // here is correct only for a direct-mapped cache; on R5000's 2-way
            // it is one bit too wide, and that extra bit **is** the way bit —
            // so a way-0 index could alias into way 1's range and the
            // `set | way << num_lines_shift` fold became meaningless.
            num_lines_mask: Self::DC_NUM_LINES_MASK as u64,
            data_mask: (DC_SIZE - 1) as u64,
            has_l2: HAS_L2,
            l2_line_shift: L2_LINE.trailing_zeros(),
            l2_num_lines_mask: if HAS_L2 { (L2_CACHE_SIZE / L2_LINE - 1) as u64 } else { 0 },
            ways: DC_WAYS as u32,
            num_lines_shift: Self::DC_NUM_LINES_SHIFT as u32,
            atomic_gen: false,
            tagless: false,
        }
    }

    fn jit_dc_lru_ptr(&self) -> *mut u8 {
        // Empty (and so no meaningful base) on a direct-mapped model — the
        // 1-way guard never reads it.
        if !Self::IS_R5K { return std::ptr::null_mut(); }
        unsafe { (*self.dc_lru.get()).as_mut_ptr() as *mut u8 }
    }

    #[cfg(feature = "jitv2")]
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

    /// The interpreter's per-instruction hit path: must inline into
    /// `step_int`. See `fill_l1i_line` for what happened when it did not.
    #[inline(always)]
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

    #[cfg(feature = "jitstats")]
    fn jit_probe<const STORE: bool>(&self, virt_addr: u64, phys_addr: u64) -> bool {
        // Mirrors the R4K hit path in `read`/`write` below. R5000 (2-way) is
        // reported as never-inlinable: the data address needs way selection
        // (`dc_data_addr` folds the way into the index), so the first emit
        // targets 1-way only.
        if Self::IS_R5K {
            return false;
        }
        let dc_idx = self.dc.get_index(virt_addr);
        let tag = self.dc.get_tag(dc_idx);
        if !tag.matches_phys(phys_addr) {
            return false;
        }
        // Stores only proceed inline on an already-dirty line — see the trait
        // doc. A clean line's dirty transition is a tag RMW, left to Rust.
        if STORE && !tag.dirty {
            return false;
        }
        // tcache: data comes from the ppmem window, so the region must be
        // mapped. Without tcache the resident line itself is authoritative
        // and no bitmap test is needed.
        tc_or!(self, self.tc_transparent(phys_addr), true)
    }

    fn read<const SIZE: usize>(&self, virt_addr: u64, phys_addr: u64) -> BusRead64 {
        self.ip28_trace("read", phys_addr, 0);
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
                if r > 1 {
                    if r == L1D_SLOW { return self.read_slow::<SIZE>(virt_addr, phys_addr); }
                    return BusRead64 { status: r, data: 0 };
                }
            }
            #[cfg(not(feature = "lightning"))]
            if devlog_is_active(LogModule::L1d) && devlog_mask(LogModule::L1d) & CACHE_LOG_HIT != 0 {
                crate::dlog!(LogModule::L1d, "read{} hit virt={:#x} phys={:#x} eidx={}", SIZE, virt_addr, phys_addr, dc_idx);
            }
            // tcache: the cache stores no data — read RAM (window or bus).
            let result = tc_or!(self, self.tc_read::<SIZE>(phys_addr), self.dc.dc_read::<SIZE>(virt_addr));
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
            if way > 1 {
                if way == L1D_SLOW { return self.read_slow::<SIZE>(virt_addr, phys_addr); }
                return BusRead64 { status: way, data: 0 };
            }
            let dc_eidx = self.dc.get_index(virt_addr) | (way as usize) << Self::DC_NUM_LINES_SHIFT;
            #[cfg(not(feature = "lightning"))]
            if devlog_is_active(LogModule::L1d) && devlog_mask(LogModule::L1d) & CACHE_LOG_HIT != 0 {
                crate::dlog!(LogModule::L1d, "read{} hit virt={:#x} phys={:#x} eidx={}", SIZE, virt_addr, phys_addr, dc_eidx);
            }
            // tcache: the cache stores no data — read RAM (window or bus).
            let result = tc_or!(self, self.tc_read::<SIZE>(phys_addr), {
                let da = Self::dc_data_addr(dc_eidx, virt_addr);
                self.dc.dc_read::<SIZE>(da)
            });
            #[cfg(feature = "debug_cache")]
            if self.is_tracking_addr(virt_addr, phys_addr) {
                println!("[CACHE DEBUG] read{} result: {} virt 0x{:016x} phys 0x{:016x} val=0x{:016x}",
                         SIZE * 8, self.tracking_label(phys_addr), virt_addr, phys_addr, result);
            }
            BusRead64::ok(result)
        }
    }

    fn write<const SIZE: usize>(&self, virt_addr: u64, phys_addr: u64, val: u64) -> u32 {
        self.ip28_trace("write", phys_addr, val);
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
                if r > 1 {
                if r == L1D_SLOW {
                    let (mask, v) = Self::store_mask::<SIZE>(virt_addr, val);
                    return self.write_slow(virt_addr, phys_addr, v, mask);
                }
                    return r;
                }
            }
            #[cfg(not(feature = "lightning"))]
            if devlog_is_active(LogModule::L1d) && devlog_mask(LogModule::L1d) & CACHE_LOG_HIT != 0 {
                crate::dlog!(LogModule::L1d, "write{} hit virt={:#x} phys={:#x} eidx={} val={:#x}", SIZE, virt_addr, phys_addr, dc_idx, val);
            }
            tc_or!(self, {
                // The cache stores no data — the write goes to RAM.
                self.tc_write::<SIZE>(phys_addr, val);
                // Writing the line makes L2's decoded slots for it stale.
                self.tc_invalidate_l2_code(phys_addr);
            }, self.dc.dc_write::<SIZE>(virt_addr, val));
            self.mark_l1d_dirty(dc_idx);
            return BUS_OK;
        }
        }
        // R5K (2-way): generic path.
        //
        // tcache **is** active here, contrary to what this comment used to
        // claim: `fill_l1d_line` sets `transparent = true` purely on
        // `cfg(tcache)` with no `IS_R5K` guard, so R5000 lines hold no data and
        // the store below goes to RAM via `tc_write` like any other model.
        //
        // `tc_invalidate_l2_code` is called for symmetry with the R4K branch and
        // is a no-op when `!HAS_L2` (R5000 has no L2). That is correct, not a
        // gap: `has_code` guards tcache's *decoded-instruction* slots in L2, and
        // with no L2 there are none to invalidate. L1-I decode slots (`ic_instrs`)
        // are populated only on an L1-I fill, so they cannot outlive the
        // guest-issued CACHE op that retires the line — which the architecture
        // already requires for self-modifying code.
        {
            let way = self.ensure_l1d_line(virt_addr, phys_addr);
            if way > 1 {
                if way == L1D_SLOW {
                    let (mask, v) = Self::store_mask::<SIZE>(virt_addr, val);
                    return self.write_slow(virt_addr, phys_addr, v, mask);
                }
                return way;
            }
            let dc_eidx = self.dc.get_index(virt_addr) | (way as usize) << Self::DC_NUM_LINES_SHIFT;
            #[cfg(not(feature = "lightning"))]
            if devlog_is_active(LogModule::L1d) && devlog_mask(LogModule::L1d) & CACHE_LOG_HIT != 0 {
                crate::dlog!(LogModule::L1d, "write{} hit virt={:#x} phys={:#x} eidx={} val={:#x}", SIZE, virt_addr, phys_addr, dc_eidx, val);
            }
            tc_or!(self, {
                self.tc_write::<SIZE>(phys_addr, val);
                self.tc_invalidate_l2_code(phys_addr);
            }, {
                let da = Self::dc_data_addr(dc_eidx, virt_addr);
                self.dc.dc_write::<SIZE>(da, val);
            });
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
                if r > 1 {
                    if r == L1D_SLOW { return self.write_slow(virt_addr, phys_addr, val, mask); }
                    return r;
                }
            }
            tc_or!(self, {
                let current = self.tc_read::<8>(phys_addr);
                self.tc_write::<8>(phys_addr, (current & !mask) | (val & mask));
                self.tc_invalidate_l2_code(phys_addr);
            }, {
                let current = self.dc.dc_read::<8>(virt_addr);
                self.dc.dc_write::<8>(virt_addr, (current & !mask) | (val & mask));
            });
            self.mark_l1d_dirty(dc_idx);
            return BUS_OK;
        }
        }
        // R5K (2-way): generic path.
        {
            let way = self.ensure_l1d_line(virt_addr, phys_addr);
            if way > 1 {
                if way == L1D_SLOW { return self.write_slow(virt_addr, phys_addr, val, mask); }
                return way;
            }
            let dc_eidx = self.dc.get_index(virt_addr) | (way as usize) << Self::DC_NUM_LINES_SHIFT;
            tc_or!(self, {
                let current = self.tc_read::<8>(phys_addr);
                self.tc_write::<8>(phys_addr, (current & !mask) | (val & mask));
                self.tc_invalidate_l2_code(phys_addr);
            }, {
                let da = Self::dc_data_addr(dc_eidx, virt_addr);
                let current = self.dc.dc_read::<8>(da);
                self.dc.dc_write::<8>(da, (current & !mask) | (val & mask));
            });
            self.mark_l1d_dirty(dc_eidx);
            BUS_OK
        }
    }

    fn cache_op(&self, cache_op: u32, virt_addr: u64, phys_addr: u64) -> u64 {
        if Self::R10K_CACHE_OPS && std::env::var_os("IRIS_IP28_CACHEOPS").is_some() {
            use std::sync::atomic::{AtomicU32, Ordering};
            static SEEN: AtomicU32 = AtomicU32::new(0);
            let bit = 1u32 << (cache_op & 0x1F);
            let all = matches!(std::env::var("IRIS_IP28_CACHEOPS").as_deref(), Ok("all"));
            if all || SEEN.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
                eprintln!("ip28: op {} raw={:#04x} va={:#018x} arg={:#018x}",
                          cache_op_name(cache_op), cache_op, virt_addr, phys_addr);
            }
        }
        if Self::ip28_watch().is_some() {
            self.ip28_trace(cache_op_name(cache_op), virt_addr, phys_addr);
        }
        // R10000 ops 5/6/7 are not the R4000 hit operations that share these
        // encodings — see C_R10K_ISD. Handled before the shared decode below.
        if Self::R10K_CACHE_OPS {
            // The R10000 meanings are scoped to particular cache selects;
            // outside those the R4000 operation on the same encoding still
            // applies. cache_r10k.h annotates each one, and the IP28 PROM uses
            // both readings of op 5: `Cache_Barrier` against the instruction
            // cache, and R4000 `Hit_Invalidate` against the secondary.
            let sel = cache_op & 3;
            match cache_op & 0x1C {
                // An ordering barrier. Nothing to do in a model with no
                // speculative memory pipeline to hold back.
                C_R10K_CBARRIER if sel == CACH_PI => return 0,
                C_R10K_ILD if matches!(sel, CACH_PI | CACH_PD | CACH_SD) => {
                    let is_l2 = sel == CACH_SD;
                    // The index is a byte offset into one way's data, and bit
                    // 0 selects the way: the same folding as the tag ops
                    // below, so data and tags name the same line.
                    if is_l2 && HAS_L2 {
                        return self.l2.data()[Self::r10k_way_slot(virt_addr, L2_DATA)];
                    }
                    return self.dc.data()[Self::r10k_way_slot(virt_addr, DC_DATA)];
                }
                C_R10K_ISD if matches!(sel, CACH_SI | CACH_SD) => {
                    if HAS_L2 {
                        self.l2.data_mut()[Self::r10k_way_slot(virt_addr, L2_DATA)] = phys_addr;
                    }
                    return 0;
                }
                _ => {}
            }
        }
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
        //
        // L2 index ops take the line from the CACHE instruction's address. The
        // executor hands that over untranslated in `virt_addr` (and in
        // `phys_addr` too, except for C_IST, where `phys_addr` carries
        // TagHi:TagLo). Hit ops and Create_Dirty_Exclusive need the
        // translated `phys_addr`.
        // R10000 index ops (Index_WB_Inv, Index_Load_Tag, Index_Store_Tag)
        // address one way: the low bits index within it and bit 0 picks the
        // way. This model is direct-mapped at the full size, so way 1 is the
        // upper half. Ignoring bit 0 sent both ways of a whole-cache flush to
        // the lower half and never flushed the upper one: dirty lines there
        // outlived the flush and were later written back over reused pages.
        let r10k_index = Self::R10K_CACHE_OPS && matches!(operation, C_IWBINV | C_ILT | C_IST);
        let idx = if is_l2 {
            if r10k_index {
                // Index ops carry the index address in virt_addr (untranslated).
                Self::r10k_way_index(virt_addr, Self::L2_LINE_SHIFT, Self::L2_NUM_LINES_SHIFT)
            } else {
                self.l2.get_index(if operation == C_IST { virt_addr } else { phys_addr })
            }
        } else if r10k_index && is_icache {
            Self::r10k_way_index(virt_addr, Self::IC_LINE_SHIFT, Self::IC_NUM_LINES_SHIFT)
        } else if r10k_index {
            Self::r10k_way_index(virt_addr, Self::DC_LINE_SHIFT, Self::DC_NUM_LINES_SHIFT)
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

        if ECC && operation == C_ILT {
            self.perr().ecc_out = self.ilt_ecc(is_icache, is_l2, idx, virt_addr);
        }

        match operation {
            // Index Invalidate (I, SI) or Index Writeback Invalidate (D, SD).
            // R5K/Triton SI/SD: reinterpreted as C_INVALL — invalidate the entire L2
            // regardless of address. PROM issues this after enabling L2 via CONFIG_SE.
            C_IINV => { // same as C_IWBINV
                if is_icache {
                    self.invalidate_l1i_line(idx, cascade);
                } else if !is_l2 {
                    self.l1d_writeback_op(idx, virt_addr, cascade);
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
                    if Self::R10K_CACHE_OPS {
                        eprintln!("ip28: C_ILT(SD) idx={idx:#x} ptag={:#x} cs={:#x} pidx={:#x}",
                                  tag.ptag(), tag.cs(), tag.pidx());
                    }
                    let state = match tag.cs() {
                        L2_CS_INVALID => 0,
                        L2_CS_CLEAN_EXCLUSIVE => 4,
                        L2_CS_DIRTY_EXCLUSIVE => 5,
                        L2_CS_SHARED => 6,
                        L2_CS_DIRTY_SHARED => 7,
                        _ => 0,
                    };
                    let fields = (state << 22) | (tag.pidx() << 19) | tag.ptag();
                    // ECC[6:0]: generated, as the R4400 does on every tag write.
                    // The R10000 tag has its own layout and no such field.
                    let ecc = if MODEL == model::R4400 { l2_tag_ecc(fields) } else { 0 };
                    ((tag.ptag() << 13) | (state << 10) | (tag.pidx() << 7) | ecc) as u64
                } else if is_icache {
                    let tag: L1ITag = self.ic.get_tag(idx);
                    let raw_ptag = (tag.ptag >> L1_PTAG_SHIFT) as u32 & L1_PTAG_MASK;
                    // L1-I TagLo, R4400 and R5000 alike:  [31:8] raw_ptag
                    // [7:6] pstate (2=valid, 0=invalid)  [0] P.
                    // The V bit reads back as PState 2 (PCLEANEXCL) on both:
                    // IDE's icachetag requires exactly that after a fill, and
                    // the Indy script runs it on every CPU it shipped with.
                    // P is even parity over PTag and V on the R4400 (UM Figure
                    // 11-3), over PTag alone on the R5000 (VR5000 UM 12-3).
                    let pstate = if tag.is_valid() { 2u32 } else { 0u32 };
                    let p = even_parity(raw_ptag)
                        ^ if Self::IS_R5K { 0 } else { tag.is_valid() as u32 };
                    ((raw_ptag << 8) | (pstate << 6) | p) as u64
                } else {
                    let tag: L1DTag = self.dc.get_tag(idx);
                    let raw_ptag = (tag.ptag >> L1_PTAG_SHIFT) as u32 & L1_PTAG_MASK;
                    // L1-D TagLo, R4400 and R5000 alike:  [31:8] raw_ptag
                    // [7:6] pstate  [0] P. Dirty reads back as DirtyExclusive (3):
                    // IRIX's IP22 parity recovery (mcparity.c) takes PState >
                    // PCLEANEXCL to mean dirty on every IP22 CPU, R5000 included.
                    // The R5000 never produces Shared (1); it has no snooping.
                    // W and its parity W' are not part of TagLo.
                    let pstate = match tag.cs as u32 {
                        L1D_CS_INVALID => 0u32,
                        L1D_CS_SHARED => 1u32,
                        L1D_CS_CLEAN_EXCLUSIVE => if tag.dirty { 3u32 } else { 2u32 },
                        L1D_CS_DIRTY_EXCLUSIVE => 3u32,
                        _ => 0u32,
                    };
                    // P: even parity over PTag and CS on the R4400 (UM Figure
                    // 11-4), over PTag alone on the R5000 (VR5000 UM 12-4).
                    let p = even_parity(raw_ptag)
                        ^ if Self::IS_R5K { 0 } else { even_parity(pstate) };
                    ((raw_ptag << 8) | (pstate << 6) | p) as u64
                }
            }

            // Index Store Tag — write CP0 TagLo into internal tag
            C_IST => {
                let tag_lo = phys_addr as u32;  // R4000-family tags are 32-bit

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
                    // L1 TagLo:  [31:8] raw_ptag  [7:6] PState, on the R4400 and
                    // the R5000 alike. Any non-zero PState is valid: IDE's set_tag
                    // writes PCLEANEXCL (2) to validate an I-cache line, and its
                    // way-pinning on R4600/R5000 stores back exactly what
                    // Index_Load_Tag returned. Must invert the C_ILT encoding.
                    let raw_ptag = (tag_lo >> 8) & L1_PTAG_MASK;
                    let ptag_line = (raw_ptag as u64) << L1_PTAG_SHIFT; // convert to line-base form
                    let pstate = (tag_lo >> 6) & 0x3;
                    let valid = pstate != 0;

                    if is_icache {
                        // Evict existing line first to maintain L1I data pointer integrity.
                        self.invalidate_l1i_line(idx, cascade);
                        self.ic.set_tag(idx, if valid { L1ITag::valid(ptag_line) } else { L1ITag::default() });
                    } else {
                        // The R5000 has no Shared state (no L1 snooping), so a
                        // stored 1 lands as CleanExclusive there.
                        let cs = match pstate {
                            0 => L1D_CS_INVALID as u8,
                            1 if Self::IS_R5K => L1D_CS_CLEAN_EXCLUSIVE as u8,
                            1 => L1D_CS_SHARED as u8,
                            2 => L1D_CS_CLEAN_EXCLUSIVE as u8,
                            _ => L1D_CS_DIRTY_EXCLUSIVE as u8,
                        };
                        let dirty = cs == L1D_CS_DIRTY_EXCLUSIVE as u8;
                        // No writeback. Index_Store_Tag installs a tag and
                        // discards whatever the line held — that is what makes
                        // it usable to initialise a cache whose tags are
                        // powerup garbage, which is exactly what PROMs do with
                        // it. Writing back here sends the line's data to an
                        // address derived from the *old* tag, corrupting
                        // unrelated memory. `C_IWBINV` above is the op that
                        // writes back; the L2 path here already gets this
                        // right.
                        self.invalidate_l1d_line(idx, true, cascade);
                        let mut tag = if cs != L1D_CS_INVALID as u8 { L1DTag::valid(ptag_line, cs, dirty) } else { L1DTag::default() };
                        if self.perr().ce && tag.is_valid() { tag.ptag |= L1DTag::SLOW; }
                        self.dc.set_tag(idx, tag);
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
                    self.l1d_unforce(idx);
                    let mut tag = L1DTag::valid(phys_addr, L1D_CS_DIRTY_EXCLUSIVE as u8, true);
                    if self.perr().ce { tag.ptag |= L1DTag::SLOW; }
                    self.dc.set_tag(idx, tag);
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
                    // With CE set the ECC register replaces the computed
                    // parity of the filled line (R4000 UM ch. 5, VR5000 UM 7.2.9).
                    let r = self.fill_l1i_line(virt_addr, phys_addr);
                    if ECC && r <= 1 && self.perr().ce {
                        if let Some(eidx) = self.hit_l1i(virt_addr, phys_addr) {
                            let ecc = self.perr().ecc;
                            self.perr().ic_forced.push((eidx, ecc));
                        }
                    } else if r > 1 && r & EXEC_IS_CACHE_ERR_BIT != 0 {
                        self.perr().op_status = r;
                    }
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
                        self.l1d_writeback_op(eidx, virt_addr, cascade);
                        self.invalidate_l1d_line(eidx, true, cascade);
                    }
                }
                0
            }

            // Hit Writeback
            C_HWB => {
                if is_icache {
                    if let Some(eidx) = self.hit_l1i(virt_addr, phys_addr) {
                        self.writeback_l1i_line(eidx, virt_addr, phys_addr & !(Self::IC_LINE_MASK as u64));
                    }
                } else {
                    if is_l2 {
                        if let Some(idx) = self.hit_l2(phys_addr) {
                            self.writeback_l2_line(idx);
                        }
                    } else {
                        if let Some(eidx) = self.hit_l1d(virt_addr, phys_addr) {
                            self.l1d_writeback_op(eidx, virt_addr, cascade);
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

    fn set_cache_ecc(&self, v: u32) {
        self.perr().ecc = v as u8;
    }

    /// CE toggles the [`L1DTag::SLOW`] mark on every line, so this is cheap
    /// unless CE actually changes — which only diagnostics do.
    fn set_error_control(&self, ce: bool, detect: bool) {
        if !ECC { return; }
        let e = self.perr();
        e.detect = detect;
        if e.ce != ce {
            e.ce = ce;
            self.mark_l1d_lines(ce);
        }
    }

    fn take_cache_err(&self) -> u32 {
        self.perr().cacheerr
    }

    fn take_op_status(&self) -> u32 {
        if !ECC { return 0; }
        std::mem::take(&mut self.perr().op_status)
    }

    fn cache_op_ecc(&self) -> u32 {
        self.perr().ecc_out
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
        let ll_addr = self.get_lladdr() << 4;
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

    fn get_lladdr(&self) -> u64 {
        unsafe { *self.lladdr.get() }
    }

    fn set_lladdr(&self, addr: u64) {
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
                    self.dc.get_tag(eidx).matches_line(phys_addr)
                });
                let mut s = format!("L1-D probe virt 0x{:016x} phys 0x{:016x} set=0x{:x}: {}\n",
                    virt_addr, phys_addr, set, if any_hit { "HIT" } else { "MISS" });
                for way in 0..num_ways {
                    let eidx = set + way * sets_per_way;
                    let tag: L1DTag = self.dc.get_tag(eidx);
                    let hit = tag.matches_line(phys_addr);
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
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8, const TC: bool> Drop for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL, TC> {
    fn drop(&mut self) {
        self.ic.stop.store(true, Ordering::Relaxed);
    }
}

// ---- Resettable ----

impl<const IC_SIZE: usize, const IC_LINE: usize, const IC_WAYS: usize, const IC_TAGS: usize,
    const DC_SIZE: usize, const DC_LINE: usize, const DC_WAYS: usize, const DC_TAGS: usize, const DC_DATA: usize,
    const L2_CACHE_SIZE: usize, const L2_LINE: usize, const L2_TAGS: usize, const L2_DATA: usize,
    const L2_NINSTRS: usize, const HAS_L2: bool,
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8, const TC: bool> Resettable for CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL, TC> {
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
    const MIPS4: bool, const PRID: u32, const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8, const TC: bool> CpuCache<IC_SIZE, IC_LINE, IC_WAYS, IC_TAGS, DC_SIZE, DC_LINE, DC_WAYS, DC_TAGS, DC_DATA, L2_CACHE_SIZE, L2_LINE, L2_TAGS, L2_DATA, L2_NINSTRS, HAS_L2, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL, TC> {
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
        t.insert("lladdr".into(),   hex_u64(lladdr));
        // 2-way only: LRU as packed u32 words, 1 bit per set (unchanged on-disk format).
        // ic_instrs / ic_raw not saved — load_cache_state rebuilds them (rebuild_l1i_words).
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
        // R5K: l2.instrs is empty; ic_instrs is rebuilt by rebuild_l1i_words below.
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

        // 2-way only: restore LRU bits.
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
            if let Some(a) = toml_u64(f) { unsafe { *self.lladdr.get() = a; } }
        }
        self.rebuild_l1i_words();
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
    use crate::dev::mem::Memory;
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

    /// Deliberately the R5000's exact shape, differing *only* in `MODEL`. If
    /// model identity is ever inferred from a shape parameter again, this
    /// aliases onto the R5000 and the test below fails.
    type NotAnR5000 = CpuCache<32768, 32, 2, 1024,
                               32768, 32, 2, 1024, 4096,
                               128, 128, 1, 16, 0, false,
                               true, 0x0000_0900, 0x0000_0900, 64, { model::R10000 }>;

    /// Two-way associativity is not an identity.
    ///
    /// `IS_R5K` used to be `IC_WAYS == 2`, which was true of every 2-way part
    /// the file knew about. The R10000 is also 2-way and shares neither the
    /// R5000's TagLo layout nor its cache-op semantics, so that inference would
    /// have handed it R5000 behaviour at all 40-odd sites that branch on it —
    /// silently, and each one plausible on its own.
    #[test]
    fn model_identity_is_explicit_and_not_inferred_from_associativity() {
        assert_eq!(<R4400Cache as CpuModel>::NAME, "R4400");
        assert_eq!(<R5000Cache as CpuModel>::NAME, "R5000");
        assert_eq!(<NotAnR5000 as CpuModel>::NAME, "R10000");

        assert!(R5000Cache::IS_R5K, "the R5000 is the R5000");
        assert!(!NotAnR5000::IS_R5K, "a 2-way cache is not what makes an R5000");
        assert!(!R4400Cache::IS_R5K);
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

    /// Index_Load_Tag on a filled line, R4400 and R5000 alike: a valid I-cache
    /// line reads PState 2, a clean D-cache line 2 (CleanExclusive), a dirty
    /// one 3 (DirtyExclusive).
    ///
    /// The R5000 shares the R4400 encoding; this is what the software SGI
    /// shipped for R5000 Indys expects, not a guess from a datasheet:
    ///
    /// - IDE's `icachetag` requires PState == 2 on every filled I-cache line,
    ///   and the IP24 script runs it on every CPU with no model check.
    /// - IRIX's IP22 parity recovery (`mcparity.c`) reads D-cache tags on any
    ///   IP22 CPU and takes PState > 2 to mean dirty, so a clean line must not
    ///   read 3.
    ///
    /// An earlier "R5000 is D(7)/V(6)" encoding read a clean line back as
    /// `0x40`, which fails `icachetag`.
    #[test]
    fn taglo_filled_lines_read_back_the_r4000_pstates_on_both_models() {
        fn check<C: MipsCache + From<Arc<dyn BusDevice>>>(name: &str, ways: u64) {
            let mem = Arc::new(Memory::new(MEM_MB));
            let cache: C = make_cache_of(mem.clone());
            let phys: u32 = 0x3000;
            let va = kseg0(phys);
            // Index ops pick the way with VA[14] on the R5000; probe each.
            let pstate_of = |sel: u32| -> u64 {
                (0..ways)
                    .map(|w| cache.cache_op(C_ILT | sel, va | (w << 14), 0))
                    .find(|t| t >> 8 == (phys >> L1_PTAG_SHIFT) as u64)
                    .map_or(0, |t| (t >> 6) & 3)
            };

            let _ = cache.fetch(va, phys as u64);
            assert_eq!(pstate_of(CACH_PI), 2, "{name}: a filled I-cache line reads PState 2");

            let _ = cache.read::<4>(va, phys as u64);
            assert_eq!(pstate_of(CACH_PD), 2, "{name}: a clean D-cache line reads CleanExclusive");

            let _ = cache.write::<4>(va, phys as u64, 0x1234_5678);
            assert_eq!(pstate_of(CACH_PD), 3, "{name}: a dirty D-cache line reads DirtyExclusive");
        }
        check::<R4400Cache>("R4400", 1);
        check::<R5000Cache>("R5000", 2);
    }

    /// `C_IST` must be the exact inverse of `C_ILT` for whichever model is
    /// active: store a tag, load it back, and get the same state bits.
    ///
    /// A *symmetry* check: it passes even when both halves share the same
    /// wrong encoding, which is exactly how the D/V encoding survived. The
    /// absolute bit positions are pinned by
    /// `taglo_filled_lines_read_back_the_r4000_pstates_on_both_models` and
    /// `taglo_store_decodes_what_ide_and_irix_write`. Kept because an
    /// asymmetric encode/decode pair is its own bug class — and IDE's way
    /// pinning on R4600/R5000 stores back exactly what it loaded.
    #[test]
    fn taglo_store_load_round_trips_on_both_models() {
        fn check<C: MipsCache + From<Arc<dyn BusDevice>>>(name: &str) {
            let cache: C = make_cache_of(Arc::new(Memory::new(MEM_MB)));
            let phys: u32 = 0x5000;
            let va = kseg0(phys);
            let raw_ptag = (phys >> L1_PTAG_SHIFT) & L1_PTAG_MASK;
            for (sel, pstate) in [(CACH_PI, 2u32), (CACH_PD, 2), (CACH_PD, 3)] {
                cache.cache_op(C_IST | sel, va, ((raw_ptag << 8) | (pstate << 6)) as u64);
                let read_back = cache.cache_op(C_ILT | sel, va, 0);
                assert_eq!((read_back >> 6) & 3, pstate as u64, "{name} sel {sel}: PState");
                assert_eq!(read_back >> 8, raw_ptag as u64, "{name} sel {sel}: physical tag");
            }
        }
        check::<R4400Cache>("R4400");
        check::<R5000Cache>("R5000");
    }

    /// `C_IST` decodes the patterns real software stores.
    ///
    /// IDE's `set_tag` writes PCLEANEXCL (`0x80`) to validate an I-cache line
    /// and PDIRTYEXCL (`0xC0`) for a dirty D-cache line, on every IP22 CPU.
    /// The old D/V encoding took `0x80` for *invalid* on the R5000. The R5000
    /// has no Shared state, so a stored 1 is taken as CleanExclusive there.
    #[test]
    fn taglo_store_decodes_what_ide_and_irix_write() {
        let cache: R5000Cache = make_cache_of(Arc::new(Memory::new(MEM_MB)));
        let phys: u32 = 0x7000;
        let va = kseg0(phys);
        let raw_ptag = (phys >> L1_PTAG_SHIFT) & L1_PTAG_MASK;
        let pstate_after = |sel: u32, pattern: u32| {
            cache.cache_op(C_IST | sel, va, ((raw_ptag << 8) | pattern) as u64);
            (cache.cache_op(C_ILT | sel, va, 0) >> 6) & 3
        };

        assert_eq!(pstate_after(CACH_PD, 0x00), 0, "D-cache 0x00: invalid");
        assert_eq!(pstate_after(CACH_PD, 0x40), 2, "D-cache 0x40: no Shared, so CleanExclusive");
        assert_eq!(pstate_after(CACH_PD, 0x80), 2, "D-cache 0x80 PCLEANEXCL: valid, clean");
        assert_eq!(pstate_after(CACH_PD, 0xc0), 3, "D-cache 0xC0 PDIRTYEXCL: valid, dirty");
        assert_eq!(pstate_after(CACH_PI, 0x80), 2, "I-cache 0x80 PCLEANEXCL: valid");
        assert_eq!(pstate_after(CACH_PI, 0x00), 0, "I-cache 0x00: invalid");
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
            let phys = (i * 4) as u64;
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
            mem.write32((i * 4) as u64, *w);
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
    fn mem_write(mem: &Memory, phys: u32, val: u32) { mem.write32(phys as u64, val); }
    fn mem_read(mem: &Memory, phys: u32) -> u32 { mem.read32(phys as u64).data }

    /// The geometry handed to codegen must describe the *sets*, not the total
    /// line count, or the 2-way inline path's `set | way << num_lines_shift`
    /// fold is nonsense: the extra mask bit IS the way bit, so a way-0 index
    /// aliases into way 1's range. R4400 hides this (1 way makes the two
    /// expressions equal), which is why it only ever broke R5000.
    #[test]
    fn jit_dc_geometry_masks_sets_not_lines() {
        // Asserted against the cache types' own constants rather than a live
        // `jit_dc_geometry()` call: under tcache that call legitimately
        // returns `unsupported()` for a bare test cache (no ppmem window), so
        // reading it here would test the null path, not the arithmetic.
        // These are exactly the values `jit_dc_geometry` copies out.
        assert_eq!(R4400Cache::DC_NUM_LINES_MASK as u64, 1023,
                   "R4400: 16K / 16B / 1 way = 1024 sets");
        assert_eq!(R5000Cache::DC_NUM_LINES_MASK as u64, 511,
                   "R5000: 32K / 32B / 2 ways = 512 sets, mask 0x1ff. NOT 1023 \
                    — that counts both ways, and the surplus bit IS the way \
                    bit, so a way-0 index would alias into way 1's range and \
                    the `set | way << num_lines_shift` fold becomes nonsense.");
        assert_eq!(R5000Cache::DC_NUM_LINES_SHIFT, 9,
                   "way bit sits just above the 512 set indices");

        // R4400 is immune to the bug by coincidence — with one way,
        // `DC_SIZE/DC_LINE - 1` and `DC_SIZE/DC_LINE/DC_WAYS - 1` are equal.
        // That is why this only ever broke R5000, and why the wrong form
        // survived review.
        assert_eq!(16384 / 16 - 1, R4400Cache::DC_NUM_LINES_MASK,
                   "the two formulas coincide at 1 way");
        assert_ne!(32768 / 32 - 1, R5000Cache::DC_NUM_LINES_MASK,
                   "and diverge at 2 ways — this is the whole bug");

        // The fold must land way 1 in the upper half of the data array, which
        // is what `dc_data_addr` documents.
        let set = 0x91u64;
        let eidx1 = set | (1u64 << R5000Cache::DC_NUM_LINES_SHIFT);
        let addr1 = (eidx1 << R5000Cache::DC_LINE_SHIFT) & (32768u64 - 1);
        assert!(addr1 >= 32768 / 2, "way 1 data must be in the upper half");

        // And the struct codegen actually receives must carry those values —
        // the constants above are only the *source*. Without tcache a bare
        // cache reports a supported geometry, so this reads the real thing.
        // (Under tcache `jit_dc_geometry` returns `unsupported()` here for
        // want of a ppmem window, which is why this half is cfg'd: asserting
        // it there would test the null path instead of the arithmetic.)
        #[cfg(not(feature = "tcache"))]
        {
            let mem = Arc::new(Memory::new(1));
            let g5 = R5000Cache::new(mem as Arc<dyn BusDevice>).jit_dc_geometry();
            assert!(g5.supported);
            assert_eq!(g5.ways, 2);
            assert_eq!(g5.num_lines_mask, 511,
                       "the geometry handed to codegen must mask SETS");
            assert_eq!(g5.num_lines_shift, 9);
        }
    }

    // Flush a single L2 set to memory.  `phys` is any address within the L2 line.
    // Only valid when L2 is present (r4k or r5k+r5ksc).
    /// A whole-cache flush the way an R10000 kernel does it -- every index of
    /// one way, then of the other, bit 0 picking the way -- reaches every line,
    /// including those this direct-mapped model keeps in its upper half.
    #[test]
    fn r10000_index_flush_reaches_both_ways() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache: R10000Cache = make_cache_of(mem.clone());
        // Upper half of the L1D (VA bit 14) and of the 1 MB L2 (PA bit 19).
        let cases = [(0x9800_0000_0000_6040u64, 0x2040u32), (0x9800_0000_0008_2080, 0x8_2080)];
        for (i, &(va, pa)) in cases.iter().enumerate() {
            let _ = cache.write::<4>(va, pa as u64, 0xa5a5_0000 + i as u64);
        }
        for va in (0..16 * 1024u64).step_by(32) {
            for way in 0..2 {
                cache.cache_op(C_IWBINV | CACH_PD, 0x8000_0000 | va | way, 0);
            }
        }
        for va in (0..512 * 1024u64).step_by(128) {
            for way in 0..2 {
                let a = 0x8000_0000 | va | way;
                cache.cache_op(C_IWBINV | CACH_SD, a, a);
            }
        }
        for (i, &(_, pa)) in cases.iter().enumerate() {
            assert_eq!(mem_read(&mem, pa), 0xa5a5_0000 + i as u32, "line at {pa:#x} not flushed");
        }
    }

    /// One physical line used under two virtual colours (index bits 12-13
    /// differ): the R4400 raises a virtual coherency exception on the
    /// secondary hit; the R10000 resolves it itself, and a store under one
    /// colour is what a load under the other sees.
    #[test]
    fn the_r10000_resolves_a_colour_mismatch_where_the_r4400_raises_vce() {
        let pa = 0x8040u64;
        let (va1, va2) = (0x0010_0040u64, 0x0010_3040u64);

        let mem = Arc::new(Memory::new(MEM_MB));
        mem_write(&mem, pa as u32, 0x1111_2222);
        let r4k: R4400Cache = make_cache_of(mem.clone());
        assert_eq!(r4k.read::<4>(va1, pa).status, BUS_OK);
        assert_eq!(r4k.read::<4>(va2, pa).status, BUS_VCE, "the R4400 raises VCE");

        let mem = Arc::new(Memory::new(MEM_MB));
        mem_write(&mem, pa as u32, 0x1111_2222);
        let r10k: R10000Cache = make_cache_of(mem.clone());
        assert_eq!(r10k.read::<4>(va1, pa).data, 0x1111_2222);
        let r = r10k.read::<4>(va2, pa);
        assert_eq!((r.status, r.data), (BUS_OK, 0x1111_2222), "the R10000 carries on");
        assert_eq!(r10k.write::<4>(va2, pa, 0x3333_4444), BUS_OK);
        assert_eq!(r10k.read::<4>(va1, pa).data, 0x3333_4444, "the old colour's copy was dropped");
    }

    /// The R10000 implements 44 virtual address bits; XContext's fields are
    /// laid out around them, so the full model must say so too.
    #[test]
    fn the_r10000_full_model_has_44_va_bits() {
        assert_eq!(<R10000Cache as CpuModel>::VA_BITS, 44);
        assert_eq!(<crate::cpu::mips_cache_shadow::R10000ShadowCache as CpuModel>::VA_BITS, 44);
        assert_eq!(<R4400Cache as CpuModel>::VA_BITS, 40);
        assert_eq!(<R5000Cache as CpuModel>::VA_BITS, 40);
    }

    /// A cache with `TC = false` holds its data in a `tcache` build too: a
    /// line written through it survives an uncached write behind it (the O2
    /// PROM keeps its stack that way while it tests RAM). `writeback_all`
    /// then puts it in RAM.
    #[test]
    fn a_cache_with_tc_off_holds_data_until_written_back() {
        type HoldingR4400 = CpuCache<16384, 16, 1, 1024, 16384, 16, 1, 1024, 2048,
            1048576, 128, 8192, 131072, 262144, true,
            false, 0x0000_0440, 0x0000_0500, 48, { model::R4400 }, false>;
        type HoldingR5000 = CpuCache<32768, 32, 2, 1024, 32768, 32, 2, 1024, 4096,
            128, 128, 1, 16, 0, false,
            true, 0x0000_2321, 0x0000_2300, 48, { model::R5000 }, false>;
        type HoldingR10000 = CpuCache<32768, 64, 1, 512, 32768, 32, 1, 1024, 4096,
            1048576, 128, 8192, 131072, 262144, true,
            true, 0x0000_0925, 0x0000_0900, 64, { model::R10000 }, false>;
        let mem = Arc::new(Memory::new(MEM_MB));
        let check = |name: &str, writeback: &dyn Fn(), write: &dyn Fn(u64, u32), read: &dyn Fn(u64, u32) -> u32| {
            let pa = 0x4040u32;
            write(kseg0(pa), 0x1234_5678);
            mem_write(&mem, pa, 0xdead_beef);
            assert_eq!(read(kseg0(pa), pa), 0x1234_5678, "{name}: the cache lost its line");
            writeback();
            assert_eq!(mem_read(&mem, pa), 0x1234_5678, "{name}: writeback_all left it in the cache");
        };
        let r10k: HoldingR10000 = make_cache_of(mem.clone());
        check("R10000", &|| r10k.writeback_all(),
            &|va, v| { let _ = r10k.write::<4>(va, va & 0x0FFF_FFFF, v as u64); },
            &|va, pa| r10k.read::<4>(va, pa as u64).data as u32);
        let r5k: HoldingR5000 = make_cache_of(mem.clone());
        check("R5000", &|| r5k.writeback_all(),
            &|va, v| { let _ = r5k.write::<4>(va, va & 0x0FFF_FFFF, v as u64); },
            &|va, pa| r5k.read::<4>(va, pa as u64).data as u32);
        let r4k: HoldingR4400 = make_cache_of(mem.clone());
        check("R4400", &|| r4k.writeback_all(),
            &|va, v| { let _ = r4k.write::<4>(va, va & 0x0FFF_FFFF, v as u64); },
            &|va, pa| r4k.read::<4>(va, pa as u64).data as u32);
    }

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

    /// Index_Store_Tag must not write the line back. Firmware initialising
    /// the cache walks every index storing an invalid tag; if that flushed,
    /// each line's contents would be written to whatever address its
    /// powerup-garbage tag happened to name.
    ///
    /// **This has to be an R5000-class model.** `writeback_l1d_line` sends
    /// dirty data straight to memory only where the L2 is non-inclusive or
    /// absent, which is the `IS_R5K` branch; on the R4400 the line goes to
    /// its inclusive L2 instead and memory never changes, so the same test
    /// written against `R4400Cache` passes whether the bug is present or not.
    /// It did, until that was noticed.
    #[test]
    #[cfg_attr(feature = "tcache", ignore = "write-through: memory is never stale under tcache")]
    fn index_store_tag_discards_the_line_instead_of_writing_it_back() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache: R5000Cache = make_cache_of::<R5000Cache>(mem.clone());

        let phys: u32 = 0x7000;
        let virt = kseg0(phys);
        mem_write(&mem, phys, 0xdead_beef);
        // Dirty the line in L1D without letting it reach memory.
        let _ = cache.write::<4>(virt, phys as u64, 0x1122_3344_u64);

        let dc_set = (phys as usize >> R5000Cache::DC_LINE_SHIFT as usize)
            & R5000Cache::DC_NUM_LINES_MASK;
        let idx_addr = (dc_set << R5000Cache::DC_LINE_SHIFT as usize) as u64;

        // Store an invalid tag over it, the way cache init does.
        cache.cache_op(C_IST | CACH_PD, idx_addr, 0);

        assert_eq!(
            mem_read(&mem, phys), 0xdead_beef,
            "Index_Store_Tag wrote the dirty line back to memory"
        );
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

    /// `Index_Store_Tag(SD)` writes the line its *index address* names, not the
    /// line its TagLo value happens to name.
    ///
    /// Both manuals say the index op takes the block from the CACHE
    /// instruction's address (R4000 UM A-42, VR5000 UM §3.4). The executor
    /// passes TagHi:TagLo in `phys_addr` for C_IST, and the L2 index was once
    /// computed from that, so a store landed in whichever set TagLo[19:7]
    /// selected. `cache_op_ilt_ist_l2` above never saw it: it stores back the
    /// tag it just read, so the read-back hits the untouched original.
    #[test]
    fn l2_index_store_tag_indexes_by_address_not_by_taglo() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());

        let line = R4400Cache::L2_LINE as u64;
        let target_set = 0x123u64;
        let index_addr = 0x8000_0000 | target_set * line;
        // CleanExclusive, PIdx 5, STag 0x1abc. TagLo[19:7] — the bits the old
        // code read as an index — name set 0xf25 instead.
        let tag_lo: u64 = (0x1abc << 13) | (4 << 10) | (5 << 7);
        let taglo_set = (tag_lo >> R4400Cache::L2_LINE_SHIFT) & (R4400Cache::L2_SIZE as u64 / line - 1);
        assert_ne!(taglo_set, target_set, "test needs TagLo to name a different set");

        cache.cache_op(C_IST | CACH_SD, index_addr, tag_lo);

        let fields = |v: u64| v & !0x7f; // ECC[6:0] is generated, not stored
        assert_eq!(fields(cache.cache_op(C_ILT | CACH_SD, index_addr, index_addr)), tag_lo,
                   "tag did not land at the index address");
        let other = 0x8000_0000 | taglo_set * line;
        assert_eq!(fields(cache.cache_op(C_ILT | CACH_SD, other, other)), 0,
                   "tag landed in the set TagLo names");
    }

    /// IDE `icache2`'s `icache_fill_writeback`, one line: the I-cache keeps the
    /// words it was filled with, and Hit_Writeback(I) puts them back after
    /// memory has changed under it. Hit_Writeback(I) was a no-op, so `icache2`
    /// reported `Cache err: … Actual:0xffffffff` on every word.
    ///
    /// Not under tcache: there a cached store goes straight to RAM, so the
    /// I-cache fill already sees the new words — the same trade
    /// `cache/hit_inv_discards` makes.
    #[test]
    #[cfg(not(feature = "tcache"))]
    fn icache_hit_writeback_restores_the_words_it_was_filled_with() {
        fn check<C: MipsCache + From<Arc<dyn BusDevice>>>(name: &str) {
            let mem = Arc::new(Memory::new(MEM_MB));
            let cache: C = make_cache_of(mem.clone());
            let phys: u32 = 0x8000;
            let va = kseg0(phys);
            let words = C::IC_LINE as u32 / 4;
            for i in 0..words { mem_write(&mem, phys + 4 * i, 0x1111_0000 + i); }

            // Dirty the D-cache with other words, fill the I-cache, then push
            // the D-cache's words all the way to memory.
            for i in 0..words { cache.write::<4>(va + 4 * i as u64, (phys + 4 * i) as u64, 0xdead_0000 + i as u64); }
            cache.cache_op(C_HWBINV | CACH_PI, va, phys as u64); // Fill(I)
            // IDE on R4600/R4700/R5000: pin the line into the way VA[14]
            // selects — load its tag, invalidate it, store the tag back by
            // index, Fill again. The Fill must refill the pinned way.
            let tag = cache.cache_op(C_ILT | CACH_PI, va, 0);
            cache.cache_op(C_HINV | CACH_PI, va, phys as u64);
            cache.cache_op(C_IST | CACH_PI, va, tag);
            cache.cache_op(C_HWBINV | CACH_PI, va, phys as u64);
            cache.cache_op(C_HWBINV | CACH_PD, va, phys as u64);
            cache.cache_op(C_HWB | CACH_SD, va, phys as u64);
            assert_eq!(mem_read(&mem, phys), 0xdead_0000, "{name}: D-cache words reached memory");

            cache.cache_op(C_HWB | CACH_PI, va, phys as u64);
            cache.cache_op(C_HWB | CACH_SD, va, phys as u64);
            for i in 0..words {
                assert_eq!(mem_read(&mem, phys + 4 * i), 0x1111_0000 + i,
                           "{name}: word {i} is not what the I-cache was filled with");
            }
        }
        check::<R4400Cache>("R4400");
        check::<R5000Cache>("R5000");
    }

    /// A snapshot carries I-cache tags but not the words, so a restore must
    /// rebuild them: a restored valid line is served without a refill. Before,
    /// it was served from empty slots, i.e. a line of NOPs.
    #[test]
    fn restored_icache_lines_hold_their_instructions() {
        fn check<C: MipsCache + From<Arc<dyn BusDevice>>>(name: &str) {
            let mem = Arc::new(Memory::new(MEM_MB));
            let phys: u32 = 0x9000;
            let va = kseg0(phys);
            for i in 0..4 { mem_write(&mem, phys + 4 * i, 0x2400_0000 + i); } // addiu
            let saved = {
                let cache: C = make_cache_of(mem.clone());
                let _ = cache.fetch(va, phys as u64);
                cache.save_cache_state()
            };
            let cache: C = make_cache_of(mem.clone());
            cache.load_cache_state(&saved).unwrap();
            for i in 0..4u32 {
                let r = cache.fetch(va + 4 * i as u64, (phys + 4 * i) as u64);
                assert_eq!(unsafe { (*r.instr).raw }, 0x2400_0000 + i, "{name}: word {i}");
            }
        }
        check::<R4400Cache>("R4400");
        check::<R5000Cache>("R5000");
    }

    /// IDE `dcache1`, one word: with Status.CE set a store takes the ECC
    /// register as its check bits, and the next load of it, with DE and ERL
    /// clear, takes a Cache Error with CacheErr.ED set. Both CPUs implement
    /// CE the same way (R4000 UM ch. 5, VR5000 UM 7.2.9).
    #[cfg(feature = "cache-ecc")]
    #[test]
    fn forced_parity_raises_a_cache_error_on_the_next_load() {
        fn check<C: MipsCache + From<Arc<dyn BusDevice>>>(name: &str) {
            let mem = Arc::new(Memory::new(MEM_MB));
            let cache: C = make_cache_of(mem.clone());
            let phys: u32 = 0xa000;
            let va = kseg0(phys);
            cache.set_error_control(false, true);
            assert_eq!(cache.write::<4>(va, phys as u64, 0), BUS_OK);

            // IDE: set_ecc(0x55), SetSR(SR_FORCE_ON), store, load.
            cache.set_cache_ecc(0x55);
            cache.set_error_control(true, true);
            assert_eq!(cache.write::<4>(va, phys as u64, 0x1010_1010), BUS_OK);
            let r = cache.read::<4>(va, phys as u64);
            assert_eq!(r.status, EXEC_CACHE_ERR, "{name}: load of a forced word");
            let err = cache.take_cache_err();
            assert_ne!(err & CACHEERR_ED, 0, "{name}: CacheErr.ED");
            assert_ne!(err & CACHEERR_ER, 0, "{name}: CacheErr.ER (data)");

            // DE (or ERL) masks it: the load completes with the data.
            cache.set_error_control(true, false);
            assert_eq!(cache.read::<4>(va, phys as u64).data, 0x1010_1010, "{name}: DE masks");

            // The other word of the doubleword was not stored with CE.
            cache.set_error_control(false, true);
            assert!(cache.read::<4>(va + 4, phys as u64 + 4).is_ok(), "{name}: untouched lanes");

            // IDE's handler: clear CE, Hit_Invalidate, store again.
            cache.cache_op(C_HINV | CACH_PD, va, phys as u64);
            assert_eq!(cache.write::<4>(va, phys as u64, 0x1010_1010), BUS_OK);
            assert!(cache.read::<4>(va, phys as u64).is_ok(), "{name}: invalidated, refilled");

            // Rewriting a forced word with CE clear also repairs it.
            cache.set_error_control(true, true);
            assert_eq!(cache.write::<4>(va, phys as u64, 0), BUS_OK);
            cache.set_error_control(false, true);
            assert_eq!(cache.write::<4>(va, phys as u64, 7), BUS_OK);
            assert_eq!(cache.read::<4>(va, phys as u64).data, 7, "{name}: repaired by a clean store");

        }
        check::<R4400Cache>("R4400");
        check::<R5000Cache>("R5000");
    }

    /// The parity machinery must leave the hit path alone: with CE clear and
    /// nothing forced, no line carries the slow mark.
    #[cfg(feature = "cache-ecc")]
    #[test]
    fn no_line_stays_marked_once_ce_clears_and_nothing_is_forced() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());
        let phys: u32 = 0xb000;
        let va = kseg0(phys);
        cache.set_error_control(false, true);
        let _ = cache.read::<4>(va, phys as u64);
        cache.set_error_control(true, true);
        let idx = cache.dc.get_index(va);
        assert_ne!(cache.dc.get_tag(idx).ptag & L1DTag::SLOW, 0, "CE marks resident lines");
        cache.set_error_control(false, true);
        assert!(cache.dc.get_tag(idx).matches_phys(phys as u64), "unmarked again");
    }

    /// IDE `icache1`, one line: with CE set, Fill(I) takes the ECC register as
    /// the line's parity, and Hit_Writeback(I) then raises a Cache Error from
    /// the CACHE instruction itself. With DE set it writes back normally; a
    /// refill without CE clears it.
    #[cfg(feature = "cache-ecc")]
    #[test]
    fn forced_icache_parity_faults_on_writeback() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());
        let phys: u32 = 0xe000;
        let va = kseg0(phys);
        cache.set_cache_ecc(0x55);
        cache.set_error_control(true, true);              // SR_FORCE_ON
        cache.cache_op(C_HWBINV | CACH_PI, va, phys as u64); // Fill(I)
        cache.set_error_control(false, true);             // SR_FORCE_OFF
        cache.cache_op(C_HWB | CACH_PI, va, phys as u64);
        assert_eq!(cache.take_op_status(), EXEC_CACHE_ERR);
        assert_ne!(cache.take_cache_err() & CACHEERR_ED, 0);
        assert_eq!(cache.take_op_status(), 0, "reading the status clears it");

        cache.set_error_control(false, false);            // DE: no fault
        cache.cache_op(C_HWB | CACH_PI, va, phys as u64);
        assert_eq!(cache.take_op_status(), 0);

        cache.set_error_control(false, true);
        cache.cache_op(C_HINV | CACH_PI, va, phys as u64);
        cache.cache_op(C_HWBINV | CACH_PI, va, phys as u64); // Fill, CE clear
        cache.cache_op(C_HWB | CACH_PI, va, phys as u64);
        assert_eq!(cache.take_op_status(), 0, "a refill without CE computes parity");
    }

    /// Fill(I) refills the way that already holds the line, not the LRU way:
    /// IDE pins a line into the way VA[14] selects with Index_Store_Tag and
    /// then fills it.
    #[test]
    fn icache_fill_refills_the_way_holding_the_line() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache: R5000Cache = make_cache_of(mem.clone());
        let phys: u32 = 0x8000;
        let va = kseg0(phys);
        mem_write(&mem, phys, 0x2400_0042);
        let raw_ptag = (phys >> L1_PTAG_SHIFT) & L1_PTAG_MASK;
        let way1 = va | 1 << 14;
        cache.cache_op(C_IST | CACH_PI, way1, ((raw_ptag << 8) | 0x80) as u64);
        cache.cache_op(C_HWBINV | CACH_PI, va, phys as u64); // Fill(I)
        let eidx1 = cache.ic.get_index(va) | 1 << R5000Cache::IC_NUM_LINES_SHIFT;
        assert_eq!(cache.l1i_word(eidx1, 0), 0x2400_0042, "way 1 was refilled");
        let eidx0 = cache.ic.get_index(va);
        assert!(!cache.ic.get_tag(eidx0).is_valid(), "way 0 untouched");
    }

    /// `L2_DATA_ECC_MASKS` is R4000 UM Figure 16-1, checked against what the
    /// figure states: 27 ones per row with the check bit, weight-3 syndromes
    /// except weight 5 for data bits 59, 58, 53, 52, 11, 10, 5 and 4, all
    /// distinct, and the text's example (data bit 0 is `0001 0011`).
    #[test]
    fn l2_data_ecc_matrix_is_the_manuals() {
        assert!(L2_DATA_ECC_MASKS.iter().all(|m| m.count_ones() + 1 == 27));
        let mut seen = std::collections::HashSet::new();
        for bit in 0..64 {
            let syndrome = l2_data_ecc(1 << bit);
            let weight = if [59, 58, 53, 52, 11, 10, 5, 4].contains(&bit) { 5 } else { 3 };
            assert_eq!(syndrome.count_ones(), weight, "data bit {bit}");
            assert!(seen.insert(syndrome), "data bit {bit} shares a syndrome");
        }
        assert_eq!(l2_data_ecc(1), 0b0001_0011);
    }

    /// IDE `scache1`, both halves, one doubleword.
    ///
    /// `scache_ecc_ram`: with CE set, a D-cache Index_Writeback_Invalidate XORs
    /// the ECC register into the check bits of the doubleword it addresses as
    /// the line goes to L2, and Index_Load_Tag(SD) reads them back into ECC.
    /// `scache_ecc_intr`: with DE clear, refilling the primary from that line
    /// takes a Cache Error with CacheErr.EC and ED set; Hit_Invalidate(SD)
    /// clears it.
    #[cfg(feature = "cache-ecc")]
    #[test]
    fn forced_secondary_ecc_reads_back_and_faults_on_refill() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());
        let phys: u32 = 0xc000;
        let va = kseg0(phys);
        cache.set_error_control(false, false); // SR_NOERR: DE set
        assert_eq!(cache.write::<4>(va, phys as u64, 0), BUS_OK);
        cache.set_cache_ecc(0x55);
        cache.set_error_control(true, false); // SR_FORCE_NOERR
        cache.cache_op(C_IWBINV | CACH_PD, va, va);
        cache.set_error_control(false, false);
        let _ = cache.read::<4>(va, phys as u64); // DE: no fault
        cache.set_cache_ecc(0);
        cache.cache_op(C_ILT | CACH_SD, va, va);
        assert_eq!(cache.cache_op_ecc(), 0x55, "data 0 has check bits 0, so ECC reads the forced pattern");
        // Other doublewords of the line were written back with computed bits.
        cache.cache_op(C_ILT | CACH_SD, va + 8, va + 8);
        assert_eq!(cache.cache_op_ecc(), 0);

        // scache_ecc_intr: refill with detection on.
        cache.cache_op(C_HINV | CACH_PD, va, phys as u64);
        cache.set_error_control(false, true);
        let r = cache.read::<4>(va, phys as u64);
        assert_eq!(r.status, EXEC_CACHE_ERR);
        let err = cache.take_cache_err();
        assert_eq!(err & (CACHEERR_EC | CACHEERR_ED), CACHEERR_EC | CACHEERR_ED, "secondary, data field");

        cache.cache_op(C_HINV | CACH_SD, va, phys as u64);
        assert!(cache.read::<4>(va, phys as u64).is_ok(), "Hit_Invalidate(SD) clears it");
    }

    /// Index_Load_Tag(SD) on ordinary data loads the computed data ECC.
    #[test]
    #[cfg(all(feature = "cache-ecc", not(feature = "tcache")))]
    fn index_load_tag_returns_the_data_ecc() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());
        let phys: u32 = 0xd000;
        let va = kseg0(phys);
        mem_write(&mem, phys, 0x1234_5678);
        mem_write(&mem, phys + 4, 0x9abc_def0);
        let _ = cache.read::<4>(va, phys as u64);
        cache.cache_op(C_ILT | CACH_SD, va, va);
        assert_eq!(cache.cache_op_ecc(), l2_data_ecc(0x1234_5678_9abc_def0) as u32);
    }

    /// `L2_TAG_ECC_MASKS` is a transcription of R4000 UM Figure 16-4, and
    /// these are the properties that figure states for itself: the "number of
    /// 1s in syndrome" row says every tag bit has a weight-3 syndrome, the row
    /// totals (check bit included) are 14,11,13,10,10,13,11 from check bit 0
    /// up, and a SECDED code needs every syndrome distinct. A mis-copied bit
    /// breaks at least one of them.
    #[test]
    fn l2_tag_ecc_matrix_is_the_manuals() {
        let rows: Vec<u32> = L2_TAG_ECC_MASKS.iter().map(|m| m.count_ones() + 1).collect();
        assert_eq!(rows, [14, 11, 13, 10, 10, 13, 11]);
        let mut seen = std::collections::HashSet::new();
        for bit in 0..25 {
            let syndrome = l2_tag_ecc(1 << bit);
            assert_eq!(syndrome.count_ones(), 3, "tag bit {bit}");
            assert!(seen.insert(syndrome), "tag bit {bit} shares a syndrome");
        }
        assert_eq!(L2_TAG_ECC_MASKS.iter().fold(0, |a, m| a | m), 0x1ff_ffff, "covers [24:0]");
    }

    /// Index_Load_Tag(SD) returns the check bits for the tag it returns, and
    /// ignores whatever ECC TagLo carried on the store (R4000 UM p. 93).
    #[test]
    fn l2_index_load_tag_generates_the_tag_ecc() {
        let mem = Arc::new(Memory::new(MEM_MB));
        let cache = make_cache(mem.clone());
        let index_addr = 0x8000_0000 | 0x40 * R4400Cache::L2_LINE as u64;
        let fields: u32 = (5 << 22) | (3 << 19) | 0x2_1357; // DirtyExcl, PIdx 3, STag
        let tag_lo = ((fields & 0x7_ffff) << 13) | ((fields >> 22) << 10) | (((fields >> 19) & 7) << 7);

        cache.cache_op(C_IST | CACH_SD, index_addr, (tag_lo | 0x55) as u64);
        let got = cache.cache_op(C_ILT | CACH_SD, index_addr, index_addr) as u32;

        assert_eq!(got & !0x7f, tag_lo);
        assert_eq!(got & 0x7f, l2_tag_ecc(fields));
        assert_ne!(got & 0x7f, 0x55, "the stored ECC field must be ignored");
    }

    /// TagLo[0] is the tag's even parity, generated by the hardware: over PTag
    /// and the state on an R4400 (UM Figures 11-3/11-4), over PTag alone on an
    /// R5000 (VR5000 UM Figures 12-3/12-4). Whatever TagLo[0] held on the store
    /// is ignored.
    #[test]
    fn l1_index_load_tag_generates_the_tag_parity() {
        fn check<C: MipsCache + From<Arc<dyn BusDevice>>>(state_counts: bool) {
            let cache: C = make_cache_of(Arc::new(Memory::new(MEM_MB)));
            for sel in [CACH_PI, CACH_PD] {
                for ptag in [0u32, 1, 3, 0x12_3457] {
                    // Valid; Dirty on the R5000 D-cache, DirtyExcl on the R4400.
                    let tag_lo = (ptag << 8) | 0xc0;
                    cache.cache_op(C_IST | sel, 0x8000_0000, (tag_lo | 1) as u64);
                    let got = cache.cache_op(C_ILT | sel, 0x8000_0000, 0x8000_0000) as u32;
                    let state = (got >> 6) & 3;
                    let want = even_parity(ptag) ^ if state_counts { even_parity(state) } else { 0 };
                    assert_eq!(got & 1, want, "sel {sel} ptag {ptag:#x} taglo {got:#x}");
                    assert_eq!(got & 0x3e, 0, "TagLo[5:1] reads zero");
                }
            }
        }
        check::<R4400Cache>(true);
        check::<R5000Cache>(false);
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
                    let mem_now = mem.read32(phys as u64).data;
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
            bank_a.write32(i as u64 * 4, v);
            bank_b.write32(i as u64 * 4, v);
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
            let a = bank_a.read32(i as u64 * 4).data;
            let b = bank_b.read32(i as u64 * 4).data;
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
            bank_a.write32(i as u64 * 4, v);
            bank_b.write32(i as u64 * 4, v);
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
                bank_a.read32(i as u64 * 4).data,
                bank_b.read32(i as u64 * 4).data,
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

    /// The bus fallback in `tc_read`/`tc_write` is not dead code: the MIPS
    /// exception vectors live at physical 0x0 / 0x80 / 0x180, which sit in
    /// bitmap region 0. Only the low 512KB of that 64MB region is ever mapped
    /// (the alias), so the bitmap cannot claim it and those accesses must take
    /// the bus. Observed live — the bench harness installs its handlers there
    /// at startup.
    #[test]
    fn exception_vectors_need_the_bus_fallback() {
        let bank = Arc::new(PpMemory::new(BANK_MB));
        let space = PpMemSpace::over(std::slice::from_ref(&*bank)).unwrap();
        space.map_bank(0, BASE, REGION, REGION).unwrap();
        space.map_alias(0, 0, 512 * 1024).unwrap();
        let tc = R4400Cache::new(bank.clone() as Arc<dyn BusDevice>);
        unsafe {
            tc.set_tcache_window_impl(space.window_base());
            space.set_bitmap_sink2(tc.tc_bitmap_ptr());
        }

        for vec in [0x0u64, 0x80, 0x180] {
            assert!(
                !tc.tc_transparent(vec),
                "vector {vec:#x} unexpectedly transparent — region 0 is only \
                 partially mapped, so the bitmap must not claim it"
            );
        }
        assert!(tc.tc_transparent(BASE), "LOMEM should still be transparent");

        // And the fallback must actually work: a write through the cache to a
        // vector address has to be readable back.
        let va = 0xFFFF_FFFF_8000_0000u64; // kseg0 phys 0
        tc.write::<4>(va, 0x0, 0x0800_0180);
        assert_eq!(
            tc.read::<4>(va, 0x0).data as u32,
            0x0800_0180,
            "bus fallback did not round-trip an exception-vector write"
        );
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
        unsafe {
            cache.set_tcache_window_impl(space.window_base());
            space.set_bitmap_sink2(cache.tc_bitmap_ptr());
        }

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
        unsafe {
            cache.set_tcache_window_impl(space.window_base());
            space.set_bitmap_sink2(cache.tc_bitmap_ptr());
        }

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
