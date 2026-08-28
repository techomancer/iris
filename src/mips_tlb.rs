// MIPS TLB (Translation Lookaside Buffer) interface and implementations

use crate::mips_exec::CacheAttr;
use std::fmt::Write;
use crate::snapshot::{u64_slice_to_toml, load_u64_slice};

/// Number of TLB entries in R4000 (48 dual-entries = 96 pages)
pub const TLB_NUM_ENTRIES: usize = 48;

// ── TLB statistics (feature = "tlbstats") ────────────────────────────────────

#[cfg(feature = "tlbstats")]
#[derive(Default, Clone)]
pub struct TlbAccessStats {
    /// Of translate() calls, how many VAs were 32-bit / sign-extended 64-bit.
    pub addr32:           u64,
    /// Of translate() calls, how many VAs were full 64-bit (non-sign-extended).
    pub addr64:           u64,
    /// NanoTLB hits (translate() never called).
    pub nano_hit:         u64,
    /// NanoTLB misses (fell through to translate()). Equals total translate() calls.
    pub nano_miss:        u64,
    /// nutlb hits (Read/Write only — Fetch stays nanotlb).
    pub nutlb_hit:        u64,
    /// nutlb misses (fell through to translate()).
    pub nutlb_miss:       u64,
    /// Of those misses, how many found the set holding a *different* valid
    /// page — i.e. conflict rather than cold/capacity. High values argue for
    /// a larger NUTLB_BITS (or associativity); low values say the working set
    /// simply doesn't fit and more sets won't help.
    pub nutlb_conflict:   u64,
    /// Vmap found an entry and ASID/global matched → Hit/Invalid/Modified.
    pub vmap_hit:         u64,
    /// Vmap found an entry but it was Invalid (V=0).
    pub vmap_invalid:     u64,
    /// Vmap found an entry but write to non-dirty (Modified).
    pub vmap_modified:    u64,
    /// Vmap found an entry but ASID mismatched (fell through to scan).
    pub vmap_asid_fall:   u64,
    /// Vmap slot was VMAP_MISS → definite miss without scan.
    pub vmap_miss:        u64,
    /// Number of MRU scan invocations (for average scan length).
    pub scan_invocations:    u64,
    /// Total TLB shadow entries walked across all scan invocations.
    pub scan_entries_walked: u64,
    /// Scan found a matching entry → Hit.
    pub scan_hit:         u64,
    /// Scan found a matching entry but it was Invalid.
    pub scan_invalid:     u64,
    /// Scan found a matching entry but write to non-dirty.
    pub scan_modified:    u64,
    /// Scan completed without finding a match → Miss.
    pub scan_miss:        u64,
}

#[cfg(feature = "tlbstats")]
#[derive(Default, Clone)]
pub struct TlbStats {
    pub by_type: [TlbAccessStats; 4], // indexed by AccessType discriminant
}

#[cfg(feature = "tlbstats")]
fn pct(num: u64, den: u64) -> f64 {
    if den == 0 { 0.0 } else { num as f64 / den as f64 * 100.0 }
}

#[cfg(feature = "tlbstats")]
impl TlbStats {
    pub fn print(&self) {
        const NAMES: [&str; 4] = ["Fetch", "Read ", "Write", "Debug"];
        eprintln!("\n=== TLB Statistics ===");
        for (i, s) in self.by_type.iter().enumerate() {
            if s.nano_hit == 0 && s.nano_miss == 0 && s.addr32 == 0 && s.addr64 == 0 { continue; }
            // nanotlb layer
            let nano_total = s.nano_hit + s.nano_miss;
            if nano_total > 0 {
                eprintln!("  [{name}]  nanotlb: total={total}  hit={nh} {nhp:.1}%  miss={nm} {nmp:.1}%",
                    name = NAMES[i], total = nano_total,
                    nh = s.nano_hit,  nhp = pct(s.nano_hit,  nano_total),
                    nm = s.nano_miss, nmp = pct(s.nano_miss, nano_total));
            }
            // nutlb layer (Read/Write only; Fetch never routes through it)
            {
                let nu_total = s.nutlb_hit + s.nutlb_miss;
                if nu_total > 0 {
                    eprintln!("  [{name}]  nutlb:   total={total}  hit={uh} {uhp:.1}%  miss={um} {ump:.1}%  (conflict={uc} {ucp:.1}% of misses)",
                        name = NAMES[i], total = nu_total,
                        uh = s.nutlb_hit,  uhp = pct(s.nutlb_hit,  nu_total),
                        um = s.nutlb_miss, ump = pct(s.nutlb_miss, nu_total),
                        uc = s.nutlb_conflict, ucp = pct(s.nutlb_conflict, s.nutlb_miss));
                }
            }
            // TLB translate() layer — may be called by paths other than nanotlb
            let tr = s.addr32 + s.addr64;
            if tr == 0 { continue; }
            eprintln!("  [{name}]  translate: calls={tr}  addr32={a32} {a32p:.1}%  addr64={a64} {a64p:.1}%",
                name = NAMES[i], tr = tr,
                a32 = s.addr32, a32p = pct(s.addr32, tr),
                a64 = s.addr64, a64p = pct(s.addr64, tr));
            // vmap outcomes (only for addr32 path)
            let vmap_total = s.vmap_hit + s.vmap_invalid + s.vmap_modified + s.vmap_asid_fall + s.vmap_miss;
            if vmap_total > 0 {
                eprintln!("           vmap: hit={vh} {vhp:.1}%  invalid={vi} {vip:.1}%  modified={vm} {vmp:.1}%  asid_fall={vf} {vfp:.1}%  miss={vms} {vmsp:.1}%",
                    vh = s.vmap_hit,      vhp  = pct(s.vmap_hit,      vmap_total),
                    vi = s.vmap_invalid,  vip  = pct(s.vmap_invalid,  vmap_total),
                    vm = s.vmap_modified, vmp  = pct(s.vmap_modified, vmap_total),
                    vf = s.vmap_asid_fall,vfp  = pct(s.vmap_asid_fall,vmap_total),
                    vms= s.vmap_miss,     vmsp = pct(s.vmap_miss,     vmap_total));
            }
            // scan outcomes
            if s.scan_invocations > 0 {
                let avg = s.scan_entries_walked as f64 / s.scan_invocations as f64;
                eprintln!("           scan: invocations={si}  avg_len={avg:.2}  hit={sh} {shp:.1}%  invalid={sinv} {sinvp:.1}%  modified={smod} {smodp:.1}%  miss={sms} {smsp:.1}%",
                    si   = s.scan_invocations,  avg = avg,
                    sh   = s.scan_hit,      shp  = pct(s.scan_hit,      s.scan_invocations),
                    sinv = s.scan_invalid,  sinvp= pct(s.scan_invalid,  s.scan_invocations),
                    smod = s.scan_modified, smodp= pct(s.scan_modified, s.scan_invocations),
                    sms  = s.scan_miss,     smsp = pct(s.scan_miss,     s.scan_invocations));
            }
        }
        eprintln!("=== End TLB Statistics ===\n");
    }
}

/// Type of memory access for translation.
/// Variants are assigned explicit discriminants so they can index arrays (0..=3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AccessType {
    Fetch = 0,
    Read  = 1,
    Write = 2,
    /// Debug access: like Read but overrides privilege to Kernel and never
    /// mutates CP0 state (BadvAddr, EntryHi, Context, XContext).
    Debug = 3,
}

/// TLB Entry structure matching R4000 TLB format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlbEntry {
    /// Page Mask - determines page size (4KB to 16MB)
    /// 64-bit register, bits used depend on addressing mode
    pub page_mask: u64,

    /// Entry Hi - contains VPN2 and ASID
    /// Bits 63:13 (or 39:13 for 40-bit addresses): VPN2 (Virtual Page Number / 2)
    /// Bits 7:0: ASID (Address Space Identifier)
    /// 64-bit register, bits used depend on addressing mode
    pub entry_hi: u64,

    /// Even/odd page mappings: index 0 = Lo0 (even), index 1 = Lo1 (odd).
    /// Bit 63:6 (or 29:6 for 32-bit PFN): PFN (Page Frame Number)
    /// Bits 5:3: Cache coherency attribute (C)
    /// Bit 2: Dirty (D) - writable if set
    /// Bit 1: Valid (V) - entry is valid if set
    /// Bit 0: Global (G) - ignore ASID if set
    pub entry_lo: [u64; 2],

    /// Bit position of the even/odd selector bit within a VA.
    /// Derived from page_mask: `trailing_ones(page_mask | 0x1FFF) - 1`.
    /// Equals `log2(selector_bit)` where `selector_bit = (page_mask|0x1FFF+1) >> 1`.
    /// `(va >> selector_bit_shift) & 1` gives the entry_lo index (0=even, 1=odd).
    pub selector_bit_shift: u8,

    /// Pre-masked VPN comparison fields (derived, not stored in hardware).
    /// `vcmp32` / `vcmp64`: VPN compare mask for 32-bit / 64-bit mode.
    /// `vpn_hi32` / `vpn_hi64`: `entry_hi & vcmp32/64` — ready to compare directly
    /// against `virt_addr & vcmp32/64` with no per-lookup masking of entry_hi.
    pub vcmp32:   u64,
    pub vpn_hi32: u64,
    pub vcmp64:   u64,
    pub vpn_hi64: u64,

    /// Page-offset mask: `(1 << selector_bit_shift) - 1`.
    /// Used to extract the within-page VA bits on every translation.
    pub offset_mask: u64,

    /// Pre-computed physical base address for each page (even=0, odd=1).
    /// `pfn_base[i] = (entry_lo[i] << 6) & (0xFF_FFFF_FFFF_000 & !offset_mask)`.
    /// Hot-path phys_addr = `pfn_base[idx] | (virt_addr & offset_mask)`.
    pub pfn_base: [u64; 2],
}

impl TlbEntry {
    /// Create a new empty/invalid TLB entry
    pub fn new() -> Self {
        Self {
            page_mask: 0,
            entry_hi: 0,
            entry_lo: [0; 2],
            // page_mask=0 → mask|0x1FFF = 0x1FFF → trailing_ones=13 → shift=12 (bit 12 = 4KB selector)
            selector_bit_shift: 12,
            // entry_hi=0 so all pre-masked VPN fields are 0
            vcmp32: !0x1FFF_u64 & 0x0000_0000_FFFF_E000,
            vpn_hi32: 0,
            vcmp64: !0x1FFF_u64 & 0xC000_00FF_FFFF_E000,
            vpn_hi64: 0,
            // page_mask=0, shift=12 → offset_mask=0xFFF; entry_lo all-zero → pfn_base=[0,0]
            offset_mask: 0xFFF,
            pfn_base: [0; 2],
        }
    }

    /// Check if this entry is valid for the even page
    #[inline]
    pub fn is_valid_even(&self) -> bool {
        (self.entry_lo[0] & 0x2) != 0
    }

    /// Check if this entry is valid for the odd page
    #[inline]
    pub fn is_valid_odd(&self) -> bool {
        (self.entry_lo[1] & 0x2) != 0
    }

    /// Check if this entry is global (ignores ASID)
    /// Per MIPS R4000 spec: G bit is stored in bit 12 of EntryHi in TLB entries
    #[inline]
    pub fn is_global(&self) -> bool {
        (self.entry_hi & 0x1000) != 0 // Check bit 12
    }

    /// Get ASID from entry_hi
    #[inline]
    pub fn asid(&self) -> u8 {
        (self.entry_hi & 0xFF) as u8
    }

    /// Get VPN2 (Virtual Page Number / 2) from entry_hi
    /// In 32-bit mode: 19 bits (31:13)
    /// In 64-bit mode: 27 bits (39:13)
    #[inline]
    pub fn vpn2(&self) -> u64 {
        (self.entry_hi >> 13) & 0x7FF_FFFF // Mask to 27 bits (covers both modes)
    }

    /// Get Region (R) field from entry_hi (64-bit mode only)
    /// Returns bits 63:62
    #[inline]
    pub fn region(&self) -> u8 {
        ((self.entry_hi >> 62) & 0x3) as u8
    }
}

impl Default for TlbEntry {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a TLB lookup
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlbResult {
    /// TLB hit - contains physical address, cache attribute, and dirty bit
    Hit {
        phys_addr: u64,
        cache_attr: CacheAttr,
        dirty: bool,
        /// Entry's G (global) bit: the mapping is valid for every ASID.
        /// Consumed by the nutlb fill path, which uses it to decide whether
        /// to ASID-qualify the cached tag (docs/nutlb-design.md §3). Threaded
        /// through here rather than re-probed, so filling costs no second
        /// lookup.
        global: bool,
    },

    /// TLB miss - no matching entry found
    /// Contains VPN2 for updating EntryHi
    Miss {
        vpn2: u64,
    },

    /// TLB invalid - entry found but not valid
    /// Contains VPN2 for updating EntryHi
    Invalid {
        vpn2: u64,
    },

    /// TLB modified - write to non-dirty page
    /// Contains VPN2 for updating EntryHi
    Modified {
        vpn2: u64,
    },
}

/// TLB interface for MIPS address translation
///
/// The TLB translates virtual page numbers to physical page numbers.
/// It receives addresses with segment bits already stripped.
pub trait Tlb {
    /// Translate a virtual address to physical address
    ///
    /// # Arguments
    /// * `virt_addr` - Full virtual address (including segment/region bits)
    /// * `asid` - Current Address Space Identifier from EntryHi
    /// * `access_type` - Type of access (Fetch, Read, Write)
    /// * `IS_64BIT` - 0 for 32-bit addressing mode, 1 for 64-bit (xtlb)
    ///
    /// # Returns
    /// TlbResult indicating hit, miss, invalid, or modified
    fn translate<const IS_64BIT: u8>(&mut self, virt_addr: u64, asid: u8, access_type: AccessType) -> TlbResult;

    /// Write a TLB entry at the specified index
    ///
    /// # Arguments
    /// * `index` - TLB index (0..num_entries-1)
    /// * `entry` - TLB entry to write
    fn write(&mut self, index: usize, entry: TlbEntry);

    /// Read a TLB entry at the specified index
    ///
    /// # Arguments
    /// * `index` - TLB index (0..num_entries-1)
    ///
    /// # Returns
    /// The TLB entry at the specified index
    fn read(&self, index: usize) -> TlbEntry;

    /// Probe for a TLB entry matching the given virtual address and ASID
    ///
    /// # Arguments
    /// * `virt_addr` - Virtual address with segment bits already masked off
    /// * `asid` - Address Space Identifier to match
    /// * `is_64bit` - True if CPU is in 64-bit addressing mode
    ///
    /// # Returns
    /// * Index of matching entry (0..num_entries-1), or
    /// * Value with bit 31 set (P bit) if no match found
    fn probe(&self, virt_addr: u64, asid: u8, is_64bit: bool) -> u32;

    /// Get the number of TLB entries
    fn num_entries(&self) -> usize;

    // Debug methods
    fn format_entry(&self, index: usize) -> String;
    fn debug_translate(&self, virt_addr: u64, asid: u8) -> String;

    // State management
    fn power_on(&mut self) {}
    fn save_state(&self) -> toml::Value { toml::Value::Table(Default::default()) }
    fn load_state(&mut self, _v: &toml::Value) -> Result<(), String> { Ok(()) }

    /// Record a nanotlb hit (default no-op; overridden by MipsTlb when tlbstats is on).
    #[cfg(feature = "tlbstats")]
    fn stats_nanotlb_hit(&mut self, _at: AccessType) {}
    /// Record a nutlb hit.
    #[cfg(feature = "tlbstats")]
    #[inline(always)]
    fn stats_nutlb_hit(&mut self, _at: AccessType) {}
    /// Record a nutlb miss.
    #[cfg(feature = "tlbstats")]
    #[inline(always)]
    fn stats_nutlb_miss(&mut self, _at: AccessType) {}
    /// Record a nutlb *conflict* miss: the set was occupied by a different
    /// page. Distinguishes "needs more sets" from cold/capacity misses — the
    /// number that decides whether NUTLB_BITS should grow.
    #[cfg(feature = "tlbstats")]
    #[inline(always)]
    fn stats_nutlb_conflict(&mut self, _at: AccessType) {}
    /// Record a nanotlb miss.
    #[cfg(feature = "tlbstats")]
    fn stats_nanotlb_miss(&mut self, _at: AccessType) {}
    /// Print collected statistics.
    #[cfg(feature = "tlbstats")]
    fn stats_print(&self) {}

    /// Attempt to clone this TLB as a concrete `MipsTlb`.
    /// Returns `None` for implementations that are not `MipsTlb` (e.g. `PassthroughTlb`).
    fn clone_as_mips_tlb(&self) -> Option<MipsTlb> { None }

    /// Restore TLB state from a `MipsTlb` snapshot (used by JIT rollback).
    /// Default no-op for implementations that don't support rollback.
    fn restore_from_mips_tlb(&mut self, _src: &MipsTlb) {}

    /// Run internal consistency checks (feature = "tlbcheck" only).
    /// Default no-op; overridden by `MipsTlb`. Violations are printed via
    /// `eprintln!`; the caller is expected to stop execution (return
    /// `EXEC_BREAKPOINT`) when this returns `true`, so a bogus TLBWI/TLBWR is
    /// caught right where it happens instead of many instructions later at a
    /// confusing IRIX "duplicate TLB entry" panic.
    ///
    /// Returns `true` if any violation was found.
    #[cfg(feature = "tlbcheck")]
    fn consistency_check(&self, _context: &str) -> bool { false }
}

const VMAP_SIZE: usize = 524288; // 4GB / 8KB = 2^19 slots
const VMAP_MISS: u8 = 0xFF;
/// Set on a vmap slot when more than one TLB entry's VPN2 range covers it
/// (aliasing — e.g. two 64-bit VAs sharing bits 13..31 but differing in
/// R-field/upper bits, or genuinely overlapping entries). A slot with this
/// bit set can't be trusted for the O(1) fast path; translate() must fall
/// back to the linear MRU scan. Low 7 bits hold the entry index (0-47) when
/// this bit is set, so it's always ORed onto a real index, never combined
/// with VMAP_MISS (0xFF is checked as a whole byte before masking).
const VMAP_MULTI: u8 = 0x80;
/// Mask to recover the raw entry index from a vmap slot (strips VMAP_MULTI).
const VMAP_IDX_MASK: u8 = 0x7F;

/// Sentinel: end of MRU list / invalid slot index.
const MRU_NONE: u8 = 0xFF;
/// One MRU list per AccessType discriminant (Fetch=0, Read=1, Write=2, Debug=3).
const MRU_LISTS: usize = 4;

/// Cache-friendly shadow of a TLB entry containing only the fields needed for
/// address translation.  All values are pre-decoded so the hot scan touches no
/// architectural TlbEntry data at all.
///
/// Layout is tuned to fit in one 64-byte cache line:
///   vcmp32/vpn_hi32/vcmp64/vpn_hi64 : 32 B
///   pfn_base[2]                       : 16 B
///   offset_mask                       :  8 B
///   valid_dirty[2] + cache_attr[2]    :  4 B  (bit0=valid, bit1=dirty)
///   asid + global + selector_bit_shift:  3 B
///   _pad                              :  1 B
///                                     = 64 B
#[derive(Clone, Copy)]
#[repr(C)]
struct ShadowEntry {
    vcmp32:             u64,
    vpn_hi32:           u64,
    vcmp64:             u64,
    vpn_hi64:           u64,
    pfn_base:           [u64; 2],
    offset_mask:        u64,
    /// Per-page flags: bit 0 = valid, bit 1 = dirty.
    valid_dirty:        [u8; 2],
    cache_attr:         [CacheAttr; 2],
    asid:               u8,
    global:             bool,
    selector_bit_shift: u8,
    _pad:               u8,
}

impl ShadowEntry {
    fn invalid() -> Self {
        Self {
            vcmp32: 0, vpn_hi32: !0,
            vcmp64: 0, vpn_hi64: !0,
            pfn_base: [0; 2],
            offset_mask: 0xFFF,
            valid_dirty: [0; 2],
            cache_attr: [CacheAttr::Uncached; 2],
            asid: 0,
            global: false,
            selector_bit_shift: 12,
            _pad: 0,
        }
    }

    fn from_entry(e: &TlbEntry) -> Self {
        let decode_cache_attr = |lo: u64| match (lo >> 3) & 0x7 {
            2 => CacheAttr::Uncached,
            3 => CacheAttr::Cacheable,
            5 => CacheAttr::CacheableCoherent,
            _ => CacheAttr::Uncached,
        };
        Self {
            vcmp32:             e.vcmp32,
            vpn_hi32:           e.vpn_hi32,
            vcmp64:             e.vcmp64,
            vpn_hi64:           e.vpn_hi64,
            pfn_base:           e.pfn_base,
            offset_mask:        e.offset_mask,
            valid_dirty:        [
                ((e.entry_lo[0] & 0x2 != 0) as u8) | ((e.entry_lo[0] & 0x4 != 0) as u8) << 1,
                ((e.entry_lo[1] & 0x2 != 0) as u8) | ((e.entry_lo[1] & 0x4 != 0) as u8) << 1,
            ],
            cache_attr:         [
                decode_cache_attr(e.entry_lo[0]),
                decode_cache_attr(e.entry_lo[1]),
            ],
            asid:               e.asid(),
            global:             e.is_global(),
            selector_bit_shift: e.selector_bit_shift,
            _pad: 0,
        }
    }
}

/// Real R4000 TLB implementation
///
/// Implements a fully associative JTLB (Joint TLB) with 48 dual-entries.
///
/// **32-bit mode (and 64-bit sign-extended ±2GB) fast path**: a 512KB `vmap`
/// array indexed by VA[31:13] gives O(1) lookup.  Each slot holds the TLB
/// entry index (0-47) or VMAP_MISS.  After the index is found we still verify
/// ASID/Global and the valid/dirty bits — but the linear scan over 48 entries
/// is eliminated.
///
/// For 64-bit VAs that are sign-extended 32-bit values (upper 32 bits all-zero
/// or all-ones), the same vmap applies: we use VA[31:13] as the key.
///
/// Full 64-bit addresses (xuseg, xsseg, etc.) and ASID-aliased 32-bit VAs fall
/// back to an MRU-ordered linear scan over `shadow[]`.  The shadow array holds
/// only translation-relevant fields (pre-decoded, cache-line sized) so the scan
/// never touches the architectural `entries[]` array.
/// Each access type (Fetch/Read/Write/Debug) has its own MRU permutation.
/// Debug lookups never disturb the Fetch/Read/Write ordering.
#[derive(Clone)]
pub struct MipsTlb {
    /// Architectural TLB entries (read/written by TLBR/TLBWI/TLBWR/TLBP).
    entries: [TlbEntry; TLB_NUM_ENTRIES],
    /// Cache-friendly shadow used by translate() and probe().  Kept in sync
    /// with `entries` — rebuilt whenever an entry is written.
    shadow: [ShadowEntry; TLB_NUM_ENTRIES],
    /// Head of each MRU list (slot index, or MRU_NONE).
    mru_head: [u8; MRU_LISTS],
    /// `mru_next[list][slot]` — next slot in that list, or MRU_NONE.
    mru_next: [[u8; TLB_NUM_ENTRIES]; MRU_LISTS],
    /// O(1) lookup for 32-bit (and sign-extended 64-bit) VAs.
    /// Indexed by VA[31:13] (19 bits).  Value = entry index or VMAP_MISS.
    vmap: [u8; VMAP_SIZE],
    /// Every VPN2 slot `vmap_fill_range`/`vmap_clear` has ever touched (across the
    /// whole session, union — never shrinks except on power_on/load_state).
    /// `vmap` starts as `[VMAP_MISS; VMAP_SIZE]` and nothing but those two
    /// functions ever mutates it, so any slot NOT in this set is guaranteed
    /// still VMAP_MISS without needing to read it. Lets
    /// `find_consistency_violations`'s vmap check examine only the slots
    /// that could possibly be wrong instead of walking all 524288 of them on
    /// every single TLBWI/TLBWR (~800us/call measured, dominating boot time).
    #[cfg(feature = "tlbcheck")]
    vmap_touched: std::collections::HashSet<usize>,
    #[cfg(feature = "tlbstats")]
    pub stats: TlbStats,
}

impl MipsTlb {
    pub fn new(num_entries: usize) -> Self {
        assert_eq!(num_entries, TLB_NUM_ENTRIES,
            "MipsTlb currently requires exactly {} entries", TLB_NUM_ENTRIES);
        let mut tlb = Self {
            entries: [TlbEntry::new(); TLB_NUM_ENTRIES],
            shadow: [ShadowEntry::invalid(); TLB_NUM_ENTRIES],
            mru_head: [0u8; MRU_LISTS],
            mru_next: [[MRU_NONE; TLB_NUM_ENTRIES]; MRU_LISTS],
            vmap: [VMAP_MISS; VMAP_SIZE],
            #[cfg(feature = "tlbcheck")]
            vmap_touched: std::collections::HashSet::new(),
            #[cfg(feature = "tlbstats")]
            stats: TlbStats::default(),
        };
        for list in 0..MRU_LISTS {
            tlb.mru_head[list] = 0;
            for i in 0..TLB_NUM_ENTRIES - 1 {
                tlb.mru_next[list][i] = (i + 1) as u8;
            }
            // slot 47 already MRU_NONE from array initialisation
        }
        tlb
    }

    /// Move `target` to the front of `list`.
    /// `prev` is the predecessor of `target` in the list, or MRU_NONE if already head.
    #[inline]
    fn mru_promote(&mut self, list: usize, target: u8, prev: u8) {
        if prev == MRU_NONE {
            return;
        }
        let after = self.mru_next[list][target as usize];
        self.mru_next[list][prev as usize] = after;
        self.mru_next[list][target as usize] = self.mru_head[list];
        self.mru_head[list] = target;
    }

    /// The [start, start+count) VPN2 slot range `entries[entry_idx]` covers.
    /// The vmap is keyed on VA[31:13] only (bits 13..31), so this uses
    /// entry_hi[31:13] regardless of the R field / upper bits — those only
    /// distinguish VAs that alias within this same range, which is exactly
    /// what VMAP_MULTI (and the post-lookup VPN verify in translate()) is for.
    #[inline]
    fn vmap_range(entry: &TlbEntry) -> (usize, usize) {
        let mask = entry.page_mask | 0x1FFF;
        let count = ((mask + 1) >> 13).max(1) as usize;
        // Align down: entry_hi bits [14:13] are offset bits for large pages and may
        // be non-zero (MIPS spec doesn't require software to clear them).
        let start = (((entry.entry_hi as u32) >> 13) as usize) & !(count - 1);
        (start, count)
    }

    /// Unconditionally blank every slot in `[start, start+count)` to VMAP_MISS.
    #[inline]
    fn vmap_clear(&mut self, start: usize, count: usize) {
        #[cfg(feature = "tlbcheck")]
        for i in 0..count {
            let slot = start.wrapping_add(i);
            if slot < VMAP_SIZE { self.vmap_touched.insert(slot); }
        }
        for i in 0..count {
            let slot = start.wrapping_add(i);
            if slot < VMAP_SIZE {
                self.vmap[slot] = VMAP_MISS;
            }
        }
    }

    /// Claim every slot in `[start, start+count)` for `entry_idx`: a slot
    /// currently VMAP_MISS is set to `entry_idx` outright; a slot already
    /// claimed by another entry is set to `entry_idx | VMAP_MULTI` (an
    /// existing VMAP_MULTI slot stays VMAP_MULTI, now naming `entry_idx` as
    /// its "most recent" index — translate() must fall back to the linear
    /// scan for it regardless of which index is recorded).
    #[inline]
    fn vmap_fill_range(&mut self, entry_idx: usize, start: usize, count: usize) {
        let tag = entry_idx as u8;
        #[cfg(feature = "tlbcheck")]
        for i in 0..count {
            let slot = start.wrapping_add(i);
            if slot < VMAP_SIZE { self.vmap_touched.insert(slot); }
        }
        for i in 0..count {
            let slot = start.wrapping_add(i);
            if slot < VMAP_SIZE {
                self.vmap[slot] = if self.vmap[slot] == VMAP_MISS { tag } else { tag | VMAP_MULTI };
            }
        }
    }

    /// Replace the vmap's claim on TLB slot `index` with the entry now
    /// stored there (`write()` has already updated `self.entries[index]`).
    /// Algorithm (see rules/testing/ for the bug this replaces a simpler,
    /// broken erase+fill with): the entry being overwritten may have been
    /// the sole owner of some VPN2 slots that *other* still-valid entries
    /// also cover — naive erase-by-tag left those slots dangling at
    /// VMAP_MISS forever. So:
    ///   1. Compute the range the *old* entry at `index` covered.
    ///   2. Unconditionally clear that whole range (not just slots tagged
    ///      `index` — a slot may already show a different tag from an
    ///      earlier aliasing write, but it's still being vacated here).
    ///   3. Walk every *other* entry and re-fill any of its coverage that
    ///      falls inside the vacated range, so a still-valid aliasing entry
    ///      reclaims its slot instead of being left at VMAP_MISS.
    ///   4. Finally fill the *new* entry's own (possibly different) range —
    ///      unbounded, i.e. its full range regardless of what was vacated —
    ///      so it gets priority (and VMAP_MULTI) over anything restored in
    ///      step 3 that it now also aliases.
    #[inline]
    fn vmap_replace(&mut self, index: usize, old_range: (usize, usize)) {
        let (old_start, old_count) = old_range;
        self.vmap_clear(old_start, old_count);
        let old_end = old_start + old_count;
        for i in 0..TLB_NUM_ENTRIES {
            if i == index {
                continue;
            }
            let (start, count) = Self::vmap_range(&self.entries[i]);
            let end = start + count;
            // Intersect [start, end) with the vacated [old_start, old_end).
            let lo = start.max(old_start);
            let hi = end.min(old_end);
            if lo < hi {
                self.vmap_fill_range(i, lo, hi - lo);
            }
        }
        let (new_start, new_count) = Self::vmap_range(&self.entries[index]);
        self.vmap_fill_range(index, new_start, new_count);
    }

    /// Full internal consistency check (feature = "tlbcheck").
    /// Returns a list of human-readable violation descriptions (empty if the
    /// TLB is consistent).
    ///
    /// Checks, in order:
    /// 1. No two valid entries overlap the same VPN2+region for an
    ///    ASID/global combination that could both match a lookup (this is
    ///    the condition IRIX itself detects and panics on as "duplicate TLB
    ///    entry").
    /// 2. `shadow[i]` is bit-for-bit the re-derivation of `entries[i]`.
    /// 3. Each MRU list is a permutation of all TLB_NUM_ENTRIES slots with
    ///    no cycle and no dropped/duplicated entry.
    /// 4. (feature = "tlbvmap") Every vmap slot either correctly names the
    ///    unique entry covering that VPN2, or is VMAP_MISS if none does.
    #[cfg(feature = "tlbcheck")]
    fn find_consistency_violations(&self) -> Vec<String> {
        let mut violations = Vec::new();
        let mut report = |msg: String| violations.push(msg);

        // 1. Duplicate/overlapping entries.
        // Two entries conflict if their VPN2 ranges intersect (after masking
        // by the wider of the two page masks) and they'd both be visible to
        // the same lookup: either entry is global, or both share an ASID.
        for i in 0..TLB_NUM_ENTRIES {
            let a = &self.entries[i];
            if !a.is_valid_even() && !a.is_valid_odd() {
                continue; // fully-invalid entries can't conflict
            }
            for j in (i + 1)..TLB_NUM_ENTRIES {
                let b = &self.entries[j];
                if !b.is_valid_even() && !b.is_valid_odd() {
                    continue;
                }
                if a.region() != b.region() {
                    continue;
                }
                if !(a.is_global() || b.is_global() || a.asid() == b.asid()) {
                    continue;
                }
                let mask = a.page_mask | b.page_mask | 0x1FFF;
                if (a.entry_hi & !mask) == (b.entry_hi & !mask) {
                    report(format!(
                        "duplicate/overlapping entries {} and {} (VPN2 {:07x} / {:07x}, ASID {:02x}/{:02x}, G {}/{})",
                        i, j, a.vpn2(), b.vpn2(), a.asid(), b.asid(), a.is_global() as u8, b.is_global() as u8
                    ));
                }
            }
        }

        // 2. Shadow vs architectural entries.
        // Skip the VPN/pfn/offset comparison for a page that's invalid on
        // both sides: ShadowEntry::invalid() (used by new()/power_on() for
        // untouched slots) fills vpn_hi32/64 with the sentinel !0 so a VA of
        // 0 never matches, while ShadowEntry::from_entry(&TlbEntry::new())
        // (an all-zero architectural entry) derives 0 for the same fields.
        // Both are correctly unmatchable — the bit patterns just differ by
        // construction — so comparing them here would be a false positive.
        for i in 0..TLB_NUM_ENTRIES {
            let expected = ShadowEntry::from_entry(&self.entries[i]);
            let s = &self.shadow[i];
            if s.valid_dirty != expected.valid_dirty || s.asid != expected.asid
                || s.global != expected.global || s.selector_bit_shift != expected.selector_bit_shift {
                report(format!("shadow[{}] does not match entries[{}] (valid/asid/global/shift)", i, i));
                continue;
            }
            let page_dead = expected.valid_dirty[0] & 0x1 == 0 && expected.valid_dirty[1] & 0x1 == 0;
            if page_dead {
                continue;
            }
            let mismatch = s.vcmp32 != expected.vcmp32
                || s.vpn_hi32 != expected.vpn_hi32
                || s.vcmp64 != expected.vcmp64
                || s.vpn_hi64 != expected.vpn_hi64
                || s.pfn_base != expected.pfn_base
                || s.offset_mask != expected.offset_mask;
            if mismatch {
                report(format!("shadow[{}] does not match entries[{}] (vpn/pfn/offset)", i, i));
            }
        }

        // 3. MRU list integrity: each list must visit every slot exactly once.
        for list in 0..MRU_LISTS {
            let mut seen = [false; TLB_NUM_ENTRIES];
            let mut cur = self.mru_head[list];
            let mut count = 0usize;
            while cur != MRU_NONE {
                let slot = cur as usize;
                if slot >= TLB_NUM_ENTRIES {
                    report(format!("MRU list {} contains out-of-range slot {}", list, slot));
                    break;
                }
                if seen[slot] {
                    report(format!("MRU list {} has a cycle at slot {}", list, slot));
                    break;
                }
                seen[slot] = true;
                count += 1;
                cur = self.mru_next[list][slot];
            }
            let missing: Vec<usize> = (0..TLB_NUM_ENTRIES).filter(|&i| !seen[i]).collect();
            if !missing.is_empty() {
                report(format!("MRU list {} is missing slot(s) {:?} (visited {} of {})", list, missing, count, TLB_NUM_ENTRIES));
            }
        }

        // 4. vmap vs full scan (only meaningful with tlbvmap, but the vmap
        // array and vmap_replace are unconditional in this codebase, so we
        // can always check it).
        // Note: vmap_replace runs unconditionally on every write(),
        // independent of the V bit — an entry written with V=0 still claims
        // its VPN2 slots (translate()'s vmap fast path relies on this to
        // return Invalid rather than falling through to Miss). Entries never
        // touched by write() still hold TlbEntry::new()'s all-zero default
        // and so "cover" VPN2=0 in this re-derivation too, exactly matching
        // vmap_replace's real behavior — see its doc comment: a never-written
        // entry claiming VPN2=0 just costs one VMAP_MULTI fallback-to-scan
        // until every entry is written, which happens almost immediately in
        // practice, so all TLB_NUM_ENTRIES entries are considered here
        // unconditionally, with no separate "was this written" tracking.
        //
        // Iterates TLB_NUM_ENTRIES entries (each covering `count` VPN2
        // slots) instead of the naive VMAP_SIZE(524288) x
        // TLB_NUM_ENTRIES(48) double loop — the naive version made every
        // TLBWI/TLBWR cost ~25M iterations, dominating emulated boot time by
        // orders of magnitude. `self.vmap_touched` (maintained by
        // vmap_clear/vmap_fill_range) is the set of every slot that could
        // possibly be non-default — nothing outside it needs checking, since
        // `vmap` starts at `[VMAP_MISS; VMAP_SIZE]` and only those two
        // functions ever mutate it.
        #[cfg(feature = "tlbvmap")]
        {
            use std::collections::HashMap;
            let mut owners: HashMap<usize, (usize, bool)> = HashMap::new(); // vpn2 -> (most_recent_entry_idx, multiple)
            for i in 0..TLB_NUM_ENTRIES {
                let (start, count) = Self::vmap_range(&self.entries[i]);
                for k in 0..count {
                    let vpn2 = start.wrapping_add(k);
                    if vpn2 >= VMAP_SIZE {
                        continue;
                    }
                    owners.entry(vpn2)
                        .and_modify(|(owner, multiple)| { *owner = i; *multiple = true; })
                        .or_insert((i, false));
                }
            }
            // Every slot that could possibly be non-VMAP_MISS is in
            // vmap_touched; check each against its expected owner (or lack
            // thereof) in one pass instead of walking the full vmap array.
            for &vpn2 in &self.vmap_touched {
                let slot_val = self.vmap[vpn2];
                match owners.get(&vpn2) {
                    None => {
                        if slot_val != VMAP_MISS {
                            report(format!("vmap[{:05x}] = {:#04x} but no entry covers this VPN2", vpn2, slot_val));
                        }
                    }
                    Some(&(_owner, multiple)) => {
                        let slot_multi = slot_val & VMAP_MULTI != 0;
                        if multiple {
                            if !slot_multi {
                                report(format!("vmap[{:05x}] = {:#04x} (no VMAP_MULTI) but multiple entries cover this VPN2", vpn2, slot_val));
                            }
                        } else {
                            // Exactly one covering entry: VMAP_MULTI must be
                            // clear and the index must be that entry. (A stale
                            // VMAP_MULTI left over from an entry that moved
                            // away is exactly the class of bug this check
                            // exists to catch.)
                            if slot_multi {
                                report(format!("vmap[{:05x}] = {:#04x} has stale VMAP_MULTI but only one entry covers this VPN2", vpn2, slot_val));
                            } else if slot_val as usize != _owner {
                                report(format!("vmap[{:05x}] = {} but entry {} covers this VPN2", vpn2, slot_val, _owner));
                            }
                        }
                    }
                }
            }
        }

        violations
    }

    /// Run `find_consistency_violations` and print any findings, tagged with
    /// `context` (e.g. "TLBWI idx=3 pc=...") so the offending write is easy
    /// to find in logs. Returns `true` if any violation was found, so the
    /// caller can stop execution right at the offending TLBWI/TLBWR instead
    /// of running on to a confusing IRIX "duplicate TLB entry" panic many
    /// instructions later.
    #[cfg(feature = "tlbcheck")]
    fn do_consistency_check(&self, context: &str) -> bool {
        let violations = self.find_consistency_violations();
        if violations.is_empty() {
            return false;
        }
        for msg in &violations {
            eprintln!("TLBCHECK VIOLATION [{}]: {}", context, msg);
        }
        eprintln!("TLBCHECK: dumping full TLB state for [{}]:", context);
        for i in 0..TLB_NUM_ENTRIES {
            eprintln!("{}", self.format_entry(i));
        }
        true
    }
}

impl Default for MipsTlb {
    fn default() -> Self {
        Self::new(TLB_NUM_ENTRIES)
    }
}

impl Tlb for MipsTlb {
    #[inline]
    fn translate<const IS_64BIT: u8>(&mut self, virt_addr: u64, asid: u8, access_type: AccessType) -> TlbResult {
        #[cfg(feature = "tlbstats")]
        if IS_64BIT == 0 {
            self.stats.by_type[access_type as usize].addr32 += 1;
        } else {
            self.stats.by_type[access_type as usize].addr64 += 1;
        }

        // Fast path: O(1) vmap lookup, keyed on VA bits 13..31 regardless of
        // IS_64BIT — two VAs sharing those bits but differing in R-field /
        // upper bits (true 64-bit aliasing) can share a vmap slot; that's
        // resolved below either by VMAP_MULTI forcing the linear scan, or by
        // the explicit vcmp/vpn_hi verify against the candidate entry.
        //
        // Gated so the vmap can actually be switched off for measurement
        // (`--no-default-features`). Previously `tlbvmap` gated only
        // tlbcheck's verification of the vmap, not the vmap itself, so
        // building without it changed nothing on this path — which made the
        // question "is the vmap still earning its keep now that nutlb
        // absorbs most lookups?" unanswerable. Falling through to the MRU
        // scan below is always correct, just slower.
        #[cfg(feature = "tlbvmap")]
        {
            let vmap_idx = ((virt_addr as u32) >> 13) as usize;
            let slot = self.vmap[vmap_idx];
            if slot != VMAP_MISS {
                // Always try the recorded candidate first, multi or not —
                // when several entries alias this VPN2 slot the recorded
                // index is merely "most recent, not guaranteed to match,"
                // but if it DOES match there's no need to pay for the linear
                // scan.
                let multi = slot & VMAP_MULTI != 0;
                let entry_idx = slot & VMAP_IDX_MASK;
                let s = &self.shadow[entry_idx as usize];
                let (vcmp, vpn_hi) = if IS_64BIT != 0 {
                    (s.vcmp64, s.vpn_hi64)
                } else {
                    (s.vcmp32, s.vpn_hi32)
                };

                // Verify the candidate entry actually matches this VA (not
                // just bits 13..31 — the vmap key).
                let vpn_match = (virt_addr & vcmp) == vpn_hi;
                if vpn_match && (s.global || s.asid == asid) {
                    let idx = ((virt_addr >> s.selector_bit_shift) & 1) as usize;
                    let vd = s.valid_dirty[idx];

                    if vd & 0x1 == 0 {
                        #[cfg(feature = "tlbstats")]
                        { self.stats.by_type[access_type as usize].vmap_invalid += 1; }
                        return TlbResult::Invalid { vpn2: virt_addr >> 13 };
                    }

                    let dirty = vd & 0x2 != 0;
                    if access_type == AccessType::Write && !dirty {
                        #[cfg(feature = "tlbstats")]
                        { self.stats.by_type[access_type as usize].vmap_modified += 1; }
                        return TlbResult::Modified { vpn2: virt_addr >> 13 };
                    }

                    #[cfg(feature = "tlbstats")]
                    { self.stats.by_type[access_type as usize].vmap_hit += 1; }
                    return TlbResult::Hit {
                        phys_addr:  s.pfn_base[idx] | (virt_addr & s.offset_mask),
                        cache_attr: s.cache_attr[idx],
                        dirty,
                        global:     s.global,
                    };
                }
                // Mismatch (VPN or ASID/global). If nothing else aliases
                // this VPN2 slot (!multi), this candidate was the *only*
                // entry in the whole TLB covering VA bits 13..31 at all — if
                // it doesn't match (wrong VPN, or right VPN but wrong ASID
                // and not global), no other entry could possibly match
                // either, since a second claimant on this slot (of any ASID)
                // would have set VMAP_MULTI when it was written. Definite
                // miss, no need to pay for the scan. With multi set, another
                // aliasing candidate wasn't even checked here — only the
                // scan can tell if one of them matches.
                if !multi {
                    #[cfg(feature = "tlbstats")]
                    { self.stats.by_type[access_type as usize].vmap_miss += 1; }
                    return TlbResult::Miss { vpn2: virt_addr >> 13 };
                }
                #[cfg(feature = "tlbstats")]
                { self.stats.by_type[access_type as usize].vmap_asid_fall += 1; }
            } else {
                // vmap says no entry covers VA bits 13..31 at all — definite miss.
                #[cfg(feature = "tlbstats")]
                { self.stats.by_type[access_type as usize].vmap_miss += 1; }
                return TlbResult::Miss { vpn2: virt_addr >> 13 };
            }
        }

        // MRU-ordered fallback scan for full 64-bit VAs or ASID-aliased 32-bit VAs.
        let list = access_type as usize;

        #[cfg(feature = "tlbstats")]
        { self.stats.by_type[list].scan_invocations += 1; }

        let mut prev: u8 = MRU_NONE;
        let mut cur: u8  = self.mru_head[list];

        while cur != MRU_NONE {
            let slot = cur as usize;
            let next = self.mru_next[list][slot];
            let s = self.shadow[slot]; // Copy — releases borrow before mru_promote

            #[cfg(feature = "tlbstats")]
            { self.stats.by_type[list].scan_entries_walked += 1; }

            let (vcmp, vpn_hi) = if IS_64BIT != 0 {
                (s.vcmp64, s.vpn_hi64)
            } else {
                (s.vcmp32, s.vpn_hi32)
            };
            if (virt_addr & vcmp) != vpn_hi {
                prev = cur; cur = next; continue;
            }
            if !s.global && s.asid != asid {
                prev = cur; cur = next; continue;
            }

            let idx = ((virt_addr >> s.selector_bit_shift) & 1) as usize;
            let vd = s.valid_dirty[idx];

            if vd & 0x1 == 0 {
                #[cfg(feature = "tlbstats")]
                { self.stats.by_type[list].scan_invalid += 1; }
                self.mru_promote(list, cur, prev);
                return TlbResult::Invalid { vpn2: virt_addr >> 13 };
            }

            let dirty = vd & 0x2 != 0;
            if access_type == AccessType::Write && !dirty {
                #[cfg(feature = "tlbstats")]
                { self.stats.by_type[list].scan_modified += 1; }
                self.mru_promote(list, cur, prev);
                return TlbResult::Modified { vpn2: virt_addr >> 13 };
            }

            #[cfg(feature = "tlbstats")]
            { self.stats.by_type[list].scan_hit += 1; }
            self.mru_promote(list, cur, prev);
            return TlbResult::Hit {
                phys_addr:  s.pfn_base[idx] | (virt_addr & s.offset_mask),
                cache_attr: s.cache_attr[idx],
                dirty,
                global:     s.global,
            };
        }

        #[cfg(feature = "tlbstats")]
        { self.stats.by_type[list].scan_miss += 1; }
        TlbResult::Miss { vpn2: virt_addr >> 13 }
    }

    fn write(&mut self, index: usize, mut entry: TlbEntry) {
        if index < self.entries.len() {
            let mask = entry.page_mask | 0x1FFF;
            entry.selector_bit_shift = (mask.trailing_ones() - 1) as u8;
            entry.vcmp32   = !mask & 0x0000_0000_FFFF_E000;
            entry.vpn_hi32 = entry.entry_hi & entry.vcmp32;
            entry.vcmp64   = !mask & 0xC000_00FF_FFFF_E000;
            entry.vpn_hi64 = entry.entry_hi & entry.vcmp64;
            entry.offset_mask = (1u64 << entry.selector_bit_shift) - 1;
            let pfn_mask = 0xFF_FFFF_FFFF_000 & !entry.offset_mask;
            entry.pfn_base = [
                (entry.entry_lo[0] << 6) & pfn_mask,
                (entry.entry_lo[1] << 6) & pfn_mask,
            ];
            let old_range = Self::vmap_range(&self.entries[index]);
            self.entries[index] = entry;
            self.shadow[index] = ShadowEntry::from_entry(&entry);
            self.vmap_replace(index, old_range);
        }
    }

    fn read(&self, index: usize) -> TlbEntry {
        if index < self.entries.len() {
            self.entries[index]
        } else {
            TlbEntry::new()
        }
    }

    fn probe(&self, virt_addr: u64, asid: u8, is_64bit: bool) -> u32 {
        for (i, s) in self.shadow.iter().enumerate() {
            // 1. Check VPN match first (per MIPS R4000 manual flowchart)
            // Apply same masking logic as translate()
            // Per MIPS R4000 manual:
            // - 32-bit mode: Compare highest 7-19 bits (depending on page size) of VA to TLB VPN
            // - 64-bit mode: Compare highest 15-27 bits (depending on page size) of VA to TLB VPN
            //   plus R field in bits 63:62
            let (vcmp, vpn_hi) = if is_64bit {
                // In 64-bit mode, include R field (bits 63:62) and VPN (bits 39:13)
                (s.vcmp64, s.vpn_hi64)
            } else {
                // In 32-bit mode, only compare VPN in bits 31:13 (and below based on page size)
                (s.vcmp32, s.vpn_hi32)
            };
            if (virt_addr & vcmp) != vpn_hi {
                continue;
            }

            // 2. Check Global bit or ASID match (after VPN match per R4000 manual flowchart)
            if !s.global && s.asid != asid {
                continue;
            }

            return i as u32;
        }
        0x80000000 // Not found (P bit set)
    }

    fn num_entries(&self) -> usize {
        self.entries.len()
    }

    fn format_entry(&self, index: usize) -> String {
        if index >= self.entries.len() {
            return format!("Index {} out of bounds", index);
        }
        let e = &self.entries[index];
        let vpn2 = e.vpn2();
        let asid = e.asid();
        let mask = e.page_mask;
        let region = e.region();

        // Calculate full 64-bit address from VPN2 and region
        // VPN2 is bits 39:13, region is bits 63:62
        let full_vpn2_addr = ((region as u64) << 62) | (vpn2 << 13);

        // Format: [Index] R=... VPN2=... (addr=...) ASID=... Mask=...
        //         Even: PFN=... C=... D=... V=... G=...
        //         Odd:  PFN=... C=... D=... V=... G=...

        let fmt_lo = |lo: u64| {
            let pfn = (lo >> 6) & 0xFF_FFFF_FFFF;
            let c = (lo >> 3) & 0x7;
            let d = (lo & 0x4) != 0;
            let v = (lo & 0x2) != 0;
            let g = (lo & 0x1) != 0;
            format!("PFN={:014x} C={} D={} V={} G={}", pfn, c, d as u8, v as u8, g as u8)
        };

        format!("[{:02}] R={} VPN2={:07x} (addr={:016x}) ASID={:02x} Mask={:016x}\n      Even: {}\n      Odd:  {}",
            index, region, vpn2, full_vpn2_addr, asid, mask, fmt_lo(e.entry_lo[0]), fmt_lo(e.entry_lo[1]))
    }

    fn debug_translate(&self, virt_addr: u64, asid: u8) -> String {
        let mut output = String::new();
        writeln!(output, "Translating VA={:016x} ASID={:02x}", virt_addr, asid).unwrap();

        for (i, entry) in self.entries.iter().enumerate() {
            // 1. Check Global bit or ASID match
            if !entry.is_global() && entry.asid() != asid {
                continue;
            }

            // 2. Check VPN match
            let mask = entry.page_mask | 0x1FFF;
            let vpn_compare_mask = !mask;

            if (virt_addr & vpn_compare_mask) == (entry.entry_hi & vpn_compare_mask) {
                writeln!(output, "Match found at Index {}", i).unwrap();
                writeln!(output, "{}", self.format_entry(i)).unwrap();

                let is_odd = ((virt_addr >> entry.selector_bit_shift) & 1) != 0;
                writeln!(output, "Selected page: {}", if is_odd { "Odd" } else { "Even" }).unwrap();

                let lo_entry = entry.entry_lo[is_odd as usize];

                if (lo_entry & 0x2) == 0 {
                    writeln!(output, "Result: Invalid (V=0)").unwrap();
                } else {
                    let phys_addr = entry.pfn_base[is_odd as usize]
                                  | (virt_addr & entry.offset_mask);
                    let c = (lo_entry >> 3) & 0x7;
                    let d = (lo_entry & 0x4) != 0;

                    writeln!(output, "Result: PhysAddr={:016x} CacheAttr={} Dirty={}", phys_addr, c, d).unwrap();
                }
                return output;
            }
        }
        writeln!(output, "No match found (TLB Miss)").unwrap();
        output
    }

    fn power_on(&mut self) {
        self.entries = [TlbEntry::new(); TLB_NUM_ENTRIES];
        self.shadow  = [ShadowEntry::invalid(); TLB_NUM_ENTRIES];
        for list in 0..MRU_LISTS {
            self.mru_head[list] = 0;
            for i in 0..TLB_NUM_ENTRIES - 1 {
                self.mru_next[list][i] = (i + 1) as u8;
            }
            self.mru_next[list][TLB_NUM_ENTRIES - 1] = MRU_NONE;
        }
        self.vmap.fill(VMAP_MISS);
        #[cfg(feature = "tlbcheck")]
        { self.vmap_touched.clear(); }
    }

    fn save_state(&self) -> toml::Value {
        // Each entry stored as [page_mask, entry_hi, entry_lo0, entry_lo1]
        let arr: Vec<toml::Value> = self.entries.iter().map(|e| {
            let words = [e.page_mask, e.entry_hi, e.entry_lo[0], e.entry_lo[1]];
            u64_slice_to_toml(&words)
        }).collect();
        toml::Value::Array(arr)
    }

    fn load_state(&mut self, v: &toml::Value) -> Result<(), String> {
        if let toml::Value::Array(arr) = v {
            for (i, item) in arr.iter().enumerate() {
                if i >= TLB_NUM_ENTRIES { break; }
                let mut words = [0u64; 4];
                load_u64_slice(item, &mut words);
                let page_mask = words[0];
                let entry_hi  = words[1];
                let mask = page_mask | 0x1FFF;
                let vcmp32 = !mask & 0x0000_0000_FFFF_E000;
                let vcmp64 = !mask & 0xC000_00FF_FFFF_E000;
                let shift = (mask.trailing_ones() - 1) as u8;
                let offset_mask = (1u64 << shift) - 1;
                let lo = [words[2], words[3]];
                let pfn_mask = 0xFF_FFFF_FFFF_000 & !offset_mask;
                self.entries[i] = TlbEntry {
                    page_mask,
                    entry_hi,
                    entry_lo: lo,
                    selector_bit_shift: shift,
                    vcmp32,
                    vpn_hi32: entry_hi & vcmp32,
                    vcmp64,
                    vpn_hi64: entry_hi & vcmp64,
                    offset_mask,
                    pfn_base: [(lo[0] << 6) & pfn_mask, (lo[1] << 6) & pfn_mask],
                };
            }
        }
        for i in 0..TLB_NUM_ENTRIES {
            self.shadow[i] = ShadowEntry::from_entry(&self.entries[i]);
        }
        // Reset MRU lists to canonical order so snapshot restores are deterministic.
        for list in 0..MRU_LISTS {
            self.mru_head[list] = 0;
            for i in 0..TLB_NUM_ENTRIES - 1 {
                self.mru_next[list][i] = (i + 1) as u8;
            }
            self.mru_next[list][TLB_NUM_ENTRIES - 1] = MRU_NONE;
        }
        self.vmap.fill(VMAP_MISS);
        for i in 0..TLB_NUM_ENTRIES {
            let (start, count) = Self::vmap_range(&self.entries[i]);
            self.vmap_fill_range(i, start, count);
        }
        Ok(())
    }

    #[cfg(feature = "tlbstats")]
    fn stats_nutlb_hit(&mut self, at: AccessType) {
        self.stats.by_type[at as usize].nutlb_hit += 1;
    }
    #[cfg(feature = "tlbstats")]
    fn stats_nutlb_miss(&mut self, at: AccessType) {
        self.stats.by_type[at as usize].nutlb_miss += 1;
    }
    #[cfg(feature = "tlbstats")]
    fn stats_nutlb_conflict(&mut self, at: AccessType) {
        self.stats.by_type[at as usize].nutlb_conflict += 1;
    }
    #[cfg(feature = "tlbstats")]
    fn stats_nanotlb_hit(&mut self, at: AccessType) {
        self.stats.by_type[at as usize].nano_hit += 1;
    }
    #[cfg(feature = "tlbstats")]
    fn stats_nanotlb_miss(&mut self, at: AccessType) {
        self.stats.by_type[at as usize].nano_miss += 1;
    }
    #[cfg(feature = "tlbstats")]
    fn stats_print(&self) {
        self.stats.print();
    }

    fn clone_as_mips_tlb(&self) -> Option<MipsTlb> { Some(self.clone()) }

    fn restore_from_mips_tlb(&mut self, src: &MipsTlb) { *self = src.clone(); }

    #[cfg(feature = "tlbcheck")]
    fn consistency_check(&self, context: &str) -> bool {
        self.do_consistency_check(context)
    }
}

/// Passthrough TLB implementation for testing
///
/// This implementation performs simple identity mapping for low addresses
/// and returns TLB miss for everything else. Useful for testing without
/// a full TLB implementation.
pub struct PassthroughTlb {
    /// Maximum address to identity-map (addresses below this are mapped 1:1)
    max_identity_addr: u64,
}

impl PassthroughTlb {
    /// Create a new passthrough TLB
    ///
    /// # Arguments
    /// * `max_identity_addr` - Maximum address for identity mapping (default: 0x20000000 / 512MB)
    pub fn new(max_identity_addr: u64) -> Self {
        Self { max_identity_addr }
    }

    /// Create a passthrough TLB with default settings (512MB identity mapping for Indy)
    pub fn default() -> Self {
        Self::new(0x20000000) // 512MB identity mapping (Indy physical address space)
    }
}

impl Tlb for PassthroughTlb {
    fn translate<const IS_64BIT: u8>(&mut self, virt_addr: u64, _asid: u8, _access_type: AccessType) -> TlbResult {
        // Identity map addresses below max_identity_addr.
        // We mask the address to 29 bits (512MB) to simulate physical offset behavior,
        // allowing it to work with both masked and full virtual addresses.
        let masked_addr = virt_addr & 0x1FFFFFFF;

        if virt_addr < self.max_identity_addr {
            TlbResult::Hit {
                phys_addr: masked_addr,
                cache_attr: CacheAttr::Uncached,
                dirty: true, // All pages are writable in passthrough mode
                // Identity mapping has no ASID concept — valid for all.
                global: true,
            }
        } else {
            let vpn2 = virt_addr >> 13;
            TlbResult::Miss { vpn2 }
        }
    }

    fn write(&mut self, _index: usize, _entry: TlbEntry) {
        // Passthrough TLB ignores writes
    }

    fn read(&self, _index: usize) -> TlbEntry {
        // Return empty entry
        TlbEntry::new()
    }

    fn probe(&self, _virt_addr: u64, _asid: u8, _is_64bit: bool) -> u32 {
        // Always return "not found" (P bit set)
        0x80000000
    }

    fn num_entries(&self) -> usize {
        0 // Passthrough has no real entries
    }

    fn format_entry(&self, _index: usize) -> String {
        "Passthrough TLB (No entries)".to_string()
    }

    fn debug_translate(&self, virt_addr: u64, _asid: u8) -> String {
        format!("Passthrough: {:016x} -> {:016x}", virt_addr, virt_addr & 0x1FFFFFFF)
    }
}

#[cfg(test)]
#[path = "mips_tlb_test.rs"]
mod tests;
