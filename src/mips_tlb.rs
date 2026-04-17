// MIPS TLB (Translation Lookaside Buffer) interface and implementations

use crate::mips_exec::CacheAttr;
use std::fmt::Write;
use crate::snapshot::{u64_slice_to_toml, load_u64_slice};

/// Number of TLB entries in R4000 (48 dual-entries = 96 pages)
pub const TLB_NUM_ENTRIES: usize = 48;

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

    /// Entry Lo0 - even page mapping
    /// Bit 63:6 (or 29:6 for 32-bit PFN): PFN (Page Frame Number)
    /// Bits 5:3: Cache coherency attribute (C)
    /// Bit 2: Dirty (D) - writable if set
    /// Bit 1: Valid (V) - entry is valid if set
    /// Bit 0: Global (G) - ignore ASID if set
    /// 64-bit register, bits used depend on addressing mode
    pub entry_lo0: u64,

    /// Entry Lo1 - odd page mapping
    /// Same format as entry_lo0
    /// 64-bit register, bits used depend on addressing mode
    pub entry_lo1: u64,
}

impl TlbEntry {
    /// Create a new empty/invalid TLB entry
    pub fn new() -> Self {
        Self {
            page_mask: 0,
            entry_hi: 0,
            entry_lo0: 0,
            entry_lo1: 0,
        }
    }

    /// Check if this entry is valid for the even page
    #[inline]
    pub fn is_valid_even(&self) -> bool {
        (self.entry_lo0 & 0x2) != 0
    }

    /// Check if this entry is valid for the odd page
    #[inline]
    pub fn is_valid_odd(&self) -> bool {
        (self.entry_lo1 & 0x2) != 0
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
    /// * `is_64bit` - True if CPU is in 64-bit addressing mode
    ///
    /// # Returns
    /// TlbResult indicating hit, miss, invalid, or modified
    fn translate(&mut self, virt_addr: u64, asid: u8, access_type: AccessType, is_64bit: bool) -> TlbResult;

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

    /// Attempt to clone this TLB as a concrete `MipsTlb`.
    /// Returns `None` for implementations that are not `MipsTlb` (e.g. `PassthroughTlb`).
    fn clone_as_mips_tlb(&self) -> Option<MipsTlb> { None }

    /// Restore TLB state from a `MipsTlb` snapshot (used by JIT rollback).
    /// Default no-op for implementations that don't support rollback.
    fn restore_from_mips_tlb(&mut self, _src: &MipsTlb) {}
}

/// Sentinel: end of MRU list / invalid vmap entry.
const MRU_NONE: u8 = 0xFF;
/// One MRU list per AccessType discriminant (Fetch=0, Read=1, Write=2, Debug=3).
const MRU_LISTS: usize = 4;

/// Number of slots in the 32-bit vmap: 4GB / 8KB = 524288.
/// Each slot stores a TLB entry index (0..47) or VMAP_MISS (0xFF).
const VMAP_SIZE: usize = 524288; // 2^19
const VMAP_MISS: u8 = 0xFF;

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
/// Full 64-bit addresses (xuseg, xsseg, etc.) fall back to the MRU-ordered
/// linear scan.
///
/// Each access type (Fetch/Read/Write/Debug) has its own MRU-ordered
/// permutation of the 48 slots for the fallback path.
/// Debug lookups never disturb the Fetch/Read/Write ordering.
#[derive(Clone)]
pub struct MipsTlb {
    entries: [TlbEntry; TLB_NUM_ENTRIES],
    /// Head of each MRU list (slot index, or MRU_NONE).
    mru_head: [u8; MRU_LISTS],
    /// `mru_next[list][slot]` — next slot in that list, or MRU_NONE.
    mru_next: [[u8; TLB_NUM_ENTRIES]; MRU_LISTS],
    /// O(1) lookup for 32-bit (and sign-extended 64-bit) VAs.
    /// Indexed by VA[31:13] (19 bits).  Value = entry index or VMAP_MISS.
    vmap: Box<[u8; VMAP_SIZE]>,
}

impl MipsTlb {
    pub fn new(num_entries: usize) -> Self {
        assert_eq!(num_entries, TLB_NUM_ENTRIES,
            "MipsTlb currently requires exactly {} entries", TLB_NUM_ENTRIES);
        let mut tlb = Self {
            entries: [TlbEntry::new(); TLB_NUM_ENTRIES],
            mru_head: [0u8; MRU_LISTS],
            mru_next: [[MRU_NONE; TLB_NUM_ENTRIES]; MRU_LISTS],
            vmap: Box::new([VMAP_MISS; VMAP_SIZE]),
        };
        // Initialise all four lists as 0 → 1 → … → 47 → NONE.
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
    /// `prev` is the predecessor of `target` in the list,
    /// or MRU_NONE when `target` is already the head.
    #[inline]
    fn mru_promote(&mut self, list: usize, target: u8, prev: u8) {
        if prev == MRU_NONE {
            return; // already at front, nothing to do
        }
        // Splice out.
        let after = self.mru_next[list][target as usize];
        self.mru_next[list][prev as usize] = after;
        // Insert at head.
        self.mru_next[list][target as usize] = self.mru_head[list];
        self.mru_head[list] = target;
    }

    /// Erase all vmap slots that currently point to `entry_idx`.
    /// Called before overwriting a TLB entry.
    /// The vmap is keyed on VA[31:13] only, so we use entry_hi[31:13] regardless
    /// of the R field / upper bits (those only matter in full 64-bit mode, which
    /// bypasses the vmap entirely).
    #[inline]
    fn vmap_erase(&mut self, entry_idx: usize) {
        let old = &self.entries[entry_idx];
        let tag = entry_idx as u8;
        let mask = old.page_mask | 0x1FFF;
        let count = ((mask + 1) >> 13).max(1) as usize;
        let vpn2 = ((old.entry_hi as u32) >> 13) as usize; // VA[31:13] of even page
        for i in 0..count {
            let slot = vpn2.wrapping_add(i);
            if slot < VMAP_SIZE && self.vmap[slot] == tag {
                self.vmap[slot] = VMAP_MISS;
            }
        }
    }

    /// Populate vmap slots for `entry_idx` using the entry now stored at that index.
    /// Always uses entry_hi[31:13] as the key — the upper bits (R field) are only
    /// relevant for full 64-bit VAs which skip the vmap anyway.
    #[inline]
    fn vmap_fill(&mut self, entry_idx: usize) {
        let entry = &self.entries[entry_idx];
        let mask = entry.page_mask | 0x1FFF;
        let count = ((mask + 1) >> 13).max(1) as usize;
        let vpn2 = ((entry.entry_hi as u32) >> 13) as usize;
        let tag = entry_idx as u8;
        for i in 0..count {
            let slot = vpn2.wrapping_add(i);
            if slot < VMAP_SIZE {
                self.vmap[slot] = tag;
            }
        }
    }
}

impl Default for MipsTlb {
    fn default() -> Self {
        Self::new(TLB_NUM_ENTRIES)
    }
}

impl Tlb for MipsTlb {
    fn translate(&mut self, virt_addr: u64, asid: u8, access_type: AccessType, is_64bit: bool) -> TlbResult {
        // Fast path: O(1) vmap lookup for 32-bit VAs and 64-bit sign-extended ±2GB VAs.
        // A 64-bit VA is sign-extended 32-bit when upper 32 bits are all-zero (user/kuseg)
        // or all-ones (kernel kseg0/kseg1/kseg2/kseg3 in 64-bit compatibility mode).
        let upper32 = (virt_addr >> 32) as u32;
        if !is_64bit || upper32 == 0 || upper32 == 0xFFFF_FFFF {
            let vmap_idx = ((virt_addr as u32) >> 13) as usize;
            let entry_idx = self.vmap[vmap_idx];
            if entry_idx != VMAP_MISS {
                let entry = &self.entries[entry_idx as usize];

                // Verify ASID / Global match.
                // On mismatch we must fall through to linear scan: a different entry
                // for the same VPN but a different ASID may exist (TLB aliasing).
                if entry.is_global() || entry.asid() == asid {
                    let mask = entry.page_mask | 0x1FFF;
                    let selector_bit = (mask + 1) >> 1;
                    let lo_entry = if (virt_addr & selector_bit) != 0 { entry.entry_lo1 } else { entry.entry_lo0 };

                    // Valid bit.
                    if (lo_entry & 0x2) == 0 {
                        return TlbResult::Invalid { vpn2: virt_addr >> 13 };
                    }

                    // Dirty bit for writes.
                    let dirty = (lo_entry & 0x4) != 0;
                    if access_type == AccessType::Write && !dirty {
                        return TlbResult::Modified { vpn2: virt_addr >> 13 };
                    }

                    // Physical address.
                    let pfn = (lo_entry >> 6) & 0xFF_FFFF_FFFF;
                    let offset_mask = selector_bit - 1;
                    let effective_pfn = pfn & !(offset_mask >> 12);
                    let phys_addr = (effective_pfn << 12) | (virt_addr & offset_mask);

                    let cache_attr = match (lo_entry >> 3) & 0x7 {
                        2 => CacheAttr::Uncached,
                        3 => CacheAttr::Cacheable,
                        5 => CacheAttr::CacheableCoherent,
                        _ => CacheAttr::Uncached,
                    };

                    return TlbResult::Hit { phys_addr, cache_attr, dirty };
                }
                // ASID mismatch on a non-global entry — fall through to linear scan
                // to check if another entry exists for this VA+ASID combination.
            } else {
                // vmap says no entry for this VA — definite miss.
                return TlbResult::Miss { vpn2: virt_addr >> 13 };
            }
        }

        // Slow path: full 64-bit VA (or ASID-aliased 32-bit VA), MRU linear scan.
        let list = access_type as usize;
        let mode_mask: u64 = if is_64bit {
            0xC000_00FF_FFFF_E000
        } else {
            0x0000_0000_FFFF_E000
        };

        let mut prev: u8 = MRU_NONE;
        let mut cur: u8  = self.mru_head[list];

        while cur != MRU_NONE {
            let idx = cur as usize;
            let next = self.mru_next[list][idx];
            let entry = &self.entries[idx];

            // 1. VPN match
            let mask = entry.page_mask | 0x1FFF;
            let vpn_compare_mask = !mask & mode_mask;
            if (virt_addr & vpn_compare_mask) != (entry.entry_hi & vpn_compare_mask) {
                prev = cur;
                cur  = next;
                continue;
            }

            // 2. Global bit or ASID match
            if !entry.is_global() && entry.asid() != asid {
                prev = cur;
                cur  = next;
                continue;
            }

            // 3. Select Even (Lo0) or Odd (Lo1) entry
            let selector_bit = (mask + 1) >> 1;
            let lo_entry = if (virt_addr & selector_bit) != 0 { entry.entry_lo1 } else { entry.entry_lo0 };

            // 4. Valid bit
            if (lo_entry & 0x2) == 0 {
                self.mru_promote(list, cur, prev);
                return TlbResult::Invalid { vpn2: virt_addr >> 13 };
            }

            // 5. Dirty bit for writes
            let dirty = (lo_entry & 0x4) != 0;
            if access_type == AccessType::Write && !dirty {
                self.mru_promote(list, cur, prev);
                return TlbResult::Modified { vpn2: virt_addr >> 13 };
            }

            // 6. Physical address
            let pfn = (lo_entry >> 6) & 0xFF_FFFF_FFFF;
            let offset_mask = selector_bit - 1;
            let effective_pfn = pfn & !(offset_mask >> 12);
            let phys_addr = (effective_pfn << 12) | (virt_addr & offset_mask);

            // 7. Cache attribute
            let cache_attr = match (lo_entry >> 3) & 0x7 {
                2 => CacheAttr::Uncached,
                3 => CacheAttr::Cacheable,
                5 => CacheAttr::CacheableCoherent,
                _ => CacheAttr::Uncached,
            };

            // Promote to front of this access type's MRU list.
            self.mru_promote(list, cur, prev);

            return TlbResult::Hit { phys_addr, cache_attr, dirty };
        }

        TlbResult::Miss { vpn2: virt_addr >> 13 }
    }

    fn write(&mut self, index: usize, entry: TlbEntry) {
        if index < self.entries.len() {
            self.vmap_erase(index);
            self.entries[index] = entry;
            self.vmap_fill(index);
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
        for (i, entry) in self.entries.iter().enumerate() {
            // 1. Check VPN match first (per MIPS R4000 manual flowchart)
            let mask = entry.page_mask | 0x1FFF;
            let mut vpn_compare_mask = !mask;

            // Apply same masking logic as translate()
            // Per MIPS R4000 manual:
            // - 32-bit mode: Compare highest 7-19 bits (depending on page size) of VA to TLB VPN
            // - 64-bit mode: Compare highest 15-27 bits (depending on page size) of VA to TLB VPN
            //   plus R field in bits 63:62
            if is_64bit {
                // In 64-bit mode, include R field (bits 63:62) and VPN (bits 39:13)
                vpn_compare_mask &= 0xC000_00FF_FFFF_E000;
            } else {
                // In 32-bit mode, only compare VPN in bits 31:13 (and below based on page size)
                vpn_compare_mask &= 0xFFFF_E000;
            }

            if (virt_addr & vpn_compare_mask) != (entry.entry_hi & vpn_compare_mask) {
                continue;
            }

            // 2. Check Global bit or ASID match (after VPN match per R4000 manual flowchart)
            if !entry.is_global() && entry.asid() != asid {
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
            index, region, vpn2, full_vpn2_addr, asid, mask, fmt_lo(e.entry_lo0), fmt_lo(e.entry_lo1))
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

                let selector_bit = (mask + 1) >> 1;
                let is_odd = (virt_addr & selector_bit) != 0;
                writeln!(output, "Selected page: {}", if is_odd { "Odd" } else { "Even" }).unwrap();

                let lo_entry = if is_odd { entry.entry_lo1 } else { entry.entry_lo0 };

                if (lo_entry & 0x2) == 0 {
                    writeln!(output, "Result: Invalid (V=0)").unwrap();
                } else {
                    let pfn = (lo_entry >> 6) & 0xFF_FFFF_FFFF;
                    let offset_mask = selector_bit - 1;
                    let pfn_mask = !(offset_mask >> 12);
                    let effective_pfn = pfn & pfn_mask;
                    let phys_addr = (effective_pfn << 12) | (virt_addr & offset_mask);
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
        // Reset all MRU lists to initial order 0 → 1 → … → 47 → NONE.
        for list in 0..MRU_LISTS {
            self.mru_head[list] = 0;
            for i in 0..TLB_NUM_ENTRIES - 1 {
                self.mru_next[list][i] = (i + 1) as u8;
            }
            self.mru_next[list][TLB_NUM_ENTRIES - 1] = MRU_NONE;
        }
        self.vmap.fill(VMAP_MISS);
    }

    fn save_state(&self) -> toml::Value {
        // Each entry stored as [page_mask, entry_hi, entry_lo0, entry_lo1]
        let arr: Vec<toml::Value> = self.entries.iter().map(|e| {
            let words = [e.page_mask, e.entry_hi, e.entry_lo0, e.entry_lo1];
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
                self.entries[i] = TlbEntry {
                    page_mask: words[0],
                    entry_hi:  words[1],
                    entry_lo0: words[2],
                    entry_lo1: words[3],
                };
            }
        }
        // Rebuild vmap from loaded entries.
        self.vmap.fill(VMAP_MISS);
        for i in 0..TLB_NUM_ENTRIES {
            self.vmap_fill(i);
        }
        Ok(())
    }

    fn clone_as_mips_tlb(&self) -> Option<MipsTlb> { Some(self.clone()) }

    fn restore_from_mips_tlb(&mut self, src: &MipsTlb) { *self = src.clone(); }
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
    fn translate(&mut self, virt_addr: u64, _asid: u8, _access_type: AccessType, _is_64bit: bool) -> TlbResult {
        // Identity map addresses below max_identity_addr.
        // We mask the address to 29 bits (512MB) to simulate physical offset behavior,
        // allowing it to work with both masked and full virtual addresses.
        let masked_addr = virt_addr & 0x1FFFFFFF;

        if virt_addr < self.max_identity_addr {
            TlbResult::Hit {
                phys_addr: masked_addr,
                cache_attr: CacheAttr::Uncached,
                dirty: true, // All pages are writable in passthrough mode
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
