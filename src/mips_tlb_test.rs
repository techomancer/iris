#[cfg(test)]
mod tests {
    use super::*;
    use crate::mips_tlb::*;
    use crate::mips_exec::CacheAttr;

    #[test]
    fn test_tlb_entry_flags() {
        let mut entry = TlbEntry::new();

        // Test invalid entry
        assert!(!entry.is_valid_even());
        assert!(!entry.is_valid_odd());
        assert!(!entry.is_global());

        // Set valid bit on even page
        entry.entry_lo[0] = 0x2; // V bit
        assert!(entry.is_valid_even());
        assert!(!entry.is_valid_odd());

        // Set valid bit on odd page
        entry.entry_lo[1] = 0x2; // V bit
        assert!(entry.is_valid_odd());

        // Set global bit (stored in EntryHi bit 12 per MIPS R4000 spec)
        entry.entry_hi = 0x1000; // G bit in bit 12
        assert!(entry.is_global());
    }

    #[test]
    fn test_tlb_entry_asid_vpn() {
        let mut entry = TlbEntry::new();

        // Set ASID to 42
        entry.entry_hi = 42;
        assert_eq!(entry.asid(), 42);

        // Set VPN2 to 0x1234
        entry.entry_hi = (0x1234 << 13) | 42;
        assert_eq!(entry.vpn2(), 0x1234);
        assert_eq!(entry.asid(), 42);
    }

    #[test]
    fn test_passthrough_tlb() {
        let mut tlb = PassthroughTlb::new(0x20000000); // 512MB

        // Test identity mapping for low addresses
        match tlb.translate::<0>(0x1000, 0, AccessType::Read) {
            TlbResult::Hit { phys_addr, cache_attr, dirty, .. } => {
                assert_eq!(phys_addr, 0x1000);
                assert_eq!(cache_attr, CacheAttr::Uncached);
                assert!(dirty);
            }
            _ => panic!("Expected TLB hit for low address"),
        }

        // Test identity mapping at upper boundary
        match tlb.translate::<0>(0x1FFFFFFF, 0, AccessType::Read) {
            TlbResult::Hit { phys_addr, cache_attr, dirty, .. } => {
                assert_eq!(phys_addr, 0x1FFFFFFF);
                assert_eq!(cache_attr, CacheAttr::Uncached);
                assert!(dirty);
            }
            _ => panic!("Expected TLB hit for address at 512MB boundary"),
        }

        // Test TLB miss for addresses beyond 512MB
        match tlb.translate::<0>(0x20000000, 0, AccessType::Read) {
            TlbResult::Miss { vpn2 } => {
                assert_eq!(vpn2, 0x20000000 >> 13); // VPN2 should be address >> 13
            }
            _ => panic!("Expected TLB miss for address beyond 512MB"),
        }

        // Test probe always returns not found
        assert_eq!(tlb.probe(0x1000, 0, false), 0x80000000);
    }

    #[test]
    fn test_real_tlb_translation() {
        let mut tlb = MipsTlb::new(TLB_NUM_ENTRIES);
        let asid = 10;

        // Setup a TLB entry
        // VPN2: 0x100 (Virtual Address 0x00200000)
        // PageMask: 0 (4KB)
        // ASID: 10
        let mut entry = TlbEntry::new();
        entry.page_mask = 0;
        entry.entry_hi = (0x100 << 13) | (asid as u64);

        // Even Page (Lo0): PFN 0x50, Cacheable, Dirty, Valid
        // Maps 0x00200000 -> 0x00050000
        entry.entry_lo[0] = (0x50 << 6) | (3 << 3) | (1 << 2) | (1 << 1);

        // Odd Page (Lo1): PFN 0x51, Uncached, Not Dirty, Valid
        // Maps 0x00201000 -> 0x00051000
        entry.entry_lo[1] = (0x51 << 6) | (2 << 3) | (0 << 2) | (1 << 1);

        // Write to index 5
        tlb.write(5, entry);

        // Test 1: Even page translation
        let va_even = 0x00200000;
        match tlb.translate::<0>(va_even, asid, AccessType::Read) {
            TlbResult::Hit { phys_addr, cache_attr, dirty, .. } => {
                assert_eq!(phys_addr, 0x50000);
                assert_eq!(cache_attr, CacheAttr::Cacheable);
                assert!(dirty);
            }
            res => panic!("Expected Hit for even page, got {:?}", res),
        }

        // Test 2: Odd page translation
        let va_odd = 0x00201000;
        match tlb.translate::<0>(va_odd, asid, AccessType::Read) {
            TlbResult::Hit { phys_addr, cache_attr, dirty, .. } => {
                assert_eq!(phys_addr, 0x51000);
                assert_eq!(cache_attr, CacheAttr::Uncached);
                assert!(!dirty);
            }
            res => panic!("Expected Hit for odd page, got {:?}", res),
        }

        // Test 3: TLB Miss (Unmapped address)
        let va_miss = 0x00300000;
        match tlb.translate::<0>(va_miss, asid, AccessType::Read) {
            TlbResult::Miss { vpn2 } => {
                assert_eq!(vpn2, va_miss >> 13);
            }
            res => panic!("Expected Miss for unmapped address, got {:?}", res),
        }

        // Test 4: TLB Miss (Wrong ASID)
        match tlb.translate::<0>(va_even, asid + 1, AccessType::Read) {
            TlbResult::Miss { vpn2 } => {
                assert_eq!(vpn2, va_even >> 13);
            }
            res => panic!("Expected Miss for wrong ASID, got {:?}", res),
        }
    }

    #[test]
    fn test_real_tlb_global_page() {
        let mut tlb = MipsTlb::new(TLB_NUM_ENTRIES);
        let entry_asid = 10;
        let request_asid = 20; // Different ASID

        // Setup a Global TLB entry
        // VPN2: 0x200 (Virtual Address 0x00400000)
        // Per MIPS R4000 spec: G bit is stored in EntryHi bit 12 in TLB entries
        let mut entry = TlbEntry::new();
        entry.page_mask = 0;
        entry.entry_hi = (0x200 << 13) | (entry_asid as u64) | 0x1000; // G bit in bit 12

        // Even Page (Lo0): PFN 0x60, Cacheable, Dirty, Valid
        // Maps 0x00400000 -> 0x00060000
        entry.entry_lo[0] = (0x60 << 6) | (3 << 3) | (1 << 2) | (1 << 1);

        // Odd Page (Lo1): PFN 0x61, Cacheable, Dirty, Valid
        // Maps 0x00401000 -> 0x00061000
        entry.entry_lo[1] = (0x61 << 6) | (3 << 3) | (1 << 2) | (1 << 1);

        // Write to index 10
        tlb.write(10, entry);

        // Test: Translation should succeed despite ASID mismatch because it is Global
        let va = 0x00400000;
        match tlb.translate::<0>(va, request_asid, AccessType::Read) {
            TlbResult::Hit { phys_addr, .. } => {
                assert_eq!(phys_addr, 0x60000);
            }
            res => panic!("Expected Hit for global page with different ASID, got {:?}", res),
        }
    }

    #[test]
    fn test_tlb_entry_region_field() {
        let mut entry = TlbEntry::new();

        // Test region field extraction (64-bit mode)
        // Set R=2 (bits 63:62), VPN2=0x1234 (bits 39:13), ASID=42 (bits 7:0)
        entry.entry_hi = (2u64 << 62) | (0x1234 << 13) | 42;
        assert_eq!(entry.region(), 2);
        assert_eq!(entry.vpn2(), 0x1234);
        assert_eq!(entry.asid(), 42);

        // Test R=3
        entry.entry_hi = (3u64 << 62) | (0x5678 << 13) | 10;
        assert_eq!(entry.region(), 3);
        assert_eq!(entry.vpn2(), 0x5678);
        assert_eq!(entry.asid(), 10);
    }

    #[test]
    fn test_tlb_entry_field_masking() {
        let mut entry = TlbEntry::new();

        // Test EntryLo PFN is 24 bits (bits 29:6)
        // Set PFN to max value (0xFFFFFF), C=3, D=1, V=1, G=1
        entry.entry_lo[0] = (0xFFFFFF << 6) | (3 << 3) | 0x7;
        assert_eq!(entry.entry_lo[0] & 0x3FFFFFFF, entry.entry_lo[0]);

        // Verify PFN extraction
        let pfn = (entry.entry_lo[0] >> 6) & 0xFFFFFF;
        assert_eq!(pfn, 0xFFFFFF);

        // Test PageMask is only bits 24:13
        entry.page_mask = 0xFFFFFFFF_FFFFFFFF;
        assert_eq!(entry.page_mask & 0x01FFE000, 0x01FFE000);
    }

    #[test]
    fn test_tlb_32bit_vs_64bit_mode() {
        let mut tlb = MipsTlb::new(TLB_NUM_ENTRIES);
        let asid = 10;

        // Setup a TLB entry with 64-bit EntryHi
        // R=2 (bits 63:62), VPN2=0x100 (bits 39:13), ASID=10
        let mut entry = TlbEntry::new();
        entry.page_mask = 0;
        entry.entry_hi = (2u64 << 62) | (0x100 << 13) | (asid as u64);
        entry.entry_lo[0] = (0x50 << 6) | (3 << 3) | 0x6; // PFN 0x50, Cacheable, Dirty, Valid
        entry.entry_lo[1] = (0x51 << 6) | (2 << 3) | 0x2; // PFN 0x51, Uncached, Valid

        tlb.write(5, entry);

        // Test 32-bit mode translation - should match on bits 31:13 only
        let va_32bit = 0x00200000; // VPN2=0x100 in bits 31:13
        match tlb.translate::<0>(va_32bit, asid, AccessType::Read) {
            TlbResult::Hit { phys_addr, .. } => {
                assert_eq!(phys_addr, 0x50000);
            }
            res => panic!("Expected Hit in 32-bit mode, got {:?}", res),
        }

        // Test 64-bit mode translation - should match R field and extended VPN2
        let va_64bit = (2u64 << 62) | 0x00200000; // R=2, VPN2=0x100
        match tlb.translate::<1>(va_64bit, asid, AccessType::Read) {
            TlbResult::Hit { phys_addr, .. } => {
                assert_eq!(phys_addr, 0x50000);
            }
            res => panic!("Expected Hit in 64-bit mode, got {:?}", res),
        }

        // Test 64-bit mode with wrong R field - should miss
        let va_wrong_r = (1u64 << 62) | 0x00200000; // R=1 instead of R=2
        match tlb.translate::<1>(va_wrong_r, asid, AccessType::Read) {
            TlbResult::Miss { .. } => {
                // Expected - R field doesn't match
            }
            res => panic!("Expected Miss for wrong R field, got {:?}", res),
        }
    }

    #[test]
    fn test_tlb_probe_32bit_vs_64bit() {
        let mut tlb = MipsTlb::new(TLB_NUM_ENTRIES);
        let asid = 10;

        // Setup a TLB entry with 64-bit EntryHi
        // R=2 (bits 63:62), VPN2=0x100 (bits 39:13), ASID=10
        let mut entry = TlbEntry::new();
        entry.page_mask = 0;
        entry.entry_hi = (2u64 << 62) | (0x100 << 13) | (asid as u64);
        entry.entry_lo[0] = (0x50 << 6) | (3 << 3) | 0x6;

        tlb.write(5, entry);

        // Test 32-bit mode probe - should match on bits 31:13 only
        let va_32bit = 0x00200000; // VPN2=0x100 in bits 31:13
        let result = tlb.probe(va_32bit, asid, false);
        assert_eq!(result, 5, "Expected probe to find entry at index 5 in 32-bit mode");

        // Test 64-bit mode probe - should match R field and extended VPN2
        let va_64bit = (2u64 << 62) | 0x00200000; // R=2, VPN2=0x100
        let result = tlb.probe(va_64bit, asid, true);
        assert_eq!(result, 5, "Expected probe to find entry at index 5 in 64-bit mode");

        // Test 64-bit mode probe with wrong R field - should miss
        let va_wrong_r = (1u64 << 62) | 0x00200000; // R=1 instead of R=2
        let result = tlb.probe(va_wrong_r, asid, true);
        assert_eq!(result & 0x80000000, 0x80000000, "Expected probe to miss with wrong R field");
    }

    /// Phase 1.7 round-trip: a fresh TLB loaded from a captured save_state must
    /// re-serialize byte-identically. Catches load_state forgetting a field
    /// that save_state writes.
    #[test]
    fn save_load_round_trip() {
        // MipsTlb contains a 512KB vmap array; two instances exceed the default
        // test thread stack (8KB on Linux), so run the body in a larger thread.
        std::thread::Builder::new()
            .stack_size(4 * 1024 * 1024)
            .spawn(|| {
                let mut src = MipsTlb::new(TLB_NUM_ENTRIES);
                for (slot, vpn2) in [(0usize, 0x100u64), (5, 0x800), (17, 0x4000), (47, 0xffff)].iter().copied() {
                    let mut e = TlbEntry::new();
                    e.page_mask = (slot as u64) << 13;
                    e.entry_hi  = (2u64 << 62) | (vpn2 << 13) | (slot as u64 & 0xff);
                    e.entry_lo[0] = ((slot as u64) << 6) | (3 << 3) | 0x6;
                    e.entry_lo[1] = ((slot as u64 + 1) << 6) | (3 << 3) | 0x6;
                    src.write(slot, e);
                }
                let v1 = src.save_state();

                let mut dst = MipsTlb::new(TLB_NUM_ENTRIES);
                dst.load_state(&v1).expect("load_state");
                let v2 = dst.save_state();

                assert_eq!(v1, v2, "MipsTlb save_state mismatch after load_state round-trip");
            })
            .expect("spawn")
            .join()
            .expect("thread panicked");
    }

    #[cfg(feature = "tlbcheck")]
    #[test]
    fn tlbcheck_clean_tlb_passes() {
        std::thread::Builder::new()
            .stack_size(4 * 1024 * 1024)
            .spawn(|| {
                let mut tlb = MipsTlb::new(TLB_NUM_ENTRIES);
                for slot in 0..TLB_NUM_ENTRIES {
                    let mut e = TlbEntry::new();
                    e.entry_hi = ((slot as u64) << 13) | 3; // distinct VPN2 per slot, ASID=3
                    e.entry_lo[0] = (0x100 << 6) | (3 << 3) | 0x6; // V|D
                    e.entry_lo[1] = (0x101 << 6) | (3 << 3) | 0x6;
                    tlb.write(slot, e);
                }
                let violations = tlb.find_consistency_violations();
                assert!(violations.is_empty(), "expected no violations on a clean TLB, got: {:?}", violations);
            })
            .expect("spawn")
            .join()
            .expect("thread panicked");
    }

    #[cfg(feature = "tlbcheck")]
    #[test]
    fn tlbcheck_detects_duplicate_entries() {
        std::thread::Builder::new()
            .stack_size(4 * 1024 * 1024)
            .spawn(|| {
                let mut tlb = MipsTlb::new(TLB_NUM_ENTRIES);
                let mut e = TlbEntry::new();
                e.entry_hi = (0x100u64 << 13) | 3; // ASID=3
                e.entry_lo[0] = (0x50 << 6) | (3 << 3) | 0x6;
                e.entry_lo[1] = (0x51 << 6) | (3 << 3) | 0x6;
                tlb.write(1, e);
                tlb.write(2, e); // same VPN2+ASID written to a second slot
                let violations = tlb.find_consistency_violations();
                assert!(
                    violations.iter().any(|v| v.contains("duplicate/overlapping entries 1 and 2")),
                    "expected a duplicate-entry violation, got: {:?}", violations
                );
            })
            .expect("spawn")
            .join()
            .expect("thread panicked");
    }

    #[cfg(feature = "tlbcheck")]
    #[test]
    fn tlbcheck_detects_corrupted_mru() {
        std::thread::Builder::new()
            .stack_size(4 * 1024 * 1024)
            .spawn(|| {
                let mut tlb = MipsTlb::new(TLB_NUM_ENTRIES);
                // Break list 0 (Fetch): point slot 0's successor at itself,
                // forming a cycle that drops the rest of the list.
                tlb.mru_next[0][0] = 0;
                let violations = tlb.find_consistency_violations();
                assert!(
                    violations.iter().any(|v| v.contains("MRU list 0") && v.contains("cycle")),
                    "expected an MRU cycle violation, got: {:?}", violations
                );
            })
            .expect("spawn")
            .join()
            .expect("thread panicked");
    }

    #[cfg(all(feature = "tlbcheck", feature = "tlbvmap"))]
    #[test]
    fn tlbcheck_detects_corrupted_vmap() {
        std::thread::Builder::new()
            .stack_size(4 * 1024 * 1024)
            .spawn(|| {
                let mut tlb = MipsTlb::new(TLB_NUM_ENTRIES);
                let mut e = TlbEntry::new();
                e.entry_hi = (0x200u64 << 13) | 3;
                e.entry_lo[0] = (0x50 << 6) | (3 << 3) | 0x6;
                tlb.write(4, e);
                // Corrupt the vmap slot for this VPN2 to point at the wrong entry.
                tlb.vmap[0x200] = 9;
                let violations = tlb.find_consistency_violations();
                assert!(
                    violations.iter().any(|v| v.contains("vmap[00200]")),
                    "expected a vmap-mismatch violation, got: {:?}", violations
                );
            })
            .expect("spawn")
            .join()
            .expect("thread panicked");
    }

    // Regression test for a false positive found live during early PROM boot:
    // a "clear all TLB entries" loop that writes indices 0..N-1 (V=0, whatever
    // stale EntryHi happens to be sitting around) but never touches the last
    // index leaves that slot holding TlbEntry::new()'s all-zero default,
    // which "covers" VPN2=0 by coincidence — but was never vmap_fill()'d, so
    // it has no real vmap claim. Before `written` tracking was added, check 4
    // treated it as owning vmap[0] anyway and flagged a phantom mismatch.
    #[cfg(all(feature = "tlbcheck", feature = "tlbvmap"))]
    #[test]
    fn tlbcheck_untouched_entry_is_not_a_vmap_owner() {
        std::thread::Builder::new()
            .stack_size(4 * 1024 * 1024)
            .spawn(|| {
                let mut tlb = MipsTlb::new(TLB_NUM_ENTRIES);
                // Simulate PROM's clear loop: write indices 0..TLB_NUM_ENTRIES-1
                // with a fixed, unrelated VPN2 and V=0, but leave the very last
                // entry (index 47) untouched at its all-zero power-on default.
                for slot in 0..TLB_NUM_ENTRIES - 1 {
                    let mut e = TlbEntry::new();
                    e.entry_hi = 0x7fc0000u64 << 13; // stale reset-garbage VPN2, V=0 throughout
                    tlb.write(slot, e);
                }
                let violations = tlb.find_consistency_violations();
                assert!(
                    violations.is_empty(),
                    "untouched entry 47 (all-zero default) must not be treated as owning vmap[0]: {:?}",
                    violations
                );
            })
            .expect("spawn")
            .join()
            .expect("thread panicked");
    }
}
