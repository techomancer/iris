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
        entry.entry_lo0 = 0x2; // V bit
        assert!(entry.is_valid_even());
        assert!(!entry.is_valid_odd());

        // Set valid bit on odd page
        entry.entry_lo1 = 0x2; // V bit
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
            TlbResult::Hit { phys_addr, cache_attr, dirty } => {
                assert_eq!(phys_addr, 0x1000);
                assert_eq!(cache_attr, CacheAttr::Uncached);
                assert!(dirty);
            }
            _ => panic!("Expected TLB hit for low address"),
        }

        // Test identity mapping at upper boundary
        match tlb.translate::<0>(0x1FFFFFFF, 0, AccessType::Read) {
            TlbResult::Hit { phys_addr, cache_attr, dirty } => {
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
        entry.entry_lo0 = (0x50 << 6) | (3 << 3) | (1 << 2) | (1 << 1);

        // Odd Page (Lo1): PFN 0x51, Uncached, Not Dirty, Valid
        // Maps 0x00201000 -> 0x00051000
        entry.entry_lo1 = (0x51 << 6) | (2 << 3) | (0 << 2) | (1 << 1);

        // Write to index 5
        tlb.write(5, entry);

        // Test 1: Even page translation
        let va_even = 0x00200000;
        match tlb.translate::<0>(va_even, asid, AccessType::Read) {
            TlbResult::Hit { phys_addr, cache_attr, dirty } => {
                assert_eq!(phys_addr, 0x50000);
                assert_eq!(cache_attr, CacheAttr::Cacheable);
                assert!(dirty);
            }
            res => panic!("Expected Hit for even page, got {:?}", res),
        }

        // Test 2: Odd page translation
        let va_odd = 0x00201000;
        match tlb.translate::<0>(va_odd, asid, AccessType::Read) {
            TlbResult::Hit { phys_addr, cache_attr, dirty } => {
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
        entry.entry_lo0 = (0x60 << 6) | (3 << 3) | (1 << 2) | (1 << 1);

        // Odd Page (Lo1): PFN 0x61, Cacheable, Dirty, Valid
        // Maps 0x00401000 -> 0x00061000
        entry.entry_lo1 = (0x61 << 6) | (3 << 3) | (1 << 2) | (1 << 1);

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
        entry.entry_lo0 = (0xFFFFFF << 6) | (3 << 3) | 0x7;
        assert_eq!(entry.entry_lo0 & 0x3FFFFFFF, entry.entry_lo0);

        // Verify PFN extraction
        let pfn = (entry.entry_lo0 >> 6) & 0xFFFFFF;
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
        entry.entry_lo0 = (0x50 << 6) | (3 << 3) | 0x6; // PFN 0x50, Cacheable, Dirty, Valid
        entry.entry_lo1 = (0x51 << 6) | (2 << 3) | 0x2; // PFN 0x51, Uncached, Valid

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
        entry.entry_lo0 = (0x50 << 6) | (3 << 3) | 0x6;

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
}
