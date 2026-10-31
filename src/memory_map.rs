//! Where a machine has memory: its RAM, what aliases RAM, and its boot ROM.
//!
//! Each machine builds one of these when it is constructed, from its own
//! address map and configuration, and hands it to whatever needs to tell
//! memory from everything else: the CPU's undo buffer, the JIT's page table,
//! and the monitor. That code then works on every machine without knowing
//! where any of them keep their RAM.
//!
//! A range says where the memory controller can answer, which is not always
//! what is mapped at this moment. The firmware programs the controller at boot,
//! and until it does, or in a bank it has not filled, an address in a range may
//! have nothing behind it. Asking the bus remains the way to find out.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemKind {
    Ram,
    /// RAM again at a second address, with the same contents.
    Alias,
    Rom,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemRange {
    pub name: &'static str,
    pub kind: MemKind,
    pub base: u64,
    pub size: u64,
}

impl MemRange {
    pub fn new(name: &'static str, kind: MemKind, base: u64, end: u64) -> Self {
        Self { name, kind, base, size: end - base }
    }

    pub fn end(&self) -> u64 {
        self.base + self.size
    }

    pub fn contains(&self, addr: u64) -> bool {
        (self.base..self.end()).contains(&addr)
    }
}

impl fmt::Display for MemRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {:#x}..{:#x}", self.name, self.base, self.end())
    }
}

/// A machine's memory ranges, lowest address first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryMap {
    ranges: Vec<MemRange>,
}

impl MemoryMap {
    /// Panics on overlapping ranges: each machine's list is written by hand,
    /// and an address has to belong to one range.
    pub fn new(mut ranges: Vec<MemRange>) -> Self {
        ranges.sort_by_key(|r| r.base);
        for pair in ranges.windows(2) {
            assert!(pair[0].end() <= pair[1].base, "{} overlaps {}", pair[0], pair[1]);
        }
        Self { ranges }
    }

    pub fn ranges(&self) -> &[MemRange] {
        &self.ranges
    }

    pub fn find(&self, addr: u64) -> Option<&MemRange> {
        self.ranges.iter().find(|r| r.contains(addr))
    }

    pub fn contains(&self, addr: u64) -> bool {
        self.find(addr).is_some()
    }

    /// True for RAM and for its aliases.
    pub fn is_ram(&self, addr: u64) -> bool {
        self.find(addr).is_some_and(|r| r.kind != MemKind::Rom)
    }

    /// The address just past the highest range.
    pub fn end(&self) -> u64 {
        self.ranges.last().map_or(0, MemRange::end)
    }
}

impl fmt::Display for MemoryMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, r) in self.ranges.iter().enumerate() {
            if i != 0 { f.write_str(", ")?; }
            write!(f, "{r}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1 << 20;

    #[test]
    fn indy_has_two_windows_a_low_alias_and_a_prom() {
        let m = crate::ip22_bus::memory_map(false);
        assert!(m.is_ram(0x1000), "the low alias is RAM");
        assert!(!m.contains(0x0008_0000), "the alias stops at 512 KB");
        assert!(m.is_ram(0x0800_0000));
        assert!(m.is_ram(0x17ff_ffff));
        assert!(!m.contains(0x1800_0000));
        assert_eq!(m.find(0x1fc0_0000).unwrap().kind, MemKind::Rom);
        assert!(!m.is_ram(0x1fc0_0000), "the PROM is memory but not RAM");
        assert!(m.is_ram(0x2fff_ffff));
        assert_eq!(m.end(), 0x3000_0000);
    }

    #[test]
    fn ip28_has_no_lomem() {
        let m = crate::ip22_bus::memory_map(true);
        assert!(!m.contains(0x0800_0000));
        assert!(m.is_ram(0x2000_0000));
        assert!(m.is_ram(0x9fff_ffff), "four 512 MB banks fit");
    }

    #[test]
    fn o2_memory_is_a_gigabyte_at_its_non_zero_base() {
        let m = crate::ip32::memory_map();
        assert!(m.is_ram(0x0fff_ffff), "the low alias");
        assert!(!m.contains(0x1000_0000));
        assert!(m.is_ram(0x4000_0000));
        assert!(m.is_ram(0x7fff_ffff));
        assert!(!m.contains(0x8000_0000), "the CPU folds the no-ECC alias before the bus");
    }

    #[test]
    fn octane_ram_follows_the_configured_banks() {
        let full = crate::ip30::memory_map(&[2048, 2048, 2048, 2048]);
        assert!(full.is_ram(0x2000_0000));
        assert!(full.is_ram(0x2_1fff_ffff), "the last byte of 8 GB");
        assert!(!full.contains(0x2_2000_0000));
        assert_eq!(full.end(), 0x2_2000_0000);

        let small = crate::ip30::memory_map(&[64, 0, 0, 0]);
        assert_eq!(small.end(), 0x2000_0000 + 64 * MB);
        assert!(small.is_ram(0x100), "low memory");
        assert!(!small.contains(0x4000), "low memory is 16 KB");
    }

    #[test]
    fn octane_range_holds_mixed_banks_however_the_prom_aligns_them() {
        let m = crate::ip30::memory_map(&[64, 2048, 0, 0]);
        assert_eq!(m.end(), 0x2000_0000 + 4096 * MB, "64 MB, a hole to 2 GB, then 2 GB");
    }

    #[test]
    #[should_panic(expected = "overlaps")]
    fn overlapping_ranges_are_refused() {
        MemoryMap::new(vec![
            MemRange::new("a", MemKind::Ram, 0, 2 * MB),
            MemRange::new("b", MemKind::Ram, MB, 3 * MB),
        ]);
    }
}
