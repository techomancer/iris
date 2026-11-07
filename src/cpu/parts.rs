//! The processors SGI shipped, one const each. A [`Part`] is a CPU as `hinv`
//! reports it, and everything iris knows about a part is here and nowhere
//! else: no machine keeps its own PRId, clock or revision. A machine lists the
//! parts it shipped with, the default first, as its own `CPUS`.
//!
//! The `_EARLY` suffix marks earlier silicon of the same family and clock; it
//! exists only where two shipped parts would otherwise share a name.

/// A processor family, numbered as it names the parts (`r10000_195`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    R4400 = 4400,
    R5000 = 5000,
    R10000 = 10000,
    R12000 = 12000,
    R14000 = 14000,
}

impl Family {
    /// PRId bits 15:8 (R10000 UM, PRId register).
    const fn implementation(self) -> u32 {
        match self {
            Self::R4400 => 0x04,
            Self::R5000 => 0x23,
            Self::R10000 => 0x09,
            Self::R12000 => 0x0e,
            Self::R14000 => 0x0f,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Part {
    pub family: Family,
    pub mhz: u32,
    /// Major and minor revision, as `hinv` prints them ("Revision: 2.7").
    pub rev: (u8, u8),
    /// Secondary cache in KB. The R5000 reads 0: its cache is not modelled.
    pub l2_kb: u32,
}

pub const R4400_150: Part = Part { family: Family::R4400, mhz: 150, rev: (5, 0), l2_kb: 1024 };
pub const R5000_180: Part = Part { family: Family::R5000, mhz: 180, rev: (2, 1), l2_kb: 0 };
pub const R5000_180_EARLY: Part = Part { rev: (1, 0), ..R5000_180 };
pub const R10000_195: Part = Part { family: Family::R10000, mhz: 195, rev: (2, 7), l2_kb: 1024 };
pub const R10000_195_EARLY: Part = Part { rev: (2, 5), ..R10000_195 };
pub const R10000_250: Part = Part { mhz: 250, rev: (3, 4), ..R10000_195 };
pub const R12000_360: Part = Part { family: Family::R12000, mhz: 360, rev: (3, 5), l2_kb: 2048 };
pub const R12000_400: Part = Part { mhz: 400, ..R12000_360 };
pub const R14000_600: Part = Part { family: Family::R14000, mhz: 600, rev: (2, 4), l2_kb: 2048 };

/// Every part, for the bare-metal bench rig, which takes any. A name matches
/// the first listed, so the rig runs the later silicon of a clock that has two.
pub const ALL: &[&Part] = &[
    &R4400_150, &R5000_180_EARLY, &R5000_180, &R10000_195, &R10000_195_EARLY, &R10000_250,
    &R12000_360, &R12000_400, &R14000_600,
];

impl Part {
    /// Implementation, then major and minor revision.
    pub const fn prid(&self) -> u32 {
        self.family.implementation() << 8 | (self.rev.0 as u32) << 4 | self.rev.1 as u32
    }

    /// CP0 Count ticks at half the pipeline clock.
    pub const fn count_hz(&self) -> u64 {
        self.mhz as u64 * 500_000
    }

    pub fn is_r10000_family(&self) -> bool {
        !matches!(self.family, Family::R4400 | Family::R5000)
    }

    /// `r10000_195`, as `cpu = ` takes it.
    pub fn name(&self) -> String {
        format!("r{}_{}", self.family as u32, self.mhz)
    }

    /// "MIPS R10000 195 MHz".
    pub fn label(&self) -> String {
        format!("MIPS R{} {} MHz", self.family as u32, self.mhz)
    }
}

/// The part in `cpus` that `name` means: its full name, or its family
/// (`r10000`), which is the first of that family listed.
pub fn find(cpus: &[&'static Part], name: &str) -> Option<&'static Part> {
    let name = name.trim().to_ascii_lowercase();
    cpus.iter().copied().find(|p| name == p.name() || name == format!("r{}", p.family as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The PRId is implementation, major, minor: the Indy's R5000 is rev 1.0
    /// and the O2's 2.1. A family name takes the first part of it a machine
    /// lists.
    #[test]
    fn prid_is_implementation_then_revision_and_a_family_name_is_the_first_listed() {
        assert_eq!((R5000_180_EARLY.prid(), R5000_180.prid()), (0x2310, 0x2321));
        assert_eq!(find(&[&R5000_180, &R10000_250, &R12000_400], "r10000"), Some(&R10000_250));
    }
}
