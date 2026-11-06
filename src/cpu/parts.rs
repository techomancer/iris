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
    R16000 = 16000,
}

impl Family {
    /// PRId bits 15:8 (R10000 UM, PRId register).
    const fn implementation(self) -> u32 {
        match self {
            Self::R4400 => 0x04,
            Self::R5000 => 0x23,
            Self::R10000 => 0x09,
            Self::R12000 => 0x0e,
            // The R16000 reuses the R14000's implementation byte.
            Self::R14000 | Self::R16000 => 0x0f,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Part {
    pub family: Family,
    pub mhz: u32,
    /// Major and minor revision, as `hinv` prints them ("Revision: 2.7"). For
    /// an R16000 the PRId's major is this major plus 2 (see [`Part::prid`]).
    pub rev: (u8, u8),
    /// Secondary cache in KB. The R5000 reads 0: its cache is not modelled.
    pub l2_kb: u32,
}

pub const R4400_150: Part = Part { family: Family::R4400, mhz: 150, rev: (5, 0), l2_kb: 1024 };
pub const R5000_180: Part = Part { family: Family::R5000, mhz: 180, rev: (2, 1), l2_kb: 0 };
pub const R5000_180_EARLY: Part = Part { rev: (1, 0), ..R5000_180 };
pub const R10000_195: Part = Part { family: Family::R10000, mhz: 195, rev: (2, 7), l2_kb: 1024 };
pub const R10000_195_EARLY: Part = Part { rev: (2, 5), ..R10000_195 };
/// The Origin 200's processor modules: the 180 MHz R10000 (rev 2.6, 1 MB
/// secondary cache) and the 270 MHz R12000 (rev 2.3, 4 MB).
pub const R10000_180: Part = Part { mhz: 180, rev: (2, 6), ..R10000_195 };
pub const R12000_270: Part = Part { mhz: 270, rev: (2, 3), l2_kb: 4096, ..R12000_360 };
pub const R10000_250: Part = Part { mhz: 250, rev: (3, 4), ..R10000_195 };
pub const R12000_360: Part = Part { family: Family::R12000, mhz: 360, rev: (3, 5), l2_kb: 2048 };
pub const R12000_400: Part = Part { mhz: 400, ..R12000_360 };
pub const R14000_600: Part = Part { family: Family::R14000, mhz: 600, rev: (2, 4), l2_kb: 2048 };
/// The Fuel's R14000A: rev 2.3 at 500 MHz with a 2 MB secondary cache, as the IP35
/// `hinv` dumps show it.
pub const R14000_500: Part = Part { mhz: 500, rev: (2, 3), ..R14000_600 };

/// The Tezro's and Origin 350's earliest R16000: rev 2.1, 4 MB secondary cache.
/// A 700 MHz R16000 also shipped as rev 2.2, in the Fuel, so this one takes
/// the `_EARLY` suffix and the plain `R16000_700` stays free for that part
/// when the Fuel takes it. The configuration name `r16000_700` is this part.
pub const R16000_700_EARLY: Part = Part { family: Family::R16000, mhz: 700, rev: (2, 1), l2_kb: 4096 };
pub const R16000_800: Part = Part { mhz: 800, rev: (2, 2), ..R16000_700_EARLY };
pub const R16000_900: Part = Part { mhz: 900, rev: (3, 0), l2_kb: 8192, ..R16000_700_EARLY };
pub const R16000_1000: Part = Part { mhz: 1000, rev: (3, 0), l2_kb: 16384, ..R16000_700_EARLY };

/// Every part, for the bare-metal bench rig, which takes any. A name matches
/// the first listed, so the rig runs the later silicon of a clock that has two.
pub const ALL: &[&Part] = &[
    &R4400_150, &R5000_180_EARLY, &R5000_180, &R10000_195, &R10000_195_EARLY, &R10000_250,
    &R10000_180, &R12000_270, &R12000_360, &R12000_400, &R14000_600, &R16000_700_EARLY, &R16000_800,
    &R16000_900, &R16000_1000,
];

impl Part {
    /// Implementation, then major and minor revision, as the PRId register
    /// holds them. `rev` is what `hinv` prints, and for an R16000 that is two
    /// below the register's major: the Linux and OpenBSD CPU probes both
    /// display an R16000's revision that way (`rev.0 + 2` here undoes it).
    pub const fn prid(&self) -> u32 {
        let major = match self.family {
            Family::R16000 => self.rev.0 + 2,
            _ => self.rev.0,
        };
        self.family.implementation() << 8 | (major as u32) << 4 | self.rev.1 as u32
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

    /// An R16000 prints a major two below the register's: the Tezro's 2.1,
    /// 2.2 and 3.0 are raw 0x41, 0x42 and 0x50, and the R14000 beside it in
    /// the table keeps its own major.
    #[test]
    fn an_r16000_register_major_is_the_displayed_one_plus_two() {
        assert_eq!(
            (R16000_700_EARLY.prid(), R16000_800.prid(), R16000_900.prid(), R16000_1000.prid()),
            (0x0f41, 0x0f42, 0x0f50, 0x0f50)
        );
        assert_eq!(R14000_600.prid(), 0x0f24);
        assert_eq!(find(&[&R16000_800, &R16000_1000], "r16000"), Some(&R16000_800));
        assert_eq!(find(&[&R16000_700_EARLY], "r16000_700"), Some(&R16000_700_EARLY));
    }
}
