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
/// The 400 MHz R12000 rev 3.5 with an 8 MB secondary cache, as a posted
/// Origin 3200 `hinv -mv` prints it for each C-brick processor.
pub const R12000_400_8M: Part = Part { l2_kb: 8192, ..R12000_400 };
pub const R14000_600: Part = Part { family: Family::R14000, mhz: 600, rev: (2, 4), l2_kb: 2048 };
/// The Fuel's R14000A: rev 2.3 at 500 MHz with a 2 MB secondary cache, as the IP35
/// `hinv` dumps show it.
pub const R14000_500: Part = Part { mhz: 500, rev: (2, 3), ..R14000_600 };

/// The Tezro's and Origin 350's earliest R16000: rev 2.1, 4 MB secondary cache.
/// A 700 MHz R16000 also shipped as rev 2.2, in the Fuel, so this one takes
/// the `_EARLY` suffix. The configuration name `r16000_700` is this part.
pub const R16000_700_EARLY: Part = Part { family: Family::R16000, mhz: 700, rev: (2, 1), l2_kb: 4096 };
pub const R16000_800: Part = Part { mhz: 800, rev: (2, 2), ..R16000_700_EARLY };
pub const R16000_900: Part = Part { mhz: 900, rev: (3, 0), l2_kb: 8192, ..R16000_700_EARLY };
pub const R16000_1000: Part = Part { mhz: 1000, rev: (3, 0), l2_kb: 16384, ..R16000_700_EARLY };

/// An R10000-family part as a node board is populated with it, with its mode word: the reset
/// configuration the processor latches, whose bits 24:0 land in CP0 Config. It
/// depends on the system clock as well as the part, so a module is named for both.
#[derive(Debug, PartialEq, Eq)]
pub struct CpuModule {
    pub part: &'static Part,
    pub mode_word: u32,
}

impl CpuModule {
    /// The pipeline clock in hertz.
    pub fn clock_hz(&self) -> u64 {
        u64::from(self.part.mhz) * 1_000_000
    }

    /// Config at cold reset: the mode word's bits 24:0 under the hardwired
    /// 32 KiB primary-cache fields.
    pub fn config(&self) -> u32 {
        0x6c00_0000 | (self.mode_word & 0x01ff_ffff)
    }
}

/// The 180 MHz R10000 on a 90 MHz system clock: the Origin 200's IP29 and the
/// Origin 2000 and Onyx2 node boards.
pub const R10000_180_SYSCLK_90: CpuModule = CpuModule { part: &R10000_180, mode_word: 0x1211_a785 };
/// The 270 MHz R12000 on a 90 MHz system clock, on the same boards.
pub const R12000_270_SYSCLK_90: CpuModule = CpuModule { part: &R12000_270, mode_word: 0x1213_ab85 };
/// The 400 MHz, 8 MiB R12000 on a 200 MHz system clock: the IP35 C-brick of
/// the Origin 3200 and the Onyx 3000.
pub const R12000_400_8M_SYSCLK_200: CpuModule = CpuModule { part: &R12000_400_8M, mode_word: 0x5414_a785 };
/// The 500 MHz R14000 on a 200 MHz system clock: the Fuel's IP34.
pub const R14000_500_SYSCLK_200: CpuModule = CpuModule { part: &R14000_500, mode_word: 0x549a_a985 };
/// The R16000s on a 200 MHz system clock: the Tezro's IP53, and the IP59 for
/// the 1 GHz part.
pub const R16000_700_EARLY_SYSCLK_200: CpuModule = CpuModule { part: &R16000_700_EARLY, mode_word: 0x589b_ad85 };
pub const R16000_800_SYSCLK_200: CpuModule = CpuModule { part: &R16000_800, mode_word: 0x549b_af85 };
pub const R16000_900_SYSCLK_200: CpuModule = CpuModule { part: &R16000_900, mode_word: 0x569c_b185 };
pub const R16000_1000_SYSCLK_200: CpuModule = CpuModule { part: &R16000_1000, mode_word: 0x4aed_b385 };

/// The processor modules an IP27 node shipped with: the 180 MHz R10000
/// (revision 2.6, 1 MiB secondary cache) and the 270 MHz R12000 (revision
/// 2.3, 4 MiB).
///
/// The R10000 module runs on the Octane's R10000 cache type, which the PROM
/// accepts; the R12000 module has its own 4 MiB type, since the Octane's 2 MiB
/// fails the PROM's secondary-cache tag test.
pub const MODULES: [&CpuModule; 2] = [&R10000_180_SYSCLK_90, &R12000_270_SYSCLK_90];

/// The processors an IP27 node takes, the default first.
pub const CPUS: &[&Part] = &[MODULES[0].part, MODULES[1].part];

/// The IP27 module built on `part`, if a node takes one.
pub fn cpu_module(part: &Part) -> Option<&'static CpuModule> {
    MODULES.iter().copied().find(|m| m.part == part)
}

/// Every part, for the bare-metal bench rig, which takes any. A name matches
/// the first listed, so the order picks the silicon of a clock that has two:
/// the Indy's rev 1.0 R5000, which the bench runs, and the rev 2.7 R10000.
pub const ALL: &[&Part] = &[
    &R4400_150, &R5000_180_EARLY, &R5000_180, &R10000_195, &R10000_195_EARLY, &R10000_250,
    &R10000_180, &R12000_270, &R12000_360, &R12000_400, &R12000_400_8M, &R14000_600,
    &R14000_500, &R16000_700_EARLY, &R16000_800, &R16000_900, &R16000_1000,
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
        assert_eq!([R16000_700_EARLY.rev, R16000_800.rev, R16000_900.rev, R16000_1000.rev], [(2, 1), (2, 2), (3, 0), (3, 0)]);
        assert_eq!(find(&[&R16000_800, &R16000_1000], "r16000"), Some(&R16000_800));
        assert_eq!(find(&[&R16000_700_EARLY], "r16000_700"), Some(&R16000_700_EARLY));
    }

    /// A module's mode word EC field (bits 12:9) is 2 * pipeline / system - 1.
    #[test]
    fn a_modules_mode_word_divides_its_part_down_to_its_system_clock() {
        let ec = |m: &CpuModule| (m.mode_word >> 9) & 0xf;
        for m in [&R10000_180_SYSCLK_90, &R12000_270_SYSCLK_90] {
            assert_eq!(2 * m.part.mhz / 90 - 1, ec(m), "{}", m.part.name());
        }
        for m in [
            &R12000_400_8M_SYSCLK_200, &R14000_500_SYSCLK_200, &R16000_700_EARLY_SYSCLK_200,
            &R16000_800_SYSCLK_200, &R16000_900_SYSCLK_200, &R16000_1000_SYSCLK_200,
        ] {
            assert_eq!(2 * m.part.mhz / 200 - 1, ec(m), "{}", m.part.name());
        }
    }

    /// Config is the mode word's low 25 bits under the 32 KiB primary-cache
    /// fields, whatever the part: the Origin 200's 180 MHz R10000, the Fuel's
    /// R14000 and the Tezro's 1 GHz R16000.
    #[test]
    fn a_modules_config_keeps_the_mode_words_low_bits_under_the_cache_fields() {
        assert_eq!(R10000_180_SYSCLK_90.config(), 0x6c11_a785);
        assert_eq!(R12000_270_SYSCLK_90.config(), 0x6c13_ab85);
        assert_eq!(R14000_500_SYSCLK_200.config(), 0x6c9a_a985);
        assert_eq!(R16000_1000_SYSCLK_200.config(), 0x6ced_b385);
        assert_eq!(R14000_500_SYSCLK_200.clock_hz(), 500_000_000);
    }

    /// The bench rig takes any part by name from `ALL`, the Fuel's 500 MHz
    /// R14000 too, and the bare family name still picks the 600 MHz part
    /// listed before it. Every listed name resolves (a clock with two
    /// revisions to the first of them).
    #[test]
    fn every_part_is_on_the_bench_rig_and_r14000_names_the_600_mhz_part() {
        assert_eq!(find(ALL, "r14000_500"), Some(&R14000_500));
        assert_eq!(find(ALL, "r14000"), Some(&R14000_600));
        for part in ALL {
            let found = find(ALL, &part.name()).unwrap_or_else(|| panic!("{} is not found", part.name()));
            assert_eq!(found.name(), part.name());
        }
    }
}
