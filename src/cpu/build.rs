//! Building the configured processor: the one place a [`Part`] becomes a
//! cache and executor type.
//!
//! [`Part`]: crate::cpu::parts::Part

/// Run `$body` with `$C` the cache type of `$part`, chosen once at machine
/// construction so the hot path carries no per-model branch. `$shadow` picks
/// the shadow caches for the R4400 and R5000, which also have the functional
/// ones (`CacheModel`). `$o2` selects the O2's R10000 and R12000 types, whose
/// secondary caches have 64-byte lines where the others have 128. The R16000's
/// secondary cache size is a type parameter, so `l2_kb` picks the type here, once.
#[macro_export]
macro_rules! with_cpu {
    ($part:expr, $shadow:expr, $o2:expr, $C:ident => $body:expr) => {{
        use $crate::cpu::mips_cache_shadow as sh;
        use $crate::cpu::mips_cache_v2 as full;
        use $crate::cpu::parts::Family as F;
        match ($part.family, $shadow, $o2) {
            (F::R4400, true, _) => { type $C = sh::R4400ShadowCache; $body }
            (F::R4400, false, _) => { type $C = full::R4400Cache; $body }
            (F::R5000, true, _) => { type $C = sh::R5000ShadowCache; $body }
            (F::R5000, false, _) => { type $C = full::R5000Cache; $body }
            (F::R10000, _, true) => { type $C = sh::R10000O2ShadowCache; $body }
            (F::R10000, _, false) => { type $C = sh::R10000ShadowCache; $body }
            (F::R12000 | F::R14000, _, true) => { type $C = sh::R12000O2ShadowCache; $body }
            (F::R12000 | F::R14000, _, false) => { type $C = sh::R12000ShadowCache; $body }
            (F::R16000, _, _) if $part.l2_kb == 8192 => { type $C = sh::R16000Scache8ShadowCache; $body }
            (F::R16000, _, _) if $part.l2_kb == 16384 => { type $C = sh::R16000Scache16ShadowCache; $body }
            (F::R16000, _, _) => { type $C = sh::R16000ShadowCache; $body }
        }
    }};
}

#[cfg(test)]
mod tests {
    use crate::cpu::mips_cache_v2::{CpuModel, MipsCache};
    use crate::cpu::parts::{R16000_1000, R16000_700_EARLY, R16000_800, R16000_900};

    /// `l2_kb` picks the R16000 type: 4, 8 or 16 MiB of secondary cache.
    #[test]
    fn an_r16000_part_gets_the_cache_type_of_its_secondary_size() {
        for (part, mib) in [(&R16000_700_EARLY, 4), (&R16000_800, 4), (&R16000_900, 8), (&R16000_1000, 16)] {
            let (name, bytes) = crate::with_cpu!(part, true, false, C => (<C as CpuModel>::NAME, <C as MipsCache>::L2_SIZE));
            assert_eq!((name, bytes >> 20), ("R16000", mib), "{}", part.name());
        }
    }
}
