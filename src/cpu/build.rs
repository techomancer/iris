//! Building the configured processor: the one place a [`Part`] becomes a
//! cache and executor type.
//!
//! [`Part`]: crate::cpu::parts::Part

/// Run `$body` with `$C` the cache type of `$part`, chosen once at machine
/// construction so the hot path carries no per-model branch. `$shadow` picks
/// the shadow caches for the R4400 and R5000, which also have the functional
/// ones (`CacheModel`). `$o2` selects the O2's R10000 and R12000 types, whose
/// secondary caches have 64-byte lines where the others have 128.
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
        }
    }};
}
