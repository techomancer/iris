//! Building the configured processor: the one place a [`CpuModel`] becomes a
//! cache and executor type.

/// Run `$body` with `$C` the cache type of `$model`, chosen once at machine
/// construction so the hot path carries no per-model branch. `$shadow` picks
/// the shadow caches for the R4400 and R5000, which also have the functional
/// ones (`CacheModel`). The O2's R10000 and R12000 have their own types:
/// their secondary caches have 64-byte lines where the others have 128.
/// The chip's PRId (`MachineProfile::cpu_prid`) goes in through
/// `MipsCpuConfig::with_prid`.
#[macro_export]
macro_rules! with_cpu {
    ($model:expr, $shadow:expr, $C:ident => $body:expr) => {{
        use $crate::config::CpuModel as M;
        use $crate::cpu::mips_cache_shadow as sh;
        match $model {
            M::R4400_150 if $shadow => { type $C = sh::R4400ShadowCache; $body }
            M::R4400_150 => { type $C = $crate::cpu::mips_cache_v2::R4400Cache; $body }
            M::R5000_180 if $shadow => { type $C = sh::R5000ShadowCache; $body }
            M::R5000_180 => { type $C = $crate::cpu::mips_cache_v2::R5000Cache; $body }
            M::R10000_250 => { type $C = sh::R10000O2ShadowCache; $body }
            M::R10000_195 => { type $C = sh::R10000ShadowCache; $body }
            M::R12000_360 | M::R14000_600 => { type $C = sh::R12000ShadowCache; $body }
            M::R12000_400 => { type $C = sh::R12000O2ShadowCache; $body }
        }
    }};
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::config::{CpuModel, MachineProfile};
    use crate::cpu::mips_cache_v2::{CpuModel as Model, MipsCache};

    /// What `with_cpu!` builds for `model` on `profile`: PRId (the machine's,
    /// else the type's), cache type, secondary cache size.
    pub(crate) fn probe(profile: MachineProfile, model: CpuModel, shadow: bool) -> (u32, &'static str, u32) {
        crate::with_cpu!(model, shadow, C => (
            profile.cpu_prid(model),
            std::any::type_name::<C>(),
            <C as MipsCache>::L2_SIZE as u32,
        ))
    }

    pub(crate) fn row<C: Model + MipsCache>(prid: u32) -> (u32, &'static str, u32) {
        (prid, std::any::type_name::<C>(), C::L2_SIZE as u32)
    }
}
