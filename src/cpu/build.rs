//! Building the configured processor: the one place a [`Part`] becomes a
//! cache and executor type.
//!
//! [`Part`]: crate::cpu::parts::Part

use std::sync::Arc;

use crate::cpu::mips_cache_v2::CpuModel;
use crate::cpu::mips_exec::{MipsCpuConfig, MipsExecutor};
use crate::cpu::mips_tlb::MipsTlb;
use crate::cpu::parts::CpuModule;
use crate::memory_map::MemoryMap;
use crate::traits::BusDevice;

/// A processor of type `C` on an IP27 or IP35 node, populated as `module` says: `bus` is
/// its uncached connection and `cached` the node's memory, which its cache
/// fills from and its JIT reads through. `count_hz` is CP0 Count's rate, which
/// the configuration may override. With `smp` a store-conditional is a
/// host compare-exchange.
pub fn build_cpu<C>(
    bus: Arc<dyn BusDevice>,
    cached: Arc<dyn BusDevice>,
    memory: Arc<MemoryMap>,
    module: &CpuModule,
    timers: &Arc<crate::hptimer::TimerManager>,
    count_hz: u64,
    cpu_id: usize,
    smp: bool,
) -> Box<MipsExecutor<MipsTlb, C>>
where
    C: CpuModel + From<Arc<dyn BusDevice>>,
{
    let cfg = MipsCpuConfig::for_model::<C>(memory).with_prid(module.part.prid());
    let mut exec = Box::new(MipsExecutor::new(bus, MipsTlb::new(cfg.tlb_entries), &cfg));
    exec.phys_mask = crate::window_map::CPU_PHYS_MASK;
    exec.core.cp0_config = module.config();
    exec.core.set_timer_manager(timers.clone());
    exec.core.set_count_hz(count_hz);
    exec.core.count_write_samples = true;
    exec.core.cpu_id = cpu_id as u8;
    exec.core.smp = smp;
    exec.set_cached_bus(cached.clone());
    exec.cache = C::from(cached);
    exec
}

/// Run `$body` with `$C` the cache type of `$part`, chosen once at machine
/// construction so the hot path carries no per-model branch. `$shadow` picks
/// the shadow caches for the R4400 and R5000, which also have the functional
/// ones (`CacheModel`). `$o2` selects the O2's R10000 and R12000 types, whose
/// secondary caches have 64-byte lines where the others have 128. The R16000's
/// secondary cache size is a type parameter, so `l2_kb` picks the type here, once;
/// an R12000 with 8 MB has its own.
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
            (F::R12000, _, false) if $part.l2_kb == 8192 => { type $C = sh::R12000Scache8ShadowCache; $body }
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
    use crate::cpu::parts::{R12000_360, R12000_400_8M, R16000_1000, R16000_700_EARLY, R16000_800, R16000_900};

    /// `l2_kb` picks the R16000 type: 4, 8 or 16 MiB of secondary cache.
    #[test]
    fn an_r16000_part_gets_the_cache_type_of_its_secondary_size() {
        for (part, mib) in [(&R16000_700_EARLY, 4), (&R16000_800, 4), (&R16000_900, 8), (&R16000_1000, 16)] {
            let (name, bytes) = crate::with_cpu!(part, true, false, C => (<C as CpuModel>::NAME, <C as MipsCache>::L2_SIZE));
            assert_eq!((name, bytes >> 20), ("R16000", mib), "{}", part.name());
        }
    }

    /// An R12000 with 8 MB gets the 8 MB type and the Octane's 2 MB part keeps
    /// the 2 MB one; both keep the R12000's name and PRId.
    #[test]
    fn an_r12000_with_8_mb_gets_its_own_cache_type() {
        for (part, mib) in [(&R12000_400_8M, 8), (&R12000_360, 2)] {
            let (name, bytes) = crate::with_cpu!(part, true, false, C => (<C as CpuModel>::NAME, <C as MipsCache>::L2_SIZE));
            assert_eq!((name, bytes >> 20), ("R12000", mib), "{}", part.name());
        }
        assert_eq!(R12000_400_8M.prid(), R12000_360.prid());
    }
}
