//! The time base of the Indy's clock chips (DS1386, 8254): host time, or under
//! `ci_clock` `hot.cycles` at `NS_PER_GUEST_CYCLE`, the base CP0 Count runs on.

use std::time::Duration;
#[cfg(not(feature = "ci_clock"))]
use std::time::Instant;

/// A moment on the clock: host `Instant`, or guest nanoseconds.
#[cfg(not(feature = "ci_clock"))]
pub type Stamp = Instant;
#[cfg(feature = "ci_clock")]
pub type Stamp = u64;

/// Clones share the cycle counter, so a timer callback's copy sees it once set.
#[derive(Clone, Default)]
pub struct Clock {
    #[cfg(feature = "ci_clock")]
    cycles: std::sync::Arc<std::sync::OnceLock<crate::cpu::mips_core::CyclesPtr>>,
}

impl Clock {
    #[cfg(not(feature = "ci_clock"))]
    pub fn now(&self) -> Stamp {
        Instant::now()
    }

    #[cfg(not(feature = "ci_clock"))]
    pub fn since(&self, start: Stamp) -> Duration {
        start.elapsed()
    }

    #[cfg(feature = "ci_clock")]
    pub fn now(&self) -> Stamp {
        self.cycles.get().map_or(0, |c| c.get()).wrapping_mul(crate::cpu::mips_core::NS_PER_GUEST_CYCLE)
    }

    #[cfg(feature = "ci_clock")]
    pub fn since(&self, start: Stamp) -> Duration {
        Duration::from_nanos(self.now().saturating_sub(start))
    }

    /// The CPU's cycle counter. Until it is set the clock stands still.
    #[cfg(feature = "ci_clock")]
    pub fn set_cpu_cycles(&self, cycles: crate::cpu::mips_core::CyclesPtr) {
        let _ = self.cycles.set(cycles);
    }
}
