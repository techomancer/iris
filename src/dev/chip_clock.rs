//! The time base of the Indy's clock chips (DS1386, 8254).
//!
//! Host time, or under `ci_clock` guest nanoseconds: `hot.cycles` at
//! `NS_PER_GUEST_CYCLE` each, the base CP0 Count runs on there. The PROM and
//! IRIX find the CPU clock by counting Count across one tick of a clock chip,
//! so the two need one time base, or under `ci_clock` they measure how fast
//! the host runs the guest.

use std::time::Duration;
#[cfg(not(feature = "ci_clock"))]
use std::time::Instant;

/// A moment on the clock: host `Instant`, or guest nanoseconds.
#[cfg(not(feature = "ci_clock"))]
pub type Stamp = Instant;
#[cfg(feature = "ci_clock")]
pub type Stamp = u64;

/// Cheap to clone: clones share the cycle counter, so one `set_cpu_cycles`
/// reaches a copy a timer callback holds.
#[derive(Clone, Default)]
pub struct Clock {
    #[cfg(feature = "ci_clock")]
    cycles: std::sync::Arc<std::sync::OnceLock<crate::cpu::mips_core::CyclesPtr>>,
    /// The last raw reading and the guest time before it, for a CPU reset.
    #[cfg(feature = "ci_clock")]
    epoch: std::sync::Arc<std::sync::Mutex<Epoch>>,
}

/// A hard CPU reset zeroes the cycle counter, but the chips are not reset
/// with the CPU and their time goes on. A raw reading below the last one
/// means such a reset, so the time so far becomes the base the new count
/// runs on, and stamps taken before it stay valid. A reset is seen at the
/// first reading after it, so one that is not read until the new run has
/// passed the old count goes unnoticed.
#[cfg(feature = "ci_clock")]
#[derive(Default)]
struct Epoch {
    last: u64,
    base: u64,
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
        // Read the counter under the lock: two threads sharing a Clock must
        // see readings in the order they are applied, or the later one looks
        // like a reset.
        let mut e = self.epoch.lock().unwrap_or_else(|p| p.into_inner());
        let cycles = self.cycles.get().map_or(0, |c| c.get());
        let raw = cycles.wrapping_mul(crate::cpu::mips_core::NS_PER_GUEST_CYCLE);
        if raw < e.last {
            e.base += e.last;
        }
        e.last = raw;
        e.base + raw
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

#[cfg(all(test, feature = "ci_clock"))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    /// Threads that share a Clock (the PIT's CPU thread and its timer
    /// callback) must never see it jump. The counter only rises here, so a
    /// reading that goes back, or runs ahead of the counter, is a false reset.
    /// The window is nanoseconds wide, so this can pass on the old code; it
    /// cannot fail on the new.
    #[test]
    fn ci_clock_readers_on_two_threads_never_see_a_false_reset() {
        let counter = Box::leak(Box::new(AtomicU64::new(1)));
        let clock = Clock::default();
        clock.set_cpu_cycles(crate::cpu::mips_core::CyclesPtr::new(counter.as_ptr()));
        let stop = AtomicBool::new(false);
        let ns = crate::cpu::mips_core::NS_PER_GUEST_CYCLE;
        std::thread::scope(|s| {
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            });
            let readers: Vec<_> = (0..2).map(|_| s.spawn(|| {
                let mut last = 0;
                for _ in 0..200_000 {
                    let t = clock.now();
                    assert!(t >= last, "the clock went back: {t} after {last}");
                    last = t;
                }
            })).collect();
            for r in readers { r.join().unwrap(); }
            stop.store(true, Ordering::Relaxed);
        });
        let end = counter.load(Ordering::Relaxed) * ns;
        assert!(clock.now() <= end, "the clock ran ahead of the counter");
    }
}
