//! The machine clock: one time base for a machine's guest-visible clocks. It holds still while the machine is stopped.

use std::sync::{atomic::{AtomicBool, AtomicU64, Ordering::{Acquire, Relaxed, Release}}, Arc};

#[derive(Default)]
struct Inner {
    /// Source minus machine nanoseconds, while running.
    shift: AtomicU64,
    /// The reading while stopped.
    held: AtomicU64,
    stopped: AtomicBool,
    /// ci_clock, the Indy's clock only: the source is the first CPU's retired instructions.
    #[cfg(feature = "ci_clock")]
    cycles: Option<std::sync::OnceLock<crate::cpu::mips_core::CyclesPtr>>,
    #[cfg(test)]
    stepped: Option<AtomicU64>,
    /// Runs on every reading, so a test can act at the instant a device reads the time.
    #[cfg(test)]
    on_read: Option<Box<dyn Fn() + Send + Sync>>,
}

/// Machine nanoseconds. Clones share one clock.
#[derive(Clone)]
pub struct Clock(Arc<Inner>);

impl Clock {
    pub fn new() -> Self {
        Clock(Arc::default())
    }

    fn source(&self) -> u64 {
        #[cfg(test)]
        if let Some(f) = &self.0.on_read {
            f();
        }
        #[cfg(test)]
        if let Some(s) = &self.0.stepped {
            return s.load(Relaxed);
        }
        #[cfg(feature = "ci_clock")]
        if let Some(c) = &self.0.cycles {
            return c.get().map_or(0, |c| c.get()).wrapping_mul(crate::cpu::mips_core::NS_PER_GUEST_CYCLE);
        }
        crate::platform::get_host_ticks()
    }

    /// A clock that counts the CPU's instructions once `set_cpu_cycles` gives it the counter.
    #[cfg(feature = "ci_clock")]
    pub fn counting() -> Self {
        Clock(Arc::new(Inner { cycles: Some(Default::default()), ..Default::default() }))
    }

    #[cfg(feature = "ci_clock")]
    pub fn set_cpu_cycles(&self, cycles: crate::cpu::mips_core::CyclesPtr) {
        let _ = self.0.cycles.as_ref().map(|o| o.set(cycles));
    }

    pub fn now(&self) -> u64 {
        if self.0.stopped.load(Acquire) {
            return self.0.held.load(Relaxed);
        }
        self.source().wrapping_sub(self.0.shift.load(Relaxed))
    }

    /// `stop`, `start` and `restore` are called from one thread at a time. A reader racing one can
    /// see a spike or a small step back; machine stops are rare, so neither is guarded.
    pub fn stop(&self) {
        self.0.held.store(self.now(), Relaxed);
        self.0.stopped.store(true, Release);
    }

    pub fn start(&self) {
        if self.0.stopped.load(Acquire) {
            self.restore(self.0.held.load(Relaxed));
            self.0.stopped.store(false, Release);
        }
    }

    pub fn restore(&self, ns: u64) {
        self.0.shift.store(self.source().wrapping_sub(ns), Relaxed);
        self.0.held.store(ns, Relaxed);
    }

    #[cfg(test)]
    pub fn stepped() -> Self {
        Clock(Arc::new(Inner { stepped: Some(AtomicU64::new(0)), ..Default::default() }))
    }

    /// A stepped clock that runs `f` on every reading.
    #[cfg(test)]
    pub fn stepped_on_read(f: impl Fn() + Send + Sync + 'static) -> Self {
        Clock(Arc::new(Inner { stepped: Some(AtomicU64::new(0)), on_read: Some(Box::new(f)), ..Default::default() }))
    }

    #[cfg(test)]
    pub fn step(&self, ns: u64) {
        self.0.stepped.as_ref().unwrap().fetch_add(ns, Relaxed);
    }
}

/// An optional machine time as one atomic word, so a timer can read it
/// without the device's lock: the time plus one, zero for none. `u64::MAX` is not
/// representable (it saturates to `MAX - 1`).
pub(crate) fn encode(t: Option<u64>) -> u64 {
    t.map_or(0, |t| t.saturating_add(1))
}

pub(crate) fn decode(n: u64) -> Option<u64> {
    n.checked_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn an_optional_time_round_trips_through_its_word() {
        assert_eq!(encode(None), 0);
        for t in [0, 1, u64::MAX - 1] {
            assert_eq!(decode(encode(Some(t))), Some(t));
        }
        assert_eq!(decode(0), None);
    }

    #[test]
    fn a_stopped_clock_holds_and_a_restore_moves_it() {
        let c = Clock::stepped();
        c.step(100);
        c.stop();
        c.step(1_000);
        c.restore(40);
        c.start();
        c.step(2);
        assert_eq!(c.now(), 42);
    }
}
