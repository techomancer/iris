//! The machine clock: one time base for a machine's guest-visible clocks. It holds still while the machine is stopped.

use std::sync::{atomic::{AtomicBool, AtomicU64, Ordering::{Acquire, Relaxed, Release}}, Arc};

#[derive(Default)]
struct Inner {
    /// Source minus machine nanoseconds, while running.
    shift: AtomicU64,
    /// The reading while stopped.
    held: AtomicU64,
    stopped: AtomicBool,
    /// The instruction clock: the source is the first CPU's retired instructions, `insn_ns` each.
    cycles: Option<std::sync::OnceLock<crate::cpu::mips_core::CyclesPtr>>,
    insn_ns: u64,
    /// Instruction time: the CPU sleeps while it is ahead of the host.
    paced: bool,
    /// Host ticks less machine nanoseconds when the two were last levelled, for pacing.
    host_anchor: AtomicU64,
    #[cfg(test)]
    stepped: Option<AtomicU64>,
    /// Runs before every reading, so a test can act at the moment the clock is read.
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
        if let Some(hook) = &self.0.on_read {
            hook();
        }
        #[cfg(test)]
        if let Some(s) = &self.0.stepped {
            return s.load(Relaxed);
        }
        if let Some(c) = &self.0.cycles {
            return c.get().map_or(0, |c| c.get()).wrapping_mul(self.0.insn_ns);
        }
        crate::platform::get_host_ticks()
    }

    /// A clock that counts the CPU's instructions, `insn_ns` each, once `set_cpu_cycles` gives it the counter.
    pub fn counting(insn_ns: u64, paced: bool) -> Self {
        Clock(Arc::new(Inner { cycles: Some(Default::default()), insn_ns, paced, ..Default::default() }))
    }

    pub fn set_cpu_cycles(&self, cycles: crate::cpu::mips_core::CyclesPtr) {
        let _ = self.0.cycles.as_ref().map(|o| o.set(cycles));
    }

    /// Nanoseconds each retired instruction adds, or 0 if this clock follows the host.
    pub fn insn_ns(&self) -> u64 {
        if self.0.cycles.is_some() { self.0.insn_ns } else { 0 }
    }

    /// Whether the CPU sleeps to stay level with the host.
    pub fn paced(&self) -> bool {
        self.0.paced
    }

    /// Nanoseconds the machine clock is ahead of the host's since `level`; negative while it lags.
    pub fn lead_ns(&self) -> i64 {
        let host = crate::platform::get_host_ticks().wrapping_sub(self.0.host_anchor.load(Relaxed));
        self.now().wrapping_sub(host) as i64
    }

    /// Call the machine clock and the host's level from here on: pacing measures against this point.
    /// The CPU thread does it as it starts, so the time before the guest runs is nobody's lead or lag.
    pub fn level(&self) {
        self.0.host_anchor.store(crate::platform::get_host_ticks().wrapping_sub(self.now()), Relaxed);
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

    /// A stepped clock that runs `hook` before every reading.
    #[cfg(test)]
    pub fn stepped_with(hook: impl Fn() + Send + Sync + 'static) -> Self {
        Clock(Arc::new(Inner {
            stepped: Some(AtomicU64::new(0)),
            on_read: Some(Box::new(hook)),
            ..Default::default()
        }))
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
    fn a_counting_clock_follows_the_cycles_and_levels_to_the_host() {
        let cycles = std::cell::UnsafeCell::new(0u64);
        let c = Clock::counting(4, true);
        c.set_cpu_cycles(crate::cpu::mips_core::CyclesPtr::new(cycles.get()));
        assert_eq!((c.insn_ns(), c.paced()), (4, true));
        c.level();
        // SAFETY: only this thread writes it; the clock reads it here, between.
        unsafe { *cycles.get() = 1_000_000_000 };
        assert_eq!(c.now(), 4_000_000_000);
        assert!(c.lead_ns() > 0, "4 s of guest time against well under that of host time");
        unsafe { *cycles.get() = 0 };
        std::thread::sleep(std::time::Duration::from_millis(2));
        assert!(c.lead_ns() < 0, "no cycles retired: the guest lags");
        c.level();
        assert!(c.lead_ns().abs() < 1_000_000_000);
        assert_eq!(Clock::new().insn_ns(), 0);
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
