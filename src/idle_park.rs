//! Kernel idle-loop detection and in-place CPU thread parking.
//!
//! Shared by the interpreter run loop (`mips_exec.rs`) and the JIT dispatch
//! loop (`jit/dispatch.rs`). See `rules/perf/idle-pause-work.md`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::mips_core::{CAUSE_IP_MASK, STATUS_IM_MASK};
use crate::mips_core::MipsCore;

const IDLE_RING: usize = 32;
const SLICE_NS: u64 = 1_000_000;

/// The CPU thread while it is parked in [`IdleParkState::park`], so an
/// interrupt source can wake it at once instead of leaving it to notice on its
/// next slice. There is one CPU thread, so one slot.
static PARKER: parking_lot::Mutex<Option<std::thread::Thread>> = parking_lot::const_mutex(None);
/// Set while the CPU thread is between its last look at the pending word and
/// its sleep. Paired with the writers' order (set the bit, then read this) so
/// a wakeup cannot be lost.
static PARKED: AtomicBool = AtomicBool::new(false);

/// Wake the parked CPU thread, if it is parked. Call after setting a bit in
/// `hot.interrupts` from any thread; one relaxed-cost atomic load when the CPU
/// is running.
#[inline]
pub fn wake() {
    if PARKED.load(Ordering::SeqCst) {
        if let Some(t) = PARKER.lock().as_ref() {
            t.unpark();
        }
    }
}

/// Tracks recent architectural-state hashes to detect polling idle loops.
#[derive(Default)]
pub struct IdleParkState {
    ring: [u64; IDLE_RING],
    ring_len: usize,
    ring_pos: usize,
}

impl IdleParkState {
    /// Hash PC + GPRs (excluding k0/k1 scratch registers).
    fn hash_state(core: &MipsCore) -> u64 {
        let mut h = core.pc;
        for (i, &g) in core.gpr.iter().enumerate() {
            if i == 26 || i == 27 {
                continue;
            }
            h = h.rotate_left(7) ^ g;
        }
        h
    }

    /// Update idle ring. Returns true when the current state repeated (safe to park).
    pub fn update(&mut self, core: &MipsCore) -> bool {
        let ie = core.interrupts_enabled();
        let pending = core.hot.interrupts.load(Ordering::Relaxed) as u32;
        let ip = (core.cp0_cause | pending) & CAUSE_IP_MASK;
        let im = core.cp0_status & STATUS_IM_MASK;
        let interrupt_ready = (ip & im) != 0;

        // With every mask bit clear nothing can satisfy `park`'s
        // `(ip & im) != 0`, so parking here never wakes.
        if im == 0 {
            self.ring_len = 0;
            self.ring_pos = 0;
            return false;
        }

        if !(ie && !interrupt_ready) {
            self.ring_len = 0;
            self.ring_pos = 0;
            return false;
        }

        let h = Self::hash_state(core);
        if self.ring[..self.ring_len].contains(&h) {
            return true;
        }
        self.ring[self.ring_pos] = h;
        self.ring_pos = (self.ring_pos + 1) % IDLE_RING;
        if self.ring_len < IDLE_RING {
            self.ring_len += 1;
        }
        false
    }

    /// Park in ≤1 ms slices until an interrupt is pending or the CPU stops.
    ///
    /// The Count==Compare interrupt needs no special handling here anymore:
    /// the compare timer fires on the hptimer thread and ORs IP7 into
    /// `hot.interrupts` exactly like a device line, and CP0 Count itself is
    /// virtual (materialized from the wall clock on read), so nothing has to
    /// advance it during the sleep. Only `hot.cycles` is advanced — at a
    /// nominal ~100 MIPS — so cross-thread cycle readers (Wd33c93a's
    /// deferred-interrupt spin-wait, CP0 Random) keep seeing progress.
    pub fn park(&self, core: &mut MipsCore, running: &AtomicBool) {
        // Only park once the guest has actually armed a Compare deadline.
        // Before that (PROM), Compare use is ad-hoc and there may be nothing
        // armed to wake us. cp0_compare is zero out of reset and the guest
        // must write it to schedule anything, so a non-zero value is the
        // signal that parking is safe.
        if core.cp0_compare == 0 {
            return;
        }

        *PARKER.lock() = Some(std::thread::current());
        loop {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            // Announce the park before the last look at the pending word: a
            // writer sets its bit and then reads PARKED, so either we see the
            // bit here or the writer sees PARKED and unparks us.
            PARKED.store(true, Ordering::SeqCst);
            let pending = core.hot.interrupts.load(Ordering::SeqCst) as u32;
            let ip = (core.cp0_cause | pending) & CAUSE_IP_MASK;
            let im = core.cp0_status & STATUS_IM_MASK;
            if (ip & im) != 0 {
                break;
            }
            // ci_clock has no hptimer — the fire point is a cycles threshold
            // checked in step()'s preamble, so stop parking once we cross it.
            #[cfg(feature = "ci_clock")]
            if core.hot.cycles >= core.count_fire_cycle {
                break;
            }

            let t0 = Instant::now();
            // Still a bounded slice — `running` and the ci_clock threshold are
            // only polled — but an interrupt now ends it at once.
            std::thread::park_timeout(Duration::from_nanos(SLICE_NS));
            let elapsed_ns = t0.elapsed().as_nanos() as u64;
            core.hot.cycles = core.hot.cycles.wrapping_add(elapsed_ns / 10);
        }
        // Every exit, including the ci_clock one, leaves the flag clear: a
        // stale `true` would put `wake` on the mutex for a running CPU.
        PARKED.store(false, Ordering::SeqCst);
    }
}

pub fn idle_park_enabled() -> bool {
    std::env::var_os("IRIS_NO_IDLE").is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mips_core::{MipsCore, STATUS_IE, STATUS_IM_SHIFT};

    /// A state the detector would otherwise park on: interrupts enabled, none
    /// ready, and the same architectural state seen twice.
    fn repeated_idle_state(status: u32) -> (IdleParkState, MipsCore) {
        let mut core = MipsCore::default();
        core.cp0_status = status;
        core.cp0_cause = 0;
        core.pc = 0x8000_0100;
        let mut st = IdleParkState::default();
        st.update(&core); // first sighting fills the ring
        (st, core)
    }

    #[test]
    fn update_parks_on_a_repeated_state_when_an_interrupt_could_arrive() {
        // Control for the test below: with a mask bit set, `park`'s
        // `(ip & im) != 0` is satisfiable, so parking is safe and expected.
        let (mut st, core) = repeated_idle_state(STATUS_IE | (1 << (STATUS_IM_SHIFT + 7)));
        assert!(st.update(&core), "a repeated idle state with IM set should park");
    }

    #[test]
    fn update_declines_to_park_when_every_interrupt_is_masked() {
        // IE set but IM zero: the guest would take an interrupt, but none can
        // be delivered, so the wait `park` performs can never end.
        let (mut st, core) = repeated_idle_state(STATUS_IE);
        assert!(!st.update(&core), "IM == 0 makes park's wake condition unsatisfiable");
    }
}

#[cfg(test)]
mod wake_tests {
    use super::*;
    use crate::mips_core::{MipsCore, CAUSE_IP7, STATUS_IE, STATUS_IM_SHIFT};
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;

    /// `park` holds `&mut MipsCore`; the interrupt writers reach the same word
    /// through `MipsCpu::interrupts_ptr`, which is what this models.
    struct InterruptsPtr(*const AtomicU64);
    unsafe impl Send for InterruptsPtr {}

    /// Time from an interrupt bit being set on another thread to `park`
    /// returning on the CPU thread. `raise_after` picks where in a slice the
    /// interrupt lands.
    fn delivery_latency(raise_after: Duration) -> Duration {
        let mut core = MipsCore::default();
        core.cp0_status = STATUS_IE | (1 << (STATUS_IM_SHIFT + 7));
        core.cp0_compare = 1; // park returns immediately while this is zero
        let ptr = InterruptsPtr(&core.hot.interrupts as *const AtomicU64);
        let running = Arc::new(AtomicBool::new(true));
        let raised = Arc::new(parking_lot::Mutex::new(None::<Instant>));
        let raised_tx = raised.clone();

        let raiser = std::thread::spawn(move || {
            let p = ptr;
            std::thread::sleep(raise_after);
            *raised_tx.lock() = Some(Instant::now());
            unsafe { &*p.0 }.fetch_or(CAUSE_IP7 as u64, Ordering::SeqCst);
            wake();
        });

        let st = IdleParkState::default();
        st.park(&mut core, &running);
        let returned = Instant::now();
        raiser.join().unwrap();
        let at = raised.lock().expect("raiser set the bit");
        returned - at
    }

    /// One pass over the slice. Sweeping matters: a single sample can be fast
    /// by luck, on an interrupt that landed just before a slice boundary.
    fn worst_over_a_slice() -> Duration {
        (0..5)
            .map(|i| delivery_latency(Duration::from_micros(2_100 + i * 200)))
            .max()
            .unwrap()
    }

    #[test]
    fn an_interrupt_ends_the_park_without_waiting_out_the_slice() {
        // Without `wake` every sweep contains a phase that waits out most of a
        // slice, so no number of retries makes this pass; with it, each phase
        // costs a thread wakeup (~6 us here). The retries are only so a
        // scheduling stall on a loaded CI runner does not fail the build.
        const LIMIT: Duration = Duration::from_micros(300);
        let mut seen = Vec::new();
        for _ in 0..3 {
            let worst = worst_over_a_slice();
            if worst < LIMIT {
                return;
            }
            seen.push(worst);
        }
        panic!(
            "worst-phase latency {seen:?} over 3 sweeps, all above {LIMIT:?}; \
             a slice is {:?}",
            Duration::from_nanos(SLICE_NS)
        );
    }
}
