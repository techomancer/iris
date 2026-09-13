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

        loop {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            let pending = core.hot.interrupts.load(Ordering::Relaxed) as u32;
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
            std::thread::sleep(Duration::from_nanos(SLICE_NS));
            let elapsed_ns = t0.elapsed().as_nanos() as u64;
            core.hot.cycles = core.hot.cycles.wrapping_add(elapsed_ns / 10);
        }
    }
}

pub fn idle_park_enabled() -> bool {
    std::env::var_os("IRIS_NO_IDLE").is_none()
}
