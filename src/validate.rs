//! Phase 3.3: snapshot determinism validator.
//!
//! Loads a saved snapshot twice, runs the CPU `n` instructions inline (with
//! all peripheral threads stopped to eliminate scheduling jitter), and
//! diffs the resulting CPU state digests. Two passes over the same starting
//! state should produce bit-identical CPU registers — any divergence points
//! at non-determinism in `load_snapshot` (host wallclock leakage, missing
//! `load_state` field, uninitialised structure) that would silently corrupt
//! CI replays.
//!
//! With JIT descoped (Phase 2.5), the original "JIT vs interp lockstep"
//! framing is gone; this is the snapshot-determinism portion. Peripheral
//! threads are stopped during the test so device-side timing variance
//! doesn't leak into the result.

use crate::machine::Machine;
use crate::cpu::mips_exec::{CpuStateDigest, EXEC_RETRY};

/// Result of `validate_snapshot_determinism`.
#[derive(Debug)]
pub struct DeterminismReport {
    pub instructions_run: u64,
    pub deterministic: bool,
    /// Per-field divergence list. Empty when `deterministic` is true.
    pub diffs: Vec<(String, String, String)>,
    pub state_a: CpuStateDigest,
    pub state_b: CpuStateDigest,
}

impl DeterminismReport {
    pub fn summary(&self) -> String {
        if self.deterministic {
            format!(
                "deterministic for {} instructions (PC=0x{:016x})",
                self.instructions_run, self.state_a.pc
            )
        } else {
            let mut s = format!(
                "DIVERGED after {} instructions ({} field(s)):",
                self.instructions_run,
                self.diffs.len()
            );
            for (field, a, b) in &self.diffs {
                s.push_str(&format!("\n  {}: A={} B={}", field, a, b));
            }
            s
        }
    }
}

/// Run two passes of `load_snapshot(name); step n; capture` and diff the
/// resulting CPU state digests. Side effects: leaves the machine stopped
/// after both passes, with the second-pass state loaded. Caller is
/// responsible for any subsequent `start`/`restart_peripherals`.
pub fn validate_snapshot_determinism(
    machine: &mut Machine,
    name: &str,
    n_instructions: u64,
) -> Result<DeterminismReport, String> {
    // Pass A: load with everything paused → step inline → capture.
    // load_snapshot_paused leaves CPU and peripheral threads stopped, so no
    // thread runs between load and digest. This is the key to surfacing
    // genuine load_state determinism issues vs. thread-scheduling jitter.
    machine.load_snapshot_paused(name)?;
    let executed_a = machine.cpu_step_n_inline(n_instructions)?;
    let state_a = machine.cpu_state_digest()?;

    // Pass B: same starting snapshot, fresh load.
    machine.load_snapshot_paused(name)?;
    let executed_b = machine.cpu_step_n_inline(n_instructions)?;
    let state_b = machine.cpu_state_digest()?;

    if executed_a != executed_b {
        return Err(format!(
            "step counts disagree: A ran {}, B ran {} (CPU stopped itself differently)",
            executed_a, executed_b
        ));
    }

    let diffs = state_a.diff(&state_b);
    Ok(DeterminismReport {
        instructions_run: executed_a,
        deterministic: diffs.is_empty(),
        diffs,
        state_a,
        state_b,
    })
}

/// A single divergence between the interpreter-only reference run and the
/// real-JIT-dispatch run — either the one `validate_jit_determinism` stopped
/// at, or one of the `skip` ones it reconverged past on the way there (see
/// `JitDeterminismReport`). Gated `jitv2 + developer` — every field and
/// every `Machine` method `validate_jit_determinism` calls (`cpu_set_jitv2_dispatch_enabled`,
/// `cpu_step_one_inline_counting_instructions`, etc.) only exist under that
/// same combination; the `jitcheck` monitor command that's this type's only
/// real caller carries the identical gate.
#[cfg(all(feature = "jitv2", feature = "developer"))]
#[derive(Debug)]
pub struct JitDivergence {
    /// 1-indexed architectural instruction count at which this was detected
    /// (i.e. "diverged right after retiring instruction N").
    pub instruction: u64,
    /// The replay (JIT-dispatch) pass's own PC after this instruction —
    /// shown unconditionally (not just when `pc` itself is one of the
    /// diverged fields) since it's the single most useful piece of context
    /// for going and looking at what instruction actually caused this.
    pub replay_pc: u64,
    pub diffs: Vec<(String, String, String)>,
}

/// Result of `validate_jit_determinism`.
#[cfg(all(feature = "jitv2", feature = "developer"))]
#[derive(Debug)]
pub struct JitDeterminismReport {
    /// The divergence that actually stopped the run — `None` means no
    /// divergence across the whole `n_instructions` window (after skipping
    /// past `skipped`, if any).
    pub stopped_at: Option<JitDivergence>,
    /// Divergences that were reconverged past because they fell within the
    /// caller's `skip` budget, in the order encountered. Always empty when
    /// `skip == 0`. Never silently discarded — every one of these was a
    /// real divergence, just one the caller already told us to treat as
    /// known-benign and step past.
    pub skipped: Vec<JitDivergence>,
}

/// Resurrects this module's original "JIT vs interp lockstep" purpose (see
/// module doc comment) at a coarser, real-full-speed-JIT granularity than
/// `jitv2_lockstep` (which recompiles and compares every single instruction
/// via Cranelift, on every dispatch, in-process — correct but very slow, and
/// only ever compiled into a `jitv2_lockstep` build). This runs the
/// interpreter-only reference pass and the real-JIT-dispatch pass as two
/// genuinely separate full passes over the same starting `LiveCheckpoint`,
/// comparing per-instruction, on any normal `jitv2` (non-lockstep) build —
/// usable directly against a live, already-running, already-mid-boot
/// machine (no named on-disk snapshot required — see `LiveCheckpoint`'s doc
/// comment for what that does and doesn't capture: no SCSI/disk state, safe
/// for early-boot PROM debugging, not a general save-state substitute once
/// real disk I/O is in play).
///
/// Stops at the first *unskipped* divergence, full stop — no open-ended
/// reconverge-and-keep-scanning-for-more. An earlier version of this
/// function reconverged (force-restored the reference pass's register
/// state) unconditionally and kept going, collecting up to `max_reported`
/// divergences per run — in practice this just buried the one real,
/// actionable divergence under a pile of downstream noise from continuing
/// to run code that had already diverged (every later "divergence" is
/// comparing two engines that may no longer be executing the same logical
/// program at all). When it stops, the machine is left exactly at the
/// JIT-dispatch pass's own (wrong) state — not force-corrected back to the
/// reference — so it can be inspected directly (`dt`, `regs`, MCP tools) to
/// find out what actually went wrong, which is the entire point.
///
/// `skip`: once a divergence at instruction N has been manually inspected
/// and confirmed benign (e.g. a known, already-understood modeling gap —
/// `cp0_count`/`count_hz` themselves never count as divergences here at
/// all, see below, but there can be others), re-running with `skip >= 1`
/// reconverges past that *specific* one (restores the reference pass's
/// register/cop0 state at that exact point) and keeps comparing, so it
/// doesn't block finding whatever comes after it. Each reconverge is a
/// deliberate, individually-justified skip, not a blanket "ignore anything
/// that looks like this field" — every skipped divergence is still visible
/// in `JitDeterminismReport::skipped` so nothing silently vanishes.
///
/// Memory cost: the reference pass's per-instruction digests
/// (`CpuStateDigest`, ~344 bytes each) are all held at once for the diff
/// pass — `n_instructions` of them, so ~344 bytes × `n_instructions` (e.g.
/// ~34 MB for a million instructions). No ring buffer / lookback cap: both
/// passes run to completion sequentially over the exact same instruction
/// count from the exact same starting checkpoint, so the replay pass can
/// never need to compare against anything outside `0..n_instructions` —
/// there's nothing to prune, and a bounded lookback would only add a "we
/// don't actually need this parameter, and getting it wrong produces a
/// confusing error" foot-gun for no benefit.
///
/// Side effects: leaves the CPU stopped on return (both success and error
/// paths) with the replay pass's (real-JIT-dispatch) final state loaded —
/// useful on its own for interactive debugging (inspect exactly where a
/// real divergence left the machine via `dt`/`regs`/etc. right after
/// `jitcheck` reports one, without the CPU racing ahead). Peripherals
/// (MC/HPC3/REX3) are left running — `restore_live_checkpoint` restarts
/// them itself — so the rest of the monitor console stays responsive.
/// Caller is responsible for `machine.cpu.start()` if normal CPU execution
/// should resume.
#[cfg(all(feature = "jitv2", feature = "developer"))]
pub fn validate_jit_determinism(
    machine: &mut Machine,
    n_instructions: u64,
    skip: u64,
) -> Result<JitDeterminismReport, String> {
    let checkpoint = machine.capture_live_checkpoint();

    // Reference pass: interpreter-only. digests[i] is the state after
    // instruction i+1 (0-indexed); digest after instruction n_instructions
    // is state_final. Fixup recording is armed for this whole pass so every
    // real read from a HW_READ_FIXUP_ADDRS address (e.g. RPSS_CTR) gets
    // captured into the digest it lands in — see hw_read_fixup_recording's
    // doc comment.
    machine.restore_live_checkpoint(&checkpoint)?;
    let was_dispatch_enabled = machine.cpu_set_jitv2_dispatch_enabled(false)?;
    machine.cpu_set_hw_read_fixup_recording(true)?;

    // A retry (EXEC_RETRY — bus busy, breakpoint mid-access) doesn't retire
    // anything: `hot.cycles` still ticks once (step()'s own unconditional
    // per-call increment, see step_one_inline_counting_instructions' doc
    // comment), so naively trusting its "retired" count here would count a
    // retry as a real instruction and push a digest for state that never
    // architecturally changed. Loop at the same pos_a until a step actually
    // retires (step_status != EXEC_RETRY) before advancing/recording —
    // matches the interpreter's own contract: nothing observable happens on
    // a retry, so it must not consume a slot in `digests`.
    // MAX_RETRIES_PER_INSTRUCTION: a real, bus-busy retry clears within a
    // handful of host cycles (the device finishing whatever real work made
    // it busy) — but a device that's stuck permanently busy (e.g. a
    // dependency this checkpoint-restore's inline single-stepping doesn't
    // actually satisfy, like a worker thread that needs more real wall-clock
    // time than a tight retry loop gives it) must not spin `jitcheck`
    // forever holding the executor lock, hanging the whole monitor console
    // with no way to break in and inspect what's stuck (found live: exactly
    // this, capturing a checkpoint mid-REX3-GFIFO-drain hung on the very
    // first instruction of the reference pass, unrecoverable short of
    // killing the process). A bounded cap turns that into a clear,
    // diagnosable error instead.
    const MAX_RETRIES_PER_INSTRUCTION: u32 = 100_000;

    let mut digests: Vec<CpuStateDigest> = Vec::with_capacity(n_instructions as usize);
    let mut pos_a: u64 = 0;
    let mut retry_count: u32 = 0;
    while pos_a < n_instructions {
        let retired = machine.cpu_step_one_inline_counting_instructions()? as u64;
        let digest = machine.cpu_state_digest()?;
        if digest.step_status == EXEC_RETRY {
            retry_count += 1;
            if retry_count > MAX_RETRIES_PER_INSTRUCTION {
                return Err(format!(
                    "reference pass: instruction at pc={:#018x} retried more than {} times without completing — a device is stuck busy (not a transient retry); machine left stopped at the stuck point for inspection",
                    digest.pc, MAX_RETRIES_PER_INSTRUCTION
                ));
            }
            continue;
        }
        retry_count = 0;
        pos_a = (pos_a + retired).min(n_instructions);
        digests.push(digest);
    }
    let state_final = machine.cpu_state_digest()?;
    machine.cpu_set_jitv2_dispatch_enabled(was_dispatch_enabled)?;
    machine.cpu_set_hw_read_fixup_recording(false)?;

    // Replay pass: real JIT dispatch on, same starting checkpoint. Before
    // each step, arm hw_read_fixup_replay with the reference pass's
    // recorded hardware-read values for that same step, so a
    // wall-clock-driven register (RPSS_CTR) reads back bit-identical to
    // what the reference pass saw instead of drifting with real elapsed
    // time between the two passes — see HW_READ_FIXUP_ADDRS's doc comment
    // for why that drift is not a JIT bug and would otherwise make a tight
    // polling loop diverge on every single iteration.
    machine.restore_live_checkpoint(&checkpoint)?;
    machine.cpu_set_jitv2_dispatch_enabled(true)?;

    let mut skipped: Vec<JitDivergence> = Vec::new();
    let mut pos_b: u64 = 0;
    while pos_b < n_instructions {
        let upcoming = digests.get(pos_b as usize).unwrap_or(&state_final);
        machine.cpu_set_hw_read_fixup_replay(Some(upcoming.hw_reads.clone()))?;

        // Resync cp0_count/count_hz to the reference's own PRE-step
        // values before each step attempt (not just after, per-instruction,
        // the way this used to work). `before_step` — NOT `upcoming`, which
        // is the state AFTER this step (digests[pos_b], correctly used above
        // for hw_read_fixup_replay's own "what this step's reads will
        // produce" purpose) — must be the state after the *previous*
        // instruction, i.e. digests[pos_b - 1], or the raw checkpoint's own
        // starting values for the very first step (pos_b == 0, nothing has
        // run yet, digests is empty at that index). Using `upcoming` here
        // instead was a real bug (found live: instruction 2's replay-pass
        // step ran with cp0_count already resynced to instruction 2's own
        // POST-step reference value, one full instruction too far ahead —
        // same off-by-one shape as the `expected` calculation below already
        // had to fix once) — step()'s own timer_fired check
        // (cp0_compare.wrapping_sub(prev) <= count_step) is exquisitely
        // sensitive to `prev` right at reset (cp0_compare starts at 0), so
        // even a one-step-ahead cp0_count was enough to flip CAUSE_IP7
        // between the two passes on literally the second instruction ever
        // compared.
        //
        // cp0_count/count_hz are deliberately excluded from the diff
        // below (unlike validate_snapshot_determinism's use of the same
        // `diff`, where both passes replay the identical instruction stream
        // against the identical host-timing calibration and matching is a
        // genuine determinism signal): the interpreter-only and
        // real-JIT-dispatch passes are two separate wall-clock timed runs,
        // so cp0_count legitimately differs by real elapsed host time
        // between them even with zero JIT correctness bug. But step()'s own
        // top-of-call timer check reads live cp0_count/cp0_compare *during*
        // the step, not after — resyncing only post-hoc (the original
        // design) fixes the value for the next `mfc0` read but does nothing
        // to stop CAUSE_IP7 itself computing differently, especially across
        // a fast-forward loop that can now retry a different number of
        // times than the reference pass for the same logical instruction
        // (same host-timing-drift class as RPSS_CTR). Resyncing here,
        // before every attempt including every retry, keeps
        // cp0_count/cp0_compare identical to the reference at the exact
        // moment each step() evaluates timer_fired, so CAUSE_IP7 comes out
        // identically too — no separate masking needed downstream.
        let before_step = if pos_b == 0 { None } else { digests.get((pos_b - 1) as usize) };
        if let Some(before_step) = before_step {
            machine.cpu_fixup_cp0_count(before_step)?;
        }

        // Fast-forward through any retry exactly like the reference pass
        // does — keep calling (the real exec_decoded dispatch loop's own
        // behavior: a retry-bail just re-enters and gets a fresh result
        // next call, see emit_check_mem_exc's doc comment) until this word
        // actually retires. hw_read_fixup_replay/cp0_count are re-armed on
        // every attempt — a device that legitimately reads differently
        // across retries within the *same* logical instruction isn't a
        // case either engine's own real dispatch loop treats specially, so
        // neither does this.
        let mut retry_count: u32 = 0;
        let (retired, replay_digest) = loop {
            let retired = machine.cpu_step_one_inline_counting_instructions()? as u64;
            let digest = machine.cpu_state_digest()?;
            if digest.step_status != EXEC_RETRY {
                break (retired, digest);
            }
            retry_count += 1;
            if retry_count > MAX_RETRIES_PER_INSTRUCTION {
                return Err(format!(
                    "replay pass: instruction at pc={:#018x} retried more than {} times without completing — a device is stuck busy (not a transient retry); machine left stopped at the stuck point for inspection",
                    digest.pc, MAX_RETRIES_PER_INSTRUCTION
                ));
            }
            if let Some(before_step) = before_step {
                machine.cpu_fixup_cp0_count(before_step)?;
            }
        };
        pos_b = (pos_b + retired).min(n_instructions);

        let expected = if pos_b < n_instructions {
            // digests[i] holds the state after instruction i+1 (0-indexed
            // push order from the reference pass's loop above) — so "state
            // after instruction pos_b" is digests[pos_b - 1], not
            // digests[pos_b]. Off-by-one here was a real bug: whenever a
            // step retired more than 1 instruction (any JIT-compiled
            // branch+delay-slot unit), pos_b jumped straight past the index
            // that actually corresponds to "after this step", comparing
            // against the state one instruction further ahead instead —
            // guaranteed spurious divergence on every single JIT-dispatched
            // branch, which is exactly what surfaced this (DIVERGED at
            // instruction 2, right after the reset vector's j+delay-slot).
            &digests[(pos_b - 1) as usize]
        } else {
            &state_final
        };

        // Post-step resync too (the original behavior, still needed): keeps
        // cp0_count correct for whatever runs *after* this comparison, same
        // reasoning as above.
        machine.cpu_fixup_cp0_count(expected)?;

        let diffs: Vec<_> = expected.diff(&replay_digest).into_iter()
            .filter(|(field, _, _)| field != "cp0_count" && field != "count_hz")
            .collect();
        if !diffs.is_empty() {
            let divergence = JitDivergence { instruction: pos_b, replay_pc: replay_digest.pc, diffs };

            if (skipped.len() as u64) < skip {
                // Within the caller's skip budget: this one was already
                // inspected and confirmed benign, so force-correct CPU
                // state back to the reference (undoing exactly this
                // divergence, nothing more) and keep comparing. Every
                // reconverged field write goes through the normal
                // cop0-status-aware restore path, so mode-dependent
                // pointers (pcp/translate/fpr) stay consistent afterward.
                machine.cpu_restore_state_digest(expected)?;
                skipped.push(divergence);
                continue;
            }

            // Skip budget exhausted (or skip == 0): stop immediately — do
            // NOT reconverge, do NOT keep scanning. The machine is left
            // exactly where the JIT-dispatch pass put it
            // (`cpu_set_jitv2_dispatch_enabled` is intentionally not
            // restored to its prior value below either, on this path — the
            // caller inspecting a real divergence wants the machine left
            // stopped, not silently toggled back and resumed). See this
            // function's doc comment for why continuing further was
            // actively counterproductive. Peripherals are already running —
            // restore_live_checkpoint restarts them itself right after
            // loading device state, so there's nothing to do here beyond
            // leaving the CPU alone. hw_read_fixup_replay IS cleared here —
            // it's pure input plumbing for this function's own replay loop,
            // not part of the CPU/JIT state the caller wants to inspect, and
            // leaving it armed would silently corrupt any real bus read the
            // caller makes afterward (e.g. single-stepping past the
            // divergence from the monitor).
            machine.cpu_set_hw_read_fixup_replay(None)?;
            return Ok(JitDeterminismReport { stopped_at: Some(divergence), skipped });
        }
    }
    machine.cpu_set_jitv2_dispatch_enabled(was_dispatch_enabled)?;
    machine.cpu_set_hw_read_fixup_replay(None)?;

    Ok(JitDeterminismReport { stopped_at: None, skipped })
}
