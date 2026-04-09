//! Adaptive JIT dispatch loop with tiered compilation and speculative execution.
//!
//! Interpreter-first architecture: the interpreter runs in short bursts, with
//! cache probes after each burst. One JIT block per probe, then back to interpreter.
//! Blocks start at Tier 0 (ALU only) and earn promotion through stable execution.
//!
//! The probe interval adapts dynamically: frequent cache hits → shorter interval
//! (probe more often), frequent misses → longer interval (less overhead).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::mips_exec::{MipsExecutor, DecodedInstr, EXEC_BREAKPOINT, decode_into};
use crate::mips_tlb::Tlb;
use crate::mips_cache_v2::MipsCache;

use super::cache::{BlockTier, CodeCache, TierConfig};
use super::compiler::BlockCompiler;
use super::context::{JitContext, EXIT_NORMAL, EXIT_EXCEPTION};
use super::helpers::HelperPtrs;
use super::profile::{self, ProfileEntry};
use super::snapshot::CpuRollbackSnapshot;
use super::trace::{TraceWriter, TraceRecord};

const MAX_BLOCK_LEN: usize = 64;

/// How many interpreter steps in one outer batch (controls flush_cycles frequency).
const BATCH_SIZE: u32 = 10000;

/// Adaptive probe interval controller.
///
/// Asymmetric adjustment: hits pull the interval down aggressively (we want to
/// exploit hot code), misses push it up gently (don't overreact to cold regions).
/// Cache size provides a floor — more compiled blocks means shorter intervals
/// even when the instantaneous hit rate is low.
struct ProbeController {
    /// Current probe interval (interpreter steps between cache probes).
    interval: u32,
    /// Minimum allowed interval.
    min_interval: u32,
    /// Maximum allowed interval.
    max_interval: u32,
    /// Exponentially weighted hit rate (0..256 fixed-point, 256 = 100%).
    ewma_hit_rate: u32,
    /// Number of compiled blocks (updated externally).
    cache_size: u32,
    /// Simple LFSR for jitter (avoids lock-step with OS timers).
    lfsr: u32,
}

impl ProbeController {
    fn new() -> Self {
        let base = std::env::var("IRIS_JIT_PROBE").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(200u32);
        let min = std::env::var("IRIS_JIT_PROBE_MIN").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(100u32);
        let max = std::env::var("IRIS_JIT_PROBE_MAX").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(2000u32);
        Self {
            interval: base.clamp(min, max),
            min_interval: min,
            max_interval: max,
            ewma_hit_rate: 0,
            cache_size: 0,
            lfsr: 0xACE1u32,
        }
    }

    /// Record a cache hit — aggressively pull interval down.
    fn record_hit(&mut self) {
        // EWMA with alpha ~1/8 for hits (fast response to hot code)
        self.ewma_hit_rate = self.ewma_hit_rate - (self.ewma_hit_rate / 8) + 32; // +32 = 1/8 of 256

        // Each hit immediately nudges interval down by ~3%
        self.interval = (self.interval * 31 / 32).max(self.min_interval);
    }

    /// Record a cache miss — gently push interval up.
    fn record_miss(&mut self) {
        // EWMA with alpha ~1/32 for misses (slow response, don't overreact)
        self.ewma_hit_rate = self.ewma_hit_rate.saturating_sub(self.ewma_hit_rate / 32);

        // Misses push interval up by ~1% (3x slower than hit pull-down)
        self.interval = (self.interval * 33 / 32).min(self.max_interval);
    }

    /// Update cache size — provides an interval floor.
    fn set_cache_size(&mut self, size: u32) {
        self.cache_size = size;
    }

    /// Get current interval with jitter, incorporating cache size pressure.
    fn next_interval(&mut self) -> u32 {
        // Cache size pressure: more blocks compiled → gently push interval down.
        // Uses sqrt so 100 blocks barely changes anything, 10000 blocks halves it,
        // but never goes below min_interval.
        // 100 blocks → factor 1.0 (no change), 1000 → 0.68, 10000 → 0.46, 50000 → 0.31
        let cache_factor = if self.cache_size > 100 {
            1.0f32 / (self.cache_size as f32 / 100.0).sqrt().max(1.0)
        } else {
            1.0
        };
        let cache_adjusted = (self.interval as f32 * cache_factor) as u32;
        let effective = cache_adjusted.clamp(self.min_interval, self.max_interval);

        // Galois LFSR for cheap pseudo-randomness
        let bit = self.lfsr & 1;
        self.lfsr >>= 1;
        if bit != 0 { self.lfsr ^= 0xB400; }

        // Jitter: ~0.85x to ~1.15x using 3 bits of LFSR
        let jitter_bits = (self.lfsr & 0x7) as u32; // 0-7
        let jittered = effective * (17 + jitter_bits) / 21; // range ~0.81x to ~1.14x
        jittered.clamp(self.min_interval, self.max_interval)
    }
}

pub fn run_jit_dispatch<T: Tlb, C: MipsCache>(
    exec: &mut MipsExecutor<T, C>,
    running: &AtomicBool,
) {
    let jit_enabled = std::env::var("IRIS_JIT").map(|v| v == "1").unwrap_or(false);

    if !jit_enabled {
        eprintln!("JIT: interpreter-only mode (set IRIS_JIT=1 to enable compilation)");
        interpreter_loop(exec, running);
        return;
    }

    let exec_ptr: *mut MipsExecutor<T, C> = exec as *mut _;

    // IRIS_JIT_MAX_TIER: cap the highest tier blocks can reach (0=Alu, 1=Loads, 2=Full)
    let max_tier = match std::env::var("IRIS_JIT_MAX_TIER").ok().and_then(|v| v.parse::<u8>().ok()) {
        Some(0) => BlockTier::Alu,
        Some(1) => BlockTier::Loads,
        _ => BlockTier::Full,
    };
    // IRIS_JIT_VERIFY=1: after each JIT block, re-run via interpreter and compare
    let verify_mode = std::env::var("IRIS_JIT_VERIFY").map(|v| v == "1").unwrap_or(false);
    let tier_cfg = TierConfig::from_env();
    let mut probe = ProbeController::new();
    eprintln!("JIT: adaptive mode (max_tier={:?}, verify={}, probe={} [{}-{}], stable={}, promote={}, demote={})",
        max_tier, verify_mode, probe.interval, probe.min_interval, probe.max_interval,
        tier_cfg.stable, tier_cfg.promote, tier_cfg.demote);
    let helpers = HelperPtrs::new::<T, C>();
    let mut compiler = BlockCompiler::new(&helpers);
    let mut cache = CodeCache::new();
    let mut ctx = JitContext::new();
    ctx.executor_ptr = exec_ptr as u64;

    let mut trace_writer = TraceWriter::from_env();

    let mut total_jit_instrs: u64 = 0;
    let mut total_interp_steps: u64 = 0;
    let mut blocks_compiled: u64 = 0;
    let mut promotions: u64 = 0;
    let mut demotions: u64 = 0;
    let mut rollbacks: u64 = 0;

    // Load saved profile and eagerly compile hot blocks
    {
        let exec = unsafe { &mut *exec_ptr };
        let profile_entries = profile::load_profile();
        let mut profile_compiled = 0u64;
        for entry in &profile_entries {
            let tier = if entry.tier > max_tier { max_tier } else { entry.tier };
            if tier == BlockTier::Alu {
                continue;
            }
            let instrs = trace_block(exec, entry.virt_pc, tier);
            if !instrs.is_empty() {
                if let Some(mut block) = compiler.compile_block(&instrs, entry.virt_pc, tier) {
                    block.phys_addr = entry.phys_pc;
                    cache.insert(entry.phys_pc, entry.virt_pc, block);
                    blocks_compiled += 1;
                    profile_compiled += 1;
                }
            }
        }
        if profile_compiled > 0 {
            eprintln!("JIT profile: pre-compiled {} blocks from profile", profile_compiled);
        }
    }

    while running.load(Ordering::Relaxed) {
        let mut steps_in_batch: u32 = 0;

        while steps_in_batch < BATCH_SIZE {
            let burst = probe.next_interval();

            // Interpreter burst
            {
                let exec = unsafe { &mut *exec_ptr };
                #[cfg(feature = "lightning")]
                for _ in 0..burst {
                    exec.step();
                }
                #[cfg(not(feature = "lightning"))]
                for _ in 0..burst {
                    let status = exec.step();
                    if status == EXEC_BREAKPOINT {
                        running.store(false, Ordering::SeqCst);
                        break;
                    }
                }
            }
            steps_in_batch += burst;
            total_interp_steps += burst as u64;

            if !running.load(Ordering::Relaxed) { break; }

            // Probe the JIT code cache
            let (pc, in_delay_slot) = {
                let exec = unsafe { &*exec_ptr };
                (exec.core.pc, exec.in_delay_slot)
            };
            let pc32 = pc as u32;

            let in_prom = (pc32 >= 0x9FC00000 && pc32 < 0xA0000000) || (pc32 >= 0xBFC00000);
            let in_exc = pc32 >= 0x80000000 && pc32 < 0x80000400;
            if in_prom || in_exc || in_delay_slot {
                probe.record_miss();
                continue;
            }

            let phys_pc = {
                let exec = unsafe { &mut *exec_ptr };
                match translate_pc(exec, pc) {
                    Some(p) => p,
                    None => { probe.record_miss(); continue; }
                }
            };

            if let Some(block) = cache.lookup(phys_pc, pc) {
                probe.record_hit();

                let block_len = block.len_mips;
                let block_tier = block.tier;
                let is_speculative = block.speculative;

                // Snapshot CPU if speculative OR verify mode
                let snapshot = if is_speculative || verify_mode {
                    let exec = unsafe { &*exec_ptr };
                    exec.tlb.clone_as_mips_tlb().map(|tlb| {
                        CpuRollbackSnapshot::capture(exec, tlb)
                    })
                } else {
                    None
                };

                // Sync and run
                {
                    let exec = unsafe { &mut *exec_ptr };
                    ctx.sync_from_executor(exec);
                }

                ctx.exit_reason = 0;
                let entry: extern "C" fn(*mut JitContext) = unsafe {
                    std::mem::transmute(block.entry)
                };
                entry(&mut ctx);

                {
                    let exec = unsafe { &mut *exec_ptr };
                    ctx.sync_to_executor(exec);

                    if ctx.exit_reason == EXIT_EXCEPTION {
                        if let Some(snap) = &snapshot {
                            if is_speculative {
                                snap.restore(exec);
                                rollbacks += 1;

                                if let Some(block) = cache.lookup_mut(phys_pc, pc) {
                                    block.hit_count += 1;
                                    block.exception_count += 1;
                                    block.stable_hits = 0;

                                    if block.exception_count >= tier_cfg.demote {
                                        if let Some(lower) = block.tier.demote() {
                                            demotions += 1;
                                            eprintln!("JIT: demote {:016x} {:?}→{:?} ({}exc)",
                                                pc, block.tier, lower, block.exception_count);
                                            recompile_block_at_tier(
                                                &mut compiler, &mut cache, exec,
                                                phys_pc, pc, lower,
                                                &mut blocks_compiled,
                                            );
                                        } else {
                                            block.speculative = false;
                                        }
                                    }
                                }
                            } else if verify_mode {
                                snap.restore(exec);
                            }
                        }
                        // Advance cp0_count for instructions that executed before the fault.
                        // ctx.pc was set to the faulting instruction by the load/store emitter.
                        let instrs_before_fault = ctx.pc.wrapping_sub(pc) / 4;
                        if instrs_before_fault > 0 {
                            let advance = exec.core.count_step.wrapping_mul(instrs_before_fault);
                            let prev = exec.core.cp0_count;
                            exec.core.cp0_count = prev.wrapping_add(advance);
                            if exec.core.cp0_compare.wrapping_sub(prev) <= advance {
                                exec.core.cp0_cause |= crate::mips_core::CAUSE_IP7;
                            }
                            exec.core.local_cycles += instrs_before_fault;
                        }
                        exec.step();
                        total_interp_steps += 1;
                        steps_in_batch += 1;
                        continue;
                    }

                    // Normal exit
                    if verify_mode {
                        if let Some(snap) = &snapshot {
                            let jit_gpr = exec.core.gpr;
                            let jit_pc = exec.core.pc;
                            let jit_hi = exec.core.hi;
                            let jit_lo = exec.core.lo;

                            snap.restore(exec);
                            for _ in 0..block_len {
                                exec.step();
                            }

                            let interp_gpr = exec.core.gpr;
                            let interp_pc = exec.core.pc;
                            let interp_hi = exec.core.hi;
                            let interp_lo = exec.core.lo;

                            let mut mismatch = false;
                            for i in 0..32 {
                                if jit_gpr[i] != interp_gpr[i] {
                                    eprintln!("JIT VERIFY FAIL at {:016x} (tier={:?}, len={}): gpr[{}] jit={:016x} interp={:016x}",
                                        pc, block_tier, block_len, i, jit_gpr[i], interp_gpr[i]);
                                    mismatch = true;
                                }
                            }
                            if jit_pc != interp_pc {
                                eprintln!("JIT VERIFY FAIL at {:016x}: pc jit={:016x} interp={:016x}",
                                    pc, jit_pc, interp_pc);
                                mismatch = true;
                            }
                            if jit_hi != interp_hi {
                                eprintln!("JIT VERIFY FAIL at {:016x}: hi jit={:016x} interp={:016x}",
                                    pc, jit_hi, interp_hi);
                                mismatch = true;
                            }
                            if jit_lo != interp_lo {
                                eprintln!("JIT VERIFY FAIL at {:016x}: lo jit={:016x} interp={:016x}",
                                    pc, jit_lo, interp_lo);
                                mismatch = true;
                            }

                            if mismatch {
                                // Check if this is a timing false positive:
                                // interpreter took an exception (PC in exception vectors)
                                // while JIT didn't. This happens because the interpreter
                                // re-run occurs at a different wall-clock time and sees
                                // different external interrupt state via the atomic.
                                let interp_pc32 = interp_pc as u32;
                                let interp_in_exc = (interp_pc32 >= 0x80000000 && interp_pc32 < 0x80000400)
                                    || interp_pc32 == 0x80000180; // general exception vector
                                let jit_pc32 = jit_pc as u32;
                                let jit_not_exc = jit_pc32 < 0x80000000 || jit_pc32 >= 0x80000400;

                                if interp_in_exc && jit_not_exc {
                                    // Timing false positive — interpreter took an interrupt
                                    // the JIT didn't see. Don't invalidate the block.
                                    // Use the interpreter's result (it's authoritative).
                                    eprintln!("JIT VERIFY: timing false positive at {:016x} (interp took exception to {:016x}), keeping block",
                                        pc, interp_pc);
                                } else {
                                    // Real codegen mismatch — dump and invalidate
                                    let instrs = trace_block(exec, pc, block_tier);
                                    eprintln!("JIT VERIFY: block at {:016x} ({} instrs):", pc, instrs.len());
                                    for (idx, (raw, d)) in instrs.iter().enumerate() {
                                        let ipc = pc.wrapping_add(idx as u64 * 4);
                                        eprintln!("  {:016x}: {:08x} op={} rs={} rt={} rd={} funct={} imm={:04x}",
                                            ipc, raw, d.op, d.rs, d.rt, d.rd, d.funct, d.imm as u16);
                                    }
                                    cache.invalidate_range(phys_pc, phys_pc + 4);
                                }
                                total_jit_instrs += block_len as u64;
                                continue;
                            }
                        }
                    }

                    // Advance cp0_count and check interrupts for the N instructions
                    // the JIT block executed. The interpreter's step() does this per-
                    // instruction; we must do it in bulk here or timing drifts and
                    // the kernel panics from missed timer interrupts.
                    {
                        let n = block_len as u64;
                        // Advance cp0_count by block_len * count_step
                        let count_advance = exec.core.count_step.wrapping_mul(n);
                        let prev = exec.core.cp0_count;
                        exec.core.cp0_count = prev.wrapping_add(count_advance);
                        if exec.core.cp0_compare.wrapping_sub(prev) <= count_advance {
                            exec.core.cp0_cause |= crate::mips_core::CAUSE_IP7;
                            exec.core.fasttick_count.fetch_add(1, Ordering::Relaxed);
                        }
                        // Credit local_cycles so the stats display shows correct MHz
                        exec.core.local_cycles += n;

                        // Merge external interrupt bits into cp0_cause so the
                        // interpreter sees them on its next step. Don't call exec.step()
                        // here — that would double-count cp0_count (the post-block
                        // advancement above already accounted for all block instructions,
                        // and step() would add yet another count_step tick per interrupt).
                        let pending = exec.core.interrupts.load(Ordering::Relaxed);
                        if pending != 0 {
                            use crate::mips_core::{CAUSE_IP2, CAUSE_IP3, CAUSE_IP4, CAUSE_IP5, CAUSE_IP6};
                            let ext_mask = CAUSE_IP2 | CAUSE_IP3 | CAUSE_IP4 | CAUSE_IP5 | CAUSE_IP6;
                            exec.core.cp0_cause = (exec.core.cp0_cause & !ext_mask)
                                | (pending as u32 & ext_mask);
                        }
                    }

                    // Update stats and check for promotion
                    if let Some(block) = cache.lookup_mut(phys_pc, pc) {
                        block.hit_count += 1;
                        block.stable_hits += 1;
                        block.exception_count = 0;

                        if block.speculative && block.stable_hits >= tier_cfg.stable {
                            block.speculative = false;
                        }

                        if !block.speculative && block.stable_hits >= tier_cfg.promote {
                            if let Some(next) = block.tier.promote().filter(|t| *t <= max_tier) {
                                promotions += 1;
                                eprintln!("JIT: promote {:016x} {:?}→{:?} ({}hits)",
                                    pc, block.tier, next, block.hit_count);
                                recompile_block_at_tier(
                                    &mut compiler, &mut cache, exec,
                                    phys_pc, pc, next,
                                    &mut blocks_compiled,
                                );
                            }
                        }
                    }

                    total_jit_instrs += block_len as u64;
                    steps_in_batch += block_len;
                }
            } else {
                probe.record_miss();
                // Cache miss — compile at Alu tier
                let exec = unsafe { &mut *exec_ptr };
                let instrs = trace_block(exec, pc, BlockTier::Alu);
                if !instrs.is_empty() {
                    if let Some(mut block) = compiler.compile_block(&instrs, pc, BlockTier::Alu) {
                        block.phys_addr = phys_pc;
                        cache.insert(phys_pc, pc, block);
                        blocks_compiled += 1;
                        probe.set_cache_size(cache.len() as u32);
                        if blocks_compiled <= 10 || blocks_compiled % 500 == 0 {
                            eprintln!("JIT: compiled #{} at {:016x} ({} instrs, tier=Alu, cache={})",
                                blocks_compiled, pc, instrs.len(), cache.len());
                        }
                    }
                }
            }
        }

        {
            let exec = unsafe { &mut *exec_ptr };
            exec.flush_cycles();
        }

        // Write trace record at 100K instruction milestones.
        // Both JIT and interpreter runs log at the same milestones so
        // records align for offline comparison.
        if let Some(tw) = &mut trace_writer {
            let total = total_interp_steps + total_jit_instrs;
            let prev_total = total.saturating_sub(BATCH_SIZE as u64);
            let milestone = 100_000u64;
            if total / milestone != prev_total / milestone {
                let exec = unsafe { &*exec_ptr };
                tw.write_record(&TraceRecord {
                    insn_count: (total / milestone) * milestone,
                    pc: exec.core.pc,
                    cp0_count: exec.core.cp0_count,
                    cp0_status: exec.core.cp0_status,
                    cp0_cause: exec.core.cp0_cause,
                    in_delay_slot: exec.in_delay_slot as u8,
                    _pad: [0; 7],
                    gpr_hash: TraceRecord::hash_gprs(&exec.core.gpr),
                });
            }
        }

        let total = total_interp_steps + total_jit_instrs;
        if total % 10000000 < BATCH_SIZE as u64 {
            let exec = unsafe { &*exec_ptr };
            let jit_pct = if total > 0 { total_jit_instrs as f64 / total as f64 * 100.0 } else { 0.0 };
            let effective_probe = {
                let cf = if probe.cache_size > 100 {
                    1.0f32 / (probe.cache_size as f32 / 100.0).sqrt().max(1.0)
                } else { 1.0 };
                ((probe.interval as f32 * cf) as u32).clamp(probe.min_interval, probe.max_interval)
            };
            eprintln!("JIT: {} total ({:.1}% jit), {} blocks, {}↑ {}↓ {}⟲, probe={}(eff {}), pc={:016x}",
                total, jit_pct, blocks_compiled, promotions, demotions, rollbacks,
                probe.interval, effective_probe, exec.core.pc);
        }
    }

    {
        let exec = unsafe { &mut *exec_ptr };
        exec.flush_cycles();
    }
    let total = total_interp_steps + total_jit_instrs;
    let jit_pct = if total > 0 { total_jit_instrs as f64 / total as f64 * 100.0 } else { 0.0 };
    eprintln!("JIT: shutdown. {} blocks, {} jit / {} interp / {} total ({:.1}% jit), {}↑ {}↓ {}⟲, final_probe={}",
        blocks_compiled, total_jit_instrs, total_interp_steps, total,
        jit_pct, promotions, demotions, rollbacks, probe.interval);

    // Save profile: all blocks above Alu tier
    let profile_entries: Vec<ProfileEntry> = cache.iter()
        .filter(|(_, block)| block.tier > BlockTier::Alu)
        .map(|(&(phys_pc, _virt_pc), block)| ProfileEntry {
            phys_pc,
            virt_pc: block.virt_addr,
            tier: block.tier,
        })
        .collect();
    if !profile_entries.is_empty() {
        if let Err(e) = profile::save_profile(&profile_entries) {
            eprintln!("JIT profile: save failed: {}", e);
        }
    }
}

/// Recompile a block at a different tier, replacing the existing cache entry.
fn recompile_block_at_tier<T: Tlb, C: MipsCache>(
    compiler: &mut BlockCompiler,
    cache: &mut CodeCache,
    exec: &mut MipsExecutor<T, C>,
    phys_pc: u64,
    virt_pc: u64,
    tier: BlockTier,
    blocks_compiled: &mut u64,
) {
    let instrs = trace_block(exec, virt_pc, tier);
    if !instrs.is_empty() {
        if let Some(mut block) = compiler.compile_block(&instrs, virt_pc, tier) {
            block.phys_addr = phys_pc;
            cache.replace(phys_pc, virt_pc, block);
            *blocks_compiled += 1;
        }
    }
}

fn interpreter_loop<T: Tlb, C: MipsCache>(
    exec: &mut MipsExecutor<T, C>,
    running: &AtomicBool,
) {
    let mut trace_writer = TraceWriter::from_env();
    let mut total_steps: u64 = 0;

    while running.load(Ordering::Relaxed) {
        #[cfg(feature = "lightning")]
        for _ in 0..1000 {
            exec.step(); exec.step(); exec.step(); exec.step(); exec.step();
            exec.step(); exec.step(); exec.step(); exec.step(); exec.step();
        }
        #[cfg(not(feature = "lightning"))]
        for _ in 0..1000 {
            let status = exec.step();
            if status == EXEC_BREAKPOINT {
                running.store(false, Ordering::SeqCst);
                break;
            }
        }
        total_steps += 10000;
        exec.flush_cycles();

        if let Some(tw) = &mut trace_writer {
            let prev = total_steps.saturating_sub(10000);
            let milestone = 100_000u64;
            if total_steps / milestone != prev / milestone {
                tw.write_record(&TraceRecord {
                    insn_count: (total_steps / milestone) * milestone,
                    pc: exec.core.pc,
                    cp0_count: exec.core.cp0_count,
                    cp0_status: exec.core.cp0_status,
                    cp0_cause: exec.core.cp0_cause,
                    in_delay_slot: exec.in_delay_slot as u8,
                    _pad: [0; 7],
                    gpr_hash: TraceRecord::hash_gprs(&exec.core.gpr),
                });
            }
        }
    }
}

fn translate_pc<T: Tlb, C: MipsCache>(
    exec: &mut MipsExecutor<T, C>,
    virt_pc: u64,
) -> Option<u64> {
    // Use debug_translate (translate_impl::<true>) to avoid CP0 side effects.
    // The non-debug path writes cp0_badvaddr, cp0_entryhi, cp0_context,
    // cp0_xcontext on TLB miss — corrupting state that the next real TLB
    // exception handler depends on. We only need the physical address for
    // cache lookup; we don't want to touch CP0 state.
    let result = exec.debug_translate(virt_pc);
    if result.is_exception() { None } else { Some(result.phys as u64) }
}

fn trace_block<T: Tlb, C: MipsCache>(
    exec: &mut MipsExecutor<T, C>,
    start_pc: u64,
    tier: BlockTier,
) -> Vec<(u32, DecodedInstr)> {
    let mut instrs = Vec::with_capacity(MAX_BLOCK_LEN);
    let mut pc = start_pc;

    for _ in 0..MAX_BLOCK_LEN {
        let raw = match exec.debug_fetch_instr(pc) {
            Ok(w) => w,
            Err(_) => break,
        };

        let mut d = DecodedInstr::default();
        d.raw = raw;
        decode_into::<T, C>(&mut d);

        if !is_compilable_for_tier(&d, tier) { break; }

        let is_branch = is_branch_or_jump(&d);
        // Terminate Full-tier blocks after each store to keep blocks short.
        // Long blocks with multiple load/store helper calls create complex CFG
        // (ok_block/exc_block diamonds) that triggers Cranelift regalloc2 issues
        // on x86_64, causing rare but fatal codegen corruption.
        let is_store = tier == BlockTier::Full && is_compilable_store(&d);
        instrs.push((raw, d));

        if is_store {
            break;
        }

        if is_branch {
            pc = pc.wrapping_add(4);
            let mut delay_ok = false;
            if let Ok(delay_raw) = exec.debug_fetch_instr(pc) {
                let mut delay_d = DecodedInstr::default();
                delay_d.raw = delay_raw;
                decode_into::<T, C>(&mut delay_d);
                // Exclude stores from delay slots: if the delay slot faults,
                // the JIT exception path loses delay-slot context (sync_to clears
                // in_delay_slot), so handle_exception sets wrong cp0_epc/BD bit,
                // and on ERET the branch is permanently skipped → crash.
                if is_compilable_for_tier(&delay_d, tier) && !is_compilable_store(&delay_d) {
                    instrs.push((delay_raw, delay_d));
                    delay_ok = true;
                }
            }
            if !delay_ok { instrs.pop(); }
            break;
        }

        pc = pc.wrapping_add(4);
    }

    instrs
}

fn is_compilable_for_tier(d: &DecodedInstr, tier: BlockTier) -> bool {
    if is_compilable_alu(d) || is_branch_or_jump(d) { return true; }
    match tier {
        BlockTier::Alu => false,
        BlockTier::Loads => is_compilable_load(d),
        BlockTier::Full => is_compilable_load(d) || is_compilable_store(d),
    }
}

fn is_compilable_alu(d: &DecodedInstr) -> bool {
    use crate::mips_isa::*;
    match d.op as u32 {
        OP_SPECIAL => matches!(d.funct as u32,
            FUNCT_SLL | FUNCT_SRL | FUNCT_SRA |
            FUNCT_SLLV | FUNCT_SRLV | FUNCT_SRAV |
            FUNCT_MOVZ | FUNCT_MOVN |
            FUNCT_MFHI | FUNCT_MTHI | FUNCT_MFLO | FUNCT_MTLO |
            FUNCT_MULT | FUNCT_MULTU | FUNCT_DIV | FUNCT_DIVU |
            FUNCT_DMULT | FUNCT_DMULTU | FUNCT_DDIV | FUNCT_DDIVU |
            FUNCT_ADDU | FUNCT_SUBU | FUNCT_AND | FUNCT_OR |
            FUNCT_XOR | FUNCT_NOR | FUNCT_SLT | FUNCT_SLTU |
            FUNCT_DADDU | FUNCT_DSUBU |
            FUNCT_DSLL | FUNCT_DSRL | FUNCT_DSRA |
            FUNCT_DSLL32 | FUNCT_DSRL32 | FUNCT_DSRA32 |
            FUNCT_DSLLV | FUNCT_DSRLV | FUNCT_DSRAV |
            FUNCT_SYNC
        ),
        OP_ADDIU | OP_DADDIU | OP_SLTI | OP_SLTIU |
        OP_ANDI | OP_ORI | OP_XORI | OP_LUI => true,
        _ => false,
    }
}

fn is_compilable_load(d: &DecodedInstr) -> bool {
    use crate::mips_isa::*;
    matches!(d.op as u32,
        OP_LB | OP_LBU | OP_LH | OP_LHU | OP_LW | OP_LWU | OP_LD
    )
}

fn is_compilable_store(d: &DecodedInstr) -> bool {
    use crate::mips_isa::*;
    matches!(d.op as u32,
        OP_SB | OP_SH | OP_SW | OP_SD
    )
}

fn is_branch_or_jump(d: &DecodedInstr) -> bool {
    use crate::mips_isa::*;
    match d.op as u32 {
        OP_BEQ | OP_BNE | OP_BLEZ | OP_BGTZ => true,
        OP_J | OP_JAL => true,
        OP_SPECIAL => matches!(d.funct as u32, FUNCT_JR),
        _ => false,
    }
}
