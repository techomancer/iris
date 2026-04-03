//! JIT dispatch loop: interpreter-first architecture with inline cache probes.
//!
//! The interpreter runs in tight batches. Every PROBE_INTERVAL steps within
//! the batch, we check if the current PC has a compiled block. If so, we
//! execute it and return to the interpreter. This gives high JIT hit rates
//! while keeping zero overhead on most interpreter steps.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::mips_exec::{MipsExecutor, DecodedInstr, EXEC_BREAKPOINT, decode_into};
use crate::mips_tlb::{Tlb, AccessType};
use crate::mips_cache_v2::MipsCache;

use super::cache::CodeCache;
use super::compiler::BlockCompiler;
use super::context::{JitContext, EXIT_NORMAL, EXIT_EXCEPTION};
use super::helpers::HelperPtrs;

const MAX_BLOCK_LEN: usize = 64;

/// How many interpreter steps between cache probes within a batch.
const PROBE_INTERVAL: u32 = 1000;

/// How many interpreter steps in one outer batch.
const BATCH_SIZE: u32 = 10000;

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

    // CRITICAL: Convert &mut to raw pointer. We must never hold &mut MipsExecutor
    // across a JIT block call, because the JIT's memory helpers create their own
    // &mut from the raw pointer. Two simultaneous &mut is UB, and with lto="fat"
    // LLVM exploits the noalias guarantee to cache/hoist loads across the call,
    // causing stale TLB/cache/CP0 state and kernel panics.
    let exec_ptr: *mut MipsExecutor<T, C> = exec as *mut _;

    eprintln!("JIT: enabled (interpreter-first, probe every {} steps)", PROBE_INTERVAL);
    let helpers = HelperPtrs::new::<T, C>();
    let mut compiler = BlockCompiler::new(&helpers);
    let mut cache = CodeCache::new();
    let mut ctx = JitContext::new();
    ctx.executor_ptr = exec_ptr as u64;

    let mut total_jit_instrs: u64 = 0;
    let mut total_interp_steps: u64 = 0;
    let mut blocks_compiled: u64 = 0;

    while running.load(Ordering::Relaxed) {
        let mut steps_in_batch: u32 = 0;

        while steps_in_batch < BATCH_SIZE {
            // Borrow exec for interpreter batch — no JIT call happens here
            {
                let exec = unsafe { &mut *exec_ptr };
                #[cfg(feature = "lightning")]
                for _ in 0..PROBE_INTERVAL {
                    exec.step();
                }
                #[cfg(not(feature = "lightning"))]
                for _ in 0..PROBE_INTERVAL {
                    let status = exec.step();
                    if status == EXEC_BREAKPOINT {
                        running.store(false, Ordering::SeqCst);
                        break;
                    }
                }
            } // &mut exec dropped here
            steps_in_batch += PROBE_INTERVAL;

            if !running.load(Ordering::Relaxed) { break; }

            // Probe the JIT code cache — borrow briefly for reads
            let (pc, in_delay_slot) = {
                let exec = unsafe { &*exec_ptr };
                (exec.core.pc, exec.in_delay_slot)
            };
            let pc32 = pc as u32;

            let in_prom = (pc32 >= 0x9FC00000 && pc32 < 0xA0000000) || (pc32 >= 0xBFC00000);
            let in_exc = pc32 >= 0x80000000 && pc32 < 0x80000400;
            if in_prom || in_exc || in_delay_slot {
                continue;
            }

            let phys_pc = {
                let exec = unsafe { &mut *exec_ptr };
                match translate_pc(exec, pc) {
                    Some(p) => p,
                    None => continue,
                }
            };

            if let Some(block) = cache.lookup(phys_pc) {
                // Cache hit — execute compiled block.
                // NO &mut MipsExecutor exists during the JIT call.
                let entry: extern "C" fn(*mut JitContext) = unsafe {
                    std::mem::transmute(block.entry)
                };
                let block_len = block.len_mips;

                {
                    let exec = unsafe { &mut *exec_ptr };
                    ctx.sync_from_executor(exec);
                } // &mut dropped before JIT call

                ctx.exit_reason = 0;
                entry(&mut ctx); // Helpers create their own &mut from exec_ptr — no aliasing

                {
                    let exec = unsafe { &mut *exec_ptr };
                    ctx.sync_to_executor(exec);

                    if ctx.exit_reason == EXIT_EXCEPTION {
                        // A load/store hit a TLB miss or other exception.
                        // ctx.pc has the faulting instruction's PC (stored before the helper call).
                        // GPRs are current (stored by the exc_block).
                        // Re-execute the faulting instruction through the interpreter,
                        // which will handle the exception properly (set EPC, jump to handler).
                        exec.step();
                        steps_in_batch += 1;
                        // Reset exit_reason for next block
                        ctx.exit_reason = 0;
                    } else {
                        // Normal exit — advance cp0_count per-instruction
                        for _ in 0..block_len {
                            let prev = exec.core.cp0_count;
                            exec.core.cp0_count = prev.wrapping_add(exec.core.count_step) & 0x0000_FFFF_FFFF_FFFF;
                            if exec.core.cp0_compare != 0 && prev < exec.core.cp0_compare && exec.core.cp0_count >= exec.core.cp0_compare {
                                exec.core.cp0_cause |= crate::mips_core::CAUSE_IP7;
                                exec.core.fasttick_count.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        exec.local_cycles += block_len as u64;
                        steps_in_batch += block_len;
                        total_jit_instrs += block_len as u64;
                    }
                } // &mut dropped
            } else {
                // Cache miss — try to compile
                let exec = unsafe { &mut *exec_ptr };
                let instrs = trace_block(exec, pc);
                if !instrs.is_empty() {
                    if let Some(mut block) = compiler.compile_block(&instrs, pc) {
                        block.phys_addr = phys_pc;
                        cache.insert(phys_pc, block);
                        blocks_compiled += 1;
                        if blocks_compiled <= 10 || blocks_compiled % 500 == 0 {
                            eprintln!("JIT: compiled #{} at {:016x} ({} instrs, cache={})",
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
        total_interp_steps += steps_in_batch as u64;

        if total_interp_steps % 10000000 < BATCH_SIZE as u64 {
            let exec = unsafe { &*exec_ptr };
            eprintln!("JIT: {} steps, {} JIT instrs ({:.1}%), {} blocks, pc={:016x}",
                total_interp_steps, total_jit_instrs,
                if total_interp_steps > 0 { total_jit_instrs as f64 / total_interp_steps as f64 * 100.0 } else { 0.0 },
                blocks_compiled, exec.core.pc);
        }
    }

    {
        let exec = unsafe { &mut *exec_ptr };
        exec.flush_cycles();
    }
    eprintln!("JIT: shutdown. {} blocks, {} JIT instrs / {} total steps ({:.1}%)",
        blocks_compiled, total_jit_instrs, total_interp_steps,
        if total_interp_steps > 0 { total_jit_instrs as f64 / total_interp_steps as f64 * 100.0 } else { 0.0 });
}

fn interpreter_loop<T: Tlb, C: MipsCache>(
    exec: &mut MipsExecutor<T, C>,
    running: &AtomicBool,
) {
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
        exec.flush_cycles();
    }
}

fn translate_pc<T: Tlb, C: MipsCache>(
    exec: &mut MipsExecutor<T, C>,
    virt_pc: u64,
) -> Option<u64> {
    let result = (exec.translate_fn)(exec, virt_pc, AccessType::Fetch);
    if result.is_exception() { None } else { Some(result.phys as u64) }
}

fn trace_block<T: Tlb, C: MipsCache>(
    exec: &mut MipsExecutor<T, C>,
    start_pc: u64,
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

        if !is_compilable(&d) { break; }

        let is_branch = is_branch_or_jump(&d);
        instrs.push((raw, d));

        if is_branch {
            pc = pc.wrapping_add(4);
            let mut delay_ok = false;
            if let Ok(delay_raw) = exec.debug_fetch_instr(pc) {
                let mut delay_d = DecodedInstr::default();
                delay_d.raw = delay_raw;
                decode_into::<T, C>(&mut delay_d);
                if is_compilable_alu(&delay_d) || is_compilable_mem(&delay_d) {
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

fn is_compilable(d: &DecodedInstr) -> bool {
    is_compilable_alu(d) || is_compilable_mem(d) || is_branch_or_jump(d)
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

fn is_compilable_mem(d: &DecodedInstr) -> bool {
    use crate::mips_isa::*;
    matches!(d.op as u32,
        OP_LB | OP_LBU | OP_LH | OP_LHU | OP_LW | OP_LWU | OP_LD |
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
