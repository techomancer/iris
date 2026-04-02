//! JIT dispatch loop: traces, compiles, and executes MIPS basic blocks.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::mips_exec::{MipsExecutor, DecodedInstr, ExecStatus, EXEC_BREAKPOINT, EXEC_IS_EXCEPTION, decode_into};
use crate::mips_tlb::{Tlb, AccessType};
use crate::mips_cache_v2::MipsCache;

use super::cache::CodeCache;
use super::compiler::BlockCompiler;
use super::context::{JitContext, EXIT_NORMAL, EXIT_INTERPRET};

/// Maximum number of instructions per compiled block.
const MAX_BLOCK_LEN: usize = 64;

/// Run the JIT dispatch loop. Replaces the inner `while running` loop in MipsCpu::start().
pub fn run_jit_dispatch<T: Tlb, C: MipsCache>(
    exec: &mut MipsExecutor<T, C>,
    running: &AtomicBool,
) {
    let mut compiler = BlockCompiler::new();
    let mut cache = CodeCache::new();
    let mut ctx = JitContext::new();
    let mut steps_since_flush: u32 = 0;

    while running.load(Ordering::Relaxed) {
        let pc = exec.core.pc;

        // Translate PC to physical address for cache lookup
        let phys_pc = match translate_pc(exec, pc) {
            Some(p) => p,
            None => {
                // Translation failed — let interpreter handle the exception
                exec.step();
                steps_since_flush += 1;
                if steps_since_flush >= 1000 {
                    exec.flush_cycles();
                    steps_since_flush = 0;
                }
                continue;
            }
        };

        if let Some(block) = cache.lookup(phys_pc) {
            // Cache hit — execute compiled block
            let entry: extern "C" fn(*mut JitContext) = unsafe {
                std::mem::transmute(block.entry)
            };
            let block_len = block.len_mips;

            ctx.sync_from_executor(exec);
            entry(&mut ctx);
            ctx.sync_to_executor(exec);

            // Advance cp0_count by block length
            let count_advance = exec.core.count_step.wrapping_mul(block_len as u64);
            let prev = exec.core.cp0_count;
            exec.core.cp0_count = prev.wrapping_add(count_advance) & 0x0000_FFFF_FFFF_FFFF;
            if exec.core.cp0_compare != 0 && prev < exec.core.cp0_compare && exec.core.cp0_count >= exec.core.cp0_compare {
                exec.core.cp0_cause |= crate::mips_core::CAUSE_IP7;
            }

            exec.local_cycles += block_len as u64;
            steps_since_flush += block_len;

            match ctx.exit_reason {
                EXIT_NORMAL => {}
                EXIT_INTERPRET => {
                    // The block ended before an uncompilable instruction.
                    // PC is set to the uncompilable instruction — interpret it.
                    exec.step();
                    steps_since_flush += 1;
                }
                _ => {}
            }

            // Check interrupts between blocks
            exec.cached_pending = unsafe {
                let ptr = std::ptr::addr_of!(exec.core.interrupts) as *const std::sync::atomic::AtomicU64;
                (*ptr).load(Ordering::Relaxed)
            };
            if (exec.cached_pending | exec.core.cp0_cause as u64) != 0 {
                // Let the interpreter handle the interrupt
                exec.step();
                steps_since_flush += 1;
            }
        } else {
            // Cache miss — try to trace and compile a block
            let instrs = trace_block(exec, pc);
            if instrs.is_empty() {
                // First instruction isn't compilable — interpret it
                exec.step();
                steps_since_flush += 1;
            } else {
                if let Some(mut block) = compiler.compile_block(&instrs, pc) {
                    block.phys_addr = phys_pc;
                    cache.insert(phys_pc, block);
                    // Next iteration will hit the cache
                } else {
                    // Compilation failed — interpret one instruction
                    exec.step();
                    steps_since_flush += 1;
                }
            }
        }

        if steps_since_flush >= 1000 {
            exec.flush_cycles();
            steps_since_flush = 0;
        }
    }

    exec.flush_cycles();
}

/// Translate a virtual PC to a physical address for code cache lookup.
fn translate_pc<T: Tlb, C: MipsCache>(
    exec: &mut MipsExecutor<T, C>,
    virt_pc: u64,
) -> Option<u64> {
    let result = (exec.translate_fn)(exec, virt_pc, AccessType::Fetch);
    if result.is_exception() {
        None
    } else {
        Some(result.phys as u64)
    }
}

/// Trace a basic block: walk instructions from `virt_pc`, collecting compilable
/// instructions until we hit a non-compilable op, a branch, or the max block size.
fn trace_block<T: Tlb, C: MipsCache>(
    exec: &mut MipsExecutor<T, C>,
    start_pc: u64,
) -> Vec<(u32, DecodedInstr)> {
    let mut instrs = Vec::with_capacity(MAX_BLOCK_LEN);
    let mut pc = start_pc;

    for _ in 0..MAX_BLOCK_LEN {
        // Fetch instruction word without side effects
        let raw = match exec.debug_fetch_instr(pc) {
            Ok(w) => w,
            Err(_) => break, // fetch failed (unmapped, etc.)
        };

        // Decode into a DecodedInstr
        let mut d = DecodedInstr::default();
        d.raw = raw;
        decode_into::<T, C>(&mut d);

        // Check if this instruction is compilable (ALU only in Phase 2)
        if !is_compilable(&d) {
            break;
        }

        instrs.push((raw, d));
        pc = pc.wrapping_add(4);
    }

    instrs
}

/// Returns true if the instruction can be compiled by the JIT (Phase 2: integer ALU only).
fn is_compilable(d: &DecodedInstr) -> bool {
    use crate::mips_isa::*;
    let op = d.op as u32;
    let funct = d.funct as u32;

    match op {
        OP_SPECIAL => matches!(funct,
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
