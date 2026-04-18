//! JitContext: `#[repr(C)]` bridge struct between JIT-compiled code and emulator state.
//!
//! Contains the hot subset of MipsCore and MipsExecutor state that compiled blocks
//! read and write directly. Synced to/from the interpreter before and after JIT execution.

use crate::mips_core::NanoTlbEntry;
use crate::mips_exec::MipsExecutor;
use crate::mips_tlb::Tlb;
use crate::mips_cache_v2::MipsCache;

// Exit reason constants set by JIT code before returning to dispatch.
pub const EXIT_NORMAL: u32 = 0;
pub const EXIT_INTERPRET: u32 = 1;
pub const EXIT_EXCEPTION: u32 = 2;
pub const EXIT_INTERRUPT_CHECK: u32 = 3;
pub const EXIT_HALT: u32 = 4;

/// Max stores we can speculatively track per block. Exceeding this forces the
/// block to be non-speculative (disables rollback for stores past this limit).
pub const WRITE_LOG_CAP: usize = 128;

/// Single entry in the speculative store write log. Records the pre-store
/// value at `addr` so rollback can restore it if the block exceptions.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct WriteLogEntry {
    pub addr: u64,
    pub old_val: u64,
    pub size: u8,
    pub _pad: [u8; 7],
}

impl WriteLogEntry {
    pub const fn empty() -> Self {
        Self { addr: 0, old_val: 0, size: 0, _pad: [0; 7] }
    }
}

#[repr(C)]
pub struct JitContext {
    // General purpose registers
    pub gpr: [u64; 32],

    // Special registers
    pub pc: u64,
    pub hi: u64,
    pub lo: u64,

    // FPU registers
    pub fpr: [u64; 32],
    pub fpu_fcsr: u32,

    // CP0 state (hot subset for interrupt/exception checking)
    pub cp0_status: u32,
    pub cp0_cause: u32,
    pub cp0_epc: u64,
    pub cp0_count: u64,
    pub cp0_compare: u64,
    pub count_step: u64,
    pub cp0_badvaddr: u64,

    // Nano-TLB (3 entries: Fetch/Read/Write)
    pub nanotlb: [NanoTlbEntry; 3],

    // Delay slot state
    pub in_delay_slot: bool,
    pub delay_slot_target: u64,

    // Interrupt handling (cached from executor)
    pub cached_pending: u64,
    pub local_cycles: u64,

    // JIT dispatch state
    pub exit_reason: u32,
    pub block_instrs_executed: u32,

    // Type-erased pointer to MipsExecutor — used by memory helper callouts
    pub executor_ptr: u64,
    // Exception status from failed memory access (set by helpers)
    pub exception_status: u32,
    _pad0: u32,

    // Speculative store write log. Each entry records the pre-store value at
    // an address. On block rollback (speculative exception), replay in reverse
    // to restore memory. On normal exit, reset write_log_len to 0.
    pub write_log_len: u32,
    _pad1: u32,
    pub write_log: [WriteLogEntry; WRITE_LOG_CAP],
}

impl JitContext {
    pub fn new() -> Self {
        Self {
            gpr: [0; 32],
            pc: 0,
            hi: 0,
            lo: 0,
            fpr: [0; 32],
            fpu_fcsr: 0,
            cp0_status: 0,
            cp0_cause: 0,
            cp0_epc: 0,
            cp0_count: 0,
            cp0_compare: 0,
            count_step: 0,
            cp0_badvaddr: 0,
            nanotlb: [NanoTlbEntry::default(); 3],
            in_delay_slot: false,
            delay_slot_target: 0,
            cached_pending: 0,
            local_cycles: 0,
            exit_reason: EXIT_NORMAL,
            block_instrs_executed: 0,
            executor_ptr: 0,
            exception_status: 0,
            _pad0: 0,
            write_log_len: 0,
            _pad1: 0,
            write_log: [WriteLogEntry::empty(); WRITE_LOG_CAP],
        }
    }

    /// Byte offset of `gpr[i]` from the start of JitContext.
    pub fn gpr_offset(i: usize) -> i32 {
        (std::mem::offset_of!(JitContext, gpr) + i * 8) as i32
    }

    pub fn hi_offset() -> i32 { std::mem::offset_of!(JitContext, hi) as i32 }
    pub fn lo_offset() -> i32 { std::mem::offset_of!(JitContext, lo) as i32 }
    pub fn pc_offset() -> i32 { std::mem::offset_of!(JitContext, pc) as i32 }
    pub fn exit_reason_offset() -> i32 { std::mem::offset_of!(JitContext, exit_reason) as i32 }
    pub fn block_instrs_offset() -> i32 { std::mem::offset_of!(JitContext, block_instrs_executed) as i32 }
    pub fn executor_ptr_offset() -> i32 { std::mem::offset_of!(JitContext, executor_ptr) as i32 }
    pub fn exception_status_offset() -> i32 { std::mem::offset_of!(JitContext, exception_status) as i32 }

    /// Copy emulator state into JitContext.
    pub fn sync_from_executor<T: Tlb, C: MipsCache>(
        &mut self,
        exec: &MipsExecutor<T, C>,
    ) {
        self.gpr = exec.core.gpr;
        self.pc = exec.core.pc;
        self.hi = exec.core.hi;
        self.lo = exec.core.lo;
        self.fpr = exec.core.fpr;
        self.fpu_fcsr = exec.core.fpu_fcsr;
        self.cp0_status = exec.core.cp0_status;
        self.cp0_cause = exec.core.cp0_cause;
        self.cp0_epc = exec.core.cp0_epc;
        self.cp0_count = exec.core.cp0_count;
        self.cp0_compare = exec.core.cp0_compare;
        self.count_step = exec.core.count_step;
        self.cp0_badvaddr = exec.core.cp0_badvaddr;
        self.nanotlb = exec.core.nanotlb;
        self.in_delay_slot = exec.in_delay_slot;
        self.delay_slot_target = exec.delay_slot_target;
        self.cached_pending = exec.cached_pending;
        self.local_cycles = exec.core.local_cycles;
    }

    /// Copy JitContext state back to the emulator.
    ///
    /// ONLY writes back fields that compiled blocks actually modify (GPRs, hi, lo, PC).
    /// Fields managed by the interpreter or helpers (cp0_*, nanotlb, fpr) are NOT
    /// written back — they're updated directly on the executor by helpers/interpreter.
    pub fn sync_to_executor<T: Tlb, C: MipsCache>(
        &self,
        exec: &mut MipsExecutor<T, C>,
    ) {
        // These are modified by compiled code (stored in the block epilogue)
        exec.core.gpr = self.gpr;
        exec.core.pc = self.pc;
        exec.core.hi = self.hi;
        exec.core.lo = self.lo;

        // Compiled blocks handle delay slots internally (the branch emitter
        // computes the target, emits the delay slot, and sets the exit PC).
        // Clear the interpreter's delay slot state so subsequent exec.step()
        // calls don't jump to a stale target.
        exec.in_delay_slot = false;
        exec.delay_slot_target = 0;

        // DO NOT write back: cp0_status, cp0_cause, cp0_epc, cp0_badvaddr,
        // cp0_count, cp0_compare, count_step, nanotlb, fpr, fpu_fcsr —
        // these are managed by the interpreter and memory helpers directly
        // on the executor. Writing them back would clobber changes made by
        // exception handlers and TLB fill operations.
    }
}
