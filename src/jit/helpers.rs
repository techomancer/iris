//! `extern "C"` bridge functions called by JIT-compiled code for memory access.
//!
//! CRITICAL: All pointer casts use `std::hint::black_box` to prevent LLVM from
//! tracking pointer provenance through LTO. Without this, LLVM can prove the
//! exec_ptr derives from a &mut in the dispatch loop and apply noalias
//! optimizations that cause stale reads.

use super::context::{JitContext, EXIT_EXCEPTION};
use crate::mips_exec::{MipsExecutor, EXEC_COMPLETE};
use crate::mips_tlb::Tlb;
use crate::mips_cache_v2::MipsCache;

/// Opaque cast that defeats LLVM's alias analysis and pointer provenance tracking.
/// `#[inline(never)]` ensures LLVM can't see through this to recover provenance.
#[inline(never)]
fn opaque_exec<T: Tlb, C: MipsCache>(ptr: *mut u8) -> *mut MipsExecutor<T, C> {
    std::hint::black_box(ptr as *mut MipsExecutor<T, C>)
}

#[inline(never)]
fn opaque_ctx(ptr: *mut JitContext) -> *mut JitContext {
    std::hint::black_box(ptr)
}

// ─── Read helpers ────────────────────────────────────────────────────────────

pub extern "C" fn jit_read_u8<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    match exec.read_data::<1>(virt_addr) {
        Ok(value) => value,
        Err(status) => { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; 0 }
    }
}

pub extern "C" fn jit_read_u16<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    match exec.read_data::<2>(virt_addr) {
        Ok(value) => value,
        Err(status) => { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; 0 }
    }
}

pub extern "C" fn jit_read_u32<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    match exec.read_data::<4>(virt_addr) {
        Ok(value) => value,
        Err(status) => { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; 0 }
    }
}

pub extern "C" fn jit_read_u64<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    match exec.read_data::<8>(virt_addr) {
        Ok(value) => value,
        Err(status) => { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; 0 }
    }
}

// ─── Write helpers ───────────────────────────────────────────────────────────

pub extern "C" fn jit_write_u8<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64, value: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    let status = exec.write_data::<1>(virt_addr, value);
    if status != EXEC_COMPLETE { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; }
    0
}

pub extern "C" fn jit_write_u16<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64, value: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    let status = exec.write_data::<2>(virt_addr, value);
    if status != EXEC_COMPLETE { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; }
    0
}

pub extern "C" fn jit_write_u32<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64, value: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    let status = exec.write_data::<4>(virt_addr, value);
    if status != EXEC_COMPLETE { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; }
    0
}

pub extern "C" fn jit_write_u64<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64, value: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    let status = exec.write_data::<8>(virt_addr, value);
    if status != EXEC_COMPLETE { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; }
    0
}

// ─── Interpreter fallback ────────────────────────────────────────────────────

/// Execute one interpreter step for a delay slot that can't be compiled at
/// the current JIT tier. The caller (JIT block) has already flushed modified
/// GPRs and set ctx.pc to the delay slot PC. This function:
/// 1. Syncs JitContext → executor (so interpreter sees JIT's register state)
/// 2. Calls exec.step() (executes the instruction + full bookkeeping)
/// 3. Syncs executor → JitContext (so JIT sees the result, e.g. loaded value)
pub extern "C" fn jit_interp_one_step<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    ctx.sync_to_executor(exec);
    exec.step();
    ctx.sync_from_executor(exec);
    0
}

// ─── CP0 helpers ─────────────────────────────────────────────────────────────

/// MFC0: read CP0 register `rd` as 32-bit sign-extended to 64.
/// Random (rd=1) depends on cycle count; flush before read.
pub extern "C" fn jit_mfc0<T: Tlb, C: MipsCache>(
    _ctx_ptr: *mut JitContext, exec_ptr: *mut u8, rd: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let rd_u32 = rd as u32;
    if rd_u32 == 1 { exec.flush_cycles(); }
    let v = exec.core.read_cp0(rd_u32);
    // sign-extend 32→64 to match interpreter exec_mfc0
    v as u32 as i32 as i64 as u64
}

/// DMFC0: read CP0 register `rd` as full 64-bit value.
pub extern "C" fn jit_dmfc0<T: Tlb, C: MipsCache>(
    _ctx_ptr: *mut JitContext, exec_ptr: *mut u8, rd: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let rd_u32 = rd as u32;
    if rd_u32 == 1 { exec.flush_cycles(); }
    exec.core.read_cp0(rd_u32)
}

/// MTC0: write low 32 bits of `value` (sign-extended) into CP0 register `rd`.
/// write_cp0 handles side effects (Status→translate_fn, Compare→timer, etc.).
pub extern "C" fn jit_mtc0<T: Tlb, C: MipsCache>(
    _ctx_ptr: *mut JitContext, exec_ptr: *mut u8, rd: u64, value: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    exec.core.write_cp0(rd as u32, value as u32 as i32 as i64 as u64);
    0
}

/// DMTC0: write full 64-bit `value` into CP0 register `rd`.
pub extern "C" fn jit_dmtc0<T: Tlb, C: MipsCache>(
    _ctx_ptr: *mut JitContext, exec_ptr: *mut u8, rd: u64, value: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    exec.core.write_cp0(rd as u32, value);
    0
}

/// Collection of monomorphized helper function pointers.
pub struct HelperPtrs {
    pub read_u8:  *const u8,
    pub read_u16: *const u8,
    pub read_u32: *const u8,
    pub read_u64: *const u8,
    pub write_u8:  *const u8,
    pub write_u16: *const u8,
    pub write_u32: *const u8,
    pub write_u64: *const u8,
    pub interp_step: *const u8,
    pub mfc0: *const u8,
    pub dmfc0: *const u8,
    pub mtc0: *const u8,
    pub dmtc0: *const u8,
}

impl HelperPtrs {
    pub fn new<T: Tlb, C: MipsCache>() -> Self {
        Self {
            read_u8:  jit_read_u8::<T, C>  as *const u8,
            read_u16: jit_read_u16::<T, C> as *const u8,
            read_u32: jit_read_u32::<T, C> as *const u8,
            read_u64: jit_read_u64::<T, C> as *const u8,
            write_u8:  jit_write_u8::<T, C>  as *const u8,
            write_u16: jit_write_u16::<T, C> as *const u8,
            write_u32: jit_write_u32::<T, C> as *const u8,
            write_u64: jit_write_u64::<T, C> as *const u8,
            interp_step: jit_interp_one_step::<T, C> as *const u8,
            mfc0:  jit_mfc0::<T, C>  as *const u8,
            dmfc0: jit_dmfc0::<T, C> as *const u8,
            mtc0:  jit_mtc0::<T, C>  as *const u8,
            dmtc0: jit_dmtc0::<T, C> as *const u8,
        }
    }
}
