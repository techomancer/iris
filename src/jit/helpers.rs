//! `extern "C"` bridge functions called by JIT-compiled code for memory access.
//!
//! CRITICAL: All pointer casts use `std::hint::black_box` to prevent LLVM from
//! tracking pointer provenance through LTO. Without this, LLVM can prove the
//! exec_ptr derives from a &mut in the dispatch loop and apply noalias
//! optimizations that cause stale reads.

use super::context::{JitContext, EXIT_EXCEPTION};
use crate::mips_exec::{MipsExecutor, MemAccessSize, EXEC_COMPLETE};
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
    match exec.read_data(virt_addr, MemAccessSize::Byte) {
        Ok(value) => value,
        Err(status) => { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; 0 }
    }
}

pub extern "C" fn jit_read_u16<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    match exec.read_data(virt_addr, MemAccessSize::Half) {
        Ok(value) => value,
        Err(status) => { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; 0 }
    }
}

pub extern "C" fn jit_read_u32<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    match exec.read_data(virt_addr, MemAccessSize::Word) {
        Ok(value) => value,
        Err(status) => { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; 0 }
    }
}

pub extern "C" fn jit_read_u64<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    match exec.read_data(virt_addr, MemAccessSize::Double) {
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
    let status = exec.write_data(virt_addr, value, MemAccessSize::Byte, 0xFF);
    if status != EXEC_COMPLETE { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; }
    0
}

pub extern "C" fn jit_write_u16<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64, value: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    let status = exec.write_data(virt_addr, value, MemAccessSize::Half, 0xFFFF);
    if status != EXEC_COMPLETE { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; }
    0
}

pub extern "C" fn jit_write_u32<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64, value: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    let status = exec.write_data(virt_addr, value, MemAccessSize::Word, 0xFFFF_FFFF);
    if status != EXEC_COMPLETE { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; }
    0
}

pub extern "C" fn jit_write_u64<T: Tlb, C: MipsCache>(
    ctx_ptr: *mut JitContext, exec_ptr: *mut u8, virt_addr: u64, value: u64,
) -> u64 {
    let exec = unsafe { &mut *opaque_exec::<T, C>(exec_ptr) };
    let ctx = unsafe { &mut *opaque_ctx(ctx_ptr) };
    let status = exec.write_data(virt_addr, value, MemAccessSize::Double, 0xFFFF_FFFF_FFFF_FFFF);
    if status != EXEC_COMPLETE { ctx.exit_reason = EXIT_EXCEPTION; ctx.exception_status = status; }
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
        }
    }
}
