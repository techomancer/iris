//! `extern "C"` bridge functions called by JIT-compiled code.
//!
//! Phase 1: stubs only. These are populated in Phase 3+ when compiled blocks
//! need to call back into the interpreter for memory access, exceptions, etc.

use super::context::JitContext;

/// Read memory via the interpreter's full memory subsystem.
/// Called by JIT-compiled code when a memory load hits the slow path.
pub extern "C" fn jit_helper_read_data(
    _ctx: *mut JitContext,
    _virt_addr: u64,
    _size: u32,
) -> u64 {
    unimplemented!("JIT memory read helper not yet implemented")
}

/// Write memory via the interpreter's full memory subsystem.
pub extern "C" fn jit_helper_write_data(
    _ctx: *mut JitContext,
    _virt_addr: u64,
    _value: u64,
    _size: u32,
) -> u32 {
    unimplemented!("JIT memory write helper not yet implemented")
}
