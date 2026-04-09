//! Cranelift-based JIT compiler for MIPS R4400.
//!
//! Feature-gated under `#[cfg(feature = "jit")]`.
//! Phase 1: dispatch infrastructure with full interpreter fallback.

pub mod context;
pub mod cache;
pub mod compiler;
pub mod dispatch;
pub mod helpers;
pub mod profile;
pub mod snapshot;
pub mod trace;
pub mod codegen_test;

pub use context::JitContext;
pub use cache::{CodeCache, CompiledBlock};
pub use snapshot::CpuRollbackSnapshot;
pub use compiler::BlockCompiler;
