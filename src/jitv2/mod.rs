//! JIT v2: physical-page PIC region compiler. See `rules/jitv2/jit-v2-design.md`.

pub mod jitv2;
pub mod comp;
pub mod opcode_support;
pub mod analyzer;
pub mod codegen;
pub mod paged_memory;
#[cfg(feature = "j2wp")]
pub mod pcp_dump;
pub mod equiv_test;

#[cfg(not(feature = "j2wp"))]
pub use jitv2::JitEntry;
pub use jitv2::{
    CompileQueue, CompileRequest, JitFn, JitStats, Jitv2, PageSlot, Pfn, PhysicalCodePage,
    ARENA_RESERVE_SIZE, BITMAP_WORDS, CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES, COMPILE_QUEUE_CAPACITY,
    ENTRIES_PER_PAGE, JITV2_INITIAL_PAGE_CAPACITY, PAGE_SIZE,
    min_calls_before_compile, set_min_calls_before_compile,
};
pub use paged_memory::{PagedArenaMemoryProvider, PagedArenaState};
#[cfg(feature = "developer")]
pub use jitv2::{BatchFlushReason, CodeSizeBucket, RejectReason, REJECT_REASON_COUNT};

/// The jitv2 dirty-page probe — see `rules/jitv2/dirty-cache-page-probe.md`.
/// Absent under `tcache`, which closes that blind spot by construction.
#[cfg(not(feature = "tcache"))]
pub use jitv2::{install_jit_page_probe, clear_jit_page_probe, jit_page_has_dirty_lines};
