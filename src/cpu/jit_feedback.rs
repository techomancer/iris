//! Cross-module JIT status-bar feedback (jitv2 arena/queue fullness, flush
//! flashes). A single global `static` of plain atomics — not threaded
//! through constructors — because the producers (the jitv2 compile thread,
//! the inline-compile path on the CPU thread) and the one consumer (`disp.rs`'s
//! `StatusBar`, on the REX3 refresh thread) have no other object in common;
//! `Rex3` doesn't hold a `Jitv2` reference and shouldn't need to just to
//! report a percentage. Mirrors `crate::devlog::DEVLOG`'s global-singleton
//! shape, simplified: every field here has a const default, so no `OnceLock`
//! init step is needed.

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};

pub struct JitFeedback {
    /// `Codegen::packing_stats().1 (reserved_bytes) * 255 /
    /// CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES` — how full the real Cranelift
    /// code arena is relative to the threshold that triggers `mega_flush`,
    /// not the page-pool slot count (that's a separate, much larger and
    /// less interesting resource — the code arena is what actually runs
    /// out and forces a flush). Updated once per compile from whichever
    /// side is actually compiling (async worker or inline). 0..=255
    /// fixed-point fraction rather than a raw bytes/threshold pair — the
    /// status bar only ever wants a fill fraction, and a single-byte atomic
    /// is cheaper to publish than two u64s that'd need to be read together
    /// to mean anything.
    pub arena_fill: AtomicU8,
    /// `consumer.slots() * 255 / COMPILE_QUEUE_CAPACITY`, updated from
    /// `CompileQueue::worker_loop` (the consumer side) once per compile —
    /// deliberately not from `CompileQueue::send` (the producer side),
    /// which runs on the CPU/exec thread's hot dispatch-gate path on every
    /// compile trigger; an extra atomic store there would tax exactly the
    /// thread the async queue exists to keep off compile work. Stays 0
    /// under inline-compile mode (`j2 inline on`, the default) — there is
    /// no queue to fill when every compile runs synchronously on the CPU
    /// thread that requested it.
    pub queue_fill: AtomicU8,
    /// Bumped by `Jitv2::mega_flush` (both call sites — CPU-thread and
    /// jit-thread flush) each time a flush actually happens. The status bar
    /// samples this once per frame and flashes if it changed since the last
    /// sample — a counter, not a bool, so a flush that happens and clears
    /// again between two frames still gets shown (a bool set-then-cleared
    /// in the same gap would be invisible).
    pub flush_events: AtomicU32,
}

pub static JIT_FEEDBACK: JitFeedback = JitFeedback {
    arena_fill: AtomicU8::new(0),
    queue_fill: AtomicU8::new(0),
    flush_events: AtomicU32::new(0),
};

impl JitFeedback {
    pub fn set_arena_fill(&self, reserved_bytes: u64, threshold_bytes: u64) {
        let frac = if threshold_bytes == 0 { 0 } else { (reserved_bytes * 255 / threshold_bytes).min(255) as u8 };
        self.arena_fill.store(frac, Ordering::Relaxed);
    }

    pub fn set_queue_fill(&self, occupancy: usize, capacity: usize) {
        let frac = if capacity == 0 { 0 } else { (occupancy * 255 / capacity).min(255) as u8 };
        self.queue_fill.store(frac, Ordering::Relaxed);
    }

    pub fn record_flush(&self) {
        self.flush_events.fetch_add(1, Ordering::Relaxed);
    }
}
