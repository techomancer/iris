//! JIT v2 core data structures.
//!
//! See `rules/jitv2/jit-v2-design.md` for the full design. This module holds the
//! per-page metadata the rest of the engine builds on (§2.4):
//!
//! - [`PhysicalCodePage`]: the mips executor's view of "what's compiled here" for
//!   the physical page it's currently executing out of.
//! - Generation counters live in the owning `BusDevice` (one per page for RAM,
//!   a single shared never-bumped counter for ROM — `BusDevice::gen_ptr`,
//!   `src/mem.rs`, `src/prom.rs`) and are read through a raw pointer here so the
//!   hot path avoids an indirect call through the device trait object.
//!
//! Threading model: the mips exec thread owns `PhysicalCodePage` management
//! (arrival, promotion — §6.1) and pushes compile requests to the compile thread
//! over an SPSC fifo (§6.4); the compile thread publishes finished artifacts back
//! into the page's `entry_table`/`entry_bits` (§6.1.3), which is why requests carry
//! a mutable pointer. Only the compile-request queue itself is added in this pass
//! — the compile thread and publish path land with codegen (Phase 2).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;

use parking_lot::Mutex;

use crate::mips_core::MipsCore;
use crate::mips_exec::ExecStatus;
use crate::traits::{BusDevice, Device};

// ============================================================================
// Tunables & initial settings
//
// Every knob that shapes JIT v2's steady-state behavior — page geometry,
// pool/queue capacities, and the thresholds that decide when to flush or
// force-seal — lives in this block so they can be reviewed and retuned
// together instead of hunting through the file. See each constant's doc
// comment for the reasoning behind its specific value.
// ============================================================================

/// Page size for JIT v2 (§2.4) — matches the MIPS TLB/cache page granularity
/// used throughout the codebase. Canonical home for this constant; `mem.rs`
/// re-exports it as `JITV2_PAGE_SIZE` for its own generation-counter indexing.
pub const PAGE_SIZE: u32 = 4096;

/// Number of possible entry offsets per page: one per 4-byte-aligned word
/// (MIPS instructions are always word-aligned), i.e. `PAGE_SIZE / 4` (§2.4:
/// `entry_bits` 16×u64 = 1024 bits, `entry_table` 1024 entries).
pub const ENTRIES_PER_PAGE: usize = (PAGE_SIZE / 4) as usize;

/// u64 words needed for a 1-bit-per-entry bitmap over `ENTRIES_PER_PAGE` offsets.
pub const BITMAP_WORDS: usize = ENTRIES_PER_PAGE / 64;

/// Default page-pool capacity for `Jitv2::new()` as embedded in `MipsExecutor`.
/// Sizing is a Phase 0 measurement per the design doc (§9, "Max live entries per
/// epoch"); `mega_flush` absorbs it being wrong in either direction. Now that
/// the whole pool is a single array allocated once at this capacity (see
/// `Jitv2::new`'s doc comment — `PhysicalCodePage` is no longer boxed inside,
/// so a larger capacity is a real memory cost, not just reserved address
/// space), 4096 is a deliberately modest working-set size rather than a
/// generous upper bound — `mega_flush`'s cost of getting this wrong low
/// (a pool-exhaustion flush) is cheap relative to permanently carrying a much
/// larger array.
pub const JITV2_INITIAL_PAGE_CAPACITY: usize = 4096;

/// Depth of the compile-request SPSC ring (§6.4 "bounded queue; drop on full —
/// hot pages re-trigger"). A starting guess, like `JITV2_INITIAL_PAGE_CAPACITY`
/// — doubled from 1024 after a live `j2 status` reading showed the compile
/// thread genuinely falling behind at that size (20.9% of dispatches
/// dropped for a full queue, average depth at dispatch 248.6/1024, out of
/// 1,166,218 total dispatches during one session) rather than the queue
/// mostly sitting near-empty.
pub const COMPILE_QUEUE_CAPACITY: usize = 2048;

/// Flush threshold for the shared `Codegen`'s Cranelift arena, in bytes
/// actually reserved (`Codegen::packing_stats()`'s `reserved` — real
/// host-page-rounded arena footprint, not the function-count proxy this
/// constant used before batching landed). `cranelift_jit::Memory` never
/// frees on drop/replace (`Codegen::reset`'s own doc comment), so nothing
/// else bounds arena growth — a long-enough-running compile (real IRIX boot,
/// not just PROM) will otherwise exhaust the whole `Codegen::ARENA_RESERVE_SIZE`
/// reservation.
///
/// Function count stopped being a good proxy for arena growth once
/// deferred-finalize batching (`j2 batch`) started letting many small
/// functions pack into a shared host-page segment instead of each getting
/// its own — the byte size actually reserved is now directly measurable
/// (`PagedArenaState`), so there's no reason to keep estimating it from a
/// count. 128MiB leaves comfortable headroom under `Codegen::ARENA_RESERVE_SIZE`
/// (512MiB) for the batch that happens to be in flight when this trips (a
/// batch isn't finalized/counted until it flushes, so the real reservation
/// can run slightly ahead of this threshold between checks) while still
/// flushing well before the arena's own exhaustion error could ever fire —
/// that error path (`comp::handle_request`'s exhaustion match arm) stays as
/// a belt-and-suspenders backstop, not the primary trigger.
#[cfg(not(feature = "j2wp"))]
pub const CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES: u64 = 256 * 1024 * 1024;

/// `j2wp`'s own Cranelift arena reservation — bigger than the default
/// path's `Codegen::ARENA_RESERVE_SIZE` (512MiB) because a one-function-
/// per-page compiled unit is larger on average than a one-function-per-
/// entry-point unit, so the whole-page redesign needs more headroom before
/// hitting its own flush threshold below.
#[cfg(feature = "j2wp")]
pub const ARENA_RESERVE_SIZE: usize = 2 * 1024 * 1024 * 1024;
#[cfg(feature = "j2wp")]
pub const CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES: u64 = ARENA_RESERVE_SIZE as u64 - 128 * 1024 * 1024;

/// Force-seal trigger for a continuously busy batching worker — see
/// `worker_loop`'s own comment at its call site. `handle_request_deferred`
/// finalizes every compile immediately, but a non-forced finalize only
/// seals a page the bump cursor has already moved past, so `pending` can
/// otherwise grow without bound while the queue never goes empty long
/// enough to reach the idle-timeout sweep. Small and arbitrary — this only
/// needs to be "small enough that pending never grows unbounded," not tuned
/// to any particular page size (unlike the old page-cross trigger it
/// replaces, this one has no reason to line up with `ENTRIES_PER_PAGE`).
const PENDING_FORCE_SEAL_THRESHOLD: usize = 64;

/// Queue-drain fallback: how long the compile-request queue must stay
/// continuously empty (wall-clock, tracked across repeated empty polls, not
/// "first empty poll") before `worker_loop` force-seals whatever's left
/// rather than waiting for a page to fill on its own — see `worker_loop`'s
/// own comment at its call site for the two-phase (non-forced then forced)
/// sealing design this backstops.
const IDLE_FORCE_SEAL_THRESHOLD: std::time::Duration = std::time::Duration::from_millis(100);

/// Backoff between empty-queue polls in `worker_loop`. Short enough that
/// `IDLE_FORCE_SEAL_THRESHOLD`'s wall-clock idle window still resolves in a
/// handful of ticks, long enough not to spin the compile thread on cycles
/// while genuinely idle.
const WORKER_IDLE_POLL_BACKOFF: std::time::Duration = std::time::Duration::from_micros(200);

/// Bounded re-check interval for a leader waiting on followers to park at the
/// compile-pool flush barrier (`wait_for_followers_or_abandon`). Not on any
/// latency-sensitive path — only needs to be short enough that `stop()`
/// racing a quiesce cycle doesn't feel hung; see that closure's own comment
/// for why an unbounded `cv.wait` would be unsafe here.
const BARRIER_FOLLOWER_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(20);

/// Minimum interpreter dispatches an offset must accumulate before
/// `exec_decoded`'s dispatch gate (`mips_exec.rs`) will send its first
/// `CompileRequest` — `j2 min-calls [N]` tunes this at runtime. A nonzero
/// value trades a few extra interpreted dispatches of a genuinely-cold
/// offset for never paying compile cost on paths that only ever run once or
/// twice — real hot loops clear any small threshold within a handful of
/// iterations, so this mostly filters out one-shot/rare code, not anything
/// actually hot. See `PhysicalCodePage::count_dispatch_and_check_threshold`
/// for the counter mechanism (reuses the per-entry `gen` slot, unused before
/// first publish). Callers that must stay deterministic/immediate (tests,
/// `jitv2_inline_compile`, lockstep/verification harnesses — none of which
/// go through `exec_decoded`'s real gate at all, `jitv2_lockstep` least of
/// all since it's a separate code path entirely) are exempt by construction,
/// not by passing a special value: they never call
/// `count_dispatch_and_check_threshold`, so this setting never applies to
/// them regardless of its value.
///
/// Defaults to 0 ("always ready," the original behavior) under `developer` —
/// diagnostics builds want every compilable offset compiled immediately, not
/// filtered by a production-tuned call-count floor — and to 4 otherwise.
static MIN_CALLS_BEFORE_COMPILE: AtomicU64 = AtomicU64::new(if cfg!(feature = "developer") { 0 } else { 4 });

pub fn set_min_calls_before_compile(n: u64) {
    MIN_CALLS_BEFORE_COMPILE.store(n, Ordering::Relaxed);
}

pub fn min_calls_before_compile() -> u64 {
    MIN_CALLS_BEFORE_COMPILE.load(Ordering::Relaxed)
}

// ============================================================================
// End tunables
// ============================================================================

/// Physical frame number. Physical addresses are keyed by PFN, never by VA (§2.1).
pub type Pfn = u32;

/// Compiled-function ABI (§6.1.2's "handler ABI", simplified for this storage
/// pass — no `DecodedInstr`/state-struct plumbing yet, just direct MipsCore
/// access): takes a pointer to the executor's `MipsCore` and returns the same
/// `ExecStatus` every interpreter handler returns. `vbase` derivation, the
/// two mirrored checks (§3.2), and exit-stub materialization all live inside
/// the compiled body once codegen exists — this signature is just the call
/// boundary.
pub type JitFn = unsafe extern "C" fn(*mut MipsCore) -> ExecStatus;

/// `JitEntry::flags` bit — set iff this offset holds a published,
/// dispatchable function. Authoritative for dispatch (probed on un-promoted
/// arrivals, §6.1) and the kill path — Release-set by `publish`, cleared by
/// `kill`. This bit is the single source of truth for "is this entry live";
/// `entries[i].func` being non-null is not itself checked (kill clears the
/// flag but may leave `func` populated until the slot is reused — see
/// `JitEntry::func`'s doc comment).
const ENTRY_VALID: u8 = 1 << 0;
/// `JitEntry::flags` bit — set iff this offset was permanently refused by
/// the compiler (too-short region below the yield threshold, excluded first
/// instruction, 0xFFC/slot hazard, etc — §6.4 "sticky rejection"). Consulted
/// by arrival/queueing to stop re-requesting a compile that will only be
/// declined again; cleared on a gen bump (re-classify against new bytes)
/// alongside `ENTRY_VALID`.
const ENTRY_DENYLISTED: u8 = 1 << 1;
/// `JitEntry::flags` bit — set iff a `CompileRequest` for this offset has
/// been sent to the async compile-thread queue and hasn't been decided yet
/// (published or denylisted). `exec_decoded`'s dispatch gate consults this
/// before building/sending another request for the same offset — without
/// it, every dispatch of a not-yet-compiled offset that keeps
/// re-satisfying the gate's trigger conditions (e.g. a hot loop back-edge
/// landing on the same still-uncompiled word every iteration, or
/// `jit_trigger` now also being set by JIT-to-JIT jump exits — see
/// `MipsCore::jit_trigger`) sends a fresh, redundant `CompileRequest` for
/// the exact same (page, offset) every single time, flooding the
/// compile-thread's queue with requests that all do the same work.
/// `try_schedule` is the only setter (a compare-exchange, so only the first
/// caller for a given offset actually wins and sends); cleared by
/// `clear_scheduled` once `handle_request` (`comp.rs`) has decided the
/// offset one way or the other, so a later legitimate re-request (e.g.
/// after a gen bump invalidates a stale artifact) isn't permanently
/// blocked. Irrelevant to the synchronous `jitv2_inline_compile` path —
/// that path can't re-enter before `comp::handle_request` returns, so
/// there's no queue to flood.
const ENTRY_SCHEDULED: u8 = 1 << 2;

/// Single entry in a page's compiled-function table (§2.4 `entry_table`).
/// AoS layout (one `JitEntry` per offset) rather than the design doc's literal
/// SoA (`entry_bits` bitmap + separate `entry_table` pointer array): `gen` is
/// consulted together with `func` at every dispatch (staleness check against
/// the page's current generation, §4.1/§6.5), so keeping them in the same
/// cache line avoids a second, unrelated array touch on the hot path.
/// `flags` (ENTRY_VALID/ENTRY_DENYLISTED/ENTRY_SCHEDULED) is colocated here
/// too, rather than `PhysicalCodePage` holding three separate
/// `[AtomicU64; BITMAP_WORDS]` bitmaps — the dispatch-hot read
/// (`is_entry_valid`, checked on every dispatch) used to mean touching a
/// bitmap word in one cache line and then `func`/`gen` in a completely
/// separate one; a single `flags` byte right next to `func`/`gen` means the
/// first touch of this entry already pulls the whole thing into L1 together.
/// `saved_bits` (corpus-collection scaffolding only, not part of the design
/// doc's per-page metadata — see its own field doc) is the one bitmap that
/// stays a separate SoA array: it's scanned/tested completely independently
/// of dispatch and is slated for deletion once the real compiler replaces
/// the dump-to-disk stub anyway.
pub struct JitEntry {
    /// Compiled function pointer for this offset, or null if unpublished.
    /// Validity is owned entirely by `flags`' `ENTRY_VALID` bit (§6.1.2's
    /// "the remove check IS this load" — a raw pointer, not `Option`, keeps
    /// that the single source of truth instead of letting callers branch on
    /// `func.is_some()` as a second, potentially-stale answer to the same
    /// question). Callers must check `ENTRY_VALID` before calling this.
    pub func: *const (),
    /// Generation this entry was compiled against (§6.5 `gen_snap`). An entry
    /// is valid iff `gen == page.current_gen()` — mismatch means the page
    /// mutated since compilation and the entry must be treated as stale
    /// (downgrade to interpreter, §6.1.2) regardless of what `flags` says.
    pub gen: AtomicU64,
    /// See `ENTRY_VALID`/`ENTRY_DENYLISTED`/`ENTRY_SCHEDULED`'s own doc
    /// comments. A single byte holding all three independent per-offset
    /// flags — they're set/cleared/tested independently of each other
    /// (never as a combined mask), just packed together for locality.
    pub flags: AtomicU8,
    /// Dev-only diagnostics for `j2 pcp`: how many instructions this
    /// entry's compiled region covers (set once at publish time — not
    /// atomic since it's write-once-then-read, same lifecycle as `func`)
    /// and how many times `exec_decoded`'s dispatch gate has actually
    /// called this entry's `func` (incremented on every dispatch, hence
    /// atomic — the exec thread is the only writer today, but this is read
    /// concurrently from the monitor console thread via `j2 pcp`). Added to
    /// help diagnose "lockstep boots fine but normal dispatch stalls
    /// somewhere" class bugs: a hot loop stuck calling the same
    /// under-sized/wrong region over and over shows up immediately as one
    /// offset's `call_count` growing without bound while PC-visible
    /// progress stalls, without needing to add ad-hoc eprintln!s each time.
    #[cfg(feature = "developer")]
    pub instr_count: u16,
    /// Dev-only diagnostic (`j2 pcp`/`j2 stats`): size in bytes of this
    /// entry's compiled machine code, read from Cranelift's own
    /// `CompiledCode::buffer` right after `define_function` — see
    /// `Codegen::compile_region`'s doc comment. Summed across every
    /// published entry (`Jitv2::code_bytes_used`) as the best available
    /// proxy for the shared `Codegen`'s Cranelift memory-arena size, since
    /// `cranelift_jit::Memory` exposes no size/usage API of its own
    /// (`pub(crate)`, not reachable from outside the crate).
    #[cfg(feature = "developer")]
    pub code_size: u32,
    #[cfg(feature = "developer")]
    pub call_count: AtomicU64,
    /// Dev-only diagnostic (`j2 pcp`): saturating count of how many distinct
    /// compiled regions have *included this word* (visited it during their
    /// reachability walk), incremented once per word per successful compile in
    /// `comp::handle_request`/`handle_request_deferred`. Unlike `instr_count`
    /// (a property of the region rooted *at this offset*) this accumulates
    /// across every region that walks *through* this offset from any entry —
    /// so a word covered by several overlapping blocks reads high, giving a
    /// direct picture of block overlap / redundant recompilation of the same
    /// straight-line code from different entry points. Saturating `u8`: the
    /// exact count past 255 doesn't matter, "pegged" is the signal. Atomic
    /// because the async compile thread and the inline-compile CPU-thread path
    /// can both write it, and `j2 pcp` reads it from the monitor thread.
    #[cfg(feature = "developer")]
    pub block_include_count: AtomicU8,
}

impl Default for JitEntry {
    fn default() -> Self {
        Self {
            func: std::ptr::null(),
            gen: AtomicU64::new(0),
            flags: AtomicU8::new(0),
            #[cfg(feature = "developer")]
            instr_count: 0,
            #[cfg(feature = "developer")]
            code_size: 0,
            #[cfg(feature = "developer")]
            call_count: AtomicU64::new(0),
            #[cfg(feature = "developer")]
            block_include_count: AtomicU8::new(0),
        }
    }
}

// Safety: `func`, when non-null, points to finalized JIT-compiled code owned
// by the compile-thread arena, which outlives every PhysicalCodePage entry
// referencing it until the next mega_flush (mirrors PhysicalCodePage's own
// Send/Sync rationale for `gen`).
unsafe impl Send for JitEntry {}
unsafe impl Sync for JitEntry {}

/// Process-wide JIT event counters, displayed by `j2 status` only under the
/// `developer` feature — plain `AtomicU64`s rather than per-`Jitv2` state
/// protected by a lock, since these are touched from both the compile
/// thread (`comp::handle_request`, on every request, inline or threaded)
/// and the CPU thread (`jit_kill_entry`, on every FR-mismatch bail —
/// `mips_exec.rs`) and none of them need to be read together atomically
/// with anything else; a lock here would just be hot-path contention for no
/// correctness benefit. The struct and its `Arc<JitStats>` field on `Jitv2`
/// exist unconditionally (not `#[cfg(feature = "developer")]`) so nothing
/// on the path from `Jitv2::new`/`CompileQueue::start` down to
/// `handle_request` needs a second, feature-gated copy of itself just to
/// thread one more `Arc` through — only the actual `.fetch_add()` call
/// sites and the `j2 status` display are gated. Survives `mega_flush`
/// deliberately (counts are lifetime totals, not "since last reset" —
/// `Codegen::function_count`'s own since-reset counter already covers that
/// angle).
/// Why a compile request was declined — `failed_compiles`'s single counter
/// doesn't distinguish these, but they're different enough situations to
/// want separately (`j2 status`'s "rejections by reason" breakdown):
/// a codegen gap you could go implement an emitter for reads very
/// differently from a Cranelift verifier bug, which reads differently again
/// from "the analyzer and codegen's emitter tables disagree" (should be
/// structurally impossible after `opcode_support::has_emitter` unified them —
/// a nonzero count here means they've drifted again).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "developer")]
pub enum RejectReason {
    /// `walk_bounded` reported the entry offset itself unreachable —
    /// architecturally excluded (COP0, CACHE, LL/SC, ...) or a codegen gap
    /// at the very first instruction, `analyzer::classify` returned
    /// `Excluded` either way (`comp::handle_request`'s `!non_empty` arm).
    EntryExcluded,
    /// `compile_region`'s upfront rejection loop found a visited instruction
    /// with no emitter in any of the four lookup tables. Per
    /// `opcode_support::has_emitter` being the single source of truth both
    /// `analyzer::classify` and this loop consult, this should be
    /// unreachable in practice — every instruction the analyzer walked past
    /// as `Sequential`/`Branch`/`Jump`/`RegJump` already passed the same
    /// check. Tracked anyway as a canary: a nonzero count here means the
    /// tables have drifted apart again, the exact bug class
    /// `opcode_support.rs` was built to close structurally.
    AnalyzerCodegenDisagreement,
    /// Cranelift's `define_function` returned `ModuleError::Compilation` —
    /// this module emitted IR the verifier rejected, a real bug in some
    /// emitter, not an unsupported-instruction-shape decline (those are all
    /// caught by the upfront loop before Cranelift is ever invoked).
    CraneliftVerifierError,
    /// The walked region's instruction count fell below
    /// `comp::min_instrs_to_compile()` (`j2 min-instrs`) — not a codegen gap
    /// at all, just not judged worth the fixed per-compile cost. See
    /// `comp::MIN_INSTRS_TO_COMPILE`'s own doc comment.
    TooShort,
}
#[cfg(feature = "developer")]
pub const REJECT_REASON_COUNT: usize = 4;
#[cfg(feature = "developer")]
impl RejectReason {
    pub fn index(self) -> usize {
        match self {
            RejectReason::EntryExcluded => 0,
            RejectReason::AnalyzerCodegenDisagreement => 1,
            RejectReason::CraneliftVerifierError => 2,
            RejectReason::TooShort => 3,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            RejectReason::EntryExcluded => "entry excluded (unsupported first instruction / architectural exclusion)",
            RejectReason::AnalyzerCodegenDisagreement => "analyzer/codegen disagreement (should be unreachable — see doc comment)",
            RejectReason::CraneliftVerifierError => "Cranelift verifier error (real emitter bug)",
            RejectReason::TooShort => "region too short (below j2 min-instrs threshold)",
        }
    }
}

#[derive(Default)]
pub struct JitStats {
    /// Successful `compile_region` calls that got published
    /// (`comp::handle_request`'s `Some(jit_fn)` arm).
    pub compiles: AtomicU64,
    /// `compile_region` declines or `walk_bounded` exclusions
    /// (`comp::handle_request`'s two `page.denylist(offset)` sites) — codegen
    /// gaps and analyzer-excluded entry offsets alike, not distinguished
    /// further (both end up sticky-denylisted the same way). See
    /// `reject_reasons` for the breakdown by cause.
    pub failed_compiles: AtomicU64,
    /// Per-`RejectReason` breakdown of `failed_compiles`, indexed by
    /// `RejectReason::index()`. Excludes the arena-out-of-memory outcome
    /// (`comp::handle_request`'s `last_compile_ran_out_of_memory` arm) —
    /// that one isn't a rejection of the *region*, it retries on its own,
    /// so it doesn't belong in a "why did this instruction never compile"
    /// breakdown; it's counted in `failed_compiles` but not here.
    #[cfg(feature = "developer")]
    pub reject_reasons: [AtomicU64; REJECT_REASON_COUNT],
    /// `jit_kill_entry` calls — a compiled unit's FR-mode-mismatch guard
    /// bailing and un-publishing its own entry (`emit_fpu_entry_guard`'s CU1/FR
    /// mismatch design, `MipsCore::kill_entry_fn`'s doc comment).
    pub kill_entry_calls: AtomicU64,
    /// Total `CompileQueue::send` calls (both accepted and dropped) —
    /// `j2 status`'s FIFO-fullness section denominator. `developer`-only:
    /// unlike `compiles`/`failed_compiles`/`kill_entry_calls` (rare events,
    /// off the hot path), `send` runs on every dispatch-gate arrival that
    /// misses the JIT cache — real per-instruction-adjacent traffic, so the
    /// extra atomic touch these three fields cost is worth avoiding outside
    /// a diagnostics build.
    #[cfg(feature = "developer")]
    pub compile_queue_dispatches: AtomicU64,
    /// `CompileQueue::send` calls that dropped the request because the ring
    /// buffer (`COMPILE_QUEUE_CAPACITY`) was already full — see `send`'s own
    /// doc comment for why a drop here is a normal, non-fatal outcome (the
    /// hot page just re-triggers the request on a later arrival), not an
    /// error; this counter exists to tell whether that's happening often
    /// enough to be a real bottleneck, not a rare edge case.
    #[cfg(feature = "developer")]
    pub compile_queue_full: AtomicU64,
    /// Running sum of `Producer::slots()`-derived occupancy
    /// (`capacity - free_slots`) sampled at every `send` call — divide by
    /// `compile_queue_dispatches` for the mean queue depth the compile
    /// thread is actually running at. A cumulative sum rather than a
    /// separate min/max/histogram: cheap (one more atomic add on an
    /// already-atomic-touching path), and mean depth is the number that
    /// actually answers "is the queue usually near-empty (compile thread
    /// keeping up) or usually near-full (compile thread falling behind)."
    #[cfg(feature = "developer")]
    pub compile_queue_depth_sum: AtomicU64,
    /// `try_force_seal` calls triggered by `pending` crossing
    /// `PENDING_FORCE_SEAL_THRESHOLD` on a continuously busy worker — `j2
    /// stats`'s batching section, split from `batch_flushes_queue_drain` so
    /// it's possible to tell whether a worker is getting throttled by
    /// sustained load (this counter) or is mostly idling between bursts
    /// (that one).
    #[cfg(feature = "developer")]
    pub batch_flushes_pending_threshold: AtomicU64,
    /// `try_force_seal` calls triggered by the compile queue draining empty
    /// (`worker_loop`'s `None` arm's idle-timeout sweep) — see
    /// `batch_flushes_pending_threshold`.
    #[cfg(feature = "developer")]
    pub batch_flushes_queue_drain: AtomicU64,
    /// High-water mark of `pending` just before any force-seal sweep
    /// actually published something (i.e. sampled right before the sweep
    /// runs, not after) — `j2 stats`'s "biggest backlog we ever let build
    /// up" figure. A `Relaxed` compare-and-swap loop, not a simple
    /// `fetch_max` (stabilized in std but this codebase's MSRV predates it
    /// for `AtomicU64` — matches the CAS-loop pattern already used elsewhere
    /// in this codebase for high-water-mark tracking).
    #[cfg(feature = "developer")]
    pub batch_max_pending: AtomicU64,
    /// Interpreter-fallback words compiled into published regions
    /// (`comp::handle_request`, incremented once per `is_fallback` word). The
    /// direct answer to "did the fallback path actually run this session?" — a
    /// clean boot with `j2 fallback on` means nothing if this is 0 (the flag was
    /// flipped after the boot-critical regions already compiled without it).
    /// Also `fallback_regions`: published regions that contained at least one
    /// fallback word, so you can tell a few big fallback loops from many tiny
    /// ones.
    #[cfg(feature = "developer")]
    pub fallback_words: AtomicU64,
    #[cfg(feature = "developer")]
    pub fallback_regions: AtomicU64,
}

/// Why a force-seal sweep ran — see `JitStats::batch_flushes_pending_threshold`/
/// `batch_flushes_queue_drain`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "developer")]
pub enum BatchFlushReason {
    PendingThreshold,
    QueueDrain,
}

#[cfg(feature = "developer")]
impl JitStats {
    /// Bump both `failed_compiles` and the per-reason breakdown together —
    /// the two call sites (`comp::handle_request`'s two denylist arms,
    /// `Codegen::compile_region`'s two `None` returns) always want to record
    /// both, and a shared helper means they can't drift out of sync (one
    /// incremented without the other).
    pub fn record_reject(&self, reason: RejectReason) {
        self.failed_compiles.fetch_add(1, Ordering::Relaxed);
        self.reject_reasons[reason.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// Record one force-seal sweep: bump the trigger-specific counter and
    /// update the high-water mark from `pending_len` (the backlog size just
    /// before this sweep drained it).
    pub fn record_batch_flush(&self, pending_len: usize, reason: BatchFlushReason) {
        match reason {
            BatchFlushReason::PendingThreshold => self.batch_flushes_pending_threshold.fetch_add(1, Ordering::Relaxed),
            BatchFlushReason::QueueDrain => self.batch_flushes_queue_drain.fetch_add(1, Ordering::Relaxed),
        };
        let pending_len = pending_len as u64;
        let mut cur = self.batch_max_pending.load(Ordering::Relaxed);
        while pending_len > cur {
            match self.batch_max_pending.compare_exchange_weak(cur, pending_len, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }
}

/// A compile request pushed from the mips exec thread to the compile thread
/// over the SPSC fifo (§6.4). Carries the page pointer rather than a snapshotted
/// generation: the compile thread reads `gen` itself at snapshot time (§6.5 step
/// 2, `gen_snap = gen`) and re-reads it at publish time — the generation at
/// queue time is never consulted, only current-at-compile and current-at-publish.
/// The pointer is mutable because publish (§6.1.3) writes into the page's
/// `entry_table`/`entry_bits`/`artifact_list`.
///
/// # Safety
/// `page` must outlive the request — pages live for the lifetime of their
/// owning device (see [`PhysicalCodePage`]'s Send/Sync safety note).
#[derive(Debug)]
pub struct CompileRequest {
    pub page: *mut PhysicalCodePage,
    pub offset: u16,
    /// Live `STATUS_FR` bit at enqueue time, threaded through because the
    /// compile thread has no `MipsCore` to read it from itself — codegen's
    /// FPR-access emitters are FR-mode-specific and must match whatever mode
    /// the executor will actually be in when it calls the compiled function
    /// (same value `exec_decoded` used to run the interpreter fallback for
    /// this same arrival).
    pub compiled_for_fr1: bool,
}

unsafe impl Send for CompileRequest {}

/// Per-physical-page code cache metadata, as tracked by the mips executor
/// (§2.4). One instance per physical RAM/ROM page that has ever been a JIT
/// compilation target; the executor holds a pointer to the page it is
/// currently executing out of.
///
/// Does not yet own `queued_bits`/`artifact_list` (§2.4) — those land with
/// the compile-thread/dispatcher work. `entries` (`JitEntry::flags` plus
/// `func`/`gen`) is the `entry_bits`/`entry_table` pair from the design
/// doc, laid out AoS per-entry rather than the document's literal SoA
/// split.
/// Fallback generation counter for a page whose backing `BusDevice` doesn't
/// implement `gen_ptr` (MMIO, etc — the trait's default returns null). A
/// single `static`, shared by every such page rather than one dummy per page:
/// it never advances (same as a ROM's shared, never-bumped counter — see
/// `PhysicalCodePage::gen`'s doc comment), so there's no reason for each page
/// to have its own copy. Pointing `gen` here instead of leaving it null means
/// `current_gen()`/every dispatch-path deref is unconditionally valid — no
/// null check needed anywhere, ever (`claim()` is the only place that decides
/// whether a page's real `gen_ptr` or this fallback gets used). If code
/// somehow executes out of true MMIO, this makes the JIT treat it like an
/// immutable ROM page instead — the JIT is not in the business of making
/// that scenario correct or fast, just of not crashing on it.
static NEVER_COMPILABLE_GEN: AtomicU64 = AtomicU64::new(0);

pub struct PhysicalCodePage {
    pub pfn: Pfn,
    /// Pointer to this page's generation counter, obtained from the owning
    /// `BusDevice` via `gen_ptr` (§2.4, §7). RAM devices return one counter
    /// per page; ROM devices point every page at a single counter that is
    /// initialized to 0 and never bumped, since ROM content is immutable.
    /// Never null — a device with no real gen tracking (MMIO, etc — `gen_ptr`
    /// returns null) gets pointed at the shared [`NEVER_COMPILABLE_GEN`]
    /// fallback instead, by `claim()`.
    gen: *const AtomicU64,
    /// Per-offset compiled-function slots (§2.4 `entry_table`). Inline, not
    /// boxed: `Jitv2::pages` is a single array allocated once, up front, at
    /// full capacity (`Jitv2::new`'s own doc comment) — every
    /// `PhysicalCodePage` is constructed exactly once and never moved again
    /// (no more `Vec::push` growing the pool one page at a time), so the
    /// "avoid copying the 1024-entry table on move" concern a `Box` used to
    /// exist for doesn't apply anymore; the indirection would just be extra
    /// pointer-chasing on every entry access for no benefit.
    pub entries: [JitEntry; ENTRIES_PER_PAGE],
    /// Scaffolding for corpus collection only (`jitv2/comp.rs`) — NOT part of
    /// the design doc's per-page metadata (§2.4). One bit per entry offset:
    /// set once that (pfn, offset) pair's page snapshot has been dumped to
    /// `jitv2_corpus/`, so a hot page revisited many times only gets saved
    /// once. Safe to delete once the real compiler (reachability walk +
    /// codegen) replaces the dump-to-disk stub in the worker loop.
    pub saved_bits: [AtomicU64; BITMAP_WORDS],
}

// Safety: `gen` points into the owning BusDevice's storage, which outlives
// every PhysicalCodePage derived from it (devices are held in Arcs/statics
// for the lifetime of the machine). The pointee is only ever read or
// atomically incremented, never moved or freed while a PhysicalCodePage
// referencing it exists.
unsafe impl Send for PhysicalCodePage {}
unsafe impl Sync for PhysicalCodePage {}

impl PhysicalCodePage {
    /// Construct an as-yet-unclaimed page descriptor: `pfn = 0`, every
    /// bitmap zeroed, every entry at its default (unpublished) state. Used
    /// both to build `Jitv2::pages`' full-capacity array up front (every slot
    /// starts unclaimed) and, functionally identically, by
    /// [`Self::claim`]/[`Self::reset_in_place`] to return a slot to this same
    /// state in place without reallocating anything.
    ///
    /// `gen` null is accepted here (callers pass it for an as-yet-unclaimed
    /// slot, or when the real device has no gen tracking) and normalized to
    /// [`NEVER_COMPILABLE_GEN`] — `self.gen` itself is never null, so nothing
    /// downstream ever needs to check.
    pub fn new(pfn: Pfn, gen: *const AtomicU64) -> Self {
        Self {
            pfn,
            gen: if gen.is_null() { &NEVER_COMPILABLE_GEN } else { gen },
            entries: std::array::from_fn(|_| JitEntry::default()),
            saved_bits: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// Zero every bitmap and reset every entry (including `flags`) to its
    /// default (unpublished) state, in place — no reallocation, `entries`'s
    /// 1024 slots are written through, not replaced. Called only from
    /// [`Self::reset_to_unclaimed`] (`Jitv2::mega_flush`'s per-slot reset) —
    /// a fresh, never-claimed slot is already zeroed by
    /// `PhysicalCodePage::new` and doesn't need this (see [`Self::claim`]'s
    /// doc comment for why re-running it there on every ordinary page
    /// arrival would be wasted, hot-path work).
    ///
    /// Explicitly zeroing every `entries[i].gen` here (not just `func`/
    /// `flags`) is the important part: that field doubles as
    /// `PhysicalCodePage::count_dispatch_and_check_threshold`'s pre-publish
    /// call counter (`j2 min-calls`) when the entry has never been
    /// published — without this, a slot reused after a flush would inherit
    /// whatever count was sitting there from its previous physical page's
    /// occupancy, silently skewing the new page's own warm-up window.
    fn reset_entries_and_bitmaps(&mut self) {
        for word in self.saved_bits.iter() { word.store(0, std::sync::atomic::Ordering::Relaxed); }
        for entry in self.entries.iter_mut() {
            entry.func = std::ptr::null();
            entry.gen.store(0, std::sync::atomic::Ordering::Relaxed);
            entry.flags.store(0, std::sync::atomic::Ordering::Relaxed);
            #[cfg(feature = "developer")]
            {
                entry.instr_count = 0;
                entry.code_size = 0;
                entry.call_count.store(0, std::sync::atomic::Ordering::Relaxed);
                entry.block_include_count.store(0, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Bump-allocate this (already-constructed, already-clean) slot into
    /// service for a newly-arrived physical page — the in-place counterpart
    /// to what used to be a fresh `PhysicalCodePage::new(pfn, gen)` +
    /// `Vec::push`. Deliberately does **not** reset bitmaps/entries itself —
    /// `page_for` calls this on every distinct-PFN arrival, which happens
    /// constantly during real execution (every newly-touched physical page,
    /// not just at startup), so re-zeroing all 1024 entries here on the hot
    /// path would be wasted work almost every time: a slot only ever reaches
    /// `claim` in one of two states, both already clean — freshly
    /// constructed (`PhysicalCodePage::new` zeroes everything) or freshly
    /// reset by `mega_flush` (`reset_to_unclaimed`, called on exactly the
    /// slots it's returning to the unclaimed pool, before any of them can be
    /// claimed again). The `debug_assert!` below exists specifically to
    /// catch a violation of that invariant (e.g. a future caller reusing a
    /// slot without going through `mega_flush` first) in a diagnostics
    /// build, rather than paying a real-build cost to re-verify it on every
    /// single claim.
    pub fn claim(&mut self, pfn: Pfn, gen: *const AtomicU64) {
        debug_assert!(std::ptr::eq(self.gen, &NEVER_COMPILABLE_GEN) && self.pfn == 0,
            "claim() called on a slot that wasn't clean (pfn={}) — every path that reuses a slot must reset it first (see reset_to_unclaimed)",
            self.pfn);
        debug_assert!((0..ENTRIES_PER_PAGE).all(|i| !self.is_published(i)),
            "claim() called on a slot with a still-published entry — mega_flush must reset_to_unclaimed before this slot can be reused");
        self.pfn = pfn;
        self.gen = if gen.is_null() { &NEVER_COMPILABLE_GEN } else { gen };
    }

    /// Return this slot to the fully-unclaimed state (`pfn = 0`, `gen`
    /// pointed at the shared [`NEVER_COMPILABLE_GEN`] fallback) —
    /// `Jitv2::mega_flush`'s in-place counterpart to what used to be
    /// `Vec::clear()` dropping every `PhysicalCodePage` outright.
    pub fn reset_to_unclaimed(&mut self) {
        self.reset_entries_and_bitmaps();
        self.pfn = 0;
        self.gen = &NEVER_COMPILABLE_GEN;
    }

    /// Current generation count for this page. `self.gen` is never null (see
    /// its own doc comment) — a page whose backing device has no real gen
    /// tracking (MMIO, etc) reads the shared, never-bumped
    /// [`NEVER_COMPILABLE_GEN`] fallback here instead, so this is
    /// unconditionally safe to call on any claimed or unclaimed page, no
    /// branch needed on the caller's part.
    #[inline]
    pub fn current_gen(&self) -> u64 {
        // Relaxed: publish-time (§6.5) and mutation-time (§7) orderings are
        // established by the fetch_or/re-read pair at those call sites, not here.
        unsafe { (*self.gen).load(std::sync::atomic::Ordering::Relaxed) }
    }

    /// Whether `entries[offset_word]`'s `ENTRY_VALID` flag is set — i.e. some
    /// compile has published a function here, without regard to whether it's
    /// still fresh against the page's current gen. Callers that need "is this
    /// dispatchable right now" want [`Self::is_entry_valid`]; this is for the
    /// dispatch-trigger gate (§6.1.2's `entry_bits[pfn].test(offset)`), which
    /// probes the flag first and only then decides exec-vs-recompile from gen.
    #[inline]
    pub fn is_published(&self, offset_word: usize) -> bool {
        self.entries[offset_word].flags.load(std::sync::atomic::Ordering::Acquire) & ENTRY_VALID != 0
    }

    /// Whether `entries[offset_word]` is both published (`ENTRY_VALID`) and
    /// still fresh (its recorded gen matches the page's current gen, §6.5).
    /// A set valid flag whose gen has drifted is stale — the caller should
    /// treat it as unpublished (downgrade to interpreter, §6.1.2) rather than
    /// dispatch it.
    ///
    /// The `gen` load is Acquire, not Relaxed: it's the real synchronization
    /// point for `entries[offset_word].func` (see `publish`'s doc comment) —
    /// `ENTRY_VALID` alone doesn't provide fresh ordering across a recompile
    /// of an already-published entry, since the flag's *value* doesn't change
    /// on that path. Callers must not read `func` after this returns `true`
    /// without going through this same `gen` load (i.e. don't cache
    /// `is_published`'s result and reuse it — always call `is_entry_valid`
    /// fresh right before trusting `func`), or the ordering guarantee is
    /// lost.
    #[inline]
    pub fn is_entry_valid(&self, offset_word: usize) -> bool {
        self.is_published(offset_word)
            && self.entries[offset_word].gen.load(std::sync::atomic::Ordering::Acquire) == self.current_gen()
    }

    /// Whether `offset_word` has been sticky-rejected by the compiler (§6.4).
    #[inline]
    pub fn is_denylisted(&self, offset_word: usize) -> bool {
        self.entries[offset_word].flags.load(std::sync::atomic::Ordering::Relaxed) & ENTRY_DENYLISTED != 0
    }

    /// Count one interpreter dispatch of `offset_word` *before* it has ever
    /// been compiled, returning `true` once `threshold` dispatches have been
    /// observed (the caller should send a `CompileRequest` now) and `false`
    /// otherwise (not hot enough yet — stay on the interpreter this time).
    ///
    /// Reuses `entries[offset_word].gen` as the counter storage: that field
    /// has no meaning before an entry is ever published (`is_entry_valid`
    /// only ever reads it after first checking `is_published`, per that
    /// method's own doc comment — a pre-publish `gen` value is simply never
    /// consulted by anything), so borrowing it here for an unrelated purpose
    /// during that same pre-publish window is safe: the two uses never
    /// overlap in time for a given entry, and `publish()` unconditionally
    /// overwrites it with the entry's real `gen_snap` the moment it does get
    /// compiled, erasing whatever count was here. No new field needed.
    ///
    /// `threshold == 0` means "always ready" (returns `true` on the very
    /// first call, matching today's behavior with the gate absent entirely)
    /// — the caller (`exec_decoded`) is expected to pass 0 for every
    /// dispatch path that must stay deterministic/immediate (tests,
    /// `jitv2_inline_compile`'s synchronous "run it now" contract,
    /// verification harnesses), and the real tunable value
    /// (`j2 min-calls`) otherwise.
    #[inline]
    pub fn count_dispatch_and_check_threshold(&self, offset_word: usize, threshold: u64) -> bool {
        if threshold == 0 {
            return true;
        }
        let prev = self.entries[offset_word].gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        prev + 1 >= threshold
    }

    /// Sticky-reject `offset_word` (§6.4): the compiler declined this offset
    /// (no emitter for some visited instruction, or `walk` found it excluded
    /// outright) and arrival/queueing should stop re-requesting a compile
    /// that will only be declined again. Cleared on a gen bump alongside
    /// `ENTRY_VALID` — not implemented yet (no gen-triggered reclassification
    /// exists until invalidation lands, §7).
    #[inline]
    pub fn denylist(&self, offset_word: usize) {
        self.entries[offset_word].flags.fetch_or(ENTRY_DENYLISTED, std::sync::atomic::Ordering::Relaxed);
    }

    /// Sticky-reject every offset on this page at once. Not on any hot path
    /// (a page-wide, one-time operation) — used to make a whole page
    /// permanently non-dispatchable, e.g. tests that need a deterministic
    /// "this page never compiles/dispatches, everything runs on the
    /// interpreter" guarantee.
    pub fn denylist_all(&self) {
        for entry in self.entries.iter() { entry.flags.fetch_or(ENTRY_DENYLISTED, std::sync::atomic::Ordering::Relaxed); }
    }

    /// Un-publish `offset_word`'s entry — clears `ENTRY_VALID` only, NOT
    /// denylist: the artifact itself isn't wrong in general, just stale for
    /// *this* dispatch (`emit_fpu_entry_guard`'s FR-mismatch case — the
    /// region was compiled for the wrong FR mode, so every future dispatch
    /// through the normal gate would hit this exact same guard failure
    /// again, forever, without ever getting a chance at a fresh compile for
    /// the FR mode that's actually live). Unlike `denylist`, a later
    /// dispatch is expected and welcome to recompile this offset — most
    /// likely for the *other* FR mode, but `try_schedule`'s normal miss path
    /// handles that exactly like any other never-yet-compiled offset.
    /// `entries[offset_word].func` itself is deliberately left in place
    /// (same "may be stale until slot reuse" contract as `JitEntry::func`'s
    /// own doc comment) — nothing between clearing this flag and the next
    /// `publish` ever reads `func` without first re-checking `is_entry_valid`.
    #[inline]
    pub fn kill(&self, offset_word: usize) {
        self.entries[offset_word].flags.fetch_and(!ENTRY_VALID, std::sync::atomic::Ordering::Release);
    }

    /// Test-and-set `offset_word`'s `ENTRY_SCHEDULED` flag: returns `true`
    /// (and sets the flag) only if it was previously clear, i.e. only the
    /// first caller for a given offset should actually build and send a
    /// `CompileRequest` — every other concurrent/subsequent caller sees
    /// `false` and skips it, since a request for this offset is already in
    /// flight. `fetch_or` alone (like `denylist`'s) isn't enough here: unlike
    /// sticky rejection, where every caller doing the same idempotent write
    /// is fine, this flag exists specifically to distinguish "I am the one
    /// who should send the request" from "someone else already did" —
    /// `fetch_or`'s return value (the *previous* bits) gives exactly that
    /// distinction for free, no separate compare-exchange needed.
    #[inline]
    pub fn try_schedule(&self, offset_word: usize) -> bool {
        let prev = self.entries[offset_word].flags.fetch_or(ENTRY_SCHEDULED, std::sync::atomic::Ordering::Relaxed);
        prev & ENTRY_SCHEDULED == 0
    }

    /// Clear `offset_word`'s `ENTRY_SCHEDULED` flag — called once
    /// `handle_request` (`comp.rs`) has decided the offset one way or the
    /// other (published or denylisted), so a future legitimate re-request
    /// for this offset (e.g. after a gen bump invalidates a stale artifact)
    /// isn't permanently blocked by a stale scheduled flag from a request
    /// that already finished.
    #[inline]
    pub fn clear_scheduled(&self, offset_word: usize) {
        self.entries[offset_word].flags.fetch_and(!ENTRY_SCHEDULED, std::sync::atomic::Ordering::Relaxed);
    }

    /// Publish a freshly compiled function at `offset_word` (§6.5 step 4):
    /// write `func` first, then Release-store `gen` — in that order, always.
    /// `gen_snap` is the page generation read *before* the compile started
    /// (§6.5 step 2); if the page has mutated since (current gen no longer
    /// matches), the publish is aborted before the bit is ever (re)set, so a
    /// racing writer that invalidated this compile during codegen can't have
    /// its mutation silently shadowed by a stale artifact. Returns `true` if
    /// the entry was actually published.
    ///
    /// The `func`-then-`gen` order (and `gen`'s Release/Acquire ordering,
    /// paired with `is_entry_valid`'s Acquire load) is load-bearing, not
    /// cosmetic — it's what makes *recompiling* an already-published entry
    /// safe, which `ENTRY_VALID` alone cannot do. §2.5's "no recompile of
    /// existing artifacts" design intent means `handle_request`
    /// (`comp.rs`) never calls this on an entry that's currently
    /// `is_entry_valid` — but an entry whose gen has drifted stale (page
    /// mutated, flag still set) *does* get recompiled in place. On that path,
    /// `ENTRY_VALID`'s Release/Acquire pairing provides no fresh
    /// synchronization at all: the flag doesn't change value (it was already
    /// set), so a dispatcher's Acquire-load of it can be satisfied by a stale
    /// cached observation from the *original* publish, with no
    /// happens-before relationship to this recompile's writes whatsoever.
    /// Without `gen` itself carrying the ordering, a dispatcher could
    /// observe the *new* `gen` (matching `current_gen()`, so `is_entry_valid`
    /// reports true) paired with the *old* `func` pointer — silently calling
    /// a compiled function for a page state it was never actually compiled
    /// against. Making `gen` the synchronization point (write `func` before
    /// it, Release it, Acquire-load it before ever trusting `func`) closes
    /// that window using the field that already exists for exactly this
    /// "has this artifact been superseded" purpose, rather than adding a
    /// second one.
    /// `instr_count`: dev-only diagnostic (`JitEntry::instr_count`, `j2 pcp`)
    /// — the number of instructions the just-compiled region covers.
    /// `code_size`: dev-only diagnostic (`JitEntry::code_size`) — compiled
    /// machine code size in bytes. Both ignored (but still accepted, to keep
    /// this signature stable across feature combinations) when the
    /// `developer` feature is off.
    pub fn publish(&self, offset_word: usize, func: *const (), gen_snap: u64, #[allow(unused_variables)] instr_count: usize, #[allow(unused_variables)] code_size: u32) -> bool {
        if gen_snap != self.current_gen() {
            return false; // page already mutated past gen_snap; discard rather than publish a stale artifact
        }
        // Safety: `func` is a raw pointer write on a `JitEntry` shared behind
        // `&self` — sound because no concurrent reader trusts `func` without
        // first Acquire-loading `gen` below and observing it equal to
        // current_gen() (see this function's doc comment for why that's the
        // synchronization point, not `ENTRY_VALID`), and no other writer
        // targets the same offset concurrently (the compile thread is
        // single-threaded and processes requests in order).
        unsafe {
            let entries = self.entries.as_ptr() as *mut JitEntry;
            (*entries.add(offset_word)).func = func;
            #[cfg(feature = "developer")]
            {
                (*entries.add(offset_word)).instr_count = instr_count as u16;
                (*entries.add(offset_word)).code_size = code_size;
            }
        }
        self.entries[offset_word].gen.store(gen_snap, std::sync::atomic::Ordering::Release);
        // fetch_or on an already-set ENTRY_VALID (the recompile-of-a-stale-entry
        // case) is a no-op on the flag's value but still executes as a Release
        // op — harmless to keep doing unconditionally here since it costs
        // nothing extra and keeps first-publish's existing Acquire/Release
        // contract intact for `is_published`'s own callers. Only ever touches
        // the ENTRY_VALID bit — ENTRY_DENYLISTED/ENTRY_SCHEDULED (if either
        // happens to also be set on this same byte) are left untouched.
        self.entries[offset_word].flags.fetch_or(ENTRY_VALID, std::sync::atomic::Ordering::Release);
        true
    }

    /// Corpus-collection scaffolding (`saved_bits`, see its field doc):
    /// whether this offset's page snapshot has already been dumped to disk.
    #[inline]
    pub fn is_saved(&self, offset_word: usize) -> bool {
        let word = offset_word >> 6;
        let bit = offset_word & 63;
        self.saved_bits[word].load(std::sync::atomic::Ordering::Relaxed) & (1 << bit) != 0
    }

    /// Corpus-collection scaffolding: mark `offset_word` as saved. Returns
    /// `true` if this call is the one that set the bit (i.e. the caller
    /// should actually write the file) — using `fetch_or`'s previous value
    /// means concurrent duplicate work is impossible even if this is ever
    /// called from more than one thread for the same page.
    #[inline]
    pub fn mark_saved(&self, offset_word: usize) -> bool {
        let word = offset_word >> 6;
        let bit = offset_word & 63;
        let prev = self.saved_bits[word].fetch_or(1 << bit, std::sync::atomic::Ordering::Relaxed);
        prev & (1 << bit) == 0
    }

    /// Dev-only (`j2 pcp`): saturating-increment `offset_word`'s
    /// `block_include_count` (see the field's doc). Called once per visited
    /// word by `comp::handle_request`/`handle_request_deferred` after a
    /// successful compile, so overlapping regions accumulate. Saturates at
    /// `u8::MAX` rather than wrapping — the exact count past 255 is
    /// uninteresting, "pegged" is the signal.
    #[cfg(feature = "developer")]
    #[inline]
    pub fn note_block_include(&self, offset_word: usize) {
        let c = &self.entries[offset_word].block_include_count;
        // Relaxed CAS loop: this is a diagnostic counter with no ordering
        // relationship to any other state, and contention is effectively nil
        // (compiles for a given page are serialized), so a plain
        // try_update-style saturating add is all that's needed.
        let _ = c.try_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |v| Some(v.saturating_add(1)),
        );
    }
}

/// Slot index into [`Jitv2::pages`].
pub type PageSlot = u32;

/// JIT v2 engine state embedded in the mips executor.
///
/// Owns the [`PhysicalCodePage`] pool (§2.4): a single array, allocated once
/// at full `capacity` in `Jitv2::new` (every slot pre-built as unclaimed —
/// see `PhysicalCodePage::new`'s doc comment), never resized or reallocated
/// afterward. Pages are handed out on demand (first arrival at a given PFN)
/// by bump-claiming the next unclaimed slot (`PhysicalCodePage::claim`) — no
/// per-page allocation or move at claim time, matching §6.3's arena-allocator
/// model ("Bump-only allocation, reset only at `flush_all()`" — D6.2 lock-in
/// 1, generalized here to the page pool itself since PCPs, like wrapper
/// slots, only ever grow monotonically between flushes). This is stronger
/// than the old `Vec::push`-based growth: that also never reallocated once
/// capacity was reserved, but each `push` still moved a freshly-constructed
/// `PhysicalCodePage` value into place — with `entries` now inline (not
/// boxed), that move would copy the whole 1024-entry table. Claiming a
/// pre-existing slot in place avoids that entirely: nothing is ever copied
/// after `Jitv2::new` builds the array.
///
/// Lookup from `pfn` to pool slot goes through `pfn_to_slot`. This is a
/// HashMap for now — simplest thing that works. If page-switch lookup shows
/// up hot in profiling, the design doc's dense pfn-indexed alternative
/// (§2.4) is the fallback; not built preemptively.
pub struct Jitv2 {
    /// The full-capacity page pool, allocated once — see this struct's own
    /// doc comment. Indices are stable for the pool's entire lifetime,
    /// including across `mega_flush` (`mega_flush` resets slots in place,
    /// it never shrinks or reallocates this array), so pointers into it —
    /// e.g. the executor's current-PCP pointer, `CompileRequest::page` —
    /// stay valid for the process's whole lifetime, not just "until the next
    /// flush" as before. A `CompileRequest`/PCP pointer surviving a flush it
    /// raced against and landing back in a slot that's since been reclaimed
    /// for a different physical page is still a real hazard (unchanged from
    /// before — `worker_loop`'s `drain_pending` and the various
    /// gen-mismatch/`is_entry_valid` checks are what actually guard against
    /// stale content, not pointer validity) — array-lifetime stability alone
    /// doesn't imply content freshness.
    pages: Box<[PhysicalCodePage]>,
    /// Number of slots in `pages` that have been claimed (bump-allocated,
    /// `PhysicalCodePage::claim`) since construction or the last
    /// `mega_flush` — the pool's actual live extent; `pages[next_free..]`
    /// are still in the unclaimed state. Replaces the old `Vec::len()`
    /// (which used to double as this count, back when the pool grew via
    /// `push`).
    next_free: usize,
    /// pfn -> index into `pages`. Consulted only on a page switch (fetch
    /// lands on a different PFN than the currently-tracked one) — not on
    /// every fetch.
    pfn_to_slot: HashMap<Pfn, PageSlot>,
    /// Pool capacity, fixed at construction (== `pages.len()`). Claiming past
    /// this triggers `mega_flush` (the "ran out of PCPs" resource-exhaustion
    /// trigger).
    capacity: usize,
    /// Compile-request fifo + worker thread (§6.4/§9). Constructed alongside
    /// the pool; `start()`/`stop()` are separate calls so the executor
    /// controls the compile thread's lifetime independently (mirrors
    /// `MipsCpu`'s own start/stop, e.g. no compile thread while paused).
    pub compile_queue: CompileQueue,
    /// The one `Codegen` (and therefore its one Cranelift `JITModule`/memory
    /// arena) shared between `jitv2_inline_compile` and the async compile
    /// thread — the two modes are mutually exclusive at any given moment (a
    /// monitor command, `j2 inline on|off`, is the only way to switch), so
    /// there is no reason for each to carry its own separate arena that then
    /// each needs its own independent flush bookkeeping. `None` exactly when
    /// the compile-thread worker currently owns it (moved out by
    /// `CompileQueue::start`, handed back by `stop`); `Some` otherwise —
    /// whenever inline mode is what's actually running, or the queue simply
    /// isn't started. `flush` operates on whichever `Some` it finds; callers
    /// on the inline path (`exec_decoded`'s dispatch gate,
    /// `jitv2_track_pcp`'s pool-exhaustion handler) take it via
    /// `codegen.lock()` for the duration of one compile/reset, same
    /// exclusion discipline the compile thread already has by construction
    /// (only it ever touches its own moved-out copy).
    pub codegen: Mutex<Option<crate::jitv2::codegen::Codegen>>,
    /// Event counters, read only under `j2 status` (dev-only display — see
    /// `JitStats`'s own doc comment for why the *fields themselves* still
    /// exist and get threaded through unconditionally: it's cheaper to carry
    /// one always-present `Arc` clone than to duplicate every function on
    /// its path behind `#[cfg(feature = "developer")]`). `Arc`, not embedded
    /// by value, so `CompileQueue::start` can clone a handle into the worker
    /// thread once (same pattern as `function_count`/`cpu`/`jitv2` below)
    /// instead of every `handle_request` call locking `Jitv2` just to touch
    /// a counter.
    pub stats: Arc<JitStats>,
}

/// Per-instruction-count bucket of a `Jitv2::code_size_by_instr_count()`
/// scan: how many published entries landed at this instruction count, and
/// the raw (not host-page-rounded — that rounding only matters for the
/// arena-usage estimate, `code_bytes_used`; here we want the real
/// per-function code size Cranelift actually emitted) `code_size` bytes
/// across them, as count/sum/min/max — enough to report both an average and
/// a spread without keeping every individual size around.
#[cfg(feature = "developer")]
#[derive(Debug, Clone, Copy)]
pub struct CodeSizeBucket {
    pub count: u32,
    pub sum_bytes: u64,
    pub min_bytes: u32,
    pub max_bytes: u32,
}

impl Jitv2 {
    /// Allocate the full page pool up front, `capacity` slots, every one
    /// starting unclaimed (`PhysicalCodePage::new(0, null)`) — see `Jitv2`'s
    /// own doc comment for why this is a single one-shot allocation rather
    /// than lazy `push`-based growth. Sizing is a Phase 0 measurement per the
    /// design doc (§9); start with whatever the caller passes and let
    /// `mega_flush` absorb a too-small guess rather than trying to size this
    /// perfectly up front — getting it wrong now just means an earlier
    /// flush, not a correctness problem. Does not start the compile thread —
    /// call `compile_queue.start()`.
    pub fn new(capacity: usize) -> Self {
        Self {
            pages: (0..capacity).map(|_| PhysicalCodePage::new(0, std::ptr::null())).collect(),
            next_free: 0,
            pfn_to_slot: HashMap::new(),
            capacity,
            compile_queue: CompileQueue::new(),
            codegen: Mutex::new(Some(crate::jitv2::codegen::Codegen::new())),
            stats: Arc::new(JitStats::default()),
        }
    }

    /// Look up the pool slot for `pfn`, claiming the next unclaimed slot
    /// in place (`PhysicalCodePage::claim`, `gen_ptr(phys_addr)` on the bus)
    /// if this is the first arrival at this page. Returns `None` if the pool
    /// is exhausted — the caller (mips exec thread) is responsible for
    /// running `mega_flush` and retrying.
    ///
    /// `phys_addr` must be the physical address whose containing page is
    /// `pfn` (i.e. `pfn == phys_addr / PAGE_SIZE`) — passed separately
    /// rather than reconstructed here because callers already have it from
    /// translation and multiplying back out is wasted work on the hot path.
    pub fn page_for(&mut self, pfn: Pfn, phys_addr: u32, bus: &dyn BusDevice) -> Option<PageSlot> {
        if let Some(&slot) = self.pfn_to_slot.get(&pfn) {
            return Some(slot);
        }
        if self.next_free >= self.capacity {
            return None;
        }
        let gen = bus.gen_ptr(phys_addr);
        let slot = self.next_free as PageSlot;
        self.pages[slot as usize].claim(pfn, gen);
        self.next_free += 1;
        self.pfn_to_slot.insert(pfn, slot);
        Some(slot)
    }

    /// Raw pointer to the page at `slot`. Valid for the process's entire
    /// lifetime (see the `pages` field doc — the array itself never moves or
    /// resizes, including across `mega_flush`) — used to set the executor's
    /// current-PCP pointer without holding a borrow of `self`.
    #[inline]
    pub fn page_ptr(&mut self, slot: PageSlot) -> *mut PhysicalCodePage {
        &mut self.pages[slot as usize] as *mut PhysicalCodePage
    }

    /// Number of pool slots currently claimed (since construction or the
    /// last `mega_flush`). Exit-time diagnostic — see `MipsCpu::stop`.
    #[inline]
    pub fn pages_used(&self) -> usize {
        self.next_free
    }

    /// Pool capacity, as passed to `new()`.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Sum, across every published entry in every pooled page, of each
    /// entry's `JitEntry::code_size` **rounded up to
    /// `Codegen::HOST_PAGE_SIZE`** — dev-only diagnostic (`j2 stats`), the
    /// best available proxy for the shared `Codegen`'s actual Cranelift
    /// memory-arena usage (see `JitEntry::code_size`'s doc comment for why
    /// nothing more direct is available). Rounding matters: `code_size` is
    /// raw compiled-machine-code bytes (~215 bytes/function observed for a
    /// single-instruction region), but `ArenaMemoryProvider` gives every
    /// function its own segment, always rounded up to a full host page
    /// regardless of actual size (`CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES`'s doc
    /// comment — confirmed live: the arena exhausted at exactly
    /// `ARENA_RESERVE_SIZE / HOST_PAGE_SIZE` functions, not the ~2.4M a
    /// byte-size-only estimate would predict) — summing raw `code_size`
    /// alone would under-report real arena consumption by roughly 19x at
    /// that average function size, which is exactly the gap that made the
    /// original OOM investigation confusing (small `code_bytes_used`
    /// numbers right up until the arena actually exhausted). O(pages ×
    /// ENTRIES_PER_PAGE); fine for an on-demand monitor command, not called
    /// from any hot path.
    #[cfg(feature = "developer")]
    pub fn code_bytes_used(&self) -> u64 {
        let page_size = crate::jitv2::codegen::Codegen::HOST_PAGE_SIZE;
        self.pages.iter()
            .map(|page| {
                (0..ENTRIES_PER_PAGE)
                    .filter(|&off| page.is_published(off))
                    .map(|off| {
                        let raw = page.entries[off].code_size as u64;
                        raw.div_ceil(page_size) * page_size
                    })
                    .sum::<u64>()
            })
            .sum()
    }

    /// Histogram of `JitEntry::instr_count` across every published entry in
    /// every pooled page, paired with each bucket's code-size distribution —
    /// dev-only diagnostic (`j2 status`), a full scan like `code_bytes_used`
    /// (O(pages × ENTRIES_PER_PAGE), fine for an on-demand monitor command).
    /// Indexed by instruction count directly (`result[n]` = stats for
    /// published entries whose region covers exactly `n` instructions);
    /// `n == 0` is always absent (a published entry's region always covers
    /// at least its own head instruction). Lets you see the real
    /// distribution against `MAX_INSTRS_PER_COMPILE` (`comp.rs`) — regions
    /// land at fewer instructions than the budget whenever a branch/excluded
    /// instruction/page boundary cuts the walk short, so this is ground
    /// truth, not just "the budget was N" — and, per bucket, whether code
    /// size scales roughly linearly with instruction count or has wide
    /// per-region variance (e.g. FPU regions paying the CU1/FR guard's fixed
    /// overhead regardless of instruction count).
    #[cfg(feature = "developer")]
    pub fn code_size_by_instr_count(&self) -> Vec<Option<CodeSizeBucket>> {
        let mut hist: Vec<Option<CodeSizeBucket>> = Vec::new();
        for page in self.pages.iter() {
            for off in 0..ENTRIES_PER_PAGE {
                if !page.is_published(off) { continue; }
                let entry = &page.entries[off];
                let n = entry.instr_count as usize;
                let size = entry.code_size;
                if n >= hist.len() { hist.resize(n + 1, None); }
                match &mut hist[n] {
                    Some(bucket) => {
                        bucket.count += 1;
                        bucket.sum_bytes += size as u64;
                        bucket.min_bytes = bucket.min_bytes.min(size);
                        bucket.max_bytes = bucket.max_bytes.max(size);
                    }
                    slot @ None => {
                        *slot = Some(CodeSizeBucket { count: 1, sum_bytes: size as u64, min_bytes: size, max_bytes: size });
                    }
                }
            }
        }
        hist
    }

    /// Every claimed page in the pool (`pages[..next_free]`) — same live
    /// extent `code_bytes_used`/`code_size_by_instr_count` scan, exposed as a
    /// borrow rather than folded into an aggregate for callers (`j2 html`)
    /// that need the raw per-page detail instead of a summary statistic.
    /// `pages` itself stays private (index stability, see the field's own
    /// doc comment, is an invariant only this module should rely on).
    pub fn claimed_pages(&self) -> &[PhysicalCodePage] {
        &self.pages[..self.next_free]
    }

    /// Reset the JIT to its initial state: drop every compiled artifact and
    /// every tracked page, and reset the pool allocator. The MAME-style
    /// "flush the world" response to running out of any bump-allocated JIT
    /// resource — page pool slots here; code arena and wrapper slots join
    /// this call as those pieces land (§6.3's `flush_all()`, of which this
    /// is the first caller: arena-full, `restore`, `rollback` all route
    /// through one routine).
    ///
    /// Does not yet demote promoted decode-entry handlers or null
    /// entry_table slots (§6.1.3, §6.3) — there are none to demote until
    /// the dispatcher/compiler land. Once they exist, this is where that
    /// walk goes, on the executor thread, before the reset loop below.
    ///
    /// Resets every claimed slot in place (`PhysicalCodePage::reset_to_unclaimed`)
    /// rather than the old `pages.clear()` — the array itself is never
    /// dropped/reallocated (see `Jitv2`'s own doc comment), so returning
    /// every slot to its unclaimed state, including zeroing every entry's
    /// `gen` (doubling as the pre-publish call counter — see
    /// `reset_entries_and_bitmaps`'s doc comment for why that specifically
    /// matters here, not just for `func`/`flags`), is what "flush"
    /// means now.
    ///
    /// Private: real callers want [`Self::flush`], which wraps this with the
    /// compile-queue pause every caller of this used to have to remember
    /// (see that method's doc comment for why bundling it here, rather than
    /// leaving it the caller's responsibility, is what makes this whole
    /// operation self-contained now that `Jitv2` owns its own compile-queue
    /// lifecycle independently of `MipsCpu::stop()`/`start()`).
    fn mega_flush(&mut self) {
        for page in self.pages[..self.next_free].iter_mut() {
            page.reset_to_unclaimed();
        }
        self.next_free = 0;
        self.pfn_to_slot.clear();
        // Status-bar feedback (disp.rs's StatusBar): reset the code-arena
        // gauge to empty and bump the flush-event counter so the bar
        // flashes once this frame — see JitFeedback's own doc comment for
        // why a counter, not a bool. Arena fill itself is tracked from
        // Codegen::packing_stats() at each compile (worker_loop /
        // jitv2_inline_compile), not here — this only needs to zero it
        // because mega_flush is also what resets the real Cranelift arena
        // (Codegen::reset, called by both of this fn's callers right after).
        crate::jit_feedback::JIT_FEEDBACK.set_arena_fill(0, CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES);
        crate::jit_feedback::JIT_FEEDBACK.record_flush();
    }

    /// Self-contained page-pool + compiled-code-arena flush, called FROM the
    /// CPU thread (`jitv2_track_pcp`'s pool-exhaustion handler). The caller
    /// is already "as good as stopped" — it's the one executing this,
    /// synchronously, not racing itself — so this never touches the CPU at
    /// all, only the *compile* queue (the other side): pauses it if it's
    /// running (every in-flight/queued `CompileRequest` points into
    /// `self.pages`, which `mega_flush` is about to clear — the worker must
    /// be fully joined and drained first, or it could dereference a page
    /// mid-drop out from under it; `stop()` also hands back whatever
    /// `Codegen` the worker was using), clears the pool, resets the
    /// `Codegen` (frees the Cranelift memory arena — `Codegen::function_count`'s
    /// doc comment for why nothing else ever does), and hands the reset
    /// `Codegen` back to wherever it came from — restarting the compile
    /// queue if (and only if) it was the one running, or back into
    /// `self.codegen` (idle slot) if inline dispatch owned it instead. Which
    /// of those it was is exactly what `compile_queue.stop()`'s `Option`
    /// tells us; blindly restarting the queue regardless used to silently
    /// steal the `Codegen` away from inline dispatch (see the code's own
    /// comment for the failure mode this caused).
    ///
    /// The caller (`jitv2_track_pcp`) is still responsible for its own
    /// `nanotlb_invalidate()`/`self.pcp = null` afterward — this type has no
    /// executor access to do that itself. See [`Self::flush_from_jit_thread`]
    /// for the mirror-image case (compile thread detects its own growth,
    /// must pause the *CPU* instead).
    ///
    /// # Safety
    /// Same contract as `Codegen::reset` — no `JitFn` this `Codegen` ever
    /// produced may still be reachable/callable after this returns.
    /// Guaranteed here: every `PhysicalCodePage` that could reference such a
    /// function is cleared by `mega_flush` in the same operation, and the
    /// compile queue is fully stopped (joined) before that clear runs.
    pub unsafe fn flush_from_cpu_thread(&mut self, bus: Arc<dyn BusDevice>) {
        // `compile_queue.stop()` returns non-empty only if the async worker
        // was actually running (i.e. threaded/`j2 inline off` mode) —
        // that's also the only case this should restart it afterward. When
        // it returns empty, the codegen was already idle in `self.codegen`
        // (inline/`j2 inline on` mode, the default), and it must be reset
        // and stay there, NOT go to the compile queue — unconditionally
        // restarting the queue here regardless of which mode was actually
        // active used to silently steal the codegen out from under inline
        // dispatch: every inline compile after the first pool-exhaustion
        // flush would find `self.codegen` empty and silently no-op
        // (mips_exec.rs's `if let Some(codegen) = codegen.as_mut()` guard
        // swallows it with no error), while `j2 inline` still reported "on"
        // the whole time.
        let stopped = self.compile_queue.stop();
        let was_threaded = !stopped.is_empty();
        let mut function_count = 0;
        // `stop()` already drains the ring itself (see its own doc comment)
        // — this second call is a harmless no-op belt-and-suspenders: every
        // request still queued at stop-time points into the pool
        // `mega_flush` is about to clear, and this must happen before that
        // clear runs, not after (see `drain_pending`'s own doc comment for
        // the live-confirmed crash that reasoning guards against).
        self.compile_queue.drain_pending_queue();
        if was_threaded {
            // The threaded queue's own Codegen(s) are discarded (dropped,
            // not reset — `start()` will build fresh ones internally on
            // restart below, same as it always does on a first start).
            // `Codegen`'s `mem::forget`-on-drop semantics (see `reset`'s own
            // doc comment) mean this leaks the old arena, same as any other
            // orphaned Codegen — acceptable here since mega_flush below is
            // about to invalidate every PhysicalCodePage entry that could
            // have pointed into it anyway.
            function_count = stopped.iter().map(|c| c.function_count()).sum();
        } else if let Some(codegen) = self.codegen.get_mut().as_mut() {
            function_count = codegen.function_count();
            unsafe { codegen.reset(); }
        }
        eprintln!(
            "jitv2: mega_flush (from cpu thread) — {} / {} pages used, {} functions compiled",
            self.pages_used(), self.capacity(), function_count,
        );
        self.mega_flush();
        if was_threaded {
            let stats = self.stats.clone();
            self.compile_queue.start(bus, stats);
        }
    }

    /// Mirror image of [`Self::flush_from_cpu_thread`], called FROM the
    /// compile thread (`CompileQueue::worker_loop`'s `run_leader_flush`,
    /// when the shared arena's growth crosses
    /// `CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES`). The compile thread can't
    /// pause itself (`CompileQueue::stop()` joins the very thread that
    /// would be calling it — a self-join deadlock), so this pauses the
    /// *CPU* instead: `cpu.stop()` fully joins the CPU's OS thread and
    /// establishes `pcp == null` as a stop-time invariant (`MipsCpu::stop`'s
    /// own doc comment) before returning, which is what makes it safe for
    /// this to clear the page pool directly despite §6.1.3's usual
    /// CPU-thread-only contract — the CPU is provably not running for the
    /// whole operation.
    ///
    /// Page-pool clear only — no longer resets any `Codegen`/arena itself
    /// (unlike before the compile-pool redesign): `run_leader_flush` is the
    /// one that builds the fresh shared arena and rebuilds every worker's
    /// `Codegen` on top of it, since that needs direct access to
    /// `SharedArena`/`BarrierState` types that have no reason to leak into
    /// `Jitv2` itself. `function_count` is just for the log line (the
    /// caller's own `codegen.function_count()` at the moment of the flush —
    /// `Jitv2` has no `Codegen` of its own to read this from anymore).
    ///
    /// # Safety
    /// No `JitFn` any worker's `Codegen` ever produced may still be
    /// reachable/callable anywhere after this returns — guaranteed by the
    /// caller having fully stopped the CPU (and, once a real pool exists,
    /// having every other worker parked) before this runs.
    ///
    /// Must be called with `self` NOT already locked by the caller:
    /// `cpu.stop()` locks the executor and, through it, this same
    /// `Mutex<Jitv2>` again (to print its own page-pool stats — see
    /// `MipsCpu::stop`'s doc comment) — calling this while already holding
    /// `Jitv2`'s lock (e.g. via `jit.lock().flush_from_jit_thread(...)`)
    /// self-deadlocks the compile thread on its own non-reentrant lock.
    /// Callers must take the lock only for the `mega_flush` portion, not
    /// across `cpu.stop()`/`cpu.start()` — see `CompileQueue::worker_loop`'s
    /// `run_leader_flush` for the correct call shape.
    pub unsafe fn flush_from_jit_thread(&mut self, function_count: u32) {
        eprintln!(
            "jitv2: mega_flush (from jit thread) — {} / {} pages used, {} functions compiled",
            self.pages_used(), self.capacity(), function_count,
        );
        self.mega_flush();
    }
}

/// Shared state behind the compile-pool's flush barrier — see
/// `CompileQueue::park_at_barrier`/`run_leader_flush` for the protocol this
/// drives. Wrapped as `Arc<(parking_lot::Mutex<BarrierState>, parking_lot::Condvar)>`,
/// cloned into every worker thread. At today's fixed `thread_count == 1`
/// this is exercised only through its degenerate no-wait path (see
/// `run_leader_flush`'s own doc comment) — the park/wake logic itself is
/// unit-tested directly against real spawned test threads
/// (`tests::barrier_parks_followers_until_leader_resumes_them`), independent
/// of the single-worker `worker_loop` integration, since N=1 alone can never
/// exercise a real park.
/// Shared state behind the compile-pool's single quiesce-everyone barrier —
/// covers both kinds of pool-wide pause a worker can trigger: a full arena
/// flush (rebuild every `Codegen` on a fresh arena) and a forced-seal sweep
/// (mprotect whatever's queued, no arena replacement). These used to be two
/// entirely separate barriers (`BarrierState`/`SealBarrierState`, each with
/// its own leader-election `AtomicBool`) — confirmed live as a genuine
/// deadlock: worker A wins the seal-leader election and waits for worker B
/// to park at the seal barrier, while worker B simultaneously wins the
/// flush-leader election and waits for worker A to park at the flush
/// barrier; neither ever parks for the other since each is itself busy
/// leading its own barrier wait. One barrier with one leader-election gate
/// closes that off structurally: only one kind of pool-wide pause can ever
/// be "in flight" at a time, and every follower always parks at the same
/// single point regardless of which kind the leader is running.
struct BarrierState {
    /// How many non-leader workers are currently parked, waiting for this
    /// generation to end. Reset to 0 by the leader once it resumes everyone.
    parked_count: usize,
    /// Bumped by the leader exactly once, right before `notify_all()` —
    /// after finishing a flush and publishing a fresh arena into
    /// `current_arena`/`current_state` below, or after finishing a
    /// forced-seal sweep (which touches neither field). A parked worker
    /// compares this against the value it observed when it started waiting
    /// — guards against lost/spurious wakeups (a `Condvar::wait` can return
    /// without a matching `notify`, and without this check a worker could
    /// misread that as "the cycle is done" before it actually is).
    generation: u64,
    /// Whether the cycle that just ended (as of the most recent `generation`
    /// bump) was a full flush — if so, every follower must rebuild its own
    /// `Codegen` on `current_arena`/`current_state` before resuming (the old
    /// arena is gone); if not (a seal-only cycle), followers resume in
    /// place with no rebuild at all, since nothing about the arena's
    /// identity changed, only its sealed watermark.
    was_flush: bool,
    /// The fresh shared arena the leader just built — `None` until the
    /// first flush ever completes (never read before then, since nothing
    /// can be parked before any worker has ever detected a flush trigger),
    /// and never touched by a seal-only cycle (see `was_flush`).
    current_arena: Option<Arc<parking_lot::Mutex<crate::jitv2::paged_memory::SharedArena>>>,
    current_state: Option<Arc<crate::jitv2::paged_memory::PagedArenaState>>,
}

impl BarrierState {
    fn new() -> Self {
        Self { parked_count: 0, generation: 0, was_flush: false, current_arena: None, current_state: None }
    }
}

/// The compile thread pool and its inbound MPMC queue (§6.4/§9 Phase 1:
/// "mips exec thread pushes work via fifo, jit thread(s) compile").
///
/// One shared `crossbeam_queue::ArrayQueue` (`queue`) — the CPU thread
/// pushes, every worker thread pops from the same instance, no
/// producer/consumer split needed. Restartable: `stop()` joins every worker
/// thread and collects their `Codegen`s, but the queue itself (an `Arc`) is
/// never taken apart — it just sits idle (still holding whatever's left
/// un-popped) between a `stop()` and the next `start()`. This is what lets
/// `mega_flush` stop-drain-restart the pool around a page-pool reset instead
/// of the queue being single-use (see `mega_flush`'s call site in
/// `mips_exec.rs`, `jitv2_track_pcp`, for why that's required: every
/// `CompileRequest::page` in flight or still queued points into the `Vec`
/// `mega_flush` is about to clear, so every worker must be fully joined,
/// and the queue drained, before that clear happens).
/// Shared body of `CompileQueue::send`: push `req` onto the raw queue,
/// recording `stats` counters under `developer`. Free function (not a
/// `CompileQueue` method) so `MipsExecutor` can call it directly through a
/// bare `Arc<ArrayQueue<CompileRequest>>` (`CompileQueue::queue_handle`)
/// without going through `Jitv2`'s outer mutex at all — the hot dispatch-gate
/// path only ever needs this one push, never the pool/codegen state that
/// mutex actually protects.
pub(crate) fn push_compile_request(
    queue: &crossbeam_queue::ArrayQueue<CompileRequest>,
    req: CompileRequest,
    #[allow(unused_variables)] stats: &JitStats,
) -> bool {
    #[cfg(feature = "developer")]
    {
        // len() before push(): occupancy right now, not after this one
        // lands — matches "depth the compile pool is running at" rather
        // than counting this dispatch's own contribution to it.
        let occupancy = queue.len();
        stats.compile_queue_dispatches.fetch_add(1, Ordering::Relaxed);
        stats.compile_queue_depth_sum.fetch_add(occupancy as u64, Ordering::Relaxed);
    }
    let accepted = queue.push(req).is_ok();
    #[cfg(feature = "developer")]
    if !accepted {
        stats.compile_queue_full.fetch_add(1, Ordering::Relaxed);
    }
    accepted
}

pub struct CompileQueue {
    /// The one shared bounded MPMC ring: the CPU thread pushes
    /// (`CompileQueue::send`), every worker thread pops from the same
    /// `Arc` — no producer/consumer handoff needed (unlike the old
    /// `rtrb`-based SPSC design, `ArrayQueue::push`/`pop` both take `&self`,
    /// so this can just be cloned into each worker thread and left there
    /// permanently, alive across `stop()`/`start()` cycles). Bounded, with
    /// `push` returning `Err` on full rather than blocking — preserves
    /// §6.4's "drop on full, hot pages re-trigger" semantics exactly.
    queue: Arc<crossbeam_queue::ArrayQueue<CompileRequest>>,
    running: Arc<AtomicBool>,
    /// One `JoinHandle` per worker thread — empty when not running. Each
    /// thread hands back its own `Codegen` on exit (see `worker_loop`'s
    /// return type); `stop()` collects all of them.
    threads: Vec<JoinHandle<crate::jitv2::codegen::Codegen>>,
    /// Weak handle to the CPU device, so the worker can stop/start it itself
    /// when a memory-growth flush is needed (`Codegen::function_count()`
    /// crossing `CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES` — see `worker_loop`). Set
    /// once, via `set_cpu`, right after `Arc<MipsCpu<T,C>>` is constructed in
    /// `Machine::new` (mirrors `Mc::set_cpu`'s own `Arc::downgrade`
    /// injection — the CPU doesn't exist yet when `Jitv2`/`CompileQueue` are
    /// first built, so this can't be a constructor parameter). `Weak`, not
    /// `Arc`: a strong reference here would be a real cycle (MipsCpu owns the
    /// executor, which owns `Jitv2`, which would own this) that nothing
    /// would ever break — `Weak::upgrade` failing just means the machine is
    /// mid-teardown and there's nothing to stop/start anymore, which is fine
    /// to skip.
    cpu: Mutex<Option<Weak<dyn Device>>>,
    /// Weak handle back to the owning `Jitv2` (behind whatever `Arc<Mutex<..>>`
    /// the caller shares it in) — the worker needs this to clear the page
    /// pool during its own flush (`worker_loop`'s doc comment explains why
    /// that's safe despite §6.1.3's usual CPU-thread-only contract: the CPU
    /// is provably stopped for the whole operation). Set via `set_owner`,
    /// same injection-after-construction reasoning as `cpu`.
    jitv2: Mutex<Option<Weak<Mutex<Jitv2>>>>,
    /// Per-worker mirror of each worker's own `codegen.function_count()`,
    /// one slot per pool thread, updated after every compile that worker
    /// performs — exists purely so `j2 stats` can read a total function
    /// count while each worker owns its own real `Codegen` by value (not
    /// behind any lock a status command could take without contending the
    /// hot compile path). `function_count()` sums across all slots. Built
    /// fresh (length `thread_count`) by `start_inner` on every start, since
    /// the slot count itself can only change between runs, never while a
    /// pool is live. Each worker's own slot is reset to 0 alongside every
    /// real flush its `Codegen` goes through (`run_leader_flush`/
    /// `park_at_barrier`'s resume path) so it never drifts from what
    /// `Codegen::function_count()` would report if you could ask that
    /// worker's real `Codegen` directly.
    function_counts: Vec<Arc<AtomicU32>>,
    /// Fixed pool size — read once by `start()`, never changed at runtime
    /// (no "j2 threads N" mutation; see `set_thread_count`'s own doc
    /// comment). Defaults to 1 so every existing caller (tests,
    /// `equiv_test.rs`, any non-`Machine::new` construction path) keeps
    /// today's single-worker behavior without having to know this field
    /// exists at all.
    thread_count: usize,
    /// The compile-pool flush barrier — see `BarrierState`'s own doc
    /// comment. Shared (`Arc`) so it can be cloned into the worker thread
    /// (and, once a real pool exists, every worker thread) alongside
    /// `running`.
    barrier: Arc<(parking_lot::Mutex<BarrierState>, parking_lot::Condvar)>,
    /// Leader-election gate for a quiesce cycle (either kind — full flush or
    /// forced-seal-only, see `BarrierState`'s own doc comment for why
    /// there's only one gate covering both): the first worker to detect
    /// either trigger (`compare_exchange(false, true)`) becomes leader and
    /// runs the corresponding sequence; every other worker that also
    /// detects a trigger (or polls this flag at its own next loop-top
    /// check) falls straight to `park_at_barrier` instead of racing to also
    /// lead. Cleared by the leader after publishing the result and
    /// releasing the barrier. At today's `thread_count == 1` this is still
    /// real — exactly one worker exists, so it always "wins" trivially —
    /// but the mechanism is exercised for real once a genuine pool shares
    /// it.
    quiesce_in_progress: Arc<AtomicBool>,
}

impl CompileQueue {
    /// Construct the queue without starting any worker threads. Call
    /// [`Self::start`] to spawn the pool.
    pub fn new() -> Self {
        Self {
            queue: Arc::new(crossbeam_queue::ArrayQueue::new(COMPILE_QUEUE_CAPACITY)),
            running: Arc::new(AtomicBool::new(false)),
            threads: Vec::new(),
            cpu: Mutex::new(None),
            jitv2: Mutex::new(None),
            function_counts: Vec::new(),
            thread_count: 1,
            barrier: Arc::new((parking_lot::Mutex::new(BarrierState::new()), parking_lot::Condvar::new())),
            quiesce_in_progress: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set the pool's fixed worker-thread count — must be called before the
    /// queue is ever `start()`-ed (later calls are a no-op, guarded by
    /// `debug_assert!`, since the whole point is "fixed at startup, no
    /// runtime mutation"). `Machine::new` is the only real caller today
    /// (reading `[jitv2].threads`/`--jitv2-threads` from config — not yet
    /// wired; still hardcoded to the default of 1 everywhere until that
    /// config plumbing lands). `n.max(1)`: a misconfigured 0 would
    /// otherwise silently degrade to "no compile threads at all."
    pub fn set_thread_count(&mut self, n: usize) {
        debug_assert!(self.threads.is_empty(), "set_thread_count must be called before the pool ever starts");
        self.thread_count = n.max(1);
    }

    /// Configured pool size — `j2 threads`'s read-only report.
    #[inline]
    pub fn thread_count(&self) -> usize {
        self.thread_count
    }

    /// Whether the worker pool is currently spawned — the real signal for
    /// "does the async compile thread own compilation right now," since
    /// `Jitv2::codegen` being `Some`/`None` no longer tracks that (the pool
    /// leaves `Jitv2::codegen`'s idle slot alone the whole time it runs,
    /// reserved for `j2 inline on` to take later — see `Machine::new`'s own
    /// comment on `compile_queue.start()`). `j2 status`/`j2 stats` should
    /// check this, not `jit.codegen.lock().is_some()`, to decide whether to
    /// report from the idle slot or from `function_count()`/pool-wide state.
    #[inline]
    pub fn is_running(&self) -> bool {
        !self.threads.is_empty()
    }


    /// Sum, across every worker's own slot, of `Codegen::function_count()`
    /// — see `function_counts`' field doc comment for why this is per-worker
    /// rather than one shared mirror. Meaningful only while the pool is
    /// actually running (`j2 stats`/`j2 status` callers should prefer
    /// `Jitv2::codegen.lock()` when it's `Some`, and fall back to this only
    /// when it's `None`, i.e. the pool currently owns every `Codegen`).
    #[inline]
    pub fn function_count(&self) -> u32 {
        self.function_counts.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    /// Current occupancy of the compile-request queue — readable regardless
    /// of whether any worker is running (the queue is a plain shared `Arc`,
    /// not handed off to a consumer). `j2 status`'s live "how full is it
    /// right now" reading, distinct from `JitStats::compile_queue_depth_sum`'s
    /// historical average.
    #[inline]
    pub fn queue_occupancy(&self) -> usize {
        self.queue.len()
    }

    /// Clone of the underlying `Arc<ArrayQueue<CompileRequest>>` — lets a
    /// caller that only ever needs to `push` (the CPU thread's dispatch-gate
    /// `send`) hold a handle directly, bypassing `Jitv2`'s outer mutex
    /// entirely on the hot path. Safe to hand out freely: the queue is
    /// already the shared, `&self`-only primitive every worker thread pops
    /// from (see the `queue` field's own doc comment), so an extra clone
    /// here is exactly as sound as the ones `start_inner` already makes per
    /// worker.
    pub fn queue_handle(&self) -> Arc<crossbeam_queue::ArrayQueue<CompileRequest>> {
        self.queue.clone()
    }

    /// Inject the CPU device handle — see the `cpu` field's doc comment for
    /// why this is a separate setter rather than a constructor parameter.
    /// Safe to call at any time (even while the worker is running); takes
    /// effect on that worker's next flush-threshold check, or immediately
    /// for a not-yet-started queue.
    pub fn set_cpu(&self, cpu: Weak<dyn Device>) {
        *self.cpu.lock() = Some(cpu);
    }

    /// Inject the weak handle back to the owning `Jitv2` — see the `jitv2`
    /// field's doc comment. Same call-after-construction reasoning as
    /// `set_cpu` (the `Arc<Mutex<Jitv2>>` doesn't exist until after this
    /// `CompileQueue`, which lives inside it, has already been constructed).
    pub fn set_owner(&self, jitv2: Weak<Mutex<Jitv2>>) {
        *self.jitv2.lock() = Some(jitv2);
    }

    /// Push a compile request. Non-blocking: per §6.4, a full queue drops the
    /// request rather than backing up the exec thread — the page that wanted
    /// it stays hot and will re-trigger the request on a later arrival.
    /// Returns `false` if the request was dropped. `stats` is `Jitv2::stats`
    /// — under `developer`, records this dispatch, whether it was dropped
    /// for a full queue, and the queue's occupancy at this exact moment
    /// (`JitStats`'s own doc comments on the three `compile_queue_*` fields)
    /// for `j2 status`'s FIFO-fullness section. Accepted unconditionally
    /// (not `#[cfg]`-gated itself) so this signature doesn't change across
    /// feature combinations; the instrumentation work inside is what's
    /// gated, to keep the extra `slots()`/atomic touches off this
    /// per-dispatch-gate-miss hot path outside a diagnostics build.
    ///
    /// Takes `&self`, not `&mut self`: `self.queue` is an `Arc<ArrayQueue>`
    /// whose own `push`/`len` only need `&self`, and every counter this
    /// touches is an atomic — nothing here actually needs exclusive access.
    /// This lets a caller send through a bare `&CompileQueue`/`queue_handle`
    /// obtained once at construction instead of relocking the whole `Jitv2`
    /// mutex on every dispatch just to reach this one `Arc`-backed field
    /// (see `MipsExecutor::jitv2_compile_queue`/`jitv2_stats`).
    pub fn send(&self, req: CompileRequest, stats: &JitStats) -> bool {
        push_compile_request(&self.queue, req, stats)
    }

    /// Discard every `CompileRequest` currently sitting in the shared queue
    /// without processing it. Every `CompileRequest::page` is a raw pointer
    /// into `Jitv2::pages` (`Vec<PhysicalCodePage>`) — a request enqueued
    /// before a `mega_flush` still holds a pointer into the just-cleared
    /// `Vec` after it, and `comp::handle_request` dereferencing it is a real
    /// use-after-free (confirmed live: `PhysicalCodePage::publish` segfault,
    /// `jitv2-compile` thread, immediately after a `flush_from_jit_thread`
    /// that never drained the queue first). Must run with no worker thread
    /// concurrently popping — true both when called from the flush leader
    /// itself right after `cpu.stop()` has fully joined the CPU and every
    /// other worker has parked (nothing pops anymore), and when called by
    /// `flush_from_cpu_thread`'s caller right after `stop()` has joined
    /// every worker thread (provably not popping).
    fn drain_pending(queue: &crossbeam_queue::ArrayQueue<CompileRequest>) {
        while queue.pop().is_some() {}
    }

    /// Public entry point for [`Self::drain_pending`] when the caller only
    /// has `&CompileQueue`, not direct access to the shared queue — see
    /// `drain_pending`'s own doc comment for why this must run and when
    /// it's safe to call (no worker concurrently popping).
    pub fn drain_pending_queue(&mut self) {
        Self::drain_pending(&self.queue);
    }

    /// Spawn the compile-thread pool with one freshly-reserved shared arena
    /// — today's normal entry point (`Machine::new`'s startup path). See
    /// `start_inner`'s own doc comment for the full contract.
    pub fn start(&mut self, bus: Arc<dyn BusDevice>, stats: Arc<JitStats>) {
        let state = Arc::new(crate::jitv2::paged_memory::PagedArenaState::default());
        #[cfg(not(feature = "j2wp"))]
        let reserve_size = crate::jitv2::codegen::Codegen::ARENA_RESERVE_SIZE;
        #[cfg(feature = "j2wp")]
        let reserve_size = ARENA_RESERVE_SIZE;
        let shared = crate::jitv2::paged_memory::PagedArenaMemoryProvider::new_shared(
            reserve_size, state.clone(),
        ).expect("CompileQueue::start: failed to reserve a fresh jitv2 arena");
        self.start_inner(bus, stats, shared, state);
    }

    /// Spawn the compile-thread pool with every worker's `Codegen` built on
    /// top of an *already-reserved* shared arena instead of a fresh one —
    /// the `j2 inline off` mode-switch path (reusing whatever arena inline
    /// mode's own `Codegen` was already using, rather than abandoning it —
    /// see that handler's own comment).
    pub fn start_with_shared_arena(
        &mut self,
        bus: Arc<dyn BusDevice>,
        stats: Arc<JitStats>,
        shared: Arc<parking_lot::Mutex<crate::jitv2::paged_memory::SharedArena>>,
        state: Arc<crate::jitv2::paged_memory::PagedArenaState>,
    ) {
        self.start_inner(bus, stats, shared, state);
    }

    /// No-op if already running. `bus` is the executor's `sysad` — every
    /// worker reads the page snapshot off it at compile time (§6.5 step 2).
    /// `stats` is `Jitv2::stats`, cloned in by the caller (mirrors
    /// `cpu`/`jitv2` below — see `JitStats`'s own doc comment for why this
    /// is an `Arc` handed in rather than reached via a lock on every
    /// `handle_request` call). Threaded through unconditionally (cheap: one
    /// more `Arc` clone per worker) rather than duplicating this whole
    /// function behind `#[cfg(feature = "developer")]` — the actual counter
    /// increments are what's gated, in `comp::handle_request`.
    ///
    /// Spawns exactly `self.thread_count` `jitv2-compile-N` OS threads, each
    /// with its own `Codegen` built via `Codegen::new_with_shared_arena` on
    /// top of the ONE `shared`/`state` pair passed in — one arena
    /// reservation for the whole pool, N independent `JITModule`s sharing
    /// it (see `paged_memory`'s module doc comment for why that's safe:
    /// `allocate()` is mutex-serialized inside `SharedArena`, and sealing is
    /// worker-agnostic by construction). Every worker pops from the same
    /// `self.queue` (`crossbeam_queue::ArrayQueue`) — no per-worker ring, no
    /// dispatch routing needed on the send side.
    fn start_inner(
        &mut self,
        bus: Arc<dyn BusDevice>,
        stats: Arc<JitStats>,
        shared: Arc<parking_lot::Mutex<crate::jitv2::paged_memory::SharedArena>>,
        state: Arc<crate::jitv2::paged_memory::PagedArenaState>,
    ) {
        if !self.threads.is_empty() {
            return;
        }
        self.running.store(true, Ordering::SeqCst);
        self.function_counts = (0..self.thread_count).map(|_| Arc::new(AtomicU32::new(0))).collect();
        for i in 0..self.thread_count {
            let codegen = crate::jitv2::codegen::Codegen::new_with_shared_arena(shared.clone(), state.clone());
            let queue = self.queue.clone();
            let running = self.running.clone();
            let bus = bus.clone();
            let cpu = self.cpu.lock().clone();
            let jitv2 = self.jitv2.lock().clone();
            let function_count = self.function_counts[i].clone();
            let stats = stats.clone();
            let barrier = self.barrier.clone();
            let quiesce_in_progress = self.quiesce_in_progress.clone();
            let thread_count = self.thread_count;
            self.threads.push(
                std::thread::Builder::new()
                    .name(format!("jitv2-compile-{i}"))
                    .spawn(move || Self::worker_loop(queue, running, bus, codegen, cpu, jitv2, function_count, stats, barrier, quiesce_in_progress, thread_count))
                    .expect("jitv2-compile spawn"),
            );
        }
    }

    /// Stop the worker thread(s) and join them, reclaiming the consumer(s)
    /// and every worker's own `Codegen` so a later `start()` can resume
    /// (with fresh `Codegen`s of its own — see `start()`'s own doc comment
    /// on why these returned ones are no longer handed back in). Empty
    /// `Vec` if not running; otherwise one `Codegen` per worker thread, in
    /// no particular order — every caller (`flush_from_cpu_thread`, the `j2
    /// inline` handler) already treats this as "however many were running,"
    /// not "exactly one."
    ///
    /// Drains every request still sitting in the shared queue (the CPU
    /// thread can keep enqueueing right up until the moment `running` flips
    /// false, so this is never guaranteed empty just because every worker
    /// joined) and zeroes the status-bar queue-fill gauge — without this, a
    /// request dropped here silently vanishes (harmless on its own, same as
    /// any other §6.4 drop-on-full/drop-on-flush case; the offset just
    /// re-triggers on its next arrival) but the status bar's occupancy
    /// reading was confirmed live to otherwise stay stuck at whatever it
    /// last was before the stop, forever, since nothing calls
    /// `set_queue_fill` again once no worker is running.
    pub fn stop(&mut self) -> Vec<crate::jitv2::codegen::Codegen> {
        self.running.store(false, Ordering::SeqCst);
        let result: Vec<_> = self.threads.drain(..).enumerate().filter_map(|(i, h)| {
            let r = h.join();
            r.ok()
        }).collect();
        self.drain_pending_queue();
        crate::jit_feedback::JIT_FEEDBACK.set_queue_fill(0, COMPILE_QUEUE_CAPACITY);
        result
    }

    /// Block this (non-leader) worker until the current quiesce cycle's
    /// leader finishes and releases the barrier. Registers itself as parked
    /// (bumping `parked_count`, which is what the leader's own wait
    /// condition — `parked_count == thread_count - 1` — counts against),
    /// then waits on `generation` advancing past whatever value was current
    /// at entry (guards spurious/lost wakeups — see `BarrierState::generation`'s
    /// own doc comment). Returns `Some((arena, state))` if the cycle that
    /// just ended was a full flush (`BarrierState::was_flush`) — the caller
    /// must rebuild its own `Codegen` from it
    /// (`Codegen::new_with_shared_arena`) before resuming its loop — or
    /// `None` if it was a seal-only cycle, where the caller resumes in
    /// place with nothing to rebuild.
    fn park_at_barrier(
        barrier: &Arc<(parking_lot::Mutex<BarrierState>, parking_lot::Condvar)>,
    ) -> Option<(Arc<parking_lot::Mutex<crate::jitv2::paged_memory::SharedArena>>, Arc<crate::jitv2::paged_memory::PagedArenaState>)> {
        let (mutex, cv) = &**barrier;
        let mut state = mutex.lock();
        let my_generation = state.generation;
        state.parked_count += 1;
        cv.notify_all(); // wake the leader, which is waiting for parked_count to reach thread_count - 1
        while state.generation == my_generation {
            cv.wait(&mut state);
        }
        if !state.was_flush {
            return None;
        }
        Some((
            state.current_arena.clone().expect("park_at_barrier: a flush leader must publish an arena before bumping generation"),
            state.current_state.clone().expect("park_at_barrier: a flush leader must publish a state before bumping generation"),
        ))
    }

    /// Worker body: pop requests until stopped and hand each to
    /// `comp::handle_request` (reachability walk + codegen + publish, §6.5;
    /// dump-to-disk corpus collection lives behind the `jitv2_corpus_dump`
    /// feature in the same function, see `jitv2/comp.rs`). Owns the
    /// `Analyzer` scratch state for the thread's whole lifetime (meant to be
    /// reused across jobs, not rebuilt per request); `codegen` is the shared
    /// one moved in by `start`. Backs off briefly when the queue is empty
    /// rather than busy-spinning — compile requests are bursty (arrival
    /// threshold crossings), not latency-critical.
    ///
    /// Every compile routes through `comp::handle_request_deferred`, never
    /// `comp::handle_request` — `handle_request`'s forced-seal contract
    /// (mprotect a whole host page to RX immediately, unconditionally) is
    /// only safe with a single caller, since it assumes nothing else could
    /// possibly still be bump-allocating or `copy_nonoverlapping`-ing into
    /// that same page; under a real N>1 pool, another worker routinely can
    /// be — confirmed live as a SIGSEGV inside `CompiledBlob::new`'s own
    /// write the first time this pool ran with `thread_count > 1` outside
    /// tests. `handle_request` stays in use elsewhere (inline compile,
    /// `mips_exec.rs`'s `jitv2_inline_compile` — single-threaded on the CPU
    /// thread, where the forced-seal assumption still holds), just never
    /// here. `handle_request_deferred` still finalizes every successful
    /// compile immediately (`Codegen::finalize_batch_nonforced`,
    /// called with just that one `FuncId` — cheap, and this is what patches
    /// relocations; see that function's own doc comment for why deferring
    /// the finalize call itself, not just the seal, used to be a real bug
    /// under multi-worker concurrency). What's deferred is only the *seal*
    /// (mprotect) step: `finalize_batch_nonforced` pushes the just-finalized
    /// range onto the shared arena's seal queue and opportunistically seals
    /// whatever's now a contiguous prefix from the watermark — the common
    /// case publishes right away, same call. If this range is gap-blocked
    /// behind an earlier, not-yet-finalized range from another worker, it
    /// comes back unsealed and `handle_request_deferred` pushes exactly one
    /// `PendingPublish` onto `pending` for a later retry — see
    /// `paged_memory`'s module doc comment for why sealing (not finalizing)
    /// is what actually lets functions pack tightly instead of each getting
    /// its own page. `pending` is drained two ways: (1) on every queue-empty
    /// poll (the `None` arm), a *non-forced* `comp::publish_ready_nonforced`
    /// attempt runs first (safe/cheap — publishes anything already sealable
    /// without forcing a page closed early), and only once the queue has
    /// been continuously empty for a real ~100ms idle threshold does
    /// `comp::force_publish_pending` force-seal whatever's still stuck — so
    /// a lone straggler that never grows into a full page still gets
    /// published eventually, at the cost of at most one prematurely-closed
    /// partial page per idle burst (see `Codegen::force_seal_pending`'s own
    /// doc comment). `pending` is normally at most one or two entries per
    /// worker (genuinely gap-blocked ranges), not an accumulated batch.
    /// `pending` is discarded (never flushed) immediately before `do_flush` runs:
    /// `do_flush` resets `codegen` (freeing the whole arena — every
    /// not-yet-finalized `FuncId` in `pending` would dangle) and clears the
    /// page pool (every `PendingPublish::page` would dangle too), so there's
    /// nothing safe left to publish by that point — same reasoning as
    /// `drain_pending` already applies to in-flight `CompileRequest`s for
    /// the identical reset.
    ///
    /// After every successful compile, checks `codegen.function_count()`
    /// against the flush threshold (`Codegen::function_count`'s doc comment)
    /// — if crossed, the first worker to detect it becomes leader
    /// (`quiesce_in_progress.compare_exchange`) and runs `run_leader_flush`:
    /// stops the CPU (via `cpu`, upgraded from `Weak`; skipped entirely if
    /// unset or already gone — nothing to flush for if the machine has no
    /// CPU to pause), waits for every other worker to park
    /// (`park_at_barrier`, called by any worker whose own loop-top check —
    /// or its own trigger detection — sees `quiesce_in_progress` already
    /// set), flushes the page pool (`Jitv2::flush_from_jit_thread` — NOT
    /// `Jitv2::flush`, which would try to stop this same compile queue from
    /// within itself), builds one fresh shared arena, rebuilds every
    /// worker's own `Codegen` on top of it
    /// (`Codegen::new_with_shared_arena`/`reset_with_shared_arena`),
    /// publishes it into the barrier, and restarts the CPU. At today's
    /// `thread_count == 1` there are never any followers to wait for — the
    /// leader's own barrier wait is immediately satisfied — so this is
    /// observably identical to the pre-pool `do_flush` sequence; the
    /// election/park machinery itself is proven separately, with real
    /// concurrent threads, in `tests::barrier_parks_followers_until_leader_resumes_them`.
    /// Returns `codegen` on exit so `stop()` can collect it — the shared
    /// `queue` itself is never handed back (it's an `Arc`, cloned into every
    /// worker up front and simply left running/idle after they all exit).
    fn worker_loop(
        queue: Arc<crossbeam_queue::ArrayQueue<CompileRequest>>,
        running: Arc<AtomicBool>,
        bus: Arc<dyn BusDevice>,
        mut codegen: crate::jitv2::codegen::Codegen,
        cpu: Option<Weak<dyn Device>>,
        jitv2: Option<Weak<Mutex<Jitv2>>>,
        function_count: Arc<AtomicU32>,
        stats: Arc<JitStats>,
        barrier: Arc<(parking_lot::Mutex<BarrierState>, parking_lot::Condvar)>,
        quiesce_in_progress: Arc<AtomicBool>,
        thread_count: usize,
    ) -> crate::jitv2::codegen::Codegen {
        let mut analyzer = crate::jitv2::analyzer::Analyzer::new();
        let mut pending: crate::jitv2::comp::PendingCount = 0;
        // Wall-clock start of the current unbroken stretch of "queue empty
        // AND pending still non-empty after a non-forced publish attempt" —
        // see the `Err(_)` arm below (queue-drain fallback) for the
        // idle-timeout scheme this drives. `None` whenever there's nothing
        // idle to time (queue non-empty, or pending fully drained).
        let mut idle_since: Option<std::time::Instant> = None;
        // Wait for every other worker to park, same as a plain `cv.wait`
        // loop on `parked_count`, but periodically re-checks `running` so a
        // `CompileQueue::stop()` racing against an in-flight leader
        // sequence can never wedge forever: `stop()` sets `running = false`
        // then joins every thread in turn; a *follower* thread's own
        // top-of-loop check can see `running == false` and exit
        // `worker_loop` entirely (not park) before the leader's wait
        // condition is ever satisfied — confirmed live as a genuine hang
        // (leader stuck waiting for parked_count to reach a follower that
        // had already exited, `stop()`'s `.join()` on the leader thread
        // never returning). Returns `true` once every other worker has
        // actually parked (safe to proceed with the real flush/seal work),
        // or `false` if `running` went false first (the caller must abandon
        // the quiesce cleanly and return, not touch shared state further).
        let wait_for_followers_or_abandon = |barrier: &Arc<(parking_lot::Mutex<BarrierState>, parking_lot::Condvar)>| -> bool {
            let (mutex, cv) = &**barrier;
            let mut state = mutex.lock();
            while state.parked_count < thread_count.saturating_sub(1) {
                if !running.load(Ordering::Relaxed) {
                    return false;
                }
                // Bounded wait, not an unbounded `cv.wait` — this is what
                // lets the `running` re-check above actually happen instead
                // of blocking indefinitely on a notify that a stopping pool
                // will never send (the follower that's supposed to park has
                // already exited instead). The timeout only needs to be
                // short enough that `stop()` doesn't feel hung — it is not
                // on any latency-sensitive path otherwise.
                cv.wait_for(&mut state, BARRIER_FOLLOWER_POLL_INTERVAL);
            }
            true
        };
        // Release the barrier without publishing anything (no arena, no
        // seal result) — the shared cleanup for every way a leader sequence
        // can end without actually doing the flush/seal work: `cpu`/`jitv2`
        // failed to upgrade, or `running` went false while waiting for
        // followers (`wait_for_followers_or_abandon` returned `false`).
        // Bumping `generation` here still matters even with nothing to
        // publish — any follower already parked (or about to park) must
        // still wake up and see `was_flush: false`, not hang forever behind
        // a leader that's giving up.
        let abandon_quiesce = |pending: &mut crate::jitv2::comp::PendingCount| {
            *pending = 0;
            let (mutex, cv) = &*barrier;
            let mut state = mutex.lock();
            state.parked_count = 0;
            state.was_flush = false;
            state.generation += 1;
            cv.notify_all();
            drop(state);
            quiesce_in_progress.store(false, Ordering::Release);
        };
        // Runs the full leader sequence for an arena flush — see
        // worker_loop's own doc comment for the step-by-step. Called only
        // after this thread has already won leadership via
        // `quiesce_in_progress.compare_exchange`.
        let run_leader_flush = |codegen: &mut crate::jitv2::codegen::Codegen,
                                 function_count: &Arc<AtomicU32>,
                                 pending: &mut crate::jitv2::comp::PendingCount| {
            let Some((cpu, jit)) = cpu.as_ref().and_then(Weak::upgrade).zip(jitv2.as_ref().and_then(Weak::upgrade)) else {
                // `set_cpu`/`set_owner` were never called, or their target
                // has since been dropped — this thread already won
                // leadership (quiesce_in_progress is true) and every other
                // worker will either park here too or block waiting for
                // this generation to end, so failing to release the barrier
                // would deadlock the whole pool forever, not just skip one
                // flush. A running pool with no cpu/jitv2 wired is a real
                // bug (Machine::new always wires both before load can ever
                // reach a flush threshold) — assert in debug/test builds so
                // it's caught immediately, but still recover in release so
                // production never wedges silently.
                debug_assert!(false, "jitv2 compile pool flush triggered with no cpu/jitv2 wired (set_cpu/set_owner never called, or their target was dropped)");
                abandon_quiesce(pending);
                return;
            };
            {
                // Wait for every OTHER worker to park before touching
                // anything shared — at thread_count == 1 there are none, so
                // this is immediately satisfied (parked_count < 0 never
                // holds for the unsigned saturating_sub(1) below). Bails
                // out cleanly (no flush performed) if `running` goes false
                // first — see `wait_for_followers_or_abandon`'s own doc
                // comment for the `CompileQueue::stop()` race this guards.
                if !wait_for_followers_or_abandon(&barrier) {
                    abandon_quiesce(pending);
                    return;
                }
                // Discard, not flush: every PendingPublish::page and every
                // not-yet-finalized FuncId this batch is holding is about to
                // dangle (the page-pool clear below, plus every worker's
                // Codegen getting rebuilt on a fresh arena) — see
                // worker_loop's own doc comment for the full reasoning,
                // same as drain_pending's existing treatment of in-flight
                // CompileRequests below.
                *pending = 0;
                // cpu.stop() must run with Jitv2's lock NOT held — it locks
                // the executor and, through it, this same Mutex<Jitv2>
                // again (own page-pool stats print), which would
                // self-deadlock this thread on its own non-reentrant lock
                // if taken here first. Lock only for the actual flush
                // (flush_from_jit_thread's own doc comment), release before
                // cpu.start().
                cpu.stop();
                // Every request still sitting in the shared queue right now
                // points into the pool flush_from_jit_thread is about to
                // clear — drain them before the flush, not after, or the
                // next pop() dereferences a dangling PhysicalCodePage
                // pointer (see drain_pending's own doc comment for the
                // crash this was confirmed to cause). Safe to drain once
                // here even though the queue is shared with every other
                // (already-parked, not popping) worker.
                Self::drain_pending(&queue);
                unsafe { jit.lock().flush_from_jit_thread(codegen.function_count()); }
                // Build ONE fresh shared arena and rebuild this (leader's
                // own) Codegen on top of it — every other, parked worker
                // rebuilds its own once it wakes (park_at_barrier's return
                // value), from the same arena published below.
                let fresh_state = std::sync::Arc::new(crate::jitv2::paged_memory::PagedArenaState::default());
                #[cfg(not(feature = "j2wp"))]
                let reserve_size = crate::jitv2::codegen::Codegen::ARENA_RESERVE_SIZE;
                #[cfg(feature = "j2wp")]
                let reserve_size = ARENA_RESERVE_SIZE;
                let fresh_arena = crate::jitv2::paged_memory::PagedArenaMemoryProvider::new_shared(
                    reserve_size, fresh_state.clone(),
                ).expect("run_leader_flush: failed to reserve a fresh jitv2 arena");
                unsafe { codegen.reset_with_shared_arena(fresh_arena.clone(), fresh_state.clone()); }
                {
                    let (mutex, cv) = &*barrier;
                    let mut state = mutex.lock();
                    state.current_arena = Some(fresh_arena);
                    state.current_state = Some(fresh_state);
                    state.parked_count = 0;
                    state.was_flush = true;
                    state.generation += 1;
                    cv.notify_all();
                }
                quiesce_in_progress.store(false, Ordering::Release);
                cpu.start();
                // flush_from_jit_thread reset codegen back to 0.
                function_count.store(0, Ordering::Relaxed);
            }
        };
        // Runs the full leader sequence for a forced-seal-only quiesce — no
        // arena replacement, no CPU stop, just "wait for everyone else to
        // park (guaranteeing none of them is mid-compile — no lower-`start`
        // range can still arrive), force-seal whatever's queued under that
        // guarantee, then resume everyone in place." Called only after this
        // thread has already won leadership via the SAME
        // `quiesce_in_progress.compare_exchange` a flush leader would use —
        // see `BarrierState`'s own doc comment for why this and
        // `run_leader_flush` share one gate/barrier instead of two.
        let run_leader_seal = |codegen: &mut crate::jitv2::codegen::Codegen,
                                pending: &mut crate::jitv2::comp::PendingCount| {
            // Bails out cleanly (no seal performed) if `running` goes false
            // first — see `wait_for_followers_or_abandon`'s own doc comment
            // for the `CompileQueue::stop()` race this guards.
            if !wait_for_followers_or_abandon(&barrier) {
                abandon_quiesce(pending);
                return;
            }
            crate::jitv2::comp::force_publish_pending(codegen, pending);
            {
                let (mutex, cv) = &*barrier;
                let mut state = mutex.lock();
                state.parked_count = 0;
                state.was_flush = false;
                state.generation += 1;
                cv.notify_all();
            }
            quiesce_in_progress.store(false, Ordering::Release);
        };
        // Leader election: the first worker whose own compile triggers
        // EITHER a flush condition or a forced-seal condition wins this one
        // shared compare_exchange and runs the corresponding sequence;
        // every other worker that also observes a trigger (its own, or via
        // this same check at the top of its next loop iteration) instead
        // drains/discards its own state and parks at the one shared
        // barrier, regardless of which kind of leader it's waiting on — see
        // `BarrierState`'s own doc comment for why one gate covers both. At
        // thread_count == 1 there is only ever one worker, so this always
        // wins trivially — real contention is exercised separately
        // (`tests::barrier_parks_followers_until_leader_resumes_them`).
        let try_flush = |codegen: &mut crate::jitv2::codegen::Codegen,
                          function_count: &Arc<AtomicU32>,
                          pending: &mut crate::jitv2::comp::PendingCount| {
            if quiesce_in_progress.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                run_leader_flush(codegen, function_count, pending);
            } else {
                *pending = 0;
                if let Some((arena, state)) = Self::park_at_barrier(&barrier) {
                    unsafe { codegen.reset_with_shared_arena(arena, state); }
                } else {
                }
                function_count.store(codegen.function_count(), Ordering::Relaxed);
            }
        };
        let try_force_seal = |codegen: &mut crate::jitv2::codegen::Codegen,
                               pending: &mut crate::jitv2::comp::PendingCount| {
            if quiesce_in_progress.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                run_leader_seal(codegen, pending);
            } else {
                *pending = 0;
                if let Some((arena, state)) = Self::park_at_barrier(&barrier) {
                    unsafe { codegen.reset_with_shared_arena(arena, state); }
                    function_count.store(codegen.function_count(), Ordering::Relaxed);
                } else {
                }
            }
        };
        while running.load(Ordering::Relaxed) {
            // A worker idling between compiles (about to pop its next
            // request) that sees another worker already leading a quiesce
            // cycle (flush OR seal-only — one shared gate, see
            // `BarrierState`'s own doc comment) must park immediately
            // rather than pop/compile anything new — its own next compile
            // would target a page pool/arena that's about to be cleared out
            // from under it, or would race the leader's forced seal. At
            // thread_count == 1 this can only ever be this same thread
            // mid-leader-sequence, which never reaches back here until it's
            // done (the flag is cleared before the leader function
            // returns), so it's a no-op today; real contention is exercised
            // separately.
            if quiesce_in_progress.load(Ordering::Acquire) {
                pending = 0;
                if let Some((arena, state)) = Self::park_at_barrier(&barrier) {
                    unsafe { codegen.reset_with_shared_arena(arena, state); }
                    function_count.store(codegen.function_count(), Ordering::Relaxed);
                } else {
                }
                continue;
            }
            match queue.pop() {
                Some(req) => {
                    // Always the deferred/non-forced path — never
                    // handle_request's forced-seal one. Forced sealing
                    // mprotects a WHOLE host page to RX immediately,
                    // unconditionally safe only when nothing else can be
                    // mid-allocate/mid-copy on that same page — true for
                    // inline compile (single-threaded, its only other
                    // caller) but not here: a second worker can easily
                    // still be bump-allocating into (or, worse,
                    // mid-`copy_nonoverlapping` writing compiled bytes
                    // into) the very page this worker's own forced seal
                    // would just mprotect out from under it. Confirmed
                    // live as a real SIGSEGV (copy_nonoverlapping crashing
                    // inside CompiledBlob::new) the first time this pool
                    // actually ran with thread_count > 1 outside tests —
                    // `j2 batch off` (the `developer`-build default) used
                    // to route here into handle_request instead, which is
                    // what triggered it.
                    let ran_out_of_memory = {
                        #[cfg(feature = "developer")]
                        { crate::jitv2::comp::handle_request_deferred(&req, &bus, &mut analyzer, &mut codegen, &mut pending, &stats) }
                        #[cfg(not(feature = "developer"))]
                        { crate::jitv2::comp::handle_request_deferred(&req, &bus, &mut analyzer, &mut codegen, &mut pending) }
                    };
                    // Keep CompileQueue::function_count's mirror in sync —
                    // see that field's doc comment for why it exists
                    // (`j2 stats` can't read the real codegen.function_count()
                    // directly while this thread owns `codegen` by value).
                    function_count.store(codegen.function_count(), Ordering::Relaxed);
                    // Status-bar feedback (disp.rs's StatusBar), updated
                    // from this side (the compile thread, once per actual
                    // compile) rather than from CompileQueue::send on the
                    // CPU/exec thread — send() runs on the hot dispatch-gate
                    // path on every compile trigger, and an extra atomic
                    // store there would tax exactly the thread this whole
                    // async-queue design exists to keep off the compile
                    // work. Reusing packing_stats() here means this doesn't
                    // even add a second call — just reads the tuple this
                    // iteration was already computing for the threshold
                    // check below.
                    let reserved_bytes = codegen.packing_stats().1;
                    crate::jit_feedback::JIT_FEEDBACK.set_arena_fill(reserved_bytes, CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES);
                    crate::jit_feedback::JIT_FEEDBACK.set_queue_fill(queue.len(), COMPILE_QUEUE_CAPACITY);
                    if ran_out_of_memory {
                        // The compile that just ran couldn't get memory —
                        // flush immediately, regardless of the byte
                        // threshold below (the arena is provably already
                        // full, so there's nothing to gain from checking).
                        // The request that just failed is gone
                        // (handle_request already returned), so it'll only
                        // be retried on this offset's next real arrival —
                        // same as any other "retry later" outcome.
                        try_flush(&mut codegen, &function_count, &mut pending);
                    } else if reserved_bytes > CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES {
                        // Real bytes reserved in the arena (not a
                        // function-count proxy — see
                        // CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES's own doc
                        // comment) crossing the threshold means this
                        // thread's own Cranelift memory arena has grown
                        // unboundedly (nothing else ever frees it — see
                        // Codegen::reset's doc comment) and needs flushing
                        // pre-emptively, before it actually runs out.
                        try_flush(&mut codegen, &function_count, &mut pending);
                    } else if pending >= PENDING_FORCE_SEAL_THRESHOLD {
                        // A continuously busy worker (queue never goes
                        // empty for it) never reaches the `None` arm's
                        // idle-timeout sweep below — every compile finalizes
                        // immediately (see handle_request_deferred's own doc
                        // comment) but a non-forced finalize only seals a
                        // page the bump cursor has already moved past, so a
                        // steady stream of small compiles can accumulate in
                        // `pending` indefinitely with nothing to force a
                        // seal. Once `pending` grows past a small bound,
                        // force-seal it now rather than wait for a queue-
                        // empty tick that may never come — bounds `pending`
                        // under sustained load at the cost of at most one
                        // prematurely-closed partial page per threshold
                        // crossing, same trade-off as the idle-timeout sweep.
                        // Goes through try_force_seal (the shared quiesce
                        // barrier), not a direct force_publish_pending call
                        // — see BarrierState's own doc comment for why a
                        // forced seal is only safe once every other worker
                        // is provably parked (not mid-compile).
                        #[cfg(feature = "developer")]
                        stats.record_batch_flush(pending, crate::jitv2::BatchFlushReason::PendingThreshold);
                        try_force_seal(&mut codegen, &mut pending);
                        function_count.store(codegen.function_count(), Ordering::Relaxed);
                    }
                }
                None => {
                    // Queue-drain fallback: even under batching, a lone
                    // compile followed by a quiet period must not sit
                    // unpublished indefinitely waiting for a page to fill.
                    //
                    // Two-phase, matching the non-forced sealing design
                    // (`paged_memory`'s module doc comment,
                    // `Codegen::finalize_batch_nonforced`/`force_seal_pending`):
                    // on every empty poll, try a non-forced publish attempt
                    // first (`publish_ready_nonforced` — cheap, harmless,
                    // and at today's N=1 worker count it's what actually
                    // does the real work here, converging with the page-cross
                    // trigger above). Only once the queue has been
                    // continuously empty for the real idle threshold
                    // (~100ms, tracked in wall-clock time across repeated
                    // empty polls, not "first empty poll") does this
                    // force-seal whatever's left — so a lone straggler that
                    // never grows into a full page still gets published
                    // eventually, but a genuinely busy queue that's just
                    // between two back-to-back bursts doesn't pay for a
                    // premature forced seal on every backoff tick
                    // (`WORKER_IDLE_POLL_BACKOFF`).
                    // At N=1 this is behaviorally a no-op relative to
                    // today (nothing here can ever actually be blocked on
                    // another worker), but it's real, reachable machinery —
                    // this is where the same idle-sweep this worker will
                    // need once a real multi-worker pool shares one arena
                    // gets exercised and proven first.
                    if pending > 0 {
                        crate::jitv2::comp::publish_ready_nonforced(&mut codegen, &mut pending);
                        if pending > 0 {
                            if idle_since.is_none() {
                                idle_since = Some(std::time::Instant::now());
                            }
                            if idle_since.is_some_and(|t| t.elapsed() >= IDLE_FORCE_SEAL_THRESHOLD) {
                                #[cfg(feature = "developer")]
                                stats.record_batch_flush(pending, crate::jitv2::BatchFlushReason::QueueDrain);
                                // Same shared quiesce barrier as the pending-
                                // threshold trigger above — see
                                // BarrierState's own doc comment.
                                try_force_seal(&mut codegen, &mut pending);
                                idle_since = None;
                            }
                        } else {
                            idle_since = None;
                        }
                        function_count.store(codegen.function_count(), Ordering::Relaxed);
                    } else {
                        idle_since = None;
                    }
                    std::thread::sleep(WORKER_IDLE_POLL_BACKOFF);
                }
            }
        }
        codegen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64};

    /// Pins `comp::MAX_INSTRS_PER_COMPILE` to an explicit value for the
    /// lifetime of the guard, restoring the prior value on drop (including
    /// on panic/unwind) — tests must never rely on whatever the process-wide
    /// default happens to be, since it's a single global shared with every
    /// other test in this binary (potentially running concurrently) and with
    /// production code's own default.
    struct MaxInstrsGuard(usize);
    impl MaxInstrsGuard {
        fn set(n: usize) -> Self {
            let prev = crate::jitv2::comp::max_instrs_per_compile();
            crate::jitv2::comp::set_max_instrs_per_compile(n);
            Self(prev)
        }
    }
    impl Drop for MaxInstrsGuard {
        fn drop(&mut self) {
            crate::jitv2::comp::set_max_instrs_per_compile(self.0);
        }
    }

    /// Proves `park_at_barrier`'s park/wake protocol directly, with real
    /// spawned threads simulating N=3 workers — independent of `worker_loop`,
    /// since a real single-worker (`thread_count == 1`) integration can
    /// never actually exercise a park (see `BarrierState`'s own doc
    /// comment). Two "followers" call `park_at_barrier`; the main thread
    /// plays "leader": waits for both to register as parked, publishes a
    /// fresh arena/state, bumps `generation`, and releases them — then
    /// asserts both followers woke with exactly that published arena/state,
    /// not a stale or missing one.
    #[test]
    fn barrier_parks_followers_until_leader_resumes_them() {
        let barrier = Arc::new((parking_lot::Mutex::new(BarrierState::new()), parking_lot::Condvar::new()));

        let followers: Vec<_> = (0..2)
            .map(|_| {
                let barrier = barrier.clone();
                std::thread::spawn(move || CompileQueue::park_at_barrier(&barrier))
            })
            .collect();

        // Wait for both followers to register as parked (mirrors what a
        // real leader's own wait loop does — see run_leader_flush, landed
        // in a later step).
        {
            let (mutex, cv) = &*barrier;
            let mut state = mutex.lock();
            while state.parked_count < 2 {
                cv.wait(&mut state);
            }
        }

        let fresh_state = Arc::new(crate::jitv2::paged_memory::PagedArenaState::default());
        let fresh_arena = crate::jitv2::paged_memory::PagedArenaMemoryProvider::new_shared(1 << 20, fresh_state.clone()).unwrap();
        {
            let (mutex, cv) = &*barrier;
            let mut state = mutex.lock();
            state.current_arena = Some(fresh_arena.clone());
            state.current_state = Some(fresh_state.clone());
            state.parked_count = 0;
            state.was_flush = true;
            state.generation += 1;
            cv.notify_all();
        }

        for handle in followers {
            let (arena, state) = handle.join().unwrap()
                .expect("was_flush was set — a parked worker must wake with Some((arena, state))");
            assert!(Arc::ptr_eq(&arena, &fresh_arena), "a parked worker must wake with the exact arena the leader published");
            assert!(Arc::ptr_eq(&state, &fresh_state), "a parked worker must wake with the exact state the leader published");
        }
    }

    #[test]
    fn barrier_parks_followers_and_resumes_them_in_place_for_a_seal_only_cycle() {
        // Companion to the flush test above: a seal-only cycle
        // (`was_flush: false`, no arena/state published) must wake a parked
        // worker with `None` — nothing to rebuild — not panic on the
        // `.expect()` an arena would otherwise require.
        let barrier = Arc::new((parking_lot::Mutex::new(BarrierState::new()), parking_lot::Condvar::new()));

        let followers: Vec<_> = (0..2)
            .map(|_| {
                let barrier = barrier.clone();
                std::thread::spawn(move || CompileQueue::park_at_barrier(&barrier))
            })
            .collect();

        {
            let (mutex, cv) = &*barrier;
            let mut state = mutex.lock();
            while state.parked_count < 2 {
                cv.wait(&mut state);
            }
        }

        {
            let (mutex, cv) = &*barrier;
            let mut state = mutex.lock();
            state.parked_count = 0;
            state.was_flush = false;
            state.generation += 1;
            cv.notify_all();
        }

        for handle in followers {
            let result = handle.join().unwrap();
            assert!(result.is_none(), "a seal-only cycle must resume a parked worker with None (nothing to rebuild)");
        }
    }

    #[test]
    fn null_gen_ptr_reads_the_never_compilable_fallback_without_panicking() {
        // A device with no real gen tracking (gen_ptr returns null, e.g.
        // most MMIO) must never leave PhysicalCodePage::gen null itself —
        // current_gen() has to be unconditionally safe to call on any page,
        // claimed or not, real gen or not (see NEVER_COMPILABLE_GEN's doc
        // comment) — so this must read 0, not panic/deref-null.
        let page = PhysicalCodePage::new(0, std::ptr::null());
        assert_eq!(page.current_gen(), 0);
    }

    #[test]
    fn physical_code_page_size_is_within_expected_bounds() {
        // Guardrail, not a strict contract: entries is now inline
        // (JITV2_INITIAL_PAGE_CAPACITY's own doc comment — no more Box
        // hiding the real per-page cost), so an accidental field bloat here
        // directly multiplies by the whole pool capacity at startup. Fails
        // loudly if PhysicalCodePage ever grows past a sanity ceiling rather
        // than silently ballooning Jitv2::new's one-shot allocation.
        let page_size = std::mem::size_of::<PhysicalCodePage>();
        let entry_size = std::mem::size_of::<JitEntry>();
        println!("size_of::<PhysicalCodePage>() = {page_size} bytes ({} bytes/entry x {ENTRIES_PER_PAGE} entries + bitmaps)", entry_size);
        assert!(page_size < 128 * 1024, "PhysicalCodePage grew unexpectedly large: {page_size} bytes — check for accidental field bloat before raising this ceiling");
    }

    #[test]
    fn reads_through_gen_pointer() {
        let counter = AtomicU64::new(42);
        let page = PhysicalCodePage::new(7, &counter as *const AtomicU64);
        assert_eq!(page.current_gen(), 42);
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(page.current_gen(), 43);
    }

    #[test]
    fn entry_starts_unpublished_and_undenylisted() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        assert!(!page.is_entry_valid(4));
        assert!(!page.is_denylisted(4));
        assert!(page.entries[4].func.is_null());
    }

    /// `try_schedule` must be a genuine test-and-set: the first caller for a
    /// given offset wins (returns `true`); every subsequent caller for the
    /// same offset, before `clear_scheduled` runs, must lose (returns
    /// `false`) — this is what stops `exec_decoded`'s dispatch gate from
    /// sending a duplicate `CompileRequest` for the same offset every time a
    /// hot PC re-satisfies the gate's trigger conditions while the first
    /// request is still in flight (see `ENTRY_SCHEDULED`'s doc comment).
    #[test]
    fn try_schedule_is_test_and_set() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        assert!(page.try_schedule(4), "first caller for a fresh offset must win");
        assert!(!page.try_schedule(4), "second caller before clear_scheduled must lose");
        assert!(!page.try_schedule(4), "still losing on a third call");

        // A different offset is independent.
        assert!(page.try_schedule(5), "a different offset must not be blocked by offset 4's bit");
    }

    #[test]
    fn clear_scheduled_allows_a_fresh_try_schedule() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        assert!(page.try_schedule(4));
        assert!(!page.try_schedule(4));

        page.clear_scheduled(4);
        assert!(page.try_schedule(4), "after clear_scheduled, a fresh request for the same offset must be allowed again");
    }

    #[test]
    fn clear_scheduled_on_an_unset_offset_is_a_harmless_no_op() {
        // The jitv2_inline_compile path calls handle_request (and therefore
        // clear_scheduled, via its scope guard) without ever having called
        // try_schedule first — must not panic or affect other offsets.
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        page.clear_scheduled(4);
        assert!(page.try_schedule(4), "offset must still be schedulable after a no-op clear");
    }

    #[test]
    fn entry_valid_only_when_bit_set_and_gen_matches() {
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 100usize;

        // gen matches but bit not set: still invalid.
        page.entries[offset].gen.store(5, Ordering::Relaxed);
        assert!(!page.is_entry_valid(offset));

        // Publish: set the bit -> now valid.
        page.entries[offset].flags.fetch_or(ENTRY_VALID, Ordering::Release);
        assert!(page.is_entry_valid(offset));

        // Page mutates (gen bumps past what the entry was compiled against):
        // bit is still set, but the entry must read as stale.
        counter.store(6, Ordering::Relaxed);
        assert!(!page.is_entry_valid(offset), "stale entry (gen mismatch) must not be reported valid");
    }

    #[test]
    fn kill_clears_valid_bit_but_not_denylist() {
        // emit_fpu_entry_guard's FR-mismatch arm (jit_kill_entry) uses this
        // to un-publish a compiled-for-the-wrong-FR-mode entry: the JIT gate
        // must stop dispatching it, but a later visit is expected and
        // welcome to recompile the same offset fresh — unlike denylist,
        // which is permanent (§6.4 sticky rejection).
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 4usize;
        page.entries[offset].gen.store(0, Ordering::Relaxed);
        page.entries[offset].flags.fetch_or(ENTRY_VALID, Ordering::Release);
        assert!(page.is_entry_valid(offset));

        page.kill(offset);

        assert!(!page.is_published(offset), "kill must clear the valid bit");
        assert!(!page.is_entry_valid(offset));
        assert!(!page.is_denylisted(offset), "kill must not sticky-reject the offset — a fresh compile is expected to follow");

        // A later re-publish (simulating the next visit's fresh compile)
        // must work normally — kill leaves the offset fully recompilable.
        page.entries[offset].gen.store(0, Ordering::Relaxed);
        page.entries[offset].flags.fetch_or(ENTRY_VALID, Ordering::Release);
        assert!(page.is_entry_valid(offset), "offset must be re-publishable after kill");
    }

    #[test]
    fn publish_recompiling_a_stale_entry_updates_func_and_gen_together() {
        // Regression test for the recompile-ordering race: an entry whose
        // gen has drifted stale (page mutated, ENTRY_VALID still set) gets
        // recompiled in place by handle_request (comp.rs) — the ONE case
        // where publish() is called on an offset whose flag is already set.
        // ENTRY_VALID's Release/Acquire pairing gives no fresh synchronization
        // on that path (the flag's value doesn't change), so `gen` itself
        // must be the ordering point: func must be visible before gen ever
        // reads as matching current_gen(). This test can't force a true
        // concurrent interleaving, but it does verify the sequential
        // contract publish() must uphold for that ordering argument to hold
        // at all: func actually gets updated, and is_entry_valid only
        // reports true once gen matches (i.e. after publish() completes).
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 100usize;

        let old_fn = 0x1000usize as *const ();
        assert!(page.publish(offset, old_fn, 5, 1, 0));
        assert!(page.is_entry_valid(offset));
        assert_eq!(page.entries[offset].func, old_fn);

        // Page mutates: entry goes stale (bit stays 1, gen no longer matches).
        counter.store(6, Ordering::Relaxed);
        assert!(!page.is_entry_valid(offset), "must read stale immediately after the page mutates");

        // Recompile in place (comp.rs's handle_request path for a
        // stale-but-still-published entry) — gen_snap=6 was captured before
        // this second compile started, matching the page's now-current gen.
        let new_fn = 0x2000usize as *const ();
        assert!(page.publish(offset, new_fn, 6, 1, 0));
        assert!(page.is_entry_valid(offset), "recompiled entry must read valid once publish completes");
        assert_eq!(page.entries[offset].func, new_fn, "func must be the NEW function, not the stale one, once gen reads as current");
    }

    #[test]
    fn denylist_bit_is_independent_of_valid_bit() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 7usize;
        page.entries[offset].flags.fetch_or(ENTRY_DENYLISTED, Ordering::Relaxed);
        assert!(page.is_denylisted(offset));
        assert!(!page.is_entry_valid(offset), "denylisting must not itself mark an entry valid");
    }

    /// Minimal BusDevice whose gen_ptr always returns the same fixed counter,
    /// standing in for a real RAM/ROM device in these pool-only tests.
    struct FakeDevice(AtomicU64);
    impl BusDevice for FakeDevice {
        fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u32, _val: u8) -> u32 { crate::traits::BUS_ERR }
        fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u32, _val: u16) -> u32 { crate::traits::BUS_ERR }
        fn read32(&self, _addr: u32) -> BusRead32 { BusRead32::err() }
        fn write32(&self, _addr: u32, _val: u32) -> u32 { crate::traits::BUS_ERR }
        fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u32, _val: u64) -> u32 { crate::traits::BUS_ERR }
        fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 { &self.0 as *const AtomicU64 }
    }

    #[test]
    fn page_for_allocates_once_and_caches_by_pfn() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(4);
        let slot_a = jit.page_for(3, 3 * PAGE_SIZE, &dev).unwrap();
        let slot_b = jit.page_for(3, 3 * PAGE_SIZE, &dev).unwrap();
        assert_eq!(slot_a, slot_b, "second arrival at the same pfn must reuse the slot");
        let slot_c = jit.page_for(4, 4 * PAGE_SIZE, &dev).unwrap();
        assert_ne!(slot_a, slot_c);
    }

    #[test]
    fn page_for_returns_none_when_pool_exhausted() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(1);
        assert!(jit.page_for(0, 0, &dev).is_some());
        assert!(jit.page_for(1, PAGE_SIZE, &dev).is_none(), "pool of 1 must reject a second distinct pfn");
    }

    #[test]
    fn mega_flush_resets_pool_and_lookup() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(1);
        let first = jit.page_for(0, 0, &dev).unwrap();
        jit.mega_flush();
        let second = jit.page_for(0, 0, &dev).unwrap();
        assert_eq!(first, second, "slots renumber from 0 after a flush");
        assert!(jit.page_for(1, PAGE_SIZE, &dev).is_none(), "pool capacity still enforced after flush");
    }

    #[test]
    fn mega_flush_clears_per_entry_gen_so_a_reused_slot_starts_with_a_fresh_call_counter() {
        // Regression test for the pre-publish call-counter staleness bug a
        // reused (post-flush) slot could otherwise have: entries[i].gen
        // doubles as PhysicalCodePage::count_dispatch_and_check_threshold's
        // counter before an entry is ever published — mega_flush's in-place
        // reset (PhysicalCodePage::reset_to_unclaimed) must zero it just
        // like func/flags, or a slot claimed for a brand-new physical
        // page after a flush would inherit whatever count was left over from
        // its previous occupant.
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(1);
        let slot = jit.page_for(0, 0, &dev).unwrap() as usize;

        // Drive offset 4's pre-publish counter up close to a real threshold
        // without actually publishing it.
        for _ in 0..3 {
            assert!(!jit.pages[slot].count_dispatch_and_check_threshold(4, 10));
        }
        assert_eq!(jit.pages[slot].entries[4].gen.load(Ordering::Relaxed), 3);

        jit.mega_flush();
        let new_slot = jit.page_for(1, PAGE_SIZE, &dev).unwrap() as usize;
        assert_eq!(slot, new_slot, "single-capacity pool reuses the same physical slot");
        assert_eq!(jit.pages[new_slot].entries[4].gen.load(Ordering::Relaxed), 0,
            "a reused slot's pre-publish call counter must start fresh, not inherit the previous occupant's count");
    }

    #[test]
    #[cfg(feature = "developer")]
    fn code_size_by_instr_count_buckets_by_instr_count_and_tracks_min_max_sum() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(1);
        let slot = jit.page_for(0, 0, &dev).unwrap() as usize;
        let page = &jit.pages[slot];

        // Two entries at instr_count=3 (sizes 100, 300), one at instr_count=5 (size 50).
        assert!(page.publish(0, 0x1000 as *const (), 0, 3, 100));
        assert!(page.publish(1, 0x2000 as *const (), 0, 3, 300));
        assert!(page.publish(2, 0x3000 as *const (), 0, 5, 50));

        let hist = jit.code_size_by_instr_count();
        let bucket3 = hist[3].expect("instr_count=3 bucket must be present");
        assert_eq!(bucket3.count, 2);
        assert_eq!(bucket3.sum_bytes, 400);
        assert_eq!(bucket3.min_bytes, 100);
        assert_eq!(bucket3.max_bytes, 300);

        let bucket5 = hist[5].expect("instr_count=5 bucket must be present");
        assert_eq!(bucket5.count, 1);
        assert_eq!(bucket5.sum_bytes, 50);
        assert_eq!(bucket5.min_bytes, 50);
        assert_eq!(bucket5.max_bytes, 50);

        assert!(hist[4].is_none(), "instr_count=4 has no published entries");
    }

    #[test]
    #[cfg(feature = "developer")]
    fn record_reject_increments_both_failed_compiles_and_the_reason_bucket() {
        let stats = JitStats::default();
        stats.record_reject(RejectReason::EntryExcluded);
        stats.record_reject(RejectReason::EntryExcluded);
        stats.record_reject(RejectReason::CraneliftVerifierError);

        assert_eq!(stats.failed_compiles.load(Ordering::Relaxed), 3);
        assert_eq!(stats.reject_reasons[RejectReason::EntryExcluded.index()].load(Ordering::Relaxed), 2);
        assert_eq!(stats.reject_reasons[RejectReason::CraneliftVerifierError.index()].load(Ordering::Relaxed), 1);
        assert_eq!(stats.reject_reasons[RejectReason::AnalyzerCodegenDisagreement.index()].load(Ordering::Relaxed), 0);
    }

    #[test]
    fn compile_queue_start_stop_drains_without_hanging() {
        // FakeDevice's read32 always errors, so handle_request bails before
        // any filesystem I/O — no cwd/tempdir isolation needed here.
        let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.start(dev, std::sync::Arc::new(JitStats::default()));
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        let stats = JitStats::default();
        for i in 0..8u16 {
            assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: i, compiled_for_fr1: true }, &stats));
        }
        // stop() joins the worker; must return promptly even with requests in flight.
        q.stop();
    }

    /// Minimal `Device` (not `BusDevice`) stand-in for `CompileQueue::set_cpu` —
    /// tests that need `run_leader_flush`'s real flush path (not its
    /// no-cpu-wired recovery path) to actually run just need something a
    /// `Weak<dyn Device>` can upgrade to and call `stop()`/`start()` on; no
    /// test here cares whether cycles are actually stepped.
    struct NoopDevice;
    impl Device for NoopDevice {
        fn step(&self, _cycles: u64) {}
        fn stop(&self) {}
        fn start(&self) {}
        fn is_running(&self) -> bool { true }
        fn get_clock(&self) -> u64 { 0 }
    }

    /// Every word decodes as `ADDIU r1, r0, 1` — a real, compilable
    /// instruction, unlike `FakeDevice` (which always errors so
    /// `handle_request` bails before ever reaching codegen). Needed for a
    /// genuine end-to-end batching test: `handle_request_deferred` must
    /// actually produce a `FuncId` for there to be anything to batch.
    /// Two real `ADDIU r1, r0, 1` words (offsets 0 and 1) at the start of
    /// every physical page, followed by the JIT region-boundary sentinel at
    /// every other word — a genuine short, terminated region, matching how
    /// real code actually looks (a handful of instructions, then a branch/
    /// boundary — never "every word is a walkable instruction forever").
    /// Two instructions, not one: `comp::min_instrs_to_compile()` defaults
    /// to 2 outside `developer` builds (the build these tests actually run
    /// under), and a region below that floor is sticky-denylisted rather
    /// than compiled — confirmed live: an earlier one-instruction version of
    /// this device made every request get silently denylisted, 0/N ever
    /// published. Returning ADDIU unconditionally for every address (no
    /// sentinel at all) was tried before that and confirmed a different
    /// wrong shape: with `MAX_INSTRS_PER_COMPILE == usize::MAX` (a
    /// deliberate production setting, not a bug — see that constant's own
    /// doc comment in `comp.rs`), an all-ADDIU device makes every walk
    /// consume the entire 1024-word page, compiling a ~1000-instruction
    /// function instead of the couple of instructions each test actually
    /// intends — still correct, but ~300x slower per request than
    /// intended, confirmed live via direct instrumentation during the
    /// compile-pool work.
    struct AddiuDevice(AtomicU64);
    impl BusDevice for AddiuDevice {
        fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u32, _val: u8) -> u32 { crate::traits::BUS_ERR }
        fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u32, _val: u16) -> u32 { crate::traits::BUS_ERR }
        fn read32(&self, addr: u32) -> BusRead32 {
            let offset_word = (addr % PAGE_SIZE) / 4;
            if offset_word < 2 {
                BusRead32::ok((crate::mips_isa::OP_ADDIU << 26) | (1 << 16) | 1)
            } else {
                BusRead32::ok(crate::mips_isa::JIT_REGION_BOUNDARY_SENTINEL)
            }
        }
        fn write32(&self, _addr: u32, _val: u32) -> u32 { crate::traits::BUS_ERR }
        fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u32, _val: u64) -> u32 { crate::traits::BUS_ERR }
        fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 { &self.0 as *const AtomicU64 }
    }

    #[test]
    fn batching_eventually_publishes_via_queue_drain_fallback() {
        // End-to-end: a real worker thread with batching on, one compile
        // request, no second request to ever trigger a page-crossing flush.
        // Without the queue-drain fallback this entry would sit in `pending`
        // forever (nothing else would ever flush it) — this is the specific
        // regression the fallback trigger exists to prevent, exercised
        // through the real threaded path rather than just the synchronous
        // handle_request_deferred/flush_pending_batch unit tests.
        //
        // The polling loop below runs inside catch_unwind specifically so
        // q.stop() always executes afterward, even if the deadline assert!
        // fires: without that, a timeout panic would unwind straight out of
        // this function with the worker thread still running, holding a raw
        // pointer (CompileRequest::page) into `page` below — which is about
        // to be dropped as this function unwinds. That's a genuine
        // use-after-free from the still-live orphaned thread, not a benign
        // leaked-thread annoyance — confirmed live as the actual cause of an
        // intermittent SIGSEGV in unrelated, later-running tests during a
        // full-workspace parallel run (this test's deadline is only ever at
        // real risk of firing under heavy system load, which isolated/
        // single-test runs never reproduce — hence why this wasn't caught
        // immediately).
        let dev: Arc<dyn BusDevice> = Arc::new(AddiuDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.start(dev, std::sync::Arc::new(JitStats::default()));

        // A real (non-null) gen counter: page.publish() calls current_gen(),
        // which is always safe now (reads the shared NEVER_COMPILABLE_GEN
        // fallback for a null gen), but this test wants a real, independent
        // counter behind it since it's exercising a real compile+publish,
        // not just checking current_gen() doesn't crash.
        let gen_counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &gen_counter as *const AtomicU64);
        let stats = JitStats::default();
        // offset 0, matching every other AddiuDevice-based test in this
        // file: AddiuDevice's own read32 only decodes a real ADDIU for word
        // offsets < 2, sentinel otherwise (see its own doc comment) — offset
        // 4 (word index 4) would land straight on the sentinel with nothing
        // to compile at all.
        assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true }, &stats));

        // 30s, not the original 5s: this test passes in ~0.1s in isolation,
        // but the compile-worker thread genuinely needs real CPU time to get
        // scheduled — 5s proved too tight under a full-workspace parallel
        // test run (confirmed live: this deadline firing, not any actual
        // correctness bug, was tripping under heavy contention from every
        // other test's threads competing for the same cores).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            while !page.is_entry_valid(0) {
                assert!(std::time::Instant::now() < deadline, "entry never published — queue-drain fallback did not fire");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }));
        q.stop();
        result.unwrap();
    }

    #[test]
    fn batching_page_cross_trigger_publishes_without_waiting_for_queue_drain() {
        // Send enough compile requests, each targeting entry offset 0 of its
        // own distinct physical page, to force at least one page-crossing
        // flush (in the Cranelift arena's host-page-segment sense — not to
        // be confused with the distinct MIPS *physical* pages these requests
        // target, a coincidental naming overlap) before the queue ever
        // drains. Deliberately one request per PhysicalCodePage, all at
        // offset 0: a single walk_bounded from offset 0 with
        // MAX_INSTRS_PER_COMPILE=usize::MAX would otherwise treat many
        // requests against *different offsets of the same page* as one
        // giant sequential region (every word here decodes as the same
        // branch-free ADDIU), which isn't 300 independent compiles at
        // all — this was tried first and confirmed the wrong shape for this
        // test; distinct pages avoids that entirely and matches how real
        // page-cross-worthy traffic (many distinct hot pages arriving close
        // together) actually looks. Pinned explicitly rather than relying on
        // the ambient default (which no longer happens to be usize::MAX in
        // production) so this reasoning stays true regardless of future
        // default changes.
        let _max_instrs_guard = MaxInstrsGuard::set(usize::MAX);
        let dev: Arc<dyn BusDevice> = Arc::new(AddiuDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.start(dev, std::sync::Arc::new(JitStats::default()));

        const N: usize = 300;
        let gen_counters: Vec<AtomicU64> = (0..N).map(|_| AtomicU64::new(0)).collect();
        let mut pages: Vec<PhysicalCodePage> = gen_counters.iter()
            .enumerate()
            .map(|(i, counter)| PhysicalCodePage::new(i as Pfn, counter as *const AtomicU64))
            .collect();
        let stats = JitStats::default();
        for page in pages.iter_mut() {
            assert!(q.send(CompileRequest { page: page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true }, &stats));
        }

        // Poll for the *first* page specifically, well before the full
        // queue could possibly have drained (300 requests, 200µs backoff
        // only between empty-queue polls — draining that many keeps the
        // worker continuously busy, not backed off): if only the page-cross
        // trigger (not queue-drain) is what eventually publishes it, this
        // still succeeds this early. `stop()` itself does NOT guarantee a
        // full drain (it just flips `running` and joins whatever the loop's
        // current iteration is, same as any other "stop now" contract) — so
        // this must observe the page-cross trigger firing before stopping,
        // not rely on stop() to finish the queue first.
        //
        // catch_unwind for the same reason as the queue-drain-fallback test
        // above: without it, a deadline timeout here would unwind out of
        // this function with the worker thread still running against
        // `pages`/`gen_counters`, about to be dropped — a genuine
        // use-after-free from the orphaned thread, not a benign leaked
        // thread. This was the confirmed root cause of an intermittent
        // SIGSEGV surfacing in unrelated, later-running tests under
        // full-workspace parallel load.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            while !pages[0].is_entry_valid(0) {
                assert!(std::time::Instant::now() < deadline, "entry never published within the timeout");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }));
        q.stop();
        result.unwrap();
    }

    #[test]
    fn multi_worker_pool_compiles_and_publishes_every_request() {
        // Real end-to-end proof that N>1 worker threads actually compile
        // concurrently and correctly: set_thread_count(4), send far more
        // distinct-page requests than any single worker could plausibly
        // have handled alone within the deadline if the "pool" were secretly
        // still just one thread, and assert every single one gets published
        // — not just "doesn't deadlock" or "doesn't crash," but genuinely
        // produces correct output for all of them, exercising the shared
        // arena (SharedArena::allocate serialized across 4 concurrent
        // Codegens), the shared queue (crossbeam_queue::ArrayQueue popped
        // concurrently by 4 threads), and — if the arena ever fills —
        // real leader-election/barrier contention among genuinely live
        // worker threads, not the single-worker degenerate case every other
        // test in this file exercises.
        let dev: Arc<dyn BusDevice> = Arc::new(AddiuDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.set_thread_count(4);
        assert_eq!(q.thread_count(), 4);
        // Wire real cpu/jitv2 stand-ins before starting: 2000 requests
        // across 4 workers sharing one arena is expected to cross
        // CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES and exercise a real
        // leader-election flush — run_leader_flush needs both Weaks to
        // upgrade to do anything (see its own doc comment for what happens
        // if they can't).
        let cpu_stub: Arc<dyn Device> = Arc::new(NoopDevice);
        q.set_cpu(Arc::downgrade(&cpu_stub));
        let jit_owner = Arc::new(Mutex::new(Jitv2::new(JITV2_INITIAL_PAGE_CAPACITY)));
        q.set_owner(Arc::downgrade(&jit_owner));
        q.start(dev, std::sync::Arc::new(JitStats::default()));

        const N: usize = 2000;
        let gen_counters: Vec<AtomicU64> = (0..N).map(|_| AtomicU64::new(0)).collect();
        let mut pages: Vec<PhysicalCodePage> = gen_counters.iter()
            .enumerate()
            .map(|(i, counter)| PhysicalCodePage::new(i as Pfn, counter as *const AtomicU64))
            .collect();
        let stats = JitStats::default();
        for page in pages.iter_mut() {
            assert!(q.send(CompileRequest { page: page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true }, &stats));
        }

        // catch_unwind for the same reason as the other threaded tests in
        // this file: a deadline timeout must not unwind out of this
        // function with worker threads still live against `pages` — see
        // the page-cross test's own comment for the confirmed
        // use-after-free this guards.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loop {
                let published = pages.iter().filter(|p| p.is_entry_valid(0)).count();
                if published == N {
                    break;
                }
                assert!(std::time::Instant::now() < deadline, "only {published}/{N} entries published within the timeout — pool may be stuck");
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }));
        q.stop();
        result.unwrap();
    }

    #[test]
    fn compile_queue_send_before_start_still_delivered_after_start() {
        let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        let stats = JitStats::default();
        assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true }, &stats));
        q.start(dev, std::sync::Arc::new(JitStats::default()));
        q.stop();
    }

    #[test]
    fn compile_queue_send_drops_when_full() {
        let mut q = CompileQueue::new();
        // Don't start the worker: nothing drains, so capacity fills exactly.
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        let stats = JitStats::default();
        let mut accepted = 0;
        for i in 0..(COMPILE_QUEUE_CAPACITY + 10) {
            if q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: i as u16, compiled_for_fr1: true }, &stats) {
                accepted += 1;
            }
        }
        assert_eq!(accepted, COMPILE_QUEUE_CAPACITY, "queue must drop pushes past capacity, not block or panic");
    }

    #[test]
    fn compile_queue_start_is_idempotent() {
        let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.start(dev.clone(), std::sync::Arc::new(JitStats::default()));
        q.start(dev, std::sync::Arc::new(JitStats::default())); // must not panic or spawn a second thread
        q.stop();
    }

    #[test]
    fn compile_queue_stop_without_start_is_a_noop() {
        let mut q = CompileQueue::new();
        q.stop(); // must not panic
    }
}
