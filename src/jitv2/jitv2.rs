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

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;

use parking_lot::Mutex;

use crate::mips_core::MipsCore;
use crate::mips_exec::ExecStatus;
use crate::traits::{BusDevice, Device};

/// A compile request pushed from the mips exec thread to the compile thread
/// over the SPSC fifo (§6.4). Carries the page pointer rather than a
/// snapshotted generation: the compile thread reads `gen` itself at snapshot
/// time (§6.5 step 2, `gen_snap = gen`) and re-reads it at publish time — the
/// generation at queue time is never consulted, only current-at-compile and
/// current-at-publish. The pointer is mutable because publish (§6.1.3) writes
/// into the page's `entry_table`/`entry_bits`/`artifact_list`.
/// The compile thread snapshots the page's whole `requested` bitmap
/// fresh at dequeue time, picking up whatever accumulated by then,
/// not just whichever offset happened to trigger this send.
///
/// # Safety
/// `page` must outlive the request — pages live for the lifetime of their
/// owning device (see [`PhysicalCodePage`]'s Send/Sync safety note).
#[derive(Debug)]
pub struct CompileRequest {
    pub page: *mut PhysicalCodePage,
    /// Live `STATUS_FR` bit at enqueue time, threaded through because the
    /// compile thread has no `MipsCore` to read it from itself — codegen's
    /// FPR-access emitters are FR-mode-specific and must match whatever mode
    /// the executor will actually be in when it calls the compiled function
    /// (same value `exec_decoded` used to run the interpreter fallback for
    /// this same arrival).
    pub compiled_for_fr1: bool,
}

unsafe impl Send for CompileRequest {}

// ============================================================================
// Tunables & initial settings. Single source of truth: every knob that shapes
// JIT v2's steady-state memory footprint and flush/queue behavior lives here.
// See each constant's doc comment for the reasoning behind its specific value.
// ============================================================================

/// Number of 4 KiB physical frames in the 32-bit physical address space the
/// guest can address (`phys_addr: u32` everywhere on the fetch path) — the
/// exact entry count of `PfnMap` below.
pub const PFN_MAP_ENTRIES: usize = (u32::MAX as usize / PAGE_SIZE as usize) + 1;

/// Direct-mapped `pfn -> PageSlot` lookup, replacing the `HashMap<Pfn,
/// PageSlot>` both page pools used to carry.
///
/// One `u32` per physical frame: 1 Mi entries x 4 B = 4 MiB, allocated once
/// and never resized. `PFN_MAP_EMPTY` means "no pool slot for this frame".
///
/// Why not a hash map: `page_for` is on the page-crossing path (see
/// `jitv2_track_pcp`), and a profile of a real IRIX boot put it at ~4.5% of
/// total runtime with `alloc::alloc::Global`/`RawTable` frames directly above
/// it — the map's own incremental rehash-and-grow, since the pool claims a
/// fresh pfn on every page switch it hasn't seen. A direct index has no hash,
/// no probe sequence, no allocation ever, and no `pages[slot]` bounds-check
/// dance beyond the one the caller already does.
///
/// Boxed rather than inline so `Jitv2` itself stays small enough to move
/// cheaply (it is constructed into an `Arc<Mutex<..>>`).
pub struct PfnMap(Box<[u32]>);

/// `PfnMap`'s own "no slot here" sentinel. Numerically identical to each
/// impl module's private `NO_SLOT` (both `u32::MAX`), spelled separately
/// because `PfnMap` is shared code that predates either module in this file
/// and cannot name a `mod`-private constant.
pub const PFN_MAP_EMPTY: u32 = u32::MAX;

impl PfnMap {
    /// One allocation, every frame empty.
    pub fn new() -> Self {
        PfnMap(vec![PFN_MAP_EMPTY; PFN_MAP_ENTRIES].into_boxed_slice())
    }

    /// The slot for `pfn`, or `None` when unmapped.
    #[inline(always)]
    pub fn get(&self, pfn: u32) -> Option<u32> {
        let slot = self.0[pfn as usize];
        if slot == PFN_MAP_EMPTY { None } else { Some(slot) }
    }

    #[inline(always)]
    pub fn insert(&mut self, pfn: u32, slot: u32) {
        self.0[pfn as usize] = slot;
    }

    #[inline(always)]
    pub fn remove(&mut self, pfn: u32) {
        self.0[pfn as usize] = PFN_MAP_EMPTY;
    }

    /// Drop every mapping (`mega_flush`). A 4 MiB memset — flushes are rare
    /// and already far more expensive than this, so no generation-tag
    /// indirection is worth paying for on every lookup to avoid it.
    pub fn clear(&mut self) {
        self.0.fill(PFN_MAP_EMPTY);
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const u32 {
        self.0.as_ptr()
    }
}


impl Default for PfnMap {
    fn default() -> Self { Self::new() }
}

/// Upfront reservation for the shared `Codegen`'s Cranelift `ArenaMemoryProvider`
/// (`Codegen::new_module`'s own doc comment for why this exists at all). A
/// `PROT_NONE` virtual-address-space reservation (cheap — nothing is actually
/// committed/faulted-in until code is written there), not a RAM commitment, so
/// there's no real memory cost to reserving more than the steady-state working
/// set needs. Must stay in lockstep with `CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES`
/// below.
pub const ARENA_RESERVE_SIZE: usize = 512 * 1024 * 1024;

/// Flush threshold for the shared `Codegen`'s Cranelift arena, in bytes
/// actually reserved (`Codegen::packing_stats()`'s `reserved` — real
/// host-page-rounded arena footprint, not a function-count proxy).
/// `cranelift_jit::Memory` never frees on drop/replace (`Codegen::reset`'s own
/// doc comment), so nothing else bounds arena growth — a long-enough-running
/// compile (real IRIX boot, not just PROM) will otherwise exhaust the whole
/// `ARENA_RESERVE_SIZE` reservation. 16MiB of headroom covers the batch that
/// happens to be in flight when this trips (a batch isn't finalized/counted
/// until it flushes, so the real reservation can run slightly ahead of this
/// threshold between checks) while still flushing well before the arena's own
/// exhaustion error could ever fire — that error path (`comp::handle_request`'s
/// exhaustion match arm) stays as a belt-and-suspenders backstop, not the
/// primary trigger.
pub const CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES: u64 = ARENA_RESERVE_SIZE as u64 - 16 * 1024 * 1024;

/// Default page-pool capacity for `Jitv2::new()` as embedded in `MipsExecutor`.
/// Sizing is a Phase 0 measurement per the design doc (§9, "Max live entries per
/// epoch"); `mega_flush` absorbs it being wrong in either direction. Now that
/// the whole pool is a single array allocated once at this capacity, a larger
/// capacity is a real memory cost, not just reserved address space — 4096 is a
/// deliberately modest working-set size rather than a generous upper bound;
/// `mega_flush`'s cost of getting this wrong low (a pool-exhaustion flush) is
/// cheap relative to permanently carrying a much larger array.
pub const JITV2_INITIAL_PAGE_CAPACITY: usize = 4096;

/// Depth of the compile-request SPSC ring (§6.4 "bounded queue; drop on full —
/// hot pages re-trigger"). A starting guess, doubled from 1024 after a live
/// `j2 status` reading showed the compile thread genuinely falling behind at
/// that size (20.9% of dispatches dropped for a full queue, average depth at
/// dispatch 248.6/1024, out of 1,166,218 total dispatches during one session)
/// rather than the queue mostly sitting near-empty.
pub const COMPILE_QUEUE_CAPACITY: usize = 2048;


/// Page size for JIT v2 (§2.4) — matches the MIPS TLB/cache page granularity
/// used throughout the codebase, and the granularity of the per-page
/// generation counters. An architectural property of the machine: JIT v2
/// keys pages by PFN at this size, and `mem.rs` re-exports it as
/// `JITV2_PAGE_SIZE` for its own generation-counter indexing. Canonical home
/// for this constant.
pub const PAGE_SIZE: u32 = 4096;

/// Number of possible entry offsets per page: one per 4-byte-aligned word
/// (MIPS instructions are always word-aligned), i.e. `PAGE_SIZE / 4` (§2.4:
/// `entry_bits` 16×u64 = 1024 bits, `entry_table` 1024 entries).
pub const ENTRIES_PER_PAGE: usize = (PAGE_SIZE / 4) as usize;

/// u64 words needed for a 1-bit-per-entry bitmap over `ENTRIES_PER_PAGE` offsets.
pub const BITMAP_WORDS: usize = ENTRIES_PER_PAGE / 64;

/// Compile-time dirty-page probe: a type-erased handle to the CPU's cache,
/// installed by `MipsExecutor::install_jit_mem_ptrs` and read by the compile
/// worker before it publishes anything for a page.
///
/// # Why this exists
///
/// The compile worker snapshots a page by reading it off the bus
/// (`comp::handle_request*`), which shows it **RAM**. The guest CPU's view is
/// RAM overlaid with its own dirty cache lines. The PROM flushes after
/// relocating, so its writes land in RAM before we ever look — but IRIX
/// relies on the R4400's L1I/L1D coherency and inclusive L2 to make freshly
/// written code visible to the fetch path without an explicit writeback.
/// Correct for the CPU, invisible to us: the page generation counter is
/// bumped by *bus* writes, so a store that retires entirely inside L1-D bumps
/// nothing, and the worker compiles pre-store bytes and publishes them as
/// valid. This is the one hole the generation counter does not cover.
///
/// # Shape
///
/// A `(ctx, fn)` pair rather than a trait object: `MipsCache` is const-generic
/// over cache geometry and the executor owns its cache by value, so the `fn`
/// is a monomorphised thunk that casts `ctx` back to the concrete cache type.
/// No vtable, no `Arc`, no allocation on the compile path.
///
/// A process-global rather than a `Jitv2` field, matching
/// `MIN_CALLS_BEFORE_COMPILE` above: every consumer is on the compile
/// worker's snapshot path, which would otherwise have to thread a new
/// argument through both designs' `handle_request`/`handle_request_deferred`
/// and every test that calls them. There is one CPU per process here, so a
/// global loses nothing.
///
/// Null `ctx` means "no probe installed" — every consumer then behaves exactly
/// as it did before this existed.
///
/// The whole mechanism is compiled out under `tcache`, which removes the blind
/// spot by construction rather than probing for it.
#[cfg(not(feature = "tcache"))]
static JIT_PAGE_PROBE_CTX: std::sync::atomic::AtomicPtr<()> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
#[cfg(not(feature = "tcache"))]
static JIT_PAGE_PROBE_FN: AtomicU64 = AtomicU64::new(0);

/// Install the dirty-page probe. Called by the CPU thread from
/// `MipsExecutor::install_jit_mem_ptrs`, which already re-runs on every event
/// that can move or replace the cache (snapshot restore, reconfigure).
///
/// # Safety
/// `ctx` must point at a live cache of the type `probe` was monomorphised for,
/// and must stay valid and un-moved for as long as a compile worker can run —
/// until a later `install_jit_page_probe` replaces it or
/// `clear_jit_page_probe` retires it.
#[cfg(not(feature = "tcache"))]
pub unsafe fn install_jit_page_probe(ctx: *const (), probe: fn(*const (), u64) -> bool) {
    // fn first, then ctx: `jit_page_has_dirty_lines` gates on ctx being
    // non-null, so publishing ctx last means it can never observe a live ctx
    // paired with a stale or zero fn.
    JIT_PAGE_PROBE_FN.store(probe as usize as u64, Ordering::Relaxed);
    JIT_PAGE_PROBE_CTX.store(ctx as *mut (), Ordering::Release);
}

/// Test-only exclusion for the probe global.
///
/// The probe is one process-wide `(ctx, fn)` pair, but several places install
/// into it: `mips_exec`'s executor setup, and the unit tests in both `comp.rs`
/// and `mips_cache_v2.rs`. `cargo test` runs those concurrently, so any test
/// that installs a stub must hold *this* lock rather than a private one — two
/// separate mutexes guarding one global exclude nothing, and the resulting
/// interference shows up as a test seeing no probe installed (or another
/// test's stub answering for its page).
#[cfg(all(test, not(feature = "tcache")))]
pub fn probe_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Retire the probe. After this, compiles stop consulting the cache and behave
/// as they did before it existed. Must be called before the cache handed to
/// `install_jit_page_probe` can go away.
#[cfg(not(feature = "tcache"))]
pub fn clear_jit_page_probe() {
    JIT_PAGE_PROBE_CTX.store(std::ptr::null_mut(), Ordering::Release);
}

/// Retire the probe only if `ctx` is still the installed one.
///
/// For a dying owner (`Drop for MipsExecutor`) that must not stomp a probe some
/// *other*, still-live executor installed after it — the common shape in a test
/// binary, where executors are created and dropped continuously while the probe
/// global is process-wide. An unconditional clear there would silently disable
/// the live executor's dirty-page check instead of just retiring the dead one's.
#[cfg(not(feature = "tcache"))]
pub fn clear_jit_page_probe_if(ctx: *const ()) {
    let _ = JIT_PAGE_PROBE_CTX.compare_exchange(
        ctx as *mut (),
        std::ptr::null_mut(),
        Ordering::Release,
        Ordering::Relaxed,
    );
}

/// Does the CPU hold dirty cache lines for the 4KB physical page at
/// `page_base`?
///
/// `true` means the compile worker must abort and publish nothing: RAM does
/// not match what the guest CPU would execute. `false` means "no reason found
/// to abort" — **not** a guarantee.
///
/// # This read is deliberately unsynchronised
///
/// It runs on the compile thread while the CPU thread freely mutates the very
/// tag arrays it walks. No lock, no atomics on the tags, no ordering: plain
/// racy reads that may be torn or stale. That is an accepted trade:
///
/// - The alternatives measured worse. Flushing the cache for every page the
///   compiler touches wrecks guest performance, and `tcache` (cache reads RAM
///   transparently, removing the blind spot by construction) costs roughly
///   10-20% across the benchmark suite. This is one short tag scan per
///   *compile*, not per memory access, and it is off the CPU thread.
/// - A false `true` (we saw a dirty tag being cleaned right now) is free: one
///   skipped compile, retried later. Pure lost work, no correctness effect.
/// - A false `false` (a store lands just after we looked) leaves exactly the
///   hole that exists today — no worse. The generation counter still guards
///   every bus-visible write and `publish` re-checks it against `gen_snap`.
///
/// So this strictly shrinks an existing race rather than introducing a new
/// one. Treat it as a mitigation, never as a correctness barrier — do not
/// build anything on top of it that assumes a definite answer.
#[cfg(not(feature = "tcache"))]
pub fn jit_page_has_dirty_lines(page_base: u64) -> bool {
    let ctx = JIT_PAGE_PROBE_CTX.load(Ordering::Acquire);
    if ctx.is_null() {
        return false;
    }
    let raw = JIT_PAGE_PROBE_FN.load(Ordering::Relaxed);
    if raw == 0 {
        return false;
    }
    // SAFETY: `install_jit_page_probe`'s contract — `ctx` is a live cache of
    // the type `probe` was monomorphised for, kept alive until cleared. The
    // tag *contents* it goes on to read race with the CPU thread by design;
    // the pointer itself does not.
    let probe: fn(*const (), u64) -> bool = unsafe { std::mem::transmute(raw as usize) };
    probe(ctx as *const (), page_base)
}

// ============================================================================
// Local Tunables & Sentinels
// ============================================================================

/// How many of the most-recently-used pages survive a flush with their
/// `requested`/`denied` bitmaps (and pfn/gen/hash entry) intact, to be


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
/// `CompileRequest` — `j2 min-calls [N]` tunes this at runtime.
///
/// **Disabled for the §13 page-consolidation experiment**: always 0
/// ("always ready" — every eligible offset gets a `CompileRequest` on its
/// first arrival, no per-offset arrival counting). §13.1's `PhysicalCodePage`
/// no longer has a spare pre-publish field to borrow as a counter (the old
/// mechanism reused each `JitEntry`'s own `gen` slot, which doesn't exist per-
/// offset anymore — see §13.7 for the page-level replacement this needs
/// eventually). `set_min_calls_before_compile` is kept as a no-op setter so
/// `j2 min-calls` doesn't need to be ripped out of the monitor console while
/// this lands.
static MIN_CALLS_BEFORE_COMPILE: AtomicU64 = AtomicU64::new(0);

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

/// Sentinel `pfn` for an unclaimed slot (`PhysicalCodePage::new`/
/// `reset_to_unclaimed`) — deliberately NOT `0`: pfn 0 is a real, legitimate
/// physical page (low RAM), so zeroing `pfn` on reset let any stale reference
/// to a freed slot (a dangling raw pointer held past a `mega_flush` eviction,
/// or a `pfn_to_slot` entry that wasn't removed before the reset) silently
/// read back as "page 0's real data" instead of something obviously wrong —
/// undetectable by inspection (`j2 pcp` on a stale pointer would show a
/// plausible-looking pfn/gen/entry_gen for the wrong page, not a crash or an
/// obvious sentinel). `Pfn::MAX` can never collide with a real pfn (would
/// require 16TB+ of physical address space at `PAGE_SIZE` granularity).
pub const UNCLAIMED_PFN: Pfn = Pfn::MAX;

/// No FR re-pin outstanding — see `PhysicalCodePage::fr_repin`. Zero so a
/// freshly-zeroed or reset slot means "nothing demanded", the safe default.
pub const FR_REPIN_NONE: u8 = 0;
/// An FR-guard bail observed live FR=0 and demands the page be re-pinned to
/// FR0 at its next compile.
pub const FR_REPIN_FR0: u8 = 1;
/// An FR-guard bail observed live FR=1 and demands the page be re-pinned to
/// FR1 at its next compile.
pub const FR_REPIN_FR1: u8 = 2;

/// Compiled-function ABI (§6.1.2's "handler ABI", simplified for this storage
/// pass — no `DecodedInstr`/state-struct plumbing yet, just direct MipsCore
/// access): takes a pointer to the executor's `MipsCore` and returns the same
/// `ExecStatus` every interpreter handler returns. `vbase` derivation, the
/// two mirrored checks (§3.2), and exit-stub materialization all live inside
/// the compiled body once codegen exists — this signature is just the call
/// boundary.
pub type JitFn = unsafe extern "C" fn(*mut MipsCore) -> ExecStatus;

/// A page-level bitmap over `ENTRIES_PER_PAGE` offsets — one bit per
/// 4-byte-aligned word (§13.1). `BITMAP_WORDS` (=16) `AtomicU64`s.
pub type EntryBitmap = [AtomicU64; BITMAP_WORDS];

fn new_bitmap() -> EntryBitmap {
    std::array::from_fn(|_| AtomicU64::new(0))
}

/// All-ones bitmap — the fresh/reset state for `denied` (§13.1: inverted
/// sense, 1 = still eligible to compile), so a freshly-claimed or
/// just-reset page means "everything eligible" without a separate memset
/// pass anywhere that constructs one.
fn new_bitmap_all_set() -> EntryBitmap {
    std::array::from_fn(|_| AtomicU64::new(u64::MAX))
}

#[inline]
fn bitmap_test(bm: &EntryBitmap, offset: usize, order: Ordering) -> bool {
    bm[offset >> 6].load(order) & (1u64 << (offset & 63)) != 0
}

#[inline]
fn bitmap_set(bm: &EntryBitmap, offset: usize, order: Ordering) {
    bm[offset >> 6].fetch_or(1u64 << (offset & 63), order);
}

/// Test-and-set a single bit; returns `true` iff this call is the one that
/// set it (i.e. it was previously clear) — same "first caller wins" contract
/// as the old per-entry `ENTRY_SCHEDULED` flag's `try_schedule`.
#[inline]
fn bitmap_test_and_set(bm: &EntryBitmap, offset: usize, order: Ordering) -> bool {
    let prev = bm[offset >> 6].fetch_or(1u64 << (offset & 63), order);
    prev & (1u64 << (offset & 63)) == 0
}

/// Snapshot every word of a bitmap into a plain array (relaxed loads — see
/// §13.3 step 3: this need not be seqlock'd against anything, under- or
/// over-collecting relative to a single instant is harmless here).
fn bitmap_snapshot(bm: &EntryBitmap) -> [u64; BITMAP_WORDS] {
    std::array::from_fn(|i| bm[i].load(Ordering::Relaxed))
}

/// Whether every set bit in `subset` is also set in `bm` (relaxed loads).
fn bitmap_is_subset_of(subset: &[u64; BITMAP_WORDS], bm: &EntryBitmap) -> bool {
    (0..BITMAP_WORDS).all(|i| subset[i] & !bm[i].load(Ordering::Relaxed) == 0)
}

/// [`bitmap_is_subset_of`] for two plain arrays — the churn-avoidance
/// snapshot's bitmaps (`CompileSnapshot`) are ordinary `[u64; _]` behind a
/// `Mutex`, not atomics, since nothing reads them lock-free.
fn bitmap_is_subset_of_plain(subset: &[u64; BITMAP_WORDS], of: &[u64; BITMAP_WORDS]) -> bool {
    (0..BITMAP_WORDS).all(|i| subset[i] & !of[i] == 0)
}

/// OR `bits` into `bm`, word by word, at the given ordering — used to publish
/// a compile's newly-covered entry points into `compiled` (§13.3 step 6).
fn bitmap_union_from(bm: &EntryBitmap, bits: &[u64; BITMAP_WORDS], order: Ordering) {
    for i in 0..BITMAP_WORDS {
        if bits[i] != 0 {
            bm[i].fetch_or(bits[i], order);
        }
    }
}

/// Whether a bitmap is entirely zero (relaxed loads) — used by the
/// pre-compile/pre-publish subsumption checks (§13.3 steps 4/6) to skip the
/// "is X a subset of Y" walk when there's nothing requested at all.
fn bitmap_is_empty(bm: &[u64; BITMAP_WORDS]) -> bool {
    bm.iter().all(|&w| w == 0)
}

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
    /// The dirty-page probe (`jit_page_has_dirty_lines`) found cache lines
    /// for this page that have not reached RAM, so the bus snapshot the
    /// compile would have been built from is known-stale. Not a codegen gap
    /// and not sticky: the offset stays un-denylisted and is retried on its
    /// next arrival, by which point the lines have usually drained on their
    /// own. A steadily climbing count here means the guest is writing code
    /// into a page it keeps executing from — expected during module load or
    /// relocation, suspicious if it never settles.
    PageDirtyInCache,
}
#[cfg(feature = "developer")]
pub const REJECT_REASON_COUNT: usize = 5;
#[cfg(feature = "developer")]
impl RejectReason {
    /// Every variant, in display order. The single source of truth for
    /// anything that enumerates reasons (`j2 stats`'s breakdown) — a new
    /// variant missing from here would silently never be reported, which is
    /// exactly how `PageDirtyInCache` first went unnoticed. `index()`'s
    /// exhaustive `match` is what forces this to be updated: adding a variant
    /// without extending it fails to compile, and the length assertion below
    /// catches a variant added to the match but not to this list.
    pub const ALL: [RejectReason; REJECT_REASON_COUNT] = [
        RejectReason::EntryExcluded,
        RejectReason::AnalyzerCodegenDisagreement,
        RejectReason::CraneliftVerifierError,
        RejectReason::TooShort,
        RejectReason::PageDirtyInCache,
    ];

    pub fn index(self) -> usize {
        match self {
            RejectReason::EntryExcluded => 0,
            RejectReason::AnalyzerCodegenDisagreement => 1,
            RejectReason::CraneliftVerifierError => 2,
            RejectReason::TooShort => 3,
            RejectReason::PageDirtyInCache => 4,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            RejectReason::EntryExcluded => "entry excluded (unsupported first instruction / architectural exclusion)",
            RejectReason::AnalyzerCodegenDisagreement => "analyzer/codegen disagreement (should be unreachable — see doc comment)",
            RejectReason::CraneliftVerifierError => "Cranelift verifier error (real emitter bug)",
            RejectReason::TooShort => "region too short (below j2 min-instrs threshold)",
            RejectReason::PageDirtyInCache => "page dirty in CPU cache (snapshot would be stale — retried later)",
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
    /// bailing and un-publishing its own entry (`emit_fr_mode_guard`'s FR
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
        // A dirty-page abort is a *deferral*, not a failed compile: the page
        // is fine, it just is not in RAM yet, and the same offsets recompile
        // successfully on a later arrival. Counting it under
        // `failed_compiles` would make codegen look like it was regressing
        // while it was working correctly, so it gets its own bucket only.
        if !matches!(reason, RejectReason::PageDirtyInCache) {
            self.failed_compiles.fetch_add(1, Ordering::Relaxed);
        }
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

/// Per-physical-page code cache metadata (§13.1 — one compiled function per
/// page, internal dispatch by entry offset, superseding the old one-function-
/// per-entry `JitEntry` table). One instance per physical RAM/ROM page that
/// has ever been a JIT compilation target; the executor holds a pointer to
/// the page it is currently executing out of.
pub struct PhysicalCodePage {
    pub pfn: Pfn,
    /// Free list pointer: singly-linked free list threaded through
    /// `Jitv2::free_head`. `NO_SLOT` when claimed or at tail of free list.
    next: u32,
    /// Pointer to this page's generation counter, obtained from the owning
    /// `BusDevice` via `gen_ptr` (§2.4, §7). RAM devices return one counter
    /// per page; ROM devices point every page at a single counter that is
    /// initialized to 0 and never bumped, since ROM content is immutable.
    /// Never null — a device with no real gen tracking (MMIO, etc — `gen_ptr`
    /// returns null) gets pointed at the shared [`NEVER_COMPILABLE_GEN`]
    /// fallback instead, by `claim()`.
    gen: *const AtomicU64,
    /// Offsets some dispatch has asked for an entry point at (§13.2) — a
    /// superset of `compiled` in general (a bit can be requested long before
    /// any compile picks it up, and stays requested across however many
    /// compiles it takes to eventually get covered).
    requested: EntryBitmap,
    /// Inverted sticky-rejection bitmap: **1 = still eligible to compile, 0 =
    /// permanently denied** for the page's current generation (§13.1, §13.6).
    /// Inverted specifically so it composes with `bitmap_is_subset_of`/
    /// dispatch's other bitmaps the same way — but that means a fresh/reset
    /// page must be explicitly initialized all-ones (`new_bitmap_all_set`,
    /// not the plain zeroed `new_bitmap` every other bitmap here uses), or
    /// every offset would read as denylisted before anything ever ran.
    /// Reset to all-ones again whenever a publish actually advances
    /// `entry_gen` (§13.6) — a pure entry-coverage publish against unchanged
    /// bytes leaves it alone.
    denied: EntryBitmap,
    /// Offsets that are live entry points into `func` right now (§13.2).
    /// Authoritative for dispatch together with `entry_gen` — see
    /// `is_runnable`.
    compiled: EntryBitmap,
    /// The page's one compiled function, or null if nothing has published
    /// yet. Validity is owned by `compiled`'s bits together with `entry_gen`
    /// matching `current_gen()` — see `is_runnable`.
    func: std::sync::atomic::AtomicPtr<()>,
    /// Generation this `func`/`compiled` pair was last published against
    /// (§13.3 step 6). Only ever written to a value `>=` its current value;
    /// numerically advances only when the publish that wrote it was actually
    /// anchored to newer page bytes than what was live before (the
    /// gen-bump case) — a publish that only adds entry-point coverage against
    /// unchanged bytes rewrites the same value. Readers never need to
    /// distinguish the two cases: `entry_gen == current_gen()` is the only
    /// fact dispatch needs (§13.2).
    entry_gen: AtomicU64,
    /// Page-level "a `CompileRequest` for this page is already in flight"
    /// flag (§13.2). Distinct from `requested`'s per-offset bits: those track
    /// *what* has been asked for and stay set until a compile actually covers
    /// them (§13.3's coalescing — many discoveries share one in-flight
    /// request), while this is purely a dedup gate on *sending* — cleared
    /// once `handle_request` has decided the page one way or another (some
    /// coverage published, or nothing new to offer), mirroring the old
    /// per-offset `ENTRY_SCHEDULED` flag's contract but page-scoped instead
    /// of offset-scoped, since one request now serves however many offsets
    /// have accumulated in `requested` by the time it's dequeued.
    page_scheduled: std::sync::atomic::AtomicBool,
    /// Pinned FR mode for this page, set once at page-claim time from live
    /// `STATUS_FR` and held fixed for the page's whole lifetime until the
    /// next `mega_flush`/`reset_to_unclaimed` (TODO revisit — see below).
    /// `true` = FR1.
    ///
    /// **Why pinned, not per-compile** (as it was under the old one-function-
    /// per-entry design, where `CompileRequest::compiled_for_fr1` could
    /// legitimately differ compile to compile): §13's one-function-per-page
    /// model means every entry point on a page shares the SAME compiled
    /// function, and FR mode is baked into every FPR-access emitter in that
    /// function at compile time (§4.2.1) — there is nowhere to put a second
    /// version for a second FR mode. If two entries on one page genuinely
    /// wanted different FR modes, only one can ever be served; pinning at
    /// claim time picks whichever mode was live first and commits to it.
    ///
    /// **No dispatch-side check**: unlike an earlier draft of this field,
    /// `exec_decoded`'s gate does NOT compare live `STATUS_FR` against this
    /// value before calling `func` — that would tax every single dispatch
    /// through the hot gate, forever, on every page. Instead the existing
    /// in-function FR-mode guard (`emit_fr_mode_guard`) does the job,
    /// unchanged in its existing gating: only emitted when the compiled
    /// region actually contains a CP1 instruction (`has_fpu`,
    /// `compile_region_uncommitted`) — a page with no FPU usage anywhere in
    /// it has no FR-mode-dependent code to protect, so it pays nothing at
    /// all, dispatch-side or in-function. Where the guard IS present, a
    /// mismatch kills just the triggering entry (`jit_kill_entry`/
    /// `page.kill`) — a page whose workload keeps arriving in the wrong mode
    /// simply has its FPU-touching entries killed one at a time as each is
    /// dispatched, self-limiting to at most (published entry count) guard
    /// failures rather than one check per dispatch forever.
    ///
    /// **Re-pinned, as of the fix below.** The original TODO here said `fr1`
    /// was never re-pinned, and justified it as "measured at ~2 occurrences
    /// per real IRIX boot (rare)". That measurement was of a *different*
    /// scenario — a page whose entries all get killed — and said nothing
    /// about **page recycling**, which is the damaging case: a page freed by
    /// an o32 (FR0) process and reallocated to an n32 (FR1) one kept the old
    /// pin even though every byte on it was new. Real symptom: 4Dwm/fm/
    /// background core-dumping at session start (a burst of exactly that
    /// recycling), cleared by `j2 flush`. Two resets now cover it, and
    /// neither subsumes the other:
    ///
    /// 1. [`Self::publish`] clears the pin on a real invalidation (the guest
    ///    wrote new bytes), so the next compile re-derives from the live
    ///    mode. This sits beside the existing `compiled`/`denied` resets,
    ///    which already reason "the bytes changed, so what was decided
    ///    against the OLD bytes deserves a fresh chance" — `fr1` had
    ///    identical justification and was the one field left behind.
    /// 2. [`Self::request_fr_repin`] records the *demanded* mode when the
    ///    in-function FR guard bails, for the case (1) cannot see: the same
    ///    bytes arriving in a new mode (the shared `checkfp.s` page).
    ///
    /// Why the guard alone was never sufficient: `emit_fr_mode_guard`
    /// computes `band(cu1_set, ...)`, deliberately suppressing the FR check
    /// when CU1 is clear so it can't pre-empt the real per-instruction CU1
    /// fault. So a stale pin is *not* always caught, and wrong FR packing on
    /// `ldc1`/`lwc1` puts an FP value in the wrong register half — silent
    /// corruption, not a fault.
    ///
    /// Entries already published against the old mode are retired **lazily**:
    /// the per-entry guard kills them one at a time and `mega_flush` recycles
    /// the page eventually. A page-wide kill on re-pin would discard correct
    /// code — non-FP entries on the same page are mode-independent — to save
    /// a bounded number of guard failures.
    fr1: std::sync::atomic::AtomicBool,
    /// FR mode demanded by the most recent in-function FR-guard bail, or
    /// `None` if no bail is outstanding. Read and cleared by the next compile
    /// for this page, which re-pins [`Self::fr1`] to it.
    ///
    /// Stores the **demanded mode**, not a "flip me" bit, deliberately.
    /// Several bails normally land before the next compile — the guard kills
    /// a page's FPU-touching entries one at a time as each is dispatched, and
    /// each kill records a demand — so a flip bit would toggle once per
    /// killed entry and an even number of kills would settle back on the
    /// wrong mode. Recording the observed live FR is idempotent: N bails all
    /// demanding FR1 say "FR1" N times, and the recompile reads a value
    /// rather than applying a delta.
    ///
    /// Encoded as a `u8` rather than `Option<bool>` so it can live in an
    /// atomic: [`FR_REPIN_NONE`], [`FR_REPIN_FR0`], [`FR_REPIN_FR1`].
    fr_repin: std::sync::atomic::AtomicU8,
    /// Set by [`Self::publish`] on a real invalidation: the page's bytes were
    /// replaced, so the FR mode pinned against the OLD bytes carries no
    /// information about what the new ones want. The next compile re-derives
    /// [`Self::fr1`] from the live mode rather than keeping the stale pin.
    ///
    /// Separate from [`Self::fr_repin`] because the two answer different
    /// questions — `fr_repin` says "a bail proved the pin wrong, use THIS
    /// mode", while this says "the pin is merely unjustified, use whatever is
    /// live at the next compile". A bail's demand is the stronger signal and
    /// wins when both are set.
    fr_unpinned: std::sync::atomic::AtomicBool,
    /// Serializes the publish step (§13.3 step 6/§13.5) — the compare/union/
    /// swap of `func`+`compiled`+`entry_gen` together needs to happen as one
    /// step even though individual dispatch reads of those fields remain
    /// independently orderable. Does **not** guard the expensive work
    /// (snapshot/analyze/codegen, §13.3 steps 1-5) — compiles proceed
    /// unlocked and in parallel, including two compiles of the same page;
    /// only the cheap publish tail is serialized here.
    publish_lock: Mutex<()>,
    /// Page-granular compile diagnostics for `j2 pcp`/`j2 status` (§13.1 —
    /// page-granular now that there's one function; the old per-entry
    /// `instr_count`/`code_size`/`call_count`/`block_include_count` don't
    /// have a meaningful per-offset home anymore since overlap between
    /// entries is structurally impossible under this design).
    ///
    /// **`instr_count`/`code_size` are NOT `developer`-gated**, unlike the
    /// counters below. They are the emitted-code-size metric
    /// (`code_size_by_instr_count`, `j2 status`' bytes/instruction), and
    /// gating them made that metric available *only* in the build that
    /// invalidates it: `developer` forces `opt_level=none` **and** emits
    /// `emit_dev_trace_bp` — a 4-argument `call_indirect` per instruction,
    /// which is both large and an opaque clobber that kills the GPR
    /// forwarding the metric is usually being read to evaluate. Measured
    /// live, that build reports ~380 code bytes per guest instruction where
    /// the same pages compile to ~92 without `developer`: a 4x error, and
    /// the whole point of the number is to compare codegen changes.
    ///
    /// Cost of keeping them always-on: 8 bytes per pooled page (two u32),
    /// i.e. 32 KiB at the default 4096-page capacity, written once per
    /// successful `publish`. Nothing reads them on any hot path. See
    /// `rules/jitv2/block-fragmentation-blocks-cse.md`'s "measure without
    /// `developer`" rule, which this makes possible to actually follow.
    pub instr_count: std::sync::atomic::AtomicU32,
    pub code_size: std::sync::atomic::AtomicU32,
    #[cfg(feature = "developer")]
    pub call_count: AtomicU64,
    /// How many times `publish()` has actually committed a fresh compile for
    /// this page — incremented once per successful `publish()` call (the two
    /// early `return false`s, stale-generation and subsumed-by-existing-
    /// coverage, don't count: nothing new was compiled in). `j2 html`'s
    /// per-page stats use this to show recompile churn (a page compiled 40
    /// times is either very hot or thrashing on a denylist/invalidation
    /// loop — both worth seeing at a glance).
    #[cfg(feature = "developer")]
    pub compile_count: std::sync::atomic::AtomicU32,
    /// Same trigger as `compile_count` (bumped once per successful
    /// `publish()`), but with the opposite lifetime and always present, not
    /// `developer`-gated: `compile_count` is a lifetime-total churn counter
    /// deliberately preserved across a page's reset (see that field's own
    /// doc comment), whereas this one is cleared by `reset_to_unclaimed` —
    /// so it answers
    /// "has this page compiled at all since its last flush/claim" directly,
    /// without needing a `developer` build or reasoning through generation
    /// numbers to rule out "never even tried" vs "tried and is legitimately
    /// stuck." `j2 pcp` prints it unconditionally for exactly that reason —
    /// a page reporting `gen` in the thousands with `compiles_since_flush=0`
    /// is a real red flag (scheduling/dispatch never reached it, or every
    /// attempt bailed before `publish`), not just "hasn't gotten hot yet."
    pub compiles_since_flush: std::sync::atomic::AtomicU32,
    /// How many times `prepare_multi_entry_compile`/`handle_request(_deferred)`
    /// (`comp.rs`) has REJECTED a compile attempt for this page — denylisted
    /// (per-entry codegen exclusion, or a whole-region decline), too-short,
    /// or out-of-memory-retry — as opposed to a successful `publish()`
    /// (tracked by `compiles_since_flush`/`compile_count` instead). Always
    /// present, not `developer`-gated, and NOT cleared by any reset path
    /// (lifetime total, same reasoning as `compile_count`'s own doc comment
    /// on why churn history outlives a flush the page survived) — the point
    /// is distinguishing "this page has been attempted repeatedly and keeps
    /// getting rejected" from "this page has never been attempted at all"
    /// (`compiles_since_flush=0` alone can't tell those apart), directly in
    /// `j2 pcp` without needing a `developer` build's `j2 status` rejection
    /// breakdown.
    pub rejected_compiles: std::sync::atomic::AtomicU32,
    /// The `rejected_compiles` subset decided during the analyze stage
    /// (`prepare_multi_entry_compile`, `comp.rs`): a candidate excluded at
    /// its own entry during the walk (`EntryExcluded`), or the whole merged
    /// region coming in too short (`TooShort`). Distinct from
    /// `codegen_rejected` — this page's request never even reached
    /// Cranelift. Always present, not `developer`-gated, lifetime total.
    pub analyze_rejected: std::sync::atomic::AtomicU32,
    /// The `rejected_compiles` subset decided during the codegen stage
    /// (`handle_request`/`handle_request_deferred`'s non-OOM `None =>` arm,
    /// `comp.rs`): Cranelift itself declined — no emitter for some visited
    /// instruction, or a real verifier rejection. Always present, not
    /// `developer`-gated, lifetime total.
    pub codegen_rejected: std::sync::atomic::AtomicU32,
    /// How many times a dispatch has REACHED the `try_schedule_page()`
    /// test-and-set for this page (`exec_decoded`'s dispatch gate,
    /// `mips_exec.rs`) — every such dispatch, win or lose, not just the one
    /// that actually goes on to send (that subset is `sends_attempted`
    /// instead). `schedule_attempts == 0` on a page with real `requested`
    /// bits means dispatch never even reached this branch for it at all — a
    /// different, earlier problem than anything downstream of the
    /// test-and-set. `schedule_attempts` vs `sends_attempted` shows the
    /// attempt/win ratio directly (a hot page loses the race often, a cold
    /// one wins almost every time). Lifetime total, not cleared by any
    /// reset. `developer`-gated: this fires on EVERY dispatch that reaches
    /// the trigger gate, win or lose — by far the hottest of the three
    /// dispatch-side counters, not worth an atomic increment on the CPU's
    /// hot path in a normal/`lightning` build (the root cause this whole
    /// family of counters was built to chase turned out to be in the arena
    /// allocator, not dispatch scheduling — see `codegen_oom_bounced`'s
    /// sibling comment history).
    #[cfg(feature = "developer")]
    pub schedule_attempts: std::sync::atomic::AtomicU32,
    /// How many times a `CompileRequest` for this page was actually handed to
    /// `push_compile_request` (`mips_exec.rs`'s dispatch gate) — i.e. this
    /// dispatch won `try_schedule_page()` AND the send call was reached.
    /// Should track `schedule_attempts` 1:1 today (nothing currently sits
    /// between winning the flag and calling send) — kept as its own counter
    /// rather than assumed equal, so a future divergence between the two is
    /// itself a diagnostic signal. Lifetime total. `developer`-gated — see
    /// `schedule_attempts`'s own doc comment for why the whole dispatch-side
    /// family moved behind it.
    #[cfg(feature = "developer")]
    pub sends_attempted: std::sync::atomic::AtomicU32,
    /// How many times `push_compile_request` returned `false` for this page
    /// — the bounded compile queue was full at that exact moment, so the
    /// request was dropped before ever reaching a worker thread (never
    /// `handle_request`/`handle_request_deferred`, so neither `compiles_since_flush`
    /// nor `rejected_compiles` ever sees it). The dispatch gate clears
    /// `page_scheduled` itself when this happens (see that call site's own
    /// comment) so the page isn't permanently starved, but the compile
    /// itself is still lost — this counter is what makes that loss visible
    /// in `j2 pcp` instead of silently returning to zero. Lifetime total.
    /// `developer`-gated — see `schedule_attempts`'s own doc comment for why
    /// the whole dispatch-side family moved behind it.
    #[cfg(feature = "developer")]
    pub sends_dropped_queue_full: std::sync::atomic::AtomicU32,
    /// How many times `prepare_multi_entry_compile` (`comp.rs`) bailed out
    /// via one of its three silent, non-denylisting early returns for this
    /// page: a `bus.read32` failure mid-snapshot, the 8-retry seqlock never
    /// stabilizing (`current_gen()` kept changing under it), or the
    /// subsumption check finding nothing new to offer. None of these touch
    /// `rejected_compiles` (nothing was actually rejected — no offset was
    /// denylisted, no codegen ran) yet they still consume the one
    /// `CompileRequest` this dispatch's `try_schedule_page()` win produced,
    /// same as any other outcome — so a page could show `sends_attempted`
    /// far above `rejected_compiles + compiles_since_flush` with this being
    /// the reason why: NOT a queue drop, but every attempt individually
    /// losing the seqlock race (or hitting a transient bus-read failure)
    /// before ever reaching the actual compiler. A sufficiently hot
    /// self-modifying/shared-DMA page bumping `gen` faster than 8 retries
    /// can outrun is the realistic way this happens repeatedly, not a one-off.
    /// Always present, not `developer`-gated, lifetime total (not cleared by
    /// any reset — same reasoning as `rejected_compiles`).
    pub prepare_bounced: std::sync::atomic::AtomicU32,
    /// How many times `Codegen::compile_region_uncommitted` returned `None`
    /// specifically because `last_compile_ran_out_of_memory()` was true for
    /// this page's compile — the shared Cranelift arena was full at that
    /// exact moment, NOT a real decline (no emitter gap, no verifier
    /// rejection). Distinct from `rejected_compiles`: the caller
    /// deliberately does NOT denylist anything here (see that call site's
    /// own comment — "leave everything un-denylisted so it retries
    /// naturally on a later arrival") and instead triggers a pool-wide arena
    /// flush. If this page's compiled region is large/expensive enough that
    /// its turn in the queue keeps landing exactly when the arena is
    /// already near its flush threshold, EVERY attempt can bounce here,
    /// forever, with nothing else this page shows (`rejected_compiles`,
    /// `prepare_bounced`, `sends_dropped_queue_full` all staying 0) hinting
    /// at why — this counter is what finally makes that visible. Always
    /// present, not `developer`-gated, lifetime total (not cleared by any
    /// reset).
    pub codegen_oom_bounced: std::sync::atomic::AtomicU32,
    /// How many times this page's compile fully succeeded in codegen
    /// (`compile_region_uncommitted` returned a real `FuncId`) but
    /// `finalize_batch_nonforced` came back empty — gap-blocked behind an
    /// earlier, still-unpatched entry in the shared arena's strictly-ordered
    /// seal queue (`paged_memory.rs`'s bump allocator only ever returns a
    /// contiguous prefix). Not a loss by itself — the idle-timeout/
    /// pending-threshold sweep is supposed to eventually force-seal it — but
    /// a page whose `codegen_oom_bounced`/this counter both climb while
    /// `compiles_since_flush` stays 0 is a real, visible red flag that
    /// something downstream of a successful compile keeps consuming this
    /// page's request without ever actually publishing it. Always present,
    /// not `developer`-gated, lifetime total (not cleared by any reset).
    pub seal_gap_blocked: std::sync::atomic::AtomicU32,
    /// How many times this page's `finalize_batch_nonforced` call came back
    /// empty because `module.finalize_definitions()` itself errored — a
    /// real Cranelift/JIT-backend failure, NOT a normal gap-blocked seal
    /// (that's `seal_gap_blocked`, which self-resolves once the earlier
    /// entry closes the gap). This one never will: confirmed as a real bug
    /// — `handle_request_deferred` couldn't tell the two apart before
    /// `Codegen::last_finalize_failed` existed, so a page hitting this kept
    /// getting bumped into `pending` and waiting on a sweep to "unblock" an
    /// entry that was never blocked, just permanently failed. Always
    /// present, not `developer`-gated, lifetime total (not cleared by any
    /// reset).
    pub finalize_failed: std::sync::atomic::AtomicU32,
    /// How many compile requests for this page were answered entirely by the
    /// churn-avoidance snapshot — no walk, no codegen, no finalize, just an
    /// `entry_gen` re-validation of the code already installed. See
    /// [`Self::try_skip_redundant_compile`]; `Jitv2::redundant_compile_totals`
    /// sums these across the pool for the process-wide ratio.
    ///
    /// Lives per-page rather than on `JitStats` because a page is the only
    /// thing the compile path has in hand at the point of the decision
    /// (`prepare_multi_entry_compile` has no `JitStats` outside a
    /// `developer` build), and because the per-page breakdown is the more
    /// useful reading anyway — "which pages churn" rather than just "how
    /// much".
    ///
    /// A page with a high count here is one whose generation churns for
    /// reasons unrelated to its code — data sharing the page, or DMA into
    /// it — and is exactly the case this cache was built for. Always
    /// present, not `developer`-gated.
    ///
    /// **Per-flush-epoch, not a lifetime total** — cleared by
    /// `reset_to_unclaimed`, unlike
    /// `rejected_compiles`/`prepare_bounced` next door. A boot's exec storm
    /// racks up hundreds of thousands of skips (measured: ~286k, 98% of
    /// everything checked), and as a lifetime total that figure would sit in
    /// `j2 status` forever, drowning out whatever the system is doing now.
    /// The flush is the honest epoch boundary anyway: it invalidates the very
    /// snapshots these counted, so the old tallies describe a state that no
    /// longer exists.
    pub redundant_skipped: std::sync::atomic::AtomicU32,
    /// The complement of [`Self::redundant_skipped`] for this page: requests
    /// that had a snapshot to compare against and still had to compile. A
    /// page where this dominates is genuinely being rewritten (or flipping
    /// FR mode), not merely churning generations.
    pub redundant_rejected: std::sync::atomic::AtomicU32,
    /// Churn-avoidance cache: what the last successful compile of this page
    /// actually looked at, kept so a later request whose only difference is
    /// a bumped generation can be answered without recompiling at all. See
    /// [`CompileSnapshot`] for the contents and
    /// [`Self::try_skip_redundant_compile`] for the comparison rule.
    ///
    /// Inline rather than `Option<Box<..>>` deliberately, matching the same
    /// call made for `entries` under the old design: the pool array is
    /// allocated once up front, so a boxed payload would buy nothing but a
    /// pointer-chase and an allocation on every page's first compile, while
    /// the flat version's cost is bounded, predictable, and mostly untouched
    /// zero pages for slots that never compile.
    ///
    /// Set by [`Self::kill`], cleared by [`Self::stage_compile_snapshot`]:
    /// "some entry was killed since the snapshot below was taken."
    ///
    /// A kill means "stop dispatching into `func` here", so a skip must not
    /// re-advertise the killed offset. `compiled` cannot answer that question
    /// on its own, because a narrowing skip clears bits there for an entirely
    /// unrelated reason — hence a separate flag. Page-granular rather than
    /// per-offset because the skip is all-or-nothing anyway: one kill on the
    /// page simply costs that page one real recompile, which re-derives the
    /// coverage honestly.
    killed_since_snapshot: std::sync::atomic::AtomicBool,
    /// **Unsynchronized, and sound because a page is never compiled twice at
    /// once.** `page_scheduled` is a test-and-set taken by the dispatch gate
    /// before a `CompileRequest` is sent and released only by
    /// `handle_request`/`handle_request_deferred`'s exit scope guard — so it
    /// stays held for the *whole* compile, not just the enqueue. A second
    /// request for the same page therefore cannot even be sent, let alone
    /// dequeued, while the first is still running, at any `thread_count`.
    /// That is the same invariant that lets `Analyzer`'s 1024-entry buffer be
    /// reused across compiles without locking.
    ///
    /// A `Mutex` here would guard against nothing reachable while adding two
    /// atomics and a 4KB compare inside a critical section. Dispatch never
    /// touches this field at all — only a compiling thread does, plus the
    /// exec thread's reset paths, which are mutually exclusive with compiles
    /// by the same flag.
    compile_snapshot: UnsafeCell<CompileSnapshot>,
}

/// What a successful compile of a page looked at, recorded so the next
/// compile request for the same page can prove it would produce identical
/// code and skip the work (§ churn reduction).
///
/// The whole 4KB is stored, not a hash: the comparison must be exact (a hash
/// collision here would publish code for bytes that no longer exist), and at
/// this size a `memcmp` of the used words is cheaper than re-reading and
/// re-walking the page anyway. The `used` mask is what makes the comparison
/// *useful* rather than merely correct — a page where the guest rewrote a
/// data constant, or any word the walk never reached, still matches, because
/// only words the compile actually consumed are compared.
struct CompileSnapshot {
    /// Whether the fields below describe a real, still-installed compile.
    /// False on a fresh page and after any reset that drops `func` — a
    /// snapshot without the function it describes must never authorize a
    /// skip, since there would be nothing left to re-publish.
    ///
    /// A record starts life with this `false` ([`PhysicalCodePage::stage_compile_snapshot`])
    /// and only becomes `true` once the publish it describes actually lands
    /// ([`PhysicalCodePage::commit_compile_snapshot`]). The deferred compile
    /// path (`comp.rs`'s `handle_request_deferred`) needs that split: it
    /// hands its compile to the shared arena's seal queue and the matching
    /// publish can happen later, on another worker's `finalize` call, with
    /// only a `PublishInfo` in hand — far too late and too small to carry a
    /// 4KB page snapshot through. Staging at compile time and committing at
    /// publish time keeps the 4KB on the page it belongs to while still
    /// guaranteeing a snapshot is never consultable before the code it
    /// describes is installed.
    valid: bool,
    /// The generation the staged record was compiled against, used by
    /// [`PhysicalCodePage::commit_compile_snapshot`] to confirm the publish
    /// it is being asked to vouch for is the compile that staged this record.
    /// The case this guards is a reset landing under an in-flight compile:
    /// [`PhysicalCodePage::invalidate_compile_snapshot`] poisons this to
    /// `u64::MAX`, which no real generation can equal, so the compile's later
    /// commit finds no match and the record stays dead instead of vouching
    /// for a function the reset already discarded.
    staged_gen: u64,
    /// The page bytes as the recorded compile saw them (its seqlock
    /// snapshot, `prepare_multi_entry_compile` step 2). Only the words
    /// selected by `used` are ever compared.
    words: [u32; ENTRIES_PER_PAGE],
    /// Words the recorded compile actually decoded — every instruction the
    /// merged multi-entry walk visited (`instrs_linear`), including delay
    /// slots. A word outside this set had no influence on the emitted code,
    /// so a change to it cannot change what a recompile would produce.
    used: [u64; BITMAP_WORDS],
    /// Entry points the recorded compile published (`new_entries` as handed
    /// to `publish`). A new request asking for an entry point outside this
    /// set needs a real compile: the published `func`'s dispatch switch has
    /// no case for it, so no amount of byte equality helps.
    entries: [u64; BITMAP_WORDS],
    /// FR mode the recorded compile was specialized for
    /// (`CompileRequest::compiled_for_fr1`). Compared alongside the bytes,
    /// and NOT optional: FR mode is baked into every FPR-access emitter at
    /// compile time (§4.2.1) and is not visible anywhere in the page bytes,
    /// so identical bytes compiled for the other mode are genuinely
    /// different code. Skipping on a mismatch would leave the old mode's
    /// `func` installed while the requester expects the new one — and
    /// because `emit_fr_mode_guard` suppresses itself when CU1 is clear (see
    /// [`PhysicalCodePage::fr1`]'s doc comment), the in-function guard is not
    /// a reliable backstop: the failure mode is `ldc1`/`lwc1` packing an FP
    /// value into the wrong register half, which is silent corruption rather
    /// than a fault. A page whose FR pin legitimately flips therefore always
    /// takes a real recompile, exactly as it did before this cache existed.
    fr1: bool,
}

impl CompileSnapshot {
    fn new() -> Self {
        Self {
            valid: false,
            staged_gen: u64::MAX,
            words: [0u32; ENTRIES_PER_PAGE],
            used: [0u64; BITMAP_WORDS],
            entries: [0u64; BITMAP_WORDS],
            fr1: false,
        }
    }
}

// Safety: `gen` points into the owning BusDevice's storage, which outlives
// every PhysicalCodePage derived from it (devices are held in Arcs/statics
// for the lifetime of the machine). The pointee is only ever read or
// atomically incremented, never moved or freed while a PhysicalCodePage
// referencing it exists. `func`, when non-null, points to finalized
// JIT-compiled code owned by the compile-thread arena, which outlives every
// PhysicalCodePage referencing it until the next mega_flush.
//
// `compile_snapshot`'s UnsafeCell is what makes the Sync impl load-bearing
// rather than merely a formality for the raw `gen` pointer: it is genuinely
// unsynchronized data behind `&self`. Sound because a page is never compiled
// twice at once and nothing but a compile (or a reset, mutually exclusive
// with one) ever touches it — see that field's own doc comment for the
// `page_scheduled` argument.
unsafe impl Send for PhysicalCodePage {}
unsafe impl Sync for PhysicalCodePage {}

impl PhysicalCodePage {
    /// Construct an as-yet-unclaimed page descriptor: `pfn = 0`, every
    /// bitmap zeroed, no compiled function. Used both to build `Jitv2::pages`'
    /// full-capacity array up front (every slot starts unclaimed) and,
    /// functionally identically, by [`Self::claim`]/[`Self::reset_to_unclaimed`]
    /// to return a slot to this same state in place without reallocating
    /// anything.
    ///
    /// `gen` null is accepted here (callers pass it for an as-yet-unclaimed
    /// slot, or when the real device has no gen tracking) and normalized to
    /// [`NEVER_COMPILABLE_GEN`] — `self.gen` itself is never null, so nothing
    /// downstream ever needs to check.
    pub fn new(pfn: Pfn, gen: *const AtomicU64) -> Self {
        Self {
            pfn,
            next: NO_SLOT,
            gen: if gen.is_null() { &NEVER_COMPILABLE_GEN } else { gen },
            requested: new_bitmap(),
            denied: new_bitmap_all_set(),
            compiled: new_bitmap(),
            func: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            entry_gen: AtomicU64::new(0),
            page_scheduled: std::sync::atomic::AtomicBool::new(false),
            fr1: std::sync::atomic::AtomicBool::new(false),
            fr_repin: std::sync::atomic::AtomicU8::new(FR_REPIN_NONE),
            fr_unpinned: std::sync::atomic::AtomicBool::new(false),
            publish_lock: Mutex::new(()),
            redundant_skipped: std::sync::atomic::AtomicU32::new(0),
            redundant_rejected: std::sync::atomic::AtomicU32::new(0),
            killed_since_snapshot: std::sync::atomic::AtomicBool::new(false),
            compile_snapshot: UnsafeCell::new(CompileSnapshot::new()),
            instr_count: std::sync::atomic::AtomicU32::new(0),
            code_size: std::sync::atomic::AtomicU32::new(0),
            #[cfg(feature = "developer")]
            call_count: AtomicU64::new(0),
            #[cfg(feature = "developer")]
            compile_count: std::sync::atomic::AtomicU32::new(0),
            compiles_since_flush: std::sync::atomic::AtomicU32::new(0),
            rejected_compiles: std::sync::atomic::AtomicU32::new(0),
            analyze_rejected: std::sync::atomic::AtomicU32::new(0),
            codegen_rejected: std::sync::atomic::AtomicU32::new(0),
            #[cfg(feature = "developer")]
            schedule_attempts: std::sync::atomic::AtomicU32::new(0),
            #[cfg(feature = "developer")]
            sends_attempted: std::sync::atomic::AtomicU32::new(0),
            #[cfg(feature = "developer")]
            sends_dropped_queue_full: std::sync::atomic::AtomicU32::new(0),
            prepare_bounced: std::sync::atomic::AtomicU32::new(0),
            codegen_oom_bounced: std::sync::atomic::AtomicU32::new(0),
            seal_gap_blocked: std::sync::atomic::AtomicU32::new(0),
            finalize_failed: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Zero every bitmap and drop the compiled function, in place — no
    /// reallocation. Called only from [`Self::reset_to_unclaimed`]
    /// (`Jitv2::mega_flush`'s per-slot reset) — a fresh, never-claimed slot
    /// is already zeroed by `PhysicalCodePage::new` and doesn't need this.
    fn reset_entries_and_bitmaps(&mut self) {
        for bm in [&self.requested, &self.compiled] {
            for word in bm.iter() { word.store(0, Ordering::Relaxed); }
        }
        // `denied` resets to all-ones, not zero — see its own field doc
        // (inverted sense: 0 means "still eligible" would make every reset
        // page permanently denylisted everywhere).
        for word in self.denied.iter() { word.store(u64::MAX, Ordering::Relaxed); }
        self.func.store(std::ptr::null_mut(), Ordering::Relaxed);
        // The snapshot describes the `func` just dropped — see
        // `invalidate_compile_snapshot`.
        self.invalidate_compile_snapshot();
        self.entry_gen.store(0, Ordering::Relaxed);
        self.page_scheduled.store(false, Ordering::Relaxed);
        self.compiles_since_flush.store(0, Ordering::Relaxed);
        // Churn-avoidance tallies are per-flush-epoch (see the
        // `redundant_skipped` field), and this slot is additionally about to
        // be handed to a different physical page entirely — carrying the
        // previous tenant's counts into it would misattribute them.
        self.redundant_skipped.store(0, Ordering::Relaxed);
        self.redundant_rejected.store(0, Ordering::Relaxed);
        self.instr_count.store(0, Ordering::Relaxed);
        self.code_size.store(0, Ordering::Relaxed);
        #[cfg(feature = "developer")]
        {
            self.call_count.store(0, Ordering::Relaxed);
            self.compile_count.store(0, Ordering::Relaxed);
        }
    }

    /// Bump-allocate this (already-constructed, already-clean) slot into
    /// service for a newly-arrived physical page — the in-place counterpart
    /// to what used to be a fresh `PhysicalCodePage::new(pfn, gen)` +
    /// `Vec::push`. Deliberately does **not** reset bitmaps itself — see
    /// `reset_entries_and_bitmaps`'s call sites for why a slot only ever
    /// reaches `claim` already clean.
    ///
    /// `fr1`: live `STATUS_FR` at the moment of first arrival. Pinned, but no
    /// longer for the page's whole lifetime — it is re-derived on a real
    /// invalidation and on an FR-guard bail (see [`Self::fr1`]'s doc comment).
    #[inline]
    pub fn is_claimed(&self) -> bool {
        self.pfn != UNCLAIMED_PFN
    }

    pub fn claim(&mut self, pfn: Pfn, gen: *const AtomicU64, fr1: bool) {
        debug_assert!(std::ptr::eq(self.gen, &NEVER_COMPILABLE_GEN) && self.pfn == UNCLAIMED_PFN,
            "claim() called on a slot that wasn't clean (pfn={:#x}) — every path that reuses a slot must reset it first (see reset_to_unclaimed)",
            self.pfn);
        debug_assert!(self.func.load(Ordering::Relaxed).is_null(),
            "claim() called on a slot with a still-published function — mega_flush must reset_to_unclaimed before this slot can be reused");
        self.pfn = pfn;
        self.gen = if gen.is_null() { &NEVER_COMPILABLE_GEN } else { gen };
        self.fr1.store(fr1, Ordering::Relaxed);
        // A freshly-claimed slot has no bail outstanding: `fr1` was just set
        // from the live mode, so there is nothing to correct.
        self.fr_repin.store(FR_REPIN_NONE, Ordering::Relaxed);
        self.fr_unpinned.store(false, Ordering::Relaxed);
    }

    /// This page's pinned FR mode (§13, see [`Self::fr1`]'s own doc comment).
    #[inline]
    pub fn is_fr1(&self) -> bool {
        self.fr1.load(Ordering::Relaxed)
    }

    /// Return every offset to eligible-for-compile, discarding the sticky
    /// denylist. For the FR-mode-flip case on the flush-survivor path: the
    /// denylist records decisions made while compiling for the OLD mode, so
    /// once the page is re-pinned those verdicts no longer describe the code
    /// that will actually be generated. `publish` has its own equivalent for
    /// the real-invalidation case (new bytes deserve a fresh chance, §13.6).
    #[inline]
    pub fn reset_denied(&self) {
        for word in self.denied.iter() { word.store(u64::MAX, Ordering::Relaxed); }
    }

    /// Record that an in-function FR-guard bail observed live FR = `fr1` and
    /// that this page's pin should be corrected to it at the next compile.
    /// Called from `jit_kill_entry` (mips_exec.rs), which already holds both
    /// the executor (for live `cp0_status`) and this page.
    ///
    /// Idempotent by construction — it stores the demanded mode, so repeated
    /// bails demanding the same mode are indistinguishable from one. Last
    /// writer wins, which is correct: the most recent observation of the live
    /// mode is the best available estimate of what the next compile should
    /// target.
    #[inline]
    pub fn request_fr_repin(&self, fr1: bool) {
        self.fr_repin.store(if fr1 { FR_REPIN_FR1 } else { FR_REPIN_FR0 }, Ordering::Relaxed);
    }

    /// Consume any outstanding FR re-pin request, applying it to [`Self::fr1`]
    /// and returning the mode the next compile should target.
    ///
    /// Takes `live_fr1` so a page with no request outstanding keeps its
    /// existing pin rather than tracking whatever mode happens to be live at
    /// compile time — a page is normally dispatched in its pinned mode, and
    /// re-deriving unconditionally would let an unrelated dispatch re-pin a
    /// page that was never mismatched.
    #[inline]
    pub fn take_fr_repin(&self, live_fr1: bool) -> bool {
        match self.fr_repin.swap(FR_REPIN_NONE, Ordering::Relaxed) {
            FR_REPIN_FR0 => { self.fr1.store(false, Ordering::Relaxed); false }
            FR_REPIN_FR1 => { self.fr1.store(true, Ordering::Relaxed); true }
            // No bail outstanding. If `publish` cleared the pin on a real
            // invalidation the page is unpinned and takes the live mode;
            // otherwise it keeps what it has.
            _ => {
                if self.fr_unpinned.swap(false, Ordering::Relaxed) {
                    self.fr1.store(live_fr1, Ordering::Relaxed);
                    live_fr1
                } else {
                    self.fr1.load(Ordering::Relaxed)
                }
            }
        }
    }

    /// Return this slot to the fully-unclaimed state (`pfn = UNCLAIMED_PFN`,
    /// `gen` pointed at the shared [`NEVER_COMPILABLE_GEN`] fallback) —
    /// `Jitv2::mega_flush`'s in-place counterpart to what used to be
    /// `Vec::clear()` dropping every `PhysicalCodePage` outright. `pfn` is
    /// poisoned to [`UNCLAIMED_PFN`], not `0` (see that constant's own doc
    /// comment) — a dangling raw pointer or leftover `pfn_to_slot` entry that
    /// reads this slot after it's been evicted and reused reads back an
    /// obviously-bogus pfn instead of silently aliasing page 0's real data.
    pub fn reset_to_unclaimed(&mut self) {
        self.reset_entries_and_bitmaps();
        self.pfn = UNCLAIMED_PFN;
        self.gen = &NEVER_COMPILABLE_GEN;
    }

    /// Reset `requested`/`denied`/`compiled` bitmaps, the compiled `func`
    /// pointer, and every diagnostic counter back to a freshly-claimed
    /// page's defaults — but, unlike [`Self::reset_to_unclaimed`], leaves
    /// `pfn`/`gen` untouched: this page stays claimed under the same
    /// physical frame, it just looks like nothing has ever been compiled for
    /// it yet. `requested`/`denied` are cleared too — this is a deliberate
    /// "forget everything you learned about this page" for
    /// `j2 clear <paddr>`.
    pub fn reset_compiled_state(&mut self) {
        self.reset_entries_and_bitmaps();
    }

    /// Current generation count for this page.    /// Current generation count for this page. `self.gen` is never null (see
    /// its own doc comment) — a page whose backing device has no real gen
    /// tracking (MMIO, etc) reads the shared, never-bumped
    /// [`NEVER_COMPILABLE_GEN`] fallback here instead, so this is
    /// unconditionally safe to call on any claimed or unclaimed page, no
    /// branch needed on the caller's part.
    #[inline]
    pub fn current_gen(&self) -> u64 {
        // Relaxed: publish-time (§13.3) and mutation-time (§7) orderings are
        // established by the fetch_or/re-read pair at those call sites, not here.
        unsafe { (*self.gen).load(Ordering::Relaxed) }
    }

    /// Relaxed read of `entry_gen` — the generation `func`/`compiled` were
    /// last published against. Used both by `j2 pcp` diagnostics and by
    /// `comp.rs`'s pre-compile subsumption check (§13.3 step 4 — comparing
    /// against this is what makes that check gate on "compiled for THIS
    /// generation" rather than blindly re-reading `compiled`'s bits, which
    /// can hold stale-generation leftovers, see `publish`'s own doc
    /// comment). Not part of the dispatch-hot path — `is_runnable` does
    /// its own Acquire load inline rather than calling this.
    #[inline]
    pub fn entry_gen(&self) -> u64 {
        self.entry_gen.load(Ordering::Relaxed)
    }

    /// How many times `publish()` has succeeded since this page's last
    /// `reset_to_unclaimed` — see the field's own
    /// doc comment for why this exists alongside `compile_count` (opposite
    /// lifetime, always present). Not on the hot dispatch path — only read
    /// by `j2 pcp`.
    #[inline]
    pub fn compiles_since_flush(&self) -> u32 {
        self.compiles_since_flush.load(Ordering::Relaxed)
    }

    /// Record one analyze-stage rejected compile attempt for this page (a
    /// candidate excluded at its own entry, or the merged region too short)
    /// — bumps both the stage-specific `analyze_rejected` and the umbrella
    /// `rejected_compiles` total. See both fields' own doc comments. Called
    /// from `comp.rs`'s `prepare_multi_entry_compile`.
    #[inline]
    pub fn mark_analyze_rejected(&self) {
        self.analyze_rejected.fetch_add(1, Ordering::Relaxed);
        self.rejected_compiles.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn analyze_rejected(&self) -> u32 {
        self.analyze_rejected.load(Ordering::Relaxed)
    }

    /// Record one codegen-stage rejected compile attempt for this page
    /// (Cranelift declined: no emitter, or a real verifier rejection) —
    /// bumps both the stage-specific `codegen_rejected` and the umbrella
    /// `rejected_compiles` total. See both fields' own doc comments. Called
    /// from `comp.rs`'s `handle_request`/`handle_request_deferred`'s non-OOM
    /// `None =>` arm.
    #[inline]
    pub fn mark_codegen_rejected(&self) {
        self.codegen_rejected.fetch_add(1, Ordering::Relaxed);
        self.rejected_compiles.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn codegen_rejected(&self) -> u32 {
        self.codegen_rejected.load(Ordering::Relaxed)
    }

    /// Lifetime total of every rejected compile attempt for this page,
    /// across both the analyze and codegen stages — see
    /// `rejected_compiles`'s own field doc comment. Not on the hot dispatch
    /// path — only read by `j2 pcp`.
    #[inline]
    pub fn rejected_compiles(&self) -> u32 {
        self.rejected_compiles.load(Ordering::Relaxed)
    }

    /// Record that a dispatch just won `try_schedule_page()` for this page
    /// — see `schedule_attempts`'s own field doc comment.
    #[cfg(feature = "developer")]
    #[inline]
    pub fn mark_schedule_attempt(&self) {
        self.schedule_attempts.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "developer")]
    #[inline]
    pub fn schedule_attempts(&self) -> u32 {
        self.schedule_attempts.load(Ordering::Relaxed)
    }

    /// Record that a `CompileRequest` for this page was actually handed to
    /// `push_compile_request` — see `sends_attempted`'s own field doc comment.
    #[cfg(feature = "developer")]
    #[inline]
    pub fn mark_send_attempted(&self) {
        self.sends_attempted.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "developer")]
    #[inline]
    pub fn sends_attempted(&self) -> u32 {
        self.sends_attempted.load(Ordering::Relaxed)
    }

    /// Record that a send for this page was dropped because the compile
    /// queue was full — see `sends_dropped_queue_full`'s own field doc
    /// comment.
    #[cfg(feature = "developer")]
    #[inline]
    pub fn mark_send_dropped_queue_full(&self) {
        self.sends_dropped_queue_full.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "developer")]
    #[inline]
    pub fn sends_dropped_queue_full(&self) -> u32 {
        self.sends_dropped_queue_full.load(Ordering::Relaxed)
    }

    /// Record one of `prepare_multi_entry_compile`'s three silent early-bounce
    /// outcomes for this page — see `prepare_bounced`'s own field doc comment.
    #[inline]
    pub fn mark_prepare_bounced(&self) {
        self.prepare_bounced.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn prepare_bounced(&self) -> u32 {
        self.prepare_bounced.load(Ordering::Relaxed)
    }

    /// Record that this page's compile bounced on arena-out-of-memory in
    /// codegen — see `codegen_oom_bounced`'s own field doc comment.
    #[inline]
    pub fn mark_codegen_oom_bounced(&self) {
        self.codegen_oom_bounced.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn codegen_oom_bounced(&self) -> u32 {
        self.codegen_oom_bounced.load(Ordering::Relaxed)
    }

    /// Record that this page's successful compile came back gap-blocked
    /// (unsealed) from `finalize_batch_nonforced` — see `seal_gap_blocked`'s
    /// own field doc comment.
    #[inline]
    pub fn mark_seal_gap_blocked(&self) {
        self.seal_gap_blocked.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn seal_gap_blocked(&self) -> u32 {
        self.seal_gap_blocked.load(Ordering::Relaxed)
    }

    /// Record that this page's `finalize_batch_nonforced` call failed
    /// outright (`Codegen::last_finalize_failed`) rather than merely
    /// gap-blocking — see `finalize_failed`'s own field doc comment.
    #[inline]
    pub fn mark_finalize_failed(&self) {
        self.finalize_failed.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn finalize_failed(&self) -> u32 {
        self.finalize_failed.load(Ordering::Relaxed)
    }

    /// Whether `offset_word` is a live entry point into `func` right now,
    /// without regard to whether `func` is still fresh against the page's
    /// current gen. Callers that need "is this dispatchable right now" want
    /// [`Self::is_runnable`]; this is for the dispatch-trigger gate
    /// (§13.2), which probes coverage first and only then decides
    /// exec-vs-recompile from gen.
    #[inline]
    pub fn is_published(&self, offset_word: usize) -> bool {
        bitmap_test(&self.compiled, offset_word, Ordering::Acquire)
    }

    /// Whether `offset_word` is both a live entry (`compiled` bit set) and
    /// `func` is still fresh (the page's `entry_gen` matches its current
    /// gen, §13.3). A set compiled bit whose gen has drifted is stale — the
    /// caller should treat it as unpublished (downgrade to interpreter,
    /// §13.2) rather than dispatch it.
    ///
    /// The `entry_gen` load is Acquire, not Relaxed: it's the real
    /// synchronization point for `func` (see `publish`'s doc comment) — the
    /// `compiled` bit alone doesn't provide fresh ordering across a recompile
    /// that only adds coverage without moving `entry_gen`, since the bit, once
    /// set, doesn't change value on that path either. Callers must not read
    /// `func` after this returns `true` without going through this same
    /// `entry_gen` load (i.e. don't cache `is_published`'s result and reuse
    /// it — always call `is_runnable` fresh right before trusting `func`),
    /// or the ordering guarantee is lost.
    #[inline]
    pub fn is_runnable(&self, offset_word: usize) -> bool {
        self.is_published(offset_word)
            && self.entry_gen.load(Ordering::Acquire) == self.current_gen()
    }

    /// The page's compiled function, or null if unpublished. Callers must
    /// check [`Self::is_runnable`] first (in that order — see its doc
    /// comment on why the `entry_gen` Acquire load must happen-before this
    /// read) before trusting the returned pointer.
    #[inline]
    pub fn func(&self) -> *const () {
        self.func.load(Ordering::Relaxed) as *const ()
    }

    /// Whether `offset_word` has been sticky-rejected by the compiler for the
    /// page's current generation (§13.6). `denied` is inverted-sense: a clear
    /// bit means denied.
    #[inline]
    pub fn is_denylisted(&self, offset_word: usize) -> bool {
        !bitmap_test(&self.denied, offset_word, Ordering::Relaxed)
    }

    /// Sticky-reject `offset_word` for the page's current generation (§13.6).
    #[inline]
    pub fn denylist(&self, offset_word: usize) {
        // `denied` is inverted (1 = allowed) — "deny" clears the bit, so this
        // can't be a plain fetch_or the way the old ENTRY_DENYLISTED flag was;
        // AND with everything except this bit instead.
        self.denied[offset_word >> 6].fetch_and(!(1u64 << (offset_word & 63)), Ordering::Relaxed);
    }

    /// Sticky-reject every offset on this page at once. Not on any hot path
    /// (a page-wide, one-time operation) — used to make a whole page
    /// permanently non-dispatchable, e.g. tests that need a deterministic
    /// "this page never compiles/dispatches, everything runs on the
    /// interpreter" guarantee.
    pub fn denylist_all(&self) {
        for word in self.denied.iter() { word.store(0, Ordering::Relaxed); }
    }

    /// Clear `offset_word`'s `compiled` bit only — NOT `denied`. Two
    /// distinct callers:
    ///
    /// - `emit_fr_mode_guard`'s FR-mismatch case (`jit_kill_entry`): the
    ///   artifact itself isn't wrong in general, just stale for *this*
    ///   dispatch (compiled for the wrong FR mode). A later dispatch is
    ///   expected and welcome to recompile — most likely for the *other* FR
    ///   mode — so `denied` is left alone.
    /// - `comp.rs`'s `prepare_multi_entry_compile`, paired with `denylist`:
    ///   a re-walk that excludes a previously-`compiled` offset (§13's
    ///   one-function-per-page model folds prior coverage into every
    ///   compile's candidates) leaves that offset with no case in the fresh
    ///   `func`'s dispatch switch at all — `compiled`'s bit must not survive
    ///   or `is_runnable` keeps reporting it dispatchable into a function
    ///   that doesn't recognize it, turning every future hit into a silent
    ///   JIT-entry-then-EXEC_FALLBACK round trip forever. Here `denied` is
    ///   *not* left alone (the paired `denylist` call sticky-rejects it, on
    ///   the theory that a walk which just excluded it once is likely to
    ///   again) — the two calls are deliberately paired at that call site.
    ///
    /// `func` itself is deliberately left in place either way; nothing
    /// between clearing this bit and the next `publish` ever reads `func`
    /// without first re-checking `is_runnable`.
    #[inline]
    pub fn kill(&self, offset_word: usize) {
        self.compiled[offset_word >> 6].fetch_and(!(1u64 << (offset_word & 63)), Ordering::Release);
        // Block the churn-avoidance skip until a real compile re-derives this
        // page's coverage — see `killed_since_snapshot`.
        self.killed_since_snapshot.store(true, Ordering::Release);
    }

    /// Mark `offset_word` as a requested entry point (§13.2) — unconditional,
    /// not a test-and-set: unlike the old per-offset `ENTRY_SCHEDULED` flag,
    /// `requested` bits are meant to accumulate and stay set across however
    /// many compile passes it takes to eventually get covered (§13.3's
    /// coalescing), not be claimed by a single "first caller" and cleared
    /// per-offset once decided. Idempotent — setting an already-set bit is a
    /// no-op.
    #[inline]
    pub fn mark_requested(&self, offset_word: usize) {
        bitmap_set(&self.requested, offset_word, Ordering::Relaxed);
    }

    /// Whether `offset_word` currently has a live `requested` bit — i.e. some
    /// dispatch has asked for it and no compile has covered/denylisted it
    /// since. Diagnostic use (`j2 pcp`); the compile-request path reads the
    /// whole bitmap at once via `snapshot_compile_candidates` instead.
    #[inline]
    pub fn is_requested(&self, offset_word: usize) -> bool {
        bitmap_test(&self.requested, offset_word, Ordering::Relaxed)
    }

    /// Test-and-set the page-level "a `CompileRequest` is already in flight
    /// for this page" flag: returns `true` (and sets it) only if it was
    /// previously clear, i.e. only the first caller should actually build and
    /// send a `CompileRequest` for this page right now — every other
    /// concurrent/subsequent caller sees `false` and skips it, since whatever
    /// gets dequeued will snapshot `requested` fresh and pick up every bit
    /// accumulated by then anyway (§13.2/§13.3), not just the one that
    /// triggered this particular call. Distinct from `mark_requested`: this
    /// gate is purely about not flooding the queue with redundant page-level
    /// sends, not about tracking which offsets are wanted.
    #[inline]
    pub fn try_schedule_page(&self) -> bool {
        !self.page_scheduled.swap(true, Ordering::Relaxed)
    }

    /// Read-only peek at the page-level in-flight flag — diagnostic use
    /// (`j2 pcp`) only. Never use this to decide whether to send a
    /// `CompileRequest`: that's `try_schedule_page`'s test-and-set job, and a
    /// plain load-then-branch here would race it (two callers could both
    /// observe `false` and both send).
    #[inline]
    pub fn is_scheduled(&self) -> bool {
        self.page_scheduled.load(Ordering::Relaxed)
    }

    /// Clear the page-level in-flight flag — called once `handle_request`
    /// (`comp.rs`) has decided this page's request one way or another (some
    /// coverage published, or nothing new to offer), so a later legitimate
    /// re-request (e.g. a fresh discovery that arrived after this compile's
    /// snapshot was already taken) isn't permanently blocked by a stale flag
    /// from a request that already finished.
    #[inline]
    pub fn clear_scheduled(&self) {
        self.page_scheduled.store(false, Ordering::Relaxed);
    }

    /// Clear every `requested` bit this compile's snapshot covered, after a
    /// successful publish — so a later re-request for the same offset (e.g.
    /// after a future gen bump) isn't shadowed by a stale bit this compile
    /// already acted on. Bits set by a *newer* discovery (arrived after this
    /// compile's snapshot, not part of `covered`) are left untouched — they
    /// still need a future compile to pick them up.
    #[inline]
    pub fn clear_requested_bits(&self, covered: &[u64; BITMAP_WORDS]) {
        for i in 0..BITMAP_WORDS {
            if covered[i] != 0 {
                self.requested[i].fetch_and(!covered[i], Ordering::Relaxed);
            }
        }
    }

    /// Snapshot the `requested` bitmap (§13.3 step 3).
    #[inline]
    pub fn snapshot_requested(&self) -> [u64; BITMAP_WORDS] {
        bitmap_snapshot(&self.requested)
    }

    /// Snapshot the `compiled` bitmap — every offset that's a live entry
    /// point into `func` right now (before any `entry_gen` staleness check;
    /// see `is_runnable`). Diagnostic use only (`j2 dump-pcp`) — the hot
    /// dispatch path never needs the whole bitmap at once, only
    /// `is_published`'s single-bit test, and `comp.rs`'s own use of
    /// `compiled` alongside `requested` goes through
    /// [`Self::snapshot_compile_candidates`] instead.
    #[inline]
    pub fn snapshot_compiled(&self) -> [u64; BITMAP_WORDS] {
        bitmap_snapshot(&self.compiled)
    }

    /// Snapshot the `denied` bitmap, in its raw (inverted: 1 = allowed)
    /// sense — see the field's own doc comment. Diagnostic use only.
    #[inline]
    pub fn snapshot_denied_raw(&self) -> [u64; BITMAP_WORDS] {
        bitmap_snapshot(&self.denied)
    }

    /// `comp.rs`'s `prepare_multi_entry_compile` candidate set, computed in
    /// one pass: `(requested | compiled) & denied` per word (`denied`'s raw,
    /// inverted sense — a set bit there means "still eligible", so ANDing
    /// with it is exactly "not denylisted").
    ///
    /// `include_compiled`: whether to fold `compiled` in at all — the caller
    /// must only pass `true` when `entry_gen == gen_snap == current_gen()`
    /// (a `compiled` bit from a superseded generation doesn't describe the
    /// bytes this compile is about to snapshot; `publish`'s own
    /// `is_real_invalidation` path already resets the bitmap for that case,
    /// so this method doesn't re-check gen itself — it has no `gen_snap` of
    /// its own to compare against).
    ///
    /// Folding `compiled` in (when eligible) is what makes every recompile
    /// re-cover previously-published entries, not just this round's fresh
    /// `requested` bits: §13's one-function-per-page model replaces `func`
    /// wholesale on every compile, and its dispatch switch only recognizes
    /// the entries this exact walk covers (codegen.rs's `compile_region`
    /// builds the switch from nothing else) — an old entry left out of a
    /// merged walk has no case in the new `func` at all, not just stale
    /// coverage, so every future dispatch into it would silently fall to
    /// EXEC_FALLBACK forever.
    #[inline]
    pub fn snapshot_compile_candidates(&self, include_compiled: bool) -> [u64; BITMAP_WORDS] {
        let mut out = [0u64; BITMAP_WORDS];
        for i in 0..BITMAP_WORDS {
            let mut bits = self.requested[i].load(Ordering::Relaxed);
            if include_compiled {
                bits |= self.compiled[i].load(Ordering::Acquire);
            }
            out[i] = bits & self.denied[i].load(Ordering::Relaxed);
        }
        out
    }

    /// Whether every bit set in `bits` is already covered by `compiled`
    /// (§13.3 steps 4/6's subsumption check).
    #[inline]
    pub fn requested_subsumed_by_compiled(&self, bits: &[u64; BITMAP_WORDS]) -> bool {
        bitmap_is_subset_of(bits, &self.compiled)
    }

    /// Publish a freshly compiled function covering `new_entries` (§13.3 step
    /// 6): re-check page gen and subsumption, then union `new_entries` into
    /// `compiled`, install `func`, and write `entry_gen` — all under
    /// `publish_lock` so the three-field update happens as one step relative
    /// to any concurrent publisher for this same page (§13.5). Returns `true`
    /// iff this call actually published (i.e. wasn't stale/subsumed).
    ///
    /// `snap_gen` is the page generation the compile snapshot was confirmed
    /// against (§13.3 step 2's seqlock). Two independent reasons abort:
    ///
    /// - `current_gen() > snap_gen`: the page mutated after this compile's
    ///   snapshot was taken — the compiled bytes are stale regardless of
    ///   entry coverage.
    /// - `new_entries ⊆ compiled` (current, re-read under the lock): some
    ///   other compile already published a superset of what this compile
    ///   covers — redundant.
    ///
    /// `func`-then-`compiled`-then-`entry_gen` write order (all Release) means
    /// a reader that observes a `compiled` bit set (Acquire-paired via
    /// `is_runnable`'s `entry_gen` load) never observes a stale/null
    /// `func` for it. `entry_gen` is written unconditionally on every actual
    /// publish, but only **numerically advances** when `snap_gen >
    /// entry_gen` (a real page-bump-triggered recompile, §13.6/§13.3's
    /// discussion) — a pure entry-coverage publish against unchanged bytes
    /// (`snap_gen == entry_gen` already) rewrites the same value. Readers
    /// never need to distinguish the two: `entry_gen == current_gen()` is
    /// the only fact `is_runnable` needs.
    ///
    /// When `snap_gen` genuinely advances `entry_gen`, `denied` resets to
    /// all-allowed (§13.6): a page-gen bump means the bytes are provably
    /// different from whatever got denied last time.
    pub fn publish(
        &self,
        new_entries: &[u64; BITMAP_WORDS],
        func: *const (),
        snap_gen: u64,
        instr_count: usize,
        code_size: u32,
    ) -> bool {
        let _guard = self.publish_lock.lock();

        if self.current_gen() > snap_gen {
            return false; // page mutated past this compile's snapshot; discard rather than publish stale code
        }
        // The subsumption check only means anything when comparing against
        // `compiled` bits published for the SAME generation this compile
        // targets: a bit set under an older, now-superseded generation
        // doesn't actually cover this offset for the bytes this compile
        // just analyzed — it's leftover coverage from before the page
        // changed, about to be wiped by the entry_gen advance below. Only
        // skip-as-redundant when entry_gen already matches snap_gen (a pure
        // entry-coverage publish racing another one for the same bytes).
        if self.entry_gen.load(Ordering::Relaxed) == snap_gen
            && bitmap_is_subset_of(new_entries, &self.compiled)
        {
            return false; // some other compile already published everything this one covers, for this same generation
        }

        // Safety: `func` is a raw pointer write behind `&self` — sound
        // because no concurrent reader trusts it without first Acquire-
        // loading `entry_gen` (via `is_runnable`) and observing it equal
        // to current_gen(); `publish_lock` excludes any concurrent writer of
        // these same three fields.
        self.func.store(func as *mut (), Ordering::Release);
        self.compiles_since_flush.fetch_add(1, Ordering::Relaxed);
        self.instr_count.store(instr_count as u32, Ordering::Relaxed);
        self.code_size.store(code_size, Ordering::Relaxed);
        #[cfg(feature = "developer")]
        {
            self.compile_count.fetch_add(1, Ordering::Relaxed);
        }

        let prev_entry_gen = self.entry_gen.load(Ordering::Relaxed);
        let is_real_invalidation = snap_gen > prev_entry_gen;
        if is_real_invalidation {
            // The old `compiled` bits refer to a now-superseded generation —
            // offsets compiled against bytes that no longer exist. Replace,
            // don't union: this compile's `new_entries` becomes the entire
            // coverage set for the new generation, not an addition to stale
            // leftovers (which would let a reader dispatch into `func` for
            // an offset this fresh compile never actually analyzed).
            for i in 0..BITMAP_WORDS {
                self.compiled[i].store(new_entries[i], Ordering::Release);
            }
        } else {
            bitmap_union_from(&self.compiled, new_entries, Ordering::Release);
        }

        self.entry_gen.store(snap_gen, Ordering::Release);
        if is_real_invalidation {
            // Real invalidation, not just added coverage: the bytes changed,
            // so whatever got sticky-denied against the OLD bytes deserves a
            // fresh chance against the new ones (§13.6).
            for word in self.denied.iter() { word.store(u64::MAX, Ordering::Relaxed); }
            // `fr1` has exactly the same justification and used to be the one
            // field left behind here. A physical page freed by an o32 (FR0)
            // process and reallocated to an n32 (FR1) one arrived with new
            // bytes but kept the previous tenant's pin, so the next compile
            // specialized fresh code for the wrong FR mode — and because
            // `emit_fr_mode_guard` suppresses itself when CU1 is clear, that
            // was not reliably caught. Don't pin a mode here (the compiling
            // thread's notion of "live" is the one that matters, and it is
            // read at compile-request time); just drop the stale claim so the
            // next compile re-derives it. See `fr1`'s own doc comment.
            self.fr_unpinned.store(true, Ordering::Relaxed);
        }
        true
    }

    /// Record what a compile looked at, **provisionally** — the record is
    /// stored but left invalid, so it cannot authorize a skip until
    /// [`Self::commit_compile_snapshot`] vouches for it with the publish
    /// that actually installed the code.
    ///
    /// Called right after codegen succeeds, before the publish (which, on
    /// the deferred path, may happen later and on another thread). `used`
    /// must be every word the walk decoded, including delay slots; `entries`
    /// exactly the bitmap that will be handed to `publish`.
    ///
    /// Overwrites any earlier staged-but-uncommitted record — which can only
    /// be one left behind by a compile whose publish never landed (it was
    /// stale, subsumed, or the page was reset under it), since this page
    /// cannot be compiled twice at once. `staged_gen` is what lets
    /// [`Self::commit_compile_snapshot`] tell its own record from such a
    /// leftover.
    pub fn stage_compile_snapshot(
        &self,
        words: &[u32; ENTRIES_PER_PAGE],
        used: &[u64; BITMAP_WORDS],
        entries: &[u64; BITMAP_WORDS],
        gen_snap: u64,
        fr1: bool,
    ) {
        // Safety: see `compile_snapshot`'s field doc — only a compiling
        // thread reaches here, and a page is never compiled twice at once.
        let snap = unsafe { &mut *self.compile_snapshot.get() };
        // This compile re-derived coverage from scratch, so any kill that
        // predates it is already accounted for in what it published.
        self.killed_since_snapshot.store(false, Ordering::Release);
        snap.valid = false;
        snap.staged_gen = gen_snap;
        snap.words.copy_from_slice(words);
        snap.used.copy_from_slice(used);
        snap.entries.copy_from_slice(entries);
        snap.fr1 = fr1;
    }

    /// Promote a staged record to usable, now that the publish it describes
    /// has actually installed its code.
    ///
    /// `gen_snap` and `entries` identify the publish doing the vouching; the
    /// record is only committed if they match what was staged. The mismatch
    /// case that matters is a reset landing between stage and publish
    /// (`invalidate_compile_snapshot` poisons `staged_gen` precisely so this
    /// cannot re-validate a snapshot for a function the reset discarded) —
    /// the record is dropped rather than committed, costing a future
    /// recompile instead of risking one that skips against the wrong bytes.
    pub fn commit_compile_snapshot(&self, gen_snap: u64, entries: &[u64; BITMAP_WORDS]) {
        // Safety: as `stage_compile_snapshot`.
        let snap = unsafe { &mut *self.compile_snapshot.get() };
        if snap.staged_gen == gen_snap && snap.entries == *entries {
            snap.valid = true;
        } else {
            snap.valid = false;
        }
    }

    /// Whether this page has a recorded compile snapshot to compare a new
    /// request against. `false` on a page that has never compiled, or whose
    /// snapshot was invalidated by a reset — those requests are not
    /// candidates for a skip at all, and the caller counts neither outcome
    /// for them (see [`Self::redundant_skipped`]).
    #[inline]
    pub fn has_compile_snapshot(&self) -> bool {
        // Safety: as `stage_compile_snapshot`.
        unsafe { (*self.compile_snapshot.get()).valid }
    }

    /// Count one compile request answered by the snapshot — see
    /// [`Self::redundant_skipped`].
    #[inline]
    pub fn mark_redundant_compile_skipped(&self) {
        self.redundant_skipped.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one compile request that had a snapshot but still had to
    /// compile — see [`Self::redundant_rejected`].
    #[inline]
    pub fn mark_redundant_compile_rejected(&self) {
        self.redundant_rejected.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn redundant_skipped(&self) -> u32 {
        self.redundant_skipped.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn redundant_rejected(&self) -> u32 {
        self.redundant_rejected.load(Ordering::Relaxed)
    }

    /// Drop the recorded compile snapshot — the `func` it describes is gone
    /// or no longer trustworthy, so it must not authorize a skip.
    ///
    /// Every reset path that clears `func` calls this. There is no path that
    /// legitimately keeps a snapshot past its function: a skip works by
    /// re-validating the *installed* code, so a stale snapshot would
    /// advance `entry_gen` over a null or superseded `func` and send
    /// dispatch into nothing.
    #[inline]
    pub fn invalidate_compile_snapshot(&self) {
        // Safety: the reset paths that call this run on the exec thread with
        // no compile in flight for this page (see the field doc).
        let snap = unsafe { &mut *self.compile_snapshot.get() };
        snap.valid = false;
        // Poison the staging slot too, not just `valid`: a compile that was
        // in flight across this reset would otherwise come back and commit
        // its record, re-validating a snapshot for a `func` this reset just
        // threw away. `u64::MAX` can never be a real `gen_snap` (the counter
        // starts at 0 and only increments), so no commit can match it.
        snap.staged_gen = u64::MAX;
    }

    /// Churn-avoidance fast path (§13.3 step 2.5): decide whether the compile
    /// this request is asking for would reproduce, byte for byte, the code
    /// already installed — and if so, re-validate that code against the new
    /// generation and skip the compile entirely.
    ///
    /// The caller has already taken its seqlock byte snapshot (`words`,
    /// coherent with `gen_snap`) and computed the entry points it wants
    /// (`wanted_entries`). Three things must all hold to skip:
    ///
    /// 1. **The wanted entries are a subset of what's published.** The
    ///    installed `func`'s dispatch switch only has cases for the entries
    ///    its own compile covered, so an entry outside that set cannot be
    ///    served no matter how identical the bytes are.
    /// 2. **Every word the recorded compile decoded is unchanged.** Words
    ///    outside `used` are irrelevant by construction — the walk never
    ///    read them, so they had no influence on the emitted code. This is
    ///    what makes the check pay off on a page whose generation bumps
    ///    because the guest wrote *data* sharing the page with code.
    /// 3. **The FR mode matches.** See [`CompileSnapshot::fr1`].
    ///
    /// On success `entry_gen` is advanced to `gen_snap` under
    /// `publish_lock`, which is exactly what makes `is_runnable` start
    /// returning true again — the same single-field write `publish` would
    /// have performed, minus the `func`/`compiled` stores (unchanged) and
    /// minus the `denied` reset, which would be actively wrong here: the
    /// bytes are provably identical, so an offset denied against them is
    /// still correctly denied, and clearing it would re-walk and re-reject
    /// it on every generation bump — the very churn this exists to remove.
    ///
    /// Returns `true` iff the compile was skipped.
    pub fn try_skip_redundant_compile(
        &self,
        words: &[u32; ENTRIES_PER_PAGE],
        wanted_entries: &[u64; BITMAP_WORDS],
        gen_snap: u64,
        fr1: bool,
    ) -> bool {
        // Safety: as `stage_compile_snapshot`.
        let snap = unsafe { &*self.compile_snapshot.get() };
        if !snap.valid || snap.fr1 != fr1 {
            return false;
        }
        // A published `func` is what the snapshot describes; without one
        // there is nothing to re-validate. (Belt and braces — every path
        // that nulls `func` also invalidates the snapshot.)
        if self.func.load(Ordering::Relaxed).is_null() {
            return false;
        }
        if !bitmap_is_subset_of_plain(wanted_entries, &snap.entries) {
            return false;
        }
        for i in 0..BITMAP_WORDS {
            let mut bits = snap.used[i];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let w = i * 64 + bit;
                if words[w] != snap.words[w] {
                    return false;
                }
            }
        }
        // Re-validate the installed code against the new generation. Taken
        // under `publish_lock` for the same reason `publish` does: this page
        // is never compiled twice at once, but a dispatch-side `kill` can
        // still land concurrently, and the gen write must not interleave with
        // the coverage check below.
        let _guard = self.publish_lock.lock();
        // Re-check under the lock, exactly as `publish` does — a mutation
        // may have landed between the byte snapshot and here, in which case
        // the bytes we just compared are already history.
        if self.current_gen() > gen_snap {
            return false;
        }
        // Never move `entry_gen` backwards: a concurrent publish may have
        // already installed newer coverage at a higher generation, and
        // rewinding it would invalidate that publish's own code.
        if self.entry_gen.load(Ordering::Relaxed) > gen_snap {
            return false;
        }
        // Deliberately NOT `wanted_entries ⊆ compiled`. That is the wrong
        // question: `compiled` is what is currently *advertised*, which a
        // previous skip may legitimately have narrowed (see the coverage
        // write below), whereas what matters is whether the installed `func`
        // has a case for these offsets — `snap.entries`, checked above, which
        // does not change while that same `func` stays installed. Requiring
        // the stricter condition refuses to re-cover an entry an earlier skip
        // dropped even though the very same function can still serve it,
        // forcing a pointless recompile: a second churn source with the same
        // "skips a lot, runs slower" signature as the first.
        //
        // What DOES have to bail is a `kill` — it means "stop dispatching
        // into `func` here", so a skip must never re-advertise a killed
        // offset. `killed_since_snapshot` is the exact condition;
        // `compiled` cannot express it, because a narrowing skip clears bits
        // there for an unrelated reason. In production every `kill` is
        // already paired with either an FR repin (caught by `snap.fr1 !=
        // fr1`) or a `denylist` (caught by `denied`, which the caller folds
        // into `wanted_entries`), so this is defence in depth rather than the
        // only guard — but it is the one that states the actual invariant.
        if self.killed_since_snapshot.load(Ordering::Relaxed) {
            return false;
        }

        // Leave exactly the coverage a real `publish` of this same candidate
        // set would have left, because that is what the installed `func` can
        // actually serve.
        //
        // This is NOT a formality — getting it wrong is a throughput
        // regression that masquerades as the optimization working. After a
        // gen bump the caller's candidate set is `requested & denied`
        // (`snapshot_compile_candidates(same_gen=false)` deliberately drops
        // `compiled`), and `publish` treats `snap_gen > entry_gen` as a real
        // invalidation and REPLACES `compiled` wholesale with that set. A
        // skip that merely bumped `entry_gen` would leave the older, wider
        // bitmap published: `is_runnable` then returns true for offsets this
        // check never verified, dispatch calls `func` for one, the switch
        // `compile_region` built has no case for it, and it returns
        // EXEC_FALLBACK — on every arrival, forever, each one re-requesting a
        // compile. Measured as high skip counts with *lower* throughput,
        // worst on R4K.
        //
        // Mirrors `publish`'s own is_real_invalidation branch, and for the
        // same reason: replace on a generation advance (old bits describe
        // bytes that no longer exist), union when the generation is
        // unchanged (pure added coverage against the same bytes).
        if gen_snap > self.entry_gen.load(Ordering::Relaxed) {
            for i in 0..BITMAP_WORDS {
                let dropped = self.compiled[i].swap(wanted_entries[i], Ordering::Release)
                    & !wanted_entries[i];
                // Entries this narrowing drops are still legitimately wanted:
                // unlike `publish`'s equivalent branch, the bytes here are
                // provably UNCHANGED, so an offset that was worth compiling a
                // moment ago is worth compiling still — it just is not in
                // this round's candidate set (it was covered, so nothing
                // re-`requested` it, and `snapshot_compile_candidates` drops
                // `compiled` once the generation moves). Folding it back into
                // `requested` is what gives it a path to a real recompile;
                // without it the entry is silently stranded on the
                // interpreter until something happens to fetch it again,
                // which is the same "no automatic path back" failure a
                // dropped-but-still-wanted entry always risks, guarded
                // against the same way.
                if dropped != 0 { self.requested[i].fetch_or(dropped, Ordering::Relaxed); }
            }
        } else {
            bitmap_union_from(&self.compiled, wanted_entries, Ordering::Release);
        }
        self.entry_gen.store(gen_snap, Ordering::Release);
        true
    }

    // ---- `pcp_dump` capture accessors (`j2 dumppcp` / `j2 corpus`) ----
    //
    // Thin forwarders to this implementation's real fields. They exist so
    // `PcpDump::capture` can be written once against a single accessor set
    // and work under both `comp.rs` implementations — the default one has no
    // `requested`/`compiled`/`denied` bitmaps and no page-level `entry_gen`,
    // and reconstructs all of it from its per-`JitEntry` state behind these
    // same names (see its own copy for what each reconstruction assumes).
    pub fn dump_requested(&self) -> [u64; BITMAP_WORDS] { self.snapshot_requested() }
    pub fn dump_compiled(&self) -> [u64; BITMAP_WORDS] { self.snapshot_compiled() }
    pub fn dump_denied_raw(&self) -> [u64; BITMAP_WORDS] { self.snapshot_denied_raw() }
    pub fn dump_entry_gen(&self) -> u64 { self.entry_gen() }
    pub fn dump_fr1(&self) -> bool { self.is_fr1() }

    /// Dispatches into this page's compiled code. Page-level here (§13's one
    /// function per page); 0 when the counter is compiled out, which is the
    /// dump format's documented "unknown".
    pub fn dump_call_count(&self) -> u64 {
        #[cfg(feature = "developer")]
        { self.call_count.load(Ordering::Relaxed) }
        #[cfg(not(feature = "developer"))]
        { 0 }
    }
}

/// Slot index into [`Jitv2::pages`].
pub type PageSlot = u32;

/// Sentinel for [`PhysicalCodePage::prev`]/[`PhysicalCodePage::next`]: "no
/// link" (list head's `prev`, list tail's `next`, or a not-yet-linked slot).
/// `pages.len()` can never reach `u32::MAX` in practice (would be 16TB+ of
/// `PhysicalCodePage`s), so this can't collide with a real slot index.
const NO_SLOT: u32 = u32::MAX;
const _: () = assert!(NO_SLOT == PFN_MAP_EMPTY);

/// JIT v2 engine state embedded in the mips executor.
///
/// Owns the [`PhysicalCodePage`] pool (§2.4): a single array, allocated once
/// at full `capacity` in `Jitv2::new` (every slot pre-built as unclaimed —
/// see `PhysicalCodePage::new`'s doc comment), never resized or reallocated
/// afterward. This is stronger than the old `Vec::push`-based growth: that
/// also never reallocated once capacity was reserved, but each `push` still
/// moved a freshly-constructed `PhysicalCodePage` value into place — with
/// `entries` now inline (not boxed), that move would copy the whole
/// 1024-entry table. Claiming a pre-existing slot in place avoids that
/// entirely: nothing is ever copied after `Jitv2::new` builds the array.
///
/// Slots are handed out via [`Self::free_head`], a singly-linked free list
/// threaded through every unclaimed slot's `PhysicalCodePage::next`. A slot
/// is claimed iff its `pfn != UNCLAIMED_PFN` (see
/// [`PhysicalCodePage::is_claimed`]) — there is no separate recency list:
/// `mega_flush` resets the whole pool to empty and rebuilds the free list
/// from slot 0, so after a flush "next unclaimed slot" is once again the
/// whole array. (An earlier design preserved the most-recently-used slots
/// across a flush, threaded on an MRU list; that was removed — preserved
/// entry points could be stale, and keeping them was at best neutral.)
///
/// Lookup from `pfn` to pool slot goes through `pfn_to_slot` — the design
/// doc's dense pfn-indexed alternative (§2.4), now built: page-switch lookup
/// *did* show up hot in profiling (~4.5% of a real IRIX boot, with the
/// HashMap's own `RawTable` grow-and-rehash allocation directly above it),
/// which is exactly the trigger this comment used to name. See [`PfnMap`].
pub struct Jitv2 {
    /// The full-capacity page pool, allocated once — see this struct's own
    /// doc comment. Indices are stable for the pool's entire lifetime,
    /// including across `mega_flush` (slots are reset/relinked in place, the
    /// array itself never shrinks or reallocates), so pointers into it —
    /// e.g. the executor's current-PCP pointer, `CompileRequest::page` —
    /// stay valid for the process's whole lifetime, not just "until the next
    /// flush" as before. A `CompileRequest`/PCP pointer surviving a flush it
    /// raced against and landing back in a slot that's since been reclaimed
    /// for a different physical page is still a real hazard (unchanged from
    /// before — `worker_loop`'s `drain_pending` and the various
    /// gen-mismatch/`is_runnable` checks are what actually guard against
    /// stale content, not pointer validity) — array-lifetime stability alone
    /// doesn't imply content freshness. Every request still in flight at
    /// flush time is drained before any slot is reclaimed (`mega_flush`'s own
    /// doc comment), so an evicted slot handed to a different pfn is never
    /// racing a stale in-flight reference to begin with.
    pages: Box<[PhysicalCodePage]>,
    /// Head of the free-slot singly-linked list, threaded through
    /// `PhysicalCodePage::next`. `NO_SLOT` when the pool is fully claimed.
    /// Every slot starts here (`Jitv2::new` links the whole array into this
    /// list up front); `page_for` pops from the head on a fresh claim,
    /// `mega_flush` resets the pool and rebuilds the free list.
    free_head: u32,
    /// pfn -> index into `pages`. Consulted only on a page switch (fetch
    /// lands on a different PFN than the currently-tracked one) — not on
    /// every fetch. Direct-mapped, not hashed — see [`PfnMap`].
    pfn_to_slot: PfnMap,
    /// Pool capacity, fixed at construction (== `pages.len()`). Claiming past
    /// this (i.e. with the free list exhausted) triggers `mega_flush` (the
    /// "ran out of PCPs" resource-exhaustion trigger).
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

    /// L1-D geometry for the inline load/store fast path, published once by
    /// the CPU thread at startup (`MipsExecutor::publish_jit_dc_geometry`)
    /// and copied into `Codegen` when the compile worker takes it. Held here
    /// rather than written straight into `Codegen` because the worker owns
    /// that by value while compiling.
    pub dc_geometry: Mutex<crate::mips_cache_v2::JitDcGeometry>,
    /// Compile-time constants for the one core these workers compile for —
    /// see `codegen::JitConsts`. Same publication route as `dc_geometry`.
    pub jit_consts: Mutex<crate::jitv2::codegen::JitConsts>,
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
/// Not `developer`-gated: `code_size_by_instr_count` is the
/// emitted-code-size metric and has to be readable in a production-shaped
/// build — see `PhysicalCodePage::instr_count`'s field doc for why measuring
/// it under `developer` gives a ~4x wrong answer.
#[derive(Debug, Clone, Copy)]
pub struct CodeSizeBucket {
    pub count: u32,
    pub sum_bytes: u64,
    pub min_bytes: u32,
    pub max_bytes: u32,
}

impl Jitv2 {
    /// Allocate the full page pool up front, `capacity` slots, every one
    /// starting unclaimed (`PhysicalCodePage::new(UNCLAIMED_PFN, null)`) —
    /// see `Jitv2`'s own doc comment for why this is a single one-shot
    /// allocation rather than lazy `push`-based growth. Sizing is a Phase 0
    /// measurement per the design doc (§9); start with whatever the caller
    /// passes and let `mega_flush` absorb a too-small guess rather than
    /// trying to size this perfectly up front — getting it wrong now just
    /// means an earlier flush, not a correctness problem. Does not start the
    /// compile thread — call `compile_queue.start()`.
    pub fn new(capacity: usize) -> Self {
        let mut pages: Box<[PhysicalCodePage]> = (0..capacity).map(|_| PhysicalCodePage::new(UNCLAIMED_PFN, std::ptr::null())).collect();
        // Thread every slot onto the free list up front, index order —
        // `page_for`'s claim path only ever pops from the head, so initial
        // order just decides which physical slot backs the first `capacity`
        // claims, which doesn't matter.
        for i in 0..capacity {
            pages[i].next = if i + 1 < capacity { (i + 1) as u32 } else { NO_SLOT };
        }
        Self {
            pages,
            free_head: if capacity > 0 { 0 } else { NO_SLOT },
            pfn_to_slot: PfnMap::new(),
            capacity,
            compile_queue: CompileQueue::new(),
            codegen: Mutex::new(Some(crate::jitv2::codegen::Codegen::new())),
            dc_geometry: Mutex::new(crate::mips_cache_v2::JitDcGeometry::unsupported()),
            jit_consts: Mutex::new(crate::jitv2::codegen::JitConsts::default()),
            stats: Arc::new(JitStats::default()),
        }
    }

    /// Release `slot` back to the free list.
    fn free_page(&mut self, slot: PageSlot) {
        let pfn = self.pages[slot as usize].pfn;
        self.pfn_to_slot.remove(pfn);
        self.pages[slot as usize].reset_to_unclaimed();
        self.pages[slot as usize].next = self.free_head;
        self.free_head = slot;
    }

    /// Look up the pool slot for `pfn`, claiming the next unclaimed slot
    /// in place (`PhysicalCodePage::claim`, `gen_ptr(phys_addr)` on the bus)
    /// if this is the first arrival at this page. Returns `None` if the pool
    /// is exhausted — the caller (mips exec thread) is responsible for
    /// running `mega_flush` and retrying.
    pub fn page_for(&mut self, pfn: Pfn, phys_addr: u32, bus: &dyn BusDevice, fr1: bool) -> Option<PageSlot> {
        if let Some(slot) = self.pfn_to_slot.get(pfn) {
            debug_assert_eq!(self.pages[slot as usize].pfn, pfn,
                "pfn_to_slot[{:#x}] -> slot {} whose own pfn is {:#x} — the map and the slot it points at have                  desynced (a slot was reused/evicted without this map entry being updated to match)",
                pfn, slot, self.pages[slot as usize].pfn);
            return Some(slot);
        }
        if self.free_head == NO_SLOT {
            return None;
        }
        let slot = self.free_head;
        self.free_head = self.pages[slot as usize].next;
        let gen = bus.gen_ptr(phys_addr);
        self.pages[slot as usize].claim(pfn, gen, fr1);
        self.pages[slot as usize].next = NO_SLOT;
        self.pfn_to_slot.insert(pfn, slot);
        Some(slot)
    }

    /// Pointers for lock-free `pfn -> *mut PhysicalCodePage` fast lookup.
    pub fn fast_lookup_ptrs(&mut self) -> (*const u32, *mut PhysicalCodePage) {
        (self.pfn_to_slot.as_ptr(), self.pages.as_mut_ptr())
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
    ///
    /// Scans `pages` rather than consulting the pfn map: [`PfnMap`] is a
    /// direct-mapped array with no live-entry count of its own, while
    /// `pfn != UNCLAIMED_PFN` is the authoritative claimed test. O(capacity),
    /// and this is a diagnostic, not a hot path.
    ///
    /// Reads 0 immediately after a `mega_flush`, which resets every slot —
    /// so this is "claimed right now", not a cumulative total. It is NOT
    /// comparable against `Codegen::function_count`, which resets on a
    /// Cranelift arena flush instead; the two count over different windows.
    #[inline]
    pub fn pages_used(&self) -> usize {
        self.pages.iter().filter(|p| p.is_claimed()).count()
    }

    /// Pool-wide churn-avoidance totals as `(skipped, rejected)`: compile
    /// requests answered by re-validating already-installed code, vs. those
    /// that had a snapshot to compare against and had to compile anyway.
    pub fn redundant_compile_totals(&self) -> (u64, u64) {
        let (mut skipped, mut rejected) = (0u64, 0u64);
        for page in self.pages.iter().filter(|p| p.is_claimed()) {
            skipped += page.redundant_skipped() as u64;
            rejected += page.redundant_rejected() as u64;
        }
        (skipped, rejected)
    }

    /// Pool capacity, as passed to `new()`.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Sum, across every pooled page with a published function, of the
    /// page's `code_size` **rounded up to `Codegen::HOST_PAGE_SIZE`** —
    /// dev-only diagnostic (`j2 stats`), the best available proxy for the
    /// shared `Codegen`'s actual Cranelift memory-arena usage. Rounding
    /// matters: `code_size` is raw compiled-machine-code bytes, but
    /// `ArenaMemoryProvider` gives every function its own segment, always
    /// rounded up to a full host page regardless of actual size — summing
    /// raw `code_size` alone would under-report real arena consumption.
    /// §13: page-granular now (one function per page), not per-offset — see
    /// `PhysicalCodePage::code_size`'s own field doc.
    pub fn code_bytes_used(&self) -> u64 {
        let page_size = crate::jitv2::codegen::Codegen::HOST_PAGE_SIZE;
        self.pages.iter()
            .filter(|page| !page.func().is_null())
            .map(|page| {
                let raw = page.code_size.load(Ordering::Relaxed) as u64;
                raw.div_ceil(page_size) * page_size
            })
            .sum()
    }

    /// Histogram of `PhysicalCodePage::instr_count` across every pooled page
    /// with a published function, paired with each bucket's code-size
    /// distribution — dev-only diagnostic (`j2 status`). Indexed by
    /// instruction count directly (`result[n]` = stats for published pages
    /// whose most recent compile covered exactly `n` instructions); `n == 0`
    /// is always absent. §13: page-granular now, not per-offset — see
    /// `PhysicalCodePage::instr_count`'s own field doc for why per-offset
    /// overlap accounting no longer applies (overlap between entries is
    /// structurally impossible under the one-function-per-page design).
    pub fn code_size_by_instr_count(&self) -> Vec<Option<CodeSizeBucket>> {
        let mut hist: Vec<Option<CodeSizeBucket>> = Vec::new();
        for page in self.pages.iter() {
            if page.func().is_null() { continue; }
            let n = page.instr_count.load(Ordering::Relaxed) as usize;
            let size = page.code_size.load(Ordering::Relaxed);
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
        hist
    }

    /// Every claimed page in the pool, exposed as a borrowing iterator
    /// rather than folded into an aggregate, for callers (`j2 html`) that
    /// need the raw per-page detail instead of a summary statistic. Not
    /// necessarily a contiguous `pages[..n]` prefix — `free_page` can return
    /// an individual slot mid-run, so claimed slots may be scattered anywhere
    /// in the array — hence the full scan filtered on
    /// [`PhysicalCodePage::is_claimed`]. `pages` itself stays private (index
    /// stability, see the field's own doc comment, is an invariant only this
    /// module should rely on).
    pub fn claimed_pages(&self) -> impl Iterator<Item = &PhysicalCodePage> {
        self.pages.iter().filter(|p| p.is_claimed())
    }

    /// Reset the whole page pool to empty — the MAME-style "flush the world"
    /// response to running out of any bump-allocated JIT resource (§6.3's
    /// `flush_all()`: arena-full, `restore`, `rollback` all route through
    /// this one routine).
    ///
    /// Every slot gets [`PhysicalCodePage::reset_to_unclaimed`] (which clears
    /// `pfn` back to `UNCLAIMED_PFN`, so `pages_used()` reads 0 afterwards),
    /// `pfn_to_slot` is cleared, and the free list is relinked across the
    /// entire array from slot 0. Slot *indices* stay stable — the array is
    /// reset in place and never reallocates (see the `pages` field's own doc
    /// comment) — but a slot's *contents* do not survive, so any surviving
    /// pointer to one must re-check pfn/gen before trusting it.
    ///
    /// An earlier design preserved the `pfn`/`requested`/`denied` of the N
    /// most-recently-used pages across a flush and immediately requeued them
    /// for recompilation, to avoid re-warming the page cache from nothing.
    /// That was removed: the preserved entry points could be stale, and
    /// accumulating bad ones was measured as at best neutral and slightly
    /// harmful. The MRU list it depended on is gone with it.
    ///
    /// Does not yet demote promoted decode-entry handlers or null
    /// entry_table slots (§6.1.3, §6.3) — there are none to demote until
    /// the dispatcher/compiler land. Once they exist, this is where that
    /// walk goes, on the executor thread, before the reset loop below.
    ///
    /// Private: real callers want [`Self::flush_from_cpu_thread`]/
    /// [`Self::flush_from_jit_thread`], which wrap this with the compile-queue
    /// pause/restart and requeue every caller of this used to have to
    /// remember (see those methods' doc comments for why bundling it there,
    /// rather than leaving it the caller's responsibility, is what makes the
    /// whole operation self-contained now that `Jitv2` owns its own
    /// compile-queue lifecycle independently of `MipsCpu::stop()`/`start()`).
    fn mega_flush(&mut self) {
        self.pfn_to_slot.clear();
        let cap = self.pages.len();
        for (i, page) in self.pages.iter_mut().enumerate() {
            page.reset_to_unclaimed();
            page.next = (i + 1) as u32;
        }
        if cap > 0 {
            self.pages[cap - 1].next = NO_SLOT;
            self.free_head = 0;
        } else {
            self.free_head = NO_SLOT;
        }
        crate::jit_feedback::JIT_FEEDBACK.set_arena_fill(0, CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES);
        crate::jit_feedback::JIT_FEEDBACK.record_flush();
    }

    /// Self-contained page-pool + compiled-code-arena flush, called FROM the
    /// CPU thread (`jitv2_track_pcp`'s pool-exhaustion handler).
    pub unsafe fn flush_from_cpu_thread(&mut self, bus: Arc<dyn BusDevice>) {
        let stopped = self.compile_queue.stop();
        let was_threaded = !stopped.is_empty();
        let mut function_count = 0;
        self.compile_queue.drain_pending_queue();
        if was_threaded {
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
    /// compile thread.
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
///
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

    /// Read-only snapshot of the seal queue's front — `j2 seal-queue`'s only
    /// caller. `None` if the pool has never started (no arena has ever been
    /// built yet). Briefly locks the shared `Mutex<SharedArena>` — not on
    /// any hot path, safe for a monitor command to call at any time
    /// (contends real compiles for the duration of one queue-front peek,
    /// same as any other `try_seal_ready` call already does).
    pub fn seal_queue_snapshot(&self) -> Option<crate::jitv2::paged_memory::SealQueueSnapshot> {
        let (mutex, _cv) = &*self.barrier;
        let arena = mutex.lock().current_arena.clone()?;
        let snapshot = arena.lock().seal_queue_snapshot();
        Some(snapshot)
    }

    /// Every seal-queue entry, front to back — `j2 seal-queue list`'s only
    /// caller. `None` if the pool has never started. See
    /// `SharedArena::seal_queue_entries`'s own doc comment; not bounded, the
    /// caller decides how much to print.
    pub fn seal_queue_entries(&self) -> Option<Vec<(usize, usize, bool, std::thread::ThreadId, *mut crate::jitv2::PhysicalCodePage)>> {
        let (mutex, _cv) = &*self.barrier;
        let arena = mutex.lock().current_arena.clone()?;
        let entries = arena.lock().seal_queue_entries();
        Some(entries)
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
    ///
    /// Clears each dropped request's page-level `page_scheduled` flag before
    /// discarding it — this specific request is never reaching
    /// `handle_request_deferred`, which is the only place that would
    /// otherwise clear it (its own `ClearScheduledOnDrop` guard). Today
    /// `mega_flush`, called by both callers right after this, independently
    /// resets `page_scheduled` for every page still in the pool anyway (both
    /// its preserve and evict branches), so this is currently redundant in
    /// practice — but making it explicit here, at the actual drop site,
    /// means the invariant ("every request this function discards leaves its
    /// page immediately re-schedulable") holds on its own rather than by
    /// coincidence of what a *different* function happens to do right
    /// after. `req.page` is still safely dereferenceable here — the pool
    /// hasn't been cleared yet at either call site (see this function's own
    /// doc comment above for exactly why that ordering is guaranteed).
    fn drain_pending(queue: &crossbeam_queue::ArrayQueue<CompileRequest>) {
        while let Some(req) = queue.pop() {
            unsafe { (*req.page).clear_scheduled(); }
        }
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
        let shared = crate::jitv2::paged_memory::PagedArenaMemoryProvider::new_shared(
            ARENA_RESERVE_SIZE, state.clone(),
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
        // Publish the arena every worker is about to build its own `Codegen`
        // on into the shared barrier state too — `j2 seal-queue`'s only way
        // to reach a live `SharedArena` for a read-only snapshot, since no
        // single worker's own (thread-owned, not externally reachable)
        // `Codegen` is available to a monitor command. Without this, the
        // mirror would stay `None` until the first real flush ever
        // completes (only `run_leader_flush`/`flush_from_cpu_thread`
        // otherwise write these two fields) — this is the one place besides
        // those two that ever hands out a fresh arena, so it's the one
        // place besides those two that must also record it here.
        {
            let (mutex, _cv) = &*self.barrier;
            let mut bstate = mutex.lock();
            bstate.current_arena = Some(shared.clone());
            bstate.current_state = Some(state.clone());
        }
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
    /// `comp::handle_request` (reachability walk + codegen + publish, §6.5 —
    /// no filesystem access of any kind; corpus capture is now an on-demand
    /// `j2 corpus` walk of this pool, see `jitv2/pcp_dump.rs`). Owns the
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
        // Pick up the L1-D geometry the CPU published for the inline
        // load/store path. The worker owns `codegen` by value for its whole
        // life, so the CPU cannot stamp it directly — it goes through
        // `Jitv2::dc_geometry` instead.
        if let Some(j) = jitv2.as_ref().and_then(|w| w.upgrade()) {
            let g = j.lock();
            codegen.dc_geometry = *g.dc_geometry.lock();
            codegen.jit_consts = *g.jit_consts.lock();
        }
        // First worker to get here builds the shared helpers into the arena's
        // lowest range. Startup only — the flush path rebuilds them from the
        // leader while everyone else is parked (see `run_leader_flush`).
        //
        // Serialized on a process-wide latch rather than done per worker: a
        // forced seal mprotects a whole host page to RX, which is only safe
        // when nothing else can still be bump-allocating into it (see this
        // function's own note on `handle_request`'s single-caller contract).
        // Every other worker waits here until the winner is done, so nobody
        // allocates into that page mid-seal.
        {
            static HELPERS_BUILT: parking_lot::Mutex<bool> = parking_lot::Mutex::new(false);
            let mut built = HELPERS_BUILT.lock();
            if !*built {
                codegen.emit_mem_helpers();
                *built = true;
            }
        }


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
                let fresh_arena = crate::jitv2::paged_memory::PagedArenaMemoryProvider::new_shared(
                    ARENA_RESERVE_SIZE, fresh_state.clone(),
                ).expect("run_leader_flush: failed to reserve a fresh jitv2 arena");
                unsafe { codegen.reset_with_shared_arena(fresh_arena.clone(), fresh_state.clone()); }
                // Build the shared memory helpers here and nowhere else.
                // Every other worker is parked at the barrier and the fresh
                // arena has had nothing else allocated into it, so this is
                // the one moment a forced seal — which mprotects a whole
                // host page to RX — cannot race another worker still
                // bump-allocating into that page. Helpers land in the
                // arena's lowest range; every region compiled afterwards
                // sits above them and seals normally.
                codegen.emit_mem_helpers();
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

    /// A claimed, otherwise-untouched page slot for the FR-pin tests below.
    /// `claim` debug-asserts the slot is clean, so it must start unclaimed.
    fn claimed_page(fr1: bool) -> Box<PhysicalCodePage> {
        let mut page = Box::new(PhysicalCodePage::new(UNCLAIMED_PFN, std::ptr::null()));
        page.claim(0x1234, std::ptr::null(), fr1);
        page
    }

    /// A page recycled between processes of different ABIs (o32 FR=0 ->
    /// n32 FR=1) must not keep the previous tenant's FR pin. `publish` already
    /// replaces `compiled` and resets `denied` on a real invalidation,
    /// reasoning that decisions made against the OLD bytes are worthless once
    /// the bytes change; `fr1` has identical justification and used to be the
    /// one field left behind.
    ///
    /// This is the 4Dwm/fm session-start core-dump shape: a burst of process
    /// creation recycles pages, the stale pin makes the next compile
    /// specialize for the wrong FR mode, and because `emit_fr_mode_guard`
    /// suppresses itself when CU1 is clear that is not reliably caught —
    /// wrong FPR packing on `ldc1`/`lwc1` silently corrupts FP data.
    #[test]
    fn real_invalidation_drops_the_stale_fr_pin() {
        let page = claimed_page(false); // o32 tenant: FR0
        assert!(!page.is_fr1());

        // A real invalidation: the guest replaced this page's bytes, so the
        // publish carries a generation strictly newer than `entry_gen`.
        let entries = [0u64; BITMAP_WORDS];
        let published = page.publish(&entries, std::ptr::null(), 1, 1, 0);
        assert!(published, "setup: a fresh generation must actually publish");

        // The next compile arrives from an n32 process (live FR=1). Before
        // the fix this returned the stale `false` and compiled FR0 code for
        // FR1 bytes.
        assert!(page.take_fr_repin(true), "a page whose bytes were replaced must re-derive FR from the live mode, not keep the old tenant's pin");
    }

    /// Coverage added to an EXISTING generation is not an invalidation — the
    /// bytes are unchanged, so the pin still describes them and an unrelated
    /// dispatch in another mode must not silently re-specialize the page.
    /// Guards the obvious over-correction of the test above.
    #[test]
    fn coverage_only_publish_keeps_the_fr_pin() {
        let page = claimed_page(false);
        let mut entries = [0u64; BITMAP_WORDS];
        entries[0] = 1;
        assert!(page.publish(&entries, std::ptr::null(), 0, 1, 0));

        assert!(!page.take_fr_repin(true), "a coverage-only publish leaves the bytes alone, so the pin still describes them");
    }

    /// The FR guard's bail records the mode the dispatch actually demanded,
    /// and the next compile adopts it. Without this a page whose live mode
    /// flipped without its bytes changing (the shared `checkfp.s` page)
    /// rebuilds the same wrong-mode code and guard-fails again, forever.
    #[test]
    fn fr_guard_bail_repins_to_the_demanded_mode() {
        let page = claimed_page(false);
        page.request_fr_repin(true); // guard observed live FR=1

        assert!(page.take_fr_repin(false), "the compile must adopt the mode the bail demanded, not the stale pin");
        assert!(page.is_fr1(), "the demand must be committed to the pin, not just returned once");
        // Consumed: a second compile with no new bail keeps the new pin.
        assert!(page.take_fr_repin(false), "the request is one-shot; the re-pinned mode persists");
    }

    /// The re-pin request stores the DEMANDED mode, not a "flip me" bit.
    /// Multiple bails before one compile is the NORMAL case, not a race: the
    /// guard kills a page's FPU-touching entries one at a time as each is
    /// dispatched, so a page with two such entries records two demands. A
    /// flip-bit encoding would toggle twice and land back on the wrong mode.
    #[test]
    fn repeated_repin_requests_are_idempotent() {
        let page = claimed_page(false);
        page.request_fr_repin(true);
        page.request_fr_repin(true);
        assert!(page.take_fr_repin(false), "two bails demanding FR1 must still mean FR1, not a double flip back to FR0");
    }

    /// With nothing outstanding, an ordinary dispatch must not re-pin a page
    /// that was never mismatched — a page is normally dispatched in its
    /// pinned mode, and re-deriving unconditionally would let any passing
    /// dispatch re-specialize it.
    #[test]
    fn take_fr_repin_is_inert_without_a_request() {
        let page = claimed_page(true);
        assert!(page.take_fr_repin(false), "no request and no invalidation: keep the existing pin");
        assert!(page.is_fr1());
    }

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
        // Guardrail, not a strict contract: §13 replaced the old 1024-entry
        // AoS table with a handful of fixed-size bitmaps + one func pointer,
        // so this should be dramatically smaller than before — fails loudly
        // if PhysicalCodePage ever grows past a sanity ceiling rather than
        // silently ballooning Jitv2::new's one-shot allocation.
        //
        // The churn-avoidance `CompileSnapshot` (see its own doc comment)
        // deliberately put ~4.25KB of that back: a 4KB copy of the page
        // bytes plus two bitmaps. That is the dominant term now — roughly
        // 20MB across the default 4096-slot pool, traded knowingly against
        // the analyze+codegen+finalize passes it avoids. Slots that never
        // compile never touch their copy, so it stays untouched zero pages
        // rather than resident memory. If this assert fires, check whether
        // something OTHER than that snapshot grew before raising the
        // ceiling.
        let page_size = std::mem::size_of::<PhysicalCodePage>();
        println!("size_of::<PhysicalCodePage>() = {page_size} bytes");
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
        assert!(!page.is_runnable(4));
        assert!(!page.is_denylisted(4));
        assert!(page.func().is_null());
    }

    /// `try_schedule` must be a genuine test-and-set: the first caller for a
    /// given offset wins (returns `true`); every subsequent caller for the
    /// same offset, before `clear_scheduled` runs, must lose (returns
    /// `false`) — this is what stops `exec_decoded`'s dispatch gate from
    /// sending a duplicate `CompileRequest` for the same PAGE every time a
    /// hot PC re-satisfies the gate's trigger conditions while the first
    /// request is still in flight (§13.2 — page-scoped now, not per-offset;
    /// see `try_schedule_page`'s own doc comment).
    #[test]
    fn try_schedule_page_is_test_and_set() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        assert!(page.try_schedule_page(), "first caller for a fresh page must win");
        assert!(!page.try_schedule_page(), "second caller before clear_scheduled must lose");
        assert!(!page.try_schedule_page(), "still losing on a third call");
    }

    #[test]
    fn clear_scheduled_allows_a_fresh_try_schedule_page() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        assert!(page.try_schedule_page());
        assert!(!page.try_schedule_page());

        page.clear_scheduled();
        assert!(page.try_schedule_page(), "after clear_scheduled, a fresh request for the same page must be allowed again");
    }

    #[test]
    fn clear_scheduled_on_an_unset_page_is_a_harmless_no_op() {
        // The jitv2_inline_compile path calls handle_request (and therefore
        // clear_scheduled, via its scope guard) without ever having called
        // try_schedule_page first — must not panic.
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        page.clear_scheduled();
        assert!(page.try_schedule_page(), "page must still be schedulable after a no-op clear");
    }

    #[test]
    fn is_scheduled_reflects_try_schedule_page_and_clear_scheduled_without_mutating() {
        // `is_scheduled` (j2 pcp's "scheduled(in-flight)" diagnostic column)
        // must be a pure read — checking it repeatedly must never itself
        // flip the flag the way try_schedule_page's test-and-set does.
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        assert!(!page.is_scheduled());
        assert!(!page.is_scheduled(), "a plain read must not consume/flip the flag");

        assert!(page.try_schedule_page());
        assert!(page.is_scheduled());
        assert!(page.is_scheduled(), "still in flight on a repeated read");

        page.clear_scheduled();
        assert!(!page.is_scheduled());
    }

    #[test]
    fn is_requested_reflects_mark_requested_per_offset() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        assert!(!page.is_requested(4));

        page.mark_requested(4);
        assert!(page.is_requested(4));
        assert!(!page.is_requested(5), "a different offset must not be marked");
    }

    #[test]
    fn mark_requested_is_idempotent_and_per_offset_independent() {
        // §13.2: unlike the old per-offset ENTRY_SCHEDULED test-and-set,
        // mark_requested is unconditional/idempotent — it just accumulates
        // bits for whatever a future compile's walk should cover, with no
        // "first caller wins" contract at the offset level (that dedup now
        // lives at the page level, try_schedule_page).
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        page.mark_requested(4);
        let snap = page.snapshot_requested();
        assert_eq!(snap[0] & (1 << 4), 1 << 4, "offset 4 must be marked requested");
        assert_eq!(snap[0] & (1 << 5), 0, "a different offset must not be marked");

        page.mark_requested(4); // idempotent
        page.mark_requested(5);
        let snap = page.snapshot_requested();
        assert_eq!(snap[0] & ((1 << 4) | (1 << 5)), (1 << 4) | (1 << 5), "both offsets stay marked");
    }

    #[test]
    fn snapshot_compile_candidates_unions_requested_and_compiled_masked_by_denied() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        // offset 4: requested only.
        page.mark_requested(4);
        // offset 8: published (compiled) only, no fresh request.
        let mut bits = [0u64; BITMAP_WORDS];
        bits[0] |= 1u64 << 8;
        assert!(page.publish(&bits, 0x1000 as *const (), 0, 1, 0));
        // offset 12: requested, but denylisted — must never appear regardless
        // of include_compiled.
        page.mark_requested(12);
        page.denylist(12);

        // include_compiled=false: only offset 4 (the fresh request); offset
        // 8's compiled-only coverage is invisible, offset 12 stays excluded.
        let without_compiled = page.snapshot_compile_candidates(false);
        assert_eq!(without_compiled[0] & (1 << 4), 1 << 4, "requested offset must appear");
        assert_eq!(without_compiled[0] & (1 << 8), 0, "compiled-only offset must not appear when include_compiled is false");
        assert_eq!(without_compiled[0] & (1 << 12), 0, "denylisted offset must never appear");

        // include_compiled=true: offset 8 joins offset 4; offset 12 still excluded.
        let with_compiled = page.snapshot_compile_candidates(true);
        assert_eq!(with_compiled[0] & (1 << 4), 1 << 4, "requested offset must still appear");
        assert_eq!(with_compiled[0] & (1 << 8), 1 << 8, "compiled offset must appear when include_compiled is true");
        assert_eq!(with_compiled[0] & (1 << 12), 0, "denylisted offset must never appear even when include_compiled is true");
    }

    #[test]
    fn entry_valid_only_when_bit_set_and_gen_matches() {
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 100usize;

        // Bit not set at all: invalid regardless of gen.
        assert!(!page.is_runnable(offset));

        // Publish: set the bit -> now valid.
        let mut bits = [0u64; BITMAP_WORDS];
        bits[offset >> 6] |= 1u64 << (offset & 63);
        assert!(page.publish(&bits, 0x1000 as *const (), 5, 1, 0));
        assert!(page.is_runnable(offset));

        // Page mutates (gen bumps past what the entry was compiled against):
        // bit is still set, but the entry must read as stale.
        counter.store(6, Ordering::Relaxed);
        assert!(!page.is_runnable(offset), "stale entry (gen mismatch) must not be reported valid");
    }

    #[test]
    fn kill_clears_valid_bit_but_not_denylist() {
        // emit_fr_mode_guard's FR-mismatch arm (jit_kill_entry) uses this
        // to un-publish a compiled-for-the-wrong-FR-mode entry: the JIT gate
        // must stop dispatching it, but a later visit is expected and
        // welcome to recompile the same offset fresh — unlike denylist,
        // which is permanent (§6.4 sticky rejection).
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 4usize;
        let mut bits = [0u64; BITMAP_WORDS];
        bits[offset >> 6] |= 1u64 << (offset & 63);
        assert!(page.publish(&bits, 0x1000 as *const (), 0, 1, 0));
        assert!(page.is_runnable(offset));

        page.kill(offset);

        assert!(!page.is_published(offset), "kill must clear the valid bit");
        assert!(!page.is_runnable(offset));
        assert!(!page.is_denylisted(offset), "kill must not sticky-reject the offset — a fresh compile is expected to follow");

        // A later re-publish (simulating the next visit's fresh compile)
        // must work normally — kill leaves the offset fully recompilable.
        // Same gen (0): a pure entry-coverage republish, not a real
        // invalidation — publish() still succeeds because `kill` cleared the
        // `compiled` bit, so the previous "subsumed" check no longer blocks it.
        assert!(page.publish(&bits, 0x1000 as *const (), 0, 1, 0));
        assert!(page.is_runnable(offset), "offset must be re-publishable after kill");
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
        // at all: func actually gets updated, and is_runnable only
        // reports true once gen matches (i.e. after publish() completes).
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 100usize;
        let mut bits = [0u64; BITMAP_WORDS];
        bits[offset >> 6] |= 1u64 << (offset & 63);

        let old_fn = 0x1000usize as *const ();
        assert!(page.publish(&bits, old_fn, 5, 1, 0));
        assert!(page.is_runnable(offset));
        assert_eq!(page.func(), old_fn);

        // Page mutates: entry goes stale (bit stays 1, gen no longer matches).
        counter.store(6, Ordering::Relaxed);
        assert!(!page.is_runnable(offset), "must read stale immediately after the page mutates");

        // Recompile in place (comp.rs's handle_request path for a
        // stale-but-still-published entry) — gen_snap=6 was captured before
        // this second compile started, matching the page's now-current gen.
        let new_fn = 0x2000usize as *const ();
        assert!(page.publish(&bits, new_fn, 6, 1, 0));
        assert!(page.is_runnable(offset), "recompiled entry must read valid once publish completes");
        assert_eq!(page.func(), new_fn, "func must be the NEW function, not the stale one, once gen reads as current");
    }

    /// Build the page-byte/used/entries triple for the churn-avoidance
    /// tests: `entries` bits set for each offset in `entry_offsets`, `used`
    /// bits set for each word in `used_words`, and `words` filled with a
    /// recognisable pattern.
    fn churn_fixture(
        entry_offsets: &[usize],
        used_words: &[usize],
    ) -> ([u32; ENTRIES_PER_PAGE], [u64; BITMAP_WORDS], [u64; BITMAP_WORDS]) {
        let words: [u32; ENTRIES_PER_PAGE] = std::array::from_fn(|i| 0x1000_0000 + i as u32);
        let mut used = [0u64; BITMAP_WORDS];
        for &w in used_words { used[w >> 6] |= 1u64 << (w & 63); }
        let mut entries = [0u64; BITMAP_WORDS];
        for &o in entry_offsets { entries[o >> 6] |= 1u64 << (o & 63); }
        (words, used, entries)
    }

    /// Stage + commit a compile snapshot against a real publish, the way
    /// `comp.rs` does — the churn tests all start from a page in this state.
    fn publish_with_snapshot(
        page: &PhysicalCodePage,
        words: &[u32; ENTRIES_PER_PAGE],
        used: &[u64; BITMAP_WORDS],
        entries: &[u64; BITMAP_WORDS],
        gen: u64,
        fr1: bool,
        func: *const (),
    ) {
        page.stage_compile_snapshot(words, used, entries, gen, fr1);
        assert!(page.publish(entries, func, gen, 1, 0), "fixture publish must succeed");
        page.commit_compile_snapshot(gen, entries);
        assert!(page.has_compile_snapshot(), "fixture must leave a usable snapshot");
    }

    #[test]
    fn a_bumped_gen_with_unchanged_code_words_skips_the_compile() {
        // The case the whole mechanism exists for: something wrote to the
        // page (gen bumped) but not to any word the compile actually
        // decoded, so recompiling would emit byte-identical code. The skip
        // must re-validate the installed function against the new gen —
        // leaving it dispatchable — without touching func or compiled.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11, 12]);
        let func = 0x1000usize as *const ();
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, func);

        // A write lands somewhere on the page that is NOT one of the decoded
        // words — data sharing the page, DMA, a neighbouring structure.
        let mut changed = words;
        changed[900] = 0xdead_beef;
        counter.store(6, Ordering::Relaxed);
        assert!(!page.is_runnable(10), "gen bump must make the entry stale first");

        assert!(page.try_skip_redundant_compile(&changed, &entries, 6, false),
            "unchanged decoded words must skip the compile");
        assert!(page.is_runnable(10), "the skip must re-validate the installed code against the new gen");
        assert_eq!(page.func(), func, "a skip must leave the installed function alone");
    }

    #[test]
    fn a_changed_decoded_word_forces_a_real_compile() {
        // The complement: the write DID land on a word the compile decoded,
        // so the installed code no longer describes the page and a real
        // recompile is mandatory.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11, 12]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        let mut changed = words;
        changed[11] = 0xdead_beef; // a decoded word
        counter.store(6, Ordering::Relaxed);

        assert!(!page.try_skip_redundant_compile(&changed, &entries, 6, false),
            "a changed decoded word must NOT be skippable");
        assert!(!page.is_runnable(10), "a rejected skip must leave the entry stale, not re-validate it");
    }

    #[test]
    fn an_fr_mode_flip_forces_a_real_compile_despite_identical_bytes() {
        // FR mode is baked into every FPR-access emitter and is invisible in
        // the page bytes, so identical bytes compiled for the other mode are
        // genuinely different code. Skipping here would leave the wrong
        // mode's function installed — silent FP register-half corruption,
        // since emit_fr_mode_guard suppresses itself when CU1 is clear.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        counter.store(6, Ordering::Relaxed);
        assert!(!page.try_skip_redundant_compile(&words, &entries, 6, true),
            "identical bytes in the OTHER FR mode must never be skipped");
        // Same request in the mode it was compiled for still skips, proving
        // the rejection above was the FR check and nothing else.
        assert!(page.try_skip_redundant_compile(&words, &entries, 6, false),
            "the same request in the compiled-for mode must still skip");
    }

    #[test]
    fn a_wanted_entry_point_outside_published_coverage_forces_a_compile() {
        // The installed function's dispatch switch only has cases for the
        // entries its own compile covered. An entry outside that set cannot
        // be served no matter how identical the bytes are — skipping would
        // leave every dispatch into it falling through to the interpreter
        // forever while `entry_gen` claimed it was runnable.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        let mut wants_more = entries;
        wants_more[0] |= 1u64 << 20; // offset 20 was never compiled
        counter.store(6, Ordering::Relaxed);

        assert!(!page.try_skip_redundant_compile(&words, &wants_more, 6, false),
            "asking for an uncompiled entry point must force a real compile");
    }

    #[test]
    fn a_killed_entry_forces_a_compile_even_with_identical_bytes() {
        // `kill` clears a `compiled` bit without dropping `func` (an
        // FR-guard bail, or a denylist landing on a previously-published
        // offset). The skip path re-checks coverage against the LIVE
        // `compiled` bitmap under the lock precisely so a killed entry
        // cannot be silently resurrected by a byte comparison.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        page.kill(10);
        counter.store(6, Ordering::Relaxed);
        assert!(!page.try_skip_redundant_compile(&words, &entries, 6, false),
            "an entry killed since publish must not be revived by a skip");
    }

    #[test]
    fn a_snapshot_is_unusable_until_its_publish_commits_it() {
        // Staging alone must never authorize a skip: on the deferred path
        // the matching publish can be gap-blocked in the seal queue for a
        // while, and a snapshot consulted in that window would re-validate
        // a `func` that was never installed.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);

        page.stage_compile_snapshot(&words, &used, &entries, 5, false);
        assert!(!page.has_compile_snapshot(), "a staged-but-uncommitted record must not be usable");
        assert!(!page.try_skip_redundant_compile(&words, &entries, 5, false),
            "a staged-but-uncommitted record must never authorize a skip");

        assert!(page.publish(&entries, 0x1000usize as *const (), 5, 1, 0));
        page.commit_compile_snapshot(5, &entries);
        assert!(page.has_compile_snapshot(), "the commit must make it usable");
    }

    #[test]
    fn a_commit_for_a_different_compile_drops_the_staged_record() {
        // A commit must only ever validate its OWN record. A page is never
        // compiled twice at once (`page_scheduled` is held for the whole
        // compile), so the reachable way the staging slot stops matching is a
        // reset landing under an in-flight compile — covered separately by
        // `a_late_commit_cannot_revive_a_snapshot_invalidated_by_a_reset`.
        // This pins the matching rule itself, which is what makes that
        // protection work: a commit whose identity doesn't match what was
        // staged drops the record rather than validating it. Costs one
        // future recompile; committing it would let a later skip compare
        // against the wrong bytes.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);

        page.stage_compile_snapshot(&words, &used, &entries, 5, false);
        assert!(page.publish(&entries, 0x1000usize as *const (), 5, 1, 0));
        // A different compile's publish tries to vouch for it.
        let mut other_entries = [0u64; BITMAP_WORDS];
        other_entries[0] |= 1u64 << 42;
        page.commit_compile_snapshot(5, &other_entries);
        assert!(!page.has_compile_snapshot(),
            "a commit whose entries don't match the staged record must drop it, not validate it");
    }

    #[test]
    fn dropping_the_compiled_function_invalidates_the_snapshot() {
        // A snapshot outliving the `func` it describes would let a skip
        // advance `entry_gen` over a null function and send dispatch into
        // nothing. Every reset that clears `func` must clear the snapshot.
        let counter = AtomicU64::new(5);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());
        page.reset_to_unclaimed();
        assert!(!page.has_compile_snapshot(),
            "reset_to_unclaimed drops func, so the snapshot must go with it");
    }

    #[test]
    fn a_late_commit_cannot_revive_a_snapshot_invalidated_by_a_reset() {
        // A compile in flight across a flush comes back and tries to commit.
        // The reset poisoned `staged_gen`, so the commit must find no match
        // and leave the snapshot invalid rather than re-validating a record
        // for a function the flush threw away.
        let counter = AtomicU64::new(5);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);

        page.stage_compile_snapshot(&words, &used, &entries, 5, false);
        page.reset_to_unclaimed();
        page.commit_compile_snapshot(5, &entries);
        assert!(!page.has_compile_snapshot(),
            "a commit arriving after a reset must not revive the snapshot");
    }

    #[test]
    fn a_skip_never_moves_entry_gen_backwards() {
        // A concurrent publish may have already installed newer coverage at
        // a higher generation. Rewinding `entry_gen` to this compile's older
        // snapshot would invalidate that publish's code while leaving it
        // installed.
        let counter = AtomicU64::new(9);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 9, false, 0x1000usize as *const ());

        // A stale request (snapshot taken at gen 7) arrives after the page
        // has already published at gen 9.
        assert!(!page.try_skip_redundant_compile(&words, &entries, 7, false),
            "a skip must refuse to rewind entry_gen");
        assert_eq!(page.entry_gen(), 9, "entry_gen must be left where the newer publish put it");
    }

    #[test]
    fn a_skip_requeues_the_entries_its_narrowing_drops() {
        // Companion to `a_skip_must_not_leave_coverage_the_check_never_verified`:
        // narrowing `compiled` to the verified set is necessary, but it must
        // not silently strand the entries it drops. Unlike `publish`'s
        // equivalent branch, the bytes here are provably UNCHANGED, so a
        // dropped entry is still legitimately wanted — nothing re-requested
        // it only because it was already covered. Folding it back into
        // `requested` is what gives it a path to a real recompile, the same
        // guard `publish` applies for a dropped-but-still-wanted entry.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10, 40], &[10, 11, 40, 41]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());
        // Both were covered, so neither is outstanding as a request.
        page.clear_requested_bits(&entries);
        assert!(!page.is_requested(40));

        counter.store(6, Ordering::Relaxed);
        let mut asks_for_one = [0u64; BITMAP_WORDS];
        asks_for_one[0] |= 1u64 << 10;
        assert!(page.try_skip_redundant_compile(&words, &asks_for_one, 6, false));

        assert!(!page.is_runnable(40), "the dropped entry must not stay published");
        assert!(page.is_requested(40),
            "the dropped entry must be re-requested, or it is stranded on the interpreter \
             with no automatic path back to compiled dispatch");
    }

    #[test]
    fn churn_counters_reset_on_flush_but_rejected_compiles_does_not() {
        // The churn tallies are per-flush-epoch on purpose: a boot storm
        // racks up ~286k skips, and as a lifetime total that would sit in
        // `j2 status` forever instead of describing current behaviour. This
        // pins that, and pins the contrast with `rejected_compiles`, which
        // deliberately DOES survive a flush (pure churn history) — the two
        // sit next to each other and the difference is easy to "tidy up" by
        // mistake later.
        let counter = AtomicU64::new(5);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        page.mark_redundant_compile_skipped();
        page.mark_redundant_compile_rejected();
        page.mark_analyze_rejected(); // bumps rejected_compiles
        assert_eq!((page.redundant_skipped(), page.redundant_rejected()), (1, 1));

        page.reset_to_unclaimed();
        assert_eq!((page.redundant_skipped(), page.redundant_rejected()), (0, 0),
            "reset_to_unclaimed must clear churn counters");
        assert_eq!(page.rejected_compiles(), 1,
            "rejected_compiles is lifetime history and must NOT be reset by a flush");

        // An evicted-and-recycled slot must not carry the previous tenant's
        // counts into a different physical page either.
        page.mark_redundant_compile_skipped();
        page.reset_to_unclaimed();
        assert_eq!(page.redundant_skipped(), 0,
            "a recycled slot must not inherit the previous page's churn counts");
    }

    #[test]
    fn a_skip_must_not_leave_coverage_the_check_never_verified() {
        // THE CHURN BUG. After a gen bump `same_gen` is false, so the
        // caller's candidate set is `requested & denied` — it deliberately
        // EXCLUDES `compiled`. A real compile then walks exactly that set and
        // `publish` REPLACES `compiled` with it wholesale (is_real_invalidation).
        //
        // A skip that only bumps `entry_gen` leaves the OLD, wider `compiled`
        // in place, so `is_runnable` starts reporting true for entries this
        // check never verified. Dispatch calls `func` for one, the installed
        // function's switch has no case for it, and it returns EXEC_FALLBACK
        // — forever, on every arrival, plus a re-request each time. That is
        // strictly worse than having recompiled: high skip counts, lower
        // throughput.
        //
        // So a skip must end with the same coverage a real publish of the
        // same candidate set would have produced.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        // Page was compiled with two entry points.
        let (words, used, entries) = churn_fixture(&[10, 40], &[10, 11, 40, 41]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        // Gen bumps; only offset 10 is freshly requested this round.
        counter.store(6, Ordering::Relaxed);
        let mut asks_for_one = [0u64; BITMAP_WORDS];
        asks_for_one[0] |= 1u64 << 10;
        assert!(page.try_skip_redundant_compile(&words, &asks_for_one, 6, false));

        assert!(page.is_runnable(10), "the verified entry must be runnable");
        assert!(!page.is_runnable(40),
            "offset 40 was NOT in this round's candidate set, so a real compile would have \
             dropped it from `compiled` — a skip that leaves it published sends every \
             dispatch into EXEC_FALLBACK forever");
    }

    #[test]
    fn a_re_requested_dropped_entry_can_still_skip_against_the_same_function() {
        // The snapshot's `entries` records what the installed `func` CAN
        // serve (its switch cases), which does not change while that `func`
        // stays installed. `compiled` records what is currently advertised,
        // which a skip narrows to the set it verified. Keeping those two
        // separate is what lets an entry dropped by one skip come back and be
        // served by a later one WITHOUT a recompile — narrowing the snapshot
        // to match `compiled` would forfeit exactly the reuse this whole
        // mechanism exists for.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10, 40], &[10, 11, 40, 41]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        // First skip verifies only offset 10; 40 is dropped and re-requested.
        counter.store(6, Ordering::Relaxed);
        let mut only_10 = [0u64; BITMAP_WORDS];
        only_10[0] |= 1u64 << 10;
        assert!(page.try_skip_redundant_compile(&words, &only_10, 6, false));
        assert!(!page.is_runnable(40));

        // 40 comes back around. The same `func` still has a case for it, so
        // this must skip too rather than forcing a recompile.
        counter.store(7, Ordering::Relaxed);
        let mut both = [0u64; BITMAP_WORDS];
        both[0] |= (1u64 << 10) | (1u64 << 40);
        assert!(page.try_skip_redundant_compile(&words, &both, 7, false),
            "an entry the installed function can still serve must be re-coverable without a recompile");
        assert!(page.is_runnable(10));
        assert!(page.is_runnable(40), "both entries must be advertised again");
    }

    #[test]
    fn a_stale_denial_carried_by_a_skip_self_heals_on_the_next_real_compile() {
        // A skip deliberately does NOT reset `denied`, unlike `publish` on a
        // gen bump. `publish`'s reset is justified as "the bytes changed, so
        // old denials deserve a fresh chance" — and on the skip path the
        // DECODED bytes provably did not change, so the denials that matter
        // (ones earned while walking those bytes) are still valid.
        //
        // The gap: a denial can be earned against bytes outside `used`, which
        // a skip never compares. Carrying it forward is conservative in the
        // SAFE direction — an offset stays on the interpreter that might have
        // compiled. It is lost opportunity, never wrong code, and it is
        // bounded because it self-heals the moment a decoded word changes and
        // a real compile runs.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        counter.store(6, Ordering::Relaxed);
        page.denylist(40);

        // Skips keep the denial while the decoded words hold still.
        counter.store(7, Ordering::Relaxed);
        let mut only_10 = [0u64; BITMAP_WORDS];
        only_10[0] |= 1u64 << 10;
        assert!(page.try_skip_redundant_compile(&words, &only_10, 7, false));
        assert!(page.is_denylisted(40), "a skip must not resurrect denials it did not re-derive");

        // A decoded word finally changes: no skip, real compile, and
        // `publish`'s own is_real_invalidation reset clears the denial.
        let mut changed = words;
        changed[11] = 0xdead_beef;
        counter.store(8, Ordering::Relaxed);
        assert!(!page.try_skip_redundant_compile(&changed, &only_10, 8, false),
            "a changed decoded word must force a real compile");
        assert!(page.publish(&only_10, 0x2000usize as *const (), 8, 1, 0));
        assert!(!page.is_denylisted(40),
            "the real compile's publish must reset denied, healing any stale denial a skip carried");
    }

    #[test]
    fn a_rewritten_excluded_entry_word_forces_a_real_compile() {
        // Every production denylist is a COMPILER verdict, with three
        // reasons: (1) the entry word itself was excluded, (2) the merged
        // region was too short, (3) Cranelift declined. For (2) and (3) the
        // offending code is a COVERED offset, so its word is in `used` by
        // construction and a rewrite fails the byte compare on its own.
        //
        // (1) is the odd one out: `visit` returns false, so
        // `walk_multi_entry` never marks that word visited and
        // `instrs_linear` never reports it. `prepare_multi_entry_compile`
        // therefore folds the excluded entry's own word into `used`
        // explicitly — without it, the guest could rewrite that instruction
        // into something compilable and the sticky denial would survive
        // every skip forever, when a real compile would have reset `denied`.
        //
        // This test drives the page API directly with a `used` mask built the
        // way that fold builds it, i.e. covered words PLUS the excluded
        // entry's word.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        // Covered {10} (words 10,11); offset 40 was a candidate the walk
        // excluded — so 40 is denied AND 40 is folded into `used`.
        let (words, used, entries) = churn_fixture(&[10], &[10, 11, 40]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());
        page.denylist(40);

        let mut only_10 = [0u64; BITMAP_WORDS];
        only_10[0] |= 1u64 << 10;

        // Unchanged: the skip is still available (the denial is still valid
        // against these exact bytes, so re-deriving it would be pure churn).
        counter.store(6, Ordering::Relaxed);
        assert!(page.try_skip_redundant_compile(&words, &only_10, 6, false),
            "an unchanged excluded word must not cost a recompile");

        // Rewritten: the excluded word is in `used`, so the compare fails and
        // a real compile runs — which resets `denied` and retries offset 40.
        let mut rewritten = words;
        rewritten[40] = 0x2021_0000;
        counter.store(7, Ordering::Relaxed);
        assert!(!page.try_skip_redundant_compile(&rewritten, &only_10, 7, false),
            "rewriting an excluded+denied entry word must force a real compile, or the \
             sticky denial outlives the instruction that earned it");
    }

    #[test]
    fn a_denied_offsets_word_stays_compared_across_a_later_same_gen_compile() {
        // `denied` is sticky page state that outlives the compile that set
        // it, so "this compile's denials are covered by this compile's
        // `used`" is NOT sufficient. A later same-generation compile (pure
        // added coverage — it does not reset `denied`) re-stages the
        // snapshot, and its own walk has no reason to touch a word denied
        // long ago: `denied` masks that offset out of the candidate set, so
        // nothing walks it. The denial would stay live while the evidence for
        // it dropped out of the snapshot, and from then on no skip could ever
        // notice the guest rewriting that instruction into something
        // compilable — the denial outliving the code that earned it,
        // permanently.
        //
        // `prepare_multi_entry_compile` therefore folds the WHOLE `denied`
        // set into `used` when it stages, not just the offsets it denied
        // itself. This test models that second compile's staging directly.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        // Compile A covers {10}, denies 40.
        let (words, used_a, entries_a) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used_a, &entries_a, 5, false, 0x1000usize as *const ());
        page.denylist(40);

        // Compile B, SAME generation: adds entry 20. Its walk covers only
        // {10,11,20,21} — nothing walks word 40. Staging applies the fold,
        // so word 40 is still carried into `used`.
        let (_w, mut used_b, entries_b) = churn_fixture(&[10, 20], &[10, 11, 20, 21]);
        for (i, w) in page.snapshot_denied_raw().iter().enumerate() {
            used_b[i] |= !*w;
        }
        page.stage_compile_snapshot(&words, &used_b, &entries_b, 5, false);
        assert!(page.publish(&entries_b, 0x2000usize as *const (), 5, 1, 0));
        page.commit_compile_snapshot(5, &entries_b);
        assert!(page.is_denylisted(40), "a same-gen publish must not reset denied");

        let mut only_10 = [0u64; BITMAP_WORDS];
        only_10[0] |= 1u64 << 10;

        // Unchanged bytes still skip — the fold must not cost churn.
        counter.store(6, Ordering::Relaxed);
        assert!(page.try_skip_redundant_compile(&words, &only_10, 6, false),
            "folding denied words in must not break the ordinary skip");

        // Rewriting the denied word is now noticed, even though the compile
        // that staged this snapshot never walked it.
        let mut rewritten = words;
        rewritten[40] = 0x2021_0000;
        counter.store(7, Ordering::Relaxed);
        assert!(!page.try_skip_redundant_compile(&rewritten, &only_10, 7, false),
            "a denied offset's word must stay compared across later compiles, or the denial \
             outlives the instruction that earned it with no way back");
    }

    #[test]
    fn every_cpu_thread_denial_path_also_kills_which_blocks_the_skip() {
        // Why the churn-avoidance `denied` fold does not need to worry about
        // denials originating on the CPU thread.
        //
        // Structurally, it could not matter anyway: the CPU thread only ever
        // reaches an offset it was actually EXECUTING, which means that
        // offset was published and dispatchable, which means it was in
        // `covered` at its last compile — so its word is already in `used` by
        // construction and its bytes are already compared.
        //
        // Mechanically there is also a backstop, which is what this test
        // pins: every CPU-thread path that touches denial state (`j2 deny`
        // in the monitor, and the FR-guard's `jit_kill_entry`) also calls
        // `kill()`, and `kill()` sets `killed_since_snapshot`, which refuses
        // the next skip outright so a real compile re-derives everything.
        //
        // If a future CPU-thread path ever denylists WITHOUT killing, this
        // test won't catch it directly — but the invariant it records is the
        // thing to re-check at that point.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        let mut only_10 = [0u64; BITMAP_WORDS];
        only_10[0] |= 1u64 << 10;
        counter.store(6, Ordering::Relaxed);

        // The `j2 deny` / FR-guard shape: denylist + kill together.
        page.denylist(40);
        page.kill(40);

        assert!(!page.try_skip_redundant_compile(&words, &only_10, 6, false),
            "a kill from the CPU thread must block the next skip, so a real compile re-derives \
             coverage and denial state rather than a skip carrying stale assumptions forward");
    }

    #[test]
    fn a_skip_does_not_reset_the_denylist() {
        // A gen bump normally clears `denied` (§13.6: new bytes deserve a
        // fresh chance). A skip proves the opposite — the decoded bytes are
        // IDENTICAL — so an offset denied against them is still correctly
        // denied. Clearing it would re-walk and re-reject it on every gen
        // bump, which is exactly the churn this exists to remove.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());
        page.denylist(200);
        assert!(page.is_denylisted(200));

        counter.store(6, Ordering::Relaxed);
        assert!(page.try_skip_redundant_compile(&words, &entries, 6, false));
        assert!(page.is_denylisted(200),
            "a skip must leave the denylist alone — the bytes are provably unchanged");
    }

    #[test]
    fn denylist_bit_is_independent_of_valid_bit() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 7usize;
        page.denylist(offset);
        assert!(page.is_denylisted(offset));
        assert!(!page.is_runnable(offset), "denylisting must not itself mark an entry valid");
    }

    #[test]
    fn denylist_alone_does_not_clear_an_already_published_entry() {
        // The real reason `prepare_multi_entry_compile` (comp.rs) must pair
        // every `denylist` call on a previously-`compiled` offset with an
        // explicit `kill()`: dispatch (`exec_decoded`'s gate, mips_exec.rs)
        // only ever consults `is_published`/`is_runnable`, never
        // `is_denylisted` — denylisting purely gates whether a FUTURE
        // compile is offered this offset as a candidate again, it does
        // nothing to stop dispatch from jumping into whatever `func`
        // currently claims to cover it. An offset that was published, then
        // denylisted without also being killed, would still read
        // `is_runnable() == true` forever, dispatchable into a `func`
        // that may no longer actually have a case for it.
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 7usize;
        let mut bits = [0u64; BITMAP_WORDS];
        bits[offset / 64] |= 1u64 << (offset % 64);
        assert!(page.publish(&bits, 0x1000 as *const (), 0, 1, 0));
        assert!(page.is_runnable(offset));

        page.denylist(offset);
        assert!(page.is_denylisted(offset));
        assert!(page.is_runnable(offset),
            "denylist alone must not clear a previously-published entry's valid bit — \
             proving kill() is the mechanism that actually has to do that");

        page.kill(offset);
        assert!(!page.is_runnable(offset), "kill() is what actually un-publishes the entry");
    }

    #[test]
    fn compiles_since_flush_counts_successful_publishes_and_is_cleared_by_both_reset_paths() {
        // Always-present, opposite-lifetime counterpart to the developer-only
        // compile_count: answers "has this page compiled at all since its
        // last flush/claim" without needing a developer build.
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        assert_eq!(page.compiles_since_flush(), 0);

        let mut bits = [0u64; BITMAP_WORDS];
        bits[0] |= 1u64 << 4;
        assert!(page.publish(&bits, 0x1000 as *const (), 0, 1, 0));
        assert_eq!(page.compiles_since_flush(), 1);

        // A redundant publish (nothing new, same generation) must not count.
        assert!(!page.publish(&bits, 0x1000 as *const (), 0, 1, 0));
        assert_eq!(page.compiles_since_flush(), 1, "a subsumed/no-op publish must not increment the counter");

        let mut bits2 = [0u64; BITMAP_WORDS];
        bits2[0] |= 1u64 << 8;
        assert!(page.publish(&bits2, 0x1000 as *const (), 0, 1, 0));
        assert_eq!(page.compiles_since_flush(), 2);

        page.reset_to_unclaimed();
        assert_eq!(page.compiles_since_flush(), 0, "reset_to_unclaimed must clear it");

        assert!(page.publish(&bits, 0x2000 as *const (), 0, 1, 0));
        assert_eq!(page.compiles_since_flush(), 1);
        page.reset_to_unclaimed();
        assert_eq!(page.compiles_since_flush(), 0, "reset_to_unclaimed must clear it too");
    }

    #[test]
    fn rejected_compiles_accumulates_and_survives_both_reset_paths() {
        // Distinguishes "this page has been attempted repeatedly and keeps
        // getting rejected" from "never attempted at all" — compiles_since_flush=0
        // alone can't tell those apart. Lifetime total, same reasoning as
        // compile_count's own doc comment: unlike compiles_since_flush, this
        // is NOT cleared by either reset path.
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        assert_eq!(page.rejected_compiles(), 0);

        page.mark_analyze_rejected();
        assert_eq!(page.rejected_compiles(), 1);
        page.mark_analyze_rejected();
        page.mark_codegen_rejected();
        assert_eq!(page.rejected_compiles(), 3, "the umbrella total must sum both stage-specific counters");
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
        let slot_a = jit.page_for(3, 3 * PAGE_SIZE, &dev, false).unwrap();
        let slot_b = jit.page_for(3, 3 * PAGE_SIZE, &dev, false).unwrap();
        assert_eq!(slot_a, slot_b, "second arrival at the same pfn must reuse the slot");
        let slot_c = jit.page_for(4, 4 * PAGE_SIZE, &dev, false).unwrap();
        assert_ne!(slot_a, slot_c);
    }

    #[test]
    fn page_for_returns_none_when_pool_exhausted() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(1);
        assert!(jit.page_for(0, 0, &dev, false).is_some());
        assert!(jit.page_for(1, PAGE_SIZE, &dev, false).is_none(), "pool of 1 must reject a second distinct pfn");
    }

    #[test]
    fn mega_flush_resets_all_pages_and_frees_all_slots() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(4);
        for i in 0..4 as Pfn {
            jit.page_for(i, i * PAGE_SIZE, &dev, false).unwrap();
        }
        assert_eq!(jit.pages_used(), 4);
        assert!(jit.page_for(4, 4 * PAGE_SIZE, &dev, false).is_none());

        jit.mega_flush();
        assert_eq!(jit.pages_used(), 0);

        for i in 0..4 as Pfn {
            assert!(jit.page_for(i + 10, (i + 10) * PAGE_SIZE, &dev, false).is_some());
        }
        assert_eq!(jit.pages_used(), 4);
    }

    #[test]
    fn fast_lookup_ptrs_lookup_hit() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(4);
        let slot = jit.page_for(7, 7 * PAGE_SIZE, &dev, false).unwrap();
        let (pfn_map_ptr, pages_base) = jit.fast_lookup_ptrs();
        unsafe {
            let mapped_slot = *pfn_map_ptr.add(7);
            assert_eq!(mapped_slot, slot);
            let page = &*pages_base.add(mapped_slot as usize);
            assert_eq!(page.pfn, 7);
        }
    }

    #[test]
    #[cfg(feature = "developer")]
    fn code_size_by_instr_count_buckets_by_instr_count_and_tracks_min_max_sum() {
        // §13: instr_count/code_size are page-level now (one function per
        // page), so distinct buckets need distinct pages, not distinct
        // offsets on the same page.
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(3);

        let bits0: [u64; BITMAP_WORDS] = { let mut b = [0u64; BITMAP_WORDS]; b[0] |= 1; b };

        // Two pages at instr_count=3 (sizes 100, 300), one at instr_count=5 (size 50).
        let slot_a = jit.page_for(0, 0, &dev, false).unwrap() as usize;
        assert!(jit.pages[slot_a].publish(&bits0, 0x1000 as *const (), 0, 3, 100));
        let slot_b = jit.page_for(1, PAGE_SIZE, &dev, false).unwrap() as usize;
        assert!(jit.pages[slot_b].publish(&bits0, 0x2000 as *const (), 0, 3, 300));
        let slot_c = jit.page_for(2, 2 * PAGE_SIZE, &dev, false).unwrap() as usize;
        assert!(jit.pages[slot_c].publish(&bits0, 0x3000 as *const (), 0, 5, 50));

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
            let _ = i;
            assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true }, &stats));
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
        // offset 0, matching every other AddiuDevice-based test in this
        // file: AddiuDevice's own read32 only decodes a real ADDIU for word
        // offsets < 2, sentinel otherwise (see its own doc comment) — offset
        // 4 (word index 4) would land straight on the sentinel with nothing
        // to compile at all. §13.2: mark_requested BEFORE send — a request
        // carries no offset, the compile thread reads `requested` fresh at
        // dequeue time.
        page.mark_requested(0);
        let stats = JitStats::default();
        assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true }, &stats));

        // 30s, not the original 5s: this test passes in ~0.1s in isolation,
        // but the compile-worker thread genuinely needs real CPU time to get
        // scheduled — 5s proved too tight under a full-workspace parallel
        // test run (confirmed live: this deadline firing, not any actual
        // correctness bug, was tripping under heavy contention from every
        // other test's threads competing for the same cores).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            while !page.is_runnable(0) {
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
            // §13.2: same as the queue-drain-fallback test above.
            page.mark_requested(0);
            assert!(q.send(CompileRequest { page: page as *mut PhysicalCodePage, compiled_for_fr1: true }, &stats));
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
            while !pages[0].is_runnable(0) {
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
            // §13.2: a request carries no offset anymore — the compile
            // thread snapshots `requested` fresh at dequeue time, so the
            // offset must be marked BEFORE sending, or handle_request_deferred
            // finds zero candidates and silently no-ops.
            page.mark_requested(0);
            assert!(q.send(CompileRequest { page: page as *mut PhysicalCodePage, compiled_for_fr1: true }, &stats));
        }

        // catch_unwind for the same reason as the other threaded tests in
        // this file: a deadline timeout must not unwind out of this
        // function with worker threads still live against `pages` — see
        // the page-cross test's own comment for the confirmed
        // use-after-free this guards.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loop {
                let published = pages.iter().filter(|p| p.is_runnable(0)).count();
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
        assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true }, &stats));
        q.start(dev, std::sync::Arc::new(JitStats::default()));
        q.stop();
    }

    #[test]
    fn compile_queue_send_drops_when_full() {
        let q = CompileQueue::new();
        // Don't start the worker: nothing drains, so capacity fills exactly.
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        let stats = JitStats::default();
        let mut accepted = 0;
        for i in 0..(COMPILE_QUEUE_CAPACITY + 10) {
            if q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true }, &stats) {
                accepted += 1;
            }
        }
        assert_eq!(accepted, COMPILE_QUEUE_CAPACITY, "queue must drop pushes past capacity, not block or panic");
    }

    #[test]
    fn drain_pending_queue_clears_scheduled_on_every_discarded_request() {
        // A request sitting in the queue when a flush drains it never
        // reaches handle_request_deferred, the only other place that clears
        // page_scheduled (via its ClearScheduledOnDrop guard) — drain_pending
        // must do it itself, or (absent mega_flush's own independent reset
        // covering the same page) that page would be permanently blocked
        // from ever being scheduled again.
        let mut q = CompileQueue::new();
        // Don't start the worker: nothing pops on its own, so the request is
        // still sitting there for drain_pending_queue to discard directly.
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        assert!(page.try_schedule_page());
        let stats = JitStats::default();
        assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true }, &stats));
        assert!(page.is_scheduled(), "sanity: the send above left the page marked in-flight");

        q.drain_pending_queue();
        assert!(!page.is_scheduled(),
            "drain_pending must clear_scheduled() every request it discards, or the page is stuck \
             thinking a compile is still in flight for it forever");
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
