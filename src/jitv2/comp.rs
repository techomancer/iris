//! JIT v2 compiler-thread logic. See `rules/jitv2/jit-v2-design.md`. Two
//! complete implementations selected by the `j2wp` feature: `old_impl` is
//! the default one-function-per-entry-point compile-from-arrival protocol
//! (single `req.offset`, `Codegen::compile_region`); `new_impl` is §13's
//! whole-page multi-entry protocol (`page.requested` bitmap snapshot,
//! `Codegen::compile_region_uncommitted` + deferred finalize). Both are
//! straight copies of their respective commit's real, previously-working
//! `comp.rs` — see each module's own doc comment for its actual protocol
//! description.

#[cfg(not(feature = "j2wp"))]
mod old_impl {
//! JIT v2 compiler-thread logic. See `rules/jitv2/jit-v2-design.md`.
//!
//! `handle_request` is the compile-from-snapshot protocol (§6.5): read the
//! page's 4KB snapshot off the bus, bounded-walk it from the requested
//! offset, hand the walked region to `Codegen::compile_region`, and publish
//! the result (or sticky-deny the offset if codegen declined it). Bounded to
//! `MAX_INSTRS_PER_COMPILE` *head* instructions — started at 1 (the smallest
//! possible working JIT) and growing incrementally from there, not a
//! redesign each time (the walk bound is just `max_instrs` on
//! `Analyzer::walk_bounded`). A branch/jump's mandatory delay slot is never
//! charged against this budget (`Analyzer::visit_slot` — a slot can never be
//! omitted, so it was never a truncation candidate) — including a nested
//! delay-slot chain (branch-in-delay-slot, "unusual but legal" on real
//! hardware), which keeps extending for free until it terminates or runs
//! off the page, at which point the walk declines the whole region rather
//! than compiling a partial chain.
//!
//! Corpus collection (raw page dump to `jitv2_corpus/`, used to develop the
//! analyzer/codegen offline against real captured pages) is preserved behind
//! the `jitv2_corpus_dump` feature — with it off, `handle_request` never
//! touches the filesystem.

use std::sync::Arc;

use crate::jitv2::analyzer::Analyzer;
use crate::jitv2::codegen::Codegen;
use crate::jitv2::{CompileRequest, ENTRIES_PER_PAGE, PAGE_SIZE};
use crate::traits::BusDevice;

/// Instruction budget for a compile-from-arrival region (see module doc):
/// head instructions only — a branch/jump's mandatory delay slot (or nested
/// slot-chain) is free and always included regardless of this number
/// (`Analyzer::visit_slot`). Raised incrementally from 1 (the original
/// "smallest possible working JIT" milestone) through 2/3/4/8, all booted
/// clean — a live `j2 status` histogram at 8 showed real regions landing
/// anywhere from 1 to 16 instructions (the tail past the nominal budget
/// comes from a branch's mandatory delay slot counting toward the total but
/// not the budget itself), clustering 8-11, with only a handful ever
/// reaching 16. Defaults to 128 — real regions cluster 8-11 instructions, so
/// this is a generous headroom rather than a tight cap, chosen to bound the
/// rare pathological case (long branch-free/self-chaining-delay-slot runs)
/// where an unbounded walk would otherwise let the analyzer grow one region
/// arbitrarily large, producing a single huge, slow-to-compile Cranelift
/// function and a single huge arena allocation for it. Any value at or
/// above `ENTRIES_PER_PAGE` (1024 words/page) is equivalent to no budget at
/// all, since the walk was always going to decline rather than compile a
/// region longer than fits on one physical page anyway (module doc: "runs
/// off the page... declines the whole region") — the page boundary, not
/// this constant, is the real ceiling past that point (same as
/// `Analyzer::walk`'s own unbounded case). `j2 max-instrs [N]` tunes this at
/// runtime (e.g. to shrink compiled regions further for debugging/bisection,
/// or raise it back toward `usize::MAX`); both `Analyzer::walk_bounded` and
/// `Codegen::compile_region` were already written generically against
/// `max_instrs` (fallthrough-edge wiring for a multi-instruction
/// straight-line region already exists, per `compile_region`'s Pass 2), so
/// this is just a config read, not a redesign.
static MAX_INSTRS_PER_COMPILE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(128);

pub fn set_max_instrs_per_compile(n: usize) {
    MAX_INSTRS_PER_COMPILE.store(n.max(1), std::sync::atomic::Ordering::Relaxed);
}

pub fn max_instrs_per_compile() -> usize {
    MAX_INSTRS_PER_COMPILE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Minimum walked-region instruction count a compile must reach to actually
/// get compiled — regions shorter than this are sticky-denylisted instead,
/// same treatment as any other decline (§6.4). Below this floor, the
/// per-compile fixed overhead (Cranelift IR building + one arena allocation
/// + one entry_table publish) very likely costs more than the region will
/// ever save over just interpreting it: a true single-instruction region is
/// the worst case for this tradeoff and, per `MAX_INSTRS_PER_COMPILE`'s own
/// doc comment, real regions cluster 8-11 instructions anyway — a
/// one-instruction region is far more often either a rare cold path or an
/// analyzer/codegen edge case than a genuinely hot single-instruction loop
/// body worth paying compile cost for. `j2 min-instrs [N]` tunes this at
/// runtime. Applies identically to `handle_request` and
/// `handle_request_deferred` — both consult `min_instrs_to_compile()` right
/// after a successful, non-empty walk.
///
/// Defaults to 1 (no filtering) under `developer` — diagnostics builds want
/// to see and measure every compile the analyzer/codegen would otherwise
/// attempt, not have some silently skipped by a production-tuned floor — and
/// to 2 otherwise, since real-world usage wants a *little* filtering by
/// default rather than requiring a manual `j2 min-instrs` on every run.
static MIN_INSTRS_TO_COMPILE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(
    if cfg!(feature = "developer") { 1 } else { 2 }
);

pub fn set_min_instrs_to_compile(n: usize) {
    MIN_INSTRS_TO_COMPILE.store(n.max(1), std::sync::atomic::Ordering::Relaxed);
}

pub fn min_instrs_to_compile() -> usize {
    MIN_INSTRS_TO_COMPILE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Handle a single compile request end to end (§6.5): snapshot the page,
/// walk from `req.offset`, codegen, and publish. No-ops (declining silently)
/// if the page's snapshot read fails, the walk finds the entry offset
/// excluded, or codegen has no emitter for something in the walked region —
/// the last two sticky-denylist the offset so arrival stops re-requesting a
/// compile that will only be declined again (§6.4).
///
/// Returns `true` iff `codegen`'s arena ran out of memory on this call
/// (`Codegen::last_compile_ran_out_of_memory`) — the caller (`worker_loop`
/// or the inline dispatch path) owns the `Jitv2`/CPU-pause machinery this
/// function doesn't have access to, so it's responsible for actually
/// flushing when this is `true`. `false` for every other outcome
/// (success, a real codegen decline, or a transient bus-read failure).
pub fn handle_request(
    req: &CompileRequest,
    bus: &Arc<dyn BusDevice>,
    analyzer: &mut Analyzer,
    codegen: &mut Codegen,
    #[cfg(feature = "developer")] stats: &crate::jitv2::JitStats,
) -> bool {
    let page = unsafe { &*req.page };
    let offset = req.offset as usize;

    // Clears this offset's ENTRY_SCHEDULED flag on every return path,
    // including early returns and panics — `exec_decoded`'s dispatch gate
    // (mips_exec.rs) sets this bit via `try_schedule` before sending a
    // request, specifically so a hot PC that keeps re-satisfying the gate's
    // trigger conditions on every dispatch (e.g. a loop back-edge landing on
    // the same still-uncompiled word every iteration) doesn't flood the
    // compile queue with duplicate requests for the exact same offset while
    // the first one is still in flight. Whatever this function decides
    // (publish, denylist, or "retry later" on a bus read failure), the bit
    // must come back down afterward or that offset can never be
    // (re-)requested again — a scope guard rather than clearing at each of
    // the several return points below so a future added early-return can't
    // forget it.
    struct ClearScheduledOnDrop<'a> { page: &'a crate::jitv2::PhysicalCodePage, offset: usize }
    impl Drop for ClearScheduledOnDrop<'_> {
        fn drop(&mut self) { self.page.clear_scheduled(self.offset); }
    }
    let _clear_scheduled = ClearScheduledOnDrop { page, offset };

    if page.is_denylisted(offset) || page.is_runnable(offset) {
        return false; // already decided (rejected or already has a fresh artifact)
    }

    let phys_base = page.pfn * PAGE_SIZE;
    // gen_snap is read *before* the copy (§6.5 step 2) so a mutation that
    // lands between this read and the snapshot copy below is never missed:
    // worst case it's captured in the copy too, and publish's re-check
    // catches anything after.
    let gen_snap = page.current_gen();

    // Dirty-cache gate. The snapshot below reads RAM off the bus; the guest
    // CPU sees RAM overlaid with its own dirty L1-D/L2 lines, and a store
    // that retires inside the cache bumps no generation counter. Compiling
    // through that would publish pre-store bytes as a valid region. Abort
    // instead and publish nothing — nothing is denylisted, so this offset is
    // retried on its next arrival, by which point the lines have usually
    // aged out to RAM on their own.
    //
    // Checked *before* the 4KB bus read: there is no point paying for a
    // snapshot we are about to discard. The probe is racy by design and can
    // miss a store that lands after it looks — see
    // `jitv2::jit_page_has_dirty_lines` for why that is strictly better than
    // the status quo rather than a new hazard.
    //
    // Compiled out entirely under `tcache`: there the cache reads and writes
    // RAM through the ppmem window, so a store is in RAM the moment it
    // retires and there is no hidden dirty data for the probe to find. The
    // executor installs no probe in that build, so this would be a global
    // load that can only ever answer false.
    #[cfg(not(feature = "tcache"))]
    if crate::jitv2::jit_page_has_dirty_lines(phys_base as u64) {
        #[cfg(feature = "developer")]
        stats.record_reject(crate::jitv2::RejectReason::PageDirtyInCache);
        return false;
    }

    let mut words = [0u32; ENTRIES_PER_PAGE];
    for (i, w) in words.iter_mut().enumerate() {
        let r = bus.read32(phys_base + (i as u32) * 4);
        if !r.is_ok() {
            return false; // page not readable right now; the offset stays un-denylisted and can retry later
        }
        *w = r.data;
    }

    #[cfg(feature = "jitv2_corpus_dump")]
    dump_corpus_snapshot(page, req.offset, &words);

    let (instrs, non_empty) = analyzer.walk_bounded(&words, req.offset, phys_base, max_instrs_per_compile());
    if !non_empty {
        page.denylist(offset); // entry offset itself is excluded (§6.4)
        #[cfg(feature = "developer")]
        stats.record_reject(crate::jitv2::RejectReason::EntryExcluded);
        return false;
    }

    let instr_count = crate::jitv2::analyzer::instrs_linear(instrs).count();
    if instr_count < min_instrs_to_compile() {
        // Too short to be worth the fixed per-compile cost — see
        // MIN_INSTRS_TO_COMPILE's own doc comment. Sticky-denylisted like
        // any other decline (§6.4): the region's instruction count can't
        // change without the page itself mutating (a gen bump, which clears
        // ENTRY_DENYLISTED alongside it), so there's nothing to gain from
        // re-evaluating this offset on a later arrival against the same
        // unchanged bytes.
        page.denylist(offset);
        #[cfg(feature = "developer")]
        stats.record_reject(crate::jitv2::RejectReason::TooShort);
        return false;
    }

    let mut instrs_owned = *instrs;
    // skip_entry_preamble=true: this compile is always reached from
    // mips_exec.rs's step() dispatch loop, which already ran the equivalent
    // IP7/pending-interrupt checks for req.offset's exact PC immediately
    // before ever arriving here — see compile_region's doc comment.
    let func = codegen.compile_region(&mut instrs_owned, req.offset, req.compiled_for_fr1, true);
    match func {
        Some(jit_fn) => {
            #[cfg(feature = "developer")]
            let code_size = codegen.last_code_size();
            #[cfg(not(feature = "developer"))]
            let code_size = 0;
            // Re-probe the cache before committing, exactly as `publish` re-checks
            // `gen_snap`: the pre-read probe above is only accurate for the instant
            // it ran, and the compile since then took real time. A store landing in
            // that window would otherwise be baked into a published region.
            //
            // The pair is exhaustive. A store made during the compile is either
            // still in a cache — its line is dirty, so this probe sees it — or it
            // reached RAM, which only happens via `writeback_l1d_line`/
            // `writeback_l2_line`, both of which drain through
            // `BusDevice::write_block`, which bumps the page generation, which
            // `publish`'s own `gen_snap` check then rejects. There is no third
            // state, and a partial writeback bumps the counter just the same.
            #[cfg(not(feature = "tcache"))]
            if crate::jitv2::jit_page_has_dirty_lines(phys_base as u64) {
                #[cfg(feature = "developer")]
                stats.record_reject(crate::jitv2::RejectReason::PageDirtyInCache);
                return false;
            }
            page.publish(offset, jit_fn as *const (), gen_snap, instr_count, code_size, req.compiled_for_fr1);
            #[cfg(feature = "developer")]
            {
                // Per-word block-overlap diagnostic (j2 pcp): every word this
                // region visited gets its saturating include-count bumped, so
                // a word covered by several overlapping regions reads high.
                // Also count fallback words / regions (j2 status) — the direct
                // confirmation that the fallback path actually ran.
                let mut fb = 0u64;
                for instr in crate::jitv2::analyzer::instrs_linear(&instrs_owned) {
                    page.note_block_include(instr.word as usize);
                    if instr.is_fallback { fb += 1; }
                }
                if fb > 0 {
                    stats.fallback_words.fetch_add(fb, std::sync::atomic::Ordering::Relaxed);
                    stats.fallback_regions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                stats.compiles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        None if codegen.last_compile_ran_out_of_memory() => {
            // Not a real decline — the offset is perfectly compilable, the
            // shared arena is just out of room right now (see
            // `Codegen::last_compile_ran_out_of_memory`'s doc comment).
            // Denylisting it would be wrong (permanently, until an
            // unrelated future flush happens to clear ENTRY_DENYLISTED too);
            // instead, leave it un-denylisted so it retries naturally on
            // its own next arrival, and tell the caller to flush — it has
            // the `Jitv2`/CPU-pause machinery this function doesn't.
            #[cfg(feature = "developer")]
            stats.failed_compiles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }
        None => {
            page.denylist(offset); // some visited instruction has no emitter yet, or a real Cranelift verifier rejection
            #[cfg(feature = "developer")]
            {
                let reason = if codegen.last_decline_was_verifier_error() {
                    crate::jitv2::RejectReason::CraneliftVerifierError
                } else {
                    crate::jitv2::RejectReason::AnalyzerCodegenDisagreement
                };
                stats.record_reject(reason);
            }
        }
    }
    false
}

/// How many of this worker's own compiles are sitting gap-blocked in the
/// shared arena's seal queue right now (fully compiled, finalized, and
/// patched with a real `PublishInfo` — see `paged_memory::PublishInfo`'s own
/// doc comment — just not yet sealed because an earlier-allocated range
/// elsewhere hasn't been patched in yet). Not a `Vec` of entries: once an
/// entry is pushed/patched into the shared seal queue
/// (`Codegen::finalize_batch_nonforced`), it is fully self-describing there
/// — nothing about *which* entry is still needed here, only a rough count to
/// decide whether `publish_ready_nonforced`/`force_publish_pending` are
/// worth calling at all. A plain counter can also be conservative in the
/// harmless direction: `publish_sealed`-derived saturating decrements mean
/// it can undercount slightly (a sweep triggered by *this* worker's own
/// gap-blocked entry can incidentally also unblock and publish some *other*
/// worker's entries, which never incremented this counter — see
/// `publish_all`'s own doc comment) but never overcounts into a permanent
/// "stuck thinking there's still something pending" state, since it only
/// ever grows by exactly 1 per genuinely gap-blocked compile.
pub type PendingCount = usize;

/// Deferred counterpart to `handle_request`: identical compile-from-snapshot
/// protocol (walk, codegen) and identical *finalize* timing — this calls
/// `Codegen::finalize_batch_nonforced` for its own single `FuncId`
/// immediately after compiling it, same as `handle_request` does via
/// `compile_region`'s own `finalize_batch` call. The only difference from
/// `handle_request` is what "finalize" *means* here: non-forced, so it
/// doesn't necessarily seal the page immediately (see `paged_memory`'s
/// module doc comment) — this call's own return value decides whether this
/// offset (and possibly others, unblocked as a side effect) is dispatchable
/// right now (published immediately, same as `handle_request`) or still
/// gap-blocked behind another worker's not-yet-finalized range elsewhere in
/// the shared arena (`*pending` incremented by one, for the caller to retry
/// via `publish_ready_nonforced`/`force_publish_pending` on a later tick —
/// see `worker_loop`'s own doc comment). Every other outcome (excluded
/// entry, bus-read failure, OOM, real codegen decline) is handled exactly
/// like `handle_request` — those don't produce a `FuncId` to finalize at
/// all, so there's nothing deferred-specific about them.
///
/// Finalizing per-compile like this (rather than accumulating many compiles
/// into one big batch before ever finalizing any of them) is deliberate: an
/// earlier version of this function accumulated a whole worker's compiles
/// into `pending` and only finalized the batch on an external trigger
/// (arena page-crossing, or the queue going empty) — under real multi-worker
/// concurrency, a worker that stays continuously busy (queue never goes
/// empty for it) could accumulate dozens of compiles with none of them ever
/// finalized, permanently blocking every OTHER worker's later-allocated
/// ranges from ever sealing (their contiguous-prefix scan can never get
/// past this worker's still-unfinalized gap) — confirmed live as a genuine
/// stuck-forever compile-pool bug under real N>1 contention. Finalizing
/// immediately, every time, removes that failure mode entirely: the only
/// thing ever deferred now is the mprotect/seal itself, which is exactly
/// what `paged_memory`'s watermark design is built to tolerate no matter
/// how long it's deferred.
///
/// Returns the same `bool` as `handle_request` (arena-OOM signal) for the
/// same reason.
pub fn handle_request_deferred(
    req: &CompileRequest,
    bus: &Arc<dyn BusDevice>,
    analyzer: &mut Analyzer,
    codegen: &mut Codegen,
    pending: &mut PendingCount,
    #[cfg(feature = "developer")] stats: &crate::jitv2::JitStats,
) -> bool {
    let page = unsafe { &*req.page };
    let offset = req.offset as usize;

    struct ClearScheduledOnDrop<'a> { page: &'a crate::jitv2::PhysicalCodePage, offset: usize }
    impl Drop for ClearScheduledOnDrop<'_> {
        fn drop(&mut self) { self.page.clear_scheduled(self.offset); }
    }
    let _clear_scheduled = ClearScheduledOnDrop { page, offset };

    if page.is_denylisted(offset) || page.is_runnable(offset) {
        return false;
    }

    let phys_base = page.pfn * PAGE_SIZE;
    let gen_snap = page.current_gen();

    // Dirty-cache gate. The snapshot below reads RAM off the bus; the guest
    // CPU sees RAM overlaid with its own dirty L1-D/L2 lines, and a store
    // that retires inside the cache bumps no generation counter. Compiling
    // through that would publish pre-store bytes as a valid region. Abort
    // instead and publish nothing — nothing is denylisted, so this offset is
    // retried on its next arrival, by which point the lines have usually
    // aged out to RAM on their own.
    //
    // Checked *before* the 4KB bus read: there is no point paying for a
    // snapshot we are about to discard. The probe is racy by design and can
    // miss a store that lands after it looks — see
    // `jitv2::jit_page_has_dirty_lines` for why that is strictly better than
    // the status quo rather than a new hazard.
    //
    // Compiled out entirely under `tcache`: there the cache reads and writes
    // RAM through the ppmem window, so a store is in RAM the moment it
    // retires and there is no hidden dirty data for the probe to find. The
    // executor installs no probe in that build, so this would be a global
    // load that can only ever answer false.
    #[cfg(not(feature = "tcache"))]
    if crate::jitv2::jit_page_has_dirty_lines(phys_base as u64) {
        #[cfg(feature = "developer")]
        stats.record_reject(crate::jitv2::RejectReason::PageDirtyInCache);
        return false;
    }

    let mut words = [0u32; ENTRIES_PER_PAGE];
    for (i, w) in words.iter_mut().enumerate() {
        let r = bus.read32(phys_base + (i as u32) * 4);
        if !r.is_ok() {
            return false;
        }
        *w = r.data;
    }

    #[cfg(feature = "jitv2_corpus_dump")]
    dump_corpus_snapshot(page, req.offset, &words);

    let (instrs, non_empty) = analyzer.walk_bounded(&words, req.offset, phys_base, max_instrs_per_compile());
    if !non_empty {
        page.denylist(offset);
        #[cfg(feature = "developer")]
        stats.record_reject(crate::jitv2::RejectReason::EntryExcluded);
        return false;
    }

    let instr_count = crate::jitv2::analyzer::instrs_linear(instrs).count();
    if instr_count < min_instrs_to_compile() {
        page.denylist(offset);
        #[cfg(feature = "developer")]
        stats.record_reject(crate::jitv2::RejectReason::TooShort);
        return false;
    }

    let mut instrs_owned = *instrs;
    // compile_region_uncommitted reads is_entry_point off the buffer itself
    // (§13.4 — codegen's replacement for a separate entry_word scalar) —
    // this path only ever has one entry point, so mark it directly rather
    // than going through Analyzer::walk_multi_entry's multi-offset API.
    // has_fpu is likewise now a caller-computed input rather than something
    // codegen scans for internally — same one-pass check compile_region's
    // own wrapper uses.
    instrs_owned[offset].is_entry_point = true;
    let has_fpu = crate::jitv2::analyzer::instrs_linear(&instrs_owned).any(|i| crate::jitv2::analyzer::is_fpu_instruction(i.raw));
    let func_id = codegen.compile_region_uncommitted(&mut instrs_owned, req.compiled_for_fr1, true, has_fpu, req.page);
    match func_id {
        Some(func_id) => {
            #[cfg(feature = "developer")]
            let code_size = codegen.last_code_size();
            #[cfg(not(feature = "developer"))]
            let code_size = 0;
            // Finalize immediately, every compile — see this function's own
            // doc comment for why deferring finalize itself (rather than
            // just the seal) was the actual bug. finalize_batch_nonforced
            // patches this entry's PublishInfo into the seal-queue slot
            // compile_region_uncommitted already reserved for it
            // (push_placeholder) and tries to seal — the returned entries
            // may include this one (the common case — nothing else is
            // blocking it) and/or other workers' own entries that this
            // seal attempt happened to unblock too; every entry returned
            // here is fully self-describing (paged_memory::PublishInfo's
            // own doc comment) and gets published unconditionally,
            // regardless of whose compile produced it. If nothing came
            // back at all, this range is itself gap-blocked behind an
            // earlier, still-unpatched entry — retry later via the
            // idle-timeout/pending-threshold sweep.
            let publish = crate::jitv2::paged_memory::PublishInfo {
                page: req.page, offset, gen_snap, instr_count, code_size,
                compiled_for_fr1: req.compiled_for_fr1,
                jit_fn: None,
            };
            let sealed = codegen.finalize_batch_nonforced(func_id, publish);
            if sealed.is_empty() {
                *pending += 1;
            } else {
                publish_all(&sealed);
            }
            #[cfg(feature = "developer")]
            {
                // Per-word block-overlap diagnostic (j2 pcp) — counted at
                // compile time regardless of whether publish happened
                // immediately or was deferred. Safe to touch `page` here in
                // either case — the include-count is an independent
                // diagnostic field with no gen/validity contract. Also count
                // fallback words/regions and the successful-compile total —
                // same bookkeeping `handle_request`'s own success arm does,
                // needed here too since `j2 status`/`j2 stats` read these
                // process-wide counters regardless of which path compiled.
                let mut fb = 0u64;
                for instr in crate::jitv2::analyzer::instrs_linear(&instrs_owned) {
                    page.note_block_include(instr.word as usize);
                    if instr.is_fallback { fb += 1; }
                }
                if fb > 0 {
                    stats.fallback_words.fetch_add(fb, std::sync::atomic::Ordering::Relaxed);
                    stats.fallback_regions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                stats.compiles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        None if codegen.last_compile_ran_out_of_memory() => {
            #[cfg(feature = "developer")]
            stats.failed_compiles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }
        None => {
            page.denylist(offset);
            #[cfg(feature = "developer")]
            {
                let reason = if codegen.last_decline_was_verifier_error() {
                    crate::jitv2::RejectReason::CraneliftVerifierError
                } else {
                    crate::jitv2::RejectReason::AnalyzerCodegenDisagreement
                };
                stats.record_reject(reason);
            }
        }
    }
    false
}

/// Publish every entry a seal sweep just returned, unconditionally — each
/// `PublishInfo` is fully self-describing (see its own doc comment), so
/// there's no `FuncId`/`Codegen` lookup needed here, and no reason to
/// filter: every entry `try_seal_ready`/`try_seal_ready_forced` hands back
/// is, by construction, freshly sealed and ready to dispatch. May include
/// entries pushed by a *different* worker than the one that triggered this
/// particular sweep — that's fine and expected (see this function's own
/// callers' doc comments).
fn publish_all(sealed: &[crate::jitv2::paged_memory::PublishInfo]) {
    for entry in sealed {
        let page = unsafe { &*entry.page };
        let jit_fn = entry.jit_fn.expect("publish_all: a sealed entry must always carry a resolved JitFn");
        // Same pre-publish re-probe as the inline path in `handle_request` —
        // see the comment there for why this plus `publish`'s `gen_snap`
        // check covers every case. Deferred entries sat in the seal queue for
        // even longer than an inline compile, so the window is wider here.
        #[cfg(not(feature = "tcache"))]
        if crate::jitv2::jit_page_has_dirty_lines((page.pfn * PAGE_SIZE) as u64) {
            continue;
        }
        page.publish(entry.offset, jit_fn as *const (), entry.gen_snap, entry.instr_count, entry.code_size, entry.compiled_for_fr1);
    }
}

/// Non-forced retry sweep for the async worker's idle-timeout scheme (see
/// `worker_loop`'s own doc comment): re-attempts sealing whatever the
/// shared arena's seal queue currently has queued (this worker's own
/// gap-blocked entries — see `PendingCount`'s own doc comment — or another
/// worker's) and publishes everything that comes back. Decrements `*pending`
/// by how many entries this sweep actually published, saturating at 0 since
/// a sweep triggered by this worker's own gap-blocked entry can incidentally
/// publish other workers' entries too, which never incremented this
/// worker's own counter.
pub fn publish_ready_nonforced(codegen: &mut Codegen, pending: &mut PendingCount) {
    if *pending == 0 {
        return;
    }
    let sealed = codegen.try_seal_ready();
    *pending = pending.saturating_sub(sealed.len());
    publish_all(&sealed);
}

/// Idle-timeout counterpart: force-seal whatever the shared arena's seal
/// queue still has queued without waiting for a further contiguous-prefix
/// match — see `Codegen::force_seal_pending`'s own doc comment. Same
/// saturating-decrement/publish-everything contract as
/// `publish_ready_nonforced`.
pub fn force_publish_pending(codegen: &mut Codegen, pending: &mut PendingCount) {
    if *pending == 0 {
        return;
    }
    let sealed = codegen.force_seal_pending();
    *pending = pending.saturating_sub(sealed.len());
    publish_all(&sealed);
}

#[cfg(feature = "jitv2_corpus_dump")]
pub const CORPUS_DIR: &str = "jitv2_corpus";

#[cfg(feature = "jitv2_corpus_dump")]
fn dump_corpus_snapshot(page: &crate::jitv2::PhysicalCodePage, offset: u16, words: &[u32; ENTRIES_PER_PAGE]) {
    use std::io::Write;

    if !page.mark_saved(offset as usize) {
        return; // already dumped for this (pfn, offset)
    }
    let out_dir = std::path::Path::new(CORPUS_DIR);
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("jitv2 corpus: failed to create {}: {}", CORPUS_DIR, e);
        return;
    }
    let path = out_dir.join(format!("pfn_{:08x}_off_{:04x}.bin", page.pfn, offset));
    let write = || -> std::io::Result<()> {
        let mut f = std::fs::File::create(&path)?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words))
        };
        f.write_all(bytes)
    };
    if let Err(e) = write() {
        eprintln!("jitv2 corpus: failed to write snapshot for pfn={:#010x} offset={:#06x}: {}", page.pfn, offset, e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jitv2::PhysicalCodePage;
    use crate::mips_isa::{OP_ADDIU, OP_SPECIAL};
    use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_ERR};
    use std::sync::atomic::AtomicU64;

    /// Test-only wrapper hiding `handle_request`'s `developer`-only `stats`
    /// parameter — every test below just wants "run a request," not the
    /// stats side effects, so a fresh throwaway `JitStats` under `developer`
    /// keeps every call site identical regardless of which build this runs
    /// under (`--features jitv2` alone must build and pass exactly like
    /// `--features jitv2,developer` does).
    fn handle_request_for_test(req: &CompileRequest, bus: &Arc<dyn BusDevice>, analyzer: &mut Analyzer, codegen: &mut Codegen) -> bool {
        #[cfg(feature = "developer")]
        {
            let stats = crate::jitv2::JitStats::default();
            handle_request(req, bus, analyzer, codegen, &stats)
        }
        #[cfg(not(feature = "developer"))]
        {
            handle_request(req, bus, analyzer, codegen)
        }
    }

    /// Every word decodes as `ADDIU r1, r0, 1` (opcode 0x09, rs=0, rt=1,
    /// imm=1) — a plain Sequential instruction with a real codegen emitter,
    /// so `handle_request` can exercise the whole walk+compile+publish path
    /// without needing a real memory device.
    struct AddiuDevice;
    impl BusDevice for AddiuDevice {
        fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u32, _val: u8) -> u32 { BUS_ERR }
        fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u32, _val: u16) -> u32 { BUS_ERR }
        fn read32(&self, _addr: u32) -> BusRead32 {
            BusRead32::ok((OP_ADDIU << 26) | (1 << 16) | 1)
        }
        fn write32(&self, _addr: u32, _val: u32) -> u32 { BUS_ERR }
        fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u32, _val: u64) -> u32 { BUS_ERR }
        fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 { std::ptr::null() }
    }

    /// Every word decodes as the JIT region-boundary sentinel — a genuine
    /// zero-instruction region (analyzer `Classify::RegionBoundary`), used to
    /// exercise `handle_request`'s denylist exit path deterministically. (An
    /// excluded instruction like SYSCALL no longer works for this: with
    /// interpreter-fallback it's kept in the region as a fallback head and
    /// compiles/publishes successfully rather than denylisting — the sentinel is
    /// now the only thing that reliably yields an empty region.)
    struct BoundaryDevice;
    impl BusDevice for BoundaryDevice {
        fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u32, _val: u8) -> u32 { BUS_ERR }
        fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u32, _val: u16) -> u32 { BUS_ERR }
        fn read32(&self, _addr: u32) -> BusRead32 {
            BusRead32::ok(crate::mips_isa::JIT_REGION_BOUNDARY_SENTINEL)
        }
        fn write32(&self, _addr: u32, _val: u32) -> u32 { BUS_ERR }
        fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u32, _val: u64) -> u32 { BUS_ERR }
        fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 { std::ptr::null() }
    }

    struct ErrDevice;
    impl BusDevice for ErrDevice {
        fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u32, _val: u8) -> u32 { BUS_ERR }
        fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u32, _val: u16) -> u32 { BUS_ERR }
        fn read32(&self, _addr: u32) -> BusRead32 { BusRead32::err() }
        fn write32(&self, _addr: u32, _val: u32) -> u32 { BUS_ERR }
        fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u32, _val: u64) -> u32 { BUS_ERR }
        fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 { std::ptr::null() }
    }

    #[test]
    fn handle_request_publishes_a_compilable_instruction() {
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 4, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);

        assert!(page.is_runnable(4), "a plain ADDIU must compile and publish");
        assert!(!page.is_denylisted(4));
    }

    /// The dirty-page gate must actually be wired into *this* design's compile
    /// path — not merely exist somewhere in the file.
    ///
    /// This exists because it was originally missed: both gates were placed in
    /// `old_impl`, leaving `j2wp` builds with no probe at all. Nothing caught
    /// it — every unit test passed, IRIX booted, and the only symptom was
    /// `PageDirtyInCache` silently reading zero in `j2 stats` forever. A test
    /// that drives the real entry point with a probe forced to "dirty" fails
    /// loudly in whichever design forgot to consult it.
    #[cfg(not(feature = "tcache"))]
    #[test]
    fn a_dirty_page_is_never_compiled() {
        let _g = dirty_probe_lock();

        // Sanity first: with no probe installed this exact request publishes.
        // Without this the test could pass for the wrong reason (e.g. the page
        // was never compilable to begin with).
        crate::jitv2::clear_jit_page_probe();
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 4, compiled_for_fr1: true };
        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
        assert!(page.is_runnable(4),
            "precondition: this request must publish when no probe vetoes it");

        // Now the real assertion: a probe reporting the page dirty must stop
        // the compile before anything is published.
        fn always_dirty(_ctx: *const (), _page_base: u64) -> bool { true }
        static ANCHOR: u8 = 0;
        // SAFETY: `always_dirty` ignores ctx; ANCHOR is a 'static non-null
        // stand-in so the installed-probe check sees a live pointer.
        unsafe { crate::jitv2::install_jit_page_probe(&ANCHOR as *const u8 as *const (), always_dirty) };

        let counter2 = AtomicU64::new(0);
        let mut page2 = PhysicalCodePage::new(0, &counter2 as *const AtomicU64);
        let req2 = CompileRequest { page: &mut page2 as *mut PhysicalCodePage, offset: 4, compiled_for_fr1: true };
        handle_request_for_test(&req2, &bus, &mut analyzer, &mut codegen);

        crate::jitv2::clear_jit_page_probe();

        assert!(!page2.is_runnable(4),
            "a page reported dirty in the CPU cache must not publish — the compile would be built \
             from a stale RAM snapshot (this design's compile path is not consulting the probe)");
        // The abort must be retryable, not sticky: the offset gets another
        // chance once the cache lines drain to RAM on their own.
        assert!(!page2.is_denylisted(4),
            "a dirty-page abort must leave the offset eligible for a later retry, not denylist it");
    }

    /// Exclusion for the probe global. Delegates to `jitv2::probe_test_lock`
    /// rather than owning a mutex here: `mips_cache_v2`'s probe tests install
    /// into the same global, and two independent locks would exclude nothing.
    #[cfg(not(feature = "tcache"))]
    fn dirty_probe_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::jitv2::jitv2::probe_test_lock()
    }

    #[test]
    fn handle_request_skips_already_valid_entry() {
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
        assert!(page.is_runnable(0));

        // Bump gen so a naive re-compile would behave differently if it ran;
        // since is_runnable is already true, handle_request must bail
        // before touching the bus again.
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
        assert!(page.is_runnable(0), "gen bump alone must not un-publish without going through it");
    }

    #[test]
    fn handle_request_leaves_offset_undecided_on_bus_error() {
        let bus: Arc<dyn BusDevice> = Arc::new(ErrDevice);
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);

        assert!(!page.is_runnable(0));
        assert!(!page.is_denylisted(0), "a transient read failure must not become a sticky rejection");
    }

    /// Regression test: `exec_decoded`'s dispatch gate sets `ENTRY_SCHEDULED`
    /// (via `try_schedule`) before sending a `CompileRequest` to stop a hot
    /// PC from flooding the queue with duplicate requests for the same
    /// offset. `handle_request` must clear that flag again once it decides —
    /// on every exit path, not just the success path — or the offset can
    /// never be scheduled again after this one request, even once it's
    /// fully resolved. Covers all three decision outcomes: publish success,
    /// sticky denylist, and "bus not readable, retry later."
    #[test]
    fn handle_request_clears_scheduled_bit_on_every_outcome() {
        let counter = AtomicU64::new(0);

        // Outcome 1: publishes successfully.
        {
            let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
            let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
            assert!(page.try_schedule(4));
            let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 4, compiled_for_fr1: true };
            let mut analyzer = Analyzer::new();
            let mut codegen = Codegen::new();
            handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
            assert!(page.is_runnable(4));
            assert!(page.try_schedule(4), "scheduled bit must be cleared after a successful publish, so a future recompile request isn't blocked");
        }

        // Outcome 2: sticky-denylisted (the region-boundary sentinel yields a
        // zero-instruction region — the only reliable empty-region case now
        // that excluded instructions compile as interpreter-fallback heads).
        {
            let bus: Arc<dyn BusDevice> = Arc::new(BoundaryDevice);
            let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
            assert!(page.try_schedule(0));
            let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true };
            let mut analyzer = Analyzer::new();
            let mut codegen = Codegen::new();
            handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
            assert!(page.is_denylisted(0));
            assert!(page.try_schedule(0), "scheduled bit must be cleared after a denylist decision");
        }

        // Outcome 3: bus read fails, offset left undecided ("retry later").
        {
            let bus: Arc<dyn BusDevice> = Arc::new(ErrDevice);
            let mut page = PhysicalCodePage::new(0, std::ptr::null());
            assert!(page.try_schedule(0));
            let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true };
            let mut analyzer = Analyzer::new();
            let mut codegen = Codegen::new();
            handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
            assert!(!page.is_runnable(0));
            assert!(!page.is_denylisted(0));
            assert!(page.try_schedule(0), "scheduled bit must be cleared even on a transient bus-read failure, so the offset can be retried");
        }
    }

    /// Same `stats`-hiding wrapper as `handle_request_for_test`, for the
    /// deferred path.
    fn handle_request_deferred_for_test(req: &CompileRequest, bus: &Arc<dyn BusDevice>, analyzer: &mut Analyzer, codegen: &mut Codegen, pending: &mut PendingCount) -> bool {
        #[cfg(feature = "developer")]
        {
            let stats = crate::jitv2::JitStats::default();
            handle_request_deferred(req, bus, analyzer, codegen, pending, &stats)
        }
        #[cfg(not(feature = "developer"))]
        {
            handle_request_deferred(req, bus, analyzer, codegen, pending)
        }
    }

    #[test]
    fn handle_request_deferred_leaves_a_lone_small_compile_pending_until_its_page_seals() {
        // Non-forced sealing never seals a still-open page — even with a
        // solo arena and exactly one compile, nothing else will ever grow
        // `position` into a new page on its own (see
        // Codegen::finalize_batch_nonforced_does_not_seal_a_still_open_page's
        // own test, at the arena level). handle_request_deferred must
        // therefore leave this offset pending, exactly as the idle-timeout/
        // force-seal machinery (force_publish_pending) expects to find it —
        // this isn't "gap-blocked behind another worker," just "not yet a
        // whole page's worth."
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 4, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        let mut pending: PendingCount = 0;
        handle_request_deferred_for_test(&req, &bus, &mut analyzer, &mut codegen, &mut pending);

        assert_eq!(pending, 1, "a lone small compile's page is still open — non-forced sealing must not publish it yet");
        assert!(!page.is_runnable(4), "must not be published until something actually seals its page");

        force_publish_pending(&mut codegen, &mut pending);
        assert_eq!(pending, 0, "the idle-timeout force-seal sweep must finish the job");
        assert!(page.is_runnable(4), "now published, once forced");
    }

    #[test]
    fn handle_request_deferred_denylists_empty_region_entry_without_accumulating() {
        let bus: Arc<dyn BusDevice> = Arc::new(BoundaryDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 0, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        let mut pending: PendingCount = 0;
        handle_request_deferred_for_test(&req, &bus, &mut analyzer, &mut codegen, &mut pending);

        assert_eq!(pending, 0, "an empty-region (boundary) entry produces no FuncId to defer");
        assert!(page.is_denylisted(0));
    }

    #[test]
    fn handle_request_deferred_clears_scheduled_bit() {
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        assert!(page.try_schedule(4));
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, offset: 4, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        let mut pending: PendingCount = 0;
        handle_request_deferred_for_test(&req, &bus, &mut analyzer, &mut codegen, &mut pending);

        assert!(page.try_schedule(4), "scheduled bit must be cleared even though publish itself is deferred");
    }
}

}
#[cfg(not(feature = "j2wp"))]
pub use old_impl::*;

#[cfg(feature = "j2wp")]
mod new_impl {
//! JIT v2 compiler-thread logic. See `rules/jitv2/jit-v2-design.md`.
//!
//! `handle_request` is the compile-from-snapshot protocol (§6.5): read the
//! page's 4KB snapshot off the bus, bounded-walk it from the requested
//! offset, hand the walked region to `Codegen::compile_region`, and publish
//! the result (or sticky-deny the offset if codegen declined it). Bounded to
//! `MAX_INSTRS_PER_COMPILE` *head* instructions — started at 1 (the smallest
//! possible working JIT) and growing incrementally from there, not a
//! redesign each time (the walk bound is just `max_instrs` on
//! `Analyzer::walk_bounded`). A branch/jump's mandatory delay slot is never
//! charged against this budget (`Analyzer::visit_slot` — a slot can never be
//! omitted, so it was never a truncation candidate) — including a nested
//! delay-slot chain (branch-in-delay-slot, "unusual but legal" on real
//! hardware), which keeps extending for free until it terminates or runs
//! off the page, at which point the walk declines the whole region rather
//! than compiling a partial chain.
//!
//! Corpus collection (raw page dump to `jitv2_corpus/`, used to develop the
//! analyzer/codegen offline against real captured pages) is preserved behind
//! the `jitv2_corpus_dump` feature — with it off, `handle_request` never
//! touches the filesystem.

use std::sync::Arc;

use crate::jitv2::analyzer::Analyzer;
use crate::jitv2::codegen::Codegen;
use crate::jitv2::{CompileRequest, ENTRIES_PER_PAGE, PAGE_SIZE};
use crate::traits::BusDevice;

/// Instruction budget for a compile-from-arrival region (see module doc):
/// head instructions only — a branch/jump's mandatory delay slot (or nested
/// slot-chain) is free and always included regardless of this number
/// (`Analyzer::visit_slot`). Raised incrementally from 1 (the original
/// "smallest possible working JIT" milestone) through 2/3/4/8, all booted
/// clean — a live `j2 status` histogram at 8 showed real regions landing
/// anywhere from 1 to 16 instructions (the tail past the nominal budget
/// comes from a branch's mandatory delay slot counting toward the total but
/// not the budget itself), clustering 8-11, with only a handful ever
/// reaching 16. Defaults to 128 — real regions cluster 8-11 instructions, so
/// this is a generous headroom rather than a tight cap, chosen to bound the
/// rare pathological case (long branch-free/self-chaining-delay-slot runs)
/// where an unbounded walk would otherwise let the analyzer grow one region
/// arbitrarily large, producing a single huge, slow-to-compile Cranelift
/// function and a single huge arena allocation for it. Any value at or
/// above `ENTRIES_PER_PAGE` (1024 words/page) is equivalent to no budget at
/// all, since the walk was always going to decline rather than compile a
/// region longer than fits on one physical page anyway (module doc: "runs
/// off the page... declines the whole region") — the page boundary, not
/// this constant, is the real ceiling past that point (same as
/// `Analyzer::walk`'s own unbounded case). `j2 max-instrs [N]` tunes this at
/// runtime (e.g. to shrink compiled regions further for debugging/bisection,
/// or raise it back toward `usize::MAX`); both `Analyzer::walk_bounded` and
/// `Codegen::compile_region` were already written generically against
/// `max_instrs` (fallthrough-edge wiring for a multi-instruction
/// straight-line region already exists, per `compile_region`'s Pass 2), so
/// this is just a config read, not a redesign.
static MAX_INSTRS_PER_COMPILE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(ENTRIES_PER_PAGE);

pub fn set_max_instrs_per_compile(n: usize) {
    MAX_INSTRS_PER_COMPILE.store(n.max(1), std::sync::atomic::Ordering::Relaxed);
}

pub fn max_instrs_per_compile() -> usize {
    MAX_INSTRS_PER_COMPILE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Minimum walked-region instruction count a compile must reach to actually
/// get compiled — regions shorter than this are sticky-denylisted instead,
/// same treatment as any other decline (§6.4). Below this floor, the
/// per-compile fixed overhead (Cranelift IR building + one arena allocation
/// + one entry_table publish) very likely costs more than the region will
/// ever save over just interpreting it: a true single-instruction region is
/// the worst case for this tradeoff and, per `MAX_INSTRS_PER_COMPILE`'s own
/// doc comment, real regions cluster 8-11 instructions anyway — a
/// one-instruction region is far more often either a rare cold path or an
/// analyzer/codegen edge case than a genuinely hot single-instruction loop
/// body worth paying compile cost for. `j2 min-instrs [N]` tunes this at
/// runtime. Applies identically to `handle_request` and
/// `handle_request_deferred` — both consult `min_instrs_to_compile()` right
/// after a successful, non-empty walk.
///
/// Defaults to 1 (no filtering) under `developer` — diagnostics builds want
/// to see and measure every compile the analyzer/codegen would otherwise
/// attempt, not have some silently skipped by a production-tuned floor — and
/// to 2 otherwise, since real-world usage wants a *little* filtering by
/// default rather than requiring a manual `j2 min-instrs` on every run.
static MIN_INSTRS_TO_COMPILE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_min_instrs_to_compile(n: usize) {
    MIN_INSTRS_TO_COMPILE.store(n, std::sync::atomic::Ordering::Relaxed);
}

pub fn min_instrs_to_compile() -> usize {
    MIN_INSTRS_TO_COMPILE.load(std::sync::atomic::Ordering::Relaxed)
}

/// §13.3 compile-from-snapshot protocol, multi-entry (§13.10 build-order
/// step 2): the request carries no offset — snapshot the page's whole
/// `requested` bitmap fresh at dequeue time, seqlock-snapshot the page bytes
/// (gen-before, copy, gen-after — retry on mismatch, so the bytes this
/// compile analyzes are provably coherent with the gen they get tagged
/// with), walk every still-eligible requested offset in one merged pass
/// (`Analyzer::walk_multi_entry`), compile, and publish the union. Any
/// requested offset that turns out excluded-at-entry or produces a region
/// below the instruction floor is sticky-denylisted individually — it does
/// not fail the whole compile, the same way a single bad offset never used
/// to fail a single-entry request either.
///
/// Returns `true` iff `codegen`'s arena ran out of memory on this call
/// (`Codegen::last_compile_ran_out_of_memory`) — the caller (`worker_loop`
/// or the inline dispatch path) owns the `Jitv2`/CPU-pause machinery this
/// function doesn't have access to, so it's responsible for actually
/// flushing when this is `true`. `false` for every other outcome (success,
/// a real codegen decline, or a transient bus-read failure).
/// Outcome of [`prepare_multi_entry_compile`]: either an early-decided
/// `bool` (the caller should return it directly — every one of
/// `handle_request`'s/`handle_request_deferred`'s old early-return cases,
/// now shared) or a walked, ready-to-compile region living in `analyzer`
/// (read back via `Analyzer::covered`/`Analyzer::has_fpu`, per the "state
/// lives on self" design — see `Analyzer`'s own doc comment) plus the
/// `gen_snap`/`instr_count` the eventual `publish` call needs.
enum PrepareOutcome {
    Done(bool),
    Ready { gen_snap: u64, instr_count: usize },
}

/// §13.3 steps 1-5, shared front half of `handle_request`/
/// `handle_request_deferred`: seqlock-snapshot the page bytes, snapshot
/// `requested`, pre-compile subsumption check, collect still-eligible
/// candidates, walk them all via `Analyzer::walk_multi_entry`, sticky-deny
/// any candidate excluded at its own entry, and check the merged region's
/// instruction floor. Everything up to (not including) the actual
/// `compile_region_uncommitted`/publish call — the two callers diverge only
/// there (immediate `finalize_batch` + direct publish vs. deferred
/// `finalize_batch_nonforced` + seal-queue publish).
fn prepare_multi_entry_compile(
    req: &CompileRequest,
    bus: &Arc<dyn BusDevice>,
    analyzer: &mut Analyzer,
    #[cfg(feature = "developer")] stats: &crate::jitv2::JitStats,
) -> PrepareOutcome {
    let page = unsafe { &*req.page };
    let phys_base = page.pfn * PAGE_SIZE;

    // Dirty-cache gate. The seqlock snapshot below reads RAM off the bus; the
    // guest CPU sees RAM overlaid with its own dirty L1-D/L2 lines, and a
    // store that retires inside the cache bumps no generation counter — so
    // the seqlock cannot see it either, and neither can publish's gen
    // re-check. This is the one mutation class those mechanisms structurally
    // miss.
    //
    // The case it exists for: IRIX loads an executable, relocates it in
    // place, and jumps straight to it while the relocated words are still
    // dirty in L2. The interpreter runs correctly out of the cache, but the
    // execution triggers compiles that read RAM and get the *pre-relocation*
    // bytes straight from disk — stale addresses baked into compiled code,
    // crashing 4Dwm/fm at startup. It clears "on its own" once writeback puts
    // the relocations in RAM and a flush forces recompiles, which is exactly
    // why `j2 flush` appears to fix it.
    //
    // Abort and publish nothing; nothing is denylisted, so these offsets are
    // retried on their next arrival, by which point the lines have usually
    // drained. Checked *before* the 4KB read: no point paying for a snapshot
    // we are about to discard.
    //
    // Compiled out entirely under `tcache`: there the cache reads and writes
    // RAM through the ppmem window, so a store is in RAM the moment it
    // retires and there is nothing hidden for the probe to find.
    #[cfg(not(feature = "tcache"))]
    if crate::jitv2::jit_page_has_dirty_lines(phys_base as u64) {
        #[cfg(feature = "developer")]
        stats.record_reject(crate::jitv2::RejectReason::PageDirtyInCache);
        page.mark_prepare_bounced();
        return PrepareOutcome::Done(false);
    }

    // §13.3 step 2: seqlock-snapshot the page bytes. Bounded retries — a
    // page under sustained concurrent SMC just keeps losing the race to
    // itself, correctly punted to the next arrival's request rather than
    // compiled against torn bytes.
    const MAX_SEQLOCK_RETRIES: u32 = 8;
    let mut words = [0u32; ENTRIES_PER_PAGE];
    let mut gen_snap = 0u64;
    let mut snapshot_ok = false;
    for _ in 0..MAX_SEQLOCK_RETRIES {
        let gen0 = page.current_gen();
        let mut read_ok = true;
        for (i, w) in words.iter_mut().enumerate() {
            let r = bus.read32(phys_base + (i as u32) * 4);
            if !r.is_ok() {
                read_ok = false;
                break;
            }
            *w = r.data;
        }
        if !read_ok {
            page.mark_prepare_bounced();
            return PrepareOutcome::Done(false); // page not readable right now; retry later
        }
        let gen1 = page.current_gen();
        if gen0 == gen1 {
            gen_snap = gen1;
            snapshot_ok = true;
            break;
        }
    }
    if !snapshot_ok {
        // A sufficiently hot self-modifying/shared-DMA page can bounce here
        // EVERY single attempt (gen keeps changing faster than
        // MAX_SEQLOCK_RETRIES can outrun it) — confirmed live via
        // `schedule_attempts`/`sends_attempted` both nonzero on a page with
        // 0 `rejected_compiles`/`compiles_since_flush`/`sends_dropped_queue_full`:
        // every accepted send individually lost this race, not a queue drop.
        page.mark_prepare_bounced();
        return PrepareOutcome::Done(false); // page under sustained concurrent mutation; punt to a later request
    }

    // §13.3 step 3: snapshot `requested` (relaxed — under- or over-collecting
    // relative to a single instant is harmless, see the field's own doc).
    let requested_snapshot = page.snapshot_requested();

    // §13.3 step 4: pre-compile subsumption check — nothing new to offer
    // against the CURRENT generation, don't even compile. Mirrors publish's
    // own subsumption check (§13.3 step 6/PhysicalCodePage::publish's doc
    // comment): `compiled` bits only mean "already covered" when they were
    // published for THIS SAME generation — a bit left over from an older,
    // now-superseded generation (page bytes changed since) doesn't actually
    // cover this offset against the new bytes, so the check must gate on
    // `entry_gen == gen_snap`, not just re-read `compiled` blindly. If the
    // page has moved past gen_snap already (a mutation raced this
    // snapshot), skip this check entirely: it's about to be stale
    // regardless, and the real gen-staleness abort happens at publish time
    // either way.
    let same_gen = page.current_gen() == gen_snap && page.entry_gen() == gen_snap;
    if same_gen && page.requested_subsumed_by_compiled(&requested_snapshot) {
        page.mark_prepare_bounced();
        return PrepareOutcome::Done(false);
    }

    // Collect every offset still eligible to compile: `requested` (this
    // round's fresh asks) unioned with `compiled` (re-included so a fresh
    // `func` re-covers every previously-published entry too — §13's
    // one-function-per-page model replaces `func` wholesale on every
    // compile, and its dispatch switch only recognizes the entries THIS
    // exact walk covers, codegen.rs's `compile_region` builds the switch
    // from nothing else — an old entry left out of a merged walk has no
    // case in the new `func` at all, and every future dispatch into it
    // would silently fall to EXEC_FALLBACK forever, confirmed live via a
    // boot traceback showing `[jit:entry]` immediately followed by a plain
    // interpreter re-execution of the very same instruction), masked by
    // `denied` (a denylisted offset stays denylisted without being
    // re-walked every time some OTHER offset on the same page gets
    // requested — see `snapshot_compile_candidates`'s own doc comment for
    // the exact bit algebra and the `include_compiled` gen guard).
    let candidate_bits = page.snapshot_compile_candidates(same_gen);
    let mut candidates: Vec<u16> = Vec::new();
    for word_idx in 0..crate::jitv2::BITMAP_WORDS {
        let bits = candidate_bits[word_idx];
        if bits == 0 { continue; }
        for bit in 0..64 {
            if bits & (1u64 << bit) == 0 { continue; }
            candidates.push((word_idx * 64 + bit) as u16);
        }
    }
    if candidates.is_empty() {
        return PrepareOutcome::Done(false); // everything requested is either already covered or already denylisted
    }

    #[cfg(feature = "jitv2_corpus_dump")]
    for &offset in &candidates {
        dump_corpus_snapshot(page, offset, &words);
    }

    // instr_count computed immediately, right after the walk: `instrs`
    // borrows `analyzer` mutably (it's `&self.instrs`), so it must be done
    // being used before any other `analyzer.*()` call (`covered()`, an
    // immutable borrow) can happen below.
    let instr_count = {
        let instrs = analyzer.walk_multi_entry(&words, &candidates, phys_base, max_instrs_per_compile());
        crate::jitv2::analyzer::instrs_linear(instrs).count()
    };

    // Per-offset declines: any candidate NOT in `covered` was excluded at
    // its own entry — sticky-denylist it individually, same as a
    // single-entry request always did, without failing the rest.
    //
    // Also `kill()` it — clear any stale `compiled` bit left over from an
    // earlier successful compile. `candidates` can now contain previously-
    // `compiled` offsets (folded in above so a fresh `func` re-covers them),
    // and a re-walk can legitimately exclude one this time (budget/shape
    // changed) even though it published fine before. Denylisting alone
    // isn't enough for those: `compiled`'s bit stays set from the earlier
    // publish, `is_runnable` keeps reporting it dispatchable, but this
    // compile's `func` has no case for it — the exact EXEC_FALLBACK-forever
    // shape the candidates fold-in was meant to fix in the first place, just
    // triggered by a denylist instead of a plain omission.
    for &offset in &candidates {
        if !analyzer.covered().contains(&offset) {
            page.denylist(offset as usize);
            page.kill(offset as usize);
            page.mark_analyze_rejected();
            #[cfg(feature = "developer")]
            stats.record_reject(crate::jitv2::RejectReason::EntryExcluded);
        }
    }
    if analyzer.covered().is_empty() {
        return PrepareOutcome::Done(false); // every candidate was excluded at its own entry
    }

    if instr_count < min_instrs_to_compile() {
        // Too short to be worth the fixed per-compile cost — see
        // MIN_INSTRS_TO_COMPILE's own doc comment. Denies every covered
        // entry: the region's instruction count is a property of the whole
        // merged walk, not any one entry, so if it's too short with all of
        // them included, it's too short full stop. `kill()` alongside for
        // the same reason as the loop above — some of `covered` may be
        // previously-`compiled` offsets whose stale bit must not survive.
        for &offset in analyzer.covered() {
            page.denylist(offset as usize);
            page.kill(offset as usize);
        }
        page.mark_analyze_rejected();
        #[cfg(feature = "developer")]
        stats.record_reject(crate::jitv2::RejectReason::TooShort);
        return PrepareOutcome::Done(false);
    }

    PrepareOutcome::Ready { gen_snap, instr_count }
}

/// §13.3 compile-from-snapshot protocol, multi-entry (§13.10 build-order
/// step 2): the request carries no offset — see [`prepare_multi_entry_compile`]
/// for the shared front half (snapshot/candidates/walk/instruction-floor
/// check); this function is the immediate-finalize back half: compile,
/// finalize right away, publish the union, clear the covered `requested`
/// bits on success.
///
/// Returns `true` iff `codegen`'s arena ran out of memory on this call
/// (`Codegen::last_compile_ran_out_of_memory`) — the caller (`worker_loop`
/// or the inline dispatch path) owns the `Jitv2`/CPU-pause machinery this
/// function doesn't have access to, so it's responsible for actually
/// flushing when this is `true`. `false` for every other outcome (success,
/// a real codegen decline, or a transient bus-read failure).
pub fn handle_request(
    req: &CompileRequest,
    bus: &Arc<dyn BusDevice>,
    analyzer: &mut Analyzer,
    codegen: &mut Codegen,
    #[cfg(feature = "developer")] stats: &crate::jitv2::JitStats,
) -> bool {
    let page = unsafe { &*req.page };

    // Clears the page-level in-flight flag on every return path, including
    // early returns and panics — `exec_decoded`'s dispatch gate (mips_exec.rs)
    // sets it via `try_schedule_page` before sending a request, specifically
    // so a hot page that keeps re-satisfying the gate's trigger conditions
    // doesn't flood the compile queue with duplicate requests while the
    // first one is still in flight. Whatever this function decides, the
    // flag must come back down afterward or this page can never be
    // (re-)requested again — a scope guard rather than clearing at each of
    // the several return points below so a future added early-return can't
    // forget it.
    struct ClearScheduledOnDrop<'a> { page: &'a crate::jitv2::PhysicalCodePage }
    impl Drop for ClearScheduledOnDrop<'_> {
        fn drop(&mut self) { self.page.clear_scheduled(); }
    }
    let _clear_scheduled = ClearScheduledOnDrop { page };

    #[cfg(feature = "developer")]
    let outcome = prepare_multi_entry_compile(req, bus, analyzer, stats);
    #[cfg(not(feature = "developer"))]
    let outcome = prepare_multi_entry_compile(req, bus, analyzer);
    let (gen_snap, instr_count) = match outcome {
        PrepareOutcome::Done(early) => return early,
        PrepareOutcome::Ready { gen_snap, instr_count } => (gen_snap, instr_count),
    };

    let mut instrs_owned = analyzer.instrs_snapshot();
    // skip_entry_preamble=true: this compile is always reached from
    // mips_exec.rs's step() dispatch loop, which already ran the equivalent
    // IP7/pending-interrupt checks for the live PC immediately before ever
    // arriving here — see compile_region_uncommitted's doc comment.
    let func_id = codegen.compile_region_uncommitted(&mut instrs_owned, req.compiled_for_fr1, true, analyzer.has_fpu(), req.page);
    match func_id {
        Some(func_id) => {
            let jit_fn = codegen.finalize_batch(&[func_id]).into_iter().next()
                .expect("finalize_batch of exactly one FuncId must return exactly one JitFn");
            #[cfg(feature = "developer")]
            let code_size = codegen.last_code_size();
            #[cfg(not(feature = "developer"))]
            let code_size = 0;
            let mut new_entries = [0u64; crate::jitv2::BITMAP_WORDS];
            for &offset in analyzer.covered() {
                new_entries[offset as usize >> 6] |= 1u64 << (offset % 64);
            }
            // Re-probe the cache before committing, exactly as `publish`
            // re-checks `gen_snap`: the probe in `prepare_multi_entry_compile`
            // is only accurate for the instant it ran, and the compile since
            // then took real time. A store landing in that window would
            // otherwise be baked into a published region.
            //
            // The pair is exhaustive. A store made during the compile is
            // either still in a cache — its line is dirty, so this probe sees
            // it — or it reached RAM, which only happens via
            // `writeback_l1d_line`/`writeback_l2_line`, both of which drain
            // through `BusDevice::write_block`, which bumps the page
            // generation, which `publish`'s own `gen_snap` check then rejects.
            // There is no third state, and a partial writeback bumps the
            // counter just the same.
            #[cfg(not(feature = "tcache"))]
            if crate::jitv2::jit_page_has_dirty_lines((page.pfn * PAGE_SIZE) as u64) {
                #[cfg(feature = "developer")]
                stats.record_reject(crate::jitv2::RejectReason::PageDirtyInCache);
                return false;
            }
            if page.publish(&new_entries, jit_fn as *const (), gen_snap, instr_count, code_size) {
                page.clear_requested_bits(&new_entries);
            }
            #[cfg(feature = "developer")]
            {
                let mut fb = 0u64;
                for instr in crate::jitv2::analyzer::instrs_linear(&instrs_owned) {
                    if instr.is_fallback { fb += 1; }
                }
                if fb > 0 {
                    stats.fallback_words.fetch_add(fb, std::sync::atomic::Ordering::Relaxed);
                    stats.fallback_regions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                stats.compiles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        None if codegen.last_compile_ran_out_of_memory() => {
            // Not a real decline — every candidate is perfectly compilable,
            // the shared arena is just out of room right now (see
            // `Codegen::last_compile_ran_out_of_memory`'s doc comment).
            // Denylisting would be wrong; instead, leave everything
            // un-denylisted so it retries naturally on a later arrival, and
            // tell the caller to flush — it has the `Jitv2`/CPU-pause
            // machinery this function doesn't. `mark_codegen_oom_bounced`
            // is what makes a page stuck repeatedly hitting exactly this
            // arm visible — see that field's own doc comment for why every
            // OTHER counter can legitimately read 0 while this is the real,
            // recurring cause.
            page.mark_codegen_oom_bounced();
            #[cfg(feature = "developer")]
            stats.failed_compiles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }
        None => {
            // Some visited instruction (from any covered entry's reachable
            // set) has no emitter yet, or a real Cranelift verifier
            // rejection — deny every covered entry, since the merged region
            // that failed included all of them.
            for &offset in analyzer.covered() {
                page.denylist(offset as usize);
            }
            page.mark_codegen_rejected();
            #[cfg(feature = "developer")]
            {
                let reason = if codegen.last_decline_was_verifier_error() {
                    crate::jitv2::RejectReason::CraneliftVerifierError
                } else {
                    crate::jitv2::RejectReason::AnalyzerCodegenDisagreement
                };
                stats.record_reject(reason);
            }
        }
    }
    false
}

/// How many of this worker's own compiles are sitting gap-blocked in the
/// shared arena's seal queue right now (fully compiled, finalized, and
/// patched with a real `PublishInfo` — see `paged_memory::PublishInfo`'s own
/// doc comment — just not yet sealed because an earlier-allocated range
/// elsewhere hasn't been patched in yet). Not a `Vec` of entries: once an
/// entry is pushed/patched into the shared seal queue
/// (`Codegen::finalize_batch_nonforced`), it is fully self-describing there
/// — nothing about *which* entry is still needed here, only a rough count to
/// decide whether `publish_ready_nonforced`/`force_publish_pending` are
/// worth calling at all. A plain counter can also be conservative in the
/// harmless direction: `publish_sealed`-derived saturating decrements mean
/// it can undercount slightly (a sweep triggered by *this* worker's own
/// gap-blocked entry can incidentally also unblock and publish some *other*
/// worker's entries, which never incremented this counter — see
/// `publish_all`'s own doc comment) but never overcounts into a permanent
/// "stuck thinking there's still something pending" state, since it only
/// ever grows by exactly 1 per genuinely gap-blocked compile.
pub type PendingCount = usize;

/// Deferred counterpart to `handle_request`: identical compile-from-snapshot
/// protocol (walk, codegen) and identical *finalize* timing — this calls
/// `Codegen::finalize_batch_nonforced` for its own single `FuncId`
/// immediately after compiling it, same as `handle_request` does via
/// `compile_region`'s own `finalize_batch` call. The only difference from
/// `handle_request` is what "finalize" *means* here: non-forced, so it
/// doesn't necessarily seal the page immediately (see `paged_memory`'s
/// module doc comment) — this call's own return value decides whether this
/// offset (and possibly others, unblocked as a side effect) is dispatchable
/// right now (published immediately, same as `handle_request`) or still
/// gap-blocked behind another worker's not-yet-finalized range elsewhere in
/// the shared arena (`*pending` incremented by one, for the caller to retry
/// via `publish_ready_nonforced`/`force_publish_pending` on a later tick —
/// see `worker_loop`'s own doc comment). Every other outcome (excluded
/// entry, bus-read failure, OOM, real codegen decline) is handled exactly
/// like `handle_request` — those don't produce a `FuncId` to finalize at
/// all, so there's nothing deferred-specific about them.
///
/// Finalizing per-compile like this (rather than accumulating many compiles
/// into one big batch before ever finalizing any of them) is deliberate: an
/// earlier version of this function accumulated a whole worker's compiles
/// into `pending` and only finalized the batch on an external trigger
/// (arena page-crossing, or the queue going empty) — under real multi-worker
/// concurrency, a worker that stays continuously busy (queue never goes
/// empty for it) could accumulate dozens of compiles with none of them ever
/// finalized, permanently blocking every OTHER worker's later-allocated
/// ranges from ever sealing (their contiguous-prefix scan can never get
/// past this worker's still-unfinalized gap) — confirmed live as a genuine
/// stuck-forever compile-pool bug under real N>1 contention. Finalizing
/// immediately, every time, removes that failure mode entirely: the only
/// thing ever deferred now is the mprotect/seal itself, which is exactly
/// what `paged_memory`'s watermark design is built to tolerate no matter
/// how long it's deferred.
///
/// Returns the same `bool` as `handle_request` (arena-OOM signal) for the
/// same reason.
pub fn handle_request_deferred(
    req: &CompileRequest,
    bus: &Arc<dyn BusDevice>,
    analyzer: &mut Analyzer,
    codegen: &mut Codegen,
    pending: &mut PendingCount,
    #[cfg(feature = "developer")] stats: &crate::jitv2::JitStats,
) -> bool {
    let page = unsafe { &*req.page };

    struct ClearScheduledOnDrop<'a> { page: &'a crate::jitv2::PhysicalCodePage }
    impl Drop for ClearScheduledOnDrop<'_> {
        fn drop(&mut self) { self.page.clear_scheduled(); }
    }
    let _clear_scheduled = ClearScheduledOnDrop { page };

    #[cfg(feature = "developer")]
    let outcome = prepare_multi_entry_compile(req, bus, analyzer, stats);
    #[cfg(not(feature = "developer"))]
    let outcome = prepare_multi_entry_compile(req, bus, analyzer);
    let (gen_snap, instr_count) = match outcome {
        PrepareOutcome::Done(early) => return early,
        PrepareOutcome::Ready { gen_snap, instr_count } => (gen_snap, instr_count),
    };

    let mut new_entries = [0u64; crate::jitv2::BITMAP_WORDS];
    for &offset in analyzer.covered() {
        new_entries[offset as usize >> 6] |= 1u64 << (offset % 64);
    }

    let mut instrs_owned = analyzer.instrs_snapshot();
    let func_id = codegen.compile_region_uncommitted(&mut instrs_owned, req.compiled_for_fr1, true, analyzer.has_fpu(), req.page);
    match func_id {
        Some(func_id) => {
            #[cfg(feature = "developer")]
            let code_size = codegen.last_code_size();
            #[cfg(not(feature = "developer"))]
            let code_size = 0;
            // Finalize immediately, every compile — see this function's own
            // doc comment for why deferring finalize itself (rather than
            // just the seal) was the actual bug. finalize_batch_nonforced
            // patches this entry's PublishInfo into the seal-queue slot
            // compile_region_uncommitted already reserved for it
            // (push_placeholder) and tries to seal — the returned entries
            // may include this one (the common case — nothing else is
            // blocking it) and/or other workers' own entries that this
            // seal attempt happened to unblock too; every entry returned
            // here is fully self-describing (paged_memory::PublishInfo's
            // own doc comment) and gets published unconditionally,
            // regardless of whose compile produced it. If nothing came
            // back at all, this range is itself gap-blocked behind an
            // earlier, still-unpatched entry — retry later via the
            // idle-timeout/pending-threshold sweep. UNLESS
            // `last_finalize_failed()` says finalize itself errored, in
            // which case this entry will never resolve no matter how many
            // sweeps run — see `Codegen::last_finalize_failed`'s own doc
            // comment for the real bug this distinction fixes.
            let publish = crate::jitv2::paged_memory::PublishInfo {
                page: req.page, new_entries, gen_snap, instr_count, code_size,
                compiled_for_fr1: req.compiled_for_fr1,
                jit_fn: None,
            };
            let sealed = codegen.finalize_batch_nonforced(func_id, publish);
            if sealed.is_empty() {
                if codegen.last_finalize_failed() {
                    page.mark_finalize_failed();
                } else {
                    page.mark_seal_gap_blocked();
                    *pending += 1;
                }
            } else {
                publish_all(&sealed);
            }
            #[cfg(feature = "developer")]
            {
                // Count fallback words/regions and the successful-compile
                // total — same bookkeeping `handle_request`'s own success arm
                // does, needed here too since `j2 status`/`j2 stats` read
                // these process-wide counters regardless of which path
                // compiled.
                let mut fb = 0u64;
                for instr in crate::jitv2::analyzer::instrs_linear(&instrs_owned) {
                    if instr.is_fallback { fb += 1; }
                }
                if fb > 0 {
                    stats.fallback_words.fetch_add(fb, std::sync::atomic::Ordering::Relaxed);
                    stats.fallback_regions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                stats.compiles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        None if codegen.last_compile_ran_out_of_memory() => {
            page.mark_codegen_oom_bounced();
            #[cfg(feature = "developer")]
            stats.failed_compiles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }
        None => {
            for &offset in analyzer.covered() {
                page.denylist(offset as usize);
            }
            page.mark_codegen_rejected();
            #[cfg(feature = "developer")]
            {
                let reason = if codegen.last_decline_was_verifier_error() {
                    crate::jitv2::RejectReason::CraneliftVerifierError
                } else {
                    crate::jitv2::RejectReason::AnalyzerCodegenDisagreement
                };
                stats.record_reject(reason);
            }
        }
    }
    false
}

/// Publish every entry a seal sweep just returned, unconditionally — each
/// `PublishInfo` is fully self-describing (see its own doc comment), so
/// there's no `FuncId`/`Codegen` lookup needed here, and no reason to
/// filter: every entry `try_seal_ready`/`try_seal_ready_forced` hands back
/// is, by construction, freshly sealed and ready to dispatch. May include
/// entries pushed by a *different* worker than the one that triggered this
/// particular sweep — that's fine and expected (see this function's own
/// callers' doc comments).
fn publish_all(sealed: &[crate::jitv2::paged_memory::PublishInfo]) {
    for entry in sealed {
        let page = unsafe { &*entry.page };
        let jit_fn = entry.jit_fn.expect("publish_all: a sealed entry must always carry a resolved JitFn");
        // Same pre-publish re-probe as the inline path in `handle_request` —
        // see the comment there for why this plus `publish`'s `gen_snap` check
        // covers every case. Deferred entries sat in the seal queue for even
        // longer than an inline compile, so the window is wider here.
        #[cfg(not(feature = "tcache"))]
        if crate::jitv2::jit_page_has_dirty_lines((page.pfn * PAGE_SIZE) as u64) {
            continue;
        }
        if page.publish(&entry.new_entries, jit_fn as *const (), entry.gen_snap, entry.instr_count, entry.code_size) {
            page.clear_requested_bits(&entry.new_entries);
        }
    }
}

/// Non-forced retry sweep for the async worker's idle-timeout scheme (see
/// `worker_loop`'s own doc comment): re-attempts sealing whatever the
/// shared arena's seal queue currently has queued (this worker's own
/// gap-blocked entries — see `PendingCount`'s own doc comment — or another
/// worker's) and publishes everything that comes back. Decrements `*pending`
/// by how many entries this sweep actually published, saturating at 0 since
/// a sweep triggered by this worker's own gap-blocked entry can incidentally
/// publish other workers' entries too, which never incremented this
/// worker's own counter.
pub fn publish_ready_nonforced(codegen: &mut Codegen, pending: &mut PendingCount) {
    if *pending == 0 {
        return;
    }
    let sealed = codegen.try_seal_ready();
    *pending = pending.saturating_sub(sealed.len());
    publish_all(&sealed);
}

/// Idle-timeout counterpart: force-seal whatever the shared arena's seal
/// queue still has queued without waiting for a further contiguous-prefix
/// match — see `Codegen::force_seal_pending`'s own doc comment. Same
/// saturating-decrement/publish-everything contract as
/// `publish_ready_nonforced`.
pub fn force_publish_pending(codegen: &mut Codegen, pending: &mut PendingCount) {
    if *pending == 0 {
        return;
    }
    let sealed = codegen.force_seal_pending();
    *pending = pending.saturating_sub(sealed.len());
    publish_all(&sealed);
}

#[cfg(feature = "jitv2_corpus_dump")]
pub const CORPUS_DIR: &str = "jitv2_corpus";

#[cfg(feature = "jitv2_corpus_dump")]
fn dump_corpus_snapshot(page: &crate::jitv2::PhysicalCodePage, offset: u16, words: &[u32; ENTRIES_PER_PAGE]) {
    use std::io::Write;

    if !page.mark_saved(offset as usize) {
        return; // already dumped for this (pfn, offset)
    }
    let out_dir = std::path::Path::new(CORPUS_DIR);
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("jitv2 corpus: failed to create {}: {}", CORPUS_DIR, e);
        return;
    }
    let path = out_dir.join(format!("pfn_{:08x}_off_{:04x}.bin", page.pfn, offset));
    let write = || -> std::io::Result<()> {
        let mut f = std::fs::File::create(&path)?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words))
        };
        f.write_all(bytes)
    };
    if let Err(e) = write() {
        eprintln!("jitv2 corpus: failed to write snapshot for pfn={:#010x} offset={:#06x}: {}", page.pfn, offset, e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jitv2::PhysicalCodePage;
    use crate::mips_isa::{OP_ADDIU, OP_SPECIAL};
    use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_ERR};
    use std::sync::atomic::AtomicU64;

    /// Test-only wrapper hiding `handle_request`'s `developer`-only `stats`
    /// parameter — every test below just wants "run a request," not the
    /// stats side effects, so a fresh throwaway `JitStats` under `developer`
    /// keeps every call site identical regardless of which build this runs
    /// under (`--features jitv2` alone must build and pass exactly like
    /// `--features jitv2,developer` does).
    fn handle_request_for_test(req: &CompileRequest, bus: &Arc<dyn BusDevice>, analyzer: &mut Analyzer, codegen: &mut Codegen) -> bool {
        #[cfg(feature = "developer")]
        {
            let stats = crate::jitv2::JitStats::default();
            handle_request(req, bus, analyzer, codegen, &stats)
        }
        #[cfg(not(feature = "developer"))]
        {
            handle_request(req, bus, analyzer, codegen)
        }
    }

    /// Every word decodes as `ADDIU r1, r0, 1` (opcode 0x09, rs=0, rt=1,
    /// imm=1) — a plain Sequential instruction with a real codegen emitter,
    /// so `handle_request` can exercise the whole walk+compile+publish path
    /// without needing a real memory device.
    struct AddiuDevice;
    impl BusDevice for AddiuDevice {
        fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u32, _val: u8) -> u32 { BUS_ERR }
        fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u32, _val: u16) -> u32 { BUS_ERR }
        fn read32(&self, _addr: u32) -> BusRead32 {
            BusRead32::ok((OP_ADDIU << 26) | (1 << 16) | 1)
        }
        fn write32(&self, _addr: u32, _val: u32) -> u32 { BUS_ERR }
        fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u32, _val: u64) -> u32 { BUS_ERR }
        fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 { std::ptr::null() }
    }

    /// Every word decodes as the JIT region-boundary sentinel — a genuine
    /// zero-instruction region (analyzer `Classify::RegionBoundary`), used to
    /// exercise `handle_request`'s denylist exit path deterministically. (An
    /// excluded instruction like SYSCALL no longer works for this: with
    /// interpreter-fallback it's kept in the region as a fallback head and
    /// compiles/publishes successfully rather than denylisting — the sentinel is
    /// now the only thing that reliably yields an empty region.)
    struct BoundaryDevice;
    impl BusDevice for BoundaryDevice {
        fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u32, _val: u8) -> u32 { BUS_ERR }
        fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u32, _val: u16) -> u32 { BUS_ERR }
        fn read32(&self, _addr: u32) -> BusRead32 {
            BusRead32::ok(crate::mips_isa::JIT_REGION_BOUNDARY_SENTINEL)
        }
        fn write32(&self, _addr: u32, _val: u32) -> u32 { BUS_ERR }
        fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u32, _val: u64) -> u32 { BUS_ERR }
        fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 { std::ptr::null() }
    }

    struct ErrDevice;
    impl BusDevice for ErrDevice {
        fn read8(&self, _addr: u32) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u32, _val: u8) -> u32 { BUS_ERR }
        fn read16(&self, _addr: u32) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u32, _val: u16) -> u32 { BUS_ERR }
        fn read32(&self, _addr: u32) -> BusRead32 { BusRead32::err() }
        fn write32(&self, _addr: u32, _val: u32) -> u32 { BUS_ERR }
        fn read64(&self, _addr: u32) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u32, _val: u64) -> u32 { BUS_ERR }
        fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 { std::ptr::null() }
    }

    #[test]
    fn handle_request_publishes_a_compilable_instruction() {
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        page.mark_requested(4);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);

        assert!(page.is_runnable(4), "a plain ADDIU must compile and publish");
        assert!(!page.is_denylisted(4));
    }

    /// The dirty-page gate must actually be wired into *this* design's compile
    /// path — not merely exist somewhere in the file.
    ///
    /// This exists because it was originally missed: both gates were placed in
    /// `old_impl`, leaving `j2wp` builds with no probe at all. Nothing caught
    /// it — every unit test passed, IRIX booted, and the only symptom was
    /// `PageDirtyInCache` silently reading zero in `j2 stats` forever. A test
    /// that drives the real entry point with a probe forced to "dirty" fails
    /// loudly in whichever design forgot to consult it.
    #[cfg(not(feature = "tcache"))]
    #[test]
    fn a_dirty_page_is_never_compiled() {
        let _g = dirty_probe_lock();

        // Sanity first: with no probe installed this exact request publishes.
        // Without this the test could pass for the wrong reason (e.g. the page
        // was never compilable to begin with).
        crate::jitv2::clear_jit_page_probe();
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        page.mark_requested(4);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true };
        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
        assert!(page.is_runnable(4),
            "precondition: this request must publish when no probe vetoes it");

        // Now the real assertion: a probe reporting the page dirty must stop
        // the compile before anything is published.
        fn always_dirty(_ctx: *const (), _page_base: u64) -> bool { true }
        static ANCHOR: u8 = 0;
        // SAFETY: `always_dirty` ignores ctx; ANCHOR is a 'static non-null
        // stand-in so the installed-probe check sees a live pointer.
        unsafe { crate::jitv2::install_jit_page_probe(&ANCHOR as *const u8 as *const (), always_dirty) };

        let counter2 = AtomicU64::new(0);
        let mut page2 = PhysicalCodePage::new(0, &counter2 as *const AtomicU64);
        page2.mark_requested(4);
        let req2 = CompileRequest { page: &mut page2 as *mut PhysicalCodePage, compiled_for_fr1: true };
        handle_request_for_test(&req2, &bus, &mut analyzer, &mut codegen);

        crate::jitv2::clear_jit_page_probe();

        assert!(!page2.is_runnable(4),
            "a page reported dirty in the CPU cache must not publish — the compile would be built \
             from a stale RAM snapshot (this design's compile path is not consulting the probe)");
        // The abort must be retryable, not sticky: the offset gets another
        // chance once the cache lines drain to RAM on their own.
        assert!(!page2.is_denylisted(4),
            "a dirty-page abort must leave the offset eligible for a later retry, not denylist it");
    }

    /// Exclusion for the probe global. Delegates to `jitv2::probe_test_lock`
    /// rather than owning a mutex here: `mips_cache_v2`'s probe tests install
    /// into the same global, and two independent locks would exclude nothing.
    #[cfg(not(feature = "tcache"))]
    fn dirty_probe_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::jitv2::jitv2::probe_test_lock()
    }

    #[test]
    fn a_second_incremental_compile_does_not_drop_the_first_entry() {
        // §13's one-function-per-page model fully replaces `func` on every
        // compile; its dispatch switch only recognizes the entry_words THAT
        // walk covered. If a later, unrelated compile (offset 8) doesn't
        // re-walk offset 4 alongside it, offset 4's `compiled` bit stays set
        // (publish unions rather than clearing) and `is_runnable` keeps
        // reporting it dispatchable, but the new `func` has no case for it —
        // every dispatch into it would fall through to EXEC_FALLBACK forever
        // (confirmed live via a boot traceback: `[jit:entry]` immediately
        // followed by the interpreter silently re-executing the same
        // instruction, on every single hit of an older entry).
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();

        page.mark_requested(4);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true };
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
        assert!(page.is_runnable(4), "offset 4 must compile and publish on its own");
        let func_after_first_compile = page.func();

        page.mark_requested(8);
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
        assert!(page.is_runnable(8), "offset 8 must compile and publish");
        assert_ne!(page.func(), func_after_first_compile,
            "sanity: the second compile really did replace func with a new one");
        assert!(page.is_runnable(4),
            "offset 4's compiled bit must stay dispatchable after an unrelated later compile — \
             a fresh func that doesn't re-walk it has no dispatch case for it at all, silently \
             turning every future hit into a JIT-entry-then-EXEC_FALLBACK round trip forever");
    }

    #[test]
    fn handle_request_skips_already_valid_entry() {
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        page.mark_requested(0);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
        assert!(page.is_runnable(0));

        // Bump gen so a naive re-compile would behave differently if it ran;
        // since is_runnable is already true, handle_request must bail
        // before touching the bus again. Re-mark requested: publish clears
        // the covered bit on success (clear_requested_bits), so without
        // this the second call would find zero candidates and no-op for an
        // unrelated reason (nothing requested), not the one this test means
        // to exercise (gen-bump-alone must not un-publish).
        page.mark_requested(0);
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
        assert!(page.is_runnable(0), "gen bump alone must not un-publish without going through it");
    }

    #[test]
    fn handle_request_leaves_offset_undecided_on_bus_error() {
        let bus: Arc<dyn BusDevice> = Arc::new(ErrDevice);
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        page.mark_requested(0);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);

        assert!(!page.is_runnable(0));
        assert!(!page.is_denylisted(0), "a transient read failure must not become a sticky rejection");
        assert_eq!(page.rejected_compiles(), 0, "a bus-read bounce is not a real rejection");
        assert_eq!(page.prepare_bounced(), 1,
            "a bus-read failure must be visible as a prepare-stage bounce, distinct from both a real \
             rejection and a queue-level drop — this is exactly the silent-vanish shape j2 pcp needs \
             to distinguish from `sends_dropped_queue_full`");
    }

    /// Regression test: `exec_decoded`'s dispatch gate sets a page-level
    /// in-flight flag (`try_schedule_page`) before sending a `CompileRequest`
    /// to stop a hot page from flooding the queue with duplicate requests.
    /// `handle_request` must clear that flag again once it decides — on
    /// every exit path, not just the success path — or the page can never
    /// be scheduled again after this one request, even once it's fully
    /// resolved. Covers all three decision outcomes: publish success,
    /// sticky denylist, and "bus not readable, retry later."
    #[test]
    fn handle_request_clears_scheduled_bit_on_every_outcome() {
        let counter = AtomicU64::new(0);

        // Outcome 1: publishes successfully.
        {
            let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
            let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
            page.mark_requested(4);
            assert!(page.try_schedule_page());
            let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true };
            let mut analyzer = Analyzer::new();
            let mut codegen = Codegen::new();
            handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
            assert!(page.is_runnable(4));
            assert!(page.try_schedule_page(), "scheduled bit must be cleared after a successful publish, so a future recompile request isn't blocked");
        }

        // Outcome 2: sticky-denylisted (the region-boundary sentinel yields a
        // zero-instruction region — the only reliable empty-region case now
        // that excluded instructions compile as interpreter-fallback heads).
        {
            let bus: Arc<dyn BusDevice> = Arc::new(BoundaryDevice);
            let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
            page.mark_requested(0);
            assert!(page.try_schedule_page());
            let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true };
            let mut analyzer = Analyzer::new();
            let mut codegen = Codegen::new();
            handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
            assert!(page.is_denylisted(0));
            assert!(page.try_schedule_page(), "scheduled bit must be cleared after a denylist decision");
        }

        // Outcome 3: bus read fails, offset left undecided ("retry later").
        {
            let bus: Arc<dyn BusDevice> = Arc::new(ErrDevice);
            let mut page = PhysicalCodePage::new(0, std::ptr::null());
            page.mark_requested(0);
            assert!(page.try_schedule_page());
            let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true };
            let mut analyzer = Analyzer::new();
            let mut codegen = Codegen::new();
            handle_request_for_test(&req, &bus, &mut analyzer, &mut codegen);
            assert!(!page.is_runnable(0));
            assert!(!page.is_denylisted(0));
            assert!(page.try_schedule_page(), "scheduled bit must be cleared even on a transient bus-read failure, so the offset can be retried");
        }
    }

    /// Same `stats`-hiding wrapper as `handle_request_for_test`, for the
    /// deferred path.
    fn handle_request_deferred_for_test(req: &CompileRequest, bus: &Arc<dyn BusDevice>, analyzer: &mut Analyzer, codegen: &mut Codegen, pending: &mut PendingCount) -> bool {
        #[cfg(feature = "developer")]
        {
            let stats = crate::jitv2::JitStats::default();
            handle_request_deferred(req, bus, analyzer, codegen, pending, &stats)
        }
        #[cfg(not(feature = "developer"))]
        {
            handle_request_deferred(req, bus, analyzer, codegen, pending)
        }
    }

    #[test]
    fn handle_request_deferred_leaves_a_lone_small_compile_pending_until_its_page_seals() {
        // Non-forced sealing never seals a still-open page — even with a
        // solo arena and exactly one compile, nothing else will ever grow
        // `position` into a new page on its own (see
        // Codegen::finalize_batch_nonforced_does_not_seal_a_still_open_page's
        // own test, at the arena level). handle_request_deferred must
        // therefore leave this offset pending, exactly as the idle-timeout/
        // force-seal machinery (force_publish_pending) expects to find it —
        // this isn't "gap-blocked behind another worker," just "not yet a
        // whole page's worth."
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        page.mark_requested(4);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        let mut pending: PendingCount = 0;
        handle_request_deferred_for_test(&req, &bus, &mut analyzer, &mut codegen, &mut pending);

        assert_eq!(pending, 1, "a lone small compile's page is still open — non-forced sealing must not publish it yet");
        assert!(!page.is_runnable(4), "must not be published until something actually seals its page");

        force_publish_pending(&mut codegen, &mut pending);
        assert_eq!(pending, 0, "the idle-timeout force-seal sweep must finish the job");
        assert!(page.is_runnable(4), "now published, once forced");
    }

    #[test]
    fn handle_request_deferred_denylists_empty_region_entry_without_accumulating() {
        let bus: Arc<dyn BusDevice> = Arc::new(BoundaryDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        page.mark_requested(0);
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        let mut pending: PendingCount = 0;
        handle_request_deferred_for_test(&req, &bus, &mut analyzer, &mut codegen, &mut pending);

        assert_eq!(pending, 0, "an empty-region (boundary) entry produces no FuncId to defer");
        assert!(page.is_denylisted(0));
    }

    #[test]
    fn handle_request_deferred_clears_scheduled_bit() {
        let bus: Arc<dyn BusDevice> = Arc::new(AddiuDevice);
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        page.mark_requested(4);
        assert!(page.try_schedule_page());
        let req = CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true };

        let mut analyzer = Analyzer::new();
        let mut codegen = Codegen::new();
        let mut pending: PendingCount = 0;
        handle_request_deferred_for_test(&req, &bus, &mut analyzer, &mut codegen, &mut pending);

        assert!(page.try_schedule_page(), "scheduled bit must be cleared even though publish itself is deferred");
    }
}

}
#[cfg(feature = "j2wp")]
pub use new_impl::*;
