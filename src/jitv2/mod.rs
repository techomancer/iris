//! JIT v2: physical-page PIC region compiler. See `rules/jitv2/jit-v2-design.md`.

pub mod jitv2;
pub mod comp;
pub mod opcode_support;
pub mod analyzer;
pub mod codegen;
pub mod paged_memory;
// Always compiled in, under both `comp.rs` implementations: `j2 dumppcp`
// and `j2 corpus` are monitor-console diagnostics, and the corpus capture
// they feed has to be available in the same production-shaped build whose
// emitted code anyone would want to measure (see `zz_corpus_sizes` below,
// and `rules/jitv2/block-fragmentation-blocks-cse.md` on why measuring a
// `developer` build answers the wrong question).
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
pub use jitv2::{install_jit_page_probe, clear_jit_page_probe, clear_jit_page_probe_if, jit_page_has_dirty_lines};

#[cfg(test)]
mod zz_corpus {
    use crate::jitv2::analyzer::Analyzer;
    use crate::jitv2::codegen::Codegen;
    use crate::jitv2::JitFn;
    use crate::jitv2::jitv2::ENTRIES_PER_PAGE;
    use crate::jitv2::pcp_dump;

    /// Compile every entry point of every `.pcp` page in `IRIS_CORPUS_DIR`
    /// and report the total emitted bytes. Same input set for two builds =
    /// an apples-to-apples measure of emitted code volume on real guest code.
    ///
    /// Capture a corpus with `j2 corpus [dir]` from a running emulator (see
    /// `jitv2/pcp_dump.rs`). That replaced the old `jitv2_corpus_dump` Cargo
    /// feature, and with it the `IRIS_CORPUS_LIST` file-of-filenames this
    /// test used to take: a `.pcp` carries the page's whole `requested`
    /// bitmap, so every entry offset for a page comes from the file itself
    /// rather than from one offset encoded per filename as `_off_XXXX.bin`.
    /// That also means the measurement now covers *every* entry point a page
    /// has, not just whichever one happened to be dumped first.
    ///
    /// **Do not run this under `developer`** — see
    /// `rules/jitv2/block-fragmentation-blocks-cse.md`: that feature flips
    /// `opt_level` to `none` and injects a per-instruction trace callout, so
    /// the numbers describe code production never emits.
    #[test]
    fn zz_corpus_sizes() {
        let dir = match std::env::var("IRIS_CORPUS_DIR") { Ok(v)=>v, Err(_)=>return };
        // Match the real emulator: opt_level is a process-wide static that
        // defaults to `none` under `developer`. Production runs `speed`.
        let speed = std::env::var("IRIS_OPT_SPEED").is_ok();
        Codegen::set_opt_level_speed(speed);
        println!("OPTLEVEL speed={}", speed);
        let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("IRIS_CORPUS_DIR {}: {}", dir, e))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map_or(false, |x| x == "pcp"))
            .collect();
        // Sorted so two runs over the same corpus visit pages in the same
        // order: the totals wouldn't change, but per-page `IRIS_JIT_DISASM`
        // output is only diffable between builds if the order is stable.
        paths.sort();
        // A real, hook-installed core for the constants to point at. Leaked
        // deliberately: compiled code bakes its address, so it must outlive
        // every region this test compiles.
        let consts = {
            use crate::mips_core::MipsCore;
            let core: &'static mut MipsCore = Box::leak(Box::new(MipsCore::new()));
            // The hook fields still hold their not-installed panic sentinels,
            // which is fine — nothing here *runs* the compiled code, and the
            // sentinels are real function addresses, so the emitted shape
            // (baked immediate vs load) is identical to production.
            crate::jitv2::codegen::JitConsts {
                core: core::num::NonZeroUsize::new(core as *mut MipsCore as usize),
            }
        };
        // R4400 L1-D: 16 KiB direct-mapped, 32-byte lines; 1 MiB L2 with
        // 128-byte lines. Matches `CpuCache::jit_dc_geometry` for the Indy.
        let geom = crate::mips_cache_v2::JitDcGeometry {
            supported: true,
            line_shift: 5,
            num_lines_mask: (16 * 1024 / 32) - 1,
            data_mask: 16 * 1024 - 1,
            has_l2: true,
            l2_line_shift: 7,
            l2_num_lines_mask: (1024 * 1024 / 128) - 1,
            // Direct-mapped, so the guard takes its 1-way path; the way bit
            // never participates. `num_lines_shift` is unread when ways == 1.
            ways: 1,
            num_lines_shift: 0,
        };
        let mut total: u64 = 0;
        let mut n_ok = 0u64;
        let mut n_decl = 0u64;
        // Entries the *analyzer* refused, before codegen ever saw them
        // (`walk_bounded` returning `!ok`: the entry word itself is Excluded,
        // §6.4). Counted separately from `n_decl`, which is codegen declining
        // a region it was actually handed. Both are real "this entry produced
        // no code" outcomes and the two must sum with `ok` to `entries`, or
        // some entries are being dropped silently — which is exactly what
        // happened when only `n_decl` existed: 178 entries went missing
        // between `entries=77690` and `ok=77512` with nothing accounting for
        // them, while `jitv2_pcp_dump` showed 13 walk-declined entries on the
        // very first page of the corpus.
        let mut n_walk_decl = 0u64;
        let mut n_pages = 0u64;
        let mut n_entries = 0u64;
        // Dispatch-weighted total: the same emitted bytes, each page's
        // contribution scaled by how many times the guest actually ran it
        // (`call_count`, recorded by `j2 corpus`). A flat byte count treats a
        // page dispatched two million times and one dispatched twice as
        // equals; this doesn't. 0 for every page on a corpus captured from a
        // non-`developer` build, where the counter doesn't exist — reported
        // separately rather than folded in, so a flat total is always
        // comparable between runs.
        let mut weighted: u128 = 0;
        let mut have_weights = false;

        // One `Analyzer` and one `Codegen`, reused across the whole corpus
        // rather than built per entry point. The `Analyzer` is just scratch
        // state meant to be reused (same as `worker_loop`, which holds one
        // for the thread's lifetime). The `Codegen` matters more:
        //
        // `cranelift_jit::Memory`'s `Drop` deliberately `mem::forget`s every
        // executable allocation rather than freeing it, so that a dangling
        // `JitFn` can never become a use-after-free — dropping or leaking a
        // `Codegen` reclaims nothing either way. The old shape here built a
        // fresh `Codegen` per entry and `mem::forget`'d it, which was
        // survivable over a 300-page corpus and is not over a real one: at
        // ~78k regions this reached **30 GB RSS in 8 minutes** and was still
        // climbing, with no output to say how far along it was.
        //
        // `Codegen::reset()` is the one thing that actually frees that
        // memory (`JITModule::free_memory`). It's `unsafe` because every
        // `JitFn` the module ever returned dangles afterwards — a contract
        // trivially satisfied here: this test never calls a compiled
        // function, it only measures `last_code_size()`, and each `f` is
        // dropped before the next reset.
        let mut an = Analyzer::new();
        let mut cg = Codegen::new();
        // Stamp the same compile-time constants a live worker would get, so
        // this measures the code the emulator actually runs. Without it
        // `hook_addr` returns `None` and every callee falls back to a
        // register load — which silently made this benchmark blind to the
        // whole constant-baking change.
        cg.jit_consts = consts;
        // Real L1-D geometry, so `emit_inline_mem_guard` actually emits the
        // inline fast path. With the default `unsupported()` the guard is
        // skipped entirely and this benchmark silently measures callout-only
        // code — invisible to any change in the inline path.
        cg.dc_geometry = geom;
        // Reclaim on a byte budget, not an entry count: region sizes vary by
        // more than an order of magnitude, so "every N entries" would still
        // swing wildly. 256 MB keeps peak RSS bounded and small while costing
        // one `free_memory` + fresh `JITModule` per ~256 MB emitted.
        const RESET_AFTER_BYTES: u64 = 256 * 1024 * 1024;
        let mut since_reset: u64 = 0;

        let started = std::time::Instant::now();
        for (path_idx, path) in paths.iter().enumerate() {
            let bytes = match std::fs::read(path) { Ok(b)=>b, Err(_)=>continue };
            let dump = match pcp_dump::PcpDump::from_bytes(&bytes) {
                Ok(d) => d,
                Err(e) => { println!("SKIP {}: {}", path.display(), e); continue; }
            };
            n_pages += 1;
            if dump.call_count > 0 { have_weights = true; }
            let pw = dump.words;
            // Every entry point the page actually had, not one guessed from
            // the filename.
            //
            // The UNION of `requested` and `compiled`, not `requested`
            // alone — same entry set `jitv2_pcp_dump`'s offline walk uses,
            // and for the same reason. `requested` is not a cumulative
            // record of every entry ever asked for: under `j2wp` a bit is
            // cleared once a compile covers it, so on a page whose compiles
            // have all landed it reads as *empty* while `compiled` holds the
            // real entry set. Measured on the first 1427-page IRIX corpus:
            // 381 requested bits total against 77,309 compiled ones, with
            // 1337 of the 1427 pages showing `requested` empty. Keying the
            // walk off `requested` alone would silently discard 99.5% of the
            // corpus and report a total_bytes that looked plausible.
            //
            // Denylisted offsets are deliberately NOT excluded here: a
            // region the analyzer declines costs a `compile_region` call
            // that returns `None` and is counted under `declined=`, which is
            // itself a number worth watching between builds (a change that
            // starts declining regions it used to compile is a regression
            // this measurement should surface, not hide).
            let entry_set: Vec<usize> = (0..ENTRIES_PER_PAGE)
                .filter(|&o| dump.is_requested(o) || dump.is_compiled(o))
                .collect();
            for off in entry_set {
                let off = off as u16;
                n_entries += 1;
                let (walked, ok) = an.walk_bounded(&pw, off, 0x8000_0000u32, usize::MAX);
                if !ok { n_walk_decl += 1; continue; }
                let mut ins = *walked;
                let f: Option<JitFn> = cg.compile_region(&mut ins, off, true, false);
                // `last_code_size` used to be `developer`-gated, which made
                // this measurement report 0 bytes in the only build worth
                // measuring (see that field's own doc comment).
                let sz = cg.last_code_size() as u64;
                if f.is_some() {
                    total += sz;
                    n_ok += 1;
                    weighted += (sz as u128) * (dump.call_count as u128);
                } else { n_decl += 1; }
                // Explicit: `f` must not outlive the reset below.
                drop(f);
                since_reset += sz;
                if since_reset >= RESET_AFTER_BYTES {
                    // Safety: no `JitFn` from this `Codegen` is live — none is
                    // ever stored, and the only one produced this iteration
                    // was dropped immediately above.
                    unsafe { cg.reset() };
                    cg.jit_consts = consts;
                    cg.dc_geometry = geom;
                    since_reset = 0;
                }
            }

            // Progress: a real corpus is tens of thousands of regions and
            // minutes of wall time. Without this the run is indistinguishable
            // from a hang — which is exactly how the 30 GB leak above went
            // unnoticed until it had eaten a quarter of host RAM.
            if path_idx % 100 == 99 || path_idx + 1 == paths.len() {
                println!("  ... {}/{} pages, {} entries, {} bytes, {:.0}s",
                    path_idx + 1, paths.len(), n_entries, total, started.elapsed().as_secs_f64());
            }
        }
        println!("CORPUS pages={} entries={} ok={} walk_declined={} codegen_declined={} total_bytes={}",
            n_pages, n_entries, n_ok, n_walk_decl, n_decl, total);
        // Every entry must land in exactly one bucket. A mismatch means the
        // loop grew a path that drops entries on the floor, which would make
        // `total_bytes` quietly non-comparable between builds.
        assert_eq!(n_ok + n_walk_decl + n_decl, n_entries,
            "corpus accounting lost entries: ok={} walk_declined={} codegen_declined={} != entries={}",
            n_ok, n_walk_decl, n_decl, n_entries);
        if have_weights {
            println!("CORPUS weighted_bytes={} (sum of region bytes x page dispatch count)", weighted);
        } else {
            println!("CORPUS weighted_bytes=n/a (corpus has no dispatch counts: captured from a non-developer build)");
        }
        if cfg!(feature = "developer") {
            println!("WARNING: `developer` is on — opt_level=none plus a per-instruction trace callout. \
                      These numbers do not describe production codegen; see \
                      rules/jitv2/block-fragmentation-blocks-cse.md.");
        }
    }
}

/// Measures the ceiling on block merging: how much of a real corpus sits in
/// straight-line runs that could share one Cranelift block.
///
/// Answers "is it worth building" for the optimization
/// `rules/jitv2/block-fragmentation-blocks-cse.md` ranks first. That note
/// established the *symptom* (median machine block of 3 instructions, 11,522
/// cross-block duplicate loads of which 58% are separated by a block boundary
/// and nothing else) but not how much of the instruction stream is actually
/// mergeable under the safety rules codegen has to obey.
///
/// Reports nothing about correctness — it only counts. Read-only over a
/// corpus, so it costs one pass and no codegen changes.
///
/// Usage: `IRIS_CORPUS_DIR=jitv2_corpus cargo test --release --features jitv2 \
///   zz_corpus_runs -- --nocapture`
#[cfg(test)]
mod zz_runs {
    use crate::jitv2::analyzer::{instrs_linear, Analyzer};
    use crate::jitv2::jitv2::ENTRIES_PER_PAGE;
    use crate::jitv2::pcp_dump;

    #[test]
    fn zz_corpus_runs() {
        let dir = match std::env::var("IRIS_CORPUS_DIR") { Ok(v)=>v, Err(_)=>return };
        let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("IRIS_CORPUS_DIR {}: {}", dir, e))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map_or(false, |x| x == "pcp"))
            .collect();
        paths.sort();

        let mut an = Analyzer::new();
        // Histogram of run lengths, in head instructions. Index = length,
        // saturating into the last bucket.
        const MAX_BUCKET: usize = 32;
        let mut hist = [0u64; MAX_BUCKET + 1];
        let mut total_heads = 0u64;
        let mut total_runs = 0u64;
        let mut heads_in_runs_ge2 = 0u64;
        // Why each run ended, so the ranking says what to attack next.
        let (mut end_branch_target, mut end_entry, mut end_exit, mut end_fallback, mut end_page) =
            (0u64, 0u64, 0u64, 0u64, 0u64);

        for path in &paths {
            let Ok(bytes) = std::fs::read(path) else { continue };
            let Ok(dump) = pcp_dump::PcpDump::from_bytes(&bytes) else { continue };
            let pw = dump.words;
            // Same entry set `zz_corpus_sizes` uses, and for the same reason
            // (`requested` alone is not cumulative — see that test).
            let entry_words: Vec<u16> = (0..ENTRIES_PER_PAGE)
                .filter(|&o| dump.is_requested(o) || dump.is_compiled(o))
                .map(|o| o as u16)
                .collect();
            if entry_words.is_empty() { continue; }
            // ONE walk per page covering every entry point, not one walk per
            // entry. Regions overlap heavily (a page with 85 entry points
            // re-walks most of its words), so per-entry walking counts shared
            // words once per entry that reaches them — which double-counts
            // both heads and runs, and cannot be repaired by deduplicating
            // only one of the two (an earlier version deduplicated heads but
            // not runs and reported "922% of heads in runs of >=2", which is
            // what caught it). `walk_multi_entry` is what the real compiler
            // uses for exactly this reason.
            let walked = an.walk_multi_entry(&pw, &entry_words, 0x8000_0000u32, usize::MAX);
            {

                // Walk the region's head instructions in address order and cut
                // a run wherever the next word cannot legally share a block.
                //
                // The predicate mirrors `try_emit_fused_lui`'s, which is the
                // codebase's existing, battle-tested answer to "may these two
                // adjacent words be treated as one unit":
                //
                //   - `is_branch_target`  — something jumps straight here, so
                //                           this word is a join point. A value
                //                           defined before it has no single
                //                           definition at it.
                //   - `is_entry_point` / `is_branch_fallback_successor`
                //                         — reachable from outside the region
                //                           entirely (external dispatch, or a
                //                           foreign delay-slot arrival).
                //   - `is_fallback`       — runs in the interpreter.
                //
                // A word whose own edges leave the region (`taken_exit` or
                // `fallthrough_exit` set) ends a run too: control does not
                // continue to word+1 in a way a shared block could express.
                //
                // NOT a cut: faulting. A load that may fault does not need its
                // own block — `emit_inline_mem_guard` already emits a side
                // exit to a cold bail block and continues in the fallthrough,
                // and SSA values stay live across that. Only *join points*
                // force a value back through memory.
                let mut run_len = 0u64;
                let mut prev_word: Option<u16> = None;

                for instr in instrs_linear(walked) {
                    if instr.is_slot_only { continue; }
                    let w = instr.word;
                    total_heads += 1;

                    // Does this word continue the run the previous one
                    // started? "Contiguous" has to skip over inlined delay
                    // slots: a branch's slot occupies word+1 but is
                    // `is_slot_only` and never gets its own block (it is
                    // emitted inline into the branch), so the next *head*
                    // after a branch sits at word+2. Testing `w == p + 1`
                    // alone therefore reported every ordinary branch as a
                    // "non-contiguous" run end — 62% of all run ends in the
                    // first version of this measurement, which is what
                    // flagged the bug.
                    let contiguous = prev_word.map_or(false, |p| {
                        // Every word strictly between the previous head and
                        // this one must be an inlined slot, not a gap.
                        w > p && (p + 1..w).all(|g| {
                            let gi = &walked[g as usize];
                            gi.visited && gi.is_slot_only
                        })
                    });
                    let joinable = !instr.is_branch_target
                        && !instr.is_entry_point
                        && !instr.is_branch_fallback_successor
                        && !instr.is_fallback;

                    if run_len > 0 && contiguous && joinable {
                        run_len += 1;
                    } else {
                        // Close the previous run and attribute why it ended.
                        if run_len > 0 {
                            let b = (run_len as usize).min(MAX_BUCKET);
                            hist[b] += 1;
                            total_runs += 1;
                            if run_len >= 2 { heads_in_runs_ge2 += run_len; }
                            if !contiguous { end_page += 1; }
                            else if instr.is_branch_target { end_branch_target += 1; }
                            else if instr.is_entry_point || instr.is_branch_fallback_successor { end_entry += 1; }
                            else if instr.is_fallback { end_fallback += 1; }
                        }
                        run_len = 1;
                    }

                    // This word's own edges leaving the region end the run
                    // here, regardless of what word+1 looks like.
                    if instr.taken_exit.is_some() || instr.fallthrough_exit.is_some() {
                        let b = (run_len as usize).min(MAX_BUCKET);
                        hist[b] += 1;
                        total_runs += 1;
                        if run_len >= 2 { heads_in_runs_ge2 += run_len; }
                        end_exit += 1;
                        run_len = 0;
                        prev_word = None;
                        continue;
                    }
                    prev_word = Some(w);
                }
                if run_len > 0 {
                    let b = (run_len as usize).min(MAX_BUCKET);
                    hist[b] += 1;
                    total_runs += 1;
                    if run_len >= 2 { heads_in_runs_ge2 += run_len; }
                    end_page += 1;
                }
            }
        }

        println!("RUNS pages={} heads={} runs={}", paths.len(), total_heads, total_runs);
        if total_runs == 0 { return; }
        println!("  mean run length: {:.2} head instructions", total_heads as f64 / total_runs as f64);
        println!("  heads in runs of >=2: {} ({:.1}% of all heads)",
            heads_in_runs_ge2, 100.0 * heads_in_runs_ge2 as f64 / total_heads.max(1) as f64);
        // The merge saves (run_len - 1) block boundaries per run: that is the
        // number of instruction boundaries that stop being a barrier.
        let boundaries_removed: u64 = (2..=MAX_BUCKET)
            .map(|b| hist[b] * (b as u64 - 1))
            .sum();
        println!("  block boundaries removable: {} ({:.1}% of the {} that exist)",
            boundaries_removed,
            100.0 * boundaries_removed as f64 / total_heads.max(1) as f64,
            total_heads);
        println!("  run-length histogram (length: count):");
        for b in 1..=MAX_BUCKET {
            if hist[b] > 0 {
                let label = if b == MAX_BUCKET { format!("{}+", b) } else { b.to_string() };
                println!("    {:>4}: {:>9}  {:.1}%", label, hist[b], 100.0 * hist[b] as f64 / total_runs as f64);
            }
        }
        println!("  why runs end: region-exit={} branch-target={} entry/foreign-slot={} fallback={} non-contiguous={}",
            end_exit, end_branch_target, end_entry, end_fallback, end_page);
    }
}

/// Scratch: how does Cranelift lower a constant base address used many times?
/// `IRIS_CONST_MODE` selects the shape. Answers whether a baked constant can
/// ever match `disp(%reg)` density for struct-field access.
#[cfg(test)]
mod zz_constdedup {
    #[test]
    fn zz_const_dedup() {
        use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlagsData};
        use cranelift_codegen::settings::{self, Configurable};
        use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
        use cranelift_codegen::Context;
        let Some(mode) = std::env::var_os("IRIS_CONST_MODE") else { return };
        let mode = mode.to_string_lossy().to_string();

        let mut fb = settings::builder();
        fb.set("opt_level", "speed").unwrap();
        let isa = cranelift_native::builder().unwrap()
            .finish(settings::Flags::new(fb)).unwrap();

        let mut ctx = Context::new();
        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.returns.push(AbiParam::new(types::I64));
        let mut bctx = FunctionBuilderContext::new();
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut bctx);
            let blk = b.create_block();
            b.append_block_params_for_function_params(blk);
            b.switch_to_block(blk);
            b.seal_block(blk);
            const ADDR: i64 = 0x7f_abcd_1234_5678u64 as i64;
            let mut acc = b.ins().iconst(types::I64, 0);
            let shared = b.ins().iconst(types::I64, ADDR);
            let param  = b.block_params(blk)[0];
            for i in 0..40 {
                let off = (i * 8) as i32;
                let v = match mode.as_str() {
                    // baked constant re-materialized per use
                    "iconst"  => { let base = b.ins().iconst(types::I64, ADDR);
                                   b.ins().load(types::I64, MemFlagsData::trusted(), base, off) }
                    // one SSA constant, reused
                    "shared"  => b.ins().load(types::I64, MemFlagsData::trusted(), shared, off),
                    // const + explicit iadd_imm, then load at 0  (struct-ish)
                    "addimm"  => { let p = b.ins().iadd_imm(shared, off as i64);
                                   b.ins().load(types::I64, MemFlagsData::trusted(), p, 0) }
                    // base arrives in a register (today's design)
                    "param"   => b.ins().load(types::I64, MemFlagsData::trusted(), param, off),
                    // const forced through an opaque use first
                    "opaque"  => { let p = b.ins().bor_imm(shared, 0);
                                   b.ins().load(types::I64, MemFlagsData::trusted(), p, off) }
                    // Call through a baked function-pointer constant vs a
                    // pointer loaded from the struct: the case where baking
                    // should actually win (no load, no indirect predictor slot).
                    "callconst" => {
                        let mut sig = cranelift_codegen::ir::Signature::new(
                            isa.default_call_conv());
                        sig.params.push(AbiParam::new(types::I64));
                        sig.returns.push(AbiParam::new(types::I64));
                        let sr = b.import_signature(sig);
                        let callee = b.ins().iconst(types::I64, ADDR);
                        let c = b.ins().call_indirect(sr, callee, &[param]);
                        b.inst_results(c)[0]
                    }
                    "callload" => {
                        let mut sig = cranelift_codegen::ir::Signature::new(
                            isa.default_call_conv());
                        sig.params.push(AbiParam::new(types::I64));
                        sig.returns.push(AbiParam::new(types::I64));
                        let sr = b.import_signature(sig);
                        let callee = b.ins().load(types::I64, MemFlagsData::trusted(), param, off);
                        let c = b.ins().call_indirect(sr, callee, &[param]);
                        b.inst_results(c)[0]
                    }
                    _ => panic!("bad mode"),
                };
                acc = b.ins().iadd(acc, v);
            }
            b.ins().return_(&[acc]);
            b.finalize(isa.frontend_config());
        }
        ctx.set_disasm(true);
        let _ = ctx.compile(&*isa, &mut Default::default()).unwrap();
        let cc = ctx.compiled_code().unwrap();
        let vc = cc.vcode.as_ref().unwrap();
        let movabs = vc.lines().filter(|l| l.contains("movabsq")).count();
        println!("CONSTMODE {:8} movabs={:3} size={}", mode, movabs, cc.code_info().total_size);
        if std::env::var_os("IRIS_CONST_DUMP").is_some() {
            for l in vc.lines().filter(|l| l.contains("call") || l.contains("movabs")).take(8) { println!("  | {}", l); }
        }
    }
}

#[cfg(test)]
mod zz_offsets {
    #[test]
    fn zz_print_offsets() {
        if std::env::var("IRIS_PRINT_OFFSETS").is_err() { return; }
        use crate::mips_core::MipsCore;
        println!("OFF gpr    = {:#x}", std::mem::offset_of!(MipsCore, gpr));
        println!("OFF pc     = {:#x}", std::mem::offset_of!(MipsCore, pc));
        println!("OFF hot    = {:#x}", std::mem::offset_of!(MipsCore, hot));
        println!("OFF fpr    = {:#x}", std::mem::offset_of!(MipsCore, fpr));
        println!("OFF nutlb  = {:#x}", std::mem::offset_of!(MipsCore, nutlb));
        println!("SIZE core  = {:#x}", std::mem::size_of::<MipsCore>());
    }
}

/// Force the entry word's own interrupt preamble to be emitted, instead of
/// bypassing it via the dispatch head's body blocks (`skip_entry_preamble`).
///
/// Default **off**, matching the shipping behaviour. `IRIS_ENTRY_PREAMBLE=1`
/// at startup, or `j2 entrypre on` from the monitor (followed by `j2 flush`,
/// since already-compiled regions keep whatever shape they were built with).
///
/// Why it exists: an externally-dispatched entry word currently runs with no
/// interrupt check of its own, so delivery is deferred by one dispatch. That is
/// bounded and never *lost* — every internal back-edge onto an entry word still
/// pays the preamble — but the window scales with how fast the JIT retires
/// code, which makes it a candidate whenever a fault reproduces under a fast
/// JIT and disappears under the interpreter or lockstep.
static ENTRY_PREAMBLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// One-shot startup read of `IRIS_ENTRY_PREAMBLE`, so the env var works without
/// a monitor command.
static ENTRY_PREAMBLE_ENV: std::sync::OnceLock<()> = std::sync::OnceLock::new();

#[inline]
pub fn entry_preamble_forced() -> bool {
    ENTRY_PREAMBLE_ENV.get_or_init(|| {
        if std::env::var("IRIS_ENTRY_PREAMBLE").map(|v| v == "1").unwrap_or(false) {
            ENTRY_PREAMBLE.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    ENTRY_PREAMBLE.load(std::sync::atomic::Ordering::Relaxed)
}

/// `j2 entrypre on|off`. Caller must `j2 flush` afterward — already-compiled
/// regions are unaffected until rebuilt.
pub fn set_entry_preamble_forced(on: bool) {
    let _ = entry_preamble_forced(); // pin the env read first
    ENTRY_PREAMBLE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Serializes tests that flip [`set_entry_preamble_forced`].
///
/// `ENTRY_PREAMBLE` is a process-global read by **every** compile, so a test
/// that turns it on while another test is compiling a region in parallel gives
/// that region an entry-word interrupt check it never asked for — a flake that
/// shows up only in a full run, never when the test is filtered. Any test that
/// writes the toggle must hold this for the whole time it is non-default, and
/// restore the previous value before releasing.
#[cfg(test)]
pub static ENTRY_PREAMBLE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
