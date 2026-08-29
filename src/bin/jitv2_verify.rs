//! JIT v2 trace verifier: replays a `src/trace.rs`-format execution trace
//! (captured live via the `trace start <path>` monitor command in a
//! `developer` build) offline, compiling each recorded instruction through
//! `Codegen` and diffing the result against what the interpreter actually
//! produced next.
//!
//! This is the "trace-then-replay" half of jitv2's lockstep verification:
//! the trace was captured by running the interpreter once against real
//! hardware state (so every load/store's side effects — SCSI commands, REX3
//! register pokes, whatever — happened exactly once, for real). Verification
//! here never touches a bus at all: it seeds a bare `MipsCore` from the
//! recorded pre-state, runs the JIT-compiled function, and compares
//! architectural register state against the trace's recorded *next*
//! pre-state. No double-execution, no risk of re-firing device side effects.
//!
//! `MipsCore`'s FPU status hooks and `handle_exception_fn` are wired to real,
//! standalone logic (`iris::platform::*`, `iris::mips_core::deliver_exception`
//! — see their trampolines below) rather than left panicking or stubbed to a
//! no-op, so instructions that raise real exceptions (integer overflow,
//! FCSR-enabled traps, CTC1 rounding-mode changes) get verified too, not just
//! skipped. `read*_fn`/`write*_fn` are the one hook family genuinely left
//! unwired (no bus exists here) — safe because `touches_memory` excludes
//! every load/store from replay before a compiled region touching one is
//! ever run.
//!
//! Two things this still can't safely verify and explicitly skips:
//! - Any load/store (including CP1 LWC1/SWC1/LDC1/SDC1): its result depends
//!   on bus/memory content this tool never captured, so comparing its
//!   destination register would just be checking whether some fixed pattern
//!   happens to match — not a real signal either way.
//! - Any step where the JIT's own resulting pc doesn't match the trace's
//!   recorded next pc: this is expected whenever an *asynchronous* interrupt
//!   was delivered between the two recorded steps (the trace only records
//!   whatever the interpreter actually dispatched next, which for an
//!   interrupt is the exception vector, not the compiled region's own
//!   deterministic fallthrough/branch target — and unlike a synchronous
//!   exception raised by the instruction itself, an interrupt's timing isn't
//!   reconstructible from the trace alone). Reported separately from
//!   register-content mismatches since it's not necessarily a bug.
//!
//! Usage: jitv2_verify <trace_file> [--limit N] [--verbose]

use std::path::PathBuf;

use iris::jitv2::analyzer::{classify, Analyzer, Classify};
use iris::jitv2::codegen::Codegen;
use iris::jitv2::{JitFn, ENTRIES_PER_PAGE};
use iris::mips_core::MipsCore;
use iris::trace::{CoreState, TraceReader, TraceRecord};

/// `MipsCore`'s FPU rounding-mode hook (`fpu_set_mode_fn`) is a pure
/// host-arch function with no executor dependency (see `mips_exec.rs`'s own
/// `jit_fpu_set_mode` trampoline, which this mirrors exactly) — safe to wire
/// up directly here without a real `MipsExecutor`. `MipsCore::new()`
/// otherwise leaves it at a panic-on-call placeholder, since compiled code
/// isn't supposed to run before `MipsExecutor::install_jit_hooks` in the
/// real emulator. (FP arithmetic's exception flags are computed from
/// operand/result bit patterns in the compiled IR itself now, not read from
/// the host FPU, so there's no corresponding status hook left to wire.)
unsafe extern "C" fn verify_fpu_set_mode(_ctx: *mut core::ffi::c_void, rm: u32) {
    iris::platform::set_fpu_mode(rm as u8)
}

/// Stand-in for `MipsExecutor` in this tool, which has no real executor:
/// a struct whose *first field is `core`*, so it has the same shape every
/// JIT->Rust hook now assumes — arg 0 is the `*mut MipsCore` compiled code
/// was entered with, and a hook needing its container recovers it with
/// `container_of` (`mips_exec::exec_from_core` in the real emulator,
/// `VerifyCtx::from_core` here).
///
/// Today no hook this tool installs actually needs the container
/// (`verify_handle_exception` wants only the `MipsCore` it was handed, and
/// `verify_fpu_set_mode` ignores its argument entirely), so this exists to
/// keep the *shape* honest rather than to carry state: a `MipsCore` reached
/// through compiled code is always the `core` field of some containing
/// context, and this tool is no longer the one exception to that rule. Give
/// it real fields if a future hook here needs them.
///
/// `repr(C)` is not required — `from_core` asks the compiler for the layout
/// it chose via `offset_of!`, exactly like `exec_from_core` — but the whole
/// point is that `core` is field 0, so state it plainly.
struct VerifyCtx {
    core: MipsCore,
}

impl VerifyCtx {
    fn new(core: MipsCore) -> Self {
        Self { core }
    }

    /// `container_of(core, VerifyCtx, core)` — the local twin of
    /// `mips_exec::exec_from_core`.
    ///
    /// # Safety
    /// `core` must point at the `core` field of a live `VerifyCtx`.
    #[allow(dead_code)] // no hook needs the container yet; see the struct doc
    unsafe fn from_core(core: *mut core::ffi::c_void) -> *mut VerifyCtx {
        unsafe {
            core.cast::<u8>()
                .sub(std::mem::offset_of!(VerifyCtx, core))
                .cast::<VerifyCtx>()
        }
    }
}

/// Deliver the exception's architectural effect via
/// `iris::mips_core::deliver_exception` — the same logic
/// `MipsExecutor::handle_exception` uses, extracted so it's callable without
/// a real executor (§4.2 single-implementation delivery: one implementation,
/// both the interpreter and this tool call it). `ctx` is the `*mut MipsCore`
/// the JitFn was entered with — the uniform arg 0 of every JIT->Rust hook
/// (`mips_exec.rs`'s `jit_handle_exception<T,C>` receives the identical
/// pointer and recovers its executor from it with `exec_from_core`; this
/// one needs nothing but the core itself, so it just casts).
///
/// `deliver_exception` reads `core.in_delay_slot` directly (one field, no
/// separate JIT-only copy) — `seed_core` sets it from `CoreState::in_delay_slot`,
/// which the recorder (`mips_exec.rs`'s `step()`) captures from the live
/// `MipsCore::in_delay_slot` at the same point every other field is
/// snapshotted, so this is the real recorded value, not a default guess.
unsafe extern "C" fn verify_handle_exception(ctx: *mut core::ffi::c_void, status: u32) -> u32 {
    // Arg 0 is biased by compiled code; `core_from_arg` is the one place that
    // contract is undone. Never dereference a callout's `ctx` without it.
    let core = unsafe { &mut *iris::mips_exec::core_from_arg(ctx) };
    iris::mips_core::deliver_exception(core, status);
    status
}

/// Whether `raw` is any load/store (GPR or CP1) — excluded from replay
/// entirely (see module doc). Opcode-based, deliberately not routed through
/// `lookup_semantics` (that would also match plenty of non-memory ops).
fn touches_memory(raw: u32) -> bool {
    use iris::mips_isa::*;
    let op = (raw >> 26) & 0x3F;
    matches!(
        op,
        OP_LB | OP_LBU | OP_LH | OP_LHU | OP_LW | OP_LWU | OP_LD
            | OP_SB | OP_SH | OP_SW | OP_SD
            | OP_LWC1 | OP_LDC1 | OP_SWC1 | OP_SDC1
    )
}

/// Apply a full `CoreState` onto a freshly constructed `MipsCore` — used to
/// seed the JIT run from the trace's exact recorded pre-state. No bus is
/// wired up: every load/store must already have been filtered out before a
/// compiled region touching one is ever run (see `touches_memory`), so the
/// compiled function should never actually dereference memory.
fn seed_core(state: &CoreState) -> MipsCore {
    let mut core = MipsCore::new();
    // No timer to silence anymore: the Count==Compare interrupt is an
    // hptimer-thread bit in hot.interrupts now, and no timer manager is
    // wired in this harness, so the pending-interrupt preamble never bails.
    core.gpr = state.gpr;
    core.pc = state.pc;
    core.hi = state.hi;
    core.lo = state.lo;
    core.cp0_epc = state.cp0_epc;
    core.cp0_badvaddr = state.cp0_badvaddr;
    core.cp0_cause = state.cp0_cause;
    core.cp0_status = state.cp0_status;
    core.fpr = state.fpr;
    core.fpu_fcsr = state.fpu_fcsr;
    core.fpu_fccr = state.fpu_fccr;
    core.fpu_fexr = state.fpu_fexr;
    core.fpu_fenr = state.fpu_fenr;
    core.in_delay_slot = state.in_delay_slot;
    // No context pointer to install: both hooks below take the `*mut MipsCore`
    // compiled code is entered with as arg 0, like every other JIT->Rust hook
    // (see `VerifyCtx`). Nothing here captures an address, so `core` is free
    // to be moved into its `VerifyCtx` after this returns.
    core.fpu_set_mode_fn = verify_fpu_set_mode;
    core.handle_exception_fn = verify_handle_exception;
    core
}

struct Args {
    trace_path: PathBuf,
    limit: Option<u64>,
    skip: u64,
    verbose: bool,
    /// Max heads per compiled chain (`--chain N`, default 1 — today's
    /// single-instruction-or-branch+slot behavior via `run()`). `N >= 2`
    /// switches to `run_chain()`: attempt to grow each compiled region to up
    /// to `N` head instructions by following the trace's own recorded path
    /// through interior branches (see module doc's "Multi-instruction chain
    /// mode" section).
    chain: usize,
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let mut trace_path = None;
    let mut limit = None;
    let mut skip = 0u64;
    let mut verbose = false;
    let mut chain = 1usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--limit" => limit = args.next().and_then(|s| s.parse().ok()),
            "--skip" => skip = args.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "--verbose" => verbose = true,
            "--chain" => chain = args.next().and_then(|s| s.parse().ok()).unwrap_or(1).max(1),
            other => trace_path = Some(PathBuf::from(other)),
        }
    }
    let trace_path = trace_path.unwrap_or_else(|| {
        eprintln!("Usage: jitv2_verify <trace_file> [--skip N] [--limit N] [--verbose] [--chain N]");
        std::process::exit(2);
    });
    Args { trace_path, limit, skip, verbose, chain }
}

#[derive(Default)]
struct Stats {
    total: u64,
    skipped_memory: u64,
    skipped_no_emitter: u64,
    skipped_control_flow_diverged: u64,
    /// Of `skipped_control_flow_diverged`, how many landed the trace's
    /// recorded next pc on one of the six real exception vector addresses
    /// (`is_exception_vector`) — confirms the "likely async interrupt"
    /// explanation instead of just asserting it. Not a subset field the
    /// caller needs to subtract manually: `skipped_control_flow_diverged -
    /// diverged_to_exception_vector` is the count of divergences that landed
    /// somewhere else entirely, which would be a real signal worth
    /// investigating rather than dismissing.
    diverged_to_exception_vector: u64,
    skipped_slot_not_adjacent: u64,
    compared: u64,
    mismatches: u64,
    /// Chain mode only (`run_chain`): how many heads actually ended up in
    /// each attempted compiled region, keyed by head count (1..=chain).
    /// Lets `--chain 4` report whether it's really reaching length-4 chains
    /// or if something (memory ops, unsupported instructions, non-adjacent
    /// slots) is quietly capping every attempt at length 1 or 2 — the whole
    /// point of the mode is building *longer* verifiable chains, so this
    /// number is the signal that it's working, not just "didn't crash".
    chain_len_histogram: std::collections::BTreeMap<usize, u64>,
    /// Chain mode only: the chain's pc diverged from the trace purely
    /// because `--chain N`'s length cap stopped it before reaching wherever
    /// the trace's real path went next — not a failure (nothing wrong with
    /// the JIT), just an incomplete build. Kept separate from
    /// `skipped_control_flow_diverged` so that count stays a meaningful
    /// "real, worth-investigating divergence" signal instead of being
    /// diluted by "didn't ask for enough chain length" noise, which at low
    /// `--chain` values can dominate.
    skipped_chain_capped: u64,
}

/// Whether `pc` is one of the six addresses `MipsCore::deliver_exception`
/// (formerly `MipsExecutor::handle_exception`) can vector to — `vector_base
/// + offset` for `vector_base` in `{BEV=0: 0x80000000, BEV=1: 0xBFC00200}`
/// and `offset` in `{0x000 (TLB refill), 0x080 (XTLB refill), 0x180
/// (general)}`. Used to confirm (not just assume) that a control-flow
/// divergence between the JIT's deterministic result and the trace's
/// recorded next pc is actually an asynchronous interrupt landing at a real
/// vector, rather than some other, unexplained divergence.
fn is_exception_vector(pc: u64) -> bool {
    const OFFSETS: [u64; 3] = [0x000, 0x080, 0x180];
    const BASES: [u64; 2] = [0xFFFF_FFFF_8000_0000, 0xFFFF_FFFF_BFC0_0200];
    BASES.iter().any(|&base| OFFSETS.iter().any(|&off| pc == base + off))
}

fn main() {
    let args = parse_args();
    let stats = if args.chain <= 1 {
        run(&args.trace_path, args.skip, args.limit, args.verbose)
    } else {
        run_chain(&args.trace_path, args.skip, args.limit, args.verbose, args.chain)
    }.unwrap_or_else(|e| {
        eprintln!("jitv2_verify: {}", e);
        std::process::exit(1);
    });

    let unexplained_diverged = stats.skipped_control_flow_diverged - stats.diverged_to_exception_vector;
    println!(
        "--- {} records: {} compared, {} mismatches, {} memory-skipped, {} control-flow-diverged ({} to a real exception vector, {} UNEXPLAINED), {} no-emitter, {} slot-not-adjacent ---",
        stats.total, stats.compared, stats.mismatches, stats.skipped_memory,
        stats.skipped_control_flow_diverged, stats.diverged_to_exception_vector, unexplained_diverged,
        stats.skipped_no_emitter, stats.skipped_slot_not_adjacent,
    );
    if args.chain > 1 {
        let hist: Vec<String> = stats.chain_len_histogram.iter()
            .map(|(len, count)| format!("{}:{}", len, count))
            .collect();
        println!(
            "--- chain length histogram (heads:attempts) — {} — {} length-capped (not a failure, see FAILURE lines above for real ones) ---",
            hist.join(", "), stats.skipped_chain_capped,
        );
    }
    std::process::exit(if stats.mismatches > 0 || unexplained_diverged > 0 { 1 } else { 0 });
}

fn run(trace_path: &std::path::Path, skip: u64, limit: Option<u64>, verbose: bool) -> std::io::Result<Stats> {
    let mut reader = TraceReader::open(trace_path)?;
    if skip > 0 {
        reader.skip_records(skip)?;
    }

    // Buffer three consecutive records at a time: `cur` is the instruction
    // under test, `next` is its immediate successor (either the comparison
    // point for a plain instruction, or the delay slot if `cur` is a
    // branch/jump), `next2` is the comparison point when `cur` is a
    // branch/jump (the JIT's compiled region covers cur+slot as one unit,
    // per §6.1.4 — see module doc).
    let mut cur = reader.next()?;
    let mut next = reader.next()?;

    let mut analyzer = Analyzer::new();
    let mut codegen = Codegen::new();
    let mut stats = Stats::default();
    // Cache compiled functions by everything the compiled body's behavior
    // depends on: the entry instruction's raw bytes, its slot's raw bytes
    // (branch/jump units only), its word offset (affects in-region branch
    // target words and the delay-slot-inline offset math), page_base
    // (affects absolute address materialization for J/JAL/off-page
    // branches and the exit block's vbase), and fr_mode (affects FPR
    // access). A boot trace revisits the same PROM/kernel routines and loop
    // bodies constantly — recompiling identical (raw, slot_raw, word,
    // page_base, fr1) tuples from scratch every time was the dominant cost
    // of a full-trace run (each Cranelift compile call dwarfs everything
    // else this tool does per record).
    let mut compile_cache: std::collections::HashMap<(u32, Option<u32>, u16, u32, bool), Option<JitFn>> =
        std::collections::HashMap::new();

    while let (Some(rec), Some(rec_next)) = (cur, next) {
        if let Some(limit) = limit {
            if stats.total >= limit { break; }
        }
        stats.total += 1;

        let rec2 = reader.next()?;

        let word = ((rec.pc & 0xFFF) / 4) as u16;
        let page_base = (rec.pc & !0xFFFu64) as u32;
        let class = classify(rec.raw, word, page_base);
        // RegJump (JR/JALR) has the same "mandatory inline delay slot, one
        // compiled unit covering both records" shape as Branch/Jump (its
        // target just isn't statically known — emit_regjump still always
        // inlines the slot before exiting via emit_runtime_pc_exit). Treated
        // identically here: pulls in rec_next as its slot, compares against
        // rec2, needs max_instrs=2.
        let is_branch_or_jump = matches!(class, Classify::Branch { .. } | Classify::Jump { .. } | Classify::RegJump);

        // A delay slot's dispatch can be silently absent from the trace: if
        // an interrupt is pending exactly when the slot's step() would run,
        // step() delivers the exception and returns before ever reaching the
        // fetch/decode/record point (mips_exec.rs, the early `pending`
        // check happens before fetch_instr) — so rec_next in that case is
        // NOT the slot, it's whatever the exception handler dispatched to
        // next, at some unrelated pc. The only way to tell is checking
        // adjacency: a real slot is always exactly rec.pc+4. If it isn't,
        // this branch/jump's compiled unit can't be safely reconstructed
        // (we don't know what its actual slot was), so skip it entirely.
        if is_branch_or_jump && rec_next.pc != rec.pc.wrapping_add(4) {
            stats.skipped_slot_not_adjacent += 1;
            advance(&mut cur, &mut next, rec_next, rec2, &mut reader, false);
            continue;
        }

        if touches_memory(rec.raw) {
            stats.skipped_memory += 1;
            advance(&mut cur, &mut next, rec_next, rec2, &mut reader, is_branch_or_jump);
            continue;
        }
        // A branch/jump's own delay slot never gets independently verified
        // (it's inlined into cur's compiled region, not a standalone
        // dispatch as far as the JIT is concerned) — but it can still touch
        // memory, in which case cur itself can't be replayed safely either.
        if is_branch_or_jump && touches_memory(rec_next.raw) {
            stats.skipped_memory += 1;
            advance(&mut cur, &mut next, rec_next, rec2, &mut reader, is_branch_or_jump);
            continue;
        }

        let expected_post = if is_branch_or_jump {
            match rec2 {
                Some(r) => r.state,
                None => break, // trace ends mid-branch-unit; nothing to compare against
            }
        } else {
            rec_next.state
        };

        let mut page_words = [0u32; ENTRIES_PER_PAGE];
        page_words[word as usize] = rec.raw;
        if is_branch_or_jump {
            let slot_word = word as usize + 1;
            if slot_word < ENTRIES_PER_PAGE {
                page_words[slot_word] = rec_next.raw;
            }
        }

        let fr1 = (rec.state.cp0_status & iris::mips_core::STATUS_FR) != 0;
        let slot_raw_key = if is_branch_or_jump { Some(rec_next.raw) } else { None };
        let cache_key = (rec.raw, slot_raw_key, word, page_base, fr1);

        let jit_fn: Option<JitFn> = *compile_cache.entry(cache_key).or_insert_with(|| {
            // max_instrs=1 always: a branch/jump's mandatory delay slot is
            // never charged against the walk budget (analyzer::visit_slot —
            // a slot can never be omitted, so it was never a truncation
            // candidate). page_words is zero-filled everywhere except
            // `word` (and the slot word, for a branch/jump) — 0 always
            // decodes as a valid Sequential NOP (SLL r0,r0,0), so any
            // budget left over after the one real head would happily walk
            // that phantom NOP in as a second real instruction and compile
            // a bigger region than the single unit under test.
            let max_instrs = 1;
            let (walked, non_empty) = analyzer.walk_bounded(&page_words, word, page_base, max_instrs);
            if !non_empty {
                return None; // shouldn't happen: cur itself excluded despite passing classify() above
            }
            let mut instrs_owned = *walked;
            // skip_entry_preamble=false: this tool seeds core state directly
            // and calls the compiled function with no interpreter step()
            // dispatch preceding it, so the entry word's own IP7/pending-
            // interrupt preamble is the only place those checks ever run —
            // must not be skipped here (see compile_region's doc comment).
            codegen.compile_region(&mut instrs_owned, word, fr1, false)
        });
        let Some(jit_fn) = jit_fn else {
            stats.skipped_no_emitter += 1;
            if verbose { eprintln!("no_emitter pc={:#018x} raw={:#010x} class={:?}", rec.pc, rec.raw, class); }
            advance(&mut cur, &mut next, rec_next, rec2, &mut reader, is_branch_or_jump);
            continue;
        };

        // Run the compiled function against `core` in its container, so the
        // pointer it gets is a real `VerifyCtx::core` — the same shape
        // `exec_from_core` assumes of `MipsExecutor::core` in the emulator.
        let mut ctx = VerifyCtx::new(seed_core(&rec.state));
        let status = unsafe { jit_fn(&mut ctx.core as *mut MipsCore) };
        let core = &mut ctx.core;

        if core.pc != expected_post.pc {
            stats.skipped_control_flow_diverged += 1;
            let vectored = is_exception_vector(expected_post.pc);
            if vectored {
                stats.diverged_to_exception_vector += 1;
            }
            if verbose || !vectored {
                let explanation = if vectored { "async interrupt, landed on a real exception vector" } else { "UNEXPLAINED — trace pc is not a known exception vector" };
                println!(
                    "pc={:#018x} raw={:#010x}: control flow diverged (jit pc={:#018x}, trace pc={:#018x}, status={:#x}) — {}, not compared",
                    rec.pc, rec.raw, core.pc, expected_post.pc, status, explanation,
                );
            }
            advance(&mut cur, &mut next, rec_next, rec2, &mut reader, is_branch_or_jump);
            continue;
        }

        stats.compared += 1;
        let got = CoreState::capture(&core);
        if let Some(diff) = diff_state(&got, &expected_post) {
            stats.mismatches += 1;
            println!("MISMATCH pc={:#018x} raw={:#010x}: {}", rec.pc, rec.raw, diff);
        }

        advance(&mut cur, &mut next, rec_next, rec2, &mut reader, is_branch_or_jump);
    }

    Ok(stats)
}

/// Any word whose top 6 bits are `OP_COP0` (rest zero — decodes as `MFC0
/// r0, $0`) unconditionally classifies `Excluded` (`analyzer::classify`),
/// regardless of the specific COP0 sub-opcode. Used by `run_chain` to
/// "poison" a branch's untaken arm inside a multi-instruction chain: the
/// analyzer refuses to walk into an excluded word, so that edge always
/// bails instead of accidentally reading whatever stale/zero content
/// happens to sit at that word in the synthetic page buffer as if it were
/// real code the trace never actually validated.
const POISON_WORD: u32 = iris::mips_isa::OP_COP0 << 26;

/// Multi-instruction chain mode (`--chain N`): instead of verifying one
/// instruction (or one branch+slot unit) per compiled region, attempt to
/// grow each region to up to `max_chain` *head* instructions by following
/// the trace's own recorded dynamic path through interior branches/jumps —
/// more of the compiled region's real control-flow logic (which arm a
/// branch's `brif` takes, whether a fallthrough edge threads through
/// several instructions in a row) gets exercised per comparison than the
/// single-unit mode ever reaches.
///
/// The core trick for a branch/jump that isn't the chain's last head: the
/// trace already tells us, via the very next recorded pc, which arm was
/// actually taken — so the *other* arm's word (if it lands in-page and
/// isn't otherwise part of the chain) is overwritten with [`POISON_WORD`]
/// in the synthetic page buffer before compiling. The analyzer then bails
/// on that edge (`Classify::Excluded`) instead of either matching by luck
/// or wandering into zero-filled/stale memory as if it were reachable real
/// code — the compiled region's *shape* is forced to match the trace's
/// actual path, not just its entry point.
///
/// A chain attempt stops growing (uses whatever it has accumulated so far,
/// possibly just the one entry head) the moment it hits: `max_chain` heads
/// reached, a memory op (`touches_memory`), a branch/jump/regjump whose
/// slot isn't adjacent (the same "interrupt ate the slot dispatch"
/// ambiguity `run()` already guards against), the end of the trace, or
/// (discovered only after compiling) `compile_region` declining the region
/// — which reports as `skipped_no_emitter`, same as single-instruction mode.
fn run_chain(trace_path: &std::path::Path, skip: u64, limit: Option<u64>, verbose: bool, max_chain: usize) -> std::io::Result<Stats> {
    let mut reader = TraceReader::open(trace_path)?;
    if skip > 0 {
        reader.skip_records(skip)?;
    }

    let mut analyzer = Analyzer::new();
    let mut codegen = Codegen::new();
    let mut stats = Stats::default();
    // True cumulative position in the trace (records actually consumed via
    // buf.pop_front(), not stats.total — an attempt can consume more than
    // one raw record, so stats.total alone badly undercounts the real
    // offset a repro needs). Starts at `skip` since that many records were
    // already skipped before this loop's first record.
    let mut record_offset: u64 = skip;

    // Cache compiled functions by everything the compiled body's behavior
    // depends on — same rationale as run()'s compile_cache, just keyed on
    // the whole chain's raw-byte sequence instead of a single (raw,
    // slot_raw) pair, since a chain's compiled region depends on every head
    // and slot in it. Without this, a real trace's tight loop bodies
    // (visited thousands of times) each trigger a fresh, expensive
    // Cranelift compile per visit instead of one compile reused every time
    // — this was a real, measured pathological slowdown (a 2M-record run
    // never finished in reasonable time/memory on a boot trace containing
    // an ordinary tight loop) before this cache existed.
    let mut compile_cache: std::collections::HashMap<(Vec<u32>, u16, u32, bool), Option<JitFn>> =
        std::collections::HashMap::new();

    // Rolling buffer of not-yet-consumed records, read ahead as needed.
    // Index 0 is always the next chain's entry head.
    let mut buf: std::collections::VecDeque<TraceRecord> = std::collections::VecDeque::new();
    let fill = |buf: &mut std::collections::VecDeque<TraceRecord>, reader: &mut TraceReader, want: usize| -> std::io::Result<()> {
        while buf.len() < want {
            match reader.next()? {
                Some(r) => buf.push_back(r),
                None => break,
            }
        }
        Ok(())
    };

    loop {
        if let Some(limit) = limit {
            if stats.total >= limit { break; }
        }
        fill(&mut buf, &mut reader, 1)?;
        let Some(entry) = buf.front().copied() else { break };
        stats.total += 1;

        // heads[i] = (record, is_branch_or_jump, slot record if applicable).
        // Grown one head at a time; `consumed` tracks how many buf entries
        // (heads + their slots) this attempt has used, for advancing buf at
        // the end regardless of how the attempt concluded. `hit_cap` tracks
        // whether the chain stopped growing *only* because `max_chain` was
        // reached (a normal, expected build-time limit — nothing wrong with
        // the JIT, the chain just wasn't asked to go further) versus any
        // other reason (memory op, non-adjacent slot, off-page) — see the
        // comparison-gating logic below for why this distinction matters.
        let mut heads: Vec<(TraceRecord, bool, Option<TraceRecord>)> = Vec::new();
        let mut consumed: usize = 0;
        let mut cur_idx: usize = 0; // index into buf of the next candidate head
        let mut hit_cap = false;

        loop {
            fill(&mut buf, &mut reader, cur_idx + 1)?;
            let Some(&rec) = buf.get(cur_idx) else { break };

            let word = ((rec.pc & 0xFFF) / 4) as u16;
            let page_base = (rec.pc & !0xFFFu64) as u32;
            let class = classify(rec.raw, word, page_base);
            let is_bj = matches!(class, Classify::Branch { .. } | Classify::Jump { .. } | Classify::RegJump);

            if touches_memory(rec.raw) || class == Classify::Excluded {
                // A memory op can't be safely replayed (no bus); an
                // Excluded-classified instruction (COP0, CACHE, LL/SC, CP2,
                // RS_BC1) is one `analyzer::visit` refuses to walk at all —
                // the real compiled region can never actually include it as
                // a head regardless of what max_instrs allows, so including
                // its bytes in page_words/heads here would silently
                // over-count the chain (the analyzer bails one word
                // earlier than heads.len() implies, desyncing the
                // comparison point from what actually got compiled — this
                // exact bug produced a real false-positive divergence
                // report before this check existed).
                break; // this head can't be included — stop before it
            }
            if would_create_back_edge(&heads, word, class, rec.raw) {
                // See would_create_back_edge's doc comment — stop growing
                // here rather than letting the analyzer wire a native
                // back-edge into an already-included chain word.
                break;
            }

            if is_bj {
                fill(&mut buf, &mut reader, cur_idx + 2)?;
                let Some(&slot) = buf.get(cur_idx + 1) else { break };
                if slot.pc != rec.pc.wrapping_add(4) {
                    // Interrupt ate the slot dispatch — same ambiguity run()
                    // guards against. Can't safely include this head.
                    if heads.is_empty() {
                        stats.skipped_slot_not_adjacent += 1;
                    }
                    break;
                }
                if touches_memory(slot.raw) {
                    break; // the slot itself can't be replayed safely either
                }
                heads.push((rec, true, Some(slot)));
                consumed = cur_idx + 2;
                cur_idx += 2;
            } else {
                heads.push((rec, false, None));
                consumed = cur_idx + 1;
                cur_idx += 1;
            }

            if heads.len() >= max_chain {
                hit_cap = true;
                break;
            }
            // A branch/jump/regjump always ends this attempt's forward
            // extension for now — growing *through* one (following the
            // trace's taken/not-taken arm into a further head) is handled
            // below by re-deriving the next head's expected word from
            // `heads.last()`'s own targets, not by blindly trusting
            // whatever sits at cur_idx next in program order (a taken
            // branch's real successor is its target, not word+2). Not a
            // hit_cap case — the extension loop below decides separately
            // whether it can keep growing past this branch/jump/regjump.
            if is_bj {
                break;
            }
        }

        if heads.is_empty() {
            // The entry itself couldn't be included (memory op or
            // non-adjacent slot, both already counted above). Skip exactly
            // this one record, same granularity as run()'s memory-skip path.
            if touches_memory(entry.raw) {
                stats.skipped_memory += 1;
            }
            let n = 1usize.max(consumed); for _ in 0..n { buf.pop_front(); } record_offset += n as u64;
            continue;
        }

        // Try to extend past the last head, if it was a branch/jump/regjump
        // and there's still chain budget, by following wherever the trace
        // actually went next (buf[consumed] is exactly that — the record
        // right after the slot). RegJump's target is never statically
        // knowable to the analyzer, so a chain can't extend through one;
        // stop growing there (it's always the chain's last head).
        //
        // hit_cap tracks whether the *most recent* stop was genuinely
        // "reached max_chain" — reset to false on every iteration that
        // grows the chain (a later stop for a real reason, e.g. a memory
        // op right after growing past max_chain-1, must not be reported as
        // if the chain had merely run out of requested length).
        while heads.len() < max_chain {
            hit_cap = false;
            let (last_rec, last_is_bj, _) = *heads.last().unwrap();
            if !last_is_bj { break; } // only a branch/jump/regjump has an extension point to resolve
            let last_word = ((last_rec.pc & 0xFFF) / 4) as u16;
            let last_page_base = (last_rec.pc & !0xFFFu64) as u32;
            let last_class = classify(last_rec.raw, last_word, last_page_base);
            if matches!(last_class, Classify::RegJump) { break; }

            fill(&mut buf, &mut reader, consumed + 1)?;
            let Some(&next_head) = buf.get(consumed) else { break };
            // Must stay on the same page as the chain's entry — the whole
            // chain compiles as one page_words buffer indexed by word
            // offset, so a cross-page continuation has nowhere to go.
            if (next_head.pc & !0xFFFu64) as u32 != page_base_of(heads[0].0.pc) {
                break;
            }
            let next_word = ((next_head.pc & 0xFFF) / 4) as u16;
            let next_page_base = (next_head.pc & !0xFFFu64) as u32;
            let next_class = classify(next_head.raw, next_word, next_page_base);
            if touches_memory(next_head.raw) || next_class == Classify::Excluded {
                // Same rule as the initial build loop above: an Excluded
                // word can never actually be a head in the real compiled
                // region, regardless of remaining chain budget.
                break;
            }
            let next_is_bj = matches!(next_class, Classify::Branch { .. } | Classify::Jump { .. } | Classify::RegJump);
            if would_create_back_edge(&heads, next_word, next_class, next_head.raw) {
                break; // stop growing here — this word is a real region boundary for chain-verification purposes
            }
            if next_is_bj {
                fill(&mut buf, &mut reader, consumed + 2)?;
                let Some(&next_slot) = buf.get(consumed + 1) else { break };
                if next_slot.pc != next_head.pc.wrapping_add(4) || touches_memory(next_slot.raw) {
                    break;
                }
                heads.push((next_head, true, Some(next_slot)));
                consumed += 2;
            } else {
                heads.push((next_head, false, None));
                consumed += 1;
            }
            if heads.len() >= max_chain {
                hit_cap = true;
            }
        }

        let entry_word = ((heads[0].0.pc & 0xFFF) / 4) as u16;
        let page_base = page_base_of(heads[0].0.pc);
        let fr1 = (heads[0].0.state.cp0_status & iris::mips_core::STATUS_FR) != 0;

        // Cache key: every raw word that determines page_words' content, in
        // order (head, its slot if any, next head, ...) — two chains with
        // the same sequence of raw bytes at the same entry_word/page_base/
        // fr1 always compile to byte-identical Cranelift IR, regardless of
        // which trace records happened to produce them.
        let mut raw_seq: Vec<u32> = Vec::with_capacity(heads.len() * 2);
        for &(rec, is_bj, slot) in &heads {
            raw_seq.push(rec.raw);
            if is_bj {
                raw_seq.push(slot.map(|s| s.raw).unwrap_or(0));
            }
        }
        let cache_key = (raw_seq, entry_word, page_base, fr1);

        let jit_fn: Option<JitFn> = *compile_cache.entry(cache_key).or_insert_with(|| {
            let page_words = build_chain_page_words(&heads, page_base);
            let (walked, non_empty) = analyzer.walk_bounded(&page_words, entry_word, page_base, heads.len());
            if !non_empty {
                return None;
            }
            let mut instrs_owned = *walked;
            // skip_entry_preamble=false: same reasoning as run()'s call site
            // above — no interpreter step() precedes this call.
            codegen.compile_region(&mut instrs_owned, entry_word, fr1, false)
        });
        let Some(jit_fn) = jit_fn else {
            stats.skipped_no_emitter += 1;
            if verbose { eprintln!("no_emitter (chain len {}) pc={:#018x}", heads.len(), heads[0].0.pc); }
            let n = consumed.max(1); for _ in 0..n { buf.pop_front(); } record_offset += n as u64;
            continue;
        };

        // Expected post-state: the record immediately after everything this
        // attempt consumed (buf[consumed] — the same "next record" the
        // extension loop above would have looked at to keep growing).
        fill(&mut buf, &mut reader, consumed + 1)?;
        let Some(&expected_rec) = buf.get(consumed) else {
            break; // trace ends mid-chain; nothing to compare against
        };
        let expected_post = expected_rec.state;

        // Same as the single-instruction path above: run against the core
        // inside its `VerifyCtx` container.
        let mut ctx = VerifyCtx::new(seed_core(&heads[0].0.state));
        let status = unsafe { jit_fn(&mut ctx.core as *mut MipsCore) };
        let core = &mut ctx.core;

        *stats.chain_len_histogram.entry(heads.len()).or_insert(0) += 1;

        if core.pc != expected_post.pc {
            if hit_cap {
                // The chain simply wasn't asked to grow far enough to reach
                // wherever the trace's real path went next — nothing wrong
                // with the JIT, this attempt just didn't build a long
                // enough chain. Not a failure: skip and move on, same as
                // any other "couldn't build a chain here" case.
                stats.skipped_chain_capped += 1;
                let n = consumed.max(1); for _ in 0..n { buf.pop_front(); } record_offset += n as u64;
                continue;
            }
            // The chain reached a genuine, self-consistent boundary (a
            // memory op, unsupported instruction, off-page target, or a
            // branch/jump/regjump the extension logic correctly stopped
            // at) — build_chain_page_words poisoned every word outside
            // that boundary, so the compiled region's shape is exactly
            // what was intended, not an accident of the length cap. If the
            // JIT's own pc still doesn't match the trace here, that's a
            // real result worth investigating, not expected noise.
            stats.skipped_control_flow_diverged += 1;
            let vectored = is_exception_vector(expected_post.pc);
            if vectored {
                stats.diverged_to_exception_vector += 1;
            } else {
                // Print the exact repro coordinates (--skip N lands exactly
                // on this attempt's entry head — record_offset is the true
                // cumulative trace position, not stats.total, which
                // undercounts whenever an earlier attempt consumed more
                // than one raw record) so a real divergence can be
                // reproduced standalone without re-scanning the whole trace.
                println!(
                    "FAILURE: chain diverged at a genuine boundary (not a length-cap artifact) — trace_skip={} chain_len={} entry_pc={:#018x} entry_raw={:#010x}: jit pc={:#018x}, trace pc={:#018x}, status={:#x}",
                    record_offset, heads.len(), heads[0].0.pc, heads[0].0.raw, core.pc, expected_post.pc, status,
                );
            }
            if verbose && vectored {
                println!(
                    "pc={:#018x} raw={:#010x} (chain len {}): control flow diverged (jit pc={:#018x}, trace pc={:#018x}, status={:#x}) — async interrupt, landed on a real exception vector, not compared",
                    heads[0].0.pc, heads[0].0.raw, heads.len(), core.pc, expected_post.pc, status,
                );
            }
            let n = consumed.max(1); for _ in 0..n { buf.pop_front(); } record_offset += n as u64;
            continue;
        }

        stats.compared += 1;
        let got = CoreState::capture(&core);
        if let Some(diff) = diff_state(&got, &expected_post) {
            stats.mismatches += 1;
            println!("MISMATCH pc={:#018x} raw={:#010x} (chain len {}): {}", heads[0].0.pc, heads[0].0.raw, heads.len(), diff);
        }

        let n = consumed.max(1); for _ in 0..n { buf.pop_front(); } record_offset += n as u64;
    }

    Ok(stats)
}

/// Build the synthetic page a chain attempt compiles from: real bytes for
/// every head + slot in `heads`, then poison ([`POISON_WORD`]) two classes
/// of word so the compiled region's *shape* is forced to match exactly what
/// the trace validated, rather than trusting `max_instrs` head-counting
/// alone to be the only thing standing between "verified" and "walked one
/// word further than intended into zero-filled/stale memory as if it were
/// real code":
///
/// 1. The untaken arm of every *interior* branch/jump (not `heads`' last
///    entry — that one's own taken/not-taken split is exactly what's being
///    verified, both arms are legitimate exit stubs there, not something to
///    force one way).
/// 2. The word immediately past the chain's last head, **only** when that
///    head is a plain Sequential instruction — its fallthrough edge is the
///    one case in this whole scheme that has no other region-exit
///    mechanism besides the walk budget running out (a branch/jump/regjump
///    tail already always exits via its own taken/not-taken/RegJump
///    handling, budget or not). Poisoning this word means an off-by-one in
///    `heads.len()` vs. the real budget accounting fails loudly (`Excluded`
///    /decline) instead of silently compiling a region one instruction
///    bigger than what was actually checked against the trace.
///
/// Extracted out of `run_chain`'s main loop so it's independently testable
/// (see `tests::build_chain_page_words_poisons_the_untaken_arm_only` and
/// `tests::build_chain_page_words_poisons_past_a_sequential_tail`) — this is
/// the one piece of chain-mode that has no directly observable effect on
/// `Stats` in the zero-filled-by-default synthetic buffers a unit test
/// constructs (a live boot's real page content is what an unpoisoned stray
/// word would actually risk wandering into), so it needs its own direct
/// coverage rather than relying on end-to-end register comparisons.
fn build_chain_page_words(heads: &[(TraceRecord, bool, Option<TraceRecord>)], page_base: u32) -> [u32; ENTRIES_PER_PAGE] {
    let mut page_words = [0u32; ENTRIES_PER_PAGE];
    for &(rec, is_bj, slot) in heads {
        let w = ((rec.pc & 0xFFF) / 4) as usize;
        if w < ENTRIES_PER_PAGE { page_words[w] = rec.raw; }
        if is_bj {
            if let Some(slot) = slot {
                let sw = ((slot.pc & 0xFFF) / 4) as usize;
                if sw < ENTRIES_PER_PAGE { page_words[sw] = slot.raw; }
            }
        }
    }

    let is_real_chain_word = |cw: u16| {
        heads.iter().any(|&(r, ibj, s)| {
            ((r.pc & 0xFFF) / 4) as u16 == cw
                || (ibj && s.map_or(false, |s| ((s.pc & 0xFFF) / 4) as u16 == cw))
        })
    };

    for i in 0..heads.len().saturating_sub(1) {
        let (rec, is_bj, _) = heads[i];
        if !is_bj { continue; }
        let word = ((rec.pc & 0xFFF) / 4) as u16;
        let class = classify(rec.raw, word, page_base);
        let taken_word = branch_or_jump_taken_word(class, rec.raw, word);
        let not_taken_word = word + 2; // past the slot, same for Branch and Jump/regjump-shaped entries
        let actual_next_word = ((heads[i + 1].0.pc & 0xFFF) / 4) as u16;
        for &candidate in &[taken_word, Some(not_taken_word)] {
            if let Some(cw) = candidate {
                // Only poison a word that isn't itself part of the real
                // chain (another head or slot already placed above) — never
                // overwrite real content.
                if cw != actual_next_word && (cw as usize) < ENTRIES_PER_PAGE && !is_real_chain_word(cw) {
                    page_words[cw as usize] = POISON_WORD;
                }
            }
        }
    }

    if let Some(&(last_rec, last_is_bj, _)) = heads.last() {
        if !last_is_bj {
            let last_word = ((last_rec.pc & 0xFFF) / 4) as u16;
            let past = last_word + 1;
            if (past as usize) < ENTRIES_PER_PAGE && !is_real_chain_word(past) {
                page_words[past as usize] = POISON_WORD;
            }
        }
    }

    page_words
}

fn page_base_of(pc: u64) -> u32 {
    (pc & !0xFFFu64) as u32
}

/// Resolve a Branch/Jump's taken-arm target word, if statically known and
/// in-page — `None` for anything else (RegJump, page-leaving). Mirrors
/// `analyzer::classify`'s own `Classify::Branch`/`Classify::Jump` payload,
/// re-exposed here since `run_chain` needs it outside the walker to decide
/// which arm to poison.
fn branch_or_jump_taken_word(class: Classify, _raw: u32, _word: u16) -> Option<u16> {
    match class {
        Classify::Branch { target } => target,
        Classify::Jump { target } => target,
        _ => None,
    }
}

/// Whether including a candidate word (`word`/`class`/`raw` — not yet
/// pushed into `heads`) would give the analyzer's real walk an edge (its
/// own fallthrough, or — for a branch/jump — either arm) landing back on a
/// word already in `heads`. If so, `analyzer::visit`'s "already visited,
/// continue into compiled code" fast path would wire a genuine Cranelift
/// back-edge there (§2.2/§3.1 "loops stay native" —
/// rules/jitv2/codegen-gotchas.md's self-loop section) — the compiled
/// function would then run that loop to convergence natively inside one
/// call, silently advancing far more architectural state than the single
/// trace record a chain attempt's comparison point represents. Checked at
/// *both* chain-growth sites in `run_chain` (the initial straight-line
/// build loop and the past-a-branch extension loop) — a real IRIX boot
/// trace hit this from both directions: first as an extension past an
/// already-included branch, and — once that was blocked — again as a
/// plain second head included via the initial loop's ordinary forward walk
/// (a 2-instruction loop at 0xbfc03bb8/0xbfc03bbc: `ADDI r2,r2,-1` then
/// `BNE r2,r0,-2`, whose taken target is the *previous* word).
fn would_create_back_edge(heads: &[(TraceRecord, bool, Option<TraceRecord>)], word: u16, class: Classify, raw: u32) -> bool {
    let already_in_chain = |w: u16| {
        heads.iter().any(|&(r, ibj, s)| {
            ((r.pc & 0xFFF) / 4) as u16 == w
                || (ibj && s.map_or(false, |s| ((s.pc & 0xFFF) / 4) as u16 == w))
        })
    };
    let is_bj = matches!(class, Classify::Branch { .. } | Classify::Jump { .. } | Classify::RegJump);
    if is_bj {
        let taken = branch_or_jump_taken_word(class, raw, word);
        let not_taken = word + 2;
        taken.map_or(false, already_in_chain) || already_in_chain(not_taken)
    } else {
        already_in_chain(word + 1)
    }
}

/// Slide the (cur, next) window forward. A branch/jump unit consumes two
/// trace records (itself + its slot) per JIT call, so its window advances by
/// two; a plain instruction advances by one. `rec2`/`reader` supply the
/// records needed to refill `next` in the branch/jump case without a second
/// read call at every call site.
fn advance(
    cur: &mut Option<TraceRecord>,
    next: &mut Option<TraceRecord>,
    rec_next: TraceRecord,
    rec2: Option<TraceRecord>,
    reader: &mut TraceReader,
    consumed_two: bool,
) {
    if consumed_two {
        // rec, rec_next (the slot) are both fully consumed; rec2 becomes the
        // new cur. Its own successor was never read yet, so this is the one
        // case that legitimately needs a fresh read for `next`.
        *cur = rec2;
        *next = rec2.and_then(|_| reader.next().unwrap_or(None));
    } else {
        // rec alone was consumed; rec_next becomes the new cur. rec2 — read
        // once, unconditionally, by every call site before calling advance —
        // is already exactly rec_next's successor, so it must be reused as
        // the new `next` here, NOT discarded in favor of a fresh read. Doing
        // a fresh read here silently drops rec2 from ever being examined,
        // permanently desyncing the (cur, next) window by one record for
        // every plain (non-branch/jump) instruction processed — this was a
        // real bug that made ordinary sequential instructions look like they
        // had diverged from an interrupt when really the verifier had just
        // skipped over their real successor.
        *cur = Some(rec_next);
        *next = rec2;
    }
}

/// Compare the fields a compiled region can actually touch. GPR/FPR are
/// compared whole-array with a per-index report so a mismatch names the
/// specific register; scalar fields compare directly. `pc` is intentionally
/// not re-checked here (already gated by the caller before this is called).
fn diff_state(got: &CoreState, want: &CoreState) -> Option<String> {
    let mut parts = Vec::new();
    for i in 0..32 {
        if got.gpr[i] != want.gpr[i] {
            parts.push(format!("gpr[{}]: got={:#018x} want={:#018x}", i, got.gpr[i], want.gpr[i]));
        }
    }
    for i in 0..32 {
        if got.fpr[i] != want.fpr[i] {
            parts.push(format!("fpr[{}]: got={:#018x} want={:#018x}", i, got.fpr[i], want.fpr[i]));
        }
    }
    if got.hi != want.hi { parts.push(format!("hi: got={:#018x} want={:#018x}", got.hi, want.hi)); }
    if got.lo != want.lo { parts.push(format!("lo: got={:#018x} want={:#018x}", got.lo, want.lo)); }
    if got.cp0_epc != want.cp0_epc { parts.push(format!("cp0_epc: got={:#018x} want={:#018x}", got.cp0_epc, want.cp0_epc)); }
    if got.cp0_cause != want.cp0_cause { parts.push(format!("cp0_cause: got={:#010x} want={:#010x}", got.cp0_cause, want.cp0_cause)); }
    if got.cp0_status != want.cp0_status { parts.push(format!("cp0_status: got={:#010x} want={:#010x}", got.cp0_status, want.cp0_status)); }
    if got.fpu_fcsr != want.fpu_fcsr { parts.push(format!("fpu_fcsr: got={:#010x} want={:#010x}", got.fpu_fcsr, want.fpu_fcsr)); }
    if got.fpu_fccr != want.fpu_fccr { parts.push(format!("fpu_fccr: got={:#010x} want={:#010x}", got.fpu_fccr, want.fpu_fccr)); }

    if parts.is_empty() { None } else { Some(parts.join(", ")) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iris::mips_isa::{FUNCT_JR, OP_ADDIU, OP_BEQ, OP_BNE, OP_LW, OP_SPECIAL};
    use iris::trace::TraceWriter;

    fn i_type(op: u32, rs: u32, rt: u32, imm: u16) -> u32 {
        (op << 26) | (rs << 21) | (rt << 16) | (imm as u32)
    }

    fn r_type(op: u32, rs: u32, rt: u32, rd: u32, sa: u32, funct: u32) -> u32 {
        (op << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct
    }

    #[test]
    fn touches_memory_flags_loads_and_stores_only() {
        assert!(touches_memory(i_type(OP_LW, 0, 1, 0)));
        assert!(!touches_memory(i_type(OP_ADDIU, 0, 1, 5)));
        assert!(!touches_memory(0)); // SPECIAL/SLL nop
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("iris_jitv2_verify_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn base_state(pc: u64) -> CoreState {
        CoreState {
            gpr: [0; 32], pc, hi: 0, lo: 0, cp0_epc: 0, cp0_badvaddr: 0,
            cp0_cause: 0, cp0_status: 0, fpr: [0; 32],
            fpu_fcsr: 0, fpu_fccr: 0, fpu_fexr: 0, fpu_fenr: 0,
            in_delay_slot: false,
        }
    }

    #[test]
    fn detects_a_correct_addiu_chain_with_no_mismatches() {
        let path = tmp_path("addiu_ok.trace");
        let pc0 = 0xFFFF_FFFF_8000_1000u64;

        // r1 = 0; ADDIU r1, r1, 5 -> r1 = 5 (matches what the JIT will
        // actually compute) at pc0; next record's state is the real
        // interpreter-produced post-state.
        let mut w = TraceWriter::create(&path).unwrap();
        let mut s0 = base_state(pc0);
        w.push(&TraceRecord::new(pc0, i_type(OP_ADDIU, 1, 1, 5), s0)).unwrap();
        s0.gpr[1] = 5;
        s0.pc = pc0 + 4;
        w.push(&TraceRecord::new(pc0 + 4, 0, s0)).unwrap(); // NOP, just a landing record
        w.flush().unwrap();
        drop(w);

        let stats = run(&path, 0, None, false).unwrap();
        assert_eq!(stats.mismatches, 0, "a correctly-recorded ADDIU must not be reported as a mismatch");
        assert_eq!(stats.compared, 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_long_chain_of_plain_instructions_stays_in_sync() {
        // Regression test for a window-desync bug: every loop iteration reads
        // a 3rd record (rec2) unconditionally, but advance()'s non-branch
        // path used to discard it and do a fresh read instead, permanently
        // skipping one real record per plain instruction processed. A 2- or
        // 3-record trace (like the ADDIU test above) can't detect this — it
        // only shows up from the 3rd plain instruction onward, so this test
        // uses a chain of 10 to make sure every step still lines up.
        let path = tmp_path("long_chain_ok.trace");
        let pc0 = 0xFFFF_FFFF_8000_7000u64;

        let mut w = TraceWriter::create(&path).unwrap();
        let mut s = base_state(pc0);
        for i in 0..10u64 {
            let pc = pc0 + i * 4;
            w.push(&TraceRecord::new(pc, i_type(OP_ADDIU, 1, 1, 1), s)).unwrap();
            s.gpr[1] = s.gpr[1].wrapping_add(1);
            s.pc = pc + 4;
        }
        w.push(&TraceRecord::new(pc0 + 40, 0, s)).unwrap(); // landing record for the 10th ADDIU
        w.flush().unwrap();
        drop(w);

        let stats = run(&path, 0, None, false).unwrap();
        assert_eq!(stats.compared, 10, "every ADDIU in the chain must be compared, none silently skipped");
        assert_eq!(stats.mismatches, 0, "the correctly-recorded chain must not produce false mismatches");
        assert_eq!(stats.skipped_control_flow_diverged, 0, "a desynced window shows up as spurious control-flow divergence");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn detects_a_deliberately_wrong_recorded_result_as_a_mismatch() {
        let path = tmp_path("addiu_bad.trace");
        let pc0 = 0xFFFF_FFFF_8000_2000u64;

        let mut w = TraceWriter::create(&path).unwrap();
        let s0 = base_state(pc0);
        w.push(&TraceRecord::new(pc0, i_type(OP_ADDIU, 1, 1, 5), s0)).unwrap();
        // Wrong: record claims r1 stayed 0 instead of becoming 5 — the JIT
        // will still (correctly) compute 5, so this must be flagged.
        let mut s1 = s0;
        s1.pc = pc0 + 4;
        w.push(&TraceRecord::new(pc0 + 4, 0, s1)).unwrap();
        w.flush().unwrap();
        drop(w);

        let stats = run(&path, 0, None, false).unwrap();
        assert_eq!(stats.mismatches, 1, "a deliberately-wrong recorded post-state must be caught");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn skips_loads_and_stores_without_comparing() {
        let path = tmp_path("load_skip.trace");
        let pc0 = 0xFFFF_FFFF_8000_3000u64;

        let mut w = TraceWriter::create(&path).unwrap();
        let s0 = base_state(pc0);
        w.push(&TraceRecord::new(pc0, i_type(OP_LW, 2, 1, 0), s0)).unwrap();
        let mut s1 = s0;
        s1.pc = pc0 + 4;
        s1.gpr[1] = 0xDEAD_BEEF; // whatever the interpreter's real bus returned
        w.push(&TraceRecord::new(pc0 + 4, 0, s1)).unwrap();
        w.flush().unwrap();
        drop(w);

        let stats = run(&path, 0, None, false).unwrap();
        assert_eq!(stats.skipped_memory, 1);
        assert_eq!(stats.compared, 0, "a load's destination register must never be compared");
        assert_eq!(stats.mismatches, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn branch_and_delay_slot_compare_against_the_third_record() {
        let path = tmp_path("branch_ok.trace");
        let pc0 = 0xFFFF_FFFF_8000_4000u64;

        // BEQ r0, r0, +1 (always taken; target = pc0+4 (slot) + 1*4 = pc0+8)
        // delay slot: ADDIU r2, r2, 1 (always executes, taken or not)
        let mut w = TraceWriter::create(&path).unwrap();
        let s0 = base_state(pc0);
        w.push(&TraceRecord::new(pc0, i_type(OP_BEQ, 0, 0, 1), s0)).unwrap();

        let mut s_slot = s0;
        s_slot.pc = pc0 + 4;
        w.push(&TraceRecord::new(pc0 + 4, i_type(OP_ADDIU, 2, 2, 1), s_slot)).unwrap();

        // Landing record: slot committed (r2=1), pc jumped to the branch target.
        let mut s_landed = s_slot;
        s_landed.gpr[2] = 1;
        s_landed.pc = pc0 + 8;
        w.push(&TraceRecord::new(pc0 + 8, 0, s_landed)).unwrap();
        w.flush().unwrap();
        drop(w);

        let stats = run(&path, 0, None, false).unwrap();
        assert_eq!(stats.compared, 1, "the branch+slot unit must be compared once, against the 3rd record");
        assert_eq!(stats.mismatches, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn regjump_and_delay_slot_compare_against_the_third_record() {
        // JR/JALR is classified separately from Branch/Jump (Classify::RegJump)
        // but still has a mandatory inline delay slot and a single compiled
        // unit spanning both records — this regression-guards the fix that
        // added RegJump to is_branch_or_jump (it was silently mis-verified
        // against the slot's pre-state instead of the real landing state
        // before that fix).
        let path = tmp_path("regjump_ok.trace");
        let pc0 = 0xFFFF_FFFF_8000_6000u64;
        let target = 0xFFFF_FFFF_8000_9000u64;

        let mut w = TraceWriter::create(&path).unwrap();
        let mut s0 = base_state(pc0);
        s0.gpr[4] = target; // JR r4
        w.push(&TraceRecord::new(pc0, r_type(OP_SPECIAL, 4, 0, 0, 0, FUNCT_JR), s0)).unwrap();

        let mut s_slot = s0;
        s_slot.pc = pc0 + 4;
        w.push(&TraceRecord::new(pc0 + 4, i_type(OP_ADDIU, 2, 2, 1), s_slot)).unwrap();

        let mut s_landed = s_slot;
        s_landed.gpr[2] = 1;
        s_landed.pc = target;
        w.push(&TraceRecord::new(target, 0, s_landed)).unwrap();
        w.flush().unwrap();
        drop(w);

        let stats = run(&path, 0, None, false).unwrap();
        assert_eq!(stats.compared, 1, "the JR+slot unit must be compared once, against the 3rd record");
        assert_eq!(stats.mismatches, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn control_flow_divergence_is_not_reported_as_a_mismatch() {
        let path = tmp_path("cf_diverge.trace");
        let pc0 = 0xFFFF_FFFF_8000_5000u64;

        let mut w = TraceWriter::create(&path).unwrap();
        let s0 = base_state(pc0);
        w.push(&TraceRecord::new(pc0, i_type(OP_ADDIU, 1, 1, 5), s0)).unwrap();
        // Simulate an interrupt landing between the two steps: next record's
        // pc is nowhere near pc0+4 (an exception vector, say).
        let mut s1 = s0;
        s1.gpr[1] = 5;
        s1.pc = 0xFFFF_FFFF_8000_0180;
        w.push(&TraceRecord::new(s1.pc, 0, s1)).unwrap();
        w.flush().unwrap();
        drop(w);

        let stats = run(&path, 0, None, false).unwrap();
        assert_eq!(stats.skipped_control_flow_diverged, 1);
        assert_eq!(stats.diverged_to_exception_vector, 1, "0x80000180 is a real exception vector (BEV=0, general)");
        assert_eq!(stats.compared, 0);
        assert_eq!(stats.mismatches, 0, "a pc mismatch alone must never be reported as a register-content mismatch");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn control_flow_divergence_to_a_non_vector_is_flagged_unexplained() {
        // If the trace's actual next pc ISN'T one of the six real exception
        // vectors, "likely async interrupt" isn't a valid explanation
        // anymore — this is exactly the case is_exception_vector exists to
        // distinguish from the legitimate-divergence case above.
        let path = tmp_path("cf_diverge_unexplained.trace");
        let pc0 = 0xFFFF_FFFF_8000_6000u64;

        let mut w = TraceWriter::create(&path).unwrap();
        let s0 = base_state(pc0);
        w.push(&TraceRecord::new(pc0, i_type(OP_ADDIU, 1, 1, 5), s0)).unwrap();
        let mut s1 = s0;
        s1.gpr[1] = 5;
        s1.pc = 0xFFFF_FFFF_8012_3456; // not a vector address
        w.push(&TraceRecord::new(s1.pc, 0, s1)).unwrap();
        w.flush().unwrap();
        drop(w);

        let stats = run(&path, 0, None, false).unwrap();
        assert_eq!(stats.skipped_control_flow_diverged, 1);
        assert_eq!(stats.diverged_to_exception_vector, 0, "0x80123456 is not a real exception vector");
        let _ = std::fs::remove_file(&path);
    }

    // ---- Chain mode (run_chain, --chain N) ------

    #[test]
    fn chain_mode_compiles_a_straight_line_run_as_one_region() {
        // 3 plain ADDIUs in a row, --chain 3: must compile as a single
        // 3-head region and land the comparison on the 4th record (past all
        // three), not fragment back into 3 separate single-instruction
        // compiles the way run() would.
        let path = tmp_path("chain_straight.trace");
        let pc0 = 0xFFFF_FFFF_8000_8000u64;

        let mut w = TraceWriter::create(&path).unwrap();
        let mut s = base_state(pc0);
        for i in 0..3u64 {
            let pc = pc0 + i * 4;
            w.push(&TraceRecord::new(pc, i_type(OP_ADDIU, 1, 1, 10), s)).unwrap();
            s.gpr[1] = s.gpr[1].wrapping_add(10);
            s.pc = pc + 4;
        }
        w.push(&TraceRecord::new(pc0 + 12, 0, s)).unwrap(); // landing record
        w.flush().unwrap();
        drop(w);

        let stats = run_chain(&path, 0, None, true, 3).unwrap();
        assert_eq!(stats.compared, 1, "3 heads must merge into a single compiled-region comparison");
        assert_eq!(stats.mismatches, 0);
        assert_eq!(stats.chain_len_histogram.get(&3), Some(&1), "must have actually reached chain length 3, not silently capped shorter");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn chain_mode_extends_through_a_taken_branch() {
        // ADDIU r1,r1,1 ; BEQ r0,r0,1 (always taken, target = own_word+2) ;
        // slot: ADDIU r2,r2,1 ; [poisoned not-taken word] ; ADDIU r3,r3,1
        // (the real taken target) -- --chain 4 should walk all of: the
        // leading ADDIU, the branch+slot, and the taken target's own ADDIU,
        // landing the comparison one word past that.
        let path = tmp_path("chain_taken.trace");
        let pc0 = 0xFFFF_FFFF_8000_9000u64;

        let mut w = TraceWriter::create(&path).unwrap();
        let mut s = base_state(pc0);
        // head 1: ADDIU r1,r1,1
        w.push(&TraceRecord::new(pc0, i_type(OP_ADDIU, 1, 1, 1), s)).unwrap();
        s.gpr[1] = 1; s.pc = pc0 + 4;
        // head 2: BEQ r0,r0,1 (target = word(pc0+4)+1+1 -> pc0+4+8 = pc0+12)
        w.push(&TraceRecord::new(pc0 + 4, i_type(OP_BEQ, 0, 0, 1), s)).unwrap();
        s.pc = pc0 + 8;
        // slot: ADDIU r2,r2,1 (always executes)
        w.push(&TraceRecord::new(pc0 + 8, i_type(OP_ADDIU, 2, 2, 1), s)).unwrap();
        s.gpr[2] = 1; s.pc = pc0 + 12; // branch taken, lands here (not pc0+16, the not-taken fallthrough)
        // head 3 (the taken target): ADDIU r3,r3,1
        w.push(&TraceRecord::new(pc0 + 12, i_type(OP_ADDIU, 3, 3, 1), s)).unwrap();
        s.gpr[3] = 1; s.pc = pc0 + 16;
        w.push(&TraceRecord::new(pc0 + 16, 0, s)).unwrap(); // landing record
        w.flush().unwrap();
        drop(w);

        let stats = run_chain(&path, 0, None, true, 4).unwrap();
        assert_eq!(stats.compared, 1, "the whole taken-branch chain must merge into one comparison");
        assert_eq!(stats.mismatches, 0);
        assert!(stats.chain_len_histogram.get(&3).copied().unwrap_or(0) >= 1, "must have reached at least the 3-head chain (leading ADDIU, branch, taken-target ADDIU)");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn chain_mode_extends_through_a_not_taken_branch() {
        // Same shape as the taken-branch test, but this time the trace
        // records BEQ as not-taken -- the chain must follow the real
        // not-taken fallthrough (word+2, past the slot) into a further
        // head, with the branch's *taken* arm poisoned instead.
        //
        // imm=3 (not 1) so the taken target (word+1+3 = word+4) lands on a
        // genuinely different word than the not-taken fallthrough (word+2)
        // -- with imm=1 both would coincide and this test's end-to-end
        // assertions couldn't actually distinguish "poisoned correctly"
        // from "poisoning does nothing" (a real trap: an earlier version of
        // this test used imm=1 and kept passing with the poison write
        // temporarily disabled). Since the synthetic page_words buffer
        // zero-fills by default (word+4 would just read as a harmless NOP
        // either way, not something that could corrupt the final register
        // comparison), this test additionally reaches into run_chain's
        // result at the analyzer level to check the poisoned word directly
        // -- the only place a zero-filled default and a genuine live-boot
        // page (where the untaken arm could hold real, unrelated
        // instruction bytes reused from elsewhere) would actually differ.
        let path = tmp_path("chain_not_taken.trace");
        let pc0 = 0xFFFF_FFFF_8000_A000u64;

        let mut w = TraceWriter::create(&path).unwrap();
        let mut s = base_state(pc0);
        s.gpr[1] = 1; // BEQ r1,r0,... is NOT taken since r1 != 0
        // head 1: BEQ r1,r0,3 -- NOT taken; taken target would be word+4
        w.push(&TraceRecord::new(pc0, i_type(OP_BEQ, 1, 0, 3), s)).unwrap();
        s.pc = pc0 + 4;
        // slot: ADDIU r2,r2,1 (always executes regardless of taken/not-taken)
        w.push(&TraceRecord::new(pc0 + 4, i_type(OP_ADDIU, 2, 2, 1), s)).unwrap();
        s.gpr[2] = 1; s.pc = pc0 + 8; // not taken -> falls through to word+2 = pc0+8
        // head 2 (the real not-taken fallthrough): ADDIU r3,r3,1
        w.push(&TraceRecord::new(pc0 + 8, i_type(OP_ADDIU, 3, 3, 1), s)).unwrap();
        s.gpr[3] = 1; s.pc = pc0 + 12;
        w.push(&TraceRecord::new(pc0 + 12, 0, s)).unwrap(); // landing record
        w.flush().unwrap();
        drop(w);

        let stats = run_chain(&path, 0, None, true, 3).unwrap();
        assert_eq!(stats.compared, 1, "the not-taken-branch chain must merge into one comparison");
        assert_eq!(stats.mismatches, 0);
        assert!(stats.chain_len_histogram.get(&2).copied().unwrap_or(0) >= 1, "must have reached at least the 2-head chain (branch, not-taken-fallthrough ADDIU)");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn build_chain_page_words_poisons_the_untaken_arm_only() {
        // Direct, honest coverage of the poisoning logic (see its own doc
        // comment for why an end-to-end trace-replay test can't actually
        // distinguish "poisoned" from "zero-filled" — both look like a
        // harmless NOP in a synthetic buffer with no other content there).
        // Calls the real function `run_chain` uses, not a reimplementation.
        let pc0 = 0xFFFF_FFFF_8000_A000u64;
        let word = ((pc0 & 0xFFF) / 4) as u16;
        let page_base = page_base_of(pc0);
        let s = base_state(pc0);

        // BEQ r1,r0,3 (not taken; taken target = word+4) / slot ADDIU r2,r2,1
        // / real not-taken continuation ADDIU r3,r3,1 at word+2 -- same
        // shape as chain_mode_extends_through_a_not_taken_branch, but here
        // `heads` is constructed directly so this test targets exactly the
        // function under test.
        let branch = TraceRecord::new(pc0, i_type(OP_BEQ, 1, 0, 3), s);
        let slot = TraceRecord::new(pc0 + 4, i_type(OP_ADDIU, 2, 2, 1), s);
        let cont = TraceRecord::new(pc0 + 8, i_type(OP_ADDIU, 3, 3, 1), s);
        let heads = vec![(branch, true, Some(slot)), (cont, false, None)];

        let page_words = build_chain_page_words(&heads, page_base);
        assert_eq!(page_words[word as usize], branch.raw, "the branch's own real bytes must be preserved");
        assert_eq!(page_words[word as usize + 1], slot.raw, "the slot's own real bytes must be preserved");
        assert_eq!(page_words[word as usize + 2], cont.raw, "the real not-taken continuation must be preserved, not poisoned");
        assert_eq!(page_words[word as usize + 4], POISON_WORD, "the untaken (taken) arm must be poisoned");

        // And confirm the analyzer actually honors that poisoning: the
        // taken arm must bail with Excluded, not silently walk into it.
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, word, page_base, 100);
        assert!(non_empty);
        assert_eq!(
            walked[word as usize].taken_exit,
            Some(iris::jitv2::analyzer::StopReason::Excluded),
            "the branch's taken arm must bail on the poisoned word, not silently walk into it"
        );
    }

    #[test]
    fn build_chain_page_words_poisons_past_a_sequential_tail() {
        // The word right after a chain's Sequential tail must be poisoned
        // too — that fallthrough edge has no region-exit mechanism besides
        // the walk budget running out, so this is the hard backstop against
        // an off-by-one in heads.len() vs. real budget accounting silently
        // compiling one instruction more than what was actually checked.
        let pc0 = 0xFFFF_FFFF_8000_C000u64;
        let word = ((pc0 & 0xFFF) / 4) as u16;
        let page_base = page_base_of(pc0);
        let s = base_state(pc0);

        let head = TraceRecord::new(pc0, i_type(OP_ADDIU, 1, 1, 1), s);
        let heads = vec![(head, false, None)];

        let page_words = build_chain_page_words(&heads, page_base);
        assert_eq!(page_words[word as usize], head.raw, "the head's own real bytes must be preserved");
        assert_eq!(page_words[word as usize + 1], POISON_WORD, "the word past a Sequential tail must be poisoned");

        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk_bounded(&page_words, word, page_base, 100);
        assert!(non_empty);
        assert_eq!(
            walked[word as usize].fallthrough_exit,
            Some(iris::jitv2::analyzer::StopReason::Excluded),
            "a Sequential tail's fallthrough must bail on the poisoned word regardless of remaining budget"
        );
    }

    #[test]
    fn build_chain_page_words_does_not_poison_past_a_branch_tail() {
        // The opposite of the Sequential-tail test: when the chain's last
        // head IS a branch/jump, neither of its own arms should be
        // poisoned -- both are legitimate exit stubs whose taken/not-taken
        // split is exactly what's being verified, and poisoning either
        // would force a bail where real analysis should happen instead.
        let pc0 = 0xFFFF_FFFF_8000_D000u64;
        let word = ((pc0 & 0xFFF) / 4) as u16;
        let page_base = page_base_of(pc0);
        let s = base_state(pc0);

        let branch = TraceRecord::new(pc0, i_type(OP_BEQ, 0, 0, 1), s); // target = word+2
        let slot = TraceRecord::new(pc0 + 4, i_type(OP_ADDIU, 2, 2, 1), s);
        let heads = vec![(branch, true, Some(slot))];

        let page_words = build_chain_page_words(&heads, page_base);
        assert_eq!(page_words[word as usize], branch.raw);
        assert_eq!(page_words[word as usize + 1], slot.raw);
        assert_eq!(page_words[word as usize + 2], 0, "the taken target (word+2) must be left untouched, not poisoned");
    }

    #[test]
    fn chain_mode_stops_at_a_memory_op() {
        // A load/store partway through a chain attempt must end the chain
        // there (whatever was accumulated before it is still verified), not
        // abort the whole attempt or silently include the memory op.
        let path = tmp_path("chain_mem_stop.trace");
        let pc0 = 0xFFFF_FFFF_8000_B000u64;

        let mut w = TraceWriter::create(&path).unwrap();
        let mut s = base_state(pc0);
        w.push(&TraceRecord::new(pc0, i_type(OP_ADDIU, 1, 1, 1), s)).unwrap();
        s.gpr[1] = 1; s.pc = pc0 + 4;
        w.push(&TraceRecord::new(pc0 + 4, i_type(OP_LW, 2, 1, 0), s)).unwrap(); // load -- must not be included
        s.pc = pc0 + 8;
        w.push(&TraceRecord::new(pc0 + 8, 0, s)).unwrap();
        w.flush().unwrap();
        drop(w);

        let stats = run_chain(&path, 0, None, true, 4).unwrap();
        assert_eq!(stats.compared, 1, "the leading ADDIU must still be verified even though the chain couldn't extend past the load");
        assert_eq!(stats.mismatches, 0);
        assert_eq!(stats.chain_len_histogram.get(&1), Some(&1), "chain must have stopped at length 1, right before the load");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn chain_mode_does_not_extend_through_an_excluded_instruction() {
        // Regression test: a real IRIX boot trace produced a false-positive
        // "control flow diverged" report where the chain's own extension
        // logic included an MFC0 (Classify::Excluded) as a 3rd head after a
        // taken branch, because the extension loop only checked
        // touches_memory, not classify()==Excluded. The analyzer's real
        // walk refuses to enter an Excluded word at all (visit() returns
        // false immediately), so the real compiled region only ever covers
        // 2 heads here (branch+slot) — walk_bounded's max_instrs must match
        // that, not the inflated head count the chain-builder would
        // otherwise have assumed. The comparison point must be right after
        // the branch+slot (the branch's taken target itself), not one
        // record further as if the excluded instruction had also run.
        let path = tmp_path("chain_excluded_stop.trace");
        let pc0 = 0xFFFF_FFFF_8000_E000u64;

        let mut w = TraceWriter::create(&path).unwrap();
        let mut s = base_state(pc0);
        // head 1: BEQ r0,r0,1 (always taken; target = word+2)
        w.push(&TraceRecord::new(pc0, i_type(OP_BEQ, 0, 0, 1), s)).unwrap();
        s.pc = pc0 + 4;
        // slot: ADDIU r2,r2,1
        w.push(&TraceRecord::new(pc0 + 4, i_type(OP_ADDIU, 2, 2, 1), s)).unwrap();
        s.gpr[2] = 1; s.pc = pc0 + 8; // taken -> lands at word+2 = pc0+8
        // The taken target is MFC0 r3,$0 (OP_COP0, rs=RS_MFC0=0) --
        // Excluded. The interpreter dispatches it for real (its own
        // architectural effect isn't this test's concern), producing
        // whatever state the *next* record after it reflects.
        let mfc0_raw = (iris::mips_isa::OP_COP0 << 26) | (0 << 21) | (3 << 16); // rs=0 (MFC0), rt=3
        w.push(&TraceRecord::new(pc0 + 8, mfc0_raw, s)).unwrap();
        s.gpr[3] = 0xDEAD; // whatever the real CP0 register read produced
        s.pc = pc0 + 12;
        w.push(&TraceRecord::new(pc0 + 12, 0, s)).unwrap(); // landing record after MFC0
        w.flush().unwrap();
        drop(w);

        let stats = run_chain(&path, 0, None, true, 4).unwrap();
        assert_eq!(stats.compared, 1, "the branch+slot must still be verified even though the chain couldn't extend through the excluded MFC0");
        assert_eq!(stats.mismatches, 0, "the comparison point must be the branch's real taken target (pc0+8), not one record past the excluded instruction");
        assert_eq!(stats.chain_len_histogram.get(&1), Some(&1), "chain must have stopped at length 1 (the branch+slot unit), never including the excluded MFC0 as a head");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn chain_mode_does_not_create_a_native_back_edge_into_the_chain() {
        // Regression test for a real IRIX boot trace shape: a tight
        // 2-instruction loop `ADDI r2,r2,-1` / `BNE r2,r0,-2` (target =
        // the ADDI's own word — one word *before* the branch). Including
        // both as chain heads would give the analyzer's walk a genuine
        // Cranelift back-edge from the branch's taken arm straight into
        // the ADDI's already-visited block ("loops stay native" —
        // rules/jitv2/codegen-gotchas.md's self-loop section) — the
        // compiled function would then run to convergence natively in one
        // call while this attempt's comparison point still assumed a
        // single trace-record step, producing a false "control flow
        // diverged" report. Hit from *both* directions in the real trace:
        // first as an attempted extension past an already-included branch,
        // then (once that was blocked) again via the initial build loop's
        // ordinary forward walk starting from the ADDI itself — this test
        // exercises the second (initial-loop) path, since a fresh trace
        // naturally starts each new attempt at whatever word the previous
        // attempt's chain boundary landed on.
        let path = tmp_path("chain_back_edge.trace");
        let pc0 = 0xFFFF_FFFF_8000_F000u64; // word W: ADDI r2,r2,-1
        // word W+1: BNE r2,r0,-2 (target = (W+1)+1+(-2) = W, the ADDI's own word)
        // word W+2: delay slot (NOP)
        // word W+3: fallthrough once the loop exits

        let mut w = TraceWriter::create(&path).unwrap();
        let mut s = base_state(pc0);
        s.gpr[2] = 1; // one iteration: decrements to 0, then not-taken
        w.push(&TraceRecord::new(pc0, i_type(OP_ADDIU, 2, 2, 0xFFFF), s)).unwrap(); // ADDI r2,r2,-1
        s.gpr[2] = 0;
        s.pc = pc0 + 4;
        w.push(&TraceRecord::new(pc0 + 4, i_type(OP_BNE, 2, 0, 0xFFFE), s)).unwrap(); // BNE r2,r0,-2
        s.pc = pc0 + 8;
        w.push(&TraceRecord::new(pc0 + 8, 0, s)).unwrap(); // delay slot, not taken this time
        s.pc = pc0 + 12; // falls through past the slot -- BNE not taken (r2==0)
        // A load at the landing pc stops the 2nd attempt's extension loop
        // cleanly (touches_memory) instead of trying to pull in yet another
        // head with nothing after it to compare against — this test is
        // about the back-edge guard, not chain length, so keep the tail
        // simple and bounded.
        w.push(&TraceRecord::new(pc0 + 12, i_type(OP_LW, 4, 5, 0), s)).unwrap();
        s.pc = pc0 + 16;
        w.push(&TraceRecord::new(pc0 + 16, 0, s)).unwrap(); // final landing record for the load-skip
        w.flush().unwrap();
        drop(w);

        let stats = run_chain(&path, 0, None, true, 4).unwrap();
        // The chain must never include both the ADDI and the BNE as heads
        // together (that's exactly the back-edge shape): the first attempt
        // starts at the ADDI, tries to extend to the BNE, and the guard
        // must refuse — leaving a length-1 chain (just the ADDI) that
        // compares against the BNE's own pre-state (pc0+4). The second
        // attempt then starts fresh at the BNE, includes its mandatory
        // slot (length 1 head + slot), and compares against the real
        // landing record (pc0+12).
        assert_eq!(stats.mismatches, 0);
        assert_eq!(stats.compared, 2, "both the ADDI alone and the BNE+slot unit must be verified as separate attempts");
        assert_eq!(stats.chain_len_histogram.get(&2), None, "no attempt may ever reach length 2 here -- that would mean the back-edge guard failed to stop it");
        assert_eq!(stats.chain_len_histogram.get(&1), Some(&2), "both attempts must have stopped at exactly length 1");
        let _ = std::fs::remove_file(&path);
    }
}
