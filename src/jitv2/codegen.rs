//! JIT v2 codegen: translate a walked region (`jitv2/analyzer.rs`) into
//! Cranelift IR (§3.1 — "one Cranelift IR block per instruction").
//!
//! This is the first pass only: allocate a Cranelift `Block` for every
//! instruction the analyzer marked `visited`, plus one entry block that
//! jumps straight to the entry instruction's block. No instruction
//! semantics are emitted yet — every per-instruction block is empty (would
//! trap if run as-is). Filling in real semantics, wiring the fallthrough/
//! taken edges between blocks, and materializing exit stubs for
//! `fallthrough_exit`/`taken_exit` land in follow-up passes.
//!
//! Standalone `cranelift_codegen::ir::Block` handles aren't stored in
//! `analyzer.rs`'s `CompiledInstr` — that module stays free of the Cranelift
//! dependency, so `block_id` is a plain `Option<u32>` there. `Block`
//! implements `EntityRef`/`From<u32>`, so the conversion here is a bare cast.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{self, AbiParam, Block, InstBuilder, MemFlagsData, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::Module;

use crate::jitv2::analyzer::{instrs_linear, CompiledInstr, WordOffset};
use crate::jitv2::{ENTRIES_PER_PAGE, PAGE_SIZE};
use crate::mips_core::MipsCore;
use crate::mips_exec::{EXEC_COMPLETE, EXEC_IS_EXCEPTION};

/// Cranelift codegen context, reused across compile jobs like `Analyzer`
/// (§2.3's per-page scratch-buffer pattern) — `Context`/`FunctionBuilderContext`
/// own reusable scratch allocations Cranelift would otherwise reallocate
/// per function.
pub struct Codegen {
    module: cranelift_jit::JITModule,
    ctx: Context,
    builder_ctx: FunctionBuilderContext,
    func_id_counter: u32,
    /// Compiled machine code size, in bytes, of the most recent successful
    /// `compile_region` call — see that function's "Read code size before
    /// clearing context" comment for why this can't just be returned
    /// directly. Dev-only (`j2 pcp`/`j2 stats` diagnostic); `handle_request`
    /// reads it immediately after `compile_region` returns `Some` and
    /// forwards it to `page.publish` as `JitEntry::code_size`.
    #[cfg(feature = "developer")]
    last_code_size: u32,
    /// Set right before `compile_region` returns `None` iff that failure
    /// was `ModuleError::Allocation` — the `ArenaMemoryProvider` running out
    /// of its `ARENA_RESERVE_SIZE` reservation (real message observed live:
    /// "pre-allocated jit memory region exhausted") — as opposed to
    /// `ModuleError::Compilation` (a genuine codegen-gap/verifier failure,
    /// which should sticky-denylist the offset, not trigger a flush).
    /// Unconditional, not `developer`-gated: unlike `last_code_size`, this
    /// drives real behavior (`comp::handle_request`'s flush-on-exhaustion),
    /// not just a diagnostic. `CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES`'s
    /// function-count proxy was sized against small, single-instruction
    /// regions; now that `MAX_INSTRS_PER_COMPILE` (`comp.rs`) allows much
    /// larger straight-line functions, a handful of unusually large
    /// compiles can exhaust the byte-size-bounded arena before the
    /// function-*count* threshold ever trips — this is the belt-and-suspenders
    /// catch for exactly that gap, same reasoning as
    /// `CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES`'s own doc comment already
    /// anticipated ("handled too... as a belt-and-suspenders").
    last_compile_ran_out_of_memory: bool,
    /// Set when the most recent `None` return from `compile_region` came
    /// from `self.module.define_function`'s `ModuleError::Compilation` arm
    /// (a real Cranelift verifier rejection) rather than the upfront
    /// no-emitter-for-this-instruction loop — `comp::handle_request` reads
    /// this via `last_decline_was_verifier_error()` immediately after a
    /// `None` result to classify the rejection for `JitStats::reject_reasons`
    /// (`RejectReason::CraneliftVerifierError` vs
    /// `RejectReason::AnalyzerCodegenDisagreement`). Always cleared at the
    /// top of `compile_region` (mirrors `last_compile_ran_out_of_memory`'s
    /// own reset-at-entry discipline) so a stale `true` from a previous call
    /// can never leak into a later, unrelated decline.
    #[cfg(feature = "developer")]
    last_decline_was_verifier_error: bool,
    /// Set when `finalize_batch_nonforced`'s own `module.finalize_definitions()`
    /// call returned `Err` — a real Cranelift/JIT-backend failure at the
    /// relocation-patching step, distinct from (and much rarer than) a
    /// gap-blocked seal (`try_seal_ready` returning empty because an earlier
    /// entry hasn't sealed yet — that's a normal, self-resolving wait, not an
    /// error). Both outcomes make `finalize_batch_nonforced` return an empty
    /// `Vec`, which is indistinguishable to the caller without this flag —
    /// confirmed as a real silent-loss bug: `handle_request_deferred` was
    /// bumping `pending` and waiting for a later sweep to "unblock" an entry
    /// that was never blocked at all, just permanently failed, so it could
    /// never actually succeed no matter how many times a sweep retried it.
    /// Read via `last_finalize_failed()` immediately after a call whose
    /// return was empty; always cleared at the top of
    /// `finalize_batch_nonforced` so a stale `true` can never leak into a
    /// later, unrelated call.
    last_finalize_failed: bool,
    /// Shared with the `PagedArenaMemoryProvider` this `Codegen`'s `module`
    /// was built with — see `PagedArenaState`'s own doc comment for why this
    /// indirection exists. Polled via `provider_crossed_page`/`packing_stats`
    /// after every compile.
    paged_state: std::sync::Arc<crate::jitv2::paged_memory::PagedArenaState>,
    /// A second handle over the exact same shared arena as `module`'s own
    /// (opaque, unreachable) provider — see `new_module`'s doc comment.
    /// Exists purely so `Codegen` can drive non-forced sealing
    /// (`try_seal_pending`, `set_force_seal`) from outside `module` without
    /// needing anything back from it — `JITModule` never exposes its own
    /// provider once constructed.
    seal_handle: crate::jitv2::paged_memory::PagedArenaMemoryProvider,
    /// Byte range (arena-relative offsets) each still-unpublished `FuncId`'s
    /// machine code landed in, recorded by `compile_region_uncommitted`
    /// right after its `define_function` call, read via
    /// `seal_handle.take_last_allocation()` — see that call site's own
    /// comment. `finalize_batch_nonforced` uses this to match
    /// `try_seal_ready`'s sealed-range results back to the `FuncId`s they
    /// belong to. Entries
    /// are removed once a `FuncId` is actually finalized (either forced,
    /// immediately, or non-forced, once its range comes back sealed) —
    /// nothing here should ever reference a `FuncId` from a previous
    /// `reset()`'s arena, so `reset()` clears this too.
    func_ranges: std::collections::HashMap<cranelift_module::FuncId, (usize, usize)>,
}

/// Per-instruction emission context: the three values that are the same for
/// every emit call made while compiling one instruction's body (`builder`,
/// `module`, `core_ptr` — Cranelift's own builder/module handles plus the
/// live `MipsCore*` argument), together with `raw`/`word`, which genuinely
/// describe *this* instruction (its encoding and its compile-time word
/// offset within the page) — not an arbitrary register number or an
/// unrelated branch/slot target word, which stay explicit parameters on the
/// handful of helpers that operate on those instead (`emit_read_gpr(reg)`,
/// `emit_word_addr(word)` for a *target*, etc.).
///
/// `word` is what `emit_check_mem_exc` needs to synthesize the correct
/// exception PC on a fault (see that function's doc comment): a plain
/// sequential instruction's body never writes `core.pc` per-instruction
/// (only exit points do, to keep a straight-line compiled run cheap — that's
/// the entire point of compiling it), so without `word` here, `core.pc`
/// stays stale from wherever the compiled unit last wrote it, and a mid-unit
/// fault reports the wrong EPC.
///
/// `entry_word` is the region's own entry word (`compile_region`'s own
/// parameter of the same name, copied in here unmodified for the whole
/// compile) — needed by `emit_exception_exit` to know when `ctx.word ==
/// entry_word`, the one case where synthesizing `core.pc` from `ctx.word`
/// would be wrong: a physical word can be compiled once as an ordinary
/// entry and *also*, on a later dispatch, be reached because the
/// interpreter's own dispatch loop landed on it as some other branch's
/// delay slot (`core.in_delay_slot` already true, `core.pc` already correct,
/// both set by the interpreter *before* calling into this compiled
/// function — see `MipsExecutor::branch_delay`/`handle_exec_complete`).
/// Every other word in the region never has this ambiguity: a non-entry
/// head instruction's `in_delay_slot` is deterministically false (this
/// compiled unit's own control flow guarantees it), and a slot's is
/// explicitly managed by `emit_slot_semantics` itself. See
/// `emit_exception_exit`'s doc comment for the actual check.
///
/// `exit_block` is the compiled function's one shared exit-to-interpreter
/// block (`BlockSkeleton::exit_block`/`compile_region`'s own local of the
/// same name) — needed by `emit_check_mem_exc` to bail there (via
/// `emit_bail`) on a non-exception nonzero status (`EXEC_RETRY`/
/// `EXEC_BREAKPOINT`) instead of routing it into `emit_exception_exit`,
/// which must only ever run for a real exception (`EXEC_IS_EXCEPTION` set).
struct EmitCtx<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
    module: &'a mut dyn cranelift_module::Module,
    core_ptr: Value,
    raw: u32,
    word: WordOffset,
    /// Compile-time-known Cause.BD value for whatever `emit_exception_exit`
    /// call site is currently being emitted with this `ctx` — `true` only
    /// while emitting a delay slot's own inlined semantics
    /// (`emit_slot_semantics` sets this before calling into the slot's
    /// semantics emitter), `false` for every ordinary (non-slot) head
    /// instruction, including one that's independently reachable as this
    /// same region's own branch/jump target (§6.1.4 dual semantics) — that
    /// occurrence is a structurally separate `ctx`/emission from the
    /// inlined-slot one, with its own `bd = false`, never sharing state with
    /// it. `emit_exception_exit` passes this straight through to
    /// `exception_other_word_block` as a literal, so BD is always written
    /// unconditionally at the exception-exit call site itself rather than
    /// trusted from whatever's already live in `core.in_delay_slot`.
    bd: bool,
    /// `true` iff an exception raised while emitting with this `ctx` must
    /// **trust the live `core.pc`/`core.in_delay_slot`** (route through
    /// `exception_entry_word_block`) rather than overwriting them from the
    /// compile-time `word`/`bd` (`exception_other_word_block`). Set for any
    /// entry word (`instrs[word].is_entry_point`, §13.4 — state set by the
    /// interpreter dispatch that reached it) and for a *branch-fallback
    /// successor* — the delay slot of a BC1 interpreter-fallback, reached
    /// with `core.pc = slot_addr` and `in_delay_slot = true` already correct
    /// from the fallback's interpreter run; overwriting them (BD would
    /// become the compile-time `false`) would give a faulting slot the wrong
    /// `Cause.BD`/EPC. Distinct from `bd`: `bd` is a compile-time literal for
    /// the *other-word* path; this bool selects *which* path entirely.
    trust_live_pc_bd_on_exc: bool,
    exit_block: Block,
    /// Two-stage shared exception-raise blocks (see their own doc comments
    /// on `BlockSkeleton`) — `emit_exception_exit` picks which outer stage
    /// to jump to at *compile* time (`ctx.trust_live_pc_bd_on_exc` is always
    /// known when emitting a given call site — see its own doc comment), so
    /// no call site ever pays a runtime check for something that's actually
    /// fixed for that site.
    exception_call_block: Block,
    exception_entry_word_block: Block,
    exception_other_word_block: Block,
    /// Compile-time-only running total of retired-but-not-yet-stored
    /// instructions since the last `core.hot.cycles` flush — see the
    /// analyzer's `CompiledInstr::cycles_delta`/`cycles_flush` doc comments
    /// for the full design (batching `emit_increment_cycles`'s old
    /// per-instruction store into one store per flush point). Lives outside
    /// `EmitCtx` itself, in `compile_region_uncommitted`'s pass-2 loop —
    /// this is a `&mut` borrow of it, not owned state, because a fresh
    /// `EmitCtx` is constructed every loop iteration (one per head
    /// instruction) but the pending count must survive across iterations
    /// until a flush resets it to 0. Threaded via `EmitCtx` rather than as
    /// a plain function parameter because `emit_slot_semantics`/
    /// `try_emit_fused_nop_slot` (which also need to add to it) are called
    /// from several different call chains several frames below the pass-2
    /// loop (`emit_branch_or_jump`, `emit_regjump`,
    /// `emit_nested_branch_slot`, `emit_nested_regjump_slot`) — `EmitCtx`
    /// is already threaded through every one of those as `&mut ctx`, so
    /// this rides along for free instead of widening every signature on
    /// the path.
    cycles_pending: &'a mut u32,
}

/// Result of the block-allocation first pass: every visited instruction's
/// word offset paired with the `Block` codegen created for it, plus the
/// dedicated entry block that jumps to `entry_word`'s block, plus the shared
/// exit-to-interpreter block. Not yet a finished function — no instruction
/// bodies, no edges between the per-instruction blocks besides the entry jump.
pub struct BlockSkeleton {
    /// The function-entry Cranelift block (unsealed until the caller finishes
    /// wiring the CFG). Jumps unconditionally to `entry_word`'s block.
    pub entry_block: Block,
    /// Single shared "exit to interpreter" block for the whole function
    /// (§3.3 — every clean exit, preamble bail or fallthrough_exit/taken_exit
    /// stub alike, jumps here instead of each emitting its own copy of the
    /// materialize-PC-and-return sequence). Takes one `i64` block param: the
    /// exiting instruction's word offset. Its body is emitted here, in
    /// `build_block_skeleton`, since it never varies — only the jumps into
    /// it do. Left unsealed: its predecessors (every bail site in the
    /// function) aren't all known until the whole function is built; the
    /// caller must seal it once emission is complete. Use [`emit_bail`] at
    /// every exit site instead of duplicating this block's body.
    pub exit_block: Block,
    /// Shared "raise an architectural exception" machinery for the whole
    /// function — the code-density counterpart of `exit_block`, but for
    /// `emit_exception_exit`'s call sites instead of `emit_bail`'s (overflow
    /// traps, TEQ/TNE-family traps, memory faults, FPU exceptions, the
    /// CU1/FR guard: every one of them used to emit its own full inline copy
    /// of the delay-slot-synthesis check + `handle_exception_fn` call, which
    /// showed up directly in compiled code size for any region touching more
    /// than one of them). Split into three blocks, two-stage, rather than
    /// one block taking `word` as a runtime param: `ctx.trust_live_pc_bd_on_exc`
    /// is always known at *compile* time for any given `emit_exception_exit`
    /// call site, so there is no need to pay a runtime check for it — each
    /// call site picks the right outer stage directly.
    ///
    /// - [`Self::exception_other_word_block`]: outer stage for every
    ///   non-entry-word call site — unconditionally synthesizes `fault_pc`
    ///   AND writes `core.in_delay_slot` from its own `bd` param (never
    ///   trusts either field's live value on entry) before falling into the
    ///   inner stage. Params: `(core_ptr, word: i64, bd: i8, status: i32)` —
    ///   `word`/`bd` are genuine runtime values here since this one block is
    ///   shared across every non-entry word in the region, both inside a
    ///   delay slot's own inlined semantics (`bd=true`) and not (`bd=false`)
    ///   — see `emit_exception_exit`'s doc comment.
    /// - [`Self::exception_entry_word_block`]: outer stage for entry-word
    ///   call sites — unconditional, no synthesis and no check at all (just
    ///   falls into the inner stage). Correct because `core.pc`/
    ///   `core.in_delay_slot` are guaranteed already correct on arrival at
    ///   entry_word by construction: the interpreter's own dispatch sets
    ///   them correctly on the external-entry path, and `entry_word_block`
    ///   (the target of every *internal* edge into entry_word — see its doc
    ///   comment in `compile_region_uncommitted`) materializes them itself
    ///   before falling into the shared body, since that's the only point
    ///   that knows the edge is internal. See
    ///   `emit_exception_entry_word_block_body`'s doc comment for why a
    ///   runtime check *here* cannot substitute for that — by this point
    ///   there is no signal left to distinguish the two arrival paths.
    ///   Params: `(core_ptr, status: i32)`.
    /// - [`Self::exception_call_block`]: inner stage, the actual
    ///   `handle_exception_fn` call + return — assumes `core.pc`/
    ///   `core.in_delay_slot` are already correct on entry (both outer
    ///   stages guarantee this before falling through). Params:
    ///   `(core_ptr, status: i32)`.
    ///
    /// All three bodies are emitted once here, in `build_block_skeleton`.
    /// Left unsealed for the same reason `exit_block` is — the caller must
    /// seal them once every exception-exit site has been emitted.
    pub exception_call_block: Block,
    pub exception_entry_word_block: Block,
    pub exception_other_word_block: Block,
    /// (word offset, allocated block) for every instruction in the region,
    /// in ascending word-offset order (mirrors `instrs_linear`'s order).
    pub instr_blocks: Vec<(WordOffset, Block)>,
}

/// `opt_level` used by every `Codegen::new_module()` call from here on —
/// process-wide, not per-`Codegen`, since `opt_level` is baked into the ISA
/// `Flags` at module-construction time and can't be changed for an
/// already-built `JITModule`. `j2 opt [none|speed]` flips this; it only
/// takes effect on the *next* `Codegen::new()`/`reset()` (a flush, or a
/// fresh process), not retroactively for functions already compiled — same
/// "takes effect on the next relevant event" contract as `j2 batch`/`j2
/// inline`. `AtomicBool` (not the setting's own string) since only two
/// values are ever offered here; `true` = "speed".
///
/// Defaults to `false` (`none`) under `developer` — diagnostics builds want
/// the simplest, most predictable codegen output, not real Cranelift
/// optimization passes reordering/eliding IR mid-investigation — and to
/// `true` (`speed`) otherwise, since production runs want the faster
/// generated code `speed` produces.
static CODEGEN_OPT_LEVEL_SPEED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(!cfg!(feature = "developer"));

impl Codegen {
    /// Set the `opt_level` used by future `Codegen::new()`/`reset()` calls.
    /// `speed` trades slower compiles for faster generated code (real
    /// register allocation, instruction selection heuristics tuned for
    /// runtime cost rather than compile cost); `none` (the default) is what
    /// this codegen has always used — compile latency matters more than
    /// codegen quality for `opt_level=none`-sized single/few-instruction
    /// regions, but `speed` is worth comparing once regions grow larger
    /// (`MAX_INSTRS_PER_COMPILE`).
    pub fn set_opt_level_speed(speed: bool) {
        CODEGEN_OPT_LEVEL_SPEED.store(speed, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn opt_level_speed() -> bool {
        CODEGEN_OPT_LEVEL_SPEED.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Host mmap page granularity `ArenaMemoryProvider` rounds every
    /// function's segment up to (`memory/arena.rs`'s own `align_up(size,
    /// page::size())`, via the `region` crate — a transitive dependency of
    /// `cranelift-jit`, not depended on directly here, so this is a named
    /// assumption rather than a call to `region::page::size()`). 4KiB is
    /// correct for every platform this project actually targets
    /// (`target-cpu=native` x86-64/aarch64 Linux and macOS) — if that ever
    /// changes, this is the one place to update, not a magic number buried
    /// in `code_bytes_used`'s arithmetic.
    ///
    /// `CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES`'s own doc comment already derived
    /// this fact independently (confirmed live: the arena exhausts at
    /// *exactly* `ARENA_RESERVE_SIZE / HOST_PAGE_SIZE` functions, not a
    /// byte-size estimate) — this constant makes that same fact available
    /// to `Jitv2::code_bytes_used`'s per-entry accounting instead of just
    /// the flush-threshold math.
    pub const HOST_PAGE_SIZE: u64 = 4096;

    /// Default path's arena reservation — see this constant's own call site
    /// for the shared doc comment on why one big reservation beats
    /// per-function mmap. `j2wp` uses `crate::jitv2::jitv2::ARENA_RESERVE_SIZE`
    /// instead (2GiB, not 512MiB — a page-sized compiled function is larger
    /// than an entry-sized one, so the whole-page redesign needs more
    /// headroom before hitting `CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES`).
    #[cfg(not(feature = "j2wp"))]
    pub(crate) const ARENA_RESERVE_SIZE: usize = 512 * 1024 * 1024;

    /// `cranelift_jit`'s default `SystemMemoryProvider` mmaps (or
    /// `alloc::alloc`s, which on Linux still routes through mmap for
    /// large/page-sized allocations) a fresh chunk sized to fit exactly
    /// each function that doesn't fit whatever's left in the current chunk
    /// (`memory.rs`'s own `// TODO: Allocate more at a time.` — it never
    /// over-allocates or reuses freed space). With small `opt_level=none`
    /// regions and no coalescing, that's close to one distinct VMA per
    /// compiled function — confirmed live: a process with ~130k compiled
    /// functions was sitting at exactly `vm.max_map_count` (262144)
    /// mappings, and the next mmap call of any kind (even an unrelated new
    /// thread's stack guard page) then failed. `ArenaMemoryProvider`
    /// reserves one big contiguous region up front instead and
    /// bump-allocates every function into it — one mapping total,
    /// regardless of function count, for as long as it fits within
    /// `ARENA_RESERVE_SIZE`.
    /// Build a `JITModule` bound to `shared` (an existing shared arena) if
    /// given, or a brand-new private one-owner arena otherwise. Two handles
    /// are always built over whichever arena is used: `module_handle` is
    /// moved into the `JITModule` (which owns its memory provider opaquely,
    /// by value — see `paged_memory`'s module doc comment — so nothing can
    /// ever be read back out of it), and `seal_handle` is kept by the
    /// caller (`Codegen`) so it can drive non-forced sealing
    /// (`try_seal_pending`) and cross-arena bookkeeping
    /// (`compile_region_uncommitted`'s range tracking) without needing
    /// anything back from the module.
    fn new_module(
        shared: Option<std::sync::Arc<parking_lot::Mutex<crate::jitv2::paged_memory::SharedArena>>>,
        state: Option<std::sync::Arc<crate::jitv2::paged_memory::PagedArenaState>>,
    ) -> (
        cranelift_jit::JITModule,
        std::sync::Arc<crate::jitv2::paged_memory::PagedArenaState>,
        crate::jitv2::paged_memory::PagedArenaMemoryProvider,
    ) {
        let mut flag_builder = settings::builder();
        let opt_level = if Self::opt_level_speed() { "speed" } else { "none" };
        flag_builder.set("opt_level", opt_level).unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        let isa_builder = cranelift_native::builder().expect("host ISA not supported");
        let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();
        let mut jit_builder = cranelift_jit::JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let (shared, paged_state) = match shared {
            Some(shared) => (shared, state.expect("new_module: shared arena given without its PagedArenaState")),
            None => {
                let paged_state = std::sync::Arc::new(crate::jitv2::paged_memory::PagedArenaState::default());
                #[cfg(not(feature = "j2wp"))]
                let reserve_size = Self::ARENA_RESERVE_SIZE;
                #[cfg(feature = "j2wp")]
                let reserve_size = crate::jitv2::jitv2::ARENA_RESERVE_SIZE;
                let shared = crate::jitv2::paged_memory::PagedArenaMemoryProvider::new_shared(reserve_size, paged_state.clone())
                    .expect("failed to reserve jitv2 Codegen arena");
                (shared, paged_state)
            }
        };
        // Shared only between these two sibling handles (not the whole
        // arena/pool) — see `PagedArenaMemoryProvider::last_allocation`'s
        // own doc comment for why this exists: `seal_handle` needs to learn
        // each individual `allocate()` call's exact range, race-free,
        // without racing other workers' concurrent allocations the way
        // reading the shared arena's own live bump cursor would. Only
        // `module_handle`'s own `allocate()` calls ever write to it (it's
        // the one actually inside `JITModule`); `seal_handle` only ever
        // reads via `take_last_allocation`.
        let last_allocation = std::sync::Arc::new(parking_lot::Mutex::new(None));
        let module_handle = crate::jitv2::paged_memory::PagedArenaMemoryProvider::from_shared_with_mailbox(shared.clone(), last_allocation.clone());
        let seal_handle = crate::jitv2::paged_memory::PagedArenaMemoryProvider::from_shared_with_mailbox(shared, last_allocation);
        jit_builder.memory_provider(Box::new(module_handle));
        (cranelift_jit::JITModule::new(jit_builder), paged_state, seal_handle)
    }

    /// Standalone `Codegen` with its own private, freshly-reserved arena —
    /// today's only construction path (inline mode, `equiv_test.rs`,
    /// `jitv2_lockstep`, this module's own tests). Behaviorally unchanged
    /// by the shared-arena machinery: with exactly one handle ever over
    /// this arena, every `finalize()` call's pushed range is always the
    /// entire contiguous-from-watermark prefix, so `try_seal_ready` always
    /// seals it immediately.
    pub fn new() -> Self {
        Self::from_module(Self::new_module(None, None))
    }

    /// `Codegen` built on top of an *already-reserved* shared arena — the
    /// compile-pool worker constructor (each worker gets its own `Codegen`/
    /// `JITModule`, but they all share one arena reservation, so `Codegen`
    /// itself can't be the one to `new_with_size` it) and the `j2 inline`
    /// mode-switch path (switching modes rebuilds `Codegen`(s) on top of
    /// whichever arena the outgoing mode's `Codegen` was already using,
    /// rather than reserving a fresh one and leaking the old — see
    /// `PagedArenaMemoryProvider::shared`'s own doc comment).
    pub fn new_with_shared_arena(
        shared: std::sync::Arc<parking_lot::Mutex<crate::jitv2::paged_memory::SharedArena>>,
        state: std::sync::Arc<crate::jitv2::paged_memory::PagedArenaState>,
    ) -> Self {
        Self::from_module(Self::new_module(Some(shared), Some(state)))
    }

    fn from_module(
        (module, paged_state, seal_handle): (
            cranelift_jit::JITModule,
            std::sync::Arc<crate::jitv2::paged_memory::PagedArenaState>,
            crate::jitv2::paged_memory::PagedArenaMemoryProvider,
        ),
    ) -> Self {
        Self {
            ctx: module.make_context(),
            module,
            builder_ctx: FunctionBuilderContext::new(),
            func_id_counter: 0,
            #[cfg(feature = "developer")]
            last_code_size: 0,
            last_compile_ran_out_of_memory: false,
            #[cfg(feature = "developer")]
            last_decline_was_verifier_error: false,
            last_finalize_failed: false,
            paged_state,
            seal_handle,
            func_ranges: std::collections::HashMap::new(),
        }
    }

    /// True iff the most recent successful `compile_region`/
    /// `compile_region_uncommitted` call's allocation started a brand-new
    /// host-page segment (as opposed to packing into, or growing, the
    /// previously-allocated one) — the exact, non-heuristic page-crossing
    /// signal `PagedArenaMemoryProvider` exists to provide (see its module
    /// doc comment). Edge-triggered: reading this clears it, so each call
    /// reports only what happened since the last read, not a sticky "have we
    /// ever crossed" flag. `worker_loop`'s batching loop polls this after
    /// every deferred compile to decide when to flush the pending batch —
    /// unconditional (not `developer`-gated), unlike `packing_stats` below:
    /// this drives real batching behavior, not just a diagnostic.
    pub fn provider_crossed_page(&self) -> bool {
        self.paged_state.crossed_page()
    }

    /// `(bytes_actually_used, bytes_reserved)` across every segment the
    /// underlying `PagedArenaMemoryProvider` has ever allocated — see
    /// `PagedArenaState::packing_stats`. `j2 stats`'s packing-quality report,
    /// but also unconditional (not `developer`-gated): the `.1` (reserved)
    /// half now drives the real flush decision in both `worker_loop` and the
    /// inline-compile path (`CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES`), so it
    /// can't be diagnostics-only anymore.
    pub fn packing_stats(&self) -> (u64, u64) {
        self.paged_state.packing_stats()
    }

    /// The `(shared arena, paged state)` pair this `Codegen`'s own module is
    /// built on — for a caller that wants to build a *sibling* `Codegen`
    /// over the exact same arena (`new_with_shared_arena`) rather than
    /// reserving a fresh one, e.g. a `j2 inline` mode switch reusing the
    /// outgoing mode's still-live arena instead of leaking it.
    pub fn shared_arena(&self) -> (
        std::sync::Arc<parking_lot::Mutex<crate::jitv2::paged_memory::SharedArena>>,
        std::sync::Arc<crate::jitv2::paged_memory::PagedArenaState>,
    ) {
        (self.seal_handle.shared(), self.paged_state.clone())
    }

    /// `(base_address, len)` of the underlying arena's real reservation —
    /// see `PagedArenaState::arena_range`. `j2 hugepages` only.
    #[cfg(feature = "developer")]
    pub fn arena_range(&self) -> (u64, u64) {
        self.paged_state.arena_range()
    }

    /// `j2 seal-queue`'s only caller — see
    /// `SharedArena::seal_queue_snapshot`'s own doc comment.
    pub fn seal_queue_snapshot(&self) -> crate::jitv2::paged_memory::SealQueueSnapshot {
        self.seal_handle.shared().lock().seal_queue_snapshot()
    }

    /// `j2 seal-queue list`'s only caller — see
    /// `SharedArena::seal_queue_entries`'s own doc comment.
    pub fn seal_queue_entries(&self) -> Vec<(usize, usize, bool, std::thread::ThreadId, *mut crate::jitv2::PhysicalCodePage)> {
        self.seal_handle.shared().lock().seal_queue_entries()
    }

    /// Reclaim every executable-memory page this `Codegen` has ever
    /// finalized and start fresh with a new, empty `JITModule`. Plain
    /// `Drop`/replacement (`*self = Codegen::new()`) does **not** do this —
    /// `cranelift_jit::Memory`'s `Drop` impl deliberately `mem::forget`s
    /// every allocation instead of freeing it, specifically so a dangling
    /// `JitFn` pointer can never become a use-after-free; the only way to
    /// actually reclaim the memory is `JITModule::free_memory`, which is
    /// `unsafe` for exactly the mirror-image reason (every `JitFn` this
    /// module ever returned becomes an immediately-dangling pointer the
    /// moment this returns).
    ///
    /// # Safety
    /// The caller must guarantee no `JitFn` obtained from this `Codegen`
    /// (via any earlier `compile_region` call) is still reachable/callable
    /// anywhere after this returns — e.g. `jitv2_lockstep`'s
    /// `lockstep_flush_if_grown`, which clears `lockstep_cache` (the only
    /// place those pointers are ever stored — never published into the real
    /// `page`/`entries` table) in the same operation.
    pub unsafe fn reset(&mut self) {
        unsafe { self.reset_inner(Self::new_module(None, None)) }
    }

    /// Same contract as `reset()`, but rebuilds on top of an
    /// already-reserved shared arena instead of freeing the old one and
    /// reserving a fresh one — the compile-pool flush-leader path (a fresh
    /// arena IS reserved once, by the leader, then every worker including
    /// itself rebuilds onto that same one) and the `j2 inline` mode-switch
    /// path (see `new_with_shared_arena`'s own doc comment) both want this
    /// instead of `reset()`'s always-fresh-reservation behavior.
    ///
    /// # Safety
    /// Same as `reset()` — no `JitFn` this `Codegen` ever produced may
    /// still be reachable/callable anywhere after this returns. Additionally,
    /// `shared` must not be the same arena this `Codegen` was already using
    /// (that would make `old_module.free_memory()` below invalidate the very
    /// arena this call is trying to keep using) — always a *fresh* arena in
    /// every real caller (the pool flush leader's newly-built one, or the
    /// other mode's still-live one during a switch).
    pub unsafe fn reset_with_shared_arena(
        &mut self,
        shared: std::sync::Arc<parking_lot::Mutex<crate::jitv2::paged_memory::SharedArena>>,
        state: std::sync::Arc<crate::jitv2::paged_memory::PagedArenaState>,
    ) {
        unsafe { self.reset_inner(Self::new_module(Some(shared), Some(state))) }
    }

    unsafe fn reset_inner(
        &mut self,
        (new_module, new_paged_state, new_seal_handle): (
            cranelift_jit::JITModule,
            std::sync::Arc<crate::jitv2::paged_memory::PagedArenaState>,
            crate::jitv2::paged_memory::PagedArenaMemoryProvider,
        ),
    ) {
        let old_module = std::mem::replace(&mut self.module, new_module);
        unsafe { old_module.free_memory(); }
        self.paged_state = new_paged_state;
        // Replace the old seal_handle too — it's a second handle over the
        // exact same Arc<Mutex<SharedArena>> the old module's own provider
        // just freed; holding onto it (or worse, calling try_seal_pending
        // through it) after that free would touch freed memory.
        self.seal_handle = new_seal_handle;
        self.ctx = self.module.make_context();
        self.func_id_counter = 0;
        self.func_ranges.clear();
        #[cfg(feature = "developer")]
        { self.last_code_size = 0; }
        self.last_compile_ran_out_of_memory = false;
    }

    /// Number of functions compiled into this `Codegen`'s `JITModule` since
    /// construction or the last `reset()` — a simple, dependency-free proxy
    /// for its Cranelift memory-arena growth (no Cranelift-internal byte-size
    /// API needed, same reasoning as `jitv2_lockstep`'s own
    /// `LOCKSTEP_CACHE_FLUSH_THRESHOLD`). Callers with an unbounded compile
    /// lifetime (`comp::handle_request`'s worker loop, `jitv2_inline_compile`)
    /// should flush once this crosses a threshold — `cranelift_jit::Memory`
    /// never frees on drop/replace (see `reset`'s own doc comment), so
    /// nothing else naturally bounds this.
    pub fn function_count(&self) -> u32 {
        self.func_id_counter
    }

    /// Compiled machine code size, in bytes, of the most recent successful
    /// `compile_region` call — see `last_code_size`'s own field doc comment.
    #[cfg(feature = "developer")]
    pub fn last_code_size(&self) -> u32 {
        self.last_code_size
    }

    /// Whether the most recent `compile_region` call returned `None`
    /// because the arena is out of memory, as opposed to a real codegen
    /// decline — see `last_compile_ran_out_of_memory`'s own field doc
    /// comment. `comp::handle_request` checks this immediately after a
    /// `None` result to decide whether to sticky-denylist the offset or
    /// flush and let it retry.
    pub fn last_compile_ran_out_of_memory(&self) -> bool {
        self.last_compile_ran_out_of_memory
    }

    /// Whether the most recent `finalize_batch_nonforced` call returned an
    /// empty `Vec` because `module.finalize_definitions()` itself errored,
    /// as opposed to a normal gap-blocked seal — see `last_finalize_failed`'s
    /// own field doc comment. Callers must check this immediately after an
    /// empty return, before assuming "just gap-blocked, safe to wait."
    pub fn last_finalize_failed(&self) -> bool {
        self.last_finalize_failed
    }

    /// Whether the most recent `compile_region` call's `None` return came
    /// from Cranelift's own verifier (`ModuleError::Compilation`) rather
    /// than the upfront no-emitter-for-this-instruction loop — see the
    /// field's own doc comment. Meaningless (always `false`) after a
    /// successful compile or an arena-out-of-memory decline; callers should
    /// only consult this immediately after a `None` result that
    /// `last_compile_ran_out_of_memory()` says isn't OOM.
    #[cfg(feature = "developer")]
    pub fn last_decline_was_verifier_error(&self) -> bool {
        self.last_decline_was_verifier_error
    }

    /// First codegen pass over a walked region (§3.1): create the function
    /// signature (`JitFn`'s shape — `extern "C" fn(*mut MipsCore) -> ExecStatus`,
    /// i.e. one pointer param, one `i32` return), then allocate one Cranelift
    /// `Block` per visited instruction plus the dedicated entry block that
    /// jumps to `entry_word`'s block.
    ///
    /// `instrs` is the analyzer's walked buffer (`Analyzer::walk`'s first
    /// return value); `entry_word` must be a visited offset in it (the same
    /// one passed to `walk`).
    ///
    /// Writes each instruction's allocated block back into
    /// `instrs[offset].block_id` (as `block.as_u32()`) so later passes can
    /// look up "what block did instruction X get" directly from the walked
    /// buffer instead of re-deriving it from `BlockSkeleton::instr_blocks`.
    ///
    /// Blocks are left unsealed — Cranelift requires every predecessor of a
    /// block to be known before sealing it, and this pass only establishes
    /// the entry edge; internal edges (fallthrough/taken) are wired by the
    /// next pass, which is also responsible for sealing.
    pub fn build_block_skeleton(
        &mut self,
        instrs: &mut [CompiledInstr; ENTRIES_PER_PAGE],
        entry_word: WordOffset,
    ) -> BlockSkeleton {
        self.ctx.func.signature = self.jit_fn_signature();
        self.ctx.func.name = ir::UserFuncName::user(0, self.func_id_counter);
        self.func_id_counter += 1;

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);

        let mut instr_blocks = Vec::new();
        let mut entry_word_block = None;
        for instr in instrs_linear(instrs) {
            let block = builder.create_block();
            if instr.word == entry_word {
                entry_word_block = Some(block);
            }
            instr_blocks.push((instr.word, block));
        }
        let entry_word_block = entry_word_block
            .expect("entry_word must be a visited offset in the walked buffer");

        builder.switch_to_block(entry_block);
        builder.ins().jump(entry_word_block, &[]);
        builder.seal_block(entry_block); // entry_block's only predecessor is the caller — always sealable immediately

        // Shared exit-to-interpreter block (see BlockSkeleton::exit_block's
        // doc comment). Params: (core_ptr, word_offset). core_ptr must be
        // its own block param — not the entry_block value captured above —
        // because a block with multiple/deferred predecessors can only use
        // values defined in blocks that dominate it; threading it as a param
        // is how Cranelift wants a value made available across an
        // as-yet-unknown set of incoming edges.
        let exit_block = builder.create_block();
        let ptr_ty = builder.func.signature.params[0].value_type;
        let exit_core_ptr = builder.append_block_param(exit_block, ptr_ty);
        let word_offset_param = builder.append_block_param(exit_block, ir::types::I64);
        builder.switch_to_block(exit_block);
        emit_exit_block_body(&mut builder, &mut self.module, exit_core_ptr, word_offset_param);
        // Not sealed: predecessors are every bail site across the whole
        // function, established incrementally as later passes emit them.

        // Shared exception-raise machinery (see BlockSkeleton's own doc
        // comment for the two-stage rationale) — same block-param pattern as
        // exit_block above, split into three blocks so no call site ever
        // pays a runtime word==entry_word check.
        let exception_call_block = builder.create_block();
        let call_core_ptr = builder.append_block_param(exception_call_block, ptr_ty);
        let call_status_param = builder.append_block_param(exception_call_block, ir::types::I32);
        builder.switch_to_block(exception_call_block);
        emit_exception_call_block_body(&mut self.module, &mut builder, call_core_ptr, call_status_param);

        let exception_other_word_block = builder.create_block();
        let other_core_ptr = builder.append_block_param(exception_other_word_block, ptr_ty);
        let other_word_param = builder.append_block_param(exception_other_word_block, ir::types::I64);
        let other_bd_param = builder.append_block_param(exception_other_word_block, ir::types::I8);
        let other_status_param = builder.append_block_param(exception_other_word_block, ir::types::I32);
        builder.switch_to_block(exception_other_word_block);
        emit_exception_other_word_block_body(&mut builder, other_core_ptr, other_word_param, other_bd_param, other_status_param, exception_call_block);

        let exception_entry_word_block = builder.create_block();
        let entry_exc_core_ptr = builder.append_block_param(exception_entry_word_block, ptr_ty);
        let entry_exc_status_param = builder.append_block_param(exception_entry_word_block, ir::types::I32);
        builder.switch_to_block(exception_entry_word_block);
        emit_exception_entry_word_block_body(&mut builder, entry_exc_core_ptr, entry_exc_status_param, exception_call_block);
        // None of the three sealed here: predecessors (every emit_exception_exit
        // call site, plus the two outer stages' own jumps into
        // exception_call_block) are established incrementally as later
        // passes emit them.

        // Write blocks back into the walked buffer so later passes can find
        // "the block for instruction at word W" via instrs[w].block_id
        // directly, without threading instr_blocks through separately.
        for &(word, block) in &instr_blocks {
            instrs[word as usize].block_id = Some(block.as_u32());
        }

        // builder is dropped here (not finalized) — this pass only stakes
        // out the block graph's nodes and the single entry edge. The next
        // pass reopens each block via switch_to_block, emits its
        // instruction's semantics, wires fallthrough/taken edges, and seals
        // once all of a block's predecessors are known.
        drop(builder);

        BlockSkeleton { entry_block, exit_block, exception_call_block, exception_entry_word_block, exception_other_word_block, instr_blocks }
    }

    /// Signature for `JitFn` (`jitv2/jitv2.rs`): `extern "C" fn(*mut MipsCore) -> ExecStatus`.
    fn jit_fn_signature(&self) -> ir::Signature {
        let mut sig = self.module.make_signature();
        let ptr_type = self.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_type)); // *mut MipsCore
        sig.returns.push(AbiParam::new(ir::types::I32)); // ExecStatus (u32)
        sig
    }

    /// Compile a walked region (`Analyzer::walk`/`walk_bounded`'s output)
    /// into a callable `JitFn`, end to end: one Cranelift block per
    /// **head** instruction (§3.1) — branch/jump delay slots do not get
    /// their own block, see below — both per-unit preambles (§3.2), each
    /// head instruction's semantics/control-flow, and the edges between
    /// blocks or out to the shared exit block.
    ///
    /// Two-pass block wiring: pass 1 allocates every head instruction's
    /// block; pass 2 emits bodies and every outgoing edge (fallthrough,
    /// taken, exit) *without sealing*, since a branch's taken target can be
    /// any word in the region — including backward (loop back-edges) — so a
    /// block's full predecessor set isn't known until every other
    /// instruction has been emitted; pass 3 seals every block once all
    /// edges exist. (The single-linear-chain shape used before this
    /// function supported branches was a special case where "seal
    /// immediately, next block is always the very next list entry" happened
    /// to be safe — no longer true in general.)
    ///
    /// **Delay slots are inlined, never independently chained (§6.1.4
    /// "indivisible unit" / "never a CFG edge").** The interpreter's
    /// `branch_delay`/`in_delay_slot`/`delay_slot_target` two-dispatch
    /// mechanism is executor-private state compiled code cannot reach (and
    /// does not need to): a branch/jump's delay slot always executes exactly
    /// once, regardless of whether the branch is taken, so its semantics are
    /// emitted directly inside the branch/jump's own block, unconditionally,
    /// before the condition test / target jump. The analyzer still marks the
    /// slot's own `CompiledInstr` as `visited` (so `instrs_linear` yields it)
    /// — pass 1 recognizes it (`is_delay_slot_of`) and skips allocating an
    /// independent block for it, so it is never reachable as its own
    /// fallthrough-chained node.
    ///
    /// **Batch scope**: `Sequential` instructions (`lookup_semantics`),
    /// conditional/unconditional branches and jumps including the annulling
    /// "Likely" family and link-writing variants (`lookup_branch_or_jump`),
    /// and register-indirect JR/JALR (`lookup_regjump` — target only known
    /// at runtime, always exits via [`emit_runtime_pc_exit`] rather than
    /// the shared `exit_block`). A branch/jump/regjump at 0xFFC
    /// (`StopReason::ForeignPageSlot`) is a normal visited head like any
    /// other — `emit_branch_or_jump`/`emit_regjump`'s own `foreign_page_slot`
    /// check handles the missing on-page slot by arming
    /// `core.in_delay_slot`/`core.delay_slot_target` instead of inlining it,
    /// same as the interpreter's `branch_delay`.
    ///
    /// Returns `None` if any visited instruction has no emitter yet.
    /// Returns `Some(JitFn)` — a raw function pointer valid for as long as
    /// `self` (its `JITModule`) lives; callers must not drop `self` while
    /// the returned function might still be called.
    ///
    /// No `page_base` parameter, deliberately: the compiled function is
    /// position-independent (§2.2) — every address it ever materializes into
    /// `core.pc` (branch/jump targets, delay-slot exception EPC, exit-stub
    /// retries) is derived from a runtime load of live `core.pc`, never from
    /// a compile-time constant baked in at codegen time. A `page_base`
    /// parameter here previously existed and was physical (mirroring
    /// `comp.rs`'s `phys_base`, the real production caller's only page
    /// address) — using it to synthesize *virtual* addresses was a real bug
    /// (harmless only for kseg0/kseg1, where physical and virtual-low-32-bits
    /// happen to coincide) fixed this session; removing the parameter
    /// entirely makes that class of mistake impossible to reintroduce rather
    /// than just fixing this generation of it. `Analyzer::walk_bounded`
    /// still takes a `page_base` — that one is a genuinely different,
    /// necessary input: the *compile-time decision* of whether a branch/jump
    /// target is on- or off-page, not a value ever written into `core.pc`.
    /// `skip_entry_preamble`: when `true`, the entry word's own IP7/pending-
    /// interrupt checks (§3.2's "check 1"/"check 2") are omitted from the
    /// path reached directly from the function's entry — the production
    /// caller (`comp.rs`'s `handle_request`, reached from
    /// `mips_exec.rs::step()`'s dispatch) already ran both checks for this
    /// exact PC, in the interpreter loop, immediately before ever calling
    /// into compiled code; repeating them here would just recheck the same
    /// state a second time for no reason. `false` for every other caller
    /// (the equivalence-test harness, `jitv2_verify`): none of them are
    /// preceded by a real interpreter dispatch loop, so the entry word's
    /// preamble is the *only* place those checks would ever run for it —
    /// omitting them there would silently under-test/under-implement the
    /// checks these tools exist to verify.
    ///
    /// Only ever affects entry words themselves: every other head in the
    /// region still gets its full preamble unconditionally, since nothing
    /// but the interpreter's own dispatch (which only ever lands on an entry
    /// word) can have already performed those checks for it. If an
    /// *internal* edge (e.g. a backward branch) targets an entry word, that
    /// transfer never went through the interpreter's dispatch either — see
    /// `block_for_word`'s construction below, which always resolves an entry
    /// word to its full-preamble block, never its skipped-preamble body
    /// block, regardless of this flag.
    ///
    /// §13.4: takes every entry point's own reachability walk, already
    /// merged into one buffer via [`crate::jitv2::analyzer::Analyzer::walk_multi_entry`]
    /// — `instrs[w].is_entry_point` (set by that walk) is now what
    /// distinguishes an entry word from an ordinary internal instruction,
    /// wherever this function used to compare against a single scalar
    /// `entry_word` parameter. Every entry point analyzed from one buffer
    /// shares this compile's `compiled_for_fr1`/`skip_entry_preamble` — see
    /// `CompileRequest::compiled_for_fr1`'s own doc comment for why that's
    /// correct (FR mode and the entry-skip contract are both properties of
    /// the *compile*, not of any one entry within it).
    ///
    /// Compile a region into Cranelift IR and hand it to `define_function`,
    /// but do **not** finalize or return a callable pointer — the shared
    /// first half of `compile_region`'s (immediate) and `finalize_batch`'s
    /// (deferred) paths. Returns the `FuncId` `finalize_batch` needs to
    /// retrieve the eventual `JitFn`; callers that need a callable pointer
    /// right away should use `compile_region` instead, not this directly.
    ///
    /// `has_fpu`: whether this region contains any CP1/FPU instruction —
    /// computed by the walk itself (`Analyzer::walk_multi_entry`'s own
    /// return value, `analyzer::is_fpu_instruction`) and passed straight
    /// through here rather than re-scanned, so this function never needs
    /// its own opinion on which opcodes are FPU-shaped.
    pub fn compile_region_uncommitted(
        &mut self,
        instrs: &mut [CompiledInstr; ENTRIES_PER_PAGE],
        compiled_for_fr1: bool,
        skip_entry_preamble: bool,
        has_fpu: bool,
        page: *mut crate::jitv2::PhysicalCodePage,
    ) -> Option<cranelift_module::FuncId> {
        let fr_mode = if compiled_for_fr1 { FrMode::Fr1 } else { FrMode::Fr0 };
        #[cfg(feature = "developer")]
        { self.last_decline_was_verifier_error = false; }
        // Reject anything this pass doesn't support before touching
        // Cranelift at all. `ForeignPageSlot`/`RegJump`/other exit reasons
        // that land on a *visited* instruction (rather than terminating an
        // edge out of the region) aren't possible per the analyzer's
        // contract — every visited instruction must have a semantics or
        // branch/jump emitter, full stop.
        for instr in instrs_linear(instrs) {
            // A fallback head (analyzer-Excluded, kept in the region) has no
            // native emitter by definition — it runs through the interpreter
            // (emit_interp_fallback_head), so it's exempt from the
            // must-have-an-emitter rule below.
            if instr.is_fallback {
                continue;
            }
            if lookup_semantics(instr.raw).is_none()
                && lookup_branch_or_jump(instr.raw).is_none()
                && lookup_regjump(instr.raw).is_none()
                && lookup_cp1_semantics(instr.raw).is_none()
            {
                return None; // no emitter for this instruction yet (includes excluded-adjacent shapes)
            }
        }

        self.ctx.func.signature = self.jit_fn_signature();
        self.ctx.func.name = ir::UserFuncName::user(0, self.func_id_counter);

        // One FunctionBuilder for the whole function — block allocation,
        // preamble/semantics emission, and finalization all happen in this
        // single session. Splitting this across two `FunctionBuilder::new`
        // calls (build-skeleton-then-drop, reopen-and-continue) is unsound:
        // the builder owns SSA-construction state in `builder_ctx` that a
        // fresh `::new` would reset against a function that already has
        // blocks/instructions in it. `build_block_skeleton` stays a
        // self-contained probe for the block-allocation-only tests, which
        // never resume the builder afterward.
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);

        // Pass 1: allocate one block per head instruction (visited, and not
        // some other instruction's inlined delay slot), in ascending word
        // order (instrs_linear's guarantee) — order doesn't matter for
        // correctness here (every edge is wired explicitly in pass 2 by
        // looking up the target word's block), just for determinism.
        //
        // `is_slot_only` is the analyzer's own bookkeeping (`visit_slot`/
        // `visit`'s promotion — §6.1.4 dual semantics), not recomputed here:
        // a word visited *only* as a delay slot never gets a block (it's
        // inlined into its predecessor unconditionally, `emit_slot_semantics`);
        // a word that's also reached as a genuine branch/jump target was
        // already promoted by the walker and has real edges — those get a
        // block like any other head.
        let mut instr_blocks: Vec<(WordOffset, Block)> = Vec::new();
        let mut entry_words: Vec<WordOffset> = Vec::new();
        for instr in instrs_linear(instrs) {
            if instr.is_slot_only {
                continue;
            }
            let block = builder.create_block();
            if instr.is_entry_point {
                entry_words.push(instr.word);
            }
            instr_blocks.push((instr.word, block));
        }
        assert!(!entry_words.is_empty(), "at least one entry_word must be a visited, non-slot offset in the walked buffer");
        // Deterministic order (ascending, matching instr_blocks/instrs_linear)
        // — nothing depends on it structurally, but a fixed iteration order
        // keeps codegen output reproducible across runs of the same input.
        entry_words.sort_unstable();
        let block_for_word_prelim: std::collections::HashMap<WordOffset, Block> =
            instr_blocks.iter().copied().collect();
        let entry_word_block_for = |w: WordOffset| -> Block {
            *block_for_word_prelim.get(&w)
                .expect("entry_word must be a visited, non-slot offset in the walked buffer")
        };

        // When skipping an entry word's own preamble (production path — see
        // compile_region's doc comment on skip_entry_preamble), EVERY entry
        // word needs two distinct blocks, one pair per entry (§13.4): its
        // ordinary block (preamble + semantics, the target for every
        // *internal* edge — a backward branch landing on an entry word never
        // went through the interpreter's dispatch, so it must still pay the
        // checks) and a dedicated body block, reached directly from the
        // function's dispatch head, bypassing the preamble. The ordinary
        // block's own body becomes just "run the preamble, then jump to the
        // body block" in this mode instead of containing the semantics
        // itself. One map, keyed by entry word, rather than a single
        // `Option<Block>` — the old single-entry shape.
        let mut entry_body_blocks: std::collections::HashMap<WordOffset, Block> = std::collections::HashMap::new();
        if skip_entry_preamble {
            for &w in &entry_words {
                entry_body_blocks.insert(w, builder.create_block());
            }
        }

        // entry_block must be the first block Cranelift lays out (it must
        // stay the function's actual entry point, matching the function
        // signature's param count) — `switch_to_block` is what places a
        // block into layout order, not `create_block`, so entry_block must
        // be switched into before exit_block's body is emitted, even though
        // exit_block's *handle* was allocated earlier up in the function.
        // (Caught by the Cranelift verifier: an earlier ordering here
        // switched to exit_block first, silently making IT the two-param
        // "entry block" the verifier then rejected against the one-param
        // function signature.)
        builder.switch_to_block(entry_block);
        let core_ptr = builder.block_params(entry_block)[0];

        let exit_block = builder.create_block();
        let ptr_ty = builder.func.signature.params[0].value_type;
        let exit_core_ptr = builder.append_block_param(exit_block, ptr_ty);
        let word_offset_param = builder.append_block_param(exit_block, ir::types::I64);

        let exception_call_block = builder.create_block();
        let call_core_ptr = builder.append_block_param(exception_call_block, ptr_ty);
        let call_status_param = builder.append_block_param(exception_call_block, ir::types::I32);

        let exception_other_word_block = builder.create_block();
        let other_core_ptr = builder.append_block_param(exception_other_word_block, ptr_ty);
        let other_word_param = builder.append_block_param(exception_other_word_block, ir::types::I64);
        let other_bd_param = builder.append_block_param(exception_other_word_block, ir::types::I8);
        let other_status_param = builder.append_block_param(exception_other_word_block, ir::types::I32);

        let exception_entry_word_block = builder.create_block();
        let entry_exc_core_ptr = builder.append_block_param(exception_entry_word_block, ptr_ty);
        let entry_exc_status_param = builder.append_block_param(exception_entry_word_block, ir::types::I32);

        // §13.4 internal dispatch head: this page's one compiled function may
        // cover several external entry points, so the function itself must
        // find out which one it's being called for. No parameter needed:
        // the offset is already implicit in live `core.pc`, exactly the way
        // the interpreter's own dispatch derives it — `(pc & 0xFFF) >> 2`.
        // Computed here, before the FR-mode guard, because the guard's own
        // kill/fallback bail (on an FR mismatch) also needs to know the live
        // entry offset — see emit_fr_mode_guard's doc comment.
        let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
        let mem = MemFlagsData::trusted();
        let live_pc = builder.ins().load(ir::types::I64, mem, core_ptr, pc_off);
        let page_off_mask = builder.ins().iconst(ir::types::I64, 0xFFF);
        let masked = builder.ins().band(live_pc, page_off_mask);
        let live_entry_offset = builder.ins().ushr_imm_u(masked, 2);

        // Region-wide FR-mode guard ONLY (not CU1 — see emit_cp1_cu1_guard,
        // called per-instruction in pass 2 below), emitted once when this
        // region actually contains a CP1 instruction (`has_fpu`, computed
        // once by the walk itself — see this function's own doc comment).
        // `compiled_for_fr1` is resolved once, at compile time, from the
        // live STATUS_FR bit when the compile started (per-page pinned FR
        // mode, §13 — PhysicalCodePage::fr1); this is also the mode every
        // FPR-access emitter in the region uses (threaded via `fr_mode` in
        // pass 2). Still positioned in entry_block here — legitimate as a
        // region-wide, entry-time check because FR mode (unlike CU1) really
        // is a whole-compile invariant — see emit_fr_mode_guard's own doc
        // comment.
        if has_fpu {
            // Not part of any head instruction's own retirement — this
            // guard runs (at most) once in entry_block, before any
            // instruction's cycles_delta/cycles_flush bookkeeping begins,
            // so a throwaway local is correct here (never read back).
            let mut unused_cycles_pending = 0u32;
            let mut guard_ctx = EmitCtx { builder: &mut builder, module: &mut self.module, core_ptr, raw: 0, word: 0, bd: false, trust_live_pc_bd_on_exc: true, exit_block, exception_call_block, exception_entry_word_block, exception_other_word_block, cycles_pending: &mut unused_cycles_pending };
            emit_fr_mode_guard(&mut guard_ctx, live_entry_offset, compiled_for_fr1);
        }

        // Dispatch: a `Switch` over every entry this compile covers, jumping
        // to that entry's own body block (bypassing its preamble, when
        // skip_entry_preamble) or its ordinary block otherwise — Cranelift
        // lowers this to a `br_table` when the case count/density justify
        // it, or a compare chain otherwise (no separate policy needed here).
        // Any offset none of `entry_words` recognizes (including "some
        // other compile's coverage was clobbered when this compile's
        // publish overwrote the page's shared `func` slot" — a page-level
        // race, not a per-compile concern) bails via EXEC_FALLBACK,
        // untouched core.pc, so the caller's normal interpret-or-recompile
        // path takes over exactly as if this function had never been
        // called.
        let dispatch_miss_block = builder.create_block();
        let mut switch = cranelift_frontend::Switch::new();
        for &w in &entry_words {
            let target = entry_body_blocks.get(&w).copied().unwrap_or_else(|| entry_word_block_for(w));
            switch.set_entry(w as u128, target);
        }
        switch.emit(&mut builder, live_entry_offset, dispatch_miss_block);
        builder.seal_block(entry_block); // entry_block's only predecessor is the caller — always sealable immediately

        builder.switch_to_block(dispatch_miss_block);
        builder.seal_block(dispatch_miss_block);
        let fallback_status = builder.ins().iconst(ir::types::I32, crate::mips_exec::EXEC_FALLBACK as i64);
        builder.ins().return_(&[fallback_status]);

        builder.switch_to_block(exit_block);
        emit_exit_block_body(&mut builder, &mut self.module, exit_core_ptr, word_offset_param);
        // Left unsealed until every bail site below has been emitted.

        builder.switch_to_block(exception_call_block);
        emit_exception_call_block_body(&mut self.module, &mut builder, call_core_ptr, call_status_param);

        builder.switch_to_block(exception_other_word_block);
        emit_exception_other_word_block_body(&mut builder, other_core_ptr, other_word_param, other_bd_param, other_status_param, exception_call_block);

        builder.switch_to_block(exception_entry_word_block);
        emit_exception_entry_word_block_body(&mut builder, entry_exc_core_ptr, entry_exc_status_param, exception_call_block);
        // None left sealed until every emit_exception_exit call site below
        // has been emitted — same reasoning as exit_block above.

        for &(word, block) in &instr_blocks {
            instrs[word as usize].block_id = Some(block.as_u32());
        }

        // Word offset -> head block, for target resolution in pass 2
        // (branch/jump taken targets, and Sequential's `word+1` for the
        // "did I land on a block, or does this edge exit" check). A target
        // that's *purely* a delay slot's own word (`is_slot_only`, never
        // independently reached as a genuine target) has no block here —
        // §6.1.4's "never a CFG edge into the slot's block" — but a word
        // that's *both* a slot and a real target (promoted by the walker)
        // does have one, same as any other head; `emit_target_edge`'s
        // `None` case relies on this.
        let block_for_word: std::collections::HashMap<WordOffset, Block> =
            instr_blocks.iter().copied().collect();

        // Pass 2: emit every head instruction's body and outgoing edges.
        // Nothing is sealed here — a block's predecessor set (especially a
        // backward branch target's) isn't complete until this whole pass
        // finishes.
        //
        // Compile-time-only running total of retired-but-unflushed
        // instructions, threaded into each iteration's `ctx` via
        // `cycles_pending` — see `EmitCtx::cycles_pending`'s doc comment.
        // Declared once, outside the loop, specifically because it must
        // survive across iterations (only a `cycles_flush` word resets it)
        // even though a fresh `ctx` is constructed every iteration.
        let mut cycles_pending: u32 = 0;
        for &(word, block) in &instr_blocks {
            builder.switch_to_block(block);

            let raw = instrs[word as usize].raw;
            // A branch-fallback successor (BC1 slot) arrives with correct live
            // pc/in_delay_slot from the fallback's interpreter run, same as the
            // entry word — both must trust those on an exception (see the field
            // doc). Set at ctx construction so every emit_exception_exit within
            // this word's emission (including a faulting slot instruction) picks
            // the right exception outer stage.
            let is_entry_point = instrs[word as usize].is_entry_point;
            let trust_live_pc_bd_on_exc = is_entry_point || instrs[word as usize].is_branch_fallback_successor;
            let mut ctx = EmitCtx { builder: &mut builder, module: &mut self.module, core_ptr, raw, word, bd: false, trust_live_pc_bd_on_exc, exit_block, exception_call_block, exception_entry_word_block, exception_other_word_block, cycles_pending: &mut cycles_pending };

            if is_entry_point && entry_body_blocks.contains_key(&word) {
                // This entry word's ordinary block is reached only by
                // internal in-region edges (emit_target_edge's None arm —
                // always a plain fallthrough/taken-branch edge, never a
                // delay-slot transfer) once skip_entry_preamble is set — the
                // real external-dispatch entry bypasses straight to this
                // entry's own body block instead (the dispatch head's
                // Switch above), never running any of this. So
                // core.pc/in_delay_slot are stale here for this specific
                // word and must be materialized before the preambles below
                // (which is why this is emitted first, still in this
                // entry's ordinary block, rather than after them —
                // deliberately NOT switching to the body block yet, so the
                // preambles still run exactly once, same as always;
                // switching early would make them run a second time inside
                // the body block too, wrongly subjecting the external-entry
                // bypass path to checks it must never pay): unconditionally
                // false for in_delay_slot (an internal edge into an entry
                // word is never a delay-slot landing) and vbase|word*4 for
                // pc. This is what lets exception_entry_word_block below
                // assume state is already correct with no runtime check of
                // its own — see its doc comment.
                //
                // Placed before the IP7/pending-interrupt preambles below,
                // not after, purely defensively — both preambles bail via
                // emit_bail, which recomputes core.pc itself from the
                // compile-time `word` constant regardless of order, and
                // never touches core.in_delay_slot at all. In principle a
                // stale in_delay_slot=true could leak through such a bail if
                // this internal-edge visit somehow followed an earlier,
                // same-call external foreign-slot visit to this same word —
                // but that specific sequence can't actually happen: the
                // pre-existing foreign-slot check on this entry word's own
                // fallthrough exit (below, `is_foreign_slot`) always exits
                // the function immediately on that path, so a genuine
                // foreign-slot arrival never continues on to any internal
                // edge within the same call. Kept first anyway since it
                // costs nothing extra either way and removes the ordering as
                // a thing to ever reason about again.
                let mem = MemFlagsData::trusted();
                let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
                let flag_off = ir::immediates::Offset32::new(core_offset_of_in_delay_slot());
                let pc = ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, pc_off);
                let vbase = ctx.builder.ins().band_imm_s(pc, !(PAGE_SIZE as i64 - 1));
                let fault_pc = ctx.builder.ins().iadd_imm_s(vbase, (word as i64) * 4);
                ctx.builder.ins().store(mem, fault_pc, ctx.core_ptr, pc_off);
                let false_flag = ctx.builder.ins().iconst(ir::types::I8, 0);
                ctx.builder.ins().store(mem, false_flag, ctx.core_ptr, flag_off);
            }

            // An entry word's ordinary-block preamble is unconditional
            // (every internal edge into it — a backward branch, say — still
            // needs it, skip_entry_preamble or not); the only thing that
            // changes is where the *semantics* get emitted: in that
            // ordinary block itself normally, or in its dedicated body
            // block when skip_entry_preamble is set, with the ordinary
            // block reduced to just "run the preamble, then jump there".
            //
            // A single preamble suffices now: the CP0 Compare timer fires on
            // the hptimer thread and raises IP7 through `hot.interrupts`
            // like every device line, so the pending-interrupt check *is*
            // the timer check — there's no per-instruction cp0_count advance
            // to mirror anymore (Count is virtual, materialized on read).
            emit_pending_interrupt_preamble(&mut ctx, exit_block, word);
            // Developer per-instruction hook (dt traceback + PC breakpoints),
            // right after the interrupt check — the same per-instruction point
            // the interpreter's step() does its trace/breakpoint work, so a
            // compiled region's instructions are visible in `dt` (tagged with
            // their real arrival class) and can hit breakpoints. Emitted in
            // whichever block the preamble runs in: for the entry word under
            // skip_entry_preamble that's entry_word_block (internal back-edges
            // ONLY — step()'s external dispatch bypasses this whole block, so
            // every call reaching here for word==entry_word is a back-edge, not
            // double-recording the external entry step() already logged).
            // is_branch_fallback_successor is reached here for both its
            // external arrival (straight from the fallback's own interpreter
            // run) and any internal back-edge — the foreign-slot runtime check
            // right below can't be resolved until after this call, so both
            // collapse to one FALLBACK_SUCCESSOR tag rather than paying for a
            // second runtime branch just to split a diagnostic tag further.
            // A fallback head's own dispatch is recorded separately, tagged
            // FallbackWord, by interp_dispatch_one itself when
            // emit_interp_fallback_head below actually calls into the
            // interpreter — skip this call entirely for one, or the word
            // gets double-recorded (once here as a misleading Jit/back-edge
            // tag before the interpreter ever runs it, once correctly after)
            // and traceback lookups keyed on pc find the wrong, earlier one.
            #[cfg(feature = "developer")]
            if !instrs[word as usize].is_fallback {
                let origin = if is_entry_point {
                    dev_trace_origin::JIT_ENTRY_BACK_EDGE
                } else if instrs[word as usize].is_branch_fallback_successor {
                    dev_trace_origin::FALLBACK_SUCCESSOR
                } else {
                    dev_trace_origin::JIT
                };
                emit_dev_trace_bp(&mut ctx, origin);
            }
            // Past the preamble: this instruction is actually going to
            // execute (the preamble didn't bail to the interpreter for this
            // word), matching step()'s per-instruction cycle count exactly —
            // see emit_account_for_cycles's doc comment for why this
            // can't go before the preamble (a bail here falls through to
            // the interpreter's step() re-entering at this same PC within
            // the *same* step() call, which already incremented once at its
            // own entry; incrementing here too would double-count).
            emit_account_for_cycles(&mut ctx, instrs, word);

            if is_entry_point {
                if let Some(&body_block) = entry_body_blocks.get(&word) {
                    ctx.builder.ins().jump(body_block, &[]);
                    ctx.builder.switch_to_block(body_block);
                }
            }

            if instrs[word as usize].is_fallback {
                // Interpreter-fallback head (analyzer kept an Excluded word in
                // the region). Takes priority over the branch/regjump/semantics
                // lookups below: an excluded word's raw bits may superficially
                // resemble a branch/regjump shape, but it has no native emitter
                // by definition — it runs through the interpreter. The int-check
                // preamble already ran above (compile_region's loop), so this is
                // the whole body. Note: entry-word special handling below is
                // skipped for a fallback head — a fallback calls the interpreter,
                // which handles any foreign-slot state itself.
                emit_interp_fallback_head(&mut ctx, exit_block, &block_for_word, instrs[word as usize].fallthrough_exit);
            } else if let Some(branch) = lookup_branch_or_jump(raw) {
                emit_branch_or_jump(&mut ctx, exit_block, &block_for_word, instrs, branch, fr_mode);
            } else if let Some(regjump) = lookup_regjump(raw) {
                emit_regjump(&mut ctx, instrs, regjump, fr_mode);
            } else {
                // jitv2_lockstep: bracket this straight-line (Sequential/CP1)
                // instruction — run the interpreter reference before, compare
                // the JIT's result after. Branch/jump/regjump aren't bracketed
                // (their inlined-slot / single-dispatch model mismatch needs
                // separate handling); fallback has its own semantics below.
                //
                // Skip the entry word: its core.pc/in_delay_slot come from
                // OUTSIDE (the interpreter dispatch that reached this region set
                // them, possibly as a foreign delay slot with in_delay_slot=true
                // — see the word==entry_word handling below). The lockstep
                // brackets materialize pc/bd from compile-time constants, which
                // would clobber that inherited state; and the entry instruction
                // was already the interpreter's own dispatch target, so it needs
                // no separate reference run.
                // Also skip a region-ending Sequential (fallthrough_exit set):
                // emit_lockstep_compare_seq materializes core.pc = word+1, but
                // the region-exit stub right after re-derives pc from live
                // core.pc's page base — at the 0xFFC boundary that word+1 write
                // lands pc on the *next* page, which the exit's vbase re-derive
                // then compounds by another page. The last instruction of a
                // region is verified when it's re-entered as part of the next
                // region anyway; skipping it here avoids the boundary hazard.
                // An entry word or branch-fallback successor (BC1 delay slot)
                // inherits core.pc/in_delay_slot/delay_slot_target from its
                // arrival, and resolves its final pc in the foreign-slot check
                // below (word+1 or delay_slot_target). It's lockstep-verified via
                // the `trust_live` path: emit_lockstep_step(true) preserves the
                // inherited state (LOCKSTEP_BD_LIVE sentinel), and the compare
                // (emit_lockstep_compare_live) runs inside BOTH foreign-slot arms
                // after pc is final. A plain straight-line head uses the ordinary
                // path (emit_lockstep_step(false) + emit_lockstep_compare_seq,
                // pc materialized to word+1). Region-ending words
                // (fallthrough_exit set) are still skipped — the exit stub's own
                // pc re-derivation collides with the compare's pc write (0xFFC
                // boundary); they're verified when re-entered as the next
                // region's head.
                // Step (run the interpreter reference, stash its expected
                // result) has nothing to do with whether this word's
                // fallthrough happens to leave the region — that's purely a
                // COMPARE-side concern (materializing pc=word+1 and, at the
                // 0xFFC page boundary, avoiding emit_bail's own re-derivation
                // double-applying the page-crossing offset — see
                // emit_lockstep_compare_seq's `region_ending` doc comment).
                // The two were previously tied to the same `fallthrough_exit
                // .is_none()` condition, which meant a region-ending
                // entry/branch-fallback-successor word never got its
                // interpreter reference run AT ALL — not skipped-but-safe,
                // just silently unverified (found live: badaddr_val's own
                // entry, the last word on its page, diverged from the
                // interpreter with zero lockstep coverage — the ADDIU that
                // computed `sp` was never cross-checked once).
                // No exclusion anymore: every word, region-ending or not,
                // gets a real interpreter reference + compare. The live-boot
                // hang this exclusion used to work around was the divergence
                // path only ever restoring `core.pc`, never `in_delay_slot`/
                // `delay_slot_target` alongside it — a stale slot flag left
                // over from the JIT's wrong run could make the interpreter
                // misinterpret a plain instruction as mid-delay-slot after a
                // break. `lockstep_compare` (mips_exec.rs) now restores all
                // three as one unit from `ls_before`/`ls_delay_target_before`,
                // so there's no remaining reason to special-case a
                // region-ending word out of lockstep coverage — the
                // page-boundary double-jump hazard on the *non-divergent*
                // continuing path is still handled below (see
                // `emit_lockstep_compare_seq`'s `region_ending` restore and
                // the `plain_block` arm's matching one).
                #[cfg(feature = "jitv2_lockstep")]
                let region_ending = instrs[word as usize].fallthrough_exit.is_some();
                #[cfg(feature = "jitv2_lockstep")]
                let ls_live = is_entry_point || instrs[word as usize].is_branch_fallback_successor;
                #[cfg(feature = "jitv2_lockstep")]
                let ls_bracket = !ls_live;
                #[cfg(feature = "jitv2_lockstep")]
                if ls_bracket { emit_lockstep_step(&mut ctx, false); }
                #[cfg(feature = "jitv2_lockstep")]
                if ls_live { emit_lockstep_step(&mut ctx, true); }

                // A real NOP (`raw == 0`, i.e. `SLL $0,$0,0`) has no
                // architectural effect at all — `emit_sll` would still emit
                // a real read/shift/sextend under this crate's opt_level=none
                // (see try_emit_fused_nop_slot's doc comment on why that
                // matters), all of it dead since emit_write_gpr already skips
                // the store for rd==0. Skip the dispatch entirely outside
                // jitv2_lockstep/developer, which still need the real
                // dispatch above (emit_lockstep_step's bracketing) to run for
                // per-instruction verification/tracing to mean anything.
                #[cfg(any(feature = "jitv2_lockstep", feature = "developer"))]
                let skip_nop_dispatch = false;
                #[cfg(not(any(feature = "jitv2_lockstep", feature = "developer")))]
                let skip_nop_dispatch = raw == 0;
                // LUI+ORI/ADDIU 32-bit-immediate fusion (see
                // try_emit_fused_lui's doc comment) — `extra_skip` is 1 when
                // it applied (fold word+1 into this write, fall through to
                // word+2 instead) or 0 otherwise (ordinary single-word LUI,
                // or any other opcode).
                let extra_skip = if raw & 0xFC00_0000 == (crate::mips_isa::OP_LUI << 26) {
                    try_emit_fused_lui(&mut ctx, instrs, word)
                } else {
                    0
                };
                if extra_skip == 0 && !skip_nop_dispatch {
                    if let Some(emit) = lookup_semantics(raw) {
                        emit(&mut ctx);
                    } else {
                        let emit = lookup_cp1_semantics(raw).expect("checked above");
                        // Per-instruction CU1 check, right before this CP1
                        // instruction's own semantics — mirrors the
                        // interpreter's own per-handler check exactly (see
                        // emit_cp1_cu1_guard's doc comment for why this
                        // can't be hoisted to entry/region granularity).
                        emit_cp1_cu1_guard(&mut ctx);
                        emit(&mut ctx, fr_mode);
                    }
                }

                // region_ending: emit_lockstep_compare_seq does nothing useful
                // (no materialize, no compare) — plain_fallthrough's Some(_)
                // arm below routes through emit_bail instead, whose shared
                // target block writes core.pc for real and runs the compare
                // right after, since that's the only point core.pc is
                // genuinely final for a region-ending word. See
                // emit_lockstep_compare_seq's own doc comment.
                #[cfg(feature = "jitv2_lockstep")]
                if ls_bracket && !region_ending { emit_lockstep_compare_seq(&mut ctx); }

                // `extra_skip == 1` (LUI fused with word+1's ORI/ADDIU):
                // word+1's own semantics were already folded into the
                // combined write above, so the edge that matters here is
                // word+1's fallthrough (whether word+2 continues into the
                // region), not the LUI's own — `word` alone never had a
                // fallthrough_exit computed against word+2 at all, since the
                // analyzer walked LUI as an ordinary single-word Sequential.
                // `extra_skip == 0`: ordinary single-word case, unchanged.
                let last_fused_word = word + extra_skip;
                let fallthrough_word = last_fused_word + 1;
                let plain_fallthrough = |ctx: &mut EmitCtx| {
                    match instrs[last_fused_word as usize].fallthrough_exit {
                        Some(_) => {
                            // Region ends here — mirrors handle_exec_complete's
                            // `pc += 4` (§3.3 "plain boundary").
                            emit_bail(ctx, exit_block, fallthrough_word);
                        }
                        None => {
                            let next_block = *block_for_word.get(&fallthrough_word)
                                .expect("fallthrough_exit is None -> analyzer guarantees the next word continues into the region");
                            ctx.builder.ins().jump(next_block, &[]);
                        }
                    }
                };

                // The entry word — and now also a *branch-fallback successor*
                // (the delay slot of a BC1 interpreter-fallback,
                // `is_branch_fallback_successor`) — can be reached with a
                // pending delay-slot transfer already armed (`in_delay_slot`/
                // `delay_slot_target` set): the entry word because the
                // interpreter's previous dispatch landed here as some *other*
                // branch's slot (the `entry_offset == 0` always-probe, or a
                // cross-page 0xFFC slot); the branch-fallback successor because
                // the BC1 fallback just ran the interpreter, which armed the
                // slot and left this word as its target's delay slot. Both need
                // the identical runtime check: after this word's semantics run,
                // if `core.in_delay_slot` is set, consume `core.delay_slot_target`
                // (exit there, clear the flag) — exactly mirroring
                // `handle_exec_complete` — instead of the compile-time-known
                // plain fallthrough, or the pending transfer is silently
                // discarded (found live: the IRIX PROM reset vector's `j
                // realstart` slot compiled standalone; and BC1-as-fallback,
                // this fix). The check is a runtime superset that's correct for
                // an ordinary (no-pending-slot) arrival too — `in_delay_slot`
                // false falls through normally — so one block serves both the
                // pending-transfer and plain arrival paths (the entry word never
                // needed two static versions here either).
                let needs_foreign_slot_check =
                    is_entry_point || instrs[word as usize].is_branch_fallback_successor;
                if needs_foreign_slot_check {
                    let mem = MemFlagsData::trusted();
                    let flag_off = ir::immediates::Offset32::new(core_offset_of_in_delay_slot());
                    let in_delay_slot = ctx.builder.ins().load(ir::types::I8, mem, ctx.core_ptr, flag_off);
                    let is_foreign_slot = ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, in_delay_slot, 0);

                    let foreign_slot_block = ctx.builder.create_block();
                    let plain_block = ctx.builder.create_block();
                    ctx.builder.ins().brif(is_foreign_slot, foreign_slot_block, &[], plain_block, &[]);

                    // Foreign-slot arm: consume the pending transfer (pc <-
                    // delay_slot_target, clear flag). Under lockstep (ls_live),
                    // materialize pc now so it's final, run the compare, then
                    // exit to the target.
                    ctx.builder.switch_to_block(foreign_slot_block);
                    ctx.builder.seal_block(foreign_slot_block);
                    let target_off = ir::immediates::Offset32::new(core_offset_of_delay_slot_target());
                    let target = ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, target_off);
                    let zero = ctx.builder.ins().iconst(ir::types::I8, 0);
                    ctx.builder.ins().store(mem, zero, ctx.core_ptr, flag_off);
                    #[cfg(feature = "jitv2_lockstep")]
                    if ls_live {
                        let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
                        ctx.builder.ins().store(mem, target, ctx.core_ptr, pc_off);
                        emit_lockstep_compare_live(&mut ctx);
                    }
                    emit_absolute_pc_exit(&mut ctx, target);

                    // Plain arm: no pending transfer. Under lockstep (ls_live),
                    // materialize pc = word+1 (the JIT didn't advance it) and
                    // compare before the ordinary fallthrough — except when
                    // region_ending, where materializing here would create the
                    // exact double-page-jump emit_lockstep_compare_seq's doc
                    // comment describes (plain_fallthrough below is about to
                    // route through emit_bail, whose shared target block
                    // writes core.pc for real and runs the compare right
                    // after — the only point core.pc is genuinely final for a
                    // region-ending word). Skip entirely in that case; nothing
                    // here would be more than a throwaway value emit_bail's
                    // own vbase re-derivation would then double-cross.
                    ctx.builder.switch_to_block(plain_block);
                    ctx.builder.seal_block(plain_block);
                    #[cfg(feature = "jitv2_lockstep")]
                    if ls_live && !region_ending {
                        let next_pc = emit_word_addr(&mut ctx, word + 1);
                        let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
                        ctx.builder.ins().store(mem, next_pc, ctx.core_ptr, pc_off);
                        emit_lockstep_compare_live(&mut ctx);
                    }
                    plain_fallthrough(&mut ctx);
                } else {
                    plain_fallthrough(&mut ctx);
                }
            }
        }

        // Pass 3: every block's predecessor set is now fully known.
        for &(_, block) in &instr_blocks {
            builder.seal_block(block);
        }
        // Each entry's body block has a single predecessor (that entry's own
        // ordinary-block preamble-then-jump), known from the moment it was
        // emitted above — sealable here alongside every other block, same
        // as the rest of pass 3.
        for &body_block in entry_body_blocks.values() {
            builder.seal_block(body_block);
        }
        builder.seal_block(exit_block);
        builder.seal_block(exception_call_block);
        builder.seal_block(exception_other_word_block);
        builder.seal_block(exception_entry_word_block);
        builder.finalize(self.module.target_config());

        // Anonymous, not named: this module never looks a compiled region
        // up by name again (get_finalized_function(func_id) is always
        // reached via the FuncId returned right here), so declare_function's
        // format!("region_{}", ...) string allocation + `names: HashMap`
        // insert (cranelift_module::ModuleDeclarations, `declare_function`'s
        // own `// TODO: Can we avoid allocating names so often?`) was pure
        // per-compile overhead with no benefit — declare_anonymous_function
        // skips both.
        let func_id = self.module
            .declare_anonymous_function(&self.ctx.func.signature)
            .expect("declare_anonymous_function");
        // Two genuinely different failure shapes land here, and callers
        // need to tell them apart (`last_compile_ran_out_of_memory`'s own
        // doc comment): `ModuleError::Compilation` means this module
        // emitted invalid IR — a real codegen bug, not a "this instruction
        // shape isn't supported yet" decline (those are all caught earlier,
        // before Cranelift is ever invoked) — the offset should be
        // sticky-denylisted, same as any other codegen gap.
        // `ModuleError::Allocation` means `ArenaMemoryProvider` ran out of
        // its reservation — not a bug in this specific region at all, and
        // denylisting it would be wrong (it's perfectly compilable, just
        // not right now); the caller needs to flush and retry instead.
        // `debug_assert!` alone is silent in release builds (debug_assertions
        // off) and was hiding exactly this in an earlier version: the
        // verifier's own error-formatting path (building the
        // mismatched-successor debug string) became the actual hot path
        // once real per-compile allocation failures started happening,
        // invisibly, in a release run. Surfaced per build mode instead:
        // full error text under `developer` (the diagnostics build),
        // nothing under `lightning` (perf-critical, no per-compile string
        // formatting ever), and a minimal one-line notice otherwise —
        // silence here is what let a real problem masquerade as a
        // performance cliff.
        // define_function's own allocate() call (through the module's own,
        // opaque provider handle) reports its exact range through the
        // mailbox both handles share — read via seal_handle right after,
        // since seal_handle never allocates anything itself and so never
        // clobbers it. This is how finalize_batch_nonforced later learns
        // which byte range a given FuncId's machine code landed in, without
        // needing get_finalized_function pre-finalize (not callable — see
        // that method's own doc comment) or any change to what
        // define_function itself does. Race-free under real multi-worker
        // concurrency because the mailbox is per-Codegen (not pool-wide):
        // only this Codegen's own module_handle ever writes to it — unlike
        // an earlier version of this code, which bracketed
        // seal_handle.position() (the shared arena's own live, pool-wide
        // bump cursor) before/after this call instead, which raced with
        // OTHER workers' concurrent allocate() calls moving that same
        // cursor in between — confirmed live as the cause of a real
        // stuck-forever compile-pool bug under genuine multi-thread
        // contention (workers' func_ranges ending up with wrong/overlapping
        // recorded ranges, so try_seal_ready's sealed results could never
        // match back to the FuncId that actually owned them).
        if let Err(e) = self.module.define_function(func_id, &mut self.ctx) {
            let is_oom = matches!(e, cranelift_module::ModuleError::Allocation { .. });
            self.last_compile_ran_out_of_memory = is_oom;
            #[cfg(feature = "developer")]
            { self.last_decline_was_verifier_error = !is_oom; }
            #[cfg(feature = "developer")]
            eprintln!("jitv2: compile_region failed: {e}");
            #[cfg(not(any(feature = "developer", feature = "lightning")))]
            eprintln!("jitv2: codegen rejected a compiled region (run a developer build for details)");
            return None;
        }
        self.last_compile_ran_out_of_memory = false;
        let range = self.seal_handle.take_last_allocation()
            .expect("define_function must have allocated real memory for a successful compile");
        self.func_ranges.insert(func_id, range);
        // Reserve this range's seal-queue slot right now — see
        // push_placeholder's own doc comment for why this can't wait until
        // finalize time. finalize_batch/finalize_batch_nonforced fill in
        // the real PublishInfo (patch_pending_publish) once they have one.
        self.seal_handle.push_placeholder(range.0, range.1, page);
        // Read code size before clearing context (compiled_code is cleared
        // by clear_context) — same pattern as rex3_jit/compiler.rs and
        // jit/compiler.rs. Captured into a field rather than returned
        // directly, to keep this function's `Option<JitFn>` return type
        // stable for its several other callers (equiv_test, lockstep, …).
        #[cfg(feature = "developer")]
        {
            self.last_code_size = self.ctx.compiled_code()
                .map(|cc| cc.code_buffer().len() as u32)
                .unwrap_or(0);
        }
        self.module.clear_context(&mut self.ctx);
        self.func_id_counter += 1;
        // Heartbeat: cranelift-jit exposes no arena-size/mmap-count API of
        // its own (`cranelift_jit::Memory` is pub(crate) — see
        // `CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES`'s doc comment), so function_count
        // is the only signal we have for "how far did we get before an
        // out-of-address-space abort." Printed unconditionally (not just
        // under `developer`) since this is exactly what you want visible in
        // a crash log when CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES/mega_flush turns
        // out not to be catching something — a crash with no heartbeat line
        // near the threshold count means the flush path itself isn't firing;
        // a crash well past several heartbeats with flushes interleaved
        // (see flush_from_cpu_thread/flush_from_jit_thread's own eprintln!s)
        // means the threshold is set too high for whatever's compiling.
        if self.func_id_counter % 10_000 == 0 {
            eprintln!("jitv2: {} functions compiled into this Codegen's arena since last reset", self.func_id_counter);
        }
        Some(func_id)
    }

    /// Compile a region and immediately finalize+return a callable `JitFn` —
    /// today's synchronous contract, used by every caller that needs the
    /// result ready to call right away (the equivalence-test harness,
    /// `jitv2_lockstep`, and `jitv2_inline_compile`'s "run it immediately"
    /// path — see that path's own doc comment in `mips_exec.rs` for why it
    /// specifically cannot tolerate a deferred pointer). Thin wrapper around
    /// `compile_region_uncommitted` + an immediate one-function
    /// `finalize_batch` call.
    /// §13.4 single-entry compatibility wrapper: same signature every
    /// existing caller (equivalence tests, `jitv2_verify`, this module's own
    /// unit tests) already uses — takes an `instrs` buffer produced by a
    /// plain `Analyzer::walk`/`walk_bounded` call (which knows nothing about
    /// `is_entry_point` or `has_fpu`, both new §13 concepts) and adapts it
    /// to `compile_region_uncommitted`'s real multi-entry-capable signature:
    /// marks `entry_word` as this region's one entry point (mirroring what
    /// `Analyzer::walk_multi_entry` would have done for a real multi-entry
    /// caller) and computes `has_fpu` the same way `walk_multi_entry` does,
    /// since a plain single-entry `walk` never had a reason to. `j2wp`
    /// production code (`comp.rs`'s `handle_request`/`handle_request_deferred`)
    /// does NOT go through this — it calls `walk_multi_entry` directly and
    /// threads its `has_fpu` straight into `compile_region_uncommitted`, no
    /// re-derivation needed. The default (`not(j2wp)`) path's `comp.rs`
    /// still uses this directly as its real, single-entry production
    /// compile call — `page` is passed as null there too (this synchronous
    /// path finalizes immediately below, so the seal-queue placeholder never
    /// dangles long enough for `j2 seal-queue`'s page stamp to matter).
    pub fn compile_region(
        &mut self,
        instrs: &mut [CompiledInstr; ENTRIES_PER_PAGE],
        entry_word: WordOffset,
        compiled_for_fr1: bool,
        skip_entry_preamble: bool,
    ) -> Option<crate::jitv2::JitFn> {
        instrs[entry_word as usize].is_entry_point = true;
        let has_fpu = instrs_linear(instrs).any(|i| crate::jitv2::analyzer::is_fpu_instruction(i.raw));
        // No real PhysicalCodePage available at this API's call sites
        // (test-only — see this function's own doc comment) — null is fine,
        // same as any other diagnostic-only field with nothing real to
        // report; this compile always finalizes immediately below anyway,
        // so its seal-queue placeholder is never left dangling long enough
        // for the page stamp to matter.
        let func_id = self.compile_region_uncommitted(instrs, compiled_for_fr1, skip_entry_preamble, has_fpu, std::ptr::null_mut())?;
        self.finalize_batch(&[func_id]).into_iter().next()
    }

    /// Finalize every `FuncId` in `ids` in one `finalize_definitions()` call
    /// and return their now-callable `JitFn` pointers, in the same order —
    /// the batched counterpart to `compile_region`'s immediate one-at-a-time
    /// finalize. Callers accumulate `FuncId`s from `compile_region_uncommitted`
    /// across several compiles (deferring the finalize so they pack into the
    /// same host-page segment instead of each getting its own — see
    /// `paged_memory`'s module doc comment) and call this once, when
    /// `provider_crossed_page()` says the next allocation would spill onto a
    /// new page or the caller otherwise decides to stop batching (queue
    /// drained, etc). `ids` empty is a valid no-op (returns an empty `Vec`)
    /// rather than a caller-side special case.
    pub fn finalize_batch(&mut self, ids: &[cranelift_module::FuncId]) -> Vec<crate::jitv2::JitFn> {
        if ids.is_empty() {
            return Vec::new();
        }
        // `.ok()` swallows a real error here the same way compile_region's
        // old inline call did (`.ok()?`) — finalize_definitions() failing
        // for a batch that already passed define_function successfully for
        // every member is not a documented/expected outcome for this
        // provider (it can only fail via the JITMemoryProvider::finalize
        // trait method, which PagedArenaMemoryProvider's impl never errors
        // from), so there's nothing more specific to report; every id in
        // `ids` simply won't be callable and the caller's batch is lost —
        // acceptable since compiled-but-never-published functions are inert
        // (never reachable from any page's entry_table).
        if self.module.finalize_definitions().is_err() {
            return Vec::new();
        }
        ids.iter()
            .map(|&id| {
                let (start, end) = self.func_ranges.remove(&id)
                    .expect("finalize_batch: id has no reserved seal-queue range — compile_region_uncommitted must run first");
                let code_ptr = self.module.get_finalized_function(id);
                let jit_fn = unsafe { std::mem::transmute::<*const u8, crate::jitv2::JitFn>(code_ptr) };
                // Forced: this caller's whole contract is "give me a
                // callable pointer right now" (compile_region/inline mode),
                // so the underlying page(s) must actually be sealed to RX
                // before returning, not just have their placeholder patched
                // — see patch_pending_publish's own doc comment. No real
                // page/offset/gen_snap to give (this path hands the raw
                // JitFn straight back to its caller instead of going
                // through page.publish()) — a null/zeroed PublishInfo is
                // fine here since nothing ever reads it back out for this
                // entry (try_seal_ready's return value is discarded below).
                let publish = crate::jitv2::paged_memory::PublishInfo {
                    jit_fn: Some(jit_fn),
                    ..crate::jitv2::paged_memory::PublishInfo::blank()
                };
                self.seal_handle.patch_pending_publish(start, end, publish, true);
                jit_fn
            })
            .collect()
    }

    /// Non-forced counterpart to `finalize_batch`, for the async worker:
    /// finalizes this one `FuncId` (patches relocations — real memory
    /// writes, must happen before anything seals) and, once
    /// `finalize_definitions()` returns, resolves its real `JitFn` and
    /// patches the complete `PublishInfo` into the seal-queue slot
    /// `compile_region_uncommitted` already reserved for it
    /// (`push_placeholder`) — see `PagedArenaMemoryProvider::patch_pending_publish`'s
    /// own doc comment. `publish` is everything except `jit_fn` (the
    /// caller's own page/offset/gen_snap/instr_count/code_size); `jit_fn`
    /// is filled in here, immediately after it becomes valid to read.
    ///
    /// Returns whatever this call's own seal attempt reported as newly
    /// sealed — almost always just this one entry's own `PublishInfo` (the
    /// common case, nothing blocking it), occasionally more (this entry
    /// happened to complete a longer contiguous run that had been waiting
    /// on it), or empty if this entry itself is now the one blocked behind
    /// an earlier gap (retry later: another non-forced call, or the
    /// seal-quiesce barrier's forced sweep) OR `finalize_definitions()`
    /// itself failed outright — check `last_finalize_failed()` immediately
    /// after an empty return to tell the two apart; only the gap-blocked
    /// case will ever resolve on its own.
    pub fn finalize_batch_nonforced(&mut self, id: cranelift_module::FuncId, publish: crate::jitv2::paged_memory::PublishInfo) -> Vec<crate::jitv2::paged_memory::PublishInfo> {
        self.last_finalize_failed = false;
        let (start, end) = *self.func_ranges.get(&id).expect("finalize_batch_nonforced: id has no reserved seal-queue range — compile_region_uncommitted must run first");
        if self.module.finalize_definitions().is_err() {
            self.last_finalize_failed = true;
            self.func_ranges.remove(&id);
            return Vec::new();
        }
        self.func_ranges.remove(&id);
        let code_ptr = self.module.get_finalized_function(id);
        let publish = crate::jitv2::paged_memory::PublishInfo {
            jit_fn: Some(unsafe { std::mem::transmute::<*const u8, crate::jitv2::JitFn>(code_ptr) }),
            ..publish
        };
        self.seal_handle.patch_pending_publish(start, end, publish, false)
    }

    /// Idle-timeout/seal-quiesce sweep: force-seal whatever is still queued
    /// (any worker's not-yet-sealed ranges from a prior
    /// `finalize_batch_nonforced` call) without waiting for a further
    /// `finalize_definitions()` call to trigger it. Only ever called from
    /// under the seal-quiesce barrier (`CompileQueue::run_seal_leader` — see
    /// `SealBarrierState`'s own doc comment for why forcing needs that
    /// guarantee), so it's safe to seal past whatever's currently queued —
    /// nothing else can arrive with a lower `start` while every worker is
    /// parked. Returns every `PublishInfo` this sweep newly sealed,
    /// regardless of which `Codegen` originally pushed each one — the
    /// caller just publishes all of them; see `paged_memory::PublishInfo`'s
    /// own doc comment for why no `FuncId`/`func_ranges` lookup is needed
    /// here anymore.
    pub fn force_seal_pending(&mut self) -> Vec<crate::jitv2::paged_memory::PublishInfo> {
        self.seal_handle.try_seal_ready_forced()
    }

    /// Non-forced counterpart to `force_seal_pending` — re-attempts sealing
    /// whatever the shared arena's seal queue currently has queued, without
    /// pushing anything new, and without forcing past a still-open page.
    /// See `paged_memory::PagedArenaMemoryProvider::try_seal_ready`'s own
    /// doc comment.
    pub fn try_seal_ready(&mut self) -> Vec<crate::jitv2::paged_memory::PublishInfo> {
        self.seal_handle.try_seal_ready()
    }
}

/// Emit the pending-interrupt preamble every compiled unit needs before its
/// own semantics (§3.2 "check 1" — mirror the interpreter's own pending
/// check, verbatim, at every unit boundary): an atomic load of
/// `core.hot.interrupts`, tested against zero, bail if nonzero. Same field, same
/// predicate as `step()`'s own `let pending = self.core.hot.interrupts.load(...)`
/// — this preamble does *not* replicate step()'s enabled/masked delivery
/// logic (whether the interrupt actually gets delivered, soft-reset
/// handling, etc.); it only decides whether to bail. On bail, the
/// interpreter's `step()` re-enters at the top and does that full logic as
/// the single authoritative implementation (§7a).
///
/// On bail: jumps to `exit_block` (the function's shared exit-to-interpreter
/// block, `BlockSkeleton::exit_block`) via [`emit_bail`], which materializes
/// `core.pc = vbase | (word_offset * 4)` and returns `EXEC_COMPLETE`.
///
/// Must be called with `builder` positioned in the block that will hold this
/// instruction's preamble; leaves `builder` positioned in a new, sealed
/// continuation block (the "nothing pending" path) where the next preamble
/// or the instruction's real semantics should be emitted next.
fn emit_pending_interrupt_preamble(ctx: &mut EmitCtx, exit_block: Block, word_offset: WordOffset) {
    let mem = MemFlagsData::trusted();

    // Genuine atomic load, not a plain `load`: on an aligned, naturally-atomic
    // width (u64, 8-byte aligned) these compile to the identical single load
    // instruction on x86_64/aarch64 at the hardware level, but a plain
    // Cranelift `load` carries none of Rust/LLVM-style atomics' *compiler*
    // reordering guarantees — under opt_level=none (this codegen's default)
    // that distinction was moot (no reordering optimization passes run to
    // exploit it), but `Codegen::set_opt_level_speed`'s `speed` mode runs
    // real optimization passes that are legally free to hoist, sink, or
    // otherwise reorder a plain load across other memory operations in this
    // region — which a pending-interrupt check must never allow (it needs to
    // observe interrupts.store()'s effect promptly, not some stale
    // hoisted-to-entry snapshot). `atomic_load` is specified as sequentially
    // consistent (stronger than the interpreter's own Ordering::Relaxed
    // load, which is a safe direction to differ in — this instruction still
    // observes everything a Relaxed load would, just with the reordering
    // freedom taken away). `atomic_load`'s format takes a bare pointer, not
    // an (Offset32) field access like the plain `load` this replaced — the
    // interrupts field's offset has to be folded into the pointer explicitly
    // first via `iadd_imm_s`.
    let interrupts_ptr = ctx.builder.ins().iadd_imm_s(ctx.core_ptr, core_offset_of_interrupts() as i64);
    let pending = ctx.builder.ins().atomic_load(ir::types::I64, mem, interrupts_ptr);
    let zero = ctx.builder.ins().iconst(ir::types::I64, 0);
    let has_pending = ctx.builder.ins().icmp(IntCC::NotEqual, pending, zero);

    let bail_block = ctx.builder.create_block();
    let continue_block = ctx.builder.create_block();
    ctx.builder.ins().brif(has_pending, bail_block, &[], continue_block, &[]);

    // Cold: a pending interrupt on any given instruction dispatch is the
    // rare case (interrupts are relatively infrequent compared to the
    // instruction stream), same reasoning as every other bail/exception
    // block in this module.
    ctx.builder.switch_to_block(bail_block);
    ctx.builder.set_cold_block(bail_block);
    ctx.builder.seal_block(bail_block);
    emit_bail(ctx, exit_block, word_offset);

    ctx.builder.switch_to_block(continue_block);
    ctx.builder.seal_block(continue_block);
}

/// Region-wide FR-mode guard: emitted once, in `entry_block`, only when the
/// region being compiled contains at least one CP1 instruction — legitimate
/// as a region-wide check because `STATUS_FR` cannot change mid-region (the
/// only instructions that touch CP0.Status, MTC0/ERET, are `Excluded` and
/// end the region on contact, §4.4), so the very first entry checks it once
/// on behalf of everything the rest of the region will FPR-access-emit.
///
/// **Does NOT check CU1** — see `emit_cp1_cu1_guard`'s own doc comment for
/// why that must be a per-CP1-instruction check, not a region/entry-wide
/// one (§13.4 correction: an earlier version of this function checked both
/// here, which was wrong on two counts — CU1-clear is a real per-instruction
/// exception, and "this function" is no longer "one region, one entry" once
/// a page's many entries can share it, so a region-wide entry check fired
/// for entries whose own reachable code never touches CP1 at all).
///
/// FR mismatch (believed unreachable for any real compiled guest binary —
/// see the `FrMode` doc comment): NOT an architectural exception at all —
/// it means this *entire compiled artifact* was built assuming a
/// `STATUS_FR` value that's no longer live, so every FPR-access emitter in
/// it uses the wrong register-packing scheme. Continuing (even via the
/// interpreter) isn't enough — this exact function must never be
/// dispatched again for this entry. Kills the entry (`kill_entry_fn` — see
/// its own doc comment) so the JIT gate stops re-selecting it, then forces
/// one real interpreter dispatch (`interp_fallback_fn`) so this instruction
/// still makes progress today; the *next* visit to this PC gets a genuine
/// fresh compile against whatever FR mode is live then.
///
/// `compiled_for_fr1` is the FR mode this whole region was compiled against
/// (`FrMode::Fr1` if `STATUS_FR` was set when compilation started); the
/// guard checks whether the live bit still agrees.
///
/// §13.4: `entry_offset_val` (runtime I32, `core.pc`'s live entry offset —
/// see the dispatch head this runs just before, in `entry_block`) replaces
/// the old compile-time `entry_word: WordOffset` — see `emit_kill_entry`'s
/// own doc comment for why a compile-time constant no longer exists at this
/// point once a function can serve more than one entry.
fn emit_fr_mode_guard(ctx: &mut EmitCtx, entry_offset_val: Value, compiled_for_fr1: bool) {
    let mem = MemFlagsData::trusted();
    let status_off = ir::immediates::Offset32::new(core_offset_of_cp0_status());
    let status = ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, status_off);

    // CU1 gates everything on real hardware — a CPU never even looks at FR
    // mode if CU1 is clear (this function's own doc comment). Since CU1 is
    // now checked per-instruction (emit_cp1_cu1_guard), not here, THIS check
    // must not fire at all when CU1 is clear: the real fault (if the
    // dispatched entry's control flow ever reaches a CP1 instruction) has to
    // come from that per-instruction check instead, at the actual
    // instruction, not from this entry-time check jumping the queue and
    // reporting a wrong-shaped fault (found live: a region compiled for
    // FR1, dispatched with CU1 AND FR both clear, hit this FR-mismatch arm
    // before the real CU1 fault ever got a chance to fire — this exact
    // ordering bug).
    let cu1_bit = ctx.builder.ins().band_imm_s(status, crate::mips_core::STATUS_CU1 as i64);
    let cu1_set = ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, cu1_bit, 0);

    let fr_bit = ctx.builder.ins().band_imm_s(status, crate::mips_core::STATUS_FR as i64);
    let fr_mismatch_if_cu1_set = if compiled_for_fr1 {
        ctx.builder.ins().icmp_imm_s(IntCC::Equal, fr_bit, 0)
    } else {
        ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, fr_bit, 0)
    };
    let fr_mismatch = ctx.builder.ins().band(cu1_set, fr_mismatch_if_cu1_set);

    let bail_block = ctx.builder.create_block();
    let continue_block = ctx.builder.create_block();
    ctx.builder.ins().brif(fr_mismatch, bail_block, &[], continue_block, &[]);

    // Cold: per this function's own doc comment, an FR mismatch is believed
    // unreachable for any real compiled guest binary — the coldest of every
    // cold block this module marks.
    ctx.builder.switch_to_block(bail_block);
    ctx.builder.set_cold_block(bail_block);
    ctx.builder.seal_block(bail_block);
    // I32 for kill_entry_fn's ABI (narrow extern "C" params aren't reliably
    // zero-extended — see kill_entry_fn's doc comment); I64 for
    // emit_interp_fallback_exit's word-offset arithmetic against vbase.
    let entry_offset_i32 = ctx.builder.ins().ireduce(ir::types::I32, entry_offset_val);
    emit_kill_entry(ctx, entry_offset_i32);
    // Not a plain emit_bail: see emit_interp_fallback_exit's doc comment for
    // why a bail here can't force the interpreter's real semantics to
    // actually run (also true here: without this, the killed entry's own
    // interpreter re-dispatch would depend on exec_decoded's gate falling
    // through correctly, an extra dependency this makes unnecessary). No PC
    // materialization needed — see emit_interp_fallback_exit's own doc
    // comment: live core.pc is already correct here, at the top of
    // entry_block.
    emit_interp_fallback_exit(ctx);

    ctx.builder.switch_to_block(continue_block);
    ctx.builder.seal_block(continue_block);
}

/// Per-CP1-instruction CU1 (coprocessor-1-usable) check — emitted right
/// before EVERY CP1 instruction's own semantics, mirroring the interpreter
/// exactly: each CP1 handler (`exec_cfc1`, `exec_add_s`, ...) independently
/// checks `if (cp0_status & STATUS_CU1) == 0 { return cpu_unusable(1) }` at
/// its own top, not once per function/region/entry.
///
/// §13.4 correction: an earlier version of this codegen checked CU1 once,
/// region-wide, at function entry (`emit_fr_mode_guard`'s predecessor,
/// folded both checks together) — wrong on real-hardware semantics (CU1 is
/// a *per-instruction* fault: an entry whose reachable code never executes
/// a CP1 instruction on a given path must never fault at all, and a path
/// that reaches CP1 only after some branches must fault exactly where the
/// real CP1 instruction is, not at function/entry start) and, once one
/// function serves many entries (§13), wrong on top of that: entries with
/// no CP1 in their own reachable set were paying (and could spuriously
/// fault from) a check that had nothing to do with them, since `has_fpu`
/// is a whole-region property, not a per-entry one.
///
/// Uses `ctx` as-is (the calling instruction's own `word`/`bd`/
/// `trust_live_pc_bd_on_exc`) so `emit_exception_exit` (via
/// `emit_materialize_cpu_unusable`) routes and materializes EPC exactly
/// like any other per-instruction fault this module detects (a memory
/// access fault, an overflow trap, ...) — no special-casing needed here
/// beyond calling it at the right point in pass 2, immediately before the
/// CP1 instruction's real emitter runs.
fn emit_cp1_cu1_guard(ctx: &mut EmitCtx) {
    let mem = MemFlagsData::trusted();
    let status_off = ir::immediates::Offset32::new(core_offset_of_cp0_status());
    let status = ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, status_off);
    let cu1_clear = ctx.builder.ins().band_imm_s(status, crate::mips_core::STATUS_CU1 as i64);
    let cu1_bad = ctx.builder.ins().icmp_imm_s(IntCC::Equal, cu1_clear, 0);

    let cu1_block = ctx.builder.create_block();
    let continue_block = ctx.builder.create_block();
    ctx.builder.ins().brif(cu1_bad, cu1_block, &[], continue_block, &[]);

    // Cold: CU1 disabled is the rare case for any region that actually
    // contains FPU instructions (real guest code enables it once, early,
    // and leaves it on).
    ctx.builder.switch_to_block(cu1_block);
    ctx.builder.set_cold_block(cu1_block);
    ctx.builder.seal_block(cu1_block);
    emit_materialize_cpu_unusable(ctx);

    ctx.builder.switch_to_block(continue_block);
    ctx.builder.seal_block(continue_block);
}

/// Materialize a real `EXC_CPU` (Coprocessor Unusable, CP1) exception and
/// deliver it via `handle_exception_fn` — the `emit_cp1_cu1_guard`
/// counterpart to `MipsExecutor::cpu_unusable(1)`. `Cause.CE` needs a direct
/// store first: `deliver_exception` (what `handle_exception_fn` calls) only
/// ever touches `Cause.ExcCode`, never the CE field — mirrors
/// `cpu_unusable`'s own `cp0_cause = (cause & !CAUSE_CE_MASK) | ((ce&3) <<
/// CAUSE_CE_SHIFT)` exactly, with `ce` hardcoded to `1` (CP1) since this
/// guard only ever exists for a region containing CP1 instructions. The
/// exception code itself (`EXC_CPU`) is a compile-time constant — no
/// interpreter re-dispatch needed to determine what happened, unlike a
/// memory-access fault's status (`emit_check_mem_exc`), which only the bus
/// device knows until the access actually runs. Terminates the current
/// block (delegates to `emit_exception_exit`, itself a terminator).
fn emit_materialize_cpu_unusable(ctx: &mut EmitCtx) {
    let mem = MemFlagsData::trusted();
    let cause_off = ir::immediates::Offset32::new(core_offset_of_cp0_cause());
    let cause = ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, cause_off);
    let ce_cleared = ctx.builder.ins().band_imm_s(cause, !(crate::mips_core::CAUSE_CE_MASK as i64));
    let ce_bit = 1i64 << crate::mips_core::CAUSE_CE_SHIFT; // CP1
    let new_cause = ctx.builder.ins().bor_imm_s(ce_cleared, ce_bit);
    ctx.builder.ins().store(mem, new_cause, ctx.core_ptr, cause_off);

    let status = ctx.builder.ins().iconst(
        ir::types::I32,
        crate::mips_exec::exec_exception(crate::mips_exec::EXC_CPU) as i64,
    );
    emit_exception_exit(ctx, status);
}

/// Call `core.kill_entry_fn(jit_ctx, entry_offset)` — see
/// `MipsCore::kill_entry_fn`'s doc comment. Not a terminator; caller
/// continues on (typically into `emit_interp_fallback_exit` right after).
///
/// §13.4: `entry_offset_val` is a runtime `Value` (I32), not a compile-time
/// constant — the FPU/FR guard (this function's only caller) runs once in
/// `entry_block`, *before* the dispatch head has resolved which of this
/// compile's possibly-several entry points the live call is actually for,
/// so there is no single compile-time offset to bake in anymore. The caller
/// passes the same `live_entry_offset` value the dispatch head itself
/// computes from `core.pc`, truncated to I32 (see its own call site).
fn emit_kill_entry(ctx: &mut EmitCtx, entry_offset_val: Value) {
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();

    let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
    let jit_ctx = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, jit_ctx_off);

    let fn_off = ir::immediates::Offset32::new(core_offset_of_kill_entry_fn());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);

    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty)); // jit_ctx
    sig.params.push(AbiParam::new(ir::types::I32)); // entry_offset
    let sig_ref = ctx.builder.import_signature(sig);

    ctx.builder.ins().call_indirect(sig_ref, callee, &[jit_ctx, entry_offset_val]);
}

/// Jump to the function's shared exit-to-interpreter block (`BlockSkeleton::
/// exit_block`) with `(core_ptr, word_offset)` as block arguments, instead of
/// emitting a fresh copy of the materialize-PC-and-return sequence at every
/// bail site. Use this from every clean exit — preambles here, and
/// `fallthrough_exit`/`taken_exit` stubs (§3.3) once those land. Does not
/// seal or switch blocks; call from within the block that decided to bail,
/// as its terminator.
fn emit_bail(ctx: &mut EmitCtx, exit_block: Block, word_offset: WordOffset) {
    let word_offset_val = ctx.builder.ins().iconst(ir::types::I64, word_offset as i64);
    ctx.builder.ins().jump(exit_block, &[ir::BlockArg::Value(ctx.core_ptr), ir::BlockArg::Value(word_offset_val)]);
}

/// Force real forward progress through `core.interp_fallback_fn` instead of
/// a plain bail, for a condition compiled code detected but can't itself
/// resolve (`emit_fr_mode_guard`'s FR mismatch — the only caller).
/// A plain `emit_bail` just re-sets `core.pc` back to this same instruction
/// and returns `EXEC_COMPLETE`, which `exec_decoded`'s caller can't tell
/// apart from a real retirement — if this word is still published/hot, the
/// very next dispatch calls this identical compiled function again, which
/// bails again, forever, without the interpreter's real semantics (the
/// `cpu_unusable` exception, in the FPU guard's case) ever actually running
/// (found live: `cfc1` with CU1 clear spun in place indefinitely). Instead,
/// call `interp_fallback_fn` (which fetches/decodes/dispatches whatever's
/// at live `core.pc`, same as any other interpreter step) and return its
/// status directly — whatever the interpreter's real handler actually did
/// (retired, faulted, retried) is the true result of this compiled unit's
/// dispatch, not a synthetic "nothing happened, try again" signal.
/// Terminates the current block.
///
/// §13.4: does **not** materialize `core.pc` before calling — this
/// function's only caller (the FPU/FR guard) runs at the very top of
/// `entry_block`, before this compiled function has touched `core.pc` at
/// all, so live `core.pc` is already exactly this dispatch's real entry
/// address (the same value the dispatch head's own `live_entry_offset` was
/// just derived from a few instructions earlier) — writing it again would
/// recompute a value already sitting there unchanged. If this function ever
/// gains a second caller from somewhere `core.pc` might be stale, that
/// caller is responsible for materializing it first.
fn emit_interp_fallback_exit(ctx: &mut EmitCtx) {
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();

    let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
    let jit_ctx = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, jit_ctx_off);

    let fn_off = ir::immediates::Offset32::new(core_offset_of_interp_fallback_fn());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);

    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty)); // jit_ctx
    sig.returns.push(AbiParam::new(ir::types::I32)); // ExecStatus
    let sig_ref = ctx.builder.import_signature(sig);

    let call = ctx.builder.ins().call_indirect(sig_ref, callee, &[jit_ctx]);
    let status = ctx.builder.inst_results(call)[0];
    ctx.builder.ins().return_(&[status]);
}

/// Emit an interpreter-fallback **head** (`CompiledInstr::is_fallback`): an
/// analyzer-`Excluded` instruction the walker kept in the region instead of
/// ending it (see `analyzer::visit`'s `Classify::Excluded` arm). Runs after
/// this word's normal per-instruction preamble (int-check + cycle increment,
/// emitted by `compile_region`'s loop like every head), so the int-check has
/// already had its chance to bail — that is what gives a fallback the
/// per-instruction interrupt check the old entry instruction relied on
/// `step()` for (`fallback_performs_int_check_before_running`).
///
/// Shape:
///   1. materialize `core.pc = word*4 | vbase` (this instruction's own
///      address) so `interp_fallback_fn` (`interp_dispatch_one`) fetches and
///      dispatches *this* excluded instruction through the real interpreter
///      handler;
///   2. call it, take its `ExecStatus`;
///   3. if status != EXEC_COMPLETE, `return status` — the handler already
///      delivered whatever exception/fault it raised (SYSCALL/BREAK/COP2/
///      CACHE-fault, etc.); falling through would run the successor after an
///      exception (`fallback_exception_status_short_circuits_successor`);
///   4. else check `core.pc == word*4+4 | vbase`. A fallback can retire
///      EXEC_COMPLETE yet *relocate* PC (ERET -> EPC, taken BC1) — for those
///      the successor block's compile-time PC assumption is wrong, so
///      `return EXEC_COMPLETE` and let the interpreter re-dispatch at the new
///      PC (`fallback_that_moves_pc_does_not_run_successor`). This check is NOT
///      inherited from entry semantics (entry only checks in_delay_slot and
///      never runs a prior instruction that could move PC);
///   5. else `core.pc`/`core.in_delay_slot` are exactly what a normal
///      interpreter step leaves (pc = this+4, in_delay_slot false — see
///      `handle_exec_complete`), i.e. the successor is already in the
///      identical state to a fresh external entry ("entry-like successor").
///      Continue: jump to the successor's block if it's in-region, or
///      `return EXEC_COMPLETE` if the fallback's fallthrough exits the region
///      (pc already correct, no `emit_bail` recompute needed).
///
/// Terminates the current block on every path.
fn emit_interp_fallback_head(
    ctx: &mut EmitCtx,
    exit_block: Block,
    block_for_word: &std::collections::HashMap<WordOffset, Block>,
    fallthrough_exit: Option<crate::jitv2::analyzer::StopReason>,
) {
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();
    let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
    let word = ctx.word;

    // (1) core.pc = this instruction's own address. `own_pc` is captured here,
    // BEFORE the fallback call, and reused for the step-(4) advance check —
    // deriving the expected successor address from the *post-call* core.pc
    // (via a fresh emit_word_addr) would be wrong: the fallback may have moved
    // core.pc to a different page (ERET -> EPC, a taken branch), so its vbase
    // no longer names this instruction's page. `own_pc + 4` is the one true
    // "advanced by exactly one word on this page" address.
    let own_pc = emit_word_addr(ctx, word);
    ctx.builder.ins().store(mem, own_pc, ctx.core_ptr, pc_off);
    let expected_next = ctx.builder.ins().iadd_imm_s(own_pc, 4);

    // (2) call core.interp_fallback_fn(jit_ctx) -> ExecStatus.
    let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
    let jit_ctx = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, jit_ctx_off);
    let fn_off = ir::immediates::Offset32::new(core_offset_of_interp_fallback_fn());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);
    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty)); // jit_ctx
    sig.returns.push(AbiParam::new(ir::types::I32)); // ExecStatus
    let sig_ref = ctx.builder.import_signature(sig);
    let call = ctx.builder.ins().call_indirect(sig_ref, callee, &[jit_ctx]);
    let status = ctx.builder.inst_results(call)[0];

    // (3) status != EXEC_COMPLETE -> return it directly.
    let complete = ctx.builder.ins().icmp_imm_s(IntCC::Equal, status, EXEC_COMPLETE as i64);
    let advanced_check_block = ctx.builder.create_block();
    let not_complete_block = ctx.builder.create_block();
    ctx.builder.ins().brif(complete, advanced_check_block, &[], not_complete_block, &[]);

    ctx.builder.switch_to_block(not_complete_block);
    ctx.builder.seal_block(not_complete_block);
    ctx.builder.ins().return_(&[status]);

    // (4) EXEC_COMPLETE but did PC advance by exactly one word? Compare live
    // core.pc against `own_pc + 4` captured before the call (see step (1)).
    ctx.builder.switch_to_block(advanced_check_block);
    ctx.builder.seal_block(advanced_check_block);
    let live_pc = ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, pc_off);
    let advanced_one = ctx.builder.ins().icmp(IntCC::Equal, live_pc, expected_next);
    let continue_block = ctx.builder.create_block();
    let moved_block = ctx.builder.create_block();
    ctx.builder.ins().brif(advanced_one, continue_block, &[], moved_block, &[]);

    // PC moved elsewhere (ERET/BC1/...): pc is already correct for wherever the
    // interpreter went; just return so the outer loop re-dispatches there.
    ctx.builder.switch_to_block(moved_block);
    ctx.builder.seal_block(moved_block);
    let complete_status = ctx.builder.ins().iconst(ir::types::I32, EXEC_COMPLETE as i64);
    ctx.builder.ins().return_(&[complete_status]);

    // (5) plain one-word advance: continue into the successor.
    ctx.builder.switch_to_block(continue_block);
    ctx.builder.seal_block(continue_block);
    match fallthrough_exit {
        Some(_) => {
            // Region ends here. core.pc is already this+4 (just verified), so
            // return EXEC_COMPLETE directly — no emit_bail pc recompute needed.
            let s = ctx.builder.ins().iconst(ir::types::I32, EXEC_COMPLETE as i64);
            ctx.builder.ins().return_(&[s]);
        }
        None => {
            let next_block = *block_for_word.get(&(word + 1))
                .expect("fallback fallthrough_exit is None -> analyzer guarantees the successor is in-region");
            ctx.builder.ins().jump(next_block, &[]);
        }
    }
}

/// Developer-only per-instruction hook (`developer` builds): call
/// Origin codes passed to `core.dev_trace_bp_fn`'s 4th (`bd`-shaped, now
/// repurposed) parameter — must match `mips_exec::InstrOrigin::from_u32`
/// exactly (that function is the single decode point; these are the encode
/// side). Each `emit_dev_trace_bp` call site passes the constant matching
/// its own arrival class, so `dt` can show exactly how an instruction was
/// reached instead of a collapsed jit/interp bit. `u32`, not `u8` — see
/// `MipsCore::dev_trace_bp_fn`'s doc comment: a narrow `extern "C"` param
/// isn't reliably zero-extended by every caller/ABI, and this was in fact
/// silently broken as `u8` (the hook fired millions of times per
/// `j2 stats`, but `dt` showed no tags at all on a live boot — found live).
#[cfg(feature = "developer")]
mod dev_trace_origin {
    pub const JIT: u32 = 1;
    pub const JIT_ENTRY_BACK_EDGE: u32 = 2;
    pub const JIT_DELAY_SLOT: u32 = 3;
    pub const FALLBACK_SUCCESSOR: u32 = 5;
    pub const FALLBACK_SUCCESSOR_BACK_EDGE: u32 = 6;
}

/// `core.dev_trace_bp_fn(jit_ctx, pc, raw, origin)` with this instruction's
/// synthesized address, compile-time-known `raw`, and `origin` (one of
/// `dev_trace_origin`'s constants, identifying which arrival class this call
/// site is — plain body, entry-word back-edge, delay slot, fallback
/// successor, etc.), so it lands in the `dt` traceback tagged with that
/// origin and can hit PC breakpoints — the visibility the interpreter's
/// `step()` has that a compiled region otherwise runs straight past. On a
/// breakpoint hit (hook returns `EXEC_BREAKPOINT`) this materializes
/// `core.pc = pc` and `core.in_delay_slot = bd` and returns
/// `EXEC_BREAKPOINT`, stopping the monitor *before* the instruction executes
/// with state correct for resume; otherwise control falls through to the
/// instruction's own semantics. Emitted right after the interrupt preamble
/// (`emit_pending_interrupt_preamble`) — the same per-instruction point the
/// interpreter does its trace/breakpoint work — and NOT on the entry word's
/// external-dispatch arm (`step()` already recorded that PC; only the
/// preamble-bearing internal-edge block reaches this — see `compile_region`).
///
/// Does not terminate the current block on the common (no-breakpoint) path —
/// caller continues emitting the instruction's semantics after it. Terminates
/// only the cold breakpoint arm (its own block).
#[cfg(feature = "developer")]
fn emit_dev_trace_bp(ctx: &mut EmitCtx, origin: u32) {
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();
    let word = ctx.word;

    let pc_val = emit_word_addr(ctx, word);
    let raw_val = ctx.builder.ins().iconst(ir::types::I32, ctx.raw as i64);
    let origin_val = ctx.builder.ins().iconst(ir::types::I32, origin as i64);

    let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
    let jit_ctx = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, jit_ctx_off);
    let fn_off = ir::immediates::Offset32::new(core_offset_of_dev_trace_bp_fn());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);

    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty));         // jit_ctx
    sig.params.push(AbiParam::new(ir::types::I64)); // pc
    sig.params.push(AbiParam::new(ir::types::I32)); // raw
    sig.params.push(AbiParam::new(ir::types::I32)); // origin
    sig.returns.push(AbiParam::new(ir::types::I32)); // ExecStatus
    let sig_ref = ctx.builder.import_signature(sig);
    let call = ctx.builder.ins().call_indirect(sig_ref, callee, &[jit_ctx, pc_val, raw_val, origin_val]);
    let status = ctx.builder.inst_results(call)[0];

    let is_bp = ctx.builder.ins().icmp_imm_s(IntCC::Equal, status, crate::mips_exec::EXEC_BREAKPOINT as i64);
    let bp_block = ctx.builder.create_block();
    let continue_block = ctx.builder.create_block();
    ctx.builder.ins().brif(is_bp, bp_block, &[], continue_block, &[]);

    // Cold: a breakpoint hit is rare and interactive.
    ctx.builder.switch_to_block(bp_block);
    ctx.builder.set_cold_block(bp_block);
    ctx.builder.seal_block(bp_block);
    // core.pc/in_delay_slot already correct for resume: pc = this word, bd =
    // ctx.bd. (pc_val recomputed rather than threaded — trivial and keeps this
    // self-contained.)
    let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
    let flag_off = ir::immediates::Offset32::new(core_offset_of_in_delay_slot());
    let pc_again = emit_word_addr(ctx, word);
    ctx.builder.ins().store(mem, pc_again, ctx.core_ptr, pc_off);
    let bd_store = ctx.builder.ins().iconst(ir::types::I8, ctx.bd as i64);
    ctx.builder.ins().store(mem, bd_store, ctx.core_ptr, flag_off);
    let bp_status = ctx.builder.ins().iconst(ir::types::I32, crate::mips_exec::EXEC_BREAKPOINT as i64);
    ctx.builder.ins().return_(&[bp_status]);

    ctx.builder.switch_to_block(continue_block);
    ctx.builder.seal_block(continue_block);
}

/// `jitv2_lockstep` per-instruction STEP bracket (`emit_lockstep_step`): under
/// lockstep only, materialize this instruction's starting `core.pc`/
/// `in_delay_slot` (the JIT doesn't keep `core.pc` live for straight-line
/// instructions, so lockstep does — see the module doc) and call
/// `core.lockstep_step_fn(jit_ctx, pc, raw, bd)`, which runs the interpreter
/// reference for this instruction and leaves the starting state restored for
/// the JIT to run against. Emitted right after the interrupt preamble, before
/// the instruction's own semantics. Not a terminator.
/// `trust_live`: for an entry word or delay slot (branch-fallback successor),
/// `core.pc`/`in_delay_slot`/`delay_slot_target` are already correct from the
/// arrival and must NOT be materialized from compile-time constants — pass the
/// `LOCKSTEP_BD_LIVE` sentinel so the callback preserves the live state, and
/// skip the pc/bd stores. For a plain straight-line head (`trust_live=false`)
/// the JIT doesn't keep core.pc live, so materialize it here as before.
#[cfg(feature = "jitv2_lockstep")]
fn emit_lockstep_step(ctx: &mut EmitCtx, trust_live: bool) {
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();
    let word = ctx.word;

    let pc_val = emit_word_addr(ctx, word);
    let bd_arg = if trust_live {
        // Trust live pc/in_delay_slot — don't overwrite them.
        crate::mips_exec::LOCKSTEP_BD_LIVE
    } else {
        // Plain head: materialize the starting pc/in_delay_slot so both engines
        // are anchored (the JIT doesn't keep core.pc live for straight-line ops).
        let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
        ctx.builder.ins().store(mem, pc_val, ctx.core_ptr, pc_off);
        let flag_off = ir::immediates::Offset32::new(core_offset_of_in_delay_slot());
        let bd_v = ctx.builder.ins().iconst(ir::types::I8, ctx.bd as i64);
        ctx.builder.ins().store(mem, bd_v, ctx.core_ptr, flag_off);
        ctx.bd as u32
    };

    // I32, not I8, for the call param — see LOCKSTEP_BD_LIVE's doc comment
    // (a narrow extern "C" param isn't reliably zero-extended by every
    // caller/ABI; core.in_delay_slot's own I8 STORE above is unrelated and
    // stays I8, it's real memory, not a call argument).
    let bd_val = ctx.builder.ins().iconst(ir::types::I32, bd_arg as i64);
    let raw_val = ctx.builder.ins().iconst(ir::types::I32, ctx.raw as i64);
    let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
    let jit_ctx = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, jit_ctx_off);
    let fn_off = ir::immediates::Offset32::new(core_offset_of_lockstep_step_fn());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);
    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty));         // jit_ctx
    sig.params.push(AbiParam::new(ir::types::I64)); // pc
    sig.params.push(AbiParam::new(ir::types::I32)); // raw
    sig.params.push(AbiParam::new(ir::types::I32)); // bd
    let sig_ref = ctx.builder.import_signature(sig);
    ctx.builder.ins().call_indirect(sig_ref, callee, &[jit_ctx, pc_val, raw_val, bd_val]);
}

/// `jitv2_lockstep` compare bracket for an entry word / delay slot, called
/// *after* the foreign-slot check has resolved the final `core.pc` (to
/// `word+1` on a plain arrival, or `delay_slot_target` on a pending-transfer
/// one). Unlike `emit_lockstep_compare_seq` it does NOT materialize pc — the
/// caller already set it correctly on the current arm — it just runs the
/// compare hook and, on a divergence (`EXEC_BREAKPOINT`), returns that (pc is
/// already at the right place for the monitor). Both arms of the foreign-slot
/// check call this. Terminates the current block only on the divergence arm.
#[cfg(feature = "jitv2_lockstep")]
fn emit_lockstep_compare_live(ctx: &mut EmitCtx) {
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();

    let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
    let jit_ctx = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, jit_ctx_off);
    let fn_off = ir::immediates::Offset32::new(core_offset_of_lockstep_compare_fn());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);
    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty)); // jit_ctx
    sig.returns.push(AbiParam::new(ir::types::I32)); // ExecStatus
    let sig_ref = ctx.builder.import_signature(sig);
    let call = ctx.builder.ins().call_indirect(sig_ref, callee, &[jit_ctx]);
    let status = ctx.builder.inst_results(call)[0];

    let is_bp = ctx.builder.ins().icmp_imm_s(IntCC::Equal, status, crate::mips_exec::EXEC_BREAKPOINT as i64);
    let bp_block = ctx.builder.create_block();
    let continue_block = ctx.builder.create_block();
    ctx.builder.ins().brif(is_bp, bp_block, &[], continue_block, &[]);

    ctx.builder.switch_to_block(bp_block);
    ctx.builder.set_cold_block(bp_block);
    ctx.builder.seal_block(bp_block);
    // pc is already final for this arm — leave it, just return the break.
    let bp_status = ctx.builder.ins().iconst(ir::types::I32, crate::mips_exec::EXEC_BREAKPOINT as i64);
    ctx.builder.ins().return_(&[bp_status]);

    ctx.builder.switch_to_block(continue_block);
    ctx.builder.seal_block(continue_block);
}

/// `jitv2_lockstep` per-instruction COMPARE bracket (`emit_lockstep_compare`):
/// under lockstep only, materialize the JIT's final `core.pc = pc + 4` for a
/// straight-line instruction that didn't write pc itself (so the pc compare is
/// direct against the interpreter's advanced pc), then call
/// `core.lockstep_compare_fn(jit_ctx)` which compares JIT vs the interpreter
/// reference `lockstep_step` stashed and panics on divergence. Emitted after
/// the instruction's own semantics, before the fallthrough edge. Only used for
/// straight-line (Sequential/CP1) instructions here — branch/jump/regjump and
/// fallback manage pc/control themselves and are bracketed (or skipped)
/// separately.
///
/// If the compare hook returns `EXEC_BREAKPOINT` (a divergence — it already
/// printed the report), this materializes `core.pc = word` (the divergent
/// instruction's own address, so the monitor stops there / resume re-runs it)
/// and returns `EXEC_BREAKPOINT` from the compiled function — a cold terminator
/// arm. The no-divergence path falls through to the instruction's own
/// fallthrough edge (not a terminator).
///
/// `region_ending`: `true` when this word's fallthrough exits the region
/// (`instrs[word].fallthrough_exit.is_some()`). This function does NOTHING at
/// all in that case — no materialize, no compare call — and the caller must
/// skip calling it (see the `ls_bracket && !region_ending` guard at the call
/// site). The compare still happens, just not here: `plain_fallthrough`'s
/// `Some(_)` arm calls `emit_bail`, whose shared target block
/// (`emit_exit_block_body`) writes `core.pc` to its one true final value AND
/// runs the compare right after, since that's the only place `core.pc` is
/// genuinely final for a region-ending word — anywhere in *this* function is
/// necessarily before that real write. An earlier version of this function
/// wrote a second, throwaway `core.pc = word+1` here purely to give the
/// compare something to check, then tried to undo it before falling through
/// to `emit_bail` — but `emit_bail`'s own vbase re-derivation reads
/// *whatever's currently in `core.pc`* to figure out which page it's on, so
/// that throwaway write (and the undo's own reload of the same,
/// already-clobbered cell) doubled the page-crossing instead of just
/// happening once (found live:
/// `sequential_pair_ending_at_0xffc_falls_through_to_next_page`). Not
/// materializing anything here at all is what actually closes that class of
/// bug, rather than trying to get the undo's bookkeeping right.
#[cfg(feature = "jitv2_lockstep")]
fn emit_lockstep_compare_seq(ctx: &mut EmitCtx) {
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();
    let word = ctx.word;

    // Straight-line instruction: the interpreter advanced pc to word+1, but the
    // JIT's semantics didn't touch core.pc. Materialize pc+4 so the compare's
    // pc field matches. in_delay_slot: a plain Sequential/CP1 op never changes
    // it, and the interpreter leaves it false after a non-slot retire — the
    // start bracket already set it to ctx.bd (false for a non-slot head), so
    // it's already correct.
    let next_pc = emit_word_addr(ctx, word + 1);
    let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
    ctx.builder.ins().store(mem, next_pc, ctx.core_ptr, pc_off);

    let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
    let jit_ctx = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, jit_ctx_off);
    let fn_off = ir::immediates::Offset32::new(core_offset_of_lockstep_compare_fn());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);
    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty)); // jit_ctx
    sig.returns.push(AbiParam::new(ir::types::I32)); // ExecStatus
    let sig_ref = ctx.builder.import_signature(sig);
    let call = ctx.builder.ins().call_indirect(sig_ref, callee, &[jit_ctx]);
    let status = ctx.builder.inst_results(call)[0];

    // On a divergence, bail to the monitor: EXEC_BREAKPOINT. core.pc/
    // in_delay_slot/delay_slot_target are already back at the divergent
    // instruction's own starting values — lockstep_compare (mips_exec.rs)
    // restores all three from `ls_before`/`ls_delay_target_before` itself,
    // in Rust, before returning — codegen has nothing left to fix up here.
    let is_bp = ctx.builder.ins().icmp_imm_s(IntCC::Equal, status, crate::mips_exec::EXEC_BREAKPOINT as i64);
    let bp_block = ctx.builder.create_block();
    let continue_block = ctx.builder.create_block();
    ctx.builder.ins().brif(is_bp, bp_block, &[], continue_block, &[]);

    ctx.builder.switch_to_block(bp_block);
    ctx.builder.set_cold_block(bp_block);
    ctx.builder.seal_block(bp_block);
    let bp_status = ctx.builder.ins().iconst(ir::types::I32, crate::mips_exec::EXEC_BREAKPOINT as i64);
    ctx.builder.ins().return_(&[bp_status]);

    ctx.builder.switch_to_block(continue_block);
    ctx.builder.seal_block(continue_block);
}

/// Body of the shared exit-to-interpreter block (`BlockSkeleton::exit_block`),
/// emitted once by `build_block_skeleton`: materialize `core.pc = vbase |
/// (word_offset * 4)` (the exiting instruction's own address) and return
/// `EXEC_COMPLETE`, so the interpreter's `step()` re-enters at the top
/// and re-runs whatever check triggered the bail as the authoritative
/// implementation. `core_ptr`/`word_offset` here are the block's own params
/// (runtime `Value`s), not compile-time constants — every bail site feeds its
/// own `word_offset` in via [`emit_bail`]. `vbase` is derived from the live
/// `core.pc & !0xFFF` rather than threaded as a separate parameter (§2.2's
/// vbase and the exiting instruction's own current page base are the same
/// value by construction, since the whole compiled region is one page).
///
/// Under `jitv2_lockstep`: the compare runs HERE, after `exit_pc` is written,
/// not at the per-instruction bracket site — this is the one place `core.pc`
/// is genuinely final for every bail alike, preamble or post-semantics. A
/// per-instruction bracket (`emit_lockstep_compare_seq`) only needs to
/// materialize `core.pc` itself for a *plain, in-region* fallthrough, where
/// nothing else will ever write it; a bail already has real control-flow
/// (this function) writing the true final `core.pc`, so materializing a
/// second, throwaway value earlier — as an older version of this code did,
/// purely to give the compare something to check — created exactly the
/// double-page-jump this design avoids: that throwaway write and this
/// function's own vbase re-derivation from *whatever's currently in core.pc*
/// would stack two page-crossings when the throwaway value had already
/// crossed the boundary once (found live:
/// `sequential_pair_ending_at_0xffc_falls_through_to_next_page`). Comparing
/// here means the compare never needs its own pc write at all — it reads
/// whatever this function already wrote for real. A preamble bail (nothing
/// staged this dispatch) is a harmless no-op, same as every other lockstep
/// compare call.
fn emit_exit_block_body(builder: &mut FunctionBuilder, module: &mut dyn cranelift_module::Module, core_ptr: Value, word_offset: Value) {
    let mem = MemFlagsData::trusted();
    let i64t = ir::types::I64;
    let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());

    let pc = builder.ins().load(i64t, mem, core_ptr, pc_off);
    let vbase = builder.ins().band_imm_s(pc, !(PAGE_SIZE as i64 - 1));
    let byte_offset = builder.ins().imul_imm_s(word_offset, 4);
    // iadd, not bor: a Sequential instruction's fallthrough off the page's
    // last word (0xFFC hazard only special-cases branch/jump/regjump there,
    // not Sequential) makes word_offset = WORDS_PER_PAGE, so byte_offset is
    // exactly PAGE_SIZE — bor silently drops that page carry whenever
    // vbase's own PAGE_SIZE bit happens to already be set, landing back on
    // this page instead of the next one. iadd carries correctly there and is
    // equivalent to bor for every in-range word_offset, so this isn't a
    // behavior change elsewhere. (Found via jitv2_verify against a real
    // IRIX 5.3 boot trace.)
    let exit_pc = builder.ins().iadd(vbase, byte_offset);
    builder.ins().store(mem, exit_pc, core_ptr, pc_off);

    #[cfg(feature = "jitv2_lockstep")]
    {
        let ptr_ty = module.target_config().pointer_type();
        let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
        let jit_ctx = builder.ins().load(ptr_ty, mem, core_ptr, jit_ctx_off);
        let fn_off = ir::immediates::Offset32::new(core_offset_of_lockstep_compare_fn());
        let callee = builder.ins().load(ptr_ty, mem, core_ptr, fn_off);
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty)); // jit_ctx
        sig.returns.push(AbiParam::new(ir::types::I32)); // ExecStatus
        let sig_ref = builder.import_signature(sig);
        let call = builder.ins().call_indirect(sig_ref, callee, &[jit_ctx]);
        let cmp_status = builder.inst_results(call)[0];

        // Divergence: lockstep_compare (mips_exec.rs) already restored
        // core.pc/in_delay_slot/delay_slot_target to the divergent
        // instruction's own starting values from `ls_before`/
        // `ls_delay_target_before` before returning — nothing left to fix up
        // here, just propagate the breakpoint status instead of EXEC_COMPLETE.
        let is_bp = builder.ins().icmp_imm_s(IntCC::Equal, cmp_status, crate::mips_exec::EXEC_BREAKPOINT as i64);
        let bp_block = builder.create_block();
        let continue_block = builder.create_block();
        builder.ins().brif(is_bp, bp_block, &[], continue_block, &[]);

        builder.switch_to_block(bp_block);
        builder.set_cold_block(bp_block);
        builder.seal_block(bp_block);
        let bp_status = builder.ins().iconst(ir::types::I32, crate::mips_exec::EXEC_BREAKPOINT as i64);
        builder.ins().return_(&[bp_status]);

        builder.switch_to_block(continue_block);
        builder.seal_block(continue_block);
    }

    let status = builder.ins().iconst(ir::types::I32, EXEC_COMPLETE as i64);
    builder.ins().return_(&[status]);
}

fn core_offset_of_pc() -> i32 { std::mem::offset_of!(MipsCore, pc) as i32 }
fn core_offset_of_jit_trigger() -> i32 { std::mem::offset_of!(MipsCore, jit_trigger) as i32 }

/// Mark `core.jit_trigger` so the interpreter's `exec_decoded` dispatch gate
/// probes this exit's target PC as a fresh compile-worthy arrival — the
/// JIT-compiled-code counterpart to the interpreter's own
/// `handle_exec_complete`/`exec_complete_pc_set` (see `MipsCore::jit_trigger`'s
/// doc comment). Called from every jump/branch exit stub
/// (`emit_absolute_pc_exit`, `emit_runtime_pc_exit`) right before the exit
/// stores `core.pc` and returns — *not* from `emit_exit_block_body` (the
/// plain sequential-fallthrough exit path), which mirrors the interpreter's
/// own asymmetry: a straight-line `pc += 4` fallthrough never sets the
/// trigger either, only an actual taken transfer does.
fn emit_set_jit_trigger(ctx: &mut EmitCtx) {
    let mem = MemFlagsData::trusted();
    let off = ir::immediates::Offset32::new(core_offset_of_jit_trigger());
    let one = ctx.builder.ins().iconst(ir::types::I8, 1);
    ctx.builder.ins().store(mem, one, ctx.core_ptr, off);
}
fn core_offset_of_cycles() -> i32 {
    (std::mem::offset_of!(MipsCore, hot) + std::mem::offset_of!(crate::mips_core::Hot, cycles)) as i32
}

/// Account for one retiring architectural instruction against
/// `core.hot.cycles` — the JIT-compiled-code counterpart to the
/// interpreter's `step()` incrementing it once per `step()` call
/// (`src/mips_exec.rs`: a real, direct write — see `Hot::cycles`'s own doc
/// comment for why it must never silently stop advancing). A compiled unit
/// never calls the interpreter's `step()` for the instructions it covers, so
/// without this, `cycles` — and everything that depends on it being visibly
/// live while a hot guest loop runs entirely inside JIT-compiled code (e.g.
/// `Wd33c93a`'s BSD SCSI deferred-interrupt spin-wait, on a completely
/// different thread) — would silently stop advancing for however many
/// instructions ran under real JIT dispatch.
///
/// Batched, not a store per instruction: adds `word`'s
/// `CompiledInstr::cycles_delta` (always 1 — see that field's doc comment)
/// to `*ctx.cycles_pending` unconditionally, then stores the running total
/// to memory — resetting it to 0 — only when `word`'s `cycles_flush` is
/// set. The analyzer's `compute_cycles_flush` post-pass guarantees this is
/// safe: `cycles_flush` is true at every region exit and at every loop
/// re-entry point (ordinary back-edge target, or the region's own
/// `entry_word` when it doubles as a loop head), which is exactly
/// everywhere `Hot::cycles`'s cross-thread-observability contract requires
/// a fresh value in memory — see that field's doc comment for the full
/// reasoning. A plain interior word between two such points has
/// `cycles_flush = false` and just grows the pending count, no store at
/// all.
///
/// Called once per head instruction (the per-instruction emission loop in
/// `compile_region_uncommitted`) and once per inlined delay slot
/// (`emit_slot_semantics`) — a branch/jump's delay slot is a second,
/// separate architectural instruction retiring, even though it has no
/// head-instruction loop iteration of its own (it's always inlined into its
/// branch's compiled body, §6.1.4). Both call sites pass the word whose
/// `cycles_delta`/`cycles_flush` apply — the head's own `word` for the
/// first, the slot's own word for the second.
///
/// A branch/jump/regjump head with a real inline slot never has
/// `cycles_flush = true` on its own row, even when it's a region exit —
/// `compute_cycles_flush` deliberately pushes that decision onto the slot
/// word (`has_inline_slot`'s doc comment). This function doesn't need to
/// know that; it just trusts whatever `cycles_flush` says for the exact
/// `word` it's given, which is already correct by construction as long as
/// every call site passes the *actually retiring* word (the head's own for
/// the head-loop call, the slot's own for the slot call) rather than
/// assuming the head "speaks for" its slot.
///
/// Plain load/store here, not `ptr::read_volatile`/`write_volatile` the way
/// the interpreter's own increment site does it in Rust: Cranelift's
/// `MemFlagsData::trusted()` load/store already genuinely touches memory on
/// every call (there's no equivalent "the compiler proves this loop's writes
/// are dead and elides them" risk inside a single compiled unit the way
/// there theoretically is for a pure-Rust unbounded loop) — the volatile
/// requirement is specific to the interpreter's own increment, not this one.
fn emit_account_for_cycles(ctx: &mut EmitCtx, instrs: &[CompiledInstr; ENTRIES_PER_PAGE], word: WordOffset) {
    let instr = &instrs[word as usize];
    *ctx.cycles_pending += instr.cycles_delta;
    if !instr.cycles_flush {
        return;
    }
    let mem = MemFlagsData::trusted();
    let off = ir::immediates::Offset32::new(core_offset_of_cycles());
    let prev = ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, off);
    let next = ctx.builder.ins().iadd_imm_s(prev, *ctx.cycles_pending as i64);
    ctx.builder.ins().store(mem, next, ctx.core_ptr, off);
    *ctx.cycles_pending = 0;
}
fn core_offset_of_interrupts() -> i32 {
    (std::mem::offset_of!(MipsCore, hot) + std::mem::offset_of!(crate::mips_core::Hot, interrupts)) as i32
}
fn core_offset_of_hi() -> i32 { std::mem::offset_of!(MipsCore, hi) as i32 }
fn core_offset_of_lo() -> i32 { std::mem::offset_of!(MipsCore, lo) as i32 }
fn core_offset_of_cp0_status() -> i32 { std::mem::offset_of!(MipsCore, cp0_status) as i32 }
fn core_offset_of_cp0_cause() -> i32 { std::mem::offset_of!(MipsCore, cp0_cause) as i32 }
fn core_offset_of_fpu_fcsr() -> i32 { std::mem::offset_of!(MipsCore, fpu_fcsr) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_fpu_set_mode_fn() -> i32 { std::mem::offset_of!(MipsCore, fpu_set_mode_fn) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_fpu_cvt_to_int_fn() -> i32 { std::mem::offset_of!(MipsCore, fpu_cvt_to_int_fn) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_fpu_cvt_int_to_float_fn() -> i32 { std::mem::offset_of!(MipsCore, fpu_cvt_int_to_float_fn) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_fpu_cvt_d_to_s_fn() -> i32 { std::mem::offset_of!(MipsCore, fpu_cvt_d_to_s_fn) as i32 }
fn core_offset_of_fpu_fir() -> i32 { std::mem::offset_of!(MipsCore, fpu_fir) as i32 }
fn core_offset_of_fpu_fexr() -> i32 { std::mem::offset_of!(MipsCore, fpu_fexr) as i32 }
fn core_offset_of_fpu_fenr() -> i32 { std::mem::offset_of!(MipsCore, fpu_fenr) as i32 }
fn core_offset_of_fpu_fccr() -> i32 { std::mem::offset_of!(MipsCore, fpu_fccr) as i32 }

/// Byte offset of `core.fpr[reg]`. `fpr` is `[u64; 32]` (`MipsCore::fpr`) —
/// same shape as `core_offset_of_gpr`.
fn core_offset_of_fpr(reg: u32) -> i32 {
    std::mem::offset_of!(MipsCore, fpr) as i32 + (reg as i32) * 8
}

/// Load `core.hi`/`core.lo` as I64 (plain, non-atomic — same access pattern
/// as `emit_read_gpr`).
fn emit_read_hi(ctx: &mut EmitCtx) -> Value {
    let mem = MemFlagsData::trusted();
    let off = ir::immediates::Offset32::new(core_offset_of_hi());
    ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, off)
}
fn emit_read_lo(ctx: &mut EmitCtx) -> Value {
    let mem = MemFlagsData::trusted();
    let off = ir::immediates::Offset32::new(core_offset_of_lo());
    ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, off)
}
fn emit_write_hi(ctx: &mut EmitCtx, value: Value) {
    let mem = MemFlagsData::trusted();
    let off = ir::immediates::Offset32::new(core_offset_of_hi());
    ctx.builder.ins().store(mem, value, ctx.core_ptr, off);
}
fn emit_write_lo(ctx: &mut EmitCtx, value: Value) {
    let mem = MemFlagsData::trusted();
    let off = ir::immediates::Offset32::new(core_offset_of_lo());
    ctx.builder.ins().store(mem, value, ctx.core_ptr, off);
}

/// Byte offset of `core.gpr[reg]`. `gpr` is `[u64; 32]` (`MipsCore::gpr`) —
/// index arithmetic on the base offset rather than a second `offset_of!`
/// call per register.
fn core_offset_of_gpr(reg: u32) -> i32 {
    std::mem::offset_of!(MipsCore, gpr) as i32 + (reg as i32) * 8
}

#[cfg(feature = "jitv2")]
fn core_offset_of_jit_ctx() -> i32 { std::mem::offset_of!(MipsCore, jit_ctx) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_read8_fn() -> i32 { std::mem::offset_of!(MipsCore, read8_fn) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_read16_fn() -> i32 { std::mem::offset_of!(MipsCore, read16_fn) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_read32_fn() -> i32 { std::mem::offset_of!(MipsCore, read32_fn) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_read64_fn() -> i32 { std::mem::offset_of!(MipsCore, read64_fn) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_write8_fn() -> i32 { std::mem::offset_of!(MipsCore, write8_fn) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_write16_fn() -> i32 { std::mem::offset_of!(MipsCore, write16_fn) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_write32_fn() -> i32 { std::mem::offset_of!(MipsCore, write32_fn) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_write64_fn() -> i32 { std::mem::offset_of!(MipsCore, write64_fn) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_write64_masked_fn() -> i32 { std::mem::offset_of!(MipsCore, write64_masked_fn) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_handle_exception_fn() -> i32 { std::mem::offset_of!(MipsCore, handle_exception_fn) as i32 }
fn core_offset_of_interp_fallback_fn() -> i32 { std::mem::offset_of!(MipsCore, interp_fallback_fn) as i32 }
fn core_offset_of_kill_entry_fn() -> i32 { std::mem::offset_of!(MipsCore, kill_entry_fn) as i32 }
#[cfg(feature = "developer")]
fn core_offset_of_dev_trace_bp_fn() -> i32 { std::mem::offset_of!(MipsCore, dev_trace_bp_fn) as i32 }
#[cfg(feature = "jitv2_lockstep")]
fn core_offset_of_lockstep_step_fn() -> i32 { std::mem::offset_of!(MipsCore, lockstep_step_fn) as i32 }
#[cfg(feature = "jitv2_lockstep")]
fn core_offset_of_lockstep_compare_fn() -> i32 { std::mem::offset_of!(MipsCore, lockstep_compare_fn) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_jit_mem_exc() -> i32 { std::mem::offset_of!(MipsCore, jit_mem_exc) as i32 }
#[cfg(feature = "jitv2")]
fn core_offset_of_in_delay_slot() -> i32 { std::mem::offset_of!(MipsCore, in_delay_slot) as i32 }
fn core_offset_of_delay_slot_target() -> i32 { std::mem::offset_of!(MipsCore, delay_slot_target) as i32 }

/// Memory access width, mirroring `MipsExecutor::read_data::<SIZE>`/
/// `write_data::<SIZE>`'s const-generic `SIZE` parameter (bytes).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MemSize { B1, B2, B4, B8 }

impl MemSize {
    fn read_fn_offset(self) -> i32 {
        match self {
            MemSize::B1 => core_offset_of_read8_fn(),
            MemSize::B2 => core_offset_of_read16_fn(),
            MemSize::B4 => core_offset_of_read32_fn(),
            MemSize::B8 => core_offset_of_read64_fn(),
        }
    }
    fn write_fn_offset(self) -> i32 {
        match self {
            MemSize::B1 => core_offset_of_write8_fn(),
            MemSize::B2 => core_offset_of_write16_fn(),
            MemSize::B4 => core_offset_of_write32_fn(),
            MemSize::B8 => core_offset_of_write64_fn(),
        }
    }
    /// Cranelift type of the *stored/loaded* value at this width, before any
    /// zero/sign-extension the caller applies afterward — `read*_fn` always
    /// returns the full u64 (already zero-extended by the wrapper on the Rust
    /// side, see `MipsCore`'s field doc comment), so this is only used for
    /// `write*_fn`'s narrower value parameter.
    fn ir_type(self) -> ir::Type {
        match self {
            MemSize::B1 => ir::types::I8,
            MemSize::B2 => ir::types::I16,
            MemSize::B4 => ir::types::I32,
            MemSize::B8 => ir::types::I64,
        }
    }
    /// Access width in bytes — used by the unaligned load/store family
    /// (LWL/LWR/LDL/LDR/SWL/SWR/SDL/SDR) to compute the alignment mask
    /// (`width_bytes() - 1`) and total bit width (`width_bytes() * 8`) for
    /// their runtime-variable shift/mask arithmetic.
    fn width_bytes(self) -> u32 {
        match self {
            MemSize::B1 => 1,
            MemSize::B2 => 2,
            MemSize::B4 => 4,
            MemSize::B8 => 8,
        }
    }
}

/// Emit a call through `core.read{8,16,32,64}_fn(core.jit_ctx, vaddr)`
/// (§3.3 "memory access = the interpreter's own access routine"). Returns
/// the loaded value as I64 (already zero-extended by the wrapper — see
/// `MipsCore::read32_fn`'s doc comment) — callers needing sign extension
/// (LB/LH/LW) must `sextend` from the narrower width themselves, same as the
/// interpreter's own handlers do after calling `read_data`.
///
/// Does not itself check the exception result — call [`emit_check_mem_exc`]
/// immediately after with the same `word_offset`/`exit_block` to bail out on
/// a fault before trusting the returned value, exactly like every
/// interpreter load handler's `match self.read_data(...) { Ok(v) => ...,
/// Err(status) => ... }`.
fn emit_mem_read(ctx: &mut EmitCtx, vaddr: Value, size: MemSize) -> Value {
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();

    let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
    let jit_ctx = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, jit_ctx_off);

    let fn_off = ir::immediates::Offset32::new(size.read_fn_offset());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);

    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty)); // jit_ctx
    sig.params.push(AbiParam::new(ir::types::I64)); // vaddr
    sig.returns.push(AbiParam::new(ir::types::I64)); // value
    let sig_ref = ctx.builder.import_signature(sig);

    let call = ctx.builder.ins().call_indirect(sig_ref, callee, &[jit_ctx, vaddr]);
    ctx.builder.inst_results(call)[0]
}

/// Emit a call through `core.write{8,16,32,64}_fn(core.jit_ctx, vaddr, value)`.
/// `value` is always passed as I64, regardless of `size` — never narrowed to
/// `size.ir_type()` here or by the caller. This is deliberate, not a
/// simplification: the x86-64 SysV C ABI doesn't guarantee a sub-word
/// integer argument's upper register bits are zeroed by the caller (nor does
/// this hand-built `call_indirect` signature impose that on itself), so a
/// narrower `AbiParam`/`Value` here previously let whatever garbage sat in
/// the unused high bits of the argument register leak into the callee's own
/// `val as u64` widening on the Rust side (observed live: SB writes coming
/// through with garbage-prefixed values like `0xffffff00` where only the low
/// byte, `0x00`, was architecturally meaningful). `MipsCore::write8_fn`
/// (etc.)'s signature was changed to match — always `u64`, with the Rust
/// wrapper (`jit_write8` in `mips_exec.rs`) masking to the real width
/// itself, exactly like `emit_mem_read`'s callers already mask/extend a
/// full-width return value down to what they need. Returns the `ExecStatus`
/// result (I32) — 0 on success, an exception status otherwise; also
/// mirrored into `core.jit_mem_exc` by the wrapper (see
/// [`emit_check_mem_exc`]), so callers can use the same check helper
/// uniformly for both loads and stores rather than branching on this return
/// value directly.
fn emit_mem_write(ctx: &mut EmitCtx, vaddr: Value, value: Value, size: MemSize) {
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();

    let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
    let jit_ctx = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, jit_ctx_off);

    let fn_off = ir::immediates::Offset32::new(size.write_fn_offset());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);

    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty)); // jit_ctx
    sig.params.push(AbiParam::new(ir::types::I64)); // vaddr
    sig.params.push(AbiParam::new(ir::types::I64)); // value — always I64, see doc comment
    sig.returns.push(AbiParam::new(ir::types::I32)); // ExecStatus
    let sig_ref = ctx.builder.import_signature(sig);

    ctx.builder.ins().call_indirect(sig_ref, callee, &[jit_ctx, vaddr, value]);
    // Return value intentionally unused here — emit_check_mem_exc reads
    // core.jit_mem_exc instead, so loads and stores share one check path.
}

/// Emit a call through `core.write64_masked_fn(core.jit_ctx, aligned_addr,
/// val, mask)` — the SWL/SWR/SDL/SDR counterpart to [`emit_mem_write`].
/// `aligned_addr` must already be doubleword-aligned (callers compute this
/// themselves, same contract as `MipsExecutor::write_data64_masked`); `val`
/// and `mask` are both full 64-bit values already positioned in
/// doubleword-space by the caller (`emit_swl`/etc.'s own dword-shift
/// promotion), not narrowed/widened here.
fn emit_mem_write_masked(ctx: &mut EmitCtx, aligned_addr: Value, val: Value, mask: Value) {
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();

    let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
    let jit_ctx = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, jit_ctx_off);

    let fn_off = ir::immediates::Offset32::new(core_offset_of_write64_masked_fn());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);

    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty)); // jit_ctx
    sig.params.push(AbiParam::new(ir::types::I64)); // aligned_addr
    sig.params.push(AbiParam::new(ir::types::I64)); // val
    sig.params.push(AbiParam::new(ir::types::I64)); // mask
    sig.returns.push(AbiParam::new(ir::types::I32)); // ExecStatus
    let sig_ref = ctx.builder.import_signature(sig);

    ctx.builder.ins().call_indirect(sig_ref, callee, &[jit_ctx, aligned_addr, val, mask]);
    // Return value intentionally unused — emit_check_mem_exc reads
    // core.jit_mem_exc instead, same convention as emit_mem_write.
}

/// After [`emit_mem_read`]/[`emit_mem_write`]: load `core.jit_mem_exc` and
/// route it exactly like the interpreter's own `finish_status` does for a
/// status straight out of `read_data`/`write_data` — retry/breakpoint-only
/// possible here, since loads/stores are the only emitters that ever call
/// this (§the interpreter's `finish_status` doc comment: "dispatches to
/// handle_exception if EXEC_IS_EXCEPTION is set, otherwise passes
/// EXEC_BREAKPOINT/EXEC_RETRY straight through unchanged"):
///
/// - `0` (`EXEC_COMPLETE`): no fault, fall through — the common case.
/// - `EXEC_IS_EXCEPTION` bit set (a real MIPS exception — ADEL/ADES/DBE/…):
///   deliver via [`emit_exception_exit`] (BD/entry-word-aware `core.pc`
///   synthesis, then `handle_exception_fn`).
/// - nonzero but `EXEC_IS_EXCEPTION` clear (`EXEC_RETRY`/`EXEC_BREAKPOINT`):
///   **not** an exception — the bus was busy, or a memory breakpoint fired
///   mid-access — nothing architectural happened and nothing should retire.
///   Bail to the interpreter at `ctx.word` via [`emit_bail`]/`exit_block`
///   instead of `emit_exception_exit`: a bail only writes `core.pc` (never
///   touches `in_delay_slot`, never calls `deliver_exception`), so the
///   interpreter's very next `step()` simply re-dispatches this exact
///   instruction from scratch and gets a fresh status — identical
///   observable effect to what the interpreter's own retry does, and
///   correct even for the entry-word-as-foreign-slot case (an already-armed
///   `in_delay_slot` survives untouched, same reasoning as `emit_bail`'s own
///   doc comment). (Found live via `jitcheck`'s new `step_status` digest
///   field: `EXEC_RETRY` from a transient bus-busy read was being routed
///   into `emit_exception_exit` and vectored as a real exception — the
///   interpreter, given the identical read, simply retried and moved on.)
///
/// Leaves `builder` positioned in a new, sealed continuation block (the
/// "no fault" path) if status was `0` — callers continue emitting there.
/// Must be called with `builder` positioned in the block holding the memory
/// op just emitted.
fn emit_check_mem_exc(ctx: &mut EmitCtx) {
    let mem = MemFlagsData::trusted();
    let exc_off = ir::immediates::Offset32::new(core_offset_of_jit_mem_exc());
    let exc = ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, exc_off);
    let zero = ctx.builder.ins().iconst(ir::types::I32, 0);
    let has_status = ctx.builder.ins().icmp(IntCC::NotEqual, exc, zero);

    let status_block = ctx.builder.create_block();
    let continue_block = ctx.builder.create_block();
    ctx.builder.ins().brif(has_status, status_block, &[], continue_block, &[]);

    ctx.builder.switch_to_block(status_block);
    // Cold: the common case for any memory access is a plain success status
    // (0) — see this function's other `set_cold_block` calls for the full
    // rationale, repeated at every block on this rare-status path.
    ctx.builder.set_cold_block(status_block);
    ctx.builder.seal_block(status_block);
    let is_exception = ctx.builder.ins().band_imm_s(exc, EXEC_IS_EXCEPTION as i64);
    let is_exception = ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, is_exception, 0);

    let exception_block = ctx.builder.create_block();
    let retry_block = ctx.builder.create_block();
    ctx.builder.ins().brif(is_exception, exception_block, &[], retry_block, &[]);

    ctx.builder.switch_to_block(exception_block);
    // Cold: a real fault (TLB miss, address error, bus error) is the rare
    // path for every memory access this emits into — Cranelift moves cold
    // blocks out of the hot straight-line path, so the branch-not-taken case
    // (no fault) stays branch-predictor- and icache-friendly. Every other
    // exception/trap-exit block this module creates gets the same
    // treatment, for the same reason — see this comment's siblings at each
    // other `create_block()` site feeding into `emit_exception_exit`.
    ctx.builder.set_cold_block(exception_block);
    ctx.builder.seal_block(exception_block);
    emit_exception_exit(ctx, exc);

    ctx.builder.switch_to_block(retry_block);
    // Cold: a transient bus-busy retry (not a hard fault, but still the rare
    // case relative to plain success).
    ctx.builder.set_cold_block(retry_block);
    ctx.builder.seal_block(retry_block);
    // EXEC_BREAKPOINT (a real memory watchpoint hit, or — under jitv2_lockstep
    // — a lockstep_jit_read/write divergence) must NOT go through emit_bail:
    // emit_bail's target (the shared exit_block) unconditionally returns
    // EXEC_COMPLETE regardless of what status sent it there (see
    // emit_exit_block_body), so the breakpoint status would be silently
    // discarded and the run loop would never stop — it would just re-dispatch
    // this same compiled unit fresh next step() (harmless-looking for a
    // watchpoint whose condition re-fires, but for a lockstep divergence it
    // re-runs the same already-diverged instruction with no diagnostic ever
    // reaching the caller). Hard-return EXEC_BREAKPOINT directly instead, same
    // as the exception path's terminator — core.pc is already this
    // instruction's own address (loads/stores never advance it), so the
    // monitor lands exactly here.
    let is_breakpoint = ctx.builder.ins().icmp_imm_s(IntCC::Equal, exc, crate::mips_exec::EXEC_BREAKPOINT as i64);
    let breakpoint_block = ctx.builder.create_block();
    let true_retry_block = ctx.builder.create_block();
    ctx.builder.ins().brif(is_breakpoint, breakpoint_block, &[], true_retry_block, &[]);

    ctx.builder.switch_to_block(breakpoint_block);
    ctx.builder.set_cold_block(breakpoint_block);
    ctx.builder.seal_block(breakpoint_block);
    let bp_status = ctx.builder.ins().iconst(ir::types::I32, crate::mips_exec::EXEC_BREAKPOINT as i64);
    ctx.builder.ins().return_(&[bp_status]);

    ctx.builder.switch_to_block(true_retry_block);
    ctx.builder.set_cold_block(true_retry_block);
    ctx.builder.seal_block(true_retry_block);
    emit_bail(ctx, ctx.exit_block, ctx.word);

    ctx.builder.switch_to_block(continue_block);
    ctx.builder.seal_block(continue_block);
}

/// Call `core.fpu_set_mode_fn(core.jit_ctx, rm)` — mirrors
/// `MipsCore::write_fpu_control`'s `platform::set_fpu_mode(rm)` call on an
/// FCSR (reg 31) write. `rm` is a runtime `Value` (the low 2 bits of the
/// newly-written FCSR), not a compile-time constant.
///
/// `AbiParam::new(I32)` here is narrower than `emit_mem_write`'s value
/// parameter (always I64 — see that function's doc comment for why: the
/// SysV C ABI doesn't guarantee a caller zero-extends a sub-word integer
/// argument, and a live bug from exactly that gap was found and fixed for
/// I8/I16 write values). This one's been left I32 rather than widened
/// defensively too: unlike I8/I16, a 32-bit x86-64 ALU/mov destination
/// register is architecturally zero-extended to 64 bits as a side effect,
/// which is almost certainly why this specific width never showed the same
/// corruption in practice (`jit_fpu_set_mode`'s `rm as u8` truncation would
/// have surfaced it identically if it existed here). Worth revisiting if
/// `rm` is ever seen wrong — the fix would be the same widen-to-I64 pattern.
fn emit_fpu_set_mode(ctx: &mut EmitCtx, rm: Value) {
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();

    let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
    let jit_ctx = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, jit_ctx_off);

    let fn_off = ir::immediates::Offset32::new(core_offset_of_fpu_set_mode_fn());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);

    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty)); // jit_ctx
    sig.params.push(AbiParam::new(ir::types::I32)); // rm
    let sig_ref = ctx.builder.import_signature(sig);
    ctx.builder.ins().call_indirect(sig_ref, callee, &[jit_ctx, rm]);
}

/// MFC1 rt, fs: rt = sign_extend32(fpr_w[fs]). Mirrors `exec_mfc1`.
fn emit_mfc1(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let rt = field_rt(ctx.raw);
    let bits = emit_read_fpr_w(ctx, fs, fr_mode);
    let sext = ctx.builder.ins().sextend(ir::types::I64, bits);
    emit_write_gpr(ctx, rt, sext);
}

/// DMFC1 rt, fs: rt = fpr_l[fs] (full 64 bits). Mirrors `exec_dmfc1`.
fn emit_dmfc1(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let rt = field_rt(ctx.raw);
    let value = emit_read_fpr_l(ctx, fs, fr_mode);
    emit_write_gpr(ctx, rt, value);
}

/// MTC1 rt, fs: fpr_w[fs] = rt[31:0]. Mirrors `exec_mtc1`.
fn emit_mtc1(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let rt = field_rt(ctx.raw);
    let rt_val = emit_read_gpr(ctx, rt);
    let rt_32 = ctx.builder.ins().ireduce(ir::types::I32, rt_val);
    emit_write_fpr_w(ctx, fs, rt_32, fr_mode);
}

/// DMTC1 rt, fs: fpr_l[fs] = rt (full 64 bits). Mirrors `exec_dmtc1`.
fn emit_dmtc1(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let rt = field_rt(ctx.raw);
    let rt_val = emit_read_gpr(ctx, rt);
    emit_write_fpr_l(ctx, fs, rt_val, fr_mode);
}

/// CFC1 rt, fs: rt = sign_extend32(read_fpu_control(fs)). Mirrors
/// `exec_cfc1`/`MipsCore::read_fpu_control` — reg 0=FIR, 25=FCCR (packed
/// from FCSR's condition-code bits), 26=FEXR, 28=FENR, 31=FCSR; anything
/// else reads 0. `fs` is a compile-time constant (part of the fixed
/// instruction encoding), so this compiles to a single load of whichever
/// field `fs` selects — no runtime branch needed.
fn emit_cfc1(ctx: &mut EmitCtx, _fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let rt = field_rt(ctx.raw);
    let mem = MemFlagsData::trusted();

    let value = match fs {
        0 => ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, ir::immediates::Offset32::new(core_offset_of_fpu_fir())),
        25 => {
            // fccr_from_fcsr: cc0 = fcsr[23], cc1..7 = fcsr[25:31] -> packed as cc0 | (cc1_7 << 1)
            let fcsr = ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, ir::immediates::Offset32::new(core_offset_of_fpu_fcsr()));
            let cc0 = ctx.builder.ins().ushr_imm_s(fcsr, 23);
            let cc0 = ctx.builder.ins().band_imm_s(cc0, 1);
            let cc1_7 = ctx.builder.ins().ushr_imm_s(fcsr, 25);
            let cc1_7 = ctx.builder.ins().band_imm_s(cc1_7, 0x7F);
            let cc1_7_shifted = ctx.builder.ins().ishl_imm_s(cc1_7, 1);
            ctx.builder.ins().bor(cc0, cc1_7_shifted)
        }
        26 => ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, ir::immediates::Offset32::new(core_offset_of_fpu_fexr())),
        28 => ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, ir::immediates::Offset32::new(core_offset_of_fpu_fenr())),
        31 => ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, ir::immediates::Offset32::new(core_offset_of_fpu_fcsr())),
        _ => ctx.builder.ins().iconst(ir::types::I32, 0),
    };
    let sext = ctx.builder.ins().sextend(ir::types::I64, value);
    emit_write_gpr(ctx, rt, sext);
}

/// CTC1 rt, fs: write_fpu_control(fs, rt[31:0]), mirroring
/// `exec_ctc1`/`MipsCore::write_fpu_control` exactly: reg 0 (FIR) is
/// read-only (no-op); reg 25 (FCCR) scatters into FCSR's condition-code
/// bits; reg 26/28 (FEXR/FENR) are plain stores; reg 31 (FCSR) stores the
/// whole value, re-derives FCCR from it, reprograms the host rounding mode
/// (`emit_fpu_set_mode`), and — the one path with control flow — re-checks
/// pending cause bits against enables and raises `EXC_FPE` immediately if
/// warranted (`(fcsr & FCSR_CE) != 0 || ((fcsr & FCSR_CM) >> 5) & (fcsr &
/// FCSR_EM) != 0`). `fs` is compile-time constant, so which case applies is
/// resolved at compile time — no runtime branch on `fs` itself, only
/// (for reg 31) on the pending-exception check.
fn emit_ctc1(ctx: &mut EmitCtx, _fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let rt = field_rt(ctx.raw);
    let mem = MemFlagsData::trusted();

    let rt_val = emit_read_gpr(ctx, rt);
    let rt_32 = ctx.builder.ins().ireduce(ir::types::I32, rt_val);

    match fs {
        0 => {} // FIR read-only, write ignored
        25 => {
            let fccr_off = ir::immediates::Offset32::new(core_offset_of_fpu_fccr());
            let value_masked = ctx.builder.ins().band_imm_s(rt_32, 0xFF);
            ctx.builder.ins().store(mem, value_masked, ctx.core_ptr, fccr_off);
            // fcsr_with_fccr: scatter cc0/cc1..7 back into FCSR's cc bits.
            let fcsr_off = ir::immediates::Offset32::new(core_offset_of_fpu_fcsr());
            let fcsr = ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, fcsr_off);
            let cleared = ctx.builder.ins().band_imm_s(fcsr, !((1i64 << 23) | (0x7Fi64 << 25)));
            let cc0 = ctx.builder.ins().band_imm_s(value_masked, 1);
            let cc0_shifted = ctx.builder.ins().ishl_imm_s(cc0, 23);
            let cc1_7 = ctx.builder.ins().ushr_imm_s(value_masked, 1);
            let cc1_7 = ctx.builder.ins().band_imm_s(cc1_7, 0x7F);
            let cc1_7_shifted = ctx.builder.ins().ishl_imm_s(cc1_7, 25);
            let with_cc0 = ctx.builder.ins().bor(cleared, cc0_shifted);
            let new_fcsr = ctx.builder.ins().bor(with_cc0, cc1_7_shifted);
            ctx.builder.ins().store(mem, new_fcsr, ctx.core_ptr, fcsr_off);
        }
        26 => {
            let off = ir::immediates::Offset32::new(core_offset_of_fpu_fexr());
            ctx.builder.ins().store(mem, rt_32, ctx.core_ptr, off);
        }
        28 => {
            let off = ir::immediates::Offset32::new(core_offset_of_fpu_fenr());
            ctx.builder.ins().store(mem, rt_32, ctx.core_ptr, off);
        }
        31 => {
            let fcsr_off = ir::immediates::Offset32::new(core_offset_of_fpu_fcsr());
            ctx.builder.ins().store(mem, rt_32, ctx.core_ptr, fcsr_off);
            // fccr_from_fcsr(value) -> fpu_fccr
            let cc0 = ctx.builder.ins().ushr_imm_s(rt_32, 23);
            let cc0 = ctx.builder.ins().band_imm_s(cc0, 1);
            let cc1_7 = ctx.builder.ins().ushr_imm_s(rt_32, 25);
            let cc1_7 = ctx.builder.ins().band_imm_s(cc1_7, 0x7F);
            let cc1_7_shifted = ctx.builder.ins().ishl_imm_s(cc1_7, 1);
            let fccr = ctx.builder.ins().bor(cc0, cc1_7_shifted);
            let fccr_off = ir::immediates::Offset32::new(core_offset_of_fpu_fccr());
            ctx.builder.ins().store(mem, fccr, ctx.core_ptr, fccr_off);

            let rm = ctx.builder.ins().band_imm_s(rt_32, 0x3);
            emit_fpu_set_mode(ctx, rm);

            // Pending-cause-vs-enabled recheck (mirrors exec_ctc1 exactly).
            const FCSR_CE: i64 = 0x0002_0000;
            const FCSR_CM: i64 = 0x0001_f000;
            const FCSR_EM: i64 = 0x0000_0f80;
            let ce_set = ctx.builder.ins().band_imm_s(rt_32, FCSR_CE);
            let ce_nonzero = ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, ce_set, 0);
            let cm = ctx.builder.ins().band_imm_s(rt_32, FCSR_CM);
            let cm_shifted = ctx.builder.ins().ushr_imm_s(cm, 5);
            let em = ctx.builder.ins().band_imm_s(rt_32, FCSR_EM);
            let cause_and_enable = ctx.builder.ins().band(cm_shifted, em);
            let cause_nonzero = ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, cause_and_enable, 0);
            let should_raise = ctx.builder.ins().bor(ce_nonzero, cause_nonzero);

            let raise_block = ctx.builder.create_block();
            let continue_block = ctx.builder.create_block();
            ctx.builder.ins().brif(should_raise, raise_block, &[], continue_block, &[]);

            // Cold: an FCSR write that immediately re-triggers a pending
            // unmasked exception is a rare, deliberate case.
            ctx.builder.switch_to_block(raise_block);
            ctx.builder.set_cold_block(raise_block);
            ctx.builder.seal_block(raise_block);
            let status = crate::mips_exec::exec_exception_const(crate::mips_exec::EXC_FPE);
            let status_val = ctx.builder.ins().iconst(ir::types::I32, status as i64);
            emit_exception_exit(ctx, status_val);

            ctx.builder.switch_to_block(continue_block);
            ctx.builder.seal_block(continue_block);
        }
        _ => {} // undefined registers ignored
    }
}

/// After an FP arithmetic op has computed its result but *before* it is
/// committed: fold in caller-computed exception flags (never read from the
/// host FPU — see `rules/jitv2/fpu-flags-are-computed-not-read.md`), update
/// `core.fpu_fcsr`'s Cause bits (rewritten every instruction, never OR'd —
/// R4000/VR5000 manuals: "the results of only one instruction"), and either
/// commit the result via `write_result` and raise `EXC_FPE` if any enabled
/// exception fired, or — if trapping — leave the destination register and
/// the sticky Flag field untouched, mirroring `MipsExecutor::fpu_update_fcsr`/
/// `fpu_update_fcsr_full` exactly (same bit math, same commit-only-if-not-
/// trapping shape). Must be called with `builder` positioned in the block
/// holding the FP op just computed, before its result has been written.
/// Leaves `builder` positioned in a new, sealed continuation block if no
/// exception fired — callers continue emitting there (their block's
/// fallthrough/exit wiring happens exactly like a plain `Sequential`
/// instruction's, from the caller's perspective).
///
/// `flags`: an `I32` SSA value holding bits [6:2] (FV,FZ,FO,FU,FI) this
/// instruction raised, computed by the caller in IR from operand/result bit
/// patterns — e.g. `emit_fpu_arith_flags_snan_only_s/d` for a signalling-NaN
/// operand, `emit_fpu_arith_flags_div_s/d` for divide-by-zero. Always an
/// `iconst 0` for plain ADD/SUB/MUL unless a check applies — Inexact and
/// Overflow are deliberately never set here, matching the interpreter.
fn emit_fpu_update_fcsr(ctx: &mut EmitCtx, flags: Value, write_result: impl Fn(&mut EmitCtx)) {
    let mem = MemFlagsData::trusted();
    let i32t = ir::types::I32;

    const FCSR_CM: i64 = 0x0001_f000;
    const FCSR_EM: i64 = 0x0000_0f80;
    const FCSR_FM: i64 = 0x0000_007c;

    // Cause holds only the last instruction's exceptions and is rewritten
    // every FP instruction, unlike the sticky Flag field — clear it
    // unconditionally, even when this instruction raised nothing.
    let fcsr_off = ir::immediates::Offset32::new(core_offset_of_fpu_fcsr());
    let fcsr0 = ctx.builder.ins().load(i32t, mem, ctx.core_ptr, fcsr_off);
    let fcsr_cause_cleared = ctx.builder.ins().band_imm_s(fcsr0, !FCSR_CM);

    let zero = ctx.builder.ins().iconst(i32t, 0);
    let has_flags = ctx.builder.ins().icmp(IntCC::NotEqual, flags, zero);

    let update_block = ctx.builder.create_block();
    let no_flags_block = ctx.builder.create_block();
    let continue_block = ctx.builder.create_block();
    ctx.builder.ins().brif(has_flags, update_block, &[], no_flags_block, &[]);

    // No exception this instruction: Cause is still rewritten to zero above,
    // just commit the result and continue.
    ctx.builder.switch_to_block(no_flags_block);
    ctx.builder.seal_block(no_flags_block);
    ctx.builder.ins().store(mem, fcsr_cause_cleared, ctx.core_ptr, fcsr_off);
    write_result(ctx);
    ctx.builder.ins().jump(continue_block, &[]);

    ctx.builder.switch_to_block(update_block);
    ctx.builder.seal_block(update_block);

    // causes = (flags & FCSR_FM) << 10
    let flags_fm = ctx.builder.ins().band_imm_s(flags, FCSR_FM);
    let causes = ctx.builder.ins().ishl_imm_s(flags_fm, 10);
    let fcsr1 = ctx.builder.ins().bor(fcsr_cause_cleared, causes);
    ctx.builder.ins().store(mem, fcsr1, ctx.core_ptr, fcsr_off);

    // Raise FPE if any cause bit has its corresponding enable bit set.
    // Causes are 5 bits above enables: (causes >> 5) aligns them with enables.
    let causes_shifted = ctx.builder.ins().ushr_imm_s(causes, 5);
    let fcsr1_em = ctx.builder.ins().band_imm_s(fcsr1, FCSR_EM);
    let enabled_cause = ctx.builder.ins().band(causes_shifted, fcsr1_em);
    let has_enabled_cause = ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, enabled_cause, 0);

    let raise_block = ctx.builder.create_block();
    let no_raise_block = ctx.builder.create_block();
    ctx.builder.ins().brif(has_enabled_cause, raise_block, &[], no_raise_block, &[]);

    // Cold: most FP ops don't raise an enabled exception cause.
    ctx.builder.switch_to_block(raise_block);
    ctx.builder.set_cold_block(raise_block);
    ctx.builder.seal_block(raise_block);
    // Trapped: neither the destination register nor the sticky Flag field
    // are touched — only Cause (already set above).
    let status = crate::mips_exec::exec_exception_const(crate::mips_exec::EXC_FPE);
    let status_val = ctx.builder.ins().iconst(i32t, status as i64);
    emit_exception_exit(ctx, status_val);

    // Not trapping: commit the result and accumulate the sticky Flag field.
    ctx.builder.switch_to_block(no_raise_block);
    ctx.builder.seal_block(no_raise_block);
    write_result(ctx);
    let fcsr2 = ctx.builder.ins().load(i32t, mem, ctx.core_ptr, fcsr_off);
    let fcsr3 = ctx.builder.ins().bor(fcsr2, flags_fm);
    ctx.builder.ins().store(mem, fcsr3, ctx.core_ptr, fcsr_off);
    ctx.builder.ins().jump(continue_block, &[]);

    ctx.builder.switch_to_block(continue_block);
    ctx.builder.seal_block(continue_block);
}

/// Unimplemented Operation (Cause.E): R4000/VR5000 manuals — no Enable or
/// Flag bit, always traps, destination register untouched. Mirrors
/// `MipsExecutor::fpu_unimplemented`. Terminates the current block; caller
/// does not get control back (matches `emit_exception_exit`'s shape).
fn emit_fpu_unimplemented(ctx: &mut EmitCtx) {
    let mem = MemFlagsData::trusted();
    let i32t = ir::types::I32;
    const FCSR_CM: i64 = 0x0001_f000;
    const FCSR_CE: i64 = 0x0002_0000;
    let fcsr_off = ir::immediates::Offset32::new(core_offset_of_fpu_fcsr());
    let fcsr = ctx.builder.ins().load(i32t, mem, ctx.core_ptr, fcsr_off);
    let fcsr_cleared = ctx.builder.ins().band_imm_s(fcsr, !FCSR_CM);
    let fcsr_with_e = ctx.builder.ins().bor_imm_s(fcsr_cleared, FCSR_CE);
    ctx.builder.ins().store(mem, fcsr_with_e, ctx.core_ptr, fcsr_off);
    let status = crate::mips_exec::exec_exception_const(crate::mips_exec::EXC_FPE);
    let status_val = ctx.builder.ins().iconst(i32t, status as i64);
    emit_exception_exit(ctx, status_val);
}

/// ABS.fmt/NEG.fmt: unlike a MOV, these are arithmetic operations (R4000
/// manual: "absolute value (ABS) and negate (NEG) ... cause this exception
/// if one or both operands is a signaling NaN") — mirrors
/// `MipsExecutor::fpu_check_snan_operand` exactly: sets Cause.V always on a
/// signalling operand, traps (destination/Flag untouched) if EV is enabled,
/// otherwise commits `write_result` and sets sticky Flag.V too.
fn emit_check_snan_operand(ctx: &mut EmitCtx, bits: Value, is_d: bool, write_result: impl Fn(&mut EmitCtx)) {
    let mem = MemFlagsData::trusted();
    let i32t = ir::types::I32;
    const FCSR_CM: i64 = 0x0001_f000;
    const FCSR_CV: i64 = 0x0001_0000;
    let is_snan = if is_d { emit_is_snan_d(ctx, bits) } else { emit_is_snan_s(ctx, bits) };

    let fcsr_off = ir::immediates::Offset32::new(core_offset_of_fpu_fcsr());
    let fcsr = ctx.builder.ins().load(i32t, mem, ctx.core_ptr, fcsr_off);
    let fcsr_cleared = ctx.builder.ins().band_imm_s(fcsr, !FCSR_CM);
    let cv_bit = ctx.builder.ins().uextend(i32t, is_snan);
    let cv_bit = ctx.builder.ins().ishl_imm_s(cv_bit, 16); // FCSR_CV = 1<<16
    let fcsr_with_cause = ctx.builder.ins().bor(fcsr_cleared, cv_bit);
    ctx.builder.ins().store(mem, fcsr_with_cause, ctx.core_ptr, fcsr_off);

    let snap_block = ctx.builder.create_block();
    let no_snan_block = ctx.builder.create_block();
    ctx.builder.ins().brif(is_snan, snap_block, &[], no_snan_block, &[]);

    // Cold: a signalling-NaN operand to ABS/NEG is a rare edge case.
    ctx.builder.switch_to_block(snap_block);
    ctx.builder.set_cold_block(snap_block);
    ctx.builder.seal_block(snap_block);
    let ev_set = ctx.builder.ins().band_imm_s(fcsr_with_cause, 0x800);
    let ev_nonzero = ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, ev_set, 0);
    let raise_block = ctx.builder.create_block();
    let untrapped_snan_block = ctx.builder.create_block();
    ctx.builder.ins().brif(ev_nonzero, raise_block, &[], untrapped_snan_block, &[]);

    ctx.builder.switch_to_block(raise_block);
    ctx.builder.seal_block(raise_block);
    let status = crate::mips_exec::exec_exception_const(crate::mips_exec::EXC_FPE);
    let status_val = ctx.builder.ins().iconst(i32t, status as i64);
    emit_exception_exit(ctx, status_val);

    let continue_block = ctx.builder.create_block();

    ctx.builder.switch_to_block(untrapped_snan_block);
    ctx.builder.seal_block(untrapped_snan_block);
    write_result(ctx);
    let fcsr2 = ctx.builder.ins().load(i32t, mem, ctx.core_ptr, fcsr_off);
    let fcsr3 = ctx.builder.ins().bor_imm_s(fcsr2, 0x40); // sticky Flag V
    ctx.builder.ins().store(mem, fcsr3, ctx.core_ptr, fcsr_off);
    ctx.builder.ins().jump(continue_block, &[]);

    ctx.builder.switch_to_block(no_snan_block);
    ctx.builder.seal_block(no_snan_block);
    write_result(ctx);
    ctx.builder.ins().jump(continue_block, &[]);

    ctx.builder.switch_to_block(continue_block);
    ctx.builder.seal_block(continue_block);
}

/// `bits` is a signalling NaN: exponent all-1s, mantissa nonzero, mantissa
/// MSB (the "is quiet" bit) clear. Mirrors `mips_exec.rs`'s `is_snan_s`/`_d`.
fn emit_is_snan_s(ctx: &mut EmitCtx, bits: Value) -> Value {
    let b = &mut ctx.builder;
    let exp_is_max = b.ins().band_imm_s(bits, 0x7F80_0000);
    let exp_is_max = b.ins().icmp_imm_s(IntCC::Equal, exp_is_max, 0x7F80_0000);
    let mantissa_nonzero = b.ins().band_imm_s(bits, 0x007F_FFFF);
    let mantissa_nonzero = b.ins().icmp_imm_s(IntCC::NotEqual, mantissa_nonzero, 0);
    let is_nan = b.ins().band(exp_is_max, mantissa_nonzero);
    let quiet_bit = b.ins().band_imm_s(bits, 0x0040_0000);
    let quiet_bit_clear = b.ins().icmp_imm_s(IntCC::Equal, quiet_bit, 0);
    b.ins().band(is_nan, quiet_bit_clear)
}
fn emit_is_snan_d(ctx: &mut EmitCtx, bits: Value) -> Value {
    let b = &mut ctx.builder;
    let exp_is_max = b.ins().band_imm_s(bits, 0x7FF0_0000_0000_0000u64 as i64);
    let exp_is_max = b.ins().icmp_imm_s(IntCC::Equal, exp_is_max, 0x7FF0_0000_0000_0000u64 as i64);
    let mantissa_nonzero = b.ins().band_imm_s(bits, 0x000F_FFFF_FFFF_FFFFu64 as i64);
    let mantissa_nonzero = b.ins().icmp_imm_s(IntCC::NotEqual, mantissa_nonzero, 0);
    let is_nan = b.ins().band(exp_is_max, mantissa_nonzero);
    let quiet_bit = b.ins().band_imm_s(bits, 0x0008_0000_0000_0000u64 as i64);
    let quiet_bit_clear = b.ins().icmp_imm_s(IntCC::Equal, quiet_bit, 0);
    b.ins().band(is_nan, quiet_bit_clear)
}

/// `bits` is a quiet NaN (any NaN that is not a signalling NaN) — mirrors
/// `mips_exec.rs`'s `is_qnan_s`/`_d`. Used by `emit_fpu_arith_flags_sqrt_s/d`
/// to exclude qNaN from the negative-operand test, mirroring the
/// interpreter's `is_neg_nonzero_s/d` exactly. RECIP/RSQRT have no JIT
/// codegen (they always fall back to the interpreter, so this path is only
/// ever reached via SQRT, which already traps qNaN upstream through
/// `emit_check_denorm_operand`) — the exclusion is dead in practice today
/// but kept for exact parity with the interpreter helper it mirrors, and in
/// case RECIP/RSQRT ever gain JIT codegen (see `emit_fpu_arith_flags_div_s/d`'s
/// doc comment on the interpreter side).
fn emit_is_qnan_s(ctx: &mut EmitCtx, bits: Value) -> Value {
    let is_snan = emit_is_snan_s(ctx, bits);
    let exp_is_max = {
        let e = ctx.builder.ins().band_imm_s(bits, 0x7F80_0000);
        ctx.builder.ins().icmp_imm_s(IntCC::Equal, e, 0x7F80_0000)
    };
    let mantissa_nonzero = {
        let m = ctx.builder.ins().band_imm_s(bits, 0x007F_FFFF);
        ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, m, 0)
    };
    let is_nan = ctx.builder.ins().band(exp_is_max, mantissa_nonzero);
    let not_snan = ctx.builder.ins().bnot(is_snan);
    ctx.builder.ins().band(is_nan, not_snan)
}
fn emit_is_qnan_d(ctx: &mut EmitCtx, bits: Value) -> Value {
    let is_snan = emit_is_snan_d(ctx, bits);
    let exp_is_max = {
        let e = ctx.builder.ins().band_imm_s(bits, 0x7FF0_0000_0000_0000u64 as i64);
        ctx.builder.ins().icmp_imm_s(IntCC::Equal, e, 0x7FF0_0000_0000_0000u64 as i64)
    };
    let mantissa_nonzero = {
        let m = ctx.builder.ins().band_imm_s(bits, 0x000F_FFFF_FFFF_FFFFu64 as i64);
        ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, m, 0)
    };
    let is_nan = ctx.builder.ins().band(exp_is_max, mantissa_nonzero);
    let not_snan = ctx.builder.ins().bnot(is_snan);
    ctx.builder.ins().band(is_nan, not_snan)
}

/// R4000/VR5000 manuals: a denormalized or quiet-NaN operand to a
/// computational instruction is Unimplemented Operation (excepting Compare
/// and Moves). Mirrors `MipsExecutor::fpu_check_denorm_operand_s`/`_d`. Call
/// once per source operand before computing; if this returns and leaves
/// `ctx.builder` in a new block, the check passed and computation may
/// proceed there — callers should structure the two-operand case as
/// sequential checks (fs then ft), each potentially terminating the region.
fn emit_check_denorm_operand(ctx: &mut EmitCtx, bits: Value, is_d: bool) {
    let is_denorm = if is_d {
        emit_is_subnormal_or_qnan_d(ctx, bits)
    } else {
        emit_is_subnormal_or_qnan_s(ctx, bits)
    };
    let bad_block = ctx.builder.create_block();
    let ok_block = ctx.builder.create_block();
    ctx.builder.ins().brif(is_denorm, bad_block, &[], ok_block, &[]);

    // Cold: a denormal/qNaN operand is a rare edge case.
    ctx.builder.switch_to_block(bad_block);
    ctx.builder.set_cold_block(bad_block);
    ctx.builder.seal_block(bad_block);
    emit_fpu_unimplemented(ctx);

    ctx.builder.switch_to_block(ok_block);
    ctx.builder.seal_block(ok_block);
}

/// `bits` is subnormal-nonzero or a quiet NaN, single precision. `is_snan_s`
/// mirrors `mips_exec.rs`'s bit-pattern helper; subnormal is exponent==0 and
/// mantissa!=0 (zero itself, exponent==0 mantissa==0, is explicitly not
/// denormal and must compute normally — e.g. `0.0 + 0.0`).
fn emit_is_subnormal_or_qnan_s(ctx: &mut EmitCtx, bits: Value) -> Value {
    let b = &mut ctx.builder;
    let exp = b.ins().band_imm_s(bits, 0x7F80_0000);
    let exp_is_zero = b.ins().icmp_imm_s(IntCC::Equal, exp, 0);
    let mantissa = b.ins().band_imm_s(bits, 0x007F_FFFF);
    let mantissa_nonzero = b.ins().icmp_imm_s(IntCC::NotEqual, mantissa, 0);
    let is_subnormal = b.ins().band(exp_is_zero, mantissa_nonzero);

    let exp_is_max = b.ins().icmp_imm_s(IntCC::Equal, exp, 0x7F80_0000);
    let is_nan = b.ins().band(exp_is_max, mantissa_nonzero);
    let quiet_bit = b.ins().band_imm_s(bits, 0x0040_0000);
    let quiet_bit_set = b.ins().icmp_imm_s(IntCC::NotEqual, quiet_bit, 0);
    let is_qnan = b.ins().band(is_nan, quiet_bit_set);

    b.ins().bor(is_subnormal, is_qnan)
}
fn emit_is_subnormal_or_qnan_d(ctx: &mut EmitCtx, bits: Value) -> Value {
    let b = &mut ctx.builder;
    let exp = b.ins().band_imm_s(bits, 0x7FF0_0000_0000_0000u64 as i64);
    let exp_is_zero = b.ins().icmp_imm_s(IntCC::Equal, exp, 0);
    let mantissa = b.ins().band_imm_s(bits, 0x000F_FFFF_FFFF_FFFFu64 as i64);
    let mantissa_nonzero = b.ins().icmp_imm_s(IntCC::NotEqual, mantissa, 0);
    let is_subnormal = b.ins().band(exp_is_zero, mantissa_nonzero);

    let exp_is_max = b.ins().icmp_imm_s(IntCC::Equal, exp, 0x7FF0_0000_0000_0000u64 as i64);
    let is_nan = b.ins().band(exp_is_max, mantissa_nonzero);
    let quiet_bit = b.ins().band_imm_s(bits, 0x0008_0000_0000_0000u64 as i64);
    let quiet_bit_set = b.ins().icmp_imm_s(IntCC::NotEqual, quiet_bit, 0);
    let is_qnan = b.ins().band(is_nan, quiet_bit_set);

    b.ins().bor(is_subnormal, is_qnan)
}

/// Result-only half: subnormal-and-nonzero, no NaN check (a computed
/// arithmetic result's denormal-ness is what finding 7's Table 9-1 gates on;
/// zero itself, exponent==0 mantissa==0, is explicitly not denormal).
fn emit_is_subnormal_s(ctx: &mut EmitCtx, bits: Value) -> Value {
    let b = &mut ctx.builder;
    let exp = b.ins().band_imm_s(bits, 0x7F80_0000);
    let exp_is_zero = b.ins().icmp_imm_s(IntCC::Equal, exp, 0);
    let mantissa = b.ins().band_imm_s(bits, 0x007F_FFFF);
    let mantissa_nonzero = b.ins().icmp_imm_s(IntCC::NotEqual, mantissa, 0);
    b.ins().band(exp_is_zero, mantissa_nonzero)
}
fn emit_is_subnormal_d(ctx: &mut EmitCtx, bits: Value) -> Value {
    let b = &mut ctx.builder;
    let exp = b.ins().band_imm_s(bits, 0x7FF0_0000_0000_0000u64 as i64);
    let exp_is_zero = b.ins().icmp_imm_s(IntCC::Equal, exp, 0);
    let mantissa = b.ins().band_imm_s(bits, 0x000F_FFFF_FFFF_FFFFu64 as i64);
    let mantissa_nonzero = b.ins().icmp_imm_s(IntCC::NotEqual, mantissa, 0);
    b.ins().band(exp_is_zero, mantissa_nonzero)
}

/// Result-side half of finding 7: after computing (not yet writing) an
/// ADD/SUB/MUL/DIV/SQRT result, check whether it's a denormalized nonzero
/// value and apply the R4000/VR5000 manuals' Table 9-1/7-1 Underflow rule —
/// mirrors `MipsExecutor::fpu_update_fcsr_full`'s `result_is_denorm` handling
/// exactly (FS clear, or FS set with U/I enabled: Unimplemented Operation,
/// untouched destination; FS set with neither enabled: flush to a signed
/// zero of `result_is_negative` and force Cause.U+I, no trap). When the
/// result is not denormal, falls through to the ordinary
/// `emit_fpu_update_fcsr` path (`flags` — see that function's doc comment).
/// `write_result`/`write_zero` are each called on exactly one path — the
/// real result when not flushing, a signed zero of the given width when
/// flushing.
fn emit_fpu_update_fcsr_arith(
    ctx: &mut EmitCtx,
    flags: Value,
    result_is_denorm: Value,
    result_is_negative: Value,
    write_result: impl Fn(&mut EmitCtx),
    write_zero: impl Fn(&mut EmitCtx, Value),
) {
    let mem = MemFlagsData::trusted();
    let i32t = ir::types::I32;
    const FCSR_FS: i64 = 0x0100_0000;
    const FCSR_UM_IM: i64 = 0x0000_0180; // Enable: U (bit 8) | I (bit 7)
    const FCSR_CU_CI: i64 = 0x0000_3000; // Cause: U (bit 13) | I (bit 12)
    const FCSR_FU_FI: i64 = 0x0000_000c; // Flag: U (bit 3) | I (bit 2)
    const FCSR_CM: i64 = 0x0001_f000;

    let denorm_block = ctx.builder.create_block();
    let normal_block = ctx.builder.create_block();
    ctx.builder.ins().brif(result_is_denorm, denorm_block, &[], normal_block, &[]);

    // Cold: a denormalized/underflowed result is a rare edge case.
    ctx.builder.switch_to_block(denorm_block);
    ctx.builder.set_cold_block(denorm_block);
    ctx.builder.seal_block(denorm_block);
    let fcsr_off = ir::immediates::Offset32::new(core_offset_of_fpu_fcsr());
    let fcsr = ctx.builder.ins().load(i32t, mem, ctx.core_ptr, fcsr_off);
    let fs_set = ctx.builder.ins().band_imm_s(fcsr, FCSR_FS);
    let fs_set = ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, fs_set, 0);
    let ui_enabled = ctx.builder.ins().band_imm_s(fcsr, FCSR_UM_IM);
    let ui_enabled = ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, ui_enabled, 0);
    let fs_clear = ctx.builder.ins().bnot(fs_set);
    let must_trap = ctx.builder.ins().bor(fs_clear, ui_enabled);

    let trap_block = ctx.builder.create_block();
    let flush_block = ctx.builder.create_block();
    ctx.builder.ins().brif(must_trap, trap_block, &[], flush_block, &[]);

    ctx.builder.switch_to_block(trap_block);
    ctx.builder.seal_block(trap_block);
    emit_fpu_unimplemented(ctx);

    ctx.builder.switch_to_block(flush_block);
    ctx.builder.seal_block(flush_block);
    let fcsr_cleared = ctx.builder.ins().band_imm_s(fcsr, !FCSR_CM);
    let fcsr_with_cause = ctx.builder.ins().bor_imm_s(fcsr_cleared, FCSR_CU_CI);
    ctx.builder.ins().store(mem, fcsr_with_cause, ctx.core_ptr, fcsr_off);
    write_zero(ctx, result_is_negative);
    let fcsr_with_flag = ctx.builder.ins().bor_imm_s(fcsr_with_cause, FCSR_FU_FI);
    ctx.builder.ins().store(mem, fcsr_with_flag, ctx.core_ptr, fcsr_off);
    let continue_block = ctx.builder.create_block();
    ctx.builder.ins().jump(continue_block, &[]);

    ctx.builder.switch_to_block(normal_block);
    ctx.builder.seal_block(normal_block);
    emit_fpu_update_fcsr(ctx, flags, &write_result);
    ctx.builder.ins().jump(continue_block, &[]);

    ctx.builder.switch_to_block(continue_block);
    ctx.builder.seal_block(continue_block);
}

/// IR counterparts of `mips_exec.rs`'s `fpu_arith_flags_*` free functions —
/// same bit-pattern checks, no host FPU status read, feeding `flags` into
/// `emit_fpu_update_fcsr`/`emit_fpu_update_fcsr_arith`. See
/// `rules/jitv2/fpu-flags-are-computed-not-read.md`.
const FCSR_FV_I64: i64 = 0x0000_0040;
const FCSR_FZ_I64: i64 = 0x0000_0020;

/// ADD/SUB/MUL (S or D): Invalid if either operand is a signalling NaN,
/// else 0 — mirrors `fpu_arith_flags_snan_only_s/d`.
fn emit_fpu_arith_flags_snan_only_s(ctx: &mut EmitCtx, fs_bits: Value, ft_bits: Value) -> Value {
    let fs_snan = emit_is_snan_s(ctx, fs_bits);
    let ft_snan = emit_is_snan_s(ctx, ft_bits);
    let any_snan = ctx.builder.ins().bor(fs_snan, ft_snan);
    let any_snan = ctx.builder.ins().uextend(ir::types::I32, any_snan);
    ctx.builder.ins().ishl_imm_s(any_snan, FCSR_FV_I64.trailing_zeros() as i64)
}
fn emit_fpu_arith_flags_snan_only_d(ctx: &mut EmitCtx, fs_bits: Value, ft_bits: Value) -> Value {
    let fs_snan = emit_is_snan_d(ctx, fs_bits);
    let ft_snan = emit_is_snan_d(ctx, ft_bits);
    let any_snan = ctx.builder.ins().bor(fs_snan, ft_snan);
    let any_snan = ctx.builder.ins().uextend(ir::types::I32, any_snan);
    ctx.builder.ins().ishl_imm_s(any_snan, FCSR_FV_I64.trailing_zeros() as i64)
}

/// DIV (S or D — the only JIT caller today; RECIP has no JIT codegen):
/// Invalid takes priority over divide-by-zero, else Z when the divisor is
/// zero — mirrors `fpu_arith_flags_div_s/d`.
fn emit_fpu_arith_flags_div_s(ctx: &mut EmitCtx, fs_bits: Value, ft_bits: Value) -> Value {
    let snan = emit_fpu_arith_flags_snan_only_s(ctx, fs_bits, ft_bits);
    let ft_zero = {
        let masked = ctx.builder.ins().band_imm_s(ft_bits, 0x7FFF_FFFF);
        ctx.builder.ins().icmp_imm_s(IntCC::Equal, masked, 0)
    };
    let z_flag = ctx.builder.ins().uextend(ir::types::I32, ft_zero);
    let z_flag = ctx.builder.ins().ishl_imm_s(z_flag, FCSR_FZ_I64.trailing_zeros() as i64);
    let snan_is_zero = ctx.builder.ins().icmp_imm_s(IntCC::Equal, snan, 0);
    ctx.builder.ins().select(snan_is_zero, z_flag, snan)
}
fn emit_fpu_arith_flags_div_d(ctx: &mut EmitCtx, fs_bits: Value, ft_bits: Value) -> Value {
    let snan = emit_fpu_arith_flags_snan_only_d(ctx, fs_bits, ft_bits);
    let ft_zero = {
        let masked = ctx.builder.ins().band_imm_s(ft_bits, 0x7FFF_FFFF_FFFF_FFFFu64 as i64);
        ctx.builder.ins().icmp_imm_s(IntCC::Equal, masked, 0)
    };
    let z_flag = ctx.builder.ins().uextend(ir::types::I32, ft_zero);
    let z_flag = ctx.builder.ins().ishl_imm_s(z_flag, FCSR_FZ_I64.trailing_zeros() as i64);
    let snan_is_zero = ctx.builder.ins().icmp_imm_s(IntCC::Equal, snan, 0);
    ctx.builder.ins().select(snan_is_zero, z_flag, snan)
}

/// SQRT (S or D — the only JIT caller today; RECIP/RSQRT have no JIT
/// codegen, see `emit_is_qnan_s`'s doc comment): Invalid for a signalling-
/// NaN operand or a negative nonzero operand, else 0 — mirrors
/// `fpu_arith_flags_sqrt_s/d`. qNaN is excluded from the negative-operand
/// test for exact parity with that interpreter helper, though it's dead
/// today since `emit_check_denorm_operand` already traps a qNaN operand
/// before this runs.
fn emit_fpu_arith_flags_sqrt_s(ctx: &mut EmitCtx, fs_bits: Value) -> Value {
    let is_snan = emit_is_snan_s(ctx, fs_bits);
    let is_qnan = emit_is_qnan_s(ctx, fs_bits);
    let sign_set = ctx.builder.ins().icmp_imm_s(IntCC::SignedLessThan, fs_bits, 0);
    let nonzero = {
        let masked = ctx.builder.ins().band_imm_s(fs_bits, 0x7FFF_FFFF);
        ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, masked, 0)
    };
    let not_nan = {
        let any_nan = ctx.builder.ins().bor(is_snan, is_qnan);
        ctx.builder.ins().bnot(any_nan)
    };
    let neg_nonzero = ctx.builder.ins().band(sign_set, nonzero);
    let neg_nonzero = ctx.builder.ins().band(neg_nonzero, not_nan);
    let invalid = ctx.builder.ins().bor(is_snan, neg_nonzero);
    let invalid = ctx.builder.ins().uextend(ir::types::I32, invalid);
    ctx.builder.ins().ishl_imm_s(invalid, FCSR_FV_I64.trailing_zeros() as i64)
}
fn emit_fpu_arith_flags_sqrt_d(ctx: &mut EmitCtx, fs_bits: Value) -> Value {
    let is_snan = emit_is_snan_d(ctx, fs_bits);
    let is_qnan = emit_is_qnan_d(ctx, fs_bits);
    let sign_set = ctx.builder.ins().icmp_imm_s(IntCC::SignedLessThan, fs_bits, 0);
    let nonzero = {
        let masked = ctx.builder.ins().band_imm_s(fs_bits, 0x7FFF_FFFF_FFFF_FFFFu64 as i64);
        ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, masked, 0)
    };
    let not_nan = {
        let any_nan = ctx.builder.ins().bor(is_snan, is_qnan);
        ctx.builder.ins().bnot(any_nan)
    };
    let neg_nonzero = ctx.builder.ins().band(sign_set, nonzero);
    let neg_nonzero = ctx.builder.ins().band(neg_nonzero, not_nan);
    let invalid = ctx.builder.ins().bor(is_snan, neg_nonzero);
    let invalid = ctx.builder.ins().uextend(ir::types::I32, invalid);
    ctx.builder.ins().ishl_imm_s(invalid, FCSR_FV_I64.trailing_zeros() as i64)
}


/// Deliver `status` via `core.handle_exception_fn(core.jit_ctx, status)`
/// (§4.2 — the interpreter's own `handle_exception`, the only implementation
/// of EPC/Cause/BD/vectoring) and return `EXEC_COMPLETE`. Terminates the
/// current block; call as the block's terminator (like `emit_bail`). Not
/// itself sealing/switching anything afterward — there is nothing after a
/// return.
///
/// `deliver_exception` (called via `handle_exception_fn`) computes EPC
/// directly from live `core.pc` (`core.pc`, or `core.pc - 4` when
/// `core.in_delay_slot`). A plain sequential head instruction's own body
/// never writes `core.pc` per-instruction — only exit points do, to keep a
/// straight-line compiled run cheap, which is the entire point of compiling
/// it — so without correcting it here, a head instruction's fault would
/// report EPC from wherever `core.pc` was last left (the compiled unit's
/// entry, or a prior unit's exit), not the actual faulting instruction.
/// `ctx.word` always names the instruction whose body is currently being
/// emitted (kept accurate by `compile_region`'s per-head-instruction loop
/// and updated to the slot's own word by `emit_slot_semantics` while
/// compiling a delay slot's body — see that function's own `core.pc`
/// pre-write, which this mirrors: for a slot, `core.pc` is already exactly
/// `vbase | (ctx.word * 4)`, so this store is a same-value no-op; for a head
/// instruction, it's the actual fix), so `core.pc` is resynchronized to
/// `vbase | (ctx.word * 4)` before the handler runs, exactly like
/// `emit_exit_block_body`'s vbase derivation. (Found live: an ADEL fault
/// inside a compiled `lw` reported EPC pointing at an unrelated, later PROM
/// routine — whatever `core.pc` was stale with — instead of the real
/// faulting instruction.)
///
/// Exception: at `entry_word` (only there — see `EmitCtx::entry_word`'s doc
/// comment for why every other word is unambiguous), the synthesis above is
/// skipped if live `core.in_delay_slot` is true at the fault site: a
/// physical word compiled once as an ordinary entry can, on a later
/// dispatch, be entered because the interpreter's own dispatch loop landed
/// on it as some *other* branch's delay slot — `core.in_delay_slot`/
/// `core.pc` are already correct in that case (set by the interpreter via
/// `branch_delay`/`handle_exec_complete` before this compiled function was
/// even called), and overwriting `core.pc` here would discard that and make
/// `deliver_exception` compute EPC/BD as if this were an ordinary
/// (non-delay-slot) fault — wrong `Cause.BD` (should be set, would read
/// clear) and wrong EPC (should be `core.pc - 4`, i.e. the real branch's
/// address, would read this word's own address instead).
///
/// Two-stage split (see `BlockSkeleton`'s own doc comment for the full
/// rationale): `word == entry_word` is always known at compile time for any
/// given call site, so `emit_exception_exit` below picks the right outer
/// stage directly — no call site ever pays a runtime check for a fact that
/// was already fixed when it was emitted. This is the inner stage, shared
/// by every call site regardless of which word it's at: assumes
/// `core.pc`/`core.in_delay_slot` are already correct (both outer stages
/// below guarantee this before jumping here) and just delivers the
/// exception.
fn emit_exception_call_block_body(
    module: &mut dyn cranelift_module::Module,
    builder: &mut FunctionBuilder,
    core_ptr: Value,
    status: Value,
) {
    let mem = MemFlagsData::trusted();
    let ptr_ty = module.target_config().pointer_type();

    let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
    let jit_ctx = builder.ins().load(ptr_ty, mem, core_ptr, jit_ctx_off);

    let fn_off = ir::immediates::Offset32::new(core_offset_of_handle_exception_fn());
    let callee = builder.ins().load(ptr_ty, mem, core_ptr, fn_off);

    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty)); // jit_ctx
    sig.params.push(AbiParam::new(ir::types::I32)); // status
    sig.returns.push(AbiParam::new(ir::types::I32)); // ExecStatus (== status, unused)
    let sig_ref = builder.import_signature(sig);

    builder.ins().call_indirect(sig_ref, callee, &[jit_ctx, status]);
    let ret_status = builder.ins().iconst(ir::types::I32, EXEC_COMPLETE as i64);
    builder.ins().return_(&[ret_status]);
}

/// Outer stage for every non-entry-word `emit_exception_exit` call site,
/// shared across all of them (`word`/`bd` are genuine runtime params here,
/// unlike the entry-word stage — this one block really does serve many
/// different words, both delay-slot and non-delay-slot alike). Unconditionally
/// writes `core.pc = vbase | (word * 4)` and `core.in_delay_slot = bd` from
/// its own params — never trusts either field's live value on entry, so no
/// call site needs to rely on any upstream code having left them correct
/// (see `emit_exception_exit`'s doc comment for why leaving `in_delay_slot`
/// to inheritance was fragile) — then falls into the inner stage. Every call
/// site's `bd` is a compile-time-known literal (`emit_exception_exit` reads
/// `ctx.bd`): `true` only while inlined inside a delay slot's own semantics
/// (`emit_slot_semantics` sets `ctx.bd = true` before calling in), `false`
/// for every ordinary, non-slot head — including one that's independently
/// reachable as this same region's own branch/jump target (§6.1.4 dual
/// semantics), which is always a *different*, freshly-constructed `ctx` with
/// its own default `bd = false`.
fn emit_exception_other_word_block_body(
    builder: &mut FunctionBuilder,
    core_ptr: Value,
    word: Value,
    bd: Value,
    status: Value,
    call_block: Block,
) {
    let mem = MemFlagsData::trusted();
    let i64t = ir::types::I64;
    let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
    let flag_off = ir::immediates::Offset32::new(core_offset_of_in_delay_slot());

    let pc = builder.ins().load(i64t, mem, core_ptr, pc_off);
    let vbase = builder.ins().band_imm_s(pc, !(PAGE_SIZE as i64 - 1));
    let byte_offset = builder.ins().imul_imm_s(word, 4);
    let fault_pc = builder.ins().iadd(vbase, byte_offset);
    builder.ins().store(mem, fault_pc, core_ptr, pc_off);
    builder.ins().store(mem, bd, core_ptr, flag_off);
    builder.ins().jump(call_block, &[ir::BlockArg::Value(core_ptr), ir::BlockArg::Value(status)]);
}

/// Outer stage for entry-word `emit_exception_exit` call sites — see
/// `BlockSkeleton::exception_entry_word_block`'s doc comment for why
/// `entry_word` is baked in as a compile-time constant here rather than
/// threaded as a block param (there's only ever one per region).
///
/// Unconditional, same shape as `emit_exception_other_word_block_body` — no
/// runtime check. A runtime check *here* cannot work: by the time control
/// reaches this block, `core.in_delay_slot`'s live value no longer
/// distinguishes "external interpreter dispatch landed on entry_word"
/// (state already correct) from "an internal in-region branch landed on
/// entry_word" (state stale) — both are just some bit pattern in `core`,
/// with no third signal available here to tell them apart. The
/// disambiguation has to happen at the branch site instead: entry_word_block
/// (the target of every *internal* edge into entry_word, per its own doc
/// comment in `compile_region_uncommitted`) unconditionally forces
/// `core.in_delay_slot = false` and `core.pc = vbase | entry_word*4` before
/// falling into entry_word_body_block — internal edges into entry_word are
/// always ordinary fallthrough/taken-branch edges (`emit_target_edge`'s
/// `None` arm), never a delay-slot transfer, so `in_delay_slot` is always
/// `false` on that path. That leaves this block free to just assume state is
/// already correct unconditionally, exactly like the non-entry-word stage.
fn emit_exception_entry_word_block_body(
    builder: &mut FunctionBuilder,
    core_ptr: Value,
    status: Value,
    call_block: Block,
) {
    builder.ins().jump(call_block, &[ir::BlockArg::Value(core_ptr), ir::BlockArg::Value(status)]);
}

/// Jump to the region's shared exception-raise machinery instead of emitting
/// a fresh copy of the whole delay-slot-check-and-raise sequence at every
/// call site — the exception-exit counterpart of `emit_bail`. Picks
/// `ctx.exception_entry_word_block` or `ctx.exception_other_word_block` at
/// *compile* time based on `ctx.trust_live_pc_bd_on_exc` (always known when
/// emitting a given call site — see `BlockSkeleton`'s doc comment for why
/// this avoids the runtime check a single fully-shared block would need).
fn emit_exception_exit(ctx: &mut EmitCtx, status: Value) {
    if ctx.trust_live_pc_bd_on_exc {
        // entry word (state set by the interpreter dispatch that reached it)
        // or a branch-fallback successor (state set by the BC1 fallback's
        // interpreter run) — `core.pc`/`core.in_delay_slot` are already
        // correct and must NOT be overwritten from the compile-time word/bd
        // (which would clobber a slot's BD=true), so route through
        // exception_entry_word_block, which trusts the live values.
        ctx.builder.ins().jump(ctx.exception_entry_word_block, &[
            ir::BlockArg::Value(ctx.core_ptr),
            ir::BlockArg::Value(status),
        ]);
    } else {
        let word_val = ctx.builder.ins().iconst(ir::types::I64, ctx.word as i64);
        let bd_val = ctx.builder.ins().iconst(ir::types::I8, ctx.bd as i64);
        ctx.builder.ins().jump(ctx.exception_other_word_block, &[
            ir::BlockArg::Value(ctx.core_ptr),
            ir::BlockArg::Value(word_val),
            ir::BlockArg::Value(bd_val),
            ir::BlockArg::Value(status),
        ]);
    }
}

/// Exit stub for a runtime-computed target address (JR/JALR — §2.3, the
/// target is a register value, not a compile-time word offset, so this
/// can't go through the shared `exit_block`/`emit_bail`, which only knows
/// how to materialize `vbase | word_offset*4` from a compile-time constant).
/// Writes `core.pc = target_addr` directly (already a full virtual address
/// — no vbase math needed) and returns `EXEC_COMPLETE`, exactly like
/// `emit_exception_exit`'s "state is already correct, just return" shape.
/// Terminates the current block; call as the block's terminator.
fn emit_runtime_pc_exit(ctx: &mut EmitCtx, target_addr: Value) {
    let mem = MemFlagsData::trusted();
    let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
    ctx.builder.ins().store(mem, target_addr, ctx.core_ptr, pc_off);
    emit_set_jit_trigger(ctx);
    let status = ctx.builder.ins().iconst(ir::types::I32, EXEC_COMPLETE as i64);
    ctx.builder.ins().return_(&[status]);
}

/// Emit a JR/JALR unit: read the target register first (before the delay
/// slot might overwrite it — mirrors the interpreter's `exec_jr`/
/// `exec_jalr`, which reads `rs` before dispatching the delay slot via
/// `branch_delay`), write the link register if JALR, then the delay slot
/// inline (§6.1.4, always executes — RegJump is never annulling), then exit
/// via [`emit_runtime_pc_exit`] with the register value read at the start.
/// Terminates the current block.
fn emit_regjump(ctx: &mut EmitCtx, instrs: &[CompiledInstr; ENTRIES_PER_PAGE], regjump: RegJump, fr_mode: FrMode) {
    let word = ctx.word;
    let raw = ctx.raw;
    let target_addr = emit_read_gpr(ctx, field_rs(raw));

    if regjump.link {
        // JALR writes its own `rd` field, NOT always r31 (unlike J/JAL/
        // BLTZAL/BGEZAL, which are architecturally hardcoded to r31) —
        // mirrors exec_jalr's `self.core.write_gpr(rd_reg, ...)` exactly.
        let this_pc_word = word as i64 + 2; // this instruction's address + 8 bytes = +2 words
        emit_write_link_register(ctx, this_pc_word, field_rd(raw));
    }

    let slot_word = word + 1;
    // `ForeignPageSlot` (word == 1023, analyzer's `is_0xffc_branch`): no
    // `instrs[slot_word]` to inline at all (`slot_word == WORDS_PER_PAGE`,
    // one past the array) — arm the pending transfer instead of executing a
    // slot that isn't on this page, exactly like emit_branch_or_jump's own
    // `foreign_page_slot` handling.
    if slot_word as usize >= ENTRIES_PER_PAGE {
        emit_foreign_page_slot_exit(ctx, word, target_addr);
        return;
    }

    let slot_raw = instrs[slot_word as usize].raw;
    if !try_emit_fused_nop_slot(ctx, instrs, slot_word, slot_raw) {
        // Recurse into the slot's own emission with ctx.raw/ctx.word switched
        // to the slot's — restored after, mirroring emit_slot_semantics' own
        // core.pc save/restore around the same call.
        ctx.raw = slot_raw;
        ctx.word = slot_word;
        let slot_terminated = emit_slot_semantics(ctx, instrs, fr_mode, target_addr);
        ctx.raw = raw;
        ctx.word = word;
        if slot_terminated {
            // A nested branch/regjump in this slot already exited (wrote its
            // own final core.pc and returned) — this RegJump's own target
            // (read from a register *before* the slot ran, per exec_jr/
            // exec_jalr's ordering) never takes effect; the nested transfer
            // superseded it, exactly like real hardware's "innermost
            // dispatched branch_delay wins" nested-delay-slot semantics.
            return;
        }
    }

    emit_runtime_pc_exit(ctx, target_addr);
}

/// Load `core.gpr[reg]` as I64. Matches `MipsCore::read_gpr` (a plain,
/// non-atomic load — GPRs are only ever touched by the owning exec thread).
fn emit_read_gpr(ctx: &mut EmitCtx, reg: u32) -> Value {
    let mem = MemFlagsData::trusted();
    let off = ir::immediates::Offset32::new(core_offset_of_gpr(reg));
    ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, off)
}

/// Store `value` to `core.gpr[reg]`, matching `MipsCore::write_gpr`'s
/// architectural effect. The interpreter's version always stores then
/// unconditionally re-zeros `gpr[0]` to avoid a runtime branch on `reg`; here
/// `reg` is a compile-time constant (baked into each fixed-shape
/// instruction's emitter), so the equivalent is simpler: skip the store
/// entirely when `reg == 0` (gpr[0] already reads as 0 and is never written
/// by any other emitter either, so it stays 0).
fn emit_write_gpr(ctx: &mut EmitCtx, reg: u32, value: Value) {
    if reg == 0 {
        return;
    }
    let mem = MemFlagsData::trusted();
    let off = ir::immediates::Offset32::new(core_offset_of_gpr(reg));
    ctx.builder.ins().store(mem, value, ctx.core_ptr, off);
}

/// Compile-time FPU register-file addressing mode, resolved once per
/// compiled region from the live `STATUS_FR` bit (`compile_region` reads it
/// when it decides to compile a region containing CP1 instructions at all;
/// see the entry-block guard preamble, which re-checks the same bit at
/// runtime and bails if it no longer matches). Mirrors
/// `MipsExecutor::update_fpr_mode`'s fn-pointer swap — baked into which
/// emitter runs, not a runtime branch per FPR access, since FR mode cannot
/// change within a single region (any instruction that could change it,
/// MTC0, is `Excluded` and ends the region).
#[derive(Clone, Copy, PartialEq, Eq)]
enum FrMode { Fr0, Fr1 }

/// Read `reg` as a 32-bit single/word FPR value (bits, not interpreted as
/// float yet — callers do `f32::from_bits`-equivalent via `bitcast` as
/// needed). Mirrors `read_fpr_w_fr0`/`_fr1` (`mips_core.rs`): FR=0 packs two
/// registers per 64-bit slot (odd reg = upper 32 bits of the even slot);
/// FR=1 is a flat 64-bit-per-register array, low 32 bits.
fn emit_read_fpr_w(ctx: &mut EmitCtx, reg: u32, fr_mode: FrMode) -> Value {
    let mem = MemFlagsData::trusted();
    match fr_mode {
        FrMode::Fr0 => {
            let slot_off = ir::immediates::Offset32::new(core_offset_of_fpr(reg & !1));
            let slot = ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, slot_off);
            let shifted = if reg & 1 != 0 { ctx.builder.ins().ushr_imm_s(slot, 32) } else { slot };
            ctx.builder.ins().ireduce(ir::types::I32, shifted)
        }
        FrMode::Fr1 => {
            let off = ir::immediates::Offset32::new(core_offset_of_fpr(reg));
            ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, off)
        }
    }
}

/// Write `value` (32-bit single/word bits) to `reg`. Mirrors
/// `write_fpr_w_fr0`/`_fr1`: FR=0 must preserve the other half of the
/// 64-bit slot (read-modify-write with a shift+mask); FR=1 stores the low
/// 32 bits directly, leaving the slot's upper 32 bits untouched (matches
/// `MipsCore::write_fpr_w`'s `(self.fpr[reg] & 0xFFFFFFFF_00000000) | value`).
fn emit_write_fpr_w(ctx: &mut EmitCtx, reg: u32, value: Value, fr_mode: FrMode) {
    let mem = MemFlagsData::trusted();
    match fr_mode {
        FrMode::Fr0 => {
            let slot_off = ir::immediates::Offset32::new(core_offset_of_fpr(reg & !1));
            let slot = ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, slot_off);
            let value_64 = ctx.builder.ins().uextend(ir::types::I64, value);
            let result = if reg & 1 != 0 {
                let masked = ctx.builder.ins().band_imm_s(slot, 0xFFFF_FFFF);
                let shifted = ctx.builder.ins().ishl_imm_s(value_64, 32);
                ctx.builder.ins().bor(masked, shifted)
            } else {
                let masked = ctx.builder.ins().band_imm_s(slot, 0xFFFF_FFFF_0000_0000u64 as i64);
                ctx.builder.ins().bor(masked, value_64)
            };
            ctx.builder.ins().store(mem, result, ctx.core_ptr, slot_off);
        }
        FrMode::Fr1 => {
            let off = ir::immediates::Offset32::new(core_offset_of_fpr(reg));
            let slot = ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, off);
            let masked = ctx.builder.ins().band_imm_s(slot, 0xFFFF_FFFF_0000_0000u64 as i64);
            let value_64 = ctx.builder.ins().uextend(ir::types::I64, value);
            let result = ctx.builder.ins().bor(masked, value_64);
            ctx.builder.ins().store(mem, result, ctx.core_ptr, off);
        }
    }
}

/// Read `reg` as a full 64-bit double/long FPR value. Mirrors
/// `read_fpr_d_fr0`/`_l_fr0` (force even register, full slot) vs the FR=1
/// direct `read_fpr_l` (flat per-register slot, `reg` used as-is).
fn emit_read_fpr_l(ctx: &mut EmitCtx, reg: u32, fr_mode: FrMode) -> Value {
    let mem = MemFlagsData::trusted();
    let reg = if fr_mode == FrMode::Fr0 { reg & !1 } else { reg };
    let off = ir::immediates::Offset32::new(core_offset_of_fpr(reg));
    ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, off)
}

/// Write a full 64-bit double/long value to `reg`. Mirrors
/// `write_fpr_d_fr0`/`_l_fr0` (forces even) vs FR=1's direct `write_fpr_l`.
fn emit_write_fpr_l(ctx: &mut EmitCtx, reg: u32, value: Value, fr_mode: FrMode) {
    let mem = MemFlagsData::trusted();
    let reg = if fr_mode == FrMode::Fr0 { reg & !1 } else { reg };
    let off = ir::immediates::Offset32::new(core_offset_of_fpr(reg));
    ctx.builder.ins().store(mem, value, ctx.core_ptr, off);
}

/// MIPS instruction field extraction — mirrors `mips_isa.rs`'s decode
/// shapes, narrowed to just what codegen needs (register numbers). Free
/// functions on the raw `u32`, not tied to `DecodedInstr` — codegen works
/// directly from `CompiledInstr::raw` (`analyzer.rs`), the same way
/// `analyzer::classify` does, rather than dragging in the interpreter's
/// decode/dispatch machinery.
fn field_rs(raw: u32) -> u32 { (raw >> 21) & 0x1F }
fn field_rt(raw: u32) -> u32 { (raw >> 16) & 0x1F }
fn field_rd(raw: u32) -> u32 { (raw >> 11) & 0x1F }
fn field_sa(raw: u32) -> u32 { (raw >> 6) & 0x1F }

// ---- Branches and jumps ------
//
// Distinct from `SemanticsEmitter`: a branch/jump terminates its own block
// (condition test + brif/jump to the resolved target block or an exit
// bail) rather than committing data and letting the caller wire a plain
// fallthrough. `compile_region` dispatches to this path via
// `lookup_branch_or_jump` before falling back to `lookup_semantics`.

/// How a branch's condition is evaluated, or `Always` for an unconditional
/// jump (J/JAL). Mirrors the interpreter's `exec_beq`/`exec_bne`/`exec_blez`/
/// `exec_bgtz`/`exec_bltz`/`exec_bgez` predicates exactly — see `emit_cond`.
#[derive(Clone, Copy)]
enum BranchCond {
    Always,
    Eq,     // BEQ:  rs == rt
    Ne,     // BNE:  rs != rt
    LeZero, // BLEZ: rs as i64 <= 0
    GtZero, // BGTZ: rs as i64 > 0
    LtZero, // BLTZ (REGIMM rt=0): rs as i64 < 0
    GeZero, // BGEZ (REGIMM rt=1): rs as i64 >= 0
}

/// A branch or jump instruction's shape, as far as codegen cares: how the
/// condition is evaluated, whether it writes a link register, and whether
/// its delay slot is annulled on the not-taken path ("Likely" branches —
/// §4.3 "annul semantics compiled explicitly": the slot's effects must
/// never be committed when not taken, unlike a plain branch's slot which
/// always executes regardless of the outcome). Batch scope so far: plain
/// and link-writing conditional branches, their annulling Likely
/// counterparts, and J/JAL. RegJump (JR/JALR) is not produced by
/// `lookup_branch_or_jump` yet.
#[derive(Clone, Copy)]
struct BranchOrJump {
    cond: BranchCond,
    link: bool,
    annul: bool,
}

/// Look up a branch/jump's shape, or `None` if `raw` isn't one of the kinds
/// this pass supports yet (RegJump — see `compile_region`'s doc comment for
/// the full batch plan). Deliberately narrower than `analyzer::classify`'s
/// `Branch`/`Jump` cases: this only recognizes the specific opcodes each
/// batch has actually wired an emitter for, so an as-yet-unimplemented
/// shape correctly falls through to `compile_region`'s `None` rejection
/// instead of being silently mis-emitted.
fn lookup_branch_or_jump(raw: u32) -> Option<BranchOrJump> {
    use crate::mips_isa::*;
    let op = (raw >> 26) & 0x3F;
    let rt = (raw >> 16) & 0x1F;
    match op {
        OP_BEQ => Some(BranchOrJump { cond: BranchCond::Eq, link: false, annul: false }),
        OP_BNE => Some(BranchOrJump { cond: BranchCond::Ne, link: false, annul: false }),
        OP_BLEZ => Some(BranchOrJump { cond: BranchCond::LeZero, link: false, annul: false }),
        OP_BGTZ => Some(BranchOrJump { cond: BranchCond::GtZero, link: false, annul: false }),
        OP_BEQL => Some(BranchOrJump { cond: BranchCond::Eq, link: false, annul: true }),
        OP_BNEL => Some(BranchOrJump { cond: BranchCond::Ne, link: false, annul: true }),
        OP_BLEZL => Some(BranchOrJump { cond: BranchCond::LeZero, link: false, annul: true }),
        OP_BGTZL => Some(BranchOrJump { cond: BranchCond::GtZero, link: false, annul: true }),
        OP_REGIMM => match rt {
            RT_BLTZ => Some(BranchOrJump { cond: BranchCond::LtZero, link: false, annul: false }),
            RT_BGEZ => Some(BranchOrJump { cond: BranchCond::GeZero, link: false, annul: false }),
            RT_BLTZAL => Some(BranchOrJump { cond: BranchCond::LtZero, link: true, annul: false }),
            RT_BGEZAL => Some(BranchOrJump { cond: BranchCond::GeZero, link: true, annul: false }),
            RT_BLTZL => Some(BranchOrJump { cond: BranchCond::LtZero, link: false, annul: true }),
            RT_BGEZL => Some(BranchOrJump { cond: BranchCond::GeZero, link: false, annul: true }),
            RT_BLTZALL => Some(BranchOrJump { cond: BranchCond::LtZero, link: true, annul: true }),
            RT_BGEZALL => Some(BranchOrJump { cond: BranchCond::GeZero, link: true, annul: true }),
            _ => None,
        },
        OP_J => Some(BranchOrJump { cond: BranchCond::Always, link: false, annul: false }),
        OP_JAL => Some(BranchOrJump { cond: BranchCond::Always, link: true, annul: false }),
        _ => None,
    }
}

/// JR/JALR's shape: whether the link register (always r31... no — whatever
/// `rd` decodes to, per `exec_jalr`) is written. Unlike `BranchOrJump`, a
/// RegJump's target is a runtime register value (§2.3 "always page-leaving/
/// region-end") — there is no on-page/off-page distinction to make at
/// compile time, `taken_exit` is unconditionally `Some(StopReason::RegJump)`
/// for every visited RegJump (the analyzer's contract), so
/// `emit_regjump` never consults `block_for_word`/`taken_exit` at all —
/// every RegJump exits via [`emit_runtime_pc_exit`].
#[derive(Clone, Copy)]
struct RegJump {
    link: bool,
}

fn lookup_regjump(raw: u32) -> Option<RegJump> {
    use crate::mips_isa::*;
    let op = (raw >> 26) & 0x3F;
    if op != OP_SPECIAL {
        return None;
    }
    match raw & 0x3F {
        FUNCT_JR => Some(RegJump { link: false }),
        FUNCT_JALR => Some(RegJump { link: true }),
        _ => None,
    }
}

/// Recompute a branch's on-page target word offset, matching
/// `analyzer::branch_target`'s math exactly (word + 1 + sign-extended
/// imm16 — target is relative to the delay slot's own address, one word
/// past the branch, not two) — the analyzer doesn't persist the resolved
/// target in `CompiledInstr` (only whether the edge exits, via
/// `taken_exit`), so codegen re-derives it from the same inputs. Only
/// meaningful when `taken_exit` is `None` (i.e. the analyzer already
/// confirmed this target is on-page and was walked) — callers must check
/// that first.
fn branch_target_word(raw: u32, word: WordOffset) -> WordOffset {
    let imm16 = (raw & 0xFFFF) as i16 as i32;
    (word as i32 + 1 + imm16) as u16
}

/// Recompute J/JAL's on-page target word offset, matching
/// `analyzer::jump_target`'s math exactly. Only meaningful when `taken_exit`
/// is `None` — see `branch_target_word`. No `page_base` parameter: the
/// result only depends on `word` and `raw`'s target26 bits — the final `&
/// (PAGE_SIZE-1)` mask discards whatever page_base would have contributed,
/// so threading a (possibly wrong — see `emit_jump_target_addr`'s doc
/// comment) page base through here just to have it cancel out would invite
/// exactly the physical/virtual confusion that function exists to avoid.
fn jump_target_word(raw: u32, word: WordOffset) -> WordOffset {
    let target26 = raw & 0x03FF_FFFF;
    let imm28 = target26 << 2;
    let pc = (word as u32) * 4;
    let addr = (pc.wrapping_add(4) & 0xF000_0000) | imm28;
    ((addr & (PAGE_SIZE - 1)) / 4) as u16
}

/// Load live `core.pc` and mask to its containing page (`vbase = pc &
/// !(PAGE_SIZE-1)`) — the one runtime-derived quantity every position-
/// independent address materialization in this module is built from
/// (§2.2: the compiled function must work correctly no matter which virtual
/// alias it's re-entered from, so nothing here may bake in a compile-time
/// absolute address). Matches `emit_exit_block_body`/`emit_write_link_register`'s
/// existing derivation exactly — this is that same three-instruction
/// sequence, pulled out as a shared helper.
fn emit_vbase(ctx: &mut EmitCtx) -> Value {
    let mem = MemFlagsData::trusted();
    let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
    let pc = ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, pc_off);
    ctx.builder.ins().band_imm_s(pc, !(PAGE_SIZE as i64 - 1))
}

/// `vbase | (word * 4)` as a runtime `Value` — the in-page address of
/// `word`, derived from live `core.pc` rather than the compile-time
/// `page_base` parameter `compile_region` is handed (which is a *physical*
/// address, `comp.rs`'s `phys_base` — conflating it with the virtual address
/// this gets stored into `core.pc` was a real bug this session: harmless
/// for kseg0/kseg1 where physical and virtual-low-32-bits coincide, wrong in
/// general, and a violation of §2.2's position-independence contract either
/// way, since it would bake a compile-time constant into what must stay a
/// runtime-computed value).
fn emit_word_addr(ctx: &mut EmitCtx, word: WordOffset) -> Value {
    let vbase = emit_vbase(ctx);
    ctx.builder.ins().iadd_imm_s(vbase, (word as i64) * 4)
}

/// J/JAL's real target as a full virtual address (not the in-page-only word
/// offset `jump_target_word` computes) — used for the `PageLeaving` case,
/// where the target is off-page and therefore has no meaningful in-page
/// word offset at all. Mirrors `MipsExecutor::exec_j`'s own math exactly:
/// `(pc+4) & 0xFFFFFFFF_F0000000 | imm28` — a **64-bit** mask. Using the
/// 32-bit-only `0xF000_0000` here (as `analyzer::jump_target`'s
/// physical-address-only classify() correctly does — it's deciding on-page
/// membership, not building a value for `core.pc`) would zero the upper 32
/// bits of `pc+4` instead of preserving them, destroying the
/// `0xFFFFFFFF` prefix every kseg0/kseg1 virtual address needs — the same
/// sign-extension mistake this session already hit twice, wearing a new
/// disguise (a 32-bit-shaped mask constant instead of a 32-bit-shaped
/// page_base value).
fn emit_jump_target_addr(ctx: &mut EmitCtx, word: WordOffset, raw: u32) -> Value {
    let this_word_pc = emit_word_addr(ctx, word);
    let pc_plus_4 = ctx.builder.ins().iadd_imm_s(this_word_pc, 4);
    let region_base = ctx.builder.ins().band_imm_s(pc_plus_4, 0xFFFFFFFF_F0000000u64 as i64);
    let target26 = raw & 0x03FF_FFFF;
    let imm28 = (target26 << 2) as i64;
    ctx.builder.ins().iadd_imm_s(region_base, imm28)
}

/// A conditional branch's real taken-arm target as a full virtual address —
/// the `PageLeaving`/`ForeignPageSlot`-case counterpart to
/// `branch_target_word`, same rationale as `emit_jump_target_addr`. Matches
/// `analyzer::branch_target`'s address math (target relative to the delay
/// slot's own address, i.e. `word + 1`, not `word`). `word + 1` can exceed
/// `WORDS_PER_PAGE` here (`ForeignPageSlot`, `word == 1023`) — safe because
/// `emit_word_addr` is `iadd`-based, not `bor`-based, so it carries into the
/// next page's bits correctly regardless.
fn emit_branch_target_addr(ctx: &mut EmitCtx, word: WordOffset, raw: u32) -> Value {
    let imm16 = (raw & 0xFFFF) as i16 as i64;
    let slot_pc = emit_word_addr(ctx, word + 1);
    ctx.builder.ins().iadd_imm_s(slot_pc, imm16 * 4)
}

/// Materialize an absolute (already fully computed, runtime `Value`) target
/// address directly into `core.pc` and exit — the `PageLeaving` counterpart
/// to `emit_bail`. Unlike `emit_bail` (which re-enters the interpreter at
/// some *word in this region* for it to re-dispatch, including re-running a
/// delay slot that may not have run yet), the branch/jump's delay slot has
/// *already* executed by the time this is called (§6.1.4 inlines it
/// unconditionally, before the condition/target logic) — so re-dispatching
/// from the branch/jump's own PC would run that slot a second time. Writing
/// the real target directly, like `emit_runtime_pc_exit` does for RegJump,
/// sidesteps re-dispatching anything: the interpreter's next step starts
/// fresh at the true destination, matching a from-scratch interpreter run
/// exactly.
fn emit_absolute_pc_exit(ctx: &mut EmitCtx, target_addr: Value) {
    let mem = MemFlagsData::trusted();
    let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
    ctx.builder.ins().store(mem, target_addr, ctx.core_ptr, pc_off);
    emit_set_jit_trigger(ctx);
    let status = ctx.builder.ins().iconst(ir::types::I32, EXEC_COMPLETE as i64);
    ctx.builder.ins().return_(&[status]);
}

/// Exit stub for a branch/jump/regjump at 0xFFC whose delay slot lives on
/// the next physical page (`analyzer::StopReason::ForeignPageSlot`) — the
/// codegen-time counterpart to the interpreter's `branch_delay`/
/// `handle_branch_not_taken`. There is no `instrs[word+1]` to inline (that
/// index is one past `ENTRIES_PER_PAGE`), so instead of executing the slot
/// now, this arms `core.in_delay_slot`/`core.delay_slot_target` exactly as
/// those two interpreter helpers do and advances `core.pc` by one word (into
/// the slot, which the *next* dispatch — on the next page — will actually
/// execute). `target_addr` is the branch/jump's real destination (already a
/// full virtual address, same shape `emit_branch_target_addr`/
/// `emit_jump_target_addr`/a RegJump's register read produce) — always the
/// next page's word 0 in address terms (`this_word_addr + 4`), but computed
/// from the caller's already-resolved value rather than re-deriving it here,
/// since a RegJump's target isn't expressible as a compile-time offset at
/// all. The entry side (`exec_decoded`'s `entry_offset == 0` always-probe,
/// `codegen.rs`'s `word == entry_word` foreign-slot check) is what actually
/// consumes this state on the next dispatch — this function's only job is to
/// arm it correctly before handing off. Deliberately does *not* call
/// `emit_set_jit_trigger` — unlike `emit_absolute_pc_exit`/
/// `emit_runtime_pc_exit` (which land on a transfer's real, already-executed
/// destination, matching `handle_exec_complete`'s slot-retirement path),
/// this lands on the delay slot itself, one word into the *next* page, with
/// the real transfer still pending — exactly `branch_delay`'s own contract,
/// which likewise never sets `jit_trigger`. Terminates the current block.
fn emit_foreign_page_slot_exit(ctx: &mut EmitCtx, word: WordOffset, target_addr: Value) {
    let mem = MemFlagsData::trusted();
    let flag_off = ir::immediates::Offset32::new(core_offset_of_in_delay_slot());
    let target_off = ir::immediates::Offset32::new(core_offset_of_delay_slot_target());
    let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());

    ctx.builder.ins().store(mem, target_addr, ctx.core_ptr, target_off);
    let one = ctx.builder.ins().iconst(ir::types::I8, 1);
    ctx.builder.ins().store(mem, one, ctx.core_ptr, flag_off);
    let next_pc = emit_word_addr(ctx, word + 1);
    ctx.builder.ins().store(mem, next_pc, ctx.core_ptr, pc_off);

    let status = ctx.builder.ins().iconst(ir::types::I32, EXEC_COMPLETE as i64);
    ctx.builder.ins().return_(&[status]);
}

/// The annulling-Likely not-taken arm at 0xFFC: the slot never executes at
/// all (mirrors `handle_branch_likely_skip`'s direct `pc += 8`, no
/// delay-slot dispatch), so there's nothing to arm — just advance two words
/// past this instruction (`this_word_addr + 8`, landing on the next page's
/// word 4) and exit. No `emit_set_jit_trigger` either, matching
/// `handle_branch_likely_skip` itself, which doesn't set it. Terminates the
/// current block.
fn emit_foreign_page_annulled_not_taken_exit(ctx: &mut EmitCtx, word: WordOffset) {
    let mem = MemFlagsData::trusted();
    let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
    let next_pc = emit_word_addr(ctx, word + 2);
    ctx.builder.ins().store(mem, next_pc, ctx.core_ptr, pc_off);
    let status = ctx.builder.ins().iconst(ir::types::I32, EXEC_COMPLETE as i64);
    ctx.builder.ins().return_(&[status]);
}

/// Emit a branch/jump's full unit: the delay slot's semantics inlined
/// unconditionally first (§6.1.4 — the slot always executes exactly once,
/// regardless of whether the branch is taken; never a CFG edge, never
/// independently chained), then the condition test and the taken/
/// not-taken edges, each resolved to either another head instruction's
/// block (`block_for_word`) or an exit bail using the analyzer's recorded
/// `taken_exit`/`fallthrough_exit` reason. Terminates the current block —
/// call as the last thing emitted for this instruction; `compile_region`'s
/// pass 2 does not emit anything else for it afterward.
///
/// Link-writing (JAL, and eventually BLTZAL/BGEZAL) commits `core.gpr[31] =
/// (this instruction's own word + 2) * 4 + vbase`-equivalent — mirrors
/// `exec_jal`'s `self.core.pc + 8` (this instruction's address + 8 bytes,
/// i.e. two words past it, skipping the delay slot) — computed the same way
/// the exit block derives an exit PC (`vbase | offset*4`), since the link
/// value is architecturally a full virtual address, not just a word index.
fn emit_branch_or_jump(
    ctx: &mut EmitCtx,
    exit_block: Block,
    block_for_word: &std::collections::HashMap<WordOffset, Block>,
    instrs: &[CompiledInstr; ENTRIES_PER_PAGE],
    branch: BranchOrJump,
    fr_mode: FrMode,
) {
    let word = ctx.word;
    let raw = ctx.raw;
    // The delay slot's raw bytes — always inlined via emit_slot_semantics
    // below, never dispatched as an independent head by compile_region's
    // main loop (pass 1 skips any word still `is_slot_only` at that point).
    // Guaranteed non-Excluded and present by the
    // analyzer's walk (visit_slot only marks it visited if compilable) and
    // to have a semantics emitter (compile_region's upfront rejection loop
    // already checked every visited instruction, slot or not) — true
    // whether or not this branch ends up annulling it; the analyzer treats
    // "is this slot compilable" as independent of annul semantics, which is
    // a codegen-only concern (§4.3).
    //
    // Non-annulling branches (plain + link-writing conditional, J/JAL):
    // the slot always executes exactly once, taken or not (§6.1.4) — emit
    // it inline, unconditionally, before the condition/target. Annulling
    // "Likely" branches (§4.3): the slot executes ONLY on the taken path;
    // the not-taken path skips it entirely (`handle_branch_likely_skip`'s
    // `pc += 8`, no delay-slot dispatch at all) — its effects must never be
    // committed, so it can only be emitted inside the taken arm.
    let slot_word = word + 1;
    // `ForeignPageSlot` (analyzer's `is_0xffc_branch`, word == 1023): the
    // slot is on the next page, there's nothing at `instrs[slot_word]` to
    // inline — `slot_word == WORDS_PER_PAGE` here, one past the array.
    // `emit_slot`/`emit_branch_taken_edge`/`emit_target_edge` below are all
    // skipped or redirected accordingly; see the taken/not-taken arms.
    let foreign_page_slot = slot_word as usize >= ENTRIES_PER_PAGE;
    let slot_raw = if foreign_page_slot { 0 } else { instrs[slot_word as usize].raw };

    if branch.link {
        // J/JAL/BLTZAL/BGEZAL (and their Likely counterparts) are all
        // architecturally hardcoded to r31 — unlike JALR's `rd` field.
        let this_pc_word = word as i64 + 2; // this instruction's address + 8 bytes = +2 words
        emit_write_link_register(ctx, this_pc_word, 31);
    }

    // Recurse into the slot's own emission with ctx.raw/ctx.word switched to
    // the slot's — restored after each use, mirroring emit_slot_semantics'
    // own core.pc save/restore around the same call. Never called at all
    // when `foreign_page_slot` — every call site below checks that first.
    // `target`: this branch's real delay_slot_target for jitv2_lockstep's
    // benefit (see emit_slot_semantics' doc comment) — the branch's actual
    // destination for Always/taken, or the taken-vs-fallthrough `select` for
    // a non-annulling conditional (§6.1.4: its slot runs exactly once,
    // unconditionally, *before* the condition's taken/not-taken split is
    // even materialized as separate blocks, so this must already reflect
    // both outcomes as one runtime value — real hardware's `branch_delay`
    // likewise arms the slot's target before the branch's own commit,
    // using whichever destination the condition resolved to).
    let emit_slot = |ctx: &mut EmitCtx, target: Value| -> bool {
        if try_emit_fused_nop_slot(ctx, instrs, slot_word, slot_raw) {
            return false;
        }
        ctx.raw = slot_raw;
        ctx.word = slot_word;
        let terminated = emit_slot_semantics(ctx, instrs, fr_mode, target);
        ctx.raw = raw;
        ctx.word = word;
        terminated
    };

    match branch.cond {
        BranchCond::Always => {
            // J/JAL are never annulling (Always has no not-taken arm at all).
            // If the slot is itself a nested branch/regjump, it already
            // exited with its own final core.pc — this J/JAL's own taken
            // target never takes effect (nested-delay-slot semantics: the
            // innermost dispatched transfer wins), so nothing more to emit.
            // `foreign_page_slot`: no slot to inline at all — the taken edge
            // (always `ForeignPageSlot` in that case, per
            // `finish_visit_foreign_page_slot`) arms the pending transfer
            // instead, exactly like `emit_jump_taken_edge`'s `ForeignPageSlot`
            // arm.
            if !foreign_page_slot {
                let target_addr = emit_jump_target_addr(ctx, word, raw);
                if emit_slot(ctx, target_addr) {
                    return;
                }
            }
            emit_jump_taken_edge(ctx, exit_block, block_for_word, instrs[word as usize].taken_exit, word, raw);
        }
        _ if !branch.annul => {
            let cond_val = emit_cond(ctx, raw, branch.cond);
            // Same nested-supersedes-outer rule as the Always case above:
            // if the slot terminated, this branch's own condition result
            // (already computed, but never consumed) never matters — the
            // nested transfer already won. Skipped entirely when
            // `foreign_page_slot` (no slot to inline).
            if !foreign_page_slot {
                let taken_addr = emit_branch_target_addr(ctx, word, raw);
                let fallthrough_addr = emit_word_addr(ctx, word + 2);
                let target = ctx.builder.ins().select(cond_val, taken_addr, fallthrough_addr);
                if emit_slot(ctx, target) {
                    return;
                }
            }

            let taken_block = ctx.builder.create_block();
            let not_taken_block = ctx.builder.create_block();
            ctx.builder.ins().brif(cond_val, taken_block, &[], not_taken_block, &[]);

            // Both branch-internal blocks have exactly one predecessor (this
            // instruction's own block, via the brif above) — sealable as
            // soon as their own single terminator is emitted, same as a
            // preamble's fired/continue sub-blocks. Not part of
            // compile_region's instr_blocks (they're not a head instruction
            // for any other edge to target), so pass 3 never touches them.
            ctx.builder.switch_to_block(taken_block);
            ctx.builder.seal_block(taken_block);
            emit_branch_taken_edge(ctx, exit_block, block_for_word, instrs[word as usize].taken_exit, word, raw);

            ctx.builder.switch_to_block(not_taken_block);
            ctx.builder.seal_block(not_taken_block);
            if foreign_page_slot {
                // Not-taken still executes the slot exactly once (§6.1.4 —
                // the delay slot always runs, taken or not), but it's on the
                // next page: arm it with target = this_word_addr + 8,
                // mirroring `handle_branch_not_taken`'s `branch_delay(pc+8)`
                // exactly (not landing directly on `core.pc` — that would
                // skip the slot instead of deferring it, wrong for the
                // non-annulling case; only the annulling not-taken arm below
                // does that).
                let target_addr = emit_word_addr(ctx, word + 2);
                emit_foreign_page_slot_exit(ctx, word, target_addr);
            } else {
                let fallthrough_word = word + 2; // past the slot
                emit_target_edge(ctx, exit_block, block_for_word, instrs[word as usize].fallthrough_exit, fallthrough_word);
            }
        }
        _ => {
            // Annulling Likely branch: slot only emitted on the taken arm.
            let cond_val = emit_cond(ctx, raw, branch.cond);

            let taken_block = ctx.builder.create_block();
            let not_taken_block = ctx.builder.create_block();
            ctx.builder.ins().brif(cond_val, taken_block, &[], not_taken_block, &[]);

            ctx.builder.switch_to_block(taken_block);
            ctx.builder.seal_block(taken_block);
            // If the slot is a nested branch/regjump, it already exited
            // (own final core.pc) — this branch's own taken target never
            // takes effect, same nested-supersedes-outer rule as above.
            // Likely's slot only ever runs here, on the taken arm, so its
            // target is unconditionally the real branch destination (no
            // select needed, unlike the non-annulling arm above).
            let slot_terminated = if foreign_page_slot {
                false
            } else {
                let taken_addr = emit_branch_target_addr(ctx, word, raw);
                emit_slot(ctx, taken_addr)
            };
            if foreign_page_slot || !slot_terminated {
                emit_branch_taken_edge(ctx, exit_block, block_for_word, instrs[word as usize].taken_exit, word, raw);
            }

            // Not taken: annulled — the slot's effects are never committed,
            // and its instruction is never "executed" at all, matching
            // handle_branch_likely_skip's direct pc+=8 (no delay-slot
            // dispatch, no slot-word landing in between). Exit target is
            // word+2 (past the slot), same numeric destination as the
            // non-annulling case's not-taken arm — the difference is
            // entirely about whether the slot's semantics were committed on
            // the way there, not about where execution resumes. Independent
            // of the taken arm's own block, so always safe to emit
            // regardless of whether emit_slot_semantics terminated it.
            ctx.builder.switch_to_block(not_taken_block);
            ctx.builder.seal_block(not_taken_block);
            if foreign_page_slot {
                // Nothing to arm — the annulled slot never executes, so
                // there's no pending transfer to defer onto the next page's
                // dispatch, matching handle_branch_likely_skip exactly
                // (direct pc+=8, no in_delay_slot involvement at all).
                emit_foreign_page_annulled_not_taken_exit(ctx, word);
            } else {
                let fallthrough_word = word + 2;
                emit_target_edge(ctx, exit_block, block_for_word, instrs[word as usize].fallthrough_exit, fallthrough_word);
            }
        }
    }
}

/// J/JAL's taken edge: on-page continues via `emit_target_edge` as usual;
/// `PageLeaving` needs `emit_jump_target_addr`'s runtime-computed address
/// instead of `emit_target_edge`'s word-offset bail (see `emit_target_edge`'s
/// doc comment on why that case can't be handled generically there).
fn emit_jump_taken_edge(
    ctx: &mut EmitCtx,
    exit_block: Block,
    block_for_word: &std::collections::HashMap<WordOffset, Block>,
    taken_exit: Option<crate::jitv2::analyzer::StopReason>,
    word: WordOffset,
    raw: u32,
) {
    if taken_exit == Some(crate::jitv2::analyzer::StopReason::PageLeaving) {
        let target_addr = emit_jump_target_addr(ctx, word, raw);
        emit_absolute_pc_exit(ctx, target_addr);
    } else if taken_exit == Some(crate::jitv2::analyzer::StopReason::ForeignPageSlot) {
        let target_addr = emit_jump_target_addr(ctx, word, raw);
        emit_foreign_page_slot_exit(ctx, word, target_addr);
    } else {
        let target_word = jump_target_word(raw, word);
        emit_target_edge(ctx, exit_block, block_for_word, taken_exit, target_word);
    }
}

/// A conditional branch's taken edge — same shape as `emit_jump_taken_edge`,
/// using `emit_branch_target_addr` for the `PageLeaving`/`ForeignPageSlot`
/// cases.
fn emit_branch_taken_edge(
    ctx: &mut EmitCtx,
    exit_block: Block,
    block_for_word: &std::collections::HashMap<WordOffset, Block>,
    taken_exit: Option<crate::jitv2::analyzer::StopReason>,
    word: WordOffset,
    raw: u32,
) {
    if taken_exit == Some(crate::jitv2::analyzer::StopReason::PageLeaving) {
        let target_addr = emit_branch_target_addr(ctx, word, raw);
        emit_absolute_pc_exit(ctx, target_addr);
    } else if taken_exit == Some(crate::jitv2::analyzer::StopReason::ForeignPageSlot) {
        let target_addr = emit_branch_target_addr(ctx, word, raw);
        emit_foreign_page_slot_exit(ctx, word, target_addr);
    } else {
        emit_target_edge(ctx, exit_block, block_for_word, taken_exit, branch_target_word(raw, word));
    }
}

/// Resolve one outgoing edge to either the target word's head block (if
/// `exit_reason` is `None`, meaning the analyzer confirmed this edge
/// continues into compiled code) or a bail to the shared exit block (if
/// `Some`, using the analyzer's recorded reason). Terminates the current
/// block.
///
/// `target_word` is the branch/jump's *computed* on-page target — valid to
/// bail to directly for every exit reason except `PageLeaving`: the
/// analyzer's `classify` only classifies a branch/jump as page-leaving when
/// the target address genuinely isn't on this page, so `branch_target_word`/
/// `jump_target_word`'s page-relative masking produces a meaningless (wrong
/// page) word offset in that one case.
///
/// `PageLeaving` additionally can't reuse a plain word-offset bail
/// (`emit_bail`) at all, even with the *right* word: by the time this is
/// called, the branch/jump's mandatory delay slot has already executed
/// inline (§6.1.4 — unconditional, taken or not, emitted before this edge).
/// A bail re-enters the interpreter at some word *in this region* for it to
/// re-dispatch — which for the branch/jump's own word would re-run that
/// already-committed slot a second time. Instead, `target_addr` (the real,
/// fully-computed absolute target — `jump_target_addr`/`branch_target_addr`)
/// is written directly into `core.pc` via `emit_absolute_pc_exit`, the same
/// technique `emit_runtime_pc_exit` uses for RegJump's register-derived
/// target: no re-dispatch of anything, the interpreter's next step starts
/// fresh at the true destination. (Observed live before this fix: a JIT'd
/// off-page J landing thousands of words away from its real target on an
/// IRIX 5.3 boot trace, from bailing to the page-masked word offset; fixing
/// that alone would have reintroduced a silent double-execution of the delay
/// slot, which is why this uses a direct absolute-address exit instead of
/// just correcting the bail's target word.)
fn emit_target_edge(
    ctx: &mut EmitCtx,
    exit_block: Block,
    block_for_word: &std::collections::HashMap<WordOffset, Block>,
    exit_reason: Option<crate::jitv2::analyzer::StopReason>,
    target_word: WordOffset,
) {
    match exit_reason {
        Some(crate::jitv2::analyzer::StopReason::PageLeaving) => unreachable!(
            "PageLeaving must be handled by the caller via emit_target_edge_page_leaving \
             (needs a runtime-computed target address, which only the caller — knowing \
             whether this is a J/JAL or a conditional branch — can compute correctly)"
        ),
        Some(_) => emit_bail(ctx, exit_block, target_word),
        None => {
            let target_block = *block_for_word.get(&target_word)
                .expect("exit_reason is None -> analyzer guarantees target_word continues into the region");
            ctx.builder.ins().jump(target_block, &[]);
        }
    }
}

/// Emit a delay slot's semantics inline (§6.1.4) — same dispatch as a plain
/// `Sequential` instruction (`lookup_semantics`) for the common case, but
/// without any of the caller-side fallthrough/exit wiring `compile_region`'s
/// main loop does for head instructions, since a slot is never itself a
/// region-exit point *when it's Sequential* — control always continues on to
/// the branch's own condition/target logic immediately after. `module` is
/// threaded through unchanged — a slot can be any `Sequential`-classified
/// instruction, including a load/store, which needs the real module for its
/// `call_indirect` (`emit_mem_read`/`emit_mem_write`).
///
/// **Nested delay slot** (the slot's own raw bits are themselves a
/// branch/jump/regjump — "unusual but legal" on real hardware, already
/// supported by the interpreter's nested `branch_delay`, and walked
/// recursively by `analyzer::visit_slot`): recurses into
/// `emit_nested_branch_slot`, which *is* a region-exit point (every one of
/// its own edges — taken, not-taken, or a further-nested slot's eventual
/// edges — always exits this compiled unit directly, since the minimal
/// region here never contains any other word's block to jump into). See
/// that function's doc comment for the full recursive shape. `instrs` is
/// only needed for this nested case (to fetch the next slot's raw bytes);
/// threaded through unconditionally for a uniform signature.
///
/// Brackets the slot's semantics with `core.in_delay_slot = true` / `= false`
/// — the same field the interpreter's `branch_delay`/`handle_exec_complete`
/// set on the plain dispatch path, no separate JIT-only copy — and writes
/// `core.pc = slot_addr` before running them: if the slot itself raises an
/// exception (a store fault, integer overflow, an FCSR-enabled trap),
/// `emit_exception_exit`'s `handle_exception_fn` call (`deliver_exception`,
/// shared with the interpreter — §4.2 single-implementation delivery)
/// computes EPC as `core.pc - 4` when `in_delay_slot` is set, mirroring the
/// interpreter's own model where `core.pc` already points at the slot's own
/// address by the time it's dispatched (the branch already advanced past
/// itself via `branch_delay`). The JIT has no equivalent per-instruction PC
/// advance — without writing `core.pc` here first, `deliver_exception`'s
/// formula would subtract 4 from the *branch's* address instead, landing EPC
/// one word too early. Setting `in_delay_slot` without also correcting
/// `core.pc` doesn't fully fix this by itself — both are needed together.
///
/// Neither write threads an `in_delay_slot`/`slot_addr` parameter through
/// `lookup_semantics`/`lookup_cp1_semantics`'s ~60 emitters — a plain head
/// instruction never calls this function, so its own `core.pc` write (via
/// the shared exit block / absolute-address exits) is unaffected; only a
/// slot's *own* possible exception exit ever observes this early write.
///
/// `in_delay_slot`/`core.pc` are reset/restored after the slot completes
/// without trapping and without exiting the function — only the Sequential
/// base case can reach that point; every nested-branch path (taken,
/// not-taken, or a still-deeper nested slot's eventual edges) is a
/// terminator and returns before ever reaching the restore, since those
/// paths write their own final `core.pc` and that must not be clobbered by
/// this function's own restore-and-continue tail.
///
/// Returns `true` if this call **terminated the current Cranelift block**
/// (the nested branch/regjump cases — every arm of those exits via
/// `emit_absolute_pc_exit`/`emit_runtime_pc_exit`, both terminators) or
/// `false` if control falls through normally (the Sequential base case).
/// Callers **must** check this and stop emitting further IR into the
/// current block when `true` — Cranelift's verifier rejects any instruction
/// after a block's terminator (`"a terminator instruction was encountered
/// before the end of block"`), which a caller unconditionally continuing to
/// emit its own condition-test/exit-wiring after this call would trigger
/// whenever the slot itself branches, since nothing in this module used to
/// need a slot to ever end the block early.
/// Fast path for the single most common delay slot in practice: a real NOP
/// (`raw == 0`, i.e. `SLL $0,$0,0`) — compiler-inserted after nearly every
/// branch/jump that doesn't have useful slot-fill work. Mirrors the
/// interpreter's own `exec_jr_nop`/`exec_j_nop`/`exec_beq_nop`/etc. fusion
/// (mips_exec.rs): skip the slot dispatch entirely — no `in_delay_slot`
/// flag, no `core.pc` save/store/restore, no BD bookkeeping, no dev-trace/bp
/// hook — since a NOP has no architectural effect and (being `raw == 0`) can
/// never itself be a nested branch/jump/regjump, never raises an exception,
/// and is never worth single-stepping to. Only `emit_account_for_cycles`
/// runs here, once more, to keep `core.hot.cycles` bookkeeping current for
/// the fused slot's own retirement — the branch/jump itself already
/// accounted for its own retirement from `compile_region_uncommitted`'s
/// per-head-instruction loop before this runs, so this call accounts for
/// the fused slot only, giving the pair a combined pending-count
/// contribution of 2 without this function double-counting the branch's
/// own share.
///
/// This compiled unit's `opt_level = "none"` (see
/// `rules/jit/cranelift-opt-levelnone-is-the-right-trade-for-throughput-jits.md`)
/// means none of `emit_slot_semantics`' bracketing IR would otherwise be
/// eliminated even though a NOP's own semantics emitter (`emit_sll` with
/// `rd == 0`, skipped by `emit_write_gpr`) is already free — every load/
/// store in the bracketing becomes real native code, so skipping it here is
/// a real, not just theoretical, win.
///
/// Excluded under `jitv2_lockstep` (needs the slot individually
/// step-bracketed to compare against the interpreter reference) and
/// `developer` (needs the slot individually addressable for `dt`
/// tracing/breakpoints) — both fall back to the full `emit_slot_semantics`
/// path unconditionally, matching every other verification/tracing
/// exclusion in this module.
///
/// Returns `true` if fused (caller must skip calling `emit_slot_semantics`
/// at all — there is no block-terminating case here, unlike that function,
/// so callers don't need to check for early-return the way they do for its
/// `bool` result), `false` if the slot needs the normal path.
#[cfg_attr(any(not(feature = "jitv2_opcodefusion"), feature = "jitv2_lockstep", feature = "developer"), allow(unused))]
fn try_emit_fused_nop_slot(ctx: &mut EmitCtx, instrs: &[CompiledInstr; ENTRIES_PER_PAGE], slot_word: WordOffset, slot_raw: u32) -> bool {
    #[cfg(any(not(feature = "jitv2_opcodefusion"), feature = "jitv2_lockstep", feature = "developer"))]
    {
        let _ = (instrs, slot_word, slot_raw);
        false
    }
    #[cfg(all(feature = "jitv2_opcodefusion", not(any(feature = "jitv2_lockstep", feature = "developer"))))]
    {
        if slot_raw != 0 {
            return false;
        }
        emit_account_for_cycles(ctx, instrs, slot_word);
        true
    }
}

fn emit_slot_semantics(ctx: &mut EmitCtx, instrs: &[CompiledInstr; ENTRIES_PER_PAGE], fr_mode: FrMode, delay_slot_target: Value) -> bool {
    let slot_raw = ctx.raw;
    let slot_word = ctx.word;
    // From here until this function returns, ctx.word/raw are the slot's
    // own — so any emit_exception_exit reached from within the slot's own
    // semantics emitter (called below) must report Cause.BD = true. Never
    // restored back to false afterward: ctx is not reused for anything else
    // once this returns (the pass-2 loop constructs a fresh ctx per head).
    ctx.bd = true;
    let mem = MemFlagsData::trusted();
    let flag_off = ir::immediates::Offset32::new(core_offset_of_in_delay_slot());
    let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
    let one = ctx.builder.ins().iconst(ir::types::I8, 1);
    ctx.builder.ins().store(mem, one, ctx.core_ptr, flag_off);
    // jitv2_lockstep only: arm core.delay_slot_target with the branch's real
    // destination (already resolved by the caller — a register read for
    // RegJump, the branch-target/fallthrough Value for a conditional/J/JAL)
    // so the interpreter reference lockstep_step runs for this slot below
    // sees the same value handle_exec_complete would use on a real dispatch.
    // Without this, delay_slot_target is whatever was last written (often 0
    // in a synthetic/first-dispatch test), and handle_exec_complete's
    // `core.pc = core.delay_slot_target` on the slot's own retire produces a
    // bogus reference pc — not a real JIT/interpreter divergence, just an
    // uninitialized comparison target. Not needed outside lockstep: the
    // non-lockstep JIT path never reads delay_slot_target for an inlined
    // slot at all (the outer branch/regjump writes its own final core.pc
    // directly via emit_absolute_pc_exit/emit_runtime_pc_exit, independent
    // of this field).
    #[cfg(feature = "jitv2_lockstep")]
    {
        let target_off = ir::immediates::Offset32::new(core_offset_of_delay_slot_target());
        ctx.builder.ins().store(mem, delay_slot_target, ctx.core_ptr, target_off);
    }
    #[cfg(not(feature = "jitv2_lockstep"))]
    let _ = delay_slot_target;
    // Save the region's real entry pc before overwriting it — every later
    // exit in this same compiled unit (emit_exit_block_body's `vbase = pc &
    // !(PAGE_SIZE-1)`, emit_bail's retry word, an outer branch/jump's own
    // link-register write) needs core.pc to still reflect the entry
    // instruction's real page once the slot completes normally, not
    // whatever the slot's own address was. Restored below on the
    // slot-completed-without-trapping path only — if the slot itself raises
    // an exception, control never returns here (emit_exception_exit is a
    // block terminator), so there's nothing to restore on that path: the
    // slot's `core.pc` write is exactly what deliver_exception needs to see
    // in that case. The slot's address itself is derived from this same
    // live pc load (emit_word_addr's vbase, §2.2 position independence) —
    // never from compile-time page_base, which is a physical address in
    // production (`comp.rs`'s `phys_base`) and would be wrong to bake into
    // a value written into core.pc (a virtual address).
    let saved_pc = ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, pc_off);
    let slot_addr_val = emit_word_addr(ctx, slot_word);
    ctx.builder.ins().store(mem, slot_addr_val, ctx.core_ptr, pc_off);
    // The delay slot always executes exactly once here (§6.1.4 — never
    // conditional, never skippable), so this is unconditional too, unlike
    // the head-instruction loop's post-preamble placement — a slot has no
    // IP7/interrupt preamble of its own to bail out of first. Covers the
    // nested-branch-in-delay-slot case too (this call's own slot instruction
    // still retires before `emit_nested_branch_slot` recurses into whatever
    // *that* slot's own delay slot is, which gets its own separate
    // `emit_slot_semantics` call and its own accounting).
    emit_account_for_cycles(ctx, instrs, slot_word);

    // Developer per-instruction hook (dt traceback + PC breakpoints), same
    // as the head-instruction loop's own emit_dev_trace_bp call — without
    // this a delay slot (including a nested branch-in-slot, handled by the
    // recursive calls below) is invisible to `dt` tagging entirely and can't
    // hit a PC breakpoint, even though it's real, independently addressed
    // architectural state (§6.1.4). ctx.word/raw/bd are already the slot's
    // own (set above), matching what the hook needs to record.
    #[cfg(feature = "developer")]
    emit_dev_trace_bp(ctx, dev_trace_origin::JIT_DELAY_SLOT);

    if let Some(branch) = lookup_branch_or_jump(slot_raw) {
        emit_nested_branch_slot(ctx, instrs, slot_word, slot_raw, branch, fr_mode);
        return true; // every arm is a terminator; the restore below is unreachable from here
    }
    if let Some(regjump) = lookup_regjump(slot_raw) {
        emit_nested_regjump_slot(ctx, instrs, slot_word, slot_raw, regjump, fr_mode);
        return true; // emit_runtime_pc_exit is a terminator
    }

    // jitv2_lockstep: bracket the slot's own instruction like any other
    // ALU/load-store/FPU dispatch — this was the missing piece that let a
    // slot's real memory access (e.g. a store using a stale/wrong address)
    // through unverified: emit_mem_read/write's lockstep hooks compare
    // unconditionally against whatever core.lockstep_mem currently holds,
    // with no check that *this* instruction was ever step-bracketed, so an
    // unbracketed slot compared against a stale leftover capture from
    // whatever head instruction last ran lockstep_step — sometimes matching
    // by coincidence, sometimes not, but never a real per-instruction check.
    // `trust_live=true`: core.pc/in_delay_slot were just set to this slot's
    // own address/true above (lines 3634/3652), exactly the state
    // lockstep_step's LOCKSTEP_BD_LIVE contract expects (same as an entry
    // word or branch-fallback successor). Not emitted for a nested
    // branch/jump/regjump slot (returned above already) — those still need
    // the two-dispatch model reconciliation this doesn't solve.
    #[cfg(feature = "jitv2_lockstep")]
    emit_lockstep_step(ctx, true);

    // slot_raw == 0 only reaches here under jitv2_lockstep/developer (the
    // try_emit_fused_nop_slot fast path in emit_branch_or_jump/emit_regjump
    // already intercepts a NOP slot everywhere else) — same dead-dispatch
    // skip as the head-instruction loop above, gated the same way.
    #[cfg(any(feature = "jitv2_lockstep", feature = "developer"))]
    let skip_nop_dispatch = false;
    #[cfg(not(any(feature = "jitv2_lockstep", feature = "developer")))]
    let skip_nop_dispatch = slot_raw == 0;
    if !skip_nop_dispatch {
        if let Some(emit) = lookup_semantics(slot_raw) {
            emit(ctx);
        } else {
            let emit = lookup_cp1_semantics(slot_raw)
                .expect("slot instruction must have a semantics emitter (checked in compile_region)");
            emit(ctx, fr_mode);
        }
    }

    // Compare before the pc/bd restore below overwrites the fields the
    // interpreter reference's snapshot needs to match against. Unlike a
    // plain straight-line head (emit_lockstep_compare_seq's slot_word+1),
    // this instruction retires FROM in_delay_slot=true: handle_exec_complete
    // sends the interpreter's real pc to core.delay_slot_target, not
    // slot_word+1 — so the expected post-state is `delay_slot_target` (the
    // same value stored into core.delay_slot_target above) with
    // in_delay_slot=false. Materialize that right before it gets overwritten
    // anyway by the real restore-for-the-outer-branch below.
    #[cfg(feature = "jitv2_lockstep")]
    {
        ctx.builder.ins().store(mem, delay_slot_target, ctx.core_ptr, pc_off);
        let zero8 = ctx.builder.ins().iconst(ir::types::I8, 0);
        ctx.builder.ins().store(mem, zero8, ctx.core_ptr, flag_off);

        let ptr_ty = ctx.module.target_config().pointer_type();
        let jit_ctx_off = ir::immediates::Offset32::new(core_offset_of_jit_ctx());
        let jit_ctx = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, jit_ctx_off);
        let fn_off = ir::immediates::Offset32::new(core_offset_of_lockstep_compare_fn());
        let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);
        let mut sig = ctx.module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty));
        sig.returns.push(AbiParam::new(ir::types::I32));
        let sig_ref = ctx.builder.import_signature(sig);
        let call = ctx.builder.ins().call_indirect(sig_ref, callee, &[jit_ctx]);
        let status = ctx.builder.inst_results(call)[0];

        let is_bp = ctx.builder.ins().icmp_imm_s(IntCC::Equal, status, crate::mips_exec::EXEC_BREAKPOINT as i64);
        let bp_block = ctx.builder.create_block();
        let continue_block = ctx.builder.create_block();
        ctx.builder.ins().brif(is_bp, bp_block, &[], continue_block, &[]);

        ctx.builder.switch_to_block(bp_block);
        ctx.builder.set_cold_block(bp_block);
        ctx.builder.seal_block(bp_block);
        // pc already points at the slot's own next-instruction address on
        // divergence's "left off here" semantics — same convention as
        // emit_lockstep_compare_seq, back the pc up to the slot's own
        // address so the monitor stops at the instruction that diverged,
        // not the one after it.
        let slot_addr_again = emit_word_addr(ctx, slot_word);
        ctx.builder.ins().store(mem, slot_addr_again, ctx.core_ptr, pc_off);
        let bp_status = ctx.builder.ins().iconst(ir::types::I32, crate::mips_exec::EXEC_BREAKPOINT as i64);
        ctx.builder.ins().return_(&[bp_status]);

        ctx.builder.switch_to_block(continue_block);
        ctx.builder.seal_block(continue_block);
    }

    let zero = ctx.builder.ins().iconst(ir::types::I8, 0);
    ctx.builder.ins().store(mem, zero, ctx.core_ptr, flag_off);
    ctx.builder.ins().store(mem, saved_pc, ctx.core_ptr, pc_off);
    false
}

/// A nested delay slot whose own raw bits are a branch/jump (`emit_cond`'s
/// `raw`/`slot_word` here refer to *this* nested instruction, one level
/// deeper than whatever called `emit_slot_semantics`). Every edge exits the
/// compiled unit directly — there is no "continue into compiled code" case
/// at any nesting level below the outermost head instruction, since the
/// minimal region this JIT compiles (`comp.rs`'s `MAX_INSTRS_PER_COMPILE`)
/// only ever contains the head plus its slot-chain, never another word's
/// block to jump into. This mirrors the outermost branch's own
/// `PageLeaving` handling (`emit_absolute_pc_exit`) rather than
/// `emit_branch_or_jump`'s in-region `block_for_word` wiring.
///
/// `core.pc` must already equal this nested instruction's own address
/// (`emit_slot_semantics` wrote it before calling in) — `emit_cond`,
/// `emit_write_link_register`, and the target-address helpers all read it
/// live, matching every other position-independent address computation in
/// this module (§2.2).
fn emit_nested_branch_slot(
    ctx: &mut EmitCtx,
    instrs: &[CompiledInstr; ENTRIES_PER_PAGE],
    word: WordOffset,
    raw: u32,
    branch: BranchOrJump,
    fr_mode: FrMode,
) {
    let inner_slot_word = word + 1;
    let inner_slot_raw = instrs[inner_slot_word as usize].raw;

    if branch.link {
        let this_pc_word = word as i64 + 2;
        emit_write_link_register(ctx, this_pc_word, 31);
    }

    // Recurse into the inner slot's own emission with ctx.raw/ctx.word
    // switched to its — restored after, same pattern as emit_branch_or_jump's
    // own emit_slot closure. `target`: see that closure's doc comment —
    // this nested branch's real delay_slot_target for jitv2_lockstep.
    let emit_inner_slot = |ctx: &mut EmitCtx, target: Value| -> bool {
        ctx.raw = inner_slot_raw;
        ctx.word = inner_slot_word;
        let terminated = emit_slot_semantics(ctx, instrs, fr_mode, target);
        ctx.raw = raw;
        ctx.word = word;
        terminated
    };

    match branch.cond {
        BranchCond::Always => {
            // A still-deeper nested branch/regjump in this slot already
            // exited with its own final core.pc — same nested-supersedes-
            // outer rule as emit_branch_or_jump's Always arm.
            let target_addr = emit_jump_target_addr(ctx, word, raw);
            if emit_inner_slot(ctx, target_addr) {
                return;
            }
            emit_absolute_pc_exit(ctx, target_addr);
        }
        _ if !branch.annul => {
            let cond_val = emit_cond(ctx, raw, branch.cond);
            let taken_addr = emit_branch_target_addr(ctx, word, raw);
            let fallthrough_addr = emit_word_addr(ctx, word + 2);
            let target = ctx.builder.ins().select(cond_val, taken_addr, fallthrough_addr);
            if emit_inner_slot(ctx, target) {
                return;
            }

            let taken_block = ctx.builder.create_block();
            let not_taken_block = ctx.builder.create_block();
            ctx.builder.ins().brif(cond_val, taken_block, &[], not_taken_block, &[]);

            ctx.builder.switch_to_block(taken_block);
            ctx.builder.seal_block(taken_block);
            emit_absolute_pc_exit(ctx, taken_addr);

            ctx.builder.switch_to_block(not_taken_block);
            ctx.builder.seal_block(not_taken_block);
            emit_absolute_pc_exit(ctx, fallthrough_addr);
        }
        _ => {
            // Annulling Likely: nested slot only on the taken arm, matching
            // the outermost branch's own annul handling exactly.
            let cond_val = emit_cond(ctx, raw, branch.cond);

            let taken_block = ctx.builder.create_block();
            let not_taken_block = ctx.builder.create_block();
            ctx.builder.ins().brif(cond_val, taken_block, &[], not_taken_block, &[]);

            ctx.builder.switch_to_block(taken_block);
            ctx.builder.seal_block(taken_block);
            let target_addr = emit_branch_target_addr(ctx, word, raw);
            if !emit_inner_slot(ctx, target_addr) {
                emit_absolute_pc_exit(ctx, target_addr);
            }

            ctx.builder.switch_to_block(not_taken_block);
            ctx.builder.seal_block(not_taken_block);
            let fallthrough_addr = emit_word_addr(ctx, word + 2);
            emit_absolute_pc_exit(ctx, fallthrough_addr);
        }
    }
}

/// A nested delay slot whose own raw bits are JR/JALR — always exits via the
/// register-derived target, exactly like the outermost `emit_regjump`, just
/// without that function's own `taken_exit`/edge bookkeeping (never
/// meaningful here — see `emit_nested_branch_slot`'s doc comment).
fn emit_nested_regjump_slot(
    ctx: &mut EmitCtx,
    instrs: &[CompiledInstr; ENTRIES_PER_PAGE],
    word: WordOffset,
    raw: u32,
    regjump: RegJump,
    fr_mode: FrMode,
) {
    let target_addr = emit_read_gpr(ctx, field_rs(raw));

    if regjump.link {
        let this_pc_word = word as i64 + 2;
        emit_write_link_register(ctx, this_pc_word, field_rd(raw));
    }

    let inner_slot_word = word + 1;
    let inner_slot_raw = instrs[inner_slot_word as usize].raw;
    ctx.raw = inner_slot_raw;
    ctx.word = inner_slot_word;
    let slot_terminated = emit_slot_semantics(ctx, instrs, fr_mode, target_addr);
    ctx.raw = raw;
    ctx.word = word;
    if slot_terminated {
        // A still-deeper nested branch/regjump already exited — this
        // RegJump's own register-derived target (read before the slot ran)
        // never takes effect, same nested-supersedes-outer rule as
        // emit_regjump's own top-level version.
        return;
    }

    emit_runtime_pc_exit(ctx, target_addr);
}

/// Evaluate a conditional branch's predicate as an I8 boolean Cranelift
/// value (nonzero = taken), mirroring the interpreter's `exec_be*`/`exec_bl*`
/// comparisons exactly (all against a signed 64-bit `rs`, except Eq/Ne which
/// compare `rs`/`rt` directly with no sign interpretation needed since
/// equality doesn't care).
fn emit_cond(ctx: &mut EmitCtx, raw: u32, cond: BranchCond) -> Value {
    let rs_val = emit_read_gpr(ctx, field_rs(raw));
    match cond {
        BranchCond::Always => unreachable!("Always has no condition to evaluate"),
        BranchCond::Eq => {
            let rt_val = emit_read_gpr(ctx, field_rt(raw));
            ctx.builder.ins().icmp(IntCC::Equal, rs_val, rt_val)
        }
        BranchCond::Ne => {
            let rt_val = emit_read_gpr(ctx, field_rt(raw));
            ctx.builder.ins().icmp(IntCC::NotEqual, rs_val, rt_val)
        }
        BranchCond::LeZero => {
            let zero = ctx.builder.ins().iconst(ir::types::I64, 0);
            ctx.builder.ins().icmp(IntCC::SignedLessThanOrEqual, rs_val, zero)
        }
        BranchCond::GtZero => {
            let zero = ctx.builder.ins().iconst(ir::types::I64, 0);
            ctx.builder.ins().icmp(IntCC::SignedGreaterThan, rs_val, zero)
        }
        BranchCond::LtZero => {
            let zero = ctx.builder.ins().iconst(ir::types::I64, 0);
            ctx.builder.ins().icmp(IntCC::SignedLessThan, rs_val, zero)
        }
        BranchCond::GeZero => {
            let zero = ctx.builder.ins().iconst(ir::types::I64, 0);
            ctx.builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, rs_val, zero)
        }
    }
}

/// Write the link register (r31) for JAL/JALR-shaped instructions: `core.pc
/// = vbase + (link_word * 4)`, matching `exec_jal`'s `self.core.pc + 8`
/// (this instruction's own address plus two words, skipping the delay slot
/// — the return address is the instruction *after* the delay slot). `vbase`
/// is derived from the live `core.pc & !0xFFF`. Must be an add, not an or:
/// unlike every other `vbase | word*4` site in this file, `link_word` here
/// is `entry_word + 2` and can legitimately reach exactly `ENTRIES_PER_PAGE`
/// (0x400) when the branch/jump sits at the page's second-to-last word — the
/// return address then falls on the *next* page. OR silently drops that
/// carry whenever `vbase`'s own bit 12 happens to already be set (observed
/// live: a JAL at word 0x3fe on page ...173000 linked to ...173000 instead
/// of the correct ...174000 — bit 12 of 0x173000 is already 1, so ORing in
/// 0x1000 changed nothing). `vbase` and the link word are still guaranteed
/// within 32 bits of each other (one page apart at most), so a plain add
/// can't spuriously touch the kseg0/kseg1 `0xFFFFFFFF` upper-32 prefix.
fn emit_write_link_register(ctx: &mut EmitCtx, link_word: i64, link_reg: u32) {
    let mem = MemFlagsData::trusted();
    let pc_off = ir::immediates::Offset32::new(core_offset_of_pc());
    let pc = ctx.builder.ins().load(ir::types::I64, mem, ctx.core_ptr, pc_off);
    let vbase = ctx.builder.ins().band_imm_s(pc, !(PAGE_SIZE as i64 - 1));
    let link_addr = ctx.builder.ins().iadd_imm_s(vbase, link_word * 4);
    emit_write_gpr(ctx, link_reg, link_addr);
}

/// Emit one instruction's semantics (§3.1's "unit semantics... committed as
/// one"). Called with `builder` positioned in the instruction's own block
/// (preambles already emitted); must leave `builder` positioned to continue
/// — either still in the same block (falls through to whatever the caller
/// wires next) or already terminated (a branch/jump/regjump emitted its own
/// terminator and jumped/branched away). Returns nothing: control-flow
/// wiring between instructions is the caller's job (`compile_region`), not
/// this function's — this only commits the one unit's data effects, exactly
/// like an interpreter `exec_*` handler's body minus the `ExecStatus` return
/// and PC advance (both handled uniformly by the caller for every
/// instruction shape, not per-emitter).
/// Takes `&mut EmitCtx` — `ctx.raw`/`ctx.word` are this instruction's own
/// encoding and compile-time word offset within the page. Most emitters
/// only touch `ctx.raw`; `emit_load`/`emit_store` (via `emit_exception_exit`)
/// also use `ctx.word`, to synthesize the correct exception PC on a fault
/// instead of leaving `core.pc` stale (see `emit_exception_exit`'s doc
/// comment).
type SemanticsEmitter = fn(&mut EmitCtx);

// ---- CP1 (FPU) ------
//
// Distinct emitter type from `SemanticsEmitter`: every CP1 register access
// needs `fr_mode` (STATUS_FR is resolved once per region at compile time —
// see `FrMode`'s doc comment), which plain integer ops never touch. CP1
// field reuse mirrors the interpreter's `DecodedInstr` layout exactly (see
// mips_exec.rs's decode): fs = rd-position bits[15:11], ft = rt-position
// bits[20:16], fd = sa-position bits[10:6], fr = rs-position bits[25:21] —
// the same `field_rd`/`field_rt`/`field_sa`/`field_rs` helpers already used
// for integer decode, just relabeled at each call site via local `let`
// bindings for readability.
type Cp1Emitter = fn(&mut EmitCtx, fr_mode: FrMode);

/// Binary FP arithmetic op shape shared by ADD/SUB/MUL/DIV, S or D format:
/// read fs/ft, apply `op`, write fd, then `emit_fpu_update_fcsr`. Mirrors
/// `exec_fadd_s`/`exec_fsub_d`/etc.'s common structure exactly — CU1 is
/// already handled by the region-wide entry guard, not repeated per
/// instruction. `flags_fn` computes the [6:2] flags `Value` from the raw
/// operand bits (`emit_fpu_arith_flags_snan_only_s/d` for ADD/SUB/MUL,
/// `emit_fpu_arith_flags_div_s/d` for DIV) — never read from the host FPU.
fn emit_fbinop_s(
    ctx: &mut EmitCtx,
    fr_mode: FrMode,
    op: fn(&mut FunctionBuilder, Value, Value) -> Value,
    flags_fn: fn(&mut EmitCtx, Value, Value) -> Value,
) {
    let raw = ctx.raw;
    let fs = field_rd(raw);
    let ft = field_rt(raw);
    let fd = field_sa(raw);

    let fs_bits = emit_read_fpr_w(ctx, fs, fr_mode);
    let ft_bits = emit_read_fpr_w(ctx, ft, fr_mode);
    emit_check_denorm_operand(ctx, fs_bits, false);
    emit_check_denorm_operand(ctx, ft_bits, false);
    let flags = flags_fn(ctx, fs_bits, ft_bits);
    let fs_val = ctx.builder.ins().bitcast(ir::types::F32, MemFlagsData::new(), fs_bits);
    let ft_val = ctx.builder.ins().bitcast(ir::types::F32, MemFlagsData::new(), ft_bits);
    let result = op(ctx.builder, fs_val, ft_val);
    let result_bits = ctx.builder.ins().bitcast(ir::types::I32, MemFlagsData::new(), result);
    let is_denorm = emit_is_subnormal_s(ctx, result_bits);
    let is_negative = ctx.builder.ins().icmp_imm_s(IntCC::SignedLessThan, result_bits, 0);
    emit_fpu_update_fcsr_arith(
        ctx, flags, is_denorm, is_negative,
        |ctx| emit_write_fpr_w(ctx, fd, result_bits, fr_mode),
        |ctx, neg| {
            let zero = ctx.builder.ins().iconst(ir::types::I32, 0);
            let neg32 = ctx.builder.ins().uextend(ir::types::I32, neg);
            let sign = ctx.builder.ins().ishl_imm_s(neg32, 31);
            let signed_zero = ctx.builder.ins().bor(zero, sign);
            emit_write_fpr_w(ctx, fd, signed_zero, fr_mode)
        },
    );
}
fn emit_fbinop_d(
    ctx: &mut EmitCtx,
    fr_mode: FrMode,
    op: fn(&mut FunctionBuilder, Value, Value) -> Value,
    flags_fn: fn(&mut EmitCtx, Value, Value) -> Value,
) {
    let raw = ctx.raw;
    let fs = field_rd(raw);
    let ft = field_rt(raw);
    let fd = field_sa(raw);

    let fs_bits = emit_read_fpr_l(ctx, fs, fr_mode);
    let ft_bits = emit_read_fpr_l(ctx, ft, fr_mode);
    emit_check_denorm_operand(ctx, fs_bits, true);
    emit_check_denorm_operand(ctx, ft_bits, true);
    let flags = flags_fn(ctx, fs_bits, ft_bits);
    let fs_val = ctx.builder.ins().bitcast(ir::types::F64, MemFlagsData::new(), fs_bits);
    let ft_val = ctx.builder.ins().bitcast(ir::types::F64, MemFlagsData::new(), ft_bits);
    let result = op(ctx.builder, fs_val, ft_val);
    let result_bits = ctx.builder.ins().bitcast(ir::types::I64, MemFlagsData::new(), result);
    let is_denorm = emit_is_subnormal_d(ctx, result_bits);
    let is_negative = ctx.builder.ins().icmp_imm_s(IntCC::SignedLessThan, result_bits, 0);
    emit_fpu_update_fcsr_arith(
        ctx, flags, is_denorm, is_negative,
        |ctx| emit_write_fpr_l(ctx, fd, result_bits, fr_mode),
        |ctx, neg| {
            let zero = ctx.builder.ins().iconst(ir::types::I64, 0);
            let neg64 = ctx.builder.ins().uextend(ir::types::I64, neg);
            let sign = ctx.builder.ins().ishl_imm_s(neg64, 63);
            let signed_zero = ctx.builder.ins().bor(zero, sign);
            emit_write_fpr_l(ctx, fd, signed_zero, fr_mode)
        },
    );
}

fn fop_add(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value { builder.ins().fadd(a, b) }
fn fop_sub(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value { builder.ins().fsub(a, b) }
fn fop_mul(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value { builder.ins().fmul(a, b) }
fn fop_div(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value { builder.ins().fdiv(a, b) }

fn emit_fadd_s(ctx: &mut EmitCtx, fr_mode: FrMode) { emit_fbinop_s(ctx, fr_mode, fop_add, emit_fpu_arith_flags_snan_only_s); }
fn emit_fadd_d(ctx: &mut EmitCtx, fr_mode: FrMode) { emit_fbinop_d(ctx, fr_mode, fop_add, emit_fpu_arith_flags_snan_only_d); }
fn emit_fsub_s(ctx: &mut EmitCtx, fr_mode: FrMode) { emit_fbinop_s(ctx, fr_mode, fop_sub, emit_fpu_arith_flags_snan_only_s); }
fn emit_fsub_d(ctx: &mut EmitCtx, fr_mode: FrMode) { emit_fbinop_d(ctx, fr_mode, fop_sub, emit_fpu_arith_flags_snan_only_d); }
fn emit_fmul_s(ctx: &mut EmitCtx, fr_mode: FrMode) { emit_fbinop_s(ctx, fr_mode, fop_mul, emit_fpu_arith_flags_snan_only_s); }
fn emit_fmul_d(ctx: &mut EmitCtx, fr_mode: FrMode) { emit_fbinop_d(ctx, fr_mode, fop_mul, emit_fpu_arith_flags_snan_only_d); }
fn emit_fdiv_s(ctx: &mut EmitCtx, fr_mode: FrMode) { emit_fbinop_s(ctx, fr_mode, fop_div, emit_fpu_arith_flags_div_s); }
fn emit_fdiv_d(ctx: &mut EmitCtx, fr_mode: FrMode) { emit_fbinop_d(ctx, fr_mode, fop_div, emit_fpu_arith_flags_div_d); }

/// SQRT.S/D fd, fs: fd = sqrt(fs). Unlike the binary ops, only one operand
/// (`fs`; `ft`/`d.rt` is unused/ignored, matching the interpreter). Still
/// goes through `emit_fpu_update_fcsr` (sqrt of a negative raises Invalid).
fn emit_fsqrt_s(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let fd = field_sa(ctx.raw);
    let fs_bits = emit_read_fpr_w(ctx, fs, fr_mode);
    emit_check_denorm_operand(ctx, fs_bits, false);
    let flags = emit_fpu_arith_flags_sqrt_s(ctx, fs_bits);
    let fs_val = ctx.builder.ins().bitcast(ir::types::F32, MemFlagsData::new(), fs_bits);
    let result = ctx.builder.ins().sqrt(fs_val);
    let result_bits = ctx.builder.ins().bitcast(ir::types::I32, MemFlagsData::new(), result);
    let is_denorm = emit_is_subnormal_s(ctx, result_bits);
    let is_negative = ctx.builder.ins().icmp_imm_s(IntCC::SignedLessThan, result_bits, 0);
    emit_fpu_update_fcsr_arith(
        ctx, flags, is_denorm, is_negative,
        |ctx| emit_write_fpr_w(ctx, fd, result_bits, fr_mode),
        |ctx, neg| {
            let zero = ctx.builder.ins().iconst(ir::types::I32, 0);
            let neg32 = ctx.builder.ins().uextend(ir::types::I32, neg);
            let sign = ctx.builder.ins().ishl_imm_s(neg32, 31);
            let signed_zero = ctx.builder.ins().bor(zero, sign);
            emit_write_fpr_w(ctx, fd, signed_zero, fr_mode)
        },
    );
}
fn emit_fsqrt_d(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let fd = field_sa(ctx.raw);
    let fs_bits = emit_read_fpr_l(ctx, fs, fr_mode);
    emit_check_denorm_operand(ctx, fs_bits, true);
    let flags = emit_fpu_arith_flags_sqrt_d(ctx, fs_bits);
    let fs_val = ctx.builder.ins().bitcast(ir::types::F64, MemFlagsData::new(), fs_bits);
    let result = ctx.builder.ins().sqrt(fs_val);
    let result_bits = ctx.builder.ins().bitcast(ir::types::I64, MemFlagsData::new(), result);
    let is_denorm = emit_is_subnormal_d(ctx, result_bits);
    let is_negative = ctx.builder.ins().icmp_imm_s(IntCC::SignedLessThan, result_bits, 0);
    emit_fpu_update_fcsr_arith(
        ctx, flags, is_denorm, is_negative,
        |ctx| emit_write_fpr_l(ctx, fd, result_bits, fr_mode),
        |ctx, neg| {
            let zero = ctx.builder.ins().iconst(ir::types::I64, 0);
            let neg64 = ctx.builder.ins().uextend(ir::types::I64, neg);
            let sign = ctx.builder.ins().ishl_imm_s(neg64, 63);
            let signed_zero = ctx.builder.ins().bor(zero, sign);
            emit_write_fpr_l(ctx, fd, signed_zero, fr_mode)
        },
    );
}

/// ABS.S/D, NEG.S/D, MOV.S/D fd, fs: unlike ADD/SUB/MUL/DIV/SQRT, these
/// never touch FCSR — mirrors `exec_fabs_s`/`exec_fneg_s`/`exec_fmov_s`
/// (and their `_d` counterparts) exactly: no `clear_fpu_status` call, no
/// `fpu_update_fcsr` tail call, just the raw bit/value operation and a
/// plain write. MOV additionally moves the full 64-bit register value via
/// `fpr_read_l`/`fpr_write_l` even in S format (matches
/// `exec_fmov_s`'s use of the `_l` accessors, not `_w`) — a straight
/// register-to-register copy needs no width truncation either way.
fn emit_fabs_s(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let fd = field_sa(ctx.raw);
    let fs_bits = emit_read_fpr_w(ctx, fs, fr_mode);
    let fs_val = ctx.builder.ins().bitcast(ir::types::F32, MemFlagsData::new(), fs_bits);
    let result = ctx.builder.ins().fabs(fs_val);
    let result_bits = ctx.builder.ins().bitcast(ir::types::I32, MemFlagsData::new(), result);
    emit_check_snan_operand(ctx, fs_bits, false, |ctx| emit_write_fpr_w(ctx, fd, result_bits, fr_mode));
}
fn emit_fabs_d(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let fd = field_sa(ctx.raw);
    let fs_bits = emit_read_fpr_l(ctx, fs, fr_mode);
    let fs_val = ctx.builder.ins().bitcast(ir::types::F64, MemFlagsData::new(), fs_bits);
    let result = ctx.builder.ins().fabs(fs_val);
    let result_bits = ctx.builder.ins().bitcast(ir::types::I64, MemFlagsData::new(), result);
    emit_check_snan_operand(ctx, fs_bits, true, |ctx| emit_write_fpr_l(ctx, fd, result_bits, fr_mode));
}
fn emit_fneg_s(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let fd = field_sa(ctx.raw);
    let fs_bits = emit_read_fpr_w(ctx, fs, fr_mode);
    let fs_val = ctx.builder.ins().bitcast(ir::types::F32, MemFlagsData::new(), fs_bits);
    let result = ctx.builder.ins().fneg(fs_val);
    let result_bits = ctx.builder.ins().bitcast(ir::types::I32, MemFlagsData::new(), result);
    emit_check_snan_operand(ctx, fs_bits, false, |ctx| emit_write_fpr_w(ctx, fd, result_bits, fr_mode));
}
fn emit_fneg_d(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let fd = field_sa(ctx.raw);
    let fs_bits = emit_read_fpr_l(ctx, fs, fr_mode);
    let fs_val = ctx.builder.ins().bitcast(ir::types::F64, MemFlagsData::new(), fs_bits);
    let result = ctx.builder.ins().fneg(fs_val);
    let result_bits = ctx.builder.ins().bitcast(ir::types::I64, MemFlagsData::new(), result);
    emit_check_snan_operand(ctx, fs_bits, true, |ctx| emit_write_fpr_l(ctx, fd, result_bits, fr_mode));
}
fn emit_fmov_s(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let fd = field_sa(ctx.raw);
    let value = emit_read_fpr_l(ctx, fs, fr_mode);
    emit_write_fpr_l(ctx, fd, value, fr_mode);
}
fn emit_fmov_d(ctx: &mut EmitCtx, fr_mode: FrMode) {
    emit_fmov_s(ctx, fr_mode);
}

/// MOVCF.fmt fd, fs, cc, tf: fd = fs if FPU condition code `cc` == `tf`
/// (no-op otherwise). Mirrors `MipsExecutor::exec_fmovcf_s`/`_d` exactly —
/// same FCSR cc-bit extraction as `emit_movci` (cc0 at bit 23, cc1..cc7 at
/// bits 24..30, both compile-time constants from the fixed encoding), just
/// gating an FPR-to-FPR copy instead of a GPR write. Like `emit_fmov_s`,
/// format (S vs D) doesn't matter for a raw copy, so both functs share this
/// body via the same full-64-bit-slot `emit_read_fpr_l`/`emit_write_fpr_l`
/// pair `emit_fmov_d` already relies on.
/// Emits the FCSR load + cc-bit compare shared by MOVCF.s/MOVCF.d, and
/// returns the `taken` boolean `Value`. Mirrors `emit_movci`'s identical
/// FCSR bit extraction (cc0 at bit 23, cc1..cc7 at bits 24..30).
fn emit_fmovcf_taken(ctx: &mut EmitCtx) -> Value {
    let cc = (ctx.raw >> 18) & 0x7;
    let tf = ((ctx.raw >> 16) & 0x1) != 0;
    let bit = if cc == 0 { 23 } else { 24 + cc };

    let mem = MemFlagsData::trusted();
    let fcsr = ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, ir::immediates::Offset32::new(core_offset_of_fpu_fcsr()));
    let cc_bit = ctx.builder.ins().ushr_imm_s(fcsr, bit as i64);
    let cc_value = ctx.builder.ins().band_imm_s(cc_bit, 1);
    let want = if tf { 1 } else { 0 };
    ctx.builder.ins().icmp_imm_s(ir::condcodes::IntCC::Equal, cc_value, want)
}

/// MOVCF.S fd, fs, cc, tf: fd = fs if FPU condition code `cc` == `tf`
/// (no-op otherwise). Mirrors `MipsExecutor::exec_fmovcf_s` exactly,
/// including its use of `fpr_read_w`/`fpr_write_w` (32-bit word, not the
/// full 64-bit slot) — unlike `emit_fmov_s`/`_d` (which alias to the same
/// full-slot copy because `exec_fmov_s` itself uses `fpr_read_l`/`_write_l`
/// even for the `.s` funct), `exec_fmovcf_s` and `exec_fmovcf_d` use
/// genuinely different-width accessors, so `.s` and `.d` need separate
/// bodies here.
fn emit_fmovcf_s(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let fd = field_sa(ctx.raw);
    let taken = emit_fmovcf_taken(ctx);

    let write_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();
    ctx.builder.ins().brif(taken, write_block, &[], merge_block, &[]);

    ctx.builder.switch_to_block(write_block);
    ctx.builder.seal_block(write_block);
    let value = emit_read_fpr_w(ctx, fs, fr_mode);
    emit_write_fpr_w(ctx, fd, value, fr_mode);
    ctx.builder.ins().jump(merge_block, &[]);

    ctx.builder.switch_to_block(merge_block);
    ctx.builder.seal_block(merge_block);
}
/// MOVCF.D fd, fs, cc, tf — see `emit_fmovcf_s`'s doc comment for why this
/// isn't just a delegate to it: `exec_fmovcf_d` uses `fpr_read_d`/
/// `fpr_write_d` (full 64-bit slot, same shape as `emit_read_fpr_l`/
/// `emit_write_fpr_l`), not the 32-bit word accessors `.s` uses.
fn emit_fmovcf_d(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let fd = field_sa(ctx.raw);
    let taken = emit_fmovcf_taken(ctx);

    let write_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();
    ctx.builder.ins().brif(taken, write_block, &[], merge_block, &[]);

    ctx.builder.switch_to_block(write_block);
    ctx.builder.seal_block(write_block);
    let value = emit_read_fpr_l(ctx, fs, fr_mode);
    emit_write_fpr_l(ctx, fd, value, fr_mode);
    ctx.builder.ins().jump(merge_block, &[]);

    ctx.builder.switch_to_block(merge_block);
    ctx.builder.seal_block(merge_block);
}

// ---- CP1 conversions ------
//
// Two families, both funneled through the shared `cvt_to_int_and_commit`/
// `cvt_int_to_float_and_commit` call-indirect (core.fpu_cvt_to_int_fn/
// fpu_cvt_int_to_float_fn — see emit_fcvt_to_int/emit_fcvt_from_int) rather
// than emit_fpu_update_fcsr (all conversions touch FCSR, unlike ABS/NEG/MOV):
//   float<->float (CVT.D.S, CVT.S.D): plain widen/narrow, no rounding choice.
//   int<->float (CVT.S/D.W/L, CVT.W/L.S/D, ROUND/TRUNC/CEIL/FLOOR.W/L.S/D):
//     int->float is a plain signed conversion; float->int applies a
//     rounding function first, matching the interpreter's fixed-function
//     choice (`.round()`/`.trunc()`/`.ceil()`/`.floor()` — CVT.W/L itself
//     always uses `.round()`, i.e. the same operation as the ROUND
//     variant; MIPS's dynamic-FCSR-rounding-mode CVT semantics are NOT
//     replicated by this interpreter either, so nothing to diverge from).

/// MIPS FCSR.RM encoding — mirrors `mips_exec.rs`'s `RM_*` constants exactly
/// (kept as a separate copy rather than shared across the crate boundary
/// since this file only needs the raw `u8` values as IR immediates).
const RM_NEAREST_EVEN: i64 = 0;
const RM_TOWARD_ZERO: i64 = 1;
const RM_TOWARD_POS_INF: i64 = 2;
const RM_TOWARD_NEG_INF: i64 = 3;

/// Round `x` (an F32 or F64 SSA value) to the nearest integer *value* (still
/// a float, not yet cast to an integer type) per `rm` (an I8 SSA value
/// carrying the MIPS FCSR.RM encoding above) — pure bit manipulation on the
/// mantissa/exponent, no `ceil`/`floor`/`trunc`/`nearest` IR op anywhere,
/// so it can't lower to a hardware rounding instruction whose result
/// secretly depends on ambient host FPU control-register state.
///
/// This mirrors `mips_exec.rs`'s `round_f32_to_int_mode`/
/// `round_f64_to_int_mode` bit-for-bit (see that function's doc comment for
/// why: `f32`/`f64::round()`/`.trunc()`/`.ceil()`/`.floor()` — and by
/// extension Cranelift's `nearest`/`trunc`/`ceil`/`floor` IR ops, which
/// lower to the same SSE `ROUNDSS`/`ROUNDSD` family — were empirically
/// found to produce a different, wrong answer depending on the host's live
/// MXCSR rounding-control bits on this build, even for MIPS operations that
/// are supposed to always round the same way regardless of FCSR.RM). Kept
/// branchless (`select` instead of control flow) since this runs inside a
/// single Cranelift basic block alongside the rest of the CP1 handler body.
///
/// `rm` is always in `0..=3` at every call site (either an `iconst` for the
/// four fixed ROUND/TRUNC/CEIL/FLOOR modes, or FCSR's live low 2 bits for
/// the two dynamic-RM plain CVT.W/CVT.L instructions), so the branch
/// dispatch only needs 2-bit comparisons, not a full 0..=255 range check.
fn emit_round_to_int_mode(builder: &mut FunctionBuilder, x: Value, rm: Value) -> Value {
    let float_ty = builder.func.dfg.value_type(x);
    let (int_ty, mantissa_bits, exp_bias): (ir::Type, i64, i64) = if float_ty == ir::types::F32 {
        (ir::types::I32, 23, 127)
    } else {
        (ir::types::I64, 52, 1023)
    };

    let bits = builder.ins().bitcast(int_ty, MemFlagsData::new(), x);
    let sign_shift = if int_ty == ir::types::I32 { 31 } else { 63 };
    let sign_bit = builder.ins().ushr_imm_s(bits, sign_shift);
    let is_negative = builder.ins().icmp_imm_s(IntCC::NotEqual, sign_bit, 0);

    let exp_mask: i64 = if int_ty == ir::types::I32 { 0xFF } else { 0x7FF };
    let biased_exp = builder.ins().ushr_imm_s(bits, mantissa_bits);
    let biased_exp = builder.ins().band_imm_s(biased_exp, exp_mask);
    let exp = builder.ins().iadd_imm_s(biased_exp, -exp_bias);

    let rm_tz = builder.ins().iconst(ir::types::I8, RM_TOWARD_ZERO);
    let rm_pi = builder.ins().iconst(ir::types::I8, RM_TOWARD_POS_INF);
    let rm_ni = builder.ins().iconst(ir::types::I8, RM_TOWARD_NEG_INF);
    let is_rm_tz = builder.ins().icmp(IntCC::Equal, rm, rm_tz);
    let is_rm_pi = builder.ins().icmp(IntCC::Equal, rm, rm_pi);
    let is_rm_ni = builder.ins().icmp(IntCC::Equal, rm, rm_ni);
    // Rounds away from zero for this sign under this mode; RM_NEAREST_EVEN
    // is handled by its own tie-break logic per regime below, not here.
    let away_from_zero_for_dir = {
        let is_positive = builder.ins().icmp_imm_s(IntCC::Equal, is_negative, 0);
        let pos_away = builder.ins().band(is_rm_pi, is_positive);
        let neg_away = builder.ins().band(is_rm_ni, is_negative);
        builder.ins().bor(pos_away, neg_away)
    };

    // ---- Regime 1: exp >= mantissa_bits — already integer-valued (or the
    // magnitude is large enough that every mantissa bit is an integer bit),
    // nothing to round.
    let no_frac_bits = builder.ins().icmp_imm_s(IntCC::SignedGreaterThanOrEqual, exp, mantissa_bits);

    // ---- Regime 2: exp < 0 — |x| < 1.0, result is 0 or +-1.
    let is_exp_negative = builder.ins().icmp_imm_s(IntCC::SignedLessThan, exp, 0);
    let mantissa_field_mask: i64 = (1i64 << mantissa_bits) - 1;
    let mantissa_field = builder.ins().band_imm_s(bits, mantissa_field_mask);
    let mantissa_is_zero = builder.ins().icmp_imm_s(IntCC::Equal, mantissa_field, 0);
    let biased_exp_is_zero = builder.ins().icmp_imm_s(IntCC::Equal, biased_exp, 0);
    let is_zero_mantissa_and_exp = builder.ins().band(mantissa_is_zero, biased_exp_is_zero); // x == +-0.0
    let is_exp_minus_one = builder.ins().icmp_imm_s(IntCC::Equal, exp, -1); // |x| in [0.5, 1.0)
    let is_lt_half = builder.ins().icmp_imm_s(IntCC::Equal, is_exp_minus_one, 0); // exp < -1 (or the zero case above)
    // |x| < 0.5 (and not exactly zero): rounds to 0 for TZ/Nearest; for
    // PI/NI, rounds away from zero to +-1 per `away_from_zero_for_dir`.
    let one_f = if float_ty == ir::types::F32 { builder.ins().f32const(1.0) } else { builder.ins().f64const(1.0) };
    let signed_one = builder.ins().fcopysign(one_f, x);
    let zero_f = if float_ty == ir::types::F32 { builder.ins().f32const(0.0) } else { builder.ins().f64const(0.0) };
    let signed_zero = builder.ins().fcopysign(zero_f, x);
    let lt_half_nonzero_result = builder.ins().select(away_from_zero_for_dir, signed_one, signed_zero);
    let lt_half_result = builder.ins().select(is_zero_mantissa_and_exp, x, lt_half_nonzero_result);
    // |x| in [0.5, 1.0) exactly: nearest-even ties round to 0 (the even
    // integer); TZ always 0; PI/NI still governed by `away_from_zero_for_dir`.
    let mantissa_all_zero_at_half = mantissa_is_zero; // reused: |x|==0.5 exactly iff mantissa field is all-zero
    let nearest_rounds_away_at_half = builder.ins().icmp_imm_s(IntCC::Equal, mantissa_all_zero_at_half, 0); // non-exact-half nearest-mode values in [0.5,1) always round away (up to 1)
    let is_rm_nearest = {
        let rm_ne = builder.ins().iconst(ir::types::I8, RM_NEAREST_EVEN);
        builder.ins().icmp(IntCC::Equal, rm, rm_ne)
    };
    let half_regime_away = builder.ins().select(is_rm_nearest, nearest_rounds_away_at_half, away_from_zero_for_dir);
    let half_regime_result = builder.ins().select(half_regime_away, signed_one, signed_zero);
    let exp_negative_result = builder.ins().select(is_lt_half, lt_half_result, half_regime_result);

    // ---- Regime 3 (the common case): 0 <= exp < mantissa_bits — split the
    // mantissa into an integer part and a fractional remainder at bit
    // position (mantissa_bits - exp).
    let mantissa_bits_val = builder.ins().iconst(int_ty, mantissa_bits);
    let frac_bits = builder.ins().isub(mantissa_bits_val, exp); // mantissa_bits - exp, as int_ty
    let one_i = builder.ins().iconst(int_ty, 1);
    let one_shl_frac = builder.ins().ishl(one_i, frac_bits);
    let frac_mask = builder.ins().iadd_imm_s(one_shl_frac, -1);
    let implicit_leading_bit = builder.ins().ishl_imm_s(one_i, mantissa_bits);
    let mantissa_mask = builder.ins().iadd_imm_s(implicit_leading_bit, -1);
    let full_mantissa = builder.ins().band(bits, mantissa_mask);
    let full_mantissa = builder.ins().bor(full_mantissa, implicit_leading_bit);
    let frac = builder.ins().band(full_mantissa, frac_mask);
    let frac_is_zero = builder.ins().icmp_imm_s(IntCC::Equal, frac, 0);

    let not_frac_mask = builder.ins().bnot(frac_mask);
    let truncated_bits = builder.ins().band(bits, not_frac_mask);

    let half = builder.ins().ushr_imm_s(one_shl_frac, 1);
    let frac_gt_half = builder.ins().icmp(IntCC::UnsignedGreaterThan, frac, half);
    let frac_eq_half = builder.ins().icmp(IntCC::Equal, frac, half);
    let truncated_lsb = builder.ins().ushr(full_mantissa, frac_bits);
    let truncated_is_odd = builder.ins().band_imm_s(truncated_lsb, 1);
    let truncated_is_odd = builder.ins().icmp_imm_s(IntCC::NotEqual, truncated_is_odd, 0);
    let tie_rounds_up = builder.ins().band(frac_eq_half, truncated_is_odd);
    let nearest_rounds_up = builder.ins().bor(frac_gt_half, tie_rounds_up);

    let round_up_normal = builder.ins().select(is_rm_nearest, nearest_rounds_up, away_from_zero_for_dir);
    let not_rm_tz = builder.ins().icmp_imm_s(IntCC::Equal, is_rm_tz, 0);
    let round_up_normal = builder.ins().band(round_up_normal, not_rm_tz);

    let incremented_bits = builder.ins().iadd(truncated_bits, one_shl_frac);
    let rounded_bits = builder.ins().select(round_up_normal, incremented_bits, truncated_bits);
    let rounded_bits = builder.ins().select(frac_is_zero, bits, rounded_bits);
    let normal_regime_result = builder.ins().bitcast(float_ty, MemFlagsData::new(), rounded_bits);

    let result = builder.ins().select(is_exp_negative, exp_negative_result, normal_regime_result);
    builder.ins().select(no_frac_bits, x, result)
}

/// ROUND/TRUNC/CEIL/FLOOR pass a fixed MIPS FCSR.RM value regardless of
/// FCSR's live contents (`Fixed`); the two unprefixed CVT.W/CVT.L
/// instructions honor FCSR.RM dynamically instead (`Dynamic`) — see
/// `exec_fcvt_w_s`'s interpreter-side counterpart.
#[derive(Clone, Copy)]
enum RoundMode { Fixed(i64), Dynamic }

/// Shared body for every float-source rounding conversion (ROUND/TRUNC/
/// CEIL/FLOOR/CVT.W/CVT.L, source format S or D, dest width W(i32) or
/// L(i64)). Unlike the old Cranelift-IR-side rounding/saturation, this is a
/// single `call_indirect` into `core.fpu_cvt_to_int_fn`
/// (`mips_exec.rs::jit_cvt_to_int`, which calls the same
/// `cvt_to_int_and_commit` the interpreter's `fpu_cvt_to_int` calls) — no
/// Cranelift-side rounding math, no host MXCSR/FPSR read at all. See
/// `cvt_to_int_and_commit`'s doc comment for the full rationale: reading
/// host FP status after a hardware conversion instruction races the flag
/// write against the read on out-of-order hardware (found live via this
/// file's own equivalence tests), and this also fixes the result itself to
/// real MIPS saturation (always the largest-magnitude representable
/// integer, e.g. 0x7FFFFFFF for a negative overflow too) instead of
/// Rust/Cranelift's own saturate-toward-sign convention. The function reads/
/// writes `core.fpr`/`core.fpu_fcsr` directly, so nothing computed here
/// needs to survive across the call as a cached `Value` — only the plain
/// scalar register indices and mode flags are passed in.
fn emit_fcvt_to_int(ctx: &mut EmitCtx, fr_mode: FrMode, src_f64: bool, dst_i64: bool, mode: RoundMode) {
    let fs = field_rd(ctx.raw);
    let fd = field_sa(ctx.raw);
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();
    let i32t = ir::types::I32;

    let rm = match mode {
        RoundMode::Fixed(rm) => ctx.builder.ins().iconst(i32t, rm),
        RoundMode::Dynamic => {
            let fcsr_off = ir::immediates::Offset32::new(core_offset_of_fpu_fcsr());
            let fcsr = ctx.builder.ins().load(i32t, mem, ctx.core_ptr, fcsr_off);
            ctx.builder.ins().band_imm_s(fcsr, 0x3)
        }
    };
    let fs_val = ctx.builder.ins().iconst(i32t, fs as i64);
    let fd_val = ctx.builder.ins().iconst(i32t, fd as i64);
    let fr1_val = ctx.builder.ins().iconst(i32t, matches!(fr_mode, FrMode::Fr1) as i64);
    let src_f64_val = ctx.builder.ins().iconst(i32t, src_f64 as i64);
    let dst_i64_val = ctx.builder.ins().iconst(i32t, dst_i64 as i64);

    let fn_off = ir::immediates::Offset32::new(core_offset_of_fpu_cvt_to_int_fn());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);
    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty)); // ctx (the MipsExecutor pointer)
    sig.params.push(AbiParam::new(i32t)); // fs_reg
    sig.params.push(AbiParam::new(i32t)); // fd_reg
    sig.params.push(AbiParam::new(i32t)); // fr1
    sig.params.push(AbiParam::new(i32t)); // src_f64
    sig.params.push(AbiParam::new(i32t)); // dst_i64
    sig.params.push(AbiParam::new(i32t)); // rm
    sig.returns.push(AbiParam::new(i32t)); // trapped (nonzero) or not (0)
    let sig_ref = ctx.builder.import_signature(sig);
    let call = ctx.builder.ins().call_indirect(sig_ref, callee, &[ctx.core_ptr, fs_val, fd_val, fr1_val, src_f64_val, dst_i64_val, rm]);
    let trapped = ctx.builder.inst_results(call)[0];
    emit_trap_if_nonzero(ctx, trapped);
}

/// Shared body for int-source-to-float conversions (CVT.S.W/D.W/S.L/D.L):
/// a single `call_indirect` into `core.fpu_cvt_int_to_float_fn`
/// (`mips_exec.rs::jit_cvt_int_to_float`, calling the same
/// `cvt_int_to_float_and_commit` the interpreter's `fpu_cvt_int_to_float`
/// calls) — no Cranelift-side `fcvt_from_sint` + host MXCSR/FPSR read, same
/// race-avoidance shape as `emit_fcvt_to_int`/`cvt_to_int_and_commit`.
fn emit_fcvt_from_int(ctx: &mut EmitCtx, fr_mode: FrMode, src_i64: bool, dst_f64: bool) {
    let fs = field_rd(ctx.raw);
    let fd = field_sa(ctx.raw);
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();
    let i32t = ir::types::I32;

    let fs_val = ctx.builder.ins().iconst(i32t, fs as i64);
    let fd_val = ctx.builder.ins().iconst(i32t, fd as i64);
    let fr1_val = ctx.builder.ins().iconst(i32t, matches!(fr_mode, FrMode::Fr1) as i64);
    let src_i64_val = ctx.builder.ins().iconst(i32t, src_i64 as i64);
    let dst_f64_val = ctx.builder.ins().iconst(i32t, dst_f64 as i64);

    let fn_off = ir::immediates::Offset32::new(core_offset_of_fpu_cvt_int_to_float_fn());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);
    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty));
    sig.params.push(AbiParam::new(i32t));
    sig.params.push(AbiParam::new(i32t));
    sig.params.push(AbiParam::new(i32t));
    sig.params.push(AbiParam::new(i32t));
    sig.params.push(AbiParam::new(i32t));
    sig.returns.push(AbiParam::new(i32t));
    let sig_ref = ctx.builder.import_signature(sig);
    let call = ctx.builder.ins().call_indirect(sig_ref, callee, &[ctx.core_ptr, fs_val, fd_val, fr1_val, src_i64_val, dst_f64_val]);
    let trapped = ctx.builder.inst_results(call)[0];
    emit_trap_if_nonzero(ctx, trapped);
}

/// CVT.D.S: widening float<->float, always exact per the R4000/VR5000
/// manuals (Table 7-2/9-1 list no exception CVT.D.S can raise) — no FCSR
/// interaction and so no host-status-read race to avoid; safe to keep as a
/// plain Cranelift `fpromote`.
fn emit_fcvt_d_s(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let fd = field_sa(ctx.raw);
    let bits = emit_read_fpr_w(ctx, fs, fr_mode);
    let val = ctx.builder.ins().bitcast(ir::types::F32, MemFlagsData::new(), bits);
    let result = ctx.builder.ins().fpromote(ir::types::F64, val);
    let result_bits = ctx.builder.ins().bitcast(ir::types::I64, MemFlagsData::new(), result);
    emit_write_fpr_l(ctx, fd, result_bits, fr_mode);
}
/// CVT.S.D: single `call_indirect` into `core.fpu_cvt_d_to_s_fn`
/// (`mips_exec.rs::jit_cvt_d_to_s`, calling `cvt_d_to_s_and_commit`) — same
/// race-avoidance shape as `emit_fcvt_from_int`/`emit_fcvt_to_int`; unlike
/// the widen direction, narrowing can raise Inexact or Overflow.
fn emit_fcvt_s_d(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let fs = field_rd(ctx.raw);
    let fd = field_sa(ctx.raw);
    let mem = MemFlagsData::trusted();
    let ptr_ty = ctx.module.target_config().pointer_type();
    let i32t = ir::types::I32;

    let fs_val = ctx.builder.ins().iconst(i32t, fs as i64);
    let fd_val = ctx.builder.ins().iconst(i32t, fd as i64);
    let fr1_val = ctx.builder.ins().iconst(i32t, matches!(fr_mode, FrMode::Fr1) as i64);

    let fn_off = ir::immediates::Offset32::new(core_offset_of_fpu_cvt_d_to_s_fn());
    let callee = ctx.builder.ins().load(ptr_ty, mem, ctx.core_ptr, fn_off);
    let mut sig = ctx.module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty));
    sig.params.push(AbiParam::new(i32t));
    sig.params.push(AbiParam::new(i32t));
    sig.params.push(AbiParam::new(i32t));
    sig.returns.push(AbiParam::new(i32t));
    let sig_ref = ctx.builder.import_signature(sig);
    let call = ctx.builder.ins().call_indirect(sig_ref, callee, &[ctx.core_ptr, fs_val, fd_val, fr1_val]);
    let trapped = ctx.builder.inst_results(call)[0];
    emit_trap_if_nonzero(ctx, trapped);
}

/// Shared tail for the `fpu_cvt_*_fn` family: `trapped` is the nonzero-if-
/// trapped `I32` a call just returned. Raises `EXC_FPE` via
/// `emit_exception_exit` on trap; otherwise leaves `ctx.builder` positioned
/// in a new, sealed continuation block, same contract as
/// `emit_fpu_update_fcsr`.
fn emit_trap_if_nonzero(ctx: &mut EmitCtx, trapped: Value) {
    let i32t = ir::types::I32;
    let trapped_bool = ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, trapped, 0);
    let raise_block = ctx.builder.create_block();
    let continue_block = ctx.builder.create_block();
    ctx.builder.ins().brif(trapped_bool, raise_block, &[], continue_block, &[]);

    ctx.builder.switch_to_block(raise_block);
    ctx.builder.set_cold_block(raise_block);
    ctx.builder.seal_block(raise_block);
    let status = crate::mips_exec::exec_exception_const(crate::mips_exec::EXC_FPE);
    let status_val = ctx.builder.ins().iconst(i32t, status as i64);
    emit_exception_exit(ctx, status_val);

    ctx.builder.switch_to_block(continue_block);
    ctx.builder.seal_block(continue_block);
}

fn emit_fcvt_w_s(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, false, false, RoundMode::Dynamic); }
fn emit_fcvt_l_s(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, false, true, RoundMode::Dynamic); }
fn emit_fcvt_w_d(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, true, false, RoundMode::Dynamic); }
fn emit_fcvt_l_d(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, true, true, RoundMode::Dynamic); }

fn emit_fround_w_s(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, false, false, RoundMode::Fixed(RM_NEAREST_EVEN)); }
fn emit_ftrunc_w_s(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, false, false, RoundMode::Fixed(RM_TOWARD_ZERO)); }
fn emit_fceil_w_s(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, false, false, RoundMode::Fixed(RM_TOWARD_POS_INF)); }
fn emit_ffloor_w_s(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, false, false, RoundMode::Fixed(RM_TOWARD_NEG_INF)); }
fn emit_fround_l_s(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, false, true, RoundMode::Fixed(RM_NEAREST_EVEN)); }
fn emit_ftrunc_l_s(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, false, true, RoundMode::Fixed(RM_TOWARD_ZERO)); }
fn emit_fceil_l_s(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, false, true, RoundMode::Fixed(RM_TOWARD_POS_INF)); }
fn emit_ffloor_l_s(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, false, true, RoundMode::Fixed(RM_TOWARD_NEG_INF)); }

fn emit_fround_w_d(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, true, false, RoundMode::Fixed(RM_NEAREST_EVEN)); }
fn emit_ftrunc_w_d(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, true, false, RoundMode::Fixed(RM_TOWARD_ZERO)); }
fn emit_fceil_w_d(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, true, false, RoundMode::Fixed(RM_TOWARD_POS_INF)); }
fn emit_ffloor_w_d(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, true, false, RoundMode::Fixed(RM_TOWARD_NEG_INF)); }
fn emit_fround_l_d(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, true, true, RoundMode::Fixed(RM_NEAREST_EVEN)); }
fn emit_ftrunc_l_d(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, true, true, RoundMode::Fixed(RM_TOWARD_ZERO)); }
fn emit_fceil_l_d(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, true, true, RoundMode::Fixed(RM_TOWARD_POS_INF)); }
fn emit_ffloor_l_d(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_to_int(ctx, f, true, true, RoundMode::Fixed(RM_TOWARD_NEG_INF)); }

fn emit_fcvt_s_w(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_from_int(ctx, f, false, false); }
fn emit_fcvt_d_w(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_from_int(ctx, f, false, true); }
fn emit_fcvt_s_l(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_from_int(ctx, f, true, false); }
fn emit_fcvt_d_l(ctx: &mut EmitCtx, f: FrMode) { emit_fcvt_from_int(ctx, f, true, true); }

// ---- CP1 compares: C.cond.fmt ------

/// Evaluate one of the 16 MIPS FP compare conditions against Cranelift's
/// `fcmp`, matching `fpu_compare_s`/`fpu_compare_d`'s match table exactly.
/// `cond` (`funct & 0xF`) is a compile-time constant (part of the fixed
/// instruction encoding), so this resolves to one `fcmp` (or a constant
/// `false`, for F/SF) — no runtime branch on which condition it is.
/// Cranelift's `fcmp` with `IntCC`-style codes doesn't exist for floats;
/// Cranelift has its own `FloatCC` enum whose `Equal`/`LessThan`/
/// `LessThanOrEqual`/`Unordered`/`NotEqual`... — MIPS's condition set maps
/// directly onto small ORs of `FloatCC::{LessThan,Equal,Unordered}`-shaped
/// primitives the same way `fpu_compare_s` builds `less`/`equal`/`unordered`
/// from three separate host comparisons and ORs them — so this mirrors that
/// exact structure (three `fcmp`s + boolean logic) rather than searching
/// for a single closer `FloatCC` variant, to keep the correspondence to the
/// interpreter's own formula obvious.
fn emit_fpu_compare(builder: &mut FunctionBuilder, a: Value, b: Value, cond: u32) -> Value {
    use cranelift_codegen::ir::condcodes::FloatCC;
    let less = builder.ins().fcmp(FloatCC::LessThan, a, b);
    let equal = builder.ins().fcmp(FloatCC::Equal, a, b);
    let unordered = builder.ins().fcmp(FloatCC::Unordered, a, b);
    let less_or_equal = builder.ins().bor(less, equal);
    let unordered_or_less = builder.ins().bor(unordered, less);
    let unordered_or_equal = builder.ins().bor(unordered, equal);
    let unordered_or_less_or_equal = builder.ins().bor(unordered, less_or_equal);
    let always_false = builder.ins().iconst(ir::types::I8, 0);
    match cond & 0xF {
        0x0 => always_false,        // F
        0x1 => unordered,           // UN
        0x2 => equal,               // EQ
        0x3 => unordered_or_equal,  // UEQ
        0x4 => less,                // OLT
        0x5 => unordered_or_less,   // ULT
        0x6 => less_or_equal,       // OLE
        0x7 => unordered_or_less_or_equal, // ULE
        0x8 => always_false,        // SF
        0x9 => unordered,           // NGLE
        0xA => equal,               // SEQ
        0xB => unordered_or_equal,  // NGL
        0xC => less,                // LT
        0xD => unordered_or_less,   // NGE
        0xE => less_or_equal,       // LE
        0xF => unordered_or_less_or_equal, // NGT
        _ => unreachable!("cond is masked to 4 bits"),
    }
}

/// Shared body for C.cond.S/C.cond.D. Mirrors `exec_fcc_s`/`exec_fcc_d`:
/// CU1 already handled by the region-wide entry guard; signaling
/// comparisons (cond bit 3 set) raise FCSR.V (Cause+Flag) and — if EV
/// (Enable.V) is set — `EXC_FPE` immediately when either operand is NaN;
/// otherwise evaluate the condition and write the selected condition-code
/// bit via the same FCSR bit-math as `MipsCore::set_fpu_cc`.
fn emit_fcc(ctx: &mut EmitCtx, fr_mode: FrMode, is_d: bool) {
    let raw = ctx.raw;
    let fs = field_rd(raw);
    let ft = field_rt(raw);
    let funct = raw & 0x3F;
    let mem = MemFlagsData::trusted();

    let (a_bits, b_bits) = if is_d {
        (emit_read_fpr_l(ctx, fs, fr_mode), emit_read_fpr_l(ctx, ft, fr_mode))
    } else {
        (emit_read_fpr_w(ctx, fs, fr_mode), emit_read_fpr_w(ctx, ft, fr_mode))
    };
    let (a, b) = if is_d {
        (ctx.builder.ins().bitcast(ir::types::F64, MemFlagsData::new(), a_bits),
         ctx.builder.ins().bitcast(ir::types::F64, MemFlagsData::new(), b_bits))
    } else {
        (ctx.builder.ins().bitcast(ir::types::F32, MemFlagsData::new(), a_bits),
         ctx.builder.ins().bitcast(ir::types::F32, MemFlagsData::new(), b_bits))
    };

    // Invalid is raised whenever either operand is a signalling NaN
    // (unconditional on the predicate), and additionally for the eight
    // signalling predicates (funct bit 3) whenever either operand is any
    // NaN — mirrors `exec_fcc_s`/`exec_fcc_d`'s `invalid` computation
    // exactly. `is_snan`: exponent all-1s, mantissa nonzero, mantissa MSB
    // (the "is quiet" bit) clear.
    let is_snan = |b: &mut cranelift_frontend::FunctionBuilder, bits: ir::Value, is_d: bool| -> ir::Value {
        if is_d {
            let exp_mantissa_nonzero = {
                let masked = b.ins().band_imm_s(bits, 0x7FFF_FFFF_FFFF_FFFFu64 as i64);
                let exp_is_max = b.ins().band_imm_s(bits, 0x7FF0_0000_0000_0000u64 as i64);
                let exp_is_max = b.ins().icmp_imm_s(IntCC::Equal, exp_is_max, 0x7FF0_0000_0000_0000u64 as i64);
                let mantissa_nonzero = b.ins().band_imm_s(masked, 0x000F_FFFF_FFFF_FFFFu64 as i64);
                let mantissa_nonzero = b.ins().icmp_imm_s(IntCC::NotEqual, mantissa_nonzero, 0);
                b.ins().band(exp_is_max, mantissa_nonzero)
            };
            let quiet_bit_clear = {
                let quiet_bit = b.ins().band_imm_s(bits, 0x0008_0000_0000_0000u64 as i64);
                b.ins().icmp_imm_s(IntCC::Equal, quiet_bit, 0)
            };
            b.ins().band(exp_mantissa_nonzero, quiet_bit_clear)
        } else {
            let exp_is_max = b.ins().band_imm_s(bits, 0x7F80_0000);
            let exp_is_max = b.ins().icmp_imm_s(IntCC::Equal, exp_is_max, 0x7F80_0000);
            let mantissa_nonzero = b.ins().band_imm_s(bits, 0x007F_FFFF);
            let mantissa_nonzero = b.ins().icmp_imm_s(IntCC::NotEqual, mantissa_nonzero, 0);
            let exp_mantissa_nonzero = b.ins().band(exp_is_max, mantissa_nonzero);
            let quiet_bit = b.ins().band_imm_s(bits, 0x0040_0000);
            let quiet_bit_clear = b.ins().icmp_imm_s(IntCC::Equal, quiet_bit, 0);
            b.ins().band(exp_mantissa_nonzero, quiet_bit_clear)
        }
    };
    let a_snan = is_snan(ctx.builder, a_bits, is_d);
    let b_snan = is_snan(ctx.builder, b_bits, is_d);
    let has_snan = ctx.builder.ins().bor(a_snan, b_snan);

    let invalid = if funct & 0x8 != 0 {
        use cranelift_codegen::ir::condcodes::FloatCC;
        let a_nan = ctx.builder.ins().fcmp(FloatCC::Unordered, a, a);
        let b_nan = ctx.builder.ins().fcmp(FloatCC::Unordered, b, b);
        let either_nan = ctx.builder.ins().bor(a_nan, b_nan);
        ctx.builder.ins().bor(has_snan, either_nan)
    } else {
        has_snan
    };
    let invalid = ctx.builder.ins().uextend(ir::types::I32, invalid);

    // Cause reflects only the last instruction (rewritten every compare,
    // not just when this one raises Invalid), matching the interpreter.
    const FCSR_CM: i64 = 0x0001_f000;
    const FCSR_CV: i64 = 0x0001_0000;
    let fcsr_off = ir::immediates::Offset32::new(core_offset_of_fpu_fcsr());
    let fcsr = ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, fcsr_off);
    let fcsr_cause_cleared = ctx.builder.ins().band_imm_s(fcsr, !FCSR_CM);
    let cv_bit = ctx.builder.ins().ishl_imm_s(invalid, 16); // FCSR_CV = 1<<16
    let fcsr_with_cause = ctx.builder.ins().bor(fcsr_cause_cleared, cv_bit);
    ctx.builder.ins().store(mem, fcsr_with_cause, ctx.core_ptr, fcsr_off);

    let invalid_nonzero = ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, invalid, 0);
    let raise_v_block = ctx.builder.create_block();
    let after_v_block = ctx.builder.create_block();
    ctx.builder.ins().brif(invalid_nonzero, raise_v_block, &[], after_v_block, &[]);

    // Cold: a signaling-NaN operand on a compare is a rare edge case.
    ctx.builder.switch_to_block(raise_v_block);
    ctx.builder.set_cold_block(raise_v_block);
    ctx.builder.seal_block(raise_v_block);
    let ev_set = ctx.builder.ins().band_imm_s(fcsr_with_cause, 0x800);
    let ev_nonzero = ctx.builder.ins().icmp_imm_s(IntCC::NotEqual, ev_set, 0);
    let raise_fpe_block = ctx.builder.create_block();
    let no_raise_block = ctx.builder.create_block();
    ctx.builder.ins().brif(ev_nonzero, raise_fpe_block, &[], no_raise_block, &[]);

    // Cold: reached only after the already-rare signaling-NaN case above.
    ctx.builder.switch_to_block(raise_fpe_block);
    ctx.builder.set_cold_block(raise_fpe_block);
    ctx.builder.seal_block(raise_fpe_block);
    // Trapped: Cause is already set above, but the sticky Flag field is
    // not — R4000 manual: flag bits are not set when an exception is taken.
    let status = crate::mips_exec::exec_exception_const(crate::mips_exec::EXC_FPE);
    let status_val = ctx.builder.ins().iconst(ir::types::I32, status as i64);
    emit_exception_exit(ctx, status_val);

    ctx.builder.switch_to_block(no_raise_block);
    ctx.builder.seal_block(no_raise_block);
    // Untrapped: also set the sticky Flag.V bit.
    let fcsr_with_flag = ctx.builder.ins().bor_imm_s(fcsr_with_cause, FCSR_CV >> 10);
    ctx.builder.ins().store(mem, fcsr_with_flag, ctx.core_ptr, fcsr_off);
    ctx.builder.ins().jump(after_v_block, &[]);

    ctx.builder.switch_to_block(after_v_block);
    ctx.builder.seal_block(after_v_block);

    let cond_result = emit_fpu_compare(ctx.builder, a, b, funct);
    let cond_i32 = ctx.builder.ins().uextend(ir::types::I32, cond_result);

    // set_fpu_cc: cc = (raw >> 8) & 0x7; bit = if cc==0 {23} else {24+cc}.
    let cc = (raw >> 8) & 0x7;
    let bit = if cc == 0 { 23 } else { 24 + cc };
    let fcsr = ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, fcsr_off);
    let cleared = ctx.builder.ins().band_imm_s(fcsr, !(1i64 << bit));
    let bit_val = ctx.builder.ins().ishl_imm_s(cond_i32, bit as i64);
    let new_fcsr = ctx.builder.ins().bor(cleared, bit_val);
    ctx.builder.ins().store(mem, new_fcsr, ctx.core_ptr, fcsr_off);

    // set_fpu_cc also refreshes fpu_fccr from the new FCSR (fccr_from_fcsr).
    let cc0 = ctx.builder.ins().ushr_imm_s(new_fcsr, 23);
    let cc0 = ctx.builder.ins().band_imm_s(cc0, 1);
    let cc1_7 = ctx.builder.ins().ushr_imm_s(new_fcsr, 25);
    let cc1_7 = ctx.builder.ins().band_imm_s(cc1_7, 0x7F);
    let cc1_7_shifted = ctx.builder.ins().ishl_imm_s(cc1_7, 1);
    let fccr = ctx.builder.ins().bor(cc0, cc1_7_shifted);
    let fccr_off = ir::immediates::Offset32::new(core_offset_of_fpu_fccr());
    ctx.builder.ins().store(mem, fccr, ctx.core_ptr, fccr_off);
}

fn emit_fcc_s(ctx: &mut EmitCtx, fr_mode: FrMode) {
    emit_fcc(ctx, fr_mode, false);
}
fn emit_fcc_d(ctx: &mut EmitCtx, fr_mode: FrMode) {
    emit_fcc(ctx, fr_mode, true);
}

/// Look up the CP1 semantics emitter for a decoded instruction word, or
/// `None` if unimplemented. Only recognizes `Sequential`-shaped CP1 ops
/// (arithmetic/convert/compare/move) — `RS_BC1` is `Excluded` by the
/// analyzer and never reaches codegen at all (§4.4).
fn lookup_cp1_semantics(raw: u32) -> Option<Cp1Emitter> {
    use crate::mips_isa::*;
    let op = (raw >> 26) & 0x3F;
    // LWC1/LDC1/SWC1/SDC1 are architecturally plain memory ops (separate
    // top-level opcodes, not OP_COP1-encoded), but they read/write an FPR —
    // routed through this table anyway, not lookup_semantics, so (a)
    // is_fpu_instruction's single check keeps being the one true trigger for
    // emit_fr_mode_guard's region-wide FR check (a region containing only
    // e.g. LWC1 must still get it) and (b) they go through the same
    // per-instruction emit_cp1_cu1_guard call every other CP1-table entry
    // does (pass 2's `lookup_cp1_semantics` arm) — putting these in the
    // integer table would silently skip both.
    match op {
        OP_LWC1 => return Some(emit_lwc1),
        OP_LDC1 => return Some(emit_ldc1),
        OP_SWC1 => return Some(emit_swc1),
        OP_SDC1 => return Some(emit_sdc1),
        _ => {}
    }
    if op != OP_COP1 {
        return None;
    }
    let rs = (raw >> 21) & 0x1F; // fmt selector
    let funct = raw & 0x3F;
    match rs {
        RS_S => match funct {
            FUNCT_FADD => Some(emit_fadd_s),
            FUNCT_FSUB => Some(emit_fsub_s),
            FUNCT_FMUL => Some(emit_fmul_s),
            FUNCT_FDIV => Some(emit_fdiv_s),
            FUNCT_FSQRT => Some(emit_fsqrt_s),
            FUNCT_FABS => Some(emit_fabs_s),
            FUNCT_FNEG => Some(emit_fneg_s),
            FUNCT_FMOV => Some(emit_fmov_s),
            // MOVF.fmt/MOVT.fmt is MIPS IV; gate like MOVZ/MOVN/MOVCI above.
            #[cfg(feature = "mips4")]
            FUNCT_FMOVCF => Some(emit_fmovcf_s),
            #[cfg(not(feature = "mips4"))]
            FUNCT_FMOVCF => None,
            FUNCT_FCVT_D => Some(emit_fcvt_d_s),
            FUNCT_FCVT_W => Some(emit_fcvt_w_s),
            FUNCT_FCVT_L => Some(emit_fcvt_l_s),
            FUNCT_FROUND_W => Some(emit_fround_w_s),
            FUNCT_FTRUNC_W => Some(emit_ftrunc_w_s),
            FUNCT_FCEIL_W => Some(emit_fceil_w_s),
            FUNCT_FFLOOR_W => Some(emit_ffloor_w_s),
            FUNCT_FROUND_L => Some(emit_fround_l_s),
            FUNCT_FTRUNC_L => Some(emit_ftrunc_l_s),
            FUNCT_FCEIL_L => Some(emit_fceil_l_s),
            FUNCT_FFLOOR_L => Some(emit_ffloor_l_s),
            FUNCT_FC_F..=FUNCT_FC_NGT => Some(emit_fcc_s),
            _ => None,
        },
        RS_D => match funct {
            FUNCT_FADD => Some(emit_fadd_d),
            FUNCT_FSUB => Some(emit_fsub_d),
            FUNCT_FMUL => Some(emit_fmul_d),
            FUNCT_FDIV => Some(emit_fdiv_d),
            FUNCT_FSQRT => Some(emit_fsqrt_d),
            FUNCT_FABS => Some(emit_fabs_d),
            FUNCT_FNEG => Some(emit_fneg_d),
            FUNCT_FMOV => Some(emit_fmov_d),
            #[cfg(feature = "mips4")]
            FUNCT_FMOVCF => Some(emit_fmovcf_d),
            #[cfg(not(feature = "mips4"))]
            FUNCT_FMOVCF => None,
            FUNCT_FCVT_S => Some(emit_fcvt_s_d),
            FUNCT_FCVT_W => Some(emit_fcvt_w_d),
            FUNCT_FCVT_L => Some(emit_fcvt_l_d),
            FUNCT_FROUND_W => Some(emit_fround_w_d),
            FUNCT_FTRUNC_W => Some(emit_ftrunc_w_d),
            FUNCT_FCEIL_W => Some(emit_fceil_w_d),
            FUNCT_FFLOOR_W => Some(emit_ffloor_w_d),
            FUNCT_FROUND_L => Some(emit_fround_l_d),
            FUNCT_FTRUNC_L => Some(emit_ftrunc_l_d),
            FUNCT_FCEIL_L => Some(emit_fceil_l_d),
            FUNCT_FFLOOR_L => Some(emit_ffloor_l_d),
            FUNCT_FC_F..=FUNCT_FC_NGT => Some(emit_fcc_d),
            _ => None,
        },
        RS_W => match funct {
            FUNCT_FCVT_S => Some(emit_fcvt_s_w),
            FUNCT_FCVT_D => Some(emit_fcvt_d_w),
            _ => None,
        },
        RS_L => match funct {
            FUNCT_FCVT_S => Some(emit_fcvt_s_l),
            FUNCT_FCVT_D => Some(emit_fcvt_d_l),
            _ => None,
        },
        RS_MFC1 => Some(emit_mfc1),
        RS_DMFC1 => Some(emit_dmfc1),
        RS_CFC1 => Some(emit_cfc1),
        RS_MTC1 => Some(emit_mtc1),
        RS_DMTC1 => Some(emit_dmtc1),
        RS_CTC1 => Some(emit_ctc1),
        _ => None,
    }
}

fn emit_addu(ctx: &mut EmitCtx) {
    // ADDU rd, rs, rt: rd = sign_extend32(rs[31:0] + rt[31:0]), no overflow trap.
    // Mirrors MipsExecutor::exec_addu exactly: truncate both operands to
    // u32, wrapping_add, then sign-extend the 32-bit result back to 64 bits.
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rs_32 = ctx.builder.ins().ireduce(ir::types::I32, rs_val);
    let rt_32 = ctx.builder.ins().ireduce(ir::types::I32, rt_val);
    let sum_32 = ctx.builder.ins().iadd(rs_32, rt_32);
    let sum_64 = ctx.builder.ins().sextend(ir::types::I64, sum_32);
    emit_write_gpr(ctx, field_rd(ctx.raw), sum_64);
}

/// ADD rd, rs, rt: rd = rs[31:0] + rt[31:0] (signed), trapping on 32-bit
/// signed overflow — mirrors `MipsExecutor::exec_add`'s `i32::checked_add`.
/// Uses Cranelift's overflow-detecting add (`iadd_ifcout` — I32 result plus
/// an overflow flag in one instruction) rather than a separate compare
/// against min/max bounds.
fn emit_add(ctx: &mut EmitCtx) {
    emit_add_sub_trapping(ctx, false);
}
/// SUB rd, rs, rt: rd = rs[31:0] - rt[31:0] (signed), trapping on 32-bit
/// signed overflow — mirrors `MipsExecutor::exec_sub`'s `i32::checked_sub`.
fn emit_sub(ctx: &mut EmitCtx) {
    emit_add_sub_trapping(ctx, true);
}

fn emit_add_sub_trapping(ctx: &mut EmitCtx, is_sub: bool) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rs_32 = ctx.builder.ins().ireduce(ir::types::I32, rs_val);
    let rt_32 = ctx.builder.ins().ireduce(ir::types::I32, rt_val);

    let (result_32, overflow) = if is_sub {
        ctx.builder.ins().ssub_overflow(rs_32, rt_32)
    } else {
        ctx.builder.ins().sadd_overflow(rs_32, rt_32)
    };

    let ok_block = ctx.builder.create_block();
    let trap_block = ctx.builder.create_block();
    ctx.builder.ins().brif(overflow, trap_block, &[], ok_block, &[]);

    // Cold: signed overflow is the rare case for ADD/SUB.
    ctx.builder.switch_to_block(trap_block);
    ctx.builder.set_cold_block(trap_block);
    ctx.builder.seal_block(trap_block);
    let status = crate::mips_exec::exec_exception_const(crate::mips_exec::EXC_OV);
    let status_val = ctx.builder.ins().iconst(ir::types::I32, status as i64);
    emit_exception_exit(ctx, status_val);

    ctx.builder.switch_to_block(ok_block);
    ctx.builder.seal_block(ok_block);
    let result_64 = ctx.builder.ins().sextend(ir::types::I64, result_32);
    emit_write_gpr(ctx, field_rd(ctx.raw), result_64);
}

fn emit_subu(ctx: &mut EmitCtx) {
    // SUBU rd, rs, rt: rd = sign_extend32(rs[31:0] - rt[31:0]), no overflow trap.
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rs_32 = ctx.builder.ins().ireduce(ir::types::I32, rs_val);
    let rt_32 = ctx.builder.ins().ireduce(ir::types::I32, rt_val);
    let diff_32 = ctx.builder.ins().isub(rs_32, rt_32);
    let diff_64 = ctx.builder.ins().sextend(ir::types::I64, diff_32);
    emit_write_gpr(ctx, field_rd(ctx.raw), diff_64);
}

fn emit_and(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let result = ctx.builder.ins().band(rs_val, rt_val);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_or(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let result = ctx.builder.ins().bor(rs_val, rt_val);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_xor(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let result = ctx.builder.ins().bxor(rs_val, rt_val);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_nor(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let or_val = ctx.builder.ins().bor(rs_val, rt_val);
    let result = ctx.builder.ins().bnot(or_val);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_slt(ctx: &mut EmitCtx) {
    // SLT rd, rs, rt: rd = (rs <s rt) ? 1 : 0 (signed 64-bit compare).
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let lt = ctx.builder.ins().icmp(IntCC::SignedLessThan, rs_val, rt_val);
    let result = ctx.builder.ins().uextend(ir::types::I64, lt);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_sltu(ctx: &mut EmitCtx) {
    // SLTU rd, rs, rt: rd = (rs <u rt) ? 1 : 0 (unsigned 64-bit compare).
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let lt = ctx.builder.ins().icmp(IntCC::UnsignedLessThan, rs_val, rt_val);
    let result = ctx.builder.ins().uextend(ir::types::I64, lt);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_sll(ctx: &mut EmitCtx) {
    // SLL rd, rt, sa: rd = sign_extend32(rt[31:0] << sa).
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rt_32 = ctx.builder.ins().ireduce(ir::types::I32, rt_val);
    let shifted = ctx.builder.ins().ishl_imm_s(rt_32, field_sa(ctx.raw) as i64);
    let result = ctx.builder.ins().sextend(ir::types::I64, shifted);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_srl(ctx: &mut EmitCtx) {
    // SRL rd, rt, sa: rd = sign_extend32((u32)rt[31:0] >> sa) — logical shift.
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rt_32 = ctx.builder.ins().ireduce(ir::types::I32, rt_val);
    let shifted = ctx.builder.ins().ushr_imm_s(rt_32, field_sa(ctx.raw) as i64);
    let result = ctx.builder.ins().sextend(ir::types::I64, shifted);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_sra(ctx: &mut EmitCtx) {
    // SRA rd, rt, sa: rd = sign_extend32((i32)rt[31:0] >> sa) — arithmetic shift.
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rt_32 = ctx.builder.ins().ireduce(ir::types::I32, rt_val);
    let shifted = ctx.builder.ins().sshr_imm_s(rt_32, field_sa(ctx.raw) as i64);
    let result = ctx.builder.ins().sextend(ir::types::I64, shifted);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_sllv(ctx: &mut EmitCtx) {
    // SLLV rd, rt, rs: rd = sign_extend32(rt[31:0] << (rs & 0x1F)).
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rt_32 = ctx.builder.ins().ireduce(ir::types::I32, rt_val);
    let sa = ctx.builder.ins().band_imm_s(rs_val, 0x1F);
    let sa_32 = ctx.builder.ins().ireduce(ir::types::I32, sa);
    let shifted = ctx.builder.ins().ishl(rt_32, sa_32);
    let result = ctx.builder.ins().sextend(ir::types::I64, shifted);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_srlv(ctx: &mut EmitCtx) {
    // SRLV rd, rt, rs: rd = sign_extend32((u32)rt[31:0] >> (rs & 0x1F)).
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rt_32 = ctx.builder.ins().ireduce(ir::types::I32, rt_val);
    let sa = ctx.builder.ins().band_imm_s(rs_val, 0x1F);
    let sa_32 = ctx.builder.ins().ireduce(ir::types::I32, sa);
    let shifted = ctx.builder.ins().ushr(rt_32, sa_32);
    let result = ctx.builder.ins().sextend(ir::types::I64, shifted);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_srav(ctx: &mut EmitCtx) {
    // SRAV rd, rt, rs: rd = sign_extend32((i32)rt[31:0] >> (rs & 0x1F)).
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rt_32 = ctx.builder.ins().ireduce(ir::types::I32, rt_val);
    let sa = ctx.builder.ins().band_imm_s(rs_val, 0x1F);
    let sa_32 = ctx.builder.ins().ireduce(ir::types::I32, sa);
    let shifted = ctx.builder.ins().sshr(rt_32, sa_32);
    let result = ctx.builder.ins().sextend(ir::types::I64, shifted);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_mfhi(ctx: &mut EmitCtx) {
    let hi = emit_read_hi(ctx);
    emit_write_gpr(ctx, field_rd(ctx.raw), hi);
}
fn emit_mthi(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    emit_write_hi(ctx, rs_val);
}
fn emit_mflo(ctx: &mut EmitCtx) {
    let lo = emit_read_lo(ctx);
    emit_write_gpr(ctx, field_rd(ctx.raw), lo);
}
fn emit_mtlo(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    emit_write_lo(ctx, rs_val);
}

/// MULT rs, rt: {hi,lo} = sext32(rs) * sext32(rt) (signed 32x32->64).
/// Mirrors `MipsExecutor::exec_mult`: both operands narrowed to their low 32
/// bits and sign-extended to i64 first (so the i64 multiply can't overflow),
/// then the 64-bit product's low/high 32-bit halves each get independently
/// sign-extended back to 64 bits for lo/hi — NOT a plain 128-bit split,
/// which is why this doesn't reuse `smulhi`/widening multiply idioms.
fn emit_mult(ctx: &mut EmitCtx) {
    emit_mult_impl(ctx, true);
}
fn emit_multu(ctx: &mut EmitCtx) {
    emit_mult_impl(ctx, false);
}
fn emit_mult_impl(ctx: &mut EmitCtx, signed: bool) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rs_32 = ctx.builder.ins().ireduce(ir::types::I32, rs_val);
    let rt_32 = ctx.builder.ins().ireduce(ir::types::I32, rt_val);
    let (rs_64, rt_64) = if signed {
        (ctx.builder.ins().sextend(ir::types::I64, rs_32), ctx.builder.ins().sextend(ir::types::I64, rt_32))
    } else {
        (ctx.builder.ins().uextend(ir::types::I64, rs_32), ctx.builder.ins().uextend(ir::types::I64, rt_32))
    };
    let product = ctx.builder.ins().imul(rs_64, rt_64);

    let lo_32 = ctx.builder.ins().ireduce(ir::types::I32, product);
    let lo = ctx.builder.ins().sextend(ir::types::I64, lo_32);
    let hi_shifted = ctx.builder.ins().ushr_imm_s(product, 32);
    let hi_32 = ctx.builder.ins().ireduce(ir::types::I32, hi_shifted);
    let hi = ctx.builder.ins().sextend(ir::types::I64, hi_32);

    emit_write_lo(ctx, lo);
    emit_write_hi(ctx, hi);
}

/// DIV rs, rt: {lo,hi} = {quotient, remainder} of sext32(rs) / sext32(rt)
/// (signed, wrapping — `wrapping_div`/`wrapping_rem` so `i32::MIN / -1`
/// doesn't panic-overflow, matching `exec_div`). Divide-by-zero is a no-op
/// (hi/lo left unchanged) — MIPS leaves this architecturally undefined and
/// the interpreter chooses "do nothing" rather than trapping; mirrored
/// exactly rather than picking a different (also-valid) undefined behavior.
fn emit_div(ctx: &mut EmitCtx) {
    emit_div_impl(ctx, true);
}
fn emit_divu(ctx: &mut EmitCtx) {
    emit_div_impl(ctx, false);
}
fn emit_div_impl(ctx: &mut EmitCtx, signed: bool) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rs_32 = ctx.builder.ins().ireduce(ir::types::I32, rs_val);
    let rt_32 = ctx.builder.ins().ireduce(ir::types::I32, rt_val);

    let zero = ctx.builder.ins().iconst(ir::types::I32, 0);
    let is_zero = ctx.builder.ins().icmp(IntCC::Equal, rt_32, zero);

    let divide_block = ctx.builder.create_block();
    let skip_block = ctx.builder.create_block();
    ctx.builder.ins().brif(is_zero, skip_block, &[], divide_block, &[]);

    ctx.builder.switch_to_block(divide_block);
    ctx.builder.seal_block(divide_block);

    if signed {
        // i32::MIN / -1 traps (#DE) on Cranelift's plain sdiv/srem (they
        // lower straight to the host idiv, which faults on this input) —
        // unlike Rust's wrapping_div/wrapping_rem, which exec_div relies on
        // to define this case as (i32::MIN, 0) without panicking. Must be
        // special-cased explicitly; this is not optional/defensive, it's
        // the only way to reach exec_div's actual defined behavior here.
        let min = ctx.builder.ins().iconst(ir::types::I32, i32::MIN as i64);
        let neg1 = ctx.builder.ins().iconst(ir::types::I32, -1);
        let rs_is_min = ctx.builder.ins().icmp(IntCC::Equal, rs_32, min);
        let rt_is_neg1 = ctx.builder.ins().icmp(IntCC::Equal, rt_32, neg1);
        let is_min_over_neg1 = ctx.builder.ins().band(rs_is_min, rt_is_neg1);

        let overflow_block = ctx.builder.create_block();
        let normal_div_block = ctx.builder.create_block();
        ctx.builder.ins().brif(is_min_over_neg1, overflow_block, &[], normal_div_block, &[]);

        // Cold: i32::MIN / -1 is a narrow edge case, rare relative to
        // ordinary division.
        ctx.builder.switch_to_block(overflow_block);
        ctx.builder.set_cold_block(overflow_block);
        ctx.builder.seal_block(overflow_block);
        let lo = ctx.builder.ins().iconst(ir::types::I64, i32::MIN as i64);
        let hi = ctx.builder.ins().iconst(ir::types::I64, 0);
        emit_write_lo(ctx, lo);
        emit_write_hi(ctx, hi);
        ctx.builder.ins().jump(skip_block, &[]);

        ctx.builder.switch_to_block(normal_div_block);
        ctx.builder.seal_block(normal_div_block);
        let quotient = ctx.builder.ins().sdiv(rs_32, rt_32);
        let remainder = ctx.builder.ins().srem(rs_32, rt_32);
        let lo = ctx.builder.ins().sextend(ir::types::I64, quotient);
        let hi = ctx.builder.ins().sextend(ir::types::I64, remainder);
        emit_write_lo(ctx, lo);
        emit_write_hi(ctx, hi);
        ctx.builder.ins().jump(skip_block, &[]);
    } else {
        // Unsigned division has no equivalent overflow case (there is no
        // -1/MIN in an unsigned range) — udiv/urem are safe with only the
        // zero-divisor guard already in place above.
        let quotient = ctx.builder.ins().udiv(rs_32, rt_32);
        let remainder = ctx.builder.ins().urem(rs_32, rt_32);
        let lo = ctx.builder.ins().sextend(ir::types::I64, quotient);
        let hi = ctx.builder.ins().sextend(ir::types::I64, remainder);
        emit_write_lo(ctx, lo);
        emit_write_hi(ctx, hi);
        ctx.builder.ins().jump(skip_block, &[]);
    }

    ctx.builder.switch_to_block(skip_block);
    ctx.builder.seal_block(skip_block);
}

// ---- Batch 5: 64-bit ALU ops ------
// Full-width equivalents of the 32-bit ALU ops above: no truncate-to-I32 /
// re-extend dance, since the result is already the architectural 64-bit
// value. DADD/DSUB trap on 64-bit signed overflow (mirrors exec_dadd/
// exec_dsub's i64::checked_add/checked_sub); DADDU/DSUBU wrap.

fn emit_daddu(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let result = ctx.builder.ins().iadd(rs_val, rt_val);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_dsubu(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let result = ctx.builder.ins().isub(rs_val, rt_val);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_dadd(ctx: &mut EmitCtx) {
    emit_dadd_dsub_trapping(ctx, false);
}
fn emit_dsub(ctx: &mut EmitCtx) {
    emit_dadd_dsub_trapping(ctx, true);
}
fn emit_dadd_dsub_trapping(ctx: &mut EmitCtx, is_sub: bool) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let (result, overflow) = if is_sub {
        ctx.builder.ins().ssub_overflow(rs_val, rt_val)
    } else {
        ctx.builder.ins().sadd_overflow(rs_val, rt_val)
    };

    let ok_block = ctx.builder.create_block();
    let trap_block = ctx.builder.create_block();
    ctx.builder.ins().brif(overflow, trap_block, &[], ok_block, &[]);

    // Cold: signed overflow is the rare case for this add/sub-family trap.
    ctx.builder.switch_to_block(trap_block);
    ctx.builder.set_cold_block(trap_block);
    ctx.builder.seal_block(trap_block);
    let status = crate::mips_exec::exec_exception_const(crate::mips_exec::EXC_OV);
    let status_val = ctx.builder.ins().iconst(ir::types::I32, status as i64);
    emit_exception_exit(ctx, status_val);

    ctx.builder.switch_to_block(ok_block);
    ctx.builder.seal_block(ok_block);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_dsll(ctx: &mut EmitCtx) {
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let result = ctx.builder.ins().ishl_imm_s(rt_val, field_sa(ctx.raw) as i64);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}
fn emit_dsrl(ctx: &mut EmitCtx) {
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let result = ctx.builder.ins().ushr_imm_s(rt_val, field_sa(ctx.raw) as i64);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}
fn emit_dsra(ctx: &mut EmitCtx) {
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let result = ctx.builder.ins().sshr_imm_s(rt_val, field_sa(ctx.raw) as i64);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}
/// DSLL32/DSRL32/DSRA32: shift amount is `sa + 32` — the "32" variants exist
/// because `sa` is only 5 bits (max 31) but a 64-bit shift needs up to 63;
/// mirrors exec_dsll32/exec_dsrl32/exec_dsra32 exactly.
fn emit_dsll32(ctx: &mut EmitCtx) {
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let result = ctx.builder.ins().ishl_imm_s(rt_val, field_sa(ctx.raw) as i64 + 32);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}
fn emit_dsrl32(ctx: &mut EmitCtx) {
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let result = ctx.builder.ins().ushr_imm_s(rt_val, field_sa(ctx.raw) as i64 + 32);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}
fn emit_dsra32(ctx: &mut EmitCtx) {
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let result = ctx.builder.ins().sshr_imm_s(rt_val, field_sa(ctx.raw) as i64 + 32);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

fn emit_dsllv(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let sa = ctx.builder.ins().band_imm_s(rs_val, 0x3F);
    let result = ctx.builder.ins().ishl(rt_val, sa);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}
fn emit_dsrlv(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let sa = ctx.builder.ins().band_imm_s(rs_val, 0x3F);
    let result = ctx.builder.ins().ushr(rt_val, sa);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}
fn emit_dsrav(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let sa = ctx.builder.ins().band_imm_s(rs_val, 0x3F);
    let result = ctx.builder.ins().sshr(rt_val, sa);
    emit_write_gpr(ctx, field_rd(ctx.raw), result);
}

/// DMULT rs, rt: {hi,lo} = full 128-bit product of rs * rt (signed 64x64).
/// Mirrors `MipsExecutor::exec_dmult`'s `i128` product exactly — unlike
/// `MULT`'s "widen 32-bit operands to i64, multiply, split the 64-bit
/// result" trick (which works because a 32x32 product always fits in 64
/// bits), a 64x64 product can be a genuine 128-bit value with no native
/// I128 arithmetic needed: `imul` gives the low 64 bits directly, and
/// Cranelift's `smulhi`/`umulhi` compute the high 64 bits of a widening
/// multiply in one instruction (typically a single `mul`/`imul` +
/// high-half-register host instruction) — together exactly the 128-bit
/// product's two halves, no I128 type or manual widening required.
fn emit_dmult(ctx: &mut EmitCtx) {
    emit_dmult_impl(ctx, true);
}
fn emit_dmultu(ctx: &mut EmitCtx) {
    emit_dmult_impl(ctx, false);
}
fn emit_dmult_impl(ctx: &mut EmitCtx, signed: bool) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let lo = ctx.builder.ins().imul(rs_val, rt_val);
    let hi = if signed {
        ctx.builder.ins().smulhi(rs_val, rt_val)
    } else {
        ctx.builder.ins().umulhi(rs_val, rt_val)
    };
    emit_write_lo(ctx, lo);
    emit_write_hi(ctx, hi);
}

/// DDIV rs, rt: {lo,hi} = {quotient, remainder} of rs / rt (signed 64-bit).
/// Mirrors `MipsExecutor::exec_ddiv` exactly, including its own explicit
/// "do nothing" arms: divide-by-zero AND `i64::MIN / -1` are both no-ops
/// here (hi/lo left unchanged) — unlike `DIV`'s 32-bit version (which
/// computes a defined `wrapping_div`/`wrapping_rem` result for the
/// MIN/-1 case, since 32-bit wrapping division has a well-defined answer),
/// `exec_ddiv` chose to treat 64-bit MIN/-1 the same as divide-by-zero
/// rather than compute `i64::MIN.wrapping_div(-1)` — this mirrors that
/// choice bit-for-bit rather than reusing DIV's MIN/-1 special-case
/// pattern, which would produce a different (also technically valid, but
/// not what the interpreter does) result.
fn emit_ddiv(ctx: &mut EmitCtx) {
    emit_ddiv_impl(ctx, true);
}
fn emit_ddivu(ctx: &mut EmitCtx) {
    emit_ddiv_impl(ctx, false);
}
fn emit_ddiv_impl(ctx: &mut EmitCtx, signed: bool) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));

    let zero = ctx.builder.ins().iconst(ir::types::I64, 0);
    let is_zero = ctx.builder.ins().icmp(IntCC::Equal, rt_val, zero);

    let divide_block = ctx.builder.create_block();
    let skip_block = ctx.builder.create_block();
    ctx.builder.ins().brif(is_zero, skip_block, &[], divide_block, &[]);

    ctx.builder.switch_to_block(divide_block);
    ctx.builder.seal_block(divide_block);

    if signed {
        // i64::MIN / -1 traps (#DE) on Cranelift's plain sdiv/srem (lowers
        // to the host idiv, which faults on this input) — exec_ddiv treats
        // this case as a no-op (same as divide-by-zero), so skip the
        // division entirely rather than computing any result for it, same
        // rationale as emit_div's MIN/-1 guard but a "do nothing" outcome
        // instead of a computed one.
        let min = ctx.builder.ins().iconst(ir::types::I64, i64::MIN);
        let neg1 = ctx.builder.ins().iconst(ir::types::I64, -1i64);
        let rs_is_min = ctx.builder.ins().icmp(IntCC::Equal, rs_val, min);
        let rt_is_neg1 = ctx.builder.ins().icmp(IntCC::Equal, rt_val, neg1);
        let is_min_over_neg1 = ctx.builder.ins().band(rs_is_min, rt_is_neg1);

        let overflow_block = ctx.builder.create_block();
        let normal_div_block = ctx.builder.create_block();
        ctx.builder.ins().brif(is_min_over_neg1, overflow_block, &[], normal_div_block, &[]);

        // Cold: i64::MIN / -1 is a narrow edge case, rare relative to
        // ordinary division.
        ctx.builder.switch_to_block(overflow_block);
        ctx.builder.set_cold_block(overflow_block);
        ctx.builder.seal_block(overflow_block);
        ctx.builder.ins().jump(skip_block, &[]);

        ctx.builder.switch_to_block(normal_div_block);
        ctx.builder.seal_block(normal_div_block);
        let quotient = ctx.builder.ins().sdiv(rs_val, rt_val);
        let remainder = ctx.builder.ins().srem(rs_val, rt_val);
        emit_write_lo(ctx, quotient);
        emit_write_hi(ctx, remainder);
        ctx.builder.ins().jump(skip_block, &[]);
    } else {
        let quotient = ctx.builder.ins().udiv(rs_val, rt_val);
        let remainder = ctx.builder.ins().urem(rs_val, rt_val);
        emit_write_lo(ctx, quotient);
        emit_write_hi(ctx, remainder);
        ctx.builder.ins().jump(skip_block, &[]);
    }

    ctx.builder.switch_to_block(skip_block);
    ctx.builder.seal_block(skip_block);
}

/// Sign-extend a 16-bit immediate field to I64, matching
/// `DecodedInstr::immu64`'s "imm as i16 as i32 as i64 as u64" convention —
/// used for address-calc immediates (loads/stores/ADDIU/...).
fn field_imm16_sext(builder: &mut FunctionBuilder, raw: u32) -> Value {
    let imm16 = (raw & 0xFFFF) as i16 as i64;
    builder.ins().iconst(ir::types::I64, imm16)
}

/// Zero-extend a 16-bit immediate field to I64, matching
/// `DecodedInstr::immi64`'s "imm as u64" convention (imm is already
/// zero-extended at decode for ANDI/ORI/XORI) — used for bitwise-immediate
/// ops, which treat imm16 as unsigned.
fn field_imm16_zext(builder: &mut FunctionBuilder, raw: u32) -> Value {
    let imm16 = (raw & 0xFFFF) as i64;
    builder.ins().iconst(ir::types::I64, imm16)
}

fn emit_addiu(ctx: &mut EmitCtx) {
    // ADDIU rt, rs, imm: rt = sign_extend32(rs[31:0] + sext(imm16)), no trap.
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let rs_32 = ctx.builder.ins().ireduce(ir::types::I32, rs_val);
    let imm_32 = ctx.builder.ins().ireduce(ir::types::I32, imm);
    let sum_32 = ctx.builder.ins().iadd(rs_32, imm_32);
    let result = ctx.builder.ins().sextend(ir::types::I64, sum_32);
    emit_write_gpr(ctx, field_rt(ctx.raw), result);
}

/// ADDI rt, rs, imm: rt = rs[31:0] + sext(imm16) (signed), trapping on
/// 32-bit signed overflow — mirrors `MipsExecutor::exec_addi`'s
/// `i32::checked_add`.
fn emit_addi(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let rs_32 = ctx.builder.ins().ireduce(ir::types::I32, rs_val);
    let imm_32 = ctx.builder.ins().ireduce(ir::types::I32, imm);
    let (result_32, overflow) = ctx.builder.ins().sadd_overflow(rs_32, imm_32);

    let ok_block = ctx.builder.create_block();
    let trap_block = ctx.builder.create_block();
    ctx.builder.ins().brif(overflow, trap_block, &[], ok_block, &[]);

    // Cold: signed overflow is the rare case for this add/sub-family trap.
    ctx.builder.switch_to_block(trap_block);
    ctx.builder.set_cold_block(trap_block);
    ctx.builder.seal_block(trap_block);
    let status = crate::mips_exec::exec_exception_const(crate::mips_exec::EXC_OV);
    let status_val = ctx.builder.ins().iconst(ir::types::I32, status as i64);
    emit_exception_exit(ctx, status_val);

    ctx.builder.switch_to_block(ok_block);
    ctx.builder.seal_block(ok_block);
    let result_64 = ctx.builder.ins().sextend(ir::types::I64, result_32);
    emit_write_gpr(ctx, field_rt(ctx.raw), result_64);
}

/// DADDI rt, rs, imm: rt = rs + sext(imm16) (signed 64-bit), trapping on
/// 64-bit signed overflow — mirrors `MipsExecutor::exec_daddi`'s
/// `i64::checked_add`. Unlike `ADDI`, no 32-bit truncate/sign-extend step:
/// the whole operation is native 64-bit width.
fn emit_daddi(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let (result, overflow) = ctx.builder.ins().sadd_overflow(rs_val, imm);

    let ok_block = ctx.builder.create_block();
    let trap_block = ctx.builder.create_block();
    ctx.builder.ins().brif(overflow, trap_block, &[], ok_block, &[]);

    // Cold: signed overflow is the rare case for this add/sub-family trap.
    ctx.builder.switch_to_block(trap_block);
    ctx.builder.set_cold_block(trap_block);
    ctx.builder.seal_block(trap_block);
    let status = crate::mips_exec::exec_exception_const(crate::mips_exec::EXC_OV);
    let status_val = ctx.builder.ins().iconst(ir::types::I32, status as i64);
    emit_exception_exit(ctx, status_val);

    ctx.builder.switch_to_block(ok_block);
    ctx.builder.seal_block(ok_block);
    emit_write_gpr(ctx, field_rt(ctx.raw), result);
}

/// DADDIU rt, rs, imm: rt = rs + sext(imm16) (wrapping, no trap) — mirrors
/// `MipsExecutor::exec_daddiu`'s `wrapping_add`. Same relationship to
/// `emit_daddi` that `emit_addiu` has to `emit_addi` (drop the overflow
/// check, wrap instead), at native 64-bit width like `emit_daddi` (no 32-bit
/// truncate/sign-extend step, unlike `emit_addiu`).
fn emit_daddiu(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let result = ctx.builder.ins().iadd(rs_val, imm);
    emit_write_gpr(ctx, field_rt(ctx.raw), result);
}

fn emit_andi(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_zext(ctx.builder, ctx.raw);
    let result = ctx.builder.ins().band(rs_val, imm);
    emit_write_gpr(ctx, field_rt(ctx.raw), result);
}

fn emit_ori(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_zext(ctx.builder, ctx.raw);
    let result = ctx.builder.ins().bor(rs_val, imm);
    emit_write_gpr(ctx, field_rt(ctx.raw), result);
}

fn emit_xori(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_zext(ctx.builder, ctx.raw);
    let result = ctx.builder.ins().bxor(rs_val, imm);
    emit_write_gpr(ctx, field_rt(ctx.raw), result);
}

fn emit_slti(ctx: &mut EmitCtx) {
    // SLTI rt, rs, imm: rt = (rs <s sext(imm16)) ? 1 : 0.
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let lt = ctx.builder.ins().icmp(IntCC::SignedLessThan, rs_val, imm);
    let result = ctx.builder.ins().uextend(ir::types::I64, lt);
    emit_write_gpr(ctx, field_rt(ctx.raw), result);
}

fn emit_sltiu(ctx: &mut EmitCtx) {
    // SLTIU rt, rs, imm: rt = (rs <u sext(imm16)) ? 1 : 0 — imm is still
    // sign-extended at decode (immu64), only the comparison is unsigned.
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let lt = ctx.builder.ins().icmp(IntCC::UnsignedLessThan, rs_val, imm);
    let result = ctx.builder.ins().uextend(ir::types::I64, lt);
    emit_write_gpr(ctx, field_rt(ctx.raw), result);
}

fn emit_lui(ctx: &mut EmitCtx) {
    // LUI rt, imm: rt = sign_extend32(imm16 << 16). Mirrors
    // MipsExecutor::exec_lui / DecodedInstr::set_imm_lui + immu64.
    let imm32 = ((ctx.raw & 0xFFFF) << 16) as i32 as i64;
    let result = ctx.builder.ins().iconst(ir::types::I64, imm32);
    emit_write_gpr(ctx, field_rt(ctx.raw), result);
}

/// Detect the 32-bit-immediate-materialization idiom `lui rX,hi; {ori,addiu}
/// rX,rX,lo` — jitv2's counterpart to the interpreter's `opcodefusion`
/// (`exec_lui_imm32`/`exec_lui_simm32`, mips_exec.rs). `lui_raw`/`next_raw`
/// are the two adjacent words (word, word+1); returns the combined 32-bit
/// value (sign-extended to i64, matching `emit_lui`'s own result type) if
/// `next_raw` is same-register ORI or ADDIU, `None` otherwise. ORI can't
/// carry (pure OR of disjoint hi/lo halves); ADDIU's sign-extending add can
/// carry into bit 16 when lo16's sign bit is set, so this replicates
/// `exec_addiu`'s wrapping-add semantics exactly, not just an OR — same
/// split the interpreter's own decode-time combine makes.
fn fused_lui_imm32(lui_raw: u32, next_raw: u32) -> Option<i64> {
    let rt = field_rt(lui_raw);
    let next_op = (next_raw >> 26) & 0x3F;
    let next_rs = field_rs(next_raw);
    let next_rt = field_rt(next_raw);
    if next_rs != rt || next_rt != rt {
        return None;
    }
    if next_op == crate::mips_isa::OP_ORI {
        let hi = (lui_raw & 0xFFFF) << 16;
        let lo = next_raw & 0xFFFF;
        Some((hi | lo) as i32 as i64)
    } else if next_op == crate::mips_isa::OP_ADDIU {
        let hi = ((lui_raw & 0xFFFF) << 16) as i32;
        let lo = (next_raw & 0xFFFF) as i16 as i32;
        Some(hi.wrapping_add(lo) as i64)
    } else {
        None
    }
}

/// Fast path for `emit_lui`'s LUI+ORI/ADDIU fusion: if the next word
/// (`word + 1`) is a same-register ORI/ADDIU forming the idiom above, write
/// the combined constant directly and report the extra word count to skip
/// (1, i.e. jump to `word + 2` instead of `word + 1`) — `0` if it doesn't
/// apply and the caller must fall back to plain `emit_lui` + ordinary
/// word+1 fallthrough.
///
/// Excluded (returns `0` unconditionally) under three conditions, mirroring
/// `try_emit_fused_nop_slot`'s own gating logic and rationale:
///
/// - `jitv2_lockstep`/`developer`: same reasoning as the NOP fusion — a
///   verification/tracing build needs the LUI and ORI/ADDIU to remain two
///   separately dispatched, separately addressable instructions (lockstep's
///   live per-instruction compare would otherwise materialize `core.pc =
///   word+1` and compare against the *unfused* interpreter's post-LUI-only
///   state, which the fused JIT side never produces — a spurious mismatch,
///   not a real divergence; `developer`'s dt trace/breakpoints need the pair
///   individually addressable exactly like the opcodefusion feature's own
///   Cargo.toml doc comment already accepts for the interpreter).
/// - `word + 1` is off the end of the page (`entries_per_page` boundary):
///   nothing to peek at, and 0xFFC-adjacent words have their own hazards
///   this function doesn't need to reason about — just don't fuse.
/// - **`word` can be arrived at as a foreign delay slot at runtime**
///   (`instrs[word].is_entry_point`, or `instrs[word].is_branch_fallback_successor`):
///   mirrors the interpreter's own `exec_lui_imm32`/`exec_lui_simm32` guard
///   (`if self.core.in_delay_slot { ...don't fuse... }`, mips_exec.rs). Such
///   an arrival means *this* LUI may actually be some other, outside-the-
///   region branch's delay slot — `core.delay_slot_target`, not `word + 1`,
///   is the real next PC, decided only at runtime (the foreign-slot check
///   emitted right after this word's semantics, see `needs_foreign_slot_check`
///   below). Fusing would unconditionally execute word+1's ORI/ADDIU and
///   jump to word+2 before that runtime check ever runs, corrupting rt with
///   a combine against an instruction that has nothing to do with this LUI
///   and silently discarding the pending foreign transfer. A plain interior
///   head can never carry a live `in_delay_slot` into itself this way (every
///   in-region edge that lands here is a compile-time-known plain
///   fallthrough or branch/jump target, never a delay-slot handoff), so
///   fusion stays safe for the ordinary case — only these two arrival kinds
///   need the exclusion.
/// - **`instrs[word + 1].is_branch_target`**: the ORI/ADDIU is independently
///   reachable — something in this region branches/jumps directly to it,
///   skipping the LUI. That arrival needs the real, un-fused ORI/ADDIU to
///   run (`word + 1` keeps its own normal Cranelift block regardless — pass
///   1 allocates blocks for every visited head independent of any fusion
///   decision made here in pass 2 — so that arrival is still correct); what
///   must never happen is the LUI's own block *also* running the combined
///   write and then falling through into that same block a second time,
///   double-applying the ORI/ADDIU. Skipping fusion whenever word+1 is a
///   branch target sidesteps the ambiguity entirely rather than trying to
///   make the fused LUI conditionally skip only for non-branch-target
///   arrivals (impossible — block choice is a single static edge, not a
///   per-arrival runtime branch).
#[cfg_attr(any(not(feature = "jitv2_opcodefusion"), feature = "jitv2_lockstep", feature = "developer"), allow(unused))]
fn try_emit_fused_lui(ctx: &mut EmitCtx, instrs: &[CompiledInstr; ENTRIES_PER_PAGE], word: WordOffset) -> u16 {
    #[cfg(any(not(feature = "jitv2_opcodefusion"), feature = "jitv2_lockstep", feature = "developer"))]
    {
        let _ = (instrs, word);
        0
    }
    #[cfg(all(feature = "jitv2_opcodefusion", not(any(feature = "jitv2_lockstep", feature = "developer"))))]
    {
        if instrs[word as usize].is_entry_point || instrs[word as usize].is_branch_fallback_successor {
            return 0;
        }
        let next_word = word + 1;
        if next_word as usize >= ENTRIES_PER_PAGE {
            return 0;
        }
        let next = &instrs[next_word as usize];
        if !next.visited || next.is_slot_only || next.is_fallback || next.is_branch_target {
            return 0;
        }
        match fused_lui_imm32(ctx.raw, next.raw) {
            Some(combined) => {
                let result = ctx.builder.ins().iconst(ir::types::I64, combined);
                emit_write_gpr(ctx, field_rt(ctx.raw), result);
                1
            }
            None => 0,
        }
    }
}

/// How a loaded value's width is extended to fill the 64-bit GPR — mirrors
/// each `exec_l*` handler's own post-load conversion (`value as iN as i64 as
/// u64` for signed, `value as u64` for zero-extending/full-width loads).
#[derive(Clone, Copy)]
enum LoadExtend { Sign, Zero }

/// Shared body for every `rt, imm(rs)` load: address calc, `read*_fn` call,
/// exception check, extend-and-write-back. `size` selects the hook/access
/// width; `extend` selects how the loaded value fills the 64-bit GPR.
/// Mirrors the common shape of `exec_lb`/`exec_lbu`/`exec_lh`/`exec_lhu`/
/// `exec_lw`/`exec_lwu`/`exec_ld` — they differ only in these two axes.
fn emit_load(ctx: &mut EmitCtx, size: MemSize, extend: LoadExtend) {
    let base = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let vaddr = ctx.builder.ins().iadd(base, imm);

    let loaded = emit_mem_read(ctx, vaddr, size);
    emit_check_mem_exc(ctx); // leaves ctx.builder in the no-fault continuation

    // read*_fn always returns the value zero-extended to u64 on the Rust
    // side (MipsCore's read*_fn field doc comments) — narrow back to the
    // access's true width before re-extending, so a Sign extend actually
    // sign-extends from the right bit rather than from bit 63.
    let result = match (size, extend) {
        (MemSize::B8, _) => loaded, // already the full 64 bits either way
        (_, LoadExtend::Zero) => loaded, // already zero-extended by read*_fn
        (_, LoadExtend::Sign) => {
            let narrow = ctx.builder.ins().ireduce(size.ir_type(), loaded);
            ctx.builder.ins().sextend(ir::types::I64, narrow)
        }
    };
    emit_write_gpr(ctx, field_rt(ctx.raw), result);
}

fn emit_lb(ctx: &mut EmitCtx) {
    emit_load(ctx, MemSize::B1, LoadExtend::Sign);
}
fn emit_lbu(ctx: &mut EmitCtx) {
    emit_load(ctx, MemSize::B1, LoadExtend::Zero);
}
fn emit_lh(ctx: &mut EmitCtx) {
    emit_load(ctx, MemSize::B2, LoadExtend::Sign);
}
fn emit_lhu(ctx: &mut EmitCtx) {
    emit_load(ctx, MemSize::B2, LoadExtend::Zero);
}
fn emit_lw(ctx: &mut EmitCtx) {
    emit_load(ctx, MemSize::B4, LoadExtend::Sign);
}
fn emit_lwu(ctx: &mut EmitCtx) {
    emit_load(ctx, MemSize::B4, LoadExtend::Zero);
}
fn emit_ld(ctx: &mut EmitCtx) {
    emit_load(ctx, MemSize::B8, LoadExtend::Zero);
}

/// LWL rt, imm(rs) / LDL rt, imm(rs): load the "left" (high-address-end, in
/// this big-endian machine's byte numbering) portion of a word/doubleword
/// from an unaligned address, merging into `rt`'s existing low bytes.
/// Mirrors `MipsExecutor::exec_lwl`/`exec_ldl` exactly: align the address
/// down to the access width, read the whole aligned unit, then shift by a
/// **runtime-variable** amount derived from the low alignment bits of the
/// original (unaligned) address — unlike every other load/store emitter in
/// this file, the shift/mask here can't be a compile-time constant, so this
/// uses Cranelift's variable-shift `ishl`/`ushr` (not the `_imm` forms
/// `emit_load`'s siblings all use).
///
/// `width` is 4 (LWL, `MemSize::B4`) or 8 (LDL, `MemSize::B8`); `align_mask`
/// is `width - 1` (0x3 or 0x7) and `total_bits` is `width * 8` (32 or 64) —
/// passed explicitly rather than derived from `size.ir_type()` because the
/// shift-amount arithmetic needs the *bit* width as a runtime constant
/// operand, and LWL's 32-bit merge still writes a sign-extended 64-bit GPR
/// result (`exec_lwl`'s `self.core.write_gpr(rt_reg, result as u64 as i32
/// as i64 as u64)`-equivalent via the final sign-extend), while LDL's is
/// already the full 64-bit value.
fn emit_lwl_ldl(ctx: &mut EmitCtx, size: MemSize) {
    let base = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let vaddr = ctx.builder.ins().iadd(base, imm);

    let align_mask: i64 = size.width_bytes() as i64 - 1;
    let aligned_addr = ctx.builder.ins().band_imm_s(vaddr, !align_mask);
    let byte_offset = ctx.builder.ins().band_imm_s(vaddr, align_mask);

    let loaded = emit_mem_read(ctx, aligned_addr, size);
    emit_check_mem_exc(ctx);

    let ity = size.ir_type();
    let mem_val = if size == MemSize::B8 { loaded } else { ctx.builder.ins().ireduce(ity, loaded) };
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rt_narrow = if size == MemSize::B8 { rt_val } else { ctx.builder.ins().ireduce(ity, rt_val) };

    let byte_offset_narrow = if size == MemSize::B8 { byte_offset } else { ctx.builder.ins().ireduce(ity, byte_offset) };
    let shift = ctx.builder.ins().imul_imm_s(byte_offset_narrow, 8);
    let all_ones = ctx.builder.ins().iconst(ity, -1);
    let mask = ctx.builder.ins().ishl(all_ones, shift);
    let shifted_mem = ctx.builder.ins().ishl(mem_val, shift);
    let not_mask = ctx.builder.ins().bnot(mask);
    let preserved = ctx.builder.ins().band(rt_narrow, not_mask);
    let result_narrow = ctx.builder.ins().bor(shifted_mem, preserved);

    let result = if size == MemSize::B8 {
        result_narrow
    } else {
        ctx.builder.ins().sextend(ir::types::I64, result_narrow)
    };
    emit_write_gpr(ctx, field_rt(ctx.raw), result);
}

/// LWR rt, imm(rs) / LDR rt, imm(rs): load the "right" (low-address-end)
/// portion — the mirror-image shift direction of [`emit_lwl_ldl`], see that
/// function's doc comment for the shared shape. Mirrors
/// `MipsExecutor::exec_lwr`/`exec_ldr` exactly: shift amount is
/// `(width-1-byte_offset)*8`, the opposite sense from LWL/LDL's plain
/// `byte_offset*8`.
fn emit_lwr_ldr(ctx: &mut EmitCtx, size: MemSize) {
    let base = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let vaddr = ctx.builder.ins().iadd(base, imm);

    let align_mask: i64 = size.width_bytes() as i64 - 1;
    let aligned_addr = ctx.builder.ins().band_imm_s(vaddr, !align_mask);
    let byte_offset = ctx.builder.ins().band_imm_s(vaddr, align_mask);

    let loaded = emit_mem_read(ctx, aligned_addr, size);
    emit_check_mem_exc(ctx);

    let ity = size.ir_type();
    let mem_val = if size == MemSize::B8 { loaded } else { ctx.builder.ins().ireduce(ity, loaded) };
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rt_narrow = if size == MemSize::B8 { rt_val } else { ctx.builder.ins().ireduce(ity, rt_val) };

    let byte_offset_narrow = if size == MemSize::B8 { byte_offset } else { ctx.builder.ins().ireduce(ity, byte_offset) };
    let inverted_offset = ctx.builder.ins().iadd_imm_s(byte_offset_narrow, -(align_mask));
    let neg_inverted = ctx.builder.ins().ineg(inverted_offset);
    let shift = ctx.builder.ins().imul_imm_s(neg_inverted, 8);
    let all_ones = ctx.builder.ins().iconst(ity, -1);
    let mask = ctx.builder.ins().ushr(all_ones, shift);
    let shifted_mem = ctx.builder.ins().ushr(mem_val, shift);
    let not_mask = ctx.builder.ins().bnot(mask);
    let preserved = ctx.builder.ins().band(rt_narrow, not_mask);
    let result_narrow = ctx.builder.ins().bor(shifted_mem, preserved);

    let result = if size == MemSize::B8 {
        result_narrow
    } else {
        ctx.builder.ins().sextend(ir::types::I64, result_narrow)
    };
    emit_write_gpr(ctx, field_rt(ctx.raw), result);
}

fn emit_lwl(ctx: &mut EmitCtx) {
    emit_lwl_ldl(ctx, MemSize::B4);
}
fn emit_lwr(ctx: &mut EmitCtx) {
    emit_lwr_ldr(ctx, MemSize::B4);
}
fn emit_ldl(ctx: &mut EmitCtx) {
    emit_lwl_ldl(ctx, MemSize::B8);
}
fn emit_ldr(ctx: &mut EmitCtx) {
    emit_lwr_ldr(ctx, MemSize::B8);
}

/// Shared body for every `rt, imm(rs)` store: address calc, truncate `rt` to
/// `size`, `write*_fn` call, exception check. Mirrors `exec_sb`/`exec_sh`/
/// `exec_sw`/`exec_sd`'s common shape.
fn emit_store(ctx: &mut EmitCtx, size: MemSize) {
    let base = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let vaddr = ctx.builder.ins().iadd(base, imm);

    // Passed to emit_mem_write unnarrowed — see that function's doc comment
    // on why `write*_fn`'s value parameter is always I64/u64 regardless of
    // `size`, not `ireduce`d to the store's real width here.
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));

    emit_mem_write(ctx, vaddr, rt_val, size);
    emit_check_mem_exc(ctx);
}

fn emit_sb(ctx: &mut EmitCtx) {
    emit_store(ctx, MemSize::B1);
}
fn emit_sh(ctx: &mut EmitCtx) {
    emit_store(ctx, MemSize::B2);
}
fn emit_sw(ctx: &mut EmitCtx) {
    emit_store(ctx, MemSize::B4);
}
fn emit_sd(ctx: &mut EmitCtx) {
    emit_store(ctx, MemSize::B8);
}

/// SWL rt, imm(rs): store the "left" (high-address-end) portion of `rt`'s
/// low 32 bits to an unaligned address. Mirrors `MipsExecutor::exec_swl`
/// exactly, including its two-stage shift: first a 32-bit word-space
/// shift/mask (`word_shift = byte_offset*8`, runtime-variable like every
/// other emitter in the unaligned load/store family), then promotion of
/// that 32-bit `(word_val, word_mask)` pair into 64-bit doubleword-aligned
/// space (`dw_shift`, selecting which half of the doubleword this word
/// falls in) before the single masked 64-bit write. `is_left` selects SWL
/// vs SWR's shift direction; the doubleword-promotion math is otherwise
/// identical between them.
fn emit_swl_swr(ctx: &mut EmitCtx, is_left: bool) {
    let base = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let vaddr = ctx.builder.ins().iadd(base, imm);

    let byte_offset64 = ctx.builder.ins().band_imm_s(vaddr, 3);
    let byte_offset = ctx.builder.ins().ireduce(ir::types::I32, byte_offset64);

    let word_shift = if is_left {
        ctx.builder.ins().imul_imm_s(byte_offset, 8)
    } else {
        // (3 - byte_offset) * 8
        let inverted = ctx.builder.ins().iadd_imm_s(byte_offset, -3);
        let neg_inverted = ctx.builder.ins().ineg(inverted);
        ctx.builder.ins().imul_imm_s(neg_inverted, 8)
    };

    let rt_val64 = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rt_val = ctx.builder.ins().ireduce(ir::types::I32, rt_val64);
    let all_ones_32 = ctx.builder.ins().iconst(ir::types::I32, -1i64);
    let (word_mask, word_val) = if is_left {
        (ctx.builder.ins().ushr(all_ones_32, word_shift), ctx.builder.ins().ushr(rt_val, word_shift))
    } else {
        (ctx.builder.ins().ishl(all_ones_32, word_shift), ctx.builder.ins().ishl(rt_val, word_shift))
    };
    let word_mask64 = ctx.builder.ins().uextend(ir::types::I64, word_mask);
    let word_val64 = ctx.builder.ins().uextend(ir::types::I64, word_val);

    // Promote word mask/val into doubleword space at the dword-aligned
    // address — mirrors exec_swl/exec_swr's own aligned8/half/dw_shift math
    // exactly (half = 0 selects the upper dword half, i.e. dw_shift = 32).
    let aligned8 = ctx.builder.ins().band_imm_s(vaddr, !7i64);
    let half64 = ctx.builder.ins().band_imm_s(vaddr, 4);
    let half = ctx.builder.ins().ireduce(ir::types::I32, half64);
    let four = ctx.builder.ins().iconst(ir::types::I32, 4);
    let four_minus_half = ctx.builder.ins().isub(four, half);
    let dw_shift = ctx.builder.ins().ishl_imm_s(four_minus_half, 3);
    let dw_shift64 = ctx.builder.ins().uextend(ir::types::I64, dw_shift);

    let val64 = ctx.builder.ins().ishl(word_val64, dw_shift64);
    let mask64 = ctx.builder.ins().ishl(word_mask64, dw_shift64);

    emit_mem_write_masked(ctx, aligned8, val64, mask64);
    emit_check_mem_exc(ctx);
}

fn emit_swl(ctx: &mut EmitCtx) {
    emit_swl_swr(ctx, true);
}
fn emit_swr(ctx: &mut EmitCtx) {
    emit_swl_swr(ctx, false);
}

/// SDL rt, imm(rs): store the "left" portion of `rt`'s full 64 bits to an
/// unaligned address. Mirrors `MipsExecutor::exec_sdl`/`exec_sdr` exactly —
/// unlike SWL/SWR, no dword-promotion step is needed (the value is already
/// doubleword-width and the aligned address is already doubleword-aligned),
/// just the runtime-variable shift/mask directly in 64-bit space.
fn emit_sdl_sdr(ctx: &mut EmitCtx, is_left: bool) {
    let base = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let vaddr = ctx.builder.ins().iadd(base, imm);

    let byte_offset = ctx.builder.ins().band_imm_s(vaddr, 7);
    let shift = if is_left {
        ctx.builder.ins().imul_imm_s(byte_offset, 8)
    } else {
        // (7 - byte_offset) * 8
        let inverted = ctx.builder.ins().iadd_imm_s(byte_offset, -7);
        let neg_inverted = ctx.builder.ins().ineg(inverted);
        ctx.builder.ins().imul_imm_s(neg_inverted, 8)
    };

    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let all_ones = ctx.builder.ins().iconst(ir::types::I64, -1i64);
    let (mask, val) = if is_left {
        (ctx.builder.ins().ushr(all_ones, shift), ctx.builder.ins().ushr(rt_val, shift))
    } else {
        (ctx.builder.ins().ishl(all_ones, shift), ctx.builder.ins().ishl(rt_val, shift))
    };

    let aligned8 = ctx.builder.ins().band_imm_s(vaddr, !7i64);
    emit_mem_write_masked(ctx, aligned8, val, mask);
    emit_check_mem_exc(ctx);
}

fn emit_sdl(ctx: &mut EmitCtx) {
    emit_sdl_sdr(ctx, true);
}
fn emit_sdr(ctx: &mut EmitCtx) {
    emit_sdl_sdr(ctx, false);
}

/// LWC1 ft, imm(rs): load a word from memory into FPR `ft`'s low 32 bits.
/// Mirrors `MipsExecutor::exec_lwc1`'s `(self.fpr_write_w)(...)` — the CU1
/// check that handler does first is `emit_cp1_cu1_guard`, emitted per this
/// instruction's own dispatch site (pass 2's `lookup_cp1_semantics` arm),
/// same as every other CP1-table entry, triggered by this being registered
/// in `lookup_cp1_semantics` (not `lookup_semantics`) despite not being
/// `OP_COP1`-encoded — see that function's doc comment for why: LWC1/LDC1/
/// SWC1/SDC1 must get both the per-instruction CU1 check and count toward
/// `is_fpu_instruction`'s region-wide FR-mode-guard trigger.
fn emit_lwc1(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let base = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let vaddr = ctx.builder.ins().iadd(base, imm);

    let loaded = emit_mem_read(ctx, vaddr, MemSize::B4);
    emit_check_mem_exc(ctx);

    let value_32 = ctx.builder.ins().ireduce(ir::types::I32, loaded);
    emit_write_fpr_w(ctx, field_rt(ctx.raw), value_32, fr_mode);
}

/// LDC1 ft, imm(rs): load a doubleword from memory into FPR `ft`. Mirrors
/// `MipsExecutor::exec_ldc1`'s `(self.fpr_write_l)(...)`.
fn emit_ldc1(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let base = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let vaddr = ctx.builder.ins().iadd(base, imm);

    let loaded = emit_mem_read(ctx, vaddr, MemSize::B8);
    emit_check_mem_exc(ctx);

    emit_write_fpr_l(ctx, field_rt(ctx.raw), loaded, fr_mode);
}

/// SWC1 ft, imm(rs): store FPR `ft`'s low 32 bits to memory. Mirrors
/// `MipsExecutor::exec_swc1`'s `(self.fpr_read_w)(...)`.
fn emit_swc1(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let base = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let vaddr = ctx.builder.ins().iadd(base, imm);

    let value_32 = emit_read_fpr_w(ctx, field_rt(ctx.raw), fr_mode);
    // emit_mem_write always takes I64 regardless of size — see its own doc
    // comment on why (ABI upper-bits-undefined gotcha, same as emit_store).
    let value_64 = ctx.builder.ins().uextend(ir::types::I64, value_32);
    emit_mem_write(ctx, vaddr, value_64, MemSize::B4);
    emit_check_mem_exc(ctx);
}

/// SDC1 ft, imm(rs): store FPR `ft`'s full 64 bits to memory. Mirrors
/// `MipsExecutor::exec_sdc1`'s `(self.fpr_read_l)(...)`.
fn emit_sdc1(ctx: &mut EmitCtx, fr_mode: FrMode) {
    let base = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let vaddr = ctx.builder.ins().iadd(base, imm);

    let value_64 = emit_read_fpr_l(ctx, field_rt(ctx.raw), fr_mode);
    emit_mem_write(ctx, vaddr, value_64, MemSize::B8);
    emit_check_mem_exc(ctx);
}

/// MOVZ rd, rs, rt: rd = rs if rt == 0 (no-op otherwise). Mirrors
/// `MipsExecutor::exec_movz` exactly, including that `rd` isn't touched at
/// all (not even re-written with its own value) when the condition is
/// false.
fn emit_movz(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rd = field_rd(ctx.raw);

    let taken = ctx.builder.ins().icmp_imm_s(ir::condcodes::IntCC::Equal, rt_val, 0);
    let write_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();
    ctx.builder.ins().brif(taken, write_block, &[], merge_block, &[]);

    ctx.builder.switch_to_block(write_block);
    ctx.builder.seal_block(write_block);
    emit_write_gpr(ctx, rd, rs_val);
    ctx.builder.ins().jump(merge_block, &[]);

    ctx.builder.switch_to_block(merge_block);
    ctx.builder.seal_block(merge_block);
}

/// MOVN rd, rs, rt: rd = rs if rt != 0 (no-op otherwise). Mirrors
/// `MipsExecutor::exec_movn`; same shape as `emit_movz` with the condition
/// inverted.
fn emit_movn(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let rd = field_rd(ctx.raw);

    let taken = ctx.builder.ins().icmp_imm_s(ir::condcodes::IntCC::NotEqual, rt_val, 0);
    let write_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();
    ctx.builder.ins().brif(taken, write_block, &[], merge_block, &[]);

    ctx.builder.switch_to_block(write_block);
    ctx.builder.seal_block(write_block);
    emit_write_gpr(ctx, rd, rs_val);
    ctx.builder.ins().jump(merge_block, &[]);

    ctx.builder.switch_to_block(merge_block);
    ctx.builder.seal_block(merge_block);
}

/// MOVCI rd, rs, cc, tf: rd = rs if FPU condition code `cc` == `tf`
/// (no-op otherwise). Mirrors `MipsExecutor::exec_movci`/`MipsCore::get_fpu_cc`
/// exactly: cc0 lives at FCSR bit 23, cc1..cc7 at FCSR bits 24..30. `cc`/`tf`
/// are both compile-time constants (part of the fixed instruction encoding,
/// same as CFC1/CTC1's `fs`), so the bit position is resolved at compile
/// time — only the FCSR load and the taken/not-taken branch are runtime.
/// Despite reading FPU state, this is `OP_SPECIAL`-funct-encoded (not
/// `OP_COP1`), matching `exec_movci`'s dispatch through the integer funct
/// table — registered in `lookup_semantics`, not `lookup_cp1_semantics`.
fn emit_movci(ctx: &mut EmitCtx) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rd = field_rd(ctx.raw);
    let cc = (ctx.raw >> 18) & 0x7;
    let tf = ((ctx.raw >> 16) & 0x1) != 0;
    let bit = if cc == 0 { 23 } else { 24 + cc };

    let mem = MemFlagsData::trusted();
    let fcsr = ctx.builder.ins().load(ir::types::I32, mem, ctx.core_ptr, ir::immediates::Offset32::new(core_offset_of_fpu_fcsr()));
    let cc_bit = ctx.builder.ins().ushr_imm_s(fcsr, bit as i64);
    let cc_value = ctx.builder.ins().band_imm_s(cc_bit, 1);
    let want = if tf { 1 } else { 0 };
    let taken = ctx.builder.ins().icmp_imm_s(ir::condcodes::IntCC::Equal, cc_value, want);

    let write_block = ctx.builder.create_block();
    let merge_block = ctx.builder.create_block();
    ctx.builder.ins().brif(taken, write_block, &[], merge_block, &[]);

    ctx.builder.switch_to_block(write_block);
    ctx.builder.seal_block(write_block);
    emit_write_gpr(ctx, rd, rs_val);
    ctx.builder.ins().jump(merge_block, &[]);

    ctx.builder.switch_to_block(merge_block);
    ctx.builder.seal_block(merge_block);
}

/// Shared body for the six register-register trap instructions (TGE/TGEU/
/// TLT/TLTU/TEQ/TNE): compare rs against rt per `cc` and `signed`, and raise
/// EXC_TR (via the same `emit_exception_exit` shared-infrastructure path
/// `emit_daddi`'s overflow trap already uses — deliver_exception's Cause/
/// EPC/Status.EXL/vector-jump side effects apply uniformly to every
/// architectural exception, trap included, not something specific to this
/// instruction class) when the condition holds. Mirrors `exec_tge`/
/// `exec_tgeu`/`exec_tlt`/`exec_tltu`/`exec_teq`/`exec_tne` exactly,
/// including that no-trap falls straight through with no other effect
/// (these never write a register).
fn emit_trap_rr(ctx: &mut EmitCtx, cc: IntCC) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let rt_val = emit_read_gpr(ctx, field_rt(ctx.raw));
    let taken = ctx.builder.ins().icmp(cc, rs_val, rt_val);

    let trap_block = ctx.builder.create_block();
    let ok_block = ctx.builder.create_block();
    ctx.builder.ins().brif(taken, trap_block, &[], ok_block, &[]);

    // Cold: the trap condition holding is the rare case — these are
    // defensive checks, not control flow real code takes routinely.
    ctx.builder.switch_to_block(trap_block);
    ctx.builder.set_cold_block(trap_block);
    ctx.builder.seal_block(trap_block);
    let status = crate::mips_exec::exec_exception_const(crate::mips_exec::EXC_TR);
    let status_val = ctx.builder.ins().iconst(ir::types::I32, status as i64);
    emit_exception_exit(ctx, status_val);

    ctx.builder.switch_to_block(ok_block);
    ctx.builder.seal_block(ok_block);
}

fn emit_tge(ctx: &mut EmitCtx) { emit_trap_rr(ctx, IntCC::SignedGreaterThanOrEqual); }
fn emit_tgeu(ctx: &mut EmitCtx) { emit_trap_rr(ctx, IntCC::UnsignedGreaterThanOrEqual); }
fn emit_tlt(ctx: &mut EmitCtx) { emit_trap_rr(ctx, IntCC::SignedLessThan); }
fn emit_tltu(ctx: &mut EmitCtx) { emit_trap_rr(ctx, IntCC::UnsignedLessThan); }
fn emit_teq(ctx: &mut EmitCtx) { emit_trap_rr(ctx, IntCC::Equal); }
fn emit_tne(ctx: &mut EmitCtx) { emit_trap_rr(ctx, IntCC::NotEqual); }

/// Shared body for the six REGIMM trap-immediate instructions (TGEI/TGEIU/
/// TLTI/TLTIU/TEQI/TNEI): compare rs against sign-extended imm16 per `cc`,
/// raise EXC_TR when it holds. Mirrors `exec_tgei`/`exec_tgeiu`/`exec_tlti`/
/// `exec_tltiu`/`exec_teqi`/`exec_tnei` — note the immediate is always
/// sign-extended even for the "unsigned" comparisons (`DecodedInstr::immu64`
/// and `imms64` are identical; only the *comparison* is unsigned for TGEIU/
/// TLTIU, not the immediate's sign-extension), matching `field_imm16_sext`
/// here.
fn emit_trap_ri(ctx: &mut EmitCtx, cc: IntCC) {
    let rs_val = emit_read_gpr(ctx, field_rs(ctx.raw));
    let imm = field_imm16_sext(ctx.builder, ctx.raw);
    let taken = ctx.builder.ins().icmp(cc, rs_val, imm);

    let trap_block = ctx.builder.create_block();
    let ok_block = ctx.builder.create_block();
    ctx.builder.ins().brif(taken, trap_block, &[], ok_block, &[]);

    // Cold: the trap condition holding is the rare case — see emit_trap_rr's
    // sibling comment.
    ctx.builder.switch_to_block(trap_block);
    ctx.builder.set_cold_block(trap_block);
    ctx.builder.seal_block(trap_block);
    let status = crate::mips_exec::exec_exception_const(crate::mips_exec::EXC_TR);
    let status_val = ctx.builder.ins().iconst(ir::types::I32, status as i64);
    emit_exception_exit(ctx, status_val);

    ctx.builder.switch_to_block(ok_block);
    ctx.builder.seal_block(ok_block);
}

fn emit_tgei(ctx: &mut EmitCtx) { emit_trap_ri(ctx, IntCC::SignedGreaterThanOrEqual); }
fn emit_tgeiu(ctx: &mut EmitCtx) { emit_trap_ri(ctx, IntCC::UnsignedGreaterThanOrEqual); }
fn emit_tlti(ctx: &mut EmitCtx) { emit_trap_ri(ctx, IntCC::SignedLessThan); }
fn emit_tltiu(ctx: &mut EmitCtx) { emit_trap_ri(ctx, IntCC::UnsignedLessThan); }
fn emit_teqi(ctx: &mut EmitCtx) { emit_trap_ri(ctx, IntCC::Equal); }
fn emit_tnei(ctx: &mut EmitCtx) { emit_trap_ri(ctx, IntCC::NotEqual); }

/// SYNC/PREF/PREFX: no-ops on this emulator (no weak memory model to fence,
/// no prefetch cache to hint) — mirrors whatever the interpreter does for
/// these (nothing architecturally observable). Registered purely so
/// `has_emitter` stops excluding them and they stop being region boundaries.
fn emit_nop(_ctx: &mut EmitCtx) {}

/// Look up the semantics emitter for a decoded instruction word, or `None`
/// if this instruction isn't wired up yet. `None` here is a codegen gap, not
/// an architectural exclusion — those are `analyzer::Classify::Excluded` and
/// never reach codegen at all (§4.4). Callers (`compile_region`) should
/// treat `None` as "can't compile this region" (deny/reject), not panic —
/// the instruction set is being filled in incrementally.
fn lookup_semantics(raw: u32) -> Option<SemanticsEmitter> {
    use crate::mips_isa::*;
    let op = (raw >> 26) & 0x3F;
    let funct = raw & 0x3F;
    match op {
        OP_SPECIAL => match funct {
            FUNCT_ADD => Some(emit_add),
            FUNCT_ADDU => Some(emit_addu),
            FUNCT_SUB => Some(emit_sub),
            FUNCT_SUBU => Some(emit_subu),
            FUNCT_AND => Some(emit_and),
            FUNCT_OR => Some(emit_or),
            FUNCT_XOR => Some(emit_xor),
            FUNCT_NOR => Some(emit_nor),
            FUNCT_SLT => Some(emit_slt),
            FUNCT_SLTU => Some(emit_sltu),
            FUNCT_SLL => Some(emit_sll),
            FUNCT_SRL => Some(emit_srl),
            FUNCT_SRA => Some(emit_sra),
            FUNCT_SLLV => Some(emit_sllv),
            FUNCT_SRLV => Some(emit_srlv),
            FUNCT_SRAV => Some(emit_srav),
            FUNCT_MFHI => Some(emit_mfhi),
            FUNCT_MTHI => Some(emit_mthi),
            FUNCT_MFLO => Some(emit_mflo),
            FUNCT_MTLO => Some(emit_mtlo),
            FUNCT_MULT => Some(emit_mult),
            FUNCT_MULTU => Some(emit_multu),
            FUNCT_DIV => Some(emit_div),
            FUNCT_DIVU => Some(emit_divu),
            FUNCT_DMULT => Some(emit_dmult),
            FUNCT_DMULTU => Some(emit_dmultu),
            FUNCT_DDIV => Some(emit_ddiv),
            FUNCT_DDIVU => Some(emit_ddivu),
            FUNCT_DADD => Some(emit_dadd),
            FUNCT_DADDU => Some(emit_daddu),
            FUNCT_DSUB => Some(emit_dsub),
            FUNCT_DSUBU => Some(emit_dsubu),
            FUNCT_DSLL => Some(emit_dsll),
            FUNCT_DSRL => Some(emit_dsrl),
            FUNCT_DSRA => Some(emit_dsra),
            FUNCT_DSLL32 => Some(emit_dsll32),
            FUNCT_DSRL32 => Some(emit_dsrl32),
            FUNCT_DSRA32 => Some(emit_dsra32),
            FUNCT_DSLLV => Some(emit_dsllv),
            FUNCT_DSRLV => Some(emit_dsrlv),
            FUNCT_DSRAV => Some(emit_dsrav),
            // MOVZ/MOVN/MOVCI are MIPS IV; without the feature they must not
            // be compiled here so the analyzer/interpreter fallback can
            // raise Reserved Instruction (mirrors mips_exec.rs's decode gate).
            #[cfg(feature = "mips4")]
            FUNCT_MOVZ => Some(emit_movz),
            #[cfg(feature = "mips4")]
            FUNCT_MOVN => Some(emit_movn),
            #[cfg(feature = "mips4")]
            FUNCT_MOVCI => Some(emit_movci),
            #[cfg(not(feature = "mips4"))]
            FUNCT_MOVZ | FUNCT_MOVN | FUNCT_MOVCI => None,
            FUNCT_TGE => Some(emit_tge),
            FUNCT_TGEU => Some(emit_tgeu),
            FUNCT_TLT => Some(emit_tlt),
            FUNCT_TLTU => Some(emit_tltu),
            FUNCT_TEQ => Some(emit_teq),
            FUNCT_TNE => Some(emit_tne),
            FUNCT_SYNC => Some(emit_nop),
            _ => None,
        },
        OP_REGIMM => match (raw >> 16) & 0x1F {
            RT_TGEI => Some(emit_tgei),
            RT_TGEIU => Some(emit_tgeiu),
            RT_TLTI => Some(emit_tlti),
            RT_TLTIU => Some(emit_tltiu),
            RT_TEQI => Some(emit_teqi),
            RT_TNEI => Some(emit_tnei),
            _ => None,
        },
        // PREF is MIPS IV; without the feature it must not be compiled here
        // so the interpreter fallback can raise Reserved Instruction.
        #[cfg(feature = "mips4")]
        OP_PREF => Some(emit_nop),
        #[cfg(not(feature = "mips4"))]
        OP_PREF => None,
        OP_ADDI => Some(emit_addi),
        OP_ADDIU => Some(emit_addiu),
        OP_DADDI => Some(emit_daddi),
        OP_DADDIU => Some(emit_daddiu),
        OP_SLTI => Some(emit_slti),
        OP_SLTIU => Some(emit_sltiu),
        OP_ANDI => Some(emit_andi),
        OP_ORI => Some(emit_ori),
        OP_XORI => Some(emit_xori),
        OP_LUI => Some(emit_lui),
        OP_LB => Some(emit_lb),
        OP_LBU => Some(emit_lbu),
        OP_LH => Some(emit_lh),
        OP_LHU => Some(emit_lhu),
        OP_LW => Some(emit_lw),
        OP_LWU => Some(emit_lwu),
        OP_LD => Some(emit_ld),
        OP_LWL => Some(emit_lwl),
        OP_LWR => Some(emit_lwr),
        OP_LDL => Some(emit_ldl),
        OP_LDR => Some(emit_ldr),
        OP_SB => Some(emit_sb),
        OP_SH => Some(emit_sh),
        OP_SW => Some(emit_sw),
        OP_SD => Some(emit_sd),
        OP_SWL => Some(emit_swl),
        OP_SWR => Some(emit_swr),
        OP_SDL => Some(emit_sdl),
        OP_SDR => Some(emit_sdr),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jitv2::analyzer::Analyzer;
    use crate::mips_isa::*;

    fn r_type(op: u32, rs: u32, rt: u32, rd: u32, sa: u32, funct: u32) -> u32 {
        (op << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct
    }

    #[test]
    fn skeleton_allocates_one_block_per_visited_instruction() {
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = 0; // nop
        page[1] = 0; // nop
        page[2] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // jr ra
        page[3] = 0; // delay slot

        let mut analyzer = Analyzer::new();
        let (instrs, non_empty) = analyzer.walk(&page, 0, 0);
        assert!(non_empty);
        let visited_count = instrs_linear(instrs).count();
        assert_eq!(visited_count, 4); // words 0,1,2,3

        let mut instrs_owned = *instrs; // Codegen needs &mut; copy out of the analyzer's borrow

        let mut codegen = Codegen::new();
        let skeleton = codegen.build_block_skeleton(&mut instrs_owned, 0);

        assert_eq!(skeleton.instr_blocks.len(), visited_count);
        // Every visited word offset has a corresponding block.
        for word in [0u16, 1, 2, 3] {
            assert!(skeleton.instr_blocks.iter().any(|&(w, _)| w == word), "missing block for word {}", word);
        }
        // block_id was written back into the buffer for every visited instruction.
        for instr in instrs_linear(&instrs_owned) {
            assert!(instr.block_id.is_some(), "word {} missing block_id after skeleton pass", instr.word);
        }
    }

    #[test]
    fn skeleton_entry_block_targets_the_entry_word_block() {
        let mut page = [0u32; ENTRIES_PER_PAGE];
        // Straight-line region starting at word 2 (simulate a non-zero entry).
        page[2] = 0;
        page[3] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR);
        page[4] = 0;

        let mut analyzer = Analyzer::new();
        let (instrs, non_empty) = analyzer.walk(&page, 2, 0);
        assert!(non_empty);
        let mut instrs_owned = *instrs;

        let mut codegen = Codegen::new();
        let skeleton = codegen.build_block_skeleton(&mut instrs_owned, 2);

        let entry_word_block_id = instrs_owned[2].block_id.expect("entry word must have a block");
        // The skeleton's instr_blocks entry for word 2 must be the same block
        // the entry_block jumps to (verified indirectly: both come from the
        // same allocation, so their raw ids must match).
        let (_, block_for_word_2) = skeleton.instr_blocks.iter().find(|&&(w, _)| w == 2).unwrap();
        assert_eq!(block_for_word_2.as_u32(), entry_word_block_id);
    }

    #[test]
    #[should_panic(expected = "entry_word must be a visited offset")]
    fn skeleton_panics_if_entry_word_was_never_visited() {
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = 0; // nop
        page[1] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // region ends here
        page[2] = 0; // delay slot
        // word 999 is well past the region and never reached.
        let mut analyzer = Analyzer::new();
        let (instrs, _) = analyzer.walk(&page, 0, 0);
        let mut instrs_owned = *instrs;
        let mut codegen = Codegen::new();
        codegen.build_block_skeleton(&mut instrs_owned, 999);
    }

    /// Compile a standalone function of the form:
    ///   entry_block: jump to preamble_block
    ///   preamble_block: emit(word_offset) then `return EXEC_COMPLETE`
    ///   exit_block: the shared bail target (built the same way
    ///               build_block_skeleton builds it)
    /// and return it as a callable `JitFn`. Exercises a preamble emitter
    /// end-to-end (real generated code including the shared exit block, not
    /// just IR construction) without needing the rest of the per-instruction
    /// emission pipeline, which doesn't exist yet.
    fn compile_preamble_only(
        name: &str,
        emit: impl FnOnce(&mut EmitCtx, Block, WordOffset),
        word_offset: WordOffset,
    ) -> crate::jitv2::JitFn {
        use crate::mips_exec::EXEC_COMPLETE;

        let mut codegen = Codegen::new();
        codegen.ctx.func.signature = codegen.jit_fn_signature();
        codegen.ctx.func.name = ir::UserFuncName::user(0, 0);

        let func_id = codegen.module
            .declare_function(name, cranelift_module::Linkage::Local, &codegen.ctx.func.signature)
            .unwrap();

        {
            let mut builder = FunctionBuilder::new(&mut codegen.ctx.func, &mut codegen.builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            let core_ptr = builder.block_params(entry_block)[0];

            // Shared exit block, built the same way build_block_skeleton does.
            let exit_block = builder.create_block();
            let ptr_ty = builder.func.signature.params[0].value_type;
            let exit_core_ptr = builder.append_block_param(exit_block, ptr_ty);
            let exit_word_offset = builder.append_block_param(exit_block, ir::types::I64);

            // Shared exception-exit machinery: constructed for EmitCtx's sake
            // (the preamble emitter this harness exercises,
            // emit_pending_interrupt_preamble, only ever calls emit_bail,
            // never emit_exception_exit) but never actually jumped to here —
            // still needs real, sealed blocks with valid bodies or the
            // verifier rejects the dangling references at finalize() time.
            let exception_call_block = builder.create_block();
            let call_core_ptr = builder.append_block_param(exception_call_block, ptr_ty);
            let call_status_param = builder.append_block_param(exception_call_block, ir::types::I32);

            let exception_other_word_block = builder.create_block();
            let other_core_ptr = builder.append_block_param(exception_other_word_block, ptr_ty);
            let other_word_param = builder.append_block_param(exception_other_word_block, ir::types::I64);
            let other_bd_param = builder.append_block_param(exception_other_word_block, ir::types::I8);
            let other_status_param = builder.append_block_param(exception_other_word_block, ir::types::I32);

            let exception_entry_word_block = builder.create_block();
            let entry_exc_core_ptr = builder.append_block_param(exception_entry_word_block, ptr_ty);
            let entry_exc_status_param = builder.append_block_param(exception_entry_word_block, ir::types::I32);

            {
                // Test harness for preamble emitters only (see this
                // function's doc comment) — never touches cycles bookkeeping.
                let mut unused_cycles_pending = 0u32;
                let mut ctx = EmitCtx { builder: &mut builder, module: &mut codegen.module, core_ptr, raw: 0, word: word_offset, bd: false, trust_live_pc_bd_on_exc: true, exit_block, exception_call_block, exception_entry_word_block, exception_other_word_block, cycles_pending: &mut unused_cycles_pending };
                emit(&mut ctx, exit_block, word_offset);
            }
            // Not-fired/not-pending path continues here (the preamble leaves
            // the builder positioned in its continuation block, already sealed).
            let status = builder.ins().iconst(ir::types::I32, EXEC_COMPLETE as i64);
            builder.ins().return_(&[status]);
            builder.seal_block(entry_block);

            builder.switch_to_block(exit_block);
            emit_exit_block_body(&mut builder, &mut codegen.module, exit_core_ptr, exit_word_offset);
            builder.seal_block(exit_block); // only predecessor in this harness is the preamble's bail site

            builder.switch_to_block(exception_call_block);
            emit_exception_call_block_body(&mut codegen.module, &mut builder, call_core_ptr, call_status_param);

            builder.switch_to_block(exception_other_word_block);
            emit_exception_other_word_block_body(&mut builder, other_core_ptr, other_word_param, other_bd_param, other_status_param, exception_call_block);

            builder.switch_to_block(exception_entry_word_block);
            emit_exception_entry_word_block_body(&mut builder, entry_exc_core_ptr, entry_exc_status_param, exception_call_block);

            // Sealed together, after every predecessor edge into any of the
            // three (the two outer stages' jumps into exception_call_block,
            // above) has been emitted — never actually jumped to *into* from
            // outside this trio in this harness, but exception_call_block's
            // own in-trio predecessors must still be established first.
            builder.seal_block(exception_call_block);
            builder.seal_block(exception_other_word_block);
            builder.seal_block(exception_entry_word_block);

            builder.finalize(codegen.module.target_config());
        }

        codegen.module.define_function(func_id, &mut codegen.ctx).unwrap();
        codegen.module.clear_context(&mut codegen.ctx);
        // This harness bypasses compile_region_uncommitted entirely (calls
        // module.define_function directly), so nothing has reserved this
        // range's seal-queue slot yet — push_placeholder/patch_pending_publish
        // must be driven by hand, same as Codegen::finalize_batch does
        // internally, or the memory stays RW forever and the returned
        // function pointer segfaults on call.
        let range = codegen.seal_handle.take_last_allocation()
            .expect("define_function must have allocated real memory");
        codegen.seal_handle.push_placeholder(range.0, range.1, std::ptr::null_mut());
        codegen.module.finalize_definitions().unwrap();
        let code_ptr = codegen.module.get_finalized_function(func_id);
        let publish = crate::jitv2::paged_memory::PublishInfo {
            jit_fn: Some(unsafe { std::mem::transmute::<*const u8, crate::jitv2::JitFn>(code_ptr) }),
            ..crate::jitv2::paged_memory::PublishInfo::blank()
        };
        codegen.seal_handle.patch_pending_publish(range.0, range.1, publish, true);
        // Leak the module so the JIT-compiled code stays valid for the
        // caller — fine for a test, mirrors what a long-lived Codegen would
        // do (code lives as long as the module, which normally lives for
        // the compile thread's whole run).
        std::mem::forget(codegen.module);
        unsafe { std::mem::transmute::<*const u8, crate::jitv2::JitFn>(code_ptr) }
    }

    #[test]
    fn pending_preamble_continues_when_nothing_pending() {
        let jit_fn = compile_preamble_only("test_pending_none", emit_pending_interrupt_preamble, 7);
        let mut core = MipsCore::new();
        core.pc = 0xFFFFFFFF_80002000;
        core.hot.interrupts.store(0, std::sync::atomic::Ordering::Relaxed);
        let orig_pc = core.pc;

        let status = unsafe { jit_fn(&mut core as *mut MipsCore) };

        assert_eq!(status, crate::mips_exec::EXEC_COMPLETE);
        assert_eq!(core.pc, orig_pc, "pc must be untouched when nothing is pending");
    }

    #[test]
    fn pending_preamble_exits_with_retry_pc_when_pending() {
        let word_offset: WordOffset = 7;
        let jit_fn = compile_preamble_only("test_pending_some", emit_pending_interrupt_preamble, word_offset);
        let mut core = MipsCore::new();
        core.pc = 0xFFFFFFFF_80002000;
        core.hot.interrupts.store(1 << 10, std::sync::atomic::Ordering::Relaxed); // some arbitrary nonzero bit

        let status = unsafe { jit_fn(&mut core as *mut MipsCore) };

        assert_eq!(status, EXEC_COMPLETE);
        let orig_vbase = 0xFFFFFFFF_80002000u64 & !(PAGE_SIZE as u64 - 1);
        assert_eq!(core.pc, orig_vbase | ((word_offset as u64) * 4),
            "pc must be set to this instruction's own address for the interpreter to retry");
    }

    /// Compile `emit_round_to_int_mode` as a standalone `fn(f64, i8) -> i64`
    /// (returning the result's raw bits — `extern "C" fn(..) -> f64` return
    /// values are fine too, but bits make bit-exact comparison in the test
    /// unambiguous, e.g. for -0.0) bypassing the `*mut MipsCore` calling
    /// convention entirely, so it can be unit-tested directly against
    /// `mips_exec.rs`'s `round_f64_to_int_mode` without needing a full
    /// CVT.W.D instruction dispatch.
    fn compile_round_to_int_mode_f64() -> extern "C" fn(f64, i8) -> i64 {
        let mut codegen = Codegen::new();
        let mut sig = codegen.module.make_signature();
        sig.params.push(AbiParam::new(ir::types::F64));
        sig.params.push(AbiParam::new(ir::types::I8));
        sig.returns.push(AbiParam::new(ir::types::I64));
        codegen.ctx.func.signature = sig;
        codegen.ctx.func.name = ir::UserFuncName::user(0, 0);

        let func_id = codegen.module
            .declare_function("test_round_to_int_mode_f64", cranelift_module::Linkage::Local, &codegen.ctx.func.signature)
            .unwrap();

        {
            let mut builder = FunctionBuilder::new(&mut codegen.ctx.func, &mut codegen.builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            let x = builder.block_params(entry_block)[0];
            let rm = builder.block_params(entry_block)[1];
            let result = emit_round_to_int_mode(&mut builder, x, rm);
            let bits = builder.ins().bitcast(ir::types::I64, MemFlagsData::new(), result);
            builder.ins().return_(&[bits]);
            builder.seal_block(entry_block);
            builder.finalize(codegen.module.target_config());
        }

        codegen.module.define_function(func_id, &mut codegen.ctx).unwrap();
        codegen.module.clear_context(&mut codegen.ctx);
        // See compile_preamble_only's own comment on the equivalent lines —
        // this harness bypasses compile_region_uncommitted too.
        let range = codegen.seal_handle.take_last_allocation()
            .expect("define_function must have allocated real memory");
        codegen.seal_handle.push_placeholder(range.0, range.1, std::ptr::null_mut());
        codegen.module.finalize_definitions().unwrap();
        let code_ptr = codegen.module.get_finalized_function(func_id);
        let publish = crate::jitv2::paged_memory::PublishInfo {
            jit_fn: Some(unsafe { std::mem::transmute::<*const u8, crate::jitv2::JitFn>(code_ptr) }),
            ..crate::jitv2::paged_memory::PublishInfo::blank()
        };
        codegen.seal_handle.patch_pending_publish(range.0, range.1, publish, true);
        std::mem::forget(codegen.module);
        unsafe { std::mem::transmute::<*const u8, extern "C" fn(f64, i8) -> i64>(code_ptr) }
    }

    /// Live-boot regression: `CVT.W.D $f10, $f10` on `-0.9757914543151855`
    /// under FCSR.RM=0 diverged jit=0 vs interp=-1 (correct: -1, since |x| is
    /// closer to 1 than 0). Root cause: `emit_round_to_int_mode` used
    /// `bnot` (bitwise NOT) on several Cranelift `icmp`-produced boolean
    /// `Value`s expecting logical negation — Cranelift booleans are encoded
    /// as plain `0`/`1` (not all-ones/all-zeros), so `bnot(1) = 0xFE`, which
    /// is *truthy* in a later `select`, silently flipping the wrong branch
    /// for exactly the `exp < 0` (magnitude < 1.0) regime this test's value
    /// falls into. Fixed by replacing every boolean negation with
    /// `icmp_imm(Equal, v, 0)` instead of `bnot(v)` (kept `bnot` only for
    /// its two genuine bitwise-NOT uses: `not_frac_mask`, a real bitmask,
    /// and `emit_nor`, MIPS's actual NOR instruction). Isolates the
    /// primitive directly (bypassing full CVT.W.D dispatch) so a future
    /// regression here doesn't need re-deriving this whole investigation.
    #[test]
    fn round_to_int_mode_f64_matches_interpreter_primitive() {
        let f = compile_round_to_int_mode_f64();
        let cases: [(f64, i8); 16] = [
            (-0.9757914543151855, 0),
            (0.9757914543151855, 0),
            (-0.6, 0), (0.6, 0),
            (-0.6, 1), (0.6, 1),
            (-0.6, 2), (0.6, 2),
            (-0.6, 3), (0.6, 3),
            (-0.5, 0), (0.5, 0),
            (-0.3, 0), (0.3, 0),
            (-0.3, 2), (0.3, 3),
        ];
        for (x, rm) in cases {
            let jit_result = f64::from_bits(f(x, rm) as u64);
            let interp_result = crate::mips_exec::round_f64_to_int_mode(x, rm as u8);
            assert_eq!(jit_result, interp_result, "round_to_int_mode({x}, rm={rm}): jit={jit_result} interp={interp_result}");
        }
    }

    /// Compiles a single-instruction `ADDIU r1, r0, 1` region at `entry_word`
    /// via `compile_region_uncommitted` — the smallest region that reliably
    /// gets a real emitter and a real `FuncId`, for exercising the
    /// non-forced finalize/idle-sweep path below without needing a real
    /// `handle_request_deferred`/bus round trip.
    fn compile_one_addiu(codegen: &mut Codegen, entry_word: WordOffset) -> cranelift_module::FuncId {
        // Every word besides the entry itself is the JIT region-boundary
        // sentinel, not the implicit `SLL r0,r0,0` NOP a zeroed word would
        // decode as — a real one-instruction region, not a walk that keeps
        // going across the whole page. Using an all-zero page here was
        // tried first and confirmed the wrong shape: zero is a genuine,
        // walkable NOP with a real emitter, so an unbounded `analyzer.walk`
        // would keep extending through the entire 1024-word page, compiling
        // a ~1000-instruction function per call instead of the one
        // instruction each test actually intends — still correct, but
        // ~300x slower per call than intended (confirmed live via direct
        // instrumentation during the compile-pool work).
        let mut page = [JIT_REGION_BOUNDARY_SENTINEL; ENTRIES_PER_PAGE];
        page[entry_word as usize] = (OP_ADDIU << 26) | (1 << 16) | 1;
        let mut analyzer = Analyzer::new();
        let (instrs, non_empty) = analyzer.walk(&page, entry_word, 0);
        assert!(non_empty);
        let mut instrs_owned = *instrs;
        instrs_owned[entry_word as usize].is_entry_point = true;
        codegen.compile_region_uncommitted(&mut instrs_owned, true, true, false, std::ptr::null_mut())
            .expect("a plain ADDIU must have a real emitter")
    }

    fn dummy_publish() -> crate::jitv2::paged_memory::PublishInfo {
        crate::jitv2::paged_memory::PublishInfo::blank()
    }

    #[test]
    fn finalize_batch_nonforced_does_not_seal_a_still_open_page() {
        // A single small compile's own page is never "vacated" (the bump
        // cursor sits nowhere near the next page boundary) — a non-forced
        // finalize must leave it queued, unsealed, exactly like
        // paged_memory's own non_forced_finalize_lets_a_later_allocation_pack_into_the_same_page
        // test already proves at the arena level. This is the whole point
        // of non-forced: it must NOT behave like the forced path.
        let mut codegen = Codegen::new();
        let id = compile_one_addiu(&mut codegen, 0);
        let sealed = codegen.finalize_batch_nonforced(id, dummy_publish());
        assert!(sealed.is_empty(), "a lone small batch's page is still open — non-forced must not seal it");
    }

    #[test]
    fn seal_queue_snapshot_stamps_the_real_page_and_thread_of_a_still_unpatched_entry() {
        // j2 seal-queue's whole reason to exist: naming exactly which
        // compile is stuck (page + thread), not just "some placeholder,
        // somewhere." A page that's never been finalized (compile_region_uncommitted
        // ran, patch_pending_publish never did) must still show up here with
        // its real identity — that's the unpatched-placeholder case this
        // diagnostic exists for.
        use crate::jitv2::PhysicalCodePage;
        use std::sync::atomic::AtomicU64;
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0x1234, &counter as *const AtomicU64);
        let mut codegen = Codegen::new();
        let mut instrs = [JIT_REGION_BOUNDARY_SENTINEL; ENTRIES_PER_PAGE];
        instrs[0] = (OP_ADDIU << 26) | (1 << 16) | 1;
        let mut analyzer = Analyzer::new();
        let (walked, non_empty) = analyzer.walk(&instrs, 0, 0);
        assert!(non_empty);
        let mut instrs_owned = *walked;
        instrs_owned[0].is_entry_point = true;
        let this_thread = std::thread::current().id();
        codegen.compile_region_uncommitted(&mut instrs_owned, true, true, false, &mut page as *mut PhysicalCodePage)
            .expect("a plain ADDIU must have a real emitter");

        let snap = codegen.seal_queue_snapshot();
        assert_eq!(snap.queue_len, 1, "the just-reserved, still-unpatched placeholder must be the only entry");
        assert_eq!(snap.front_is_unpatched_placeholder, Some(true),
            "compile_region_uncommitted alone (no finalize_batch_nonforced call yet) must leave the entry unpatched");
        assert_eq!(snap.front_thread_id, Some(this_thread),
            "the entry must be stamped with the thread that actually called push_placeholder");
        assert_eq!(snap.front_page, Some(&mut page as *mut PhysicalCodePage),
            "the entry must be stamped with the real page passed to compile_region_uncommitted, \
             not PublishInfo::page (which stays null until patch_pending_publish runs)");
    }

    #[test]
    fn finalize_batch_nonforced_leaves_gap_blocked_entries_pending() {
        // Compile two regions but only finalize the SECOND one first
        // (skipping the first's own finalize_definitions() call) — this
        // simulates one worker's compile landing in the queue behind
        // another still-unfinalized range. The second's own finalize must
        // not report itself sealed (it's blocked behind the first's gap:
        // its own push_placeholder entry, from compile_one_addiu, is still
        // unpatched — see paged_memory::PublishInfo's own doc comment for
        // why an unpatched entry blocks the contiguity scan).
        let mut codegen = Codegen::new();
        let first = compile_one_addiu(&mut codegen, 0);
        let second = compile_one_addiu(&mut codegen, 4);

        let sealed = codegen.finalize_batch_nonforced(second, dummy_publish());
        assert!(sealed.is_empty(), "second's range must stay blocked behind first's un-patched gap");
        assert!(!codegen.last_finalize_failed(),
            "a normal gap-block is not a finalize failure — last_finalize_failed must stay false, \
             or comp.rs's handle_request_deferred would wrongly skip incrementing `pending` for an \
             entry that's genuinely still waiting on a sweep, not permanently stuck");

        // Finalizing the first now makes the combined range contiguous from
        // the watermark, but still just two tiny instructions — nowhere
        // near a full page — so a non-forced finalize still must not seal
        // either (only force_seal_pending, exercised in the next test,
        // does that for a still-open page).
        let sealed = codegen.finalize_batch_nonforced(first, dummy_publish());
        assert!(sealed.is_empty(), "combined range is contiguous now but still within one still-open page — non-forced must not seal it");
        assert!(!codegen.last_finalize_failed(), "still just a gap-block, not a real failure");
    }

    #[test]
    fn seal_queue_snapshot_finds_the_real_gap_behind_a_patched_contiguous_front() {
        // Confirmed live shape: j2 seal-queue's front entry can be fully
        // patched (unpatched_placeholder=false) with a queue_len in the
        // thousands and nothing draining — the front alone is the wrong
        // place to look. Build exactly that: entry #0 finalized (patched,
        // contiguous from the watermark), entry #1 left as a bare
        // push_placeholder (never finalized) — the real, actionable gap.
        let mut codegen = Codegen::new();
        let front = compile_one_addiu(&mut codegen, 0);
        let stuck = compile_one_addiu(&mut codegen, 4);

        let sealed = codegen.finalize_batch_nonforced(front, dummy_publish());
        // front's own page is still open (too small to seal non-forced —
        // same reasoning as finalize_batch_nonforced_does_not_seal_a_still_open_page),
        // so it stays queued: patched, but not yet popped.
        assert!(sealed.is_empty());

        let snap = codegen.seal_queue_snapshot();
        assert_eq!(snap.queue_len, 2);
        assert_eq!(snap.front_is_unpatched_placeholder, Some(false),
            "sanity: the front entry (front's compile) must be patched — this is the shape that makes \
             the front alone misleading");
        assert_eq!(snap.first_gap_index, Some(1),
            "the scan must walk past the patched, contiguous front and stop at entry #1 (stuck's \
             still-unpatched placeholder) — that's the entry actually worth investigating");
        assert_eq!(snap.first_gap_is_unpatched_placeholder, Some(true));

        let entries = codegen.seal_queue_entries();
        assert_eq!(entries.len(), 2);
        assert!(!entries[0].2, "entries[0] (front) must report patched");
        assert!(entries[1].2, "entries[1] (the real gap) must report unpatched");
    }

    #[test]
    fn force_seal_pending_seals_a_dangling_partial_page() {
        // The real idle-timeout scenario: a lone small compile, finalized
        // non-forced (so its page stays open, per the test above), then
        // force_seal_pending (the ~100ms-idle-timeout sweep) closes it out
        // anyway — this is what guarantees a partial page never sits
        // un-dispatchable forever just because it never got big enough to
        // fill a page on its own.
        let mut codegen = Codegen::new();
        let id = compile_one_addiu(&mut codegen, 0);
        let sealed = codegen.finalize_batch_nonforced(id, dummy_publish());
        assert!(sealed.is_empty());

        let swept = codegen.force_seal_pending();
        assert_eq!(swept.len(), 1);
        assert!(swept[0].jit_fn.is_some());

        // And a second sweep with nothing new pending must be a harmless
        // no-op, not an error or a duplicate report.
        let swept_again = codegen.force_seal_pending();
        assert!(swept_again.is_empty());
    }

    #[test]
    fn force_seal_pending_blocks_on_a_range_that_was_compiled_but_never_finalized() {
        // A compiled-but-never-finalized FuncId still has its push_placeholder
        // entry sitting in the shared seal queue (reserved at compile time,
        // not at finalize time — see push_placeholder's own doc comment),
        // unpatched (jit_fn: None). force_seal_pending must not skip over
        // it — an unpatched entry blocks the contiguity scan regardless of
        // `force`, so the sweep must report nothing at all here.
        let mut codegen = Codegen::new();
        let _never_finalized = compile_one_addiu(&mut codegen, 0);

        let swept = codegen.force_seal_pending();
        assert!(swept.is_empty(), "an unpatched placeholder must block the sweep, not be skipped");
    }

    #[test]
    fn new_with_shared_arena_lets_two_codegens_pack_into_one_reservation() {
        // The compile-pool shape: N Codegens, one arena. Two Codegens built
        // via new_with_shared_arena over the same Arc<Mutex<SharedArena>>
        // must genuinely share address space — proven here by having each
        // compile+forced-finalize one function and asserting their packing
        // stats (read off the one shared PagedArenaState) accumulate
        // together rather than each starting its own separate reservation.
        let state = std::sync::Arc::new(crate::jitv2::paged_memory::PagedArenaState::default());
        let shared = crate::jitv2::paged_memory::PagedArenaMemoryProvider::new_shared(1 << 20, state.clone()).unwrap();

        let mut a = Codegen::new_with_shared_arena(shared.clone(), state.clone());
        let mut b = Codegen::new_with_shared_arena(shared, state.clone());

        let (used0, _) = state.packing_stats();
        assert_eq!(used0, 0);

        let id_a = compile_one_addiu(&mut a, 0);
        let jit_fn_a = a.finalize_batch(&[id_a]);
        assert_eq!(jit_fn_a.len(), 1);
        let (used1, _) = state.packing_stats();
        assert!(used1 > 0, "a's compile must have consumed real bytes from the shared arena");

        let id_b = compile_one_addiu(&mut b, 4);
        let jit_fn_b = b.finalize_batch(&[id_b]);
        assert_eq!(jit_fn_b.len(), 1);
        let (used2, _) = state.packing_stats();
        assert!(used2 > used1, "b's compile must have consumed MORE bytes from the SAME shared arena, on top of a's");
    }

    #[test]
    fn reset_with_shared_arena_reuses_the_given_arena_not_a_fresh_one() {
        // reset_with_shared_arena's whole point (vs. plain reset()) is NOT
        // reserving a new mmap — verify the resulting Codegen's own arena
        // base address matches the one explicitly handed in, not some new
        // reservation.
        let state = std::sync::Arc::new(crate::jitv2::paged_memory::PagedArenaState::default());
        let shared = crate::jitv2::paged_memory::PagedArenaMemoryProvider::new_shared(1 << 20, state.clone()).unwrap();
        let expected_base = crate::jitv2::paged_memory::PagedArenaMemoryProvider::from_shared(shared.clone()).arena_base();

        let mut codegen = Codegen::new(); // starts with its OWN private arena
        let original_base = codegen.seal_handle.arena_base();
        assert_ne!(original_base, expected_base, "sanity: Codegen::new()'s own arena must not coincidentally be the same reservation");

        unsafe { codegen.reset_with_shared_arena(shared, state); }
        assert_eq!(codegen.seal_handle.arena_base(), expected_base, "reset_with_shared_arena must rebuild on top of the given arena, not reserve a fresh one");
    }
}
