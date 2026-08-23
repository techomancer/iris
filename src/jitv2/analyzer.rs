//! JIT v2 static analyzer: reachability walker over page snapshots (§2.3).
//!
//! Standalone from the interpreter's `decode_into`/`MipsExecutor` machinery on
//! purpose — that decoder dispatches to executor handler function pointers and
//! drags in the whole cache/TLB stack. The analyzer only needs opcode/funct/
//! target classification, so it's built directly on `mips_isa.rs` constants,
//! matching exactly the same opcode patterns the interpreter's branch/jump
//! handlers (`mips_exec.rs`) use.
//!
//! This module provides [`Analyzer`] (the walker + its reusable scratch
//! buffer) and its classifier (`classify`); `src/bin/jitv2_analyze.rs` is the
//! runner that invokes it over a directory of corpus snapshots
//! (`jitv2/comp.rs`'s dump format).

use crate::jitv2::{ENTRIES_PER_PAGE, PAGE_SIZE};
use crate::mips_isa::*;

/// Word offset within a page (0..1024), matching `PhysicalCodePage`'s entry
/// indexing (§2.4).
pub type WordOffset = u16;

/// Runtime toggle for interpreter-fallback region admission (`j2 fallback
/// [on|off]`). When `false` (the default), an `Classify::Excluded` instruction
/// ends the region exactly as it did before fallback existed (the caller
/// records the boundary on its own edge, the excluded word is never visited);
/// when `true`, it's admitted as an `is_fallback` head and codegen runs it
/// through `interp_fallback_fn` (see `analyzer::visit` and
/// `codegen::emit_interp_fallback_head`). A process global (matching
/// `Codegen`'s own speed global) rather than a threaded param: the analyzer has
/// no other per-call config, and this only ever changes from the monitor
/// console, followed by a `mega_flush` so already-compiled regions are
/// rebuilt under the new policy. Defaults **off** while the fallback path is
/// still being validated against live boots.
static FALLBACK_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Enable/disable interpreter-fallback region admission (see
/// [`FALLBACK_ENABLED`]). The caller is responsible for flushing already-
/// compiled regions afterward (`Jitv2::mega_flush`) so the change takes effect.
pub fn set_fallback_enabled(on: bool) {
    FALLBACK_ENABLED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether interpreter-fallback region admission is currently on.
pub fn fallback_enabled() -> bool {
    FALLBACK_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The single process-wide lock serializing every test (in any module) that
/// forces [`FALLBACK_ENABLED`] on — one mutex, shared, so an analyzer-module
/// fallback test and an equiv_test-module fallback test can never run
/// concurrently and stomp each other's flag. Both modules acquire this exact
/// static via [`test_fallback_guard`]; two separate locks would not be mutually
/// exclusive (the bug this consolidation fixes).
#[cfg(test)]
pub(crate) static FALLBACK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII test helper (any module): hold [`FALLBACK_TEST_LOCK`] and force
/// fallback ON for the guard's lifetime, restoring OFF on drop (incl. panic).
#[cfg(test)]
#[must_use]
pub(crate) fn test_fallback_guard() -> TestFallbackGuard {
    let lock = FALLBACK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_fallback_enabled(true);
    TestFallbackGuard { _lock: lock }
}

#[cfg(test)]
pub(crate) struct TestFallbackGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
}
#[cfg(test)]
impl Drop for TestFallbackGuard {
    fn drop(&mut self) {
        set_fallback_enabled(false);
    }
}

const WORDS_PER_PAGE: u16 = (PAGE_SIZE / 4) as u16;

/// Byte offset 0xFFC — the last word of a page. A branch/jump here has its
/// delay slot on the next physical page, which the walker (or the compiler)
/// can't see (§2.3's "0xFFC hazard").
const OFFSET_0XFFC_WORD: u16 = 0xFFC / 4;

/// Classification of a single decoded instruction, as far as reachability
/// walking cares. Deliberately coarser than the interpreter's `DecodedInstr`
/// — no register/immediate extraction beyond what's needed to compute branch
/// targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classify {
    /// Falls through to the next word. The overwhelming majority of instructions.
    Sequential,
    /// Conditional branch (BEQ/BNE/BLEZ/BGTZ/BLTZ/BGEZ/BLTZAL/BGEZAL and the
    /// "likely" variants). `target` is the word offset within *this* page if
    /// the branch stays on-page; `None` if the target is off-page.
    Branch { target: Option<WordOffset> },
    /// J/JAL. `target` is always `None` (page-leaving) — see `jump_target`'s
    /// doc comment for why this is never resolved on-page, unlike `Branch`:
    /// J/JAL's encoding is an absolute word position within a fixed 256MB
    /// region, and classifying it against a specific page's identity would
    /// make the compiled function's own shape depend on which physical page
    /// happens to be compiling, breaking position independence. Kept as an
    /// `Option` (matching `Branch`'s shape) rather than a bare unit variant
    /// so callers don't need a separate match arm from `Branch`'s on/off-page
    /// handling — always taking the `None` arm.
    Jump { target: Option<WordOffset> },
    /// JR/JALR — register-indirect, always page-leaving/region-end (§2.3).
    RegJump,
    /// CP0 moves/TLB ops/CACHE/ERET/SYSCALL/BREAK/WAIT (§4.4), LL/SC, CP1's
    /// conditional branch (RS_BC1 — condition-code-dependent target, not
    /// resolved by this walker), and CP2 (unimplemented on this platform).
    /// Note this is narrower than §4.4's literal "all CP1/FPU" — plain CP1
    /// arithmetic/move/compare/load/store ops classify as `Sequential`
    /// instead (see `classify`'s doc comment for the rationale).
    Excluded,
    /// The `JIT_REGION_BOUNDARY_SENTINEL` word — a hard region end that is
    /// never visited, never compiled, and never run through the interpreter
    /// fallback (unlike `Excluded`, which is now kept in the region as a
    /// fallback head). Behaves exactly like `Excluded` did before
    /// interpreter-fallback existed: the walk stops at the predecessor's edge.
    /// A test/tooling device (see `mips_isa::JIT_REGION_BOUNDARY_SENTINEL`), not
    /// something a real guest binary contains.
    RegionBoundary,
}

/// Decode `raw` (a 32-bit MIPS instruction word) into its reachability
/// classification. `offset_word` is this instruction's word offset within
/// the page, used by `branch_target`'s PC-relative math. `page_base` is
/// still threaded through for the walker's own recursive traversal
/// (`visit`/`visit_slot`'s branch-target follow-up), even though `classify`
/// itself no longer consults it directly — J/JAL used to (`jump_target`
/// resolving on-page-ness against it), until that was found to break
/// position independence (see `jump_target`'s doc comment) and J/JAL was
/// moved to the same always-page-leaving treatment as RegJump.
pub fn classify(raw: u32, offset_word: u16, page_base: u32) -> Classify {
    // Test/tooling region-boundary sentinel — checked before opcode decode so
    // no real opcode arm can ever shadow it (see the constant's doc comment).
    if raw == JIT_REGION_BOUNDARY_SENTINEL {
        return Classify::RegionBoundary;
    }
    let op = (raw >> 26) & 0x3F;
    let rs = (raw >> 21) & 0x1F;
    let rt = (raw >> 16) & 0x1F;
    let funct = raw & 0x3F;

    match op {
        OP_SPECIAL => match funct {
            FUNCT_JR | FUNCT_JALR => branch_category_gate(raw, Classify::RegJump),
            FUNCT_SYSCALL | FUNCT_BREAK => Classify::Excluded,
            _ => sequential_or_excluded(raw),
        },
        OP_REGIMM => match rt {
            RT_BLTZ | RT_BGEZ | RT_BLTZL | RT_BGEZL | RT_BLTZAL | RT_BGEZAL
            | RT_BLTZALL | RT_BGEZALL => branch_category_gate(raw, branch_target(raw, offset_word)),
            // Trap-immediate REGIMM variants (TGEI/TGEIU/TLTI/TLTIU/TEQI/TNEI):
            // no unresolved control flow either way (they either trap —
            // exception, out of scope for reachability — or fall through),
            // so whether they're a region boundary is purely a "does codegen
            // have an emitter yet" question, same as everything else routed
            // through sequential_or_excluded.
            _ => sequential_or_excluded(raw),
        },
        OP_J | OP_JAL => branch_category_gate(raw, jump_target()),
        OP_BEQ | OP_BNE | OP_BLEZ | OP_BGTZ
        | OP_BEQL | OP_BNEL | OP_BLEZL | OP_BGTZL => branch_category_gate(raw, branch_target(raw, offset_word)),
        OP_COP0 => Classify::Excluded, // MFC0/MTC0/CFC0/CTC0/TLB*/ERET/WAIT all live under COP0
        // CP1 arithmetic/move/compare ops are plain data-flow (moves, add/sub/
        // mul/div/sqrt/convert/compare) — no unresolved control flow, so
        // whether they compile is purely an emitter-coverage question.
        // FCSR-enabled-exception risk applies uniformly across all of them,
        // not just a subset, and is a codegen concern (materialize state
        // before the op so a raised CP1 exception can be delivered
        // correctly), not a reachability one — see the design doc's Phase 3
        // note. RS_BC1 is the one real exception: it's a CP1-conditional
        // branch, and the walker doesn't resolve condition-code-dependent
        // targets, so it stays a region boundary. `rs == RS_BC1` (0x08) is
        // only meaningful as a format selector for OP_COP1 — for OP_COP1X
        // the same bit position is the *base register* for indexed loads/
        // stores (LWXC1/SWXC1/etc.), not a branch selector, so OP_COP1X
        // must never share this arm (a legitimate `lwxc1 $f0, ($8)` would
        // otherwise be misread as a BC1 branch and wrongly excluded).
        OP_COP1 => match rs {
            RS_BC1 => Classify::Excluded,
            _ => sequential_or_excluded(raw),
        },
        OP_COP1X => sequential_or_excluded(raw),
        OP_COP2 => Classify::Excluded, // unimplemented coprocessor
        OP_CACHE => Classify::Excluded,
        OP_LL | OP_LLD | OP_SC | OP_SCD => Classify::Excluded,
        // FPU loads/stores are plain memory ops, same as any other load/store.
        OP_LWC1 | OP_LDC1 | OP_SWC1 | OP_SDC1 => sequential_or_excluded(raw),
        OP_LWC2 | OP_LDC2 | OP_SWC2 | OP_SDC2 => Classify::Excluded, // CP2, unimplemented — treat as excluded
        _ => sequential_or_excluded(raw),
    }
}

/// Whether an `Classify::Excluded` instruction is a **control-transfer** that
/// arms a delay slot when run through the interpreter fallback — i.e. one whose
/// `interp_dispatch_one` can leave `core.in_delay_slot = true` with a pending
/// `delay_slot_target`, so its successor (the delay slot) must be treated as an
/// entry-like word (honor the pending transfer) rather than a plain fallthrough.
/// Today the only such excluded opcode is `BC1` (branch on CP1 condition):
/// SYSCALL/BREAK/COP2/CACHE/LL/SC don't transfer control; ERET/COP0 move `core.pc`
/// but never arm a delay slot (the fallback's off-page / pc-moved check already
/// handles those). Kept as a standalone predicate (not a `CompiledInstr` field)
/// so codegen can call it directly from `ctx.raw`; the analyzer only needs to
/// tag the *successor* word (`is_branch_fallback_successor`).
pub fn is_fallback_branch(raw: u32) -> bool {
    let op = (raw >> 26) & 0x3F;
    let rs = (raw >> 21) & 0x1F;
    op == OP_COP1 && rs == RS_BC1
}

/// Whether `raw` is any CP1/FPU instruction — arithmetic, move, compare,
/// convert (`OP_COP1`, any `rs` including `RS_BC1`), indexed FPU load/store
/// (`OP_COP1X`), or plain FPU load/store (`OP_LWC1`/`OP_LDC1`/`OP_SWC1`/
/// `OP_SDC1`). Used by [`Analyzer::walk_multi_entry`]/[`Analyzer::walk_bounded`]
/// to compute a region's `has_fpu` flag as a byproduct of the walk itself
/// (one scan, at analyzer time) rather than codegen re-scanning the walked
/// buffer afterward via its own `lookup_cp1_semantics` table — this is a
/// pure opcode-shape question, the same kind of thing `classify` already
/// answers, so it belongs here, not duplicated as a second classification
/// authority in codegen.
pub fn is_fpu_instruction(raw: u32) -> bool {
    let op = (raw >> 26) & 0x3F;
    matches!(op, OP_COP1 | OP_COP1X | OP_LWC1 | OP_LDC1 | OP_SWC1 | OP_SDC1)
}

/// `Sequential` if `codegen.rs` actually has an emitter for `raw`
/// (`opcode_support::has_emitter`, the single source of truth shared with
/// codegen's own lookup tables), `Excluded` otherwise. Before this existed,
/// every arm below fell straight through to `Classify::Sequential`
/// unconditionally — correct only as long as `codegen.rs`'s emitter
/// coverage matched that blanket assumption, which it didn't (several
/// opcodes, e.g. MOVZ/MOVN/DADDI/the unaligned load-store family/all of
/// OP_COP1X, had no emitter at all — see
/// `rules/jitv2/unsupported-instructions.md`). `compile_region`'s upfront
/// rejection loop declines the *whole* region if any visited instruction
/// lacks an emitter, so a wrongly-`Sequential` opcode didn't just fail to
/// compile itself — it silently poisoned every other instruction in
/// whatever region it happened to be walked into. Routing through
/// `has_emitter` makes an unimplemented opcode a clean `Excluded` region
/// boundary instead (same as an architecturally-excluded one), and makes
/// adding a new emitter automatically un-exclude it — no analyzer.rs edit
/// needed.
fn sequential_or_excluded(raw: u32) -> Classify {
    if crate::jitv2::opcode_support::has_emitter(raw) {
        Classify::Sequential
    } else {
        Classify::Excluded
    }
}

/// Runtime-toggle gate for the `Branch`/`Jump`/`RegJump` arms: these never
/// call `has_emitter` (they're resolved by `classify`'s own construction,
/// not an emitter-coverage lookup), so `j2 branch [on|off]` /
/// `j2 <instr> [on|off]` would otherwise never apply to them.
/// `already_classified` is whatever `classify` already computed
/// (`Branch { target }`/`Jump { target }`/`RegJump`); this only downgrades
/// it to `Excluded` when `raw`'s `InstrKind` is currently disabled in
/// `opcode_support`'s per-instruction table — same "clean region boundary
/// instead of poisoning the region" contract as `sequential_or_excluded`.
fn branch_category_gate(raw: u32, already_classified: Classify) -> Classify {
    let op = (raw >> 26) & 0x3F;
    let rs = (raw >> 21) & 0x1F;
    let rt = (raw >> 16) & 0x1F;
    let funct = raw & 0x3F;
    let kind = crate::mips_instr_stats::classify_instr(op as u8, rs as u8, rt as u8, funct as u8);
    if crate::jitv2::opcode_support::instr_enabled(kind) {
        already_classified
    } else {
        Classify::Excluded
    }
}

/// Compute a conditional branch's target word offset, `None` if it leaves
/// the page. `imm16` is sign-extended and measured in words from the delay
/// slot's own address (`offset_word + 1`), matching MIPS's
/// `PC + 4 + (imm16 << 2)` with `PC` = this instruction's address.
fn branch_target(raw: u32, offset_word: u16) -> Classify {
    // target = PC_branch + 4 + (imm16 << 2), i.e. in word units:
    // offset_word + 1 (the delay slot's word) + imm16 — NOT + 2. (A branch's
    // target is defined relative to the delay slot's address, one word past
    // the branch itself, not two.) Matches MipsExecutor::exec_beq et al.'s
    // `self.core.pc.wrapping_add(4).wrapping_add(d.immu64())` exactly.
    let imm16 = (raw & 0xFFFF) as i16 as i32;
    let target = offset_word as i32 + 1 + imm16;
    let in_page = (0..WORDS_PER_PAGE as i32).contains(&target);
    Classify::Branch { target: in_page.then_some(target as u16) }
}

/// J/JAL is always treated as leaving the page — same footing as RegJump
/// (JR/JALR), never resolved to an on-page word offset at analysis time.
/// J/JAL's real target, `((pc+4) & 0xFFFFFFFF_F0000000) | imm28`
/// (`MipsExecutor::exec_j`/`exec_jal`), is an *absolute* word position
/// within a fixed 256MB region baked into the instruction's own encoding —
/// unlike `branch_target`'s pure PC-relative delta, whether that lands on
/// "this" page can only be known by comparing against the page's actual
/// identity. Classifying it as on-page would make the *compiled function's
/// own shape* — which branch of codegen's `emit_jump_taken_edge` gets taken,
/// on-page-jump vs. absolute-address bail — depend on which physical page
/// happens to be compiling, breaking position independence (§2.2): the same
/// raw instruction word could compile to two different function shapes on
/// two different pages, for a JIT explicitly designed to make compiled code
/// carry no page identity of its own. Codegen's off-page path
/// (`emit_jump_target_addr` + `emit_absolute_pc_exit`) already computes the
/// exact right target from the live `core.pc` at runtime regardless — it's
/// no less correct or slower in the common on-page case than a dedicated
/// on-page branch would be, just uniform.
fn jump_target() -> Classify {
    Classify::Jump { target: None }
}

/// Why a particular outgoing edge of an instruction exits the compiled
/// region instead of continuing into another compiled instruction. Never
/// attributed to an excluded instruction itself — an excluded instruction is
/// by definition never compiled/visited (§4.4) — always to the edge of
/// whichever compiled instruction leads into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The edge's target left the page.
    PageLeaving,
    /// JR/JALR — target unknown statically, always exits.
    RegJump,
    /// The edge's target is an excluded instruction (§4.4) — control exits
    /// to the interpreter here rather than continuing into it.
    Excluded,
    /// A branch/jump/regjump's mandatory delay slot is on the next physical
    /// page (offset 0xFFC — §2.3's "0xFFC hazard"), so it can't be inlined.
    /// Recorded on *every* edge this instruction has (both taken and
    /// not-taken/fallthrough, or the sole edge for RegJump/Jump) — codegen
    /// materializes `core.in_delay_slot`/`core.delay_slot_target` and exits
    /// directly to the next page, mirroring `branch_delay`/
    /// `handle_branch_not_taken` exactly, rather than inlining a slot that
    /// isn't there. The entry-side counterpart is `exec_decoded`'s
    /// `entry_offset == 0` always-probe, which already consumes this
    /// runtime state correctly regardless of which page armed it.
    ForeignPageSlot,
    /// The walk's instruction budget (`Analyzer::walk_bounded`) ran out
    /// before this edge's target could be visited. Test/tooling scaffolding
    /// only — the unbounded `Analyzer::walk` used by the real compiler never
    /// produces this. Distinct from `Excluded` so callers can't mistake
    /// "asked for a small region on purpose" for a genuine exclusion boundary.
    Truncated,
}

/// One slot in the per-page compile record array (§2.3/§6.1.2's "one
/// Cranelift IR block per instruction" — this is the pre-codegen scaffolding
/// for that: the walker only marks reachability, actual codegen fills in
/// `block_id` later per instruction).
///
/// Every instruction has up to two outgoing edges codegen cares about
/// independently — e.g. a conditional branch whose taken arm continues into
/// compiled code but whose not-taken arm hits an excluded instruction: the
/// branch still compiles (test + conditional jump to the taken arm's IR
/// block), but the not-taken side needs its own exit stub (materialize
/// `pc = bd_slot + 4`, return to the interpreter) distinct from whatever the
/// taken side does. A single `stop` field can't represent "one arm continues,
/// the other exits," hence two separate fields:
/// - `fallthrough_exit`: the "didn't branch" edge — `Sequential`'s only edge,
///   or a `Branch`'s not-taken arm (both mean the same thing: "control
///   reaches the next word in program order without transferring").
/// - `taken_exit`: the "did branch/jump" edge — `Branch`/`Jump`'s target, or
///   `RegJump`'s sole (always-exiting) edge. `None` for `Sequential`, which
///   has no such edge.
#[derive(Clone, Copy)]
pub struct CompiledInstr {
    /// Whether the walk reached this word offset. `false` for every slot
    /// until the walker visits it; checked before recursing so converging
    /// paths (loop back-edges, two branches into the same target) stop
    /// immediately instead of re-walking. Reset to `false` for the whole
    /// array at the start of every [`Analyzer::walk`] call.
    pub visited: bool,
    pub word: WordOffset,
    pub raw: u32,
    /// Set by codegen (not the walker) once this instruction has an emitted
    /// Cranelift IR block — an opaque id here rather than a real
    /// `cranelift_codegen::ir::Block` so the analyzer stays free of the
    /// Cranelift dependency until codegen actually exists.
    pub block_id: Option<u32>,
    /// Exit reason for the "fall through / not taken" edge, if that edge
    /// exits the region. `None` means this edge continues into another
    /// compiled instruction (or doesn't exist, e.g. `RegJump`/`Jump`).
    pub fallthrough_exit: Option<StopReason>,
    /// Exit reason for the "branch/jump taken" edge, if that edge exits the
    /// region. `None` means this edge continues into another compiled
    /// instruction (or doesn't exist, e.g. plain `Sequential`).
    pub taken_exit: Option<StopReason>,
    /// `true` iff this word has been visited *only* as some branch/jump's
    /// inline delay slot (or as a slot's own nested slot, arbitrarily deep —
    /// `visit_slot`), never as a genuine head instruction with its own
    /// computed `fallthrough_exit`/`taken_exit`. Codegen uses this directly
    /// to decide block allocation (§6.1.4 "same-offset dual semantics" —
    /// only a *purely* slot word skips getting its own Cranelift block; a
    /// word that's also reached as a real branch/jump target, even if it
    /// started out slot-only, gets promoted — see `visit`'s doc comment —
    /// and this flips to `false`). Meaningless when `visited` is `false`.
    pub is_slot_only: bool,
    /// `true` iff this word is an `Classify::Excluded` instruction that the
    /// walker kept in the region as an **interpreter-fallback** head (rather
    /// than ending the region on contact, the pre-fallback behavior). Codegen
    /// emits its int-check preamble, then a call to `core.interp_fallback_fn`
    /// (running the excluded instruction through the real interpreter handler),
    /// then either returns the handler's non-`EXEC_COMPLETE` status or — if the
    /// handler advanced `core.pc` by exactly one word — falls through to the
    /// successor, which becomes "entry-like" (materializes pc/bd from `core`,
    /// see codegen). Its `fallthrough_exit` follows the same rule as a
    /// `Sequential`'s (set iff the successor exits the region). Only ever set
    /// on a *head* fallback: an excluded instruction in a delay-slot position
    /// still declines the whole branch (`visit_slot` returns `false`), a
    /// separately-scoped follow-up. Meaningless when `visited` is `false`.
    pub is_fallback: bool,
    /// `true` iff this word is the immediate successor (in program order) of a
    /// **branch** interpreter-fallback (`is_fallback_branch`, i.e. a `BC1`) —
    /// so it is that branch's delay slot. Codegen must emit it as an
    /// entry-like word (the same `core.in_delay_slot`/`delay_slot_target`
    /// foreign-slot check the region's real entry word does), because after the
    /// branch fallback runs the interpreter, this word is reached with
    /// `in_delay_slot = true` and a pending transfer armed — a plain
    /// fallthrough here would silently drop the branch (found: BC1 taken/
    /// not-taken-non-likely lose the transfer). A non-branch fallback's
    /// successor never has this set and stays a plain fallthrough.
    /// Meaningless when `visited` is `false`.
    pub is_branch_fallback_successor: bool,
    /// `true` iff this word is the explicit `target` of some `Branch`/`Jump`
    /// in the region — i.e. reachable by an edge other than falling straight
    /// off the previous word. Set at the two `finish_visit` sites that
    /// recurse into a branch/jump's resolved `target` (never for the
    /// not-taken/past-the-slot `offset + 2` edge, which is ordinary
    /// sequential flow, not a jump destination). Codegen consults this
    /// before fusing a `Sequential` instruction with its immediate successor
    /// (LUI+ORI/ADDIU 32-bit-immediate materialization, `opcodefusion`'s
    /// jitv2 counterpart): fusing must never let the producer's own block
    /// skip past a word that something else can jump directly into, since
    /// that arrival needs the *un-fused* instruction to run — this flag is
    /// exactly "would fusing orphan a real predecessor of word+1". Distinct
    /// from `is_slot_only`'s narrower "reached only as a mandatory,
    /// never-independently-targeted delay slot" — a branch target is the
    /// opposite case, explicitly and independently reachable.
    pub is_branch_target: bool,
    /// The on-page word the fallthrough (not-taken/sequential) edge
    /// continues to, when that edge does *not* exit the region — mirrors
    /// `fallthrough_exit`'s "one of these two is set, never both" split
    /// (see this struct's own doc comment on why a `Branch`'s two edges
    /// each need independent bookkeeping): `None` whenever
    /// `fallthrough_exit` is `Some` (the edge exits instead) or the
    /// instruction has no fallthrough edge at all (`RegJump`/`Jump`).
    /// Filled in by the same `finish_visit`/`visit` call sites that
    /// already resolve `fallthrough_exit` — exists purely so
    /// `compute_cycles_flush` (a separate post-pass — see its own doc
    /// comment for why it can't run inline during the walk) can look up
    /// "what word does this edge land on" without re-deriving branch/jump
    /// target arithmetic a second time. Meaningless when `visited` is
    /// `false`.
    pub continues_to_fallthrough: Option<WordOffset>,
    /// The on-page word the taken (branch/jump) edge continues to, when
    /// that edge does *not* exit the region — the `taken_exit` counterpart
    /// to `continues_to_fallthrough`, same rationale. `None` whenever
    /// `taken_exit` is `Some` or the instruction has no taken edge
    /// (`Sequential`).
    pub continues_to_taken: Option<WordOffset>,
    /// `true` iff this word is a `Branch`/`Jump`/`RegJump` head with a real,
    /// on-page mandatory delay slot inlined at `word + 1` (§6.1.4) — `false`
    /// for `Sequential`/`is_fallback` heads (no slot at all) and for the
    /// 0xFFC foreign-page-slot case (`finish_visit_foreign_page_slot`'s
    /// callers — a branch/jump/regjump whose slot is on the *next*,
    /// unwalkable page, so there's no `instrs[word+1]` on this page to
    /// inline). Exists purely for `compute_cycles_flush`: emission order
    /// puts the slot's own `emit_account_for_cycles` call strictly after
    /// the head's (the head is accounted for in the pass-2 loop; the slot
    /// is accounted for down inside whichever branch/jump emitter
    /// processes it, called afterward) — so a flush decided for the head
    /// itself would fire *before* the slot's own +1 ever accrues, silently
    /// losing it (found live: a one-instruction `JR`+NOP region reported
    /// `cycles` advancing by 1, not 2). `compute_cycles_flush` uses this
    /// flag to push a would-be head flush onto the slot word instead,
    /// since the slot is always the true last-to-retire word of the pair.
    pub has_inline_slot: bool,
    /// Always 1 — every visited word (a real head or a delay-slot-only
    /// word alike) retires exactly one architectural instruction. Kept as
    /// an explicit field rather than codegen using a bare literal so "how
    /// much does this word contribute to the pending cycles count" and
    /// "does this word flush that pending count to memory"
    /// (`cycles_flush`) stay visibly independent concerns. See the
    /// cycles-batching design: `rules/jitv2/` (or this module's own
    /// `compute_cycles_flush` doc comment) for the full rationale — in
    /// short, `Hot::cycles` must be current at every region exit and at
    /// every loop re-entry point (for cross-thread observability of a
    /// long-running compiled loop that never exits), but a straight-line
    /// run of instructions between two such points can defer the store to
    /// wherever the run ends instead of paying one store per instruction.
    pub cycles_delta: u32,
    /// `true` iff `core.hot.cycles` must be stored (flushed with whatever
    /// has accrued since the previous flush) on this instruction's
    /// outgoing edge, before control leaves along it. Computed by
    /// `compute_cycles_flush`, a separate pass run after the reachability
    /// walk completes (`is_branch_target` can be set on a word *later* in
    /// the walk than the word itself was visited, so this can't be decided
    /// inline during `visit`/`finish_visit`). `true` for every region-exit
    /// word (`is_region_exit()`) and every instruction with a continuing
    /// edge (`continues_to_fallthrough`/`continues_to_taken`) that lands on
    /// an `is_branch_target` word or on `entry_word` itself (loop re-entry,
    /// including the case where the
    /// region's own entry word is the loop head) — both need cycles fresh
    /// on arrival for the same cross-thread-observability reason exits do.
    /// `false` for a plain interior word whose only edge continues
    /// straight into another compiled instruction with no other
    /// predecessor — codegen defers that word's `cycles_delta` into
    /// whichever later word's flush eventually covers it. Meaningless when
    /// `visited` is `false`.
    pub cycles_flush: bool,
    /// §13.4: `true` iff this word is one of the compile's external entry
    /// points (a bit set in the `entry_words` passed to
    /// [`Analyzer::walk_multi_entry`]) — codegen's replacement for the old
    /// single-`entry_word` scalar comparison (`word == entry_word`) at every
    /// per-instruction emission site (preamble skip, exception two-stage
    /// routing, FPU-guard bail target, dual-semantics block choice). Set
    /// directly by the walk driver on every word it walked *as* an entry
    /// (not inferred from reachability — a word can be fully reachable as an
    /// internal instruction and still not be one of this particular
    /// compile's requested entries). Mirrors `is_slot_only`'s existing
    /// "upgrade on a later, different-role visit" pattern: a word walked
    /// first as a plain internal instruction from one entry, then later
    /// found to itself be a second entry point, gets this flipped `true` by
    /// the walk driver without needing to re-walk or re-classify it — `visit`
    /// already computed correct edges for it on the first pass. Meaningless
    /// when `visited` is `false`.
    pub is_entry_point: bool,
}

impl CompiledInstr {
    /// `true` if this instruction has no exit edges at all — every edge it
    /// has continues into another compiled instruction. Convenience for
    /// callers that don't care which edge, just whether the instruction is
    /// an internal region node vs. a region boundary.
    pub fn is_region_exit(&self) -> bool {
        self.fallthrough_exit.is_some() || self.taken_exit.is_some()
    }
}

impl Default for CompiledInstr {
    fn default() -> Self {
        Self {
            visited: false, word: 0, raw: 0, block_id: None, fallthrough_exit: None, taken_exit: None, is_slot_only: false, is_fallback: false, is_branch_fallback_successor: false, is_branch_target: false,
            continues_to_fallthrough: None, continues_to_taken: None, has_inline_slot: false, cycles_delta: 0, cycles_flush: false, is_entry_point: false,
        }
    }
}

/// Reachability walker with a reusable per-page scratch buffer (§2.3). One
/// instance is meant to live for the compile thread's lifetime and be reused
/// across every job — a page is always exactly 4KB/1024 words, so there's no
/// reason to heap-allocate a fresh `[CompiledInstr; 1024]` per compile
/// request. [`Self::walk`] resets the buffer in place before each walk.
///
/// §13.4: [`Self::walk_multi_entry`] additionally stores `has_fpu`/`covered`
/// on `self` as it walks, rather than returning them as a tuple — callers
/// (`comp.rs`'s `handle_request`/`handle_request_deferred`) read them back
/// via [`Self::has_fpu`]/[`Self::covered`] once the walk returns, so the
/// walk's own derived facts live alongside the buffer they were derived
/// from instead of being threaded separately through every caller.
/// Overwritten by the next `walk`/`walk_bounded`/`walk_multi_entry` call,
/// same reuse contract as `instrs` itself.
pub struct Analyzer {
    instrs: Box<[CompiledInstr; ENTRIES_PER_PAGE]>,
    has_fpu: bool,
    covered: Vec<WordOffset>,
}

impl Analyzer {
    pub fn new() -> Self {
        Self { instrs: Box::new([CompiledInstr::default(); ENTRIES_PER_PAGE]), has_fpu: false, covered: Vec::new() }
    }

    /// Whether the most recent [`Self::walk_multi_entry`] call's merged
    /// region contained any CP1/FPU instruction (`is_fpu_instruction`).
    /// Meaningless before the first `walk_multi_entry` call.
    #[inline]
    pub fn has_fpu(&self) -> bool {
        self.has_fpu
    }

    /// The subset of `entry_words` the most recent [`Self::walk_multi_entry`]
    /// call actually admitted (didn't decline as excluded-at-entry) —
    /// borrowed, not cloned; callers that need to keep it past the next walk
    /// call should copy it out.
    #[inline]
    pub fn covered(&self) -> &[WordOffset] {
        &self.covered
    }

    /// Owned copy of the most recent walk's buffer — codegen needs `&mut
    /// [CompiledInstr; ENTRIES_PER_PAGE]` (it writes `block_id` back into
    /// it), so callers that also want to keep reading `self` (`covered()`/
    /// `has_fpu()`) via other calls need their own copy rather than holding
    /// the borrow `walk_multi_entry`'s return value ties to `self`.
    #[inline]
    pub fn instrs_snapshot(&self) -> [CompiledInstr; ENTRIES_PER_PAGE] {
        *self.instrs
    }

    /// Walk a region starting at `entry_word` within `page` (§2.3): decode
    /// from the entry, follow both arms of every intra-page conditional
    /// branch and same-page J/JAL — each arm a separate recursive call,
    /// skipped entirely if its target was already visited — stop at
    /// page-leaving jumps/branches, `jr`/`jalr`, and the 0xFFC hazard. An
    /// excluded instruction (§4.4) is never itself compiled/visited; the
    /// instruction that would have fallen through/branched/jumped into it
    /// gets its `fallthrough_exit`/`taken_exit` (whichever edge led there)
    /// set to `Excluded` instead, and the excluded instruction is left
    /// untouched for the interpreter.
    ///
    /// Returns `(instrs, non_empty)`: `instrs` is a reference to the
    /// (freshly reset) scratch buffer — only `visited` entries are
    /// meaningful, collect them via [`instrs_linear`]. `non_empty` is
    /// `false` iff `entry_word` itself is excluded — the only way to get a
    /// zero-instruction region (§6.4's "excluded-first-instruction"
    /// sticky-rejection reason; a non-excluded entry always visits at least
    /// itself) — so callers can distinguish "nothing to compile, reject this
    /// entry" from every other case without re-scanning the buffer.
    pub fn walk(&mut self, page: &[u32; ENTRIES_PER_PAGE], entry_word: WordOffset, page_base: u32) -> (&[CompiledInstr; ENTRIES_PER_PAGE], bool) {
        self.walk_bounded(page, entry_word, page_base, usize::MAX)
    }

    /// Same as [`Self::walk`], but stops growing the region once `max_instrs`
    /// distinct *head* instructions have been visited — a branch/jump's
    /// mandatory delay slot (or nested slot-chain, for the "unusual but
    /// legal" branch-in-delay-slot case) is never charged against
    /// `max_instrs` (`visit_slot` — a slot can never be omitted, so it was
    /// never a truncation candidate). Building block for codegen tests that
    /// want a single, small, hand-picked region (e.g. exactly one
    /// instruction wrapped by entry/exit) instead of whatever the unbounded
    /// walk from a real corpus page would produce — see
    /// `jitv2/codegen_test.rs`. `max_instrs = usize::MAX` (via [`Self::walk`])
    /// recovers the unbounded behavior used everywhere else.
    ///
    /// A branch/jump whose taken and/or fallthrough targets are cut off by
    /// the budget gets [`StopReason::Truncated`] on that edge — a distinct
    /// reason from `Excluded`, so callers (and the region-size stats in
    /// `jitv2_analyze`) never confuse "hit the instruction budget" with a
    /// genuine excluded-instruction boundary.
    pub fn walk_bounded(&mut self, page: &[u32; ENTRIES_PER_PAGE], entry_word: WordOffset, page_base: u32, max_instrs: usize) -> (&[CompiledInstr; ENTRIES_PER_PAGE], bool) {
        self.instrs.fill(CompiledInstr::default());
        let mut budget = Budget::new(max_instrs, entry_word);
        let non_empty = visit(&mut self.instrs, page, page_base, entry_word, &mut budget);
        if non_empty {
            compute_cycles_flush(&mut self.instrs, entry_word, budget.min, budget.max);
        }
        (&self.instrs, non_empty)
    }

    /// §13.4: walk every offset set in `entry_words` into one shared buffer,
    /// merging their reachable sets. A word reachable from more than one
    /// entry point is analyzed once (`visit`'s own already-visited
    /// short-circuit — see its doc comment) and its Cranelift body still only
    /// gets emitted once by codegen; this is the actual duplication
    /// elimination §13 exists for. No analyzer-side "is this word also an
    /// entry point" bookkeeping is needed: entry-hood is purely a codegen-time
    /// property (bitmap membership), since `visit` already computes correct,
    /// promotable edges for a word regardless of whether this is its first or
    /// a later visit (`is_slot_only` promotion is the existing precedent for
    /// exactly this kind of "upgrade an already-walked word" case).
    ///
    /// Returns the merged buffer (same scratch storage `walk_bounded` uses).
    /// Also stores, on `self` (read back via [`Self::covered`]/
    /// [`Self::has_fpu`] — see the struct's own doc comment for why these
    /// live alongside the buffer instead of being returned as a tuple):
    /// `covered`, the subset of `entry_words` that actually produced a
    /// non-empty walk (i.e. weren't excluded-at-entry) — callers should
    /// treat any bit in `entry_words` but not in `covered` as a per-offset
    /// decline (§6.4 sticky-denylist candidate for that one offset), not
    /// fail the whole compile; and `has_fpu` (`is_fpu_instruction`, see its
    /// own doc comment), computed as a byproduct of the same
    /// `instrs_linear` pass every caller already needs to run over the
    /// result — codegen's region-wide FR-mode guard (`emit_fr_mode_guard`)
    /// consumes this instead of re-scanning the buffer itself (the
    /// per-instruction CU1 check, `emit_cp1_cu1_guard`, doesn't need this at
    /// all — it's driven directly by each instruction's own opcode during
    /// pass 2, not by a region-wide flag). Empty `entry_words` produces an empty buffer, empty
    /// `covered`, `has_fpu = false` — not a caller error (a request whose
    /// every bit turned out already-covered/denied by the time of the
    /// pre-compile subsumption check, §13.3 step 4, may legitimately have
    /// nothing left to walk).
    pub fn walk_multi_entry(
        &mut self,
        page: &[u32; ENTRIES_PER_PAGE],
        entry_words: &[WordOffset],
        page_base: u32,
        max_instrs: usize,
    ) -> &[CompiledInstr; ENTRIES_PER_PAGE] {
        self.instrs.fill(CompiledInstr::default());
        self.covered.clear();
        let mut any_covered = false;
        let mut min_visited = WordOffset::MAX;
        let mut max_visited = 0;
        for &entry_word in entry_words {
            let mut budget = Budget::new(max_instrs, entry_word);
            if visit(&mut self.instrs, page, page_base, entry_word, &mut budget) {
                // Mark AFTER visit returns, unconditionally — `visit`'s own
                // already-visited-as-head short-circuit means a word already
                // marked a real head by an earlier entry's walk is never
                // reconstructed by this call, so this can't be clobbered by
                // a later entry in the same loop (see this method's own doc
                // comment on the `is_slot_only`-promotion precedent this
                // mirrors).
                instrs_mark_entry_point(&mut self.instrs, entry_word);
                self.covered.push(entry_word);
                any_covered = true;
                min_visited = min_visited.min(budget.min);
                max_visited = max_visited.max(budget.max);
            }
        }
        if any_covered {
            // `compute_cycles_flush` only needs *a* valid re-entry word — it
            // checks `instrs[t].is_entry_point` for every other entry, so a
            // single representative (the first covered one) is enough to
            // drive its `entry_word`-specific "never flush on its own row"
            // rule (see that function's doc comment); every other entry
            // point is covered by the `is_entry_point` check the same as any
            // branch target.
            compute_cycles_flush(&mut self.instrs, self.covered[0], min_visited, max_visited);
        }
        self.has_fpu = instrs_linear(&self.instrs).any(|i| is_fpu_instruction(i.raw));
        &self.instrs
    }
}

/// Remaining instruction-visit budget for a bounded walk (`Analyzer::walk_bounded`),
/// plus the min/max word offsets visited so far. Threaded through
/// `visit`/`visit_slot` by `&mut` alongside `instrs`/`page` — `min`/`max`
/// piggyback on the same threading `remaining` already needs, rather than a
/// separate parameter, since every "mark visited" site already has a
/// `&mut Budget` in scope. Used after the walk completes
/// (`Analyzer::walk_bounded`) to bound `compute_cycles_flush`'s post-pass to
/// `min..=max` instead of scanning the full page.
struct Budget {
    remaining: usize,
    min: WordOffset,
    max: WordOffset,
}

impl Budget {
    fn new(remaining: usize, entry_word: WordOffset) -> Self {
        Self { remaining, min: entry_word, max: entry_word }
    }

    /// Record `offset` as freshly marked visited — call from every site that
    /// sets `instrs[offset].visited = true` for the first time.
    fn mark_visited(&mut self, offset: WordOffset) {
        self.min = self.min.min(offset);
        self.max = self.max.max(offset);
    }
}

/// Collect every visited instruction from a walked buffer, ascending by word
/// offset — the shape codegen wants to hand to Cranelift (one IR block per
/// instruction, emitted in address order so forward branches can be patched
/// once their target block exists). The buffer is already stored in offset
/// order, so this is just a filter — no separate visit-order log needed.
pub fn instrs_linear(instrs: &[CompiledInstr; ENTRIES_PER_PAGE]) -> impl Iterator<Item = &CompiledInstr> {
    instrs.iter().filter(|i| i.visited)
}

/// Flip `offset`'s `is_entry_point` flag — [`Analyzer::walk_multi_entry`]'s
/// own post-visit marking step, factored out as a free function so it stays
/// obviously separate from `visit`'s own field-construction sites (this never
/// runs *during* a walk, only after one entry's walk has fully committed its
/// edges).
fn instrs_mark_entry_point(instrs: &mut [CompiledInstr; ENTRIES_PER_PAGE], offset: WordOffset) {
    instrs[offset as usize].is_entry_point = true;
}

/// Mark `offset` visited as a **delay slot** — never charged against
/// `budget` (see doc below) and never gated by it either.
///
/// Real branch-in-delay-slot is "unusual but legal" on real hardware and
/// already supported by the interpreter (`mips_exec.rs`'s `branch_delay`
/// nests) — so if the slot's own raw bits decode as a branch/jump/regjump,
/// *that* instruction's own mandatory slot must be walked too (recursively —
/// a chain of nested delay-slot branches needs a chain of nested inline
/// slots, exactly mirroring however deep the interpreter's own nesting
/// goes). This is distinct from ordinary reachability recursion ([`visit`]'s
/// fallthrough/taken arms): a slot's *own* target (if it's a branch/jump) is
/// never walked here — mid-chain slots are inlined unconditionally
/// (§6.1.4), so only the *next* slot in the chain matters, not where any of
/// them branch to. `visit`'s ordinary target/fallthrough recursion still
/// applies if this offset is *also* independently reached as a genuine jump
/// target from elsewhere in the region (§6.1.4 dual semantics) — this
/// function only ever walks the slot-chain role.
///
/// **Not charged against `budget`**: a delay slot can never be omitted
/// while its branch is compiled (§6.1.4) — it was never a truncation
/// candidate, so `max_instrs` counts head instructions only (a branch/jump
/// plus its mandatory slot-chain no longer competes with the caller's
/// budget for compiling more of the region). A slot chain still has a hard
/// stop of its own — running off the page, hitting an excluded instruction,
/// or the 0xFFC hazard — any of which declines the whole walk exactly like
/// before, just never because the *head* budget happened to be spent.
///
/// Returns `false` (never visited) iff `offset` (or any offset in a nested
/// slot-chain) is excluded or off-page — a slot (or slot-chain) that can't
/// complete disqualifies the outermost branch exactly like an excluded slot
/// always did.
fn visit_slot(instrs: &mut [CompiledInstr; ENTRIES_PER_PAGE], page: &[u32; ENTRIES_PER_PAGE], page_base: u32, offset: WordOffset, budget: &mut Budget) -> bool {
    if offset >= WORDS_PER_PAGE {
        return false;
    }
    if instrs[offset as usize].visited {
        // Already visited, whether as a real head (a promoted word, or one
        // walked as a genuine branch/jump target elsewhere in the region)
        // or as another branch's slot — either way its raw bytes and
        // semantics are already correctly captured; this occurrence just
        // shares the same word, no new bookkeeping needed. Does *not*
        // touch `is_slot_only` — a word that's a real head stays a real
        // head; a word that's still slot-only stays slot-only until
        // `visit` promotes it.
        return true;
    }
    let raw = page[offset as usize];
    let class = classify(raw, offset, page_base);
    if class == Classify::Excluded || class == Classify::RegionBoundary {
        // Excluded: an excluded instruction can never be a delay slot (a
        // fallback runs it via the interpreter, which needs it in head
        // position, not inlined); the branch is declined. RegionBoundary: a
        // hard end, same treatment.
        return false;
    }

    // 0xFFC applies to a nested slot exactly like a head: a branch/jump/
    // regjump landing here has its own mandatory slot on the next
    // (unreachable) page — it stays inlined into its own predecessor
    // (§6.1.4, nested delay-slot chain), but its own chain stops here.
    // codegen's emit_nested_branch_slot/emit_nested_regjump_slot never read
    // taken_exit/fallthrough_exit off a nested slot (every edge is always a
    // fresh runtime address computation, never a block/bail lookup), so this
    // is recorded only for introspection symmetry with the head case, not
    // because any codegen path consumes it.
    if offset == OFFSET_0XFFC_WORD && !matches!(class, Classify::Sequential) {
        instrs[offset as usize] = CompiledInstr {
            visited: true, word: offset, raw, block_id: None,
            fallthrough_exit: None, taken_exit: Some(StopReason::ForeignPageSlot),
            is_slot_only: true, is_fallback: false, is_branch_fallback_successor: false,
            is_branch_target: false, continues_to_fallthrough: None, continues_to_taken: None, has_inline_slot: false, cycles_delta: 1, cycles_flush: false, is_entry_point: false,
        };
        budget.mark_visited(offset);
        return true;
    }

    // Nested delay slot: this slot's own raw bits are themselves a
    // branch/jump/regjump, which mandates its own inline slot one word
    // further — walk it the same way, recursively, before this slot can be
    // considered complete. A Sequential slot (the overwhelming common case)
    // has no further chain and stops here immediately.
    if !matches!(class, Classify::Sequential) {
        let next_slot = offset + 1;
        if !visit_slot(instrs, page, page_base, next_slot, budget) {
            return false;
        }
    }

    instrs[offset as usize] = CompiledInstr {
        visited: true, word: offset, raw, block_id: None,
        fallthrough_exit: None, taken_exit: None, is_slot_only: true, is_fallback: false, is_branch_fallback_successor: false,
        is_branch_target: false, continues_to_fallthrough: None, continues_to_taken: None, has_inline_slot: false, cycles_delta: 1, cycles_flush: false, is_entry_point: false,
    };
    budget.mark_visited(offset);
    true
}

/// Visit `offset`, recursing into whatever it branches/falls through to.
/// Returns `true` if `offset` (and, for a branch/jump, its mandatory delay
/// slot — §6.1.4's "slot instance is emitted inline... an indivisible unit",
/// walked via [`visit_slot`], not recursively as a branch/jump itself) was
/// successfully entered into the compiled region; `false` if `offset` itself
/// is excluded, or is a branch/jump whose delay slot is excluded — in either
/// case nothing at `offset` is marked visited, and the caller is responsible
/// for recording `StopReason::Excluded` on *itself* instead of treating this
/// as a normal fall-through/branch/jump.
///
/// No-ops (returns `true` immediately) if already visited as a **head**
/// (`is_slot_only` is `false`) — this is what lets two converging paths (a
/// loop back-edge, two branches into the same target) terminate instead of
/// re-walking. But if `offset` was only ever visited as someone's delay slot
/// so far (`is_slot_only` is `true` — §6.1.4 dual semantics: a word can be
/// both a slot instance, inlined into its predecessor, *and* a genuine
/// jump/branch target with its own real edges), this call promotes it:
/// computes its real fallthrough/taken edges the same as a fresh visit,
/// without re-decrementing `budget` (already charged, if at all — see
/// `visit_slot`) or re-walking its own slot (`visit_slot` already did, and a
/// slot is never re-classified as a branch/jump for the *target* role, only
/// re-used for edge computation — same raw bits, same `class`). A word can
/// only ever be promoted once; after promotion `is_slot_only` is cleared, so
/// a later `visit` call sees a real head and no-ops normally.
///
/// Free function (not a method) so it can recurse without re-borrowing
/// `Analyzer` — `instrs`/`page` are disjoint borrows for the whole walk.
/// `budget` bounds total *head* instructions visited across the whole walk
/// (`Analyzer::walk_bounded`); `Analyzer::walk` passes `usize::MAX`, making
/// this unconditionally true in practice for the real compiler's unbounded
/// walk.
fn visit(instrs: &mut [CompiledInstr; ENTRIES_PER_PAGE], page: &[u32; ENTRIES_PER_PAGE], page_base: u32, offset: WordOffset, budget: &mut Budget) -> bool {
    if offset >= WORDS_PER_PAGE {
        return false; // ran off the page; shouldn't happen given the stop conditions below
    }
    let already_visited = instrs[offset as usize].visited;
    if already_visited && !instrs[offset as usize].is_slot_only {
        return true; // real head already walked — ordinary loop-termination case
    }
    if !already_visited && budget.remaining == 0 {
        return false; // caller records StopReason::Truncated on itself
    }

    let raw = page[offset as usize];
    let class = classify(raw, offset, page_base);

    if class == Classify::RegionBoundary {
        // Hard region end (test/tooling sentinel): never visited, never a
        // fallback — the caller records the boundary on its own edge, exactly
        // as it did for every Excluded word before interpreter-fallback existed.
        return false;
    }

    if class == Classify::Excluded && !fallback_enabled() {
        // Fallback disabled (the default): an excluded instruction ends the
        // region here, exactly as it always did — never visited, the caller
        // records StopReason::Excluded on its own edge.
        return false;
    }

    if class == Classify::Excluded {
        // Interpreter-fallback head: keep the excluded instruction *in* the
        // region rather than ending it here (the pre-fallback behavior was
        // `return false`, making the caller record StopReason::Excluded on
        // itself). It behaves exactly like a `Sequential` for reachability —
        // one fall-through edge to offset+1 — but is tagged `is_fallback` so
        // codegen emits the interpreter-fallback path instead of native
        // semantics. Charged against the budget like any other head. Promotion
        // (`already_visited`) can never land here: an excluded word reached
        // only as a slot was never marked visited (`visit_slot` returns false
        // and declines the branch), so there's nothing to promote.
        debug_assert!(!already_visited, "an excluded word cannot have been visited as a slot (visit_slot declines it)");
        instrs[offset as usize] = CompiledInstr {
            visited: true, word: offset, raw, block_id: None,
            fallthrough_exit: None, taken_exit: None, is_slot_only: false,
            is_fallback: true, is_branch_fallback_successor: false,
            is_branch_target: false, continues_to_fallthrough: None, continues_to_taken: None, has_inline_slot: false, cycles_delta: 1, cycles_flush: false, is_entry_point: false,
        };
        budget.remaining -= 1;
        budget.mark_visited(offset);
        // Same fall-through recursion as Sequential (finish_visit's Sequential
        // arm), inlined here so finish_visit's `class` stays a real
        // (non-Excluded) Classify and its `unreachable!` on Excluded holds.
        let successor_in_region = visit(instrs, page, page_base, offset + 1, budget);
        if !successor_in_region {
            let reason = if budget.remaining == 0 { StopReason::Truncated } else { StopReason::Excluded };
            instrs[offset as usize].fallthrough_exit = Some(reason);
        } else {
            instrs[offset as usize].continues_to_fallthrough = Some(offset + 1);
        }
        if successor_in_region && is_fallback_branch(raw) {
            // A branch fallback (BC1) arms a delay slot when run: its successor
            // is that slot and, on a taken/not-taken-non-likely arm, is reached
            // with core.in_delay_slot=true + a pending delay_slot_target. Mark
            // it so codegen gives it the entry-word foreign-slot treatment
            // (honor the pending transfer) instead of a plain fallthrough.
            // Note: OR into the flag — a word reached both this way AND some
            // other way stays flagged, and codegen's entry-word check is a
            // superset that's correct for a plain arrival too (in_delay_slot
            // false -> falls through normally).
            instrs[(offset + 1) as usize].is_branch_fallback_successor = true;
        }
        return true;
    }

    if already_visited {
        // Promotion (§6.1.4 dual semantics): `visit_slot` already walked
        // this word's own slot-chain (if it has one) and already validated
        // it's not excluded/0xFFC-hazarded — none of that needs redoing.
        // Only the edge computation below (the `match class` block) is new
        // for this word. Not charged against `budget`: this word's slot
        // role was never charged either (`visit_slot`), and it's still the
        // *same* physical instruction being counted once, not twice, when
        // both roles happen to land on it.
        instrs[offset as usize].is_slot_only = false;
        return finish_visit(instrs, page, page_base, offset, class, budget);
    }

    // 0xFFC: any branch/jump/regjump sitting at the last word of the page has
    // its mandatory delay slot on the next (unwalkable) page — there is no
    // `instrs[offset+1]` to inline via visit_slot. This no longer excludes
    // the word from compilation (see ForeignPageSlot): codegen materializes
    // core.in_delay_slot/delay_slot_target itself instead of inlining the
    // slot, exactly what the interpreter's own branch_delay/
    // handle_branch_not_taken do — the entry-side counterpart
    // (`entry_offset == 0` in exec_decoded's dispatch gate) already consumes
    // that state correctly for a page arriving with a foreign slot pending,
    // closing the loop. So: skip the slot walk entirely for this offset, and
    // finish_visit (below) is taught to force both edges to
    // StopReason::ForeignPageSlot without attempting any on-page target
    // arithmetic — branch_target's `offset_word + 1 + imm16` is relative to
    // the delay slot's own address, which for offset 0xFFC is on the *next*
    // page, so a small/negative imm16 could otherwise land back in this
    // page's `0..1024` range and be misread as an on-page target when it
    // actually names a next-page address (position-independence would also
    // break, same rationale as J/JAL always being page-leaving regardless of
    // the encoded absolute target).
    let is_0xffc_branch = offset == OFFSET_0XFFC_WORD && !matches!(class, Classify::Sequential);

    // A branch/jump's delay slot is indivisible from it (§6.1.4): walk it
    // first, atomically (visit_slot, not visit — the slot is never itself
    // recursed into as a branch/jump target, only as a slot — see
    // visit_slot's doc comment for the nested-branch-in-slot case it does
    // handle). If the slot (or its own nested slot-chain) comes back
    // excluded or runs off the page, the branch/jump can't be compiled
    // either — neither gets marked visited. Not charged against `budget` —
    // a delay slot was never a truncation candidate (§6.1.4). Skipped
    // entirely for a 0xFFC branch/jump/regjump — there is no slot to walk.
    let slot = offset + 1;
    if !is_0xffc_branch && !matches!(class, Classify::Sequential) && !visit_slot(instrs, page, page_base, slot, budget) {
        return false;
    }

    // This word has a real, on-page inline slot iff it's a Branch/Jump/
    // RegJump AND not the 0xFFC foreign-slot case (whose mandatory slot is
    // on the next, unwalkable page — nothing at `instrs[offset+1]` on this
    // page belongs to it). See `has_inline_slot`'s doc comment for why
    // `compute_cycles_flush` needs this.
    let has_inline_slot = !is_0xffc_branch && !matches!(class, Classify::Sequential);

    instrs[offset as usize] = CompiledInstr {
        visited: true, word: offset, raw, block_id: None,
        fallthrough_exit: None, taken_exit: None, is_slot_only: false, is_fallback: false, is_branch_fallback_successor: false,
        is_branch_target: false, continues_to_fallthrough: None, continues_to_taken: None, has_inline_slot, cycles_delta: 1, cycles_flush: false, is_entry_point: false,
    };
    budget.remaining -= 1;
    budget.mark_visited(offset);

    if is_0xffc_branch {
        return finish_visit_foreign_page_slot(instrs, offset, class);
    }
    finish_visit(instrs, page, page_base, offset, class, budget)
}

/// Shared edge-computation tail for [`visit`]'s fresh-visit and promotion
/// paths — both have already ensured `instrs[offset]` is marked visited
/// (with the right `raw`) and any mandatory slot-chain is walked; this just
/// computes `fallthrough_exit`/`taken_exit` from `class`. Always returns
/// `true` (matching `visit`'s "entered into the region" contract — by the
/// time this runs, `offset` itself is unconditionally part of the region).
fn finish_visit(instrs: &mut [CompiledInstr; ENTRIES_PER_PAGE], page: &[u32; ENTRIES_PER_PAGE], page_base: u32, offset: WordOffset, class: Classify, budget: &mut Budget) -> bool {
    match class {
        Classify::Sequential => {
            // Sequential's only edge is the fall-through — if it exits,
            // that's this instruction's fallthrough_exit, mirroring exactly
            // what a Branch's not-taken arm means. Budget exhaustion takes
            // priority over Excluded in the reported reason only in the
            // sense that visit() itself distinguishes them via budget check
            // ordering above (budget checked before exclusion) — so a
            // truncated edge reads Truncated, not Excluded.
            if !visit(instrs, page, page_base, offset + 1, budget) {
                let reason = if budget.remaining == 0 { StopReason::Truncated } else { StopReason::Excluded };
                instrs[offset as usize].fallthrough_exit = Some(reason);
            } else {
                instrs[offset as usize].continues_to_fallthrough = Some(offset + 1);
            }
        }
        Classify::Excluded | Classify::RegionBoundary => unreachable!("handled above"),
        Classify::RegJump => instrs[offset as usize].taken_exit = Some(StopReason::RegJump),
        Classify::Branch { target } => {
            // Slot already walked above (excluded-free, checked before this
            // instruction was marked visited). Not-taken arm falls through
            // past the slot to offset+2 — recorded on fallthrough_exit if it
            // exits. Taken arm (if in-page) recurses via visit with full
            // branch/jump semantics — the dual-semantics case (§6.1.4): if
            // `t` also happens to be someone's delay slot reached via
            // visit_slot elsewhere, that's fine, both paths agree on
            // whether it's excluded. Both edges are independent — a branch
            // can have one continue and the other exit, or both exit, or
            // neither.
            if !visit(instrs, page, page_base, offset + 2, budget) {
                let reason = if budget.remaining == 0 { StopReason::Truncated } else { StopReason::Excluded };
                instrs[offset as usize].fallthrough_exit = Some(reason);
            } else {
                instrs[offset as usize].continues_to_fallthrough = Some(offset + 2);
            }
            match target {
                Some(t) => {
                    if visit(instrs, page, page_base, t, budget) {
                        // Set after visit() returns, never before: a fresh
                        // first-visit for `t` (Classify::Excluded's own
                        // CompiledInstr construction, or the plain-head
                        // construction below) fully overwrites
                        // instrs[t], which would silently clobber this
                        // flag if it were set beforehand.
                        instrs[t as usize].is_branch_target = true;
                        instrs[offset as usize].continues_to_taken = Some(t);
                    } else {
                        let reason = if budget.remaining == 0 { StopReason::Truncated } else { StopReason::Excluded };
                        instrs[offset as usize].taken_exit = Some(reason);
                    }
                }
                None => instrs[offset as usize].taken_exit = Some(StopReason::PageLeaving),
            }
        }
        Classify::Jump { target } => {
            // Unconditional — no not-taken arm, only the taken_exit edge.
            match target {
                Some(t) => {
                    if visit(instrs, page, page_base, t, budget) {
                        // See the Branch arm above for why this is set
                        // after visit() returns, not before.
                        instrs[t as usize].is_branch_target = true;
                        instrs[offset as usize].continues_to_taken = Some(t);
                    } else {
                        let reason = if budget.remaining == 0 { StopReason::Truncated } else { StopReason::Excluded };
                        instrs[offset as usize].taken_exit = Some(reason);
                    }
                }
                None => instrs[offset as usize].taken_exit = Some(StopReason::PageLeaving),
            }
        }
    }
    true
}

/// `finish_visit`'s counterpart for a branch/jump/regjump sitting at offset
/// 0xFFC (`visit`'s `is_0xffc_branch`): every edge this instruction has is
/// forced to `StopReason::ForeignPageSlot`, with no on-page target/
/// fallthrough resolution attempted at all — `branch_target`'s
/// `offset_word + 1 + imm16` arithmetic is relative to the delay slot's own
/// address, which for this offset is on the *next* page, so letting it run
/// here could misclassify a next-page destination as landing back in this
/// page's `0..1024` range (see `visit`'s comment on `is_0xffc_branch`).
/// `RegJump`/`Jump{None}` already have no `Classify`-level target to
/// resolve, so those two arms just confirm that; `Branch`'s two edges (and
/// the annulling-vs-not distinction) are entirely a codegen-time concern
/// here — both get the same `ForeignPageSlot` reason regardless of `cond`
/// or `annul`, since the analyzer doesn't track those distinctions (unlike
/// `classify`'s `Branch`, which has already collapsed them into a target-only
/// question by the time this runs).
fn finish_visit_foreign_page_slot(instrs: &mut [CompiledInstr; ENTRIES_PER_PAGE], offset: WordOffset, class: Classify) -> bool {
    match class {
        Classify::Branch { .. } => {
            instrs[offset as usize].fallthrough_exit = Some(StopReason::ForeignPageSlot);
            instrs[offset as usize].taken_exit = Some(StopReason::ForeignPageSlot);
        }
        Classify::Jump { .. } | Classify::RegJump => {
            instrs[offset as usize].taken_exit = Some(StopReason::ForeignPageSlot);
        }
        Classify::Sequential | Classify::Excluded | Classify::RegionBoundary => unreachable!("is_0xffc_branch guarantees a branch/jump/regjump class"),
    }
    true
}

/// Second pass over an already-`visit`ed buffer: fills in `cycles_flush` for
/// every visited word now that every `is_branch_target` bit is final (a
/// word can be visited before some *later* branch in the walk marks it as a
/// target, so `cycles_flush` can't be decided inline during
/// `visit`/`finish_visit` the way `cycles_delta` — a pure function of the
/// word's own class — can be). Called once by `Analyzer::walk_bounded`
/// right after `visit` returns, bounded to `min_visited..=max_visited`
/// (tracked incrementally during the walk via `Budget::mark_visited`) so it
/// never scans more of the 1024-entry page than the walk actually touched.
///
/// A linear word-offset-ascending scan is sufficient here — no need to
/// re-walk edges in chain order — because every field this reads
/// (`is_region_exit()`, `continues_to_fallthrough`/`continues_to_taken`,
/// `is_branch_target`) is already final for the whole buffer by the time
/// this runs; `cycles_flush` is a pure per-instruction/per-edge fact, not
/// an accumulated distance, so there is no ordering dependency between
/// words to respect.
///
/// Sets `cycles_flush = true` for:
/// - every region-exit word (`is_region_exit()`) — cycles must be current
///   before control leaves the compiled unit, unconditionally;
/// - every word whose continuing fallthrough or taken edge lands on an
///   `is_branch_target` word, or on `entry_word` itself — a loop
///   re-entering there (whether an ordinary in-region back-edge, or the
///   case where the region's own entry word doubles as the loop head)
///   must have cycles fresh on arrival, so a busy-wait spinning on another
///   thread (see `Hot::cycles`'s doc comment) keeps observing it advance
///   even if this compiled unit never otherwise exits.
///
/// `entry_word` deliberately never gets `cycles_flush` set on its *own*
/// row for this reason — the flush obligation lands on whichever
/// predecessor's edge targets it, exactly like any other branch-target
/// arrival; entry_word's block is shared between external dispatch
/// (cycles already current, nothing pending) and internal back-edge
/// arrival (predecessor already flushed before jumping in), so both modes
/// see a body that starts counting fresh from zero pending, uniformly.
fn compute_cycles_flush(instrs: &mut [CompiledInstr; ENTRIES_PER_PAGE], entry_word: WordOffset, min_visited: WordOffset, max_visited: WordOffset) {
    for word in min_visited..=max_visited {
        if !instrs[word as usize].visited {
            continue;
        }
        let mut flush = instrs[word as usize].is_region_exit();
        if let Some(t) = instrs[word as usize].continues_to_fallthrough {
            flush |= instrs[t as usize].is_branch_target || instrs[t as usize].is_entry_point || t == entry_word;
        }
        if let Some(t) = instrs[word as usize].continues_to_taken {
            flush |= instrs[t as usize].is_branch_target || instrs[t as usize].is_entry_point || t == entry_word;
        }
        if !flush {
            continue;
        }
        // A word with a real inline slot (`has_inline_slot`) is never the
        // last one to retire for its own edge — codegen always emits its
        // slot's `emit_account_for_cycles` call strictly after the head's
        // (`emit_slot_semantics`, invoked from within whichever
        // branch/jump/regjump emitter processes this word). Deciding the
        // flush *here*, on the head, would fire before the slot's own
        // pending contribution ever accrues, silently losing it — push the
        // obligation onto the slot word (`word + 1`, always the real
        // on-page slot per `has_inline_slot`'s doc comment) instead; the
        // head keeps `cycles_flush = false` and just contributes its
        // `cycles_delta` like any other batched word.
        if instrs[word as usize].has_inline_slot {
            instrs[(word + 1) as usize].cycles_flush = true;
        } else {
            instrs[word as usize].cycles_flush = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mips_isa::*;

    /// Delegates to the crate-wide [`super::test_fallback_guard`] so this
    /// module's fallback tests share the one lock with every other module's.
    fn fallback_on_guard() -> super::TestFallbackGuard {
        super::test_fallback_guard()
    }

    fn r_type(op: u32, rs: u32, rt: u32, rd: u32, sa: u32, funct: u32) -> u32 {
        (op << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct
    }
    fn i_type(op: u32, rs: u32, rt: u32, imm: u16) -> u32 {
        (op << 26) | (rs << 21) | (rt << 16) | imm as u32
    }

    #[test]
    fn classify_sequential_nop() {
        assert_eq!(classify(0, 0, 0), Classify::Sequential);
    }

    #[test]
    fn classify_jr_is_regjump() {
        let instr = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR);
        assert_eq!(classify(instr, 5, 0), Classify::RegJump);
    }

    #[test]
    fn classify_excluded_mtc0() {
        let instr = r_type(OP_COP0, RS_MTC0, 0, 12, 0, 0);
        assert_eq!(classify(instr, 5, 0), Classify::Excluded);
    }

    #[test]
    fn classify_excluded_cache() {
        let instr = i_type(OP_CACHE, 0, 0, 0);
        assert_eq!(classify(instr, 5, 0), Classify::Excluded);
    }

    #[test]
    fn classify_plain_fpu_arithmetic_is_sequential() {
        // FADD.D
        let instr = r_type(OP_COP1, RS_D, 3, 4, 5, FUNCT_FADD);
        assert_eq!(classify(instr, 5, 0), Classify::Sequential);
    }

    #[test]
    fn classify_fpu_register_moves_are_sequential() {
        let mfc1 = r_type(OP_COP1, RS_MFC1, 2, 3, 0, 0);
        assert_eq!(classify(mfc1, 5, 0), Classify::Sequential);
        let mtc1 = r_type(OP_COP1, RS_MTC1, 2, 3, 0, 0);
        assert_eq!(classify(mtc1, 5, 0), Classify::Sequential);
    }

    #[test]
    fn classify_fpu_compare_is_sequential() {
        // C.EQ.D
        let instr = r_type(OP_COP1, RS_D, 5, 4, 0, FUNCT_FC_EQ);
        assert_eq!(classify(instr, 5, 0), Classify::Sequential);
    }

    #[test]
    fn classify_fpu_load_store_is_sequential() {
        // OP_LWC1/OP_SDC1 are plain memory ops architecturally (no
        // unresolved control flow) and now have real emitters, routed
        // through lookup_cp1_semantics specifically so the region-wide
        // CU1/FR guard's trigger check (has_fpu) still catches them — see
        // codegen::lookup_cp1_semantics's doc comment. sequential_or_excluded
        // correctly reports Sequential now that the emitters exist.
        assert_eq!(classify(i_type(OP_LWC1, 0, 0, 0), 5, 0), Classify::Sequential);
        assert_eq!(classify(i_type(OP_SDC1, 0, 0, 0), 5, 0), Classify::Sequential);
    }

    #[test]
    fn classify_cop1x_madd_is_excluded_until_an_emitter_exists() {
        // OP_COP1X has no emitters at all yet (lookup_cp1_semantics only
        // matches op == OP_COP1) — sequential_or_excluded correctly reports
        // Excluded here too, not Sequential.
        let instr = r_type(OP_COP1X, 1, 2, 3, 4, FUNCT_MADD_S);
        assert_eq!(classify(instr, 5, 0), Classify::Excluded);
    }

    #[test]
    fn classify_cop1x_with_rs_equal_to_bc1_encoding_is_not_treated_as_a_branch() {
        // Regression: OP_COP1X's `rs` field is a base register (e.g.
        // LWXC1's indexed-load base), not a format/branch selector like
        // OP_COP1's — rs == RS_BC1's numeric value (0x08) here must not be
        // misread as a BC1 conditional branch. Currently Excluded (no
        // COP1X emitter exists yet) rather than Sequential, but critically
        // NOT via the Excluded-as-BC1-branch path — once a COP1X emitter
        // lands this must become Sequential, never get stuck as a phantom
        // branch exclusion.
        let instr = r_type(OP_COP1X, RS_BC1, 2, 3, 4, FUNCT_LWXC1);
        assert_eq!(classify(instr, 5, 0), Classify::Excluded);
    }

    #[test]
    fn classify_bc1_is_excluded() {
        // BC1F/BC1T — CP1 conditional branch, condition-code-dependent target.
        let instr = r_type(OP_COP1, RS_BC1, 0, 0, 0, 0);
        assert_eq!(classify(instr, 5, 0), Classify::Excluded);
    }

    #[test]
    fn classify_cop2_is_excluded() {
        let instr = r_type(OP_COP2, 0, 0, 0, 0, 0);
        assert_eq!(classify(instr, 5, 0), Classify::Excluded);
    }

    #[test]
    fn branch_in_page_target_resolves() {
        // BEQ at word 10, imm=+2 words -> target word offset = 10 + 1 + 2 = 13
        // (target is relative to the delay slot's address, word 11, not word 12).
        let instr = i_type(OP_BEQ, 1, 2, 2);
        assert_eq!(classify(instr, 10, 0), Classify::Branch { target: Some(13) });
    }

    #[test]
    fn branch_off_page_low_end() {
        // Backward branch far enough to leave the page (offset 0, big negative imm).
        let instr = i_type(OP_BEQ, 1, 2, 0xFF00u16); // large negative offset
        assert_eq!(classify(instr, 0, 0), Classify::Branch { target: None });
    }

    #[test]
    fn walk_straight_line_to_regjump() {
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = 0; // nop
        page[1] = 0; // nop
        page[2] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // jr ra
        page[3] = 0; // delay slot
        let mut a = Analyzer::new();
        let (result, non_empty) = a.walk(&page, 0, 0);
        assert!(non_empty);
        let v: Vec<_> = instrs_linear(result).collect();
        assert!(v.iter().any(|i| i.word == 0));
        assert!(v.iter().any(|i| i.word == 1));
        assert!(v.iter().any(|i| i.word == 2 && i.taken_exit == Some(StopReason::RegJump)));
    }

    #[test]
    fn walk_branch_reaches_both_arms() {
        let mut page = [0u32; ENTRIES_PER_PAGE];
        // word 0: BEQ r1,r2, +2 (target = 0+1+2 = 3, relative to the delay
        // slot's own address at word 1)
        page[0] = i_type(OP_BEQ, 1, 2, 2);
        page[1] = 0; // delay slot
        page[2] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // fall-through path ends here
        page[3] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // taken-branch path ends here
        let mut a = Analyzer::new();
        let (result, _) = a.walk(&page, 0, 0);
        let visited: Vec<u16> = result.iter().filter(|i| i.visited).map(|i| i.word).collect();
        for expect in [0u16, 1, 2, 3] {
            assert!(visited.contains(&expect), "expected offset {} reached", expect);
        }
    }

    #[test]
    fn walk_excluded_entry_is_a_one_instruction_fallback_region() {
        let _fb = fallback_on_guard();
        // Interpreter-fallback: an excluded entry is no longer an empty region.
        // It's a one-instruction region with the excluded word visited and
        // tagged `is_fallback` — codegen runs it via core.interp_fallback_fn.
        // (Pre-fallback this reported non_empty=false / nothing visited.)
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = r_type(OP_COP0, RS_MTC0, 0, 12, 0, 0);
        page[1] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // clean region end after the fallback
        let mut a = Analyzer::new();
        let (result, non_empty) = a.walk(&page, 0, 0);
        assert!(non_empty, "excluded entry is now a compilable fallback region");
        assert_eq!(instrs_linear(result).count(), 3, "fallback head + JR successor + JR's delay slot");
        assert!(result[0].visited, "the excluded entry is now visited as a fallback head");
        assert!(result[0].is_fallback, "excluded head must be tagged is_fallback");
    }

    #[test]
    fn walk_branch_taken_target_excluded_continues_into_fallback() {
        let _fb = fallback_on_guard();
        // word 0: BEQ r1,r2, +2 (target = 0+1+2 = 3). The taken target (word 3)
        // is excluded — with interpreter-fallback it's now visited as a
        // fallback head and the branch *continues into* it (taken_exit None),
        // rather than the branch recording StopReason::Excluded and exiting.
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = i_type(OP_BEQ, 1, 2, 2);
        page[1] = 0; // delay slot
        page[2] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // fall-through path ends here
        page[3] = r_type(OP_COP0, RS_MTC0, 0, 12, 0, 0); // taken target: excluded -> fallback
        let mut a = Analyzer::new();
        let (result, non_empty) = a.walk(&page, 0, 0);
        assert!(non_empty);
        let branch = result[0];
        assert!(branch.visited);
        assert_eq!(branch.taken_exit, None, "taken arm now continues into the fallback word, no exit");
        assert!(result[3].visited, "excluded taken target is now visited as a fallback head");
        assert!(result[3].is_fallback);
    }

    #[test]
    fn walk_jump_is_always_page_leaving() {
        // J/JAL is never resolved on-page, even when target26 happens to
        // encode a word offset that would fall on this same page — unlike
        // Branch, J/JAL's encoding is an absolute word position within a
        // fixed 256MB region, and resolving it against a specific page's
        // identity at analysis time would make the compiled function's own
        // shape depend on which physical page happens to be compiling,
        // breaking position independence (see jump_target's doc comment).
        // Same treatment as RegJump (JR/JALR): always taken_exit =
        // PageLeaving, regardless of what instruction (if any) a "would-be"
        // on-page interpretation of target26 would land on — word 2 here is
        // deliberately excluded (COP0) specifically to prove it's never even
        // looked at.
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = crate::mips_isa::OP_J << 26 | 2; // J, target26=2 (would be word 2 if resolved on-page)
        page[1] = 0; // delay slot
        page[2] = r_type(OP_COP0, RS_MTC0, 0, 12, 0, 0); // never visited: J never resolves here
        let mut a = Analyzer::new();
        let (result, non_empty) = a.walk(&page, 0, 0);
        assert!(non_empty);
        let jump = result[0];
        assert!(jump.visited);
        assert_eq!(jump.taken_exit, Some(StopReason::PageLeaving), "J must always be page-leaving, never resolved on-page");
        assert!(!result[2].visited, "J's target must never be visited at analysis time — it's not resolved until runtime");
    }

    #[test]
    fn walk_excluded_successor_is_a_fallback_head_in_region() {
        let _fb = fallback_on_guard();
        // word 0 is a plain sequential instruction whose fall-through (word 1)
        // is excluded. With interpreter-fallback the region no longer ends at
        // word 0: word 1 is visited as a fallback head, so word 0's fallthrough
        // continues into it (no exit). (Word 1's own fall-through to word 2, a
        // nop, continues on — the region keeps growing normally.)
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = 0; // nop, falls through to word 1
        page[1] = r_type(OP_COP0, RS_MTC0, 0, 12, 0, 0); // excluded -> fallback
        page[2] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // clean region end
        let mut a = Analyzer::new();
        let (result, non_empty) = a.walk(&page, 0, 0);
        assert!(non_empty);
        assert_eq!(result[0].fallthrough_exit, None, "word 0 now continues into the fallback word");
        assert!(result[1].visited, "the excluded successor is now visited as a fallback head");
        assert!(result[1].is_fallback);
        assert!(!result[1].is_slot_only);
    }

    #[test]
    fn walk_excluded_delay_slot_disqualifies_the_branch_too() {
        // A branch whose delay slot is excluded can't be compiled either
        // (§6.1.4: slot is an indivisible part of the branch unit) -- the
        // instruction that led into the branch must record the exit instead,
        // and neither the branch nor its slot may be visited.
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = 0; // nop, falls through to word 1
        page[1] = i_type(OP_BEQ, 1, 2, 3); // branch at word 1, target = 1+1+3 = 5
        page[2] = r_type(OP_COP0, RS_MTC0, 0, 12, 0, 0); // delay slot: excluded
        let mut a = Analyzer::new();
        let (result, non_empty) = a.walk(&page, 0, 0);
        assert!(non_empty);
        let v: Vec<_> = instrs_linear(result).collect();
        assert_eq!(v.len(), 1, "only word 0 should be in the region");
        assert_eq!(v[0].word, 0);
        assert_eq!(v[0].fallthrough_exit, Some(StopReason::Excluded));
        assert!(!result[1].visited, "branch with an excluded delay slot must not be visited");
        assert!(!result[2].visited);
    }

    #[test]
    fn walk_branch_not_taken_arm_excluded_continues_into_fallback() {
        let _fb = fallback_on_guard();
        // BEQ's not-taken side (fall-through past the delay slot, word 2) is
        // excluded. With interpreter-fallback the not-taken arm continues into
        // word 2 as a fallback head (no exit) instead of the branch becoming a
        // region boundary. The taken arm (word 3) is a good in-page target as
        // before. NOTE: word 2's fall-through target is word 3 (the JR), so the
        // two arms converge — fine.
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = i_type(OP_BEQ, 1, 2, 2); // target = 0+1+2 = 3
        page[1] = 0; // delay slot
        page[2] = r_type(OP_COP0, RS_MTC0, 0, 12, 0, 0); // not-taken arm: excluded -> fallback
        page[3] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // taken arm / fallback's successor
        let mut a = Analyzer::new();
        let (result, non_empty) = a.walk(&page, 0, 0);
        assert!(non_empty);
        assert_eq!(result[0].fallthrough_exit, None, "not-taken arm now continues into the fallback word");
        assert_eq!(result[0].taken_exit, None, "taken arm continues into compiled code, no exit needed");
        assert!(result[1].visited, "delay slot is still compiled");
        assert!(result[2].visited, "excluded not-taken word is now visited as a fallback head");
        assert!(result[2].is_fallback);
        assert!(result[3].visited, "taken arm is still walked");
    }

    #[test]
    fn walk_entry_at_0xffc_is_accepted_as_foreign_page_slot() {
        // A branch/jump sitting at a page's last word has no room for its
        // mandatory delay slot on this page — codegen arms
        // core.in_delay_slot/core.delay_slot_target instead of inlining it
        // (mirroring the interpreter's branch_delay), so this is no longer
        // rejected: it's a normal visited head whose sole edge (RegJump has
        // no not-taken arm) is StopReason::ForeignPageSlot.
        let mut page = [0u32; ENTRIES_PER_PAGE];
        let last = (ENTRIES_PER_PAGE - 1) as u16;
        page[last as usize] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR);
        let mut a = Analyzer::new();
        let (result, non_empty) = a.walk(&page, last, 0);
        assert!(non_empty, "a branch/jump AT the 0xFFC word is now a normal compilable entry");
        assert!(result[last as usize].visited);
        assert_eq!(result[last as usize].taken_exit, Some(StopReason::ForeignPageSlot));
    }

    #[test]
    fn walk_reaches_0xffc_as_a_branch_target_and_visits_it_as_foreign_page_slot() {
        // Same 0xFFC word, reached as a branch's taken target instead of as
        // the walk's own entry — the branch leading into it compiles
        // normally (word 0's BEQ, delay slot at word 1), and the target word
        // itself is now visited too (promoted, same as any other in-region
        // branch target), with its own sole edge recorded as
        // ForeignPageSlot rather than pulling in a slot that isn't there.
        let mut page = [0u32; ENTRIES_PER_PAGE];
        let last = (ENTRIES_PER_PAGE - 1) as u16;
        let target = last as i32;
        let imm16 = (target - (0 + 1)) as i16 as u16;
        page[0] = i_type(OP_BEQ, 1, 2, imm16);
        page[1] = 0; // delay slot
        page[last as usize] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR);
        let mut a = Analyzer::new();
        let (result, non_empty) = a.walk(&page, 0, 0);
        assert!(non_empty);
        assert_eq!(result[0].taken_exit, None, "taken arm continues into compiled code, no exit needed");
        assert!(result[last as usize].visited, "the 0xFFC word is now visited as a branch target");
        assert_eq!(result[last as usize].taken_exit, Some(StopReason::ForeignPageSlot));
    }

    #[test]
    fn walk_does_not_revisit_converging_paths() {
        let mut page = [0u32; ENTRIES_PER_PAGE];
        // Two branches that both target word 4, which then JRs.
        page[0] = i_type(OP_BEQ, 1, 2, 3); // target = 0+1+3 = 4
        page[1] = 0;
        page[2] = i_type(OP_BNE, 1, 2, 1); // target = 2+1+1 = 4
        page[3] = 0;
        page[4] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR);
        let mut a = Analyzer::new();
        let (result, _) = a.walk(&page, 0, 0);
        // offset 4 appears exactly once in the buffer by construction (it's
        // an array indexed by offset) — the real assertion is that visiting
        // it twice didn't panic/misbehave and it's marked visited.
        assert!(result[4].visited);
        assert_eq!(result[4].word, 4);
    }

    #[test]
    fn linear_is_sorted_by_word_offset() {
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = i_type(OP_BEQ, 1, 2, 2); // target = 0+1+2 = 3
        page[1] = 0;
        page[2] = 0;
        page[3] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR);
        let mut a = Analyzer::new();
        let (result, _) = a.walk(&page, 0, 0);
        let words: Vec<u16> = instrs_linear(result).map(|i| i.word).collect();
        let mut sorted = words.clone();
        sorted.sort();
        assert_eq!(words, sorted, "instrs_linear must be in ascending word-offset order for codegen");
    }

    #[test]
    fn analyzer_buffer_is_reused_and_reset_across_calls() {
        let mut page_a = [0u32; ENTRIES_PER_PAGE];
        page_a[0] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR);
        let mut page_b = [0u32; ENTRIES_PER_PAGE];
        page_b[10] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR);

        let mut a = Analyzer::new();
        let (first, _) = a.walk(&page_a, 0, 0);
        assert!(first[0].visited);
        assert!(!first[10].visited);

        // Second call on a different page/entry must not leak state from the first.
        let (second, _) = a.walk(&page_b, 10, 0);
        assert!(!second[0].visited, "stale visited bit from previous walk leaked through");
        assert!(second[10].visited);
    }

    #[test]
    fn walk_bounded_stops_at_exactly_one_instruction() {
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = 0; // nop
        page[1] = 0; // nop, would continue if unbounded
        let mut a = Analyzer::new();
        let (result, non_empty) = a.walk_bounded(&page, 0, 0, 1);
        assert!(non_empty);
        let v: Vec<_> = instrs_linear(result).collect();
        assert_eq!(v.len(), 1, "budget of 1 must admit exactly one instruction");
        assert_eq!(v[0].word, 0);
        assert_eq!(v[0].fallthrough_exit, Some(StopReason::Truncated));
        assert!(!result[1].visited, "word past the budget must never be visited");
    }

    #[test]
    fn walk_bounded_budget_excludes_delay_slot() {
        // A branch/jump's mandatory delay slot is never charged against
        // `max_instrs` (only *head* instructions are — comp.rs's
        // MAX_INSTRS_PER_COMPILE=1 relies on this: "1" means one head, with
        // its slot-chain always included for free, however deep). With a
        // budget of exactly 1, the branch itself spends the whole budget,
        // its slot is still visited (never a truncation candidate), but the
        // branch's taken arm (a second head-shaped instruction) is cut off.
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = i_type(OP_BEQ, 1, 2, 2); // target = 0+1+2 = 3
        page[1] = 0; // delay slot
        page[3] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // taken arm, cut off by budget
        let mut a = Analyzer::new();
        let (result, non_empty) = a.walk_bounded(&page, 0, 0, 1);
        assert!(non_empty);
        let v: Vec<_> = instrs_linear(result).collect();
        assert_eq!(v.len(), 2, "branch + its free delay slot are visited, but nothing past the 1-head budget");
        assert!(result[1].visited, "delay slot must still be visited (indivisible from the branch)");
        assert!(!result[3].visited, "taken arm must not be visited once the head budget is spent");
        assert_eq!(result[0].taken_exit, Some(StopReason::Truncated));
        // Fallthrough here (offset 2, past the slot) is also cut off by the
        // same exhausted budget, and must likewise read Truncated, not
        // Excluded — offset 2 is a perfectly compilable nop, just unreached
        // because nothing is left to spend.
        assert_eq!(result[0].fallthrough_exit, Some(StopReason::Truncated));
    }

    #[test]
    fn walk_bounded_max_recovers_unbounded_behavior() {
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = 0;
        page[1] = 0;
        page[2] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR);
        page[3] = 0;
        let mut a = Analyzer::new();
        let (result, non_empty) = a.walk_bounded(&page, 0, 0, usize::MAX);
        assert!(non_empty);
        assert_eq!(instrs_linear(result).count(), 4);
    }

    #[test]
    fn walk_excluded_entry_bounded_is_a_one_instruction_fallback_region() {
        let _fb = fallback_on_guard();
        // Interpreter-fallback: an excluded entry is a fallback head charged
        // against the budget like any other head. A budget of 1 admits exactly
        // the one fallback instruction (mirrors
        // walk_excluded_entry_is_a_one_instruction_fallback_region, bounded).
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = r_type(OP_COP0, RS_MTC0, 0, 12, 0, 0);
        let mut a = Analyzer::new();
        let (result, non_empty) = a.walk_bounded(&page, 0, 0, 1);
        assert!(non_empty, "excluded entry is now a compilable fallback region");
        assert_eq!(instrs_linear(result).count(), 1);
        assert!(result[0].visited && result[0].is_fallback);
    }

    // --- cycles_delta / cycles_flush ---

    #[test]
    fn cycles_delta_is_always_one_for_every_visited_word() {
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = 0; // nop
        page[1] = 0; // nop
        page[2] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // jr ra
        page[3] = 0; // delay slot
        let mut a = Analyzer::new();
        let (result, _) = a.walk(&page, 0, 0);
        for i in instrs_linear(result) {
            assert_eq!(i.cycles_delta, 1, "word {} should contribute exactly 1", i.word);
        }
    }

    #[test]
    fn cycles_flush_straight_line_only_flushes_at_the_exit() {
        // A plain straight-line region with no internal branch target: only
        // the exit should flush. Every interior word batches into it. The
        // regjump's own flush is deliberately pushed onto its delay slot
        // (word 4), not the regjump itself (word 3) -- codegen emits the
        // slot's own emit_account_for_cycles call strictly after the
        // head's, so flushing at the head would fire before the slot's own
        // +1 ever accrues (see has_inline_slot's doc comment).
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = 0; // nop
        page[1] = 0; // nop
        page[2] = 0; // nop
        page[3] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // jr ra
        page[4] = 0; // delay slot
        let mut a = Analyzer::new();
        let (result, _) = a.walk(&page, 0, 0);
        assert!(!result[0].cycles_flush, "word 0 has no branch target/exit on its edge");
        assert!(!result[1].cycles_flush);
        assert!(!result[2].cycles_flush);
        assert!(!result[3].cycles_flush, "the regjump's flush is deferred to its own delay slot");
        assert!(result[4].cycles_flush, "the delay slot is the true last-to-retire word of the pair");
    }

    #[test]
    fn cycles_flush_forward_branch_target_flushes_the_branch_but_not_the_straight_run() {
        // word 0: BEQ r1,r2, +3 -> taken target = 0+1+3 = 4 (word 4, marked
        // is_branch_target). Not-taken (fallthrough) arm runs straight
        // through word 2 (nop) to word 3 (another nop) before reaching word
        // 4 too -- both paths converge on word 4, so both the branch's own
        // taken edge (word 0, pushed onto its own slot at word 1 --
        // has_inline_slot) AND the last fallthrough hop into it (word 3)
        // must flush; only the interior word 2, whose edge lands on
        // non-branch-target word 3, gets to batch. Word 4 (jr ra) is itself
        // a region exit, but that flush is likewise deferred to its own
        // slot (word 5).
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = i_type(OP_BEQ, 1, 2, 3);
        page[1] = 0; // delay slot -- branch's flush lands here (has_inline_slot)
        page[2] = 0; // not-taken arm: nop, falls through to word 3 (not a branch target)
        page[3] = 0; // another nop, falls through to word 4 (a branch target) -- must flush
        page[4] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // branch target AND fallthrough landing, then exits
        page[5] = 0; // delay slot -- word 4's own exit-flush lands here
        let mut a = Analyzer::new();
        let (result, _) = a.walk(&page, 0, 0);
        assert!(result[4].is_branch_target, "word 4 is BEQ's taken target");
        assert!(!result[0].cycles_flush, "branch's own flush is deferred to its slot (word 1)");
        assert!(result[1].cycles_flush, "branch's taken edge lands on a branch-target word, flush pushed to the slot");
        assert!(!result[2].cycles_flush, "word 2's edge lands on word 3, which is not a branch target");
        assert!(result[3].cycles_flush, "word 3's fallthrough edge lands on branch-target word 4");
        assert!(!result[4].cycles_flush, "word 4's own exit-flush is deferred to its slot (word 5)");
        assert!(result[5].cycles_flush, "word 4's delay slot is the true last-to-retire word for its exit");
    }

    #[test]
    fn cycles_flush_back_edge_loop_flushes_at_the_branch_targeting_the_loop_head() {
        // word 0: nop (loop head, also entry_word)
        // word 1: BEQ r1,r2, -2 -> target = 1+1+(-2) = 0 (back-edge to entry_word)
        // word 2: delay slot -- branch's flush (targeting entry_word) lands here
        // word 3: jr ra (not-taken arm falls through here and exits)
        // word 4: jr ra's own delay slot -- its exit-flush lands here
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = 0; // nop, entry_word and loop head
        page[1] = i_type(OP_BEQ, 1, 2, (-2i16) as u16);
        page[2] = 0; // delay slot
        page[3] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // not-taken arm exits here
        page[4] = 0; // delay slot
        let mut a = Analyzer::new();
        let (result, _) = a.walk(&page, 0, 0);
        assert_eq!(result[0].word, 0);
        assert!(!result[0].cycles_flush, "entry_word's own row is never independently a flush trigger");
        assert!(!result[1].cycles_flush, "the branch's own flush is deferred to its slot (word 2)");
        assert!(result[2].cycles_flush, "the branch's taken edge targets entry_word -- flush pushed to the slot, the true last-to-retire word");
        assert!(!result[3].cycles_flush, "not-taken arm's exit-flush is deferred to its own slot (word 4)");
        assert!(result[4].cycles_flush, "jr ra's delay slot is the true last-to-retire word for its exit");
    }

    #[test]
    fn cycles_flush_single_instruction_regjump_region_flushes_at_the_slot_not_the_head() {
        // Regression test for a real bug: a one-head region (entry_word ==
        // the regjump itself, e.g. `jr ra` with a real NOP slot,
        // max_instrs=1) must flush at word 1 (the slot), not word 0 (the
        // regjump). Getting this wrong meant codegen's pass-2 loop flushed
        // `cycles_pending` right after accounting for the head (before the
        // slot's own emit_account_for_cycles call ever ran), storing 1
        // instead of 2 -- caught by
        // equiv_test::tests::jr_with_nop_slot_fuses_and_still_advances_cycles_by_two.
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // jr ra
        page[1] = 0; // real NOP delay slot
        let mut a = Analyzer::new();
        let (result, _) = a.walk_bounded(&page, 0, 0, 1);
        assert!(result[0].has_inline_slot);
        assert!(!result[0].cycles_flush, "the regjump's own flush must be deferred to its slot");
        assert!(result[1].cycles_flush, "the slot is the true last-to-retire word for this region's only exit");
    }

    #[test]
    fn cycles_flush_post_pass_is_bounded_by_min_max_visited() {
        // Regression guard for the min/max bound compute_cycles_flush is
        // scanned over: entry_word starts mid-page, so words before it (and
        // far beyond the walked region) must never have cycles_flush set --
        // if the post-pass scanned the whole page instead of
        // [min_visited, max_visited], unvisited words would still read
        // false (they default to it), so this mostly guards against a
        // panic/out-of-bounds if the bound were ever wrong, and documents
        // the intent.
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[10] = 0; // nop
        page[11] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // jr ra
        page[12] = 0; // delay slot
        let mut a = Analyzer::new();
        let (result, _) = a.walk(&page, 10, 0);
        assert!(!result[0].visited);
        assert!(!result[0].cycles_flush);
        assert!(result[12].cycles_flush, "jr ra's exit-flush is deferred to its own slot (word 12)");
    }

    #[test]
    fn cycles_flush_multi_entry_flushes_on_arrival_at_a_non_primary_entry_point() {
        // walk_multi_entry generalizes compute_cycles_flush's single
        // `entry_word` special case ("no flush needed, a re-entry there
        // always arrives with cycles already fresh") to every entry point,
        // not just the first one walked. Word 1 here is reachable two ways:
        // as word 0's plain fallthrough, AND as its own independent entry
        // (e.g. some other guest PC dispatches straight into it) -- if only
        // the representative entry_word passed to compute_cycles_flush got
        // the "is a valid re-entry" treatment, word 0's fallthrough edge
        // into word 1 would wrongly go unflushed, since word 1 is not
        // is_branch_target (nothing branches to it) and isn't the
        // representative entry_word (word 0 is).
        let mut page = [0u32; ENTRIES_PER_PAGE];
        page[0] = 0; // nop (entry A)
        page[1] = r_type(OP_SPECIAL, 31, 0, 0, 0, FUNCT_JR); // jr ra (entry B)
        page[2] = 0; // real NOP delay slot
        let mut a = Analyzer::new();
        let result = a.walk_multi_entry(&page, &[0, 1], 0, usize::MAX);
        assert!(result[1].is_entry_point, "word 1 must be recorded as its own entry point");
        assert!(
            result[0].cycles_flush,
            "word 0's fallthrough lands on word 1, a real entry point -- must flush on arrival just like a branch target"
        );
    }
}
