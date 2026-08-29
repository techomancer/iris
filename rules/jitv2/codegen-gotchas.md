# JIT v2 codegen gotchas

## Cranelift `icmp`/`fcmp` booleans are `0`/`1`, not all-ones/all-zeros — `bnot` on one is not logical negation

`builder.ins().icmp(...)`/`fcmp(...)` return an `I8`-typed boolean `Value`
encoded as plain `0` (false) or `1` (true) in this Cranelift version — *not*
the all-ones-for-true/all-zeros-for-false convention some other IRs (and
LLVM's `i1`, sort of) use. `bnot` is genuine bitwise NOT: `bnot(1) = 0xFE`
(254), which is **truthy** (nonzero) when fed into a later `select` or
`band`/`bor` — i.e. `bnot`-negating a boolean silently does *not* flip it in
the way `!` would in Rust.

Found via a live-boot `jitv2_lockstep` divergence: `emit_round_to_int_mode`
(the portable bit-manipulation rounding primitive — see the
`f32/f64::round()` MXCSR-sensitivity entry below) used `bnot` on four
`icmp`-derived booleans expecting logical negation (`is_positive =
bnot(is_negative)`, `is_lt_half = bnot(is_exp_minus_one)`, etc.) — every one
of them silently selected the wrong branch for any input landing in the
`exp < 0` (magnitude < 1.0) regime. Diagnosed by compiling
`emit_round_to_int_mode` standalone as an isolated `fn(f64, i8) -> f64` (no
`MipsCore`/CVT.W.D dispatch needed) and dumping every intermediate boolean
value through debug out-parameters — `is_lt_half` printed as `254`, not `0`
or `1`, immediately pointing at the encoding mismatch rather than a logic
error in the branch conditions themselves.

**Fix**: use `icmp_imm(IntCC::Equal, v, 0)` to logically negate a
boolean-typed `Value`, never `bnot`. Reserve `bnot` for genuine bitwise-NOT
on real integer bitmasks/registers (e.g. a `frac_mask` built via
`iadd_imm(1 << n, -1)`, or MIPS's actual NOR instruction) — those are correct
as-is since they're not boolean values to begin with.

## Cranelift `sdiv`/`srem` trap on `i32::MIN / -1`

Cranelift's plain `sdiv`/`srem` lower straight to the host `idiv` instruction,
which raises `#DE` (SIGFPE) on x86_64 for `i32::MIN / -1` — same as C's
`INT_MIN / -1`. This is *not* guarded by Cranelift itself.

MIPS `DIV`'s interpreter implementation (`exec_div`, `src/mips_exec.rs`) uses
Rust's `wrapping_div`/`wrapping_rem`, which define this case as
`(i32::MIN, 0)` rather than panicking. A JIT emitter that naively lowers to
`sdiv`/`srem` will SIGFPE the whole process on this input instead of matching
the interpreter.

Fix: explicitly branch on `rs == i32::MIN && rt == -1` before emitting
`sdiv`/`srem`, and materialize `(i32::MIN, 0)` directly on that path. See
`emit_div_impl` in `src/jitv2/codegen.rs`.

Caught by the jitv2-vs-interpreter equivalence harness
(`src/jitv2/equiv_test.rs`) — `div_matches_interpreter`'s
`i32::MIN / -1` case SIGFPE'd the test binary before the fix, confirming the
harness is doing real work rather than rubber-stamping.

Unsigned division (`udiv`/`urem`, DIVU) has no equivalent trap case — there's
no `-1`/`MIN` boundary in an unsigned range, so only the zero-divisor guard
(shared with DIV) is needed there.

## Analyzer's branch target formula was off by one word

`analyzer::branch_target` computed `target = offset_word + 2 + imm16`. The
correct formula is `offset_word + 1 + imm16` — a branch's target is defined
relative to its **delay slot's own address** (one word past the branch),
not two words past it:

> `target = PC_branch + 4 + (imm16 << 2)`, and the delay slot sits at
> `PC_branch + 4` — so in word units, `target_word = branch_word + 1 + imm16`.

Confirmed against `MipsExecutor::exec_beq` et al.'s actual arithmetic:
`self.core.pc.wrapping_add(4).wrapping_add(d.immu64())`, where `d.immu64()`
is `imm16 << 2` bytes and `self.core.pc` is the branch's own address.

This bug was self-consistent within `analyzer.rs`'s own unit tests (they
computed expected targets with the same wrong formula, so nothing caught
it) — it only surfaced once codegen's branch/jump batch cross-checked
against the *real* interpreter via the equivalence harness
(`equiv_test.rs`), which is exactly the point of that harness. Both
`analyzer::branch_target` and codegen's independent re-derivation
(`branch_target_word` in `codegen.rs`, which recomputes the same value from
`raw`/`word` since the analyzer doesn't persist resolved targets in
`CompiledInstr`) had to be fixed together — they must always agree.

## Cranelift's "entry block" is layout order, not creation order

`FunctionBuilder::create_block()` just allocates a `Block` handle — it does
not place the block into the function's layout. A block only enters layout
order on its first `switch_to_block` call. Cranelift's verifier treats
whichever block ends up first in layout order as the function's entry
block, and requires its param count to match the function signature
exactly.

`compile_region` creates `entry_block` first (correct — one param, matching
the `*mut MipsCore` signature) but, in an intermediate refactor while adding
the FPU entry guard, called `switch_to_block(exit_block)` (two params:
`core_ptr`, `word_offset`) before ever switching into `entry_block`. Even
though `entry_block`'s handle was allocated earlier, `exit_block` became
first-in-layout and thus the verifier's "entry block" — rejected with
`entry block parameters (2) must match function signature (1)`.

Symptom was misleading: `compile_region` returned `None` silently (the
`.ok()?` on `define_function`'s `Result` swallows the verifier error),
which looked exactly like "declined to compile" rather than "compiled
something malformed." Diagnosed by temporarily swapping `.ok()?` for
`.expect(...)` to surface the real `VerifierErrors`.

Rule: whichever block must be the function's actual entry has to be the
first one `switch_to_block`'d, regardless of `create_block()` order. It's
fine to allocate a later block's handle early (so an earlier block can
reference it as a branch target) — just don't call `switch_to_block` on it
before the real entry block.

## Writing cp0_status directly in tests must call update_fpr_mode()

`MipsExecutor::fpr_read_w`/`fpr_write_w`/etc. are cached fn pointers, not
re-derived on every access — they're only re-bound in `update_fpr_mode()`,
which the real dispatch path calls from `on_cp0_status_changed` (wired up
via `write_cp0`'s callback). A test harness that pokes
`exec.core.cp0_status` directly (bypassing `write_cp0`) leaves those fn
pointers stale at whatever `MipsExecutor::new()` initialized them to,
silently reading/writing FPRs under the *old* FR mode regardless of what
the field now says.

Symptom: an FR=1 equivalence test wrote its ADD.S result into the wrong
half of the wrong register — the interpreter side had FR=0 accessors still
installed (odd/even packing), so `fs`/`ft`/`fd` all resolved to different
physical slots than the JIT (which reads the live `STATUS_FR` bit correctly
at compile time). Fixed by calling `exec.update_fpr_mode()` right after any
direct `cp0_status` write in test setup (`fpu_seeded_executor` in
`equiv_test.rs`).

## An exit-stub bail is a retry, not a delivery — equivalence tests for exception paths need two steps

`emit_fpu_entry_guard` (and every other preamble/guard bail) materializes a
*retry* PC and returns `EXEC_COMPLETE_NO_INC` — it does not itself deliver
the exception. The real exception delivery (`cpu_unusable`/`EXC_CPU` for a
CU1 guard, `handle_exception`'s EPC/Cause/vector math) only happens on the
interpreter's *next* dispatch at that same retried address, when it
re-decodes the instruction and hits the real check itself (§3.3's "plain
boundary" contract).

A test that calls the compiled `JitFn` once and immediately snapshots state
will see *no* exception at all on that path — pc unchanged, nothing
vectored — because the bail only rewound the PC; nothing has re-executed
yet. This looks exactly like "the guard didn't fire," which is a much
scarier (and wrong) diagnosis. Fix: after calling the compiled function,
assert `core.pc` landed back at the expected retry address, then call
`exec()` (or however the harness drives the interpreter) once more at that
address — mirroring what the real dispatch loop's next iteration would do —
before comparing snapshots.

## Rust's f32/f64::round()/trunc()/ceil()/floor() are not safe to build MIPS rounding semantics on — superseded by a portable bit-manipulation primitive

(Original finding, kept for history: `.round()` used to set MXCSR's
Precision/Inexact flag in debug builds but not release, meaning the
interpreter's own FCSR.Inexact "ground truth" for CVT.W/ROUND.* wasn't even
stable across the project's own build profiles. Superseded — see below.)

A live IRIX boot under `jitv2_lockstep` produced a genuine, reproducible
divergence: `CVT.W.D $f4, $f6` on `65535.5` under FCSR.RM=1 (round-toward-
zero) gave JIT=65536 (round-half-away-from-zero) vs interp=65535
(round-toward-zero) — even though at the time *neither* engine's `CVT.W.D`
handler consulted FCSR.RM at all; both unconditionally called `.round()`
(interpreter) or its Cranelift IR equivalent (JIT). Instrumenting
`exec_fcvt_w_d` directly (an `IRIS_DEBUG_CVT_W_D` env-gated eprintln) proved
`f64::round()` itself returned `65535.0` for `65535.5` on this exact build,
with the host's live MXCSR rounding-control bits (`mxcsr_rc`) reading `0x3`
(RZ) at the call site — i.e. `.round()`'s *result value* silently tracked
ambient MXCSR state, contradicting a from-scratch standalone repro (same
Rust version, same explicit MXCSR-RZ setup via inline `ldmxcsr`) that
reliably returned `65536.0`, and contradicting the disassembly of the real
binary's `exec_fcvt_w_d`, which showed a `vroundsd $0xb, ...` with a fixed,
non-MXCSR-select immediate that per the Intel SDM should ignore MXCSR.RC
entirely. The exact mechanism was never fully explained (LTO/codegen-units=1
whole-program optimization producing different code than an isolated
single-file `rustc -O` build is the leading suspect, but unconfirmed) — the
behavior was 100% reproducible against the real binary regardless, so it
couldn't be trusted no matter the cause.

**This means the earlier "MXCSR Inexact differs by build profile" finding
above was actually a symptom of a bigger problem**: `.round()`/`.trunc()`/
`.ceil()`/`.floor()` (and by extension Cranelift's `nearest`/`trunc`/`ceil`/
`floor` IR ops, which lower to the same SSE `ROUNDSS`/`ROUNDSD` family) can't
be trusted to produce a fixed, MIPS-spec-defined answer independent of
ambient host FPU control-register state on this codebase's actual build —
not just "sometimes sets a spurious flag" but "can silently return a
different *value*".

**Fix**: replaced every host rounding call/IR op in the whole ROUND/TRUNC/
CEIL/FLOOR/CVT.W/CVT.L handler family (all 20: S/D source × W/L dest × 4
modes + plain CVT, both engines) with a portable, pure bit-manipulation
round-to-integer primitive that never touches a hardware rounding
instruction at all — `mips_exec.rs::round_f32_to_int_mode`/
`round_f64_to_int_mode` (interpreter: splits the mantissa into integer/
fractional parts by exponent, computes round-up-or-down per MIPS FCSR.RM
encoding via plain integer comparisons, increments via *integer addition on
the raw bit pattern* — critical, since `truncated_value + 2f64.powi(exp)`
doesn't correctly propagate a mantissa-overflow carry into the exponent
field, only bit-pattern integer addition does) and
`codegen.rs::emit_round_to_int_mode` (JIT: the same algorithm as branchless
Cranelift IR via `select` instead of control flow, parameterized by an SSA
`rm: Value` so it serves both the four fixed-mode ROUND/TRUNC/CEIL/FLOOR
call sites via `iconst` and the two plain-CVT.W/CVT.L call sites via a
runtime load of FCSR's live low 2 bits).

Closed a second, previously-deferred, already-documented spec gap in the
same pass since it's the same root cause: plain (unprefixed) `CVT.W`/`CVT.L`
are architecturally specified to honor FCSR.RM dynamically, not hardcode
round-half-away-from-zero — both engines now do (see
`project_fpu_rounding_spec_gap` memory).

Regression coverage: `mips_exec::round_to_int_mode_tests::*` (unit tests on
the primitive directly, including the exact `65535.5`/RM=1 case and a
mantissa-carry-into-exponent case) and `jitv2::equiv_test::tests::lockstep_fpu_cvt_w_d_*`
/ `lockstep_fpu_round_w_s_ignores_fcsr_rm` / `lockstep_fpu_trunc_ceil_floor_ignore_fcsr_rm`
(full JIT-vs-interpreter lockstep agreement across all 4 RM values, plus
confirming ROUND/TRUNC/CEIL/FLOOR still correctly ignore FCSR.RM).

## Exit-block pc materialization used `bor` instead of `iadd` — broke exactly at a page-crossing fallthrough

`emit_exit_block_body` (the shared `exit_block` every `emit_bail` call jumps
to) computed the bail target as `vbase | (word_offset * 4)`, not
`vbase + (word_offset * 4)`. This is only safe when `word_offset <
WORDS_PER_PAGE` (1024) — every in-range case has `byte_offset < 0x1000`,
which never overlaps `vbase`'s already-zeroed low 12 bits, so `bor` and
`iadd` agree.

But a **Sequential** (non-branch) instruction sitting at the page's last
word (offset 0xFFC, word 1023) legitimately falls through to word 1024 —
"next page, word 0." (The 0xFFC hazard special-case in `analyzer::visit`
only applies to branch/jump/regjump, which need an inline delay slot;
Sequential has no such need and just falls through normally, ending up with
`word_offset == WORDS_PER_PAGE`.) At exactly that value, `byte_offset ==
PAGE_SIZE (0x1000)`, whose bit 12 collides with `vbase`'s own bit 12 —
`vbase | 0x1000 == vbase` whenever that bit happens to already be set,
silently dropping the page carry and landing the JIT back on the *same*
page instead of the next one.

Found via `jitv2_verify` against a real IRIX 5.3 boot trace, showing up as
an "UNEXPLAINED control-flow diverged" line with the JIT's pc exactly one
page behind the trace's — e.g. a Sequential instruction at `0x...9fc0fffc`
landed the JIT at `0x9fc0f000` instead of the correct `0x9fc10000`. Confirmed
at the disassembly level (the compiled `or r10, rax` was executing exactly
as encoded — this was never a Cranelift bug, just wrong IR going in).

**Fix**: use `iadd` instead of `bor` to combine `vbase` and `byte_offset` —
identical result for every in-range `word_offset`, correct carry at the
`WORDS_PER_PAGE` boundary. Regression test:
`sequential_at_last_word_falls_through_to_next_page_not_same_page`
(`equiv_test.rs`), deliberately using a page base whose next-page-boundary
bit is set so a reverted fix fails loudly instead of hiding behind a
lucky-bit-pattern page address.

## Self-looping backward branches can't be validated by single-instruction jitv2_verify mode (not a codegen bug) — resolved by chain mode's back-edge guard

A branch whose taken target lands on an already-visited word within the same
walked region (most commonly a literal self-loop, e.g. `BGTZ r2,-1` with a
decrementing delay slot, or a 2-instruction loop like `ADDI r2,r2,-1` /
`BNE r2,r0,-2` whose target is the *previous* word) hits `analyzer::visit`'s
"already visited, stop recursing" base case and gets `taken_exit = None` —
"continues into compiled code." Per the design (`jit-v2-design.md` §2.2/§3.1,
"loops stay native"), this is intentional: codegen wires a real Cranelift
back-edge, and the compiled function runs the *entire* loop to convergence
natively in one call, never returning to the interpreter between iterations.

`run()`'s single-instruction/single-branch-unit trace-replay model assumes
one compiled-region call advances state by exactly one trace record (or one
branch+slot pair). A self-looping region breaks that assumption by
construction — it's correct JIT behavior colliding with a verifier that has
no way to represent "this call did N iterations." Confirmed by hand-tracing
the real delay slot from a captured boot trace: the JIT's branch condition
correctly reads the pre-slot register value (matching real MIPS delay-slot
semantics), and nesting a debug build with an artificial `cp0_compare` far
in the future to disable the IP7 timer preamble shows the loop genuinely
converges to the same final register state the trace implies over many
iterations — just not divided into single-record steps the verifier can
check.

**Resolved in chain mode** (`--chain N`, see below): `run_chain`'s
`would_create_back_edge` guard detects, *before* including a candidate word
as a chain head, whether doing so would give the analyzer's walk an edge
landing back on a word already in the chain — and if so, stops growing the
chain there instead of silently letting the analyzer wire a native
back-edge into a region the comparison logic can't represent. This isn't
"skip the loop" — the loop's own two instructions (the decrement and the
branch+slot) still each get verified individually, correctly, as their own
separate single-head chain attempts; only the specific *pairing* that would
create a native back-edge is refused. Confirmed against the real trace: the
same `0xbfc03bb8`/`0xbfc03bbc` region that previously produced ~2000
UNEXPLAINED divergences per 500K-record sample now produces zero — the
divergences were entirely this artifact, not a separate real bug.
`run()` (single-instruction mode) still has no equivalent guard and will
still report this class as UNEXPLAINED — that's fine, chain mode is strictly
more capable here, not a replacement for `run()`'s speed on a full-trace
first pass.

## Branch-in-delay-slot ("unusual but legal") panicked the JIT compiler live

A real IRIX boot hit `panicked at src/jitv2/codegen.rs:1749:18: exit_reason
is None -> analyzer guarantees target_word continues into the region` —
not in `jitv2_verify`, in the *live* JIT compiler thread. Root cause: a
branch/jump whose target word coincides with a word that's *also* someone's
delay slot (§6.1.4's "same-offset dual semantics" — the design doc already
named this case, but the implementation didn't actually handle it).

Three compounding gaps, found and fixed in this order:

1. **`comp.rs`'s live budget (`MAX_INSTRS_PER_COMPILE=2`) charged the
   mandatory slot against the same budget as the head instruction.** A
   delay slot can never be omitted (§6.1.4) — it was never a real
   truncation candidate. Changed to `max_instrs=1` (head instructions
   only); `analyzer::visit_slot` no longer decrements `budget` at all.
   Side effect: this also lets a nested branch-in-delay-slot's own slot
   extend the walk for free instead of instantly blowing the budget.

2. **A branch/jump/regjump sitting *in* a delay slot decoded as inert.**
   Real hardware allows this ("unusual but legal"; the interpreter's
   `branch_delay` already nests correctly). `visit_slot` used to mark any
   slot's raw bits visited without ever checking whether they themselves
   decode as a branch/jump/regjump needing their *own* mandatory slot.
   Fixed by making `visit_slot` recurse into a nested slot-chain,
   arbitrarily deep, stopping only at a genuine hard limit (off-page,
   excluded, 0xFFC hazard) — matching the user's framing: "if bdslot
   happens to be a jump as well it should continue bdslot extensions...
   if that happens we should just give up and denylist it" (a chain that
   can't complete just fails the walk, same as an excluded slot always did).

3. **A word visited only as a slot never got promoted when later reached as
   a genuine branch/jump target.** `analyzer::visit`'s "already visited,
   no-op" fast path didn't distinguish "already a real head" from "only
   ever seen as someone's slot content" — so a target landing on a
   slot-only word silently kept `taken_exit`/`fallthrough_exit` as `None`
   ("continues into the region") while codegen's own local `is_slot`
   heuristic (recomputed from scratch, blind to the analyzer's walk)
   still excluded that word from `block_for_word` — the literal cause of
   the panic. Fixed by adding `CompiledInstr::is_slot_only: bool`
   (analyzer-owned ground truth) and a promotion path in `visit`: the
   first time a slot-only word is reached as a real target, it computes
   real edges and flips `is_slot_only` to `false`, without re-walking its
   slot-chain or double-charging budget. Codegen's pass 1 now reads
   `instr.is_slot_only` directly instead of recomputing its own version —
   analyzer and codegen agree on ground truth instead of each guessing.

**A fourth bug surfaced once the above three let nested-slot codegen
actually run**: `emit_slot_semantics` always assumed control fell through
to the caller after emitting a slot's semantics, so the caller could keep
emitting its own condition-test/exit-wiring IR into the same Cranelift
block. That's false for a nested branch/regjump slot — every one of its
arms is a terminator (`emit_absolute_pc_exit`/`emit_runtime_pc_exit`,
matching real hardware's rule that the *innermost* dispatched transfer
wins, superseding whatever the outer instruction "wanted" to do) — so
Cranelift's verifier correctly rejected the outer instruction's dead code
after that terminator (`"a terminator instruction was encountered before
the end of block"`). Fixed by making `emit_slot_semantics` return `bool`
(did this call terminate the block?) and having every call site
(`emit_branch_or_jump`, `emit_regjump`, and the new
`emit_nested_branch_slot`/`emit_nested_regjump_slot` for deeper nesting)
check it and skip their own remaining emission when `true`.

**Test-suite fallout**: the budget semantics change (slots no longer
charged) meant every existing test hardcoding `max_instrs=2` for "a single
branch/jump plus its slot" was implicitly relying on the old "slot costs 1"
accounting — with slots free, `2` now means "walk two *head* instructions,"
which pulled in an extra (previously-truncated) instruction and shifted
several tests' expected exit points by one word. Fixed by changing every
such call site to `max_instrs=1`, and updated
`walk_bounded_budget_includes_delay_slot`'s test (renamed
`..._excludes_delay_slot`) to assert the new contract directly instead of
the old one.

Regression tests: `nested_regjump_in_delay_slot_compiles_and_innermost_transfer_wins`
and `self_loop_with_decrement_in_delay_slot_converges_natively`
(`equiv_test.rs`) cover the dual-semantics promotion, the nested-slot
termination fix, and (as a byproduct) confirm the self-loop-convergence
behavior described in the section above actually produces the
hand-computed-correct final register state, not just "doesn't crash."

## jitv2_verify's multi-instruction chain mode (`--chain N`) — real bugs found building it

Added `--chain N` to `src/bin/jitv2_verify.rs`: instead of verifying one
instruction (or one branch+slot unit) per compiled region, grow each
attempt to up to `N` head instructions by following the trace's own
recorded dynamic path through interior branches — more of the compiled
region's real control-flow logic gets exercised per comparison. Three real
bugs surfaced while building and testing it against a real IRIX 5.3 boot
trace:

**1. `run()`'s `max_instrs=2` convention was stale.** Earlier this session,
the analyzer changed so a branch/jump's mandatory delay slot is never
charged against the walk budget (`analyzer::visit_slot` — see the "Fixed
budget/`is_slot_only`" sections above). `run()`'s own single-instruction
compile path was never updated to match — it still passed `max_instrs=2`
for a branch/jump ("branch + slot"), which under the new semantics means
"walk 2 *head* instructions," letting the walk pull in one extra
(previously-truncated) instruction past the branch's own unit. Fixed by
changing it to `max_instrs=1` unconditionally (slots are free at any
`max_instrs` value now). Caught by a previously-passing unit test
(`branch_and_delay_slot_compare_against_the_third_record`) that had
silently started failing without anyone noticing — a reminder to actually
run existing test suites after a "should be behavior-preserving" analyzer
change, not just the tests directly related to the change.

**2. The chain-builder didn't check `Classify::Excluded` before including a
word as a head.** Only `touches_memory` was checked when deciding whether
to extend a chain into the next trace record — an `Excluded`-classified
instruction (COP0/MFC0/MTC0/TLB*/ERET/WAIT, CACHE, LL/SC, CP2, the CP1
conditional branch RS_BC1) is one `analyzer::visit` refuses to walk at all,
so the *real* compiled region silently ends one head earlier than the
chain-builder assumed, desyncing the comparison point from what actually
got compiled. Manifested as false-positive "control flow diverged"
UNEXPLAINED reports at real trace addresses (found via a `bfc01a48` region
in a real boot trace, entry `ADDI r3,r3,1` / `BNE ...` taken into an
`MFC0`). Fixed by checking `class == Classify::Excluded` alongside
`touches_memory` at both chain-extension sites (initial build loop and the
past-a-branch extension loop). Regression test:
`chain_mode_does_not_extend_through_an_excluded_instruction`.

**3. `run_chain` had no compile cache at all** (unlike `run()`, which has
had one since early this session for exactly this reason — see its own doc
comment). A real boot trace revisits the same loop bodies constantly; a
tight loop found at `0xbfc03e0c`–`0xbfc03e24` repeats thousands of times in
sequence. Without caching, every single visit triggered a fresh Cranelift
compile of the identical chain shape — a `--chain 4 --limit 2000000` run
never got past ~6500 records in several minutes of CPU time with steadily
climbing memory (2.1GB→2.4GB+), while a `--chain 1`-equivalent (`run()`,
which does cache) processes the same trace region instantly. Diagnosed by
directly inspecting the trace records around the stuck offset
(`TraceReader::skip_records` + a scratch dump tool) and recognizing the
repeating `pc` sequence as a loop, not a stall. Fixed by adding a
`compile_cache: HashMap<(Vec<u32>, u16, u32, bool), Option<JitFn>>` keyed
on the whole chain's raw-byte sequence (head + slot raw words, in order)
plus `entry_word`/`page_base`/`fr1` — mirrors `run()`'s cache exactly, just
with a `Vec<u32>` key instead of a fixed-size tuple since a chain's length
varies. After the fix, a `--chain 4 --limit 500000` run against the same
trace region completed in well under the time the uncached version took to
process 6500 records. **Lesson**: any new codegen-invoking loop over a real
boot trace needs a compile cache from the start — real traces are
loop-heavy by nature (that's most of what a running OS does), so "compile
once per visit" is never viable, not just slow.

**4. Chains could silently create their own native back-edge** — the exact
"loops stay native" mechanism described in this file's self-loop section
above, but now self-inflicted by the chain-builder itself: including a
2-instruction loop's decrement (`ADDI r2,r2,-1`) and its branch
(`BNE r2,r0,-2`, target = the decrement's own word) as two heads in the
same chain gives the analyzer's walk a genuine back-edge from the branch's
taken arm into the already-visited decrement — the compiled function then
runs to convergence natively in one call, desyncing from the chain's
single-trace-record comparison point. Manifested as ~2000 UNEXPLAINED
divergences per 500K-record sample, all at one real trace loop. Fixed by
`would_create_back_edge` (checked at both chain-growth sites — see this
file's dedicated section above for the fix's full detail and confirmation
that it eliminates the divergences entirely, down to zero, without
suppressing or skipping the loop's own instructions).

## A standalone-compiled word can silently be someone else's delay slot

`exec_decoded`'s dispatch gate (`mips_exec.rs`) unconditionally probes
`entry_offset == 1` for compilation on *every* dispatch to that offset,
regardless of why PC landed there (§6.1.4's "one statically always-checkable
offset" heuristic). This means a page's word 1 can get compiled and
dispatched as an ordinary, standalone `compile_region` entry even when the
interpreter's *immediately preceding* dispatch reached it by executing some
other instruction's `j`/branch at word 0 — i.e. while `core.in_delay_slot`
is true and `core.delay_slot_target` holds the real, pending destination.

The standalone compile has no way to know this: `compile_region` only
manages `in_delay_slot`/`delay_slot_target` explicitly for a branch/jump's
own *inlined* slot (`emit_slot_semantics`, entered via
`emit_branch_or_jump`) — a plain `Sequential` head instruction's exit
(`emit_exit_block_body`) always wrote `core.pc` from its own *compile-time*
`fallthrough_word`, with zero runtime awareness that this exact dispatch
might really be someone else's armed delay slot. Before the fix, that
silently discarded the pending transfer — found live on the real IRIX PROM
reset vector: `j realstart` (word 0) followed by a `nop` delay slot (word
1); word 1 got compiled standalone via the `entry_offset == 1` probe and
looped back to the next sequential word instead of `realstart`, corrupting
downstream execution non-deterministically depending on JIT dispatch timing
(single-stepping reproduced it every time; a from-scratch interpreter-only
run never did — see below for why `jitcheck`/`jitv2_lockstep` never caught
it either).

**Fix** (`compile_region`'s plain-`Sequential` exit path, entry-word only —
see `EmitCtx`'s doc comment on `entry_word` for why only the entry word can
have this ambiguity: every other word in a compiled region is deterministically
not-a-foreign-slot by the region's own control flow): load `core.in_delay_slot`
at runtime before falling through; if set, exit via `core.delay_slot_target`
(clearing the flag) instead of the compile-time fallthrough word — mirroring
`handle_exec_complete` exactly. This also required moving `delay_slot_target`
itself from `MipsExecutor` onto `MipsCore` (alongside `in_delay_slot`, which
was already there for the same reason) — JIT-compiled code only ever gets a
raw `MipsCore*`, so a field the JIT needs to read must live there, not on the
executor wrapper.

**Why `jitv2_lockstep`/`jitcheck` never found this**: both always compile a
region keyed to *exactly* what's being dispatched right now, with the
interpreter's real state as ground truth for that same dispatch — a
branch+slot pair is walked and compiled together as one unit
(`lockstep_check`'s `walk_bounded` from the branch's own entry word). Neither
has any notion of a *separate*, later, decoupled compile of a bare word
triggered independently of whatever dispatch armed `in_delay_slot` earlier —
that decoupling only exists in the real async/inline compile-request path
(`comp::handle_request`, reached via the `entry_offset == 1` always-probe),
which lockstep deliberately never touches (its own dispatch gate is compiled
out entirely under `jitv2_lockstep`, precisely so nothing else intercepts a
word before lockstep's own comparison does). Regression coverage:
`jitv2::equiv_test::tests::standalone_compile_of_a_foreign_delay_slot_honors_pending_transfer`
compiles a bare word with `in_delay_slot`/`delay_slot_target` pre-armed (as
the interpreter would leave them) and asserts the compiled function honors
the pending transfer rather than its own fallthrough.

## The CVT/ROUND/TRUNC/CEIL/FLOOR helper callouts must pass `core.jit_ctx`, not `ctx.core_ptr`

`emit_fcvt_to_int` / `emit_fcvt_from_int` / `emit_fcvt_s_d` call out to
`jit_cvt_to_int` / `jit_cvt_int_to_float` / `jit_cvt_d_to_s`
(`mips_exec.rs`), whose first parameter is the **`MipsExecutor<T,C>`**
pointer — each does `&mut *(ctx as *mut MipsExecutor<T,C>)` then
`cvt_*_and_commit(&mut exec.core, …)`.

Every other JIT->Rust callout in this file loads that pointer from
`core.jit_ctx` (`load(ptr_ty, core_ptr, core_offset_of_jit_ctx())`) — the
`MipsCore` field `install_jit_hooks` stamps with `self as *mut Self`. These
three passed the bare `ctx.core_ptr` (the `*mut MipsCore` the JitFn is
called with, `jit_fn(&mut self.core …)`) instead.

`ctx.core_ptr == core.jit_ctx` **only if `core` sits at offset 0 of
`MipsExecutor`**, which `repr(Rust)` does not guarantee — and doesn't
deliver here: with a big align-8 first field followed by an `Arc`, bools, a
`Vec`, etc., rustc packs the small fields ahead of `core`
(`offset_of!(MipsExecutor, core)` came out 32 on a representative layout).
So the helper received `self + off`, treated it as the executor base, and
read/wrote the FPR file at `self + 2*off` — corrupting `sysad`/`tlb`/`cache`
bytes while leaving the real `core.fpr` untouched.

Found via a `jitv2_lockstep` divergence: `trunc.w.d f0, f0` (and, in another
run, `cvt.d.l f3, f3`) reported the JIT's destination FPR **unchanged** —
`jit=<input>  interp=<converted>` — with nothing else diverging and no
crash. The tell is "the result register just never updated": the callout
ran, the conversion math is shared/deterministic, so a silent no-op on the
real `core` means the write landed somewhere else.

**Fix**: load `core.jit_ctx` and pass that, exactly like `emit_mem_read_callout`
and friends. In normal (non-lockstep) execution this bug silently corrupts
`MipsExecutor` memory on every CVT/ROUND/TRUNC/CEIL/FLOOR/`cvt.s.d`, so it is
not lockstep-specific — lockstep just makes it a hard, immediate failure.

## Every FP computational op rewrites FCSR.Cause — even the ones that raise nothing (CVT.D.S)

The R4000 Cause field holds *only the last FP instruction's* exceptions and is
rewritten by every floating-point operation, including to zero when the op
raised nothing (`fpu_update_fcsr_full`'s unconditional `fpu_fcsr &= !FCSR_CM`,
and its own doc comment). The interpreter honours this everywhere — arith via
`fpu_update_fcsr`, ABS/NEG via `fpu_check_snan_operand`, compare via
`exec_fcc_*`, the CVT commit helpers, and — the easy one to miss —
`exec_fcvt_d_s` via `fpu_update_fcsr(0, …)`.

`emit_fcvt_d_s` was a plain `fpromote` + FPR write with **no FCSR touch at
all**, on the (correct but incomplete) reasoning that CVT.D.S is always exact
and raises no exception. It still has to clear Cause. Skipping it let a stale
`Cause.I` from a preceding inexact CVT ride through a CVT.D.S, surfacing as a
`jitv2_lockstep` `fcsr` divergence: `jit=0x00001004 interp=0x00000004`
(JIT kept Cause.I set, interp cleared it — interp/spec correct).

**Fix**: route CVT.D.S's write through `emit_fpu_update_fcsr(ctx, iconst 0, …)`,
same as the interpreter's `fpu_update_fcsr(0, …)`. When adding any new FP
computational emitter, the default is "clears Cause"; the only ops that leave
FCSR entirely untouched are MOV.fmt (and MOVCF.fmt) — ABS/NEG do clear it (via
the SNaN-operand check).

## `LOCKSTEP_BD_LIVE` must still anchor `core.pc` — only `in_delay_slot` is genuinely "live"

`emit_lockstep_step(trust_live=true)` (entry words, delay slots,
`is_branch_fallback_successor` words) passes `bd = LOCKSTEP_BD_LIVE`, and
`lockstep_step` used to skip setting `core.pc` entirely in that case, on the
comment's claim that live `core.pc` "already equals the instruction's own
address."

That holds when the word was reached by **interpreter dispatch** or a **BC1
interpreter fallback** — both write `core.pc`. It does **not** hold when a JIT
branch/jump took its **in-region edge** into the word's block:
`emit_target_edge`'s `None` arm is a bare `builder.ins().jump(target_block)`
with no `core.pc` write, so `core.pc` is left stale at whatever the last
materialization set — typically the branch's own address, via
`emit_slot_semantics`' `saved_pc` restore (itself the value a *preceding*
instruction's lockstep compare wrote as `core.pc = word+1`).

Collision case: a word that is both `is_branch_fallback_successor` (so it gets
the `trust_live` bracket) **and** a JIT branch target (so it's reached with a
stale pc). Found live: `beq` taken to a `ldc1` fallback-successor —
`interp.pc` came out one word past the branch (`exec_ldc1` + `handle_exec_complete`
anchored at the stale branch address), `jit.pc` was correct (the block's own
foreign-slot check materialized it), spurious `pc` divergence.

**Fix**: `lockstep_step` sets `self.core.pc = pc` **unconditionally** (`pc` is
always `emit_word_addr(word)` — the instruction's own address, authoritative
however it was reached). Only `in_delay_slot`/`delay_slot_target` stay
trusted-live for `LOCKSTEP_BD_LIVE` — a fallback-successor really can arrive
mid-delay-slot and a compile-time `bd` would clobber the inherited pending
transfer.
