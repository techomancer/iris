# BC1 is an ordinary PC-relative branch, and compiling it is worth ~4x

Done 2026-09-22. `BC1F`/`BC1T`/`BC1FL`/`BC1TL` used to classify as
`Classify::Excluded`, which made every FP conditional branch a **region
boundary** rather than a one-instruction fallback — a hot FP loop containing
one `bc1t` got chopped into fragments.

## The reasoning that kept it excluded was wrong

The comment in `analyzer::classify` said BC1 had a
*"condition-code-dependent target"*. It does not. **The target is the same
PC-relative 16-bit offset every other conditional branch uses**, computed in
`exec_bc1` as `pc + 4 + (imm << 2)` — statically resolvable from the
instruction word alone. Only the *predicate* comes from the FPU, and `BEQ`'s
predicate is equally a runtime value that nobody considers unresolvable.

The phrase conflated the condition with the target. Once separated, BC1 is
structurally identical to `BEQL`: a conditional branch, optionally annulling.

## What it took

- `analyzer::classify`: `RS_BC1` now goes to
  `branch_category_gate(raw, branch_target(raw, offset_word))`.
- `BranchCond::Fcc`, and `lookup_branch_or_jump` returning it for
  `OP_COP1` with `rs == RS_BC1`. `nd` (raw bit 17) is this table's `annul`,
  exactly as for `BEQL`.
- `emit_fcc_taken` — factored out of the existing `emit_fmovcf_taken`, since
  MOVCF and BC1 share the `cc`/`tf` encoding (cc at raw[20:18], tf at
  raw[16]) and the FCSR bit layout (cc0 at bit 23, cc1..7 at 24+cc).
- `Bc1` added to `has_jitv2_support`.

### The CU1 guard is the trap

BC1 reads the FPU, so unlike every other branch **it can fault**: `exec_bc1`
checks `STATUS_CU1` first and raises Coprocessor Unusable when clear.

Branches reach codegen through `lookup_branch_or_jump`, which — unlike
`lookup_cp1_semantics` — gets **no** automatic `emit_cp1_cu1_guard`. It is
emitted explicitly in `emit_cond`'s `Fcc` arm.

Position matters: every `emit_cond` call site evaluates the condition
*before* `emit_slot` inlines the delay slot, which is the interpreter's order
too (the exception is raised before the delay-slot instruction runs). Do not
sink it past the slot.

## Measuring it needed assembly

Dhrystone and Whetstone show **nothing** — Whetstone's own text contains a
single `bc1f`; the 852 `bc1t`/`bc1f` are inside `libm.so`. Three attempts at
a C microbenchmark calling `sin`/`cos`/`exp` all collapsed under MIPSpro
`-Ofast`: 24M transcendental calls "completing" in about a second, which is
not physically plausible on an emulator. Making the input non-periodic did
not help either.

**Write the kernel in assembly.** `clanker/jitv2/bc1/bc1bench.s` is a loop
whose only interesting content is `c.lt.d` + `bc1t`, with the compare
operands swapped each iteration so the branch alternates taken/not-taken and
cannot settle into one direction. The accumulator self-checks: alternating
+2/+1 must give exactly `1.5 * n`.

Same kernel, same guest, two emulator binaries, 200M iterations, host wall
clock, warm reps:

| | warm reps | |
|---|---|---|
| BC1 interpreted | 7, 7, 6, 7, 11 s | **~7 s** |
| BC1 compiled | 1, 2, 1, 2, 2 s | **~1.6 s** |

**~4x**, and understated: roughly a second of each figure is ssh overhead, so
the compute ratio is larger. Both arms print `acc=300000000`.

## Tests

`bc1_all_conditions_match_interpreter` covers all 8 condition codes x both
`tf` polarities x both `nd` settings x condition set/clear x both FR modes —
128 combinations, each against the interpreter.
`bc1_likely_not_taken_annuls_its_delay_slot` asserts the annul directly
(and the taken case too, so an emitter that simply never ran the slot could
not pass). `bc1_without_cu1_raises_coprocessor_unusable_like_the_interpreter`
covers the guard.

### Harness gotcha

BC1 tests need a multi-word page (branch + delay slot) *and* CP1 state, which
neither existing harness provides, hence `run_*_page_bc1`. **The FR mode
passed to `compile_region` must match the mode the core is actually in.** Any
region containing a CP1 instruction gets `emit_fr_mode_guard`, and on a
mismatch it calls `jit_kill_entry`, which aborts in a test harness with no
tracked `PhysicalCodePage` ("jit_kill_entry reached with no tracked
PhysicalCodePage"). The integer page harness hardcodes `true` and gets away
with it only because integer regions have no FPU instruction and so no guard.

Related: [`instructions-jitv2-still-interprets.md`](instructions-jitv2-still-interprets.md).
