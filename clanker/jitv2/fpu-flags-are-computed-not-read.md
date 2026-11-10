# FPU exception flags are computed from bit patterns, never read from the host

Five source sites reference this note (`mips_exec.rs:140`, `:6681`, `:6810`;
`codegen.rs:4410`, `:4807`). It did not exist until 2026-09-11 — written here
retroactively from those comments and from the behaviour they describe.

## The rule

An FPU handler's `flags` argument (FCSR bits [6:2] = FV, FZ, FO, FU, FI) is
**computed by the caller from operand and result bit patterns**. It is never
read back from the host FPU's status word.

Why: the interpreter and the JIT must agree exactly, and a JIT-emitted sequence
has no portable way to read host FPU status — Cranelift does not expose it.
Reading it in one engine and not the other produced real interpreter/JIT
divergence before this rule was adopted. Computing from bit patterns is the only
formulation both engines can implement identically.

## What is deliberately NOT computed

For plain ADD/SUB/MUL/DIV/SQRT/RECIP/RSQRT and the MADD family, `flags` is **0**
unless a specific condition applies (signalling-NaN operand -> FV,
divide-by-zero -> FZ, and the result rules below). In
particular:

- **Inexact (FI) is never set by arithmetic on its own.** It is set only as
  part of an Overflow.
- **Overflow (FO) and Invalid (FV) are computed for ADD/SUB/MUL/DIV only**,
  from the result: one range test on the result's magnitude, which replaced
  the denormal-result test on the same path, sends zero, denormals, ±MAX,
  ±infinity and NaN to a cold side. There `fpu_binop_top_flags_s/d`
  (`mips_exec.rs`) classifies the top of the range: a NaN from non-NaN
  operands is Invalid (inf-inf, 0*inf, 0/0, inf/inf) and delivers the MIPS
  default NaN 0x7FBFFFFF / 0x7FF7FFFF_FFFFFFFF; an infinity, or a ±MAX whose
  exact value reaches 2^(emax+1), from finite operands is Overflow+Inexact.
  The JIT reaches the same function through `fpu_binop_commit_fn`, one
  call that also does the operand checks and the denormal-result rule for
  a denormal, infinite or NaN operand or a denormal or top-of-range result
  (zero operands and results stay inline), so the engines cannot disagree
  and each ADD/SUB/MUL/DIV carries one call, not a copy of the rules. 0/0 is Invalid, not divide-by-zero; FZ needs a
  finite nonzero dividend. The Octane PROM's IDE `fpu` test checks both
  (its "invalid result" and "overflow result" cases). SQRT, RECIP,
  RSQRT, the MADD, MSUB, NMADD and NMSUB families and CVT.S.D follow the
  Invalid rule only (next bullet) and still compute no Overflow or Inexact:
  an overflowing result from any of them is delivered as the host rounds
  it, with FCSR untouched. CVT.S.D has its own overflow check but no
  Inexact-with-Overflow.
- **Invalid results of the operations above deliver the MIPS default NaN.**
  The host's NaN for an invalid result (x86: 0xFFC00000) has the quiet bit
  set, which is a *signalling* NaN under the MIPS encoding, so an
  instruction that read it would raise V where hardware takes E.
  `fpu_fix_invalid_s/d` (interpreter) and `emit_fix_invalid` (JIT) replace
  a NaN result with 0x7FBFFFFF / 0x7FF7FFFF_FFFFFFFF and add V when an
  operand was a signalling NaN or no operand was a NaN (infinity times
  zero, infinity minus infinity, the square root of a negative number). A
  quiet NaN operand propagates unchanged. Both engines gate it on one
  compare of the result's magnitude, with the classification on the cold
  side. A finite result proves the operands raise nothing (a NaN operand
  makes the host result a NaN, a zero divisor or an infinity makes it an
  infinity, a negative square-rooted operand makes it a NaN), so for RECIP,
  RSQRT and the MADD family the operand flags are not even evaluated on that
  path (`fpu_fix_invalid_s/d` takes them as a closure,
  `emit_commit_finite_first` as an emitter). CVT.S.D, which runs in `cvt_d_to_s_and_commit` for both engines,
  does the same for a signalling NaN source.

  Known limit: RECIP, RSQRT and the MADD family have no operand check, so a
  quiet-NaN operand (quiet bit clear, the MIPS encoding) propagates with the
  host's quieting, which sets the fraction MSB. The result, for example
  0x7FE00000, then reads as a *signalling* NaN to the next operation, which
  raises V where hardware probably takes E. SQRT traps a quiet-NaN operand
  with E before this point. What an R10000 does with a quiet-NaN operand to
  these operations is not stated in the manuals we have, so it is left.
  A quiet-NaN source to CVT.S.D has a related quirk: it is never equal to
  its widened result, so it leaves the exact path and sets Cause and Flag I.

Stated in `fpu_update_fcsr`'s doc comment as: *"not worth the IR to compute for
ordinary arithmetic, and nothing in IRIX depends on them."* That trade is sound
for the emulator's actual workload — but it is the direct cause of the standing
cpu-tests failures below, so it is a **known gap, not a regression**.

## The cpu-tests it accounts for

Five of the 15 standing failures (see
`clanker/testing/cpu-tests-known-failure-baseline.md`). All five are flag-only:
the computed *result* is correct in every case.

| test | symptom |
|---|---|
| `fpu/vec_arith_single` | `o.flags` got `0x0`, want `0x1`/`0x5` — missing FI |
| `fpu/vec_arith_double` | same |
| `fpu/vec_sqrt` | `o.flags [sqrt.s]` got `0x0`, want `0x1` — missing FI |
| `fpu/cvt_s_d_rounds` | got `0x4`, want `0x5` — has FO, missing FI (CVT.S.D, not covered by the ADD/SUB/MUL/DIV overflow rule) |
| `fpu/cvt_out_of_range` | `cvt.w.s` of 2^31 does not set FV (though `+inf` does) |

`fpu/cvt_out_of_range` is the odd one out: it is about Invalid on an
out-of-range conversion rather than Inexact, so it is a *different* gap in the
same family — the conversion path classified `+inf` but not a finite value whose
integer result does not fit. Fixed since: the word range test compared against
`i32::MAX as f32`, which rounds up to 2^31 and so admitted it.

These are distinct from the exception-model findings in
`clanker/cpu/fpu-exception-model-vs-r4400.md`, which are about *when a trap is
taken and what it writes*. These five are about *which flag bits a
correctly-computed result should have raised*.

## Before "fixing" any of these

Setting FI on arithmetic means computing inexactness for every operation in both
engines — including in emitted IR. That is exactly the cost the current design
declined to pay. If IRIX ever needs it, or if the goal becomes a green
cpu-tests run, the work is real and should be scoped deliberately rather than
patched per-test. Do not make the interpreter compute it without making the JIT
compute it identically, or the lockstep/equivalence harnesses will start
failing for a *good* reason.
