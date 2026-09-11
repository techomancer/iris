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
divide-by-zero -> FZ). In particular:

- **Inexact (FI) is never set by arithmetic.**
- **Overflow (FO) is never set by arithmetic.**

Stated in `fpu_update_fcsr`'s doc comment as: *"not worth the IR to compute for
ordinary arithmetic, and nothing in IRIX depends on them."* That trade is sound
for the emulator's actual workload — but it is the direct cause of the standing
cpu-tests failures below, so it is a **known gap, not a regression**.

## The cpu-tests it accounts for

Five of the 15 standing failures (see
`rules/testing/cpu-tests-known-failure-baseline.md`). All five are flag-only:
the computed *result* is correct in every case.

| test | symptom |
|---|---|
| `fpu/vec_arith_single` | `o.flags` got `0x0`, want `0x1`/`0x5` — missing FI |
| `fpu/vec_arith_double` | same |
| `fpu/vec_sqrt` | `o.flags [sqrt.s]` got `0x0`, want `0x1` — missing FI |
| `fpu/cvt_s_d_rounds` | got `0x4`, want `0x5` — has FO, missing FI |
| `fpu/cvt_out_of_range` | `cvt.w.s` of 2^31 does not set FV (though `+inf` does) |

`fpu/cvt_out_of_range` is the odd one out: it is about Invalid on an
out-of-range conversion rather than Inexact, so it is a *different* gap in the
same family — the conversion path classifies `+inf` but not a finite value whose
integer result does not fit.

These are distinct from the exception-model findings in
`rules/testing/fpu-exception-model-vs-r4400.md`, which are about *when a trap is
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
