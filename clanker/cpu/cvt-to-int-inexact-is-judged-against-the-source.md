# A float-to-integer conversion's Inexact is judged against the source, in both engines

Status: fixed 2026-11-10.

## Symptom

`TRUNC`, `ROUND`, `CEIL`, `FLOOR`, `CVT.W` and `CVT.L` never set FCSR
Inexact (Cause bit 12, Flag bit 2), and an enabled Inexact trap never fired
on them. `trunc.w.s` of 3.7 left FCSR at 0. The only test that asserted it,
`lockstep_fpu_trunc_w_s_non_integer_value_sets_inexact`, builds only with
`jitv2_lockstep` and had been failing unnoticed.

## Cause

Two independent places, which agreed with each other and so passed every
equivalence test:

- `cvt_to_int_and_commit` (interpreter, and the JIT's callout) took Inexact
  from `cvt_f32/f64_to_int_by_value`, which is handed the value *already
  rounded to an integer* and compares the integer result with that. They are
  always equal.
- jitv2's inline fast path in `emit_fcvt_to_int` reported no Inexact at all,
  and `cvt_to_int_case` in `equiv_test.rs` masked the Inexact bits out of its
  comparison. Both dated from when the interpreter's answer came from host
  FPU flags and differed between debug and release builds.

## Rule

Inexact for these instructions is `rounded != source`, by value, never from a
host flag and never from the rounded intermediate. The interpreter computes
it in `cvt_to_int_and_commit`; the inline path computes the same comparison.

The inline path (James's ad61be60) takes the callout for every source while
the Inexact enable is set, so an inexact result traps there. Otherwise it
stores `(fcsr & !Cause) | (rounded != x ? Cause.I|Flag.I : 0)` without a
branch. Testing the enable alone, folded into the range check that already
branches, measured cheaper than testing exactness as well.

## Alternative not taken

A version that compared the new FCSR with the old one and stored only on a
difference (in a cold block, which also took the callout for an enabled
trap) kept enabled-Inexact code inline. It cost about 1 ns per conversion
on `cvt_perf_probe` over the old fast path, and it added two cold blocks to
every conversion site. Code that enables the Inexact trap is rare, so the
simpler version won. A first version that branched on exactness, with the
destination write in both arms, cost up to 25% on `trunc` because of
mispredicts. The benchmark suite cannot see any of this: its image holds
four such instructions.

`cvt_perf_probe` (an ignored test in `equiv_test.rs`) has exact (79.0) and
fractional (3.7) source cases for measuring this path.

## Checked with

With ad61be60's inline path and the full-FCSR tests:
`cargo test --release --features jitv2 --lib cpu::` (823 passed) and the same
with `jitv2,jitv2_lockstep,developer` (850 passed).
