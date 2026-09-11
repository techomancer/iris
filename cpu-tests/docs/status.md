# Status

*Last full run: 2026-09-11, all four matrix cells, on `main`.*
*A full run is a few minutes per cell on a quiet machine.*

> **Run it with `FORCE_BUILD=1 run/matrix.sh`.** `build_iris` reuses any existing
> `build/iris-<cpu>-<engine>` binary, and a stale one fails in a way that looks
> like a test failure rather than a skipped cell (an Aug-21 binary here predated
> the `--cpu` flag and every r4400 cell reported `error: unexpected argument
> '--cpu'`). See `rules/testing/cpu-tests-known-failure-baseline.md`.

## Coverage

| group | tests | what it covers |
|---|---:|---|
| `identity` | 5 | PRId, FIR, cache geometry, Config.K0, TLB size |
| `alu` | 29 | sign extension across the 32/64-bit boundary, overflow traps, shifts, logic, SLT |
| `muldiv` | 18 | mult/div in both widths, HI/LO, the unspecified cases |
| `mem` | 18 | load/store widths, the whole unaligned family at every offset, alignment faults, KSEG0/KSEG1 |
| `branch` | 15 | every conditional, likely-nullification, link registers, delay slots, faults in delay slots |
| `excep` | 15 | traps, reserved instructions, coprocessor usability, EXL/ERET, vector selection |
| `cp0` | 21 | read-only registers, reserved-bit masks, 64-bit access, Count/Compare, LL/SC |
| `tlb` | 10 | entry round-trip over all 48, TLBP, every page size, real translation, V/D bits, ASIDs, refill |
| `fpu` | 88 | see below |
| `cache` | 8 | geometry, tag round-trip, cached/uncached views, I-cache coherency |
| `mips4` | 13 | every MIPS IV addition — computes on R5000, must raise RI on R4400 |
| **total** | **240** | |

### Inside `fpu`

Eight files, because the group grew from 26 tests to 88 and one file could no
longer say what it was about:

| file | tests | what it covers |
|---|---:|---|
| `fpu.c` | 26 | both formats, signed zeros, infinities, NaNs, rounding modes, comparisons, FCSR, FR aliasing |
| `fpu_trap.c` | 14 | trapped exceptions: the Cause/Enable/Flag interaction, what a trap does and does not write, EPC and Cause.BD |
| `fpu_denorm.c` | 9 | denormal operands and results, underflow, and FCSR.FS — the Unimplemented Operation path |
| `fpu_compare.c` | 7 | all sixteen `C.cond` predicates in both formats, signalling vs quiet NaNs, the condition bit's ownership |
| `fpu_vectors.c` | 10 | arithmetic, all four rounding modes, and every integer conversion, against generated tables |
| `fpu_double.c` | 8 | the double-precision paths with no vector: NaN results, bit operations, format conversions |
| `fpu_fr0.c` | 8 | the FR=0 paired register file — the mode every o32 IRIX binary runs in |
| `fpu_breadth.c` | 6 | all 32 registers, signalling-NaN operands, out-of-range conversions, unaligned FP access |

Expectations for `fpu_vectors.c` come from `gen/fpvectors.py`, which computes
them with exact rational arithmetic and cross-checks against the host FPU
before writing (`make vectors`). That is what makes it practical to assert the
*flags* as well as the results, exactly, for every vector.

## Results

Measured on this branch, which carries **tests only** — no emulator changes, so
the suite reports every finding rather than hiding any.

| cell | pass | fail | failing tests |
|---|---:|---:|---|
| R4400, interpreter | 2101 | 61 | 15 |
| R4400, jitv2 | 2101 | 61 | 15 |
| R5000, interpreter | 2071 | 61 | 15 |
| R5000, jitv2 | 2071 | 61 | 15 |

**All four cells fail the same 15 tests**, verified by diffing the failing-test
names, not just the counts. Two things follow, and both are the point of having
a matrix:

- **interp and jitv2 agree exactly**, on both CPUs. A guest-visible ISA suite is
  the cleanest available JIT differential test, and it currently finds nothing.
- **R4400 and R5000 agree on which tests fail.** The 2101-vs-2071 *check* count
  is the MIPS IV split working as intended: the 13 `mips4/` tests all pass on
  both, executing on R5000 and raising Reserved Instruction on R4400, but the
  R4400 path runs fewer individual checks getting there.

The pass counts moved a long way from the 2026-08-19 measurement (2041/121/29
for R4400-interp). Gone since: the 11 `mips4/` failures (finding 2),
`cp0/wired_reserved_bits` and `fpu/fcsr_reserved` (findings 3-4). The remaining
15 are listed below.

`cp0/count_writable` is **load-sensitive**, not a stable pass: it writes
`0x12345678` to Count and checks the top 16 bits, so a busy host can let Count
cross `0x1235_0000` first. A run reporting 2100/61+1 with it failing is equally
healthy. (`cp0/compare_sets_ip7` drifts the *total* by one for the separate
reason noted below.)

The PROM row is unmeasured in this run; it was the same binary reaching the same
answer down the path a bootable disc uses.


Every failure is a recorded finding — see [findings.md](findings.md). The 15
that remain, all `fpu/`, all on every cell:

| failing tests | finding |
|---|---|
| the six `fpu/trap_*` tests | 6 — a trapped exception still writes its result and its Flag bit |
| `fpu/invalid_operations`, `fpu/inexact_flag`, `fpu/overflow_flag`, `fpu/double_invalid_ops` | 3/4/6 — Cause/Flag handling around a raised exception |
| `fpu/vec_arith_single`, `fpu/vec_arith_double`, `fpu/vec_sqrt` | **unclassified — see below** |
| `fpu/cvt_s_d_rounds`, `fpu/cvt_out_of_range` | **unclassified — see below** |

### Fixed since 2026-08-19

- **The eleven `mips4/` failures** (finding 2, "MIPS IV executes instead of
  raising RI" on R4400). All 13 `mips4/` tests now pass on every cell.
  Re-verified in the configuration that specifically provoked it — a binary with
  **both** `mips4` and `jitv2` compiled in, run with `--cpu r4400`, where the
  JIT has MIPS IV emitters available but the model must trap: 2101/61, all
  `mips4/` PASS, including the `CHECK_RI` assertions.
- `cp0/wired_reserved_bits` and `fpu/fcsr_reserved` (findings 3-4).
- `mips4/multi_fp_cc` on R5000 (finding 1) — now reports a skip rather than a
  failure.

### Unclassified: the `fpu/vec_*` and `fpu/cvt_*` failures

These five are **not** explained by the exception-model findings (6-10), which
are all about *when a trap is taken and what it writes*. `fpu_vectors.c` checks
computed results and flags against tables generated by `gen/fpvectors.py` with
exact rational arithmetic. So these are about **what the FPU computes**, a
different and more serious class.

Note this contradicts the paragraph that used to sit here claiming "every vector
in `fpu_vectors.c` ... passes on both CPUs". That was true at the 2026-08-19
measurement and is not true now. Whether the tests or the emulator changed has
not been determined — start by bisecting `fpu/vec_sqrt`, the narrowest of them.

What remains is now entirely FPU, and entirely the same on all four cells. Most
of it is the exception model rather than the arithmetic (see
`rules/testing/fpu-exception-model-vs-r4400.md`): IRIS computes on the host FPU
and delivers the correct IEEE answer where an R4400 would refuse denormals and
trap for software assist, so it looks *more* conformant than the hardware, not
less.

The exception to that — and the part worth investigating — is the five
`fpu/vec_*` / `fpu/cvt_*` failures, which are about computed results. All 128
predicate results in `fpu_compare.c` still pass on both CPUs.

> The total check count drifts by one or two between runs. `cp0/compare_sets_ip7`
> executes two checks when the timer fires and one when it reports a skip, and
> whether it fires depends on the wallclock-anchored counter — see finding 5.

## Not yet done

- **The EFS CD.** The volume-header path is proven — `boot -f dksc(0,2,8)cputest`
  runs the whole suite — but an EFS writer does not exist yet. See PLAN.md §12.
- ~~**The JIT cells.**~~ Run as of 2026-09-11: both `r4400-jitv2` and
  `r5000-jitv2` agree exactly with their interpreter counterparts (same 15
  failing tests). What remains unrun is the **`j2wp` whole-page** variant, which
  `matrix.sh` does not define a cell for.
- **Interrupt delivery.** Everything runs with `Status.IE` clear, so the suite
  tests that interrupts become *pending* but never that one is *taken*. That
  needs a handler that can distinguish an interrupt from a fault, and is the
  most valuable single addition left.
- **Supervisor and user mode.** Everything runs in kernel mode. The privileged-
  instruction and address-space tests that need a mode switch are absent.
