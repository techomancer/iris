# cpu-tests: the known-failure baseline, and how to use it

`make -C cpu-tests run` does **not** exit 0 on a healthy tree. The suite has a
standing set of failures, all of them recorded findings about IRIS's FPU
exception model — not regressions. Compare against this baseline before
concluding that a change broke something.

## Baseline (measured 2026-09-10/11, R4400 + interpreter)

```
RESULT: 2101 checks passed, 61 failed  (240 tests)
```

**15 stable failing tests**, all FPU:

```
fpu/cvt_out_of_range          fpu/trap_disabled_control
fpu/cvt_s_d_rounds            fpu/trap_enable_selective
fpu/double_invalid_ops        fpu/trap_inexact
fpu/inexact_flag              fpu/trap_invalid
fpu/invalid_operations        fpu/trap_overflow
fpu/overflow_flag             fpu/trap_overflow_via_i
fpu/vec_arith_double          fpu/vec_arith_single
fpu/vec_sqrt
```

### Plus one flaky test: `cp0/count_writable` (2101/61 <-> 2100/62)

Not a regression indicator in either direction. The test
(`cpu-tests/tests/cp0/cp0.c:296`) writes `0x12345678` to Count and checks the
top 16 bits still read `0x1234`, with its own comment conceding *"It keeps
counting, so only the high bits are stable enough to check."* If Count crosses
`0x1235_0000` between the write and the read — pure host timing — it fails.

Observed: it failed on the one run that competed with seven parallel `cargo test`
suites, then passed on four consecutive runs on an idle machine. So it is
**load-sensitive, not randomly flaky** — the same machine business that trips
the 600s timeout below also stretches the gap between the write and the read.

A run reporting **2100/62 with `cp0/count_writable` in the list is equally
healthy**. Diff the identities: if the only difference is this test, ignore it,
and re-run on an idle machine if you want the clean number.

`cpu-tests/docs/status.md` documents a **second** source of count drift:
`cp0/compare_sets_ip7` runs two checks when the timer fires and one when it
reports a skip, depending on the wallclock-anchored counter (finding 5). It
skipped in both runs measured here, so it did not contribute to the 2100<->2101
delta — but it can move the *total* by one without any test failing, which is
another reason to compare identities rather than counts.

**`make run`'s exit code is the failure count**, not a boolean: `Error 62` here
means 62 failed checks, not "error 62". `Error 124` is different — that is
`timeout(1)`, i.e. the run was killed (see below).

## The 2101/61 number is STALE for the current tree (2026-09-12)

Re-measured `25f86c1` (default features, R4400, interpreter — the same config
the 2101/61 figure claims):

```
RESULT: 2040 checks passed, 124 failed  (240 tests)     28 failing tests
```

So **~124 failures are pre-existing on this tree**, not 61, and the 15-test list
below is incomplete. A change measured against 61 looks like it broke ~65
checks when it broke nothing.

**The reliable procedure is to measure your own baseline, every time:**

```sh
git stash && cargo build --release
make -C cpu-tests run; grep -E "FAIL" cpu-tests/build/serial.log | sort > /tmp/base.txt
git stash pop && cargo build --release
make -C cpu-tests run; grep -E "FAIL" cpu-tests/build/serial.log | sort > /tmp/mine.txt
diff /tmp/base.txt /tmp/mine.txt          # empty == clean
```

That diff is the only trustworthy signal here. It is what confirmed the
T1/T2/T3/C1 batch clean (identical identities) after the raw count had already
sent one investigation down the wrong path.

**Read the last PASSing line before the stall.** A run that times out names the
culprit for free: the T3 hang stopped right after `identity/cache_geometry`, and
the next test in the file is `identity/config_k0` — precisely what the change
touched. Two reverts were wasted on a different suspect before reading it.

## The baseline is CONFIG-SPECIFIC — check your binary first

**Measured on R4400 + interpreter**, i.e. `cargo build --release` with *default
features*. `make -C cpu-tests run` just uses whatever `../target/release/iris`
happens to be, and says nothing about how it was built — so if you last built
with `--features lightning,rex-jit,jitv2,j2wp,tcache` (the recommended run
config), `make run` silently measures **that** binary against an
interpreter-only baseline.

Observed 2026-09-12 doing exactly this: **2038/126 with 29 failing tests**
against the documented 2101/61 with 15. The 14 extra are not regressions; they
are what the JIT build scores. Recognisable shapes:

```
mem/lwr_all_offsets        sign-extension differs from the interpreter
cache/hit_inv_discards     invalidate leaves stale data visible
excep/cop2_unusable        CU2-set case takes an exception it shouldn't
mips4/recip_rsqrt{,_d}     \  present whenever the `mips4` emitters are
mips4/fp_cond_move_{s,d}   /  compiled in
```

**Before comparing, rebuild to match:**

```sh
cargo build --release                     # default features == the baseline
make -C cpu-tests run
```

Or point the runner at a specific binary without disturbing `target/`:

```sh
IRIS=/path/to/iris-plain LOG=build/serial-plain.log \
  cpu-tests/run/run-local.sh cpu-tests/build/cputest.elf
```

Recording this because the failure mode is quiet and convincing: a
count-and-identity diff against the wrong binary produces a long, specific,
entirely bogus regression list, and the extra tests look plausible enough
(memory, cache, exceptions) to send someone bisecting a change that had nothing
to do with them.

## Compare identities, not just the total

A matching count with a *different* test failing is a regression that a
count-only comparison hides. Save the list and diff it:

```sh
grep -E "FAIL" cpu-tests/build/serial.log | awk '{print $1}' | sort > /tmp/fails.txt
diff /tmp/fails-baseline.txt /tmp/fails.txt
```

## Why these fail — mostly one documented cause

Most of the FPU entries are the exception-model simplifications written up in
`rules/testing/fpu-exception-model-vs-r4400.md` and at length in
`cpu-tests/docs/findings.md` §6-§10. In short: IRIS computes on the host FPU
and is *more* IEEE-conformant than an R4400, which refuses denormals and traps
for software assist. The six `fpu/trap_*` failures are finding 6 — a trapped
exception still writes its result and its Flag bit, where hardware stores
nothing. `fpu/inexact_flag`/`overflow_flag`/`invalid_operations` are the same
family (Cause/Flag handling, findings 3-4).

None of this affects IRIX, which is why it has stayed this way: nothing in the
system relies on the FP-assist path.

The other five — `fpu/vec_arith_single`, `fpu/vec_arith_double`, `fpu/vec_sqrt`,
`fpu/cvt_s_d_rounds`, `fpu/cvt_out_of_range` — are a **different** gap and were
passing at the 2026-08-19 measurement. In all five the computed *result* is
correct and only the FCSR flag bits are wrong (missing Inexact, or missing
Invalid on an out-of-range finite conversion). Cause: FPU flags are computed
from bit patterns rather than read from the host FPU (so both engines can agree
exactly), and Inexact/Overflow are deliberately not computed for ordinary
arithmetic. See `rules/jitv2/fpu-flags-are-computed-not-read.md` — which five
source comments referenced but which did not exist until it was written up.

## `cpu-tests/docs/status.md` is stale — do not use it as the baseline

That file's results table is dated **2026-08-19** and reports
`R4400, interpreter: 2041 pass / 121 fail / 29 tests`. 99 commits have landed
since. Today's measurement is 2100/62/16 — *better*, and the deltas are real:

- The **11 `mips4/` failures are gone** (finding 2, "MIPS IV executes instead of
  raising RI"). Those are exactly what cpucritique's **J6** describes, so
  whatever fixed them, the R4400-profile decode gate is now behaving in this
  configuration.
- `cp0/wired_reserved_bits` and `fpu/fcsr_reserved` no longer fail.
- But `fpu/vec_arith_single`, `fpu/vec_arith_double` and `fpu/vec_sqrt` **do**
  fail, while status.md claims "every vector in `fpu_vectors.c` ... passes on
  both CPUs". Also newly failing vs the doc: `cp0/count_writable`,
  `fpu/cvt_*`, `fpu/double_invalid_ops`.

So the doc is wrong in both directions and should be re-measured, not trusted.
The `fpu/vec_*` and `fpu/cvt_*` entries are the ones worth investigating — they
are about *computed results*, not the exception model, which is a different and
more serious class than the rest of this list.

## `run/matrix.sh` silently reuses stale binaries

`build_iris` skips the build whenever `cpu-tests/build/iris-<cpu>-<engine>`
already exists, unless `FORCE_BUILD=1` is set:

```sh
if [[ -x "$target" ]] && [[ -z "${FORCE_BUILD:-}" ]]; then
    echo "  (reusing $target)"; return 0
fi
```

Those binaries are **not** cleaned by `make clean` (they live under `build/`,
which is removed — but only if you run it) and they can be arbitrarily old.
Observed here: `iris-r4400-interp` and `iris-r4400-jitv2` dated Aug 21 against a
Sep 11 tree, i.e. ~3 weeks and a hundred commits stale.

The failure mode is not subtle but it is easy to skim past — the cell logs
contained only:

```
error: unexpected argument '--cpu' found
```

because `--cpu` was added to the CLI *after* those binaries were built (both
cache models are now compiled into every build and selected at runtime; the
`r5k` cargo feature is vestigial for model selection). The matrix reported the
cells as failures rather than as "did not run".

**Always run `FORCE_BUILD=1 run/matrix.sh`** when the tree has moved, or
`rm -f cpu-tests/build/iris-*` first.

## `Error 124` means the run was killed, not that tests failed

`run/run-local.sh` wraps IRIS in `timeout 600`, and its own comment calls that
"a hang detector, not a performance budget... a few minutes of emulated time on
a quiet machine and rather more on a busy one."

**Running cpu-tests concurrently with several `cargo test` suites will trip it.**
Observed in this session: an error-124 run alongside seven parallel cargo
suites, which completed normally (2100/62) once the machine was quiet. A
timeout is a *no-result*, not a failure — re-run it on an idle machine before
drawing any conclusion.

The prebuilt `cpu-tests/build/cputest.elf` runs fine without a MIPS
cross-compiler in `PATH`; the toolchain is only needed to *rebuild* it. A stale
ELF is not the reason for a timeout.
