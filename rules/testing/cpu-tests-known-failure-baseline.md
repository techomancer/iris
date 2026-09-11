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
