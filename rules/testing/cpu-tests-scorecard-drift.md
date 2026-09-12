# The cpu-tests scorecard in README.md is stale

`cpu-tests/README.md` publishes an "Expected results" table. Measured on
2026-09-10 from a clean build of the current tree, both rows are wrong:

| | README says | actually measured |
|---|---|---|
| R4400 | 2041 pass / 121 fail / 29 failing tests | **2101 / 61 / 15** |
| R5000 | 2095 pass / 37 fail / 13 failing tests | **2071 / 61 / 15** |

Both cells now fail the **same 15 tests**, and every one of them is under
`fpu/`: `invalid_operations`, `inexact_flag`, `overflow_flag`, the six `trap_*`,
the three `vec_*`, `double_invalid_ops`, `cvt_s_d_rounds`, `cvt_out_of_range`.

An identical failing set across two different CPU models is worth more than the
raw count: it points at one systemic problem in FP exception and flag handling
rather than fifteen independent bugs. The R5000 row having *more* failures than
the README claims, while R4400 has far fewer, also means the numbers did not
simply drift in one direction — do not treat the table as a baseline to diff
against.

**Both are now superseded again**: the suite was corrected against real hardware
on 2026-09-11, and IRIS `--cpu r4400` reports **2040 / 124 across 27 tests**
against an Indy's 240/240. See `cpu-tests/docs/findings.md`.

Check the numbers before concluding a change regressed something, and re-measure
rather than trusting the table. `run/matrix.sh`'s header comment has a second
stale claim in the same area — see
[cpu-model-is-runtime-not-compile-time.md](cpu-model-is-runtime-not-compile-time.md).

Note the failure *count* is checks, not tests: one failing test contributes
several. `IRIS-CPUTEST-DONE rc=` saturates at 100, so read the `RESULT:` line.

## Related

- [cpu-count-is-wallclock-derived.md](cpu-count-is-wallclock-derived.md) — one
  further check flakes run-to-run, so ±1 is expected noise.
