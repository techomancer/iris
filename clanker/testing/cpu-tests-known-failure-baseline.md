# cpu-tests: the known-failure baseline, and how to use it

`make -C cpu-tests run` does **not** exit 0 on a healthy tree: the suite has a
standing set of failing tests, each an IRIS bug that passes on a real Indy. The
baseline is the per-cell list in `cpu-tests/expected-failures.txt`, and

```sh
make -C cpu-tests check-run CPU=r4400 ENGINE=interp   # IRIS=<a build for it>
```

judges a run against it by test name (`cpu-tests/run/check.py`). CI runs the
same target. A fix that makes a listed test pass fails check-run until its line
is deleted, in the same commit.

Names, not counts. CI used to compare the number of failing checks with a
per-cell constant, which cannot see one test starting to fail as another
starts to pass, or a run that derailed into the same count.

## The baseline is specific to the engine and the CPU

`make run` uses whatever `../target/release/iris` is, and says nothing about how
it was built. A jitv2 build run as `interp` shows `cache/hit_inv_discards` as
a surprise, and a default build run as `jitv2` shows it as a test that now
passes; `--cpu r5000` drops `mem/lwr_all_offsets` and the `mips4/` tests. These
look like regressions in memory, cache and exceptions and have sent people
bisecting changes that touched none of them. Build for the cell:

```sh
cargo build --release                     # interp
cargo build --release --features jitv2    # jitv2
```

or pass `IRIS=/path/to/iris` to check-run.

## `run/matrix.sh` reuses binaries

`matrix.sh` skips the build when `cpu-tests/build/iris-<cpu>-<engine>` exists.
After the tree moves, run `FORCE_BUILD=1 run/matrix.sh`; an old binary can fail
every cell with `unexpected argument '--cpu'`, which reads as a failure rather
than "did not run".

## `Error 124` is the timeout, not a failing test

`run/run-local.sh` kills IRIS after `TIMEOUT` seconds (600 by default, 900 under
check-run). A run next to several parallel `cargo test` suites can hit it.
That is no result: rerun on a quiet machine. A killed run reports fewer tests
than its plan and check-run fails it for that. The exit code of a finished run
is its failing-check count, saturated at 100.

## Where the reasons live

The header of `expected-failures.txt` groups the failures by cause. The FPU
ones are written up in `cpu-tests/docs/findings.md` and
`clanker/jitv2/fpu-flags-are-computed-not-read.md`. `cpu-tests/docs/status.md`
is a 2026-08-19 snapshot; do not use it as a baseline.
