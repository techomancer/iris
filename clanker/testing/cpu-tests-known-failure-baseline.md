# cpu-tests: the known-failure baseline, and how to use it

`make -C cpu-tests run` does **not** exit 0 on a healthy tree: the suite has a
standing set of failing tests, each an IRIS bug that passes on a real Indy. The
baseline is the per-cell list in `cpu-tests/expected-failures.txt`, and

```sh
cargo test --release [--features jitv2] --test cpu_tests -- --ignored
```

judges a run on each CPU against it by test name (`tests/cpu_tests.rs`). CI
runs the same test under `cargo nextest` and annotates each failing guest test
on the pull request. A fix that makes a listed test pass fails it until its
line is deleted, in the same commit. A run must also be whole (plan, numbering,
`IRIS-CPUTEST-DONE`); the rules are in
[TESTING.md](../../TESTING.md#strict-tap).

Names, not counts. CI used to compare the number of failing checks with a
per-cell constant, which cannot see one test starting to fail as another
starts to pass, or a run that derailed into the same count.

## The baseline is specific to the engine and the CPU

The cargo test builds the emulator for the features on its command line, so
it always runs the engine it names. `make run` does not: it uses whatever
`../target/release/iris` is, and says nothing about how it was built. A jitv2 build run as `interp` shows `cache/hit_inv_discards` as
a surprise, and a default build run as `jitv2` shows it as a test that now
passes; `--cpu r5000` drops `mem/lwr_all_offsets` and the `mips4/` tests. These
look like regressions in memory, cache and exceptions and have sent people
bisecting changes that touched none of them. Build for the cell:

```sh
cargo build --release                     # interp
cargo build --release --features jitv2    # jitv2
```

or pass `IRIS=/path/to/iris` to `make run`.

## A timeout is no result, not a failing test

The cargo test kills IRIS after 900 s (`IRIS_SUITE_TIMEOUT` overrides it) and
fails with "timed out"; `run/run-local.sh` exits 124 after `TIMEOUT` seconds
(600 by default). A run next to several parallel `cargo test` suites can hit
either. Rerun on a quiet machine. The exit code of a finished run is its
failing-check count, saturated at 100.

## Where the reasons live

The header of `expected-failures.txt` groups the failures by cause. The FPU
ones are written up in `cpu-tests/docs/findings.md` and
`clanker/jitv2/fpu-flags-are-computed-not-read.md`. `cpu-tests/docs/status.md`
is a 2026-08-19 snapshot; do not use it as a baseline.
