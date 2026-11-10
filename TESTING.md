# Testing and benchmarking

The suite commands and benchmarking guidance previously in README.md.
Build-feature and configuration combinations are listed in
[FEATURES.md](FEATURES.md); outstanding functionality is tracked in
[TODO.md](clanker/TODO.md).

## Testing and benchmarking

Two bare-metal MIPS suites run on the emulated CPU with no operating system in
the way. They answer different questions and neither replaces the other.

**`cpu-tests/`** — is this instruction correct? ~250 self-checking tests over
ALU, FPU, TLB, caches, exceptions and the MIPS IV additions, one instruction at
a time with clean state. The expectations are validated on real Indys (R4400
and R5000, both passing every check); see
[cpu-tests/README.md](cpu-tests/README.md).

```sh
make -C cpu-tests all    # build the guest; compiler: see cpu-tests/docs/toolchain.md
make -C cpu-tests run    # one run on ../target/release/iris, PASS/FAIL per test, no verdict
cargo test --release [--features jitv2] --test cpu_tests -- --ignored   # judged, both CPUs
```

`make run` only prints. The verdict is the cargo test, which builds the
emulator for the engine you name and compares the failing tests with
`cpu-tests/expected-failures.txt` (see [Strict TAP](#strict-tap)).

**`bench/`** — how fast is this build, and is it still right after ten million
of them? 46 kernels covering integer, FPU, the cache hierarchy, image and video
editing inner loops, compression, and the emulator-only paths (TLB refill,
exception round trip, cache maintenance, uncached I/O). Every kernel checksums
its result against a golden value computed by building the same C natively, so
each run reports an accuracy percentage next to its throughput — and per-kernel
**guest instructions per host second**, which is directly comparable between the
interpreter and jitv2.

```sh
cargo build --release --bin iris-bench
./target/release/iris-bench run         # ~60 s; --quick for about half that
./target/release/iris-bench run --quick
cargo test --release [--features jitv2] --test bench_accuracy -- --ignored   # judged: accuracy on R4400, R5000, R10000

make -C bench all && make -C bench hostbench        # only to change the suite; compiler as above
make -C bench prebuilt      # only after changing a kernel; see bench/prebuilt/README.md
./target/release/iris-bench matrix      # builds and runs every CPU x engine cell
./target/release/iris-bench host        # the same kernels, natively, for the ratio
```

`run` needs **no MIPS toolchain and no build step**: a known-good guest binary
is checked in at `bench/prebuilt/` and linked into `iris`, and the run happens
in-process on a headless machine the emulator builds for itself. That is also
what the GUI's **Benchmark tab** does, on every platform and inside the App
Store sandbox — one button, and the accuracy score sits next to the speed.

Includes Dhrystone 2.1 (DMIPS) and LINPACK 100x100 (MFLOPS), so an emulated
Indy can be put next to published figures for a real one, plus a Whetstone mix
(reported in passes/s — see bench/README.md for why not MWIPS). See
[bench/README.md](bench/README.md) — and
[clanker/testing/benchmark-suite-gotchas.md](clanker/testing/benchmark-suite-gotchas.md)
before adding a kernel.

To ask whether a change made the emulator slower, compare two `iris` builds with
`iris-bench ab --a BASE --b CHANGE`: paired rounds on one guest image, each
kernel judged by an interval and reported inconclusive when the run cannot tell.
See "Comparing two builds" in [bench/README.md](bench/README.md).

`bench/irix/` is the other half: real workloads under a booted IRIX (filesystem,
buffer cache, IRIX's own tools, X on REX3), driven over `iris-ci`.

**GPU parity tests** — with the `wgpu` feature, the host-GPU raster path
(`src/wgpu`, `src/dev/odyssey/wgpu`) is checked against the CPU pipeline: the
same triangles, shaded fragments and clears go through both, and the two board
memories are compared word for word. They need a GPU, or lavapipe, and print
`skipped` without one.

```sh
cargo test --features wgpu --lib wgpu::tests odyssey::wgpu
```

**GL suites in a live guest** — `test/gltest/glsuite.sh` runs every GL check
program inside a booted IRIX (glstate, blendtest, seltest, cicopy, pixtest in
four visuals, the visual, texture and line sweeps, glprim scenes read back
through GL and captured with `scrsave`, and piglit's VPro list), and
`test/gltest/xsweep.sh` restarts the X server in each root visual (PseudoColor
8 and 12, TrueColor 12, 15, 24 and 30, DirectColor 24) and checks a GL scene
in each. Both write to `/shared/results/TAG`; the host-side
`test/gltest/glsuite-compare.sh A B` lists every difference between two runs.
Running both on the CPU and the GPU raster unit and comparing is how the GPU
unit is checked against the reference end to end.
The same suite on a real board, from the same binaries, against a guest
emulating that board is how the model is checked against hardware; the
board's quirks then show up as differences instead of as failures. Cases
that the GL HLE can replay without a guest become unit tests against the
board's own read-backs, captured on a real High IMPACT and kept in
`src/dev/mgras/testdata/real_high_impact_*.txt`: `real_board_lines` draws every
case of `linesweep.sh` and compares it pixel for pixel,
`real_board_smooth_lines_and_points` the antialiased lines and points of
`linetest`, and `real_board_fog` and `real_board_fogtest` glprim's fog scenes
and `fogtest`'s fourteen fog cases (all in `src/dev/mgras/mgras_tests.rs`).

```sh
mkdir -p shared/gltest && cp test/gltest/* shared/gltest/       # piglit: shared/piglit (clanker/testing/piglit-on-irix.md)
tools/irix-sh.py run 'cp /shared/gltest/glsuite.sh /shared/gltest/xsweep.sh /usr/tmp/' \
    '(nohup sh -c "sh /usr/tmp/glsuite.sh gpu; sh /usr/tmp/xsweep.sh gpu" > /usr/tmp/suite.out 2>&1 &)'
test/gltest/glsuite-compare.sh shared/results/cpu shared/results/gpu
```

The suite copies `/shared/gltest` to `/usr/tmp` and runs the programs from
there, so build them once in the guest (`sh irix.sh` in a copy, see the
`irix-gfx-loop` skill) and copy the binaries back into `shared/gltest`. The whole run takes about 45 minutes, most of it piglit. Short
recordings of a failing case (`vpro rec`) replay on both raster units offline:
`the_raster_units_agree_on_recorded_sessions` replays every recording in
`ignore/recordings/gpu-cases` (or `VPRO_PARITY_DIR`) on the CPU unit, the GPU
unit and the GPU unit with live-style slice ends, and compares what the board
wrote to guest memory and its final memory.

## Strict synthetic REX3 replay corpus

The redistributable [REX3 corpus](src/dev/ng1/testdata/README.md) contains
independently authored register programs on zeroed state, not guest captures.
It is functional coverage, not a representative desktop workload or benchmark.
Run the mandatory CPU gate and the manifest, negative, and in-memory
byte-for-byte regeneration tests with:

```sh
cargo test -p iris --lib --features gputrace,rex-jit,rexdiag rex3_replay_testdata -- --nocapture
cargo test -p iris --lib --features gputrace,rex-jit,rexdiag rex3_corpus_ -- --nocapture
```

The first command requires every named case, verifies reads and checkpoints,
and proves actual Cranelift dispatch (no interpreter fallbacks) before
comparing full state with the interpreter. Omitting `rex-jit` or `rexdiag`
allows an interpreter-only test but does **not** satisfy the strict CPU gate.
Linux CI runs both commands and an explicitly selected filesystem security
test; no GPU is required. Discovery rejects unexpected names before reading
content, rejects links and reparse points beneath the checkout root, and caps
the manifest/README at 64 KiB and each recording at 256 KiB. Checkout paths
reached through a symlink or junction ancestor remain valid. The required
`.rex3-corpus-root` anchor is allowed in the directory, but is not a recording
or generated content.

The ordinary negative tests do not write the source tree. To also test real
filesystem symlinks, junctions and atomic replacement, provide a writable
build directory. The test provisions its own anchor; on Windows it also
checks checkout-junction retargeting, non-listable checkout parents (using
`icacls` on a test directory), and attribute-only junction mutation while the
anchor is held, after guard release, and after anchor removal:

```sh
REX3_CORPUS_IO_DIR=target cargo test -p iris --lib --features gputrace,rex-jit,rexdiag rex3_corpus_filesystem_security -- --ignored --nocapture
```

The hardware GPU gate is explicitly ignored unless selected. It requires an
integrated/discrete adapter, checks nonzero handled work and declared
fallbacks, and reports unavailable hardware rather than passing it. The batch
case compares framebuffer state and completed readback at the instant the
display fence is published, before any subsequent idle flush. The second
selected GPU test injects an early signal and requires this check to reject
stale pixels even when the later final state matches:

```sh
REX3_GPU_BACKEND=vulkan cargo test -p iris --lib --features rex-gpu,rex-jit,rexdiag rex3_corpus_gpu_ -- --ignored --nocapture --test-threads=1
```

To deliberately regenerate, set `REX3_CORPUS_OUT` to an existing relative
directory with a pre-existing regular `.rex3-corpus-root` file and run
`rex3_corpus_generate -- --ignored --nocapture` with the strict CPU feature
set. The checked-in corpus has its anchor; for a new output directory create
the anchor before invoking the generator. For example, on PowerShell:

```powershell
$env:REX3_CORPUS_OUT = 'src\dev\ng1\testdata'
cargo test -p iris --lib --features gputrace,rex-jit,rexdiag rex3_corpus_generate -- --ignored --nocapture
```

Review the generated files and rerun both CPU commands. Ordinary tests do not
write the corpus. Generator outputs are individually atomically replaced;
links within the relative output path and existing link destinations are
rejected, while checkout-ancestor aliases are allowed. The generator pins the
anchor for its entire run and never creates or overwrites it; on Windows this
prevents directory reparse conversion while handle-relative writes are active.
To replay a separate recording instead, explicitly select
the optional ignored test, e.g.
`REX3_REPLAY=<file> cargo test -p iris --lib --features gputrace rex3_replay_file -- --ignored --nocapture`.
That single-file run is not corpus coverage.

## Continuous integration

| Workflow | Runs | What it checks |
|---|---|---|
| `rust.yml` | every pull request and push to `main` | `cargo nextest run` on the workspace, a build of the release feature set, the interpreter's hot paths |
| `suites.yml` | pull requests and pushes that touch the CPU, `cpu-tests/`, `bench/` or the test code | the two bare-metal suites on each engine |
| `build.yml` | nightly and release | packages for every OS, then boots IRIX 5.3 to the login prompt on each CLI build |

### Which pull requests run what

`rust.yml` tests a pull request on Linux x86-64 (`ubuntu-24.04`), Linux arm64
(`ubuntu-24.04-arm`) and Windows (`windows-latest`), and builds the release
feature set on Linux x86-64 and Windows. Only the Linux x86-64 test step fails
its job. On the others a failing test is a red step and the job stays green, so
read the step when the change touches host-specific code (JIT memory, windowing,
audio, sockets). macOS (`macos-26`) joins both jobs on pushes to `main`, with
the same informational test step.

macOS is main-only because its runners wait longest for a slot, and a break
that only macOS sees has been rare: from 2026-10-13 to 2026-11-10 no pull
request's macOS job failed on its code where Linux passed.

A draft pull request runs the Linux x86-64 test and the script lint only, so
work in progress does not hold runners that pull requests being landed need.
Marking it ready for review starts the rest. `suites.yml` has no draft rule: it
runs on a draft whose changes touch its paths.

Every job has a time limit, about twice the slowest recent successful run, so a
hung job fails in under an hour instead of holding its concurrency group for
six. A new push cancels the run it replaces on a pull request; each push to
`main` gets its own run and none is cancelled.

The Linux x86-64 test job compiles the `iris` crate fewer times than the
others: it skips `cargo build --workspace` and the SIMD lane step because
nextest builds and runs the same code, and runs the REX3 matcher test through
`cargo test`, which compiles its replay example. Pull requests restore `main`'s
build cache but never save one, so they cannot push `main`'s entries out of the
repository's cache quota. The Linux, AppImage and Windows package builds in
`build.yml` cache only the cargo registry and git checkouts, not their target
trees.

### The bare-metal suites

`suites.yml` builds each guest binary once, in the pinned compiler container,
and each engine's emulator once. The engine (`interp`, `jitv2`) is a cargo
feature and the CPU a runtime flag, so one build runs every CPU. The `suites`
job is, per engine:

```sh
CARGO_UNSTABLE_PANIC_ABORT_TESTS=true IRIS_SUITE_JUNIT=$PWD/target/nextest/ci \
cargo nextest run --profile ci --release --features jitv2 \
    --test cpu_tests --test bench_accuracy --run-ignored all --test-threads 1
```

You can run the same line locally. `--test-threads 1` is there because the
bench cases time the guest, and emulators sharing cores would time each other.
`--profile ci` comes from `.config/nextest.toml`, shared with `rust.yml`: no stop
at the first failure, and nextest's own JUnit file at `target/nextest/ci/junit.xml`. The cases write
their per-guest-test files into the directory `IRIS_SUITE_JUNIT` names; leave it
unset and they write none. `CARGO_UNSTABLE_PANIC_ABORT_TESTS` builds the tests with `panic=abort`, so they
share the one copy of the `iris` library the release binaries link instead of
compiling a second.

Each case runs the emulator in a child process under a timeout, so a hung guest
fails its own case and a crashed emulator does not take the rest down. Speed is
never judged: a shared runner's throughput varies by more than most real
regressions. Bench figures are collected as artifacts and summarised by the
comparison report, for people to read.

### Reading a pull request's results

* **Annotations.** `mikepenz/action-junit-report` reads nextest's `junit.xml`
  (`rust.yml`) or every XML file in `target/nextest/ci/` (`suites.yml`) and marks
  each failing case on the pull request's Checks tab and in the diff. It only annotates; the job's own status is what blocks.
* **Two levels of case.** nextest's `junit.xml` has one case per Rust test: a
  CPU (`r4400`, `r5000`, `r10000_195`) for one engine, with a message naming the
  guest tests or kernels that broke. Beside it, `cpu-tests-<cpu>-<engine>.xml`
  has a case per guest test (`fpu/trap_overflow`), filed under the class
  `cpu-tests.<cpu>.<engine>`, and `bench-<cpu>-<engine>.xml` a case per kernel
  (`img/dct8x8`) under `bench.<cpu>.<engine>`. A guest test or kernel that broke
  fails by name. A listed expected failure, a test the guest skipped on that CPU
  and a kernel with no golden value to check are reported skipped. A problem
  with the whole run (no plan, no `IRIS-CPUTEST-DONE`) is a failed case named
  `run`.
* **Artifacts.** `suites-<engine>` holds each run's console log
  (`target/tmp/bare-metal/`) and the XML files; `bench-results-<engine>` the
  parsed bench results; `bench-report` the comparison table, also in the run's
  summary.

### JUnit from the tools

`iris-bench run --junit FILE` writes the run's kernels as a JUnit report, one
case per kernel: failed on a checksum mismatch, skipped where there was nothing
to check. The accuracy test passes it for you when `IRIS_SUITE_JUNIT` is set.
The reports are built by `src/junit.rs`, which new reporters should use. The
workspace tests, the hot-path checks and the suites need nothing extra: nextest
writes their reports. `iris-ci junit` does the same for the checks of a shell
script, as the nightly boot test uses it.

### Strict TAP

cpu-tests prints TAP, and `tests/common/cputest.rs` accepts a run only when it
is complete. A cell fails on any of:

* no `1..N` plan line before the first test;
* a result number that is not its position (a repeated or missing line);
* fewer results than the plan, or more;
* no `IRIS-CPUTEST-DONE rc=N` line after the last test;
* the emulator killed by a signal, or no answer within 900 s
  (`IRIS_SUITE_TIMEOUT` overrides);
* a test that fails and is not listed for that CPU and engine in
  `cpu-tests/expected-failures.txt`;
* a listed test that now passes. Delete its line in the commit that fixed it.

A derailed run can print no more failures than a healthy one, so a count would
read it as no change. Results are matched by name, which ends at the first
space; a skipped test (`# SKIP`) reports `ok`.

### Nightly boot

`build.yml` boots IRIX 5.3 with each CLI build, on Linux, macOS and Windows,
when the `IRIX53_BOOT_IMAGE_URL` secret is set. `scripts/ci-boot-test.sh` seeds
NVRAM on a first headless run, boots with the console on serial and waits up to
15 minutes for `login:`, then checks that iris quits cleanly. Each check is a
JUnit case, so a failed run is annotated with the check that failed, and the run
keeps the logs. [clanker/testing/ci-unattended-irix-boot.md](clanker/testing/ci-unattended-irix-boot.md)
has the details.

## Disk persistence on exit

`src/exit_persistence_tests.rs` builds every machine (Indy, Indigo2, Indigo2
IMPACT, O2, Octane) with one disk of each kind (raw with an overlay, raw in
place, compressed CHD, CHD with the copy-on-write flag, scratch volume), writes
sectors with WRITE(10), ends the session each way iris can end (guest power-off,
`quit` over the CI socket, `quit` at the terminal console, `main` returning,
drop), with and without a guest reset first, then builds the machine again over
the same files and reads every sector back. Another test fails on any
`process::exit` in the library that does not go through
`workstation::exit_process`, which makes its caller flush the disks first.

```sh
cargo nextest run --release --lib exit_persistence_tests
```

## Interpreter hot paths

`iris-checks/tests/hot_paths.rs` reads a lightning release binary and fails
when interpreter code outside its cold list calls a formatting function, or
when a hit-path cache `fetch` has a symbol of its own. Its tests are ignored in
a normal run; CI runs them on Linux x86-64 after the release build. The rule
they enforce is in [hot-path-inlining-must-be-explicit.md](clanker/perf/hot-path-inlining-must-be-explicit.md).

```sh
cargo build --release --features lightning
cargo test -p iris-checks -- --ignored   # IRIS_HOT_PATH_BINARY picks another binary
```
