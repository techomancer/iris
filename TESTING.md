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

## Real PROM images

`real_prom_files_load_to_their_payload` (in `src/dev/prom_image.rs`) loads the
real PROM files through the shared loader: Origin, Fuel and Tezro `.img`
containers against the `.payload.bin` beside them, the O2's `ip32prom.image`
and the Octane's `IP30prom.bin`. It searches `ignore/proms` (or
`IRIS_PROM_DIR`) for them and skips when there are none; the images are not in
git.

## Host audio

A sound card can break anywhere between guest memory and the host's
speakers, and a CI runner has no speakers. Two backends hear what a card
sends to the host:

- **`[audio] file`**: the card's host ring played into a WAV file in real
  time, silent where the card was late. It needs no sound card. It cannot
  see cpal, the operating system's sound server or its mixer.
- **The real device**: the card plays into the host's default output and
  the test records it back from a loopback. `IRIS_AUDIO_LOOPBACK` names it
  as `<output host>:<capture source>`: the `[audio] host` the card plays
  through (`alsa`, `pulseaudio` or `default`) and what hears the default
  output. On Linux that is a PulseAudio source, recorded with `parec`. On
  Windows it is `default-render`, cpal's WASAPI loopback of the default
  output. On macOS it is an input device's name, `BlackHole 2ch`, recorded
  through cpal. With the variable unset
  these tests print `SKIPPED: IRIS_AUDIO_LOOPBACK not set` and pass. With it
  set, a missing source or silence fails them.

The tests, for HAL2, O2 audio, Octane RAD1, the Tezro's ICE1724 and the host
output on its own:

- `a_tone_in_guest_memory_reaches_the_host_on_the_real_clock` (file only):
  the card built as a machine builds it, playing a tone from memory laid
  out as its driver lays it out. The file must hold the guest's samples in
  order, each channel on its own side.
- `a_tone_survives_an_underrun_into_a_file` and
  `a_tone_survives_an_underrun_on_the_real_device`: a tone with a different
  frequency on each channel, then a 0.3 s pause that runs the host stream
  dry, then the tone again. The pause is the guest falling behind for HAL2,
  and the guest stopping the output for the others. On both sides of the
  pause each channel must keep its own frequency within 1%. Channels that
  swap after an underrun fail here. Each peak must reach half the tone's
  level. The sound must last no more than 110% of what the card played, at
  the capture's rate, and at least half of it: a busy host can make the
  card catch up in a burst that overflows the host ring. The card's
  underrun counter must move during the pause. HAL2, the ICE1724 and the host
  test play 440 and 880 Hz. The O2's and RAD1's 512-frame rings hold whole
  cycles of 468.75 and 937.5 Hz, and of 430.7 and 861.3 Hz. The ICE1724's
  guest writes eight-slot frames just ahead of the address it reads and looks
  away for 30 ms twice; its tone rises 2 Hz a second, and into a file its
  frames must come out in the guest's order, none replayed.

nextest runs the real-device tests one at a time (the `real-audio` group in
`.config/nextest.toml`): two cards on one output would mix. Under plain
`cargo test` a lock in the test does the same.

To run the real-device tests on Linux without a sound card, give them a
PulseAudio null sink. `scripts/audio-loopback-test.sh` starts a server if
none answers and makes a null sink the default output. It checks that noise
played into the sink comes back from its monitor, then runs a command once
per output host in `IRIS_AUDIO_HOSTS` (default `alsa`; `alsa pulseaudio` adds
iris's own PulseAudio client):

```sh
scripts/audio-loopback-test.sh cargo nextest run -p iris --lib -E 'test(/_on_the_real_device$/)'
```

It needs `pulseaudio`, `pulseaudio-utils` and `libasound2-plugins` (the ALSA
run goes through the pulse plugin). On a desktop it changes the default
output, so run it in a container or a throwaway session. One way is to build
the test binary (`cargo test -p iris --lib --no-run`), then run it in a
container with those three packages installed:
`scripts/audio-loopback-test.sh <test-binary> --test-threads=1 _on_the_real_device`.

What these prove: the card's samples reach the host's real output through
cpal and the sound server, in order, on the right channels, at the right
rate, and the stream recovers from an underrun. What they do not prove:
anything about a real sound card's driver, latency, or whether a person
hears it.

CI runs the ALSA pass on Linux x86-64 in the Test job, and it gates. The
Windows and macOS runners have no sound device, so the Test job gives them
a virtual one first. They do not gate yet: nobody has seen the setup work on
those runners.

- Windows: `scripts/audio-loopback-windows.ps1` installs VB-CABLE, then the
  tests run with `IRIS_AUDIO_LOOPBACK=default:default-render`. VB-CABLE is
  donationware (see the script's header): fetch it from the vendor, don't
  commit or redistribute it.
- macOS: `scripts/audio-loopback-macos.sh` installs BlackHole 2ch and makes it
  the default output and input, then the tests run with
  `IRIS_AUDIO_LOOPBACK="default:BlackHole 2ch"`. BlackHole's source is
  GPL-3.0; the Homebrew cask installs the vendor's binary build, which is
  only run as a tool.

Both scripts install system drivers: use a throwaway machine. If the bare
`a_tone_survives_an_underrun_on_the_real_device` test fails alone, the
virtual device or its setup is broken. On macOS that includes the
runner's microphone permission, which may block the recording or make it
silent: not verified either way. If it passes and a card's test fails, the
card is.

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
| `loom.yml` | pull requests that change a file holding loom models (`cfg(loom)`), nightly on `main` at 04:30 UTC, and by hand from Actions | the `loom_tests` concurrency models, e.g. `intx::loom_tests`; locally `cargo loom` |
| `tsan.yml` | nightly on `main` at 03:30 UTC, and by hand from Actions; gates nothing | the threaded device tests under ThreadSanitizer, to catch data races; the `tsan` nextest profile selects them; locally `RUSTFLAGS="-Zsanitizer=thread" CARGO_PROFILE_RELEASE_PANIC=unwind cargo nextest run -Zbuild-std --target x86_64-unknown-linux-gnu --release --lib --profile tsan` |
| `suites.yml` | pull requests and pushes that touch the CPU, `cpu-tests/`, `bench/` or the test code | the two bare-metal suites on each engine |
| `nightly.yml` | every day at 05:00 UTC when `main` has moved, and by hand from Actions | the `build.yml` packages and IRIX boot, then publishes; a job beside them runs the default-feature tests and the `ci_clock` tests (`cargo ci-clock`) the pull-request run leaves out. It gates no publish |
| `build.yml` | nightly and release | packages for every OS, then boots every IRIX image to the login prompt on each CLI build |

### Which pull requests run what

`rust.yml` tests a pull request on Linux x86-64 (`ubuntu-24.04`), Linux arm64
(`ubuntu-24.04-arm`) and Windows (`windows-latest`), and builds the release
feature set on Linux x86-64 and Windows. A failing test fails the job on all
three. macOS (`macos-26`) joins both jobs on pushes to `main`, and there a
failing test is a red step that leaves the job green, so read the step when the
change touches host-specific code (JIT memory, windowing, audio, sockets).

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

CI builds two feature sets: the default features, and the default plus `jitv2`,
`rex-gpu` and `rex-jit`. `.cargo/config.toml` owns every test command and the
workflows only call its aliases: `cargo ci-test` (the second set, on the Linux
jobs), `cargo ci-test-default`, and `cargo ci-clock` (the nightly clock tests).
`jitv2` implies `tcache`, which compiles out the real-cache path, so the default
build stays: Windows and macOS test it on pull requests and the nightly workflow
tests it on x86-64 Linux. Windows also runs `cargo ci-test`. The Linux job
compiles the `iris` crate fewer times than the others: it skips
`cargo build --workspace` and the SIMD lane step because nextest builds and
runs the same code, and runs the REX3 GPU matcher and replay corpus tests in the
same build (the one ignored filesystem test runs in a second invocation over
that build). Pull requests restore `main`'s
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

`build.yml` boots every IRIX image with each CLI build, on Linux, macOS and
Windows: a platform by image matrix in which each image is listed once, from a
pinned, checksummed stock disk. IRIX 5.3 boots on the serial and graphics
consoles ([clanker/testing/ci-irix53-disk.md](clanker/testing/ci-irix53-disk.md));
IRIX 6.5.22 and 6.5.30 do the same, each against a golden screenshot of its own
login (`scripts/boot-goldens/`). No 6.5.30 Indy disk is published, so 6.5.30
boots an O2 (`--machine o2_ip32`) on its own PROM image, pinned in the matrix.

`iris-ci boot-test` seeds NVRAM on a first headless run, boots with the console
on serial and waits up to 15 minutes for `login:`, then does the same with the
console on the graphics board and a screenshot to match. It reads the seeded
NVRAM back from the PROM and from the booted guest, and checks that iris quits
cleanly. What it seeds and checks is `tests/irix/boot.toml`. Each check is a
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
