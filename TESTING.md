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
make -C cpu-tests all && make -C cpu-tests run   # compiler: see cpu-tests/docs/toolchain.md
cargo test --release [--features jitv2] --test cpu_tests -- --ignored   # judged, both CPUs
```

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
cargo test --release [--features jitv2] --test bench_accuracy -- --ignored   # accuracy, every CPU

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
