# Testing and benchmarking

How to run the test suites and benchmarks, and what CI runs.
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
cargo test --release [--features jitv2] --test cpu_tests -- --ignored   # judged, every CPU
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
cargo test --features wgpu --lib -- wgpu::tests odyssey::wgpu
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
`IRIS_PROM_DIR`) for them and skips when there are none. The SGI PROMs
themselves are built in (`data/proms`), so tests that only need a
PROM no longer look for a file. `cargo run -p iris-prom -- show FILE` prints what a PROM
file holds (its kind, version and segments, each checked) and `extract` writes its
payload or one segment, such as the BaseIO PROM inside the IP35 file; `download` and
`pack` rebuild `data/proms` from the catalog.

## Repeated PROM boots

A PROM that fails one start in a hundred needs hundreds of starts to show it.
`tools/prom-boot-loop.py` starts iris with one config again and again, watches
the serial log the config names, and counts each start as ok (the PROM menu,
`Option?`), failed (`SL-9600`, the O2's serial loader, or `EXC`, a PROM
exception; both can be changed with `--good` and `--bad`), timed out or
exited. It stops iris with SIGTERM once a start is decided, so use it only on
a config whose disks the guest cannot write: none, or `overlay = true` on each.
The serial log and iris's output of every start that was not ok are kept. The
script's help has an O2 config with no disks that reaches the menu in about
five seconds. Races between the CPU thread and a device thread show up more
often when the host is busy, so run it beside a few busy loops.

The O2 PROM clears memory with its rendering engine and reuses it after one
status read, so it reaches its menu only while that read waits for the
engine's memory transfers. `IRIS_IP32_RE_LAG_US` holds each transfer on the
engine's thread for that many microseconds, which opens the window on any host
without loading it:

```bash
IRIS_IP32_RE_LAG_US=5000 tools/prom-boot-loop.py --iris target/release/iris --config o2.toml --count 10 --timeout 60
```

Every start must reach the menu, at 1500, 5000 or 20000 alike; a start that
stops at `SL-9600-8E>` or hangs with no output means a status read no longer
waits for the engine.

## Host audio

A sound card can break anywhere between guest memory and the host's
speakers, and a CI runner has no speakers. Two backends hear what a card
sends to the host:

- **`[audio] file`**: the card's host stream played into a WAV file in real
  time, silent where the card was late. It needs no sound card. It goes
  through the stream's prebuffer and rate conversion (no clock following,
  so 48 kHz is copied exactly), but cannot see the host's audio system, its
  sound server or its mixer.
- **The real device** (Linux): the card plays into the host's default output
  and the test records it back from a loopback. `IRIS_AUDIO_LOOPBACK` names
  it as `<output host>:<capture source>`: the `[audio] host` the card plays
  through (`default` or `pulse`) and the PulseAudio source that hears the
  default output, recorded with `parec`. With the variable unset these tests
  print `SKIPPED: IRIS_AUDIO_LOOPBACK not set` and pass. With it set, a
  missing source or silence fails them. There is no recorder for Windows or
  macOS: the WASAPI and CoreAudio backends are exercised by the `iris-audio`
  crate's own examples (`iris-audio/README.md`).

The tests, for HAL2, O2 audio, Octane RAD1, the Tezro's ICE1724 and the host
output on its own:

- `a_tone_in_guest_memory_reaches_the_host_on_the_real_clock` (file only):
  the card built as a machine builds it, playing a tone from memory laid
  out as its driver lays it out. The file must hold the guest's tones at
  full level, each channel on its own side.
- `a_tone_survives_an_underrun_into_a_file` and
  `a_tone_survives_an_underrun_on_the_real_device`: a tone with a different
  frequency on each channel, then a 0.3 s pause that runs the host stream
  dry, then the tone again. The pause is the guest falling behind for HAL2,
  and the guest stopping the output for the others. On both sides of the
  pause each channel must keep its own frequency within 1%. Channels that
  swap after an underrun fail here. Each peak must reach half the tone's
  level. The sound must last no more than 110% of what the card played, at
  the capture's rate, and at least half of it: a busy host can make the
  card catch up in a burst larger than the host stream queues. The card's
  underrun counter must move during the pause. HAL2, the ICE1724 and the host
  test play 440 and 880 Hz. The O2's and RAD1's 512-frame rings hold whole
  cycles of 468.75 and 937.5 Hz, and of 430.7 and 861.3 Hz. The ICE1724's
  guest writes eight-slot frames just ahead of the address it reads and looks
  away for 30 ms twice; its tone rises 2 Hz a second, and the frames the
  card hands the host stream must be in the guest's order, none replayed.

nextest runs the real-device tests one at a time (the `real-audio` group in
`.config/nextest.toml`): two cards on one output would mix. Under plain
`cargo test` a lock in the test does the same.

To run the real-device tests on Linux without a sound card, give them a
PulseAudio null sink. `scripts/audio-loopback-test.sh` starts a server if
none answers and makes a null sink the default output. It checks that noise
played into the sink comes back from its monitor, then runs a command once
per output host in `IRIS_AUDIO_HOSTS` (default `default`):

```sh
scripts/audio-loopback-test.sh cargo nextest run -p iris --lib -E 'test(/_on_the_real_device$/)'
```

It needs `pulseaudio` and `pulseaudio-utils`. On a desktop it changes the
default output, so run it in a container or a throwaway session. One way is
to build the test binary (`cargo test -p iris --lib --no-run`), then run it
in a container with those packages installed:
`scripts/audio-loopback-test.sh <test-binary> --test-threads=1 _on_the_real_device`.

What these prove: the card's samples reach the host's real output through
the host stream and the sound server, on the right channels, at the right
rate, and the stream recovers from an underrun. What they do not prove:
anything about a real sound card's driver, latency, or whether a person
hears it.

CI runs this on Linux x86-64 in the Test job, and it gates. If the bare
`a_tone_survives_an_underrun_on_the_real_device` test fails alone, the null
sink or its setup is broken. If it passes and a card's test fails, the card
is.

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

The **resident Newport raster** has CPU-only limit, status and GFIFO gate
tests, plus ignored hardware parity tests. Production activation uses the same
backend; the hardware tests retain direct injection so they can isolate plane
authority, ordering and failure behavior. The hardware
tests require the named hardware adapter/backend rather than silently skipping;
they compare both complete linear planes and the final architectural
`Rex3Context` with the CPU oracle, including overlapping fills, clipping,
masks, tile boundaries, retries, and offscreen columns. They exercise the real
GFIFO, reset, and partial-prefix restore. Scissor/CID clip modes outside the
gated slice are rejected without advancing context or falling back. Explicit
observation selects RGB, auxiliary or both and returns a snapshot; installing
and acknowledging a CPU shadow is separate. These tests do not depend on the production configuration selector; live
activation is validated separately below.

The opt-in CPU-plane residency tripwire copies each board shadow into its own
page-aligned OS mapping, seeds the GPU, then protects both complete mappings.
It counts slice accesses and also catches previously cached raw pointers:
child processes deliberately read and write the first and last words of each
protected plane and must terminate with an access violation or segmentation
fault. The GPU no-touch test holds both guards across real GFIFO fills, a
display fence, idle, direct test-only composition and a GPU frame-consumer
submission. It checks zero CPU-plane accesses, raw-plane uploads, readbacks,
and staging-map attempts until an explicit observation, then compares both
full GPU planes with the CPU reference and checks attempted/success bytes by
reason. The mappings are never enabled for ordinary tests unless explicitly
requested.

The gated direct-frame tests also drive an explicit GFIFO marker, including a
later-draw race and an XMAP fence published at the cut. They compare composed
palette/gamma/cursor/topscan output with the software compositor, exercise
stale-frame and capture failures, and use protected CPU planes while checking
that ordinary presentation causes no plane traffic or readback. Headless
tests request composed frames without a renderer, compare a real
encoded screenshot with the CPU compositor, and verify an actual loopback VNC
raw update. Failed captures must leave requests pending and not advance the
published frame sequence. Additional cases inject loss into the shared
asynchronous device-error sink before and after submission and while idle,
including a full GFIFO and a full ordinary machine event queue. They require
a single first-fault notification and reject restart.
These tests opt in to the manual display seam rather than production startup,
so failures remain attributable to scanout/capture behavior. Production tests
and live runs cover automatic startup, board-level snapshot/monitor consumers
and the visible same-device window. On Windows and Linux, an additional
hardware test sends two
ordered draws and a resize through the common wgpu presenter on a native
window. It explicitly reads the presenter's unpacked frame texture, including
padded rows after the resize, and compares it pixel-for-pixel with the software
compositor. This diagnostic readback is not part of ordinary presentation and
does not inspect the final swapchain or on-screen pixels. The window test is
excluded on macOS, where a separate main-thread harness is not yet provided.
Resident refresh skips CPU DID decoding. A GPU kernel expands changed VC2 DID
tables before scanout, and traffic assertions require zero decoded-map uploads
and less than 128 KiB for the complete initial display-state transfer. The
hardware oracle covers disabled DID, malformed line pointers, line-table
termination with stale-row preservation, zero scanline length, and width and
height changes on both backends. It explicitly reads the GPU DID metadata for
byte-for-byte comparison with the CPU map; ordinary frames do not perform that
test readback. The tests also cover scanout replacement, reset and restore
epochs, switching displays, inactive display controls, and rejection of
unreconstructible initial DID history or a different GPU device.
The explicitly selected compatibility case does the opposite by design: it
requires one successful, attributed
`Compatibility` observation of both complete planes (16 MiB), composes from a
separate CPU cache, leaves the board shadows stale and protected, and matches
the software oracle. It is not a fallback from direct presentation.
Three CPU command-ABI tests lock the 64-byte command and 256-byte state-slot
packing, normalized fill capture, and bounded transactional payload ranges.
The hardware fill cases consume the same records through WGSL and retain
overlapping command order across bounded batch reuse.
Four flat-fill normalizer tests additionally compare blocks, spans, consistent
reverse-direction resident blocks and active `COLORVRAM` fast clears with the
CPU walker, including exact context advancement. The dual-backend hardware
fill case includes the reverse fast-clear command.
Two line-normalizer tests compare payload coverage and exact context
completion with the CPU walker across all octants, both BRESROUND tie modes,
fractional endpoints, skip-first/last and antialiased endpoint filtering.
Their HostW coverage also checks transactional integer, fractional and
antialiased completion, mixed shade/pattern state, single-step continuation,
`LENGTH32` outer retries and restored fractional endpoint state. The hardware
line case executes ordinary integer, fractional and antialiased commands
against complete seeded RGB and auxiliary planes on both backends.
The hardware clipping case compares complete planes after offset blocks and
lines using SMASK0, OR-combined SMASK1-4 and CIDMATCH. It also verifies that a
CID write earlier in one ordered batch affects a later overlapping RGB
command without a submission boundary.
The hardware pixel-format case covers RGB/RGBA and CI depths 4, 8, 12 and 24,
both `DBLSRC` slots, OLAY/PUP/CID packing, alpha pass and rejection, all logic
operations used by the matrix, WRMASK, Bayer dithering and YFLIP blocks and
lines. Complete seeded RGB and auxiliary planes are compared with the CPU
oracle on both hardware backends.
The hardware blending case runs the complete 6x6 matrix of defined source and
destination factors with both `BLENDALPHA` behaviors, then covers
`BACKBLEND`, 4/8/12-bit expansion and recompression, dithering, `DBLSRC`,
OLAY/PUP/CID packing and a YFLIP line. Both complete planes and final context
state are compared with the CPU oracle.
The hardware pattern case covers transparent and opaque Z and line patterns,
combined tests, `LSREPEAT`/`LSLENGTH`, `LSADVLAST`, row reset with a continuing
repeat counter, clipped phase, reverse blocks, YFLIP, fractional and
antialiased lines, and `COLORBACK` through the ordinary pixel pipeline. It
compares both complete planes and final pattern context on both backends.
The hardware screen-copy case covers safe and feedback-producing horizontal
and vertical overlap orders, packed-slot copies, clipping, logic/WRMASK
feedback, ignored compare and pattern tests with state advancement,
SKIPFIRST, auxiliary planes, spans and all three line families. It compares
both complete planes and final context with the CPU walker on both backends.
The hardware continuation case covers every block stop-bit combination,
partial `XSAVE` rows, reverse YSTRIDE, fractional and no-plane draws,
LENGTH32 spans and lines, single-step integer/fractional/antialiased lines,
partial overlapping copies, NOOP, and an armed non-host HostRW token's
no-progress retry. It compares complete planes and the full context after
each direct segment and after the board-level retry on both backends.
Two CPU HostW walker tests compare transactional completion with the generic
walker, including restored noncanonical line-pattern phase, and pin the
FASTCLEAR+ALPHAHOST field consumption and retirement. A separate
interpreter/Cranelift oracle pins
mixed host-source selection. The hardware HostW case stages PIO and multiword BLOCK/SPAN and line transfers
and compares both complete planes and full context with the CPU oracle. Its
segment matrix covers COLORHOST, ALPHAHOST and both, packed and unpacked
4/8/12/32-bit fields, RWDOUBLE, SWAPENDIAN, residual input state, row-boundary
continuation, PIO replication, clipping, logic, blending, WRMASK,
RGB/auxiliary planes and board-level outer GO integration. It also covers
transparent and opaque combined patterns, row reset, shade with host color
and/or alpha, endpoint skips, LENGTH32 continuation and increasing and
decreasing LR-only semantics. The line matrix adds all integer octants,
fractional and antialiased lines, BRESROUND state, restored fractional
endpoint filtering across multiword `LENGTH32` retries, single-step
continuation, YFLIP/XYOFFSET, shade, transparent and opaque patterns, partial
clipping, CID and alpha rejection, packed/unpacked input and command-free
no-progress retirement. Canonical FASTCLEAR+ALPHAHOST fill and line records
prove that HostRW fields are consumed while conflicting shade, pattern, alpha,
logic, blend and dither state is bypassed; complete planes also cover clipping
and WRMASK. A 6x4 `glPixelZoom`-style source grid verifies that
24 full-block PIO records remain unsubmitted until one explicit batch flush,
then match the CPU plane. Overlapping HostW-to-ordinary-to-HostW draws verify
both ordering boundaries, while the board-level one-slot backend proves that
capacity exhaustion submits the accepted prefix and retries the next HostW
record without duplicated context advancement. The producer reliability
coverage refuses a bulk VDMA transfer with `BUS_BUSY` while the GFIFO lacks
room, proves that neither token nor payload prefix was published, and retries
the complete transfer exactly once. Concurrent scalar and batch producers run
2048 records apiece across repeated ring wraps. The hardware oracle alternates
24 PIO and VDMA HostW records through the board, then uses a four-word payload
arena, three state slots and two reusable GPU sets to force repeated payload
and state exhaustion. Complete planes, context, counters and stream positions
must remain CPU-identical on both backends.

Two focused CPU tests pin resident HostR board gating, sticky failure and
batch retirement across PIO, scalar DMA and bulk DMA ports. The hardware
observation case runs the existing CPU HostR walker against one reusable
selected-plane snapshot and checks RGB/auxiliary cache reuse. The board oracle
writes distinct GPU-resident RGB words, verifies PIO read-then-advance and
bulk-DMA prime-then-read values, proves repeated GOs use one attributed
full-plane map, then writes the selected plane and requires exactly one new
map. The board CPU framebuffer remains stale throughout.

```sh
cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag resident_ -- --test-threads=1
cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag dev::ng1::rex3_flat_fill::tests -- --test-threads=1
cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag dev::ng1::wgpu::line::tests -- --test-threads=1
cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag dev::ng1::wgpu::host::tests -- --test-threads=1
cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag mixed_host_sources_match_between_interpreter_and_jit -- --test-threads=1
cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag test_hostr_board_ -- --test-threads=1
cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag line_pattern_closed_form_matches_scalar_advancement -- --test-threads=1
cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag dev::ng1::wgpu::command::tests -- --test-threads=1
cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag direct_ng1_scanout_ -- --test-threads=1
REX3_RESIDENT_BACKEND=vulkan cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag resident_gpu_ -- --ignored --nocapture --test-threads=1
REX3_RESIDENT_BACKEND=vulkan cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag dev::ng1::wgpu::direct_tests -- --ignored --nocapture --test-threads=1
REX3_RESIDENT_BACKEND=dx12 cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag resident_gpu_ -- --ignored --nocapture --test-threads=1
REX3_RESIDENT_BACKEND=dx12 cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag dev::ng1::wgpu::direct_tests -- --ignored --nocapture --test-threads=1
```

`rexdiag` makes the zero CPU-dispatch assertions meaningful; the no-touch
test also requires the REX3 JIT to be enabled, so unset `IRIS_NO_JIT`. An explicitly
requested backend must resolve to that backend on an integrated or discrete
GPU: an unavailable adapter, a backend mismatch or a software adapter fails
the test rather than skipping. On Windows, the complete required Vulkan+D3D12
gate is (the machine-fault fixture disables NVRAM/EEPROM file persistence,
and the script isolates and removes its JIT profile under `target`). The
PowerShell session must permit running local scripts:

```powershell
$env:LIBCHDMAN_FORCE_SOURCE = '1'  # when the prebuilt libchdman does not link
Remove-Item Env:\IRIS_NO_JIT -ErrorAction SilentlyContinue
.\scripts\test-resident-gate.ps1
```

The exact 131-case script requires 44 `resident_` CPU tests, four flat-fill
tests, two line tests, two HostW walker tests, one HostW interpreter/JIT
source test, one shade-DDA test, one pattern-state test, two resident HostR
port/retirement tests, three command-ABI tests and three direct-state tests,
then **24 resident GPU tests and ten direct scanout tests on each backend**.
That is 131 tests in total. The window test requires an available native
desktop and a GPU that can present to it. The script checks each libtest pass
count, rejects hardware skips and prints the selected adapter; both Vulkan and
D3D12 must succeed for the full gate. `.\scripts\test-resident-gate.ps1
-Backend vulkan` or `-Backend dx12` runs just one hardware backend and is not
the complete two-backend gate. For manual PowerShell runs set
`$env:REX3_RESIDENT_BACKEND = 'vulkan'` or `'dx12'` before each command
instead of the shell prefix. A host that lacks either required hardware
backend cannot claim the complete gate. Metal is not covered by this script.

The direct group covers the explicit resident frontend matrix: CPU
software/GL composition, CPU planes through wgpu scanout, same-device
resident wgpu scanout, and explicitly labeled resident readback through the
ordinary CPU renderer. Both resident modes exercise transactional screenshot,
headless frame and loopback VNC capture. Capture failures must keep tickets
pending, preserve the last published pixels and `frame_seq`, and leave the
selected display mode unchanged. Assertions distinguish per-mode composed
capture attempts, successes and bytes from attributed raw-plane
`Compatibility` observations.

The lifecycle cases stop and serialize producers at an empty GFIFO boundary,
capture registers and both complete authoritative planes once, and restore
them into a new epoch before restart. They cover disk snapshots, CI rollback,
live debugger checkpoints, a second Newport head, reset, monitor exclusion,
HostW continuation and sticky failure. Hardware cases round-trip complete
GPU-written planes and both resident display modes while the CPU framebuffer
mappings remain protected. Missing, duplicate, incomplete or failed snapshot
halves must not publish registers, pixels, captures or a restartable board.
Metal remains outside this gate.

Production activation has two additional ignored hardware regressions. They
construct a normal Newport machine rather than injecting the backend directly:
both heads must share one `Arc<Gpu>`, repeated peripheral start must leave both
processor/display workers alive, direct mode must require the same-device
presenter, and explicit readback mode must advertise its two-second minimum
capture timeout. Run them on both supported backends:

```powershell
$env:REX3_RESIDENT_BACKEND = 'vulkan'
cargo test --locked -p iris --lib --features wgpu production_newport_wgpu_tests -- --ignored --nocapture --test-threads=1
$env:REX3_RESIDENT_BACKEND = 'dx12'
cargo test --locked -p iris --lib --features wgpu production_newport_wgpu_tests -- --ignored --nocapture --test-threads=1
cargo test --locked -p iris --lib --features wgpu power_on_reset_fits_default_service_thread_stack -- --test-threads=1
```

The final live matrix uses an ordinary `iris` binary and CI socket, not a test
fixture. For each of Vulkan and D3D12 it boots the embedded PROM in direct
mode, requests a composed screenshot, saves and restores a disk snapshot,
performs CI rollback, resets through the monitor, restarts, captures again and
checks `rex status` for `no_cpu_fallback=true` and zero unsupported work.
Vulkan additionally exercises a visible same-device wgpu window, explicit
readback compatibility and a real RFB 3.8 loopback VNC framebuffer update.
The direct VNC run must show composed direct bytes with zero compatibility raw
bytes. These runs establish production lifecycle wiring independently of the
representative application run below.

The application-level acceptance run uses an installed IRIX 6.5 IP22 system
with Xsgi, 4Dwm, toolchest, Showcase and the Atlantis demo. It boots the same
disk through three isolated overlays: CPU raster, Vulkan resident direct and
D3D12 resident direct. A scripted Showcase rectangle/ellipse workload must
produce the same canvas pixels on all three paths. Save/restore and rollback
must reproduce the complete pre-mutation resident screenshot exactly.
Atlantis then runs for a 15-second warm-up and a 30-second measured interval.
On the RTX 5070 / 16-logical-CPU acceptance host, the measured results were:

| Engine | REX3 GO/s | Host core equivalents | Throughput vs CPU | Host CPU vs CPU |
|---|---:|---:|---:|---:|
| CPU raster/JIT | 4,761.5 | 1.673 | baseline | baseline |
| Vulkan direct | 6,765.6 | 1.574 | +42.1% | -5.9% |
| D3D12 direct | 5,918.0 | 1.545 | +24.3% | -7.7% |

CPU throughput is the `rex status` `GO TOTAL` delta. Resident throughput is
the telemetry accepted-command delta, and host core equivalents are process
CPU seconds divided by wall seconds. Both resident samples require
`unsupported=0`, `no_cpu_fallback=true`, no increase in raw/composed readback
counters during the timed interval, and conservative p95 presentation buckets
no higher than 4.2 ms. The accepted Showcase canvas crop SHA-256 was
`b733d604c1a34554578e69fd9874d9ab892a470d54e67428d6a7fc081eec4190`
on all three paths. Screenshots and snapshot plane observations occur outside
the timing window and remain explicitly attributed.

The M8 cases add checked histogram/delta accounting, exact GFIFO
full-versus-contention telemetry, bounded-set wait accounting, a 4 ms
continuous-stream batch-age boundary and profiling-only validation of the
GPU-generated active-tile dispatch. The independent mixed-workload hardware
gate compares complete planes and context against the CPU engine after fills,
copies, integer/fractional/antialiased lines, HostW, HostR, direct
presentations and explicit observations. It also requires zero ordinary
raw-plane/composed readback and reports typed batch, GFIFO and latency
aggregates.

An optional local recorder file can be replayed through the resident backend
without printing records or pixels. The file is not a test artifact and must
not be committed. `REX3_RESIDENT_PROFILE` enables a 64-byte metadata map and
wait per nonempty batch; omit it to verify the default map-free path.
`REX3_RESIDENT_REPLAY_FRAMES` optionally presents the final replayed state
through direct wgpu scanout that many times without requesting capture. The
recording must contain valid final display state when that option is used.
For example, on PowerShell:

```powershell
$env:REX3_RESIDENT_REPLAY = 'C:\local\newport.rec'
$env:REX3_RESIDENT_BACKEND = 'vulkan' # repeat with dx12
$env:REX3_RESIDENT_PROFILE = '1'
$env:REX3_RESIDENT_REPLAY_FRAMES = '8'
cargo test --locked -p iris --lib --features wgpu,rex-jit,rexdiag,gputrace resident_replay_aggregate_file -- --ignored --nocapture --test-threads=1
```

The aggregate reports records and checked reads, accepted/no-op/unsupported
operations, typed primitive/mode/coverage/HostRW distributions, nonempty
batches and flush reasons, GFIFO/backpressure, attributed transfer bytes and
conservative latency buckets. Replay throughput and optional post-replay
presentations are test-path measurements, not live guest wall-clock timing.

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
| `rust.yml` | every pull request and push to `main` | `cargo nextest run` on the workspace, the audio tests through a PulseAudio null sink, a build of the release feature set, the interpreter's hot paths, the REX3 dirty-upload tests and a lint of the distribution scripts; pushes to `main` also run the full-size GR2 and IMPACT JIT sweeps (`rss_jit_tests`, `re3_jit_tests`) |
| `loom.yml` | pull requests that change a file holding loom models (`cfg(loom)`), nightly on `main` at 04:30 UTC, and by hand from Actions | the `loom_tests` concurrency models; locally `cargo loom` (see [Loom models](#loom-models-and-threadsanitizer)) |
| `tsan.yml` | nightly on `main` at 03:30 UTC, and by hand from Actions; gates nothing | the threaded device tests under ThreadSanitizer, to catch data races (see [below](#loom-models-and-threadsanitizer)) |
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
`rex-jit` and `gputrace`. `.cargo/config.toml` owns every test command and the
workflows only call its aliases: `cargo ci-test` (the second set, on the Linux
jobs), `cargo ci-test-default`, and `cargo ci-clock` (the nightly clock tests).
`jitv2` implies `tcache`, which compiles out the real-cache path, so the default
build stays: Windows and macOS test it on pull requests and the nightly workflow
tests it on x86-64 Linux. Windows also runs `cargo ci-test`. The Linux job
compiles the `iris` crate fewer times than the others: it skips
`cargo build --workspace` and the SIMD lane step because nextest builds and
runs the same code, and runs the REX3 replay corpus tests in the same build
(the one ignored filesystem test runs in a second invocation over that build).
`rex-gpu` is not in either set: it turns on the whole `wgpu` feature, whose GPU
suites need a named adapter. The other platforms only compile it, with a
`cargo check` of the `rex3-gpu-probe` example. Pull requests restore `main`'s
build cache but never save one, so they cannot push `main`'s entries out of the
repository's cache quota. The Linux, AppImage and Windows package builds in
`build.yml` cache only the cargo registry and git checkouts, not their target
trees.

### Loom models and ThreadSanitizer

The lock-free and condition-variable protocols have [loom](https://docs.rs/loom)
models in `loom_tests` modules beside the code they check: the GFIFO ring
(`src/gfifo.rs`), the interrupt guard `IrqMutex` (`src/irq.rs`), PCI INTx
routing (`src/pci/intx.rs`) and the jitv2 compile pool's flush barrier
(`src/cpu/jitv2/pool_barrier.rs`).
They are built only under `--cfg loom`, where `src/sync.rs` swaps
`parking_lot`'s `Mutex` and `Condvar` for loom's, so the model drives the code
the device runs rather than a copy of it. Run them all with:

```sh
cargo loom    # alias in .cargo/config.toml: the lib's loom_tests, built with --cfg loom and jitv2
```

`jitv2` is on so the compile pool's models, which exist only with it, are
built. Name a new model module `loom_tests` and `cargo loom` and the workflow
pick it up.

ThreadSanitizer sees races that functional tests and loom miss, such as a
plain `u64` written from another thread. The `tsan` profile in
`.config/nextest.toml` selects the threaded device tests by name (`ioc3`,
`hub`, `intx`, `bridge`, `isp1040`, `isp12160`) and leaves out `ppmem`, whose
plain read of an atomically updated word is the lock-free design. Locally, on
x86-64 Linux with the `rust-src` component:

```sh
RUSTFLAGS="-Zsanitizer=thread" CARGO_PROFILE_RELEASE_PANIC=unwind \
cargo nextest run -Zbuild-std --target x86_64-unknown-linux-gnu --release --lib --profile tsan
```

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
  CPU for one engine (`cpu_tests`: `r4400`, `r4400_other_cache`, `r5000`,
  `r5000_other_cache`, `r10000`, `r12000`, `r14000`; `bench_accuracy`:
  `r4400`, `r5000`, `r10000_195`), with a message naming the
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
login (`scripts/boot-goldens/`). The 6.5.22 Indy disk has the HAL2 audio
driver in its kernel. No 6.5.30 Indy disk is published, so 6.5.30 boots an O2
(`--machine o2_ip32`) on the O2's built-in PROM.

`iris-ci boot-test` seeds NVRAM on a first headless run, boots with the console
on serial and waits up to 15 minutes for `login:`, then does the same with the
console on the graphics board and a screenshot to match. It reads the seeded
NVRAM back from the PROM and from the booted guest, and checks that iris quits
cleanly. What it seeds and checks is data: `tests/irix/boot.toml` (built in, and
used for 5.3), or the image's own `--steps` file (`tests/irix/boot-irix65.toml`,
`tests/irix/boot-irix6530-o2.toml`). The nightly gives each boot 10 minutes
(`--timeout 600`). Each check is a
JUnit case, so a failed run is annotated with the check that failed, and the run
keeps the logs. When the headless run never reaches the PROM's menu or a prompt,
`phase-a.log` holds everything the PROM printed and `phase-a-cpu.log` the CPU's
registers where it stopped (on the O2, also the words at physical 0x1000 and
0x4000). [clanker/testing/ci-unattended-irix-boot.md](clanker/testing/ci-unattended-irix-boot.md)
has the details.

## Disk persistence on exit

`src/exit_persistence_tests.rs` builds the Indy, Indigo2, Indigo2 IMPACT, O2,
Octane and Tezro, each with one disk of each kind (raw with an overlay, raw in
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
