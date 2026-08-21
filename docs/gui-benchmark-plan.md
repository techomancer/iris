# Benchmark in iris-gui — feature map

Goal: an App Store user presses one button and gets a meaningful score for their
machine, on every platform, with no toolchain, no subprocess, and no files
written outside the sandbox.

Status: **plan only.** Nothing below is built. The developer-facing path
(`iris-bench`, `bench/`, the Benchmark tab hidden under `!appstore`) already
works and is what this reuses.

---

## The fact that makes this tractable

`iris-gui` depends on `iris` as a **library** and already runs the emulator
**in-process** on a worker thread (`handle.rs:382`, `Machine::new(cfg_owned)`
inside `catch_unwind`). There is no `iris` subprocess to sandbox, no `cargo`, no
`iris-bench` binary to ship.

So the App Store benchmark is not "drive the developer tool from a GUI". It is:
build a `MachineConfig`, load an ELF that is already inside the app, run it, read
the report the guest prints. Most of the work is making the emulator *embeddable*
for that, not building UI.

Four things stand in the way, all in `iris`, none large:

| Blocker | Where | Why it matters in-process |
|---|---|---|
| `TestDevice::exit()` calls `std::process::exit` | `testdev.rs:164` | Guest finishing the suite would **quit the app** |
| `TestDevice::putc()` writes to `stdout` | `testdev.rs:129` | GUI never sees the report |
| `Machine::load_elf` takes a path | `machine.rs:1064` | Suite has to be a file on disk |
| `Machine::new` calls `process::exit(1)` on the ultra64/test-device slot clash | `machine.rs:565` | Config mistake would quit the app |

`load_elf` is a five-line refactor — it is already `fs::read` → `elf::parse` →
load segments (`mips_exec.rs:8441`). The other three are the real work, and
`TestDevice::exit` is the one to be careful with: the guest is mid-store on the
CPU thread when it fires.

---

## What the user sees

**One primary action.** "Benchmark this Mac" (or PC). No cell picker, no engine
picker, no mention of R4400 vs R5000 — the App Store build is one binary with one
CPU and one engine, so there is nothing to choose.

**While it runs:** a progress bar, the current kernel name, elapsed and estimated
remaining. Not a log tail — the current tab streams subprocess stdout, which is
right for a developer and wrong for everyone else.

**When it finishes** — this is the shipping state, with an empty reference
table. Design for it first; it is what every user sees until someone measures
their machine:

```
  Emulated Indy        71 DMIPS      ← interpreter: the store build has no JIT
  Emulator throughput  51 MIPS
  Accuracy            100%  (40/40)  ← correctness, not speed

  Integer    38.5 M ops/s        Imaging     1.0 M px/s
  Floating    7.3 M ops/s        Codec       7.6 MB/s
  Memory     30.7 MB/s

  Reference statistics not gathered for this platform.

  [ Details ]  [ Copy ]  [ Save report… ]
```

Once a matching row exists in `data/bench_reference.json`, the same block gains
a comparison column and the sentence is replaced:

```
  Integer    38.5 M ops/s   ████████████░░  1.4× vs MacBook Air (M1)
  Floating    7.3 M ops/s   ███░░░░░░░░░░░  0.7×
  …
```

- **The absolute numbers carry the screen.** DMIPS has forty years of published
  figures behind it, guest MIPS is meaningful on its own, and accuracy needs no
  baseline at all — so an empty reference table costs the user very little.
  Comparison is an enhancement, never a dependency.
- **Accuracy is shown as prominently as speed.** It is the differentiator: no
  other emulator reports whether it computed the right answer. A user seeing
  100% learns something real, and a user seeing 97% has found a bug worth
  reporting.
- **Nothing is uploaded.** Results live in the app container; export is an
  explicit save panel. (See `PRIVACY.md`.)

Numbers above are the interpreter's, because that is what ships — see the JIT
note under Risks. A source build with `jitv2` scores ~203 MIPS / ~213 DMIPS on
the same host, which is why every stored result carries its engine.

**Quick vs full.** The full suite is ~160 s interpreted, ~80 s with jitv2. That
is too long for a consumer button as the default. Ship a **quick mode** (~20 s)
as the default and full as an option.

**Progress, not a log tail.** The suite already streams its table a line at a
time; the runner parses those lines anyway to know which kernel is running, so
the progress display and the raw console are the same data at two levels of
detail. Progress bar on top, `Show details ▸` for the console underneath.

---

## Architecture

**Today (developer path):**

```
iris-gui ──spawn──> iris-bench ──spawn──> iris (process)
                         │                    │ --load-elf bench/build/irisbench.elf
                         │                    │ --test-device
                         │                 stdout ──> IRIS-BENCH-BEGIN…END
                         └── parses the block, writes results/*.json
```
Needs: a built ELF on disk, two binaries, subprocess spawn, filesystem writes.
None of that survives the sandbox.

**Proposed (embedded path):**

```
iris-gui
  └── iris::bench_runner::run(opts, progress_cb)
        ├── MachineConfig { headless, no_audio, no scsi, test_device, banks }
        ├── Machine::new(cfg)                    (in-process, worker thread)
        ├── machine.load_elf_bytes(BENCH_ELF)    (include_bytes!, ~285 KB)
        ├── TestDevice sink ──> Vec<u8> ──> progress_cb per line
        └── on TESTDEV_EXIT ──> stop ──> parse ──> Report
```
Needs: nothing outside the process. Works identically on macOS, Windows, Linux.

The same `bench_runner` backs `iris-bench run` too, so there is one
implementation of "run the suite and parse the answer" rather than two.

---

## Work items, in dependency order

### P0 — make the emulator embeddable  *(small, `iris` crate only)*

1. `Machine::load_elf_bytes(&self, bytes: &[u8]) -> Result<String, String>`.
   Refactor `load_elf` to call it. `MipsCpu::load_elf` splits the same way.
2. `TestDevice` gains an output mode:
   - `TestDevice::new_embedded(sink: Arc<Mutex<Vec<u8>>>, on_exit: Box<dyn Fn(u32) + Send + Sync>)`
     alongside today's `new(dump_path)`.
   - `putc` writes to the sink when embedded, `stdout` otherwise.
   - `exit(code)` calls `on_exit(code)` and then **parks the CPU** instead of
     `process::exit`. Getting this right is the one genuinely delicate piece:
     the store that triggers it is executing on the CPU thread, so the handler
     must not block on anything the CPU thread owns. Signal an `AtomicBool` +
     `Condvar` and let the *runner* thread do the stopping.
   - `dump()` becomes a no-op when no path is configured.
3. `Machine::new`'s `process::exit(1)` (`machine.rs:565`) becomes an error or a
   panic. `Machine::new` already panics on bad input and the GUI already catches
   that (`handle.rs:381`), so a full `Result` refactor is optional — converting
   the one `exit` call to a panic is enough and is smaller.

**Test**: a `#[test]` in `iris` that runs the embedded suite in quick mode and
asserts the report parses and accuracy is 100%. That single test covers most of
P0 and P1 and is the regression net for the whole feature.

### P1 — ship the suite as an asset  *(small)*

4. Check in `bench/prebuilt/irisbench.elf` (~285 KB) plus the `golden.h` hash it
   was built against. `include_bytes!` it from a new `src/benchsuite.rs`.
   Precedent: the 512 KB PROM is already embedded (`src/prombin.rs`) — though as
   a 3.2 MB Rust hex array, which is *not* the pattern to copy. `include_bytes!`
   on a real file is smaller, faster to compile, and diffable as a binary.
5. Extend `.github/workflows/bench.yml`: rebuild the ELF and fail if it differs
   from the checked-in one. Same discipline as `golden.h` and `fpvectors.c`
   already have — a checked-in build product that can silently drift is worse
   than no build product.
6. Move the report parser out of `src/bin/iris_bench.rs` into `iris` so the
   library, the CLI and the GUI share one copy.

### P2 — the runner  *(medium)*

7. `iris::bench_runner`:
   ```rust
   pub struct BenchOptions { pub quick: bool, pub groups: u32, pub banks: [u32; 4] }
   pub enum Progress { Started { total: usize }, Kernel { name: String, index: usize }, Line(String) }
   pub fn run(opts: BenchOptions, progress: impl FnMut(Progress) + Send) -> Result<Report, String>;
   ```
8. Rewire `iris-bench run` onto it (drops the subprocess for the local case;
   `matrix` keeps spawning, because comparing builds inherently means comparing
   binaries).

### P3 — the GUI screen  *(medium — this is where the design effort goes)*

9. Replace `bench_ui.rs`'s subprocess + log tail with the runner + a results
   view. The state machine is small (Idle → Running → Done/Failed); the work is
   the results presentation, not the plumbing.
10. **No IRIS Index in v1** — see the reference-table section. Headline the
    numbers that stand alone.
11. **Progress primary, console secondary.** The suite already streams its
    human table a line at a time — deliberately, so a run that prints nothing
    for two minutes is not mistaken for a hang. The runner has to consume that
    stream anyway to know which kernel is running, so the progress events *are*
    the parsed console lines and showing the raw text underneath costs nothing
    extra. Progress bar + current kernel on top; `Show details ▸` reveals the
    console.
    - Do **not** route it through `serial_console.rs`: that is a TCP client to
      `127.0.0.1:8881`, so it would require standing up the loopback serial
      server for a benchmark that has no other use for it. Take the test-device
      sink directly and reuse only the *view* half of that widget (scrollback
      cap, autoscroll, monospace). Splitting `SerialConsole` into transport and
      view is worth doing for its own sake.
    - Raw console as the *primary* surface reads as "something went wrong" to a
      non-technical user. It belongs one click down.
12. Keep matrix/cells/host-baseline buttons behind `!appstore`.

### P4 — host baseline in-process  *(medium, one build-system decision)*

12. The kernels already compile natively — that is how `golden.h` is generated.
    Compile `bench/kernels/*.c` + `bench/gen/hostplat.c` into `iris` with the
    `cc` crate so the native comparison runs in-process. That preserves the
    property the whole suite is built on: **the same C on both sides**, so the
    native ratio is a real number rather than two benchmarks pretending to be
    comparable.
13. **Decision needed**: `cc` means a C compiler becomes a build requirement for
    the `iris` crate. Either (a) gate it behind a `bench-host` feature that
    release builds turn on — contributors' builds then differ from shipped ones;
    or (b) always on, with build.rs degrading to a stub when no compiler is
    found. Recommend (b): the failure mode is a missing feature, not a broken
    build, and shipped and local builds stay identical.

### P5 — quick mode  *(small, guest-side)*

14. The suite deliberately has no runtime selector — a bare-metal binary loaded
    with `--load-elf` has nowhere to take arguments from. Cleanest fix: **a new
    test-device register** the guest reads at startup (`TESTDEV_CONFIG`, next to
    `TESTDEV_CAPS`), carrying a group bitmask and a target-time scale. ~20 lines
    in `harness/main.c`, one register in `testdev.rs`, and it composes with the
    capability probe already there.
    Alternatives considered: a second smaller ELF (doubles the asset and the
    golden discipline), or poking a word into RAM after `load_elf_bytes`
    (works, but invents a second ABI nobody documents).

### P6 — sandbox and store  *(small, mostly audit)*

15. Unhide the tab for `appstore`. Audit: no path outside the container, no
    subprocess, no `process::exit`, export only via `rfd` save panel.
16. Confirm what the App Store workflow actually builds (`appstore.yml` is not in
    this branch). See the open question below.

---

## The reference table

`data/bench_reference.json` — checked in, `include_str!`'d, updated by hand when
someone measures a machine worth recording. **It ships empty, and empty is a
normal state**: a machine with no row gets *"reference statistics not gathered
for this platform"* rather than a comparison. No upload, no download, no
user-writable override, no import/export. It is a static file and a pull
request.

Deliberately cut from an earlier draft of this plan: the layered
bundled/override/import design, and the IRIS Index. The index needs a frozen
normalization vector to mean anything, and there is nothing to freeze until the
table has entries — so v1 shows the numbers that stand on their own (guest MIPS,
DMIPS, accuracy) and an index can arrive later if it earns its place.

Note this is a different file from the golden checksums, which are the
correctness oracle and are already compiled *into the MIPS ELF*. Those must
never be user-editable — editing them would let someone "fix" an accuracy
failure. Externalising the performance table therefore carries no correctness
risk at all.

### Adding a row  *(built — works today)*

```sh
./target/release/iris-bench run --label my-machine
./target/release/iris-bench reference \
    --id m1-max-interp --label "MacBook Pro (M1 Max) — interpreter" \
    --into data/bench_reference.json
```

`reference` with no `--from` takes the newest result in `bench/build/results/`;
without `--into` it prints the row for pasting. See
`data/bench_reference.README.md`.

### Two fields that keep it honest

**`suite_id`** — blake3 of the guest binary the numbers came from, recorded on
every result. Reference figures only mean something against the exact suite that
produced them: add or change a kernel and every stored number silently becomes a
comparison between two different workloads. So the merge refuses when a result
disagrees with a populated table, and the GUI treats a mismatch exactly like an
empty table — one fallback path covers both. An empty table adopts the suite of
the first row merged into it.

**`cpu` and `engine`** — the two engines differ by roughly 4x, so a row without
them cannot be compared with anything. This matters concretely: the App Store
build forces `IRIS_NO_JIT=1`, so rows meant for comparison against it must be
`"engine": "interp"`.

### Multiple contributors

Not designed for. It is a file in the repo; rows arrive as pull requests. If
that ever becomes unwieldy the problem will announce itself, and the schema
already carries everything a merge would need.

---

## Risks and open questions

| | Risk | Mitigation |
|---|---|---|
| **1** | `TestDevice::exit` firing on the CPU thread mid-store | Signal + park; let the runner thread stop the machine. Cover with the P0 test. |
| **2** | Two `Machine`s at once (benchmark while IRIX is running) | Refuse. `handle.rs:358` already has the one-machine guard. It also gives a clean measurement, so this is a feature. |
| **3** | Laptop measurement validity — thermal throttling, efficiency cores, other apps | Already best-of-two per kernel. Surface the spread; warn when the repeats disagree by >5%; say plainly that a laptop on battery will score lower. |
| **4** | Embedded ELF drifting from `golden.h` | CI check (item 5). Non-negotiable — a stale ELF against fresh goldens reports false accuracy failures to users. |
| **5** | A user reading a low score as "IRIS is broken" | Lead with the reference comparison, not the raw number. |
| **6** | App Store review reading it as a hardware-diagnostic utility | It benchmarks *the app's own emulator*, reports nothing about the host beyond a CPU model string, and uploads nothing. Frame the UI that way. |

**Settled — the App Store build has no JIT at all.** `main.rs:116` forces
`IRIS_NO_JIT=1` under `feature = "appstore"`, and the comment there explains
why: Cranelift allocates executable memory with `mmap`+`mprotect`, not
`MAP_JIT`, and the sandbox only permits `MAP_JIT` pages
(`com.apple.security.cs.allow-jit` is the only code-signing entitlement the
store accepts; `allow-unsigned-executable-memory` and
`disable-executable-page-protection` are rejected by review). The first JITed
REX3 draw gets SIGKILL'd. So it is not just jitv2 — the REX3 draw-shader JIT is
off too.

Consequences, all of which the design has to absorb rather than work around:

- The App Store headline is **~51 MIPS / ~70 DMIPS**, not 203 / 213.
- **The shipped reference table must be interpreter numbers.** A jitv2 row next
  to an App Store result is a 4x apples-to-oranges comparison.
- The benchmark is still worth shipping: "how fast is your Mac at emulating an
  Indy" is the user-facing question, and the interpreter is what they have.
- If Cranelift is ever made `MAP_JIT`-aware, this reverses — which is another
  reason entries carry an explicit `engine` field rather than an implied one.

---

## What stays developer-only

`iris-bench matrix` — building a separate emulator per cell is inherently a
source-checkout activity. Same for `--force-build`, the cell picker, and the
`bench/irix/` guest-OS suite (needs an IRIX image and a CI socket).

The split is clean: **one embedded run** is a product feature, **comparing
builds** is a developer tool.

---

## Rough shape of the effort

| Phase | Crates touched | Size | Risk |
|---|---|---|---|
| P0 embeddability | `iris` | small | **the exit path is the one delicate piece** |
| P1 asset + CI | `iris`, workflows | small | low |
| P2 runner | `iris`, `iris-bench` | medium | low |
| P3 GUI screen | `iris-gui` | medium | low (design effort, not technical) |
| P4 host baseline | `iris` + build.rs | medium | build-system decision |
| P5 quick mode | `bench/`, `iris` | small | low |
| P6 sandbox/store | `iris-gui`, workflows | small | gated on the jitv2 question |

P0 + P1 + P2 is the spine: at the end of it `iris-bench run` works with no
subprocess and no ELF on disk, and the GUI is a view over something already
proven. P3 onwards is additive.
