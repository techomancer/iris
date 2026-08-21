# Benchmark in iris-gui

Goal: a user presses one button and gets a meaningful score for their machine,
on every platform, with no toolchain, no subprocess, and no files written
outside the sandbox.

**Status: built.** The spine (an embeddable emulator, the suite as a linked-in
asset, an in-process runner, the GUI screen over it) is implemented and tested.
What is left is listed under [Not built](#not-built) at the end — one of the two
items is a build-system decision rather than work.

---

## What it does now

```
iris-gui  Benchmark tab
  └── iris::bench_runner::run(opts, progress_cb)
        ├── MachineConfig { headless, no_audio, no scsi, banks: 128+128 }
        ├── Machine::new_with_testdev(cfg, TestDevice::new_embedded(…))
        ├── machine.load_elf_bytes(benchsuite::SUITE_ELF)   include_bytes!, 285 KB
        ├── TestDevice sink ──> Vec<u8> ──> Progress events per line
        └── on TESTDEV_EXIT ──> stop ──> parse ──> Run
```

Nothing outside the process; identical on macOS, Windows and Linux. `iris-bench
run` is the same code path, so "run the suite and parse the answer" has one
implementation rather than two.

The tab now ships in every build, App Store included. Its developer half — the
matrix runner and the native host baseline, which need a source checkout and
build a separate emulator per cell — is folded into a "Developer tools"
disclosure that is compiled out under `feature = "appstore"`.

---

## The measurement got 2.5x faster, and that changed the design

The plan this document replaces was written against a 160-second interpreted
run and built a "quick mode" around it. Measuring where that time actually went
found something better.

The suite prints its table one row at a time, over the SCC. `scc_putc` spins on
`RR0.TX_BUFFER_EMPTY` with a 100,000-iteration bound before giving up. But
WR5.TX_ENABLE is programmed by the PROM, and a `--load-elf` image never runs the
PROM — so the transmitter is never enabled, the four-byte holding queue fills,
`TX_BUFFER_EMPTY` never comes back, and **every character costs the full spin**.

| r4400 lightning, full suite | wall | timed regions | accuracy |
|---|---|---|---|
| before | 117 s | 12.5 s | 40/40 |
| after | **46 s** | 12.4 s | 40/40 |

The fix is six lines in `cpu-tests/harness/console.c` — latch the port off once
it has proved it will not transmit, but only when a test device is present, so
that a run with no other sink still gets its (slow) serial output and a
PROM-booted run is untouched. `cpu-tests` was paying the same tax and gets the
same speedup. Written up in
`rules/testing/scc-serial-output-from-bare-metal-code.md`.

Measured on the reference host, plain release interpreter — which is what the
App Store build is:

| | wall | guest MIPS | DMIPS | accuracy |
|---|---|---|---|---|
| full | 58 s | 51.7 | 74.4 | 100% |
| quick | 33 s | 51.9 | 72.6 | 100% |

Quick mode therefore reports the same numbers to within a couple of percent.
What it gives up is precision, and only that.

### Why quick mode does not drop kernels

The original plan's quick mode ran fewer groups. It no longer needs to, and
should not: **accuracy is the number the shipping build leads with**, and a
short run that quietly checked less would report the same 100% while covering
less ground. So quick mode scales the per-kernel target time and drops from
best-of-two to a single pass, and every kernel still runs and still verifies.

The floor is each kernel's verification pass — one exact workload, because that
is what the golden checksum was computed against — plus the kernels whose base
iteration count is already 1 (`codec/lz` alone is 1.5 s and cannot be made
smaller). That is about 22 s of the 33, and it is why quick mode lands near
half a full run rather than near a tenth of one.

---

## What the user sees

**One primary action** — "Benchmark this Mac" / "this PC" — with a Quick
checkbox. No cell picker and no engine picker: the shipping build is one binary
with one CPU and one engine, so there is nothing to choose.

**While it runs:** a progress bar, the current kernel, elapsed, and an estimate
that only appears once there is evidence for one. The first rows are the cheap
integer kernels and the last are the expensive codec ones, so extrapolating from
row two is confidently wrong; before four rows it says "about 35 s in total"
instead of pretending.

Progress is honest because the guest says up front how many kernels it will run
(`IRIS-BENCH-PLAN benches=46`). It has to come from the guest: the count depends
on the CPU (some kernels are R5000-only) and on the group mask, and only the
guest can resolve either.

**When it finishes:**

```
  Emulated Indy        74 DMIPS      ← interpreter: the store build has no JIT
  Emulator throughput  52 MIPS
  Accuracy            100%  (40/40)  ← correctness, not speed

  Integer    38.5 M ops/s
  Floating    7.3 M ops/s
  Memory     30.7 MB/s
  Imaging     1.0 M px/s
  Codec       7.6 MB/s

  Reference statistics not gathered for this platform.

  [ Copy ]  [ Save report… ]        ▸ Details
```

- **The absolute numbers carry the screen.** DMIPS has forty years of published
  figures behind it, guest MIPS is meaningful on its own, and accuracy needs no
  baseline at all — so an empty reference table costs the user very little.
  Comparison is an enhancement, never a dependency.
- **Accuracy is as prominent as speed.** It is the differentiator: no other
  emulator reports whether it computed the right answer. 100% tells a user
  something real, and 97% has found a bug worth reporting.
- **Caveats are stated, not left to be discovered.** A quick run says it was
  one. An interpreter build says a JIT build scores about four times higher and
  is not comparable. A kernel that took an unexpected exception is named,
  because the harness steps over faults and the kernel still reports a
  throughput — for doing something other than what it claims.
- **The console is one click down.** A wall of monospace as the primary surface
  reads as "something went wrong" to a reader who did not ask for a log.
- **Nothing is uploaded.** Export is an explicit `rfd` save panel.

A benchmark is refused while a machine is running. Two emulators sharing the
host would measure whatever IRIX happened to be doing, and refusing is simpler
than explaining the result afterwards.

---

## How the pieces work

### Embeddability (`iris`)

| Was | Now |
|---|---|
| `TestDevice::exit` → `process::exit` | `new_embedded` takes an `ExitHook`; the standalone path is unchanged |
| `TestDevice::putc` → stdout | a `Vec<u8>` sink the embedder drains |
| `Machine::load_elf` takes a path | `load_elf_bytes` alongside it; `load_elf` calls it |
| `Machine::new` → `process::exit(1)` on the ultra64 slot clash | panics, which `iris-gui` already catches |

**`exit` returning is safe, and that is the whole trick.** The store lands on
the CPU thread mid-instruction, which is why the original plan flagged this as
the delicate piece. It is not, because every guest reaches `EXIT` through
`testdev_exit()`, which spins forever afterwards — a bare-metal image has
nowhere to return to. So the hook fires, the store completes, the CPU thread
loops harmlessly in guest code, and the runner stops the machine from its own
thread. A hook that *blocked* would deadlock against `Machine::stop`'s join.

Collected in `rules/testing/embedding-the-emulator-in-process.md`, along with the
three other things that bite (skip `register_system_controller`; spawn with a
16 MB stack; `Machine::start` does not start the CPU in a debug build).

### The suite as an asset (`bench/prebuilt/`)

A known-good `irisbench.elf` is checked in and linked with `include_bytes!`.
Precedent: the 512 KB PROM is already embedded, though as a generated Rust array
— `include_bytes!` on a real file is smaller, faster to compile, and diffable as
the binary it is.

**Drift is the danger, and CI is the answer.** Accuracy is scored against golden
checksums compiled *into* the image, so a stale image against fresh goldens
reports failures to users that are not real. `suites.yml` rebuilds it and fails
on any difference; `make -C bench prebuilt` refreshes it.

### Quick mode (`TESTDEV_RUN_CONFIG`)

A bare-metal image has no argv and no environment, so the host leaves its
request in a register the guest reads at startup:

```
  31            16 15   12 11             0
  +---------------+-------+---------------+
  |     groups    |repeats|    time_pct   |
  +---------------+-------+---------------+
```

Every field means "unrestricted" when zero — which is exactly what an emulator
predating the register returns — so the guest reads it unconditionally.
Verified both ways: the current guest image runs correctly on an emulator built
before the register existed.

The effective configuration is echoed back in the report (`#run …`) and stored
on every result, so a shortened run can never be mistaken for a full one.
`iris-bench reference` refuses to put one in the reference table.

### The machine inventory

Every result now records what it ran on, read out of the hardware rather than
reported from what the runner configured: CPU identity and revision plus the
L1/L2 geometry from CP0 Config, and the RAM banks from the memory controller's
MEMCFG registers. It works with no PROM and no POST because `--load-elf`
programs MEMCFG exactly as POST would before the image starts.

It is not decoration. The `mem/` kernels are a direct readout of the cache
hierarchy, so two results with different L1 sizes are not measuring the same
thing — and nothing in a stored result used to say so.

The CPU is named from PRId alone, so it is right on any MIPS machine and not
just the two this emulator models. The suite used to *refuse* to run on
anything else, on the stated grounds that the goldens were selected by PRId;
they are not — `golden.h` is one flat CPU-independent table and no kernel is
CPU-gated — so it now runs and says what it found.

### The report model (`iris::bench_report`)

The parser, the data model and the reference table moved out of
`src/bin/iris_bench.rs` into the library: three callers need them and only one
of them is that binary. One parser, one schema, one definition of what
"accuracy" means.

---

## The reference table

`data/bench_reference.json` — checked in, `include_str!`'d, updated by hand.
**It ships empty, and empty is a normal state**: a machine with no row gets
"reference statistics not gathered for this platform" rather than a comparison.
No upload, no download, no user-writable override. It is a static file and a
pull request.

Three things must agree before a comparison means anything, and
`ReferenceTable::matching` enforces all three: the **suite** (`suite_id`, a
blake3 of the guest binary — two different workloads under one name is not a
comparison), the **emulated CPU** (the R4400 and R5000 cache models differ
deeply), and the **engine** (interpreter and jitv2 are about 4x apart). A
mismatch on any of them is treated exactly like an empty table, so there is one
fallback path rather than four.

No IRIS Index in v1. An index needs a frozen normalization vector to mean
anything, and there is nothing to freeze until the table has entries.

Note this is a different file from the golden checksums, which are the
correctness oracle and are compiled *into* the guest image. Those must never be
user-editable — editing them would let someone "fix" an accuracy failure.
Externalising the performance table therefore carries no correctness risk.

---

## Risks, and what became of them

| | Risk | Outcome |
|---|---|---|
| 1 | `TestDevice::exit` firing on the CPU thread mid-store | Not a problem — the guest spins after it. Covered by the P0 test. |
| 2 | Two `Machine`s at once | Refused: the button is disabled while a machine runs. |
| 3 | Laptop measurement validity — throttling, efficiency cores, other apps | Still real. Best-of-two per kernel in a full run; quick mode says it is a quick mode. Surfacing per-kernel spread is [not built](#not-built). |
| 4 | Embedded ELF drifting from `golden.h` | CI rebuilds and diffs both. |
| 5 | A user reading a low score as "IRIS is broken" | The interpreter caveat is stated on the results screen. |
| 6 | App Store review reading it as a hardware-diagnostic utility | It benchmarks the app's own emulator, reports nothing about the host beyond a CPU model string, and uploads nothing. |

**The App Store build has no JIT at all.** `main.rs` forces `IRIS_NO_JIT=1` under
`feature = "appstore"`: Cranelift allocates executable memory with
`mmap`+`mprotect` rather than `MAP_JIT`, and the sandbox only permits `MAP_JIT`
pages (`com.apple.security.cs.allow-jit` is the only code-signing entitlement
review accepts). The REX3 draw-shader JIT is off for the same reason.

Consequences the design absorbs rather than works around: the headline is
~52 MIPS / ~74 DMIPS, not 203 / 213; the shipped reference table must hold
interpreter rows; and every stored result carries an explicit `engine` field
rather than an implied one, so this reverses cleanly if Cranelift is ever made
`MAP_JIT`-aware.

---

## Not built

**Host baseline in-process.** The kernels already compile natively — that is how
`golden.h` is generated — so compiling `bench/kernels/*.c` plus
`bench/gen/hostplat.c` into `iris` with the `cc` crate would let the native
comparison run in-process too, preserving the property the suite rests on: the
same C on both sides.

The decision it needs: `cc` makes a C compiler a build requirement for the
`iris` crate. Either gate it behind a `bench-host` feature that release builds
turn on — contributors' builds then differ from shipped ones — or leave it
always on with `build.rs` degrading to a stub when no compiler is found.
Recommended: the latter. The failure mode is a missing feature rather than a
broken build, and shipped and local builds stay identical.

Not required for anything above; the emulated numbers stand on their own, which
is the whole reason the results screen leads with them.

**Anything but an Indy or Indigo2.** The bare-metal harness is written for
IP22/IP24 — the load address, the console and the memory inventory all assume
that machine — so the suite does not run on an O2, Octane or Origin, and no run
on real SGI hardware of any kind is recorded in this repo. It costs nothing
today, since IP22/IP24 is what IRIS emulates; it would matter for a
real-hardware reference number, which is the only kind that is not the
emulator's opinion of itself. Written up, with what a port would involve and
which parts of the analysis were actually tested, in
`rules/testing/bare-metal-harness-platform-assumptions.md`.

**Measurement-spread warning.** Risk 3. The harness keeps the best of two timed
passes but does not report how far apart they were, so a thermally throttled
laptop looks the same as a quiet desktop. Reporting the spread and warning above
~5% needs one more field per row in the machine block.
