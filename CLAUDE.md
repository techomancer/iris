# Claude Instructions — IRIS

IRIS is an SGI Indy (IP24) and Indigo2 (IP22) emulator written in Rust, with an
R4400 or R5000 CPU selected at runtime. It boots IRIX 6.5 and 5.3 to a usable
system (shell, networking, X11). It is **not** cycle-accurate
— IRIX doesn't need it and accuracy would only make it slower.

## Read these first

- `HACKING.md` — architecture: data path/endianness, concurrency model, the
  MC bus/device/port abstraction. **Read before touching device or CPU code.**
- `HELP.md` — running it: serial ports, monitor console, NVRAM/MAC setup, disk
  image prep.
- `README.md` — overview, feature flags, current status.
- `iris-gui-README.md` — the optional egui front-end (`-p iris-gui`).
- `CHANGELOG.md` — what changed, by area.
- `docs/` — per-device notes and design docs (hal2, rex3, wd33c93a, ppmem,
  tcache, nutlb, …) plus the hardware datasheets (PDFs).
- `rules/` — accumulated, hard-won findings about emulator behaviour
  (`jitv2/`, `rex3/`, `snapshot/`, `irix/`, `testing/`, `gui/`, `perf/`,
  `scsi/`, `macos/`, `build/`). The IRIX install guide is
  `rules/irix/irix-install.md`. Check here before re-deriving a
  gotcha; when you confirm a non-obvious fix, write it up here as a short
  markdown note so the next session doesn't relearn it.

## Build & run

```
cargo run --release                                       # interpreter
cargo run --release --features lightning,rex-jit          # recommended for speed
cargo run --release --features jitv2,rex-jit              # enable MIPS JIT v2 (experimental)
cargo run --release -- --cpu r5000                        # CPU is a runtime choice, not a feature
cargo run -p iris-gui --release                           # GUI front-end
```

The toolchain is pinned to nightly (`rust-toolchain.toml`). `lightning` and
`developer` are mutually exclusive; `r5ksc`/`r5ksc_triton` deliberately fail to
build.

Binaries: `iris` (the emulator), `iris-ci` (CI/automation socket client),
`iris-bench` (benchmark driver), `coffdump`, `mkvh` (SGI volume headers),
`chd_extract` (`chd`), `jitv2_analyze`/`jitv2_verify`/`jitv2_pcp_dump`
(`jitv2`), and `iris-gui` in the workspace. Feature flags are documented in
`README.md`.

## Testing and benchmarking

- `cargo test --workspace` — unit tests (per-device snapshot round trips, CPU,
  TLB, REX3; add `--features jitv2` for the JIT equivalence tests).
- `cpu-tests/` — bare-metal MIPS III/IV correctness suite. "Is this instruction
  right", one instruction at a time. `make -C cpu-tests run`.
- `bench/` — bare-metal benchmark suite. "How fast is this build, and is it
  still right after ten million of them." Reports throughput, guest MIPS and an
  accuracy score per kernel; `iris-bench matrix` sweeps R4400/R5000 x
  interpreter/jitv2. Read `rules/testing/benchmark-suite-gotchas.md` before
  adding a kernel — the accuracy check catches endianness, uninitialised
  memory and out-of-bounds reads, and every one of those has already happened.
- Both share `cpu-tests/harness` (toolchain probe, SCC console, startup and
  exception dispatch). Changing those files affects both suites.

## Hard invariants (from HACKING.md)

- **Endianness lives only at "The Edge."** Host `u32`/`u64` are bit-containers;
  byte-swapping happens at PROM/disk I/O via `swap_on_load`, never in CPU/bus/MC
  logic. **Do not suggest `.to_be()` / `.to_le()` for memory or register code.**
- **Concurrency is per-device.** CPU, REX3, SCSI, and ethernet run on their own
  threads and lock their own state. Deadlocks live in callbacks *up* to a parent
  device (e.g. SCSI → HPC3) — be careful there.

## Automation & CI

- `iris-ci` is the canonical socket interface for driving a running emulator
  (snapshots, scripted input, headless runs). Prefer it over ad-hoc serial
  poking. See `rules/snapshot/` and the CI section of `README.md`.
- Install IRIX only from original media (see `rules/irix/irix-install.md`). Never use a
  pre-built MAME CHD as a shortcut.
- After changing PROM env (`setenv`/`unsetenv`) or NVRAM, run `rtc save` from the
  monitor console before halting, or the change is lost.
