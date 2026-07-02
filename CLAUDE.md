# Claude Instructions — IRIS

IRIS is an SGI Indy (MIPS R4400) emulator written in Rust. It boots IRIX 6.5
and 5.3 to a usable system (shell, networking, X11). It is **not** cycle-accurate
— IRIX doesn't need it and accuracy would only make it slower.

## Read these first

- `HACKING.md` — architecture: data path/endianness, concurrency model, the
  MC bus/device/port abstraction. **Read before touching device or CPU code.**
- `HELP.md` — running it: serial ports, monitor console, NVRAM/MAC setup, disk
  image prep.
- `README.md` — overview, feature flags, current status.
- `docs/` — per-device notes (hal2, rex3, irix-install, …).
- `rules/` — accumulated, hard-won findings about emulator behaviour
  (`jit/`, `snapshot/`, `irix/`, `testing/`). Check here before re-deriving a
  gotcha; when you confirm a non-obvious fix, write it up here as a short
  markdown note so the next session doesn't relearn it.
- `rules/git/contributing-workflow.md` — **Cursor git workflow**: read before any
  commit, push, or PR so changes stay tracked (branch from `main`, push to `fork`,
  PR to techomancer/iris, one change at a time).

## Build & run

```
cargo run --release                                       # interpreter
cargo run --release --features lightning,rex-jit          # recommended for speed
IRIS_JIT=1 cargo run --release --features jit             # enable MIPS JIT
```

Binaries: `iris` (the emulator), `iris-ci` (CI/automation socket client),
`coffdump`, `chd_extract`. Feature flags and `IRIS_JIT_*` env vars are
documented in `README.md`.

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
  poking. See `rules/snapshot/` and `manual_test_runbook.md`.
- Install IRIX only from original media (see `docs/irix-install.md`). Never use a
  pre-built MAME CHD as a shortcut.
- After changing PROM env (`setenv`/`unsetenv`) or NVRAM, run `rtc save` from the
  monitor console before halting, or the change is lost.
