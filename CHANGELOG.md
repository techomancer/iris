# Changelog

All notable user-facing and developer-facing changes to iris.

## [Unreleased] — 2026-05-03

The headline of this release is a complete snapshot/rollback stack: capture
the full machine state to disk, restore it, roll back inside a session, ship
snapshots between machines over HTTP, and validate that any of the above
produces deterministic results.

### Added

#### Snapshot system

- **Save/restore/rollback** (`save_snapshot` / `load_snapshot` /
  `ci_restore` / `ci_rollback` on `Machine`). Captures CPU, MC, IOC, HPC3,
  REX3, RTC, EEPROM, SCSI, Seeq, and all RAM banks plus the COW disk
  overlay. Snapshots live under `saves/<name>/`.
- **In-memory rollback checkpoint** (Phase 2.1): `ci_rollback` skips disk
  by replaying a cached `RollbackCheckpoint` taken at the last `ci_restore`.
  Measured ~42 ms per rollback on M2 vs 145–213 ms for the disk path.
- **Reflink overlay capture** (Phase 1.3): on APFS / btrfs / xfs, snapshot
  copies of multi-GB COW overlays use `clonefile(2)` / `FICLONE` and consume
  ~18 MB actual disk for a 4 GB apparent overlay.
- **Auto-fork-on-restore** (Phase 2.3): `ci_restore` captures the overlay's
  dirty-sector set so the running session can mutate the disk without
  poisoning the parent snapshot.
- **Scratch SCSI volume** (Phase 2.4): a host-controlled raw block device
  for file injection/extraction without networking. Configure with
  `scratch = true` in `iris.toml`; iris pre-formats it with a minimal SGI
  Volume Header so IRIX surfaces it as `/dev/rdsk/dks0dNs0`. CI commands
  `scratch-write` / `scratch-read` / `scratch-clear` / `scratch-info`. New
  module `src/sgi_vh.rs`.
- **Content-addressable chunked RAM** (Phase 3.1): each RAM bank and
  framebuffer is split into 64 KB chunks, BLAKE3-hashed, stored once under
  `saves/.cas/`. Snapshots reference chunks by hash; identical chunks
  across snapshots share storage. A second snapshot of an unchanged
  machine adds **zero bytes** to disk. New module `src/chunk_store.rs`.
- **Snapshot determinism validator** (Phase 3.3): `validate <name>
  [<n_instructions>]` loads the snapshot twice with peripheral threads
  stopped, steps each pass `n_instructions` times in-line, and diffs the
  resulting CPU register digests. 1M instructions in 265 ms. Surfaces
  `load_state` field omissions, host-wallclock leakage at load time, and
  unrestored TLB/cache structures. New module `src/validate.rs`.
- **Snapshot library commands** (Phase 3.2):
  - `tree` — render snapshot parent-chain hierarchy
  - `diff <a> <b>` — per-device, per-RAM-chunk, per-COW-sector delta
  - `gc` — sweep CAS chunks not referenced by any kept snapshot
- **HTTP snapshot registry** (Phase 3.4): `pull <url> <name>` and `push
  <url> <name>` ship snapshots between machines. URL layout mirrors disk
  layout, so any static HTTP server (`python3 -m http.server` against
  `saves/`) works as a read-only pull source. Pull validates each chunk's
  BLAKE3 hash; push uploads chunks first and the manifest last so an
  interrupted push never publishes an incomplete snapshot. Hand-rolled
  HTTP/1.1 client over `std::net` — no new dependency. New module
  `src/registry.rs`. Demonstrated 138× speedup on warm pulls (21 ms vs
  2.9 s) thanks to local-CAS dedup.

#### CI control socket

`--ci` enables a Unix-domain control plane at `/tmp/iris.sock`. New
newline-delimited JSON commands beyond the existing `start` / `quit` /
`serial-{send,read}` / `wait-serial` / `screenshot`:

- `save` / `restore` / `rollback` / `list` / `info` / `delete`
- `validate`
- `tree` / `diff` / `gc`
- `scratch-write` / `scratch-read` / `scratch-clear` / `scratch-info`
- `pull` / `push`

#### Snapshot manifest

A `snapshot.toml` at the top of every snapshot directory records:
- `schema_version` (currently 3)
- `host_arch` (cross-arch loads are refused — FPU bit-layout differs)
- `iris_git_rev` (warns on mismatch)
- `created_at_unix`
- `parent` (snapshot name this was restored from, if any)
- `description`
- `installed_bundles`

`tree` walks `parent` to render snapshot lineage; `diff` uses it to
report what changed between two related snapshots; `gc` uses it to
compute the live chunk set.

#### Tests and validation

- **Per-device round-trip property tests** (Phase 1.7): every `Saveable`
  device has a `save_load_round_trip` test that mutates state, captures
  v1 = `save_state()`, loads v1 into a fresh device, captures v2 =
  `save_state()`, asserts v1 == v2. Catches `load_state` field omissions
  before they corrupt snapshots silently. Covers 10 devices:
  `eeprom_93c56`, `ds1x86`, `ioc`, `pit8254`, `mc`, `mips_tlb`, `ps2`,
  `z85c30`, `wd33c93a`, `seeq8003`.
- **CiSerialBackend regression test**: round-trips a 53-char single-line
  `dd` command through the loopback to prevent regression of the chunked-
  input drop bug (see Fixed below).
- 28+ new unit tests across the new modules; all 198+ lib tests pass.

### Changed

- **Snapshot schema version bumped twice this release**:
  - **v0 → v1** (Phase 1.2): added `snapshot.toml` manifest with
    `schema_version`, `host_arch`, `parent`, etc.
  - **v1 → v2** (Phase 2.2): per-device state moved from `*.toml` (hex
    strings) to `*.bin` (postcard-encoded `BinValue`). cpu state file
    shrunk 24% (3.65 MB → 2.79 MB) and parses 3.4× faster (19.7 ms → 5.8 ms).
  - **v2 → v3** (Phase 3.1): RAM banks and framebuffers moved from raw
    `bank{N}.bin`/`rex3_*.bin` files to the content-addressable chunk
    store at `saves/.cas/`. Each snapshot writes a tiny `chunks.bin`
    manifest of per-bank/per-framebuffer chunk hashes.
  - **Backward compatibility**: load reads any of v0/v1/v2/v3; the
    appropriate code path is dispatched off `manifest.schema_version`.
    New saves write the highest version.
- **`load_snapshot` refactored** into `load_snapshot_inner` (private) +
  `load_snapshot` (public, auto-starts CPU + peripherals on return) +
  `load_snapshot_paused` (used by the determinism validator; leaves all
  threads stopped).
- **`Machine::with_paused`** helper: briefly stops all device threads to
  perform a host-side mutation (used by scratch-write etc.), then
  resumes — but only restarts the CPU if it was running before, so
  pre-`start` operations don't auto-launch the CPU.
- **iris.toml**: documented `[scsi.2]` scratch-volume block (commented
  out by default). New optional fields `scratch: bool` and `size_mb:
  Option<u32>` on `ScsiDeviceConfig`.

### Fixed

- **`cp0_compare` write recalibration: synthetic clock available behind
  `--features ci_clock`.** The previous implementation in
  `src/mips_core.rs` measured `Instant::now()` between successive
  Compare writes to compute a wallclock-stretched `count_step`. Two
  passes from the same starting state would see different host
  scheduling → different `dt_ns` → different `count_step` → different
  timer-interrupt timing → divergent guest execution. With
  `--features ci_clock` we swap in `dt_ns = (cycles since last Compare
  write) * 10ns` (R4400 ~100 MIPS), giving the Phase 3.3 validator
  `deterministic: true` at any N. Default builds keep the wallclock
  path so interactive desktop sessions retain real-time IRIX timing.
  Tradeoff under `ci_clock`: guest wall-clock no longer tracks host
  wall-clock — exactly what reproducible CI wants.
- **CiSerialBackend chunked-input loss** (Phase 3.5). The SCC channel-A
  RX worker silently dropped bytes when its 8-byte `rx_queue` was full,
  producing the symptom `dd if=/dev/rdsk/dks0d2s0 bs=512` arriving at
  the IRIX shell as `dd if=/d=512`. Fixed by holding the byte in a
  local `pending: Option<u8>` slot and retrying instead of dropping —
  proper flow control: bytes only leave `host_to_guest` when there's
  downstream space. Regression test `long_input_round_trips_without_loss`
  in `src/z85c30.rs`.
- **EEPROM round-trip**: discovered during 1.7 testing that the EEPROM
  has 128 words (not 256). Test corrected.
- **IOC round-trip**: `load_state` re-runs `update_interrupts()` which
  re-derives the MAP_INT0/MAP_INT1 cascade bits in `l0_stat`/`l1_stat`.
  Test now calls `update_interrupts` before the first save so the saved
  state already reflects the cascade — matches what a real running
  machine always shows.
- **Z85c30 default constructor binds TCP** 8880/8881 on `new()`; tests
  use `new_null()` instead so two test instances don't race on the same
  ports. Also the right choice for CI mode (which already used it).
- **Snapshot restore killed a booted IRIX 6.5 guest**: `impl From<L1DTag> for u32`
  (`src/mips_cache_v2.rs:277`) packed `dirty` into bit 27 of the tag word, which
  is `raw_ptag` bit 19 and therefore physical address bit 31. Every dirty L1
  D-cache line deserialized `0x8000_0000` off its own address, never matched
  again, and was never written back, so every store still in L1D at save time
  silently reverted to the value in RAM. `dirty` now lives in bit 0, outside both
  the ptag and `cs` fields. Restore survival went from 0 of 22 to 20 of 20 across
  two independent guests, reproduced on a second harness at 7 of 7. The on-disk
  `dc_tags` encoding changes, so `save_cache_state` writes a `dc_tag_format` key
  and older snapshots are migrated on load rather than misread. See
  `rules/snapshot/l1d-dirty-bit-aliased-physical-address-bit-31.md`.
- **The memory controller charged the guest for time it spent stopped**:
  `last_host_ticks` is a raw host tick count that nothing re-anchored, so the
  first MC read after any pause credited `update_timers` with the whole interval
  and stepped `RPSS_CTR` forward by billions of counts. Observed as a single step
  of `cpu_cycles=16630691385` after a restore, which is 332 s at 50 MHz. Now
  re-anchored in `MemoryController::start()`, which covers restore, the monitor
  and the gdb stub.
- **SCC restore left rr0 contradicting the emptied FIFOs**: `channel_from_toml`
  cleared both queues but kept `status` verbatim. `RX_CHAR_AVAILABLE` restored
  over an empty `rx_queue` has no reachable setter, since `read_data` returns 0
  without touching the bit, so a guest polling RR0 spins on zeroes for good.
  Measured at roughly 1 save in 10 when input is injected shortly beforehand,
  which is the shape of any `iris-ci run` followed by `iris-ci save`.
  `TX_BUFFER_EMPTY` restored clear deadlocks a guest that gates its write on the
  bit, though `save_snapshot` cannot produce that state: the TX thread drains
  the FIFO during the 3.7 to 8.4 ms between `cpu.stop()` and the SCC join.
  Now sets `TX_BUFFER_EMPTY`, clears `RX_CHAR_AVAILABLE`, re-arms
  `tx_int_pending` when WR1 has `TX_INT_EN` and WR5 has `TX_ENABLE`, and calls
  `update_ip` so the SCC and the IOC's `map_stat` agree after `Ioc::load_state`
  restores it wholesale. Independent of the L1D tag fix above: the post-restore
  panic was byte-identical with and without this change. See
  `rules/snapshot/scc-restore-rr0-contradicts-the-emptied-fifos.md`.
- **CHD snapshots captured no disk state**, recorded but not fixed:
  `cow_export` and `cow_import` match only `DiskBackend::Cow`, while `is_cow`
  reports true for a `ChdHd` with a diff. RAM is restored while the disk stays
  in the present. See `rules/snapshot/chd-snapshots-do-not-capture-the-disk.md`.

### Deprecated / Descoped

- **Persistent JIT cache** (was Phase 2.5): descoped. Interp on M2 hits
  Indy parity (60–100 MIPS for integer code). The plan-cited 1.5–2× JIT
  win wasn't worth the maintenance burden of an unstable JIT (still-open
  POST hang on M2, prior Loads-tier and store-correctness issues). JIT
  code stays mothballed behind the existing `--features jit` flag —
  re-enable if a future workload outgrows interp.

### Module map

New modules under `src/`:

| Module | Purpose |
|---|---|
| `sgi_vh.rs` | Minimal SGI Volume Header writer for the scratch volume |
| `chunk_store.rs` | Content-addressable chunk store (BLAKE3, 64 KB) |
| `validate.rs` | Snapshot determinism check (interp two-pass diff) |
| `registry.rs` | Hand-rolled HTTP/1.1 client for snapshot pull/push |

Existing modules with significant changes:

| Module | Changes |
|---|---|
| `snapshot.rs` | Manifest, BinValue (postcard), ChunksManifest, write_state/read_state, write_chunks_manifest |
| `machine.rs` | save/load/restore/rollback orchestration, with_paused, scratch_path, schema-version-aware dispatch |
| `ci.rs` | 15+ new commands |
| `mips_exec.rs` | step_n_inline, state_digest, CpuStateDigest |
| `mips_core.rs` | Deterministic `cp0_compare` recalibration |
| `cow_disk.rs` | Reflink-based overlay capture |
| `z85c30.rs` | RX worker pending-byte hold, save_load_round_trip + long_input_round_trips_without_loss tests |
| `config.rs` | scratch + size_mb on ScsiDeviceConfig |

### Performance numbers (M2 interp)

| Metric | Value |
|---|---|
| Cold restore (disk) | 145–213 ms |
| In-memory rollback | 42 ms |
| Save (warm CAS, no guest changes) | 232 ms |
| Save (cold CAS, first save) | 851 ms |
| 1 MB scratch-write while CPU running | 31 ms |
| 1M-instruction determinism check | 265 ms |
| Snapshot pull (cold local CAS) | 2.9 s / 268 MB |
| Snapshot pull (warm local CAS) | 21 ms / 3.5 MB metadata |
| 100 snapshots from same parent (estimated) | ~1.5 GB total vs ~27 GB without dedup |

### Dependencies added

- `postcard = "1"` — non-self-describing binary serde format for v2 device state and v3 chunks manifest.
- `blake3 = "1"` — content hashing for the CAS chunk store.

No HTTP client dependency added — `registry.rs` uses `std::net::TcpStream`
directly.

---

### `iris-ci` wrapper binary

Driving the CI socket via raw `printf … | nc -U /tmp/iris.sock` proved tedious
and error-prone in real use (long lines, brittle JSON quoting, hand-managed
timeouts, bs=512 foot-guns). New `iris-ci` companion binary replaces all of
that.

#### Subcommands

**Direct passthroughs to socket commands:**
`ping`, `start`, `quit`, `save`, `restore`, `rollback`, `list`, `info`,
`delete`, `tree`, `diff`, `gc`, `validate`, `screenshot`, `pull`, `push`,
`serial-send`, `serial-read`, `serial-wait`, `scratch read`, `scratch write`,
`scratch clear`, `scratch info`.

**High-level macros** for the multi-step rituals that dominate a real CI loop:

- `iris-ci boot` — the full PROM-menu-to-login dance (start CPU + wait
  `Option?` + send `1` + wait `IRIS console login`) in one command.
- `iris-ci login [USER]` — sends username + handles vt100 prompt + waits for
  `#`. Defaults to `root`.
- `iris-ci run "<cmd>"` — sends a shell command, waits for the prompt,
  prints just the captured stdout, returns non-zero on guest failure. Uses
  csh `$status` by default; `--shell sh` switches to `$?`. Solves the SCC
  echo-of-input ambiguity by waiting for `\nIRIS-CI-RC=` (only matches at
  the start of the output line, never inside the typed-input echo line).
- `iris-ci put HOST_FILE [--to GUEST_PATH]` — copies a host file into the
  guest. Stages bytes in the scratch volume, drives the guest with
  `dd if=/dev/rdsk/dks0d2s0 of=… bs=512 count=N` where N is computed
  automatically, then truncates the destination to the original byte length
  with `dd if=/dev/null of=… bs=1 seek=N count=0`. **The user never types
  bs=512 or sector counts.**
- `iris-ci get GUEST_PATH [--to HOST_FILE]` — pulls a guest file out.
  Zeros scratch, drives the guest `dd … bs=512 conv=sync,notrunc` to write
  with sector padding, looks up the byte count via `wc -c`, reads back
  exactly that many bytes from scratch.
- `iris-ci script FILE` — runs a sequence of iris-ci commands from a file
  (one per line, `#` comments, double-quoted args). Each step prints
  `[ok Nms] <line>` or `[FAIL Nms] <line>: <error>`. Aborts on first
  failure with non-zero overall exit.

#### Connection options

- Default socket `/tmp/iris.sock`; override with `--socket PATH` or
  `IRIS_SOCKET` environment variable.
- `--json` for raw JSON responses (scriptable). `--quiet` for silent-on-success.
- Exit codes: 0 success, 1 socket/connection error, 2 iris error response,
  3 local error (file not found, etc.).

#### Implementation

- New binary `iris-ci` at `src/iris_ci_main.rs` (~700 lines), declared as
  `[[bin]]` in `Cargo.toml`. No new dependencies — reuses the existing
  `clap`, `serde_json`, and `std::os::unix::net`.
- Single-request, single-response per invocation. Connects, sends one
  newline-delimited JSON request, reads one line of response, shuts down
  the write side so the server's read loop exits cleanly.

#### What this replaced in the manual test runbook

| Before | After |
|---|---|
| 6-step PROM-to-shell ritual via `printf` + `nc` | `iris-ci boot && iris-ci login` |
| `printf '%s\n' '{"cmd":"serial-send",...}' \| nc …` | `iris-ci serial send "..."` |
| Hand-built `dd if=… bs=512 count=K` recipes for file injection | `iris-ci put localfile.tar` |
| Hand-built `dd … conv=sync,notrunc` + `wc -c` for extraction | `iris-ci get /tmp/foo --to ./foo.tar` |
| Multi-line shell sequences with manual error handling | `iris-ci script tests/scenario.iris` |
| JSON output piped through `head -c` and visually parsed | Pretty-printed tables + `--json` opt-in |
