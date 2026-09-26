# Changelog

All notable user-facing and developer-facing changes to iris.

IRIS has no numbered releases. Binary builds are tagged `v<YYYY-MM-DD-HH-MM>` by
the Release workflow when someone publishes one. This file is grouped by month,
newest first, and within a month by area. Commit hashes are given where a change
is easiest to understand by reading the commit.

## September 2026

### Graphics (REX3)

- **Drawing engine refactor** (`35a3b18`). One generic draw routine
  (`src/rex3_generic.rs`, mode decoding in `src/rex3_shape.rs`) is specialised
  ahead of time into 462 native draw functions (`src/rex3_shaders.rs`, generated
  by `tools/gen_rex3_shaders.py` from a corpus of the draw modes the IRIX desktop
  uses). Most desktop drawing now runs through LLVM-optimised specialised code in
  every build, not only with `rex-jit`. The REX3 JIT and the precompiled set share
  one dispatch table. `src/rex3_simd.rs` is gone; the JIT profile moved to
  `src/rex3_profile.rs`.
- VDMA copes with REX3 reporting busy; `advlast` fixed in the JIT; the REX3
  diagnostic counters were moved off the hot path behind the new default-on
  `rexdiag` feature, so a last-drops build can drop them with
  `--no-default-features` (`f1d0fcb`).
- **Crash fix: `rex-jit` CIDMATCH probe on aux-plane draws.** A draw into an
  overlay plane (OLAY/PUP/CID) with CID checking live re-derived the CID probe's
  offset against `fb_rgb` while the pixel pointer was already `fb_aux`-based,
  reading `fb_aux + (fb_aux - fb_rgb) + off` — a wild pointer that segfaulted the
  REX3 thread under X11. `Dm1::use_aux()` now picks the base in both shader
  emitters (`rules/rex3/cidmatch-aux-plane-base.md`).
- The GFIFO push is retryable, and a shader rejected by a full compile queue can
  be requested again (`rules/testing/rex-jit-queue-retry.md`).
- `CIDMATCH` is a mask of permitted CIDs, not an equality value
  (`rules/rex3/cidmatch-is-a-mask.md`).
- Blend-alpha handling fixed (`rules/rex3/blendalpha-and-alpha-blending.md`).
- REX3 benchmarking tests; a triangle benchmark in `gltest`.

### CPU and timing

- **CP0 Count runs at a fixed 33 MHz** (`066935b`). The slow/fast tick detection
  and Count/IP7 frequency inference are gone; a constant rate proved more stable.
  IRIX reports it as a 66 MHz CPU. `[clock] fixed_mhz` / `--clock-fixed-mhz`
  override it.
- **Guest clock offset** (`[rtc_offset]`, iris-gui General → Real-time clock).
  Start the DS1386 RTC shifted from host time by signed years, months, days,
  hours, minutes and seconds, e.g. `years = -18`. Applied only when the RTC is
  seeded from the host at startup; clamped to the chip's 1970–2039 range.
- IP7 delivery improved for Linux guests' timer checks and calibration.
- Misaligned-fetch exception, and Config.K0 cache modes including the reserved
  ones (`rules/irix/cache-attributes-and-fetch-alignment.md`).
- FR=1 FPU mode handled better, with a guard for `MOVCI`
  (`rules/jitv2/fr1-pin-must-be-re-derived.md`,
  `rules/testing/movci-is-cp1-by-behaviour-not-encoding.md`).

### JIT v2

- **Corpus capture moved to the pcp cache; `jitv2_corpus_dump` removed.** The
  old Cargo feature dumped a raw 4KB page per compile request from inside the
  compile worker, with one entry offset encoded per filename and a
  `PhysicalCodePage::saved_bits` bitmap to dedup them. It had to be compiled in
  *before* the run that would produce the interesting pages, so a corpus could
  only be captured by deciding in advance you wanted one — and the corpus every
  `rules/jitv2/` measurement cites was a local artifact that got lost, with no
  way to reproduce it.

  `j2 corpus [dir]` now walks the live JIT page cache on demand and writes one
  `.pcp` per page — the format `j2 dumppcp` already used, bumped to `IRISPCP2`
  with a `call_count`, for weighting a measurement by how hot a page actually
  was. The format stays entirely physical: no virtual address is recorded,
  because jitv2 compiles PIC precisely because one physical page is shared
  across processes, so no single VA is meaningful. `IRISPCP1` files still
  read. Both
  `j2 dumppcp` and `j2 corpus` now work under **both** `comp.rs`
  implementations, not just `j2wp`: the default impl reconstructs the entry
  bitmaps from its per-`JitEntry` state. `zz_corpus_sizes` consumes a directory
  of `.pcp` files via `IRIS_CORPUS_DIR` instead of a file of filenames.

  Also fixed along the way: `Codegen::last_code_size` was `developer`-gated,
  which made the one number a codegen-size measurement needs available only in
  the build that invalidates such a measurement (`developer` forces
  `opt_level=none` and injects a per-instruction trace callout) — so
  `zz_corpus_sizes` reported `total_bytes=0` in exactly the build it was meant
  to measure. See `rules/jitv2/corpus-capture-from-the-pcp-cache.md`, including
  the `requested`-vs-`compiled` trap that silently discards 99.5% of a corpus.

- **Compile churn avoidance (`j2wp`).** Each page now remembers the bytes its
  last compile decoded, the entry points it published and the FR mode it was
  built for. A later compile request whose generation moved but whose *decoded*
  words are all unchanged re-validates the installed function instead of
  recompiling it — the common case where code and data share a 4KB page, so a
  write to the data bumps the generation and used to force a full
  analyze+codegen+finalize that emitted byte-identical code. On an IRIX 6.5
  boot this skips ~286k compiles, 98% of everything checked. Costs ~20MB
  (`PhysicalCodePage` 720 → 4976 bytes across the 4096-slot pool). `j2 status`
  and `j2 pcp` report the skipped/recompiled split (since last flush, and
  available in `lightning` builds, not just `developer`). See
  `rules/jitv2/compile-churn-avoidance-snapshot.md`.
- Inline load/store for the R5000 cache model, not only the R4400.
- Physical code pages are found through a flat pfn array instead of a hash map;
  PC/BD stores are emitted only when needed; the last instruction on a page and
  excluded instructions share one path (early September).

### Networking

- **Guest DNS goes to the host's DNS server** (`src/host_dns.rs`): the first IPv4
  `nameserver` in `/etc/resolv.conf`, or the active adapter's server on Windows,
  re-read every few seconds so a VPN coming or going needs no restart. `8.8.8.8`
  is only the fallback.

### Testing

- **cpu-tests validated on real SGI hardware** (`5150db0`). An Indy R4400 rev 6.0
  and an Indy R5000 rev 1.0 both pass every check; the logs are in
  `cpu-tests/oracle/`. All 15 original FP failures are confirmed IRIS bugs.
  `run/diff-hw.py` classifies a hardware log against an emulator one.
- CI now gates cpu-tests on the real failing count per CPU instead of on zero.
- New tests: CP0 instructions from User/Supervisor mode must raise Coprocessor
  Unusable; `identity/config_k0` is back; `mem/load_then_use`,
  `load_then_trap`, `load_then_more` and `fpu/trap_behind_a_load` cover the
  instructions right behind a load; an R4600 case with every expectation's
  source written down (`cpu-tests/docs/r4600.md`).
- `cp0/count_writable` no longer varies between runs.
- The prebuilt bench guest image was refreshed.

### iris-gui

- **Optional native macOS front-end** (`--features macos-gui`): menus in the
  system menu bar, the configuration editor and every dialog in their own OS
  windows, and the run state in the window title, so the main window holds only
  the display. Off by default and ignored off macOS; the default layout is
  unchanged. See `rules/gui/macos-gui-front-end.md`.
- Scaling and resize fixes (`e93c5bb`): the VM screen scale is now the maximum
  draw scale, so a larger window centres the picture instead of stretching it;
  a snap-to-size request made while fullscreen is applied when fullscreen ends.
- **Windows crash fixes** (#94): window resizes are sent to the main event-loop
  thread instead of being called from other threads, the GL surface and the first
  `make_current` happen on the main thread, and further cross-thread UI fixes.
- `crash_diag`: silent Windows deaths (`0xC000041D`) are logged with a symbolised
  stack to `iris-crash.log` (`rules/gui/windows-silent-exit-0xc000041d.md`).

## August 2026

### CPU

- **The CPU model is a runtime setting** (`e677edf`, `29158fc`). R4400 vs R5000
  used to be 99 `#[cfg(feature = "r5k")]` sites and a rebuild per CPU. Both
  models are now types monomorphised into every binary; `[machine] cpu`,
  `--cpu`, the GUI and `iris-bench` pick one. Snapshots record the CPU and refuse
  a crossed restore. The `r5k` feature is vestigial; `r5ksc`/`r5ksc_triton`
  refuse to build until the R5000 L1I bugs are fixed.
- R5000 IRIX boot hang fixed: the PROM is told there is no L2 (`98332ed`).
- Six `CpuDevice` methods that recursed instead of forwarding were fixed.
- `WAIT` in the disassembler; debugger fixes (breakpoints no longer alias
  adjacent instructions, `EXEC_RETRY` is retried in both debugger paths, a PC
  breakpoint waits until its instruction is fetchable, the step-off-breakpoint
  skip also applies to `step_one`); the GDB stub returns an error for unreadable
  memory instead of zeros.
- FPU flag synchronisation and rounding-mode fixes found by cpu-tests; R5000 L1
  cache test fixes.
- `MipsCore` reorganised: the interrupt word, `in_delay_slot` and its target live
  in the core; fields reordered for cache locality; the cycles counter is a plain
  variable. Cranelift upgraded 0.116 → 0.134, and every dependency except the
  egui stack updated (`rules/build/dependency-upgrade-gotchas.md`).
- IP7 is driven by a host timer instead of instruction counting (`1e05210`), and a
  fixed CP0 Count clock option was added (superseded in September).

### Memory, caches and translation

- **ppmem**: host-MMU-backed physical memory (`--features ppmem`,
  `docs/ppmem-design.md`).
- **tcache**: transparent cache on top of ppmem (`--features tcache`,
  `docs/tcache-design.md`), with `tcache_verify`.
- **nutlb**: a direct-mapped data-side translation cache, later reworked around a
  validity bitmask and made permanent (`docs/nutlb-design.md`). nanotlb is
  invalidated on ASID changes. `tlbcheck` walks the JTLB after every write;
  `jitstats` instruments the inline-memory path.

### JIT v2 (new) and the old JIT (removed)

- **jitv2** (`c340118`): a physical-page Cranelift compiler with memory-resident
  registers and no speculation (`rules/jitv2/jit-v2-design.md`). Over the month:
  a multi-threaded compile pool with a lock-free queue (`[jitv2] threads`,
  `--jitv2-threads`), compiles triggered on page transitions, FMOVCF and DADDIU
  emitters, interpreter fallbacks that don't break streaks, NOP elimination,
  lazily materialised cycle counts, inline L1D loads/stores for lines already in
  cache, a dirty-cache-page probe, self-modifying-code detection, and a Windows
  x64 callout ABI that returns status in registers.
- `jitv2_opcodefusion` (LUI+ORI/ADDIU, branch+NOP) exists but is **off by
  default** after it broke Linux (`rules/jitv2/jitv2_lui_fusion_foreign_delay_slot_hazard.md`).
- `j2wp` whole-page compile, `jitv2_lockstep`, `jitv2_smc_check`,
  the `j2` monitor command (`clear`, `deny`, `pagewb`,
  `html` physical code page visualiser, …) and the `jitv2_analyze`,
  `jitv2_verify`, `jitv2_pcp_dump` tools.
- Status-bar feedback for JIT activity.
- **The original tiered MIPS JIT was removed** (`33c4e68`), along with its
  `jit` feature, `IRIS_JIT*` environment variables and `rules/jit/`.

### Bare-metal testing and benchmarking

- **cpu-tests** (`866925c` onwards): a self-checking MIPS III/IV suite that runs on
  the emulated CPU with no OS — ALU, mul/div, memory, branches, exceptions, CP0,
  TLB, FPU (88 tests), caches and MIPS IV — plus `run/matrix.sh` for R4400/R5000 ×
  interpreter/jitv2 and a CI workflow.
- Emulator support for it: `--load-elf` and the `loadelf`/`loadbin` monitor
  commands (`src/elf.rs`); a default-off **test device** in GIO slot 0 with guest
  console, machine-state JSON dump and exit code (`--test-device`,
  `src/testdev.rs`), later with a host clock and a retired-instruction counter.
- **bench/** (`07d8a3b`): 46 kernels in six groups, each checksummed against a
  golden value, reporting throughput, guest MIPS and an accuracy score.
  **iris-bench** runs, compares and sweeps builds (`matrix`, `host`).
  `bench/irix/` covers workloads under a booted IRIX.

#### Benchmark, for everyone

- **The benchmark runs in-process, on every platform.** `iris-bench run` and
  the GUI's Benchmark tab no longer spawn anything: the guest binary is linked
  into `iris` (`bench/prebuilt/`, `src/benchsuite.rs`) and runs on a headless
  machine the emulator builds for itself (`iris::bench_runner`). No MIPS cross
  toolchain, no ELF on disk, no subprocess, and nothing written outside the
  application container — so the Benchmark tab ships in App Store builds.
  `iris-bench run --iris PATH` still measures a separate binary in a subprocess,
  which is what `matrix` needs.
- **The bare-metal suite got 2.5x faster** (117 s → 46 s for a full r4400
  lightning run). `--load-elf` skips the PROM, so the SCC's transmitter was never
  enabled and every character the guest printed burned a 100,000-iteration spin
  waiting for a TX-empty bit that would never come. `cpu-tests` gets the same
  speedup. See `rules/testing/scc-serial-output-from-bare-metal-code.md`.
- **Quick mode** (`iris-bench run --quick`, and the GUI's default): about half
  the wall clock for the same numbers to within a couple of percent. It never
  runs fewer kernels, it only shortens the timed passes. Requested through a
  test-device register (`TESTDEV_RUN_CONFIG`); every field means "unrestricted"
  when zero, so older emulators are unaffected. Refused by `iris-bench reference`.
- **Every result records the machine it measured**: CPU identity and revision
  and the L1/L2 geometry from CP0 Config, and the RAM banks from MEMCFG, printed
  as a header and as `#cache` / `#memory` lines.
- **Any MIPS CPU is identified and runs.** The suite names R4000, R4400, R4600,
  R4700, R5000, R8000, R10000, R12000, R14000, RM5200 and RM7000 from PRId and
  runs on anything else too; `golden.h` was always CPU-independent.
- **Platform limits documented** in
  `rules/testing/bare-metal-harness-platform-assumptions.md`.
- `iris::bench_report` holds the report parser, data model and reference table,
  shared by the CLI, the runner and the GUI.

### Platforms and devices

- **Indigo2 (IP22)**: its own PROM with an embedded fallback (`src/prombini2.rs`),
  INT2 and the fullhouse interrupt layout, a second SCSI controller, NVRAM in the
  93CS56 serial EEPROM (`nveeprom`), MC revision bump for the PROM, and vertical
  retrace delivery for Indigo2 graphics.
- **DaynaPort SCSI/Link** (`--features daynaport`, `734660b`): Ethernet over the
  SCSI bus as a per-ID target with its own NAT or PCAP backend
  (`docs/daynaport.md`, `docs/iris-daynaport-target.md`).
- **TFTP server** in the NAT gateway for PROM network boot (`--tftp-dir`,
  `src/tftp.rs`).
- **SGI volume headers**: volume-directory support and the `mkvh` tool.
- SCSI fixes for Linux (phantom LUNs, mode pages, MODE SENSE(10)) and the Indigo2.
- PS/2: report that the aux mux is unsupported (fixes the mouse in Debian 7);
  keyboard and mouse enable/disable toggles.
- REX3 framebuffer dump (`rex fbdump`); 8-bit DCB writes packed into the top of
  the word; alpha compare fixed.

### iris-gui

- **Benchmark tab** (in-process, ships everywhere) and a **CPU picker** shown
  wherever the machine is described.
- JIT and Ultra64 are opt-in build features rather than always on.
- Sends physical key positions instead of layout-translated keys (#72); captures
  the macOS window handle on the main thread (abort on first frame); file dialogs
  open where the file is; never pairs a file name with a directory on macOS.
- Installer: the R5000 build got its own AppId, then was dropped when the CPU
  became a setting.

### Tooling, CI and diagnostics

- **Release and App Store pipelines** moved into this repo (`f40a1c1`):
  manual dispatch, dry run by default, publish on request. One variant per
  platform: lightning + rex-jit + camera + chd.
- One `suites.yml` workflow for both bare-metal suites.
- `iris-ci`: the socket read timeout follows the caller's deadline instead of a
  fixed 300 s.
- Stack sizes raised for machine construction and tests.
- Build warnings cleaned up; the `r5ksc_triton` and non-JIT builds fixed.
- `scripts/build-manifest.sh` emits a per-build crate manifest.
- Snapshot fixes: the MC timebase is re-anchored at start, HPC3 PDMA latched
  flags are serialized, SCC RR0 is normalised against emptied FIFOs, and the L1D
  dirty bit is encoded where the hardware keeps it (late July).

## July 2026

### Graphics

- Removed dirty-rectangle tracking; fixed line (`fline`) drawing, line stipple
  (`lspattern`/`zpattern`) and accelerated quads/spans; fixed a packed HOSTRW read
  that drew black bars; fixed SoftWindows corruption (#46).
- GL compositor falls back from GL 3.2 / GLSL 1.50 to GL 2.1 / GLSL 1.20; the 2px
  framebuffer offset is gone, and the 1024x768 mode displays correctly.
- Screenshots after the first work with the GL compositor.
- The status bar shows the IP7 rate instead of the (constant) refresh rate.

### CPU

- **Interpreter opcode fusion** (`252efb5`, part of `lightning`): branch+NOP,
  LUI+ADDIU/ORI and add/sub+load/store pairs dispatch as one.
- Inlined completion tails.
- Per-instruction statistics (`instr_stats`); FPU rounding modes and control-bit
  reads/writes fixed; targeted FPU-instruction logging.
- Delay-slot fix for a branch that is not taken.

### Platforms and devices

- **Indigo2 IP22 platform** (`f2d0bff`): `[machine] profile`, fullhouse MC/IOC,
  a Newport XL on the GIO graphics slot, plus preview stubs for the Indy XZ/Elan
  board (`src/xz.rs`) and Indigo2 IMPACT (`src/mgras.rs`), dual-head Newport and
  forced Newport resolutions.
- IndyCam CDMC register map and power-on defaults corrected
  (`rules/irix/indycam-cdmc-register-map.md`).
- NFS resolves `.` and `..` server-side.
- NAT: port forwards to rsh/rlogin use a reserved source port, probing for a free
  one (`rules/irix/rsh-forward-needs-reserved-source-port.md`).
- Recent HAL2 changes reverted after they broke audio.
- `libchdman-rs` 0.288.8 (BSD-3-Clause; drops GPL-3.0), then 0.288.9 (bin/cue fix).
- Machine and other large objects are constructed with bigger stacks (#60).

## June 2026 (and late May)

### iris-gui (new)

- **Optional egui front-end** (`a8b8262`, contributed via danifunker's fork):
  named machines with autosave in `gui.json`, iris.toml import/export, embedded
  framebuffer, mouse and keyboard capture, safe-stop dialog, icons.
- Live MIPS readout; fewer framebuffer copies; X11 mouse capture fix;
  IntelliMouse wheel support in the core.
- **Mac App Store support**: security-scoped bookmarks, interpreter-only under the
  App Sandbox, notarised-distribution entitlements, winit's private blur API
  stubbed out (guideline 2.5.1, `rules/macos/appstore-private-api.md`), camera
  and network entitlements made testable, keyboard capture, CHD folder grants,
  licenses window, and `PRIVACY.md`.
- Adaptive framebuffer filtering and the left control column; windowed-first
  sizing with a VM-screen scale; NVRAM at a stable per-user path, seeded on
  first run, with automatic MAC detection and writing; managed location for new
  disk images; NET status light; redesigned Networking tab with live subnet
  changes and a "Check networking" dialog; CHD copy-on-write UI; powered-off
  overlay; `bundled` build feature and "Use embedded PROM".
- GL teardown runs on the refresh thread that owns the context (fixes a segfault
  on power-off and window close).

### Networking

- **In-process NFS server** (`src/nfsudp.rs`, eight increments ending `685f534`):
  NFSv2 for IRIX 5.3 and NFSv3 for 6.x, MOUNT v1/v3, a duplicate-request cache,
  IP-fragment reassembly, and READDIR that respects `count`. The external
  `unfsd` and its `--unfsd`, `--nfs-port` and `--mountd-port` options are gone.
- **PCAP bridged networking** (`--features pcap`, `2c83ac8`), with capture-
  permission elevation and installer plumbing, an NFS responder on a virtual LAN
  IP (`nfs_pcap_ip`), live NIC changes, and a fix for RX starvation on busy LANs.
- **XDMCP** reverse-proxy helper (`src/xdmcp.rs`, `docs/xdmcp.md`).
- FTP passive-mode helper for inbound port forwards; port forwards can be
  rebound live; NAT adoption and "networking off" diagnostics.
- Time and NTP answered by the gateway (late May).

### Storage

- **Hot-swappable CD-ROM** (#47): load a disc at runtime with RCtrl+F12 (CLI) or
  Ctrl/Cmd+F12 (GUI), empty trays, and three changer bugs fixed.
- **CHD copy-on-write** (`fe8d449`): writes go to a `.diff.chd` sidecar and are
  folded back into the base on exit ("Synchronizing disks…").
- **WD33C93A rewrite** for OpenBSD (`1cdf836`), plus SCSI fixes for Linux, NetBSD
  and OpenBSD and a `scsi_deferred_int` setting for the BSDs.

### Other emulation

- **Ultra64 N64 development board** (`--features ultra64`) in GIO slot 0, bridged
  over shared memory to a modified gopher64.
- GL-based display compositor, window resizing and fullscreen.
- vmap TLB indexing fixed for NetBSD's large pages; serial TX-empty interrupt no
  longer fires constantly (Gentoo).
- Mouse clicks on the IndyCam image fixed by clip-mode and CID checks in the REX3
  JIT; compositor modularised.
- Indycam capture works on Linux hosts (V4L).

### Late May

- **VINO / IndyCam** end to end: pixel pipeline, CDMC and SAA7191, host camera
  capture on macOS, SYSID bit 4 so IRIX attaches the driver, I2C fixes, capture
  on IRIX 6.5, colour and interlace fixes (`rules/irix/vino-*`,
  `rules/irix/indycam-end-to-end-capture.md`).
- **Idle park** (`--features idle-pause`, off by default): the CPU thread sleeps
  while IRIX idles. The REX3 GFIFO consumer parks instead of spinning, the refresh
  thread skips unchanged frames, and the winit loop waits when idle.
- IRIX install guide for 5.3 and 6.5.22 (`rules/irix/irix-install.md`) with
  `tools/inst-*.py` helpers; config templates `iris-irix53.toml` and
  `iris-irix65.toml`; per-config `nvram = "..."`.
- SCSI: IRIX miniroot install hang fixed
  (`rules/irix/miniroot-install-hang-scsi0-dma-irq-storm.md`).
- `--serial-log` mirrors ttyd1 output; the monitor port stays bound under `--ci`;
  telnet option negotiation on the serial and monitor listeners.
- `iris-ci rtc-save`, `cdrom-eject`, `cdrom-load`; `get`/`put` work under a
  `/bin/sh` guest shell.
- Monitor: `ps2 type`/`enter`/`status`, `proc info`.
- `chd_extract` tool.
- First R5000 support (slower than R4400 under the interpreter because every
  cache access probes two ways).
- Enabled build features are printed at startup.
- A configured SCSI device that can't attach is a fatal error.

## May 2026 — snapshots, CHD and CI

- **CHD images** (`--features chd`, May 18–20) for SCSI disks and CD-ROMs, via the
  `libchdman-rs` crate.
- Configurable NAT subnet; unprivileged ICMP on macOS.
- `tlbvmap` on by default, TLB translation statistics, a shadow TLB with cooked
  values.
- `gr_osview` and `jot` fixed on IRIX 5.3; 12bpp colour-index decoding fixed.
- CD-ROM enabled by default; TCP forwarding regression from the CI work fixed.
- NetBSD no longer hangs on DCB access.

The snapshot and CI work below landed on 2026-05-03.

The headline is a complete snapshot/rollback stack: capture
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

### Deprecated / Descoped

- **Persistent JIT cache** (was Phase 2.5): descoped. Interp on M2 hits
  Indy parity (60–100 MIPS for integer code). The plan-cited 1.5–2× JIT
  win wasn't worth the maintenance burden of an unstable JIT (still-open
  POST hang on M2, prior Loads-tier and store-correctness issues). At the
  time the JIT stayed mothballed behind `--features jit`; that JIT was
  removed entirely in August 2026 and replaced by jitv2.

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

#### `iris-ci` wrapper binary

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

#### What this replaced in the manual test runbook (since deleted)

| Before | After |
|---|---|
| 6-step PROM-to-shell ritual via `printf` + `nc` | `iris-ci boot && iris-ci login` |
| `printf '%s\n' '{"cmd":"serial-send",...}' \| nc …` | `iris-ci serial send "..."` |
| Hand-built `dd if=… bs=512 count=K` recipes for file injection | `iris-ci put localfile.tar` |
| Hand-built `dd … conv=sync,notrunc` + `wc -c` for extraction | `iris-ci get /tmp/foo --to ./foo.tar` |
| Multi-line shell sequences with manual error handling | `iris-ci script tests/scenario.iris` |
| JSON output piped through `head -c` and visually parsed | Pretty-printed tables + `--json` opt-in |

## April 2026 — initial release

- First public code (2026-04-01): an SGI Indy (R4400) emulator booting IRIX 6.5
  and 5.3 with Newport graphics, HAL2 audio, SCSI, the SEEQ Ethernet with a NAT
  gateway, PS/2 input, a monitor console and serial ports.
- Early additions: 2x window scale, ICMP on Windows, unfs3-based file sharing
  (replaced in June), port forwarding, headless mode, copy-on-write disk
  overlays, the first MIPS JIT (removed in August), the REX3 draw-shader JIT
  (`rex-jit`), a custom GFIFO, TLB vmap and nanotlb fast paths, many interpreter
  micro-optimisations, the GDB stub, screenshots, CD-ROM block size fixes and
  monitor commands for the CD changer.
