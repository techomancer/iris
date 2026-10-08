# Changelog

All notable user-facing and developer-facing changes to iris.

IRIS has no numbered releases. Binary builds are tagged `v<YYYY-MM-DD-HH-MM>` by
the Release workflow when someone publishes one. This file is grouped by month,
newest first, and within a month by area. Dates use the Git committer date
(`git show -s --format=%cs`), which records when a change landed in this
history; author dates can differ after rebases or cherry-picks. Historical
feature names describe the build at the time, not flags to use today. Commit
hashes are given where a change is easiest to understand by reading the commit.

## October 2026

### UI / input

- **The `iris` window lets go of held keys and buttons when it loses
  focus.** Cmd-Tab or Alt-Tab is pressed in the window and released in the
  next one, so the guest kept those keys (and any held mouse button) down: a
  stuck Shift made Motif scrollbars ignore every click. On focus loss the
  window now releases everything the guest still sees held.
- **Right Cmd releases the grab on macOS**, like Right Ctrl, which Mac
  keyboards lack.
- **Japanese (JIS) keyboards.** With the PROM set to the Japanese layout, `]`
  typed `\`, `}` typed `|` and `_` typed nothing. The new `keyboard = "jis"`
  setting (`--keyboard jis`; also `iso`) sends the set-3 code of an SGI
  keyboard for the key left of Enter (`0x53`, not the US backslash's `0x5C`).
  In iris-gui it's **Keyboard** on the Display tab. ろ, ¥, 無変換, 変換 and
  かな now send keys in all three scancode sets. Codes
  are from xkeyboard-config's `sgi_vndr/indy(jp106)`.

### Build / features

- **IMPACT raster JIT (`gr4-jit`).** The registers that shape an IMPACT
  primitive's pixel pipeline reduce to a 64-bit key, and Cranelift compiles
  one shader per key. It covers fills, X lines, character stipple, transfer
  lines, and GL triangles and lines with their tests, blending, texturing
  and fog. Shaders are bit-exact with the interpreter, which a key-space
  sweep proves board by board (`src/dev/mgras/rss_jit_tests.rs`). They are
  2.5-6x faster on representative primitives. Shaders compile in the
  background, and `mgras jit` in the monitor shows and switches the JIT. The
  RSS and TE1 state is now plain `#[repr(C)]` data: `Option`s gave way to a
  plain `Slot`, and the sampler and setup structs became flat.
- **Retired the `chd`, `camera`, `ultra64`, `daynaport`, `ip28`, `ppmem`,
  `mips4` and `r5k` cargo features.** CHD images, the host camera, the
  Ultra64 dev board, DaynaPort and the IP28 / R10000 machine are always built
  in and enabled per machine in the config. Physical RAM is always ppmem
  (host-MMU mapped); RAM banks stay bus devices, so DMA and other bus-path
  accesses are unchanged. iris-gui drops its matching passthrough features
  and the "rebuild with --features ..." hints. Old snapshots that recorded
  `chd`/`camera` in their manifest still restore.
- **MIPS IV is purely a property of the configured CPU.** The interpreter
  already monomorphised on `C::MIPS4` (R4400 gets its own MIPS III decoder);
  jitv2's `Analyzer` carries the same answer as a runtime flag, now defaulting
  to MIPS IV for tools that have no CPU to ask.
- **`jitv2` implies `tcache`.** The JIT now always runs over the transparent
  cache. The compile-time dirty-page probe (`jit_page_has_dirty_lines`,
  `RejectReason::PageDirtyInCache`) and the non-tcache inline load/store path
  (`jit_dc_data`) existed only for jitv2 without tcache and are gone.
  `tcache` can still be enabled alone for an interpreter build.

### iris-gui

- **NVRAM EEPROM gets the same stable-path treatment as NVRAM.** `nveeprom`
  (Indigo2/IP28's motherboard EEPROM — where `eaddr` and PROM env actually live
  on those profiles, not in the NVRAM file) used to default to a bare
  `"nveeprom.bin"` with no anchoring, no migration, and no config-editor field,
  unlike `nvram`. Since iris-gui never changes its working directory, that bare
  default resolved against whatever CWD the OS happened to launch it with —
  often not the same place twice between `cargo run` and a bundled `.app` — so
  a machine could silently load a different (usually blank) EEPROM each
  launch, and "Reset NVRAM"'s MAC write (which only ever touched `nvram`) had
  no effect on Indigo2/IP28's actual `eaddr`. Now: a stable
  `<config dir>/iris/nveeprom.bin` default (`GuiSettings::default_nveeprom_path`),
  the same relative-path migration `nvram` gets on load, and a "NVRAM EEPROM
  file" row on the General tab right below NVRAM file.
- **"Reset NVRAM" and the no-MAC-yet pre-flight now also patch the NVRAM
  EEPROM.** Both only ever touched the DS1386 `nvram` file — on Indigo2/IP28,
  which read `eaddr` from `nveeprom` instead (see above), this had no effect
  on the guest's actual Ethernet MAC at all: the toast would report a fresh
  MAC, but IRIX kept reporting the static `config::DEFAULT_MAC` because
  `nveeprom` stayed blank and core's own blank-EEPROM fallback filled it in
  with that default on every boot. Added `nveeprom`-side equivalents of every
  `nvram` MAC helper (`nveeprom_has_mac`, `write_nveeprom_mac`,
  `ensure_nveeprom_exists`, `reset_nveeprom`) and call them alongside the
  `nvram` ones, so both chips get a real MAC regardless of machine profile.
### Graphics and host OpenGL

- **2026-10-06 — IMPACT OpenGL and texture pipeline** (`862d0fe`): GE11
  command HLE, shared matrix/clipping/lighting routines in `src/dev/gl/`,
  triangle rasterization, context storage and eviction through ERAM, and
  tiled pixel memory selected by DRB/XMAP pointers. TE1 handles texture
  downloads, RGBA4 decoding, wrapping, interpolation, and LOD for workloads
  including OKR. Scanout follows guest VC3 timings, with 2048×2048 raster
  storage and larger shared display buffers. Double-buffer selection,
  Z/stencil, and complete board snapshots remain unfinished.
- **2026-10-03 — HostGL accumulation buffer** (`6bc2082`): implements
  `glAccum` operations with a software accumulation buffer and pixel readback;
  restores GL state after returning accumulated pixels to the colour buffer.
- **2026-10-03 — HostGL packed pixel enums** (`181c8b1`): keep the standard
  numeric values for packed pixel types instead of conflicting definitions.
- **2026-10-02 — GR2 rendering and transfer fixes** (`0cd1bae`, `7728ca2`,
  `f9eceab`): Mandelbrot/XZ rendering, pixel read/write DMA for snoop, and
  irisGL vertex colour, lighting, and ambient-light behaviour for X backgrounds.
- **2026-10-02 — IMPACT opaque line stipples** (`2231d58`): stipple gaps draw
  in the background colour instead of remaining transparent.

### IP28, memory, and battery-backed state

- **2026-10-04 — IP28 512 MB banks** (`c16e18d`): the MEMCFG installed-size
  decoder accepts the PROM's `(size_field=31, rank=1)` encoding. Two 512 MB
  banks provide 1 GB; IRIX 6.5 reports 1024 MB after POST. The GUI offers
  512 MB banks and 768/1024 MB totals. IP22/IP24 still reject banks above
  128 MB; the fourth-bank policy is unchanged. See
  [IP28 512 MB banks](rules/irix/ip28-512mb-banks.md).
- **2026-10-04 — IP28 boot chime** (`ce56123`): report R10000 revision 2.5
  so the PROM takes its supported audio path; correct PBUS register decoding
  and masking used by the tune's DMA setup.
- **2026-10-04 — Save NVRAM and EEPROM on Stop** (`5abfa7f`): after the CPU
  stops, HPC3 saves the DS1386 and the Indigo2 motherboard EEPROM to their
  configured paths. Normal GUI Stop/Quit and guest power-off retain PROM
  changes. Automatic and manual saves use a flushed temporary file followed
  by replacement, preserving the previous image on failure. Fresh IP28 EEPROMs
  get PROM defaults and a valid checksum before boot, including volume `80`
  and boottune `1`; existing settings are preserved.
- **2026-10-04 — IRIX memory stress utility** (`e966d54`): adds
  `test/memstress/` for walking an allocated mapping in both directions,
  verifying indexed 64-bit patterns, and checking pages after discard/refill.

### Monitor and serial ports

- **2026-10-04 — Configurable monitor port** (`d8b8131`): top-level
  `monitor_port` / `--monitor-port`, default `8888`. The terminal connects
  only to its own machine's bound listener; a failed bind no longer attaches
  it to another emulator's monitor.
- **2026-10-04 — Configurable serial ports** (`d6184b8`):
  `serial_port_a` / `--serial-port-a` and `serial_port_b` / `--serial-port-b`,
  defaults `8880` and `8881`. The GUI serial console and halt action follow
  channel B's configured port. Occupied serial ports warn and use null backends.

### Build, features, and CI

- **2026-10-01 — Retired device and CPU Cargo features** (`4d6e97c`): `chd`,
  `camera`, `ultra64`, `daynaport`, `ip28`, `ppmem`, `mips4`, and `r5k` are
  removed. Devices are built in and selected per machine. Physical RAM always
  uses ppmem; banks remain bus devices for DMA. MIPS IV follows the configured
  CPU in both engines. Old snapshots listing `chd`/`camera` still restore.
  `jitv2` implies `tcache`; the non-tcache JIT memory path and dirty-page probe
  are removed. Interpreter builds can enable `tcache` separately.
- **2026-10-01 — Source layout** (`af9433f`, `51789b2`, `484ca1a`, `e9e52c6`,
  `061b347`): CPU and JIT code move to `src/cpu/`, devices to `src/dev/`,
  Newport to `src/dev/ng1/`, GR2/IMPACT to their device directories, and
  network services to `src/net/`.
- **2026-10-01 — Bare-metal CI expectations** (`2414459`, `320c38a`): build
  matrix drops retired flags; cpu-tests baselines account for `jitv2`'s
  mandatory transparent-cache semantics. Failing-check limits are 124/126
  for R4400 interpreter/JIT and 108/110 for R5000 interpreter/JIT.
- **2026-10-02 — Workflow runtime updates** (`e0f0e66`, `0529517`): opt
  GitHub Actions into Node.js 24 and update App Store artifact upload.

### iris-gui and configuration

- **2026-10-01 — Stable motherboard EEPROM paths** (`9f17c3d`): default
  `nveeprom` to `<config dir>/iris/nveeprom.bin`, migrate relative paths on
  load, and expose the file in General. Indigo2/IP28 PROM environment and
  MAC live in this chip; Indy uses the DS1386 instead.
- **2026-10-01 — Reset both battery-backed chips** (`36bfb4c`): Reset NVRAM
  and the no-MAC pre-flight also seed and patch the motherboard EEPROM, so
  the GUI's selected MAC reaches Indigo2/IP28's actual `eaddr`.
- **2026-10-01 — Persistent JIT cache settings** (`71790bd`): expose
  `[jitv2] cache` and `cache_dir` rather than a Debug-only environment toggle.
  Default false/blank settings preserve externally supplied `IRIS_JIT_CACHE`
  / `IRIS_JIT_CACHE_DIR`; explicit enabled/nonempty settings replace them.
  With no environment override, defaults leave the cache off.
- **2026-10-02 — Cache control in General** (`5582e96`): persistent JIT cache
  controls remain accessible in lightning builds, which hide the Debug tab.

## September 2026

### Configuration and desktop usability

- **2026-09-29 — Debug environment precedence** (`e6bde88`): externally
  supplied debug environment variables survive default false/blank GUI/TOML
  settings. Explicit enabled/nonempty settings still replace those variables.
- **2026-09-30 — Desktop and GUI refinements** (`e38f38a`, `05242a6`,
  `6aa3a21`): UI updates, resize locking, and gr_osview startup fixes.
- **2026-09-30 — Example configuration defaults** (`08be293`): restore
  standard settings in `iris.toml` after diagnostic runs.

### Graphics (GR2 and IMPACT)

- **2026-09-28 — GR2 XZ/Extreme** (`5d07b4d`): replaces the XZ preview stub
  with an HQ2 command interpreter, RE3 software rasterizer, GE7 diagnostic
  storage, VC1/XMAP5/Bt457 display path, and framebuffer presentation.
  XZ is selectable on Indy/IP22; Extreme is restricted to IP22. Later
  September fixes add irisGL tokens, glyphs, polygons, window-ID clipping,
  context state, and homogeneous clipping (`2383f26`, `30fb4ff`, `a528f18`,
  `e806066`, `95b7ad0`, landed 2026-09-29). The GUI gains the corresponding
  GR2 board picker on the same date (`bb286ea`).
- **2026-09-30 — IMPACT/MGRAS board model** (`0b1ea38`): replaces the ID stub
  with HQ3 command processing, GE11 diagnostic storage, RSS 2D rasterization,
  VC3/XMAP/colormap/DAC scanout, interrupts, tracing, and replayable recordings.
  The OpenGL, TE1, and tiled pixel-memory work lands on October 6 (above).
  Solid, High, and Maximum configurations share the generic `GfxDisplay`
  output used by the CLI and GUI. See [MGRAS design](rules/mgras/DESIGN.md).
- **2026-09-30 — Graphics DMA and FIFO fixes** (`5787d02`, `b5cc363`): improved
  DMA reads, FIFO handling, and fillrate benchmark crash fixes.

### Host services

- **2026-09-30 — Host calls and HostGL** (`c8dc330`, `1a93808`, `0be4d1c`):
  optional `hostcall` traps private user-mode syscalls 3000–3009; `hostgl`
  registers service 3000 to replay the guest replacement libGL's commands on
  the host GPU. The backend is macOS CGL; other platforms register no GL
  service. Guest libraries live in `atomchild411/iris-guest-tools`.
  `iris-hostcall` is a workspace member; `iris-hostgl` is excluded and builds
  only when requested through the feature.

### Graphics (REX3)

- **Drawing engine refactor** (`35a3b18`). One generic draw routine
  (`src/dev/ng1/rex3_generic.rs`, mode decoding in `src/dev/ng1/rex3_shape.rs`) is specialised
  ahead of time into 462 native draw functions (`src/dev/ng1/rex3_shaders.rs`, generated
  by `tools/gen_rex3_shaders.py` from a corpus of the draw modes the IRIX desktop
  uses). Most desktop drawing now runs through LLVM-optimised specialised code in
  every build, not only with `rex-jit`. The REX3 JIT and the precompiled set share
  one dispatch table. `src/rex3_simd.rs` is gone; the JIT profile moved to
  `src/dev/ng1/rex3_profile.rs`.
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
  Landed 2026-09-18 (`44f4b24`).
- The GFIFO push is retryable, and a shader rejected by a full compile queue can
  be requested again (`rules/testing/rex-jit-queue-retry.md`).
- `CIDMATCH` is a mask of permitted CIDs, not an equality value
  (`rules/rex3/cidmatch-is-a-mask.md`).
- Blend-alpha handling fixed (`rules/rex3/blendalpha-and-alpha-blending.md`).
- REX3 benchmarking tests; a triangle benchmark in `gltest`.

### CPU, timing, and audio

- **2026-09-30 — Indigo2 IMPACT IP28 / R10000** (`8579614`): fullhouse machine
  variant with 16 MB MEMCFG granules, RAM at `0x20000000`, board revisions,
  R10000 cache-operation semantics, 64 TLB entries, and 44-bit virtual
  addresses. Loads/stores/fetches use memory directly; shadow tag/data arrays
  answer CACHE operations and PROM diagnostics. Default Count is 97.5 MHz.
  Requires an external IP28 PROM and IMPACT graphics for the IP28 kernel.
- **2026-09-29 — CPU/cache correctness** (`a09186a`, `b7d2a18`, `70a5e39`):
  MTC0 retains full values for 64-bit CP0 registers, XContext fields follow
  the CPU's VA width, and Index_Store_Tag discards old line data without
  writing it back. MEMCFG changes update only affected device-map slots
  (`772e01c`).
- **2026-09-29 — Compare deadline handling** (`da4223a`): a far-future Compare
  value is not treated as a missed deadline; writing Compare acknowledges
  its pending match without delivering it again (`18109d8`, 2026-09-19).
- **2026-09-18 — Idle-pause interrupt handling** (`f1a561d`, `1390bf6`): wake
  a parked CPU as soon as a device raises an interrupt, and do not park when
  all interrupt sources are masked.
- **2026-09-29 — HAL2 pacing and output** (`38e510c`): wall-clock frame pacing
  on 250 µs timer wakes, bounded catch-up, DMA read-ahead tied to guest polls,
  a persistent host output stream, and resampling to its rate. Fixes stretched
  audio, repeated stale chunks, and lock-related hangs. Codec CLKID uses BRES
  generator numbers 1–3 (`39d304f`, 2026-09-19).
- **2026-09-18 — VDMA implementation split** (`e603daf`): specialised transfer
  paths in `src/dev/mc_vdma.rs` preserve translation and GIO packing rules.

- **CP0 Count runs at a fixed 33 MHz** (`066935b`). The slow/fast tick detection
  and Count/IP7 frequency inference are gone; a constant rate proved more stable.
  IRIX reports it as a 66 MHz CPU. `[clock] fixed_mhz` / `--clock-fixed-mhz`
  override it.
- **2026-09-19 — Guest clock offset** (`eab2df2`, `[rtc_offset]`, iris-gui General → Real-time clock).
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

- **2026-09-21 — Whole-page compilation becomes the only implementation**
  (`8dfc365`): remove per-entry compilation; `j2wp` remains a compatibility
  alias for `jitv2`. Flushes no longer requeue preserved entry sets, and
  dispatch uses fast page lookup (`44e9b9e`, 2026-09-20; `da2bb7d`,
  2026-09-21). Shared absolute-PC exit blocks reduce code size (`e03b9b8`).
- **2026-09-29 — Broader compiled regions and persistent cache** (`36ec12b`):
  safe CP0 operations and LL/SC stay in regions through their interpreter
  handlers; add MIPS IV/FPU coverage and fix flush/reclamation failures.
  An optional disk cache reuses verified compiled pages between runs, keyed
  by build identity, page bytes, FR mode, and codegen settings. See
  [Persistent JIT cache](docs/jitv2-persistent-cache.md).
- **2026-09-30 — R10000 inline memory** (`839155a`, `afb8d1b`): direct loads
  and stores use the transparent memory window without cache-tag probes;
  the transparent-region gate tests the bitmap in one load.
- **2026-09-21 — FPU codegen** (`bea0af6`): CP1 usability checks in delay
  slots, conditional FPU moves, additional FP loads/stores, and literal zero
  IR operands.
- **2026-09-20 — Interrupt-check coalescing** (`3f51ff0`): `j2 intrun` can
  combine checks across eligible short instruction runs; default remains 1,
  and lockstep forces per-instruction checks.
- **2026-09-18 — Verifier and diagnostics** (`66d5151`, `156936a`): poison
  the verifier with its region-boundary sentinel and report JIT verification
  and policy flags in the build feature list.

- **2026-09-20 — Corpus capture moved to the pcp cache; `jitv2_corpus_dump` removed** (`e76520e`). The
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
  read. At introduction both compiler implementations could capture pages;
  the per-entry implementation was removed the following day. `zz_corpus_sizes` consumes a directory
  of `.pcp` files via `IRIS_CORPUS_DIR` instead of a file of filenames.

  Also fixed along the way: `Codegen::last_code_size` was `developer`-gated,
  which made the one number a codegen-size measurement needs available only in
  the build that invalidates such a measurement (`developer` forces
  `opt_level=none` and injects a per-instruction trace callout) — so
  `zz_corpus_sizes` reported `total_bytes=0` in exactly the build it was meant
  to measure. See `rules/jitv2/corpus-capture-from-the-pcp-cache.md`, including
  the `requested`-vs-`compiled` trap that silently discards 99.5% of a corpus.

- **2026-09-19 — Compile churn avoidance** (`d1bf817`). Each page now remembers the bytes its
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

### Networking and SCSI

- **2026-09-29 — NAT TCP backpressure** (`3e38408`): retain guest TCP data
  when a host reader is slow, instead of acknowledging bytes before the host
  has accepted them.
- **2026-09-19 — DNS response source** (`2b55ed6`): replies use the resolver
  address the guest queried, even when the host's upstream resolver differs.
- **2026-09-19 — Ethernet RX descriptors** (`b7bca7c`): stop the receive
  channel when its descriptor chain is exhausted so the guest can restart it.
- **2026-09-19 — SCSI write data phases** (`8434446`, `a68e7a3`): outbound
  PIO does not inherit DMA direction; data after a WRITE CDB is retained as
  data rather than decoded as a second command. Fixes Linux/NetBSD paths.

- **Guest DNS goes to the host's DNS server** (`src/net/host_dns.rs`): the first IPv4
  `nameserver` in `/etc/resolv.conf`, or the active adapter's server on Windows,
  re-read every few seconds so a VPN coming or going needs no restart. `8.8.8.8`
  is only the fallback.

### Testing and builds

- **2026-09-30 — Bare-metal suite coverage** (`2a796e2`, `5038af9`,
  `98a523e`): consecutive load/store and TLB-load cases, plus User-mode tests
  with UX enabled under KX/SX disabled, matching IRIX 6 n32 execution.
  Prebuilt benchmark provenance refreshed (`0f5c65c`; earlier `b6440aa`,
  2026-09-19).
- **2026-09-27 — Linux packaging** (`c5e3ad4`): riscv64 deb/rpm packages and
  pinned Ubuntu 24.04 runners; Anylinux/AppImage build fixes landed
  2026-09-25 (`b0e6f7b`).
- **2026-09-18 — CLI logging backend** (`edccb81`): install `env_logger` so
  warnings are visible; developer-mode CPU startup crash workaround
  (`45f26b8`).

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

- **2026-09-30 — UI consistency** (`e38f38a`): shared graphics board picker,
  memory controls, and processor clock controls across machine setup paths.
  Kernel tick rate appears next to MIPS; mouse injection is available from
  the monitor (`837b048`). The status bar also visualizes graphics FIFO depth
  (`a6f773e`, 2026-09-20).
- **2026-09-30 — First-frame resize locking** (`05242a6`): release the window
  size lock before calling `inner_size()` to avoid a recursive lock.
- **2026-09-19 — Machine lifetime cleanup** (`6ab3f11`): monitor/CI handles
  share the machine drop slot rather than retaining a dead `Machine` pointer.

- **2026-09-29 — Graphics board picker** in Configuration → General: Newport, GR2 XZ or GR2
  Extreme (Indigo2 only). Picking a GR2 board resets heads, resolution and
  `[impact]` to values `validate()` accepts; the Newport heads control is
  hidden for GR2.
- **2026-09-30 — IP28 / R10000 and IMPACT in the GUI.** The machine and CPU
  dropdowns include IP28 and R10000. New Machine selects R10000, disables the
  embedded PROM option, and presets IMPACT graphics for IP28. Initially gated
  by `ip28`; all profiles are built in after the October 1 feature cleanup.
- **2026-09-30 — Unified graphics selection.** The picker writes
  `[graphics] board` for Newport, GR2 XZ/Extreme, or IMPACT Solid/High/Maximum;
  there is no separate `[impact]` section. Non-Newport boards reset heads and
  resolution to supported values. Extreme is restricted to IP22; changing to
  an unsupported profile falls back to XZ.
- **2026-09-30 — CP0 Count clock (Processor section, General tab):** a `[clock] fixed_mhz`
  control (was CLI/TOML only) with an "Auto" reset to the profile's default
  (33 MHz, or 97.5 MHz on IP28).
- **2026-09-30 — Kernel Hz next to MIPS in the status footer** (`Machine::fasttick_count`,
  new in core): the guest's own clock-tick rate — CP0 Compare matches, or the
  IOC's 8254 timer interrupts when IRIX uses those instead — distinct from
  the MIPS readout's host emulation throughput. The CLI's baked status bar
  has shown this since `837b048`; iris-gui had no equivalent readout at all.
- **2026-09-30 — Host services / host OpenGL build status on the Debug tab.** New passthrough
  `hostcall`/`hostgl` crate features (`iris/hostcall`, `iris/hostgl` — see
  `c8dc330`, `1a93808`): private syscalls 3000-3009 let an IRIX program built
  against the replacement libGL (iris-guest-tools) ask the host for something
  the emulated machine doesn't have, with host OpenGL the first service. There
  is no per-machine config for either — a guest either gets the trap or
  doesn't — so the GUI's only job is reporting what's built in (and, for
  `hostgl`, that only macOS/CGL has a backend so far).
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
  `src/dev/testdev.rs`), later with a host clock and a retired-instruction counter.
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
  `src/net/tftp.rs`).
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
  a Newport XL on the GIO graphics slot, plus the then-existing Indy XZ/Elan
  and Indigo2 IMPACT preview stubs (removed/replaced in September), dual-head Newport and
  forced Newport resolutions.
- IndyCam CDMC register map and power-on defaults corrected
  (`rules/irix/indycam-cdmc-register-map.md`).
- NFS resolves `.` and `..` server-side.
- NAT: port forwards to rsh/rlogin use a reserved source port, probing for a free
  one (`rules/irix/rsh-forward-needs-reserved-source-port.md`).
- Recent HAL2 changes reverted after they broke audio.
- `libchdman-rs` 0.288.8 (BSD-3-Clause; drops GPL-3.0), then 0.288.9 (bin/cue fix).
- Machine and other large objects are constructed with bigger stacks (#60).

## June 2026

### iris-gui (new)

- The optional egui front-end introduced on May 31 continued to gain
  machine management, display, input, and distribution improvements.
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

- **In-process NFS server** (`src/net/nfsudp.rs`, eight increments ending `685f534`):
  NFSv2 for IRIX 5.3 and NFSv3 for 6.x, MOUNT v1/v3, a duplicate-request cache,
  IP-fragment reassembly, and READDIR that respects `count`. The external
  `unfsd` and its `--unfsd`, `--nfs-port` and `--mountd-port` options are gone.
- **PCAP bridged networking** (`--features pcap`, `2c83ac8`), with capture-
  permission elevation and installer plumbing, an NFS responder on a virtual LAN
  IP (`nfs_pcap_ip`), live NIC changes, and a fix for RX starvation on busy LANs.
- **XDMCP** reverse-proxy helper (`src/net/xdmcp.rs`, `docs/xdmcp.md`).
- FTP passive-mode helper for inbound port forwards; port forwards can be
  rebound live; NAT adoption and "networking off" diagnostics.

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

## May 2026 — snapshots, CHD and CI

### Late May: GUI and devices

- **2026-05-31 — Optional egui front-end** (`a8b8262`, contributed by Dani
  Sarfati): named machines with autosave in `gui.json`, iris.toml import/export,
  embedded framebuffer, input capture, safe-stop dialog, and icons.

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

### Storage and automation

- **CHD images** (`--features chd`, May 18–20) for SCSI disks and CD-ROMs, via the
  `libchdman-rs` crate.
- Configurable NAT subnet; unprivileged ICMP on macOS.
- Time and NTP answered by the gateway (late May).
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
  `src/cpu/mips_core.rs` measured `Instant::now()` between successive
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
  in `src/dev/z85c30.rs`.
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

- **Persistent JIT cache** (was Phase 2.5): descoped at the time; a different
  compiled-page cache for JIT v2 landed on 2026-09-29 (see September). Interp on M2 hits
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
