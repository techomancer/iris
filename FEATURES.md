# Build features and configuration reference

Complete feature inventory for **iris**, **iris-gui** and the crates they
build on. The tables list every declared build feature, shared machine-config
field, standalone CLI switch, and persisted GUI setting. Diagnostic environment controls follow
the user-facing settings. Detailed feature explanations and build examples
are below the inventory.

## Build feature inventory

### iris (core and standalone CLI)

Source: [Cargo.toml](Cargo.toml). Default features are `tlbvmap,rexdiag,lineedit,embedded-proms,http-media`.
Select with `cargo build --release --features <comma-separated-list>`; add
`--no-default-features` to drop the core default bundle.

| Feature | Purpose | Enables |
|---|---|---|
| `debug_cache` | Track a selected cache line through cache operations. | — |
| `developer` | Undo/trace buffers and developer monitor tools; CPU starts paused. | `gputrace` |
| `gputrace` | REX3 bus taps: annotated trace (`rex trace`) and replayable recording (`rex rec`); see [clanker/rex3/trace-and-recording.md](clanker/rex3/trace-and-recording.md). GR2, IMPACT, O2 and VPro annotated traces (`gr2 trace`, `mgras trace`, `re trace`, `vpro trace`); O2 engine recording (`re rec`). Off: no per-access check at all. Lightning-compatible. | — |
| `developer_ip7` | CP0 Compare/timer calibration statistics and debug prints. | — |
| `developerx` | Extended exception diagnostics; break on IBE/DBE/ADEL/ADES/TLB errors. | — |
| `llstats` | Per-address LL/SC reservation histogram (`ll stats`); lightning-compatible. | — |
| `perfetto` | Perfetto trace recording (`--perfetto FILE`, monitor `perfetto`); see [docs/perfetto.md](docs/perfetto.md). Lightning-compatible. | — |
| `rexdiag` | REX3 per-GO activity and dispatch counters. Default on; removable in a core build. | — |
| `fetchverify` | Compare executed instruction bytes with memory in interpreter/JIT; expensive stale-code detector. | — |
| `ci_clock` | Synthetic deterministic Count clock for CI/snapshot validation; replaces real-time Count timing. Also times the Indy's DS1386 and 8254 on the cycle count. | — |
| `opcodefusion` | Interpreter instruction fusion; breakpoints on a fused second instruction cannot fire. | — |
| `lightning` | Strip breakpoints/traceback for speed; hides GUI Debug controls. | `opcodefusion` |
| `tlbvmap` | O(1) TLB fast path (the vmap); without it lookups use the MRU scan. On by default. | — |
| `tlbstats` | TLB path/entries-walked counters, printed at exit. | — |
| `jitstats` | Measure eligibility for JIT inline memory accesses. | — |
| `tlbcheck` | Full JTLB consistency checks after TLB writes; diagnostic overhead. | — |
| `instr_stats` | Per-opcode decode/execute counters; interpreter only, refused with JIT v2. | — |
| `r5ksc` | Disabled external secondary-cache model; deliberately refuses to compile. | — |
| `r5ksc_triton` | Disabled O2/Triton integrated L2 model; deliberately refuses to compile. | `r5ksc` |
| `idle-pause` | Park the CPU in recognized guest idle loops; opt-in. | — |
| `tcache` | Keep cache tags/state but use authoritative ppmem RAM for cacheable data. | — |
| `vprodiag` | VPro texture packing check: each texture level is checked against the upload that wrote its memory, reported by `vpro status`. Off by default; small cost per upload. | — |
| `cache-ecc` | Cache parity/ECC injection and Cache Error exceptions for SGI IDE diagnostics; adds miss/exception checks. Off by default. | — |
| `decode-cache` | Compatibility flag; decoded-instruction caching for the shadow model is unconditional. | — |
| `tcache_verify` | Assert transparent-cache/RAM coherence on accesses; very slow. | `tcache` |
| `embedded-proms` | The SGI PROMs built into the binary, each inflated when its machine starts. Without it a machine needs a `prom` file. | — |
| `default` | Default bundle: `tlbvmap`, `rexdiag`, `lineedit`, `embedded-proms` and `http-media`. | `tlbvmap`, `rexdiag`, `lineedit`, `embedded-proms`, `http-media` |
| `rex-dirty-upload` | Track conservative Newport framebuffer damage and copy/upload only dirty regions. Opt-in; the default retains the original REX3 and presentation paths. | — |
| `rex-jit` | Cranelift compiler for Newport draw shaders; complements precompiled shaders. | `cranelift-codegen`, `cranelift-frontend`, `cranelift-jit`, `cranelift-module`, `cranelift-native` |
| `wgpu` | Host-GPU rasterisation through the common wgpu layer (VPro, IMPACT and experimental GPU-authoritative Newport), plus wgpu window composition. Newport production activation is initially validated on Vulkan and D3D12 and fails closed without CPU fallback. Work in progress; see `clanker/plans/GPUTodo.md`. | `dep:bytemuck`, `dep:pollster`, `dep:wgpu`, `dep:ash` |
| `gr4-jit` | Cranelift compiler for IMPACT (GR4) raster pipelines (RSS/TE1 shaders). | `cranelift-codegen`, `cranelift-frontend`, `cranelift-jit`, `cranelift-module`, `cranelift-native` |
| `gr2-jit` | Cranelift compiler for GR2 (XZ / Extreme) RE3 span pipelines (SHADED / FLAT shaders). | `cranelift-codegen`, `cranelift-frontend`, `cranelift-jit`, `cranelift-module`, `cranelift-native` |
| `gpujit` | Every graphics JIT; the recommended build. Hosts that forbid JIT code (the Mac App Store sandbox) set `IRIS_NO_JIT` at run time. | `rex-jit`, `gr4-jit`, `gr2-jit` |
| `jitv2` | Experimental whole-page MIPS compiler; implies transparent caching. | `tcache`, `cranelift-codegen`, `cranelift-frontend`, `cranelift-jit`, `cranelift-module`, `cranelift-native`, `target-lexicon`, `region`, `wasmtime-jit-icache-coherence` |
| `jitv2_opcodefusion` | Optional JIT fusion, off by default after live delay-slot failures. | `jitv2` |
| `j2wp` | Compatibility alias for `jitv2`; no separate compiler mode. | `jitv2` |
| `jitv2_lockstep` | Per-dispatch JIT/interpreter verifier for ALU/load-store/CP1; branch/jump verification incomplete. | `jitv2`, `developer` |
| `jitv2_smc_check` | Report writes to the executing code page; disable inline memory to observe all stores. | `jitv2` |
| `pcap` | Raw Ethernet bridge backend; needs host capture library/driver and capture permissions. | `dep:pcap` |
| `hostcall` | Private user-mode guest syscall services 3000–3009; IRIX handles other calls. | `dep:iris-hostcall` |
| `hostgl` | Guest replacement-libGL command replay; macOS (CGL) and Linux (EGL) backends. | `hostcall`, `dep:iris-hostgl` |
| `lineedit` | Terminal console line editing (history, Ctrl-R, cursor keys, Tab completion of command names) when stdin is a terminal. Default on; `--no-default-features` drops rustyline. | `dep:rustyline` |
| `http-media` | CD-ROM images given as `http://` or `https://` URLs, read with HTTP `Range` requests (see [STORAGE.md](STORAGE.md#cd-rom-images-over-http)). Default on; without it such a URL is refused when the disc is opened and the build has no HTTP or TLS code. | `dep:ureq` |

Cargo also exposes these **implicit optional-dependency features**:
`cranelift-codegen`, `cranelift-frontend`, `cranelift-jit`, `cranelift-module`, `cranelift-native`, `region`, `target-lexicon`, `wasmtime-jit-icache-coherence`. They enable library dependencies,
not complete emulator capabilities; use `jitv2` or `gpujit` for a working
compiler configuration. `dep:` entries above suppress implicit feature names
for the wgpu, PCAP, host-service, line-editing and HTTP dependencies.

The core always depends on the `iris-audio` crate with its default backends;
see [iris-audio](#iris-audio) below.

### iris-gui

Source: [iris-gui/Cargo.toml](iris-gui/Cargo.toml). There is no GUI-specific
default feature bundle. Its `iris` dependency always enables `gpujit` (the
App Store build turns the JITs off at run time with `IRIS_NO_JIT`) and retains the core default features. GUI `--no-default-features` alone does not
turn off those dependency defaults.

| GUI feature | Purpose | Enables |
|---|---|---|
| `bundled` | Distribution UI: hides source-checkout tools; machine TOML storage is shared by all builds. | — |
| `pcap` | Expose PCAP NIC selection and bridge support in the embedded core. | `iris/pcap` |
| `premiere` | Performance bundle for the embedded core; lightning plus idle parking. | `iris/lightning`, `iris/idle-pause` |
| `hostcall` | Enable private guest syscall services in the embedded core. | `iris/hostcall` |
| `hostgl` | Enable hostcall and host OpenGL on macOS (CGL) and Linux (EGL) in the embedded core. | `hostcall`, `iris/hostgl` |
| `perfetto` | Perfetto recording controls in the Debug tab. | `iris/perfetto` |
| `rex-dirty-upload` | Enable opt-in Newport damage tracking and partial uploads in the embedded core. | `iris/rex-dirty-upload` |
| `wgpu` | Enable experimental GPU-authoritative raster backends in the embedded core. iris-gui can host Newport only in explicit `newport_mode = "readback"`; direct presentation still requires the standalone frontend. | `iris/wgpu` |

Every core feature above is also selectable for the GUI via `iris/<feature>`:

```sh
cargo build -p iris-gui --release --features iris/jitv2
cargo build -p iris-gui --release --features premiere,hostgl
cargo build -p iris-gui --release --features wgpu
```

The GUI offers every machine profile and compatible graphics boards. **Advanced**
contains platform firmware and consoles, XIO/PCI expansion and storage, virtio
serial, rendering and diagnostic settings. Its **Complete configuration (TOML)**
editor accepts any supported key in the inventory below, validates before
applying, and requires the machine to be stopped. Optional build features still
need to be enabled when compiling; a saved setting does not compile them in.

### iris-gui-appkit (the Mac app)

`cargo build -p iris-gui-appkit`; `iris-gui/frontends/appkit/scripts/bundle.sh`
makes `IRIS.app`.

| Feature | Effect | Implies |
|---|---|---|
| `bundled` | Distribution UI, as for `iris-gui`. | `iris-gui-core/bundled` |
| `appstore` | Mac App Store build: the App Sandbox's security-scoped bookmarks, the container's data folder, CHD diffs in the container, the JITs off at startup (`IRIS_NO_JIT`), and Devices > Grant a disk folder. `bundle.sh --appstore` builds it with `lightning`. | `bundled`, `iris-gui-core/appstore` |
| `pcap` | Bridged networking, as for `iris-gui`. | `iris-gui-core/iris-pcap` |
| `lightning`, `jitv2`, `idle-pause` | The emulator's speed features. | `iris/...` |
| `hostgl` | Host OpenGL for IRIX programs. | `iris/hostgl`, `iris/hostcall` |

### iris-audio

Source: [iris-audio/Cargo.toml](iris-audio/Cargo.toml). The host audio output
every sound device plays through (HAL2, MACE audio, RAD1, ICE1724), selected at
run time with `[audio] host`. It is a workspace of its own, built through the
core's path dependency, and the core takes its default features.

| Feature | Purpose | Enables |
|---|---|---|
| `default` | Every backend; each is compiled only on its own platform. | `pulse`, `wasapi`, `coreaudio` |
| `pulse` | PulseAudio native-protocol backend, which also reaches PipeWire's pipewire-pulse. Pure Rust; Unix only. | `dep:pulseaudio` |
| `wasapi` | WASAPI backend, using only libraries that are part of Windows. | `dep:windows`, `dep:windows-core` |
| `coreaudio` | CoreAudio backend, using only macOS frameworks. | — |

### iris-prom

Source: [iris-prom/Cargo.toml](iris-prom/Cargo.toml). A workspace tool with no
features of its own: it downloads, packs and inspects the PROM images that
`embedded-proms` builds in (`data/proms`). See
[Companion binaries](#companion-binaries).

### Conflicts and unavailable combinations

[src/lib.rs](src/lib.rs) rejects `lightning + developer`,
`jitv2_lockstep + lightning/opcodefusion/jitv2_opcodefusion`, and
`instr_stats + jitv2`. `r5ksc` and `r5ksc_triton` always fail intentionally.
Consequently, `--all-features` is not a supported build command.
`jitv2_lockstep` implies `developer`, so it also inherits that conflict.
App Store builds use interpreter execution and disable the REX3 JIT; adding
`iris/jitv2` is not a supported sandbox build configuration. PCAP is a source-
build option and is absent from the standard release/App Store builds.

CHD, host camera, DaynaPort, Ultra64, IP28/R10000, ppmem, and MIPS IV code are
built in. The old `chd,camera,daynaport,ultra64,ip28,ppmem,mips4,r5k` Cargo
features are removed. Select devices/ISA at runtime instead.

### Build profiles, workspace targets, and version controls

The repository pins **nightly** in [rust-toolchain.toml](rust-toolchain.toml).
A plain `cargo build` selects the root `iris` package; `-p iris-gui` selects
the frontend, and `--workspace` also includes `iris-hostcall`, `iris-prom`
and the test-only `iris-checks`. `iris-hostgl` and `iris-audio` are excluded
from the workspace: `iris-hostgl` is built only through the optional `hostgl`
dependency, and `iris-audio` (a workspace of its own) through the core's path
dependency.
Select Cargo's target triple with `--target <triple>`; platform backends and
packaging still impose the host restrictions described above.

| Profile | Selection | Settings |
|---|---|---|
| `dev` | `cargo build` / `cargo run` | Cargo development defaults: unoptimized, with debug information. |
| `release` | `--release` | Optimization level 3, fat LTO, one codegen unit, abort on panic, debug level 1. |
| `developer` | `--profile developer` | Optimization level 3 with thin LTO and 16 codegen units, so rebuilds after an edit take one to three minutes instead of about nine; runs about 4% slower than `release`. Full debug information. Enable `--features developer` separately for instrumentation. |
| `profiling` | `--profile profiling` | Inherits release optimization; full debug information for profiling. |

[iris-gui/build.rs](iris-gui/build.rs) reads `RELEASE_VERSION` at build time
and exports `APP_VERSION` to the frontend. Without an override it uses
`CARGO_PKG_VERSION`, adding `-dev` for a debug-profile build. It also exports
`APP_COMMIT`, the checkout's short commit hash (`unknown` outside a Git
checkout), for the Help menu's version line. `PROFILE` and `CARGO_PKG_VERSION` are supplied
by Cargo. These build-time controls are
separate from the runtime environment inventory below.

### Offline and distribution builds

CHD support comes from the `libchdman-rs` crate, which iris builds with its
`prebuilt` feature: by default its build script downloads a static archive of
MAME's CHD code for the host target. Distribution build systems have no
network, so set `LIBCHDMAN_FORCE_SOURCE=1`. The build script then compiles the
MAME CHD sources shipped inside the crate (and therefore inside `cargo
vendor` output) with the host C/C++ toolchain, and nothing is downloaded. No
other dependency fetches anything at build time.

```sh
cargo vendor vendor >> .cargo/config.toml     # online, once: copies every crate
                                               # into vendor/ and points Cargo at it
LIBCHDMAN_FORCE_SOURCE=1 cargo build --release --offline --locked
LIBCHDMAN_FORCE_SOURCE=1 cargo build --release --offline --locked \
    -p iris-gui --features iris/lightning,bundled   # the GUI, as the release builds it
```

CHD support stays on; there is no build without it.

A prebuilt archive downloaded earlier is cached in `~/Library/Caches/libchdman-rs`
(macOS), `~/.cache/libchdman-rs` (Linux) or `$XDG_CACHE_HOME/libchdman-rs`, and
an offline build without `LIBCHDMAN_FORCE_SOURCE` uses it. Such a build passing
on a developer machine therefore says nothing about a clean build host.
[tools/offline-build-test.sh](tools/offline-build-test.sh) runs the steps above
with the network denied and with a fresh `CARGO_HOME`, cache and target
directory. Then the `chd-extract` it built unpacks a CHD made by MAME's
`chdman`. With `--expect-download-failure` it first shows that the same build
without `LIBCHDMAN_FORCE_SOURCE` fails.

| Variable | Effect |
|---|---|
| `LIBCHDMAN_FORCE_SOURCE` | Any value: compile the bundled MAME CHD sources instead of downloading a prebuilt archive. Needs a C++20 compiler. |
| `LIBCHDMAN_CPU` | `portable` compiles the FLAC codec as generic C. Without it an x86_64 source build passes `-mavx2 -mfma`, and the binary can fail with an illegal instruction on CPUs older than Haswell. Packages for a generic x86_64 baseline should set `portable`. |
| `LIBCHDMAN_PREBUILT_FALLBACK` | Any value: compile from source if the prebuilt download fails, instead of stopping the build. |

## Shared machine configuration inventory

Source: [src/config.rs](src/config.rs), `MachineConfig` and nested structs.
The CLI reads `iris.toml` (or `--config FILE`); iris-gui reads and saves the same
TOML schema in `<config dir>/iris/machines/<name>/<name>.toml`. `gui.json` holds
GUI preferences. The GUI uses the selected machine folder as the working
directory, stores config paths as relative paths, and defaults to `disks/` for
disk images and `nvram.bin`/`nveeprom.bin` for private battery-backed state.

Defaults below are **core schema/runtime defaults**, not this checkout's
sample iris.toml or the GUI New Machine dialog. GUI IP28 creation selects
R10000/Solid IMPACT and the embedded IP28 PROM; changing `machine.profile` in TOML
does not automatically replace every other hardware setting. Omit optional
keys to use defaults; TOML has no `null` value. Put top-level scalars before
the first `[section]`, because TOML tables continue until the next header.

### Top-level scalars

| Key | Default | Values and behavior |
|---|---|---|
| `prom` | `"prom.bin"` | PROM path, an override: every machine has its own PROM built in (the Indy and Indigo2 chip images; the IRIX 6.5.30 media images for the O2, Octane, Origin 200/2000/Onyx2 and Fuel/Tezro), which runs when this is left at the default. `(embedded)` selects it outright. A file named here must load. At the default, a machine uses `prom.bin`, or else the file name its releases shipped under, if either is in the working directory: `070-1367-012.bin` (IP22), `070-1477-002.bin` (IP28), `ip32prom.rev4.18.bin` (O2), `IP30prom.bin` (Octane), `ip27prom.img` (Origin 200/2000, Onyx2). |
| `nvram` | `"nvram.bin"` | DS1386 backing file on the Indy and Indigo2, its clock in `<nvram>.clock`; the PROM's environment on the O2 (its flash); the platform RTC/NVRAM store on the Octane, SN0 and IP35 machines. GUI paths are relative to the selected machine folder. |
| `nveeprom` | `"nveeprom.bin"` | Indigo2 motherboard EEPROM file. GUI paths are relative to the selected machine folder. |
| `banks` | `[128,128,0,0]` | RAM bank sizes in MB, up to eight entries (two banks on the Fuel, eight on the Tezro and the IP35 brick machines, four elsewhere; entries beyond the profile's bank count must be 0). On the O2 each entry is a DIMM pair's kit (32, 64, 128 or 256), largest first. On the Octane each entry is a pair of DIMMs, 64, 128, 256, 512, 1024 or 2048, filled from bank 0, 8192 in all. See profile constraints below. |
| `scale` | `1` | Standalone window scale: 1–4. GUI uses its separate vm_scale. |
| `headless` | `false` | Suppress the standalone window. Newport is not fitted; O2 uses a serial console. Other graphics boards can remain fitted. Audio is independent. |
| `no_audio` | `false` | Disable host audio output. The Indy and Indigo2 then have no HAL2; the O2's MACE audio, the Octane's and Onyx2's RAD1 and the Tezro's ICE1724 stay fitted and run silently. |
| `keyboard` | `ansi` | Physical shape of the host keyboard: `ansi` (US, `\|` above Enter), `iso` (European) or `jis` (Japanese 106/109; use it with the PROM's `keybd` set to Japanese). It decides which SGI key the host's ambiguous key left of Enter sends; the guest applies the character layout. `--keyboard` overrides. |
| `gdb_port` | `unset` | Start the GDB TCP stub; debugging is unavailable in lightning builds. |
| `monitor_port` | `8888 when unset` | Monitor listener on loopback. |
| `serial_port_a` | `8880 when unset` | SCC channel A listener (tty2); the O2's first serial port, its console when headless. |
| `serial_port_b` | `8881 when unset` | SCC channel B listener (tty1/console); the O2's second serial port. GUI halt uses this port. |
| `serial_socket_a` | `unset` | Also serve serial channel A on this Unix socket (raw bytes, one client at a time). |
| `serial_socket_b` | `unset` | Likewise for channel B. With `ci` the console stays readable through `serial-read` and `wait-serial` as well. |
| `load_elf` | `unset` | Load a static big-endian MIPS ELF32 binary before starting the CPU. |
| `cheritest_dump_hook` | `false` | CP0 register 26 dump convention; needs a `test_device` device, unsafe for normal PROM boot. |
| `nat_subnet` | `192.168.0.0/24 when unset` | CIDR; NAT gateway network+1 and guest network+2. |
| `ci` | `false` | CI server/serial backend; hides host window unless ci_display, keeps offscreen graphics. |
| `ci_socket` | `iris.sock` in `$TMPDIR` (else `/tmp`) on Unix; `127.0.0.1:19851` on Windows | Unix path, host:port, or tcp:host:port. |
| `ci_display` | `false` | Keep host graphics window visible in CI mode. |
| `mouse_scroll_pixels_per_line` | `40.0` | Host scrolling distance per guest PS/2 detent. |
| `lock_aspect_ratio` | `true` | Constrain standalone window aspect ratio; false allows letterboxing. |
| `renderer` | `gl` | What draws the window: `gl`, or `wgpu` (needs a build with the `wgpu` feature). With `wgpu` the window is presented through wgpu on the same GPU as a VPro or IMPACT whose raster unit is on the GPU, and its frames are shown without being read back. The iris-gui front end ignores it. |
| `vsync` | `false` | Wait for the host monitor's vertical sync when a frame is shown. Off, frames are shown at once and the guest's video timing is independent of the host monitor; on an X11 desktop with no compositor this can tear. On, a present can wait up to one host frame, which also delays the guest's vertical retrace. The iris-gui front end ignores it. |
| `serial_log` | `unset` | Log what the guest sends on its serial console (channel B on the Indy and Indigo2, A on the Octane and O2). An existing file is added to, or started over with `serial_log_overwrite`. Used by CI and GUI serial capture. `-` writes to standard output (for a container's log collector). |
| `serial_log_overwrite` | `false` | Start `serial_log`, `serial_log_a` and `serial_log_b` over each run instead of adding to them, so a script tailing the file sees only this run. `--serial-log-overwrite` sets it. |
| `serial_log_a`, `serial_log_b` | `unset` | Log what the guest sends on channel A or B when that channel is not the console (A on the Indy and Indigo2, B on the Octane and O2; `serial_log` is the console's), appended to like `serial_log`. |
| `serial_log_timestamps` | `off` | `off` (raw transcript), `wall` (prefix each completed line with a UTC ISO-8601 stamp) or `ecs` (one Elastic Common Schema JSON document per non-empty line, logger `irix.guest`, control characters dropped). A partial line is held until its newline and written on stop. `--serial-log-timestamps` overrides. |
| `scsi_deferred_int` | `true` | Defer SCSI status interrupts for BSD compatibility. |

### [machine]

| Key | Default | Values and behavior |
|---|---|---|
| `machine.profile` | `indy_ip24` | indy_ip24, indigo2_ip22, indigo2_ip28, o2_ip32 (configured by `[o2]`, see [the O2 guide](docs/o2-ip32.md)), octane_ip30 (configured by `[octane]`, see [the Octane guide](docs/octane-ip30.md)), origin200_ip29 (configured by `[origin200]`, see [the Origin 200 guide](docs/origin200-ip29.md)), origin2000_ip27 (configured by `[origin2000]`, see [the Origin 2000 guide](docs/origin2000-ip27.md)), onyx2_ip27 (configured by `[onyx2]`, see [the Onyx2 guide](docs/onyx2-ip27.md)), fuel_ip34 (configured by `[fuel]`, see [the Fuel guide](docs/fuel-ip34.md)), tezro_ip53 (configured by `[tezro]`, see [the Tezro guide](docs/tezro-ip53.md)), origin3200_ip35 (configured by `[origin3200]`, see [the Origin 3200 guide](docs/origin3200-ip35.md)), onyx3000_ip35 (the Origin 3200 with a G-brick or a V-brick; `[graphics] board` picks it, see [the Onyx 3000 guide](docs/onyx3000-ip35.md)). |
| `machine.cpu` | the machine's first | A family (`r4400`, `r5000`, `r10000`, `r12000`, `r14000`, `r16000`), meaning the machine's part of that family, or a part (`r14000_600`); independent of build features. With `test_device` set (the bare-metal benchmark rig) any part is accepted on any machine, `r14000_500` and the R16000 parts among them. Indy and `indigo2_ip22`: r4400, r5000; `indigo2_ip28`: r10000; O2: r5000, r10000, r12000; Octane: r10000, r12000, r14000; Origin 200, Origin 2000 and Onyx2: r10000 (`r10000_180`), r12000 (`r12000_270`); Fuel: r14000 (`r14000_500`); Tezro: r16000 (a bare `r16000` is `r16000_700`; `r16000_800`, `r16000_900` and `r16000_1000` are the other parts it ships with); Origin 3200: r12000 (`r12000_400`, with 8 MiB of secondary cache). The first listed is the default. |
| `machine.cache` | `auto` | How the CPU caches are modelled: `full` (tags, line data, LRU and write-back, as on a real R4400 or R5000; transparent in a `tcache` or `jitv2` build, which keeps tags and states but writes through to memory), `shadow` (memory is the only store; tag arrays only answer CACHE instructions, the faster model under jitv2) or `auto` (`shadow` in a `jitv2` build, `full` otherwise). The R10000 family always uses `shadow`. `--cache` overrides. |
| `machine.cpus` | `1`; `4` on the Tezro | Processors fitted on each node board, each run on its own host thread. 2 on the Octane; see [its guide](docs/octane-ip30-multiprocessor.md). 1, 2 or 4 on the Tezro; one selects the one-processor board with one PIC, two or four the board with two PICs. 1, 2 or 4 on each C-brick of the Origin 3200 and Onyx 3000, so two C-bricks run up to eight. 1 or 2 on each node board of the Origin 200, Origin 2000 and Onyx2, so a two-board deskside runs four. |
| `machine.emu_id` | `0` | 0 to 4095; only the Fuel and Tezro take it, and other machines refuse a nonzero one. Mints the machine's system id (`sysinfo -s`, `lmhostid`) in iris's layout. When nonzero it gives the station address `08:00:69:e2:<emu_id>`, which is the Fuel's id. The Tezro's L1 system serial is `LEMU2<emu_id>0`, whose fold is the Tezro's id. 0 is the generic id. See [the Fuel guide](docs/fuel-ip34.md) and [the Tezro guide](docs/tezro-ip53.md). |

The keys in the sections below, down to `[onyx3000]`, and those of
[`[o2]`](#o2) and [`[octane]`](#octane), are `[machine]` keys too: what one
machine has beyond the shared keys. Write them in `[machine]`; a key the
profile does not have is refused. Two are named differently there: `nodes`
(also accepted as `c_bricks`) is the node boards of an Origin 2000 or Onyx2 and
the C-bricks of an Origin 3200 or Onyx 3000. The tables named for a machine are the older spelling. They are
still read, a `[machine]` key wins over the same key there, and a config saved
by iris-gui writes `[machine]`.

### [origin200], [origin2000] and [onyx2]

The SN0 machines read one table each, named for the product and holding the
same keys: `[origin200]` when `machine.profile = "origin200_ip29"`,
`[origin2000]` for `"origin2000_ip27"` and `[onyx2]` for `"onyx2_ip27"`. A
table that is not the profile's own is refused, with the name of the one the
profile reads.

| Key | Default | Values and behavior |
|---|---|---|
| `nodes` | `1` | Node boards on a deskside: 1 or 2. Two are joined by the direct cable. Every node gets the `banks` memory. The Origin 200 has one. |
| `flash` | unset | Persistent 1 MiB node boot-flash file (of the first node on a deskside; the second keeps its own at the name with `.1` added). Unset, the flash is volatile. |
| `controller_store` | unset | Persistent file for the module controller's configuration store. Unset, it is volatile. |
| `io_prom` | unset | Deskside only. The BaseIO card's I/O PROM image (`io6prom.img`, beside the node PROM on the IRIX 6.5.30 media), loaded into its boot flash. Unset, or `(embedded)`, loads the PROM built into iris. |

A table is written to a saved config only when one of its keys is set. See the
[Origin 200](docs/origin200-ip29.md), [Origin 2000](docs/origin2000-ip27.md) and
[Onyx2](docs/onyx2-ip27.md) guides.

### [fuel]

Fuel settings apply when `machine.profile = "fuel_ip34"`.

| Key | Default | Values and behavior |
|---|---|---|
| `flash` | unset | Persistent 16 MiB boot-flash file. Unset, the flash is volatile. |
| `console` | `l1` | `l1`, `ioc3_a` or `ioc3_b`. Selects the guest console connection; the console serves CI and `serial_log`. |
| `l1_port` | `8882` | Host TCP port for the L1 console. The IOC3's ports use `serial_port_a` and `serial_port_b`. |
| `l1_log` | unset | Optional serial log path for L1 output when an IOC3 port is selected as console. |
| `ps2_input` | `true` | Fit the IOC3's PS/2 keyboard and mouse and feed them host input. |
| `usb_input` | `false` | Plug a USB keyboard and mouse into the OHCI controller's first two root ports and send host input to them instead of the PS/2 pair, which stays fitted unless `ps2_input = false`. |
| `l1_revision` | `1.48.1` | L1 firmware revision as `major.minor.patch`, decimal. The guest reads it with L1 request `0x0005`. |
| `brick_serial` | `EMUN00` | L1 brick serial, at most 37 bytes. The guest reads it with L1 request `0x0004`. The default is a made-up serial, not a real one. |

The machine has one 500 MHz R14000 and 512 MiB of RAM in two 256 MiB
banks (`banks` must be `[256, 256]`), and runs on a serial console unless
`[graphics] board` fits a VPro. A `[fuel]` table is written to a
saved config only when one of these keys is set. See the
[Fuel guide](docs/fuel-ip34.md).

### [tezro]

Tezro-specific settings apply when `machine.profile = "tezro_ip53"`.

| Key | Default | Values and behavior |
|---|---|---|
| `flash` | unset | Persistent 16 MiB boot-flash file. Unset, the flash is volatile. |
| `console` | `l1` | `l1`, `ioc4_a`, `ioc4_b`, `ioc4_c` or `ioc4_d`. Selects the host connection only. The PROM uses the L1 until `console ioc4` is run in its Command Monitor and the machine is reset; it then uses UART A (`ioc4_a`) only, and keeps the choice in the `flash` file. |
| `l1_port` | `8882` | Host TCP port for the L1 console. |
| `l1_log` | unset | Optional serial log path for L1 output when an IOC4 port is selected as console. |
| `serial_port_c`, `serial_port_d` | `8883`, `8884` | Host TCP ports of IOC4 ports 2 and 3; ports 0 and 1 use `serial_port_a` and `serial_port_b`. |
| `audio` | `false` | Add the ICE1724 audio controller on the IO9's PCI bridge. Off, the bridge's bus is empty. On, it plays to the host as `[audio]` sets, or silently with `no_audio`. See the [Tezro guide](docs/tezro-ip53.md). |

The machine runs the 700 MHz R16000 (`r16000_700`, the default), or the 800, 900
or 1000 MHz one (`r16000_800`, `r16000_900`, `r16000_1000`, with 4, 8 and 16 MiB
secondary caches). The 700 MHz module accepts 1, 2 or 4 processors, the 800
and 900 MHz modules accept 4, and the 1 GHz module accepts 2 or 4. `banks`
takes 512 or 1024 MiB banks from bank 0; left unset, it is the 4 GiB layout of
1024, 1024 and four 512 MiB banks. A `[tezro]` table is written to a saved
config only when one of these keys is set. See the
[Tezro guide](docs/tezro-ip53.md), which lists its limits.

### [origin3200]

Origin 3200 settings apply when `machine.profile = "origin3200_ip35"`.

| Key | Default | Values and behavior |
|---|---|---|
| `c_bricks` | `1`; `2` on an Onyx 3000 with a V12 | 1 or 2 C-bricks, each with `machine.cpus` processors. Two are joined by the direct NUMAlink cable. |
| `flash` | unset | Persistent 16 MiB boot-flash file of the first C-brick; the second keeps its own at the name with `.1` added. Unset, the flash is volatile. |
| `console` | `l1` | `l1`, `ioc3_a` or `ioc3_b`. Selects the guest console connection. |
| `l1_port` | `8882` | Host TCP port for the first C-brick's L1 console. |
| `l1_log` | unset | Optional serial log path for L1 output when an IOC3 port is selected as console. |
| `l1_revision`, `brick_serial` | `1.48.1`, `EMUN00` | What the first C-brick's L1 reports, as for the Fuel. |

`banks` is per C-brick and defaults to two 256 MiB banks; 256 and 512 MiB banks
are accepted, in the matched pairs a lane of the memory controller holds. The I-brick's Fibre Channel
adapter is always fitted; its disks are `hdd` devices at `xio15/pci0/1/lN`. A `[origin3200]` table is
written to a saved config only when one of these keys is set. See the
[Origin 3200 guide](docs/origin3200-ip35.md), which lists its limits.

### [onyx3000]

Onyx 3000 settings apply when `machine.profile = "onyx3000_ip35"`. The keys are
those of `[origin3200]`, with the same defaults: `c_bricks` (`[machine] nodes`), `flash`, `console`,
`l1_port`, `l1_log`, `l1_revision` and `brick_serial`. Either graphics board
takes one or two C-bricks; `c_bricks` left unset is 2 with a V12, whose V-brick
then has a C-brick of its own. A `[onyx3000]` table is written to a saved config only when
one of them is set. See [the Onyx 3000 guide](docs/onyx3000-ip35.md).

### [graphics]

| Key | Default | Values and behavior |
|---|---|---|
| `graphics.board` | `newport`; `solidimpact` on the Octane | newport, xz, extreme, solidimpact, highimpact, maximpact, v6, v8, v10, v12, ir2, ir2:2rm, ir2e, ir2e:2rm, ir3, none. `ir2` and `ir2:2rm` are the Onyx2 deskside's InfiniteReality2 pipe with one or two raster managers; `ir2e` and `ir2e:2rm` are the InfiniteReality2E (GE16-4, RM9, 64 MB texture per raster manager), which also fits the deskside. The Onyx2 takes only the first four or `none`; the default is `none`. Extreme is Indigo2 only. The Octane takes only `solidimpact` (SI), `highimpact` (SI with texture option), `maximpact` (MXI), `v6`, `v8`, `v10`, `v12` (the VPro boards, also `vpro:v12` and so on; need an IRIX install with the VPro driver, IRIX 6.5.10 or later and PROM 4.0, or 4.5 for the V10 and V12, which shipped only in the Octane2 and so give the machine the Octane2's Crossbow 1.4 and Bridge revision) or `none` (no board: the machine runs on its serial console, with no window). The Fuel takes `v12` or `v10` in its graphics slot, or `none`, its default. The Tezro takes `v12` at widget 12, or `none`, its default. The Onyx 3000 (`onyx3000_ip35`) takes `ir3`, its default, an InfiniteReality3 pipe on a G-brick at widget 11 of the I-brick, or `v12`, on a V-brick with a second C-brick (widget 8, an approximation) or with one C-brick at widget 11 of the I-brick; it has no `none`. |
| `graphics.flash` | unset | InfiniteReality2: the file that keeps the pipe's 512 KiB flash EEPROM between runs; created when absent. Unset keeps it in memory. |
| `graphics.resolution` | `guest` | guest, 1024x768, 1280x960, 1280x1024; presets require Newport. |
| `graphics.threads` | `2` | Worker threads the board's fragment pipeline draws on, at least 1. VPro and InfiniteReality: the screen tiles' owners, besides its front-end and raster threads; more help on a host with cores to spare. |
| `graphics.tile` | `32` | VPro and InfiniteReality: the fragment pipeline's screen tile size in pixels, a power of two from 8 to 256. Each tile's work runs in order on one worker at a time; with many threads, 64 can be faster. |
| `graphics.raster` | `cpu` | VPro, IMPACT and Newport: where the raster unit runs. `cpu` keeps the existing board path; `wgpu` draws on the host GPU through wgpu compute (needs a build with the `wgpu` feature). Newport's experimental production mode makes its complete RGB and auxiliary planes GPU-authoritative and has no unsupported-operation CPU fallback. An IMPACT board then has one raster engine, whatever `IRIS_MGRAS_ENGINES` says. |
| `graphics.newport_mode` | `direct` when Newport uses wgpu | `direct` shares the resident GPU with the wgpu window presenter and fails startup rather than falling back to GL/CPU; `readback` explicitly selects the slow compatibility/validation compositor, synchronizing both complete planes for each newly composed frame. It applies only to Newport with `raster = "wgpu"`. |
| `graphics.gpu_backend` | `auto` | The wgpu backend for `raster = "wgpu"`: auto, vulkan, metal, dx12 or gl. `WGPU_BACKEND` overrides it. Experimental production Newport initially accepts only Vulkan or D3D12; `auto` must resolve to one of them. |

<a id="devices"></a>

### [[devices]]

Disks, CD-ROMs and the cards that go in a machine are `[[devices]]` entries, the same on every machine.
Each has a `type`, an optional `at` (where it goes) and optional `name` (a local alias for that slot, so
`at = "cage/pci0"` starts from the device named `cage`), then the type's own keys. A name uses lowercase
letters, digits, `_` and `-`, and must not look like a slot (`pci2`, `xio15`, `t1`) or shadow a machine alias.

`at` is a slot path: tagged hops separated by `/`, such as `xio15/pci0/0/t1` (target 1 of the Octane's first
SCSI controller), `pci0/3` (the O2's PCI option slot) or `gio1`. Each machine also gives some slots
aliases, such as `scsi0/t1`, `scsi1/t4`, `pci_slot`, `io2` or `bus1-slot2`. Left out, or naming a bus
(`xio13/pci0`), `at` picks the first free slot that fits: devices with a full slot claim first, then
those that name a bus, then the rest, each in file order. A SCSI disk with no slot takes the first free
target. `iris --list-slots` prints every slot of a machine with its aliases, what is in it and what it
takes; `--device [NAME:]TYPE[@SLOT][,KEY=VALUE]...` adds or replaces one from the command line.

| `type` | Goes in | Keys |
|---|---|---|
| `hdd` | a SCSI target, or a Fibre Channel loop ID | `path` (raw or CHD), `overlay` (copy-on-write sidecar `PATH.overlay`, or a CHD's `.diff.chd`; one emulator at a time can use it, and a `--ci` run keeps its own in the temporary directory; see [STORAGE.md](STORAGE.md)), `write_cache` (default `true`: a write completes once the host has it and a sync follows within about 5 s; `false` stores every write first), `scratch` (a host-managed raw volume; no overlay) and `size_mb` (64 when unset, for a new scratch image) |
| `cdrom` | a SCSI target | `path` (a file, or an `http://` or `https://` URL, [STORAGE.md](STORAGE.md#cd-rom-images-over-http); empty means an empty tray), `discs` (more changer images), `overlay` |
| `daynaport` | a SCSI target | `mac` (`00:80:19:44:50:<target>` when unset), `subnet` (`192.168.10.0/24` when unset, separate from ec0's). Refused on the machines with built-in Ethernet only (the O2, the Origin 200, Origin 2000 and Onyx2, the Fuel and the Tezro) and with `network.mode = "stream"`. |
| `nvme` | a PCI slot | `path` (raw or CHD, at most 2 TiB), `overlay`, `write_cache` (as for `hdd`). An NVM Express controller with one namespace; any number per machine; see [docs/nvme.md](docs/nvme.md). |
| `qla2200` | a PCI slot | A QLogic ISP2200 Fibre Channel adapter; its disks are `hdd` entries at `CARD/lN`, N the loop ID 0 to 125. The Origin 3200 and Onyx 3000 have one fitted at `xio15/pci0/1`. See [docs/fibre-channel.md](docs/fibre-channel.md). |
| `ice1724` | a PCI slot | The IC Ensemble ICE1724 audio card; plays as `[audio]` sets. |
| `pci_cage`, `pci_shoehorn` | an Octane XIO slot (`xio9` to `xio13`; the cage at `xio13`, alias `pci_module`) | The PCI module (slots 1 to 3) and the single-slot XIO-to-PCI adapter; see [docs/pci.md](docs/pci.md#option-boards-in-xio-slots). |
| `pci_bridge` | a PCI slot | A PCI-to-PCI bridge, with one slot behind it at `BRIDGE/pci0/0`. |
| `virtio_serial` | a spare slot (a GIO slot on the Indy and Indigo2) | `port` tables, `[[devices.port]]`, at most 31: `name`, `console`, and `tcp_port` (raw TCP on 127.0.0.1) or `socket` (a Unix socket, exclusive with `tcp_port`); one console on TCP 8890 when none. See [docs/dev/virtio-serial.md](docs/dev/virtio-serial.md). |
| `virtio_9p` | a spare slot | `path` (required), `tag` (`"share"`), `readonly`, `owners` (`"squash"` or `"passthrough"`), `uid`, `gid`, `uid_map`, `gid_map`. Needs a Unix host; see [docs/dev/virtio-9p.md](docs/dev/virtio-9p.md). |
| `ultra64`, `test_device`, `newport` | `gio0` | The N64 development board (needs an external gopher64 peer), the bare-metal test device (`dump`, `run_config` as `time_pct=N,repeats=N,groups=MASK`; with `cheritest_dump_hook`) and the Indy's or Indigo2's second Newport head. |
| `hostserial` | a serial port, `serialN` (IRIX's `/dev/ttydN`: `serial1` and `serial2`, on the Tezro `serial3` and `serial4`); `at` is required | `path` (required): a host serial port such as `"/dev/ttyUSB0"`. Its rate, framing, DTR, RTS and break follow the guest; its CTS, DCD, DSR and RI are the guest's modem inputs. Linux and macOS. See the serial ports section of [HELP.md](HELP.md). |
| `midi_synth` | a serial port, as `hostserial` | `path` (required): a General MIDI SoundFont (`.sf2`), loaded at start. `gain` (`1.0`, 0 to 16): times the synth's own level, which leaves room for many voices; the host mix clips above full scale. Plays into the host's audio as a stream of its own (muted by `no_audio`); with `[audio] file`, into a file beside it (`out.wav` → `out-midi.wav`); `IRIS_MIDI_CAPTURE=file.wav` also records it. |
| `midi_raw` | a serial port, as `hostserial` | `path` (required): a host MIDI port. A raw MIDI device (Linux: `"/dev/snd/midiC1D0"` or `"/dev/midi1"`) passes the bytes as they are, both ways; a host serial port with a MIDI interface (`"/dev/ttyUSB0"`) is set up as `hostserial` does. |
| `graphics` | a graphics port or slot (`gfx`, `xio11`, `node1/xio8`) | Where the board goes; which board it is stays `[graphics] board`. Left out, the machine places it. |

virtio devices take only the spare slots a board has for them, never a real slot (the Octane's Bridge
devices 7, 5 and 6, the Tezro's IO9 bus 1); `at` can name any slot. A snapshot records each device at its slot: see [Snapshots](#snapshots).

#### Old keys

The keys iris wrote before `[[devices]]` still load, each with a warning saying what it became, and will be
removed in a later release: `[scsi.N]` tables (with `controller`, `kind` and `cdrom`), `[ultra64]`,
`[virtio_serial]` and `[virtio_9p]` (with `slot`), `graphics.heads = 2`, `test_device` and its two companions,
and the command-line `--scsi1`/`--cdrom4` family and `--test-device`.

`[nvme]`, `[[xio_board]]`, `[fibre_channel]`, `[machine] audio_card`, the `audio` flags, `[graphics] node` and
`widget`, and `bridge_slot` no longer load. A file with one is refused, and the error names the `[[devices]]`
entry to write instead.

### Snapshots

A snapshot names every disk, overlay and card by its slot, so any number of NVMe, virtio and audio
cards work. Saving a machine writes each device's slot into the manifest; a restore refuses a device that
sits at another slot in this machine and says which `at` keeps it, and one added to the config afterwards
is refused too, since the saved bus has no slot for it. A virtio-9p share cannot be saved; the snapshot is refused. Snapshots from before slots
(schema 10 and older) are refused.

### [network]

| Key | Default | Values and behavior |
|---|---|---|
| `network.mode` | `nat` | nat, pcap, stream or off (no Ethernet; every machine but the Indy and Indigo2). Stream needs `network.stream`. PCAP needs build support and host capture permissions. PCAP works on every Ethernet; stream on every onboard one, not on a DaynaPort. |
| `network.stream` | `unset` | `tcp:IP:PORT` or `unix:PATH`: the server of the private segment for `mode = "stream"` (QEMU `-netdev stream` framing). See NETWORKING.md. |
| `network.pcap_interface` | `auto-pick when unset` | Interface name or numeric list index; PCAP only. |
| `network.nfs_pcap_ip` | `unset` | Virtual LAN IP for the built-in NFS responder in PCAP mode. |
| `network.mac` | `08:00:69:12:34:56 when unset` | ec0 MAC; injected only into blank PROM battery-backed MAC slots. |
| `network.tftp_dir` | `unset` | Read-only TFTP root at the NAT gateway for PROM network boot. |
| `network.rsh_dir` | `unset` | Root for the built-in rsh server at the NAT gateway (TCP 514): `rsh <gateway> cat\|dd\|ls\|echo\|pwd`, read-only, nothing runs on the host. |
| `network.rx_hold` | `true` | Indy and Indigo2: frames for the guest wait in iris, off the virtual wire, while the next one would fill the last descriptor of its receive ring before the driver has reaped the filled ones, and for at most 100 ms. A frame that reaches a stopped receive channel is lost either way, as on the wire. `false` never waits. |

### [nfs]

| Key | Default | Values and behavior |
|---|---|---|
| `nfs.shared_dir` | `required when [nfs] exists` | Host directory exported by in-process NFS; no external unfsd. |
| `nfs.version` | `auto` | auto, v2, v3 (serialized lowercase enum names). |

Omit `[nfs]` to disable the share; the default forward list is empty.

### [[port_forward]]

| Key | Default | Values and behavior |
|---|---|---|
| `port_forward[].proto` | `required` | tcp or udp. |
| `port_forward[].host_port` | `required` | Host listener port. |
| `port_forward[].guest_port` | `required` | Guest destination port. |
| `port_forward[].bind` | `localhost` | localhost or any; any listens on all interfaces. |

### [vino]

| Key | Default | Values and behavior |
|---|---|---|
| `vino.source` | `off` | off, test_pattern, camera, black. |
| `vino.standard` | `ntsc` | ntsc (60 fields/s) or pal (50 fields/s). |
| `vino.camera_index` | `0` | Host camera index; meaningful only for source=camera. |

### [vnc]

| Key | Default | Values and behavior |
|---|---|---|
| `vnc.listen` | `unset` (no server) | `host:port`, or a port on 127.0.0.1, to serve the screen, keyboard and mouse over VNC (RFB 3.8; Raw and ZRLE; VNC Authentication when the standalone `iris` has a password, see `password_file`; without one, keep it on loopback). Every board, with a window or `ci`; not `headless`. Standalone `iris` and iris-gui. See [HELP.md](HELP.md#vnc) and [iris-gui-README.md](iris-gui-README.md#vnc). |
| `vnc.max_fps` | `30` | Most updates a second sent to a viewer (1–120). |
| `vnc.password_file` | `unset` | File holding the VNC password (RFB VNC Authentication: first 8 bytes; a trailing newline is dropped). `--vnc-password-file` sets it. Unset: `IRIS_VNC_PASSWORD_FILE`, then `IRIS_VNC_PASSWORD` (the password itself) from the environment, then no password. Standalone `iris` only; see [HELP.md](HELP.md#vnc). |

### [audio]

| Key | Default | Values and behavior |
|---|---|---|
| `audio.host` | `"default"` | Host audio system: `"default"` (CoreAudio on macOS, WASAPI on Windows, a PulseAudio server or PipeWire's pipewire-pulse on Linux and the other Unixes), `"pulse"`, `"wasapi"`, `"coreaudio"`, or `"null"` (the devices run and their sound is discarded). `"alsa"` and `"pulseaudio"` from older configurations mean `"default"` and `"pulse"`. The stream follows a change of the host's default output device, and keeps running with no device or server until one appears. `--audio-host` overrides. |
| `audio.prebuf_ms` | `20` | How much sound a stream holds before it plays and keeps queued afterwards: the margin against the emulated device running late, and the latency iris adds. Raise it if sound breaks up on a busy host. |
| `audio.latency_ms` | `unset` | How much the host's device should buffer. Unset: the host system's low-latency default. |
| `audio.channels` | `2` | Speakers the Tezro's multichannel card plays to: 2, 4 (quad), 6 (5.1) or 8 (7.1). The card folds its channels down to this many; any other value plays stereo. Other sound devices are stereo. |
| `audio.file` | `unset` | Play into this WAV file instead of a sound card: 48 kHz 16-bit, written in real time as the host stream would play it, silence included. For headless runs and for checking what a guest played. |

### [jitv2]

| Key | Default | Values and behavior |
|---|---|---|
| `jitv2.threads` | one per CPU, at most 4, capped by the host's CPUs | Fixed compile-pool size at startup; must be at least 1 (0 is rejected). Unset, the pool has one thread per guest CPU, up to 4 — capped by the host's usable CPUs, one of which is held back for the emulator's device threads. The guest's own CPU threads are not counted against the pool: they are the ones waiting for the compiles, and they keep interpreting while the pool works. `--jitv2-threads` overrides. |
| `jitv2.compile` | `async` | Where a hot region is compiled: `async` on the pool's threads while the CPU keeps interpreting it (fastest), or `sync` on the CPU thread, so the compiled code runs at once and every run executes the same host code. `sync` needs one CPU and is refused in a `lightning` build. `--jitv2-compile` overrides. |
| `jitv2.cache` | `false` | Opt-in compiled-page disk cache; needs jitv2. |
| `jitv2.cache_dir` | `""` | Blank selects platform user cache directory plus iris/jitv2. |

### [debug]

| Key | Default | Values and behavior |
|---|---|---|
| `debug.gui_gl_capture` | `false` | GUI GL compositor capture path (IRIS_GUI_GL). |
| `debug.no_idle` | `false` | Suppress idle parking (IRIS_NO_IDLE); useful with idle-pause builds. |
| `debug.log` | `""` | Trace points to turn on at startup (`--log`), as for the monitor's `log`: `heart,l1d.*,-l1d.hit`. Some points need a `developer` build. Was `debug_log`, still read. |

### [perf]

| Key | Default | Values and behavior |
|---|---|---|
| `perf.thread_affinity` | `false` | Enable supported host thread-affinity setup. |
| `perf.cpu_core` | `automatic when unset` | Host logical CPU index for the guest CPU thread. |
| `perf.rex3_core` | `automatic when unset` | Host logical CPU index for Newport drawing. |
| `perf.refresh_core` | `automatic when unset` | Host logical CPU index for display refresh. |

### [clock]

| Key | Default | Values and behavior |
|---|---|---|
| `clock.fixed_mhz` | `half the part's clock (75 on an R4400 Indy)` | CP0 Count MHz; IRIX's CPU inventory and the O2's PROM report twice this rate. |

### [rtc_offset]

| Key | Default | Values and behavior |
|---|---|---|
| `rtc_offset.years` | `0` | Signed years offset from host UTC when seeding RTC. |
| `rtc_offset.months` | `0` | Signed months offset from host UTC when seeding RTC. |
| `rtc_offset.days` | `0` | Signed days offset from host UTC when seeding RTC. |
| `rtc_offset.hours` | `0` | Signed hours offset from host UTC when seeding RTC. |
| `rtc_offset.minutes` | `0` | Signed minutes offset from host UTC when seeding RTC. |
| `rtc_offset.seconds` | `0` | Signed seconds offset from host UTC when seeding RTC. |

The RTC's battery keeps its clock running while the machine is off. Once
the guest sets the clock, with a write that moves it by more than two
seconds, iris keeps it in a file beside the RTC's NVRAM
file, `<file>.clock`, and the next run carries on from it with the time
in between added. Until then the clock starts where `[rtc_offset]` puts it
at every run, so an edit to `[rtc_offset]` takes effect. Once the `.clock`
file exists it wins, and iris says so at startup when `[rtc_offset]` is
set. Delete the `.clock` file to start from `[rtc_offset]` again. A guest
that writes its own clock back at shutdown, as IRIX does, sets it if its
clock has drifted. An Octane's NVRAM file with no `.clock` beside it opens
in QEMU at the time of its last save, since QEMU then takes the image's
calendar without adding the time since.

### [o2]

Used with `machine.profile = "o2_ip32"`. An O2 is configured by the shared
keys: `prom` (its PROM image, which overrides the embedded 4.18; `prom.bin`, then `ip32prom.rev4.18.bin`, is tried
when `prom` is left at its default; the raw 512 KiB flash or the IRIX
installer's `ip32prom.image`), `nvram` (the PROM's flash, which holds
its environment; the DS1687's RAM goes beside it in `<nvram>.rtc`, the
Octane's 512-byte image, and its clock in `<nvram>.rtc.clock`), `banks`
(the kit in each of the four DIMM pairs: 32, 64,
128 or 256 MB, largest first), `serial_port_a`/`serial_port_b`/`serial_log`,
`headless` (serial console instead of the graphics console and keyboard),
`[clock] fixed_mhz` (CP0 Count; the CPU runs at twice it, so 90 MHz is a
180 MHz O2), `[network]` including `mac` and `mode = "off"` for an unplugged
Ethernet, `[[port_forward]]`, `nat_subnet`, `[nfs]` and `[[devices]]`
(`scsi0` and `scsi1` are its buses). See [the O2 guide](docs/o2-ip32.md). The old
O2-only spellings (`o2.prom`, `o2.flash`, `o2.ram_mb`, `o2.dimms`,
`o2.disks`, `o2.serial_port`, `o2.serial_log`, `o2.graphics`, `o2.cpu_mhz`,
`o2.net`, `o2.forwards`) are deprecated: they are folded into those keys when
the file is loaded, with a note, and never written back.

| Key | Default | Values and behavior |
|---|---|---|
| `o2.mouse` | `true` | PS/2 mouse on the second port (with the graphics console). |
| `o2.arcs_fpu` | `false` | Test aid: lend a kernel the FPU while it runs inside the PROM. |

### [octane]

Used with `machine.profile = "octane_ip30"` (an R10000, R12000 or R14000,
one or two of them). An Octane is configured by the shared keys: `prom` (the IP30 PROM,
which overrides the embedded 6.5.30 image; `prom.bin`, then `IP30prom.bin`, is tried when it is left at its default), `nvram` (the
DS1685's NVRAM, behind IOC3, its clock in `<nvram>.clock`), `banks` (the MB in each HEART bank: 64, 128, 256,
512, 1024 or 2048, filled from bank 0 with no gap, 8192 in all), `serial_port_a` and
`serial_port_b`/`serial_log` (IOC3's two ports, the first the console),
`headless`, `[clock] fixed_mhz` (CP0 Count; 97.5 for the 195 MHz R10000),
`[network]`, `[[port_forward]]`, `nat_subnet`, `[[devices]]` (`scsi0` is the internal bus, `scsi1` the external) and `[graphics] board`. See [the Octane guide](docs/octane-ip30.md).

| Key | Default | Values and behavior |
|---|---|---|
| `octane.flash` | unset | Boot flash file, keeping the PROM's records between runs. Created from `prom` when missing and then used as it is, so delete it after changing `prom`; a file of another size is refused. Unset, the flash is in memory and every start is from `prom`. |

`octane.graphics` is retired: a config that still has it is refused with a
message naming `[graphics] board`, where the board is now set.

### Hardware constraints and precedence

On the Indy and Indigo2 bank values are 0, 8, 16, 32, 64, or 128 MB, and only
IP28 accepts 256/512 MB. On the O2 a bank is a DIMM pair's kit, 32, 64, 128 or
256 MB, fitted from pair A largest first; on the Octane it is a matched DIMM
pair, 64 MB to 2 GB, filled from bank 0. The Origin 200, Origin 2000 and Onyx2
take 64, 128, 256 or 512 MB per bank, filled from bank 0 (no 256 MB bank with
the 180 MHz R10000, whose PROM disables them); the Fuel takes exactly
`[256, 256]`; the Tezro 512 or 1024 MB banks; the Origin 3200 and Onyx 3000
256 or 512 MB banks per C-brick, in matched pairs. `banks` has eight entries;
Fuel uses two banks, Tezro and IP35 brick machines use up to eight, and
the other profiles use four. Entries beyond the profile's bank count must be 0.
IP22/IP24 four-bank IRIX 6.5 behavior remains separate from IP28's verified
two-bank 1 GB layout. Extreme requires IP22. All non-Newport boards require
`heads=1` and `resolution="guest"`. IP28 IRIX needs IMPACT; its PROM has an
embedded fallback.
The config validator permits SCSI controller 1 only on the machines with two
controllers: IP22, the O2, the Octane, the Origin 200, Origin 2000 and Onyx2,
the Fuel and the Tezro.

Standalone CLI options update the loaded config where supplied; boolean enable
switches do not provide a general way to clear every saved true setting.
For `[debug]` and `[jitv2]`, false/blank values preserve an existing environment
variable; enabled/nonempty config values replace it. This applies even when a
previous GUI Start set the variable. Do not assume environment values always
override explicit config, or that an unchecked GUI option always clears them.
Hardware edits take effect on Stop/Start; GUI networking/forward controls have
live-update paths. See [HELP.md](HELP.md) for procedures and [TODO.md](clanker/TODO.md)
for fidelity and validation gaps.

## Standalone iris CLI inventory

All declared options in `config::Cli` are listed here. Built-in `--help`/`-h`
prints the active parser help; `iris --config FILE` selects the config file.
iris-gui does not parse this standalone option set; use its named-machine UI.

| Switch | Purpose |
|---|---|
| `--config` | Path to iris.toml config file [default: iris.toml] |
| `--prom` | Path to PROM image |
| `--machine NAME` | The machine to emulate, by `[machine] profile` name (`octane_ip30`); overrides the config file. `--machine help` lists the names and exits. |
| `--ip22` | Same as `--machine indigo2_ip22`. Hidden from `--help`; cannot be combined with `--machine`. |
| `--nvram` | Path to NVRAM file (default: nvram.bin in cwd) |
| `--nveeprom` | Path to Indigo2 motherboard EEPROM file (default: nveeprom.bin in cwd) |
| `--bank0` | RAM bank 0 size in MB (0/8/16/32/64/128; IP28 also supports 256/512) |
| `--bank1` | RAM bank 1 size in MB (0/8/16/32/64/128; IP28 also supports 256/512) |
| `--bank2` | RAM bank 2 size in MB (0/8/16/32/64/128; IP28 also supports 256/512) |
| `--bank3` | RAM bank 3 size in MB (0/8/16/32/64/128; IP28 also supports 256/512) |
| `--jitv2-threads` | jitv2 compile-pool threads (default: one per guest CPU, at most 4, capped by the host's usable CPUs; see jitv2.threads) |
| `--jitv2-compile MODE` | Where jitv2 compiles: `async` on the pool's threads (default) or `sync` on the CPU thread, so a run executes the same host code every time. One CPU only. Sets `jitv2.compile`. |
| `--device SPEC` | Place a device: `[NAME:]TYPE[@SLOT][,KEY=VALUE]...`, as a `[[devices]]` entry (`hdd@scsi0/t1,path=irix65.raw`; a value is read as TOML when it parses as one). A named one replaces the file's device of that name. |
| `--list-slots` | Print every slot on the machine, its aliases, what is in it and what it takes, then exit. |
| `--2x` | 2× window scaling for HiDPI/4K monitors |
| `--headless` | Suppress the standalone window (platform-specific graphics/console behavior, as `headless`; audio unaffected) |
| `--noaudio` | Disable host audio output, as `no_audio`; graphics is unaffected |
| `--audio-host <HOST>` | Host audio system: `default`, `pulse`, `wasapi`, `coreaudio` or `null` (overrides `[audio] host`) |
| `--nfs-dir` | Enable NFS share: path to the directory to export (enables NFS) |
| `--nat-subnet` | NAT subnet in CIDR notation (e.g. 192.168.5.0/24). Gateway gets .1, guest (IRIX) gets .2. Default: 192.168.0.0/24. |
| `--net-mode` | Networking backend: "nat" (default, software gateway), "pcap" (bridge onto a real host interface; requires --features pcap), "stream" (a private segment; needs `network.stream`) or "off". |
| `--pcap-interface` | Host interface to bridge onto in PCAP mode (e.g. eth0, en0). Implies --net-mode pcap. List candidates with --list-net-interfaces. |
| `--validate` | Check the configuration (parse, settings, files, directories, ports) without starting the machine; exit 0 or 1. See [HELP.md](HELP.md#checking-a-configuration). |
| `--list-net-interfaces` | Print the host network interfaces libpcap can bridge onto, then exit. Requires a build with --features pcap. |
| `--no-scsi-deferred-int` | Disable deferred SCSI status interrupts (default: enabled for OpenBSD/NetBSD compatibility). |
| `--no-write-cache` | Turn every hard disk's write cache off (`write_cache = false` in each `hdd` and `nvme` device): slower, but a host crash loses no completed write. |
| `--cpu` | Emulated CPU: a family (`r4400`, `r5000`, `r10000`, `r12000`, `r14000`, `r16000`) or a part (`r14000_600`) the machine takes. Overrides `[machine] cpu`. |
| `--cache MODEL` | Cache model: `auto` (default: `shadow` in a jitv2 build, else `full`), `full` or `shadow`. Overrides `[machine] cache`. The R10000 family is always `shadow`. |
| `--graphics` | Graphics board: `newport` (default), `xz` (2 GE7 on Indigo2, 4 on Indy), `elan` (4 GE7, Indigo2), `extreme` (8 GE7, Indigo2), `solidimpact`, `highimpact`, `maximpact` (or `impact:solid`, `impact:high`, `impact:max`), `v6`, `v8`, `v10`, `v12` (`vpro:v12` and so on, Octane; `v10` and `v12` on the Fuel, `v12` on the Tezro and Onyx 3000), `ir2`, `ir2:2rm`, `ir2e`, `ir2e:2rm` (Onyx2), `ir3` (Onyx 3000) or `none`. Overrides `[graphics] board`. |
| `--raster` | VPro, IMPACT and Newport: `cpu` (default) or `wgpu`, the raster unit on the host GPU (`[graphics] raster`; needs a build with the `wgpu` feature). |
| `--renderer` | What draws the window: `gl` (default) or `wgpu` (`renderer`; needs a build with the `wgpu` feature). |
| `--gdb-port` | Enable GDB stub on the given TCP port (e.g. --gdb-port 1234). Connect with: target remote localhost:<port> |
| `--monitor-port` | Monitor console port on 127.0.0.1 (default 8888). Give each iris running at once its own. |
| `--vnc-password-file FILE` | Ask VNC viewers for the password in FILE (first 8 bytes). Without it, `IRIS_VNC_PASSWORD_FILE` or `IRIS_VNC_PASSWORD` (the password itself) in the environment is used. Sets `vnc.password_file`. |
| `--vnc [ADDR]` | Serve the screen, keyboard and mouse over VNC on ADDR (`host:port` or a port; default 127.0.0.1:5900). Sets `vnc.listen`. |
| `--serial-port-a` | Serial channel A (tty2) port on 127.0.0.1 (default 8880). |
| `--serial-port-b` | Serial channel B (tty1, the console) port on 127.0.0.1 (default 8881). |
| `--serial-socket-a`, `--serial-socket-b` | Also serve channel A or B on a Unix socket. |
| `--keyboard LAYOUT` | Physical host keyboard: `ansi` (default), `iso` or `jis`. Use `jis` for a Japanese keyboard with the PROM's `keybd` set to JP. Sets `keyboard`. |
| `--serial-log-a`, `--serial-log-b` | Log what the guest sends on channel A or B when it is not the console. |
| `--device test_device@gio0` | Map the bare-metal test device into GIO expansion slot 0: SIGNATURE, PUTC (guest console → stdout), DUMP (machine state → JSON) and EXIT (terminate with the guest's exit code). Off by default. Its `dump` key says where the machine-state dump goes; `run_config` asks a bare-metal guest for a shorter run, as `time_pct=N,repeats=N,groups=MASK`, any of them. |
| `--cheritest-dump-hook` | cheritest convention: a guest write to CP0 register 26 dumps machine state. Needs a `test_device` device. Never enable for a normal boot — CP0 26 is ECC on a real R4400 and the PROM writes it during cache init. |
| `--tftp-dir` | Serve this directory read-only over TFTP at the gateway address, so the PROM can network-boot from it: `boot -f bootp()<file>`. Off when unset. |
| `--rsh-dir` | Let the guest `rsh <gateway> cat\|dd\|ls\|echo\|pwd` against this directory, read-only, through a built-in rsh server at the gateway (TCP 514). Nothing is executed on the host. Off when unset. |
| `--perfetto FILE` | Record a Perfetto trace from startup (`perfetto` builds). `--perfetto-cats LIST` (default `all`) and `--perfetto-clock host\|guest` (default `host`) select what and on which timeline. See [docs/perfetto.md](docs/perfetto.md). |
| `--load-elf` | Load a static ELF32 MSB (big-endian MIPS) binary into RAM before the CPU starts and set PC to its entry point, instead of booting the PROM. For bare-metal test binaries; see also the monitor's `loadelf`. |
| `--ci` | CI mode: enable the control socket and apply speed-favoring fidelity shortcuts. Hides the window unless --ci-display is set; offscreen graphics remain active unless --headless is explicitly set. |
| `--ci-socket` | Override the default control-socket path (`iris.sock` in `$TMPDIR`, else `/tmp`; `iris-ci` finds it there only under the same `TMPDIR`). |
| `--ci-display` | With --ci, keep the Newport window visible for interactive test development (deferred rendering at 10–15 fps). |
| `--serial-log` | Write every byte the guest emits on its serial console to this file, as `serial_log`. Useful for live tailing during an install. |
| `--serial-log-overwrite` | Start the serial logs over each run instead of adding to them, as `serial_log_overwrite`. |
| `--serial-log-timestamps MODE` | `off` (raw transcript), `wall` (prefix each completed serial-log line with a UTC ISO-8601 timestamp) or `ecs` (one JSON document a line), as `serial_log_timestamps`. |
| `--clock-fixed-mhz` | Override CP0 Count MHz (default half the part's clock: 75 on an R4400 Indy, 97.5 on the IP28's R10000). IRIX reports twice this rate as CPU MHz, e.g. --clock-fixed-mhz 50. |
| `--help`, `-h` | Display parser help. |

`--graphics` has visible aliases `--board` and `--graphics-board`.
Board aliases include `xl`, `gr2`, `gr2_xz`, `gr2-xz`, `gr2_extreme`,
`gr2-extreme`, `impact:solid`, `solid_impact`, `impact:high`, `high_impact`,
`impact:max`, `max_impact`, `vpro:v6` and `vpro_v6` (likewise for V8, V10 and
V12), `ir2:1rm`, `ir2_1rm`, `infinitereality2`, `ir2_2rm`, `ir2x2` and
`infinitereality3`. TOML board parsing also accepts hyphenated IMPACT, VPro
and InfiniteReality2 names and `solid`, `high`, `max`, `maximumimpact`;
canonical values are in the table. A `--config` file named on the command line
must exist; the default `iris.toml` may be missing, and the defaults are used.
See [debug.md](debug.md) for monitor commands and [STORAGE.md](STORAGE.md) for iris-ci; they are runtime tools,
not Cargo build features.

### Companion binaries

Each prints its own usage with `--help` (or with no arguments).

| Binary | Purpose |
|---|---|
| `iris-ci` | Client for the CI control socket: snapshots, scripted console input, file transfer, screenshots. `--socket PATH` (or `$IRIS_SOCKET`) names the socket, defaulting to iris's own (`iris.sock` in `$TMPDIR`, else `/tmp`); `--json` prints raw responses and `-q` is silent on success. See [STORAGE.md](STORAGE.md) and [TESTING.md](TESTING.md). |
| `iris-bench` | Benchmark driver: `run` (in process, or another emulator with `--iris`; `--cpu` takes any part, `--cache` a cache model, `--deterministic` a repeatable run), `host`, `matrix` (CPU x engine cells), `report`, `irix` and `reference`. See [bench/README.md](bench/README.md). |
| `iris-prom` | `cargo run -p iris-prom -- <command>`: `download` and `pack` the catalog's PROM files (`--catalog`, default `data/proms/proms.toml`), `show FILE` (kind, version, segments and banners of a PROM file) and `extract FILE` (its payload or one `--segment`, to `--out`). |
| `mkvh` | `mkvh build <out.img> [options] <name=path> ...` writes an SGI volume header image; `mkvh dump <image>` lists one. |
| `chd-extract` | `chd-extract <input.chd> <output.raw>`: a SCSI hard-disk CHD to a raw image. |
| `coffdump` | Dumps a MIPS COFF file's headers and sections. |
| `jitv2-analyze`, `jitv2-verify`, `jitv2-pcp-dump` | JIT v2 developer tools; built only with `jitv2`. |

## iris-gui persisted settings inventory

Source: [iris-gui/core/src/model/settings.rs](iris-gui/core/src/model/settings.rs). These keys belong
to GUI JSON rather than machine TOML files. View controls update the scales;
the GUI saves machine TOMLs and sandbox access state for you.

| GUI JSON key | Normal load default | Purpose |
|---|---|---|
| `ui_scale` | `1.25` | egui controls scale; sanitized to supported range. |
| `vm_scale` | `0.75` | Maximum framebuffer magnification, independent of UI scale. |
| `language` | `""` | GUI language code (a folder under `iris-gui/locales`); empty follows the system language. |
| `active_machine` | unset | Selected machine folder name. |
| `machines` | — | Legacy machine configs kept in the JSON file; read once to migrate them to TOML folders and never written back. |
| `recent_configs` | empty list | Legacy recent TOML paths; retained for compatibility. |
| `last_config` | unset | Legacy TOML migration source; cleared after conversion. |
| `bookmarks` | empty map | macOS sandbox security-scoped bookmarks by absolute path. |
| `disk_folders` | empty list | Granted recursive sandbox disk-folder access. |
| `library_folders` | empty list | The library's own folders, in order, each with its machines; a machine in none is listed under its family. |
| `display_mode` | `"actual"` | Guest display size: `actual` (one guest pixel to a point; `scale` is read as this), `fit` (keep the aspect ratio) or `stretch`. |
| `integer_only` | `false` | With `fit`, draw only at whole multiples of the guest's pixels. |
| `display_filter` | `"auto"` | Scaling filter: `auto` (sharp at whole multiples, smooth otherwise), `nearest` or `linear`. |
| `pointer_mode` | `"capture"` | Guest mouse: `capture` it, or `follow` the host pointer (for remote desktops). |
| `convert_declined` | empty set | Absolute paths of compressed CHDs the user chose to keep: no more suggestions to convert them. |

eframe also persists its own viewport/window state. These framework preferences
are separate from the shared machine schema. First-launch fallback window size
is 1512×1024 logical points; monitor fitting and running framebuffer size refine
it. GUI scale defaults refer to the normal load/sanitization path.

## Environment and diagnostic controls

These controls are process-wide and many are read once. Set them before launch;
monitor toggles are preferable when available. Except for the documented host
QoS behavior and Windows crash handler, opt-in diagnostic controls are inactive
when unset; JIT/inline-memory dispatch is normally enabled when built. Test-only
and offline-tool controls are listed in their own table below so they are not
mistaken for GUI settings.

| Variable | Effect / accepted value | Reader |
|---|---|---|
| `GR2_JIT_DISASM` | Any presence prints each GR2 span shader's machine code as it compiles. | [src/dev/gr2/re3_jit/compiler.rs](src/dev/gr2/re3_jit/compiler.rs) |
| `GR4_JIT_DISASM` | Any presence prints each IMPACT raster shader's machine code as it compiles. | [src/dev/mgras/rss_jit/compiler.rs](src/dev/mgras/rss_jit/compiler.rs) |
| `IRIS_BAKE_HOOKS` | Any presence bakes callout hook addresses into JIT code (experimental). | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_AUDIO_DUMP` | File to record what every sound card plays (HAL2, MACE audio, RAD1, ICE1724) at the card's own rate, before conversion. RAD1 and ICE1724 record only with `[audio]` enabled. The layout is in the module. | [src/dev/audio_dump.rs](src/dev/audio_dump.rs) |
| `IRIS_BREAK_CPU` | 1 stops CPU after exception handling for diagnostic inspection. | [src/cpu/mips_exec.rs](src/cpu/mips_exec.rs) |
| `IRIS_PPMEM_BACKEND` | `mach` backs guest RAM with Mach memory entries instead of POSIX shared memory (macOS), as the App Sandbox build does; for testing that path. | [src/ppmem/map_unix.rs](src/ppmem/map_unix.rs) |
| `IRIS_CHD_DIFF_DIR` | Redirect CHD diff sidecars; GUI App Store sets a writable container directory. | [src/chd_disk.rs](src/chd_disk.rs) |
| `IRIS_CRASH_DIAG` | off disables Windows crash handler installation. | [src/crash_diag.rs](src/crash_diag.rs) |
| `IRIS_CRASH_SELFTEST` | Standalone crash diagnostic self-test selector; deliberately crashes. | [src/main.rs](src/main.rs) |
| `IRIS_DISPLAY_VERIFY` | 1 checks the boards' "nothing changed" verdicts: a frame IMPACT or VPro would have left standing is composed anyway and compared with the one on screen; a difference is reported on stderr, with a count every 600 checks. For development; it composes every frame. | [src/display/verify.rs](src/display/verify.rs) |
| `IRIS_ENTRY_PREAMBLE` | 0 turns off jitv2's interrupt check at an externally dispatched entry word (on by default; `j2 entrypre`). | [src/cpu/jitv2/mod.rs](src/cpu/jitv2/mod.rs) |
| `IRIS_GPRCACHE` | Guest registers a JIT v2 function with a loop keeps in host registers, 0 to 31 (`j2 gprcache`; default 6, 10 on AArch64). | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_GPRCACHE_EVERYWHERE` | Presence caches registers in functions without a loop too. For test runs: slower to compile, no faster to run. | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_GR2_JIT` | Starting mode of the GR2 span JIT (`gr2-jit` builds): `off` or `0`, `sync` (compile on first use and wait), `async`, `on` or `1` (the default). `gr2 jit` changes it later. | [src/dev/gr2/re3_jit/mod.rs](src/dev/gr2/re3_jit/mod.rs) |
| `IRIS_GR4_JIT` | Starting mode of the IMPACT raster JIT (`gr4-jit` builds): `off` or `0`, `sync`, `async`, `on` or `1` (the default). `mgras jit` changes it later. | [src/dev/mgras/rss_jit/mod.rs](src/dev/mgras/rss_jit/mod.rs) |
| `IRIS_GUI_GL` | 1 selects the GUI GL capture path. | [iris-gui/core/src/model/framebuffer.rs](iris-gui/core/src/model/framebuffer.rs) |
| `IRIS_HAL2_CAPTURE` | HAL2 host audio capture: a WAV of what the host device is given, at the device's rate. | [src/dev/hal2.rs](src/dev/hal2.rs) |
| `IRIS_HOSTGL_CALLS` | Substring filter for HostGL call logging (bounded log). | [iris-hostgl/src/service.rs](iris-hostgl/src/service.rs) |
| `IRIS_HOSTGL_INLINE` | 1 runs the HostGL service on the CPU thread instead of its own (for comparison; the CPU stalls while a buffer replays). | [iris-hostgl/src/lib.rs](iris-hostgl/src/lib.rs) |
| `IRIS_HOSTGL_NOSURFACE` | 1 disables CGL IOSurface presentation. | [iris-hostgl/src/backend/cgl.rs](iris-hostgl/src/backend/cgl.rs) |
| `IRIS_HOSTGL_QOS` | default disables HostGL thread QoS adjustment. | [iris-hostgl/src/qos.rs](iris-hostgl/src/qos.rs) |
| `IRIS_HOSTGL_STRINGS` | host passes through host GL strings instead of compatibility strings. | [iris-hostgl/src/service.rs](iris-hostgl/src/service.rs) |
| `IRIS_ICE1724_AUDIO_CAPTURE` | Tezro (ICE1724) host audio capture: a WAV of what the host device is given, at the device's rate. | [src/dev/ice1724.rs](src/dev/ice1724.rs) |
| `IRIS_IDLE_PEER_US` | `idle-pause` builds, on a multiprocessor: the first slice in microseconds after which a parked CPU looks for work again while another CPU is busy (default 100; it doubles up to eight times that while none turns up). | [src/cpu/idle_park.rs](src/cpu/idle_park.rs) |
| `IRIS_IDLE_SLICE_US` | `idle-pause` builds, on a multiprocessor: the longest park in microseconds while every CPU is idle (default 10000); a CPU that ran an interrupt handler or other work wakes the others sooner. | [src/cpu/idle_park.rs](src/cpu/idle_park.rs) |
| `IRIS_IDLE_WARMUP_SECS` | `idle-pause` builds: seconds after CPU start before idle parking engages (default 60; 0 disables the warm-up). | [src/cpu/idle_park.rs](src/cpu/idle_park.rs) |
| `IRIS_INTRUN` | Integer interrupt-check coalescing budget for JIT v2 code; also used by offline tools. | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs), [src/bin/jitv2_pcp_dump.rs](src/bin/jitv2_pcp_dump.rs) |
| `IRIS_IP32_RE_TRACE` | Number of O2 rendering-engine primitives to record for `o2 retrace PATH`. | [src/dev/o2gfx.rs](src/dev/o2gfx.rs) |
| `IRIS_IP32_RE_LAG_US` | Test aid: the O2 rendering engine waits this many microseconds before each memory transfer, as on a host that keeps its thread waiting. Used with `tools/prom-boot-loop.py` to check that the PROM still reaches its menu (see [TESTING.md](TESTING.md#repeated-prom-boots)). | [src/dev/o2gfx.rs](src/dev/o2gfx.rs) |
| `IRIS_JIT_ARENA_FLUSH_MB` | Treat a code arena as full when it holds this many MiB, so the compile pool switches arenas and retires the oldest often. For testing the arena rotation. | [src/cpu/jitv2/jitv2.rs](src/cpu/jitv2/jitv2.rs) |
| `IRIS_JIT_PAGES` | Size of JIT v2's page pool in pages (default 4096, at least 16), so the pool recycles page slots often. For testing the recycling. | [src/cpu/jitv2/jitv2.rs](src/cpu/jitv2/jitv2.rs) |
| `IRIS_JIT_ATOMIC_GEN` | Any presence makes inline stores bump the page generation atomically with one CPU too, as they always do with several. | [src/cpu/mips_exec.rs](src/cpu/mips_exec.rs) |
| `IRIS_JIT_CACHE` | 1 or on enables persistent JIT page reuse. | [src/cpu/jitv2/pcache.rs](src/cpu/jitv2/pcache.rs) |
| `IRIS_JIT_CACHE_DIR` | Persistent JIT cache base directory. | [src/cpu/jitv2/pcache.rs](src/cpu/jitv2/pcache.rs) |
| `IRIS_JIT_CLIF` | Any presence prints Cranelift input IR. | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_JIT_DISASM` | Any presence requests generated machine-code disassembly. | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_JIT_DISPATCH` | off or 0 disables normal JIT v2 dispatch at startup. | [src/cpu/mips_exec.rs](src/cpu/mips_exec.rs) |
| `IRIS_JIT_HASHSTATS` | 1 enables page/entry reuse diagnostics. | [src/cpu/jitv2/hashstats.rs](src/cpu/jitv2/hashstats.rs) |
| `IRIS_JIT_HASHSTATS_LOG` | Append per-compile reuse diagnostic records to this file. | [src/cpu/jitv2/hashstats.rs](src/cpu/jitv2/hashstats.rs) |
| `IRIS_JIT_OPT` | none or speed sets JIT v2's Cranelift optimisation level from the start of the run (`j2 opt` changes it later). | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_JIT_PIC` | Any presence disables core-address constants for JIT code. | [src/cpu/mips_exec.rs](src/cpu/mips_exec.rs) |
| `IRIS_JIT_REGALLOC` | single_pass selects Cranelift's one-pass register allocator for JIT v2: compiles about a third faster, code about half as fast. For measurement. | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_JIT_VERIFY` | 1 or on runs Cranelift's IR verifier on every JIT v2 compile. Already on in unit tests and `--features developer` builds. | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_MEM_HELPERS` | Any presence enables the experimental JIT memory-helper path. | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_MGRAS_ENGINES` | IMPACT raster engine threads: `1`, `2` or `4` (default 2 on Maximum IMPACT, 1 on the others). More threads split the scanlines without changing what the guest sees. | [src/dev/mgras/mod.rs](src/dev/mgras/mod.rs) |
| `IRIS_MGRAS_REC` | Record IMPACT bus operations from power-on to this path. | [src/dev/mgras/mod.rs](src/dev/mgras/mod.rs) |
| `IRIS_MGRAS_TRACE` | Trace IMPACT accesses from power-on to this path (`gputrace` builds). | [src/dev/mgras/mod.rs](src/dev/mgras/mod.rs) |
| `IRIS_NO_EXIT_ON_POWEROFF` | Any presence prevents process exit on guest power-off/CI quit; set by GUI. | [src/ci.rs](src/ci.rs) |
| `IRIS_NO_FETCH_VERIFY` | 1 suppresses fetchverify checks in a fetchverify build. | [src/cpu/mips_exec.rs](src/cpu/mips_exec.rs) |
| `IRIS_NO_IDLE` | Any presence disables idle parking at startup (`idle-pause` builds); `idle on` re-enables it. | [src/cpu/idle_park.rs](src/cpu/idle_park.rs) |
| `IRIS_NO_INLINE_MEM` | Any presence disables JIT inline memory emission. | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_NO_JIT` | Any presence disables the REX3 shader JIT, the IMPACT raster JIT and the GR2 span JIT; does not select the MIPS engine. | [src/dev/ng1/rex3.rs](src/dev/ng1/rex3.rs), [src/dev/mgras/rss_jit/mod.rs](src/dev/mgras/rss_jit/mod.rs), [src/dev/gr2/re3_jit/mod.rs](src/dev/gr2/re3_jit/mod.rs) |
| `IRIS_O2_AUDIO_CAPTURE` | O2 host audio capture: a WAV of what the host device is given, at the device's rate. | [src/dev/mace_audio.rs](src/dev/mace_audio.rs) |
| `IRIS_O2_TRACE` | Trace the O2's rendering engine, GBE and CRIME from power-on to this path (`gputrace` builds). | [src/dev/o2gfx_trace.rs](src/dev/o2gfx_trace.rs) |
| `IRIS_OCTANE_AUDIO_CAPTURE` | Octane and Onyx2 (RAD1) host audio capture: a WAV of what the host device is given, at the device's rate. | [src/dev/rad1.rs](src/dev/rad1.rs) |
| `IRIS_ODSY_BACKEND` | `inline` keeps the VPro fragment pipeline's shading and back end (depth, stencil, blend and stores) on the raster thread, with no worker pool. | [src/dev/odyssey/raster_unit.rs](src/dev/odyssey/raster_unit.rs) |
| `IRIS_ODSY_REC` | Record VPro bus stores, loads and DMA reads from power-on to this path (gzip; replayed by the tests in `src/dev/odyssey/golden/`; `vpro rec` does the same from the monitor). | [src/dev/odyssey/record.rs](src/dev/odyssey/record.rs) |
| `IRIS_ODSY_SHADER` | `inline` walks and shades VPro filled triangles on the raster thread; the tile workers then only store. | [src/dev/odyssey/raster_unit.rs](src/dev/odyssey/raster_unit.rs) |
| `IRIS_ODSY_TRACE` | Trace VPro command words, GL commands, bus accesses and DMA descriptors from power-on to this path (`gputrace` builds; `vpro trace` from the monitor). Also traces an offline replay (`VPRO_REPLAY`); a test that replays several sessions adds `.1`, `.2` and so on for the later ones. | [src/dev/odyssey/trace.rs](src/dev/odyssey/trace.rs) |
| `IRIS_QOS` | HostGL QoS: default/off disables promotion; quiet suppresses logging. | [iris-hostgl/src/qos.rs](iris-hostgl/src/qos.rs) |
| `IRIS_REX3_GPU_STATS` | Set (any value) to add the costlier Newport GPU-raster telemetry to `rex status`: command generation time per GO and which state-slot words change from one primitive to the next. Read when the GPU raster starts. | [src/dev/ng1/wgpu/resident.rs](src/dev/ng1/wgpu/resident.rs) |
| `IRIS_REX3_REC` | Record REX3 bus operations from board start to this path (`gputrace` builds). | [src/dev/ng1/rex3_rec.rs](src/dev/ng1/rex3_rec.rs) |
| `IRIS_REX3_TRACE` | Trace REX3 GFIFO and CPU accesses from board start to this path (`gputrace` builds). | [src/dev/ng1/rex3_trace.rs](src/dev/ng1/rex3_trace.rs) |
| `IRIS_REX_JIT_PROFILE` | Where the REX3 shader JIT keeps its profile of draw modes between runs, instead of `~/.iris/rex-jit-profile.bin` (`rex-jit` builds). | [src/dev/ng1/rex3_profile.rs](src/dev/ng1/rex3_profile.rs) |
| `IRIS_SNAPSHOT_SKIP_CHECK` | 1 bypasses selected snapshot compatibility checks; diagnostic escape hatch. | [src/machine.rs](src/machine.rs) |
| `IRIS_SOCKET` | iris-ci default socket override (Unix path or TCP address). | [src/iris_ci_main.rs](src/iris_ci_main.rs) |
| `IRIS_VNC_PASSWORD` | The VNC password itself (first 8 bytes), used when neither `[vnc] password_file` nor `IRIS_VNC_PASSWORD_FILE` is set. Standalone `iris` only. | [src/vnc/auth.rs](src/vnc/auth.rs) |
| `IRIS_VNC_PASSWORD_FILE` | File holding the VNC password, used when `[vnc] password_file` is unset; tried before `IRIS_VNC_PASSWORD`. | [src/vnc/auth.rs](src/vnc/auth.rs) |
| `IRIS_WGPU_DUMP` | With the `wgpu` feature: a directory the composed WGSL of every GPU kernel is written to as it is compiled, for `naga` and `spirv-val`. | [src/wgpu/shader.rs](src/wgpu/shader.rs) |
| `IRIS_WGPU_LOAD` | With the `wgpu` feature: a directory of kernels to compile instead of the built-in ones. A kernel is replaced when the directory holds a file of the name `IRIS_WGPU_DUMP` writes for it, so a dumped kernel can be edited and tried without a rebuild. | [src/wgpu/shader.rs](src/wgpu/shader.rs) |
| `MGRAS_BIN_EDGES` | With the `wgpu` feature: `0` makes the IMPACT GPU unit's bin pass place GL triangles by their box alone, instead of also testing their span bounds at each 32x32 coarse tile and 8x8 tile; for comparing the two. | [src/dev/mgras/wgpu/mod.rs](src/dev/mgras/wgpu/mod.rs) |
| `REX3_FORCE_FULL_UPLOAD` | `1` restores the two-full-plane Newport texture upload in a `rex-dirty-upload` build, for comparison. | [src/gl_compositor.rs](src/gl_compositor.rs) |
| `REX3_GPU_BACKEND` | Backend for the GPU raster in REX3 replay: `auto`, `vulkan`, `metal`, `dx12` or `gl`. | [src/dev/ng1/rex3_rec.rs](src/dev/ng1/rex3_rec.rs) |
| `VPRO_BIN_EDGES` | With the `wgpu` feature: `0` makes the VPro GPU unit's bin pass place triangles by their box alone, instead of also testing their edges at each 32x32 coarse tile and 8x8 tile; for comparing the two. | [src/dev/odyssey/wgpu/mod.rs](src/dev/odyssey/wgpu/mod.rs) |
| `VPRO_TILE_CACHE` | With the `wgpu` feature: `1` makes the VPro GPU raster kernel hold each 8x8 tile's blocks of the destination in workgroup memory for the batch and store the changed ones back at the end of the tile, instead of an atomic per access; off by default (slower on the measured workloads). | [src/dev/odyssey/wgpu/mod.rs](src/dev/odyssey/wgpu/mod.rs) |
| `WGPU_BACKEND` | With the `wgpu` feature: the wgpu backend to open, overriding `[graphics] gpu_backend` (a comma list of `vulkan`, `metal`, `dx12`, `gl`). The other variables wgpu reads itself apply too, for example `WGPU_VALIDATION`, `WGPU_DEBUG`, `WGPU_GL_FENCE_BEHAVIOR` and `WGPU_GLES_MINOR_VERSION`. | [src/wgpu/device.rs](src/wgpu/device.rs) |
| `WGPU_BATCH_LOG` | With `WGPU_BATCH_TIMING`: write one line per batch to this file: primitives, busy tiles, tile-primitive pairs, bin ns, raster ns, graphics and memory clock in MHz (0 when unknown). | [src/wgpu/batch.rs](src/wgpu/batch.rs) |
| `WGPU_BATCH_TIMING` | With the `wgpu` feature, on a device with timestamp queries: time the bin and raster passes of every GPU batch and count its busy tiles; the VPro and IMPACT replay tests print the totals. Each batch is then waited for, so the GPU times are true and the wall time is not. On Linux with an AMD card whose `gpu_metrics` it can read, the report also gives the average graphics and memory clocks the raster pass ran at: the driver lowers them while the GPU is mostly idle, so compare GPU times only at the same clocks (pin them with `power_dpm_force_performance_level`). | [src/wgpu/batch.rs](src/wgpu/batch.rs) |
| `WGPU_NO_SUBGROUPS` | With the `wgpu` feature: the raster kernel's quads exchange values through workgroup memory instead of subgroup quad operations, as on a device without subgroups. For testing that path. | [src/wgpu/batch.rs](src/wgpu/batch.rs) |

### Test, replay and internal controls

These are read by unit and integration tests, ignored replay and benchmark
tests, and subprocesses the tests start. A normal run never needs them.

| Variable | Effect / accepted value | Reader |
|---|---|---|
| `BENCH_MB` | Megabytes written by the ignored block-device sync benchmark (default 256). | [src/block_node_sync_bench.rs](src/block_node_sync_bench.rs) |
| `BENCH_SECS`, `BENCH_SCENE` | Seconds per measurement, and one scene, for the ignored VPro raster and display and InfiniteReality benchmarks. | [src/dev/odyssey/raster_bench.rs](src/dev/odyssey/raster_bench.rs), [src/dev/odyssey/display_bench.rs](src/dev/odyssey/display_bench.rs), [src/dev/kona/back/bench.rs](src/dev/kona/back/bench.rs) |
| `GR2_BENCH_N` | Primitive count of the ignored GR2 benchmark (default 20000). | [src/dev/gr2/hq2_tests.rs](src/dev/gr2/hq2_tests.rs) |
| `GR2_JIT_SWEEP`, `GR2_JIT_SEED` | Widen and reseed the GR2 span JIT's key sweep against the interpreter. | [src/dev/gr2/re3_jit_tests.rs](src/dev/gr2/re3_jit_tests.rs) |
| `GR2_REPLAY` | Replay input path for opt-in GR2 tests. | [src/dev/gr2/hq2_tests.rs](src/dev/gr2/hq2_tests.rs) |
| `GR2_REPLAY_OUT` | GR2 replay dump prefix (default gr2replay). | [src/dev/gr2/hq2_tests.rs](src/dev/gr2/hq2_tests.rs) |
| `GR4_JIT_SWEEP`, `GR4_JIT_SEED` | Widen and reseed the IMPACT raster JIT's key sweep against the interpreter. | [src/dev/mgras/rss_jit_tests.rs](src/dev/mgras/rss_jit_tests.rs) |
| `IRIS_AUDIO_LOOPBACK` | Real-device audio tests: `<output host>:<capture source>`, the `[audio] host` a card plays through (`default` or `pulse`) and the PulseAudio source that hears the default output (`default:iris_sink.monitor`). Linux; the tests skip when it is unset. | [src/host_audio.rs](src/host_audio.rs) |
| `IRIS_BENCH_ELF`, `IRIS_BENCH_RESULTS` | The benchmark-accuracy test: a rebuilt suite binary instead of `bench/prebuilt/irisbench.elf`, and a directory to collect the parsed results in. | [tests/bench_accuracy.rs](tests/bench_accuracy.rs) |
| `IRIS_CONST_DUMP` | Presence prints selected disassembly lines in constant-dedup tests. | [src/cpu/jitv2/mod.rs](src/cpu/jitv2/mod.rs) |
| `IRIS_CONST_MODE` | Select the opt-in constant-dedup codegen test shape. | [src/cpu/jitv2/mod.rs](src/cpu/jitv2/mod.rs) |
| `IRIS_CORPUS_DIR` | Corpus directory consumed by opt-in JIT corpus tests. | [src/cpu/jitv2/mod.rs](src/cpu/jitv2/mod.rs) |
| `IRIS_CPUTEST_ELF` | The cpu-tests binary the integration tests run, instead of `cpu-tests/build/cputest.elf`. | [tests/common/cputest.rs](tests/common/cputest.rs) |
| `IRIS_GUI_STORAGE_TEST` | Internal subprocess switch of the GUI storage tests: the machine-storage root. | [iris-gui/src/main.rs](iris-gui/src/main.rs) |
| `IRIS_HOT_PATH_BINARY` | The release binary the hot-path checks disassemble, instead of `target/release/iris`. | [iris-checks/tests/hot_paths.rs](iris-checks/tests/hot_paths.rs) |
| `IRIS_HTTP_TEST_URL` | `http(s)://` image URL for the ignored HTTP media test against a real server. | [src/http_disk.rs](src/http_disk.rs) |
| `IRIS_IP30_PROM` | Test-only PROM path for PROM-backed Octane tests; defaults to `/inputs/IP30prom.bin` and skips when missing. | [src/ip30_machine.rs](src/ip30_machine.rs) |
| `IRIS_KILL_CHD` | Disk-image path used by the ignored killed-process CHD persistence benchmark. | [src/block_node_sync_bench.rs](src/block_node_sync_bench.rs) |
| `IRIS_OPT_SPEED` | Presence selects speed optimization in offline PCP/corpus measurements. | [src/bin/jitv2_pcp_dump.rs](src/bin/jitv2_pcp_dump.rs) |
| `IRIS_PERFETTO_DEMO` | Output file for the ignored Perfetto demonstration trace test. | [src/perfetto/tests.rs](src/perfetto/tests.rs) |
| `IRIS_PRINT_OFFSETS` | Presence runs the opt-in layout-offset print test. | [src/cpu/jitv2/mod.rs](src/cpu/jitv2/mod.rs) |
| `IRIS_PROM_DIR` | Test-only directory of real PROM images, default `ignore/proms`; payload decoder tests skip when no images are found. | [src/dev/prom_image.rs](src/dev/prom_image.rs) |
| `IRIS_RESIDENT_FULL_FIFO_LOSS_CHILD`, `IRIS_RESIDENT_SCREENSHOT_CHILD`, `IRIS_REX3_LEGACY_ENV_CHILD`, `IRIS_REX3_RESIDENT_GATE_CHILD`, `IRIS_REX3_PLANE_FAULT_PROBE` | Internal child-process selectors of the Newport GPU and plane tests. | [src/dev/ng1/rex3_frame.rs](src/dev/ng1/rex3_frame.rs), [src/dev/ng1/wgpu/direct_tests.rs](src/dev/ng1/wgpu/direct_tests.rs), [src/dev/ng1/rex3_tests.rs](src/dev/ng1/rex3_tests.rs), [src/dev/ng1/rex3_plane_tests.rs](src/dev/ng1/rex3_plane_tests.rs) |
| `IRIS_RUN_CL_PROBE` | Presence runs the opt-in Cranelift probe test. | [src/cpu/jitv2/mod.rs](src/cpu/jitv2/mod.rs) |
| `IRIS_SUITE_JUNIT`, `IRIS_SUITE_TIMEOUT` | Integration suites: a directory for their JUnit reports, and a timeout override. | [tests/common/mod.rs](tests/common/mod.rs) |
| `IRIS_TEST_APPLY_ENV_EXTERNAL`, `IRIS_TEST_APPLY_ENV_OURS` | Variables the `[debug]`/`[jitv2]` environment-publishing tests set and check. | [src/config.rs](src/config.rs) |
| `IRIS_TEST_CAPTURE` | Capture variable the host-audio unit tests open their output with. | [src/host_audio.rs](src/host_audio.rs) |
| `IRIS_TEST_GPR_CACHE_MODE` | Internal child-process selector used by the JIT register-cache equivalence tests. | [src/cpu/jitv2/equiv_test.rs](src/cpu/jitv2/equiv_test.rs) |
| `IRIS_TEST_PRINT_CI_SOCKET` | Internal subprocess-test switch that prints the resolved default CI socket. | [src/config.rs](src/config.rs) |
| `KONA_RECORDINGS`, `KONA_GL_RECORDING`, `KONA_QUAKE_RECORDING` | Recordings for the ignored InfiniteReality replay tests. `KONA_SHOT_DIR` keeps composed frames, `KONA_STOP_AT=<n>` ends a replay after n records and `KONA_PROGRESS=<n>` reports progress every n records. | [src/dev/kona/back/replay.rs](src/dev/kona/back/replay.rs) |
| `LINETEST_OUT` | Output directory of the ignored VPro line-drawing test. | [src/dev/odyssey/golden/linetest.rs](src/dev/odyssey/golden/linetest.rs) |
| `MGRAS_GPU_CHECK_EACH` | With the `wgpu` feature: the IMPACT GPU unit's parity tests compare memories after every primitive instead of every step, to find the primitive that differs. | [src/dev/mgras/wgpu/tests.rs](src/dev/mgras/wgpu/tests.rs) |
| `MGRAS_REBASELINE`, `MGRAS_REBASELINE_OUT` | Input and output of the ignored IMPACT recording re-baseline test. | [src/dev/mgras/record.rs](src/dev/mgras/record.rs) |
| `MGRAS_REPLAY` | Replay input path for opt-in IMPACT golden tests. `MGRAS_REPLAY_GPU=1` replays on the GPU raster unit (`wgpu` feature) and compares its scan-out with the CPU compose at every checkpoint (`MGRAS_REPLAY_NOSCAN=1` skips that, for timing); `MGRAS_REPLAY_OCTANE=1` sets the board up as the Octane does (a recording made there); `MGRAS_REPLAY_WARM=1` replays once untimed first so the raster JIT's shaders are compiled; `MGRAS_REPLAY_INIT=<reg>=<val>,...` writes raster registers (hex) before the stream, for a recording that starts mid-session (`16e=9000025a`: the DRBsize of Maximum IMPACT at 1024x768); `MGRAS_REPLAY_HASHES_OUT=<file>` writes every checkpoint's state hash and `MGRAS_REPLAY_HASHES_IN=<file>` reports the first that differs from such a list; `MGRAS_REPLAY_DUMP_AT=<n>` with `MGRAS_REPLAY_DUMP=<file>` dumps pixel memory and clip IDs after checkpoint n; `MGRAS_REPLAY_TRACE=<file>` with `MGRAS_REPLAY_TRACE_MASK=hq,rss,cpu,tex` traces the replay. | [src/dev/mgras/record.rs](src/dev/mgras/record.rs) |
| `MGRAS_REPLAY_OUT` | Optional IMPACT replay image output path. | [src/dev/mgras/record.rs](src/dev/mgras/record.rs) |
| `MGRAS_REPLAY_TRACE` | IMPACT replay tracing selector (`gputrace` builds). | [src/dev/mgras/record.rs](src/dev/mgras/record.rs) |
| `MGRAS_SIZE_PROBE` | Input for the opt-in IMPACT size/probe test. | [src/dev/mgras/mgras_tests.rs](src/dev/mgras/mgras_tests.rs) |
| `MGRAS_TEST_BOARD`, `MGRAS_REPLAY_BOARD`, `MGRAS_REPLAY_NOCHECK` | `max` runs the IMPACT board tests, or a replay, on Maximum IMPACT's two raster engines; `MGRAS_REPLAY_NOCHECK` replays a recording that starts mid-session for timing and its final state, unchecked. | [src/dev/mgras/record.rs](src/dev/mgras/record.rs) |
| `REX3_CORPUS_IO_DIR`, `REX3_CORPUS_MUTATOR_EXPECT`, `REX3_CORPUS_MUTATOR_LINK`, `REX3_CORPUS_MUTATOR_TARGET` | The ignored corpus-writer filesystem test's build directory, and its child process's internal selectors. | [src/dev/ng1/rex3_corpus_io.rs](src/dev/ng1/rex3_corpus_io.rs) |
| `REX3_CORPUS_OUT` | Existing directory, relative to the repository, the ignored REX3 shader-corpus generator writes into. | [src/dev/ng1/rex3_corpus.rs](src/dev/ng1/rex3_corpus.rs) |
| `REX3_REPLAY` | Recording for the ignored one-file REX3 replay test. `REX3_REPLAY_ENGINE` is `jit` (default), `interp`, or `resident` (the GPU raster, `wgpu`, with `REX3_GPU_BACKEND`); `REX3_REPLAY_LIMIT=<n>` replays the first n records; `REX3_REPLAY_NOCHECK` skips the checkpoints; `REX3_REPLAY_TRACE=<file>` traces the replay; `REX3_REPLAY_STATE=<file>` writes the final register state as TOML; `REX3_REPLAY_OUT=<png>` saves the screen. | [src/dev/ng1/rex3_rec.rs](src/dev/ng1/rex3_rec.rs) |
| `REX3_RESIDENT_BACKEND` | Test-only resident GPU adapter backend: `auto`, `vulkan`, `dx12`, `metal` or `gl`. The ignored hardware tests require the selected backend and a hardware adapter; this variable does not enable a live renderer. See [TESTING.md](TESTING.md). | [src/dev/ng1/wgpu/resident_tests.rs](src/dev/ng1/wgpu/resident_tests.rs) |
| `RE_REPLAY`, `RE_SPEED` | O2 rendering-engine traces (`IRIS_IP32_RE_TRACE`, saved with `retrace`) for the ignored replay and replay-speed tests. `RE_REPLAY_PNG` names the replay's image (default `replay.png`); `RE_REPLAY_BUF`, `RE_REPLAY_LOW`, `RE_REPLAY_HALF`, `RE_REPLAY_PX`, `RE_REPLAY_PRELOAD`, `RE_REPLAY_PRELOAD_EVERY` and `RE_GOURAUD` select what it shows and preloads; `RE_SPEED_PASSES` (default 5) the timed passes. | [src/dev/o2gfx.rs](src/dev/o2gfx.rs) |
| `VPRO_GOLDEN` | `tolerance` lets a VPro golden frame whose hash changed pass inside its stored limits. | [src/dev/odyssey/golden/mod.rs](src/dev/odyssey/golden/mod.rs) |
| `VPRO_GOLDEN_BLESS`, `VPRO_GOLDEN_SCENE` | The ignored `bless` test requires `VPRO_GOLDEN_BLESS` before replacing reference hashes/images; `VPRO_GOLDEN_SCENE` restricts it to one scene. | [src/dev/odyssey/golden/mod.rs](src/dev/odyssey/golden/mod.rs) |
| `VPRO_GOLDEN_OUT` | Directory for images of failing golden frames; does not replace the references. | [src/dev/odyssey/golden/mod.rs](src/dev/odyssey/golden/mod.rs) |
| `VPRO_GPU_TRACE_WAITS` | With the `wgpu` feature: the VPro GPU unit counts its readbacks by the board function that caused them; the replay prints the histogram. | [src/dev/odyssey/wgpu/mod.rs](src/dev/odyssey/wgpu/mod.rs) |
| `VPRO_PARITY_DIR` | Recording directory for CPU/GPU replay parity tests, default `ignore/recordings/gpu-cases`. | [src/dev/odyssey/golden/replay.rs](src/dev/odyssey/golden/replay.rs) |
| `VPRO_REPLAY` | Replay input path for one VPro recording; `VPRO_REPLAY_DIR` picks the directory that is replayed otherwise, `VPRO_REPLAY_THREADS=1` replays with the raster and back-end threads running, `VPRO_REPLAY_GPU=1` replays on the GPU raster unit (`wgpu` feature; with `VPRO_REPLAY_LAZY=1` the unit skips the readbacks between checkpoints, to time the GPU path alone), `VPRO_REPLAY_LIMIT=<n>` stops after n records and checks only the checkpoints reached (with `VPRO_REPLAY_DUMP` to see the board there), `VPRO_REPLAY_TICK=<n>` ends a board slice every n stores as a running machine does (on the GPU unit each slice end can submit a batch), `VPRO_REPLAY_WRITES=<file>` logs every write the board makes to guest memory (pixel reads, fences, query results); the report line gives their hash and the final memory hash, to compare two replays, and the time the stores took to draw without the checkpoints' hashing ("(x s drawing)"), to time two builds. `VPRO_GPU=1` puts every VPro test board (the golden scenes among them) on the GPU raster unit. | [src/dev/odyssey/golden/replay.rs](src/dev/odyssey/golden/replay.rs) |
| `VPRO_REPLAY_BLESS`, `VPRO_REPLAY_REQUIRE` | The ignored checkpoint-replacement test requires `VPRO_REPLAY_BLESS`; `VPRO_REPLAY_REQUIRE` makes an empty replay corpus fail instead of skip. | [src/dev/odyssey/golden/replay.rs](src/dev/odyssey/golden/replay.rs) |
| `VPRO_REPLAY_OUT`, `VPRO_REPLAY_SHOTS`, `VPRO_REPLAY_EVERY` | Save the final replay screen to a PNG, save each checkpoint screen in a directory, or additionally save a screen every N records. | [src/dev/odyssey/golden/replay.rs](src/dev/odyssey/golden/replay.rs) |
| `VPRO_REPLAY_POOL` | With threaded replay, `<threads>,<tile>` selects the worker count and tile size (default `2,32`). | [src/dev/odyssey/golden/replay.rs](src/dev/odyssey/golden/replay.rs) |
| `VPRO_SCANOUT`, `VPRO_SCANOUT_PNG` | Comma-separated VPro display captures for the ignored compose test, and a directory for its PNGs. | [src/dev/odyssey/display_bench.rs](src/dev/odyssey/display_bench.rs) |
| `WGPU_TEST_BACKEND` | wgpu backend for the ignored common-layer GPU tests (default `auto`). | [src/wgpu/tests/mod.rs](src/wgpu/tests/mod.rs) |

`RUST_LOG` configures the host Rust logger (errors only when unset);
`--log` configures device logging. Standard host variables (`HOME`,
`XDG_CACHE_HOME`, `LOCALAPPDATA`, `TMPDIR`, `PATH`, `USER`, `DISPLAY`,
`WAYLAND_DISPLAY`, `APPIMAGE`, `PROCESSOR_IDENTIFIER`, `SystemRoot`) affect
platform paths/launch behavior, not guest hardware; `TMPDIR` also places the
default CI socket and `--ci` overlays. Build-time metadata uses
`IRIS_GIT_REV`, `APP_VERSION`, `APP_COMMIT` and `CARGO_MANIFEST_DIR`; GUI premiere export
sets `IRIS_GUI_FEATURES`. `IRIS_SHADOW_CACHEOPS` is retired: use devlog `l2c`
instead. Shell capture/packaging scripts can also have their own job variables.

## Build examples and diagnostic feature details

The following material was moved from README.md. The inventory above covers
both crates; these examples and implementation notes explain the tradeoffs.

Build variants:
```
cargo run --release --features lightning,gpujit      # recommended for best speed
cargo run --release --features lightning             # disable emulator breakpoints for a little bit more speed
cargo run --release --features gpujit                # enable the graphics JITs (rex-jit + gr4-jit + gr2-jit)
cargo run --release --features jitv2,gpujit          # MIPS JIT v2 (experimental; see "JIT compilers")
cargo run --release --features idle-pause            # park the CPU thread while the guest idles instead of spinning a host core
cargo run --release --features ci_clock              # synthetic deterministic CP0 Count clock (CI/snapshot validator only; loses realtime desktop timing)
cargo run --release --features pcap                  # bridge guest networking onto a real host interface via libpcap instead of the built-in NAT gateway. See [network] in iris.toml.
cargo run -p iris-gui --release                      # the egui front-end, see iris-gui-README.md
```

`lightning` and `developer` are mutually exclusive, and `lightning` implies the
interpreter's `opcodefusion`. The emulator prints the features it was built with
at startup.

CHD images, the host camera (IndyCam source), DaynaPort, the Ultra64 dev board
and the IP28 / R10000 machine are always built in; nothing to enable. MIPS IV
follows the configured CPU (R4400 is MIPS III, R5000/R10000 are MIPS IV) in
both the interpreter and jitv2.

<details>
<summary>Diagnostic and experimental features</summary>

| Feature | What it does |
|---|---|
| `developer` | Undo buffer, execution trace, extra monitor commands; CPU starts paused. Use `--features developer`; `--profile developer` only selects build optimization/debug symbols. |
| `developer_ip7` | CP0 Compare / timer delivery stats and debug prints |
| `developerx` | Break into the monitor on IBE/DBE/ADEL/ADES/TLB errors |
| `rexdiag` | REX3 per-GO activity bits and dispatch counters. **On by default**; drop with `--no-default-features` to measure their cost |
| `vprodiag` | VPro texture packing check (`vpro status`, "texture packing"): every texture level the guest names is checked against the upload that wrote that memory. Off by default; works with `lightning` |
| `gputrace` | REX3 annotated trace (`rex trace`) and replayable bus recording (`rex rec`); GR2, IMPACT, O2 and VPro annotated traces (`gr2 trace`, `mgras trace`, `re trace`, `vpro trace`); O2 engine recording (`re rec`). On with `developer`; opt in for plain and `lightning` builds |
| `llstats` | Per-address LL/SC reservation histogram (`ll stats`); lightning-compatible |
| `perfetto` | Record Perfetto traces for ui.perfetto.dev: per-thread event rings, a writer thread, host or guest timeline. See [docs/perfetto.md](docs/perfetto.md) |
| `fetchverify` | Check every executed instruction word against memory (stale-code detector); lightning-compatible |
| `opcodefusion` | Interpreter branch+NOP, LUI+ORI/ADDIU and address-calc+load/store fusion. Breakpoints on a fused second instruction never fire |
| `tlbstats` / `tlbcheck` | TLB translation counters / full JTLB consistency check after every TLB write |
| `jitstats` | Counts how far each load/store gets through the JIT inline-memory checks |
| `instr_stats` | Per-opcode decode/execute counters (interpreter only; refused with `jitv2`) |
| `tcache` / `tcache_verify` | Transparent cache over the ppmem window ([docs/tcache-design.md](docs/tcache-design.md)) / its self-check. Always on with `jitv2`; optional for an interpreter build |
| `jitv2_lockstep` | Verify every JIT instruction against the interpreter (implies `developer`) |
| `jitv2_smc_check` | Report writes into the page the CPU is executing (run with `j2 inline_mem off`) |
| `jitv2_opcodefusion` | jitv2 LUI+ORI/ADDIU and branch+NOP fusion (off by default; see "JIT compilers") |
| `j2wp` | Compatibility alias for `jitv2`; whole-page compilation is the only implementation |
| `hostcall` / `hostgl` | Private host-service syscalls / host OpenGL for IRIX programs; `hostgl` has macOS (CGL) and Linux (EGL) backends |
| `debug_cache` | Track one cache line across all operations |
| `tlbvmap` | The vmap TLB fast path; on by default |
| `r5ksc`, `r5ksc_triton` | Refuse to build: no working R5000 secondary-cache model yet (`clanker/cpu/r5k-l1i-cache-bugs.md`) |

</details>

## Emulated CPU

The Indy and Indigo2 (IP22) run an R4400 (the default) or an R5000, the IP28 an
R10000, and the other machines R10000-family parts (R10000, R12000, R14000,
R16000) as `[machine] cpu` lists for each. The CPU is chosen at runtime: every
model is compiled into every binary, and there is no separate CPU build or
download. The table compares the three cores the Indy and Indigo2 run; the
R12000, R14000 and R16000 share the R10000's model with their own revisions and
secondary-cache sizes.

| | R4400 | R5000 | R10000 |
|---|---|---|---|
| L1 I/D | 16KB direct-mapped, 16B lines | 32KB 2-way, 32B lines | 32KB 2-way, I: 64B / D: 32B lines |
| Secondary cache | 1MB unified L2, 128B lines | none (Config `SC=1`) | 1MB, 128B lines; shadow arrays for CACHE diagnostics |
| ISA | MIPS III | MIPS IV | MIPS IV |
| PRId / FPU FIR | `0x00000440` / `0x00000500` | `0x00002310` (Indy and Indigo2, rev 1.0; the O2's rev 2.1 is `0x00002321`) / `0x00002300` | `0x00000925` (IP28) / `0x00000900` |

R10000 loads, stores, and fetches access memory directly; CACHE instructions
operate on shadow tag/data arrays for PROM diagnostics. This models functional
behaviour, not the physical CPU's timing or out-of-order execution. The R4400
and R5000 can use either cache model (`[machine] cache`): `full` keeps line
data as the hardware does, `shadow` keeps only the tags CACHE instructions read.

This is a different machine to the guest, not a speed knob: IRIX reads PRId and
configures itself from it, and an R4400 raises Reserved Instruction on the MIPS IV
opcodes an R5000 executes.

Which one is faster depends on the engine, so don't compare scores across CPUs.
On the bare-metal suite the R5000 is about 17% slower under the interpreter —
2-way associativity means probing both ways on every fetch, read and write — but
about 10% faster under jitv2, where the larger cache and 32-byte lines pay off
and the probe is not on the critical path.

Pick it in the GUI (Machine menu, or the General tab of the configuration area),
in a config file, or on the command line:

```
cargo run --release -- --cpu r5000        # or r4400, the default
```

```toml
[machine]
cpu = "r5000"
```

A snapshot records the CPU it was taken on and refuses to restore onto a
different CPU model, since the captured state assumes that machine.


## JIT compilers

### MIPS JIT v2 (`--features jitv2`) — experimental

A physical-page compiler built on Cranelift, with memory-resident
registers and no speculation, with generation checks at publication and
dispatch. Equivalence tests check agreement with the interpreter. Not the default
engine yet. Enabled automatically at runtime once the feature is compiled in.
See `clanker/jitv2/jit-v2-design.md` for the full design and `HACKING.md`'s
JIT v2 section for tuning. (The original speculative, tiered MIPS JIT was
removed in August 2026; jitv2 replaces it.)

```
cargo run --release --features jitv2,gpujit
```

What it does today: compiles on a pool of background threads (`[jitv2]
threads` / `--jitv2-threads`, default one per guest CPU up to 4, capped by
the host's usable CPUs), or on the CPU
thread for a repeatable run (`[jitv2] compile = "sync"` / `--jitv2-compile sync`,
one CPU only), inlines translated RAM accesses
through the direct memory window with shadow caches, retains L1 checks for
R4400/R5000 `full` caches, detects
self-modifying code through per-page generation counters, and falls back to the
interpreter for unsupported instructions. Each physical page has one function
with a dispatch switch for its entry points. Selected CP0 operations and LL/SC
stay in regions as calls to their interpreter handlers.

Extra features: `jitv2_lockstep` (cross-checks every compiled instruction
against the interpreter — slow, diagnostic only), `jitv2_smc_check`,
`jitv2_opcodefusion` (LUI+ORI/ADDIU and
branch/jump+NOP delay-slot fusion, jitv2's counterparts to the interpreter's
`opcodefusion` — OFF by default, unlike the interpreter's own fusion, due to a
history of live-boot bugs; see
`clanker/jitv2/jitv2_lui_fusion_foreign_delay_slot_hazard.md`). Developer tools:
`jitv2-analyze`, `jitv2-verify` and `jitv2-pcp-dump` binaries, and the `j2`
monitor command.

Compiled pages can be reused across runs with the optional persistent cache:

```toml
[jitv2]
threads = 1
cache = true
# cache_dir = "/path/to/cache"   # blank uses the per-user cache directory
```

In iris-gui, this is **General → Persistent JIT code cache**. It is off by
default and needs a build with `jitv2`. Cache entries are checked against the
executable, codegen settings, and page bytes. See
[Persistent JIT cache](docs/jitv2-persistent-cache.md).

#### Measuring emitted code against a real corpus

`j2 corpus [dir]` (default `jitv2_corpus/`) writes every page in the live JIT
page cache to a `.pcp` file — the same format `j2 dumppcp` produces for a
single page, carrying the page's bytes, its entry-point bitmaps, its
generation and its dispatch count. Boot the guest, do whatever workload you
care about, then take the dump; the pages are already cached, so this costs
nothing until you ask for it.

```
(monitor) j2 corpus                 # -> jitv2_corpus/pcp_<pfn>.pcp, one per page
IRIS_CORPUS_DIR=jitv2_corpus IRIS_OPT_SPEED=1   cargo test --release --features jitv2 zz_corpus_sizes -- --nocapture
```

That compiles every entry point of every captured page and reports total
emitted bytes, plus a dispatch-weighted total, so a codegen change can be
measured against real guest code instead of a microbenchmark.

**Do not measure under `developer`.** It forces `opt_level=none` and injects a
per-instruction trace callout, so its output describes code production never
emits — this has produced a completely wrong conclusion before, and the test
prints a warning if you try. See
`clanker/jitv2/block-fragmentation-blocks-cse.md`.

### REX3 drawing and the graphics JIT (`--features rex-jit`)

Every build draws through one generic REX3 draw routine that is specialised
ahead of time into 400+ native draw functions (`src/dev/ng1/rex3_shaders.rs`, generated
by `tools/gen_rex3_shaders.py` from a corpus of the DrawMode0/DrawMode1/clip
combinations the IRIX desktop actually uses). Most desktop drawing already runs
through one of those, with no JIT involved.

`rex-jit` adds a Cranelift JIT for combinations outside that corpus. It compiles
a specialised native "shader" per unique draw mode, inlining the entire draw
loop — coordinate stepping, clipping, shade DDA, pattern advance — into a single
function. Shaders compile in the background on first use and share the dispatch
table with the precompiled set; the profile of modes seen persists across
sessions (`~/.iris/rex-jit-profile.bin`) for instant warm-up on next boot. `rex jit status|list|on|off` in the
monitor inspects and controls it.

```
cargo run --release --features rex-jit
```

### Newport GPU-authoritative raster (experimental, `--features wgpu`)

With `[graphics] raster = "wgpu"`, Newport uses the production resident
backend rather than the CPU-canonical prototype below. One validated
`Arc<Gpu>` is shared by both Newport heads. Complete linear 2048 x 1024 RGB
and auxiliary planes are seeded once, then ordinary DRAW, SCR2SCR, HostW,
HostR ordering and scanout run through the bounded resident command stream.
Unsupported work and device loss are fatal to the machine; neither can switch
to CPU raster or claim success from stale CPU shadows.

The default direct mode requires the visible window to use
`renderer = "wgpu"` on that same device. Vulkan and D3D12 are the initial
validated production backends. Direct-mode screenshots, CI capture and VNC
map only requested composed output; ordinary presentation maps neither raw
planes nor composed output. If the required presenter cannot be created,
startup fails and the already-started machine is stopped and flushed.

Set `[graphics] newport_mode = "readback"` only when deliberately validating
through the ordinary GL/CPU compositor. Each newly composed frame observes
both full planes (16 MiB) and is attributed as compatibility traffic. The
frontend gives that explicit operation a two-second minimum timeout. Readback
mode is fixed at startup and is never an automatic fallback from direct mode.
An iris-gui build with its local `wgpu` feature exposes this readback path
through its CPU or optional GL capture renderer. iris-gui does not yet provide
a same-device eframe wgpu presenter, so it cannot host Newport direct mode.

`rex status` reports the selected mode, resolved backend and adapter,
`no_cpu_fallback=true`, aggregate command/batch/GFIFO/latency telemetry, and
all attributed raw and composed transfer bytes. Snapshots, restore, CI
rollback, reset, Stop/restart and dual-head construction retain GPU authority.
The embedded PROM command monitor and an installed IRIX 6.5 Indigo Magic
desktop have booted and captured frames on Vulkan and D3D12 with zero
unsupported operations. Showcase produced pixels identical to CPU raster on
both backends, and the animated Atlantis demo improved live REX3 throughput by
42.1% on Vulkan and 24.3% on D3D12 while using 5.9% and 7.7% less host CPU,
respectively, on the acceptance host. The mode remains experimental and
opt-in while that application/adapter matrix is broadened.

### Newport framebuffer presentation

`rex-dirty-upload` is an opt-in feature that tracks conservative 64x32 damage
tiles independently for Newport's RGB and auxiliary planes. The refresh thread
copies only dirty rectangles into its cached snapshot, and the GL compositor
uploads only those rectangles to the existing source textures. The CPU
framebuffer and the interpreter/precompiled/Cranelift raster engines remain
authoritative.

The feature is not in the default or release feature bundles. Without it, the
compiler omits coordinate classification and tile atomics from the REX3 draw
path and retains the original coarse dirty flag, full snapshot behavior, and
full-plane GL uploads.

```
cargo run --release --features rex-dirty-upload
cargo run -p iris-gui --release --features rex-dirty-upload
```

The first frame after GL texture creation uploads both complete 2048x1024
planes. Later palette-, cursor-, topscan-, and heartbeat-only redraws can
reuse both source textures without uploading framebuffer data. Set
`REX3_FORCE_FULL_UPLOAD=1` to restore the previous two-full-plane upload
behavior for comparison or fallback in a feature-enabled build.

With `--features perfetto`, the raster, snapshot copy, per-plane upload,
composition, and readback stages have independent slices and
byte/tile/rectangle counters; see
[docs/perfetto.md](docs/perfetto.md#newport-framebuffer-damage-and-presentation).

### IMPACT raster JIT (`--features gr4-jit`)

`gr4-jit` compiles IMPACT's raster pipelines (`src/dev/mgras/rss_jit`). The
registers that shape a primitive's pipeline reduce to a 64-bit key, and each
key gets its own monomorphised shader. The key covers the primitive (fills,
X lines, character stipple, transfer lines, GL triangles and GL lines),
clipping, draw buffers, pixel format, logic op, and the alpha, stencil and
depth tests, blending, texture environment, texture format, filtering and
wrap modes, and fog. Colours, masks, references, page pointers and plane
coefficients reach a shader as data in a per-board context.

Shaders compile on a background thread. Until a primitive's shader is ready
the interpreter draws it. Shaders are bit-exact with the interpreter:
`src/dev/mgras/rss_jit_tests.rs` sweeps the key space and compares whole
boards (`GR4_JIT_SWEEP=<factor>` and `GR4_JIT_SEED=<n>` widen it). Measured
on representative primitives, they are 2.5-6x faster than the interpreter:
fills and transfers 5-6x, lines and text about 3x, shaded and textured
triangles 2.5x.

- `IRIS_GR4_JIT=off|async|sync` sets the starting mode (`on` is `async`,
  the default). `sync` compiles on first use and waits.
- `mgras jit [on|off|sync|list]` in the monitor shows and changes it.
- `GR4_JIT_DISASM=1` prints each shader's machine code as it compiles.

### GR2 span JIT (`--features gr2-jit`)

`gr2-jit` compiles GR2's RE3 span pipelines (`src/dev/gr2/re3_jit`), on the
pattern of `gr4-jit`. On GR2 the HQ2/GE7 side walks primitives and RE3 draws
spans, so the unit is an RE3 instruction: SHADED (every iterator steps per
pixel, with depth, stencil and blending) and FLAT / FLAT4. The state that
shapes a span's pipeline reduces to a key: the instruction, frame-buffer
pixel format and dither, raster op, pattern mode, window-ID test, depth and
stencil tests and blend factors. Masks, scissor, colours and steps, stencil
references and pattern bits are read from the RE3 registers as the shader
starts. Row spans (DX = 1, DY = 0, what GL fills send) and FLAT spans are
clipped to the scissor once, not per pixel.

Shaders compile on a background thread; until a span's shader is ready the
interpreter draws it. They are bit-exact with the interpreter:
`src/dev/gr2/re3_jit_tests.rs` sweeps the keys and compares whole boards
(`GR2_JIT_SWEEP=<factor>`, `GR2_JIT_SEED=<n>`). Fills run about 3x faster
(a 24-bit quad fill went from 78 to 245 Mpx/s in IRIX, Z-buffered from 58
to 115), smooth shading 2-3x. Small triangles gain nothing: the HQ2 side
(edge walking and span setup) bounds them, not RE3.

- `IRIS_GR2_JIT=off|async|sync` sets the starting mode (`on` is `async`,
  the default). `sync` compiles on first use and waits.
- `gr2 jit [on|off|sync|list]` in the monitor shows and changes it.
- `GR2_JIT_DISASM=1` prints each shader's machine code as it compiles.

```
cargo run --release --features lightning,gpujit   # = lightning,rex-jit,gr4-jit,gr2-jit
```
