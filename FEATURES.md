# Build features and configuration reference

Complete feature inventory for **iris** and **iris-gui**. The tables list
every declared build feature, shared machine-config field, standalone CLI
switch, and persisted GUI setting. Diagnostic environment controls follow
the user-facing settings. Detailed feature explanations and build examples
are below the inventory.

## Build feature inventory

### iris (core and standalone CLI)

Source: [Cargo.toml](Cargo.toml). Default features are `tlbvmap,rexdiag`.
Select with `cargo build --release --features <comma-separated-list>`; add
`--no-default-features` to drop the core default bundle.

| Feature | Purpose | Enables |
|---|---|---|
| `debug_cache` | Track a selected cache line through cache operations. | — |
| `developer` | Undo/trace buffers and developer monitor tools; CPU starts paused. | `gputrace` |
| `gputrace` | REX3 bus taps: annotated trace (`rex trace`) and replayable recording (`rex rec`); see [clanker/rex3/trace-and-recording.md](clanker/rex3/trace-and-recording.md). GR2 and IMPACT annotated traces (`gr2 trace`, `mgras trace`). Off: no per-access check at all. Lightning-compatible. | — |
| `developer_ip7` | CP0 Compare/timer calibration statistics and debug prints. | — |
| `developerx` | Extended exception diagnostics; break on IBE/DBE/ADEL/ADES/TLB errors. | — |
| `llstats` | Per-address LL/SC reservation histogram (`ll stats`); lightning-compatible. | — |
| `perfetto` | Perfetto trace recording (`--perfetto FILE`, monitor `perfetto`); see [docs/perfetto.md](docs/perfetto.md). Lightning-compatible. | — |
| `rexdiag` | REX3 per-GO activity and dispatch counters. Default on; removable in a core build. | — |
| `fetchverify` | Compare executed instruction bytes with memory in interpreter/JIT; expensive stale-code detector. | — |
| `ci_clock` | Synthetic deterministic Count clock for CI/snapshot validation; replaces real-time Count timing. | — |
| `opcodefusion` | Interpreter instruction fusion; breakpoints on a fused second instruction cannot fire. | — |
| `lightning` | Strip breakpoints/traceback for speed; hides GUI Debug controls. | `opcodefusion` |
| `tlbvmap` | Compatibility flag: vmap translation is unconditional. | — |
| `tlbstats` | TLB path/entries-walked counters, printed at exit. | — |
| `jitstats` | Measure eligibility for JIT inline memory accesses. | — |
| `tlbcheck` | Full JTLB consistency checks after TLB writes; diagnostic overhead. | — |
| `instr_stats` | Per-opcode decode/execute counters; interpreter only, refused with JIT v2. | — |
| `r5ksc` | Disabled external secondary-cache model; deliberately refuses to compile. | — |
| `r5ksc_triton` | Disabled O2/Triton integrated L2 model; deliberately refuses to compile. | `r5ksc` |
| `idle-pause` | Park the CPU in recognized guest idle loops; opt-in. | — |
| `tcache` | Keep cache tags/state but use authoritative ppmem RAM for cacheable data. | — |
| `tcache_verify` | Assert transparent-cache/RAM coherence on accesses; very slow. | `tcache` |
| `default` | Default bundle: `tlbvmap` and `rexdiag`. | `tlbvmap`, `rexdiag` |
| `rex-jit` | Cranelift compiler for Newport draw shaders; complements precompiled shaders. | `cranelift-codegen`, `cranelift-frontend`, `cranelift-jit`, `cranelift-module`, `cranelift-native` |
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

Cargo also exposes these **implicit optional-dependency features**:
`cranelift-codegen`, `cranelift-frontend`, `cranelift-jit`, `cranelift-module`, `cranelift-native`, `region`, `target-lexicon`, `wasmtime-jit-icache-coherence`. They enable library dependencies,
not complete emulator capabilities; use `jitv2` or `gpujit` for a working
compiler configuration. `dep:` entries above suppress implicit feature names
for PCAP/host-service dependencies.

### iris-gui

Source: [iris-gui/Cargo.toml](iris-gui/Cargo.toml). There is no GUI-specific
default feature bundle. Its `iris` dependency always enables `gpujit` (the
App Store build turns the JITs off at run time with `IRIS_NO_JIT`) and retains the core default features. GUI `--no-default-features` alone does not
turn off those dependency defaults.

| GUI feature | Purpose | Enables |
|---|---|---|
| `bundled` | Distribution UI: hides source-checkout tools; machine TOML storage is shared by all builds. | — |
| `appstore` | Mac App Store UI, sandbox bookmarks, private-API patch, CHD diff redirection, and REX3 JIT disabled at startup; hides CI/Ultra64 controls. | `bundled` |
| `pcap` | Expose PCAP NIC selection and bridge support in the embedded core. | `iris/pcap` |
| `premiere` | Performance bundle for the embedded core; lightning plus idle parking. | `iris/lightning`, `iris/idle-pause` |
| `hostcall` | Enable private guest syscall services in the embedded core. | `iris/hostcall` |
| `hostgl` | Enable hostcall and macOS host OpenGL in the embedded core. | `hostcall`, `iris/hostgl` |

Every core feature above is also selectable for the GUI via `iris/<feature>`:

```sh
cargo build -p iris-gui --release --features iris/jitv2
cargo build -p iris-gui --release --features premiere,hostgl
```

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
the frontend, and `--workspace` also includes `iris-hostcall`. `iris-hostgl`
is excluded from the workspace and built through the optional dependency.
Select Cargo's target triple with `--target <triple>`; platform backends and
packaging still impose the host restrictions described above.

| Profile | Selection | Settings |
|---|---|---|
| `dev` | `cargo build` / `cargo run` | Cargo development defaults: unoptimized, with debug information. |
| `release` | `--release` | Optimization level 3, fat LTO, one codegen unit, abort on panic, debug level 1. |
| `developer` | `--profile developer` | Inherits release optimization; full debug information. Enable `--features developer` separately for instrumentation. |
| `profiling` | `--profile profiling` | Inherits release optimization; full debug information for profiling. |

[iris-gui/build.rs](iris-gui/build.rs) reads `RELEASE_VERSION` at build time
and exports `APP_VERSION` to the frontend. Without an override it uses
`CARGO_PKG_VERSION`, adding `-dev` for a debug-profile build. `PROFILE` and
`CARGO_PKG_VERSION` are supplied by Cargo. These build-time controls are
separate from the runtime environment inventory below.

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
| `prom` | `"prom.bin"` | PROM path; IP24, IP22, and IP28 each have an embedded fallback. Indigo2 also tries `070-1367-012.bin` (IP22) or `070-1477-002.bin` (IP28) in the working directory; the O2, which has no embedded PROM, tries `ip32prom.rev4.18.bin`. |
| `nvram` | `"nvram.bin"` | DS1386 backing file on the Indy; the PROM's environment on the O2 (its flash) and the Octane. GUI defaults to an absolute user-config path. |
| `nveeprom` | `"nveeprom.bin"` | Motherboard EEPROM file; GUI anchors it in its user-config directory. |
| `banks` | `[128,128,0,0]` | RAM bank sizes in MB, up to eight entries (every machine today has four; the rest must be 0). On the O2 each entry is a DIMM pair's kit (32, 64, 128 or 256), largest first. On the Octane each entry is 64, 128, 256 or 512, filled from bank 0, 2048 in all. See profile constraints below. |
| `scale` | `1` | Standalone window scale: 1–4. GUI uses its separate vm_scale. |
| `headless` | `false` | Disable the standalone window and graphics; the O2 then boots on its serial console. Audio is independent. |
| `no_audio` | `false` | Disable audio emulation (HAL2, MACE or RAD). |
| `gdb_port` | `unset` | Start the GDB TCP stub; debugging is unavailable in lightning builds. |
| `monitor_port` | `8888 when unset` | Monitor listener on loopback. |
| `serial_port_a` | `8880 when unset` | SCC channel A listener (tty2); the O2's first serial port, its console when headless. |
| `serial_port_b` | `8881 when unset` | SCC channel B listener (tty1/console); the O2's second serial port. GUI halt uses this port. |
| `load_elf` | `unset` | Load a static big-endian MIPS ELF32 binary before starting the CPU. |
| `test_device` | `false` | Map the bare-metal test device in GIO slot 0. |
| `test_device_dump` | `iris-testdev-dump.json when unset` | Test-device machine-state JSON output. |
| `cheritest_dump_hook` | `false` | CP0 register 26 dump convention; needs test_device, unsafe for normal PROM boot. |
| `nat_subnet` | `192.168.0.0/24 when unset` | CIDR; NAT gateway network+1 and guest network+2. |
| `ci` | `false` | CI server/serial backend; hides host window unless ci_display, keeps offscreen graphics. |
| `ci_socket` | `/tmp/iris.sock on Unix; 127.0.0.1:19851 on Windows` | Unix path, host:port, or tcp:host:port. |
| `ci_display` | `false` | Keep host graphics window visible in CI mode. |
| `mouse_scroll_pixels_per_line` | `40.0` | Host scrolling distance per guest PS/2 detent. |
| `lock_aspect_ratio` | `true` | Constrain standalone window aspect ratio; false allows letterboxing. |
| `serial_log` | `unset` | Append guest channel-B output (the O2's first serial port); used by CI and GUI serial capture. |
| `serial_log_timestamps` | `off` | `off` (raw transcript) or `wall`: prefix each completed serial-log line with a UTC ISO-8601 stamp. A partial line is held until its newline and written on stop. |
| `scsi_deferred_int` | `true` | Defer SCSI status interrupts for BSD compatibility. |

### [machine]

| Key | Default | Values and behavior |
|---|---|---|
| `machine.profile` | `indy_ip24` | indy_ip24, indigo2_ip22, indigo2_ip28, o2_ip32 (configured by `[o2]`, see [the O2 guide](docs/o2-ip32.md)), octane_ip30 (configured by `[octane]`, see [the Octane guide](docs/octane-ip30.md)), fuel_ip34 (see [the Fuel guide](docs/fuel-ip34.md)). |
| `machine.cpu` | `r4400` | r4400, r5000, r10000. CPU/ISA choice is independent of build features. A Fuel always runs its one R14000, and the key must be `r10000`; any other value is refused. |
| `machine.cpus` | `1` | Processors fitted. 2 only on the Octane; see [its guide](docs/Octane/multiprocessor.md). |

### [fuel]

Fuel-specific settings apply when `machine.profile = "fuel_ip34"`.

| Key | Default | Values and behavior |
|---|---|---|
| `flash` | unset | Persistent 16 MiB boot-flash file. Unset, the flash is volatile. |
| `console` | `l1` | `l1`, `ioc3a` or `ioc3b`. Selects the guest console connection. |
| `l1_port` | `8882` | Host TCP port for the L1 console. Used when `console = "l1"`. |
| `l1_log` | unset | Optional serial log path for L1 output when an IOC3 port is selected as console. |

The machine uses one 500 MHz R14000 and fixed 512 MiB RAM in two 256 MiB
banks. Its IOC3 has no keyboard or mouse ports, so it runs on a serial
console. A `[fuel]` table is written to a saved config only when one of
these keys is set. The sample configuration and supported PROM inputs are in the
[Fuel guide](docs/fuel-ip34.md).

### [graphics]

| Key | Default | Values and behavior |
|---|---|---|
| `graphics.board` | `newport`; `solidimpact` on the Octane | newport, xz, extreme, solidimpact, highimpact, maximpact, none. Extreme is Indigo2 only. The Octane takes only `solidimpact` (SI), `highimpact` (SI with texture option), `maximpact` (MXI) or `none` (no board: the machine runs on its serial console, with no window). |
| `graphics.heads` | `1` | 1 or 2; dual-head requires Newport. |
| `graphics.resolution` | `guest` | guest, 1024x768, 1280x960, 1280x1024; presets require Newport. |

### [scsi.<ID>]

| Key | Default | Values and behavior |
|---|---|---|
| `scsi.<ID>.path` | `""` | Disk/CHD/ISO path; empty CD-ROM means an empty tray. |
| `scsi.<ID>.discs` | `[]` | Additional CD changer images. |
| `scsi.<ID>.cdrom` | `false` | Legacy spelling for kind=cdrom. |
| `scsi.<ID>.kind` | `disk` | disk, cdrom, daynaport; serialized key is kind. |
| `scsi.<ID>.mac` | `00:80:19:44:50:<SCSI ID> when unset` | DaynaPort-only station MAC. |
| `scsi.<ID>.subnet` | `192.168.10.0/24 when unset` | DaynaPort NAT network; separate from ec0. |
| `scsi.<ID>.overlay` | `false` | Raw-image COW sidecar; CHD uses its own diff sidecar. |
| `scsi.<ID>.scratch` | `false` | Host-managed raw scratch volume; cannot be CD-ROM/overlay. |
| `scsi.<ID>.size_mb` | `64 when unset` | Size of a newly created scratch image; existing files are unchanged. |
| `scsi.<ID>.controller` | `0` | 0 or 1; controller 1 is accepted on indigo2_ip22 and o2_ip32 (on the O2 it is the external bus). A key of 16 + ID names controller 1 directly and does not need it. |

IDs are 1–7. The key is a slot: `[scsi.N]` with N 1–7 is id N on controller 0
(or on `controller`, when set), and N = 16 + id is that id on controller 1,
so both controllers can carry the same id; two tables naming one target are
rejected. If `scsi` is entirely omitted, core defaults attach SCSI #1 as
`scsi1.raw` and SCSI #4 as `cdrom4.iso`. An explicit map replaces that
default map; missing slots are unattached. `kind = "daynaport"` needs no image.
An explicit non-disk kind takes precedence over the legacy cdrom flag.

### [network]

| Key | Default | Values and behavior |
|---|---|---|
| `network.mode` | `nat` | nat, pcap or off (no Ethernet on O2, Octane and Fuel). PCAP needs build support and host capture permissions. It works on every machine with Ethernet. |
| `network.pcap_interface` | `auto-pick when unset` | Interface name or numeric list index; PCAP only. |
| `network.nfs_pcap_ip` | `unset` | Virtual LAN IP for the built-in NFS responder in PCAP mode. |
| `network.mac` | `08:00:69:12:34:56 when unset` | ec0 MAC; injected only into blank PROM battery-backed MAC slots. |
| `network.tftp_dir` | `unset` | Read-only TFTP root at the NAT gateway for PROM network boot. |
| `network.rsh_dir` | `unset` | Root for the built-in rsh server at the NAT gateway (TCP 514): `rsh <gateway> cat\|dd\|ls\|echo\|pwd`, read-only, nothing runs on the host. |

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
| `vnc.listen` | `unset` (no server) | `host:port`, or a port on 127.0.0.1, to serve the screen, keyboard and mouse over VNC (RFB 3.8; Raw and ZRLE; no password). Every board, with a window or `ci`; not `headless`. Standalone `iris` only. See [HELP.md](HELP.md#vnc). |
| `vnc.max_fps` | `30` | Most updates a second sent to a viewer (1–120). |

### [audio]

| Key | Default | Values and behavior |
|---|---|---|
| `audio.host` | `"default"` | Host audio system: `"default"` (CoreAudio on macOS, WASAPI on Windows, ALSA on Linux), `"alsa"`, or `"pulseaudio"` (Linux: a PulseAudio server or PipeWire's pipewire-pulse, through cpal's own client; ALSA if none answers). `--audio-host` overrides. |
| `audio.prebuf_ms` | `20` | HAL2 prebuffer duration in milliseconds. Not used yet: HAL2 prebuffers 20 ms. |
| `audio.cpal_buffer_frames` | `host default when unset` | Host buffer size in frames (a period; PulseAudio buffers two). Unset: the host's default, except PulseAudio's (about 2 s), which gets 512. A size the device refuses falls back to the default. |

### [jitv2]

| Key | Default | Values and behavior |
|---|---|---|
| `jitv2.threads` | `1` | Fixed compile-pool size at startup; must be at least 1 (0 is rejected). |
| `jitv2.cache` | `false` | Opt-in compiled-page disk cache; needs jitv2. |
| `jitv2.cache_dir` | `""` | Blank selects platform user cache directory plus iris/jitv2. |

### [debug]

| Key | Default | Values and behavior |
|---|---|---|
| `debug.gui_gl_capture` | `false` | GUI GL compositor capture path (IRIS_GUI_GL). |
| `debug.no_idle` | `false` | Suppress idle parking (IRIS_NO_IDLE); useful with idle-pause builds. |
| `debug.debug_log` | `""` | Device log specification (IRIS_DEBUG_LOG); developer instrumentation required. |

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
| `clock.fixed_mhz` | `33 on IP22/IP24; 97.5 on IP28; 90 on the O2 when unset` | CP0 Count MHz; IRIX's CPU inventory and the O2's PROM report twice this rate. |

### [rtc_offset]

| Key | Default | Values and behavior |
|---|---|---|
| `rtc_offset.years` | `0` | Signed years offset from host UTC when seeding RTC. |
| `rtc_offset.months` | `0` | Signed months offset from host UTC when seeding RTC. |
| `rtc_offset.days` | `0` | Signed days offset from host UTC when seeding RTC. |
| `rtc_offset.hours` | `0` | Signed hours offset from host UTC when seeding RTC. |
| `rtc_offset.minutes` | `0` | Signed minutes offset from host UTC when seeding RTC. |
| `rtc_offset.seconds` | `0` | Signed seconds offset from host UTC when seeding RTC. |

### [ultra64]

| Key | Default | Values and behavior |
|---|---|---|
| `ultra64.enabled` | `false` | Enable the GIO N64 development board; requires an external gopher64 peer. |

### [o2]

Used with `machine.profile = "o2_ip32"`. An O2 is configured by the shared
keys: `prom` (its PROM image, not embedded; `ip32prom.rev4.18.bin` is tried
when `prom` is left at its default), `nvram` (the PROM's flash, which holds
its environment), `banks` (the kit in each of the four DIMM pairs: 32, 64,
128 or 256 MB, largest first), `serial_port_a`/`serial_port_b`/`serial_log`,
`headless` (serial console instead of the graphics console and keyboard),
`[clock] fixed_mhz` (CP0 Count; the CPU runs at twice it, so 90 MHz is a
180 MHz O2), `[network]` including `mac` and `mode = "off"` for an unplugged
Ethernet, `[[port_forward]]`, `nat_subnet`, `[nfs]` and `[scsi.N]`
(`controller` is the bus). See [the O2 guide](docs/o2-ip32.md). The old
O2-only spellings (`o2.prom`, `o2.flash`, `o2.ram_mb`, `o2.dimms`,
`o2.disks`, `o2.serial_port`, `o2.serial_log`, `o2.graphics`, `o2.cpu_mhz`,
`o2.net`, `o2.forwards`) are deprecated: they are folded into those keys when
the file is loaded, with a note, and never written back.

| Key | Default | Values and behavior |
|---|---|---|
| `o2.mouse` | `true` | PS/2 mouse on the second port (with the graphics console). |
| `o2.control_port` | unset | TCP port on 127.0.0.1 serving the `o2` monitor commands for scripts. |
| `o2.arcs_fpu` | `false` | Test aid: lend a kernel the FPU while it runs inside the PROM. |

### [octane]

Used with `machine.profile = "octane_ip30"`, which needs `machine.cpu` of
`r10000`, `r12000` or `r14000`. An Octane is configured by the shared keys: `prom` (the IP30 PROM,
not embedded; `IP30prom.bin` is tried when it is left unset), `nvram` (the
DS1685's NVRAM, behind IOC3), `banks` (the MB in each HEART bank: 64, 128, 256
or 512, filled from bank 0 with no gap, 2048 in all), `serial_port_a` and
`serial_port_b`/`serial_log` (IOC3's two ports, the first the console),
`headless`, `[clock] fixed_mhz` (CP0 Count; 97.5 for the 195 MHz R10000),
`[network]`, `[[port_forward]]`, `nat_subnet`, `[scsi.N]` (`controller` 0
is the internal bus, 1 the external) and `[graphics] board`. See [the Octane guide](docs/octane-ip30.md).

| Key | Default | Values and behavior |
|---|---|---|
| `octane.flash` | unset | Boot flash file, keeping the PROM's records between runs. Created from `prom` when missing and then used as it is, so delete it after changing `prom`; a file of another size is refused. Unset, the flash is in memory and every start is from `prom`. |

### Hardware constraints and precedence

On the Indy and Indigo2 bank values are 0, 8, 16, 32, 64, or 128 MB, and only
IP28 accepts 256/512 MB. On the O2 a bank is a DIMM pair's kit, 32, 64, 128 or
256 MB, fitted from pair A largest first; on the Octane it is a matched DIMM
pair, 64, 128, 256 or 512 MB, filled from bank 0. `banks` has eight entries;
every machine today uses the first four and the rest must be 0.
IP22/IP24 four-bank IRIX 6.5 behavior remains separate from IP28's verified
two-bank 1 GB layout. Extreme requires IP22. All non-Newport boards require
`heads=1` and `resolution="guest"`. IP28 IRIX needs IMPACT; its PROM has an
embedded fallback.
The config validator permits SCSI controller 1 only on IP22, the O2, Octane
and Fuel. These machines have two controllers.

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
| `--ip22` | Emulate an SGI Indigo2 (IP22) instead of the default Indy (IP24). Overrides `[machine].profile` in the config file. |
| `--nvram` | Path to NVRAM file (default: nvram.bin in cwd) |
| `--nveeprom` | Path to Indigo2 motherboard EEPROM file (default: nveeprom.bin in cwd) |
| `--bank0` | RAM bank 0 size in MB (0/8/16/32/64/128; IP28 also supports 256/512) |
| `--bank1` | RAM bank 1 size in MB (0/8/16/32/64/128; IP28 also supports 256/512) |
| `--bank2` | RAM bank 2 size in MB (0/8/16/32/64/128; IP28 also supports 256/512) |
| `--bank3` | RAM bank 3 size in MB (0/8/16/32/64/128; IP28 also supports 256/512) |
| `--jitv2-threads` | jitv2 compile-pool thread count (fixed at startup, see jitv2.threads) |
| `--scsi1` | SCSI ID 1 image path (HDD) |
| `--scsi2` | SCSI ID 2 image path (HDD) |
| `--scsi3` | SCSI ID 3 image path (HDD) |
| `--cdrom4` | SCSI ID 4 image path (CD-ROM, primary disc) |
| `--cdrom5` | SCSI ID 5 image path (CD-ROM, primary disc) |
| `--cdrom6` | SCSI ID 6 image path (CD-ROM, primary disc) |
| `--scsi7` | SCSI ID 7 image path (HDD) |
| `--cdrom4-extra` | Additional ISO images for CD-ROM ID 4 (can be specified multiple times) |
| `--cdrom5-extra` | Additional ISO images for CD-ROM ID 5 (can be specified multiple times) |
| `--cdrom6-extra` | Additional ISO images for CD-ROM ID 6 (can be specified multiple times) |
| `--2x` | 2× window scaling for HiDPI/4K monitors |
| `--headless` | Run headless: no window, no REX3 graphics (audio unaffected; use --noaudio to disable) |
| `--noaudio` | Disable audio emulation (no HAL2); graphics still works |
| `--audio-host <HOST>` | Host audio system for HAL2: `default`, `alsa` or `pulseaudio` (overrides `[audio] host`) |
| `--nfs-dir` | Enable NFS share: path to the directory to export (enables NFS) |
| `--nat-subnet` | NAT subnet in CIDR notation (e.g. 192.168.5.0/24). Gateway gets .1, guest (IRIX) gets .2. Default: 192.168.0.0/24. |
| `--net-mode` | Networking backend: "nat" (default, software gateway) or "pcap" (bridge onto a real host interface; requires --features pcap). |
| `--pcap-interface` | Host interface to bridge onto in PCAP mode (e.g. eth0, en0). Implies --net-mode pcap. List candidates with --list-net-interfaces. |
| `--list-net-interfaces` | Print the host network interfaces libpcap can bridge onto, then exit. Requires a build with --features pcap. |
| `--no-scsi-deferred-int` | Disable deferred SCSI status interrupts (default: enabled for OpenBSD/NetBSD compatibility). |
| `--cpu` | Emulated CPU: `r4400` (default), `r5000`, or `r10000`. Overrides `[machine] cpu`.  |
| `--graphics` | Graphics board: `newport` (default), `xz`, `extreme`, `solidimpact`, `highimpact`, `maximpact` (or `impact:solid`, `impact:high`, `impact:max`). |
| `--gdb-port` | Enable GDB stub on the given TCP port (e.g. --gdb-port 1234). Connect with: target remote localhost:<port> |
| `--monitor-port` | Monitor console port on 127.0.0.1 (default 8888). Give each iris running at once its own. |
| `--vnc [ADDR]` | Serve the screen, keyboard and mouse over VNC on ADDR (`host:port` or a port; default 127.0.0.1:5900). Sets `vnc.listen`. |
| `--serial-port-a` | Serial channel A (tty2) port on 127.0.0.1 (default 8880). |
| `--serial-port-b` | Serial channel B (tty1, the console) port on 127.0.0.1 (default 8881). |
| `--test-device` | Map the bare-metal test device into GIO expansion slot 0: SIGNATURE, PUTC (guest console → stdout), DUMP (machine state → JSON) and EXIT (terminate with the guest's exit code). Off by default. |
| `--test-device-dump` | Where the test device writes its machine-state dump. |
| `--cheritest-dump-hook` | cheritest convention: a guest write to CP0 register 26 dumps machine state. Needs --test-device. Never enable for a normal boot — CP0 26 is ECC on a real R4400 and the PROM writes it during cache init. |
| `--tftp-dir` | Serve this directory read-only over TFTP at the gateway address, so the PROM can network-boot from it: `boot -f bootp()<file>`. Off when unset. |
| `--rsh-dir` | Let the guest `rsh <gateway> cat\|dd\|ls\|echo\|pwd` against this directory, read-only, through a built-in rsh server at the gateway (TCP 514). Nothing is executed on the host. Off when unset. |
| `--perfetto FILE` | Record a Perfetto trace from startup (`perfetto` builds). `--perfetto-cats LIST` (default `all`) and `--perfetto-clock host\|guest` (default `host`) select what and on which timeline. See [docs/perfetto.md](docs/perfetto.md). |
| `--load-elf` | Load a static ELF32 MSB (big-endian MIPS) binary into RAM before the CPU starts and set PC to its entry point, instead of booting the PROM. For bare-metal test binaries; see also the monitor's `loadelf`. |
| `--ci` | CI mode: enable the control socket and apply speed-favoring fidelity shortcuts. Hides the window unless --ci-display is set; offscreen graphics remain active unless --headless is explicitly set. |
| `--ci-socket` | Override the default control-socket path (/tmp/iris.sock). |
| `--ci-display` | With --ci, keep the Newport window visible for interactive test development (deferred rendering at 10–15 fps). |
| `--serial-log` | With --ci, append every byte the guest emits on ttyd1 (IRIX serial console) to this file. Useful for live tailing during an install. |
| `--clock-fixed-mhz` | Override CP0 Count MHz (default 33 on IP22/IP24, 97.5 on IP28). IRIX reports twice this rate as CPU MHz, e.g. --clock-fixed-mhz 50. |
| `--help`, `-h` | Display parser help. |

`--graphics` has visible aliases `--board` and `--graphics-board`.
Board aliases include `xl`, `gr2`, `gr2_xz`, `gr2-xz`, `gr2_extreme`,
`gr2-extreme`, `impact:solid`, `solid_impact`, `impact:high`, `high_impact`,
`impact:max`, and `max_impact`. TOML board parsing also accepts hyphenated
IMPACT names and `solid`, `high`, `max`; canonical values are in the table.
See [debug.md](debug.md) for monitor commands and [STORAGE.md](STORAGE.md) for iris-ci; they are runtime tools,
not Cargo build features.

## iris-gui persisted settings inventory

Source: [iris-gui/src/settings.rs](iris-gui/src/settings.rs). These keys belong
to GUI JSON rather than machine TOML files. View controls update the scales;
the GUI saves machine TOMLs and sandbox access state for you.

| GUI JSON key | Normal load default | Purpose |
|---|---|---|
| `ui_scale` | `1.25` | egui controls scale; sanitized to supported range. |
| `vm_scale` | `0.75` | Maximum framebuffer magnification, independent of UI scale. |
| `active_machine` | unset | Selected machine folder name. |
| `recent_configs` | empty list | Legacy recent TOML paths; retained for compatibility. |
| `last_config` | unset | Legacy TOML migration source; cleared after conversion. |
| `bookmarks` | empty map | macOS sandbox security-scoped bookmarks by absolute path. |
| `disk_folders` | empty list | Granted recursive sandbox disk-folder access. |

eframe also persists its own viewport/window state. These framework preferences
are separate from the shared machine schema. First-launch fallback window size
is 1512×1024 logical points; monitor fitting and running framebuffer size refine
it. GUI scale defaults refer to the normal load/sanitization path.

## Environment and diagnostic controls

These controls are process-wide and many are read once. Set them before launch;
monitor toggles are preferable when available. Except for the documented host
QoS behavior and Windows crash handler, opt-in diagnostic controls are inactive
when unset; JIT/inline-memory dispatch is normally enabled when built. Test-only
and offline-tool controls are included so they are not mistaken for GUI settings.

| Variable | Effect / accepted value | Reader |
|---|---|---|
| `GR2_REPLAY` | Replay input path for opt-in GR2 tests. | [src/dev/gr2/hq2_tests.rs](src/dev/gr2/hq2_tests.rs) |
| `GR2_REPLAY_OUT` | GR2 replay dump prefix (default gr2replay). | [src/dev/gr2/hq2_tests.rs](src/dev/gr2/hq2_tests.rs) |
| `IRIS_BAKE_HOOKS` | Any presence bakes callout hook addresses into JIT code (experimental). | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_BREAK_CPU` | 1 stops CPU after exception handling for diagnostic inspection. | [src/cpu/mips_exec.rs](src/cpu/mips_exec.rs) |
| `IRIS_CHD_DIFF_DIR` | Redirect CHD diff sidecars; GUI App Store sets a writable container directory. | [src/chd_disk.rs](src/chd_disk.rs) |
| `IRIS_CONST_DUMP` | Presence prints selected disassembly lines in constant-dedup tests. | [src/cpu/jitv2/mod.rs](src/cpu/jitv2/mod.rs) |
| `IRIS_CONST_MODE` | Select the opt-in constant-dedup codegen test shape. | [src/cpu/jitv2/mod.rs](src/cpu/jitv2/mod.rs) |
| `IRIS_CORPUS_DIR` | Corpus directory consumed by opt-in JIT corpus tests. | [src/cpu/jitv2/mod.rs](src/cpu/jitv2/mod.rs) |
| `IRIS_CRASH_DIAG` | off disables Windows crash handler installation. | [src/crash_diag.rs](src/crash_diag.rs) |
| `IRIS_CRASH_SELFTEST` | Standalone crash diagnostic self-test selector; deliberately crashes. | [src/main.rs](src/main.rs) |
| `IRIS_DEBUG_LOG` | Device log specification; applied to developer devlog after machine creation. | [src/main.rs](src/main.rs) |
| `IRIS_ENTRY_PREAMBLE` | 1 forces entry-word preamble in the opt-in codegen test. | [src/cpu/jitv2/mod.rs](src/cpu/jitv2/mod.rs) |
| `IRIS_GUI_GL` | 1 selects the GUI GL capture path. | [iris-gui/src/framebuffer.rs](iris-gui/src/framebuffer.rs) |
| `IRIS_HAL2_CAPTURE` | HAL2 host audio capture output path. | [src/dev/hal2.rs](src/dev/hal2.rs) |
| `IRIS_HOSTGL_CALLS` | Substring filter for HostGL call logging (bounded log). | [iris-hostgl/src/service.rs](iris-hostgl/src/service.rs) |
| `IRIS_HOSTGL_INLINE` | 1 runs the HostGL service on the CPU thread instead of its own (for comparison; the CPU stalls while a buffer replays). | [iris-hostgl/src/lib.rs](iris-hostgl/src/lib.rs) |
| `IRIS_HOSTGL_NOSURFACE` | 1 disables CGL IOSurface presentation. | [iris-hostgl/src/backend/cgl.rs](iris-hostgl/src/backend/cgl.rs) |
| `IRIS_HOSTGL_QOS` | default disables HostGL thread QoS adjustment. | [iris-hostgl/src/qos.rs](iris-hostgl/src/qos.rs) |
| `IRIS_HOSTGL_STRINGS` | host passes through host GL strings instead of compatibility strings. | [iris-hostgl/src/service.rs](iris-hostgl/src/service.rs) |
| `IRIS_INTRUN` | Integer interrupt-check coalescing budget; also used by offline tools. | [src/bin/jitv2_pcp_dump.rs](src/bin/jitv2_pcp_dump.rs) |
| `IRIS_IP28_CACHEOPS` | Presence traces cache ops in the legacy R10K cache model; all broadens logging. | [src/cpu/mips_cache_v2.rs](src/cpu/mips_cache_v2.rs) |
| `IRIS_IP28_EXC_VADDR` | IP28 exception virtual-address trace filter. | [src/cpu/mips_exec.rs](src/cpu/mips_exec.rs) |
| `IRIS_IP28_TLBW` | wired enables extra IP28 Wired/TLB-write tracing. | [src/cpu/mips_exec.rs](src/cpu/mips_exec.rs) |
| `IRIS_IP28_WATCH` | IP28 physical-address/window access trace filter. | [src/cpu/mips_cache_v2.rs](src/cpu/mips_cache_v2.rs) |
| `IRIS_IP28_WATCHGPR` | Guest GPR index whose writes are traced in IP28 bring-up. | [src/cpu/mips_core.rs](src/cpu/mips_core.rs) |
| `IRIS_IP32_HOT` | Any presence counts O2 register accesses (MACE, GBE, rendering engine) for finding a driver polling a bit nothing sets; costs speed. | [src/ip32.rs](src/ip32.rs) |
| `IRIS_IP32_MEC_TRACE` | Any presence traces O2 Ethernet frames (sent, received, dropped) on stderr. | [src/dev/mec.rs](src/dev/mec.rs) |
| `IRIS_IP32_RE_TRACE` | Number of O2 rendering-engine primitives to record for `o2 retrace PATH`. | [src/dev/o2gfx.rs](src/dev/o2gfx.rs) |
| `IRIS_JIT_CACHE` | 1 or on enables persistent JIT page reuse. | [src/cpu/jitv2/pcache.rs](src/cpu/jitv2/pcache.rs) |
| `IRIS_JIT_CACHE_DIR` | Persistent JIT cache base directory. | [src/cpu/jitv2/pcache.rs](src/cpu/jitv2/pcache.rs) |
| `IRIS_JIT_CLIF` | Any presence prints Cranelift input IR. | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_JIT_DISASM` | Any presence requests generated machine-code disassembly. | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_JIT_DISPATCH` | off or 0 disables normal JIT v2 dispatch at startup. | [src/cpu/mips_exec.rs](src/cpu/mips_exec.rs) |
| `IRIS_JIT_HASHSTATS` | 1 enables page/entry reuse diagnostics. | [src/cpu/jitv2/hashstats.rs](src/cpu/jitv2/hashstats.rs) |
| `IRIS_JIT_HASHSTATS_LOG` | Append per-compile reuse diagnostic records to this file. | [src/cpu/jitv2/hashstats.rs](src/cpu/jitv2/hashstats.rs) |
| `IRIS_JIT_ARENA_FLUSH_MB` | Flush compiled code when the arena holds this many MiB, for testing the compile workers' flush. | [src/cpu/jitv2/jitv2.rs](src/cpu/jitv2/jitv2.rs) |
| `IRIS_JIT_ATOMIC_GEN` | Any presence makes inline stores bump the page generation atomically with one CPU too, as they always do with several. | [src/cpu/mips_exec.rs](src/cpu/mips_exec.rs) |
| `IRIS_JIT_PIC` | Any presence disables core-address constants for JIT code. | [src/cpu/mips_exec.rs](src/cpu/mips_exec.rs) |
| `IRIS_MEM_HELPERS` | Any presence enables the experimental JIT memory-helper path. | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_MGRAS_REC` | Record IMPACT bus operations from power-on to this path. | [src/dev/mgras/mod.rs](src/dev/mgras/mod.rs) |
| `IRIS_MGRAS_TRACE` | Trace IMPACT accesses from power-on to this path (`gputrace` builds). | [src/dev/mgras/mod.rs](src/dev/mgras/mod.rs) |
| `IRIS_O2_AUDIO_CAPTURE` | O2 host audio capture output path (what the host plays). | [src/dev/mace_audio.rs](src/dev/mace_audio.rs) |
| `IRIS_O2_AUDIO_DUMP` | File to record the O2's audio output at the guest's rate, before resampling. | [src/dev/mace_audio.rs](src/dev/mace_audio.rs) |
| `IRIS_O2_AUDIO_TRACE` | Any presence traces every O2 audio register access and codec operation on stderr. | [src/dev/mace_audio.rs](src/dev/mace_audio.rs) |
| `IRIS_REX3_REC` | Record REX3 bus operations from board start to this path (`gputrace` builds). | [src/dev/ng1/rex3_rec.rs](src/dev/ng1/rex3_rec.rs) |
| `IRIS_REX3_TRACE` | Trace REX3 GFIFO and CPU accesses from board start to this path (`gputrace` builds). | [src/dev/ng1/rex3_trace.rs](src/dev/ng1/rex3_trace.rs) |
| `IRIS_NO_EXIT_ON_POWEROFF` | Any presence prevents process exit on guest power-off/CI quit; set by GUI. | [src/ci.rs](src/ci.rs) |
| `IRIS_NO_FETCH_VERIFY` | 1 suppresses fetchverify checks in a fetchverify build. | [src/cpu/mips_exec.rs](src/cpu/mips_exec.rs) |
| `IRIS_NO_IDLE` | Any presence disables idle parking at startup; `idle on` re-enables it. | [src/cpu/idle_park.rs](src/cpu/idle_park.rs) |
| `IRIS_IDLE_WARMUP_SECS` | Seconds after CPU start before idle parking engages (default 60; 0 disables the warm-up). | [src/cpu/idle_park.rs](src/cpu/idle_park.rs) |
| `IRIS_NO_INLINE_MEM` | Any presence disables JIT inline memory emission. | [src/cpu/jitv2/codegen.rs](src/cpu/jitv2/codegen.rs) |
| `IRIS_NO_JIT` | Any presence disables the REX3 shader JIT and the IMPACT raster JIT; does not select the MIPS engine. | [src/dev/ng1/rex3.rs](src/dev/ng1/rex3.rs) |
| `IRIS_OPT_SPEED` | Presence selects speed optimization in offline PCP/corpus measurements. | [src/bin/jitv2_pcp_dump.rs](src/bin/jitv2_pcp_dump.rs) |
| `IRIS_PRINT_OFFSETS` | Presence runs the opt-in layout-offset print test. | [src/cpu/jitv2/mod.rs](src/cpu/jitv2/mod.rs) |
| `IRIS_QOS` | HostGL QoS: default/off disables promotion; quiet suppresses logging. | [iris-hostgl/src/qos.rs](iris-hostgl/src/qos.rs) |
| `IRIS_REX_JIT_PROFILE` | REX3 shader profile path; see rex3_profile.rs for off/empty behavior. | [src/dev/ng1/rex3_profile.rs](src/dev/ng1/rex3_profile.rs) |
| `IRIS_RSH_TRACE` | Any presence logs each built-in rsh request to stderr (command, bytes moved, duration, unfinished requests), in any build. | [src/net/rsh.rs](src/net/rsh.rs) |
| `IRIS_RUN_CL_PROBE` | Presence runs the opt-in Cranelift probe test. | [src/cpu/jitv2/mod.rs](src/cpu/jitv2/mod.rs) |
| `IRIS_SNAPSHOT_SKIP_CHECK` | 1 bypasses selected snapshot compatibility checks; diagnostic escape hatch. | [src/machine.rs](src/machine.rs) |
| `IRIS_SOCKET` | iris-ci default socket override (Unix path or TCP address). | [src/iris_ci_main.rs](src/iris_ci_main.rs) |
| `IRIS_TFTP_TRACE` | Any presence logs each built-in TFTP request to stderr (file, found or not, bytes, blocks, retransmits, duplicate ACKs, slowest ACK, total time), in any build. | [src/net/tftp.rs](src/net/tftp.rs) |
| `MGRAS_REPLAY` | Replay input path for opt-in IMPACT golden tests. | [src/dev/mgras/record.rs](src/dev/mgras/record.rs) |
| `MGRAS_REPLAY_OUT` | Optional IMPACT replay image output path. | [src/dev/mgras/record.rs](src/dev/mgras/record.rs) |
| `MGRAS_REPLAY_TRACE` | IMPACT replay tracing selector (`gputrace` builds). | [src/dev/mgras/record.rs](src/dev/mgras/record.rs) |
| `MGRAS_SIZE_PROBE` | Input for the opt-in IMPACT size/probe test. | [src/dev/mgras/mgras_tests.rs](src/dev/mgras/mgras_tests.rs) |

`RUST_LOG` configures the host Rust logger; `IRIS_DEBUG_LOG` configures device
logging. Standard host variables (`HOME`, `XDG_CACHE_HOME`, `LOCALAPPDATA`,
`PATH`, `USER`, `WAYLAND_DISPLAY`, `APPIMAGE`, `PROCESSOR_IDENTIFIER`) affect
platform paths/launch behavior, not guest hardware. Build-time metadata uses
`IRIS_GIT_REV`, `APP_VERSION`, and `CARGO_MANIFEST_DIR`; GUI premiere export
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
| `gputrace` | REX3 annotated trace (`rex trace`) and replayable bus recording (`rex rec`); GR2 and IMPACT annotated traces (`gr2 trace`, `mgras trace`). On with `developer`; opt in for plain and `lightning` builds |
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
| `tlbvmap` | Vestigial; the vmap TLB fast path is always on |
| `r5ksc`, `r5ksc_triton` | Refuse to build: no working R5000 secondary-cache model yet (`clanker/cpu/r5k-l1i-cache-bugs.md`) |

</details>

## Emulated CPU

R4400 (the default), R5000, or R10000, chosen per machine at runtime.
All three models are compiled into every binary; there is no separate CPU build
or download. Pair R10000 with the IP28 profile, which has an embedded IP28 PROM.

| | R4400 | R5000 | R10000 |
|---|---|---|---|
| L1 I/D | 16KB direct-mapped, 16B lines | 32KB 2-way, 32B lines | 32KB 2-way, I: 64B / D: 32B lines |
| Secondary cache | 1MB unified L2, 128B lines | none (Config `SC=1`) | 1MB, 128B lines; shadow arrays for CACHE diagnostics |
| ISA | MIPS III | MIPS IV | MIPS IV |
| PRId / FPU FIR | `0x00000440` / `0x00000500` | `0x00002321` / `0x00002300` | `0x00000925` / `0x00000900` |

R10000 loads, stores, and fetches access memory directly; CACHE instructions
operate on shadow tag/data arrays for PROM diagnostics. This models functional
behaviour, not the physical CPU's timing or out-of-order execution.

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
threads` / `--jitv2-threads`, default 1), inlines L1 data-cache loads and stores
for R4400/R5000, uses the direct memory window for R10000, detects
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

- `IRIS_GR4_JIT=off|on|sync` sets the starting mode. `sync` compiles on
  first use and waits.
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

- `IRIS_GR2_JIT=off|on|sync` sets the starting mode. `sync` compiles on
  first use and waits.
- `gr2 jit [on|off|sync|list]` in the monitor shows and changes it.
- `GR2_JIT_DISASM=1` prints each shader's machine code as it compiles.

```
cargo run --release --features lightning,gpujit   # = lightning,rex-jit,gr4-jit,gr2-jit
```
