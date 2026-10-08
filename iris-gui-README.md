# iris-gui

An optional egui front-end for the IRIS SGI Indy / Indigo2 emulator. It runs the
emulator in-process, draws the framebuffer in its own window, and adds
named-machine storage with autosave, a configuration editor, disk and CD-ROM
management, networking diagnostics, an in-app serial console, and the benchmark.

iris-gui is a separate workspace crate. A plain `cargo build` of the repo
still builds only the standalone `iris` CLI — the GUI's dependencies
(`eframe` / `egui` / `rfd` plus the iris additive features) only land in the
build when you ask for `-p iris-gui`.

Pre-built installers (Windows, macOS, Linux AppImage/deb/rpm) come out of the
Release workflow, and a sandboxed build ships on the Mac App Store.

---

See [FEATURES.md](FEATURES.md) for the complete GUI/core build-feature list,
shared machine configuration, persisted GUI settings, defaults, and conflicts.

## 1. Build and run

### Default build (recommended)

```
cargo run -p iris-gui --release
```

The first build compiles libchdman-rs, the camera backends (nokhwa / V4L),
and Cranelift. CHD and camera support are unconditional in the core; iris-gui
also enables `rex-jit`. Device use is configured at runtime.

A debug build (`cargo run -p iris-gui` without `--release`) is fine for
iterating on the GUI itself but uses an unoptimized iris core, which means
emulation will be noticeably slow.

### Features

| Feature | What it does |
|---|---|
| `pcap` | PCAP bridged networking; the Networking tab lists host interfaces. Needs libpcap / a WinPcap-compatible SDK |
| `hostcall` | Host services for IRIX programs over private syscalls 3000-3009. No per-machine setting — the Debug tab just reports it's built in |
| `hostgl` | Host OpenGL for IRIX programs (implies `hostcall`); IRIX's replacement libGL replays its GL calls on the host GPU. macOS (CGL) only for now — builds elsewhere but registers no backend |
| `premiere` | `iris/lightning` + `iris/idle-pause` for maximum in-process speed |
| `bundled` | Distributed build: hides source-checkout tools such as premiere preparation. Set by the Release workflow |
| `appstore` | Mac App Store build: implies `bundled`, hides the CI tab, enables security-scoped bookmarks and folder grants |

DaynaPort, Ultra64, IP28/R10000, and the graphics board models are built in.
The `chd`, `camera`, `daynaport`, `ultra64`, `ip28`, `ppmem`, `mips4`, and
`r5k` features have been removed. Ultra64 needs the external gopher64 emulator
and is hidden in App Store builds.

Core features that change how the executor is built pass straight through to
iris:

```
cargo build -p iris-gui --release --features iris/lightning
cargo build -p iris-gui --release --features iris/jitv2
```

- `iris/lightning` — strips breakpoint checks and the traceback buffer. The
  Release workflow builds with it. The Debug tab is hidden in lightning builds
  (`iris::build_features::LIGHTNING`), since the paths it drives are compiled
  out.
- `iris/jitv2` — the experimental MIPS JIT; nothing in the GUI needs to change.
- `iris/idle-pause`, `iris/tlbstats`, `iris/ci_clock`, `iris/developer*`,
  `iris/debug_cache` — various core tweaks. `iris/r5ksc` and
  `iris/r5ksc_triton` refuse to build.

**Help → Diagnostics** lists what's compiled in.

### Verify that the default iris build is unaffected

```
cargo build --release                  # builds only the iris binary
cargo tree -p iris | grep -E 'egui|eframe|rfd'   # should print nothing
```

---

## 2. First run

When no machine TOML files exist, the **New machine** dialog opens
automatically. You'll be asked for:

- **Name** — names the machine folder and its TOML file. Conflicts get a
  numeric suffix (`indy`, `indy-2`, …). Names cannot contain path separators
  or reserved filename characters.
- **Machine model** — Indy (IP24), Indigo2 (IP22), or Indigo2 IMPACT (IP28).
  **Processor** offers R4400, R5000, and R10000. Selecting IP28 defaults to
  R10000 and IMPACT Solid graphics.
- **PROM image** — defaults to "Use embedded PROM (bundled with iris)",
  which lets iris fall back to its built-in PROM blob with no disk file
  needed for IP24, IP22, or IP28. Each profile selects its own embedded image.
- **NVRAM file** — defaults to `nvram.bin` in the machine folder (see Storage) and is
  seeded with a default NVRAM on first use. Indigo2 profiles also have a
  **NVRAM EEPROM file**, which stores their PROM environment and MAC.
- **Total RAM** — preset totals. Tick **Advanced: configure individual banks**
  to set each of the four banks yourself (0, 8, 16, 32, 64, 128 MB; IP28
  also allows 256 and 512 MB). IP28 presets extend to 1024 MB.
- **Boot disk (SCSI #1)** — optional path. For a fresh install, create a blank
  image afterwards from the SCSI menu (**Create blank HDD image…**).
- **CD-ROM (SCSI #4)** — optional install media.

Click **Create**. The machine is saved to `machines/<name>/<name>.toml` and becomes active. Start it
from the **Machine** menu.

If the NVRAM or motherboard EEPROM has no Ethernet MAC, iris-gui writes one
before boot and holds a
MAC-less machine at the PROM instead of booting IRIX into a network that can't
work.

---

## 3. UI tour

### Layout

The window is split into a **control column** on the left and the **emulator
screen** on the right. The screen shows the live framebuffer once a machine is
running (with an overlay while it is powered off). The **VM screen** scale in
the View menu is the maximum draw scale: a larger window (fullscreen, maximised,
resized) centres the picture at that scale, and only a smaller window shrinks it
to fit.

The control column holds, top to bottom: the drop-down menus, the capture
button and hint, the configuration editor, and a status footer (run state,
machine name, MIPS, kernel tick **Hz**, graphics FIFO depth, and the **NET**
light for the internal network).

### Menus

| Menu | Contents |
| --- | --- |
| **File** | New machine… / Switch to machine / Rename current… / Delete current machine / Prepare for premiere… (source builds) / Disk folder access (App Store) / Quit |
| **Machine** | Start / Stop / Reset / Reset NVRAM (fresh PRAM) / Processor (R4400, R5000, or R10000, applies at next Start) / Save and Restore state / Screenshot… / Serial console… |
| **Memory** | Total presets, plus per-bank submenus |
| **SCSI** | Per-ID submenu (SCSI #1 … #7) with context-appropriate actions, plus per-disk **Commit changes to disk** / **Discard changes** for COW overlays and CHD diffs while stopped |
| **View** | Fullscreen (F11), UI scale, VM screen scale |
| **Help** | Version, Diagnostics (build features), Serial console…, How camera & networking work, Mount the shared folder in IRIX, N64 development board, Licenses, Privacy policy |

The **SCSI** menu is the recommended way to attach / detach / replace
drives. Each ID shows its current state inline:

- *(empty)*: Attach HDD… / Attach CD-ROM… / Create blank HDD image…
- *HDD attached*: Enable/Disable COW overlay / Replace image… / Detach
- *CD-ROM attached*: Eject / Insert disc… / Detach

CD-ROM changes apply to a running guest: loading or ejecting a disc signals
IRIX with a media-change Unit Attention, no restart needed.

### Configuration tabs

| Tab | What's there |
| --- | --- |
| **General** | Platform, CPU, CP0 Count clock, persistent JIT code cache (jitv2 builds), RTC offset, graphics board (Newport, GR2 XZ/Extreme, IMPACT Solid/High/Maximum), Newport heads/resolution, PROM, NVRAM, motherboard EEPROM, ttyd1 serial log |
| **Disks** | SCSI devices: image paths, CD-ROM discs, COW overlay, scratch volume, DaynaPort, second controller (IP22) |
| **Networking** | NAT subnet (applied live, with conflict checks against host interfaces), port forwards (added/removed live), NFS share, PCAP interface, **Check networking** diagnostics |
| **Memory** | RAM banks and the resulting total |
| **Display** | Display resolution, window scale, headless, audio on/off and buffering, keyboard shape (ANSI/ISO/JIS) |
| **Video-In** | VINO source: **off** (default), **test_pattern**, **camera** (with a Test Camera preview), **black**; standard; camera index |
| **Debug** | Build features, GDB stub port, capture renderer, idle park, devlog spec, thread affinity. Hidden in lightning builds |
| **CI / Automation** | CI socket, `--ci-display`, serial transcript, SCSI interrupt deferral. Hidden in App Store builds |
| **Benchmark** | The bare-metal benchmark suite, run in-process on a headless machine (see `bench/README.md`) |

### Input and shortcuts

Click the screen to capture mouse and keyboard. Keys are sent by **physical
position**, so IRIX's own `keybd` layout applies on top.

| Keys | Action |
| --- | --- |
| **Ctrl+Alt** (macOS: **Option+Command**) | Release capture |
| **Ctrl+Alt+Esc** | Release capture (fallback) |
| **F11** | Toggle fullscreen |
| **Ctrl+Alt+F11** | Send F11 to IRIX |
| **Ctrl/Cmd+F12** | Pick a disc and load it into the first CD-ROM drive |
| **Ctrl/Cmd + =**, **−**, **0** | Zoom UI in / out / reset |

Ctrl+C / Ctrl+X / Ctrl+V reach the guest while captured (egui would otherwise
turn them into clipboard commands), and switching away from the window releases
capture after a short grace period.

### Serial console and networking help

**Serial console…** (Machine and Help menus) opens an in-app IRIX serial console
(a client of the machine's `serial_port_b`, default `127.0.0.1:8881`). Networking has a "check / fix guest networking" flow that compares `ec0`
against the NAT subnet and can issue the IRIX commands to fix it, and a mount
helper that shows the exact `mount` command for the NFS share (including the
PCAP-mode NFS IP).

---

## 4. Storage model

### Where things live

The storage root is `<config dir>/iris`: `~/Library/Application Support/iris`
on macOS, `$XDG_CONFIG_HOME/iris` (default `~/.config/iris`) on Linux, including
AppImage builds, and `%APPDATA%/iris` on Windows. Mac App Store builds use
Foundation's container home, so the root is inside
`~/Library/Containers/<bundle id>/Data/Library/Application Support/iris`.

```text
iris/
├── gui.json
└── machines/
    ├── indy/
    │   ├── indy.toml
    │   ├── nvram.bin
    │   ├── nveeprom.bin
    │   └── disks/
    │       └── scsi1.raw
    └── indigo2/
        ├── indigo2.toml
        └── disks/
```

`gui.json` holds GUI preferences and the active machine name. Each machine's
TOML file is its configuration source, using the same `MachineConfig` schema
as the standalone `iris` CLI. At launch, the GUI lists
`machines/<name>/<name>.toml` and selects the last opened machine. If that
machine is unavailable, it selects the first available machine or opens
**New machine** when the list is empty. Invalid TOML files produce visible
errors and remain untouched.

The selected machine folder is the emulator's working directory. Config paths
stay relative to that folder, including files selected with **Browse**:

```toml
prom = "(embedded)"
nvram = "nvram.bin"
nveeprom = "nveeprom.bin"

[scsi.1]
path = "disks/scsi1.raw"
```

New disk images default to `disks/scsiN.raw`. External images stay in place;
the TOML references them with relative paths such as `../../../../media/root.chd`.
The same resolution applies to disc changers, NFS shares, TFTP folders, serial
logs, ELF binaries, test dumps, filesystem CI sockets, and JIT cache folders.
TCP CI addresses and network interface names retain their original spelling.
On Windows, external paths must be on the same volume as the machine folder
to be expressible as relative paths.

NVRAM and motherboard EEPROM default to separate files in each machine folder.
App Store CHD diffs use that machine's `chd-diffs/` folder. Security-scoped
bookmarks remain in `gui.json` and use resolved absolute paths, including
external resources reached through `..`.

To run the same machine with the CLI, change to its folder and pass its TOML:

```sh
cd "$HOME/Library/Application Support/iris/machines/indy"
/path/to/iris --config indy.toml
```

### `gui.json` shape

```json
{
  "ui_scale": 1.25,
  "vm_scale": 0.75,
  "active_machine": "indy",
  "recent_configs": [],
  "last_config": null,
  "bookmarks": {},
  "disk_folders": []
}
```

The legacy `recent_configs` and `last_config` keys remain compatible with
older preferences. Machine configurations are never serialized into this JSON.
The File menu has no TOML import or export actions: place a machine TOML in its
matching folder instead. **Prepare for premiere** uses the existing machine
TOML; it no longer creates a separate exported config.

### Autosave and machine management

Config edits save to the active TOML after **~600 ms of inactivity**. Hard
flushes also occur before **Start**, **Quit**, **Switch to machine**, and
**Rename current**. Preference saves do not rewrite machine TOMLs. Selecting
or starting a machine rereads its file, so external edits take effect too.

Switching, creating, renaming, and deleting machines require the emulator to
be stopped and its queued file operations to finish. Renaming moves the
machine folder and renames its TOML together, retaining its disks and
battery-backed state. Deleting removes the TOML registration and retains the
folder's disk and NVRAM files. Retained folders reserve their names to prevent
accidental reuse.

### Migration

On the first launch with machines embedded in `gui.json`, the GUI keeps
`gui.json.pre-toml.bak`, writes each machine to its TOML folder, and removes
only the machine payloads from the preferences. It copies battery-backed
state into each machine's folder, preserving PROM settings and MAC addresses.
Existing disks stay in place and their paths become relative. App Store CHD
diffs are copied from the shared redirect into the machine's redirect, and
the original files remain available for rollback. Names that cannot be used
as folder names receive deterministic safe names.

A legacy `last_config` pointer also migrates once into a machine folder; its
source TOML stays untouched. If migration fails, the original JSON and its
backup remain intact, and preference writes stop until the error is resolved.


---

## 5. Crash and corruption safety

The iris core occasionally calls `std::process::exit` directly. From a
GUI host that's terminal — `catch_unwind` can't intercept it. iris-gui
guards every reachable exit:

| Exit site | When | iris-gui guard |
| --- | --- | --- |
| SCSI attach failure in `Machine::new` | configured image file missing | **Pre-flight**: `App::missing_disks()` runs before sending `Cmd::Start`. If any image is missing, a modal lists them with **Cancel / Edit Disks tab / Detach missing & start**. |
| PowerOff handler (`machine.rs`) | IRIX `halt` finishes | iris-gui sets **`IRIS_NO_EXIT_ON_POWEROFF=1`** in `main()`. iris skips the `exit(0)`; the machine is still stopped cleanly. |
| CI socket `quit` (`ci.rs`) | CI tab enabled | Same env-var guard. |
| Anywhere a `Machine` method panics | such as a bad image or a parse failure | **Worker thread `catch_unwind`** in `handle.rs` around `Machine::new`/`start`/`stop`. The worker stays alive and emits `Evt::Error` instead. |

Behavior of the standalone `iris` binary is unchanged — it doesn't set
the env var, so the soft-power-off `exit(0)` still happens as before.

On Windows, `iris::crash_diag` records otherwise-silent deaths (e.g.
`0xC000041D`) with a symbolised stack in `iris-crash.log`.

### Single instance and TCP ports

iris binds the monitor (`8888`) and two serial listeners (`8880` channel A,
`8881` channel B / ttyd1) on loopback, so two emulators can't share a host
unless their configs give them other ports (`monitor_port`, `serial_port_a`,
`serial_port_b`; the serial console window follows `serial_port_b`). On
startup iris-gui terminates a still-running previous instance (Unix; tracked
with a pidfile) to reclaim those ports. A crashed instance needs no cleanup —
the OS frees its ports.

If a port is still taken, the bind fails soft: the serial channel falls back to
a null backend and the machine boots without it.

### Safe-stop dialog

Clicking **Stop** invokes `safe_stop::evaluate`. If the CPU has halted (clean
shutdown or idle at the PROM) stopping is always safe. Otherwise the decision is
made **from config**: an abrupt power-off only risks the on-disk image when
some attached device writes through to its **base image** — i.e. a plain
read-write hard disk. If one is attached, a modal lists the affected `scsi` IDs
with **Cancel / Send IRIX halt / Force stop**.

These cases are treated as safe:

- **CD-ROM** — read-only.
- **COW overlay** (`overlay = true`) — writes go to a `{path}.overlay`
  sidecar; the base image is never modified.
- **Scratch volume** (`scratch = true`) — a transient host-side file.
- **CHD** (`*.chd`) — writes go to a `.diff.chd` sidecar; the base CHD is
  never modified.

"Send IRIX halt" uses the machine's channel B port (`serial_port_b`, default
`8881`) and writes `halt\n`. This requires a guest shell ready to accept it.

### NVRAM persistence

Normal **Stop**, **Quit**, and guest power-off save the DS1386 NVRAM and, on
Indigo2, the motherboard EEPROM. Writes use a temporary file and replacement
so a failed save preserves the previous image. Forced process termination
bypasses Stop; use `rtc save` / `nveeprom save` before terminating a process.
See [Battery-backed state on Stop](rules/irix/nvram-persistence.md).

### Disk synchronization on exit

CHD diffs are folded back into their base images when you quit ("Synchronizing
disks…", with per-disk progress). Under the macOS App Sandbox the fold needs
write access to the *folder* holding the CHD, which is why the App Store build
asks for folder grants (`rules/macos/chd-fold-needs-folder-grant-not-file-grant.md`).

### Stop timeout for a wedged machine

`Machine::stop()` begins with `cpu.stop()`, which blocks until the CPU thread
acknowledges the halt — a thoroughly wedged guest can make that never return.
To keep the GUI alive, the worker runs the stop on a detached helper thread
(`handle.rs::stop_machine_timed`) and waits at most **5 s**. On timeout it
emits an error toast and reports the machine as stopped so you regain control;
the wedged `Machine` and helper thread are abandoned (they leak, but the OS
reclaims them at exit). The same bound is applied on **Quit**.

---

## 6. Architecture

### Crate layout

```
iris/
├── Cargo.toml         [workspace] iris, iris-gui, iris-hostcall; iris-hostgl excluded
├── src/               iris library + CLI
└── iris-gui/
    ├── Cargo.toml     depends on iris with rex-jit; CHD/camera built into core
    ├── build.rs       APP_VERSION (from RELEASE_VERSION or the crate version)
    ├── assets/        icons, default NVRAM
    └── src/
        ├── main.rs            App: control column, menus, modals, update loop
        ├── handle.rs          EmulatorHandle: worker thread, Cmd/Evt channels
        ├── framebuffer.rs     CaptureRenderer + FrameSink (GfxDisplay → egui texture)
        ├── input.rs           capture, egui → PS/2 keyboard+mouse pump
        ├── config_ui.rs       configuration tabs
        ├── bench_ui.rs        Benchmark tab
        ├── scsi_menu.rs       SCSI menu (per-ID actions, disc pickers)
        ├── safe_stop.rs       Stop-safety evaluator
        ├── settings.rs        GUI preferences, migration, NVRAM seeding and MAC helpers
        ├── machines.rs        TOML folders, relative paths, discovery, and rename
        ├── ram.rs             RAM presets shared by menus and tabs
        ├── netplan.rs         subnet math for the Networking tab
        ├── netfix.rs          "check / fix guest networking" logic
        ├── capture_access.rs  per-OS packet-capture permission (PCAP)
        ├── camera_test.rs     Test Camera preview
        ├── serial_console.rs  in-app ttyd1 console
        ├── filedialog.rs      where file dialogs open
        ├── macos_sandbox.rs   security-scoped bookmarks (App Store)
        ├── single_instance.rs previous-instance reclaim
        └── dialogs/
            ├── new_machine.rs Startup "New machine" dialog
            └── create_disk.rs Blank-HDD-image creator
```

### Thread model

```
                +------------------+        cmd_tx          +------------------+
                |   eframe / egui  | ---------------------> |  worker thread   |
                |   (main thread)  |                        |  (handle.rs)     |
                |                  | <--------------------- |                  |
                +------------------+        evt_rx          +------------------+
                       owns                 (crossbeam)            owns
                  GuiSettings,                                  Option<Machine>
                  MachineConfig,                                + catch_unwind
                  egui::Context                                 around all calls
```

- The eframe app owns the single `winit::EventLoop` for the process.
  iris's own `src/ui.rs` event loop is **not** used — iris-gui never
  calls `Ui::run`, so iris never opens its own window. The selected graphics
  board runs its display loop; iris-gui installs a custom `Renderer`
  (`framebuffer.rs::CaptureRenderer`) through `Machine::get_display()`
  and the `GfxDisplay` trait before the CPU starts, then uploads frames to an
  `egui::TextureHandle`.
- Window operations (resize, fullscreen, surface creation) must run on the
  main/event-loop thread; other threads send requests to it. Calling them from
  elsewhere crashed on Windows and macOS
  (`rules/gui/winit-window-calls-must-be-on-event-thread.md`,
  `rules/macos/winit-030-window-handle-main-thread-only.md`).
- A dedicated worker thread (64 MB stack — machine construction builds large
  objects on the stack) owns the `Machine` when one exists. Commands go over a
  `crossbeam_channel::Sender<Cmd>`; events come back over a `Receiver<Evt>`
  that the main thread drains each frame.
- Input flows the other way: the GUI thread reads egui events each frame and,
  while captured, drives the guest's `Ps2Controller` directly.

### Command and event vocabulary

```
enum Cmd { Start(Box<MachineConfig>), Stop, HaltIrix,
           SetNatSubnet(String), SetPortForwards(Vec<..>), SetPcapInterface(Option<String>),
           SaveState(name), RestoreState(name), Screenshot(path),
           SyncDisks, CowCommit { base, chd }, CowReset { base, chd },
           LoadDisc { id, path, remount }, EjectCdrom { id }, RemountCdrom { id },
           Quit }
enum Evt { Started, Stopped, PowerOff, StateSaved(name), StateRestored(name),
           Screenshot(path), Error(msg), Status(Status),
           SyncProgress { disk, total, fraction }, SyncDone(n), CowDone { committed } }
```

### Build-time feature detection

`src/lib.rs` exposes `iris::build_features` (`PCAP`, `JITV2`, `REX_JIT`,
`HOSTCALL`, `HOSTGL`, `LIGHTNING`, `IDLE_PAUSE`, plus `enabled()` and `banner()`
for the full list). iris-gui reads these to list the build in
Help → Diagnostics, hide the Debug tab in lightning builds, and gate PCAP
and persistent JIT cache controls. The emulated CPU is not
a build feature — read `MachineConfig::machine.cpu`.

### Conventions

- `Result<T, String>` for fallible APIs (matches in-tree iris style; no
  anyhow / thiserror).
- `log` macros for diagnostics; the existing iris `devlog` module
  remains the routing layer.
- `crossbeam-channel` for cross-thread messaging.
- `parking_lot::Mutex` where sharing is needed.

`rules/gui/` collects the hard-won findings (drop order of `CyclesPtr`, GL
teardown on the refresh thread, keyboard capture, layouts, mouse integration,
Windows silent exits); `rules/macos/` covers the App Store and sandbox.
