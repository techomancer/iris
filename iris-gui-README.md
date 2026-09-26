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

## 1. Build and run

### Default build (recommended)

```
cargo run -p iris-gui --release
```

The first build is slow because iris-gui always enables three heavyweight
*additive* iris features so they're available at runtime: `chd`
(libchdman-rs), `camera` (nokhwa / V4L), and `rex-jit` (Cranelift).
Subsequent builds are fast.

A debug build (`cargo run -p iris-gui` without `--release`) is fine for
iterating on the GUI itself but uses an unoptimized iris core, which means
emulation will be noticeably slow.

### Features

| Feature | What it does |
|---|---|
| `pcap` | PCAP bridged networking; the Networking tab lists host interfaces. Needs libpcap / a WinPcap-compatible SDK |
| `daynaport` | DaynaPort SCSI/Link targets on the Disks tab (without it the option is shown with a rebuild hint) |
| `ultra64` | N64 development board toggle and help window |
| `premiere` | `iris/lightning` + `iris/idle-pause` for maximum in-process speed |
| `bundled` | Distributed build: hides the iris.toml import/export items. Set by the Release workflow |
| `appstore` | Mac App Store build: implies `bundled`, hides the CI tab, enables security-scoped bookmarks and folder grants |
| `macos-gui` | Native macOS front-end: system menu bar, dialogs in OS windows, status in the window title (see [Native macOS front-end](#native-macos-front-end)). Ignored off macOS |
| `r5k` | Vestigial. The CPU is a runtime setting (Machine menu / General tab) |

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

On a cold start (no `gui.json`), the **New machine** dialog opens
automatically. You'll be asked for:

- **Name** — used for the machine entry in `gui.json`. Conflicts get a
  numeric suffix (`indy`, `indy-2`, …).
- **Machine model** — Indy (IP24) or Indigo2 (IP22), the **Processor** (R4400
  or R5000), and a display resolution (or leave it to IRIX).
- **PROM image** — defaults to "Use embedded PROM (bundled with iris)",
  which lets iris fall back to its built-in PROM blob with no disk file
  needed. Untick to point at your own `prom.bin`.
- **NVRAM file** — defaults to a stable per-user path (see Storage) and is
  seeded with a default NVRAM on first use.
- **Total RAM** — preset totals. Tick **Advanced: configure individual banks**
  to set each of the four banks yourself (valid sizes: 0, 8, 16, 32, 64, 128 MB).
- **Boot disk (SCSI #1)** — optional path. For a fresh install, create a blank
  image afterwards from the SCSI menu (**Create blank HDD image…**).
- **CD-ROM (SCSI #4)** — optional install media.

Hit **Create**. The machine is saved to `gui.json` and becomes active. Start it
from the **Machine** menu.

If the NVRAM has no Ethernet MAC, iris-gui writes one before boot and holds a
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
machine name, MIPS readout, and the **NET** light for the internal network).

### Native macOS front-end

Building with `--features macos-gui` swaps the layout on macOS. The
default layout above is untouched, and on other platforms the feature does
nothing.

```
cargo run -p iris-gui --release --features macos-gui
```

- The File / Machine / Memory / SCSI / View / Help menus are in the **system
  menu bar**, and the window holds only the emulator screen (or the welcome
  panel while stopped). Save and restore state use fixed slots (snap1–4),
  since a menu has nowhere to type a name. Help → About IRIS lists the build
  features.
- **File → Configuration…** (⌘,) opens the configuration editor in a
  **separate window**. Every dialog is also its own window, so none of them
  ever covers the picture.
- The **window title** carries the status footer: machine name, run state,
  MIPS, networking, on-screen scale, capture state and the latest
  notification.
- Extra shortcuts: ⌘R / ⇧⌘R start / stop, ⌘K capture, ⌘F fullscreen, ⌘N new
  machine, ⌘Q quit (through the same close handling as closing the window, so
  pending CHD changes are still folded back).

The implementation is in `src/macos_native/`. See
`rules/gui/macos-gui-front-end.md`.

### Menus

| Menu | Contents |
| --- | --- |
| **File** | New machine… / Switch to machine / Import iris.toml… / Export current to iris.toml… / Prepare for premiere… (source builds) / Disk folder access (App Store) / Quit |
| **Machine** | Start / Stop / Reset / Reset NVRAM (fresh PRAM) / Processor (R4400 or R5000, applies at next Start) / Save and Restore state / Screenshot… / Serial console… |
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
| **General** | Platform (Indy / Indigo2), CPU, Newport heads, display resolution, PROM, NVRAM, ttyd1 serial log |
| **Disks** | SCSI devices: image paths, CD-ROM discs, COW overlay, scratch volume, DaynaPort, controller (Indigo2) |
| **Networking** | NAT subnet (applied live, with conflict checks against host interfaces), port forwards (added/removed live), NFS share, PCAP interface, **Check networking** diagnostics |
| **Memory** | RAM banks and the resulting total |
| **Display** | Display resolution, window scale, headless, audio on/off and buffering |
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
(a client of `127.0.0.1:8881`). Networking has a "check / fix guest networking" flow that compares `ec0`
against the NAT subnet and can issue the IRIX commands to fix it, and a mount
helper that shows the exact `mount` command for the NFS share (including the
PCAP-mode NFS IP).

---

## 4. Storage model

### Where things live

- `<config dir>/iris/gui.json` — **the system of record.** Contains all saved
  machines, the active machine pointer, UI scale, VM screen scale, and
  fullscreen pref. `<config dir>` is `dirs::config_dir()`: `~/.config` on Linux,
  `~/Library/Application Support` on macOS, `%APPDATA%` on Windows.
- `<config dir>/iris/nvram.bin` — the default NVRAM path. It is absolute on
  purpose, so the NVRAM is the same however the app was launched; older relative
  `nvram.bin` entries are migrated.
- `iris.toml` — the **standalone iris CLI's** config format. iris-gui
  treats it as *import/export only* via the File menu (hidden in `bundled`
  builds), so a machine configured in the GUI can still be booted with
  `cargo run -- --config exported.toml`.

### `gui.json` shape

```json
{
  "ui_scale": 1.15,
  "fullscreen": false,
  "active_machine": "indy",
  "machines": {
    "indy":     { "prom": "(embedded)", "nvram": "/home/me/.config/iris/nvram.bin", ... },
    "irix-65":  { ... }
  },
  "recent_configs": [...],
  "last_config": null
}
```

`MachineConfig` (defined in `src/config.rs`) is serde-serialized directly, so the
schema follows the canonical iris config. Existing `gui.json` files from earlier
iris-gui builds upgrade automatically (missing fields default).

### Autosave

Every form field, dialog result, and menu action that mutates the config
calls `App::mark_dirty()`. Each frame, `App::maybe_autosave()` flushes after
**~600 ms of inactivity** — debouncing keystrokes without leaving you in a
"did it save?" state.

Hard flushes also occur before **Start**, on **Quit**, on **Switch to machine**,
and on **Import iris.toml…**.

### Migration

If you had an older iris-gui that pointed `last_config` at an
`iris.toml`, the first launch of the new build will import that TOML as
a named machine (using the file stem), clear the legacy pointer, and
adopt it as `active_machine`. No manual steps required.

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
`8881` channel B / ttyd1) on loopback, so two emulators can't share a host. On
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

"Send IRIX halt" connects to `127.0.0.1:8881` and writes `halt\n`.

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
├── Cargo.toml         [workspace] { members = ["iris-gui"] }
├── src/               iris library + CLI
└── iris-gui/
    ├── Cargo.toml     depends on iris with chd, camera, rex-jit on
    ├── build.rs       APP_VERSION (from RELEASE_VERSION or the crate version)
    ├── assets/        icons, default NVRAM
    └── src/
        ├── main.rs            App: control column, menus, modals, update loop
        ├── handle.rs          EmulatorHandle: worker thread, Cmd/Evt channels
        ├── framebuffer.rs     CaptureRenderer + FrameSink (REX3 → egui texture)
        ├── input.rs           capture, egui → PS/2 keyboard+mouse pump
        ├── config_ui.rs       configuration tabs
        ├── bench_ui.rs        Benchmark tab
        ├── scsi_menu.rs       SCSI menu (per-ID actions, disc pickers)
        ├── safe_stop.rs       Stop-safety evaluator
        ├── settings.rs        GuiSettings: gui.json, NVRAM seeding and MAC helpers
        ├── ram.rs             RAM presets shared by menus and tabs
        ├── netplan.rs         subnet math for the Networking tab
        ├── netfix.rs          "check / fix guest networking" logic
        ├── capture_access.rs  per-OS packet-capture permission (PCAP)
        ├── camera_test.rs     Test Camera preview
        ├── serial_console.rs  in-app ttyd1 console
        ├── filedialog.rs      where file dialogs open
        ├── macos_sandbox.rs   security-scoped bookmarks (App Store)
        ├── single_instance.rs previous-instance reclaim
        ├── macos_native/      optional native macOS front-end (`macos-gui`)
        │   ├── mod.rs         hooks called from main.rs, config/About windows, title
        │   ├── menus.rs       menu model + action dispatcher
        │   ├── menubar.rs     AppKit NSMenu glue
        │   └── window.rs      egui::Window stand-in that opens OS windows
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
  calls `Ui::run`, so iris never opens its own window. REX3 still
  runs its refresh loop; iris-gui intercepts the per-frame output via a
  custom `Renderer` impl (`framebuffer.rs::CaptureRenderer`) installed
  into `Rex3::renderer` before the CPU starts, and uploads it to an
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

`src/lib.rs` exposes `iris::build_features` (`CHD`, `CAMERA`, `PCAP`, `JITV2`,
`REX_JIT`, `ULTRA64`, `DAYNAPORT`, `LIGHTNING`, `IDLE_PAUSE`, plus `enabled()`
and `banner()` for the full list). iris-gui reads these to list the build in
Help → Diagnostics, hide the Debug tab in lightning builds, gate the DaynaPort,
PCAP and Ultra64 controls, and label the camera source. The emulated CPU is not
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
