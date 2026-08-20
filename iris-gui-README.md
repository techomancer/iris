# iris-gui

An optional egui-based front-end for the iris SGI Indy emulator. Provides a
menu-driven launcher and configuration UI on top of the existing `iris` core
library, with named-machine storage, auto-save, and panic-safe runtime
control.

iris-gui is a separate workspace crate. A plain `cargo build` of the repo
still builds only the standalone `iris` CLI — the GUI's dependencies
(`eframe` / `egui` / `rfd` plus the iris additive features) only land in the
build when you explicitly opt in.

---

## 1. Build & run

### Default build (recommended)

```
cargo run -p iris-gui --release
```

The first build is slow because iris-gui forces several heavyweight
*additive* iris features on so they're available at runtime: `chd`
(libchdman-rs), `camera` (nokhwa / AVFoundation), and `rex-jit`
(cranelift). Subsequent builds are fast.

A debug build (`cargo run -p iris-gui` without `--release`) is fine for
iterating on the GUI itself but uses an unoptimized iris core, which means
emulation will be noticeably slow.

### Adding compile-time-only features

The features above are runtime-selectable (e.g. `.chd` paths and
`vino.source = "camera"`). A second class of iris features is
**compile-time pervasive** — they change how the executor itself is built
and can't be toggled at runtime. To opt in, pass them through the iris dep:

```
cargo build -p iris-gui --release --features iris/lightning,iris/r5k
```

Notable Group-B features:
- `iris/lightning` — strips breakpoint checks + traceback for ~5% speed.
  Interactive debugging (GDB stub) is non-functional in this build. The
  GUI detects this at runtime (`iris::build_features::LIGHTNING`) and
  greys out the GDB port input on the Debug tab.
- `iris/r5k` — switch the emulated CPU to an R5000 with 2-way caches
  (default is R4400 direct-mapped). `iris/r5ksc`/`iris/r5ksc_triton` (the
  secondary-cache variants) don't currently build — see
  rules/testing/r5k-l1i-cache-bugs.md.
- `iris/tlbstats`, `iris/ci_clock`, `iris/gdc`, `iris/mouseabs`,
  `iris/developer*`, `iris/debug_cache` — various core tweaks.

The **Help → Build features** menu lists what's compiled in.

### Verifying that the default iris build is unaffected

```
cargo build --release                  # builds just the iris binary
cargo tree -p iris | grep -E 'egui|eframe|rfd'   # should print nothing
```

---

## 2. First run

On a cold start (no `gui.json`), the **New machine** dialog opens
automatically. You'll be asked for:

- **Name** — used for the machine entry in `gui.json`. Conflicts get a
  numeric suffix (`indy`, `indy-2`, …).
- **PROM image** — defaults to "Use embedded PROM (bundled with iris)",
  which lets iris fall back to its built-in PROM blob with no disk file
  needed. Untick to point at your own `prom.bin`.
- **NVRAM file** — created on first write.
- **Total RAM** — preset 32 / 64 / 96 / 128 / 192 / 256 MB. Tick
  **Advanced: configure individual banks** to set each of the four banks
  yourself (valid sizes: 0, 8, 16, 32, 64, 128 MB).
- **Boot disk (SCSI #1)** — optional path, plus a "treat as empty for
  fresh install" hint.
- **CD-ROM (SCSI #4)** — optional install media.

Hit **Create**. The machine is saved to `gui.json` and becomes active.
Hit **▶ Start** to boot it.

---

## 3. UI tour

### Menu bar

| Menu | Purpose |
| --- | --- |
| **File** | New machine… / Switch to machine → / Rename / Delete / Import iris.toml… / Export to iris.toml… / Quit |
| **Machine** | Start, Stop, Reset, Save state, Restore state, Screenshot |
| **Memory** | Total presets, plus per-bank submenus |
| **SCSI** | Per-ID submenu (SCSI #1 … #7) with context-appropriate actions |
| **View** | Fullscreen (F11), UI scale |
| **Help** | Version + build feature listing |

The **SCSI** menu is the recommended way to attach / detach / replace
drives. Each ID shows its current state inline:

- *(empty)*: Attach HDD… / Attach CD-ROM… / Create blank HDD image…
- *HDD attached*: Enable/Disable COW overlay / Replace image… / Detach
- *CD-ROM attached*: Eject / Insert disc… / Detach

### Toolbar

- **▶ Start** / **■ Stop** (Stop opens the safe-stop dialog if needed)
- **💾 Save state** / **↶ Restore state** — calls `Machine::save_snapshot` / `Machine::ci_restore` against `saves/<name>/`
- **Edit config… / Hide config editor** — toggles the collapsible config
  side panel (see Central panel below) for advanced settings that aren't
  surfaced as menu actions (Network, Video-In, Debug, CI)
- Right side: status pill (PROM / IRIX running / halted / stopped) and
  MIPS counter

### Central panel

The central panel **always shows the emulator screen** once a machine is
running — the live REX3 framebuffer, drawn aspect-fit and centered. While
idle it falls back to the **welcome / status panel**: active machine name,
PROM/NVRAM/RAM summary, attached drive list, network mode, and the big
Start button.

### Config side panel

The **Edit config… / Hide config editor** toolbar toggle slides in a
collapsible **right-hand side panel** with the tabbed editor (Network /
Video-In / Debug / CI). It no longer covers the central panel, so you
can change settings while watching the emulator screen. The panel is
resizable — drag its left edge to trade width with the screen — and is
dismissed with either the toolbar toggle or the **✕** in its header. It
starts collapsed on each launch (open/closed state isn't persisted).

The **Video-In** tab's source selector offers:

- **off (disabled)** — the default. VINO stays memory-mapped (IRIX can
  still probe it) but no video source is installed and the DMA pump thread
  never starts. Use this unless you specifically need IndyCam.
- **test_pattern** — SMPTE-style colour bars + animated luma ramp.
- **camera** — live host camera (needs a `--features camera` build; first
  use on macOS triggers the camera permission prompt).
- **black** — solid black field; drivers attach but no capture.

### Keyboard shortcuts

- **F11** — toggle fullscreen
- **Ctrl =** / **Ctrl -** / **Ctrl 0** — zoom in / out / reset (helps on
  Linux/Wayland where egui's default text size can look small)

---

## 4. Storage model

### Where things live

- `~/.config/iris/gui.json` — **the system of record.** Contains all
  saved machines, the active machine pointer, window state, UI scale,
  and fullscreen pref.
- `iris.toml` — the **standalone iris CLI's** config format. iris-gui
  treats it as *import/export only* via the File menu, so a machine
  configured in the GUI can still be booted with `cargo run -- --config
  exported.toml`.

### `gui.json` shape

```json
{
  "ui_scale": 1.15,
  "fullscreen": false,
  "active_machine": "indy",
  "machines": {
    "indy":     { "prom": "(embedded)", "nvram": "nvram.bin", ... },
    "irix-65":  { ... }
  },
  "recent_configs": [...],
  "last_config": null
}
```

`MachineConfig` (defined in `iris/src/config.rs`) is serde-serialized
directly, so the schema follows the canonical iris config. Existing
`gui.json` files from earlier iris-gui builds upgrade automatically
(missing fields default to empty/None).

### Autosave

Every form field, dialog result, and menu action that mutates the config
calls `App::mark_dirty()`. This sets `cfg_dirty = true` and stamps
`cfg_dirty_since`. Each frame, `App::maybe_autosave()` checks the
timestamp and flushes after **~600 ms of inactivity** — debouncing
keystrokes without leaving you in a "did it save?" state.

Hard flushes also occur:
- Before **Start** (so the on-disk copy matches what we're booting).
- On **Quit**.
- On **Rename current…** and **Switch to machine**.
- On **Import iris.toml…**.

The welcome panel shows `(autosave pending…)` during the debounce window
and the status bar shows a trailing `*` next to the machine name.

### Migration

If you had an older iris-gui that pointed `last_config` at an
`iris.toml`, the first launch of the new build will import that TOML as
a named machine (using the file stem), clear the legacy pointer, and
adopt it as `active_machine`. No manual steps required.

---

## 5. Crash & corruption safety

The iris core occasionally calls `std::process::exit` directly. From a
GUI host that's terminal — `catch_unwind` can't intercept it. iris-gui
guards every reachable exit:

| Exit site | When | iris-gui guard |
| --- | --- | --- |
| `machine.rs:291` SCSI attach fatal | configured image file missing | **Pre-flight**: `App::missing_disks()` runs before sending `Cmd::Start`. If any image is missing, a modal lists them with **Cancel / Edit Disks tab / Detach missing & start**. |
| `machine.rs:585` PowerOff handler | IRIX `halt` finishes | iris-gui sets **`IRIS_NO_EXIT_ON_POWEROFF=1`** in `main()`. iris reads it inside the PowerOff arm and skips the `exit(0)`. The machine is still `.stop()`'d cleanly. |
| `ci.rs:244` CI socket `quit` | future use | Same env-var guard. |
| Anywhere a `Machine` method panics | bad image, parse failure, etc. | **Worker thread `catch_unwind`** in `handle.rs::worker_loop` around `Machine::new`/`start`/`stop`. The worker stays alive and emits `Evt::Error("start failed: <msg>")` instead. |

Behavior of the standalone `iris` binary is unchanged — it doesn't set
the env var, so the soft-power-off `exit(0)` still happens as before.

### Serial TCP ports & stale processes

iris binds two TCP serial backends on fixed ports —
`127.0.0.1:8880` (channel A / tty2) and `127.0.0.1:8881` (channel B /
tty1, used by **Send IRIX halt**). Two consequences:

- **You can't run two emulators at once.** A second instance can't bind
  the same ports; its serial channels fall back to null backends (see
  below).
- **A crashed run can orphan the ports.** If iris-gui dies without fully
  tearing down (a hard crash, `kill -9`), the emulator threads may keep
  the listeners open. The next start then hits `AddrInUse` on bind.

Previously that bind failure `.expect()`-panicked and aborted the whole
process. It now fails soft: `TcpSocketBackend::new` logs a warning and
the channel falls back to a `NullBackend`, so the machine still boots —
that serial channel is just unavailable until the port frees.

If you *need* the serial console back, clear the stale listener:

```
lsof -nP -iTCP:8880 -sTCP:LISTEN     # find who's holding it
pkill -f iris-gui                    # or: kill <pid> from the line above
```

### Safe-stop dialog

Clicking **Stop** invokes `safe_stop::evaluate`. The core does not expose
live dirty-sector state, so the decision is made **from config**: an abrupt
power-off only risks the on-disk image when some attached device writes
through to its **base image** — i.e. a plain read-write hard disk. If one is
attached, a modal lists the affected `scsi` IDs with **Cancel / Send IRIX
halt / Force stop**.

If *nothing* persists guest writes to a base image, stopping won't damage the
hard disk, so the dialog is skipped and the machine powers off immediately.
These cases are all treated as safe:

- **CD-ROM** — read-only.
- **COW overlay** (`overlay = true`) — writes go to a `{path}.overlay`
  sidecar; the base image is never modified.
- **Scratch volume** (`scratch = true`) — a transient host-side file.
- **CHD** (`*.chd`) — writes go to a `.diff.chd` sidecar; the base CHD is
  never modified.

> The "Send IRIX halt" button is wired (TCP `127.0.0.1:8881` → `halt\n`).
> Note the heuristic is intentionally config-based, not runtime: it can't see
> *whether* the guest has actually written to a read-write disk, only that one
> is attached. Exposing `Machine::is_in_prom()` / `dirty_cow_sectors()` /
> event subscription would let the dialog also clear a read-write disk that's
> idle or already halted; see the Phase C notes at the bottom of this doc.

### Stopping a wedged machine

`Machine::stop()` begins with `cpu.stop()`, which blocks until the CPU thread
acknowledges the halt — a thoroughly wedged guest can make that never return.
To keep the GUI alive, the worker runs the stop on a detached helper thread
(`handle.rs::stop_machine_timed`) and waits at most **5 s**. On timeout it
emits an error toast and reports the machine as stopped so you regain control;
the wedged `Machine` and helper thread are abandoned (they leak, but the OS
reclaims them at exit). The same bound is applied on **Quit**, so closing the
app on a hung guest can't hang exit either.

---

## 6. Architecture

### Crate layout

```
iris/
├── Cargo.toml         [workspace] { members = ["iris-gui"] }
├── src/               iris library + CLI (unchanged)
└── iris-gui/
    ├── Cargo.toml     depends on iris with chd, camera, rex-jit on
    └── src/
        ├── main.rs            App, menu bar, toolbar, modals, update loop
        ├── handle.rs          EmulatorHandle: worker thread, command/event channels
        ├── framebuffer.rs     CaptureRenderer + FrameSink (REX3 → egui texture)
        ├── input.rs           egui → PS/2 keyboard+mouse pump
        ├── config_ui.rs       Tabbed editor (Network / Video-In / Debug / CI)
        ├── scsi_menu.rs       Top-level SCSI submenu (per-ID actions)
        ├── safe_stop.rs       Stop-safety evaluator
        ├── settings.rs        GuiSettings: gui.json read/write, machine map
        └── dialogs/
            ├── mod.rs
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
  calls `Ui::run`, so iris never opens its own winit window. REX3 still
  runs its refresh loop; we just intercept the per-frame output via a
  custom `Renderer` impl (`framebuffer.rs::CaptureRenderer`) installed
  into `Rex3::renderer` before the CPU starts. The captured pixels are
  uploaded to an `egui::TextureHandle` and drawn aspect-fit in the
  central panel.
- A dedicated worker thread (8 MB stack to satisfy
  `Physical::device_map`'s allocation) owns the `Machine` when one
  exists. Commands are sent over a `crossbeam_channel::Sender<Cmd>`;
  events come back over a `Receiver<Evt>` that the main thread drains
  each frame.
- Input flows the other way: the GUI thread reads egui events each
  frame and, when the cursor is over the framebuffer rect, drives the
  guest's `Ps2Controller` directly via a handle shared from the worker
  through `EmulatorHandle.ps2: Arc<Mutex<Option<Arc<Ps2Controller>>>>`.

### Command / event vocabulary

```
enum Cmd  { Start(MachineConfig), Stop, SaveState(name), RestoreState(name),
            Screenshot(path), Quit }
enum Evt  { Started, Stopped, PowerOff, StateSaved(name), StateRestored(name),
            Screenshot(path), Error(msg), Status(Status) }
```

`Status` carries `running`, `in_prom`, `power_off_seen`, `dirty_cow`,
`mips`. The worker currently emits `Started` / `Stopped` / `StateSaved`
/ `StateRestored` / `Screenshot` / `Error`. `PowerOff` and `Status` are
reserved for when the iris core exposes `Machine::subscribe_events` and
status accessors (see Phase C notes).

### Build-time feature detection

`iris/src/lib.rs` exposes a `build_features` module:

```rust
pub mod build_features {
    pub const CHD:       bool = cfg!(feature = "chd");
    pub const CAMERA:    bool = cfg!(feature = "camera");
    pub const JITV2:     bool = cfg!(feature = "jitv2");
    pub const REX_JIT:   bool = cfg!(feature = "rex-jit");
    pub const LIGHTNING: bool = cfg!(feature = "lightning");
}
```

iris-gui reads these to:
- Surface the active feature set in **Help → Build features**.
- Warn on the **Disks** form when a `.chd` path is entered into a
  non-CHD build (currently always on for iris-gui).
- Hide the GDB stub port input on the Debug tab under a lightning
  build (interactive debugging is non-functional there).
- Adjust the "camera" label on the Video-In tab.

### Conventions

- `Result<T, String>` for fallible APIs (matches in-tree iris style; no
  anyhow / thiserror).
- `log` macros for diagnostics; the existing iris `devlog` module
  remains the routing layer.
- `crossbeam-channel` for cross-thread messaging.
- `parking_lot::Mutex` where sharing is needed.
- No new `rustfmt.toml` / `clippy.toml`; defaults apply.

A short companion note lives at `rules/gui/01-overview.md` per the
CLAUDE.md convention for hard-won emulator findings.

---

## 7. Phase B status

Phase B is live. iris-gui now hosts the emulator in-process with the
REX3 framebuffer rendered inside the egui central panel and keyboard /
mouse routed through to the guest.

| Item | Status | Where |
| --- | --- | --- |
| Embedded REX3 framebuffer | ✅ landed | `framebuffer.rs::CaptureRenderer` installed in `Rex3::renderer`. Frame copied per `render()` call into a `FrameSink`. GUI uploads to an `egui::TextureHandle` on change and draws aspect-fit, centered. |
| egui → PS/2 input | ✅ landed | `input.rs::pump`. Modifiers diff-synthesized as `Shift/Control/Alt/SuperLeft`. egui `Key` → `winit::KeyCode` mapping covers letters/digits/punctuation/F-keys/navigation. Mouse events only fire when the cursor is inside the framebuffer rect; F11 stays reserved for fullscreen. |
| Save state / Restore state | ✅ landed | `Machine::save_snapshot` (stops the CPU, snapshot to `saves/<name>/`, restart) + `Machine::ci_restore`. Errors surface as `Evt::Error` toasts. |
| Screenshot | ✅ landed | PNG-encoded from the latest `FrameSink` snapshot via the `png` crate, written off the GUI thread inside the worker. |
| Send IRIX halt | ✅ landed | TCP-connects to `127.0.0.1:8881` (iris's standing ttyd1 listener in non-CI mode) and writes `halt\n`. User waits for IRIX shutdown to complete, then clicks Stop. |
| Empty-media CD-ROM | ✅ landed (Task #11) | `ScsiDevice.backend: Option<DiskBackend>`, sense `0x3A` (MEDIUM NOT PRESENT) on `TEST UNIT READY` / `READ CAPACITY` / `READ` when no media. New `Wd33c93a::insert_disc(id, path)` and `eject_to_empty(id)`. SCSI menu gained **Attach empty CD-ROM drive**. |

### Remaining polish (Phase C nice-to-haves)

- **Live SCSI swap**: the SCSI menu currently edits the *config*, which
  means attach / eject changes take effect at the next reset.
  `Wd33c93a::insert_disc` and `eject_to_empty` are wired in the core —
  exposing them through `Machine` and a runtime-only Cmd would let
  inserts / ejects take effect on a live guest.
- **Live status accessors**: `Machine::is_in_prom()`,
  `dirty_cow_sectors()`, and `subscribe_events() -> Receiver<…>` would
  let the safe-stop dialog refine its current config-based heuristic
  (warn whenever a read-write base-image disk is attached) into accurate
  runtime corruption-risk reporting — e.g. clearing a read-write disk
  that is idle or already halted.
- **`rfd` pickers everywhere**: a few remaining text-field path entries
  in the Edit-config tabs don't have browse buttons yet.

Each is independent and small enough to land on its own.
