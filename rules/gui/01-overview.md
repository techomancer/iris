# iris-gui — overview

Optional egui front-end for iris. Lives as a separate workspace crate
(`iris-gui/`); default `cargo build` does not include it. Build with
`cargo build -p iris-gui --release`. The user-facing tour, storage model and
module map are in `iris-gui-README.md`; this note keeps the invariants that are
easy to break.

## Process / thread model

- The eframe app owns the **single** `winit::EventLoop` in the process. iris's
  own `src/ui.rs` event loop is **not** used by iris-gui. The GUI does *not*
  force `headless = true`: REX3 still runs its refresh thread, and the GUI
  installs a capture `Renderer` to receive frames.
- **Window calls belong to the event-loop thread.** Resize, fullscreen, surface
  creation and the first GL `make_current` must run there; other threads send
  requests. Violations crashed Windows (#94) and macOS
  (`winit-window-calls-must-be-on-event-thread.md`,
  `../macos/winit-030-window-handle-main-thread-only.md`).
- **GL teardown belongs to the refresh thread** that owns the context
  (`gl-teardown-must-run-on-the-refresh-thread.md`).
- A worker thread (`iris-gui/src/handle.rs`) owns the `Machine`, with a 64 MB
  stack because machine construction builds large objects on the stack. GUI ↔
  worker communication is via `crossbeam_channel` (`Cmd` and `Evt`); every
  `Machine` call on the worker is wrapped in `catch_unwind`.
- `CyclesPtr` (the live MIPS readout) must be cleared before the `Machine` is
  dropped (`cycles-ptr-must-be-cleared-before-dropping-the-machine.md`).
- Settings **and machine configs** persist to `<config dir>/iris/gui.json`
  (`dirs::config_dir()`), which is the system of record. `iris.toml` is
  import/export only, hidden in `bundled` builds.

## Frames and input

The GUI installs `framebuffer.rs::CaptureRenderer` in `Rex3::renderer`
immediately after `Machine::new`, before the CPU starts. Each `render` call from
the REX3 refresh thread does a stride-aware copy into a `FrameSink`; the main
thread uploads it to an `egui::TextureHandle`. `IRIS_GUI_GL=1`
(`[debug] gui_gl_capture`) uses the GL compositor's capture path instead.

PS/2 input flows through `input.rs::pump`, only while captured:

- Keys are sent by **physical position**, never layout-translated
  (`keyboard-layout-send-physical-keys-not-logical.md`, #72).
- egui swallows Tab/arrows/Esc and turns Ctrl/Cmd+C/X/V into clipboard
  commands; the pump re-synthesises them
  (`keyboard-capture-egui-steals-tab-arrows-esc-and-ctrl-cxv.md`).
- Modifiers arrive as real left/right key events; AltRight is AltGr.
- Capture is released by Ctrl+Alt (Option+Command on macOS), Ctrl+Alt+Esc, or
  focus loss after a grace period. Plain F11 is the fullscreen toggle;
  Ctrl+Alt+F11 sends F11 to the guest.
- Mouse capture details: `gui_mouse_integration.md`.

## Safe-stop logic (`iris-gui/src/safe_stop.rs`)

Stopping without a prompt is safe when the CPU has halted (clean shutdown,
soft power-off, or idle at the PROM), or when no attached device writes guest
data straight into its base image. CD-ROMs, COW overlays, scratch volumes and
CHDs (`.diff.chd` sidecars) are all safe; a plain read-write hard disk is not.
The decision is config-based: the core exposes no live dirty-sector count.
Otherwise a modal offers **Cancel / Send IRIX halt / Force stop**; "Send IRIX
halt" writes `halt\n` to `127.0.0.1:8881`.

The standalone binary exits the process on soft power-off; iris-gui sets
`IRIS_NO_EXIT_ON_POWEROFF=1` so the machine only stops. A wedged `Machine::stop`
is abandoned after 5 s so the GUI stays responsive.

## What the GUI knows about iris

Only the public API: `MachineConfig`, `Machine`, `iris::build_features`, and
accessors added as `pub fn` on the existing types — never private fields. The
emulated CPU is not a build feature; read `MachineConfig::machine.cpu`.

## Empty-media CD-ROM

`ScsiDevice.backend` is `Option<DiskBackend>`. `None` represents "drive
present, tray empty": INQUIRY still answers, TEST UNIT READY / READ
CAPACITY / READ / READ TOC return `CHECK CONDITION` with sense key
`0x02` (NOT_READY) + ASC `0x3A` (MEDIUM NOT PRESENT). Construct via
`ScsiDevice::new_empty_cdrom()`; mount/swap media with
`Wd33c93a::insert_disc(id, path)`; unload with
`Wd33c93a::eject_to_empty(id)`. In `iris.toml` an empty-tray CD-ROM is
`cdrom = true` with an empty `path` and no `discs` — `MachineConfig::
validate` accepts this state. The GUI's `Cmd::LoadDisc` / `EjectCdrom` /
`RemountCdrom` drive these on a running machine
(`cdrom-hot-insert-remount.md`).
