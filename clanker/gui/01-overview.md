# iris-gui — overview

Optional egui front-end for iris. Lives as a separate workspace crate
(`iris-gui/`); default `cargo build` does not include it. Build with
`cargo build -p iris-gui --release`. The user-facing tour, storage model and
module map are in `iris-gui-README.md`; this note keeps the invariants that are
easy to break.

## Process / thread model

- The eframe app owns the **single** `winit::EventLoop` in the process. iris's
  own `src/ui.rs` event loop is **not** used by iris-gui. The GUI does *not*
  force `headless = true`: the selected board still runs its refresh thread, and the GUI
  installs a capture `Renderer` to receive frames.
- **Window calls belong to the event-loop thread.** Resize, fullscreen, surface
  creation and the first GL `make_current` must run there; other threads send
  requests. Violations crashed Windows (#94) and macOS
  (`winit-window-calls-must-be-on-event-thread.md`,
  `../macos/winit-030-window-handle-main-thread-only.md`).
- **GL teardown belongs to the refresh thread** that owns the context
  (`gl-teardown-must-run-on-the-refresh-thread.md`).
- A worker thread (`iris-gui/src/handle.rs`) owns a `Box<dyn Workstation>`, with a 64 MB
  stack because machine construction builds large objects on the stack. GUI ↔
  worker communication is via `crossbeam_channel` (`Cmd` and `Evt`); every
  machine command dispatch on the worker is wrapped in `catch_unwind`.
- `CyclesPtr` (the live MIPS readout) must be cleared before the `Machine` is
  dropped (`cycles-ptr-must-be-cleared-before-dropping-the-machine.md`).
- GUI preferences persist to `<config dir>/iris/gui.json`; machine configs use
  the core TOML schema at `machines/<name>/<name>.toml`. The selected machine
  folder is the process working directory. Paths inside the folder are stored relative; external paths stay absolute and are never rewritten to `../..` chains. Wait for
  queued worker operations and machine teardown before changing cwd.
  App Store builds use Foundation's container home for storage and resolve
  external relative paths before harvesting security-scoped bookmarks.

## Frames and input

The GUI installs `framebuffer.rs::CaptureRenderer` through
`Workstation::get_display()` / `GfxDisplay::install_renderer` immediately after
construction, before the CPU starts. Every graphics board shares this interface. The renderer publishes frames into a `FrameSink` — pooled buffers,
only the damaged band of scanlines, and a wake so the event loop sleeps until
a frame exists; the event-loop thread uploads that band to an
`egui::TextureHandle` (`gui-frame-sink-carries-only-the-damaged-band.md`).
`IRIS_GUI_GL=1` (`[debug] gui_gl_capture`) selects the board's available GL
capture path.

The sink is the GUI's frame path only. Screenshots (`Cmd::Screenshot`) and VNC
read `screen.rgba` through `gfx_display::fresh_frame` / the renderers' mirror,
never the sink.

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

The config-based heuristic considers `Status::cpu_halted` (including zero
measured throughput), then `[scsi]` entries. It warns for plain writable disks,
excluding CD-ROM, scratch, overlays, DaynaPort and names ending in `.chd`.
This is the UI's classification, not proof that every such image is protected:
uncompressed CHD can write in place, expansion disks are not inspected, and a
zero-throughput window is not a guest shutdown acknowledgement.
The modal offers **Cancel / Send IRIX halt / Force stop**. "Send IRIX halt"
uses `Workstation::inject_serial_console` in-process; the separate console
viewer uses the platform's selected TCP listener.

The standalone binary exits the process on soft power-off; iris-gui sets
`IRIS_NO_EXIT_ON_POWEROFF=1` so the machine only stops. A wedged `Machine::stop`
is abandoned after 5 s so the GUI stays responsive.

## What the GUI knows about iris

Only the public API: `MachineConfig`, `Workstation`, `iris::build_features`, and
accessors added as `pub fn` on the existing types — never private fields. The
emulated CPU is not a build feature; read `MachineConfig::machine.cpu`.

## Empty-media CD-ROM

A CD-ROM uses `Media::Changer(ChangerNode)`; the changer can have no active
node while the drive remains present. INQUIRY still answers, while medium
commands report NOT READY / MEDIUM NOT PRESENT. Construct with
`ScsiDevice::new_empty_cdrom(identity)`. Media changes run through the shared
`DiscChanger` / `ScsiTargets` interface in `src/scsi_monitor.rs`, not a
WD33C93A-specific backend. In TOML, `kind = "cdrom"` (or `cdrom = true`) with
empty `path` and `discs` selects an empty tray. The GUI's `Cmd::LoadDisc`,
`EjectCdrom` and `RemountCdrom` operate on the running machine
(`cdrom-hot-insert-remount.md`).
