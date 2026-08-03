# GUI mouse integration — current approach, and the classic-Mac absolute-mouse pattern

Status: reference / design analysis. Captures why iris-gui uses pointer
**capture** for the framebuffer, and why the seamless absolute-mouse trick used
by some classic Macintosh emulators does **not** port to IRIX without
significant new machinery. Read this before proposing "make the mouse seamless."

## Current iris-gui approach: capture (grab + hide)

The guest's PS/2 mouse is **relative** — it reports motion deltas, and IRIX/X11
draws its own pointer with its own acceleration. A relative guest pointer can
never stay pixel-aligned with a *visible* host cursor; the two drift, and the
host→guest sensitivity is wrong on top of that.

So iris-gui uses the standard emulator model (mirroring `src/ui.rs`):

- **Click the framebuffer to capture.** On capture we hide the host cursor and
  lock it in place (`egui::ViewportCommand::CursorGrab(CursorGrab::Locked)` +
  `CursorVisible(false)`). Only the guest's own pointer is visible, so there is
  nothing to misalign.
- **Motion uses raw deltas.** eframe forwards `winit DeviceEvent::MouseMotion`
  as `egui::Event::MouseMoved(delta)` regardless of grab mode
  (`eframe .../glow_integration.rs` → `egui_winit::on_mouse_motion`). We read
  those deltas and feed them straight to the PS/2 controller — natural 1:1
  feel, no scaling, no warp-to-center, no edge-piling.
- **Left Ctrl+Alt — left Option+Cmd on macOS — (or focus loss) releases.** Hold
  both and press nothing else; it fires on key-up so Ctrl+Alt+F11 still works.
  Ctrl+Alt+Esc remains as an explicit fallback. Needs egui ≥ 0.35 for discrete
  left/right modifier keys — see
  [keyboard-layout-send-physical-keys-not-logical.md](keyboard-layout-send-physical-keys-not-logical.md).
  Input is gated on capture: while captured,
  keyboard + mouse go to the guest; while not, they stay with egui (so menu
  clicks and config-side-panel typing don't leak into IRIX).

Implementation: `iris-gui/src/input.rs` (`pump`, `release_capture`,
`force_release`); capture is also force-released when the emulator stops so the
host cursor can't get stuck hidden.

> Note: iris's `mouseabs` cargo feature is **misnamed** — it is still grab +
> warp-to-center + relative deltas (`src/ui.rs:532`), *not* absolute
> positioning. There is no hidden absolute backend to tap.

## The absolute pattern (classic Mac OS)

Some classic Macintosh emulators get seamless, capture-free, 1:1 mouse
alignment via an **absolute** mode that bypasses the emulated mouse hardware
entirely. It works because **classic Mac OS exposes a stable, documented,
memory-mapped cursor position you may overwrite, and cooperatively re-reads
it**:

- The host cursor position is written directly into classic Mac OS **low-memory
  globals** — `MTemp` (MouseTemp) and `RawMouse` — and the `CrsrNew` flag is
  set.
- Mac OS polls those globals every tick and "jumps" its cursor to the new
  position. The emulated ADB mouse stays relative but is sidestepped in
  absolute mode.
- The frontend exposes a `MouseMode { Absolute, RelativeHw, Disabled }` seam and
  dispatches an absolute vs. relative update per event.

## Why it does not port to IRIS/IRIX cheaply

IRIX has no equivalent of that mechanism:

- **No fixed mouse globals.** IRIX is Unix + X11. Pointer position lives in the
  X server's dynamically-allocated state at addresses that vary per boot/build —
  there is no constant to poke like `MTemp`.
- **The cursor is a hardware sprite** programmed by the X server through
  REX3/VC2/RAMDAC, and X derives pointer position from *relative* input-device
  events plus its own acceleration curve.
- **No "set absolute pointer via memory" convention.** X's supported absolute
  paths are the input protocol (absolute valuators / XInput) or
  `XWarpPointer`/XTEST — none of which the emulated SGI PS/2-style mouse exposes.

## What an absolute mode would actually require here

One of:

1. **Emulate an absolute pointing device** IRIX already has a driver for (e.g. a
   tablet/touch valuator on the input bus) and feed normalized coordinates —
   new device emulation, depends on IRIX driver support.
2. **A guest-side agent** that calls `XWarpPointer` from coordinates passed over
   a channel — requires installing software inside the guest.
3. **A feedback hack**: locate the X server's pointer coordinates in guest RAM
   at runtime and synthesize relative deltas toward the host position —
   fragile, version-specific, not "without altering much."

None of these is a small port.

## Recommendation

Capture is the correct, standard approach for an X11 guest — it is what
SGI/Unix emulators do, and what the classic-Mac absolute approach itself falls
back to (a relative-hardware mode) when absolute isn't available. The one piece
genuinely worth borrowing is the clean frontend seam: a `MouseMode` enum +
`update_mouse(abs, rel)`. Adopting that abstraction now (even with only
relative/capture wired up) would make options 1 or 2 drop-in later, without
committing to the absolute backend today.
