# iris-gui's frame sink carries only the damaged band

Status: rule, 2026-10-31. `iris-gui/core/src/model/framebuffer.rs`.

## Symptom

A screenshot taken from the GUI was a picture with stale rows: the changed
part current, everything else from an older frame. No build error, no panic —
just a wrong PNG.

## Cause

`FrameSink` is a hand-off between the board's refresh thread and egui, not a
frame store. It carries one frame at a time, and after the damage-tracking
change that frame holds only the band of scanlines that differ from the frame
the GUI's texture already has (`pack_frame` fills `dirty_y..dirty_y+dirty_h`
and leaves the rest of the recycled buffer alone). That is the whole point of
the band: egui uploads it with `set_partial` (`tex_sub_image_2d` of just those
rows), so a settled desktop costs nothing and a blinking cursor costs one row.

The band is only meaningful against the texture the GUI already holds. A
reader that wants a whole picture has no such baseline, so reading the
published frame gives it garbage outside the band.

## Rule

The sink answers "what changed" for one consumer (egui's texture). Anything
that needs the whole picture must go to the board:

- **Screenshots** (`Cmd::Screenshot`): `gfx_display::fresh_frame(display, …)`,
  which requests a frame and copies `screen.rgba` — the same path the CLI and
  `iris-ci` use. It costs nothing until a screenshot is taken, because
  `need_readback` is what makes the refresh loop keep `rgba` current.
- **VNC**: `screen.rgba` too, mirrored band-wise by the renderers when
  `need_readback` (see `clanker/vnc/frame-capture-and-damage.md`).

Two consequences worth remembering when touching this code:

- `RowDamage::remember` runs **only on a successful publish**. A frame
  dropped because the GUI had not taken the previous one yet would move the
  baseline away from the texture, and the next band would then be wrong.
  Same reason the dropped frame's `need_readback` path mirrors the *whole*
  frame rather than its band.
- `status_bar_only` short-circuits `present` before any of this. It is
  computed with `!frame_request`, so a `request_frame` always forces a full
  refresh and cannot be starved by the heartbeat-only case.
