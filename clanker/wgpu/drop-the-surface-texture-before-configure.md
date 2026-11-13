# Drop a surface texture before configuring the surface again

Status: fixed 2026-11-08 (5b9c80b2), confirmed by dragging the window border.

## Symptom

Dragging the wgpu window's border left it broken for good, with this pair
repeating on every frame:

```
In Surface::configure
  The `SurfaceOutput` returned by `get_current_texture` must be dropped before re-configuring ...
In Queue::present
  Surface is not configured for presentation
```

## Cause

`Presenter::present` (`src/wgpu/present.rs`) answered a `Suboptimal`
texture by calling `Surface::configure` while still holding that texture.
wgpu rejects the configure, so the surface keeps its old size; the next
acquire is `Suboptimal` again and the failure repeats on every frame.
It was not a threading problem: the winit thread only drops the new size in
a mailbox and the board's render thread applies it before presenting.

## Rule

Never call `Surface::configure` while a `SurfaceTexture` from that surface
is alive. On `Suboptimal`, drop the texture, configure, acquire again; if
the second texture is still suboptimal (the window moved again mid-drag)
draw it anyway and let the next frame catch up. `Outdated`/`Lost` carry no
texture, so configuring there is fine.
