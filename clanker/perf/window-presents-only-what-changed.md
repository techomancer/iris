# The window presents only what changed

Symptom: an idle IRIX desktop still kept the display thread busy. Every
heartbeat (ten a second) uploaded the whole framebuffer to the GPU, rendered
the picture into the FBO again and rebuilt the picture's mip chain, although
the guest had drawn nothing since the last frame. On a small host the
display thread was the largest thread after the CPU.

Three costs, one change:

## 1. The upload and the render

`Slot::frame` (`src/display/mod.rs`) knows whether the board gave a new
picture: `Composed::Frame`/`Gpu` means new, `Composed::Unchanged` means the
frame on screen still stands. It writes that into `screen.status_bar_only`
(`!new_frame && self.shown`). The compositor (`src/compositor.rs`,
`src/gl_compositor.rs`) keeps its own `picture_current` flag and the size it
composed for; when the screen says the picture did not change and the size
is the same, `compose()` hands back the texture it has without uploading the
framebuffer, the window IDs or the lookup tables, and without rendering the
picture into the FBO again.

The flag belongs to the compositor, not to the screen: a fresh compositor
(after `disp compositor`, or a swap between the GL and the software
compositor) holds no picture and must compose one. `destroy()` clears it.

This is safe because every board that answers `Composed::Unchanged` leaves
the screen's framebuffer alone: IMPACT's `Taken::None`, VPro's
`reader::Composed::No`, the O2's `render_direct` with nothing written. A
board that changes the picture and still answers `Unchanged` would show a
stale window; `IRIS_DISPLAY_VERIFY=1` is what catches that.

## 2. The mip chain

`generate_mipmap` on the 2048x2048 RGBA8 output texture writes and reads
about 22 MB per call, and the old code called it for every texture of every
present (the main picture, the overlay, the status bar).

A chain is only wanted when the window shows the picture smaller than the
guest drew it (`scale_f < 1`): a magnified picture reads level 0 either way.
So the chain is built when the picture first needs it and again only when
the picture changed (`Compositor::picture_changed`). The integer
(`texelFetch`) path needs no chain at all, so its min filter is LINEAR and
the texture stays complete.

## 3. The marks

IMPACT's `marks_show` (`src/dev/mgras/disp.rs`) tested every pixel of a
mark's box: it painted the row's window IDs (a 2048-byte fill per row) and
then scanned them. The VC3 line table already names the window IDs of a row
as runs, so the test now walks the runs (`frame::for_each_did_in_row`) and
asks about the window IDs that reach into the box. The answer is the same:
the walk reproduces the painted row exactly, because a later run overwrites
an earlier one's pixels and the union of the later runs' spans is one
interval, so one bound (`hi`) is enough.

What is left on the table: the overlay and the status bar are drawn from
their own textures and are uploaded every present. They are small (the
status bar is 2048x16), and the status bar changes every second.
