# Mirai: flipped glDrawPixels and the scissor-enable readback

Two GE HLE gaps found with Mirai 1.0 on Maximum Impact (mirai.log,
2026-10-12).

## Negative pixel zoom: 0xDA8 rows run down

The splash came out upside down. glDrawPixels with glPixelZoom(1, -1)
arrives as an ordinary SEND_PIXELS; only GE pixel state 0xDA8 shows the
flip. Its rectangle [x0, y0, x1, y1, size, 1] gives the first row at
y0 = -1 and the last at y1 = -rows, relative to the raster position, so
the image runs down from it with its first row on top. Matador's upright
images have y0 = 0, y1 = rows - 1. The same goes for x when x1 < x0.

`draw_pixels_words` keeps the raster-engine block ascending and picks
the source row and column per output pixel. The RE4's own zoom registers
(xfrfactor 0x155, xfrinitfactor 0x154) that libGLcore writes are not
modelled; their values did not decode as a zoom. Test:
`gl_draw_pixels_negative_zoom_runs_down`.

## GE readback 0xB9C = glIsEnabled(GL_SCISSOR_TEST)

Parts of the menu bar and toolbar vanished on every other swap. Mirai
keeps its two buffers in step by hand: after each swap it reads the
changed region from the front (GET_PIXELS) and draws it into the back
(SEND_PIXELS). Before copying it asks the GE at 0xB9C, and turns
scissoring off for the copy only when the answer says it is on.
libGLcore's `__glMgrim_IsEnabled` sends GL_SCISSOR_TEST to the GE; it is
the symbol-table entry just before the scissor box's four (0xB98-0xB9B).
We answered 0. The last widget's box (39x20) stayed active, the
full-window copy into the back buffer was clipped to it, and that bank
kept a stale menu bar. It showed whenever the bank was displayed.

Unknown readbacks are answered 0 and noted once (`hq3.rs`, CP_RETURN_MODE).
A wrong 0 can steer an application like this, so check the notes when an
app misbehaves. Mirai also asks 0x293, 0x3, 0x240 and 0x2d0.
