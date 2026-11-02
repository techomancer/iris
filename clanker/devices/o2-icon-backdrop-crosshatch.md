# O2 icon backdrops crosshatch because GL dithers the window's own colour

Status: open on one point only, whether real CRIME does the same. Needs a
lossless capture of a real O2 running 6.5.22 or later. Seen 2026-11-01 on
6.5.22 at 1280x1024.

## Symptom

In the Icon Catalog each icon sits on a faintly crosshatched box against the
flat panel. Pixels read `0x2a15` (panel) next to `0xaa36` and the like.

## Cause

Capture with `re trace FILE re` and change catalog page. The desktop's icons
are GL in a double-pix A1R5G5B5 window (BufMode 0x226/0x227). Each icon starts
with two triangles and each label with two more, with
`[GL SHADE AA_LINE BLEND DITHER]`, shade 81.999 131.998 172.997, alpha 255.

That colour is the panel's own. X holds it as 5-bit (10, 16, 21) and reports
it as 0x5252 0x8484 0xadad; the desktop divides by 65536 rather than 65535, so
times 255 it lands just under 82, 132, 173. The spec truncates the fraction:
81, 131, 172. Without dithering all three truncate to the panel's 5 bits and
the box vanishes. With `GL_DITHER` on, its default, 1, 3 and 4 above a 5-bit
step get bumped on part of the 4 x 4 cell.

The panel itself stays flat because GL's clear packs 5 bits without dithering.

## What is established

- The trace above is the guest's own drawing. It does not depend on our engine.
- The DrawMode bit is dither (driver, NetBSD and OpenBSD agree against the spec).
- The matrix is libGLcore's `__glDitherTable`, identical to `dither5`'s:
  0 8 2 10 / 12 4 14 6 / 3 11 1 9 / 15 7 13 5.
- Add then truncate: libGLcore's clear treats a 3:3:2 colour as unchanged by
  dithering only when the low bits are zero or the channel is 255. A scaling
  dither would have made k * 255 / 7 the grid.
- The spec's dithering section is "TBD". Amplitude at 5 bits is inferred:
  (c + b / 2) >> 3, which equals the software path's bias of (2b + 1) / 32 of a
  step once c is an 8-bit integer.

## What is not

Whether the chip really produces this. Any full-amplitude dither gives a
pattern from these inputs, so expect it on hardware, but nobody here has looked.
The screenshots found online are older releases whose panels are not 5-bit
colours, so they do not compare. `re dither off` shows the undithered picture
for a side-by-side.

## Rule

Do not "fix" this by dropping the fraction differently or by skipping dither on
flat colours: the spec and the driver are explicit about truncation, and the
driver's clear path relies on flat rectangles being dithered. All fragments
dither now, as the spec's pipeline order has it (blend, dither, logic op):
triangles, GL lines, points and rectangles.

Not done: 3:3:2 windows (8-bit RGB). `pack` has no such layout; only 5-bit
double-pix windows dither.
