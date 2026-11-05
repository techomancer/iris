# First board-against-emulator comparison: lines, points, pixel paths, state

Status: measured 2026-11-03. Board: a real V10 (see `v10-hardware-limits.md`).
Emulator: the V12 model at 9aeaf4aa+ (quad ring, 12-bit back end). Both on
an exact RGBA 8,8,8,8 visual (the boards offer different visual lists: the
32 MB V10 has no double-buffered 8888 with depth, the emulated 128 MB V12
does; pick visuals explicitly or the runs differ).

## How

- `test/gltest/linesweep.sh OUT`: 30 line and point cases through glprim
  (`--read front --readout`, 64x48 window, `--visual 8,8,8,8`); compare
  with `tools/ppmdiff.py HW_DIR EMU_DIR --out CMP_DIR`.
- `pixtest --rgba8888`, `glstate`, `blendtest`: PASS/FAIL per case on both,
  compared case by case.
- Moving files out of the emulated guest: `irix-sh.py pull` of a binary
  lost two bytes; uuencode it first. The real board's pull was intact.
- A GL client in the guest blocks while xdm's greeter grabs the server:
  arm autologin (`irix-guest-loop` skill) before running anything.

## Lines and points: the legacy rules are the board's

Aliased lines of widths 1 to 10 (1.5 and 2.5 too), three line stipples
(also on a wide line), points of sizes 1 to 10 and polygon modes line and
point: **pixel-identical**. Port these walks into the pipeline bit-exact.

Different:
- Smooth (antialiased) lines: the board ends a line at its endpoint
  pixel with full coverage (no fade past the end); the model treats the
  line as a rectangle and fades the ends. Across the line the board's
  coverage is not the pixel's area: a width-1 line at y = 9.6 gives 219
  and 42 (area: 230 and 26), wider lines' edge rows run about 0.08 above
  the area and not symmetrically. Needs a sweep of sub-pixel offsets and
  widths to fit.
- Smooth points: 12 pixels differ (edge coverage), up to 53.
- Smooth-shaded lines: 69 pixels a step apart (colour interpolation or
  rounding along a line).

## Pixel paths (pixtest): about 45 emulator failures the board passes

All in the front end's pixel paths, none in the fragment pipeline:
colour-index DrawPixels (shift, offset, maps; BITMAP as index), scale and
bias, 256-entry maps, byte-swapped and 5_5_5_1 draws, ABGR_EXT, stencil
draws and copies, a CopyPixels with zoom (1, -1), and every signed
BYTE/SHORT/INT texture upload (the board clamps negative components to
0; the model stores 0x11).

The board fails five cases the model passes: depth read-backs (DrawPixels
then ReadPixels, CopyPixels, scale-bias, UNSIGNED_INT, FLOAT) come back
about 2e-6 low, and a byte-swapped UNSIGNED_SHORT colour read is a little
off. The board keeps less depth precision than the model's encoder.
Decided (2026-11-03, the user): do not match it. The model stays more
precise; matching would cost work and buy back the board's reputed depth
precision trouble, and nothing the guest does depends on it.

## State (glstate) and blending (blendtest)

glstate: one difference, GL_FOG_INDEX reads back 0 on the model (the
value set on the board). blendtest: all 508 cells pass on both (it allows
a step).
