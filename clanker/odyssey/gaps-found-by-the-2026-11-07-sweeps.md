# Gaps the 2026-11-07 test sweeps found (both raster units)

Status: open, 2026-11-07. Measured on the V12 model, IRIX 6.5.30, with
`test/gltest/glsuite.sh` and `xsweep.sh` (results compared with
`test/gltest/glsuite-compare.sh`). Every item fails the same way on the CPU and
the GPU raster unit: these are model gaps, not GPU regressions.

- **Colour-index 8 reads return index 0.** `pixtest --ci 8`: 9 pass, 32
  fail, starting with `ci8/baseline/rects` and `ci8/baseline/clear` (index
  0 read where 17 and 5 were drawn). `scrsave` of a PseudoColor 8 root
  (`xsweep.sh` mode `pc8`) is almost entirely black for the same reason.
  The screen shows the right pixels, so the draw is right and the read
  (format choice or packing for 8-bit index windows) is wrong.
- **X screen reads over some windows are noise.** `scrsave` of the
  TrueColor 24 desktop is right except inside the icon catalog's panel;
  over GL windows (the glprim scenes) the window area is streaks. The GL
  readbacks of the same windows are right, so it is the X server's
  `XGetImage` path (its reads through the board), not the drawing.
- **Visuals that fail on both units:** 10,10,10,2 pbuffers (4 cases) and
  12,12,12,12 with a 16-bit depth buffer (4 cases) in `vissweep.sh`: the
  12-bit z16 window draws nothing (clear colour and clear depth read back).
- **`gltest --db --spin` shows only the clear.** Double buffered, the
  window shows the clear colour and never the triangle, on both units,
  while the single-buffered run shows it. Swaps happen (the displayed
  buffer bit toggles).
- **`cicopy`**: the colour-index back-to-front copy to another position
  leaves the destination at the clear index (`front, copied from back`).
- **glstate**: 5 failures (fog index, index write mask, index clear value,
  `PIXEL_TEXTURE_SGIS` enable), as before.
- **pixtest RGBA**: 53 failures, the ones the first V10 comparison listed
  (`v10-vs-emulator-first-comparison.md`) plus the texture RGBA8 signed
  types.
- **TyrQuake started from a telnet shell** (`+timedemo demo1`) stays on its
  console and draws nothing, on the GPU unit at least; started from the
  desktop it plays (the user's runs). Not chased.

Also seen: `texsweep.sh` read back a lower-precision texture for one or two
of the 12- and 16-bit luminance and intensity formats on some runs only
(different formats each run). Replaying the run's recording on the CPU unit,
the GPU unit and the GPU unit with slice ends gives byte-identical host
writes, so the boards agree on what they were sent; the guest's GL library
sent something different that run (a smaller internal format, by the look of
the 4-bit steps). Recording: `ignore/recordings/gpu-cases/texsweep.rec`.
