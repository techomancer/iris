# Real Indy XZ against the emulated XZ

Status: open, 2026-11-24. Both IRIX 6.5.22m, R5000, GR3-XZ (4 GE7: GL
says GR3-Elan on both), the same `test/gltest` sources built with MIPSpro
7.4.4m on each, the same `run-xz.sh` (glsuite with
`PIGLIT_LIST=express.list`, vissweep over XZ's visuals, then `xsweep.sh`
pc8 pc12 tc12 tc24 with `ximgtest` per mode). Results:
`shared/results/real-xz` (real), `emu2` (emulator before the colour and
quad fixes), `emu3` (after, state/pix/scenes/quanttest only); comparison
`shared/results/real-xz-vs-emu.txt` (`test/gltest/glsuite-compare.sh`).

## Running it on a real machine

- No `/shared` there: push a uuencoded tarball with
  `tools/irix-sh.py --host H push` into a local `/shared`
  (`clanker/testing/piglit-on-irix.md`), `sh irix.sh` to build,
  `nohup sh run-xz.sh TAG`.
- Anything over the test windows takes pixel ownership and every GL read
  returns 0: ARP "address in use" alerts, a running screen saver. glsuite
  runs `xset s reset`; check xsweep's `desktop.rgb` before trusting a run.
- `glx@glx-window-life` kills Xsgi on the real Indy and in the emulator
  (the emulator also on `glx_ext_import_context@free context`); xdm then
  shows the login window, whose grab hangs every later test, and after a
  few deaths stops restarting the server. piglit-run.sh now checks for
  toolchest after each test and runs `PIGLIT_XRESTART` (glsuite sets
  `xrestart.sh`); `DIR/xdied` lists the killers.

## Fixed (2026-11-24)

- Colour quantisation: min(255, floor(v * 256)) for RGB; quanttest now
  matches the real board on all 2179 samples (clear, flat, dithered).
- Quads, quad strips and polygons are drawn as a fan from the first
  vertex; their >2 differences are gone.
- 4 GE7 on Indy (variant `Elan`), so GL_RENDERER matches.

After these, scenes match the real board exactly for clipplane, depth,
dlist, points and stencil, and within 1-2 elsewhere, except below.

## Still different (emulator side)

- **EXP2 fog** (`fogexp2`): 22 % of pixels differ by more than 2 over the
  denser half. EXP and linear agree within rounding. `fogtest` prints the
  factor curve on both machines.
- **Local light with spot** (`litlocal`): 13 %, two wedges, green.
- Smooth interpolation is 1-2 off on 10-35 % of pixels of shaded scenes:
  GE slope / iterator precision.
- **X, XYBitmap XPutImage** does not land in any visual; real draws fg/bg.
- **X, the 4-bit 1:2:1 TrueColor visual** round-trips nothing; real
  round-trips every ximgtest case.
- **XReadDisplay of the 2- and 4-bit overlay visuals**: colour in the
  wrong byte (emulator 0x00009e, real 0x9e3200).
- Root reads over the 4-bit TrueColor window differ in the low blue bits.

## Real board departs from the spec (emulator too exact)

Suite expectations came from the spec and the emulator. Model the
hardware or XFAIL with the reason:

- glstate current/COLOR (8-bit quantised), COLOR/integer (component 0 reads
  0), NORMAL (x and z read 0), RASTER_COLOR. quanttest's
  glGetFloatv(CURRENT_COLOR) reads 0 for most values on the real board.
- blendtest `add` (5 cells).
- pixtest: depth draw/read/copy (#303-306), glBitmap (#314-317; CI raster
  index #45), a set of texture uploads (#321-#476).
- piglit (clean runs: real 79 pass, emulator 85): real fails, emulator
  passes gl-1.0-scissor-bitmap, clipflat, gl-1.1-xor, point-line-no-cull,
  readpix-z, triangle-rasterization-overdraw; glx-window-life crashes on
  the real board. No test passes on the real board and fails here.
