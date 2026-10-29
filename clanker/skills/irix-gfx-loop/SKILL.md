---
name: irix-gfx-loop
description: >-
  Debug and test emulated SGI graphics through the real IRIX stack: Newport
  (REX3, XL/XGE on Indy and Indigo2), GR2 (XZ/Extreme) and IMPACT (MGRAS:
  Solid/High/Maximum). Boot IRIS, log in to a desktop, build and run the
  test/gltest programs in the guest, capture board traces and recordings from
  the monitor around each run, check rendered pixels, replay captures
  offline as unit tests, and inject mouse/keyboard input. Use for any
  graphics bug, board bring-up, GL/X coverage work or graphics performance
  investigation.
---

# Graphics debugging loop (REX3, GR2, IMPACT)

The loop: put one known drawing through the guest's real graphics stack,
capture what the board saw, compare the pixels with what the drawing should
produce, fix, replay the capture offline, repeat. Guest restarts cost
minutes, so capture once and iterate on replays wherever possible.

Read first: the `irix-guest-loop` skill (booting, telnet, NFS, autologin,
shutdown), `debug.md` (every monitor command) and the board's notes in
`clanker/rex3/`, `clanker/gr2/` or `clanker/mgras/`.

## 1. Pick the board

| Board | Config | Monitor prefix | Model |
|---|---|---|---|
| Newport (REX3) | `[graphics] board = "newport"` (Indy, Indigo2 IP22) | `rex`, plus `vc2`, `xmap`, `cmap`, `bt445`, `disp` | register-level REX3 (`src/dev/ng1/`), optional `rex-jit` |
| GR2 XZ / Extreme | `board = "xz"` or `"extreme"` (Extreme on IP22 only) | `gr2`, `re3` | HQ2 command HLE + RE3 raster (`src/dev/gr2/`), optional `gr2-jit` |
| IMPACT (MGRAS) | `board = "solidimpact"`, `"highimpact"`, `"maximpact"` (IP28) | `mgras`, `rss` | HQ3/GE HLE + RSS raster (`src/dev/mgras/`), optional `gr4-jit` |

On IMPACT, `[machine] profile = "indigo2_ip28"` and `cpu = "r10000"` (see
`docs/indigo2-ip28.md`). GR2 and IMPACT need one head and
`resolution = "guest"`.

Build with `developer` (monitor tooling, `gputrace`; the CPU starts paused,
`start` it), or add `--features gputrace` to a `lightning` build when speed
matters. `developer` and `lightning` cannot be combined. Add the board's JIT
feature (`rex-jit`, `gr2-jit`, `gr4-jit`, or `gpujit` for all three) to test
the compiled path. If the user has a build they use, build into your own
`--target-dir` instead of overwriting it.

## 2. Bring up a desktop

1. Boot and wait for telnet (`irix-guest-loop` §2).
2. Make sure autologin is armed (`irix-guest-loop` §4): GL and X clients need
   a logged-in session, not xdm's login window.
3. Wait for `4Dwm` and `toolchest` in `ps -ef`, then ~15 s more.
4. `xset s off`, or the screen saver turns every probe black after 10 min.

## 3. Build the test programs in the guest

`test/gltest` holds board-neutral GL/X programs: `glprim` (one primitive per
run, `--help` lists every option), `gltest` (scenes and `--bench`),
`pixtest`, `blendtest`, `glstate`, `seltest` (selection hit records), plus helpers `xgetpix` (read root-window
pixels through XGetImage), `xclick`, `xresize`. `irix.sh` builds them with
MIPSpro.

```sh
mkdir -p shared/gltest && cp test/gltest/* shared/gltest/
tools/irix-sh.py run 'mkdir -p /tmp/gltest && cp /shared/gltest/* /tmp/gltest/' \
                     'cd /tmp/gltest && sh irix.sh glprim xgetpix 2>&1'
```

`/tmp` is emptied at boot: redo this every boot.

## 4. One test case

### GR2 and IMPACT: `tools/glprim-case.py`

```sh
tools/glprim-case.py CASE [--probe gx,gy[=rrggbb]]... [--notrace] -- <glprim options>
```

It detects the board from the monitor (`--board gr2|mgras` overrides),
starts the board trace, runs glprim, reads its real window origin, probes
pixels in GL window coordinates with `<board> pix`, dumps the framebuffer,
crops the window to `$GLPRIM_OUT/CASE.png`, stops the trace and exits with
the number of failed probes. It talks to the monitor on 127.0.0.1:8888
directly and works alongside the MCP server.

### By hand (any board)

```
<board> trace /tmp/case.log all                  # gr2 / mgras / rex (gputrace builds)
tools/irix-sh.py run 'cd /tmp/gltest && ./glprim <options> --hold 4000'
<board> pix <x> <y>                              # gr2 / mgras: one display pixel, decoded
<board> fbdump /tmp/case                         # planes + composed screen as PNG
<board> trace off
```

Run the captures while `--hold` keeps the window up (background the guest
command with `&` if you need to sample during the hold). Newport has no
`rex pix`: use `rex fbdump DIR` and read the PNG, `xgetpix X Y [W H]` in the
guest, or `iris-ci screenshot`.

glprim prints its **real window origin** (`window origin OX,OY size WxH`);
the window manager moves windows. GL window pixel (gx, gy) is display pixel
`(OX + gx, OY + H - 1 - gy)`.

Default geometry: the unit square scaled to the viewport (glOrtho 0..w,
0..h); triangle (0.2w,0.2h) red, (0.8w,0.2h) green, (0.5w,0.8h) blue. Flat
shading uses the provoking vertex (the last vertex for
triangles/strips/fans/quads, the FIRST for GL_POLYGON). Default clear is
pink (1, 0.5, 0.75) = `ff80bf`.

## 5. Verify

- Compute expected values from the GL spec: pixel centres at +0.5, the
  top-left-style fill rule, flat = provoking vertex colour, smooth =
  barycentric interpolation (allow ±2 per channel). Check the interior, the
  first pixel in and out on each edge, and one pixel just outside the window,
  which must be untouched.
- IMPACT's scanout applies the DAC gamma (SGI default 1.7): mid-level
  colours read back brighter than on GR2 (the pink clear reads `ffaad7`).
  Probe saturated colours, or compare raw values with glprim
  `--read ... --check`.
- IMPACT: glprim picks the deepest visual by default (12,12,12), which the
  raster JIT leaves to the interpreter. Use `--visual 8,8,8,8` to exercise
  the JIT.
- Read `fbdump` PNGs with your image viewer tool; compare against a known
  good dump rather than by eye where you can.

Change one option at a time. A useful order: clear only (`-p none`), flat
triangle, `--smooth`, `--mono`, every `-p` primitive, `--vtx
2f/4f/2i/3i/2s/3d`, `--color 4f/3ub/4ub/once`, `--viewport`, `--scissor`,
`--cull`/`--cw`, `--polymode`, `--persp`, `--depth`, `--db`. `--scene
depth|stencil|alphatest|blend` draws small multi-primitive scenes with known
images (see `draw_scene` in glprim.c).

## 6. Reading traces

- **REX3**: `rex trace FILE [gfifo,cpu|all]` logs register writes as the
  GFIFO consumer sees them, decoded. `rex status` dumps the drawing
  registers; `rex jit list` shows compiled shaders and failures.
- **GR2**: `HQ exec` lines are the HLE's view: `VERTEX (...) -> window
  (...)`, `GL triangle ...`, `GL_CLEAR ... rect`. Unknown tokens show as
  `tokNNN` / `(N words, not implemented)` with their modifier bits decoded.
- **IMPACT**: `mgras trace FILE hq,rss,cpu,tex`; `rss regs` decodes the IR
  and shows each raster engine's backlog; `mgras stats` counts commands and
  DMAs; `locks` shows who holds what if the board seems stuck.

New hardware facts for GR2/IMPACT belong in the board headers if you have
the local, unpublished `ignore/gr2` or `ignore/gr4` trees (ask the user);
emulator behaviour and gotchas go in `clanker/<board>/`.

## 7. Capture once, replay offline

| Board | Capture | Replay as a unit test |
|---|---|---|
| REX3 | `rex rec FILE` ... `rex rec off` (`rex rec mark` adds a checkpoint) | `REX3_REPLAY=FILE cargo test --release rex3_replay_file -- --nocapture` (`REX3_REPLAY_ENGINE=interp\|jit`, `REX3_REPLAY_NOCHECK`) |
| GR2 | `gr2 trace` of the case, cut to the HQ2 FIFO writes from the window setup (`0x1e5`) to the `GL_FINISH` after the draw | `GR2_REPLAY=FILE cargo test --release --lib gr2_replay_file -- --ignored` (`GR2_REPLAY_OUT`); pinned cases use `replay_trace()` in `src/dev/gr2/hq2_tests.rs` |
| IMPACT | `mgras rec FILE` ... `mgras rec off` | `MGRAS_REPLAY=FILE cargo test --release --lib --features gr4-jit mgras_replay_file -- --nocapture` (`MGRAS_REPLAY_BOARD=high\|max`, `MGRAS_REPLAY_NOCHECK=1`) |

`tools/rex3-rec.py` and `tools/mgras-rec.py` inspect and slice recordings.
Small captures that pin a fix go in the board's `testdata/` as regression
tests. Record the commit a capture was made at: a recording made before a
fix can stop matching after it.

Guest-side benchmarks lie when the board is the bottleneck: board-bound
runs slow the guest clock (`clanker/mgras/guest-gl-benchmarks-miss-board-time.md`).
Time replays on the host instead.

## 8. Driving input

- **Monitor, any PS/2 machine**: `ps2 mouse <dx> <dy> [buttons]` (1 left,
  2 right, 4 middle; send the press, then a packet with 0 to release),
  `ps2 type <text>`, `ps2 enter`, `ps2 esc`. Moves are relative and go
  through X pointer acceleration: push into the top-left corner with a few
  large negative moves, then step to the target in small ones.
- **In the guest, XTest**: `xclick NAME [BUTTON [X Y]]` or `xclick NAME key
  KEYSYM` clicks at exact window coordinates in the first window whose name
  starts with NAME; `xresize NAME W1 H1 W2 H2 [COUNT] [DELAY_MS]` resizes
  repeatedly. Build them with `sh irix.sh xclick xresize`.

Demos live in `/usr/demos/General_Demos/<name>/<name>`. Some wait for input
(`ideas` sits on its title screen until clicked). Their sources were
published on opengl.org: read them before guessing what a demo sends. Test
demos one at a time.

## 9. Gotchas

- GL state lives per context; a trace that starts mid-session has no context
  switch for the running client.
- A GR2 client spinning on `version` bit 0 waits for a FIN3 (Finish) we did
  not answer; bit 1 is FIN2 (kernel context switch). See `clanker/gr2/`.
- GR2 window clipping arrives as `0x1E5` only when switching into a context
  whose clip changed: start the trace before moving the covering window.
- Several GL clients share IMPACT texture memory through the texture
  manager's page saves and restores; a bug there damages every client.
- Board-specific notes: `clanker/rex3/`, `clanker/gr2/`, `clanker/mgras/`.
  Check them before re-deriving anything.
