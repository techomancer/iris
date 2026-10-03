---
name: gr2-gl-loop
description: Autonomous GR2 (XZ/Extreme) OpenGL bring-up loop on IRIX 6.5.22 - boot IRIS, build and run test/gltest/glprim in the guest over telnet, capture gr2 traces around each run, and verify rendered pixels. Use when iterating GL pipeline configurations (primitives, shading, vertex/colour forms, depth, scissor, viewport) against the GR2 HLE.
---

# GR2 OpenGL test loop

Drive one GL configuration at a time through the real IRIX GL stack
(libglcore EXPRESS -> HQ2 FIFO -> our HLE in `src/dev/gr2/{hq2,gl}.rs`) and
check the pixels. Each `glprim` run draws exactly one primitive, so its FIFO
trace is short and every option flips one piece of state or one address bit.

## Pieces

| What | Where |
|---|---|
| Test app | `test/gltest/glprim.c` (`--help` lists every option), built by `test/gltest/irix.sh` with MIPSpro `cc` |
| Guest shell | `tools/irix-sh.py` (telnet via host port 2323, root, no password; single-shot) |
| File exchange | built-in NFS: host `./shared` = guest `/shared` (mount with `tools/irix-sh.py mount-shared`). **noexec**: copy to `/tmp` before building/running |
| Monitor | iris MCP (`mcp__iris__run_command`), i.e. the monitor on 127.0.0.1:8888 |
| Token map | `ignore/gr2/HQ2.h` "OpenGL token addressing" (source of truth; add findings there) |

## Boot

1. `cargo build --release --features developer` (developer: monitor tooling;
   do not combine with `lightning`).
2. Start it with the Bash tool's `run_in_background: true`, as a command of
   its own (`target/release/iris > <scratchpad>/iris-run.log 2>&1`). A
   `nohup ... &` inside a chained command does not survive the command, and
   starting it right after killing the old process can fail: check
   `pgrep -a iris` before going on. (config: `iris.toml`, Indy R5000 + XZ,
   disk `scsi1.raw`.)
3. Start the CPU through the MCP: `run_command("cpu start")` (check with
   `get_status` if unsure).
4. `tools/irix-sh.py wait --timeout 900` returns once telnet answers with a
   login prompt (network and inetd are up; X/xdm come up around then too).
5. `tools/irix-sh.py mount-shared`.
6. GL needs a logged-in desktop (xdm's clogin holds the display otherwise).
   root autologin is set up: `/etc/autologin` = `root`, and the ONE-SHOT flag
   `/etc/autologin.on`, which clogin deletes when it logs in. Re-create it
   (`touch /etc/autologin.on`) every boot before shutdown. The session takes
   ~1 min after telnet comes up: wait until `ps -ef | grep 4Dwm` shows it.
7. The screen saver (`haven` + `/usr/sbin/bongo`, octahedra) takes over the
   whole screen after 10 min idle and every probe reads black. The desktop
   re-applies its saver settings after `/root/.sgisession`, so run
   `/usr/bin/X11/xset s off` once the session is up, and check
   `xset q | grep timeout:` = 0.
8. Stopping iris: kill it by its exact pid (`ps -eo pid,args | grep
   '[t]arget/release/iris$'`). `pgrep -f target/release/iris` also matches
   the shell running the command and kills it.

`iris.toml` currently writes the disk image directly (no overlay): never
kill iris while IRIX is running (see Shutdown).

## Build glprim in the guest

```
mkdir -p shared/gltest && cp test/gltest/{glprim.c,main.c,irix.sh} shared/gltest/
tools/irix-sh.py run 'mkdir -p /tmp/gltest && cp /shared/gltest/* /tmp/gltest/' \
                     'cd /tmp/gltest && sh irix.sh glprim 2>&1'
```

Every session gets `DISPLAY=:0.0`, `/bin/sh`, cwd `/tmp`. The exit status of
`irix-sh.py run` is the last command's status (124 = timeout/connection).

## One test case

Preferred: `tools/glprim-case.py CASE [--probe gx,gy[=rrggbb]]... [--notrace] -- <glprim options>`
does all of the below (trace, run, window origin, probes in GL window
coordinates, fbdump, crop of the window to `$GLPRIM_OUT/CASE.png`, trace off,
kill glprim) and exits with the number of failed probes. It talks to the
monitor on 127.0.0.1:8888 directly (works alongside the MCP). Stage the
sources first (`cp test/gltest/* shared/gltest/`, then copy to /tmp in the
guest and `sh irix.sh glprim`).

By hand:

```
run_command("gr2 trace /tmp/gl-<case>.log all")        # MCP
tools/irix-sh.py run 'cd /tmp/gltest && ./glprim <options> --hold 4000'
run_command("gr2 fbdump /tmp/gl-<case>")                # during the hold, or
run_command("gr2 pix <x> <y>")                          # spot checks (display coords)
run_command("gr2 trace off")
```

Run the capture commands while `--hold` keeps the window up (launch the
guest command in the background with `&` if you need to sample during the
hold). `glprim` prints its configuration and its **real window origin**
(`window origin OX,OY size WxH`; the window manager moves windows). A GL
window pixel (gx, gy) is display pixel `(OX + gx, OY + H - 1 - gy)`.

Default geometry (unit square scaled to the viewport, glOrtho 0..w, 0..h):
triangle (0.2w,0.2h) red, (0.8w,0.2h) green, (0.5w,0.8h) blue. Flat shading
uses the provoking vertex (last vertex for triangles/strips/fans/quads, the
FIRST vertex for GL_POLYGON). Clear default is pink (1, 0.5, 0.75) =
`rgb=ff80bf`.

## Verify

- `gr2 pix x y` prints the VRAM word, DID, XMAP mode and composed rgb.
- `gr2 fbdump DIR` writes VRAM planes and the composed screen as PNGs; view
  them with the Read tool.
- Expected values: compute from the GL spec (pixel centres at +0.5, fill rule
  top-left style; flat = provoking vertex colour; smooth = barycentric
  interpolation, allow +-2 per channel). Check interior, each edge's first
  in/out pixel, and a pixel just outside the window (must be untouched).
- In the trace, `HQ exec` lines show the HLE's view: `VERTEX (...) -> window
  (...)`, `GL triangle ...`, `GL_CLEAR ... rect`. Unknown tokens appear as
  `tokNNN` / `(N words, not implemented)` with their modifier bits decoded
  (`V3|USEV:VERTEX`, `C3:COLOR`, `LOADV:...`).

## Iterate

`--scene depth|stencil|alphatest|blend` draws small multi-primitive scenes
with known images (see the comment above `draw_scene` in glprim.c); verified
probes are in the history of this skill's cases (depth: (170,150) red,
(230,150) blue; stencil: (200,120) blue, (20,20) pink; blend: (200,120)
800080, (300,120) 8040df; alphatest: (200,200) ~1c1cc6, (90,62) pink).

Change one option at a time. Suggested order: clear only (`-p none`), flat
triangle, `--smooth`, `--mono`, every `-p` primitive, `--vtx 2f/4f/2i/3i/2s/3d`,
`--color 4f/3ub/4ub/once`, `--viewport`, `--scissor`, `--cull`/`--cw`,
`--polymode`, `--persp`, `--depth`, `--db`. When a new token or format shows
up, decode it against `ignore/gr2/libglcore/decomp/EXPRESS` (struct offset
X = FIFO index (X - 0x2000)/4), record it in `HQ2.h` with the emitter as
citation, implement in `src/dev/gr2/gl.rs`, add a replay test in
`src/dev/gr2/hq2_tests.rs` built from the trace (see `glprim_triangle`), and
re-run the case.

## Offline replay of real streams

A guest restart costs ~6 minutes, so capture once and iterate offline:
extract the HQ2 FIFO writes of a case (from the window setup `0x1e5` to the
`GL_FINISH` after the draw, 2D command groups 0x12C..0x1EA dropped) into
`src/dev/gr2/testdata/glprim_<scene>.trace` ("index value" hex per line) and
replay it in a unit test with `replay_trace()` (hq2_tests.rs); the window
origin is the first two words after `1e5`. See the lighting/fog tests.

## Demos

`/usr/demos/General_Demos/<name>/<name>` (atlantis, ideas, buttonfly, flight,
powerflip, ...). GL state is swapped per context on GE_HQMSAV; still, test demos one at a
time.

Demos that wait for input need it injected: `ideas` sits on its title
screen until clicked (and needs another click to restart after its
animation), so a run without a click only ever draws the title.

## Driving input (mouse, keyboard)

Two ways, both scriptable:

- **Monitor, `ps2`** (any PS/2 machine: Indy, Indigo2/IP28):
  `ps2 mouse <dx> <dy> [buttons]` sends one relative mouse packet (buttons:
  1 left, 2 right, 4 middle; send a packet with the button, then one with 0
  to release), `ps2 type <ascii text>` types a string, `ps2 enter` presses
  Enter, `ps2 status` shows the controller state. Moves are relative and go
  through X's pointer acceleration: to reach a spot, first push the pointer
  into the top-left corner with a few large negative moves, then step to the
  target in small moves.
  ```
  run_command("ps2 mouse -500 -500")  # corner (repeat to be sure)
  run_command("ps2 mouse 40 30")      # small steps to the target
  run_command("ps2 mouse 0 0 1")      # press left
  run_command("ps2 mouse 0 0 0")      # release
  ```
- **In the guest, XTest**: `test/gltest/xclick.c` (`xclick NAME [BUTTON [X Y]]`
  or `xclick NAME key KEYSYM`) clicks or presses a key at exact window
  coordinates in the first window whose name starts with NAME;
  `test/gltest/xresize.c` (`xresize NAME W1 H1 W2 H2 [COUNT] [DELAY_MS]`)
  resizes a window repeatedly. Build with `sh irix.sh xclick xresize`.
  Precise, no acceleration, but needs a telnet session.

Demo sources (ideas, atlantis, ...) are published on opengl.org: read them
before guessing what a demo sends.

## Shutdown

```
tools/irix-sh.py run 'touch /etc/autologin.on; sync; shutdown -p -g0 -y'
```
Then wait about a minute (the developer build does not exit on power-off),
check `/tmp/iris-run.log` / the monitor for the power-off, and only then kill
the iris process.

## Gotchas

- The host shell's `cd` may be aliased; use `builtin cd` or absolute paths.
- `/shared` is noexec: always build and run from `/tmp`.
- Stock IRIX has no bash (scripts must be `/bin/sh`), no `pkill`, and
  `chmod` wants `a+x` rather than `+x`.
- GL state is per context: GE_HQMSAV (0x1F0, kernel Gr2PcxSwap) swaps the
  HLE's GlState by context id; state 0 = new context = defaults. A trace
  that starts mid-session has no switch for the running client: replays keep
  the live state for it.
- MIPSpro is C89-ish: declarations at block start, no `<stdint.h>`.
- A GL client that waits forever on `version` bit 0 (FIN3) is waiting for a
  Finish (`0xA3`) we did not answer; a stuck client on bit 1 is FIN2 (kernel
  context switch).
- Window clipping arrives as 0x1E5 only on the switch into a GL context after
  its clip changed (see rules/gr2/gl-window-clipping.md). To trace clipping,
  start the trace before moving the covering window.
