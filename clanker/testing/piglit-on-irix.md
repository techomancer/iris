# piglit on IRIX: comparing the emulated boards with real ones

Status: howto, 2026-11-05.

piglit's `irix` branch (local checkout `ignore/piglit`) builds the
fixed-function subset of piglit with MIPSpro and runs it with a Bourne
shell driver; its `irix/README.md` is the manual. This note is how to use
it against IRIS.

## The lists

`irix/lists/{vpro,impact,express,newport}.list`, from the boards' glxinfo
under IRIX 6.5.22: VPro 421 tests (OpenGL 1.2), the three OpenGL 1.1
boards 341 each. Regenerate with `irix/gen/gen.py` after changing
`irix/gen/boards.py` (needs a CMake build tree of piglit; see the README).

## Getting files in and out

- Emulated guest: put the tree in the NFS share (`./shared/piglit`), then
  copy it to `/usr/tmp` in the guest (`/shared` is noexec). Results come
  back the same way (`tar cf /shared/x.tar ...`).
- Real machine on the LAN: `tools/irix-sh.py --host H --port 23 push` a
  uuencoded `.tgz` (IRIX has `uudecode`, `gzip`, `tar`), `pull` a
  uuencoded result tarball back. The IRIX tty cuts a line at 255
  characters (`clanker/irix/telnet-lines-are-cut-at-256.md`): push scripts
  for anything long. A machine whose clock is far off makes `make` rebuild
  everything if files keep their mtimes: unpack with `tar xmf`.
- Build once on the fastest machine (a real Octane builds the whole set in
  about five minutes with `smake -J 2 -k`) and run the same binaries
  everywhere: n32 MIPS IV runs on R5000, R10000 and later.

## Running and comparing

    cd /usr/tmp/piglit/irix && ./piglit-run.sh -t 300 lists/vpro.list
    ./piglit-summary.sh results/vpro-octopus results/vpro-IRIS

Use `-t 300` in the emulator (some tests take minutes there). Compare the
emulated board with the same model of real board: a V12 offers 12-bit
visuals a V10 does not, which changes which visual a test gets. Set
`[graphics] board = "v10"` to compare with a 32 MB V10.

Tests that draw only open an override-redirect window at (0, 0); xdm's
login box does not cover it, but some images' xdm grabs the server until
a console login (then even glxinfo hangs).

## Where it stood (2026-11-05)

Real V10 (octopus, 32 MB, IRIX 6.5.22): 146 pass, 38 fail, 19 crash (X
errors from the IRIX server's GLX), 216 skip, 1 timeout, 1 warn. Emulated
V10 (IRIX 6.5.30): 94 pass before the fixes of that day, 145 after (window
colour reads, edge flags, stencil-or-index reads, two-sided lighting).

Still failing in the emulator, passing on the V10:
`drawbuffer-modes` (`clanker/odyssey/front-and-back-arrives-as-front.md`),
`drawpixels-depth-test` (`clanker/odyssey/drawpixels-skips-fragment-operations.md`),
`gl-1.0-texgen` (sphere map, not modelled), `two-sided-lighting-separate-specular`
(separate specular colour, not modelled), `read-pixels-after-display-list`
(a texture uploaded through the pixel maps), `draw-elements-vs-inputs`
(a colour 0.005 off), `glx-multithread`, one `dlist-beginend` subtest.

Passing in the emulator, failing on the V10: `glx-make-current`,
`glx-multi-context-single-window`, `long-line-loop`, `user-clip-all-planes`,
`hiz`, `line-smooth-stipple`. Most run on 4- or 5-bit visuals, where the
board probably dithers and the model does not; unconfirmed.
