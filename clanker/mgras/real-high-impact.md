# Comparing the emulator with a real High IMPACT (purple)

Status: howto, 2026-11-08.

**The machine.** `purple` on the user's LAN (ask for its address): Indigo2 IP28,
R10000 195 MHz, 1 GB, High IMPACT with the TRAM option (gfxinfo
"IMPACTPC", 1 GE, 1 RE, 4 TRAMs; MGRAS rev 3, RA 5, GE11 rev B, RE4 rev A,
PP1 rev A), Presenter flat-panel adapter, IRIX 6.5.22 with MIPSpro, X at
1024x768. Telnet as root; the password is the user's (ask, never write it
down): `IRIS_ROOT_PASSWORD=... tools/irix-sh.py --host ADDRESS --port 23
run ...`. `ignore/ip28/ip28_6522_realhw.raw` is a clone of its disk.

**Setup on the real machine.**
- Share: the user's NAS, `mount -o vers=3 NAS:/share /nas` (the NAS, not
  the router), then `ln -s /nas/iris-purple /shared`; the host sees the same directory
  through its own mount of the NAS. Root writes are not squashed.
- Kill the screen saver first (`haven -n bongo`, which runs octahedra over
  every window) and `xset s off`: covered windows fail every front-buffer
  read (blendtest 440 failures, pixtest 280).
- Build test/gltest there with `sh irix.sh` (MIPSpro, fast), then
  `sh glsuite.sh TAG` as for the emulator (TESTING.md); results land in
  /shared/results/TAG. Long jobs under `nohup`: a dropped telnet session
  kills the script. Command lines over 219 characters: push a script.

**The emulated twin.** A copy of `ignore/ip28/iris.toml` with
`board = "highimpact"`, `overlay = true` on both disks, its own NVRAM
copies, `[nfs] shared_dir` at the NAS directory (both machines then see the
same /shared), 512 MB. Not 4 x 256 MB banks: bank 3 comes up with a 16 MB
limit on IP28. The cloned image needs, after every boot, from the
desktop's winterm (`ptr click`, `key type`):
`ifconfig ec0 inet 192.168.0.2 netmask 0xffffff00; killall routed; route
add default 192.168.0.1; mv /etc/resolv.conf /etc/resolv.conf.real`
(the clone keeps purple's static LAN address; routed rewrites the routes a minute
later; resolv.conf points at the LAN router, whose reverse lookups stall
telnetd logins), and the real root password.

**Identity.** `test/gltest/bdinfo` dumps the GFX_GETBOARDINFO record.
Against the real board the emulated High IMPACT differs in board revision
(byte 0x31: 3 vs 0), RA revision (0x32: 5 vs 7), GE11 revision (0x38: 2
vs 1), RE4 revision (0x39: 0 vs 1), PP1 revision (0x3B: 0 vs 1) and the
flat-panel probe (0x49, the Presenter). Not changed yet: the GE rev B
microcode and the revision checks in libGLcore may take other paths.

**Findings so far** (each with its own note or code comment):
- Detail textures: `detail-textures-follow-the-drivers-table.md`.
- Subtract blends: `subtract-blend-follows-the-real-board.md`.
- GL_CURRENT_RASTER_DISTANCE is the raster vertex's eye z, not divided by
  w (`test/gltest/rdist.c`; glstate XFAIL).
- GE memory in a fresh context (`gepeek ge:0-0xFFF`, twice): words
  0x200..0x241 are [value, RSS register] pairs, 0x000..0x031 RSS command
  words; 0x1D0..0x1D6 (the CI copy's mask state) hold the previous
  context's leftovers; 0xDBB = 4, 0xDBE = 2, 0xDBF = 0x3F, 0xDC0 = 0xFFF
  every time. The emulator answers the words glquery models, and a new
  context's pixel memory holds those four ERAM words (`pixpipe::FRESH`;
  the element type replaced the old guess).
- pixtest differences, real board failing where the emulator passes
  (open): 8_8_8_8 and 4_4_4_4 reads (component order), ABGR draws of
  8_8_8_8 / 10_10_10_2 (alpha 255), ALPHA and INTENSITY textures loaded
  from RGBA (all ones), gettex RGB/RED/GREEN/BLUE/LUMINANCE (zeros), zoom
  0.5; the CI8 read XFAILs pass on the board (no planes 11:8 leak).
- Host DMA windows: `dma-past-the-mapped-window.md`.
- Quads: the board splits GL_QUADS / GL_QUAD_STRIP along the 1-3
  diagonal (v1-v2 for a strip's quad); fixed in `gl.rs` `polygon`.
- Open, measured (glsuite 2026-11-08, real `purple` vs emu-cpu):
  - Fog: fixed 2026-11-09 (`fog-per-pixel-through-a-table.md`; a table
    over window depth, pixel pairs; `real_board_fog`, `real_board_fogtest`).
  - Lines and points: fixed 2026-11-09 (`gl-lines-follow-diamond-exit.md`);
    aliased cases exact (`real_board_lines`), antialiased lines and
    points fitted (`real_board_smooth_lines_and_points`), including the
    point box the GE latches at glPointSize (glprim's AA points).
  - Smooth triangles, strips, fans, polygons, lighting, blends: off by one
    level at most (rounding).
  - texsweep: ALPHA and INTENSITY textures (4/8/12/16) read back as the
    first texel everywhere on the board (0x11111111; pixtest sees all
    ones); L12A4, L12A12, L16A16 read back in another byte order.
  - CI reads as GL_BYTE: the board returns index & 0x7F (XPASS), we return
    the truncated 32-bit index; XFRMASKLOW / HIGH (0x156/0x157) are not
    modelled.
  - vissweep: 4,4,4,4 db pbuffer passes on the board, fails here.

**Piglit** (`lists/vpro.list`, 2026-11-08/09): the board 130 pass, 41
fail, 24 crash, 224 skip; the emulated twin (results/emu2, before the fog
and flat-line fixes) 116 pass, 54 fail. Passing on the board, failing
here: hiz stencil reads x2 and masked-clear (a stencil index read as
GL_UNSIGNED_INT comes back 0x05050505 for 5: the byte widened like a
colour), edgeflag x3 (edges drawn that the flags turn off),
drawbuffer-modes (GL_NONE still writes the front), rastercolor, texgen
(sphere map), no-op-paths, dlist-color-material (glGetMaterial after
a display-listed colour material), clear-accum, stencil-drawpixels,
levelclamp / lodclamp / lodclamp-between, streaming-texture-leak
(timeout); fixed since: line-flat-clip-color, fog-linear-floor. Failing on
the board, passing here: dlist-bitmap, long-line-loop,
scissor-depth-clear, scissor-stencil-clear, infinite-spot-light. The
twin's network used to die ~20 minutes after boot
(`clanker/net/nat-readopts-the-gateway-when-the-guest-moves-subnet.md`).

**Measured 2026-11-09** (results/purple-lf on the share): `linetest` in
every mode (default, profile, vprofile, slopes, vslopes, fine, points,
pointsdc, pointsal), `pttest` (results/purple-lf/pt), `fogtest` and `fogtest 8`, glprim smooth-point variants
(ptvar). The fits are in the notes above; the board data the unit tests
use is in `src/dev/mgras/testdata/real_high_impact_*.txt`.

**Data** (on the share, `results/`): purple/ (the real suite),
emu-cpu/, gepeek-*.txt, bdinfo-*.txt, dettex/ (sweeps, dumps), blend/.
