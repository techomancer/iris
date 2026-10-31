# Odyssey (VPro) public references: two console-level drivers, no GL stream

Status: survey, 2026-10-25 (before the model landed; see bringup.md). This note
records what public code exists for the board so the Octane/Tezro work does
not re-search it. Verify the repositories before relying on details; the
T2 pages and patch trees move.

## The board

Odyssey is SGI's VPro graphics (V6/V8/V10/V12): a dual-chip OpenGL engine,
Buzz (geometry, transform, raster) and PB&J (video controller, gamma). It is
the only graphics option on Tezro and Fuel and the late option on Octane
and Octane2. There is no CPU-visible linear frame buffer; everything goes
through a command FIFO (CFIFO) and a data FIFO (DFIFO). The Octane2 and the
Tezro/Fuel cards differ in XIO connector but not, as far as the public code
shows, in programming model.

## What exists in public

| Source | Licence | Covers |
|---|---|---|
| T2 SDE kernel patch `4006_all-add-odyssey-driver.patch` | GPL | fbdev driver: init sequences, FIFO protocol, 2D accel |
| T2 SDE kernel patch `4007_all-add-odyssey_early.patch` | GPL | early boot console on the same hardware |
| `rxrbln/xf86-video-vpro` (GitHub) | MIT-style (check `COPYING`) | X.org DDX: shadow FB upload, fills, lines, copies, HW cursor, modesetting started |
| OpenBSD `odyssey(4)` (Joel Sing) | BSD | wscons console driver; independent implementation of the same init and FIFO path |
| IRIX PROM textport and kernel/X/GL binaries | proprietary | the only source for the full GL command stream |

All of the Linux and X code was developed on Octane/Octane2 (IP30). The
Kconfig entry also lists IP35, but no Linux IP35 port has ever existed, so
nothing has been tested on a Tezro or Fuel.

### The Linux fbdev driver

Lives in T2's `architecture/mips64/package/linux/` against Linux 6.11
(the generic `package/kernel/linux/` directory holds only IP32 and xtalk
hotfixes). About 1,000 lines: `drivers/video/fbdev/odyssey.c` plus
`include/video/odyssey.h`.

Lineage, from the copyright block: Stanislaw Skowronek (skylark) 2005,
Johannes Dickgreber 2007-2009 (platform-driver conversion), Joshua Kinard
(Gentoo) 2011-2015 (fixes), René Rebe 2018-2025 (rebase and revival). It is
Skylark's reverse-engineered driver carried forward, not a rewrite.

What it implements:

- Probe through `xtalk_get_swin()`; one 0x410000-byte MMIO window.
- Five init sequences: Buzz GFE, Buzz transform, Buzz raster, PB&J video
  controller, PB&J gamma. Mostly magic constants the header says came from
  reverse-engineering IRIX.
- CFIFO/DFIFO writes with wait helpers; status from `ODY_STATUS0` and
  `ODY_DBESTAT`.
- One fixed mode: 1280x1024, 24 bpp TrueColor (packed 32 bpp).
- Accelerated fillrect (COPY and XOR as GL logic ops), copyarea, and
  imageblit (8 bpp PIO, a 1 bpp stipple fast path for 8x16 glyphs, a slow
  1 bpp PIO fallback).
- An mmap of the raw register window for userspace; accel paths back off
  while a user holds the device open.

What it does not implement: cursor, DMA, blanking (`TODO`), `fb_read` /
`fb_write` / `fb_ioctl`, any direct frame-buffer mapping (`smem_len` is 0).

### The X.org driver

`xf86-video-vpro`, published 11-12 July 2026, six commits, all by René Rebe.
The README says the FIFO engine in `src/vpro_accel.c` was ported from the
fbdev driver and the IRIX Odyssey PROM textport code. It renders to a
host-side shadow frame buffer and uploads dirty regions through the FIFO.
Listed: hardware bring-up, solid rectangles and lines, pixel upload,
screen-to-screen copy, a hardware cursor, modesetting begun. Same single
1280x1024 mode. No release notes or user reports; treat "works" as
unverified. The T2 Octane page dated September 2025 still says no VPro X
driver exists, so it predates this repository.

### Mainline Linux

Octane IP30 platform support (HEART, Xtalk, Bridge, IOC3) has been in
mainline since 2020 (Thomas Bogendoerfer). Only the Impact and Odyssey
frame-buffer drivers are out of tree. T2's statement that IP30 is "not in
the vanilla kernel" is stale.

## What this buys an emulator, and what it does not

Enough for a textport: the init sequences, the FIFO handshake and the
register window are what the PROM and a console driver exercise, and three
independent implementations (Linux, X, OpenBSD) agree on them. A serial-
console Octane or Tezro plus a PROM textport is reachable from these sources.

Not enough for a desktop: the IRIX X server and libGL drive the GL pipeline
(vertex, lighting, texture, pixel ops) that none of the public drivers touch.
That part is a reverse-engineering job on the scale of IMPACT or larger, from
the IRIX driver, X DDX and libGLcore binaries, the same way `ignore/gr4` was
built. Expect the raster and display blocks to be the tractable part and the
geometry/transform block the hard one.

## Rules for using these sources

- Linux code is GPL and the X driver is a separate licence: take register
  offsets, bit positions, FIFO token layouts and the order of init steps;
  do not copy code or comments. Same rule as for SGI sources
  (`clanker/CLANKER.md`).
- OpenBSD's driver is the friendliest licence and an independent second
  reading of the same hardware; when the Linux constants look wrong, diff
  against it.
- Hardware facts recovered from IRIX binaries go into a local `ignore/`
  header set like `ignore/gr2` and `ignore/gr4`, not into `clanker/`.

## Links

- T2 Octane page: https://t2linux.com/hardware/desktop/Sgi/Octane/
- T2 mips64 kernel patches: https://svn.exactcode.de/t2/trunk/architecture/mips64/package/linux/
- Odyssey fbdev patch: https://svn.exactcode.de/t2/trunk/architecture/mips64/package/linux/4006_all-add-odyssey-driver.patch
- Odyssey early console patch: https://svn.exactcode.de/t2/trunk/architecture/mips64/package/linux/4007_all-add-odyssey_early.patch
- X.org driver: https://github.com/rxrbln/xf86-video-vpro
- Impact X.org driver (same author's fork): https://github.com/rxrbln/xf86-video-impact
- OpenBSD odyssey(4): https://man.openbsd.org/odyssey.4
- Mainline IP30 patch (2019): https://lkml.rescloud.iu.edu/1910.1/01791.html
- Related: `clanker/cpu/mp-llsc-is-a-host-compare-exchange.md` (Octane MP
  prerequisites), `docs/Octane/` (Xbow, Bridge, Xbridge ASIC specs).
