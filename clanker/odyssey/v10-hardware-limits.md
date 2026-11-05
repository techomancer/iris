# What a real V10 reports: visuals, limits, pbuffers, antialiasing

Status: measured 2026-11-03 on a real Octane2 (IP30, 2x R14000, V10
"VPRO/B/32", 32 MB board memory, IRIX 6.5.22 with the VPro patch, display
1024x768 at 60 Hz) with `glxinfo`, `gfxinfo -vv` and `test/gltest/vprolimits.c`
(build in the guest or on the machine: `cc -o vprolimits vprolimits.c -lGL -lX11`).
The machine is on the LAN; ask the user for its address and login.

## Visuals and pixel formats

- Colour index: 8 and 12 bits, single and double buffered.
- RGBA (single and double buffered, with and without 24-bit depth and
  8-bit stencil): 8-bit "RGB" (`r 8`, pseudo-colour class), 12-bit RGB,
  and true-colour 4:4:4:4 and 5:5:5:1 (16 bits a pixel), 8:8:8:8 and
  10:10:10:2 (32 bits a pixel). No 12/12/12/12 visual on this board and
  memory size.
- Accumulation: 16 bits a channel on the window visuals, 24 bits on the
  pbuffer-only fbconfigs (drawable type 0x6: pixmap and pbuffer).
- No multisample visual; `GL_SGIS_multisample` is not offered.

The model stores every RGBA window as RGBA8: 4:4:4:4, 5:5:5:1 and
10:10:10:2 windows are wrong in it (see the quad-pipeline plan,
milestone 9 and after).

## Limits (GL 1.2, an RGBA8888 pbuffer context)

| | |
|---|---|
| GL_MAX_VIEWPORT_DIMS | 3840 x 3840 |
| GL_MAX_TEXTURE_SIZE | 512 |
| GL_MAX_3D_TEXTURE_SIZE | 16384 (as reported) |
| GL_SUBPIXEL_BITS | 3 (as reported; the model snaps to 1/256 from the frames) |
| point size, line width | 0.5 to 10, granularity 0.125 |
| clip planes, lights | 6, 8 |

## Pbuffers

Every pbuffer-capable fbconfig reports GLX_MAX_PBUFFER_WIDTH/HEIGHT
1024x768 and 786432 pixels: the managed screen's size. 512x512 and
1024x1024 pbuffers are created and read back (a red clear reads
ff0000ff at the far corner); 2048x2048 and 4096x4096 fail with BadAlloc.
Pbuffers live in the board's framebuffer memory, sized by the display
mode and the memory, not by a fixed 4k target.

## Antialiasing in the driver

The VPro libGLcore (`/usr/gfx/arch/IP30ODSY32/libGLcore.so`) has, by its
symbol names: a hardware 24-bit accumulation path (`__glOdyHardAccum24`,
`hardLoadAccumulate24`, `sendBlenderColorTokensAccum24`, a scratch buffer
in board memory, `initAccumGfxImages`) and a software one
(`__glOdySoftAccum`, `softAccumulate`); smooth points, lines and polygons
(the raster words carry the smooth bits; the generic `__glAntiAlias*`
routines are there too). No multisample. Full-scene antialiasing on VPro
is the accumulation buffer, jittered; the model has no accumulation
buffer yet, so a hardware accumulate's commands are something to trace.
