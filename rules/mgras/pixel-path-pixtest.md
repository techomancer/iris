# IMPACT pixel path: pixtest and what it settled

`test/gltest/pixtest.c` exercises glDrawPixels, glReadPixels, glCopyPixels,
glTex(Sub)Image2D, glCopyTex(Sub)Image2D, glGetTexImage, glBitmap, stencil
and the transfer state (PixelStore, PixelTransfer, PixelMap, zoom) one GL
call a case. Each case checks its own result. As of 2026-10-13 the IP28
MaxImpact (6.5.22) result is 348 pass, 0 fail, 12 XFAIL, 108 skipped (the
skips are format/type pairs GL rejects).

Run: `sh irix.sh pixtest` in the guest, then `./pixtest [--only S]
[--skip S] [--list]`. Record with `mgras rec F` and read the
recording with `tools/mgras-pixdecode.py F --ucode shared/ucode/mgras_ge.u
--results pix.txt --summary [--case N|NAME]`. The summary lists each
operation's GE module chain with its states, plus the raster registers.

The hardware map is in ignore/gr4: GE11.h "IMAGING ENGINE" (modules and
states), HQ3.h "PIXEL OPERATIONS" (tokens, payloads, readbacks, libGLcore
deviations), RE4.h XFRMODE (fields, zoom).

Gotchas that cost time:

- **XFAIL means libGLcore does it on real hardware too**, checked in its
  code (ignore/gr4/libglcore). Do not "fix" these in the emulator:
  - REPLACE is sent as MODULATE.
  - Reads get no pixel transfer.
  - glBitmap drops `skip_pixels % 8`.
  - glGetTexImage(GL_ALPHA) fails.
  - Packed 5551, 1010102 and 332 reads come back in a reversed layout.
  - glTexSubImage narrower than 32 texels fails (next item).
- **glTexSubImage2D INVALID_VALUE**: mgrTM_TexSubImage2D_setup wants
  width >= 32 (or the full width), with the right edge on a multiple of 32
  (or at the texture's edge). Test with legal sizes. When a guest app's
  sub-image updates "do nothing", check this first.
- **PIO read order is set per client by XFRMODE bit 18.** Without it
  (libGLcore), RGB(A) comes in the reverse of the DMA order. With it (the
  X server's XGetImage, 0x40080), it comes in the DMA order. Check
  XGetImage after changing any read path: `test`-style probe xgetpix
  (XGetImage of a root pixel) against `mgras pix`. The unit test is
  `pio_read_order_follows_rgborder_and_bit_18`.
- **Index outputs wrap, colours clamp**: stencil and colour-index draws
  keep the low bits of the pipeline's value (shift/offset past 255).
- **Texture loads into L/LA/A/I/RGB** read RGBA from the host and end in
  COMPRESS. The input is 4 components, not the output count.
- The decompiler's function boundaries in libglcore/decomp are wrong in
  places (labels lag by a function). Read `asm/` when the C makes no sense.
