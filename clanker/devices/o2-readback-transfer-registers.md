# O2 read-back: a frame-buffer-to-linear pixel transfer, by the spec's registers

Status: rule (fixed 2026-10-24, branch o2-readpixels).

**Symptom.** `glReadPixels` on the O2 returned zeros; the engine logged
`xfer unaligned linear end (src 0/4, dst 4/4)`.

**How IRIX reads pixels.** The GL library mmaps a 128 KB scratch buffer from
`/dev/zero` and calls `ioctl(/dev/opengl, 0x5605)`; the kernel maps the buffer
into linear A (`TLB_LINEAR_A`) and runs a pixel transfer from tiled buffer 0
(the frame buffer) into linear buffer 4, then the library copies the pixels
out. No aperture read (0x1000_0000 stays unused) and no register read.

**The registers (CRIME 1.5 table 7-3), as IRIX uses them for this:**
- Primitive 0x0300_0000: rectangle (op-code 3), edge type 0 = left to right,
  bottom to top. The X server's transfers use edge type 2 (top to bottom).
- DrawMode has the GL bit (22), so the corners are the GL vertices at
  0x080..0x08c (13.6 fixed point, window coordinates plus the 4096 guard),
  both ends included. The X vertices at 0x070/0x074 hold stale X values.
  A 1x1 read sends a 2x1 rectangle; a 4x2 read sends exactly 4x2.
- 0x0a0 `PixelXfer.src.addr`: the first source pixel, window-relative, rows
  counted from the top. The source window offset (0xf020f020 for a window at
  (32,32): origin - 4096) places it on the screen modulo 4096.
- 0x0a8 / 0x0ac `src.xStep` / `src.yStep`: source steps in pixels (1, 1),
  in the edge type's directions (bottom to top = up the screen).
- 0x0b0 `dst.linAddr`: byte address in the linear buffer.
- 0x0b4 `dst.linStride`: bytes from a row's last pixel to the next row's
  first, the convention of a linear source's step: 4 for IRIX GL's 32-bit
  reads, 1 for its X server's 8-bit screen-to-linear copies, packed rows in
  both.

The old code read 0x0a8/0x0ac as the linear end's byte steps and placed the
rectangle by the X vertices plus the window offset, which clamped to zero
width. Linear-to-frame-buffer transfers (the X server's) keep the old path.

**Repro.** A GLX program that clears a 64x64 window, draws quads of known
colours and reads single pixels and a 4x2 block back with `glReadPixels`.
`--startup "re rec FILE"` records the register writes; the read-backs are the transfers into buffer 4 with DrawMode
bit 22 set.
