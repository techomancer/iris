# What the GL library sends for glReadPixels, glCopyPixels and glCopyTexImage

Status: traced 2026-11-07 on IRIX 6.5.30 with pixtest (`test/gltest/pixtest.c`,
the `read/`, `copy/`, `depth/`, `stencil/`, `buffers/`, `copytex/` cases) on
a V12 at 1280x1024, 8:8:8:8 double-buffered visual with depth and stencil.
Recordings of the three runs are in `ignore/recordings/gpu-cases/`
(`pixtest-read.rec`, `pixtest-copy.rec`, `pixtest-copytex.rec`), where the
CPU/GPU parity test replays them.

## glReadPixels

- A plain read of colour sends no board command: a pixel DMA to the host
  from the read drawable (`DMA_DESC_UPLOAD 0xe01`, descriptor `dest
  0x30000019`, format word `0x12633042` for RGBA8 bytes, `0x1263300a` with
  a skip for RGB bytes, `0x10433012` for depth and stencil). On the GPU
  unit this is the one readback that cannot go away: the pixels leave the
  board.
- A read with pixel transfer (scale and bias) is a copy first: `COPY
  0x2031` from the drawable into a scratch surface at `0x16b800` under
  `TEX_SCALE_BIAS 0x203b`, then the DMA to the host from the scratch
  (`dest 0x16b800`, format `0x8002500b`, eight-byte lanes). Colour-index
  map reads do the same (`read/.../map256` is an XFAIL for the map itself).

## glCopyPixels

- `GL_COLOR` at zoom 1, 1 or 1, -1: inside a `GL_BEGIN 0xa` image: the
  image box (the destination rectangle, with its direction), a
  `VERTEX_2I 0 0`, then one `COPY 0x2031` from the read drawable to the
  draw drawable, same field, then `GL_END`. The library sets `CLIP_BOX
  0x81d` to the destination rectangle around it and `TEX_SCALE_BIAS` to the
  transfer (identity when none).
- Any other zoom (2, 2; -1, 1; fractional): a copy into the scratch at
  `0x16b800`, then an image draw from it (textured, `TEX_MODE 0x2400`).
- `GL_DEPTH` and `GL_STENCIL`: no copy block at all. The library reads the
  rectangle to the host (a DMA get) and draws it back as a depth or stencil
  image.
- Under the depth test, a colour copy comes inside an image whose box
  carries the raster depth (operands 6 and 9), exactly as a depth-tested
  glDrawPixels does: the copied pixels go through the fragment back end.
  The model did not do this before 2026-11-08 (pixtest
  `depth/colour-copy/less` failed on both raster units); it now routes them
  as `Board::image_put` routes an image's pixels.
- Front to back (`buffers/copy-front-to-back`): one copy within the
  drawable, source field 12 (word 4 bit 31 set: the front) to field 4.

## glCopyTexImage2D, glCopyTexSubImage2D

One `COPY 0x2031` from the read drawable into the texture's level in board
memory (`0x16b800...`, field 0, four bytes), inside an image whose box is
reversed in y, with a `CLIP_BOX` of the texture rectangle. The sub-image
copies name the sub-rectangle's offset as the destination (x, y). No DMA.

## The copy's direction bits and the image box

Word 7 bits 31 and 30 of the copy say the source is walked in decreasing x
and y; the image box says which way the destination is walked (reversed
where x2 < x1 or y2 < y1). Where the two disagree on an axis the copy
mirrors along it. The cases that establish it:

| case | box | bits 31:30 | result |
|---|---|---|---|
| copy +40,+25, zoom 1,1 | forward, forward | 00 | straight |
| copy +5,+3 (overlapping, destination right of source) | x reversed | 10 | straight; the walk clears the overlap |
| copy -5,-3 (destination below) | y reversed | 01 | straight |
| copy +7,-2 | both reversed | 11 | straight |
| copy +10,+40, zoom 1,-1 | both reversed | 10 | y mirrored |
| glCopyTexImage | y reversed | 00 | y mirrored (the texture is flipped) |
| front to back | y reversed | 01 | straight |

Before 2026-11-08 the model mirrored y whenever one side was a drawable and
the other memory, and never otherwise (pixtest `copy/color/+10,+40/zoom1,-1`
failed). With no image open (the X server's copies) that older rule still
applies, since the box is not set.

## On the GPU unit

Since 2026-11-08 a copy the unit can take runs as the copy kernel
(`src/dev/odyssey/wgpu/shaders/copy.wgsl`, `GpuRaster::copy`): the
records before it are submitted, then a gather of the source into a
scratch buffer and a scatter into the destination (mirroring, the window's
clip list and id test, the scale-and-bias filter, the destination's codec,
raw byte moves under the window-id masks), all on the queue, with no
readback and no wait. What stays on the CPU: a depth-tested copy (its
pixels are fragments; the readback path with the fragment route), a copy
of more than 2M pixels, and a view the kernels cannot address. The quake1
timedemo's 16 copies are the kernel's; its only readbacks left are the
replay's own checkpoints. A glReadPixels still reads back its rectangle.

The scale-and-bias filter is now exact integer arithmetic on both units
(`Board::copy_post_filter`: byte x scale + 255 x bias in 3.12, clamped and
rounded, ties up), so the kernel and the CPU agree bit for bit; the old
f64 formula could round a tie either way.
