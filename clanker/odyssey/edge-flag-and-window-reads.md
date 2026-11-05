# Edge flags are two command words; a window's colour reads use its display mode

Status: rule, 2026-11-05. Found with piglit on IRIX (ignore/piglit, irix
branch), comparing the emulated VPro with a real V10.

## glEdgeFlag

The library sends glEdgeFlag as one of two zero-operand words:
`0x2f00c02e` for a true flag, `0x3000c02e` for false (the false form for a
zero flag, read from the shipped libGLcore). The model named them `SYNC`
and only recorded them for the current-value query, so every edge was drawn
in polygon line mode: piglit's `gl-1.0-edgeflag`, `-edgeflag-const` and
`-edgeflag-quads` failed where the V10 passes. They are
`gl_cmd::EDGE_FLAG_TRUE`/`FALSE` now and set `assembly.edge`.

## Colour reads of a window

`glReadPixels` from a window, like a copy from one (the TyrQuake liquid
textures, `e376eae1`), names the packing it wants in the DMA header's
format byte, not the format the window's colour is stored in. A
10:10:10:2 window read as the 8:8:8:8 it asked for returned pure green as
(0, 0.247, 0.941) and spoiled about 40 piglit tests (scissor, clears,
masks, texturing, clipping: anything that probes a pixel). `dma_fields`
now takes a drawable read's codec from the display mode of the window at
the read's first pixel (`Board::drawable_codec`), as copies and the
scan-out do.

Rule: a read or copy whose source is a window gets the colour layout from
the window's display mode (0x2900 + main id), never from the format byte.

## Stencil or index: the window decides

A DMA read with header class 0 and field codes (0,0) is a stencil read of
a storage-2 window or a colour index read of a storage-0 window, and the
header alone does not say which. The model served every one as a colour
index; on an RGB window that returned garbage (128) for
`glReadPixels(GL_STENCIL_INDEX)` and failed piglit's
`hiz-stencil-read-window-*`, which pass on the V10. GL reads indices only
from an index window, so `dma_read_layout` now asks the display mode of
the window at the read's origin: an index group means a colour index read,
anything else a stencil read.

## Two-sided lighting

The lit programs the window records mark two-sided (`WindowRecord`) now
light the back with the back material and the reversed normal
(`dev::gl::light::Lighting::light_vertex` already did both faces), and the
vertices carry the back colour in `vtx::BACK_COLOR` with
`Prim::back_colours` set, which setup already honoured.

## Separate specular, sphere maps, pixel transfer (2026-11-05)

- **GL_SEPARATE_SPECULAR_COLOR.** Block 0x005 (`TEX_MODE`) bit 19 says the
  raster adds a second, specular colour after texturing
  (`tex::MODE_SEPARATE_SPECULAR`). The model lights such vertices with
  `Lighting::light_vertex_separate`, carries the specular in
  `vtx::SPECULAR`/`BACK_SPECULAR`, interpolates it in setup and adds it in
  `ShaderState::shade12`. Points, lines and polygons drawn in line, point
  or smooth mode take the sum before shading (exact untextured). The lit
  program is often an uploaded window record: record 15's body also
  arrives entered at 0x738 (0x738 0x4c5 0x4c5).
- **GL_SPHERE_MAP.** In the precompiled unlit programs the list continues
  with 0x06b (S and T from planes) or 0x06c (S and T sphere map) then
  0x077, without the 0x033 the assembled lists lead with. `sphere_texgen`
  computes s, t in `Board::sphere_map`.
- **glPixelTransfer scale and bias** arrive as block 0x203b before an image
  and apply to image pixels drawn into a window (`Board::image_transfer`),
  as to copies.
- **Still open:** pixel maps (uploaded by DMA to 0xcb000, not used),
  colour-index images into RGBA windows, signed texel uploads one or two
  steps below the board, ABGR with BYTE, byte-swapped INT and 5:5:5:1
  draws, glCopyPixels with zoom (1, -1), the depth of glDrawPixels
  (`drawpixels-skips-fragment-operations.md`).
