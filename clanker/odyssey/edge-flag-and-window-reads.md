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
