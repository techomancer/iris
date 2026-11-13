# glDrawPixels' depth rides in the image box, not the raster position blocks

Status: resolved for depth, 2026-11-05. Stencil, alpha test and blending of
image pixels that are not depth-tested are still not applied.

Symptom: piglit's `gl-1.0-drawpixels-depth-test` and pixtest's
`depth/colour-draw/less` passed on the real V10 and failed in the model; a
pixel drawn behind the stored depth was not discarded.

Where the depth is: the raster position is computed by the library on the
host, not on the board. The blocks 0x3cf8-0x3cfb (16 zeros each) only route
the vertex to the board's lighting and texgen for the raster colour and
texture coordinate. The depth is in the image box `0xcf80a92f`: operands 6
and 9 (both the same) carry normalised-device z on the combined matrix's
depth-row scale, `DEPTH_S * (1 + ndc)`. The library sends it only when the
image is depth-tested; otherwise they are 0. glDepthRange is not folded in:
the board maps the value through the depth range like a vertex's.

Measured with a probe (raster z 0, 0.5, 1 under `glOrtho(..., 2, -2)`, then
`glDepthRange(0.25, 0.75)`, then depth test off), on octopus and in IRIS:

| raster z | range | box operand 6 | V10 stored depth |
|---|---|---|---|
| 0 | 0..1 | 4294951000 | 0.5000010 |
| 0.5 | 0..1 | 5368688600 | 0.6249990 |
| 1 | 0..1 | 6442426400 | 0.7499999 |
| 0 | 0.25..0.75 | 4294951000 | 0.5000010 |
| 1 | 0.25..0.75 | 5368688600 | 0.6249990 |
| 0.5, no depth test | 0..1 | 0 | 1.0 (untouched) |

The model now matches every row.

Gotcha: do not divide by `Board::gl_depth_scale()` (the client's block-16
sign word). The image runs under the library's own raster state, and that
word then gives a depth of 1 for every image.

Fix: `Board::image_box_loaded()` (draw.rs) keeps the box words, decodes the
back-end state and depth word once per box, and `image_put` sends each RGBA
pixel through `BackendState::fragment_q` (depth test and write, stencil,
blend, logic op, masks). Boxes with no depth, index images and scratch
destinations take the old plain store. The FIFO trace prints the box's ten
operands and, for a depth-tested box, the decoded depth and back-end state.
Test: `a_depth_tested_image_takes_the_box_depth`.
