# glDrawPixels skips the fragment operations in the model

Status: open, 2026-11-05.

piglit's `gl-1.0-drawpixels-depth-test` passes on the real V10 and fails on
the emulated one: a pixel drawn at raster depth 1.0 under GL_LESS over a
stored 0.5 must be discarded, and the model draws it.

The library sends glDrawPixels through the primitive path: the raster
position blocks 0x3cf8-0x3cfb (16 operands each), then `GL_BEGIN` with
mode 10, the image box, one `VERTEX_2I` at the window position, the image
header and the burst. The model skips the four raster position blocks
("unmodelled word ... skipping 16 operands") and paints image pixels with
`Board::paint_pixel`: window clip and id tests only, no depth, stencil,
alpha test or blending.

The back end has the per-fragment path (`BackendState::fragment_q`), so
the store side is ready; what is missing is the fragment's depth. On the
board the raster position is transformed by the geometry microcode from
those blocks, which the model does not run. Next step: record the blocks'
operands for known raster positions (extend the FIFO trace to print the
operands of skipped words) and find the depth among them.
