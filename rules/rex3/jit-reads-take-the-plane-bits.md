# Compiled READ and SCR2SCR reads take the plane's bits, as the interpreter does

The interpreter reads a pixel through `read_plane` / `plane_shift_mask`
(rex3_generic.rs): the aux framebuffer for overlay, popup and CID planes at
their own bit offsets (OLAY 8/16, PUP 2/6, CID 0/4), and for RGB depths
4, 8 and 12 the second buffer at bits 4, 8 or 12 when DBLSRC is set. Then
the depth mask, then expansion to 24-bit in RGB mode.

The Cranelift HOSTR read (READ opcode) only masked the raw word, so it read
the first buffer whatever DBLSRC said, and the low bits of an aux plane. Its
SCR2SCR source read handled the aux planes but ignored DBLSRC for RGB.
Destination reads (blending, logic ops) were already right.

Found through IRIX's glReadPixels from a 12-bit double-buffered RGB window:
DRAWMODE0 `0x65`, DRAWMODE1 `0x3565fbb1` (DBLSRC, 32-bit host words,
SWAPENDIAN, RWPACKED). The first read ran on the interpreter while Cranelift
compiled the shape, and every later one read the other buffer
(scratch probe: "left 136 0 0" where 0 0 0 belonged). The prebuilt shaders
were never affected: they are the interpreter's own pipeline instantiated
with constants.

Bisect a bad compiled shape with `rex jit disable <dm0> <dm1> <cm>` (it takes
the shape out of dispatch) after `rex jit list`.

Tests: `jit_hostr_rgb12_dblsrc_swapendian_irix_readpixels`,
`jit_hostr_rgb_depths_and_buffers`, `jit_hostr_aux_planes`,
`jit_scr2scr_rgb12_dblsrc` (rex3_tests.rs, `--features rex-jit`).
