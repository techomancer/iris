# Fastclear packs COLORVRAM to the plane depth in RGB mode

Status: rule, found 2026-11-26.

Symptom: in an 8-bit TrueColor X window (visual 0x25, R 0x07 G 0x38 B 0xc0)
every XFillRectangle stored the colour's red byte: pixel 0x01 read back 0x24,
0x38 read 0x00, 0xc0 read 0x00. XPutImage into the same window was right.

Xsgi clears and fills these windows by fastclear. Each fill writes
DRAWMODE1 = RGB, 8bpp (`0x3000f109`), then COLORVRAM with a 24-bit BGR
colour (`0x00000024`, `0x0000ff00`, `0x00ff0000`), then sets FASTCLR and
draws a block. The spec says COLORVRAM is loaded after RGBMODE and DRAWDEPTH,
that is, its meaning depends on them.

Rule: in RGB mode a fastclear packs COLORVRAM like any other RGB write,
without dither: `rgb24_to_rgb4/8/12`. CI mode uses it as the index. 24bpp is
unchanged. `fastclear_color` (interpreter, precompiled shaders, and the GPU
raster, which takes its fastclear colour from it on the host) and the JIT's
inline copy in `rex3_jit/compiler.rs` agree. 12bpp already did this; 4 and 8
took the low bits.

Tests: `fastclear_packs_colorvram_to_the_rgb_plane_depth` pins the stored
values; `jit_fastclear_rgb_packed_depths` compares the JIT at 4/8/12bpp.
The JIT equivalence tests alone could not catch it, since both engines
shared the bug.
