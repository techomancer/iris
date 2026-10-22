# The ip28 guest's X server defaults to 24-bit TrueColor

`/var/X11/xdm/Xservers` on the ip28 image starts Xsgi with
`-depth 24 -class TrueColor`. SGI's stock default is an 8-bit PseudoColor
root visual, and some period software assumes that default.

Symptom (SGI's sgiquake1 `quake.sw`, 2026-10-13): the picture is squeezed
into the left quarter of the window, in noisy colours, with black beyond.
The program creates a shared-memory XImage in the default visual (32 bits
per pixel, 1280-byte rows at 320 pixels) and writes 8-bit palette indices
into the start of each row. The X server then pushes those rows as 32-bit
pixels (xfrmode 0x80, DMA line_bytes = stride = 1280), so every four
indices become one pixel.

This is not an emulator bug. Real hardware with the same Xservers line
shows the same thing. In a trace, the tell is that each DMA's line 0 holds
byte-wide indices, not 32-bit pixels.

Fix: run the program on an 8-bit visual (`quake.sw -visualid 32`, visual
0x20 in `xdpyinfo`), or drop `-depth 24 -class TrueColor` from Xservers.
With `-visualid 32` Quake draws correctly, so the 8-bit PseudoColor path in
the 24-bit planes works.
