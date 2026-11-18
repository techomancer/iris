# 8-bit frame blits on XZ: the per-frame sync costs, not X vs GL

Status: measure, 2026-11-17, emulated Indy XZ (R5000), IRIX 6.5.

**Question.** For a game that draws 8-bit frames itself (SDL "potato mode",
chocolate-doom), is XShmPutImage, XPutImage or glDrawPixels(GL_COLOR_INDEX)
the fastest way onto a GR2?

**Same path on the board.** XShmPutImage and an unzoomed glDrawPixels of a
640x480 CI frame both reach the HQ2 as GEDMA, 76,800 words per frame (four
pixels per word). Board-side they cost the same.

**glPixelZoom is a trap.** A zoomed glDrawPixels (320x200 scaled 2 x 2.4)
makes libGL send one GL_FRAGMENT per output pixel (~300k FIFO commands a
frame): 3 fps against 20. Scale on the CPU and draw 1:1.

**The difference is synchronisation.** 640x480, 100 frames:

| Mode | fps |
|---|---|
| XShmPutImage + XSync each frame | 11.7 |
| XShmPutImage + XFlush only | 19.5 |
| glDrawPixels, no wait | 19.9 |
| glDrawPixels + glFinish each frame | 14.2 |

**But a real game doesn't gain from async.** chocolate-doom with updates
that wait only for the previous frame: 16.6 fps against 16.3 synchronous
(the Indy has one CPU; X server and game run in turn either way). The
probe is `blitbench` (not in the tree; a 100-line Xlib/GL loop).

**Doom numbers** (potato mode, -timedemo demo1, 640x480 window): XL24 on an
R4400 26 fps, XZ on an R5000 16-19 fps (57 with -noblit, so ~40 ms a frame
is the blit). IRIX's OpenGL on Newport textures in software: the stock SDL
GL renderer path takes seconds per frame there.
