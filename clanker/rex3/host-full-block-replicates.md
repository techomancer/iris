# A host DRAW BLOCK with STOPONY fills the whole block from HOSTRW

`DRAW BLOCK COLORHOST` with **STOPONY** set is not the segmented host upload.
It is a full block. One GO runs it to the end, and the shifter reloads from
HOSTRW every time it empties. With a single PIO word (`host_len == 1`), the
same word is re-read, so its pixels repeat across the block.

The segmented upload, one word per GO (rex3.pdf §3.5.4.1 "Segments I"), is
specified with STOPONX=STOPONY=0. IRIX always uses that form for pixmap and
icon uploads: `DRAWMODE0 = 0x46`.

## Evidence

IRIX glPixelZoom, as used by `snoop`, draws each zoomed pixel like this:

```
DRAWMODE1 = 0x3009f3b1   RGB 12bpp DBLSRC host:32bpp RWPACKED DITHER
DRAWMODE0 = 0x000003c6   DRAW BLOCK COLORHOST ALPHAHOST STOPONX STOPONY
XYSTARTI (0,1699)  XYENDI (0,1694)  XSTARTENDI 1376..1381  SETUP
HOSTRW1 = c ; HOSTRW0 GO = c     <- one colour, one GO, 6x6 block
```

We used to stop after one word in every host mode. Snoop then showed one dot
every 6 pixels on black instead of 6x6 blocks. MAME's `do_rex3_command` block
loop gives the same fill: its outer `stop_on_y` loop keeps going, and
`get_host_color` keeps returning the same 32bpp word.

## Scope

- Only the block walker with DRAW, COLORHOST and STOPONY is affected. Spans,
  STOPONX-only blocks, READ, and ALPHAHOST-only modes still stop at a word
  boundary, as in MAME.
- In a full host block, a row boundary is still a word boundary: `hostcnt` is
  reset and the primitive continues into the next row.
- The rex JIT declines this mode, because its `host_xstop` assumes one word per
  call. Only the zoom path uses it, so the interpreter fallback costs nothing
  noticeable.
- Test: `test_hostw_full_block_replicates_single_word`. The other host-write
  tests used to set STOPONX|STOPONY while expecting one word per GO. They now
  use IRIX's real segmented mode (`DM0_HOSTW_BLOCK` has no stop bits).
