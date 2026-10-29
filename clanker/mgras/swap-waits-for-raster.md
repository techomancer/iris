# Report SCHEDULE_SWAP only once the raster engines have drawn the frame

Status: rule (2026-10-28).

**Symptom.** Octane (IP30) IMPACT, double-buffered GL fullscreen (`bongo
-geometry 1280x1024+0+0`, the octahedra saver): flicker and tearing, frames
shown half drawn. Shots taken with the CPU stopped (`stop`, `mgras shot`)
showed the displayed half partly cleared and redrawn. The IP28 was fine.

**Cause.** The HQ3 model raised the swap flag (FLAG_CP0, HQ4 pending bit
10) when SCHEDULE_SWAP reached it in `hq_fifo`. The frame's raster writes
could still be queued in `rss_fifo`. The Octane kernel's retrace handler
reads STATUS and flips XMAP buf_select once the swap is pending, even on
0x55 (HQ idle, board busy). The real pipe behind the HQ is shallow; ours
is not. The Indigo2 kernel flushes FIFOSTATUS to 0 (whole board idle,
`cfifo_level`) before each swap, so it never saw the race.

Not the cause, checked: bank tracking. VALIDATE_BANKS alternates [0,0] and
[1,1], context restores carry the same bank in image words 16-17, and every
flip shows the other half. The gr4 raster JIT was not the cause either:
`mgras jit off` showed the same partial frames.

**Fix.** `Hq3Sink::finish_raster` (drain `rss_fifo`) before `set_flags
(FLAG_CP0)` in the CP_SCHEDULE_SWAP handler. Perfetto: `swap waits raster`
on `IMPACT swap wait`.

**Visual note.** On the Octane the SI offers no double-buffered 24-bit
visual, so GLUT's double-buffered RGB gets the 12-bit pair visual
(DRAW_BUFFER [2, 1], XMAP format 7, both buffers in one word); on the IP28
it gets 24-bit [4, 1]. A bug in one of these paths shows on one machine
only.
