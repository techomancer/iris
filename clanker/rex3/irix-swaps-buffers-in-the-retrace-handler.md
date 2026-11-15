# IRIX swaps buffers in the retrace handler, and Newport's does not wait for drawing

Status: observed 2026-11-14 in the IRIX 6.5.22 kernel graphics modules as
shipped (Newport, GR2, IMPACT), disassembled and decompiled locally
(`ignore/ip24/guest-files/ng1/`, `ignore/gr2/gr2_kernel/`,
`ignore/gr4/mgras_kernel/`). It is the reason the display refresh thread
captures and composes after the vertical blank
(`clanker/plans/display-refresh-common-layer.md`).

## What the kernel does

All three boards' drivers work the same way.

- **A swap request only schedules the swap.** With retrace interrupts
  masked, the driver records in a 32-entry table (one per window ID) the
  window's new display mode, a countdown (the swap interval, at least 1),
  and the buffer to hand back. The client's access to the board stays
  taken away until the swap has happened.
- **The retrace handler does the swap.** For every entry whose countdown
  has run out it writes the new mode to the board, then gives the buffer's
  access back to its client. Entries with a longer interval count down and
  wait for a later retrace.
- So after a retrace has been serviced, the visible buffer is finished and
  its client draws the next frame into the other one. Before it is
  serviced, a client that asked for a swap is not drawing at all.

## Newport

- The interrupt routine reads REX3 STATUS once (which clears VRINT) and
  runs the retrace handler when the VRINT bit was set. On Indigo2 it then
  pulses the retrace-clear bit in the port configuration register.
- The swap is an XMAP9 mode-register write through the DCB: the window ID
  in the top byte, the mode in the low 24 bits, with **bit 0 (buffer
  select) set or cleared**. The handler picks the DCB timing from the
  video format's refresh rate (below 60, 60 to 119, 120 and up) and
  restores DCBMODE when it is done.
- Before each write the handler waits for **BACKBUSY** to clear and for the
  XMAP9's FIFO status. It never looks at GFXBUSY and does not wait for the
  graphics FIFO to drain.

That last point is why the emulator has a fence of its own. On the board the
FIFO is 32 deep and drains in microseconds, so a flip in the blank cannot
overtake drawing. Ours holds far more, so the mode-register write can reach
the XMAP while drawing for that frame is still queued. Every mode-table
write therefore bumps `xmap_fence` and queues a sync entry; the display must
not capture until the processor has passed it (`gfifo_fence`). Keep this
when Newport moves to the refresh thread: the fence guards the capture step.

BACKBUSY in the emulator is raised only by DCB address 12 (for 1 ms), not by
XMAP writes, so the handler's waits cost nothing.

## GR2 and IMPACT

- GR2's handler polls the vertical status level for up to ten 1 µs delays
  before it writes, so it expects the blank level still up when it runs;
  then it writes the XMAP mode for each due window.
- IMPACT's handler has the same table and the same hand-back.

## What this means for the display thread

The guest has the length of the blank (0.66 ms at 1280x1024, on the host's
clock) to service the retrace before the frame is captured. If a desktop
tears, the guest is servicing it later than that; the lever is to hold the
blank open until the interrupt is acknowledged, not to re-read or re-compose.
