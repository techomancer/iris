# The display frame is taken after the retrace, not before the pulse

Status: history (rule 2026-10-28, reversed 2026-11-14 by the display refresh
thread; see `clanker/plans/display-refresh-common-layer.md` and
`clanker/rex3/irix-swaps-buffers-in-the-retrace-handler.md`).

**The old rule.** Double-buffered GL (octahedra, the `bongo` saver) tore on
the Octane. A trace showed the guest correct: SCHEDULE_SWAP, the kernel's
XMAP `buf_select` flip about 7 ms later in its retrace handler, and the next
frame's drawing only after the flip. The refresh thread raised the retrace,
slept 500 us, then copied the visible frame (a millisecond or more); the
handler's flip could land inside that copy, and the client began drawing
into the buffer being copied. So IMPACT and GR2 took their frame first and
pulsed the retrace after, and a flip made at retrace N showed from frame
N+1.

That was not the whole Octane tearing: the main cause was the swap being
reported before the raster engines finished the frame
(`swap-waits-for-raster.md`).

**Now.** Every board follows one order on the refresh thread: vblank start,
the blank (0.66 ms at 1280x1024), vblank end, capture, compose. The capture
reads the buffer the handler has made visible, which is the frame the guest
means to show, one frame sooner than the old order.

**What to watch.** The 7 ms above is a measurement of how late the handler
ran on the host's clock in that trace: ten times the blank. When the handler
runs after the capture has finished, the capture read the old front buffer,
complete and untouched (the client has no access until the handler hands it
back), and the swap shows a frame later, as before. Only a flip landing
inside the capture can tear. Octahedra and windowed demos looked right on
Newport, XZ, IMPACT and VPro on 2026-11-14. If tearing comes back, hold the
blank open until the retrace interrupt is acknowledged instead of going back
to the old order.
