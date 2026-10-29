# Take the IMPACT display frame before the retrace pulse, not after

Status: rule (2026-10-28).

**Symptom.** Double-buffered GL (octahedra, the `bongo` saver) tore on the
Octane, although a trace showed the guest correct: SCHEDULE_SWAP, the
kernel's XMAP `buf_select` flip about 7 ms later in its retrace handler, and
the next frame's drawing only after the flip.

**Cause.** The refresh thread raised the retrace, slept 500 us, then copied
the visible frame (a millisecond or more). The handler's flip landed inside
that copy, and the client began drawing into the buffer being copied.

**Rule.** Snapshot first, then pulse the retrace (`Mgras::refresh_loop`).
A flip made at retrace N shows from frame N+1, as scanout does. The STATUS
vertical-blank window (`begin_frame`) moves with the pulse.

GR2 (`Gr2::display_loop`) had the same order (pulse, hold the vertical
status 700 us, then compose) and now composes first too. REX3 was already
right: it copies the framebuffer and the XMAP/VC2 state into the screen
caches, then raises VBLANK; it only presents the live framebuffer when no
pixel changed since the last copy.
