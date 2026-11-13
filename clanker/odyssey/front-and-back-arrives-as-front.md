# FRONT_AND_BACK draws front only on a 16-byte frame buffer: the library, not the board

Status: rule, 2026-11-05 (was open). Resolved by comparing with a V10.

**Symptom.** piglit `gl-1.0-drawbuffer-modes` and pixtest's front-and-back
cases failed on the emulated board: geometry drawn under
glDrawBuffer(GL_FRONT_AND_BACK) (or GL_LEFT) landed in the front only. The
stream carries draw mask 0x424 = 1 for that geometry, word for word the
GL_FRONT stream.

**Cause.** The 6.5.30m library's clear (`ody_clear`) splits a
FRONT_AND_BACK clear into a back pass and a front pass and leaves the
front-only mask selected afterwards when context +0x7b74 is zero.
`MakeCurrent` sets that word to 1 unless the drawable's pixel-size field is
3, a 16-byte frame buffer. The real V10 runs an 8-byte frame buffer (its
`/var/X11/Xvc/Odyssey0_TimingTable` reads `1024x768_60`, `8`, `0`: `setmon
-d 8`), takes the one-pass clear and keeps mask 5 (dbx:
`__glOdysseySetDrawbuf(ctx, 1, 0, 1, 0)`). The emulated guest's timing
table was empty, so X took its default, 16 bytes.

**Rule.** To compare the emulated board with a real one, give the guest
the real machine's frame-buffer depth: `/usr/gfx/setmon -x -d 8
1280x1024_60` in the guest, then reboot. With the same setting the
emulated V10's visual list matches the real V10's (33 visuals) and the
test passes. On a 16-byte frame buffer the library's behaviour above is
what a real board would also get (inferred from the code, not measured).
