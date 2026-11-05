# FRONT_AND_BACK geometry arrives with the front draw mask

Status: 2026-11-05, open. The V10 draws both buffers; the emulated board
receives nothing that says so.

**Symptom.** pixtest `buffers/geometry-front-back` and
`buffers/drawpixels-front-back` fail with "blue missing in back" on a
double-buffered 8:8:8:8 visual.

**What the guest sends** (trace of the single case):
- glDrawBuffer(GL_BACK): the kernel's context load writes draw mask
  0x424 = 4.
- glDrawBuffer(GL_FRONT): mask 1.
- glDrawBuffer(GL_FRONT_AND_BACK): mask 1 again. The quad's command words
  are the same as the front-only quad's, apart from colour and vertices.
- glClear under GL_FRONT_AND_BACK is two passes, mask 4 then mask 1. The
  geometry and the image are sent once.

Nothing in the stream tells the board to write the back for the third
quad, so no model of the board can pass this case from these words.

**Settled on the board (2026-11-05): it draws both.** piglit's
`gl-1.0-drawbuffer-modes` passes on the real V10 with a double-buffered
8:8:8:8 visual (0x3e, no depth): after glDrawBuffer(GL_FRONT_AND_BACK)
the rectangle is in the back as well as the front, and after
glDrawBuffer(GL_FRONT) it is in the front only. GL_LEFT behaves as
FRONT_AND_BACK.

On the emulated V10 the same test fails for FRONT_AND_BACK and LEFT, and a
trace (`vpro trace ... all`) shows the FRONT_AND_BACK geometry and the
later FRONT geometry arrive the same way: draw mask 0x424 = 1, write mask
0x452 = 0x01f00000, the same drawable words 0x608/0x609 = 0x80002000, one
pair of triangles. The clear is two passes (mask 4, then mask 1) in both.
So the difference is not in the FIFO words the emulated board receives.
Candidates: the library takes another path on the real board because of
something the board or the kernel reports (an identity, revision or
capability the model answers differently), or the kernel changes state
outside the FIFO (CPU writes are in the trace too, and show nothing).
Next step: trace the same test on the real V10 (needs the kernel's view,
since the board cannot be traced there), or compare what libGLcore reads
from the board at make-current between the two machines.
