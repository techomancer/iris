# FRONT_AND_BACK geometry arrives with the front draw mask

Status: 2026-11-04, open, not a model bug as far as the stream shows.

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

**Not settled on the board.** The V10 (octopus, 32 MB) offers pixtest no
double-buffered visual, so the case is skipped there. A board with more
memory, or a visual list that includes one, would show whether the
library draws front only.
