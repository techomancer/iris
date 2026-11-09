# IMPACT fogs RGBA from the window depth through a table

Status: rule, 2026-11-09 (real High IMPACT, Indigo2 IP28, IRIX 6.5.22;
`test/gltest/fogtest`, glprim `--scene fog|fogexp|fogexp2`;
`mgras_tests.rs` `real_board_fog`, `real_board_fogtest`).

**Symptom.** EXP and EXP2 fog across one quad were straight ramps here (up
to 59 levels off) and curves on the board; linear fog with start or end
inside the primitive, or under a perspective projection, was wrong too
(a per-vertex factor, interpolated).

**What the board does** (fitted to fogtest's 14 cases within 1-2 levels):
- The fog factor is a function of the **window depth** d, looked up in a
  table built from the projection and the depth range. Perspective fog is
  therefore exact in eye z (fogtest's frustum cases curve as GL says).
- The index carries a bias of one table step: d (N - 1) / N + 1 / N, with
  N = 32 steps, linearly interpolated, for EXP and EXP2, and N = 512 for
  linear fog. In eye terms at ortho: z (31/32) + far/32 for EXP.
- The factor is floored to 8 bits on a 256 scale (255 at most) and
  multiplies the colour as factor/255.
- Pixels are fogged in **pairs**, an even window x and the odd one after
  it, each pair taking the depth at (x & !1) + 2. Along a line the same
  on the major axis. (Pairs counted from each span's start fit the scenes
  less well, 1.2 levels against 0.9.)
- 8- and 12-bit visuals give the same factors; GL_FOG_HINT changes
  nothing.

**Model.** The GE HLE builds a 257-point table over the biased depth
(`gl.rs` `fog_state` / `fog_table`: EXP's 33 points with the chords
between them exactly, linear fog's curve directly) into the model-private
`rss::reg::FOG_LUT` (129 words) with the index's scale and offset
(`FOG_IDX_K`, `FOG_IDX_B`), resent when the fog parameters, the
projection's depth terms or the depth range change, and once per batch.
The depth plane is loaded whenever fog is on. The RSS looks the factor up
from the depth at the pair's sample (`rss::fog_factor`,
`fog_pair_offset`); the JIT and the GPU unit do the same integer steps.
Colour-index fog is still per vertex.

Error against the board: glprim's scenes 0.5-1.2 levels mean (it was 2.8
linear, 40 EXP, 25 EXP2), fogtest within 3 levels everywhere.

The hardware facts behind it: RE4 fogs per pixel with 8 bits through a
table it shares with the texture colour tables (Octane technical report;
ignore/gr4/TE1.h). How the GE11 loads that table is not visible on the
bus, hence the model-private registers.
