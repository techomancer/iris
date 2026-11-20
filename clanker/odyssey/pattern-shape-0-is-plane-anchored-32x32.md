# Pattern shape 0 is a 32x32 pattern anchored at the plane

Status: rule (inferred from the traced data), 2026-11-18. Fixed in
`Board::pattern_enabled` / `pattern_bit` (draw.rs) and `gl_quad`
(geometry.rs).

## Symptom

In IRIS GL pop-up menus (powerflip's right-button menu), a checked toggle
showed a black bar about 16x8, twice the box's width, instead of a check
mark.

## Cause

The X server draws the check mark as a quad, 16x8 at (85, 117) in the
overlay/window-id buffer (0x142000).

- The quad is drawn with `PATTERN_MODE 0x00080018` and `PATTERN_SHAPE 0`.
- Before it, the X server loads the glyph into the 32-word pattern buffer,
  replicated across it (each 16-bit row twice in a word, 8 rows repeated).
- The model took shape 0 for "no pattern" (its constants said "0x0 = off").
- So the quad went down the id-plane solid-fill path (`clear_planes`) and
  filled the whole rectangle.

## Rule

Shape 0 is a 32x32 pattern, MSB first, rows top-down, sampled at plane
coordinates: bit `31 - (x & 31)` of word `y & 31`.

Shapes 4, 5 and 6 (32x32, 64x16, 16x64) stay anchored at the quad's top-left
vertex.

Only the plane-anchored reading turns the traced words into a check mark;
quad-relative and window-relative readings give noise. The X server's
window origin was 0 in the trace, so plane and drawable origin could not be
told apart.

GL's polygon stipple also uses shape 0 (`pipe/state.rs`), with its own
bottom-up, LSB-first conversion in the library.

Test: `a_shape_0_pattern_quad_paints_the_glyph_anchored_at_the_plane`
(`src/dev/odyssey/tests/pixels.rs`).
