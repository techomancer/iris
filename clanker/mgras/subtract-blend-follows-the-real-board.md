# IMPACT's subtract blends drop or un-invert factors; the HLE copies the board

Status: rule, 2026-11-08 (real High IMPACT, Indigo2 IP28, IRIX 6.5.22;
blendtest `-a`, one random cell per legal factor pair).

**Symptom.** On the real board blendtest fails 29 of 156 SUBTRACT cells
and 51 of 156 REVERSE_SUBTRACT cells; ADD, MIN, MAX and the logic ops all
pass. The emulator passed all 508, i.e. it followed GL where the board does
not.

**What the board does** (`gl.rs` `impact_subtract_factors`, 309 of 312
cells reproduced):

- SUBTRACT: the destination term is dropped when the source factor is an
  alpha or constant one (SRC_ALPHA, DST_ALPHA, CONSTANT_COLOR,
  CONSTANT_ALPHA, their ONE_MINUS forms, SRC_ALPHA_SATURATE) and the
  destination factor is ONE_MINUS of a different one of those; SATURATE
  also against ONE_MINUS_SRC_COLOR. Same-quantity pairs
  (SRC_ALPHA / ONE_MINUS_SRC_ALPHA) and colour factors are fine.
- REVERSE_SUBTRACT: a source factor ONE_MINUS_{SRC_ALPHA, DST_ALPHA,
  CONSTANT_COLOR, CONSTANT_ALPHA} loses its ONE_MINUS, and the destination
  term is then dropped under the same conflict or for SRC_COLOR;
  ONE_MINUS_DST_COLOR as the source factor drops the destination term for
  SRC_COLOR and the alpha / constant factors.
- Unexplained: REVERSE_SUBTRACT with an inverted source factor over
  SRC_COLOR (3 cells) matches no pair of factors.

It looks like one shared alpha/constant operand and one inverter between
the two factors, but that is a guess; the rule is the measurement.

**Where.** The BlendFactor encoding past the two 4-bit factor fields is
ours (the GE HLE writes it), so the quirk is applied where the GE builds
the register from GL state, not in `rss::blend_rgba`. blendtest still
checks GL's formula: the same 80 cells fail on the board and the emulator.

**Data.** The board's `blendtest -a` output and the fitting scripts:
`clanker/mgras/real-high-impact.md`.
