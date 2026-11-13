# IMPACT detail textures follow the driver's table, and add w (d - 0.53) / 32

Status: rule, 2026-11-08; measured on a real High IMPACT (Indigo2 IP28,
IRIX 6.5.22, `purple`) with dettex `--fill`, `--func` and `--dump`.

**Model.** `te1.rs` `Sampler::detail` (interpreter), `rss_jit` `detail`
(JIT, key flag `detail`), `te1.wgsl` `smp_detail` (GPU unit) compute the
same thing, bit for bit: when magnifying (lambda <= 0) with TEXMODE2 bit 9,
out = clamp(base + F (detail - DETAIL_CENTER)) on all four components; the
detail texel is bilinear and repeating at the level-0 coordinate scaled by
2^-L; F is the 32-entry DETAILSCALE table read at k = 32 2^lambda with an
8-bit blend between entries (`exp2_q8`), an entry of 31 weighing 31/32
(`DETAIL_SCALE`), and F = 0 from the TXLOD index (TXLOD 15:10 / 2) on.
The register facts are in `ignore/gr4/TE1.h`.

**What the real board does** (uniform base b and detail d, `--fill b,d`,
table weight w from `--func -10,X,0,X`):

- out = b + w (d' - 2185) / 32 in 12-bit units, texels widened to 12 bits
  by replication (d' = 16 d + d >> 4), rounded: 161 of 165 points exact,
  the rest one level off. Not the extension's b + F (2 d - 1): the slope
  is half and the zero sits at d = 136.6, not 127.5.
- The weight is linear in w (w = 1..31 measured), 31 = 31/32.
- The cutoff is sharp: a table with entries 15 and 16 at 31 and TXLOD
  index 16 gives full weight at k = 15.52 and none at k = 16; a constant
  function (TXLOD 62) gives full weight at k = 30.1 and none at 31.0.
  Between entries the board blends, within about one table step.
- k = 32 2^lambda holds (ramps over -6..0 measured at 17 sizes).
- Component order and the detail coordinate (2^-L texels a level-0 texel,
  no offset, bilinear) match; with random textures the residual is about
  4 levels RMS at L = -3, 8 at L = -4, unexplained (not fraction or
  coordinate quantisation).
- **Level -2 is broken on the board**: seams and a tear along the quad's
  diagonal (dettex `add_level2`). The driver sends levels above -2 as -2,
  so applications get it; we do not model the fault.

**What the spec test says.** `test/gltest/dettex.c` checks the
SGIS_detail_texture formulas. On IMPACT, run it with `--impact`: the
reference then uses the driver's table, the board's combine and cutoff,
and expects ADD for every mode. Without `--impact` these fail by design:
`add_color`, `add_alpha`, `modulate` (the driver sends ADD's registers),
non-constant functions away from their points (the table is linear in
k = 32 2^LOD, the spec's F in LOD), `add_half` (the driver rounds 0.5 x 31
down to 15), levels above -2.

The driver only enables detail on a mipmapped base with a 128 or 256
square detail image; use `--mipmap` and `--detail 128|256`.

**Checks.** `cargo test --features gr4-jit,wgpu mgras` sweeps detail
through the interpreter, JIT and GPU (`test_gen` turns it on for a quarter
of the textured states); `te1::tests::detail_adds_the_scaled_detail_image`
pins the model to the measured combine and cutoff. `sh dettexsweep.sh DIR
[fill weight lod cutoff]` (test/gltest) repeats the measurements on any
board; the real board and its emulated twin are described in
`real-high-impact.md`.
