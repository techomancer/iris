# VPro lines on the triangle arithmetic, then on the GPU

Status: steps 1, 3 and 4 done 2026-11-07 (CPU on the triangle arithmetic, ring line commands, REC_LINE kernel); step 2 aliased part done on octopus (pixels match); smoothed lines open (see clanker/odyssey/vpro-line-rasterisation-on-a-real-v10.md); t stays the major-axis form.

## Why

`Draw::line_projected` (pipe/prims.rs) interpolates in f64. That came
from the original port, not from the hardware. Every line pixel is shaded
on the CPU. The GPU unit gets them as packed fragment records; the
threaded CPU unit gets one ring quad per pixel.

SolidView (`ignore/recordings/solid.rec`: 465k aliased lines of about 9
pixels) with lines skipped against drawn, 2026-11-07:

| unit | with lines | lines skipped | lines cost |
|---|---|---|---|
| CPU inline | 2.59 s | 2.14 s | 0.44 s |
| CPU threaded (live config) | 3.32 s | 1.74 s | 1.57 s |
| GPU | 2.56 s | 1.95 s | 0.61 s |

## Number formats (the evaluation)

- **Pixel selection: integer.** It already is (`centre_minor`: subpixel
  coordinates, 8 bits, exact rational, i64). Keep it.
- **Attributes: the triangle's arithmetic, not f64, f32-only or pure
  u32.** Triangles are bit-exact on both units today:
  - f32 barycentrics;
  - 0.15 perspective weights from `interp::weights4`;
  - 12.12 colours through `fixed::interpolate`;
  - f32 depth through `depth_ratio`/`window_depth32`;
  - texture through `TriTex`;
  - on the GPU, the `nc` and `div_exact` guards.

  A line pixel is that triangle with l1 = t and l2 = 0 (vertex c a copy
  of a). Reusing it gives CPU/GPU parity for free. A wireframe line also
  gets the same depth and colour arithmetic as the polygon edge it
  outlines. A pure fixed-point DDA would be closer in spirit to hardware,
  but it needs its own perspective divide and its own WGSL twin, and we
  have no measurement of the VPro's line interpolation to match anyway.
- **The parameter t: the major-axis DDA.** Use t = (c_major - a_major) /
  (b_major - a_major), integers converted with the same rounding as the
  barycentrics, instead of the GL projection formula ((c-a)·(b-a)/|b-a|²).
  Hardware steps along the major axis; GL allows either. It drops a dot
  product per pixel. To be checked against octopus.
- **User clip distances:** today f64 for triangles too. Move them to
  f32 the same way (the remaining reason fogged or user-clipped
  primitives stay CPU-shaded); a separate step.
- **Smooth lines:** box-filter coverage by polygon area in f64. Rare
  (1,050 of 465k in SolidView). Leave them on the CPU as packed fragments
  first. Also narrow their candidate scan, which still covers the whole
  minor extent (O(length²)).

## Steps

1. **CPU reformulation.** `line_projected` computes the integer t and
   feeds the triangle's shading (a `Tri` with l2 = 0, or a small
   `LineTri` sharing `weights4`, `fixed::interpolate`, `TriTex` and the
   depth path). Keep pixel order, stipple counting, width replicas,
   ownership and scissor exactly as now. Re-take the line goldens under
   `VPRO_GOLDEN=tolerance` and inspect.
2. **Validate against hardware.** On octopus, run the glprim line cases,
   the line sweep (30 readbacks) and the piglit line tests. Compare with
   the emulated V10 before and after. Prefer whichever t formulation
   matches.
3. **Threaded CPU unit.** Push an aliased line as one ring primitive that
   workers walk, as they walk triangle quads, instead of one quad per
   pixel.
4. **GPU kernel (REC_LINE).** One record per line, or per segment of at
   most N major pixels so the binning box stays thin. Contents: subpixel
   endpoints, the t numerator and denominator terms, 12.12 colours,
   1/w, depth terms, texture values, width, stipple start count and
   pattern.
   - A lane covers (px, py) when py (x-major) lies in the minor span of
     the exact line at the column centre. Expressed as 64-bit
     multiply-compares (needs a 32x32 to 64 multiply in i64.wgsl; there
     is no division), with `centre_minor`'s tie rule as strict and
     non-strict bounds.
   - The stipple bit index is the major distance from the segment start
     plus the start count.
   - Shading reuses the triangle kernel's helpers with l2 = 0.
   - A `bin_accept` hook (ArchShaders) can drop tiles the line misses.
5. Then user-clip distances in f32, and smooth-line coverage as a
   table, to retire the remaining CPU-shaded line paths.

## Expected

Most of the 0.6 s on the GPU unit (CPU setup per line stays) and most of
the 1.6 s on the threaded CPU unit for SolidView; less per-pixel work on
the inline CPU unit (no f64 divides per pixel).
