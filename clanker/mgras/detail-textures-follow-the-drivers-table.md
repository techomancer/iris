# IMPACT detail textures follow the driver's table, not the spec's F(LOD)

Status: rule, 2026-11-08 (IRIX 6.5.22, Octane Maximum IMPACT; dettex, sgiquake).

**Model.** `te1.rs` `Sampler::detail` (interpreter), `rss_jit` `detail`
(JIT, key flag `detail`), `te1.wgsl` `smp_detail` (GPU unit) compute the
same thing, bit for bit: when magnifying (lambda <= 0) with TEXMODE2 bit 9,
out = clamp(base + F (2 detail - 1)) on all four components; the detail
texel is bilinear and repeating at the level-0 coordinate scaled by 2^-L;
F is the 32-entry DETAILSCALE table read at k = 32 2^lambda with an 8-bit
blend between entries (`exp2_q8`), entries past TXLOD bits 15:10 / 2 zero.
The register facts are in `ignore/gr4/TE1.h`.

**What the spec test says.** `test/gltest/dettex.c` checks the
SGIS_detail_texture formulas. On IMPACT, run it with `--impact`: the
reference then builds F from the table the driver writes and expects ADD
for every mode. Without `--impact` these fail by design:

- `add_color`, `add_alpha`, `modulate`: the driver sends ADD's registers.
- Any non-constant detail function (`ramp_*`, `peak_lod`, `func`) away from
  its points: the driver's table is linear in k = 32 2^LOD, the spec's F is
  linear in LOD.
- `add_half`: the driver rounds 0.5 x 31 down to 15.
- Levels above -2 (`add_level2` with `--level -2`): sent as -2.

The driver only enables detail on a mipmapped base with a 128 or 256
square detail image; use `--mipmap` and `--detail 128|256`.

**Why zero past the TXLOD index, not clamp to it.** For a ramp ending at
weight 0 the driver writes `31 31 27 22 18 13 9 4 0 0 ...` and TXLOD 14
(index 7, the last nonzero): clamping would hold 4/31 where the function
is 0. For a function ending nonzero it writes the held value to entry 31
and TXLOD 62. Zeroing past the index fits both; nothing observable here
settles what the board does, real hardware would.

**Checks.** `cargo test --features gr4-jit,wgpu mgras` sweeps detail
through the interpreter, JIT and GPU (`test_gen` turns it on for a quarter
of the textured states); `te1::tests::detail_adds_the_scaled_detail_image`
pins the model to the spec formula. `mgrasq1.rec` (sgiquake, mid-session:
see `mid-session-replays-need-the-live-drbsize.md`) replays with all 518
checkpoints equal on CPU and GPU.
