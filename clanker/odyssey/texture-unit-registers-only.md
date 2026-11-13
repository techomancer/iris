# The texture unit reads registers, a latch and board memory, nothing else

Status: 2026-11-04, current.

## Rule

In `src/dev/odyssey`, a data structure is either a hardware register, or it
needs a reason: it caches state that is expensive to derive (latched when a
primitive is issued, behind a dirty bit), or it holds data in flight between
parallel stages (a draw job, a ring slot, a triangle record). There is no
converted, intermediate, validated or GL-shaped state. Use hardware enums,
raw pointers to data, and only the error checking that names what the model
does not decode yet. `Option` is suspect and `Mutex` is wrong. Types are
`repr(C)` (enums `repr(u32)`), because a JIT will read them.

The hardware does not check, so the model does not refuse. A level address
is masked to board memory (`Vram::new` rounds up to a power of two), and a
level that points at garbage samples garbage.

## Shape (`pipe/texture.rs`, `pipe/slot.rs`)

- `TexRegs`: the texture registers as written (path, sampler, env, LOD,
  first level, level table, scale/bias, offset, colour tables, detail
  function). Field accessors return hardware enums (`Target`, `Wrap`, `Min`,
  `Mag`, `Class`, `Env`).
- `Latch`: per level the byte address, tile row, texel storage, log2 size and
  volume slice stride; the level count, the LOD clamps and bias scales, the
  detail image, the max clamp, and the border colour. `unimplemented` bits
  are logged once each by `Board::issue_state`.
- `Slots`: 64 `DrawState { tex, latch, dirty, users }`. `Board::tex_mut()`
  copies the current slot to a free one when a queued draw holds it (and
  waits for the raster unit if every slot is held). `Board::issue_state()`
  latches a dirty slot and counts the draw as a user. `GlDrawJob::release`
  is called from `give_back` (and from `Drop`).
- `Unit::sample_quad(&TexQuad, cov, &mut [[i32; 4]; 4])`: the quad's four
  pixels' homogeneous S, T, R, Q in; it divides, takes one lambda for the
  quad from the diagonals, picks the filter and levels once, and per level
  gathers the covered pixels' tap box (up to 8x8, two slices for a volume)
  into an uninitialised stack window, then filters each pixel from it
  (12-bit components, 8-bit weights, 16 fraction bits) and applies detail,
  scale/bias and the colour table into 12.12. A box over 8x8 (clamped
  lambda, anisotropy) gathers each pixel's own 2x2. Triangles fill the
  `TexQuad` from barycentrics (`TriTex::quad`); lines, points and smoothed
  polygons give one pixel plus helper pixels one step right and down
  (`Frag::dx`, `dy`).
- Two function pointers, chosen at latch: `Latch::addr[axis]`
  (`addr::<WRAP, AXIS>`, 12 variants) turns indices into wrapped, tiled
  offset parts (a tiled offset is a column part plus a row part, so a
  bilinear footprint is two plus two, not four addresses) and marks
  border-colour indices; `Level::gather` (`gather::<BYTES, FIELD, CLASS>`,
  8 storages x 6 classes) reads and decodes a window, class expansion
  included.

## What went

`sampler.rs` (Sampler, refusals, `texture_setup`), `TextureChain`,
`TexelTables`/`TexelMemory`, the invented slice tables, `Tex`, the per-thread
texel cache and prefill, the planes fallback and the f64 footprint, and the
texture-validity drops in setup and prims.

## Cost

Quake timedemo replay, first 20 M records, host time (2026-11-04): the
first rewrite (per-tap decode, no cache) 146 s; monomorphised address and
gather per pixel 138 s; with the quad window 120 s, about the old decoded
texel cache's speed (scaled from 590 s for the whole replay). Zeroing the
2 KB window per level per quad cost 25 s of that: keep it uninitialised.
The goldens moved by at most one 8-bit step from the cached sampler, and by
up to 3 in the detail scenes (the 12-bit filter); the window is bit-exact
with per-pixel gathering.
