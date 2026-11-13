# Plan: the VPro fragment pipeline from hardware state, in quads, on threads

Status: 2026-11-04, in progress. Milestones 1 to 3 landed, 4 and 5 in
part, 6 (one thread per stage) and 9 (legacy paths retired: no GL
context, no SoftDraw; the board's words go to primitive assembly,
`assembly.rs`, and the stage records). The texture path was rewritten to
the registers-only rule (`clanker/odyssey/texture-unit-registers-only.md`):
texture registers in draw state slots (`pipe/slot.rs`), one latch, and
`Unit::sample_quad`. Next: 7, fixed point for the rest; the raster and
back-end words into the slots too; scaling threads (8).

## Goal

Replace the VPro (Odyssey) software rasteriser's fragment pipeline with one
that reads the board's own state words, grouped by the stage that reads
them (rasteriser, shader, back end), walks primitives into 2x2 quads over
8x8 tiles, and stores pixels through packers that build each plane's
pixel once. Generic branchy functions first; const-generic
monomorphisation (as the REX3 JIT's key does) or a Cranelift JIT later.
Then the three stages go on threads behind a quad ring, the arithmetic
moves to integers and the quads to SIMD lanes.

The workload is Quake 1's `timedemo demo1` on the 6.5.30 guest at
`ignore/ip30-ody` (1280x1024, textured, depth tested), recorded as
`ignore/recordings/quake1-timedemo.rec` (81.8 M records, 165 checkpoints
in `src/dev/odyssey/testdata/replay/quake1-timedemo.hashes`, blessed at
the pre-rewrite model). Quake 2 and TyrQuake are installed in the same
guest under `/usr/games/`.

## What the model does today, and what stays

- The board thread decodes each state block into GL calls on a GL context
  ported from qemu-sgi-origami (`gl_context.rs`): the blend word becomes
  `glBlendFunc` arguments, the raster enable word a set of `glEnable`
  flags, and the rasteriser (`raster*.rs`, about 7,000 lines) draws from
  that GL state plus a `RasterRules` struct of board facts. The raw words
  are mostly kept on the board already (`pattern_mode`, `gl_depth_func`,
  `gl_write_mask`, `blend_word`, `const_colour`, `gl_fog`, `gl_clip_box`,
  the texture blocks in `tex`, the window words in `win`); the alpha,
  stencil, logic-op, polygon-mode, polygon-offset and line/point words
  were decoded on arrival and not kept.
- The transform side is the same kind of HLE: matrices, lights and
  materials in the GL context stand in for the transform engine's
  microcode. **It stays** for this plan. Its output, window-space
  vertices with colour, texture coordinates, 1/w and the depth divisor,
  is the pipeline's input either way. Replacing it with a transform
  engine model is a separate project.
- The planes are byte arrays written a byte at a time through cells with
  a packing enum per plane (`memory.rs`, `GlPlaneRow`). That path is what
  the back end's packers replace.
- The hardware does what the quad pipeline does (US 6,259,460): the
  raster unit makes perspective-corrected barycentrics per pixel and
  hands 2x2 quads on with the triangle's attributes, the shader
  interpolates colour in 12-bit modules and depth in floating point, and
  the per-fragment tests come last. The driver drains the board before
  every state block (settle groups), so no state is in flight behind a
  primitive on the real board either.

## The state, by stage (`src/dev/odyssey/pipe/state.rs`)

Each record holds the words themselves; accessors decode the established
fields (`ignore/odsy/docs/tokens.md`, `words.md`) and nothing more. A
field the notes leave unknown is reported through the existing
unmodelled-word reports, not given a meaning.

| Stage | Words |
|---|---|
| Raster | 0x404 mode (cull, winding, flat, stipple, smooth, offset enables), 0x415 polygon mode, 0x458 polygon offset, 0x81d clip box, 0x841 origin, 0x204b clip rectangles, 0x840 id compare, 0x84a line and point size, 0xc48 line stipple, 0x3c4e/0x3c4f pattern, 0x43f pattern shape, 0x420 depth function (for the pre-test) |
| Shader | 0x404 mode (fog, alpha test, flat), 0x405 path (texture target, clip planes, colours per vertex), 0x1c16 sampler, 0x141b environment, 0x1013 level of detail, 0x203b scale and bias, 0xc3e offset, 0x406 level 0, the level descriptors, 0x181a fog, 0x41e alpha test |
| Back end | 0x404 mode (depth test, stencil test, blend, dither, logic ops), 0x420 depth function, 0x452 write mask, 0x41f stencil, 0xc21 blend (3 words), 0x422 logic op, 0x1017 destination select, 0x1023 plane mask, the colour-mode words, and the plane descriptors (format, packing, base, stride) of the colour buffers, depth, stencil and id planes |

`Board::fragment_words()` captures the three records at the point a draw
job is built today (`gl_draw`), so they travel with the job.

## The stages

- **Raster.** Vertices snapped to 1/256 pixel (established), three exact
  integer edge functions, an 8x8 tile walk with trivial reject and trivial
  accept, quads within a tile; lines and points to quads by the same
  record (centre sampling along the major axis, established; eight
  subsamples for the smooth modes, inferred from the patent). Per quad:
  coverage, barycentrics, window depth, the depth pre-test (test only,
  never a write, so it is safe against stale depth under threads).
- **Shader.** Perspective-corrected interpolation of colour and the
  clip-plane signs from the barycentrics; texture coordinates as plane
  equations with one divide (established on the frames); the fixed-point
  sampler ported, not rewritten (it is the part nearest the hardware:
  1/256 texel positions, 8-bit weights, the diagonal level of detail per
  quad); fog from the fog block's 12.8 coefficients; the alpha test on
  the 12-bit reference.
- **Back end.** Stencil and depth tests and operations from the words,
  blend and logic op on 12-bit components, colour mask, and packers: one
  function per plane format that builds the 32-bit, 16-bit or byte pixel
  and stores it with one write, chosen once per draw. Depth is the 24-bit
  float word; stencil a byte.

## Milestones

1. **State records.** Landed: `pipe/state.rs` with the three records,
   their decoders, the six words the board did not keep (alpha, stencil,
   logic op, polygon mode, polygon offset, line/point size), and
   `Board::fragment_words()`. Tests: every established field lands where
   its stage reads it.
2. **Back end.** Landed: `pipe/backend.rs`: `BackendState` from the words,
   `BackendPlanes` over a job's spans, packers (RGBA8, Z24, index) with
   one write per pixel, the fragment test-and-store; 24,000 random
   fragments match the old `rgba_result` for every blend factor,
   equation, logical operation and mask.
3. **Raster.** Triangles landed: `pipe/raster.rs`: `RasterState`,
   `Triangle` setup, the 8x8 tile walk into `Quad`s with the depth
   pre-test; 40 random triangles (clipped into fans as the old path
   clips) give the same pixels and depth words as the old walk. Lines and
   points to quads are next; then the nine odydiags frames (hardware truth
   for positions, the alpha test and nearest texels) through
   `ignore/odsy/scripts/odydiags_frame_check.py`'s rule set.
4. **Shader.** `pipe/shader.rs`: interpolation, the ported sampler, fog,
   alpha test.
5. **Switch-over.** Draw jobs run the new pipeline; odydiags frames
   bit-exact, the replays and the 78 golden scenes under tolerance; the
   old `raster*.rs` deleted; `debug.md` and the skill updated.
6. **Threads.** Three threads, one per stage (`odyssey-raster`,
   `odyssey-shader`, `odyssey-backend`), on one ring of 64-byte quad slots
   with three cursors (`pipe/ring.rs`, as first designed; c1ca14b7 had a
   ring per hand-over, copying each quad). The shader shades slots in
   place; quads the raster thread shades itself go in already shaded.
   Drains at every `raster_wait`, completion, non-draw job and (for now)
   early-z mode change; none when the queue runs dry (aec14e3c).
   `IRIS_ODSY_SHADER=inline` / `IRIS_ODSY_BACKEND=inline` fold the stages
   back into the raster thread. Replaced 2026-11-04 by a worker pool, then
   by screen tiles with one owner each (`clanker/odyssey/screen-tiles.md`):
   about 4.6x the three stages on the Quake replay.
7. **Integers, SIMD, specialisation.** In progress: every Quake fragment
   now samples in fixed point (detail images, af641c26); the shader's
   colour, environment, fog, alpha test and the detail combine are 12.12
   (`pipe/fixed.rs`). Per pixel there is no double left on the common
   path. A triangle is one record for its life (`TriShade`: the walk's
   integer edge setup, its vertices once, the shading constants); the walk
   gives every pixel its barycentrics, which the ring's slot carries, and
   depth, colour weights and texture positions all come from them and the
   vertex values (`pipe/interp.rs`), with no plane fitted; one division
   per pixel in 32-bit floats, integers after it.
   Per-triangle setup stays in double. Speed did not move: what is left is
   the texel cache, the walk and the back end, which SIMD over a quad's
   four lanes and specialisation address. Decided (2026-11-03, with the user): colour stays perspective
   correct; colour is 12.12; shader and texture arithmetic fit in 32 bits
   as a rule (a wider product is the exception, and says why). Then 12-bit colour in 16-bit lanes,
   fixed-point weights as `mgras/fixed.rs`, depth stays floating point;
   `fearless_simd` (already in the lock file via `vello_common`) for the
   four lanes of a quad; then const-generic keys or a JIT for the hot
   combinations. Judged against the `blend.SUM` residue (332 one-step
   pixels today).
8. **Scaling the stages.** Only after 7: the per-stage costs it leaves
   decide which stage gets more threads. Shader workers dealt quads
   round-robin need the back end to restore command order (a reorder
   buffer); screen-space partitioning of the back end, as GPUs split
   their render back ends, keeps order per pixel without one. Not before
   the fixed-point stages exist, or the tuning is done on the wrong code.

9. **Retire the legacy paths and the GL enums.** Every plane write goes
   through the ring and the stage records, and nothing converts the
   board's words into GL state and back. To move: clears and rectangle
   operations (`gl_raster_ops`, `RasterOp`) as back-end quads or spans
   (RGBA clears done: span slots in the ring, `backend::ClearDraw`, no
   drain; the clear still takes its values from `RasterOp`, built from
   the board's rectangle words and some GL state, and index and aux
   clears still go the old way);
   lines, points and smoothed polygons walked in `pipe/raster.rs`; the
   image and copy paths; colour-index windows and auxiliary targets.
   Then delete the old fill loops (`raster_fill.rs`, `QuadRow`,
   `QuadGeneral`, `SoftTexSpan`), the row lending (`GlPlanes::row`,
   `GlPlaneRow`, `RowsMut`), `SoftDraw`, `SoftFrag`, `SoftVertex`
   beyond what the transform side hands over, the double samplers once
   no fallback needs them, `IRIS_ODSY_PIPE=old`, and the GL enums the
   stages still match on (`GL_REPLACE`, `GL_LINEAR_DETAIL_*`, `GL_REPEAT`,
   formats, `GlDrawState` reads listed below): each stage decodes the
   board's own fields (texture environment code, filter and wrap codes,
   formats) once per draw into its record. The transform HLE keeps its
   GL context; the boundary is the words it hands the stages.
   After this, 12/12/12/12 colour storage for 12-bit windows: the
   planes widen (every writer and the display read them), which is one
   change once the ring and the stage records are the only writers, and
   five while the legacy paths store bytes of their own.
10. **Display: drop per-row dirty tracking.** The raster unit, the back
   end and the display keep per-row bitsets of what changed
   (`TouchedRows`, `DrawnRows`, `PublishedRows`, `RowMarks`) so the
   display resolves only changed rows. No other board model does this;
   it costs the back end (17% of its cycles before rows were published
   in batches) and adds code where a missed mark leaves stale rows on
   screen. The user suspects the occasional missing text in a console
   window comes from it: check that first (a missing mark would show
   exactly that). Then resolve whole frames and let the compositor deal
   with the screen, as the other boards do.

11. **Verification against hardware.** A real Octane2 with a V10 is on
   the LAN (ask the user for its address and login; never write them into
   the repository). Directed tests: build `test/gltest` there and run the
   same cases on the emulator and the board, comparing read-back pixels
   (glReadPixels) case by case; settle the open precision questions this
   way (blend rounding at 12 bits, NEAREST ties, the 12-to-8-bit store,
   subpixel snap, depth word rounding, fog). As root, read the board's
   registers and, where it is mapped, board memory, to compare plane
   layouts and formats. Final verification: piglit, extracted from a mesa
   clone into its own repository (`piglit-irix`), ported to build with
   MIPSpro on IRIX 6.5, run on both, results compared test by test.
   First facts from the board are in `clanker/odyssey/v10-hardware-limits.md`:
   the model's fixed 1280x1024 RGBA8 planes are a model choice (the board
   runs 1024x768 here, its viewport reaches 3840, pbuffers are sized by
   the display mode and memory, and RGBA windows come in 4:4:4:4, 5:5:5:1,
   8:8:8:8 and 10:10:10:2). Accumulation (hardware 24-bit via the blender
   and a scratch buffer) is VPro's full-scene antialiasing and the model
   has none.
   First comparison (`clanker/odyssey/v10-vs-emulator-first-comparison.md`):
   the aliased line, stipple and point rules match the board pixel for
   pixel, so milestone 9 ports them bit-exact; smooth lines and points
   need fitting to the board; the pixel paths have about 45 failures to
   fix; the board's depth read-backs are less precise than the model's,
   which stays as it is (decided: more precision, no cost).

12. **Depth metadata: fast clears and hierarchical Z (deferred; design
   only).** Not built now: clears write the depth plane (9aeaf4aa) and
   the pre-test reads it per pixel. If it comes back, the shape is:
   - One table beside the depth plane, an entry per 8x8 tile (the walk's
     tile): 1280x1024 is 20480 entries of 8 bytes, 160 KB, in cache. An
     entry holds a state (expanded, or cleared to the draw-independent
     clear word it carries) and the tile's nearest and farthest stored
     words (`zmin`, `zmax`; words compare as integers).
   - One writer: the back end, in ring order, as it is for the plane. A
     clear span covering whole tiles sets them cleared (no plane write);
     the first fragment stored into a cleared tile fills it with the clear
     word first. Partial tiles are filled as now. `zmin`/`zmax` move only
     as far as a store proves (a full-tile clear sets both; a store
     widens the range; nothing ever narrows it except a clear).
   - Every other reader resolves through one call: the front end's depth
     reads and DMA (`resolve(rect)`: fill the cleared tiles it touches,
     under the drain those paths already take), the per-pixel pre-test
     (reads the clear word of a cleared tile instead of the plane).
   - Hierarchical Z in the raster stage: before walking a tile, compare
     the triangle's nearest depth over the tile with the tile's `zmax`
     (LESS/LEQUAL; `zmin` for GREATER/GEQUAL) and drop the tile. The table
     lags the raster thread by the ring, which is safe only when stale
     entries are conservative: the same rule as today's pre-test (one
     comparison direction since the last drain, `pretest_less`), and no
     reject while a depth clear is still in the ring (`depth_clear_at`).
   - Off for stencil-writing draws, depth-write-off draws keep the entry,
     a change of the depth plane's format or size drops the table.
   Measure first: it pays only where clears or overdraw dominate, and
   Quake's frame is neither.

## What the transform side hands over (`assembly::XfState`)

- The model-view, projection and texture matrices, the viewport (which
  bounds the walk), the depth range and the user clip planes.
- The depth row and divisor (`gl.rs::DepthRow`) and whether vertices
  carry back colours.

## Measurements

| What | Pre-rewrite model (2026-11-02) |
|---|---|
| Quake 1 timedemo, guest | 969 frames in 919 s, 1.1 fps |
| Replay of the recording, host, old binary | 917 s (the guest's own time: the board was the bottleneck) |
| `raster_bench` flat fill / smooth depth / textured bilinear depth | 52 / 22 / 9.5 Mpix/s |
| Quake replay, fixed sampler for alpha test, inline / raster+shader+back-end threads | 788 s / 665 s (raster and back end alone 736 s) |
| Where the double sampler's draws came from (Quake, first 20 M records) | every refused draw: LINEAR_DETAIL_COLOR_SGIS magnification with a detail image over LINEAR_MIPMAP_LINEAR; 254 M of 634 M textured fragments |
| Quake replay, fixed sampler takes detail images (af641c26), inline / three threads | 508 s / 386 s |
| ... and every fixed triangle sampled per quad in the pipe (bit-exact) | 475 s inline |
| Profile after both, first 20 M records inline (`perf record -e cycles:u` works; kernel sampling does not) | tri_shade_quad 29%, clipped_fan (setup and walk) 17%, back end 11%, tri_prep 11% before the per-quad change, ShaderState::shade 7%, fixed_detail 5% |
| Drains on the replay before the in-flight count (aec14e3c), first 20 M records: draws / drained when the queue ran dry / other jobs / object-store waits | 341 k / 106 k / 29 k / 12 k |
| Quake replay, in-flight count, three threads | 368 s |
| Quake replay, integer shader (12.12 colour, environment, fog, alpha test, detail), inline / three threads | 466 s / 350 s |
| Profile after the integer shader, 20 M records inline | tri_shade_quad 41% (sampler taps, the perspective division and footprint in double), setup and walk 19%, back end 12%, fixed_detail12 5% |
| Perspective unit (f32 planes, one division per pixel, integer positions, weights and level of detail) and 32-bit depth planes, 20 M records inline, three builds at once | 98 s, against 100 s for the integer shader alone: no measurable change; the double arithmetic was not the cost |
| Full replay inline, alone, perspective unit and 32-bit depth | 486 s (466 s for the integer shader alone, run on another occasion) |
| Profile after the unit: tri_shade_quad 45%, clipped_triangle (setup and walk) 18%, back end 12% | the cost is spread over the texel cache's tag checks, the walk, the unit's lanes and the back end, no line above 3% |
| Single ring, three threads (shared the machine with a second replay) / shader inline | 366 s / 468 s, every checkpoint; the shader is the bottleneck (the raster thread spends ~20% waiting for room) |
| One triangle record, barycentrics in the slot, no plane fits: inline / three threads | 465 s / 329 s, every checkpoint (NEAREST picks the other texel where a pixel centre lies exactly on a texel edge in two mip scenes) |
| 12-bit back end (slot carries 12-bit components, integer blend on a widened destination): inline / three threads | 477 s / 340 s, every checkpoint; unblended pixels unchanged, 20 blended scenes move a step |
| Where the raster thread's time went, first 20 M records inline | 183 s: front end 9, walk and stores 36, shading 137 |
| Quake replay, pipeline inline / raster and back-end threads (2026-11-03) | 828 s / 772 s, all 165 checkpoints both ways |
| Pipeline, generic branchy stages, against the first draft's M1 (`BENCH_PIPE=old`) | flat fill 0.48, smooth and textured 0.65 to 0.92, lines and smoothed polygons 1.27 (new packers) |
| First draft's M1 (per-quad stages inside the old fill, record round trip) | replay 933 s, every checkpoint matched; `raster_bench` 0.71 flat, 0.85 smooth, 0.97 to 1.04 textured of the above |

## Idea: a quad worker pool (the user, 2026-11-04)

Instead of one shader and one back-end thread, balance the work across a
pool:

- The rasteriser stamps quads into a buffer. As soon as it has about 16
  quads, or finishes a primitive, a pool thread takes the batch and runs
  each quad to completion: shade, test, store.
- When the pool is dry, the rasteriser runs a single quad to completion
  itself, rather than wait.
- Order: the threads complete quads in command order at the back-end
  stage through atomics. Either one sequence counter, or a 4x4 array of
  counters indexed by the quad's position in an 8x8 grid (quad x & 3,
  quad y & 3), so only quads at the same screen place wait for each other.
  Quads elsewhere complete out of order, which nothing can observe.

Questions to settle first: the cost of the per-quad order check against
the current ring's batch cursors, the state lifetime (a draw's
`BackendDraw` is held until every quad of it is stored, as `hold` does
now), and texel reads racing an upload (`Board::raster_wait` still
drains everything).

## Risks

- The 78 golden scenes pin the old model's pixels, not the hardware's;
  re-blessing them is expected where the frames do not pin a value.
- The display reads rows lock-free; the back end must publish touched
  rows as `RasterCtx::publish` does now.
- Everything that is not a triangle, line or point (bitmaps, images,
  copies, clears) keeps its path until the back end's packers serve it.
