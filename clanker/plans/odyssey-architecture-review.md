# The VPro (Odyssey) model — architecture review after the merge

**Status:** review, 2026-11-02. **Range:** `c11cb062..bf5f78b9` (six commits
by James Braid, 2026-10-29 to 2026-11-02), `src/dev/odyssey/` 38,826 lines
in 39 source files, 293 unit tests, plus `golden/` (2,405 lines) and
`testdata/`. **Origin:** a port of `hw/misc/sgi-odyssey.c` and the shared
`hw/display/sgi/` code in jamesbraid/qemu-sgi-origami; every file header
names the QEMU file it came from. **Hardware notes:** `ignore/odsy/`
(James's reverse-engineering tree: `BUZZ.h`, `PBJ.h`, `docs/*.md`).

This note answers the questions asked of the merge: how the board is
driven, the thread and front-end/raster split, debug and regression
tooling against the other boards, overlap with GR2 and MGRAS geometry,
float versus fixed point in the raster, the texture path and its
specialisations, JIT feasibility, and whether the raster is tiled.
James's own design notes are `clanker/odyssey/threads-and-locks.md`,
`bringup.md`, `gotchas.md`; read them first, this note does not repeat
them where they already answer.

## 1. How the board is driven

**Commands by PIO stores, data by DMA.** There is no linear frame buffer
and no command DMA. The processor stores command words into two
command-FIFO ports in the widget's MMIO window; those stores are the whole
GL and 2D stream (vertices, colours, state blocks, primitive begin/end,
texture level tables). libGLcore, the X module and the PROM textport all
write this stream directly (`ignore/odsy/docs/words.md`, `opengl.md`).

DMA is used only for bulk pixel data: texture level uploads, glDrawPixels
image uploads, readbacks (glReadPixels, the kernel's context save), query
results, and fence/sync words that the board writes into guest memory. A
DMA is itself a command: a descriptor opcode in the FIFO (`DESC_UPLOAD`,
`DESC_READBACK`) with four operand words naming a page table in guest
memory (`command.rs` ~1049, `dma.rs::dma_descriptor`). The board walks the
table, reads the header (destination buffer or object, format, row
length, control bits), and copies through `pixel_dma.rs` in page-sized
chunks. Completion is a fence word written back and, when the header asks,
an interrupt packet to HEART.

So the data path is: CPU store → `GFifo` ring (64K entries) → front-end
thread decodes → either integer/2D draw directly into planes, a GL draw
packaged as a job for the raster thread, a DMA walked synchronously on the
front-end thread, or a register write. The guest sees a 32,768-word FIFO
with low/high-water bits (a model choice; the hardware FIFO depth is
unknown, see `fifo.rs`).

**Two-processor Octane.** The board accepts stores from two CPUs; the ring's
own lock orders them and a small `submit` lock keeps two processors from
both passing the "FIFO drained" check and borrowing the board. The board's
own threads never take that lock.

## 2. Threads and the front-end / raster split

Four threads touch the board (`mod.rs` header, `threads-and-locks.md`):

| Thread | Hardware stand-in | Does |
|---|---|---|
| CPU thread(s) | widget accesses | push FIFO stores; front registers answered at once; everything else answered `BUS_BUSY` until both queues drain |
| `odyssey-work` | BUZZ front end (GFE + transform) | decode FIFO words, register writes, matrix loads, lighting, primitive assembly, clipping, texture setup, DMA, 2D drawing, object store |
| `odyssey-raster` | BUZZ raster unit | triangles/lines/points, texturing, fragment ops, plane writes, fences and query publication |
| `odyssey-display` | PB&J | retrace at 60 Hz, swap latch, colour map, cursor, resolve to the window |

Yes, there is a front-end/RE split, and it is one level deeper than
MGRAS's. The split point follows the patent (US 6,417,858) that puts a
FIFO between transform and raster. What crosses it is `GlDrawJob`
(`gl.rs` ~899): raster rules, draw state, a vertex buffer, the texture
chain and texel tables, and plane spans. Jobs are recycled through a
return ring so a draw allocates nothing. Non-GL raster work (clears,
pattern rectangles, fences, queries) crosses as closures.

The design mirrors MGRAS's threaded refactor (`clanker/mgras/DESIGN.md`):
`UnsafeCell` state owned by one thread, raw plane pointers for the display,
the shared `GFifo` ring, `BUS_BUSY` back-pressure. The differences:

- One raster head, not two. `RasterQueue<const HEADS>` is written so a
  second head could be added; the Maximum IMPACT model's scanline
  interleave is the pattern to copy if a dual-BUZZ board is ever wanted.
- The front end is bounded by *estimated pixels* queued
  (`RASTER_WORK_BUDGET`, 2^17), not by job count. That was measured
  (Smoke & Mirrors swaps/s) and is the right knob; IMPACT bounds on ring
  depth.
- Front-end plane work (2D, DMA into planes, object store) waits for the
  raster queue to drain (`Board::raster_wait`). A texture DMA into an
  object that already has rows is instead staged as a raster job
  (`object_stage`), so uploads do not stall the pipeline. This is the
  same hazard class as IMPACT's "stale block flushed by the display
  thread" and is handled better here.

**Assessment.** Sound and consistent with the house rules in `HACKING.md`
("no locks the hardware doesn't have"). The `unsafe impl Sync` contracts
are written at each cell. The one place to keep an eye on is the display's
lock-free row copy: it is correct by the "mark after store, take marks
before copy" argument, but it depends on every plane writer marking rows.
A new writer that forgets the mark shows as a frame that never refreshes,
not as a crash; the `vpro dirty` counter is the diagnostic.

## 3. Debug, tracing, recording and replay

| Facility | VPro | IMPACT (MGRAS) | GR2 | REX3 |
|---|---|---|---|---|
| Monitor commands | `vpro refusals`, `vpro dirty` only | `mgras status|hq|gl|ge|ucode|vc3|xmap|cmap|dac|pix|stats|jit|fbdump|shot|dump|trace|rec`, `rss regs|pix` | `gr2 status|hq|ge|shram|ucode|vc1|xmap|clut|dac|pix|trace`, `re3 regs|pix` | `rex status|jit|debug|cmap|trace`, `vc2`, `xmap`, `cmap`, `bt445` |
| Perfetto | `VPro cfifo level`, `VPro irq`, `VPro display/capture/present`, `slice`/`drain`, `raster slice`/`raster wait`, `dma descriptor` | yes | yes | yes |
| Recording | `IRIS_ODSY_REC=<file>` at build time only; gzip; W/R/B/b records (bus stores, non-repeated loads, DMA reads) | `IRIS_MGRAS_REC` **and** `mgras rec <file>|mark|off` from the monitor; W/R/M/m/F records | `.rec` via monitor | `.rec` via monitor |
| Replay test | `odyssey::golden::replay`, hashes every 500,000 records, `VPRO_REPLAY_OUT` PNG | `mgras_replay_file`, `MGRAS_REPLAY_TRACE` annotated trace, `mgras_rebaseline_file` | yes | yes |
| Draw-level goldens | 78 scenes through `Board::gl_draw`, bit-exact hashes with a per-scene tolerance mode, PNGs in `testdata/golden/frames/`; vertex goldens for 6 matrix and 6 light setups | power-on replay goldens | pixel checks in `irix-gfx-loop` | pixel checks |
| Benchmarks | `raster_bench` (fill rate per scene, `BENCH_FILL=table|dynamic|off`), `display_bench` (resolve) | sweep benches in `rss_jit_tests` | re3 bench | gfifo bench |
| Interactive dump | none | `fbdump`, `shot`, `dump` | `pix` | screenshot via CI socket |

**What matches established patterns.** The record format is the MGRAS one
with the tags renamed; the headless replay with hash checkpoints and a
bless step is the same workflow; recordings stay out of git. The display
reader, the row marks, perfetto tracks and the `GFifo` are the shared
pieces. The draw-level golden net is *new* and better than what the other
boards have: it tests the rasteriser from a Rust entry point with no
guest, so a change to a fragment rule fails in seconds with a PNG diff.
Worth copying to IMPACT and GR2.

**What is missing against the pattern.**

1. No monitor state inspection. There is no way to print the GL draw
   state, the texture words, the current context, the FIFO level, the
   object store or the planes from the monitor, and no screenshot or
   plane dump. Every other board has these and `irix-gfx-loop` depends on
   them. Add a `vpro` subcommand set modelled on `mgras`: `status`,
   `fifo`, `gl` (GlDrawState and texture sampler), `objects`, `pix x y`,
   `fbdump`, `shot`. Update `debug.md` and `iris_mcp.py` in the same change.
2. Recording cannot be started or marked from the monitor. Add
   `vpro rec <file>|mark|off` as MGRAS has; the `Recorder` already
   flushes periodically so the plumbing is small. A `mark` record would
   also let the replay test checkpoint at meaningful points rather than
   every 500,000 records.
3. No per-draw trace. IMPACT and GR2 have a `trace` command that logs
   decoded tokens; the VPro has only `dlog_dev!` lines under `developer`.
   A FIFO-word decoder trace (word class, block id, operands) switched
   from the monitor would make the `words.md` findings checkable live.
4. `debug.md` does not list the `vpro` command. It must (CLAUDE.md rule).
5. Snapshots: `Odyssey` has no `Saveable`, and the Octane refuses save
   with a VPro fitted (CHANGELOG). IMPACT is in the same state, so this
   is parity, not regression.

## 4. T&L: overlap with GR2 and MGRAS

The three boards share `src/dev/gl/` (1,169 lines: `math.rs` matrices,
`vertex.rs` clip planes, polygon clipping, viewport mapping; `light.rs`
the OpenGL 1.1 lighting equation and fog). The VPro uses it for:

- lighting (`geometry.rs::gl_light_fill` through `dev::gl::light::Lighting`),
- the modelview and projection transform (`dev::gl::math::xform`, f32),
- the view-volume clipper (`dev::gl::vertex::Clip` with the `NearW`
  policy, `VolumeVertex: ClipVertex`),
- the viewport mapping (`Viewport::project_by::<NearW,...>`).

So the generic geometry is already deduplicated; James added the `NearW`
policy and `Wv` generics to `vertex.rs` for it rather than forking. What
is board-specific and should stay so:

- **Program recognition** (`geometry.rs::Program`, `gl_program`): the
  VPro's libGLcore uploads transform microcode and the model recognises
  which resident program is running (lit, two-sided, texgen, clip
  planes) from the entry words and constant-memory list. GR2 and MGRAS
  do not execute microcode either but their token protocols name the
  state directly; nothing to share.
- **Depth model**: the VPro's window depth is a `depth_row` dot product
  with eye coordinates, perspective-corrected per fragment, stored as a
  24-bit unsigned float (`depth_word.rs`). The others use a normal
  Z-buffer. Keep separate.
- **Window matrix mode** (`GlDrawState::window_matrix_given`): the X
  server's eye-to-window transform that replaces projection + viewport.

**Deduplication candidates, ranked by payoff:**

1. `GlDrawState` (gl_context.rs ~451) is a third copy of "the GL
   fragment state a draw is made under". `mgras/gl.rs` and `gr2/gl.rs`
   each keep their own. A shared `dev::gl::state::FragmentState` with
   the OpenGL 1.2 fields (depth, stencil, blend, logic op, fog, alpha,
   colour mask, scissor, cull, polygon offset, clip planes) and each
   board layering its own extras would let the three JIT keys and the
   three validity checkers (`gl_blend_factor_valid` and friends exist in
   all three) converge. Medium effort, mechanical, well covered by tests
   on all three boards.
2. Primitive assembly (`gl_context.rs` glBegin/glEnd, strips, fans,
   quads, polygon `GlPolygonAssembly`). GR2 does the same on
   vertex-buffer indices, MGRAS on its own vertex records. A shared
   assembler over a `VertexSource` trait is feasible but the per-board
   provoking-vertex and edge-flag rules differ in detail; lower payoff.
3. Polygon offset, depth range and fog factor helpers are small and
   duplicated three ways; trivial to lift into `dev::gl`.

Do **not** move the VPro's clip/transform onto a single `dev::gl` path
without the far-plane test `gotchas.md` describes: the depth maths is
f64 on purpose.

## 5. Float or fixed point in the raster

**Double precision, with one fixed-point fast path.** `raster.rs` has 104
uses of `f64` against 15 of `f32`. Vertex positions snap to an integer
subpixel grid (`SoftVertex.x/y: i64`, `subpixel_bits`), edge functions
are exact `i64`, but barycentrics, `1/w`, depth, colour, texture
coordinates, blending, fog and the texture environment are all `f64`.
The texture sampler on the general path (`raster_texture.rs::texture`) is
`f64` too. Only `raster_fixed_texture.rs` is integer: texel positions in
1/256, 8-bit bilinear weights, lambda in 1/256 through a 10-bit log2
table, four 8-bit channels packed in 16-bit lanes of a `u64` so one
multiply weights all four. It is documented as "faster, not more
accurate", within 3/255 of the double path, and falls back per fragment.

MGRAS is the opposite: `mgras/fixed.rs` models RE4/TE1 as 12.16 fixed
point in `i32` because the hardware is an integer machine and the
microcode's constants are known. The VPro's arithmetic widths are
*unknown* (`ignore/odsy/docs/open-questions.md`: "Filter-weight and
sub-texel widths are in no published source"; the colour interpolation
residue against the `odydiags` simulator frames is unexplained at
332/13,934 pixels). So the f64 choice is honest: it is the reference
until the hardware's widths are recovered, and the golden tolerance mode
exists precisely for the day they are and pixels move.

**Sharing with MGRAS.** Little at the arithmetic level today, because one
is f64-as-reference and the other is bit-modelled fixed point. What can be
shared is structural:

- the texel cache and level-selection scaffolding
  (`SoftTexelCache`, `soft_texture_select`) versus `te1.rs`;
- blend factor / logic op / stencil op tables (`soft_blend_factor`,
  `soft_logic_op`, `stencil_op`): identical GL semantics, three copies;
- the `Planes` trait (`load_rgba/store_rgba/load_depth/...`) is a good
  shape for a shared plane abstraction; `rss.rs` reaches `pixmem`
  directly.

If the `odydiags` residue work in `ignore/odsy` lands on a fixed-point
model of the colour interpolator, the right move is to port `fixed.rs`'s
*style* (one module of named widths, interpreter and JIT both built on
it), not its constants.

## 6. The texture path and the fragment/texture split

**Yes, the split is orthogonal.** Texture *object* state (format, levels,
filters, wraps, border, LOD limits, detail/sharpen tables) is decoded
once per draw from the board's texture words by `sampler.rs::texture_setup`
into `GlTexImage` + `TextureChain` + `TexelTables`, carried on the job,
and reached through the `Texels` trait (`texel`, `border_texel`,
`texel_3d`, `post_filter`, `filter_weight`). Fragment state
(`GlDrawState`) holds only the texture *environment* (mode, colour) and
the enables, as the GL spec partitions it. The `Fill` trait's const
parameters mix the two dimensions on purpose (KERNEL, REPEAT, FORMAT from
the sampler; ENV, DEPTH, BLEND... from the fragment state) because that
is what a specialised loop needs; the inputs stay separate.

**Per-draw flow.** `clipped_triangle` projects three vertices, computes
the texture planes (S/W, T/W, Q/W as plane equations), culls, then picks
one of three row loops: the fixed-point textured span
(`fixed_span_select`), the untextured fill span (`fill_select`), or the
general per-fragment path (`fragment_at`) for anything else (alpha test,
logic op, polygon stipple, non-REPEAT/EDGE wraps, 3D textures, sharpen
and detail, index mode). Texels are optionally prefetched per draw into
flat per-level arrays (`fixed_prefill`, above 2,048 pixels) so the inner
loop reads a `Vec<u64>` rather than board memory through the object
store.

**The specialisations** (`raster_fill.rs`): a `FillMode<13 const
params>` type implements `Fill`; `fill_quads<M: Fill, P: Planes>` is the
*one* loop body, and `FILL_DYN` in a parameter makes the accessor read
the state at run time. 43 instantiations are enumerated in `fill_rows!`:
24 textured (nearest/linear/trilinear × replace/modulate × RGB/RGBA ×
LESS/LEQUAL), one edge-wrapped trilinear for perfly's truck, two blended
ones for Smoke & Mirrors glass and perfly sky, 12 untextured (depth
none/LESS/LEQUAL × flat/smooth × blend off/over), and two "Any" fallbacks.
Tests compare every loop pixel for pixel against the general path.

**This is already the REX3 pattern.** `ng1/rex3_generic.rs` writes the
pipeline once against a `Mode` trait with `ConstMode` (const generics)
and `DynMode` (fields); `rex3_shaders.rs` is a *generated* table of 498
wrappers produced by `tools/gen_rex3_shaders.py` from a recorded profile
of modes seen. The VPro's `Fill`/`FillMode`/`FillDyn` is the same idea
written by hand with a 43-entry macro table and a hand-written selector.
Generalising it the REX3 way is straightforward:

1. Make the selection a key, not a decision tree. Replace
   `fill_select`/`fixed_span_select` with a `FillKey` struct (the 13
   fields plus the ones currently excluded: alpha test, logic op,
   stipple, stencil, clip planes, 3D) and a `pack()`; this is the VPro
   equivalent of `rex3_shape::unpack` and MGRAS's `PipeKey`
   (`rss_jit/key.rs`). The three `Any` fallbacks become "key not in
   table".
2. Profile the key stream from the replay corpus (Smoke & Mirrors,
   perfly, the IRIX desktop) and generate the `fill_rows!` table from
   it, as `gen_rex3_shaders.py` does. The hand-picked perfly truck and
   sky entries show the table is already being tuned by eye.
3. Widen what `fill_quads` admits. Today alpha test, logic op and
   stipple leave the loop entirely (`fill_admits`); REX3 folds those in
   as const arms. Lines and points (`raster_line.rs`) have no
   specialised loop at all and the X server's 2D path draws through
   `draw.rs` integer code; both are candidates once a key exists.

Effort: a few days for steps 1 and 2 because the loop body and the
equivalence tests already exist; the risk is compile time (43
monomorphisations of a large loop is already noticeable; 500 will need
the REX3 table's `#[inline(never)]` wrappers and probably a feature flag).

## 7. JIT feasibility

Three raster JITs exist on the same pattern: `ng1/rex3_jit` (3,819
lines), `mgras/rss_jit` (3,667, "modelled on the REX3 shader JIT"),
`gr2/re3_jit` (1,721). Each has a `key.rs` (mixed-radix `PipeKey`), a
`compiler.rs` (Cranelift straight-line pixel pipeline), background
compilation with interpreter fallback, `IRIS_<BOARD>_JIT=off|sync|async`,
and a sweep test proving bit-exactness against the interpreter.

A `odyssey/fill_jit` on that pattern is feasible and the groundwork is
better than MGRAS had when `gr4-jit` started:

- The pipeline is already written once against a trait with every
  decision hoisted (`Fill`), so the compiler emits exactly the
  `fill_quads` body with the `M::` accessors folded. The arithmetic is
  f64, which Cranelift handles directly; MGRAS's JIT had to reproduce
  12.16 shifts.
- Setup (edges, texture planes, lambda table, prefilled texels) stays in
  Rust and goes in a per-draw context struct, as `Rss::tri_setup` does.
- The equivalence oracle exists: the general path and the 43 loops are
  already compared pixel for pixel, and the 78 golden scenes plus two
  replay recordings pin the result.
- The job boundary is the natural call site: `GlDrawJob::run` on the
  raster thread, so the JIT never runs on the CPU thread.

Where to expect the work: the fixed-point sampler's lane tricks
(`u64` 16-bit lanes) and the texel cache need Cranelift equivalents or
helper calls; the per-quad LOD (`soft_quad_rho`) should be computed once
per quad in the JIT as it is now. The f64 divides per pixel
(barycentric `e/area`, `1/sum` for perspective) are the real cost in
the interpreter too, so measure with `raster_bench` first; the fixed
texture path already gave the big win and a JIT may buy less here than
on REX3, where the interpreter decided per pixel.

Recommendation: do the key/generated-table step first (section 6). It
costs little, is the prerequisite for the JIT key, and will show from the
profile which states actually matter.

## 8. Scanline or tiled

**Scanline, in 2x2 quads, with edge functions. Not tiled.** The model
walks a triangle's bounding box two rows at a time (`clipped_triangle`:
`qy` steps by 2 from an even row), solves each row's owned run from the
three integer edge functions (`owned_run`, exact since each edge is
linear in x), then `fill_quads` steps across the two rows in 2x2 quads.
A quad's four pixels interpolate together and share one level of detail
(`soft_quad_rho`, the diagonal-difference rule); each pixel is then
tested and stored on its own. The tie rule (`owns`/`owned_run`) gives a
shared edge's centres to the left or top edge of a counter-clockwise
triangle, with rows counted up from the bottom, so two triangles sharing
an edge never both draw it.

The hardware evidence in `ignore/odsy` supports a quad-based (not tile-
based) rasteriser: `docs/open-questions.md` records the quad LOD rule as
inferred from the `odydiags` simulator frames and cites the patent's
shared-edge rule "decided by consulting the neighbouring pixels of the
quad". `docs/raster.md` establishes centre sampling, the diamond-exit
line rule and `floor(256 c)` colour storage from CRC-matched frames, none
of which say anything about traversal order. The word "tile" in
`BUZZ.h` and `opengl.md` refers to the **memory layout**: board memory,
the FIFO and texture levels are described in 16-texel and 2/32-bpp
tiles. The model keeps planes linear (`memory.rs`: `base + y*stride +
x*bpp`) and does not model the tiled layout, which is fine as long as
nothing reads board memory through an address the guest computed
tile-wise. The object store (`objects.rs`) indexes textures by their
board address and width, so a DMA that lands in the middle of a tiled
level would show as a misplaced texel region; none of the recordings
show it.

Nothing in the model is a tile renderer in the deferred, bin-and-render
sense, and nothing in the RE notes suggests BUZZ is one.

## 9. Things to act on

Ordered by value per hour, all independent of each other:

1. Monitor surface: `vpro status|fifo|gl|objects|pix|fbdump|shot|rec|trace`
   plus `debug.md` and `iris_mcp.py` (section 3). Without it the
   `irix-gfx-loop` skill does not apply to this board.
2. `FillKey` + generated specialisation table from the replay corpus
   (section 6); keep `FillDyn` as the fallback.
3. Lift the shared fragment state and the validity tables into
   `dev::gl` (section 4, item 1), then let the three JIT keys build on it.
4. Copy the draw-level golden net (`golden/scenes.rs` style, no guest)
   to MGRAS and GR2.
5. JIT only after 2 shows where the fragments go and `raster_bench`
   shows the interpreter loop, not the divides or the texel reads, is the
   bottleneck.
6. Keep `ignore/odsy/docs/open-questions.md` "Colour interpolation
   arithmetic" and "Texture arithmetic" linked from `gotchas.md`: when
   either is answered, the f64 paths become a fixed-point model and the
   golden tolerance mode is the tool for the transition.
