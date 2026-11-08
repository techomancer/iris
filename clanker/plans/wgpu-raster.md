# Plan: the boards' raster units on the host GPU (wgpu compute)

Status: 2026-11-06, branch `wgpu-raster`, in progress. Milestone 0 landed:
the common layer in `src/wgpu` (device, shader assembly, batch recorder,
bin and raster kernels, GPU memory with readback) with an end-to-end test
board that matches a CPU reference word for word on the RX 6600 XT.
Milestone 1a landed: the VPro raster unit itself (`src/dev/odyssey/wgpu`:
records, slots, the triangle, fragment and clear kernels over the model's
word rule, the generic back end) matches the CPU pipeline word for word in
eight parity tests (`cargo test --features wgpu --lib odyssey::wgpu`),
after two GPU facts were guarded (`clanker/wgpu/`). Milestone 1b landed:
the unit is in the board (`[graphics] raster = "wgpu"`; `VPRO_GPU=1` for
the test boards, `VPRO_REPLAY_GPU=1` for the replay), with the CPU `Vram`
as a mirror the unit keeps current
(`clanker/wgpu/the-mirror-and-its-dirty-maps.md`); every golden scene and
the quake1 replay's checkpoints pass on it. Next: the front end's memory
paths as records (DMA puts and image puts from a staged upload buffer,
gets as staged readbacks, copies as records), then texture on the GPU.

## Goal

Replace the software raster units of IMPACT (RSS engines), VPro (the
quad pipeline) and, if it ever pays, GR2 (RE3) with compute kernels on
the host GPU through wgpu. Board memory lives on the GPU and nothing on
the CPU maps it: every CPU read of it is a readback. The front ends stay
as they are: command decode, state tracking, the HLEs and triangle setup
remain CPU work; the boundary is where a primitive reaches the raster
unit today. Bit exactness is the acceptance test, through the existing
replay hashes and goldens, so the kernels are the models' arithmetic
again, not a host-GL approximation (that path was tried for VPro and
retired; see `odyssey-quad-pipeline.md`).

Newport (REX3) is out of scope: its primitives are spans, and no batching
amortises a GPU round trip over a span. See the `rex-gpu` replay prototype
for what that looks like.

## Why compute, and why nothing is mapped

- The host rasteriser cannot be used for coverage plus a shader for the
  rest: two fragments of different primitives on one pixel have no
  ordering guarantee against a storage buffer. The fixes (fragment
  shader interlock, rasterizer ordered views, raster order groups) are
  not in wgpu and not on every driver. Depth has the same problem: the
  boards' depth words and interpolation are not the host's.
- Hardware samplers cannot read the boards' layouts (16x16 tiles, per
  engine pages, TRAM bank bit) or reproduce the 12-bit filter math. Texels
  are gathered by the kernel, as `sample_quad` does today.
- A compute rasteriser needs storage buffers, atomics and indirect
  dispatch and nothing else, which is wgpu's portable core. It runs
  unchanged on Metal and D3D12 through naga, and identically on lavapipe,
  so CI can run the goldens with no GPU.
- Memory stays unmapped because every path that touches it on the CPU
  side already waits on the raster unit first (`raster_wait`, `drain_rss`,
  `drain_re3`); those waits become "flush the batch, fence, read what
  you need back". No ownership protocol, no imported memory, no page
  tracking. The CPU rasteriser remains the other instantiation of the
  same board, chosen at machine build (const generic or trait, below),
  never a per-draw fallback.

## The common layer (`src/wgpu`, landed)

| Piece | What it is |
|---|---|
| `device.rs` `Gpu` | adapter, device, queue, adapter-sized storage limits, optional features (subgroups, int64, int64 atomic min/max, timestamps) recorded in `Caps`; async errors captured for `check()` |
| `shader.rs` `Source` | WGSL from `include_str!` fragments with `#include "name"` and `define`d constants; compile errors come back as text with the composed source numbered |
| `tiles.rs` | 8x8 tiles, `MAX_PRIMS` 4096 a batch, 128 mask words a tile, `TileGrid::tile_box`, the header and list layouts shared with `shaders/common.wgsl` |
| `batch.rs` `Batch<R, S>` | the recorder: `slot(S)`, `prim(R, pixel box)`; `flush` uploads records, slots, boxes and header, resets the busy count and the indirect arguments, dispatches bin then raster, returns the submission |
| `batch.rs` `BatchPipelines` | builds both kernels around a board's `ArchShaders`: its raster hook fragments, an optional `bin_accept`, its defines, its bind group layouts for groups 2 and up |
| `memory.rs` `GpuMemory`, `Staging` | a board memory: device-local storage buffer, `write`, `clear`, `read` (copy to a staging buffer, wait, map); `Staging` is the readback buffer any GPU buffer can be read through |
| `scanout.rs` `ScanoutPipeline`, `Scanout` | the display's compose: a board's `scanout_pixel(x, y)` kernel over its memory into a frame on the GPU, one lane a pixel, read back for the window; group 0 the frame, group 1 the board's display slot, groups 2 and up the board's memory |
| `shaders/bin.wgsl` | one workgroup per primitive, lanes over its tile rows: `atomicOr` the primitive's bit into each tile's mask, append first-touched tiles to the busy list, raise the indirect arguments |
| `shaders/raster.wgsl` | one workgroup per busy tile, dispatched indirectly, one pixel a lane: `arch_tile_begin`, then every set bit in mask order through `arch_prim`, then `arch_tile_end`; clears its mask words and busy flag on the way |

Rules the kernels rely on:

- **One writer per tile.** Only the tile's workgroup stores to its pixels
  during a batch, so primitive order per pixel is the loop order and no
  atomics touch memory. Two pixels in one word (overlay cells, 16-bit
  pixels) are handled inside the group.
- **Order is the bit position.** Setting bits commutes; the busy list's
  order does not matter; the records array is in submission order.
- **Hooks run in uniform control flow.** The tile index and each mask
  word are read with `workgroupUniformLoad`, so hooks may use barriers
  and subgroup quad operations as long as every lane takes the same
  branches and only stores are gated on coverage.
- **Bindings.** Group 0 is the batch; group 1 is records (binding 0) and
  slots (binding 1); groups 2 and up are the board's. The bin pass binds
  the indirect arguments at group 0 binding 5 through its own layout: a
  buffer cannot be read-write storage and the indirect source in one
  dispatch (wgpu validation, found in milestone 0).
- **Workgroup grids cap at 65535.** The busy count folds over x and y;
  the bin dispatch is one group per primitive, under `MAX_PRIMS`.

## What a board provides (`src/dev/<board>/wgpu`)

- A setup record type `R: Pod` and a slot type `S: Pod`, laid out as its
  WGSL declares them. A record names its slot; a slot is the stage state
  as register words, decoded by the board's WGSL functions, so the
  kernel adapts to the front end without the common layer knowing it.
  Each board has its own shader set; shared helpers (edge functions,
  fixed-point, Bayer tables) go in `src/wgpu/shaders` as library
  fragments when two boards want them.
- One raster shader per primitive kind is the aim (triangle, line, point,
  rect or span, pixel list). Until the bin pass can carry a per-record
  kind to a per-kind dispatch, `arch_prim` switches on a kind field in
  the record; the switch is uniform per workgroup so it costs a scalar
  branch.
- Its memory as one or more `GpuMemory` (VPro: one; IMPACT: pixel pages,
  CID planes, TRAM), bound from group 2 up.
- The batch boundary rule: close and fence when the front end would
  observe memory, at frame end, on `Full`, and on a time cap (Windows
  removes the device after a two-second dispatch).
- Readback helpers for every CPU-side reader the board has (listed per
  board below), each a flush, a wait and a `GpuMemory::read` of the
  bytes concerned.
- Scanout as a kernel into a frame buffer on the GPU, read back into the
  board's `Rex3Screen` (5 MB a frame at 1280x1024) until the GUI shares
  the device.

Selection: a const generic on the board (`Board<const GPU: bool>` or a
raster-unit trait with two implementations), instantiated when the
machine is built from `[graphics] raster = "cpu" | "wgpu"` and
`[wgpu] backend`. The CPU instantiation keeps its code; nothing is
decided per draw.

## VPro (Odyssey), first

The boundary is the raster thread's job: `GlDrawJob` becomes a
`PipeDraw` and `pipe::setup::run` walks it. The GPU instantiation turns
the same inputs into records and slots instead.

- **Records.** Filled, non-smoothed triangles already travel as setup
  records (`setup::Tri` with `raster::Triangle`: exact i64 edge values
  at pixel (0,0), per-pixel steps, owning-edge flags, area, 12.12 vertex
  colours, S/W..Q/W per vertex, z and divisor over w per vertex, inverse
  w, user-plane distances). The GPU record is that struct narrowed to
  what the kernel reads, i64 edges split or reduced (a 1/256 subpixel
  grid over 65536 pixels fits i32 edge *steps*; the start values need
  care). Clears (`ClearDraw`, `push_rect`) become rect records. Lines,
  points, smoothed primitives and polygon modes are emitted today as
  single shaded fragments through `Draw::fragment` (depth word, 12-bit
  colour, coverage); they become a *fragment list* record kind (a box and
  a range into a fragment buffer) first, and WGSL walkers later.
- **Slots.** `FragmentWords` (raster 0x404/0x415/0x458/0x81d/0x841/
  0x204b/0x840/0x84a/0xc48/0x3c4e/0x43f/0x420; shader 0x404/0x181a/0x41e;
  back end 0x404/0x420/0x452/0x41f/0xc21/0x422/0x1017/0x1023/0x424 and the
  decoded `DestViews`) plus the draw-state slot's texture registers
  (`TexRegs`: path, sampler, env, lod, level 0, 192 level words, scale and
  bias, offset, colour tables, detail) and its `Latch`. The 64 draw-state
  slots and the latch rule stay on the CPU; a GPU slot is written when a
  draw issues a slot the batch has not seen.
- **Memory.** `Vram` (power of two, 128 MB on a V12) as a `GpuMemory` of
  u32 words. The model's word is a u64 whose lowest address is the
  highest byte; the WGSL accessors reproduce `VramPtr::load/store(a, n)`
  bit for bit on the u32 pair, so hashes of the readback equal
  `address_bytes()` today.
- **Back end and shader.** `store_quad<Z, OP, C>`, `fragment_q`, the
  packers, `depth_word`, `sample_quad` (tiled texel addressing, 8x8x2
  windows, 8-bit filter weights, 1/256 lambda, 12.12 output, detail,
  post) port function for function. `BackendState::select`'s table
  becomes uniform branches on the slot.
- **Other memory readers and writers**, each with its treatment:

  | Today | GPU instantiation |
  |---|---|
  | `dest_mem()` paths (integer drawing, `image_put`, texture builds, zoom) after `raster_wait` | flush + fence, then GPU copy or scatter kernels from an upload ring; `image_put` is the hot one (Quake's texture bursts) |
  | pixel DMA put (`dma_fields` → `Rows::store_word`) | upload ring + scatter kernel |
  | pixel DMA get (`dma_read_pack`) | flush + fence + rect readback, then the existing pack |
  | `copy_block` | GPU copy kernel |
  | `ctx_switch`, `mmio_read` of waiting registers, `reset` | flush + fence; reset clears the buffer |
  | display: `Frame::compose` every 16.7 ms over all of VRAM | scanout kernel into a GPU frame; readback into `Rex3Screen` |
  | monitor `vpro pix/raw/peek/shot/fbdump/hash` | quiesce as now, then readback |
  | fences (`Board::fence`, `query_finish`, completions on the quad ring tail) | `Gpu::on_done` after the batch that precedes them: guest-RAM token write and softsync, as the closures do today |
  | replay: `raster_wait` then `memory_hash` | flush + wait, `read_all`, blake3 over the same byte order |
  | snapshots | refused for VPro today; unchanged |

- **Threads.** The raster thread and the pool go away in the GPU
  instantiation; the front end records and flushes. `RasterQueue`'s
  budget and `wait_room` become the batch's `Full` and a bound on
  in-flight batches.

### What landed (milestone 1a, `src/dev/odyssey/wgpu`)

- `Rec` (256 bytes): a triangle (the three edge functions as i64 at the
  centre of plane pixel (0, 0) plus their steps, the owning-edge bits,
  the inverse area, the z and divisor planes as `depth4` takes them,
  1/w at the vertices, the depth range and slack, the 12.12 vertex and
  provoking colours, the offset steps, the clipped plane box, the window
  origin, the alpha test's words), a CPU-shaded quad (`BackQuad`: plane
  position, coverage, depth words, 12-bit colours) or a clear's
  rectangle (its values and masks). `Slot` (368 bytes): the back-end
  words (0x404, 0x420, 0x452, 0x41f, 0xc21 x3, 0x422, 0x424) decoded in
  WGSL as `BackendState::of` decodes them, the draw facts the words lack
  (stencil planes, index mode, overlay context), the six destination
  views packed, the clip rectangles and the id-compare word, the polygon
  stipple and its background.
- Kernels: `vram.wgsl` (the u64 word rule on u32 pairs, atomics),
  `views.wgsl` (tiled addressing, RGBA8 / Z24 / byte fields),
  `backend.wgsl` (`fragment_q` generically: stencil, depth, blend with
  every factor and equation, logical operations, colour and index masks,
  both targets; `clear_pixel`), `vpro.wgsl` (the hooks: the walk's
  bounds, exact edges, `owned`, depth, stipple, flat or
  perspective-correct weights, the alpha test). Shared:
  `src/wgpu/shaders/i64.wgsl` and `float.wgsl`.
- `GpuRaster`: `push(rec, &slot)` with slot reuse and flush on `Full`,
  `flush`, `wait` (= `raster_wait`), `write_words`, `read_words`, `hash`
  in `address_bytes` order.
- Parity: random triangles of every size (edge values past 2^24 exercise
  the i64 path), flat and smooth, alpha tests, stipples with and without
  a background, forty random back-end states (depth functions and
  writes, blends over the whole factor table, logic ops, colour masks,
  stencil ops and masks, both targets, depth ranges, offsets), clears
  under id tests and masks interleaved with triangles and shaded quads,
  index draws into the 2-byte overlay surface. Whole memories compared.
- Every colour field the model has: RGBA8, Z24, bytes, the 12-bit
  lanes and the 2- and 4-byte texel codecs (`views.wgsl` ports
  field.rs). Not yet on the GPU: texture, fog, the separate specular,
  user clip planes (f64 on the CPU), lines, points and smoothed
  primitives (they arrive as shaded quads, which the GPU stores).

Measured on the way (`clanker/wgpu/gpu-floats-are-not-the-hosts.md`):
RADV's division is a reciprocal and a multiply, and it fuses multiply-adds;
both broke depth-word parity until `div_exact` and the multiply-by-one
guard went in. The quad pipeline's "fixed point for the rest" would remove
the need for both.

### Milestone 1b (landed): the unit in the board

- **Where it sits.** `RasterCtx.gpu: Option<Box<GpuRaster>>`
  (gpu_unit.rs aliases it; without the feature the type is uninhabited
  and the dispatch folds away). With a unit no raster thread starts, so
  every job runs inside its submit on the front end (`RasterQueue`'s
  unthreaded path, unchanged). `PipeDraw` carries the unit's pointer and
  the draw's `Slot`, built in `PipeDraw::new` from the back-end words,
  planes, index and overlay flags, the clip list and the stipple;
  `emit` becomes `push_frag`, `clipped_triangle` pushes a triangle record
  for a plain fill (no texture, fog, specular or user planes) and walks
  and shades the rest on the CPU into fragment records; `gl_raster_ops`
  pushes rectangles. A draw whose planes the kernels cannot address falls
  back to CPU stores on a mirror brought current first (none left since
  the coded fields landed).
- **Memory.** The board's `Vram` is the unit's mirror: CPU stores mark
  16 KB blocks (`DirtyBlocks` behind `VramPtr::store`), uploaded before
  each batch; records mark the blocks their boxes touch in their slot's
  views, read back at `raster_wait` into the mirror. So the front end's
  own paths (integer drawing, image puts, copies, DMA, context, the
  texture unit's samples, hashes and dumps) run unchanged and in order,
  at the price of a wait and a readback whenever they run with drawing
  pending. The board thread also brings the mirror current once a frame
  (`gpu_display_tick`) so the display follows the drawing until a
  scanout kernel feeds it.
- **Selection.** `[graphics] raster = "cpu" | "wgpu"` and
  `gpu_backend`, through `Odyssey::new` to `Board::enable_gpu`, which
  must run before the threads start. A runtime `Option`, not yet the
  const generic: the dispatch points are a handful and coarse.
- **Checks.** The parity tests (nine, coded fields included), the 76
  golden scenes with `VPRO_GPU=1`, and the quake1 replay with
  `VPRO_REPLAY_GPU=1`: all 165 checkpoints, bit for bit.
- **Measured** (quake1 replay, RX 6600 XT, release, 2026-11-06):
  925 M records in 226,779 batches (4,080 a batch: the batches fill),
  265 MiB uploaded, 20,149 MiB read back in 1,869 readbacks (10.8 MiB
  each, the frame's blocks), 515 s wall against the CPU pipeline's 152 s
  with two workers. The records are almost all fragment quads from
  CPU-shaded textured triangles (one record a quad: `MAX_PRIMS` is the
  bottleneck, not the GPU), and the readbacks are the image puts and
  DMA between frames. Those two numbers are what milestone 2 (texture on
  the GPU, fragment lists) and the record conversions below must bring
  down.

- **Where the time goes** (the first 10 M records, 21 checkpoints,
  release, 2026-11-06; `VPRO_REPLAY_LIMIT` with `VPRO_REPLAY_LAZY=1`
  skipping the readbacks between checkpoints):

  | | wall | readbacks |
  |---|---|---|
  | CPU, one thread | 46.1 s | |
  | CPU, two workers | 18.3 s | |
  | GPU unit | 59.7 s | 233 (2.7 GiB) |
  | GPU unit, no readbacks | 58.1 s | 20 |

  The readbacks are 3 % of it. The GPU path loses on the CPU side: 107 M
  fragment records (one 256-byte record per CPU-textured quad, 26,270
  full batches), built, uploaded and binned, on top of the CPU texturing
  itself. Texture on the GPU removes the stream and the shading; that
  is milestone 2 and comes before the DMA conversions. A compact
  fragment list (quads in a side buffer, one record per CPU walk tile)
  stays worth having for lines, points and smoothed polygons.

### Milestone 2a (landed): texture on the GPU

`shaders/texture.wgsl` ports the texture unit function for function:
the quad's positions from its four pixels' barycentrics (each lane
recomputed its quad's edge values at first; since 2026-11-08 the quad's
lanes exchange their positions, see
`clanker/odyssey/quake-gpu-unit-profile-2026-11-07.md`),
the diagonal level of detail through `fixed::log2_q8`'s table (generated
into the shader), the filter and levels lambda chooses, the taps wrapped
and gathered from board memory with every texel codec, the 8-bit filter
weights, the mip blend and the 12.12 conversion with their 64-bit
products split, the detail image, the scale and bias and the colour
tables, then the environment (`apply12x4`) and the separate specular. A
`TexSlot` (4.8 KB) is the draw's registers the unit reads at sample time
and the CPU's `Latch` (levels, LOD clamps and scales as the CPU computed
them, so no `exp2` runs on the GPU); slots are deduplicated by hash per
batch in their own buffer, and the record names its slot. The record
grew to 352 bytes for the texture values and specular colours.

Checks: `textured_triangles_sample_as_the_cpu_unit` (random 1D, 2D and
3D textures, every wrap, filter, class and environment, borders, detail,
LOD limits, clamps, scale and bias, colour tables, specular; whole
memories compared) and the 23 textured golden scenes, all exact. Still
on the CPU, stored by the unit: fog (f64 on the CPU) and user clip
planes (f64), lines, points and smoothed polygons.

Found on the way: the CPU unit's `filter` read the second row of a
one-row window for a 1D LINEAR texture and multiplied it by zero; the
read is now skipped (no value changes).

### Milestone 2b (landed): image puts as blit records

`paint_pixel`, which every image put and integer primitive ends in,
hands its pixel to the unit when it is active: `GpuRaster::blit_pixel`
collects consecutive pixels of a row into one blit record (`REC_BLIT`)
whose words sit in a staged buffer uploaded with the batch, under the
slot's view, a mask for the byte planes, the window's tests already done
on the CPU (which reads the id bytes without waiting: they are the X
server's, never the raster unit's). Any other thing the unit does first
closes the run, so the pixels keep their place in the order. A
depth-tested image's pixels go through the unit as fragments under a
slot latched with the box. A texture or scratch image through the FIFO
(`image_put` into a scratch destination) stores into the mirror untracked
and the unit uploads its rectangle when it closes
(`GpuRaster::note_region`): the records before it are submitted first and
the upload lands before the next batch, so nothing waits.

The mip-level builder (`texture_build_level`) stores into its level the
same way. Found on the way (`clanker/wgpu/read-storage-in-place.md`):
the texture slot copied by value into private memory gave wrong texels
once the kernel grew by one dispatch arm; records, slots and texture
slots are now read from their buffers in place, and the log2 table lives
in a buffer.

Measured after these (2026-11-06): the first 10 M records in 5.9 s with
47 readbacks (874 MiB: 20 checkpoints, 18 context switches, 8 copies,
1 clear), against 5.3 s with none; the whole demo, 165 checkpoints
exact, in 63 s before the scratch and builder changes (CPU: 152 s with
two workers). The remaining time is the front end itself; the next
readbacks to remove are the context switch and the copies, and the
display's once the scanout runs on the GPU.

### Milestone 3a (landed): the scan-out on the GPU

`src/wgpu/scanout.rs` is the common driver: one lane per screen pixel
calls the board's `scanout_pixel`, into a frame buffer on the GPU read
back through `Staging`. The VPro kernel (`odyssey/wgpu/shaders/
scanout.wgsl`) is `Frame::pixel` and `cursor_row`: the window-id byte,
the overlay, the window's display mode and its field decoded through
`load_kind`, the gamma tables or the lookup RAM, the cursor or crosshair.
Its `DisplaySlot` (38 KB a frame) is what `Frame::snapshot` resolves on
the CPU: surfaces, per-id decisions and fields, the overlay table, the
gamma tables, the cursor registers and the resolved LUT. The display
thread owns a `VproScanout` over a clone of the memory buffer
(`DisplayShared::gpu_scanout`) and composes with it in
`ScanoutReader::update`, falling back to the CPU compose on an error; it
no longer reads the mirror, so the 16 ms mirror readback is gone. Because
the kernel reads the GPU's memory, the front end now sends the mirror's
dirty blocks up at every flush, records or not, and once per run of
stores (`gpu_upload_tick`), so CPU-drawn pixels reach the screen. The
first live boot showed the other half of that: records and blit runs
stayed in the open batch until it filled or something waited, so the PROM
text and X painting reached the screen late or not at all. `GpuRaster::tick`
now submits the open batch at the end of a run of stores when no store
waits (the board going idle) or when the batch has been open 4 ms
(`MAX_OPEN`); a GL game fills batches faster than that and pays nothing.
`wait` also skips the device poll when nothing was submitted since the
last wait, since a poll with no index waits for the display's scanout too.

Swap ordering (the user's design, 2026-11-07): rather than fences the
display waits on, the model uses stream positions. `CommandFifo` counts
stores `pushed` and `retired` (absolute, monotone; the ring's indices
wrap). A swap arm or swap-control write records `pushed` at the time
(`HostInterface::swap_after`, set from `stream_pos` by the bus path), the
GPU unit's `tick` takes `retired` and publishes it to
`DisplayShared::covered` once everything recorded up to it is submitted
(at every flush, or at a tick with nothing recorded), and
`HostInterface::retrace(covered)` latches flips only when
`swap_after <= covered`. Submitted is enough, not completed: the display's
scanout submission queues behind the batch on the same queue. The CPU unit
leaves `covered` at the maximum (its own raster-thread lag is the
pre-existing behaviour). Test:
`a_flip_waits_until_the_raster_unit_covers_the_stores_before_its_arm`.
Still to do from the same idea: make the status words the guest polls for
idle (DMA busy, the sync words) report "busy until the fence completes"
instead of `raster_wait`'s flush and wait; that needs a completion counter
(`on_submitted_work_done` per submission) and the registers' busy bits.
`vpro gpu [trace on|off]` prints the unit's counters live, with readbacks by
cause when traced.
Checks: `scanout_composes_as_the_cpu_frame` (random modes, LUT, overlays,
cursors and crosshairs over random memory, whole frames compared) and the
22 display tests with `VPRO_GPU=1`.

The readback stays until the GUI runs on eframe's wgpu renderer and can
present the frame as a texture on the same device; that move, with the
Newport GL compositor ported to WGSL on this driver, is the next project
for the display.

### Milestone 3b (landed): the front end stores through the unit

The direction (2026-11-06): no more work on syncing the mirror; what
draws goes through the unit, and the mirror comes back only where a
reader needs bytes (a DMA get's rectangle, the hashes, the monitor,
`dest_mem` for the few general paths left).

- Blit records carry a logical operation with the result's and/or
  masks (`paint_integer`'s index or RGB fold), the plane mask for byte
  fields, and optionally their own view (a DMA put's surface). So
  `paint_integer`, `paint_aux` (the X server's window-id and overlay
  writes), `clear_planes`, `buffer_store`, screen `dma_put`s and the
  depth and stencil image writes all store through the unit, no CPU read.
- `Board::raster_wait` on the GPU unit is a flush and a GPU wait, no
  readback: the register reads that wait (`mmio_read`), the context
  switch (it touches no board memory) and the slot waits cost no memory
  traffic. `planes_in(box)` reads back only the blocks the drawing marked
  under a box, and nothing when none is marked; `dest_mem` (the general
  path) and `gpu_sync` bring everything back. Test helpers that read the
  mirror after a wait call `gpu_sync`.
- The region readback (`GpuRaster::read_back_view`) is also how a DMA
  get stages its rectangle before the existing pack writes it to the
  guest, and how a copy's rectangles come back before the CPU copy. The
  copy kernel (gather, then scatter with the codecs, flip, byte masks and
  the scale-and-bias filter, submitted between batches) is the next
  conversion; after it the mirror is written by nothing hot.

An aside from the same day: a replay run that shared the GPU with a
live IRIS session doing whole-screen readbacks hit an amdgpu ring timeout
and reported 41 GiB of uploads; neither reproduced on an idle GPU (10 M
records: 148 MiB uploaded, 4 MiB of them dirty blocks, 16 MiB regions).

### Copies as a kernel (landed 2026-11-08), then the walkers

Copies are the kernel now (`shaders/copy.wgsl`, `GpuRaster::copy`): the
batch is flushed, then a gather of the source rectangle into a 16 MiB
scratch buffer and a scatter into the destination run as two dispatches
on the queue, with the mirror rule, the window's clip list and id test,
the per-pixel front/back choice from the window-id plane, the
scale-and-bias filter and raw byte moves under the window-id masks. A
depth-tested copy (fragments), a copy over 2M pixels or a view the kernels
cannot address falls back to the readback path. The replay prints the
count (`N copies`); on quake1 the 16 copies leave only the checkpoints'
own readbacks. What the library sends for reads and copies is in
`clanker/odyssey/gl-reads-and-copies-as-traced.md`. The remaining
readbacks are the DMA gets (glReadPixels; a rectangle each, but one per
row today in the recordings: 670 for 27 reads in the pixtest copy run).


Texture and scratch uploads (`dma_to_buffer` into memory no drawing
touches) no longer wait: they store through an untracked pointer and
upload their own tiles (`GpuRaster::upload_region`; the rule is in
`clanker/wgpu/the-mirror-and-its-dirty-maps.md`).

The direction (set 2026-11-06): a DMA *get* stages its rectangle in
front-end memory (a readback of just that span) and writes it to the
guest through the bus and DMA abstraction as now; a DMA *put* or image
put becomes a blit record whose pixels sit in a staged upload buffer the
kernel fetches from, so it joins the batch in order and nobody waits;
copies become copy records. Each conversion removes one class of wait
from the replay's readback count, which is the measure. After that,
texture on the GPU (milestone 2) ends the last CPU shading, and the
scanout kernel (milestone 3) ends the display's readbacks.

Milestone 2 (the rest): fog and the user planes once the quad pipeline
has them in f32 or fixed point, a compact fragment list for lines and
points, scanout on the GPU; the quake1 timedemo's 165 hashes on the
Radeon and on lavapipe. Milestone 3: live guest (fences, DMA, monitor,
display readback).

## IMPACT (MGRAS), second

The boundary is the raster FIFO: register writes with an execute bit,
plus the out-of-band entries (PIO advance, DMA line with payload, GL
enter and leave, exit). The GPU instantiation is one engine that owns
every row (the layout still follows DRBsize, as a single real engine's
does): it keeps the 0x400-word register file and `execute()`'s decode on
the CPU and emits a record per primitive instead of drawing.

- **Records** per IR kind: block fill and character stipple (corners from
  0x046/0x047, colour by mode, stipple bits), X line and point, GL
  triangle (`TriSetup`: biased float positions, s.24 slopes, 12.12 colour
  planes, z.12, 2^32 texture iterators, fog), GL line (`GlLineSetup`),
  transfers in (a line of pixels from the upload ring with the
  `XFRMODE` format). `rss_jit`'s `PipeKey` and `RasterCtx` builders are
  the slot and record content already; the WGSL mirrors
  `compiler.rs`'s emitters as that mirrors the interpreter.
- **Memory.** `PixMem` (48 MiB of u64 words, `Buffer::locate` with
  per-engine regions, 192x16 pages, overlay cells of 9 bits), the CID
  planes (4 MiB) and TRAM (4 MiB, nibble cells, border page) as three
  `GpuMemory`s. 12:12:12 (`deep36`) is in scope from the start; it is
  the interpreter-only case today.
- **Other readers and writers:** display `Frame::snapshot` through
  engine 0 (scanout kernel + readback); PIO reads and `dma_read_line`
  after `drain_rss` (flush + readback); DMA lines in (upload ring:
  TE loads to TRAM, transfer puts to pages); host GL composite
  (`hostgl`, writes pages directly: a write); `state_bytes` for
  recordings (readback of the regions it hashes); monitor commands.
  Multi-engine TE synchronisation disappears with the second engine.
- **Test harness:** the desktop recordings in `ignore/mgras-golden/`
  and the gr4-jit sweep (two boards, random state, word-for-word
  pages, CID and stipple positions) rerun against the GPU board.

### What landed (2026-11-08, `src/dev/mgras/wgpu`)

The raster FIFO stays the boundary, and the CPU keeps everything up to
drawing: `Rss` (registers, `execute()`, the transfer and stipple state
machines, TE tables, `tri_setup` / `gl_line_setup`). An `Rss` with a unit
attached (`Rss::gpu`, a `GpuLink`) hands each primitive over as one
record at the point the interpreter would walk it: `fill`, `point`,
`line`, `stipple_bits` (one record a chunk), `put_line` (one record a
transfer line, pixels decoded on the CPU into the stage), `triangle`,
`gl_line`. The slot is the PP1/TE registers the pixel path reads plus the
sampler `Te1::sampler` resolves and the polygon stipple rows.

- **No scanlines.** Each lane tests its own pixel: the block rectangle;
  an X line's Bresenham step in closed form (`xline_step`, n =
  floor((2 minor k + major) / (2 major)), brute-forced against the
  interpreter's loop); the stipple or transfer row and column; a
  triangle's span bounds at the lane's row and a GL line's minor extent
  at its step. The model walks both in integers since 2026-11-08
  (interpreter, JIT and kernels alike; `clanker/mgras/gr4-jit.md`): an
  edge is (C, B) with the bound `-((B j + C) >> 25)`, a GL line a DDA
  `(qk + qs p) >> 32`, i64 products and shifts per lane (`wide.wgsl`,
  which also carries the texture unit's 64/128-bit arithmetic). The first
  version evaluated the f64 walk in exact software doubles
  (`src/wgpu/shaders/f64.wgsl`, kept as a tested library), about ten
  times the per-lane work.
- **Memory.** Pixel pages (binding 0, 48 MiB, atomics: four overlay cells
  share a word), clip-ID planes (binding 1, atomics), TRAM (binding 2,
  read only), the transfer stage (binding 3, per batch set), RECIP and
  log2 tables (binding 4). TRAM's only writer is the CPU's texture
  loader: `Te1::dirty` marks 16 KB pages; `before_tram_write` submits a
  batch holding a textured primitive first, and dirty pages go up before
  the next textured record's batch.
- **Board.** `[graphics] raster = "wgpu"` reaches IMPACT through
  `Mgras::with_raster` (Indigo2 `machine.rs`, Octane `ImpactXio::
  with_raster`): one engine thread, the unit owned by the board and
  linked into engine 0. The planes are the mirror: the RSS thread syncs
  them (flush, wait, read back the pages and clip-ID rows records marked)
  before it reports idle, and every 16 ms while busy, so every reader
  that waits for idle (DMA and PIO reads, monitor, hashes, swaps) and the
  display see current memory. A host GL composite uploads the rows it
  wrote. TRAM reads (glGetTexImage, page saves) read the CPU's TRAM,
  which is authoritative.
- **Checks.** `cargo test --release --features wgpu --lib mgras::wgpu`
  (`--test-threads 2..4`): two engines from the JIT sweep's generators
  (`src/dev/mgras/test_gen.rs`, shared with `rss_jit_tests`), one
  interpreting, one with the unit, compared word for word every step over
  all pixel memory and clip IDs plus the stipple positions: fills, X
  lines, character stipple, transfers (put, fragment, depth, stencil),
  triangles flat/smooth/textured/fogged/stippled, steep slivers, GL
  lines, TRAM loads between textured draws (and a targeted test that a
  load lands between the triangles around it). Twelve deliberate kernel
  mutations and the TRAM-ordering one were each caught.
- **Rule found:** pixels that alias one memory word race across tiles
  (`clanker/wgpu/aliased-pixels-race-across-tiles.md`); the GPU tests use
  unaliased layouts, as IRIX's are.

- **Goldens and a live guest (2026-11-08).** `MGRAS_REPLAY_GPU=1`
  replays a recording on the unit: `desktop2.rec` passes 35/35
  checkpoints, and `dense`, `desktop`, `flat`, `threaded` and
  `irix6522-boot` end in the CPU engine's final state (those predate
  later fixes and fail checkpoints on the CPU too). The goldens hold no
  GL triangles; a live recording of tex_cube and atlantis (about 200k GL
  triangles, Maximum Impact) replays to the same final state on both.
  The IP30 guest (`ignore/ip30/iris.toml` with `raster = "wgpu"`, a COW
  overlay, `lightning,wgpu`) boots on the unit through the PROM's
  power-on screen, xdm and the 4Dwm desktop, and runs both demos.

- **Scan-out on the GPU (2026-11-08).** `wgpu/scanout.rs` and
  `shaders/scanout.wgsl` are `Frame::snapshot` and `Frame::compose` over
  the unit's pages: the slot is the frame's decoded state (XMAP modes,
  colormap 0, gamma, the cursor glyph and colours, the VC3 window-ID runs
  as a run list per row). With `renderer = "wgpu"` the window shares the
  unit's device (`GfxDisplay::gpu`) and frames are presented from there;
  a frame is read back only when someone reads `screen.rgba` (monitor
  `screen`, VNC capture, the screenshot hotkey, the GL renderer). The RSS
  thread no longer syncs the mirror: it submits on idle and every 4 ms
  while busy. Readers sync what they read: a guest's PIO or DMA read the
  dirty pages under its block in its source buffer
  (`Rss::gpu_sync_transfer`, `GpuRaster::sync_box`), `state_bytes` and the
  monitor's pixel commands everything (`Mgras::mirror_now`). Checks:
  `scanout_composes_as_the_cpu_frame` (random state, whole frames; nine
  kernel mutations caught), every golden replay comparing the scan-out
  with the CPU compose at each checkpoint (396 frames, both boards), and
  `transfer_reads_see_the_drawing`. Live on the IP30 guest: about 6,000
  frames presented with 0 bytes read back through the PROM.

Open: per-row edge sharing (eight lanes of a tile row compute the same
bounds); the host GL composite still uploads rows the CPU wrote.

## GR2 (XZ / Extreme), open

The boundary is `Re3Sink` into `re3_fifo` (one RE3; the GE7 only stores
microcode, all geometry is HLE in `hq2.rs`/`gl.rs`): register writes
with IR execute (SHADED, FLAT, FLAT4 spans with iterators in 14-bit
fractions), the private ops (copy, Z fill, pixel format, Z control,
stencil, blend, alpha) and reads after `drain_re3`. Memory is two
1280x1024 u32 planes. Spans are short and a frame has few of them, so
the batch amortisation argument is weak; decide after IMPACT with
numbers from the span counts of a GL scene. The REX3 lesson applies
until then.

## Measurements to take before each milestone closes

- Batch count and primitives per batch on the quake1 replay (how often
  the front end observes memory).
- Bin and raster pass time (timestamp queries) against the CPU pipeline's
  tile time (`quake-replay-is-fragment-bound.md`: 151.6 s wall at 2
  workers).
- Readback bytes a frame (display, DMA gets) and their wait time.
- Hash parity on radv and on lavapipe.

## Open questions

- **Float parity.** Settled for add, multiply, divide and the int
  conversions (`clanker/wgpu/gpu-floats-are-not-the-hosts.md`); the
  texture unit's `sqrt` and `log2` paths, when they arrive, need the same
  treatment (an exact `sqrt` the same way as `div_exact`; `log2` is a
  table on the CPU already). The quad pipeline's "fixed point for the
  rest" would make the guards unnecessary.
- **Atomics for every memory access.** Correct and simple; the cost is
  unmeasured (`clanker/wgpu/vram-accesses-are-atomic.md`).
- **Fragment lists.** One record per emitted fragment would exhaust
  `MAX_PRIMS` on a text-heavy frame; the fragment buffer with range
  records is the first design, the WGSL walkers the second.
- **Per-kind dispatch.** Whether a per-record kind in the bin pass and
  one raster pipeline per kind beats a uniform switch inside one kernel.
- **GUI.** iris-gui is on glow. Readback first; eframe's wgpu backend and a
  shared device when the raster path has earned it.
- **Trusted shaders.** wgpu bounds-checks storage accesses; turn it off
  per kernel (`create_shader_module_trusted`) once the goldens pass.
