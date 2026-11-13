# The GPU unit's board memory has a CPU mirror, and two dirty maps keep the two in step

Status: 2026-11-06, design, `src/dev/odyssey/wgpu/mod.rs` (`GpuRaster`),
`src/dev/odyssey/vram.rs` (`DirtyBlocks`), `Board::raster_wait`.

The plan says board memory lives on the GPU and nothing on the CPU maps
it. The front end still has code that reads and writes pixels itself
(integer drawing, image puts, copies, pixel DMA, context save, the
texture unit's samples, the hashes and dumps). Rather than rewrite each
of those for the first milestone, the board keeps its `Vram` as a
*mirror* of the GPU buffer, and the raster unit keeps the two in step:

- **CPU to GPU.** `Vram::track_dirty` gives the memory a map of 16 KB
  blocks; `VramPtr::store` marks the block it writes (one relaxed
  `fetch_or`, skipped when no map is kept, which is the CPU unit's case).
  `GpuRaster::flush` takes the map and uploads the marked runs with
  `queue.write_buffer` before it submits the batch, so the batch sees
  them. The display's scanout kernel reads the GPU buffer, so the blocks
  also go up at the end of every run of stores (`GpuRaster::tick`), and
  the open batch is submitted there too once the board is going idle or
  the batch has been open 4 ms: a batch that waited to fill held PROM
  text and X painting off the screen in the first live boot.
- **GPU to CPU.** `GpuRaster::push` marks, in its own map, the blocks
  each record's box may touch in each view of its slot (the tile-major
  layout keeps a box's tiles in one contiguous span of addresses).
  `read_back` (what `Board::raster_wait` becomes) flushes, waits, reads
  the marked runs back in one copy and writes them into the mirror
  without marking the CPU map (`write_words_untracked`).

The invariant everything rests on is the one the CPU raster thread
already imposed: **the front end waits for the raster unit before it
touches memory itself.** A CPU store with records still in the batch
would be uploaded ahead of them at the next flush, reordering it before
drawing that was issued earlier. The one known hole is the same as the
CPU pipeline's: a texture the GPU drew (a pbuffer used as a texture)
is sampled from the mirror, which is current only as of the last wait.

**A writer that skips the wait must not use the block map.** A block is
16 KB; if the mirror is stale in part of it (the GPU drew there and no
readback has run), uploading the whole block clobbers the GPU's bytes
with the mirror's. So a path that writes without waiting, which is only
allowed into memory no pending drawing touches (texture and scratch
uploads, `dma_to_buffer` with `memory`), stores through
`VramPtr::untracked()` and then calls `GpuRaster::upload_region` for the
rectangle it wrote: the upload is those tiles alone. The same tile spans
are what a blit record will reference once puts into the screen carry
their pixels in the batch.

What this buys: every CPU path is correct unchanged, and a draw the
kernels cannot take (a coded colour field, texture, fog, user planes,
lines) still goes through the unit in order, either as CPU-shaded quads
the unit stores (`emit` becomes `push_frag`) or, for an unsupported
destination, as CPU stores onto a mirror brought current first.

Since 2026-11-06 the front end's own pixel paths store *through the
unit* as blit records (logical operations, masks and own views included),
`raster_wait` is a GPU wait with no readback, and only readers bring
blocks back, by the box they read (`planes_in`, `read_back_view`): a DMA
get, a copy's rectangles, the hashes and the monitor. The invariant above
still governs the paths that store into the mirror.

What it costs: a readback of the touched blocks under a reader's box, and
the upload of every CPU write. The replay prints both
(`gpu unit: ... MiB uploaded, ... MiB read back in N readbacks`). The
way out, as the plan's VPro table says, is to turn the CPU paths that run
between draws into records: pixel DMA puts and image puts as blit records
reading from a staged upload buffer, copies as copy records, and DMA gets
as readbacks of their own rectangle staged in the front end and written
to the guest through the bus. Each one removes a wait.
