# The Quake replay is fragment-bound; fences and die placement cost the rest

Status: measured 2026-11-05 at 51cd68ac; see "Since" for the changes after (2990WX, release build, whole
`ignore/recordings/quake1-timedemo.rec`, `VPRO_REPLAY_THREADS=1`).

## Where the time goes (2 workers, tile 32)

About 200 s wall, 2.7 cores busy. The run is roughly 970 frames:

- 3.08 G textured fragments in 792 M quads (3.9 per quad);
- 1.27 G clear pixels (one 1280x1024 clear a frame);
- 1.49 M draws (about 1,500 a frame);
- 74,113 fences (about 76 a frame).

| Thread | Busy | Waiting |
|---|---|---|
| front end | ~27 s of its own work | 111 s for room (budget), 59 s in `raster_wait` (2,369 calls) |
| raster thread | 5 s of triangle set-up for all 1.49 M draws; 168 s running tiles while draining | 21 s asleep |
| each pool worker | ~172 s on tiles | ~24 s asleep (209 k sleeps in all) |

- The board is the bottleneck: the front end waits most of the run.
- Tile work in all is 513 s, about 167 ns (≈550 cycles) per textured fragment.
- The split of worker time, from perf:
  - texture sampling 48% (`sample_quad`, `level_quad`, `gather<2,6,4>`);
  - walk and depth 19% (`run_tile` self; the hottest instruction is the
    scalar `vdivss` in `Triangle::depth_ratio`, once per pixel);
  - shading 19% (`shade_quad`, `shade12`);
  - back-end stores and blend 10%;
  - clears 3%.
- Every fence (`Board::fence`, a `submit_completion` job) makes the raster
  thread `drain_back()` the whole tile ring before it writes the token. That
  is about 76 barriers a frame, and it is where the workers sleep: two to
  three sleeps per fence.
- Removing the barriers is worth at most about 12%. The per-fragment cost is
  the larger lever.

## Since: fences on the ring, quads as lanes (same day)

- **Fences.** Fences now ride the quad ring (`QuadRing::complete`) instead
  of draining it.
  - Drains fell from 121 k to 43 k, and worker sleeps halved in count.
  - Wall time did not move (189 s against 192-197 s): the barriers' tails
    were short.
- **What the front end still blocks on.** Over the first 25 M records, 697
  `raster_wait`s from `image_put` on a scratch destination cost 19.6 s of a
  52.8 s run. These are Quake's texture and lightmap uploads by pixel burst,
  which drain the pipe before their first pixel.
  - Next: hand the upload to the raster thread as an ordered job, so the
    front end keeps parsing behind it.
- **Lanes.** The quad's data is now `[component][pixel]`:
  - the walk's edges, coverage and depth;
  - `TexQuad`, the sampler's positions and taps, and the texel it returns;
  - the shader's colours, weights and alpha mask.

  The back end's per-pixel rgba stays AoS: its loads and stores are per
  pixel, and its blend is already 4-wide across components.
- **Auto-vectorization.**
  - LLVM vectorized the lane products and integer interpolation.
  - It left every saturating `as i32` scalar, and the walk's depth too.
  - Those three kernels run on fearless_simd (`pipe/lanes.rs`; the depth
    encoder is in `depth_word.rs`), with tests against the scalar formulas
    at NaN, the infinities and the i32 edges.
- **Results.** Full replay at 2 workers: 151.6 s wall, 413 s CPU (from about
  193 s and 540 s), all 165 checkpoints. The first 25 M records went from
  50.5 to 41.2 s.
- **Portable build.** A portable build (`-C target-cpu=x86-64`, the level
  detected at run time) gains more: 62.6 s to 44.1 s over the first 25 M
  records, with the same checkpoints. That baseline target vectorized
  almost none of the old code.
- **Order now:** `gather<2,6,4>` 5.4%, `run_tile` 5.2%, `level_quad` 4.9%,
  `shade_quad` 4.5%, `sample_quad` 4.1% of samples.

## Scaling

| Workers | Wall | Tile CPU |
|---|---|---|
| 1 | 284 s | 502 s |
| 2 | 197-203 s | 513 s |
| 4 | 203 s | 837 s |
| 4, pinned to NUMA node 0 | 151 s | 596 s |
| 8 | 143 s | 941 s |

- The raster thread runs tiles while it drains, so N workers means N+1
  shading threads.
- On the 2990WX, unpinned workers spread across dies, and the same pixels
  cost 60% more CPU. Pinned to one node, 4 workers scale as expected.
- `[perf] thread_affinity` pins only the CPU and REX3 threads, not the VPro
  ones.

## Test-side cost

Fixed in the same session: `memory_hash_bytes` read the 128 MB board one
byte at a time, about 0.25 s per checkpoint over 167 checkpoints. It now reads
whole words and produces the same hashes.

## How it was measured

- perf at user level only (`perf_event_paranoid` 2: no scheduler or
  context-switch events).
- Wait timers at every wait site: the ring's slot, tile-queue and drain
  waits; pool sleeps; `wait_idle` and `wait_room`; the raster thread's sleep
  and job kinds. Plus fragment and fence counters.
- The timers are temporary, not committed.
