# Quake timedemo on the GPU unit: user planes, then the raster kernel

Status: measured 2026-11-07 and 2026-11-08 (RX 6600 XT, RADV; 2990WX), release build,
`ignore/recordings/quake1-timedemo.rec`. The recording is SGI's
sgiquake1 (`/usr/sgiquake1/quake.gl`), not TyrQuake: 60% of its
triangles sample with a detail texture.

## User planes sent every triangle to the CPU

- sgiquake1 keeps user clip plane 0 enabled with every vertex about +4080
  inside it. Draws with any enabled plane were shaded per pixel on the CPU
  and stored as fragment records: 16.65 M records, 5661 batches, 26 s
  drawing, about 22% of host time in CPU texture sampling.
- Fix: `Transform::user_side()`. All vertices inside every enabled plane:
  no per-fragment test (GPU kernel as usual); all outside one plane: the
  triangle is dropped; otherwise per fragment. Exact: the interpolated
  plane value is a weighted mean of the vertices' with weights >= 0.
- After: 1.10 M records, 1970 batches, 18 s drawing, memory hash
  unchanged (`65a51d3cd35bf925`).

## Then the raster kernel is the bottleneck

- `WGPU_BATCH_TIMING=1`: raster 12.9 s GPU, bin 0.8 s; 559 prims and
  ~9,900 busy tiles a batch; 554 batches over 10 ms.
- amdgpu_top GRBM "Graphics Pipe" busy ~41% over the whole replay,
  which includes the harness's hashing; it says work is resident, not how
  many waves run.
- `RADV_DEBUG=shaderstats`: the VPro raster kernel has 168 VGPRs,
  6 subgroups/SIMD, 52 spilled SGPRs, 36.7 k instructions. The bin kernels
  have 16 VGPRs.
- `RADV_PERFTEST=cswave32` was slower (14.0 s raster). Do not tune per
  vendor; use RADV numbers only to measure.
- All 164 readbacks in the run are the harness's checkpoints
  (`VPRO_GPU_TRACE_WAITS`).

## Then the registers (2026-11-08)

Raster kernel 168 -> 96 VGPRs (6 -> 10 waves/SIMD), raster 12.9 -> 8.2 s
GPU, memory hash unchanged. Workgroup-memory fallback
(`WGPU_NO_SUBGROUPS`): 128 VGPRs, 9.4 s. Code 193 KB -> 83 KB.

What each step was worth, as pre-scheduling demand (the real figure; the
allocated count snaps to wave steps: 128 = 8 waves, 96 = 10):

| step | demand |
|---|---|
| start | 164 |
| one `shade_pixel` and one `fragment_q` call site, quad exchange in `tri_in` | 113 |
| texel sampled before the colours are interpolated | 106 |
| blend a component at a time from the packed destination | 106 (SGPR spills 34 -> 21) |
| texture taps as one loop with a variable trip count | 95 |

- Each record kind alone was at most 116; together 164. Inlined copies of
  the shading and the back end stack in one kernel.
- The quad exchange alone did not lower the count (116 -> 108 on the
  triangle path only): the peak was the unrolled 2x2x2 tap loop, which
  kept every tap live. A loop over `nx * ny * nz` cannot be unrolled.
- Tools: `RADV_DEBUG=shaderstats,nocache` (without `nocache` an unchanged
  kernel prints nothing; pick the largest compute shader), `Pre-Sched
  VGPRs`; `RADV_DEBUG=shaders,nocache ACO_DEBUG=liveinfo` prints demand
  per instruction. Edit the dumped kernel (`IRIS_WGPU_DUMP`) and compile it
  with `IRIS_WGPU_LOAD`: no rebuild per experiment.

## A texel cache in LDS: not worth it (measured 2026-11-08)

Stubs in the dumped kernel, whole replay, raster GPU time (full 7.5 s):
texel load and unpack 0.82 s (11%), the whole tap with wrap and address
1.44 s (19%), all sampling with LOD and detail 1.94 s (26%). A 16x16 LDS
window per tile could at best take part of the 19%, against a fill, two
barriers a primitive a tile, slivers that fill more than they read, and a
fallback for footprints that do not fit. The GPU's caches already absorb
the repeated reads. Not pursued; 74% of the time is outside texturing.

## A tile cache in LDS for the destination: slower (measured 2026-11-09)

`VPRO_TILE_CACHE=1` (`shaders/tile_cache.wgsl`, off by default): each 8x8
tile's block of every surface its primitives draw to is held in workgroup
memory for the batch (a tile is a quarter of a 16x16 memory tile, so a
block is 8 contiguous rows of 8 << log2 bytes); fragments read and write
it; the blocks a tile changed are stored back once at tile end. Lane 0
plans the directory per primitive, the workgroup fetches (a coalesced
load or four a lane) and evicts between barriers. Exact: the replay's
memory hash is unchanged (`65a51d3cd35bf925`), the parity tests pass.

Raster GPU time, back to back:

| kernel | VGPRs | raster |
|---|---|---|
| atomics on board memory (default) | 96 | 5.89 s |
| fragments in LDS, no fetch or write-back (the ceiling, wrong output) | 96 | 5.66 s |
| first cache: lookup loop, origin and lane held per lane | 128 | 7.88 s |
| the cache with the register fixes below | 96 | 6.92 s |

- The ceiling is 4%: the atomics' latency is hidden by the other waves;
  the destination's accesses are not what the kernel waits on.
- What is left, +17%, is the block traffic and a barrier a primitive. A
  busy tile sees 10.1 primitives a batch on average (measured with the
  list binning below, which counts them), so the traffic is not
  amortised over one primitive; the ceiling, not the traffic, is what
  rules the cache out on this workload.

Texel reads through a plain binding instead of `atomicLoad` (`glc dlc`,
L2): 5.90 -> 5.85 s raster, A/B in one binary, memory hash unchanged.
Neither the destination's nor the texels' memory round trips are what
this kernel waits on; an RGP thread trace (instruction timing) is the
next step, not more static analysis.

**Where the 32 registers went** (Pre-Sched demand; the allocation snaps
96 -> 128, i.e. 10 -> 8 waves a SIMD, as soon as demand passes ~95):
- 24: the directory lookup at every field access, a loop to `tc_count`
  read from workgroup memory, inlined at ~20 sites. Unrolled compares
  against the four entries (unused ones hold an impossible base): 122 -> 98.
- 2: the tile origin as `px & ~7`, a per-lane value the next primitive's
  `tc_prepare` and `tc_end` need, so held across all the fragment code.
  From the tile index (`batch` fields) it stays in SGPRs: 98 -> 96.
- 1: the lane index (`v_mbcnt`), held for the same reason; recomputed
  from (px, py), which the hooks take anyway: 96 -> 95, allocated 96.
- General rule for this kernel: anything per lane that a later hook needs
  costs a VGPR across the whole shading path; derive it from values that
  are live anyway, or from uniform ones.
- Method: `RADV_DEBUG=shaders,nocache ACO_DEBUG=liveinfo`, then compare
  the histogram of per-instruction demand of two kernels (a constant
  shift means values held across the peak, not a new peak) and look up
  the extra live-in temp's definition. Kernels edited through
  `IRIS_WGPU_DUMP`/`IRIS_WGPU_LOAD` must be dumped from the same
  workload: the dump bakes in `VRAM_WORD_MASK`, and a parity test's 4 MiB
  mask in the 128 MiB replay gave wrong output and wrong timings.

## Binning per coarse tile into lists (2026-11-09)

The bin pass went from one workgroup per primitive, global `atomicOr`s
into a 132-word mask per 8x8 tile (35 MB), to one workgroup per 32x32
coarse tile with its masks in workgroup memory, writing each busy tile's
primitives as an ordered run of a list (`src/wgpu/shaders/bin.wgsl`).
A/B, same replay, memory hash unchanged:

| | bin | raster | busy tiles/batch |
|---|---|---|---|
| masks | 1.07 s | 5.82 s | 9,859 |
| lists | 0.20 s | 6.11 s | 9,872 |

10.1 primitives per busy tile. The raster pass is ~4% slower with lists
and it is not the walk: prefetching the next index (6.09 against 6.12 s)
and running the busy tiles in screen order (6.05/6.10 s) changed nothing.
Unexplained; the RGP traces in `ignore/rgp/2026-11-09-quake-texel-plain/`
are from the mask build, a list-build capture would show the difference.

## Binning by edges, not boxes (2026-11-09)

With lists, `WGPU_BATCH_TIMING` showed 10.1 primitives per busy tile:
binning by box sends a triangle to every tile of its box, and each costs
a full setup in all 64 lanes. The board's `bin_accept(p, tx0, ty0, tx1,
ty1)` now rejects tile rectangles the primitive cannot draw in, asked
once per coarse tile (32x32) and once per tile.

- VPro (`vpro_bin.wgsl`): each edge function at the rectangle's corner
  its steps point to (its largest value there), against the edge's
  owning test. Conservative: three edges may pass at three corners.
- IMPACT (`src/dev/mgras/wgpu/shaders/rss_bin.wgsl`): the span bounds
  are monotonic in the row, so per run of rows on one minor edge, the
  leftmost left bound and rightmost right bound at the run's ends.

A/B with `VPRO_BIN_EDGES`/`MGRAS_BIN_EDGES=0`, back to back, output
unchanged (VPro memory hash `65a51d3cd35bf925`; IMPACT final state
`35e972ff3b58fadf`):

| | bin | raster | busy tiles/batch | prims/busy tile |
|---|---|---|---|---|
| VPro quake1, boxes | 0.20 s | 6.09 s | 9,872 | 10.1 |
| VPro, edges per tile only | 0.48 s | 5.06 s | 9,547 | 4.9 |
| VPro, edges per coarse tile and tile | 0.37 s | 5.11 s | 9,547 | 4.9 |
| IMPACT sgiquake, boxes | 0.14 s | 3.72 s | 3,600 | 3.6 |
| IMPACT, edges per coarse tile and tile | 0.31 s | 3.08 s | 1,400 | 1.3 |

The coarse test is what keeps the bin pass cheap: without it every
candidate is tested at each of the 16 tiles. IMPACT's busy tiles fall by
more than half because its boxes are padded by a pixel on each side
(`tri_box`). The sgiquake replay's wall time (11-15 s) is the CPU side
and does not show this.

## At a pinned clock (2026-11-09)

Every GPU time above was taken with the card's clocks floating: the
replay keeps the GPU about a third busy, so amdgpu ran it near 1.2 of
2.6 GHz (memory 480 of 1000 MHz). `WGPU_BATCH_TIMING` now prints the
clocks; pin them with `echo profile_peak >
/sys/class/drm/card0/device/power_dpm_force_performance_level` (root;
`auto` undoes it). Pinned (2590 MHz, 995 MHz), two runs each, which then
agree within 0.5%:

| | bin | raster |
|---|---|---|
| VPro quake1, masks (a62360f0) | 0.314 s | 3.323 s |
| VPro, lists, boxes | 0.071 s | 3.185 s |
| VPro, lists, edges | 0.158 s | 2.535 s |
| VPro, lists, edges, tile cache | 0.158 s | 3.60 s |
| IMPACT sgiquake, boxes | 0.062 s | 1.41 s |
| IMPACT, edges | 0.125 s | 1.09 s |

- The lists' "raster +4%" was clock noise: pinned, the list walk is 4%
  faster than the mask walk.
- Edges: bin + raster -17% on both boards; against masks -26% on VPro.
- The tile cache is +42% raster, worse than unpinned: with memory at
  full clock the atomics it saves cost even less.
- Per frame: VPro 2.8 ms (970 frames), IMPACT 6.0 ms (203 frames).
- Live, sgiquake `timedemo demo1` on the IP28 guest (High IMPACT,
  1024x768, jitv2): 27.2 fps on the GPU unit, 26.3 pinned; guest CPU
  thread 100%, HQ3 and RSS threads 40-50%, GPU 55% busy. The guest CPU
  is the limit. On the CPU rasteriser: 1.9 fps, RSS thread 97%.

## Batches: why they end, and the mask walk (2026-11-08)

- Not by size: 559 prims a batch against MAX_PRIMS 4096. With
  `VPRO_GPU_TRACE_WAITS` the replay lists flushes by caller: 1603 of 1970
  from `note_region` (a texture image or mip level going up untracked:
  the open batch is flushed first, whether or not it touches that
  memory), 232 waits (mostly the harness), 81 capacity. Sweeping
  MAX_PRIMS would change little; flushing only when the batch touches the
  upload's memory would.
- The raster pass walked all MASK_WORDS (128) words a tile, an atomic
  exchange and a barrier each, whatever the batch held. Walking the
  batch's words: raster 6.43 -> 4.95 s. Then a per-tile summary bit per
  word (bin pass sets it when a word gets its first bit): raster
  6.53 -> 5.97 s, bin 0.81 -> 1.03 s, net -5%. Setting the summary bit
  unconditionally (non-returning atomics) made no difference.
- Absolute GPU times move between runs (8.2, 7.5, 6.4 s for one kernel
  the same day): compare back to back only.

## Next

The back end and the line's made-up quad are the largest remaining parts
(about 19 and 9 registers). Bin pass 0.6 s; a batch averages 4.2 ms.
