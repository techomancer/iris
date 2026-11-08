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

## Next

The back end and the line's made-up quad are the largest remaining parts
(about 19 and 9 registers). Bin pass 0.6 s; a batch averages 4.2 ms.
