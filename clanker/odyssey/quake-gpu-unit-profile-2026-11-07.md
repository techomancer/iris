# Quake timedemo on the GPU unit: user planes, then the raster kernel

Status: measured 2026-11-07 (RX 6600 XT, RADV; 2990WX), release build,
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

## Next

Quads aligned to the plane on both units, and the GPU's four lanes of a
quad sharing their values through quad operations instead of each lane
recomputing its three neighbours (`clanker/plans/wgpu-raster.md`).
