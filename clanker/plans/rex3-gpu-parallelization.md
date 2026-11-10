# REX3 GPU parallelization: tile-binned shader, tile-major plane

Status: implemented 2026-11-10 (candidate 3 below) and carried into the
experimental live-renderer PR. On Vulkan the final tree passed both gates in
three of three runs (raster ~2.5x, tiny ~0.96x); D3D12 passed raster and
missed tiny at 0.947x. The tiny margin is a few percent; see "Where the tiny
gate stands". The step measurements below were taken on the hardened tree (process
lease, deferred stop). The PR's lean tree, same shader, bins and plane
layout, measured 2026-11-10 on Vulkan, three runs: raster 3.323 / 3.305 /
3.272x (CPU ~1,315 Mpx/s, GPU ~4,340 Mpx/s) and tiny 1.532 / 1.552 / 1.541x
(CPU ~67.5 Mpx/s, GPU ~104 Mpx/s). The likely difference is the hardened
tree's framebuffer-access gate, taken for every GFIFO register in `rex-gpu`
builds, which costs the GPU path more than the CPU path. Not yet isolated;
the tiny-gate analysis below applies to the hardened tree.

## What was built

- **CPU binning** (`TileBins` in `rex3_gpu.rs`). The screen is 42 × 32 tiles
  of 32 × 32 pixels. Each accepted fill appends its batch index to every tile
  it covers. At submit the bins are encoded as one `u32` array: 1,345 offsets
  into the array itself, then each tile's indices in submission order. A batch
  is submitted early once it would hold more than 2^20 tile references.
- **Shader** (`rex3_gpu.wgsl`). One workgroup per tile, dispatched over all
  1,344 tiles; empty tiles return at once. Each of the 256 invocations owns
  four pixels, loads them once, applies the tile's fills in order in
  registers, and stores them once. No two invocations share a word, so there
  are no barriers, and draw order is preserved per pixel.
- **Tile-major GPU plane.** The GPU copy holds only the 1,344 × 1,024 screen
  (5.25 MiB, was 8 MiB), each tile's 32 rows contiguous. Upload swizzles the
  CPU plane once per reseed. A checkpoint copies back only the tile rows
  damaged since the last checkpoint — one buffer copy per run of damaged
  tiles — and the CPU scatters only those tiles. Every other tile still equals
  the CPU plane: only accepted fills change the GPU plane, and any CPU
  mutation first revokes GPU authority. `bytes_read_back` now reports the
  bytes actually transferred.
- **One bind group for the backend's lifetime**: all three bindings are whole
  fixed buffers; only the bin offsets select fills.
- **Default batch 1,024 → 16,384 draws.** A submission costs ~50 µs of host
  time on the processor thread. Journal caps are unchanged (65,536 draws,
  4,194,304 pixels). The strict GPU corpus pins its own 1,024-draw batch
  because `batch_1025_fence` was authored around that boundary.
- **Per-draw host cuts** (hardened tree). One failure-lock acquisition per
  accepted draw instead of two, no `Arc` clone; a pending callback error is
  still checked on every non-accepting result. The lean PR tree checks the
  failure slot once per attempt.
  `execute_go` computes CPU damage only after the GPU refuses a draw.
- **Map wait** yield-spins for its first 2 ms before sleeping in 100 µs steps.

Tests: `tile_bins_preserve_order_and_cover_exactly_each_fill` runs the WGSL
algorithm statement for statement on the CPU against `FlatFill::replay`;
`damaged_tile_runs_cover_damage_in_tile_major_order` covers the readback
runs; the hardware test `gpu_tiled_fills_match_cpu_replay_at_every_checkpoint`
compares 8,000 random fills (edges, overlaps, masks, empty draws, odd batch
and journal limits) bit-exact against CPU replay at every checkpoint. The
release micro-benchmark `gpu_tiny_fill_host_cost` measures the processor-side
cost per tiny fill without the GFIFO.

## Measurements

Release, `rex-gpu,rex-jit`, RTX 5070 / NVIDIA 617.42, timestamps off, five
interleaved one-second samples per workload, median ratio. CPU samples vary
±5% between runs and the host was noisier later in the day, so compare
ratios across several runs, not single values.

| Step (Vulkan) | Raster (1,280 px) | Tiny (8 px) |
|---|---:|---:|
| Baseline: barrier shader, 1,024 batch | 0.758x | 0.463x |
| Tile-binned shader | 1.984x | 0.730x |
| + 16,384 batch, no per-draw `Arc` clone | 1.789x | 0.872x |
| + one failure lock per accepted draw, deferred CPU damage | 1.828x | 0.906x |
| + map spin, damage-limited CPU copy (3 runs, + D3D12) | 2.33 / 2.36 / 2.35 (2.29) | 0.962 / 0.945 / 0.956 (0.951) |
| + tile-major plane (3 runs, noisy host) | 2.45 / 2.88 / 2.84 | 0.928 / 0.951 / 0.972 |
| Final tree, quieter host (3 runs) | 2.468 / 2.484 / 2.548 | 0.966 / 0.965 / 0.961 |
| Final tree, D3D12 (1 run) | 2.472 | 0.947 |

Stage evidence (one ~1 s tiny sample unless noted):

- Baseline `device_wait` was 704 ms (raster) and 402 ms (tiny): compute queued
  behind checkpoints. After the tiled shader tiny `device_wait` was 26 ms.
- Tile-major cut tiny checkpoint transfer from ~470 MB to ~125 MB per sample
  and `device_wait` from ~25 ms to ~13 ms, but tiny throughput did not move:
  checkpoints were no longer the bound.
- `gpu_tiny_fill_host_cost`: 37.8 ns per tiny fill on the processor thread =
  ~10 matcher + ~20 accept (including one ~8 ns uncontended lock pair) + 3
  submit + 6 checkpoint.

A tried-and-reverted change: making the recovery journal the single fill store
(batch as a journal range) and deriving damage from the bins. It removed a
32-byte copy and the per-fill damage math but measured no gain (38.3 ns in the
micro-benchmark; gate 0.943 / 0.938 / 0.938), so it was not kept.

## Where the tiny gate stands

The CPU renderer spends ~260 ns per 8-pixel draw on the processor thread, and
most of that is GFIFO handling both paths share: six register entries, each
taking the framebuffer-access lock in a `rex-gpu` build, plus GO dispatch. The
GPU-specific cost is ~38 ns plus the `gpu_replay` slot lock (~8 ns). The CPU
path's own draw cost (shader lookup, 8 pixel writes, damage atomics, rexdiag
counters) is of the same order, so the ratio is near 1 by construction and a
5% swing is ~14 ns. Reducing *shared* overhead would push the ratio further
from 1, not closer.

What could still buy margin, with the honest cost of each:

1. **Elide the `gpu_replay` slot lock on the processor thread** (~8 ns, ~3%).
   Every mutation of the slot already happens under `framebuffer_access`,
   which the processor holds for the whole GO, but report readers
   (`gpu_replay_report`) take only the slot lock. They would need a
   lock-free snapshot or `framebuffer_access`, and the soundness argument has
   to be written down first.
2. **Asynchronous checkpoints**: no longer worth it for the gate — the whole
   checkpoint now costs ~5 ns per tiny draw.
3. **Revisit the gate itself.** At 8 pixels per draw the workload measures
   GFIFO dispatch, not rasterization. Whether 0.95x is the right bar for that
   case, or whether the gate should require ≥0.95x on a mix that includes real
   guest draws, is a project decision, not something to tune around.

## Constraints kept

- The gate asserts every draw was GPU-handled with zero fallbacks; no small
  fills are routed to the CPU.
- `check_limits` requires three storage buffers, 256-invocation workgroups,
  1,344 workgroups per dimension and bindings large enough for the plane,
  bins and parameters. (The hardened tree also pins these in
  `preflight_device_requirements` and keeps an out-of-range
  `InvalidParameterWrite` fault; neither is in the lean PR.)

## Original cost model (2026-11-09, confirmed)

The old shader ran a whole batch in one 256-invocation workgroup with a
`storageBarrier()` after every draw: 669 µs per 1,024 tiny draws and
1,421 µs per 1,024 raster spans. The GPU consumer does not wait on
`BatchFull` submits, only at checkpoints, so both gates were bounded by compute
queued behind a checkpoint. Candidates were a pixel-ownership shader over the
batch bounding box (rejected: scattered small draws cost up to 43,008 tests
each), conflict-free epochs (rejected: a dispatch per epoch, degenerates on
overlap), and tile binning (built).
