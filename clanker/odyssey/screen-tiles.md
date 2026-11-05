# The VPro fragment pipeline runs on screen tiles, one owner each

Status: 2026-11-04, current (`pipe/ring.rs`).

## Design

The raster thread sets primitives up (clip, project, triangle record) and
writes one command per primitive into a slot ring: a triangle (record and
bounding box), a quad it shaded itself (lines, points, smoothed polygons),
or a clear rectangle (one per clip rectangle). The slot's index goes on the
queue of every tile (`[graphics] tile`, default 32, in plane pixels on a
4096-pixel grid that wraps) the command touches, and the slot counts them.
A tile with work goes on a lock-free ready queue; a worker pops it and owns
it until its queue is empty: it walks each triangle clipped to the tile and
shades and stores each quad at once, stores shaded quads, clears its part of
rectangles. The raster thread retires slots whose count is zero in order,
so `hold`, `reclaim` and `drain` keep their meaning.

Workers are `[graphics] threads` (default 2: most hosts are not 32-core
machines); the replay takes `VPRO_REPLAY_POOL=<threads>,<tile>`.

## Rules

- One owner per tile is the only ordering. Every command must go to every
  tile it can touch (`commit` takes its bounding box); a box too small lets
  a pixel be written out of order.
- The walk is clipped to the tile but keeps the window's 2x2 quad grid, so
  quads and their level of detail are the same as an unclipped walk: output
  is bit-exact with the inline pipeline.
- The depth pre-test is exact in a tile owner's walk. On the raster thread
  (draws not walked by the tiles, `IRIS_ODSY_SHADER=inline`) it is off.
- A tile owner copies the draw's context for each triangle: a cached copy
  could belong to a freed draw whose address a later draw took (the same
  bug the gate pool had).

## Measurements (Quake timedemo replay, first 20 M records, threaded)

| | time |
|---|---|
| three stages (raster, shader, back end) | 86 s |
| gate pool (jobs of 128 quads, primitive gate, 4 workers) | 37-40 s |
| tiles 32, 4 / 8 / 16 workers | 33 / 23 / 20 s |
| tiles 64, 4 / 8 / 16 / 24 / 32 workers | 36 / 22.5 / 18.6 / 18.2 / 18.4 s |
| tiles 16, 16 workers | 29 s |
| tiles 128, 16 / 32 workers | 27 / 26 s |

At 16 workers on 64-pixel tiles no thread is saturated (raster 73%, front
end 53%, workers about 54%): the limit is serialisation, most likely the
drains before texture uploads and other board-memory jobs (`raster_wait`),
or tiles that get much more work than others. A perfetto trace would tell.

The gate pool's sweep (jobs of 4 to 256 quads, 1 to 16 workers, storing per
quad or per job) is in git history (`c72e83d3`, `quad-worker-pool.md`):
small jobs collapse with more workers, and the raster thread capped it.

## When to revisit

Tile owners walk, shade and store their own quads, which suits Quake: cheap
shading, many small primitives. A shader-heavy workload (long texture
chains, many fragments per primitive on few tiles) may want the work split
differently again, for example shading spread over more threads than own
tiles. Measure before changing it.

