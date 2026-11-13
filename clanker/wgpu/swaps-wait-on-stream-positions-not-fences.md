# Swaps wait on stream positions, not on fences

Status: design, 2026-11-07 (ea947636). One part open, see the end.

**Problem.** With the GPU raster unit the drawing a swap should show may
still sit in an unsubmitted batch when the retrace latches the flip, and
the display composes from GPU memory. Making the display thread wait for
the GPU per frame costs the overlap we want.

**Design.** Order by position in the command stream, not by GPU fences:

- `CommandFifo` counts stores `pushed` and `retired` (absolute, monotone;
  the ring's own indices wrap).
- A swap arm or swap-control write records `pushed` at that moment
  (`HostInterface::swap_after`, from `stream_pos`, which the bus path sets
  before the front register write).
- `GpuRaster::tick(idle, retired)` runs at the end of every board slice; it
  publishes `retired` to `DisplayShared::covered` once everything recorded
  up to it is **submitted** (every flush, or a tick with nothing recorded).
- `HostInterface::retrace(covered)` latches flips only when
  `swap_after <= covered`; otherwise the flip stays armed for a later
  retrace.

Submitted is enough, completed is not needed: the scan-out's own
submission goes on the same wgpu queue, behind the batch, so the composed
frame shows it. The CPU raster unit leaves `covered` at `u64::MAX` and
flips latch at once, as before.

Test: `a_flip_waits_until_the_raster_unit_covers_the_stores_before_its_arm`
(tests/display.rs).

**Open, minor.** The status words a guest polls for idle (DMA busy, the
sync words, `access_waits` reads) still go through `raster_wait`: flush and
wait for the GPU. The same idea would let them answer "busy" until a
per-submission completion counter (`queue.on_submitted_work_done`) passes
the position of the last store before the read, without blocking the
board's thread. Measured live it does not matter yet: steady-state GL
Quake made 72 waits over the whole session and no readbacks.
