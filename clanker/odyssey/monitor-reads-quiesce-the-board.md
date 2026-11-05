# Monitor reads of VPro state quiesce the board; pixel reads go lock-free

Status: design, 2026-11-02; pixel side updated 2026-11-04 for board memory.
Implemented in `src/dev/odyssey/debug.rs`.

## The problem

`Board` has no lock. The board's thread (`odyssey-work`) owns it while it
runs stores, and a processor touches it only after a waiting read has drained
the FIFO and the raster queue (`BoardLocks::bus_read`, `interface.rs`). The
monitor runs on a third thread and the other boards' pattern, MGRAS's
"read-only peek, may tear" through an `UnsafeCell`, is unsafe here: the
GL context's `Vec`s are mutated by the worker, and a torn read of those is
UB, not a stale number.

## The rule

- **State commands** (`status`, `regs`, `fifo`, `gl`, `geom`, `tex`, `win`,
  `mem`, `backend`, `unimp`, `hash`) go through
  `BoardLocks::quiesce(timeout)`: take the processors' `submit` lock, and if
  the FIFO is drained and the raster queue empty, hand out the `BoardGuard`,
  the same contract a waiting read uses. Otherwise drop the lock, `hurry` the
  worker, sleep 200 us and look again, until the timeout (500 ms). With no
  board thread (tests, a stopped machine) the waiting stores run in the
  caller, as `work_drain` does for a waiting access.
- The guard holds `submit`, so processors stall on the bus while it is held.
  Format the whole answer into a `String` under the guard and write to the
  monitor socket after dropping it; never hold it across I/O.
- A board that stays busy answers `board busy` and the FIFO and raster
  levels. `vpro status` prints the front-side half regardless and only marks
  the board half as not read.
- **Recording control** (`rec <file>`, `rec mark`, `rec off`) quiesces too,
  and takes the recorder's lock under the guard: the processors' lock first,
  then the recorder, the order the bus path uses (`BoardLocks::taken`). The
  DMA path takes the recorder alone. See `trace-and-recording.md`.
- **The trace** (`trace ...`) never quiesces: it only switches the mask and
  the sink, which is a leaf lock.
- **Pixel commands** (`pix`, `raw`, `peek`, `shot`, `fbdump`, `dispmode`,
  `lut`, `cursor`) never quiesce. They read board memory through the pointer
  fixed at `Board::allocate` (`DisplayShared::vram`) and the registers
  through `SharedRegs` (atomics), exactly as the display thread does, so they
  never stall the guest. A pixel read under a draw in flight may be a frame
  stale, which is what a screenshot is anyway.

## Why not quiesce the pixel commands too

`fbdump` and `shot` read up to 128 MB of board memory and compose 1.3 M
pixels. Holding
`submit` that long stalls every processor for tens of milliseconds, and a
guest that keeps the FIFO fed might never give a 500 ms window. A dump that
is one frame stale is the same thing `screen shot` gives on every board.

## Where `vpro hash` fits

`memory_hash` (once `plane_hash`) lives in `debug.rs` so the live monitor
and the replay test hash the same bytes: all of board memory. It quiesces so
the hash is of a settled board.
