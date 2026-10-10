# Guest-side GL benchmarks do not see the board's drawing time

`gltest --bench` (fill rate) on IMPACT reports nonsense for short runs:
3,000-10,000 Mpx/s for 40 full-window quads, and a first run of 4,453 Mpx/s
next to 85 for the rest. Longer runs (200 quads) look plausible but still
under-count.

Why (traced 2026-10-10, `mgras trace ... cpu,hq` around `gltest --bench 3`):
glFinish is a `CP_RETURN_MODE` round trip. The client polls FLAG_GE_DATA and
then reads GE_READBACK_LO, and that read waits (bus busy, retried) until the
board is idle, so glFinish does wait for the raster engines. But host time
from the last quad to the readback was ~10 ms, and the client's own timer
measured 3 ms: guest time hardly advances while the CPU retries the busy
load. Only work that also keeps the guest CPU busy (FIFO back-pressure on
long runs, triangle setup) shows up in guest timings.

So:
- Compare raster performance offline: record the session in the guest
  (`mgras rec <file>` ... `mgras rec off`) and time its replay:
  `MGRAS_REPLAY_NOCHECK=1 MGRAS_REPLAY_BOARD=high|max MGRAS_REPLAY=<file>
  cargo test --release --lib --features gr4-jit mgras_replay_file -- --nocapture`
  prints the wall time and the final state hash (which must match across
  boards).
- Treat guest-side numbers as "how long the CPU was busy", not fill rate.
- Run at least two warmup passes: the first runs also time the raster JIT
  compiling its shaders.
