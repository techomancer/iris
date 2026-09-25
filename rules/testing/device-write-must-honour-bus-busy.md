# Test harnesses calling `write32` on a device must honour `BUS_BUSY` — 2026-09-15

`Rex3::write32`/`write64` return `BUS_BUSY` (== `EXEC_RETRY`) when the GFIFO
is full. On the real CPU path that makes the store re-execute; in a test that
calls `rex.write32()` directly and ignores the return value, the entry is
**silently dropped**.

A host thread pushes far faster than the painter drains, so any throughput
test that pumps more than ~64K entries fills the ring and then spends its
time in the drop path. `gfifo_pressure_sweep` reported 22 Gpx/s for
1280-pixel Gouraud spans, with spans/s flat (~18M) across a 160x range of
span lengths — the signature of timing failed `try_push` calls (~8 ns each)
rather than draws. The single-span validation step could not catch it
because it writes into an empty queue.

Fix in `src/rex3_tests.rs`: `w32`/`w64` spin while the status is `BUS_BUSY`;
`reg()`, `reg_go()` and the direct `write32(go_addr(..))` call sites use them.

Rule: if a number from a device-level throughput test looks better than the
per-pixel arithmetic could possibly allow, check whether the writes are
being accepted before believing anything else about it.
