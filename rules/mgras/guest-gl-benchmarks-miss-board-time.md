# Board-bound GL work slows the guest clock (and guest benchmarks lie)

`gltest --bench` (fill rate) on IMPACT reports 2,000-10,000 Mpx/s for
40-quad runs. Measured 2026-10-10 (Maximum Impact, 1024x768): guest `date`
advanced 6 s while the host took 33 s for the same run, and `ioc ticks`
over a run showed only 219 of 1,182 system-clock ticks acknowledged (963
merged, waits up to 140 ms); at idle all 1,009 are acked within 0.5 ms.
Guest time is lost because clock interrupts are not taken, not because
retries skip the interrupt poll: CP0 Count follows host time
(`materialize_count`), and a BUS_BUSY retry goes back through
`step_preamble!`, which polls interrupts.

Where the CPU sits without taking ticks (`mgras trace F cpu,hq`, gaps
between accesses):
- ~100 ms after each `__MGR_RETURN_MODE` (glFinish): the HQ3 raises
  FLAG_GE_DATA only after the raster engines finish the queued quads; the
  client polls the flag a handful of times meanwhile.
- ~100 ms after a context switch: the kernel reads FIFOSTATUS, which our
  model answers only once the whole board is idle (`read_needs_idle`).
Ticks that go unacked through a wait that does poll interrupts mean the
guest is waiting with interrupts masked (most likely the kernel at splhi);
not confirmed with a PC sample.

Why it is worse than on the board: our hq_fifo/rss_fifo hold 64K entries
each, so a client can queue a large fraction of a second of drawing without
ever blocking, and FIFOSTATUS/STATUS then wait for all of it. The real
CFIFO is shallow, reports its level at once, and raises the FIFO high-water
interrupt (cfifo_hw/lw) so the kernel can put a writer to sleep. Possible
fixes, none made yet: answer FIFOSTATUS with the real level without waiting;
model the high-water interrupt; bound the queued work so writers block on
BUS_BUSY in user mode.

Until then:
- Compare raster performance offline: record the session in the guest
  (`mgras rec <file>` ... `mgras rec off`) and time its replay:
  `MGRAS_REPLAY_NOCHECK=1 MGRAS_REPLAY_BOARD=high|max MGRAS_REPLAY=<file>
  cargo test --release --lib --features gr4-jit mgras_replay_file -- --nocapture`
  prints the wall time and the final state hash (which must match across
  boards).
- Treat guest-side fill rates as meaningless; `ioc ticks` shows whether a
  workload is losing ticks.
- Run at least two warmup passes: the first ones also time the raster JIT
  compiling its shaders.
