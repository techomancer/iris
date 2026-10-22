# Pipeline fences (idea, not built)

Our FIFOs are far deeper than the board's (`hq_fifo` and `rss_fifo` hold
64K entries each), so "is the work queued before X done?" is answered
today only by "is the whole board idle?" (`idle()`, `drain_rss`,
`wait_idle`). Fences would answer the precise question cheaply.

## Mechanism

- A producer holding `submit` takes the next `seq` (u64) and pushes
  `tag::FENCE(seq)` into `hq_fifo` (`try_push`: full means "busy", as now).
- The HQ3 thread, on popping it: `hq_done.store(seq, Release)`, then
  forwards the fence into `rss_fifo`. Host DMA runs synchronously on the
  HQ3 thread, so `hq_done >= n` means every command and host-memory DMA
  read before fence n is finished.
- Each raster engine, on reaching its copy: `re_done[id].store(seq,
  Release)`. Drawing before n is done when `min(re_done[..nrss]) >= n`
  (Maximum Impact's two engines read one stream at their own pace).
- Checks are two or three Acquire loads. The fence seq goes into
  recordings so replays stay deterministic.

A check must name a fence issued after the work it cares about. For a
poll: issue a fence only if the `hq_fifo` tail moved since the last one,
otherwise reuse the outstanding one, so a polling loop does not flood
the FIFO.

## Where it would fit

- **DMABUSY / FIFOSTATUS / STATUS idle bits.** Same answer as `idle()`
  while the polling CPU is the only producer, so no gain yet; listed
  for completeness.
- **Display writes after queued drawing.** DCB writes (buffer swaps,
  colormaps, window-ID tables) take effect at once, ahead of drawing
  still in `rss_fifo`. Tagging them with the current fence and applying
  them in the refresh path once `re_done` reaches it would keep swaps in
  step with drawing without a full drain. Best candidate for leftover
  sync glitches (ideas in motion).
- **Shared TRAM on Maximum Impact.** `Rss::te_write` hands off per texel
  row; one fence per load (at the load-start XFRCONTROL) would do.
- **Read-backs.** TRAM saves, `dma_read_line`, `transfer()` drain
  `rss_fifo` completely; a fence would wait only for what precedes them.
- **Perfetto.** A fence instant per stage shows queue latency (issue to
  HQ3 to each RE) directly.

Context: the status-register rework (612ab398, ad5be1d3) and
`guest-gl-benchmarks-miss-board-time.md`.
