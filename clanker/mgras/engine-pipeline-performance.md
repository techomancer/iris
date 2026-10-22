# The CPU -> HQ -> raster pipeline: what made it slow

Found profiling the Quake 3 timedemo (`demo four`) on High IMPACT with macOS
`sample`. Read before changing the FIFOs, the engine loops or the texture paths.

- Big 32-bit textures do not fit TRAM, so libGLcore's texture manager reloads
  them every frame (Quake 3: ~500 DMAs, ~13 MB a frame, mostly TRAM restores of
  RGB at 3 bytes a texel). The texture-load path is a hot path, not a cold one.
- The CPU polls DMABUSY after each DMA, so anything that makes the HQ wait on
  the engines makes the CPU wait too. Nothing on the HQ's host-to-board DMA
  path may drain the raster engines; the engines check the armed transfer when
  the lines arrive.
- The raster FIFO must hold most of a frame's loads (1 M entries). At 64 K the
  HQ spun for room on every load and the CPU, the HQ and the engines took
  turns. 4 M gained only another 2.6%.
- Per-entry stores to shared atomics are false sharing: the HQ's and the
  engines' busy flags and `dirty` sit on one line. Store a flag only when it
  changes. Likewise unpark a consumer only when it is parked (`sleep`/`wake`,
  SeqCst fences on both sides).
- TRAM stores of 8-bit components are whole bytes (`Te1::load_line`); the
  nibble read-modify-write path is for 4- and 12-bit components.
- With several engine threads on one `Te1`, a handshake per texel line costs
  more than the line. Engines other than 0 pass texel loads and wait for them
  (`te_settle`) only before an entry that may sample.
- Engine threads: rows go to y % n for n = 1, 2 or 4 (`IRIS_MGRAS_ENGINES`) on
  any board; layout follows DRBsize, not n. A first-row skip must be
  `(engine - fy) * ysign & mask`: the XOR form agrees with it only for n = 2.
- Measured and not worth keeping: padding `dirty`/`trace_mask`, reading
  `general` before the swap, a doubleword copy in `read_host`, keeping the HQ
  up 200 us before parking, word-wise ERAM packing, one fence per `wake_rss`.
