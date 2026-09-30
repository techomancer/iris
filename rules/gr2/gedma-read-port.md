# HQ2_GEDMA reads: a ring the HQ2 fills, busy while the HQ2 has work

The read side of HQ2_GEDMA (0x6a068) is the HQ2's output to the host. The
kernel queues a request and starts the VDMA read at once, with no barrier:

- 0x1E1 context save (Gr2PcxSwap / _Gr2CXSaveRestore);
- 0x152 pixel DMA read (Xsgi expReadImage*, XGetImage over 1024 pixels);
- 0x0AC pixel DMA read (IRIS GL lrectread, OpenGL glReadPixels KDMA).

Our HQ2 is a thread behind a deep FIFO, so the first read often beats it.
`Gr2::gedma_out` is a single-producer ring (inline AtomicU32 array, head /
tail counters) the HQ2 thread fills when it executes the request; a read:

- a word in the ring: return it;
- empty, and the HQ2 had work (FIFO not empty or `hq_busy`) sampled BEFORE
  looking at the ring: bus busy (the VDMA worker and the CPU retry);
- empty and the HQ2 idle: overrun, log, return 0.

Sampling order matters: the HQ2 pushes its words before it drops
`hq_busy`, so an idle HQ2 seen first means everything it produced is
visible. Checking the ring first races (empty, then the HQ2 pushes and goes
idle, then "idle" = false overrun), same as the FIN2 stall
(fin2-wait-must-stall.md).

Each transfer starts with `gedma_begin`, which drops words a previous one
left unread (they would shift this one). The HQ2 waits while the ring is
full, up to 2 s without a reader, then drops the rest.

What went wrong without it:
- save: a read that beat 0x1E1 returned the previous image's tail, the
  image shifted one word, the restore was rejected and another context's
  GL state stayed live (cx-save-must-wait-for-hq.md);
- 0x152 / 0x0AC unhandled: zeros, no FIN2, "Gr2PixelDma: TIMEOUT", the
  kernel reset the board (and with a GL client, panicked).

Tests: `gl_context_save_read_waits_for_hq`,
`ddx_dma_read_pixels_streams_gedma_and_sets_fin2`,
`gedma_read_with_hq_idle_is_an_overrun`, `gl_dma_read_*`.
