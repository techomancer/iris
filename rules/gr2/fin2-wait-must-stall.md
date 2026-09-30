# FIN2 polls must not outrun the HQ2 (pixel DMA timeout)

The kernel acks FIN2 (write 0x6A04C) right before a FIN2 command (pixel DMA
0x147, context save/restore) and then polls version (0x6A040) bit 1 in a
counted loop: 100,000 x us_delay(1) for pixel DMA (_Gr2DMAtrigger /
_Gr2MCDMAtrigger). Hardware answers in microseconds. Our HQ2 thread draws a
large pixel DMA row by row through RE3 and can take longer than that budget
in emulated time.

Symptom (IRIX 5.3, XZ): console "Gr2PixelDma: TIMEOUT gfx DMA did not
complete (finish flag not set)", "Graphics error: GE PC = 0x0"; the kernel
detaches Xsgi (blank screen), Xsgi's input stream stays linked, and the next
mouse event panics in GfxProto calling PositionCursor through the detached
board's NULL function table. The NULL call is only the aftermath; `gr2 hq`
afterwards shows FIN2 set: it arrived, late.

Fix (mod.rs `fin2_wait`): the FIN2 ack arms a wait; while armed, a version
read with FIN2 clear returns bus busy as long as the HQ2 has queued or
in-progress work, so the kernel's first poll after the DMA sees FIN2. With
the HQ2 idle and no FIN2 the read returns at once and the kernel times out,
as on hardware. The stall gives up after 2 s of host time WITHOUT
PROGRESS (FIN2_STALL_LIMIT; progress = the HQ2 or RE3 FIFO consumer moved,
`GFifo::consumed`), in case the pipeline can never finish (RE3 held by a
CPU-driven RWDATA readback). Not counted from the ack (the DMA in between
can take long on a busy host; that failed this test on a GitHub CI runner),
and not a fixed cap from the first poll either: see gltest below. Sample "HQ2 working" BEFORE reading the register (the HQ2
raises FIN2 before it consumes the entry and drops hq_busy); sampling after
lost the race in about one run in three. Test:
`kernel_pixel_dma_fin2_poll_waits_for_hq`.

Related: fin3-must-track-pending-finish.md (the opposite problem: a stale
FIN3 seen too early).

The same wait covers the context switch (Gr2PcxSwap: GE_HQMSAV, 0x1E0 /
0x1E6, save / restore; 1,000,000 x us_delay(1)). With our deep FIFO a GL
client can queue far more work than hardware's 512 words allow: `gltest
--bench N` is N full-screen 800x600 quads (a few FIFO words, 480k pixels
each) then glFinish. N = 500 is ~2.4 s of RE3 work; a context switch behind
it outlived the old fixed 2 s cap, the kernel's FIN2 poll timed out and it
crashed in Gr2PcxSwap's 0x1E5 clip loop (PC 0x882adcac, IRIX 6.5.22 XZ);
N = 300..400 stayed under 2 s and passed. Hardware passes at 1000. Hence the
limit counts from the last progress, not from the first poll.

Second cause of the same console message (IRIX 5.3 MRI software): the pixel
DMA command itself was unimplemented. lrectwrite goes out as token 0x0B5
(0x0B8 when zoomed), not the 2D 0x147; the HQ2 never raised FIN2 however long
the kernel waited. Check the trace for "tok0xNN ... not implemented" right
after `wr hq.dmasync = 2` / `wr hq.fin2 = 0` before blaming timing. Test:
`gl_pixel_dma_lrectwrite_ci16`.
