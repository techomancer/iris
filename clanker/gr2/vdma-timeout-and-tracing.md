# Tracing GR2 can push a pixel DMA past IRIX's VDMA timeout

Status: rule, 2026-11-24 (issue 844, SOFTIMAGE 2.66 on IRIX 5.3, Indy XZ).

**Symptom.** `WARNING: VDMA  TIMEOUT, vdma status 40`, then
`Gr2PixelDma: DMA write failed`, `Graphics error: GE PC = 0x0`, a board
reset, and sometimes a kernel fault after it. Seen with a full GR2 trace
running (`si.log`), on a 648x50 lrectwrite (token 0x0B5, 32 400
words through HQ2_GEDMA).

**Why.** The kernel's polled VDMA wait is 500 000 iterations of
`DELAY(1)` plus one MC DMA_RUN read. DELAY is a spin loop calibrated at boot
(`decinsperloop` in the PDA, 0xFFFFA15C on 5.3; 5 on this guest, i.e. 200
spins per "us"), so the budget is a few hundred ms of host time at most. A
trace writes two lines per GEDMA word on the VDMA side but none for the
CPU's DMA_RUN polls, so the transfer slows (160+ ms) while the timeout does
not. On timeout the kernel writes DMA_STDMA = 0 and waits for RUNNING to
clear; our MC does not abort on that write, so the kernel waits for the
transfer to finish before it reports.

**Rule.** Treat a VDMA timeout seen only under tracing as a tracing
artefact first; reproduce untraced with `mc vdma on`, whose `VDMA end` lines
give each transfer's host time. Keep the GIO32 path fast: it translates the
memory side once per page (a per-byte walk took the DMA lock the CPU polls
and read a PTE four times a word, ~0.75 us a word; now ~0.15 us).
Context save/restore through the read port still costs ~1 us a word; it is
paced by the HQ2 thread producing the words.
