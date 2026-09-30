# Context save: GEDMA reads must wait for the queued 0x1E1

Gr2PcxSwap / _Gr2CXSaveRestore queue 0x1E1 (save main) and start the VDMA
read of HQ2_GEDMA right away. On hardware the microcode runs 0x1E1 before any
GEDMA data exists. Our HQ2 is a thread behind a deep FIFO: when it was still
busy, the first VDMA read beat the HQ2 and returned the end of the previous
image (0), so the saved image was shifted one word. Its restore then failed
the magic check and the incoming context kept the outgoing one's live GL
state: another demo (ideas) drew into amesh's window with amesh's window,
clip and matrices. Rare: took minutes of two GL demos running together.

Fix: HQ2_GEDMA reads wait (bus busy) while the port is empty and the HQ2
still has work; see gedma-read-port.md (the first fix counted queued 0x1E1
tokens, `cx_save_pending`; replaced by the general read-port rule when pixel
DMA reads needed the same). Also: a rejected restore now resets GlState to
defaults instead of keeping the previous context's state (draws nothing
until the next window / matrix tokens, rather than into someone else's
window).

Trace diagnostics: GE_HQMSAV lines show `(live was <ctx>)` and flag
`LEAK?` for state 1 over another context's state; restores report
`REJECTED: <reason>`. Test: `gl_context_save_read_waits_for_hq`.

