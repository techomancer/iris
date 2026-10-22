# GR2 FIN3 must not be satisfied by an older Finish

Symptom: an OpenGL program (ideas) flickers after a resize / window fiddling
and keeps flickering; restarting it sometimes fixes it. Buffer masks and the
kernel's flips are correct.

Cause: FIN3 is a level flag. libglcore Finish = write 0x0A3, spin on
hq.version bit 0, then ack (0x6B000 = 0). If FIN3 is already 1 when the wait
starts (an older Finish executed late, or the kernel restored a context's saved
FIN3 bit at a switch), the wait passes immediately. On hardware the pipeline
drains in microseconds, so that is harmless; with our deep HQ FIFO the frame is
still queued, SwapBuffers goes ahead, and the kernel flips at retrace to a
buffer the HQ is still drawing. The late Finish then sets FIN3 for the next
wait, so the pipeline stays exactly one frame behind forever.

Found with a one-off event log (logpoints on the kernel's RRM / GR2 swap
path plus GR2 token and XMAP events in one file; kept on the
`evlog-instrumentation` branch, not merged): the HQ-executed token sequence
equalled the CPU-written sequence shifted by one frame, while the kernel's
unmap / fault / retrace flip / revalidate sequence was correct.

Fix (src/dev/gr2/mod.rs, `fin3_pending`): count 0x0A3 / 0x155 tokens when the
CPU queues them, count down when the HQ executes them; `version` bit 0 and the
0x6B000 read show FIN3 only when none is pending. Test:
`fin3_waits_for_queued_finish`.
