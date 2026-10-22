# Colour-index glCopyPixels runs a pipeline whose state is never loaded

Symptom: PowerAnimator 9 (colour-index double-buffered main window) lost
its viewports or its menu/prompt strip. It copies front to back after a
swap (glCopyPixels), then draws only what changed and copies small strips
back. Every copy wrote index 0.

Cause: libGLcore implements glCopyPixels on IMPACT through host memory:
GET_PIXELS reads the strip into host RAM, then the GE pixel pipeline sends
it back (SEND_PIXELS routine 0x49D0 via _WRITE_DMAGESETUP). The fast copy
path (`__glMgrFastCopyPixels`) builds the program at GE 0x13B (`mask`
0x1A3E with state 0x1D6, then `pipecontrol`) but never calls
`__glMgrInitPixelPipe`, so the mask state and the element type (ERAM
0xDBB) are never sent. Our zero-initialised GE memory made the AND mask 0,
and type 0 made the 16-bit indices read as bytes.

glDrawPixels from the host does send them (`0x1D6: [ptrs.., 0xFFFFFFFF]`,
`0xDB7..0xDBC` with type 4). pixtest never caught it because its draws ran
before its copies in the same context and left the state loaded.

Model: `PixMem::loaded` tracks which words a state write loaded. A `mask`
whose state was never loaded passes indices through (GL's copy with no
transfer operations); an unloaded element type takes the 32-bit code's
rule (size from the bytes that came). Where a real GE gets those words is
not known (no CRAM image in the microcode files, no kernel write); see
ignore/gr4/GE11.h, `mask`.

Repro: `test/gltest/cicopy.c` (`sh irix.sh cicopy; ./cicopy`), and pixtest
`--ci 12 --only ci12/copy/`.

Debugging that found it: a power-on recording (`IRIS_MGRAS_REC`) replays
to the live state with all checkpoints; a mid-session `mgras rec` does
not (the GL HLE never sees the context's CI format). Replay with the
raster JIT off and on gave the same final hash, which put the bug ahead of
the raster engine. Keep recordings out of the repo root: a later run with
the variable still set overwrites them.
