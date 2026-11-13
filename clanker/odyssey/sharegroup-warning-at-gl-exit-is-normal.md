# "attempted sharegroup operation from unbound gfxp" at GL exit is normal

Status: measured 2026-11-05 on octopus (real V10, IRIX 6.5.22).

**Symptom.** Quitting Quake (or any GL program) under the emulated VPro
prints `WARNING: odsy: attempted sharegroup operation from unbound gfxp` on
the console, many times.

**Cause.** The kernel's memory-manager ioctls (alloc, dealloc, make resident,
query, transfer) print it when the calling process's resource node has no
per-client context record. At exit libGLcore still frees texture memory
after that record is gone. Details are in `ignore/odsy` `docs/kernel.md`.

**Rule.** It is not an emulator fault. The real board prints it too: tyrquake
quitting cleanly on octopus logged it 57 times in one second, and piglit's
tests log it at exit. Only worry if it appears while a program runs, or
after a CTXSW, MKRES, GLGET or DMA timeout and "Graphics error" (a board
teardown).
