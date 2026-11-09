# Host DMA stops at the kernel's mapped window (TLB-miss refill not modelled)

Status: open, 2026-11-08 (IP28 High IMPACT, IRIX 6.5.22, the cloned
`purple` image).

**Symptom.** During glsuite's visual sweep the emulated IP28 wedged: the
CPU looped inside the general exception vector (0x80000180..0x80000240,
EXL set, ADES, BadVAddr 0xffffffffffffff15). The vector page had been
overwritten: zeros, then pixel-like words (`40040010 00004004 00100000`
repeating, or `ff0fff0f`). Physical write breakpoints (`bp add 0x180 pw`,
`0x20000180 pw`) never fired, so no CPU store did it.

**Cause.** A raster-to-host DMA (`DMA read pool 0 ... base 0x748 stride
1542 lines 256 bytes 1542`, about 97 pages) through a pool whose table the
kernel had filled for 4 pages: `MAX_IM1_PTR_OFFSET` (0x830) = 4,
`DMA_PG_RANGE_IM1` (0x82B) = 2. Entry 4 is 0, so the model wrote page 4
onwards to frame 0: on IP28 that is the low-memory alias holding the
exception vectors. The real board's HAG is a virtual DMA engine (HQ3.h:
graphics TLB, flag bit 7 TLB miss): past the window it asks the kernel for
more, and the kernel refills the table (mgras_dma: Fill/Plug/EditAppend).

**Now.** `hq3.rs` `dma` stops a transfer at entry `MAX_PTR_OFFSET + pool`
and logs it once ("stops at page N, past the M the kernel mapped"); any
transfer that still reads a zero frame is logged too. Such a read returns
only its first window.

**To do.** Model the miss: stop at the window, raise flag bit 7 and its
interrupt, resume from the same line when the kernel has refilled the
table. Check first which client issued it (the desktop's powerflip
background, IRIS GL, runs throughout, and the log's colour-table and
evaluator tokens are not glsuite's), and what DMA_PG_RANGE means (2 for a
4-page window: log2?).
