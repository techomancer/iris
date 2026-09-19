# PREFETCH primes the PIO read pipeline — the `0xdeadbeef` GO is the primer

A PIO pixel readback from IRIX looks like this in the bus log:

```
reg=0000(DRAWMODE1) val=0000000004007189  ; RGB 8bpp host:8bpp cmp:7 logicop:ZERO RWPACKED CI PREFETCH
reg=0004(DRAWMODE0) val=0000000000000045  ; READ BLOCK COLORHOST
reg=0120(BRESOCTINC1) val=0000000000000000
------- GO reg=0230(HOSTRW0) val=00000000deadbeef -------
------- PURE_GO -------
```

The `0xdeadbeef` is not data. It is a **throwaway write whose only purpose is to
carry the GO bit**, and it is the software half of the PREFETCH protocol.

## What the spec says

`DRAWMODE1` bit 26, PREFETCH — "Enables host framebuffer pixel prefetch
mechanism for PIO reads" (rex3.pdf §3.10, *Framebuffer PIO and DMA*):

> For pixel reads, the DRAWMODE1 PREFETCH bit must be set to 1, with the
> DRAWMODE0 OPCODE=read. The set up of the DRAWMODE registers should be
> performed with a write to the GO (address+800H) command, **prefetching the
> data, reducing the I/O latency of subsequent transfers.** Pixel data may then
> be read from the HOSTRW register, again with the "GO" command.

So the read pipeline is *read-then-advance*: every GO-space read returns the
currently latched word **and** starts fetching the next one. That leaves the
first word with nothing to return, which is what the priming GO exists to fix —
it runs one primitive with no preceding read, so the first real read already has
a value waiting.

This is the hardware counterpart of the discard loop in
xf86-video-newport's `NewportXAAReadPixmap`, which issues N-1 GO reads then one
final non-GO read for N words.

## Two conditions that are easy to miss

**PREFETCH must be OFF for DMA reads.** Same section, and it is a hard
requirement, not an optimisation:

> To insure correct operation with the MC, all pixel DMA read transfers must be
> performed with **DRAWMODE1 PREFETCH = 0**. In addition, a pixel DMA read
> transfer may not begin until the graphics pipe is idle (STATUS GFXBUSY = 0).

That is why `dma_read64` inverts the ordering instead of imitating PIO: VDMA has
no software-side discard loop, so every `dma_read64` result must already be real
data. It pushes a GO-only entry, waits for it to execute, *then* reads — see the
doc comment on `Rex3::dma_read64` and
`rules/rex3/hostrw-batching-design.md`.

**STOPONX/STOPONY should be zero for PIO**, "indicating one GIO word per
primitive GO" — the one-word-per-GO feed. That is the same `!stopony` shape
whose batch handling dropped every row after the first; see
`rules/rex3/hostrw-tests-must-use-partial-word-rows.md`.

## What IRIS does with the bit today

**Nothing.** `prefetch()` is decoded at `src/rex3.rs:425` and used only to print
` PREFETCH` in the register log (`src/rex3.rs:263`). No path branches on it.

That is currently harmless rather than correct, and it is worth knowing which:

* The priming write puts `0xdeadbeef` into `ctx.hostrw` via the `REX3_HOSTRW0`
  arm of `process_register`, then the GO on the same entry runs the primitive,
  which overwrites `hostrw` with the first real pixel. The trash value is
  clobbered before anything can read it.
* So our read pipeline behaves as if prefetch were always on, and PIO readback
  comes out right for the wrong reason.

The latent risk is a guest that sets `OPCODE=read` **without** PREFETCH and
expects the un-primed, one-behind sequence. The spec says the driver "must" set
it, so that path is probably never exercised by IRIX — but nothing in IRIS would
diverge gracefully if it were: we would return the first pixel where hardware
returns the previous latch.

## Other sources

MAME's `newport.cpp` decodes the DRAWMODE1 bits but models no prefetch
behaviour — it has no pipeline latch to prime, so there is nothing to compare
against. Not a useful reference here.

## Related

- `rules/rex3/hostrw-batching-design.md` — the batch token/payload protocol and
  why DMA reads invert the PIO ordering.
- `rules/rex3/hostrw-tests-must-use-partial-word-rows.md` — the STOPONX/STOPONY
  shapes and why test widths must not be multiples of 8.
- rex3.pdf §3.10 *Framebuffer PIO and DMA*, §3.11 *FIFO Management* (the
  GFXBUSY/GFIFOLEVEL idle check before a read series).
