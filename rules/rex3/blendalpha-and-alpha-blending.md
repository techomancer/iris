# REX3 BLENDALPHA: what it selects, and why the `blast` billboard exposes it

## The bit

`DRAWMODE1` bit 27, `BLENDALPHA`. Register table (spec Table 11):

> Selects SFACTOR BF_SA source alpha: '1' = source alpha, '0' = 1.0.

§3.8 adds a qualifier whose trailing clause is load-bearing:

> When source multiplier is set to source alpha (SFACTOR=4), alpha component can
> be blended in two different ways depending on how BLENDALPHA ... is set. When
> BLENDALPHA is set to 0, the source multiplier for blending alpha is one instead
> of source alpha **and destination multiplier is defined by DFACTOR**. When
> BLENDALPHA is set to 1, alpha is blended the way defined by SFACTOR and DFACTOR.

So the substitution applies to **SFACTOR only**. DFACTOR keeps its own definition
and still evaluates against the *real* source alpha. Substituting into both
factors zeroes `BF_MSA` and discards the destination entirely — the spec never
says that, and it visibly breaks rendering (bright low-alpha texels get promoted
to full intensity, producing white speckle).

## REX has no destination alpha — anywhere

This is the key structural fact. The SFACTOR/DFACTOR tables (Tables 13/14) offer
only:

    000 BF_ZERO   001 BF_ONE
    010 BF_DC/BF_SC     011 BF_MDC/BF_MSC     (destination/source *colour*)
    100 BF_SA           101 BF_MSA            (source alpha)

There is **no `BF_DA`**. No blend factor reads destination alpha in any mode.
Only two framebuffer formats in Table 22 have an alpha field at all
(`RGBa-DB 3324+3324`, `RGBa-SB 444 8`), and those are alpha the host *writes*,
never a blend input.

Consequence: §3.8's "the source multiplier for blending alpha" cannot mean
"producing a destination alpha value". At 12bpp (`RGB-DB 444+444`) there is no
alpha plane to write one into either.

## Why this matters: the `blast` demo billboard

96.8% of that demo's draws (544,055 of 562,149) use one configuration:

    DRAWMODE0  DRAW SPAN  (+ DOSETUP SHADE on 79,303 of them), STOPONX=0
    DRAWMODE1  0x3165d011 / 0x3165d031  (DBLSRC is the only varying bit)
               PLANES=RGB DRAWDEPTH=12bpp RGBMODE=1 DITHER=1
               COMPARE=101 (src!=ref) ALPHAREF=0
               BLEND=1 SFACTOR=BF_SA DFACTOR=BF_MSA
               BLENDALPHA=0  LOGICOP=LO_SRC

`STOPONX=0` means **one pixel per GO** — `draw_span` breaks after a single
iteration. `xend-xstart` is the remaining distance to the polygon edge, not the
drawn extent; do not read it as a span width.

The app sets up textbook `GL_SRC_ALPHA / GL_ONE_MINUS_SRC_ALPHA` but with
BLENDALPHA=0. Taken literally that makes `BF_SA = 1.0`, so the source is never
attenuated — and the billboard's transparent surround (alpha 1..15 carrying
luma 7..27) writes at full intensity, quantising to a visible grey/pink haze at
12bpp instead of vanishing.

## What is NOT the cause (each checked against a real trace)

- **Alpha is not corrupted by interpolation.** `SLOPEALPHA` is written **0 times**
  in the entire log (SLOPERED/GRN/BLUE: 72 each), so `shade_add`'s
  `coloralpha += slopealpha` adds zero even on SHADE draws.
- **The clamp is not eating it.** `iterate_shade_rgb_clamp` clamps in place but is
  idempotent at these values.
- **AFUNCTION is correct and independent.** §3.3 says it compares source alpha
  "either from DDA or host, for bit ALPHAHOST=0,1" against ALPHAREF — the real
  alpha, unaffected by BLENDALPHA. Our `sa_src` is scoped inside `blend()`; the
  alpha test reads `raw_src >> 24` separately.
- **COMPARE is not an enum**, though ours behaves identically. Table 11 defines it
  as three OR'ed condition enables (bit2 `src>dest`, bit1 `src=dest`, bit0
  `src<dest`). Checked over all 8 values × zero and non-zero ALPHAREF: the enum
  form and the OR'ed form agree everywhere. `0b101` = `!=`.
- **The alpha test cannot clear the surround.** With ALPHAREF=0 and `!=`, only
  alpha *exactly* 0 is discarded — 0.7% of billboard draws. The surround arrives
  as alpha 1..15, 8.6% of draws.

## The `blast` demo artifact is NOT a REX3 bug

Replaying the trace into an image of exactly what the host wrote to the colour
registers (`tools/rex3_replay_png.py`) settles it:

* The texture is `orion.rgb` — an SGI RLE image, **256x256, z=3, NO ALPHA
  channel**. All transparency in this demo comes from texels being *black*, never
  from alpha. It is a glow sprite of the Orion Nebula, not a cutout.
* The source texture is **42.1% pure black**. The quad we render is only **1.6%
  black** — measured from the host's register writes, before any blending,
  dithering or quantisation of ours.
* REX3 has **no texture unit**. The ~395,000 COLORRED/GRN/BLUE writes are one per
  pixel: IRIX's software GL samples the texture on the CPU and hands REX3
  finished texels, one per single-pixel GO.

So the wrong colours are computed by MIPS code on the emulated CPU before REX3
sees them. The bug is upstream — in the software rasteriser's texture sampling
(and the triangular band along the quad's left edge looks like an edge-walk or
clipping error in that same code), not in the blender. Reproduced under the pure
interpreter, so it is not a CPU JIT codegen bug either; suspect FPU/conversion
behaviour (see the outstanding CVT equiv_test failures).

## MAME divergence

`newport.cpp` `blend_pixel()` gates the **RGB** source multiplier on bit 27
(SFACTOR case 4/5), so with BLENDALPHA=0 it uses `sbb = sb` (case 4, no
attenuation) or `sbb = 0` (case 5). Its DFACTOR path does not gate. MAME shows
the same billboard artifact, so agreeing with it is not evidence of correctness.
