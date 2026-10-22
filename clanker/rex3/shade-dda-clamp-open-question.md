# Shade DDA clamping: clamped accumulator, or clamped output? — OPEN, 2026-09-15

**Unresolved. Needs a real XL Indy to settle.** Do not "optimise" the shade path
on the assumption that the loop-carried DDA value is already clamped until this
is answered — a JIT change resting on exactly that assumption is parked in a
stash for this reason (see the bottom of this note).

## The question

`rex3.pdf` (line ~6834 of `pdftotext` output) says:

> DDA values of R,G,B,A are clamped each iteration before sending down the
> pipeline. As each of these components has an additional, overflow bit at the
> DDA, a normalized range of [-.5 to +1.5) is handled prior to clamping.

and immediately after:

> Color index DDA values can be clamped to desired range by setting the
> DRAWMODE0 bit ENCICLAMP.

Two readings, and they are not equivalent:

**(A) Clamped accumulator.** The clamp is applied to the DDA register itself
each iteration, so the next `+= slope` starts from the clamped value. This is
what IRIS currently implements: `iterate_shade_rgb_clamp` (rex3.rs) writes the
clamped result back into `ctx.colorred/grn/blue/alpha`.

**(B) Clamped output only.** The accumulator keeps running unclamped and the
clamp happens where the value is handed to the rest of the pixel path — i.e.
when packing to RGBA8888. "Another turn of the interpolator" is arguably not
"down the pipeline", so the clamp would sit at the boundary, not on the state.

## Why it matters

The two agree for any monotonic ramp that stays in range, and for one that
saturates and never comes back. They diverge when a component **overshoots and
returns**:

- Under (A) the value is pinned at the clamp (0 or 0x7FFFF) and a negative slope
  walks back from the *clamped* value.
- Under (B) the accumulator sailed past, and a negative slope walks back from
  the true value, so the visible colour stays clamped for longer and then
  resumes at a different place.

Long Gouraud spans with steep slopes are where this shows. Nothing in the
current test suite distinguishes them: `jit_shade_ramp_255_to_0` ramps down from
255 and never exceeds range, and `jit_shade_span_saturate` saturates upward and
stays there.

## What the other implementations do

- **IRIS interpreter**: reading (A). RGB clamps every iteration
  unconditionally; ENCICLAMP consults only the CI paths
  (`iterate_shade_ci8_clamp` / `ci12`), matching "ENCICLAMP governs CI".
- **IRIS rex-jit**: same as the interpreter (`clamp_shade` on the loop-carried
  values), so the two engines agree with each other either way.
- **MAME `newport.cpp`** (`iterate_shade`, ~line 3212): gates **even the RGB
  clamp** on CICLAMP (DRAWMODE0 bit 21). With CICLAMP clear it does not clamp
  RGB at all. That contradicts the spec text above on its face, and contradicts
  IRIS. MAME is the reference for TLB conformance; it is evidently *not* a
  reference here.

So all three differ in some respect, and the spec sentence is ambiguous enough
to support two of them.

## How to settle it on hardware

Draw one long Gouraud span in RGB mode whose red component overshoots the top of
range and then comes back — e.g. start red near max, positive slope for the
first stretch, then re-issue with a negative slope (or use a slope large enough
that the 21.11 value wraps well past 0x7FFFF and back). Capture the span.

- Reading (A): red stays pinned at the ceiling and only starts descending after
  the *clamped* value minus accumulated slope drops below it.
- Reading (B): red stays at the ceiling longer, then reappears at the value the
  unclamped accumulator actually reached.

Compare against both IRIS engines. Also worth checking the same with CICLAMP set
and clear in RGB mode, which discriminates IRIS from MAME directly.

## Facts already established (no hardware needed)

Brute-forced over the full u32 domain, assuming reading (A):

- `clamp_shade` can **never** produce a value whose `(c >> 11) & 0x1FF` exceeds
  0xFF — 0 cases. Its own saturation constant `0x0007_FFFF` has `val == 0xFF`
  exactly. So the `val > 0xFF` arm of `clamp_color_component` is dead code on
  shade input, and the per-pixel colour extraction collapses to a plain
  shift-and-mask.
- On *raw* (unclamped) input the cheap form differs from the full clamp in
  2.68e9 cases, so under reading (B) the per-pixel clamp is load-bearing and
  cannot be reduced.

That is precisely why the answer changes the optimisation: under (A) the
per-pixel clamp is nearly free to remove; under (B) it is mandatory.

## The parked work

A JIT change implementing the (A)-only optimisation — entry-block pre-clamp plus
a reduced per-pixel extraction, gated on `dm1.rgbmode()` — is in
`git stash` (`WIP on main: 3f4aec5`). It passed 530 tests and looked like
roughly +24% on long-span Gouraud across two runs, but that measurement never
got a same-session baseline and one intervening sweep was thermally
contaminated, so the number is unconfirmed as well as the premise.
