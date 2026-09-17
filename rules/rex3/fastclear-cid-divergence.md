# FASTCLEAR + CID checking: three code paths, two of which ignored CID — 2026-09-15

**Status: FIXED.** `rex3_simd` deleted; JIT clears the FASTCLEAR bit when CID checking is
on; regression test `jit_fastclear_with_cid_checking_matches_interp` added and verified to
fail without the fix (interpreter `0x605040` vs JIT `0xabcdef`).

Kept because the *shape* of the bug is worth remembering: two independent code paths
silently agreed on wrong behaviour, so a comparison test could not see it, and the
first diagnosis was wrong in a way that only measurement caught.

## What the spec says

`rex3.pdf` says it three times:

- DRAWMODE1 bit 17: "**FASTCLEAR** — Enables fast-clear write mode **when CID checking
  disabled** (CLIPMODE CIDMATCH = 0xF). Valid with DRAW SPAN/BLOCK only."
- §3.5.5: "No support for any per pixel operations, such as shade, stipple, dither,
  blend. Flat fill only, via value previously written by host into the COLORVRAM
  register. […] **CID checking is not allowed for this drawing mode.**"
- Programming notes: "**REX3 will disable FASTCLEAR mode if CID checking is enabled.**
  Therefore host must setup fast clear operation by writing COLORVRAM and also setting up
  DRAWMODE and COLORI (for example) register. This is necessary because GL will not know
  if window system invokes CID check."

So with CID checking on, hardware ignores FASTCLEAR and the draw proceeds as an ordinary
one — which is why the host is told to set up COLORI as well.

## What IRIS actually does

There are **three** fastclear implementations, and they do not agree:

| Path | CID honoured? |
|---|---|
| `execute_go` processor select ([rex3.rs:3862](../../src/rex3.rs)) — `fastclear() && no_cid && no_host` | ✅ yes |
| `rex3_simd::try_fastclear_block` (`src/rex3_simd.rs`, since deleted) | ❌ **no** |
| rex-jit `emit_pixel_write` ([compiler.rs:543](../../src/rex3_jit/compiler.rs)) — `dm1.fastclear() && !is_hostw` | ❌ **no** |

`try_fastclear_block` runs as a pre-loop bailout at the top of `draw_block`
([rex3.rs:1772](../../src/rex3.rs)), *before* `execute_go` selects a pixel processor. It
gates on fastclear, patterns, host mode and BLOCK adrmode — but never on CIDMATCH — then
fills the whole block with `fastclear_color` and returns `true`.

**Consequence: the `no_cid` term in `execute_go` is unreachable for BLOCK draws.** The
SIMD bailout has already handled them. It only has effect for SPAN, where no such bailout
runs (`try_src_span_rgb`'s call site is commented out).

Measured with a 16x16 fastclear BLOCK, `CIDMATCH=0x1`, COLORVRAM=0xABCDEF: interpreter
and JIT both write `0xabcdef` to every pixel. The interpreter *does* select
`process_pixel_draw` — the gate works — but the pixel loop never runs, because
`try_fastclear_block` returned true before it.

## What I got wrong

The first version of this note claimed a clean interpreter-vs-JIT divergence: interpreter
correct, JIT wrong. That came from reading `execute_go`'s processor selection and stopping
there. A test built on that prediction **passed**, which is what exposed the error.

Two lessons worth keeping:

1. **The pre-loop bailouts in `draw_block` bypass processor selection entirely.** Any
   reasoning about "which pixel path runs" that starts at `execute_go` is incomplete
   while `rex3_simd` exists.
2. The test that passed was not useless — it falsified the hypothesis. But it was written
   to confirm an expected failure, and had it been written with an `assert_ne!` it would
   have "passed" for the wrong reason. Probing what each engine actually wrote is what
   settled it.

A second error in that test, found while debugging: `COLORRED` was set to `0x00112233` as
though it were packed RGB24, but in RGB mode `get_colori` reads it as an o12.11 DDA
register (`>> 11`, then clamp), so the general path would have produced black, not the
distinct colour intended. Same class of bug as the vacuous-test fixes in `9340810`.

## The fix that landed

1. **`rex3_simd.rs` deleted entirely** (module, both call sites in `draw_block`, the
   commented-out span call, and the `simd_fill_rows` counter and its `perf snapshot`
   line). It was never SIMD — a prior session confirmed zero vector instructions by
   objdump — just the interpreter loop with mode checks hoisted, and its gates shadowed
   the interpreter's own. Removing it changed no test result, confirming it was pure
   redundancy. The interpreter immediately became spec-correct here, because
   `execute_go`'s `no_cid` term finally became reachable for BLOCK draws.
2. **JIT clears the FASTCLEAR bit when `cidmatch != 0xF`**, in `compile_shader` right
   after `normalize_dm1` — one edit covering all four `emit_pixel_write` call sites,
   rather than threading `cidmatch` through each.
3. **Regression test** `jit_fastclear_with_cid_checking_matches_interp`, verified to fail
   with the fix disabled: interpreter `0x605040`, JIT `0xabcdef`. Colour registers are
   written as `component << 11` because `get_colori` reads them as o12.11 DDA values —
   packed RGB24 would clamp to black and make the test vacuous.

Suite: 546/0 with `rex-jit`, 498/0 without.

## Why this was hard to see, and what to take from it

Two independent implementations were wrong **in the same direction**, so the
JIT-vs-interpreter comparison harness — the tool built precisely to catch engine
disagreement — could not detect it. Agreement between engines is evidence only when at
least one of them is independently known to be right.

The general lesson: a pre-loop bailout that intercepts a primitive before dispatch can
silently override correctness logic downstream of it, and nothing type-checks that its
gate conditions match the ones it is bypassing.
