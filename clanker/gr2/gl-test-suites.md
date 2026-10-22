# GR2: the GL test suites (pixtest, blendtest, glstate, vissweep)

The four board-neutral suites in `test/gltest` run on XZ unchanged. They
were written for IMPACT, and these points matter on GR2. Results as of
2026-10-13 (Indy XZ, IRIX 6.5.22):

| Suite | Result |
|---|---|
| pixtest | 366 pass, 0 fail, 7 XFAIL |
| pixtest --ci 8 | 48 pass, 1 XFAIL |
| pixtest --ci 12 | 48 pass, 1 XFAIL |
| glstate | 396 pass, 0 fail, 11 XFAIL |
| blendtest, every RGB visual | 0 fail |
| vissweep | 7 of 7 visuals pass |

The hardware facts these runs settled are in ignore/gr2: HQ2.h (pixel
section, 0x0A7, 0x0BA, 0x70A0, READ_RECT) and GR2.h (blend equations).

## Visuals

- XZ has no alpha planes, no pbuffers, and 8,8,8 only single-buffered.
  Double-buffered RGB is 4,4,4 (12-bit), 3,3,2 is single-buffered, and
  1,2,1 is double-buffered (MakeCurrent 0).
- Depth/stencil layouts are z24 s0 and z20 s4.
- Each suite falls back to a visual XZ has: pixtest and blendtest to
  single-buffered RGB8, glstate to a visual with stencil.
- vissweep needs XZ's lists:
  `COLORS="1,2,1 3,3,2 4,4,4 8,8,8" ZS="none z24 s4z20" sh vissweep.sh`.
- `blendtest --visual 0x27` (and the others) checks the low-depth visuals.
  It models storage as each channel's top bits and widening by bit
  replication.

## GR2 XFAIL lists

The tests detect GR2 from GL_RENDERER (it starts with "GR" or "GU"). Each
xfail entry carries a `gr2_only` flag. All of them are libGLcore EXPRESS
departures, checked in its code. Do not "fix" them in the emulator:

- **Subtract and reverse-subtract blending is applied twice** (software
  blend: `__glBlendSpan`, then `Store_B`).
- **Stencil reads return Z's low bits.** On z20 s4 the stencil sits above
  Z, but libGLcore's stencil `Fetch` masks with 0xF.
- **PACK_SWAP_BYTES is ignored on Pack1 reads.**
- **glGetTexImage returns LUMINANCE as R+G+B.**
- **The raster distance is returned as eye z.**
- **Index mask and clear index are dropped in RGBA mode.**
- **A right-shifted UNSIGNED_BYTE index is rounded.**

On GR2 only the current and raster colour, index and normal queries reach
the board (`__glExpDoGet`). Everything else in glstate tests libGLcore
itself.

## Gotchas

- IRIX `grep` has no `\|`. Filter with a plain word, or use `egrep`.
- GL software paths make it look like the board is wrong when it isn't.
  Look at the fragment colour libGLcore sends (trace `GL_FRAGMENT`) before
  touching the HLE. Colour-index fragments carry the index (+0.5) as a
  float in the first word.
- The guest's /tmp is cleared on boot: stage the tests again after every
  restart.
