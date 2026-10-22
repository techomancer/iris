# The raster JIT (`gr4-jit`) and its Cranelift pitfalls

`src/dev/mgras/rss_jit` compiles RSS/TE1 pixel pipelines. The rules below
are what it took to make shaders exact and fast. Read them before changing
an emitter.

## Exactness

- Every emitter mirrors an interpreter routine (named in its doc comment)
  and evaluates the same f64/f32 expressions in the same order. Setup is
  shared Rust code (`Rss::tri_setup`, `Rss::gl_line_setup`, `Te1::sampler`).
  Change the interpreter and the emitter together, then run
  `cargo test --release --features gr4-jit --lib rss_jit`.
- Rust and Cranelift differ in these places, and the emitters spell Rust out:
  - `f64::round` rounds halves away from zero, while `nearest` rounds them
    to even (`E::round`, or `round_nonneg` once a clamp has made the value
    non-negative or NaN).
  - `clamp` and `min` let NaN through or prefer the number, while
    `fmin`/`fmax` propagate NaN.
  - `as` conversions saturate (`fcvt_to_*_sat`).
- No libm and no float in the pixel path: everything a fragment computes
  is defined in `src/dev/mgras/fixed.rs`, and the JIT emits the same
  integer operations. Change one and you must change the other. Component and byte scaling come from tables
  filled by the interpreter's own divisions. Shaders make no calls.
- Booleans from `icmp`/`fcmp` are 0/1 bytes. `bnot`, `bor_not` and
  `band_not` on them give 0xFE/0xFF, which branch as true. That made every
  texture skip. Negate a comparison by flipping its condition code.

- Colour is 12.16 fixed point end to end (`src/dev/mgras/fixed.rs`):
  iterators, texels widened to 12 bits, 8-bit bilinear weights, the 8-bit
  mip fraction, texture environment, fog and blending, then a shift down
  on write. Depth is a z.12 integer DDA. Texture coordinates are integer
  too: S/W, T/W, 1/W planes at 2^32, a reciprocal unit (1024-entry seed
  table plus one Newton step, `fixed::recip`), s and t in Q31, texels in
  Q16 per level, and the level of detail in Q8 from a log2 table. Only
  the triangle edge walk is still f64.

## Speed

- Cranelift rematerialises ALU ops that take an immediate (`iadd`, `band`,
  `bor`, ... with an `iconst`, `opts/remat.isle`) at every use. Anything
  computed from them is dragged back into the pixel loop, and loop-invariant
  code motion never sees it. Plain block parameters do not help: with one
  incoming value, "remove constant phis" folds them away. `E::pin` makes
  per-row values opaque instead: a block entered from a never-false branch
  (pixel memory pointer != 0) with the values, and from an unreachable path
  with zeros.
- `can_move` loads are sunk to their uses, into the loops. Context
  invariants are loaded once, in the entry block, as plain readonly loads,
  and only those the key needs. Every invariant held through a loop costs
  the loop a register.
- A call from a shader clobbers every float register it holds (all XMM
  registers are caller-saved), so shaders make none. Textured triangles
  still spill: about 25 plane coefficients live in 16 XMM registers.
- Per-primitive cost matters as much as per-pixel cost (X draws many tiny
  primitives). The board keeps a persistent context (`Rss::jit_ctx`). Only
  writes to registers outside `is_data_reg` invalidate its cached,
  register-derived part, and each primitive memoises its last shader.

`GR4_JIT_DISASM=1` prints a shader's machine code;
`bench_interpreter_vs_jit` (ignored test) times the two engines.
