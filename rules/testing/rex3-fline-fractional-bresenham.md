# REX3 F_LINE fractional Bresenham — fixes and open A_LINE follow-up

## Fixed: base `d` formula was I_LINE's, not F_LINE's — and the fix lives in `setup()`, not `draw_line_bresenham`

`fline_apply_fract` (rex3.rs) received `*d` already computed by `setup()`
using the I_LINE decision-variable formula (`incr1 - major` = `2*minor -
major`) and only added the fractional cross-term on top. Per the REX3
hardware manual (`ignore/docs/rex/rex3_pdf.md` §3.6.1.2, "Code for aliased
line with fractional endpoints"): F_LINE's base `d` is `3*minor - 2*major`,
not I_LINE's `2*minor - major`. The two formulas differ by exactly `(minor -
major)`. Missing this correction let the fractional term get added to the
wrong baseline, which could flip `d`'s sign on the very first Bresenham step
for near-degenerate lines (small minor-axis component), producing a spurious
extra step that the line never recovered from — confirmed via a real
interpreter run that silently drew a wrong pixel row and, worse, tripped
`debug_assert!` in the REX3-Processor background thread. Because that thread
panic is silent (Rust background-thread panics don't crash the process) and
`wait_idle()` spins forever waiting for a `gfxbusy` flag that a dead thread
will never clear, this bug manifested as a **hang**, not a visible failure —
worth remembering next time a REX3 test mysteriously never returns.

Independently confirmed against MAME's `newport.cpp::do_fline` (same `d =
3*dy - 2*dx` formula) — but see the MAME caveat below before trusting that
file for anything beyond the `d` formula.

**Architectural note — where the fix had to live:** the fractional
correction was originally applied inside `fline_apply_fract`, called from
`draw_line_bresenham` — the **interpreter's own draw routine**. That's the
wrong place: `execute_go` calls `self.setup(ctx)` (octant/incr1/incr2/base-d
derivation) unconditionally on `DOSETUP`, *before* dispatching to either the
interpreter or a JIT-compiled shader. The JIT-compiled entry point never
calls back into `draw_line_bresenham`/`fline_apply_fract` — it only reads
whatever `bresd`/`bresoctinc1`/`bresrndinc2`/`xstart`/`ystart` `setup()`
already wrote into `ctx` and replays the walk. So applying the correction
only in `draw_line_bresenham` left the JIT silently drawing plain-I_LINE
trajectories for F_LINE (and A_LINE) — confirmed via `emit_draw_iline`
(`rex3_jit/compiler.rs`) having zero references to fractional nibbles/`xf`/
`yf`/`>> 7`/`dosetup`. **The fix now lives in `setup()` itself** (gated on
`ctx.drawmode0.adrmode()` being F_LINE/A_LINE): it calls
`fline_apply_fract`, then writes the corrected `x`/`y` back into
`ctx.xstart`/`ctx.ystart` and the corrected `d` into `ctx.bresd`. This makes
the JIT correct **for free** — no changes needed in `rex3_jit/compiler.rs`
at all, since the JIT just reads the already-fixed context state.
`draw_line_bresenham`'s old `fline_apply_fract` call was removed (would have
double-applied the correction). The shared octant table used by both
`setup()` and `draw_line_bresenham` is now `REX3_BRES_OCTANTS` (module-level
const, rex3.rs) rather than a local `const BRES` duplicated in each
function — verify both call sites still reference the same table if editing
this again.

Verify this "JIT reads ctx state setup() wrote, never re-derives it"
principle before assuming any other REX3 primitive's JIT/interpreter split
works differently — it generalizes to Bresenham state, pattern bits
(`pat_bit`/`zpat_bit`, see below), and shade DDA, all of which are `ctx`
fields populated on the interpreter side of `execute_go` and merely
*consumed* by JIT-compiled shaders.

## Fixed: `debug_assert!` exact-endpoint check was I_LINE-only, wrongly applied to F_LINE

`draw_line_bresenham`'s per-pixel loop asserted the walk landed exactly on
`(x2,y2)` on the last step. That's a valid invariant for I_LINE (integer
endpoints → integer Bresenham is provably exact), but not for F_LINE: a
fractional start biases the initial error term, but the loop still steps by
whole pixels along the major axis. When the fractional remainder doesn't
divide evenly, the walk lands on the closest integer approximation to the
true line at the endpoint's major-axis column/row — not necessarily the
literal requested minor-axis coordinate. This is expected fractional-DDA/
Bresenham behavior, not a bug (confirmed via web search: fractional Bresenham
variants guarantee landing in the endpoint's integer pixel column/row, not
its exact sub-pixel position — an exact match would require re-checking a
fixed-point DDA at every step instead of just the initial bias). Fix: gate
the assert on `fract` (only check for I_LINE, i.e. `!fract`).

**Do not "fix" F_LINE by trying to force the Bresenham walk to land exactly
on the target endpoint** — that was tried first (see "abandoned approaches"
below) and is the wrong invariant entirely.

## Test coordinate gotcha: XSTARTF/YSTARTF/XENDF/YENDF cannot hold biased coordinates

REX3 internally represents all coordinates with `REX3_COORD_BIAS` (4096)
baked in (`XYSTARTI`/`XYENDI`'s MMIO handler adds it). The F-suffixed
"GL fast path" registers (`XSTARTF` etc.) are documented in
`rex3_pdf.md` table 7 as "GL version of XSTART, **(zeros 4 msbs)**" — a
genuinely narrower register than plain `XSTART`. Their write handler
(`from12_4_7(val) = val & 0x007fff80`) only keeps 16 value bits and silently
truncates any biased coordinate (bias=4096 pushes the value past the mask).
Writing through `XSTARTF` with a biased value produces a wrong, unbiased
`ctx.xstart`; combined with a properly-biased `XYENDI` on the other endpoint,
`dx`/`dy` come out enormously wrong (confirmed: this alone made a test
compute octant 5 as octant 4, drawing a diagonal instead of a horizontal
line). **Test helpers that need to set fractional coordinates must use the
plain `XSTART`/`YSTART`/`XEND`/`YEND` registers** (`from16_4_7`, which
sign-extends via `Rex3RegisterOps::rexset` and correctly round-trips biased
values), not the F-suffixed aliases — see `write_xstartf`/`write_ystartf`/
`write_xendf`/`write_yendf` in `rex3_tests.rs` for the working pattern
(despite the `f` in their names, for historical reasons — they write to
`REX3_XSTART` etc., not `REX3_XSTARTF`).

## Abandoned approaches (don't re-try these)

- **Tuning `major`/`minor` rounding** (ceil, round-half-up, round-half-to-
  even) to force the walk onto the exact endpoint. Each variant fixed the
  1-2 cases it was tuned against and broke ~14-51% of a broader sweep
  (8064 cases across varying centers/radii/angles/fractions). This is
  curve-fitting, not a real fix — the underlying assumption (exact endpoint
  match) was wrong, not the rounding rule.
- **Porting MAME's `do_fline` verbatim.** MAME's `dx_i`/`dy_i` loop-count
  computation (`newport.cpp` around line 2768) has its own unit-scale bug:
  `x10 = x1 & ~0xf` masks off the fraction bits but keeps the 16.4
  fixed-point *scale*, instead of actually converting to integer pixels
  (`x1 >> 4`). This makes MAME's loop count ~16× too large if taken
  literally. MAME is useful for confirming the `d` formula shape, not as a
  drop-in reference for the full algorithm.

## Fixed: `compare_jit_interp` test harness bug — pat_bit/zpat_bit leaked across its own two internal runs

While chasing what looked like a JIT/interpreter parity bug in
`jit_lspattern_span_rgb24` (unrelated to F_LINE — found while running the
full `--features rex-jit` suite to verify the F_LINE fix didn't regress
anything), found a real bug in the **test helper**, not the emulator.

`compare_jit_interp` (`rex3_tests.rs`, `mod jit_tests`) runs a GO **twice**
against the same `rex_jit` instance: once to trigger JIT compilation (which
executes via the interpreter fallback while compilation happens
asynchronously), and once more after compilation completes, to actually
capture the JIT's output for comparison. For any `dm0` **without DOSETUP
set** (a legitimate, common case — e.g. `DM0_DRAW_BLOCK` has no DOSETUP
bit), `ctx.pat_bit`/`ctx.zpat_bit` never get reset (`execute_go` only resets
them `if ctx.drawmode0.dosetup()`). Since `pat_bit`/`zpat_bit` are pure
internal `ctx` state with no MMIO register mapping, neither `rex3init()` nor
the test's `setup` closure (which only issues `reg()`/register writes) can
reset them between the two internal runs. Result: the first "trigger
compile" GO's interpreter-fallback draw advances `pat_bit` as a side effect,
and the second "real" JIT-dispatched GO inherits that *already-advanced*
pattern position — while the separate `rex_interp` comparison instance
(which only ever runs the GO once) starts from a pristine `pat_bit`. The two
runs were never comparing equivalent starting states for any
LSPATTERN-using, non-DOSETUP `dm0`.

**Fix:** in `compare_jit_interp`, explicitly zero `ctx.pat_bit`/`ctx.zpat_bit`
(direct unsafe field access, matching the existing pattern already used to
read `ctx.clipmode` in the same function) after the "trigger compile" GO and
before the real comparison run. `jit_lspattern_span_rgb24` now passes.

## Fixed: 3 more pre-existing `--features rex-jit` failures, unrelated to F_LINE

Confirmed via `git stash` (running the full `cargo test --lib --features
rex-jit` suite against unmodified `main`) that these 3 already failed
independent of any F_LINE or `pat_bit` work. All three are now fixed —
distinct root causes, listed separately since none of them are really
"F_LINE" bugs; they just surfaced while verifying the F_LINE change didn't
regress the wider `rex-jit` suite.

- **`jit_fastclear_rgb24`** (`rex3_simd.rs`): `try_fastclear_block`,
  `try_src_block_rgb`, and `try_src_span_rgb` all compute their bounding box
  via `(ctx.xstart >> 11).clamp(0, REX3_SCREEN_WIDTH - 1)` — but
  `ctx.xstart >> 11` is a **biased** coordinate (REX3_COORD_BIAS = 4096
  baked in, same convention `calculate_fb_address` expects and un-biases
  internally), while `[0, REX3_SCREEN_WIDTH-1]` is an **unbiased** screen
  range. Since every real on-screen coordinate is `>= 4096`, `clamp` always
  collapsed it down to `REX3_SCREEN_WIDTH-1` (or `REX3_SCREEN_HEIGHT-1`)
  regardless of the actual requested region — these three fast paths were
  unconditionally processing a single degenerate off-screen pixel instead of
  the real bounding box, for every invocation, always. Fix: clamp against
  `[REX3_COORD_BIAS, REX3_COORD_BIAS + REX3_SCREEN_WIDTH - 1]` (new
  `REX3_BIASED_X_MIN/MAX`/`REX3_BIASED_Y_MIN/MAX` consts) instead. Per-pixel
  bounds rejection still happens correctly inside `calculate_fb_address` —
  these clamps only exist to cap the loop's iteration range before that, not
  to do the real bounds check. Likely invisible in practice because real
  IRIX/JIT usage rarely falls into these specific SIMD fast-path gate
  conditions (SRC logicop, no blend/pattern/host/shade) via the interpreter
  fallback — the JIT handles most solid-fill hot paths instead.

- **`jit_lspattern_lsopaque_block_ci8`** and **`jit_triangle_exact_mode`**
  (`rex3_jit/compiler.rs`, both `emit_shader` and `emit_draw_iline` — the
  logic is duplicated in each): `f2d0bff` flipped the interpreter's pattern
  cursor advance from decrement to increment (`advance_zpat`/`advance_lspat`
  in rex3.rs now both do `wrapping_add(1) & 31`; a stale doc comment at
  rex3.rs ~2090 still said "zpat_bit = (zpat_bit - 1) & 31" — also fixed),
  but the JIT's copies of this logic were never updated and still did
  `pat_bit32 - 1` / `zpat_bit_v - 1`. Beyond the direction flip, LSPATTERN
  has a second, more serious gap: on wrap (cursor reaches the pattern's end),
  the interpreter's `advance_lspat` does `ctx.lspattern =
  ctx.lspattern.rotate_left(1)` (recirculation) — the JIT never did this at
  all, anywhere (confirmed zero `rotate_left`/`rotl` references in
  `compiler.rs` before this fix). Fixed both: corrected the `+1` direction
  for `zpat_bit`/`pat_bit`, and added `lspattern` rotation-on-wrap
  (`(v << 1) | (v >> 31)`, implemented manually rather than via a Cranelift
  `rotl` op to avoid an API-availability assumption). The rotation fix
  required threading a new `lspattern`/`new_lspattern` `Value` through the
  same loop-carried plumbing `pat_bit`/`lsmode` already use in both
  functions: `loop_header`'s Cranelift block parameters (order-sensitive —
  added right after the existing `lsmode` param), the initial
  `jump(loop_header, &init_args)` call, the block-param extraction at loop
  entry, `emit_writeback`'s signature (`emit_shader` only — `emit_draw_iline`
  writes back via direct `st32e!` instead), and every `back_args`/
  `make_back_args!` construction + `jump(loop_header, ...)` site (2 functions
  × ~4-5 sites each). Row-boundary resets (`pat_bit_reset = 31`) do **not**
  reset `lspattern` itself — only the cursor resets per row; the pattern
  data only changes via `dosetup` or wrap-recirculation, matching the
  interpreter's `ctx.pat_bit = 31` (not `ctx.lspattern = ...`) at row ends.

## Still open: A_LINE

A_LINE tests were removed from this pass (out of scope) — `draw_aline`'s
AWEIGHT-LUT endpoint-suppression logic (rex3.rs `draw_aline`) and the test
helper `aline_endpoint_skip`/`DM0_DRAW_ALINE`/`DM0_ENDPTFILTER` in
`rex3_tests.rs` are kept (marked `#[allow(dead_code)]`) for a future
session. Whatever fix A_LINE needs should build on the same base-`d`-formula
fix here (A_LINE reuses `fline_apply_fract` via `draw_line_bresenham(ctx,
true, ...)`), plus its own AWEIGHT lookup — start there rather than
re-deriving the Bresenham math again.
