# REX3 HOSTR row-wrap under STOPONY: two independent word-boundary bugs

CI flaked on `test_hostr_ci8_read_block_32bit` — `cargo test --workspace`
unifies features across the workspace, and `iris-gui/Cargo.toml` depends on
`iris` with `features = [..., "rex-jit"]`, so the `iris` lib's own plain
interpreter tests actually run with `rex-jit` compiled in even though nothing
in the test asks for it. That's why "REX JIT: started" appeared ahead of a
test that looks JIT-unrelated, and why the failure was flaky: a background
compile completing mid-sequence flips a later GO from interpreter to JIT.

Chasing it surfaced two separate, independently-introduced bugs in host-mode
(HOSTR/HOSTW) BLOCK draws under `DM0_STOPONXY` (the mode `DM0_READ_BLOCK`
actually uses) — one in the interpreter, one in the JIT, both about the same
invariant: **a row boundary in host mode is always a forced word boundary**,
full or not, because hardware sends exactly one word per GO.

## Interpreter bug (`src/rex3.rs`, `draw_block`)

The row-wrap branch (`x_end_reached` under `stopony`, more rows remaining)
fell through into the shared `stop_on_word && ctx.hostcnt == 0` check below
it. That check only fires for a word that reached exactly `host_count`
pixels — it never fires for a row whose width doesn't divide evenly into
`host_count` (e.g. CI8 packs 4px/word; a 17px-wide row's last word only gets
1 of 4 slots filled, so `hostcnt` is 3, never 0, at the row boundary). The
loop kept running into the next row's pixels, packing them into the *same*
still-open shifter meant for the previous row's leftover partial word,
corrupting several words downstream before the corruption happened to
resolve itself.

Fix: added an explicit `if stop_on_word && ctx.hostcnt > 0 { break; }` right
after the row-wrap's `y_end_reached` check, so *any* open word — complete or
not — stops the loop at the row boundary. The existing unconditional
`self.flush_host_pixel(ctx)` call after the loop (for READ) then pads and
sends the partial word correctly; it was already correct; it just never used
to run because the loop didn't reliably break there.

Existing tests didn't catch this because `test_hostr_ci8_partial_word` is
single-row (hits `y_end_reached`, which already broke correctly) and the
existing multirow test uses `!stopony`.

## JIT bug (`src/rex3_jit/compiler.rs`, `emit_shader`'s `y_cont_block`)

Cranelift codegen computes `host_xstop_v` — the "this word is done" x
threshold — **once, at shader-function entry**, from the x position at the
time `entry()` was called. That's correct only within the row the shader
started on. Under `stopony`, when a row wraps but the block isn't finished,
`y_cont_block` used to reset x/host state and `jump(loop_header, ...)` to
keep iterating *inside the same `entry()` call*, straight into the next row —
never calling `emit_store_hostrw` for the row that just ended, and now
comparing the new row's x positions against a stale threshold computed for
the old row. The threshold effectively stops firing, so the shader runs past
the word boundary the caller is waiting on, silently leaving `ctx.hostrw`
unwritten for the swallowed word(s).

Fix: `y_cont_block` in host mode (`is_hostw || is_hostr`) now treats the row
wrap itself as a forced word boundary — flush the (possibly padded) partial
word via the same `flushed_shifter`/`emit_store_hostrw` logic the
whole-block-done path (`y_done_block`) already uses, write back state
positioned at the new row's start, and `jump(loop_end, ...)` to return to the
caller instead of continuing internally. The caller's next GO/`entry()` call
recomputes `host_xstop_v` fresh for the new row. `mid_primitive` is left
`true` (unlike `y_done_block`, which clears it) since the block isn't
actually finished.

## Regression coverage

`src/rex3_tests.rs`, `jit_tests::hostr_stress` (+ 3 callers,
`jit_hostr_ci8_stress_large_block`/`_odd_width_block`/
`jit_hostr_rgb24_stress_large_block`): differential interp-vs-JIT tests over
larger multi-row blocks, including odd widths that force every row to end
mid-word. Uses the same JIT pre-warm pattern as
`jit_hostr_ci8_block_multirow_no_stopony` (throwaway pass + `wait_compiled`)
to make the "already compiled by the time it matters" race deterministic
instead of relying on background-thread timing.

**Gotcha in the fill pattern**: an early version of the stress test varied
only the high bits of each synthetic pixel value across x/y. CI8's 8-bit
depth mask discards everything above bit 7, so every pixel read back
identical regardless of position — a solid-fill test in disguise that hides
exactly this class of bug (see the "distinct per-pixel values" comment
already on the older multirow test — same trap, easy to reintroduce). Fixed
by varying the low byte directly (`0x1234_5600 | (1 + i % 0xFE)`).
