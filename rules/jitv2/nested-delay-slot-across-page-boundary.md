# Nested delay slot across a page boundary (jitv2 codegen)

## Symptom

Panic while compiling, from `src/jitv2/codegen.rs`:

```
index out of bounds: the len is 1024 but the index is 1024
```

Hit live running **SoftWindows 95** under IRIX (which JITs its own code, so it
emits instruction layouts ordinary compilers rarely produce).

## Shape that triggers it

A branch/jump at word **1022** whose delay slot at word **1023 (0xFFC)** is
*itself* a branch/jump/JR. That nested transfer's own mandatory delay slot is
`word + 1 == 1024` — the **next physical page**, one past the
`[CompiledInstr; ENTRIES_PER_PAGE]` array.

```
0x...FFC-4  BEQ  ...      <- head, slot is next word
0x...FFC    BEQ  ...      <- nested branch; ITS slot is on the next page
0x(next)000 ...           <- not in this page's instrs[]
```

## Why the analyzer let it through

The analyzer deliberately permits it. `analyzer::visit_slot`'s
`OFFSET_0XFFC_WORD` arm marks word 1023 visited with
`taken_exit: StopReason::ForeignPageSlot` and **does not** recurse into word
1024 — the exact mirror of what `visit`'s `is_0xffc_branch` does for a head
instruction. So the region compiles, and codegen is expected to handle the
foreign slot.

`emit_branch_or_jump` (head level) *did* handle it, via its `foreign_page_slot`
flag. The two **nested** emitters did not: `emit_nested_branch_slot` and
`emit_nested_regjump_slot` both indexed `instrs[word + 1]` unconditionally.

## The fix

Nested emitters now mirror the head-level handling. Key point, easy to get
wrong: **the slot is not skipped and is not a NOP.** It still executes — just
on the next dispatch, after the next page is entered. So the exit must *arm the
pending transfer*, exactly like the interpreter's `branch_delay`:

- `core.delay_slot_target = <this branch's real target>`
- `core.in_delay_slot = 1`
- `core.pc = word + 1` (the slot itself, next page's word 0)
- return `EXEC_COMPLETE`, **no** `jit_trigger` (the transfer hasn't retired yet)

That is `emit_foreign_page_slot_exit`. Handled up front in
`emit_nested_branch_slot`, before the `emit_inner_slot` closure exists, so no
path can recurse into a foreign slot with bogus raw bits.

Per outcome (`emit_nested_foreign_page_slot_branch`):

| nested branch                | behaviour                                                        |
|------------------------------|------------------------------------------------------------------|
| `Always` (J/JAL)             | arm the jump target                                              |
| non-annulling conditional    | **both** arms arm the slot — taken with the branch target, not-taken with `word + 2` (§6.1.4: the slot runs exactly once either way; `handle_branch_not_taken`'s `branch_delay(pc + 8)`) |
| annulling Likely, taken      | arm the branch target as usual                                   |
| annulling Likely, not taken  | slot annulled — nothing new to arm; plain `pc += 8` (`handle_branch_likely_skip`) |
| JR/JALR                      | arm the register-derived target (single unconditional outcome)   |

## The second bug, found by comparing against the interpreter

The annulled-Likely-not-taken arm is the **only** exit in this family that
leaves `in_delay_slot` set without writing `delay_slot_target` of its own. The
*outer* branch's transfer is still pending there (the interpreter armed it one
dispatch earlier; `handle_branch_likely_skip` only does `pc += 8` and touches
neither field).

`emit_slot_semantics` stores `delay_slot_target` **only under
`jitv2_lockstep`**, justified by a comment saying the non-lockstep path "never
reads delay_slot_target for an inlined slot" — true only while every inlined
slot ended by writing a final `core.pc` directly. This path broke that
assumption, so the outer target is now threaded down
(`outer_delay_slot_target`) and stored on that arm. Without it the JIT left
`delay_slot_target = 0` where the interpreter had the real target, and the next
dispatch would retire the pending transfer to pc 0.

**This was only caught because the test compared against the interpreter rather
than asserting hand-derived constants** — a hand-written assertion had guessed
`in_delay_slot == false` on the annulled path, which is wrong for the nested
case (the outer branch's flag is still live).

## The `is_inlinable` helper (follow-up audit)

The bug was possible because "can I inline this word?" was open-coded at five
sites, four of which tested **only** `word < ENTRIES_PER_PAGE`. That is the
weaker test. All three ways the answer can be "no" collapse to one
analyzer-side fact — the walker never set `visited`:

| reason | where | what the analyzer records |
|---|---|---|
| off the page | `visit`/`visit_slot` early-return on `offset >= WORDS_PER_PAGE` | `StopReason::ForeignPageSlot` (the 0xFFC hazard) |
| forbidden | `Classify::Excluded` (fallback off) / `RegionBoundary` | `StopReason::Excluded` on the caller's edge |
| over budget | `budget.remaining == 0` in `visit` | `StopReason::Truncated` on the caller |

And the consequence is identical in all three: the word is not part of the
compiled unit, so its semantics must be **deferred to the next dispatch**, not
emitted here and not skipped.

`codegen::is_inlinable(instrs, word)` now names this:

```rust
(word as usize) < ENTRIES_PER_PAGE && instrs[word as usize].visited
```

Why `visited` matters and not just bounds:
`compile_region_uncommitted`'s upfront rejection loop iterates `instrs_linear`
— **visited words only** — so an unvisited word has never been checked for
having a semantics emitter at all. Inlining one is exactly as unsound as
indexing past the array; it just fails later and less loudly.

Converted sites: `emit_regjump`, `emit_branch_or_jump` (its `foreign_page_slot`
flag), `emit_nested_branch_slot`, `emit_nested_regjump_slot`, and
`try_emit_fused_lui`. Fusion was the only site that already paired both tests by
hand — the helper is that pairing, named once.

Not a second live bug: the analyzer declines a branch outright
(`visit` returns `false`) when its slot chain can't be walked, so today
`visited` holds wherever bounds hold. It is defense-in-depth against exactly
the invariant that broke for word 1024. `is_inlinable_rejects_out_of_bounds_and_unvisited_alike`
(in `codegen.rs`) pins both halves.

`emit_account_for_cycles` and `try_emit_fused_nop_slot` still index `instrs`
unguarded, which is fine — every path to them is behind an `is_inlinable` gate.

## Tests

`src/jitv2/equiv_test.rs`, via the `check_nested_foreign_page_slot` helper —
runs both engines over the same two-word region and asserts the JIT matches the
interpreter's `pc` / `in_delay_slot` / `delay_slot_target`:

- `nested_branch_at_last_word_with_foreign_page_slot_matches_interpreter`
- `nested_jump_at_last_word_with_foreign_page_slot_matches_interpreter`
- `nested_regjump_at_last_word_with_foreign_page_slot_matches_interpreter`
- `nested_branch_at_last_word_not_taken_arms_foreign_slot_like_interpreter`
- `nested_likely_at_last_word_not_taken_annuls_foreign_slot_like_interpreter`
- `nested_likely_at_last_word_taken_arms_foreign_slot_like_interpreter`

The test page base is deliberately `0x...9FC0_F000` so an OR-vs-ADD slip in the
next-page address can't hide (same rationale as
`sequential_at_last_word_falls_through_to_next_page_not_same_page`).

## Note

`cargo test --features jitv2,jitv2_lockstep --lib` SIGSEGVs on the full suite —
**pre-existing**, reproduces on stock `main` with these changes stashed.
Unrelated to this fix; the tests above pass under both feature sets when run
filtered.
