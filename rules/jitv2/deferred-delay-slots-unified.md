# A branch whose delay slot can't be inlined: one path, not two

**2026-09-02.** Until this change, jitv2 had two unrelated answers to the same
question, and one of them was a leftover from before the answer was known.

## The question

A branch/jump's delay slot is architecturally indivisible from it (§6.1.4):
the slot always executes exactly once, so codegen inlines it into the
branch's own compiled unit. Sometimes it can't:

- the slot is on the **next physical page** (branch at offset 0xFFC), or
- the slot is an **`Excluded`** instruction (COP0/MTC0, `cache`, `eret`,
  `syscall`, `break`, LL/SC, BC1, CP2, any unimplemented opcode), which by
  definition has no native emitter and must run through the interpreter in
  *head* position, or
- the slot is otherwise unvisited.

## The two old answers

| slot | branch | mechanism |
|---|---|---|
| off-page (0xFFC) | **compiled**, slot deferred | `is_0xffc_branch` skips `visit_slot`; `taken_exit: ForeignPageSlot`; codegen's `emit_foreign_page_slot_exit` arms the pending transfer |
| excluded | **declined outright** | `visit_slot` returns `false`, branch never marked visited, falls out of the region |

The second is what the first used to do, before the foreign-page case was
worked out. The analyzer's own comment gave it away — *"a slot that can't
complete disqualifies the outermost branch exactly like an excluded slot
always did"* — describing inertia, not a reason.

`is_inlinable`'s doc comment had *already* asserted the unification: all
three ways a slot can fail to inline "collapse to the same analyzer-side fact
and the same codegen-side consequence: deferred to the next dispatch." The
analyzer just didn't act on it.

## The unified rule

**If the slot can't be inlined, compile the branch and hand the interpreter a
pending transfer** — arm `core.delay_slot_target`, set `core.in_delay_slot`,
land `core.pc` on the slot word, return `EXEC_COMPLETE`. The interpreter runs
the slot and retires the transfer. It does not care *why* the slot was
deferred.

The slot's address is derivable identically in both cases:
`emit_word_addr(ctx, word + 1)`. At word 1023 that is `vbase + 1024*4` =
`vbase + 0x1000` — the next page's word 0 — because it is an `iadd`, so the
carry into bit 12 just works. (An on-page excluded slot is the easier case:
no carry at all.) There is no address asymmetry between the two; that was
the one thing that made them *look* like different problems.

## What changed

- **analyzer `visit`**: a failed `visit_slot` now sets `deferred_slot`
  instead of `return false`. The branch is visited with
  `has_inline_slot = false` and both edges forced to
  `StopReason::ForeignPageSlot` (whose doc comment now says it covers both
  causes — the name is historical).
- **`is_inlinable` (codegen)**: now also rejects `is_fallback` heads.
  **Necessary, not cosmetic**: an `Excluded` word *can* be `visited` — as a
  fallback head admitted by some other path — which would otherwise make it
  look inlinable purely because the `visited` bit was set.
- **codegen otherwise unchanged.** It was already keyed on `is_inlinable`
  rather than on `word == 1023`, so the deferred case flows through the
  existing foreign-slot emitters untouched. That is the payoff: one predicate,
  one path.

## Payoff: simplification, not speed

Measured honestly, on the 300-page IRIX corpus: **22 branch-with-excluded-slot
occurrences across 300 pages**, and emitted instruction count identical
(480,780 before and after). The shape is rare, and where it occurs the region
often ended nearby anyway.

Do not expect a benchmark to move. The reason to have done it is that jitv2
is hairy enough already, and this removes a special case that existed only
because of the order things were figured out in. A delay slot at a page break
is genuinely plausible in real code; a weird excluded instruction in one
mostly is not.

## Tests

- `analyzer::tests::walk_excluded_delay_slot_defers_the_slot_and_still_compiles_the_branch`
  (rewritten — it previously asserted the old decline-the-branch contract).
- `equiv_test::tests::{branch_taken,branch_not_taken,jump}_with_excluded_delay_slot_defers_like_the_interpreter`
  — execution-level, checking the JIT reproduces the interpreter's
  `pc`/`in_delay_slot`/`delay_slot_target` exactly. Analyzer-level tests alone
  would only prove the region compiles, not that it *runs* right.

Related: [[inlined-slot-pc-bd-bracket]] — same area, same session; that change
is what made `emit_foreign_page_annulled_not_taken_exit`'s inherited
`in_delay_slot` an explicit parameter.
