# jitv2_lockstep: region-ending words landed 1-2 pages off (double-page-jump)

## Symptom

`jitv2_lockstep` write-mismatch divergence at a freshly-entered region whose
entry word sits at a page's last word (offset 0xFFC), e.g.:

```
=== jitv2_lockstep DIVERGENCE at pc=...: write mismatch ===
  value jit=<correct-looking ra>  interp=<stale, unrelated value>
```

Also reproduced without any real divergence, as a plain equivalence-test
failure: a region-ending Sequential instruction's post-state `pc` landed one
or two pages past where it should (`sequential_pair_ending_at_0xffc_falls_
through_to_next_page`, `sequential_at_last_word_falls_through_to_next_page_
not_same_page`).

## Root cause

`emit_lockstep_compare_seq` (src/jitv2/codegen.rs) materialized a **throwaway**
`core.pc = word+1` purely so the lockstep compare hook had a `pc` field to
check — codegen doesn't otherwise keep `core.pc` live between straight-line
instructions. For a region-ending word, `word+1` crosses onto the next page.
An older version tried to "undo" that write (`core.pc = word`'s own address)
before falling through to `plain_fallthrough`'s `emit_bail`. But the shared
exit stub, `emit_exit_block_body`, derives its own exit address by **reloading
and re-masking whatever's currently in `core.pc`** and adding its own
`word_offset*4` — so:

1. The undo's own reload of `core.pc` happened *after* the throwaway
   `next_pc` write had already landed, so in some code paths it re-derived
   `vbase` from the *already-crossed* page instead of the original one.
2. Even when the undo worked, the general shape (materialize a pc, run
   compare, un-materialize before a *separate* function re-derives from
   the same mutable cell) is fragile — two different concerns (compare's
   "what should pc be" and the exit stub's "what page am I really on")
   sharing one memory cell with a manual dance in between.

This is exactly why an earlier attempt to extend lockstep coverage to
region-ending words caused a live-boot hang, and the original (reverted)
fix was to exclude region-ending words (and entry-words-that-are-also-
region-ending, e.g. an 0xFFC-only 1-instruction region) from lockstep
brackets entirely — silently unverified, not skipped-but-safe.

## Fix

Stop having the per-instruction compare bracket write `core.pc` for a
region-ending word at all. Instead:

- The lockstep compare call moved **into `emit_exit_block_body`** (the
  shared exit-to-interpreter block every `emit_bail` call routes through),
  right after it writes the real, final `core.pc` (`vbase | word_offset*4`)
  and before returning. This is the one point `core.pc` is genuinely final
  for *every* bail alike — preamble bails (nothing staged, compare is a
  no-op) and post-semantics region-ending bails (real value, real compare)
  both go through the same path with no special-casing.
- `emit_lockstep_compare_seq` (the ordinary in-region fallthrough bracket)
  and the `needs_foreign_slot_check`'s `plain_block` arm now skip their own
  materialize+compare entirely when the word is region-ending — that case is
  now handled solely by the exit stub described above. The materialize+
  compare they still do for plain in-region fallthrough is unaffected and
  correct as before (no page-crossing risk there by construction).
- Divergence-path state recovery (previously only ever restored `core.pc`,
  never `in_delay_slot`/`delay_slot_target` alongside it — the actual source
  of the earlier hang, per the reverted fix's own comment) moved to
  `lockstep_compare` (mips_exec.rs), which now restores all three atomically
  from `ls_before` (a `LockstepSnapshot`, already captured pre-instruction)
  plus a new `ls_delay_target_before: u64` field (mirroring the existing
  `ls_delay_target_expected`).
- With the hazard closed at its source, the `region_ending` exclusion on
  `ls_live`/`ls_bracket` was deleted — every JIT-emitted instruction, no
  exceptions (entry words, region-ending words, both at once), now gets a
  real interpreter-reference step + compare.

## Key invariant learned

`pc`, `in_delay_slot`, and `delay_slot_target` must always be read/written/
restored **together, as one unit** — never `pc` alone. A stale slot flag or
delay target left over from a divergent JIT run, combined with a "fixed" pc,
produces an internally inconsistent state the interpreter can misinterpret
on resume (e.g. landing on a plain instruction's address while still
believing it's mid-delay-slot).

## Related

- [[project_jitv2_lockstep_status]]
- `rules/jit/emit_absolute_pc_exit-in_delay_slot-followup.md` — a related,
  *not yet fixed* fragility found during this same investigation:
  `emit_absolute_pc_exit` never clears `core.in_delay_slot` itself, relying
  on every caller to have already cleared it first.
