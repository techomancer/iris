# The inlined delay slot's `core.pc` / `in_delay_slot` bracket looks dead. It is not.

**Status: a removal was attempted on 2026-09-01 and reverted the same day.**
This note exists so the next person who spots this "obviously redundant"
bracket does not repeat it.

`emit_slot_semantics` (src/jitv2/codegen.rs) wraps every **inlined** delay
slot in six memory operations:

```
in_delay_slot = 1
saved_pc = load core.pc          ; save
core.pc  = <slot's own address>
  ... the slot instruction's real semantics ...
in_delay_slot = 0                ; restore
core.pc  = saved_pc              ; restore
```

Measured on 300 real IRIX corpus pages (`zz_corpus_sizes`,
`IRIS_JIT_DISASM=1`, `opt_level=speed`, no `developer`), that is 3,574 `pc`
stores + 3,574 `in_delay_slot` stores + 1,741 `pc` loads out of 24,222 total
emitted stores — about 29% of all store traffic, all inline hot-path. It is
genuinely expensive, and it is genuinely necessary.

## Why it looks dead

Every argument below is *true* and still leads to the wrong conclusion:

- An in-region branch edge is a plain `jump` to the target's block
  (`emit_target_edge`) — it writes neither field.
- A region-leaving exit writes its own final `core.pc` (`emit_bail`,
  `emit_absolute_pc_exit`, `emit_runtime_pc_exit`).
- Only bits 12..63 of `core.pc` are used for in-region *addressing*: every
  address is `emit_vbase` (`pc & !0xFFF`) plus a compile-time word offset,
  and an inlined slot is always on its branch's own page (the 0xFFC
  cross-page slot is never inlined — `is_inlinable` rejects
  `word >= ENTRIES_PER_PAGE`). So the low bits look irrelevant.
- `emit_exception_exit` picks its outer stage from the **compile-time**
  `ctx.bd`, and `exception_other_word_block` stores both `core.pc` and
  `core.in_delay_slot` itself — so the inline pair looks overwritten before
  anything reads it.

## Why it is actually live

**`deliver_exception` (mips_core.rs) reads both fields straight out of
memory**, and the JIT's exception path gives it no other channel:

```rust
if core.in_delay_slot {
    cause |= CAUSE_BD;
    core.cp0_epc = core.pc.wrapping_sub(4);
} else {
    cause &= !CAUSE_BD;
    core.cp0_epc = core.pc;
}
```

`emit_exception_call_block_body` calls `handle_exception` with **only
`(core_ptr, status)`** — `Cause.BD` is not an argument. `ctx.bd` selects
*which block runs*, not what the callee sees; `exception_other_word_block`
stores `ctx.bd` into `core.in_delay_slot` precisely *because* the callee is
about to read it back out of memory.

And EPC needs the **exact word**, not the page: `cp0_epc = core.pc - 4` for a
delay-slot fault. The "only bits 12..63 matter" argument is correct about
in-region addressing and irrelevant here.

So a faulting inlined delay slot needs, live in memory at the moment it
faults: `in_delay_slot = true` and `core.pc` = the slot's own address.
Removing either produces `BD=0, EPC=pc` where the interpreter produces
`BD=1, EPC=pc-4`.

## How the removal was caught

`cargo test --release --features jitv2 --lib jitv2::` — six `equiv_test`
failures, all delay-slot exception shapes:

```
adel_in_delay_slot_epc_and_bd_match_interpreter
ades_in_delay_slot_epc_and_bd_match_interpreter
overflow_in_delay_slot_traps_with_epc_and_bd_matching_interpreter
overflow_in_delay_slot_not_taken_traps_with_epc_and_bd_matching_interpreter
nested_likely_at_last_word_not_taken_annuls_foreign_slot_like_interpreter
word_both_inlined_delay_slot_and_independent_branch_target_faults_on_slot_pass_matches_interpreter
```

with `cp0_cause` differing by exactly bit 31 (`0x30` vs `0x8000_0030`).

**`cpu-tests` did NOT catch it** — it ran 2101 passed / 61 failed, byte-identical
to baseline, because its coverage of delay-slot exception BD/EPC is thin.
IRIX also booted fine with the broken build. Neither is sufficient
validation for a codegen change in this area; `equiv_test` is the suite that
covers it, and it must be run before believing any change to
`emit_slot_semantics`, the exception stages, or anything else touching
`core.pc`/`in_delay_slot`.

## If you want this traffic back

The stores are only observable through `handle_exception`. To remove them you
would have to change the JIT→Rust exception ABI so BD and the faulting word
are passed as *arguments* (they are already compile-time constants at every
inlined-slot call site — `ctx.bd` and `ctx.word`), and have the callee use
those instead of reading `core.in_delay_slot`/`core.pc`. That is a real
design change to `emit_exception_call_block_body` +
`deliver_exception`, not a local cleanup, and it has to keep working for the
interpreter's own callers of `deliver_exception`, which genuinely do maintain
those fields live.

Related: [[emit_absolute_pc_exit-in_delay_slot-followup]] (the clear belongs
inside `emit_absolute_pc_exit`, not in each caller), and
[[block-fragmentation-blocks-cse]] (where this measurement came from, and
what the real optimization lever turned out to be).
