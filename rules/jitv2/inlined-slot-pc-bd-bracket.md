# The inlined delay slot's `core.pc` / `in_delay_slot` bracket

**History, in order — read all three parts before touching this.**

1. It looked dead. It was not.
2. It was made dead, deliberately, by changing the exception ABI.
3. Two other things silently depended on it. Both are now explicit.

## What it is

`emit_slot_semantics` (src/jitv2/codegen.rs) wraps every **inlined** delay
slot:

```
in_delay_slot = 1
saved_pc = load core.pc
core.pc  = <slot's own address>
  ... the slot instruction's real semantics ...
in_delay_slot = 0
core.pc  = saved_pc
```

As of 2026-09-02 all six of those memory operations are
`#[cfg(any(feature = "jitv2_lockstep", feature = "developer"))]`.

## Part 1 — why it was NOT removable (2026-09-01, reverted)

A first attempt cfg-gated the bracket on the argument that nothing in a
compiled region reads either field back: an in-region branch edge is a plain
`jump` (`emit_target_edge`), a region-leaving exit writes its own `core.pc`,
and only bits 12..63 of `core.pc` matter for in-region addressing.

All true, and all beside the point. **`deliver_exception` (mips_core.rs) read
both fields straight out of memory:**

```rust
if core.in_delay_slot {
    cause |= CAUSE_BD;
    core.cp0_epc = core.pc.wrapping_sub(4);
}
```

and `emit_exception_call_block_body` called `handle_exception` with only
`(core_ptr, status)`. `ctx.bd` selected *which stage block ran*, not what the
callee saw. EPC also needs the **exact word**, not the page, so the
"low bits are dead" argument did not apply.

Caught by six `equiv_test` delay-slot exception tests, `cp0_cause` differing
by exactly bit 31. **`cpu-tests` passed the broken build (2101/61, identical
to baseline) and IRIX booted fine** — the common case self-heals, because
delivering with the stale pc resumes at the branch and simply re-executes it.
It breaks only where a handler *inspects* rather than retries (reading
`Cause.BD` to find the faulting instruction, or a non-restartable
trap/overflow/breakpoint in a slot, which would loop).

## Part 2 — how it was actually removed (2026-09-02)

Not by deleting the stores, but by removing the reason they existed: **pass
EPC and BD as arguments instead of through memory.**

- `mips_core::deliver_exception_at(core, status, fault_pc, bd)` holds the
  logic; `deliver_exception(core, status)` is now a two-line wrapper reading
  the fields, so interpreter and `jitv2_verify` callers are untouched.
- `MipsExecutor::handle_exception_at`, and a new
  `MipsCore::handle_exception_at_fn` FFI hook `(ctx, status, fault_pc, bd)`.
- Codegen splits the exit into two wrappers over **one** shared call block
  (the two outer stage blocks are deleted):
  - `emit_exception_exit_const` — `emit_word_addr(ctx.word)` + `iconst(ctx.bd)`.
    Every ordinary in-region instruction, **including an inlined slot**.
  - `emit_exception_exit_live` — two loads. Only for the entry word and
    branch-fallback successor, which inherit state from outside the region.

Measured over 300 real IRIX corpus pages (`zz_corpus_sizes`,
`IRIS_JIT_DISASM=1`, `opt_level=speed`, no `developer`):

| | before | after |
|---|---|---|
| `pc` stores | 5,884 | 1,243 |
| `in_delay_slot` stores | 4,900 | 259 |
| `pc` loads | 3,959 | 1,967 |
| `gpr` stores | 6,790 | 6,790 |
| **total stores** | **24,222** | **12,431** |

Half of all emitted store traffic. (`gpr` unchanged is the correctness
check — that traffic is architectural and must not move.)

## Part 3 — the two hidden dependencies it was masking

Both were silent inheritances of the bracket's unconditional writes, and both
now set what they need explicitly:

**`trust_live_pc_bd_on_exc` leaked into slots.** When a branch is itself an
entry word (or branch-fallback successor) that flag is set on its `ctx`, and
`emit_slot_semantics` inherited it — routing the *slot's* fault down
`emit_exception_exit_live`, which loads a flag the bracket no longer writes.
Fixed by clearing it alongside `ctx.bd = true`: a slot's fault state is
always compile-time known.

**`emit_foreign_page_annulled_not_taken_exit` inherited `in_delay_slot`** —
and its two callers need **opposite** values. Head-level branch-likely at
0xFFC: nothing pending, `false`. Nested: the *outer* branch's transfer is
still live, `true`. Now a `pending_outer_transfer` parameter. This is exactly
the fragility [[emit_absolute_pc_exit-in_delay_slot-followup]] flagged for the
mirror-image case, hit from the other direction.

## What still needs the bracket

- **`jitv2_lockstep`** — the compare reads `core.pc`/`in_delay_slot` as the
  slot's post-state (and `delay_slot_target` as the expected pc, since a slot
  retires *from* `in_delay_slot = true`).
- **`developer`** — `emit_dev_trace_bp` reports the slot's own pc for `dt`.

## Testing rule

**`equiv_test` is the only suite that covers delay-slot exception BD/EPC.**
cpu-tests and a full IRIX boot both passed the broken build. Run
`cargo test --release --features jitv2 --lib jitv2::` before believing any
change to `emit_slot_semantics`, the exception path, or anything touching
`core.pc`/`core.in_delay_slot`.

Related: [[block-fragmentation-blocks-cse]] (where the measurement came from),
[[deferred-delay-slots-unified]] (the sibling cleanup in the same area).
