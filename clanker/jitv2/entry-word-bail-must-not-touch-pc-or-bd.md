# An entry-word bail must not touch `pc` or `in_delay_slot`

`j2 entrypre on` (`IRIS_ENTRY_PREAMBLE=1`) makes an externally-dispatched entry
word sample `hot.interrupts` before running, instead of deferring to the next
head. Getting the *bail* wrong panics IRIX within seconds of kernel start.

## Why the feature exists

Normally the dispatch head jumps straight to an entry word's **body block**,
bypassing its preamble (`skip_entry_preamble`). Delivery is therefore deferred
by one dispatch — bounded, never lost, since every internal back-edge onto an
entry word still pays the preamble. But the window scales with how fast the JIT
retires code.

That matters because a full `jitv2_lockstep` boot — every JIT instruction
verified against the interpreter inline, hence very slow — reaches a working
desktop. So instruction emulation is **correct**, and what differs under a fast
JIT is *when* interrupts are sampled, not what the code computes. This toggle
closes that window so the difference can be tested against a live boot.

## The wrong implementation (tried, reverted)

Emptying `entry_body_blocks` so the dispatch head falls through to the entry
word's **ordinary** block. One flag, looks equivalent, and it is not:

```
PANIC: tlbmiss: invalid kptbl entry
```

(`kern/os/trap.c:3110` — a KSEG2 TLB miss whose page-table entry is invalid, in
kernel mode.)

The ordinary block opens by unconditionally storing `in_delay_slot = false`.
Its own comment says why that is sound:

> unconditionally false for in_delay_slot (**an internal edge into an entry word
> is never a delay-slot landing**)

and, about external arrivals:

> the real external-dispatch entry bypasses straight to this entry's own body
> block instead (the dispatch head's Switch above), **never running any of this**

An **external** dispatch *can* land on a delay-slot word — that is exactly what
the foreign-slot protocol is for. Routing external arrivals through that block
destroys the armed transfer, the guest branches somewhere it should not, and the
kernel faults on a garbage KSEG2 address.

## The right implementation

Emit the check in `entry_block`, **before** the dispatch `Switch` — the same
slot `emit_fr_mode_guard` already occupies, and for the same reason: at that
point the function has not touched `core.pc` or `core.in_delay_slot`, so the
live values are still exactly what the dispatching `step_jit` set up, foreign
delay-slot arrivals included.

The bail (`emit_entry_interrupt_bail`) is a bare `return EXEC_FALLBACK` that
stores **neither** field. `step_jit` turns that into `step_int()`, which
re-dispatches at the untouched PC and runs the interpreter's real
`step_preamble!`.

**Do not use `emit_bail` here.** It jumps to the shared exit stub, which
recomputes and stores `core.pc` from the compile-time word offset
(`emit_exit_block_body`). Correct for an interior instruction; wrong for an
entry word, whose PC is already right on arrival. Same reasoning as
`emit_interp_fallback_exit`'s "does not materialize core.pc before calling".

## Test

`entrypre_bail_preserves_pc_and_delay_slot_state` seeds the exact combination
that broke — a foreign delay-slot arrival with an interrupt already pending —
and asserts `EXEC_FALLBACK`, semantics did not run, `pc` unchanged,
`in_delay_slot` still set, `delay_slot_target` intact. Verified to fail against
**both** wrong shapes: toggle off, and bail-via-`emit_bail`.

## The toggle is a process-global — lock it in tests

`ENTRY_PREAMBLE` is read by *every* compile, so a test that flips it while
another test compiles a region in parallel gives that region an entry check it
never asked for. That surfaced as a 1-in-N failure visible only in full runs and
never when the test was filtered. Any test touching it must hold
`ENTRY_PREAMBLE_TEST_LOCK` for the whole window and restore the prior value.

At runtime the same compile-time-read property means `j2 entrypre on` needs a
`j2 flush` (CPU stopped) to affect already-compiled regions — same contract as
`j2 fallback`.
