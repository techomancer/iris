# Follow-up: `emit_absolute_pc_exit` should own its own `in_delay_slot=false` clear

`emit_absolute_pc_exit` (src/jitv2/codegen.rs) stores `target_addr` into
`core.pc` and returns `EXEC_COMPLETE`, but never touches
`core.in_delay_slot`. Every one of its ~8 call sites today is safe only
because something upstream already cleared the flag first:

- `needs_foreign_slot_check`'s `foreign_slot_block` arm explicitly zeros it
  before calling.
- Every J/JAL/branch taken-edge call (`emit_jump_taken_edge`/
  `emit_branch_taken_edge`/`emit_nested_branch_slot`) runs after
  `emit_slot_semantics`'s non-terminating tail, which unconditionally clears
  the flag and restores `saved_pc` before returning.
  **(2026-09-02: this guarantee is GONE.)** The bracket is now
  `#[cfg(any(feature = "jitv2_lockstep", feature = "developer"))]` — the
  exception ABI passes `Cause.BD` and EPC as arguments, so nothing reads
  those fields back for an inlined slot. See [[inlined-slot-pc-bd-bracket]].
  Compiled code no longer *sets* `in_delay_slot` for an inlined slot either,
  so these call sites remain correct — the flag is simply never true there to
  begin with — but they are now correct **by luck of what runs before them**,
  not by an upstream guarantee. Which is exactly what this note warned about:
  the removal immediately broke
  `emit_foreign_page_annulled_not_taken_exit`, which had been silently
  inheriting `in_delay_slot = 1` from the bracket. That one now takes an
  explicit `pending_outer_transfer` parameter (its two callers need opposite
  values). The general fix below — move the clear *inside*
  `emit_absolute_pc_exit` — is still not done, and is now more clearly worth
  doing rather than less.
- The annulling-Likely not-taken arm never sets the flag in the first place
  (the slot is skipped entirely, mirroring `handle_branch_likely_skip`).

This is a real but non-local invariant — a future call site that forgets to
clear the flag first would silently leave a stale `in_delay_slot=true`
resulting in the interpreter's next `step()` misinterpreting a plain
instruction as mid-delay-slot after re-entry.

**Direction for the fix, when someone gets to it**: the clear belongs
*inside* `emit_absolute_pc_exit` itself, not in each caller — an absolute-PC
exit is inherently "we are now at a plain, non-delay-slot instruction," so
the function that owns that contract should enforce it, the same way
`emit_exit_block_body`'s "plain boundary" (`handle_exec_complete`'s `pc+=4`)
implies `in_delay_slot=false` structurally. Every current caller already has
the flag false at the call site, so adding an unconditional store inside
`emit_absolute_pc_exit` is a no-op today and only guards future call sites.

Found while auditing `emit_absolute_pc_exit` call sites during the
jitv2_lockstep region-boundary divergence fix (see
[[jitv2_lockstep_region_boundary_divergence]] if that note exists, or the
`ls_before`/`ls_delay_target_before` restore-on-divergence work in
mips_exec.rs `lockstep_compare`). Not fixed in that session — scoped out to
avoid touching unrelated codegen while the lockstep fix was in flight.

---

## RESOLVED (2026-09-10)

The clear now lives inside `emit_absolute_pc_exit`, unconditionally, as this note
recommended.

**Why unconditional, and a wrong turn worth recording.** The first attempt gated the store
on `cfg(any(jitv2_lockstep, developer))`, reasoning that those are the only features under
which *compiled code* stores a non-zero `in_delay_slot` (the `emit_slot_semantics` bracket),
so a release build would find the flag already zero and the store would be pure cost.

That reasoning was wrong, and the gate would have left the original bug in release builds.
The flag does not only arrive from within the region — it arrives **set from outside it**:

- `emit_foreign_page_slot_exit` stores `in_delay_slot = 1`, sets `pc` to word 0 of the next
  page, and returns. The interpreter's `branch_delay` does the same.
- The next dispatch therefore *enters* a region at a word that is a delay slot, with the
  flag live. This is the foreign-page-slot protocol (0xFFC branch, slot on the next page),
  not a debug path, and it has no cfg gate at all.
- If that region then leaves via `emit_absolute_pc_exit`, the stale flag goes out with it
  and the interpreter's next `step()` treats a plain instruction as mid-delay-slot.

So the hazard is live in exactly the configuration the gate would have skipped. **Any entry
word that is a delay slot can have `in_delay_slot == true` regardless of features.**

**Cost.** ~2%, measured with `iris-bench run`: 227.5 / 227.7 MIPS without the store versus
223.0 / 223.0 with (three runs each; the clusters are tight, so this is real, not noise).
It lands on all eight call sites, which are the hot branch/jump transfer paths. Paid
deliberately — a correctness hazard on a non-debug path is not worth 2%.

If that 2% ever needs reclaiming, the route is *not* a cfg gate: it is tracking at compile
time whether a given region can be entered at a delay-slot word (the analyzer already knows
which words are slots) and emitting the store only in regions where it can.

---

## Do NOT gate that store on word kind (attempted and reverted, 2026-09-20)

The note above suggests reclaiming the 2% by "tracking at compile time whether
a given region can be entered at a delay-slot word ... and emitting the store
only in regions where it can". Two things went wrong when that was tried:

1. **A region-level predicate is always true.** "Can this region be entered at
   a delay-slot word" reduces to `is_entry_point || is_branch_fallback_successor`
   over the region's entries, and every region has at least one entry point.
   It gates nothing.

2. **A per-word predicate is not safe.** The obvious refinement — emit the
   store only for entry/fallback-successor words, since an interior word
   "cannot carry a live foreign-slot flag" — breaks flow control. The JIT
   *writes* `in_delay_slot` on exit paths (`emit_foreign_page_slot_exit` stores
   1), so the flag's value at an exit is not a function of the word's arrival
   kind alone. Removing the clear at interior words can let a set flag escape.
   Reverted before measuring.

**And it would not help forwarding anyway.** Exception/bail exits do not block
store-to-load forwarding at all — see
[[what-actually-blocks-gpr-forwarding]], shapes `exitbr`/`exitbr2`. This store
costs code size and ~2% MIPS, nothing else. Anyone reclaiming it needs a real
flow-sensitive analysis of what the flag holds at each exit, not an arrival-kind
heuristic.

## Deferred: skip the pc/bd reload on the `_live` exception exit

`emit_exception_exit_live` loads `core.pc` and `core.in_delay_slot` purely to
pass them as arguments to `handle_exception_at_fn(core, status, fault_pc, bd)`.
The callee already has `core` and could read both fields itself — on this path
they are exactly what the interpreter left there, unchanged, which is *why*
this variant loads them instead of materializing compile-time constants.

So the two loads are redundant with the callee's own reach. Removing them
would need a second hook (`handle_exception_live_fn`) that takes only
`(core, status)`, since the existing ABI is shared with
`emit_exception_exit_const`, whose whole point is passing compile-time values
that are *not* in memory.

**Measured payoff: 20 sites, 40 loads, 4.4% of CLIF loads on one real region —
all on cold exception paths.** Near-zero runtime effect. Recorded for
completeness, not recommended: it adds a hook and an ABI variant to save code
that does not execute.
