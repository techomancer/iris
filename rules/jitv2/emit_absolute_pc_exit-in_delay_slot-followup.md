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
