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
