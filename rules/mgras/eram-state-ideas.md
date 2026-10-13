# GE state and ERAM: ideas for later (not built)

Today (`hq3.rs`): one HQ thread, one live `super::gl::Gl`. On a context
switch it is copied into the outgoing context's ERAM slot
(`eram.words[0]`) and copied out of the incoming one. Maximum Impact's
second GE11 gets `eram.words[1]`, which only the kernel's per-GE
eviction transfers touch and nothing reads (see
`eram-transfers-come-once-per-ge.md`).

## 1. Run the GL state in place in ERAM

Make the live `Gl` a view of the loaded context's slot (`words[0][slot..]`
cast to `&mut Gl`) instead of a copy, so a switch changes a slot offset
instead of copying the record twice.

- `Gl` is already plain `repr(C)` numbers, valid for any bit pattern,
  and the record fits the kernel's 0x17C7-word slot (const assert). It
  must stay u32-aligned with no padding-sensitive fields.
- The borrow has to be re-derived from `eram` plus the slot on each use
  (or `Hq3Engine` hands out `&mut Gl` through one accessor). Do not keep
  a long-lived pointer: `eram` is boxed, but the slot changes.
- No slot yet (before the first context load), or a slot outside ERAM:
  this needs a scratch `Gl` to point at.
- First load (image word 1 bit 31): zero the slot in place, as
  `load_context` zeroes the copy today.
- `CP_GE_NOSAVE_NEXT_SWITCH` changes meaning: there is no save left to
  skip, and anything the kernel's context does to its `Gl` lands in its
  slot. Harmless, since that is its own slot, but check
  `swap_buffers` / `set_draw_bank` on the kernel context.
- The kernel's ERAM reads and writes must see and replace the live state
  if they ever target the loaded slot. In place they do, without a copy.
  Today a pass-through read of the loaded slot returns the state as of
  the last switch.
- Gain: two copies of the record (GL_WORDS; `mgras gl` prints it) per
  switch, about 750 switches in a two-minute Matador trace. Small. The
  better reason is that the board's GE state becomes plain memory, which
  helps snapshots later.

## 2. Alias GE1 onto GE0: reads from GE0, writes dropped

The two GE11s hold the same GL attribute state (state commands go to
both), and a context switch drains per-GE scratch. So:

- GE1 read: serve it from `words[0]`, one word late (the kernel reads it
  at S - 1). Both halves of the host buffer then hold the same record,
  as on hardware. Today GE1's half is zeros the first time round.
- GE1 write set: drop it. It carries the same words GE0's set just
  wrote.
- Then delete `words[1]` (512 KB per board).
- Keep the GE0/GE1 classification (a transfer repeating GE0's last one).
  The bug was GE1's write landing on GE0's copy, so dropping it still
  needs to know which set is GE1's.
- Test: `gl_context_survives_two_ge_eviction` must still pass. Add a
  check that both host halves match after the save.
