# GE state and ERAM: ideas for later (not built)

Today (`hq3.rs`): one HQ thread, one live `super::gl::Gl`. On a context
switch it is copied into the outgoing context's ERAM slot
(`eram.words[0]`) and copied out of the incoming one. Maximum Impact's
second GE11 gets `eram.words[1]`, which only the kernel's per-GE
eviction transfers touch and nothing reads (see
`eram-transfers-come-once-per-ge.md`).

## 1. Run the GL state in place in ERAM

How the board seems to work: the GE11 microcode runs on its working
state inside the chip (WRAM, CRAM, registers). On a context switch the
microcode saves the outgoing state to that context's ERAM slot and
loads the incoming one's (HQ3.h, "Where the state lives"). The rest of
ERAM holds resources (libGLcore's imaging tables and the like). So ERAM
is the save area, not where the GE runs. Before the first context
switch after boot, the GE still has state, its internal one, but no
slot owns it. Our live `Gl` struct plays the part of that internal
state, and copying it on a switch matches the hardware.

The idea, purely as an emulator shortcut: make the live `Gl` a view of
the loaded context's slot (`words[0][slot..]` cast to `&mut Gl`), so a
switch changes an offset instead of copying the record twice.

- `Gl` is already plain `repr(C)` numbers, valid for any bit pattern,
  and the record fits the kernel's 0x17C7-word slot (const assert).
- Re-derive the borrow from `eram` plus the slot on each use. Do not keep
  a long-lived pointer.
- Before the first context load (`slot: None`), and for a slot outside
  ERAM, the view needs a scratch `Gl`: the "internal" state above.
- First load (image word 1 bit 31): zero the slot in place.
- Behaviour that differs from the hardware's save/load model:
  - `CP_GE_NOSAVE_NEXT_SWITCH` has nothing left to skip.
  - A kernel ERAM read of the loaded slot sees live state, where the
    board (and today's code) gives the state as of the last save.
  - A kernel ERAM write to the loaded slot changes live state, where the
    board would overwrite it at the next save.
  The kernel probably never touches the loaded slot, but check before
  relying on it.
- Gain: two copies of the record (GL_WORDS; `mgras gl` prints it) per
  switch, about 750 switches in a two-minute Matador trace. Small.
  Snapshots would gain little too: the live state has to be saved
  either way.

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
