# A lanes field's view is eight bytes long

Status: 2026-11-04, fixed in f2652441.

Symptom: moving a window left ghost text in the title bar. `vpro pix` showed
bytes 6 and 7 of the A field still holding the old pixel.

Cause: the X server copies a 16-byte screen pixel as two 8-byte halves (copy
block 0x2031). The view of a `Codec::Lanes` field gave its length as 6
bytes, so every copy left bytes 6, 7, 14 and 15 behind.

Rule: an 8-byte lanes field's view length is 8 (`view.rs`,
`Kind::Coded(Codec::Lanes) => 8`). Test:
`a_window_move_copies_every_byte_of_both_halves` in `tests/pixels.rs`.
