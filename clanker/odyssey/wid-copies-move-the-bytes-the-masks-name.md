# Window-id copies move whole pixels, under the draw and plane masks

Status: 2026-11-04, fixed in b6597ce0 and 64f3ed7f.

**Symptom.** Popup menus (powerflip's right-button menu) drew their
title but showed the window underneath through the body.

**What the X server sends.** It fills a menu body in the overlay by
putting one 16x16 tile (an image burst under draw mask 0xc) and then
doubling it with block-0x2031 copies inside the window-id buffer
(0x142000, stride 0x50). Each copy names 2-byte pixels and a 2-byte
field at offset 0: the id byte and the overlay byte together. The draw
mask is 0xc (overlay byte only) and the plane mask 0xff000. The body's
window-id quad (mask 0x3, plane 0x7000, index 2) comes between two rounds
of copies.

**Two bugs.**
- The copy took the pixel mode's 1-byte id field as its destination
  codec, so it moved only the id byte.
- Copying the whole pixel then overwrote the id bytes that the quad had
  written in between, with the source's id 0.

**Rule.**
- A copy uses the pixel mode's field only when that field is at least as
  wide as the copy's field. The window-id copies (2 bytes over 1-byte
  fields) move raw pixels. The library's read copies (a 4-byte colour into
  an 8-byte, 12-bit-lanes scratch) still convert.
- A whole-pixel copy into the window-id layout writes byte 0 only if the
  draw mask selects buffer 0, and byte 1 only if it selects buffer 1, both
  under the plane mask, as `paint_aux` does.
- Bytes move raw only when both sides hold a word the same way.

**Check.** `vpro pix` on the menu body: the overlay id in the id byte and
the menu's index in the overlay byte. Test:
`tests/pixels.rs` `an_id_plane_copy_moves_the_byte_the_drawing_mask_names`
and `a_read_copy_into_a_lanes_scratch_converts_the_colour`.
