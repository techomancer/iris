# The write pair names a field's packing, not the selected pixel mode

Status: rule, 2026-11-18. Fix in `copy::write_codec`, used by `copy_block`
and `dma_to_buffer` (screen puts only).

## Symptom

Icon Catalog on a 12-bit visual (Tezro, VPro), both raster units:

- After a scroll, the moved icons came out pale and scrambled. The newly
  exposed rows were drawn correctly.
- The tab strip showed red and green tabs instead of grey ones.

## Cause

The model packed a transfer's destination field the way the currently
selected pixel mode (block 0x017) lays it out. The X server and the library
often leave another window's mode selected.

- The scroll copies (block 0x2031) read RGBA 4:4:4:4 (word 4 `0x83331069`,
  class 1 code 13) and write RGBA/US_4_4_4_4 (word 7 `0x323300cb`: format
  0xb, type 6). They arrive with the index mode `0x20086a00` selected, so
  the model decoded 4:4:4:4 and re-encoded the result as an index.
- The tab strip is a CI8 DMA put (control `0x32050041`: COLOR_INDEX, bytes).
  Its header names field 6+2. It arrives right after an X fill in a
  4:4:4:4 window, with mode `0x000069d0` still selected, so the indices were
  stored as 4:4:4:4 colour.

## Rule

When word 7's (or the DMA control word's) write format and type name the
packing, they decide it. The library picks that pair from the destination's
pixel class (`ignore/odsy` devices/buzz/pixel-paths.md, host-to-board table).

- COLOR_INDEX: index.
- RGBA/US_4_4_4_4 in 2 bytes: 4:4:4:4.

Otherwise the pixel mode's layout still applies. Window-id copies keep
their whole-byte rule, and textured copies are left as they were.

A copy whose read and write packings are equal then moves its bits raw.
The GPU unit gets the same raw flag and destination view.

Copy ordering was checked at the same time and is correct. The CPU stages
the whole source before writing, after `raster_wait`. The GPU unit flushes
earlier records first, then gathers into scratch and scatters in separate
compute passes, so every read lands before any write.

Tests: `a_4444_scroll_copy_with_an_index_mode_selected_moves_the_bits` and
`a_ci8_put_with_a_4444_mode_selected_lands_indices` in
`src/dev/odyssey/tests/pixels.rs`. Run them with and without `VPRO_GPU=1`.
