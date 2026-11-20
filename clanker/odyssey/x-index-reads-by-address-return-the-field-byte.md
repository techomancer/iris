# The X server's colour-index screen reads name board memory by address, and must be answered

Status: fixed 2026-11-17 (`dma_read_pack`).

**Symptom.** On an 8-bit (PseudoColor) desktop, a Motif text field leaves a
solid black block, the cursor's size (6x20 at clogin's font), at every place
its text cursor has been: clogin's "Login name" field shows one before the
first character as soon as the screen comes up. Window and board memory
agree (`screen shot` = `vpro shot`), so it is drawn, not a stale frame.

**What the X server sends** (`vpro trace`, a `gputrace` build, one arrow key
in the field):

1. the restore of the old place: an image box over the cursor's 6x20 and one
   two-word PIO image burst a row (8-bit index pixels);
2. the save of the new place: a DMA to the host, header word 0 `0x2000` (the
   frame buffer by address, not the read-drawable sentinel), word 4
   `0x83331008` (front, field at unit 3, class 0 code 1), word 7 format byte
   `0x41` (X colour index), one byte a pixel, rows of 6 bytes;
3. the I-beam: a stippled quad (pattern shape 4).

**Cause.** `dma_read_pack` refused every index layout (`BYTE_INDEX`,
`BYTE_INDEX_HIGH`) whose source was not a drawable, because there is no
window to ask for a display mode. The read wrote nothing, the host buffer
kept zeros, and the next restore painted index 0.

**Rule.** An index read of board memory named by address returns the
field's byte as it stands: the low byte for `BYTE_INDEX`, the next for
`BYTE_INDEX_HIGH`. That is the byte the scan-out shows through the window's
colour map. The display-mode and range checks stay for drawable reads.

Test: `tests/dma.rs`
`an_index_read_of_board_memory_by_address_returns_the_field_byte`.

Probably the same read: the 2026-11-07 sweep's `scrsave` of a PseudoColor 8
root coming back almost entirely black
(`gaps-found-by-the-2026-11-07-sweeps.md`); not re-run.
