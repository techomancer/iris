# The 30-bit visual's save-under reads with write code 0x10c, a packed type

Status: fixed 2026-11-21 (`DMA_X_PACKED`, `dma_read_layout`).

**Symptom.** In a Motif text field on the TrueColor desktop (nedit's Save
As name field, any X edit field), characters the text cursor has passed over
lose the columns under it, so `/root/hello` turns into `/root/ŀ:ll:`.
The 8-bit login field (`x-index-reads-by-address-return-the-field-byte.md`)
was fine.

**What the X server sends** (`vpro trace` + `vpro rec`): the same three
steps as the index case, but the save is word 4 `0x82233062` (front, field
at unit 2, 4-byte fields) and word 7 `0xb301010c`. Its write code
(`write::CODE_MASK`, bits 8:0) is **0x10c**: format 0xc (ABGR), type 8,
ABGR 2_10_10_10_REV, four bytes a pixel. The restore puts the saved words
back with an image header of the same code (`0x0fa1110c`), burst tag 0x12.

**Cause.** `dma_read_layout` looked the code up by its low byte only
(`image::CODE_MASK`, 0x0c) in `DMA_X_LAYOUTS`, which has the byte codes
0x4c/0x4b/0x49/0x41. No match, the read was refused, and the X server's
buffer kept whatever it held before: in the recording every one of the 19
restores was the flat background word `0x11c4711c`.

**Rule.** An X read whose write code has bit 8 set is looked up by the whole
nine-bit code in `DMA_X_PACKED`, laid out as the write table
(`pixel_write.rs`) packs that code. 0x10c and 0x10b (RGBA 10_10_10_2) are
listed; add others when a trace shows them.

**How it was found.** Decode the restore's burst words straight out of the
`vpro rec` file (`W` records at offset 0x100000 after each `0x4570`
header): a restore that is all one value while the screen under the cursor
held text means the save, not the put, is wrong.

Test: `tests/dma.rs` `a_30_bit_save_under_returns_abgr_2_10_10_10_rev`.
