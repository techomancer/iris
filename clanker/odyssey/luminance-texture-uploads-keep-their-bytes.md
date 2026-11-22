# Luminance and luminance-alpha texture uploads by address keep their bytes

Status: fixed 2026-11-21 (`dma_fields`); check added (`check_level_packing`, feature `vprodiag`, in `packing.rs`).

**Symptom.** buttonfly's button labels (a texture font) drew every glyph as
a pale opaque box with the letter faint inside it. glprim's 256x256
LUMINANCE and INTENSITY textures were stored the same wrong way (found by
the check below, not by eye).

**What the library sends.** A texture big enough to go by DMA is uploaded
into board memory with no level selected: word 4 `0x80001000` (2-byte
pixel and fields, format byte 0) and word 7 code `0x48` (LUMINANCE_ALPHA
bytes) or `0x42` (LUMINANCE bytes; INTENSITY is sent as luminance). The
draws then name the level as format index 0xb (luminance byte, alpha byte)
or 4 (one byte), `texel_storage`. buttonfly also builds mip levels 1-3 on
the board with `TEX_RECT` draws (`texture_build_level`); those were right.

**Cause.** With no level selected, `dma_fields` took the field's packing
from `copy::source_codec(2, 0)`: four 4-bit components. A clear texel
(L 0xff, A 0) was stored as 0xfff0 and sampled with alpha 0xf0.

**Rule.** A DMA upload to board memory by address with 2-byte fields
stores code 0x42 as `FIELD_ONE` and code 0x48 as `FIELD_TWO`. Other codes
keep `source_codec`; the copy path is unchanged (its tests send the same
word 4 for 2-byte id/overlay pairs). An ALPHA source by DMA was not seen
disagreeing; no rule for it.

**How to find the next one.** Every upload into board memory (DMA, FIFO
image into a scratch surface, board-built mip level) records its surface and
packing; each `TEX_LEVEL` descriptor over that memory is checked against it
once (`Board::check_level_packing`; build with `--features vprodiag`). `vpro status` shows "texture packing: N
levels checked ..., M disagree" and the distinct kinds of disagreement;
`log odyssey on` logs each. Run `test/gltest/texpack.sh` in the guest and
read the status. On 2026-11-21 it covered all 32 sized formats, the six
unsized source formats, both upload paths, mipmaps, ABGR, 16-bit sources,
sub-images and `gluBuild2DMipmaps`: only the luminance case disagreed.
A texture packed one way and sampled another can still round-trip through
`glGetTexImage`, which is why the texture sweep's `texread` missed these.

Tests: `tests/dma.rs` `a_luminance_alpha_texture_upload_keeps_its_alpha`,
`a_luminance_texture_upload_keeps_its_byte`,
`a_level_is_checked_against_the_upload_that_wrote_it`,
`an_eight_byte_level_over_lanes_is_the_same_packing`.
