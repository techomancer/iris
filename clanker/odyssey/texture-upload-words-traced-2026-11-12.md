# Texture uploads: word 4's field size, the image header's swap and type bits

Status: rule, 2026-11-12 (IRIX 6.5.30, V12 model, `ignore/ip30-ody`).

## A DMA upload's word 4 can name a field wider than its pixel

- Symptom: sgiquake1 (`/usr/games/sgiquake1/quake.gl.r10k`) showed its HUD,
  weapon and most world textures as noise on both raster units, from
  `fa2c91c9` (2026-11-11) on.
- Its texture levels are RGBA4 and RGB5_A1 (2-byte texels). The library
  selects the level (select words base, pitch, `0x08000060`) and uploads
  4-byte RGBA or ABGR pixels by DMA with word 4 `0x80003000`: bits 15:13
  name a 4-byte field, bits 19:16 a 2-byte pixel.
- `fa2c91c9` took the selected mode's packing only when its field size
  equals word 4's (for 4Dwm's icons: word 4 `0x8223305a`, a 4-byte field in
  the 16-byte frame-buffer pixel, after a stale index select). The texture
  uploads then packed 4-byte fields into 2-byte texels.
- Rule (`Board::dma_fields`): the selected mode's packing applies when its
  field has word 4's size, or when word 4's field does not fit word 4's
  own pixel. Test: `a_texture_upload_naming_wider_fields_than_its_pixel_packs_as_the_level`.

## Image header (FIFO) codes: bits 8:0 and bit 10

Traced with piglit's `texture-packed-formats` (format word low bits =
format, bits 8:5 = type, `pixel_write.rs`):

| GL type | code |
|---|---|
| RGBA, UNSIGNED_INT_8_8_8_8 / BYTE | 0x4b |
| RGBA, 8_8_8_8_REV | 0x4c |
| BGRA, 8_8_8_8 / 8_8_8_8_REV | 0x51 / 0x50 |
| RGBA 10_10_10_2 / 2_10_10_10_REV | 0x10b / 0x10c |
| 4_4_4_4 / _REV (RGBA, BGRA) | 0xcb 0xcc / 0xd1 0xd0 |
| 5_5_5_1 / 1_5_5_5_REV | 0xeb 0xec / 0xf1 0xf0 |
| RGB, BGR 5_6_5 | 0x149, 0x14d **or** 0x189, 0x18d, by internal format |
| 3_3_2 / 2_3_3_REV | 0xa9 / 0xad |

- `GL_UNPACK_SWAP_BYTES` sets header bit 10 (`image::HEADER_SWAP`) and
  leaves the code alone. The model reverses each element (a 16/32-bit
  component, else the whole pixel: 8_8_8_8 arrives as 0x4b), INFERRED from
  GL; the board's own swap is unseen.
- `image.code` is the header's low byte. 0x149's low byte 0x49 is the RGB
  byte code, so a type-8+ code that the write table knows must skip the
  byte-stream path (`image_burst_word`).
- Unit tests in `pixel_write.rs`. The DMA path's word 7 has no swap bit
  seen yet.

## Comparing the two raster units by screenshot

- The gamma ramp (colour RAM 0x1a00..) changes during a session (a fresh
  boot had 0/133/200 at entries 0/0x80/0xff, a later session 44/188/252):
  screenshots from two sessions differ everywhere. Read `vpro lut 0x1a00 1`
  etc. before and after a sweep and compare only under the same ramp.
- Same ramp, final build: all 101 glprim texture shots (32 sized formats x
  8x8, 64x64, 256x32, plus mipmapped and linear cases) were pixel-identical
  on the CPU and GPU units.
