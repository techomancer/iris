# Plan: VPro board memory as the hardware has it

Status: 2026-11-04, steps 1-5 done on `odyssey-vram` (the switch is in, the
old layers deleted, unit tests and goldens pass); step 6 in progress
(pixtest in the guest: 31 failures, all of them predating the series;
see Follow-ups). Replaces the model's fixed
1280x1024 planes and its object store with one board memory addressed
the way the board addresses it. Prerequisite for real pixel formats
(4:4:4:4, 5:5:5:1, 10:10:10:2, 12-bit), pbuffers, render-to-texture,
accumulation and any display mode other than 1280x1024.

## Why

The board has one pool of SDRAM (32 to 128 MB, size in the strap word)
holding the command FIFO, the frame buffer, overlay and window ids,
accumulation, textures, pbuffers and scratch images. The driver's memory
manager allocates it; the board only ever sees a base address, a pitch in
tiles and a format word. Nothing on the board knows where one allocation
ends.

The model instead keeps colour, depth and stencil as fixed 1280x1024
arrays outside board memory, and invents an "object store": every base a
select names becomes an object with a width, a row count, per-word written
bits and least-recently-used eviction. Consequences seen so far:

- A copy between two board addresses (the library's signed-texture
  upload: burst into scratch at 0x16b000, copy through scale/bias into the
  level at 0x16f000) does nothing: the copy code knows only window to
  object and object to window. The texture stays "unwritten" and samples
  white. (Real V10: correct.)
- The sampler treats an unwritten object as empty; real memory holds what
  was last written.
- Display mode, pbuffer size, pixel depth and format are fixed by the
  model, not by what the driver configured (the V10 here runs 1024x768).
- Paths that alias memory cannot work: rendering into a texture level,
  accumulation through a scratch buffer, reading a pbuffer as a texture.

## What is known (James's `ignore/odsy` notes)

- Memory is managed in 16 x 16-pixel tiles; an allocation unit is 1 KB
  (16 x 16 x 4 bytes); depth 16 and 8 pack two and four tiles a unit
  (`docs/display.md`, `odsyConfigFb`).
- `odsyConfigFb` lays out the frame buffer: width and height rounded up to
  tiles, the tile row to an even count, start on a four-unit boundary;
  then the overlay and window-id buffer (2 bytes a pixel), the unused
  coarse-Z slot, up to four 8-bit accumulation buffers, and the first free
  address for textures and pbuffers. It pushes six scan-out commands:
  overlay base, pixel base, width, height, stride, depth code (4, 3, 2 for
  32, 16, 8). The published figure: on-screen memory is pixels x 10 or 18
  bytes, an 8- or 16-byte frame buffer fixed at X start.
- Buffer words are board addresses: an allocation's first unit shifted
  left by 5 (`odsy_mm_commit`); region starts shifted left 9 or 13 in the
  words handed back.
- Block 0x017 (destination select) is the drawable's buffer descriptor:
  word 0 the draw buffer's location, word 1 its width in tiles, word 2
  format and mode fields, word 3 the second buffer's location.
- Texture level descriptors (0x280 + n): word 0 the level's board address,
  word 1 its pitch in tiles.
- Scan-out (PB&J) reads BUZZ memory through base, pitch and depth
  registers in tile units (0x2800, 0x2801, 0x2807, 0x2809).
- The command FIFO is the first 16 tiles of 32-bpp memory (0x6b0 its base
  tile, 0x6b1 its size).

## What is unknown, and how to settle it on the real V10

- The layout inside a tile: pixel order (row-major or swizzled), and how a
  pixel's bytes hold front and back colour, depth and stencil in the 8- and
  16-byte frame buffer. Settle with directed tests on the board: draw a
  known pattern (distinct colours, depths, stencils per pixel), then read
  the frame buffer's raw memory back by a raw-format DMA (the model already
  knows a raw 64-bit read form, `dma_raw64`) or a copy into a texture level
  read back through glGetTexImage; and as root, read BUZZ registers and any
  mapped memory window directly.
- Address units of every base word (bytes, 32-byte, 512-byte, tile units)
  per command: settle by tracing the guest's own buffer words against the
  layout `odsyConfigFb` reports, and on the board by drawing at known
  addresses and reading back.
- How texel formats sit in memory (4/8/12/16 bits a component, packed
  forms), and what the copy and DMA format words convert.
- Depth and stencil encoding in memory (the 24-bit float word, its place
  in the pixel).

## Settled (2026-11-03)

Details and evidence in `ignore/odsy/docs/measured.md` sections 4 and 5.

- Every board address in a command counts 16-byte units (the memory manager,
  the 6.5.7 frame-buffer code and the traced region sizes agree). Allocation
  tiles are 16 x 16 pixels at 32 bytes = 8 KB = 512 units; the command FIFO is
  the first 16 (128 KB), so the frame buffer starts at 0x2000.
- Block 0x017 is the pixel mode: origin, stride (bits 11:0, tiles), status,
  coarse-Z origin. Status: FB_DEPTH 15:13 and WIN_DEPTH 12:10 (bytes =
  2 << code), PIXEL_FMT 9:7, FIELD_FMT 6:4, PIXEL_INDEX 19, SCRATCH_BUFFER 27
  (inferred), 23 the swap/clear transient.
- The frame buffer is one region of 8 or 16 bytes a pixel (X at depth 24 on
  the 128 MB board: 16), then the 2-byte overlay/WID region, then accumulation
  (8 bytes a pixel each, if any), then free memory. Back buffer, depth and
  stencil have no regions: they are inside the frame-buffer pixel.
- Front/back is block 0x024 (4-bit mask, 0x3 front, 0xc back), not 0x017.
- Copy and DMA address words: low 29 bits address, bit 28 = relative to a
  drawable register (0x10000019 draw, 0x10000010 read), the model's old
  "sentinels".

## The model's choices where the board cannot be seen

- Tile order row-major, pixels row-major inside a tile.
- Fields inside a pixel from the status word's WIN_DEPTH and PIXEL_FMT,
  through the library's own field-offset table (`field.rs`, `window_fields`;
  `ignore/odsy/docs/measured.md` section 8). The X server's double-buffered
  window on 16-byte pixels: front at byte 12, back at 4, depth at 8.
- Depth: the 24-bit word big-endian in the field's first three bytes,
  stencil in the fourth. Colour in an 8-byte field: 12-bit lanes. The
  window-id buffer (FB 2 bytes, PIXEL_FMT 1): id byte, then overlay byte.

## Shape (2026-11-03, user direction)

No plane layer, no object layer, no inferred buffers. Three parts, as the
MGRAS model has them (`src/dev/mgras/pixmem.rs`, `frame.rs`):

1. **Memory** (`vram.rs`): one byte array; addresses, strides and pixel sizes
   exactly as commands give them.
2. **Drawing** reads its destination from the board's own registers each time
   they change: the pixel mode (block 0x017: origin, stride, status) and the
   buffer select (block 0x024). The back end, image bursts, clears, copies and
   DMA compute byte addresses from those words, nothing else. A texture fetch
   uses the level descriptor words (0x280 + n). A copy uses its own eight
   operands (both strides and pixel sizes are in them). A DMA uses its header
   (word 0 origin, word 2 stride, word 4 bits 19:16 pixel size, 15:13 field
   size).
3. **Scan-out** is a `Frame::snapshot` taken by the display thread after the
   retrace, from the PB&J registers the display FIFO writes (0x2800 overlay
   and WID base, 0x2801 pixel base, 0x2802/0x2803 size, 0x2807 stride, 0x2809
   depth code), the per-WID swap bits (BUZZ 0x10c4/0x10cc), the per-WID
   display modes, gamma and cursor; then `Frame::compose`, one pass over the
   whole screen. No dirty rows.

The only model choices are the ones nothing can observe: pixel order inside a
tile, and which bytes of a frame-buffer pixel hold front colour, back colour,
depth and stencil (below).

## Steps

On a branch (`odyssey-vram`); the old planes, objects and scan-out go in
one switch, since a half-switched model would draw where nothing scans.

Steps 1-5 are done (2026-11-04); step 6 is in progress.

1. Done: measurements, `vram.rs`.
2. PB&J registers from the display FIFO, and `Frame::snapshot`/`compose`
   from memory (shape of `mgras/frame.rs`), behind the existing presenter.
3. Destination state from blocks 0x017 and 0x024: the back end (fragment
   pipeline), the 2D stream, clears and image bursts write memory through
   it; depth and stencil in the same pixel.
4. Texture fetch from level descriptors; DMA up and down from headers;
   copies from their operands, converting by the two sides' formats.
5. Delete `objects.rs`, `Planes`, the dirty-row machinery, `memory.rs`'s
   plane layer, and the debug commands' plane dumps (`vpro fbdump` from
   memory instead).
6. Goldens and replays re-checked; `sigtex`, `texsweep` and the first board
   comparison against the V10.

Each step keeps the replays and golden scenes (re-blessed only where the
hardware comparison says the model was wrong) and the board comparisons
(`clanker/odyssey/v10-vs-emulator-first-comparison.md`) as the checks.

## Follow-ups (after the switch is checked)

- **Specialise memory access.** Today `view::View` decodes the field kind and
  pixel size at run time and every `Rows` access matches on them per pixel.
  Next: zero-sized field-kind types (`Rgba8`, `Z24`, `Byte`, `Rgba4444`,
  `Rgb5a1`, `Rgb10a2`, `Lanes12`, `Index16`) with `const BYTES` and
  load/store, the pixel size as a const shift, and one dispatch per draw on
  (colour kind, pixel size) into monomorphised back-end functions; depth and
  stencil stay Z24 plus a byte at a per-draw offset. Quad addressing: the
  tile base once per 2x2 quad, the lanes at {0, 1, 16, 17} << log2. Span
  paths (clears, 2D fills, image bursts) and texel fetch walk runs of up to
  16 contiguous pixels within a tile. The branchy path stays as the
  reference: check bit-exact against it, and time on the Quake replay.
- Settle the order inside a tile and the depth/stencil split on the V10
  with raw `MMxfr` reads at shifted origins (`ignore/odsy/docs/measured.md`
  section 8).
- Copy direction bits (word 7 bits 31:30): check with a known overlapping
  copy on the V10.
- pixtest (2026-11-04, 8:8:8:8 double-buffered visual): 31 failures at
  8e48d6b6, against 57 on the commit before the series (92bb43be); every
  one of the 31 fails there too. Board memory fixed the 26 signed texture
  uploads. Left: colour-index draws, scale and bias, maps, byte-swapped
  and 5:5:5:1 draws, ABGR_EXT bytes, CopyPixels zoom (1,-1), colour
  images under the depth test (bursts never test depth), LUMINANCE
  GetTexImage, the 4:4:4:4/swap read, and FRONT_AND_BACK geometry
  (`clanker/odyssey/front-and-back-arrives-as-front.md`).
- Replay checkpoints: quake1-timedemo re-taken at 8e48d6b6; mirrors and
  perfly still hold the old model's (their recordings are not here).
