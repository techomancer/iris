# Two pixels that share a memory word race on the GPU unit; IMPACT's layouts can alias, IRIX's do not

Status: rule, 2026-11-08, `src/dev/mgras/wgpu`.

**Symptom.** The IMPACT GPU unit's triangle sweep failed against the
interpreter: hundreds of words where both sides drew, in different
colours, the GPU's saturated. The kernel computed the right colour for
the pixel it drew.

**Cause.** The generators (`src/dev/mgras/test_gen.rs`, from the JIT
sweep) pick narrow and overlapping layouts. IMPACT has no linear
framebuffer: pixel (x, y) is in page `ptr + (y / 16) * xtiles + x / 192`
(`pixmem::Buffer::locate`). With `DRBsize` giving one tile a row,
pixel (192, y) is the word of pixel (0, y + 16 * engines); buffer A at
page 0x40 runs into B at 0x80, and the depth buffer from page 0 into A.
The interpreter settles two writes to one word by scanline order, so the
later pixel wins. The GPU unit's tiles are separate workgroups with no
order between them (`clanker/plans/wgpu-raster.md`, "one writer per
tile"), so the two writes race.

**Rule.**
- The GPU unit reproduces the interpreter where no two framebuffer
  pixels of a batch share a memory word. IRIX's layouts keep that: the
  PROM's 1280x1024 is 7 tiles (1344 pixels) a row, buffers in disjoint
  page runs, and drawing is clipped to the screen.
- Tests that compare the GPU unit with the interpreter use unaliased
  layouts (`wgpu/tests.rs` `unaliased`: rows of 960 to 1344 pixels, depth
  from page 0, A from 0x100, B from 0x200, overlay at 0x300; screen mask
  0 where slopes reach past a row). The JIT sweep keeps the aliasing
  layouts: the JIT walks scanlines as the interpreter does.
- A guest that does alias (drawing past `DRBsize`'s row width, or two
  buffers' page runs overlapping) gets one of the racing values. No
  ordering is added for it.

**How it was found.** The test's report gained "which side left the
word" (snapshot before the step) and a per-primitive check
(`MGRAS_GPU_CHECK_EACH=1`); the dumped kernel, edited to store `di`, `dj`
and the plane words as the colour (`IRIS_WGPU_DUMP` / `IRIS_WGPU_LOAD`,
no rebuild), showed the kernel's colour was right for its pixel.
