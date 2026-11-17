# Texture colour tables come as device writes outside any pixel operation

Status: rule, 2026-11-17. Modelled in the interpreter, the raster JIT and
the wgpu unit; matched against purple (real High IMPACT) with `texlut`.

**Symptom.** `glColorTableSGI(GL_TEXTURE_COLOR_TABLE_SGI, ...)` plus
`glEnable(GL_TEXTURE_COLOR_TABLE_SGI)` changed nothing on the emulated
IMPACT; a real board maps the texels.

**What the driver sends.** libGLcore loads RE4's three 256-entry tables as
`RSS_REG_SHADOW` (0x07C) triples right after `RESTORE_RSS`, outside any
pixel operation: `DEVICE_ADDR` 0x15C = 0x9000_0000 once, then 768
`DEVICE_DATA` 0x15D words (R, then G, then B, 8 bits each). The address is
never resent: the device auto-increments. Smaller tables come expanded to
256, L tables copied into all three, RGBA tables without alpha. The enable
is TEXMODE1 bits 15:13 (all three, whatever the formats).

**Why it was dropped.** `hq3.rs` keeps `RSS_REG_SHADOW` triples sent
outside a pixel operation as GE shadow only (they are a context's raster
state), apart from the TE registers `Gl::te_write` takes. The two device
registers now go straight to the raster engine from `te_write`: the tables
are board state (RE4 RAM), not context state, and a `Gl` copy would not
fit the kernel's context slot anyway (`KERNEL_SLOT_WORDS`).

**The lookup** (`te1::tex_lut`, `Rss::textured`, `rss_jit` `tex_lut`,
`te1.wgsl` `tex_lut`): after filtering, before the environment. RGB texels
per channel; luminance and intensity through R only (the result stays
luminance); alpha textures not at all; index = the component's top 8 bits.
The JIT key has one bit (`Tex::lut`); which tables are on is context data
(`GlCtx::lut_mask`). Making room for that bit meant packing the dependent
texture flags (mip mode, clamp group) into shared digits: the key was
within a factor 1.7 of 64 bits.

**Not yet.** Fog through the shared table; alpha (unobservable on the RGB
visual texlut used).

**Seen alongside, separate.** GL_MODULATE rounds differently from the
board by one level: colour 0.5 is 0x80 on the board, 0x7f here, and the
board rounds products up (1 x 254/255 -> 1, here 0). Visible in every
texlut case, table on or off.

**Test.** `texlut [on|off] [size N] [env replace|modulate] [tex l|la|rgb]
[fmt rgb|rgba|l]` (a 256x64 ramp, prints readback pixels);
`te1::tests::texture_colour_tables_follow_the_board` pins purple's values.
GL_REPLACE reaching the board as modulate is libGLcore, not us.
