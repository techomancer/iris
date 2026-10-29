# The Octane's IMPACT microcode keeps ERAM state 0x8C words higher

Status: rule (IRIX 6.5.22, measured 2026-10-28). Check again on 6.5.30.

**Symptom.** On the Octane, pixtest failed 314 of 486 cases on IRIX 6.5.22:
glDrawPixels of RGB images came out shifted by one component (8-byte RGBA16
pixels read from 6-byte RGB16 rows), and glGetFloatv(GL_COLOR_MATRIX_SGI)
read zeros. The same binaries pass on an IP28 with 6.5.22.

**Cause.** libGLcore and the GE11 microcode put state where the microcode
wants it, and the HQ4 build moves some blocks:

| What | Indigo2 (HQ3) | Octane (HQ4) |
|---|---|---|
| GL state from the stipple enables up (viewport 0xB91, clear values, ...) | I2 address | +0x8C |
| pixel pipeline ERAM blocks (expand state 0xD05, transfer size 0xDA8, texture load 0xDB1-0xDBB) | I2 address | +0x8C |
| pipeline program (0x13B), module states in GE memory (0x187-0x1EF), zoom (0x191) | same | same |
| colour matrix readback (0x1B9-0x1C8) | same | same |

The GE11 program is identical on both 6.5.22 boards (fingerprint
`5467b0da1d106495`); only the HQ microcode differs (`mgras status`).

**Rule.** `Hq4Quirks` translates to the Indigo2 layout on the way in:
`ge_state` for readbacks, `pixel_state_address` for PIXEL_STATE /
INIT_PIXEL_STATE_ERAM and for the module state addresses in a program write
(`Hq3Engine::canonical_pixel_state`). The pixel HLE and `PixMem` only know
the Indigo2 layout. A new release's layout should be told apart by the
microcode fingerprint (`src/dev/mgras/microcode.rs`), not by the board.

**Also.** IRIX base 6.5 on the IP28 reads viewport, scissor, clear values and
raster state from addresses the 6.5.22 map does not answer (glstate: 31
failures). Not mapped yet.
