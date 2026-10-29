# glGetTexImage of ALPHA, LUMINANCE, INTENSITY and RGB textures is libGLcore's, not the board's

Status: rule (IRIX 6.5.22, 2026-10-28).

**Symptom.** texsweep's `texread` lines report differing bytes for most
sized formats: ALPHA textures read back `ff000000` (alpha in red), RGB,
LUMINANCE and INTENSITY with alpha 0. The output is byte-for-byte the same
on an IP28 High IMPACT, an Octane SI with texture and an MXI.

**Cause.** libGLcore asks the TE for the texture's own components (XFRMODE
format 9 for one, 0xA two, 7 RGB, 8 RGBA; the DMA lengths match) and then
expands them on the host by component count alone: one component to red,
two to red and alpha, anything missing 0. GL 1.1 would give (0,0,0,A),
(L,0,0,1) and (R,G,B,1). Details in the local `ignore/gr4/GE11.h`.

**Rule.** Do not "fix" the model's texel readback (`Te1::read_line`) to
produce GL's answer: it returns what the board returns, and libGLcore's
expansion is what an application on real IRIX sees. Counts that are GL-
correct: RGBA formats 0, rgba2/rgba4 176, luminance-alpha 120 (plus 4-bit
rounding).
