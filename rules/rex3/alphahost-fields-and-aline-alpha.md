# REX3 source alpha: host fields and antialiased lines

Two places where the source alpha does not come from COLORALPHA. Both were
invisible while BLENDALPHA was misread (see `blendalpha-and-alpha-blending.md`):
with BF_SA forced to 1.0 a wrong alpha only changed the destination term.

## ALPHAHOST without COLORHOST: the host field is the alpha

Spec §3.9: "ALPHAHOST=1 with COLORHOST=0 specifies the HOSTRW1,0 alpha fields
are to be used to blend the DDA R,G,B components". §3.10: each value sits in a
field of 8, 16 or 32 bits chosen by HOSTDEPTH, leftmost first.

IRIX's OpenGL and IRIS GL draw antialiased points this way: one I_LINE GO per
pixel, DRAWMODE0 `0x8a` (ILINE ALPHAHOST), HOSTDEPTH 0 (4-bit, so 8-bit
fields), unpacked, and the CPU-computed coverage in the top byte of a
32-bit HOSTRW0 write (`0xdd000000`, `0x22000000`, ...). The colour comes from
COLORRED/GRN/BLUE. Unpacking that field as a 4-bit colour gives alpha 0.

For 8-bit fields (host depths 4 and 8) the whole field is the alpha. 32-bit
fields are ABGR, alpha in the top byte, as before. Where an alpha sits in a
16-bit field (host depth 12) is not stated and has not been seen in a trace.

## A_LINE: alpha is coverage

IRIX's OpenGL draws GL_LINE_SMOOTH as A_LINE (`0x00440b32`: DOSETUP, SHADE,
SKIPLAST, ENDPTFILTER) with blending SA/MSA and never loads COLORALPHA for it:
the register still holds whatever the previous primitive left. The spec's line
algorithms write each pixel with "alpha represents pixel coverage" (§3.6), the
coverage coming from the AWEIGHT tables. IRIS does not model coverage, so
A_LINE pixels count as fully covered (alpha 255): solid, not antialiased.
`blast` draws its HUD lines as A_LINE with blending off, so it is unaffected.

## How these were found

`rex buslog on` in a `developer` build (GFIFO register log), around a small GL
program that draws one primitive of each kind, compared against the same
program on the emulated XZ.
