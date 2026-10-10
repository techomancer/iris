# STEPZ: a GO that steps over one pixel

rex3.pdf's register table: `0x0034 STEPZ` "Enables ZPATTERN (Z test fail) for
one iteration, (current pixel)". The value written does not matter; the GO
does. That GO iterates like any other (XSTART, the shade DDAs and the
patterns advance), but its pixel fails the Z pattern test, so it is not
written, or is written in COLORBACK under ZPOPAQUE, as for any Z pattern miss.

IRIX's software rasteriser on Newport uses it to skip pixels inside a span.
Textured, alpha-tested polygons (blast's billboards, `blast -T` on an XL) are
drawn one GO per pixel: DRAWMODE0 `0x40022` (DRAW SPAN DOSETUP SHADE, no
STOPONX), `ZPATTERN` written 0, then for each pixel either an ordinary GO or a
GO on STEPZ for a texel it has rejected (transparent, or alpha-tested away
under DRAWMODE1 compare 5, `alpha != ALPHAREF`).

Treating STEPZ as a plain register write plus GO drew every rejected pixel:
blast's billboards showed their transparent corners as smeared rows and solid
white triangles.

Emulation (`Rex3::execute_stepz_go`): the GO runs with ENZPATTERN forced on
and only the current pattern bit cleared, so every later pixel of the same GO
passes. The guest's ZPATTERN is put back afterwards, and so is the pattern bit
position when the guest had not enabled the pattern itself. Running it as an
ordinary ENZPATTERN GO means the interpreter, the precompiled shaders and the
REX JIT all handle it with code they already have.

Test: `stepz_go_skips_its_pixel` (rex3_tests.rs) draws a span one pixel per
GO, every other GO on STEPZ, with and without ZPOPAQUE.
