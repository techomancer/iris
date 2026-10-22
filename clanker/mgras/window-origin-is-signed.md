# The GL window origin's halves are signed

XYWIN and the GL window's origin (`Window::origin`, from CP_WINDOW and the
context image: y in the high 16 bits, x in the low) are two **signed** 16-bit
values. A window that reaches past the screen's top or left edge has a
negative one: Quake III at 1280x1024 on a 1280x1024 screen, its window taller
than the screen with the title bar, gets origin 0xffe00008 (x 8, y -32).

Decode them with `rss::signed16`, never `as i32` on the masked or shifted
word. Read unsigned, y was 65504: `Gl::clear_block`'s rectangle landed off
screen, clamped to the scissor box it was empty, and the clear was dropped.
The colour clear is usually drawn over and hides it; the depth clear does not.
The depth buffer kept the previous window's values, so the 3D view showed
only inside the old window's rectangle, and stale frames piled up elsewhere.

How to spot it: geometry that depth-tests vanishes outside some old
rectangle while depth-less drawing (HUD, 2D) still appears there, and `mgras
gl` shows an origin with a high (or low) half of 0xff... (A colour clear does
not help to see it: it goes through the same `clear_block` and was dropped
too.)
