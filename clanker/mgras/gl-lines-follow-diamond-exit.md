# IMPACT GL lines follow OpenGL's diamond exit, minor offset and all

Status: rule, 2026-11-09 (real High IMPACT, Indigo2 IP28, IRIX 6.5.22;
`test/gltest/linesweep.sh`, checked offline by `mgras_tests.rs`
`real_board_lines` against `testdata/real_high_impact_lines.txt`).

**Symptom.** glprim's line loop (vertices at x 12.8 / 51.2, y 9.6 / 38.4 in
a 64x48 window): the board lit each segment's start pixel and the model
did not, so every corner lacked a pixel and every stipple pattern ran one
pixel late. At even widths (2, 4, 10) the board, unlike at odd widths,
did *not* draw the start pixel; polygon-mode lines dropped the apex pixel.

**Rule** (`Rss::gl_line_setup`, shared by the interpreter, the JIT and the
GPU unit): a pixel is drawn when the line leaves its diamond within
(start, end]. With d the line's distance from the pixel centre across the
major axis, the diamond's far corner is d short of the pixel's far edge
(x + 1 going right, x going left). d is taken at each end. A wide line is
judged by its spine, which for even widths sits half a pixel off the
centres (d from `b + 0.5`). The old rule, pixel centres in [start, end),
is the same only when d = 0.5.

Also on the board:
- Colours are evaluated at each pixel centre even when the first centre
  lies before the start vertex (extrapolated, then clamped): the start
  pixel shows the start colour exactly, the next one t = 0.7 along.
- The stipple counter runs across the whole strip or loop, one step per
  pixel drawn (we already did; the missing pixel made it look shifted).
- Widths round half up (1.5 -> 2, 2.5 -> 3, 3.5 -> 4).
- GL_LINE_SMOOTH only antialiases width-1 lines; wider smooth lines are
  exactly the aliased line of the rounded width.

**Antialiased width-1 lines** (`gl.rs` `smooth_line_window`, fitted to
`linetest` profile, vprofile, slopes, vslopes and its eight-angle case;
`real_board_smooth_lines_and_points`, 1-5 levels mean):
- every pixel whose centre is within half a pixel past either end along
  the major axis and less than a pixel from the line across it;
- coverage = profile(|minor distance|) x sqrt(1 + slope^2) x the end
  ramps clamp(u + 1/2, 0, 1) (u: how far inside the line from that end);
- the profile is measured, and differs by major axis: x-major peaks at
  171/255 and bends down (two pixels either side sum to about 0.8), y-major
  peaks at 201 and falls nearly linearly; y-major lines lose a little more
  near 45 degrees (x (1 - 0.1 slope^2)); 45 degrees counts as x-major.
The GE draws them pixel by pixel as one-pixel lines with alpha scaled.

Points ride on the same lines: a point is a horizontal line from the left
edge of its first pixel to the right edge of its last (`gl.rs`
`point_window`), and all nine aliased sizes match the board. Antialiased
points: size 1 is a one-pixel antialiased line from the point one pixel
right (exact); larger ones are discs with alpha S(r - d), the integral of a
tent 0.9 pixels either side of the edge, over the pixels floor(c) - ceil(r)
to floor(c) + ceil(r) (which lights the far side's last pixel and not the
near side's; 1 level mean). Open: in glprim (64x48 window, three points
in one glBegin) sizes 3 and 6.5 lack the near side's whole ring on the
board, up to 105 levels from the model: a size-3 point on a pixel corner
is a 3x3 box there (unblended: all 255), 4x4 in linetest. Not the hint
(`linetest pointsdc` is identical), not the visual or the blend function
(glprim variants on the board, results/purple-lf/ptvar). Left: the
window, several points per glBegin, the clear between linetest's draws;
an `mgras trace` of both in the emulated guest would show what libGLcore
sends.
