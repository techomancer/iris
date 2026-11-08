# VPro line rasterisation, measured on a real V10

Status: measure / open (smoothed lines), 2026-11-07.

**Tool.** `test/gltest/linetest.c` (`sh irix.sh linetest`).
- Run with no argument, it draws 8 angles (0 to 90 degrees, 60-pixel
  lines from fractional starts) in 6 cases: aliased widths 1, 2 and 3;
  smoothed widths 1 and 2, white with GL_SRC_ALPHA, GL_ONE onto black, so
  red reads back as coverage; and an aliased line shaded red to green.
- `linetest profile` draws smoothed horizontal lines at 64 offsets of
  1/64 pixel, at widths 1, 1.5, 2 and 3.
- Octopus output is in `ignore/linetest/octopus.txt` and
  `octopus-profile.txt`. `ignore/linetest/analyse.py` compares them with
  the model's pixel rule.

**Aliased lines: the model is right.** 2,532 of 2,534 pixels match over
all angles and widths. The one miss is at 15 degrees, width 2: one pixel
moved.
- One pixel per major-axis column, the one whose span holds the line at
  the column centre (`centre_minor`, 8 subpixel bits).
- A column is drawn when its centre is in [start, end).
- A wide line is replicated along the minor axis, shifted by
  (width-1)/2. Its ends are flat and perpendicular to the major axis.

**Smoothed lines: the model is wrong.** It uses box-filter area coverage
of the width x length rectangle (`line_coverage`). The board instead:
- Uses columns along the major axis, with the same [start, end) rule. The
  ends are flat with no fade, and there are no caps.
- Within a column, coverage depends only on the minor-axis distance dv
  from the pixel centre to the line at the column centre: 255 for
  |dv| <= (width-1)/2, then one falloff table over the next pixel. The
  table is the same for widths 1, 1.5, 2 and 3, in 1/64 steps, and
  slightly asymmetric between the sides (+1: 248 / -1: 245, +32: 139 /
  -32: 139, +64: 3 / -63: 6). The full table is in
  `ignore/linetest/octopus-profile.txt`.
- Each column carries about the same total at any angle (width 1: about
  264-277), so total intensity against geometric area falls toward 45
  degrees (0.78 there). The area model gives 1.0 everywhere.
- Near 45 degrees the pixel nearer the line on the lower side reads 7
  below the table, and a third pixel 1.3 away reads 6, keeping the
  column's total. This is probably the "diagonal intensity" the marketing
  material mentions. Its rule is not decoded yet; at 15 and 26.6 degrees
  the table matches to within about 2.

**Next.**
- Model smoothed lines as the column walk plus the table, and decode the
  near-diagonal spill with 45-degree and 30-60 degree offset sweeps.
- Then smoothed lines can join the integer walk and the GPU kernel.
- Case 5 (shaded) is the reference for the line colour arithmetic. It
  has not been compared with the emulator yet, because the emulated guest
  was not running.
