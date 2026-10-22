IRIS GL software-Z lines: ZPATTERN reload and BRESROUND

Symptom: fsn (IRIS GL) label text drawn in fragments, strokes missing or a
row off; the same shape of damage in other GL apps' small line text.

How IRIS GL draws a Z-buffered line on Newport (fns.log, `rex trace`): the
CPU does the depth test and each stroke becomes
  XSTARTF YSTARTF XENDF YENDF        fractional endpoints
  DRAWMODE0 = 0x0000930e             F_LINE STOPONX STOPONY ENZPAT LEN32, no DOSETUP
  SETUP = 0                          setup without iterating
  ZPATTERN GO = 0xe0000000           one mask bit per pixel, MSB = first pixel
with BRESRNDINC2 = 0x001f.... (BRESROUND byte 0). Glyph strokes are 1-4
pixels, so every stroke is far shorter than the 32-bit pattern.

Two bugs, both fixed:

1. ZPATTERN is a recirculating register, "msb = first pixel": writing it
   reloads it. We kept the value plus a bit index (`zpat_bit`) that only
   DOSETUP and the end of a span row reset, so a stroke's mask was tested
   from wherever the previous stroke stopped (bit 28 after a 3-pixel stroke;
   bit 0 on a fresh board). Spans (polygons) never showed it: a full 32-pixel
   chunk wraps the index back to 31 and row ends reset it. A ZPATTERN write
   now sets zpat_bit = 31.

2. BRESROUND (BRESRNDINC2[31:24], one bit per octant, msb = spec octant 1;
   hardware octant {XMAJOR,XDEC,YDEC} 5,1,3,7,6,2,0,4 = spec 1..8) picks the
   step when d == 0: set = diagonal (d < 0 goes straight), clear = straight
   (d <= 0 goes straight). We ignored it and always behaved as "set". IRIS GL
   loads 0, the X server 0xff. It decides lines that pass exactly through a
   pixel-row boundary at a column centre (a 2x1 I_LINE starts at d == 0).
   It first looked like the cause of horizontal F_LINEs dropping a row; that
   was fault 3's y mirroring, and those lines no longer tie. Both engines now
   honour it (`bres_straight_on_zero`; JIT: `round_set_v` in
   emit_draw_iline). The generated shaders call the generic code, so they
   follow automatically.

3. F_LINE/A_LINE setup put pixels a row or column off (strays sticking
   out of glyphs, and the spotlight showing floor through the label where
   IRIS GL's Z buffer said text was). Four separate faults in setup():
   - slope from the integer endpoint deltas; the spec uses the fixed-point
     ones (a 2.5 x 1.44 stroke became 2 x 2);
   - the spec's octant table has y pointing UP (its octant 1 = hardware
     octant 5, YDEC set): its y fraction is 16 - the register's. Applying
     the table to the register fraction mirrored y the wrong way round in
     every octant;
   - the fractional term was rounded (>> 4); the errata's BRESD is s18.8,
     so d, incr1, incr2 are now kept in 1/256 pixel (fits the 20/21/27-bit
     fields for any on-screen line);
   - no first-pixel adjustment at all: the first pixel is the start
     point's own pixel. The pseudocode's E test and the errata's "E and G
     tests" both move it, and both are wrong on this REX3 (see below).
   And the pixel count must be along the octant's major axis, not
   max(|dx|, |dy|) of the integer deltas (they disagree near 45 degrees).
   Result: the start point's pixel, then in each column of the major axis
   the row holding the true line at the column's centre (pixel n covers
   [n, n+1)), moving at most one row per column. The test oracle
   `fline_geom` is that rule, written from geometry, not from the code; a
   Python model of the setup matched it on 29547/29547 random lines (exact
   row-boundary ties excluded: BRESROUND's call).
   Ground truth: fsn draws its labels before the spotlight disc, and the
   disc spans are masked by IRIS GL's software Z at the pixels GL computed
   for each stroke. Where our stroke pixel differed, the floor colour
   showed through inside the disc. Replaying fns.log (trace -> recording)
   and counting those holes found and confirmed each fix.
   The first-pixel rule needed more: fsn.rec (a `rex rec` of fsn redrawing
   two labelled files), replayed on the interpreter with every stroke's
   endpoints and pixels logged, plus every later soft-Z span pixel with its
   mask bit. A 0 bit = IRIS GL thinks something nearer (the text) is there.
   On the strokes where candidate rules put the first pixel in different
   places and the next span's bits disagree: start point's own pixel 113/119,
   pseudocode E 87, errata-style E/G (centre rule for the first pixel too)
   6; on later pixels 35/36. The losers are 1-pixel strokes next to others.
   Errata-style E/G gave the "position dependent" stubs: a 45-degree stroke
   evaluated half a pixel behind its start lands in the neighbouring row.

Tests: test_zpattern_write_restarts_pattern_for_soft_z_lines (fsn's stroke
sequence), test_bresround_decides_the_d_zero_step (both settings, both
engines), test_fline_fractional_both_ends_matches_geometry (2000 random
lines, both engines), test_fline_fsn_label_stroke.

MAME is no reference here: its line code ignores ZPATTERN and steps x/y in
1/16-pixel units, which hides any wrong first step.

Also found with fsn.rec (REX3_REPLAY_ENGINE=interp vs jit, bisected with
REX3_REPLAY_LIMIT, diffed with REX3_REPLAY_STATE): the Cranelift shader ended
a SPAN like a BLOCK row (x back to XSAVE, y + 1); a span leaves XSTART one
past XEND and YSTART alone (spec §3.5.3, draw_span, MAME). And at a row end
it left a disabled pattern's bit position stale where the interpreter
restarts both at 31. Only shapes not in the precompiled table run through
Cranelift, so this showed up in one span of 4M records (IRIS GL reads
XSTART/YSTART back). Both engines now replay fsn.rec to the same state.

Not covered: A_LINE is still one pixel per step at full coverage (the spec
draws two, T and S, weighted from AWEIGHT), and its ENDPTFILTER drops end
pixels by a guessed rule instead of scaling coverage. fsn uses it rarely
(16 GOs in fns.log); check blast's HUD before assuming it is the same bug.
