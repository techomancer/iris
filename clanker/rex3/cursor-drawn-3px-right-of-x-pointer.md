# Newport draws the hardware cursor 3 pixels right of the X pointer

Status: open, measured 2026-10-29 on IRIX 6.5 at 1280x1024 (REX3 decodes
the mode as 1282x1024, `cursor_x_adjust` = 5).

**How it was seen.** `ptr calibrate` pins the X pointer at 0,0 and reads
the VC2 cursor position as the compositor draws it. xdm's X-shaped cursor
(cursor font hotspot 7,7) pinned with its glyph at (-4,-7); the 4Dwm arrow
(hotspot 1,1) at (2,-1). Both say the glyph is drawn 3 px right of where X
put it and exactly right vertically. On GR2 the arrow pins with its tip on
0,0: no offset there.

**Effect.** Cosmetic in the window and VNC (the arrow tip is 3 px right of
the real pointer). `ptr`/VNC clicks are unaffected: they learn hotspots in
drawn coordinates, so negative hotspots are allowed.

**The 1282 width is not ours.** The PROM and IRIX program the VC2 timing
wider than the 1280-pixel X screen, and the X server scissors one column on
the left to black: probably a workaround for a display hardware quirk. So
the decoded width is right and should stay. The cursor offset may be tied to
the same quirk (`cursor_x_adjust` and its -2 bias in `decode_vc2_timings`);
check against a real Indy before changing it.
