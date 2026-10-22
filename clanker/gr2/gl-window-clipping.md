# GR2 GL window clipping comes from 0x1E5, per context

GL windows are clipped by the kernel's 0x1E5 packet (window, wid, obscured,
pieces, up to 4 rectangles; format in HQ2.h "0x1E5 HQ2_GL_WINDOW").

Gotchas found the hard way:

- The piece count decides, not the obscured flag. A partly covered window
  arrives with `obscured = 0, pieces = 2`. Gating the rectangles on
  `obscured` left atlantis unclipped (its clear and fish drew into the window
  in front).
- The packet only arrives when the kernel switches into the context after the
  clip changed (Gr2PcxSwap re-sends on a clip-generation mismatch), not while
  the window is being dragged. A trace that starts after the windows were
  arranged has no 0x1E5 at all: the clip is already in the context's saved
  state. To see one, capture across a window move plus a few frames.
- `pieces > 4`: no rectangles; the HLE turns on the RE3 WID test (window-id
  planes painted by Xsgi's 2D_CID_WRITE).
- `obscured = 1, pieces = 0`: the one rectangle the kernel sends is the
  window's bounding box, not its visible region. WID test too, bounded by
  the box. Treating it as a visible rectangle let twilight's root-window
  background paint over every desktop window (the draw is 24-bit RGB, so
  all planes of those windows went).
- Clipping is at span level in `gl_tri` (`GlState::span_pieces`): colour,
  alpha and Z iterators must be started at each piece's left end, or shading
  jumps at the cut (`gl_clip_to_visible_pieces` checks this).
- Symptom of missing clipping on a 12-bit double-buffered GL window: odd flat
  colour (the clear) and GL geometry inside the X window in front, because
  X's 8-bit CI pixels share VRAM bits 7:0 with GL buffer 0.

- The pieces are XOR'ed, not unioned: a window with a hole (C shape)
  arrives as [whole window, covered rectangle], 2 pieces, obscured = 0,
  wid = 0 (Xsgi DDX exp_window.c expValidateClip; full table in HQ2.h).
  The union let atlantis draw under ideas (flicker) whenever ideas covered
  the middle of atlantis's side; L shapes (disjoint pieces) were fine, which
  hid it. `span_pieces` pairs the sorted piece edges of each row;
  `visible_rects` bands the XOR for clears. Test: `gl_clip_pieces_are_xor`.
- Window origin words in 0x1E5 are signed (off-screen windows).
- Open: in the WID-test case (>= 5 visible rectangles) the wid sent is the
  RRM window id (0x10 in traces); we compare `wid & 0xf` with the 4-bit CID
  planes, i.e. 0, the CID of plain X windows. Probably wrong; check what
  rrmValidateCID paints for that window before trusting WID clipping.
