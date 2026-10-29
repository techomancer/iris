# Pointer steering: what the hardware cursor can and cannot tell

Status: design notes for `src/hostinput.rs`, 2026-10-29.

- **X acceleration on IRIX 6.5 (default `xset m`)**: a packet of d mickeys
  moves the pointer d when d <= 4, else `2d - 4` (measured: 5->6, 8->12,
  20->36, 40->76). Steps of up to 3 are exact; the steering loop learns a
  gain from larger ones.
- **The hotspot is not in the hardware.** The glyph's tip is a bad guess: the
  xdm X cursor's hotspot is its centre, and Newport adds a 3 px offset
  (`clanker/rex3/cursor-drawn-3px-right-of-x-pointer.md`). The reliable
  source is pinning in the top-left corner: X clamps the pointer to 0,0,
  so hotspot = -glyph position. The first move does this once
  (`calibrate_locked`).
- **Only the top/left edges pin reliably.** The board's visible size can
  differ from X's screen (Newport is programmed 1282 wide for X's 1280, as on real hardware), so the
  bottom/right pin position is off by that difference.
- **New shapes** are learned when they appear after an exact (<= 3) step from
  a known shape; otherwise guessed as tip + the last learned (hotspot - tip).
- **Busy guests**: during desktop start-up X ignores the mouse for seconds
  and GR2's X server turns the cursor off for a moment while drawing. The
  loop waits for queued mouse bytes to drain before sending more, tolerates
  the cursor being off for 400 ms, waits for the cursor to settle 20 ms after
  each step (the O2 shows a move a frame late; steering from the old position
  oscillated by 2-3 px), and gives up after 3 s without motion.
- **GR2 glyph**: the arrow sits in columns 16..31 of the 32x32 mask; its
  hotspot is (16,0).
