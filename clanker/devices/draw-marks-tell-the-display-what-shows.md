# A draw marks its planes and box; the display composes only what shows

Status: in use on IMPACT and VPro (2026-11-15). `src/display/marks.rs`,
`mgras/disp.rs` `marks_show`, `odyssey/frame.rs` `Frame::shows`.

## Why

On both boards a window's back buffer is a field of the same pixel as its
front buffer (IMPACT: the other half of a 12-bit pair, or a second page;
VPro: another byte range of the pixel). "Memory was written" therefore marked
every frame of a double-buffered program as changed, and the display
composed 550 to 600 frames of 606 to show 114 to 148 swaps.

## Rule

- The raster thread marks after it stores: the store (IMPACT: buffer page
  and kind; VPro: surface base, pitch and pixel size), the planes of a pixel
  it can change (IMPACT: the plane mask where the pixel type stores through
  one; VPro: a bit per byte of the pixel), and a box (IMPACT: the keep-inside
  screen masks; VPro: the command's box inside its tile).
- Once a frame the display takes the marks and asks, for each, whether a
  window ID inside the box displays that store and reads one of those
  planes. Display-state changes (the kernel's flip at the retrace, colour
  maps, the cursor) are damage of their own.
- A path that cannot name planes and box says "anything": `Marks::mark_all`
  on IMPACT (host GL composites), the plain written flag on VPro (every
  store that is not a tile worker's).
- IMPACT: a command that keeps the engines across a whole frame counts as a
  change (`Scan::consumed`): the X server's full-screen fill of the root
  window takes about 50 ms as one command and marks only when done.

## Checking it

`IRIS_DISPLAY_VERIFY=1` composes every frame a board would have left and
compares (`src/display/verify.rs`). "missed" is a change with no mark for two
frames running; "late" is one the next frame's verdict reported, which is a
draw caught between its stores and its mark.

2026-11-15, IRIX 6.5.22 / 6.5.30, boot to desktop, atlantis, window drag,
`xrefresh`: IMPACT 7800 verdicts, 0 missed, 9 late; VPro 18600, 0 missed, 7
late.

## Measured (atlantis, 10 s, headless, same host)

| | Before | After |
|---|---|---|
| IMPACT composes / retraces | 550 / 606 | 156 / 655 (156 swaps) |
| IMPACT display thread busy | 80% | 20% |
| VPro composes / retraces | 603 / 606 | 155 / 666 (156 swaps) |
| VPro display thread busy | 39% | 9% |

IMPACT's frame was a snapshot of the planes and display state (8 ms) and a
compose from it (6 ms). A thread that mostly sleeps is clocked down by the
host, and the pair then took 20 ms cold, longer than a frame. The display's
frames are now composed straight from the pixel pages
(`frame::compose_direct`: no copy between, a span of one window ID at a time
through tables that already hold the gamma, the overlay skipped a word at a
time where it is 0), as VPro's are: 6.5 ms cold, done 7 ms after the retrace.
At 76 Hz the retrace intervals stay within 12.92 to 13.43 ms (1st to 99th
percentile) with atlantis swapping. `Frame::snapshot` and `Frame::compose`
remain for screenshots, the monitor and as the reference the direct compose
is tested against.

## Found on the way

- IMPACT's DAC gamma table is written through select 1, which did not count
  as a display change; it does now.
