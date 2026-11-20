# GR2's refresh rate: the VC1's table totals at the PLL's clock

Status: in use (2026-11-15). `Vc1::totals`, `debug::pll_hz`,
`Gr2::vblank_start` (`gr2disp.rs`). `gr2 vc1` prints the timing and
`gr2 status` the clock. The table and clock formats are also in
`ignore/gr2/VC1.h`.

## Tables (VC1 SRAM, byte addressed)

- Frame table at VID_EP (0x800, or 0x900 for the cursor-workaround set): a
  word, the lines of a frame, then three bytes a line type (line-state table
  address, repeat count) until the repeats reach that word. What follows is
  not part of the frame: the PROM's table ends with 0xFFFF there, IRIX's
  with `00 ff ff` and a few more entries.
- Line-state table: runs of state bytes, the last with bit 7 set, then a
  byte whose low seven bits are the run's length in units of five pixel
  clocks and whose bit 7 ends the line.
- A line is in the picture when the last state byte of one of its runs has
  bits 6, 1 and 0 set (0x43). Inferred from the counts it gives.

## Clock

The seven bytes bit-banged to the clock port (boards of revision 4 on):

    clock = 11.375 MHz x (6 (b4 + 1) + b5) / (b6 + 1)
            doubled when b1 = 0x00, halved when b1 = 0x20 (0x10: as is)

Fitted: with it, every one of IRIX's fourteen format files
(`/usr/gfx/ucode/GR2/vof/*.u`: line tables, two frame tables at 0x3c0 and
0x7e4, the clock bytes in the last eight) gives its named rate from its own
totals. The "132 MHz" table of GR2.h makes 130 MHz.

| File | Clock | Totals | Frame rate | Picture lines, fields |
|---|---|---|---|---|
| 60hz | 107.352 MHz | 1680 x 1065 | 60.000 | 1025, 1 |
| 72hz | 130.000 | 1690 x 1064 | 72.296 | 1024, 1 |
| 50hz | 89.449 | 1680 x 1065 | 49.994 | 1025, 1 |
| 48hz | 107.352 | 1680 x 1331 | 48.009 | 1025, 1 |
| 30hz | 53.676 | 1580 x 1131 | 30.037 | 1024, 2 |
| st492 | 107.352 | 1680 x 1064 | 60.056 | 1022, 2 |
| ntsc | 12.273 | 780 x 525 | 29.971 | 487, 2 |
| pal | 15.003 | 960 x 625 | 25.005 | 578, 2 |
| ccir50 / ccir60 | 97.754 / 118.590 | 1680 x 1164 / 1850 x 1065 | 49.989 / 60.191 | 1025 / 1024, 1 |
| iris3k / presenter | 63.984 | 1320 x 813 / 803 | 59.622 / 60.365 | 768 / 769, 1 |
| rs343a | 49.720 | 1620 x 1023 | 30.001 | 961, 2 |

The early boards' byte-written clock generator is not decoded: the timing
stays as it was (60 Hz).

## Checked live (Indy XZ, GR2 revision 4, IRIX 6.5, `setmon -n` from a
logged-in session)

| Format | Decoded | Retraces delivered |
|---|---|---|
| 60HZ | 1680 x 1065 at 107.352 MHz, 60.000 Hz | 60.000 Hz |
| 72HZ | 1690 x 1064 at 130 MHz, 72.296 Hz | 72.296 Hz |
| 50HZ | 1680 x 1065 at 89.449 MHz, 49.994 Hz | 49.994 Hz |
| STR_RECT | 1680 x 1064, 2 fields | 120.11 Hz |

## The O2, for the record

`Gbe::video_totals` takes VT_XYMAX as the totals (the PROM writes 1680 and
1065, the totals of IRIX's 1280x1024_60) at the dot clock, 20 MHz x 16 / 3 =
106.67 MHz, so 59.62 Hz. The O2 composes every frame (GBE reads RAM with no
mark to test). That took about 25 ms through a copy of the frame's pages and
a per-pixel renderer, so retraces came in bursts; `Gbe::render_direct` scans
RAM in place through per-entry tables in 3.7 ms (PROM screen, same pixels),
and the retrace is steady. `Gbe::render` stays for screenshots and the
monitor, and as the reference in the test. Its vertical-timing interrupts
are still raised by its service thread on guest time.
