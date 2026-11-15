# VPro's refresh rate is the timing tables' totals at the clock part's frequency

Status: in use (2026-11-15). `DisplayFifo::table_totals`, `ClockParts`,
`TwoWire` and `timed_mode` (`src/dev/odyssey/display_fifo.rs`); the mode
reaches the refresh thread through `ScanoutMode::video_timing`
(`Scan::vblank_start`). `vpro status` prints it.

## Tables

- Frame table, display address 0x2400: one word a line type, repeat count in
  the high half, position in the line table (in words) in the low; a zero
  word ends it. The counts add up to the lines of a frame.
- Line table, 0x2000: 16-bit entries, two a word, low half first, in the
  VC2's scheme. A run's first entry: length in pixel clocks in bits 14:8
  (units of one clock, not the VC2's two), states in bits 6:0, bit 7 set
  when no second entry follows, bit 15 ends the line. A second entry is
  states only and takes no time.
- With that reading every line type of every format checked adds up to the
  format's line length. The earlier reading (entries with bits 14:12 = 100
  are untimed events) held only for the built-in 60 Hz timing and gave 1537
  for 1280x1024_72's 1680.
- State bit 6 marks a line of the picture: 1024 of 1066, and two runs for
  the stereo formats, which is how the fields are counted.

## Clock

The kernel programs one of two parts over the main two-wire port (display
address 0x2780), a byte per control word: bits 7:0 the byte, 8 start, 9
transfer, 11 stop, 12 send, 13 and 14 force stop and idle. A register write
is the address byte with start, the register number, its byte. Both parts
take 12 MHz.

- ICS1523 at 0x26 (board revision 0): 12 MHz x (reg 2 + 8 + 256 x reg 3
  low nibble) / (reg 7 low seven bits + 2).
- The part at 0x58 (later boards, the emulated V12): 12 MHz x (reg 4:3) /
  (reg 1:0). The kernel's built-in timing (before X loads a format) gives
  108.000 MHz; the 1280x1024_60 file 107.964476.

The mode load ends with a write to the timing generator's control entry
(0x2481), after the tables and the clock: that is when the mode is published
again.

## Checked live (Octane V12, IRIX 6.5.30, `setmon -n`)

| Format | Decoded | Retraces delivered |
|---|---|---|
| 1280x1024_60 | 1688 x 1066 at 107.964 MHz, 59.999 Hz | 60.000 Hz |
| 1280x1024_72 | 1680 x 1065 at 129.25 MHz, 72.238 Hz | |
| 1280x1024_75 | 1688 x 1066 at 134.956 MHz, 75.000 Hz | 75.000 Hz |
| 1280x1024_85 | 1728 x 1072 at 157.455 MHz, 84.999 Hz | 85.000 Hz |
| 1024x768_60 | 1344 x 806 at 64.956 MHz, 59.963 Hz | |
| 1024x768_75 | 1312 x 800 at 78.72 MHz, 75.000 Hz | |
| 1280x492_120s | 1680 x 1064, 2 fields, 60.000 frames | |
| 1280x1024_96s | 1572 x 2148 at 162.079 MHz, 2 fields, 47.999 frames | 96.000 Hz |

`1600x1200_60` and `1920x1035_30i` were refused by `setmon` on this
configuration ("No available formats match"); not looked into.

Not done: an interlaced format's two fields (none could be loaded), the
V6/V8 pass with the other clock part live, and the dual-channel formats.
