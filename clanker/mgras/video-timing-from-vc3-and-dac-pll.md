# IMPACT's refresh rate is the VC3's totals at the DAC PLL's clock

Status: in use (2026-11-15). `Dcb::video_timing` (`src/dev/mgras/dcb.rs`),
decoded at a retrace after any display-bus write and handed to the refresh
thread (`HeadHandle::set_timing`). `mgras vc3` prints it.

## Totals

The VC3's video timing tables have the VC2's layout
(`disp::decode_vc2_totals`): register 0 points at the frame table, pairs of
(line table address, repeat count) up to a zero count; a line is runs whose
first word holds the duration in bits 14:8, in units of two pixel clocks,
bit 7 says there is no second word, bit 15 of either word ends the line, and
the word after the line is the address of the line that follows.

- 1280x1024_60: line durations sum to 840 (1680 clocks), repeat counts to
  1065. Frame table at 0x380, line tables from 0.
- A visible line has bit 5 of every run's first word set, a blank line never
  (1024 of 1065). The VC2's composite-blank bit (second word bit 6) is set
  on blank lines too on the VC3, which is why `Vc3::timing_size` reads 1065
  visible lines (clamped to 1024) and a loose width.
- Fields a frame = runs of visible lines: 2 for 1280x492_120s, the `_2f`
  formats and 1280x960_30i.

## Pixel clock

DAC registers 0x09 (control), 0x0C (R), 0x0F (V):

    clock = 33 MHz * (4 (V + 2) + control[7:6]) / ((R + 2) * post)
    post  = 1, 2 or 4 by control[5:4], doubled by control[2]

Fitted, not read from the part's datasheet: IRIX's format files
(`/usr/gfx/ucode/MGRAS/vof/*.sdb`) carry `DacPll_cntl`, `DacPll_r`,
`DacPll_v` and `AchievedFrequency`, and the formula gives all fifteen distinct
clocks exactly (49.5 to 159.92 MHz; the table is the unit test). Control bit 0
is set in every file.

The kernel's writes of these registers land in the model's register file at
`reg << 8` (0x900, 0xC00, 0xF00), the PROM's writes of the others at `reg`:
the two send the 16-bit address with its bytes the other way round.
`Dac::pixel_clock_hz` reads either. The pixel mask (4) and command 1 (5) are
at both 0x4 / 0x5 and 0x400 / 0x500 for the same reason, and the model reads
the mask at 4 only. Open: whether IRIX ever blanks the screen through the
mask (it would write 0x400, which nothing reads).

## The format files

`.sdb`: a symbol table near the end (a count, then for each symbol its name
length, the name padded to four bytes, a type, a zero and a file offset;
type 2 is a 32-bit integer, 0x11 a double), values before it.
`TotalPixelsPerLine`, `TotalLinesPerFrame`, `FieldsPerFrame` and
`AchievedFrequency` (MHz) are what the decode must reproduce.
`FramesPerSecond` is nominal: 1024x768_60 says 59.963 with totals that give
60.031.

## Checked live (Indigo2 High IMPACT, IRIX 6.5.22)

`/usr/gfx/setmon -n <format>` loads a format at once, with no X restart.

| Format | Decoded | Retraces delivered |
|---|---|---|
| 1280x1024_60 | 1680 x 1065 at 107.25 MHz, 59.942 Hz | 59.943 Hz |
| 1280x1024_72 | 1680 x 1065 at 129.25 MHz, 72.238 Hz | |
| 1280x1024_76 | 1712 x 1079 at 140.25 MHz, 75.923 Hz | 75.925 Hz |
| 1024x768_60 | 1304 x 813 at 63.643 MHz, 60.031 Hz | 60.032 Hz |
| 1024x768_76 | 1336 x 812 at 82.5 MHz, 76.048 Hz | |
