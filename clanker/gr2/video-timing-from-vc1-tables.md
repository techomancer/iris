# GR2's refresh rate: the VC1's table totals at one of two known clocks

Status: in use (2026-11-15), checked at the PROM only. `Vc1::totals`,
`debug::pll_hz`, `Gr2::vblank_start` (`gr2disp.rs`). `gr2 vc1` prints it. The
table format is also in `ignore/gr2/VC1.h`.

- Frame table at VID_EP (0x800, or 0x900 for the cursor-workaround set): a
  word, the lines of a frame, then three bytes a line type (line-state table
  address, repeat count) up to an address of 0xFFFF. PROM 60 Hz: 0x0429
  (1065), and the repeats add up to it.
- Line-state table: runs of state bytes, the last with bit 7 set, then a
  byte whose low seven bits are the run's length in units of five pixel
  clocks and whose bit 7 ends the line. Two line types of the PROM's table
  both add up to 336 units, 1680 clocks: exactly 60 Hz at 107.352 MHz.
- Clock: only the two PLL programmings in GR2.h are known (107.352 and
  132 MHz). Any other, and the early boards' byte-written clock generator,
  leaves the timing as it was (60 Hz).
- Blank lines: the frame's lines less the window-ID frame table's extent.

Not checked: the 72 Hz tables and stereo, which need IRIX on an XZ (the
scratch configuration here boots the PROM only). Fields are always 1.

O2, for the record: `Gbe::video_totals` takes VT_XYMAX as the totals (the
PROM writes 1680 and 1065, the totals of IRIX's 1280x1024_60) at the dot
clock, 20 MHz x 16 / 3 = 106.67 MHz, so 59.62 Hz. The O2 composes every
frame (GBE reads RAM with no mark to test). That took about 25 ms through a
copy of the frame's pages and a per-pixel renderer, so retraces came in
bursts; `Gbe::render_direct` scans RAM in place through per-entry tables in
3.7 ms (PROM screen, same pixels), and the retrace is steady. `Gbe::render`
stays for screenshots and the monitor, and as the reference in the test. Its
vertical-timing interrupts are still raised by its service thread on guest
time, which this does not change.
