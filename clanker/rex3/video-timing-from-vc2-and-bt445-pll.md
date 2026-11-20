# Newport's refresh rate is the VC2's totals at the Bt445 PLL's clock

Status: in use (2026-11-15). `Vc2::totals`, `Bt445::pixel_clock_hz`,
`Rex3::head_vblank_start`. `bt445 status` prints both and the timing.

- Totals: `disp::decode_vc2_totals` over the VC2 RAM, decoded when a RAM
  write or the video entry pointer moved `Vc2::timing_gen`. A line is in the
  picture while the composite blank to the XMAPs (second word, bit 6) is off.
  1280x1024_60 on a revision 0 board: 1682 x 1065, 1024 visible. Tables with
  no vertical blank are ignored: the host's `[graphics] resolution` presets
  load such a stand-in.
- Clock: 20 MHz x M / ((N + 1) x 2^P); M is rate register 0, N the low six
  bits of rate register 1 and P its top two. Fitted: the format files
  (`/usr/gfx/ucode/NG1/vof/*.u`, text) end with "M,N,P" and name the clock in
  whole MHz in their first line's last field; eleven settings from 12.27 to
  140 MHz all agree within the rounding, and the registers were read back
  after `setmon` for three of them (43,7,0 -> 2b 07; 28,3,0 -> 1c 03;
  51,7,1 -> 33 47). The datasheet in `docs/datasheets/indy` is a scan with
  no text layer and was not read for this.
- Format file layout: line 1 `table address, width, height, rate, MHz`;
  line 2 the frame table; line 3 the line table; then M,N,P and seven DAC
  setup bytes. `_r3` and `_r4` variants are for later board revisions.

Checked live (Indy, NG1 revision 0, IRIX 6.5.22, `setmon -n` as the logged-in
user: at the xdm login screen it cannot open the display):

| Format | Decoded | Retraces delivered |
|---|---|---|
| 1280x1024_60 | 1682 x 1065 at 107.5 MHz, 60.011 Hz | 60.011 Hz |
| 1280x1024_72 | 1692 x 1064 at 130 MHz, 72.210 Hz | 72.211 Hz |
| 1280x1024_76 | 1710 x 1079 at 140 MHz, 75.877 Hz | 75.877 Hz |
| 1280x1024_50 | 1682 x 1065 at 89.23 MHz, 49.812 Hz | 49.813 Hz |
