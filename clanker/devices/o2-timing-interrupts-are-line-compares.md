# The O2's timing interrupts are line compares, raised by the refresh thread

Status: current (2026-11-15).

GBE has no vertical-blank interrupt. `VT_INTR01` and `VT_INTR23` hold four
12-bit lines; the beam reaching line n latches CRIME input 16+n.

What IRIX 6.5.22 programs at 1280x1024_60 (1680 x 1065 total, blank from
line 1024 = `VT_VCMAP` bits 23:12):

- interrupt 0 at line 1028, four lines into the blank;
- interrupt 1 before the picture ends; 992, 976 and 948 were seen in one
  session (the kernel moves it);
- interrupts 2 and 3 at 4095: a line past the frame never fires.

It writes two shadow registers every frame, so the shadow latch at the
start of the blank sits between the two interrupts and their order matters.

## How they are raised

The display refresh thread runs the frame. At each start of vertical
blanking `O2Display::vblank_start` calls `Gbe::retrace`: shadows go live,
the beam `VT_XY` reads is anchored at the first blank line, and the four
lines come back as lines after the retrace. They go to
`HeadHandle::set_line_events`, and `Head::line` calls
`Gbe::timing_interrupt`. `Head::line` runs on the hptimer thread
(`Refresh::set_timers`): the refresh thread arms the first line at the
retrace and each callback reschedules itself for the next, so a compose
does not delay a line; a compose that overruns delays the next retrace
and its whole sequence.

The compares are sampled once a frame, at the retrace. IRIX does move
interrupt 1 (a different line on different boots and over a session) but
300 samples, idle and under buttonfly, showed no change within a run of
frames. A guest that rewrites a compare mid-frame and expects it that
frame would need the compare evaluated per line instead. A bus without a display (unit tests, the bring-up
harness) keeps the old path, `Gbe::service` on its own clock; the first
`retrace` switches a `Gbe` over for good, and a snapshot load switches the
counting back until the next one.

`VT_XYMAX` holds the totals themselves, not the last index: the beam and
the crossings count modulo 1680 x 1065, the same frame the refresh rate is
decoded from.

## Measured (interpreter, 25-30 guest MIPS, idle desktop)

| | on the instruction count (before) | on the refresh thread |
|---|---|---|
| timing interrupts in 10 s | 67 of each | 602 of each |
| retraces in 10 s | 602 | 602 |
| interrupt 0 after the retrace | unrelated | 0.071 ms, p99 0.082 (4 lines: 0.063) |
| interrupt 1 after the retrace | unrelated | 15.780 ms, p99 16.069 (line 960: 15.766) |

## The hang that kept them on the instruction count

The O2's first commit ran the beam on the CPU's instruction count, with a
comment that on the host clock IRIX's interrupt dispatcher found GBE
latched again before each pass ended and never returned. Nobody recorded a
reproduction. At 25 MIPS on the interpreter IRIX boots to the desktop and
runs with both interrupts at 59.6 Hz (and a 1 kHz clock tick beside them),
so the hang is taken to have been an early CRIME latching bug, fixed since.
If it comes back, look at CRIME's acknowledge path, not at the rate.
