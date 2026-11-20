# A second Newport is found only in GIO slot 0, and on an Indy only from board revision 4

Status: confirmed live 2026-11-15 (Indy, IRIX 6.5: `hinv` lists two boards,
both windows draw). Indigo2 dual head not re-tested after the move.

## Symptom

`graphics.heads = 2` created the second REX3, but neither the PROM nor IRIX
listed it.

## Cause

Three things had to agree with the guest, and none did.

- **Address.** IRIX's Newport driver has a two-entry board table: REX3 at
  `0x1F0F0000` (graphics slot) and `0x1F4F0000` (expansion slot 0). We had
  the second board in slot 1 (`0x1F6F0000`), which the kernel never probes.
- **Board revision.** On an Indy, the PROM and the kernel both skip the probe
  of any board after the first unless SYS_ID's board revision (bits 4:1) is
  at least 4. We report 3 (`0x26`). Nothing else on an Indy depends on the
  value beyond a "2 or later" check, so a dual-head Indy reports 4.
- **Interrupt.** On an Indy the driver registers board 0 on GIO interrupts 0
  and 2 of the graphics slot (FIFO, retrace) and board 1 on the single pin of
  its expansion slot: `GIO_EXP0`, map status bit 6. Retrace and FIFO of the
  second board are therefore OR-ed onto that pin (`IrqLine::wired_or`). We
  had retrace on `GIO_EXP1` and FIFO on the graphics slot's FIFO line.
  On Indigo2 both boards use GIO interrupts 0 and 1 of their slot through the
  EXT_IO fan-out; the shared local-0 bit must be the OR of the slots.

The handler is the same for every case: it reads STATUS once and acts on
VRINT, the video bit and the FIFO bits, so any registered line will do.

## Rule

- Slot 0 is shared with the Ultra64 board and `test_device`: two heads turn
  the first off and refuse the second; virtio devices go to slot 1.
- A unit test cannot probe the board by writing a REX3 register and reading
  it back: writes go through the GFIFO and the drawing thread is not running.
  Read STATUS instead.

## The second board is an "NG2" with its own video table

IRIX calls board 1 "NG2" and programs it differently for the same
1280x1024_60: pixel clock 107.27 MHz (107.5 on board 0), 1680 pixels a line
(1682), 1296 unblanked (1282), the picture starting at x = 8 (2), and 30
pixels from the HPOS edge to the first unblanked one (40). So head 1's window
is 1296 wide with 8 black pixels each side; that is the guest's doing. The
30-pixel gap makes the cursor adjustment -1, which the decoder used to refuse.

With two heads the monitor has numbered commands (`rex1`, `vc21`, `bt4451`;
`rex0` and so on for the first), head 1's thread is `REX3-Processor.1` and
its locks `rex3.1::*`.
