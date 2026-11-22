# IOC status bits shared by several sources are the OR of their levels

Status: fixed 2026-11-19 (issue techomancer/iris#262), unit-tested and
checked end to end by `tools/ioc-shared-bit-e2e.py`; the Indy second-head
mapping is still an open question (below).

## Symptom

Two sources on one IOC status bit overwrote each other. When one deasserted
while the other still asserted, the bit dropped and the second source's
interrupt was lost until it rose again. On an Indigo2 every GIO slot's
`GioSg*`/`GioS0*`/`GioS1*` source sets the same `FIFO_FULL`, `GRAPHICS` or
`VERTICAL_RETRACE` bit; the direct `FifoFull`/`Graphics`/`VerticalRetrace`
sources share them too.

In a configuration IRIS builds today, the case left after 36bb98dc is an
Indigo2 with virtio-serial or virtio-9p in GIO expansion slot 1 (their
default): the device drives `GioS1Graphics`, which shares L0 `GRAPHICS` with
the graphics board (Newport's retrace, IMPACT's general interrupt). Each
Newport retrace release cleared the bit under a pending virtio interrupt,
and a virtio acknowledgement cleared it under a pending retrace. The e2e
harness reproduces both in every iteration on a build without the fix.

## Cause

`IocState::set_source` set or cleared the bit from the caller's level alone.
36bb98dc already derived the bit from the graphics slot's and slot 0's EXT_IO
bits, but slot 1 (no EXT_IO bit) and the direct sources still wrote the bit
straight, and a PORT_CONFIG retrace clear for one slot dropped the bit for
both.

## Rule

- `IocState.levels` holds one bit per `IocInterrupt` line.
  `IocInterrupt::status_bit` names the register and bit a source drives;
  `set_source` updates the level, the source's EXT_IO bit, and sets the
  status bit to the OR of every source that shares it (`SHARERS`, built at
  compile time).
- A guest acknowledgement that clears a latch drops that source's level
  through `IocState::release`, not by clearing the status bit: GEN_CNTL bit 0
  (Indy GR2, `VerticalRetrace`) and PORT_CONFIG `CLR_SG/S0_RETRACE_N`
  (`GioSgRetrace`/`GioS0Retrace`). The bit then stays up while another
  source still asserts.
- The 8254 timer bits (MAP 0/1) are latches set by the timer callback and
  cleared by TMR_CLR; they do not go through levels.
- Snapshots save `levels` and `ext_io`. An older snapshot loads with every
  level low and the status bits as saved; the first source on a bit to
  change sets it from the levels, which is how the old code behaved.

## Open: which pins a second head on an Indy uses

The maintainer's comment on #262: the second head on an Indy needs its own
interrupts, but the documents are unclear and nobody has measured it on a
real multihead Indy. IRIS keeps the second Newport on its slot's single pin
(`GIO_EXP0`, MAP bit 6), with retrace and FIFO joined by
`IrqLine::wired_or` (see [second-newport-needs-slot0-and-board-rev-4.md](second-newport-needs-slot0-and-board-rev-4.md)).
That is where IRIX registers the handler, and dual head works with it, but
it is not confirmed against hardware. On the Indigo2, the maintainer says
the slots' bits are set together by design and IRIX acks them together,
multiplexed by EXT_IO; that is what the OR models. Testing real Indy and
Indigo2 multihead to pin the exact mapping is still to do.
