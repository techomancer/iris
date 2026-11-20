# Octane PROM: loading an 11 MB ELF from the volume header overruns a DMA list

Status: open (PROM 4.17 as iris runs it, 2026-11-15). Not established whether
real hardware does the same; network boot is the workaround.

**Symptom.** `boot -f dksc(0,1,8)vmlinux` with a Linux 7.3 kernel of 11.5 MB
(one PT_LOAD of 0xaf3e30 bytes at 0xa800000020004000) prints the size and
then `Read address error exception, bad address: 0x40001f400008` at PROM PC
0x9fc422a4, followed by `PANIC: driver_acquire: io_state.active deadlock`
for ever. The same file loads and runs from `bootp()vmlinux` (iris's NAT
TFTP). The SCSI trace ends with READ(10) of the ELF header (block 8, one
block): the crash comes while the PROM issues the segment's read.

**What was found.** The PROM polls its SCSI adapters through a list whose
head is at 0xa800000020f17e40; the node for controller 1 at 0x20f0e9a0 had
next = 0x...20f0dfc0, base = 0x900000001f400000 before the boot and read
`0x00000000a0128000, 0x000040001f400000, 0x00000000a012c000,
0x0000400000050000` after it. Those are 16-byte records `(0, PCI address,
0x4000)` whose PCI addresses are the Bridge's direct window (0x8000_0000 +
physical 0x2012_8000, +0x4000, ...): a DMA page list for the 11 MB segment,
one record per 16 KB page, written past its buffer into the adapter node.
No CPU store to the node hit a physical write breakpoint, and iris's cache
model here is the shadow model (no dirty lines to lose), so the list is
written by the PROM (through the fast store path) or by a DMA.

**Open.** What sizes that list on real hardware, i.e. whether the PROM
splits a large read by something iris answers (INQUIRY, MODE SENSE pages,
the ISP1040 firmware's limits) or real Octanes cannot load a kernel this
large from the volume header either.
