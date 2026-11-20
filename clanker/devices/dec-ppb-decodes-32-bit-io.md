# The DEC bridge in front of a card decodes 32-bit I/O

Status: rule, 2026-11-15.

## Symptom

Octane with the PCI card cage and `[fibre_channel] bridge = true`: IRIX 6.5.30
prints `WARNING: *** ppb IO base/limit conflict` at boot, programs the
bridge's I/O window as 0x0000..0xffff (0x1c = `f000`, 0x30 = 0) and puts the
ISP2200's I/O BAR at 0x200000, outside it. An I/O cycle to the BAR ends in a
master time-out on the cage's Bridge; touched from the monitor, IRIX panics
on the Bridge error interrupt. The FC disks' memory BAR is unaffected, so
nothing else showed it.

## Cause

`PciToPciBridge::new` built a plain type 1 header: I/O base and limit read 0
in bits 3:0 (16-bit I/O) and the upper 16-bit registers at 0x30/0x32 read
zero. The cards' bridge reports DEC `1011:0026`, the 21154, whose family
(21152/21153/21154) decodes 32-bit I/O: Intel's 21153 configuration note
(278070-001, tables 2 and 3) gives 1Ch/1Dh a reset value of 01h, the upper
16-bit registers, and a 4 GB I/O range at 4 KB granularity.

## Fix

`PciToPciBridge::with_io32_windows` (`PciConfig::type1_io32`) for the NVMe
and ISP2200 `bridge = true` slots. With it IRIX reads the capability, writes
0x30 = `0x0020_0020` (window 0x200000..0x20ffff) and the warning is gone.

Two things the change also needed:

- `PciBus::config_write` rebuilt the bus in front only on dwords 1 and 4..9.
  A bridge's upper halves (10..12) and bridge control (15, ISA enable) feed
  `forwards()` too, so a write to 0x30 alone left the window stale. Type 1
  headers now refresh on those dwords as well.
- A saved header from before keeps 0x1c's low nibbles at 0; `load_state`
  re-asserts the read-only capability.

ISA enable (bridge control bit 2): below 64 KiB the window forwards only the
low 256 bytes of each 1 KiB block (PCI-to-PCI Bridge Architecture 1.2, bridge control; the
21153 note, section 8). A BAR there wider than 256 bytes is not forwarded at
all, by the existing whole-range rule.

How other OSes read it: Linux `pci_read_bridge_io` takes the upper halves only
when 0x1c's type is `PCI_IO_RANGE_TYPE_32` and probes the window with a word
of `0xe0f0`; NetBSD `pciconf.c` disables a 16-bit bridge's I/O window and any
BAR above 0xffff behind it; OpenBSD `ppb.c` reads and writes 0x30 without
checking the type.

Not modelled: the 21154's 64-bit prefetchable window (24h reads 01h on the
real part; here it stays 32-bit, upper halves zero) and VGA enable.
