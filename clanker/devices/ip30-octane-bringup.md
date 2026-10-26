# The Octane (IP30) bring-up: the PROM to its serial prompt

Status: open (started 2026-10-19 on branch ip30 of slaapliedje/iris, rebuilt
2026-10-26 on branch ip30-machine as its own machine; the PROM's menu and
command monitor work on the serial console).

**Where it stands.** `[machine] profile = "octane_ip30"` builds an
`OctaneMachine` (`src/ip30_machine.rs`) over `Ip30Bus` (`src/ip30.rs`):
HEART, the Xbow, the Bridge with its PCI configuration space and device
windows, the IOC3's two 16550s, DS1687 and 1-wire parts, the flash, and
memory placed by MEMCFG. The QL1040s, the RAD and the graphics widget are
register files. The map every offset came from is
`docs/Octane/ip30-address-map.md`. PROM 4.17 runs its power-on
diagnostics, builds its device tree (`hinv` shows the IOC3's serial
ports, keyboard, Ethernet, both QLISP1040s and the RAD) and reaches the
System Maintenance Menu and the Command Monitor on UART A
(`serial_port_a`). `IRIS_IP30_TRACE=1` logs the first access to each
register, `=2` every IOC3 access in order; the monitor's `ip30` command
shows memory, HEART, the Bridge and the unmapped accesses.

**What the PROM insists on, found one stall at a time.**

- *Memory sizing.* The PROM programs a bank as 1 GB at 2 GB (MEMCFG
  0x819f0040) and looks for where its contents repeat: at +256 bytes,
  +128 MB, then +1 GB or +64 MB, one DIMM address line each. A rev D HEART
  (the Octane's) sizes only 1 GB and 2 GB pairs correctly from a plain
  mirror, so configure 1024 MB banks. It also probes every bank, so a
  bank with no DIMMs must take writes and read back zero (`Decode::Empty`)
  rather than bus-error; the probe places banks beyond 4 GB for a moment,
  where the 32-bit bus cannot follow.
- *The Bridge's device windows at reset are memory space at base 0.* The
  PROM enables the RAD's memory decode and writes through window 3 with
  neither the RAD's BAR nor device register 3 programmed; the window has
  to reach PCI address 0, where the unmoved BAR answers
  (`PciConfig::decode_unmoved`).
- *The Bridge status register must say PCI.* Bit 5 (PCI, not GIO) clear
  makes the PROM skip the PCI bus entirely: no IOC3 node in its tree, no
  serial device, "Cannot open serial(0) for input", "PANIC: Cannot open
  any console!". Nothing in the trace points at it; the device tree
  (dumped from the root at 0xa800000020f24040 with the monitor) did.
- *Empty PCI slots must bus-error on configuration reads.* All ones
  makes the PROM list "slot 4 vendor 0xffff part 0xffff rev 255" devices.
- *IOC3 details.* The SuperIO control register's bit 22 (no DMA request
  in flight) must read set, or the PROM spins after resetting the serial
  DMA engines; SSCR_A/B's and EMCR's reset bits (31) self-clear. The
  DS1687's address port is byte bus 1 (IOC3 + 0xa0000) and its data port
  byte bus 2 (+0xc0000), as OpenBSD's dsrtc.c has it, not the SuperIO RTC
  slot at 0x20168. The 16550s are at 0x20178 (A, the console) and 0x20170
  (B), registers at consecutive bytes.
- *The NICs.* The MAC part is a DS1981U (family 0x91): the PROM reads 14
  bytes after Read Memory at 0 (the first the CRC8 of the command) into
  nic[13..0] and takes the address from nic[2..8], so the MAC sits
  reversed at memory 5..11 as the end of a TMEX page. The board parts are
  DS2505s (family 0x0b, 2 KB): Read Status at 0x100 returns the page
  redirection table in groups of eight bytes each followed by an inverted
  CRC16 (over the command, address and bytes for the first group, the
  bytes alone after; the PROM's `nic_read_mfg` checks them: "invalid crc16
  reading redirection map"), and Read Memory returns the pages with no
  CRC8 in front. Page layout as OpenBSD's owserial.c reads it: 0x01, the
  barcode (20), nine characters of the part number; page 1: three more,
  the revision, six 0xff bytes, the name (14); each page ends in its
  inverted CRC16. The PROM uses Search ROM on each wire, so several parts
  share one `OneWire`. The system board's and CPU module's parts are on
  the Bridge's wire, the front plane's and power supply's on the IOC3's
  (with the MAC part); the PROM never drives HEART's MicroLAN master. The
  CPU board NIC diagnostic still fails: what the PROM wants from the PM10
  part is not known yet.
- *What the PROM programs in the Bridge.* IOC3 BAR0 = 0x00400000 and
  device register 2 = 0x1004 (memory space, base 0x004 << 20, no PIO
  swap); interrupt destination upper 0x09080000 / lower 0x80 (vector 9 for
  the Bridge's errors, HEART widget 8's ISR); DIR_MAP 0x00800000 (DMA to
  widget 8, offset 0); RRB even 0x1; INT_RST_STAT 0x7f; interrupt address
  registers 6 and 7 = 0x29 and 0x39 (the power switch and AC-fail
  vectors). HEART widget control 0x4008, request timeout 0x28, error type
  and mask 0x3ff7fc0f.
- *HEART's widget registers are 64-bit.* Unlike every other widget's
  (32-bit in the odd word), HEART's are addressed at 8-byte boundaries; the
  PROM reads the ID at widget offset 0 as a doubleword.
- *Byte lanes.* The Bridge presents the CPU's view of a device window:
  the IOC3, wired for a big-endian bus, has its registers at the offsets
  the CPU uses, and the PROM reaches the ISP1040's 16-bit registers at
  PCI offset ^ 2 through a window whose swap bit is clear. A
  little-endian function's model therefore takes CPU-view offsets and
  applies the ^2/^3 itself; the window's swap bit (device register bit
  13) is modelled as lane and data swapping but nothing has used it yet.
- *The device tree.* Finding out why a console device will not open is
  quickest from the tree: the root pointer is at 0xa800000020f24040, a
  node's type is at +4, key at +16, peer at +64, child at +72, and one of
  the doublewords from +24 on points at its name.

**Not done.** The CPU board NIC check; the keyboard (the PROM polls
K_RD after writing 0xf2 to K_WD and gives up, which is fine without a
keyboard); large and huge widget windows (physical 0x8_0000_0000 and up)
and memory above 4 GB need the bus widened past `u32`, which the O2's
`phys_alias` trick does not cover; SCSI needs the ISP1040 model
(`docs/Octane/qla1040.md`: the PROM's "QL SCSI controller initialization
failed" is the register file answering its mailbox); Ethernet the IOC3's
ring DMA; graphics HQ4 in front of the IMPACT emulation (the widget is
left off the Xbow's links so the PROM takes the serial console);
snapshots the usual `Saveable` work.
