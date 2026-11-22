# Running mainline Linux (7.3) on the Octane

Status: howto (Linux 7.3.4, PROM 4.17, 2026-11-16). Boots to a shell on two
CPUs with the onboard SCSI, IOC3 serial and Ethernet, the cage's ISP2200 and
NVMe behind 21152s, virtio-serial and virtio-9p.

**Kernel.** `ip30_defconfig` plus: W1, W1_MASTER_SGI, W1_SLAVE_DS250X (the
Bridge driver defers its probe for ever until it can read each board's
part number from the one-wire NIC); RTC_DRV_DS1685_FAMILY; DEVTMPFS;
`qlogic/1040.bin` built in for qla1280 (it calls request_firmware before any
root exists); qla2xxx wants `ql2200_fw.bin`. The PROM loads the ELF64
`vmlinux` (load address 0xa800000020004000) from `bootp()`, and from the
volume header (`dksc(0,1,8)vmlinux`) once `mkvh build` has split its segment
into 512 KB pieces (`ip30-prom-disk-load-of-a-large-elf-overruns-a-dma-list.md`).

**Kernel-side problems, not iris's** (fixed with a patch to
`arch/mips/pci/pci-xtalk-bridge.c`):

- The driver sets `PCI_PROBE_ONLY` and claims what firmware assigned, and
  assumes PCI bus address = small-window offset. The PROM gives BaseIO
  windows 2 and up a base 1 MB below their offset (IOC3 BAR 0x500000 through
  window 2 at 0x600000) and leaves an option board's Bridge at reset values
  with nothing assigned behind its 21152s. Pointing each Device(x) window at
  its own offset (except the one the PROM console's IOC3 uses; `prom_putchar`
  hard-codes widget 15 + 0x600000) and calling
  `pci_assign_unassigned_bus_resources` after the claim makes the cage work.
- The root-bus scan goes to device 31; device 8 of type 0 space is the type 1
  aperture, so with the type 1 register at bus 0 the unnumbered 21152 claims
  the cycle and a phantom ISP2200 appears at 00:08.0 (a bridge with
  secondary = subordinate = 0 does claim a type 1 cycle for bus 0; iris is
  right). Stop the scan at `BRIDGE_DEV_CNT`.

**iris fixes it took:** ISP2200 Login Loop Port and
request-ring mailbox entries
(`scsi/isp2200-linux-logs-in-through-ring-mailbox-entries.md`); virtio-pci
offering ACCESS_PLATFORM; virtio-serial off slot 4's interrupt pin
(`virtio-serial-design.md`, a separate change).

**Other behaviour to expect.** No software power-off (`poweroff` halts; the
DS1685 driver powers down only on the power button). The 21152's I/O window
lands at bus 0x200000, which the CPU cannot reach (all Bridge windows are
memory space); qla2xxx runs on the memory BAR. ext4 sends WRITE SAME(10), which the SCSI disk refuses
(noted, harmless). `make depend` of OpenSSL 1.0.2 takes hours without the JIT.
