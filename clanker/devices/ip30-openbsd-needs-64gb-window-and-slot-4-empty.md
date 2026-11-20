# OpenBSD 6.9 on the Octane needs the Bridge's 64 GB window and an empty slot 4

Status: confirmed 2026-11-15 with OpenBSD 6.9 (the last release with the sgi
port), `bsd.rd.IP30` netbooted from PROM 6.5 Rev 4.9, two R10000s, 512 MB, the
full card load (PCI module with ISP2200 and NVMe behind DEC 21154 bridges,
empty shoehorn, virtio-serial and virtio-9p).

## 1. The kernel reaches the IOC3 through the 64 GB window

Symptom: `bootp()bsd.rd.IP30` loads, prints `Found SGI-IP30, setting up.`,
then the PROM's exception handler reports a Data Bus error at
`ip30_setup+0x298` with `v0 = 0x900000f040500048`.

Cause: OpenBSD's `ip30_setup` (sys/arch/sgi/sgi/ip30_machdep.c) builds the
early console on `ip30_widget_long(0, 15) + BRIDGE_PCI0_MEM_SPACE_BASE +
0x500000`, i.e. physical `0xF0_4050_0000`: widget 15's 64 GB window (`w << 36`),
XIO offset `0x4000_0000` (the 32-bit PCI memory alias), PCI address 0x500000
(the IOC3 BAR the PROM programs). `xbridge(4)` then maps every onboard BAR and
the option boards' BARs through the same window (`xbridge_mapping_setup`).
iris mapped only the small and 2 GB windows of a Bridge.

Fix: `Bridge::huge_window` (`XioWindow` with a 36-bit mask) attached by
`Ip30Bus::attach_huge_window` for every Bridge; BAR runs in the 32-bit alias
are mapped in both the 2 GB and the 64 GB window. The 64-bit alias
(0x8000_0000) and PCI I/O (0x1_0000_0000) decode through the Bridge.
`xbridge(4)` uses direct I/O space only on Bridge revision 4 and later
(`XF_NO_DIRECT_IO`), so on the onboard Bridge (rev 3) and the PCI module (rev 3
placeholder) I/O BARs go through device windows; only the shoehorn (rev 4)
would use the I/O alias.

## 2. ioc(4) takes the SuperIO interrupt from the first empty slot

Symptom: the kernel's polled output is fine, then userland's console output
stops after 16 bytes (`erase ^?, werase` from the installer's `stty`): the
tty waits for a transmit interrupt that never comes.

Cause: `xbridge_intr_map` gives the onboard IOC3's interrupt pin D (the
SuperIO: both UARTs, keyboard, parallel) "pin A of the first empty PCI slot",
which is slot 4 on a real Octane. With virtio-serial in slot 4 and virtio-9p
in 5, OpenBSD listened on Bridge pin 6, while iris drives the SuperIO on pin 4.

Fix: keep slot 4 empty. The defaults are now virtio-serial in slot 7 (on
INTB) and virtio-9p in 5, which also keeps Linux's console alive; with
`bridge_slot`, any placement that leaves slot 4 empty works.

## What else OpenBSD finds

`qlw0/1` (ISP1040, "nvram corrupt" is harmless), `ioc0` with `com0` console,
`iec0` + `icsphy0`, `dsrtc0`; `xbridge1` (shoehorn, widget 10, empty),
`xbridge2` (PCI module, widget 13) with `ppb0`/`ppb1` ("DEC 21154"), `qla0`
(ISP2200) with both loop disks as `sd1`/`sd2`. No driver for the NVMe
(GENERIC-IP30 has no `nvme`), virtio (none on sgi) or the RAD1.

## Booting it unattended

- The installer puts `boot` (boot64) in the volume header. PROM settings:
  `OSLoader=boot`, `OSLoadFilename=bsd`, `SystemPartition=dksc(0,1,8)`,
  `AutoLoad=Yes`, and `OSLoadPartition` in the ARCS form
  `xio(0)pci(15)scsi(0)disk(1)rdisk(0)partition(0)`: the kernel matches its
  root disk against that path (sys/arch/sgi/sgi/autoconf.c); with
  `dksc(0,1,0)` it prints `boot device: 'dksc(0,1,0)' unrecognized.` and stops
  at a `root device:` prompt.
- `shutdown -p now` ends in `System Power Down.` and the machine's soft
  power-off: iris exits.
- `halt` on the two-processor kernel is not clean (open): the PROM takes over
  on CPU 0 while CPU 1 still runs OpenBSD's clock interrupt, whose lightbar
  update reads the IOC3 GPIO register at PCI 0x50003c; that master-times-out
  (Bridge `PCI_MST_TIMEOUT`, address 0x50003c), CPU 1 takes a data bus error
  into the PROM's handler, and after the restart the PROM finds the NVRAM
  checksum wrong and resets the environment. OpenBSD's `ds1687_settime` leaves
  the DS1687 in bank 1 (it never clears register A's bank bit), which is a
  candidate for the checksum. Not chased; keep a golden NVRAM copy.
- A flood ping (`ping -f`) of the gateway reports one round trip of about
  -1000 ms among 20000 (min -999.6 ms). The `heart` timecounter iris serves is
  monotonic; not chased.

## Guest gotchas (not iris)

- `/var/tmp` is a link to `/tmp`, which `rc` empties at boot.
- The installer's `encrypt -b a` (bcrypt with rounds calibrated to the CPU)
  for a root password took 16 minutes with the JIT off.
- `iris-ci serial-send` appends a CR already; sending `"x\r"` answers two
  prompts.
- An IRIX volume header on a disk limits OpenBSD's `c` partition to the size
  it records.
