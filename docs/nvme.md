# NVM Express controller

iris can fit one NVM Express controller in the O2, or in an Octane's PCI module
or shoehorn. It is written
from the NVM Express Base Specification (revision 1.4) and the PCI
specifications; it models the specification and nothing vendor-specific. A
guest needs a driver for it: IRIX has none of its own, and Linux and the BSDs
carry theirs.

```toml
[nvme]
path = "nvme.img"     # raw or CHD; the card is present when this is set
overlay = true        # writes go to nvme.img.overlay, as for [scsi.N]
bridge = false        # true: behind a PCI-to-PCI bridge in the slot
board = 0             # Octane: which [[xio_board]] (the first by default)
slot = 1              # Octane: its PCI device (the first free by default)
```

On the Octane the card needs a board to sit in, since its onboard Bridge keeps
only its built-in devices:

```toml
[[xio_board]]
kind = "pci-cage"     # or "pci-shoehorn"; see docs/pci.md

[nvme]
path = "nvme.img"
slot = 2              # the module's slots are PCI devices 1 to 3; a shoehorn has 1
```

## Where it sits

| Machine | Slot | Interrupt |
|---|---|---|
| O2 | "PCI option", device 3 | INTA, CRIME input 10 |
| Octane | Slot `d` of a PCI module or shoehorn's Bridge | INTA, that Bridge's pin `d` |

The Octane's boards are a Bridge of their own on a Crossbow port
([docs/pci.md](pci.md#option-boards-in-xio-slots)): its pins go to HEART, and
IRIX finds the card in the hardware graph under that port's `pci` node.

With `bridge = true` the slot holds a PCI-to-PCI bridge (type 1 header) and
the controller is device 0 on the bus behind it. Firmware numbers the bus and
opens the bridge's memory window; a pin of the card reaches the bridge's pins
by the usual swizzle ([docs/pci.md](pci.md)). The card's DMA is the slot's.

## What it does

- Controller registers in a 64-bit memory BAR0 (8 KiB): CAP, VS 1.4, INTMS,
  INTMC, CC, CSTS, AQA, ASQ, ACQ, and doorbells with a 4-byte stride.
- Admin commands: Create and Delete I/O Submission and Completion Queue,
  Identify (controller, namespace, active namespace list, namespace
  descriptors), Get Log Page (error information, SMART / health, firmware
  slot), Get and Set Features, Abort, Async Event Request.
- NVM commands: Read, Write, Write Zeroes, Flush on one namespace of 512-byte
  blocks, PRP data pointers, up to 128 KiB a command.
- Up to 16 I/O queue pairs of 2 to 1024 entries, physically contiguous
  (CAP.CQR). Queue 0 first, then the others in turn.
- Pin-based interrupts only (the PCI core has no MSI): the pin is asserted
  while a completion queue with interrupts enabled holds an entry the host has
  not read and INTMS bit 0 is clear. Vector 0 is the only vector.

Anything else is refused with the status the specification gives it. The
monitor's `nvme refused` lists the opcodes and features a guest asked for that
were refused; `nvme status` shows the registers, the queues, the counters, the
admin commands run and each time the controller failed (CSTS.CFS) with the
last reason; `nvme cow ...` is the copy-on-write layer's commands for the
namespace. The counters and failures outlive a reset.

## Resets

A Controller Reset (the host clearing CC.EN) returns every register and
queue to its power-on value except AQA, ASQ and ACQ, which keep what the host
wrote (NVMe 1.4 section 7.3.2). Those three are plain storage at any time and
are read when CC.EN is next set. RST# on the machine's PCI bus (on the
Octane, HEART's reset and the monitor's `reset`; the O2's resets reach no
PCI function yet) clears them too. Either reset waits for a command being run to finish, so nothing
from before it reaches memory afterwards. A guest that reboots without
resetting the machine finds the controller as it left it, as it would on a
card that saw no RST#; the host then clears CC.EN and waits for CSTS.RDY to
fall (section 7.6.1).

## Tracing

`log nvme on` in the monitor, or `IRIS_DEBUG_LOG=nvme` (`[debug] debug_log`)
from power-on to stderr, traces what the host does to the controller and what
it does back. Each line starts with `t=` and the microseconds since the
controller was made; on the Octane HEART's counter runs on the same host
clock. `log nvme mask` takes `state` (CC writes, CSTS changes, resets and
fatal errors with their reason), `admin` and `io` (doorbells, commands and
their completions) and `regs` (every register access below the doorbells),
or a hex mask of 1, 2, 8 and 4 for those. `log nvme file PATH` writes to a
file.

## Approximations

- Enable, disable and shutdown complete at once; CAP.TO (5 s) is only the
  bound a host may wait. CSTS.RDY is set by the CC write that sets CC.EN.
- Interrupt coalescing is stored and reported but not applied.
- The PCI Express and Power Management capabilities are present and
  read-only.
- The namespace is at most 2 TiB: the disk layer addresses blocks in 32 bits.
- The namespace GUID is made from the image path.
- A doorbell value outside its queue is ignored and counted; no asynchronous
  error event is raised.
- The PCI vendor and device numbers (1b36:0010) are the ones QEMU's emulated
  controller uses.

## Storage

The namespace is a disk of the shared SCSI target layer, so images, CHD,
copy-on-write overlays, the background sync and the `cow` commands are those
of [STORAGE.md](../STORAGE.md).

The namespace reports a volatile write cache (Identify Controller VWC is 1),
and `[nvme] write_cache` sets whether it powers up on. The default is `true`,
as for `[scsi.N] write_cache`: a plain Write completes once the host has the
data, Flush and a Write with FUA store what came before them, and the disk
layer's background sync follows about 5 s after a write. A host crash or power
loss can lose that much. With `false`, or `--no-write-cache`, every write is
on the host's disk before it completes. STORAGE.md says what each setting
loses.

Get and Set Features, Volatile Write Cache, read and change the setting the
way a MODE SELECT of the caching page does for a SCSI disk. Turning the cache
off first stores what it holds. Get Features with the default selector returns
what the card powers up with, and a controller reset puts the setting back to
it. The setting is not saved in a snapshot: a restored card has what
`write_cache` says.

A snapshot saves the controller's state and the overlay with the other disks,
and records the disk's size under slot 240 in its manifest.
