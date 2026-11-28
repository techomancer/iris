# QLogic ISP2200 Fibre Channel adapter

iris fits a QLogic ISP2200 Fibre Channel host adapter in an Octane's PCI
module or shoehorn, or on the I-brick of an Origin 3200 or Onyx 3000, with
a private loop of SCSI disks behind it. IRIX 6.5.30
brings it up with its own driver and shows the disks as a SCSI controller of
their own: the adapter is one controller with one bus, however many disks are
on the loop.

```toml
[[devices]]
type = "pci_cage"        # or "pci_shoehorn"; see docs/pci.md
name = "cage"
at = "pci_module"

[[devices]]
type = "qla2200"
name = "fc"
at = "cage/pci0"         # the first free slot; cage/pci0/2 names one

[[devices]]
type = "hdd"
at = "fc/loop5"             # a loop ID, 0 to 125
path = "frame0.raw"      # raw or CHD
overlay = false
```

A loop disk is an `hdd` with the same options as one on a SCSI bus (`path`,
`overlay`, `write_cache`). Only plain disks go on the loop. A disk with no
slot goes on the first free SCSI target before it goes on a loop. The adapter
takes the lowest loop ID no disk holds, or the hard address the guest asks for.
The disks report the machine-wide disk identity in INQUIRY, as a SCSI disk
does. `iris --list-slots` shows the loop's slots under the card.

Put the card behind a PCI-to-PCI bridge with a `pci_bridge` device and
`at = "bridge/pci0/0"`.

On Origin 3200 and Onyx 3000 the adapter is already fitted on the I-brick, at
`xio15/pci0/1`. Put its disks at `xio15/pci0/1/loopN`. The I-brick's slots are
`bus1-slot2`, `bus1-slot3` and `bus2-slot1`, `bus2-slot2` (the guide's
labels; their PCI device numbers are inferred, and the adapter stands for
`bus1-slot1`). They take PCI cards, not NVMe yet. See the
[Origin 3200 guide](origin3200-ip35.md).

## What the guest sees

The slot, interrupt and DMA are those of any card on an option board
([docs/pci.md](pci.md#option-boards-in-xio-slots)): INTA on the Bridge pin of
the slot's device. It is PCI vendor 1077, device 2200. Its monitor commands are
`scsi2` and `cow2` (the Octane's own two controllers are 0 and 1), which act on
the loop's disks as the other controllers' do; `scsi2 notes` lists what a guest
did that the model does not decode. A snapshot saves the adapter and each
disk's overlay with a supported Octane snapshot. Origin 3200 and Onyx 3000
whole-machine snapshots are not supported.

## What it models

The host interface and nothing behind it: a driver sees mailboxes and two rings
of entries, and the firmware it downloads is kept for its checksum and never
run. Fibre Channel itself is not modelled, with no frames, logins or loop
initialisation. A disk is the one port at its loop ID, and the guest sees loop
IDs, names and SCSI commands, which is all the firmware shows a host. No
programming manual for the part was found; the register, mailbox and ring
facts come from the public Linux, OpenBSD and NetBSD drivers and from how the
IRIX driver uses the part. Where those are silent the model makes a choice,
and the comment in `src/dev/isp2200.rs` says so.

## Limits

- IRIX 6.5.30 and Linux 7.3 (qla2xxx, with the stock `ql2200_fw.bin`) have
  been tried. Linux logs in to each port with Login Loop Port and with
  mailbox commands it puts in the request ring (login, logout and the port
  database for ADISC); the model answers those four in a ring entry and
  refuses any other command sent that way.
- Commands run one at a time. An abort or reset matches the command running,
  not the ones queued behind it.
- A disk attached or removed after start is not announced; there is no loop
  event for it. The loop is up 100 ms after initialisation or a LIP.
- A disk has one LUN. A command to another LUN up to 255 is answered as for a
  LUN the disk lacks, and one past 255 is refused. REPORT LUNS is not answered.
- Registers take 16-bit accesses; a byte access is dropped.
- Fast-post and point-to-point are refused. Loop down is never reported.
- Five mailbox commands come from IRIX alone and have no public definition, so
  their replies are made up to keep that guest on its success path
  (`IRIS_DEBUG_LOG=scsi` marks each). Any other command the model does not
  implement answers 0x4001.
- The NVRAM image and its serial (hashed from the adapter's number) are
  invented, as are the firmware version and the checksum algorithm.
