# PCI

The O2 (through MACE) and the Octane (through the onboard Bridge) have PCI
buses. Both hosts own a `PciBus` (`src/pci/`), which holds the functions on
the bus, decodes their BARs, combines their interrupt pins and resets and
saves them. A host decides only what is its own: how a CPU access becomes
configuration cycles and byte order, where an interrupt output goes, and what
a function's DMA reaches. The bus has no byte order and knows no card by name.

## Functions

A function implements `PciFunction` (`src/pci/function.rs`): BAR reads and
writes, `reset` (RST#), and a `PciFunctionState` that holds its configuration
header and, once the bus attaches it, its interrupt line and DMA. The bus resets
a function's header; a function never resets its own, so the decode table cannot
go stale behind a BAR that moved. Headers change only through the bus: a
configuration write, a reset or a snapshot load, and each updates the decode
table and the BAR mappings. A function whose chip
computes a configuration register (the indirect windows of a NIC) overrides
`config_dword` for the read, and `config_write` to take a whole-dword write
itself, which the header then never sees.

`PciBus::attach` gives the function its DMA and, if its header names an
interrupt pin, the line that pin drives, then calls `attached` for a function
that builds on them. A device number the wiring lacks, an
address already taken, or a pin the wiring does not route is a bug in the
machine and panics at attach.

## Byte order

A little-endian function behind a big-endian CPU needs a rule for how bytes
move, and the two hosts differ. A host chooses a `ByteOrder` value per window
(`src/pci/host.rs`): `Stream`, or `Values { width }` for a bus 4 or 8 bytes
wide. MACE is `Values { width: 4 }`: a byte access at CPU offset `a` reaches the
function at `a ^ 3`, a halfword at `a ^ 2`, a word as given (O2 Device Driver
Programming Guide 007-3443-002, pp. 397-398, Table 15-3). That guide does not
cover 64-bit accesses; iris sends one as two word cycles, the high word first,
and the rule is unknown. The Bridge takes its order from the window's Device(x)
register: the programmed swap bit gives `Stream`, and its 64-bit device bit
gives the width. A `BarDevice` the physical bus maps over a host window holds
its `ByteOrder` by value, so a BAR access is one match and the function's own
call.

## Interrupts

Every device pin and every board IRQ is a line routed to a host output, and
the bus tells the host the OR of what is routed to each output. A pin that
falls and rises before the host has seen the fall still reaches an edge host
as a 0 and then a 1: the Bridge sends its interrupt packet to HEART only on a
rising edge, and the guest may have cleared the bit in between. A rise that
falls again unseen is coalesced. No lock of ours is held while the host is
driven, and deliveries to one output are serialised and in order.

## Reset

`PciBus::reset` is RST#: each function's header, then `PciFunction::reset`, in
(device, function) order. Pin levels are left alone, so a function that is not
reset keeps its line. A restore is not a reset: it loads over what is there and
ends by telling the host the levels again.

| Machine | RST# | Grade |
|---|---|---|
| O2 | Not yet: the O2's cold reset does not reach its PCI functions; that comes with a later change. MACE has no software PCI reset: the ISA ring-base bit resets the ISA block alone (MACE specification p. 123; PCI registers pp. 162-165). | When it comes, inferred. MACE has a `PCIRst_N` output (MACE specification p. 188) and a `SysReset` input (p. 194), and CRIME drives system and cold reset (CRIME specification, Table 8-2); that CRIME's reset reaches `SysReset` and that `PCIRst_N` follows it on the board is not documented. |
| Octane | On a cold reset, to every function. And when a write takes Bridge Control bit `24 + n` (`RST_PIN_N(3:0)`) from 1 to 0: RST# to the functions at PCI device `n` alone, with a logged warning. | The bits, their reset value (all high) and that a 1 drives the pin high are established (Bridge ASIC specification rev 4.0, Control register and pin list: four `BUS_RSTN` outputs, "PCI Bus reset signals"), so a 0 asserts RST#. Which device a pin serves is not documented and is inferred from the driver guide's `pciio_reset()` (007-3443-002, p. 550), which resets one card "without affecting other devices on the PCI bus". |
| Octane | Not on a Bridge core reset or a Crossbow link reset. | Not established. The Bridge specification says the core reset comes from the link reset (rev 4.0, section 7.5) and does not say whether the pins pulse. |

## Wiring

**O2**, MACE bus 0. Outputs are CRIME inputs (MACE specification Tables 85, 87
and 90; established as software, and the board's INTB-D wiring is not known).

| Device | INTA | INTB | INTC | INTD | Use |
|---|---|---|---|---|---|
| 1 | CRIME 8 | | | | Internal SCSI controller 0 |
| 2 | CRIME 9 | | | | Internal SCSI controller 1 |
| 3 | CRIME 10 | CRIME 13 | CRIME 14 | CRIME 15 | The PCI option slot |
| 4 | CRIME 11 | | | | Not populated on an O2; iris keeps it spare for virtio-serial |

**Octane**, Bridge bus 0. Outputs are the Bridge's eight pins. Pins A and C of
device `d` reach pin `d`, B and D reach `d ^ 4`: inferred from software, not
established. `BoardIrq(i)` is pin `i`, shared with the GPIO inputs; the IOC3's
SuperIO interrupt is `BoardIrq(4)`. Devices 0 and 1 are the ISP1040s, 2 is the
IOC3, 3 is the RAD1, and 4 is virtio-serial when enabled.

## Octane windows

The CPU reaches PCI memory on the Bridge two ways (Bridge ASIC specification
rev 4.0; offsets in [octane-ip30-address-map.md](octane-ip30-address-map.md), section
8). A device window in the Bridge's own widget space, set by `Device(x)`; and
the 2 GB window HEART gives widget 15 at `0xF_8000_0000`, whose first 16 MB is
the widget's small window and whose range from offset `0x4000_0000` is the
32-bit alias of the low 1 GB of PCI memory. A BAR in either is mapped straight
to its function. The rest of the 2 GB window (16 MB to 1 GB) is reserved in the
specification (p. 52), which does not say what an access does; iris ends it in
a bus error, an approximation. The 64-bit alias and PCI I/O space, reached only
through the 64 GB window, are not modelled. The external ATE SSRAM (`0x080000`)
is not fitted on the Octane and reads as all ones, which is what the PROM
expects when it sizes it.

## Option boards in XIO slots

An Octane takes boards that put a PCI bus on a spare XIO slot. Each is a
Bridge (the onboard one's model) on its own Crossbow port, with its own bus,
interrupt pins to HEART and DMA through the Crossbow's live links.
`[[xio_board]]` fits them:

```toml
[[xio_board]]
kind = "pci-shoehorn"    # one slot, PCI device 1
# widget = 10            # a shoehorn's XIO slot, 9 to 12; unset: 10, then 12, 9, 11

[[xio_board]]
kind = "pci-cage"        # the Octane PCI module: three slots, PCI devices 1 to 3, port 13
```

The shoehorn is the single-slot XIO-to-PCI adapter: Bridge revision 4, one
slot at device 1 whose INTA and INTB reach the board's Bridge pins 1 and 5, and
a one-wire record naming it `XTALK_PCI` (part 030-1275-004, revision A). It does
not use the slot the graphics board is in.

The PCI module (the card cage) is the Octane's own three-slot option, on
Crossbow port 13 beside the four XIO slots. Its slots are PCI ID 1 to 3, the
guide's numbering, taken as configuration devices 1 to 3, with INTA of device
`d` on the board's Bridge pin `d` and INTB on pin `d ^ 4`. Its record names it
`XTALKPCI` (part 030-0952-005, revision E). Devices 0 and 4 to 7 hold nothing.

Which of this is established and which inferred is in `src/dev/xio_pci.rs`.
The shoehorn's slot number is established on the Origin 2000 and inferred on
the Octane; the module's slot-to-device numbering is inferred; the pins are
inferred from OpenBSD and Linux for both; the shoehorn's Bridge revision is the
Origin 2000's and the module's is a placeholder, the onboard Bridge's. A
board's Bridge has no `pci` monitor command yet; the onboard Bridge's `pci`
shows only its own bus.

## PCI-to-PCI bridges

`PciToPciBridge` is a function with a type 1 header and a bus behind it. The
bus in front merges the BARs behind the bridge that lie wholly inside an enabled
window into its own decode table and runs; a BAR that straddles a window's edge
is not forwarded, because what a bridge does with one is not defined. A pin of
the function at device `d` behind the bridge reaches pin `(pin + d) mod 4` of
the bridge's device, and its DMA is the bridge's. Raising bridge control bit 6
resets the bus behind it, and RST# on the bus in front reaches behind it too.
MACE's configuration cycles for a bus number other than 0 go to the bus firmware
has numbered that far. A configuration cycle that reaches the bus behind a
bridge and finds no function ends in master abort at the bridge: a read returns
all ones, a write is dropped, and the bridge's secondary status records received
master abort. On the Octane, the Bridge's type 1 aperture (0x028000 in its
widget space, Bridge specification table 43) sends a cycle to the bus and device
its PCI Type 1 Configuration register names (table 26); a bus number no bridge's
secondary to subordinate range holds, including 0, ends in a master time-out.
IRIX 6.5.30 scans a bus behind a bridge this way. The header is the type 1
header only (the caller names the PCI identity); the upper halves of the
prefetchable and I/O windows read zero. No machine builds one yet.

## Monitor

`pci status` lists each function: address, identity, command, assigned BARs
and where its pin goes (on the Octane too, since both machines share the bus);
`pci config <slot>` dumps a function's header.
