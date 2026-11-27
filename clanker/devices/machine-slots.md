# A machine is a list of slots; `[[devices]]` are placed into it

Status: design (read `src/topology.rs`, `src/slot_path.rs` and `src/devices.rs` before relying on details).

**What to know.** Where a device goes is data, declared once per machine and
checked in one place. Nothing else decides whether a disk or card fits.

- `MachineSlotPath` (`src/slot_path.rs`): the address of a slot, such as
  `xio15/pci0/0/t1`, `hpc3/c0/t4` or `gio1`. Each step has a tag and a number,
  so a path parses without knowing the machine and does not depend on attach
  order. The step type is private; build a child with `pci(n)`, `target(n)`,
  `loop_id(n)` and ask with `target_id()`, `gio_slot()`.
- `MachineSlot`: a path, a `MachineSlotKind` (a built-in part, a SCSI
  controller, a PCI bus, or a slot that takes devices of some
  `DeviceFamily`) and its aliases (`scsi0`, `pci_slot`). A machine's slots are
  a plain `Vec<MachineSlot>`.
- `ConfiguredDevice` (`src/devices.rs`): one `[[devices]]` entry: a
  `DeviceModel` (its type and options), an optional `at` and `name`, and a
  `default` flag for the disks a config gets when it names none.
- `DevicePlacement`: which slot each device got, the slots grown by what
  devices brought, and the PCI buses the constructor built. `new(cfg)` is
  strict; `lenient` gives an editor each device's slot whatever else is wrong.

**Placement.** `at` is a slot path whose first step may be a machine alias or
another entry's `name`. Devices claim in three passes, each in file order:
a full path to one slot, then a bus (`scsi1`), then no `at`, which takes the
first free slot of its family in list order. A slot that does not exist, does
not take the device or is already taken is an error naming the slot and the
device; a device is never dropped. Only a default device that finds no room is
left out. A device with no slot yet may wait for one a later device brings (a PCI
module's slots, a Fibre Channel loop). Spare `Extra` slots take virtio by
default and a card only when it names them.

**Giving a machine its slots.** Add `slots(s: &mut Vec<MachineSlot>)` beside
the constructor and call it from `MachineProfile::slots`. It builds no hardware:
`open_slot`, `fixed_slot`, `scsi_bus_slots` and `pci_bus_slots` add slots, and
a board's PCI bus is a `const` table of (device number, kind, name) next to the
routing the chip takes. List order is placement order. The constructor builds
`DevicePlacement::new(cfg)`, binds each PCI bus it made with `bind_pci`, and
hands its SCSI buses to `attach_from_config` in list order; a count that
disagrees fails the build, so the list cannot drift from the hardware.

**`[scsi.N]`.** `src/config/legacy.rs` turns the older tables into entries
with a warning: keys 1 to 15 are targets on `controller` (default 0), a key
of 16 or more is `16 * controller + id`, and each becomes `at = "scsiC/tN"`, so
it lands on whatever machine the command line picks. `--scsiN` and `--cdromN`
replace the image of the entry at their slot. Disc changers
still key by `16 * controller + id` where they must; `scsi_target` and
`scsi_target_slot` convert.

**`--list-slots`.** One line per slot: path, aliases, what is in it (a
built-in part, or the device with its name, and "(default)"), and what it
takes. A run of free targets on one bus shares a line (`hpc3/c0/t2-3`).
