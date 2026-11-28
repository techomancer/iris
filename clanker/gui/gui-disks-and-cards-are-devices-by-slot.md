# GUI disks and cards are `[[devices]]` entries, keyed by slot

Status: rule, 2026-11-27 (main's "place SCSI devices by slot", 707c54ed, and
"place cards by slot", 606c214c).

**What changed.** `MachineConfig` has no `scsi` map, `controller` field,
`nvme`, `fibre_channel` or `xio_board` any more. Every disk, CD-ROM,
DaynaPort, NVMe card, Fibre Channel card and option board is a
`ConfiguredDevice` in `cfg.devices`, with an `at` slot path. Older files still
load: `[scsi.N]` and friends fold into entries (`src/config/legacy.rs`), and a
save writes `[[devices]]`.

**Rules for iris-gui-core and the shells.**
- A disk's key is still `scsi_slot(controller, id)` (`16 * controller + id`).
  Use `cfg.disk(key)`, `disk_mut`, `set_disk`, `remove_disk`, and
  `cfg.disks()` (slot, index into `devices`), or
  `model::hardware::scsi_disks` for (key, &ScsiDeviceConfig) in slot order.
  `hardware::placement(key)` gives (controller, id); `scsi_name(key)` needs
  no config.
- `disks()` places every device each call (`DevicePlacement::lenient`). Call
  it once per pass, not once per key in a loop over many keys.
- Loop disks are not in `disks()`: use `cfg.loop_disks()`. NVMe cards:
  `cfg.nvmes()`. `hardware::clear_disks` removes the SCSI targets only.
- A profile switch needs the profile it leaves:
  `switch_profile(cfg, from)`, which keeps each disk at the same controller
  and target where the new machine has that slot.
- **The folded card flags still compile and do nothing.** `graphics.heads`,
  `ultra64.enabled`, `test_device`, `virtio_serial.enabled` and
  `virtio_9p.enabled` are reset by `fold_cards` when a file is read. Read
  `cfg.has_device("newport")` (the second head), `has_device("ultra64")`,
  and write with `cfg.set_device(ty, on)`. Grep for the old fields after a
  merge from main; the compiler will not find them.
- The disc changers number drives by the slot key too: a disc command for
  the CD-ROM at key 20 sends 20, not 4 (`PowerController::cd_id`).
- `workstation::build` settles slots itself (`cfg.resolved()`); the GUI does
  not need to.
