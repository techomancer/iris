# The GUI must read machine configs through `MachineConfig::from_toml_str`

Status: rule, 2026-11 (fold_cards added 2026-11-27).

**Symptom.** A machine made in iris-gui kept one jitv2 compile thread however
many CPUs it had, while the same file run with `iris --config` got one per CPU.
Older `[o2]` spellings were not folded either, and `banks` read as unset.

**Cause.** `iris-gui/core/src/model/machines.rs::read` parsed the file with a bare
`toml::from_str`. The core's loader (`MachineConfig::parse_toml`, reached by
`try_load`) does more after parsing: it records whether `banks` and
`[jitv2] threads` were given (`banks_set`, `jitv2_threads_set`, both
`#[serde(skip)]`), folds deprecated keys and settles an unset thread count from
the CPU count. A plain serde parse leaves the skip fields false and
`threads` at its serde default of 1.

**Rule.**
- Read with `machines::parse` (`machines::read` for a file):
  `MachineConfig::from_toml_str`, which is `parse_toml` made public, then
  `fold_cards`, which turns the older card flags (`graphics.heads = 2`,
  `[ultra64]`, `test_device`, `[virtio_*]`) into `[[devices]]` entries as a
  run does. Without the fold the GUI shows those cards as absent. See
  `gui-disks-and-cards-are-devices-by-slot.md`.
- Write with `machines::to_toml`, which leaves `[jitv2] threads` out while it
  is automatic, and `banks` out while it is unset and still the generic
  default (the core's own test for "unset"; the Tezro and IP35 machines then
  fit their own memory, and `[128, 128]` is not valid there). Code that sets
  `banks` without `banks_set` still saves, because the value then differs. Serde always writes `threads`, and a written key reads back as
  "set", so a plain `toml::to_string` freezes the automatic value at the first
  save. Compare configs for dirtiness with `to_toml` too, or toggling
  **Automatic** does not count as a change.
- Call `settle_jitv2_threads` before a start: the CPU count may have changed
  since the file was read.
- A new `#[serde(skip)]` "was it given" flag in `MachineConfig` needs the same
  treatment in `to_toml`.
