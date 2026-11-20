# Removed MachineConfig keys must be folded, not deleted

Status: rule (2026-11-15).

**Symptom.** iris-gui starts with `ERROR iris_gui::settings] .../gui.json:
unknown field `impact` ...`; every machine is gone and every save fails
("settings not saved" / "settings locked").

**Cause.** `MachineConfig` is `deny_unknown_fields`. Commit 862d0fe1 deleted
the `[impact]` section (replaced by `[graphics] board = "solidimpact"` etc).
Old `gui.json` files still embed legacy machine configs carrying
`"impact": {...}`, so the whole file failed to parse. `Settings::load_in`
then falls back to defaults with `migration_pending = true`, which blocks
every save — one stale key locks the user out of the GUI.

**Rule.** When you retire a `MachineConfig` key, keep it as a
`#[serde(default, skip_serializing)]` field and move it to its replacement in
`MachineConfig::fold_deprecated` (see `o2.*` and `LegacyImpactSection`).
The loaders (`MachineConfig::from_toml_str`, which the CLI and the GUI's
`machines::read` use, and the `gui.json` parse in `Settings::load_in`) all
call `fold_deprecated`, so a fold there covers the CLI and the GUI.
