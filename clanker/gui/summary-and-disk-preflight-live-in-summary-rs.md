# The stopped-machine summary and the missing-disk preflight live in `summary.rs`

Status: rule, 2026-11.

**Symptom (issue 304).** On an O2 the summary and the Start preflight looked
Indy-only: the summary listed `[scsi]` and nothing else, the missing-disk
modal named an O2 external-bus target by its raw `[scsi.KEY]` key, which reads like an internal-bus target or a
number the Disks tab never shows, and an
Origin 3200 listed the `[scsi]` map it never uses. The original report (an
O2 summary showing the Indy's PROM, memory and disks, and Start stopping at
"missing disk" until a `[scsi.N]` existed) dates from when the O2 kept its
own `o2.prom`/`o2.disks`/... keys. Those are now deprecated and folded into
the shared keys by `MachineConfig::fold_deprecated`, so reading the shared
keys is right for every profile, as long as the file goes through
`from_toml_str` (`gui-reads-configs-through-core-parse.md`).

**Rule.**
- Work out what the summary lists and what the preflight checks in
  `iris-gui/core/src/model/summary.rs` (`drives`, `network`, `missing_disks`,
  `expansion_paths`), from the config alone, so tests need no window.
- Name SCSI targets with `model::hardware::scsi_name` (controller:id), never the
  `[scsi.KEY]` key; keep the key only for detaching.
- A new place disks can live (another expansion card, another machine's own
  map) goes into `drives` and, if a missing image would make `Machine::new`
  exit, into `missing_disks` or `expansion_paths`.
- The IP35 bricks take no `[scsi]` disks; both functions skip that map there.
