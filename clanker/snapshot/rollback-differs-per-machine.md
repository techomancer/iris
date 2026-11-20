# `iris-ci rollback` is built three ways, one per machine family

Status: current (2026-11-19).

`Workstation::ci_restore` / `ci_rollback` have no shared implementation.
Each machine keeps what its rollback needs at `ci_restore` time:

- **Indy / Indigo2** (`Machine`): after the load, `capture_rollback_checkpoint`
  reads the live machine back into a `RollbackCheckpoint` (bank words, parsed
  device TOMLs, framebuffers, graphics checkpoints). Without one it falls back
  to `ci_restore` from disk.
- **O2** (`O2Machine`): keeps the `O2Snapshot` that `read_snapshot` returned
  (it already holds the memory and parsed state), boxed with the name, and
  `apply_snapshot(&snap)` puts it back. Nothing is read back from the live
  machine.
- **Octane** (`OctaneMachine`): remembers only the name and calls
  `load_snapshot` again, so every rollback rereads the snapshot.

Every one reimports the COW overlays from `saves/<name>/` (`ctrl<i>/` on the
O2). That is safe because guest writes after the import go to the live
overlay, never back into the snapshot directory. A rollback test can prove
it does not reread state by deleting `snapshot.toml` after the restore (the
O2's `ci_rollback_returns_to_the_last_restore` does).

Snapshots are written under the working directory's `saves/`. A machine test
that saves runs itself again as a child process in a private directory (see
the O2 rollback test and the REX3 head-1 lifecycle test).

A CD change (`cdrom-load`/`cdrom-eject`) needs nothing per controller: the
`DiscChanger` blanket impl over `ScsiTargets` reaches the shared `ScsiDevice`,
which arms unit attention 06/28. The AIC-7880 passes it to the HIM's
auto-sense like any other check condition
(`a_disc_change_reaches_the_him_as_unit_attention`).
