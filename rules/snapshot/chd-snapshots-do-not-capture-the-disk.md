# CHD snapshots do not capture the disk

**Keywords:** snapshot,restore,chd,cow,overlay,scsi,corruption,diff.chd
**Category:** snapshot

With a CHD image, `save` writes no disk state and `restore` rolls none back. RAM
and device state go back in time while the disk stays in the present, which is a
plausible contributor to the guest kernel corruption seen after every restore.

## Where it goes wrong

`ScsiDevice::is_cow` (`src/scsi.rs:309`) returns true for a `ChdHd` that has a
diff, so `export_overlays` lists the device as having COW state worth saving.
`cow_export` (`src/scsi.rs:279`) then matches only `DiskBackend::Cow`:

```rust
match &mut self.backend {
    Some(DiskBackend::Cow(cow)) => cow.export_overlay(dest),
    _ => Ok(Vec::new()),
}
```

`ChdHd` falls into `_` and returns an empty vector without error. `cow_import`
has the same shape, so restore rolls nothing back. `cow_dirty_count` does handle
`ChdHd`, which is what makes the gap easy to miss: the device reports dirt and
reports itself as COW, then exports nothing.

## What it looks like

Snapshot of a CHD run:

```
cow.toml:
scsi1 = []
```

No `scsi1.overlay` in the snapshot directory. The same guest against a raw image
extracted with `chd_extract`:

```
scsi1.overlay  3808.6M
cow.toml       328.4K

scsi1 = [ 266240, 266241, ... ]
```

## Not the cause of the restore panic

Worth fixing on its own terms, but it was chased as the likely cause of the
kernel panic after `iris-ci restore` and it is not:

- A raw image extracted with `chd_extract` **did** capture `scsi1.overlay` with a
  real dirty list in `cow.toml`, and restore still panicked. n=2.
- An in-place `save` then `restore` seconds later, same process, moves the disk
  by milliseconds and panics anyway.

So the disk being stale cannot be what kills the guest. See
`scc-restore-rr0-contradicts-the-emptied-fifos.md` for what else has been ruled
out and what is left.
