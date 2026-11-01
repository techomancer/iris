# CI mode overlay path is $TMPDIR/iris-ci-PID-scsiID.overlay

**Keywords:** ci,overlay,scratch,tmpdir,iris-ci,cow,snapshot,debugging
**Category:** snapshot

# CI Mode Overlay Path is in the Temporary Directory, Not Image-Sibling

When iris is invoked with `--ci`, the COW overlay file does NOT live next to the base image (`<base>.overlay`). It goes to `iris-ci-<pid>-scsi<id>.overlay` in the host's temporary directory: `std::env::temp_dir()`, which is `$TMPDIR` when set and `/tmp` otherwise. This isolates concurrent CI runs from each other and from any interactive session sharing the same base image.

## Where it's set
`ScsiDeviceConfig::ci_overlay_path` (`src/config.rs`), used by `attach_from_config` (`src/scsi/attach.rs`) when `cfg.ci` and the disk has `overlay = true`.

## Implications
- `rm -f irix65_4g.raw.overlay` before launching `--ci` is a no-op.
- To keep the overlay on the same filesystem as `saves/` (so a snapshot can reflink it), point `TMPDIR` at a directory on that filesystem.
- After the iris process exits, the CI overlay file and its `.dirty` record remain in the temporary directory until the next reboot or manual cleanup.
- `save_snapshot` captures the CI overlay regardless of path (it routes through `cow_disk::export_overlay`, which uses `self.overlay_path`).

## Verification
```
lsof -p $(pgrep -f 'target/release/iris.*--ci') | grep overlay
```
Should show `$TMPDIR/iris-ci-<pid>-scsi1.overlay` (on macOS, `TMPDIR` is under `/var/folders/`).
