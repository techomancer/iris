# CI mode overlay path is /tmp/iris-ci-PID-scsiID.overlay

**Keywords:** ci,overlay,scratch,/tmp,iris-ci,wd33c93a,cow,snapshot,debugging
**Category:** snapshot

# CI Mode Overlay Path is /tmp-Based, Not Image-Sibling

**Raw images only. CHD ignores all of this, see the carve-out below.**

When iris is invoked with `--ci`, the COW overlay file does NOT live next to the base image (`<base>.overlay`). It goes to `/tmp/iris-ci-<pid>-scsi<id>.overlay`. This isolates concurrent CI runs from each other and from any interactive session sharing the same base image.

## Carve-out: CHD images are not isolated

`Wd33c93a::add_device` (`src/wd33c93a.rs:372`) applies `overlay_path_override`
only in the `else if overlay && !is_cdrom` raw-image branch. The CHD branch calls
`ChdHd::open(path, overlay)` and never sees the override, so the diff lands at
`chd_disk::diff_path_for()`, which is `<base>.chd.diff.chd` beside the image, and
it is **not per-pid**.

Confirmed with `lsof` on a running `ci = true` emulator:

```
iris ... /home/mach/workspace/irix-actions-runner/Indy-IRIX65_dev.chd
iris ... /home/mach/workspace/irix-actions-runner/Indy-IRIX65_dev.chd.diff.chd
```

No `/tmp/iris-ci-*` file exists in that run. The same config against a raw image
does produce `/tmp/iris-ci-<pid>-scsi1.overlay`.

Consequences: concurrent `--ci` runs on one CHD share a single overlay, and
consecutive runs inherit each other's guest state. An A/B comparison that does
not isolate the overlay is worthless, since a guest that panicked in one run
boots into `savecore` in the next. Set `IRIS_CHD_DIFF_DIR` (`src/chd_disk.rs:307`)
to redirect the diff to a directory of your choice, which is cleaner than
deleting the file between runs.

## Where it's set
`src/machine.rs:197`:
```rust
let ci_overlay = format!("/tmp/iris-ci-{}-scsi{}.overlay", ci_pid, id);
hpc3.add_scsi_device_with_overlay(id as usize, &path, dev.cdrom, discs, dev.overlay, &ci_overlay)
```

`src/wd33c93a.rs:255-258` honors the override:
```rust
let overlay_path = overlay_path_override
    .map(|s| s.to_string())
    .unwrap_or_else(|| format!("{}.overlay", path));
```

## Implications
- `rm -f irix65_4g.raw.overlay` before launching `--ci` is a no-op.
- To inspect the live overlay during a `--ci` run, find it via `lsof -p <iris-pid> | grep overlay`.
- After the iris process exits, the CI overlay file remains under `/tmp` until the next reboot or manual cleanup.
- `save_snapshot` correctly captures the CI overlay regardless of path (it routes through `cow_disk::export_overlay`, which uses `self.overlay_path`).

## Verification
```
lsof -p $(pgrep -f 'target/release/iris.*--ci') | grep overlay
```
Should show: `/private/tmp/iris-ci-<pid>-scsi1.overlay`
