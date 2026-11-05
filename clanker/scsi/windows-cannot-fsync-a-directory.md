# Windows cannot fsync a directory

Status: rule (2026-11-05).

Symptom: on Windows, a raw disk with `overlay = true` boots until IRIX first
writes a new sector (or until an overlay with an existing `.dirty` sidecar is
reopened), then gets `Media error: Write error (asc=0xc)` on the root disk and
panics. iris logs `flushing SCSI id 1 failed: Access is denied. (os error 5)`.

Cause: `save_dirty_sidecar` (`src/cow_disk.rs`) renames the temp record over
the sidecar, then calls `chd_disk::fsync_dir`, which did
`File::open(dir)?.sync_all()`. Windows refuses `File::open` on a directory
with ERROR_ACCESS_DENIED. The rename itself is fine: std's rename replaces a
target that has an open append handle without trouble. The CHD paths call
the same helper but ignore its result, so CHD overlays never failed.

Rule: `fsync_dir` is a no-op on Windows (NTFS journals the rename). Do not
add a directory sync whose error propagates without a `cfg(windows)` answer.
Opening the directory with `FILE_FLAG_BACKUP_SEMANTICS` and calling
`FlushFileBuffers` might work but has not been tested on real Windows.
Test: `cow_disk::tests::the_dirty_sidecar_can_be_replaced_repeatedly` (it
only catches the failure on a Windows host).
