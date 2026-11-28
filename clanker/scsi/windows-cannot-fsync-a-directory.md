# Windows cannot fsync a directory

Status: rule (updated 2026-11-28).

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

The initial fix made `fsync_dir` a no-op on Windows. Raw overlay sidecars now
flush the temporary file and publish it with
`MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH)`, checking its result. Do not
add `COPY_ALLOWED`: replacement must stay on the same filesystem. Reject NUL
in paths before calling the UTF-16 API. Unix retains rename and directory sync.

Unlike the standard library's rename path, this replacement can fail while
an append handle to the destination is still open. `CowNode::rewrite_record`
retires both its own handle and the background flusher's handle first.
Ordinary file syncs hold `Background.files`, so `set_record(None)` also waits
for an in-flight sync to release its handle. Custom `Durable` callbacks must
stay outside that mutex because they may take backend locks.

A read-only `RawNode` has no writes to flush; both flush methods return success
without calling `FlushFileBuffers`, which requires write access on Windows.

Tests: sidecar create/replace/error propagation, repeated record replacement,
read-only raw flush, and COW write/background-sync/flush/reopen over a read-only
base. Windows-specific failures require a Windows test run.
