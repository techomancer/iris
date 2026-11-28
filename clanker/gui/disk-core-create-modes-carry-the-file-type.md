# disk-core create modes must carry the file-type bits

Status: rule (2026-11-22). Found while testing the disk browser on a copy of
a real IRIX 5.3 disk.

**Symptom.** A file the browser created (the text editor's save, an import)
listed as `crw-r--r--`: a character device, not a file. IRIX would see the
same broken inode.

**Cause.** `EditableFilesystem::create_file` and `create_directory` take
`CreateFileOptions::mode` / `CreateDirectoryOptions::mode` as the *whole*
inode mode. fs-unix's EFS stores it as given
(`new_ino.mode = options.mode.unwrap_or(0o100644)`), so a mode of `0o644`
writes an inode with no `S_IFREG`. `set_permissions`, by contrast, keeps the
type bits and replaces only the permission bits.

**Rule.**
- Pass `S_IFREG | perms` (`0o100000`) to `create_file` and `S_IFDIR | perms`
  (`0o040000`) to `create_directory`, or leave `mode` `None` for the crate's
  default. `iris-gui/core/src/controller/disk_browser.rs` has the constants.
- `set_permissions` may be given the full mode or only permissions; it keeps
  the type either way.
- `real_irix_disk` (an ignored test in that file) catches this: it edits
  `/etc/motd` and checks the listing still says `-rw-r--r--`.
