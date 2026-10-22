# macOS App Sandbox: the CHD on-exit fold needs a *folder* grant, not a file grant

Status: confirmed fix (2026-06-19). On the Mac App Store (sandboxed) build,
compressed CHDs were not being "shrunk" on exit — the "Synchronizing disks…"
fold silently did nothing and the `.diff.chd` kept growing.

## Root cause

A security-scoped bookmark minted from the file picker (the user choosing a disk
image) grants read-write to **that file**. It does **not** grant the right to
create siblings in, or rename within, the file's **directory**.

`chd_disk::flatten_diff` (the fold) is built on an atomic replace:

1. write the rebuilt CHD to `<base>.synctmp.chd` **beside the base**, then
2. `rename(synctmp, base)` over the base, then
3. remove the diff.

Steps 1 and 2 both require **directory** write access. With only a file-scoped
grant the sandbox denies them, so `flatten_diff` returns `EPERM` early — base and
diff are left untouched (correct/safe), but the disk never compacts.

Two things hid it:
- The running COW writes go to `IRIS_CHD_DIFF_DIR` (a writable container path set
  in `main.rs` for the appstore build), so *during* the session everything works
  — only the exit-time fold, which writes beside the base, fails.
- `handle.rs` swallowed the error: `m.sync_chd_disks(...).unwrap_or(0)`, so the
  user saw "0 synced" with no message.

Note you can't fix this by redirecting `synctmp` into the container too: the
final atomic `rename` over the base still needs write access to the base's
directory. And rewriting the base in place (file-level access only) would forfeit
the crash-atomicity the rename gives — a crash mid-rebuild leaves the base
neither old nor new, and the diff no longer applies. So the fold genuinely needs
the folder grant.

## The fix (all in iris-gui)

- **Surface it.** `handle.rs` `Cmd::SyncDisks` now reports the fold error via
  `Evt::Error` (a toast) instead of `.unwrap_or(0)`.
- **Preflight + gather the permission.** On Start (appstore only),
  `chd_dirs_needing_grant()` probes each attached non-scratch HDD `.chd`'s parent
  directory with `dir_writable()` (create+remove a `.iris-write-probe-<pid>`
  file — the sandbox can `stat` a dir yet deny the write, so probe a real
  create). Any in a non-writable folder pop the `ChdGrantModal`, which offers a
  per-folder "Grant …" button (a folder NSOpenPanel pre-pointed via
  `grant_disk_folder_at`, persisting a recursive directory bookmark) or "Start
  without compacting" (one-shot `skip_chd_grant_check` bypass). After a grant we
  re-probe and drop satisfied disks; when the list empties we start.

The granted directory bookmark is process-wide once `startAccessingSecurity-
ScopedResource` runs (`macos_sandbox::restore`), so the fold on the handle worker
thread gets the access — security-scoped access is not thread-scoped.

## Three layers of sandbox file access (why `dir_writable` is the right test)

A MAS build reaches files three ways, and the fold needs *directory* write from
one of them:
1. **Container** (`~/Library/Containers/<id>/Data/…`, where `dirs::data_dir()`
   points): always writable, no grant. New disks (`<data_dir>/disks`) and the COW
   diff (`IRIS_CHD_DIFF_DIR`) live here — that's why *running* works even when the
   fold doesn't.
2. **User-selected + security-scoped bookmarks** (`files.user-selected.read-write`
   + `files.bookmarks.app-scope`, both in `installer/iris-gui.entitlements`): a
   *file* bookmark (picking a disk) grants RW to that file only; a *directory*
   bookmark (picking a folder) grants recursive RW. These are **invisible** in
   System Settings → Privacy & Security → Files & Folders.
3. **TCC special-folder consent** (Desktop/Documents/Downloads): the visible
   "Documents Folder" toggle. For a sandboxed app this extends access to that
   tree — but we have **no** Documents entitlement and must not depend on it.

`dir_writable()` (a real create-probe) is ground truth across all three: if the
CHD sits in the container, in a granted folder, or under an effective Documents
grant, the probe passes and we don't prompt; otherwise we do. So we never need to
know *which* layer is providing access — only whether a write would succeed.

We deliberately did **not** force CHDs into the container (multi-GB copy into a
hidden path) or require ~/Documents (broad exposure, TCC-dependent). The
least-privilege path is a user-selected folder grant for the disk's own folder,
with UI text recommending one CHD per dedicated folder (a grant is recursive).

## When we prompt

Both at **Start** (preflight) and at **assignment time** — when a disk image is
picked via Browse in the Disks tab (`PathEdit.picked` → `TabOutcome.disk_picked`
→ `check_chd_folder_grants`). Assignment-time is gated on an actual *pick*, never
on typed text, so the dialog can't pop mid-keystroke. The grant button opens a
folder picker pre-pointed at the CHD's directory, so permissions are assigned
only when the user explicitly selects that folder.

## Detection predicate

We prompt for any non-scratch, non-CD `.chd` in a non-writable folder. We don't
gate on actual compression (which would need opening the CHD): nearly all
attached CHDs are compressed, a per-disk COW toggle would make even an
uncompressed base overlay, and a redundant folder grant is harmless. CD-ROM CHDs
are read-only (no diff) and scratch disks live in the writable container, so both
are skipped.
