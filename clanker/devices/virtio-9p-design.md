# virtio-9p: 9P2000.L over a host directory (design and gotchas)

Status: implemented 2026-11-02 for IP22/IP24/IP28 (mmio) and IP30 (pci, Bridge
slot 5). Exercised end to end on IP28 under the IRIX module in
`github.com/steigr/virtio-drivers` (`irix/virtio_9p`). IP30
and Linux guests have not run it. User docs: `docs/dev/virtio-9p.md`.

## Layout

`src/dev/virtio/p9.rs`: `Server` is the protocol and the host calls, knowing
nothing of virtio, and is what the unit tests drive. `P9` is the `VirtioDevice`
around it.

## Rules that cost time

- **Never pass a guest path to the host.** `Server::dir` walks from the root's fd
  with `openat(O_NOFOLLOW)` per component and everything else is `*at` from there.
  `canonicalize` plus a prefix check would race a guest making a symlink. A fid
  holds components, so an op on a deep path costs its depth in syscalls; nobody
  has measured that mattering.
- **A fid is a name, not a file.** Rename updates every fid at or under the old
  name, but a rename *over* a file leaves that file's fids naming the newcomer. A
  client must compare qids if it cares; the IRIX module does.
- **Open fids hold an fd**, and Tgetattr on them uses it, so an open file that has
  been unlinked still answers. Walked fids answer ENOENT.
- **Errnos and open flags are Linux's** on the wire whatever the host is: macOS
  numbers `ENOTEMPTY` 66 and `O_CREAT` 0x200. `linux_errno` and `host_flags`.
- **Treaddir snapshots at offset 0.** Offsets are indices into the snapshot, because
  host telldir values are not portable and a cookie would have to survive `rewinddir`.
- **`O_APPEND` is deliberately not passed to the host.** Twrite carries the offset,
  and on Linux hosts `pwrite` on an `O_APPEND` fd ignores it.
- **Servicing is inline** on the virtio thread (`notify`), so a slow host call holds
  `VirtioCore`'s lock and a guest register access waits on it. Fine for a local disk.
  A worker pool would need `Cx` split up.

## Open

- **Snapshots:** as virtio-serial, `VirtioCore` has no `Saveable`.
- **Tmknod, xattrs and Tlock** are refused. A FIFO would need `mkfifoat`, which is
  not in the libc crate on every host.
- **uid/gid mapping** is one owner for all; QEMU's `mapped-xattr` model would be the
  way to store per-file owners on a non-root host.
