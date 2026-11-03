# virtio-9p: a host directory for the guest

IRIS can present a directory on the host to the guest as a **virtio-9p** device
(virtio device ID 9, virtio 1.x) speaking 9P2000.L. It is the same protocol
QEMU's `-virtfs` uses, so any driver for that works: a Linux or BSD kernel on the
MIPS machines, or the IRIX filesystem module in the
[virtio drivers for IRIX](https://github.com/steigr/virtio-drivers) (`irix/virtio_9p`). No network and no
NFS server are involved.

None of the machines IRIS emulates had virtio, so **the device needs a guest
driver**. It is off unless `[virtio_9p] enabled = true`.

## Configuration

```toml
[virtio_9p]
enabled  = true
slot     = "exp0"              # "exp0" | "exp1"; Indy and Indigo2 only
path     = "/path/on/the/host"
tag      = "share"             # the name the guest mounts it by
readonly = false
uid      = 0                   # owner every file reports to the guest
gid      = 0
#owners   = "passthrough"       # the host's owners instead; see below
#uid_map  = "0:1000"            # guest uid 0 is host uid 1000 (guest:host[:count],...)
#gid_map  = "0:0"
```

`path` is required and the machine still starts if it cannot be opened: the
device then appears and every attach fails. `tag` is 1 to 64 bytes. The section
needs a Unix host, and is refused on Windows.

## Where it attaches

It uses the same transports as [virtio-serial](virtio-serial.md):

| Machine | Transport | Where |
|---|---|---|
| Indy (IP24), Indigo2 (IP22), Indigo2 IMPACT (IP28) | virtio-mmio v2 | GIO expansion slot `exp0` (`0x1F400000`) or `exp1` (`0x1F600000`, the default) |
| Octane (IP30) | virtio-pci | Bridge slot 5: vendor `0x1af4`, device `0x1049` |
| O2 (IP32) | | Not wired, as for virtio-serial |

A slot holds one device: with `[virtio_serial]` on as well, give each its own, or
the configuration is refused. An Indigo2 kernel dispatches no interrupts for
`exp1`, so use `exp0` for the device that needs speed. On that machine that is
this one: a driver on `exp1` has to poll for its replies.

## What the guest sees

Device feature `VIRTIO_9P_MOUNT_TAG` (bit 0). Config space is the tag's length as
a little-endian `u16`, then its bytes. There is one queue, of up to 256 entries;
a request is a descriptor chain of the T message followed by room for the R
message. Requests are answered in order, and a reply that will not fit the room
given is not cut short: the chain comes back with nothing written.

Messages are at most 1 MiB plus the 24-byte header, or what Tversion agrees.

### What is implemented

Tversion, Tattach, Twalk, Tclunk, Tremove, Tflush, Tlopen, Tlcreate, Tread, Twrite,
Tgetattr, Tsetattr, Treaddir, Tmkdir, Tsymlink, Treadlink, Tlink, Trename,
Trenameat, Tunlinkat, Tfsync and Tstatfs.

Refused: Tauth and the xattr and lock messages with `ENOTSUP`, so locks stay with
the guest, and Tmknod with `EPERM`, so a guest cannot make device files or FIFOs.
Existing host FIFOs and devices appear in listings but do not open.

### Inside the share

The guest names files by path and can itself create symlinks, so the device
never hands a guest's path to the host whole. Every operation walks from the
share's root one component at a time, following no symlink, and then acts on the
directory it reached. `..` stops at the root, a name holding `/` is refused, and
a symlink is only ever read: the guest resolves its target in its own namespace.
A symlink in the share therefore cannot reach outside it.

An attach's `aname` picks a directory under the share to be the root, with the
same rules: `..` is refused.

### Owners and permissions

The host's files all belong to one host user, the one running IRIS. So every file
reports the configured `uid` and `gid`, and a guest's chown succeeds without
effect. A file, directory or symlink the guest creates is handed to that `uid` and `gid` on
the host when IRIS may chown (it runs as root, or has CAP_CHOWN); otherwise it belongs to the
host user. That is how a share can be written by the guest and deleted by an unprivileged
user on the host: set `uid`/`gid` to theirs. Modes and times are the host file's, and a guest's chmod and utime change
them. The device checks nothing on the guest's behalf; everything it does, it
does as the host user, and the host's own permissions apply to that user. The
guest's kernel makes its own checks from the attributes it is given.

#### `owners = "passthrough"`

With the default, `owners = "squash"`, the guest sees one owner and the host sees the user
running IRIS (or `uid`/`gid`, see above). `passthrough` carries the guest's ids to the host and
back, as QEMU's passthrough security model does:

- `getattr` reports the host file's owner, translated host-to-guest by `uid_map`/`gid_map`
  (`"guest:host[:count],..."`; empty maps an id to itself, a non-empty map knows only what it
  lists). An owner the map lacks shows as `uid`/`gid`.
- `Tsetattr` with a uid or gid is made on the host with `fchownat`, guest-to-host through the
  maps; an id the maps lack is refused with `EPERM`. The host user must be allowed to chown
  (root, or CAP_CHOWN) for anything but its own ids.
- What the guest creates starts as `uid` and the host group its `Tlcreate`/`Tmkdir`/`Tsymlink`
  gid maps to. 9P2000.L's create carries a gid but no uid, so the driver follows with a
  `Tsetattr` for the creator's uid (p9fs does, with its `owners` mount option); a driver that
  does not leaves the entry with `uid`.

The device trusts what the guest says: a guest user that can chown can give a file to any host
uid in the map, and root in the guest is whatever its map entry says on the host. Map only ids
the guest should be able to act as.

## Limits

- **No snapshot support yet**, as with virtio-serial: restoring a snapshot leaves
  the device idle, and a guest must remount.
- Host calls happen on the device's service thread. A share on a very slow
  filesystem holds up the guest's next access to the device's registers.
- Inode numbers come from the host, mixed with the device number so that a share
  spanning host filesystems does not repeat them.
