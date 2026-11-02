# Storage, snapshots, and automation

Disk images, overlays, snapshots, the CI control interface, and scratch-volume
workflows. The full configuration inventory is in [FEATURES.md](FEATURES.md);
[HELP.md](HELP.md) covers image preparation and [debug.md](debug.md) the monitor commands. The sections
below retain the detailed material previously in README.md.

### CHD image support

IRIS mounts `.chd` hard-disk and CD-ROM images directly without first
extracting to raw. Compressed parent CHDs stay untouched — writes go to a
MAME-style `.diff.chd` sidecar.

The CHD backend (`libchdman-rs` >= 0.288.8) and the MAME CHD core it vendors
are BSD-3-Clause licensed, so IRIS stays fully BSD-3-Clause (see
`LICENSE-libchdman-rs.txt`).

See [HELP.md](HELP.md) for the full rundown: serial ports, monitor console,
NVRAM/MAC address setup, disk image prep, and more.

**Windows 11:** full build/launch guide in [wsl/README.md](wsl/README.md).


## Disk write cache

Each hard disk has a `write_cache` setting in its `[scsi.N]` table. It is the
WCE bit of the SCSI caching page: whether the disk may tell the guest a write
is done before the write is on the host's disk. `--no-write-cache` sets it to
`false` for every disk.

```toml
[scsi.1]
path = "irix.img"
write_cache = false   # default true
```

- `true` (the default): a write is done once the host has it. A thread syncs
  it to the host's disk about 5 s later. This is no slower than before the
  setting existed.
- `false`: every write is on the host's disk before the guest is told it is
  done, as on a drive with its cache off. It costs speed: in an IRIX 6.5.30
  guest on an Octane, diskperf's direct-I/O writes ran at 28 MB/s against 49
  MB/s (two runs each). Large `dd` runs and reads were unaffected.

**The risk of `true`:** a host crash or power loss can lose up to about 5 s of
the writes the guest was told were done, in any order, and that can damage the
guest's filesystem. Nothing synced a disk before this setting existed except
SYNCHRONIZE CACHE and a clean stop, so the default is no less safe than what
it replaced. Set `write_cache = false` for a disk you cannot rebuild, or an
installed system you keep. Stop iris cleanly before shutting the host down.

### What survives

A clean stop (Ctrl-C, SIGTERM, SIGHUP), a WRITE(10) with FUA, and
SYNCHRONIZE CACHE lose nothing in any mode. These cases differ:

| | `true` | `false` |
|---|---|---|
| iris killed (`kill -9`) or crashes | nothing lost | nothing lost |
| Host crashes or loses power | up to about 5 s lost | nothing lost |

A guest's `sync` or `fsync` does not change this: IRIX sends nothing the disk
acts on. A thread syncs a disk about 5 s after the first write that follows a
sync, so treat 5 s as typical, not as a bound. Measured on an Octane with
IRIX 6.5.30 on CHD disks: four kills, each during a loop that copied a file
and ran `sync`, lost none of the 58 files the loop had finished, and the root
filesystem stayed sound. A real host crash has not been tried; its window
follows from how the sync works.

A guest can switch the bit with MODE SELECT (clearing it stores what was
written so far at once). A bus reset puts it back to the configured value: on
the AIC-7880 and the ISP1040 a bus reset, on the WD33C93A a reset of the chip.

### SGI context

SGI shipped its drives with the write cache off (inferred, not measured), and
IRIX never flushes a disk. Its 6.5.30 kernels do not read the caching page,
set FUA or send SYNCHRONIZE CACHE. That is established for IP27 (Origin 2000,
Onyx2) and IP35 (Origin 350) and only reported for IP30 (Octane) and IP32
(O2). So with a cache, `sync`, `fsync` and XFS log writes cannot force
anything out, and after a crash the XFS log and the metadata it describes can
disagree. `write_cache = false` is the faithful mode: GOOD means stored, which
is what IRIX assumes. NetBSD and Linux send SYNCHRONIZE CACHE to a disk that
reports a cache, so `true` is safer for them.

### CHD disks

A write to a hunk the CHD already holds is stored in place. A write to a new
hunk appends it to the file, and the hunk's map entry waits in memory until a
sync has stored the data, so after a host crash the map only points at data
that reached the disk. In either mode the write then syncs the data, writes
the entry and syncs again, before the guest is told it is done. Left to
the background sync, the entry would be lost to a kill while later writes to
hunks already held were kept, and a filesystem cannot recover from losing an
earlier write but keeping a later one. Writes to a hunk the CHD already holds
cost one sync with `write_cache = false` and none with `true`, as on a raw
image, so the extra cost falls on a fresh `.diff.chd` filling up.

### For developers

- The interval is `BACKGROUND_FLUSH_INTERVAL` (5 s) in `src/block_node.rs`.
  `Background` is the flag and thread. `ScsiDevice::new` starts it for each
  writable disk and a node's `write` calls `Background::mark`.
- The default is `default_write_cache` in `src/config.rs`.
  `ScsiDevice::with_write_cache` applies it; a bare `ScsiDevice` has no cache.
  `perform_write` in `src/scsi.rs` calls `flush_data` before GOOD for a
  WRITE(10) with FUA and, when WCE is 0, for every write. SYNCHRONIZE CACHE
  (10) and (16) call `flush` on the whole disk and ignore the range and IMMED.

What a `BlockNode` promises (the `block_node.rs` module docs say the same):

- `write` has put the data where a killed iris leaves it, the host's file,
  before it returns. Data held only in memory dies with `kill -9`, so a
  backend that holds part of a write back (the CHD's map entry for a new
  hunk) syncs before `write` returns.
- `flush_data` returns only when every write accepted so far is on the host's
  disk and the file would open after a crash and read back as the guest wrote
  it. An on-disk index must never point at data that is not there. It may
  skip work only a clean stop needs, such as compacting an overlay's record.
  `write_cache = false` and FUA depend on it.
- `flush` is `flush_data` plus that housekeeping.
- `background` returns the node's `Background`. A backend whose syncs must go
  in an order supplies a `Durable`; its `sync_data` does what `flush_data`
  does, and the thread calls it holding no node or SCSI lock. A failed sync
  leaves the writes pending, so the thread retries.

A new backend, a qcow2 image for instance, has to: write guest data to the
host's file before `write` returns; make `flush_data` sync data clusters
before the tables that point at them, then the tables; have the thread sync in
that order through `Durable`; call `mark` after every write; and add a column
to the table above if it holds accepted writes in memory. Tests to copy, in
`src/scsi.rs`: `a_disk_without_a_write_cache_stores_every_write_before_good`,
`write_10_with_fua_syncs_before_good` and
`synchronize_cache_flushes_in_both_cdb_sizes`. For a kill test, see the
ignored `kill_mid_write_leaves_a_chd_that_opens` in
`src/block_node_sync_bench.rs`.

## Copy-on-write disk overlay

Protects disk images from corruption during development and testing. The base
`.raw` file is opened read-only and writes go to a sparse overlay file, with a
`.overlay.dirty` record beside it of which sectors the overlay holds. Delete
both to reset to the clean base.

The record grows as each write lands, so ending iris any way at all (Ctrl-C,
`kill`, `kill -9`, a crash of iris itself) keeps every write the guest had
made. To the guest that is a power cut: its filesystem recovers as it would on
real hardware, from its log.

A crash or power loss of the host is different: see
[Disk write cache](#disk-write-cache) for what each `write_cache` setting
loses.

An older iris reads only what the last flush recorded, so after a kill it
sees the disk as it was before the appended writes, torn as before. Open an
overlay that a newer iris left that way with a newer iris first.

Enable in `iris.toml`:
```toml
[scsi.1]
path = "scsi1.raw"
cdrom = false
overlay = true
```

Writes go to `scsi1.raw.overlay`. Monitor commands:
- `cow status` - show dirty sector count
- `cow commit [id]` - merge overlay into base image (permanent)
- `cow reset [id]` - discard all overlay writes

CHD images get the same protection automatically: writes go
to a `.diff.chd` sidecar. `iris-ci chd-sync` (or `iris-ci quit --sync-chd`, or
the GUI's "Commit changes to disk") folds the diff back into the base.


## Snapshots and rollback

Graphics coverage is incomplete: GR2 saves register/microcode/display tables
but omits VRAM and full drawing-engine state; IMPACT board state is not saved
yet. A snapshot of either board cannot reproduce its complete graphics state.
See [TODO.md](clanker/TODO.md) for remaining device and restore work.

Capture RAM, supported device state, and the COW overlay into
`saves/<name>/`, and restore it later. CPU, MC, IOC, HPC3, REX3, RTC, EEPROM,
SCSI controller, and the Seeq Ethernet chip all round-trip. Current schema
version is 3: postcard-encoded binary device state plus content-addressable
chunked RAM under `saves/.cas/`. A second snapshot taken from the same parent
adds **zero bytes** to disk for any RAM region that didn't change — same
storage model as Docker layers. A snapshot records which CPU it was taken on
(R4400/R5000/R10000) and refuses to restore onto a different CPU model.

The O2 and the Octane save in this format too. An Octane snapshot covers one
processor and no graphics board: with two processors or an IMPACT or VPro board
fitted, a save or load is refused with the reason. Neither board's state is
saved.

From the interactive monitor (`telnet 127.0.0.1 8888`):
```
save base/desktop          # writes saves/base/desktop/
load base/desktop          # restore saved RAM, device state, and disk overlay
```

From `iris-ci` (the wrapper; see "CI control socket and `iris-ci`"):
```bash
iris-ci save base/desktop
iris-ci restore base/desktop          # full disk-backed reload (~150 ms cold)
iris-ci rollback                      # in-memory rewind to last restore (~40 ms)
iris-ci diff base/desktop tests/grep  # what changed: devices, RAM chunks, COW sectors
iris-ci validate base/desktop -n 1000000  # bit-deterministic re-execution check (build with --features ci_clock)
iris-ci tree                          # snapshot parent-chain hierarchy
iris-ci gc                            # sweep CAS chunks no kept snapshot references
iris-ci pull http://reg/snapshots/base   # fetch a snapshot from another machine
```

Two restore tiers:
- **`restore <name>`** — full disk-backed reload. ~150 ms. Use after a hard
  reset or to switch to a different snapshot.
- **`rollback`** — in-memory rewind to the last `restore` checkpoint. ~40 ms,
  no disk I/O. Use this in tight inner test loops where you keep returning to
  the same starting state.

Reflinks are used on APFS / btrfs / xfs so capturing a snapshot of a 4 GB disk
image takes <10 ms and uses ~18 MB of actual disk. Where the filesystem cannot
reflink (NFS, ext4, NTFS, or a copy between two filesystems), a snapshot copies
only the sectors the guest wrote and leaves the rest of the overlay sparse, so
its cost follows the written size, not the disk size.

See [CHANGELOG.md](CHANGELOG.md) for the full feature set, and `clanker/snapshot/`
for the format and its gotchas.


## CI control socket and `iris-ci`

`--ci` enables a control socket for headless automation, plus a
small in-process serial backend so the harness can drive the IRIX console
directly. The default is the Unix socket `iris.sock` in the temporary
directory (`$TMPDIR`, else `/tmp`); on Windows it is TCP `127.0.0.1:19851`.
`iris` and `iris-ci` (and `tools/iris-test`) find each other through the
default only when they run with the same `TMPDIR`; otherwise give both the
path, `--ci-socket` to iris and `--socket` or `$IRIS_SOCKET` to iris-ci.
`--ci` hides the host window unless you add `--ci-display`, while keeping
offscreen graphics alive for screenshots. Explicit `--headless`
disables Newport graphics. Use `--serial-log FILE` to keep a transcript of the
IRIX console.

```
cargo run --release --features lightning -- --ci
```

`cargo build` produces a companion binary, `iris-ci`, that's the **canonical
way** to drive the socket. Don't bother with raw `nc` + JSON unless you're
debugging the wrapper itself.

```bash
# In one terminal: launch iris (Newport window opens; --ci adds a control channel)
./target/release/iris --ci

# In another terminal: drive it
./target/release/iris-ci status --wait-ready 60   # guest state + monitor readiness
./target/release/iris-ci boot          # PROM menu → IRIS console login (one cmd)
./target/release/iris-ci login         # send root + dismiss vt100 prompt + wait #
./target/release/iris-ci run 'ls /'    # send shell command, get stdout + exit code
./target/release/iris-ci save base/multiuser
./target/release/iris-ci put localfile.tar   # copy file into guest, no bs=512 math
./target/release/iris-ci get /tmp/out --to ./out.tar
./target/release/iris-ci diff base mutated   # per-device + chunk + cow-sector deltas
./target/release/iris-ci tree
./target/release/iris-ci script tests/scenario.iris   # batch-run a sequence of cmds
./target/release/iris-ci cdrom-load 4 disc2.iso       # swap a CD at runtime
./target/release/iris-ci rtc-save                     # persist NVRAM
./target/release/iris-ci quit --sync-chd              # fold CHD diffs, then exit
```

Run `iris-ci --help` for the full list, or `iris-ci <subcmd> --help` for any
subcommand. Every operation has a typed clap arg — no JSON quoting, no
hand-managed timeouts.

### Waiting for a fresh machine

The control socket is created only after `Machine::start` has bound the monitor,
so a successful connection (or `iris-ci ping`) also means the monitor is
listening. A harness that launches iris should poll, not sleep:
`iris-ci status --wait-ready 60` retries until `monitor_ready` is true and then
prints the guest state (`idle`, `starting`, `running`, `halted`) and the
monitor address. `--json` emits the same fields as one object. A `--ci` machine
sits in `starting` with the CPU parked until `iris-ci start` (or the monitor's
`machine-start`), whose reply acknowledges the transition; repeating it is a
no-op. See [debug.md](debug.md#machine) for `machine-status` on the monitor
side.

`script FILE` (also spelled `batch FILE`) runs one command per line through a
single control-socket connection, so a multi-step scenario costs one connect
rather than one per line. It prints per-step status and stops at the first
failure; `--keep-going` runs the remaining steps and still exits non-zero.

For automation that doesn't want to depend on `iris-ci`, the underlying socket
protocol is newline-delimited JSON; `cmd` and `args` per request, `{ok, data,
error}` per response. See `src/ci.rs` for the dispatch table.


## Scratch volume — file injection without networking

A SCSI device with `scratch = true` is a host-controlled raw block device for
pushing files into the guest (and pulling artifacts back out) without bringing
up NFS or anything else. iris pre-formats the underlying file with a minimal
SGI Volume Header on first run, and exposes it inside IRIX as
`/dev/rdsk/dks0d2s0`.

Enable in `iris.toml`:
```toml
[scsi.2]
path    = "scratch.raw"
cdrom   = false
overlay = false
scratch = true
size_mb = 64
```

With `iris-ci` (recommended):
```bash
iris-ci put localfile.tar                 # copies host file into the guest
iris-ci get /tmp/output.log --to ./out.log  # pulls a guest file out
```

`iris-ci put`/`get` handle the IRIX `dd bs=512` sector-alignment quirk
transparently — they compute the right block count from the host file size,
issue the right `dd` recipe to the guest, and truncate to the original byte
length on the receiving end.

Manual/raw paths (if you want to drive `dd` yourself):
- Reads MUST use `bs=512` (or any 512-multiple); `bs=64` returns "I/O error".
- Writes must be padded to `bs`; add `conv=sync` for short inputs.
- Inside IRIX: `dd if=/dev/rdsk/dks0d2s0 bs=512 | tar xf -`


