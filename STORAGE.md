# Storage, snapshots, and automation

Disk images, overlays, snapshots, the CI control interface, and scratch-volume
workflows. The full configuration inventory is in [FEATURES.md](FEATURES.md);
[HELP.md](HELP.md) covers image preparation and monitor commands. The sections
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


## Copy-on-write disk overlay

Protects disk images from corruption during development and testing. The base
`.raw` file is opened read-only and writes go to a sparse overlay file. Kill
the emulator whenever you want. Delete the overlay to reset to the clean base.

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
See [TODO.md](TODO.md) for remaining device and restore work.

Capture RAM, supported device state, and the COW overlay into
`saves/<name>/`, and restore it later. CPU, MC, IOC, HPC3, REX3, RTC, EEPROM,
SCSI controller, and the Seeq Ethernet chip all round-trip. Current schema
version is 3: postcard-encoded binary device state plus content-addressable
chunked RAM under `saves/.cas/`. A second snapshot taken from the same parent
adds **zero bytes** to disk for any RAM region that didn't change — same
storage model as Docker layers. A snapshot records which CPU it was taken on
(R4400/R5000/R10000) and refuses to restore onto a different CPU model.

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
image takes <10 ms and uses ~18 MB of actual disk.

See [CHANGELOG.md](CHANGELOG.md) for the full feature set, and `rules/snapshot/`
for the format and its gotchas.


## CI control socket and `iris-ci`

`--ci` enables a control socket for headless automation, plus a
small in-process serial backend so the harness can drive the IRIX console
directly. The default is the Unix socket `/tmp/iris.sock`; on Windows it is TCP
`127.0.0.1:19851`. `--ci` hides the host window unless you add `--ci-display`,
while keeping offscreen graphics alive for screenshots. Explicit `--headless`
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


