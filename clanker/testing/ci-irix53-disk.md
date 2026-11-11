# The stock IRIX 5.3 disk the nightly boot test uses

**Keywords:** ci,boot-test,irix53,indy,chd,sha256,golden,rotate
**Category:** testing

`build.yml`'s `boot-test` job boots this disk on Linux, macOS and Windows. It is
an unmodified IRIX 5.3 install for the Indy, with no patches, no accounts and
no root password. Nothing on it is secret, so the job needs no repository
secret and a fork can run it.

| | |
|---|---|
| URL | `https://origami-dist.irix.fans/irix/5.3/indy-3a9ea895.chd` |
| CHD size | 222,031,005 bytes |
| CHD sha256 | `3a9ea8952be5a55cdca5559916264e3be7883b037869d4ce47ea1cef00a850de` |
| Raw disk | 4 GiB (4,294,967,296 bytes, sparse), sha256 `af4b6102beb855374ef4d2b6dfde26d6d8843caa2202e097d14675fdb6dfd1fb` |

The CHD's checksum also appears in the `SHA256SUMS` file at the root of
`origami-dist.irix.fans`. The path carries the first eight hex digits of the
CHD's sha256, so a rebuilt disk is a new URL and an old URL never changes
under a pinned workflow.

## Provenance and recipe

Built on 2026-11-10 with iris itself, following the IRIX 5.3 sections of
[irix-install.md](../irix/irix-install.md) with every default:

- Media: the "IRIX 5.3 for Indy including R5000" install CD image, 527,779,840
  bytes, sha256 `00bf857d0b409900c171e1eefce288cda69aca0ca80954d1b5d279b5c63842b0`.
- Machine: Indy, 128 MB + 128 MB RAM, a blank 4 GiB raw disk on SCSI 1, the CD
  on SCSI 4.
- Seed NVRAM headless (`SystemPartition scsi(0)disk(1)rdisk(0)partition(8)`,
  `OSLoadPartition ...partition(0)`, `OSLoadFilename /unix`, `console d`), boot
  the standalone `fx` from the CD to write the default label, boot the miniroot,
  make the file systems on `s0` and `s6`, and `install *` from `/CDROM/dist`.
- Leave the installer's own kernel reconfiguration to finish, answer `y` to
  `Restart?`, then halt with `/etc/shutdown -y -g0 -i0` and quit iris.
- Compress with `chdman createhd -i irix53-indy-stock.raw -o <name>.chd`
  (MAME's chdman 0.276, default compression, 4096-byte hunks). `chdman
  extractraw` of the CHD reproduces the raw checksum and `chdman verify` passes.

On this disk `/etc/inittab` has `t1:23:respawn:/sbin/getty ttyd1 co_9600`, so
`IRIS login:` reaches the serial console in either PROM console mode. A
first boot after the install went straight to the login with no fsck and no
kernel reconfiguration, which the previous CI disk did not do.

## How CI pins and checks it

The job's `env` holds `IRIX53_IMAGE_URL` and `IRIX53_IMAGE_SHA256`. The
download step fetches the URL with `curl -fL --retry 3` and runs `sha256sum -c`
(`shasum -a 256 -c` on macOS) before anything reads the file, so a truncated or
swapped download fails the job. There is no Actions cache: the repository's
10 GB quota is full of Rust caches and a 222 MB entry was evicted before the
next night.

`scripts/ci-boot-test.sh` opens the CHD with `overlay = true`, which keeps
writes in a `<name>.chd.diff.chd` beside it and reuses that file on the next
open. A local run in a reused directory therefore starts from the last run's
writes. Copy the CHD into a fresh directory for every run.

## Rotating it

1. Build the new disk and publish it under a path that carries the new CHD
   hash, as above. Leave the old object alone.
2. Run `bash scripts/ci-boot-test.sh <bin-dir> <fresh-copy>.chd <out> 600`
   with a release build of the CLI (`cargo build --release --bin iris --bin
   iris-ci` with the nightly features in `scripts/dist/lib.sh`). It fails the
   graphics check against the old golden and leaves `gfx-screen.png`.
3. Look at `gfx-screen.png`. It must be the login screen of the new disk. Copy it
   to `scripts/boot-goldens/indy-irix53.png`, run the script again from another
   fresh copy (exit 0, six passing cases in `boot-test.xml`) and run `cargo
   test --release --bin iris-ci` for the golden's unit test.
4. Change `IRIX53_IMAGE_URL` and `IRIX53_IMAGE_SHA256` in `build.yml`, the table
   above and the golden in one commit, so the job never runs the new disk against
   the old golden or the reverse.

## What the golden looks like

The login screen of this disk shows five accounts: root, EZsetup, demos, guest
and 4Dgifts. The screenshot taken when the disk was built differs from the
golden by 56 pixels (the text cursor in the login field), and the golden
differs from the previous disk's screenshot by 1,720 pixels (an extra account),
against the 128 that `screen-wait` allows.
