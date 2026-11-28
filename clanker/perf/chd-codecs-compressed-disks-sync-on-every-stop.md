# A compressed CHD disk pays a whole-disk rebuild on every stop; uncompressed pays none

Measured 2026-11-23 on an Apple M1 (8 cores), with a 4.29 GB IRIX 6.5.22 Indy
root disk (4 KB hunks, 512-byte units), libchdman-rs 0.289.3. Single runs:
treat differences under about 5 s as noise.

## Why it matters

A compressed CHD cannot be written in place, so the guest writes to a
`.diff.chd` and every stop folds it back (`chd_disk::flatten_diff`), which
rebuilds and recompresses the **whole** disk however little changed. That is
the GUI's "Synchronizing disks…" wait. An uncompressed v5 CHD is written in
place and never folds.

## Host-side (scratch tool over libchdman-rs)

| Format (codecs) | File size | Convert from the lzma original | Rebuild in the same format (= one sync) | Read the whole disk |
|---|---|---|---|---|
| chdman HD default (lzma, zlib, huff, flac) | 1.15 GB | — | 167 s | 55–60 s |
| zstd only | 1.22 GB | 160–167 s | 140 s | 22 s |
| zlib only | 1.27 GB | 116 s | 100 s | 38 s |
| uncompressed | 2.88 GB | 62–74 s | never needed | 6.3 s |

Random 4 KB reads cost under 0.15 ms each in every format: not a factor.
The uncompressed file is not the full 4.29 GB because all-zero hunks are not
stored. zstd rebuilds only 16% faster than lzma because MAME's CHD zstd
compressor uses a high compression level: zstd decodes fast but encodes
slowly here, and zlib alone rebuilds faster than zstd.

## In the Mac app (release bundle, IRIX 6.5.22 on the Indy)

| Format | Start to telnet login | `shutdown -p` to power-off | Power-off to synced |
|---|---|---|---|
| uncompressed | 26.2 s | 59.1 s | 0 s |
| lzma (chdman default) | 30.3 s | 90.0 s | 143.2 s |
| zstd | 25.6 s | 83.6 s | about 110 s (from the rebuilt file's time) |
| zlib (converted in the app) | 30.5 s | 92.2 s | about 65 s (from the rebuilt file's time) |

## Rule

The Mac app offers this conversion (`controller::convert`), zlib first:
it keeps the file about the size it was and halves the wait after a stop,
while uncompressed (no wait at all) can be very large for a big disk.
lzma to zlib took 95 s in the app once App Nap was held off
(`clanker/macos/app-nap-throttles-background-work.md`). zstd is not worth
offering: it saves about 30 s per stop over lzma, syncs slower than zlib
(MAME compresses it at a high level) and needs chdman/MAME 0.247 or later.
