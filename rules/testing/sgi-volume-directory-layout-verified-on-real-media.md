# SGI volume directory: layout verified against real media

`struct volume_header`'s volume directory sits at **0x048**, 15 entries of 16
bytes: `char vd_name[8]; int vd_lbn; int vd_nbytes;` — big-endian, `vd_lbn` in
512-byte blocks from the start of the *volume*, `vd_nbytes` the exact byte
length. This was confirmed by parsing media IRIX itself wrote, not by
round-tripping our own writer (`mkvh dump`, Feature A):

- An installed IRIX 6.5 system disk (`Indy-IRIX65_dev.chd`, sector 0):

  ```
  bootfile  "/unix"          checksum valid
    ide      lbn        2      343040 bytes
    sash     lbn      672      343040 bytes
    slot  0  first 266240  nblks 8122368  type 10 (PT_XFS)
    slot  1  first   4096  nblks  262144  type  3 (PT_RAW)
    slot  8  first      0  nblks    4096  type  0 (PT_VOLHDR)
    slot 10  first      0  nblks 8388608  type  6 (PT_VOLUME)
  ```

  Both files start with `0x0163` (SGI ECOFF, MIPSEB) at exactly `lbn * 512`,
  and `ide` ends (1024 + 343040 = 344064) precisely where `sash` begins
  (672 * 512) — contiguous, so the field offsets can't be coincidence.

- IRIX 6.5.22 / nekoware / mkisofs-built CDs parse with a **valid checksum and
  an empty voldir**. Data CDs are not bootable-by-name; do not read an empty
  voldir as a parse failure. Only media prepared with `dvhtool` (or an SGI
  installer) carries entries.

Gotchas:

- An 8-character name fills `vd_name` with **no NUL terminator** — parse the
  8 bytes and trim, never `strlen`. `sashARCS` is exactly 8.
- A used slot is detected by `vd_name[0] != 0`; entries need not be contiguous.
- The checksum (0x1F8) covers all 128 big-endian words and must sum to zero, so
  it has to be recomputed after *every* mutation, voldir writes included.
- The volume-header partition (slot 8) must be large enough to contain the
  files the voldir points at — the real disk above uses 4096 blocks for
  ~672 KB of `sash`/`ide`, not the 8-block minimum `create_scratch_image()`
  writes for the scratch volume.
