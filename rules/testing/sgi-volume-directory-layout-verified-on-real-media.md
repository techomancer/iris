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

- The IRIX 6.5.22 *Installation Tools and Overlays (1 of 3)* CD — the reference
  for the writer, reproduced exactly by `mkvh dump`:

  ```
  magic 0x0BE5A941  rootpt 0  swappt 0  bootfile ''   csum VALID
    sgilabel lbn      32          512 bytes
    mr       lbn      33     19200000 bytes      (the miniroot)
    sash64   lbn   37533       266000 bytes
    sashARCS lbn   38053       342532 bytes      (what an Indy boots)
    slot  7  first  48736  nblks 1283016  type 5 (PT_SYSV)
    slot  8  first      0  nblks   48736  type 0 (PT_VOLHDR)
    slot 10  first      0  nblks 1331776  type 6 (PT_VOLUME)
  ```

- Data CDs (nekoware, mkisofs-built, the "full" 6.5.22 image) parse with a
  **valid checksum and an empty voldir**. They are not bootable-by-name; an
  empty voldir is not a parse failure. Only media prepared with `dvhtool` or an
  SGI installer carries entries.

Gotchas:

- An 8-character name fills `vd_name` with **no NUL terminator** — parse the
  8 bytes and trim, never `strlen`. `sashARCS` is exactly 8.
- A used slot is detected by `vd_name[0] != 0`; entries need not be contiguous.
- The checksum (0x1F8) covers all 128 big-endian words and must sum to zero, so
  it has to be recomputed after *every* mutation, voldir writes included.
- The volume-header partition (slot 8) must **span every file the voldir points
  at**. The install CD gives it 48736 blocks (~24 MB) for sash/miniroot and the
  installed disk 4096 blocks for ~672 KB of `sash`/`ide` — not the 8-block
  minimum `create_scratch_image()` writes for a file-less scratch volume. The
  filesystem partition then starts exactly where it ends (block 48736 on the
  CD), which is what `PartitionSpec { first_block: None }` emits.
- **An EFS filesystem partition is type 5 (`PT_SYSV`), not type 7 (`PT_EFS`).**
  Counter-intuitive, but it is what shipping SGI media emits: partition 7 of the
  install CD is type 5, and the EFS superblock is really there — block 1 of that
  partition has `fs_magic = 0x00072959` at offset 0x1c with `fs_size = 1280320`.
  Emit what the CD emits, not what the type name suggests.
- Both PROM loader paths appear on that one CD: `sashARCS` is ECOFF
  (`f_magic 0x0163` MIPSEBMAGIC, OMAGIC, `F_EXEC`, text at 0x10000000, entry
  0x10021020) and `sash64` is an ELF64 MSB `ET_REL` object the PROM relocates
  (the PROM's "Cannot load and relocate %s." string). 0x10000000 is therefore a
  load address this PROM demonstrably accepts.
