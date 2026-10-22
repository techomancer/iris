# Disk Image Hygiene Rules

## Always use COW overlay or fresh disk for JIT testing

Each emulator crash (kernel panic) corrupts the IRIX XFS filesystem on
scsi1.raw. Subsequent boots from the corrupted image produce TLBMISS panics
that look IDENTICAL to JIT bugs. This confounded at least 5 rounds of store
debugging — crashes were attributed to JIT codegen when they were actually
filesystem corruption from earlier test runs.

**Solutions (use one):**
1. Enable `overlay = true` in iris.toml — base image never modified
2. Re-extract scsi1.raw from the CHD archive before each test
3. Keep a known-good copy: `cp scsi1.raw scsi1.raw.clean`

## Mounting IRIX disk images on Linux

The disk image has an SGI DVH (disk volume header) with partitions. The XFS
root partition is NOT at offset 0.

To find the partition offset:
```python
# Parse SGI DVH partition table at offset 312
# 16 entries of 12 bytes: (nblks: u32be, first_lbn: u32be, type: u32be)
# Type 10 = XFS
```

For a standard IRIX 6.5 install, the root XFS partition is typically at
offset 136314880 (LBA 266240 * 512).

```bash
sudo losetup -o 136314880 /dev/loopN scsi1.raw
sudo mount /dev/loopN /mnt/irix
```

If the filesystem is dirty from a crash:
```bash
sudo xfs_repair -L /dev/loopN   # zeros the dirty log
sudo mount /dev/loopN /mnt/irix
```

The `-L` flag is required — IRIX big-endian XFS dirty logs can't be replayed
by Linux's little-endian XFS driver.
