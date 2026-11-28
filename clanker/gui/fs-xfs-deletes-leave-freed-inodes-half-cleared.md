# fs-xfs deletes leave freed inodes half-cleared; its log replay miscounts free blocks

Status: open (upstream), 2026-11-27, fs-xfs 0.2.3 (also rusty-backup main).

**Symptom.** After the disk browser edits an IRIX 6.5 XFS volume, IRIX's
`xfs_check` on the unmounted volume reports, beyond what the unedited image
already had:

- `bad nlink 1 for free inode N`, often with `bad nblocks K for free inode N`,
  one per file the browser deleted (Delete, and the old file an editor save
  replaces).
- `sb_fdblocks X, counted Y`, with Y about 30 blocks above X, after any
  browser log replay, even with no edits at all.

The files themselves are right, and IRIX mounts and boots the volume.

**Cause.**

- `free_inode` (`fs-xfs/src/xfs/edit/inode_alloc.rs`) sets the inobt free
  bit and zeroes only `di_mode`. `di_nlink` (v1 `di_onlink` too), `di_size`,
  `di_nblocks` and `di_nextents` keep the live file's values. An inode that a
  later create reuses at once hides it, so "import, replace existing" looks
  clean in a test.
- `resync_sb_counters` in the replay (`log_replay.rs`) sets `sb_fdblocks`
  to the sum of `agf_freeblks + agf_flcount + agf_btreeblks`. IRIX's own
  recovery leaves the counter as logged, and its `xfs_check` agrees with that
  value, not with the recomputed one, on these non-lazy-count volumes.

**How to check XFS edits.** Boot a clean IRIX 6.5 system disk with the edited
volume as a second drive and an unedited copy of the same image as a third.
Mount and unmount the unedited copy first (IRIX recovers its log), then
`xfs_check /dev/rdsk/dks0dNs0` on both while unmounted, and diff the output:
what the baseline also says came with the image. One action per copy
(hard links to one base CHD give each its own `.diff.chd`) tells which action
did what.

**Fix.** Upstream, in rusty-backup's fs-xfs: clear the freed dinode as XFS
does, and keep the logged `sb_fdblocks` on a volume without lazy counters.
Until then the browser's XFS edits are safe to boot, and `xfs_repair` in IRIX
tidies the leftovers.
