# Disk browser: an XFS volume's log may not have been replayed

Status: rule (2026-11-22). fs-xfs 0.2.0.

**Finding.** `irix65_Indy.chd` from the NAS (an IRIX 6.5 Indy install) has an
unreplayed XFS log: IRIX refuses `mount -r` with "Dirty XFS filesystem.
Cannot be mounted read-only", on the untouched copy as on the edited one.
fs-xfs edits the volume without looking at the log. Here IRIX replayed the
log on a read-write mount and every browser edit survived, but a replay
rewrites the metadata blocks its transactions name, so an edit to the same
blocks can be undone or the volume left inconsistent.

**Rule.** The browser opens XFS read-only, with the reason, until the user
chooses Edit Anyway (`browser-xfs-log-unknown`). The cure is in the crate: a
log-state check in fs-xfs (find the log head; clean when the last record is
an unmount record), after which the browser can edit clean volumes without
asking. Do not detect it in iris.

**How it was checked.** Boot a pristine copy of the same install with the
edited copy attached as SCSI 2, then `xfs_repair -n /dev/rdsk/dks0d2s0`
(unmounted) and `mount /dev/dsk/dks0d2s0 /mnt2` in the guest; compare with a
pristine copy as SCSI 3. Both report AG 0's unlinked-list error and orphans
12605 and 12607, which were there before any edit.
