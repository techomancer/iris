## The disk browser: a machine's disk or CD-ROM, its partitions and files.

browser-title = { $machine }: { $device }
browser-partition = Partition { $index }
browser-why-whole-volume = the whole disk
browser-why-swap = swap
browser-why-unformatted = not formatted
browser-why-unsupported = { $kind }, not browsable
browser-no-label = This disk has no SGI volume header: { $error }
browser-open-failed = Couldn't open the disk image: { $error }
browser-ro-cdrom = CD-ROMs are read-only.
browser-ro-running = The machine is running, so this is read-only, and what you see can be a few seconds out of date.
browser-ro-volume-header = The volume header is read-only.
browser-ro-crate = This filesystem can be read but not changed here: { $reason }
browser-read-only = This filesystem is read-only.
browser-check-failed-ro = The filesystem check found { $count ->
    [one] a problem
   *[other] { $count } problems
}, so it opened read-only. Check it in IRIX, or choose Edit Anyway.
browser-busy = Wait for the current task to finish.
browser-no-partition = Choose a partition first.
browser-select-first = Select something first.
browser-select-one = Select one item.
browser-not-a-file = Only files can be edited.
browser-too-big = Files over { $limit } open with Export, not the editor.
browser-not-text = This file isn't text. Export it to look at it on this computer.
browser-move-not-yet = Moving to another folder needs a newer rusty-backup release.
browser-bad-name = A name can't contain / or be . or ..
browser-name-taken = Something with that name is already here.
browser-bad-mode = Type the mode in octal, like 755 or 644.
browser-bad-owner = Type user:group, by name or number.
browser-opening = Opening
browser-reading-names = Reading users and groups
browser-checking = Checking
browser-listing = Reading the folder
browser-reading = Reading
browser-saving = Saving
browser-applying = Applying
browser-deleting = Deleting
browser-exporting = Exporting
browser-importing = Importing
browser-exported = Exported { $size }.
browser-imported = Imported { $size }.
browser-cancelled = Cancelled.
browser-failed = It didn't work
browser-check-title = Filesystem check
browser-check-clean = No errors{ $warnings ->
    [0] .
   *[other] , { $warnings } warnings.
}
browser-check-errors = { $errors } errors, { $warnings } warnings.
browser-check-unsupported = This filesystem has no check.
browser-export-title = Export to
browser-export-archive-title = Export as archive
browser-import-title = Import
browser-import-body = Copy { $count ->
    [one] one item
   *[other] { $count } items
} ({ $size }) into { $dest }.
browser-free-space = Free on this partition: { $free }.
browser-no-room = There isn't enough free space.
browser-import-clash = Already here: { $names }
browser-import-skip-existing = Import, keep existing
browser-import-replace = Import, replace existing
browser-import-unpack = Unpack
browser-import-as-file = Copy as one file
browser-unpack-title = Unpack archives?
browser-unpack-body = { $count ->
    [one] { $names } is an archive.
   *[other] { $count } archives: { $names }.
} Unpack into a folder named after { $count ->
    [one] it
   *[other] each
}, or copy { $count ->
    [one] it
   *[other] them
} in as { $count ->
    [one] one file
   *[other] files
}?
browser-unpack-more = , and { $count } more
browser-permissions-title = Permissions
browser-permissions-prompt = Mode (octal):
browser-owner-title = Owner
browser-owner-prompt = Owner and group (user:group):
browser-rename-title = Rename
browser-name-prompt = Name:
browser-delete-title = Delete { $count ->
    [one] this item
   *[other] { $count } items
}?
browser-delete-body = This deletes { $names } from the disk. It can't be undone.
browser-delete-body-dirs = This deletes { $names } and everything in them from the disk. It can't be undone.
browser-xfs-dirty = IRIX was not shut down cleanly: this filesystem's log holds { $count ->
    [one] a change
   *[other] { $count } changes
} IRIX will replay at its next boot, so it opened read-only. Boot the machine and shut IRIX down cleanly, or choose Replay Log.
browser-xfs-unrecoverable = IRIX was not shut down cleanly, and this filesystem's log cannot be replayed here ({ $reason }), so it opened read-only. Boot the machine to let IRIX recover it.
browser-replay = Replay Log
browser-replay-title = Replay the log?
browser-replay-body = This writes the { $count ->
    [one] logged change
   *[other] { $count } logged changes
} into the filesystem, as IRIX would at its next boot, and marks it cleanly unmounted. Nothing is written if the log cannot be replayed completely.
browser-replay-ok = Replay
browser-replaying = Replaying the log
browser-replayed = Replayed { $count ->
    [one] one change
   *[other] { $count } changes
}; the filesystem is clean.
browser-unix-line-endings = Text files: Unix line endings
browser-edit-anyway-title = Edit a damaged filesystem?
browser-edit-anyway-body = The check found problems. Changing the filesystem now can make them worse. Running fsck in IRIX first is safer.
browser-edit-anyway-ok = Edit anyway
# $id is the SCSI id, as "1" or "1:3".
browser-label-disk = Disk { $id }
browser-label-cdrom = CD-ROM { $id }
browser-no-disks = This machine has no disk image to browse.
browser-which-title = Which disk do you want to browse?
browser-tb-up = Up
browser-tb-refresh = Refresh
browser-tb-import = Import Files…
browser-tb-import-folder = Import Folder…
browser-tb-export = Export…
browser-tb-export-tgz = Export as .tar.gz…
browser-tb-export-tar = Export as .tar…
browser-tb-new-folder = New Folder…
browser-tb-edit = Edit Text
browser-tb-rename = Rename…
browser-tb-move = Move…
browser-tb-permissions = Permissions…
browser-tb-owner = Owner…
browser-tb-delete = Delete…
browser-tb-check = Check Filesystem
browser-edit-anyway = Edit Anyway
browser-col-name = Name
browser-col-size = Size
browser-col-mode = Mode
browser-col-owner = Owner
browser-col-group = Group
browser-col-date = Modified
browser-editor-save = Save
browser-editor-close = Close
browser-editor-read-only = Read-only
browser-empty = This folder is empty.
browser-closed-stopped = The machine stopped, so the disk browser closed. Browse Files again to edit the disk.
