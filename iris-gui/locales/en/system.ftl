# Machine folders, settings storage, the emulator worker. See ../README.md for conventions.

## Stop confirmation

# $id is a SCSI ID (a number).
safestop-writable-disk = SCSI disk { $id } is read-write. Stopping while IRIX is running can corrupt its filesystem.

## Machine folders

machines-name-invalid = Machine names can't contain slashes or colons, start with a dot, or end with a dot or space.
machines-name-reserved = This name is reserved on Windows or uses characters that aren't allowed in file names.
# A file that couldn't be read or written, followed by the system's error text.
machines-file-error = { $path }: { $error }
machines-list-error = Can't read the machines folder: { $error }
machines-migrate-error = Couldn't copy { $path } into the machine folder: { $error }
machines-folder-exists = A machine named { $name } already exists.

## Settings storage

settings-no-config-dir = Can't find the settings folder.
settings-no-storage = Can't find the settings folder or the current folder.
settings-create-error = Can't create the settings folder: { $error }
settings-backup-error = Couldn't back up your preferences: { $error }
settings-migration-failed = Couldn't move your machines to the new format: { $error }
settings-locked = Fix the problem loading your preferences before changing machines.
settings-not-saved = Preferences weren't saved because they failed to load. The file on disk is unchanged.

## Emulator
## Messages from the emulator worker, shown as notifications.

emu-already-running = The machine is already running.
emu-not-running = The machine isn't running.
# $error is the reason the machine failed to start.
emu-start-failed = Couldn't start the machine: { $error }
# $subnet is a CIDR such as 192.168.0.0/24.
emu-nat-subnet-invalid = Can't use { $subnet } as the NAT subnet: { $error }
emu-chd-compact-failed = Couldn't compact CHD disks on exit: { $error }. Grant access to the disk's folder (File > Grant a disk folder…) so IRIS can write next to it.
emu-commit-failed = Couldn't commit the disk changes: { $error }
emu-rollback-failed = Couldn't discard the disk changes: { $error }
# $name is the saved state's name.
emu-save-failed = Couldn't save state "{ $name }": { $error }
emu-save-crashed = Saving state crashed: { $error }
emu-restore-failed = Couldn't restore state "{ $name }": { $error }
emu-restore-crashed = Restoring state crashed: { $error }
emu-screenshot-no-frame = Nothing to capture yet. The machine hasn't drawn anything.
emu-screenshot-failed = Couldn't save the screenshot: { $error }
# $id is a SCSI ID; $file is a disc image file name.
emu-disc-loaded = SCSI { $id }: loaded { $file }.
emu-disc-loaded-remount = SCSI { $id }: loaded { $file } and asked IRIX to remount /CDROM.
emu-disc-ejected = SCSI { $id }: disc ejected.
emu-remount-sent = SCSI { $id }: asked IRIX to remount /CDROM.
emu-scsi-error = SCSI { $id }: { $error }
emu-stop-spawn-failed = Couldn't stop the machine: no thread was available to stop it.
emu-stop-failed = Couldn't stop the machine cleanly: { $error }
emu-stop-timeout = The machine didn't stop within 5 seconds and was abandoned.
# Stands in for an error message when a crash gave none.
emu-unknown-error = unknown error

## Memory

# $total is megabytes; $banks is the number of memory banks in use.
ram-summary = { $total } MB ({ $banks ->
    [one] 1 bank
   *[other] { $banks } banks
})
