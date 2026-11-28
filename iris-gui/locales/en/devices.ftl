# SCSI menu, serial console and camera test.

## SCSI menu

# Slot labels in the SCSI menu. $id is the SCSI ID (1 to 7).
scsi-slot-empty = SCSI { $id }: empty
scsi-slot-daynaport = SCSI { $id }: DaynaPort Ethernet
scsi-slot-cdrom-empty = SCSI { $id }: CD-ROM, no disc
# $path is the disc image file that can't be found.
scsi-slot-cdrom-missing = SCSI { $id }: CD-ROM ⚠ { $path } not found
# $name is the disc image's file name.
scsi-slot-cdrom = SCSI { $id }: CD-ROM, { $name }
# $name is the disk image's file name, $size its size in megabytes.
scsi-slot-hdd = SCSI { $id }: { $name }
scsi-slot-hdd-overlay = SCSI { $id }: { $name } [overlay]
scsi-slot-hdd-size = SCSI { $id }: { $name } ({ $size } MB)
scsi-slot-hdd-size-overlay = SCSI { $id }: { $name } ({ $size } MB) [overlay]
scsi-menu-subtitle =

scsi-attach-hdd = Attach hard disk…
scsi-attach-hdd-picker = Attach hard disk
scsi-attach-cdrom-empty = Attach empty CD-ROM drive
scsi-attach-cdrom-disc = Attach CD-ROM with disc…
scsi-attach-cdrom-disc-picker = Attach CD-ROM with disc
scsi-create-blank = Create blank disk image…
scsi-daynaport-subtitle =
scsi-detach-daynaport = Detach DaynaPort
scsi-eject = Eject disc
scsi-insert-disc = Insert disc…
scsi-change-disc = Change disc…
scsi-insert-disc-picker = Insert disc
scsi-detach-cdrom = Detach CD-ROM drive
scsi-overlay-enable = Enable write overlay
scsi-overlay-disable = Disable write overlay
scsi-overlay-tip = Writes go to a separate .overlay file and the disk image stays unchanged.
scsi-replace-image = Replace image…
scsi-replace-image-picker = Replace hard disk image
scsi-detach-hdd = Detach hard disk
scsi-filter-disk = Disk images
scsi-filter-iso = CD images
scsi-filter-all = All files

# Notifications after a SCSI change. $id is the SCSI ID.
scsi-attached-hdd = Hard disk attached to SCSI { $id }.
scsi-attached-cdrom-empty = Empty CD-ROM drive attached to SCSI { $id }. If the machine is running, restart it to use the drive.
scsi-attached-cdrom-disc = CD-ROM drive with disc attached to SCSI { $id }.
scsi-disc-inserted = Disc inserted in SCSI { $id }.
scsi-disc-ejected = Disc ejected from SCSI { $id }.
scsi-detached = SCSI { $id } detached.
scsi-overlay-on = Write overlay on for SCSI { $id }.
scsi-overlay-off = Write overlay off for SCSI { $id }.

## Serial console

# $addr is a host:port address, $error the system's error text.
serial-connect-failed =
    Couldn't connect to the serial console at { $addr }: { $error }
    Start the machine, then open the console again.
serial-socket-failed = The serial console connection failed: { $error }

## Camera test

camera-opening = Opening camera…
# egui's SCSI menu only (the Mac app no longer offers it).
scsi-mount-in-irix = Mount disc in IRIX
scsi-mount-in-irix-tip = Mounts the disc at /CDROM. A shell must be open on the console.
