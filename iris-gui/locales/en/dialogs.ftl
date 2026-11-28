# New machine and create disk dialogs.

## New machine

dialog-new-machine-title = New Machine
dialog-new-machine-subtitle =
dialog-new-machine-name = Name
dialog-new-machine-model = Model
dialog-new-machine-cpu = Processor
dialog-new-machine-resolution = Display resolution
dialog-new-machine-prom = PROM image
dialog-new-machine-prom-picker = Choose PROM image
dialog-new-machine-prom-filter = PROM image
dialog-new-machine-embedded-prom = Use built-in PROM
dialog-new-machine-nvram = NVRAM file
dialog-new-machine-nvram-picker = Choose NVRAM file
dialog-new-machine-nvram-filter = NVRAM
dialog-new-machine-eeprom = EEPROM file
dialog-new-machine-eeprom-tip = The Indigo2 keeps its Ethernet address and PROM settings here instead of in NVRAM.
dialog-new-machine-eeprom-picker = Choose EEPROM file
dialog-new-machine-eeprom-filter = EEPROM
dialog-new-machine-ram = Memory
dialog-new-machine-ram-total = Total
dialog-new-machine-ram-banks = Set each memory bank
# A memory size in megabytes, in the RAM pickers.
dialog-new-machine-mb = { $size } MB
# $bank is the bank number, 0 to 3.
dialog-new-machine-bank = Bank { $bank }
dialog-new-machine-disks = Disks
dialog-new-machine-hdd = Hard disk (SCSI 1)
dialog-new-machine-hdd-picker = Choose hard disk image
dialog-new-machine-hdd-filter = Disk image
dialog-new-machine-hdd-blank = Start with a blank disk if the file doesn't exist
dialog-new-machine-cdrom = CD-ROM (SCSI 4)
dialog-new-machine-cdrom-picker = Choose CD-ROM image
dialog-new-machine-iso-filter = ISO image
dialog-new-machine-cd-filter = CD image
dialog-new-machine-cdrom-attach = Attach a CD-ROM drive
dialog-new-machine-footer-subtitle =
dialog-new-machine-create = Create

## Create disk

# $id is the SCSI ID the new disk is for.
dialog-create-disk-title = New disk image for SCSI { $id }
dialog-create-disk-file = File
dialog-create-disk-picker-title = New disk image
dialog-create-disk-filter = Disk image
dialog-create-disk-size = Size (MB)
dialog-create-disk-subtitle =
# $size is the existing file's size in megabytes.
dialog-create-disk-exists = ⚠ { $file } already exists ({ $size } MB). Creating the image erases it.
dialog-create-disk-overwrite = Erase the existing image
dialog-create-disk-create = Create

## The New Machine wizard

wizard-step-type = Choose the machine
wizard-step-hardware = Name and hardware
wizard-step-disks = Disks
wizard-step-summary = Ready to create
# $n and $total count the steps.
wizard-step-of = Step { $n } of { $total }
wizard-back = Back
wizard-continue = Continue
wizard-none = None
wizard-no-disc = Empty drive
