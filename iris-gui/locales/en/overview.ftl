## The stopped machine's summary page (native shells).

ov-spec = { $model } ({ $board })
ov-processor = Processor
ov-memory = Memory
ov-graphics = Graphics
ov-prom = PROM
ov-prom-builtin = Built-in PROM
ov-scsi = SCSI { $id }
ov-fc = Fibre Channel { $id }
ov-empty = No image
ov-no-disc = CD-ROM, no disc
ov-cdrom = CD-ROM, { $disc }
ov-daynaport = DaynaPort Ethernet
ov-missing = missing
ov-cow = copy-on-write
ov-network = Network
ov-net-nat = NAT, { $subnet }
ov-net-bridged = Bridged to { $iface }
ov-net-stream = Stream to { $peer }
ov-forwards = { $count ->
    [one] one port forward
   *[other] { $count } port forwards
}
ov-saved-states = Saved states
ov-no-states = No saved states yet.
ov-browse = Browse files
ov-edit-settings = Edit Settings…
ov-no-machine = No machine yet. Choose File > New Machine.
ov-restore = Restore
ov-rename = Rename
ov-delete = Delete
ov-details = Details
ov-current-state = Current state
ov-not-running = Not running
ov-save-while-running = Start the machine to save its state.
ov-saved-at = Saved { $when }
