## The machine settings sheet (sections built by the settings controller).

settings-section-machine = Machine
settings-section-display = Display
settings-section-disks = Disks
settings-section-network = Network
settings-section-advanced = Advanced
settings-machine-type = Machine
settings-cpu-count = Processors
settings-one-cpu = This machine takes one kind of processor.
settings-one-board = This machine takes one graphics board.
settings-next-power-on = Takes effect at the next power on.
settings-memory-banks = Memory banks
settings-o2-display = The O2's display (GBE) is built in.
settings-heads = Heads
settings-disk-missing = The image is missing or can't be read.
settings-disks-summary = { $count ->
    [one] One disk
   *[other] { $count } disks
}
settings-pcap-interface = Host interface
settings-pcap-interface-help = The wired host interface to bridge onto. Leave empty to pick one.
settings-pcap-not-built = This build has no bridged networking; the machine uses NAT.
settings-net-off = Off
settings-advanced-summary = Every setting, as TOML
settings-toml = Complete configuration
settings-toml-help = The machine's TOML file. Applied only when it is valid.
settings-no-free-id = There's no free SCSI ID on the first controller.
settings-filter-disk-images = Disk images
settings-kind-disk = Hard disk
settings-kind-cdrom = CD-ROM
settings-kind-daynaport = DaynaPORT network
settings-add-disk = Add Disk
settings-add-cdrom = Add CD-ROM
settings-overlay = Copy-on-write
settings-overlay-tip = Writes go to a separate file and the disk image stays unchanged.
settings-scratch = Scratch disk
settings-choose = Choose…
settings-window-title = { $name } Settings
settings-no-machine = Select a machine to change its settings.

## Application preferences (app-wide, not per machine).

prefs-window-title = Preferences
prefs-display-size = Display size
prefs-display-actual = Actual Size
prefs-display-fit = Scaled (keep aspect ratio)
prefs-display-stretch = Scaled (stretch)
prefs-integer-when-fit = Used when the display is scaled keeping its aspect ratio.
prefs-integer-only = Whole multiples only
prefs-integer-only-help = Scale the guest display by 1×, 2×, 3× and so on, so every pixel is the same size.
prefs-filter = Smoothing
prefs-filter-auto = Sharp at whole multiples
prefs-filter-nearest = Always sharp
prefs-filter-linear = Always smooth
settings-pin-first = Turn on Pin threads to cores first.
settings-performance-summary = Host threads, the JIT and the VNC server
prefs-pointer = Mouse
prefs-pointer-capture = Capture it (click the display)
prefs-pointer-follow = Follow the Mac pointer
prefs-pointer-help = Follow suits remote desktops such as RustDesk or Screen Sharing, which move the pointer without reporting motion. The guest's pointer is steered after yours, so it trails fast moves a little. The keyboard reaches the guest while the display is focused; Command shortcuts stay with the Mac.
prefs-pointer-no-capture = The mouse follows the Mac pointer; there is nothing to capture (Preferences > Mouse).
settings-group-workstations = Workstations
settings-group-servers = Servers
settings-memory = Memory
settings-memory-help = The machine's memory. Only amounts it can hold are offered; the banks are filled for you (Advanced shows them).
# $machine is a machine type, such as "SGI Tezro (IP53)".
prefs-tab-general = General
prefs-tab-display = Display
