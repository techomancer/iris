# Machine configuration tabs (config_ui.rs). See ../README.md for conventions.

## Tabs

config-tab-general = General
config-tab-disks = Disks
config-tab-network = Network
config-tab-memory = Memory
config-tab-display = Display
config-tab-video-in = Video in
config-tab-debug = Debug
config-tab-ci = Automation
config-tab-bench = Benchmark

## Shared

# Shown when a setting differs between the running machine and the saved config.
# $running and $pending are a CPU model or a RAM summary such as "128 MB".
config-pending = Running: { $running }. After restart: { $pending }.
# $size is a number of megabytes.
config-megabytes = { $size } MB
# Button that fills in a suggested value; $value is an address or folder.
config-use = Use { $value }
config-remove = Remove
config-reveal-tip = Show in file manager
# Title of the file picker opened by a field's browse button.
config-browse-title = Choose a file
config-filter-prom = PROM images
config-filter-nvram = NVRAM files
config-filter-disk = Disk images
config-filter-iso = ISO images
config-filter-perfetto = Perfetto traces
config-filter-all = All files

## General

# $platform is the machine model, such as "Indy (IP24)".
config-platform = Platform: { $platform }
config-platform-subtitle =
config-viewport-subtitle =
config-indigo2-subtitle =
config-ip28-subtitle =
config-ip28-graphics-warning = IRIX has no Newport driver for this machine. Choose an IMPACT graphics board.

config-processor = Processor
config-cpu = CPU
config-cpu-locked = Stop the machine to change the CPU.
config-cpu-applies-subtitle =
config-cpu-subtitle =
config-r10000-subtitle =
config-cache-model = Cache model
config-cache-model-tip = Full is fastest with the interpreter, Shadow with the JIT.
config-cache-shadow-only = Shadow (the only R10000 option)
config-cache-auto-shadow = Auto (shadow)
config-cache-auto-full = Auto (full)
config-cache-full = Full
config-cache-shadow = Shadow
config-clock = CPU clock
# $mhz is the default clock speed in MHz.
config-clock-auto-tip = Use the default, { $mhz } MHz.

config-jit-threads = JIT compile threads
config-jit-threads-tip =
config-jit-cache = Keep compiled code between runs
config-jit-cache-tip =
config-jit-cache-dir = JIT cache folder
config-jit-cache-dir-hint = Default location

config-newport-heads = Newport heads
config-newport-dual-subtitle =

config-prom = PROM image
config-prom-builtin = Use built-in PROM
config-prom-builtin-tip =
config-prom-builtin-active = Using the built-in PROM
config-nvram = NVRAM file
config-nveeprom = EEPROM file
config-nveeprom-tip = Indigo2 keeps its Ethernet address and PROM settings here. Indy doesn't use it.
config-serial-log = Console log file

config-ultra64 = N64 development board (Ultra64)
config-ultra64-tip = Needs the gopher64 ultra64 fork running alongside IRIS.
config-autosave-subtitle =

## Host ports

config-host-ports = Host ports
config-port-monitor = Monitor console
config-port-monitor-tip = Connect with telnet to 127.0.0.1 on this port.
config-port-serial1 = Serial port 1 (console)
config-port-serial1-tip = The PROM and IRIX console.
config-port-serial2 = Serial port 2
config-port-serial2-tip =
# $port is the default port number.
config-port-default-tip = Use { $port }.
config-ports-conflict = Two of these use the same port. Only the first to start gets it.
config-host-ports-subtitle =

## Real-time clock

config-rtc = Real-time clock
config-rtc-subtitle =
config-rtc-years = Years
config-rtc-months = Months
config-rtc-days = Days
config-rtc-hours = Hours
config-rtc-minutes = Minutes
config-rtc-seconds = Seconds
# $time is a date and time such as "2001-05-17 09:30:00".
config-rtc-preview = Guest clock starts at { $time } UTC
config-rtc-reset = Reset
config-rtc-clamped = The clock chip only holds 1970 to 2039, so the time will be clamped.
config-rtc-running-subtitle =
config-rtc-applies-subtitle =

## Graphics

config-graphics-board = Graphics board
config-graphics-indigo2-only = Indigo2 only
config-graphics-locked = Stop the machine to change the graphics board.
config-graphics-gr2-subtitle =
config-graphics-impact-ip28-subtitle =
config-graphics-impact-subtitle =
config-resolution = Display resolution
config-resolution-newport-only-tip = Only Newport graphics can change resolution.
config-resolution-newport-only-subtitle =
config-resolution-subtitle =
config-resolution-locked = Stop the machine to change the resolution.

## Memory

config-ram-locked = Stop the machine to change RAM.
# $ram is a RAM summary such as "128 MB".
config-ram-running = Running: { $ram }
config-ram-applies-subtitle =
config-ram-subtitle =
config-ram-presets = Presets
config-ram-banks = Bank sizes
config-ram-banks-subtitle =
config-ram-banks-ip28-subtitle =
# $bank is the bank number, 0 to 3.
config-ram-bank = Bank { $bank }
config-ram-bank-ip28-only = 256 and 512 MB banks need the IP28 platform.
config-ram-total = Total: { $ram }

## Display

config-scale = Window scale
# $scale is a whole number, such as 2 for double size.
config-scale-factor = { $scale }×
config-headless = No graphics (headless)
config-no-audio = No audio
config-keyboard = Keyboard
config-keyboard-tip = The physical shape of your keyboard. Set the language layout in the PROM.
config-audio = Audio
config-audio-subtitle =
config-audio-host = Host audio
config-audio-host-default = Default (ALSA)
config-audio-host-alsa = ALSA
config-audio-host-pulse = PulseAudio / PipeWire
config-audio-prebuffer = Pre-buffer (ms)
config-audio-host-buffer = Host buffer (frames)
config-audio-rate-subtitle =

## Disks

config-scsi = SCSI devices
config-scsi-subtitle =
# $id is the SCSI ID, 1 to 7.
config-scsi-id = SCSI ID { $id }
config-scsi-attach = Add device
config-scsi-type = Type
config-scsi-type-disk = Hard disk
config-scsi-type-cdrom = CD-ROM
config-scsi-type-daynaport = DaynaPort (Ethernet)
config-scsi-mac = MAC address
config-scsi-mac-tip = Leave blank to derive one from the SCSI ID.
config-scsi-subnet = NAT subnet
config-scsi-subnet-tip = This adapter has its own private network. Leave blank for 192.168.10.0/24.
config-daynaport-subtitle =
config-scsi-image = Image
config-scsi-overlay = Overlay file
# $path is a file path; $size is its size in megabytes with one decimal.
config-scsi-overlay-file = { $path } ({ $size } MB)
config-scsi-overlay-tip = Changes collect here until a clean exit merges them into the disk.
config-scsi-cdrom-empty = No disc. Insert one from the SCSI menu.
config-scsi-cow = Copy-on-write
config-scsi-cow-tip = Keep this session's changes in a separate file so you can roll them back.
config-scsi-cow-check-tip =
config-scsi-scratch = Scratch disk
config-scsi-scratch-size = Scratch size (MB)
config-scsi-changer = Extra discs in the changer
config-scsi-disc-remove-tip = Remove this disc
config-scsi-add-disc = Add disc

## Network

config-net-mode = Mode
config-net-mode-nat = NAT (private network)
config-net-mode-pcap = Bridged (PCAP)
config-net-pcap-note = The guest joins your real network directly, so the NAT settings and port forwards don't apply. Packet capture needs administrator rights.
config-net-pcap-enable = Allow packet capture…
config-net-pcap-enable-tip =
config-net-nat-subtitle =
config-net-network = Network
config-net-custom = Custom
config-net-mask = Subnet mask
# $prefix is the prefix length, such as 24.
config-net-mask-custom = Custom (/{ $prefix })
# $reason explains what is wrong with the subnet.
config-net-invalid = Invalid: { $reason }
# $hosts is a count already formatted with separators, such as "254".
config-net-derived = Gateway { $gateway }, guest { $guest }, { $hosts } usable addresses, broadcast { $broadcast }
# $typed is the address the user entered; $network/$prefix is the one used instead.
config-net-snapped = { $typed } isn't a network address. Using { $network }/{ $prefix }.
config-net-warning = Warning: { $reason }
config-net-ok = OK: a private range that doesn't clash with your networks

config-forwards = Port forwards
config-forward-tcp = TCP
config-forward-udp = UDP
config-forward-host = Host port
config-forward-guest = to guest port
config-forward-bind-local = This computer only
config-forward-bind-any = All interfaces
config-forward-add = Add forward
config-forward-telnet = Telnet (host 2323 to guest 23)
config-forward-telnet-tip = Log in with telnet localhost 2323. The guest needs telnetd running.
config-forward-ftp = FTP (host 2121 to guest 21)
config-forward-ftp-tip = Forwards only the FTP control port. File transfers also need the data channel.
config-forward-ssh = SSH (host 2222 to guest 22)
config-forward-ssh-tip = Log in with ssh -p 2222 user@localhost. IRIX has no SSH server until you install one.
config-forward-vnc = VNC (host 5901 to guest 5900)
config-forward-vnc-tip = The host side is 5901 because macOS Screen Sharing uses 5900.
config-forward-xdmcp = XDMCP (host 11177 to guest 177, UDP)
config-forward-xdmcp-tip = Remote X login. Open to your whole network, and X servers must be pointed at port 11177.
config-forward-custom = Custom
config-forwards-subtitle =

config-tftp = TFTP server
config-tftp-subtitle =
config-tftp-enable = Serve a folder over TFTP
config-tftp-dir = Folder
config-tftp-dir-empty = Choose a folder to turn TFTP on.
config-tftp-dir-missing = That folder doesn't exist.
config-tftp-boot = To boot a kernel from it, press Esc at “Starting up the system” and enter:
# $gateway is the server's IP address.
config-tftp-irix = From IRIX, the server is { $gateway }:

config-rsh = rsh server
config-rsh-subtitle =
config-rsh-enable = Serve a folder over rsh (install source)
config-rsh-dir = Folder
config-rsh-dir-empty = Choose a folder to turn the rsh server on.
config-rsh-dir-missing = That folder doesn't exist.
config-rsh-inst = To install from it, give inst this location (the folder is the root):
config-rsh-readonly = Read-only: the guest can list and read files, nothing else.

config-nfs = NFS share
config-nfs-subtitle =
config-nfs-enable = Share a folder over NFS
config-nfs-dir = Shared folder
config-nfs-version = NFS version
config-nfs-v2 = v2 (IRIX 5.3)
config-nfs-v3 = v3 (IRIX 6.x)
config-nfs-appstore-grant = The shared folder must be one the app can access. Grant a disk folder first (File > Grant a disk folder…), or choose a folder above.
config-nfs-appstore-shared = Or use a “shared” folder inside a granted disk folder:
# Stands in for the server address in the mount command until one is set.
config-nfs-ip-placeholder = <NFS IP>
config-nfs-mount = On the guest, mount it with:
config-nfs-mount-subtitle =
config-nfs-pcap-ip = NFS server address
config-nfs-pcap-ip-hint = e.g. 192.168.1.213
config-nfs-pcap-suggest-tip = A free address on your network.
config-nfs-pcap-subtitle =

## Ethernet address

config-eaddr = Ethernet address
config-eaddr-nvram-tip = Stored in the NVRAM file ({ $path }).
config-eaddr-eeprom-tip = Stored in the EEPROM file ({ $path }).
config-eaddr-generate = Generate
config-eaddr-generate-tip = Make a new address in SGI's 08:00:69 range.
config-eaddr-no-file = Couldn't write { $path }. The file isn't usable yet.
config-eaddr-write-failed = Couldn't write { $path }: { $error }
config-eaddr-locked = Stop the machine to change the Ethernet address.
config-eaddr-unset = Not set. One is generated when the machine starts.

## Packet capture interface

config-pcap-iface = Interface
config-pcap-auto = Automatic (first active interface)
config-pcap-none = No interfaces found
config-pcap-list-error = Can't list interfaces: { $error }
config-pcap-not-listed = Click Refresh to list interfaces
config-pcap-refresh-tip =
config-pcap-manual = Or enter a name or number
config-pcap-manual-hint = e.g. 1 or eth0; blank for automatic
# Status words shown after an interface name in the list.
config-pcap-tag-up = up
config-pcap-tag-running = running
config-pcap-tag-loopback = loopback
config-pcap-not-built = This build has no packet capture. Rebuild iris-gui with --features pcap.

## Video in

config-vino = Video in (IndyCam)
config-vino-source = Source
config-vino-source-camera = Camera
config-vino-source-test = Test pattern
config-vino-source-black = Black
config-vino-source-off = Off
config-vino-standard = Standard
config-vino-ntsc = NTSC
config-vino-pal = PAL
config-vino-camera = Camera number
config-vino-test = Test camera…
config-vino-test-subtitle =
config-vino-permission-subtitle =

## Build features

config-build = Build features
config-build-cpu = CPU
config-build-not-built = Not built
config-build-jit-off = Built, turned off
config-build-jit-on = On
config-build-rex-jit = Newport graphics JIT
config-build-gr2-jit = GR2 graphics JIT
config-build-impact-jit = IMPACT graphics JIT
config-build-cpu-jit = CPU JIT
config-build-perfetto = Perfetto tracing
config-build-perfetto-no = No (build with --features perfetto)
config-build-lightning = Lightning
config-build-lightning-yes = Yes (no GDB stub)
config-build-idle-pause = Idle pause
config-build-pcap = Packet capture
config-build-hostgl = Host OpenGL
config-build-hostgl-on = Yes: IRIX OpenGL programs draw on the host GPU
config-build-hostgl-hostcall-only = Host services only (build with --features hostgl)
config-build-hostgl-unsupported-os = Host services only (no host OpenGL on this system)
config-build-hostgl-off = No (build with --features hostgl)
config-build-hostcall-subtitle =
config-build-mips-subtitle =

## Debug

config-debug-lightning = This is a Lightning build, so breakpoints, the GDB stub and tracebacks are off.
config-debug-gdb-port = GDB stub port
config-debug-capture = Capture
config-debug-gl-capture = OpenGL capture
config-debug-gl-capture-tip = Faster for GL demos in the GUI.
config-debug-audio-subtitle =
config-debug-no-idle = Keep the CPU running when idle
config-debug-log = Device log
config-debug-log-hint = all, or e.g. mc,mips
config-debug-profiles = Build profiles
config-debug-rebuild-cli = Rebuild Premiere CLI…
config-debug-rebuild-cli-tip =
config-debug-rebuild-gui = Rebuild Premiere GUI…
config-debug-rebuild-gui-tip =

config-perf = Host performance
config-perf-affinity = Pin threads to cores
config-perf-cpu-core = CPU thread core
config-perf-rex3-core = Graphics thread core
config-perf-refresh-core = Display refresh core

## Perfetto

config-perfetto = Perfetto trace
config-perfetto-file = Trace file
config-perfetto-categories = Categories
config-perfetto-categories-hint = all, or e.g. irq,gfx or all,-jit
# $names is a comma-separated list of category names.
config-perfetto-categories-tip = all, or a comma-separated list of: { $names }
config-perfetto-clock = Clock
config-perfetto-clock-host = Host time
config-perfetto-clock-guest = Guest instructions
config-perfetto-clock-tip = Guest time counts each instruction as 1 ns, so slices follow guest work, not host speed.
config-perfetto-start = Start trace
config-perfetto-stop = Stop trace
config-perfetto-subtitle =

## Automation

config-ci-subtitle =
config-ci-enable = CI mode
config-ci-socket = CI socket
config-ci-socket-tip = A Unix socket path or host:port.
config-ci-display = Keep window visible
config-ci-scsi-deferred = Defer SCSI interrupts
config-ci-scsi-deferred-tip = Needed for OpenBSD and NetBSD. Turn it off if SCSI commands time out.
config-ci-aspect = Lock aspect ratio
config-ci-scroll = Scroll pixels per line
config-ci-ping = Test connection
config-ci-test = Test headless boot
config-ci-test-subtitle =
