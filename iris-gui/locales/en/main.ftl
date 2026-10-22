# Main window: menus, control panel, status, framebuffer, help windows.

menu-view-language = Language
menu-view-language-system = System default

## Machine selection and developer tools

toast-pcap-enabled = Packet capture is enabled.
# $path is the machine folder; $error is the operating system's message.
toast-enter-dir-failed = Can't open the machine folder { $path }: { $error }
toast-switch-machine-busy = Stop the emulator and wait for disk operations to finish before switching machines.
toast-autosave-failed = Couldn't save the machine settings: { $error }
toast-settings-save-failed = Couldn't save settings: { $error }
toast-machine-loaded = Loaded { $name }.
toast-machine-load-failed = Couldn't load the machine: { $error }
# $path is the machine's config file.
toast-premiere-config = Config for Premiere: { $path }. Run iris with --config from the machine's folder.
# $profile is "gui" or "cli".
toast-rebuild-started = Rebuilding { $profile } in the background. Progress is shown in the terminal.
toast-rebuild-failed = Couldn't start the rebuild: { $error }
toast-ci-test-stop-first = Stop the emulator first. The CI test runs a separate copy of iris.
toast-ci-test-starting = CI test: starting iris without a window…

## Starting and stopping

toast-start-failed = Couldn't start the machine: { $error }
toast-invalid-config = The machine settings aren't valid: { $error }
# $path is the configured PROM file.
toast-prom-missing = { $path } wasn't found, so the built-in PROM is used.
toast-nvram-seeded = Created a default NVRAM.
# $mac is an Ethernet address such as 08:00:69:12:34:56.
toast-nvram-mac-written = The NVRAM had no Ethernet address, so { $mac } was written to it.
toast-nvram-mac-no-file = The NVRAM has no Ethernet address, and there's no NVRAM file to write one to yet.
toast-nvram-mac-failed = Couldn't write the Ethernet address to the NVRAM: { $error }
toast-nveeprom-mac-written = The NVRAM EEPROM had no Ethernet address, so { $mac } was written to it.
# Title of the folder picker.
main-grant-folder-title = Grant access to a disk folder
toast-folder-granted = Access granted to { $path }.
toast-emulator-started = Emulator started.
toast-emulator-stopped = Emulator stopped.
toast-guest-powered-off = IRIX powered off. It's safe to stop.
# $name is the saved state's name.
toast-state-saved = Saved state { $name }.
toast-state-restored = Restored state { $name }.
toast-screenshot-saved = Screenshot saved to { $path }.
toast-error = Error: { $error }
toast-cow-committed = Write overlay committed to the disk image.
toast-cow-done = Done.
toast-disks-synced = { $count ->
    [one] Synchronized one disk.
   *[other] Synchronized { $count } disks.
}

## Copy-on-write menu section

menu-cow-heading = Write overlays
menu-cow-stop-first = Stop the machine to commit or discard overlay changes.
# $id is the SCSI ID, $name the disk image's file name.
menu-cow-disk = SCSI { $id }: { $name }
menu-cow-commit = Commit overlay to disk
menu-cow-commit-tip = Merges the write overlay into the disk image permanently.
menu-cow-discard = Discard overlay…
menu-cow-discard-tip = Deletes the write overlay and returns the disk to its original state.

## File menu

menu-file = File
menu-file-new-machine = New machine…
menu-file-switch-machine = Switch machine
menu-file-no-machines = No saved machines
menu-file-rename-machine = Rename machine…
menu-file-rename-machine-disabled-tip = No machine is selected.
menu-file-delete-machine = Delete machine
menu-file-premiere = Prepare for Premiere…
menu-file-folders-heading = Disk folder access
menu-file-grant-folder = Grant a disk folder…
menu-file-grant-folder-tip = Choose the folder that holds your disk images so IRIS can save disk changes there.
menu-file-no-folders = No folders granted
menu-file-folder-active-tip = IRIS can read and write this folder.
menu-file-folder-lapsed-tip = Access has lapsed. Grant this folder again.
menu-file-folder-reveal-tip = Show in Finder
menu-file-folder-revoke-tip = Remove access
menu-file-quit = Quit

## Machine menu

menu-machine = Machine
menu-machine-start = Start
menu-machine-stop = Stop
menu-machine-reset = Restart
menu-machine-reset-nvram = Reset NVRAM
# $nvram and $nveeprom are file paths, shown on their own lines.
menu-machine-reset-nvram-tip =
    Restores the NVRAM and NVRAM EEPROM to their defaults and assigns a new Ethernet address.
    { $nvram }
    { $nveeprom }
toast-nvram-reset-failed = Couldn't reset the NVRAM: { $error }
toast-nveeprom-reset-failed = Couldn't reset the NVRAM EEPROM: { $error }
toast-nvram-reset = NVRAM reset. New Ethernet address: { $mac }
# $cpu is a processor name such as R4400.
menu-machine-processor = Processor: { $cpu }
toast-cpu-set = Processor set to { $cpu }. It takes effect the next time the machine starts.
menu-machine-running-pending = Running on { $cpu }. Stop the machine to apply the change.
menu-machine-running = Running on { $cpu }
menu-machine-cpu-subtitle =
menu-machine-save-state = Save state
menu-machine-restore-state = Restore state
menu-machine-restore = Restore
menu-machine-screenshot = Screenshot…
# Title of the save dialog.
menu-machine-screenshot-title = Save screenshot
menu-machine-serial-console = Serial console…
menu-machine-serial-console-tip = Shows the IRIX serial console.

## Memory menu

menu-memory = Memory
# $ram is a summary such as "64 MB (4 × 16 MB)".
menu-memory-configured = Configured: { $ram }
menu-memory-running-pending = Running with { $ram }. Stop the machine to apply the change.
menu-memory-running = Running with { $ram }
menu-memory-running-subtitle =
menu-memory-stopped-subtitle =
menu-memory-presets = Total memory
menu-memory-size = { $mb } MB
menu-memory-stop-first-tip = Stop the machine to change memory.
# $banks lists each bank's size in MB, such as [32, 32, 0, 0].
toast-ram-set = Memory set to { $ram } { $banks }.
menu-memory-banks = Memory banks
menu-memory-bank = Bank { $bank }: { $mb } MB
menu-memory-ip28-only-tip = 256 MB and 512 MB banks need the Indigo2 R10000 (IP28) profile.

## SCSI menu

menu-scsi = SCSI
toast-disc-saved = Disc saved. Restart the machine to load it.
toast-cdrom-remount-sent = SCSI { $id }: remount sent. Keep a shell window focused on the IRIX console.
toast-cdrom-remount-not-running = Start the machine before mounting /CDROM.

## View menu

menu-view = View
menu-view-fullscreen = Full screen (F11)
menu-view-exit-fullscreen = Exit full screen (F11)
menu-view-ui-scale = Interface size
menu-view-ui-scale-subtitle =
menu-view-vm-scale = Screen size
menu-view-vm-scale-subtitle =

## Help menu

menu-help = Help
menu-help-title = IRIS, an emulator for SGI workstations
# $version is the release version, $commit the source revision.
menu-help-version = Version { $version } ({ $commit })
menu-help-diagnostics = Diagnostics
menu-help-test-camera = Test camera…
menu-help-test-camera-tip = Preview the host camera that IRIX sees as the IndyCam.
menu-help-start-first-tip = Start a machine first.
menu-help-serial-console = Serial console…
# $port is a TCP port on 127.0.0.1.
menu-help-serial-console-tip = Connects to the serial port on 127.0.0.1:{ $port }.
menu-help-camera-network = Camera and networking…
menu-help-nfs = Mount the shared folder in IRIX…
menu-help-nfs-tip =
menu-help-ultra64 = N64 development board…
menu-help-ultra64-tip =
menu-help-legal = Legal
menu-help-licenses = Licenses…
menu-help-licenses-tip =
menu-help-privacy = Privacy policy…
menu-help-authors = Authors
menu-help-authors-iris = IRIS: Dominik Behr (techomancer) and contributors
menu-help-authors-gui = iris-gui: Dani Sarfati (danifunker)
menu-help-repository = Source code:
menu-help-build-features = Build features
# $name is a build feature name (not translated); $state is one of the menu-help-feature-* values.
menu-help-feature = { $name }: { $state }
menu-help-feature-on = on
menu-help-feature-off = off
menu-help-feature-off-sandbox = off (sandboxed)
menu-help-feature-on-no-debug = on (no debugger)

## Control column

main-start = ▶ Start
main-stop = ■ Stop
main-save-state = Save state
main-restore-state = Restore state
main-show-settings = Edit settings…
main-hide-settings = Hide settings
main-tab-network = Network
main-tab-video-in = Video input
main-tab-debug = Debug
main-tab-ci = Automation
main-capture-on = Mouse and keyboard captured
# $keys is the key combination that releases capture, such as "Left Ctrl+Alt".
main-capture-release = Press { $keys } to release.
main-capture-f11-subtitle =
main-capture-off = Mouse and keyboard not captured
main-capture = Capture mouse and keyboard

## Status footer

status-halted = Halted, safe to stop
status-prom = PROM
status-running = IRIX running
status-stopped = Stopped
# $mips is a whole number.
status-mips = { $mips } MIPS
status-mips-tip = Emulation speed in millions of guest instructions per second.
status-kernel-hz = { $hz } Hz
status-kernel-hz-tip = How often the IRIX clock ticks.
status-net = Network
status-net-nat = NAT
status-net-pcap = Bridged
# $iface is a host network interface name, or "Auto".
status-net-pcap-active-tip = Bridged to { $iface }. The guest is sending traffic.
status-net-pcap-idle-tip = Bridged to { $iface }. No traffic from the guest yet. Check its IP address or DHCP; only wired connections work.
status-net-pcap-off-tip = Bridged networking. The machine isn't running.
status-net-nat-active-tip = NAT networking. The guest is sending traffic.
status-net-nat-idle-tip = NAT networking. No traffic from the guest yet. It may have no IP address, or one outside the NAT subnet.
status-net-nat-off-tip = NAT networking. The machine isn't running.
status-net-off-tip = The machine isn't running.
status-net-check = Check
status-net-check-tip = Diagnose guest networking.
# $scale is a number such as 1 or 1.25.
status-scale = { $scale }× scale
status-scale-filtered = { $scale }× scale, smoothed
status-scale-tip = Size of the emulated display. At sizes that aren't whole multiples, the picture is smoothed.

## Check networking window

window-net-title = Check networking
window-net-not-running = Start the machine, then check again.
window-net-up = Networking is up
# $ip is the guest's IP address.
window-net-up-detail = { $ip } is reaching the network through IRIS.
window-net-up-detail-no-ip = The guest is reaching the network through IRIS.
window-net-gateway = Gateway: { $gateway }
window-net-no-traffic = No traffic from the guest yet
window-net-no-traffic-detail = If IRIX just booted, wait a few seconds and check again. Otherwise ec0 may not be configured, or networking is turned off in IRIX.
window-net-enable = To turn networking on, run as root:
window-net-enable-detail = Then reboot, or run /etc/init.d/network start.
window-net-fresh-hint = If ec0 isn't configured yet, use the IRIS defaults: address { $ip }, gateway { $gateway }, netmask { $netmask }.
window-net-down = Networking is down
# $network is a subnet such as 192.168.0.0/24.
window-net-guest-ip = The guest's address is { $ip } (network { $network }).
window-net-has-gateway = Its gateway is { $gateway }. IRIS adopts the guest's gateway, so networking should come up in a few seconds. Check again shortly.
window-net-has-gateway-detail = If it stays down, the gateway may be unreachable. Log in as root to check.
window-net-no-gateway = The guest has no default route, so it can only reach its own subnet. Add one; IRIS adopts whatever gateway you choose.
window-net-run-as-root = Log in as root on the serial console or in a winterm, and run:
window-net-route-temporary = This lasts until the next reboot. To keep it, set the default route in the IRIX network settings.
# $iface and $addr are the host interface name and address.
window-net-overlap = The guest's subnet { $network } overlaps your host network ({ $iface } { $addr }). IRIS won't put its NAT on your real network, so move the guest onto the IRIS subnet instead.
window-net-overlap-result = This gives ec0 the address { $ip } with gateway { $gateway }. Check again afterwards.
window-net-nat-subnet = The IRIS NAT subnet is { $network }. You can switch it to the guest's subnet.
window-net-nat-subnet-other = The IRIS NAT subnet is different. You can switch it to the guest's subnet.
window-net-use-guest-subnet = Use { $network } for NAT
toast-nat-subnet-live = NAT subnet set to { $network }.
toast-nat-subnet-next-start = NAT subnet set to { $network }. It takes effect the next time the machine starts.

## Framebuffer

fb-waiting = Waiting for the display…
fb-head-waiting = Waiting…
# $head is the screen number, starting at 0.
fb-head = Head { $head }
fb-powered-off = Powered off
toast-guest-halted = IRIX halted. Mouse and keyboard released; click the display to capture them again.
toast-guest-off-syncing = IRIX powered off. Synchronizing disks…

## Machines

toast-rename-failed = Couldn't rename the machine: { $error }
toast-renamed = Renamed to { $name }.
toast-delete-failed = Couldn't delete the machine: { $error }
toast-deleted = Deleted { $name }. Its disk images and NVRAM files were kept.
main-settings-no-machine = Default
# $name is the machine's name.
main-settings-heading = Settings: { $name }
# $id is the SCSI ID, $name the disc image's file name.
toast-disc-loaded = SCSI { $id }: loaded { $name }.

## Camera test window

window-camera-title = Test camera
window-camera-error = Camera unavailable: { $error }
window-camera-error-detail = Check that a camera is connected and that IRIS is allowed to use it. On macOS, that's in System Settings > Privacy & Security > Camera.
window-camera-starting = Starting the camera…
window-camera-starting-subtitle =

## Serial console window

window-serial-title = IRIX serial console
# $addr is a network address such as 127.0.0.1:8881.
window-serial-connected = • Connected to { $addr }
window-serial-disconnected = Disconnected
window-serial-input-hint = Type a command and press Enter
window-serial-send = Send
window-serial-clear = Clear

## License and privacy windows

window-license-title = Licenses
window-license-intro = IRIS is licensed under the BSD 3-Clause License.
window-license-chd = CHD disk support comes from libchdman-rs, which is also BSD 3-Clause, as is the MAME CHD code it includes. Its notice is below.
window-license-source = Source:
window-license-iris-heading = IRIS: BSD 3-Clause License
window-license-chd-heading = CHD support (libchdman-rs): BSD 3-Clause License
window-privacy-title = Privacy policy

## Help: camera and networking

help-info-title = Camera and networking
help-how-to-use = How to use it
help-privacy = Privacy
help-camera-heading = 📷 Camera
help-camera-body = IRIS emulates the IndyCam video input. When you choose a host camera as the video source, its live picture becomes the Indy's camera input.
help-camera-usage =
    • Help > Test camera shows a live preview while a machine is running.
    • To use it in IRIX, set the video input source to Camera on the Video input tab, boot IRIX, and run an IndyCam program such as vino or cam.
    • macOS asks for camera permission the first time. Closing the preview releases the camera.
help-camera-privacy = Camera frames only feed the emulated video input. IRIS never records, stores or sends them anywhere.
help-network-heading = 🌐 Networking
help-network-body = The emulated machine reaches the internet through a built-in NAT, like a home router. IRIS doesn't change your system's network settings or need administrator rights. To reach services running in IRIX, add port forwards on the Network tab.
help-network-subnet-heading = Guest address
help-network-subnet =
    • The NAT is one subnet, 192.168.0.0/24 by default, with the gateway at .1. You can change it on the Network tab.
    • The guest needs an address on that subnet and a default route to the gateway. Otherwise nothing routes.
    • The Check button next to the NAT light compares the guest's address with the NAT and tells you how to fix a mismatch.
    • There are no port forwards by default. Add forward maps a host port to a guest port, for example to telnet into IRIX.
help-network-serial-heading = The serial console isn't the network
# $serial and $monitor are TCP ports on 127.0.0.1.
help-network-serial = Help > Serial console is the guest's serial terminal, used to log in and to reach the PROM. It works with or without guest networking. It runs over a local connection on port { $serial }; port { $monitor } is the IRIS debug monitor.
help-network-privacy = IRIS listens only on your own computer (127.0.0.1) for the serial console, the monitor and any port forwards you add. It never opens network connections on its own; all outside traffic comes from the guest.

## Help: shared folder (NFS)

help-nfs-title = Mount the shared folder in IRIX
# Shown in place of the server address in the mount command.
help-nfs-no-server-ip = <NFS server address from the Network tab>
help-nfs-disabled = The shared folder is turned off.
help-nfs-disabled-detail = Turn on the NFS share on the Network tab and choose a folder. Then open this window again for the mount command.
help-nfs-intro-nat = IRIS shares the folder below over NFS from the NAT gateway. Boot IRIX, check that networking is up, then log in as root and run:
help-nfs-intro-bridged = IRIS shares the folder below over NFS from the address you gave it on your network. Boot IRIX, check that networking is up, then log in as root and run:
help-nfs-no-folder = No folder is chosen. Choose one for the NFS share on the Network tab first.
help-nfs-sharing = Sharing: { $path }
help-nfs-mounted = The folder then appears at /shared in IRIX. You can use any empty directory instead of /shared.
help-nfs-version-auto = NFS version: automatic. IRIX 6.5 uses NFSv3 and IRIX 5.3 uses NFSv2.
help-nfs-version-2 = NFS version: 2 only, the right choice for IRIX 5.3.
help-nfs-version-3 = NFS version: 3 only, for IRIX 6.x.
help-nfs-size-v3 = With NFSv3, file size is limited only by your disk.
help-nfs-size-v2 = ⚠ NFSv2 can't handle files larger than about 2 GB. Use NFSv3 for larger files.
help-nfs-trouble-heading = If the mount hangs or is refused
help-nfs-trouble = The IRIS NFS server only speaks UDP, and IRIX may try TCP or another version. Set both explicitly:
help-nfs-trouble-network = NFS can't mount until networking works. Use the Check button next to the network light to make sure the guest is on the IRIS subnet.
help-nfs-unmount = To unmount it later:
help-nfs-limits-heading = Limitations
help-nfs-limit-size = File size:
help-nfs-limit-size-body = About 2 GB per file over NFSv2. NFSv3 is limited only by the host disk.
help-nfs-limit-udp = UDP only:
help-nfs-limit-udp-body = NFS over TCP isn't supported.
help-nfs-limit-export = One export:
help-nfs-limit-export-body = The whole folder is shared as "/". The path after the colon in the mount command is ignored.
help-nfs-limit-permissions = Permissions:
help-nfs-limit-permissions-body = Everything is owned by root, with directories 0755 and files 0644. chmod and chown are accepted but have no effect.
help-nfs-limit-auth = No authentication:
help-nfs-limit-auth-body = Any client can connect. Don't share anything sensitive.
help-nfs-limit-locking = No file locking:
help-nfs-limit-locking-body = Locks don't work between the host and the guest.
help-nfs-limit-special = No special files:
help-nfs-limit-special-body = Symbolic links, device nodes and hard links aren't supported.
help-nfs-limit-sync = Synchronous writes:
help-nfs-limit-sync-body = Every write goes straight to disk. Data is safe, but writing is slower.

## Help: N64 development board

help-ultra64-title = N64 development board
help-ultra64-intro = IRIS emulates the Indy side of the N64 development board: the GIO card Nintendo used to build N64 games, with its 16 MB shared "RAMROM". The N64 itself runs in a separate emulator, gopher64, which uses the RAMROM as its cartridge.
help-ultra64-step1 = 1. Turn on the board in IRIS
help-ultra64-enabled = ✔ It's on (General tab). It takes effect the next time the machine starts.
help-ultra64-disabled = Turn on "N64 development board" on the General tab, then start or restart the machine.
help-ultra64-step2 = 2. Build and run the N64 emulator
help-ultra64-step2-body = You need the IRIS version of gopher64 (branch "ultra64"). Build it once and run it next to IRIS. It waits for IRIS, so the order doesn't matter:
help-ultra64-fork = Repository:
help-ultra64-step3 = 3. Load a ROM from IRIX
help-ultra64-step3-body = Boot IRIX, open a shell and load a .n64 or .z64 ROM with gload. A gopher64 window opens and starts the game:
help-ultra64-switch = Run gload again with another ROM to switch games. The N64 resets cleanly.
help-ultra64-monitor-heading = Monitor commands
help-ultra64-monitor-intro = Type these in the IRIS monitor console:
help-ultra64-cmd-status = Show the board's state
help-ultra64-cmd-reset = Reset the N64
help-ultra64-cmd-load = Load a file straight into RAMROM
help-ultra64-cmd-dump = Show RAMROM contents in hex
help-ultra64-cmd-disasm = Disassemble RAMROM
help-ultra64-footer = This is a developer feature for N64 homebrew and research. IRIS doesn't include any ROMs. Hardware notes are in docs/Indy/ultra64.md.

## Disk sync and discard dialogs

modal-sync-title = Synchronizing disks…
# Shown when several disks are synced.
modal-sync-disk = Disk { $disk } of { $total }
modal-sync-body = Writing this session's changes into the CHD disk image. Only some compressed CHD files need this step.
modal-sync-dont-quit = Don't force quit. This takes a moment.
modal-discard-title = Discard write overlay?
modal-discard-body = Every change stored in this disk's write overlay will be lost.
modal-discard-confirm = Discard overlay
toast-prom-builtin = Using the built-in PROM.

## Welcome panel

welcome-unsaved = Unsaved
# $profile is the machine type, such as "Indy (IP24)".
welcome-machine = Machine: { $name } ({ $profile })
welcome-autosave-pending-subtitle =
welcome-summary = Summary
welcome-platform = Platform
welcome-processor = Processor
welcome-processor-running = Processor (running)
welcome-prom = PROM
welcome-prom-builtin = Built-in
welcome-prom-missing = { $path } (not found, using the built-in PROM)
welcome-nvram = NVRAM
welcome-memory = Memory
welcome-memory-running = Memory (running)
welcome-drives = Drives
# $id is the SCSI ID.
welcome-drive-daynaport = SCSI { $id }: DaynaPort Ethernet adapter
welcome-drive-cdrom = SCSI { $id }: CD-ROM, { $path }
welcome-drive-disk = SCSI { $id }: disk, { $path }
welcome-drives-subtitle =
welcome-network = Network
welcome-network-bridged = Bridged to { $iface }
welcome-network-nat = NAT on { $subnet }
welcome-network-nat-default = NAT on 192.168.0.0/24 (default)
welcome-start = ▶ Start

## Status footer: machine

# $name is the machine name, $profile the machine type, $cpu the processor.
status-machine = { $name }: { $profile }, { $cpu }
status-machine-unsaved = { $name }: { $profile }, { $cpu } (unsaved)
status-cpu-pending = { $cpu } after restart
status-dirty-sectors = { $count ->
    [one] One changed disk sector
   *[other] { $count } changed disk sectors
}

## Keyboard shortcuts

# Title of the disc picker opened with Ctrl+F12.
main-load-disc-title = Load CD-ROM disc
toast-no-cdrom = No CD-ROM drive is attached.
toast-load-disc-not-running = Start the machine before loading a disc.

## Dialogs

# A bulleted line in a dialog list.
modal-list-item = • { $item }
modal-rename-title = Rename machine
modal-rename-label = New name:
modal-rename-confirm = Rename
toast-machine-created = Created { $name }.
toast-machine-create-failed = Couldn't create the machine: { $error }
# $path is the new disk image; $id the SCSI ID it's attached to.
toast-disk-created = Created { $path } and attached it as SCSI { $id }.
modal-stop-title = Stop the machine?
modal-stop-body = Stopping now could damage the disk image:
modal-stop-halt = Shut down IRIX
modal-stop-force = Force stop
toast-halt-sent = Shutting down IRIX. Stop the machine once it has halted.
modal-prom-title = Use the built-in PROM?
modal-prom-body = Replace the selected PROM with the built-in one?
modal-prom-confirm = Use built-in PROM
modal-net-title = Check network settings
modal-net-body = These network settings don't look right:
modal-net-subtitle =
# $subnet is a suggested subnet such as 192.168.0.0/24.
modal-net-use = Use { $subnet }
modal-net-override = Keep my setting
modal-pcap-title = Allow packet capture
modal-pcap-body = Bridged networking needs permission to capture packets on a network interface.
modal-pcap-enable = Allow packet capture…
modal-pcap-not-now = Not now
modal-missing-title = Missing disk images
modal-missing-body = These disk images are missing or can't be read:
modal-missing-cdrom = SCSI { $id } (CD-ROM): { $path }
modal-missing-disk = SCSI { $id } (disk): { $path }
modal-missing-sandbox = If a file exists but is listed here, macOS has withdrawn access to it. Choose it again on the Disks tab.
modal-missing-subtitle =
modal-missing-edit = Open Disks tab
modal-missing-detach = Remove missing and start
modal-grant-title = Allow access to disk folders
modal-grant-body = IRIS can't write to the folders holding these CHD disks:
modal-grant-disk = SCSI { $id }: { $path }
modal-grant-why = A compressed CHD keeps this session's changes in a separate file and merges them into the disk when you quit. That needs write access to the disk's folder. Without it, the change file keeps growing.
modal-grant-subtitle =
modal-grant-sandbox = Grant access to the folder so IRIS can merge changes on quit:
modal-grant-folder = Grant { $path }…
modal-grant-tcc = macOS is blocking access to these folders. Allow IRIS in System Settings > Privacy & Security, under Files and Folders or Full Disk Access, then check again. Full Disk Access may need IRIS to be reopened.
modal-grant-open-privacy = Open Privacy & Security…
modal-grant-recheck = Check again
modal-grant-start-anyway = Start without merging
