## The running machine: power, saved states, discs, the status bar.

session-needs-stopped = Power off the machine first.
session-needs-running = The machine isn't running.
session-syncing = Disks are being synchronized.
session-no-nvram-reset = This machine keeps its PROM settings elsewhere.
session-state-name-invalid = A name can't contain / \ or : or start with a dot.
session-state-name-taken = A saved state with that name already exists.
session-screenshot-title = Save screenshot
session-insert-disc-title = Insert disc
session-disc-images = Disc images
session-save-state-title = Save state
session-save-state-prompt = Name:
session-rename-state-title = Rename saved state
session-delete-state-title = Delete saved state { $name }?
session-delete-state-body = The saved state is removed from the machine's folder. This can't be undone.
session-state-rename-failed = Couldn't rename the saved state: { $error }
session-state-delete-failed = Couldn't delete the saved state: { $error }
session-reset-nvram-title = Reset the PROM settings?
session-reset-nvram-body = The NVRAM goes back to its factory state, with this machine's own Ethernet address. Settings made at the PROM monitor (setenv) are lost.
session-reset-nvram-ok = Reset
status-starting = Starting
status-powered-off = Powered off
status-disk-tip = Disk { $id }: { $file }
status-cdrom-tip = CD-ROM { $id }: { $disc }
status-cdrom-empty-tip = CD-ROM { $id }: no disc
status-chd-sync-pending = Changes are merged into the disk image when the machine stops.
status-net-guest-ip = Guest address: { $ip }
status-audio-tip = Audio
status-audio-playing-tip = Audio: playing
status-serial-tip = Serial console
status-serial-connected-tip = Serial console: connected

# Setting the machine's Ethernet address (Tools > Set MAC Address).
session-set-mac-title = Set MAC address
# $chip is the NVRAM or EEPROM file the address is kept in.
session-set-mac-body = The address the PROM reports, kept in { $chip }.
session-set-mac-ok = Set
session-set-mac-invalid = That is not an Ethernet address; use 08:00:69:xx:xx:xx.
# $mac is the address written.
session-set-mac-done = Ethernet address set to { $mac }.
