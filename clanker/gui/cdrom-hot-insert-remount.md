# GUI CD-ROM insert must hot-load + remount /CDROM

**Keywords:** gui, cdrom, iso, /CDROM, mediad, scsi menu, hot-swap
**Category:** gui

## Symptom

User picks an ISO in iris-gui (SCSI menu or config path) but `/CDROM` stays
empty and no desktop icon appears.

## Cause

1. **SCSI menu "Insert disc"** only updated `MachineConfig` — it did not call
   `Wd33c93a::load_disc` on the running VM. The emulator still had an empty tray.
2. **IRIX `mediad`** does not reliably remount `/CDROM` after hot insert or
   changer swap (`clanker/irix/cdrom-changer-eject-no-mediad-remount.md`).

Config-tab path picker and Ctrl+F12 already hot-loaded; SCSI menu did not.

## Fix (iris-gui)

- `Insert disc` / `Swap disc` → `Cmd::LoadDisc { remount: true }` when running
- `Mount /CDROM in IRIX` → `Cmd::RemountCdrom` (injects csh mount one-liner on tty1)
- `Attach CD-ROM with disc…` — attach config + hot-load if running
- `Machine::remount_cdrom_guest(id)` — EFS `dks0dNs7` then iso9660 `dks0dNvol`

## User workflow

1. Attach CD-ROM at **SCSI ID 4** (internal) — with disc, or empty + Insert disc
2. If VM already running: Insert disc hot-loads; keep a **shell/xterm** on the
   console so remount commands are received
3. Manual fallback in IRIX:

```csh
mount -t efs -o ro /dev/dsk/dks0d4s7 /CDROM
# or
mount -t iso9660 /dev/rdsk/dks0d4vol /CDROM
```

4. New SCSI **drive** attach/detach still requires **Stop → Start**
