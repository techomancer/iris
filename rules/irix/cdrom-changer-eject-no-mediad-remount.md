# CD changer `cdrom-eject` does not trigger a `mediad` remount

**Finding (5.3, and expected on 6.5):** `iris-ci cdrom-eject <id>` cycles
the SCSI CD changer to the next disc at the *device* level, but the guest's
`mediad` does **not** notice the media change and keeps the previous disc's
filesystem mounted at `/CDROM` (stale — you'll see the old disc's contents,
or I/O errors). On real hardware the drive raises a media-change /
unit-attention that `mediad` polls; iris's changer eject doesn't deliver
that signal to the guest.

**Symptom:** after `cdrom-eject 4`, `ls /CDROM` still shows the *old* disc.

**Workaround:** remount by hand after every eject (the CD device is
`/dev/dsk/dks0d4s7` for SCSI id 4, EFS, read-only):

```bash
ic cdrom-eject 4
ic run "umount /CDROM"
ic run "mount -t efs -o ro /dev/dsk/dks0d4s7 /CDROM"
```

This came up cycling the 3 Developer's Toolbox CDs during a 5.3 add-on
install (`rules/irix/irix-install.md` §11). The base-OS install via the changer
(6.5.22 recipe) doesn't hit it because `inst` itself reopens the
distribution path after each swap rather than relying on a `/CDROM` mount.

Related: shell driving is csh — see the csh-redirect memory note; use
`egrep` (5.3 `grep` has no `-E`), and a wedged `? ` continuation needs
Ctrl-D, not Ctrl-C.
