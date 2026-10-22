# IRIX csh + scratch raw-device gotchas (when you can't use iris-ci)

**Keywords:** irix,csh,bs512,scratch,dd,redirect,marker,wait,serial
**Category:** irix

# IRIX csh + scratch raw-device gotchas

If you're driving the CI socket without `iris-ci` (raw `nc`, foreign language harness, etc.), these are the pitfalls the wrapper handles for you. Use the wrapper if you can.

## csh redirect syntax

IRIX root logs into csh. `2>&1` is sh-only and csh fails to parse it silently. Use:

- `>& /dev/null` — combined stdout+stderr to /dev/null
- `>& file` — combined stdout+stderr to file
- `>> file` — append stdout (csh has no portable stderr-only redirect)

If you need sh semantics, wrap in `sh -c "..."`.

## csh echoes typed input

Any string in the typed command appears in the serial buffer twice — once as the literal input echo, once expanded in the output. A wait pattern of `IRIS-CI-RC=` matches the typed line (which contains the literal `IRIS-CI-RC=$status`) before the command runs.

Use `\nIRIS-CI-RC=` as the wait pattern. The typed line has the marker inline; only the output line starts a fresh line with the marker, so the newline-prefixed pattern only matches the actual output.

## Raw block-device alignment

`/dev/rdsk/dks0dNs0` (the scratch payload partition) requires:

- **Reads** in 512-byte multiples. `dd bs=64` returns `Read error: I/O error`.
- **Writes** padded to `bs`. From a 28-byte input, `dd bs=512 conv=sync,notrunc` zero-pads to 512.

After a `dd … of=FILE bs=512 count=N` from the scratch device, the guest file is N×512 bytes — too long. Truncate to the real size with `dd if=/dev/null of=FILE bs=1 seek=<original_byte_size> count=0`.

## Looking up byte counts in the guest

`ls -l` column layout varies by IRIX version. Use `wc -c < FILE`, which prints just the byte count on one line and is cleanly parseable.

## See also

- `clanker/snapshot/iris-ci-is-the-canonical-ci-socket-interface.md` — the wrapper that hides all of this
- `clanker/snapshot/scratch-scsi-volume-sgi-vh-layout-and-irix-raw-device-gotchas.md` — partition layout and SGI VH details
- `clanker/snapshot/ci-mode-overlay-path-is-tmpiris-ci-pid-scsiidoverlay.md` — where `--ci`'s COW overlay actually lives

