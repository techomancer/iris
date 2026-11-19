# Octane: the internal SCSI bus is probed only at ids 1-3

Status: confirmed 2026-11-19 (PROM 4.9, IRIX 6.5.30).

**Symptom.** A `[scsi.4]` CD-ROM (controller 0, the default) is listed by
`scsi status` but missing from PROM `hinv` and IRIX `hinv`; with nothing
else attached the PROM's "Internal SCSI device/cable diagnostic" fails.
`scsiha -p 0` in IRIX finds it afterwards (`/hw/scsi_ctlr/0/target/4`
appears), which makes it look like a boot-time model bug.

**Cause.** Not the emulator. The Octane's internal bus has three drive bays,
and the PROM sends INQUIRY only to ids 1, 2 and 3 on controller 0 (retrying
them), while it walks ids 1-15 on controller 1, the external bus. The
IRIX boot probe does the same. A device at id 4+ on controller 0 is never
asked. `log scsi` (with the `isp1040 N: request ... target T` lines) shows
the per-controller pattern directly.

**Rule.** On the Octane, put CD-ROMs and anything beyond three disks on
controller 1 (`controller = 1`, or `[scsi.20]` for id 4 there). The guest
then sees `scsi(1)cdrom(4)` / `dks1d4`.
