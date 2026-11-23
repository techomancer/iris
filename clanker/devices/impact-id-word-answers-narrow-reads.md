# OpenBSD 6.9 on a High IMPACT: the GIO ID word must answer byte and halfword reads

Status: fixed 2026-11-22 (issue #833: OpenBSD 6.9 stops after "Found SGI-IP28,
setting up." on `highimpact`; the same ISO boots on `xz`).

Symptom: the ARCS loader prints its banner and the kernel prints nothing,
forever. The guest is not hung: PC samples sit in `sched_idle` (the idle loop)
and the kernel is waiting for the installer's console it never got. Kernel
memory shows `giofb_consid` = 0 and `impact_cons` all zero: the Impact
console was never attached, so no kernel output reaches the screen, which
keeps showing the PROM's text window.

Cause: OpenBSD's `gio_id()` (sys/arch/sgi/gio/gio.c) reads the word at the
slot base, then a halfword at +2 and a byte at +3, and only trusts the word as
a 32-bit ID when they match its low bytes. The board answered the word
(0x00050190) but read 0 at +2 and +3 (a register-file lookup of an unmapped
offset), so the probe fell through to the "is it a REX3 or GR2" tests and
returned nothing. The MGRAS trace shows it: `rd gio_id = 0x00050190`,
`rd [0x00002] = 0`, `rd [0x00003] = 0`, then no more board accesses at all.
IRIX reads the ID as a word, which is why only this probe noticed.

Fix: `read_subword` in `src/dev/mgras/mod.rs` answers an 8- or 16-bit read of a
32-bit register with the part of the word at that address (high byte first). The ID word and the plain
register store use it; the DCB windows keep their own narrow-read handling,
which IRIX relies on. Writes are unchanged. The GIO bus spec (2.10.1) says only
that the ID must be read as a word, so what a real board answers to a narrow
read is unchecked (the driver accepts the right bytes or a bus error; zero is what
breaks it).

How to reproduce and look: `[machine] profile = "indigo2_ip28"`, banks
`[256, 0, 0, 0]`, `[graphics] board = "highimpact"`, the sgi install69.iso
on `[scsi.4]` with `cdrom = true`, `--ci`; dismiss the failed autoboot, open
the PROM menu's Command Monitor and `boot -f
scsi(0)cdrom(4)rdisk(0)partition(8)/sash64
scsi(0)cdrom(4)rdisk(0)partition(0)/bsd.rd.IP28`. The kernel's symbols are in
`bsd.rd.IP28` (not stripped): the ISO is an SGI volume header, then a
big-endian UFS1 filesystem at partition 0 (superblock 8 KiB in), so the
kernel is read with a small UFS reader rather than an ISO9660 one.
`mgras trace FILE all` from the boot command on shows the kernel's board
accesses.
