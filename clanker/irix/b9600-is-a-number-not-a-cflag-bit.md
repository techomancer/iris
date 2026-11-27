# IRIX 6.5's B9600 is the number 9600: never OR it into c_cflag

Status: rule, 2026-11-26.

**Symptom.** A guest test program opens `/dev/ttyd2`, sets
`t.c_cflag = B9600 | CS8 | CREAD | CLOCAL`, and a device on the port answers
in garbage after the first byte. The SCC shows WR4 = 0x47 (parity on, even),
so the host port runs 8E1 against a device sending 8N1.

**Cause.** IRIX 6.5's termios carries the speed in `c_ospeed` / `c_ispeed`,
and `B9600` is the value 9600 (0x2580), not an old CBAUD code. OR-ed into
`c_cflag` it sets PARENB (0x100), HUPCL (0x400) and CREAD.

**Rule.** In guest test programs set speeds with `cfsetospeed` /
`cfsetispeed` only. When a serial device misbehaves in the guest, read WR4
with the monitor's `serial status` before suspecting the emulator.
