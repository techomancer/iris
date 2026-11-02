# Octane: quitting during the PROM's halt leaves the next boot at "Okay to power off"

Status: rule (measured 2026-11-02 on IP30, stock 6.5.30 PROM, IRIX 6.5.30, iris 3cba53ad).

**Symptom.** A harness halts IRIX (`/etc/shutdown -y -g0 -i0`), waits until
the PROM restarts, and quits iris. The next start of the same run directory
prints the usual PDS warning on the serial port and then nothing: no
"Starting up the system", no IRIX banner, iris at about 100% CPU. A
screenshot (`iris-ci screenshot`) shows the PROM's "Okay to power off the
system now. Press any key to restart." dialog on the graphics console.
A key on the serial port does nothing because the console is the graphics
head.

**Cause.** As the machine resets for the halt, a pending-halt request lands in
the RTC's battery-backed extended RAM (byte 0x0a goes from `0x30` to `0x33`,
and the checksum at register 0x0f changes with it). The file showed it
within a second of the PROM's first PCI probe line. Whether IRIX or the
restarting PROM writes it was not traced. The PROM acts on the request once it
has redrawn the graphics console, about 6 to 7 host seconds later. Then it shows
the dialog and writes the byte back to `0x30`. `iris-ci quit` stops the
machine at once and writes the NVRAM file as the guest left it, which is what
a battery-backed part does when the power is cut. A quit inside that window
saves the request. On the next power-on the PROM reads it and goes straight to
the dialog. The NVRAM file is correct; the harness pulled the plug early.

Evidence:

- Quitting 5 s after the first PCI probe line of the PROM restart: 6 of 6
  next boots stopped at the dialog, and each saved file held `0x33`.
  Quitting as soon as that line appeared: 1 of 5 next boots stopped there.
  It is a race between the quit and the write.
- Copying the file every second through a halt showed `0x33` appear with the
  PROM restart and `0x30` return in the same second the dialog appeared.
- A start from a saved `0x33` file reached the dialog. A start from a `0x30`
  file taken after the dialog reached a login prompt. The two files differ
  only in that byte and the checksum.
- Setting the kickstart flag (bit 0 of extended control register 4A) in the
  `0x33` file does not change this. The PROM reads 4A at power-on and writes
  it back, then still shows the dialog.
- Across all 32 saved files, the next boot stopped at the dialog exactly when
  the file held `0x33`.

It has nothing to do with the boot flash. Without `[octane] flash` the flash
starts from `prom` on every run, so every start prints "Bad/Uninitialized
Flash Persistant Data Storage Header / Re-Initializing PDS", hang or not.

**Rule.** To stop an Octane guest cleanly from a script, either:

- power it off from IRIX (`halt -p`): iris exits on the
  guest's soft power-off (register 4A, PAB), or
- after a halt, wait until the NVRAM file has stopped changing for some
  seconds before `iris-ci quit`. With 10 s without a change, 9 of 9 next
  boots went past the PROM into IRIX (one of them then stalled in IRIX's
  startup scripts, a different problem). Powering off gave 9 of 9 logins.

The `irix-guest-loop` skill already powers off rather than halting.

The 100% CPU is the guest waiting at the dialog for a key: iris used about
94% of a host core there.
