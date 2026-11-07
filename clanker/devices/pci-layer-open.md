# PCI layer: open items

Status: open, 2026-11.

- **O2 RST# on cold reset** is not wired yet; the O2's cold reset leaves its
  PCI functions alone. When it is, it is inferred. MACE has a `PCIRst_N` output and
  CRIME drives the system reset, but no document says the board connects them,
  and no MACE register resets the PCI bus (MACE specification pp. 123, 162-165,
  188, 194). The O2 Device Driver Programming Guide (007-3443-002) mentions
  only the generic `pciio_reset()` interface; what that does on an O2 is
  unknown.
- **Restore is not reset, and the AIC-7880 chip reset.** A snapshot load puts
  the functions back over what is there and ends with `PciBus::redrive`; it
  does not pulse RST#. The AIC-7880 has no `reset` of its own: RST# clears
  its header and nothing else, and its chip reset is the `HCNTRL` `CHIPRST`
  write. Its load calls `reset_targets`, which is the SCSI bus reset
  (SCSIRSTO), to put the targets where a bus reset leaves them. Telling RST#,
  chip reset and SCSI bus reset apart, and making a load re-create rather
  than reset, is deferred until after the machine construction is reworked.
- **Bridge Control 27:24**: pin `n` resets PCI device `n` alone. The
  pin-to-device wiring is inferred from `pciio_reset()` ("without affecting
  other devices on the PCI bus", p. 550) and the four-pin count; the Bridge
  specification does not give it. Whether the stock PROM or IRIX ever writes a
  pin low has not been observed. A write is logged as a warning.
- **Bridge master time-out capture is an approximation.** Which device number
  the PCI error upper register names for a type 1 cycle nobody claims is not
  in the Bridge specification; iris records the device the cycle carried. For
  a PIO through the 2 GB window's PCI memory alias that nothing claims, iris
  names device 0 and records the PCI address the cycle carried.
- **Bridge 2 GB window, 16 MB to 1 GB, is an approximation.** The
  specification calls it reserved (p. 52) without saying what an access does;
  iris answers with a bus error. The 64-bit alias and PCI I/O space,
  reachable only through the 64 GB window, are not modelled.
- **No external ATE SSRAM.** The Octane does not fit it; `0x080000` reads as
  all ones, the way the PROM's size probe expects, and a write is accepted
  and dropped.
- **MACE 64-bit PIO** is not in the driver guide's table; iris splits it into
  two word cycles, high word first. MACE rev 0 sends 4-byte-aligned addresses
  on I/O cycles, so byte and halfword I/O may fail on it (MACE specification
  p. 167); iris does not model that.
- **Reset pins per device.** Which device each of the Bridge's four reset
  pins serves is inferred (see above); no board document gives it.
- **Octane pin rule** (A and C to the device's pin, B and D to device ^ 4) is
  inferred from the system's software, not established by a board document.
- **Bridge DMA** keeps its `st` lock and the `Weak` upgrade per call; that is a
  separate project.
- **A held reset pin.** While a Bridge reset pin is held low the functions
  still answer configuration cycles; iris acts on the assertion edge only.
- **AIC-7880 target state** (block size, write cache) is put back to the
  post-bus-reset state on a load rather than saved (see restore above).
- **Resettable on PCI functions.** Ioc3, Isp1040, Rad1 and VirtioPci have
  empty `Resettable` impls only because the device registry requires one.
- **A BAR that straddles a bridge window's edge** is not forwarded; what
  hardware does is undefined.
