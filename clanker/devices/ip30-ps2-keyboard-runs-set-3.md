# Octane: the PROM and IRIX run the PS/2 keyboard in scan code set 3

Status: rule (2026-10-27). Reference: `docs/dev/ioc3-ps2.md`.

**Rule.** On the Octane, PROM 4.17 and IRIX 6.5.30's `ioc3_pckm` both put
the keyboard in set 3 (F0 03), then FA (all keys typematic and
make/break), FC 14 and FC 76 (Caps Lock and Num Lock make/break only).
Linux and OpenBSD use set 2. A keyboard model that only speaks set 2 gives
IRIX the wrong codes on the Octane. `src/dev/ps2.rs` has the set-3 table
and per-key modes.

**Other things the drivers rely on.**
- K_RD/M_RD: first byte in 23:16 with VALID_0 (bit 31); a read empties
  the register. All of IRIX, the PROM, Linux and OpenBSD read slots 0, 1,
  2 in that order.
- IRIX's `pckm_pollcmd` returns the whole RD value after one command, so
  with CLAMP_THREE (KM_CSR 0x00C00000, what IRIX, the PROM and Linux set)
  an F2 answer FA AB 83 arrives in one read.
- IRIX assigns keyboard and mouse by the F2 answer, not by connector.
- IRIX's interrupt loop drains each port, then writes 0x400000 to SIO_IR
  (write-one-to-clear) and IES. Linux reads each port only once per
  interrupt, so the interrupt has to latch again for bytes that arrive
  after a read (a clamped device resumes then).

**How the PROM sequence was found.** A throwaway build with an
`eprintln!` in `Kbd::read_locked`/`write_locked` (`src/dev/ioc3/kbd.rs`), PROM 4.17, a
blank NVRAM (graphics console, so it probes the keyboard), an empty
`[scsi.4]` CD-ROM slot so no default disk is attached, run for ~100 s
under `timeout`; then `uniq -c` over the KBC lines (millions of empty
K_RD/M_RD polls after the setup). IRIX's side is from disassembling
`IP30boot/ioc3_pckm.o` (it has symbols and DWARF).
