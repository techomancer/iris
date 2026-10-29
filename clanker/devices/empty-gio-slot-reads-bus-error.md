# Reads of an empty GIO slot end in a bus error

Status: fixed 2026-10-29 (`GioBusErrorDevice`, `src/ip22_bus.rs`).

**Symptom.** Indy or Indigo2 with no graphics board (`--headless` with
Newport): the PROM printed `Cannot open video() for output` only after
200 000 `MC: GIO Timeout at 1f0f1338` lines, and IRIX later failed (the
install guide recorded a UTLB miss). The GIO slot probes of a normal boot
also "succeeded" on empty slots.

**Cause.** Unanswered GIO addresses returned all ones with `BUS_OK`. The
PROM and the kernel find boards with `badaddr()`, which expects a read of an
absent board to fault. With no fault, the PROM took the slot for a Newport,
wrote CONFIG and polled REX3 STATUS (offset 0x1338) for GFXBUSY to clear:
all ones never clears, so it ran its 100 000-iteration timeout, twice.

**Rule.** With `CPUCTRL1.ABORT_EN` (bit 4, "enable GIO bus time outs"; the
PROM sets it in its startup code) an unanswered GIO read returns `BUS_ERR`
(a data bus error exception) after recording GIO_ERROR_ADDR/STAT. Without
it, all ones as before. Writes are still dropped. Headless Indy and Indigo2
(IP28 checked) now boot IRIX to the serial login with `hinv` showing no
graphics; the PROM still says `Cannot open video()` because the NVRAM has
`console=g`, and waits in its menu instead of autobooting.
