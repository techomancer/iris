# Running cpu-tests on a real SGI Indy

Real hardware is the oracle (`cpu-tests/PLAN.md` §6, priority 4): it is the only
way to tell whether a suite failure is an IRIS bug or a bad test. This note
records the bring-up work for that run.

**Status: done.** The suite ran on a physical **SGI Indy, R4400 rev 6.0**
(PRId `0x460`, FIR `0x500`, Config `0x20c1c483`, `testdev no`) on 2026-09-11,
booted from a BlueSCSI v2 through the machine's own PROM. Logs are archived in
`cpu-tests/oracle/`.

Four runs, each correcting what the previous one exposed:

| run | result | what it found |
|---|---|---|
| 1 | 64 checks / 16 tests failed | the first oracle data this project ever had |
| 2 | 24 / 3 | 13 corrections confirmed; the cache hazard fix worked |
| 3 | 7 / 1 | NaN compare rule was half-right; diagnostics identified the last one |
| 4 | **0 / 0** | all 240 pass on silicon |

A second machine, an Indy **R5000 rev 1.0** (PRId `0x2310`), then ran the same
binary and reached **240/240, `rc=0`** — and its diff against the emulator is
`IRIS-WRONG 0, TEST-BUG 0, IRIS-BUG 22`.

Of the 31 tests that disagreed on the first R4400 run, **15 were IRIS bugs**
(they fail in the emulator and pass on hardware) and **16 were bad expectations**
that IRIS happened to share. Correcting the 16 moved IRIS's own score from 61
failed checks to 124, and the R5000 raised the confirmed bug count from 15 tests
to 22 — six of which only became visible *because* the expectations had been
corrected. The suite now measures the emulator against silicon rather than
against itself.

Only one test needed CPU gating: on a partial `LWR` an R4400 leaves the upper
half of `rt` untouched while an R5000 sign-extends. Everything else behaves
identically on both parts, including the NaN handling and the refusal to convert
a 64-bit integer past 2^53.

## The thing that stops a naive attempt: there is no console

`con_init()` is a deliberate no-op — the harness writes SCC channel B registers
directly and relies on the PROM having programmed WR5 and the baud rate. That
only happens when the PROM's own console is serial.

An Indy driven from a keyboard and monitor is on `console=g`, and then **the
suite produces no visible output whatsoever** — not the banner, not `RESULT:`,
not `IRIS-CPUTEST-DONE`. The harness never goes through the PROM console, so
nothing it prints reaches the screen. This is not a degraded mode; it is silence.

Two ways out, and they are not equivalent:

- **Serial.** `setenv console d`, capture at 9600 8N1 on `ttyd1`, which is SCC
  **channel B** (`harness/iris.h:11`), *not* channel A. Note `HELP.md:307` calls
  `ttyd1` the Mini-DIN port; channel A (`ttyd2`) is the DB-9. Cheapest by far
  when the cable exists.
- **`harness/scsilog.c`.** Mirrors the console into RAM and writes it to fixed
  LBAs on the boot disk. Needed when there is no serial adapter at all.

Setting `console=d` with nothing attached to the serial port is the worst case:
the PROM prompt goes to the serial line too, so there is no way to type `boot`.

## scsilog: getting the log off a machine with no console

`harness/scsilog.c` tees every `con_putc` byte into a 64 KB buffer and, at the
end of the run, writes it with HPC3 DMA + WD33C93A `SELECT_ATN_XFER` to
`SCSILOG_LBA` (8192) onward. Recover it with `run/extract-log.py` after moving
the SD card to a host; the region starts with `IRISLOG1` and a big-endian length.

Four things that were not obvious:

- **PIO writes are not available.** `TRANSFER_DATA_OUT` is raised in exactly one
  place in `src/wd33c93a.rs` — inside the `TRANSFER_INFO` handler — and never
  from `SELECT_ATN_XFER`. The path IRIX and the PROM use is DMA with the
  descriptor pre-armed, so that is the path to copy. Writing the PIO
  phase-stepping variant instead would exercise emulator code no real guest
  touches, and passing under IRIS would prove very little about the Indy.
- **The register values came from the firmware, not a datasheet.** Boot the
  suite through the PROM and run `scsi regs` on the IRIS monitor while it is
  live: the chip still holds what the PROM's own driver left. That yields
  `CONTROL = 0x2d` — DMA-mode field `001`, burst — where a datasheet reading
  suggests `100`/`0x80`. Also `OWN_ID = 0x0a`, `TIMEOUT = 0x40`, and
  `DEST_ID` bit 6 as the data-phase direction (set for the PROM's READ, so
  clear for our WRITE). Copying firmware verbatim is the closest thing to a
  hardware guarantee available without hardware.
- **The DMA length must equal the WRITE(10) length.** A final short chunk —
  2920 bytes of data against a 6-block (3072-byte) command — leaves the transfer
  never completing, and the tail of the log is silently zeros while the earlier
  chunks land fine. Round every chunk up to a whole sector.
- **Use the KSEG1 aliases, via `K1_PTR`.** The buffer and descriptor are only
  ever touched uncached, so no cache writeback is needed before the engine reads
  them, and the flush stays correct after the TLB and cache tests have finished
  mangling state — KSEG0/KSEG1 are unmapped, so nothing depends on the TLB. Note
  `iris.h:213` : a bare `u32` KSEG1 address zero-extends into TLB-mapped xkuseg
  in 64-bit mode, which is why the helper exists.

With no disk attached (`--load-elf` against `run/bare.toml`) the flush returns
`-3` promptly rather than hanging, so it costs nothing on the emulator path.

## Three harness fixes that only matter on hardware

None is observable under IRIS, and all three were found by reading.

**The test-device probe ran before the exception vectors existed.**
`testdev_probe()` reads `0xBF400000`, GIO expansion slot 0, which is empty on a
stock Indy and takes a bus error there. `start.S` has already set `BEV=0`, so
that dispatched to whatever the PROM left at `0x80000180` — the suite would very
likely have died before printing its banner, and it would have looked like the
BlueSCSI failed. `exc_install()` now precedes the probe.

The guard went in the **caller**, `testlib.c`, not in `console.c` as first
drafted: `console.c` is compiled into `bench/` too, and `console.h` deliberately
excludes the exception plumbing so it can be (`bench` has its own identical
`struct exc_record`). Referencing `exc` from `console.c` does not compile. Since
`bench` never calls `testdev_probe()`, the caller is also the smaller diff.

The `exc.count` test is not redundant: the default resume mode is
`EXC_RESUME_SKIP`, so the faulting load is stepped over and its destination
register keeps its prior value — the magic comparison alone is not trustworthy.

**The `scc_dead` latch never fired on hardware.** It latched only
`if (have_testdev)`, which is 0 on a real Indy, so with an unprogrammed SCC —
which is what `console=g` leaves you, since the PROM only programs its own
console channel — every character burned the full `TX_SPIN_LIMIT` of 100,000 on
an uncached read, for all ~19,000 of them. Silent and extremely slow, and easily
mistaken for a hang.

The condition is now `if (have_testdev || con_tap)`: give up on the port once
some *other* sink is capturing the run. The original reasoning still holds where
it applied — with no other sink, a slow port beats no output — it just no longer
describes a machine with the disk tap installed. A `console=d` run has a working
SCC, never trips the limit, and is unaffected, which `run-prom.sh` depends on
since it greps the serial log for its verdict.

## Verified in the emulator

- The image boots through the real PROM and runs to completion:
  `make image` (with `--size 16M`), then
  `IRIS=../target/release/iris run/run-prom.sh 2 cputest`.
- The disk log is **byte-identical** to the serial log across all 19,287 bytes
  of suite output.
- The probe reorder changes nothing: the restored ELF is byte-identical, and the
  only differing check is `cp0/count_writable`, which is non-deterministic —
  see [cpu-count-is-wallclock-derived.md](cpu-count-is-wallclock-derived.md).

## Still to do on the day

- Confirm the PROM banner CPU first. `testlib.c:172` maps only `IMP_R4400` and
  `IMP_R5000`; anything else is `cpu_kind = 0`, `UNKNOWN CPU — refusing to run`,
  exit 127. An R4600 Indy is common and will be refused. That is deliberate, not
  a bug: the expectations are selected by PRId by construction.
- Name the image `HD2_512.img` at the SD card root — `hd` prefix, third
  character is the SCSI ID and must match `boot -f dksc(0,2,8)`, block size after
  the first `_`. It is a **hard disk**, not a CD: `mkvh` emits 512-byte sectors
  and BlueSCSI defaults optical devices to 2048.
- The machine hangs after the last line by design — `testdev_exit()` spins
  forever with no test device. `IRIS-CPUTEST-DONE` is the end; power-cycle after
  it. `rc=` saturates at 100, so read `RESULT:` for the real count.
- Then `run/diff-hw.py build/emu-<cpu>.log <hardware>.log`, from the **same
  ELF**, and file each difference per the three buckets it prints.
