# Hardware oracle

Logs from real SGI Indys, and the reference the suite's expectations are
validated against. Real hardware is the oracle (`PLAN.md` §6, priority 4): it is
the only way to tell an emulator bug from a bad test, and it settled both.

## The machines

| | PRId | FIR | Config | L2 |
|---|---|---|---|---|
| R4400 rev 6.0 | `0x00000460` | `0x00000500` | `0x20c1c483` | yes |
| R5000 rev 1.0 | `0x00002310` | `0x00002310` | `0x1043e6f3` | no |

Both booted from a BlueSCSI v2 at SCSI ID 2 through the machine's own PROM,
`console=g`, with no serial cable attached. Output reached the screen through
ARCS and the log reached the disk through the PROM's own driver — see
`rules/testing/arcs-console-from-bare-metal.md`.

## The logs

| file | result |
|---|---|
| `r4400-rev6.0-run1.log` | 64 checks / 16 tests failed — the first oracle data this project had |
| `r4400-rev6.0-run2.log` | 24 / 3 |
| `r4400-rev6.0-run3.log` | 7 / 1 |
| `r5000-rev1.0-run1.log` | 11 / 3 |
| `r5000-rev1.0-run2-clean.log` | **0 / 0**, `rc=0` |
| `r4400-rev6.0-run5-clean.log` | **0 / 0**, `rc=0` — 2164 checks |
| `diff-r4400.txt`, `diff-r5000.txt` | three-bucket classification of each clean run |

`ELF.md5` pins the guest binary all of these were produced with, and notes the
one change made since. A diff is only meaningful between logs from the *same*
ELF.

Both clean runs are against the **same** current ELF, so `r4400-…-run5-clean`
also serves as the regression check on the three changes the R5000 forced
(`mem/lwr_all_offsets` gated, `fpu/vec_cvt_from_l` ungated, `identity/fir`
masked). It passed 240/240, so none of them broke the R4400 path.

## What it found

Of 31 tests that disagreed with the emulator on the first R4400 run:

- **15 were IRIS bugs** — they fail in the emulator and pass on silicon. None
  turned out to be a bad test. The suite's existing findings were all real.
- **16 were bad expectations** that IRIS happened to share, which is exactly why
  they had never shown up. Written up in `docs/gotchas.md`; the largest is that
  quiet and signalling NaN handling was swapped in both the arithmetic and the
  comparison tests.

Correcting the 16 moved IRIS from 61 failed checks to 124, and the R5000 then
raised the confirmed bug count from 15 tests to 22. The suite now measures the
emulator against silicon rather than against itself, and the gap it reports is
IRIS's real bug surface.

## Re-running

```sh
run/extract-log.py <card>/HD20*.img -o hw.log   # recover the log from the card
run/diff-hw.py build/emu-<cpu>.log hw.log       # classify it
```

Extract before re-running: the log lives at a fixed LBA (8192) and the next run
overwrites it. Keep the image at **SCSI ID 2** — `SCSILOG_ARCS_PATH` is
`dksc(0,2,10)`, and at any other ID the suite runs and prints normally while the
log write silently fails.
