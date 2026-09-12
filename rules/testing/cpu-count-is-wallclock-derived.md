# CP0 Count is wallclock-derived, so Count tests flake

IRIS derives CP0 Count from a host `Instant` anchor plus a calibrated
`count_hz`, not from a guest instruction count — `src/mips_exec.rs:9686` calls
`count_anchor_instant` a wallclock anchor and notes it is "meaningless across
runs". Count therefore advances with **host** time, and how much of it passes
between two guest instructions depends on host scheduling.

That makes any test which writes Count and reads it back a few instructions
later non-deterministic under load.

## The one that bites

`cpu-tests/tests/cp0/cp0.c:296` — `cp0/count_writable`:

```c
cp0_count_set(0x12345678u);
CHECK_EQ(cp0_count() & 0xFFFF0000u, 0x12340000u);
```

Only the top 16 bits are asserted, which sounds tolerant, but the headroom is
`0x10000 - 0x5678` = 43,400 counts — under a millisecond of host time at a
typical `count_hz`. If the CPU thread is descheduled between the write and the
read, the top half rolls to `0x1235` and the check fails.

Demonstrated: the **same** ELF, run three times, gave PASS, PASS, FAIL. It is
not caused by whatever change you just made — check for this before believing a
one-check regression in `cp0/`.

## Why it matters for the hardware diff

On a real R4400 or R5000, Count increments at exactly half the pipeline clock
and is fully deterministic, so this test should pass reliably on hardware. A
difference here would land in the "passes in IRIS, fails on hardware" bucket (or
its opposite, run to run) and mean nothing about the CPU model.

The test now **retries** — up to eight attempts, passing on the first clean
one. The property is that the write sticks, not that the host stayed on CPU.
Real silicon gets it first time, so hardware results are unaffected.

That was not optional in the end: at 124 failing checks the CI baseline is
exact, and this one check varying took two of four cells red with
"125 failing checks, baseline is 124 — 1 new". A single non-deterministic
check is affordable when nobody counts; it is not once something does.

`cpu-tests/run/diff-hw.py` still excludes it by name as belt and braces;
`--include-flaky` overrides that. Any other Count-derived assertion added later
belongs in the same list.

## Related

- [running-cpu-tests-on-real-hardware.md](running-cpu-tests-on-real-hardware.md)
- The suite's own scorecard drifts for unrelated reasons too — see
  [cpu-tests-scorecard-drift.md](cpu-tests-scorecard-drift.md).
