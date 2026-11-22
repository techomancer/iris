# Which interrupt is IRIX's clock tick, per machine

Status: rule (2026-11-22). Feeds the status bar's per-CPU tick light and the
monitor's `ticks` (`status::CpuStatus`).

| Machine | Clock interrupt | Programmed rate under IRIX 6.5 |
|---|---|---|
| Indy, Indigo2 (IP22/IP28) | IOC 8254 timer 0 (system clock) and timer 1 (fast clock) | timer 2 divides the 1 MHz input to 5 kHz; timer 0 count 50 = 100 Hz; timer 1 count 5 = 1 kHz while the fast clock runs, count 0 when it is parked. The light counts both, so 1100 Hz or 100 Hz |
| O2 (IP32) | CP0 Compare | Compare advanced by the interval over a free-running Count: 100 Hz, interleaved with the profiling clock when that runs |
| Octane (IP30) | CP0 Compare | HEART's counter is the time base, not an interrupt. Compare carries the 100 Hz tick, the 1 kHz fast clock while it runs, and timeouts aligned to them, so the rate is the mean of a mix |
| Origin 200/2000, Onyx2, Fuel, Tezro, Origin 3000 (Hub/Bedrock) | CP0 Compare, per CPU | 100 Hz. The Hub's PI RT compare is the timeout timer and the profile compare the fast clock; neither is the tick |

## Measuring the programmed Compare rate

The SN machines and the Octane zero Count at each tick and write Compare as
the interval less the time already past the old Compare, so Compare is not
the interval. `cpu::compare_rate` folds each Count write back into one
timeline (what the write took off the Count the guest last read or wrote),
so a deadline is Compare plus that offset and the interval is the distance
between successive deadlines, smoothed. Acknowledgements and parking values
(cases 1 and 3 of the Compare write's classification) are not deadlines; a
deadline before the previous one (a timeout pulled in) replaces it unsampled.

## Gotchas

- The 8254 model does not run a channel with a zero count (real silicon
  counts 65536), so a parked fast clock delivers nothing and publishes 0.
- An IOC1 Indy's 8254 is broken and IRIX falls back to Compare for the clock;
  the IOC we model is not an IOC1, so Indy and Indigo2 do not publish from
  Compare.
- Before this, CPU 0 of the Onyx2 and Onyx 3000 counted a zero `Arc`; the
  status now takes CPU 0's counter after the CPUs exist
  (`MachineStatus::without_ticks` + `set_cpu_ticks`).
