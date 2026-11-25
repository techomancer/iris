# Octane: the module record must match the active CPUs

Status: rule, found 2026-10-30.

The PROM reads the processor module's name from HEART's MicroLAN and compares it with
`STATUS`'s active bits: `PM10` with one CPU, `PM20` with two. A mismatch shows
`CPU board NIC diagnostic *FAILED*` and `Diagnostics failed. [Press any key to continue.]`,
and the boot waits there forever on a headless run, with CPU 0 looping in the PROM.
`Identity::processor_module` picks the record from the CPU part and `cpus`
only (R10000: `PM10`/`PM20`; later parts `PM10<MHz>MHZ`/`PM20<MHz>MHZ`).
The PROM checks only the prefix, not the part number.

Until 2026-11-25 a V10/V12 (Octane2) board forced `PM10360MHZ` whatever
`cpus` said, so a dual-CPU V12 Octane failed this diagnostic. Graphics
choose the chassis records (Octane2 Crossbow, Bridge, system board, front
plane, power supply), never the processor module.

On real units the board revisions track the HEART and Crossbow revisions:
the newer HEART allows denser memory (black-handle modules up to 256 MB per
slot, silver-handle up to 1 GB), and Crossbow 1.4 was SGI's stated VPro
requirement but mattered little in practice (owner, 2026-11-25). The
emulator gates neither, so the chassis revision letters are cosmetic.
