# Octane: the module record must match the active CPUs

Status: rule, found 2026-10-30.

The PROM reads the processor module's name from HEART's MicroLAN and compares it with
`STATUS`'s active bits: `PM10` with one CPU, `PM20` with two. A mismatch shows
`CPU board NIC diagnostic *FAILED*` and `Diagnostics failed. [Press any key to continue.]`,
and the boot waits there forever on a headless run, with CPU 0 looping in the PROM.
`OctaneMachine::new` picks the name from `cpus`.

The part number in the record is still the PM10's. The PROM does not check it.
