# A dropped machine must free its CPUs

Status: rule, Nov 2026.

**Symptom.** `exit_persistence_tests::tezro_power_off_then_drop` failed on
Windows CI only: restoring the snapshot after the drop could not delete a disk
image ("being used by another process", os error 32). On Unix the delete
succeeds on an open file, so the test passed there. `lsof` on the test process
after `drop(machine)` showed every Tezro disk, overlay and the flash store still
open, three seconds later too. The Octane released everything.

**Cause.** Under jitv2 the Fuel, Tezro and Origin 3200 constructors handed the
compile pool `Arc::downgrade(&cpu_device)` and then `std::mem::forget(cpu_device)`.
The weak handle was already all the pool needed; the forget leaked one strong
count, so CPU 0 (each node's CPU on the Origin) never dropped. A CPU's
`batch_hook` closure owns an `Arc` of the board (it runs `board.poll()` and the
presence holds), so the board, its SCSI buses and the flash backing lived on.

**Rule.** Never `forget` an `Arc` to keep a weak handle valid: the weak handle
upgrades as long as the machine's own `cpus` hold the CPU. Each of these
machines has a test that drops it and checks a `Weak` of its board or CPUs no
longer upgrades (`a_dropped_tezro_frees_its_board`,
`a_dropped_fuel_frees_its_board`, `a_dropped_origin_frees_its_processors`).
Anything that closes over the board from inside a CPU is a cycle candidate.
