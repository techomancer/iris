# Raw `*mut Machine` holders must share the slot that Drop nulls

**Keywords:** monitor,systemcontroller,ci,save,use-after-free,iris-gui,drop,port 8888
**Category:** snapshot

## Symptom

iris-gui crashed with SIGSEGV (`KERN_INVALID_ADDRESS` at `0x10`) in
`Machine::save_snapshot`, called from `SystemController::execute_command`
("save") on the monitor thread. The thread list showed several copies of
`machine-events`, `rex3-jit` and `VINO-DMA`: more than one `Machine` had been
created and dropped in that process.

## Cause

`register_system_controller` handed the monitor (and the `machine-events`
thread) a raw `*mut Machine`, on the assumption that the Machine lives for the
whole process — true for the CLI, false for iris-gui, which boxes a Machine on
Start and drops it on Stop. The monitor thread outlived it, so every controller
command after a Stop dereferenced freed memory. `save` was just the first to
touch it.

Because the monitor was never torn down, the first Machine's monitor also kept
port 8888. Later Machines logged `monitor disabled: failed to bind`, and every
monitor command after a Stop → Start went to the *first* (dead) Machine's
devices. Results gathered over 8888 after a restart in the same GUI process are
not trustworthy.

## Fix

- `Machine::machine_slot()` creates one `Arc<Mutex<MachineRef>>` per Machine.
  `SystemController`, the `machine-events` thread and `CiServer` all use it.
- `with_machine_slot` null-checks under the lock; `Drop for Machine` nulls the
  pointer under the same lock before any field is dropped. The lock is what
  makes this sound — Drop waits for an in-flight `save`/`reset` to finish.
- `Monitor::shutdown` (called from `Drop`) closes the listener, freeing 8888,
  and clears the device list, which also drops the old `mc` clone so the old
  `machine-events` thread's channel closes.

`src/bench_runner.rs` deliberately never registers a controller for the same
reason; it does not need the slot.

## Rules

- Never hand out `self as *mut Machine`. Use `machine_slot()` and
  `with_machine_slot`.
- No command run through the slot may drop the Machine, or `Drop` deadlocks on
  the lock its own thread holds.
- Call `machine_slot()` only once the Machine is in its final location (boxed,
  or on `main`'s stack); a debug assertion catches a move.

Regression tests: `machine::controller_lifetime_tests` and
`monitor::tests::shutdown_releases_the_port`.
