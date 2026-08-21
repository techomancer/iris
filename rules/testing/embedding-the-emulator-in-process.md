# Running a bare-metal image inside the host process

`iris` is a library, and `iris-gui` already runs a `Machine` on a worker thread.
So a bare-metal suite (`bench/`, and in principle `cpu-tests/`) can be driven
entirely in-process — no `iris` subprocess, no ELF on disk, no stdout to parse.
That is what makes the benchmark shippable: a sandboxed application has no
toolchain and no writable path to unpack an image to.

`crate::bench_runner` is the working example. What had to change, and why.

## `TestDevice::exit` can just return — no parking, no signalling

This looked like the delicate part and turned out to be the easy one. The
`EXIT` store lands on the **CPU thread**, inside the guest's own instruction, so
the obvious worry is what that thread does next.

It does not matter, because **every guest that has a test device reaches `EXIT`
through `testdev_exit()`, which spins forever afterwards**
(`cpu-tests/harness/console.c`) — a bare-metal image has nowhere to return to.
So the handler can fire a callback, return normally, and let the CPU thread go
back to looping in guest code that does nothing while the *runner* thread stops
the machine at its leisure.

Do **not** block in the handler instead. `Machine::stop` joins the CPU thread,
so a handler that waits for the runner deadlocks against the runner waiting for
the join.

## Do not call `register_system_controller`

It hands a raw pointer to the `Machine` to a thread that outlives the call. That
is fine for `main.rs`, where the machine lives as long as the process, and wrong
for a runner whose machine is dropped when the run finishes. Skip it: nothing an
embedded run needs (`reset`, `save`, `load`, guest-initiated power-off) applies.

## Spawn a thread with a big stack

`Machine::new` puts a >1 MB device map on the stack. Windows gives a thread 1 MB
by default, so construct it on a `Builder::new().stack_size(16 << 20)` thread —
`main.rs` does the same thing for the same reason. Doing it inside the runner
rather than asking callers to remember is what keeps the API a plain function.

## `Machine::start` does not always start the CPU

`self.cpu.start()` there is behind `#[cfg(not(any(debug_assertions, feature =
"developer")))]`, so a debug build — which is what `cargo test` produces —
starts every device *except* the CPU and then sits there. Call
`Machine::cpu_start()` explicitly afterwards; it is a no-op if the CPU is
already running.

## Give the guest its configuration through a register

A bare-metal image loaded with `--load-elf` has no argv and no environment.
`TESTDEV_RUN_CONFIG` (`src/testdev.rs`, `RunConfig`) is the channel. Encode it so
that **every field means "unrestricted" when zero**: that is what an emulator
predating the register returns from an undecoded address, so the guest can read
it unconditionally with no capability check of its own beyond the one that
already exists. Verified both ways — the current guest image runs correctly on
an emulator built before the register existed.

## `MachineConfig::default()` attaches a disk

`default_scsi()` puts `scsi1.raw` on ID 1, and startup is fatal when the file is
absent — which it always is for a bare-metal run. Clear `cfg.scsi` (this is the
same reason `bench/run/bare.toml` carries a present-but-empty `[scsi]`).
