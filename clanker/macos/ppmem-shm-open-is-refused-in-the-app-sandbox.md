# ppmem's shm_open is refused in the App Sandbox

Status: fixed 2026-11-27 (Mach memory entries on macOS when `shm_open` is refused).

**Symptom.** A sandboxed build (the `appstore` feature, signed with
`installer/iris-gui-sandbox-local.entitlements` or the App Store's) launches
and shows its library, but Power On panics on the emulator thread:

    ppmem: failed to allocate RAM bank: ... "ppmem: shm_open: Operation not permitted (os error 1)"

**Cause.** `src/ppmem/map_unix.rs` `SharedMem::new` backs each RAM bank with a
POSIX shared-memory object (`shm_open` + immediate `shm_unlink`), so the bank
can be mapped at several addresses (mirrors, the JIT's generation window).
The App Sandbox allows POSIX shm only under a name that starts with an
application-group ID and a slash, and macOS limits a shm name to 31 bytes, so
a `<TeamID>.<group>/` prefix leaves almost nothing. ppmem became
unconditional on 2026-10-01 (4d6e97c1, "feature flag cleanups"), after the
last App Store build (2026-08-31), so no shipped build had it.

**Fix (done).** On macOS, when `shm_open` fails with `EPERM`, `SharedMem`
backs the object with a Mach memory entry instead (`mach_vm_allocate`, then
`mach_make_memory_entry_64`, then `mach_vm_map` it with `VM_FLAGS_FIXED |
VM_FLAGS_OVERWRITE` at each address), which needs no name and no file. A
temporary file in the container's `TMPDIR` would also have worked, but dirty
guest RAM could then be written back to disk. `IRIS_PPMEM_BACKEND=mach` forces
the Mach path; the whole library suite passes on it (5122 tests), and IRIX 5.3
boots, logs in and powers off in the sandboxed Mac app.

**Test.** `scripts/build-macos.sh appstore` builds the sandboxed Mac app
locally; use a throwaway bundle id (`bundle.sh --appstore --bundle-id ...`) so
its container is not a real App Store install's.
