# Start the compile pool after the ppmem windows are published

Status: rule, found 2026-10-30 replacing the O2 cache handover. `src/machine.rs` always had
the right order; `src/ip32_machine.rs` and `src/ip30_machine.rs` were fixed then.

## Symptom

Memory-bound guest code slower under jitv2 with no error, and different from boot to boot.
On the O2 R5000, `iris_bench` gave 3.82-3.99 MB/s in one boot and 3.08-3.64 in the next,
against 4.33-4.70 on the build it replaced. Dhrystone and Whetstone overlapped. In
`j2 status`, entries of the same instruction count had less code (5-instruction entries
averaged 794 bytes against 1014). The counter that shows it directly is the exit log's
`jitv2 inline memory: emitted=N declined=M`: `declined` should be 0 on a tcache build.

## Cause

`worker_loop` copies `Jitv2::dc_geometry` into its `Codegen` once, when the thread starts.
`ShadowCache::jit_dc_geometry` reports `unsupported` until both the window and the generation
window are set. A worker that reads it before `set_tcache_gen_window` compiles every load and
store as a call out, for the life of the thread. The workers race the publishing thread, so
the mix differs per boot.

`install_jit_mem_ptrs` flushes and restarts workers only if the geometry changed with
`pages_used() > 0`. At machine build no page is used, so nothing corrects them.

The O2 hid this for as long as it had a cache handover: the switch to the shadow model
changed the geometry, and that flush restarted every worker with the right one.

## Rule

In machine construction: bitmap sinks, `set_tcache_window`, `set_tcache_gen_window`, then
`compile_queue.start`. A flush that happens to fix the speed of a machine may be hiding this.
