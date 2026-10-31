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

## Octane after the same fix (open)

Octane, R10000, 256 MB. Dhrystone 10M, host seconds per run, 3 boots of each build plus 2 of
each from an earlier build pair, always pinned to one NUMA node:

| Order | Best | Runs at 37 s or more |
|---|---|---|
| old (pool first) | 35.13 | 0 of 18 |
| new (windows first) | 35.19 | 10 of 18 |

Same best time, so compiled code is not slower as such, but the new order often lands near 38 s,
sometimes within one boot and between runs of the same binary. `iris_bench` is level. The
1 s Dhrystone clock cannot show this; use host time. Cause not found. Ruled out: memory helpers
(off unless `IRIS_MEM_HELPERS`), and mega-flushes (none). New emits about 2% fewer inline sites
(630-640k against 645-650k), unexplained. The old order showed `declined=0` in 6 boots, so no
boot here shows it costing speed on the Octane.
