# A by-value Rex3Context costs every inlining caller a megabyte stack frame

Status: rule, fixed 2026-11-22 (`prepare_host_line` is `#[inline(never)]`).

## Symptom

Under the Newport GPU raster (`raster = "wgpu"`), a GO cost the
`REX3-Resident-Processor` thread 3.3 us. `perf` put 42% of the thread in
`ResidentGpu::enqueue_at`, and 87% of that on three instructions of its
prologue: the stack probe (`sub $0x1000,%rsp; movq $0,(%rsp)` in a loop) for
a 1 MiB frame, run on every call.

## Cause

`Rex3Context` ends in `hostrw: HostRwArray`, 1 MiB inline. `line.rs`
`prepare_host_line` copies the context (`let mut simulated = *context;`) to
simulate the walk. Inlined into `enqueue_at`, that local made every caller's
frame 1 MiB, so every GO probed 256 pages, whether or not it was a host line.

## Rule

- Never copy a `Rex3Context` by value on a hot path. If a function must, mark
  it `#[inline(never)]` so only its own calls pay.
- Check: `objdump -d iris | awk '/^[0-9a-f]+ <.*>:$/ {fn=$2} /sub +\$0x10[0-9a-f]{4},%r11/ {print fn}'`
  lists every function with a frame of 1 MiB or more; only
  `prepare_host_line` should be a Newport one.

The same profile found the other fifth of the thread in `clock_gettime`:
the batch-age check ran after every GFIFO entry, not every GO.
