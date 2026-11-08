# CPU-written GPU buffers live in host memory, allocated once

Status: rule, 2026-11-07.

**Symptom.** The SolidView replay ran 6.1 s on the GPU unit against 2.5 s
on the CPU unit. 40% of all samples were in the kernel's
`__alloc_range_bias`, reached through `amdgpu_cs_ioctl` and
`amdgpu_cs_bo_validate`. One test board took about 450 MiB of VRAM and
70 MiB of the CPU-visible window. A parallel `VPRO_GPU=1` test run made
the kernel log `Not enough memory for command submission` thousands of
times, starved the desktop (VS Code windows went white), and probably
took a running iris down with it.

**Causes.**
- `Queue::write_buffer` and `write_texture` create a new MAP_WRITE
  staging buffer per call. On Vulkan, `gpu-allocator` puts CpuToGpu
  memory in device-local, host-visible memory: the BAR window. On the RX
  6600 XT that window is 256 MiB (`lspci` Region 0 size=256M, so no
  resizable BAR for it), and the desktop fills it. The kernel kept
  failing to place each new buffer there on every submission.
- Default `MemoryHints::Performance` gives 256 MiB device blocks and
  64 MiB mappable blocks, so 176 MiB of buffers took two device blocks.
- An upload ring that grows when all its slots are busy grew to sixteen
  8 MiB slots on the 128 MiB mirror upload, then kept them.

**Rule.**
- Never `Queue::write_buffer`/`write_texture`. Use `Gpu::write` and
  `Gpu::write_texture`, which go through the fixed ring (4 x 8 MiB host
  slots) and are copied ahead of the next `Gpu::submit`.
- In the ring, a slot full of unsubmitted writes is submitted, not
  replaced. It grows only when every slot is in flight from earlier
  submissions. Within one large write it waits instead.
- What the CPU writes once and the GPU reads once is a `HostBuffer`
  (`src/wgpu/host.rs`): a Vulkan buffer from `ash` in HOST_VISIBLE |
  HOST_COHERENT, not DEVICE_LOCAL, preferably uncached; mapped for life;
  wrapped with `create_buffer_from_hal`. The batch's header, boxes,
  records and slots, and the VPro's stage and texture slots, come in sets
  (`Batch::next_set`). A set is reused after its `on_submitted_work_done`
  and added only when all are busy. Off Vulkan, `HostBuffer` falls back to
  a device buffer fed by the ring.
- GPU-only buffers have no MAP usage, so they land in invisible VRAM.
  Readbacks use one fixed 16 MiB MAP_READ buffer, in pieces.
- `memory_hints: Manual { 4..32 MiB }` in `Gpu::open`.
- Run GPU test suites with `--test-threads 8` or so. Every test board is
  a device with about 190 MiB of VRAM, and the desktop shares the card.

**Check.** `/sys/class/drm/card0/device/mem_info_{vram,vis_vram,gtt}_used`
while a board runs. One board is now about 190 MiB VRAM, +12 MiB visible
and about 76 MiB GTT. `dmesg | grep -c 'Not enough memory for command'`
must not grow. The SolidView GPU replay is 2.5 s with no
`__alloc_range_bias` in perf.
