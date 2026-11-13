# GPU replay must flush before CPU fallback

**Scope:** `rex-gpu`, both recording replay and the experimental live renderer
(`IRIS_REX3_GPU`). Status 2026-11-10: tile-binned shader, tile-major GPU plane,
bounded recovery journal.

While a supported batch is outstanding, the wgpu RGB buffer is authoritative
for the tiles it has drawn. The CPU framebuffer must be made current before
anything reads or writes it outside the GPU path.

The required ordering is:

1. upload the CPU RGB plane (screen area only, tile-major) when the first
   non-empty supported draw arrives after the GPU copy was invalidated;
2. bin supported draws into 32×32 tiles in GFIFO order and submit full
   batches without waiting;
3. before CPU fallback, GFIFO idle/exit, display fences, `wait_idle`, any other
   CPU write to the plane (snapshot restore, `power_on`, resolution fill,
   framebuffer load), or accepting a draw that would exceed 65,536 journal
   entries or 4,194,304 replay pixels, submit and read back every tile damaged
   since the last checkpoint (the rest of the GPU plane still equals the CPU
   plane);
4. copy readback into the CPU plane before clearing the journal or publishing
   damage, and release `GFXBUSY`/fences only after synchronization;
5. after any CPU write, mark the GPU copy stale so the next draw re-uploads;
6. on any GPU failure (wgpu error, device loss, map failure, 10 s readback
   timeout) replay the journal's pixel effects onto the CPU plane, which still
   holds the last checkpoint, publish the merged damage, disable the engine
   and leak its wgpu objects.

Step 6 does not replay guest contexts: the context was advanced when the draw
was accepted, so only pixels are reapplied. Leaking is deliberate: with stock
wgpu 30, destroying a lost device or its queue can panic or block, and the CPU
plane is already authoritative.

The matcher accepts only complete positive-direction, integer RGB `SRC` blocks
and spans. A multi-row block also requires `XSAVE == XSTART`: the CPU walker
uses `XSTART` for the first row and `XSAVE` for later rows. A batch token
(`host_len > 1`) is refused even when the draw reads no host data, because the
CPU path resumes the primitive until the batch cursor stops moving. Keep the
complete `WRMASK` and clamped `COLORALPHA`: high-only and mixed masks write
upper bits of the CPU framebuffer word even in RGB24 mode. An accepted draw
also retires the data port (`host_len = 1`, `host_cursor = 0`) as a CPU GO
does.

Batch parameters and tile bins live in fixed buffers bound once. Queue writes
for a later submission execute after earlier submissions, so reuse is ordered.
The bin offsets, rewritten every batch, are the only thing that selects fills,
so stale entries past the current batch in the parameter buffer are never
read. Each 32×32 tile belongs to one workgroup and each pixel to one
invocation, which applies the tile's fills in submission order; no barrier is
involved.

Nonblocking polls use `Instance::poll_all(false)`: in wgpu-core 30.0.1 that
routes HAL failures to the device-lost callback instead of the fatal path of a
blocking `Device::poll`.

On 2026-11-05 the RTX 5070 probe completed two overlapping fills in one
submission and one fence, with exact readback, zero fallbacks, and no wgpu
error. `auto` selected Vulkan 1.4.351 through NVIDIA 617.42; the explicit
D3D12 path passed the same probe through driver 32.0.16.1742.

What this lean design does not do (hardened in the follow-up, see
[the follow-up plan](../plans/rex3-gpu-live-follow-ups.md)): bound machine stop
by a deadline while a readback is outstanding, retire a backend on a worker
thread, or keep a process-wide GPU lease so a failed device cannot be
re-created.
