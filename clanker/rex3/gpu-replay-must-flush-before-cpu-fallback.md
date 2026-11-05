# GPU replay must flush before CPU fallback

The replay-only `rex-gpu` path temporarily makes its wgpu RGB buffer
authoritative while a supported batch is pending. The CPU framebuffer must be
made current before any unsupported draw uses the interpreter, precompiled
shader, or Cranelift path.

The required ordering is:

1. upload the complete CPU RGB plane when the first non-empty supported draw
   enters a batch;
2. dispatch supported draws in GFIFO order;
3. before CPU fallback, or when a bounded batch fills, GFIFO becomes idle,
   `wait_idle` is called, or the processor exits, submit the batch, wait for it,
   and read the complete plane back;
4. copy readback into the CPU plane and only then publish its damage; and
5. release `GFXBUSY` only after that synchronization finishes.

This keeps the CPU framebuffer canonical at every replay observation point.
An unsupported draw never runs against stale pixels, and refresh cannot copy a
GPU-produced region before readback completes. GPU errors are retained in the
replay report so a logged failure cannot become a successful-looking replay.

The initial matcher deliberately accepts only complete positive-direction,
integer RGB `SRC` blocks and spans. A multi-row block also requires
`XSAVE == XSTART`: the CPU walker uses `XSTART` for the first row and `XSAVE`
for later rows, so any mismatch is not the rectangle the WGSL kernel executes.
Unsupported states report a stable reason and take the synchronized CPU path.

The current full-plane 8 MiB upload and 8 MiB readback are validation
boundaries, not the intended live architecture. On 2026-11-05 the RTX 5070
probe completed two overlapping fills in one submission and one fence, with
exact readback, zero fallbacks, and no wgpu error. `auto` selected Vulkan
1.4.351 through NVIDIA 617.42; the explicit D3D12 path passed the same probe
through driver 32.0.16.1742.
