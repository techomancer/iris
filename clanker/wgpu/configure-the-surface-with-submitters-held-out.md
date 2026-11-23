# Configure the surface with every submitter held out

Status: rule, fixed 2026-11-22 (`Gpu::configure_surface`).

## Symptom

On a Newport resolution change under the GPU raster and the wgpu renderer:

```
Rex3: Resolution changed to 1282x1024 cursor_x_adjust=5
wgpu error: Validation Error
  In Surface::configure
    Failed to wait for GPU to come idle before reconfiguring the Surface
```

## Cause

wgpu-core's `surface_configure` waits for the queue to drain and then
requires it to be empty; if another thread submitted during the wait it
fails with `GpuWaitTimeout` (a Validation error, despite the name). The
board's raster thread submits on the same `Gpu` all the time. The error was
uncaptured, so it stuck in `Gpu::check`, and every user of the device
after it (the presenter and the resident raster) reported it as a failure.

## Rule

Configure surfaces only through `Gpu::configure_surface`. It holds the
uploads lock, which every `queue.submit` takes (`submit_locked`), and
catches the error in an error scope so it belongs to the surface, not the
device. A presenter whose configure failed tries again on the next frame.
Every submission must keep going through `Gpu::submit`: a direct
`queue.submit` reopens the race. See also
[drop-the-surface-texture-before-configure.md](drop-the-surface-texture-before-configure.md).
