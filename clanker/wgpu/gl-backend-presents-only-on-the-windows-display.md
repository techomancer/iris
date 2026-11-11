# wgpu's GL backend presents only from an instance opened on the window's display

**Status:** fixed 2026-11-11 (`set_display` in `src/wgpu/device.rs`, called from
`src/main.rs`). Found in issue #564. Checked 2026-11-11 on an RX 6600 XT (radeonsi,
Mesa 26.1.8) with `WGPU_BACKEND=gl`, High IMPACT on IP28: "drawing the window
with wgpu", IRIX desktop up, `screen info` frame counter 0 through the boot (no
read-back), and a monitor screenshot read back in 0.3 s.

## Symptom

With `raster = "wgpu"`, `renderer = "wgpu"` and `gpu_backend = "gl"`, iris prints
"the wgpu renderer is unavailable (... cannot present to this window); drawing
with GL". Every frame then goes through `compose_gpu_into` → `Staging::read`:
composed on the GPU, read back into `screen.fb_rgb`, uploaded again by the GL
compositor. Vulkan presents fine.

## Cause

The board opens its `Gpu` while the machine is built, before the event loop
existed, with `InstanceDescriptor::new_without_display_handle()`. wgpu-hal's EGL
setup picks its platform from the instance's display handle: none means Mesa's
surfaceless platform. Surfaces made from that instance are not presentable, so
`is_surface_supported` is false and `Presenter::new` refuses the window. Vulkan,
Metal and DX12 ignore the handle.

## Rule

- An instance that may present must be created with the event loop's
  `OwnedDisplayHandle`. `main.rs` opens the event loop before building the
  machine when a window is wanted and `raster = "wgpu"`, and hands its display
  to `wgpu::set_display`. Every `Gpu::open` after that tries with the display
  first, then without it.
- Headless runs, windowless CI (`--ci` without `ci_display`) and iris-gui never
  set a display: their instances stay surfaceless, which is all compute needs.
  Their frames are read back, as before.
- winit allows one event loop per process, and a failed `EventLoop::new` uses up
  that one too. Keep the early loop's `Result` and use it for the window rather
  than creating another.
- The GL renderer is still the fallback, with a readback per frame. That path
  must keep `ApiPreference::PreferEgl` (`Ui::new`) whenever the board has a
  device: libglvnd refuses to make wgpu's EGL context current on a thread where
  glutin's GLX context is current (EGL_BAD_ACCESS), and wgpu-hal panics on that.
  iris-gui is safe: eframe's GLX-first context stays on the UI thread, the
  machine is built and draws on other threads, `HeadlessGl` is EGL, and the GUI
  does not enable `iris/wgpu` at all.
- EGL display sharing: on X11, glutin asks for the display with
  `EGL_PLATFORM_X11_SCREEN_KHR` and wgpu without attributes, so they get separate
  `EGLDisplay`s. On Wayland both ask with no attributes and share one. wgpu
  calls `eglTerminate` when its last instance on a display drops, which would
  break glutin's context. Today that happens only at exit, after the window is
  gone. Do not create and drop GL-backend instances while the GL renderer runs.
