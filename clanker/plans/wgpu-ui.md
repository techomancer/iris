# The host window on wgpu (iris `ui.rs`)

Status: steps 1-4 done 2026-11-07. Left: the GL renderer and
`gl_compositor.rs` stay as the default until the wgpu window has run on
macOS and Windows; then the default can move and GL become optional. The iris-gui front end
stays on its software compose and readback; this is the plain `iris`
window only.

Measured live (V12, GPU raster unit, `--renderer wgpu`): 9106 frames from
power-on to the desktop composed and presented with 0 bytes read back; a
`screen shot` reads back exactly one frame. The display thread used 2302
CPU ticks to the desktop against 7209 with the GL window.

## Why

The VPro composes its frame on the GPU (`VproScanout`), reads it back,
`Display::present` byte-swaps it into `Rex3Screen.rgba`, and the GL
renderer uploads it again as a texture: two copies across the bus per
frame for an image that started on the GPU. On one wgpu device the frame
stays there. The Newport GL compositor (`gl_compositor.rs`) can then move to
WGSL too, and iris stops needing GL at all.

## Shape

- **Common, `src/wgpu/present.rs`.** `Presenter`: a `wgpu::Surface` on the
  winit window, its configuration (resized on the event thread's request,
  recreated on `Outdated`/`Lost`), and one render pipeline that draws a
  frame quad letterboxed into the window: nearest sampling at integer
  scales, linear otherwise (what the GL path's integer and fallback shaders
  do), then the debug overlay (alpha blended) and the status bar under it.
  Frame sources:
  - `Cpu { rgba, width, height }`: `queue.write_texture` of the visible rows
    (Newport's software compose, GR2/IMPACT prebuilt frames, VPro on the
    CPU unit).
  - `Gpu { buffer, width, height }`: a storage buffer of 0x00RRGGBB words
    already on the device (the VPro scan-out's output), copied into the
    frame texture by a small compute pass or `copy_buffer_to_texture`; no
    readback.
- **Shared device.** `Gpu` keeps its `Instance` and `Adapter`. On Vulkan,
  Metal and DX12 the instance needs no display handle, so the window's
  surface can come from the board's instance; `adapter.is_surface_supported`
  decides. The board's `Gpu` reaches the UI through a new
  `GfxDisplay::gpu()` (None for boards without one); otherwise the UI opens
  its own. GLES (the `gl` backend) would need the event loop's
  `OwnedDisplayHandle` in the instance: not supported at first.
- **Renderer trait.** `WgpuRenderer` implements `Renderer` (the board
  display threads keep calling `present`). A new default method,
  `present_gpu_frame(buffer, w, h, overlay, status, stats) -> bool`, lets the
  VPro display skip its scan-out readback when the installed renderer takes
  the GPU frame; screenshots and VNC (`need_readback`, `set_capture`) still
  read back, on request only.
- **Overlay and status bar.** Both draw into CPU RGBA today (`DebugOverlay`,
  `StatusBar` + `StatusBarTexture` for GL). The wgpu path uploads their
  pixels as textures; `StatusBarTexture` gains a CPU form.
- **Choice.** `[ui] renderer = "gl" | "wgpu"`, CLI `--renderer`; default
  stays `gl` until the wgpu path has run on Linux, macOS and Windows.
  Threading follows the GL path's rules (`clanker/gui/`): the surface is
  created on the main thread in `Ui::new`, configured and presented on the
  board's display thread, and window-mutating calls stay on the event
  thread (`ResizeRequest`).

## Steps

1. `Gpu` keeps instance and adapter; `GfxDisplay::gpu()`; VPro returns its
   unit's `Gpu`.
2. `src/wgpu/present.rs` with the CPU frame source; `WgpuRenderer` in
   `ui.rs` behind `[ui] renderer`; every board runs on it (CPU frames).
3. GPU frame source; the VPro display presents its scan-out buffer without
   a readback when the renderer supports it.
4. Newport's compositor in WGSL (`gl_compositor.rs`'s shader, as a
   `ScanoutShaders` kernel over the REX3 planes uploaded per frame), and
   the GL renderer becomes optional. Done as `src/dev/ng1/wgpu`
   (`Ng1Scanout`), matching `SwCompositor` on every pixel of 12 random
   frames (`the_gpu_compose_matches_the_software_compositor`); seen live on
   an Indy's PROM. The planes go up whole each frame (16 MB); REX3's damage
   rectangles (`rex-dirty-upload`) could cut that, as the GL path does.
