# winit 0.30 macOS: window_handle() only works on the main thread

**Keywords:** winit,0.30,0.29,macos,appkit,raw_window_handle,window_handle,HandleError,Unavailable,build_surface_attributes,glutin,glutin-winit,surface,init_gl,refresh,thread,panic,abort,MainThreadMarker
**Category:** macos

## Symptom

CLI panics on the first presented frame — right after the guest programs a video
mode (`Rex3: Resolution changed to ...`), because `present()` is gated on
`screen.width > 0`:

    thread 'REX3-Refresh' panicked at src/ui.rs:
    surface attributes: Unavailable

## Cause

The winit 0.29 → 0.30 bump changed macOS behavior. In 0.29,
`raw_window_handle()` returned the NSView pointer from any thread. In 0.30 the
window delegate is `MainThreadBound` and both `window_handle()` /
`raw_window_handle()` return `Err(HandleError::Unavailable)` off the main
thread (`platform_impl/macos/window.rs`, gated on `MainThreadMarker::new()`).

`glutin_winit::GlWindow::build_surface_attributes` calls `window_handle()`
internally, so it must not be called from the REX3-Refresh thread — but that is
exactly where `GlRenderer::init_gl()` runs (lazily, from the first `present()`).

## Rule

Capture `RawWindowHandle` **once on the main thread** in `Ui::new()` (same place
the GL context is created — see the `not_current_context` field comment), stash
it in `GlRenderer`, and build surface attributes in `init_gl()` with
`SurfaceAttributesBuilder::build(cached_handle, w, h)` from
`window.inner_size()`. Never call `build_surface_attributes()` /
`window_handle()` off the main thread on macOS.

glutin 0.32's CGL `create_window_surface` only reads the handle out of the
attrs (no main-thread check of its own), so surface creation itself may stay on
the refresh thread — consistent with the GL-ownership rule in
`rules/gui/gl-teardown-must-run-on-the-refresh-thread.md`.

`headless_gl.rs` intentionally keeps the `.ok()?` pattern: it creates its own
`EventLoop` on the calling thread, which already fails gracefully off-main on
macOS — it degrades to `None` instead of crashing. Don't "fix" it the same way.
