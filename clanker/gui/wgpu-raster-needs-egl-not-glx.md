# A board drawing on the host GPU needs the window's GL to be EGL, not GLX

**Keywords:** opengl,gl,glx,egl,libglvnd,glutin,wgpu,wgpu-hal,gles,eglMakeCurrent,badaccess,thread,refresh,mgras,impact,raster,panic
**Category:** gui

## Symptom

IMPACT with `--raster wgpu` panics seconds into boot, on the **MGRAS-Refresh**
thread:

```
thread : MGRAS-Refresh
location: .../wgpu-hal-30.0.1/src/gles/egl.rs:184:14
message: called `Result::unwrap()` on an `Err` value: BadAccess
```

`egl.rs:184` is `EglContext::make_current`'s `.unwrap()` over
`EglInstance::make_current`, i.e. `eglMakeCurrent` returned `EGL_FALSE` with
`eglGetError() == EGL_BAD_ACCESS`. It is not a race, not a lost surface, and not
a wgpu-hal deadlock: the same `eglMakeCurrent` calls succeed hundreds of times
before and the failure is deterministic once the window's GL context has been
made current.

## Cause: libglvnd tags a thread's current context per window system

`Ui::new` builds the windowed GL context through
`glutin_winit::DisplayBuilder`, whose `ApiPreference` default is `FallbackEgl`
— on X11 that is `DisplayApiPreference::GlxThenEgl`, so **GLX wins** whenever
GLX works. `GlRenderer::ensure_init` then makes that context current **on the
refresh thread** (`refresh_loop` → `present()`), because `GlRenderer` is `Send`ed
there and the context's "current" binding is per-OS-thread.

libglvnd's `eglMakeCurrent` (`libegl.c`, `__glDispatchGetCurrentThreadState`)
refuses to make an EGL context current while the thread's current context was
made current through a different window system:

```c
glas = __glDispatchGetCurrentThreadState();
if (glas != NULL) {
    if (glas->tag != GLDISPATCH_API_EGL) {
        // Another API (probably GLX) already has a current context.
        __eglReportError(EGL_BAD_ACCESS, "eglMakeCurrent", NULL,
                "Another window API already has a current context");
        return EGL_FALSE;
    }
```

`tag` is `GLDISPATCH_API_GLX` (0) or `GLDISPATCH_API_EGL` (1); only libGLX writes
the former. So a GLX-current thread cannot make *any* EGL context current, and
wgpu-hal's GLES backend is pure EGL.

The conflict is structural, not a wgpu-hal locking bug: `AdapterContext::lock()`
takes the glow mutex and calls `make_current()` on the *same* thread that
`present()` made the window's GLX context current on. Two threads racing on the
glow mutex is a separate, pre-existing concern; the panic happens with no other
thread inside wgpu at all.

## Why only `raster = "wgpu"`

`GfxDisplay::gpu()` is `Some` only for a board with a host-GPU raster unit, and
`Mgras::with_raster` shares that one `Arc<Gpu>` with the window renderer
(`ui.rs: new_wgpu`). Without it the board composes on the CPU, wgpu-hal is never
entered from the refresh thread, and GLX stays perfectly happy. The plain
interpreter/CPU path never panics.

## Fix

Ask glutin for EGL when the board draws on the host GPU — the window system
wgpu-hal's GLES backend already uses, so the two contexts take turns being
current on the one thread, as EGL allows:

```rust
#[cfg(all(feature = "wgpu", target_os = "linux"))]
if display.gpu().is_some() {
    display_builder = display_builder.with_preference(ApiPreference::PreferEgl);
}
```

Gated to Linux: the conflict is GLVND's GLX/EGL tag, elsewhere glutin ignores the
preference (CGL on macOS) or there is no shared dispatch layer (WGL on Windows).
`PreferEgl` maps to `EglThenGlx`, so a machine with no usable EGL still gets the
old GLX behaviour rather than a hard failure.

`src/headless_gl.rs` already pins `DisplayApiPreference::Egl` on Linux; only the
windowed path in `ui.rs` was left on glutin's GLX-first default.

## Rule

If a board's refresh thread enters wgpu's GLES backend (any `--raster wgpu`
board), the window's GL context must be EGL on Linux. Never let glutin pick GLX
on that path.
