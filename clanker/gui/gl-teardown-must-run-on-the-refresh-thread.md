# GL teardown must run on the REX3-Refresh thread

**Keywords:** opengl,gl,glow,glutin,glDeleteTextures,thread,context,make_current,refresh,poweroff,shutdown,crash,segfault,compositor,destroy,rex3,renderer
**Category:** gui

## All OpenGL calls — including teardown — belong to the thread that owns the context

The GL context is created and made current lazily by `GlRenderer::init_gl`, which
runs from `present()` — and `present()` is only ever called from `Rex3::refresh_loop`
(the **REX3-Refresh** thread). That thread is the sole owner of the context; this is
documented by the `unsafe impl Send for GlRenderer` comment ("sent to the refresh
thread where it owns and uses the GL context. No other thread touches these fields").

On macOS, GL functions dispatch through the **calling thread's** current context.
A GL call from a thread with no current context dereferences a null current-context
struct and faults at a small offset (the observed crash was `glDeleteTextures` with
`KERN_INVALID_ADDRESS at 0x1e0` — a null + field offset, not a freed pointer).

## The bug this rule encodes

`Rex3::stop()` used to:
1. set `running = false`, join the REX3-Processor thread,
2. **join the REX3-Refresh thread** (the context owner is now gone), then
3. call `renderer.stop()` → `Compositor::destroy(&state.gl)` → `glDeleteTextures`
   **on whatever thread called `Rex3::stop()`** — never the refresh thread.

`Rex3::stop()` is reached from many non-owning threads: the `machine-events` thread
(guest soft power-off → `MachineEvent::PowerOff` → `Machine::stop`), the **main**
thread (`main.rs` after the winit loop returns on window close), and the monitor/CI
thread (`reset`, snapshot load/save). The GUI power-off and window-close paths have a
live context and crash; the headless/CI paths have `state == None` so teardown is a
silent no-op, which is why it went unnoticed.

## The fix

Tear the renderer down at the **end of `refresh_loop`**, right after the
`while self.running` loop exits — i.e. on the owning thread, while the context is
still current — and drop the `renderer.stop()` call from `Rex3::stop()`. Ordering is
preserved because `Rex3::stop()` joins the refresh thread, so teardown still finishes
before `stop()` returns. After a `reset`, `restart_peripherals()` calls
`rex3.start()`, which respawns the refresh thread; `present()` then re-inits GL lazily.

## Second trigger, same root cause — FIXED

`disp compositor <gl|sw>` → `GlRenderer::switch_compositor` (ui.rs) used to call
`compositor.destroy(&state.gl)` directly on the **monitor/CI socket** thread
(commands are dispatched per-connection, see `ci.rs`). Same foreign-thread GL
teardown, same fault; rarely triggered because it needs a manual monitor command.

Fixed as this file prescribed: `switch_compositor` now only records the request in
`pending_compositor: Arc<Mutex<Option<bool>>>` and returns the name the swap will
settle on. `present()` services it via `apply_pending_compositor()` at the top of the
frame — on the refresh thread, with the context current — before taking its `state`
borrow. `switch_compositor` also now clamps `use_gl` against `gl_tier`, so asking for
`gl` on a Legacy (GL 2.1) context correctly reports `sw` instead of claiming a
compositor that `GlCompositor::new()` can't back.

## Rule of thumb

Never call any `gl.*` / `glow` method (create, upload, draw, or delete) from outside
`refresh_loop`/`present`. If another thread needs GL work done, signal the refresh
thread and let it do it.
