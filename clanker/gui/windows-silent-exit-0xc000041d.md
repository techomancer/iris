# Windows: CLI vanishes with exit 0xC000041D and no message

**Keywords:** windows,winit,0.30,0.30.13,crash,STATUS_FATAL_USER_CALLBACK_EXCEPTION,0xC000041D,-1073740771,panic=abort,wndproc,window procedure,vectored exception handler,VEH,access violation,request_inner_size,SetWindowPos,WM_SIZE,re-entrant event handler,event loop,aspect ratio,issue #94,iris-crash.log,crash_diag
**Category:** gui

## Symptom

On some Windows machines the native (winit) CLI dies during startup — right
after `Rex3: Resolution changed to 1282x1024` — with exit code `0xC000041D`
(`-1073740771`, `STATUS_FATAL_USER_CALLBACK_EXCEPTION`) and **no** Rust panic
message, no backtrace, no window. `iris-gui` (eframe) is unaffected.
issue #94: ~90% of launches on one reporter's internal laptop panel, 0% on an
external monitor; unaffected by DPI scale.

## What that exit code means

`0xC000041D` is what 64-bit Windows raises when an exception escapes a
**user-mode callback invoked by the kernel** — a window procedure, hook proc,
`Enum*` callback. The kernel's callback dispatcher catches the original
exception, tears down the callback frame, and re-raises this fixed status. The
original cause (a Rust panic, or a native access violation in a driver) is gone
by the time any top-level `SetUnhandledExceptionFilter` runs.

Because the release profile is `panic = "abort"`, a Rust panic on a wndproc
stack aborts immediately — the default hook prints the message but it is easily
lost, and there is no unwind and no backtrace unless `RUST_BACKTRACE` is set.

## Getting a diagnosis

`src/crash_diag.rs` (installed first thing in `main`) exists for exactly this:

* **Panic hook** — fires even under `panic = "abort"`, before the abort. Writes
  message + thread + location + backtrace to `iris-crash.log`.
* **Vectored exception handler** (Windows) — runs *first-chance*, before the
  callback dispatcher swallows anything. For access violations / illegal /
  privileged instructions / stack overflow / heap corruption / the
  fatal-callback status it logs the exception code, faulting address, and the
  stack **twice**: a raw `module+offset` list (no heap — always makes it out),
  then a dbghelp-symbolised list (`function+off  at file:line`) resolved
  against `iris.pdb`. dbghelp's default search does not reliably include the
  exe's own directory, so `crash_diag` passes it explicitly to
  `SymInitializeW` — without that, every iris frame silently stays unresolved.

Ask a reporter to reproduce once and attach `iris-crash.log` — it is
self-symbolising as long as `iris.pdb` sits next to `iris.exe` (the release
profile emits it; ship them together). If only the raw list resolved, feed the
`iris.exe+0xNNNN` RVAs to `cdb -z iris.exe -c "ln iris+0xNNNN;q"` or
`llvm-symbolizer --obj=iris.exe --relative-address` — **not** `addr2line`, MSVC
PDBs are not DWARF.

Self-test the wiring: `IRIS_CRASH_SELFTEST=panic|thread|segv target/release/iris.exe`.
Escape hatch if the VEH ever gets noisy: `IRIS_CRASH_DIAG=off` (panic hook stays).

## What is NOT the cause (verified — don't re-investigate)

The obvious theory was re-entrancy: `WindowEvent::Resized`'s aspect-ratio lock
calls `Window::request_inner_size` from *inside* the winit event callback; on
Windows that reaches `SetWindowPos` synchronously, which re-enters the wndproc
with `WM_SIZE`.

* The **REX3-refresh-thread** `request_inner_size` in `GlRenderer::resize` is
  *not* it — cross-thread `SetWindowPos` uses `SWP_ASYNCWINDOWPOS`, so the
  resize is posted, never re-entrant. (Commit c2e085a's rationale is imprecise
  on this point.)
* The **event-thread** re-entrant `request_inner_size` is buffered, not
  crashed: vendored winit 0.30.13's `EventLoopRunner::should_buffer()` detects
  the taken handler and defers the nested `WindowEvent`. Forcing an *infinite*
  re-entrant `request_inner_size` from the `Resized` handler just ping-pongs
  the window ±1px forever without panicking.
* Could not reproduce on a desktop (single 2560×1440 @ 96 DPI) across ~130
  launches: delayed cross-thread resize, spammed resize, forced work-area
  clamp, forced infinite re-entrancy — all 0 crashes.
* The fix commits (c2e085a + 81c34ba) do **not** stop the reporter's crash.

## Upstream winit status

Not fixed. 0.30.13 is the last 0.30.x. `v0.31.0-beta.3` keeps the identical
`call_event_handler` re-entrancy assert (`"either event handler is re-entrant
(likely)…"`) and the identical `send_event` path that dispatches
`RedrawRequested` **directly, bypassing `should_buffer`** — the one genuine
re-entrancy hole. No changelog entry addresses Windows re-entrancy from
`request_inner_size`/`WM_SIZE`. Upstream's pattern for this class is deferral:
the `pending_drag` / `source_drag` fields were added so the blocking
`DoDragDrop` runs only after the app returns control to winit. Any iris fix
should follow suit — never call a synchronous window-mutating method from
inside a winit callback; queue it and apply it from outside dispatch.

## Confirmed cause (from a reporter's `iris-crash.log`)

    code       : 0xC0000005  ACCESS_VIOLATION
    access     : read @ 0x0000000000000000
    fault addr : atio6axx.dll+0x87DF84
    …
      atio6axx.dll+0x876AD3
      USER32.dll  (x3)                 ← window-procedure / hook thunk
      ntdll.dll  KiUserCallbackDispatcher
      win32u.dll                        ← NtUser… syscall
      iris.exe   (winit → a windowing call)
      … iris::main   (MAIN THREAD)

`atio6axx.dll` is the **AMD/ATI 64-bit OpenGL ICD**. It null-derefs on a
**read**, on the **main (event-loop) thread**, inside a **window procedure** —
the `win32u → KiUserCallbackDispatcher → USER32 → wndproc` sequence is Windows
running the GL window's proc *synchronously* as part of a winit windowing call
(`SetWindowPos`, `SetPixelFormat`, window show/create — symbolise the reporter's
log to see which).

OpenGL on Windows does **not** require the context to be created or made current
on the window-owning thread (WGL contexts move between threads freely; only
"current on one thread at a time" applies), and iris keeps all GL *calls* on the
REX3 refresh thread. But the AMD ICD **subclasses the GL window** and its
wndproc hook runs on whatever thread owns the window — the main thread — while
iris does `SetPixelFormat` + the first `wglMakeCurrent` (via glutin's
`create_window_surface` / `make_current` in `GlRenderer::ensure_init`) on the
**REX3 thread**. A window message reaching the AMD hook before that per-HWND GL
state is populated → null deref. AMD's ICD is historically the worst offender
for this cross-thread setup race.

## Fix

`Ui::new()` (main thread, before `Ui::run` starts pumping messages) now builds
the window `Surface` and binds the context to it once — `make_current` then
`make_not_current` — and hands both the `NotCurrentContext` and the `Surface`
to `GlRenderer` (`initial_surface: Option<Surface<WindowSurface>>`). That first
`make_current` is what installs the ICD's window subclass and populates its
per-HWND state, so by the time the event loop runs and a resize can reach the
subclass, the driver state exists.

`ensure_init()` on the REX3 thread consumes `initial_surface` on the true first
frame and only `make_current`s it on its own thread — the "move a context
between threads" handoff, which WGL supports. Rendering stays entirely on REX3;
only the one-time pixel-format + initial bind moved.

After a `stop()`/`start()` cycle (jitcheck checkpoint restore, `reset`,
snapshot load) `initial_surface` is already `None` — `ensure_init` then makes a
fresh surface on the refresh thread, as before. Safe by then: the pixel format
is already set on the HWND and the driver's window state was established at
startup, so a resize hitting the subclass is a read of valid state, not a null
deref.

macOS/Linux: unaffected or improved. The macOS `window_handle()` main-thread
constraint was already satisfied (handle captured in `Ui::new` —
`clanker/macos/winit-030-window-handle-main-thread-only.md`); Linux/GLX moving
`create_window_surface` + first `make_current` onto the window-creating thread
lines up with the proprietary-NVIDIA concern noted on the
`not_current_context` field.
