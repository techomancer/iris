# Window methods must only be called on the event-loop thread (Windows crash)

## Symptom

`iris` (the winit CLI window, not `iris-gui`) died during startup on Windows on
roughly 90% of launches, with **no Rust panic message and no backtrace**. Exit
code `-1073740771` = `0xC000041D` = `STATUS_FATAL_USER_CALLBACK_EXCEPTION`.

The last line printed was always:

```
Rex3: Resolution changed to 1282x1024 cursor_x_adjust=5
```

Reported as "internal laptop panel only, external monitor is 100% stable", at
both 100% and 150% display scaling. That framing is misleading — it is a plain
race, and the panel only changes the timing (see below).

## Root cause

`GlRenderer::resize()` runs on the **REX3-Refresh thread** (`rex3.rs`, the
`resized` branch of the refresh loop). It called
`window.request_inner_size(...)`.

On Windows that reaches `SetWindowPos`. `SetWindowPos` against a window owned by
*another* thread does not simply post a message: Win32 dispatches
`WM_WINDOWPOSCHANGING` / `WM_WINDOWPOSCHANGED` / `WM_SIZE` **synchronously into
the owning thread's wndproc**. (`SWP_ASYNCWINDOWPOS` only affects when the
*caller* returns; it does not make callback delivery async.)

So the main thread re-enters `public_window_callback` → `WM_SIZE` →
`send_event(Resized)` → `EventLoopRunner::send_event`. And winit's Windows
`EventLoopRunner` keeps its state in **`Cell` and `RefCell`, with no
synchronisation at all**:

```rust
event_handler: Cell<Option<Box<dyn FnMut(Event<T>)>>>,
event_buffer:  RefCell<VecDeque<BufferedEvent<T>>>,
```

winit's whole Windows backend assumes every window operation happens on the
event-loop thread — that is why upstream `Window` is `!Send`/`!Sync`. iris gets
around that with `Arc<Window>` plus a hand-written `unsafe impl Send` on
`GlRenderer`, which is what let the bug exist.

Three separate ways it dies, all producing 0xC000041D because a Rust panic
unwinding out of a Win32 callback is a fatal user callback exception:

1. `should_buffer()` does a non-atomic `event_handler.take()` … `.set(handler)`.
   Interleave the two threads and `call_event_handler` hits
   `.expect("either event handler is re-entrant (likely), or no event handler
   is registered")`.
2. `event_buffer.borrow_mut()` from both threads → `RefCell` "already borrowed".
3. Startup ordering: `main.rs` calls `machine.start()` (spawns REX3-Refresh)
   well before `ui.run(event_loop)`. In that gap `runner_state` is still
   `Uninitialized` and no handler is registered; `move_state_to` also has
   reachable `panic!("cannot move state to Uninitialized")`.

## Why "laptop panel only" is a red herring

- The `Resolution changed` line is the *last* message because that `resized`
  flag is precisely what triggered the offending call.
- 1282 (not 1280) comes out of `decode_video_timings`, so the initial 1280-wide
  window **always** mismatches and the resize fires on every single boot.
- 1282x1040 does not fit in a 1080p work area, so Windows clamps it and emits an
  extra `WM_GETMINMAXINFO` plus a second `WM_SIZE` — more cross-thread callback
  traffic, wider race window. The internal panel is also the one carrying the
  taskbar work-area and the iGPU's DWM composition.
- 150% scaling adds `WM_DPICHANGED` as a third source of synchronous callbacks.
- `iris-gui` (eframe) was never affected: it only touches the window from the
  event-loop thread.
- Linux/X11 never crashed because winit's X11 backend queues requests over a
  thread-safe connection instead of synchronously invoking a callback.

## The rule

**Never call a `Window` method that mutates window state from any thread other
than the one running the event loop.** That includes `request_inner_size`,
`set_fullscreen`, `set_cursor_grab`, `set_window_icon`, …

Read-only queries (`inner_size()` → `GetClientRect`, `fullscreen()`,
`is_maximized()`) are fine cross-thread and are still used by the renderer.

**Also do not call them from inside an event callback.** Even on the correct
thread, `request_inner_size` re-enters the wndproc with `WM_SIZE` while winit is
mid-dispatch. The old aspect-ratio lock in `WindowEvent::Resized` did exactly
this and was the second half of the hazard.

## The pattern used instead (`src/ui.rs`)

Both resize sources only *record* what they want:

- `GlRenderer::resize()` (REX3 thread) sets `resize_request` and pokes
  `EventLoopProxy::send_event(())`. The proxy is the right primitive here: it is
  just `PostMessageW`, which is genuinely async, thread-safe, and queues fine
  even if posted before the loop starts running.
- The RCtrl+1 / RCtrl+2 hotkey and the aspect-ratio lock (both already on the
  event thread, but inside a callback) also just set `resize_request`.

`UiApp::about_to_wait` is the single place that actually calls
`request_inner_size` — winit has finished dispatching the current message batch
by then, so there is nothing to re-enter. It also coalesces a burst of `Resized`
events during a mouse drag into one correction.

RCtrl+F11 (borderless fullscreen) goes through the same queue as
`ResizeRequest::ToggleFullscreen`. `set_fullscreen` is a whole style + geometry
change — WM_WINDOWPOSCHANGING, WM_SIZE, and WM_DPICHANGED if the window lands on
a monitor with a different scale factor — so calling it mid-callback is the same
re-entrancy hazard, just on the correct thread. `about_to_wait` handles that
variant *before* the "don't resize while fullscreen" guard, since that guard
would otherwise swallow every request to leave fullscreen.

`resize_request` is a single slot, so a burst coalesces to the last request.
That is the intent for a resize drag (many `Resized` events, one correction).
The one lossy case — an F11 landing in the same frame as a guest mode change,
dropping that mode change's one-time 1x snap — is cosmetic: `display_res` is
published separately by `resize()` and always survives, so the aspect lock never
sees stale geometry, and the renderer letterboxes correctly at whatever size the
window happens to have.

Note the direction of travel: `window_size` flows event-thread → render-thread,
`resize_request` flows render-thread → event-thread. Keep them separate.

The waker matters: with `ControlFlow::Wait`, an idle guest can generate no OS
events for a long time, so without `send_event` the queued resize would sit
unapplied until the user happened to move the mouse.

## Verifying

On Linux, run the binary windowed and check the geometry actually landed:

```
./target/release/iris &
xwininfo -root -tree | grep "Irresponsible Rust IRIX Simulator"
#  ... 1282x1040+1+18     <- 1282 x (1024 + 16 status bar). Request crossed
#                            threads and was applied.
```

Seeing the `Resolution changed to 1282x1024` line followed by continued
execution is the regression check; that line used to be the last thing printed.

## Corollary: native modal dialogs must not run on the event thread either

`rfd::FileDialog::pick_file()` (RCtrl+F12, CD-ROM hot-swap) used to be called
inline inside `window_event`. A native file dialog pumps its **own modal message
loop**, which on Windows re-enters our window procedure while winit is still
inside the event callback — the same `Cell`/`RefCell` exposure as above, and the
REX3 thread is concurrently queueing resize requests into that same runner.

It also froze emulation for as long as the dialog stayed open, since the event
thread never returned to winit.

Fixed by spawning a short-lived `cdrom-picker` thread that opens the dialog and
calls `Wd33c93a::load_disc` (which takes `&self` and locks internally, so it is
safe from any thread). Off-thread the dialog can neither re-enter the event loop
nor stall the guest.

The general shape: **anything that pumps a nested message loop — native file
dialogs, message boxes, drag-and-drop tracking, `MessageBox`, COM modal UI —
must not be invoked from inside a winit event callback.**
