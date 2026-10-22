# macOS Cmd+Q bypasses the close-intercept — fold CHD overlays on power-off, not just quit

Status: confirmed fix (2026-06-20). The CHD "Synchronizing disks…" fold wasn't
happening when a user cleanly shut IRIX down and then quit the app **without
pressing STOP**.

## Root cause

The exit-time fold runs from the close-intercept in `App::update`:

```rust
if ctx.input(|i| i.viewport().close_requested()) {
    if !self.sync_then_close && self.emu.has_pending_chd_sync() { /* SyncDisks + CancelClose */ }
}
```

That only fires on `WindowEvent::CloseRequested`. **winit 0.29.15 (what eframe
0.29 pulls in) does not hook `applicationShouldTerminate`** — grep its
`platform_impl/macos/` for it: zero hits. So on macOS the **app-menu Quit / Cmd+Q**
calls `[NSApp terminate:]`, which exits the process *without* emitting a window
CloseRequested. The event loop never runs the intercept → no fold. (The red
close button and the in-app File » Quit *do* work — File » Quit sends
`ViewportCommand::Close`, which becomes a CloseRequested.)

Diagnosis shortcut that pinned it: the **STOP button** fold (`request_stop` →
`has_pending_chd_sync` → `SyncDisks`) *did* consolidate, which proves
`chd_sync_pending` is true in the shutdown state. So the fold logic was fine; only
the quit path was being missed.

## Fix

Fold at **power-off time**, not quit time — keyed off **`cpu_stopped`**, NOT
`Evt::PowerOff`. Gotcha: `Evt::PowerOff` is declared but **never emitted** (see
the comment at `handle.rs:43` — it awaits a core `subscribe_events` API), so a
first attempt that hooked it was dead code and did nothing. The signal that
actually moves is the status field `cpu_stopped` (the same one that draws the
"Powered off" overlay):

- A guest `poweroff` writes the IOC front-panel power register (`ioc.rs:575`) →
  `MachineEvent::PowerOff` → the core dispatch thread calls `machine.stop()`
  (iris-gui sets `IRIS_NO_EXIT_ON_POWEROFF=1` so the process survives). The CPU
  thread stops → the status tick reports `cpu_stopped = true`.
- The worker is NOT told (only a user STOP raises `Evt::Stopped`), so the GUI's
  `running` stays true. Thus **`running && cpu_stopped`** uniquely means "the
  guest shut itself down" (a user STOP sets `running=false` and folds via
  `request_stop` anyway).

So in `framebuffer_panel`, edge-trigger on `cpu_stopped` (a `prev_cpu_stopped`
field, reset to false at Start): when `cpu_stopped && !prev && is_running() &&
has_pending_chd_sync() && syncing.is_none()`, kick off `Cmd::SyncDisks` with the
"Synchronizing disks…" modal. By the time the user quits — however they quit —
the disk is already consolidated. COW-mode disks are unaffected (`pending_sync`
returns `None` when `cow`).

The status loop keeps ticking after the core's `machine.stop()` because the
worker's `cycles` is only cleared on `Cmd::Stop`/`SyncDisks`, not on a guest
power-off — so `cpu_stopped` and `chd_sync_pending` stay live and accurate.

## Residual gap (not fixed)

Quitting via **Cmd+Q while the guest is still running** (or halted at the PROM
without a power-off) still bypasses the fold, because that path never reaches the
close-intercept and never raises `Evt::PowerOff`. Quitting mid-run is already
discouraged (see `dont-run-iris-long`). The complete fix would be an
`NSApplicationDelegate applicationShouldTerminate:` override (via objc2, as in
`macos_sandbox.rs`) that defers termination until the fold completes — deferred
because winit owns the app lifecycle and the interop is fiddly.
