# Phase 3 platform notes

## Unified config

`MachineConfig` now includes `[debug]` (formerly `[jit]`), `[perf]`, and `[machine]` TOML
sections. CLI applies `cfg.debug.apply_env()` at startup; iris-gui syncs Debug tab edits into
`cfg.debug` before Start.

## Windows CI

Default `ci_socket = "127.0.0.1:19851"` (TCP). Unix builds keep `/tmp/iris.sock`.
`iris-ci` connects to either form. Launch headless CI with `wsl/run-iris-ci.bat`.

## HAL2

Codec A output uses the shared **hptimer** (`TimerManager`) recurring timer at the
codec pitch rate (~44.1 kHz). A Phase 3 experiment with `thread::sleep` on a
dedicated pump thread caused scratchy audio on Windows — see
`rules/perf/hal2-pump-thread.md`. Codec B/AES still use hptimer as well.

## Graphics

- `Rex3Screen.fb_borrowed`: skip 16 MB copy when refresh uses heartbeat-only FB path.
- `CompositorSource.status_bar_only`: skip the DID texture upload for
  status-bar-only (heartbeat) frames. This is the only per-frame upload skip
  that remains — a separate per-pixel dirty-rectangle tracker
  (`CompositorSource.dirty_y0/y1/x0/x1`, `Rex3::note_fb_x/note_fb_y`) was
  tried and then fully removed. It traded the wrong resource: it moved cost
  onto the REX3 draw thread — the thing that actually gates emulation
  speed — adding an atomic CAS-loop to every single pixel write (interpreter
  and SIMD paths), just to save a full-frame block copy / texture upload on
  the display refresh thread, which mostly sleeps and was never the
  bottleneck. It also broke silently under JIT: the Cranelift JIT draw path
  never updated the dirty bounds, so JIT-drawn frames got the wrong dirty
  rectangle with no error. Always do full-frame GL uploads now.
- GUI: partial egui upload for status-bar-only frames (via `Frame::dirty_y/
  dirty_h`, driven by `status_bar_only`, not by the removed per-pixel
  tracker) + live VRAM borrow (`fb_borrowed`).

## JIT stores (obsolete)

This section described the old tiered MIPS JIT's `[jit] compile_stores` /
`IRIS_JIT_NO_STORES` switch and its write-log rollback (`rules/jit/store-compilation.md`).
That JIT, the setting and the rules were removed in August 2026 (`33c4e68`). jitv2
compiles stores unconditionally; see `rules/jitv2/`.
