# rfd on Linux dlopens libdbus: name it to quick-sharun, and watch rfd's log for "no picker"

Status: rule (2026-11-19, rfd 0.17.2; techomancer/iris#457).

**Symptom.** In the Linux AppImage every Browse button did nothing. stderr:

    rfd::backend::xdg_desktop_portal::portal::libdbus  Can't connect to a portal: libdbus-1.so not found
    rfd::backend::xdg_desktop_portal  Failed to pick file with zenity: … (os error 2)

Installing `zenity` on the host "fixed" it.

**Cause.** rfd's `xdg-portal` backend does not link libdbus; it `dlopen`s
`libdbus-1.so.3` (then `libdbus-1.so`) when a dialog opens
(`backend/xdg_desktop_portal/portal/ffi.rs`). quick-sharun's strace mode runs
the app for a few seconds and never opens a dialog, so the library was not
bundled, and sharun's loader does not search the host's library path. (The
reporter's host had a 32-bit userland, so its libdbus could not have been used
anyway.) With no portal connection rfd runs `zenity`; when that is missing too
it logs the error and returns `None`, the same as a cancel.

**Fix / rules.**

- `scripts/dist/appimage.sh` passes `/usr/lib/libdbus-1.so.3` to quick-sharun
  explicitly and fails the build if it is not in the AppDir. Any other library
  a dependency only dlopens on demand needs the same treatment.
- Every iris-gui picker goes through `filedialog::Dialog`, not
  `rfd::FileDialog`. `filedialog::LogTap` (installed by
  `filedialog::init_logging`, replacing `env_logger::init`) collects rfd's
  error records during a pick; "Failed to … with zenity" with a `None` result
  means no dialog appeared, and the window shows `modal-no-picker-*`. A cancel
  logs nothing; a portal failure zenity covered logs only the portal error.
- That detection keys on rfd 0.17's log wording. After an rfd upgrade, re-check
  `backend/xdg_desktop_portal.rs` for the zenity failure message and update
  `filedialog::no_picker_shown` and its tests.
- The standalone `iris` window (`src/ui.rs`) still calls rfd directly and has
  no notice.
