# Native macOS front-end (`--features macos-gui`)

An optional second layout for iris-gui on macOS: system menu bar, a separate OS
window for every dialog and for the configuration editor, and the status in the
window title. It lives in `iris-gui/src/macos_native/`. The default sidebar
layout is unchanged and is still what every build gets unless the feature is
on.

## Keep it opt-in

- `build.rs` turns the feature into `cfg(native_mac)`, and only on macOS.
  Enabling it elsewhere does nothing, so `--all-features` still builds on Linux
  and Windows. Gate code on `native_mac`, not on the feature name.
- `main.rs` knows about the backend only through a few `cfg(native_mac)` hooks:
  the `egui` import, a `native` field on `App`, `before_launch()` in `main`,
  `native_frame()` in place of the side panels, and `native_windows()` after the
  central panel. The sidebar methods are `allow(dead_code)` in the native build.
  Keep new backend code in `macos_native/` and don't turn the sidebar code into
  a shared abstraction.

## Dialogs become OS windows through a swapped `egui::Window`

In the native build, `main.rs` and `dialogs/*.rs` import
`crate::macos_native::egui`. That is `eframe::egui::*` with `Window` replaced by
`macos_native::window::Window`, which opens an immediate viewport. So the dialog
code is shared unchanged.

- The stand-in implements only the builder methods the call sites use (`new`,
  `open`, `resizable`, `default_width`/`height`, `collapsible`, `anchor`,
  `show`). If a dialog adds another `egui::Window` method, the native build
  fails to compile until the method is added there too. Check both builds.
- A new file that draws an `egui::Window` needs the same cfg'd import, or its
  window stays inside the main window in the native build.
- **Don't nest them.** Showing one of these windows from inside another one's
  body is not supported. The config editor (itself an OS window) therefore only
  *sets flags* (`net_sanity_modal`, `confirm_embedded_prom`), and `App::ui`
  draws those modals at the top level.
- Non-resizable windows size themselves: the content goes in an unconstrained
  `egui::Area`, its size is measured, and the builder's `inner_size` follows it,
  so egui resizes the OS window. A window with `.open(..)` gets a close button
  that clears the flag. One without it has no close button, like the in-window
  modal it replaces.
- A window is centred over the main window each time it opens. If it wasn't
  shown in the previous pass, it counts as reopened.

## Menu bar (menubar.rs)

- An `NSMenuItem` can't hold a closure. Items carry a *tag* that indexes the
  action table. A click queues the `Action`, which is applied on the next frame,
  after the menu has closed. That is why a menu item can safely open a file
  dialog.
- The model is rebuilt at most every 200 ms and handed to AppKit only when it
  changed. Naming a SCSI slot stats its image file.
- Call `setAutoenablesItems(false)` on every `NSMenu`, or AppKit ignores
  `setEnabled(false)`.
- We replace winit's whole menu bar, including its `terminate:` Quit. Our Quit
  goes through `ViewportCommand::Close`, so the exit-time CHD fold still runs
  (see `cmd-q-bypasses-close-intercept-fold-on-poweroff.md`).
- winit uses objc2 0.5 / objc2-app-kit 0.2, and this crate uses 0.6 / 0.3. That
  is fine as long as no typed object crosses between them.
- `NSWindow.allowsAutomaticWindowTabbing = false` is set before launch.
  Otherwise AppKit tabs the config window into a fullscreen main window.

## Status in the title

`window_title()` is pushed at about 4 Hz, and only when the text changed. The
native build expires the toast there, since the status footer that normally
does it isn't drawn.

## Verifying

`cargo test -p iris-gui --features macos-gui` covers the window stand-in's
headless fallback. When driving the dev binary with computer-use, it is a bare
`iris-gui` process, not the installed `IRIS.app`, so screenshots filter it out.
Use `osascript` (System Events → process "iris-gui") to click menu items and
`screencapture` to look.
