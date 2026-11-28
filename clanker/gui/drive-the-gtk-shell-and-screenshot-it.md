# Drive the GTK shell, and let it screenshot itself

Status: howto (2026-11-27), `iris-gui/frontends/gtk4`.

The GTK shell (`iris-gtk`) has the same kind of test hooks as the AppKit one
(`clanker/macos/drive-the-appkit-shell-with-the-screen-locked.md`), so an
agent can drive it and *see* it without a screenshot tool or an unlocked
screen.

- `IRIS_GTK_SCRIPT="step;step;..."` runs one step every
  `IRIS_GTK_SCRIPT_DELAY` seconds (default 3, fractions allowed). Steps so
  far: `shot:PATH` (write the window to a PNG) and `quit`. Later slices add
  command ids, `select:`, `set:`, `open:`/`pick:`, `type:` and `dump:`, as
  AppKit has them.
- Run with `HOME=<scratch>` so the library, preferences and the pidfile are
  separate from the user's — and so `AppController::launch` cannot reclaim
  (that is, kill) a front end the user is using.

**The window paints itself.** There is no way to grab another process's
window on Wayland, and this desktop has no `grim`, `import` or
`gnome-screenshot`; GNOME's portal wants a click. So `shot:` asks the window
for its own picture instead:

```rust
let renderer = window.native().and_then(|n| n.renderer())?;     // the window's GskRenderer
let paintable = gtk::WidgetPaintable::new(Some(&window));
let snapshot = gtk::Snapshot::new();
paintable.snapshot(&snapshot, w as f64, h as f64);
renderer.render_texture(&snapshot.to_node()?, None).save_to_png(path)
```

It needs no permission, no tool and no compositor support, it captures the
real widget tree (so it is also what a CI run would compare), and it works
the same under Wayland and `GDK_BACKEND=x11`. It does **not** capture the
compositor's own shadow and rounded corners, nor a pop-up that is its own
surface (a menu or a dialog is a separate `GtkNative`: screenshot that
widget's own window instead).

Gotchas found so far:

- `GApplication`'s uniqueness is per application id. A second `iris-gtk`
  hands its activation to the first and exits silently with status 0; the
  first presents its window. The pidfile
  (`iris_gui_core::model::single_instance`) is the cross-shell guard, for
  when the egui app or `iris-appkit` holds it — nothing can raise *their*
  window here, so the GTK shell says where it is and leaves it alone.
- Every front end's process name must be in
  `single_instance::is_front_end_name`, or two of them run at once and fight
  over the monitor and serial ports.
- A debug build never starts the guest CPU
  (`debug-builds-never-start-the-cpu.md`); that holds for `iris-gtk` too.
