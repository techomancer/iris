# GTK 4 has no pointer lock, so the Linux shell steers the guest's pointer

Status: rule (2026-11-28), `iris-gui/frontends/gtk4`.

**What is missing.** A captured mouse needs two things a toolkit has to
offer: the pointer confined to the window (or hidden and warped back to the
middle), and motion reported as deltas rather than positions. GTK 4 exposes
neither:

- There is no `gdk_device_warp` for applications and no pointer-grab call;
  GTK 3's `gdk_seat_grab` is gone.
- Wayland can do it through `pointer-constraints-unstable-v1` and
  `relative-pointer-unstable-v1`, but GDK keeps both to itself — nothing in
  the public API locks or confines a pointer. On X11 (and XWayland) the
  underlying calls exist, but only through the X11 backend's own headers,
  which would make capture work on one backend and not the other.

**What the shell does instead.** The core's Follow mode
(`PointerMode::Follow`, `model::pointer_follow`), which the plan already
names for hosts that cannot report motion: the guest's pointer is steered to
wherever the host's is over the picture. `GuestView` turns a motion event
into the guest pixel under it (inverting the draw rectangle the same
`display::picture_size` rule produced) and sends
`DisplayAction::PointerAt`. Nothing is captured, so nothing has to be
released; the keyboard reaches the guest while the display has the focus.

**Rules.**
- Do not add a capture path to the GTK shell that only works on X11. If
  pointer lock is wanted later, it needs the Wayland protocols through a
  backend-specific crate, and Follow has to stay for the other backend.
- The pointer steering is the core's: a shell sends `PointerAt` with guest
  pixels and buttons, and never computes where the guest's pointer should
  go. It trails a fast move a little; that is the mode, not a bug.
- `gtk::GestureClick` with `set_button(0)` reports every button; the held
  mask comes from the event state (`BUTTON1_MASK` and friends), and the
  pressed or released button has to be added to or removed from it by hand,
  because the state describes the moment *before* the event.
