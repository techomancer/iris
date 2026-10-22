# Keyboard capture: egui steals Tab/arrows/Esc, and egui-winit eats Ctrl+C/X/V

Status: confirmed fix (2026-06-19). When the framebuffer has keyboard capture,
some keys never reach the guest. There are **three independent causes**, all in
the egui/egui-winit layer, none in iris's own keymap. If "key X doesn't reach
the guest while captured" comes up again, check these before touching
`map_key`/`ps2.rs` — the scancode tables are complete (Tab/arrows/F5 are all
mapped end-to-end).

## 1. Focus-navigation keys (Tab, arrows, Esc) — egui's focus engine

`pump()` reads `ctx.input(|i| i.events)`, and egui clones **all** raw events into
`InputState.events` (`egui .../input_state/mod.rs`: `events: new.events.clone()`),
so the Tab event *is* delivered to us. The problem is the side effect:
`Memory::begin_pass` (`egui .../memory/mod.rs`) interprets an unfiltered
`Key::Tab` as `FocusDirection::Next`, arrows as directional focus moves, and
`Key::Escape` as "clear focus". On the press, egui moves keyboard focus **off**
the framebuffer `Image` onto the next focusable side-panel widget — and that
widget can then `consume_key` later keystrokes (`events.retain(...)`) before
`pump` sees them. A plain `egui::Image` never claims these keys, so by default it
loses them.

Fix: while captured, pin focus to the framebuffer and lock the focus filter
(`iris-gui/src/main.rs`, `framebuffer_panel`):

```rust
if captured {
    if !response.has_focus() { response.request_focus(); }
    ui.memory_mut(|m| m.set_focus_lock_filter(response.id, egui::EventFilter {
        tab: true, horizontal_arrows: true, vertical_arrows: true, escape: true,
    }));
}
```

`set_focus_lock_filter` only takes effect when the widget `had_focus_last_frame
&& has_focus` — so it must be re-applied every frame (TextEdit does the same),
and there's a harmless one-frame delay when capture engages via the side-panel
button (FB wasn't focused yet). `escape: true` does **not** break release: the
Ctrl+Alt+Esc chord is read globally in `pump` (`i.key_pressed(Escape) && ctrl &&
alt`) before any key is forwarded, and plain Esc still flows to the guest. It
also fixes a latent bug where a bare Esc used to clear FB focus.

## 2. Ctrl+C / Ctrl+X / Ctrl+V — egui-winit converts them to clipboard commands

`egui-winit::on_keyboard_input` checks `is_copy/cut/paste_command` and, on a
match, pushes `Event::Copy/Cut/Paste` and **`return`s without emitting the Key
event**. Those predicates use `modifiers.command`, and `command == ctrl` on
Linux/Windows — so on those platforms the guest never sees **Ctrl+C (SIGINT!)**,
Ctrl+X, or Ctrl+V. (On macOS `command == Cmd`, so real Ctrl+C still arrives as a
normal `Key::C`; only Cmd+C/X/V are swallowed, which the guest doesn't need.)

Fix: handle those events in `pump`'s loop, gated on `i.modifiers.ctrl`, and
re-synthesise the bare letter as a tap. The held Ctrl is already sent by the
modifier diff, so the guest forms the full chord:

```rust
Event::Copy     if i.modifiers.ctrl => { keys.push((KeyCode::KeyC, true)); keys.push((KeyCode::KeyC, false)); }
Event::Cut      if i.modifiers.ctrl => { keys.push((KeyCode::KeyX, true)); keys.push((KeyCode::KeyX, false)); }
Event::Paste(_) if i.modifiers.ctrl => { keys.push((KeyCode::KeyV, true)); keys.push((KeyCode::KeyV, false)); }
```

The `ctrl` gate means macOS Cmd+C is left to the host clipboard. (We forward the
keystroke, not the clipboard text — in a Unix shell Ctrl+V is "literal next", so
faithful passthrough is the correct behavior for a captured emulator.)

## 3. F5 was dropped on a wrong comment

`map_key` had `// F-keys (egui has no F5 …)` and skipped it — but `egui::Key::F5`
exists and `ps2.rs` maps `KeyCode::F5` (set 2). Added it. The PS/2 set stops at
F12, so F13+ are still legitimately dropped.

## 4. F11 is the fullscreen toggle — Ctrl+Alt+F11 is the escape hatch into IRIX

Plain F11 is the GUI fullscreen toggle (`App::update`), so `map_key` does **not**
forward it — otherwise it would both toggle fullscreen *and* reach the guest.
**Ctrl+Alt+F11** is reserved as the only way to send a real F11 to IRIX: the
fullscreen handler is gated with `!(ctrl && alt)`, and `pump` detects the chord
on the press edge and sends a **bare** F11. Because the modifier diff has already
pressed the chord's Ctrl+Alt in the guest, `pump` lifts whatever modifiers are
held (`state.held_mods`), taps F11, then re-presses them — so IRIX sees an
unmodified F11. The hint lives on the capture status block (`capture_controls`)
alongside the release-chord hint (`input::RELEASE_HINT`).

## What is NOT fixable at this layer

egui's `Key` enum collapses the numpad into `Num0..9` / `Plus`/`Minus`/etc., so
numpad keys reach the guest only as their main-row equivalents, and there are no
egui keys for NumLock/ScrollLock/CapsLock/PrintScreen/ContextMenu — even though
`ps2.rs` has scancodes for some. Distinguishing them would require reading raw
winit `KeyEvent.physical_key`/`location` instead of egui events.

Re-verified against **egui 0.35** (2026-08-03): still true. 0.35 *did* add
`Key::IntlBackslash` and discrete `ShiftLeft/Right`, `ControlLeft/Right`,
`AltLeft/Right`, `SuperLeft/Right` (all now mapped), but no numpad and none of
the lock/menu keys. `map_key` in `input.rs` covers everything egui can express.
