//! Translate egui keyboard / pointer events into iris PS2 controller writes.
//!
//! ## Mouse / keyboard capture
//!
//! The guest's PS/2 mouse is *relative*: it reports motion deltas, and IRIX
//! draws its own pointer with its own acceleration. There is no way to keep a
//! relative guest pointer pixel-aligned with a visible host cursor — they
//! drift. So we adopt the standard emulator model (mirroring `src/ui.rs`):
//! **click the framebuffer to capture.** On capture we hide the host cursor
//! and lock it in place (`CursorGrab::Locked`); raw motion then arrives as
//! `egui::Event::MouseMoved` deltas (eframe forwards `DeviceEvent::MouseMotion`
//! regardless of grab), which we feed straight to the guest. Only the guest's
//! own pointer is visible, so there is nothing to misalign. `RELEASE_CHORD` releases.
//!
//! While captured we also forward keyboard input to the guest; while *not*
//! captured we forward nothing, so menu clicks and typing into the config
//! side panel stay with egui.
//!
//! The framebuffer panel calls `pump(...)` each frame, passing whether the REX3
//! image widget itself was clicked this frame. Using the widget's own
//! `Response::clicked()` (rather than a raw point-in-rect test) means egui's
//! hit-testing already routed clicks on open menus / popups to those widgets,
//! so navigating menus over the display never gets "eaten" into a capture.

use egui::{CursorGrab, Event, Key, MouseWheelUnit, PointerButton, ViewportCommand};
use iris::ps2::Ps2Controller;
use winit::keyboard::KeyCode;

pub struct InputState {
    /// Modifier keys currently held down in the guest, so they can be lifted on release.
    held_mods: Vec<KeyCode>,
    /// Set when another key is pressed while the release chord is held, so the chord sends its modifiers to the guest instead of releasing capture.
    chord_consumed: bool,
    last_buttons: u8,         // bit0=L, bit1=R, bit2=M, bit3=B4, bit4=B5
    /// True while the host cursor is grabbed and input is routed to the guest.
    pub captured: bool,
    /// When the window first reported unfocused while captured. Used to debounce
    /// a real focus loss (alt-tab) from the spurious single-frame `focused=false`
    /// flickers macOS emits around cursor-grab / notifications / Spaces, which
    /// otherwise dropped capture mid-typing. `None` while focused.
    unfocused_since: Option<std::time::Instant>,
}

impl Default for InputState {
    fn default() -> Self {
        Self { held_mods: Vec::new(), chord_consumed: false, last_buttons: 0, captured: false, unfocused_since: None }
    }
}

/// Hold these two together (and press nothing else) to release capture.
#[cfg(target_os = "macos")]
const RELEASE_CHORD: [KeyCode; 2] = [KeyCode::AltLeft, KeyCode::SuperLeft];
#[cfg(not(target_os = "macos"))]
const RELEASE_CHORD: [KeyCode; 2] = [KeyCode::ControlLeft, KeyCode::AltLeft];

/// Human-readable name of `RELEASE_CHORD`, for the capture hint in the control column.
#[cfg(target_os = "macos")]
pub const RELEASE_HINT: &str = "Left Option+Cmd";
#[cfg(not(target_os = "macos"))]
pub const RELEASE_HINT: &str = "Left Ctrl+Alt";

fn is_modifier(kc: KeyCode) -> bool {
    matches!(kc, KeyCode::ShiftLeft | KeyCode::ShiftRight | KeyCode::ControlLeft | KeyCode::ControlRight
        | KeyCode::AltLeft | KeyCode::AltRight | KeyCode::SuperLeft | KeyCode::SuperRight)
}

/// How long the window must stay unfocused before a capture is released. Long
/// enough to ride out macOS's transient focus flicker, short enough that a real
/// alt-tab frees the cursor promptly.
const FOCUS_LOSS_GRACE: std::time::Duration = std::time::Duration::from_millis(300);

pub fn pump(ctx: &egui::Context, fb_clicked: bool, ps2: &Ps2Controller, state: &mut InputState, scroll_pixels_per_line: f64) {
    // Collect everything we need inside the input borrow, then act afterwards
    // (sending viewport commands / PS2 writes outside the `input()` closure).
    let mut want_enter = false;
    let mut want_release = false;
    let mut esc_chord = false;
    let mut focused = true;
    let mut dx = 0.0f32;
    let mut dy = 0.0f32;
    let mut dz = 0.0f32;
    let mut buttons = state.last_buttons;
    let mut keys: Vec<(KeyCode, bool)> = Vec::new();
    let mut f11_to_guest = false;

    ctx.input(|i| {
        if !state.captured {
            // Not captured: capture only when the framebuffer Image widget
            // itself was clicked. egui routes clicks on menus/popups/panels to
            // those widgets first, so this never fires for menu navigation that
            // happens to overlap the display. Everything else stays with egui.
            if fb_clicked { want_enter = true; }
            return;
        }

        // Captured. RELEASE_CHORD releases (see below); Ctrl+Alt+Esc still works as an explicit fallback, and a real focus loss (alt-tab) releases after a grace period so a one-frame flicker doesn't drop capture mid-typing.
        esc_chord = i.key_pressed(Key::Escape) && i.modifiers.ctrl && i.modifiers.alt;
        focused = i.focused;

        for ev in &i.events {
            match ev {
                // Raw relative motion (eframe → DeviceEvent::MouseMotion).
                Event::MouseMoved(d) => { dx += d.x; dy += d.y; }
                // Prefer the physical position: `key` is already layout-translated by the host, and the guest applies its own `keybd=` layout on top, which mangles every non-US layout.
                Event::Key { key, physical_key, pressed, repeat, .. } => {
                    let k = physical_key.unwrap_or(*key);
                    if k == Key::F11 {
                        // Plain F11 is the GUI's fullscreen toggle; Ctrl+Alt+F11 is the escape hatch that sends a bare F11 to IRIX.
                        if *pressed && !*repeat && i.modifiers.ctrl && i.modifiers.alt {
                            f11_to_guest = true;
                        }
                    } else if let Some(kc) = map_key(k) {
                        keys.push((kc, *pressed));
                    }
                }
                // egui-winit swallows Ctrl/Cmd + C/X/V into clipboard *commands*
                // (Cut/Copy/Paste) and never emits the underlying Key event — so
                // on Linux and Windows, where `command == ctrl`, the guest would
                // never see Ctrl+C (SIGINT in a shell), Ctrl+X, or Ctrl+V. Re-
                // synthesise the bare letter as a tap; the held Ctrl is already
                // sent by the modifier diff below, so the guest forms the full
                // chord. Gated on `ctrl` so macOS Cmd+C/X/V — where real Ctrl+C
                // still arrives as a normal Key, and the Cmd combo has no guest
                // meaning — is left to the host clipboard.
                Event::Copy     if i.modifiers.ctrl => { keys.push((KeyCode::KeyC, true)); keys.push((KeyCode::KeyC, false)); }
                Event::Cut      if i.modifiers.ctrl => { keys.push((KeyCode::KeyX, true)); keys.push((KeyCode::KeyX, false)); }
                Event::Paste(_) if i.modifiers.ctrl => { keys.push((KeyCode::KeyV, true)); keys.push((KeyCode::KeyV, false)); }
                Event::MouseWheel { unit, delta, .. } => {
                    let lines = match unit {
                        MouseWheelUnit::Line => delta.y,
                        MouseWheelUnit::Point => delta.y / scroll_pixels_per_line as f32,
                        MouseWheelUnit::Page => delta.y * 15.0,
                    };
                    dz += lines;
                }
                _ => {}
            }
        }

        let mut b = 0u8;
        if i.pointer.button_down(PointerButton::Primary)   { b |= 0x01; }
        if i.pointer.button_down(PointerButton::Secondary) { b |= 0x02; }
        if i.pointer.button_down(PointerButton::Middle)    { b |= 0x04; }
        if i.pointer.button_down(PointerButton::Extra1)    { b |= 0x08; }
        if i.pointer.button_down(PointerButton::Extra2)    { b |= 0x10; }
        buttons = b;
    });

    // Decide whether to release, outside the input borrow. The Esc chord is
    // immediate; focus loss is debounced over `FOCUS_LOSS_GRACE` so a transient
    // macOS flicker doesn't drop capture, while a genuine alt-tab still does.
    if state.captured {
        if esc_chord {
            want_release = true;
        } else if !focused {
            let now = std::time::Instant::now();
            let since = *state.unfocused_since.get_or_insert(now);
            if now.duration_since(since) >= FOCUS_LOSS_GRACE {
                want_release = true;
            }
        } else {
            state.unfocused_since = None;
        }
    }

    if want_enter {
        engage_capture(ctx, state);
        return;
    }

    if want_release {
        release_capture(ctx, ps2, state);
        return;
    }

    // Modifiers arrive as real L/R key events (egui 0.35), so forward them like any other key and track what's held.
    if f11_to_guest { state.chord_consumed = true; }
    let mut chord_release = false;
    for (kc, pressed) in keys {
        let was_chord = RELEASE_CHORD.iter().all(|k| state.held_mods.contains(k));
        if is_modifier(kc) {
            if pressed {
                if !state.held_mods.contains(&kc) { state.held_mods.push(kc); }
            } else {
                state.held_mods.retain(|k| *k != kc);
            }
        } else if pressed && was_chord {
            state.chord_consumed = true;
        }
        let now_chord = RELEASE_CHORD.iter().all(|k| state.held_mods.contains(k));
        if now_chord && !was_chord { state.chord_consumed = false; }
        if was_chord && !now_chord && !state.chord_consumed { chord_release = true; }
        ps2.push_kb(kc, pressed);
    }

    // Ctrl+Alt+F11 → a *bare* F11 to the guest: plain F11 is the GUI's fullscreen toggle, so this chord is the only path for F11 into IRIX.
    if f11_to_guest {
        let held = state.held_mods.clone();
        for kc in &held { ps2.push_kb(*kc, false); }
        ps2.push_kb(KeyCode::F11, true);
        ps2.push_kb(KeyCode::F11, false);
        for kc in &held { ps2.push_kb(*kc, true); }
    }

    if chord_release {
        release_capture(ctx, ps2, state);
        return;
    }

    // ---- mouse: raw per-frame delta + button diff + scroll. ----
    let (mdx, mdy, mdz) = (dx as i32, dy as i32, dz as i32);
    if buttons != state.last_buttons || mdx != 0 || mdy != 0 || mdz != 0 {
        ps2.push_mouse_input(buttons, mdx, mdy, mdz);
        state.last_buttons = buttons;
    }
}

/// Pick the cursor-grab mode winit actually supports on this platform/session.
///
/// winit's two grab modes are not portable: `Locked` (lock the cursor in place
/// and deliver relative motion) is supported on macOS and Wayland but **not
/// X11**, while `Confined` (keep the cursor inside the window) is supported on
/// X11 and Windows but **not macOS**. egui-winit does no fallback — it just
/// logs the error — so a blanket `Locked` silently fails on X11, leaving the
/// cursor un-grabbed: it drifts off-window, loses focus, and capture drops.
///
/// We rely on raw `DeviceEvent::MouseMotion` deltas (which arrive regardless of
/// grab mode), so `Confined` is sufficient on X11 — it just stops the cursor
/// escaping the window. Detect Wayland via `WAYLAND_DISPLAY`; otherwise assume
/// X11 on Linux. macOS/Windows keep `Locked`.
fn grab_mode() -> CursorGrab {
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("WAYLAND_DISPLAY").is_some() {
            CursorGrab::Locked
        } else {
            CursorGrab::Confined
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        CursorGrab::Locked
    }
}

/// Engage capture: grab + hide the host cursor and route input to the guest,
/// exactly as a click on the framebuffer does. Driven both by that click (via
/// `pump`) and by the side-panel "Capture" button. No-op if already captured.
pub fn engage_capture(ctx: &egui::Context, state: &mut InputState) {
    if state.captured { return; }
    state.captured = true;
    // Anchor modifier/button state so we don't leave a stale key held from before capture.
    state.held_mods.clear();
    state.chord_consumed = false;
    state.last_buttons = 0;
    state.unfocused_since = None;
    ctx.send_viewport_cmd(ViewportCommand::CursorVisible(false));
    ctx.send_viewport_cmd(ViewportCommand::CursorGrab(grab_mode()));
}

/// Release a capture: show + ungrab the host cursor and lift any modifiers
/// we'd synthesised, so the guest doesn't see stuck keys. Safe to call when
/// not captured (no-op). Used both for Esc/focus-loss and when the emulator
/// stops while the framebuffer still had the grab.
pub fn release_capture(ctx: &egui::Context, ps2: &Ps2Controller, state: &mut InputState) {
    if !state.captured { return; }
    for kc in state.held_mods.drain(..) { ps2.push_kb(kc, false); }
    state.captured = false;
    state.chord_consumed = false;
    state.last_buttons = 0;
    state.unfocused_since = None;
    ctx.send_viewport_cmd(ViewportCommand::CursorVisible(true));
    ctx.send_viewport_cmd(ViewportCommand::CursorGrab(CursorGrab::None));
}

/// Ungrab the cursor without touching the guest (it may already be gone).
/// Called when the emulator stops while the framebuffer still held capture,
/// so the host cursor doesn't stay hidden/locked. No-op when not captured.
pub fn force_release(ctx: &egui::Context, state: &mut InputState) {
    if !state.captured { return; }
    state.captured = false;
    state.held_mods.clear();
    state.chord_consumed = false;
    state.last_buttons = 0;
    state.unfocused_since = None;
    ctx.send_viewport_cmd(ViewportCommand::CursorVisible(true));
    ctx.send_viewport_cmd(ViewportCommand::CursorGrab(CursorGrab::None));
}


/// egui::Key (a physical position, per the caller) → winit KeyCode; None for keys iris's scancode mapper doesn't recognise.
fn map_key(k: Key) -> Option<KeyCode> {
    Some(match k {
        // Letters
        Key::A => KeyCode::KeyA, Key::B => KeyCode::KeyB, Key::C => KeyCode::KeyC,
        Key::D => KeyCode::KeyD, Key::E => KeyCode::KeyE, Key::F => KeyCode::KeyF,
        Key::G => KeyCode::KeyG, Key::H => KeyCode::KeyH, Key::I => KeyCode::KeyI,
        Key::J => KeyCode::KeyJ, Key::K => KeyCode::KeyK, Key::L => KeyCode::KeyL,
        Key::M => KeyCode::KeyM, Key::N => KeyCode::KeyN, Key::O => KeyCode::KeyO,
        Key::P => KeyCode::KeyP, Key::Q => KeyCode::KeyQ, Key::R => KeyCode::KeyR,
        Key::S => KeyCode::KeyS, Key::T => KeyCode::KeyT, Key::U => KeyCode::KeyU,
        Key::V => KeyCode::KeyV, Key::W => KeyCode::KeyW, Key::X => KeyCode::KeyX,
        Key::Y => KeyCode::KeyY, Key::Z => KeyCode::KeyZ,
        // Digits
        Key::Num0 => KeyCode::Digit0, Key::Num1 => KeyCode::Digit1,
        Key::Num2 => KeyCode::Digit2, Key::Num3 => KeyCode::Digit3,
        Key::Num4 => KeyCode::Digit4, Key::Num5 => KeyCode::Digit5,
        Key::Num6 => KeyCode::Digit6, Key::Num7 => KeyCode::Digit7,
        Key::Num8 => KeyCode::Digit8, Key::Num9 => KeyCode::Digit9,
        // Navigation / editing
        Key::Escape    => KeyCode::Escape,
        Key::Tab       => KeyCode::Tab,
        Key::Backspace => KeyCode::Backspace,
        Key::Enter     => KeyCode::Enter,
        Key::Space     => KeyCode::Space,
        Key::Insert    => KeyCode::Insert,
        Key::Delete    => KeyCode::Delete,
        Key::Home      => KeyCode::Home,
        Key::End       => KeyCode::End,
        Key::PageUp    => KeyCode::PageUp,
        Key::PageDown  => KeyCode::PageDown,
        Key::ArrowUp    => KeyCode::ArrowUp,
        Key::ArrowDown  => KeyCode::ArrowDown,
        Key::ArrowLeft  => KeyCode::ArrowLeft,
        Key::ArrowRight => KeyCode::ArrowRight,
        // Punctuation
        Key::Comma        => KeyCode::Comma,
        Key::Period       => KeyCode::Period,
        Key::Slash        => KeyCode::Slash,
        Key::Backslash    => KeyCode::Backslash,
        Key::Minus        => KeyCode::Minus,
        Key::Equals       => KeyCode::Equal,
        Key::Plus         => KeyCode::Equal, // shifted: same physical key
        Key::Semicolon    => KeyCode::Semicolon,
        Key::Colon        => KeyCode::Semicolon,
        Key::Quote        => KeyCode::Quote,
        Key::OpenBracket  => KeyCode::BracketLeft,
        Key::CloseBracket => KeyCode::BracketRight,
        Key::Backtick     => KeyCode::Backquote,
        // Only reachable on the logical fallback (no physical_key): shifted symbols sharing a key with Backslash/Slash.
        Key::Pipe         => KeyCode::Backslash,
        Key::Questionmark => KeyCode::Slash,
        // ISO 102nd key (< > |), left of Z on European keyboards — new in egui 0.35.
        Key::IntlBackslash => KeyCode::IntlBackslash,
        // Modifiers as real L/R keys (egui 0.35); AltRight is AltGr, which the guest needs for @ \ | { } [ ] ~ on DE/de_CH.
        Key::ShiftLeft   => KeyCode::ShiftLeft,   Key::ShiftRight   => KeyCode::ShiftRight,
        Key::ControlLeft => KeyCode::ControlLeft, Key::ControlRight => KeyCode::ControlRight,
        Key::AltLeft     => KeyCode::AltLeft,     Key::AltRight     => KeyCode::AltRight,
        Key::SuperLeft   => KeyCode::SuperLeft,   Key::SuperRight   => KeyCode::SuperRight,
        // F11 is reserved by the GUI (fullscreen); iris's scancode sets stop at F12, so F13+ are dropped.
        Key::F1 => KeyCode::F1, Key::F2 => KeyCode::F2, Key::F3  => KeyCode::F3,
        Key::F4 => KeyCode::F4, Key::F5 => KeyCode::F5, Key::F6  => KeyCode::F6,
        Key::F7 => KeyCode::F7, Key::F8 => KeyCode::F8, Key::F9  => KeyCode::F9,
        Key::F10 => KeyCode::F10,
        Key::F12 => KeyCode::F12,
        _ => return None,
    })
}
