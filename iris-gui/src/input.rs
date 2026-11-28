//! Translate egui keyboard and pointer events into display actions.
//!
//! The rules for captured input (the release chord, Ctrl+Alt+Esc, F11, the
//! focus-loss grace) are the display controller's
//! (`iris_gui_core::controller::display`). This module only turns egui's
//! events into physical keys and deltas, and grabs or frees the host cursor to
//! match the controller's capture state.
//!
//! Capture starts with a click on the framebuffer widget itself: egui routes
//! clicks on open menus and popups to those widgets first, so navigating menus
//! over the display never captures. While captured, raw motion arrives as
//! `egui::Event::MouseMoved` deltas (eframe forwards `DeviceEvent::MouseMotion`
//! regardless of grab).

use egui::{CursorGrab, Event, Key, MouseWheelUnit, PointerButton, ViewportCommand};
use iris_gui_core::controller::display::{DisplayAction, GuestKey as KeyCode};

/// What the shell has applied to the host cursor.
#[derive(Default)]
pub struct InputState {
    grabbed: bool,
}

/// Collect this frame's input as display actions. `captured` is the
/// controller's state; `fb_clicked` whether the framebuffer widget was clicked.
pub fn pump(ctx: &egui::Context, fb_clicked: bool, captured: bool) -> Vec<DisplayAction> {
    let mut out = Vec::new();
    if !captured {
        if fb_clicked {
            out.push(DisplayAction::Capture);
        }
        return out;
    }
    let (mut dx, mut dy, mut lines, mut points) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut buttons = 0u8;
    ctx.input(|i| {
        out.push(DisplayAction::Focus(i.focused));
        for ev in &i.events {
            match ev {
                Event::MouseMoved(d) => {
                    dx += d.x;
                    dy += d.y;
                }
                // Prefer the physical position: `key` is already layout-translated
                // by the host, and the guest applies its own `keybd=` layout on top.
                Event::Key { key, physical_key, pressed, repeat, .. } => {
                    let k = physical_key.unwrap_or(*key);
                    // egui has no Key for the JIS-only keys (IntlYen, IntlRo...), so
                    // they arrive with no physical key and their logical `\` or `|`.
                    // A physical `\|` key always has one, so this is the yen key.
                    let code = if physical_key.is_none() && matches!(k, Key::Backslash | Key::Pipe) {
                        Some(KeyCode::IntlYen)
                    } else if k == Key::F11 {
                        Some(KeyCode::F11)
                    } else {
                        map_key(k)
                    };
                    if let Some(code) = code {
                        out.push(DisplayAction::Key { key: code, down: *pressed, repeat: *repeat });
                    }
                }
                // egui-winit turns Ctrl+C/X/V into clipboard commands and drops
                // the key event, so on Linux and Windows the guest would never
                // see Ctrl+C. Re-synthesise the letter as a tap; the held Ctrl
                // already went to the guest. macOS Cmd+C stays with the host.
                Event::Copy if i.modifiers.ctrl => tap(&mut out, KeyCode::KeyC),
                Event::Cut if i.modifiers.ctrl => tap(&mut out, KeyCode::KeyX),
                Event::Paste(_) if i.modifiers.ctrl => tap(&mut out, KeyCode::KeyV),
                Event::MouseWheel { unit, delta, .. } => match unit {
                    MouseWheelUnit::Line => lines += delta.y,
                    MouseWheelUnit::Point => points += delta.y,
                    MouseWheelUnit::Page => lines += delta.y * 15.0,
                },
                _ => {}
            }
        }
        for (b, bit) in [
            (PointerButton::Primary, 0x01),
            (PointerButton::Secondary, 0x02),
            (PointerButton::Middle, 0x04),
            (PointerButton::Extra1, 0x08),
            (PointerButton::Extra2, 0x10),
        ] {
            if i.pointer.button_down(b) {
                buttons |= bit;
            }
        }
    });
    out.push(DisplayAction::Pointer { dx, dy, buttons, wheel_lines: lines, wheel_points: points });
    out
}

fn tap(out: &mut Vec<DisplayAction>, key: KeyCode) {
    out.push(DisplayAction::Key { key, down: true, repeat: false });
    out.push(DisplayAction::Key { key, down: false, repeat: false });
}

/// Pick the cursor-grab mode winit supports here: `Locked` on macOS, Windows
/// and Wayland, `Confined` on X11 (which has no `Locked`; raw motion deltas
/// arrive either way).
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

/// Hide and grab the host cursor while captured, free it otherwise.
pub fn sync_cursor(ctx: &egui::Context, state: &mut InputState, captured: bool) {
    if state.grabbed == captured {
        return;
    }
    state.grabbed = captured;
    ctx.send_viewport_cmd(ViewportCommand::CursorVisible(!captured));
    ctx.send_viewport_cmd(ViewportCommand::CursorGrab(if captured { grab_mode() } else { CursorGrab::None }));
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
