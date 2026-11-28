//! macOS virtual key codes (`kVK_*`, physical positions on an ANSI layout)
//! to the guest keys the PS/2 layer takes. Physical, never layout-translated:
//! the guest applies its own `keybd=` layout
//! (clanker/gui/keyboard-layout-send-physical-keys-not-logical.md).

use iris_gui_core::controller::display::GuestKey as K;
use iris_gui_core::controller::keys::Key;

/// The guest key at macOS virtual key code `code`.
pub fn guest_key(code: u16) -> Option<K> {
    Some(match code {
        0x00 => K::KeyA,
        0x01 => K::KeyS,
        0x02 => K::KeyD,
        0x03 => K::KeyF,
        0x04 => K::KeyH,
        0x05 => K::KeyG,
        0x06 => K::KeyZ,
        0x07 => K::KeyX,
        0x08 => K::KeyC,
        0x09 => K::KeyV,
        0x0A => K::IntlBackslash, // ISO section key, left of Z
        0x0B => K::KeyB,
        0x0C => K::KeyQ,
        0x0D => K::KeyW,
        0x0E => K::KeyE,
        0x0F => K::KeyR,
        0x10 => K::KeyY,
        0x11 => K::KeyT,
        0x12 => K::Digit1,
        0x13 => K::Digit2,
        0x14 => K::Digit3,
        0x15 => K::Digit4,
        0x16 => K::Digit6,
        0x17 => K::Digit5,
        0x18 => K::Equal,
        0x19 => K::Digit9,
        0x1A => K::Digit7,
        0x1B => K::Minus,
        0x1C => K::Digit8,
        0x1D => K::Digit0,
        0x1E => K::BracketRight,
        0x1F => K::KeyO,
        0x20 => K::KeyU,
        0x21 => K::BracketLeft,
        0x22 => K::KeyI,
        0x23 => K::KeyP,
        0x24 => K::Enter,
        0x25 => K::KeyL,
        0x26 => K::KeyJ,
        0x27 => K::Quote,
        0x28 => K::KeyK,
        0x29 => K::Semicolon,
        0x2A => K::Backslash,
        0x2B => K::Comma,
        0x2C => K::Slash,
        0x2D => K::KeyN,
        0x2E => K::KeyM,
        0x2F => K::Period,
        0x30 => K::Tab,
        0x31 => K::Space,
        0x32 => K::Backquote,
        0x33 => K::Backspace,
        0x35 => K::Escape,
        0x36 => K::SuperRight,
        0x37 => K::SuperLeft,
        0x38 => K::ShiftLeft,
        0x39 => K::CapsLock,
        0x3A => K::AltLeft,
        0x3B => K::ControlLeft,
        0x3C => K::ShiftRight,
        0x3D => K::AltRight,
        0x3E => K::ControlRight,
        0x41 => K::NumpadDecimal,
        0x43 => K::NumpadMultiply,
        0x45 => K::NumpadAdd,
        0x47 => K::NumLock, // Clear on a Mac keypad
        0x4B => K::NumpadDivide,
        0x4C => K::NumpadEnter,
        0x4E => K::NumpadSubtract,
        0x51 => K::NumpadEqual,
        0x52 => K::Numpad0,
        0x53 => K::Numpad1,
        0x54 => K::Numpad2,
        0x55 => K::Numpad3,
        0x56 => K::Numpad4,
        0x57 => K::Numpad5,
        0x58 => K::Numpad6,
        0x59 => K::Numpad7,
        0x5B => K::Numpad8,
        0x5C => K::Numpad9,
        0x5D => K::IntlYen, // JIS yen
        0x5E => K::IntlRo,  // JIS underscore / ro
        0x60 => K::F5,
        0x61 => K::F6,
        0x62 => K::F7,
        0x63 => K::F3,
        0x64 => K::F8,
        0x65 => K::F9,
        0x67 => K::F11,
        0x6D => K::F10,
        0x6F => K::F12,
        0x72 => K::Insert, // Help on old Mac keyboards
        0x73 => K::Home,
        0x74 => K::PageUp,
        0x75 => K::Delete,
        0x76 => K::F4,
        0x77 => K::End,
        0x78 => K::F2,
        0x79 => K::PageDown,
        0x7A => K::F1,
        0x7B => K::ArrowLeft,
        0x7C => K::ArrowRight,
        0x7D => K::ArrowDown,
        0x7E => K::ArrowUp,
        _ => return None,
    })
}

/// Whether `code` is a modifier key (reported through flagsChanged).
pub fn is_modifier(code: u16) -> bool {
    matches!(code, 0x36..=0x3E)
}

/// The key a shortcut lookup sees for `code` and the typed characters.
pub fn shortcut_key(code: u16, chars: &str) -> Option<Key> {
    Some(match code {
        0x7A => Key::F(1),
        0x78 => Key::F(2),
        0x63 => Key::F(3),
        0x76 => Key::F(4),
        0x60 => Key::F(5),
        0x61 => Key::F(6),
        0x62 => Key::F(7),
        0x64 => Key::F(8),
        0x65 => Key::F(9),
        0x6D => Key::F(10),
        0x67 => Key::F(11),
        0x6F => Key::F(12),
        0x35 => Key::Escape,
        0x24 => Key::Enter,
        0x30 => Key::Tab,
        _ => Key::Char(chars.chars().next()?),
    })
}
