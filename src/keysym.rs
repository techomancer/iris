//! One table of the keys the host can press on the guest's keyboard: the
//! physical key (winit's `KeyCode`, what `Ps2Controller::push_kb` takes), its
//! X keysyms unshifted and shifted on a US layout, its XT (scancode set 1)
//! number as QEMU's extended key event sends it (`0xE0 nn` as `0x80 | nn`),
//! and its X keysym name.
//!
//! VNC viewers send keysyms, or with the QEMU extension the scancode of the
//! key itself, which needs no layout. The monitor's `key` command takes the
//! keysym names, a few short aliases, and winit's names (`KeyA`).

use winit::keyboard::KeyCode;
use winit::keyboard::KeyCode::*;

struct Key {
    code: KeyCode,
    xt: u16,
    name: &'static str,
    sym: u32,
    shifted: u32,
}

const fn k(code: KeyCode, xt: u16, name: &'static str, sym: u32, shifted: u32) -> Key {
    Key { code, xt, name, sym, shifted }
}

#[rustfmt::skip]
static KEYS: &[Key] = &[
    k(Escape, 0x01, "Escape", 0xff1b, 0),
    k(Digit1, 0x02, "1", b'1' as u32, b'!' as u32),
    k(Digit2, 0x03, "2", b'2' as u32, b'@' as u32),
    k(Digit3, 0x04, "3", b'3' as u32, b'#' as u32),
    k(Digit4, 0x05, "4", b'4' as u32, b'$' as u32),
    k(Digit5, 0x06, "5", b'5' as u32, b'%' as u32),
    k(Digit6, 0x07, "6", b'6' as u32, b'^' as u32),
    k(Digit7, 0x08, "7", b'7' as u32, b'&' as u32),
    k(Digit8, 0x09, "8", b'8' as u32, b'*' as u32),
    k(Digit9, 0x0a, "9", b'9' as u32, b'(' as u32),
    k(Digit0, 0x0b, "0", b'0' as u32, b')' as u32),
    k(Minus, 0x0c, "minus", b'-' as u32, b'_' as u32),
    k(Equal, 0x0d, "equal", b'=' as u32, b'+' as u32),
    k(Backspace, 0x0e, "BackSpace", 0xff08, 0),
    k(Tab, 0x0f, "Tab", 0xff09, 0xfe20),
    k(KeyQ, 0x10, "q", b'q' as u32, b'Q' as u32),
    k(KeyW, 0x11, "w", b'w' as u32, b'W' as u32),
    k(KeyE, 0x12, "e", b'e' as u32, b'E' as u32),
    k(KeyR, 0x13, "r", b'r' as u32, b'R' as u32),
    k(KeyT, 0x14, "t", b't' as u32, b'T' as u32),
    k(KeyY, 0x15, "y", b'y' as u32, b'Y' as u32),
    k(KeyU, 0x16, "u", b'u' as u32, b'U' as u32),
    k(KeyI, 0x17, "i", b'i' as u32, b'I' as u32),
    k(KeyO, 0x18, "o", b'o' as u32, b'O' as u32),
    k(KeyP, 0x19, "p", b'p' as u32, b'P' as u32),
    k(BracketLeft, 0x1a, "bracketleft", b'[' as u32, b'{' as u32),
    k(BracketRight, 0x1b, "bracketright", b']' as u32, b'}' as u32),
    k(Enter, 0x1c, "Return", 0xff0d, 0),
    k(ControlLeft, 0x1d, "Control_L", 0xffe3, 0),
    k(KeyA, 0x1e, "a", b'a' as u32, b'A' as u32),
    k(KeyS, 0x1f, "s", b's' as u32, b'S' as u32),
    k(KeyD, 0x20, "d", b'd' as u32, b'D' as u32),
    k(KeyF, 0x21, "f", b'f' as u32, b'F' as u32),
    k(KeyG, 0x22, "g", b'g' as u32, b'G' as u32),
    k(KeyH, 0x23, "h", b'h' as u32, b'H' as u32),
    k(KeyJ, 0x24, "j", b'j' as u32, b'J' as u32),
    k(KeyK, 0x25, "k", b'k' as u32, b'K' as u32),
    k(KeyL, 0x26, "l", b'l' as u32, b'L' as u32),
    k(Semicolon, 0x27, "semicolon", b';' as u32, b':' as u32),
    k(Quote, 0x28, "apostrophe", b'\'' as u32, b'"' as u32),
    k(Backquote, 0x29, "grave", b'`' as u32, b'~' as u32),
    k(ShiftLeft, 0x2a, "Shift_L", 0xffe1, 0),
    k(Backslash, 0x2b, "backslash", b'\\' as u32, b'|' as u32),
    k(KeyZ, 0x2c, "z", b'z' as u32, b'Z' as u32),
    k(KeyX, 0x2d, "x", b'x' as u32, b'X' as u32),
    k(KeyC, 0x2e, "c", b'c' as u32, b'C' as u32),
    k(KeyV, 0x2f, "v", b'v' as u32, b'V' as u32),
    k(KeyB, 0x30, "b", b'b' as u32, b'B' as u32),
    k(KeyN, 0x31, "n", b'n' as u32, b'N' as u32),
    k(KeyM, 0x32, "m", b'm' as u32, b'M' as u32),
    k(Comma, 0x33, "comma", b',' as u32, b'<' as u32),
    k(Period, 0x34, "period", b'.' as u32, b'>' as u32),
    k(Slash, 0x35, "slash", b'/' as u32, b'?' as u32),
    k(ShiftRight, 0x36, "Shift_R", 0xffe2, 0),
    k(NumpadMultiply, 0x37, "KP_Multiply", 0xffaa, 0),
    k(AltLeft, 0x38, "Alt_L", 0xffe9, 0xffe7),
    k(Space, 0x39, "space", b' ' as u32, 0),
    k(CapsLock, 0x3a, "Caps_Lock", 0xffe5, 0),
    k(F1, 0x3b, "F1", 0xffbe, 0),
    k(F2, 0x3c, "F2", 0xffbf, 0),
    k(F3, 0x3d, "F3", 0xffc0, 0),
    k(F4, 0x3e, "F4", 0xffc1, 0),
    k(F5, 0x3f, "F5", 0xffc2, 0),
    k(F6, 0x40, "F6", 0xffc3, 0),
    k(F7, 0x41, "F7", 0xffc4, 0),
    k(F8, 0x42, "F8", 0xffc5, 0),
    k(F9, 0x43, "F9", 0xffc6, 0),
    k(F10, 0x44, "F10", 0xffc7, 0),
    k(NumLock, 0x45, "Num_Lock", 0xff7f, 0),
    k(ScrollLock, 0x46, "Scroll_Lock", 0xff14, 0),
    k(Numpad7, 0x47, "KP_7", 0xffb7, 0xff95),
    k(Numpad8, 0x48, "KP_8", 0xffb8, 0xff97),
    k(Numpad9, 0x49, "KP_9", 0xffb9, 0xff9a),
    k(NumpadSubtract, 0x4a, "KP_Subtract", 0xffad, 0),
    k(Numpad4, 0x4b, "KP_4", 0xffb4, 0xff96),
    k(Numpad5, 0x4c, "KP_5", 0xffb5, 0xff9d),
    k(Numpad6, 0x4d, "KP_6", 0xffb6, 0xff98),
    k(NumpadAdd, 0x4e, "KP_Add", 0xffab, 0),
    k(Numpad1, 0x4f, "KP_1", 0xffb1, 0xff9c),
    k(Numpad2, 0x50, "KP_2", 0xffb2, 0xff99),
    k(Numpad3, 0x51, "KP_3", 0xffb3, 0xff9b),
    k(Numpad0, 0x52, "KP_0", 0xffb0, 0xff9e),
    k(NumpadDecimal, 0x53, "KP_Decimal", 0xffae, 0xff9f),
    k(IntlBackslash, 0x56, "less", b'<' as u32 | 0x1_0000, 0),
    k(F11, 0x57, "F11", 0xffc8, 0),
    k(F12, 0x58, "F12", 0xffc9, 0),
    k(NumpadEnter, 0x9c, "KP_Enter", 0xff8d, 0),
    k(ControlRight, 0x9d, "Control_R", 0xffe4, 0),
    k(NumpadDivide, 0xb5, "KP_Divide", 0xffaf, 0),
    k(PrintScreen, 0xb7, "Print", 0xff61, 0),
    k(AltRight, 0xb8, "Alt_R", 0xffea, 0xfe03),
    k(Home, 0xc7, "Home", 0xff50, 0),
    k(ArrowUp, 0xc8, "Up", 0xff52, 0),
    k(PageUp, 0xc9, "Prior", 0xff55, 0),
    k(ArrowLeft, 0xcb, "Left", 0xff51, 0),
    k(ArrowRight, 0xcd, "Right", 0xff53, 0),
    k(End, 0xcf, "End", 0xff57, 0),
    k(ArrowDown, 0xd0, "Down", 0xff54, 0),
    k(PageDown, 0xd1, "Next", 0xff56, 0),
    k(Insert, 0xd2, "Insert", 0xff63, 0),
    k(Delete, 0xd3, "Delete", 0xffff, 0),
    k(SuperLeft, 0xdb, "Super_L", 0xffeb, 0),
    k(SuperRight, 0xdc, "Super_R", 0xffec, 0),
    k(ContextMenu, 0xdd, "Menu", 0xff67, 0),
];

/// Short names people type, beside the keysym names.
static ALIASES: &[(&str, KeyCode)] = &[
    ("ctrl", ControlLeft), ("control", ControlLeft), ("shift", ShiftLeft),
    ("alt", AltLeft), ("meta", AltLeft), ("super", SuperLeft), ("win", SuperLeft),
    ("enter", Enter), ("ret", Enter), ("esc", Escape), ("tab", Tab),
    ("bs", Backspace), ("backspace", Backspace), ("del", Delete), ("ins", Insert),
    ("pgup", PageUp), ("pageup", PageUp), ("pgdn", PageDown), ("pagedown", PageDown),
    ("up", ArrowUp), ("down", ArrowDown), ("left", ArrowLeft), ("right", ArrowRight),
    ("home", Home), ("end", End), ("caps", CapsLock), ("menu", ContextMenu),
    ("Page_Up", PageUp), ("Page_Down", PageDown), ("KP_Separator", NumpadDecimal),
];

/// The physical key for an X keysym. A shifted symbol (`!`) gives the key it
/// is on (`1`); the viewer sends Shift itself. Upper and lower case letters
/// are the same key.
pub fn from_keysym(sym: u32) -> Option<KeyCode> {
    KEYS.iter().find(|e| e.sym == sym || (e.shifted != 0 && e.shifted == sym)).map(|e| e.code)
}

/// The key that types an ASCII character on a US layout, and whether it
/// needs Shift. Newline and carriage return are Enter, tab is Tab.
pub fn char_key(ch: char) -> Option<(KeyCode, bool)> {
    match ch {
        '\n' | '\r' => return Some((Enter, false)),
        '\t' => return Some((Tab, false)),
        '\x08' => return Some((Backspace, false)),
        '\x1b' => return Some((Escape, false)),
        _ => {}
    }
    if !(' '..='~').contains(&ch) {
        return None;
    }
    let sym = ch as u32;
    KEYS.iter().find_map(|e| {
        if e.sym == sym && e.xt != 0x56 {
            Some((e.code, false))
        } else if e.shifted == sym {
            Some((e.code, true))
        } else {
            None
        }
    })
}

/// The physical key for an XT scancode as QEMU's extended key event numbers
/// it (`0xE0 nn` sent as `0x80 | nn`).
pub fn from_xt(xt: u32) -> Option<KeyCode> {
    KEYS.iter().find(|e| e.xt as u32 == xt).map(|e| e.code)
}

/// The physical key for a name: an X keysym name (`Return`, `F1`, `a`,
/// `Control_L`), an alias (`ctrl`, `enter`, `esc`, `pgup`), a single
/// character (`a`, `5`, `/`), or a winit name (`KeyA`). Case is ignored
/// where it does not change the key.
pub fn from_name(name: &str) -> Option<KeyCode> {
    if let Some(e) = KEYS.iter().find(|e| e.name == name) {
        return Some(e.code);
    }
    let mut chars = name.chars();
    if let (Some(c), None) = (chars.next(), chars.next()) {
        if let Some(code) = from_keysym(c as u32) {
            return Some(code);
        }
    }
    if let Some(e) = KEYS.iter().find(|e| e.name.eq_ignore_ascii_case(name)) {
        return Some(e.code);
    }
    if let Some((_, code)) = ALIASES.iter().find(|(a, _)| a.eq_ignore_ascii_case(name)) {
        return Some(*code);
    }
    KEYS.iter().find(|e| format!("{:?}", e.code) == name).map(|e| e.code)
}

/// Every key name `from_name` knows, for help text.
pub fn names() -> impl Iterator<Item = &'static str> {
    KEYS.iter().map(|e| e.name).chain(ALIASES.iter().map(|(a, _)| *a))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keysyms() {
        assert_eq!(from_keysym(b'a' as u32), Some(KeyA));
        assert_eq!(from_keysym(b'A' as u32), Some(KeyA));
        assert_eq!(from_keysym(b'!' as u32), Some(Digit1));
        assert_eq!(from_keysym(0xff0d), Some(Enter));
        assert_eq!(from_keysym(0xffe3), Some(ControlLeft));
        assert_eq!(from_keysym(0x1234), None);
    }

    #[test]
    fn characters() {
        assert_eq!(char_key('a'), Some((KeyA, false)));
        assert_eq!(char_key('A'), Some((KeyA, true)));
        assert_eq!(char_key('~'), Some((Backquote, true)));
        assert_eq!(char_key('|'), Some((Backslash, true)));
        assert_eq!(char_key('<'), Some((Comma, true)));
        assert_eq!(char_key(' '), Some((Space, false)));
        assert_eq!(char_key('\n'), Some((Enter, false)));
        for c in ' '..='~' {
            assert!(char_key(c).is_some(), "{c:?}");
        }
    }

    #[test]
    fn scancodes() {
        assert_eq!(from_xt(0x1e), Some(KeyA));
        assert_eq!(from_xt(0xc8), Some(ArrowUp));
        assert_eq!(from_xt(0x9d), Some(ControlRight));
    }

    #[test]
    fn names() {
        assert_eq!(from_name("Return"), Some(Enter));
        assert_eq!(from_name("return"), Some(Enter));
        assert_eq!(from_name("ctrl"), Some(ControlLeft));
        assert_eq!(from_name("F12"), Some(F12));
        assert_eq!(from_name("KeyQ"), Some(KeyQ));
        assert_eq!(from_name("/"), Some(Slash));
        assert_eq!(from_name("A"), Some(KeyA));
        assert_eq!(from_name("nope"), None);
    }

    #[test]
    fn scancodes_unique() {
        for (i, a) in KEYS.iter().enumerate() {
            for b in &KEYS[i + 1..] {
                assert_ne!(a.xt, b.xt, "{} and {}", a.name, b.name);
                assert_ne!(a.code, b.code, "{} and {}", a.name, b.name);
            }
        }
    }
}
