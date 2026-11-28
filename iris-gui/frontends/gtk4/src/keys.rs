//! Linux evdev scan codes (physical positions) to the guest keys the PS/2
//! layer takes, and GDK key values to the shortcut table's keys.
//!
//! Physical, never layout-translated: the guest applies its own `keybd=`
//! layout (clanker/gui/keyboard-layout-send-physical-keys-not-logical.md).
//! GDK reports the X11 keycode on both backends, which is the evdev code
//! plus 8. The application's own shortcuts are GTK accelerators, built from
//! the controller's key map in `menu.rs`, so nothing here looks at key
//! values: every key over the display is physical and goes to the guest.

use iris_gui_core::controller::display::GuestKey as K;

/// X11 keycodes are evdev codes offset by this.
const EVDEV_OFFSET: u32 = 8;

/// The guest key at GDK hardware keycode `code`.
pub fn guest_key(code: u32) -> Option<K> {
    let evdev = code.checked_sub(EVDEV_OFFSET)?;
    Some(match evdev {
        1 => K::Escape,
        2 => K::Digit1,
        3 => K::Digit2,
        4 => K::Digit3,
        5 => K::Digit4,
        6 => K::Digit5,
        7 => K::Digit6,
        8 => K::Digit7,
        9 => K::Digit8,
        10 => K::Digit9,
        11 => K::Digit0,
        12 => K::Minus,
        13 => K::Equal,
        14 => K::Backspace,
        15 => K::Tab,
        16 => K::KeyQ,
        17 => K::KeyW,
        18 => K::KeyE,
        19 => K::KeyR,
        20 => K::KeyT,
        21 => K::KeyY,
        22 => K::KeyU,
        23 => K::KeyI,
        24 => K::KeyO,
        25 => K::KeyP,
        26 => K::BracketLeft,
        27 => K::BracketRight,
        28 => K::Enter,
        29 => K::ControlLeft,
        30 => K::KeyA,
        31 => K::KeyS,
        32 => K::KeyD,
        33 => K::KeyF,
        34 => K::KeyG,
        35 => K::KeyH,
        36 => K::KeyJ,
        37 => K::KeyK,
        38 => K::KeyL,
        39 => K::Semicolon,
        40 => K::Quote,
        41 => K::Backquote,
        42 => K::ShiftLeft,
        43 => K::Backslash,
        44 => K::KeyZ,
        45 => K::KeyX,
        46 => K::KeyC,
        47 => K::KeyV,
        48 => K::KeyB,
        49 => K::KeyN,
        50 => K::KeyM,
        51 => K::Comma,
        52 => K::Period,
        53 => K::Slash,
        54 => K::ShiftRight,
        55 => K::NumpadMultiply,
        56 => K::AltLeft,
        57 => K::Space,
        58 => K::CapsLock,
        59 => K::F1,
        60 => K::F2,
        61 => K::F3,
        62 => K::F4,
        63 => K::F5,
        64 => K::F6,
        65 => K::F7,
        66 => K::F8,
        67 => K::F9,
        68 => K::F10,
        69 => K::NumLock,
        70 => K::ScrollLock,
        71 => K::Numpad7,
        72 => K::Numpad8,
        73 => K::Numpad9,
        74 => K::NumpadSubtract,
        75 => K::Numpad4,
        76 => K::Numpad5,
        77 => K::Numpad6,
        78 => K::NumpadAdd,
        79 => K::Numpad1,
        80 => K::Numpad2,
        81 => K::Numpad3,
        82 => K::Numpad0,
        83 => K::NumpadDecimal,
        85 => K::Lang5,
        86 => K::IntlBackslash,
        87 => K::F11,
        88 => K::F12,
        89 => K::IntlRo,
        90 => K::Lang3,
        91 => K::Lang4,
        92 => K::Convert,
        93 => K::KanaMode,
        94 => K::NonConvert,
        96 => K::NumpadEnter,
        97 => K::ControlRight,
        98 => K::NumpadDivide,
        99 => K::PrintScreen,
        100 => K::AltRight,
        102 => K::Home,
        103 => K::ArrowUp,
        104 => K::PageUp,
        105 => K::ArrowLeft,
        106 => K::ArrowRight,
        107 => K::End,
        108 => K::ArrowDown,
        109 => K::PageDown,
        110 => K::Insert,
        111 => K::Delete,
        113 => K::AudioVolumeMute,
        114 => K::AudioVolumeDown,
        115 => K::AudioVolumeUp,
        116 => K::Power,
        117 => K::NumpadEqual,
        119 => K::Pause,
        121 => K::NumpadComma,
        122 => K::Lang1,
        123 => K::Lang2,
        124 => K::IntlYen,
        125 => K::SuperLeft,
        126 => K::SuperRight,
        127 => K::ContextMenu,
        183 => K::F13,
        184 => K::F14,
        185 => K::F15,
        186 => K::F16,
        187 => K::F17,
        188 => K::F18,
        189 => K::F19,
        190 => K::F20,
        191 => K::F21,
        192 => K::F22,
        193 => K::F23,
        194 => K::F24,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_home_row_is_where_it_should_be() {
        // evdev 30..38 are A S D F G H J K L, plus the X11 offset.
        assert_eq!(guest_key(30 + EVDEV_OFFSET), Some(K::KeyA));
        assert_eq!(guest_key(38 + EVDEV_OFFSET), Some(K::KeyL));
        assert_eq!(guest_key(1 + EVDEV_OFFSET), Some(K::Escape));
        // Below the offset is not a key at all.
        assert_eq!(guest_key(0), None);
    }
}
