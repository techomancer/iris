//! Shortcuts as data: named commands and their default chords.
//!
//! Every view translates its native key event into a [`Key`] plus [`Mods`] and
//! asks [`KeyMap::lookup`] which [`Command`] it means; no view decides what a
//! key does. `primary` in a chord is Cmd on macOS and Ctrl elsewhere, so one
//! entry serves every platform; [`OnPlatform`] limits an entry to one side.
//!
//! Shortcuts act only while the guest display does not have input captured and
//! no text field has focus. While captured, every key goes to the guest except
//! the chords in [`CaptureChords`], which the display controller reads.
//!
//! The chords for the menu bar are proposals (plan, "Menu bar (macOS)"); this
//! table is the one place to change them.

use std::fmt;

/// A key, independent of any toolkit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    F(u8),
    /// A printable key, compared case-insensitively (`Shift` is a modifier).
    Char(char),
    Tab,
    Enter,
    Escape,
    Backspace,
    Delete,
    Space,
    Up,
    Down,
    Left,
    Right,
    PageUp,
    PageDown,
    Home,
    End,
}

/// The modifiers held when a key was pressed, as the platform reports them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mods {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    /// Cmd on macOS (the Windows / Super key elsewhere, which no binding uses).
    pub cmd: bool,
}

/// Which platforms a chord applies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnPlatform {
    All,
    MacOnly,
    PcOnly,
}

/// A key plus the modifiers a binding requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chord {
    pub key: Key,
    /// Cmd on macOS, Ctrl elsewhere.
    pub primary: bool,
    /// Ctrl on macOS when `primary` is Cmd (the Ctrl+Cmd+F full-screen chord).
    pub mac_ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    pub on: OnPlatform,
}

impl Chord {
    const fn plain(key: Key) -> Self {
        Chord { key, primary: false, mac_ctrl: false, alt: false, shift: false, on: OnPlatform::All }
    }
    const fn primary(key: Key) -> Self {
        Chord { primary: true, ..Chord::plain(key) }
    }
    const fn primary_shift(key: Key) -> Self {
        Chord { primary: true, shift: true, ..Chord::plain(key) }
    }
    const fn mac(self) -> Self {
        Chord { on: OnPlatform::MacOnly, ..self }
    }
    const fn pc(self) -> Self {
        Chord { on: OnPlatform::PcOnly, ..self }
    }

    fn applies(&self, mac: bool) -> bool {
        match self.on {
            OnPlatform::All => true,
            OnPlatform::MacOnly => mac,
            OnPlatform::PcOnly => !mac,
        }
    }

    fn matches(&self, key: Key, m: Mods, mac: bool) -> bool {
        if !self.applies(mac) || !same_key(self.key, key) {
            return false;
        }
        let (primary, ctrl) = if mac { (m.cmd, m.ctrl) } else { (m.ctrl, false) };
        primary == self.primary && ctrl == self.mac_ctrl && m.alt == self.alt && m.shift == self.shift
            && (mac || !m.cmd)
    }

    /// The chord as the platform writes it in a menu or tooltip.
    pub fn display(&self, mac: bool) -> String {
        let mut s = String::new();
        if mac {
            if self.mac_ctrl {
                s.push_str("Ctrl+");
            }
            if self.alt {
                s.push_str("Option+");
            }
            if self.shift {
                s.push_str("Shift+");
            }
            if self.primary {
                s.push_str("Cmd+");
            }
        } else {
            if self.primary {
                s.push_str("Ctrl+");
            }
            if self.alt {
                s.push_str("Alt+");
            }
            if self.shift {
                s.push_str("Shift+");
            }
        }
        s.push_str(&self.key.to_string());
        s
    }
}

fn same_key(a: Key, b: Key) -> bool {
    match (a, b) {
        (Key::Char(x), Key::Char(y)) => x.eq_ignore_ascii_case(&y),
        _ => a == b,
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Key::F(n) => write!(f, "F{n}"),
            Key::Char(',') => f.write_str(","),
            Key::Char(c) => write!(f, "{}", c.to_ascii_uppercase()),
            Key::Tab => f.write_str("Tab"),
            Key::Enter => f.write_str("Enter"),
            Key::Escape => f.write_str("Esc"),
            Key::Backspace => f.write_str("Backspace"),
            Key::Delete => f.write_str("Delete"),
            Key::Space => f.write_str("Space"),
            Key::Up => f.write_str("Up"),
            Key::Down => f.write_str("Down"),
            Key::Left => f.write_str("Left"),
            Key::Right => f.write_str("Right"),
            Key::PageUp => f.write_str("PageUp"),
            Key::PageDown => f.write_str("PageDown"),
            Key::Home => f.write_str("Home"),
            Key::End => f.write_str("End"),
        }
    }
}

/// What an application shortcut asks for. The app maps each to the owning
/// controller's action; a menu item shows the chord from [`KeyMap::chord`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Command {
    NewMachine,
    NewFolder,
    AppSettings,
    Close,
    Quit,
    ShowLibrary,
    FullScreen,
    DetachDisplay,
    SerialConsole,
    PowerOn,
    ShutDownGuest,
    SaveState,
    Screenshot,
    CaptureInput,
    MachineSettings,
    BrowseFiles,
}

impl Command {
    pub const ALL: &'static [Command] = &[
        Command::NewMachine, Command::NewFolder, Command::AppSettings, Command::Close,
        Command::Quit, Command::ShowLibrary, Command::FullScreen, Command::DetachDisplay,
        Command::SerialConsole, Command::PowerOn, Command::ShutDownGuest, Command::SaveState,
        Command::Screenshot, Command::CaptureInput, Command::MachineSettings, Command::BrowseFiles,
    ];
}

/// The default table. Power Off and Restart deliberately have no chord: they
/// can lose data in a running guest, so they are a click and a confirmation.
const DEFAULTS: &[(Command, Chord)] = &[
    (Command::NewMachine, Chord::primary(Key::Char('n'))),
    (Command::NewFolder, Chord::primary_shift(Key::Char('n'))),
    (Command::AppSettings, Chord::primary(Key::Char(',')).mac()),
    (Command::Close, Chord::primary(Key::Char('w'))),
    (Command::Quit, Chord::primary(Key::Char('q'))),
    // On the Mac (accepted 2026-11-27): Ctrl+Cmd+S for the library and
    // Ctrl+Cmd+F alone for full screen (F11 is Show Desktop there), Option+Cmd+C
    // for the serial console, and no chord for Shut Down or Detach.
    (Command::ShowLibrary, Chord::primary_shift(Key::Char('l')).pc()),
    (Command::ShowLibrary, Chord { mac_ctrl: true, ..Chord::primary(Key::Char('s')) }.mac()),
    (Command::FullScreen, Chord::plain(Key::F(11)).pc()),
    (Command::FullScreen, Chord { mac_ctrl: true, ..Chord::primary(Key::Char('f')) }.mac()),
    (Command::DetachDisplay, Chord::primary_shift(Key::Char('d')).pc()),
    (Command::SerialConsole, Chord::primary_shift(Key::Char('t')).pc()),
    (Command::SerialConsole, Chord { alt: true, ..Chord::primary(Key::Char('c')) }.mac()),
    (Command::PowerOn, Chord::primary(Key::Char('r'))),
    (Command::ShutDownGuest, Chord::primary_shift(Key::Char('h')).pc()),
    (Command::SaveState, Chord::primary(Key::Char('s'))),
    (Command::Screenshot, Chord::primary_shift(Key::Char('s'))),
    (Command::CaptureInput, Chord::primary(Key::Char('g'))),
    (Command::MachineSettings, Chord::primary(Key::Char('e'))),
    (Command::BrowseFiles, Chord::primary_shift(Key::Char('b'))),
];

/// The shortcut table in force.
#[derive(Debug, Clone)]
pub struct KeyMap {
    entries: Vec<(Command, Chord)>,
    mac: bool,
}

impl Default for KeyMap {
    fn default() -> Self {
        Self::for_platform(cfg!(target_os = "macos"))
    }
}

impl KeyMap {
    pub fn for_platform(mac: bool) -> Self {
        Self { entries: DEFAULTS.to_vec(), mac }
    }

    /// The command a key press means, if any.
    pub fn lookup(&self, key: Key, mods: Mods) -> Option<Command> {
        self.entries.iter().find(|(_, c)| c.matches(key, mods, self.mac)).map(|(cmd, _)| *cmd)
    }

    /// The first chord bound to `cmd` on this platform, for a menu item.
    pub fn chord(&self, cmd: Command) -> Option<Chord> {
        self.entries.iter().find(|(c, ch)| *c == cmd && ch.applies(self.mac)).map(|(_, ch)| *ch)
    }

    /// The chord as text for a tooltip or menu, or empty when unbound.
    pub fn chord_text(&self, cmd: Command) -> String {
        self.chord(cmd).map(|c| c.display(self.mac)).unwrap_or_default()
    }
}

/// A physical modifier key, left or right, as the guest-input rules need them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModKey {
    ControlLeft,
    ControlRight,
    AltLeft,
    AltRight,
    ShiftLeft,
    ShiftRight,
    SuperLeft,
    SuperRight,
}

/// The chords that act while the guest has input captured.
///
/// - Release: hold the two release keys and press nothing else; fires on
///   key-up, so the same modifiers with another key (Ctrl+Alt+F11) still reach
///   the guest. Left Ctrl+Alt, or left Option+Cmd on macOS.
/// - Release fallback: Ctrl+Alt+Esc.
/// - Plain F11 toggles full screen; Ctrl+Alt+F11 sends a bare F11 to the guest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureChords {
    pub release: [ModKey; 2],
}

impl CaptureChords {
    pub fn for_platform(mac: bool) -> Self {
        if mac {
            Self { release: [ModKey::AltLeft, ModKey::SuperLeft] }
        } else {
            Self { release: [ModKey::ControlLeft, ModKey::AltLeft] }
        }
    }

    /// The release chord as the user reads it.
    pub fn release_text(mac: bool) -> &'static str {
        if mac {
            "Option+Cmd"
        } else {
            "Ctrl+Alt"
        }
    }
}

impl Default for CaptureChords {
    fn default() -> Self {
        Self::for_platform(cfg!(target_os = "macos"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CMD: Mods = Mods { ctrl: false, alt: false, shift: false, cmd: true };
    const CTRL: Mods = Mods { ctrl: true, alt: false, shift: false, cmd: false };

    #[test]
    fn primary_is_cmd_on_mac_and_ctrl_elsewhere() {
        let mac = KeyMap::for_platform(true);
        let pc = KeyMap::for_platform(false);
        assert_eq!(mac.lookup(Key::Char('N'), CMD), Some(Command::NewMachine));
        assert_eq!(mac.lookup(Key::Char('n'), CTRL), None);
        assert_eq!(pc.lookup(Key::Char('n'), CTRL), Some(Command::NewMachine));
        assert_eq!(pc.lookup(Key::Char('n'), CMD), None);
    }

    #[test]
    fn full_screen_is_f11_on_a_pc_and_ctrl_cmd_f_alone_on_mac() {
        let mac = KeyMap::for_platform(true);
        assert_eq!(mac.lookup(Key::F(11), Mods::default()), None, "F11 is Show Desktop on a Mac");
        let ctrl_cmd = Mods { ctrl: true, cmd: true, ..Mods::default() };
        assert_eq!(mac.lookup(Key::Char('f'), ctrl_cmd), Some(Command::FullScreen));
        let pc = KeyMap::for_platform(false);
        assert_eq!(pc.lookup(Key::F(11), Mods::default()), Some(Command::FullScreen));
        assert_eq!(pc.chord(Command::AppSettings), None);
    }

    /// The Mac chords accepted on 2026-11-27.
    #[test]
    fn the_mac_menu_chords() {
        let mac = KeyMap::for_platform(true);
        assert_eq!(mac.chord_text(Command::ShowLibrary), "Ctrl+Cmd+S");
        assert_eq!(mac.chord_text(Command::SerialConsole), "Option+Cmd+C");
        assert_eq!(mac.chord(Command::ShutDownGuest), None);
        assert_eq!(mac.chord(Command::DetachDisplay), None);
        let pc = KeyMap::for_platform(false);
        assert_eq!(pc.chord_text(Command::ShowLibrary), "Ctrl+Shift+L");
        assert_eq!(pc.chord_text(Command::SerialConsole), "Ctrl+Shift+T");
    }

    #[test]
    fn no_two_commands_share_a_chord() {
        for mac in [false, true] {
            let applicable: Vec<_> = DEFAULTS.iter().filter(|(_, c)| c.applies(mac)).collect();
            for (i, (a, ca)) in applicable.iter().enumerate() {
                for (b, cb) in &applicable[i + 1..] {
                    assert!(!(ca == cb && a != b), "{a:?} and {b:?} share {}", ca.display(mac));
                }
            }
        }
    }

    #[test]
    fn chord_text() {
        assert_eq!(KeyMap::for_platform(true).chord_text(Command::Screenshot), "Shift+Cmd+S");
        assert_eq!(KeyMap::for_platform(false).chord_text(Command::Screenshot), "Ctrl+Shift+S");
        assert_eq!(KeyMap::for_platform(false).chord_text(Command::FullScreen), "F11");
    }
}
