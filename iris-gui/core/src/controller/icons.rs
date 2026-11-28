//! Icon meanings (D11). A controller says what a control *means*; each shell
//! picks the glyph: SF Symbols on macOS 11 and later, our own SVG set in
//! `iris-gui/assets/icons/` everywhere else (and as the macOS fallback).
//!
//! Adding a meaning: add the variant, its file stem in [`Icon::svg_name`] and
//! the SVG itself, then each shell's glyph match fails to compile until it has
//! one too.

/// What an icon stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Icon {
    // Toolbar
    ShowLibrary,
    PowerOn,
    ShutDownGuest,
    PowerOff,
    Restart,
    /// A hard reset, like the machine's reset switch.
    Reset,
    SaveState,
    SavedStates,
    Screenshot,
    InsertDisc,
    BrowseFiles,
    DetachDisplay,
    FullScreen,
    SerialConsole,
    MachineSettings,
    // Library
    NewMachine,
    NewFolder,
    Folder,
    // Status bar devices
    Disk,
    CdRom,
    Network,
    Audio,
    Camera,
    InputCapture,
    DisplayScale,
    // Generic
    Warning,
    Error,
    Info,
    Delete,
    Duplicate,
    Rename,
    Restore,
    Export,
    Import,
    Edit,
    Permissions,
    Check,
    Up,
    Refresh,
    Cancel,
}

impl Icon {
    /// The file stem of this icon in `iris-gui/assets/icons/` (`<stem>.svg`).
    pub fn svg_name(self) -> &'static str {
        match self {
            Icon::ShowLibrary => "sidebar",
            Icon::PowerOn => "power-on",
            Icon::ShutDownGuest => "shut-down",
            Icon::PowerOff => "power-off",
            Icon::Restart => "restart",
            Icon::Reset => "reset",
            Icon::SaveState => "save-state",
            Icon::SavedStates => "saved-states",
            Icon::Screenshot => "screenshot",
            Icon::InsertDisc => "insert-disc",
            Icon::BrowseFiles => "browse-files",
            Icon::DetachDisplay => "detach",
            Icon::FullScreen => "full-screen",
            Icon::SerialConsole => "terminal",
            Icon::MachineSettings => "settings",
            Icon::NewMachine => "new-machine",
            Icon::NewFolder => "new-folder",
            Icon::Folder => "folder",
            Icon::Disk => "disk",
            Icon::CdRom => "cdrom",
            Icon::Network => "network",
            Icon::Audio => "audio",
            Icon::Camera => "camera",
            Icon::InputCapture => "capture",
            Icon::DisplayScale => "scale",
            Icon::Warning => "warning",
            Icon::Error => "error",
            Icon::Info => "info",
            Icon::Delete => "delete",
            Icon::Duplicate => "duplicate",
            Icon::Rename => "rename",
            Icon::Restore => "restore",
            Icon::Export => "export",
            Icon::Import => "import",
            Icon::Edit => "edit",
            Icon::Permissions => "permissions",
            Icon::Check => "check",
            Icon::Up => "up",
            Icon::Refresh => "refresh",
            Icon::Cancel => "cancel",
        }
    }

    /// The SVG itself: 24x24 line art, black on transparent, so a shell can
    /// use it as a mask or template and colour it (the machine pictures in
    /// [`crate::controller::profile::MachineIcon::svg`] work the same way, on
    /// a 64x64 grid because they are drawn much larger).
    pub fn svg(self) -> &'static str {
        macro_rules! art {
            ($stem:literal) => {
                include_str!(concat!("../../../assets/icons/", $stem, ".svg"))
            };
        }
        match self {
            Icon::ShowLibrary => art!("sidebar"),
            Icon::PowerOn => art!("power-on"),
            Icon::ShutDownGuest => art!("shut-down"),
            Icon::PowerOff => art!("power-off"),
            Icon::Restart => art!("restart"),
            Icon::Reset => art!("reset"),
            Icon::SaveState => art!("save-state"),
            Icon::SavedStates => art!("saved-states"),
            Icon::Screenshot => art!("screenshot"),
            Icon::InsertDisc => art!("insert-disc"),
            Icon::BrowseFiles => art!("browse-files"),
            Icon::DetachDisplay => art!("detach"),
            Icon::FullScreen => art!("full-screen"),
            Icon::SerialConsole => art!("terminal"),
            Icon::MachineSettings => art!("settings"),
            Icon::NewMachine => art!("new-machine"),
            Icon::NewFolder => art!("new-folder"),
            Icon::Folder => art!("folder"),
            Icon::Disk => art!("disk"),
            Icon::CdRom => art!("cdrom"),
            Icon::Network => art!("network"),
            Icon::Audio => art!("audio"),
            Icon::Camera => art!("camera"),
            Icon::InputCapture => art!("capture"),
            Icon::DisplayScale => art!("scale"),
            Icon::Warning => art!("warning"),
            Icon::Error => art!("error"),
            Icon::Info => art!("info"),
            Icon::Delete => art!("delete"),
            Icon::Duplicate => art!("duplicate"),
            Icon::Rename => art!("rename"),
            Icon::Restore => art!("restore"),
            Icon::Export => art!("export"),
            Icon::Import => art!("import"),
            Icon::Edit => art!("edit"),
            Icon::Permissions => art!("permissions"),
            Icon::Check => art!("check"),
            Icon::Up => art!("up"),
            Icon::Refresh => art!("refresh"),
            Icon::Cancel => art!("cancel"),
        }
    }

    /// Every meaning, for tests and icon-set checks.
    pub const ALL: &'static [Icon] = &[
        Icon::ShowLibrary, Icon::PowerOn, Icon::ShutDownGuest, Icon::PowerOff, Icon::Restart, Icon::Reset,
        Icon::SaveState, Icon::SavedStates, Icon::Screenshot, Icon::InsertDisc, Icon::BrowseFiles,
        Icon::DetachDisplay, Icon::FullScreen, Icon::SerialConsole, Icon::MachineSettings,
        Icon::NewMachine, Icon::NewFolder, Icon::Folder, Icon::Disk, Icon::CdRom, Icon::Network,
        Icon::Audio, Icon::Camera, Icon::InputCapture, Icon::DisplayScale, Icon::Warning,
        Icon::Error, Icon::Info, Icon::Delete, Icon::Duplicate, Icon::Rename, Icon::Restore, Icon::Export,
        Icon::Import, Icon::Edit, Icon::Permissions, Icon::Check, Icon::Up, Icon::Refresh,
        Icon::Cancel,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_meaning_has_its_art() {
        for icon in Icon::ALL {
            let svg = icon.svg();
            assert!(svg.starts_with("<svg"), "{} is not an SVG", icon.svg_name());
            assert!(svg.contains("viewBox=\"0 0 24 24\""), "{} is not on the 24x24 grid", icon.svg_name());
            // Tinted by the shell, so the art carries no colour of its own.
            assert!(!svg.contains("stroke=\"#0") || svg.contains("stroke=\"#000\""), "{} is coloured", icon.svg_name());
        }
    }

    #[test]
    fn svg_names_are_unique() {
        let mut names: Vec<_> = Icon::ALL.iter().map(|i| i.svg_name()).collect();
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(n, names.len());
    }
}
