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
    fn svg_names_are_unique() {
        let mut names: Vec<_> = Icon::ALL.iter().map(|i| i.svg_name()).collect();
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(n, names.len());
    }
}
