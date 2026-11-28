//! Stable widget ids. Every shell sets them as accessibility identifiers
//! (AppKit `setAccessibilityIdentifier`, egui `Id::new`), so one scenario
//! test can drive every view. Never rename one without updating the tests that
//! find controls by it.

/// The main window's toolbar.
pub mod toolbar {
    pub const SHOW_LIBRARY: &str = "toolbar.show-library";
    pub const POWER_ON: &str = "toolbar.power-on";
    pub const SHUT_DOWN_GUEST: &str = "toolbar.shut-down-guest";
    pub const POWER_OFF: &str = "toolbar.power-off";
    /// Restart IRIX cleanly; Reset (in its menu) is the hard one.
    pub const RESTART: &str = "toolbar.restart";
    pub const RESET: &str = "toolbar.reset";
    pub const SAVE_STATE: &str = "toolbar.save-state";
    pub const SAVED_STATES: &str = "toolbar.saved-states";
    pub const SCREENSHOT: &str = "toolbar.screenshot";
    pub const INSERT_DISC: &str = "toolbar.insert-disc";
    pub const BROWSE_FILES: &str = "toolbar.browse-files";
    pub const DETACH_DISPLAY: &str = "toolbar.detach-display";
    pub const FULL_SCREEN: &str = "toolbar.full-screen";
    pub const SERIAL_CONSOLE: &str = "toolbar.serial-console";
    pub const MACHINE_SETTINGS: &str = "toolbar.machine-settings";
    pub const NEW_MACHINE: &str = "toolbar.new-machine";
    pub const DUPLICATE: &str = "toolbar.duplicate";
    pub const DELETE: &str = "toolbar.delete";
    /// The disc menu: insert, change or eject, for each CD-ROM.
    pub const DISC: &str = "toolbar.disc";
    pub const CAPTURE: &str = "toolbar.capture";
    /// Not an item: the flexible space between the left and right groups.
    pub const GAP: &str = "toolbar.gap";
    /// The selected machine's name and model, right of the library button
    /// (a shell that shows them in its title bar may leave it out).
    pub const TITLE: &str = "toolbar.title";
}

/// The machine library on the left.
pub mod library {
    pub const LIST: &str = "library.list";
    pub const NEW_MACHINE: &str = "library.new-machine";
    pub const NEW_FOLDER: &str = "library.new-folder";
}

/// The status bar under the content.
pub mod status {
    pub const RUN_STATE: &str = "status.run-state";
    pub const EXECUTOR: &str = "status.executor";
    pub const TOAST: &str = "status.toast";
    pub const NETWORK: &str = "status.network";
    pub const SERIAL: &str = "status.serial";
    pub const AUDIO: &str = "status.audio";
    pub const CAMERA: &str = "status.camera";
    pub const CAPTURE: &str = "status.capture";
    pub const SCALE: &str = "status.scale";
    pub const MIPS: &str = "status.mips";
    pub const HZ: &str = "status.hz";

    /// A disk light: `status.disk.<controller>.<id>`.
    pub fn disk(controller: usize, id: u8) -> String {
        format!("status.disk.{controller}.{id}")
    }
}

/// Pages of the machine settings, for `Effect::OpenSettings`.
pub mod settings {
    pub const GENERAL: &str = "settings.general";
    pub const DISKS: &str = "settings.disks";
    pub const NETWORK: &str = "settings.network";
    pub const DISPLAY: &str = "settings.display";
}

/// Per-item buttons on the summary page; the item follows the prefix.
pub mod item {
    pub const RESTORE_STATE: &str = "state.restore:";
    pub const RENAME_STATE: &str = "state.rename:";
    pub const DELETE_STATE: &str = "state.delete:";
    /// Followed by the disk's `[scsi.KEY]` key.
    pub const BROWSE_DISK: &str = "browse:";
    /// Followed by the CD-ROM's `[scsi.KEY]` key.
    pub const INSERT_DISC: &str = "disc.insert:";
    /// Followed by a machine's name: select it and power it on.
    pub const START_MACHINE: &str = "machine.start:";
    /// Each followed by the CD-ROM's key.
    pub const EJECT_DISC: &str = "disc.eject:";
    pub const MOUNT_DISC: &str = "disc.mount:";
    /// Each followed by the disk's key: merge or discard its copy-on-write changes.
    pub const COMMIT_COW: &str = "cow.commit:";
    pub const DISCARD_COW: &str = "cow.discard:";
    /// Followed by the disk's key: convert its compressed CHD to a quicker format.
    pub const CONVERT_DISK: &str = "disk.convert:";
    /// Followed by the disk's key: keep it as it is, no more suggestions.
    pub const DECLINE_CONVERT: &str = "disk.keep:";
    /// `on:<machine>:<command id>`: select the machine, then run the command
    /// (a library row's context menu). Machine names never hold a colon.
    pub const ON_MACHINE: &str = "on:";

    /// The id that runs `command` on `machine`.
    pub fn on_machine(machine: &str, command: &str) -> String {
        format!("{ON_MACHINE}{machine}:{command}")
    }

    pub fn id(prefix: &str, item: impl std::fmt::Display) -> String {
        format!("{prefix}{item}")
    }
}
