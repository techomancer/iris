//! Toolbar and menu commands by widget id: each one's icon and tooltip
//! (D11), whether it can run now and why not, and what it does. Every shell
//! routes its toolbar items and menu items through here, so the meaning of a
//! control is decided once.

use super::app::{AppAction, AppController};
use super::display::{DisplayAction, GuestChord};
use super::icons::Icon;
use super::ids::toolbar as tb;
use super::keys::{Command, KeyMap};
use super::labels::Tip;
use super::serial::SerialAction;
use super::session::SessionAction;
use super::{Controller, Effect, Enabled};

/// Menu item ids (the toolbar's are in `ids::toolbar`).
pub mod menu {
    pub const ABOUT: &str = "menu.about";
    pub const APP_SETTINGS: &str = "menu.app-settings";
    pub const QUIT: &str = "menu.quit";
    pub const NEW_MACHINE: &str = "menu.new-machine";
    pub const NEW_FOLDER: &str = "menu.new-folder";
    pub const DUPLICATE: &str = "menu.duplicate";
    pub const RENAME: &str = "menu.rename";
    pub const DELETE: &str = "menu.delete";
    pub const SHOW_IN_FINDER: &str = "menu.show-in-finder";
    pub const OPEN_CONFIG: &str = "menu.open-config";
    pub const CLOSE: &str = "menu.close";
    pub const CAPTURE: &str = "menu.capture";
    pub const SEND_F11: &str = "menu.send-f11";
    pub const SEND_CAD: &str = "menu.send-ctrl-alt-del";
    pub const SEND_ESC: &str = "menu.send-esc";
    pub const SYNC_DISKS: &str = "menu.sync-disks";
    /// App Store builds: grant the sandbox a folder of disk images, so a
    /// disk's changes can be folded in beside it.
    pub const GRANT_FOLDER: &str = "menu.grant-folder";
    /// The display's size: one guest pixel to a point, scaled keeping the
    /// aspect ratio, or stretched (app-wide, as Preferences sets it).
    pub const VIEW_ACTUAL: &str = "menu.view-actual-size";
    pub const VIEW_FIT: &str = "menu.view-fit";
    pub const VIEW_STRETCH: &str = "menu.view-stretch";
    /// Size the display's window to the picture, as large as it is drawn
    /// now: no black around it and nothing to scroll.
    pub const VIEW_FIT_WINDOW: &str = "menu.view-fit-window";
    pub const CHECK_NETWORK: &str = "menu.check-network";
    pub const BENCHMARK: &str = "menu.benchmark";
    pub const CAMERA_TEST: &str = "menu.camera-test";
    pub const DIAGNOSTICS: &str = "menu.diagnostics";
    pub const RESET_NVRAM: &str = "menu.reset-nvram";
    pub const SET_MAC: &str = "menu.set-mac";
    pub const GUIDE: &str = "menu.guide";
    pub const REPORT_ISSUE: &str = "menu.report-issue";
}

/// Commands that change only the shell's own presentation; the shell does them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellCommand {
    ToggleLibrary,
    /// Open the machine settings sheet.
    Settings,
    /// Show the saved-states list.
    SavedStates,
    /// Pop up the disc menu (insert, change, eject) for the machine's CD-ROMs.
    DiscMenu,
    NewMachine,
    About,
    AppSettings,
    Benchmark,
    Diagnostics,
    CloseWindow,
    /// Size the window that shows the display to its picture.
    FitWindow,
}

/// What a command id did.
#[derive(Debug, Default)]
pub struct Routed {
    pub effects: Vec<Effect>,
    pub shell: Option<ShellCommand>,
}

/// The icon and tooltip of a toolbar id, with its shortcut.
pub fn tip(id: &str, keys: &KeyMap) -> Option<Tip> {
    let (icon, text, cmd) = match id {
        tb::SHOW_LIBRARY => (Icon::ShowLibrary, t!("tb-show-library"), Some(Command::ShowLibrary)),
        tb::POWER_ON => (Icon::PowerOn, t!("tb-power-on"), Some(Command::PowerOn)),
        tb::SHUT_DOWN_GUEST => (Icon::ShutDownGuest, t!("tb-shut-down"), Some(Command::ShutDownGuest)),
        tb::POWER_OFF => (Icon::PowerOff, t!("tb-power-off"), None),
        tb::RESTART => (Icon::Restart, t!("tb-restart"), None),
        tb::RESET => (Icon::Reset, t!("tb-reset"), None),
        tb::SAVE_STATE => (Icon::SaveState, t!("tb-save-state"), Some(Command::SaveState)),
        tb::SAVED_STATES => (Icon::SavedStates, t!("tb-saved-states"), None),
        tb::SCREENSHOT => (Icon::Screenshot, t!("tb-screenshot"), Some(Command::Screenshot)),
        tb::INSERT_DISC => (Icon::InsertDisc, t!("tb-insert-disc"), None),
        tb::BROWSE_FILES => (Icon::BrowseFiles, t!("tb-browse-files"), Some(Command::BrowseFiles)),
        tb::DETACH_DISPLAY => (Icon::DetachDisplay, t!("tb-detach"), Some(Command::DetachDisplay)),
        tb::FULL_SCREEN => (Icon::FullScreen, t!("tb-full-screen"), Some(Command::FullScreen)),
        tb::SERIAL_CONSOLE => (Icon::SerialConsole, t!("tb-serial-console"), Some(Command::SerialConsole)),
        tb::MACHINE_SETTINGS => (Icon::MachineSettings, t!("tb-settings"), Some(Command::MachineSettings)),
        tb::NEW_MACHINE => (Icon::NewMachine, t!("tb-new-machine"), Some(Command::NewMachine)),
        tb::DUPLICATE => (Icon::Duplicate, t!("tb-duplicate"), None),
        tb::DELETE => (Icon::Delete, t!("tb-delete"), None),
        tb::DISC => (Icon::InsertDisc, t!("tb-disc"), None),
        tb::CAPTURE => (Icon::InputCapture, t!("tb-capture"), Some(Command::CaptureInput)),
        _ => return None,
    };
    let mut tip = Tip::new(icon, text);
    if let Some(c) = cmd {
        tip.shortcut = keys.chord_text(c);
    }
    Some(tip)
}

/// The menu under a toolbar button: the forced version of a graceful power
/// action (Shut Down: Power Off; Restart: Reset), empty for others. Whether
/// an entry can run is `AppController::command_enabled`, asked when it shows.
pub fn toolbar_menu(id: &str) -> Vec<MenuEntry> {
    let entry = |cmd: &str, label: String, tip: String| MenuEntry { id: cmd.to_string(), label, tooltip: Some(tip), enabled: Enabled::Yes };
    match id {
        tb::SHUT_DOWN_GUEST => vec![entry(tb::POWER_OFF, t!("tb-power-off"), t!("tb-power-off-tip"))],
        tb::RESTART => vec![entry(tb::RESET, t!("tb-reset"), t!("tb-reset-tip"))],
        _ => Vec::new(),
    }
}

/// The command id a key-map command runs.
pub fn id_of(cmd: Command) -> &'static str {
    match cmd {
        Command::NewMachine => menu::NEW_MACHINE,
        Command::NewFolder => menu::NEW_FOLDER,
        Command::AppSettings => menu::APP_SETTINGS,
        Command::Close => menu::CLOSE,
        Command::Quit => menu::QUIT,
        Command::ShowLibrary => tb::SHOW_LIBRARY,
        Command::FullScreen => tb::FULL_SCREEN,
        Command::DetachDisplay => tb::DETACH_DISPLAY,
        Command::SerialConsole => tb::SERIAL_CONSOLE,
        Command::PowerOn => tb::POWER_ON,
        Command::ShutDownGuest => tb::SHUT_DOWN_GUEST,
        Command::SaveState => tb::SAVE_STATE,
        Command::Screenshot => tb::SCREENSHOT,
        Command::CaptureInput => menu::CAPTURE,
        Command::MachineSettings => tb::MACHINE_SETTINGS,
        Command::BrowseFiles => tb::BROWSE_FILES,
    }
}

fn session_of(id: &str) -> Option<SessionAction> {
    Some(match id {
        tb::POWER_ON => SessionAction::PowerOn,
        tb::SHUT_DOWN_GUEST => SessionAction::ShutDownGuest,
        tb::POWER_OFF => SessionAction::PowerOff,
        tb::RESTART => SessionAction::RebootGuest,
        tb::RESET => SessionAction::Restart,
        tb::SAVE_STATE => SessionAction::AskSaveState,
        tb::SCREENSHOT => SessionAction::Screenshot,
        menu::SYNC_DISKS => SessionAction::SynchronizeDisks,
        menu::GRANT_FOLDER => SessionAction::GrantDiskFolder,
        menu::RESET_NVRAM => SessionAction::ResetNvram,
        menu::SET_MAC => SessionAction::SetMac,
        _ => return None,
    })
}

/// A per-item command id's session action (discs and copy-on-write).
fn item_session(id: &str) -> Option<SessionAction> {
    use super::ids::item;
    let key = |prefix: &str| id.strip_prefix(prefix).and_then(|k| k.parse::<u8>().ok());
    if let Some(key) = key(item::INSERT_DISC) {
        return Some(SessionAction::InsertDisc { key });
    }
    if let Some(key) = key(item::EJECT_DISC) {
        return Some(SessionAction::EjectDisc { key });
    }
    if let Some(key) = key(item::MOUNT_DISC) {
        return Some(SessionAction::RemountDisc { key });
    }
    if let Some(key) = key(item::COMMIT_COW) {
        return Some(SessionAction::ApplyChanges { key });
    }
    key(item::DISCARD_COW).map(|key| SessionAction::RollBack { key })
}

/// Toolbar ids that are a menu item's command under another name.
fn alias(id: &str) -> Option<&'static str> {
    Some(match id {
        tb::NEW_MACHINE => menu::NEW_MACHINE,
        tb::DUPLICATE => menu::DUPLICATE,
        tb::DELETE => menu::DELETE,
        tb::CAPTURE => menu::CAPTURE,
        _ => return None,
    })
}

/// One entry of a shell-drawn menu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MenuEntry {
    /// The command id to run.
    pub id: String,
    pub label: String,
    pub tooltip: Option<String>,
    pub enabled: Enabled,
}

/// A titled group of entries (a submenu, or a section with a heading).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MenuGroup {
    pub title: String,
    pub icon: Icon,
    pub entries: Vec<MenuEntry>,
}

impl AppController {
    /// The toolbar's items for what the machine is doing, left to right;
    /// [`tb::GAP`] is the flexible space before the right-hand group. A
    /// stopped machine gets what you do to a machine (start it, change it,
    /// copy it); a running one what you do while it runs. ▶ only ever starts,
    /// ⏻ only ever asks IRIX to shut down.
    pub fn toolbar_layout(&self) -> Vec<&'static str> {
        let s = self.session();
        if s.is_running() || s.run_state() == super::session::RunState::Starting {
            // Capture (a click on the display), detaching and full screen
            // (the window's own button) are in the menus, not here.
            let mut v = vec![
                tb::SHOW_LIBRARY,
                tb::TITLE,
                tb::GAP,
                tb::SAVE_STATE,
                tb::SCREENSHOT,
                tb::DISC,
                tb::SERIAL_CONSOLE,
                // Each with its forced version in a menu (`toolbar_menu`).
                tb::SHUT_DOWN_GUEST,
                tb::RESTART,
            ];
            // A machine with no CD-ROM drive has no use for the disc button.
            if !crate::model::hardware::scsi_disks(self.config()).iter().any(|(_, d)| d.is_cdrom()) {
                v.retain(|id| *id != tb::DISC);
            }
            v
        } else {
            vec![
                tb::SHOW_LIBRARY,
                tb::TITLE,
                tb::GAP,
                tb::DUPLICATE,
                tb::DELETE,
                tb::BROWSE_FILES,
                tb::MACHINE_SETTINGS,
                tb::POWER_ON,
            ]
        }
    }

    /// The Disks menu: one group per disk and CD-ROM of the selected
    /// machine, each with what can be done to it now.
    pub fn disk_menu(&self) -> Vec<MenuGroup> {
        let mut keys: Vec<u8> = self.config().disks().iter().map(|d| d.0).collect();
        keys.sort();
        keys.into_iter().filter_map(|k| self.device_menu(k)).collect()
    }

    /// The menu a status-bar light opens: its disk's or CD-ROM's group of
    /// the Devices menu. `None` for the other lights (they run a command).
    pub fn light_menu(&self, id: &str) -> Option<MenuGroup> {
        use super::session::LightTarget as L;
        let bar = self.status_bar();
        match bar.lights.iter().find(|l| l.id == id)?.target {
            L::Disk { key } | L::CdRom { key } => self.device_menu(key),
            _ => None,
        }
    }

    /// One disk's or CD-ROM's group of the Devices menu (SCSI key `key`).
    fn device_menu(&self, key: u8) -> Option<MenuGroup> {
        use super::ids::item;
        let cfg = self.config();
        let entry = |id: String, label: String, tooltip: Option<String>| {
            let enabled = self.command_enabled(&id);
            MenuEntry { id, label, tooltip, enabled }
        };
        let overlays = self.overlays();
        {
            let d = cfg.disk(key)?;
            let name = crate::model::hardware::scsi_name(key);
            // A running CD-ROM shows the disc in it now, which may not be the
            // one the machine started with.
            let live = d.is_cdrom().then(|| self.cd_id(key)).flatten().and_then(|id| self.session().live_disc(id));
            let path = match live {
                Some(p) => p.unwrap_or_default(),
                None => d.path.as_str(),
            };
            let file = std::path::Path::new(path).file_name().map(|f| f.to_string_lossy().into_owned());
            if d.is_cdrom() {
                let title = match &file {
                    Some(f) => t!("scsi-slot-cdrom", id = name.as_str(), name = f.as_str()),
                    None => t!("scsi-slot-cdrom-empty", id = name.as_str()),
                };
                let insert = if file.is_some() { t!("scsi-change-disc") } else { t!("scsi-insert-disc") };
                let mut entries = vec![entry(item::id(item::INSERT_DISC, key), insert, None)];
                if file.is_some() {
                    entries.push(entry(item::id(item::EJECT_DISC, key), t!("scsi-eject"), None));
                    entries.push(entry(item::id(item::BROWSE_DISK, key), t!("ov-browse"), None));
                }
                Some(MenuGroup { title, icon: Icon::CdRom, entries })
            } else if d.kind() == iris::config::ScsiKind::Disk && !d.path.trim().is_empty() {
                let title = t!("scsi-slot-hdd", id = name.as_str(), name = file.unwrap_or_default());
                let mut entries = vec![entry(item::id(item::BROWSE_DISK, key), t!("ov-browse"), None)];
                if overlays.iter().any(|(k, _, _)| *k == key) {
                    entries.push(entry(item::id(item::COMMIT_COW, key), t!("menu-cow-commit"), Some(t!("menu-cow-commit-tip"))));
                    entries.push(entry(item::id(item::DISCARD_COW, key), t!("menu-cow-discard"), Some(t!("menu-cow-discard-tip"))));
                }
                if self.convertible(key) {
                    entries.push(entry(item::id(item::CONVERT_DISK, key), t!("convert-menu"), Some(t!("convert-menu-tip"))));
                }
                Some(MenuGroup { title, icon: Icon::Disk, entries })
            } else {
                None
            }
        }
    }

    /// The View menu's display-size item that is checked now.
    pub fn display_size_item(&self) -> &'static str {
        match self.prefs().display_mode {
            super::display::DisplayMode::Actual => menu::VIEW_ACTUAL,
            super::display::DisplayMode::Fit => menu::VIEW_FIT,
            super::display::DisplayMode::Stretch => menu::VIEW_STRETCH,
        }
    }

    /// A library row's context menu for `machine`, in sections: what you do
    /// with it, what you do to it, where it lives. Each entry selects the
    /// machine first (`ids::item::ON_MACHINE`).
    pub fn machine_menu(&self, machine: &str) -> Vec<Vec<MenuEntry>> {
        use super::ids::item;
        let entry = |cmd: &str, label: String| {
            let enabled = if cmd.starts_with(item::START_MACHINE) || self.selected() == Some(machine) {
                self.command_enabled(cmd)
            } else {
                // Not selected yet: what depends on the machine is checked
                // once it is (choosing the entry selects it).
                match self.can_change_machine() {
                    Enabled::Yes => self.command_enabled(cmd),
                    no => no,
                }
            };
            MenuEntry { id: item::on_machine(machine, cmd), label, tooltip: None, enabled }
        };
        vec![
            vec![
                entry(&item::id(item::START_MACHINE, machine), t!("ctx-start")),
                entry(tb::MACHINE_SETTINGS, t!("ctx-settings")),
                entry(tb::BROWSE_FILES, t!("ctx-browse")),
            ],
            vec![
                entry(menu::DUPLICATE, t!("mb-duplicate")),
                entry(menu::RENAME, t!("mb-rename")),
                entry(menu::DELETE, t!("mb-delete")),
            ],
            vec![entry(menu::SHOW_IN_FINDER, t!("mb-show-in-finder"))],
        ]
    }

    /// Whether the command `id` can run now, and if not why.
    pub fn command_enabled(&self, id: &str) -> Enabled {
        use super::ids::item;
        if let Some((_, cmd)) = id.strip_prefix(item::ON_MACHINE).and_then(|r| r.split_once(':')) {
            return self.command_enabled(cmd);
        }
        if let Some(c) = self.light_command(id) {
            return self.command_enabled(&c);
        }
        if let Some(a) = alias(id) {
            return self.command_enabled(a);
        }
        if id == tb::DISC {
            return self.command_enabled(tb::INSERT_DISC);
        }
        if let Some(name) = id.strip_prefix(item::START_MACHINE) {
            // The running machine is started already; another waits for it to stop.
            return match (self.session().is_running(), self.selected() == Some(name)) {
                (true, true) => Enabled::No(t!("tb-already-running")),
                (true, false) => Enabled::No(t!("tb-one-at-a-time")),
                (false, _) => Enabled::Yes,
            };
        }
        if id.starts_with(item::BROWSE_DISK) {
            return self.command_enabled(tb::BROWSE_FILES);
        }
        if id.starts_with(item::CONVERT_DISK) {
            return self.convert_enabled();
        }
        if id.starts_with(item::DECLINE_CONVERT) {
            return Enabled::Yes;
        }
        if id == super::convert::CANCEL_ID {
            return Enabled::Yes;
        }
        if let Some(a) = item_session(id) {
            return self.session_enabled(&a);
        }
        if let Some(name) = id.strip_prefix(item::RESTORE_STATE) {
            return self.session_enabled(&SessionAction::RestoreState(name.into()));
        }
        if let Some(a) = session_of(id) {
            return self.session_enabled(&a);
        }
        let running = self.session().is_running();
        let machine = || Enabled::when(self.selected().is_some(), || t!("toast-no-machine"));
        let need_running = || Enabled::when(running, || t!("session-needs-running"));
        match id {
            tb::INSERT_DISC => {
                let cd = crate::model::hardware::scsi_disks(self.config()).iter().any(|(_, d)| d.is_cdrom());
                Enabled::when(cd, || t!("tb-no-cdrom"))
            }
            // A conversion replaces the disk file under the browser.
            tb::BROWSE_FILES if self.conversion.is_some() => Enabled::No(t!("convert-busy")),
            // A running machine is read through its own disk hook, which every
            // machine has (`Workstation::read_disk`).
            tb::BROWSE_FILES => machine(),
            menu::CAPTURE if super::display::follows_pointer(self.prefs()) => Enabled::No(t!("prefs-pointer-no-capture")),
            tb::DETACH_DISPLAY | tb::FULL_SCREEN | menu::SEND_F11 | menu::SEND_CAD | menu::SEND_ESC | menu::CAPTURE => {
                need_running()
            }
            tb::SERIAL_CONSOLE | menu::CHECK_NETWORK => need_running(),
            // Full screen sizes the window itself.
            menu::VIEW_FIT_WINDOW if self.display().fullscreen() => Enabled::No(t!("mb-view-fit-window-full-screen")),
            menu::VIEW_FIT_WINDOW => need_running(),
            menu::GRANT_FOLDER => Enabled::when(cfg!(feature = "appstore"), || t!("menu-file-grant-folder-tip")),
            tb::MACHINE_SETTINGS | menu::SHOW_IN_FINDER | menu::OPEN_CONFIG => machine(),
            menu::DUPLICATE | menu::RENAME | menu::DELETE => match machine() {
                Enabled::Yes => self.can_change_machine(),
                no => no,
            },
            menu::NEW_MACHINE => self.can_change_machine(),
            menu::CAMERA_TEST => need_running(),
            _ => Enabled::Yes,
        }
    }

    /// The command a status-bar light's id stands for: a disk opens the
    /// browser, a CD-ROM asks for a disc, network opens the check, serial the
    /// console, capture toggles capture. `None` for other ids and for lights
    /// that only show something.
    pub fn light_command(&self, id: &str) -> Option<String> {
        use super::ids::item;
        use super::session::LightTarget as L;
        if !id.starts_with("status.") {
            return None;
        }
        let bar = self.status_bar();
        let light = bar.lights.iter().find(|l| l.id == id)?;
        Some(match light.target {
            L::Disk { key } => item::id(item::BROWSE_DISK, key),
            L::CdRom { key } => item::id(item::INSERT_DISC, key),
            L::Network => menu::CHECK_NETWORK.to_string(),
            L::Serial => tb::SERIAL_CONSOLE.to_string(),
            L::Capture => menu::CAPTURE.to_string(),
            L::Audio | L::Scale => return None,
        })
    }

    /// Run the command `id` (a toolbar item, a menu item or a key-map chord).
    pub fn command(&mut self, id: &str) -> Routed {
        if let Some((machine, cmd)) = id.strip_prefix(super::ids::item::ON_MACHINE).and_then(|r| r.split_once(':')) {
            let mut fx = Vec::new();
            if self.selected() != Some(machine) {
                fx.extend(self.apply(AppAction::Select(machine.to_string())));
                if self.selected() != Some(machine) {
                    return Routed { effects: fx, shell: None };
                }
            }
            let mut r = self.command(cmd);
            fx.append(&mut r.effects);
            r.effects = fx;
            return r;
        }
        if let Some(c) = self.light_command(id) {
            return self.command(&c);
        }
        if let Some(a) = alias(id) {
            return self.command(a);
        }
        let mut r = Routed::default();
        if let Enabled::No(why) = self.command_enabled(id) {
            r.effects.push(Effect::Toast(why));
            return r;
        }
        if let Some(a) = session_of(id) {
            r.effects = self.apply(AppAction::Session(a));
            return r;
        }
        use super::ids::item;
        let selected = self.selected().map(str::to_string);
        let display = |a: DisplayAction| AppAction::Display(a);
        let per_item = if let Some(n) = id.strip_prefix(item::RESTORE_STATE) {
            Some(SessionAction::RestoreState(n.into()))
        } else if let Some(n) = id.strip_prefix(item::RENAME_STATE) {
            Some(SessionAction::RenameState(n.into()))
        } else if let Some(n) = id.strip_prefix(item::DELETE_STATE) {
            Some(SessionAction::DeleteState(n.into()))
        } else {
            None
        };
        if let Some(a) = per_item {
            r.effects = self.apply(AppAction::Session(a));
            return r;
        }
        if let Some(key) = id.strip_prefix(item::BROWSE_DISK).and_then(|k| k.parse().ok()) {
            r.effects = self.apply(AppAction::OpenBrowser { key });
            return r;
        }
        if let Some(a) = super::convert::action_for(id) {
            r.effects = self.apply(a);
            return r;
        }
        if let Some(name) = id.strip_prefix(item::START_MACHINE) {
            // Select it (a no-op when it is already selected), then power on.
            let mut fx = Vec::new();
            if self.selected() != Some(name) {
                fx = self.apply(AppAction::Select(name.to_string()));
            }
            if self.selected() == Some(name) {
                fx.extend(self.apply(AppAction::Session(SessionAction::PowerOn)));
            }
            r.effects = fx;
            return r;
        }
        if id == tb::DISC {
            r.shell = Some(ShellCommand::DiscMenu);
            return r;
        }
        if let Some(a) = item_session(id) {
            r.effects = self.apply(AppAction::Session(a));
            return r;
        }
        let action = match id {
            tb::SHOW_LIBRARY => {
                r.shell = Some(ShellCommand::ToggleLibrary);
                None
            }
            tb::SAVED_STATES => {
                r.shell = Some(ShellCommand::SavedStates);
                None
            }
            tb::INSERT_DISC => {
                let key = crate::model::hardware::scsi_disks(self.config()).into_iter().find(|(_, d)| d.is_cdrom()).map(|(k, _)| k);
                key.map(|key| AppAction::Session(SessionAction::InsertDisc { key }))
            }
            tb::BROWSE_FILES => Some(AppAction::BrowseFiles),
            tb::DETACH_DISPLAY => Some(display(DisplayAction::ToggleDetached)),
            tb::FULL_SCREEN => Some(display(DisplayAction::ToggleFullScreen)),
            tb::SERIAL_CONSOLE => Some(AppAction::Serial(SerialAction::Open)),
            tb::MACHINE_SETTINGS => {
                r.shell = Some(ShellCommand::Settings);
                None
            }
            menu::CAPTURE => Some(display(DisplayAction::Capture)),
            menu::VIEW_ACTUAL => Some(display(DisplayAction::SetMode(super::display::DisplayMode::Actual))),
            menu::VIEW_FIT => Some(display(DisplayAction::SetMode(super::display::DisplayMode::Fit))),
            menu::VIEW_STRETCH => Some(display(DisplayAction::SetMode(super::display::DisplayMode::Stretch))),
            menu::VIEW_FIT_WINDOW => {
                r.shell = Some(ShellCommand::FitWindow);
                None
            }
            menu::SEND_F11 => Some(display(DisplayAction::Send(GuestChord::F11))),
            menu::SEND_CAD => Some(display(DisplayAction::Send(GuestChord::CtrlAltDel))),
            menu::SEND_ESC => Some(display(DisplayAction::Send(GuestChord::Escape))),
            menu::CHECK_NETWORK => Some(AppAction::Network(super::network::NetworkAction::OpenCheck)),
            menu::CAMERA_TEST => Some(AppAction::Camera(super::camera::CameraAction::Open)),
            menu::NEW_FOLDER => Some(AppAction::NewFolder),
            menu::DUPLICATE => selected.map(|machine| AppAction::Duplicate { machine }),
            menu::RENAME => selected.map(|machine| AppAction::Rename { machine }),
            menu::DELETE => selected.map(|machine| AppAction::Delete { machine }),
            menu::SHOW_IN_FINDER => selected.map(|machine| AppAction::ShowInFileManager { machine }),
            menu::OPEN_CONFIG => selected.map(|machine| AppAction::OpenConfigFile { machine }),
            menu::QUIT => Some(AppAction::RequestQuit),
            menu::GUIDE => {
                r.effects.push(Effect::OpenUrl("https://github.com/techomancer/iris/blob/main/iris-gui-README.md".into()));
                None
            }
            menu::REPORT_ISSUE => {
                r.effects.push(Effect::OpenUrl("https://github.com/techomancer/iris/issues".into()));
                None
            }
            menu::NEW_MACHINE => {
                r.shell = Some(ShellCommand::NewMachine);
                None
            }
            menu::ABOUT => {
                r.shell = Some(ShellCommand::About);
                None
            }
            menu::APP_SETTINGS => {
                r.shell = Some(ShellCommand::AppSettings);
                None
            }
            menu::BENCHMARK => {
                r.shell = Some(ShellCommand::Benchmark);
                None
            }
            menu::DIAGNOSTICS => {
                r.shell = Some(ShellCommand::Diagnostics);
                None
            }
            menu::CLOSE => {
                r.shell = Some(ShellCommand::CloseWindow);
                None
            }
            _ => None,
        };
        if let Some(a) = action {
            r.effects.extend(self.apply(a));
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_toolbar_id_has_a_tip_and_shortcuts_show() {
        let keys = KeyMap::for_platform(true);
        for id in [
            tb::SHOW_LIBRARY, tb::POWER_ON, tb::SHUT_DOWN_GUEST, tb::POWER_OFF, tb::RESTART, tb::SAVE_STATE,
            tb::SAVED_STATES, tb::SCREENSHOT, tb::INSERT_DISC, tb::BROWSE_FILES, tb::DETACH_DISPLAY,
            tb::FULL_SCREEN, tb::SERIAL_CONSOLE, tb::MACHINE_SETTINGS,
        ] {
            let t = tip(id, &keys).unwrap_or_else(|| panic!("{id}"));
            assert!(!t.tooltip.is_empty());
        }
        assert_eq!(tip(tb::SCREENSHOT, &keys).unwrap().shortcut, "Shift+Cmd+S");
    }

    #[test]
    fn every_key_command_maps_to_an_id() {
        for c in Command::ALL {
            assert!(!id_of(*c).is_empty());
        }
    }
}
