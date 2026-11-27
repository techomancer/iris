//! The application: the machine library, the selected machine and its
//! in-memory config, autosave, app preferences, and quitting.
//!
//! Every shell owns one `AppController`. It owns the [`SessionController`] that
//! runs the selected machine, so selecting, renaming or deleting a machine can
//! wait for the worker to be idle.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use iris::config::MachineConfig;

use super::profile::{traits, Family, MachineIcon};
use super::session::SessionController;

use super::{Answer, ChoiceSpec, ConfirmSpec, Controller, Effect, Enabled, TextSpec};
use crate::model::job::Notify;
use crate::model::machines;
use crate::model::settings::{GuiSettings, LibraryFolder};

/// How long the config must sit unedited before it is written.
pub const AUTOSAVE_AFTER: Duration = Duration::from_millis(600);

pub const RENAME_MACHINE: &str = "app.rename-machine";
pub const DELETE_MACHINE: &str = "app.delete-machine";
pub const NEW_FOLDER: &str = "app.new-folder";
pub const RENAME_FOLDER: &str = "app.rename-folder";
/// Quit while a machine runs: Cancel or power off and quit.
pub const QUIT_RUNNING: &str = "app.quit-running";
/// Which disk Browse Files opens, when the machine has several.
pub const BROWSE_WHICH: &str = "app.browse-which";

/// What the user asked the application to do.
#[derive(Debug, Clone)]
pub enum AppAction {
    /// Select a machine in the library; it becomes the working directory.
    Select(String),
    /// Create and select a machine (the new-machine dialog's result).
    CreateMachine { name: String, config: Box<MachineConfig> },
    /// Ask for a new name for `machine`.
    Rename { machine: String },
    /// Copy `machine`; small files are copied, disk images shared.
    Duplicate { machine: String },
    /// Ask, then remove `machine` from the library (its disks stay).
    Delete { machine: String },
    /// Ask for a folder name and add an empty library folder.
    NewFolder,
    RenameFolder { folder: String },
    /// Remove a folder; its machines go back to their family groups.
    DeleteFolder { folder: String },
    /// Put `machine` in `folder`, or back under its family with `None`.
    MoveToFolder { machine: String, folder: Option<String> },
    ToggleFolder { folder: String },
    /// Show the machine's folder in the host's file manager.
    ShowInFileManager { machine: String },
    /// Open the machine's TOML in the user's editor.
    OpenConfigFile { machine: String },
    /// Choose the interface language (empty follows the system).
    SetLanguage(String),
    /// The application-wide UI zoom.
    SetUiScale(f32),
    /// The application-wide guest display scale (D22).
    SetDisplayScale(f32),
    /// The selected machine's in-memory config was edited.
    ConfigEdited,
    /// Write pending edits now (before an action that reads the file).
    Flush,
    /// The user asked to quit (window close, menu, Cmd+Q).
    RequestQuit,
    /// The host-camera test window.
    Camera(super::camera::CameraAction),
    /// Networking help and packet-capture permission.
    Network(super::network::NetworkAction),
    /// The serial console window.
    Serial(super::serial::SerialAction),
    /// A settings edit on the selected machine.
    Settings(super::settings::SettingsAction),
    /// A display action: capture, keys and pointer, full screen, scaling.
    Display(super::display::DisplayAction),
    /// A session action: power, saved states, discs, disks.
    Session(super::session::SessionAction),
    /// Open the disk browser on `[scsi.KEY]` of the selected machine.
    OpenBrowser { key: u8 },
    /// Open the disk browser, asking which disk when there are several.
    BrowseFiles,
    /// An action in the open disk browser.
    Browser(super::disk_browser::BrowserAction),
    /// Close the disk browser (cancels its job).
    CloseBrowser,
    /// Ask which format `[scsi.KEY]`, a compressed CHD, converts to.
    ConvertDisk { key: u8 },
    /// Stop the running disk conversion; the disk stays as it was.
    CancelConversion,
    /// Keep `[scsi.KEY]` as it is: no more suggestions to convert it.
    DeclineConvert { key: u8 },
    Answer(Answer),
}

/// A row in the library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryRow {
    pub name: String,
    /// The machine and board, as "Indy (IP24)" (as the title bar shows them).
    pub subtitle: String,
    pub icon: MachineIcon,
    pub selected: bool,
    /// Shown only for a running machine or one with a problem.
    pub dot: Option<RowDot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowDot {
    Running,
    /// Something will stop it from starting; the text says what.
    Problem(String),
}

/// Which group a library section is.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GroupKey {
    Folder(String),
    Family(Family),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryGroup {
    pub key: GroupKey,
    pub title: String,
    pub collapsed: bool,
    pub rows: Vec<LibraryRow>,
}

/// The library, the selected machine and the app preferences.
pub struct AppController {
    prefs: GuiSettings,
    cfg: MachineConfig,
    dirty_since: Option<Instant>,
    session: SessionController,
    display: super::display::DisplayController,
    settings: super::settings::SettingsController,
    serial: super::serial::SerialConsoleController,
    pub(crate) network: super::network::NetworkController,
    camera: super::camera::CameraTestController,
    /// The disk browser, while its window is open.
    browser: Option<super::disk_browser::DiskBrowserController>,
    /// A disk being converted, and a format question waiting for its answer.
    pub(crate) conversion: Option<super::convert::Conversion>,
    pub(crate) pending_convert: Option<super::convert::Asked>,
    /// CHDs just attached, to ask about once their headers are read.
    pub(crate) attach_checks: Vec<super::convert::AttachCheck>,
    /// Reset: power the machine on again once its stop has finished.
    pub(crate) restart_after_stop: bool,
    /// Disk images' readability, checked off the UI thread.
    pub(crate) probe: crate::model::disk_probe::DiskProbe,
    notify: Arc<dyn Notify>,
    /// Problems found the last time the library was checked, by machine.
    problems: std::collections::BTreeMap<String, String>,
    /// The machine a rename or delete prompt is about.
    pending_machine: Option<String>,
    pending_folder: Option<String>,
    /// Quit once the disk synchronisation the quit started has finished.
    quit_after_sync: bool,
    /// The user chose to quit although the machine was running.
    quit_anyway: bool,
    /// Effects produced while constructing, handed out by the first tick.
    startup: Vec<Effect>,
    /// Session events for the shell (display, serial console) since it last asked.
    session_events: Vec<super::session::SessionEvent>,
}

impl AppController {
    /// Start-up for a shell: claim the single-instance lock (ending a stale
    /// copy that holds the ports), load the preferences, pick the language,
    /// set the environment the embedded core reads, and select the machine.
    /// Call once, before any machine can start.
    /// Another IRIS front end already running, by PID. The app is single
    /// instance: a shell that finds one brings it forward and exits instead
    /// of launching (launching would take over its ports and its library).
    pub fn other_instance() -> Option<i32> {
        crate::model::single_instance::running_other()
    }

    pub fn launch(notify: Arc<dyn Notify>) -> Self {
        // The core must not exit the process when a guest powers off.
        std::env::set_var("IRIS_NO_EXIT_ON_POWEROFF", "1");
        // The App Sandbox only allows MAP_JIT pages; the JITs would be killed.
        #[cfg(feature = "appstore")]
        std::env::set_var("IRIS_NO_JIT", "1");
        crate::model::single_instance::acquire();
        let prefs = GuiSettings::load();
        crate::model::i18n::set_language(&prefs.language);
        crate::model::macos_sandbox::restore(&prefs.bookmarks);
        Self::new(prefs, notify)
    }

    /// The last step of a shell's exit: release the single-instance lock.
    pub fn release_instance() {
        crate::model::single_instance::release();
    }

    /// Take over `prefs` (already loaded) and select its active machine.
    pub fn new(mut prefs: GuiSettings, notify: Arc<dyn Notify>) -> Self {
        let mut cfg = MachineConfig::default();
        let mut startup: Vec<Effect> = Vec::new();
        if let Some(name) = prefs.active_machine.clone() {
            match prefs.machines.get(&name) {
                Some(stored) => cfg = stored.clone(),
                None => prefs.active_machine = None,
            }
        }
        if prefs.active_machine.is_none() {
            prefs.active_machine = prefs.machines.keys().next().cloned();
            if let Some(name) = &prefs.active_machine {
                cfg = prefs.machines[name].clone();
            }
        }
        let dir = match prefs.active_machine.as_deref() {
            Some(name) => prefs.machine_dir(name),
            None => prefs.root().map(std::path::Path::to_path_buf),
        };
        let mut errors = prefs.errors.clone();
        if let Err(e) = dir.and_then(|dir| machines::enter(&dir)) {
            errors.push(e);
        }
        if !errors.is_empty() {
            startup.push(Effect::Toast(errors.join("\n")));
        }
        let mut app = Self {
            prefs,
            cfg,
            dirty_since: None,
            session: SessionController::spawn(notify.clone()),
            display: Default::default(),
            settings: Default::default(),
            serial: Default::default(),
            network: super::network::NetworkController::new(),
            camera: Default::default(),
            browser: None,
            conversion: None,
            pending_convert: None,
            attach_checks: Vec::new(),
            restart_after_stop: false,
            probe: crate::model::disk_probe::DiskProbe::new(notify.clone()),
            notify,
            problems: Default::default(),
            pending_machine: None,
            pending_folder: None,
            quit_after_sync: false,
            quit_anyway: false,
            startup,
            session_events: Vec::new(),
        };
        app.check_library();
        app
    }

    // ---- read accessors -----------------------------------------------------

    pub fn prefs(&self) -> &GuiSettings {
        &self.prefs
    }

    /// The selected machine's name.
    pub fn selected(&self) -> Option<&str> {
        self.prefs.active_machine.as_deref()
    }

    /// The selected machine's in-memory config (what the settings show).
    pub fn config(&self) -> &MachineConfig {
        &self.cfg
    }

    /// The config for the settings editor to change in place; follow every
    /// change with [`AppAction::ConfigEdited`]. Goes away when the settings
    /// form is driven by the settings controller.
    #[doc(hidden)]
    pub fn config_mut(&mut self) -> &mut MachineConfig {
        &mut self.cfg
    }

    #[doc(hidden)]
    pub fn prefs_mut(&mut self) -> &mut GuiSettings {
        &mut self.prefs
    }

    /// Edits not yet written to the machine's TOML.
    pub fn is_dirty(&self) -> bool {
        self.dirty_since.is_some()
    }

    /// No machine exists yet: the shell opens the new-machine dialog.
    pub fn is_empty(&self) -> bool {
        self.prefs.machines.is_empty()
    }

    pub fn session(&self) -> &SessionController {
        &self.session
    }

    pub fn session_mut(&mut self) -> &mut SessionController {
        &mut self.session
    }

    pub fn camera(&self) -> &super::camera::CameraTestController {
        &self.camera
    }

    pub fn serial(&self) -> &super::serial::SerialConsoleController {
        &self.serial
    }

    /// The selected machine's settings, as sections of fields.
    pub fn settings_sections(&self) -> Vec<super::settings::Section> {
        let dir = self.machine_dir();
        let readable = |p: &str| self.disk_readable(dir.as_deref(), p);
        self.settings.sections_with(&self.cfg, self.session.is_running(), &readable)
    }

    /// Whether a disk image path (relative to the machine folder) was readable
    /// when last checked; never blocks (see `model::disk_probe`).
    pub(crate) fn disk_readable(&self, dir: Option<&std::path::Path>, path: &str) -> bool {
        match dir {
            Some(d) if !path.is_empty() && !iris::http_disk::is_url(path) => {
                self.probe.readable(&machines::resolve(d, path).to_string_lossy())
            }
            _ => self.probe.readable(path),
        }
    }

    pub fn display(&self) -> &super::display::DisplayController {
        &self.display
    }

    /// A message over the display: why there is no picture.
    pub fn display_overlay(&self) -> Option<String> {
        use super::session::RunState;
        match self.session.run_state() {
            RunState::PoweredOff => Some(t!("fb-powered-off")),
            RunState::Starting => Some(t!("status-starting")),
            RunState::Stopped | RunState::Running | RunState::Halted => None,
        }
    }

    pub fn notify(&self) -> Arc<dyn Notify> {
        self.notify.clone()
    }

    /// The selected machine's folder.
    pub fn machine_dir(&self) -> Option<PathBuf> {
        self.selected().and_then(|n| self.prefs.machine_dir(n).ok())
    }

    /// The disk browser, while it is open. A shell shows its window exactly
    /// while this is `Some`.
    pub fn browser(&self) -> Option<&super::disk_browser::DiskBrowserController> {
        self.browser.as_ref()
    }

    /// The selected machine's disks and CD-ROMs the browser can open, by key.
    pub fn browse_targets(&self) -> Vec<(u8, super::disk_browser::BrowseTarget)> {
        let (Some(name), Some(_)) = (self.selected(), self.machine_dir()) else { return Vec::new() };
        let mut keys: Vec<u8> = self.cfg.disks().iter().map(|d| d.0).collect();
        keys.sort();
        keys.into_iter()
            .filter_map(|key| {
                let d = &self.cfg.disk(key).unwrap();
                if d.path.trim().is_empty() || !(d.kind() == iris::config::ScsiKind::Disk || d.is_cdrom()) {
                    return None;
                }
                let id = crate::model::hardware::scsi_name(key);
                let label = if d.is_cdrom() { t!("browser-label-cdrom", id = id) } else { t!("browser-label-disk", id = id) };
                // As the TOML spells it, against the machine's folder (the
                // working directory, which stays put while the browser is
                // open): a CHD's `.diff.chd` is named from this spelling when
                // `IRIS_CHD_DIFF_DIR` is set, so the browser and the guest
                // then write the same diff.
                let path = d.path.clone();
                Some((key, super::disk_browser::BrowseTarget { machine: name.to_string(), label, path, cdrom: d.is_cdrom(), overlay: d.overlay }))
            })
            .collect()
    }

    fn open_browser(&mut self, key: u8, fx: &mut Vec<Effect>) {
        if let Enabled::No(why) = self.command_enabled(super::ids::toolbar::BROWSE_FILES) {
            fx.push(Effect::Toast(why));
            return;
        }
        let Some((_, target)) = self.browse_targets().into_iter().find(|(k, _)| *k == key) else {
            fx.push(Effect::Toast(t!("browser-no-disks")));
            return;
        };
        if !self.close_browser(fx) {
            return;
        }
        // Edits must reach the file the machine boots from.
        self.flush(fx);
        // A running machine's disk is read through the machine, read-only.
        let live = if self.session.is_running() {
            if self.cfg.disk(key).is_none() {
                return;
            }
            let (bus, id) = crate::model::hardware::placement(key);
            match self.session.live_disk(bus as usize, id) {
                Some(l) => Some(l),
                None => {
                    fx.push(Effect::Toast(t!("browser-unavailable-running")));
                    return;
                }
            }
        } else {
            None
        };
        self.browser = Some(super::disk_browser::DiskBrowserController::open(target, live, self.notify.clone()));
    }

    /// Close the browser; `false` (with a toast) while it is writing.
    pub(crate) fn close_browser(&mut self, fx: &mut Vec<Effect>) -> bool {
        if let Some(b) = &mut self.browser {
            // Any job, not only a write: opening or checking the disk holds
            // the image too.
            if b.writing() || !b.close() {
                fx.push(Effect::Toast(t!("browser-busy")));
                return false;
            }
        }
        self.browser = None;
        true
    }

    /// Whether the library can change now (select, rename, delete, create):
    /// settings writable, the worker idle.
    pub fn can_change_machine(&self) -> Enabled {
        if let Err(e) = self.prefs.ensure_writable() {
            return Enabled::No(e);
        }
        if self.conversion.is_some() {
            return Enabled::No(t!("convert-busy"));
        }
        Enabled::when(self.session.is_idle(), || t!("toast-switch-machine-busy"))
    }

    /// The library: user folders first, then the families that have machines.
    pub fn library(&self) -> Vec<LibraryGroup> {
        let running = self.session.is_running();
        let row = |name: &str| -> Option<LibraryRow> {
            let cfg = self.prefs.machines.get(name)?;
            let t = traits(cfg.machine.profile);
            let selected = self.selected() == Some(name);
            let dot = if selected && running {
                Some(RowDot::Running)
            } else {
                self.problems.get(name).map(|p| RowDot::Problem(p.clone()))
            };
            Some(LibraryRow {
                name: name.to_string(),
                subtitle: t!("library-row-subtitle", model = t.model, board = t.board),
                icon: t.icon,
                selected,
                dot,
            })
        };
        let mut groups = Vec::new();
        let mut placed = std::collections::BTreeSet::new();
        for f in &self.prefs.library_folders {
            let rows: Vec<_> = f.machines.iter().filter_map(|m| row(m)).collect();
            placed.extend(rows.iter().map(|r| r.name.clone()));
            groups.push(LibraryGroup {
                key: GroupKey::Folder(f.name.clone()),
                title: f.name.clone(),
                collapsed: f.collapsed,
                rows,
            });
        }
        for &family in Family::ALL {
            let rows: Vec<_> = self
                .prefs
                .machines
                .iter()
                .filter(|(n, c)| !placed.contains(*n) && traits(c.machine.profile).family == family)
                .filter_map(|(n, _)| row(n))
                .collect();
            if !rows.is_empty() {
                groups.push(LibraryGroup { key: GroupKey::Family(family), title: family.title(), collapsed: false, rows });
            }
        }
        groups
    }

    /// The folder `machine` is in, if any.
    pub fn folder_of(&self, machine: &str) -> Option<&str> {
        self.prefs
            .library_folders
            .iter()
            .find(|f| f.machines.iter().any(|m| m == machine))
            .map(|f| f.name.as_str())
    }

    // ---- internals ------------------------------------------------------------

    fn mark_dirty(&mut self) -> Vec<Effect> {
        let mut fx = Vec::new();
        if let Some(dir) = self.machine_dir() {
            if let Err(e) = machines::make_relative(&mut self.cfg, &dir) {
                fx.push(Effect::Toast(e));
            }
        }
        self.dirty_since = Some(Instant::now());
        fx
    }

    /// Write the selected machine's config. On failure the edit is kept.
    fn flush(&mut self, fx: &mut Vec<Effect>) -> bool {
        let Some(name) = self.prefs.active_machine.clone() else { return false };
        if let Err(e) = self.prefs.save_machine(&name, &mut self.cfg) {
            fx.push(Effect::Toast(t!("toast-autosave-failed", error = e)));
            return false;
        }
        self.dirty_since = None;
        true
    }

    fn guard(&self, fx: &mut Vec<Effect>) -> bool {
        match self.can_change_machine() {
            Enabled::Yes => true,
            Enabled::No(why) => {
                fx.push(Effect::Toast(why));
                false
            }
        }
    }

    pub(crate) fn save_prefs(&mut self, fx: &mut Vec<Effect>) {
        if let Err(e) = self.prefs.save() {
            fx.push(Effect::Toast(t!("toast-settings-save-failed", error = e)));
        }
    }

    /// Note which machines will not start as they are (a missing disk image),
    /// for the library's problem dots.
    fn check_library(&mut self) {
        self.problems.clear();
        let Ok(root) = self.prefs.root().map(|r| r.to_path_buf()) else { return };
        for (name, cfg) in &self.prefs.machines {
            let Ok(dir) = machines::directory(&root, name) else { continue };
            let readable = |p: &str| {
                iris::http_disk::is_url(p) || machines::resolve(&dir, p).is_file()
            };
            if let Some(m) = crate::model::summary::missing_disks(cfg, readable).first() {
                self.problems.insert(name.clone(), t!("library-problem-missing-disk", disk = m.name.as_str()));
            } else if let Err(e) = cfg.validate() {
                self.problems.insert(name.clone(), e.to_string());
            }
        }
    }

    fn select(&mut self, name: &str, fx: &mut Vec<Effect>) {
        if !self.guard(fx) {
            return;
        }
        if self.selected() != Some(name) && !self.close_browser(fx) {
            return;
        }
        if self.is_dirty() && !self.flush(fx) {
            return;
        }
        let result = (|| -> Result<MachineConfig, String> {
            let mut cfg = machines::read(&self.prefs.machine_path(name)?)?;
            let dir = self.prefs.machine_dir(name)?;
            machines::make_relative(&mut cfg, &dir)?;
            machines::enter(&dir)?;
            Ok(cfg)
        })();
        match result {
            Ok(cfg) => {
                self.cfg = cfg;
                self.prefs.machines.insert(name.into(), self.cfg.clone());
                self.prefs.active_machine = Some(name.into());
                match self.prefs.save() {
                    Err(e) => fx.push(Effect::Toast(t!("toast-settings-save-failed", error = e))),
                    Ok(()) => fx.push(Effect::Toast(t!("toast-machine-loaded", name = name))),
                }
            }
            Err(e) => fx.push(Effect::Toast(t!("toast-machine-load-failed", error = e))),
        }
    }

    fn create(&mut self, name: &str, mut cfg: MachineConfig, fx: &mut Vec<Effect>) {
        if !self.guard(fx) {
            return;
        }
        if let Err(e) = machines::validate_name(name) {
            fx.push(Effect::Toast(e));
            return;
        }
        if self.is_dirty() && !self.flush(fx) {
            return;
        }
        let name = self.prefs.unique_name(name);
        match self.prefs.save_machine(&name, &mut cfg) {
            Ok(()) => {
                self.select(&name, fx);
                fx.push(Effect::Toast(t!("toast-machine-created", name = &name)));
            }
            Err(e) => fx.push(Effect::Toast(t!("toast-machine-create-failed", error = e))),
        }
        self.check_library();
    }

    fn rename(&mut self, old: &str, new: &str, fx: &mut Vec<Effect>) {
        if !self.guard(fx) {
            return;
        }
        let new = new.trim();
        if new == old {
            return;
        }
        if let Err(e) = machines::validate_name(new) {
            fx.push(Effect::Toast(e));
            return;
        }
        let active = self.selected() == Some(old);
        if active && self.is_dirty() && !self.flush(fx) {
            return;
        }
        let name = self.prefs.unique_name(new);
        // The folder may be the working directory, and Windows won't rename that.
        let result = self.prefs.root().map(|r| r.to_path_buf()).and_then(|root| {
            std::env::set_current_dir(&root).map_err(|e| e.to_string())?;
            machines::rename(&root, old, &name)
        });
        if let Err(e) = result {
            fx.push(Effect::Toast(t!("toast-rename-failed", error = e)));
            if let Err(e) = self.machine_dir().ok_or_default().and_then(|d| machines::enter(&d)) {
                fx.push(Effect::Toast(e));
            }
            return;
        }
        if let Some(cfg) = self.prefs.machines.remove(old) {
            self.prefs.machines.insert(name.clone(), if active { self.cfg.clone() } else { cfg });
        }
        for f in &mut self.prefs.library_folders {
            for m in &mut f.machines {
                if m == old {
                    *m = name.clone();
                }
            }
        }
        if active {
            self.prefs.active_machine = Some(name.clone());
        }
        if let Err(e) = self.machine_dir().ok_or_default().and_then(|d| machines::enter(&d)) {
            fx.push(Effect::Toast(e));
            return;
        }
        match self.prefs.save() {
            Err(e) => fx.push(Effect::Toast(e)),
            Ok(()) => fx.push(Effect::Toast(t!("toast-renamed", name = &name))),
        }
        self.check_library();
    }

    fn delete(&mut self, name: &str, fx: &mut Vec<Effect>) {
        if !self.guard(fx) {
            return;
        }
        let result = self.prefs.machine_path(name).and_then(|p| std::fs::remove_file(p).map_err(|e| e.to_string()));
        if let Err(e) = result {
            fx.push(Effect::Toast(t!("toast-delete-failed", error = e)));
            return;
        }
        self.prefs.machines.remove(name);
        for f in &mut self.prefs.library_folders {
            f.machines.retain(|m| m != name);
        }
        if self.selected() == Some(name) {
            self.dirty_since = None;
            self.prefs.active_machine = None;
            if let Some(next) = self.prefs.machines.keys().next().cloned() {
                self.select(&next, fx);
            } else {
                self.cfg = MachineConfig::default();
                if let Err(e) = self.prefs.root().and_then(machines::enter) {
                    fx.push(Effect::Toast(e));
                }
                self.save_prefs(fx);
            }
        } else {
            self.save_prefs(fx);
        }
        fx.push(Effect::Toast(t!("toast-deleted", name = name)));
        self.check_library();
    }

    fn duplicate(&mut self, name: &str, fx: &mut Vec<Effect>) {
        if !self.guard(fx) {
            return;
        }
        if self.selected() == Some(name) && self.is_dirty() && !self.flush(fx) {
            return;
        }
        let copy = self.prefs.unique_name(name);
        let result = self.prefs.root().map(|r| r.to_path_buf()).and_then(|root| machines::duplicate(&root, name, &copy));
        match result {
            Ok((cfg, shared)) => {
                self.prefs.machines.insert(copy.clone(), cfg);
                if let Some(folder) = self.folder_of(name).map(str::to_string) {
                    if let Some(f) = self.prefs.library_folders.iter_mut().find(|f| f.name == folder) {
                        f.machines.push(copy.clone());
                    }
                }
                self.save_prefs(fx);
                fx.push(Effect::Toast(if shared.is_empty() {
                    t!("toast-duplicated", name = &copy)
                } else {
                    t!("toast-duplicated-shared", name = &copy, count = shared.len())
                }));
            }
            Err(e) => fx.push(Effect::Toast(t!("toast-duplicate-failed", error = e))),
        }
        self.check_library();
    }

    fn folder_names(&self) -> Vec<String> {
        self.prefs.library_folders.iter().map(|f| f.name.clone()).collect()
    }

    fn ask_folder_name(&self, purpose: &'static str, title: String, initial: String) -> Effect {
        Effect::AskText(TextSpec {
            purpose,
            title,
            prompt: t!("library-folder-name"),
            initial: initial.clone(),
            ok_label: t!("common-ok"),
            taken: self.folder_names().into_iter().filter(|n| *n != initial).collect(),
            error: None,
            validator: folder_name_rule,
        })
    }

    fn answer(&mut self, answer: Answer, fx: &mut Vec<Effect>) {
        match answer {
            Answer::Text { purpose: RENAME_MACHINE, text } => {
                if let (Some(old), Some(new)) = (self.pending_machine.take(), text) {
                    self.rename(&old, &new, fx);
                }
            }
            Answer::Confirmed { purpose: DELETE_MACHINE, yes } => {
                if let (Some(name), true) = (self.pending_machine.take(), yes) {
                    self.delete(&name, fx);
                }
            }
            Answer::Text { purpose: NEW_FOLDER, text: Some(name) } => {
                let name = name.trim().to_string();
                if folder_name_rule_with(&self.folder_names(), &name).is_ok() {
                    self.prefs.library_folders.push(LibraryFolder { name, machines: Vec::new(), collapsed: false });
                    self.save_prefs(fx);
                }
            }
            Answer::Text { purpose: RENAME_FOLDER, text: Some(name) } => {
                let name = name.trim().to_string();
                let Some(old) = self.pending_folder.take() else { return };
                let others: Vec<_> = self.folder_names().into_iter().filter(|n| *n != old).collect();
                if folder_name_rule_with(&others, &name).is_ok() {
                    if let Some(f) = self.prefs.library_folders.iter_mut().find(|f| f.name == old) {
                        f.name = name;
                    }
                    self.save_prefs(fx);
                }
            }
            Answer::Text { purpose: RENAME_FOLDER, text: None } => self.pending_folder = None,
            Answer::Chose { purpose: QUIT_RUNNING, index: 1 } => {
                self.quit_anyway = true;
                self.request_quit(fx);
            }
            // Cancelled prompts and answers to someone else's question.
            Answer::Text { .. } | Answer::Confirmed { .. } | Answer::Picked { .. } | Answer::Chose { .. } => {}
        }
    }

    fn request_quit(&mut self, fx: &mut Vec<Effect>) {
        // A guest that is still running would lose what it has not written:
        // ask first. A halted CPU (IRIX shut down, or the PROM idle) is safe.
        if self.session.is_running() && !self.session.status().cpu_halted && !self.quit_anyway {
            let machine = self.selected().unwrap_or_default().to_string();
            fx.push(Effect::Choose(ChoiceSpec {
                purpose: QUIT_RUNNING,
                title: t!("app-quit-running-title", machine = machine.as_str()),
                message: t!("app-quit-running-body"),
                choices: vec![t!("common-cancel"), t!("app-quit-running-anyway")],
                dismiss: 0,
                default: Some(0),
                destructive: vec![1],
            }));
            return;
        }
        if self.session.syncing.is_some() {
            // Still folding disks: quit when that ends.
            self.quit_after_sync = true;
            return;
        }
        if self.session.emu.has_pending_chd_sync() && !self.quit_after_sync {
            // Fold CHD diffs back into their bases first; quit on SyncDone.
            self.quit_after_sync = true;
            self.session.syncing = Some(super::session::SyncProgress { disk: 0, total: 0, fraction: 0.0 });
            self.session.emu.send(crate::model::handle::Cmd::SyncDisks);
            return;
        }
        if self.is_dirty() {
            self.flush(fx);
        } else {
            self.save_prefs(fx);
        }
        fx.push(Effect::Quit);
    }

    /// Whether the app is waiting for a disk sync before it quits.
    pub fn quitting(&self) -> bool {
        self.quit_after_sync
    }

    /// The last step of quitting, once the shell's window is going away: stop
    /// the machine and join the worker. Bounded by the worker's stop timeout.
    pub fn shutdown(&mut self) {
        if self.is_dirty() {
            let mut fx = Vec::new();
            self.flush(&mut fx);
        } else {
            let _ = self.prefs.save();
        }
        self.session.emu.shutdown();
    }
}

trait OkOrDefault<T> {
    fn ok_or_default(self) -> Result<T, String>;
}

impl<T> OkOrDefault<T> for Option<T> {
    fn ok_or_default(self) -> Result<T, String> {
        self.ok_or_else(|| t!("toast-no-machine"))
    }
}

fn folder_name_rule_with(taken: &[String], name: &str) -> Result<(), String> {
    let name = name.trim();
    if name.is_empty() {
        return Err(String::new());
    }
    if taken.iter().any(|t| t.eq_ignore_ascii_case(name)) {
        return Err(t!("library-folder-taken"));
    }
    Ok(())
}

fn folder_name_rule(spec: &TextSpec, text: &str) -> Result<(), String> {
    folder_name_rule_with(&spec.taken, text)
}

fn machine_name_rule(spec: &TextSpec, text: &str) -> Result<(), String> {
    let text = text.trim();
    if text.is_empty() || text == spec.initial {
        return Err(String::new());
    }
    machines::validate_name(text)
}

impl Controller for AppController {
    type Action = AppAction;

    fn apply(&mut self, action: AppAction) -> Vec<Effect> {
        let mut fx = Vec::new();
        match action {
            AppAction::Select(name) => self.select(&name, &mut fx),
            AppAction::CreateMachine { name, config } => {
                // The new machine is selected under the name it got, which
                // may carry a suffix ("indy-2").
                let before = self.selected().map(str::to_string);
                self.create(&name, *config, &mut fx);
                if self.selected().is_some() && self.selected().map(str::to_string) != before {
                    self.watch_attached(&Default::default());
                }
            }
            AppAction::Rename { machine } => {
                if self.guard(&mut fx) {
                    fx.push(Effect::AskText(TextSpec {
                        purpose: RENAME_MACHINE,
                        title: t!("modal-rename-title"),
                        prompt: t!("modal-rename-label"),
                        initial: machine.clone(),
                        ok_label: t!("modal-rename-confirm"),
                        taken: Vec::new(),
                        error: None,
                        validator: machine_name_rule,
                    }));
                    self.pending_machine = Some(machine);
                }
            }
            AppAction::Duplicate { machine } => self.duplicate(&machine, &mut fx),
            AppAction::Delete { machine } => {
                if self.guard(&mut fx) {
                    fx.push(Effect::Confirm(ConfirmSpec {
                        purpose: DELETE_MACHINE,
                        title: t!("library-delete-title", name = &machine),
                        message: t!("library-delete-message"),
                        confirm_label: t!("library-delete-ok"),
                        destructive: true,
                    }));
                    self.pending_machine = Some(machine);
                }
            }
            AppAction::NewFolder => fx.push(self.ask_folder_name(NEW_FOLDER, t!("library-new-folder"), String::new())),
            AppAction::RenameFolder { folder } => {
                fx.push(self.ask_folder_name(RENAME_FOLDER, t!("library-rename-folder"), folder.clone()));
                self.pending_folder = Some(folder);
            }
            AppAction::DeleteFolder { folder } => {
                self.prefs.library_folders.retain(|f| f.name != folder);
                self.save_prefs(&mut fx);
            }
            AppAction::MoveToFolder { machine, folder } => {
                for f in &mut self.prefs.library_folders {
                    f.machines.retain(|m| *m != machine);
                }
                if let Some(folder) = folder {
                    if let Some(f) = self.prefs.library_folders.iter_mut().find(|f| f.name == folder) {
                        f.machines.push(machine);
                    }
                }
                self.save_prefs(&mut fx);
            }
            AppAction::ToggleFolder { folder } => {
                if let Some(f) = self.prefs.library_folders.iter_mut().find(|f| f.name == folder) {
                    f.collapsed = !f.collapsed;
                }
                self.save_prefs(&mut fx);
            }
            AppAction::ShowInFileManager { machine } => match self.prefs.machine_dir(&machine) {
                Ok(dir) => fx.push(Effect::RevealPath(dir)),
                Err(e) => fx.push(Effect::Toast(e)),
            },
            AppAction::OpenConfigFile { machine } => {
                if self.selected() == Some(machine.as_str()) && self.is_dirty() {
                    self.flush(&mut fx);
                }
                match self.prefs.machine_path(&machine) {
                    Ok(path) => fx.push(Effect::OpenPath(path)),
                    Err(e) => fx.push(Effect::Toast(e)),
                }
            }
            AppAction::SetLanguage(lang) => {
                crate::model::i18n::set_language(&lang);
                self.prefs.language = lang;
                self.save_prefs(&mut fx);
            }
            AppAction::SetUiScale(s) => {
                self.prefs.ui_scale = s.clamp(crate::model::settings::UI_SCALE_MIN, crate::model::settings::UI_SCALE_MAX);
                self.save_prefs(&mut fx);
            }
            AppAction::SetDisplayScale(s) => {
                self.prefs.vm_scale = s;
                self.save_prefs(&mut fx);
            }
            AppAction::ConfigEdited => fx.extend(self.mark_dirty()),
            AppAction::Flush => {
                if self.is_dirty() {
                    self.flush(&mut fx);
                }
            }
            AppAction::RequestQuit => self.request_quit(&mut fx),
            AppAction::Session(a) => {
                use super::session::SessionAction as S;
                // A running guest owns its disks: the browser lets go first.
                if matches!(a, S::PowerOn | S::RestoreState(_)) && !self.close_browser(&mut fx) {
                    return fx;
                }
                self.session_apply(a, &mut fx)
            }
            AppAction::Camera(a) => self.camera.apply(a, &self.cfg),
            AppAction::Network(a) => self.network_apply(a, &mut fx),
            AppAction::Serial(a) => self.serial.apply(a, &self.cfg, self.notify.clone()),
            AppAction::Settings(a) => {
                use super::settings::{SettingsAction, SettingsController};
                let live = match &a {
                    SettingsAction::Set { field, .. } => SettingsController::live_change(field),
                    SettingsAction::Forward(_) => true,
                    _ => false,
                };
                let dir = self.machine_dir();
                let disks_before = self.disk_paths();
                let (changed, more) = self.settings.apply(a, &mut self.cfg, dir);
                fx.extend(more);
                if changed {
                    self.watch_attached(&disks_before);
                    fx.extend(self.mark_dirty());
                    if live && self.session.is_running() {
                        use crate::model::handle::Cmd;
                        if let Some(subnet) = self.cfg.nat_subnet.clone().filter(|s| iris::config::parse_nat_subnet(s).is_ok()) {
                            self.session.send(Cmd::SetNatSubnet(subnet));
                        }
                        self.session.send(Cmd::SetPortForwards(self.cfg.port_forward.clone()));
                        self.session.send(Cmd::SetPcapInterface(self.cfg.network.pcap_interface.clone()));
                    }
                }
            }
            AppAction::Display(a) => {
                let ps2 = self.session.ps2();
                let guest = ps2.as_deref().map(|p| p as &dyn super::display::GuestInput);
                let per_line = self.cfg.mouse_scroll_pixels_per_line;
                let follower = if super::display::follows_pointer(&self.prefs) { self.session.follower() } else { None };
                let pointer = follower.as_deref().map(|f| f as &dyn super::display::GuestPointer);
                fx.extend(self.display.apply_with(a, guest, pointer, &mut self.prefs, per_line));
            }
            AppAction::OpenBrowser { key } => self.open_browser(key, &mut fx),
            AppAction::BrowseFiles => {
                let targets = self.browse_targets();
                match targets.len() {
                    0 => fx.push(Effect::Toast(t!("browser-no-disks"))),
                    1 => self.open_browser(targets[0].0, &mut fx),
                    n => {
                        let mut choices: Vec<String> = targets.iter().map(|(_, t)| t.label.clone()).collect();
                        choices.push(t!("common-cancel"));
                        fx.push(Effect::Choose(ChoiceSpec {
                            purpose: BROWSE_WHICH,
                            title: t!("browser-which-title"),
                            message: String::new(),
                            choices,
                            dismiss: n,
                            default: Some(0),
                            destructive: Vec::new(),
                        }));
                    }
                }
            }
            AppAction::Browser(a) => {
                if let Some(b) = &mut self.browser {
                    fx.extend(b.apply(a));
                }
            }
            AppAction::CloseBrowser => {
                self.close_browser(&mut fx);
            }
            AppAction::ConvertDisk { key } => self.ask_convert(key, &mut fx),
            AppAction::CancelConversion => self.cancel_convert(),
            AppAction::DeclineConvert { key } => self.decline_convert(key, &mut fx),
            AppAction::Answer(a) => {
                if self.convert_answer(&a, &mut fx) {
                    return fx;
                }
                if a.purpose().starts_with("browser.") {
                    return self.apply(AppAction::Browser(super::disk_browser::BrowserAction::Answer(a)));
                }
                if let Answer::Chose { purpose: BROWSE_WHICH, index } = a {
                    if let Some((key, _)) = self.browse_targets().get(index) {
                        self.open_browser(*key, &mut fx);
                    }
                    return fx;
                }
                if a.purpose().starts_with("network.") {
                    self.network_apply(super::network::NetworkAction::Answer(a), &mut fx);
                } else if a.purpose().starts_with("settings.") {
                    fx.extend(self.apply(AppAction::Settings(super::settings::SettingsAction::Answer(a))));
                } else if a.purpose().starts_with("session.") {
                    self.session_apply(super::session::SessionAction::Answer(a), &mut fx);
                } else {
                    self.answer(a, &mut fx);
                }
            }
        }
        fx
    }

    fn tick(&mut self) -> Vec<Effect> {
        let mut fx = std::mem::take(&mut self.startup);
        self.session.drain(&mut fx);
        let events = self.route_session_events(&mut fx);
        use super::session::SessionEvent;
        // A browser reading through the machine has nothing to read once it
        // stops: close it as soon as a stop or a disk sync begins.
        let machine_going = !self.session.is_running() || self.session.stopping() || self.session.syncing().is_some();
        if machine_going && self.browser.as_ref().is_some_and(|b| b.live()) {
            // A job still reading through the machine: again next tick.
            if self.browser.as_mut().is_some_and(|b| b.close()) {
                self.browser = None;
                fx.push(Effect::Toast(t!("browser-closed-stopped")));
            }
        }
        if events.iter().any(|e| matches!(e, SessionEvent::Stopped | SessionEvent::Halted | SessionEvent::PoweredOff)) {
            if self.display.captured() && events.contains(&SessionEvent::Halted) {
                // Give the cursor back once the guest is safe to stop.
                fx.push(Effect::Toast(t!("toast-guest-halted")));
            }
            self.display.force_release();
        }
        let ps2 = self.session.ps2();
        self.display.tick(ps2.as_deref().map(|p| p as &dyn super::display::GuestInput));
        if events.contains(&SessionEvent::Started) {
            self.network.started();
            self.serial.reconnect(&self.cfg, self.notify.clone());
        }
        self.session_events.extend(events);
        self.network_tick(&mut fx);
        if let Some(b) = &mut self.browser {
            fx.extend(b.tick());
        }
        self.convert_tick(&mut fx);
        // Reset: the stop has finished (disks synchronized, nothing pending),
        // so start it again.
        if self.restart_after_stop && !self.session.is_running() && self.session.is_idle() && !self.session.stopping() {
            self.restart_after_stop = false;
            self.power_on(false, &mut fx);
        }
        if self.dirty_since.is_some_and(|t| t.elapsed() >= AUTOSAVE_AFTER) {
            self.flush(&mut fx);
        }
        fx
    }
}

impl AppController {
    /// Session events (started, stopped, halted, powered off, sync done) since
    /// the last call, for the parts of the shell that react to them.
    pub fn take_session_events(&mut self) -> Vec<super::session::SessionEvent> {
        std::mem::take(&mut self.session_events)
    }

    pub(crate) fn apply_config_edited(&mut self) -> Vec<Effect> {
        self.mark_dirty()
    }

    pub(crate) fn apply_flush(&mut self) -> Vec<Effect> {
        let mut fx = Vec::new();
        self.flush(&mut fx);
        fx
    }

    /// The time until the autosave is due, so a shell that sleeps between
    /// wakes can schedule one.
    pub fn autosave_due_in(&self) -> Option<Duration> {
        self.dirty_since.map(|t| AUTOSAVE_AFTER.saturating_sub(t.elapsed()))
    }

    /// When the controller next needs a tick with nothing else to wake the
    /// shell: the autosave, or stopping a machine IRIX has powered off.
    pub fn wake_due_in(&self) -> Option<Duration> {
        [self.autosave_due_in(), self.session.stop_due_in(), self.display.release_due_in()].into_iter().flatten().min()
    }

    /// A quit is under way: its disk sync runs, and `Effect::Quit` follows.
    pub fn quit_pending(&self) -> bool {
        self.quit_after_sync
    }

    /// A disk sync ended: if it was the quit's first step, finish quitting.
    #[doc(hidden)]
    pub fn sync_finished(&mut self) -> Vec<Effect> {
        let mut fx = Vec::new();
        if self.quit_after_sync {
            if self.is_dirty() {
                self.flush(&mut fx);
            }
            fx.push(Effect::Quit);
        }
        fx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::job::NoWake;
    use crate::model::machines::testing::TempRoot;

    fn app_in(root: &TempRoot, names: &[&str]) -> AppController {
        let mut prefs = GuiSettings::load_in(root.0.clone(), &root.0);
        for n in names {
            let mut cfg = MachineConfig::default();
            crate::model::hardware::clear_disks(&mut cfg);
            prefs.save_machine(n, &mut cfg).unwrap();
        }
        prefs.active_machine = names.first().map(|s| s.to_string());
        prefs.save().unwrap();
        AppController::new(GuiSettings::load_in(root.0.clone(), &root.0), Arc::new(NoWake))
    }

    // The working directory is process-wide: run the cwd-changing tests in
    // one thread, one after another.
    static CWD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn library_groups_by_family_and_folders_come_first() {
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("lib");
        let mut app = app_in(&root, &["alpha", "beta"]);
        let groups = app.library();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].key, GroupKey::Family(Family::Indy));
        assert_eq!(groups[0].rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), ["alpha", "beta"]);
        assert!(groups[0].rows[0].selected);
        assert_eq!(groups[0].rows[0].dot, None, "a stopped machine with no problem has no dot");

        app.apply(AppAction::NewFolder);
        app.apply(AppAction::Answer(Answer::Text { purpose: NEW_FOLDER, text: Some("Work".into()) }));
        app.apply(AppAction::MoveToFolder { machine: "beta".into(), folder: Some("Work".into()) });
        let groups = app.library();
        assert_eq!(groups[0].key, GroupKey::Folder("Work".into()));
        assert_eq!(groups[0].rows[0].name, "beta");
        assert_eq!(groups[1].rows.len(), 1, "beta is no longer under its family");

        // Folders persist in gui.json.
        let reloaded = GuiSettings::load_in(root.0.clone(), &root.0);
        assert_eq!(reloaded.library_folders[0].machines, ["beta"]);
        app.shutdown();
    }

    #[test]
    fn rename_and_delete_go_through_prompts() {
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("rename");
        let mut app = app_in(&root, &["alpha", "beta"]);
        let fx = app.apply(AppAction::Rename { machine: "alpha".into() });
        let Some(Effect::AskText(spec)) = fx.first() else { panic!("{fx:?}") };
        assert!(spec.validate("alpha").is_err(), "unchanged");
        assert!(spec.validate("gamma").is_ok());
        app.apply(AppAction::Answer(Answer::Text { purpose: RENAME_MACHINE, text: Some("gamma".into()) }));
        assert_eq!(app.selected(), Some("gamma"));
        assert_eq!(std::env::current_dir().unwrap(), app.prefs.machine_dir("gamma").unwrap());

        let fx = app.apply(AppAction::Delete { machine: "gamma".into() });
        assert!(matches!(fx.first(), Some(Effect::Confirm(c)) if c.destructive));
        // A "no" leaves it alone.
        app.apply(AppAction::Answer(Answer::Confirmed { purpose: DELETE_MACHINE, yes: false }));
        assert!(app.prefs.machines.contains_key("gamma"));
        app.apply(AppAction::Delete { machine: "gamma".into() });
        app.apply(AppAction::Answer(Answer::Confirmed { purpose: DELETE_MACHINE, yes: true }));
        assert!(!app.prefs.machines.contains_key("gamma"));
        assert_eq!(app.selected(), Some("beta"));
        app.shutdown();
    }

    #[test]
    fn edits_autosave_after_a_pause() {
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("autosave");
        let mut app = app_in(&root, &["alpha"]);
        app.config_mut().serial_log = Some("serial.log".into());
        app.apply(AppAction::ConfigEdited);
        assert!(app.is_dirty());
        assert!(app.tick().is_empty() && app.is_dirty(), "not yet");
        app.dirty_since = Some(Instant::now() - AUTOSAVE_AFTER);
        app.tick();
        assert!(!app.is_dirty());
        let saved = machines::read(&app.prefs.machine_path("alpha").unwrap()).unwrap();
        assert_eq!(saved.serial_log.as_deref(), Some("serial.log"));
        app.shutdown();
    }

    #[test]
    fn a_light_runs_the_command_it_stands_for() {
        use super::super::session::LightTarget;
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("lights");
        let mut app = app_in(&root, &["alpha"]);
        let disk = root.0.join("disk.img");
        std::fs::write(&disk, [0u8; 512]).unwrap();
        app.config_mut().set_disk(1, iris::config::ScsiDeviceConfig { path: disk.to_string_lossy().into(), ..Default::default() });
        app.config_mut().set_disk(4, iris::config::ScsiDeviceConfig { cdrom: true, ..Default::default() });
        app.apply(AppAction::ConfigEdited);
        let bar = app.status_bar();
        let id_of = |t: LightTarget| bar.lights.iter().find(|l| l.target == t).map(|l| l.id.clone()).expect("light");
        let disk_light = id_of(LightTarget::Disk { key: 1 });
        let cd_light = id_of(LightTarget::CdRom { key: 4 });
        assert_eq!(app.light_command(&disk_light).as_deref(), Some("browse:1"));
        assert_eq!(app.light_command(&cd_light).as_deref(), Some("disc.insert:4"));
        // The disk opens in the browser (the image is a blank file: no label).
        assert!(app.command(&disk_light).shell.is_none());
        assert!(app.browser().is_some());
        app.apply(AppAction::CloseBrowser);
        assert!(app.browser().is_none());
        // A CD-ROM light asks for a disc.
        let fx = app.command(&cd_light).effects;
        assert!(fx.iter().any(|e| matches!(e, Effect::PickFile(_))), "{fx:?}");
        assert_eq!(app.light_command("toolbar.power-on"), None);
        app.shutdown();
    }

    /// A disc command names a CD-ROM by its slot key, as the machine's disc
    /// changers number drives, so a drive on the second controller is
    /// reached and not the same id on the first.
    #[test]
    fn a_disc_command_names_the_drive_by_its_slot() {
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("cdslot");
        let mut app = app_in(&root, &["alpha"]);
        app.config_mut().machine.profile = iris::config::MachineProfile::Indigo2Ip22;
        crate::model::hardware::clear_disks(app.config_mut());
        let key = iris::config::scsi_slot(1, 4);
        app.config_mut().set_disk(key, iris::config::ScsiDeviceConfig { cdrom: true, ..Default::default() });
        app.config_mut().set_disk(1, iris::config::ScsiDeviceConfig::default());
        assert_eq!(app.cd_id(key), Some(20));
        assert_eq!(app.cd_id(1), None, "a disk is not a drive");
        assert_eq!(app.cd_id(4), None, "nothing at 0:4");
    }

    #[test]
    fn the_disk_menu_offers_what_each_device_can_do() {
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("diskmenu");
        let mut app = app_in(&root, &["alpha"]);
        let disk = root.0.join("disk.img");
        std::fs::write(&disk, [0u8; 512]).unwrap();
        let iso = root.0.join("irix.iso");
        std::fs::write(&iso, [0u8; 2048]).unwrap();
        crate::model::hardware::clear_disks(app.config_mut());
        app.config_mut().set_disk(1, iris::config::ScsiDeviceConfig { path: disk.to_string_lossy().into(), ..Default::default() });
        app.config_mut().set_disk(4, iris::config::ScsiDeviceConfig { path: iso.to_string_lossy().into(), cdrom: true, ..Default::default() });
        app.apply(AppAction::ConfigEdited);
        let menu = app.disk_menu();
        assert_eq!(menu.len(), 2);
        let ids: Vec<&str> = menu[1].entries.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, ["disc.insert:4", "disc.eject:4", "browse:4"]);
        // Eject needs a running machine; the controller says why.
        assert!(matches!(menu[1].entries[1].enabled, Enabled::No(_)));
        assert_eq!(menu[0].entries[0].id, "browse:1");
        // No overlay file yet: no copy-on-write entries.
        assert_eq!(menu[0].entries.len(), 1);
        app.shutdown();
    }

    #[test]
    fn a_slow_chd_is_asked_about_once_when_attached() {
        use super::super::convert::CONVERT_FORMAT;
        use libchdman_rs::hd::{create_from_reader, HdCreateOptions};
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("convert-ask");
        let mut app = app_in(&root, &["alpha"]);
        let chd = root.0.join("slow.chd");
        let data = vec![7u8; 1 << 20];
        let opts = HdCreateOptions {
            logical_size: 1 << 20,
            hunk_size: 4096,
            unit_size: 512,
            codecs: [libchdman_rs::CHD_CODEC_LZMA, 0, 0, 0],
            geometry: None,
            ident: None,
        };
        create_from_reader(&data[..], &chd, opts, &mut |_| {}, &|| false).unwrap();
        let path = chd.to_string_lossy().into_owned();
        let attach = |app: &mut AppController| {
            crate::model::hardware::clear_disks(app.config_mut());
            app.config_mut().set_disk(1, iris::config::ScsiDeviceConfig { path: path.clone(), ..Default::default() });
            app.apply(AppAction::ConfigEdited);
            app.watch_attached(&Default::default());
        };
        // Wait for the disk probe, then for the question.
        let question = |app: &mut AppController| {
            let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while std::time::Instant::now() < until {
                for e in app.tick() {
                    if let Effect::Choose(c) = e {
                        if c.purpose == CONVERT_FORMAT {
                            return Some(c);
                        }
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            None
        };
        attach(&mut app);
        let c = question(&mut app).expect("a slow CHD is asked about when attached");
        assert_eq!(c.choices, [t!("convert-to-zlib"), t!("convert-to-uncompressed"), t!("convert-keep")]);
        assert!(app.suggest_convert(1), "the machine page points it out until answered");
        app.apply(AppAction::Answer(Answer::Chose { purpose: CONVERT_FORMAT, index: 2 }));
        assert!(!app.suggest_convert(1), "Keep As Is stops the suggestion");
        let saved = GuiSettings::load_in(root.0.clone(), &root.0);
        assert!(saved.convert_declined.contains(&path), "and is remembered");
        // Attached again: no question, but the Devices menu still converts it.
        attach(&mut app);
        assert!(question(&mut app).is_none());
        assert!(app.disk_menu()[0].entries.iter().any(|e| e.id == "disk.convert:1"));
        app.shutdown();
    }

    #[test]
    fn nothing_else_touches_a_disk_while_it_converts() {
        use super::super::convert::CONVERT_FORMAT;
        use super::super::session::SessionAction;
        use libchdman_rs::hd::{create_from_reader, HdCreateOptions};
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("convert-busy");
        let mut app = app_in(&root, &["alpha"]);
        let chd = root.0.join("slow.chd");
        let opts = HdCreateOptions {
            logical_size: 1 << 20,
            hunk_size: 4096,
            unit_size: 512,
            codecs: [libchdman_rs::CHD_CODEC_LZMA, 0, 0, 0],
            geometry: None,
            ident: None,
        };
        create_from_reader(&vec![7u8; 1 << 20][..], &chd, opts, &mut |_| {}, &|| false).unwrap();
        crate::model::hardware::clear_disks(app.config_mut());
        app.config_mut().set_disk(1, iris::config::ScsiDeviceConfig { path: chd.to_string_lossy().into_owned(), ..Default::default() });
        app.apply(AppAction::ConfigEdited);
        app.watch_attached(&Default::default());
        let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut asked = false;
        while !asked && std::time::Instant::now() < until {
            asked = app.tick().iter().any(|e| matches!(e, Effect::Choose(c) if c.purpose == CONVERT_FORMAT));
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(asked, "the slow CHD is asked about");
        assert_eq!(app.command_enabled(super::super::ids::toolbar::BROWSE_FILES), Enabled::Yes);
        // Convert it: browsing it, or committing or discarding its changes,
        // waits until the conversion has finished.
        app.apply(AppAction::Answer(Answer::Chose { purpose: CONVERT_FORMAT, index: 0 }));
        assert!(app.conversion().is_some(), "the conversion runs");
        assert_eq!(app.command_enabled(super::super::ids::toolbar::BROWSE_FILES), Enabled::No(t!("convert-busy")));
        assert_eq!(app.session_enabled(&SessionAction::ApplyChanges { key: 1 }), Enabled::No(t!("convert-busy")));
        assert_eq!(app.session_enabled(&SessionAction::RollBack { key: 1 }), Enabled::No(t!("convert-busy")));
        let until = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while app.conversion().is_some() && std::time::Instant::now() < until {
            app.tick();
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(app.conversion().is_none(), "the conversion finished");
        assert_eq!(app.command_enabled(super::super::ids::toolbar::BROWSE_FILES), Enabled::Yes);
        app.shutdown();
    }

    #[test]
    fn a_rows_context_menu_selects_its_machine_first() {
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("ctxmenu");
        let mut app = app_in(&root, &["alpha", "beta"]);
        assert_eq!(app.selected(), Some("alpha"));
        let menu = app.machine_menu("beta");
        let ids: Vec<&str> = menu.iter().flatten().map(|e| e.id.as_str()).collect();
        assert_eq!(ids[0], "on:beta:machine.start:beta");
        assert!(ids.contains(&"on:beta:toolbar.machine-settings"), "{ids:?}");
        assert!(ids.contains(&"on:beta:menu.rename") && ids.contains(&"on:beta:menu.delete"), "{ids:?}");
        // Settings… on beta selects beta and opens the settings.
        let r = app.command("on:beta:toolbar.machine-settings");
        assert_eq!(app.selected(), Some("beta"));
        assert_eq!(r.shell, Some(super::super::commands::ShellCommand::Settings));
        app.shutdown();
    }

    #[test]
    fn power_on_stops_to_ask_about_a_slow_chd() {
        use super::super::convert::CONVERT_FORMAT;
        use libchdman_rs::hd::{create_from_reader, HdCreateOptions};
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("convert-start");
        let mut app = app_in(&root, &["alpha"]);
        let chd = root.0.join("slow.chd");
        let opts = HdCreateOptions {
            logical_size: 1 << 20,
            hunk_size: 4096,
            unit_size: 512,
            codecs: [libchdman_rs::CHD_CODEC_LZMA, 0, 0, 0],
            geometry: None,
            ident: None,
        };
        create_from_reader(&vec![7u8; 1 << 20][..], &chd, opts, &mut |_| {}, &|| false).unwrap();
        crate::model::hardware::clear_disks(app.config_mut());
        app.config_mut().set_disk(1, iris::config::ScsiDeviceConfig { path: chd.to_string_lossy().into(), ..Default::default() });
        app.apply(AppAction::ConfigEdited);
        let fx = app.apply(AppAction::Session(super::super::session::SessionAction::PowerOn));
        let c = fx
            .iter()
            .find_map(|e| match e {
                Effect::Choose(c) if c.purpose == CONVERT_FORMAT => Some(c.clone()),
                _ => None,
            })
            .expect("Power On asks about the slow CHD first");
        assert_eq!(c.choices.len(), 4, "{:?}", c.choices);
        assert_eq!(c.choices[2], t!("convert-start-anyway"));
        assert!(!app.session().is_running(), "and does not start yet");
        // Cancel: nothing starts, nothing is remembered.
        app.apply(AppAction::Answer(Answer::Chose { purpose: CONVERT_FORMAT, index: 3 }));
        assert!(!app.session().is_running());
        assert!(app.prefs().convert_declined.is_empty());
        app.shutdown();
    }

    #[test]
    fn every_help_page_has_a_title_and_lines() {
        use super::super::help::{HelpPage, Line};
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("help");
        let mut app = app_in(&root, &["alpha"]);
        for p in HelpPage::all().into_iter().chain([HelpPage::About]) {
            let d = app.help(p, "1.2.3", "abcdef12");
            assert!(!d.title.is_empty() && !d.lines.is_empty(), "{p:?}");
            assert_eq!(HelpPage::from_id(p.id()), Some(p));
        }
        let about = app.help(HelpPage::About, "1.2.3", "abcdef12");
        assert!(about.lines.iter().any(|l| matches!(l, Line::Text(t) if t.contains("1.2.3"))));
        let lic = app.help(HelpPage::Licenses, "", "");
        assert!(lic.lines.iter().any(|l| matches!(l, Line::Code(t) if t.contains("MIT"))));
        app.shutdown();
    }

    #[test]
    fn power_off_and_quit_answers_the_running_question() {
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("quitrun");
        let mut app = app_in(&root, &["alpha"]);
        // Cancel does nothing; Power Off and Quit quits.
        assert!(app.apply(AppAction::Answer(Answer::Chose { purpose: QUIT_RUNNING, index: 0 })).is_empty());
        let fx = app.apply(AppAction::Answer(Answer::Chose { purpose: QUIT_RUNNING, index: 1 }));
        assert_eq!(fx.last(), Some(&Effect::Quit));
        app.shutdown();
    }

    #[test]
    fn quitting_with_nothing_pending_saves_and_quits() {
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("quit");
        let mut app = app_in(&root, &["alpha"]);
        let fx = app.apply(AppAction::RequestQuit);
        assert_eq!(fx.last(), Some(&Effect::Quit));
        app.shutdown();
    }

    /// Power a disk-less Indy on through the session, see it run, and power
    /// it off: at the PROM with no disks the stop needs no question.
    #[test]
    fn power_on_and_off_through_the_session() {
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("IRIS_NO_EXIT_ON_POWEROFF", "1");
        let root = TempRoot::new("power");
        let mut prefs = GuiSettings::load_in(root.0.clone(), &root.0);
        let mut cfg = MachineConfig::default();
        cfg.banks = iris::config::banks_from4([8, 8, 0, 0]);
        crate::model::hardware::clear_disks(&mut cfg);
        cfg.headless = true;
        cfg.monitor_port = Some(0);
        prefs.save_machine("indy", &mut cfg).unwrap();
        prefs.active_machine = Some("indy".into());
        prefs.save().unwrap();
        let mut app = AppController::new(GuiSettings::load_in(root.0.clone(), &root.0), Arc::new(NoWake));
        let wait = |app: &mut AppController, what: &str, done: &dyn Fn(&AppController) -> bool| {
            let deadline = Instant::now() + Duration::from_secs(20);
            while !done(app) && Instant::now() < deadline {
                app.tick();
                std::thread::sleep(Duration::from_millis(20));
            }
            assert!(done(app), "{what}");
        };
        let fx = app.apply(AppAction::Session(super::super::session::SessionAction::PowerOn));
        assert!(fx.iter().all(|e| matches!(e, Effect::Toast(_))), "{fx:?}");
        wait(&mut app, "started", &|a| a.session().is_running());
        assert!(app.take_session_events().contains(&super::super::session::SessionEvent::Started));
        assert!(std::path::Path::new("nvram.bin").exists(), "the NVRAM was seeded");
        let bar = app.status_bar();
        assert_eq!(bar.executor, super::super::session::SessionController::executor());
        let fx = app.apply(AppAction::Session(super::super::session::SessionAction::PowerOff));
        assert!(!fx.iter().any(|e| matches!(e, Effect::Choose(_))), "no disks, no question: {fx:?}");
        wait(&mut app, "stopped", &|a| !a.session().is_running() && a.session().is_idle());
        app.shutdown();
    }

    #[test]
    fn duplicate_selects_nothing_and_joins_the_folder() {
        let _g = CWD.lock().unwrap_or_else(|e| e.into_inner());
        let root = TempRoot::new("dupapp");
        let mut app = app_in(&root, &["alpha"]);
        app.apply(AppAction::NewFolder);
        app.apply(AppAction::Answer(Answer::Text { purpose: NEW_FOLDER, text: Some("Lab".into()) }));
        app.apply(AppAction::MoveToFolder { machine: "alpha".into(), folder: Some("Lab".into()) });
        app.apply(AppAction::Duplicate { machine: "alpha".into() });
        assert!(app.prefs.machines.contains_key("alpha-2"));
        assert_eq!(app.selected(), Some("alpha"));
        assert_eq!(app.prefs.library_folders[0].machines, ["alpha", "alpha-2"]);
        app.shutdown();
    }
}
