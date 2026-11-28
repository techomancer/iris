//! The session actions that need the library: power on reads the selected
//! machine's TOML and checks it, a disc insert edits the config while the
//! machine is stopped, a folder grant goes into the preferences. The worker
//! side is [`SessionController`](super::session::SessionController).

use std::path::Path;

use iris::config::{MachineConfig, MachineProfile};

use super::app::AppController;
use super::session::{self, GrantChoice, Pending, SessionAction, SessionController, SessionEvent};
use super::{Answer, ChoiceSpec, Effect, Enabled, PickSpec, TextSpec};
use crate::model::handle::Cmd;
use crate::model::{hardware, machines, macos_sandbox, settings};

/// The prompt's rule: anything `iris::config::parse_mac` takes.
fn mac_rule(_: &super::TextSpec, text: &str) -> Result<(), String> {
    match iris::config::parse_mac(text.trim()) {
        Ok(_) => Ok(()),
        Err(_) if text.trim().is_empty() => Err(String::new()),
        Err(e) => Err(e.to_string()),
    }
}

impl AppController {
    /// Whether a session action can run now, and if not why.
    pub fn session_enabled(&self, action: &SessionAction) -> Enabled {
        if self.selected().is_none() {
            return Enabled::No(t!("toast-no-machine"));
        }
        // A conversion replaces the disk file: no copy-on-write commit or
        // discard on it meanwhile.
        if self.conversion.is_some() && matches!(action, SessionAction::ApplyChanges { .. } | SessionAction::RollBack { .. }) {
            return Enabled::No(t!("convert-busy"));
        }
        self.session().enabled(action, self.config())
    }

    /// The selected machine's saved states.
    pub fn saved_states(&self) -> Vec<session::SavedState> {
        self.machine_dir().map(|d| SessionController::saved_states(&d)).unwrap_or_default()
    }

    /// The status bar for the selected machine.
    pub fn status_bar(&self) -> session::StatusBar {
        // No serial light: the toolbar has the console, and its activity says
        // nothing the user needs.
        self.session().status_bar(self.config(), self.is_dirty())
    }

    pub(crate) fn session_apply(&mut self, action: SessionAction, fx: &mut Vec<Effect>) {
        if let Enabled::No(why) = self.session_enabled(&action) {
            if !matches!(action, SessionAction::Answer(_)) {
                fx.push(Effect::Toast(why));
                return;
            }
        }
        match action {
            SessionAction::PowerOn => self.power_on(false, fx),
            SessionAction::ShutDownGuest => {
                self.session_mut().send(Cmd::HaltIrix);
                fx.push(Effect::Toast(t!("toast-halt-sent")));
            }
            SessionAction::RebootGuest => {
                self.session_mut().send(Cmd::RebootIrix);
                fx.push(Effect::Toast(t!("toast-reboot-sent")));
            }
            SessionAction::PowerOff => {
                let cfg = self.config().clone();
                self.session_mut().stop(&cfg, fx);
            }
            // Reset: the normal stop (it asks first when IRIX is busy, and
            // synchronizes the disks), then Power On once it has stopped
            // (`AppController::tick`).
            SessionAction::Restart => {
                self.restart_after_stop = true;
                let cfg = self.config().clone();
                self.session_mut().stop(&cfg, fx);
            }
            SessionAction::AskSaveState => {
                let taken: Vec<String> = self.saved_states().into_iter().map(|s| s.name).collect();
                let mut n = 1;
                while taken.contains(&format!("snap{n}")) {
                    n += 1;
                }
                fx.push(Effect::AskText(TextSpec {
                    purpose: session::STATE_NAME,
                    title: t!("session-save-state-title"),
                    prompt: t!("session-save-state-prompt"),
                    initial: format!("snap{n}"),
                    ok_label: t!("common-save"),
                    taken,
                    error: None,
                    validator: save_name_rule,
                }));
                self.session_mut().pending = Pending::StateName;
            }
            SessionAction::SaveState(name) => match plain_state_name(&name) {
                Some(name) => self.session_mut().send(Cmd::SaveState(name)),
                None => fx.push(Effect::Toast(t!("session-state-name-invalid"))),
            },
            SessionAction::RestoreState(name) => {
                if self.session().is_running() {
                    self.session_mut().send(Cmd::RestoreState(name));
                } else {
                    // Armed only once the start has really been sent: a start
                    // that stops to ask, or fails its checks, must not leave
                    // the restore waiting for some later Power On.
                    self.power_on(false, fx);
                    if self.session().is_starting() {
                        self.session_mut().restore_when_started(name);
                    }
                }
            }
            SessionAction::RenameState(name) => {
                let taken: Vec<String> = self.saved_states().into_iter().map(|s| s.name).collect();
                fx.push(Effect::AskText(TextSpec {
                    purpose: session::RENAME_STATE,
                    title: t!("session-rename-state-title"),
                    prompt: t!("session-save-state-prompt"),
                    initial: name.clone(),
                    ok_label: t!("modal-rename-confirm"),
                    taken,
                    error: None,
                    validator: session::state_name_rule,
                }));
                self.session_mut().pending = Pending::RenameState(name);
            }
            SessionAction::DeleteState(name) => {
                fx.push(Effect::Confirm(super::ConfirmSpec {
                    purpose: session::DELETE_STATE,
                    title: t!("session-delete-state-title", name = &name),
                    message: t!("session-delete-state-body"),
                    confirm_label: t!("library-delete-ok"),
                    destructive: true,
                }));
                self.session_mut().pending = Pending::DeleteState(name);
            }
            SessionAction::Screenshot => {
                fx.push(session::screenshot_pick(self.machine_dir()));
                self.session_mut().pending = Pending::Screenshot;
            }
            SessionAction::InsertDisc { key } => {
                fx.push(session::disc_pick(None));
                self.session_mut().pending = Pending::InsertDisc(key);
            }
            SessionAction::LoadDisc { key, path } => self.load_disc(key, path, fx),
            SessionAction::EjectDisc { key } => {
                if let Some(id) = self.cd_id(key) {
                    self.session_mut().send(Cmd::EjectCdrom { id });
                }
            }
            SessionAction::RemountDisc { key } => {
                if let Some(id) = self.cd_id(key) {
                    self.session_mut().send(Cmd::RemountCdrom { id });
                }
            }
            SessionAction::ApplyChanges { key } => {
                if let Some((base, chd)) = self.overlay(key) {
                    if chd {
                        self.session_mut().syncing = Some(session::SyncProgress { disk: 0, total: 1, fraction: 0.0 });
                    }
                    self.session_mut().send(Cmd::CowCommit { base, chd });
                }
            }
            SessionAction::RollBack { key } => {
                if let Some((base, _)) = self.overlay(key) {
                    let file = Path::new(&base).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or(base);
                    let id = hardware::scsi_name(key);
                    fx.push(session::roll_back_confirm(&id, &file));
                    self.session_mut().pending = Pending::RollBack(key);
                }
            }
            SessionAction::SynchronizeDisks => self.session_mut().stop_now(fx),
            SessionAction::CheckFolderGrants => {
                let needs = machines::chd_dirs_needing_grant(self.config());
                if !needs.is_empty() {
                    fx.push(grant_question(&needs, false, &mut self.session_mut().pending));
                }
            }
            SessionAction::GrantDiskFolder => fx.push(Effect::PickFolder(PickSpec::open(
                session::GRANT_FOLDER,
                t!("main-grant-folder-title"),
            ))),
            SessionAction::RevokeDiskFolder(path) => {
                self.prefs_mut().disk_folders.retain(|x| *x != path);
                self.prefs_mut().bookmarks.remove(&path);
                let _ = self.prefs_mut().save();
            }
            SessionAction::ResetNvram => fx.push(Effect::Confirm(super::ConfirmSpec {
                purpose: session::RESET_NVRAM,
                title: t!("session-reset-nvram-title"),
                message: t!("session-reset-nvram-body"),
                confirm_label: t!("session-reset-nvram-ok"),
                destructive: true,
            })),
            SessionAction::SetMac => {
                let cfg = self.config();
                let eeprom = crate::model::hardware::has_nveeprom(cfg.machine.profile);
                let path = if eeprom { cfg.nveeprom.clone() } else { cfg.nvram.clone() };
                let dir = self.machine_dir();
                let full = dir.map(|d| d.join(&path).to_string_lossy().into_owned()).unwrap_or(path);
                let now = if eeprom { settings::nveeprom_mac(&full) } else { settings::nvram_mac(&full) };
                // Nothing set yet: offer one, as the first start would.
                let initial = now
                    .map(settings::mac_to_string)
                    .unwrap_or_else(|| settings::mac_to_string(settings::generate_mac_bytes(&full)));
                fx.push(Effect::AskText(super::TextSpec {
                    purpose: session::SET_MAC,
                    title: t!("session-set-mac-title"),
                    prompt: t!("session-set-mac-body", chip = full),
                    initial,
                    ok_label: t!("session-set-mac-ok"),
                    taken: Vec::new(),
                    error: None,
                    validator: mac_rule,
                }));
            }
            SessionAction::Answer(a) => self.session_answer(a, fx),
        }
    }

    /// Copy-on-write overlays that exist on disk: (key, base, is_chd).
    pub fn overlays(&self) -> Vec<(u8, String, bool)> {
        self.config()
            .disks()
            .into_iter()
            .filter_map(|(k, _)| self.overlay(k).map(|(b, c)| (k, b, c)))
            .collect()
    }

    fn overlay(&self, key: u8) -> Option<(String, bool)> {
        let dev = self.config().disk(key)?;
        if dev.is_cdrom() || dev.scratch || dev.path.trim().is_empty() {
            return None;
        }
        let chd = iris::chd_disk::is_chd(&dev.path);
        let exists = if chd {
            iris::chd_disk::diff_path_for(Path::new(&dev.path)).exists()
        } else {
            dev.overlay && Path::new(&format!("{}.overlay", dev.path)).exists()
        };
        exists.then(|| (dev.path.clone(), chd))
    }

    /// The drive number a disc command takes for the CD-ROM at `key`: the
    /// machine's disc changers number drives `16 * controller + id`, the
    /// slot key itself, so a drive on any controller is reached.
    pub(crate) fn cd_id(&self, key: u8) -> Option<u8> {
        let dev = self.config().disk(key)?;
        dev.is_cdrom().then_some(key)
    }

    fn load_disc(&mut self, key: u8, path: String, fx: &mut Vec<Effect>) {
        if self.session().is_running() {
            if let Some(id) = self.cd_id(key) {
                // No remount command on the console: IRIX's mediad notices a
                // new disc by itself.
                self.session_mut().send(Cmd::LoadDisc { id, path, remount: false });
            }
        } else if let Some(dev) = self.config_mut().disk_mut(key) {
            // Stopped: the disc is the drive's medium at the next power on.
            dev.path = path;
            fx.extend(self.apply_config_edited());
        }
    }

    /// Write `text` into the chip the machine reads its address from: the
    /// Indigo2's motherboard EEPROM, every other machine's NVRAM.
    fn set_mac(&mut self, text: &str, fx: &mut Vec<Effect>) {
        let Ok(mac) = iris::config::parse_mac(text.trim()) else {
            fx.push(Effect::Toast(t!("session-set-mac-invalid")));
            return;
        };
        let cfg = self.config();
        let eeprom = crate::model::hardware::has_nveeprom(cfg.machine.profile);
        let path = if eeprom { cfg.nveeprom.clone() } else { cfg.nvram.clone() };
        let dir = self.machine_dir();
        let full = dir.map(|d| d.join(&path).to_string_lossy().into_owned()).unwrap_or(path);
        let wrote = if eeprom {
            settings::ensure_nveeprom_exists(&full);
            settings::write_nveeprom_mac(&full, mac)
        } else {
            settings::ensure_nvram_seeded(&full);
            settings::write_nvram_mac(&full, mac)
        };
        fx.push(Effect::Toast(match wrote {
            Ok(_) => t!("session-set-mac-done", mac = settings::mac_to_string(mac)),
            Err(e) => t!("config-eaddr-write-failed", path = full.as_str(), error = e.to_string()),
        }));
    }

    fn reset_nvram(&mut self, fx: &mut Vec<Effect>) {
        let cfg = self.config().clone();
        let seed = self.selected().unwrap_or("indy").to_string();
        let mac = settings::generate_mac_bytes(&seed);
        if cfg.machine.profile == MachineProfile::OctaneIp30 {
            // A blank DS1685 NVRAM is the Octane's factory state: keep a
            // backup and remove the file; without a backup, keep the file.
            if std::path::Path::new(&cfg.nvram).exists() {
                if let Err(e) = std::fs::copy(&cfg.nvram, format!("{}.bak", cfg.nvram)) {
                    fx.push(Effect::Toast(t!("toast-nvram-reset-failed", error = e.to_string())));
                    return;
                }
            }
            fx.push(Effect::Toast(match std::fs::remove_file(&cfg.nvram) {
                Ok(()) => t!("toast-nvram-reset-octane"),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => t!("toast-nvram-reset-octane"),
                Err(e) => t!("toast-nvram-reset-failed", error = e.to_string()),
            }));
            return;
        }
        // Indigo2 and IP28 read eaddr and the PROM environment from the
        // EEPROM, the Indy from the NVRAM: reset and re-MAC both.
        let mut errors = Vec::new();
        match settings::reset_nvram(&cfg.nvram) {
            Ok(()) => {
                let _ = settings::write_nvram_mac(&cfg.nvram, mac);
            }
            Err(e) => errors.push(t!("toast-nvram-reset-failed", error = e.to_string())),
        }
        match settings::reset_nveeprom(&cfg.nveeprom) {
            Ok(()) => {
                let _ = settings::write_nveeprom_mac(&cfg.nveeprom, mac);
            }
            Err(e) => errors.push(t!("toast-nveeprom-reset-failed", error = e.to_string())),
        }
        fx.push(Effect::Toast(if errors.is_empty() {
            t!("toast-nvram-reset", mac = settings::mac_to_string(mac))
        } else {
            errors.join("\n")
        }));
    }

    /// Power on the selected machine: write pending edits, read the TOML
    /// back, check it, and start. `skip_grant` is the one-shot "start without
    /// merging" answer to the CHD folder question.
    pub(crate) fn power_on(&mut self, skip_grant: bool, fx: &mut Vec<Effect>) {
        if let Enabled::No(why) = self.can_change_machine() {
            fx.push(Effect::Toast(why));
            return;
        }
        // A running guest owns its disks: an editable browser lets go first,
        // however the start was asked for (the toolbar, a finished conversion,
        // a restore, a reset).
        if !self.close_browser(fx) {
            return;
        }
        if self.is_dirty() {
            fx.extend(self.apply_flush());
            if self.is_dirty() {
                return;
            }
        }
        let Some(name) = self.selected().map(str::to_string) else { return };
        let result = (|| -> Result<MachineConfig, String> {
            let cfg = machines::read(&self.prefs().machine_path(&name)?)?;
            machines::enter(&self.prefs().machine_dir(&name)?)?;
            Ok(cfg)
        })();
        let mut cfg = match result {
            Ok(cfg) => cfg,
            Err(e) => {
                fx.push(Effect::Toast(t!("toast-start-failed", error = e)));
                return;
            }
        };
        // Files edited by hand can hold absolute paths; normalise them before
        // either the GUI or the emulator uses the saved config.
        let before = machines::to_toml(&cfg).unwrap_or_default();
        let Ok(dir) = self.prefs().machine_dir(&name) else { return };
        if let Err(e) = machines::make_relative(&mut cfg, &dir) {
            fx.push(Effect::Toast(e));
            return;
        }
        if machines::to_toml(&cfg).unwrap_or_default() != before {
            if let Err(e) = self.prefs_mut().save_machine(&name, &mut cfg) {
                fx.push(Effect::Toast(e));
                return;
            }
        }
        *self.config_mut() = cfg.clone();
        self.prefs_mut().machines.insert(name.clone(), cfg.clone());
        // Re-assert sandbox access to every bookmarked file before the
        // preflight opens them; a picker's grant does not survive a restart.
        macos_sandbox::restore(&self.prefs().bookmarks);
        if iris::build_features::LIGHTNING && cfg.gdb_port.is_some() {
            // The GDB stub does nothing in a lightning build.
            cfg.gdb_port = None;
        }
        if let Err(e) = cfg.validate() {
            fx.push(Effect::Toast(t!("toast-invalid-config", error = e.to_string())));
            return;
        }
        // Machine::new exits the process on a missing image: catch it here.
        if let Some(path) = crate::model::summary::expansion_paths(&cfg).into_iter().find(|p| !machines::readable(p)) {
            fx.push(Effect::Toast(t!("extra-unreadable-disk", path = path)));
            return;
        }
        let missing = crate::model::summary::missing_disks(&cfg, machines::readable);
        if !missing.is_empty() {
            let mut message = t!("modal-missing-body");
            for m in &missing {
                message.push('\n');
                message.push_str(&super::overview::missing_line(m));
            }
            if cfg!(feature = "appstore") {
                message.push_str("\n\n");
                message.push_str(&t!("modal-missing-sandbox"));
            }
            fx.push(Effect::Choose(ChoiceSpec {
                purpose: session::MISSING,
                title: t!("modal-missing-title"),
                message,
                choices: vec![t!("common-cancel"), t!("modal-missing-edit"), t!("modal-missing-detach")],
                dismiss: 0,
                default: Some(2),
                destructive: Vec::new(),
            }));
            self.session_mut().pending = Pending::Missing(missing.iter().map(|m| m.key).collect());
            return;
        }
        if !skip_grant {
            let needs = machines::chd_dirs_needing_grant(&cfg);
            if !needs.is_empty() {
                fx.push(grant_question(&needs, true, &mut self.session_mut().pending));
                return;
            }
        }
        // A slow compressed CHD is asked about once, before the first start.
        if self.ask_before_start(fx) {
            return;
        }
        // An older GUI wrote an empty `prom` for the built-in PROM; the core
        // reads that as a file name.
        if cfg.prom.is_empty() {
            cfg.prom = "(embedded)".into();
            self.config_mut().prom = cfg.prom.clone();
            fx.extend(self.apply_config_edited());
        }
        let mut notes = Vec::new();
        if hardware::builtin_prom(&cfg) {
            notes.push(t!("toast-prom-builtin"));
        }
        cfg.debug.apply_env();
        cfg.jitv2.apply_env();
        // An automatic `[jitv2] threads` follows the CPU count chosen since.
        cfg.settle_jitv2_threads();
        // Seed the Indy/Indigo2 battery files and give the NVRAM (and the
        // Indigo2's EEPROM) an Ethernet address before the first boot, so IRIX
        // attaches ec0 at once. Other platforms keep their own stores.
        if hardware::mc_machine(cfg.machine.profile) {
            if settings::ensure_nvram_seeded(&cfg.nvram) {
                notes.push(t!("toast-nvram-seeded"));
            }
            if !settings::nvram_has_mac(&cfg.nvram) {
                let mac = settings::generate_mac_bytes(&name);
                notes.push(match settings::write_nvram_mac(&cfg.nvram, mac) {
                    Ok(true) => t!("toast-nvram-mac-written", mac = settings::mac_to_string(mac)),
                    Ok(false) => t!("toast-nvram-mac-no-file"),
                    Err(e) => t!("toast-nvram-mac-failed", error = e.to_string()),
                });
            }
            if !settings::nveeprom_has_mac(&cfg.nveeprom) {
                settings::ensure_nveeprom_exists(&cfg.nveeprom);
                let mac = settings::generate_mac_bytes(&name);
                if let Ok(true) = settings::write_nveeprom_mac(&cfg.nveeprom, mac) {
                    notes.push(t!("toast-nveeprom-mac-written", mac = settings::mac_to_string(mac)));
                }
            }
        }
        self.session_mut().start(cfg);
        if !notes.is_empty() {
            fx.push(Effect::Toast(notes.join("\n")));
        }
    }

    fn session_answer(&mut self, answer: Answer, fx: &mut Vec<Effect>) {
        let pending = std::mem::take(&mut self.session_mut().pending);
        match (answer, pending) {
            (Answer::Chose { purpose: session::STOP, index }, _) => match index {
                1 => {
                    self.session_mut().send(Cmd::HaltIrix);
                    fx.push(Effect::Toast(t!("toast-halt-sent")));
                }
                2 => {
                    self.session_mut().mark_stopping();
                    fx.push(Effect::Toast(t!("toast-stopping")));
                    self.session_mut().send(Cmd::Stop);
                }
                // Cancelled: no stop, so no reset either.
                _ => self.restart_after_stop = false,
            },
            (Answer::Chose { purpose: session::MISSING, index }, Pending::Missing(keys)) => match index {
                1 => fx.push(Effect::OpenSettings(super::ids::settings::DISKS)),
                2 => {
                    for k in keys {
                        self.config_mut().remove_disk(k);
                    }
                    fx.extend(self.apply_config_edited());
                    self.power_on(false, fx);
                }
                _ => {}
            },
            (Answer::Chose { purpose: session::GRANT, index }, Pending::Grant(choices, then_start)) => {
                match choices.get(index).cloned() {
                    Some(GrantChoice::Grant(dir)) => {
                        fx.push(Effect::PickFolder(
                            PickSpec::open(session::GRANT_FOLDER, t!("main-grant-folder-title")).in_dir(Some(dir.into())),
                        ));
                        self.session_mut().pending = Pending::GrantFolder(then_start);
                    }
                    Some(GrantChoice::OpenPrivacy) => {
                        fx.push(Effect::OpenUrl(
                            "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles".into(),
                        ));
                        // Ask again, so "Check again" is there when they return.
                        self.recheck_grants(then_start, fx);
                    }
                    Some(GrantChoice::Recheck) => self.recheck_grants(then_start, fx),
                    Some(GrantChoice::StartAnyway) if then_start => self.power_on(true, fx),
                    Some(GrantChoice::StartAnyway) | Some(GrantChoice::Cancel) | None => {}
                }
            }
            (Answer::Picked { purpose: session::GRANT_FOLDER, paths }, pending) => {
                if let Some(dir) = paths.first() {
                    let path = dir.to_string_lossy().into_owned();
                    if !self.prefs().disk_folders.contains(&path) {
                        self.prefs_mut().disk_folders.push(path.clone());
                    }
                    let _ = self.prefs_mut().save(); // mints the directory bookmark
                    macos_sandbox::restore(&self.prefs().bookmarks);
                    fx.push(Effect::Toast(t!("toast-folder-granted", path = &path)));
                }
                if let Pending::GrantFolder(then_start) = pending {
                    self.recheck_grants(then_start, fx);
                }
            }
            (Answer::Text { purpose: session::STATE_NAME, text: Some(name) }, _) => match plain_state_name(&name) {
                Some(name) => self.session_mut().send(Cmd::SaveState(name)),
                None => fx.push(Effect::Toast(t!("session-state-name-invalid"))),
            },
            (Answer::Text { purpose: session::RENAME_STATE, text: Some(new) }, Pending::RenameState(old)) => {
                let (Some(old), Some(new)) = (plain_state_name(&old), plain_state_name(&new)) else {
                    fx.push(Effect::Toast(t!("session-state-name-invalid")));
                    return;
                };
                if let Some(dir) = self.machine_dir() {
                    let saves = dir.join("saves");
                    if let Err(e) = std::fs::rename(saves.join(&old), saves.join(&new)) {
                        fx.push(Effect::Toast(t!("session-state-rename-failed", error = e.to_string())));
                    }
                }
            }
            (Answer::Confirmed { purpose: session::DELETE_STATE, yes: true }, Pending::DeleteState(name)) => {
                // Only a state's own folder: an empty or odd name would name
                // `saves` itself, or something outside it.
                let Some(name) = plain_state_name(&name) else { return };
                if let Some(dir) = self.machine_dir() {
                    if let Err(e) = std::fs::remove_dir_all(dir.join("saves").join(&name)) {
                        fx.push(Effect::Toast(t!("session-state-delete-failed", error = e.to_string())));
                    }
                }
            }
            (Answer::Confirmed { purpose: session::ROLL_BACK, yes: true }, Pending::RollBack(key)) => {
                if let Some((base, chd)) = self.overlay(key) {
                    self.session_mut().send(Cmd::CowReset { base, chd });
                }
            }
            (Answer::Confirmed { purpose: session::RESET_NVRAM, yes: true }, _) => self.reset_nvram(fx),
            (Answer::Text { purpose: session::SET_MAC, text: Some(text) }, _) => self.set_mac(&text, fx),
            (Answer::Picked { purpose: session::SCREENSHOT, paths }, _) => {
                if let Some(p) = paths.into_iter().next() {
                    self.session_mut().send(Cmd::Screenshot(p));
                }
            }
            (Answer::Picked { purpose: session::INSERT_DISC, paths }, Pending::InsertDisc(key)) => {
                if let Some(p) = paths.into_iter().next() {
                    self.load_disc(key, p.to_string_lossy().into_owned(), fx);
                }
            }
            // Cancelled questions, and answers that belong to another question.
            (_, _) => {}
        }
    }

    fn recheck_grants(&mut self, then_start: bool, fx: &mut Vec<Effect>) {
        let needs = machines::chd_dirs_needing_grant(self.config());
        if !needs.is_empty() {
            fx.push(grant_question(&needs, then_start, &mut self.session_mut().pending));
        } else if then_start {
            self.power_on(true, fx);
        }
    }

    /// Route the session's events: a finished sync may end a quit.
    pub(crate) fn route_session_events(&mut self, fx: &mut Vec<Effect>) -> Vec<SessionEvent> {
        let events = self.session_mut().take_events();
        if events.contains(&SessionEvent::SyncDone) {
            fx.extend(self.sync_finished());
        }
        events
    }
}

/// The question for CHD folders the app cannot write. The App Store build can
/// be granted a folder; a notarized build is blocked by macOS privacy, which
/// only System Settings can change.
fn grant_question(needs: &[machines::ChdNeedsGrant], then_start: bool, pending: &mut Pending) -> Effect {
    let mut message = t!("modal-grant-body");
    for d in needs {
        message.push('\n');
        message.push_str(&t!("modal-grant-disk", id = d.key, path = &d.path));
    }
    message.push_str("\n\n");
    message.push_str(&t!("modal-grant-why"));
    let mut choices: Vec<GrantChoice> = Vec::new();
    if cfg!(feature = "appstore") {
        message.push_str("\n\n");
        message.push_str(&t!("modal-grant-sandbox"));
        let mut dirs: Vec<&str> = needs.iter().map(|d| d.dir.as_str()).collect();
        dirs.sort_unstable();
        dirs.dedup();
        choices.extend(dirs.into_iter().map(|d| GrantChoice::Grant(d.to_string())));
    } else {
        message.push_str("\n\n");
        message.push_str(&t!("modal-grant-tcc"));
        choices.push(GrantChoice::OpenPrivacy);
        choices.push(GrantChoice::Recheck);
    }
    choices.push(GrantChoice::Cancel);
    choices.push(GrantChoice::StartAnyway);
    let labels = choices
        .iter()
        .map(|c| match c {
            GrantChoice::Grant(d) => t!("modal-grant-folder", path = d.as_str()),
            GrantChoice::OpenPrivacy => t!("modal-grant-open-privacy"),
            GrantChoice::Recheck => t!("modal-grant-recheck"),
            GrantChoice::StartAnyway => t!("modal-grant-start-anyway"),
            GrantChoice::Cancel => t!("common-cancel"),
        })
        .collect();
    let cancel = choices.iter().position(|c| *c == GrantChoice::Cancel).unwrap_or(0);
    let start = choices.len() - 1;
    *pending = Pending::Grant(choices, then_start);
    Effect::Choose(ChoiceSpec {
        purpose: session::GRANT,
        title: t!("modal-grant-title"),
        message,
        choices: labels,
        dismiss: cancel,
        default: Some(start),
        destructive: Vec::new(),
    })
}

/// A saved state's name as its folder under `saves/`: trimmed, and only a
/// plain folder name (not empty, no separators, not hidden). The prompts'
/// validators check the same; this holds whatever a shell sends.
fn plain_state_name(name: &str) -> Option<String> {
    let t = name.trim();
    (!t.is_empty() && !t.contains(['/', '\\', ':']) && !t.starts_with('.')).then(|| t.to_string())
}

fn save_name_rule(spec: &TextSpec, text: &str) -> Result<(), String> {
    // Saving over an existing state is allowed; only the shape is checked.
    let t = text.trim();
    if t.is_empty() {
        return Err(String::new());
    }
    let mut s = spec.clone();
    s.taken.clear();
    s.initial.clear();
    session::state_name_rule(&s, t)
}

#[cfg(test)]
mod tests {
    use super::plain_state_name;

    #[test]
    fn a_state_name_is_a_plain_folder_name() {
        assert_eq!(plain_state_name("  snap1 "), Some("snap1".to_string()), "trimmed");
        // Nothing that names `saves/` itself or anything outside it.
        for bad in ["", "   ", "..", "../x", "a/b", "a\\b", "a:b", ".hidden"] {
            assert_eq!(plain_state_name(bad), None, "{bad:?}");
        }
    }
}
