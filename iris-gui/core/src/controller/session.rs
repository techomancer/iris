//! The running machine: power, saved states, the status-bar lights, CD-ROMs,
//! copy-on-write disks and the safe-stop rules.
//!
//! One session runs at a time (D2), but the session hangs off its own
//! controller, not the app, so several remain possible later. Actions that
//! need the machine's config or the library (power on reads the TOML, a disc
//! insert edits the config while stopped) arrive through
//! [`AppAction::Session`](super::app::AppAction::Session); the app passes them
//! here with what they need.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use iris::config::{MachineConfig, NetMode};

use super::icons::Icon;
use super::{ids, Answer, ChoiceSpec, ConfirmSpec, Effect, Enabled, PickSpec, TextSpec};
use crate::model::handle::{Cmd, DiskMoved, EmulatorHandle, Evt, NetState, Status};
use crate::model::job::Notify;

/// Progress of a disk synchronisation (folding CHD diffs back into their
/// bases, or applying one copy-on-write overlay).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyncProgress {
    /// Disk being synced (0-based) and the total.
    pub disk: usize,
    pub total: usize,
    /// Fraction through the current disk.
    pub fraction: f32,
}

impl SyncProgress {
    /// Overall fraction done, for one progress bar.
    pub fn overall(&self) -> f32 {
        if self.total > 0 {
            ((self.disk as f32 + self.fraction) / self.total as f32).clamp(0.0, 1.0)
        } else {
            self.fraction.clamp(0.0, 1.0)
        }
    }
}

/// What the machine is doing, as the status bar names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunState {
    Stopped,
    /// Power on was asked for; the worker has not reported the machine up.
    Starting,
    Running,
    /// Running but executing nothing: halted at the PROM or idle. Safe to stop.
    Halted,
    /// The guest powered itself off.
    PoweredOff,
}

/// Things that happened in the session that other parts of the app react to
/// (the display releases input capture, the serial console reconnects).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionEvent {
    Started,
    Stopped,
    /// The guest stopped executing (halt, power off): release captured input.
    Halted,
    /// The guest powered itself off.
    PoweredOff,
    /// A disk synchronisation ended.
    SyncDone,
}

/// The user's session intents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionAction {
    PowerOn,
    /// Type `halt` at the IRIX console for a clean shutdown.
    ShutDownGuest,
    /// Ask IRIX to restart (graceful, over the serial console).
    RebootGuest,
    /// Stop the machine; asks first when a disk could be damaged.
    PowerOff,
    /// Power off and on again.
    Restart,
    /// Ask for a name, then save the state.
    AskSaveState,
    SaveState(String),
    /// Restore a saved state; powers on first when stopped.
    RestoreState(String),
    RenameState(String),
    DeleteState(String),
    /// Ask where, then save a PNG of the display.
    Screenshot,
    /// Ask for an image, then load it into the CD-ROM at `key`.
    InsertDisc { key: u8 },
    /// Load `path` into the CD-ROM at `key` (a picker the shell already ran).
    LoadDisc { key: u8, path: String },
    EjectDisc { key: u8 },
    /// Ask IRIX to mount the disc again.
    RemountDisc { key: u8 },
    /// Write a copy-on-write disk's changes into its base image.
    ApplyChanges { key: u8 },
    /// Ask, then discard a copy-on-write disk's changes.
    RollBack { key: u8 },
    /// Stop and fold every CHD's pending changes into its base.
    SynchronizeDisks,
    /// Factory-reset the PROM settings (NVRAM and EEPROM).
    ResetNvram,
    /// Ask now about CHD folders the app cannot write (after a disk is picked).
    CheckFolderGrants,
    /// Grant the app a whole folder (App Store sandbox): disks, CHD merge
    /// temp files and an NFS share under it.
    GrantDiskFolder,
    /// Forget a granted folder.
    RevokeDiskFolder(String),
    Answer(Answer),
}

/// One saved state of a machine (`saves/<name>/` in its folder).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavedState {
    pub name: String,
    /// Seconds since the Unix epoch the state was written.
    pub saved: Option<u64>,
    pub bytes: u64,
    /// The screen when it was saved, when one was kept.
    pub picture: Option<Picture>,
}

/// A picture file and when it was written, so a view reloads it when it changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picture {
    pub path: PathBuf,
    pub modified: u64,
}

impl Picture {
    /// `path` when it is a file.
    pub fn at(path: PathBuf) -> Option<Self> {
        let m = std::fs::metadata(&path).ok().filter(|m| m.is_file())?;
        let modified = m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_millis() as u64;
        Some(Self { path, modified })
    }
}

/// How a status-bar light shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightState {
    /// The machine is stopped: greyed out, still clickable.
    Off,
    Idle,
    Reading,
    Writing,
    /// Network or audio traffic.
    Active,
    /// Something is wrong; the tooltip says what.
    Fault,
}

/// What a light opens when clicked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LightTarget {
    Disk { key: u8 },
    CdRom { key: u8 },
    Network,
    Serial,
    Audio,
    Capture,
    Scale,
}

/// One clickable status-bar icon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceLight {
    /// Widget id (also the accessibility identifier).
    pub id: String,
    pub icon: Icon,
    pub state: LightState,
    /// What it is and what it is doing.
    pub tooltip: String,
    /// A short count beside the icon (copy-on-write dirty sectors).
    pub badge: Option<String>,
    pub target: LightTarget,
}

/// The status bar, left to right.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusBar {
    pub run_state: RunState,
    pub run_label: String,
    /// interpreter / jitv2 / lightning.
    pub executor: String,
    /// "R5000 after restart" when the CPU changed while running.
    pub cpu_pending: Option<String>,
    pub unsaved: bool,
    pub lights: Vec<DeviceLight>,
    pub mips: Option<String>,
    pub hz: Option<String>,
}

/// The emulator worker and what the GUI knows about the machine it runs.
pub struct SessionController {
    pub(crate) emu: EmulatorHandle,
    pub(crate) syncing: Option<SyncProgress>,
    /// Follow mode's steering thread for the running machine, with the host
    /// input it steers (a new machine gets a new one).
    follower: Option<(Arc<iris::hostinput::HostInput>, Arc<crate::model::pointer_follow::PointerFollower>)>,
    starting: bool,
    /// The config the running machine started with (RAM, CPU and network are
    /// fixed until it stops).
    started: Option<MachineConfig>,
    prev_halted: bool,
    prev_stopped: bool,
    events: Vec<SessionEvent>,
    /// Restore this state once the machine is up.
    restore_after_start: Option<String>,
    /// A machine stopped by a power-on preflight answer waits here.
    pub(crate) pending: Pending,
    /// The disc in each CD-ROM (by SCSI id) since this run changed it: the
    /// config keeps the disc the machine started with.
    live_discs: std::collections::HashMap<u8, Option<String>>,
    /// A stop was asked for and the worker has not reported it done.
    stopping: bool,
    /// IRIX powered off: stop the machine at this time (a short pause, so
    /// the last screen stays a moment).
    stop_at: Option<std::time::Instant>,
}

/// What a question the session asked is about.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) enum Pending {
    #[default]
    None,
    /// Missing disk images: detach these keys and start.
    Missing(Vec<u8>),
    /// CHDs in folders the app cannot write; the choices' meaning by index,
    /// and whether answering it goes on to power on.
    Grant(Vec<GrantChoice>, bool),
    /// A folder picker for a grant is up; power on afterwards if true.
    GrantFolder(bool),
    StateName,
    RenameState(String),
    DeleteState(String),
    RollBack(u8),
    InsertDisc(u8),
    Screenshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GrantChoice {
    Grant(String),
    OpenPrivacy,
    Recheck,
    StartAnyway,
    Cancel,
}

pub(crate) const STOP: &str = "session.stop";
pub(crate) const MISSING: &str = "session.missing";
pub(crate) const GRANT: &str = "session.grant";
pub(crate) const GRANT_FOLDER: &str = "session.grant-folder";
pub(crate) const STATE_NAME: &str = "session.state-name";
pub(crate) const RENAME_STATE: &str = "session.rename-state";
pub(crate) const DELETE_STATE: &str = "session.delete-state";
pub(crate) const ROLL_BACK: &str = "session.roll-back";
pub(crate) const INSERT_DISC: &str = "session.insert-disc";
pub(crate) const SCREENSHOT: &str = "session.screenshot";
pub(crate) const RESET_NVRAM: &str = "session.reset-nvram";

impl SessionController {
    pub(crate) fn spawn(notify: Arc<dyn Notify>) -> Self {
        Self {
            emu: EmulatorHandle::spawn(notify),
            syncing: None,
            live_discs: Default::default(),
            stopping: false,
            stop_at: None,
            follower: None,
            starting: false,
            started: None,
            prev_halted: true,
            prev_stopped: false,
            events: Vec::new(),
            restore_after_start: None,
            pending: Pending::None,
        }
    }

    // ---- read accessors -----------------------------------------------------

    /// Whether a machine is up (from the worker's lifecycle events).
    pub fn is_running(&self) -> bool {
        self.emu.is_running()
    }

    pub fn run_state(&self) -> RunState {
        let s = &self.emu.status;
        if !s.running {
            if self.starting {
                RunState::Starting
            } else {
                RunState::Stopped
            }
        } else if s.cpu_stopped {
            RunState::PoweredOff
        } else if s.cpu_halted {
            RunState::Halted
        } else {
            RunState::Running
        }
    }

    /// The worker's latest status sample.
    pub fn status(&self) -> &Status {
        &self.emu.status
    }

    /// The config the running machine started with.
    pub fn started_config(&self) -> Option<&MachineConfig> {
        self.started.as_ref().filter(|_| self.is_running())
    }

    /// A disk synchronisation in progress, for its progress window.
    pub fn syncing(&self) -> Option<SyncProgress> {
        self.syncing
    }

    /// Whether the worker is idle: no machine, no queued command, no machine
    /// left behind by a stop that timed out. Changing the working directory
    /// (selecting, renaming or deleting a machine) waits for this.
    pub fn is_idle(&self) -> bool {
        self.emu.can_change_directory() && self.syncing.is_none()
    }

    /// The live PS/2 controller, for the display's input pump.
    pub fn ps2(&self) -> Option<Arc<iris::dev::ps2::Ps2Controller>> {
        self.emu.ps2.lock().clone()
    }

    /// Follow mode's pointer for the running machine, started on first use.
    pub(crate) fn follower(&mut self) -> Option<Arc<crate::model::pointer_follow::PointerFollower>> {
        let Some(host) = self.emu.host.lock().clone() else {
            self.follower = None;
            return None;
        };
        match &self.follower {
            Some((h, f)) if Arc::ptr_eq(h, &host) => Some(f.clone()),
            _ => {
                let f = Arc::new(crate::model::pointer_follow::PointerFollower::start(host.clone()));
                self.follower = Some((host, f.clone()));
                Some(f)
            }
        }
    }

    /// The frame sinks for head 0 and head 1, for the shell's presenter.
    pub fn frame_sinks(&self) -> (&crate::model::framebuffer::FrameSink, &crate::model::framebuffer::FrameSink) {
        (&self.emu.frame_sink, &self.emu.frame_sink_head1)
    }

    /// Whether `action` can run now, and if not why (a disabled control's tooltip).
    pub fn enabled(&self, action: &SessionAction, cfg: &MachineConfig) -> Enabled {
        let running = self.is_running();
        let need_stopped = || t!("session-needs-stopped");
        let need_running = || t!("session-needs-running");
        let busy = self.syncing.is_some();
        if busy {
            return Enabled::No(t!("session-syncing"));
        }
        match action {
            SessionAction::PowerOn => Enabled::when(!running && !self.starting, need_stopped),
            SessionAction::ShutDownGuest
            | SessionAction::RebootGuest
            | SessionAction::PowerOff
            | SessionAction::Restart
            | SessionAction::AskSaveState
            | SessionAction::SaveState(_)
            | SessionAction::Screenshot
            | SessionAction::SynchronizeDisks => Enabled::when(running, need_running),
            SessionAction::RestoreState(_) => Enabled::Yes,
            SessionAction::RenameState(_) | SessionAction::DeleteState(_) => Enabled::Yes,
            SessionAction::InsertDisc { .. } | SessionAction::LoadDisc { .. } => Enabled::Yes,
            SessionAction::EjectDisc { .. } | SessionAction::RemountDisc { .. } => Enabled::when(running, need_running),
            SessionAction::ApplyChanges { .. } | SessionAction::RollBack { .. } => Enabled::when(!running, need_stopped),
            SessionAction::CheckFolderGrants | SessionAction::GrantDiskFolder | SessionAction::RevokeDiskFolder(_) => {
                Enabled::Yes
            }
            SessionAction::ResetNvram => {
                let p = cfg.machine.profile;
                if !(crate::model::hardware::mc_machine(p) || p == iris::config::MachineProfile::OctaneIp30) {
                    Enabled::No(t!("session-no-nvram-reset"))
                } else {
                    Enabled::when(!running, need_stopped)
                }
            }
            SessionAction::Answer(_) => Enabled::Yes,
        }
    }

    /// The machine's saved states, newest first, from `saves/` in `dir`.
    pub fn saved_states(dir: &Path) -> Vec<SavedState> {
        let Ok(rd) = std::fs::read_dir(dir.join("saves")) else { return Vec::new() };
        let mut out: Vec<SavedState> = rd
            .flatten()
            // Hidden folders (`.cas`, the store snapshots share) are not states.
            .filter(|e| e.path().is_dir() && !e.file_name().to_string_lossy().starts_with('.'))
            .map(|e| {
                let path = e.path();
                let saved = std::fs::metadata(path.join("snapshot.toml"))
                    .or_else(|_| e.metadata())
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs());
                let picture = Picture::at(path.join(crate::model::handle::STATE_SCREEN));
                SavedState { name: e.file_name().to_string_lossy().into_owned(), saved, bytes: dir_size(&path), picture }
            })
            .collect();
        out.sort_by(|a, b| b.saved.cmp(&a.saved).then(a.name.cmp(&b.name)));
        out
    }

    /// The executor this build runs guests on, for the status-bar chip.
    pub fn executor() -> &'static str {
        let jit_off = std::env::var_os("IRIS_NO_JIT").is_some();
        if iris::build_features::JITV2 && !jit_off {
            "jitv2"
        } else if iris::build_features::LIGHTNING {
            "lightning"
        } else {
            "interpreter"
        }
    }

    /// The status bar for the selected machine, whose current config is `cfg`.
    pub fn status_bar(&self, cfg: &MachineConfig, unsaved: bool) -> StatusBar {
        let state = self.run_state();
        let s = &self.emu.status;
        let live = matches!(state, RunState::Running | RunState::Halted);
        let run_label = match state {
            _ if self.syncing.is_some() => t!("status-syncing"),
            RunState::Running | RunState::Halted | RunState::PoweredOff if self.stopping => t!("status-stopping"),
            RunState::Stopped => t!("status-stopped"),
            RunState::Starting => t!("status-starting"),
            RunState::Running => t!("status-running"),
            RunState::Halted => t!("status-halted"),
            RunState::PoweredOff => t!("status-powered-off"),
        };
        let started = self.started_config();
        let cpu_pending = started
            .filter(|st| st.cpu() != cfg.cpu())
            .map(|_| t!("status-cpu-pending", cpu = super::profile::cpu_label(cfg.cpu())));
        let shown = started.unwrap_or(cfg);
        let mut lights = Vec::new();
        for (key, dev) in crate::model::hardware::scsi_disks(shown) {
            if dev.is_daynaport() {
                continue;
            }
            let cdrom = dev.is_cdrom();
            if !cdrom && dev.path.trim().is_empty() {
                continue;
            }
            let (controller, id) = crate::model::hardware::placement(key);
            let moved = s
                .disks
                .get(controller as usize)
                .and_then(|b| b.get(id as usize))
                .copied()
                .unwrap_or(DiskMoved::default());
            let light = if !live {
                LightState::Off
            } else if moved.wrote {
                LightState::Writing
            } else if moved.read {
                LightState::Reading
            } else {
                LightState::Idle
            };
            let name = crate::model::hardware::scsi_name(key);
            let file = Path::new(&dev.path).file_name().map(|f| f.to_string_lossy().into_owned());
            let tooltip = if cdrom {
                match &file {
                    Some(f) => t!("status-cdrom-tip", id = &name, disc = f.as_str()),
                    None => t!("status-cdrom-empty-tip", id = &name),
                }
            } else {
                t!("status-disk-tip", id = &name, file = file.unwrap_or_default())
            };
            lights.push(DeviceLight {
                id: ids::status::disk(controller as usize, id),
                icon: if cdrom { Icon::CdRom } else { Icon::Disk },
                state: light,
                tooltip,
                badge: None,
                target: if cdrom { LightTarget::CdRom { key } } else { LightTarget::Disk { key } },
            });
        }
        // Copy-on-write: one count for the machine (the worker sums them).
        if live && s.dirty_cow > 0 {
            if let Some(first) = lights.iter_mut().find(|l| matches!(l.target, LightTarget::Disk { .. })) {
                first.badge = Some(s.dirty_cow.to_string());
                first.tooltip = format!("{}\n{}", first.tooltip, t!("status-dirty-sectors", count = s.dirty_cow));
            }
        }
        if live && s.chd_sync_pending {
            if let Some(first) = lights.iter_mut().find(|l| matches!(l.target, LightTarget::Disk { .. })) {
                first.tooltip = format!("{}\n{}", first.tooltip, t!("status-chd-sync-pending"));
            }
        }
        // Network: the backend is the one the machine started with.
        let mode = match shown.network.mode {
            NetMode::Pcap if !iris::build_features::PCAP => NetMode::Nat,
            m => m,
        };
        if mode != NetMode::Off {
            let net = self.emu.net_state();
            let (label, state, tip) = match (live, mode) {
                (false, _) => (t!("status-net"), LightState::Off, t!("status-net-off-tip")),
                (true, NetMode::Pcap) => {
                    let iface = shown
                        .network
                        .pcap_interface
                        .clone()
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| t!("common-auto"));
                    match net {
                        NetState::Active => (t!("status-net-pcap"), LightState::Active, t!("status-net-pcap-active-tip", iface = &iface)),
                        NetState::Idle => (t!("status-net-pcap"), LightState::Fault, t!("status-net-pcap-idle-tip", iface = &iface)),
                        NetState::Off => (t!("status-net-pcap"), LightState::Idle, t!("status-net-pcap-off-tip")),
                    }
                }
                (true, NetMode::Stream) => match net {
                    NetState::Active => (t!("status-net-stream"), LightState::Active, t!("status-net-stream-active-tip")),
                    _ => (t!("status-net-stream"), LightState::Idle, t!("status-net-stream-idle-tip")),
                },
                (true, _) => match net {
                    NetState::Active => (t!("status-net-nat"), LightState::Active, t!("status-net-nat-active-tip")),
                    NetState::Idle => (t!("status-net-nat"), LightState::Fault, t!("status-net-nat-idle-tip")),
                    NetState::Off => (t!("status-net-nat"), LightState::Idle, t!("status-net-nat-off-tip")),
                },
            };
            let ip = s.net_guest_ip.map(|ip| format!("\n{}", t!("status-net-guest-ip", ip = ip.to_string()))).unwrap_or_default();
            lights.push(DeviceLight {
                id: ids::status::NETWORK.into(),
                icon: Icon::Network,
                state,
                tooltip: format!("{label}: {tip}{ip}"),
                badge: None,
                target: LightTarget::Network,
            });
        }
        if s.has_audio || !live {
            lights.push(DeviceLight {
                id: ids::status::AUDIO.into(),
                icon: Icon::Audio,
                state: if !live {
                    LightState::Off
                } else if s.audio_moving {
                    LightState::Active
                } else {
                    LightState::Idle
                },
                tooltip: if s.audio_moving { t!("status-audio-playing-tip") } else { t!("status-audio-tip") },
                badge: None,
                target: LightTarget::Audio,
            });
        }
        let rates = matches!(state, RunState::Running);
        StatusBar {
            run_state: state,
            run_label,
            executor: Self::executor().to_string(),
            cpu_pending,
            unsaved,
            lights,
            mips: rates.then(|| t!("status-mips", mips = format!("{:.0}", s.mips))),
            hz: rates.then(|| t!("status-kernel-hz", hz = format!("{:.0}", s.kernel_hz))),
        }
    }

    /// Session events since the last call (the app routes them on).
    pub(crate) fn take_events(&mut self) -> Vec<SessionEvent> {
        std::mem::take(&mut self.events)
    }

    /// The disc in the CD-ROM at SCSI id `id` while running: `Some(None)` for
    /// an empty tray, `None` if this run has not changed it.
    pub(crate) fn live_disc(&self, id: u8) -> Option<Option<&str>> {
        if !self.is_running() {
            return None;
        }
        self.live_discs.get(&id).map(|p| p.as_deref())
    }

    // ---- worker events --------------------------------------------------------

    /// Drain the worker's events into toasts and session events.
    pub(crate) fn drain(&mut self, fx: &mut Vec<Effect>) {
        for evt in self.emu.drain_events() {
            match evt {
                Evt::Started => {
                    self.starting = false;
                    self.live_discs.clear();
                    self.prev_halted = true;
                    // A CPU not started yet (debug builds wait for the
                    // monitor's `start`) is not a guest that switched off:
                    // only running -> stopped counts.
                    self.prev_stopped = true;
                    self.events.push(SessionEvent::Started);
                    fx.push(Effect::Toast(t!("toast-emulator-started")));
                    if let Some(name) = self.restore_after_start.take() {
                        self.emu.send(Cmd::RestoreState(name));
                    }
                }
                Evt::Stopped => {
                    self.emu.status.clear_worker_fields();
                    self.starting = false;
                    self.stopping = false;
                    self.live_discs.clear();
                    self.restore_after_start = None;
                    self.events.push(SessionEvent::Stopped);
                    fx.push(Effect::Toast(t!("toast-emulator-stopped")));
                }
                Evt::PowerOff => fx.push(Effect::Toast(t!("toast-guest-powered-off"))),
                Evt::StateSaved(n) => fx.push(Effect::Toast(t!("toast-state-saved", name = n))),
                Evt::StateRestored(n) => fx.push(Effect::Toast(t!("toast-state-restored", name = n))),
                Evt::Screenshot(p) => fx.push(Effect::Toast(t!("toast-screenshot-saved", path = p.display()))),
                // Only a failed start ends "starting": other errors on the way
                // up (a VNC port in use) leave the start going.
                Evt::StartFailed(e) => {
                    self.starting = false;
                    self.restore_after_start = None;
                    fx.push(Effect::Toast(t!("toast-error", error = e)));
                }
                Evt::Error(e) => fx.push(Effect::Toast(t!("toast-error", error = e))),
                Evt::Disc { id, path, message } => {
                    self.live_discs.insert(id, path);
                    fx.push(Effect::Toast(message));
                }
                Evt::Notice(message) => fx.push(Effect::Toast(message)),
                Evt::Status(_) => {}
                Evt::SyncProgress { disk, total, fraction } => {
                    self.syncing = Some(SyncProgress { disk, total, fraction });
                }
                Evt::CowDone { committed } => {
                    fx.push(Effect::Toast(if committed { t!("toast-cow-committed") } else { t!("toast-cow-done") }));
                }
                Evt::SyncDone(n) => {
                    self.syncing = None;
                    if n > 0 {
                        fx.push(Effect::Toast(t!("toast-disks-synced", count = n)));
                    }
                    self.events.push(SessionEvent::SyncDone);
                }
            }
        }
        let s = &self.emu.status;
        let running = s.running;
        let halted = running && s.cpu_halted;
        if halted && !self.prev_halted {
            self.events.push(SessionEvent::Halted);
        }
        self.prev_halted = halted || !running;
        let stopped = running && s.cpu_stopped;
        if stopped && !self.prev_stopped {
            self.events.push(SessionEvent::PoweredOff);
            // The guest shut itself down: fold CHD diffs now, so a later quit
            // (however it happens; Cmd+Q bypasses the window close) has
            // nothing left to do. clanker/gui/cmd-q-bypasses-close-intercept-fold-on-poweroff.md
            if self.emu.has_pending_chd_sync() && self.syncing.is_none() {
                fx.push(Effect::Toast(t!("toast-guest-off-syncing")));
                self.syncing = Some(SyncProgress { disk: 0, total: 0, fraction: 0.0 });
                self.emu.send(Cmd::SyncDisks);
            } else if self.syncing.is_none() {
                // Nothing to fold: stop it after a moment, as the hardware
                // would have switched off.
                self.stop_at = Some(std::time::Instant::now() + std::time::Duration::from_secs(2));
            }
        }
        self.prev_stopped = stopped;
        if let Some(at) = self.stop_at {
            if !stopped || self.stopping || self.syncing.is_some() {
                self.stop_at = None;
            } else if std::time::Instant::now() >= at {
                self.stop_at = None;
                self.stop_now(fx);
            }
        }
    }

    // ---- actions that need only the session -----------------------------------

    /// Start a machine on `cfg` (already read, checked and normalised).
    pub(crate) fn start(&mut self, cfg: MachineConfig) {
        self.starting = true;
        self.prev_halted = true;
        self.prev_stopped = true;
        self.started = Some(cfg.clone());
        self.emu.send(Cmd::Start(Box::new(cfg)));
    }

    pub(crate) fn stop(&mut self, cfg: &MachineConfig, fx: &mut Vec<Effect>) {
        let reasons = crate::model::safe_stop::evaluate(&self.emu.status, cfg);
        if !reasons.is_empty() {
            let mut message = t!("modal-stop-body");
            for line in crate::model::safe_stop::reason_lines(&reasons) {
                message.push('\n');
                message.push_str(&t!("modal-list-item", item = line));
            }
            fx.push(Effect::Choose(ChoiceSpec {
                purpose: STOP,
                title: t!("modal-stop-title"),
                message,
                choices: vec![t!("common-cancel"), t!("modal-stop-halt"), t!("modal-stop-force")],
                dismiss: 0,
                default: Some(1),
                destructive: vec![2],
            }));
            return;
        }
        self.stop_now(fx);
    }

    /// Stop without asking; folds pending CHD diffs first, and says which.
    pub(crate) fn stop_now(&mut self, fx: &mut Vec<Effect>) {
        self.stopping = true;
        let syncing = self.emu.has_pending_chd_sync();
        fx.push(Effect::Toast(if syncing { t!("toast-stopping-sync") } else { t!("toast-stopping") }));
        if syncing {
            self.syncing = Some(SyncProgress { disk: 0, total: 0, fraction: 0.0 });
            self.emu.send(Cmd::SyncDisks);
        } else {
            self.emu.send(Cmd::Stop);
        }
    }

    /// The running machine's disk at SCSI `(bus, id)`, read through the
    /// machine; `None` when it is not running or cannot be read that way.
    pub(crate) fn live_disk(&self, bus: usize, id: u8) -> Option<crate::model::handle::LiveDisk> {
        if !self.is_running() {
            return None;
        }
        self.emu.live_disk(bus, id)
    }

    /// The time until a machine IRIX powered off is stopped.
    pub(crate) fn stop_due_in(&self) -> Option<std::time::Duration> {
        self.stop_at.map(|t| t.saturating_duration_since(std::time::Instant::now()))
    }

    /// A stop was asked for and has not finished.
    pub(crate) fn stopping(&self) -> bool {
        self.stopping
    }

    /// A stop was sent another way (a forced Power Off).
    pub(crate) fn mark_stopping(&mut self) {
        self.stopping = true;
    }

    pub(crate) fn send(&self, cmd: Cmd) {
        self.emu.send(cmd);
    }

    /// Power On was sent and the machine is not up yet.
    pub(crate) fn is_starting(&self) -> bool {
        self.starting
    }

    pub(crate) fn restore_when_started(&mut self, name: String) {
        self.restore_after_start = Some(name);
    }

    /// The worker handle, for the parts of the egui shell not yet on a
    /// controller. Goes away once every slice has moved.
    #[doc(hidden)]
    pub fn handle(&self) -> &EmulatorHandle {
        &self.emu
    }

    #[doc(hidden)]
    pub fn handle_mut(&mut self) -> &mut EmulatorHandle {
        &mut self.emu
    }

    #[doc(hidden)]
    pub fn set_syncing(&mut self, s: Option<SyncProgress>) {
        self.syncing = s;
    }
}

fn dir_size(dir: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    rd.flatten()
        .map(|e| match e.metadata() {
            Ok(m) if m.is_dir() => dir_size(&e.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        })
        .sum()
}

/// A saved-state name: a plain folder name.
pub(crate) fn state_name_rule(spec: &TextSpec, text: &str) -> Result<(), String> {
    let t = text.trim();
    if t.is_empty() || t == spec.initial {
        return Err(String::new());
    }
    if t.contains(['/', '\\', ':']) || t.starts_with('.') {
        return Err(t!("session-state-name-invalid"));
    }
    if spec.taken.iter().any(|n| n == t) {
        return Err(t!("session-state-name-taken"));
    }
    Ok(())
}

/// The confirmation before discarding a copy-on-write disk's changes.
pub(crate) fn roll_back_confirm(id: &str, file: &str) -> Effect {
    Effect::Confirm(ConfirmSpec {
        purpose: ROLL_BACK,
        title: t!("modal-discard-title"),
        message: format!("{}\n{}", t!("modal-discard-body"), t!("menu-cow-disk", id = id, name = file)),
        confirm_label: t!("modal-discard-confirm"),
        destructive: true,
    })
}

/// A save dialog for a screenshot, opening in the machine folder.
pub(crate) fn screenshot_pick(dir: Option<PathBuf>) -> Effect {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Effect::PickFile(
        PickSpec::save(SCREENSHOT, t!("session-screenshot-title"), format!("screenshot_{stamp}.png"))
            .filter("PNG", &["png"])
            .in_dir(dir),
    )
}

/// A file dialog for a disc image.
pub(crate) fn disc_pick(dir: Option<PathBuf>) -> Effect {
    Effect::PickFile(
        PickSpec::open(INSERT_DISC, t!("session-insert-disc-title"))
            .filter(&t!("session-disc-images"), &["iso", "chd", "cue", "img", "raw", "toast", "cdr"])
            .in_dir(dir),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saved_states_are_listed_newest_first() {
        let root = crate::model::machines::testing::TempRoot::new("states");
        for (name, age) in [("old", 100u64), ("new", 10)] {
            let d = root.0.join("saves").join(name);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("snapshot.toml"), "x").unwrap();
            let t = std::time::SystemTime::now() - std::time::Duration::from_secs(age);
            std::fs::File::options().write(true).open(d.join("snapshot.toml")).unwrap().set_modified(t).unwrap();
        }
        let states = SessionController::saved_states(&root.0);
        assert_eq!(states.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["new", "old"]);
        assert!(states[0].bytes >= 1);
    }

    #[test]
    fn state_names_refuse_paths_and_duplicates() {
        let spec = TextSpec {
            purpose: STATE_NAME,
            title: String::new(),
            prompt: String::new(),
            initial: String::new(),
            ok_label: String::new(),
            taken: vec!["snap1".into()],
            error: None,
            validator: state_name_rule,
        };
        assert!(spec.validate("snap2").is_ok());
        assert!(spec.validate("snap1").is_err());
        assert!(spec.validate("../x").is_err());
        assert_eq!(spec.validate(" "), Err(String::new()));
    }

    #[test]
    fn a_stopped_machine_shows_its_lights_greyed() {
        let s = SessionController::spawn(Arc::new(crate::model::job::NoWake));
        let mut cfg = MachineConfig::default();
        cfg.disk_mut(1).unwrap().path = "disks/root.img".into();
        let bar = s.status_bar(&cfg, false);
        assert_eq!(bar.run_state, RunState::Stopped);
        assert!(bar.mips.is_none());
        let disk = bar.lights.iter().find(|l| l.target == LightTarget::Disk { key: 1 }).expect("disk light");
        assert_eq!(disk.state, LightState::Off);
        assert!(bar.lights.iter().all(|l| l.state == LightState::Off));
        assert!(bar.lights.iter().any(|l| l.target == LightTarget::Network));
    }
}
