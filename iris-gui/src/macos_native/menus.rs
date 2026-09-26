//! Menu model and dispatcher for the native macOS menu bar.
//!
//! The menus hold what the classic sidebar's File, Machine, Memory, SCSI,
//! View and Help menus hold. A feature added to one should normally be added to
//! the other too.

use crate::settings;
use crate::App;
use eframe::egui;
use iris::config::CpuModel;

/// Everything the menus can ask the app to do.
///
/// Deliberately data-only: a native `NSMenuItem` can't hold a closure, so a
/// click resolves to one of these and is applied later on the UI thread, after
/// the menu has closed. That also means a file picker opened by a menu item
/// (`AttachHdd` and friends) runs *after* the menu is gone, rather than
/// underneath it.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    // --- File ---
    NewMachine,
    SwitchMachine(String),
    RenameMachine,
    DeleteMachine,
    ImportToml,
    ExportToml,
    PrepareForPremiere,
    GrantDiskFolder,
    RevealFolder(String),
    RevokeFolder(String),
    Quit,

    // --- Machine ---
    Start,
    Stop,
    Reset,
    ResetNvram,
    SetCpu(CpuModel),
    SaveState(String),
    RestoreState(String),
    Screenshot,
    SerialConsole,
    ToggleCapture,

    // --- Memory ---
    SetRam(u32),
    SetBank(usize, u32),

    // --- SCSI ---
    Scsi(ScsiCmd),
    CowCommit { base: String, chd: bool },
    CowDiscard { id: u8, base: String, chd: bool },

    // --- View ---
    ToggleFullscreen,
    SetVmScale(f32),
    SetUiScale(f32),
    ShowConfig,

    // --- Help ---
    NetCheck,
    CameraTest,
    HelpInfo,
    NfsHelp,
    Ultra64Help,
    License,
    Privacy,
    About,
    OpenUrl(&'static str),
}

/// Save-state slots offered by the Machine menu.
const SAVE_SLOTS: [&str; 4] = ["snap1", "snap2", "snap3", "snap4"];

/// A SCSI bus operation. The `Pick*` variants open a file dialog when applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScsiCmd {
    PickHdd(u8),
    AttachEmptyCdrom(u8),
    PickCdromWithDisc(u8),
    PickDisc(u8),
    Eject(u8),
    Remount(u8),
    Detach(u8),
    CreateBlank(u8),
    ToggleOverlay(u8),
}

/// A keyboard equivalent. Command on macOS, Ctrl elsewhere — the modifier is
/// implied, since a menu accelerator that isn't the platform's own command key
/// would collide with keystrokes meant for the guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Accel {
    pub key: char,
    pub shift: bool,
}

/// One row of a menu.
#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Action {
        label: String,
        action: Action,
        enabled: bool,
        checked: bool,
        accel: Option<Accel>,
    },
    Sub {
        label: String,
        items: Vec<Item>,
    },
    /// Non-clickable text: a section heading, or a line of explanation.
    Info(String),
    Separator,
}

/// A top-level menu.
#[derive(Clone, Debug, PartialEq)]
pub struct Menu {
    pub title: String,
    pub items: Vec<Item>,
}

/// A clickable item. Chain `.off()` / `.checked()` / `.accel()` to refine it.
pub fn act(label: impl Into<String>, action: Action) -> Item {
    Item::Action {
        label: label.into(),
        action,
        enabled: true,
        checked: false,
        accel: None,
    }
}

impl Item {
    /// Enable only when `yes`.
    pub fn enabled_if(mut self, yes: bool) -> Self {
        if let Item::Action { enabled, .. } = &mut self {
            *enabled = yes;
        }
        self
    }
    pub fn checked_if(mut self, yes: bool) -> Self {
        if let Item::Action { checked, .. } = &mut self {
            *checked = yes;
        }
        self
    }
    pub fn accel(mut self, key: char) -> Self {
        if let Item::Action { accel, .. } = &mut self {
            *accel = Some(Accel { key, shift: false });
        }
        self
    }
    pub fn accel_shift(mut self, key: char) -> Self {
        if let Item::Action { accel, .. } = &mut self {
            *accel = Some(Accel { key, shift: true });
        }
        self
    }
}

/// "1×", "1.25×" — trailing zeros dropped so the common whole steps read clean.
pub fn scale_label(s: f32) -> String {
    if (s - s.round()).abs() < 0.005 {
        format!("{:.0}\u{00d7}", s.round())
    } else {
        format!("{s}\u{00d7}")
    }
}

/// Steps of a scale slider, as menu-friendly discrete choices.
fn scale_steps(min: f32, max: f32, step: f32) -> Vec<f32> {
    let mut out = Vec::new();
    let n = ((max - min) / step).round() as i32;
    for i in 0..=n {
        out.push(min + step * i as f32);
    }
    out
}

fn same_scale(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.01
}

impl App {
    /// Build the whole menu tree from current state.
    ///
    /// Cheap enough to call a few times a second (see the rebuild throttle in
    /// `refresh_menus`), but *not* every frame: naming a SCSI disk stats its image
    /// file to report the size.
    pub fn build_menus(&self) -> Vec<Menu> {
        vec![
            self.file_menu(),
            self.machine_menu(),
            self.memory_menu(),
            self.scsi_menu(),
            self.view_menu(),
            self.help_menu(),
        ]
    }

    fn file_menu(&self) -> Menu {
        let mut items = vec![act("New machine\u{2026}", Action::NewMachine).accel('n')];

        let mut machines: Vec<Item> = Vec::new();
        if self.prefs.machines.is_empty() {
            machines.push(Item::Info("(no saved machines yet)".into()));
        }
        for name in self.prefs.machines.keys() {
            let active = self.prefs.active_machine.as_deref() == Some(name.as_str());
            machines.push(
                act(name.clone(), Action::SwitchMachine(name.clone())).checked_if(active),
            );
        }
        items.push(Item::Sub { label: "Switch to machine".into(), items: machines });

        let has_active = self.prefs.active_machine.is_some();
        items.push(act("Rename current\u{2026}", Action::RenameMachine).enabled_if(has_active));
        items.push(act("Delete current machine", Action::DeleteMachine).enabled_if(has_active));

        items.push(Item::Separator);
        items.push(act("Configuration\u{2026}", Action::ShowConfig).accel(','));

        // iris.toml import/export is a source-build affordance for users who
        // also run the standalone `iris` CLI; the GUI's own gui.json machine
        // store is the system of record. Hidden in pre-compiled / App Store
        // builds (the `bundled` feature). See iris-gui Cargo.toml.
        if !cfg!(feature = "bundled") {
            items.push(act("Import iris.toml\u{2026}", Action::ImportToml));
            items.push(act("Export current to iris.toml\u{2026}", Action::ExportToml));
            items.push(act("Prepare for premiere\u{2026}", Action::PrepareForPremiere));
        }

        // App Store sandbox: grant a whole folder (recursive) so the disk-sync
        // fold — which writes a temp beside the base and renames over it —
        // works, and so a disk image / NFS shared subfolder under it is covered
        // by one grant. Hidden elsewhere.
        if cfg!(feature = "appstore") {
            items.push(Item::Separator);
            items.push(act("Grant a disk folder\u{2026}", Action::GrantDiskFolder));
            if self.prefs.disk_folders.is_empty() {
                items.push(Item::Info("(no folders granted yet)".into()));
            }
            for f in &self.prefs.disk_folders {
                // Live access state: a grant can lapse mid-session, and the
                // only fix is to re-grant, so say which it is.
                let live = crate::folder_accessible(f);
                let state = if live { "granted" } else { "no access \u{2014} re-grant" };
                items.push(Item::Sub {
                    label: format!("{f}  ({state})"),
                    items: vec![
                        act("Reveal in file manager", Action::RevealFolder(f.clone())),
                        act("Revoke", Action::RevokeFolder(f.clone())),
                    ],
                });
            }
        }

        items.push(Item::Separator);
        items.push(act("Quit", Action::Quit).accel('q'));
        Menu { title: "File".into(), items }
    }

    fn machine_menu(&self) -> Menu {
        let running = self.emu.is_running();
        let mut items = vec![
            act("Start", Action::Start).enabled_if(!running).accel('r'),
            act("Stop", Action::Stop).enabled_if(running).accel_shift('r'),
            act("Reset", Action::Reset).enabled_if(running),
            act("Reset NVRAM (fresh PRAM)", Action::ResetNvram).enabled_if(!running),
            Item::Separator,
        ];

        // Chosen at Machine::new, so an edit while running is pending, not live.
        let mut cpus: Vec<Item> = CpuModel::ALL
            .iter()
            .map(|&c| {
                act(c.label(), Action::SetCpu(c))
                    .checked_if(self.cfg.machine.cpu == c)
                    .enabled_if(!running)
            })
            .collect();
        if running {
            cpus.push(Item::Separator);
            match self.started_cpu {
                Some(started) if started != self.cfg.machine.cpu => cpus.push(Item::Info(
                    format!("Running: {} \u{2014} Stop to apply edits", started.label()),
                )),
                Some(started) => cpus.push(Item::Info(format!("Running: {}", started.label()))),
                None => {}
            }
        }
        items.push(Item::Sub {
            label: format!("Processor: {}", self.cfg.machine.cpu.label()),
            items: cpus,
        });

        items.push(Item::Separator);
        // Named slots rather than a typed-in name: a menu has nowhere to type,
        // and a fixed set of slots is what a save state is used as anyway.
        let slots = |kind: fn(String) -> Action| -> Vec<Item> {
            SAVE_SLOTS
                .iter()
                .map(|name| act(*name, kind(name.to_string())).enabled_if(running))
                .collect()
        };
        items.push(Item::Sub { label: "Save state".into(), items: slots(Action::SaveState) });
        items.push(Item::Sub { label: "Restore state".into(), items: slots(Action::RestoreState) });
        items.push(Item::Separator);
        items.push(act("Screenshot\u{2026}", Action::Screenshot).enabled_if(running));
        items.push(act("Serial console\u{2026}", Action::SerialConsole).enabled_if(running));
        items.push(Item::Separator);
        items.push(
            act(
                if self.input_state.captured {
                    format!("Release mouse & keyboard ({})", crate::input::RELEASE_HINT)
                } else {
                    "Capture mouse & keyboard".into()
                },
                Action::ToggleCapture,
            )
            .enabled_if(running)
            .accel('k'),
        );
        Menu { title: "Machine".into(), items }
    }

    fn memory_menu(&self) -> Menu {
        let running = self.emu.is_running();
        let mut items = vec![Item::Info(format!(
            "Config: {}",
            crate::ram_summary(&self.cfg.banks)
        ))];
        if running {
            if let Some(started) = self.started_banks {
                if started != self.cfg.banks {
                    items.push(Item::Info(format!(
                        "Running: {} \u{2014} Stop to apply edits",
                        crate::ram_summary(&started)
                    )));
                } else {
                    items.push(Item::Info(format!("Running: {}", crate::ram_summary(&started))));
                }
            }
            items.push(Item::Info("RAM changes apply after Stop \u{2192} Start".into()));
        } else {
            items.push(Item::Info("Applied at next Start".into()));
        }
        items.push(Item::Separator);
        items.push(Item::Info("Quick presets (auto-distributed)".into()));
        for &p in crate::RAM_PRESETS {
            items.push(
                act(format!("{p} MB"), Action::SetRam(p))
                    .enabled_if(!running)
                    .checked_if(self.cfg.banks == crate::distribute_ram(p)),
            );
        }
        items.push(Item::Separator);
        for i in 0..4 {
            let banks: Vec<Item> = iris::config::VALID_BANK_SIZES
                .iter()
                .map(|&sz| {
                    act(format!("{sz} MB"), Action::SetBank(i, sz))
                        .enabled_if(!running)
                        .checked_if(self.cfg.banks[i] == sz)
                })
                .collect();
            items.push(Item::Sub {
                label: format!("Bank {i}: {} MB", self.cfg.banks[i]),
                items: banks,
            });
        }
        Menu { title: "Memory".into(), items }
    }

    fn scsi_menu(&self) -> Menu {
        let mut items = Vec::new();
        for id in 1u8..=7 {
            let dev = self.cfg.scsi.get(&id);
            let label = crate::scsi_menu::render_label(id, dev);
            let sub = match dev {
                None => vec![
                    act("Attach HDD\u{2026}", Action::Scsi(ScsiCmd::PickHdd(id))),
                    // Attaching a CD-ROM gives an empty drive by default; the
                    // user loads media afterwards via "Insert disc…". Mirrors
                    // real hardware and avoids an upfront file prompt.
                    act("Attach CD-ROM drive (empty)", Action::Scsi(ScsiCmd::AttachEmptyCdrom(id))),
                    act("Attach CD-ROM with disc\u{2026}", Action::Scsi(ScsiCmd::PickCdromWithDisc(id))),
                    act("Create blank HDD image\u{2026}", Action::Scsi(ScsiCmd::CreateBlank(id))),
                ],
                Some(d) if d.is_daynaport() => vec![
                    Item::Info("DaynaPort SCSI/Link (Ethernet).".into()),
                    Item::Info("Configure its MAC and subnet on the Disks tab.".into()),
                    Item::Separator,
                    act("Detach DaynaPort", Action::Scsi(ScsiCmd::Detach(id))),
                ],
                Some(d) if d.is_cdrom() => {
                    let has_media =
                        !d.path.is_empty() && std::path::Path::new(&d.path).exists();
                    let mut v = Vec::new();
                    if has_media {
                        v.push(act("Eject (tray empty)", Action::Scsi(ScsiCmd::Eject(id))));
                    }
                    v.push(act(
                        if has_media { "Swap disc\u{2026}" } else { "Insert disc\u{2026}" },
                        Action::Scsi(ScsiCmd::PickDisc(id)),
                    ));
                    if has_media {
                        v.push(act("Mount /CDROM in IRIX\u{2026}", Action::Scsi(ScsiCmd::Remount(id))));
                    }
                    v.push(Item::Separator);
                    v.push(act("Detach CD-ROM drive", Action::Scsi(ScsiCmd::Detach(id))));
                    v
                }
                Some(d) => vec![
                    act(
                        if d.overlay {
                            "Disable COW overlay"
                        } else {
                            "Enable COW overlay (writes \u{2192} .overlay)"
                        },
                        Action::Scsi(ScsiCmd::ToggleOverlay(id)),
                    ),
                    act("Replace image\u{2026}", Action::Scsi(ScsiCmd::PickHdd(id))),
                    Item::Separator,
                    act("Detach hard drive", Action::Scsi(ScsiCmd::Detach(id))),
                ],
            };
            items.push(Item::Sub { label, items: sub });
        }
        items.push(Item::Separator);
        items.push(Item::Info(
            "CD-ROM: prefer SCSI #4. Insert/Swap hot-loads media and remounts".into(),
        ));
        items.push(Item::Info(
            "/CDROM (console shell must be active). New drives need Stop\u{2192}Start.".into(),
        ));

        // Copy-on-write overlays with something in them: commit or roll back.
        let cow = self.cow_entries();
        if !cow.is_empty() {
            items.push(Item::Separator);
            items.push(Item::Info("Copy-on-write changes".into()));
            if self.emu.is_running() {
                items.push(Item::Info("Stop the machine to commit or roll back.".into()));
            } else {
                for (id, base, is_chd) in cow {
                    let name = std::path::Path::new(&base)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(&base)
                        .to_string();
                    items.push(Item::Sub {
                        label: format!("SCSI {id}: {name}"),
                        items: vec![
                            act(
                                "Commit changes to disk",
                                Action::CowCommit { base: base.clone(), chd: is_chd },
                            ),
                            act(
                                "Discard changes (roll back)",
                                Action::CowDiscard { id, base: base.clone(), chd: is_chd },
                            ),
                        ],
                    });
                }
            }
        }
        Menu { title: "SCSI".into(), items }
    }

    fn view_menu(&self) -> Menu {
        let mut items = vec![
            act(
                if self.fullscreen { "Exit fullscreen" } else { "Fullscreen" },
                Action::ToggleFullscreen,
            )
            .accel('f'),
            Item::Separator,
        ];

        // The emulated display is drawn at a whole number of device pixels per
        // emulated pixel wherever it can be, so these are exact: 1× is the
        // guest's own resolution, one emulated pixel per logical point.
        let vm: Vec<Item> = scale_steps(
            settings::VM_SCALE_MIN,
            settings::VM_SCALE_MAX,
            settings::VM_SCALE_STEP as f32,
        )
        .into_iter()
        .map(|s| {
            act(scale_label(s), Action::SetVmScale(s))
                .checked_if(same_scale(self.prefs.vm_scale, s))
        })
        .collect();
        items.push(Item::Sub {
            label: format!("Emulator scale: {}", scale_label(self.prefs.vm_scale)),
            items: vm,
        });

        let ui_steps: Vec<Item> = scale_steps(settings::UI_SCALE_MIN, settings::UI_SCALE_MAX, 0.25)
            .into_iter()
            .map(|s| {
                act(scale_label(s), Action::SetUiScale(s))
                    .checked_if(same_scale(self.prefs.ui_scale, s))
            })
            .collect();
        items.push(Item::Sub {
            label: format!("Menu & dialog scale: {}", scale_label(self.prefs.ui_scale)),
            items: ui_steps,
        });

        Menu { title: "View".into(), items }
    }

    fn help_menu(&self) -> Menu {
        let running = self.emu.is_running();
        let mut items = vec![
            Item::Info("Diagnostics".into()),
            act("Test camera\u{2026}", Action::CameraTest).enabled_if(running),
            act("Serial console\u{2026}", Action::SerialConsole).enabled_if(running),
            act("Check networking\u{2026}", Action::NetCheck).enabled_if(running),
            Item::Separator,
            act("How camera & networking work\u{2026}", Action::HelpInfo),
            act("Mount the shared folder in IRIX\u{2026}", Action::NfsHelp),
        ];
        // N64 dev board getting-started guide. Only in builds that carry the
        // board (source builds with --features ultra64), and never in App Store
        // builds, where it can't run anyway.
        if cfg!(all(feature = "ultra64", not(feature = "appstore"))) {
            items.push(act("N64 development board (Ultra64)\u{2026}", Action::Ultra64Help));
        }
        items.push(Item::Separator);
        items.push(act("Licenses\u{2026}", Action::License));
        items.push(act("Privacy policy\u{2026}", Action::Privacy));
        items.push(Item::Separator);
        items.push(act(
            "IRIS on GitHub (upstream)",
            Action::OpenUrl("https://github.com/techomancer/iris"),
        ));
        items.push(act(
            "This fork on GitHub",
            Action::OpenUrl("https://github.com/danifunker/iris"),
        ));
        items.push(Item::Separator);
        items.push(act("About IRIS", Action::About));
        Menu { title: "Help".into(), items }
    }
}

impl App {
    /// Disks that currently have a copy-on-write overlay on disk:
    /// `(scsi id, base image path, is a CHD)`.
    pub fn cow_entries(&self) -> Vec<(u8, String, bool)> {
        let mut entries = Vec::new();
        for (&id, dev) in &self.cfg.scsi {
            if dev.cdrom || dev.scratch || dev.path.trim().is_empty() {
                continue;
            }
            let is_chd = iris::chd_disk::is_chd(&dev.path);
            let has_overlay = if is_chd {
                iris::chd_disk::diff_path_for(std::path::Path::new(&dev.path)).exists()
            } else if dev.overlay {
                std::path::Path::new(&format!("{}.overlay", dev.path)).exists()
            } else {
                false
            };
            if has_overlay {
                entries.push((id, dev.path.clone(), is_chd));
            }
        }
        entries
    }

    /// The one place a chosen menu item takes effect.
    pub fn apply_menu_action(&mut self, action: Action, ctx: &egui::Context) {
        use crate::handle::Cmd;
        use egui::ViewportCommand;
        match action {
            // --- File ---
            Action::NewMachine => self.new_machine.open(),
            Action::SwitchMachine(name) => self.switch_to(&name),
            Action::RenameMachine => {
                // Opens the rename dialog seeded with the current name.
                self.rename_buffer = self.prefs.active_machine.clone();
            }
            Action::DeleteMachine => {
                if let Some(name) = self.prefs.active_machine.clone() {
                    self.prefs.machines.remove(&name);
                    self.prefs.active_machine = self.prefs.machines.keys().next().cloned();
                    if let Some(next) = self.prefs.active_machine.clone() {
                        self.cfg = self.prefs.machines[&next].clone();
                    } else {
                        self.cfg = iris::config::MachineConfig::default();
                        self.new_machine.open();
                    }
                    let _ = self.prefs.save();
                    self.toast(format!("deleted '{name}'"));
                }
            }
            Action::ImportToml => {
                if let Some(path) =
                    crate::native_open_dialog("Import iris.toml", &[("TOML", &["toml"])])
                {
                    let cfg = iris::config::MachineConfig::load_toml(&path.to_string_lossy());
                    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("imported");
                    let name = self.prefs.unique_name(stem);
                    self.prefs.machines.insert(name.clone(), cfg.clone());
                    self.prefs.active_machine = Some(name.clone());
                    self.cfg = cfg;
                    self.cfg_path = Some(path);
                    self.flush_machine();
                    self.toast(format!("imported as '{name}'"));
                }
            }
            Action::ExportToml => {
                if let Some(path) =
                    crate::native_save_dialog("Export iris.toml", &[("TOML", &["toml"])])
                {
                    self.save_config(path);
                }
            }
            Action::PrepareForPremiere => self.prepare_for_premiere(),
            Action::GrantDiskFolder => self.grant_disk_folder(),
            Action::RevealFolder(f) => crate::config_ui::reveal_in_file_manager(&f),
            Action::RevokeFolder(f) => {
                self.prefs.disk_folders.retain(|x| x != &f);
                self.prefs.bookmarks.remove(&f);
                let _ = self.prefs.save();
            }
            Action::Quit => {
                if self.cfg_dirty {
                    self.flush_machine();
                }
                ctx.send_viewport_cmd(ViewportCommand::Close);
            }

            // --- Machine ---
            Action::Start => self.start_emulator(),
            Action::Stop => self.request_stop(),
            Action::Reset => {
                self.emu.send(Cmd::Stop);
                self.start_emulator();
            }
            Action::ResetNvram => match settings::reset_nvram(&self.cfg.nvram) {
                Ok(()) => {
                    let seed = self.prefs.active_machine.as_deref().unwrap_or("indy");
                    let mac = settings::generate_mac_bytes(seed);
                    let _ = settings::write_nvram_mac(&self.cfg.nvram, mac);
                    self.toast(format!("NVRAM reset \u{2014} new MAC {}", settings::mac_to_string(mac)));
                }
                Err(e) => self.toast(format!("NVRAM reset failed: {e}")),
            },
            Action::SetCpu(c) => {
                self.cfg.machine.cpu = c;
                self.mark_dirty();
                self.toast(format!("{} \u{2014} applies at next Start", c.label()));
            }
            Action::SaveState(slot) => self.emu.send(Cmd::SaveState(slot)),
            Action::RestoreState(slot) => self.emu.send(Cmd::RestoreState(slot)),
            Action::Screenshot => {
                if let Some(p) = crate::native_save_dialog("Save screenshot", &[("PNG", &["png"])]) {
                    self.emu.send(Cmd::Screenshot(p));
                }
            }
            Action::SerialConsole => self.open_serial_console(),
            Action::ToggleCapture => {
                if self.input_state.captured {
                    crate::input::force_release(ctx, &mut self.input_state);
                } else {
                    crate::input::engage_capture(ctx, &mut self.input_state);
                }
            }

            // --- Memory ---
            Action::SetRam(mb) => {
                self.cfg.banks = crate::distribute_ram(mb);
                self.mark_dirty();
                self.toast(format!(
                    "RAM set to {} ({:?})",
                    crate::ram_summary(&self.cfg.banks),
                    self.cfg.banks
                ));
            }
            Action::SetBank(i, sz) => {
                self.cfg.banks[i] = sz;
                self.mark_dirty();
            }

            // --- SCSI ---
            Action::Scsi(cmd) => self.apply_scsi(cmd),
            Action::CowCommit { base, chd } => {
                if chd {
                    // A CHD commit recompresses — show the progress modal.
                    self.syncing = Some(crate::SyncJob { disk: 0, total: 1, fraction: 0.0 });
                }
                self.emu.send(Cmd::CowCommit { base, chd });
            }
            Action::CowDiscard { id, base, chd } => {
                // Destructive — confirm before discarding.
                self.cow_discard_confirm = Some(crate::CowDiscard { id, base, chd });
            }

            // --- View ---
            Action::ToggleFullscreen => {
                self.toggle_fullscreen(ctx);
            }
            Action::SetVmScale(s) => {
                self.prefs.vm_scale = s;
                self.pending_fb_snap = true;
                let _ = self.prefs.save();
            }
            Action::SetUiScale(s) => {
                self.prefs.ui_scale = s;
                ctx.set_zoom_factor(s);
                // Re-fit the window so bigger/smaller controls grow the window
                // rather than squeezing the picture.
                self.pending_fb_snap = true;
                let _ = self.prefs.save();
            }
            Action::ShowConfig => self.show_config_editor = true,

            // --- Help ---
            Action::NetCheck => self.show_net_check = true,
            Action::CameraTest => self.open_camera_test(),
            Action::HelpInfo => self.show_help_info = true,
            Action::NfsHelp => self.show_nfs_help = true,
            Action::Ultra64Help => {
                #[cfg(feature = "ultra64")]
                {
                    self.show_ultra64_help = true;
                }
            }
            Action::License => self.show_license = true,
            Action::Privacy => self.show_privacy = true,
            Action::About => self.native.show_about = true,
            Action::OpenUrl(url) => open_url(url),
        }
    }

    fn apply_scsi(&mut self, cmd: ScsiCmd) {
        use crate::handle::Cmd;
        use crate::scsi_menu::{self as sm, ScsiAction};
        // The picker runs here rather than inside the menu, so it opens after
        // the menu has closed instead of underneath it.
        let picked = match cmd {
            ScsiCmd::PickHdd(id) => {
                let cur = self.cfg.scsi.get(&id).map(|d| d.path.clone()).unwrap_or_default();
                let title = if cur.is_empty() { "Attach HDD" } else { "Replace HDD image" };
                sm::pick_disk(title, &cur).map(|path| ScsiAction::AttachHdd { id, path })
            }
            ScsiCmd::AttachEmptyCdrom(id) => Some(ScsiAction::AttachEmptyCdrom { id }),
            ScsiCmd::PickCdromWithDisc(id) => sm::pick_iso("Attach CD-ROM with disc", "")
                .map(|path| ScsiAction::AttachCdromWithDisc { id, path }),
            ScsiCmd::PickDisc(id) => {
                let cur = self.cfg.scsi.get(&id).map(|d| d.path.clone()).unwrap_or_default();
                sm::pick_iso("Insert disc", &cur).map(|path| ScsiAction::InsertDisc { id, path })
            }
            ScsiCmd::Eject(id) => Some(ScsiAction::Eject { id }),
            ScsiCmd::Detach(id) => Some(ScsiAction::Detach { id }),
            ScsiCmd::ToggleOverlay(id) => Some(ScsiAction::ToggleOverlay { id }),
            ScsiCmd::CreateBlank(id) => {
                self.create_disk.open_for(id);
                None
            }
            ScsiCmd::Remount(id) => {
                if self.emu.is_running() {
                    self.emu.send(Cmd::RemountCdrom { id });
                    self.toast(format!(
                        "SCSI #{id}: remount sent \u{2014} keep a shell focused on the console"
                    ));
                } else {
                    self.toast("Mount /CDROM: start the VM first");
                }
                None
            }
        };
        let Some(action) = picked else { return };

        // Media changes reach a running machine live; a new *drive* needs a
        // Stop → Start, since the SCSI bus is built at Machine::new.
        let live = match &action {
            ScsiAction::InsertDisc { id, path } | ScsiAction::AttachCdromWithDisc { id, path } => {
                Some(Cmd::LoadDisc { id: *id, path: path.clone(), remount: true })
            }
            ScsiAction::Eject { id } => Some(Cmd::EjectCdrom { id: *id }),
            _ => None,
        };
        let was_insert = matches!(action, ScsiAction::InsertDisc { .. });
        if let Some(msg) = sm::apply(&mut self.cfg, action) {
            self.mark_dirty();
            self.toast(msg);
        }
        match (live, self.emu.is_running()) {
            (Some(cmd), true) => self.emu.send(cmd),
            (Some(_), false) if was_insert => {
                self.toast("Disc saved \u{2014} Stop\u{2192}Start to load into SCSI drive")
            }
            _ => {}
        }
    }
}

/// Open `url` in the user's browser.
fn open_url(url: &str) {
    let _ = std::process::Command::new("open").arg(url).spawn();
}
