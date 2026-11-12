use eframe::egui::{self, Color32, ComboBox, Grid, RichText, TextEdit};
use iris::config::{MachineConfig, MachineProfile, ScsiDeviceConfig, VALID_BANK_SIZES};
use iris::cpu::parts::{self, Part};
use iris::dev::ng1::vc2_timings::NewportResolution;

use crate::i18n::{subtitle, ResponseExt};
use crate::ram::ram_presets;

/// "New machine" startup dialog.
/// Pops up at first run (or on `File → New machine…`) to bootstrap a config.
pub struct NewMachineDialog {
    open: bool,
    pub name: String,
    /// Emulated SGI machine profile.
    pub profile: MachineProfile,
    pub cpu: &'static Part,
    pub graphics: iris::config::GraphicsBoard,
    /// Host-forced Newport video mode (`Guest` leaves it to IRIX/setmon).
    pub resolution: NewportResolution,
    pub prom_path: String,
    pub use_embedded_prom: bool,
    pub nvram_path: String,
    pub nveeprom_path: String,
    pub ram_total_mb: u32,
    /// If true, expose each profile's banks instead of the total preset.
    pub ram_advanced: bool,
    pub ram_banks: [u32; iris::config::MAX_BANKS],
    pub scsi1_path: String,
    pub create_blank_scsi1: bool,
    pub cdrom4_path: String,
    pub attach_cdrom: bool,
    result: Option<NewMachineResult>,
}

pub struct NewMachineResult {
    pub name: String,
    pub cfg: MachineConfig,
}

impl Default for NewMachineDialog {
    fn default() -> Self {
        Self {
            open: false,
            name: "indy".into(),
            profile: MachineProfile::default(),
            cpu: &parts::R4400_150,
            graphics: iris::config::GraphicsBoard::Newport,
            resolution: NewportResolution::default(),
            prom_path: "prom.bin".into(),
            use_embedded_prom: true,
            nvram_path: crate::settings::GuiSettings::default_nvram_path(),
            nveeprom_path: crate::settings::GuiSettings::default_nveeprom_path(),
            ram_total_mb: 256,
            ram_advanced: false,
            ram_banks: iris::config::banks_from4([128, 128, 0, 0]),
            scsi1_path: crate::settings::GuiSettings::default_disk_path(1),
            create_blank_scsi1: false,
            cdrom4_path: String::new(),
            attach_cdrom: false,
            result: None,
        }
    }
}

/// Fill the profile's banks largest first, without gaps.
pub fn distribute_ram(total: u32, profile: MachineProfile) -> [u32; iris::config::MAX_BANKS] {
    let mut remaining = total;
    let mut banks = [0u32; iris::config::MAX_BANKS];
    for slot in banks.iter_mut().take(profile.memory_banks()) {
        // Pick the largest size in VALID_BANK_SIZES that is <= remaining.
        let pick = VALID_BANK_SIZES.iter()
            .filter(|&&s| s > 0 && s <= remaining && profile.bank_sizes().contains(&s))
            .max().copied().unwrap_or(0);
        *slot = pick;
        remaining -= pick;
        if remaining == 0 { break; }
    }
    banks
}

impl NewMachineDialog {
    pub fn open(&mut self) { self.open = true; self.result = None; }
    pub fn take_result(&mut self) -> Option<NewMachineResult> { self.result.take() }

    /// Resolve picker defaults in the prospective folder, before creation.
    fn machine_dir(&self, prefs: &crate::settings::GuiSettings) -> std::path::PathBuf {
        let name = if self.name.trim().is_empty() { "indy" } else { self.name.trim() };
        if crate::machines::validate_name(name).is_ok() {
            if let Ok(dir) = prefs.machine_dir(&prefs.unique_name(name)) { return dir; }
        }
        prefs.root().map(std::path::Path::to_path_buf).unwrap_or_else(|_| std::path::PathBuf::from("."))
    }

    pub fn show(&mut self, ctx: &egui::Context, prefs: &crate::settings::GuiSettings) {
        if !self.open { return; }
        let mut close = false;
        egui::Window::new(t!("dialog-new-machine-title"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_min_width(440.0);
                subtitle(ui, t!("dialog-new-machine-subtitle"));
                ui.add_space(4.0);

                Grid::new("new_machine_grid").num_columns(2).striped(true).show(ui, |ui| {
                    ui.label(t!("dialog-new-machine-name"));
                    ui.add(TextEdit::singleline(&mut self.name).desired_width(200.0));
                    ui.end_row();

                    ui.label(t!("dialog-new-machine-model"));
                    let profile_before = self.profile;
                    ComboBox::from_id_salt("nm_profile")
                        .selected_text(self.profile.label())
                        .show_ui(ui, |ui| {
                            for p in MachineProfile::ALL {
                                ui.selectable_value(&mut self.profile, p, p.label());
                            }
                        });
                    if self.profile != profile_before {
                        self.cpu = self.profile.cpus()[0];
                        self.graphics = if self.profile.ip28() { iris::config::GraphicsBoard::SolidImpact } else { iris::config::GraphicsBoard::default_for(self.profile) };
                        self.use_embedded_prom = self.profile.has_embedded_prom();
                        self.ram_total_mb = crate::ram::ram_presets(self.profile)[0];
                        self.ram_banks = distribute_ram(self.ram_total_mb, self.profile);
                        self.resolution = NewportResolution::Guest;

                    }
                    ui.end_row();


                    ui.label(t!("dialog-new-machine-cpu"));
                    // Only the CPUs this machine can be built with; a machine
                    // with one has no choice.
                    let cpus = self.profile.cpus();
                    ui.add_enabled_ui(cpus.len() > 1, |ui| {
                        ComboBox::from_id_salt("nm_cpu")
                            .selected_text(self.cpu.label())
                            .show_ui(ui, |ui| {
                                for &c in cpus {
                                    ui.selectable_value(&mut self.cpu, c, c.label());
                                }
                            });
                    });
                    if cpus.len() == 1 {
                        ui.label(RichText::new(cpus[0].label()).weak());
                    }
                    ui.end_row();

                    ui.label(t!("config-graphics-board"));
                    ComboBox::from_id_salt("nm_graphics").selected_text(if self.profile == MachineProfile::O2Ip32 { t!("extra-gbe") } else { self.graphics.label().to_string() }).show_ui(ui, |ui| {
                        for board in iris::config::GraphicsBoard::ALL {
                            if crate::extra_ui::board_supported(board, self.profile) { ui.selectable_value(&mut self.graphics, board, board.label()); }
                        }
                    });
                    ui.end_row();

                    if self.profile.has_embedded_prom() && !self.profile.ip28() {
                        ui.label(t!("dialog-new-machine-resolution"));
                        ComboBox::from_id_salt("nm_resolution")
                            .selected_text(self.resolution.label())
                            .show_ui(ui, |ui| {
                                for mode in NewportResolution::ALL {
                                    ui.selectable_value(&mut self.resolution, mode, mode.label());
                                }
                            });
                        ui.end_row();
                    }

                    ui.label(t!("dialog-new-machine-prom"));
                    ui.horizontal(|ui| {
                        ui.add_enabled(!self.use_embedded_prom,
                            TextEdit::singleline(&mut self.prom_path).desired_width(260.0));
                        if ui.add_enabled(!self.use_embedded_prom, egui::Button::new("📁")).clicked() {
                            if let Some(p) = crate::filedialog::dialog_with_base(
                                &t!("dialog-new-machine-prom-picker"), &self.prom_path, &self.machine_dir(prefs),
                                crate::filedialog::Anchor::Data,
                                crate::filedialog::Purpose::Open,
                                &[(t!("dialog-new-machine-prom-filter").as_str(), &["bin"])]).pick_file()
                            {
                                self.prom_path = p.to_string_lossy().into_owned();
                            }
                        }
                    });
                    ui.end_row();
                    ui.label("");
                    // Only the Indy and Indigo2 family have embedded PROMs.
                    ui.add_enabled(self.profile.has_embedded_prom(),
                        egui::Checkbox::new(&mut self.use_embedded_prom, t!("dialog-new-machine-embedded-prom")));
                    ui.end_row();

                    ui.label(t!("dialog-new-machine-nvram"));
                    ui.horizontal(|ui| {
                        ui.add(TextEdit::singleline(&mut self.nvram_path).desired_width(260.0));
                        if ui.button("📁").clicked() {
                            if let Some(p) = crate::filedialog::dialog_with_base(
                                &t!("dialog-new-machine-nvram-picker"), &self.nvram_path, &self.machine_dir(prefs),
                                crate::filedialog::Anchor::Data,
                                crate::filedialog::Purpose::Save,
                                &[(t!("dialog-new-machine-nvram-filter").as_str(), &["bin"])]).save_file()
                            {
                                self.nvram_path = p.to_string_lossy().into_owned();
                            }
                        }
                    });
                    ui.end_row();

                    // The O2's NVRAM is its own PROM flash, already asked for
                    // above; it has no separate motherboard EEPROM.
                    if self.profile.has_embedded_prom() {
                        ui.label(t!("dialog-new-machine-eeprom"))
                            .tip(t!("dialog-new-machine-eeprom-tip"));
                        ui.horizontal(|ui| {
                            ui.add(TextEdit::singleline(&mut self.nveeprom_path).desired_width(260.0));
                            if ui.button("📁").clicked() {
                                if let Some(p) = crate::filedialog::dialog_with_base(
                                    &t!("dialog-new-machine-eeprom-picker"), &self.nveeprom_path, &self.machine_dir(prefs),
                                    crate::filedialog::Anchor::Data,
                                    crate::filedialog::Purpose::Save,
                                    &[(t!("dialog-new-machine-eeprom-filter").as_str(), &["bin"])]).save_file()
                                {
                                    self.nveeprom_path = p.to_string_lossy().into_owned();
                                }
                            }
                        });
                        ui.end_row();
                    }

                    if !self.ram_advanced {
                        ui.label(t!("dialog-new-machine-ram"));
                        ComboBox::from_id_salt("ram_total")
                            .selected_text(t!("dialog-new-machine-mb", size = self.ram_total_mb))
                            .show_ui(ui, |ui| {
                                for &s in ram_presets(self.profile) {
                                    ui.selectable_value(&mut self.ram_total_mb, s, t!("dialog-new-machine-mb", size = s));
                                }
                            });
                        ui.end_row();
                    } else {
                        for i in 0..self.profile.memory_banks() {
                            ui.label(t!("dialog-new-machine-bank", bank = i));
                            ComboBox::from_id_salt(("nm_bank", i))
                                .selected_text(t!("dialog-new-machine-mb", size = self.ram_banks[i]))
                                .show_ui(ui, |ui| {
                                    for &sz in VALID_BANK_SIZES {
                                        ui.add_enabled_ui(self.profile.bank_sizes().contains(&sz), |ui| {
                                            ui.selectable_value(&mut self.ram_banks[i], sz, t!("dialog-new-machine-mb", size = sz))
                                                .on_disabled_hover_text(crate::ram::bank_size_tip(self.profile));
                                        });
                                    }
                                });
                            ui.end_row();
                        }
                        let total: u32 = self.ram_banks.iter().sum();
                        ui.label(t!("dialog-new-machine-ram-total"));
                        ui.label(t!("dialog-new-machine-mb", size = total));
                        ui.end_row();
                    }
                    ui.label("");
                    ui.checkbox(&mut self.ram_advanced, t!("dialog-new-machine-ram-banks"));
                    ui.end_row();
                });

                if self.profile == MachineProfile::Indigo2Ip28 {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(t!("dialog-new-machine-ip28-note"))
                        .color(Color32::from_rgb(220, 170, 90))
                        .small(),
                    );
                }

                if self.profile == MachineProfile::OctaneIp30 {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(t!("dialog-new-machine-octane-note"))
                        .color(Color32::from_rgb(220, 170, 90))
                        .small(),
                    );
                }

                if self.profile == MachineProfile::O2Ip32 {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(t!("dialog-new-machine-o2-note"))
                        .color(Color32::from_rgb(220, 170, 90))
                        .small(),
                    );
                }

                ui.separator();
                ui.label(RichText::new(t!("dialog-new-machine-disks")).strong());
                Grid::new("new_machine_disk").num_columns(2).striped(true).show(ui, |ui| {
                    ui.label(t!("dialog-new-machine-hdd"));
                    ui.horizontal(|ui| {
                        ui.add(TextEdit::singleline(&mut self.scsi1_path).desired_width(260.0));
                        if ui.button("📁").clicked() {
                            if let Some(p) = crate::filedialog::dialog_with_base(
                                &t!("dialog-new-machine-hdd-picker"), &self.scsi1_path, &self.machine_dir(prefs),
                                crate::filedialog::Anchor::Disks,
                                crate::filedialog::Purpose::Open,
                                &[(t!("dialog-new-machine-hdd-filter").as_str(), &["raw", "img", "chd"])]).pick_file()
                            {
                                self.scsi1_path = p.to_string_lossy().into_owned();
                                self.create_blank_scsi1 = false;
                            }
                        }
                    });
                    ui.end_row();
                    ui.label("");
                    ui.checkbox(&mut self.create_blank_scsi1, t!("dialog-new-machine-hdd-blank"));
                    ui.end_row();

                    ui.label(t!("dialog-new-machine-cdrom"));
                    ui.horizontal(|ui| {
                        ui.add(TextEdit::singleline(&mut self.cdrom4_path).desired_width(260.0));
                        if ui.button("📁").clicked() {
                            if let Some(p) = crate::filedialog::dialog_with_base(
                                &t!("dialog-new-machine-cdrom-picker"), &self.cdrom4_path, &self.machine_dir(prefs),
                                crate::filedialog::Anchor::Disks,
                                crate::filedialog::Purpose::Open,
                                &[(t!("dialog-new-machine-iso-filter").as_str(), &["iso"]),
                                  (t!("dialog-new-machine-cd-filter").as_str(), &["iso", "chd"])]).pick_file()
                            {
                                self.cdrom4_path = p.to_string_lossy().into_owned();
                                self.attach_cdrom = true;
                            }
                        }
                    });
                    ui.end_row();
                    ui.label("");
                    ui.checkbox(&mut self.attach_cdrom, t!("dialog-new-machine-cdrom-attach"));
                    ui.end_row();
                });

                ui.separator();
                subtitle(ui, t!("dialog-new-machine-footer-subtitle"));

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.button(t!("common-cancel")).clicked() { close = true; }
                    let name = if self.name.trim().is_empty() { "indy" } else { self.name.trim() };
                    let valid = crate::machines::validate_name(name);
                    if let Err(e) = &valid { ui.colored_label(Color32::from_rgb(230, 140, 70), e); }
                    if ui.add_enabled(valid.is_ok(), egui::Button::new(RichText::new(t!("dialog-new-machine-create")).strong())
                        .fill(Color32::from_rgb(60, 110, 60))).clicked()
                    {
                        let mut cfg = MachineConfig::default();
                        cfg.machine.profile = self.profile;
                        cfg.set_cpu(self.cpu);
                        cfg.banks_set = true;
                        cfg.graphics.board = Some(self.graphics);
                        cfg.graphics.resolution = if self.graphics == iris::config::GraphicsBoard::Newport { self.resolution } else { NewportResolution::Guest };
                        cfg.prom = if self.use_embedded_prom { String::new() } else { self.prom_path.clone() };
                        // Empty prom path makes Machine::new fall back to embedded
                        // (the load path warns + falls back when the file is missing).
                        if cfg.prom.is_empty() { cfg.prom = "(embedded)".into(); }
                        cfg.nvram = self.nvram_path.clone();
                        cfg.nveeprom = self.nveeprom_path.clone();
                        cfg.banks = if self.ram_advanced {
                            self.ram_banks
                        } else {
                            distribute_ram(self.ram_total_mb, self.profile)
                        };
                        // Keep the PROM's flash records between runs.
                        if self.profile == MachineProfile::OctaneIp30 {
                            cfg.octane.flash = Some("flash-ip30.bin".into());
                        }
                        // SCSI defaults: drop the built-in entries unless the
                        // user explicitly opted in.
                        if !self.profile.is_ip35_bricks() { cfg.scsi.clear(); }
                        if self.profile.is_ip35_bricks() && !self.scsi1_path.is_empty() {
                            let mut fc = iris::config::FibreChannelConfig::default();
                            fc.disks.insert(1, ScsiDeviceConfig { path: self.scsi1_path.clone(), ..Default::default() });
                            cfg.fibre_channel = Some(fc);
                        }
                        if !self.profile.is_ip35_bricks() && !self.scsi1_path.is_empty() {
                            cfg.scsi.insert(1, ScsiDeviceConfig {
                                path: self.scsi1_path.clone(),
                                ..Default::default()
                            });
                        }
                        if !self.profile.is_ip35_bricks() && self.attach_cdrom && !self.cdrom4_path.is_empty() {
                            cfg.scsi.insert(4, ScsiDeviceConfig {
                                path: self.cdrom4_path.clone(),
                                cdrom: true,
                                ..Default::default()
                            });
                        }
                        let name = if self.name.trim().is_empty() { "indy".to_string() } else { self.name.trim().to_string() };
                        self.result = Some(NewMachineResult { name, cfg });
                        close = true;
                    }
                });
            });
        if close { self.open = false; }
    }
}

#[cfg(test)]
mod ram_tests {
    use super::{distribute_ram, MachineProfile};

    #[test]
    fn ip28_gigabyte_uses_two_banks() {
        assert_eq!(distribute_ram(1024, MachineProfile::Indigo2Ip28), iris::config::banks_from4([512, 512, 0, 0]));
        assert_eq!(distribute_ram(768, MachineProfile::Indigo2Ip28), iris::config::banks_from4([512, 256, 0, 0]));
        assert_eq!(distribute_ram(512, MachineProfile::IndyIp24), iris::config::banks_from4([128, 128, 128, 128]));
    }

    #[test]
    fn octane_banks_are_valid_pairs_from_bank_zero() {
        let octane = MachineProfile::OctaneIp30;
        assert_eq!(distribute_ram(128, octane), iris::config::banks_from4([128, 0, 0, 0]));
        assert_eq!(distribute_ram(192, octane), iris::config::banks_from4([128, 64, 0, 0]));
        assert_eq!(distribute_ram(2048, octane), iris::config::banks_from4([2048, 0, 0, 0]));
        assert_eq!(distribute_ram(8192, octane), iris::config::banks_from4([2048, 2048, 2048, 2048]));
        for &total in crate::ram::ram_presets(octane) {
            let mut cfg = iris::config::MachineConfig::default();
            cfg.machine.profile = octane;
            cfg.set_cpu(octane.cpus()[0]);
            cfg.banks = distribute_ram(total, octane);
            cfg.validate().unwrap_or_else(|e| panic!("{total} MB: {e}"));
        }
    }
}
