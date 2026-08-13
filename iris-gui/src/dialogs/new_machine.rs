use eframe::egui::{self, Color32, ComboBox, Grid, RichText, TextEdit};
use iris::config::{MachineConfig, MachineProfile, ScsiDeviceConfig, VALID_BANK_SIZES};
use iris::vc2_timings::NewportResolution;

use crate::ram::RAM_PRESETS;

/// "New machine" startup dialog.
/// Pops up at first run (or on `File → New machine…`) to bootstrap a config.
pub struct NewMachineDialog {
    open: bool,
    pub name: String,
    /// Emulated SGI machine model (Indy IP24 / Indigo2 IP22).
    pub profile: MachineProfile,
    /// Host-forced Newport video mode (`Guest` leaves it to IRIX/setmon).
    pub resolution: NewportResolution,
    pub prom_path: String,
    pub use_embedded_prom: bool,
    pub nvram_path: String,
    pub ram_total_mb: u32,
    /// If true the dialog exposes 4 per-bank selectors and ignores ram_total_mb.
    pub ram_advanced: bool,
    pub ram_banks: [u32; 4],
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
            resolution: NewportResolution::default(),
            prom_path: "prom.bin".into(),
            use_embedded_prom: true,
            nvram_path: crate::settings::GuiSettings::default_nvram_path(),
            ram_total_mb: 256,
            ram_advanced: false,
            ram_banks: [128, 128, 0, 0],
            scsi1_path: crate::settings::GuiSettings::default_disk_path(1),
            create_blank_scsi1: false,
            cdrom4_path: String::new(),
            attach_cdrom: false,
            result: None,
        }
    }
}

pub fn distribute_ram(total: u32) -> [u32; 4] {
    // Greedy fill banks 0..3 with the largest valid bank size that fits.
    let mut remaining = total;
    let mut banks = [0u32; 4];
    for slot in &mut banks {
        // Pick the largest size in VALID_BANK_SIZES that is <= remaining.
        let pick = VALID_BANK_SIZES.iter().filter(|&&s| s > 0 && s <= remaining).max().copied().unwrap_or(0);
        *slot = pick;
        remaining -= pick;
        if remaining == 0 { break; }
    }
    banks
}

impl NewMachineDialog {
    pub fn open(&mut self) { self.open = true; self.result = None; }
    pub fn take_result(&mut self) -> Option<NewMachineResult> { self.result.take() }

    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open { return; }
        let mut close = false;
        egui::Window::new("New machine")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_min_width(440.0);
                ui.label(RichText::new("Configure a new SGI machine").strong());
                ui.add_space(4.0);

                Grid::new("new_machine_grid").num_columns(2).striped(true).show(ui, |ui| {
                    ui.label("Name");
                    ui.add(TextEdit::singleline(&mut self.name).desired_width(200.0));
                    ui.end_row();

                    ui.label("Machine model");
                    ComboBox::from_id_salt("nm_profile")
                        .selected_text(self.profile.label())
                        .show_ui(ui, |ui| {
                            for p in MachineProfile::ALL {
                                ui.selectable_value(&mut self.profile, p, p.label());
                            }
                        });
                    ui.end_row();

                    ui.label("Display resolution");
                    ComboBox::from_id_salt("nm_resolution")
                        .selected_text(self.resolution.label())
                        .show_ui(ui, |ui| {
                            for mode in NewportResolution::ALL {
                                ui.selectable_value(&mut self.resolution, mode, mode.label());
                            }
                        });
                    ui.end_row();

                    ui.label("PROM image");
                    ui.horizontal(|ui| {
                        ui.add_enabled(!self.use_embedded_prom,
                            TextEdit::singleline(&mut self.prom_path).desired_width(260.0));
                        if ui.add_enabled(!self.use_embedded_prom, egui::Button::new("📁")).clicked() {
                            if let Some(p) = rfd::FileDialog::new()
                                .add_filter("PROM image", &["bin"])
                                .pick_file()
                            {
                                self.prom_path = p.to_string_lossy().into_owned();
                            }
                        }
                    });
                    ui.end_row();
                    ui.label("");
                    ui.checkbox(&mut self.use_embedded_prom, "Use embedded PROM (bundled with iris)");
                    ui.end_row();

                    ui.label("NVRAM file");
                    ui.horizontal(|ui| {
                        ui.add(TextEdit::singleline(&mut self.nvram_path).desired_width(260.0));
                        if ui.button("📁").clicked() {
                            if let Some(p) = rfd::FileDialog::new()
                                .add_filter("NVRAM", &["bin"]).save_file()
                            {
                                self.nvram_path = p.to_string_lossy().into_owned();
                            }
                        }
                    });
                    ui.end_row();

                    if !self.ram_advanced {
                        ui.label("Total RAM");
                        ComboBox::from_id_salt("ram_total")
                            .selected_text(format!("{} MB", self.ram_total_mb))
                            .show_ui(ui, |ui| {
                                for &s in RAM_PRESETS {
                                    ui.selectable_value(&mut self.ram_total_mb, s, format!("{s} MB"));
                                }
                            });
                        ui.end_row();
                    } else {
                        for i in 0..4 {
                            ui.label(format!("Bank {i}"));
                            ComboBox::from_id_salt(("nm_bank", i))
                                .selected_text(format!("{} MB", self.ram_banks[i]))
                                .show_ui(ui, |ui| {
                                    for &sz in VALID_BANK_SIZES {
                                        ui.selectable_value(&mut self.ram_banks[i], sz, format!("{sz} MB"));
                                    }
                                });
                            ui.end_row();
                        }
                        let total: u32 = self.ram_banks.iter().sum();
                        ui.label("Total");
                        ui.label(format!("{total} MB"));
                        ui.end_row();
                    }
                    ui.label("");
                    ui.checkbox(&mut self.ram_advanced, "Advanced: configure individual banks");
                    ui.end_row();
                });

                ui.separator();
                ui.label(RichText::new("Boot disk (optional)").strong());
                Grid::new("new_machine_disk").num_columns(2).striped(true).show(ui, |ui| {
                    ui.label("SCSI ID 1 (HDD)");
                    ui.horizontal(|ui| {
                        ui.add(TextEdit::singleline(&mut self.scsi1_path).desired_width(260.0));
                        if ui.button("📁").clicked() {
                            if let Some(p) = rfd::FileDialog::new()
                                .add_filter("Disk image", &["raw", "img", "chd"])
                                .pick_file()
                            {
                                self.scsi1_path = p.to_string_lossy().into_owned();
                                self.create_blank_scsi1 = false;
                            }
                        }
                    });
                    ui.end_row();
                    ui.label("");
                    ui.checkbox(&mut self.create_blank_scsi1,
                        "If the file doesn't exist, treat as empty (suitable for fresh IRIX install)");
                    ui.end_row();

                    ui.label("SCSI ID 4 (CD-ROM)");
                    ui.horizontal(|ui| {
                        ui.add(TextEdit::singleline(&mut self.cdrom4_path).desired_width(260.0));
                        if ui.button("📁").clicked() {
                            if let Some(p) = rfd::FileDialog::new()
                                .add_filter("ISO", &["iso"])
                                .pick_file()
                            {
                                self.cdrom4_path = p.to_string_lossy().into_owned();
                                self.attach_cdrom = true;
                            }
                        }
                    });
                    ui.end_row();
                    ui.label("");
                    ui.checkbox(&mut self.attach_cdrom, "Attach an install CD-ROM at SCSI ID 4");
                    ui.end_row();
                });

                ui.separator();
                ui.label(RichText::new(
                    "You can refine networking, JIT, video-in and CI settings from the menus after creation."
                ).color(Color32::GRAY).small());

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() { close = true; }
                    if ui.add(egui::Button::new(RichText::new("Create").strong())
                        .fill(Color32::from_rgb(60, 110, 60))).clicked()
                    {
                        let mut cfg = MachineConfig::default();
                        cfg.machine.profile = self.profile;
                        cfg.graphics.resolution = self.resolution;
                        cfg.prom = if self.use_embedded_prom { String::new() } else { self.prom_path.clone() };
                        // Empty prom path makes Machine::new fall back to embedded
                        // (the load path warns + falls back when the file is missing).
                        if cfg.prom.is_empty() { cfg.prom = "(embedded)".into(); }
                        cfg.nvram = self.nvram_path.clone();
                        cfg.banks = if self.ram_advanced {
                            self.ram_banks
                        } else {
                            distribute_ram(self.ram_total_mb)
                        };
                        // SCSI defaults: drop the built-in entries unless the
                        // user explicitly opted in.
                        cfg.scsi.clear();
                        if !self.scsi1_path.is_empty() {
                            cfg.scsi.insert(1, ScsiDeviceConfig {
                                path: self.scsi1_path.clone(),
                                ..Default::default()
                            });
                        }
                        if self.attach_cdrom && !self.cdrom4_path.is_empty() {
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
