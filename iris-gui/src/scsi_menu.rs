use eframe::egui::{RichText, Ui};
use iris::config::{MachineConfig, ScsiDeviceConfig};
use std::path::Path;

/// What the user picked from a SCSI submenu, deferred for the App to act on
/// (so we don't hold &mut MachineConfig across nested closures and dialogs).
pub enum ScsiAction {
    None,
    AttachHdd { id: u8, path: String },
    AttachEmptyCdrom { id: u8 },
    AttachCdromWithDisc { id: u8, path: String },
    InsertDisc { id: u8, path: String },
    Eject { id: u8 },
    RemountInIrix { id: u8 },
    Detach { id: u8 },
    CreateBlank { id: u8 },
    ToggleOverlay { id: u8 },
}

/// Build the top-level "SCSI" menu. Returns at most one action per frame.
pub fn draw(ui: &mut Ui, cfg: &MachineConfig) -> ScsiAction {
    let mut action = ScsiAction::None;
    ui.set_min_width(280.0);
    for id in 1u8..=7 {
        let dev = cfg.scsi.get(&id);
        let label = render_label(id, dev);
        ui.menu_button(label, |ui| {
            ui.set_min_width(220.0);
            match dev {
                None => {
                    if ui.button("Attach HDD…").clicked() {
                        if let Some(p) = pick_disk("Attach HDD", "") {
                            action = ScsiAction::AttachHdd { id, path: p };
                        }
                        ui.close();
                    }
                    // Attaching a CD-ROM gives an empty drive by default; the
                    // user loads media afterwards via "Insert disc…". Mirrors
                    // real hardware and avoids an upfront file prompt.
                    if ui.button("Attach CD-ROM drive (empty)").clicked() {
                        action = ScsiAction::AttachEmptyCdrom { id };
                        ui.close();
                    }
                    if ui.button("Attach CD-ROM with disc…").clicked() {
                        if let Some(p) = pick_iso("Attach CD-ROM with disc", "") {
                            action = ScsiAction::AttachCdromWithDisc { id, path: p };
                        }
                        ui.close();
                    }
                    if ui.button("Create blank HDD image…").clicked() {
                        action = ScsiAction::CreateBlank { id };
                        ui.close();
                    }
                }
                Some(d) if d.is_daynaport() => {
                    ui.label("DaynaPort SCSI/Link (Ethernet). Configure its MAC and \
                              subnet on the Config tab.");
                    ui.separator();
                    if ui.button("Detach DaynaPort").clicked() {
                        action = ScsiAction::Detach { id };
                        ui.close();
                    }
                }
                Some(d) if d.is_cdrom() => {
                    let has_media = !d.path.is_empty() && Path::new(&d.path).exists();
                    if has_media {
                        if ui.button("Eject (tray empty)").clicked() {
                            action = ScsiAction::Eject { id };
                            ui.close();
                        }
                    }
                    let insert_label = if has_media { "Swap disc…" } else { "Insert disc…" };
                    if ui.button(insert_label).clicked() {
                        if let Some(p) = pick_iso("Insert disc", &d.path) {
                            action = ScsiAction::InsertDisc { id, path: p };
                        }
                        ui.close();
                    }
                    if has_media {
                        if ui.button("Mount /CDROM in IRIX…").clicked() {
                            action = ScsiAction::RemountInIrix { id };
                            ui.close();
                        }
                    }
                    ui.separator();
                    if ui.button("Detach CD-ROM drive").clicked() {
                        action = ScsiAction::Detach { id };
                        ui.close();
                    }
                }
                Some(d) => {
                    // HDD
                    let overlay_label = if d.overlay {
                        "Disable COW overlay"
                    } else {
                        "Enable COW overlay (writes -> .overlay)"
                    };
                    if ui.button(overlay_label).clicked() {
                        action = ScsiAction::ToggleOverlay { id };
                        ui.close();
                    }
                    if ui.button("Replace image…").clicked() {
                        if let Some(p) = pick_disk("Replace HDD image", &d.path) {
                            action = ScsiAction::AttachHdd { id, path: p };
                        }
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Detach hard drive").clicked() {
                        action = ScsiAction::Detach { id };
                        ui.close();
                    }
                }
            }
        });
    }
    ui.separator();
    ui.label(RichText::new(
        "CD-ROM: prefer SCSI #4. Insert/Swap hot-loads media + remounts /CDROM \
         (console shell must be active). New drives need Stop→Start."
    ).weak().small());
    action
}

fn render_label(id: u8, dev: Option<&ScsiDeviceConfig>) -> String {
    match dev {
        None => format!("SCSI #{id}: (empty)"),
        Some(d) if d.is_daynaport() => format!("SCSI #{id}: DaynaPort (Ethernet)"),
        Some(d) if d.is_cdrom() => {
            if d.path.is_empty() {
                format!("SCSI #{id}: CD (no media)")
            } else if !Path::new(&d.path).exists() {
                format!("SCSI #{id}: CD ⚠ {} (missing)", d.path)
            } else {
                let name = Path::new(&d.path).file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| d.path.clone());
                format!("SCSI #{id}: CD {name}")
            }
        }
        Some(d) => {
            let name = Path::new(&d.path).file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| d.path.clone());
            let size = std::fs::metadata(&d.path).map(|m| m.len()).unwrap_or(0);
            let mb = size as f64 / (1024.0 * 1024.0);
            let suffix = if d.overlay { " [COW]" } else { "" };
            if size > 0 {
                format!("SCSI #{id}: HDD {name} ({mb:.0} MB){suffix}")
            } else {
                format!("SCSI #{id}: HDD {name}{suffix}")
            }
        }
    }
}

// Seed the picker at `cur`'s folder, else the managed disks dir — never the OS default.
fn dialog_at(title: &str, cur: &str) -> rfd::FileDialog {
    let mut d = rfd::FileDialog::new().set_title(title);
    let p = Path::new(cur);
    match p.parent().filter(|d| !d.as_os_str().is_empty() && d.is_dir()) {
        Some(dir) => {
            d = d.set_directory(dir);
            if let Some(n) = p.file_name() { d = d.set_file_name(n.to_string_lossy()); }
        }
        None => if let Some(dir) = crate::settings::GuiSettings::disks_dir().filter(|d| d.is_dir()) {
            d = d.set_directory(dir);
        }
    }
    d
}

fn pick_disk(title: &str, cur: &str) -> Option<String> {
    dialog_at(title, cur)
        .add_filter("Disk images", &["raw", "img", "chd"])
        .add_filter("All", &["*"])
        .pick_file()
        .map(|p| p.to_string_lossy().into_owned())
}

pub fn pick_iso(title: &str, cur: &str) -> Option<String> {
    dialog_at(title, cur)
        .add_filter("ISO images", &["iso", "chd"])
        .add_filter("All", &["*"])
        .pick_file()
        .map(|p| p.to_string_lossy().into_owned())
}

/// Apply an action to the config.
pub fn apply(cfg: &mut MachineConfig, action: ScsiAction) -> Option<String> {
    match action {
        ScsiAction::None => None,
        ScsiAction::AttachHdd { id, path } => {
            cfg.scsi.insert(id, ScsiDeviceConfig { path, ..Default::default() });
            Some(format!("scsi{id}: HDD attached"))
        }
        ScsiAction::AttachEmptyCdrom { id } => {
            cfg.scsi.insert(id, ScsiDeviceConfig { cdrom: true, ..Default::default() });
            Some(format!("scsi{id}: empty CD-ROM drive attached (Stop→Start if VM is running)"))
        }
        ScsiAction::AttachCdromWithDisc { id, path } => {
            cfg.scsi.insert(id, ScsiDeviceConfig {
                path: path.clone(), cdrom: true, ..Default::default()
            });
            Some(format!("scsi{id}: CD-ROM attached with disc"))
        }
        ScsiAction::InsertDisc { id, path } => {
            if let Some(d) = cfg.scsi.get_mut(&id) { d.path = path; }
            Some(format!("scsi{id}: disc inserted"))
        }
        ScsiAction::Eject { id } => {
            if let Some(d) = cfg.scsi.get_mut(&id) { d.path = String::new(); }
            Some(format!("scsi{id}: ejected"))
        }
        ScsiAction::Detach { id } => {
            cfg.scsi.remove(&id);
            Some(format!("scsi{id}: detached"))
        }
        ScsiAction::CreateBlank { .. } => {
            // App opens the CreateDiskDialog; nothing to apply yet.
            None
        }
        ScsiAction::ToggleOverlay { id } => {
            if let Some(d) = cfg.scsi.get_mut(&id) { d.overlay = !d.overlay; }
            Some(format!("scsi{id}: overlay toggled"))
        }
        ScsiAction::RemountInIrix { .. } => None,
    }
}
