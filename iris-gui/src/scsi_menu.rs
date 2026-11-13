use crate::i18n::{subtitle, ResponseExt};
use eframe::egui::Ui;
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
                    if ui.button(t!("scsi-attach-hdd")).clicked() {
                        if let Some(p) = pick_disk(&t!("scsi-attach-hdd-picker"), "") {
                            action = ScsiAction::AttachHdd { id, path: p };
                        }
                        ui.close();
                    }
                    // Attaching a CD-ROM gives an empty drive by default; the
                    // user loads media afterwards via "Insert disc…". Mirrors
                    // real hardware and avoids an upfront file prompt.
                    if ui.button(t!("scsi-attach-cdrom-empty")).clicked() {
                        action = ScsiAction::AttachEmptyCdrom { id };
                        ui.close();
                    }
                    if ui.button(t!("scsi-attach-cdrom-disc")).clicked() {
                        if let Some(p) = pick_iso(&t!("scsi-attach-cdrom-disc-picker"), "") {
                            action = ScsiAction::AttachCdromWithDisc { id, path: p };
                        }
                        ui.close();
                    }
                    if ui.button(t!("scsi-create-blank")).clicked() {
                        action = ScsiAction::CreateBlank { id };
                        ui.close();
                    }
                }
                Some(d) if d.is_daynaport() => {
                    subtitle(ui, t!("scsi-daynaport-subtitle"));
                    ui.separator();
                    if ui.button(t!("scsi-detach-daynaport")).clicked() {
                        action = ScsiAction::Detach { id };
                        ui.close();
                    }
                }
                Some(d) if d.is_cdrom() => {
                    let has_media = !d.path.is_empty() && (iris::http_disk::is_url(&d.path) || Path::new(&d.path).exists());
                    if has_media {
                        if ui.button(t!("scsi-eject")).clicked() {
                            action = ScsiAction::Eject { id };
                            ui.close();
                        }
                    }
                    let insert_label = if has_media { t!("scsi-change-disc") } else { t!("scsi-insert-disc") };
                    if ui.button(insert_label).clicked() {
                        if let Some(p) = pick_iso(&t!("scsi-insert-disc-picker"), &d.path) {
                            action = ScsiAction::InsertDisc { id, path: p };
                        }
                        ui.close();
                    }
                    if has_media {
                        if ui.button(t!("scsi-mount-in-irix")).tip(t!("scsi-mount-in-irix-tip")).clicked() {
                            action = ScsiAction::RemountInIrix { id };
                            ui.close();
                        }
                    }
                    ui.separator();
                    if ui.button(t!("scsi-detach-cdrom")).clicked() {
                        action = ScsiAction::Detach { id };
                        ui.close();
                    }
                }
                Some(d) => {
                    // HDD
                    let overlay_label = if d.overlay {
                        t!("scsi-overlay-disable")
                    } else {
                        t!("scsi-overlay-enable")
                    };
                    if ui.button(overlay_label).tip(t!("scsi-overlay-tip")).clicked() {
                        action = ScsiAction::ToggleOverlay { id };
                        ui.close();
                    }
                    if ui.button(t!("scsi-replace-image")).clicked() {
                        if let Some(p) = pick_disk(&t!("scsi-replace-image-picker"), &d.path) {
                            action = ScsiAction::AttachHdd { id, path: p };
                        }
                        ui.close();
                    }
                    ui.separator();
                    if ui.button(t!("scsi-detach-hdd")).clicked() {
                        action = ScsiAction::Detach { id };
                        ui.close();
                    }
                }
            }
        });
    }
    ui.separator();
    subtitle(ui, t!("scsi-menu-subtitle"));
    action
}

fn render_label(id: u8, dev: Option<&ScsiDeviceConfig>) -> String {
    match dev {
        None => t!("scsi-slot-empty", id = id),
        Some(d) if d.is_daynaport() => t!("scsi-slot-daynaport", id = id),
        Some(d) if d.is_cdrom() => {
            if d.path.is_empty() {
                t!("scsi-slot-cdrom-empty", id = id)
            } else if !iris::http_disk::is_url(&d.path) && !Path::new(&d.path).exists() {
                t!("scsi-slot-cdrom-missing", id = id, path = &d.path)
            } else {
                let name = Path::new(&d.path).file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| d.path.clone());
                t!("scsi-slot-cdrom", id = id, name = name)
            }
        }
        Some(d) => {
            let name = Path::new(&d.path).file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| d.path.clone());
            let size = std::fs::metadata(&d.path).map(|m| m.len()).unwrap_or(0);
            let mb = size as f64 / (1024.0 * 1024.0);
            let size_mb = format!("{mb:.0}");
            match (size > 0, d.overlay) {
                (true, false) => t!("scsi-slot-hdd-size", id = id, name = name, size = size_mb),
                (true, true) => t!("scsi-slot-hdd-size-overlay", id = id, name = name, size = size_mb),
                (false, false) => t!("scsi-slot-hdd", id = id, name = name),
                (false, true) => t!("scsi-slot-hdd-overlay", id = id, name = name),
            }
        }
    }
}

// Seed the picker at `cur`'s folder, else the managed disks dir — never the OS
// default. See `crate::filedialog`, which is where the "else" used to be a
// silent no-op and the panel opened at whatever the user last browsed.
fn dialog_at(title: &str, cur: &str) -> rfd::FileDialog {
    crate::filedialog::dialog(title, cur, crate::filedialog::Anchor::Disks,
                              crate::filedialog::Purpose::Open)
}

fn pick_disk(title: &str, cur: &str) -> Option<String> {
    dialog_at(title, cur)
        .add_filter(t!("scsi-filter-disk"), &["raw", "img", "chd"])
        .add_filter(t!("scsi-filter-all"), &["*"])
        .pick_file()
        .map(|p| crate::picked_path(&p))
}

pub fn pick_iso(title: &str, cur: &str) -> Option<String> {
    dialog_at(title, cur)
        .add_filter(t!("scsi-filter-iso"), &["iso", "chd"])
        .add_filter(t!("scsi-filter-all"), &["*"])
        .pick_file()
        .map(|p| crate::picked_path(&p))
}

/// Apply an action to the config.
pub fn apply(cfg: &mut MachineConfig, action: ScsiAction) -> Option<String> {
    match action {
        ScsiAction::None => None,
        ScsiAction::AttachHdd { id, path } => {
            cfg.scsi.insert(id, ScsiDeviceConfig { path, ..Default::default() });
            Some(t!("scsi-attached-hdd", id = id))
        }
        ScsiAction::AttachEmptyCdrom { id } => {
            cfg.scsi.insert(id, ScsiDeviceConfig { cdrom: true, ..Default::default() });
            Some(t!("scsi-attached-cdrom-empty", id = id))
        }
        ScsiAction::AttachCdromWithDisc { id, path } => {
            cfg.scsi.insert(id, ScsiDeviceConfig {
                path: path.clone(), cdrom: true, ..Default::default()
            });
            Some(t!("scsi-attached-cdrom-disc", id = id))
        }
        ScsiAction::InsertDisc { id, path } => {
            if let Some(d) = cfg.scsi.get_mut(&id) { d.path = path; }
            Some(t!("scsi-disc-inserted", id = id))
        }
        ScsiAction::Eject { id } => {
            if let Some(d) = cfg.scsi.get_mut(&id) { d.path = String::new(); }
            Some(t!("scsi-disc-ejected", id = id))
        }
        ScsiAction::Detach { id } => {
            cfg.scsi.remove(&id);
            Some(t!("scsi-detached", id = id))
        }
        ScsiAction::CreateBlank { .. } => {
            // App opens the CreateDiskDialog; nothing to apply yet.
            None
        }
        ScsiAction::ToggleOverlay { id } => {
            let mut on = false;
            if let Some(d) = cfg.scsi.get_mut(&id) { d.overlay = !d.overlay; on = d.overlay; }
            Some(if on { t!("scsi-overlay-on", id = id) } else { t!("scsi-overlay-off", id = id) })
        }
        ScsiAction::RemountInIrix { .. } => None,
    }
}
