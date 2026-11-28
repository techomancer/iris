//! The stopped machine's summary page (plan, "The window" and the 2026-11-21
//! feedback): the name, a short spec line, a warning banner, the saved
//! states as a list, and the hardware rows. Each disk and CD-ROM row has a
//! browse button at its right edge.

use super::app::AppController;
use super::icons::Icon;
use super::profile::{traits, MachineIcon};
use super::session::{LightTarget, SavedState};
use super::Severity;
use crate::model::summary::{self, Drive, Network};

/// One row of the hardware list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareRow {
    pub icon: Icon,
    pub label: String,
    pub value: String,
    /// A short note at the right ("copy-on-write", "missing").
    pub note: String,
    /// What a click opens: the device's settings, or the browser for a disk.
    pub target: Option<LightTarget>,
    /// The row has a browse-files button.
    pub browsable: bool,
    /// The disk is a CHD whose compression makes every stop slow: the row
    /// has a convert button (`ids::item::CONVERT_DISK`).
    pub convertible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Overview {
    pub name: String,
    pub icon: MachineIcon,
    /// The machine and board, as "Indy (IP24)".
    pub spec: String,
    /// The run state when it is not "stopped": no word for a stopped machine.
    pub state: Option<String>,
    pub warnings: Vec<(Severity, String)>,
    pub states: Vec<SavedState>,
    pub rows: Vec<HardwareRow>,
    /// The machine's last screen, from when it last stopped.
    pub preview: Option<super::session::Picture>,
}

/// One line for a missing image: "SCSI 1 (disk): disks/a.raw", or that no
/// image was chosen when the path is empty.
pub(crate) fn missing_line(m: &summary::MissingDisk) -> String {
    let id = m.name.as_str();
    match (m.cdrom, m.path.is_empty()) {
        (true, false) => t!("modal-missing-cdrom", id = id, path = m.path.as_str()),
        (false, false) => t!("modal-missing-disk", id = id, path = m.path.as_str()),
        (true, true) => t!("modal-missing-cdrom-none", id = id),
        (false, true) => t!("modal-missing-disk-none", id = id),
    }
}

impl AppController {
    /// The summary of the selected machine, `None` with no machine.
    pub fn overview(&self) -> Option<Overview> {
        let name = self.selected()?.to_string();
        let cfg = self.config();
        let t = traits(cfg.machine.profile);
        let mut rows = vec![
            HardwareRow {
                icon: Icon::MachineSettings,
                label: t!("ov-processor"),
                value: {
                    let n = cfg.machine.cpus();
                    if n > 1 { format!("{n} x {}", super::profile::cpu_label(cfg.cpu())) } else { super::profile::cpu_label(cfg.cpu()) }
                },
                note: String::new(),
                target: None,
                browsable: false,
                convertible: false,
            },
            HardwareRow {
                icon: Icon::Info,
                label: t!("ov-memory"),
                value: crate::model::ram::ram_summary(&cfg.banks),
                note: String::new(),
                target: None,
                browsable: false,
                convertible: false,
            },
            HardwareRow {
                icon: Icon::DisplayScale,
                label: t!("ov-graphics"),
                value: if cfg.machine.profile == iris::config::MachineProfile::O2Ip32 {
                    // The O2's display is built in; the board key does not apply.
                    t!("extra-gbe")
                } else {
                    cfg.graphics_board().label().to_string()
                },
                note: String::new(),
                target: None,
                browsable: false,
                convertible: false,
            },
            HardwareRow {
                icon: Icon::Edit,
                label: t!("ov-prom"),
                value: if crate::model::hardware::builtin_prom(cfg) { t!("ov-prom-builtin") } else { cfg.prom.clone() },
                note: String::new(),
                target: None,
                browsable: false,
                convertible: false,
            },
        ];
        let dir = self.machine_dir();
        let missing = summary::missing_disks(cfg, |p| self.disk_readable(dir.as_deref(), p));
        for d in summary::drives(cfg) {
            let shown = |p: &str| {
                let file = summary::shown_path(p, dir.as_deref());
                if file.is_empty() { t!("ov-empty") } else { file }
            };
            let key_of = |id: &str| cfg.disks().into_iter().map(|d| d.0).find(|k| crate::model::hardware::scsi_name(*k) == id);
            let row = match &d {
                Drive::Disk { id, path } => {
                    let key = key_of(id);
                    let dev = key.and_then(|k| cfg.disk(k));
                    let note = if missing.iter().any(|m| m.name == *id) {
                        t!("ov-missing")
                    } else if dev.is_some_and(|d| d.overlay) {
                        t!("ov-cow")
                    } else {
                        String::new()
                    };
                    HardwareRow {
                        icon: Icon::Disk,
                        label: t!("ov-scsi", id = id.as_str()),
                        value: shown(path),
                        note,
                        target: key.map(|key| LightTarget::Disk { key }),
                        browsable: key.is_some(),
                        convertible: key.is_some_and(|k| self.suggest_convert(k)),
                    }
                }
                Drive::Cdrom { id, path } => {
                    let key = key_of(id);
                    HardwareRow {
                        icon: Icon::CdRom,
                        label: t!("ov-scsi", id = id.as_str()),
                        value: if path.is_empty() { t!("ov-no-disc") } else { t!("ov-cdrom", disc = shown(path)) },
                        note: if missing.iter().any(|m| m.name == *id) { t!("ov-missing") } else { String::new() },
                        target: key.map(|key| LightTarget::CdRom { key }),
                        browsable: key.is_some() && !path.is_empty(),
                        convertible: false,
                    }
                }
                Drive::Daynaport { id } => HardwareRow {
                    icon: Icon::Network,
                    label: t!("ov-scsi", id = id.as_str()),
                    value: t!("ov-daynaport"),
                    note: String::new(),
                    target: None,
                    browsable: false,
                    convertible: false,
                },
                Drive::Nvme { path } => HardwareRow {
                    icon: Icon::Disk,
                    label: "NVMe".into(),
                    value: shown(path),
                    note: String::new(),
                    target: None,
                    browsable: false,
                    convertible: false,
                },
                Drive::FibreChannel { id, path } => HardwareRow {
                    icon: Icon::Disk,
                    label: t!("ov-fc", id = *id),
                    value: shown(path),
                    note: String::new(),
                    target: None,
                    browsable: false,
                    convertible: false,
                },
            };
            rows.push(row);
        }
        rows.push(HardwareRow {
            icon: Icon::Network,
            label: t!("ov-network"),
            value: match summary::network(cfg, iris::build_features::PCAP) {
                Network::Nat(s) => t!("ov-net-nat", subnet = s.unwrap_or_else(|| "192.168.0.0/24".into())),
                Network::Bridged(i) => t!("ov-net-bridged", iface = i.unwrap_or_else(|| t!("common-auto"))),
                Network::Off => t!("settings-net-off"),
                Network::Stream(p) => t!("ov-net-stream", peer = p.unwrap_or_default()),
            },
            note: if cfg.port_forward.is_empty() { String::new() } else { t!("ov-forwards", count = cfg.port_forward.len()) },
            target: Some(LightTarget::Network),
            browsable: false,
            convertible: false,
        });
        let mut warnings = Vec::new();
        for m in &missing {
            warnings.push((Severity::Warning, missing_line(m)));
        }
        for r in &rows {
            if r.convertible {
                warnings.push((Severity::Info, t!("ov-convert-suggest", disk = r.label.as_str())));
            }
        }
        if let Err(e) = cfg.validate() {
            warnings.push((Severity::Error, t!("toast-invalid-config", error = e.to_string())));
        }
        let state = match self.session().run_state() {
            super::session::RunState::Stopped => None,
            _ => Some(self.status_bar().run_label),
        };
        Some(Overview {
            name,
            icon: t.icon,
            spec: t!("ov-spec", model = t.model, board = t.board),
            state,
            warnings,
            states: self.saved_states(),
            rows,
            preview: self.machine_dir().and_then(|d| super::session::Picture::at(d.join(crate::model::handle::LAST_SCREEN))),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_disk_with_no_image_says_so() {
        let m = |path: &str, cdrom| summary::MissingDisk { key: 3, name: "3".into(), path: path.into(), cdrom };
        assert!(missing_line(&m("disks/a.raw", false)).contains("disks/a.raw"));
        for cdrom in [false, true] {
            let line = missing_line(&m("", cdrom));
            assert!(!line.trim_end().ends_with(':'), "{line}");
        }
    }
}
