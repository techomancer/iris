//! Machine settings as data (D8): sections of fields, built from the core's
//! hardware tables and the [`profile`](super::profile) traits row, so a shell
//! implements each field kind once and a new machine needs no shell code.
//!
//! The schema covers the settings people change most (machine, memory,
//! display, disks, network). Every other key is reachable through the
//! complete-config editor (the Advanced section), which takes the whole TOML,
//! validates it and replaces the config.

use std::path::PathBuf;

use iris::net::nfsudp::NfsVersion;
use iris::config::{
    ForwardBind, ForwardProto, GraphicsBoard, MachineConfig, MachineProfile, NetMode, PortForwardConfig,
    ScsiDeviceConfig, ScsiKind, VinoSource, VinoStandard, VALID_BANK_SIZES,
};
use iris::dev::ng1::vc2_timings::NewportResolution;

use super::icons::Icon;
use super::profile::traits;
use super::{Answer, Effect, Enabled, PickSpec, Severity};
use crate::model::hardware;

/// Field ids (stable; also accessibility identifiers).
pub mod field {
    pub const PROFILE: &str = "settings.machine.profile";
    pub const CPU: &str = "settings.machine.cpu";
    pub const CPUS: &str = "settings.machine.cpus";
    pub const PROM: &str = "settings.machine.prom";
    pub const BUILTIN_PROM: &str = "settings.machine.builtin-prom";
    pub const NVRAM: &str = "settings.machine.nvram";
    pub const NVEEPROM: &str = "settings.machine.nveeprom";
    pub const BANKS: &str = "settings.memory.banks";
    /// The total, as a slider on the Machine page (`model::ram::layouts`).
    pub const MEMORY: &str = "settings.machine.memory";
    pub const BOARD: &str = "settings.display.board";
    pub const HEADS: &str = "settings.display.heads";
    pub const RESOLUTION: &str = "settings.display.resolution";
    pub const DISKS: &str = "settings.disks.table";
    pub const NET_MODE: &str = "settings.network.mode";
    pub const NAT_SUBNET: &str = "settings.network.nat-subnet";
    pub const PCAP_INTERFACE: &str = "settings.network.pcap-interface";
    pub const FORWARDS: &str = "settings.network.forwards";
    pub const NFS_ENABLE: &str = "settings.network.nfs";
    pub const NFS_DIR: &str = "settings.network.nfs-dir";
    pub const NFS_VERSION: &str = "settings.network.nfs-version";
    pub const NFS_PCAP_IP: &str = "settings.network.nfs-pcap-ip";
    pub const VINO_SOURCE: &str = "settings.video-in.source";
    pub const VINO_STANDARD: &str = "settings.video-in.standard";
    pub const VINO_CAMERA: &str = "settings.video-in.camera";
    pub const AFFINITY: &str = "settings.performance.affinity";
    pub const CPU_CORE: &str = "settings.performance.cpu-core";
    pub const REX3_CORE: &str = "settings.performance.rex3-core";
    pub const REFRESH_CORE: &str = "settings.performance.refresh-core";
    pub const JIT_THREADS_AUTO: &str = "settings.performance.jit-threads-auto";
    pub const JIT_THREADS: &str = "settings.performance.jit-threads";
    pub const JIT_CACHE: &str = "settings.performance.jit-cache";
    pub const JIT_CACHE_DIR: &str = "settings.performance.jit-cache-dir";
    pub const VNC_LISTEN: &str = "settings.performance.vnc-listen";
    pub const VNC_FPS: &str = "settings.performance.vnc-fps";
    pub const VNC_PASSWORD: &str = "settings.performance.vnc-password";
    pub const TOML: &str = "settings.advanced.toml";
    pub const NOTE: &str = "settings.note";
}

/// What a file field picks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PickKind {
    File { filters: Vec<(String, Vec<String>)> },
    Folder,
}

/// One SCSI target row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskRow {
    pub key: u8,
    /// "3" or "1:3".
    pub name: String,
    pub kind: ScsiKind,
    pub path: String,
    pub overlay: bool,
    pub scratch: bool,
    /// Missing or unreadable image, said in words.
    pub problem: Option<String>,
}

impl DiskRow {
    /// Whether the row offers copy-on-write: only for what the guest can
    /// write to (a CD-ROM is read-only).
    pub fn can_overlay(&self) -> bool {
        self.kind != ScsiKind::Cdrom
    }
}

/// The memory slider's stops: among `totals`, the powers of two and the
/// amounts half-way between them (32, 48, 64, 96, 128, 192, ... MB; a Tezro's
/// 1, 1.5, 2, 3, 4, 6, 8 GB), and `mb` itself when it is neither, so the knob
/// sits on the amount the machine has. All of `totals` when fewer than two
/// qualify.
pub fn memory_stops(totals: &[u32], mb: u32) -> Vec<u32> {
    let step = |t: u32| t.is_power_of_two() || (t % 3 == 0 && (t / 3).is_power_of_two());
    let mut stops: Vec<u32> = totals.iter().copied().filter(|t| step(*t)).collect();
    if stops.len() < 2 {
        return totals.to_vec();
    }
    if totals.contains(&mb) && !stops.contains(&mb) {
        stops.push(mb);
        stops.sort_unstable();
    }
    stops
}

/// A forward the Network page suggests: a common IRIX service, with a host
/// port that does not clash with the Mac's own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardPreset {
    pub label: String,
    pub tip: String,
    pub row: ForwardRow,
}

impl ForwardPreset {
    /// The forward as the config stores it.
    pub fn config(&self) -> PortForwardConfig {
        forward_config(&self.row)
    }
}

fn forward_config(row: &ForwardRow) -> PortForwardConfig {
    PortForwardConfig {
        proto: if row.tcp { ForwardProto::Tcp } else { ForwardProto::Udp },
        host_port: row.host_port,
        guest_port: row.guest_port,
        bind: if row.any_interface { ForwardBind::Any } else { ForwardBind::Localhost },
    }
}

/// The suggested forwards, in menu order: telnet, FTP, SSH, VNC, XDMCP.
pub fn forward_presets() -> Vec<ForwardPreset> {
    let p = |label: String, tip: String, tcp: bool, host_port: u16, guest_port: u16, any_interface: bool| ForwardPreset {
        label,
        tip,
        row: ForwardRow { tcp, host_port, guest_port, any_interface },
    };
    vec![
        p(t!("config-forward-telnet"), t!("config-forward-telnet-tip"), true, 2323, 23, false),
        p(t!("config-forward-ftp"), t!("config-forward-ftp-tip"), true, 2121, 21, false),
        p(t!("config-forward-ssh"), t!("config-forward-ssh-tip"), true, 2222, 22, false),
        p(t!("config-forward-vnc"), t!("config-forward-vnc-tip"), true, 5901, 5900, false),
        p(t!("config-forward-xdmcp"), t!("config-forward-xdmcp-tip"), false, 11177, 177, true),
    ]
}

/// The Add menu for a machine forwarding `rows`: each suggestion, offered
/// unless its guest port is forwarded already.
pub fn forward_menu(rows: &[ForwardRow]) -> Vec<(ForwardPreset, Enabled)> {
    forward_presets()
        .into_iter()
        .map(|p| {
            let taken = rows.iter().any(|r| r.guest_port == p.row.guest_port);
            (p, Enabled::when(!taken, || t!("config-forward-taken")))
        })
        .collect()
}

/// One port forward row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardRow {
    pub tcp: bool,
    pub host_port: u16,
    pub guest_port: u16,
    pub any_interface: bool,
}

impl FieldKind {
    /// A choice's selected option, as a summary line.
    pub fn choice_label(&self) -> Option<String> {
        match self {
            FieldKind::Choice { options, selected } => options.get(*selected).cloned(),
            FieldKind::GroupedChoice { groups, selected } => groups.iter().flat_map(|(_, v)| v).nth(*selected).cloned(),
            _ => None,
        }
    }
}

/// How a field is edited. A shell draws each kind once.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldKind {
    Choice { options: Vec<String>, selected: usize },
    /// A choice in titled groups; `selected` and the reported index count
    /// the options alone, group after group.
    GroupedChoice { groups: Vec<(String, Vec<String>)>, selected: usize },
    Toggle(bool),
    Text { value: String, placeholder: String, error: Option<String> },
    Number { value: f64, min: f64, max: f64, step: f64, unit: String },
    Path { value: String, pick: PickKind },
    /// An amount of memory: `mb` now, the totals the machine can hold (a
    /// typed amount snaps to one), and the slider's stops ([`memory_stops`]).
    Memory { mb: u32, totals: Vec<u32>, stops: Vec<u32> },
    /// Memory banks: each bank's size in MB, and the sizes a bank takes.
    BankGrid { banks: Vec<u32>, sizes: Vec<u32>, summary: String },
    DiskTable(Vec<DiskRow>),
    PortForwardTable(Vec<ForwardRow>),
    /// The whole config as TOML.
    Toml { text: String },
    Note(Severity, String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub id: &'static str,
    pub label: String,
    pub help: Option<String>,
    pub kind: FieldKind,
    pub enabled: Enabled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub id: &'static str,
    pub title: String,
    /// One line for the device list on the left of the sheet.
    pub summary: String,
    pub icon: Icon,
    pub fields: Vec<Field>,
}

/// A value for [`SettingsAction::Set`].
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Index(usize),
    Bool(bool),
    Text(String),
    Number(f64),
}

/// Disk table edits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiskEdit {
    /// Add a target of `kind` at the first free id on controller 0.
    Add(ScsiKind),
    Remove(u8),
    SetPath { key: u8, path: String },
    /// Ask for an image file.
    Pick(u8),
    SetKind { key: u8, kind: ScsiKind },
    SetOverlay { key: u8, on: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForwardEdit {
    /// A blank forward to fill in (Custom).
    Add,
    /// One of [`forward_presets`], by index.
    AddPreset(usize),
    Remove(usize),
    Set(usize, ForwardRow),
}

/// Settings intents. Each changes the selected machine's in-memory config;
/// the app marks it dirty and autosaves it.
#[derive(Debug, Clone, PartialEq)]
pub enum SettingsAction {
    Set { field: &'static str, value: Value },
    /// Ask for a file or folder for a path field.
    Pick(&'static str),
    /// Set memory bank `index` to `size` MB.
    Bank { index: usize, size: u32 },
    Disk(DiskEdit),
    Forward(ForwardEdit),
    /// Replace the whole config with this TOML, if it is valid.
    ApplyToml(String),
    Answer(Answer),
}

pub(crate) const PICK: &str = "settings.pick";

/// The parts of a table row a control can be.
pub mod part {
    pub const KIND: &str = "kind";
    pub const PATH: &str = "path";
    pub const PICK: &str = "pick";
    pub const OVERLAY: &str = "overlay";
    pub const REMOVE: &str = "remove";
    pub const TCP: &str = "tcp";
    pub const HOST: &str = "host";
    pub const GUEST: &str = "guest";
    pub const ANY: &str = "any";
    pub const ADD_DISK: &str = "add-disk";
    pub const ADD_CDROM: &str = "add-cdrom";
    pub const ADD: &str = "add";
    pub const BROWSE: &str = "browse";
    pub const APPLY: &str = "apply";
}

/// The id a shell gives a control: the field, then for a table the row and
/// the part (`settings.disks.table/3/path`), for a bank grid the bank.
pub fn control_id(field: &str, row: Option<usize>, part: Option<&str>) -> String {
    match (row, part) {
        (Some(r), Some(p)) => format!("{field}/{r}/{p}"),
        (None, Some(p)) => format!("{field}//{p}"),
        (Some(r), None) => format!("{field}/{r}"),
        (None, None) => field.to_string(),
    }
}

/// The disk kinds a disk row's popup offers, in order.
pub const DISK_KINDS: [ScsiKind; 3] = [ScsiKind::Disk, ScsiKind::Cdrom, ScsiKind::Daynaport];

/// The popup's labels for [`DISK_KINDS`], in the same order.
pub fn disk_kind_labels() -> Vec<String> {
    vec![t!("settings-kind-disk"), t!("settings-kind-cdrom"), t!("settings-kind-daynaport")]
}

/// The popup index of a disk row's kind.
pub fn disk_kind_index(row: &DiskRow) -> usize {
    DISK_KINDS.iter().position(|k| *k == row.kind).unwrap_or(0)
}

/// The settings action a control with `id` means, given its new `value`, and
/// for table rows the row data the sections listed (`rows` for disks: keys in
/// order; `forwards`: the current rows).
pub fn action_for(id: &str, value: Value, disk_keys: &[u8], forwards: &[ForwardRow]) -> Option<SettingsAction> {
    let mut it = id.splitn(3, '/');
    let f = it.next()?;
    let row: Option<usize> = it.next().and_then(|r| r.parse().ok());
    let p = it.next();
    let field = [
        field::PROFILE, field::CPU, field::CPUS, field::PROM, field::BUILTIN_PROM, field::NVRAM, field::NVEEPROM,
        field::BANKS, field::MEMORY, field::BOARD, field::HEADS, field::RESOLUTION, field::DISKS, field::NET_MODE, field::NAT_SUBNET,
        field::PCAP_INTERFACE, field::FORWARDS, field::TOML, field::VINO_SOURCE, field::VINO_STANDARD,
        field::VINO_CAMERA, field::AFFINITY, field::CPU_CORE, field::REX3_CORE, field::REFRESH_CORE,
        field::JIT_THREADS_AUTO, field::JIT_THREADS, field::JIT_CACHE, field::JIT_CACHE_DIR, field::VNC_LISTEN,
        field::VNC_FPS, field::VNC_PASSWORD, field::NFS_ENABLE, field::NFS_DIR, field::NFS_VERSION, field::NFS_PCAP_IP,
    ]
    .into_iter()
    .find(|x| *x == f)?;
    Some(match (field, row, p, value) {
        (field::BANKS, Some(i), None, Value::Number(mb)) => SettingsAction::Bank { index: i, size: mb as u32 },
        (field::DISKS, None, Some(part::ADD_DISK), _) => SettingsAction::Disk(DiskEdit::Add(ScsiKind::Disk)),
        (field::DISKS, None, Some(part::ADD_CDROM), _) => SettingsAction::Disk(DiskEdit::Add(ScsiKind::Cdrom)),
        (field::DISKS, Some(r), Some(p), v) => {
            let key = *disk_keys.get(r)?;
            SettingsAction::Disk(match (p, v) {
                (part::KIND, Value::Index(i)) => DiskEdit::SetKind { key, kind: *DISK_KINDS.get(i)? },
                (part::PATH, Value::Text(path)) => DiskEdit::SetPath { key, path },
                (part::PICK, _) => DiskEdit::Pick(key),
                (part::OVERLAY, Value::Bool(on)) => DiskEdit::SetOverlay { key, on },
                (part::REMOVE, _) => DiskEdit::Remove(key),
                _ => return None,
            })
        }
        // The Add menu: a suggestion by its index, or (past them) Custom.
        (field::FORWARDS, None, Some(part::ADD), Value::Index(i)) if i < forward_presets().len() => {
            SettingsAction::Forward(ForwardEdit::AddPreset(i))
        }
        (field::FORWARDS, None, Some(part::ADD), _) => SettingsAction::Forward(ForwardEdit::Add),
        (field::FORWARDS, Some(r), Some(part::REMOVE), _) => SettingsAction::Forward(ForwardEdit::Remove(r)),
        (field::FORWARDS, Some(r), Some(p), v) => {
            let mut row = forwards.get(r)?.clone();
            match (p, v) {
                (part::TCP, Value::Index(i)) => row.tcp = i == 0,
                (part::HOST, Value::Text(t)) => row.host_port = t.trim().parse().ok()?,
                (part::GUEST, Value::Text(t)) => row.guest_port = t.trim().parse().ok()?,
                (part::ANY, Value::Bool(b)) => row.any_interface = b,
                _ => return None,
            }
            SettingsAction::Forward(ForwardEdit::Set(r, row))
        }
        (field::TOML, None, Some(part::APPLY), Value::Text(t)) => SettingsAction::ApplyToml(t),
        (f, None, Some(part::BROWSE), _) => SettingsAction::Pick(f),
        (f, None, None, v) => SettingsAction::Set { field: f, value: v },
        _ => return None,
    })
}

/// Remembers which field a picker is for.
#[derive(Debug, Default)]
pub struct SettingsController {
    picking: Option<(&'static str, Option<u8>)>,
    toml_error: Option<String>,
    /// "Use built-in PROM" was switched off: show the PROM image field even
    /// while the config still resolves to the built-in PROM (no file chosen
    /// yet), so the checkbox does not snap back on.
    prom_custom: bool,
}

fn disk_filters() -> Vec<(String, Vec<String>)> {
    vec![(t!("settings-filter-disk-images"), ["img", "raw", "chd", "hda", "iso", "bin", "toast", "cdr"].map(String::from).to_vec())]
}

fn any_file() -> PickKind {
    PickKind::File { filters: Vec::new() }
}

impl SettingsController {
    /// The sections for `cfg`. `running` disables what only applies at power on.
    pub fn sections(&self, cfg: &MachineConfig, running: bool) -> Vec<Section> {
        self.sections_with(cfg, running, &|p| crate::model::machines::readable(p))
    }

    /// The sections, asking `readable` whether each disk image can be read
    /// (the app passes its non-blocking probe).
    pub fn sections_with(&self, cfg: &MachineConfig, running: bool, readable: &dyn Fn(&str) -> bool) -> Vec<Section> {
        let p = cfg.machine.profile;
        let t = traits(p);
        let at_power_on = || Enabled::when(!running, || t!("settings-next-power-on"));
        let mut out = Vec::new();

        // Machine.
        let mut f = vec![Field {
            id: field::PROFILE,
            label: t!("settings-machine-type"),
            help: None,
            kind: super::profile::profile_choice(p),
            enabled: at_power_on(),
        }];
        let cpus = p.cpus();
        f.push(Field {
            id: field::CPU,
            label: t!("config-cpu"),
            help: None,
            kind: FieldKind::Choice {
                options: cpus.iter().map(|c| super::profile::cpu_label(c)).collect(),
                selected: cpus.iter().position(|c| *c == cfg.cpu()).unwrap_or(0),
            },
            enabled: if cpus.len() < 2 { Enabled::No(t!("settings-one-cpu")) } else { at_power_on() },
        });
        if p.max_cpus() > 1 {
            f.push(Field {
                id: field::CPUS,
                label: t!("settings-cpu-count"),
                help: None,
                kind: FieldKind::Number {
                    value: cfg.machine.cpus() as f64,
                    min: 1.0,
                    max: p.max_cpus() as f64,
                    step: 1.0,
                    unit: String::new(),
                },
                enabled: at_power_on(),
            });
        }
        let builtin = hardware::builtin_prom(cfg) && !self.prom_custom;
        f.push(Field {
            id: field::BUILTIN_PROM,
            label: t!("config-prom-builtin"),
            help: None,
            kind: FieldKind::Toggle(builtin),
            enabled: at_power_on(),
        });
        if !builtin {
            f.push(Field {
                id: field::PROM,
                label: t!("config-prom"),
                help: None,
                kind: FieldKind::Path {
                    value: if iris::dev::prom_image::is_embedded_keyword(&cfg.prom) { String::new() } else { cfg.prom.clone() },
                    pick: any_file(),
                },
                enabled: at_power_on(),
            });
        }
        f.push(Field {
            id: field::NVRAM,
            label: t!("config-nvram"),
            help: None,
            kind: FieldKind::Path { value: cfg.nvram.clone(), pick: any_file() },
            enabled: at_power_on(),
        });
        if hardware::has_nveeprom(p) {
            f.push(Field {
                id: field::NVEEPROM,
                label: t!("config-nveeprom"),
                help: Some(t!("config-nveeprom-tip")),
                kind: FieldKind::Path { value: cfg.nveeprom.clone(), pick: any_file() },
                enabled: at_power_on(),
            });
        }
        // Memory: one total here, after the processors; the banks it fills
        // are under Advanced.
        let totals: Vec<u32> = crate::model::ram::layouts(cfg).into_iter().map(|(t, _)| t).collect();
        if !totals.is_empty() {
            let at = f.iter().rposition(|x| x.id == field::CPUS || x.id == field::CPU).map_or(f.len(), |i| i + 1);
            f.insert(at, Field {
                id: field::MEMORY,
                label: t!("settings-memory"),
                help: Some(t!("settings-memory-help")),
                kind: {
                    let mb = cfg.banks.iter().take(p.memory_banks()).sum();
                    FieldKind::Memory { mb, stops: memory_stops(&totals, mb), totals }
                },
                enabled: at_power_on(),
            });
        }
        out.push(Section {
            id: super::ids::settings::GENERAL,
            title: t!("settings-section-machine"),
            summary: format!("{}, {}", t.model, super::profile::cpu_label(cfg.cpu())),
            icon: Icon::MachineSettings,
            fields: f,
        });

        // Display.
        let mut f = Vec::new();
        let boards: Vec<GraphicsBoard> =
            GraphicsBoard::ALL.iter().copied().filter(|b| hardware::board_supported(*b, p)).collect();
        let board = cfg.graphics_board();
        if p == MachineProfile::O2Ip32 {
            f.push(note(Severity::Info, t!("settings-o2-display")));
        } else {
            f.push(Field {
                id: field::BOARD,
                label: t!("config-graphics-board"),
                help: None,
                kind: FieldKind::Choice {
                    options: boards.iter().map(|b| b.label().to_string()).collect(),
                    selected: boards.iter().position(|b| *b == board).unwrap_or(0),
                },
                enabled: if boards.len() < 2 { Enabled::No(t!("settings-one-board")) } else { at_power_on() },
            });
        }
        let newport = hardware::mc_machine(p) && board == GraphicsBoard::Newport;
        if newport {
            f.push(Field {
                id: field::RESOLUTION,
                label: t!("config-resolution"),
                help: None,
                kind: FieldKind::Choice {
                    options: NewportResolution::ALL.iter().map(|r| r.label().to_string()).collect(),
                    selected: NewportResolution::ALL.iter().position(|r| *r == cfg.graphics.resolution).unwrap_or(0),
                },
                enabled: at_power_on(),
            });
            if matches!(p, MachineProfile::Indigo2Ip22) {
                f.push(Field {
                    id: field::HEADS,
                    label: t!("settings-heads"),
                    help: None,
                    kind: FieldKind::Choice { options: vec!["1".into(), "2".into()], selected: usize::from(cfg.has_device("newport")) },
                    enabled: at_power_on(),
                });
            }
        }
        out.push(Section {
            id: super::ids::settings::DISPLAY,
            title: t!("settings-section-display"),
            summary: if p == MachineProfile::O2Ip32 { "GBE".into() } else { board.label().to_string() },
            icon: Icon::DisplayScale,
            fields: f,
        });

        // Disks.
        if !p.is_ip35_bricks() {
            // By controller and ID, so rows (and the ids that name them)
            // stay put between edits.
            let rows: Vec<DiskRow> = hardware::scsi_disks(cfg)
                .into_iter()
                .map(|(key, d)| DiskRow {
                    key,
                    name: hardware::scsi_name(key),
                    kind: d.kind(),
                    path: d.path.clone(),
                    overlay: d.overlay,
                    scratch: d.scratch,
                    problem: (d.kind() == ScsiKind::Disk && !d.scratch && !readable(&d.path))
                        .then(|| t!("settings-disk-missing")),
                })
                .collect();
            let n = rows.iter().filter(|r| r.kind == ScsiKind::Disk).count();
            out.push(Section {
                id: super::ids::settings::DISKS,
                title: t!("settings-section-disks"),
                summary: t!("settings-disks-summary", count = n),
                icon: Icon::Disk,
                fields: vec![Field {
                    id: field::DISKS,
                    label: t!("config-scsi"),
                    help: None,
                    kind: FieldKind::DiskTable(rows),
                    enabled: Enabled::Yes,
                }],
            });
        }

        // Network.
        let modes = net_modes();
        let mode = cfg.network.mode;
        let mut f = vec![Field {
            id: field::NET_MODE,
            label: t!("config-net-mode"),
            help: None,
            kind: FieldKind::Choice {
                options: modes.iter().map(|m| net_mode_label(*m)).collect(),
                selected: modes.iter().position(|m| *m == mode).unwrap_or(0),
            },
            enabled: at_power_on(),
        }];
        match mode {
            NetMode::Nat => {
                let value = cfg.nat_subnet.clone().unwrap_or_default();
                let error = (!value.is_empty())
                    .then(|| iris::config::parse_nat_subnet(&value).err().map(|e| e.to_string()))
                    .flatten();
                f.push(Field {
                    id: field::NAT_SUBNET,
                    label: t!("config-net-network"),
                    help: None,
                    kind: FieldKind::Text { value, placeholder: "192.168.0.0/24".into(), error },
                    enabled: Enabled::Yes,
                });
                f.push(Field {
                    id: field::FORWARDS,
                    label: t!("config-forwards"),
                    help: None,
                    kind: FieldKind::PortForwardTable(
                        cfg.port_forward
                            .iter()
                            .map(|r| ForwardRow {
                                tcp: r.proto == ForwardProto::Tcp,
                                host_port: r.host_port,
                                guest_port: r.guest_port,
                                any_interface: r.bind == ForwardBind::Any,
                            })
                            .collect(),
                    ),
                    enabled: Enabled::Yes,
                });
            }
            NetMode::Pcap => {
                f.push(Field {
                    id: field::PCAP_INTERFACE,
                    label: t!("settings-pcap-interface"),
                    help: Some(t!("settings-pcap-interface-help")),
                    kind: FieldKind::Text {
                        value: cfg.network.pcap_interface.clone().unwrap_or_default(),
                        placeholder: t!("common-auto"),
                        error: None,
                    },
                    enabled: Enabled::Yes,
                });
                if !iris::build_features::PCAP {
                    f.push(note(Severity::Warning, t!("settings-pcap-not-built")));
                }
            }
            NetMode::Off | NetMode::Stream => {}
        }
        // The NFS share: a host folder IRIX mounts over the network.
        if mode != NetMode::Off {
            f.push(Field {
                id: field::NFS_ENABLE,
                label: t!("config-nfs-enable"),
                help: None,
                kind: FieldKind::Toggle(cfg.nfs.is_some()),
                enabled: at_power_on(),
            });
            if let Some(nfs) = &cfg.nfs {
                f.push(Field {
                    id: field::NFS_DIR,
                    label: t!("config-nfs-dir"),
                    help: None,
                    kind: FieldKind::Path { value: nfs.shared_dir.clone(), pick: PickKind::Folder },
                    enabled: at_power_on(),
                });
                f.push(Field {
                    id: field::NFS_VERSION,
                    label: t!("config-nfs-version"),
                    help: None,
                    kind: FieldKind::Choice {
                        options: vec![t!("common-auto"), t!("config-nfs-v2"), t!("config-nfs-v3")],
                        selected: NFS_VERSIONS.iter().position(|v| *v == nfs.version).unwrap_or(0),
                    },
                    enabled: at_power_on(),
                });
                if mode == NetMode::Pcap {
                    let value = cfg.network.nfs_pcap_ip.map(|i| i.to_string()).unwrap_or_default();
                    f.push(Field {
                        id: field::NFS_PCAP_IP,
                        label: t!("config-nfs-pcap-ip"),
                        help: None,
                        kind: FieldKind::Text { value, placeholder: t!("config-nfs-pcap-ip-hint"), error: None },
                        enabled: at_power_on(),
                    });
                }
            }
        }
        out.push(Section {
            id: super::ids::settings::NETWORK,
            title: t!("settings-section-network"),
            summary: net_mode_label(mode),
            icon: Icon::Network,
            fields: f,
        });

        // Video in: the Indy's VINO input.
        if p == MachineProfile::IndyIp24 {
            let src = VINO_SOURCES.iter().position(|v| *v == cfg.vino.source).unwrap_or(0);
            let mut f = vec![
                Field {
                    id: field::VINO_SOURCE,
                    label: t!("config-vino-source"),
                    help: None,
                    kind: FieldKind::Choice {
                        options: vec![
                            t!("config-vino-source-off"),
                            t!("config-vino-source-test"),
                            t!("config-vino-source-camera"),
                            t!("config-vino-source-black"),
                        ],
                        selected: src,
                    },
                    enabled: at_power_on(),
                },
                Field {
                    id: field::VINO_STANDARD,
                    label: t!("config-vino-standard"),
                    help: None,
                    kind: FieldKind::Choice {
                        options: vec![t!("config-vino-ntsc"), t!("config-vino-pal")],
                        selected: usize::from(cfg.vino.standard == VinoStandard::Pal),
                    },
                    enabled: at_power_on(),
                },
            ];
            if cfg.vino.source == VinoSource::Camera {
                f.push(Field {
                    id: field::VINO_CAMERA,
                    label: t!("config-vino-camera"),
                    help: None,
                    kind: FieldKind::Number { value: cfg.vino.camera_index as f64, min: 0.0, max: 15.0, step: 1.0, unit: String::new() },
                    enabled: at_power_on(),
                });
            }
            out.push(Section {
                id: "settings.video-in",
                title: t!("config-tab-video-in"),
                summary: f[0].kind.choice_label().unwrap_or_default(),
                icon: Icon::Camera,
                fields: f,
            });
        }

        // Performance: host threads, the JIT, the VNC server.
        let cores = host_cores() as f64;
        let pinned = Enabled::when(cfg.perf.thread_affinity, || t!("settings-pin-first"));
        let core = |id, label: String, value: Option<u32>, default: u32| Field {
            id,
            label,
            help: None,
            kind: FieldKind::Number { value: value.unwrap_or(default) as f64, min: 0.0, max: cores - 1.0, step: 1.0, unit: String::new() },
            enabled: pinned.clone(),
        };
        let mut f = vec![
            Field {
                id: field::AFFINITY,
                label: t!("config-perf-affinity"),
                help: Some(t!("config-perf-affinity-tip")),
                kind: FieldKind::Toggle(cfg.perf.thread_affinity),
                enabled: at_power_on(),
            },
            core(field::CPU_CORE, t!("config-perf-cpu-core"), cfg.perf.cpu_core, 0),
            core(field::REX3_CORE, t!("config-perf-rex3-core"), cfg.perf.rex3_core, 1),
            core(field::REFRESH_CORE, t!("config-perf-refresh-core"), cfg.perf.refresh_core, 2),
        ];
        if iris::build_features::JITV2 {
            let auto = !cfg.jitv2_threads_set;
            f.push(Field {
                id: field::JIT_THREADS_AUTO,
                label: t!("config-jit-threads-auto"),
                help: Some(t!("config-jit-threads-auto-tip")),
                kind: FieldKind::Toggle(auto),
                enabled: at_power_on(),
            });
            f.push(Field {
                id: field::JIT_THREADS,
                label: t!("config-jit-threads"),
                help: None,
                kind: FieldKind::Number { value: cfg.jitv2.threads as f64, min: 1.0, max: cores, step: 1.0, unit: String::new() },
                enabled: Enabled::when(!auto, || t!("config-jit-threads-auto-tip")),
            });
            f.push(Field {
                id: field::JIT_CACHE,
                label: t!("config-jit-cache"),
                help: None,
                kind: FieldKind::Toggle(cfg.jitv2.cache),
                enabled: at_power_on(),
            });
            f.push(Field {
                id: field::JIT_CACHE_DIR,
                label: t!("config-jit-cache-dir"),
                help: None,
                kind: FieldKind::Text { value: cfg.jitv2.cache_dir.clone(), placeholder: t!("config-jit-cache-dir-hint"), error: None },
                enabled: Enabled::when(cfg.jitv2.cache, || t!("config-jit-cache")),
            });
        }
        f.push(Field {
            id: field::VNC_LISTEN,
            label: t!("config-vnc-listen"),
            help: Some(t!("config-vnc-listen-tip")),
            kind: FieldKind::Text { value: cfg.vnc.listen.clone().unwrap_or_default(), placeholder: t!("config-vnc-listen-hint"), error: None },
            enabled: at_power_on(),
        });
        f.push(Field {
            id: field::VNC_FPS,
            label: t!("config-vnc-max-fps"),
            help: None,
            kind: FieldKind::Number { value: cfg.vnc.max_fps.unwrap_or(30) as f64, min: 1.0, max: 120.0, step: 1.0, unit: String::new() },
            enabled: at_power_on(),
        });
        f.push(Field {
            id: field::VNC_PASSWORD,
            label: t!("config-vnc-password-file"),
            help: Some(t!("config-vnc-password-file-tip")),
            kind: FieldKind::Path { value: cfg.vnc.password_file.clone().unwrap_or_default(), pick: any_file() },
            enabled: at_power_on(),
        });
        out.push(Section {
            id: "settings.performance",
            title: t!("config-tab-performance"),
            summary: t!("settings-performance-summary"),
            icon: Icon::Check,
            fields: f,
        });

        // Advanced: every key, as TOML.
        out.push(Section {
            id: "settings.advanced",
            title: t!("settings-section-advanced"),
            summary: t!("settings-advanced-summary"),
            icon: Icon::Edit,
            fields: vec![
                Field {
                    id: field::BANKS,
                    label: t!("settings-memory-banks"),
                    // The popups list only the sizes this machine takes; Octane
                    // and O2 also have rules about which sizes go together.
                    help: matches!(p, MachineProfile::OctaneIp30 | MachineProfile::O2Ip32).then(|| crate::model::ram::bank_size_tip(p)),
                    kind: FieldKind::BankGrid {
                        banks: cfg.banks.iter().take(p.memory_banks()).copied().collect(),
                        sizes: VALID_BANK_SIZES.iter().copied().filter(|s| *s == 0 || p.bank_sizes().contains(s)).collect(),
                        summary: crate::model::ram::ram_summary(&cfg.banks),
                    },
                    enabled: at_power_on(),
                },
                Field {
                    id: field::TOML,
                    label: t!("settings-toml"),
                    help: Some(t!("settings-toml-help")),
                    kind: FieldKind::Toml { text: crate::model::machines::to_toml(cfg).unwrap_or_default() },
                    enabled: at_power_on(),
                },
            ]
            .into_iter()
            .chain(self.toml_error.clone().map(|e| note(Severity::Error, e)))
            .collect(),
        });
        out
    }

    /// Apply `action` to `cfg`. Returns whether the config changed, and the
    /// effects to carry out.
    pub fn apply(&mut self, action: SettingsAction, cfg: &mut MachineConfig, dir: Option<PathBuf>) -> (bool, Vec<Effect>) {
        let mut fx = Vec::new();
        let changed = match action {
            SettingsAction::Set { field: field::BUILTIN_PROM, value: Value::Bool(on) } => {
                self.prom_custom = !on;
                set(cfg, field::BUILTIN_PROM, Value::Bool(on));
                true
            }
            SettingsAction::Set { field, value } => set(cfg, field, value),
            SettingsAction::Pick(field) => {
                let title = t!("common-browse");
                let spec = PickSpec::open(PICK, title).in_dir(dir);
                let effect = match field {
                    field::PROM | field::NVRAM | field::NVEEPROM | field::VNC_PASSWORD => Effect::PickFile(spec),
                    field::NFS_DIR => Effect::PickFolder(spec),
                    _ => return (false, fx),
                };
                self.picking = Some((field, None));
                fx.push(effect);
                false
            }
            SettingsAction::Bank { index, size } => {
                // Only a size this machine's banks take (0: empty).
                let fits = size == 0 || cfg.machine.profile.bank_sizes().contains(&size);
                if fits && index < cfg.machine.profile.memory_banks() && cfg.banks[index] != size {
                    cfg.banks[index] = size;
                    cfg.banks_set = true;
                    true
                } else {
                    false
                }
            }
            SettingsAction::Disk(edit) => match edit {
                DiskEdit::Add(kind) => {
                    let p = cfg.machine.profile;
                    // ID 0 is the controller itself; targets take 1 up to the bus's last id.
                    let free = hardware::scsi_ids(p, 0).into_iter().find(|id| hardware::scsi_key(cfg, 0, *id).is_none());
                    match free {
                        Some(id) => {
                            let mut dev = ScsiDeviceConfig { kind_field: kind, cdrom: kind == ScsiKind::Cdrom, ..Default::default() };
                            if kind == ScsiKind::Disk {
                                dev.path = crate::model::settings::GuiSettings::default_disk_path(id);
                            }
                            cfg.set_disk(id, dev);
                            true
                        }
                        None => {
                            fx.push(Effect::Toast(t!("settings-no-free-id")));
                            false
                        }
                    }
                }
                DiskEdit::Remove(key) => {
                    let had = cfg.disk(key).is_some();
                    cfg.remove_disk(key);
                    had
                }
                DiskEdit::SetPath { key, path } => match cfg.disk_mut(key) {
                    Some(d) if d.path != path => {
                        d.path = path;
                        true
                    }
                    _ => false,
                },
                DiskEdit::Pick(key) => {
                    self.picking = Some((field::DISKS, Some(key)));
                    fx.push(Effect::PickFile(
                        PickSpec { filters: disk_filters(), ..PickSpec::open(PICK, t!("common-browse")) }
                            .in_dir(dir.map(|d| d.join("disks"))),
                    ));
                    false
                }
                DiskEdit::SetKind { key, kind } => match cfg.disk_mut(key) {
                    Some(d) if d.kind() != kind => {
                        d.kind_field = kind;
                        d.cdrom = kind == ScsiKind::Cdrom;
                        // A CD-ROM is read-only: no copy-on-write.
                        if d.cdrom {
                            d.overlay = false;
                        }
                        true
                    }
                    _ => false,
                },
                DiskEdit::SetOverlay { key, on } => match cfg.disk_mut(key) {
                    Some(d) if d.overlay != on => {
                        d.overlay = on;
                        true
                    }
                    _ => false,
                },
            },
            SettingsAction::Forward(edit) => {
                match edit {
                    ForwardEdit::Add => {
                        let host = (2222u16..).find(|p| !cfg.port_forward.iter().any(|r| r.host_port == *p)).unwrap_or(2222);
                        cfg.port_forward.push(PortForwardConfig {
                            proto: ForwardProto::Tcp,
                            host_port: host,
                            guest_port: 22,
                            bind: ForwardBind::default(),
                        });
                    }
                    ForwardEdit::AddPreset(i) => {
                        let Some(p) = forward_presets().into_iter().nth(i) else { return (false, fx) };
                        if cfg.port_forward.iter().any(|r| r.guest_port == p.row.guest_port) {
                            return (false, fx);
                        }
                        cfg.port_forward.push(p.config());
                    }
                    ForwardEdit::Remove(i) => {
                        if i < cfg.port_forward.len() {
                            cfg.port_forward.remove(i);
                        }
                    }
                    ForwardEdit::Set(i, row) => {
                        // Real ports, and a host port no other forward of the
                        // same protocol already listens on.
                        if row.host_port == 0 || row.guest_port == 0 {
                            fx.push(Effect::Toast(t!("config-forward-port-zero")));
                            return (false, fx);
                        }
                        let proto = forward_config(&row).proto;
                        if cfg.port_forward.iter().enumerate().any(|(j, r)| j != i && r.host_port == row.host_port && r.proto == proto) {
                            fx.push(Effect::Toast(t!("config-forward-host-taken", port = row.host_port)));
                            return (false, fx);
                        }
                        if let Some(r) = cfg.port_forward.get_mut(i) {
                            *r = forward_config(&row);
                        }
                    }
                }
                true
            }
            SettingsAction::ApplyToml(text) => match hardware::parse_complete(&text) {
                Ok(new) => {
                    self.toml_error = None;
                    *cfg = new;
                    true
                }
                Err(e) => {
                    self.toml_error = Some(e);
                    false
                }
            },
            SettingsAction::Answer(Answer::Picked { purpose: PICK, paths }) => {
                let (Some((f, key)), Some(p)) = (self.picking.take(), paths.into_iter().next()) else {
                    return (false, fx);
                };
                let path = p.to_string_lossy().into_owned();
                match (f, key) {
                    (field::DISKS, Some(key)) => {
                        return self.apply(SettingsAction::Disk(DiskEdit::SetPath { key, path }), cfg, None)
                    }
                    (f, _) => set(cfg, f, Value::Text(path)),
                }
            }
            SettingsAction::Answer(_) => false,
        };
        (changed, fx)
    }

    /// The live command a change needs on a running machine, if any: the NAT
    /// subnet and port forwards change without a reboot.
    pub fn live_change(field: &'static str) -> bool {
        matches!(field, field::NAT_SUBNET | field::FORWARDS | field::PCAP_INTERFACE)
    }
}

fn note(severity: Severity, text: String) -> Field {
    Field { id: field::NOTE, label: String::new(), help: None, kind: FieldKind::Note(severity, text), enabled: Enabled::Yes }
}

fn net_modes() -> Vec<NetMode> {
    vec![NetMode::Nat, NetMode::Pcap, NetMode::Off]
}

fn net_mode_label(m: NetMode) -> String {
    match m {
        NetMode::Nat => t!("config-net-mode-nat"),
        NetMode::Pcap => t!("config-net-mode-pcap"),
        NetMode::Off => t!("settings-net-off"),
        NetMode::Stream => t!("status-net-stream"),
    }
}

/// The NFS version popup's order.
const NFS_VERSIONS: [NfsVersion; 3] = [NfsVersion::Auto, NfsVersion::V2, NfsVersion::V3];

/// The Video In source popup's order.
const VINO_SOURCES: [VinoSource; 4] = [VinoSource::Off, VinoSource::TestPattern, VinoSource::Camera, VinoSource::Black];

/// Set `slot` to `v`; whether it changed.
fn set_num<T: PartialEq>(slot: &mut T, v: T) -> bool {
    if *slot == v {
        return false;
    }
    *slot = v;
    true
}

fn set(cfg: &mut MachineConfig, field: &'static str, value: Value) -> bool {
    let p = cfg.machine.profile;
    match (field, value) {
        (field::PROFILE, Value::Index(i)) => {
            let Some(&new) = super::profile::menu_order().get(i) else { return false };
            if new == p {
                return false;
            }
            cfg.machine.profile = new;
            // Refit only on an explicit change, never per frame.
            hardware::switch_profile(cfg, p);
            true
        }
        (field::CPU, Value::Index(i)) => match p.cpus().get(i) {
            Some(&c) if c != cfg.cpu() => {
                cfg.set_cpu(c);
                true
            }
            _ => false,
        },
        (field::MEMORY, Value::Number(mb)) => {
            // The nearest total the machine holds, in its best bank layout.
            let Some((_, layout)) = crate::model::ram::layout_near(cfg, mb.max(0.0) as u32) else { return false };
            let mut banks = cfg.banks.clone();
            banks.iter_mut().for_each(|b| *b = 0);
            for (i, b) in layout.iter().enumerate() {
                if let Some(slot) = banks.get_mut(i) {
                    *slot = *b;
                }
            }
            if banks == cfg.banks {
                return false;
            }
            cfg.banks = banks;
            cfg.banks_set = true;
            true
        }
        (field::CPUS, Value::Number(n)) => {
            let n = (n.round() as u8).clamp(1, p.max_cpus());
            if p.accepts_cpus(n).is_err() || cfg.machine.cpus() == n {
                return false;
            }
            cfg.machine.cpus = Some(n);
            true
        }
        (field::BUILTIN_PROM, Value::Bool(on)) => {
            if on {
                cfg.prom = "(embedded)".into();
            } else if hardware::builtin_prom(cfg) {
                cfg.prom = iris::config::DEFAULT_PROM.into();
            }
            true
        }
        (field::NFS_ENABLE, Value::Bool(on)) => {
            if on == cfg.nfs.is_some() {
                return false;
            }
            cfg.nfs = on.then(|| iris::config::NfsConfig { shared_dir: String::new(), version: NfsVersion::Auto });
            true
        }
        (field::NFS_DIR, Value::Text(t)) => match cfg.nfs.as_mut() {
            Some(n) => replace(&mut n.shared_dir, t.trim().to_string()),
            None => false,
        },
        (field::NFS_VERSION, Value::Index(i)) => match (cfg.nfs.as_mut(), NFS_VERSIONS.get(i)) {
            (Some(n), Some(v)) => set_num(&mut n.version, *v),
            _ => false,
        },
        (field::NFS_PCAP_IP, Value::Text(t)) => match t.trim() {
            "" => set_num(&mut cfg.network.nfs_pcap_ip, None),
            t => match t.parse() {
                Ok(ip) => set_num(&mut cfg.network.nfs_pcap_ip, Some(ip)),
                Err(_) => false,
            },
        },
        (field::VINO_SOURCE, Value::Index(i)) => match VINO_SOURCES.get(i) {
            Some(v) => set_num(&mut cfg.vino.source, v.clone()),
            None => false,
        },
        (field::VINO_STANDARD, Value::Index(i)) => {
            let v = if i == 1 { VinoStandard::Pal } else { VinoStandard::Ntsc };
            let changed = v != cfg.vino.standard;
            cfg.vino.standard = v;
            changed
        }
        (field::VINO_CAMERA, Value::Number(n)) => set_num(&mut cfg.vino.camera_index, n.round().clamp(0.0, 15.0) as _),
        (field::AFFINITY, Value::Bool(b)) => set_num(&mut cfg.perf.thread_affinity, b),
        // A core this host has: 0 to its count less one.
        (field::CPU_CORE, Value::Number(n)) => set_num(&mut cfg.perf.cpu_core, Some(n.round().clamp(0.0, host_cores() as f64 - 1.0) as u32)),
        (field::REX3_CORE, Value::Number(n)) => set_num(&mut cfg.perf.rex3_core, Some(n.round().clamp(0.0, host_cores() as f64 - 1.0) as u32)),
        (field::REFRESH_CORE, Value::Number(n)) => set_num(&mut cfg.perf.refresh_core, Some(n.round().clamp(0.0, host_cores() as f64 - 1.0) as u32)),
        (field::JIT_THREADS_AUTO, Value::Bool(auto)) => {
            if cfg.jitv2_threads_set == !auto {
                return false;
            }
            cfg.jitv2_threads_set = !auto;
            if auto {
                cfg.settle_jitv2_threads();
            }
            true
        }
        (field::JIT_THREADS, Value::Number(n)) => {
            cfg.jitv2_threads_set = true;
            set_num(&mut cfg.jitv2.threads, (n.round().clamp(1.0, host_cores() as f64)) as _)
        }
        (field::JIT_CACHE, Value::Bool(b)) => set_num(&mut cfg.jitv2.cache, b),
        (field::JIT_CACHE_DIR, Value::Text(t)) => replace(&mut cfg.jitv2.cache_dir, t.trim().to_string()),
        (field::VNC_LISTEN, Value::Text(t)) => {
            let t = t.trim();
            set_num(&mut cfg.vnc.listen, (!t.is_empty()).then(|| t.to_string()))
        }
        (field::VNC_FPS, Value::Number(n)) => set_num(&mut cfg.vnc.max_fps, Some(n.round().clamp(1.0, 120.0) as u32)),
        (field::VNC_PASSWORD, Value::Text(t)) => {
            let t = t.trim();
            set_num(&mut cfg.vnc.password_file, (!t.is_empty()).then(|| t.to_string()))
        }
        (field::PROM, Value::Text(s)) => replace(&mut cfg.prom, s),
        (field::NVRAM, Value::Text(s)) => replace(&mut cfg.nvram, s),
        (field::NVEEPROM, Value::Text(s)) => replace(&mut cfg.nveeprom, s),
        (field::BOARD, Value::Index(i)) => {
            let boards: Vec<GraphicsBoard> =
                GraphicsBoard::ALL.iter().copied().filter(|b| hardware::board_supported(*b, p)).collect();
            match boards.get(i) {
                Some(&b) if b != cfg.graphics_board() => {
                    cfg.graphics.board = Some(b);
                    true
                }
                _ => false,
            }
        }
        (field::RESOLUTION, Value::Index(i)) => match NewportResolution::ALL.get(i) {
            Some(&r) if r != cfg.graphics.resolution => {
                cfg.graphics.resolution = r;
                true
            }
            _ => false,
        },
        (field::HEADS, Value::Index(i)) => {
            // The second head is a Newport card in a GIO slot.
            let dual = i >= 1;
            if dual == cfg.has_device("newport") {
                return false;
            }
            cfg.set_device("newport", dual);
            true
        }
        (field::NET_MODE, Value::Index(i)) => match net_modes().get(i) {
            Some(&m) if m != cfg.network.mode => {
                cfg.network.mode = m;
                true
            }
            _ => false,
        },
        (field::NAT_SUBNET, Value::Text(s)) => {
            let v = (!s.trim().is_empty()).then(|| s.trim().to_string());
            if v == cfg.nat_subnet {
                return false;
            }
            cfg.nat_subnet = v;
            true
        }
        (field::PCAP_INTERFACE, Value::Text(s)) => {
            let v = (!s.trim().is_empty()).then(|| s.trim().to_string());
            if v == cfg.network.pcap_interface {
                return false;
            }
            cfg.network.pcap_interface = v;
            true
        }
        // A value of the wrong kind for the field, or a field with no setter.
        _ => false,
    }
}

/// The host's processor count (8 when it cannot be read).
fn host_cores() -> usize {
    std::thread::available_parallelism().map_or(8, |n| n.get()).max(1)
}

fn replace(slot: &mut String, s: String) -> bool {
    if *slot == s {
        false
    } else {
        *slot = s;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section<'a>(s: &'a [Section], id: &str) -> &'a Section {
        s.iter().find(|x| x.id == id).unwrap()
    }

    #[test]
    fn every_profile_builds_a_schema_whose_choices_round_trip() {
        let c = SettingsController::default();
        for p in MachineProfile::ALL {
            let mut cfg = MachineConfig::default();
            let from = cfg.machine.profile;
            cfg.machine.profile = p;
            hardware::switch_profile(&mut cfg, from);
            let sections = c.sections(&cfg, false);
            for s in &sections {
                for f in &s.fields {
                    if let FieldKind::Choice { selected, options } = &f.kind {
                        assert!(*selected < options.len(), "{p:?} {}", f.id);
                        // Choosing what is selected changes nothing.
                        let mut again = cfg.clone();
                        assert!(!set(&mut again, f.id, Value::Index(*selected)), "{p:?} {}", f.id);
                    }
                }
            }
            assert_eq!(p.is_ip35_bricks(), !sections.iter().any(|s| s.id == super::super::ids::settings::DISKS));
        }
    }

    #[test]
    fn switching_profile_refits_and_validates() {
        let mut c = SettingsController::default();
        let mut cfg = MachineConfig::default();
        let octane = super::super::profile::menu_order().iter().position(|p| *p == MachineProfile::OctaneIp30).unwrap();
        let (changed, _) = c.apply(SettingsAction::Set { field: field::PROFILE, value: Value::Index(octane) }, &mut cfg, None);
        assert!(changed);
        assert_eq!(cfg.machine.profile, MachineProfile::OctaneIp30);
        cfg.validate().unwrap();
    }

    #[test]
    fn forwards_and_banks_take_only_what_works() {
        let mut c = SettingsController::default();
        let mut cfg = MachineConfig::default();
        cfg.port_forward.clear();
        let row = |host_port, guest_port| ForwardRow { tcp: true, host_port, guest_port, any_interface: false };
        for r in [row(2323, 23), row(2222, 22)] {
            cfg.port_forward.push(forward_config(&r));
        }
        let set = |c: &mut SettingsController, cfg: &mut MachineConfig, i, r| c.apply(SettingsAction::Forward(ForwardEdit::Set(i, r)), cfg, None).0;
        assert!(!set(&mut c, &mut cfg, 1, row(0, 22)), "port 0");
        assert!(!set(&mut c, &mut cfg, 1, row(2323, 22)), "a host port the first forward has");
        assert!(set(&mut c, &mut cfg, 1, row(2224, 22)));
        assert_eq!(cfg.port_forward[1].host_port, 2224);
        // A bank takes only the sizes the machine's banks do.
        let bank = |c: &mut SettingsController, cfg: &mut MachineConfig, size| c.apply(SettingsAction::Bank { index: 0, size }, cfg, None).0;
        assert!(!bank(&mut c, &mut cfg, 100), "no 100 MB bank");
        let other = *cfg.machine.profile.bank_sizes().iter().find(|s| **s != 0 && **s != cfg.banks[0]).unwrap();
        assert!(bank(&mut c, &mut cfg, other));
    }

    #[test]
    fn the_add_menu_suggests_services_not_forwarded_yet() {
        let mut c = SettingsController::default();
        let mut cfg = MachineConfig::default();
        cfg.port_forward.clear();
        let n = forward_presets().len();
        let add = |i| action_for(&control_id(field::FORWARDS, None, Some(part::ADD)), Value::Index(i), &[], &[]);
        assert_eq!(add(0), Some(SettingsAction::Forward(ForwardEdit::AddPreset(0))));
        assert_eq!(add(n), Some(SettingsAction::Forward(ForwardEdit::Add)), "past the suggestions: Custom");
        // Telnet: host 2323 to guest 23, this computer only.
        assert!(c.apply(add(0).unwrap(), &mut cfg, None).0);
        let pf = &cfg.port_forward[0];
        assert_eq!((pf.host_port, pf.guest_port), (2323, 23));
        assert!(pf.bind == ForwardBind::Localhost);
        // Offered once: then greyed out, and adding it again changes nothing.
        let rows = [ForwardRow { tcp: true, host_port: 2323, guest_port: 23, any_interface: false }];
        assert!(!forward_menu(&rows)[0].1.is_yes());
        assert!(forward_menu(&rows)[1].1.is_yes());
        assert!(!c.apply(add(0).unwrap(), &mut cfg, None).0);
        assert_eq!(cfg.port_forward.len(), 1);
    }

    #[test]
    fn the_memory_slider_stops_at_powers_of_two_and_half_way() {
        let indy = [8, 16, 24, 32, 40, 64, 96, 128, 160, 256, 384, 512];
        assert_eq!(memory_stops(&indy, 256), vec![8, 16, 24, 32, 64, 96, 128, 256, 384, 512]);
        // An amount that is neither is a stop of its own.
        assert_eq!(memory_stops(&indy, 160), vec![8, 16, 24, 32, 64, 96, 128, 160, 256, 384, 512]);
        // A Tezro: 1, 1.5, 2, 3, 4, 6 and 8 GB.
        let tezro: Vec<u32> = (2..=16).map(|i| i * 512).collect();
        assert_eq!(memory_stops(&tezro, 4096), vec![1024, 1536, 2048, 3072, 4096, 6144, 8192]);
        // Too few that qualify: every amount.
        assert_eq!(memory_stops(&[512], 512), vec![512]);
        assert_eq!(memory_stops(&[80, 160], 160), vec![80, 160]);
    }

    #[test]
    fn unticking_the_built_in_prom_shows_the_path_field() {
        let mut c = SettingsController::default();
        let mut cfg = MachineConfig::default();
        cfg.prom = "(embedded)".into();
        let fields = |c: &SettingsController, cfg: &MachineConfig| -> Vec<Field> {
            c.sections(cfg, false).into_iter().flat_map(|s| s.fields).collect()
        };
        let builtin = |f: &[Field]| f.iter().find(|f| f.id == field::BUILTIN_PROM).map(|f| f.kind.clone());
        assert_eq!(builtin(&fields(&c, &cfg)), Some(FieldKind::Toggle(true)));
        assert!(!fields(&c, &cfg).iter().any(|f| f.id == field::PROM));
        // Unticked with no PROM file on disk: off, and the path field shows.
        c.apply(SettingsAction::Set { field: field::BUILTIN_PROM, value: Value::Bool(false) }, &mut cfg, None);
        assert_eq!(builtin(&fields(&c, &cfg)), Some(FieldKind::Toggle(false)));
        assert!(fields(&c, &cfg).iter().any(|f| f.id == field::PROM));
        // Ticked again: the built-in PROM, no path field.
        c.apply(SettingsAction::Set { field: field::BUILTIN_PROM, value: Value::Bool(true) }, &mut cfg, None);
        assert_eq!(builtin(&fields(&c, &cfg)), Some(FieldKind::Toggle(true)));
        assert!(!fields(&c, &cfg).iter().any(|f| f.id == field::PROM));
    }

    #[test]
    fn running_disables_power_on_fields_but_not_live_ones() {
        let c = SettingsController::default();
        let cfg = MachineConfig::default();
        let s = c.sections(&cfg, true);
        let machine = section(&s, super::super::ids::settings::GENERAL);
        assert!(!machine.fields[0].enabled.is_yes());
        let net = section(&s, super::super::ids::settings::NETWORK);
        let subnet = net.fields.iter().find(|f| f.id == field::NAT_SUBNET).unwrap();
        assert!(subnet.enabled.is_yes());
    }

    #[test]
    fn video_in_and_performance_edit_the_config() {
        let mut c = SettingsController::default();
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::IndyIp24;
        let s = c.sections(&cfg, false);
        assert!(s.iter().any(|x| x.id == "settings.video-in"));
        let set = |c: &mut SettingsController, cfg: &mut MachineConfig, id, v| {
            c.apply(action_for(id, v, &[], &[]).expect("an action"), cfg, None).0
        };
        assert!(set(&mut c, &mut cfg, field::VINO_SOURCE, Value::Index(2)));
        assert_eq!(cfg.vino.source, VinoSource::Camera);
        assert!(c.sections(&cfg, false).iter().flat_map(|s| &s.fields).any(|f| f.id == field::VINO_CAMERA));
        assert!(set(&mut c, &mut cfg, field::AFFINITY, Value::Bool(true)));
        assert!(set(&mut c, &mut cfg, field::REX3_CORE, Value::Number(3.0)));
        assert_eq!(cfg.perf.rex3_core, Some(3));
        assert!(set(&mut c, &mut cfg, field::VNC_LISTEN, Value::Text(" 5901 ".into())));
        assert_eq!(cfg.vnc.listen.as_deref(), Some("5901"));
        assert!(!set(&mut c, &mut cfg, field::VNC_LISTEN, Value::Text("5901".into())), "no change");
        cfg.validate().unwrap();
        // The NFS share.
        assert!(set(&mut c, &mut cfg, field::NFS_ENABLE, Value::Bool(true)));
        assert!(set(&mut c, &mut cfg, field::NFS_DIR, Value::Text("/tmp/share".into())));
        assert!(set(&mut c, &mut cfg, field::NFS_VERSION, Value::Index(1)));
        assert_eq!(cfg.nfs.as_ref().map(|n| (n.shared_dir.as_str(), n.version)), Some(("/tmp/share", NfsVersion::V2)));
        let (_, fx) = c.apply(SettingsAction::Pick(field::NFS_DIR), &mut cfg, None);
        assert!(matches!(fx.as_slice(), [Effect::PickFolder(_)]));
        assert!(set(&mut c, &mut cfg, field::NFS_ENABLE, Value::Bool(false)));
        assert!(cfg.nfs.is_none());
        // Only the Indy has VINO.
        cfg.machine.profile = MachineProfile::O2Ip32;
        assert!(!c.sections(&cfg, false).iter().any(|x| x.id == "settings.video-in"));
    }

    #[test]
    fn disks_add_remove_and_bad_toml_is_refused() {
        let mut c = SettingsController::default();
        let mut cfg = MachineConfig::default();
        crate::model::hardware::clear_disks(&mut cfg);
        c.apply(SettingsAction::Disk(DiskEdit::Add(ScsiKind::Cdrom)), &mut cfg, None);
        assert!(hardware::scsi_disks(&cfg).iter().any(|(_, d)| d.is_cdrom()));
        let key = cfg.disks()[0].0;
        // Rows come in SCSI order, whatever the map's order.
        c.apply(SettingsAction::Disk(DiskEdit::Add(ScsiKind::Disk)), &mut cfg, None);
        c.apply(SettingsAction::Disk(DiskEdit::Add(ScsiKind::Disk)), &mut cfg, None);
        let rows = c.sections(&cfg, false).into_iter().flat_map(|s| s.fields).find_map(|f| match f.kind {
            FieldKind::DiskTable(r) => Some(r),
            _ => None,
        });
        let keys: Vec<u8> = rows.unwrap().iter().map(|r| r.key).collect();
        assert_eq!(keys, [1, 2, 3]);
        cfg.remove_disk(2);
        cfg.remove_disk(3);
        // ID 0 is the controller: the first target goes at 1, and the config stays valid.
        assert_eq!(key, 1);
        assert!(cfg.validate().is_ok(), "{:?}", cfg.validate());
        c.apply(SettingsAction::Disk(DiskEdit::Remove(key)), &mut cfg, None);
        assert!(cfg.disks().is_empty());
        let (changed, _) = c.apply(SettingsAction::ApplyToml("made_up = true".into()), &mut cfg, None);
        assert!(!changed);
        let s = c.sections(&cfg, false);
        let adv = section(&s, "settings.advanced");
        assert!(adv.fields.iter().any(|f| matches!(f.kind, FieldKind::Note(Severity::Error, _))));
    }

    #[test]
    fn control_ids_become_actions() {
        let fw = vec![ForwardRow { tcp: true, host_port: 2222, guest_port: 22, any_interface: false }];
        assert_eq!(
            action_for(&control_id(field::DISKS, Some(0), Some(part::PATH)), Value::Text("a.img".into()), &[4], &fw),
            Some(SettingsAction::Disk(DiskEdit::SetPath { key: 4, path: "a.img".into() }))
        );
        assert_eq!(
            action_for(&control_id(field::FORWARDS, Some(0), Some(part::HOST)), Value::Text("2323".into()), &[], &fw),
            Some(SettingsAction::Forward(ForwardEdit::Set(0, ForwardRow { host_port: 2323, ..fw[0].clone() })))
        );
        assert_eq!(
            action_for(field::PROFILE, Value::Index(3), &[], &fw),
            Some(SettingsAction::Set { field: field::PROFILE, value: Value::Index(3) })
        );
        assert_eq!(action_for(&control_id(field::PROM, None, Some(part::BROWSE)), Value::Bool(true), &[], &fw), Some(SettingsAction::Pick(field::PROM)));
        assert_eq!(action_for("nonsense", Value::Bool(true), &[], &fw), None);
    }

    #[test]
    fn banks_only_take_the_profiles_sizes() {
        let c = SettingsController::default();
        let cfg = MachineConfig::default();
        let s = c.sections(&cfg, false);
        let FieldKind::BankGrid { sizes, banks, .. } = &section(&s, "settings.advanced").fields[0].kind else { panic!() };
        assert_eq!(banks.len(), MachineProfile::IndyIp24.memory_banks());
        assert!(sizes.iter().all(|s| *s == 0 || MachineProfile::IndyIp24.bank_sizes().contains(s)));
    }
}
