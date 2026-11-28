//! `[[devices]]`: what a config places, and the device types it can name. An entry has a `type`, an optional `at` (a slot path,
//! which may start with a machine alias or another entry's `name`), a `name` and the type's options; each type is one row of [`DEVICE_TYPES`].

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use std::sync::Arc;

use crate::dev::serial_dev::{SerialDeviceConfig, SerialKind};
use crate::config::{GraphicsBoard, MachineConfig, NvmeConfig, ScsiDeviceConfig, ScsiKind, Virtio9pConfig, VirtioSerialConfig};
use crate::dev::ng1::vc2_timings::NewportResolution;
use crate::dev::ice1724::{Ice1724, Variant};
use crate::dev::isp2200::{card::Loop, Isp2200};
use crate::dev::nvme::Nvme;
use crate::hptimer::TimerManager;
use crate::dev::xio_pci::{Kind, CAGE_SLOTS, SHOEHORN_SLOTS};
use crate::slot_path::MachineSlotPath;
use crate::dev::virtio::pci::VirtioPci;
use crate::pci::{DevFn, PciToPciBridge, PCI_INTA};
use crate::topology::{loop_slots, pci_bus_slots, DeviceFamily, DevicePlacement, MachineSlot, MachineSlotKind};

/// What a device is, with its options. A family step adds a variant here and
/// its rows to [`DEVICE_TYPES`].
#[derive(Clone, Debug)]
pub enum DeviceModel {
    /// A target on a SCSI bus: `hdd`, `cdrom` or `daynaport`, by its kind.
    Scsi(ScsiDeviceConfig),
    Nvme(NvmeConfig),
    /// An XIO option board: the PCI shoehorn or the Octane's PCI module.
    XioBoard(crate::dev::xio_pci::Kind),
    /// A DEC 21154 PCI-to-PCI bridge in a slot, with one slot behind it.
    PciBridge,
    Ice1724,
    Qla2200,
    /// The N64 development board.
    Ultra64,
    /// The bare-metal harness's console, dump and exit register.
    TestDevice(TestDeviceConfig),
    Board(BoardConfig),
    /// `none`: the slot stays empty, with no default.
    Empty,
    VirtioSerial(VirtioSerialConfig),
    Virtio9p(Virtio9pConfig),
    /// A device on a serial port: `hostserial`, `midi_synth` or `midi_raw`, by its kind.
    Serial(SerialDeviceConfig),
}

/// What the test device's run-configuration register publishes and where its
/// DUMP register writes: `iris-testdev-dump.json` in the working directory
/// when unset.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TestDeviceConfig {
    pub dump: Option<String>,
    /// As `time_pct=10,repeats=3`. Unset publishes 0, which the bare-metal
    /// benchmark suite reads as "every kernel, full length".
    pub run_config: Option<String>,
}

impl TestDeviceConfig {
    pub fn run(&self) -> Result<crate::dev::testdev::RunConfig, String> {
        self.run_config.as_deref().map_or(Ok(crate::dev::testdev::RunConfig::ALL), |s| s.parse().map_err(|e| format!("run_config {s:?}: {e}")))
    }
}

/// A graphics board's options; the entry's type is the family, `model` the board in it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoardConfig {
    #[serde(default)]
    pub model: GraphicsBoard,
    #[serde(default = "one")]
    pub raster_managers: u8,
    #[serde(default)]
    pub resolution: NewportResolution,
    /// The pipe's flash EEPROM file, kept between runs; unset is volatile.
    #[serde(default)]
    pub flash: Option<String>,
}

fn one() -> u8 {
    1
}

impl Default for BoardConfig {
    fn default() -> Self {
        BoardConfig { model: GraphicsBoard::default(), raster_managers: 1, resolution: NewportResolution::default(), flash: None }
    }
}

impl BoardConfig {
    pub fn new(model: GraphicsBoard) -> BoardConfig {
        BoardConfig { model, ..Default::default() }
    }

    pub fn label(&self) -> String {
        match self.model.kona() {
            Some(_) => format!("{} ({} RM)", self.model.label(), self.raster_managers),
            None => self.model.label().to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConfiguredDevice {
    pub name: Option<String>,
    pub at: Option<String>,
    pub device: DeviceModel,
    /// One of the disks a config with none gets: placed after everything else and
    /// dropped when it finds no slot. Written like any entry, and read back as one.
    pub default: bool,
}

pub struct DeviceTypeSpec {
    pub name: &'static str,
    pub family: DeviceFamily,
    /// The option keys the type takes besides `type`, `name` and `at`.
    keys: &'static [&'static str],
    parse: fn(&DeviceTypeSpec, toml::Table) -> Result<DeviceModel, String>,
    /// The slots a device of the type brings once placed, such as the bus a
    /// bridge puts behind it. Placement adds them before it places the devices that go there.
    pub slots: Option<fn(&DeviceModel, &MachineSlotPath, &mut Vec<MachineSlot>)>,
}

/// Every device type, one row each.
pub const DEVICE_TYPES: &[DeviceTypeSpec] = &[
    DeviceTypeSpec { name: "hdd", family: DeviceFamily::Disk, keys: &["path", "write_cache", "overlay", "scratch", "size_mb"], parse: scsi_target, slots: None },
    DeviceTypeSpec { name: "cdrom", family: DeviceFamily::Scsi, keys: &["path", "discs", "overlay"], parse: scsi_target, slots: None },
    DeviceTypeSpec { name: "daynaport", family: DeviceFamily::Scsi, keys: &["mac", "subnet"], parse: scsi_target, slots: None },
    DeviceTypeSpec { name: "pci_cage", family: DeviceFamily::Cage, keys: &[], parse: |_, _| Ok(DeviceModel::XioBoard(Kind::Cage)), slots: Some(|_, at, s| pci_bus_slots(s, &at.pci(0).to_string(), "PCI module", CAGE_SLOTS)) },
    DeviceTypeSpec { name: "pci_shoehorn", family: DeviceFamily::Xio, keys: &[], parse: |_, _| Ok(DeviceModel::XioBoard(Kind::Shoehorn)), slots: Some(|_, at, s| pci_bus_slots(s, &at.pci(0).to_string(), "PCI shoehorn", SHOEHORN_SLOTS)) },
    // One slot behind it, device 0, where iris has always put a card.
    DeviceTypeSpec { name: "pci_bridge", family: DeviceFamily::Pci, keys: &[], parse: |_, _| Ok(DeviceModel::PciBridge), slots: Some(|_, at, s| pci_bus_slots(s, &at.pci(0).to_string(), "pci bridge", &[(0, MachineSlotKind::PCI_SLOT, None)])) },
    DeviceTypeSpec { name: "ice1724", family: DeviceFamily::Pci, keys: &[], parse: |_, _| Ok(DeviceModel::Ice1724), slots: None },
    DeviceTypeSpec { name: "qla2200", family: DeviceFamily::Pci, keys: &[], parse: |_, _| Ok(DeviceModel::Qla2200), slots: Some(|_, at, s| loop_slots(s, at)) },
    DeviceTypeSpec { name: "nvme", family: DeviceFamily::Pci, keys: &["path", "overlay", "write_cache"], parse: nvme, slots: None },
    DeviceTypeSpec { name: "ultra64", family: DeviceFamily::Gio, keys: &[], parse: |_, _| Ok(DeviceModel::Ultra64), slots: None },
    DeviceTypeSpec {
        name: "test_device",
        family: DeviceFamily::Gio,
        keys: &["dump", "run_config"],
        parse: |_, o| options::<TestDeviceConfig>(o).and_then(|c| c.run().map(|_| DeviceModel::TestDevice(c))),
        slots: None,
    },
    DeviceTypeSpec { name: "newport", family: DeviceFamily::Gfx, keys: &["resolution"], parse: board, slots: None },
    DeviceTypeSpec { name: "express", family: DeviceFamily::Gfx, keys: &["model"], parse: board, slots: None },
    DeviceTypeSpec { name: "impact", family: DeviceFamily::Gfx, keys: &["model"], parse: board, slots: None },
    DeviceTypeSpec { name: "vpro", family: DeviceFamily::Gfx, keys: &["model"], parse: board, slots: None },
    DeviceTypeSpec { name: "kona", family: DeviceFamily::Gfx, keys: &["model", "raster_managers", "flash"], parse: board, slots: None },
    DeviceTypeSpec { name: "none", family: DeviceFamily::Empty, keys: &[], parse: |_, _| Ok(DeviceModel::Empty), slots: None },
    DeviceTypeSpec {
        name: "virtio_serial",
        family: DeviceFamily::Virtio,
       
        keys: &["port"],
        parse: |_, o| options(o).map(|v: VirtioSerialConfig| DeviceModel::VirtioSerial(VirtioSerialConfig { enabled: true, ..v })),
        slots: None,
    },
    DeviceTypeSpec {
        name: "virtio_9p",
        family: DeviceFamily::Virtio,
       
        keys: &["path", "tag", "readonly", "uid", "gid", "owners", "uid_map", "gid_map"],
        parse: |_, o| options(o).map(|v: Virtio9pConfig| DeviceModel::Virtio9p(Virtio9pConfig { enabled: true, ..v })),
        slots: None,
    },
    DeviceTypeSpec { name: "hostserial", family: DeviceFamily::Serial, keys: &["path"], parse: serial_device, slots: None },
    DeviceTypeSpec { name: "midi_synth", family: DeviceFamily::Serial, keys: &["path", "gain"], parse: serial_device, slots: None },
    DeviceTypeSpec { name: "midi_raw", family: DeviceFamily::Serial, keys: &["path"], parse: serial_device, slots: None },
];

fn board(t: &DeviceTypeSpec, opts: toml::Table) -> Result<DeviceModel, String> {
    let b: BoardConfig = options(opts)?;
    if b.model.family() != t.name {
        let models: Vec<_> = GraphicsBoard::ALL.into_iter().filter(|m| m.family() == t.name).map(GraphicsBoard::as_str).collect();
        return Err(format!("model must be one of {}", models.join(", ")));
    }
    Ok(DeviceModel::Board(b))
}

fn serial_device(t: &DeviceTypeSpec, opts: toml::Table) -> Result<DeviceModel, String> {
    let kind = SerialKind::from_name(t.name).expect("a serial type");
    let dev = SerialDeviceConfig { kind, ..options(opts)? };
    dev.check()?;
    Ok(DeviceModel::Serial(dev))
}

fn nvme(_: &DeviceTypeSpec, opts: toml::Table) -> Result<DeviceModel, String> {
    let cfg: NvmeConfig = toml::Value::Table(opts).try_into().map_err(|e: toml::de::Error| e.message().to_string())?;
    if cfg.path.is_empty() {
        return Err("needs a path".into());
    }
    Ok(DeviceModel::Nvme(cfg))
}

fn options<T: DeserializeOwned>(opts: toml::Table) -> Result<T, String> {
    toml::Value::Table(opts).try_into().map_err(|e: toml::de::Error| e.message().to_string())
}

fn scsi_kind(name: &str) -> ScsiKind {
    match name {
        "cdrom" => ScsiKind::Cdrom,
        "daynaport" => ScsiKind::Daynaport,
        _ => ScsiKind::Disk,
    }
}

fn scsi_target(t: &DeviceTypeSpec, opts: toml::Table) -> Result<DeviceModel, String> {
    let mut dev: ScsiDeviceConfig = options(opts)?;
    dev.kind_field = scsi_kind(t.name);
    dev.cdrom = dev.kind_field == ScsiKind::Cdrom;
    Ok(DeviceModel::Scsi(dev))
}

impl DeviceModel {
    pub fn spec(&self) -> &'static DeviceTypeSpec {
        let name = match self {
            DeviceModel::Scsi(d) => match d.kind() {
                ScsiKind::Disk => "hdd",
                ScsiKind::Cdrom => "cdrom",
                ScsiKind::Daynaport => "daynaport",
            },
            DeviceModel::Nvme(_) => "nvme",
            DeviceModel::XioBoard(kind) => kind.name(),
            DeviceModel::PciBridge => "pci_bridge",
            DeviceModel::Ice1724 => "ice1724",
            DeviceModel::Qla2200 => "qla2200",
            DeviceModel::Ultra64 => "ultra64",
            DeviceModel::TestDevice(_) => "test_device",
            DeviceModel::Board(b) => b.model.family(),
            DeviceModel::Empty => "none",
            DeviceModel::VirtioSerial(_) => "virtio_serial",
            DeviceModel::Virtio9p(_) => "virtio_9p",
            DeviceModel::Serial(d) => d.kind.name(),
        };
        DEVICE_TYPES.iter().find(|t| t.name == name).expect("every variant has a type")
    }

    pub fn scsi(&self) -> Option<&ScsiDeviceConfig> {
        match self {
            DeviceModel::Scsi(d) => Some(d),
            _ => None,
        }
    }

    pub fn scsi_mut(&mut self) -> Option<&mut ScsiDeviceConfig> {
        match self {
            DeviceModel::Scsi(d) => Some(d),
            _ => None,
        }
    }

    pub fn nvme_mut(&mut self) -> Option<&mut NvmeConfig> {
        match self {
            DeviceModel::Nvme(d) => Some(d),
            _ => None,
        }
    }

    /// What `--list-slots` says is there.
    pub fn summary(&self) -> String {
        match self {
            DeviceModel::Scsi(ScsiDeviceConfig { path, .. }) | DeviceModel::Nvme(NvmeConfig { path, .. }) | DeviceModel::Serial(SerialDeviceConfig { path, .. }) if !path.is_empty() => {
                format!("{} {path}", self.spec().name)
            }
            DeviceModel::Board(b) if b.model != GraphicsBoard::Newport => format!("{} {}", self.spec().name, b.model),
            _ => self.spec().name.to_string(),
        }
    }

    /// The options that differ from the type's defaults, as a file writes them.
    fn options(&self) -> toml::Table {
        fn changed<T: Serialize + Default>(now: &T, keys: &[&str]) -> toml::Table {
            let default = toml::Table::try_from(T::default()).expect("options serialize");
            let now = toml::Table::try_from(now).expect("options serialize");
            now.into_iter().filter(|(k, v)| keys.contains(&k.as_str()) && default.get(k) != Some(v)).collect()
        }
        let keys = self.spec().keys;
        match self {
            DeviceModel::Scsi(d) => changed(d, keys),
            DeviceModel::Nvme(d) => changed(d, keys),
            DeviceModel::TestDevice(d) => changed(d, keys),
            DeviceModel::VirtioSerial(d) => changed(d, keys),
            DeviceModel::Virtio9p(d) => changed(d, keys),
            DeviceModel::Serial(d) => changed(d, keys),
            DeviceModel::Board(b) => changed(b, keys),
            DeviceModel::XioBoard(_) | DeviceModel::PciBridge | DeviceModel::Ice1724 | DeviceModel::Qla2200 | DeviceModel::Ultra64 | DeviceModel::Empty => toml::Table::new(),
        }
    }
}

impl ConfiguredDevice {
    pub fn new(at: &str, device: DeviceModel) -> ConfiguredDevice {
        ConfiguredDevice { name: None, at: Some(at.to_string()), device, default: false }
    }

    pub fn from_table(mut t: toml::Table) -> Result<ConfiguredDevice, String> {
        let mut text = |key: &str| match t.remove(key) {
            None => Ok(None),
            Some(toml::Value::String(s)) => Ok(Some(s)),
            Some(v) => Err(format!("{key} = {v} is not a string")),
        };
        let name = text("type")?.ok_or("a device needs a type")?;
        let (at, local) = (text("at")?, text("name")?);
        let ty = DEVICE_TYPES.iter().find(|t| t.name == name).ok_or_else(|| {
            format!("unknown device type \"{name}\" (types: {})", DEVICE_TYPES.iter().map(|t| t.name).collect::<Vec<_>>().join(", "))
        })?;
        if let Some(key) = t.keys().find(|k| !ty.keys.contains(&k.as_str())) {
            return Err(format!("{name} takes no key {key} (keys: type, name, at, {})", ty.keys.join(", ")));
        }
        let device = (ty.parse)(ty, t).map_err(|e| format!("{name}: {e}"))?;
        if matches!(device, DeviceModel::Empty) && at.is_none() {
            return Err("none: needs an at, the slot to leave empty".into());
        }
        Ok(ConfiguredDevice { name: local, at, device, default: false })
    }

    pub fn names_graphics(&self, profile: crate::config::MachineProfile) -> bool {
        matches!(self.device, DeviceModel::Board(_) | DeviceModel::Empty) && self.at.as_deref().is_none_or(|at| profile.is_graphics_slot(at))
    }

    /// The entry's keys in the order a file lists them: `type`, `name`, `at`, then the options.
    fn keys(&self) -> Vec<(String, toml::Value)> {
        let mut keys = vec![("type".to_string(), self.device.spec().name.into())];
        keys.extend(self.name.clone().map(|n| ("name".to_string(), n.into())));
        keys.extend(self.at.clone().map(|a| ("at".to_string(), a.into())));
        keys.extend(self.device.options());
        keys
    }

    /// `--device [NAME:]TYPE[@SLOT][,KEY=VALUE]...`. A value is read as TOML
    /// when it parses as such, else as a string, so `path=a.img` needs no quotes.
    pub fn from_arg(arg: &str) -> Result<ConfiguredDevice, String> {
        let mut parts = arg.split(',');
        let head = parts.next().unwrap_or_default();
        let (name, rest) = match head.split_once(':') {
            Some((n, r)) => (Some(n), r),
            None => (None, head),
        };
        let (ty, at) = match rest.split_once('@') {
            Some((t, a)) => (t, Some(a)),
            None => (rest, None),
        };
        let mut t = toml::Table::new();
        t.insert("type".into(), ty.into());
        name.map(|n| t.insert("name".into(), n.into()));
        at.map(|a| t.insert("at".into(), a.into()));
        for kv in parts {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("--device {arg}: `{kv}` is not KEY=VALUE"))?;
            let value = toml::from_str::<toml::Table>(&format!("v = {v}")).ok().and_then(|mut t| t.remove("v"));
            t.insert(k.into(), value.unwrap_or_else(|| v.into()));
        }
        ConfiguredDevice::from_table(t).map_err(|e| format!("--device {arg}: {e}"))
    }
}

/// Said at startup and by `--validate` when an audio card is placed with
/// `no_audio` set: the guest sees the card and nothing is played.
pub(crate) const AUDIO_WITH_NO_AUDIO: &str = "an ice1724 card is placed but no_audio is set: the card plays to nowhere";

/// An NVMe card at its slot.
pub struct NvmeCard {
    pub at: MachineSlotPath,
    pub nvme: Arc<Nvme>,
    pub cfg: NvmeConfig,
}

/// A Fibre Channel adapter at its slot, with its loop's disks as a snapshot records them.
pub struct FibreCard {
    pub at: MachineSlotPath,
    pub fc: Arc<Isp2200>,
    pub disks: Vec<crate::snapshot::DiskRef>,
}

/// What [`DevicePlacement::build_cards`] built, each at its slot, for the machine to start, stop, reset and snapshot.
#[derive(Default)]
pub struct AttachedCards {
    pub nvme: Vec<NvmeCard>,
    pub fibre: Vec<FibreCard>,
    pub virtio: Vec<(MachineSlotPath, Arc<VirtioPci>)>,
    pub audio: Vec<(MachineSlotPath, Arc<Ice1724>)>,
}

impl AttachedCards {
    pub fn devices(&self) -> impl Iterator<Item = Arc<dyn crate::traits::Device>> + '_ {
        let audio = self.audio.iter().map(|(_, a)| a.clone() as Arc<dyn crate::traits::Device>);
        let fibre = self.fibre.iter().map(|f| f.fc.clone() as Arc<dyn crate::traits::Device>);
        let nvme = self.nvme.iter().map(|c| c.nvme.clone() as Arc<dyn crate::traits::Device>);
        self.virtio.iter().map(|(_, v)| v.clone() as Arc<dyn crate::traits::Device>).chain(audio).chain(fibre).chain(nvme)
    }

    pub fn is_empty(&self) -> bool {
        self.nvme.is_empty() && self.fibre.is_empty() && self.virtio.is_empty() && self.audio.is_empty()
    }

    pub fn power_on(&self) {
        for (_, v) in &self.virtio {
            crate::traits::Resettable::power_on(&**v);
        }
        for (_, a) in &self.audio {
            a.reset();
        }
    }

    /// The type and slot of each card, for a snapshot's manifest.
    pub fn manifest(&self) -> Vec<(String, MachineSlotPath)> {
        let nvme = self.nvme.iter().map(|c| ("nvme".to_string(), c.at.clone()));
        let virtio = self.virtio.iter().map(|(at, v)| (if v.core().device_id() == crate::dev::virtio::console::DEVICE_ID { "virtio_serial" } else { "virtio_9p" }.to_string(), at.clone()));
        let fibre = self.fibre.iter().map(|f| ("qla2200".to_string(), f.at.clone()));
        nvme.chain(virtio).chain(fibre).chain(self.audio.iter().map(|(at, _)| ("ice1724".to_string(), at.clone()))).collect()
    }

    /// The NVMe cards' and the Fibre Channel loops' disks, by the card's slot.
    pub fn disk_buses(&self) -> Vec<crate::snapshot::DiskBus<'_>> {
        use crate::scsi_monitor::ScsiTargets;
        let nvme = self.nvme.iter().map(|c| crate::snapshot::DiskBus::card(c.at.clone(), c.nvme.scsi_bus()));
        nvme.chain(self.fibre.iter().map(|f| crate::snapshot::DiskBus::loop_ids(f.at.clone(), f.fc.bus()))).collect()
    }

    /// The manifest's record of each card's disks.
    pub fn disk_refs(&self) -> Vec<crate::snapshot::DiskRef> {
        let nvme = self.nvme.iter().map(|c| crate::dev::nvme::disk_ref(&c.cfg, &c.nvme, c.at.clone()));
        nvme.chain(self.fibre.iter().flat_map(|f| f.disks.clone())).collect()
    }

    /// Why a snapshot cannot hold these cards now, if it cannot.
    pub fn snapshot_blocker(&self) -> Option<String> {
        let nine_p = self.virtio.iter().find(|(_, v)| v.core().device_id() != crate::dev::virtio::console::DEVICE_ID);
        nine_p.map(|(at, _)| format!("the virtio-9p share at {at} keeps the guest's open files, which a snapshot does not hold"))
    }

    /// Each card's state under its slot.
    pub fn save_state(&self) -> toml::Value {
        use crate::traits::Saveable;
        let mut t = toml::Table::new();
        for c in &self.nvme {
            t.insert(c.at.to_string(), c.nvme.save_state());
        }
        for f in &self.fibre {
            t.insert(f.at.to_string(), f.fc.save_state());
        }
        for (at, v) in &self.virtio {
            t.insert(at.to_string(), v.core().save_state());
        }
        for (at, a) in &self.audio {
            t.insert(at.to_string(), a.save_state());
        }
        toml::Value::Table(t)
    }

    /// Put back what [`AttachedCards::save_state`] gave; a card with no entry is refused.
    pub fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        use crate::traits::Saveable;
        let state = |at: &MachineSlotPath| crate::snapshot::field(v, &at.to_string()).map_err(|_| format!("snapshot has no state for the card at {at}"));
        for c in &self.nvme {
            c.nvme.load_state(state(&c.at)?).map_err(|e| format!("nvme at {}: {e}", c.at))?;
        }
        for f in &self.fibre {
            f.fc.load_state(state(&f.at)?).map_err(|e| format!("qla2200 at {}: {e}", f.at))?;
        }
        for (at, v) in &self.virtio {
            v.core().load_state(state(at)?).map_err(|e| format!("virtio at {at}: {e}"))?;
        }
        for (at, d) in &self.audio {
            d.load_state(state(at)?).map_err(|e| format!("ice1724 at {at}: {e}"))?;
        }
        Ok(())
    }
}

impl DevicePlacement {
/// Build the PCI cards at the slots placement gave them, each bridge before the cards behind it, then check every PCI bus slot has its bus.
pub fn build_cards(&mut self, cfg: &MachineConfig, timers: &Arc<TimerManager>) -> Result<AttachedCards, String> {
    let mut cards: Vec<(ConfiguredDevice, MachineSlotPath)> = self.placed()
        .filter(|(_, at)| self.slot(at).is_some_and(|s| s.kind.takes(DeviceFamily::Pci, true)))
        .map(|(d, at)| (d.clone(), at.clone()))
        .collect();
    cards.sort_by_key(|(_, at)| at.depth());
    let mut out = AttachedCards::default();
    let profile = cfg.machine.profile;
    for (d, at) in cards {
        let bus = at.parent().and_then(|b| self.pci(&b).cloned()).ok_or_else(|| format!("{at}: no PCI bus is built there"))?;
        let slot = DevFn { device: at.last_number().unwrap_or(0), function: 0 };
        let pin = match self.slot(&at).map(|s| &s.kind) {
            Some(MachineSlotKind::Extra { pin }) => *pin,
            _ => PCI_INTA,
        };
        let virtio = |name: &str, core| {
            let dev = Arc::new(VirtioPci::with_pin(core, pin));
            bus.attach(slot, dev.clone());
            eprintln!("iris: {name} at {at}");
            dev
        };
        match &d.device {
            DeviceModel::Nvme(cfg) => {
                profile.refuse_nvme()?;
                let card = crate::dev::nvme::Card::open(cfg)?;
                card.attach(&bus, slot.device);
                eprintln!("iris: NVMe at {at}");
                out.nvme.push(NvmeCard { at: at.clone(), nvme: card.nvme, cfg: cfg.clone() });
            }
            DeviceModel::VirtioSerial(v) => out.virtio.push((at.clone(), virtio("virtio-serial", crate::dev::virtio::console::from_config(v)))),
            DeviceModel::Virtio9p(v) => out.virtio.push((at.clone(), virtio("virtio-9p", crate::dev::virtio::p9::from_config(v)))),
            DeviceModel::Ice1724 => {
                let card = Arc::new(Ice1724::new(timers, Variant::Dm8));
                if cfg.no_audio {
                    eprintln!("iris: {AUDIO_WITH_NO_AUDIO}");
                } else {
                    card.enable_output(cfg.audio.clone());
                }
                bus.attach(slot, card.clone());
                card.register_locks();
                eprintln!("iris: ICE1724 at {at}");
                out.audio.push((at.clone(), card));
            }
            DeviceModel::Qla2200 => {
                if !out.fibre.is_empty() {
                    return Err("[[devices]] qla2200: the machine takes one Fibre Channel adapter".into());
                }
                let disks = self.card_loop(&at);
                let card = crate::dev::isp2200::card::Card::open(&disks, cfg.ci, timers)?;
                card.attach(&bus, slot.device);
                card.fc.register_locks();
                eprintln!("iris: Fibre Channel at {at}, {} loop disk(s)", disks.len());
                out.fibre.push(FibreCard { disks: crate::dev::isp2200::card::disk_refs(&at, &disks), at: at.clone(), fc: card.fc });
            }
            DeviceModel::PciBridge => {
                let ppb = PciToPciBridge::dec_21154(&bus, slot);
                bus.attach(slot, ppb.clone());
                self.bind_pci(&at.pci(0).to_string(), ppb.secondary().clone())?;
            }
            other => return Err(format!("[[devices]] {} at {at}: this machine cannot build it", other.spec().name)),
        }
    }
    self.check_pci()?;
    Ok(out)
}
}

impl Serialize for ConfiguredDevice {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let keys = self.keys();
        let mut m = s.serialize_map(Some(keys.len()))?;
        for (k, v) in &keys {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

impl<'de> Deserialize<'de> for ConfiguredDevice {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        ConfiguredDevice::from_table(toml::Table::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MachineProfile;

    fn entry(text: &str) -> Result<ConfiguredDevice, String> {
        ConfiguredDevice::from_table(toml::from_str(text).unwrap())
    }

    #[test]
    fn each_scsi_type_sets_its_kind_and_round_trips() {
        for (ty, kind) in [("hdd", ScsiKind::Disk), ("cdrom", ScsiKind::Cdrom), ("daynaport", ScsiKind::Daynaport)] {
            let e = entry(&format!("type = \"{ty}\"\nat = \"scsi0/t3\"")).unwrap();
            assert_eq!(e.device.scsi().unwrap().kind(), kind);
            assert_eq!(e.device.spec().name, ty);
            let back = ConfiguredDevice::from_table(toml::Table::try_from(&e).unwrap()).unwrap();
            assert_eq!((back.device.spec().name, back.at.as_deref()), (ty, Some("scsi0/t3")));
        }
        let e = entry("type = \"hdd\"\npath = \"a.raw\"\nwrite_cache = false").unwrap();
        #[derive(Serialize)]
        struct File { devices: Vec<ConfiguredDevice> }
        let text = toml::to_string(&File { devices: vec![e] }).unwrap();
        assert_eq!(text, "[[devices]]\ntype = \"hdd\"\npath = \"a.raw\"\nwrite_cache = false\n", "type first, defaults left out");
    }

    #[test]
    fn a_key_the_type_lacks_or_an_unknown_type_is_an_error() {
        assert!(entry("type = \"cdrom\"\nscratch = true").unwrap_err().contains("cdrom takes no key scratch"));
        assert!(entry("type = \"hdd\"\ncontroller = 1").unwrap_err().contains("no key controller"));
        assert!(entry("type = \"floppy\"").unwrap_err().contains("types: hdd, cdrom, daynaport, pci_cage"));
        assert!(entry("at = \"scsi0/t1\"").unwrap_err().contains("needs a type"));
    }

    #[test]
    fn a_board_is_its_family_with_a_model_and_round_trips() {
        for (text, family, model) in [
            ("type = \"vpro\"\nmodel = \"v12\"", "vpro", GraphicsBoard::V12),
            ("type = \"impact\"\nmodel = \"maximpact\"", "impact", GraphicsBoard::MaxImpact),
            ("type = \"express\"\nmodel = \"elan\"", "express", GraphicsBoard::Elan),
            ("type = \"newport\"", "newport", GraphicsBoard::Newport),
        ] {
            let e = entry(text).unwrap();
            assert_eq!(e.device.spec().name, family);
            let DeviceModel::Board(b) = &e.device else { panic!("{text}") };
            assert_eq!(b.model, model);
            let back = ConfiguredDevice::from_table(toml::Table::try_from(&e).unwrap()).unwrap();
            assert_eq!(back.device.summary(), e.device.summary());
        }
        let k = entry("type = \"kona\"\nmodel = \"ir2e\"\nraster_managers = 2\nflash = \"p.bin\"").unwrap();
        let DeviceModel::Board(b) = &k.device else { panic!() };
        assert_eq!((b.raster_managers, b.flash.as_deref()), (2, Some("p.bin")));
        assert_eq!(toml::to_string(&k).unwrap(), "type = \"kona\"\nflash = \"p.bin\"\nmodel = \"ir2e\"\nraster_managers = 2\n");
        assert!(entry("type = \"newport\"\nresolution = \"1280x1024\"").is_ok());
    }

    #[test]
    fn a_board_needs_a_model_of_its_family_and_none_needs_a_slot() {
        assert!(entry("type = \"vpro\"").unwrap_err().contains("model must be one of v6, v8, v10, v12"));
        assert!(entry("type = \"vpro\"\nmodel = \"xz\"").unwrap_err().contains("model must be one of v6"));
        assert!(entry("type = \"vpro\"\nmodel = \"v14\"").unwrap_err().contains("unknown graphics model"));
        assert!(entry("type = \"vpro\"\nmodel = \"v12\"\nraster_managers = 2").unwrap_err().contains("vpro takes no key raster_managers"));
        assert!(entry("type = \"newport\"\nmodel = \"newport\"").unwrap_err().contains("newport takes no key model"));
        assert!(entry("type = \"none\"").unwrap_err().contains("none: needs an at"));
    }

    #[test]
    fn device_arg_takes_a_name_a_slot_and_options() {
        let e = ConfiguredDevice::from_arg("boot:hdd@scsi0/t2,path=root.img,write_cache=false").unwrap();
        assert_eq!((e.name.as_deref(), e.at.as_deref()), (Some("boot"), Some("scsi0/t2")));
        let d = e.device.scsi().unwrap();
        assert_eq!((d.path.as_str(), d.write_cache), ("root.img", false));
        assert!(ConfiguredDevice::from_arg("cdrom").unwrap().at.is_none());
        assert!(ConfiguredDevice::from_arg("hdd@scsi0/t1,path").unwrap_err().contains("not KEY=VALUE"));
    }
}
