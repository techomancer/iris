//! The keys older versions of iris wrote, read with a warning and turned into their current form. `[scsi.N]`, `--scsiN`,
//! `--cdromN` and the bare drive number live only here; the card flags keep their fields in `config.rs`.

use super::*;
use crate::devices::{BoardConfig, ConfiguredDevice, DeviceModel, TestDeviceConfig};
use crate::topology::{slots_take, DeviceFamily, DevicePlacement};

/// The older `[ultra64]` section.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Ultra64Config {
    #[serde(default)]
    pub enabled: bool,
}

/// The older `slot` of `[virtio_serial]` and `[virtio_9p]`: a GIO expansion slot.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VirtioGioSlot {
    Exp0,
    Exp1,
}

/// The older GIO flags (`[ultra64]`, `test_device`) as
/// `[[devices]]` entries in the slot they named, once the machine is known.
pub(super) fn gio_cards(cfg: &mut MachineConfig) -> Result<(), String> {
    if cfg.test_device_run_config.is_some() && !cfg.test_device {
        return Err("test_device_run_config needs test_device".into());
    }
    if !slots_take(&cfg.machine.profile.slots(), DeviceFamily::Gio) {
        return Ok(());
    }
    let mut cards: Vec<(&str, DeviceModel)> = Vec::new();
    let mut moved: Vec<&str> = Vec::new();
    if cfg.ultra64.enabled {
        cards.push(("gio0", DeviceModel::Ultra64));
        moved.push("[ultra64] -> ultra64");
    }
    if cfg.test_device {
        let run = TestDeviceConfig { dump: cfg.test_device_dump.clone(), run_config: cfg.test_device_run_config.clone() };
        run.run()?;
        cards.push(("gio0", DeviceModel::TestDevice(run)));
    }
    cfg.ultra64.enabled = false;
    (cfg.test_device, cfg.test_device_dump, cfg.test_device_run_config) = (false, None, None);
    cfg.devices.extend(cards.into_iter().map(|(at, d)| ConfiguredDevice::new(at, d)));
    if !moved.is_empty() {
        eprintln!("iris: deprecated keys folded into [[devices]] ({}); move them in the config to silence this", moved.join(", "));
    }
    Ok(())
}

/// An enabled `[virtio_serial]` or `[virtio_9p]` as a `[[devices]]` entry, in the GIO slot it named,
/// else wherever the machine has a slot for it.
pub(super) fn virtio_cards(cfg: &mut MachineConfig) {
    let gio = slots_take(&cfg.machine.profile.slots(), DeviceFamily::Gio);
    let at = |slot: Option<VirtioGioSlot>| gio.then(|| if slot == Some(VirtioGioSlot::Exp0) { "gio0" } else { "gio1" }.to_string());
    let mut moved: Vec<&str> = Vec::new();
    if cfg.virtio_serial.enabled {
        let serial = std::mem::take(&mut cfg.virtio_serial);
        cfg.devices.push(ConfiguredDevice { at: at(serial.slot), ..ConfiguredDevice::new("", DeviceModel::VirtioSerial(serial)) });
        moved.push("[virtio_serial] -> virtio_serial");
    }
    if cfg.virtio_9p.enabled {
        let p9 = std::mem::take(&mut cfg.virtio_9p);
        cfg.devices.push(ConfiguredDevice { at: at(p9.slot), ..ConfiguredDevice::new("", DeviceModel::Virtio9p(p9)) });
        moved.push("[virtio_9p] -> virtio_9p");
    }
    if !moved.is_empty() {
        eprintln!("iris: deprecated keys folded into [[devices]] ({}); move them in the config to silence this", moved.join(", "));
    }
}

/// What replaces each key that placement dropped, and the sections it was valid in (none: anywhere).
const DROPPED: &[(&str, &[&str], &str)] = &[
    ("xio_board", &[], "a `pci_cage` or `pci_shoehorn` [[devices]] entry at the XIO slot"),
    ("nvme", &[], "an `nvme` [[devices]] entry"),
    ("fibre_channel", &[], "a `qla2200` [[devices]] entry, with its disks as `hdd` entries at CARD/lN"),
    ("audio_card", &[], "an `ice1724` [[devices]] entry"),
    ("audio", &["fuel", "tezro", "origin3200", "onyx3000"], "an `ice1724` [[devices]] entry"),
    ("node", &["graphics"], "the `at` of the board's [[devices]] entry"),
    ("widget", &["graphics"], "the `at` of the board's [[devices]] entry"),
    ("bridge_slot", &[], "the `at` of the virtio [[devices]] entry"),
];

/// A parse error for a dropped key, with what to write instead; any other error as it was.
pub(super) fn name_replacement(e: toml::de::Error, text: &str) -> toml::de::Error {
    let key = e.message().strip_prefix("unknown field `").and_then(|m| m.split('`').next());
    let section = || {
        let before = text.get(..e.span()?.start)?;
        let header = before.lines().rev().find_map(|l| l.trim().strip_prefix('['))?;
        header.trim_start_matches('[').split(']').next()
    };
    match key.and_then(|k| DROPPED.iter().find(|d| d.0 == k && (d.1.is_empty() || section().is_some_and(|s| d.1.contains(&s))))) {
        Some((key, _, instead)) => <toml::de::Error as serde::de::Error>::custom(format!("{e}\n`{key}` is no longer read: write {instead}")),
        None => e,
    }
}

/// The board an older `[graphics] board` name stands for and its raster managers; the inner none is `none`.
fn old_board(name: &str) -> Option<Option<(GraphicsBoard, u8)>> {
    let k = name.trim().to_ascii_lowercase().replace(['_', '-'], ":");
    let k = k.strip_prefix("vpro:").unwrap_or(&k);
    let (k, rms) = match (k.strip_suffix(":2rm"), k) {
        (Some(k), _) => (k, 2),
        (_, "ir2x2") => ("ir2", 2),
        _ => (k.strip_suffix(":1rm").unwrap_or(k), 1),
    };
    let model = match k {
        "none" => return Some(None),
        "xl" => "newport",
        "gr2" | "gr2:xz" => "xz",
        "gr2:elan" => "elan",
        "gr2:extreme" => "extreme",
        "impact:solid" | "solid:impact" => "solid",
        "impact:high" | "high:impact" => "high",
        "impact:max" | "max:impact" | "maximum:impact" | "maximumimpact" => "max",
        "infinitereality2" => "ir2",
        "infinitereality3" => "ir3",
        k => k,
    };
    Some(Some((model.parse().ok()?, rms)))
}

pub(crate) const GRAPHICS_MOVED: &str = "moved to [[devices]] entries (--device on the command line); see HELP.md \"Graphics boards\"";

/// Refuse the old `[graphics]` board keys, or for the GUI that wrote them turn them into entries.
pub(super) fn graphics(cfg: &mut MachineConfig, migrate: bool) -> Result<(), String> {
    let impact = cfg.impact.take().and_then(|i| i.gfx).filter(|g| ["solid", "high", "max"].contains(&g.to_ascii_lowercase().as_str()));
    let g = &mut cfg.graphics;
    let from_impact = g.board.is_none() && impact.is_some();
    let (board, heads, res, flash) = (g.board.take().or(impact), g.heads.take().filter(|&h| h != 1), g.resolution.take().filter(|r| !r.is_guest()), g.flash.take());
    if board.is_none() && heads.is_none() && res.is_none() && flash.is_none() {
        return Ok(());
    }
    // An `[impact]` section, which no older iris refused, still loads.
    if !migrate && !from_impact {
        return Err(format!("[graphics] board, heads, resolution and flash {GRAPHICS_MOVED}"));
    }
    if !matches!(heads, None | Some(2)) {
        return Err("graphics.heads must be 1 or 2".into());
    }
    // What the GUI wrote for every machine: the machine's own board, or a board for a machine with no slot, is no entry.
    let profile = cfg.machine.profile;
    let default = profile.default_board().map(|m| (m, 1));
    let named = board.as_deref().map(|b| old_board(b).ok_or_else(|| format!("[graphics] board \"{b}\" is not a graphics board"))).transpose()?;
    let want = named.unwrap_or(default);
    if want.is_none() && named.is_none() && (res.is_some() || flash.is_some()) {
        return Err("[graphics] resolution and flash need a board".into());
    }
    if !profile.graphics_boards().is_empty() && (want != default || res.is_some() || flash.is_some()) {
        cfg.devices.push(match want {
            Some((model, raster_managers)) => ConfiguredDevice { at: None, ..ConfiguredDevice::new("", DeviceModel::Board(BoardConfig { model, raster_managers, resolution: res.unwrap_or_default(), flash })) },
            None => ConfiguredDevice::new("graphics", DeviceModel::Empty),
        });
    }
    if heads.is_some() {
        cfg.devices.push(ConfiguredDevice::new("gio0", DeviceModel::Board(BoardConfig { resolution: res.unwrap_or_default(), ..Default::default() })));
    }
    Ok(())
}

/// A bare drive id (`16 * controller + id`), the only way older versions
/// named a drive to load or eject.
pub(crate) fn drive_number(spec: &str) -> Option<usize> {
    let n = spec.parse().ok()?;
    eprintln!("iris: drive id {n} is deprecated; name the drive by slot, such as scsi{}/t{}", n / 16, n % 16);
    Some(n)
}

/// The current form of every older key: `[scsi.N]` tables become `[[devices]]` entries at `scsiC/tN`, and the old `[o2]`
/// spellings move to the shared keys that are still at their defaults.
pub(super) fn fold(cfg: &mut MachineConfig) -> Result<(), String> {
    if let Some(tables) = cfg.legacy_scsi.take() {
        let mut keys: Vec<(u8, toml::Table)> = Vec::new();
        for (key, t) in tables {
            let n = key.parse::<u8>().map_err(|_| format!("[scsi.{key}]: the key is not a SCSI id"))?;
            let toml::Value::Table(t) = t else { return Err(format!("[scsi.{key}] is not a table")) };
            keys.push((n, t));
        }
        keys.sort_by_key(|k| k.0);
        for (key, t) in keys {
            cfg.devices.push(scsi_table(key, t)?);
        }
    }
    let mut moved: Vec<&str> = Vec::new();
    if let Some(prom) = cfg.o2.prom.take() {
        if cfg.prom == DEFAULT_PROM { cfg.prom = prom; }
        moved.push("o2.prom -> prom");
    }
    if let Some(flash) = cfg.o2.flash.take() {
        if cfg.nvram == DEFAULT_NVRAM { cfg.nvram = flash; }
        moved.push("o2.flash -> nvram");
    }
    if let Some(mb) = cfg.o2.ram_mb.take() {
        match crate::dev::crime_mem::kits_for(mb) {
            Ok(kits) if cfg.banks == default_banks() => {
                cfg.banks = [0; MAX_BANKS];
                cfg.banks[..kits.len()].copy_from_slice(&kits);
            }
            Ok(_) => {}
            Err(e) => eprintln!("iris: o2.ram_mb = {mb}: {e}; ignored"),
        }
        moved.push("o2.ram_mb -> banks");
    }
    if let Some(kits) = cfg.o2.dimms.take() {
        if cfg.banks == default_banks() && !kits.is_empty() && kits.len() <= MAX_BANKS {
            cfg.banks = [0; MAX_BANKS];
            cfg.banks[..kits.len()].copy_from_slice(&kits);
        }
        moved.push("o2.dimms -> banks");
    }
    if !cfg.o2.disks.is_empty() {
        for d in std::mem::take(&mut cfg.o2.disks) {
            let kind = if d.cdrom { ScsiKind::Cdrom } else { ScsiKind::Disk };
            let dev = ScsiDeviceConfig { path: d.path, cdrom: d.cdrom, kind_field: kind, overlay: d.overlay, ..Default::default() };
            cfg.devices.push(ConfiguredDevice::new(&format!("scsi{}/t{}", d.bus, d.id), DeviceModel::Scsi(dev)));
        }
        moved.push("o2.disks -> [[devices]]");
    }
    if let Some(port) = cfg.o2.serial_port.take() {
        if cfg.serial_port_a.is_none() { cfg.serial_port_a = Some(port); }
        moved.push("o2.serial_port -> serial_port_a");
    }
    if let Some(log) = cfg.o2.serial_log.take() {
        if cfg.serial_log.is_none() { cfg.serial_log = Some(log); }
        moved.push("o2.serial_log -> serial_log");
    }
    if let Some(graphics) = cfg.o2.graphics.take() {
        if !graphics { cfg.headless = true; }
        moved.push("o2.graphics -> headless (inverted)");
    }
    if let Some(mhz) = cfg.o2.cpu_mhz.take() {
        if cfg.clock.fixed_mhz.is_none() { cfg.clock.fixed_mhz = Some(mhz as f64 / 2.0); }
        moved.push("o2.cpu_mhz -> [clock] fixed_mhz (half: CP0 Count)");
    }
    if let Some(net) = cfg.o2.net.take() {
        if !net { cfg.network.mode = NetMode::Off; }
        moved.push("o2.net -> [network] mode = \"off\"");
    }
    if !cfg.o2.forwards.is_empty() {
        let forwards = std::mem::take(&mut cfg.o2.forwards);
        cfg.port_forward.extend(forwards);
        moved.push("o2.forwards -> [[port_forward]]");
    }
    if !moved.is_empty() {
        eprintln!(
            "iris: deprecated [o2] keys folded into the shared settings ({}); move them in the config to silence this",
            moved.join(", ")
        );
    }
    Ok(())
}

/// One `[scsi.KEY]` table: keys 1-15 are ids on `controller` (default 0),
/// and 16 + id names controller 1 directly.
fn scsi_table(key: u8, mut t: toml::Table) -> Result<ConfiguredDevice, String> {
    let controller = match t.remove("controller") {
        None => 0,
        Some(toml::Value::Integer(c @ 0..=255)) => c as u8,
        Some(v) => return Err(format!("[scsi.{key}] controller = {v} is not a controller number")),
    };
    let (controller, id) = match key {
        16.. if controller != 0 && controller != key / 16 => {
            return Err(format!("[scsi.{key}]: the key already puts the target on controller {}; controller = {controller} contradicts it", key / 16))
        }
        16.. => (key / 16, key % 16),
        _ => (controller, key),
    };
    let mut dev: ScsiDeviceConfig = toml::Value::Table(t).try_into().map_err(|e: toml::de::Error| format!("[scsi.{key}]: {}", e.message()))?;
    dev.kind_field = dev.kind();
    dev.cdrom = dev.is_cdrom();
    let e = ConfiguredDevice::new(&format!("scsi{controller}/t{id}"), DeviceModel::Scsi(dev));
    eprintln!("iris: [scsi.{key}] is an older spelling of [[devices]] type = \"{}\" at = \"{}\"", e.device.spec().name, e.at.as_deref().unwrap_or(""));
    Ok(e)
}

/// `--scsiN PATH` and `--cdromN ISO`: a disk at `scsi0/tN`, which replaces
/// the image of whatever is there once the machine is known.
pub(super) fn shorthand(cfg: &mut MachineConfig, id: u8, path: String, cdrom: bool, discs: Vec<String>) {
    let kind = if cdrom { ScsiKind::Cdrom } else { ScsiKind::Disk };
    let dev = ScsiDeviceConfig { path, cdrom, kind_field: kind, discs, ..Default::default() };
    cfg.legacy_shorthands.push(ConfiguredDevice::new(&format!("scsi0/t{id}"), DeviceModel::Scsi(dev)));
}

/// Fold each shorthand into the device already at its slot (the image and changer list change, the rest stays), else add it.
pub(super) fn merge_shorthands(cfg: &mut MachineConfig) {
    for short in std::mem::take(&mut cfg.legacy_shorthands) {
        let placement = DevicePlacement::lenient(cfg, &cfg.devices);
        let mine = short.at.as_deref().and_then(|at| {
            let (alias, target) = at.split_once('/')?;
            Some(crate::topology::slot_alias(&placement.slots, alias)?.extend(&target.parse().ok()?))
        });
        let old = (0..cfg.devices.len()).find(|&i| placement.at(i).is_some() && placement.at(i) == mine.as_ref());
        match (&short.device, old) {
            (DeviceModel::Scsi(new), Some(i)) if cfg.devices[i].device.scsi().is_some() => {
                let d = cfg.devices[i].device.scsi_mut().expect("checked");
                d.path = new.path.clone();
                (d.cdrom, d.kind_field) = (new.cdrom, new.kind_field);
                if !new.discs.is_empty() {
                    d.discs = new.discs.clone();
                }
                cfg.devices[i].default = false;
            }
            _ => cfg.devices.push(short),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn places(cfg: &MachineConfig) -> Vec<String> {
        let p = DevicePlacement::new(cfg).unwrap();
        (0..cfg.devices.len()).filter_map(|i| Some(p.at(i)?.to_string())).collect()
    }

    /// `[scsi.17]` is id 1 on controller 1; `controller = 1` on a low key
    /// still works; a key that names a controller cannot be contradicted.
    #[test]
    fn scsi_tables_become_devices_at_their_controller_alias() {
        let cfg = MachineConfig::from_toml_str(r#"
            [machine]
            profile = "indigo2_ip22"
            [scsi.1]
            path = "internal.raw"
            [scsi.17]
            path = "external.raw"
            [scsi.4]
            path = "cd.iso"
            cdrom = true
            controller = 1
        "#).expect("parse");
        let at: Vec<_> = cfg.devices.iter().filter_map(|d| Some((d.at.clone()?, d.device.spec().name))).collect();
        assert_eq!(at, [("scsi0/t1".into(), "hdd"), ("scsi1/t4".into(), "cdrom"), ("scsi1/t1".into(), "hdd")]);
        assert_eq!(places(&cfg), ["hpc3/c0/t1", "hpc3/c1/t4", "hpc3/c1/t1"]);
        assert_eq!(cfg.disks().iter().map(|d| d.0).collect::<Vec<_>>(), [1, 17, 20]);
        let e = MachineConfig::from_toml_str("[scsi.18]\ncontroller = 2\npath = \"x\"").unwrap_err().to_string();
        assert!(e.contains("contradicts"), "{e}");
        let e = MachineConfig::from_toml_str("[scsi.4]\ncdrom = true\n[scsi.20]\ncdrom = true\n[machine]\nprofile = \"indigo2_ip22\"").unwrap();
        let mut both = e.clone();
        both.devices[0].at = Some("scsi1/t4".into());
        assert!(both.validate().unwrap_err().contains("hpc3/c1/t4 is taken"));
        assert!(e.validate().is_ok());
    }

    /// An older O2 config's `o2.disks` become devices, in place of the
    /// default disks, and are gone from the section afterwards.
    #[test]
    fn deprecated_o2_disks_fold_into_devices() {
        let cfg = MachineConfig::from_toml_str(r#"
            [machine]
            profile = "o2_ip32"
            [o2]
            disks = [ { id = 1, path = "sys.raw", overlay = true },
                      { id = 4, path = "cd.iso", cdrom = true, bus = 1 } ]
        "#).expect("parse");
        assert!(cfg.o2.disks.is_empty());
        assert_eq!(cfg.devices.len(), 2, "no default disks beside them");
        assert!(cfg.disk(1).unwrap().overlay);
        assert!(cfg.disk(20).unwrap().is_cdrom());
        assert_eq!(places(&cfg), ["pci0/1/t1", "pci0/2/t4"]);
        let s = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(!s.contains("disks") && !s.contains("[scsi"), "older keys are never written back:\n{s}");
    }

    /// The machine is picked after the file is read, so a table follows it.
    #[test]
    fn a_scsi_table_lands_on_the_machine_the_command_line_picks() {
        let file = MachineConfig::from_toml_str("[scsi.1]\npath = \"root.raw\"\n").unwrap();
        let mut cfg = Cli::parse_from(["iris", "--machine", "indigo2_ip22"]).apply(file);
        cfg.resolve_slots().unwrap();
        assert_eq!(places(&cfg), ["hpc3/c0/t1"]);
    }

    /// `--cdrom4` loads its image into the CD-ROM the file has at target 4,
    /// keeping the drive's other settings.
    #[test]
    fn cdrom4_replaces_the_image_of_the_drive_at_target_4() {
        let file = MachineConfig::from_toml_str(r#"
            [[devices]]
            type = "cdrom"
            at = "hpc3/c0/t4"
            path = "first.iso"
            overlay = true
        "#).unwrap();
        let mut cfg = Cli::parse_from(["iris", "--cdrom4", "other.iso"]).apply(file);
        cfg.resolve_slots().unwrap();
        assert_eq!(cfg.devices.iter().filter(|d| d.device.scsi().is_some()).count(), 1);
        let cd = cfg.disk(4).unwrap();
        assert_eq!((cd.path.as_str(), cd.overlay, cd.is_cdrom()), ("other.iso", true, true));
        // With no drive there, it adds one.
        let mut cfg = Cli::parse_from(["iris", "--scsi2", "two.raw"]).apply(MachineConfig::from_toml_str("devices = []").unwrap());
        cfg.resolve_slots().unwrap();
        assert_eq!(places(&cfg), ["hpc3/c0/t2"]);
    }

    /// The keys that placement replaced are no longer read.
    #[test]
    fn the_dropped_keys_are_unknown_fields() {
        for (text, key) in [
            ("[[xio_board]]\nkind = \"pci-cage\"\n", "xio_board"),
            ("[nvme]\npath = \"d.img\"\n", "nvme"),
            ("[fibre_channel]\n", "fibre_channel"),
            ("[graphics]\nnode = 1\n", "node"),
            ("[graphics]\nwidget = 9\n", "widget"),
            ("[machine]\naudio_card = true\n", "audio_card"),
            ("[machine]\nprofile = \"tezro_ip53\"\n[tezro]\naudio = true\n", "audio"),
            ("[virtio_serial]\nenabled = true\nbridge_slot = 5\n", "bridge_slot"),
        ] {
            let e = MachineConfig::from_toml_str(text).err().map(|e| e.to_string()).unwrap_or_else(|| format!("{key} parsed"));
            assert!(e.contains("unknown field") && e.contains(key), "{key}: {e}");
            assert!(e.contains(&format!("`{key}` is no longer read: write ")) && e.contains("[[devices]]"), "{key}: the error says what replaces it:\n{e}");
        }
        // A key of the same name in another section is only a typo.
        for text in ["banana = 1", "[machine]\naudio = true\n", "[tezro]\nnode = 1\n"] {
            let e = MachineConfig::from_toml_str(text).unwrap_err().to_string();
            assert!(e.contains("unknown field") && !e.contains("no longer read"), "{text}: {e}");
        }
    }

    /// A front end that folds the flags when it reads a file sees the cards, and the run's own fold finds
    /// nothing left to add.
    #[test]
    fn folding_the_cards_twice_changes_nothing() {
        let mut cfg = MachineConfig::from_toml_str("[graphics]\nheads = 2\n[virtio_serial]\nenabled = true\n").unwrap();
        cfg.fold_cards().unwrap();
        let names = |c: &MachineConfig| c.devices.iter().map(|d| d.device.spec().name).collect::<Vec<_>>();
        let once = names(&cfg);
        assert!(once.contains(&"newport") && once.contains(&"virtio_serial"), "{once:?}");
        cfg.resolve_slots().unwrap();
        assert_eq!(names(&cfg), once);
    }

    /// A file with no disks gets the two defaults; one that names a disk gets
    /// exactly that one.
    #[test]
    fn default_disks_only_when_the_file_names_none() {
        let empty = MachineConfig::from_toml_str("").unwrap();
        assert_eq!(places(&empty), ["hpc3/c0/t1", "hpc3/c0/t4"]);
        assert!(empty.devices.iter().all(|d| d.default));
        let one = MachineConfig::from_toml_str("[[devices]]\ntype = \"hdd\"\nat = \"scsi0/t2\"\npath = \"mine.raw\"\n").unwrap();
        assert_eq!(places(&one), ["hpc3/c0/t2"]);
        assert!(MachineConfig::from_toml_str("[scsi]\n").unwrap().devices.iter().all(|d| d.device.scsi().is_none()), "an empty [scsi] is no disks");
        let text = toml::to_string(&MachineConfig { devices: vec![], ..MachineConfig::default() }).unwrap();
        assert!(MachineConfig::from_toml_str(&text).unwrap().devices.iter().all(|d| d.device.scsi().is_none()), "a machine saved with none reads back with none");
    }

    /// A file that names only cards still names its devices: the default disks do not come back, as
    /// they would for a saved machine whose disks were all removed.
    #[test]
    fn a_file_with_only_cards_gets_no_default_disks() {
        let card = "[machine]\nprofile = \"octane_ip30\"\n[[devices]]\ntype = \"nvme\"\npath = \"n.img\"\n";
        let cfg = MachineConfig::from_toml_str(card).unwrap();
        assert!(cfg.devices.iter().all(|d| d.device.scsi().is_none()), "no default disks beside the card");
        let text = toml::to_string(&MachineConfig { devices: cfg.devices.clone(), ..cfg }).unwrap();
        let back = MachineConfig::from_toml_str(&text).unwrap();
        assert!(back.devices.iter().all(|d| d.device.scsi().is_none()), "saved and read back:\n{text}");
        assert_eq!(back.devices.len(), 1);
    }

    fn slots(cfg: &MachineConfig) -> Vec<(String, &'static str)> {
        let p = DevicePlacement::new(cfg).unwrap();
        p.placed().filter(|(_, at)| at.gio_slot().is_some()).map(|(d, at)| (at.to_string(), d.device.spec().name)).collect()
    }

    /// The older flags put a card in the slot they named, on the Indy and on
    /// the Indigo2, and leave nothing behind to be written back.
    #[test]
    fn gio_flags_become_cards_in_their_slots() {
        for profile in ["indy_ip24", "indigo2_ip22"] {
            let cfg = MachineConfig::from_toml_str(&format!(
                "[machine]\nprofile = \"{profile}\"\n[ultra64]\nenabled = true\n[virtio_serial]\nenabled = true\n[[virtio_serial.port]]\nname = \"a\"\n"
            )).unwrap();
            let mut cfg = cfg;
            cfg.resolve_slots().unwrap();
            assert_eq!(slots(&cfg), [("gio0".into(), "ultra64"), ("gio1".into(), "virtio_serial")], "{profile}");
            assert!(!cfg.ultra64.enabled && cfg.virtio_serial.is_default());
            let Some(DeviceModel::VirtioSerial(v)) = cfg.devices.iter().map(|d| &d.device).find(|d| matches!(d, DeviceModel::VirtioSerial(_))) else { panic!("not virtio") };
            assert_eq!(v.ports[0].name.as_deref(), Some("a"), "the ports move with the card");
        }
        let mut cfg = MachineConfig::from_toml_str("[virtio_9p]\nenabled = true\nslot = \"exp0\"\npath = \"/x\"\n").unwrap();
        cfg.resolve_slots().unwrap();
        assert_eq!(slots(&cfg), [("gio0".into(), "virtio_9p")]);
    }

    #[test]
    fn two_flags_for_one_slot_are_a_slot_clash() {
        let mut cfg = MachineConfig::from_toml_str("test_device = true\n[ultra64]\nenabled = true\n").unwrap();
        let e = cfg.resolve_slots().unwrap_err();
        assert!(e.contains("test_device at \"gio0\": gio0 is taken by the ultra64"), "{e}");
        assert_eq!(cfg.validate().unwrap_err(), e);
        let mut cfg = MachineConfig::from_toml_str("test_device = true\n[graphics]\nheads = 2\n").unwrap();
        assert!(cfg.resolve_slots().unwrap_err().contains("gio0 is taken by the newport"));
    }

    /// The GUI's reader turns the old keys into entries; `heads = 1` and `resolution = "guest"` are no keys.
    #[test]
    fn the_older_graphics_keys_are_refused_and_the_gui_reads_them_as_entries() {
        for (text, entry) in [
            ("[graphics]\nboard = \"ir2:2rm\"\n", "[[devices]]\ntype = \"kona\"\nmodel = \"ir2\"\nraster_managers = 2\n"),
            ("[graphics]\nboard = \"v12\"\n", "[[devices]]\ntype = \"vpro\"\nmodel = \"v12\"\n"),
            ("[graphics]\nboard = \"solidimpact\"\n", "[[devices]]\ntype = \"impact\"\nmodel = \"solid\"\n"),
            ("[graphics]\nboard = \"xz\"\n", "[[devices]]\ntype = \"express\"\nmodel = \"xz\"\n"),
            ("[graphics]\nboard = \"none\"\n", "[[devices]]\ntype = \"none\"\nat = \"graphics\"\n"),
            ("[graphics]\nheads = 2\n", "[[devices]]\ntype = \"newport\"\nat = \"gio0\"\n"),
            ("[graphics]\nresolution = \"1024x768\"\n", "[[devices]]\ntype = \"newport\"\nresolution = \"1024x768\"\n"),
            ("[machine]\nprofile = \"onyx2_ip27\"\n[graphics]\nboard = \"ir2\"\nflash = \"p.bin\"\n", "[[devices]]\ntype = \"kona\"\nflash = \"p.bin\"\nmodel = \"ir2\"\n"),
        ] {
            let e = MachineConfig::parse_toml(text).unwrap_err().to_string();
            assert!(e.contains("moved to [[devices]] entries") && e.contains("HELP.md"), "{text}:\n{e}");
            let cfg = MachineConfig::from_toml_str(text).unwrap();
            let entries: String = cfg.devices.iter().filter(|d| matches!(d.device, DeviceModel::Board(_) | DeviceModel::Empty)).map(|d| format!("[[devices]]\n{}", toml::to_string(d).unwrap())).collect();
            assert!(entries.contains(entry), "{text}:\n{entries}");
        }
        let defaults = "[graphics]\nheads = 1\nresolution = \"guest\"\n";
        assert!(MachineConfig::parse_toml(defaults).unwrap().devices.iter().all(|d| !matches!(d.device, DeviceModel::Board(_))));
        assert!(MachineConfig::from_toml_str("[graphics]\nboard = \"v14\"\n").unwrap_err().to_string().contains("not a graphics board"));
    }

    /// What the GUI wrote for every machine still loads: its own default board or a board for a machine with no slot is no
    /// entry, `flash` alone belongs on the default board, and a V12 with one C-brick goes on the I-brick.
    #[test]
    fn the_guis_saved_boards_load_on_every_machine() {
        let load = |profile: &str, graphics: &str| MachineConfig::from_toml_str(&format!("[machine]\nprofile = \"{profile}\"\n[graphics]\n{graphics}")).unwrap();
        for (profile, board) in [("o2_ip32", "newport"), ("origin200_ip29", "none"), ("origin2000_ip27", "none"), ("origin3200_ip35", "none"), ("indy_ip24", "newport"), ("octane_ip30", "solidimpact")] {
            let cfg = load(profile, &format!("board = \"{board}\"\nheads = 1\nresolution = \"guest\"\n"));
            assert!(cfg.devices.iter().all(|d| !matches!(d.device, DeviceModel::Board(_) | DeviceModel::Empty)), "{profile}");
            DevicePlacement::new(&cfg).unwrap_or_else(|e| panic!("{profile}: {e}"));
        }
        let kona = load("onyx3000_ip35", "flash = \"p.bin\"\n").graphics_entry().unwrap();
        assert_eq!((kona.model, kona.flash.as_deref()), (GraphicsBoard::Ir3, Some("p.bin")));
        let cfg = MachineConfig::from_toml_str("[machine]\nprofile = \"onyx3000_ip35\"\nnodes = 1\n[graphics]\nboard = \"v12\"\n").unwrap();
        assert_eq!(DevicePlacement::new(&cfg).unwrap().graphics().map(|at| at.to_string()).as_deref(), Some("xio11"));
        for name in ["xl", "gr2_xz", "vpro:v12", "impact:solid", "ir2x2", "maximumimpact"] {
            assert!(MachineConfig::from_toml_str(&format!("[graphics]\nboard = \"{name}\"\n")).is_ok(), "{name}");
        }
    }

    #[test]
    fn a_machine_without_gio_slots_refuses_a_second_head() {
        let octane = "[machine]\nprofile = \"octane_ip30\"\n";
        let mut cfg = MachineConfig::from_toml_str(&format!("{octane}[graphics]\nheads = 2\n")).unwrap();
        assert!(cfg.resolve_slots().unwrap_err().contains("has no slot gio0"));
        assert!(MachineConfig::from_toml_str("[graphics]\nheads = 3\n").unwrap_err().to_string().contains("heads must be 1 or 2"));
    }
}
