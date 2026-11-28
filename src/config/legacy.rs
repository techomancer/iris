//! The keys older versions of iris wrote, each declared once in [`OLD_KEYS`]: read and turned into its current form, or
//! refused with what to write. `--scsiN`, `--cdromN` and the bare drive number live here too.

use super::*;
use crate::devices::{ConfiguredDevice, DeviceModel, TestDeviceConfig};
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

/// When a row's fold runs: as the file is read, or once the command line has picked the machine.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum When {
    Load,
    Placed,
}

pub(super) enum How {
    /// Move the key into its current form; true when the config had it.
    Fold(fn(&mut MachineConfig) -> Result<bool, String>),
    /// An unknown field in `sections` (none: anywhere), whose replacement needs a slot choice.
    Refuse { sections: &'static [&'static str], instead: &'static str },
}

/// An older key, declared once: its translation and its error both come from the row.
pub(super) struct OldKey {
    /// The key as a file writes it, which the run's warning names.
    key: &'static str,
    when: When,
    how: How,
}

const fn fold(key: &'static str, when: When, f: fn(&mut MachineConfig) -> Result<bool, String>) -> OldKey {
    OldKey { key, when, how: How::Fold(f) }
}

const fn refuse(key: &'static str, sections: &'static [&'static str], instead: &'static str) -> OldKey {
    OldKey { key, when: When::Load, how: How::Refuse { sections, instead } }
}

fn took<T>(old: Option<T>, f: impl FnOnce(T)) -> Result<bool, String> {
    Ok(old.map(f).is_some())
}

fn banks_from(c: &mut MachineConfig, kits: &[u32]) {
    if c.banks == default_banks() && !kits.is_empty() && kits.len() <= MAX_BANKS {
        c.banks = [0; MAX_BANKS];
        c.banks[..kits.len()].copy_from_slice(kits);
    }
}

/// A card flag as a `[[devices]]` entry at gio0, on a machine that has GIO slots; the flag stays set where it has none.
fn gio_card(c: &mut MachineConfig, on: bool, card: impl FnOnce(&mut MachineConfig) -> Result<DeviceModel, String>) -> Result<bool, String> {
    if !on || !slots_take(&c.machine.profile.slots(), DeviceFamily::Gio) {
        return Ok(false);
    }
    let card = card(c)?;
    c.devices.push(ConfiguredDevice::new("gio0", card));
    Ok(true)
}

/// A virtio flag as a `[[devices]]` entry in the GIO slot it named, else wherever the machine has a slot for it.
fn virtio_card(c: &mut MachineConfig, slot: Option<VirtioGioSlot>, card: DeviceModel) -> Result<bool, String> {
    let gio = slots_take(&c.machine.profile.slots(), DeviceFamily::Gio);
    let at = gio.then(|| if slot == Some(VirtioGioSlot::Exp0) { "gio0" } else { "gio1" }.to_string());
    c.devices.push(ConfiguredDevice { at, ..ConfiguredDevice::new("", card) });
    Ok(true)
}

fn heads(c: &mut MachineConfig) -> Result<bool, String> {
    match c.graphics.heads {
        1 => return Ok(false),
        2 => {}
        n => return Err(format!("graphics.heads {n} is invalid (valid: 1, 2)")),
    }
    if !slots_take(&c.machine.profile.slots(), DeviceFamily::Gio) {
        return Err(format!("graphics.heads = 2: the {} has no GIO slot for a second head", c.machine.profile.product_name().unwrap_or(c.machine.profile.label())));
    }
    c.graphics.heads = 1;
    if !c.headless {
        c.devices.push(ConfiguredDevice::new("gio0", DeviceModel::Newport));
    }
    Ok(true)
}

/// Every key an older iris wrote and a current one still reads, or refuses with what to write instead.
pub(super) const OLD_KEYS: &[OldKey] = &[
    fold("scsi", When::Load, scsi_tables),
    fold("o2.prom", When::Load, |c| took(c.o2.prom.take(), |p| if c.prom == DEFAULT_PROM { c.prom = p })),
    fold("o2.flash", When::Load, |c| took(c.o2.flash.take(), |f| if c.nvram == DEFAULT_NVRAM { c.nvram = f })),
    fold("o2.ram_mb", When::Load, |c| {
        took(c.o2.ram_mb.take(), |mb| match crate::dev::crime_mem::kits_for(mb) {
            Ok(kits) => banks_from(c, &kits),
            Err(e) => eprintln!("iris: o2.ram_mb = {mb}: {e}; ignored"),
        })
    }),
    fold("o2.dimms", When::Load, |c| took(c.o2.dimms.take(), |kits| banks_from(c, &kits))),
    fold("o2.disks", When::Load, |c| {
        took((!c.o2.disks.is_empty()).then(|| std::mem::take(&mut c.o2.disks)), |disks| {
            for d in disks {
                let kind = if d.cdrom { ScsiKind::Cdrom } else { ScsiKind::Disk };
                let dev = ScsiDeviceConfig { path: d.path, cdrom: d.cdrom, kind_field: kind, overlay: d.overlay, ..Default::default() };
                c.devices.push(ConfiguredDevice::new(&format!("scsi{}/t{}", d.bus, d.id), DeviceModel::Scsi(dev)));
            }
        })
    }),
    fold("o2.serial_port", When::Load, |c| took(c.o2.serial_port.take(), |p| c.serial_port_a = c.serial_port_a.or(Some(p)))),
    fold("o2.serial_log", When::Load, |c| took(c.o2.serial_log.take(), |l| c.serial_log = c.serial_log.take().or(Some(l)))),
    fold("o2.graphics", When::Load, |c| took(c.o2.graphics.take(), |on| c.headless |= !on)),
    // CP0 Count runs at half the CPU clock.
    fold("o2.cpu_mhz", When::Load, |c| took(c.o2.cpu_mhz.take(), |mhz| c.clock.fixed_mhz = c.clock.fixed_mhz.or(Some(f64::from(mhz) / 2.0)))),
    fold("o2.net", When::Load, |c| took(c.o2.net.take(), |on| if !on { c.network.mode = NetMode::Off })),
    fold("o2.forwards", When::Load, |c| took((!c.o2.forwards.is_empty()).then(|| std::mem::take(&mut c.o2.forwards)), |f| c.port_forward.extend(f))),
    fold("impact.gfx", When::Load, |c| {
        took(c.impact.take(), |impact| {
            let board = match impact.gfx.as_deref().map(str::to_ascii_lowercase).as_deref() {
                Some("solid") => Some(GraphicsBoard::SolidImpact),
                Some("high") => Some(GraphicsBoard::HighImpact),
                Some("max") => Some(GraphicsBoard::MaxImpact),
                _ => None,
            };
            c.graphics.board = c.graphics.board.or(board);
        })
    }),
    fold("octane.graphics", When::Load, |c| match c.octane.graphics.take() {
        Some(old) => Err(format!("octane.graphics has moved: set the board in [graphics] as board = \"{}\"", old.trim())),
        None => Ok(false),
    }),
    fold("graphics.heads", When::Placed, heads),
    fold("ultra64", When::Placed, |c| {
        gio_card(c, c.ultra64.enabled, |c| {
            c.ultra64.enabled = false;
            Ok(DeviceModel::Ultra64)
        })
    }),
    fold("test_device", When::Placed, |c| {
        if c.test_device_run_config.is_some() && !c.test_device {
            return Err("test_device_run_config needs test_device".into());
        }
        gio_card(c, c.test_device, |c| {
            let run = TestDeviceConfig { dump: c.test_device_dump.take(), run_config: c.test_device_run_config.take() };
            run.run()?;
            c.test_device = false;
            Ok(DeviceModel::TestDevice(run))
        })
    }),
    fold("virtio_serial", When::Placed, |c| match c.virtio_serial.enabled {
        true => {
            let v = std::mem::take(&mut c.virtio_serial);
            virtio_card(c, v.slot, DeviceModel::VirtioSerial(v))
        }
        false => Ok(false),
    }),
    fold("virtio_9p", When::Placed, |c| match c.virtio_9p.enabled {
        true => {
            let v = std::mem::take(&mut c.virtio_9p);
            virtio_card(c, v.slot, DeviceModel::Virtio9p(v))
        }
        false => Ok(false),
    }),
    refuse("xio_board", &[], "a `pci_cage` or `pci_shoehorn` [[devices]] entry at the XIO slot"),
    refuse("nvme", &[], "an `nvme` [[devices]] entry"),
    refuse("fibre_channel", &[], "a `qla2200` [[devices]] entry, with its disks as `hdd` entries at CARD/lN"),
    refuse("audio_card", &[], "an `ice1724` [[devices]] entry"),
    refuse("audio", &["fuel", "tezro", "origin3200", "onyx3000"], "an `ice1724` [[devices]] entry"),
    refuse("node", &["graphics"], "the `at` of a `graphics` [[devices]] entry"),
    refuse("widget", &["graphics"], "the `at` of a `graphics` [[devices]] entry"),
    refuse("bridge_slot", &[], "the `at` of the virtio [[devices]] entry"),
];

/// Run each row's fold of `when`, noting the keys the config had.
pub(super) fn fold_all(cfg: &mut MachineConfig, when: When) -> Result<(), String> {
    for old in OLD_KEYS.iter().filter(|o| o.when == when) {
        if let How::Fold(f) = old.how {
            if f(cfg)? {
                cfg.old_keys.push(old.key);
            }
        }
    }
    Ok(())
}

impl MachineConfig {
    /// The one line a run says when it translated older keys, naming them.
    pub fn old_keys_warning(&self, file: &str) -> Option<String> {
        (!self.old_keys.is_empty()).then(|| format!("iris: {file}: older keys or options ({}), translated for this run; `--config-dump FILE` writes the new form", self.old_keys.join(", ")))
    }
}

/// A parse error for a refused key, with what to write instead; any other error as it was.
pub(super) fn name_replacement(e: toml::de::Error, text: &str) -> toml::de::Error {
    let key = e.message().strip_prefix("unknown field `").and_then(|m| m.split('`').next());
    let section = || {
        let before = text.get(..e.span()?.start)?;
        let header = before.lines().rev().find_map(|l| l.trim().strip_prefix('['))?;
        header.trim_start_matches('[').split(']').next()
    };
    let refused = OLD_KEYS.iter().find_map(|o| match o.how {
        How::Refuse { sections, instead } if Some(o.key) == key && (sections.is_empty() || section().is_some_and(|s| sections.contains(&s))) => Some((o.key, instead)),
        _ => None,
    });
    match refused {
        Some((key, instead)) => <toml::de::Error as serde::de::Error>::custom(format!("{e}\n`{key}` is no longer read: write {instead}")),
        None => e,
    }
}

/// A bare drive id (`16 * controller + id`), the only way older versions
/// named a drive to load or eject.
pub(crate) fn drive_number(spec: &str) -> Option<usize> {
    let n = spec.parse().ok()?;
    eprintln!("iris: drive id {n} is deprecated; name the drive by slot, such as scsi{}/t{}", n / 16, n % 16);
    Some(n)
}

/// The older `[scsi.N]` tables as `[[devices]]` entries at `scsiC/tN`.
fn scsi_tables(cfg: &mut MachineConfig) -> Result<bool, String> {
    let Some(tables) = cfg.legacy_scsi.take() else { return Ok(false) };
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
    Ok(true)
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
    Ok(ConfiguredDevice::new(&format!("scsi{controller}/t{id}"), DeviceModel::Scsi(dev)))
}

/// `--scsiN PATH` and `--cdromN ISO`: a disk at `scsi0/tN`, which replaces
/// the image of whatever is there once the machine is known.
pub(super) fn shorthand(cfg: &mut MachineConfig, id: u8, path: String, cdrom: bool, discs: Vec<String>) {
    let kind = if cdrom { ScsiKind::Cdrom } else { ScsiKind::Disk };
    let dev = ScsiDeviceConfig { path, cdrom, kind_field: kind, discs, ..Default::default() };
    if !cfg.old_keys.contains(&"--scsiN/--cdromN") {
        cfg.old_keys.push("--scsiN/--cdromN");
    }
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
        assert_eq!(cfg.old_keys, ["--scsiN/--cdromN"]);
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

    /// An Indigo2 config from before slots and devices: every key in it is translated, the run says so once and names
    /// them, and a dump of the result is the new form, which reads back with nothing to say.
    #[test]
    fn an_older_indigo2_config_translates_and_dumps_the_new_form() {
        let old = r#"
            [machine]
            profile = "indigo2_ip22"
            [graphics]
            heads = 2
            [impact]
            gfx = "max"
            [virtio_serial]
            enabled = true
            [scsi.1]
            path = "irix.raw"
            [scsi.20]
            path = "cd.iso"
            cdrom = true
        "#;
        let mut cfg = MachineConfig::from_toml_str(old).unwrap();
        cfg.fold_cards().unwrap();
        let cfg = cfg.resolved().unwrap();
        assert_eq!(cfg.old_keys, ["scsi", "impact.gfx", "graphics.heads", "virtio_serial"]);
        let line = cfg.old_keys_warning("indigo2.toml").unwrap();
        assert!(line.contains("indigo2.toml") && line.contains("scsi, impact.gfx, graphics.heads, virtio_serial") && line.contains("--config-dump FILE"), "{line}");
        let new = cfg.to_toml(Dump::Minimal).unwrap();
        for gone in ["[scsi", "[impact]", "[virtio_serial]", "heads"] {
            assert!(!new.contains(gone), "{gone} in\n{new}");
        }
        for want in ["board = \"maximpact\"", "type = \"newport\"", "type = \"virtio_serial\"", "at = \"scsi1/t4\"", "type = \"cdrom\""] {
            assert!(new.contains(want), "{want} in\n{new}");
        }
        let again = MachineConfig::from_toml_str(&new).unwrap().resolved().unwrap();
        assert!(again.old_keys.is_empty() && again.old_keys_warning("x").is_none());
        assert_eq!(again.to_toml(Dump::Minimal).unwrap(), new);
    }

    /// The O2's own section and an Octane's section spell what `[machine]` and the shared keys now hold.
    #[test]
    fn an_older_o2_config_translates_its_section() {
        let cfg = MachineConfig::from_toml_str("[o2]\nram_mb = 256\nserial_port = 9000\ngraphics = false\nnet = false\ncpu_mhz = 200\n[machine]\nprofile = \"o2_ip32\"\n").unwrap();
        assert_eq!(cfg.old_keys, ["o2.ram_mb", "o2.serial_port", "o2.graphics", "o2.cpu_mhz", "o2.net"]);
        let new = cfg.resolved().unwrap().to_toml(Dump::Minimal).unwrap();
        for want in ["serial_port_a = 9000", "headless = true", "mode = \"off\"", "fixed_mhz = 100.0"] {
            assert!(new.contains(want), "{want} in\n{new}");
        }
        assert!(!new.contains("[o2]"), "{new}");
        let e = MachineConfig::from_toml_str("[octane]\ngraphics = \"maximpact\"").unwrap_err().to_string();
        assert!(e.contains("[graphics] as board = \"maximpact\""), "{e}");
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
        // Without a window there is no second head to take the slot.
        let mut cfg = MachineConfig::from_toml_str("test_device = true\nheadless = true\n[graphics]\nheads = 2\n").unwrap();
        cfg.resolve_slots().unwrap();
    }

    #[test]
    fn a_machine_without_gio_slots_refuses_a_second_head() {
        let octane = "[machine]\nprofile = \"octane_ip30\"\n";
        let mut cfg = MachineConfig::from_toml_str(&format!("{octane}[graphics]\nheads = 2\n")).unwrap();
        assert!(cfg.resolve_slots().unwrap_err().contains("no GIO slot"));
        let mut three = MachineConfig::from_toml_str("[graphics]\nheads = 3\n").unwrap();
        assert!(three.resolve_slots().unwrap_err().contains("heads 3 is invalid"));
    }
}
