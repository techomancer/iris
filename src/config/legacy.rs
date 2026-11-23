//! The keys older versions of iris wrote, read with a warning and turned into their current form. `[scsi.N]`, `--scsiN`,
//! `--cdromN` and the bare drive number live only here; the card flags keep their fields in `config.rs`.

use super::*;
use crate::devices::{ConfiguredDevice, DeviceModel};
use crate::topology::DevicePlacement;

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
    if let Some(impact) = cfg.impact.take() {
        let board = match impact.gfx.as_deref().map(str::to_ascii_lowercase).as_deref() {
            Some("solid") => Some(GraphicsBoard::SolidImpact),
            Some("high") => Some(GraphicsBoard::HighImpact),
            Some("max") => Some(GraphicsBoard::MaxImpact),
            _ => None,
        };
        if let Some(board) = board {
            if cfg.graphics.board.is_none() { cfg.graphics.board = Some(board); }
            eprintln!("iris: deprecated [impact] gfx folded into [graphics] board = \"{}\"; move it in the config to silence this", board.as_str());
        }
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
}
