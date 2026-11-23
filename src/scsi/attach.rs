//! Putting the config's SCSI devices on a machine's buses.
//!
//! Every machine walks the same list: find each device's bus in the slot
//! list, make a scratch volume, send a CI run's overlay to the temporary
//! directory, attach the target, and decide what a target that will not
//! attach means. Only two things are the machine's: the buses it binds to
//! the slot list, and, on the machines that have one, how a DaynaPort is built.

use std::path::PathBuf;

use super::ScsiBus;
use crate::config::{DaynaportParams, MachineConfig, ScsiDeviceConfig};
use crate::topology::{scsi_buses, scsi_target, DevicePlacement};

/// One disk or CD-ROM, for a snapshot manifest to record.
pub struct TableRef {
    /// The slot, controller * 16 + id (`config::scsi_slot`).
    pub slot: u8,
    pub is_cdrom: bool,
    pub path: String,
    /// The image's size on the host, 0 if it cannot be read.
    pub size_bytes: u64,
}

/// A `daynaport`. The controller that can hold one builds it, since it
/// needs the machine's network.
pub struct DaynaportRef {
    pub controller: u8,
    pub id: u8,
    pub params: DaynaportParams,
}

/// What attaching left for the machine.
#[derive(Default)]
pub struct Attached {
    /// The `scratch = true` volume the CI socket reads and writes: the
    /// lowest-id one, if several are configured.
    pub scratch: Option<PathBuf>,
    /// Every disk and CD-ROM, by slot, whether or not it attached.
    pub tables: Vec<TableRef>,
    /// The DaynaPorts, not attached.
    pub daynaports: Vec<DaynaportRef>,
}

/// A target that will not attach ends a command-line run, since booting on
/// without it only yields a confusing PROM "no such device" later. An
/// embedder (the GUI sets `IRIS_NO_EXIT_ON_POWEROFF`) must never be torn
/// down by the library, so it skips the one target and boots without it; the
/// GUI's Start preflight already warns about an unreadable disk.
pub fn attach_failed(msg: String, at: &str) -> Result<(), String> {
    if std::env::var_os("IRIS_NO_EXIT_ON_POWEROFF").is_some() {
        eprintln!("iris: warning: {msg}; continuing without {at}");
        Ok(())
    } else {
        Err(msg)
    }
}

/// Attach the target one table describes at `id` of `bus`, logging it as
/// `{log_prefix}: {at}: ...` unless the prefix is `None`. `ci_overlay` is the
/// path a `--ci` run keeps the copy-on-write overlay at. The Fibre Channel
/// adapter's loop disks come through here too, so a disk is attached the same
/// way wherever it sits.
pub fn attach_table(
    bus: &ScsiBus,
    id: u8,
    at: &str,
    dev: &ScsiDeviceConfig,
    ci_overlay: Option<&str>,
    log_prefix: Option<&str>,
) -> Result<(), String> {
    let (path, discs) = dev.media();
    match bus.attach_device(id, &path, dev.is_cdrom(), discs, dev.overlay, ci_overlay, dev.write_cache) {
        Ok(blocks) => {
            if let Some(prefix) = log_prefix {
                if path.is_empty() {
                    eprintln!("{prefix}: {at}: CD-ROM, tray empty");
                } else {
                    eprintln!("{prefix}: {at}: {path} ({blocks} blocks)");
                }
            }
            Ok(())
        }
        Err(e) => {
            let msg = match ci_overlay {
                Some(ov) => format!("could not attach {path} to {at} with its --ci overlay {ov}: {e}"),
                None => format!("could not attach {path} to {at}: {e}"),
            };
            attach_failed(msg, at)
        }
    }
}

/// Attach every SCSI device of `cfg` to the bus of its slot; `buses` are the machine's, in slot-list order. Logs each device as
/// `{log_prefix}: SLOT: ...` unless the prefix is `None`. The error ends the run.
pub fn attach_from_config(cfg: &MachineConfig, log_prefix: Option<&str>, resolved: &DevicePlacement, buses: &[&ScsiBus]) -> Result<Attached, String> {
    let declared = scsi_buses(&resolved.slots).len();
    if buses.len() != declared {
        return Err(format!("the {}'s slot list declares {declared} SCSI buses, which the machine built {}", resolved.machine(), buses.len()));
    }
    let mut targets: Vec<_> = resolved.placed().filter_map(|(d, at)| {
        let (controller, id) = scsi_target(&resolved.slots, at)?;
        Some((crate::config::scsi_slot(controller, id), controller, id, d.device.scsi()?, at.clone()))
    }).collect();
    targets.sort_by_key(|t| t.0);
    let mut out = Attached::default();
    for (key, controller, id, dev, at) in targets {
        let bus = buses[usize::from(controller)];
        let label = at.to_string();
        if dev.is_daynaport() {
            let params = dev.daynaport_params(key).map_err(|e| format!("{label}: {e}"))?;
            out.daynaports.push(DaynaportRef { controller, id, params });
            continue;
        }
        out.tables.push(TableRef {
            slot: key,
            is_cdrom: dev.is_cdrom(),
            path: dev.path.clone(),
            size_bytes: std::fs::metadata(&dev.path).map(|m| m.len()).unwrap_or(0),
        });
        // The file gets an SGI volume header, without which IRIX returns I/O
        // errors on every read of the volume.
        if let Some(path) = dev.prepare_scratch(key) {
            if out.scratch.is_some() {
                println!("Note: multiple scratch SCSI devices configured; CI socket will use the lowest-id one");
            } else {
                out.scratch = Some(path);
            }
        }
        // CI mode isolates each copy-on-write overlay in the temporary
        // directory, so an interactive iris holding one and any number of CI
        // runs can share the image.
        let ci_overlay = (cfg.ci && dev.overlay && !dev.is_cdrom()).then(|| ScsiDeviceConfig::ci_overlay_path(key));
        attach_table(bus, id, &label, dev, ci_overlay.as_deref(), log_prefix)?;
    }
    out.tables.sort_by_key(|t| t.slot);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MachineProfile;
    use crate::devices::{DeviceModel, ConfiguredDevice};
    use crate::scsi::{ScsiIdentities, ScsiIdentity};
    use crate::test_temp::TempPath;
    use crate::topology::DevicePlacement;

    const ID: ScsiIdentity = ScsiIdentity::new(b"TEST    ", b"ATTACH          ", b"1.0 ");

    fn buses() -> [ScsiBus; 2] {
        std::array::from_fn(|_| ScsiBus::new(16, ScsiIdentities { disk: ID, cdrom: ID }))
    }

    fn image(tag: &str) -> TempPath {
        TempPath::file(&format!("attach-{tag}"), &[0u8; 4096])
    }

    fn indigo2(devices: Vec<ConfiguredDevice>) -> MachineConfig {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::Indigo2Ip22;
        cfg.devices = devices;
        cfg
    }

    fn at(at: &str, dev: ScsiDeviceConfig) -> ConfiguredDevice {
        ConfiguredDevice::new(at, DeviceModel::Scsi(dev))
    }

    fn disk(path: &TempPath) -> ScsiDeviceConfig {
        ScsiDeviceConfig { path: path.to_string_lossy().into_owned(), ..Default::default() }
    }

    fn attach(cfg: &MachineConfig, buses: &[ScsiBus; 2]) -> Result<Attached, String> {
        attach_from_config(cfg, None, &DevicePlacement::new(cfg)?, &buses.iter().collect::<Vec<_>>())
    }

    #[test]
    fn devices_land_on_the_bus_bound_at_their_slot() {
        let (a, b) = (image("a"), image("b"));
        let cfg = indigo2(vec![at("scsi0/t1", disk(&a)), at("hpc3/c1/t2", disk(&b)), at("scsi1/t5", disk(&b))]);
        let buses = buses();
        attach(&cfg, &buses).unwrap();
        let ids = |b: &ScsiBus| (0..16).filter(|&i| b.has_target(i)).collect::<Vec<u8>>();
        assert_eq!(ids(&buses[0]), vec![1]);
        assert_eq!(ids(&buses[1]), vec![2, 5]);
    }

    #[test]
    fn a_slot_the_machine_lacks_is_an_error() {
        let a = image("lack");
        let buses = buses();
        let e = attach(&indigo2(vec![at("scsi2/t1", disk(&a))]), &buses).err().unwrap();
        assert!(e.contains("has no slot scsi2/t1"), "{e}");
    }

    #[test]
    fn a_bus_the_map_declares_must_be_built() {
        let cfg = indigo2(vec![]);
        let buses = buses();
        let e = attach_from_config(&cfg, None, &DevicePlacement::new(&cfg).unwrap(), &[&buses[0]]).err().unwrap();
        assert!(e.contains("declares 2 SCSI buses, which the machine built 1"), "{e}");
    }

    #[test]
    fn daynaports_are_returned_for_the_machine_to_attach() {
        let dayna = ScsiDeviceConfig { kind_field: crate::config::ScsiKind::Daynaport, ..Default::default() };
        let buses = buses();
        let got = attach(&indigo2(vec![at("scsi0/t3", dayna)]), &buses).unwrap();
        assert_eq!((got.daynaports[0].controller, got.daynaports[0].id), (0, 3));
        assert!(got.tables.is_empty(), "no image behind a DaynaPort");
        assert!(!buses[0].has_target(3), "and nothing attached");
    }

    #[test]
    fn tables_come_back_by_slot_with_their_kind() {
        let (a, iso) = (image("prov"), image("iso"));
        let cd = ScsiDeviceConfig { cdrom: true, ..disk(&iso) };
        let buses = buses();
        let got = attach(&indigo2(vec![at("scsi1/t4", cd), at("scsi0/t1", disk(&a))]), &buses).unwrap();
        let kinds: Vec<_> = got.tables.iter().map(|t| (t.slot, t.is_cdrom)).collect();
        assert_eq!(kinds, vec![(1, false), (20, true)]);
        assert_eq!(got.tables[0].size_bytes, 4096);
    }
}
