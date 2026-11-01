//! Putting the `[scsi.N]` config tables on a machine's buses.
//!
//! Every machine walks the same list: place each table on a controller and
//! id, make a scratch volume, send a CI run's overlay to the temporary
//! directory, attach the target, and decide what a target that will not
//! attach means. Only two
//! things are the machine's: which bus a controller number names, and, on
//! the machines that have one, how a DaynaPort is built.

use std::path::PathBuf;

use super::ScsiBus;
use crate::config::{DaynaportParams, MachineConfig, ScsiDeviceConfig};

/// One `[scsi.N]` image table, for a snapshot manifest to record.
pub struct TableRef {
    /// The slot, controller * 16 + id (`config::scsi_slot`).
    pub slot: u8,
    pub is_cdrom: bool,
    pub path: String,
    /// The image's size on the host, 0 if it cannot be read.
    pub size_bytes: u64,
}

/// A `kind = "daynaport"` table. The controller that can hold one builds it,
/// since it needs the machine's network.
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
    /// Every disk and CD-ROM table, by slot, whether or not it attached.
    pub tables: Vec<TableRef>,
    /// The DaynaPort tables, not attached.
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

/// Attach every disk and CD-ROM table of `cfg`, logging each as
/// `{log_prefix}: SCSI ID n: ...` unless the prefix is `None`. `route` names
/// the bus a controller number is on, or says why there is none ("an O2 has
/// controllers 0 and 1"). The error is the message to end the run with.
pub fn attach_from_config<'a>(
    cfg: &MachineConfig,
    log_prefix: Option<&str>,
    route: &dyn Fn(u8) -> Result<&'a ScsiBus, String>,
) -> Result<Attached, String> {
    let mut out = Attached::default();
    let mut keys: Vec<u8> = cfg.scsi.keys().copied().collect();
    keys.sort();
    for key in keys {
        let dev = &cfg.scsi[&key];
        let (controller, id) = dev.placement(key);
        let at = crate::config::scsi_label(controller, id);
        if dev.is_daynaport() {
            let params = dev.daynaport_params(key).map_err(|e| format!("{at}: {e}"))?;
            route(controller).map_err(|e| format!("{at}: {e}"))?;
            out.daynaports.push(DaynaportRef { controller, id, params });
            continue;
        }
        out.tables.push(TableRef {
            slot: crate::config::scsi_slot(controller, id),
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
        let bus = route(controller).map_err(|e| format!("{at}: {e}"))?;
        let (path, discs) = dev.media();
        let cdrom = dev.is_cdrom();
        // CI mode isolates each copy-on-write overlay in the temporary
        // directory, so an interactive iris holding one and any number of CI
        // runs can share the image.
        let ci_overlay = (cfg.ci && dev.overlay && !cdrom).then(|| ScsiDeviceConfig::ci_overlay_path(key));
        match bus.attach_device(id, &path, cdrom, discs, dev.overlay, ci_overlay.as_deref(), dev.write_cache) {
            Ok(blocks) => {
                if let Some(prefix) = log_prefix {
                    if path.is_empty() {
                        eprintln!("{prefix}: {at}: CD-ROM, tray empty");
                    } else {
                        eprintln!("{prefix}: {at}: {path} ({blocks} blocks)");
                    }
                }
            }
            Err(e) => {
                let msg = match &ci_overlay {
                    Some(ov) => format!("could not attach {path} to {at} with its --ci overlay {ov}: {e}"),
                    None => format!("could not attach {path} to {at}: {e}"),
                };
                attach_failed(msg, &at)?
            }
        }
    }
    out.tables.sort_by_key(|t| t.slot);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scsi::{ScsiIdentities, ScsiIdentity};

    const ID: ScsiIdentity = ScsiIdentity::new(b"TEST    ", b"ATTACH          ", b"1.0 ");

    fn buses() -> [ScsiBus; 2] {
        std::array::from_fn(|_| ScsiBus::new(16, ScsiIdentities { disk: ID, cdrom: ID }))
    }

    fn route<'a>(b: &'a [ScsiBus; 2]) -> impl Fn(u8) -> Result<&'a ScsiBus, String> {
        move |c| b.get(c as usize).ok_or_else(|| format!("controller {c} (two buses here)"))
    }

    fn image(tag: &str) -> String {
        let p = std::env::temp_dir().join(format!("iris-attach-{tag}-{}", std::process::id()));
        std::fs::write(&p, vec![0u8; 4096]).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn table(path: &str, cdrom: bool, controller: u8) -> ScsiDeviceConfig {
        ScsiDeviceConfig { path: path.into(), cdrom, controller, ..Default::default() }
    }

    #[test]
    fn tables_land_on_their_controller_and_id() {
        let (a, b) = (image("a"), image("b"));
        let mut cfg = MachineConfig::default();
        cfg.scsi.clear();
        cfg.scsi.insert(1, table(&a, false, 0));
        cfg.scsi.insert(2, table(&b, false, 1)); // controller field
        cfg.scsi.insert(16 + 5, table(&b, false, 0)); // key names controller 1
        let buses = buses();
        attach_from_config(&cfg, None, &route(&buses)).unwrap();
        let ids = |b: &ScsiBus| (0..16).filter(|&i| b.has_target(i)).collect::<Vec<u8>>();
        assert_eq!(ids(&buses[0]), vec![1]);
        assert_eq!(ids(&buses[1]), vec![2, 5]);
        for p in [a, b] {
            std::fs::remove_file(p).unwrap();
        }
    }

    #[test]
    fn a_bus_the_machine_lacks_is_an_error() {
        let a = image("lack");
        let mut cfg = MachineConfig::default();
        cfg.scsi.clear();
        cfg.scsi.insert(1, table(&a, false, 2));
        let buses = buses();
        let e = attach_from_config(&cfg, None, &route(&buses)).err().unwrap();
        assert!(e.starts_with("SCSI 2:1: controller 2"), "{e}");
        std::fs::remove_file(a).unwrap();
    }

    #[test]
    fn daynaports_are_returned_for_the_machine_to_attach() {
        let mut cfg = MachineConfig::default();
        cfg.scsi.clear();
        cfg.scsi.insert(3, ScsiDeviceConfig { kind_field: crate::config::ScsiKind::Daynaport, ..Default::default() });
        let buses = buses();
        let got = attach_from_config(&cfg, None, &route(&buses)).unwrap();
        assert_eq!((got.daynaports[0].controller, got.daynaports[0].id), (0, 3));
        assert!(got.tables.is_empty(), "no image behind a DaynaPort");
        assert!(!buses[0].has_target(3), "and nothing attached");
    }

    #[test]
    fn tables_come_back_by_slot_with_their_kind() {
        let (a, iso) = (image("prov"), image("iso"));
        let mut cfg = MachineConfig::default();
        cfg.scsi.clear();
        cfg.scsi.insert(4, table(&iso, true, 0));
        cfg.scsi.insert(1, table(&a, false, 0));
        let buses = buses();
        let got = attach_from_config(&cfg, None, &route(&buses)).unwrap();
        let kinds: Vec<_> = got.tables.iter().map(|t| (t.slot, t.is_cdrom)).collect();
        assert_eq!(kinds, vec![(1, false), (4, true)]);
        assert_eq!(got.tables[0].size_bytes, 4096);
        for p in [a, iso] {
            std::fs::remove_file(p).unwrap();
        }
    }
}
