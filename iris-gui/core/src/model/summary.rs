//! What the welcome summary lists and what the Start preflight checks, worked
//! out from the config alone so every machine profile gets the same answer
//! and the tests need no window.
//!
//! Disks sit in different slots per machine: SCSI targets on every machine but
//! the IP35 bricks (where that slot list is unused), an `nvme` card on the O2 and in an
//! Octane option board, a `qla2200` loop on the machines with a PCI slot for
//! it. The SCSI targets are named by their (controller, id) placement, so the
//! O2's external bus reads `1:3`, not the `[scsi.19]` key.

use iris::config::{MachineConfig, NetMode};

/// One line of the summary's Drives row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Drive {
    Daynaport { id: String },
    Cdrom { id: String, path: String },
    Disk { id: String, path: String },
    Nvme { path: String },
    FibreChannel { id: u8, path: String },
}

/// Every drive the machine will be started with, SCSI targets in (controller,
/// id) order, then the NVMe cards, then the Fibre Channel loop.
pub fn drives(cfg: &MachineConfig) -> Vec<Drive> {
    let mut out = Vec::new();
    for (key, i) in cfg.disks() {
        let d = cfg.devices[i].device.scsi().expect("listed by disks");
        let id = crate::model::hardware::scsi_name(key);
        out.push(if d.is_daynaport() {
            Drive::Daynaport { id }
        } else if d.is_cdrom() {
            Drive::Cdrom { id, path: d.path.clone() }
        } else {
            Drive::Disk { id, path: d.path.clone() }
        });
    }
    out.extend(cfg.nvmes().map(|n| Drive::Nvme { path: n.path.clone() }));
    for (id, i) in cfg.loop_disks() {
        out.push(Drive::FibreChannel { id, path: cfg.devices[i].device.scsi().map_or_else(String::new, |d| d.path.clone()) });
    }
    out
}

/// `path` as the summary shows it: relative to the machine folder when it is
/// in it (the folder is the working directory, so a relative path already
/// is), else as written. The full path is the label's tooltip.
pub fn shown_path(path: &str, machine_dir: Option<&std::path::Path>) -> String {
    let p = std::path::Path::new(path);
    match machine_dir.and_then(|dir| p.strip_prefix(dir).ok()) {
        Some(rel) if !rel.as_os_str().is_empty() => rel.to_string_lossy().into_owned(),
        _ => path.to_string(),
    }
}

/// The summary's Network row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Network {
    /// Bridged to a host interface; `None` picks one automatically.
    Bridged(Option<String>),
    /// Built-in NAT; `None` is the default subnet.
    Nat(Option<String>),
    /// No Ethernet (every machine but the Indy and Indigo2).
    Off,
    /// A QEMU-style stream segment to this peer.
    Stream(Option<String>),
}

/// What the machine's Ethernet is attached to. A build without PCAP runs
/// NAT whatever the file says, so `pcap_built` false never reports a bridge.
pub fn network(cfg: &MachineConfig, pcap_built: bool) -> Network {
    match cfg.network.mode {
        NetMode::Pcap if pcap_built => Network::Bridged(
            cfg.network.pcap_interface.clone().filter(|s| !s.is_empty())),
        NetMode::Off => Network::Off,
        NetMode::Stream => Network::Stream(cfg.network.stream.clone()),
        NetMode::Nat | NetMode::Pcap => Network::Nat(cfg.nat_subnet.clone()),
    }
}

/// One SCSI target that is missing its backing file at Start time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingDisk {
    /// The `[scsi.KEY]` key, for detaching it.
    pub key: u8,
    /// The (controller, id) name the Disks tab shows.
    pub name: String,
    pub path: String,
    pub cdrom: bool,
}

/// The SCSI targets whose image `readable` refuses. Scratch volumes (iris
/// creates them), DaynaPorts (no image) and empty CD-ROM drives (tray empty)
/// are never missing, and a CD-ROM is present if any disc in its changer
/// list is. NVMe and Fibre Channel images are checked separately
/// ([`expansion_paths`]): they cannot be detached from this list.
pub fn missing_disks(cfg: &MachineConfig, readable: impl Fn(&str) -> bool) -> Vec<MissingDisk> {
    let mut out = Vec::new();
    for (key, i) in cfg.disks() {
        let dev = cfg.devices[i].device.scsi().expect("listed by disks");
        if dev.scratch || dev.is_daynaport() {
            continue;
        }
        if dev.is_cdrom() && dev.path.is_empty() && dev.discs.is_empty() {
            continue;
        }
        let present = readable(&dev.path)
            || (dev.is_cdrom() && dev.discs.iter().any(|d| readable(d)));
        if !present {
            out.push(MissingDisk {
                key,
                name: crate::model::hardware::scsi_name(key),
                path: dev.path.clone(),
                cdrom: dev.is_cdrom(),
            });
        }
    }
    out
}

/// The NVMe and Fibre Channel images the machine will open.
pub fn expansion_paths(cfg: &MachineConfig) -> Vec<&str> {
    cfg.loop_disks().into_iter().filter_map(|(_, i)| cfg.devices[i].device.scsi().map(|d| d.path.as_str()))
        .chain(cfg.nvmes().map(|n| n.path.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use iris::config::MachineProfile;

    fn parse(text: &str) -> MachineConfig {
        MachineConfig::from_toml_str(text).expect("config parses")
    }

    fn none_readable(_: &str) -> bool { false }

    #[test]
    fn o2_summary_lists_its_own_buses_by_placement() {
        let cfg = parse(r#"
            [machine]
            profile = "o2_ip32"
            [scsi.1]
            path = "irix.img"
            [scsi.19]
            path = "cd.iso"
            cdrom = true
            [[devices]]
            type = "nvme"
            path = "nvme.img"
        "#);
        assert_eq!(cfg.machine.profile, MachineProfile::O2Ip32);
        assert_eq!(drives(&cfg), vec![
            Drive::Disk { id: "1".into(), path: "irix.img".into() },
            Drive::Cdrom { id: "1:3".into(), path: "cd.iso".into() },
            Drive::Nvme { path: "nvme.img".into() },
        ]);
    }

    #[test]
    fn deprecated_o2_disks_reach_the_summary_and_the_disk_check() {
        // The spelling issue #304 was filed against: disks in `[o2]`, not in
        // `[scsi.N]`. The core folds them into `[[devices]]` on load.
        let cfg = parse(r#"
            [machine]
            profile = "o2_ip32"
            [o2]
            disks = [ { id = 2, path = "o2-root.img" },
                      { id = 4, bus = 1, path = "o2-cd.iso", cdrom = true } ]
        "#);
        assert_eq!(drives(&cfg), vec![
            Drive::Disk { id: "2".into(), path: "o2-root.img".into() },
            Drive::Cdrom { id: "1:4".into(), path: "o2-cd.iso".into() },
        ]);
        let missing = missing_disks(&cfg, none_readable);
        assert_eq!(missing.iter().map(|m| (m.name.as_str(), m.path.as_str(), m.cdrom)).collect::<Vec<_>>(),
                   vec![("2", "o2-root.img", false), ("1:4", "o2-cd.iso", true)]);
        // Every name maps back to a key that detaching removes.
        for m in &missing {
            assert!(cfg.disk(m.key).is_some());
        }
    }

    #[test]
    fn o2_with_readable_disks_starts_without_an_indy_disk() {
        let cfg = parse(r#"
            [machine]
            profile = "o2_ip32"
            [scsi.1]
            path = "o2.img"
        "#);
        assert!(missing_disks(&cfg, |p| p == "o2.img").is_empty());
        assert_eq!(missing_disks(&cfg, none_readable).len(), 1);
    }

    #[test]
    fn empty_disk_path_is_missing_but_empty_cdrom_and_scratch_are_not() {
        let cfg = parse(r#"
            [machine]
            profile = "o2_ip32"
            [scsi.1]
            path = ""
            [scsi.2]
            path = ""
            cdrom = true
            [scsi.3]
            path = "scratch.img"
            scratch = true
            [scsi.4]
            path = "gone.iso"
            cdrom = true
            discs = ["there.iso"]
        "#);
        let missing = missing_disks(&cfg, |p| p == "there.iso");
        assert_eq!(missing.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), vec!["1"]);
    }

    #[test]
    fn ip35_bricks_ignore_the_scsi_map() {
        let cfg = parse(r#"
            [machine]
            profile = "origin3200_ip35"
            [scsi.1]
            path = "unused.img"
            [[devices]]
            type = "hdd"
            at = "xio15/pci0/1/loop0"
            path = "fc0.img"
        "#);
        assert!(missing_disks(&cfg, none_readable).is_empty());
        assert_eq!(drives(&cfg), vec![Drive::FibreChannel { id: 0, path: "fc0.img".into() }]);
        assert_eq!(expansion_paths(&cfg), vec!["fc0.img"]);
    }

    #[test]
    fn every_nvme_card_is_listed_and_checked() {
        let cfg = parse(r#"
            [machine]
            profile = "octane_ip30"
            [[devices]]
            type = "nvme"
            path = "a.img"
            [[devices]]
            type = "nvme"
            path = "b.img"
        "#);
        assert_eq!(drives(&cfg), vec![Drive::Nvme { path: "a.img".into() }, Drive::Nvme { path: "b.img".into() }]);
        assert_eq!(expansion_paths(&cfg), vec!["a.img", "b.img"]);
    }

    #[test]
    fn indy_summary_is_unchanged() {
        let cfg = parse(r#"
            [scsi.1]
            path = "indy.img"
            [scsi.4]
            path = "cd.iso"
            cdrom = true
        "#);
        assert_eq!(cfg.machine.profile, MachineProfile::IndyIp24);
        assert_eq!(drives(&cfg), vec![
            Drive::Disk { id: "1".into(), path: "indy.img".into() },
            Drive::Cdrom { id: "4".into(), path: "cd.iso".into() },
        ]);
        assert!(expansion_paths(&cfg).is_empty());
    }

    #[test]
    fn paths_in_the_machine_folder_are_shown_relative() {
        let dir = std::path::Path::new("/m/o2");
        assert_eq!(shown_path("disks/root.img", Some(dir)), "disks/root.img");
        assert_eq!(shown_path("/m/o2/disks/root.img", Some(dir)), "disks/root.img");
        assert_eq!(shown_path("/m/o2b/x.img", Some(dir)), "/m/o2b/x.img");
        assert_eq!(shown_path("/elsewhere/x.iso", Some(dir)), "/elsewhere/x.iso");
        assert_eq!(shown_path("/m/o2", Some(dir)), "/m/o2");
        assert_eq!(shown_path("/m/o2/x", None), "/m/o2/x");
    }

    #[test]
    fn network_row_follows_the_mode() {
        let mut cfg = parse("[machine]\nprofile = \"o2_ip32\"\n");
        assert_eq!(network(&cfg, true), Network::Nat(None));
        cfg.network.mode = NetMode::Off;
        assert_eq!(network(&cfg, true), Network::Off);
        cfg.network.mode = NetMode::Stream;
        cfg.network.stream = Some("tcp:127.0.0.1:5555".into());
        assert_eq!(network(&cfg, true), Network::Stream(Some("tcp:127.0.0.1:5555".into())));
        cfg.network.mode = NetMode::Pcap;
        cfg.network.pcap_interface = Some(String::new());
        assert_eq!(network(&cfg, true), Network::Bridged(None));
        assert_eq!(network(&cfg, false), Network::Nat(None));
    }
}
