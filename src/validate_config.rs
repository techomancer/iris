//! `iris --validate`: look over a configuration without starting the machine.
//!
//! [`MachineConfig::validate`] answers whether the settings agree with each
//! other. This adds what only the host can say: that the files the machine
//! will open are there, that the directories it will write into exist, and
//! that two listeners are not asked for the same port. Nothing is built, bound
//! or opened, so it is safe next to a running iris and needs no display or
//! sound device.

use std::collections::BTreeMap;
use std::path::Path;

use crate::config::{MachineConfig, MachineProfile, ScsiKind, DEFAULT_PROM};

fn missing_file(what: &str, path: &str, problems: &mut Vec<String>) {
    match std::fs::metadata(path) {
        Ok(m) if m.is_file() => {}
        Ok(_) => problems.push(format!("{what} {path} is not a file")),
        Err(e) => problems.push(format!("{what} {path}: {e}")),
    }
}

fn missing_dir(what: &str, path: &str, problems: &mut Vec<String>) {
    match std::fs::metadata(path) {
        Ok(m) if m.is_dir() => {}
        Ok(_) => problems.push(format!("{what} {path} is not a directory")),
        Err(e) => problems.push(format!("{what} {path}: {e}")),
    }
}

/// A file iris will create must have a directory to go in.
fn missing_parent(what: &str, path: &str, problems: &mut Vec<String>) {
    let parent = match Path::new(path).parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => return,
    };
    if !parent.is_dir() {
        problems.push(format!("{what} {path}: directory {} does not exist", parent.display()));
    }
}

/// What the machine would warn about and carry on: the audio card added
/// with `no_audio` set.
pub fn warnings(cfg: &MachineConfig) -> Vec<String> {
    let mut warnings = Vec::new();
    if cfg.audio_card() && cfg.no_audio {
        warnings.push(crate::ip35::AUDIO_WITH_NO_AUDIO.to_string());
    }
    warnings
}

/// Everything wrong with `cfg`, one message each; empty when it would run.
pub fn problems(cfg: &MachineConfig) -> Vec<String> {
    let mut problems = Vec::new();
    if let Err(e) = cfg.validate() {
        problems.push(e);
    }
    let profile = cfg.machine.profile;

    // The PROM: one the user named must be there. Left at the default, or "(embedded)",
    // the machine runs the PROM built into iris.
    if cfg.prom != DEFAULT_PROM && !crate::dev::prom_image::is_embedded_keyword(&cfg.prom) {
        missing_file("prom", &cfg.prom, &mut problems);
    }

    let placement = crate::topology::DevicePlacement::lenient(cfg, &cfg.devices);
    for (i, d) in cfg.devices.iter().enumerate() {
        let (Some(dev), Some(at)) = (d.device.scsi(), placement.at(i)) else { continue };
        let what = format!("{} at {at}", d.device.spec().name);
        match dev.kind() {
            ScsiKind::Daynaport => {}
            ScsiKind::Cdrom => {
                if !dev.path.is_empty() {
                    missing_file(&what, &dev.path, &mut problems);
                }
                for disc in &dev.discs {
                    missing_file(&format!("{what} disc"), disc, &mut problems);
                }
            }
            ScsiKind::Disk if dev.scratch => missing_parent(&format!("{what} scratch volume"), &dev.path, &mut problems),
            ScsiKind::Disk if dev.path.is_empty() => problems.push(format!("{what}: a disk needs a path")),
            ScsiKind::Disk => missing_file(&what, &dev.path, &mut problems),
        }
    }

    if let Some(path) = &cfg.load_elf {
        missing_file("load_elf", path, &mut problems);
    }
    if let Some(nfs) = &cfg.nfs {
        missing_dir("nfs.shared_dir", &nfs.shared_dir, &mut problems);
    }
    for (what, dir) in [("network.tftp_dir", &cfg.network.tftp_dir), ("network.rsh_dir", &cfg.network.rsh_dir)] {
        if let Some(dir) = dir {
            missing_dir(what, dir, &mut problems);
        }
    }

    let outputs = [("serial_log", &cfg.serial_log), ("serial_log_a", &cfg.serial_log_a), ("serial_log_b", &cfg.serial_log_b)];
    let sockets = [("serial_socket_a", &cfg.serial_socket_a), ("serial_socket_b", &cfg.serial_socket_b)];
    for (what, path) in outputs.into_iter().chain(sockets) {
        if let Some(path) = path {
            missing_parent(what, path, &mut problems);
        }
    }

    // TCP ports iris listens on. A port two of them share leaves the second
    // without its listener (a console or the monitor), which a run only says
    // on stderr.
    let [a, b] = cfg.serial_ports();
    let mut ports = vec![("serial_port_a", a), ("serial_port_b", b), ("monitor_port", cfg.monitor_port.unwrap_or(crate::monitor::DEFAULT_PORT))];
    if let Some(gdb) = cfg.gdb_port {
        ports.push(("gdb_port", gdb));
    }
    // The listeners a board adds. Defaults match the machines' own.
    let l1 = |p: Option<u16>| ("machine.l1_port", p.unwrap_or(8882));
    match cfg.machine.profile {
        MachineProfile::FuelIp34 => ports.push(l1(cfg.fuel.l1_port)),
        MachineProfile::TezroIp53 => ports.extend([
            l1(cfg.tezro.l1_port),
            ("machine.serial_port_c", cfg.tezro.serial_port_c.unwrap_or(8883)),
            ("machine.serial_port_d", cfg.tezro.serial_port_d.unwrap_or(8884)),
        ]),
        MachineProfile::Origin3200Ip35 | MachineProfile::Onyx3000Ip35 => ports.push(l1(cfg.ip35_section().l1_port)),
        MachineProfile::O2Ip32 => ports.extend(cfg.o2.control_port.map(|p| ("machine.control_port", p))),
        _ => {}
    }
    let virtio = &cfg.virtio_serial;
    if virtio.enabled {
        let default = [crate::dev::virtio::console::DEFAULT_TCP_PORT];
        let tcp = virtio.ports.iter().filter_map(|p| p.tcp_port);
        ports.extend(if virtio.ports.is_empty() { default.to_vec() } else { tcp.collect() }.into_iter().map(|p| ("virtio_serial tcp_port", p)));
    }
    ports.extend(cfg.port_forward.iter().filter(|f| f.proto == crate::config::ForwardProto::Tcp).map(|f| ("port_forward host_port", f.host_port)));
    let mut seen: BTreeMap<u16, &str> = BTreeMap::new();
    for (setting, port) in ports {
        if port == 0 {
            continue;
        }
        if let Some(other) = seen.insert(port, setting) {
            problems.push(format!("{setting} and {other} both use port {port}"));
        }
    }
    problems
}

/// Run the check for `iris --validate`: print the problems, or that the
/// configuration is fine. Returns the process exit status.
pub fn report(path: &str, cfg: &MachineConfig) -> i32 {
    for w in warnings(cfg) {
        eprintln!("Warning: {w}");
    }
    let problems = problems(cfg);
    if problems.is_empty() {
        println!("iris: {path}: ok ({})", cfg.machine.profile.label());
        0
    } else {
        for p in &problems {
            eprintln!("Configuration error: {p}");
        }
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::banks_from4;

    fn dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("iris-check-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn file(dir: &Path, name: &str) -> String {
        let path = dir.join(name);
        std::fs::write(&path, b"x").unwrap();
        path.to_string_lossy().into_owned()
    }

    /// An Octane whose PROM, disk and CD image are there checks clean.
    #[test]
    fn a_good_config_has_no_problems() {
        let d = dir("good");
        let text = format!(
            "prom = {:?}\nbanks = [128, 0, 0, 0]\nserial_log = {:?}\n[machine]\nprofile = \"octane_ip30\"\ncpu = \"r10000\"\ncpus = 2\n\
             [scsi.1]\npath = {:?}\n[scsi.4]\ncdrom = true\n",
            file(&d, "prom"), d.join("console.log"), file(&d, "disk"));
        let cfg_path = d.join("octane.toml");
        std::fs::write(&cfg_path, text).unwrap();
        let cfg = MachineConfig::try_load(cfg_path.to_str().unwrap()).unwrap();
        assert_eq!(problems(&cfg), Vec::<String>::new());
        assert_eq!(report("octane.toml", &cfg), 0);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The defaults want the disk and CD image iris.toml would name, and
    /// nothing else: the Indy has its PROM built in. A config file that is
    /// not there is the defaults, as for a run.
    #[test]
    fn the_defaults_need_only_their_disk_images() {
        let d = dir("defaults");
        let cfg = MachineConfig::try_load(d.join("absent.toml").to_str().unwrap()).unwrap();
        let wanted = problems(&cfg);
        // (Both are there if the test is run beside such files.)
        assert!(wanted.iter().all(|p| p.starts_with("hdd at hpc3/c0/t1 scsi1.raw") || p.starts_with("cdrom at hpc3/c0/t4 cdrom4.iso")), "{wanted:?}");
        let mut cfg = cfg;
        cfg.devices.clear();
        assert!(problems(&cfg).is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Text that is not a configuration, or has a key iris does not know, is
    /// an error that names the file.
    #[test]
    fn a_file_that_does_not_parse_is_an_error() {
        let d = dir("parse");
        let path = d.join("bad.toml");
        std::fs::write(&path, "scale = \n").unwrap();
        assert!(MachineConfig::try_load(path.to_str().unwrap()).unwrap_err().contains("failed to parse"));
        std::fs::write(&path, "no_such_key = 1\n").unwrap();
        assert!(MachineConfig::try_load(path.to_str().unwrap()).unwrap_err().contains("no_such_key"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Settings that disagree, files that are not there and ports used twice
    /// are each reported, together.
    #[test]
    fn a_bad_config_lists_every_problem() {
        let d = dir("bad");
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::OctaneIp30;
        cfg.machine.cpu = Some("r4400".into());
        cfg.banks = banks_from4([128, 0, 0, 0]);
        cfg.prom = d.join("no-prom").to_string_lossy().into_owned();
        cfg.devices.clear();
        cfg.set_disk(1, toml::from_str(&format!("path = {:?}", d.join("no-disk").to_str().unwrap())).unwrap());
        cfg.set_disk(2, toml::from_str("path = \"\"").unwrap());
        cfg.serial_log = Some(d.join("nowhere").join("log").to_string_lossy().into_owned());
        cfg.monitor_port = Some(8880);
        let all = problems(&cfg).join("\n");
        for want in [
            "SGI Octane (IP30) takes [machine] cpu = ",
            "prom ",
            "hdd at xio15/pci0/0/t1 ",
            "hdd at xio15/pci0/0/t2: a disk needs a path",
            "serial_log ",
            "directory ",
            "monitor_port and serial_port_a both use port 8880",
        ] {
            assert!(all.contains(want), "no {want:?} in:\n{all}");
        }
        assert_eq!(report("bad.toml", &cfg), 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A board's L1 port and a Tezro's extra serial ports are listeners too.
    #[test]
    fn the_board_ports_join_the_clash_check() {
        let mut cfg = MachineConfig::default();
        cfg.devices.clear();
        cfg.machine.profile = MachineProfile::FuelIp34;
        cfg.fuel.l1_port = Some(8880);
        assert!(problems(&cfg).iter().any(|p| p == "machine.l1_port and serial_port_a both use port 8880"), "{:?}", problems(&cfg));
        cfg.machine.profile = MachineProfile::TezroIp53;
        cfg.tezro.l1_port = None;
        cfg.tezro.serial_port_c = Some(8881);
        assert!(problems(&cfg).iter().any(|p| p == "machine.serial_port_c and serial_port_b both use port 8881"), "{:?}", problems(&cfg));
        cfg.tezro.serial_port_c = None;
        assert!(!problems(&cfg).iter().any(|p| p.contains("both use port")), "{:?}", problems(&cfg));
        cfg.machine.profile = MachineProfile::O2Ip32;
        cfg.o2.control_port = Some(8888);
        assert!(problems(&cfg).iter().any(|p| p == "machine.control_port and monitor_port both use port 8888"), "{:?}", problems(&cfg));
        cfg.o2.control_port = None;
        cfg.virtio_serial.enabled = true;
        cfg.serial_port_a = Some(8890);
        assert!(problems(&cfg).iter().any(|p| p == "virtio_serial tcp_port and serial_port_a both use port 8890"), "{:?}", problems(&cfg));
    }

    /// A `prom` the user names must be there, on every machine; "(embedded)" and the
    /// default ask for the built-in PROM.
    #[test]
    fn a_named_prom_must_exist() {
        let mut cfg = MachineConfig::default();
        cfg.devices.clear();
        cfg.prom = "no-such-prom.bin".into();
        assert!(problems(&cfg).iter().any(|p| p.starts_with("prom ")), "{:?}", problems(&cfg));
        assert_eq!(report("indy.toml", &cfg), 1);
        cfg.machine.profile = MachineProfile::OctaneIp30;
        cfg.machine.cpu = Some("r10000".into());
        cfg.banks = banks_from4([128, 0, 0, 0]);
        assert!(problems(&cfg).iter().any(|p| p.starts_with("prom ")));
        cfg.prom = "(embedded)".into();
        assert!(problems(&cfg).is_empty(), "{:?}", problems(&cfg));
    }

    /// The audio card with `no_audio` set is present and silent; that
    /// is worth saying, and not an error.
    #[test]
    fn the_tezro_audio_card_with_no_audio_is_a_warning() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::TezroIp53;
        cfg.tezro.audio = true;
        assert!(warnings(&cfg).is_empty());
        cfg.no_audio = true;
        assert!(warnings(&cfg).iter().any(|w| w.contains("plays to nowhere")), "{:?}", warnings(&cfg));
        cfg.tezro.audio = false;
        assert!(warnings(&cfg).is_empty());
        // The same card on the other machines that take it.
        cfg.machine.profile = MachineProfile::FuelIp34;
        cfg.fuel.audio = true;
        assert!(warnings(&cfg).iter().any(|w| w.contains("plays to nowhere")));
        cfg.machine.profile = MachineProfile::Onyx3000Ip35;
        assert!(warnings(&cfg).is_empty());
        cfg.onyx3000.audio = true;
        assert!(warnings(&cfg).iter().any(|w| w.contains("plays to nowhere")));
    }

    /// A scratch volume only needs its directory and an empty CD-ROM tray is
    /// fine; a PROM that is named and missing is a problem on an Octane.
    #[test]
    fn scratch_volumes_and_empty_trays_are_accepted() {
        let d = dir("lenient");
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::OctaneIp30;
        cfg.machine.cpu = Some("r10000".into());
        cfg.banks = banks_from4([128, 0, 0, 0]);
        cfg.prom = d.join("absent-prom").to_string_lossy().into_owned();
        assert!(problems(&cfg).iter().any(|p| p.starts_with("prom ")), "{:?}", problems(&cfg));
        cfg.devices.clear();
        cfg.set_disk(1, toml::from_str(&format!("scratch = true\npath = {:?}", d.join("scratch.img").to_str().unwrap())).unwrap());
        cfg.set_disk(4, toml::from_str("cdrom = true").unwrap());
        cfg.prom = file(&d, "prom");
        assert_eq!(problems(&cfg), Vec::<String>::new());
        let _ = std::fs::remove_dir_all(&d);
    }
}
