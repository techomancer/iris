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

/// What the machine would warn about and carry on: a PROM file named by
/// `prom` that is not there, on a machine with a built-in one.
pub fn warnings(cfg: &MachineConfig) -> Vec<String> {
    let mut warnings = Vec::new();
    if cfg.machine.profile.has_embedded_prom() && cfg.prom != DEFAULT_PROM {
        missing_file("prom", &cfg.prom, &mut warnings);
        for w in &mut warnings {
            w.push_str("; the built-in PROM would run");
        }
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

    // The PROM: the Indy and Indigo2 run a built-in one when the file is
    // missing (see `warnings`); the others read a file, the Octane and O2
    // trying the name their releases shipped under first.
    if !profile.has_embedded_prom() {
        missing_file("prom", &profile.prom_path(&cfg.prom), &mut problems);
    }

    let mut keys: Vec<_> = cfg.scsi.keys().copied().collect();
    keys.sort();
    for key in keys {
        let dev = &cfg.scsi[&key];
        let what = format!("scsi.{key}");
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
    use crate::config::{banks_from4, CpuModel};

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
        assert!(wanted.iter().all(|p| p.starts_with("scsi.1 scsi1.raw") || p.starts_with("scsi.4 cdrom4.iso")), "{wanted:?}");
        let mut cfg = cfg;
        cfg.scsi.clear();
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
        cfg.machine.cpu = CpuModel::R4400;
        cfg.banks = banks_from4([128, 0, 0, 0]);
        cfg.prom = d.join("no-prom").to_string_lossy().into_owned();
        cfg.scsi.clear();
        cfg.scsi.insert(1, toml::from_str(&format!("path = {:?}", d.join("no-disk").to_str().unwrap())).unwrap());
        cfg.scsi.insert(2, toml::from_str("path = \"\"").unwrap());
        cfg.serial_log = Some(d.join("nowhere").join("log").to_string_lossy().into_owned());
        cfg.monitor_port = Some(8880);
        let all = problems(&cfg).join("\n");
        for want in [
            "an Octane has an R10000",
            "prom ",
            "scsi.1 ",
            "scsi.2: a disk needs a path",
            "serial_log ",
            "directory ",
            "monitor_port and serial_port_a both use port 8880",
        ] {
            assert!(all.contains(want), "no {want:?} in:\n{all}");
        }
        assert_eq!(report("bad.toml", &cfg), 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The Indy and Indigo2 run their built-in PROM when `prom` names a file
    /// that is not there, so `--validate` warns and still passes; a machine
    /// with no built-in PROM fails.
    #[test]
    fn a_missing_prom_is_a_warning_where_the_machine_has_its_own() {
        let mut cfg = MachineConfig::default();
        cfg.scsi.clear();
        cfg.prom = "no-such-prom.bin".into();
        assert!(problems(&cfg).is_empty());
        assert_eq!(warnings(&cfg).len(), 1, "{:?}", warnings(&cfg));
        assert!(warnings(&cfg)[0].contains("built-in PROM"));
        assert_eq!(report("indy.toml", &cfg), 0);
        cfg.machine.profile = MachineProfile::OctaneIp30;
        cfg.machine.cpu = CpuModel::R10000;
        cfg.banks = banks_from4([128, 0, 0, 0]);
        assert!(warnings(&cfg).is_empty());
        assert!(problems(&cfg).iter().any(|p| p.starts_with("prom ")));
    }

    /// A scratch volume only needs its directory and an empty CD-ROM tray is
    /// fine; a PROM that is named and missing is a problem on an Octane.
    #[test]
    fn scratch_volumes_and_empty_trays_are_accepted() {
        let d = dir("lenient");
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::OctaneIp30;
        cfg.machine.cpu = CpuModel::R10000;
        cfg.banks = banks_from4([128, 0, 0, 0]);
        cfg.prom = d.join("absent-prom").to_string_lossy().into_owned();
        assert!(problems(&cfg).iter().any(|p| p.starts_with("prom ")), "{:?}", problems(&cfg));
        cfg.scsi.clear();
        cfg.scsi.insert(1, toml::from_str(&format!("scratch = true\npath = {:?}", d.join("scratch.img").to_str().unwrap())).unwrap());
        cfg.scsi.insert(4, toml::from_str("cdrom = true").unwrap());
        cfg.prom = file(&d, "prom");
        assert_eq!(problems(&cfg), Vec::<String>::new());
        let _ = std::fs::remove_dir_all(&d);
    }
}
