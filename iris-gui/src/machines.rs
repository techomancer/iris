//! Machine folders and the core's TOML schema. Paths stay relative on disk and
//! in the editor; the selected machine folder is the emulator's working directory.

use iris::config::MachineConfig;
use std::path::{Component, Path, PathBuf};

pub fn validate_name(name: &str) -> Result<(), String> {
    if name.trim().is_empty()
        || name.starts_with('.')
        || name.contains(['/', '\\', ':'])
        || name.chars().any(char::is_control)
        || name.ends_with(['.', ' '])
    {
        return Err(
            "Use a machine name without path separators, leading dots, or trailing dots/spaces"
                .into(),
        );
    }
    // These names cannot be directories on Windows, even with an extension.
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    if matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
        || name.contains(['<', '>', '"', '|', '?', '*'])
    {
        return Err(
            "This machine name is reserved or contains unsupported filename characters".into(),
        );
    }
    Ok(())
}

/// Older JSON names had no filename restrictions. Give those profiles a
/// deterministic safe folder name so retries select the same migrated file.
pub fn legacy_name(name: &str) -> String {
    if validate_name(name).is_ok() {
        return name.into();
    }
    use std::hash::{Hash, Hasher};
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hash);
    let base: String = name
        .chars()
        .map(|c| {
            if c.is_control() || "/\\:<>\"|?*".contains(c) {
                '_'
            } else {
                c
            }
        })
        .collect();
    let base = base.trim().trim_matches('.').trim();
    let base = if validate_name(base).is_ok() {
        base
    } else {
        "machine"
    };
    format!("{base}-{:08x}", hash.finish() as u32)
}

pub fn directory(root: &Path, name: &str) -> Result<PathBuf, String> {
    validate_name(name)?;
    Ok(root.join("machines").join(name))
}

pub fn config_path(root: &Path, name: &str) -> Result<PathBuf, String> {
    Ok(directory(root, name)?.join(format!("{name}.toml")))
}

pub fn read(path: &Path) -> Result<MachineConfig, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Replace a config only after serialization and the complete write succeed.
pub fn write(path: &Path, text: &str) -> Result<(), String> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&tmp)?;
        use std::io::Write;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result.map_err(|e| format!("{}: {e}", path.display()))
}

/// Visit filesystem paths only. Network interfaces, MACs, subnets, debug module
/// filters, and TCP CI addresses are strings but must never be treated as paths.
pub fn visit_paths(cfg: &mut MachineConfig, mut visit: impl FnMut(&mut String)) {
    visit(&mut cfg.prom);
    visit(&mut cfg.nvram);
    visit(&mut cfg.nveeprom);
    for path in [
        &mut cfg.serial_log,
        &mut cfg.load_elf,
        &mut cfg.test_device_dump,
        &mut cfg.network.tftp_dir,
    ] {
        if let Some(path) = path {
            visit(path);
        }
    }
    visit(&mut cfg.jitv2.cache_dir);
    if !iris::config::ci_socket_is_tcp(&cfg.ci_socket) {
        visit(&mut cfg.ci_socket);
    }
    for dev in cfg.scsi.values_mut() {
        visit(&mut dev.path);
        for disc in &mut dev.discs {
            visit(disc);
        }
    }
    if let Some(nfs) = &mut cfg.nfs {
        visit(&mut nfs.shared_dir);
    }
}

pub fn resolve(base: &Path, path: &str) -> PathBuf {
    base.join(path)
}

/// Compute a relative path without requiring the destination to exist. Keep
/// symlinks intact: lexical normalization would change `symlink/../file`.
pub fn relative(base: &Path, path: &Path) -> Result<PathBuf, String> {
    let base_parts: Vec<_> = base.components().collect();
    let path_parts: Vec<_> = path.components().collect();
    if !base.is_absolute() || !path.is_absolute() || base_parts.first() != path_parts.first() {
        return Err(format!(
            "Cannot express {} relative to {}",
            path.display(),
            base.display()
        ));
    }
    let common = base_parts
        .iter()
        .zip(&path_parts)
        .take_while(|(a, b)| a == b)
        .count();
    let mut out = PathBuf::new();
    for part in &base_parts[common..] {
        if *part != Component::CurDir {
            out.push("..");
        }
    }
    for part in &path_parts[common..] {
        out.push(part.as_os_str());
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    Ok(out)
}

pub fn make_relative(cfg: &mut MachineConfig, dir: &Path) -> Result<(), String> {
    let mut error = None;
    visit_paths(cfg, |path| {
        if Path::new(path).is_absolute() {
            match relative(dir, Path::new(path)) {
                Ok(p) => *path = p.to_string_lossy().into_owned(),
                Err(e) => error = Some(e),
            }
        }
    });
    error.map_or(Ok(()), Err)
}

pub fn save(root: &Path, name: &str, cfg: &mut MachineConfig) -> Result<(), String> {
    let dir = directory(root, name)?;
    make_relative(cfg, &dir)?;
    let text = toml::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(dir.join("disks")).map_err(|e| e.to_string())?;
    write(&config_path(root, name)?, &text)
}

pub fn discover(
    root: &Path,
) -> (
    std::collections::BTreeMap<String, MachineConfig>,
    Vec<String>,
) {
    let mut machines = std::collections::BTreeMap::new();
    let mut errors = Vec::new();
    let entries = match std::fs::read_dir(root.join("machines")) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (machines, errors),
        Err(e) => return (machines, vec![format!("Cannot list machines: {e}")]),
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                errors.push(e.to_string());
                continue;
            }
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        if validate_name(&name).is_err() {
            continue;
        }
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let path = entry.path().join(format!("{name}.toml"));
        match std::fs::metadata(&path) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                errors.push(format!("{}: {e}", path.display()));
                continue;
            }
        }
        match read(&path) {
            Ok(mut cfg) => match make_relative(&mut cfg, &entry.path()) {
                Ok(()) => {
                    machines.insert(name, cfg);
                }
                Err(e) => errors.push(e),
            },
            Err(e) => errors.push(e),
        }
    }
    (machines, errors)
}

/// Migrate battery-backed state into the machine folder, preserving existing
/// files and external disk locations. The old launch directory is captured
/// before selecting a machine changes the process working directory.
pub fn migrate(root: &Path, name: &str, cfg: &MachineConfig, old_cwd: &Path) -> Result<(), String> {
    let dir = directory(root, name)?;
    let target = config_path(root, name)?;
    if target.exists() {
        read(&target)?;
        return Ok(());
    }
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut migrated = cfg.clone();
    visit_paths(&mut migrated, |p| {
        if !p.is_empty() && p != "(embedded)" {
            *p = resolve(old_cwd, p).to_string_lossy().into_owned();
        }
    });
    for (old, path, leaf) in [
        (&cfg.nvram, &mut migrated.nvram, "nvram.bin"),
        (&cfg.nveeprom, &mut migrated.nveeprom, "nveeprom.bin"),
    ] {
        let source = if !Path::new(old).is_absolute() {
            // The old GUI anchored relative battery paths by their filename.
            let filename = Path::new(old)
                .file_name()
                .filter(|p| !p.is_empty())
                .unwrap_or_else(|| std::ffi::OsStr::new(leaf));
            let shared = root.join(filename);
            if shared.exists() {
                shared
            } else if old.is_empty() {
                old_cwd.join(leaf)
            } else {
                old_cwd.join(old)
            }
        } else {
            PathBuf::from(old)
        };
        let dest = dir.join(leaf);
        if !dest.exists() {
            match std::fs::copy(&source, &dest) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(format!("Cannot migrate {}: {e}", source.display())),
            }
        }
        *path = leaf.into();
    }
    make_relative(&mut migrated, &dir)?;
    // App Store builds formerly redirected CHD diffs into one shared folder.
    // Copy any existing sidecar to the machine's redirect, using its new path
    // spelling for the hash. Keep the original for rollback.
    #[cfg(feature = "appstore")]
    for (id, dev) in &cfg.scsi {
        if !iris::chd_disk::is_chd(&dev.path) {
            continue;
        }
        let source = root.join("chd-diffs").join(diff_name(&dev.path));
        let dest = dir
            .join("chd-diffs")
            .join(diff_name(&migrated.scsi[id].path));
        if source.exists() && !dest.exists() {
            std::fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
            std::fs::copy(source, dest).map_err(|e| e.to_string())?;
        }
    }
    save(root, name, &mut migrated)
}

#[cfg(feature = "appstore")]
fn diff_name(path: &str) -> String {
    use std::hash::{Hash, Hasher};
    let path = Path::new(path);
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hash);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("disk");
    format!("{stem}.{:016x}.diff.chd", hash.finish())
}

pub fn rename(root: &Path, old: &str, new: &str) -> Result<(), String> {
    let old_dir = directory(root, old)?;
    let new_dir = directory(root, new)?;
    if new_dir.exists() {
        return Err(format!("Machine folder '{new}' already exists"));
    }
    std::fs::rename(&old_dir, &new_dir).map_err(|e| e.to_string())?;
    if let Err(e) = std::fs::rename(
        new_dir.join(format!("{old}.toml")),
        new_dir.join(format!("{new}.toml")),
    ) {
        let _ = std::fs::rename(&new_dir, &old_dir);
        return Err(e.to_string());
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub struct TempRoot(pub PathBuf);
    impl TempRoot {
        pub fn new(label: &str) -> Self {
            let suffix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "iris-machines-{label}-{}-{suffix}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path.canonicalize().unwrap())
        }
    }
    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn saved_toml_roundtrips_every_path_and_preserves_non_path_strings() {
        let root = TempRoot::new("paths");
        let dir = directory(&root.0, "Indigo2 IP28").unwrap();
        let mut cfg = MachineConfig::default();
        cfg.prom = dir.join("prom.bin").to_string_lossy().into_owned();
        cfg.nvram = dir.join("nvram.bin").to_string_lossy().into_owned();
        cfg.nveeprom = dir.join("nveeprom.bin").to_string_lossy().into_owned();
        cfg.scsi.get_mut(&1).unwrap().path =
            dir.join("disks/root.chd").to_string_lossy().into_owned();
        cfg.scsi.get_mut(&4).unwrap().path.clear();
        cfg.scsi.get_mut(&4).unwrap().discs =
            vec![root.0.join("install.iso").to_string_lossy().into_owned()];
        cfg.nfs = Some(iris::config::NfsConfig {
            shared_dir: root.0.join("shared").to_string_lossy().into_owned(),
            version: Default::default(),
        });
        cfg.network.tftp_dir = Some(root.0.join("tftp").to_string_lossy().into_owned());
        cfg.serial_log = Some(dir.join("console.log").to_string_lossy().into_owned());
        cfg.load_elf = Some(root.0.join("test.elf").to_string_lossy().into_owned());
        cfg.test_device_dump = Some(dir.join("dump.json").to_string_lossy().into_owned());
        cfg.jitv2.cache_dir = dir.join("jit-cache").to_string_lossy().into_owned();
        cfg.ci_socket = "tcp:127.0.0.1:19851".into();
        cfg.debug.debug_log = "scsi=debug,cpu=warn".into();
        cfg.network.pcap_interface = Some("\\Device\\NPF_{example}".into());
        cfg.network.mac = Some("08:00:69:12:34:56".into());
        save(&root.0, "Indigo2 IP28", &mut cfg).unwrap();
        let path = config_path(&root.0, "Indigo2 IP28").unwrap();
        let mut back = read(&path).unwrap();
        assert_eq!(back.scsi[&1].path, "disks/root.chd");
        assert_eq!(back.scsi[&4].path, "");
        assert_eq!(back.scsi[&4].discs[0], "../../install.iso");
        assert_eq!(back.nvram, "nvram.bin");
        assert_eq!(back.nveeprom, "nveeprom.bin");
        assert_eq!(back.ci_socket, "tcp:127.0.0.1:19851");
        assert_eq!(back.debug.debug_log, cfg.debug.debug_log);
        assert_eq!(back.network.pcap_interface, cfg.network.pcap_interface);
        assert_eq!(back.network.mac, cfg.network.mac);
        visit_paths(&mut back, |p| assert!(!Path::new(p).is_absolute(), "{p}"));
        assert!(dir.join("disks").is_dir());
    }

    #[test]
    fn discovery_uses_only_matching_machine_tomls_and_reports_parse_errors() {
        let root = TempRoot::new("discover");
        let mut cfg = MachineConfig::default();
        save(&root.0, "indy", &mut cfg).unwrap();
        let stray = directory(&root.0, "stray").unwrap();
        std::fs::create_dir_all(&stray).unwrap();
        std::fs::write(stray.join("other.toml"), "banks = [8,8,0,0]").unwrap();
        let broken = directory(&root.0, "broken").unwrap();
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("broken.toml"), "unknown_setting = true").unwrap();
        let (machines, errors) = discover(&root.0);
        assert_eq!(machines.keys().collect::<Vec<_>>(), vec!["indy"]);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("broken.toml"));
        assert!(errors[0].contains("unknown_setting"));
        assert_eq!(
            std::fs::read_to_string(broken.join("broken.toml")).unwrap(),
            "unknown_setting = true"
        );
    }

    #[test]
    fn legacy_migration_isolates_battery_files_and_preserves_external_disks() {
        let root = TempRoot::new("migrate");
        let old_cwd = root.0.join("old-launch");
        std::fs::create_dir_all(&old_cwd).unwrap();
        std::fs::write(root.0.join("nvram.bin"), b"original NVRAM").unwrap();
        std::fs::write(root.0.join("nveeprom.bin"), b"original EEPROM").unwrap();
        std::fs::write(old_cwd.join("scsi1.raw"), b"disk stays here").unwrap();
        let cfg = MachineConfig::default();
        for name in ["indy", "ip28"] {
            migrate(&root.0, name, &cfg, &old_cwd).unwrap();
        }
        let indy = directory(&root.0, "indy").unwrap();
        let ip28 = directory(&root.0, "ip28").unwrap();
        let migrated = read(&config_path(&root.0, "indy").unwrap()).unwrap();
        assert_eq!(
            std::fs::read(indy.join(&migrated.scsi[&1].path)).unwrap(),
            b"disk stays here"
        );
        assert_eq!(
            std::fs::read(indy.join(&migrated.nvram)).unwrap(),
            b"original NVRAM"
        );
        std::fs::write(indy.join("nvram.bin"), b"changed Indy").unwrap();
        std::fs::write(indy.join("nveeprom.bin"), b"changed Indy EEPROM").unwrap();
        assert_eq!(
            std::fs::read(ip28.join("nvram.bin")).unwrap(),
            b"original NVRAM"
        );
        assert_eq!(
            std::fs::read(ip28.join("nveeprom.bin")).unwrap(),
            b"original EEPROM"
        );
        // Retrying after a failed preferences write never resets migrated state.
        migrate(&root.0, "indy", &cfg, &old_cwd).unwrap();
        assert_eq!(
            std::fs::read(indy.join("nvram.bin")).unwrap(),
            b"changed Indy"
        );
        assert_eq!(
            std::fs::read(old_cwd.join("scsi1.raw")).unwrap(),
            b"disk stays here"
        );
    }

    #[test]
    fn rename_moves_config_disks_and_battery_state_as_one_machine() {
        let root = TempRoot::new("rename");
        let mut cfg = MachineConfig::default();
        cfg.scsi.get_mut(&1).unwrap().path = "disks/root.raw".into();
        save(&root.0, "before", &mut cfg).unwrap();
        let old_dir = directory(&root.0, "before").unwrap();
        for (leaf, data) in [
            ("disks/root.raw", b"disk".as_slice()),
            ("nvram.bin", b"nvram"),
            ("nveeprom.bin", b"eeprom"),
        ] {
            std::fs::write(old_dir.join(leaf), data).unwrap();
        }
        rename(&root.0, "before", "after").unwrap();
        let new_dir = directory(&root.0, "after").unwrap();
        assert!(!old_dir.exists());
        assert!(!new_dir.join("before.toml").exists());
        let cfg = read(&new_dir.join("after.toml")).unwrap();
        assert_eq!(
            std::fs::read(new_dir.join(&cfg.scsi[&1].path)).unwrap(),
            b"disk"
        );
        assert_eq!(std::fs::read(new_dir.join(&cfg.nvram)).unwrap(), b"nvram");
        assert_eq!(
            std::fs::read(new_dir.join(&cfg.nveeprom)).unwrap(),
            b"eeprom"
        );
        std::fs::remove_file(new_dir.join("after.toml")).unwrap();
        assert!(discover(&root.0).0.is_empty());
        assert!(new_dir.join("disks/root.raw").exists());
    }

    #[test]
    fn names_cannot_escape_storage_or_alias_hidden_directories() {
        for name in [
            "",
            ".",
            "..",
            ".hidden",
            "../escape",
            "a/b",
            "a\\b",
            "a:b",
            "a\n",
            "a.",
            "a ",
            "CON",
            "LPT1",
            "a?",
        ] {
            assert!(validate_name(name).is_err(), "{name:?}");
        }
        for name in ["Indigo2 IP28", "indy", "マシン", "irix-6.5"] {
            validate_name(name).unwrap();
        }
    }

    #[test]
    fn relative_external_resources_are_bookmarked_from_each_machine_folder() {
        let root = TempRoot::new("bookmarks");
        let dir = directory(&root.0, "indy").unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        let resource = root.0.join("external.iso");
        std::fs::write(&resource, b"media").unwrap();
        let mut cfg = MachineConfig::default();
        cfg.scsi.get_mut(&4).unwrap().path = "../../external.iso".into();
        let paths = crate::macos_sandbox::config_paths(&cfg, &dir);
        assert!(paths.contains(&resource.to_string_lossy().into_owned()));
    }

    #[cfg(feature = "appstore")]
    #[test]
    fn sandbox_migration_preserves_shared_chd_diffs_in_each_machine_folder() {
        let root = TempRoot::new("diffs");
        std::fs::create_dir_all(root.0.join("chd-diffs")).unwrap();
        let mut cfg = MachineConfig::default();
        cfg.scsi.get_mut(&1).unwrap().path = root.0.join("root.chd").to_string_lossy().into_owned();
        let old_diff = root.0.join("chd-diffs").join(diff_name(&cfg.scsi[&1].path));
        std::fs::write(&old_diff, b"guest writes").unwrap();
        migrate(&root.0, "indy", &cfg, &root.0).unwrap();
        let migrated = read(&config_path(&root.0, "indy").unwrap()).unwrap();
        let new_diff = directory(&root.0, "indy")
            .unwrap()
            .join("chd-diffs")
            .join(diff_name(&migrated.scsi[&1].path));
        assert_eq!(std::fs::read(new_diff).unwrap(), b"guest writes");
        assert!(old_diff.exists());
    }
}
