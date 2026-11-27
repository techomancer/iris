//! Machine folders and the core's TOML schema. Paths inside the machine folder stay relative on
//! disk and in the editor; external paths keep the spelling the user chose; the selected machine folder is the emulator's working directory.

use iris::config::MachineConfig;
use iris::devices::DeviceModel;
use std::path::{Component, Path, PathBuf};

pub fn validate_name(name: &str) -> Result<(), String> {
    if name.trim().is_empty()
        || name.starts_with('.')
        || name.contains(['/', '\\', ':'])
        || name.chars().any(char::is_control)
        || name.ends_with(['.', ' '])
    {
        return Err(t!("machines-name-invalid"));
    }
    // These names cannot be directories on Windows, even with an extension.
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    if matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
        || name.contains(['<', '>', '"', '|', '?', '*'])
    {
        return Err(t!("machines-name-reserved"));
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

/// A machine's TOML, with the older card flags read as the cards they
/// become, as a run builds them.
pub fn parse(text: &str) -> Result<MachineConfig, String> {
    let mut cfg = MachineConfig::from_toml_str(text).map_err(|e| e.to_string())?;
    cfg.fold_cards()?;
    Ok(cfg)
}

pub fn read(path: &Path) -> Result<MachineConfig, String> {
    let text = std::fs::read_to_string(path).map_err(|e| file_error(path, e))?;
    parse(&text).map_err(|e| file_error(path, e))
}

/// The TOML a machine is saved as. Keys the config left to the machine are
/// left out, so they keep their meaning after a reload: an automatic
/// `[jitv2] threads` (one per guest CPU, fewer on a small host), and
/// `banks` when not chosen and
/// still at the generic default (the Tezro and IP35 machines then fit their
/// own memory, as the core does for an unset `banks`).
pub fn to_toml(cfg: &MachineConfig) -> Result<String, String> {
    // Through text: `toml::Table::try_from` refuses the integer-keyed maps
    // (`[scsi.N]`) that the document writer accepts.
    let text = toml::to_string_pretty(&cfg.for_writing()).map_err(|e| e.to_string())?;
    let unset_banks = !cfg.banks_set && cfg.banks == MachineConfig::default().banks;
    if !unset_banks && cfg.jitv2_threads_set {
        return Ok(text);
    }
    let mut table: toml::Table = toml::from_str(&text).map_err(|e| e.to_string())?;
    if unset_banks {
        table.remove("banks");
    }
    if !cfg.jitv2_threads_set {
        if let Some(toml::Value::Table(jit)) = table.get_mut("jitv2") {
            jit.remove("threads");
            if jit.is_empty() {
                table.remove("jitv2");
            }
        }
    }
    toml::to_string_pretty(&table).map_err(|e| e.to_string())
}

/// A file that couldn't be read or written, as shown to the user.
pub fn file_error(path: &Path, error: impl std::fmt::Display) -> String {
    t!("machines-file-error", path = path.display(), error = error.to_string())
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
    result.map_err(|e| file_error(path, e))
}

/// Visit filesystem paths only. Network interfaces, MACs, subnets, debug module
/// filters, and TCP CI addresses are strings but must never be treated as paths.
pub fn visit_paths(cfg: &mut MachineConfig, mut visit: impl FnMut(&mut String)) {
    visit(&mut cfg.prom);
    visit(&mut cfg.nvram);
    visit(&mut cfg.nveeprom);
    for path in [
        &mut cfg.octane.flash, &mut cfg.graphics.flash,
        &mut cfg.origin200.flash, &mut cfg.origin200.controller_store, &mut cfg.origin200.io_prom,
        &mut cfg.origin2000.flash, &mut cfg.origin2000.controller_store, &mut cfg.origin2000.io_prom,
        &mut cfg.onyx2.flash, &mut cfg.onyx2.controller_store, &mut cfg.onyx2.io_prom,
        &mut cfg.fuel.flash, &mut cfg.fuel.l1_log,
        &mut cfg.tezro.flash, &mut cfg.tezro.l1_log,
        &mut cfg.origin3200.flash, &mut cfg.origin3200.l1_log,
        &mut cfg.onyx3000.flash, &mut cfg.onyx3000.l1_log,
        &mut cfg.serial_log_a, &mut cfg.serial_log_b,
        &mut cfg.serial_socket_a, &mut cfg.serial_socket_b,
        &mut cfg.vnc.password_file,
    ] {
        if let Some(path) = path { visit(path); }
    }
    visit(&mut cfg.virtio_9p.path);
    if let Some(file) = &mut cfg.audio.file {
        let mut path = file.to_string_lossy().into_owned();
        visit(&mut path);
        *file = path.into();
    }
    for port in &mut cfg.virtio_serial.ports {
        if let Some(path) = &mut port.socket { visit(path); }
    }
    for path in [
        &mut cfg.serial_log,
        &mut cfg.load_elf,
        &mut cfg.network.tftp_dir,
        &mut cfg.network.rsh_dir,
    ] {
        if let Some(path) = path {
            visit(path);
        }
    }
    visit(&mut cfg.jitv2.cache_dir);
    if !iris::config::ci_socket_is_tcp(&cfg.ci_socket) {
        visit(&mut cfg.ci_socket);
    }
    for d in &mut cfg.devices {
        match &mut d.device {
            DeviceModel::Scsi(dev) => {
                visit(&mut dev.path);
                for disc in &mut dev.discs {
                    visit(disc);
                }
            }
            DeviceModel::Nvme(n) => visit(&mut n.path),
            DeviceModel::TestDevice(t) => t.dump.iter_mut().for_each(&mut visit),
            DeviceModel::Virtio9p(v) => visit(&mut v.path),
            DeviceModel::Serial(s) => visit(&mut s.path),
            DeviceModel::VirtioSerial(v) => v.ports.iter_mut().filter_map(|p| p.socket.as_mut()).for_each(&mut visit),
            DeviceModel::XioBoard(_) | DeviceModel::PciBridge | DeviceModel::Ice1724 | DeviceModel::Qla2200 | DeviceModel::Ultra64 | DeviceModel::Newport | DeviceModel::Graphics => {}
        }
    }
    if let Some(nfs) = &mut cfg.nfs {
        visit(&mut nfs.shared_dir);
    }
}

pub fn resolve(base: &Path, path: &str) -> PathBuf {
    base.join(path)
}

/// Where the emulator keeps the `.diff.chd` of the CHD the machine's TOML
/// names as `path`. With `IRIS_CHD_DIFF_DIR` set (the App Store build) the
/// diff's name hashes the path as spelled, so this must start from the
/// TOML's spelling, not from the resolved path.
pub fn chd_diff(base: &Path, path: &str) -> PathBuf {
    base.join(iris::chd_disk::diff_path_for(Path::new(path)))
}

/// Express `path` relative to `base` only when it lives inside `base`. Paths
/// elsewhere stay absolute: a `../../..` chain to reach them is fragile, and
/// the user's choice of absolute spelling must survive a reload.
pub fn relative(base: &Path, path: &Path) -> Option<PathBuf> {
    if !base.is_absolute() || !path.is_absolute() {
        return None;
    }
    let rest = path.strip_prefix(base).ok()?;
    if rest.components().any(|c| c == Component::ParentDir) {
        return None;
    }
    Some(if rest.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        rest.to_path_buf()
    })
}

pub fn make_relative(cfg: &mut MachineConfig, dir: &Path) -> Result<(), String> {
    visit_paths(cfg, |path| {
        if let Some(p) = relative(dir, Path::new(path)) {
            *path = p.to_string_lossy().into_owned();
        }
    });
    Ok(())
}

pub fn save(root: &Path, name: &str, cfg: &mut MachineConfig) -> Result<(), String> {
    let dir = directory(root, name)?;
    make_relative(cfg, &dir)?;
    let text = to_toml(cfg)?;
    std::fs::create_dir_all(dir.join("disks")).map_err(|e| e.to_string())?;
    write(&config_path(root, name)?, &text)
}

/// True if `path` can actually be *read*. The start preflight uses this rather
/// than `Path::exists()`: under the macOS App Sandbox a file picked in an
/// earlier session with no bookmark still `stat()`s as present, and the
/// sandbox can permit `open()` yet deny `read()`, so only a real read tells.
pub fn readable(path: &str) -> bool {
    use std::io::Read;
    if iris::http_disk::is_url(path) {
        return true; // read over the network, not through the sandbox
    }
    if path.is_empty() {
        return false;
    }
    let Ok(mut f) = std::fs::File::open(path) else { return false };
    let mut probe = [0u8; 1];
    matches!(f.read(&mut probe), Ok(n) if n >= 1)
}

/// True if a file can be created (and removed) in `dir`: write access to the
/// folder itself. The exit-time CHD fold writes a temp beside the base and
/// renames it over it, which a file-scoped sandbox grant does not allow.
pub fn dir_writable(dir: &Path) -> bool {
    let probe = dir.join(format!(".iris-write-probe-{}", std::process::id()));
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// A writable CHD disk whose folder the app cannot write, so its exit-time
/// fold would be refused (macOS only: the sandbox, or TCC on a notarized build).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChdNeedsGrant {
    pub key: u8,
    pub path: String,
    /// The folder that needs a grant, absolute.
    pub dir: String,
}

/// The attached CHD disks (not CD-ROMs, not scratch) in folders the app cannot
/// write. Empty off macOS. Relative paths are taken against the working
/// directory, which is the machine's folder.
pub fn chd_dirs_needing_grant(cfg: &MachineConfig) -> Vec<ChdNeedsGrant> {
    if !cfg!(target_os = "macos") || cfg.machine.profile.is_ip35_bricks() {
        return Vec::new();
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut out: Vec<_> = crate::model::hardware::scsi_disks(cfg)
        .into_iter()
        .filter(|(_, d)| !d.scratch && !d.is_cdrom() && d.path.to_ascii_lowercase().ends_with(".chd"))
        .filter_map(|(key, d)| {
            let parent = resolve(&cwd, &d.path).parent()?.to_path_buf();
            (!dir_writable(&parent)).then(|| ChdNeedsGrant {
                key,
                path: d.path.clone(),
                dir: parent.to_string_lossy().into_owned(),
            })
        })
        .collect();
    out.sort_by_key(|c| c.key);
    out
}

/// Make `dir` the process working directory, creating it and its `disks`
/// folder first. Paths in a machine's TOML are relative to its folder, and the
/// emulator resolves them against the working directory.
pub fn enter(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir.join("disks")).map_err(|e| e.to_string())?;
    std::env::set_current_dir(dir)
        .map_err(|e| t!("toast-enter-dir-failed", path = dir.display(), error = e.to_string()))?;
    #[cfg(feature = "appstore")]
    std::env::set_var("IRIS_CHD_DIFF_DIR", dir.join("chd-diffs"));
    Ok(())
}

/// Files under this size that live in the source machine's folder are copied
/// by [`duplicate`] (NVRAM, EEPROM, flash); larger ones (disk images) are shared.
pub const DUPLICATE_COPY_LIMIT: u64 = 4 * 1024 * 1024;

/// Copy machine `src` to a new machine `dst`. Small files in the source folder
/// (battery-backed NVRAM and the like) are copied so the two machines keep
/// their own; larger ones, the disk images, are not copied: the duplicate
/// refers to the original's files by absolute path. Returns those shared
/// files, so the user can be told.
pub fn duplicate(root: &Path, src: &str, dst: &str) -> Result<(MachineConfig, Vec<String>), String> {
    let src_dir = directory(root, src)?;
    let dst_dir = directory(root, dst)?;
    let mut cfg = read(&config_path(root, src)?)?;
    std::fs::create_dir_all(dst_dir.join("disks")).map_err(|e| e.to_string())?;
    let mut shared = Vec::new();
    let mut failed = None;
    visit_paths(&mut cfg, |p| {
        if p.is_empty() || iris::dev::prom_image::is_embedded_keyword(p) || iris::http_disk::is_url(p) {
            return;
        }
        let from = resolve(&src_dir, p);
        let Some(inside) = relative(&src_dir, &from) else { return };
        let small = std::fs::metadata(&from).map(|m| m.is_file() && m.len() <= DUPLICATE_COPY_LIMIT);
        match small {
            Ok(true) => {
                let to = dst_dir.join(&inside);
                if let Some(parent) = to.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if let Err(e) = std::fs::copy(&from, &to) {
                    failed.get_or_insert_with(|| file_error(&from, e));
                }
                *p = inside.to_string_lossy().into_owned();
            }
            Ok(false) => {
                shared.push(from.to_string_lossy().into_owned());
                *p = from.to_string_lossy().into_owned();
            }
            // Not there (yet): keep the relative name, which the new machine
            // creates in its own folder when it runs.
            Err(_) => {}
        }
    });
    if let Some(e) = failed {
        return Err(e);
    }
    save(root, dst, &mut cfg)?;
    Ok((cfg, shared))
}

/// The machines under `root`, those whose file does not read (name and
/// why), and every error found.
pub fn discover(
    root: &Path,
) -> (
    std::collections::BTreeMap<String, MachineConfig>,
    std::collections::BTreeMap<String, String>,
    Vec<String>,
) {
    let mut machines = std::collections::BTreeMap::new();
    let mut needs_edit = std::collections::BTreeMap::new();
    let mut errors = Vec::new();
    let entries = match std::fs::read_dir(root.join("machines")) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (machines, needs_edit, errors),
        Err(e) => return (machines, needs_edit, vec![t!("machines-list-error", error = e.to_string())]),
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
                errors.push(file_error(&path, e));
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
            Err(e) => {
                needs_edit.insert(name, e.clone());
                errors.push(e);
            }
        }
    }
    (machines, needs_edit, errors)
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
        if !p.is_empty() && !iris::dev::prom_image::is_embedded_keyword(p) {
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
                Err(e) => {
                    return Err(t!("machines-migrate-error", path = source.display(), error = e.to_string()))
                }
            }
        }
        *path = leaf.into();
    }
    make_relative(&mut migrated, &dir)?;
    // App Store builds formerly redirected CHD diffs into one shared folder.
    // Copy any existing sidecar to the machine's redirect, using its new path
    // spelling for the hash. Keep the original for rollback.
    #[cfg(feature = "appstore")]
    for (dev, moved) in cfg.devices.iter().zip(&migrated.devices).filter_map(|(d, m)| Some((d.device.scsi()?, m.device.scsi()?))) {
        if !iris::chd_disk::is_chd(&dev.path) {
            continue;
        }
        let source = root.join("chd-diffs").join(diff_name(&dev.path));
        let dest = dir
            .join("chd-diffs")
            .join(diff_name(&moved.path));
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
        return Err(t!("machines-folder-exists", name = new));
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

/// A throwaway settings root under the system temp folder, removed on drop.
/// For tests here and in the shells.
#[doc(hidden)]
pub mod testing {
    use std::path::PathBuf;

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
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub use super::testing::TempRoot;

    #[test]
    fn duplicate_copies_small_files_and_shares_disks() {
        let root = TempRoot::new("dup");
        let mut cfg = MachineConfig::default();
        let src = directory(&root.0, "indy").unwrap();
        std::fs::create_dir_all(src.join("disks")).unwrap();
        std::fs::write(src.join("nvram.bin"), [7u8; 64]).unwrap();
        let disk = src.join("disks/root.img");
        let f = std::fs::File::create(&disk).unwrap();
        f.set_len(DUPLICATE_COPY_LIMIT + 1).unwrap();
        cfg.nvram = "nvram.bin".into();
        cfg.disk_mut(1).unwrap().path = "disks/root.img".into();
        save(&root.0, "indy", &mut cfg).unwrap();
        let (copy, shared) = duplicate(&root.0, "indy", "indy-2").unwrap();
        let dst = directory(&root.0, "indy-2").unwrap();
        assert_eq!(std::fs::read(dst.join("nvram.bin")).unwrap(), vec![7u8; 64]);
        assert_eq!(copy.nvram, "nvram.bin");
        assert_eq!(shared, vec![disk.to_string_lossy().into_owned()]);
        let reread = read(&config_path(&root.0, "indy-2").unwrap()).unwrap();
        assert_eq!(reread.disk(1).unwrap().path, disk.to_string_lossy());
    }

    #[test]
    fn saved_toml_roundtrips_every_path_and_preserves_non_path_strings() {
        let root = TempRoot::new("paths");
        let dir = directory(&root.0, "Indigo2 IP28").unwrap();
        let mut cfg = MachineConfig::default();
        cfg.prom = dir.join("prom.bin").to_string_lossy().into_owned();
        cfg.nvram = dir.join("nvram.bin").to_string_lossy().into_owned();
        cfg.nveeprom = dir.join("nveeprom.bin").to_string_lossy().into_owned();
        cfg.disk_mut(1).unwrap().path =
            dir.join("disks/root.chd").to_string_lossy().into_owned();
        cfg.disk_mut(4).unwrap().path.clear();
        cfg.disk_mut(4).unwrap().discs =
            vec![root.0.join("install.iso").to_string_lossy().into_owned()];
        cfg.nfs = Some(iris::config::NfsConfig {
            shared_dir: root.0.join("shared").to_string_lossy().into_owned(),
            version: Default::default(),
        });
        cfg.network.tftp_dir = Some(root.0.join("tftp").to_string_lossy().into_owned());
        cfg.serial_log = Some(dir.join("console.log").to_string_lossy().into_owned());
        cfg.load_elf = Some(root.0.join("test.elf").to_string_lossy().into_owned());
        cfg.set_device("test_device", true);
        if let Some(DeviceModel::TestDevice(t)) = cfg.devices.last_mut().map(|d| &mut d.device) {
            t.dump = Some(dir.join("dump.json").to_string_lossy().into_owned());
        }
        cfg.jitv2.cache_dir = dir.join("jit-cache").to_string_lossy().into_owned();
        cfg.ci_socket = "tcp:127.0.0.1:19851".into();
        cfg.debug.debug_log = "scsi=debug,cpu=warn".into();
        cfg.network.pcap_interface = Some("\\Device\\NPF_{example}".into());
        cfg.network.mac = Some("08:00:69:12:34:56".into());
        save(&root.0, "Indigo2 IP28", &mut cfg).unwrap();
        let path = config_path(&root.0, "Indigo2 IP28").unwrap();
        let back = read(&path).unwrap();
        // Saved in the host's spelling: disks\root.chd on Windows.
        assert_eq!(Path::new(&back.disk(1).unwrap().path), Path::new("disks/root.chd"));
        assert_eq!(back.disk(4).unwrap().path, "");
        // External paths keep their absolute spelling.
        assert_eq!(back.disk(4).unwrap().discs[0], root.0.join("install.iso").to_string_lossy());
        assert_eq!(back.nvram, "nvram.bin");
        assert_eq!(back.nveeprom, "nveeprom.bin");
        assert_eq!(back.ci_socket, "tcp:127.0.0.1:19851");
        assert_eq!(back.debug.debug_log, cfg.debug.debug_log);
        assert_eq!(back.network.pcap_interface, cfg.network.pcap_interface);
        assert_eq!(back.network.mac, cfg.network.mac);
        assert_eq!(back.nfs.as_ref().unwrap().shared_dir, root.0.join("shared").to_string_lossy());
        assert_eq!(back.serial_log.as_deref(), Some("console.log"));
        assert!(dir.join("disks").is_dir());
    }

    #[test]
    fn expansion_and_platform_paths_survive_save_and_reload() {
        let root = TempRoot::new("platform-paths");
        let dir = directory(&root.0, "tezro").unwrap();
        let mut cfg = MachineConfig::default();
        cfg.tezro.flash = Some(dir.join("flash.bin").to_string_lossy().into_owned());
        cfg.origin2000.io_prom = Some(dir.join("io6prom.img").to_string_lossy().into_owned());
        let nvme = iris::config::NvmeConfig { path: dir.join("disk.raw").to_string_lossy().into_owned(), ..Default::default() };
        cfg.devices.push(iris::devices::ConfiguredDevice::new("pci_slot", iris::devices::DeviceModel::Nvme(nvme)));
        cfg.machine.profile = iris::config::MachineProfile::Origin3200Ip35;
        cfg.devices.push(iris::devices::ConfiguredDevice::new("xio15/pci0/1/loop1", iris::devices::DeviceModel::Scsi(iris::config::ScsiDeviceConfig {
            path: dir.join("fc.raw").to_string_lossy().into_owned(), ..Default::default()
        })));
        cfg.virtio_serial.ports.push(iris::config::VirtioSerialPort {
            socket: Some(dir.join("port.sock").to_string_lossy().into_owned()), ..Default::default()
        });
        save(&root.0, "tezro", &mut cfg).unwrap();
        let back = read(&config_path(&root.0, "tezro").unwrap()).unwrap();
        assert_eq!(back.tezro.flash.as_deref(), Some("flash.bin"));
        assert_eq!(back.origin2000.io_prom.as_deref(), Some("io6prom.img"));
        assert_eq!(back.nvmes().next().unwrap().path, "disk.raw");
        assert_eq!(back.loop_disks().iter().map(|d| back.devices[d.1].device.scsi().unwrap().path.as_str()).collect::<Vec<_>>(), ["fc.raw"]);
        assert_eq!(back.virtio_serial.ports[0].socket.as_deref(), Some("port.sock"));
    }

    #[test]
    fn share_audio_and_vnc_paths_survive_save_and_reload() {
        let root = TempRoot::new("share-paths");
        let dir = directory(&root.0, "octane").unwrap();
        let mut cfg = MachineConfig::default();
        cfg.virtio_9p.path = dir.join("share").to_string_lossy().into_owned();
        cfg.audio.file = Some(dir.join("out.wav"));
        cfg.vnc.password_file = Some(dir.join("vnc.pass").to_string_lossy().into_owned());
        save(&root.0, "octane", &mut cfg).unwrap();
        let back = read(&config_path(&root.0, "octane").unwrap()).unwrap();
        assert_eq!(back.virtio_9p.path, "share");
        assert_eq!(back.audio.file.as_deref(), Some(Path::new("out.wav")));
        assert_eq!(back.vnc.password_file.as_deref(), Some("vnc.pass"));
    }

    /// `banks` a config left unset stays unset across a save, so a Tezro
    /// keeps its own memory; banks the user chose are kept.
    #[test]
    fn unchosen_banks_stay_unchosen_across_save() {
        let root = TempRoot::new("banks");
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = iris::config::MachineProfile::TezroIp53;
        save(&root.0, "unset", &mut cfg).unwrap();
        let back = read(&config_path(&root.0, "unset").unwrap()).unwrap();
        assert!(!back.banks_set);
        back.validate().unwrap();

        let mut cfg = MachineConfig::default();
        cfg.banks = [64, 64, 0, 0, 0, 0, 0, 0];
        cfg.banks_set = true;
        save(&root.0, "chosen", &mut cfg).unwrap();
        let back = read(&config_path(&root.0, "chosen").unwrap()).unwrap();
        assert!(back.banks_set);
        assert_eq!(back.banks, cfg.banks);
    }

    /// An unset `[jitv2] threads` stays unset across a save, so it keeps
    /// following the guest's CPU count and the host's usable CPUs; a number
    /// the user chose is kept.
    #[test]
    fn automatic_jit_threads_survive_save_and_reload() {
        let root = TempRoot::new("jit-threads");
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = iris::config::MachineProfile::TezroIp53;
        cfg.machine.cpus = Some(4);
        cfg.settle_jitv2_threads();
        save(&root.0, "auto", &mut cfg).unwrap();
        let text = std::fs::read_to_string(config_path(&root.0, "auto").unwrap()).unwrap();
        let saved: toml::Table = toml::from_str(&text).unwrap();
        assert!(saved.get("jitv2").and_then(|j| j.get("threads")).is_none(), "{text}");
        let back = read(&config_path(&root.0, "auto").unwrap()).unwrap();
        assert!(!back.jitv2_threads_set);
        // The automatic size follows the host as well as the guest, so it
        // is whatever this host would choose for a 4-CPU guest.
        let expected = iris::config::auto_jitv2_threads(4, iris::config::host_parallelism()).0;
        assert_eq!(back.jitv2.threads, expected);

        cfg.jitv2_threads_set = true;
        cfg.jitv2.threads = 2;
        save(&root.0, "fixed", &mut cfg).unwrap();
        let back = read(&config_path(&root.0, "fixed").unwrap()).unwrap();
        assert!(back.jitv2_threads_set);
        assert_eq!(back.jitv2.threads, 2);
    }

    /// The GUI sees the older card flags as the cards they become, so its
    /// settings and its slot check agree with what Start builds.
    #[test]
    fn older_card_flags_read_as_cards() {
        let root = TempRoot::new("older-cards");
        for (text, card) in [
            ("[ultra64]\nenabled = true\n", "ultra64"),
            ("[graphics]\nheads = 2\n", "newport"),
            ("test_device = true\n", "test_device"),
            ("[virtio_serial]\nenabled = true\n", "virtio_serial"),
            ("[virtio_9p]\nenabled = true\npath = \"/share\"\n", "virtio_9p"),
        ] {
            let dir = directory(&root.0, "indy").unwrap();
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(config_path(&root.0, "indy").unwrap(), text).unwrap();
            let cfg = read(&config_path(&root.0, "indy").unwrap()).unwrap();
            assert!(cfg.has_device(card), "{card}");
            assert!(to_toml(&cfg).unwrap().contains(&format!("type = \"{card}\"")), "{card}: saved as the card");
        }
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
        let (machines, needs_edit, errors) = discover(&root.0);
        assert_eq!(machines.keys().collect::<Vec<_>>(), vec!["indy"]);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("broken.toml"));
        assert!(errors[0].contains("unknown_setting"));
        assert_eq!(needs_edit.keys().collect::<Vec<_>>(), vec!["broken"]);
        assert_eq!(needs_edit["broken"], errors[0]);
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
            std::fs::read(indy.join(&migrated.disk(1).unwrap().path)).unwrap(),
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
        cfg.disk_mut(1).unwrap().path = "disks/root.raw".into();
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
            std::fs::read(new_dir.join(&cfg.disk(1).unwrap().path)).unwrap(),
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
    fn only_paths_inside_the_machine_folder_become_relative() {
        // Built on the temp folder: "/m" has no drive, so Windows calls it relative.
        let base = std::env::temp_dir();
        let dir = base.join("m/machines/indy");
        assert_eq!(
            relative(&dir, &dir.join("disks/a.raw")),
            Some(PathBuf::from("disks/a.raw"))
        );
        let outside = base.join("Volumes/X/a.chd").to_string_lossy().into_owned();
        assert_eq!(relative(&dir, Path::new(&outside)), None);
        let mut cfg = MachineConfig::default();
        cfg.disk_mut(1).unwrap().path = outside.clone();
        make_relative(&mut cfg, &dir).unwrap();
        assert_eq!(cfg.disk(1).unwrap().path, outside);
    }

    #[test]
    fn relative_external_resources_are_bookmarked_from_each_machine_folder() {
        let root = TempRoot::new("bookmarks");
        let dir = directory(&root.0, "indy").unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        let resource = root.0.join("external.iso");
        std::fs::write(&resource, b"media").unwrap();
        let mut cfg = MachineConfig::default();
        cfg.disk_mut(4).unwrap().path = "../../external.iso".into();
        let paths = crate::model::macos_sandbox::config_paths(&cfg, &dir);
        assert!(paths.contains(&resource.to_string_lossy().into_owned()));
    }

    #[cfg(feature = "appstore")]
    #[test]
    fn sandbox_migration_preserves_shared_chd_diffs_in_each_machine_folder() {
        let root = TempRoot::new("diffs");
        std::fs::create_dir_all(root.0.join("chd-diffs")).unwrap();
        let mut cfg = MachineConfig::default();
        cfg.disk_mut(1).unwrap().path = root.0.join("root.chd").to_string_lossy().into_owned();
        let old_diff = root.0.join("chd-diffs").join(diff_name(&cfg.disk(1).unwrap().path));
        std::fs::write(&old_diff, b"guest writes").unwrap();
        migrate(&root.0, "indy", &cfg, &root.0).unwrap();
        let migrated = read(&config_path(&root.0, "indy").unwrap()).unwrap();
        let new_diff = directory(&root.0, "indy")
            .unwrap()
            .join("chd-diffs")
            .join(diff_name(&migrated.disk(1).unwrap().path));
        assert_eq!(std::fs::read(new_diff).unwrap(), b"guest writes");
        assert!(old_diff.exists());
    }
}

#[cfg(test)]
mod readable_tests {
    use super::readable as disk_readable;
    use std::io::Write;

    #[test]
    fn empty_and_missing_paths_are_not_readable() {
        assert!(!disk_readable(""));
        assert!(!disk_readable("/no/such/iris/disk/image.chd"));
    }

    #[test]
    fn an_existing_readable_file_is_readable() {
        let mut p = std::env::temp_dir();
        p.push(format!("iris_readable_{}.raw", std::process::id()));
        std::fs::File::create(&p).unwrap().write_all(b"\0\0\0\0").unwrap();
        assert!(disk_readable(&p.to_string_lossy()));
        let _ = std::fs::remove_file(&p);
    }

    // The case the sandbox actually hits: a file that exists (so Path::exists()
    // is true) but cannot be opened. Simulated with chmod 000. Skipped when the
    // test runs as root, where DAC permission checks are bypassed and the open
    // would still succeed.
    #[cfg(unix)]
    #[test]
    fn an_unreadable_but_present_file_is_not_readable() {
        use std::os::unix::fs::PermissionsExt;
        if unsafe { libc_geteuid() } == 0 {
            return; // root bypasses 000 perms; the distinction can't be shown
        }
        let mut p = std::env::temp_dir();
        p.push(format!("iris_unreadable_{}.raw", std::process::id()));
        std::fs::File::create(&p).unwrap().write_all(b"\0\0\0\0").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o000)).unwrap();
        // exists() says present, but disk_readable() must say no.
        assert!(p.exists());
        assert!(!disk_readable(&p.to_string_lossy()));
        let _ = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600));
        let _ = std::fs::remove_file(&p);
    }

    #[cfg(unix)]
    extern "C" {
        #[link_name = "geteuid"]
        fn libc_geteuid() -> u32;
    }
}
