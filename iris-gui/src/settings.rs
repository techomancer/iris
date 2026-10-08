use iris::config::MachineConfig;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// GUI preferences at `<config dir>/iris/gui.json`. Machine configurations
/// use the core TOML schema in `machines/<name>/<name>.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuiSettings {
    /// egui UI scale (default `UI_SCALE_DEFAULT`, currently 1.25).
    #[serde(default = "default_ui_scale")]
    pub ui_scale: f32,
    /// Emulated-display (VM screen) magnification (default `VM_SCALE_DEFAULT`,
    /// currently 0.75). Driven by the View-menu slider (0.5×–3× in 0.25
    /// steps), **independent of `ui_scale`** — scaling the controls doesn't
    /// resize the picture, and vice-versa.
    #[serde(default = "default_vm_scale")]
    pub vm_scale: f32,

    /// Runtime machine list, discovered from TOML folders. Deserialize the old
    /// JSON field for migration, but never serialize configs back into JSON.
    /// BTreeMap keeps menu entries in alphabetical order.
    #[serde(default, skip_serializing)]
    pub machines: BTreeMap<String, MachineConfig>,
    /// Currently-selected machine (key into `machines`). None = no
    /// machine loaded yet (first run).
    #[serde(default)]
    pub active_machine: Option<String>,

    // Retained for compatibility with existing GUI preferences.
    /// Legacy recent TOML files; no longer updated by the GUI.
    #[serde(default)]
    pub recent_configs: Vec<PathBuf>,
    /// Legacy TOML pointer, consumed once during migration.
    #[serde(default)]
    pub last_config: Option<PathBuf>,

    /// macOS App Sandbox security-scoped bookmarks, keyed by the absolute file
    /// path they re-grant access to (disk image, PROM, ISO, NFS dir, …). Minted
    /// at save time and resolved at startup so user-selected files reopen across
    /// launches under the Mac App Store sandbox. Empty / unused everywhere else.
    /// See [`crate::macos_sandbox`].
    #[serde(default)]
    pub bookmarks: BTreeMap<String, Vec<u8>>,

    /// Folders the user has granted access to under the Mac App Store sandbox
    /// (via "Grant disks folder…"). A *directory* security-scoped bookmark is
    /// recursive, so granting one folder covers every disk image, the CHD diff /
    /// fold temp written beside a base, AND an NFS shared subfolder under it — so
    /// the "Synchronizing disks" fold (which needs to create a sibling temp and
    /// rename over the base) works without per-file grants. Empty off the App
    /// Store build. See [`crate::macos_sandbox`].
    #[serde(default)]
    pub disk_folders: Vec<String>,

    /// Runtime storage root, captured before the working directory changes.
    #[serde(skip)]
    root: Option<PathBuf>,
    /// Load/migration errors shown by the launcher and logged.
    #[serde(skip)]
    pub errors: Vec<String>,
    /// Protect the legacy JSON if any conversion or preference read failed.
    #[serde(skip)]
    migration_pending: bool,
}

impl Default for GuiSettings {
    fn default() -> Self {
        Self {
            ui_scale: UI_SCALE_DEFAULT,
            vm_scale: VM_SCALE_DEFAULT,
            machines: BTreeMap::new(),
            active_machine: None,
            recent_configs: Vec::new(),
            last_config: None,
            bookmarks: BTreeMap::new(),
            disk_folders: Vec::new(),
            root: None,
            errors: Vec::new(),
            migration_pending: false,
        }
    }
}

/// Byte offset of the Indy's 6-byte Ethernet MAC inside the NVRAM. The PROM
/// reads the MAC from these *raw bytes* — it is NOT the colon-separated ASCII
/// you type at `setenv` (that's just the human entry form). Reverse-engineered
/// from firmware-written NVRAMs (the SGI OUI 08:00:69 lands exactly here, with
/// zero bytes around it and no adjacent checksum). Like the `console` byte the
/// headless path patches, this is a fixed, PROM-specific offset.
pub const NVRAM_MAC_OFFSET: usize = 0x13a;

/// The 6 raw MAC bytes from an NVRAM file, if it holds a non-blank one.
pub fn nvram_mac(path: &str) -> Option<[u8; 6]> {
    let b = std::fs::read(path).ok()?;
    let m: [u8; 6] = b.get(NVRAM_MAC_OFFSET..NVRAM_MAC_OFFSET + 6)?.try_into().ok()?;
    let blank = m.iter().all(|&x| x == 0x00) || m.iter().all(|&x| x == 0xff);
    (!blank).then_some(m)
}

/// Whether the NVRAM already has an Ethernet MAC (so IRIX can attach `ec0`).
pub fn nvram_has_mac(path: &str) -> bool {
    nvram_mac(path).is_some()
}

/// Deterministic SGI-OUI MAC bytes (`08:00:69:xx:xx:xx`) from `seed` (machine
/// name) — stable per machine. Uniqueness across instances doesn't matter; each
/// runs on its own isolated NAT.
pub fn generate_mac_bytes(seed: &str) -> [u8; 6] {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut h);
    let v = h.finish();
    [0x08, 0x00, 0x69, (v >> 16) as u8, (v >> 8) as u8, v as u8]
}

/// Human form `08:00:69:xx:xx:xx` for display/logging.
pub fn mac_to_string(m: [u8; 6]) -> String {
    m.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(":")
}

/// Write 6 MAC bytes into an existing NVRAM file at [`NVRAM_MAC_OFFSET`],
/// touching only those 6 bytes so the boot env is preserved. Backs the file up
/// to `<path>.bak` first. Returns Ok(false) if there's no NVRAM file yet (a
/// bare MAC with no DS1386 structure would be useless) or it's too small.
pub fn write_nvram_mac(path: &str, mac: [u8; 6]) -> std::io::Result<bool> {
    let Ok(mut bytes) = std::fs::read(path) else { return Ok(false); };
    if bytes.len() < NVRAM_MAC_OFFSET + 6 {
        return Ok(false);
    }
    let _ = std::fs::copy(path, format!("{path}.bak")); // best-effort backup
    bytes[NVRAM_MAC_OFFSET..NVRAM_MAC_OFFSET + 6].copy_from_slice(&mac);
    std::fs::write(path, &bytes)?;
    Ok(true)
}

/// Default NVRAM image baked into the binary: the repo's known-good NVRAM (boot
/// env present) with the MAC zeroed. Lets a fresh install — especially the
/// bundled `.app`, which has nothing in its working dir to migrate — boot with
/// proper PROM env, while the auto-write fills in a per-machine MAC.
pub const DEFAULT_NVRAM: &[u8] = include_bytes!("../assets/nvram-default.bin");

/// Write the embedded default NVRAM to `path` if there's no (non-empty) file
/// there yet. Returns true if it seeded one. Creates the parent dir as needed.
pub fn ensure_nvram_seeded(path: &str) -> bool {
    if std::fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false) {
        return false;
    }
    if let Some(parent) = Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(path, DEFAULT_NVRAM).is_ok()
}

/// Overwrite the NVRAM at `path` with the embedded default (boot env, blank
/// MAC) — backs the current file up to `<path>.bak` first. Used by the
/// "Reset NVRAM / fresh PRAM" menu action.
pub fn reset_nvram(path: &str) -> std::io::Result<()> {
    if std::fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false) {
        let _ = std::fs::copy(path, format!("{path}.bak"));
    }
    if let Some(parent) = Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(path, DEFAULT_NVRAM)
}

/// Byte offset of the Ethernet MAC inside the NVRAM *EEPROM* (93CS56) file —
/// word 0x7D, 2 bytes per word, big-endian (`src/dev/eeprom_93c56.rs`'s
/// `backdoor_set_mac`: word 0x7D = MAC[0]<<8|MAC[1], …, so the three words'
/// raw bytes are the 6 MAC bytes in order). **Indigo2/IP28 read `eaddr` from
/// here, not from [`NVRAM_MAC_OFFSET`]** — that offset is the DS1386 chip,
/// which those profiles don't use for the MAC at all. A build that only ever
/// patches `NVRAM_MAC_OFFSET` silently has no effect on those profiles' guest
/// networking; see `rules/irix/networking.md`.
pub const NVEEPROM_MAC_OFFSET: usize = 0x7D * 2;

/// The blank/erased size of a 93C56: 128 words × 2 bytes.
const NVEEPROM_SIZE: usize = 128 * 2;

/// The 6 raw MAC bytes from an NVRAM EEPROM file, if it holds a non-blank one.
/// A real 93C56 reads all-`0xFF` when erased (never all-zero — that's the
/// DS1386's blank state, not this chip's), but `nvram_mac`'s both-sentinels
/// check already covers it, so the same logic works unmodified.
pub fn nveeprom_mac(path: &str) -> Option<[u8; 6]> {
    let b = std::fs::read(path).ok()?;
    let m: [u8; 6] = b.get(NVEEPROM_MAC_OFFSET..NVEEPROM_MAC_OFFSET + 6)?.try_into().ok()?;
    let blank = m.iter().all(|&x| x == 0x00) || m.iter().all(|&x| x == 0xff);
    (!blank).then_some(m)
}

/// Whether the NVRAM EEPROM already has an Ethernet MAC.
pub fn nveeprom_has_mac(path: &str) -> bool {
    nveeprom_mac(path).is_some()
}

/// Write 6 MAC bytes into an existing NVRAM EEPROM file at
/// [`NVEEPROM_MAC_OFFSET`], touching only those 6 bytes. Backs the file up to
/// `<path>.bak` first. Returns `Ok(false)` if there's no file yet or it's too
/// small — call [`ensure_nveeprom_exists`] first to create a blank one.
pub fn write_nveeprom_mac(path: &str, mac: [u8; 6]) -> std::io::Result<bool> {
    let Ok(mut bytes) = std::fs::read(path) else { return Ok(false); };
    if bytes.len() < NVEEPROM_MAC_OFFSET + 6 {
        return Ok(false);
    }
    let _ = std::fs::copy(path, format!("{path}.bak")); // best-effort backup
    bytes[NVEEPROM_MAC_OFFSET..NVEEPROM_MAC_OFFSET + 6].copy_from_slice(&mac);
    std::fs::write(path, &bytes)?;
    Ok(true)
}

/// Create a blank (all-erased, `0xFF`) 256-byte EEPROM image at `path` if
/// there's no (non-empty) file there yet. Returns true if it created one.
/// Unlike [`ensure_nvram_seeded`], there's no baked-in asset to seed from —
/// `0xFF`-erased is exactly a brand new 93C56's real power-on state, and
/// `Eeprom93c56::backdoor_set_mac_if_blank` (core, at `Machine::new`) already
/// fills in a MAC from a blank one, same as the DS1386 path. On IP28 the
/// core also initializes PROM defaults before the first boot, including
/// volume and boottune; Stop persists the initialized image.
pub fn ensure_nveeprom_exists(path: &str) -> bool {
    if std::fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false) {
        return false;
    }
    if let Some(parent) = Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(path, vec![0xFFu8; NVEEPROM_SIZE]).is_ok()
}

/// Overwrite the NVRAM EEPROM at `path` with a blank (`0xFF`-erased) image —
/// backs the current file up to `<path>.bak` first. Used by the
/// "Reset NVRAM / fresh PRAM" menu action alongside [`reset_nvram`], so a
/// reset actually clears whichever chip the current profile reads `eaddr`
/// and PROM env from, not just the DS1386 one.
pub fn reset_nveeprom(path: &str) -> std::io::Result<()> {
    if std::fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false) {
        let _ = std::fs::copy(path, format!("{path}.bak"));
    }
    if let Some(parent) = Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(path, vec![0xFFu8; NVEEPROM_SIZE])
}

/// Allowed UI-scale range, shared by the View-menu slider, the Ctrl +/-/0
/// keyboard zoom, and the load-time clamp so a stale persisted value can never
/// put the UI into a state the slider can't represent (which egui would then
/// silently re-clamp to its own bound).
pub const UI_SCALE_MIN: f32 = 1.0;
pub const UI_SCALE_MAX: f32 = 3.0;
pub const UI_SCALE_DEFAULT: f32 = 1.25;

/// Allowed VM-screen scale range and step for the View-menu slider. ¼× steps
/// (0.5, 0.75, 1.0, 1.25, …) give finer control; on a HiDPI (2×) display the
/// half-integer steps (0.5, 1.0, 1.5, …) are pixel-crisp and the ¼ steps in
/// between are bilinear-smoothed — the footer readout tags which is which.
pub const VM_SCALE_MIN: f32 = 0.5;
pub const VM_SCALE_MAX: f32 = 3.0;
pub const VM_SCALE_STEP: f64 = 0.25;
/// Default windowed VM scale. 0.75 (not native 1.0) so a from-scratch window
/// opens *target-bound* on a typical laptop — sized to the picture exactly,
/// rather than "as big as the monitor allows" which clamps to a fractional
/// scale and leaves letterbox slack around the 5:4 display.
pub const VM_SCALE_DEFAULT: f32 = 0.75;

/// First-launch window size in logical points. Sized to match the *running*
/// window for the standard 1280×1024 display so the picture doesn't visibly
/// jump when you press Start: with the left control column (~186 pt) and no
/// top/bottom chrome, the running size at the default UI scale is ≈ the native
/// 1280×1024 display plus the column width. The launcher fit (see `main`) and
/// the on-Start snap still refine this — clamping to the monitor on smaller
/// screens — so it's only the initial size and the fallback when the monitor
/// size is unknown. Once a real size is persisted to `gui.json`, that's used.
pub const WINDOW_DEFAULT_SIZE: [f32; 2] = [1512.0, 1024.0];

fn default_ui_scale() -> f32 { UI_SCALE_DEFAULT }
fn default_vm_scale() -> f32 { VM_SCALE_DEFAULT }

impl GuiSettings {
    /// Foundation returns the actual container home for sandboxed macOS apps,
    /// including launches from a shell whose HOME still points outside it.
    pub fn data_dir() -> Option<PathBuf> {
        #[cfg(all(target_os = "macos", feature = "appstore"))]
        {
            let home = objc2_foundation::NSHomeDirectory().to_string();
            Some(PathBuf::from(home).join("Library/Application Support/iris"))
        }
        #[cfg(not(all(target_os = "macos", feature = "appstore")))]
        dirs::config_dir().map(|d| d.join("iris"))
    }

    pub fn root(&self) -> Result<&Path, String> {
        self.root.as_deref().ok_or_else(|| "No GUI config directory".into())
    }

    pub fn machine_dir(&self, name: &str) -> Result<PathBuf, String> {
        crate::machines::directory(self.root()?, name)
    }

    pub fn machine_path(&self, name: &str) -> Result<PathBuf, String> {
        crate::machines::config_path(self.root()?, name)
    }

    pub fn default_nvram_path() -> String { "nvram.bin".into() }
    pub fn default_nveeprom_path() -> String { "nveeprom.bin".into() }

    /// The selected machine's folder is the process working directory.
    pub fn working_dir() -> Option<PathBuf> { std::env::current_dir().ok() }
    pub fn disks_dir() -> Option<PathBuf> { Self::working_dir().map(|d| d.join("disks")) }
    pub fn default_disk_path(scsi_id: u8) -> String { format!("disks/scsi{scsi_id}.raw") }

    pub fn load() -> Self {
        match (Self::data_dir(), std::env::current_dir()) {
            (Some(root), Ok(cwd)) => Self::load_in(root, &cwd),
            _ => Self { errors: vec!["Cannot locate GUI storage or working directory".into()], ..Default::default() },
        }
    }

    pub(crate) fn load_in(root: PathBuf, old_cwd: &Path) -> Self {
        let directory_error = std::fs::create_dir_all(&root).err();
        // cwd follows symlinks; use the same base for relative TOML paths.
        let root = root.canonicalize().unwrap_or(root);
        let path = root.join("gui.json");
        let mut s = match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<Self>(&text) {
                Ok(s) => s,
                Err(e) => Self { migration_pending: true, errors: vec![format!("{}: {e}", path.display())], ..Default::default() },
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(e) => Self { migration_pending: true, errors: vec![format!("{}: {e}", path.display())], ..Default::default() },
        };
        s.root = Some(root.clone());
        if let Some(e) = directory_error {
            s.migration_pending = true;
            s.errors.push(format!("Cannot create GUI storage: {e}"));
        }
        // Restore access before migrating any external battery-backed files.
        crate::macos_sandbox::restore(&s.bookmarks);
        s.ui_scale = if !s.ui_scale.is_finite() || s.ui_scale < UI_SCALE_MIN {
            UI_SCALE_DEFAULT
        } else { s.ui_scale.min(UI_SCALE_MAX) };
        s.vm_scale = if !s.vm_scale.is_finite() || s.vm_scale < VM_SCALE_MIN {
            VM_SCALE_DEFAULT
        } else { s.vm_scale.min(VM_SCALE_MAX) };

        if !s.migration_pending && (!s.machines.is_empty() || s.last_config.is_some()) {
            // Leave this original backup intact across retries.
            let backup = root.join("gui.json.pre-toml.bak");
            let result = (|| -> Result<(), String> {
                if !backup.exists() {
                    std::fs::copy(&path, &backup).map_err(|e| format!("Cannot back up GUI preferences: {e}"))?;
                }
                for (old_name, cfg) in &s.machines {
                    let name = crate::machines::legacy_name(old_name);
                    crate::machines::migrate(&root, &name, cfg, old_cwd)?;
                    if s.active_machine.as_deref() == Some(old_name) { s.active_machine = Some(name); }
                }
                if s.machines.is_empty() {
                    if let Some(legacy) = &s.last_config {
                        let source = old_cwd.join(legacy);
                        let cfg = crate::machines::read(&source)?;
                        let stem = source.file_stem().and_then(|n| n.to_str()).unwrap_or("imported");
                        let base = crate::machines::legacy_name(stem);
                        let name = if crate::machines::config_path(&root, &base)?.exists() { base } else { s.unique_name(&base) };
                        crate::machines::migrate(&root, &name, &cfg, old_cwd)?;
                        s.active_machine = Some(name);
                    }
                }
                Ok(())
            })();
            match result {
                Ok(()) => {
                    s.last_config = None;
                    s.machines.clear();
                    if let Err(e) = s.save() { s.migration_pending = true; s.errors.push(e); }
                }
                Err(e) => { s.migration_pending = true; s.errors.push(format!("Machine migration failed: {e}")); }
            }
        }
        let (machines, errors) = crate::machines::discover(&root);
        s.machines = machines;
        s.errors.extend(errors);
        for e in &s.errors { log::error!("{e}"); }
        s
    }

    pub fn ensure_writable(&self) -> Result<(), String> {
        if self.migration_pending {
            Err("Resolve the GUI preference/migration error before changing machines".into())
        } else { Ok(()) }
    }

    pub fn save_machine(&mut self, name: &str, cfg: &mut MachineConfig) -> Result<(), String> {
        self.ensure_writable()?;
        crate::machines::save(self.root()?, name, cfg)?;
        self.machines.insert(name.into(), cfg.clone());
        self.save()
    }

    pub fn save(&mut self) -> Result<(), String> {
        if self.migration_pending { return Err("GUI preferences retained because loading or migration failed".into()); }
        let root = self.root()?.to_path_buf();
        let paths: Vec<String> = self.machines.iter().flat_map(|(name, cfg)| {
            let dir = crate::machines::directory(&root, name).unwrap();
            crate::macos_sandbox::config_paths(cfg, &dir)
        }).collect();
        crate::macos_sandbox::harvest(
            paths.iter().map(String::as_str).chain(self.disk_folders.iter().map(String::as_str)),
            &mut self.bookmarks,
        );
        std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        crate::machines::write(&root.join("gui.json"), &text)
    }

    fn name_taken(&self, name: &str) -> bool {
        self.machines.contains_key(name)
            || self.machine_dir(name).map(|d| d.exists()).unwrap_or(true)
    }

    /// Pick a free name like "indy", "indy-2", "indy-3", …
    pub fn unique_name(&self, base: &str) -> String {
        if !self.name_taken(base) { return base.to_string(); }
        for n in 2..1000 {
            let candidate = format!("{base}-{n}");
            if !self.name_taken(&candidate) { return candidate; }
        }
        format!("{base}-{}", uuid_like())
    }
}

fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0).to_string()
}

#[cfg(test)]
mod storage_tests {
    use super::*;
    use crate::machines::{self, tests::TempRoot};

    #[test]
    fn legacy_json_migration_keeps_preferences_and_removes_only_machine_payloads() {
        let root = TempRoot::new("json");
        let mut cfg = MachineConfig::default();
        cfg.scsi.get_mut(&1).unwrap().path = "boot.raw".into();
        let original = serde_json::json!({
            "ui_scale": 1.5, "vm_scale": 1.0, "active_machine": "indy",
            "machines": {"indy": cfg}, "recent_configs": ["old.toml"],
            "last_config": null, "bookmarks": {"/old/disk": [1,2,3]},
            "disk_folders": ["/old/folder"]
        });
        let original_text = serde_json::to_string(&original).unwrap();
        std::fs::write(root.0.join("gui.json"), &original_text).unwrap();
        let mut prefs = GuiSettings::load_in(root.0.clone(), &root.0);
        assert!(prefs.errors.is_empty(), "{:?}", prefs.errors);
        assert_eq!(prefs.active_machine.as_deref(), Some("indy"));
        assert_eq!(prefs.machines["indy"].scsi[&1].path, "../../boot.raw");
        assert_eq!(std::fs::read_to_string(root.0.join("gui.json.pre-toml.bak")).unwrap(), original_text);
        prefs.save().unwrap();
        let persisted: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.0.join("gui.json")).unwrap()).unwrap();
        assert!(persisted.get("machines").is_none());
        for key in ["ui_scale", "vm_scale", "active_machine", "recent_configs", "last_config", "bookmarks", "disk_folders"] {
            assert_eq!(persisted[key], original[key], "{key}");
        }
        // Preference saves do not overwrite externally edited machine TOMLs.
        let path = prefs.machine_path("indy").unwrap();
        let mut cfg = machines::read(&path).unwrap();
        cfg.banks = [8, 8, 0, 0];
        machines::save(&root.0, "indy", &mut cfg).unwrap();
        prefs.save().unwrap();
        let reloaded = GuiSettings::load_in(root.0.clone(), &root.0);
        assert_eq!(reloaded.machines["indy"].banks, [8, 8, 0, 0]);
        assert_eq!(std::fs::read_to_string(root.0.join("gui.json.pre-toml.bak")).unwrap(), original_text);
    }

    #[test]
    fn failed_migration_leaves_original_json_intact() {
        let root = TempRoot::new("failed");
        // A non-directory machines entry forces a real filesystem failure.
        std::fs::write(root.0.join("machines"), b"occupied").unwrap();
        let text = serde_json::json!({"machines": {"indy": MachineConfig::default()}, "active_machine": "indy"}).to_string();
        std::fs::write(root.0.join("gui.json"), &text).unwrap();
        let mut prefs = GuiSettings::load_in(root.0.clone(), &root.0);
        assert!(prefs.migration_pending);
        assert!(prefs.save().is_err());
        assert_eq!(std::fs::read_to_string(root.0.join("gui.json")).unwrap(), text);
        assert_eq!(std::fs::read_to_string(root.0.join("gui.json.pre-toml.bak")).unwrap(), text);
    }

    #[test]
    fn corrupt_preferences_are_reported_and_never_overwritten() {
        let root = TempRoot::new("corrupt");
        std::fs::write(root.0.join("gui.json"), "{ invalid json").unwrap();
        let mut prefs = GuiSettings::load_in(root.0.clone(), &root.0);
        assert!(!prefs.errors.is_empty());
        assert!(prefs.save().is_err());
        assert_eq!(std::fs::read_to_string(root.0.join("gui.json")).unwrap(), "{ invalid json");
    }

    #[test]
    fn legacy_toml_pointer_migrates_once_and_preserves_source_file() {
        let root = TempRoot::new("legacy-toml");
        let source = root.0.join("old.toml");
        std::fs::write(&source, "banks = [8, 8, 0, 0]").unwrap();
        std::fs::write(root.0.join("gui.json"), serde_json::json!({"last_config": source}).to_string()).unwrap();
        let prefs = GuiSettings::load_in(root.0.clone(), &root.0);
        assert!(prefs.errors.is_empty(), "{:?}", prefs.errors);
        assert_eq!(prefs.active_machine.as_deref(), Some("old"));
        assert_eq!(prefs.machines["old"].banks, [8, 8, 0, 0]);
        assert!(prefs.last_config.is_none());
        assert_eq!(std::fs::read_to_string(source).unwrap(), "banks = [8, 8, 0, 0]");
        assert_eq!(GuiSettings::load_in(root.0.clone(), &root.0).machines.len(), 1);
    }

    #[test]
    fn legacy_names_that_are_not_filenames_migrate_deterministically() {
        let root = TempRoot::new("old-name");
        let old_name = "IRIX / 6.5";
        std::fs::write(root.0.join("gui.json"), serde_json::json!({
            "machines": {old_name: MachineConfig::default()}, "active_machine": old_name
        }).to_string()).unwrap();
        let prefs = GuiSettings::load_in(root.0.clone(), &root.0);
        assert!(prefs.errors.is_empty(), "{:?}", prefs.errors);
        let name = machines::legacy_name(old_name);
        machines::validate_name(&name).unwrap();
        assert_eq!(prefs.active_machine.as_deref(), Some(name.as_str()));
        assert!(prefs.machine_path(&name).unwrap().exists());
        assert_eq!(GuiSettings::load_in(root.0.clone(), &root.0).machines.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn storage_roots_with_symlinked_parents_use_the_actual_working_directory_base() {
        let temp = TempRoot::new("symlink-root");
        let actual = temp.0.join("actual");
        std::fs::create_dir_all(&actual).unwrap();
        std::os::unix::fs::symlink(&actual, temp.0.join("alias")).unwrap();
        let mut prefs = GuiSettings::load_in(temp.0.join("alias/iris"), &temp.0);
        assert_eq!(prefs.root().unwrap(), actual.join("iris"));
        let disk = temp.0.join("external.raw");
        std::fs::write(&disk, b"external disk").unwrap();
        let mut cfg = MachineConfig::default();
        cfg.scsi.get_mut(&1).unwrap().path = disk.to_string_lossy().into_owned();
        prefs.save_machine("indy", &mut cfg).unwrap();
        let dir = prefs.machine_dir("indy").unwrap();
        assert_eq!(std::fs::read(dir.join(&cfg.scsi[&1].path)).unwrap(), b"external disk");
    }

    #[test]
    fn folder_discovery_does_not_need_gui_json_and_names_reserve_retained_data() {
        let root = TempRoot::new("no-json");
        let mut cfg = MachineConfig::default();
        machines::save(&root.0, "indy", &mut cfg).unwrap();
        let prefs = GuiSettings::load_in(root.0.clone(), &root.0);
        assert!(prefs.machines.contains_key("indy"));
        assert_eq!(prefs.ui_scale, UI_SCALE_DEFAULT);
        assert_eq!(prefs.vm_scale, VM_SCALE_DEFAULT);
        assert_eq!(prefs.unique_name("indy"), "indy-2");
        std::fs::create_dir_all(root.0.join("machines/indy-2/disks")).unwrap();
        assert_eq!(prefs.unique_name("indy"), "indy-3");
    }
}
