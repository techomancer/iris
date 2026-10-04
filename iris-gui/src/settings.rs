use iris::config::MachineConfig;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// GUI-only persisted state. Lives at `~/.config/iris/gui.json`.
///
/// This is the **system of record** for machines: each named machine is a
/// `MachineConfig` stored here. `iris.toml` is treated as import/export
/// only, for compatibility with the standalone `iris` CLI.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GuiSettings {
    /// egui UI scale (1.0 = default).
    #[serde(default = "default_ui_scale")]
    pub ui_scale: f32,
    /// Emulated-display (VM screen) magnification: 1.0 = native (1 emulated
    /// pixel : 1 logical point). Driven by the View-menu slider (0.5×–3× in 0.5
    /// steps), **independent of `ui_scale`** — scaling the controls doesn't
    /// resize the picture, and vice-versa.
    #[serde(default = "default_vm_scale")]
    pub vm_scale: f32,

    /// All saved machines keyed by user-visible name. BTreeMap so menus
    /// list them in stable alphabetical order.
    #[serde(default)]
    pub machines: BTreeMap<String, MachineConfig>,
    /// Currently-selected machine (key into `machines`). None = no
    /// machine loaded yet (first run).
    #[serde(default)]
    pub active_machine: Option<String>,

    // --- Legacy iris.toml workflow (still supported for users who had it). ---
    /// Most-recently-imported iris.toml files (newest first, max ~10).
    #[serde(default)]
    pub recent_configs: Vec<PathBuf>,
    /// Last-imported TOML path; one-shot migration source on first launch
    /// of the new machine-store world.
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
    pub fn config_path() -> Option<PathBuf> {
        Self::data_dir().map(|d| d.join("gui.json"))
    }

    /// Stable per-user directory for GUI state (gui.json, nvram.bin, …). The OS
    /// maps this into the sandbox container automatically on the App Store
    /// build, so the *same* code resolves the right place for `cargo run` and
    /// the bundled app alike.
    pub fn data_dir() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("iris"))
    }

    /// Default absolute NVRAM path: `<data_dir>/nvram.bin`. Absolute on purpose
    /// — a relative `nvram.bin` resolves against the process's working
    /// directory, which differs between `cargo run` (repo root) and a bundled
    /// `.app`, silently loading different (often blank, MAC-less) NVRAMs. Anchor
    /// it once and every launch shares one NVRAM.
    pub fn default_nvram_path() -> String {
        Self::data_dir()
            .map(|d| d.join("nvram.bin").to_string_lossy().into_owned())
            .unwrap_or_else(|| "nvram.bin".to_string())
    }

    /// Default absolute NVRAM EEPROM path: `<data_dir>/nveeprom.bin`. Same
    /// reasoning as [`default_nvram_path`] — the core default is a bare
    /// `"nveeprom.bin"` (`src/config.rs::default_nveeprom`), which resolves
    /// against the process's working directory and silently finds a
    /// different (usually blank) file between `cargo run` and a bundled
    /// `.app`. This is the motherboard EEPROM Indigo2/IP28 actually read
    /// `eaddr` from (`Eeprom93c56`, see `rules/irix/networking.md`).
    /// `ensure_nveeprom_exists` creates an erased chip; the IP28 core seeds
    /// PROM defaults before its first boot and saves them on Stop.
    pub fn default_nveeprom_path() -> String {
        Self::data_dir()
            .map(|d| d.join("nveeprom.bin").to_string_lossy().into_owned())
            .unwrap_or_else(|| "nveeprom.bin".to_string())
    }

    /// Managed directory for newly-created disk images: `<data_dir>/disks`.
    /// Absolute and writable in every launch context — the OS maps it into the
    /// sandbox container on the App Store build, so creating a disk here needs
    /// no permission prompt. Users can still pick another location.
    pub fn disks_dir() -> Option<PathBuf> {
        Self::data_dir().map(|d| d.join("disks"))
    }

    /// Default absolute path for a new SCSI disk image: `<disks_dir>/scsiN.raw`.
    pub fn default_disk_path(scsi_id: u8) -> String {
        Self::disks_dir()
            .map(|d| d.join(format!("scsi{scsi_id}.raw")).to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("scsi{scsi_id}.raw"))
    }

    /// Anchor a machine's NVRAM path to [`data_dir`] if it's relative (the
    /// legacy default was a bare `"nvram.bin"`). Best-effort: if the anchored
    /// file doesn't exist yet but the old cwd-relative one does, copy it over so
    /// the PROM env (boot settings, any MAC) carries forward instead of starting
    /// blank. Idempotent — absolute paths are left untouched.
    pub fn migrate_nvram_path(nvram: &mut String) {
        if !nvram.is_empty() && Path::new(&nvram).is_absolute() {
            return;
        }
        let Some(dir) = Self::data_dir() else { return; };
        let _ = std::fs::create_dir_all(&dir);
        let leaf = Path::new(nvram.as_str())
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("nvram.bin");
        let dst = dir.join(leaf);
        let src = PathBuf::from(nvram.as_str()); // relative to cwd
        if !dst.exists() && !nvram.is_empty() && src.exists() {
            let _ = std::fs::copy(&src, &dst);
        }
        *nvram = dst.to_string_lossy().into_owned();
    }

    /// Same anchoring as [`migrate_nvram_path`], for the NVRAM EEPROM path.
    /// The legacy default was a bare `"nveeprom.bin"`, which could be
    /// anywhere depending on how iris-gui was launched (no `set_current_dir`
    /// call of its own) — this was never anchored at all before, so most
    /// existing machines will have nothing to find at the old relative path
    /// and just start a fresh EEPROM at the new stable location, same as a
    /// brand new machine would.
    pub fn migrate_nveeprom_path(nveeprom: &mut String) {
        if !nveeprom.is_empty() && Path::new(&nveeprom).is_absolute() {
            return;
        }
        let Some(dir) = Self::data_dir() else { return; };
        let _ = std::fs::create_dir_all(&dir);
        let leaf = Path::new(nveeprom.as_str())
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("nveeprom.bin");
        let dst = dir.join(leaf);
        let src = PathBuf::from(nveeprom.as_str()); // relative to cwd
        if !dst.exists() && !nveeprom.is_empty() && src.exists() {
            let _ = std::fs::copy(&src, &dst);
        }
        *nveeprom = dst.to_string_lossy().into_owned();
    }

    pub fn load() -> Self {
        // Load from disk when present, else start from defaults — but ALWAYS
        // fall through to the sanitizer below. A missing or unreadable file used
        // to early-return `Self::default()`, which leaves `vm_scale`/`ui_scale`
        // at the struct's zero `Default` (0.0, not the serde field defaults).
        // A 0.0 vm_scale then panics the window-fit math (`clamp` min > max), so
        // a first-ever run with no gui.json crashed instead of using defaults.
        let mut s: Self = Self::config_path()
            .and_then(|path| std::fs::read_to_string(&path).ok())
            .and_then(|text| serde_json::from_str::<Self>(&text).ok())
            .unwrap_or_default();
        // Sanitize a stale/out-of-range persisted scale. A value below the
        // minimum is junk left by an older build whose keyboard zoom floored
        // at 0.5 (the UI can no longer produce sub-minimum values), so reset
        // it to the default rather than honoring it — likewise for a
        // non-finite value from a corrupt file. Only the high end is clamped.
        s.ui_scale = if !s.ui_scale.is_finite() || s.ui_scale < UI_SCALE_MIN {
            UI_SCALE_DEFAULT
        } else {
            s.ui_scale.min(UI_SCALE_MAX)
        };
        s.vm_scale = if !s.vm_scale.is_finite() || s.vm_scale < VM_SCALE_MIN {
            VM_SCALE_DEFAULT
        } else {
            s.vm_scale.min(VM_SCALE_MAX)
        };
        // Anchor every machine's NVRAM to the stable data dir so all launch
        // methods share one file (the persisted path becomes absolute on the
        // next save).
        for m in s.machines.values_mut() {
            Self::migrate_nvram_path(&mut m.nvram);
            Self::migrate_nveeprom_path(&mut m.nveeprom);
        }
        s
    }

    pub fn save(&mut self) -> Result<(), String> {
        // Refresh macOS security-scoped bookmarks for every machine's reachable
        // files so they reopen under the App Sandbox next launch. No-op off the
        // Mac App Store build.
        let paths: Vec<String> = self
            .machines
            .values()
            .flat_map(crate::macos_sandbox::config_paths)
            .collect();
        // Harvest both per-file bookmarks and the user-granted disk folders (a
        // directory bookmark is recursive — see `disk_folders`).
        crate::macos_sandbox::harvest(
            paths.iter().map(String::as_str).chain(self.disk_folders.iter().map(String::as_str)),
            &mut self.bookmarks,
        );

        let path = Self::config_path().ok_or("no config dir")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&path, text).map_err(|e| e.to_string())
    }

    pub fn push_recent(&mut self, path: PathBuf) {
        self.recent_configs.retain(|p| p != &path);
        self.recent_configs.insert(0, path.clone());
        self.recent_configs.truncate(10);
        self.last_config = Some(path);
    }

    /// Pick a free name like "indy", "indy-2", "indy-3", …
    pub fn unique_name(&self, base: &str) -> String {
        if !self.machines.contains_key(base) { return base.to_string(); }
        for n in 2..1000 {
            let candidate = format!("{base}-{n}");
            if !self.machines.contains_key(&candidate) { return candidate; }
        }
        format!("{base}-{}", uuid_like())
    }
}

fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0).to_string()
}
