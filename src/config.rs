use clap::Parser;
use serde::{Deserialize, Serialize};

/// Valid memory bank sizes in MB.
pub const VALID_BANK_SIZES: &[u32] = &[0, 8, 16, 32, 64, 128];

/// Configuration for a single SCSI device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScsiDeviceConfig {
    /// Path to the disk image or ISO file (primary/current disc).
    pub path: String,
    /// Additional ISO images for CD-ROM changers (ignored for HDD).
    #[serde(default)]
    pub discs: Vec<String>,
    /// true = CD-ROM, false = hard disk.
    pub cdrom: bool,
}

/// Top-level machine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineConfig {
    /// Path to the PROM ROM image.
    #[serde(default = "default_prom")]
    pub prom: String,

    /// RAM bank sizes in MB. Valid values: 0 (absent), 8, 16, 32, 64, 128.
    #[serde(default = "default_banks")]
    pub banks: [u32; 4],

    /// SCSI devices keyed by ID 1–7. Missing IDs are not attached.
    #[serde(default = "default_scsi")]
    pub scsi: std::collections::HashMap<u8, ScsiDeviceConfig>,

    /// Window scale factor (1 = native, 2 = 2× for HiDPI/4K). CLI --2x overrides this.
    #[serde(default = "default_scale")]
    pub scale: u32,
}

fn default_prom() -> String {
    "prom.bin".to_string()
}

fn default_banks() -> [u32; 4] {
    [128, 128, 0, 0]
}

fn default_scale() -> u32 { 1 }

fn default_scsi() -> std::collections::HashMap<u8, ScsiDeviceConfig> {
    let mut map = std::collections::HashMap::new();
    map.insert(1, ScsiDeviceConfig {
        path: "scsi1.raw".to_string(),
        discs: vec![],
        cdrom: false,
    });
    map.insert(4, ScsiDeviceConfig {
        path: "cdrom4.iso".to_string(),
        discs: vec![],
        cdrom: true,
    });
    map
}

impl Default for MachineConfig {
    fn default() -> Self {
        Self {
            prom: default_prom(),
            banks: default_banks(),
            scsi: default_scsi(),
            scale: default_scale(),
        }
    }
}

impl MachineConfig {
    /// Load from `iris.toml` if it exists, otherwise return defaults.
    pub fn load_toml(path: &str) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self::default();
        };
        match toml::from_str::<Self>(&text) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("Warning: failed to parse {}: {}", path, e);
                Self::default()
            }
        }
    }

    /// Validate bank sizes, returns a description of any errors.
    pub fn validate(&self) -> Result<(), String> {
        if self.scale != 1 && self.scale != 2 {
            return Err(format!("scale {} is invalid (valid: 1, 2)", self.scale));
        }
        for (i, &sz) in self.banks.iter().enumerate() {
            if !VALID_BANK_SIZES.contains(&sz) {
                return Err(format!(
                    "bank{} size {} MB is invalid (valid: {:?})",
                    i, sz, VALID_BANK_SIZES
                ));
            }
        }
        for (id, dev) in &self.scsi {
            if *id == 0 || *id > 7 {
                return Err(format!("SCSI ID {} is out of range (1–7)", id));
            }
            if dev.cdrom && dev.path.is_empty() && dev.discs.is_empty() {
                return Err(format!("SCSI ID {} is a CD-ROM but has no disc", id));
            }
        }
        Ok(())
    }

    /// Return the active disc path for a CD-ROM device (first of `discs` list,
    /// falling back to `path`).
    pub fn active_disc(dev: &ScsiDeviceConfig) -> &str {
        dev.discs.first().map(|s| s.as_str()).unwrap_or(&dev.path)
    }
}

// ---------------------------------------------------------------------------
// CLI — all fields optional; presence overrides the TOML/default value.
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "iris", about = "SGI Indy (MIPS R4400) emulator")]
pub struct Cli {
    /// Path to iris.toml config file [default: iris.toml]
    #[arg(long, default_value = "iris.toml")]
    pub config: String,

    /// Path to PROM image
    #[arg(long)]
    pub prom: Option<String>,

    /// RAM bank 0 size in MB (0/8/16/32/64/128)
    #[arg(long)]
    pub bank0: Option<u32>,

    /// RAM bank 1 size in MB (0/8/16/32/64/128)
    #[arg(long)]
    pub bank1: Option<u32>,

    /// RAM bank 2 size in MB (0/8/16/32/64/128)
    #[arg(long)]
    pub bank2: Option<u32>,

    /// RAM bank 3 size in MB (0/8/16/32/64/128)
    #[arg(long)]
    pub bank3: Option<u32>,

    /// SCSI ID 1 image path (HDD)
    #[arg(long)]
    pub scsi1: Option<String>,

    /// SCSI ID 2 image path (HDD)
    #[arg(long)]
    pub scsi2: Option<String>,

    /// SCSI ID 3 image path (HDD)
    #[arg(long)]
    pub scsi3: Option<String>,

    /// SCSI ID 4 image path (CD-ROM, primary disc)
    #[arg(long)]
    pub cdrom4: Option<String>,

    /// SCSI ID 5 image path (CD-ROM, primary disc)
    #[arg(long)]
    pub cdrom5: Option<String>,

    /// SCSI ID 6 image path (CD-ROM, primary disc)
    #[arg(long)]
    pub cdrom6: Option<String>,

    /// SCSI ID 7 image path (HDD)
    #[arg(long)]
    pub scsi7: Option<String>,

    /// Additional ISO images for CD-ROM ID 4 (can be specified multiple times)
    #[arg(long = "cdrom4-extra", value_name = "ISO")]
    pub cdrom4_extra: Vec<String>,

    /// Additional ISO images for CD-ROM ID 5 (can be specified multiple times)
    #[arg(long = "cdrom5-extra", value_name = "ISO")]
    pub cdrom5_extra: Vec<String>,

    /// Additional ISO images for CD-ROM ID 6 (can be specified multiple times)
    #[arg(long = "cdrom6-extra", value_name = "ISO")]
    pub cdrom6_extra: Vec<String>,

    /// 2× window scaling for HiDPI/4K monitors
    #[arg(long = "2x", default_value_t = false)]
    pub scale2x: bool,
}

impl Cli {
    /// Merge CLI overrides into a base `MachineConfig`.
    pub fn apply(&self, mut cfg: MachineConfig) -> MachineConfig {
        if let Some(p) = &self.prom    { cfg.prom = p.clone(); }
        if let Some(v) = self.bank0    { cfg.banks[0] = v; }
        if let Some(v) = self.bank1    { cfg.banks[1] = v; }
        if let Some(v) = self.bank2    { cfg.banks[2] = v; }
        if let Some(v) = self.bank3    { cfg.banks[3] = v; }

        // Helper: insert or update a SCSI device entry.
        let apply_scsi = |map: &mut std::collections::HashMap<u8, ScsiDeviceConfig>,
                          id: u8, path: String, cdrom: bool, extra: Vec<String>| {
            let entry = map.entry(id).or_insert_with(|| ScsiDeviceConfig {
                path: String::new(),
                discs: vec![],
                cdrom,
            });
            entry.path = path;
            entry.cdrom = cdrom;
            if !extra.is_empty() {
                entry.discs = extra;
            }
        };

        if let Some(p) = self.scsi1.clone()  { apply_scsi(&mut cfg.scsi, 1, p, false, vec![]); }
        if let Some(p) = self.scsi2.clone()  { apply_scsi(&mut cfg.scsi, 2, p, false, vec![]); }
        if let Some(p) = self.scsi3.clone()  { apply_scsi(&mut cfg.scsi, 3, p, false, vec![]); }
        if let Some(p) = self.cdrom4.clone() { apply_scsi(&mut cfg.scsi, 4, p, true, self.cdrom4_extra.clone()); }
        if let Some(p) = self.cdrom5.clone() { apply_scsi(&mut cfg.scsi, 5, p, true, self.cdrom5_extra.clone()); }
        if let Some(p) = self.cdrom6.clone() { apply_scsi(&mut cfg.scsi, 6, p, true, self.cdrom6_extra.clone()); }
        if let Some(p) = self.scsi7.clone()  { apply_scsi(&mut cfg.scsi, 7, p, false, vec![]); }

        if self.scale2x { cfg.scale = 2; }

        cfg
    }
}

/// Parse CLI, load TOML, merge, and validate. Exits on error.
/// Returns (machine_config, window_scale) where window_scale is 1 or 2.
pub fn load_config() -> (MachineConfig, u32) {
    let cli = Cli::parse();
    let toml_cfg = MachineConfig::load_toml(&cli.config);
    let cfg = cli.apply(toml_cfg);
    let scale = cfg.scale;
    if let Err(e) = cfg.validate() {
        eprintln!("Configuration error: {}", e);
        std::process::exit(1);
    }
    (cfg, scale)
}
