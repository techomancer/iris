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
    /// Enable copy-on-write overlay. Base image is never modified; writes go to
    /// `{path}.overlay`. Delete the overlay file to reset to clean state.
    #[serde(default)]
    pub overlay: bool,
}

/// Protocol for port forwarding.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ForwardProto {
    Tcp,
    Udp,
}

/// Bind scope for a port forward listener.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ForwardBind {
    /// Listen only on 127.0.0.1 (loopback only).
    Localhost,
    /// Listen on 0.0.0.0 (all interfaces).
    Any,
}

impl Default for ForwardBind {
    fn default() -> Self { ForwardBind::Localhost }
}

/// One port-forward rule: host_port → guest_port on a given protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortForwardConfig {
    /// Protocol: "tcp" or "udp".
    pub proto: ForwardProto,
    /// Host-side port to listen on.
    pub host_port: u16,
    /// Guest-side port to forward to (inside the VM).
    pub guest_port: u16,
    /// Bind scope: "localhost" (loopback only) or "any" (all interfaces).
    #[serde(default)]
    pub bind: ForwardBind,
}

/// NFS share configuration (requires unfsd on the host).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NfsConfig {
    /// Directory to export over NFS.
    pub shared_dir: String,
    /// Path to the unfsd binary [default: "unfsd"].
    #[serde(default = "default_unfsd")]
    pub unfsd: String,
    /// Host-side port unfsd listens on for NFS (high port, NAT'd to 2049 inside the VM).
    #[serde(default = "default_nfs_host_port")]
    pub nfs_host_port: u16,
    /// Host-side port unfsd listens on for mountd (high port, NAT'd to 1234 inside the VM).
    #[serde(default = "default_mountd_host_port")]
    pub mountd_host_port: u16,
}

fn default_unfsd()          -> String { "unfsd".to_string() }
fn default_nfs_host_port()  -> u16    { 12049 }
fn default_mountd_host_port() -> u16  { 11234 }

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

    /// NFS share configuration. If present, unfsd is started and NFS is available inside the VM.
    #[serde(default)]
    pub nfs: Option<NfsConfig>,

    /// Port forwarding rules (host port → guest port).
    #[serde(default)]
    pub port_forward: Vec<PortForwardConfig>,

    /// Run without graphics (no window, no REX3). Use no_audio to also disable HAL2.
    /// Useful for headless/server/CI environments.
    #[serde(default)]
    pub headless: bool,

    /// Disable audio emulation (no HAL2). Independent of headless/graphics.
    #[serde(default)]
    pub no_audio: bool,
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
        overlay: false,
    });
    map.insert(4, ScsiDeviceConfig {
        path: "cdrom4.iso".to_string(),
        discs: vec![],
        cdrom: true,
        overlay: false,
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
            nfs: None,
            port_forward: vec![],
            headless: false,
            no_audio: false,
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

    /// Run headless: no window, no REX3 graphics (audio unaffected; use --noaudio to disable)
    #[arg(long, default_value_t = false)]
    pub headless: bool,

    /// Disable audio emulation (no HAL2); graphics still works
    #[arg(long = "noaudio", default_value_t = false)]
    pub no_audio: bool,

    /// Enable NFS share: path to the directory to export (enables NFS)
    #[arg(long = "nfs-dir", value_name = "DIR")]
    pub nfs_dir: Option<String>,

    /// Path to unfsd binary [default: unfsd]
    #[arg(long = "unfsd", value_name = "PATH")]
    pub unfsd: Option<String>,

    /// Host port for unfsd NFS listener [default: 12049]
    #[arg(long = "nfs-port", value_name = "PORT")]
    pub nfs_host_port: Option<u16>,

    /// Host port for unfsd mountd listener [default: 11234]
    #[arg(long = "mountd-port", value_name = "PORT")]
    pub mountd_host_port: Option<u16>,
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
                overlay: false,
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
        if self.headless  { cfg.headless  = true; }
        if self.no_audio  { cfg.no_audio  = true; }

        // NFS: --nfs-dir enables NFS; other flags refine an existing [nfs] section or the defaults.
        if let Some(dir) = &self.nfs_dir {
            let base = cfg.nfs.get_or_insert_with(|| NfsConfig {
                shared_dir:       dir.clone(),
                unfsd:            default_unfsd(),
                nfs_host_port:    default_nfs_host_port(),
                mountd_host_port: default_mountd_host_port(),
            });
            base.shared_dir = dir.clone();
        }
        if let Some(ref mut nfs) = cfg.nfs {
            if let Some(p) = &self.unfsd           { nfs.unfsd            = p.clone(); }
            if let Some(p) = self.nfs_host_port    { nfs.nfs_host_port    = p; }
            if let Some(p) = self.mountd_host_port { nfs.mountd_host_port = p; }
        }

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
