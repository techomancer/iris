use clap::Parser;
use serde::{Deserialize, Serialize};
use std::net::Ipv4Addr;

/// Valid memory bank sizes in MB.
pub const VALID_BANK_SIZES: &[u32] = &[0, 8, 16, 32, 64, 128, 256];

/// What sits at a SCSI id. `cdrom = true` remains the historical spelling for
/// `kind = "cdrom"`; either works and they mean the same thing.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ScsiKind {
    /// Hard disk (raw image, CHD, or COW overlay).
    #[default]
    Disk,
    /// CD-ROM drive (may start with an empty tray).
    Cdrom,
    /// DaynaPort SCSI/Link — a SCSI-attached Ethernet adapter. Has no disk
    /// image at all. Requires a build with `--features daynaport`.
    Daynaport,
}

impl ScsiKind {
    fn is_default(&self) -> bool { *self == ScsiKind::Disk }

    pub fn label(self) -> &'static str {
        match self {
            Self::Disk => "Hard disk",
            Self::Cdrom => "CD-ROM",
            Self::Daynaport => "DaynaPort SCSI/Link",
        }
    }
}

/// Configuration for a single SCSI device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScsiDeviceConfig {
    /// Path to the disk image or ISO file (primary/current disc).
    /// For CD-ROMs this may be omitted (defaults to empty string) to start the
    /// drive with an empty tray; media can be loaded at runtime.
    #[serde(default)]
    pub path: String,
    /// Additional ISO images for CD-ROM changers (ignored for HDD).
    #[serde(default)]
    pub discs: Vec<String>,
    /// true = CD-ROM, false = hard disk. Kept for compatibility: it is the
    /// original spelling of `kind = "cdrom"` and every existing config uses it.
    /// Use `kind` for anything that is not a disk or a CD-ROM.
    #[serde(default)]
    pub cdrom: bool,
    /// Target type. Defaults to `disk`; `cdrom = true` still selects a CD-ROM
    /// on its own. Read it through [`ScsiDeviceConfig::kind`] rather than
    /// directly, so the two spellings stay reconciled.
    #[serde(default, rename = "kind", skip_serializing_if = "ScsiKind::is_default")]
    pub kind_field: ScsiKind,
    /// DaynaPort only: explicit MAC address, e.g. `"00:80:19:12:34:56"`.
    /// Default is derived from the SCSI id so two targets never collide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mac: Option<String>,
    /// DaynaPort only: NAT subnet in CIDR notation for *this* target's gateway,
    /// e.g. `"192.168.10.0/24"`. Each DaynaPort runs its own NAT engine, so
    /// this must differ from the machine-wide `nat_subnet` used by `ec0`.
    /// Gateway gets host .1, the guest gets host .2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subnet: Option<String>,
    /// Enable copy-on-write overlay. Base image is never modified; writes go to
    /// `{path}.overlay`. Delete the overlay file to reset to clean state.
    #[serde(default)]
    pub overlay: bool,
    /// Scratch volume: a host-controlled raw block device used for file
    /// injection/extraction without networking. iris auto-creates a zero-filled
    /// file at `path` if it doesn't exist (size = `size_mb`, default 64). The
    /// CI socket exposes scratch-write/read/clear/info to mutate it from the
    /// host side. No filesystem is imposed: callers can write a tar stream and
    /// the guest reads it with `dd if=/dev/rdsk/dks0dNvh | tar xf -`.
    /// Implies !cdrom && !overlay (the volume must be host-writable directly).
    #[serde(default)]
    pub scratch: bool,
    /// Size in MB for an auto-created scratch volume. Ignored when the file
    /// already exists or `scratch=false`.
    #[serde(default)]
    pub size_mb: Option<u32>,
    /// Which HPC3 SCSI chip this target lives on: 0 (default) or 1. Indigo2
    /// (IP22 fullhouse) has two independent WD33C93A controllers; Indy
    /// (IP24 guinness) only has controller 0, so `controller = 1` is
    /// rejected outside the Indigo2 profile — see `MachineConfig::validate`.
    #[serde(default)]
    pub controller: u8,
}

impl Default for ScsiDeviceConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            discs: vec![],
            cdrom: false,
            kind_field: ScsiKind::Disk,
            mac: None,
            subnet: None,
            overlay: false,
            scratch: false,
            size_mb: None,
            controller: 0,
        }
    }
}

impl ScsiDeviceConfig {
    /// The target type, reconciling the `kind` key with the older `cdrom` bool.
    /// An explicit `kind` wins; `cdrom = true` alone still means CD-ROM.
    pub fn kind(&self) -> ScsiKind {
        match self.kind_field {
            ScsiKind::Disk if self.cdrom => ScsiKind::Cdrom,
            k => k,
        }
    }

    pub fn is_cdrom(&self) -> bool { self.kind() == ScsiKind::Cdrom }
    pub fn is_daynaport(&self) -> bool { self.kind() == ScsiKind::Daynaport }

    /// A DaynaPort target for this SCSI id, with defaults filled in.
    /// Errors describe a bad `mac` / `subnet`; `validate()` catches those first.
    pub fn daynaport_params(&self, id: u8) -> Result<DaynaportParams, String> {
        let mac = match &self.mac {
            Some(s) => parse_mac(s)?,
            None => [0x00, 0x80, 0x19, 0x44, 0x50, id],
        };
        let subnet = match &self.subnet {
            Some(cidr) => {
                let (gateway_ip, client_ip, netmask) = parse_nat_subnet(cidr)?;
                NatSubnet { gateway_ip, client_ip, netmask }
            }
            None => NatSubnet {
                gateway_ip: Ipv4Addr::new(192, 168, 10, 1),
                client_ip:  Ipv4Addr::new(192, 168, 10, 2),
                netmask:    Ipv4Addr::new(255, 255, 255, 0),
            },
        };
        Ok(DaynaportParams { mac, subnet })
    }
}

/// Resolved DaynaPort settings for one SCSI target.
#[derive(Debug, Clone, Copy)]
pub struct DaynaportParams {
    pub mac: [u8; 6],
    pub subnet: NatSubnet,
}

/// Default Ethernet station address for `ec0` when `[network] mac` is unset.
/// SGI's registered OUI (08:00:69) plus an arbitrary host part; matches the
/// address used throughout `rules/irix/networking.md` and CI test fixtures.
pub const DEFAULT_MAC: [u8; 6] = [0x08, 0x00, 0x69, 0x12, 0x34, 0x56];

/// Parse `"00:80:19:12:34:56"` (or `-` separated) into six octets.
pub fn parse_mac(s: &str) -> Result<[u8; 6], String> {
    let parts: Vec<&str> = s.split(|c| c == ':' || c == '-').collect();
    if parts.len() != 6 {
        return Err(format!("\"{}\" is not a MAC address (expected six octets)", s));
    }
    let mut mac = [0u8; 6];
    for (i, p) in parts.iter().enumerate() {
        mac[i] = u8::from_str_radix(p, 16)
            .map_err(|_| format!("\"{}\" is not a MAC address (bad octet \"{}\")", s, p))?;
    }
    if mac[0] & 0x01 != 0 {
        return Err(format!("{} is a multicast address; a station MAC must have bit 0 of the \
                            first octet clear", s));
    }
    Ok(mac)
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

/// NFS share configuration. NFS is served in-process by the NAT
/// (`src/nfsudp.rs`) — no external `unfsd`, no host sockets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NfsConfig {
    /// Directory to export over NFS.
    pub shared_dir: String,
    /// NFS protocol version to serve (Auto answers whatever the guest mounts).
    #[serde(default)]
    pub version: crate::nfsudp::NfsVersion,
}

/// Pre-parsed NAT subnet derived from a CIDR string.
#[derive(Debug, Clone, Copy)]
pub struct NatSubnet {
    pub gateway_ip: Ipv4Addr,
    pub client_ip:  Ipv4Addr,
    pub netmask:    Ipv4Addr,
}

impl Default for NatSubnet {
    fn default() -> Self {
        Self {
            gateway_ip: Ipv4Addr::new(192, 168, 0, 1),
            client_ip:  Ipv4Addr::new(192, 168, 0, 2),
            netmask:    Ipv4Addr::new(255, 255, 255, 0),
        }
    }
}

/// Selects which networking backend the SEEQ Ethernet controller is wired to.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
#[value(rename_all = "lowercase")]
pub enum NetMode {
    /// Built-in software NAT gateway (ARP/DHCP/DNS/ICMP/TCP/UDP + port forwarding).
    /// Works without any host privileges or extra libraries. This is the default.
    Nat,
    /// Bridge raw Ethernet frames onto a real host interface via libpcap
    /// (Linux/macOS) or a WinPcap-compatible driver (WinPcap or Npcap) on Windows.
    /// Requires building with `--features pcap` and elevated privileges at runtime.
    /// The guest appears as a real L2 host on the physical LAN; NAT services
    /// (DHCP/DNS/NFS/port-forward) are NOT provided — use the real network's.
    Pcap,
}

impl Default for NetMode {
    fn default() -> Self { NetMode::Nat }
}

/// Networking parameters extracted from `MachineConfig` for the NAT engine and HPC3.
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    pub nfs:          Option<NfsConfig>,
    pub port_forward: Vec<PortForwardConfig>,
    /// Parsed subnet; None means use the built-in default (192.168.0.0/24).
    pub nat_subnet:   Option<NatSubnet>,
    /// Backend selection: NAT (default) or PCAP bridged.
    pub mode:         NetMode,
    /// Host interface name to bridge onto when `mode == Pcap`. None = auto-pick
    /// the first non-loopback interface that libpcap reports as up/running.
    pub pcap_interface: Option<String>,
    /// PCAP-only virtual IP for the in-process NFS server (so a bridged guest can
    /// mount it). None = NFS-in-PCAP not configured.
    pub nfs_pcap_ip: Option<std::net::Ipv4Addr>,
    /// Ethernet station address for `ec0`, backdoor-injected into NVRAM
    /// (Indy) / serial EEPROM (Indigo2) before boot. Defaults to
    /// [`DEFAULT_MAC`] when `[network] mac` is unset.
    pub mac: [u8; 6],
    /// Directory served read-only over TFTP at the gateway, for PROM network
    /// boot. None (the default) disables the TFTP server.
    pub tftp_dir: Option<std::path::PathBuf>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            nfs: None,
            port_forward: Vec::new(),
            nat_subnet: None,
            mode: NetMode::default(),
            pcap_interface: None,
            nfs_pcap_ip: None,
            tftp_dir: None,
            mac: DEFAULT_MAC,
        }
    }
}

/// `[network]` section: backend selection and PCAP options.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct NetworkSection {
    /// Backend: "nat" (default) or "pcap".
    #[serde(default)]
    pub mode: NetMode,
    /// Host interface to bridge onto in PCAP mode (e.g. "eth0", "en0").
    /// Run `iris --list-net-interfaces` to enumerate candidates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pcap_interface: Option<String>,
    /// PCAP-only: the virtual LAN IP the in-process NFS server answers on, so a
    /// bridged guest (which is directly on your real LAN, with no NAT gateway to
    /// reach) can mount it. None = NFS-in-PCAP not configured. NAT mode ignores
    /// this and serves NFS at the gateway IP instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nfs_pcap_ip: Option<std::net::Ipv4Addr>,
    /// Ethernet station address for the built-in SEEQ controller (`ec0`), e.g.
    /// "08:00:69:12:34:56" (SGI's registered OUI). Defaults to
    /// `08:00:69:12:34:56` when unset. Real hardware has no way to leave this
    /// unset — every SGI ships with a MAC burned into NVRAM (Indy) or a serial
    /// EEPROM (Indigo2) — so iris backdoor-injects it into the emulated
    /// NVRAM/EEPROM before boot rather than requiring a guest-side `setenv`.
    /// On Indy this only patches NVRAM if the `eaddr` slot is still blank
    /// (00:00:00:00:00:00), so it never clobbers a value you've already set
    /// from the PROM monitor and saved with `rtc save`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mac: Option<String>,
    /// Directory served read-only over TFTP (UDP 69) at the gateway address, so
    /// the PROM can `boot -f bootp()<file>`. Unset disables TFTP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tftp_dir: Option<String>,
}

/// Where VINO's video-in capture should come from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VinoSource {
    /// Live host camera capture (requires building with `--features camera`).
    /// First run on macOS triggers the camera permission dialog.
    Camera,
    /// SMPTE-style colour bars + animated luma ramp.  No host capture needed.
    TestPattern,
    /// Solid black field.  Useful when you want IRIX video drivers to attach
    /// but don't want any host camera permission prompt or test pattern.
    Black,
    /// Video-In disabled: VINO stays memory-mapped (IRIX can still probe it)
    /// but no video source is installed and the DMA pump thread is never
    /// started.  Use this to skip Video-In entirely.
    Off,
}

impl Default for VinoSource {
    // Off by default: most users don't need IndyCam, and this avoids a host
    // camera permission prompt, the test-pattern source, and VINO's DMA pump
    // thread. Set a source explicitly (`[vino] source = "..."`) to enable it.
    fn default() -> Self { VinoSource::Off }
}

/// Broadcast video standard the source emits.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum VinoStandard {
    /// 525-line / 60-field interlaced (NTSC, 640×486 frame).
    Ntsc,
    /// 625-line / 50-field interlaced (PAL, 768×576 frame).
    Pal,
}

impl Default for VinoStandard {
    fn default() -> Self { VinoStandard::Ntsc }
}

/// N64 development board (Ultra64) configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Ultra64Config {
    /// Enable the N64 development board GIO device and POSIX shm IPC bridge.
    #[serde(default)]
    pub enabled: bool,
}

/// Emulated SGI machine profile (hardware layout scaffold).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MachineProfile {
    /// SGI Indy IP24 (Guinness) — single Newport GIO64. Default and fully supported.
    #[default]
    IndyIp24,
    /// SGI Indigo2 IP22 — fullhouse MC/IOC, Newport XL on GIO gfx slot.
    Indigo2Ip22,
    /// SGI Indigo2 IMPACT IP28 — an R10000 CPU module in the Indigo2 chassis.
    ///
    /// Shares the fullhouse MC/IOC/HPC3 with IP22 and differs in the decodes
    /// inside them: MEMCFG's base field is shifted by 24 rather than 22 (so
    /// its size granule is 16 MB, not 4), RAM lives at 0x20000000 with the
    /// low-memory alias following it there, and both the MC chip revision and
    /// the HPC3 board revision have to read high enough for the kernel to
    /// call the board an IP28.
    ///
    /// Graphics is IMPACT, which is a register stub — an IP28 kernel carries
    /// no Newport driver, so REX3 is not an alternative here.
    Indigo2Ip28,
}

impl MachineProfile {
    /// All selectable profiles, in display order. Single source of truth for the
    /// GUI dropdowns (Config tab + New Machine dialog) so they never drift.
    pub const ALL: [Self; 3] = [Self::IndyIp24, Self::Indigo2Ip22, Self::Indigo2Ip28];

    pub fn label(self) -> &'static str {
        match self {
            Self::IndyIp24 => "SGI Indy (IP24)",
            Self::Indigo2Ip22 => "SGI Indigo2 (IP22)",
            Self::Indigo2Ip28 => "SGI Indigo2 IMPACT (IP28)",
        }
    }

    /// MC/IOC/HPC3 Guinness vs Fullhouse layout. Indy IP24 is Guinness (`true`).
    pub fn guinness(self) -> bool {
        matches!(self, Self::IndyIp24)
    }

    /// The R10000 Indigo2. Selects the IP28 decodes inside the shared
    /// fullhouse devices — see the variant's own documentation for the list.
    pub fn ip28(self) -> bool {
        matches!(self, Self::Indigo2Ip28)
    }
}

/// Indy / Indigo2 graphics board in the GIO gfx slot.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GraphicsBoard {
    /// Newport (REX3) — fully emulated. Default.
    #[default]
    Newport,
    /// GR2 XZ (2 GE7) — Indy XZ or Indigo2 XZ (`src/dev/gr2`). In bring-up.
    Xz,
    /// GR2 Extreme (8 GE7) — Indigo2 only (`src/dev/gr2`). In bring-up.
    Extreme,
}

impl GraphicsBoard {
    pub const ALL: [Self; 3] = [Self::Newport, Self::Xz, Self::Extreme];
    pub fn label(self) -> &'static str {
        match self {
            Self::Newport => "Newport (XL)",
            Self::Xz => "GR2 XZ",
            Self::Extreme => "GR2 Extreme",
        }
    }
    /// Whether `validate()` accepts this board on `profile`.
    pub fn supports(self, profile: MachineProfile) -> bool {
        self != Self::Extreme || profile == MachineProfile::Indigo2Ip22
    }
}

/// IMPACT board occupying one GIO64 slot (Indigo2 preview scaffold).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImpactSlot {
    #[default]
    None,
    Solid,
    High,
    Max,
}

/// `[impact]` section — IMPACT/MGRAS slot population (Indigo2 IP22 preview).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ImpactSection {
    /// GIO gfx slot (`0x1F000000`). Solid IMPACT anchors here.
    #[serde(default)]
    pub gfx: ImpactSlot,
    /// GIO expansion slot 0 (`0x1F400000`). Second board for High / Max configs.
    #[serde(default)]
    pub exp0: ImpactSlot,
    /// GIO expansion slot 1 (`0x1F600000`). Third board for Maximum IMPACT.
    #[serde(default)]
    pub exp1: ImpactSlot,
}

impl ImpactSection {
    pub fn any_enabled(&self) -> bool {
        self.gfx != ImpactSlot::None
            || self.exp0 != ImpactSlot::None
            || self.exp1 != ImpactSlot::None
    }

    /// One IMPACT board, in the graphics slot; a second head is not modelled yet.
    pub fn validate(&self) -> Result<(), String> {
        if self.exp0 != ImpactSlot::None || self.exp1 != ImpactSlot::None {
            return Err("[impact] only the graphics slot (gfx) is supported so far".into());
        }
        Ok(())
    }
}

/// `[graphics]` section — Newport head count and display options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphicsSection {
    /// Graphics board: `newport` (default), or GR2 `xz` (Indy/Indigo2) / `extreme` (Indigo2).
    #[serde(default)]
    pub board: GraphicsBoard,
    /// Newport heads to emulate (1 or 2). Dual-head maps a second REX3 at GIO slot 1.
    #[serde(default = "default_graphics_heads")]
    pub heads: u8,
    /// Host-forced Newport video mode at VM start (`guest` = IRIX/setmon controls VC2).
    #[serde(default)]
    pub resolution: crate::dev::ng1::vc2_timings::NewportResolution,
}

impl GraphicsSection {
    /// Pixel size for host window layout when a preset is selected.
    pub fn host_display_size(&self) -> (u32, u32) {
        self.resolution
            .visible_size()
            .unwrap_or((1280, 1024))
    }
}

fn default_graphics_heads() -> u8 { 1 }

impl Default for GraphicsSection {
    fn default() -> Self {
        Self {
            board: GraphicsBoard::default(),
            heads: default_graphics_heads(),
            resolution: crate::dev::ng1::vc2_timings::NewportResolution::default(),
        }
    }
}

/// Emulated CPU. Runtime-selectable: each model is its own monomorphisation,
/// so the hot path carries no per-model branch.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
#[clap(rename_all = "lower")]
pub enum CpuModel {
    /// MIPS R4400, 16K direct-mapped L1s, 1 MB L2. The Indy IRIS has always shipped.
    #[default]
    R4400,
    /// MIPS R5000, 2-way 32K L1s, no secondary cache, MIPS IV.
    R5000,
    /// MIPS R10000, 2-way 32K L1s, 1 MB secondary cache, MIPS IV. The CPU in
    /// the Indigo2 IMPACT (IP28). Bring-up only — see docs/ip28-bringup.md.
    R10000,
}

impl CpuModel {
    pub const ALL: [Self; 3] = [Self::R4400, Self::R5000, Self::R10000];

    pub fn label(self) -> &'static str {
        match self {
            Self::R4400 => "MIPS R4400",
            Self::R5000 => "MIPS R5000",
            Self::R10000 => "MIPS R10000",
        }
    }
}

/// `[machine]` section — platform identity (not performance knobs).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineSection {
    #[serde(default)]
    pub profile: MachineProfile,
    #[serde(default)]
    pub cpu: CpuModel,
}

impl Default for MachineSection {
    fn default() -> Self {
        Self { profile: MachineProfile::default(), cpu: CpuModel::default() }
    }
}

/// Misc debug/capture runtime tuning (`[debug]` section). Applied to process
/// env at Start. These knobs are independent of any particular JIT engine —
/// gui_gl_capture picks iris-gui's framebuffer capture renderer, no_idle
/// disables the interpreter's idle-park path, and debug_log seeds the devlog
/// module spec — see their respective `IRIS_*` env var readers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DebugConfig {
    /// iris-gui only: use GlCompositor capture path (IRIS_GUI_GL=1).
    #[serde(default)]
    pub gui_gl_capture: bool,
    #[serde(default)]
    pub no_idle: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub debug_log: String,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            gui_gl_capture: false,
            no_idle: false,
            debug_log: String::new(),
        }
    }
}

impl DebugConfig {
    /// Apply to current process environment (CLI and iris-gui before Machine::new).
    /// Publish `[debug]` into the environment, **without** overwriting a
    /// variable the caller set.
    ///
    /// The caller's comment has always promised "env vars still override if
    /// set externally" and the code did the opposite: a config with no
    /// `debug_log` key called `remove_var`, so `IRIS_DEBUG_LOG=l2c ./iris
    /// --config foo.toml` silently deleted its own setting and produced no
    /// logging at all — devlog looked dead when only its bootstrap was.
    ///
    /// This governs all three `[debug]` keys, not just `debug_log`: a caller
    /// who sets `IRIS_NO_IDLE` or `IRIS_GUI_GL` keeps it too.
    pub fn apply_env(&self) {
        set_or_remove_env("IRIS_NO_IDLE", if self.no_idle { "1" } else { "" });
        set_or_remove_env("IRIS_GUI_GL", if self.gui_gl_capture { "1" } else { "" });
        set_or_remove_env("IRIS_DEBUG_LOG", &self.debug_log);
    }
}

/// Set `key` from the config, unless the environment already says otherwise.
///
/// An empty config value means "the config does not mention this", which is
/// not the same as "unset it": only a variable this function itself could
/// have set is cleared, and one that arrived from the caller's environment is
/// left alone.
fn set_or_remove_env(key: &str, val: &str) {
    if std::env::var_os(key).is_some() && val.is_empty() {
        return;
    }
    if val.is_empty() {
        std::env::remove_var(key);
    } else {
        std::env::set_var(key, val);
    }
}

/// jitv2 compile-pool tuning (`[jitv2]` section): the compile-pool thread
/// count, fixed at process startup, never changed at runtime — see
/// `CompileQueue::set_thread_count`'s own doc comment for why — plus the
/// persistent on-disk code cache (`src/jitv2/pcache.rs`), applied to the
/// process environment at `Start` the same way `DebugConfig` applies
/// `[debug]` (see `Jitv2Config::apply_env`). `cache`/`cache_dir` used to be
/// the undocumented `IRIS_JIT_CACHE`/`IRIS_JIT_CACHE_DIR` env vars; those
/// still work as a direct override (same "env wins if externally set" rule
/// as `[debug]`), but the config is now the documented interface.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Jitv2Config {
    #[serde(default = "default_jitv2_threads")]
    pub threads: usize,
    /// Keep compiled pages on disk across runs (`IRIS_JIT_CACHE`). 86-95%
    /// warm hits in the measurements in `docs/jitv2-persistent-cache.md`.
    #[serde(default)]
    pub cache: bool,
    /// Where to keep the cache (`IRIS_JIT_CACHE_DIR`). Blank (the default)
    /// means the platform's user cache directory — `~/Library/Caches` on
    /// macOS, `%LOCALAPPDATA%` on Windows, `$XDG_CACHE_HOME` or `~/.cache`
    /// elsewhere — with `iris/jitv2` inside it; see `pcache::default_base`.
    /// A directory that doesn't exist yet is created on first use.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub cache_dir: String,
}

fn default_jitv2_threads() -> usize { 1 }

impl Default for Jitv2Config {
    fn default() -> Self {
        Self { threads: default_jitv2_threads(), cache: false, cache_dir: String::new() }
    }
}

impl Jitv2Config {
    /// Apply to current process environment (CLI and iris-gui before
    /// Machine::new). Same "doesn't overwrite a variable the caller set"
    /// rule as `DebugConfig::apply_env` — see its doc comment.
    pub fn apply_env(&self) {
        set_or_remove_env("IRIS_JIT_CACHE", if self.cache { "1" } else { "" });
        set_or_remove_env("IRIS_JIT_CACHE_DIR", &self.cache_dir);
    }
}

/// `[clock]` section — CP0 Count/Compare timer frequency.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ClockConfig {
    /// CP0 Count frequency in MHz. Count ticks at a fixed rate; there is no
    /// runtime inference. None = `DEFAULT_COUNT_HZ` (33 MHz), which is what
    /// IRIX expects — it reports a 66 MHz CPU for a 33 MHz Count, and since
    /// these systems are interrupt-driven the emulator running at a different
    /// real speed has no ill effects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_mhz: Option<f64>,
}

impl Default for ClockConfig {
    fn default() -> Self {
        Self { fixed_mhz: None }
    }
}

/// Earliest time the DS1386 can hold: 1970-01-01 00:00:00 UTC. The emulated
/// chip counts centiseconds since the Unix epoch, so it cannot go below this.
pub const RTC_MIN_UNIX: i64 = 0;
/// Latest time the DS1386 can hold: 2039-12-31 23:59:59 UTC. The year register
/// is two BCD digits read as 1940 + n, and years before 1970 are unusable
/// (see `RTC_MIN_UNIX`), so 2039 is the last year that round-trips.
pub const RTC_MAX_UNIX: i64 = 2_208_988_799;

fn is_zero_i64(v: &i64) -> bool { *v == 0 }

/// `[rtc_offset]` section — where the guest's real-time clock starts,
/// relative to the host's current time. Every field is a signed amount, so
/// `years = -18` alone sets the clock back 18 years to the day.
///
/// Applied once, when the RTC is seeded from the host clock at startup; the
/// clock then runs forward normally. Snapshots keep the time they were saved
/// with, and IRIX setting the time itself still works as usual.
///
/// Years and months are calendar steps (applied first, day clamped to the
/// target month, so Mar 31 − 1 month is Feb 28/29). Days, hours, minutes and
/// seconds are then added as a plain duration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct RtcOffset {
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub years: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub months: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub days: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub hours: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub minutes: i64,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub seconds: i64,
}

impl RtcOffset {
    pub fn is_zero(&self) -> bool { *self == Self::default() }

    /// Shift a host Unix time (seconds, UTC) by this offset. Not clamped to
    /// what the RTC can hold; see [`RtcOffset::apply_clamped`].
    pub fn apply(&self, unix_secs: i64) -> i64 {
        let days = unix_secs.div_euclid(86_400);
        let secs_of_day = unix_secs.rem_euclid(86_400);
        let (y, m, d) = civil_from_days(days);

        // Calendar step. Clamp the year to something the day arithmetic can't
        // overflow on; anything that far out is clamped to the RTC range anyway.
        let total_months = (y * 12 + (m - 1))
            .saturating_add(self.years.saturating_mul(12))
            .saturating_add(self.months)
            .clamp(0, 10_000 * 12);
        let (y, m) = (total_months / 12, total_months % 12 + 1);
        let d = d.min(days_in_month(y, m));

        (days_from_civil(y, m, d) * 86_400 + secs_of_day)
            .saturating_add(self.days.saturating_mul(86_400))
            .saturating_add(self.hours.saturating_mul(3_600))
            .saturating_add(self.minutes.saturating_mul(60))
            .saturating_add(self.seconds)
    }

    /// [`RtcOffset::apply`], clamped to what the DS1386 can represent. The
    /// bool is true when clamping changed the result.
    pub fn apply_clamped(&self, unix_secs: i64) -> (i64, bool) {
        let t = self.apply(unix_secs);
        let c = t.clamp(RTC_MIN_UNIX, RTC_MAX_UNIX);
        (c, c != t)
    }

    /// Compact form for logs, e.g. `-3y -4mo -2d -13h -4m -27s`; `+0` if zero.
    pub fn describe(&self) -> String {
        let parts: Vec<String> = [
            (self.years, "y"), (self.months, "mo"), (self.days, "d"),
            (self.hours, "h"), (self.minutes, "m"), (self.seconds, "s"),
        ]
        .iter()
        .filter(|(v, _)| *v != 0)
        .map(|(v, unit)| format!("{:+}{}", v, unit))
        .collect();
        if parts.is_empty() { "+0".to_string() } else { parts.join(" ") }
    }
}

/// `YYYY-MM-DD HH:MM:SS` for a Unix time in seconds (UTC).
pub fn format_unix_utc(unix_secs: i64) -> String {
    let (y, m, d) = civil_from_days(unix_secs.div_euclid(86_400));
    let s = unix_secs.rem_euclid(86_400);
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", y, m, d, s / 3600, s / 60 % 60, s % 60)
}

fn is_leap_year(y: i64) -> bool { (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 }

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        2 if is_leap_year(y) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

// Proleptic Gregorian date <-> days since 1970-01-01 (Howard Hinnant's
// `days_from_civil` / `civil_from_days`).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { yoe + era * 400 + 1 } else { yoe + era * 400 }, m, d)
}

/// Host-side performance tuning (`[perf]` section).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PerfConfig {
    #[serde(default)]
    pub thread_affinity: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_core: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rex3_core: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_core: Option<u32>,
}

impl Default for PerfConfig {
    fn default() -> Self {
        Self {
            thread_affinity: false,
            cpu_core: None,
            rex3_core: None,
            refresh_core: None,
        }
    }
}

/// HAL2 / cpal audio output tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Pre-buffer duration (ms) before feeding the cpal ring. Default 20.
    #[serde(default = "default_audio_prebuf_ms")]
    pub prebuf_ms: u64,
    /// Fixed cpal buffer size in frames (stereo pairs). Unset = host default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpal_buffer_frames: Option<u32>,
}

fn default_audio_prebuf_ms() -> u64 { 20 }

impl Default for AudioConfig {
    fn default() -> Self {
        Self { prebuf_ms: default_audio_prebuf_ms(), cpal_buffer_frames: None }
    }
}

/// VINO video-in configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VinoConfig {
    /// Where the IndyCam feed comes from.
    #[serde(default)]
    pub source: VinoSource,
    /// Broadcast standard.  Affects field rate (60 vs 50 Hz) and field size.
    #[serde(default)]
    pub standard: VinoStandard,
    /// Index of the host camera to open (0 = default).  Only meaningful when
    /// `source = "camera"`.
    #[serde(default)]
    pub camera_index: u32,
}

/// (De)serialize the `scsi` map through string keys. TOML (and the `toml`
/// crate's serializer) requires map keys to be strings, but the map is keyed
/// by `u8`, so `toml::to_string` would fail with "map key was not a string".
/// JSON is unaffected (it already stringifies map keys); this just makes the
/// representation explicit and symmetric so iris.toml export round-trips.
mod scsi_keys {
    use super::ScsiDeviceConfig;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::{BTreeMap, HashMap};

    pub fn serialize<S: Serializer>(
        map: &HashMap<u8, ScsiDeviceConfig>,
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        // BTreeMap → stable, ID-sorted output.
        map.iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<BTreeMap<String, &ScsiDeviceConfig>>()
            .serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        de: D,
    ) -> Result<HashMap<u8, ScsiDeviceConfig>, D::Error> {
        HashMap::<String, ScsiDeviceConfig>::deserialize(de)?
            .into_iter()
            .map(|(k, v)| k.parse::<u8>().map(|id| (id, v)).map_err(serde::de::Error::custom))
            .collect()
    }
}

/// Top-level machine configuration.
///
/// Field order matters for TOML export: the `toml` serializer requires every
/// scalar/inline-value field to be emitted before any table or array-of-table
/// field, so all scalars are declared first and the table-valued fields
/// (`scsi`, `nfs`, `port_forward`, `vino`) come last.
///
/// `deny_unknown_fields` makes typos and misplaced keys a hard parse error
/// instead of silently ignoring them. This catches a common footgun: writing
/// `mode = "pcap"` at the top level (because `[network]` was left commented
/// out) used to be silently dropped, so PCAP never engaged and networking
/// quietly stayed on NAT.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineConfig {
    /// Path to the PROM ROM image.
    #[serde(default = "default_prom")]
    pub prom: String,

    /// Path to the NVRAM file. The emulated DS1386 NVRAM is loaded from
    /// here at startup (if the file exists) and `iris-ci rtc-save` writes
    /// back to it by default. Per-config NVRAM files avoid the footgun
    /// where two configs (e.g. iris-irix53.toml and iris-irix65.toml)
    /// otherwise share `nvram.bin` and overwrite each other's PROM env.
    #[serde(default = "default_nvram")]
    pub nvram: String,

    /// Path to the Indigo2 (IP22) motherboard EEPROM file (93CS56 — NVRAM
    /// env vars + MAC, see `nveeprom` monitor command). Loaded at startup
    /// if the file exists; `nveeprom save` writes back to it by default.
    /// Ignored on Indy (no such chip). Per-config files avoid the same
    /// cross-install footgun as `nvram`.
    #[serde(default = "default_nveeprom")]
    pub nveeprom: String,

    /// RAM bank sizes in MB. Valid values: 0 (absent), 8, 16, 32, 64, 128.
    #[serde(default = "default_banks")]
    pub banks: [u32; 4],

    /// Window scale factor (1 = native, 2 = 2× for HiDPI/4K). CLI --2x overrides this.
    #[serde(default = "default_scale")]
    pub scale: u32,

    /// Run without graphics (no window, no REX3). Use no_audio to also disable HAL2.
    /// Useful for headless/server/CI environments.
    #[serde(default)]
    pub headless: bool,

    /// Disable audio emulation (no HAL2). Independent of headless/graphics.
    #[serde(default)]
    pub no_audio: bool,

    /// If Some(port), start the GDB RSP stub on that TCP port.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gdb_port: Option<u16>,

    /// If Some(path), load this static ELF32 MSB binary into RAM at startup and
    /// set PC to its entry point (bare-metal test binaries; see --load-elf).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_elf: Option<String>,

    /// Map the bare-metal test device (guest console, machine-state dump, exit
    /// code) into GIO expansion slot 0. Off by default; see src/dev/testdev.rs.
    #[serde(default)]
    pub test_device: bool,

    /// Where the test device's DUMP register writes its JSON. Defaults to
    /// `iris-testdev-dump.json` in the working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_device_dump: Option<String>,

    /// cheritest convention: a guest write to CP0 register 26 triggers the same
    /// dump as the test device's DUMP register. Off by default and must stay
    /// that way for a normal boot — CP0 26 is ECC on a real R4400 and the PROM
    /// writes it during cache initialisation.
    #[serde(default)]
    pub cheritest_dump_hook: bool,

    /// NAT subnet in CIDR notation (e.g. "192.168.5.0/24").
    /// The gateway gets host .1 and the guest (IRIX) gets host .2.
    /// Defaults to "192.168.0.0/24" if not set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nat_subnet: Option<String>,

    /// CI mode: opens a control socket for automation, applies speed-favoring
    /// fidelity shortcuts. Implies headless unless ci_display is also set.
    #[serde(default)]
    pub ci: bool,

    /// Unix socket path for CI control. Used only when `ci` is true.
    #[serde(default = "default_ci_socket")]
    pub ci_socket: String,

    /// With `ci`, keep the Newport window visible (deferred rendering) for
    /// interactive test development.
    #[serde(default)]
    pub ci_display: bool,

    /// Pixels of host trackpad/wheel movement that equal one PS/2 scroll
    /// detent. Lower = faster scroll; higher = slower. Default 40.
    /// Tune if scroll feels too fast or too slow on your hardware.
    #[serde(default = "default_scroll_pixels_per_line")]
    pub mouse_scroll_pixels_per_line: f64,

    /// Lock the window's aspect ratio to the emulated display (picture +
    /// status bar) while resizing, so it fills the window without letterbox
    /// bars. Set to false if you have a non-standard monitor and prefer free
    /// resizing — the display is then letterboxed to fit. Default: true.
    #[serde(default = "default_lock_aspect_ratio")]
    pub lock_aspect_ratio: bool,

    /// Optional file path that will receive every byte emitted on ttyd1
    /// (the IRIX serial console) in `--ci` mode. Append-only. Useful for
    /// keeping a continuously-updated transcript of the install or test run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serial_log: Option<String>,

    // --- table / array-of-table fields: must be emitted after all scalars ---

    /// SCSI devices keyed by ID 1–7. Missing IDs are not attached.
    #[serde(default = "default_scsi", with = "scsi_keys")]
    pub scsi: std::collections::HashMap<u8, ScsiDeviceConfig>,

    /// NFS share configuration. If present, the NAT's in-process NFS server
    /// (`src/nfsudp.rs`) exports `shared_dir` to the guest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nfs: Option<NfsConfig>,

    /// Port forwarding rules (host port → guest port).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub port_forward: Vec<PortForwardConfig>,

    /// VINO video-in configuration (IndyCam emulation source).
    #[serde(default)]
    pub vino: VinoConfig,

    /// Networking backend selection (`[network]` section). Defaults to NAT.
    #[serde(default)]
    pub network: NetworkSection,

    /// Defer SCSI status interrupts so wd33c93_loop exits before INT fires.
    /// Required for OpenBSD/NetBSD; disable if observing spurious SCSI timeouts.
    #[serde(default = "default_scsi_deferred_int")]
    pub scsi_deferred_int: bool,

    /// HAL2 audio output tuning (`[audio]` section).
    #[serde(default)]
    pub audio: AudioConfig,

    /// Machine platform profile (`[machine]` section).
    #[serde(default)]
    pub machine: MachineSection,

    /// Graphics options (`[graphics]` section).
    #[serde(default)]
    pub graphics: GraphicsSection,

    /// IMPACT/MGRAS slot population (`[impact]` section, Indigo2 preview).
    #[serde(default)]
    pub impact: ImpactSection,

    /// Misc debug/capture runtime tuning (`[debug]` section).
    #[serde(default)]
    pub debug: DebugConfig,

    /// jitv2 compile-pool tuning (`[jitv2]` section).
    #[serde(default)]
    pub jitv2: Jitv2Config,

    /// Host performance tuning (`[perf]` section).
    #[serde(default)]
    pub perf: PerfConfig,

    /// CP0 Count/Compare clock tuning (`[clock]` section).
    #[serde(default)]
    pub clock: ClockConfig,

    /// Guest real-time clock offset from host time (`[rtc_offset]` section).
    #[serde(default, skip_serializing_if = "RtcOffset::is_zero")]
    pub rtc_offset: RtcOffset,

    /// N64 development board (Ultra64) — GIO slot 0 + shm IPC.
    #[serde(default)]
    pub ultra64: Ultra64Config,
}

fn default_scsi_deferred_int() -> bool { true }

#[cfg(unix)]
fn default_ci_socket() -> String { "/tmp/iris.sock".to_string() }
#[cfg(windows)]
fn default_ci_socket() -> String { "127.0.0.1:19851".to_string() }
#[cfg(not(any(unix, windows)))]
fn default_ci_socket() -> String { "/tmp/iris.sock".to_string() }

/// True when `ci_socket` is a TCP `host:port` address (Windows CI default).
pub fn ci_socket_is_tcp(path: &str) -> bool {
    let path = path.trim();
    if path.starts_with("tcp:") {
        return true;
    }
    // host:port heuristic (not a filesystem path)
    if path.contains(':') && !path.starts_with('\\') && !path.starts_with('/') {
        path.rsplit_once(':')
            .map(|(_, port)| port.parse::<u16>().is_ok())
            .unwrap_or(false)
    } else {
        false
    }
}

/// Normalize CI socket address to `host:port` for TCP clients.
pub fn ci_socket_tcp_addr(path: &str) -> String {
    let path = path.trim();
    if let Some(rest) = path.strip_prefix("tcp:") {
        rest.to_string()
    } else {
        path.to_string()
    }
}
fn default_scroll_pixels_per_line() -> f64 { 40.0 }
fn default_lock_aspect_ratio() -> bool { true }

fn default_prom() -> String {
    "prom.bin".to_string()
}

fn default_nvram() -> String {
    "nvram.bin".to_string()
}

fn default_nveeprom() -> String {
    "nveeprom.bin".to_string()
}

fn default_banks() -> [u32; 4] {
    [128, 128, 0, 0]
}

fn default_scale() -> u32 { 1 }

fn default_scsi() -> std::collections::HashMap<u8, ScsiDeviceConfig> {
    let mut map = std::collections::HashMap::new();
    map.insert(1, ScsiDeviceConfig {
        path: "scsi1.raw".to_string(),
        ..Default::default()
    });
    map.insert(4, ScsiDeviceConfig {
        path: "cdrom4.iso".to_string(),
        cdrom: true,
        ..Default::default()
    });
    map
}

impl Default for MachineConfig {
    fn default() -> Self {
        Self {
            prom: default_prom(),
            nvram: default_nvram(),
            nveeprom: default_nveeprom(),
            banks: default_banks(),
            scsi: default_scsi(),
            scale: default_scale(),
            nfs: None,
            port_forward: vec![],
            headless: false,
            no_audio: false,
            gdb_port: None,
            load_elf: None,
            test_device: false,
            test_device_dump: None,
            cheritest_dump_hook: false,
            nat_subnet: None,
            ci: false,
            ci_socket: default_ci_socket(),
            ci_display: false,
            serial_log: None,
            vino: VinoConfig::default(),
            network: NetworkSection::default(),
            mouse_scroll_pixels_per_line: default_scroll_pixels_per_line(),
            lock_aspect_ratio: default_lock_aspect_ratio(),
            scsi_deferred_int: default_scsi_deferred_int(),
            audio: AudioConfig::default(),
            machine: MachineSection::default(),
            graphics: GraphicsSection::default(),
            impact: ImpactSection::default(),
            debug: DebugConfig::default(),
            jitv2: Jitv2Config::default(),
            perf: PerfConfig::default(),
            clock: ClockConfig::default(),
            rtc_offset: RtcOffset::default(),
            ultra64: Ultra64Config::default(),
        }
    }
}


impl MachineConfig {
    /// Load from `iris.toml` if it exists, otherwise return defaults.
    ///
    /// A *missing* file is fine (defaults are used). A file that exists but
    /// fails to parse is **fatal**: previously we silently fell back to
    /// defaults, which hid config mistakes — e.g. a Windows `pcap_interface`
    /// with unescaped backslashes would discard the entire config, so the
    /// emulator would quietly auto-pick a different interface and run with NAT
    /// instead of the settings the user wrote.
    pub fn load_toml(path: &str) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self::default();
        };
        match toml::from_str::<Self>(&text) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("Configuration error: failed to parse {}:\n{}", path, e);
                // Common footgun: backslashes in a double-quoted string (Windows
                // pcap device names look like \Device\NPF_{GUID}).
                if text.contains("\\Device\\NPF") || e.to_string().contains("escape") {
                    eprintln!(
                        "\nhint: backslashes are escape characters inside a \"double-quoted\" TOML string.\n\
                         For a Windows pcap interface, either use the numeric index from\n\
                         `iris --list-net-interfaces` (e.g. pcap_interface = \"1\") or a TOML\n\
                         'single-quoted' literal string:\n\
                         \n    pcap_interface = '\\Device\\NPF_{{...}}'\n"
                    );
                }
                std::process::exit(1);
            }
        }
    }

    /// Validate bank sizes, returns a description of any errors.
    pub fn validate(&self) -> Result<(), String> {
        if self.graphics.heads != 1 && self.graphics.heads != 2 {
            return Err(format!(
                "graphics.heads {} is invalid (valid: 1, 2)",
                self.graphics.heads
            ));
        }
        if self.graphics.board != GraphicsBoard::Newport {
            let name = if self.graphics.board == GraphicsBoard::Xz { "xz" } else { "extreme" };
            if self.graphics.board == GraphicsBoard::Extreme && self.machine.profile != MachineProfile::Indigo2Ip22 {
                return Err(
                    "graphics.board \"extreme\" is only valid on Indigo2 (machine.profile = indigo2_ip22)".into(),
                );
            }
            if self.graphics.heads != 1 {
                return Err(format!("graphics.board \"{name}\" does not support dual-head (graphics.heads must be 1)"));
            }
            if !self.graphics.resolution.is_guest() {
                return Err("graphics.resolution presets require Newport (graphics.board = newport)".into());
            }
            if self.impact.any_enabled() {
                return Err(format!("graphics.board \"{name}\" and [impact] both claim the GIO gfx slot"));
            }
        }
        if self.impact.any_enabled()
            && !matches!(self.machine.profile, MachineProfile::Indigo2Ip22 | MachineProfile::Indigo2Ip28)
        {
            return Err(
                "[impact] slots need an Indigo2 (machine.profile = indigo2_ip22 or indigo2_ip28)".into(),
            );
        }
        self.impact.validate()?;
        for (&id, dev) in &self.scsi {
            if dev.controller != 0 && self.machine.profile != MachineProfile::Indigo2Ip22 {
                return Err(format!(
                    "scsi.{}: controller {} is only valid on Indigo2 (machine.profile = indigo2_ip22) — Indy has a single SCSI controller (0)",
                    id, dev.controller
                ));
            }
            if dev.controller > 1 {
                return Err(format!("scsi.{}: controller {} is invalid (valid: 0, 1)", id, dev.controller));
            }
        }
        if self.scale < 1 || self.scale > 4 {
            return Err(format!("scale {} is invalid (valid: 1, 2, 3, 4)", self.scale));
        }
        for (i, &sz) in self.banks.iter().enumerate() {
            if !VALID_BANK_SIZES.contains(&sz) {
                return Err(format!(
                    "bank{} size {} MB is invalid (valid: {:?})",
                    i, sz, VALID_BANK_SIZES
                ));
            }
            // The IP22/IP24 MC cannot express a 256 MB bank at its base
            // shift; only the IP28's can.
            if sz == 256 && !self.machine.profile.ip28() {
                return Err(format!(
                    "bank{} size 256 MB needs the IP28 (machine.profile = indigo2_ip28)",
                    i
                ));
            }
        }
        if let Some(ref s) = self.nat_subnet {
            if let Err(e) = parse_nat_subnet(s) {
                return Err(format!("nat_subnet \"{}\": {}", s, e));
            }
        }
        if let Some(ref mac) = self.network.mac {
            parse_mac(mac).map_err(|e| format!("network.mac: {}", e))?;
        }
        for (id, dev) in &self.scsi {
            if *id == 0 || *id > 7 {
                return Err(format!("SCSI ID {} is out of range (1–7)", id));
            }
            // A CD-ROM may legitimately start with an empty tray (no path, no
            // discs) and have media loaded at runtime; any discs list is valid
            // as a changer queue. So there is nothing CD-ROM-specific to check.
            if dev.is_daynaport() {
                if dev.cdrom || dev.overlay || dev.scratch {
                    return Err(format!(
                        "SCSI ID {}: kind = \"daynaport\" has no disk image, so cdrom / \
                         overlay / scratch don't apply", id));
                }
                if let Some(mac) = &dev.mac {
                    parse_mac(mac).map_err(|e| format!("SCSI ID {}: mac: {}", id, e))?;
                }
                if let Some(cidr) = &dev.subnet {
                    parse_nat_subnet(cidr)
                        .map_err(|e| format!("SCSI ID {}: subnet \"{}\": {}", id, cidr, e))?;
                }
                // Each DaynaPort runs its own NAT gateway. Sharing a subnet with
                // ec0 gives the guest two interfaces on one network and nothing
                // routes predictably.
                let dp = dev.daynaport_params(*id)
                    .map_err(|e| format!("SCSI ID {}: {}", id, e))?;
                let main = self.nat_subnet.as_deref()
                    .map(|c| parse_nat_subnet(c).map(|(g, c2, n)| NatSubnet {
                        gateway_ip: g, client_ip: c2, netmask: n }))
                    .transpose()?
                    .unwrap_or_default();
                if dp.subnet.gateway_ip == main.gateway_ip {
                    return Err(format!(
                        "SCSI ID {}: DaynaPort subnet {} collides with the machine's NAT subnet \
                         (ec0). Give the DaynaPort its own, e.g. subnet = \"192.168.10.0/24\".",
                        id, dp.subnet.gateway_ip));
                }
            }
        }
        if let Some(mhz) = self.clock.fixed_mhz {
            if !(mhz > 0.0) || !mhz.is_finite() {
                return Err(format!("clock.fixed_mhz {} is invalid (must be a positive number)", mhz));
            }
        }
        if self.jitv2.threads == 0 {
            return Err(
                "jitv2.threads must be at least 1 — 0 would silently degrade to no compile threads, everything sticky-fails to compile forever".into(),
            );
        }
        Ok(())
    }

    /// Extract network-related settings into a `NetworkConfig`.
    /// Parses `nat_subnet` from CIDR — safe to unwrap because `validate()` already accepted it.
    pub fn network(&self) -> NetworkConfig {
        let nat_subnet = self.nat_subnet.as_deref().map(|cidr| {
            let (gateway_ip, client_ip, netmask) = parse_nat_subnet(cidr)
                .expect("nat_subnet: validate() should have caught this");
            NatSubnet { gateway_ip, client_ip, netmask }
        });
        let mac = self.network.mac.as_deref()
            .map(|s| parse_mac(s).expect("network.mac: validate() should have caught this"))
            .unwrap_or(DEFAULT_MAC);
        NetworkConfig {
            nfs:          self.nfs.clone(),
            port_forward: self.port_forward.clone(),
            nat_subnet,
            mode:         self.network.mode,
            pcap_interface: self.network.pcap_interface.clone(),
            nfs_pcap_ip:  self.network.nfs_pcap_ip,
            mac,
            tftp_dir:     self.network.tftp_dir.as_ref().map(std::path::PathBuf::from),
        }
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

/// Config file used when `--config` is not given. A missing one is not an
/// error (defaults are used); a missing *explicit* one is — see `load_config`.
pub const DEFAULT_CONFIG: &str = "iris.toml";

#[derive(Parser, Debug)]
#[command(name = "iris", about = "SGI Indy (MIPS R4400) emulator")]
pub struct Cli {
    /// Path to iris.toml config file [default: iris.toml]
    #[arg(long, default_value = DEFAULT_CONFIG)]
    pub config: String,

    /// Path to PROM image
    #[arg(long)]
    pub prom: Option<String>,

    /// Emulate an SGI Indigo2 (IP22) instead of the default Indy (IP24).
    /// Overrides `[machine].profile` in the config file.
    #[arg(long, default_value_t = false)]
    pub ip22: bool,

    /// Path to NVRAM file (default: nvram.bin in cwd)
    #[arg(long)]
    pub nvram: Option<String>,

    /// Path to Indigo2 motherboard EEPROM file (default: nveeprom.bin in cwd)
    #[arg(long)]
    pub nveeprom: Option<String>,

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

    /// jitv2 compile-pool thread count (fixed at startup, see jitv2.threads)
    #[arg(long = "jitv2-threads", value_name = "N")]
    pub jitv2_threads: Option<usize>,

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

    /// NAT subnet in CIDR notation (e.g. 192.168.5.0/24).
    /// Gateway gets .1, guest (IRIX) gets .2. Default: 192.168.0.0/24.
    #[arg(long = "nat-subnet", value_name = "CIDR")]
    pub nat_subnet: Option<String>,

    /// Networking backend: "nat" (default, software gateway) or "pcap"
    /// (bridge onto a real host interface; requires --features pcap).
    #[arg(long = "net-mode", value_name = "MODE")]
    pub net_mode: Option<NetMode>,

    /// Host interface to bridge onto in PCAP mode (e.g. eth0, en0).
    /// Implies --net-mode pcap. List candidates with --list-net-interfaces.
    #[arg(long = "pcap-interface", value_name = "IFACE")]
    pub pcap_interface: Option<String>,

    /// Print the host network interfaces libpcap can bridge onto, then exit.
    /// Requires a build with --features pcap.
    #[arg(long = "list-net-interfaces", default_value_t = false)]
    pub list_net_interfaces: bool,

    /// Disable deferred SCSI status interrupts (default: enabled for OpenBSD/NetBSD compatibility).
    #[arg(long = "no-scsi-deferred-int", default_value_t = false)]
    pub no_scsi_deferred_int: bool,

    /// Emulated CPU: `r4400` (default) or `r5000`. Overrides `[machine] cpu`.
    ///
    /// Both models are compiled into every build — each is its own
    /// monomorphisation, so the hot path carries no per-model branch — and this
    /// picks between them at construction. The `r5k` cargo feature no longer
    /// selects the model and is vestigial for that purpose.
    #[arg(long = "cpu", value_name = "MODEL")]
    pub cpu: Option<CpuModel>,

    /// Enable GDB stub on the given TCP port (e.g. --gdb-port 1234).
    /// Connect with: target remote localhost:<port>
    #[arg(long = "gdb-port", value_name = "PORT")]
    pub gdb_port: Option<u16>,

    /// Map the bare-metal test device into GIO expansion slot 0: SIGNATURE,
    /// PUTC (guest console → stdout), DUMP (machine state → JSON) and EXIT
    /// (terminate with the guest's exit code). Off by default.
    #[arg(long = "test-device", default_value_t = false)]
    pub test_device: bool,

    /// Where the test device writes its machine-state dump.
    #[arg(long = "test-device-dump", value_name = "FILE")]
    pub test_device_dump: Option<String>,

    /// cheritest convention: a guest write to CP0 register 26 dumps machine
    /// state. Needs --test-device. Never enable for a normal boot — CP0 26 is
    /// ECC on a real R4400 and the PROM writes it during cache init.
    #[arg(long = "cheritest-dump-hook", default_value_t = false)]
    pub cheritest_dump_hook: bool,

    /// Serve this directory read-only over TFTP at the gateway address, so the
    /// PROM can network-boot from it: `boot -f bootp()<file>`. Off when unset.
    #[arg(long = "tftp-dir", value_name = "DIR")]
    pub tftp_dir: Option<String>,

    /// Load a static ELF32 MSB (big-endian MIPS) binary into RAM before the
    /// CPU starts and set PC to its entry point, instead of booting the PROM.
    /// For bare-metal test binaries; see also the monitor's `loadelf`.
    #[arg(long = "load-elf", value_name = "FILE")]
    pub load_elf: Option<String>,

    /// CI mode: enable the control socket and apply speed-favoring fidelity
    /// shortcuts. Implies --headless unless --ci-display is also set.
    #[arg(long, default_value_t = false)]
    pub ci: bool,

    /// Override the default control-socket path (/tmp/iris.sock).
    #[arg(long = "ci-socket", value_name = "PATH")]
    pub ci_socket: Option<String>,

    /// With --ci, keep the Newport window visible for interactive test
    /// development (deferred rendering at 10–15 fps).
    #[arg(long = "ci-display", default_value_t = false)]
    pub ci_display: bool,

    /// With --ci, append every byte the guest emits on ttyd1 (IRIX serial
    /// console) to this file. Useful for live tailing during an install.
    #[arg(long = "serial-log", value_name = "FILE")]
    pub serial_log: Option<String>,

    /// Override the fixed CP0 Count frequency in MHz (default 33, which IRIX
    /// reports as a 66 MHz CPU), e.g. --clock-fixed-mhz 50.
    #[arg(long = "clock-fixed-mhz", value_name = "MHZ")]
    pub clock_fixed_mhz: Option<f64>,
}

impl Cli {
    /// Merge CLI overrides into a base `MachineConfig`.
    pub fn apply(&self, mut cfg: MachineConfig) -> MachineConfig {
        if let Some(p) = &self.prom    { cfg.prom = p.clone(); }
        if self.ip22 { cfg.machine.profile = MachineProfile::Indigo2Ip22; }
        if let Some(p) = &self.nvram   { cfg.nvram = p.clone(); }
        if let Some(p) = &self.nveeprom { cfg.nveeprom = p.clone(); }
        if let Some(v) = self.bank0    { cfg.banks[0] = v; }
        if let Some(v) = self.bank1    { cfg.banks[1] = v; }
        if let Some(v) = self.bank2    { cfg.banks[2] = v; }
        if let Some(v) = self.bank3    { cfg.banks[3] = v; }
        if let Some(n) = self.jitv2_threads { cfg.jitv2.threads = n; }

        // Helper: insert or update a SCSI device entry.
        let apply_scsi = |map: &mut std::collections::HashMap<u8, ScsiDeviceConfig>,
                          id: u8, path: String, cdrom: bool, extra: Vec<String>| {
            let entry = map.entry(id).or_insert_with(|| ScsiDeviceConfig {
                cdrom,
                ..Default::default()
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

        if let Some(cpu) = self.cpu { cfg.machine.cpu = cpu; }
        if self.scale2x { cfg.scale = 2; }
        if self.headless  { cfg.headless  = true; }
        if self.no_audio  { cfg.no_audio  = true; }

        if self.no_scsi_deferred_int { cfg.scsi_deferred_int = false; }
        if self.ci         { cfg.ci         = true; }
        if let Some(p) = &self.ci_socket { cfg.ci_socket = p.clone(); }
        if self.ci_display { cfg.ci_display = true; }
        if let Some(p) = &self.serial_log { cfg.serial_log = Some(p.clone()); }
        if let Some(mhz) = self.clock_fixed_mhz { cfg.clock.fixed_mhz = Some(mhz); }
        // NB: --ci does NOT imply --headless. REX3 stays alive so screenshots
        // work; main.rs simply skips the host window when ci && !ci_display.

        // NFS: --nfs-dir enables the in-core NFS export.
        if let Some(dir) = &self.nfs_dir {
            let base = cfg.nfs.get_or_insert_with(|| NfsConfig {
                shared_dir: dir.clone(),
                version: Default::default(),
            });
            base.shared_dir = dir.clone();
        }

        if let Some(p) = self.gdb_port { cfg.gdb_port = Some(p); }
        if let Some(ref p) = self.load_elf { cfg.load_elf = Some(p.clone()); }
        if let Some(ref d) = self.tftp_dir { cfg.network.tftp_dir = Some(d.clone()); }
        if self.test_device { cfg.test_device = true; }
        if let Some(ref p) = self.test_device_dump { cfg.test_device_dump = Some(p.clone()); }
        if self.cheritest_dump_hook { cfg.cheritest_dump_hook = true; }
        if let Some(ref s) = self.nat_subnet { cfg.nat_subnet = Some(s.clone()); }

        if let Some(m) = self.net_mode { cfg.network.mode = m; }
        if let Some(ref iface) = self.pcap_interface {
            cfg.network.pcap_interface = Some(iface.clone());
            // Specifying an interface implies PCAP mode unless the user also
            // explicitly asked for NAT.
            if self.net_mode.is_none() {
                cfg.network.mode = NetMode::Pcap;
            }
        }

        cfg
    }
}

/// Parse CLI, load TOML, merge, and validate. Exits on error.
/// Returns (machine_config, window_scale) where window_scale is 1 or 2.
pub fn load_config() -> (MachineConfig, u32) {
    let cli = Cli::parse();

    // --list-net-interfaces: print candidate PCAP interfaces and exit. Handled
    // here (before machine construction) since it only needs the parsed CLI.
    if cli.list_net_interfaces {
        #[cfg(feature = "pcap")]
        {
            print!("{}", crate::net_pcap::format_interfaces());
            std::process::exit(0);
        }
        #[cfg(not(feature = "pcap"))]
        {
            eprintln!("iris: --list-net-interfaces requires a build with --features pcap");
            std::process::exit(1);
        }
    }

    // A missing `iris.toml` is fine — that's the "just run it" path, and
    // load_toml falls back to defaults. But a config the user *named* on the
    // command line and that isn't there is a typo or a relative path resolved
    // against the wrong cwd, and silently booting a default machine instead
    // (different disks, no DaynaPort, whatever they configured) wastes a run
    // before anyone notices.
    if cli.config != DEFAULT_CONFIG && !std::path::Path::new(&cli.config).exists() {
        eprintln!("Configuration error: --config {} does not exist", cli.config);
        std::process::exit(1);
    }

    let toml_cfg = MachineConfig::load_toml(&cli.config);
    let cfg = cli.apply(toml_cfg);
    let scale = cfg.scale;
    if let Err(e) = cfg.validate() {
        eprintln!("Configuration error: {}", e);
        std::process::exit(1);
    }
    (cfg, scale)
}

/// Parse a CIDR string like "192.168.5.0/24" and return
/// `(gateway_ip, client_ip, netmask)` where gateway=host .1, client=host .2.
///
/// Returns an error string on invalid input.
pub fn parse_nat_subnet(cidr: &str) -> Result<(std::net::Ipv4Addr, std::net::Ipv4Addr, std::net::Ipv4Addr), String> {
    let (addr_str, prefix_str) = cidr.split_once('/').ok_or("expected format IP/PREFIX (e.g. 192.168.5.0/24)")?;
    let base: std::net::Ipv4Addr = addr_str.parse().map_err(|_| format!("invalid IPv4 address \"{}\"", addr_str))?;
    let prefix: u8 = prefix_str.parse().map_err(|_| format!("invalid prefix length \"{}\"", prefix_str))?;
    if prefix > 30 {
        return Err(format!("prefix /{} is too small (minimum /30)", prefix));
    }
    let mask = if prefix == 0 { 0u32 } else { !0u32 << (32 - prefix) };
    let network = u32::from(base) & mask;
    if u32::from(base) != network {
        return Err(format!("address {} is not the network address for /{} (did you mean {}.0/{}?)",
            base, prefix,
            std::net::Ipv4Addr::from(network & 0xFFFFFF00),
            prefix));
    }
    let netmask = std::net::Ipv4Addr::from(mask);
    let gateway_ip = std::net::Ipv4Addr::from(network + 1);
    let client_ip  = std::net::Ipv4Addr::from(network + 2);
    Ok((gateway_ip, client_ip, netmask))
}

#[cfg(test)]
mod export_tests {
    use super::*;

    #[test]
    fn toml_export_roundtrips() {
        let mut cfg = MachineConfig::default();
        cfg.scsi.insert(4, ScsiDeviceConfig {
            path: "/abs/cd.chd".into(), cdrom: true, ..Default::default()
        });
        let s = toml::to_string_pretty(&cfg).expect("serialize");
        let back: MachineConfig = toml::from_str(&s).expect("deserialize");
        assert_eq!(back.scsi.len(), cfg.scsi.len());
        assert_eq!(back.scsi[&1].path, cfg.scsi[&1].path);
        assert_eq!(back.scsi[&4].cdrom, true);
        println!("--- exported toml ---\n{s}");
    }

    #[test]
    fn scsi_kind_daynaport_parses_and_round_trips() {
        let cfg: MachineConfig = toml::from_str(r#"
            [scsi.3]
            kind = "daynaport"
            mac = "00:80:19:aa:bb:cc"
            subnet = "192.168.7.0/24"
        "#).expect("parse");
        let dev = &cfg.scsi[&3];
        assert!(dev.is_daynaport());
        assert!(!dev.is_cdrom());
        cfg.validate().expect("validate");

        let params = dev.daynaport_params(3).expect("params");
        assert_eq!(params.mac, [0x00, 0x80, 0x19, 0xaa, 0xbb, 0xcc]);
        assert_eq!(params.subnet.gateway_ip, Ipv4Addr::new(192, 168, 7, 1));
        assert_eq!(params.subnet.client_ip,  Ipv4Addr::new(192, 168, 7, 2));

        let s = toml::to_string_pretty(&cfg).expect("serialize");
        let back: MachineConfig = toml::from_str(&s).expect("deserialize");
        assert!(back.scsi[&3].is_daynaport(), "kind must survive export:\n{s}");
    }

    /// `cdrom = true` predates `kind`; it must keep working, and a plain disk
    /// must not start serializing a redundant `kind = "disk"`.
    #[test]
    fn cdrom_bool_still_selects_a_cdrom() {
        let cfg: MachineConfig = toml::from_str(r#"
            [scsi.4]
            path = "cd.iso"
            cdrom = true
            [scsi.1]
            path = "disk.raw"
        "#).expect("parse");
        assert!(cfg.scsi[&4].is_cdrom());
        assert_eq!(cfg.scsi[&1].kind(), ScsiKind::Disk);
        let s = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(!s.contains("kind"), "default kind should not be emitted:\n{s}");
    }

    #[test]
    fn daynaport_defaults_to_its_own_subnet_and_a_derived_mac() {
        let cfg: MachineConfig = toml::from_str("[scsi.5]\nkind = \"daynaport\"\n").expect("parse");
        cfg.validate().expect("validate");
        let p = cfg.scsi[&5].daynaport_params(5).expect("params");
        assert_eq!(p.mac, [0x00, 0x80, 0x19, 0x44, 0x50, 5]);
        // Not the driver's own 00:80:19:00:00:NN placeholder, or the acceptance
        // test can't tell a real MAC read from a made-up one.
        assert_ne!(p.mac, [0x00, 0x80, 0x19, 0x00, 0x00, 5]);
        assert_eq!(p.subnet.gateway_ip, Ipv4Addr::new(192, 168, 10, 1));
        assert_ne!(p.subnet.gateway_ip, NatSubnet::default().gateway_ip);
    }

    #[test]
    fn daynaport_rejects_a_subnet_that_collides_with_ec0() {
        let cfg: MachineConfig = toml::from_str(r#"
            nat_subnet = "192.168.10.0/24"
            [scsi.3]
            kind = "daynaport"
        "#).expect("parse");
        let err = cfg.validate().expect_err("subnet collision must be rejected");
        assert!(err.contains("collides"), "{err}");
    }

    #[test]
    fn daynaport_rejects_disk_only_options() {
        let cfg: MachineConfig = toml::from_str(
            "[scsi.3]\nkind = \"daynaport\"\noverlay = true\n").expect("parse");
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn parse_mac_accepts_both_separators_and_rejects_junk() {
        assert_eq!(parse_mac("00:80:19:12:34:56").unwrap(), [0, 0x80, 0x19, 0x12, 0x34, 0x56]);
        assert_eq!(parse_mac("00-80-19-12-34-56").unwrap(), [0, 0x80, 0x19, 0x12, 0x34, 0x56]);
        assert!(parse_mac("00:80:19:12:34").is_err());
        assert!(parse_mac("zz:80:19:12:34:56").is_err());
        assert!(parse_mac("01:80:19:12:34:56").is_err(), "multicast bit must be rejected");
    }

    #[test]
    fn indy_ip24_profile_validates_and_is_guinness() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::IndyIp24;
        cfg.validate().expect("indy_ip24 should validate");
        assert!(cfg.machine.profile.guinness());
    }

    #[test]
    fn indigo2_profile_validates_and_is_fullhouse() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::Indigo2Ip22;
        cfg.validate().expect("indigo2_ip22 should validate on default build");
        assert!(!cfg.machine.profile.guinness());
    }

    #[test]
    fn a_256_mb_bank_is_an_ip28_bank() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::Indigo2Ip22;
        cfg.banks = [256, 128, 0, 0];
        let err = cfg.validate().expect_err("the IP22 MC cannot express a 256 MB bank");
        assert!(err.contains("256 MB"), "{err}");
        {
            cfg.machine.profile = MachineProfile::Indigo2Ip28;
            cfg.machine.cpu = CpuModel::R10000;
            cfg.validate().expect("the IP28 MC can");
        }
    }
}

#[cfg(test)]
mod rtc_offset_tests {
    use super::*;

    /// Unix time for a UTC date, via the same helper the offset uses.
    fn t(y: i64, mo: i64, d: i64, h: i64, mi: i64, s: i64) -> i64 {
        days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + s
    }

    #[test]
    fn civil_round_trip_and_known_epochs() {
        assert_eq!(t(1970, 1, 1, 0, 0, 0), 0);
        assert_eq!(t(2000, 3, 1, 0, 0, 0), 951_868_800);
        assert_eq!(t(2039, 12, 31, 23, 59, 59), RTC_MAX_UNIX);
        for days in [-1_000_000i64, -1, 0, 59, 60, 11_016, 25_567, 1_000_000] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
    }

    #[test]
    fn zero_offset_is_identity() {
        let now = t(2026, 9, 19, 14, 22, 5);
        assert_eq!(RtcOffset::default().apply(now), now);
        assert!(RtcOffset::default().is_zero());
    }

    #[test]
    fn years_back_keeps_date_and_time() {
        let off = RtcOffset { years: -18, ..Default::default() };
        assert_eq!(off.apply(t(2026, 9, 19, 14, 22, 5)), t(2008, 9, 19, 14, 22, 5));
    }

    #[test]
    fn mixed_offset() {
        let off = RtcOffset { years: -3, months: -4, days: -2, hours: -13, minutes: -4, seconds: -27 };
        // 2026-09-19 14:22:05 → 2023-05-19 14:22:05 → minus 2d 13:04:27
        assert_eq!(off.apply(t(2026, 9, 19, 14, 22, 5)), t(2023, 5, 17, 1, 17, 38));
        assert_eq!(off.describe(), "-3y -4mo -2d -13h -4m -27s");
    }

    #[test]
    fn month_step_clamps_day() {
        let back1 = RtcOffset { months: -1, ..Default::default() };
        assert_eq!(back1.apply(t(2024, 3, 31, 12, 0, 0)), t(2024, 2, 29, 12, 0, 0));
        assert_eq!(back1.apply(t(2023, 3, 31, 12, 0, 0)), t(2023, 2, 28, 12, 0, 0));
        let fwd1y = RtcOffset { years: 1, ..Default::default() };
        assert_eq!(fwd1y.apply(t(2024, 2, 29, 0, 0, 0)), t(2025, 2, 28, 0, 0, 0));
        // Months roll the year in both directions.
        let m = RtcOffset { months: 5, ..Default::default() };
        assert_eq!(m.apply(t(2026, 9, 19, 0, 0, 0)), t(2027, 2, 19, 0, 0, 0));
        let m = RtcOffset { months: -10, ..Default::default() };
        assert_eq!(m.apply(t(2026, 9, 19, 0, 0, 0)), t(2025, 11, 19, 0, 0, 0));
    }

    #[test]
    fn positive_small_units_carry() {
        let off = RtcOffset { hours: 30, minutes: 90, seconds: 3_700, ..Default::default() };
        assert_eq!(off.apply(t(2026, 12, 31, 20, 0, 0)), t(2027, 1, 2, 4, 31, 40));
    }

    #[test]
    fn clamped_to_rtc_range() {
        let now = t(2026, 9, 19, 0, 0, 0);
        let (v, c) = RtcOffset { years: 20, ..Default::default() }.apply_clamped(now);
        assert_eq!((v, c), (RTC_MAX_UNIX, true));
        let (v, c) = RtcOffset { years: -60, ..Default::default() }.apply_clamped(now);
        assert_eq!((v, c), (RTC_MIN_UNIX, true));
        // Absurd values saturate instead of overflowing.
        let (v, c) = RtcOffset { years: i64::MAX, seconds: i64::MIN, ..Default::default() }.apply_clamped(now);
        assert!(c && (RTC_MIN_UNIX..=RTC_MAX_UNIX).contains(&v));
        let (_, c) = RtcOffset { years: -18, ..Default::default() }.apply_clamped(now);
        assert!(!c);
    }

    #[test]
    fn toml_section_parses_and_omits_when_zero() {
        let cfg: MachineConfig = toml::from_str("[rtc_offset]\nyears = -18\nhours = +2\n").unwrap();
        assert_eq!(cfg.rtc_offset, RtcOffset { years: -18, hours: 2, ..Default::default() });
        let out = toml::to_string(&MachineConfig::default()).unwrap();
        assert!(!out.contains("rtc_offset"));
        assert!(toml::from_str::<MachineConfig>("[rtc_offset]\nyear = 1\n").is_err());
        let mut cfg = MachineConfig::default();
        cfg.rtc_offset = RtcOffset { years: -3, seconds: 27, ..Default::default() };
        let text = toml::to_string_pretty(&cfg).unwrap();
        assert_eq!(toml::from_str::<MachineConfig>(&text).unwrap().rtc_offset, cfg.rtc_offset);
    }
}

#[cfg(test)]
mod apply_env_tests {
    use super::set_or_remove_env;

    /// Keys unique to this test, so nothing here races another test's
    /// environment — `std::env` is process-global and the suite runs
    /// threaded.
    const EXTERNAL: &str = "IRIS_TEST_APPLY_ENV_EXTERNAL";
    const OURS: &str = "IRIS_TEST_APPLY_ENV_OURS";

    #[test]
    fn an_empty_config_value_does_not_clobber_the_caller_s_environment() {
        std::env::set_var(EXTERNAL, "l2c");
        set_or_remove_env(EXTERNAL, "");
        assert_eq!(
            std::env::var(EXTERNAL).ok().as_deref(),
            Some("l2c"),
            "a config that does not mention the key must leave the caller's value alone: \
             `IRIS_DEBUG_LOG=l2c iris --config foo.toml` used to silently delete itself",
        );
        std::env::remove_var(EXTERNAL);
    }

    #[test]
    fn a_config_value_still_wins_over_nothing_and_still_clears_itself() {
        std::env::remove_var(OURS);
        set_or_remove_env(OURS, "mips");
        assert_eq!(std::env::var(OURS).ok().as_deref(), Some("mips"));
        // Nothing external set it, so an empty config value clears it.
        std::env::remove_var(OURS);
        set_or_remove_env(OURS, "");
        assert!(std::env::var_os(OURS).is_none());
    }
}
