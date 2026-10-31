use clap::Parser;
use serde::{Deserialize, Serialize};
use std::net::Ipv4Addr;

/// Memory bank sizes in MB; each machine restricts its fitted bank sizes.
pub const VALID_BANK_SIZES: &[u32] = &[0, 8, 16, 32, 64, 128, 256, 512];

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
    /// image at all. Built in; attach it with `kind = "daynaport"`.
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
    /// Which SCSI controller this target lives on: 0 (default) or 1. Indigo2
    /// (IP22 fullhouse) has two independent WD33C93A controllers and the O2
    /// two Adaptec 7880s (0 the internal bus, 1 the external connector);
    /// Indy (IP24 guinness) only has controller 0, so `controller = 1` is
    /// rejected there — see `MachineConfig::validate`.
    #[serde(default)]
    pub controller: u8,
}

/// The `[scsi.N]` key, and the snapshot manifest's disk number, that names
/// `id` on `controller`: the id itself on controller 0, 16 + id on
/// controller 1. Sixteen slots per controller, as SCSI ids go.
pub fn scsi_slot(controller: u8, id: u8) -> u8 {
    controller * 16 + id
}

/// `SCSI ID 1` on controller 0, `SCSI 1:1` on another, for messages.
pub fn scsi_label(controller: u8, id: u8) -> String {
    if controller == 0 { format!("SCSI ID {id}") } else { format!("SCSI {controller}:{id}") }
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

    /// Where the `[scsi.KEY]` table puts this target, as (controller, id).
    /// Keys 1–7 are ids on `controller` (0 unless the key says otherwise);
    /// a key of 16 + id names controller 1 directly, so the same id can be
    /// used on both controllers. See [`scsi_slot`].
    pub fn placement(&self, key: u8) -> (u8, u8) {
        if key >= 16 { (key / 16, key % 16) } else { (self.controller, key) }
    }

    /// The media this target starts with: the active image and, for a
    /// CD-ROM, the changer list built from `path` and `discs` with empty
    /// entries skipped (an empty `path` is a drive with its tray empty, to be
    /// loaded at run time). A disk is just its path.
    pub fn media(&self) -> (String, Vec<String>) {
        if !self.is_cdrom() {
            return (self.path.clone(), vec![]);
        }
        let mut list: Vec<String> = Vec::new();
        if !self.path.is_empty() {
            list.push(self.path.clone());
        }
        for d in &self.discs {
            if !d.is_empty() && !list.contains(d) {
                list.push(d.clone());
            }
        }
        let active = list.first().cloned().unwrap_or_default();
        (active, list)
    }

    /// Where a `--ci` run keeps the copy-on-write overlay of the disk at
    /// `id`: under /tmp and named for the process, so an interactive iris
    /// holding `PATH.overlay` can coexist with any number of CI runs. The
    /// files are kept for post-mortem inspection.
    pub fn ci_overlay_path(id: u8) -> String {
        format!("/tmp/iris-ci-{}-scsi{}.overlay", std::process::id(), id)
    }

    /// For `scratch = true`: make sure the volume exists, creating it with a
    /// minimal SGI volume header and `size_mb` MB when it does not, and
    /// return its path for the CI socket's scratch commands. None when the
    /// target is not a scratch volume, or cannot be one (a CD-ROM or an
    /// overlay is not host-writable in place), which is reported.
    pub fn prepare_scratch(&self, id: u8) -> Option<std::path::PathBuf> {
        if !self.scratch {
            return None;
        }
        if self.is_cdrom() || self.overlay {
            println!("Note: SCSI ID {}: scratch=true is incompatible with cdrom/overlay; ignoring scratch flag", id);
            return None;
        }
        let path = std::path::Path::new(&self.path);
        if !path.exists() {
            let size_mb = self.size_mb.unwrap_or(64) as u64;
            match crate::sgi_vh::create_scratch_image(path, size_mb * 1024 * 1024) {
                Ok(()) => println!("iris: created scratch volume {} ({} MB, with SGI VH)", self.path, size_mb),
                Err(e) => println!("Note: could not create scratch volume {}: {}", self.path, e),
            }
        }
        Some(path.to_path_buf())
    }

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
/// address used throughout `clanker/irix/networking.md` and CI test fixtures.
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
/// (`src/net/nfsudp.rs`) — no external `unfsd`, no host sockets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NfsConfig {
    /// Directory to export over NFS.
    pub shared_dir: String,
    /// NFS protocol version to serve (Auto answers whatever the guest mounts).
    #[serde(default)]
    pub version: crate::net::nfsudp::NfsVersion,
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
    /// No Ethernet: the adapter is not plugged into anything. O2 and Octane
    /// only (`MachineConfig::validate`); the Indy and Indigo2 always have a
    /// network.
    Off,
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
    /// Directory served read-only by the gateway's built-in rsh server (TCP
    /// 514). None (the default) disables it.
    pub rsh_dir: Option<std::path::PathBuf>,
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
            rsh_dir: None,
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
    /// `off` unplugs the Ethernet (O2 only for now).
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
    /// Directory the gateway's built-in rsh server (TCP 514) works on, so the
    /// guest can `rsh <gateway> cat|ls|echo|pwd`. Read-only; nothing is
    /// executed on the host. Unset disables it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rsh_dir: Option<String>,
}

/// Where VINO's video-in capture should come from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VinoSource {
    /// Live host camera capture (built in; selected through `[vino] source`).
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
    /// Use an IMPACT board (`src/dev/mgras/`): an IP28 kernel carries no
    /// Newport driver, so REX3 is not an alternative for that guest.
    Indigo2Ip28,
    /// SGI O2 IP32: CRIME, MACE and GBE, with an R5000 or R10000. Built from
    /// the `[o2]` section, not the fullhouse/Guinness options above; see
    /// `src/ip32_machine.rs`.
    O2Ip32,
    /// SGI Octane IP30: HEART, the Crossbow and the onboard Bridge, with an
    /// R10000. Configured by the shared keys and the `[octane]` section; see
    /// `src/ip30_machine.rs` and `docs/Octane/ip30-address-map.md`.
    OctaneIp30,
    /// Serial-only Fuel IP34 board using the IP35 software platform.
    FuelIp34,
}

impl MachineProfile {
    /// All GUI-selectable profiles, in display order. O2 and Fuel remain
    /// CLI-only until the editor supports their settings. One list serves the
    /// GUI dropdowns (Config tab + New Machine dialog) so they never drift.
    pub const ALL: [Self; 4] = [Self::IndyIp24, Self::Indigo2Ip22, Self::Indigo2Ip28, Self::OctaneIp30];

    /// The bank sizes in MB (0 is an empty bank) that `validate()` accepts
    /// for this profile. The Octane's banks are matched DIMM pairs of 64 to
    /// 512 MB; the IP22/IP24 MC stops at 128 MB.
    pub fn bank_sizes(self) -> &'static [u32] {
        match self {
            Self::OctaneIp30 => &[0, 64, 128, 256, 512],
            Self::FuelIp34 => &[0, 256],
            Self::Indigo2Ip28 | Self::O2Ip32 => VALID_BANK_SIZES,
            _ => &[0, 8, 16, 32, 64, 128],
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::IndyIp24 => "SGI Indy (IP24)",
            Self::Indigo2Ip22 => "SGI Indigo2 (IP22)",
            Self::Indigo2Ip28 => "SGI Indigo2 IMPACT (IP28)",
            Self::O2Ip32 => "SGI O2 (IP32)",
            Self::OctaneIp30 => "SGI Octane (IP30)",
            Self::FuelIp34 => "SGI Fuel (IP34)",
        }
    }

    /// MC/IOC/HPC3 Guinness vs Fullhouse layout. Indy IP24 is Guinness (`true`).
    /// How many SCSI controllers the machine has: two on the Indigo2 (two
    /// WD33C93As) and the O2 (two Adaptec 7880s, the second on the external
    /// connector), one on the Indy.
    pub fn scsi_controllers(self) -> u8 {
        match self {
            Self::Indigo2Ip22 | Self::O2Ip32 | Self::OctaneIp30 | Self::FuelIp34 => 2,
            _ => 1,
        }
    }

    /// Four banks on the MC, HEART and O2, two on Fuel.
    /// `banks` has room for more; unused entries must be zero.
    pub fn memory_banks(self) -> usize {
        if self == Self::FuelIp34 { 2 } else { 4 }
    }

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum GraphicsBoard {
    /// Newport (REX3) — fully emulated. Default.
    #[default]
    #[clap(name = "newport", alias = "xl")]
    Newport,
    /// GR2 XZ (2 GE7) — Indy XZ or Indigo2 XZ (`src/dev/gr2`). In bring-up.
    #[clap(name = "xz", alias = "gr2", alias = "gr2_xz", alias = "gr2-xz")]
    Xz,
    /// GR2 Extreme (8 GE7) — Indigo2 only (`src/dev/gr2`). In bring-up.
    #[clap(name = "extreme", alias = "gr2_extreme", alias = "gr2-extreme")]
    Extreme,
    /// Solid IMPACT (1 GE11, 1 RE4).
    #[clap(name = "solidimpact", alias = "impact:solid", alias = "solid_impact")]
    SolidImpact,
    /// High IMPACT (2 GE11, 1 RE4).
    #[clap(name = "highimpact", alias = "impact:high", alias = "high_impact")]
    HighImpact,
    /// Maximum IMPACT (2 GE11, 2 RE4).
    #[clap(name = "maximpact", alias = "impact:max", alias = "max_impact")]
    MaxImpact,
    /// No graphics board: the machine runs on its serial console. Octane
    /// and Fuel.
    #[clap(name = "none")]
    None,
}

impl GraphicsBoard {
    pub const ALL: [Self; 6] = [
        Self::Newport,
        Self::Xz,
        Self::Extreme,
        Self::SolidImpact,
        Self::HighImpact,
        Self::MaxImpact,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Newport => "Newport (XL)",
            Self::Xz => "GR2 XZ",
            Self::Extreme => "GR2 Extreme",
            Self::SolidImpact => "IMPACT Solid",
            Self::HighImpact => "IMPACT High",
            Self::MaxImpact => "IMPACT Maximum",
            Self::None => "None",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Newport => "newport",
            Self::Xz => "xz",
            Self::Extreme => "extreme",
            Self::SolidImpact => "solidimpact",
            Self::HighImpact => "highimpact",
            Self::MaxImpact => "maximpact",
            Self::None => "none",
        }
    }

    pub fn is_impact(self) -> bool {
        matches!(self, Self::SolidImpact | Self::HighImpact | Self::MaxImpact)
    }

    /// Whether `validate()` accepts this board on `profile`. The Octane
    /// models only IMPACT (High as the SI with texture option, Maximum as
    /// the MXI) or no board.
    pub fn supports(self, profile: MachineProfile) -> bool {
        match profile {
            MachineProfile::OctaneIp30 => self.is_impact() || self == Self::None,
            MachineProfile::FuelIp34 => self == Self::None,
            MachineProfile::Indigo2Ip22 => self != Self::None,
            _ => !matches!(self, Self::Extreme | Self::None),
        }
    }

    /// The board a machine gets when `[graphics] board` is not set.
    pub fn default_for(profile: MachineProfile) -> Self {
        match profile {
            MachineProfile::OctaneIp30 => Self::SolidImpact,
            MachineProfile::FuelIp34 => Self::None,
            _ => Self::Newport,
        }
    }
}

impl std::str::FromStr for GraphicsBoard {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "newport" | "xl" => Ok(Self::Newport),
            "xz" | "gr2" | "gr2_xz" | "gr2-xz" => Ok(Self::Xz),
            "extreme" | "gr2_extreme" | "gr2-extreme" => Ok(Self::Extreme),
            "solidimpact" | "impact:solid" | "solid_impact" | "solid-impact" | "solid" => Ok(Self::SolidImpact),
            "highimpact" | "impact:high" | "high_impact" | "high-impact" | "high" => Ok(Self::HighImpact),
            "maximpact" | "impact:max" | "max_impact" | "max-impact" | "maximumimpact" | "maximum_impact" | "maximum-impact" | "max" => Ok(Self::MaxImpact),
            "none" => Ok(Self::None),
            _ => Err(format!(
                "invalid graphics board \"{s}\" (valid: newport, xz, extreme, solidimpact, highimpact, maximpact, impact:solid, impact:high, impact:max, none)"
            )),
        }
    }
}

impl Serialize for GraphicsBoard {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for GraphicsBoard {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse::<GraphicsBoard>().map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for GraphicsBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// `[graphics]` section — Newport head count and display options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphicsSection {
    /// Graphics board: Newport, GR2 XZ/Extreme, IMPACT Solid/High/Maximum,
    /// or none (Octane and Fuel). Unset: the machine's default
    /// ([`GraphicsBoard::default_for`]); read it with
    /// [`MachineConfig::graphics_board`]. Extreme is restricted to IP22,
    /// the Octane to IMPACT or none; non-Newport boards require one head
    /// and Guest resolution. See [`MachineConfig::validate`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub board: Option<GraphicsBoard>,
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
            board: None,
            heads: default_graphics_heads(),
            resolution: crate::dev::ng1::vc2_timings::NewportResolution::default(),
        }
    }
}

/// Physical shape of the host keyboard: which keys it has, not what they print.
/// The guest applies the character layout itself (PROM `keybd`, IRIX X11
/// keymaps); this only matters because the host reports the US `\|` key
/// (ANSI, above Enter) and the key left of Enter on ISO/JIS boards (`#~` on
/// UK, `]}` on JIS) as the same `Backslash` keycode, while an SGI keyboard
/// sends set-3 `0x5C` for the first and `0x53` for the second
/// (xkeyboard-config `sgi_vndr/indy`: pc101 `<BKSL>` = 100, pc102 `<BKSL>` and
/// jp106 `<AC12>` = 91; X keycode = set-3 code + 8).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
#[clap(rename_all = "lower")]
pub enum KeyboardLayout {
    /// US-style: `\|` above Enter.
    #[default]
    Ansi,
    /// European ISO: extra key left of Z, `#~`-style key left of Enter.
    Iso,
    /// Japanese JIS 106/109: `]}` left of Enter, plus `¥`, `ろ` and the
    /// conversion keys.
    Jis,
}

impl KeyboardLayout {
    pub const ALL: [Self; 3] = [Self::Ansi, Self::Iso, Self::Jis];

    pub fn label(self) -> &'static str {
        match self {
            Self::Ansi => "ANSI (US)",
            Self::Iso => "ISO (European)",
            Self::Jis => "JIS (Japanese)",
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
    /// the Indigo2 IMPACT (IP28). The machine uses shadow cache arrays;
    /// see `docs/indigo2-ip28.md` and `src/cpu/mips_cache_shadow.rs`.
    R10000,
    /// MIPS R12000, an R10000-family part: the Octane2's 360 MHz R12000 with
    /// 2 MiB of secondary cache, or the O2's 300 MHz part with 1 MiB.
    R12000,
    /// MIPS R14000, an R10000-family part: the Fuel's, and the Octane's
    /// revision 2.4 with 2 MiB of secondary cache.
    R14000,
}

impl CpuModel {
    pub const ALL: [Self; 5] = [Self::R4400, Self::R5000, Self::R10000, Self::R12000, Self::R14000];

    pub fn label(self) -> &'static str {
        match self {
            Self::R4400 => "MIPS R4400",
            Self::R5000 => "MIPS R5000",
            Self::R10000 => "MIPS R10000",
            Self::R12000 => "MIPS R12000",
            Self::R14000 => "MIPS R14000",
        }
    }

    /// Whether `profile` shipped with this part. The R12000 and R14000 are
    /// limited to the machines a source records them in (origami's datasheets):
    /// the Octane2's R12000 (`ip30-r12000.md`), a dual-R14000 Octane
    /// (`r14000.md`, IP30 identity and cache review), the O2's 300 MHz R12000
    /// (`o2-hardware-findings.md`) and the Fuel's R14000 (`ip34-fuel.md`). No
    /// source has a Fuel with another part. The older parts keep the choices
    /// the machines already had.
    pub fn fits(self, profile: MachineProfile) -> bool {
        match self {
            Self::R12000 => matches!(profile, MachineProfile::OctaneIp30 | MachineProfile::O2Ip32),
            Self::R14000 => matches!(profile, MachineProfile::OctaneIp30 | MachineProfile::FuelIp34),
            _ => true,
        }
    }
}

/// How the CPU's caches are modelled. A runtime choice: each is its own
/// monomorphisation, picked when the machine is built.
///
/// - `full`: the functional caches (`mips_cache_v2`): tags, line data, LRU,
///   write-back, VCE. What the interpreter is fastest with, since its decoded
///   instructions live in the cache lines.
/// - `shadow`: memory is the only store; tag arrays exist only to answer
///   CACHE instructions (`mips_cache_shadow`). jitv2's inline loads and stores
///   need no tag check against it.
/// - `auto`: `shadow` in a `jitv2` build, `full` otherwise.
///
/// The R10000 (IP28) only has the shadow model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
#[clap(rename_all = "lower")]
pub enum CacheModel {
    #[default]
    Auto,
    Full,
    Shadow,
}

impl CacheModel {
    /// Whether `cpu` runs the shadow cache under this setting.
    pub fn is_shadow(self, cpu: CpuModel) -> bool {
        match (cpu, self) {
            (CpuModel::R10000 | CpuModel::R12000 | CpuModel::R14000, _) => true,
            (_, Self::Full) => false,
            (_, Self::Shadow) => true,
            (_, Self::Auto) => cfg!(feature = "jitv2"),
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
    #[serde(default)]
    pub cache: CacheModel,
    /// Processors fitted. More than one only on the Octane.
    #[serde(default = "default_cpus")]
    pub cpus: u8,
}

fn default_cpus() -> u8 {
    1
}

impl Default for MachineSection {
    fn default() -> Self {
        Self { profile: MachineProfile::default(), cpu: CpuModel::default(), cache: CacheModel::default(), cpus: default_cpus() }
    }
}

/// What only an O2 has (`[o2]` section). The rest of an O2 is the shared
/// keys: `prom` (its PROM image, not embedded), `nvram` (the PROM's flash,
/// which holds its environment), `banks` (the kit in each DIMM pair),
/// `serial_port_a`/`serial_port_b`/`serial_log`, `headless` (a serial
/// console instead of the graphics console), `[clock] fixed_mhz` (CP0 Count,
/// half the CPU clock), `[network]`, `[[port_forward]]` and `[scsi.N]`. The
/// deprecated keys below are the old O2-only spellings of those:
/// [`MachineConfig::fold_deprecated`] moves them over when a file is loaded
/// and they are never written back.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct O2Section {
    /// A PS/2 mouse on the second port (with the graphics console).
    #[serde(default = "o2_on")]
    pub mouse: bool,
    /// A TCP port on 127.0.0.1 for driving the machine from a script: screen
    /// shots, keys, the mouse (see `src/ip32_machine.rs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_port: Option<u16>,
    /// Test aid, not hardware behaviour: lend a kernel the FPU while it is
    /// inside the PROM (NetBSD prints through the PROM's graphics console
    /// with CU1 clear, and the PROM scrolls with FPU loads).
    #[serde(default, skip_serializing_if = "is_false")]
    pub arcs_fpu: bool,

    /// Deprecated: `prom`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prom: Option<String>,
    /// Deprecated: `nvram`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flash: Option<String>,
    /// Deprecated: `banks`, fitted largest kits first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ram_mb: Option<u32>,
    /// Deprecated: `banks`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimms: Option<Vec<u32>>,
    /// Deprecated: `[scsi.N]` with `controller` = bus.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disks: Vec<O2Disk>,
    /// Deprecated: `serial_port_a`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serial_port: Option<u16>,
    /// Deprecated: `serial_log`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serial_log: Option<String>,
    /// Deprecated: `headless` (its inverse).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graphics: Option<bool>,
    /// Deprecated: `[clock] fixed_mhz`, which is half of it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_mhz: Option<u32>,
    /// Deprecated: `[network] mode = "off"` when false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub net: Option<bool>,
    /// Deprecated: `[[port_forward]]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forwards: Vec<PortForwardConfig>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// What only an Octane has (`[octane]` section), used with
/// `machine.profile = "octane_ip30"`. The rest is the shared keys: `prom`
/// (the 1 MB flash image; IRIX's `IP30prom.bin` is tried when it is left
/// unset), `nvram` (the DS1685's RAM), `banks` (MB in each HEART bank),
/// `[scsi.N]` (`controller` 0 internal, 1 external), `[network]`,
/// `serial_port_a`/`serial_port_b`/`serial_log` and `[clock] fixed_mhz`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OctaneSection {
    /// A file holding the boot flash, so what the PROM writes there (its
    /// persistent records) survives from run to run. Created from `prom`
    /// when missing; an existing file is used as it stands. Unset: the
    /// flash lives in memory only and every start is from `prom`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flash: Option<String>,
    /// Retired: the board is `[graphics] board` on every machine. Kept
    /// only so an old config gets an error naming the new key.
    #[serde(default, skip_serializing)]
    pub graphics: Option<String>,
}

impl OctaneSection {
    /// Left out of a written config when nothing in it was set.
    fn is_default(&self) -> bool {
        self.flash.is_none()
    }
}

/// An entry of the deprecated `o2.disks` list; see [`O2Section::disks`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct O2Disk {
    pub id: u8,
    /// The controller: 0 (internal, ids 1-4 under IRIX) or 1 (external).
    #[serde(default)]
    pub bus: u8,
    pub path: String,
    #[serde(default)]
    pub cdrom: bool,
    /// Copy-on-write, as `[scsi.N] overlay`: the image is never written,
    /// writes go to `{path}.overlay`, and a snapshot carries them, so its
    /// disk matches its memory.
    #[serde(default)]
    pub overlay: bool,
}

fn o2_on() -> bool {
    true
}

impl Default for O2Section {
    fn default() -> Self {
        Self {
            mouse: true,
            control_port: None,
            arcs_fpu: false,
            prom: None,
            flash: None,
            ram_mb: None,
            dimms: None,
            disks: Vec::new(),
            serial_port: None,
            serial_log: None,
            graphics: None,
            cpu_mhz: None,
            net: None,
            forwards: Vec::new(),
        }
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
    /// Publish enabled/nonempty `[debug]` values into the environment.
    /// Default false/blank values preserve variables the caller already set;
    /// explicit nonempty values overwrite the corresponding variable.
    ///
    /// The caller's comment has always promised "env vars still override if
    /// set externally" and the code did the opposite: a config with no
    /// `debug_log` key called `remove_var`, so `IRIS_DEBUG_LOG=l2c ./iris
    /// --config foo.toml` silently deleted its own setting and produced no
    /// logging at all — devlog looked dead when only its bootstrap was.
    ///
    /// This governs all three `[debug]` keys, not just `debug_log`: a caller
    /// who sets `IRIS_NO_IDLE` or `IRIS_GUI_GL` keeps it when the corresponding
    /// config value is false; an enabled setting publishes `1`.
    pub fn apply_env(&self) {
        set_or_remove_env("IRIS_NO_IDLE", if self.no_idle { "1" } else { "" });
        set_or_remove_env("IRIS_GUI_GL", if self.gui_gl_capture { "1" } else { "" });
        set_or_remove_env("IRIS_DEBUG_LOG", &self.debug_log);
    }
}

/// Publish nonempty config values; preserve existing env values for blank config.
///
/// An empty config value means "the config does not mention this", which is
/// not the same as "unset it": an existing variable is left alone regardless
/// of who originally set it. A nonempty config value replaces it.
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
/// persistent on-disk code cache (`src/cpu/jitv2/pcache.rs`), applied to the
/// process environment at `Start` the same way `DebugConfig` applies
/// `[debug]` (see `Jitv2Config::apply_env`). `cache`/`cache_dir` used to be
/// the undocumented `IRIS_JIT_CACHE`/`IRIS_JIT_CACHE_DIR` env vars; those
/// still work when config is false/blank. Explicit enabled/nonempty config
/// values replace them, as with `[debug]`.
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
    /// Machine::new). False/blank config preserves existing environment values;
    /// enabled/nonempty config replaces them, as in `DebugConfig::apply_env`.
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
    /// runtime inference. None selects the profile default: 33 MHz on
    /// IP22/IP24, 97.5 MHz on IP28, 90 MHz on the O2 (a 180 MHz CPU). IRIX
    /// reports twice the Count rate as CPU MHz, and the O2's PROM measures
    /// it the same way; host emulation throughput is independent of this
    /// setting.
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

/// Which host audio system HAL2 plays through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum AudioHost {
    /// The platform's usual one: CoreAudio on macOS, WASAPI on Windows, ALSA
    /// on Linux.
    #[default]
    Default,
    /// ALSA (Linux).
    Alsa,
    /// A PulseAudio server, or PipeWire's pipewire-pulse (Linux). Falls back
    /// to ALSA when none answers.
    Pulseaudio,
}

/// HAL2 / cpal audio output tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Host audio system to play through. Default: the platform's usual one.
    #[serde(default)]
    pub host: AudioHost,
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
        Self { host: AudioHost::Default, prebuf_ms: default_audio_prebuf_ms(), cpal_buffer_frames: None }
    }
}

/// The built-in VNC server (`[vnc]` section). Off unless `listen` is set.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct VncConfig {
    /// Address to serve on, `host:port` (a bare port means 127.0.0.1).
    /// Without a password, keep it on loopback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub listen: Option<String>,
    /// Most frames a second sent to a viewer (default 30).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_fps: Option<u32>,
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

/// Fuel's independent serial connections. The selected console serves CI and serial_log.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FuelConsole {
    #[default]
    L1,
    Ioc3A,
    Ioc3B,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FuelSection {
    /// Persistent 16 MiB boot flash store; absent means volatile flash.
    #[serde(default)]
    pub flash: Option<String>,
    #[serde(default)]
    pub console: FuelConsole,
    /// L1 host console TCP port; IOC3 keeps serial_port_a and serial_port_b.
    #[serde(default)]
    pub l1_port: Option<u16>,
    /// Separate L1 log when an IOC3 port is selected as console.
    #[serde(default)]
    pub l1_log: Option<String>,
    /// Fit the PS/2 keyboard and mouse controller in the onboard IOC3 and
    /// feed it the host's keyboard and mouse.
    #[serde(default = "default_fuel_input")]
    pub ps2_input: bool,
    /// Plug a USB keyboard and mouse into the OHCI root ports and feed them
    /// the same host input. Off by default: the Fuel's manual and the PROM
    /// treat PS/2 as the standard keyboard and mouse.
    #[serde(default)]
    pub usb_input: bool,
}

fn default_fuel_input() -> bool { true }

impl Default for FuelSection {
    fn default() -> Self {
        Self {
            flash: None,
            console: FuelConsole::default(),
            l1_port: None,
            l1_log: None,
            ps2_input: true,
            usb_input: false,
        }
    }
}

impl FuelSection {
    /// Left out of a written config when nothing in it was set.
    fn is_default(&self) -> bool {
        self.flash.is_none()
            && self.console == FuelConsole::default()
            && self.l1_port.is_none()
            && self.l1_log.is_none()
            && self.ps2_input
            && !self.usb_input
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

    /// Path to the Indigo2 (IP22/IP28) motherboard EEPROM file (93CS56 — NVRAM
    /// env vars + MAC, see `nveeprom` monitor command). Loaded at startup
    /// if the file exists; saved on Stop and by `nveeprom save` by default.
    /// Ignored on Indy (no such chip). Per-config files avoid the same
    /// cross-install footgun as `nvram`.
    #[serde(default = "default_nveeprom")]
    pub nveeprom: String,

    /// RAM bank sizes in MB, one per bank, up to [`MAX_BANKS`]. Every machine
    /// today has four (`MachineProfile::memory_banks`); the rest must be 0.
    /// On the Indy and Indigo2 a bank is 0 (absent), 8, 16, 32, 64 or 128 MB
    /// (the IP28 also 256 or 512); on the O2 it is the kit in a DIMM pair, 32,
    /// 64, 128 or 256 MB, largest first from pair A. A config may list fewer
    /// than eight; the rest are 0, and a config is written back with the
    /// trailing zeros beyond the fourth trimmed.
    #[serde(default = "default_banks", with = "banks_serde")]
    pub banks: [u32; MAX_BANKS],

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

    /// TCP port of the monitor console on 127.0.0.1; None means the usual
    /// 8888 (`crate::monitor::DEFAULT_PORT`). Two iris processes running at
    /// once need different ports: the second can't bind the first's, and
    /// then has no monitor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monitor_port: Option<u16>,

    /// TCP ports of the serial channels on 127.0.0.1: A (tty2) and B (tty1,
    /// the console). None means 8880 and 8881
    /// (`crate::dev::z85c30::DEFAULT_PORTS`); like the monitor's, they have to
    /// differ between iris processes running at once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serial_port_a: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serial_port_b: Option<u16>,

    /// Physical host keyboard: `ansi` (default), `iso` or `jis`. Set `jis`
    /// with a Japanese keyboard and the PROM's `keybd` set to Japanese. See
    /// [`KeyboardLayout`].
    #[serde(default, skip_serializing_if = "is_default_keyboard")]
    pub keyboard: KeyboardLayout,

    /// If Some(path), load this static ELF32 MSB binary into RAM at startup and
    /// set PC to its entry point (bare-metal test binaries; see --load-elf).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_elf: Option<String>,

    /// Start a Perfetto trace into this file at startup (`--perfetto`;
    /// needs the `perfetto` feature). Categories and clock as for the
    /// monitor's `perfetto start`.
    #[serde(skip)]
    pub perfetto: Option<String>,
    #[serde(skip)]
    pub perfetto_cats: Option<String>,
    #[serde(skip)]
    pub perfetto_clock: Option<String>,

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
    /// fidelity shortcuts. Hides the host window unless ci_display is set;
    /// offscreen graphics remain active unless headless is explicitly set.
    #[serde(default)]
    pub ci: bool,

    /// Unix socket path or TCP host:port for CI control. Used only when `ci` is true.
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

    /// Prefix each completed line of `serial_log` with a UTC ISO-8601 stamp.
    /// `off` (the default) writes raw bytes, byte-for-byte a transcript.
    #[serde(default)]
    pub serial_log_timestamps: crate::dev::serial_log::SerialLogTimestamps,

    // --- table / array-of-table fields: must be emitted after all scalars ---

    /// SCSI devices keyed by ID 1–7. Missing IDs are not attached.
    #[serde(default = "default_scsi", with = "scsi_keys")]
    pub scsi: std::collections::HashMap<u8, ScsiDeviceConfig>,

    /// NFS share configuration. If present, the NAT's in-process NFS server
    /// (`src/net/nfsudp.rs`) exports `shared_dir` to the guest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nfs: Option<NfsConfig>,

    /// Port forwarding rules (host port → guest port).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub port_forward: Vec<PortForwardConfig>,

    /// VINO video-in configuration (IndyCam emulation source).
    #[serde(default)]
    pub vino: VinoConfig,

    /// Built-in VNC server (`[vnc]` section).
    #[serde(default)]
    pub vnc: VncConfig,

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

    /// Misc debug/capture runtime tuning (`[debug]` section).
    #[serde(default)]
    pub debug: DebugConfig,

    /// jitv2 compile-pool tuning (`[jitv2]` section).
    #[serde(default)]
    pub jitv2: Jitv2Config,

    /// The O2 (`[o2]` section), used when `[machine] profile = "o2_ip32"`.
    #[serde(default)]
    pub o2: O2Section,

    /// The Octane (`[octane]` section), used when `[machine] profile = "octane_ip30"`.
    #[serde(default, skip_serializing_if = "OctaneSection::is_default")]
    pub octane: OctaneSection,

    /// Serial console and flash storage for the Fuel.
    #[serde(default, skip_serializing_if = "FuelSection::is_default")]
    pub fuel: FuelSection,

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

/// `prom` when a config does not set it. The O2 then tries
/// `ip32_machine::O2_PROM_FILE` as well, having no embedded PROM.
pub const DEFAULT_PROM: &str = "prom.bin";
/// `nvram` when a config does not set it: the DS1386's file on the Indy, the
/// PROM's flash on the O2, the DS1685's RAM on the Octane.
pub const DEFAULT_NVRAM: &str = "nvram.bin";

fn default_prom() -> String {
    DEFAULT_PROM.to_string()
}

fn default_nvram() -> String {
    DEFAULT_NVRAM.to_string()
}

fn default_nveeprom() -> String {
    "nveeprom.bin".to_string()
}

/// Room in `banks`: eight, for machines with more than today's four.
pub const MAX_BANKS: usize = 8;

fn default_banks() -> [u32; MAX_BANKS] {
    banks_from4([128, 128, 0, 0])
}

/// Four bank sizes as a full `banks` array.
pub fn banks_from4(four: [u32; 4]) -> [u32; MAX_BANKS] {
    let mut banks = [0u32; MAX_BANKS];
    banks[..4].copy_from_slice(&four);
    banks
}

/// `banks` as TOML: any length from 1 to [`MAX_BANKS`] is read and padded
/// with zeros; it is written with at least four entries and no trailing
/// zeros beyond them, so a four-bank machine's config looks as it always has.
mod banks_serde {
    use super::MAX_BANKS;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(banks: &[u32; MAX_BANKS], ser: S) -> Result<S::Ok, S::Error> {
        let n = banks.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1).max(4);
        banks[..n].serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<[u32; MAX_BANKS], D::Error> {
        let list = Vec::<u32>::deserialize(de)?;
        if list.is_empty() || list.len() > MAX_BANKS {
            return Err(serde::de::Error::custom(format!(
                "banks: {} entries; give 1 to {MAX_BANKS} bank sizes in MB", list.len()
            )));
        }
        let mut banks = [0u32; MAX_BANKS];
        banks[..list.len()].copy_from_slice(&list);
        Ok(banks)
    }
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
            monitor_port: None,
            serial_port_a: None,
            serial_port_b: None,
            keyboard: KeyboardLayout::default(),
            load_elf: None,
            perfetto: None,
            perfetto_cats: None,
            perfetto_clock: None,
            test_device: false,
            test_device_dump: None,
            cheritest_dump_hook: false,
            nat_subnet: None,
            ci: false,
            ci_socket: default_ci_socket(),
            ci_display: false,
            serial_log: None,
            serial_log_timestamps: Default::default(),
            vino: VinoConfig::default(),
            vnc: VncConfig::default(),
            network: NetworkSection::default(),
            mouse_scroll_pixels_per_line: default_scroll_pixels_per_line(),
            lock_aspect_ratio: default_lock_aspect_ratio(),
            scsi_deferred_int: default_scsi_deferred_int(),
            audio: AudioConfig::default(),
            machine: MachineSection::default(),
            graphics: GraphicsSection::default(),
            debug: DebugConfig::default(),
            jitv2: Jitv2Config::default(),
            o2: O2Section::default(),
            octane: OctaneSection::default(),
            fuel: FuelSection::default(),
            perf: PerfConfig::default(),
            clock: ClockConfig::default(),
            rtc_offset: RtcOffset::default(),
            ultra64: Ultra64Config::default(),
        }
    }
}


fn is_default_keyboard(k: &KeyboardLayout) -> bool { *k == KeyboardLayout::default() }

impl MachineConfig {
    /// The serial channels' TCP ports (A, B), defaults filled in.
    pub fn serial_ports(&self) -> [u16; 2] {
        let [a, b] = crate::dev::z85c30::DEFAULT_PORTS;
        [self.serial_port_a.unwrap_or(a), self.serial_port_b.unwrap_or(b)]
    }

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
            Ok(mut cfg) => {
                cfg.fold_deprecated();
                cfg
            }
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

    /// Translate keys an older iris wrote into their current form, saying so
    /// once. The old `[o2]` spellings move to the shared keys they stand for
    /// when those are still at their defaults (a shared key that is set wins);
    /// `o2.disks` becomes `[scsi.N]` entries with `controller` = bus and
    /// replaces the `scsi` map, since a config that has it never meant the
    /// Indy's default disks.
    pub fn fold_deprecated(&mut self) {
        let mut moved: Vec<&str> = Vec::new();
        if let Some(prom) = self.o2.prom.take() {
            if self.prom == DEFAULT_PROM { self.prom = prom; }
            moved.push("o2.prom -> prom");
        }
        if let Some(flash) = self.o2.flash.take() {
            if self.nvram == DEFAULT_NVRAM { self.nvram = flash; }
            moved.push("o2.flash -> nvram");
        }
        if let Some(mb) = self.o2.ram_mb.take() {
            match crate::dev::crime_mem::kits_for(mb) {
                Ok(kits) if self.banks == default_banks() => {
                    self.banks = [0; MAX_BANKS];
                    self.banks[..kits.len()].copy_from_slice(&kits);
                }
                Ok(_) => {}
                Err(e) => eprintln!("iris: o2.ram_mb = {mb}: {e}; ignored"),
            }
            moved.push("o2.ram_mb -> banks");
        }
        if let Some(kits) = self.o2.dimms.take() {
            if self.banks == default_banks() && !kits.is_empty() && kits.len() <= MAX_BANKS {
                self.banks = [0; MAX_BANKS];
                self.banks[..kits.len()].copy_from_slice(&kits);
            }
            moved.push("o2.dimms -> banks");
        }
        if !self.o2.disks.is_empty() {
            self.scsi.clear();
            for d in std::mem::take(&mut self.o2.disks) {
                self.scsi.insert(d.id, ScsiDeviceConfig {
                    path: d.path,
                    cdrom: d.cdrom,
                    overlay: d.overlay,
                    controller: d.bus,
                    ..Default::default()
                });
            }
            moved.push("o2.disks -> [scsi.N] (controller = bus)");
        }
        if let Some(port) = self.o2.serial_port.take() {
            if self.serial_port_a.is_none() { self.serial_port_a = Some(port); }
            moved.push("o2.serial_port -> serial_port_a");
        }
        if let Some(log) = self.o2.serial_log.take() {
            if self.serial_log.is_none() { self.serial_log = Some(log); }
            moved.push("o2.serial_log -> serial_log");
        }
        if let Some(graphics) = self.o2.graphics.take() {
            if !graphics { self.headless = true; }
            moved.push("o2.graphics -> headless (inverted)");
        }
        if let Some(mhz) = self.o2.cpu_mhz.take() {
            if self.clock.fixed_mhz.is_none() { self.clock.fixed_mhz = Some(mhz as f64 / 2.0); }
            moved.push("o2.cpu_mhz -> [clock] fixed_mhz (half: CP0 Count)");
        }
        if let Some(net) = self.o2.net.take() {
            if !net { self.network.mode = NetMode::Off; }
            moved.push("o2.net -> [network] mode = \"off\"");
        }
        if !self.o2.forwards.is_empty() {
            let forwards = std::mem::take(&mut self.o2.forwards);
            self.port_forward.extend(forwards);
            moved.push("o2.forwards -> [[port_forward]]");
        }
        if !moved.is_empty() {
            eprintln!(
                "iris: deprecated [o2] keys folded into the shared settings ({}); move them in the config to silence this",
                moved.join(", ")
            );
        }
    }

    /// The O2's memory as the kit in each DIMM pair from pair A: `banks` with
    /// the trailing zeros dropped.
    pub fn o2_kits(&self) -> Vec<u32> {
        let n = self.banks.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        self.banks[..n].to_vec()
    }

    /// Validate bank sizes, returns a description of any errors.
    /// The graphics board: `[graphics] board`, or the machine's default
    /// (Solid IMPACT on Octane, none on Fuel, Newport elsewhere).
    pub fn graphics_board(&self) -> GraphicsBoard {
        self.graphics.board.unwrap_or_else(|| GraphicsBoard::default_for(self.machine.profile))
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.machine.cpu.fits(self.machine.profile) {
            return Err(format!("{} is not a processor the {:?} shipped with", self.machine.cpu.label(), self.machine.profile));
        }
        if self.machine.cpus != 1 && self.machine.profile != MachineProfile::OctaneIp30 {
            return Err(format!("[machine] cpus = {}: only the Octane has more than one processor", self.machine.cpus));
        }
        if self.graphics.heads != 1 && self.graphics.heads != 2 {
            return Err(format!(
                "graphics.heads {} is invalid (valid: 1, 2)",
                self.graphics.heads
            ));
        }
        let board = self.graphics_board();
        if board != GraphicsBoard::Newport {
            let name = board.as_str();
            if board == GraphicsBoard::Extreme && self.machine.profile != MachineProfile::Indigo2Ip22 {
                return Err(
                    "graphics.board \"extreme\" is only valid on Indigo2 (machine.profile = indigo2_ip22)".into(),
                );
            }
            if !board.supports(self.machine.profile) {
                return Err(if self.machine.profile == MachineProfile::OctaneIp30 {
                    format!("graphics.board \"{name}\" is not modelled on the Octane (valid: solidimpact, highimpact, maximpact, none)")
                } else {
                    format!("graphics.board \"{name}\" is only valid on the Octane")
                });
            }
            if self.graphics.heads != 1 {
                return Err(format!("graphics.board \"{name}\" does not support dual-head (graphics.heads must be 1)"));
            }
            if !self.graphics.resolution.is_guest() {
                return Err("graphics.resolution presets require Newport (graphics.board = newport)".into());
            }
        }
        let mut placed: std::collections::HashMap<(u8, u8), u8> = std::collections::HashMap::new();
        let mut keys: Vec<u8> = self.scsi.keys().copied().collect();
        keys.sort();
        for key in keys {
            let dev = &self.scsi[&key];
            let (controller, id) = dev.placement(key);
            let max_id = if self.machine.profile == MachineProfile::FuelIp34 { 15 } else { 7 };
            if id == 0 || id > max_id {
                return Err(format!("scsi.{}: SCSI ID {} is out of range (1–{}; the controller itself is 0)", key, id, max_id));
            }
            if key >= 16 && dev.controller != 0 && dev.controller != controller {
                return Err(format!(
                    "scsi.{}: the key already puts the target on controller {}; controller = {} contradicts it",
                    key, controller, dev.controller
                ));
            }
            if controller > 1 {
                return Err(format!("scsi.{}: controller {} is invalid (valid: 0, 1)", key, controller));
            }
            if controller != 0 && self.machine.profile.scsi_controllers() < 2 {
                return Err(format!(
                    "scsi.{}: controller {} is only valid on Indigo2 (machine.profile = indigo2_ip22) and O2 (o2_ip32), which have two SCSI controllers — this machine has one (0)",
                    key, controller
                ));
            }
            if let Some(other) = placed.insert((controller, id), key) {
                return Err(format!(
                    "scsi.{} and scsi.{} both name {}", other, key, scsi_label(controller, id)
                ));
            }
            if dev.is_daynaport() && matches!(self.machine.profile, MachineProfile::O2Ip32 | MachineProfile::FuelIp34) {
                return Err(format!(
                    "scsi.{}: kind = \"daynaport\" is not available on {} (its Ethernet is built in)", key, self.machine.profile.label()
                ));
            }
        }
        if self.scale < 1 || self.scale > 4 {
            return Err(format!("scale {} is invalid (valid: 1, 2, 3, 4)", self.scale));
        }
        let bank_count = self.machine.profile.memory_banks();
        for (i, &sz) in self.banks.iter().enumerate().skip(bank_count) {
            if sz != 0 {
                return Err(format!(
                    "bank{} size {} MB: {:?} has {} memory banks, so banks beyond the {}th must be 0",
                    i, sz, self.machine.profile, bank_count, bank_count
                ));
            }
        }
        if self.machine.profile == MachineProfile::O2Ip32 {
            // A bank is the kit in a DIMM pair, filled from pair A.
            if let Some(gap) = self.banks[..bank_count].iter().position(|&b| b == 0) {
                if self.banks[gap..bank_count].iter().any(|&b| b != 0) {
                    return Err(format!("banks: DIMM pair {} is empty but a later pair is populated; fill the pairs from A", (b'A' + gap as u8) as char));
                }
            }
            crate::dev::crime_mem::check_kits(&self.o2_kits()).map_err(|e| format!("banks: {e}"))?;
        } else if self.machine.profile == MachineProfile::FuelIp34 {
            // The machine always models its one R14000; `r10000` names the
            // family, as the key did before the R14000 could be named, and a
            // config cannot claim another processor.
            if !matches!(self.machine.cpu, CpuModel::R10000 | CpuModel::R14000) {
                return Err("a Fuel has an R14000: set [machine] cpu = \"r14000\"".into());
            }
            if self.banks[..2] != [256, 256] {
                return Err("Fuel currently requires banks = [256, 256] (512 MiB)".into());
            }
            if board != GraphicsBoard::None {
                return Err("Fuel currently supports graphics.board = \"none\" only".into());
            }
        } else if self.machine.profile == MachineProfile::OctaneIp30 {
            // A HEART bank is a matched DIMM pair: 64 to 512 MB, filled from
            // bank 0 without a gap, 2 GB at most (the Octane owner's guide).
            let banks = &self.banks[..bank_count];
            if banks[0] == 0 {
                return Err("banks: an Octane needs memory in bank 0".into());
            }
            if let Some(gap) = banks.iter().position(|&b| b == 0) {
                if banks[gap..].iter().any(|&b| b != 0) {
                    return Err(format!("banks: bank {gap} is empty but a later bank is populated; fill the banks from 0"));
                }
            }
            for (i, &sz) in banks.iter().enumerate() {
                if sz != 0 && ![64, 128, 256, 512].contains(&sz) {
                    return Err(format!("bank{i} size {sz} MB is invalid for an Octane (valid: 64, 128, 256, 512)"));
                }
            }
            if let Some(old) = &self.octane.graphics {
                return Err(format!(
                    "octane.graphics has moved: set the board in [graphics] as board = \"{}\"",
                    old.trim()
                ));
            }
            if self.graphics.board == Some(GraphicsBoard::Newport) {
                return Err("graphics.board \"newport\" is not modelled on the Octane (valid: solidimpact, highimpact, maximpact, none)".into());
            }
        } else {
            for (i, &sz) in self.banks.iter().enumerate().take(bank_count) {
                if !VALID_BANK_SIZES.contains(&sz) {
                    return Err(format!(
                        "bank{} size {} MB is invalid (valid: {:?})",
                        i, sz, VALID_BANK_SIZES
                    ));
                }
                // The IP22/IP24 MC cannot express banks larger than 128 MB at its base
                // shift; only the IP28's can.
                if sz > 128 && !self.machine.profile.ip28() {
                    return Err(format!(
                        "bank{} size {} MB needs the IP28 (machine.profile = indigo2_ip28)",
                        i, sz
                    ));
                }
            }
        }
        match (self.network.mode, self.machine.profile) {
            (NetMode::Off, p) if p != MachineProfile::O2Ip32 && p != MachineProfile::OctaneIp30 && p != MachineProfile::FuelIp34 => {
                return Err("network.mode = \"off\" is only available on the O2, Octane and Fuel; the Indy and Indigo2 always have their Ethernet".into());
            }
            _ => {}
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
            rsh_dir:      self.network.rsh_dir.as_ref().map(std::path::PathBuf::from),
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

    /// RAM bank 0 size in MB (0/8/16/32/64/128; IP28 also supports 256/512)
    #[arg(long)]
    pub bank0: Option<u32>,

    /// RAM bank 1 size in MB (0/8/16/32/64/128; IP28 also supports 256/512)
    #[arg(long)]
    pub bank1: Option<u32>,

    /// RAM bank 2 size in MB (0/8/16/32/64/128; IP28 also supports 256/512)
    #[arg(long)]
    pub bank2: Option<u32>,

    /// RAM bank 3 size in MB (0/8/16/32/64/128; IP28 also supports 256/512)
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

    /// Host audio system for HAL2: default, alsa or pulseaudio (overrides [audio] host)
    #[arg(long = "audio-host", value_enum, value_name = "HOST")]
    pub audio_host: Option<AudioHost>,

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

    /// Emulated CPU: `r4400` (default), `r5000`, `r10000`, `r12000` or `r14000`. Overrides `[machine] cpu`.
    /// The last two are limited to the machines they shipped in.
    ///
    /// Every model is compiled into every build — each is its own
    /// monomorphisation, so the hot path carries no per-model branch — and this
    /// picks between them at construction. The `r5k` Cargo feature is removed.
    #[arg(long = "cpu", value_name = "MODEL")]
    pub cpu: Option<CpuModel>,

    /// Cache model: `auto` (default: `shadow` in a jitv2 build, else `full`),
    /// `full` or `shadow`. Overrides `[machine] cache`. The R10000 is always
    /// `shadow`.
    #[arg(long = "cache", value_name = "MODEL")]
    pub cache: Option<CacheModel>,

    /// Graphics board: `newport` (default), `xz`, `extreme`, `solidimpact`, `highimpact`, `maximpact` (or `impact:solid`, `impact:high`, `impact:max`).
    #[arg(long = "graphics", visible_alias = "board", visible_alias = "graphics-board", value_name = "BOARD")]
    pub graphics: Option<GraphicsBoard>,

    /// Enable GDB stub on the given TCP port (e.g. --gdb-port 1234).
    /// Connect with: target remote localhost:<port>
    #[arg(long = "gdb-port", value_name = "PORT")]
    pub gdb_port: Option<u16>,

    /// Monitor console port on 127.0.0.1 (default 8888). Give each iris
    /// running at once its own.
    #[arg(long = "monitor-port", value_name = "PORT")]
    pub monitor_port: Option<u16>,

    /// Serve the screen, keyboard and mouse over VNC on ADDR (`host:port`,
    /// or a port on 127.0.0.1; default 127.0.0.1:5900). Works with a
    /// window, with --ci, and on every graphics board.
    #[arg(long = "vnc", value_name = "ADDR", num_args = 0..=1, default_missing_value = "127.0.0.1:5900")]
    pub vnc: Option<String>,

    /// Serial channel A (tty2) port on 127.0.0.1 (default 8880).
    #[arg(long = "serial-port-a", value_name = "PORT")]
    pub serial_port_a: Option<u16>,

    /// Serial channel B (tty1, the console) port on 127.0.0.1 (default 8881).
    #[arg(long = "serial-port-b", value_name = "PORT")]
    pub serial_port_b: Option<u16>,

    /// Physical host keyboard: `ansi` (default), `iso` or `jis`. Use `jis`
    /// for a Japanese keyboard with the PROM's `keybd` set to JP.
    #[arg(long = "keyboard", value_name = "LAYOUT")]
    pub keyboard: Option<KeyboardLayout>,

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

    /// Let the guest `rsh <gateway> cat|ls|echo|pwd` against this directory,
    /// read-only, through a built-in rsh server at the gateway (TCP 514).
    /// Nothing is executed on the host. Off when unset.
    #[arg(long = "rsh-dir", value_name = "DIR")]
    pub rsh_dir: Option<String>,

    /// Load a static ELF32 MSB (big-endian MIPS) binary into RAM before the
    /// CPU starts and set PC to its entry point, instead of booting the PROM.
    /// For bare-metal test binaries; see also the monitor's `loadelf`.
    #[arg(long = "load-elf", value_name = "FILE")]
    pub load_elf: Option<String>,

    /// Record a Perfetto trace into FILE from startup (build with
    /// `--features perfetto`). Open it at ui.perfetto.dev.
    #[arg(long, value_name = "FILE")]
    pub perfetto: Option<String>,

    /// Perfetto categories, e.g. `irq,gfx` or `all,-jit` [default: all]
    #[arg(long = "perfetto-cats", value_name = "LIST")]
    pub perfetto_cats: Option<String>,

    /// Perfetto timeline: `host` (ns) or `guest` (instructions) [default: host]
    #[arg(long = "perfetto-clock", value_name = "CLOCK")]
    pub perfetto_clock: Option<String>,

    /// CI mode: enable the control socket and apply speed-favoring fidelity
    /// shortcuts. Hides the window unless --ci-display is set; offscreen
    /// graphics remain active unless --headless is explicitly set.
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

    /// `off` (raw transcript) or `wall`: prefix each completed serial-log
    /// line with a UTC ISO-8601 timestamp.
    #[arg(long = "serial-log-timestamps", value_name = "MODE")]
    pub serial_log_timestamps: Option<crate::dev::serial_log::SerialLogTimestamps>,

    /// Override CP0 Count MHz (default 33 on IP22/IP24, 97.5 on IP28).
    /// IRIX reports twice this rate as CPU MHz, e.g. --clock-fixed-mhz 50.
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
        if let Some(cache) = self.cache { cfg.machine.cache = cache; }
        if let Some(b) = self.graphics { cfg.graphics.board = Some(b); }
        if self.scale2x { cfg.scale = 2; }
        if self.headless  { cfg.headless  = true; }
        if self.no_audio  { cfg.no_audio  = true; }
        if let Some(h) = self.audio_host { cfg.audio.host = h; }

        if self.no_scsi_deferred_int { cfg.scsi_deferred_int = false; }
        if self.ci         { cfg.ci         = true; }
        if let Some(p) = &self.ci_socket { cfg.ci_socket = p.clone(); }
        if self.ci_display { cfg.ci_display = true; }
        if let Some(p) = &self.serial_log { cfg.serial_log = Some(p.clone()); }
        if let Some(mode) = self.serial_log_timestamps { cfg.serial_log_timestamps = mode; }
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
        if let Some(p) = self.monitor_port { cfg.monitor_port = Some(p); }
        if let Some(a) = &self.vnc { cfg.vnc.listen = Some(a.clone()); }
        if let Some(p) = self.serial_port_a { cfg.serial_port_a = Some(p); }
        if let Some(p) = self.serial_port_b { cfg.serial_port_b = Some(p); }
        if let Some(k) = self.keyboard { cfg.keyboard = k; }
        if let Some(ref p) = self.load_elf { cfg.load_elf = Some(p.clone()); }
        if let Some(ref p) = self.perfetto { cfg.perfetto = Some(p.clone()); }
        if let Some(ref p) = self.perfetto_cats { cfg.perfetto_cats = Some(p.clone()); }
        if let Some(ref p) = self.perfetto_clock { cfg.perfetto_clock = Some(p.clone()); }
        if let Some(ref d) = self.tftp_dir { cfg.network.tftp_dir = Some(d.clone()); }
        if let Some(ref d) = self.rsh_dir { cfg.network.rsh_dir = Some(d.clone()); }
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
            print!("{}", crate::net::net_pcap::format_interfaces());
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

    /// The R12000 and R14000 are accepted only where a source records them.
    #[test]
    fn r12000_and_r14000_are_limited_to_the_machines_they_shipped_in() {
        for (profile, r12, r14) in [
            (MachineProfile::OctaneIp30, true, true),
            (MachineProfile::O2Ip32, true, false),
            (MachineProfile::FuelIp34, false, true),
            (MachineProfile::IndyIp24, false, false),
            (MachineProfile::Indigo2Ip22, false, false),
            (MachineProfile::Indigo2Ip28, false, false),
        ] {
            assert_eq!(CpuModel::R12000.fits(profile), r12, "R12000 on {profile:?}");
            assert_eq!(CpuModel::R14000.fits(profile), r14, "R14000 on {profile:?}");
            assert!(CpuModel::R10000.fits(profile));
        }
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::IndyIp24;
        cfg.machine.cpu = CpuModel::R12000;
        assert!(cfg.validate().unwrap_err().contains("R12000"));
    }

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
    fn octane_graphics_defaults_to_solid_impact() {
        let octane = |t: &str| -> MachineConfig {
            let mut c: MachineConfig =
                toml::from_str(&format!("banks = [128, 0, 0, 0]\n[machine]\nprofile = \"octane_ip30\"\ncpu = \"r10000\"\n{t}"))
                    .expect("parse");
            c.scsi.clear();
            c
        };
        let cfg = octane("");
        assert_eq!(cfg.graphics_board(), GraphicsBoard::SolidImpact);
        cfg.validate().expect("default Octane");
        for (t, b) in [("none", GraphicsBoard::None), ("impact:solid", GraphicsBoard::SolidImpact),
                       ("maximpact", GraphicsBoard::MaxImpact), ("highimpact", GraphicsBoard::HighImpact)] {
            let cfg = octane(&format!("[graphics]\nboard = \"{t}\""));
            assert_eq!(cfg.graphics_board(), b, "{t}");
            cfg.validate().expect(t);
        }
        for t in ["xz", "extreme", "newport"] {
            assert!(octane(&format!("[graphics]\nboard = \"{t}\"")).validate().is_err(), "{t} is not modelled on the Octane");
        }
        let err = octane("[octane]\ngraphics = \"maximpact\"").validate().unwrap_err();
        assert!(err.contains("[graphics]"), "the retired key names its replacement: {err}");
        // Elsewhere the default stays Newport, and none is the Octane's alone.
        let cfg: MachineConfig = toml::from_str("").expect("parse");
        assert_eq!(cfg.graphics_board(), GraphicsBoard::Newport);
        assert!(!GraphicsBoard::None.supports(MachineProfile::IndyIp24));
    }

    #[test]
    fn audio_host_parses_and_defaults() {
        let cfg: MachineConfig = toml::from_str("").expect("parse");
        assert_eq!(cfg.audio.host, AudioHost::Default);
        let cfg: MachineConfig = toml::from_str(r#"
            [audio]
            host = "pulseaudio"
            cpal_buffer_frames = 256
        "#).expect("parse");
        assert_eq!(cfg.audio.host, AudioHost::Pulseaudio);
        assert_eq!(cfg.audio.cpal_buffer_frames, Some(256));
        let back: MachineConfig = toml::from_str(&toml::to_string_pretty(&cfg).expect("serialize")).expect("deserialize");
        assert_eq!(back.audio.host, AudioHost::Pulseaudio);
        assert!(toml::from_str::<MachineConfig>("[audio]\nhost = \"oss\"").is_err());
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
    fn two_512_mb_banks_require_ip28() {
        let mut cfg = MachineConfig::default();
        cfg.banks = banks_from4([512, 512, 0, 0]);
        for profile in [MachineProfile::IndyIp24, MachineProfile::Indigo2Ip22] {
            cfg.machine.profile = profile;
            assert!(cfg.validate().unwrap_err().contains("512 MB needs the IP28"));
        }
        cfg.machine.profile = MachineProfile::Indigo2Ip28;
        cfg.machine.cpu = CpuModel::R10000;
        cfg.validate().expect("IP28 accepts two 512 MB banks");
    }

    #[test]
    fn a_256_mb_bank_is_an_ip28_bank() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::Indigo2Ip22;
        cfg.banks = banks_from4([256, 128, 0, 0]);
        let err = cfg.validate().expect_err("the IP22 MC cannot express a 256 MB bank");
        assert!(err.contains("256 MB"), "{err}");
        {
            cfg.machine.profile = MachineProfile::Indigo2Ip28;
            cfg.machine.cpu = CpuModel::R10000;
            cfg.validate().expect("the IP28 MC can");
        }
    }

    /// The O2 reads the same `[scsi.N]` tables as the other machines, with
    /// `controller` as the bus; like the Indigo2 it has two. Its Ethernet is
    /// built in, so a DaynaPort target is refused.
    #[test]
    fn o2_takes_scsi_tables_with_controller_as_bus() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::O2Ip32;
        cfg.machine.cpu = CpuModel::R10000;
        cfg.scsi.insert(2, ScsiDeviceConfig { path: "external.raw".into(), controller: 1, ..Default::default() });
        cfg.validate().expect("the O2 has two SCSI controllers");
        cfg.machine.profile = MachineProfile::IndyIp24;
        cfg.machine.cpu = CpuModel::default();
        assert!(cfg.validate().unwrap_err().contains("controller 1"), "the Indy has one");
        cfg.machine.profile = MachineProfile::O2Ip32;
        cfg.machine.cpu = CpuModel::R10000;
        cfg.scsi.insert(3, ScsiDeviceConfig { kind_field: ScsiKind::Daynaport, ..Default::default() });
        assert!(cfg.validate().unwrap_err().contains("daynaport"));
    }

    /// `banks` reads one to eight entries and pads; it writes four or more
    /// with no trailing zeros past the fourth; banks a machine does not have
    /// must be 0; the O2 takes DIMM kits, largest first, and nothing else.
    #[test]
    fn banks_have_eight_slots_and_four_users() {
        let cfg: MachineConfig = toml::from_str("banks = [64, 64]").expect("short list pads");
        assert_eq!(cfg.banks, banks_from4([64, 64, 0, 0]));
        let s = toml::to_string(&MachineConfig::default()).expect("serialize");
        assert!(s.contains("banks = [128, 128, 0, 0]"), "four entries, no more:\n{s}");
        assert!(toml::from_str::<MachineConfig>("banks = [8, 8, 8, 8, 8, 8, 8, 8, 8]").is_err(), "nine is too many");

        let mut cfg = MachineConfig::default();
        cfg.banks[4] = 8;
        assert!(cfg.validate().unwrap_err().contains("has 4 memory banks"));

        let mut o2 = MachineConfig::default();
        o2.machine.profile = MachineProfile::O2Ip32;
        o2.machine.cpu = CpuModel::R10000;
        o2.validate().expect("two 128 MB kits are an O2's 256 MB");
        o2.banks = banks_from4([256, 256, 256, 256]);
        o2.validate().expect("1 GB");
        assert_eq!(o2.o2_kits(), vec![256, 256, 256, 256]);
        o2.banks = banks_from4([8, 0, 0, 0]);
        assert!(o2.validate().unwrap_err().contains("no 8 MB kit"));
        o2.banks = banks_from4([128, 0, 128, 0]);
        assert!(o2.validate().unwrap_err().contains("pair B is empty"));
        o2.banks = banks_from4([64, 128, 0, 0]);
        assert!(o2.validate().unwrap_err().contains("largest kit goes in pair A"));
        o2.banks = default_banks();
        o2.network.mode = NetMode::Pcap;
        o2.validate().expect("a bridged O2");
        o2.network.mode = NetMode::Off;
        o2.validate().expect("an unplugged O2");
        let mut indy = MachineConfig::default();
        indy.network.mode = NetMode::Off;
        assert!(indy.validate().unwrap_err().contains("only available on the O2"));
    }

    /// Every old `[o2]` key lands on the shared key it stood for, and the
    /// section keeps only what is the O2's alone.
    #[test]
    fn deprecated_o2_keys_fold_into_shared_settings() {
        let mut cfg: MachineConfig = toml::from_str(r#"
            [machine]
            profile = "o2_ip32"
            cpu = "r10000"
            [o2]
            prom = "o2prom.bin"
            flash = "o2flash.bin"
            ram_mb = 384
            serial_port = 2291
            serial_log = "o2.log"
            graphics = false
            cpu_mhz = 195
            net = false
            mouse = false
            control_port = 2293
            forwards = [ { proto = "tcp", host_port = 2292, guest_port = 22 } ]
        "#).expect("parse");
        cfg.fold_deprecated();
        assert_eq!(cfg.prom, "o2prom.bin");
        assert_eq!(cfg.nvram, "o2flash.bin");
        assert_eq!(cfg.banks, banks_from4([256, 128, 0, 0]), "384 MB, largest kits first");
        assert_eq!(cfg.serial_port_a, Some(2291));
        assert_eq!(cfg.serial_log.as_deref(), Some("o2.log"));
        assert!(cfg.headless);
        assert_eq!(cfg.clock.fixed_mhz, Some(97.5));
        assert_eq!(cfg.network.mode, NetMode::Off);
        assert_eq!(cfg.port_forward.len(), 1);
        assert_eq!(cfg.port_forward[0].host_port, 2292);
        assert!(!cfg.o2.mouse);
        assert_eq!(cfg.o2.control_port, Some(2293));
        cfg.validate().expect("the folded config is a valid O2");
        let s = toml::to_string_pretty(&cfg).expect("serialize");
        for gone in ["ram_mb =", "cpu_mhz =", "flash =", "forwards =", "graphics =", "serial_port =", "net ="] {
            assert!(!s.contains(gone), "{gone} must not be written back:\n{s}");
        }

        // A shared key that is set wins over the old spelling.
        let mut cfg: MachineConfig = toml::from_str(r#"
            prom = "mine.bin"
            [machine]
            profile = "o2_ip32"
            [o2]
            prom = "old.bin"
        "#).expect("parse");
        cfg.fold_deprecated();
        assert_eq!(cfg.prom, "mine.bin");
    }

    /// `[scsi.17]` is id 1 on controller 1, so one id can be on both
    /// controllers; `controller = 1` on a low key still works; a key that
    /// names a controller cannot be contradicted, two tables cannot name the
    /// same target, and id 0 and controller 2 do not exist.
    #[test]
    fn slot_keys_place_targets_on_either_controller() {
        let mut cfg: MachineConfig = toml::from_str(r#"
            [machine]
            profile = "indigo2_ip22"
            [scsi.1]
            path = "internal.raw"
            [scsi.17]
            path = "external.raw"
            [scsi.4]
            path = "cd.iso"
            cdrom = true
            controller = 1
        "#).expect("parse");
        cfg.validate().expect("two controllers, no collision");
        assert_eq!(cfg.scsi[&1].placement(1), (0, 1));
        assert_eq!(cfg.scsi[&17].placement(17), (1, 1));
        assert_eq!(cfg.scsi[&4].placement(4), (1, 4));
        assert_eq!(scsi_slot(1, 1), 17);
        assert_eq!(scsi_label(0, 1), "SCSI ID 1");
        assert_eq!(scsi_label(1, 1), "SCSI 1:1");

        cfg.scsi.insert(20, ScsiDeviceConfig { path: "twice.iso".into(), cdrom: true, ..Default::default() });
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("scsi.4 and scsi.20 both name SCSI 1:4"), "{err}");
        cfg.scsi.remove(&20);

        cfg.scsi.insert(18, ScsiDeviceConfig { path: "x.raw".into(), controller: 0, ..Default::default() });
        cfg.validate().expect("controller 0 on a slot key is just the default");
        cfg.scsi.get_mut(&18).unwrap().controller = 2;
        assert!(cfg.validate().unwrap_err().contains("contradicts"));
        cfg.scsi.remove(&18);

        cfg.scsi.insert(16, ScsiDeviceConfig { path: "x.raw".into(), ..Default::default() });
        assert!(cfg.validate().unwrap_err().contains("out of range"));
        cfg.scsi.remove(&16);
        cfg.scsi.insert(33, ScsiDeviceConfig { path: "x.raw".into(), ..Default::default() });
        assert!(cfg.validate().unwrap_err().contains("controller 2 is invalid"));
        cfg.scsi.remove(&33);

        cfg.machine.profile = MachineProfile::IndyIp24;
        assert!(cfg.validate().unwrap_err().contains("this machine has one"));
    }

    /// An older O2 config's `o2.disks` becomes `[scsi.N]` entries, replacing
    /// the default disks, and is gone from the section afterwards.
    #[test]
    fn deprecated_o2_disks_fold_into_scsi() {
        let mut cfg: MachineConfig = toml::from_str(r#"
            [machine]
            profile = "o2_ip32"
            [o2]
            disks = [ { id = 1, path = "sys.raw", overlay = true },
                      { id = 4, path = "cd.iso", cdrom = true, bus = 1 } ]
        "#).expect("parse");
        cfg.fold_deprecated();
        assert!(cfg.o2.disks.is_empty());
        assert_eq!(cfg.scsi.len(), 2, "the list replaces the default map");
        assert_eq!(cfg.scsi[&1].path, "sys.raw");
        assert!(cfg.scsi[&1].overlay);
        assert_eq!(cfg.scsi[&1].controller, 0);
        assert!(cfg.scsi[&4].is_cdrom());
        assert_eq!(cfg.scsi[&4].controller, 1);
        let s = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(!s.contains("disks"), "the deprecated key is never written back:\n{s}");
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

#[cfg(test)]
mod fuel_input_tests {
    use super::*;

    /// A config saved for another machine must not grow a `[fuel]` table,
    /// and a Fuel's own settings must survive a save and reload.
    #[test]
    fn only_a_configured_fuel_section_is_saved_and_it_round_trips() {
        let other = MachineConfig::default();
        let saved = toml::to_string(&other).unwrap();
        assert!(!saved.contains("[fuel]"), "{saved}");
        let mut fuel = MachineConfig::default();
        fuel.fuel.console = FuelConsole::Ioc3B;
        fuel.fuel.flash = Some("flash.bin".into());
        fuel.fuel.l1_port = Some(18882);
        fuel.fuel.l1_log = Some("l1.log".into());
        fuel.fuel.usb_input = true;
        fuel.fuel.ps2_input = false;
        let saved = toml::to_string(&fuel).unwrap();
        assert!(saved.contains("[fuel]"), "{saved}");
        let loaded: MachineConfig = toml::from_str(&saved).unwrap();
        assert_eq!(loaded.fuel.console, FuelConsole::Ioc3B);
        assert_eq!(loaded.fuel.flash.as_deref(), Some("flash.bin"));
        assert_eq!(loaded.fuel.l1_port, Some(18882));
        assert_eq!(loaded.fuel.l1_log.as_deref(), Some("l1.log"));
        assert!(!loaded.fuel.ps2_input && loaded.fuel.usb_input);
        assert!(!MachineProfile::ALL.contains(&MachineProfile::FuelIp34));
    }

    /// PS/2 is the Fuel's standard keyboard and mouse; USB is opt-in, and a
    /// `[fuel]` table that names neither key keeps both defaults.
    #[test]
    fn fuel_input_defaults_to_ps2_with_usb_opt_in() {
        let none: MachineConfig = toml::from_str("[fuel]\nflash = \"flash.bin\"\n").unwrap();
        assert!(none.fuel.ps2_input && !none.fuel.usb_input);
        assert!(MachineConfig::default().fuel.is_default());
        let usb: MachineConfig = toml::from_str("[fuel]\nusb_input = true\n").unwrap();
        assert!(usb.fuel.ps2_input && usb.fuel.usb_input);
        assert!(!usb.fuel.is_default(), "an opt-in is written back");
    }

    /// The Fuel's processor is fixed, so `machine.cpu` may only name it (or the
    /// family, as the key did before the R14000 had a name); the default
    /// `r4400` is refused rather than silently ignored.
    #[test]
    fn fuel_requires_the_r14000_cpu_key() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::FuelIp34;
        cfg.banks.fill(0);
        cfg.banks[0] = 256;
        cfg.banks[1] = 256;
        cfg.scsi.clear();
        assert!(cfg.validate().unwrap_err().contains("r14000"));
        cfg.machine.cpu = CpuModel::R10000;
        cfg.validate().unwrap();
        cfg.machine.cpu = CpuModel::R14000;
        cfg.validate().unwrap();
        cfg.machine.cpu = CpuModel::R12000;
        assert!(cfg.validate().is_err(), "no source has a Fuel with an R12000");
    }

    #[test]
    fn fuel_requires_its_fixed_memory_population_and_no_graphics() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::FuelIp34;
        cfg.machine.cpu = CpuModel::R10000;
        cfg.banks.fill(0);
        cfg.banks[0] = 256;
        cfg.banks[1] = 256;
        cfg.scsi.clear();
        assert_eq!(cfg.graphics_board(), GraphicsBoard::None);
        cfg.network.mode = NetMode::Off;
        cfg.validate().unwrap();
        cfg.network.mode = NetMode::Pcap;
        cfg.validate().unwrap();
        cfg.graphics.board = Some(GraphicsBoard::Newport);
        assert!(cfg.validate().unwrap_err().contains("none"));
        cfg.graphics.board = None;
        cfg.banks[1] = 0;
        assert!(cfg.validate().unwrap_err().contains("256, 256"));
    }

    #[test]
    fn fuel_accepts_wide_scsi_ids_on_both_channels_without_changing_other_profiles() {
        let text = "banks = [256, 256]\n[machine]\nprofile = \"fuel_ip34\"\ncpu = \"r10000\"\n[scsi.15]\npath = \"disk.raw\"\n[scsi.31]\npath = \"disk1.raw\"\n";
        let mut cfg: MachineConfig = toml::from_str(text).unwrap();
        cfg.validate().unwrap();
        cfg.machine.profile = MachineProfile::OctaneIp30;
        assert!(cfg.validate().unwrap_err().contains("out of range"));
    }
}
