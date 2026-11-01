//! Host-side networking: the NAT gateway the guest's Ethernet talks to, its
//! in-core services, and the PCAP alternative to it.
//!
//! A NIC model (seeq8003, DaynaPort, MEC, IOC3) hands guest frames to a
//! [`NetBackend`] over an rtrb ring and takes frames back on another. The NAT backend is
//! [`NatEngine`]; see `clanker/plans/net-stack.md` for its design.
//!
//!   packet.rs    Ethernet/ARP/IPv4/ICMP/UDP/TCP views and builders
//!   engine.rs    NatEngine: the loop, ARP, IP dispatch, frames to the guest
//!   tcp.rs       guest-facing TCP endpoint; host-socket and service backends
//!   nat_tcp.rs   TCP flows: connection setup, polling (+ FTP passive ALG)
//!   nat_udp.rs   UDP dispatch and flows
//!   nat_icmp.rs  ICMP echo, local and relayed
//!   fwd.rs       inbound port forwards (+ XDMCP ALG)
//!   services.rs  BOOTP/DHCP, DNS relay, NFS, portmap, TFTP, time, NTP
//!   frag.rs      inbound fragment reassembly
//!   portmap.rs   RPC GETPORT for the in-core NFS server
//!   nfsudp.rs    in-core NFSv2/v3 server
//!   nfs_pcap.rs  the NFS server as its own LAN host (PCAP mode)
//!   tftp.rs      read-only TFTP for PROM network boot
//!   rsh.rs       read-only rsh (TCP 514): a few built-in commands over one directory
//!   xdmcp.rs     XDMCP gateway into the guest's xdm
//!   host_dns.rs  host DNS server discovery for DNS relaying
//!   link.rs      an adapter's connection to the backend `[network]` selects
//!   net_pcap.rs  PCAP bridged backend (`pcap` feature)
//!   telnet.rs    telnet option negotiation for the monitor/serial consoles

mod engine;
mod fwd;
mod frag;
mod host_dns;
mod link;
mod nat_icmp;
mod nat_tcp;
mod nat_udp;
#[cfg_attr(not(feature = "pcap"), allow(dead_code))]
mod nfs_pcap;
#[cfg(feature = "pcap")]
pub mod net_pcap;
pub mod nfsudp;
pub mod packet;
mod portmap;
mod rsh;
mod services;
mod tcp;
pub mod wake;
pub mod telnet;
#[cfg(test)]
mod testnet;
pub mod tftp;
pub mod xdmcp;

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::config::{NatSubnet, NetMode, NfsConfig, PortForwardConfig};

pub use engine::NatEngine;
pub use link::{Link, Rings};
pub use wake::Wake;
#[cfg(feature = "pcap")]
pub(crate) use nfs_pcap::NfsVirtualHost;
pub use packet::mac_str;

/// One-line summary of an Ethernet frame for debug logging.
pub fn eth_summary(frame: &[u8]) -> String {
    packet::summary(frame)
}

// ── Gateway configuration ─────────────────────────────────────────────────────
#[derive(Clone)]
pub struct GatewayConfig {
    pub gateway_mac: [u8; 6],
    pub gateway_ip:  Ipv4Addr,
    pub client_ip:   Ipv4Addr,
    pub netmask:     Ipv4Addr,
    /// Fixed upstream for guest DNS queries. None follows the host's configured
    /// DNS server (see [`crate::net::host_dns`]), falling back to 8.8.8.8.
    pub dns_upstream: Option<SocketAddr>,
    /// NFS configuration; if Some, portmap and NAT redirection for NFS/mountd are enabled.
    pub nfs: Option<NfsConfig>,
    /// Port forwarding rules: host listens and forwards to the guest.
    pub port_forwards: Vec<PortForwardConfig>,
    /// Networking backend: NAT (software gateway) or PCAP (bridged). Default NAT.
    pub mode: NetMode,
    /// Host interface to bridge onto in PCAP mode. None = auto-pick.
    pub pcap_interface: Option<String>,
    /// PCAP-only: virtual LAN IP the in-process NFS server answers on. `Some`
    /// together with `nfs` enables the bridged NFS responder ([`NfsVirtualHost`]).
    pub nfs_pcap_ip: Option<Ipv4Addr>,
    /// Root directory served read-only over TFTP at the gateway, for PROM
    /// network boot (`boot -f bootp()<file>`). None disables TFTP entirely.
    pub tftp_dir: Option<std::path::PathBuf>,
    /// Root directory for the built-in read-only rsh server (TCP 514 at the
    /// gateway): `rsh <gateway> cat|ls|echo|pwd`. Nothing is executed on the
    /// host. None disables it, and 514 is then relayed to the host like any port.
    pub rsh_dir: Option<std::path::PathBuf>,
    /// Hold frames for the guest while its receive ring is about to end,
    /// where the hardware would stop its channel (`[network] rx_hold`; the Indy's
    /// SEEQ only).
    pub rx_hold: bool,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        let subnet = NatSubnet::default();
        Self {
            gateway_mac:  [0x02, 0x00, 0xDE, 0xAD, 0xBE, 0xEF],
            gateway_ip:   subnet.gateway_ip,
            client_ip:    subnet.client_ip,
            netmask:      subnet.netmask,
            dns_upstream: None,
            nfs:          None,
            port_forwards: vec![],
            mode:         NetMode::Nat,
            pcap_interface: None,
            nfs_pcap_ip:  None,
            tftp_dir:     None,
            rsh_dir:      None,
            rx_hold:      true,
        }
    }
}

impl GatewayConfig {
    /// The gateway `[network]`, `nat_subnet`, `[nfs]` and `[[port_forward]]`
    /// describe, as the machines hand it to their Ethernet.
    pub fn from_network(net: &crate::config::NetworkConfig) -> Self {
        let subnet = net.nat_subnet.clone().unwrap_or_default();
        Self {
            gateway_ip: subnet.gateway_ip,
            client_ip: subnet.client_ip,
            netmask: subnet.netmask,
            nfs: net.nfs.clone(),
            port_forwards: net.port_forward.clone(),
            mode: net.mode,
            pcap_interface: net.pcap_interface.clone(),
            nfs_pcap_ip: net.nfs_pcap_ip,
            tftp_dir: net.tftp_dir.clone(),
            rsh_dir: net.rsh_dir.clone(),
            rx_hold: net.rx_hold,
            ..Self::default()
        }
    }
}

// ── Network backend abstraction ───────────────────────────────────────────────
//
// The SEEQ 8003 chip is backend-agnostic: it hands outbound Ethernet frames to a
// backend over an rtrb ring and receives inbound frames back over another. Any
// type that owns those endpoints and a run loop can serve as the backend. Today
// there are two implementations:
//   - `NatEngine` (this file): a software NAT gateway/router.
//   - `PcapEngine` (net_pcap.rs, `--features pcap`): bridges frames onto a real
//     host interface via libpcap.
pub trait NetBackend: Send {
    /// Run the backend loop until the shared `running` flag goes false. Blocks
    /// the calling thread (the `seeq-net` thread).
    fn run(&mut self);
}

/// Live status of the PCAP bridged-capture backend, surfaced to an embedder (the
/// GUI) so it can prompt for privilege elevation when the raw capture can't be
/// opened. Deliberately **not** feature-gated — `NatControl` always carries the
/// field and non-`pcap` builds / the GUI reference the type unconditionally; it
/// only ever leaves `Inactive` when the `pcap` feature is built and the active
/// machine uses `[network] mode = "pcap"`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub enum PcapStatus {
    /// Not bridging — NAT mode, or PCAP not yet attempted (default).
    #[default]
    Inactive = 0,
    /// Capture handle is open; frames are bridging onto the host interface.
    Active = 1,
    /// Open failed for lack of privilege (EPERM/EACCES, libpcap "permission
    /// denied"). The embedder should offer to elevate / enable capture.
    PermissionDenied = 2,
    /// Open failed for another reason (no such device, driver missing, …).
    DeviceError = 3,
}

impl PcapStatus {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => PcapStatus::Active,
            2 => PcapStatus::PermissionDenied,
            3 => PcapStatus::DeviceError,
            _ => PcapStatus::Inactive,
        }
    }
}

// ── NAT debug/status control (shared between NatEngine thread and command handler) ─
pub struct NatControl {
    pub debug_tcp:  AtomicBool,
    pub debug_udp:  AtomicBool,
    pub debug_icmp: AtomicBool,
    pub snapshot:   Mutex<NatSnapshot>,
    /// Set to true to flush all NAT tables on the next NatEngine loop iteration.
    /// The NAT thread clears the flag after flushing.
    pub reset_nat:  AtomicBool,
    /// Count of guest-originated IP frames the NAT engine has processed (ARP and
    /// other link-layer chatter are excluded — they happen even with no/wrong
    /// IP). Monotonic for the life of the machine; the GUI samples it to light a
    /// grey/red/green "internal network" indicator.
    pub guest_frames: Arc<AtomicU64>,
    /// Set once the guest sends an IP frame *to the gateway's MAC* — i.e. it is
    /// actually routing off-subnet through NAT. Gates plug-and-play adoption:
    /// adoption locks only once the guest is really using the gateway, so local
    /// or broadcast IP chatter (e.g. a ping to x.x.x.255) no longer disarms it.
    pub routed: AtomicBool,
    /// The guest's own source IPv4 address as last seen on the wire (ARP sender
    /// address, or IP source) — `0` = none seen yet. Captured even when the
    /// guest's config is wrong and nothing routes (it still ARPs for its
    /// gateway), so the GUI can show what address the guest actually has and
    /// compare it to what NAT expects.
    pub observed_guest_ip: AtomicU32,
    /// The address the guest is ARP-ing for within its own subnet but failing
    /// to resolve (NAT doesn't own it) — almost always its configured default
    /// gateway. `0` = none seen. Lets the GUI tell what gateway the guest
    /// expects (and whether the host-side fix will satisfy it).
    pub observed_gateway: AtomicU32,
    /// Pending live subnet change (gateway / client / netmask as u32), latched
    /// by `apply_subnet`. Lets an embedder move the running NAT onto a new
    /// subnet without a reboot; applied on the NAT thread's next loop.
    pub apply_gateway: AtomicU32,
    pub apply_client:  AtomicU32,
    pub apply_netmask: AtomicU32,
    pub apply_subnet:  AtomicBool,
    /// The host's own IPv4 networks `(network, prefix)`, set by the embedder.
    /// NAT refuses to plug-and-play *adopt* a subnet that overlaps one of these,
    /// so the emulator never shadows the host's real LAN / VPN / Docker network.
    pub host_nets: Mutex<Vec<(u32, u8)>>,
    /// New port-forward rule set to bind live, latched by `apply_forwards`. Lets
    /// an embedder add/remove forwards without a reboot.
    pub pending_forwards: Mutex<Option<Vec<PortForwardConfig>>>,
    pub apply_forwards:   AtomicBool,
    /// Live PCAP backend status (a [`PcapStatus`] discriminant), set by the
    /// `PcapEngine` as it opens (or fails to open) the capture and sampled by the
    /// GUI to drive the "Enable packet capture" / elevation prompt. `Inactive`
    /// (0) in NAT mode and on non-`pcap` builds.
    pub pcap_status: AtomicU8,
    /// Pending PCAP host-interface reswap (`None` = auto-pick), latched by
    /// `request_pcap_interface`. Lets the GUI change the bridged NIC on a running
    /// machine: the `PcapEngine` reopens its capture on the next loop — no guest
    /// reboot. Unused in NAT mode.
    pub pending_pcap_iface: Mutex<Option<String>>,
    pub apply_pcap_iface:   AtomicBool,
}

impl NatControl {
    pub fn new() -> Arc<Self> {
        Self::counting(Arc::new(AtomicU64::new(0)))
    }

    /// Controls whose frame count is `frames`, shared with the backend-neutral
    /// count a [`Link`] keeps.
    pub fn counting(frames: Arc<AtomicU64>) -> Arc<Self> {
        Arc::new(Self {
            debug_tcp:  AtomicBool::new(false),
            debug_udp:  AtomicBool::new(false),
            debug_icmp: AtomicBool::new(false),
            snapshot:   Mutex::new(NatSnapshot::default()),
            reset_nat:  AtomicBool::new(false),
            guest_frames: frames,
            routed: AtomicBool::new(false),
            observed_guest_ip: AtomicU32::new(0),
            observed_gateway: AtomicU32::new(0),
            apply_gateway: AtomicU32::new(0),
            apply_client:  AtomicU32::new(0),
            apply_netmask: AtomicU32::new(0),
            apply_subnet:  AtomicBool::new(false),
            host_nets:     Mutex::new(Vec::new()),
            pending_forwards: Mutex::new(None),
            apply_forwards:   AtomicBool::new(false),
            pcap_status:      AtomicU8::new(PcapStatus::Inactive as u8),
            pending_pcap_iface: Mutex::new(None),
            apply_pcap_iface:   AtomicBool::new(false),
        })
    }

    /// Ask the running `PcapEngine` to reopen its capture on host interface
    /// `iface` (`None` = auto-pick), applied on the engine's next loop. The guest
    /// keeps running — only the host-side capture handle is swapped. No-op in NAT.
    pub fn request_pcap_interface(&self, iface: Option<String>) {
        *self.pending_pcap_iface.lock() = iface;
        self.apply_pcap_iface.store(true, Ordering::Release);
    }

    /// Record the live PCAP backend status (called by the `PcapEngine`).
    pub fn set_pcap_status(&self, s: PcapStatus) {
        self.pcap_status.store(s as u8, Ordering::Relaxed);
    }

    /// The live PCAP backend status, for an embedder to drive elevation UI.
    pub fn pcap_status(&self) -> PcapStatus {
        PcapStatus::from_u8(self.pcap_status.load(Ordering::Relaxed))
    }
    pub fn dbg_tcp(&self)  -> bool { self.debug_tcp.load(Ordering::Relaxed) }
    pub fn dbg_udp(&self)  -> bool { self.debug_udp.load(Ordering::Relaxed) }
    pub fn dbg_icmp(&self) -> bool { self.debug_icmp.load(Ordering::Relaxed) }
    /// The guest's likely default gateway (in-subnet ARP target it can't
    /// resolve), or None if none seen.
    pub fn observed_gateway(&self) -> Option<Ipv4Addr> {
        match self.observed_gateway.load(Ordering::Relaxed) {
            0 => None,
            v => Some(Ipv4Addr::from(v)),
        }
    }
    /// The guest's last-seen source IP, or None if no frame has revealed one.
    pub fn observed_guest_ip(&self) -> Option<Ipv4Addr> {
        match self.observed_guest_ip.load(Ordering::Relaxed) {
            0 => None,
            v => Some(Ipv4Addr::from(v)),
        }
    }
    /// Number of guest frames the NAT engine has seen so far.

    /// Ask the running NAT to move to a new subnet (applied on the NAT thread's
    /// next loop: config swapped + connection tables flushed). No reboot needed.
    pub fn request_subnet(&self, gateway: Ipv4Addr, client: Ipv4Addr, netmask: Ipv4Addr) {
        self.apply_gateway.store(u32::from(gateway), Ordering::Relaxed);
        self.apply_client.store(u32::from(client), Ordering::Relaxed);
        self.apply_netmask.store(u32::from(netmask), Ordering::Relaxed);
        self.apply_subnet.store(true, Ordering::Release);
    }

    /// Record the host's own IPv4 networks (network address + prefix) so NAT
    /// won't adopt a subnet that overlaps them.
    pub fn set_host_nets(&self, nets: Vec<(Ipv4Addr, u8)>) {
        *self.host_nets.lock() = nets.into_iter().map(|(n, p)| (u32::from(n), p)).collect();
    }

    /// Replace the running NAT's port-forward rules (rebound on the NAT thread's
    /// next loop). No reboot needed.
    pub fn set_port_forwards(&self, rules: Vec<PortForwardConfig>) {
        *self.pending_forwards.lock() = Some(rules);
        self.apply_forwards.store(true, Ordering::Release);
    }

    /// Whether `net/prefix` overlaps any recorded host network — i.e. adopting
    /// or moving onto it would shadow a network the host already uses.
    pub fn host_conflict(&self, net: Ipv4Addr, prefix: u8) -> bool {
        let a = u32::from(net);
        self.host_nets.lock().iter().any(|&(b, hp)| {
            let p = prefix.min(hp);
            let mask = if p == 0 { 0 } else { u32::MAX << (32 - p) };
            (a & mask) == (b & mask)
        })
    }
}

#[derive(Default)]
pub struct NatTcpInfo {
    pub remote_ip:        String,
    pub remote_port:      u16,
    pub client_port:      u16,
    pub age_secs:         u64,
    pub server_seq:       u32,
    pub server_seq_acked: u32,
    pub client_win:       u32,
    pub client_seq:       u32,
    pub fin_wait:         bool,
    pub rtx_count:        usize,  // segments in retransmit queue
    pub rtx_bytes:        usize,  // bytes in retransmit queue
}

#[derive(Default)]
pub struct NatUdpInfo {
    pub remote_ip:   String,
    pub remote_port: u16,
    pub client_port: u16,
    pub age_secs:    u64,
}

#[derive(Default)]
pub struct NatIcmpInfo {
    pub remote_ip: String,
    pub ident:     u16,
    pub age_secs:  u64,
}

#[derive(Default)]
pub struct NatSnapshot {
    pub tcp:  Vec<NatTcpInfo>,
    pub udp:  Vec<NatUdpInfo>,
    pub icmp: Vec<NatIcmpInfo>,
}
