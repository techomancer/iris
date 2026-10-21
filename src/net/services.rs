//! In-core services answered at the gateway.
//!
//! UDP services implement [`UdpService`]: they see a peer and a datagram and
//! return datagrams, never packets; the engine owns addressing, fragmentation
//! and which MAC to answer. TCP services ride the real TCP endpoint (see
//! `tcp.rs`). BOOTP/DHCP is handled by the engine itself, because its replies
//! are addressed unlike any other (broadcast, or to an address being assigned).

use std::net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4};
use std::time::{Duration, Instant};

use super::engine::NatEngine;
use super::packet::MacAddr;
use super::portmap::{self, MOUNTD_PORT, NFS_PORT, PORTMAP_PORT};
use crate::devlog::LogModule;

pub(super) const BOOTP_SERVER_PORT: u16 = 67;
pub(super) const BOOTP_CLIENT_PORT: u16 = 68;
pub(super) const DNS_PORT: u16 = 53;
pub(super) const TIME_PORT: u16 = 37;
pub(super) const NTP_PORT: u16 = 123;
pub(super) const TFTP_PORT: u16 = 69;

const BOOTP_REQUEST: u8 = 1;
const DHCP_MAGIC: [u8; 4] = [99, 130, 83, 99];

/// Seconds from the NTP/RFC 868 epoch (1900) to the Unix epoch.
const NTP_EPOCH_OFFSET: u64 = 2_208_988_800;

/// Advertised to the guest as its resolver, and the upstream when the host's
/// own resolver can't be found.
pub(super) const DNS_FALLBACK: Ipv4Addr = Ipv4Addr::new(8, 8, 8, 8);
/// How long a discovered host DNS server is trusted before re-reading it.
const HOST_DNS_REFRESH: Duration = Duration::from_secs(5);
/// Forget the MAC of a service peer not heard from for this long.
const PEER_IDLE: Duration = Duration::from_secs(600);

/// A service on a UDP port at the gateway.
pub(super) trait UdpService: Send {
    /// A datagram from `peer`; returns the datagrams to send back to it.
    fn datagram(&mut self, peer: SocketAddrV4, data: &[u8]) -> Vec<Vec<u8>>;
    /// Timer work (retransmissions): datagrams to send, and to whom.
    fn poll(&mut self) -> Vec<(SocketAddrV4, Vec<u8>)> {
        Vec::new()
    }
}

/// The services a configuration enables, with the gateway ports each answers.
pub(super) fn udp_services(config: &super::GatewayConfig) -> Vec<(Vec<u16>, Box<dyn UdpService>)> {
    let mut v: Vec<(Vec<u16>, Box<dyn UdpService>)> = vec![
        (vec![TIME_PORT], Box::new(TimeUdp)),
        (vec![NTP_PORT], Box::new(Ntp)),
    ];
    if let Some(c) = &config.nfs {
        eprintln!("iris: in-core NFS server exporting {}", c.shared_dir);
        v.push((vec![PORTMAP_PORT], Box::new(Portmap)));
        v.push((vec![NFS_PORT, MOUNTD_PORT],
                Box::new(Nfs(super::nfsudp::NfsServer::new(c.shared_dir.clone(), c.version)))));
    }
    if let Some(dir) = &config.tftp_dir {
        eprintln!("iris: TFTP server (read-only) serving {}", dir.display());
        v.push((vec![TFTP_PORT], Box::new(Tftp(super::tftp::TftpServer::new(dir.clone())))));
    }
    v
}

// ── The services ─────────────────────────────────────────────────────────────

pub(super) fn rfc868_time() -> [u8; 4] {
    let unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    ((unix + NTP_EPOCH_OFFSET) as u32).to_be_bytes()
}

/// 64-bit NTP timestamp: seconds.fraction since 1900-01-01 UTC.
fn ntp_timestamp() -> u64 {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs() + NTP_EPOCH_OFFSET;
    let frac = ((d.subsec_nanos() as u64) << 32) / 1_000_000_000;
    (secs << 32) | frac
}

/// RFC 868 time over UDP: any datagram gets the time.
struct TimeUdp;

impl UdpService for TimeUdp {
    fn datagram(&mut self, _: SocketAddrV4, _: &[u8]) -> Vec<Vec<u8>> {
        vec![rfc868_time().to_vec()]
    }
}

/// NTP: a stratum-1 server reading the host clock.
struct Ntp;

impl UdpService for Ntp {
    fn datagram(&mut self, _: SocketAddrV4, query: &[u8]) -> Vec<Vec<u8>> {
        if query.len() < 48 {
            return Vec::new();
        }
        let mut pkt = vec![0u8; 48];
        pkt[0] = 4 << 3 | 4; // LI=0, VN=4, Mode=4 (server)
        pkt[1] = 1;          // stratum 1 (primary)
        pkt[2] = 4;          // poll interval (log2 16 s)
        pkt[3] = 0xEC;       // precision ~ 2^-20 s
        pkt[12..16].copy_from_slice(b"LOCL"); // reference identifier
        let recv = ntp_timestamp();
        pkt[16..24].copy_from_slice(&recv.to_be_bytes()); // reference
        pkt[24..32].copy_from_slice(&query[40..48]);       // originate = client's transmit
        pkt[32..40].copy_from_slice(&recv.to_be_bytes()); // receive
        pkt[40..48].copy_from_slice(&ntp_timestamp().to_be_bytes()); // transmit
        vec![pkt]
    }
}

struct Portmap;

impl UdpService for Portmap {
    fn datagram(&mut self, peer: SocketAddrV4, call: &[u8]) -> Vec<Vec<u8>> {
        dlog_dev!(LogModule::Net, "NAT portmap UDP from {} → port={}", peer, portmap::lookup(call));
        vec![portmap::reply(call)]
    }
}

/// NFS and mountd (one server answers both programs).
struct Nfs(super::nfsudp::NfsServer);

impl UdpService for Nfs {
    fn datagram(&mut self, _: SocketAddrV4, call: &[u8]) -> Vec<Vec<u8>> {
        self.0.handle(call).into_iter().collect()
    }
}

struct Tftp(super::tftp::TftpServer);

impl UdpService for Tftp {
    fn datagram(&mut self, peer: SocketAddrV4, data: &[u8]) -> Vec<Vec<u8>> {
        self.0.handle((*peer.ip(), peer.port()), data, Instant::now()).into_iter().collect()
    }
    fn poll(&mut self) -> Vec<(SocketAddrV4, Vec<u8>)> {
        self.0.tick(Instant::now()).into_iter()
            .map(|((ip, port), pkt)| (SocketAddrV4::new(ip, port), pkt))
            .collect()
    }
}

// ── Engine side ──────────────────────────────────────────────────────────────

fn dhcp_message_type(opts: &[u8]) -> Option<u8> {
    let mut i = 0;
    while i < opts.len() {
        match opts[i] {
            255 => break,
            0 => i += 1,
            tag => {
                let len = *opts.get(i + 1)? as usize;
                if tag == 53 && len >= 1 {
                    return opts.get(i + 2).copied();
                }
                i += 2 + len;
            }
        }
    }
    None
}

/// A NUL-terminated string field of a BOOTP packet.
fn bootp_str(field: &[u8]) -> String {
    let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    String::from_utf8_lossy(&field[..end]).into_owned()
}

impl NatEngine {
    /// A datagram for a service at the gateway; replies come from its port.
    pub(super) fn udp_service(&mut self, mac: MacAddr, src_ip: Ipv4Addr, sport: u16, dport: u16, data: &[u8]) {
        let Some(&i) = self.udp_service_ports.get(&dport) else { return };
        let peer = SocketAddrV4::new(src_ip, sport);
        self.service_peers.insert(peer, (mac, Instant::now()));
        for reply in self.udp_services[i].datagram(peer, data) {
            self.service_reply(mac, dport, peer, &reply);
        }
    }

    /// Timer work for every service (TFTP retransmissions).
    pub(super) fn poll_services(&mut self) {
        let mut due = Vec::new();
        for i in 0..self.udp_services.len() {
            for (peer, data) in self.udp_services[i].poll() {
                // Replies come from the service's (first) port.
                let port = self.udp_service_ports.iter().filter(|(_, &j)| j == i).map(|(&p, _)| p).min();
                due.extend(port.map(|p| (p, peer, data)));
            }
        }
        for (port, peer, data) in due {
            let Some(&(mac, _)) = self.service_peers.get(&peer) else { continue };
            self.service_reply(mac, port, peer, &data);
        }
        self.service_peers.retain(|_, (_, seen)| seen.elapsed() < PEER_IDLE);
    }

    fn service_reply(&mut self, mac: MacAddr, port: u16, peer: SocketAddrV4, data: &[u8]) {
        let id = self.next_ip_id();
        for f in self.route(mac, self.config.gateway_ip, *peer.ip()).udp_fragments(id, port, peer.port(), data) {
            self.send(f);
        }
    }

    // ── BOOTP / DHCP ─────────────────────────────────────────────────────────

    pub(super) fn handle_bootp(&mut self, client_mac: MacAddr, req: &[u8]) {
        if req.len() < 236 || req[0] != BOOTP_REQUEST {
            return;
        }
        let xid = &req[4..8];
        let chaddr = &req[28..34];
        let is_dhcp = req.len() >= 240 && req[236..240] == DHCP_MAGIC;
        let dhcp_type = if is_dhcp { dhcp_message_type(&req[240..]) } else { None };
        dlog_dev!(LogModule::Net, "NAT BOOTP xid={:02x?} mac={:02x?} dhcp_type={:?}", xid, chaddr, dhcp_type);
        let reply_type = match dhcp_type {
            Some(1) => Some(2), // DISCOVER → OFFER
            Some(3) => Some(5), // REQUEST → ACK
            None => None,
            _ => return,
        };

        let mut rep = vec![0u8; 300];
        rep[0] = 2;
        rep[1] = 1;
        rep[2] = 6;
        rep[4..8].copy_from_slice(xid);
        rep[16..20].copy_from_slice(&self.config.client_ip.octets()); // yiaddr
        rep[20..24].copy_from_slice(&self.config.gateway_ip.octets()); // siaddr
        rep[28..34].copy_from_slice(chaddr);
        self.bootp_boot_file(req, &mut rep);

        if is_dhcp {
            let mut opts: Vec<u8> = DHCP_MAGIC.to_vec();
            if let Some(t) = reply_type {
                opts.extend_from_slice(&[53, 1, t]);
            }
            // Not the host's DNS server: that is often loopback (a VPN or
            // systemd-resolved stub), which the guest would take as its own.
            // Guest queries to any address are relayed, so this still works.
            let dns = match self.config.dns_upstream.map(|a| a.ip()) {
                Some(IpAddr::V4(ip)) => ip,
                _ => DNS_FALLBACK,
            };
            for (tag, value) in [(1u8, self.config.netmask.octets()), (3, self.config.gateway_ip.octets()),
                                 (6, dns.octets()), (51, 86400u32.to_be_bytes()),
                                 (54, self.config.gateway_ip.octets())] {
                opts.extend_from_slice(&[tag, 4]);
                opts.extend_from_slice(&value);
            }
            opts.push(255);
            rep[236..236 + opts.len()].copy_from_slice(&opts);
        }

        // Address the reply as bootpd does: to ciaddr when the client already
        // has an address, else to yiaddr for plain BOOTP. The PROM drops any
        // IP packet not addressed to it once it has left promiscuous mode, so a
        // broadcast reply to its second BOOTP (a backward seek reopens the file)
        // is never seen. DHCP keeps the broadcast its clients already rely on.
        let ciaddr = Ipv4Addr::new(req[12], req[13], req[14], req[15]);
        let dst_ip = if !ciaddr.is_unspecified() { ciaddr }
                     else if is_dhcp { Ipv4Addr::BROADCAST }
                     else { self.config.client_ip };
        let id = self.next_ip_id();
        let frame = self.route(client_mac, self.config.gateway_ip, dst_ip)
            .udp(id, BOOTP_SERVER_PORT, BOOTP_CLIENT_PORT, &rep);
        self.send(frame);
    }

    /// Fill sname and file for a PROM network boot (`boot -f bootp()unix`).
    /// The PROM refuses a reply with an empty file ("File ... not found on
    /// server") and TFTPs exactly the name given back, so name the requested
    /// file whenever the TFTP root has it. With no TFTP, or no such file, the
    /// fields stay empty.
    fn bootp_boot_file(&self, req: &[u8], rep: &mut [u8]) {
        let trace = super::tftp::trace_enabled();
        let file = bootp_str(&req[108..236]);
        let Some(root) = self.config.tftp_dir.as_ref() else {
            if trace && !file.is_empty() {
                eprintln!("iris: tftp BOOTP asks for {:?}: no boot file named, tftp_dir is not set", file);
            }
            return;
        };
        if file.is_empty() || !super::tftp::file_exists(root, &file) {
            dlog_dev!(LogModule::Net, "NAT BOOTP file {:?} not in the TFTP root", file);
            if trace {
                eprintln!("iris: tftp BOOTP asks for {:?}: no boot file named, {}", file,
                          if file.is_empty() { "the request names no file".to_string() }
                          else { format!("it is not a file under {}", root.display()) });
            }
            return;
        }
        if trace {
            eprintln!("iris: tftp BOOTP asks for {:?}: named as the boot file", file);
        }
        // Echo a requested server name (bootp()host:file), else name ourselves.
        let sname = match bootp_str(&req[44..108]) {
            s if s.is_empty() => "iris".to_string(),
            s => s,
        };
        let n = sname.len().min(63);
        rep[44..44 + n].copy_from_slice(&sname.as_bytes()[..n]);
        rep[108..108 + file.len()].copy_from_slice(file.as_bytes()); // < 128: it came from a 128-byte field
    }

    // ── DNS ──────────────────────────────────────────────────────────────────
    //
    // Guest queries are relayed as ordinary UDP flows (nat_udp), sent to
    // `dns_upstream` and answered from the address the guest asked: a
    // resolver that checks (NetBSD's `res_send` does, IRIX's does not)
    // discards an answer from an address it never queried. We intercept port
    // 53 to *any* destination, so that is what keeps the interception
    // transparent.

    /// The resolver guest queries go to. Re-read periodically so turning a VPN
    /// on or off needs no restart.
    pub(super) fn dns_upstream(&mut self) -> SocketAddr {
        if let Some(addr) = self.config.dns_upstream {
            return addr;
        }
        if let Some((addr, read_at)) = self.host_dns {
            if read_at.elapsed() < HOST_DNS_REFRESH {
                return addr;
            }
        }
        let ip = super::host_dns::system_dns_server().unwrap_or(DNS_FALLBACK);
        let addr = SocketAddr::from((ip, DNS_PORT));
        if self.host_dns.map(|(old, _)| old) != Some(addr) {
            dlog_dev!(LogModule::Net, "NAT DNS upstream {}", addr);
        }
        self.host_dns = Some((addr, Instant::now()));
        addr
    }
}
