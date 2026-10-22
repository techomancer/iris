//! UDP: dispatch to the in-core services, otherwise relay through one host
//! socket per guest endpoint.
//!
//! The mapping is endpoint-independent (RFC 4787): a guest port talks to any
//! number of peers through the same host socket, and replies are presented
//! from wherever they really came from. Protocols that answer from a new port
//! (TFTP, some RPC) work. Where the guest addressed something the host knows
//! by another name (the gateway is host loopback, the DNS server the guest
//! asked is our upstream), an alias maps the real address back.

use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::time::{Duration, Instant};

use super::engine::NatEngine;
use super::packet::{MacAddr, Udp};
use super::services::{BOOTP_SERVER_PORT, DNS_PORT};
use crate::devlog::LogModule;

const UDP_IDLE: Duration = Duration::from_secs(60);

/// Let `sock` send any datagram the guest can send (up to 64 KB, reassembled
/// from fragments). macOS refuses a datagram larger than the socket's send
/// buffer with EMSGSIZE, and that buffer defaults to 9216 bytes
/// (`net.inet.udp.maxdgram`); Linux and Windows already allow 64 KB.
pub(super) fn allow_large_datagrams(sock: &UdpSocket) {
    let _ = socket2::SockRef::from(sock).set_send_buffer_size(65_535);
}

/// Guest UDP endpoint: (guest IP, guest port).
pub(super) type UdpKey = (u32, u16);

pub(super) struct NatUdpEntry {
    pub(super) sock: UdpSocket,
    client_mac: MacAddr,
    /// Real peer address → the address the guest sent to.
    aliases: HashMap<SocketAddrV4, SocketAddrV4>,
    /// The last destination, for status display.
    pub(super) last_dst: SocketAddrV4,
    pub(super) last_use: Instant,
}

impl NatEngine {
    pub(super) fn handle_udp(&mut self, src_mac: MacAddr, src_ip: Ipv4Addr, dst_ip: Ipv4Addr, udp: &Udp) {
        let (sport, dport, payload) = (udp.sport(), udp.dport(), udp.payload());
        dlog_dev!(LogModule::Net, "NAT UDP {}:{} → {}:{}", src_ip, sport, dst_ip, dport);
        let to_gw = dst_ip == self.config.gateway_ip;

        // A guest reply to an inbound UDP port forward: send it back to whoever
        // sent the forwarded datagram.
        if to_gw && self.udp_forward_reply(sport, dport, payload) {
            return;
        }

        let visible = SocketAddrV4::new(dst_ip, dport);
        match dport {
            BOOTP_SERVER_PORT => self.handle_bootp(src_mac, payload),
            // Any resolver the guest asks is answered by ours, from the
            // address it asked (see `forward_dns`'s note in services.rs).
            DNS_PORT => {
                let upstream = match self.dns_upstream() {
                    SocketAddr::V4(a) => a,
                    SocketAddr::V6(_) => return,
                };
                dlog_dev!(LogModule::Net, "NAT DNS {} via {}", dst_ip, upstream);
                self.udp_relay(src_mac, src_ip, sport, visible, upstream, payload);
            }
            _ if to_gw && self.udp_service_ports.contains_key(&dport) =>
                self.udp_service(src_mac, src_ip, sport, dport, payload),
            _ => {
                // A read request that is not ours: say where it goes, since
                // "no TFTP trace" otherwise looks the same as "no request".
                if dport == super::services::TFTP_PORT && payload.starts_with(&[0, 1]) && super::tftp::trace_enabled() {
                    eprintln!("iris: tftp: guest {}:{} -> {}:{} read request {}",
                              src_ip, sport, dst_ip, dport,
                              if to_gw { "relayed to the host's port 69: tftp_dir is not set" }
                              else { "relayed to that address: it is not the gateway" });
                }
                let (ip, port) = self.gateway_to_host(dst_ip, dport);
                self.udp_relay(src_mac, src_ip, sport, visible, SocketAddrV4::new(ip, port), payload);
            }
        }
    }

    /// Send a guest datagram to `real`, remembering that the guest knows that
    /// peer as `visible`.
    fn udp_relay(&mut self, client_mac: MacAddr, src_ip: Ipv4Addr, sport: u16,
                 visible: SocketAddrV4, real: SocketAddrV4, payload: &[u8]) {
        let key: UdpKey = (u32::from(src_ip), sport);
        if !self.udp_nat.contains_key(&key) {
            let Ok(sock) = UdpSocket::bind("0.0.0.0:0") else {
                dlog_dev!(LogModule::Net, "NAT UDP: no host socket for guest :{}", sport);
                return;
            };
            let _ = sock.set_nonblocking(true);
            allow_large_datagrams(&sock);
            dlog_dev!(LogModule::Net, "NAT UDP new flow guest {}:{} via host {:?}", src_ip, sport, sock.local_addr());
            self.udp_nat.insert(key, NatUdpEntry { sock, client_mac, aliases: HashMap::new(),
                                                   last_dst: visible, last_use: Instant::now() });
        }
        let entry = self.udp_nat.get_mut(&key).unwrap();
        if real != visible {
            entry.aliases.insert(real, visible);
        }
        entry.client_mac = client_mac;
        entry.last_dst = visible;
        entry.last_use = Instant::now();
        dlog_dev!(LogModule::Net, "NAT UDP {}:{} → {} (as {}) len={}", src_ip, sport, real, visible, payload.len());
        if let Err(e) = entry.sock.send_to(payload, real) {
            dlog_dev!(LogModule::Net, "NAT UDP {}:{} → {}: send of {} bytes failed: {}", src_ip, sport, real, payload.len(), e);
        }
    }

    pub(super) fn poll_udp(&mut self) {
        self.udp_nat.retain(|key, e| {
            let live = e.last_use.elapsed() <= UDP_IDLE;
            if !live {
                dlog_dev!(LogModule::Net, "NAT UDP guest :{} expired", key.1);
            }
            live
        });
        let mut replies: Vec<(UdpKey, MacAddr, SocketAddrV4, Vec<u8>)> = Vec::new();
        let mut buf = vec![0u8; 65536];
        for (&key, entry) in &mut self.udp_nat {
            while let Ok((n, from)) = entry.sock.recv_from(&mut buf) {
                let SocketAddr::V4(from) = from else { continue };
                let shown = entry.aliases.get(&from).copied().unwrap_or(from);
                dlog_dev!(LogModule::Net, "NAT UDP reply {} (as {}) → guest :{} len={}", from, shown, key.1, n);
                entry.last_use = Instant::now();
                replies.push((key, entry.client_mac, shown, buf[..n].to_vec()));
            }
        }
        for ((guest_ip, guest_port), mac, from, data) in replies {
            // Anything from host loopback not otherwise aliased is the gateway.
            let (src_ip, src_port) = self.host_to_gateway(*from.ip(), from.port());
            let id = self.next_ip_id();
            for f in self.route(mac, src_ip, Ipv4Addr::from(guest_ip)).udp_fragments(id, src_port, guest_port, &data) {
                self.send(f);
            }
        }
    }

    /// The host address a guest packet to `dst_ip:dport` really goes to. The
    /// gateway is virtual: traffic to it goes to host loopback on the same port
    /// (the guest reaches any host service on 127.0.0.1), except an XDMCP
    /// session's X11 port, which goes to the real X server the ALG recorded.
    pub(super) fn gateway_to_host(&self, dst_ip: Ipv4Addr, dport: u16) -> (Ipv4Addr, u16) {
        if dst_ip != self.config.gateway_ip {
            return (dst_ip, dport);
        }
        if let Some(&xserver) = self.xdmcp_sessions.get(&dport) {
            return (xserver, dport);
        }
        (Ipv4Addr::LOCALHOST, dport)
    }

    /// The reverse of `gateway_to_host`: what the guest should see as the source.
    pub(super) fn host_to_gateway(&self, src_ip: Ipv4Addr, sport: u16) -> (Ipv4Addr, u16) {
        if let Some(&xserver) = self.xdmcp_sessions.get(&sport) {
            if src_ip == xserver {
                return (self.config.gateway_ip, sport);
            }
        }
        if src_ip != Ipv4Addr::LOCALHOST {
            return (src_ip, sport);
        }
        (self.config.gateway_ip, sport)
    }
}
