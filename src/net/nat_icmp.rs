//! ICMP: echo to the gateway answered locally, echo to anywhere else relayed
//! through a host ICMP socket per (destination, identifier).

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::{Duration, Instant};

use socket2::{Domain, Protocol, Socket, Type};

use super::engine::NatEngine;
use super::packet::{self, Icmp, MacAddr};
use super::wake::Watched;
use crate::devlog::LogModule;

const ICMP_IDLE: Duration = Duration::from_secs(30);

pub(super) struct NatIcmpEntry {
    pub(super) sock: Option<Watched<Socket>>, // None if socket creation failed (e.g. not admin on Windows)
    client_mac: MacAddr,
    client_ip: Ipv4Addr,
    pub(super) last_use: Instant,
}

/// Linux: unprivileged SOCK_DGRAM+ICMPV4 works (kernel ≥3.11) but Time Exceeded
///   replies are not delivered, so traceroute sees * * * for intermediate hops.
/// macOS: unprivileged SOCK_DGRAM works for any user and, unlike Linux, recv
///   yields the full IP header and delivers Time Exceeded; SOCK_RAW (root) is
///   the fallback.
/// Windows: SOCK_RAW+ICMPV4 needs admin; Time Exceeded is delivered on recv.
fn icmp_socket() -> std::io::Result<Socket> {
    let new = |t| Socket::new(Domain::IPV4, t, Some(Protocol::ICMPV4));
    #[cfg(target_os = "linux")]
    { new(Type::DGRAM) }
    #[cfg(target_os = "macos")]
    { new(Type::DGRAM).or_else(|_| new(Type::RAW)) }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { new(Type::RAW) }
}

impl NatEngine {
    pub(super) fn handle_icmp(&mut self, src_mac: MacAddr, src_ip: Ipv4Addr, dst_ip: Ipv4Addr,
                              ttl: u8, icmp: &Icmp) {
        if icmp.ty() != packet::ICMP_ECHO_REQUEST {
            return;
        }
        let (ident, seq) = (icmp.ident(), icmp.seq());

        if dst_ip == self.config.gateway_ip {
            dlog_dev!(LogModule::Net, "NAT ICMP {} → {} ident={} seq={} (local reply)", src_ip, dst_ip, ident, seq);
            let id = self.next_ip_id();
            let frame = self.route(src_mac, dst_ip, src_ip)
                .icmp(id, packet::ICMP_ECHO_REPLY, 0, icmp.rest(), icmp.body());
            self.send(frame);
            return;
        }

        let key = (u32::from(dst_ip), ident);
        dlog_dev!(LogModule::Net, "NAT ICMP {} → {} ident={} seq={} ttl={}{}", src_ip, dst_ip, ident, seq, ttl,
            if self.icmp_nat.contains_key(&key) { "" } else { " [new]" });
        if self.icmp_unavailable {
            return;
        }
        let wake = &self.tx_wake;
        let entry = self.icmp_nat.entry(key).or_insert_with(|| {
            let sock = icmp_socket().and_then(|s| {
                let _ = s.set_nonblocking(true);
                Watched::new(s, wake, true, false)
            });
            let sock = match sock {
                Ok(s) => Some(s),
                Err(e) => {
                    #[cfg(windows)]
                    eprintln!("iris: ICMP unavailable ({}); ping will time out. \
                        Run as Administrator to enable raw ICMP.", e);
                    #[cfg(not(windows))]
                    eprintln!("iris: ICMP unavailable ({}); ping will time out.", e);
                    None
                }
            };
            NatIcmpEntry { sock, client_mac: src_mac, client_ip: src_ip, last_use: Instant::now() }
        });
        let Some(sock) = entry.sock.as_ref() else {
            self.icmp_unavailable = true;
            return;
        };
        entry.last_use = Instant::now();
        // Keep the guest's TTL so routers answer Time Exceeded at the right hop.
        let _ = sock.set_ttl_v4(ttl as u32);
        let _ = sock.send_to(icmp.bytes(), &SocketAddr::new(IpAddr::V4(dst_ip), 0).into());
    }

    pub(super) fn poll_icmp(&mut self) {
        self.icmp_nat.retain(|_, e| e.last_use.elapsed() <= ICMP_IDLE);
        // (icmp message, outer source, key)
        let mut replies: Vec<(Vec<u8>, Ipv4Addr, (u32, u16))> = Vec::new();
        for (&key, entry) in &mut self.icmp_nat {
            let Some(sock) = &mut entry.sock else { continue };
            let mut buf = [std::mem::MaybeUninit::<u8>::uninit(); 1500];
            while let Ok(n) = sock.recv(&mut buf) {
                // SAFETY: recv initialised the first n bytes.
                let raw: Vec<u8> = buf[..n].iter().map(|b| unsafe { b.assume_init() }).collect();
                // Linux (SOCK_DGRAM) delivers only the ICMP message; Windows
                // (SOCK_RAW) and macOS prepend the outer IP header.
                #[cfg(not(target_os = "linux"))]
                let Some(ip) = packet::Ipv4::parse(&raw) else { continue };
                #[cfg(not(target_os = "linux"))]
                let (outer_src, icmp) = (ip.src(), ip.payload().to_vec());
                #[cfg(target_os = "linux")]
                let (outer_src, icmp) = (Ipv4Addr::from(key.0), raw);
                replies.push((icmp, outer_src, key));
            }
        }
        for (icmp, outer_src, key) in replies {
            let Some(msg) = Icmp::parse(&icmp) else { continue };
            // Time Exceeded (non-Linux only): find the flow from the probe
            // embedded in it, rewrite its source back to the guest.
            #[cfg(not(target_os = "linux"))]
            if msg.ty() == packet::ICMP_TIME_EXCEEDED {
                let Some(emb) = packet::Ipv4::parse(msg.body()) else { continue };
                let Some(probe) = Icmp::parse(&msg.body()[emb.header_len()..]) else { continue };
                let Some(entry) = self.icmp_nat.get(&(u32::from(emb.dst()), probe.ident())) else { continue };
                let (client_mac, client_ip) = (entry.client_mac, entry.client_ip);
                let mut body = msg.body().to_vec();
                body[12..16].copy_from_slice(&client_ip.octets());
                dlog_dev!(LogModule::Net, "NAT ICMP TimeExceeded {} → {} ident={}", outer_src, client_ip, probe.ident());
                let id = self.next_ip_id();
                let frame = self.route(client_mac, outer_src, client_ip)
                    .icmp(id, msg.ty(), msg.code(), msg.rest(), &body);
                self.send(frame);
                continue;
            }
            // Echo reply: restore the guest's identifier (the host kernel may
            // have rewritten it).
            let Some(entry) = self.icmp_nat.get(&key) else { continue };
            let (client_mac, client_ip) = (entry.client_mac, entry.client_ip);
            let mut rest = msg.rest();
            rest[..2].copy_from_slice(&key.1.to_be_bytes());
            dlog_dev!(LogModule::Net, "NAT ICMP reply {} → {} ident={} seq={}", outer_src, client_ip, key.1, msg.seq());
            let id = self.next_ip_id();
            let frame = self.route(client_mac, outer_src, client_ip).icmp(id, msg.ty(), msg.code(), rest, msg.body());
            self.send(frame);
        }
    }
}
