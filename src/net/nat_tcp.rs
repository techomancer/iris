//! TCP flows: each guest connection is a [`TcpConn`] whose backend is a host
//! socket (outbound NAT or an inbound forward) or an in-core service.
//!
//! Flows are keyed by what the guest sees: (remote IP, remote port, guest
//! port), the remote being the gateway for anything relayed to host loopback.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};

use super::engine::{FlowKey, NatEngine};
use super::packet::{MacAddr, Tcp, TcpFlags, TcpHeader};
use super::portmap::PORTMAP_PORT;
use super::rsh::{RshSession, RSH_PORT};
use super::services::TIME_PORT;
use super::tcp::{Backend, Endpoints, HostConn, PortmapService, ServiceConn, SideConnect, State, TcpConn, TimeService};
use crate::devlog::LogModule;

/// The guest's FTP server port, for the passive-mode ALG on inbound forwards.
const FTP_CONTROL_PORT: u16 = 21;

/// FTP application-layer gateway (passive mode). If `payload`, server→client
/// bytes on an FTP control connection, contains a `227 Entering Passive Mode
/// (h1,h2,h3,h4,p1,p2)` reply, rewrite the address tuple to advertise
/// `host`:`host_port` and return `(rewritten_payload, guest_data_port)`.
///
/// The length-changing rewrite is safe because the NAT relays the byte stream
/// between two independent TCP connections, so no seq/ack surgery is needed.
pub(super) fn ftp_pasv_rewrite(payload: &[u8], host: Ipv4Addr, host_port: u16) -> Option<(Vec<u8>, u16)> {
    // "227" at the start of a line.
    let pos = (0..payload.len().saturating_sub(2))
        .find(|&i| &payload[i..i + 3] == b"227" && (i == 0 || payload[i - 1] == b'\n'))?;
    let open = pos + payload[pos..].iter().position(|&b| b == b'(')?;
    let close = open + payload[open..].iter().position(|&b| b == b')')?;
    // Exactly six 0..=255 numbers between the parentheses.
    let nums = payload[open + 1..close]
        .split(|&b| b == b',')
        .map(|s| std::str::from_utf8(s).ok().and_then(|t| t.trim().parse::<u16>().ok()))
        .collect::<Option<Vec<u16>>>()?;
    if nums.len() != 6 || nums.iter().any(|&n| n > 255) {
        return None;
    }
    let data_port = (nums[4] << 8) | nums[5];
    let o = host.octets();
    let replacement = format!("({},{},{},{},{},{})", o[0], o[1], o[2], o[3], host_port >> 8, host_port & 0xff);
    let mut out = Vec::with_capacity(payload.len() + replacement.len());
    out.extend_from_slice(&payload[..open]);
    out.extend_from_slice(replacement.as_bytes());
    out.extend_from_slice(&payload[close + 1..]);
    Some((out, data_port))
}

/// The ALG hook for guest→host bytes on an inbound FTP control forward: a
/// passive reply gets a fresh host listener for its data connection.
fn ftp_alg(data: &[u8]) -> Option<(Vec<u8>, TcpListener, u16)> {
    ftp_pasv_rewrite(data, Ipv4Addr::LOCALHOST, 0)?;
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).ok()?;
    let port = listener.local_addr().ok()?.port();
    let _ = listener.set_nonblocking(true);
    let (rewritten, data_port) = ftp_pasv_rewrite(data, Ipv4Addr::LOCALHOST, port)?;
    dlog_dev!(LogModule::Net, "NAT FTP ALG: PASV guest data port {} -> host 127.0.0.1:{}", data_port, port);
    Some((rewritten, listener, data_port))
}

impl NatEngine {
    pub(super) fn handle_tcp(&mut self, client_mac: MacAddr, src_ip: Ipv4Addr, dst_ip: Ipv4Addr, tcp: &Tcp) {
        let key: FlowKey = (u32::from(dst_ip), tcp.dport(), tcp.sport());
        let flags = tcp.flags();
        let gw_mac = self.config.gateway_mac;

        if flags.syn() && !flags.ack() {
            // A SYN on a port pair whose old connection is finished starts afresh.
            if self.tcp_nat.get(&key).is_some_and(|c| c.state == State::Closed) {
                self.tcp_nat.remove(&key);
            }
            if !self.tcp_nat.contains_key(&key) {
                self.tcp_open(client_mac, src_ip, dst_ip, tcp, key);
                return;
            }
        }

        let Some(conn) = self.tcp_nat.get_mut(&key) else {
            if !flags.rst() {
                dlog_dev!(LogModule::Net, "NAT TCP no connection for {}:{} → {}:{} flags={:?}, sending RST",
                          src_ip, tcp.sport(), dst_ip, tcp.dport(), flags);
                self.tcp_reset(client_mac, src_ip, dst_ip, tcp);
            }
            return;
        };

        // FTP ALG on an inbound forward to the guest's ftpd: rewrite a passive
        // reply that arrives in one in-order segment.
        let alg = conn.backend().is_host() && conn.ends.remote_ip == self.config.gateway_ip
            && conn.ends.guest_port == FTP_CONTROL_PORT && tcp.seq() == conn.rcv_nxt;
        let frames = match alg.then(|| ftp_alg(tcp.payload())).flatten() {
            Some((rewritten, listener, data_port)) => {
                let frames = conn.input_rewritten(gw_mac, tcp, &rewritten);
                self.add_ftp_data_forward(listener, data_port);
                frames
            }
            None => conn.input(gw_mac, tcp),
        };
        for f in frames {
            self.send(f);
        }
    }

    /// A RST for a segment that belongs to no connection.
    fn tcp_reset(&mut self, mac: MacAddr, src_ip: Ipv4Addr, dst_ip: Ipv4Addr, tcp: &Tcp) {
        let flags = tcp.flags();
        let h = TcpHeader {
            sport: tcp.dport(), dport: tcp.sport(),
            seq: if flags.ack() { tcp.ack() } else { 0 },
            ack: tcp.seq().wrapping_add(tcp.seq_len()),
            flags: TcpFlags::RST_ACK, window: 0, mss: None,
        };
        let id = self.next_ip_id();
        let frame = self.route(mac, dst_ip, src_ip).tcp(id, &h, &[]);
        self.send(frame);
    }

    /// The guest's SYN: pick the backend and start the connection. The SYN-ACK
    /// (or RST) goes out from `poll_tcp` once the backend has connected (or
    /// failed), so a slow or black-holed host never blocks the engine.
    fn tcp_open(&mut self, client_mac: MacAddr, src_ip: Ipv4Addr, dst_ip: Ipv4Addr, syn: &Tcp, key: FlowKey) {
        let dport = syn.dport();
        let ends = Endpoints { guest_mac: client_mac, guest_ip: src_ip, guest_port: syn.sport(),
                               remote_ip: dst_ip, remote_port: dport };
        let to_gw = dst_ip == self.config.gateway_ip;
        if dport == RSH_PORT && super::rsh::trace_enabled() {
            // Where did this guest rsh go? "No trace from the built-in server"
            // usually means it never got the connection.
            let verdict = if to_gw && self.config.rsh_dir.is_some() {
                "answered by the built-in rsh server"
            } else if to_gw {
                "relayed to the host's port 514: rsh_dir is not set"
            } else {
                "relayed to that address: it is not the gateway"
            };
            eprintln!("iris: rsh: guest {}:{} -> {}:{}: {}", src_ip, syn.sport(), dst_ip, dport, verdict);
        }
        let backend = if to_gw && dport == TIME_PORT {
            Backend::Service(ServiceConn::new(Box::new(TimeService(super::services::rfc868_time()))))
        } else if to_gw && dport == PORTMAP_PORT && self.config.nfs.is_some() {
            Backend::Service(ServiceConn::new(Box::new(PortmapService::default())))
        } else if let Some(root) = self.config.rsh_dir.as_ref().filter(|_| to_gw && dport == RSH_PORT) {
            Backend::Service(ServiceConn::new(Box::new(RshSession::new(root.clone()))))
        } else {
            let (host_ip, host_port) = self.gateway_to_host(dst_ip, dport);
            dlog_dev!(LogModule::Net, "NAT TCP connect {}:{} → {}:{}", src_ip, syn.sport(), host_ip, host_port);
            match HostConn::connect(SocketAddr::new(IpAddr::V4(host_ip), host_port)) {
                Ok(h) => Backend::Host(h),
                Err(e) => {
                    dlog_dev!(LogModule::Net, "NAT TCP connect {}:{} failed: {}", host_ip, host_port, e);
                    self.tcp_reset(client_mac, src_ip, dst_ip, syn);
                    return;
                }
            }
        };
        self.tcp_nat.insert(key, TcpConn::accept(ends, syn, backend));
    }

    /// Advance every connection: connects completing, data from backends,
    /// retransmissions, window updates, closes.
    pub(super) fn poll_tcp(&mut self) {
        let gw_mac = self.config.gateway_mac;
        let mut frames = Vec::new();
        let mut budget = self.queue_room();
        let mut side_connects = Vec::new();
        for conn in self.tcp_nat.values_mut() {
            let out = conn.poll(gw_mac, budget);
            budget = budget.saturating_sub(out.len());
            frames.extend(out);
            // A service (rsh) may have just asked for a connection back to the guest.
            if let Some(side) = conn.take_side_connect() {
                side_connects.push((conn.ends, side));
            }
        }
        self.tcp_nat.retain(|key, c| {
            let keep = !c.is_done();
            if !keep {
                dlog_dev!(LogModule::Net, "NAT TCP closed {}:{} ↔ guest :{}", Ipv4Addr::from(key.0), key.1, key.2);
            }
            keep
        });
        for f in frames {
            self.send(f);
        }
        for (main, side) in side_connects {
            self.open_side_connection(main, side);
        }
    }

    /// Open the connection a service asked for toward the guest: from the
    /// gateway address the guest connected to, at the port the service named.
    /// rsh's stderr channel is the case: `rcmd` waits to accept it, and checks
    /// that it comes from a reserved port.
    fn open_side_connection(&mut self, main: Endpoints, side: SideConnect) {
        let Some(sport) = self.alloc_sport(side.guest_port, true) else {
            dlog_dev!(LogModule::Net, "NAT TCP: no reserved source port for a connection to guest :{}", side.guest_port);
            return;
        };
        dlog_dev!(LogModule::Net, "NAT TCP side connection {}:{} → guest {}:{}",
                  main.remote_ip, sport, main.guest_ip, side.guest_port);
        if super::rsh::trace_enabled() {
            eprintln!("iris: rsh: connecting back to guest {}:{} (stderr) from {}:{}",
                      main.guest_ip, side.guest_port, main.remote_ip, sport);
        }
        let ends = Endpoints { guest_port: side.guest_port, remote_port: sport, ..main };
        let conn = TcpConn::open(ends, Backend::Service(ServiceConn::new(side.service)));
        let syn = conn.syn(self.config.gateway_mac);
        self.tcp_nat.insert((u32::from(ends.remote_ip), ends.remote_port, ends.guest_port), conn);
        self.send(syn);
    }

    /// Open a connection toward the guest for an accepted forward.
    pub(super) fn tcp_open_to_guest(&mut self, ends: Endpoints, stream: std::net::TcpStream) {
        let conn = TcpConn::open(ends, Backend::Host(HostConn::connected(stream)));
        let syn = conn.syn(self.config.gateway_mac);
        self.tcp_nat.insert((u32::from(ends.remote_ip), ends.remote_port, ends.guest_port), conn);
        self.send(syn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_pasv_reply() {
        let p = b"227 Entering Passive Mode (192,168,0,2,200,21).\r\n";
        let (out, port) = ftp_pasv_rewrite(p, Ipv4Addr::new(127, 0, 0, 1), 50000).unwrap();
        assert_eq!(port, 200 * 256 + 21);
        let s = std::str::from_utf8(&out).unwrap();
        assert!(s.contains("(127,0,0,1,195,80)"), "got: {s}"); // 50000 = 195*256+80
        assert!(s.starts_with("227 Entering Passive Mode"));
        assert!(s.ends_with(".\r\n"));
    }

    #[test]
    fn ignores_non_pasv() {
        assert!(ftp_pasv_rewrite(b"USER anonymous\r\n", Ipv4Addr::LOCALHOST, 1).is_none());
        assert!(ftp_pasv_rewrite(b"200 PORT command successful.\r\n", Ipv4Addr::LOCALHOST, 1).is_none());
        // "227" not at a line start must not match.
        assert!(ftp_pasv_rewrite(b"x227 (1,2,3,4,5,6)\r\n", Ipv4Addr::LOCALHOST, 1).is_none());
        // A malformed tuple (not six bytes) is rejected.
        assert!(ftp_pasv_rewrite(b"227 (1,2,3,4,5)\r\n", Ipv4Addr::LOCALHOST, 1).is_none());
        assert!(ftp_pasv_rewrite(b"227 (1,2,3,4,5,999)\r\n", Ipv4Addr::LOCALHOST, 1).is_none());
    }
}
