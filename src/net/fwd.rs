//! Inbound port forwards: host listeners whose connections and datagrams are
//! injected into the guest as if they came from the gateway.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream, UdpSocket};

use super::engine::NatEngine;
use super::tcp::Endpoints;
use crate::config::{ForwardBind, ForwardProto, PortForwardConfig};
use crate::devlog::LogModule;

/// XDMCP control channel inside the guest.
const XDMCP_GUEST_PORT: u16 = 177;
/// X11 display N's TCP port is 6000 + N.
const X11_BASE_PORT: u16 = 6000;
/// Transient FTP-ALG data forwards kept at once (oldest dropped first).
const MAX_FTP_DATA_FWD: usize = 16;

pub(super) struct TcpFwdListener {
    pub(super) listener: TcpListener,
    pub(super) guest_port: u16,
}

pub(super) struct UdpFwdListener {
    pub(super) sock: UdpSocket,
    guest_port: u16,
    host_port: u16,
    /// Most recent sender; the guest's replies go back here.
    last_sender: Option<SocketAddr>,
}

/// Bind host listeners for a set of forward rules (at construction, and when
/// forwards are reconfigured live).
pub(super) fn bind_forwards(rules: &[PortForwardConfig]) -> (Vec<TcpFwdListener>, Vec<UdpFwdListener>) {
    let mut tcp = Vec::new();
    let mut udp = Vec::new();
    for rule in rules {
        let bind_addr = match rule.bind {
            ForwardBind::Localhost => Ipv4Addr::LOCALHOST,
            ForwardBind::Any => Ipv4Addr::UNSPECIFIED,
        };
        let addr = SocketAddr::new(IpAddr::V4(bind_addr), rule.host_port);
        match rule.proto {
            ForwardProto::Tcp => match TcpListener::bind(addr) {
                Ok(listener) => {
                    let _ = listener.set_nonblocking(true);
                    eprintln!("iris: TCP port forward {}:{} → guest:{}", bind_addr, rule.host_port, rule.guest_port);
                    tcp.push(TcpFwdListener { listener, guest_port: rule.guest_port });
                }
                Err(e) => eprintln!("iris: TCP port forward {}:{} failed to bind: {}", bind_addr, rule.host_port, e),
            },
            ForwardProto::Udp => match UdpSocket::bind(addr) {
                Ok(sock) => {
                    let _ = sock.set_nonblocking(true);
                    super::nat_udp::allow_large_datagrams(&sock);
                    eprintln!("iris: UDP port forward {}:{} → guest:{}", bind_addr, rule.host_port, rule.guest_port);
                    udp.push(UdpFwdListener { sock, guest_port: rule.guest_port, host_port: rule.host_port,
                                              last_sender: None });
                }
                Err(e) => eprintln!("iris: UDP port forward {}:{} failed to bind: {}", bind_addr, rule.host_port, e),
            },
        }
    }
    (tcp, udp)
}

impl NatEngine {
    /// Rebind the static forward listeners from a new rule set, live. The old
    /// listeners close; transient FTP-ALG data forwards and established
    /// connections are kept.
    pub(super) fn rebind_forwards(&mut self, rules: &[PortForwardConfig]) {
        let cut = self.fwd_static_count.min(self.tcp_fwd_listeners.len());
        let transient = self.tcp_fwd_listeners.split_off(cut);
        let (mut tcp, udp) = bind_forwards(rules);
        self.fwd_static_count = tcp.len();
        tcp.extend(transient);
        self.tcp_fwd_listeners = tcp;
        self.udp_fwd_listeners = udp;
    }

    /// Register a transient forward for an FTP passive data connection.
    pub(super) fn add_ftp_data_forward(&mut self, listener: TcpListener, guest_port: u16) {
        while self.tcp_fwd_listeners.len() >= self.fwd_static_count + MAX_FTP_DATA_FWD
            && self.tcp_fwd_listeners.len() > self.fwd_static_count
        {
            self.tcp_fwd_listeners.remove(self.fwd_static_count); // drop the oldest
        }
        self.tcp_fwd_listeners.push(TcpFwdListener { listener, guest_port });
    }

    /// Pick a source port for a synthesized inbound connection that isn't live
    /// for this guest port. rsh/rlogin (513/514) need a reserved 512..=1023
    /// port or the guest's daemons reject them; everything else uses
    /// 49152..=65535. Probing past ports in use keeps a wrapped counter from
    /// overwriting a live entry. None if the range is exhausted.
    fn alloc_fwd_sport(&mut self, guest_port: u16) -> Option<u16> {
        self.alloc_sport(guest_port, matches!(guest_port, 513 | 514))
    }

    /// `alloc_fwd_sport` with the range chosen by the caller: the gateway's own
    /// connections back to the guest (rsh's stderr channel) must come from a
    /// reserved port whatever port the guest listens on.
    pub(super) fn alloc_sport(&mut self, guest_port: u16, reserved: bool) -> Option<u16> {
        let (lo, hi) = if reserved { (512u16, 1023u16) } else { (49152u16, 65535u16) };
        let gw = u32::from(self.config.gateway_ip);
        let mut cur = if reserved { self.fwd_reserved_next } else { self.fwd_ephemeral_next };
        if cur < lo || cur > hi {
            cur = lo;
        }
        let mut chosen = None;
        for _ in 0..=(hi - lo) {
            let p = cur;
            cur = if p >= hi { lo } else { p + 1 };
            if !self.tcp_nat.contains_key(&(gw, p, guest_port)) {
                chosen = Some(p);
                break;
            }
        }
        if reserved { self.fwd_reserved_next = cur; } else { self.fwd_ephemeral_next = cur; }
        chosen
    }

    /// Accept host connections on the forward listeners and open each toward
    /// the guest with a SYN from gateway:ephemeral.
    pub(super) fn poll_tcp_fwd_listeners(&mut self) {
        let mut accepted: Vec<(TcpStream, u16)> = Vec::new();
        for fwd in &self.tcp_fwd_listeners {
            loop {
                match fwd.listener.accept() {
                    Ok((stream, peer)) => {
                        dlog_dev!(LogModule::Net, "NAT FWD TCP accepted {} → guest:{}", peer, fwd.guest_port);
                        accepted.push((stream, fwd.guest_port));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(e) => { dlog_dev!(LogModule::Net, "NAT FWD TCP accept error: {}", e); break; }
                }
            }
        }
        for (stream, guest_port) in accepted {
            // The guest's actual IP (learned from its traffic), not the assumed
            // client address: a statically configured guest can be anywhere.
            let guest_ip = self.ctl.observed_guest_ip().unwrap_or(self.config.client_ip);
            // The SYN must be unicast to the guest's MAC, or IRIX drops it.
            let Some(guest_mac) = self.guest_mac else {
                dlog_dev!(LogModule::Net, "NAT FWD TCP: guest MAC not yet known, dropping accept for guest:{}", guest_port);
                continue;
            };
            let Some(ephemeral) = self.alloc_fwd_sport(guest_port) else {
                dlog_dev!(LogModule::Net, "NAT FWD TCP: no free source port for guest:{}, dropping accept", guest_port);
                continue;
            };
            dlog_dev!(LogModule::Net, "NAT FWD TCP SYN {}:{} → {}:{}", self.config.gateway_ip, ephemeral, guest_ip, guest_port);
            let ends = Endpoints { guest_mac, guest_ip, guest_port,
                                   remote_ip: self.config.gateway_ip, remote_port: ephemeral };
            self.tcp_open_to_guest(ends, stream);
        }
    }

    /// Inject datagrams arriving on the UDP forward sockets into the guest,
    /// from gateway:host_port, so the guest can reply there.
    pub(super) fn poll_udp_fwd_listeners(&mut self) {
        let mut received: Vec<(Vec<u8>, SocketAddr, usize)> = Vec::new();
        let mut buf = vec![0u8; 65536];
        for (i, fwd) in self.udp_fwd_listeners.iter().enumerate() {
            loop {
                match fwd.sock.recv_from(&mut buf) {
                    Ok((n, from)) => {
                        dlog_dev!(LogModule::Net, "NAT FWD UDP {} → guest:{} len={}", from, fwd.guest_port, n);
                        received.push((buf[..n].to_vec(), from, i));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(e) => { dlog_dev!(LogModule::Net, "NAT FWD UDP recv error: {}", e); break; }
                }
            }
        }
        for (data, from, i) in received {
            let Some(guest_mac) = self.guest_mac else {
                dlog_dev!(LogModule::Net, "NAT FWD UDP: guest MAC not yet known, dropping datagram");
                continue;
            };
            let fwd = &mut self.udp_fwd_listeners[i];
            fwd.last_sender = Some(from);
            let (guest_port, host_port) = (fwd.guest_port, fwd.host_port);
            let guest_ip = self.ctl.observed_guest_ip().unwrap_or(self.config.client_ip);
            let gw_ip = self.config.gateway_ip;
            // XDMCP ALG: on the control forward, rewrite a Request's connection
            // addresses to the gateway so the guest's xdm opens the X11 session
            // at gateway:(6000+display), which `gateway_to_host` then relays to
            // the real X server (the datagram's source) recorded here.
            let data = if guest_port == XDMCP_GUEST_PORT {
                match (super::xdmcp::rewrite_request_ipv4(&data, gw_ip), from.ip()) {
                    (Some(rw), IpAddr::V4(xserver)) => {
                        let x11_port = X11_BASE_PORT.wrapping_add(rw.display_number);
                        self.xdmcp_sessions.insert(x11_port, xserver);
                        dlog_dev!(LogModule::Net, "XDMCP Request: display {} → relay gateway:{} to X server {}",
                                  rw.display_number, x11_port, xserver);
                        rw.packet
                    }
                    _ => data,
                }
            } else {
                data
            };
            let id = self.next_ip_id();
            for f in self.route(guest_mac, gw_ip, guest_ip).udp_fragments(id, host_port, guest_port, &data) {
                self.send(f);
            }
        }
    }

    /// A guest datagram to gateway:host_port from a forwarded guest_port is a
    /// reply: send it to the forward's last sender. True if it was one.
    pub(super) fn udp_forward_reply(&mut self, sport: u16, dport: u16, payload: &[u8]) -> bool {
        let Some(fwd) = self.udp_fwd_listeners.iter().find(|f| f.host_port == dport && f.guest_port == sport) else {
            return false;
        };
        if let Some(sender) = fwd.last_sender {
            dlog_dev!(LogModule::Net, "NAT FWD UDP reply guest:{} → host:{} → {}", sport, dport, sender);
            let _ = fwd.sock.send_to(payload, sender);
        }
        true
    }
}
