//! The NAT engine: the guest's gateway. Owns every flow table and service and
//! runs on the `seeq-net` thread (or another adapter's), taking guest frames from
//! one ring and putting replies on another.
//!
//! Protocol handling lives in sibling modules, each an `impl NatEngine`:
//! `nat_tcp`, `nat_udp`, `nat_icmp`, `fwd` (inbound port forwards) and
//! `services` (DHCP, DNS, NFS, portmap, TFTP, time).

use std::collections::{HashMap, VecDeque};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};


use super::fwd::{bind_forwards, TcpFwdListener, UdpFwdListener};
use super::nat_icmp::NatIcmpEntry;
use super::nat_udp::{NatUdpEntry, UdpKey};
use super::tcp::TcpConn;
use super::packet::{self, Ether, Ipv4Route, MacAddr};
use super::services::UdpService;
use super::wake::Wake;
use super::{GatewayConfig, NatControl, NatIcmpInfo, NatTcpInfo, NatUdpInfo, NetBackend};
use crate::devlog::LogModule;

/// TCP and UDP flows are keyed by (remote IP, remote port, guest port).
pub(super) type FlowKey = (u32, u16, u16);

/// Frames queued toward the guest beyond which producers hold back (TCP stops
/// reading host sockets). Datagram replies are never dropped for it.
const QUEUE_SOFT_LIMIT: usize = 192;
/// The status snapshot (`net status`, the GUI) is for people: refresh it at
/// this rate, not every pass.
const SNAPSHOT_EVERY: Duration = Duration::from_millis(200);
/// Longest sleep with nothing to do: the coarse timers (UDP and service
/// expiry, TFTP retransmission, reassembly, half-closed TCP) run at this rate.
const HOUSEKEEPING: Duration = Duration::from_millis(100);

pub struct NatEngine {
    pub(super) config: GatewayConfig,
    pub(super) host_dns: Option<(SocketAddr, Instant)>, // host DNS server and when it was read
    tx_cons: rtrb::Consumer<Vec<u8>>, // outbound frames from the enet thread
    rx_prod: rtrb::Producer<Vec<u8>>, // inbound frames to the enet thread
    rx_wake: Arc<Wake>, // signal the NIC thread when rx_prod gets a frame
    tx_wake: Arc<Wake>, // the NIC thread signals new tx frames
    running: Arc<AtomicBool>,
    pub(super) ctl: Arc<NatControl>,
    pub(super) udp_nat: HashMap<UdpKey, NatUdpEntry>,
    pub(super) tcp_nat: HashMap<FlowKey, TcpConn>,
    pub(super) icmp_nat: HashMap<(u32, u16), NatIcmpEntry>, // key: (dst_ip, identifier)
    pub(super) icmp_unavailable: bool, // true after the first failed ICMP socket creation
    // Frames for the guest not yet on the ring; they go at the end of the pass.
    to_guest: VecDeque<Vec<u8>>,
    pub(super) tcp_fwd_listeners: Vec<TcpFwdListener>,
    pub(super) udp_fwd_listeners: Vec<UdpFwdListener>,
    // Next source ports to try for synthesized inbound connections.
    pub(super) fwd_ephemeral_next: u16,
    pub(super) fwd_reserved_next: u16, // BSD r-services want 512..1023
    // Configured (static) forwards at the front of `tcp_fwd_listeners`;
    // anything past this index is a transient FTP-ALG data forward.
    pub(super) fwd_static_count: usize,
    // Guest MAC learned from any outbound frame (ARP SHA or Ethernet src).
    pub(super) guest_mac: Option<MacAddr>,
    ip_id: u16,
    reasm: super::frag::Reassembler,
    // XDMCP reverse-proxy ALG: X11 port (6000 + display) → real X server.
    pub(super) xdmcp_sessions: HashMap<u16, Ipv4Addr>,
    // In-core UDP services at the gateway, and the ports each answers.
    pub(super) udp_services: Vec<Box<dyn UdpService>>,
    pub(super) udp_service_ports: HashMap<u16, usize>,
    // Who has talked to a service lately, and from which MAC (for replies the
    // service sends on its own, like TFTP retransmissions).
    pub(super) service_peers: HashMap<SocketAddrV4, (MacAddr, Instant)>,
    snapshot_at: Instant,
}

impl NatEngine {
    pub fn new(config: GatewayConfig,
               tx_cons: rtrb::Consumer<Vec<u8>>,
               rx_prod: rtrb::Producer<Vec<u8>>,
               rx_wake: Arc<Wake>,
               tx_wake: Arc<Wake>,
               running: Arc<AtomicBool>,
               ctl:     Arc<NatControl>) -> Self {
        let (tcp_fwd_listeners, udp_fwd_listeners) = bind_forwards(&config.port_forwards);
        let fwd_static_count = tcp_fwd_listeners.len();
        let mut udp_services = Vec::new();
        let mut udp_service_ports = HashMap::new();
        for (ports, svc) in super::services::udp_services(&config) {
            for p in ports {
                udp_service_ports.insert(p, udp_services.len());
            }
            udp_services.push(svc);
        }
        if let Some(dir) = &config.rsh_dir {
            eprintln!("iris: rsh server (read-only: cat, dd, ls, echo, pwd) serving {}", dir.display());
        }
        Self { config, host_dns: None, tx_cons, rx_prod, rx_wake, tx_wake, running, ctl,
               udp_nat: HashMap::new(), tcp_nat: HashMap::new(),
               icmp_nat: HashMap::new(), icmp_unavailable: false,
               to_guest: VecDeque::new(),
               tcp_fwd_listeners, udp_fwd_listeners, fwd_static_count,
               fwd_ephemeral_next: 49152,
               fwd_reserved_next: 512,
               guest_mac: None, ip_id: 1, reasm: Default::default(),
               xdmcp_sessions: HashMap::new(),
               udp_services, udp_service_ports, service_peers: HashMap::new(),
               snapshot_at: Instant::now() }
    }

    pub fn run(&mut self) {
        while self.running.load(Ordering::Relaxed) {
            let wait = self.idle_timeout();
            self.step(wait);
        }
    }

    /// One pass of the engine: wait up to `max_wait` for a guest frame or a
    /// ready host socket, then handle guest frames and poll every host socket
    /// and timer once.
    pub(crate) fn step(&mut self, max_wait: Duration) {
        self.wait_for_work(max_wait);

        // Machine reset: flush all NAT tables, close all host sockets.
        if self.ctl.reset_nat.swap(false, Ordering::AcqRel) {
            self.flush_flows();
            self.xdmcp_sessions.clear();
            self.ctl.routed.store(false, Ordering::Relaxed); // re-arm plug-and-play adoption
        }

        // Live subnet change requested by an embedder: swap the gateway /
        // client / netmask and flush connection state so nothing lingers on
        // the old subnet. The guest reaches the new gateway as soon as it ARPs
        // for it (or routes to it); no reboot.
        if self.ctl.apply_subnet.swap(false, Ordering::AcqRel) {
            self.config.gateway_ip = Ipv4Addr::from(self.ctl.apply_gateway.load(Ordering::Relaxed));
            self.config.client_ip  = Ipv4Addr::from(self.ctl.apply_client.load(Ordering::Relaxed));
            self.config.netmask    = Ipv4Addr::from(self.ctl.apply_netmask.load(Ordering::Relaxed));
            self.flush_flows();
            self.ctl.routed.store(false, Ordering::Relaxed);
        }

        // Live port-forward reconfigure: rebind the static listeners.
        if self.ctl.apply_forwards.swap(false, Ordering::AcqRel) {
            let rules = self.ctl.pending_forwards.lock().take(); // drop the lock before rebinding
            if let Some(rules) = rules {
                self.rebind_forwards(&rules);
            }
        }

        self.flush_to_ring();

        while let Ok(frame) = self.tx_cons.pop() {
            self.process(&frame);
        }

        self.poll_udp();
        self.poll_tcp();
        self.poll_icmp();
        self.poll_tcp_fwd_listeners();
        self.poll_udp_fwd_listeners();
        self.poll_services();
        self.flush_to_ring();
        if self.snapshot_at.elapsed() >= SNAPSHOT_EVERY {
            self.snapshot_at = Instant::now();
            self.update_snapshot();
        }
    }

    /// How long the next wait may last: until the earliest timer, or a
    /// moment if frames are waiting for room on the ring.
    pub(super) fn idle_timeout(&self) -> Duration {
        if !self.to_guest.is_empty() {
            return Duration::from_millis(1);
        }
        let now = Instant::now();
        self.tcp_nat.values()
            .filter_map(|c| c.deadline())
            .map(|d| d.saturating_duration_since(now))
            .fold(HOUSEKEEPING, Duration::min)
    }

    /// Sleep until a guest frame, a ready host socket, or `timeout`.
    #[cfg(unix)]
    fn wait_for_work(&mut self, timeout: Duration) {
        use std::os::fd::AsRawFd;
        if self.short_wait(timeout) {
            return;
        }
        let mut fds: Vec<libc::pollfd> = Vec::with_capacity(8 + self.tcp_nat.len() + self.udp_nat.len());
        let mut add = |fd: i32, events: i16| fds.push(libc::pollfd { fd, events, revents: 0 });
        match self.tx_wake.fd() {
            Some(fd) => add(fd, libc::POLLIN),
            None => {
                // No self-pipe: fall back to the condvar at a 1 ms tick.
                self.tx_wake.wait(timeout.min(Duration::from_millis(1)));
                return;
            }
        }
        for c in self.tcp_nat.values() {
            if let Some((fd, ev)) = c.poll_interest() {
                add(fd, ev);
            }
        }
        for e in self.udp_nat.values() {
            add(e.sock.as_raw_fd(), libc::POLLIN);
        }
        for e in self.icmp_nat.values() {
            if let Some(s) = &e.sock {
                add(s.as_raw_fd(), libc::POLLIN);
            }
        }
        for l in &self.tcp_fwd_listeners {
            add(l.listener.as_raw_fd(), libc::POLLIN);
        }
        for l in &self.udp_fwd_listeners {
            add(l.sock.as_raw_fd(), libc::POLLIN);
        }
        // Round up so a sub-millisecond deadline doesn't spin.
        let ms = timeout.as_micros().div_ceil(1000).min(i32::MAX as u128) as i32;
        // SAFETY: fds is a valid, initialised array of pollfd for its length.
        unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, ms) };
        self.tx_wake.consume();
    }

    #[cfg(not(unix))]
    fn wait_for_work(&mut self, timeout: Duration) {
        if self.short_wait(timeout) {
            return;
        }
        // Host sockets can't join the wait here: poll them at a 1 ms tick.
        self.tx_wake.wait(timeout.min(Duration::from_millis(1)));
    }

    /// A wait shorter than a millisecond (a timer about to expire) can't go through
    /// `poll(2)` or a condvar: both round it up to a whole tick, and Windows
    /// to its ~15.6 ms timer. Sleep instead. Host sockets and guest frames are
    /// not watched for that long, which costs nothing: the pass that follows
    /// handles whatever arrived. Returns true if it handled the wait.
    fn short_wait(&mut self, timeout: Duration) -> bool {
        if timeout >= Duration::from_millis(1) {
            return false;
        }
        if !timeout.is_zero() {
            std::thread::sleep(timeout);
        }
        self.tx_wake.consume(); // step() drains tx_cons whatever the signal said
        true
    }

    /// Drop every flow and its host socket (machine reset, subnet change).
    fn flush_flows(&mut self) {
        self.tcp_nat.clear();
        self.udp_nat.clear();
        self.icmp_nat.clear();
        self.tcp_fwd_listeners.truncate(self.fwd_static_count); // drop transient FTP data forwards
    }

    fn update_snapshot(&self) {
        let now = Instant::now();
        let mut snap = self.ctl.snapshot.lock();
        snap.tcp = self.tcp_nat.iter().map(|(&(ip, rport, cport), c)| NatTcpInfo {
            remote_ip:        Ipv4Addr::from(ip).to_string(),
            remote_port:      rport,
            client_port:      cport,
            age_secs:         now.duration_since(c.last_use).as_secs(),
            server_seq:       c.snd_nxt,
            server_seq_acked: c.snd_una,
            client_win:       c.peer_win,
            client_seq:       c.rcv_nxt,
            fin_wait:         c.fin_sent,
            rtx_count:        (c.in_flight() > 0) as usize,
            rtx_bytes:        c.unacked_len(),
        }).collect();
        snap.udp = self.udp_nat.iter().map(|(&(_, cport), e)| NatUdpInfo {
            remote_ip:   e.last_dst.ip().to_string(),
            remote_port: e.last_dst.port(),
            client_port: cport,
            age_secs:    now.duration_since(e.last_use).as_secs(),
        }).collect();
        snap.icmp = self.icmp_nat.iter().map(|(&(ip, ident), e)| NatIcmpInfo {
            remote_ip: Ipv4Addr::from(ip).to_string(),
            ident,
            age_secs:  now.duration_since(e.last_use).as_secs(),
        }).collect();
    }

    // ── Frames toward the guest ──────────────────────────────────────────────

    /// Addressing for a frame from `src` (as the guest will see it) to the guest.
    pub(super) fn route(&self, guest_mac: MacAddr, src: Ipv4Addr, guest_ip: Ipv4Addr) -> Ipv4Route {
        Ipv4Route::new(guest_mac, self.config.gateway_mac, src, guest_ip)
    }

    pub(super) fn next_ip_id(&mut self) -> u16 {
        self.ip_id = self.ip_id.wrapping_add(1);
        self.ip_id
    }

    /// Queue a frame for the guest. Nothing queued is ever dropped; it waits
    /// for room on the ring.
    pub(super) fn send(&mut self, frame: Vec<u8>) {
        if self.ctl.dbg_tcp() {
            log_frame("<<", &frame);
        }
        self.to_guest.push_back(frame);
    }

    /// How many more frames a bulk producer (TCP) should queue now.
    pub(super) fn queue_room(&self) -> usize {
        QUEUE_SOFT_LIMIT.saturating_sub(self.to_guest.len())
    }

    /// Move queued frames to the ring, in order, while it has room.
    fn flush_to_ring(&mut self) {
        let mut pushed = false;
        while let Some(frame) = self.to_guest.pop_front() {
            if let Err(rtrb::PushError::Full(frame)) = self.rx_prod.push(frame) {
                self.to_guest.push_front(frame);
                break;
            }
            pushed = true;
        }
        if pushed {
            self.rx_wake.notify();
        }
    }

    // ── Frames from the guest ────────────────────────────────────────────────

    pub(super) fn process(&mut self, frame: &[u8]) {
        let Some(eth) = Ether::parse(frame) else { return };
        let src_mac = eth.src();
        if self.guest_mac.is_none() {
            self.guest_mac = Some(src_mac);
        }
        // Record the guest's own source address so the GUI can tell what IP it
        // is using, even when that IP is wrong and nothing routes. ARP carries
        // it even with zero IP traffic, since the guest ARPs for its gateway.
        let arp = eth.arp();
        let ip = eth.ipv4();
        let src_ip = arp.map(|a| a.sender_ip()).or(ip.map(|i| i.src()));
        if let Some(ip) = src_ip.filter(|ip| !ip.is_unspecified()) {
            self.ctl.observed_guest_ip.store(u32::from(ip), Ordering::Relaxed);
        }
        if let Some(a) = arp.filter(|a| a.op() == packet::ARP_REQUEST) {
            self.maybe_adopt_gateway(a.sender_ip(), a.target_ip());
        }
        dlog_dev!(LogModule::Net, "NAT TX {}", packet::summary(frame));
        if self.ctl.dbg_tcp() && ip.is_some() {
            log_frame(">>", frame);
        }
        if let Some(a) = arp {
            self.handle_arp(&a);
        } else if let Some(ip) = ip {
            // Count only IP traffic as the network-alive signal: ARP happens
            // even when the guest's IP is missing or wrong.
            self.ctl.guest_frames.fetch_add(1, Ordering::Relaxed);
            // A frame to the gateway's MAC is off-subnet traffic routed through
            // us; that, not local chatter, locks adoption to the current gateway.
            if eth.dst() == self.config.gateway_mac {
                self.ctl.routed.store(true, Ordering::Relaxed);
            }
            self.handle_ip(&ip, src_mac);
        }
    }

    /// Plug-and-play: a guest ARP request for an in-subnet address it can't
    /// resolve is almost always its configured gateway. Record it for the GUI
    /// and, while nothing routes yet, adopt it so NAT answers and traffic flows
    /// with no config change. Self-limiting: once any frame routes through us
    /// we stop, so a working setup is never moved. Refused when the subnet
    /// overlaps a host network, which adopting would shadow.
    fn maybe_adopt_gateway(&mut self, spa: Ipv4Addr, tpa: Ipv4Addr) {
        let (s, t) = (spa.octets(), tpa.octets());
        dlog_dev!(LogModule::Net, "NAT guest ARP who-has {} tell {} (nat gw={} client={} guest_frames={})",
            tpa, spa, self.config.gateway_ip, self.config.client_ip,
            self.ctl.guest_frames.load(Ordering::Relaxed));
        if tpa.is_unspecified() || t == s || t[..3] != s[..3] {
            return;
        }
        self.ctl.observed_gateway.store(u32::from(tpa), Ordering::Relaxed);
        let adopt_net = Ipv4Addr::new(t[0], t[1], t[2], 0);
        let conflicts = self.ctl.host_conflict(adopt_net, 24);
        if self.config.gateway_ip != tpa && !self.ctl.routed.load(Ordering::Relaxed) && !conflicts {
            dlog_dev!(LogModule::Net, "NAT adopting gateway {} (was {})", tpa, self.config.gateway_ip);
            self.config.gateway_ip = tpa;
            self.config.client_ip = Ipv4Addr::new(t[0], t[1], t[2], 2);
            self.config.netmask = Ipv4Addr::new(255, 255, 255, 0);
        } else if conflicts {
            dlog_dev!(LogModule::Net, "NAT refusing to adopt {}: subnet {}/24 overlaps a host network", tpa, adopt_net);
        }
    }

    fn handle_arp(&mut self, a: &packet::Arp) {
        if a.op() != packet::ARP_REQUEST || a.target_ip() != self.config.gateway_ip {
            return;
        }
        dlog_dev!(LogModule::Net, "NAT ARP: who-has {} tell {}", a.target_ip(), a.sender_ip());
        let reply = packet::arp(packet::ARP_REPLY, self.config.gateway_mac, self.config.gateway_ip,
                                a.sender_mac(), a.sender_ip());
        self.send(reply);
    }

    fn handle_ip(&mut self, ip: &packet::Ipv4, src_mac: MacAddr) {
        // A guest NFS WRITE with a large wsize arrives fragmented: reassemble
        // and dispatch the whole datagram once it is complete.
        if ip.is_fragment() {
            if let Some(whole) = self.reasm.add(ip) {
                if ip.proto() == packet::PROTO_UDP {
                    if let Some(udp) = packet::Udp::parse(&whole) {
                        self.handle_udp(src_mac, ip.src(), ip.dst(), &udp);
                    }
                }
            }
            return;
        }
        match ip.proto() {
            packet::PROTO_ICMP => if let Some(icmp) = ip.icmp() {
                self.handle_icmp(src_mac, ip.src(), ip.dst(), ip.ttl(), &icmp);
            },
            packet::PROTO_UDP => if let Some(udp) = ip.udp() {
                self.handle_udp(src_mac, ip.src(), ip.dst(), &udp);
            },
            packet::PROTO_TCP => if let Some(tcp) = ip.tcp() {
                self.handle_tcp(src_mac, ip.src(), ip.dst(), &tcp);
            },
            _ => {}
        }
    }
}

impl NetBackend for NatEngine {
    fn run(&mut self) {
        NatEngine::run(self)
    }
}

/// Log a frame in detail (the `net tcp` debug switch).
pub(super) fn log_frame(dir: &str, frame: &[u8]) {
    dlog_dev!(LogModule::Net, "  {} {}", dir, packet::summary(frame));
    let Some(t) = Ether::parse(frame).and_then(|e| e.ipv4()).and_then(|i| i.tcp()) else { return };
    dlog_dev!(LogModule::Net, "     seq={} ack={} win={} len={}",
              t.seq(), t.ack(), t.window(), t.payload().len());
    if !t.payload().is_empty() {
        let hex: String = t.payload().iter().take(64).map(|b| format!(" {:02x}", b)).collect();
        dlog_dev!(LogModule::Net, "     data:{}", hex);
    }
}
