//! Directed tests for the NAT engine: a scripted guest speaks Ethernet frames
//! to a real `NatEngine`, which talks to real host sockets on loopback.
//!
//! The guest side is deliberately small and strict: it checks every checksum
//! and address on what the engine sends, and its TCP is just enough to drive
//! a connection (in-order data, cumulative ACKs, FIN). Tests use only frames
//! in and frames out, so they hold across any refactor of the engine.

use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream, UdpSocket};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use super::packet::{self, Ether, Ipv4Route, MacAddr, TcpFlags, TcpHeader, TCP_MSS};
use super::{GatewayConfig, NatControl, NatEngine};
use crate::config::{ForwardBind, ForwardProto, PortForwardConfig};

pub const GUEST_MAC: MacAddr = [0x08, 0x00, 0x69, 0x12, 0x34, 0x56];
/// How long a test waits for something before failing.
const PATIENCE: Duration = Duration::from_secs(10);

pub struct TestNet {
    pub e: NatEngine,
    rx: rtrb::Consumer<Vec<u8>>,
    pub guest_ip: Ipv4Addr,
    pub gw_ip: Ipv4Addr,
    pub gw_mac: MacAddr,
    ip_id: u16,
    /// Frames the engine sent that a waiter did not consume.
    pub inbox: VecDeque<Vec<u8>>,
    /// The NIC side's "frames sent" signal.
    pub tx_wake: Arc<super::Wake>,
    /// The NIC side of the guest-to-engine ring, for tests that need frames
    /// to arrive the way the enet thread delivers them (see `send_via_nic`).
    tx: rtrb::Producer<Vec<u8>>,
}

impl TestNet {
    pub fn new(config: GatewayConfig) -> Self {
        let (tx, tx_cons) = rtrb::RingBuffer::new(16);
        let (rx_prod, rx) = rtrb::RingBuffer::new(4096);
        let tx_wake = super::Wake::new();
        let e = NatEngine::new(
            config, tx_cons, rx_prod,
            super::Wake::new(),
            tx_wake.clone(),
            Arc::new(AtomicBool::new(true)),
            NatControl::new(),
        );
        let (guest_ip, gw_ip, gw_mac) = (e.config.client_ip, e.config.gateway_ip, e.config.gateway_mac);
        Self { e, rx, guest_ip, gw_ip, gw_mac, ip_id: 1, inbox: VecDeque::new(), tx_wake, tx }
    }

    pub fn default() -> Self {
        Self::new(GatewayConfig::default())
    }

    /// Route for a guest frame to `dst` (always via the gateway's MAC).
    pub fn route_to(&self, dst: Ipv4Addr) -> Ipv4Route {
        Ipv4Route::new(self.gw_mac, GUEST_MAC, self.guest_ip, dst)
    }

    pub fn next_id(&mut self) -> u16 {
        self.ip_id = self.ip_id.wrapping_add(1);
        self.ip_id
    }

    /// Hand a guest frame to the engine.
    pub fn send(&mut self, frame: Vec<u8>) {
        self.e.process(&frame);
    }

    /// Put a guest frame on the NIC ring and signal, as the enet thread does.
    /// Unlike `send`, the engine meets it inside `step`, as the real guest's
    /// frames are.
    pub fn send_via_nic(&mut self, frame: Vec<u8>) {
        self.tx.push(frame).expect("NIC ring full");
        self.tx_wake.notify();
    }

    /// One pass of the engine as its own thread runs it: wait as long as the
    /// engine asks, then step. Collects what it sent, like `pump`.
    pub fn run_pass(&mut self) {
        let wait = self.e.idle_timeout();
        self.e.step(wait);
        while let Ok(f) = self.rx.pop() {
            check_frame(&f, &self.gw_mac);
            self.inbox.push_back(f);
        }
    }

    /// Run one engine pass and move everything it sent into the inbox,
    /// checking that each frame is well formed and addressed to the guest.
    pub fn pump(&mut self) {
        self.e.step(Duration::ZERO);
        // step() moves deferred frames into the ring; drain it.
        while let Ok(f) = self.rx.pop() {
            check_frame(&f, &self.gw_mac);
            self.inbox.push_back(f);
        }
    }

    /// Pump until a frame matching `pred` arrives; return it (others stay queued).
    pub fn wait_for<F: FnMut(&[u8]) -> bool>(&mut self, what: &str, mut pred: F) -> Vec<u8> {
        let deadline = Instant::now() + PATIENCE;
        loop {
            if let Some(i) = self.inbox.iter().position(|f| pred(f)) {
                return self.inbox.remove(i).unwrap();
            }
            assert!(Instant::now() < deadline, "timed out waiting for {}", what);
            self.pump();
            if self.inbox.is_empty() {
                thread::sleep(Duration::from_millis(1));
            }
        }
    }

    /// Pump for `d`, collecting whatever the engine sends.
    pub fn settle(&mut self, d: Duration) {
        let end = Instant::now() + d;
        while Instant::now() < end {
            self.pump();
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// Wait for a UDP datagram to the guest's `dport`, reassembling fragments.
    pub fn wait_udp(&mut self, dport: u16) -> (Ipv4Addr, u16, Vec<u8>) {
        let deadline = Instant::now() + PATIENCE;
        // Fragments by IP id: offset -> bytes.
        let mut frags: HashMap<u16, std::collections::BTreeMap<usize, Vec<u8>>> = HashMap::new();
        loop {
            let mut keep = VecDeque::new();
            while let Some(f) = self.inbox.pop_front() {
                let Some(ip) = Ether::parse(&f).unwrap().ipv4().filter(|i| i.proto() == packet::PROTO_UDP) else {
                    keep.push_back(f);
                    continue;
                };
                let whole = if ip.is_fragment() {
                    let set = frags.entry(ip.id()).or_default();
                    set.insert(ip.frag_offset(), ip.payload().to_vec());
                    let mut whole = Vec::new();
                    for (&off, data) in set.iter() {
                        if off != whole.len() { break; }
                        whole.extend_from_slice(data);
                    }
                    let want = (whole.len() >= 8).then(|| u16::from_be_bytes([whole[4], whole[5]]) as usize);
                    if want != Some(whole.len()) {
                        continue;
                    }
                    frags.remove(&ip.id());
                    whole
                } else {
                    ip.payload().to_vec()
                };
                assert_eq!(packet::pseudo_checksum(ip.src(), ip.dst(), packet::PROTO_UDP, &whole), 0,
                           "UDP checksum to :{}", dport);
                let u = packet::Udp::parse(&whole).unwrap();
                if u.dport() == dport {
                    keep.extend(self.inbox.drain(..));
                    self.inbox = keep;
                    return (ip.src(), u.sport(), u.payload().to_vec());
                }
                keep.push_back(f);
            }
            self.inbox = keep;
            assert!(Instant::now() < deadline, "timed out waiting for UDP to :{}", dport);
            self.pump();
            thread::sleep(Duration::from_millis(1));
        }
    }
}

/// Every frame the engine sends must be addressed correctly and carry valid checksums.
fn check_frame(f: &[u8], gw_mac: &MacAddr) {
    let eth = Ether::parse(f).expect("runt frame from engine");
    assert_eq!(&eth.src(), gw_mac, "frame not from the gateway MAC: {}", packet::summary(f));
    if let Some(ip) = eth.ipv4() {
        assert!(ip.checksum_ok(), "bad IP header checksum: {}", packet::summary(f));
        if !ip.is_fragment() {
            match ip.proto() {
                packet::PROTO_TCP | packet::PROTO_UDP => {
                    let ok = packet::pseudo_checksum(ip.src(), ip.dst(), ip.proto(), ip.payload()) == 0;
                    assert!(ok, "bad L4 checksum: {}", packet::summary(f));
                }
                packet::PROTO_ICMP => assert_eq!(packet::checksum(ip.payload()), 0, "bad ICMP checksum"),
                _ => {}
            }
        }
    }
}

pub fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 251) as u8).collect()
}

// ── A minimal guest TCP ──────────────────────────────────────────────────────

pub struct GuestTcp {
    pub sport: u16,
    pub dst: Ipv4Addr,
    pub dport: u16,
    pub snd_una: u32,
    pub snd_nxt: u32,
    pub rcv_nxt: u32,
    /// Our advertised receive window.
    pub window: u16,
    pub peer_window: u32,
    pub peer_mss: Option<u16>,
    pub received: Vec<u8>,
    pub peer_fin: bool,
    pub reset: bool,
}

impl GuestTcp {
    fn is_mine(&self, f: &[u8]) -> bool {
        let Some(ip) = Ether::parse(f).and_then(|e| e.ipv4()) else { return false };
        let Some(t) = ip.tcp() else { return false };
        ip.src() == self.dst && t.dport() == self.sport && t.sport() == self.dport
    }

    fn seg(&self, net: &mut TestNet, flags: TcpFlags, seq: u32, data: &[u8]) {
        let h = TcpHeader { sport: self.sport, dport: self.dport, seq, ack: self.rcv_nxt,
                            flags, window: self.window,
                            mss: flags.syn().then_some(TCP_MSS) };
        let id = net.next_id();
        let f = net.route_to(self.dst).tcp(id, &h, data);
        net.send(f);
    }

    /// Open a connection; returns the flags of the engine's answer (SYN-ACK or RST).
    pub fn connect(net: &mut TestNet, dst: Ipv4Addr, dport: u16, sport: u16) -> (Self, TcpFlags) {
        Self::connect_win(net, dst, dport, sport, 65535)
    }

    /// `connect` advertising receive window `window` from the SYN on.
    pub fn connect_win(net: &mut TestNet, dst: Ipv4Addr, dport: u16, sport: u16, window: u16) -> (Self, TcpFlags) {
        let iss = 0x0100_0000u32.wrapping_mul(sport as u32);
        let mut c = GuestTcp { sport, dst, dport, snd_una: iss, snd_nxt: iss.wrapping_add(1),
                               rcv_nxt: 0, window, peer_window: 0, peer_mss: None,
                               received: Vec::new(), peer_fin: false, reset: false };
        c.seg(net, TcpFlags::SYN, iss, &[]);
        let f = net.wait_for("SYN-ACK or RST", |f| c.is_mine(f));
        let t = Ether::parse(&f).unwrap().ipv4().unwrap().tcp().unwrap();
        let flags = t.flags();
        if flags.rst() {
            c.reset = true;
            return (c, flags);
        }
        assert!(flags.syn() && flags.ack(), "expected SYN-ACK, got {:?}", flags);
        assert_eq!(t.ack(), iss.wrapping_add(1), "SYN-ACK must ack our SYN");
        c.rcv_nxt = t.seq().wrapping_add(1);
        c.snd_una = t.ack();
        c.peer_window = t.window() as u32;
        c.peer_mss = t.mss();
        c.seg(net, TcpFlags::ACK, c.snd_nxt, &[]);
        (c, flags)
    }

    /// Absorb one segment from the engine. In-order data is kept and ACKed
    /// (unless `drop_data`); anything else gets a duplicate ACK.
    fn absorb(&mut self, net: &mut TestNet, f: &[u8], drop_data: bool) {
        let ip = Ether::parse(f).unwrap().ipv4().unwrap();
        let t = ip.tcp().unwrap();
        let flags = t.flags();
        if flags.rst() {
            self.reset = true;
            return;
        }
        if flags.ack() && t.ack().wrapping_sub(self.snd_una) as i32 > 0 {
            self.snd_una = t.ack();
        }
        self.peer_window = t.window() as u32;
        let data = t.payload();
        if drop_data && !data.is_empty() {
            return;
        }
        let mut advance = false;
        if t.seq() == self.rcv_nxt {
            if !data.is_empty() {
                self.received.extend_from_slice(data);
                self.rcv_nxt = self.rcv_nxt.wrapping_add(data.len() as u32);
                advance = true;
            }
            if flags.fin() {
                self.rcv_nxt = self.rcv_nxt.wrapping_add(1);
                self.peer_fin = true;
                advance = true;
            }
        } else if !data.is_empty() || flags.fin() {
            advance = true; // out of order or a retransmit: duplicate ACK
        }
        if advance {
            self.seg(net, TcpFlags::ACK, self.snd_nxt, &[]);
        }
    }

    /// Process whatever the engine has sent this connection.
    pub fn poll(&mut self, net: &mut TestNet) {
        net.pump();
        let mine: Vec<Vec<u8>> = {
            let (m, rest): (Vec<_>, Vec<_>) = net.inbox.drain(..).partition(|f| self.is_mine(f));
            net.inbox.extend(rest);
            m
        };
        for f in mine {
            self.absorb(net, &f, false);
        }
    }

    /// Send `data`, honouring the peer's window and MSS, until all of it is ACKed.
    pub fn send_all(&mut self, net: &mut TestNet, data: &[u8]) {
        let mss = self.peer_mss.unwrap_or(536) as usize;
        let base = self.snd_nxt;
        let deadline = Instant::now() + PATIENCE;
        let mut last_progress = Instant::now();
        loop {
            let acked = self.snd_una.wrapping_sub(base) as usize;
            if acked >= data.len() {
                return;
            }
            let sent = self.snd_nxt.wrapping_sub(base) as usize;
            // Go back to the first unacked byte if nothing moved for a while.
            if last_progress.elapsed() > Duration::from_millis(300) {
                self.snd_nxt = self.snd_una;
                last_progress = Instant::now();
            }
            let in_flight = sent - acked;
            let room = (self.peer_window as usize).saturating_sub(in_flight);
            if sent < data.len() && room > 0 {
                let n = room.min(mss).min(data.len() - sent);
                let seq = self.snd_nxt;
                self.seg(net, TcpFlags::PSH_ACK, seq, &data[sent..sent + n]);
                self.snd_nxt = seq.wrapping_add(n as u32);
            }
            let before = self.snd_una;
            self.poll(net);
            if self.snd_una != before {
                last_progress = Instant::now();
            }
            assert!(!self.reset, "connection reset while sending");
            assert!(Instant::now() < deadline, "send stalled at {} of {} bytes", acked, data.len());
            if room == 0 {
                thread::sleep(Duration::from_millis(1));
            }
        }
    }

    /// Receive until `n` bytes have arrived (or the peer's FIN when `n` is None).
    pub fn recv(&mut self, net: &mut TestNet, n: Option<usize>) {
        let deadline = Instant::now() + PATIENCE;
        loop {
            match n {
                Some(n) if self.received.len() >= n => return,
                None if self.peer_fin => return,
                _ => {}
            }
            assert!(!self.reset, "connection reset while receiving");
            assert!(Instant::now() < deadline, "recv stalled at {} bytes", self.received.len());
            self.poll(net);
            thread::sleep(Duration::from_millis(1));
        }
    }

    pub fn close(&mut self, net: &mut TestNet) {
        let seq = self.snd_nxt;
        self.seg(net, TcpFlags::FIN_ACK, seq, &[]);
        self.snd_nxt = seq.wrapping_add(1);
        let deadline = Instant::now() + PATIENCE;
        while self.snd_una != self.snd_nxt {
            assert!(Instant::now() < deadline, "FIN never ACKed");
            self.poll(net);
            thread::sleep(Duration::from_millis(1));
        }
    }
}

// ── Host-side helpers ────────────────────────────────────────────────────────

/// A loopback listener running `serve` on the first connection, in a thread.
fn host_server<F>(serve: F) -> (u16, thread::JoinHandle<()>)
where
    F: FnOnce(TcpStream) + Send + 'static,
{
    let l = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let port = l.local_addr().unwrap().port();
    let h = thread::spawn(move || {
        let (s, _) = l.accept().unwrap();
        s.set_read_timeout(Some(PATIENCE)).unwrap();
        serve(s);
    });
    (port, h)
}

fn free_port() -> u16 {
    TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap().local_addr().unwrap().port()
}

// ── TCP ──────────────────────────────────────────────────────────────────────

#[test]
fn tcp_echo_through_the_gateway() {
    let (port, h) = host_server(|mut s| {
        let mut buf = [0u8; 8192];
        loop {
            match s.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => s.write_all(&buf[..n]).unwrap(),
            }
        }
    });
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let (mut c, flags) = GuestTcp::connect(&mut net, gw, port, 40001);
    assert!(flags.syn());
    let data = pattern(200_000);
    c.send_all(&mut net, &data);
    c.recv(&mut net, Some(data.len()));
    assert!(c.received == data, "echo differs");
    c.close(&mut net);
    c.recv(&mut net, None); // host sees EOF, closes; its FIN reaches us
    h.join().unwrap();
}

#[test]
fn tcp_source_respects_a_small_guest_window() {
    let data = pattern(300_000);
    let sent = data.clone();
    let (port, h) = host_server(move |mut s| { s.write_all(&sent).unwrap(); });
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    // A 2 KB window: the engine must never have more than that in flight.
    let (mut c, _) = GuestTcp::connect_win(&mut net, gw, port, 40002, 2048);
    let deadline = Instant::now() + PATIENCE;
    while !c.peer_fin {
        assert!(Instant::now() < deadline, "stalled at {}", c.received.len());
        net.pump();
        let mine: Vec<_> = net.inbox.drain(..).collect();
        let window_end = c.rcv_nxt.wrapping_add(c.window as u32);
        for f in &mine {
            if !c.is_mine(f) { continue; }
            let t = Ether::parse(f).unwrap().ipv4().unwrap().tcp().unwrap();
            let end = t.seq().wrapping_add(t.payload().len() as u32);
            assert!(window_end.wrapping_sub(end) as i32 >= 0,
                    "engine sent past the advertised window ({} > {})", end, window_end);
        }
        for f in mine {
            if c.is_mine(&f) { c.absorb(&mut net, &f, false); }
        }
        thread::sleep(Duration::from_millis(1));
    }
    assert!(c.received == data);
    h.join().unwrap();
}

#[test]
fn tcp_lost_segment_is_retransmitted() {
    let data = pattern(20_000);
    let sent = data.clone();
    let (port, h) = host_server(move |mut s| {
        s.write_all(&sent).unwrap();
        let mut b = [0u8; 1];
        let _ = s.read(&mut b); // hold the connection open until the guest closes
    });
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let (mut c, _) = GuestTcp::connect(&mut net, gw, port, 40003);
    // Drop the first data segment we see.
    let first = net.wait_for("first data", |f| {
        Ether::parse(f).and_then(|e| e.ipv4()).and_then(|i| i.tcp()).is_some_and(|t| !t.payload().is_empty())
    });
    c.absorb(&mut net, &first, true);
    c.recv(&mut net, Some(data.len()));
    assert!(c.received == data, "retransmission produced a different stream");
    c.close(&mut net);
    h.join().unwrap();
}

#[test]
fn tcp_out_of_order_guest_data_is_not_skipped() {
    let (port, h) = host_server(|mut s| {
        let mut all = Vec::new();
        s.read_to_end(&mut all).unwrap();
        assert_eq!(all, b"0123456789abcdef");
    });
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let (mut c, _) = GuestTcp::connect(&mut net, gw, port, 40004);
    let base = c.snd_nxt;
    // Second half first: must not be accepted past the gap.
    c.seg(&mut net, TcpFlags::PSH_ACK, base.wrapping_add(8), b"89abcdef");
    net.settle(Duration::from_millis(20));
    c.poll(&mut net);
    assert_eq!(c.snd_una, base, "engine acked data beyond a gap");
    c.seg(&mut net, TcpFlags::PSH_ACK, base, b"01234567");
    c.snd_nxt = base.wrapping_add(8);
    c.send_all(&mut net, b""); // pumps until the first half is acked
    let deadline = Instant::now() + PATIENCE;
    while c.snd_una != base.wrapping_add(8) {
        assert!(Instant::now() < deadline);
        c.poll(&mut net);
    }
    // Retransmit the second half, now in order.
    c.seg(&mut net, TcpFlags::PSH_ACK, base.wrapping_add(8), b"89abcdef");
    c.snd_nxt = base.wrapping_add(16);
    while c.snd_una != c.snd_nxt {
        assert!(Instant::now() < deadline);
        c.poll(&mut net);
    }
    c.close(&mut net);
    h.join().unwrap();
}

#[test]
fn tcp_half_close_from_the_guest_still_receives() {
    let (port, h) = host_server(|mut s| {
        let mut req = Vec::new();
        s.read_to_end(&mut req).unwrap(); // the guest's FIN ends the request
        assert_eq!(req, b"request");
        s.write_all(b"response after your FIN").unwrap();
    });
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let (mut c, _) = GuestTcp::connect(&mut net, gw, port, 40005);
    c.send_all(&mut net, b"request");
    c.close(&mut net);
    c.recv(&mut net, None);
    assert_eq!(c.received, b"response after your FIN");
    h.join().unwrap();
}

#[test]
fn tcp_connect_to_a_closed_port_is_reset() {
    let port = free_port();
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let (c, flags) = GuestTcp::connect(&mut net, gw, port, 40006);
    assert!(flags.rst(), "expected RST, got {:?}", flags);
    assert!(c.reset);
}

#[test]
fn tcp_segment_for_no_connection_is_reset() {
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let h = TcpHeader { sport: 40007, dport: 9, seq: 1000, ack: 2000, flags: TcpFlags::ACK,
                        window: 1000, mss: None };
    let id = net.next_id();
    let f = net.route_to(gw).tcp(id, &h, b"stray");
    net.send(f);
    let r = net.wait_for("RST", |f| {
        Ether::parse(f).and_then(|e| e.ipv4()).and_then(|i| i.tcp()).is_some_and(|t| t.flags().rst())
    });
    let t = Ether::parse(&r).unwrap().ipv4().unwrap().tcp().unwrap();
    assert_eq!(t.seq(), 2000, "RST takes its sequence number from the ACK");
}

// ── UDP ──────────────────────────────────────────────────────────────────────

fn udp_echo_host() -> (u16, thread::JoinHandle<()>) {
    let s = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let port = s.local_addr().unwrap().port();
    s.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
    // The echo must send back what the engine relayed, 20 KB included.
    super::nat_udp::allow_large_datagrams(&s);
    let h = thread::spawn(move || {
        let mut buf = vec![0u8; 65536];
        while let Ok((n, from)) = s.recv_from(&mut buf) {
            if &buf[..n] == b"quit" { break; }
            s.send_to(&buf[..n], from).unwrap();
        }
    });
    (port, h)
}

#[test]
fn udp_echo_through_the_gateway() {
    let (port, h) = udp_echo_host();
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    for (i, size) in [1usize, 64, 1472].into_iter().enumerate() {
        let data = pattern(size);
        let id = net.next_id();
        let f = net.route_to(gw).udp(id, 50000 + i as u16, port, &data);
        net.send(f);
        let (src, sport, got) = net.wait_udp(50000 + i as u16);
        assert_eq!((src, sport), (gw, port), "reply must come from the address the guest used");
        assert_eq!(got, data);
    }
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 50009, port, b"quit");
    net.send(f);
    h.join().unwrap();
}

#[test]
fn udp_fragmented_both_ways() {
    let (port, h) = udp_echo_host();
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let data = pattern(20_000);
    let id = net.next_id();
    for f in net.route_to(gw).udp_fragments(id, 50100, port, &data) {
        net.send(f);
    }
    let (_, _, got) = net.wait_udp(50100);
    assert_eq!(got.len(), data.len());
    assert!(got == data);
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 50101, port, b"quit");
    net.send(f);
    h.join().unwrap();
}

// ── Link and gateway ─────────────────────────────────────────────────────────

#[test]
fn arp_for_the_gateway_is_answered() {
    let mut net = TestNet::default();
    let (gw, guest) = (net.gw_ip, net.guest_ip);
    net.send(packet::arp(packet::ARP_REQUEST, GUEST_MAC, guest, [0; 6], gw));
    let r = net.wait_for("ARP reply", |f| Ether::parse(f).and_then(|e| e.arp()).is_some());
    let eth = Ether::parse(&r).unwrap();
    let a = eth.arp().unwrap();
    assert_eq!(eth.dst(), GUEST_MAC);
    assert_eq!((a.op(), a.sender_ip(), a.sender_mac(), a.target_ip()),
               (packet::ARP_REPLY, gw, net.gw_mac, guest));
    // Once the guest routes through us, other addresses are not ours to
    // answer. (Before that, an ARP for another in-subnet address makes the
    // NAT adopt it as the gateway: plug-and-play for guests with a static
    // configuration.)
    let id = net.next_id();
    let f = net.route_to(gw).icmp(id, packet::ICMP_ECHO_REQUEST, 0, [0, 1, 0, 1], b"");
    net.send(f);
    net.settle(Duration::from_millis(10));
    net.inbox.clear();
    net.send(packet::arp(packet::ARP_REQUEST, GUEST_MAC, guest, [0; 6], Ipv4Addr::new(192, 168, 0, 77)));
    net.settle(Duration::from_millis(10));
    assert!(net.inbox.is_empty());
}

#[test]
fn ping_the_gateway() {
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let id = net.next_id();
    let f = net.route_to(gw).icmp(id, packet::ICMP_ECHO_REQUEST, 0, [0xbe, 0xef, 0, 7], b"payload");
    net.send(f);
    let r = net.wait_for("echo reply", |f| {
        Ether::parse(f).and_then(|e| e.ipv4()).and_then(|i| i.icmp()).is_some()
    });
    let ip = Ether::parse(&r).unwrap().ipv4().unwrap();
    let i = ip.icmp().unwrap();
    assert_eq!((ip.src(), i.ty(), i.ident(), i.seq(), i.body()),
               (gw, packet::ICMP_ECHO_REPLY, 0xbeef, 7, &b"payload"[..]));
}

// ── Inbound port forwards ────────────────────────────────────────────────────

fn forward_config(proto: ForwardProto, host_port: u16, guest_port: u16) -> GatewayConfig {
    GatewayConfig {
        port_forwards: vec![PortForwardConfig { proto, host_port, guest_port, bind: ForwardBind::Localhost }],
        ..GatewayConfig::default()
    }
}

#[test]
fn tcp_port_forward_into_the_guest() {
    let host_port = free_port();
    let mut net = TestNet::new(forward_config(ForwardProto::Tcp, host_port, 23));
    // The engine learns the guest's MAC from its first frame.
    let (gw, guest) = (net.gw_ip, net.guest_ip);
    net.send(packet::arp(packet::ARP_REQUEST, GUEST_MAC, guest, [0; 6], gw));
    net.settle(Duration::from_millis(5));
    net.inbox.clear();

    let client = thread::spawn(move || {
        let mut s = TcpStream::connect((Ipv4Addr::LOCALHOST, host_port)).unwrap();
        s.set_read_timeout(Some(PATIENCE)).unwrap();
        s.write_all(b"hello guest").unwrap();
        let mut got = [0u8; 11];
        s.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"hello hosts");
    });

    // The engine injects a SYN from the gateway; answer it as the guest's server.
    let syn = net.wait_for("forwarded SYN", |f| {
        Ether::parse(f).and_then(|e| e.ipv4()).and_then(|i| i.tcp()).is_some_and(|t| t.flags() == TcpFlags::SYN)
    });
    let ip = Ether::parse(&syn).unwrap().ipv4().unwrap();
    let t = ip.tcp().unwrap();
    assert_eq!((ip.src(), ip.dst(), t.dport()), (gw, guest, 23));
    let mut c = GuestTcp { sport: 23, dst: gw, dport: t.sport(), snd_una: 5000, snd_nxt: 5001,
                           rcv_nxt: t.seq().wrapping_add(1), window: 65535, peer_window: t.window() as u32,
                           peer_mss: t.mss(), received: Vec::new(), peer_fin: false, reset: false };
    c.seg(&mut net, TcpFlags::SYN_ACK, 5000, &[]);
    c.recv(&mut net, Some(11));
    assert_eq!(c.received, b"hello guest");
    c.send_all(&mut net, b"hello hosts");
    client.join().unwrap();
}

#[test]
fn udp_port_forward_into_the_guest() {
    let host_port = free_port();
    let mut net = TestNet::new(forward_config(ForwardProto::Udp, host_port, 7));
    let (gw, guest) = (net.gw_ip, net.guest_ip);
    net.send(packet::arp(packet::ARP_REQUEST, GUEST_MAC, guest, [0; 6], gw));
    net.settle(Duration::from_millis(5));

    let host = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    host.set_read_timeout(Some(PATIENCE)).unwrap();
    host.send_to(b"to the guest", (Ipv4Addr::LOCALHOST, host_port)).unwrap();
    let (src, sport, got) = net.wait_udp(7);
    assert_eq!((src, got.as_slice()), (gw, &b"to the guest"[..]));
    // The guest answers to where the datagram came from.
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 7, sport, b"from the guest");
    net.send(f);
    net.settle(Duration::from_millis(5));
    let mut buf = [0u8; 64];
    let (n, _) = host.recv_from(&mut buf).unwrap();
    assert_eq!(&buf[..n], b"from the guest");
}

// ── Services ─────────────────────────────────────────────────────────────────

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("iris-testnet-{}-{}", tag, nanos));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn tftp_read_through_the_gateway() {
    let root = tmp_dir("tftp");
    let data = pattern(700);
    std::fs::write(root.join("cputest"), &data).unwrap();
    let mut net = TestNet::new(GatewayConfig { tftp_dir: Some(root.clone()), ..GatewayConfig::default() });
    let gw = net.gw_ip;
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 4242, 69, b"\x00\x01/cputest\x00octet\x00");
    net.send(f);
    let (src, sport, p) = net.wait_udp(4242);
    assert_eq!((src, sport), (gw, 69));
    assert_eq!(&p[..4], &[0, 3, 0, 1], "DATA block 1");
    assert_eq!(&p[4..], &data[..512]);
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 4242, 69, &[0, 4, 0, 1]);
    net.send(f);
    let (_, _, p) = net.wait_udp(4242);
    assert_eq!(&p[..4], &[0, 3, 0, 2]);
    assert_eq!(&p[4..], &data[512..]);
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 4242, 69, &[0, 4, 0, 2]);
    net.send(f);
    net.settle(Duration::from_millis(20));
    assert!(net.inbox.is_empty(), "the final ACK ends the transfer silently");
    let _ = std::fs::remove_dir_all(&root);
}

/// A BOOTP request as the PROM builds it for `bootp()<file>`.
fn bootp_request(file: &str, ciaddr: Ipv4Addr) -> Vec<u8> {
    let mut bp = vec![0u8; 300];
    bp[0] = 1;
    bp[1] = 1;
    bp[2] = 6;
    bp[4..8].copy_from_slice(&0x1234u32.to_be_bytes());
    bp[12..16].copy_from_slice(&ciaddr.octets());
    bp[28..34].copy_from_slice(&GUEST_MAC);
    bp[108..108 + file.len()].copy_from_slice(file.as_bytes());
    Ipv4Route::new(packet::BROADCAST_MAC, GUEST_MAC, ciaddr, Ipv4Addr::BROADCAST).udp(1, 68, 67, &bp)
}

#[test]
fn bootp_names_the_boot_file_and_answers_the_client() {
    let root = tmp_dir("bootp");
    std::fs::write(root.join("unix"), b"kernel").unwrap();
    let mut net = TestNet::new(GatewayConfig { tftp_dir: Some(root.clone()), ..GatewayConfig::default() });
    let ip_dst = |f: &[u8]| Ether::parse(f).unwrap().ipv4().unwrap().dst();

    // No address yet: the reply goes to the one being assigned (not broadcast:
    // the PROM drops broadcasts once it has an address).
    let guest = net.guest_ip;
    net.send(bootp_request("/unix", Ipv4Addr::UNSPECIFIED));
    let f = net.wait_for("BOOTP reply", |f| ip_dst(f) == guest);
    let ip = Ether::parse(&f).unwrap().ipv4().unwrap();
    let rep = ip.udp().unwrap().payload().to_vec();
    assert_eq!(&rep[16..20], &net.guest_ip.octets(), "yiaddr");
    assert_eq!(&rep[20..24], &net.gw_ip.octets(), "siaddr = TFTP server");
    assert_eq!(&rep[108..114], b"/unix\0", "the PROM refuses an empty boot file");
    assert_eq!(&rep[44..49], b"iris\0");

    // A reopen (ciaddr set) is answered at ciaddr.
    let ciaddr = Ipv4Addr::new(192, 168, 0, 77);
    net.send(bootp_request("/unix", ciaddr));
    let f = net.wait_for("BOOTP reply to ciaddr", |f| ip_dst(f) == ciaddr);
    let rep = Ether::parse(&f).unwrap().ipv4().unwrap().udp().unwrap().payload().to_vec();
    assert_eq!(&rep[108..114], b"/unix\0");

    // A file the root doesn't have leaves the field empty.
    net.send(bootp_request("missing", Ipv4Addr::UNSPECIFIED));
    let f = net.wait_for("BOOTP reply", |_| true);
    let rep = Ether::parse(&f).unwrap().ipv4().unwrap().udp().unwrap().payload().to_vec();
    assert!(rep[108..236].iter().all(|&b| b == 0));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn dhcp_offer_advertises_the_fallback_resolver() {
    let mut net = TestNet::default();
    let mut req = vec![0u8; 244];
    req[0] = 1;
    req[28..34].copy_from_slice(&GUEST_MAC);
    req[236..].copy_from_slice(&[99, 130, 83, 99, 53, 1, 1, 255]); // DISCOVER
    let f = Ipv4Route::new(packet::BROADCAST_MAC, GUEST_MAC, Ipv4Addr::UNSPECIFIED, Ipv4Addr::BROADCAST)
        .udp(1, 68, 67, &req);
    net.send(f);
    let f = net.wait_for("DHCP offer", |_| true);
    let ip = Ether::parse(&f).unwrap().ipv4().unwrap();
    assert_eq!(ip.dst(), Ipv4Addr::BROADCAST, "DHCP replies stay broadcast");
    let rep = ip.udp().unwrap().payload().to_vec();
    let opts = &rep[240..];
    let mut i = 0;
    let mut seen = HashMap::new();
    while opts[i] != 255 {
        let len = opts[i + 1] as usize;
        seen.insert(opts[i], opts[i + 2..i + 2 + len].to_vec());
        i += 2 + len;
    }
    assert_eq!(seen[&53], vec![2], "OFFER");
    assert_eq!(seen[&6], super::services::DNS_FALLBACK.octets().to_vec(),
               "not the host's resolver, which may be loopback");
    assert_eq!(seen[&3], net.gw_ip.octets().to_vec(), "router");
}

/// A host DNS server that answers one query (setting the QR bit).
fn dns_responder() -> (std::net::SocketAddr, thread::JoinHandle<Vec<u8>>) {
    let server = UdpSocket::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    let h = thread::spawn(move || {
        let mut buf = [0u8; 512];
        let (n, peer) = server.recv_from(&mut buf).unwrap();
        buf[2] |= 0x80;
        server.send_to(&buf[..n], peer).unwrap();
        buf[..n].to_vec()
    });
    (addr, h)
}

const DNS_QUERY: &[u8] = b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01";

#[test]
fn dns_goes_to_the_configured_upstream() {
    let (upstream, h) = dns_responder();
    let mut net = TestNet::new(GatewayConfig { dns_upstream: Some(upstream), ..GatewayConfig::default() });
    let gw = net.gw_ip;
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 4321, 53, DNS_QUERY);
    net.send(f);
    let answer = h.join().unwrap();
    let (src, sport, got) = net.wait_udp(4321);
    assert_eq!((src, sport), (gw, 53));
    assert_eq!(got, answer);
}

/// DHCP hands the guest a resolver that is not the gateway, and we intercept
/// the query on its way there. A resolver that checks where answers come from
/// (NetBSD's does) drops anything not from the address it asked.
#[test]
fn dns_reply_comes_from_the_resolver_the_guest_addressed() {
    let (upstream, h) = dns_responder();
    let mut net = TestNet::new(GatewayConfig { dns_upstream: Some(upstream), ..GatewayConfig::default() });
    let resolver = super::services::DNS_FALLBACK;
    let id = net.next_id();
    let f = net.route_to(resolver).udp(id, 4322, 53, DNS_QUERY);
    net.send(f);
    h.join().unwrap();
    let (src, _, _) = net.wait_udp(4322);
    assert_eq!(src, resolver);
}

#[test]
fn host_dns_is_the_default_upstream_and_cached() {
    let mut net = TestNet::default();
    let expected = super::host_dns::system_dns_server().unwrap_or(super::services::DNS_FALLBACK);
    assert_eq!(net.e.dns_upstream(), std::net::SocketAddr::from((expected, 53)));
    let (_, read_at) = net.e.host_dns.unwrap();
    net.e.dns_upstream();
    assert_eq!(net.e.host_dns.unwrap().1, read_at, "re-read before the refresh interval");
}

#[test]
fn portmap_and_time_at_the_gateway() {
    let mut net = TestNet::new(GatewayConfig {
        nfs: Some(crate::config::NfsConfig { shared_dir: "/nonexistent".into(),
                                             version: crate::net::nfsudp::NfsVersion::Auto }),
        ..GatewayConfig::default()
    });
    let gw = net.gw_ip;
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 900, 111, &super::portmap::getport_call(9, super::portmap::PROG_NFS));
    net.send(f);
    let (_, sport, rep) = net.wait_udp(900);
    assert_eq!(sport, 111);
    assert_eq!(u32::from_be_bytes(rep[24..28].try_into().unwrap()), 2049);

    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 901, 37, b"");
    net.send(f);
    let (_, _, t) = net.wait_udp(901);
    let secs = u32::from_be_bytes(t[..4].try_into().unwrap()) as u64;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    assert!((secs.wrapping_sub(2_208_988_800) as i64 - now as i64).abs() < 5, "RFC 868 time");

    // Time over TCP: SYN-ACK, 4 bytes, FIN.
    let (mut c, flags) = GuestTcp::connect(&mut net, gw, 37, 902);
    assert!(flags.syn());
    c.recv(&mut net, None);
    assert_eq!(c.received.len(), 4);
}

// ── TCP endpoint behaviour ───────────────────────────────────────────────────

fn tcp_of(f: &[u8]) -> Option<packet::Tcp<'_>> {
    Ether::parse(f)?.ipv4()?.tcp()
}

/// A connect to an address that never answers must not stop the engine: the
/// gateway still answers a ping straight away.
#[test]
fn black_holed_connect_does_not_stall_the_engine() {
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let h = TcpHeader { sport: 41000, dport: 80, seq: 1, ack: 0, flags: TcpFlags::SYN, window: 8192, mss: Some(1460) };
    let id = net.next_id();
    let f = net.route_to(Ipv4Addr::new(10, 255, 255, 1)).tcp(id, &h, &[]);
    let t0 = Instant::now();
    net.send(f);
    assert!(t0.elapsed() < Duration::from_millis(100), "handling the SYN blocked for {:?}", t0.elapsed());
    let id = net.next_id();
    let f = net.route_to(gw).icmp(id, packet::ICMP_ECHO_REQUEST, 0, [0, 9, 0, 1], b"x");
    net.send(f);
    net.wait_for("echo reply", |f| Ether::parse(f).and_then(|e| e.ipv4()).and_then(|i| i.icmp()).is_some());
    assert!(t0.elapsed() < Duration::from_millis(200), "ping answered after {:?}", t0.elapsed());
}

#[test]
fn mss_is_advertised_and_honoured() {
    let data = pattern(20_000);
    let sent = data.clone();
    let (port, h) = host_server(move |mut s| { s.write_all(&sent).unwrap(); });
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    // A guest that can only take 536-byte segments.
    let iss = 7000u32;
    let syn = TcpHeader { sport: 41001, dport: port, seq: iss, ack: 0, flags: TcpFlags::SYN,
                          window: 65535, mss: Some(536) };
    let id = net.next_id();
    let f = net.route_to(gw).tcp(id, &syn, &[]);
    net.send(f);
    let sa = net.wait_for("SYN-ACK", |f| tcp_of(f).is_some_and(|t| t.flags().syn()));
    let t = tcp_of(&sa).unwrap();
    assert_eq!(t.mss(), Some(1460), "SYN-ACK must carry our MSS");
    let mut c = GuestTcp { sport: 41001, dst: gw, dport: port, snd_una: iss + 1, snd_nxt: iss + 1,
                           rcv_nxt: t.seq().wrapping_add(1), window: 65535, peer_window: t.window() as u32,
                           peer_mss: t.mss(), received: Vec::new(), peer_fin: false, reset: false };
    c.seg(&mut net, TcpFlags::ACK, c.snd_nxt, &[]);
    let deadline = Instant::now() + PATIENCE;
    while !c.peer_fin {
        assert!(Instant::now() < deadline);
        net.pump();
        let mine: Vec<_> = net.inbox.drain(..).filter(|f| c.is_mine(f)).collect();
        for f in mine {
            assert!(tcp_of(&f).unwrap().payload().len() <= 536, "segment larger than the guest's MSS");
            c.absorb(&mut net, &f, false);
        }
        thread::sleep(Duration::from_millis(1));
    }
    assert!(c.received == data);
    h.join().unwrap();
}

/// A host that reads slowly must not lose the guest's data: the engine stops
/// acknowledging when its buffer is full and the guest retransmits.
#[test]
fn slow_host_reader_loses_nothing() {
    let data = pattern(1_500_000);
    let expect = data.clone();
    let (port, h) = host_server(move |mut s| {
        thread::sleep(Duration::from_millis(300)); // let the buffers fill
        let mut got = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            match s.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => { got.extend_from_slice(&buf[..n]); thread::sleep(Duration::from_micros(200)); }
                Err(e) => panic!("host read: {}", e),
            }
        }
        assert_eq!(got.len(), expect.len());
        assert!(got == expect, "the host saw a different stream");
    });
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let (mut c, _) = GuestTcp::connect(&mut net, gw, port, 41002);
    c.send_all(&mut net, &data);
    c.close(&mut net);
    c.recv(&mut net, None);
    h.join().unwrap();
}

#[test]
fn portmap_over_tcp_is_a_real_connection() {
    let mut net = TestNet::new(GatewayConfig {
        nfs: Some(crate::config::NfsConfig { shared_dir: "/nonexistent".into(),
                                             version: crate::net::nfsudp::NfsVersion::Auto }),
        ..GatewayConfig::default()
    });
    let gw = net.gw_ip;
    let (mut c, flags) = GuestTcp::connect(&mut net, gw, 111, 41003);
    assert!(flags.syn());
    for xid in [1u32, 2] {
        let call = super::portmap::getport_call(xid, super::portmap::PROG_MOUNTD);
        let mut rec = (call.len() as u32 | 0x8000_0000).to_be_bytes().to_vec();
        rec.extend_from_slice(&call);
        let before = c.received.len();
        c.send_all(&mut net, &rec);
        c.recv(&mut net, Some(before + 32));
        let r = &c.received[before..];
        assert_eq!(&r[4..8], &xid.to_be_bytes());
        assert_eq!(u32::from_be_bytes(r[28..32].try_into().unwrap()), 1234);
    }
    c.close(&mut net);
    c.recv(&mut net, None);
}

/// The guest's SYN is retransmitted because our SYN-ACK was lost: answer again.
#[test]
fn lost_syn_ack_is_resent() {
    let (port, h) = host_server(|mut s| { s.write_all(b"hi").unwrap(); });
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let syn = TcpHeader { sport: 41004, dport: port, seq: 500, ack: 0, flags: TcpFlags::SYN,
                          window: 65535, mss: Some(1460) };
    let id = net.next_id();
    let f = net.route_to(gw).tcp(id, &syn, &[]);
    net.send(f.clone());
    let first = net.wait_for("SYN-ACK", |f| tcp_of(f).is_some_and(|t| t.flags().syn()));
    net.send(f); // as if `first` never arrived
    let again = net.wait_for("SYN-ACK again", |f| tcp_of(f).is_some_and(|t| t.flags().syn()));
    assert_eq!(tcp_of(&first).unwrap().seq(), tcp_of(&again).unwrap().seq(), "same ISN");
    h.join().unwrap();
}

/// The guest doesn't answer a forwarded SYN at first: it is retransmitted.
#[test]
fn forward_syn_is_retransmitted() {
    let host_port = free_port();
    let mut net = TestNet::new(forward_config(ForwardProto::Tcp, host_port, 7));
    let (gw, guest) = (net.gw_ip, net.guest_ip);
    net.send(packet::arp(packet::ARP_REQUEST, GUEST_MAC, guest, [0; 6], gw));
    net.settle(Duration::from_millis(5));
    let _client = TcpStream::connect((Ipv4Addr::LOCALHOST, host_port)).unwrap();
    let is_syn = |f: &[u8]| tcp_of(f).is_some_and(|t| t.flags() == TcpFlags::SYN);
    let a = net.wait_for("SYN", is_syn);
    let b = net.wait_for("retransmitted SYN", is_syn);
    assert_eq!(tcp_of(&a).unwrap().seq(), tcp_of(&b).unwrap().seq());
    assert_eq!(tcp_of(&a).unwrap().mss(), Some(1460));
}

/// Passive FTP to the guest's ftpd through a forward: the 227 reply is
/// rewritten to a host address, and the data connection reaches the guest.
#[test]
fn ftp_passive_reply_through_a_forward() {
    let host_port = free_port();
    let mut net = TestNet::new(forward_config(ForwardProto::Tcp, host_port, 21));
    let (gw, guest) = (net.gw_ip, net.guest_ip);
    net.send(packet::arp(packet::ARP_REQUEST, GUEST_MAC, guest, [0; 6], gw));
    net.settle(Duration::from_millis(5));

    let client = thread::spawn(move || {
        let mut ctl = TcpStream::connect((Ipv4Addr::LOCALHOST, host_port)).unwrap();
        ctl.set_read_timeout(Some(PATIENCE)).unwrap();
        let mut line = Vec::new();
        let mut b = [0u8; 1];
        while !line.ends_with(b"\r\n") {
            ctl.read_exact(&mut b).unwrap();
            line.push(b[0]);
        }
        let s = String::from_utf8(line).unwrap();
        let nums: Vec<u16> = s[s.find('(').unwrap() + 1..s.find(')').unwrap()]
            .split(',').map(|n| n.parse().unwrap()).collect();
        assert_eq!(&nums[..4], &[127, 0, 0, 1], "reply must point at the host: {}", s);
        let mut data = TcpStream::connect((Ipv4Addr::LOCALHOST, nums[4] << 8 | nums[5])).unwrap();
        data.set_read_timeout(Some(PATIENCE)).unwrap();
        let mut got = String::new();
        data.read_to_string(&mut got).unwrap();
        assert_eq!(got, "listing");
    });

    // The guest's ftpd: accept the control connection, send a 227.
    let syn = net.wait_for("control SYN", |f| tcp_of(f).is_some_and(|t| t.dport() == 21 && t.flags().syn()));
    let t = tcp_of(&syn).unwrap();
    let mut ctl = GuestTcp { sport: 21, dst: gw, dport: t.sport(), snd_una: 9001, snd_nxt: 9001,
                             rcv_nxt: t.seq().wrapping_add(1), window: 65535, peer_window: t.window() as u32,
                             peer_mss: t.mss(), received: Vec::new(), peer_fin: false, reset: false };
    ctl.seg(&mut net, TcpFlags::SYN_ACK, 9000, &[]);
    ctl.send_all(&mut net, b"227 Entering Passive Mode (192,168,0,2,200,21).\r\n");
    // The data connection arrives at the guest's port 51221.
    let syn = net.wait_for("data SYN", |f| tcp_of(f).is_some_and(|t| t.dport() == 51221 && t.flags().syn()));
    let t = tcp_of(&syn).unwrap();
    let mut data = GuestTcp { sport: 51221, dst: gw, dport: t.sport(), snd_una: 7001, snd_nxt: 7001,
                              rcv_nxt: t.seq().wrapping_add(1), window: 65535, peer_window: t.window() as u32,
                              peer_mss: t.mss(), received: Vec::new(), peer_fin: false, reset: false };
    data.seg(&mut net, TcpFlags::SYN_ACK, 7000, &[]);
    data.send_all(&mut net, b"listing");
    data.close(&mut net);
    let deadline = Instant::now() + PATIENCE;
    while !client.is_finished() {
        assert!(Instant::now() < deadline, "host FTP client stuck");
        net.pump();
        for f in net.inbox.drain(..).collect::<Vec<_>>() {
            if data.is_mine(&f) { data.absorb(&mut net, &f, false); }
            else if ctl.is_mine(&f) { ctl.absorb(&mut net, &f, false); }
        }
        thread::sleep(Duration::from_millis(1));
    }
    client.join().unwrap();
}

/// The host resets mid-stream: the guest is told.
#[test]
fn host_reset_reaches_the_guest() {
    let (port, h) = host_server(|s| {
        // SO_LINGER 0 turns the close into a RST.
        let sock = socket2::Socket::from(s);
        sock.set_linger(Some(Duration::ZERO)).unwrap();
        thread::sleep(Duration::from_millis(50));
    });
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let (mut c, _) = GuestTcp::connect(&mut net, gw, port, 41005);
    h.join().unwrap();
    let deadline = Instant::now() + PATIENCE;
    while !c.reset && !c.peer_fin {
        assert!(Instant::now() < deadline, "the guest never heard of the reset");
        c.poll(&mut net);
        thread::sleep(Duration::from_millis(1));
    }
}

// ── UDP mapping ──────────────────────────────────────────────────────────────

/// A server that answers from a fresh port (as TFTP servers do) is seen by the
/// guest at that port, so the guest can lock on to the new TID.
#[test]
fn udp_reply_from_another_port_keeps_its_source() {
    let server = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let port = server.local_addr().unwrap().port();
    let h = thread::spawn(move || {
        let mut buf = [0u8; 64];
        let (n, peer) = server.recv_from(&mut buf).unwrap();
        let other = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        other.send_to(&buf[..n], peer).unwrap();
        other.local_addr().unwrap().port()
    });
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 51000, port, b"rrq");
    net.send(f);
    let new_port = h.join().unwrap();
    let (src, sport, got) = net.wait_udp(51000);
    assert_eq!((src, sport, got.as_slice()), (gw, new_port, &b"rrq"[..]));
}

/// One guest port, two peers: one host socket, each reply from its own peer.
#[test]
fn udp_one_guest_port_two_peers() {
    let (p1, h1) = udp_echo_host();
    let (p2, h2) = udp_echo_host();
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    for (port, msg) in [(p1, &b"one"[..]), (p2, &b"two"[..])] {
        let id = net.next_id();
        let f = net.route_to(gw).udp(id, 51001, port, msg);
        net.send(f);
        let (_, sport, got) = net.wait_udp(51001);
        assert_eq!((sport, got.as_slice()), (port, msg));
    }
    assert_eq!(net.e.udp_nat.len(), 1, "one mapping per guest endpoint");
    for port in [p1, p2] {
        let id = net.next_id();
        let f = net.route_to(gw).udp(id, 51002, port, b"quit");
        net.send(f);
    }
    h1.join().unwrap();
    h2.join().unwrap();
}

/// A resolver that never answers must not stall the engine.
#[test]
fn silent_dns_upstream_does_not_stall_the_engine() {
    let silent = UdpSocket::bind("127.0.0.1:0").unwrap();
    let mut net = TestNet::new(GatewayConfig { dns_upstream: Some(silent.local_addr().unwrap()),
                                               ..GatewayConfig::default() });
    let gw = net.gw_ip;
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 4323, 53, DNS_QUERY);
    let t0 = Instant::now();
    net.send(f);
    let id = net.next_id();
    let f = net.route_to(gw).icmp(id, packet::ICMP_ECHO_REQUEST, 0, [0, 9, 0, 2], b"y");
    net.send(f);
    net.wait_for("echo reply", |f| Ether::parse(f).and_then(|e| e.ipv4()).and_then(|i| i.icmp()).is_some());
    assert!(t0.elapsed() < Duration::from_millis(200), "ping answered after {:?}", t0.elapsed());
    let mut buf = [0u8; 512];
    silent.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
    assert!(silent.recv_from(&mut buf).is_ok(), "the query still reached the upstream");
}

/// A reply to a guest frame goes on the ring in the pass that read the
/// frame. Waiting for a later pass cost a poll tick per lock-step exchange:
/// TFTP paid it once per 512-byte block, capping it near 450 KB/s on Linux
/// and 30 KB/s on Windows.
#[test]
fn tftp_blocks_are_not_paced_by_the_poll_tick() {
    const BLOCKS: usize = 300;
    let root = tmp_dir("tftp-pace");
    let data = pattern(BLOCKS * 512 + 100);
    std::fs::write(root.join("big"), &data).unwrap();
    let mut net = TestNet::new(GatewayConfig { tftp_dir: Some(root.clone()), ..GatewayConfig::default() });
    let gw = net.gw_ip;

    // DATA for `block`, in order, run the way the engine thread runs.
    let next_data = |net: &mut TestNet, block: u16| -> Vec<u8> {
        let deadline = Instant::now() + PATIENCE;
        loop {
            let hit = net.inbox.iter().position(|f| {
                let Some(u) = Ether::parse(f).and_then(|e| e.ipv4()).and_then(|i| i.udp()) else { return false };
                u.dport() == 4270 && u.payload().get(..4) == Some(&[0, 3, (block >> 8) as u8, block as u8][..])
            });
            if let Some(i) = hit {
                let f = net.inbox.remove(i).unwrap();
                let u = Ether::parse(&f).unwrap().ipv4().unwrap().udp().unwrap().payload().to_vec();
                return u;
            }
            assert!(Instant::now() < deadline, "timed out waiting for DATA {}", block);
            net.run_pass();
        }
    };

    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 4270, 69, b"\x00\x01big\x00octet\x00");
    net.send_via_nic(f);
    let mut got = next_data(&mut net, 1)[4..].to_vec();

    let start = Instant::now();
    for block in 1..=BLOCKS as u16 {
        let id = net.next_id();
        let f = net.route_to(gw).udp(id, 4270, 69, &[0, 4, (block >> 8) as u8, block as u8]);
        net.send_via_nic(f);
        if block == 1 {
            // One pass reads the ACK and puts the next DATA on the ring.
            net.run_pass();
            assert_eq!(net.inbox.len(), 1, "the reply to an ACK waited for a later pass");
        }
        got.extend_from_slice(&next_data(&mut net, block + 1)[4..]);
    }
    let elapsed = start.elapsed();
    assert_eq!(got, data);
    // The old 1 ms tick made this at least BLOCKS ms by construction.
    assert!(elapsed < Duration::from_millis(BLOCKS as u64),
            "{} blocks took {:?}: replies are waiting for the poll tick", BLOCKS, elapsed);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn tftp_retransmits_through_the_service_timer() {
    let root = tmp_dir("tftp-rtx");
    std::fs::write(root.join("f"), pattern(2000)).unwrap();
    let mut net = TestNet::new(GatewayConfig { tftp_dir: Some(root.clone()), ..GatewayConfig::default() });
    let gw = net.gw_ip;
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 4250, 69, b"\x00\x01f\x00octet\x00");
    net.send(f);
    let (_, _, first) = net.wait_udp(4250);
    // No ACK: the same block comes again after the retransmit timeout.
    let (src, sport, again) = net.wait_udp(4250);
    assert_eq!((src, sport), (gw, 69));
    assert_eq!(again, first);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn ntp_answers_with_the_host_clock() {
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let mut q = vec![0u8; 48];
    q[0] = 0x23; // v4 client
    q[40..48].copy_from_slice(&0x1122_3344_5566_7788u64.to_be_bytes());
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 4260, 123, &q);
    net.send(f);
    let (_, sport, r) = net.wait_udp(4260);
    assert_eq!(sport, 123);
    assert_eq!(r[0] & 7, 4, "server mode");
    assert_eq!(&r[24..32], &q[40..48], "originate = our transmit timestamp");
    let secs = u32::from_be_bytes(r[40..44].try_into().unwrap()) as i64 - 2_208_988_800i64;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    assert!((secs - now).abs() < 5);
}

// ── The event loop ───────────────────────────────────────────────────────────

/// The engine sleeps when idle and wakes at once when the NIC signals.
#[test]
fn engine_sleeps_until_signalled() {
    let mut net = TestNet::default();
    let t0 = Instant::now();
    net.e.step(Duration::from_millis(50));
    assert!(t0.elapsed() >= Duration::from_millis(45), "idle step returned early: {:?}", t0.elapsed());
    net.tx_wake.notify();
    let t0 = Instant::now();
    net.e.step(Duration::from_secs(5));
    assert!(t0.elapsed() < Duration::from_millis(100), "a notify didn't end the wait: {:?}", t0.elapsed());
}

/// A host socket becoming readable ends the wait (Unix: it is in the poll set).
#[cfg(unix)]
#[test]
fn engine_wakes_for_host_data() {
    let (port, h) = udp_echo_host();
    let mut net = TestNet::default();
    let gw = net.gw_ip;
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 52000, port, b"ping");
    net.send(f); // relays it to the host now; the echo comes back while we sleep
    let t0 = Instant::now();
    net.e.step(Duration::from_secs(5));
    assert!(t0.elapsed() < Duration::from_millis(500), "readable socket didn't wake the engine");
    let (_, _, got) = net.wait_udp(52000);
    assert_eq!(got, b"ping");
    let id = net.next_id();
    let f = net.route_to(gw).udp(id, 52001, port, b"quit");
    net.send(f);
    h.join().unwrap();
}

// ── Built-in rsh ─────────────────────────────────────────────────────────────

fn rsh_config(root: &std::path::Path) -> GatewayConfig {
    GatewayConfig { rsh_dir: Some(root.to_path_buf()), ..GatewayConfig::default() }
}

/// The gateway's connection back to the guest, answered as `rcmd` would: a
/// guest listening on `port`.
fn accept_side_connection(net: &mut TestNet, port: u16) -> GuestTcp {
    let (gw, guest) = (net.gw_ip, net.guest_ip);
    let syn = net.wait_for("stderr SYN", |f| {
        tcp_of(f).is_some_and(|t| t.flags() == TcpFlags::SYN && t.dport() == port)
    });
    let ip = Ether::parse(&syn).unwrap().ipv4().unwrap();
    let t = ip.tcp().unwrap();
    assert_eq!((ip.src(), ip.dst()), (gw, guest));
    assert!((512..=1023).contains(&t.sport()), "rcmd refuses a stderr connection from port {}", t.sport());
    let mut c = GuestTcp { sport: port, dst: gw, dport: t.sport(), snd_una: 7000, snd_nxt: 7001,
                           rcv_nxt: t.seq().wrapping_add(1), window: 65535, peer_window: t.window() as u32,
                           peer_mss: t.mss(), received: Vec::new(), peer_fin: false, reset: false };
    c.seg(net, TcpFlags::SYN_ACK, 7000, &[]);
    c
}

/// `rsh gateway cat ...` as BSD `rcmd` speaks it: a request from a reserved
/// port naming a stderr port, then the gateway connects back to that port.
#[test]
fn rsh_cat_with_a_stderr_connection() {
    let root = tmp_dir("rsh");
    let data = pattern(70_000);
    std::fs::write(root.join("hello.txt"), &data).unwrap();
    let mut net = TestNet::new(rsh_config(&root));
    let gw = net.gw_ip;
    let (mut out, flags) = GuestTcp::connect(&mut net, gw, 514, 1023);
    assert!(flags.syn(), "the gateway answers on 514");
    // rcmd sends the port alone and waits for the connection back before it
    // sends the user names and the command.
    out.send_all(&mut net, b"5000\0");
    let mut err = accept_side_connection(&mut net, 5000);
    out.send_all(&mut net, b"guest\0root\0cat /hello.txt nope ../hello.txt\0");

    let want_err = b"cat: nope: No such file or directory\ncat: ../hello.txt: Permission denied\n";
    out.recv(&mut net, Some(1 + data.len()));
    err.recv(&mut net, Some(want_err.len()));
    assert_eq!(out.received[0], 0, "accepted");
    assert_eq!(&out.received[1..], &data[..]);
    assert_eq!(err.received, want_err);
    // Both connections end with the gateway's FIN.
    out.recv(&mut net, None);
    err.recv(&mut net, None);
    assert_eq!(out.received.len(), 1 + data.len(), "nothing after the file");
    out.close(&mut net);
    err.close(&mut net);
    let _ = std::fs::remove_dir_all(&root);
}

/// A client that wants no stderr connection gets none, and its errors come
/// down the main connection.
#[test]
fn rsh_without_a_stderr_port() {
    let root = tmp_dir("rsh0");
    std::fs::write(root.join("a"), b"A").unwrap();
    let mut net = TestNet::new(rsh_config(&root));
    let gw = net.gw_ip;
    let (mut c, _) = GuestTcp::connect(&mut net, gw, 514, 1022);
    c.send_all(&mut net, b"0\0guest\0guest\0ls\0");
    c.recv(&mut net, None);
    assert_eq!(c.received, b"\0a\n");
    net.settle(Duration::from_millis(50));
    assert!(!net.inbox.iter().any(|f| tcp_of(f).is_some_and(|t| t.flags() == TcpFlags::SYN)),
            "no connection back to the guest");
    c.close(&mut net);

    let (mut c, _) = GuestTcp::connect(&mut net, gw, 514, 1021);
    c.send_all(&mut net, b"0\0guest\0guest\0cat /missing\0");
    c.recv(&mut net, None);
    assert_eq!(c.received, b"\0cat: /missing: No such file or directory\n");
    c.close(&mut net);
    let _ = std::fs::remove_dir_all(&root);
}

/// The request may be split across segments, and a client with a small
/// window still receives the whole file.
#[test]
fn rsh_survives_a_split_request_and_a_small_window() {
    let root = tmp_dir("rshwin");
    let data = pattern(40_000);
    std::fs::write(root.join("f"), &data).unwrap();
    let mut net = TestNet::new(rsh_config(&root));
    let gw = net.gw_ip;
    let (mut c, _) = GuestTcp::connect_win(&mut net, gw, 514, 1020, 2048);
    c.send_all(&mut net, b"0\0gu");
    c.send_all(&mut net, b"est\0guest\0cat f");
    c.send_all(&mut net, b"\0");
    c.recv(&mut net, None);
    assert_eq!(c.received[0], 0);
    assert_eq!(&c.received[1..], &data[..]);
    c.close(&mut net);
    let _ = std::fs::remove_dir_all(&root);
}

/// A bad request is refused the way rshd does: a 1 byte, the reason, a newline.
#[test]
fn rsh_refuses_a_malformed_request() {
    let root = tmp_dir("rshbad");
    let mut net = TestNet::new(rsh_config(&root));
    let gw = net.gw_ip;
    let (mut c, _) = GuestTcp::connect(&mut net, gw, 514, 1019);
    c.send_all(&mut net, b"not-a-port\0x\0y\0pwd\0");
    c.recv(&mut net, None);
    assert_eq!(c.received, b"\x01rshd: protocol failure in circuit setup\n");
    c.close(&mut net);
    let _ = std::fs::remove_dir_all(&root);
}
