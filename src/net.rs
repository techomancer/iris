// NAT gateway / network engine for the SEEQ 8003 emulator.
//
// Runs in its own thread ("seeq-nat"). Receives outbound Ethernet frames from
// the enet thread via an rtrb::Consumer<Vec<u8>>, processes them through a
// software NAT stack, and enqueues inbound frames back via rtrb::Producer<Vec<u8>>.

use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream, UdpSocket};
use socket2::{Domain, Protocol, Socket, Type};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use crate::devlog::LogModule;
use parking_lot::{Condvar, Mutex};
use std::time::{Duration, Instant};

// ── Ethernet constants ────────────────────────────────────────────────────────
const ETHERTYPE_ARP: u16        = 0x0806;
const ETHERTYPE_IP:  u16        = 0x0800;
const IP_PROTO_ICMP: u8         = 1;
const IP_PROTO_TCP:  u8         = 6;
const IP_PROTO_UDP:  u8         = 17;
const ICMP_ECHO_REQUEST: u8     = 8;
const ICMP_ECHO_REPLY:   u8     = 0;
const ARP_HW_ETHER:   u16       = 1;
const ARP_PROTO_IP:   u16       = 0x0800;
const ARP_OP_REQUEST: u16       = 1;
const ARP_OP_REPLY:   u16       = 2;
const UDP_PORT_BOOTP_SERVER: u16 = 67;
const UDP_PORT_BOOTP_CLIENT: u16 = 68;
const UDP_PORT_DNS:          u16 = 53;
const BOOTP_OP_REQUEST: u8      = 1;

// ── Gateway configuration ─────────────────────────────────────────────────────
#[derive(Clone)]
pub struct GatewayConfig {
    pub gateway_mac: [u8; 6],
    pub gateway_ip:  Ipv4Addr,
    pub client_ip:   Ipv4Addr,
    pub netmask:     Ipv4Addr,
    pub dns_upstream: SocketAddr,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            gateway_mac: [0x02, 0x00, 0xDE, 0xAD, 0xBE, 0xEF],
            gateway_ip:  Ipv4Addr::new(192, 168, 0, 1),
            client_ip:   Ipv4Addr::new(192, 168, 0, 2),
            netmask:     Ipv4Addr::new(255, 255, 255, 0),
            dns_upstream: "8.8.8.8:53".parse().unwrap(),
        }
    }
}

// ── NAT table entries ─────────────────────────────────────────────────────────
struct NatUdpEntry {
    sock:        UdpSocket,
    client_mac:  [u8; 6],
    client_ip:   Ipv4Addr,
    #[allow(dead_code)]
    client_port: u16,
    last_use:    Instant,
}

// Key: (dst_ip, icmp_identifier) — identifier plays the role of "port" for ICMP NAT.
struct NatIcmpEntry {
    sock:       Option<Socket>,  // None if raw socket creation failed (e.g. not admin on Windows)
    client_mac: [u8; 6],
    client_ip:  Ipv4Addr,
    last_use:   Instant,
}

struct RetransmitEntry {
    seq:     u32,
    data:    Vec<u8>,
    sent_at: Instant,
}

const RTO: Duration = Duration::from_millis(200);

struct NatTcpEntry {
    stream:           TcpStream,
    client_mac:       [u8; 6],
    client_ip:        Ipv4Addr,
    client_port:      u16,
    server_ip:        Ipv4Addr,  // real remote IP — used as src in all replies to client
    server_seq:       u32,       // next seq we will send to IRIX
    server_seq_acked: u32,       // last seq IRIX has ACKed (= what IRIX has consumed)
    client_win:       u32,       // IRIX's advertised receive window (bytes)
    client_seq:       u32,
    last_use:         Instant,
    fin_wait:         bool,  // IRIX sent FIN; waiting for server to close before we send FIN back
    server_fin:       bool,  // server closed its end; need to send FIN+ACK to IRIX when ring has space
    retransmit:       VecDeque<RetransmitEntry>,
}

// ── Packet helpers ────────────────────────────────────────────────────────────
pub fn r16(b: &[u8], o: usize) -> u16 { ((b[o] as u16) << 8) | b[o+1] as u16 }
fn r32(b: &[u8], o: usize) -> u32 {
    ((b[o] as u32) << 24) | ((b[o+1] as u32) << 16) | ((b[o+2] as u32) << 8) | b[o+3] as u32
}
pub fn w16(b: &mut [u8], o: usize, v: u16) { b[o] = (v>>8) as u8; b[o+1] = v as u8; }
fn w32(b: &mut [u8], o: usize, v: u32) {
    b[o]=(v>>24)as u8; b[o+1]=(v>>16)as u8; b[o+2]=(v>>8)as u8; b[o+3]=v as u8;
}

fn ip_checksum(data: &[u8]) -> u16 {
    let mut s: u32 = 0;
    let mut i = 0;
    while i+1 < data.len() { s += r16(data, i) as u32; i += 2; }
    if i < data.len() { s += (data[i] as u32) << 8; }
    while s >> 16 != 0 { s = (s & 0xffff) + (s >> 16); }
    !(s as u16)
}

fn ipv4_header(src: Ipv4Addr, dst: Ipv4Addr, proto: u8, payload_len: u16) -> [u8; 20] {
    let mut h = [0u8; 20];
    h[0] = 0x45;
    w16(&mut h, 2, 20 + payload_len);
    h[8]  = 64;
    h[9]  = proto;
    h[12..16].copy_from_slice(&src.octets());
    h[16..20].copy_from_slice(&dst.octets());
    let c = ip_checksum(&h); w16(&mut h, 10, c);
    h
}

fn udp_checksum(src: Ipv4Addr, dst: Ipv4Addr, sport: u16, dport: u16, payload: &[u8]) -> u16 {
    let udp_len = (8 + payload.len()) as u16;
    let mut p = Vec::with_capacity(12 + udp_len as usize);
    p.extend_from_slice(&src.octets());
    p.extend_from_slice(&dst.octets());
    p.push(0); p.push(IP_PROTO_UDP);
    p.push((udp_len>>8) as u8); p.push(udp_len as u8);
    p.push((sport>>8) as u8); p.push(sport as u8);
    p.push((dport>>8) as u8); p.push(dport as u8);
    p.push((udp_len>>8) as u8); p.push(udp_len as u8);
    p.push(0); p.push(0);
    p.extend_from_slice(payload);
    ip_checksum(&p)
}

fn udp_packet(src_ip: Ipv4Addr, dst_ip: Ipv4Addr, sport: u16, dport: u16, payload: &[u8]) -> Vec<u8> {
    let udp_len = 8u16 + payload.len() as u16;
    let csum = udp_checksum(src_ip, dst_ip, sport, dport, payload);
    let mut p = Vec::with_capacity(udp_len as usize);
    p.push((sport>>8) as u8); p.push(sport as u8);
    p.push((dport>>8) as u8); p.push(dport as u8);
    p.push((udp_len>>8) as u8); p.push(udp_len as u8);
    p.push((csum>>8) as u8); p.push(csum as u8);
    p.extend_from_slice(payload);
    p
}

fn tcp_checksum(src: Ipv4Addr, dst: Ipv4Addr, tcp_seg: &[u8]) -> u16 {
    let tcp_len = tcp_seg.len() as u16;
    let mut p = Vec::with_capacity(12 + tcp_seg.len());
    p.extend_from_slice(&src.octets());
    p.extend_from_slice(&dst.octets());
    p.push(0); p.push(IP_PROTO_TCP);
    p.push((tcp_len>>8) as u8); p.push(tcp_len as u8);
    p.extend_from_slice(tcp_seg);
    ip_checksum(&p)
}

fn flags_str(flags: u8) -> String {
    let mut s = String::new();
    if flags & 0x02 != 0 { s.push_str("SYN "); }
    if flags & 0x10 != 0 { s.push_str("ACK "); }
    if flags & 0x08 != 0 { s.push_str("PSH "); }
    if flags & 0x01 != 0 { s.push_str("FIN "); }
    if flags & 0x04 != 0 { s.push_str("RST "); }
    if s.is_empty() { s.push_str("---"); }
    s.trim_end().to_string()
}

/// Decode and print a full Ethernet frame as human-readable TCP info.
fn log_eth_frame(dir: &str, frame: &[u8]) {
    if frame.len() < 14 { dlog_dev!(LogModule::Net, "  {} <runt>", dir); return; }
    let dst_mac = &frame[0..6];
    let src_mac = &frame[6..12];
    let etype   = r16(frame, 12);
    if etype != 0x0800 || frame.len() < 34 {
        dlog_dev!(LogModule::Net, "  {} non-IP etype={:#06x}", dir, etype); return;
    }
    let ip = &frame[14..];
    let ihl      = ((ip[0] & 0xf) as usize) * 4;
    let ip_total = r16(ip, 2) as usize;
    let proto    = ip[9];
    let src_ip   = Ipv4Addr::new(ip[12], ip[13], ip[14], ip[15]);
    let dst_ip   = Ipv4Addr::new(ip[16], ip[17], ip[18], ip[19]);
    if proto != IP_PROTO_TCP || ip_total < ihl + 20 || frame.len() < 14 + ihl + 20 {
        dlog_dev!(LogModule::Net, "  {} non-TCP proto={}", dir, proto); return;
    }
    let ip_end = ip_total.min(frame.len() - 14);
    let tcp   = &ip[ihl..ip_end]; // bounded by IP total length, clamped to frame
    let sport = r16(tcp, 0);
    let dport = r16(tcp, 2);
    let seq   = r32(tcp, 4);
    let ack   = r32(tcp, 8);
    let doff  = ((tcp[12] >> 4) as usize) * 4;
    let flags = tcp[13];
    let win   = r16(tcp, 14);
    let plen  = if doff <= tcp.len() { tcp.len() - doff } else { 0 };
    dlog_dev!(LogModule::Net, "  {} {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} → {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
              dir,
              src_mac[0],src_mac[1],src_mac[2],src_mac[3],src_mac[4],src_mac[5],
              dst_mac[0],dst_mac[1],dst_mac[2],dst_mac[3],dst_mac[4],dst_mac[5]);
    dlog_dev!(LogModule::Net, "     IP  {} → {}  (ihl={} total={})", src_ip, dst_ip, ihl, frame.len());
    dlog_dev!(LogModule::Net, "     TCP sport={} dport={}  {} seq={} ack={} win={} doff_byte={:#04x} tcp_seg_len={} plen={}",
              sport, dport, flags_str(flags), seq, ack, win, tcp[12], tcp.len(), plen);
    if plen > 0 && doff <= tcp.len() {
        let data = &tcp[doff..];
        eprint!("     data:");
        for b in data { eprint!(" {:02x}", b); }
        dlog_dev!(LogModule::Net, "");
    }
}

fn tcp_segment(src_ip: Ipv4Addr, dst_ip: Ipv4Addr,
               sport: u16, dport: u16,
               seq: u32, ack: u32, flags: u8,
               payload: &[u8]) -> Vec<u8> {
    let mut seg = vec![0u8; 20 + payload.len()];
    w16(&mut seg, 0, sport); w16(&mut seg, 2, dport);
    w32(&mut seg, 4, seq);   w32(&mut seg, 8, ack);
    seg[12] = 0x50;
    seg[13] = flags;
    w16(&mut seg, 14, 65535);
    if !payload.is_empty() { seg[20..].copy_from_slice(payload); }
    let c = tcp_checksum(src_ip, dst_ip, &seg); w16(&mut seg, 16, c);
    seg
}

pub fn eth_frame(dst: &[u8; 6], src: &[u8; 6], etype: u16, payload: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(60.max(14 + payload.len()));
    f.extend_from_slice(dst); f.extend_from_slice(src);
    f.push((etype>>8) as u8); f.push(etype as u8);
    f.extend_from_slice(payload);
    // Pad to minimum Ethernet frame size (60 bytes excluding CRC)
    if f.len() < 60 { f.resize(60, 0); }
    f
}

fn ip_frame(dst_mac: &[u8; 6], gw_mac: &[u8; 6],
            src_ip: Ipv4Addr, dst_ip: Ipv4Addr, proto: u8, payload: &[u8]) -> Vec<u8> {
    let iph = ipv4_header(src_ip, dst_ip, proto, payload.len() as u16);
    let mut ip = iph.to_vec();
    ip.extend_from_slice(payload);
    eth_frame(dst_mac, gw_mac, ETHERTYPE_IP, &ip)
}

pub fn mac_str(m: &[u8; 6]) -> String {
    format!("{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}", m[0],m[1],m[2],m[3],m[4],m[5])
}

/// One-line summary of an Ethernet frame for debug logging.
pub fn eth_summary(frame: &[u8]) -> String {
    if frame.len() < 14 {
        return format!("<runt {} bytes>", frame.len());
    }
    let dst: &[u8; 6] = frame[0..6].try_into().unwrap();
    let src: &[u8; 6] = frame[6..12].try_into().unwrap();
    let etype = r16(frame, 12);

    let inner = match etype {
        ETHERTYPE_ARP if frame.len() >= 14 + 28 => {
            let a = &frame[14..];
            let spa = Ipv4Addr::new(a[14], a[15], a[16], a[17]);
            let tpa = Ipv4Addr::new(a[24], a[25], a[26], a[27]);
            let op = r16(a, 6);
            let op_str = if op == 1 { "who-has" } else { "is-at" };
            format!("ARP {} {} tell {}", op_str, tpa, spa)
        }
        ETHERTYPE_IP if frame.len() >= 34 => {
            let ip = &frame[14..];
            let ihl = ((ip[0] & 0xf) as usize) * 4;
            let proto = ip[9];
            let src_ip = Ipv4Addr::new(ip[12], ip[13], ip[14], ip[15]);
            let dst_ip = Ipv4Addr::new(ip[16], ip[17], ip[18], ip[19]);
            let proto_str = match proto {
                IP_PROTO_ICMP => "ICMP".to_string(),
                IP_PROTO_TCP  => {
                    if frame.len() >= 14 + ihl + 4 {
                        let t = &ip[ihl..];
                        format!("TCP :{}->{}", r16(t, 0), r16(t, 2))
                    } else { "TCP".to_string() }
                }
                IP_PROTO_UDP  => {
                    if frame.len() >= 14 + ihl + 4 {
                        let u = &ip[ihl..];
                        format!("UDP :{}->{}", r16(u, 0), r16(u, 2))
                    } else { "UDP".to_string() }
                }
                n => format!("proto={}", n),
            };
            format!("IPv4 {} > {}  {}", src_ip, dst_ip, proto_str)
        }
        _ => format!("etype={:#06x}", etype),
    };

    format!("{} > {}  {}  {} bytes", mac_str(src), mac_str(dst), inner, frame.len())
}

fn parse_dhcp_type(opts: &[u8]) -> Option<u8> {
    let mut i = 0;
    while i < opts.len() {
        let tag = opts[i];
        if tag == 255 { break; }
        if tag == 0   { i += 1; continue; }
        if i + 1 >= opts.len() { break; }
        let len = opts[i+1] as usize;
        if tag == 53 && len >= 1 && i+2 < opts.len() { return Some(opts[i+2]); }
        i += 2 + len;
    }
    None
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
}

impl NatControl {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            debug_tcp:  AtomicBool::new(false),
            debug_udp:  AtomicBool::new(false),
            debug_icmp: AtomicBool::new(false),
            snapshot:   Mutex::new(NatSnapshot::default()),
            reset_nat:  AtomicBool::new(false),
        })
    }
    pub fn dbg_tcp(&self)  -> bool { self.debug_tcp.load(Ordering::Relaxed) }
    pub fn dbg_udp(&self)  -> bool { self.debug_udp.load(Ordering::Relaxed) }
    pub fn dbg_icmp(&self) -> bool { self.debug_icmp.load(Ordering::Relaxed) }
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

// ── NAT engine ────────────────────────────────────────────────────────────────
pub struct NatEngine {
    config:  GatewayConfig,
    tx_cons: rtrb::Consumer<Vec<u8>>, // outbound frames from enet thread
    rx_prod: rtrb::Producer<Vec<u8>>, // inbound frames to enet thread
    rx_wake: Arc<(Mutex<()>, Condvar)>, // signal enet thread when rx_prod gets a frame
    tx_wake: Arc<(Mutex<()>, Condvar)>, // wait on this for new tx frames from enet thread
    running: Arc<AtomicBool>,
    ctl:     Arc<NatControl>,
    udp_nat:   HashMap<(u32, u16, u16), NatUdpEntry>,
    tcp_nat:   HashMap<(u32, u16, u16), NatTcpEntry>,
    tcp_tw:    HashMap<(u32, u16, u16), Instant>,  // TIME_WAIT: absorb final ACKs silently
    icmp_nat:  HashMap<(u32, u16), NatIcmpEntry>,  // key: (dst_ip, identifier)
    icmp_unavailable: bool,  // true after first failed raw socket creation (Windows non-admin)
}

impl NatEngine {
    pub fn new(config: GatewayConfig,
               tx_cons: rtrb::Consumer<Vec<u8>>,
               rx_prod: rtrb::Producer<Vec<u8>>,
               rx_wake: Arc<(Mutex<()>, Condvar)>,
               tx_wake: Arc<(Mutex<()>, Condvar)>,
               running: Arc<AtomicBool>,
               ctl:     Arc<NatControl>) -> Self {
        Self { config, tx_cons, rx_prod, rx_wake, tx_wake, running, ctl,
               udp_nat: HashMap::new(), tcp_nat: HashMap::new(), tcp_tw: HashMap::new(),
               icmp_nat: HashMap::new(), icmp_unavailable: false }
    }

    pub fn run(&mut self) {
        while self.running.load(Ordering::Relaxed) {
            // Wait for new TX frames from the enet thread, or timeout to poll sockets.
            // Timeout of 10ms is enough for UDP/TCP response polling.
            {
                let (lock, cvar) = &*self.tx_wake;
                let mut guard = lock.lock();
                let _ = cvar.wait_for(&mut guard, Duration::from_millis(10));
            }

            // Machine reset: flush all NAT tables, close all host sockets.
            if self.ctl.reset_nat.swap(false, Ordering::AcqRel) {
                self.tcp_nat.clear();  // drops all TcpStreams, closing connections
                self.tcp_tw.clear();
                self.udp_nat.clear();  // drops all UdpSockets
                self.icmp_nat.clear(); // drops all ICMP raw sockets
            }

            // Drain all pending outbound frames
            while let Ok(frame) = self.tx_cons.pop() {
                self.process(&frame);
            }

            self.poll_udp();
            self.poll_tcp();
            self.poll_icmp();
            self.update_snapshot();
        }
    }

    fn update_snapshot(&self) {
        let now = Instant::now();
        let mut snap = self.ctl.snapshot.lock();
        snap.tcp = self.tcp_nat.iter().map(|(&(ip, rport, cport), e)| NatTcpInfo {
            remote_ip:        Ipv4Addr::from(ip).to_string(),
            remote_port:      rport,
            client_port:      cport,
            age_secs:         now.duration_since(e.last_use).as_secs(),
            server_seq:       e.server_seq,
            server_seq_acked: e.server_seq_acked,
            client_win:       e.client_win,
            client_seq:       e.client_seq,
            fin_wait:         e.fin_wait,
            rtx_count:        e.retransmit.len(),
            rtx_bytes:        e.retransmit.iter().map(|r| r.data.len()).sum(),
        }).collect();
        snap.udp = self.udp_nat.iter().map(|(&(ip, rport, cport), e)| NatUdpInfo {
            remote_ip:   Ipv4Addr::from(ip).to_string(),
            remote_port: rport,
            client_port: cport,
            age_secs:    now.duration_since(e.last_use).as_secs(),
        }).collect();
        snap.icmp = self.icmp_nat.iter().map(|(&(ip, ident), e)| NatIcmpInfo {
            remote_ip: Ipv4Addr::from(ip).to_string(),
            ident,
            age_secs:  now.duration_since(e.last_use).as_secs(),
        }).collect();
    }

    fn enqueue_rx(&mut self, frame: Vec<u8>) {
        if self.ctl.dbg_tcp() && r16(&frame, 12) == ETHERTYPE_IP {
            dlog_dev!(LogModule::Net, "NAT TX (NAT→IRIX):");
            log_eth_frame("<<", &frame);
        }
        // If the ring is full, drop the frame rather than block
        let _ = self.rx_prod.push(frame);
        // Wake the enet thread so it drains rx_prod promptly
        self.rx_wake.1.notify_one();
    }

    fn process(&mut self, frame: &[u8]) {
        if frame.len() < 14 { return; }
        let src_mac: [u8; 6] = frame[6..12].try_into().unwrap();
        let etype = r16(frame, 12);
        dlog_dev!(LogModule::Net, "NAT TX {}", eth_summary(frame));
        if self.ctl.dbg_tcp() && etype == ETHERTYPE_IP {
            dlog_dev!(LogModule::Net, "NAT RX (IRIX→NAT):");
            log_eth_frame(">>", frame);
        }
        match etype {
            ETHERTYPE_ARP => self.handle_arp(frame, &src_mac),
            ETHERTYPE_IP  => self.handle_ip(frame, &src_mac),
            _ => {}
        }
    }

    // ── ARP ───────────────────────────────────────────────────────────────────
    fn handle_arp(&mut self, frame: &[u8], _src_mac: &[u8; 6]) {
        if frame.len() < 14 + 28 { return; }
        let a = &frame[14..];
        if r16(a,0) != ARP_HW_ETHER || r16(a,2) != ARP_PROTO_IP
           || a[4] != 6 || a[5] != 4 || r16(a,6) != ARP_OP_REQUEST { return; }

        let sender_mac: [u8; 6] = a[8..14].try_into().unwrap();
        let sender_ip = Ipv4Addr::new(a[14], a[15], a[16], a[17]);
        // ARP layout: sha[8..14] spa[14..18] tha[18..24] tpa[24..28]
        let target_ip = Ipv4Addr::new(a[24], a[25], a[26], a[27]);

        if target_ip != self.config.gateway_ip { return; }

        dlog_dev!(LogModule::Net, "NAT ARP: who-has {} tell {}", target_ip, sender_ip);

        let mut arp = [0u8; 28];
        w16(&mut arp, 0, ARP_HW_ETHER); w16(&mut arp, 2, ARP_PROTO_IP);
        arp[4] = 6; arp[5] = 4;
        w16(&mut arp, 6, ARP_OP_REPLY);
        arp[8..14].copy_from_slice(&self.config.gateway_mac);
        arp[14..18].copy_from_slice(&self.config.gateway_ip.octets());
        arp[18..24].copy_from_slice(&sender_mac);
        arp[24..28].copy_from_slice(&sender_ip.octets());

        let reply = eth_frame(&sender_mac, &self.config.gateway_mac, ETHERTYPE_ARP, &arp);
        self.enqueue_rx(reply);
    }

    // ── IP dispatch ───────────────────────────────────────────────────────────
    fn handle_ip(&mut self, frame: &[u8], src_mac: &[u8; 6]) {
        if frame.len() < 34 { return; }
        let ip = &frame[14..];
        let ihl = ((ip[0] & 0xf) as usize) * 4;
        if frame.len() < 14 + ihl { return; }
        let proto  = ip[9];
        let src_ip = Ipv4Addr::new(ip[12], ip[13], ip[14], ip[15]);
        let dst_ip = Ipv4Addr::new(ip[16], ip[17], ip[18], ip[19]);
        // ip_total is the total IP datagram length (header + data).
        // Use it to strip Ethernet padding (frames padded to 60-byte minimum).
        let ip_total = r16(ip, 2) as usize;
        if ip_total < ihl || frame.len() < 14 + ihl { return; }
        // Clamp to actual frame size in case ip_total > frame bytes available.
        let ip_end = ip_total.min(frame.len() - 14);
        let payload = &ip[ihl..ip_end];
        match proto {
            IP_PROTO_ICMP => self.handle_icmp(src_mac, src_ip, dst_ip, payload),
            IP_PROTO_UDP  => self.handle_udp(src_mac, src_ip, dst_ip, payload),
            IP_PROTO_TCP  => self.handle_tcp(src_mac, src_ip, dst_ip, payload),
            _ => {}
        }
    }

    // ── ICMP echo ─────────────────────────────────────────────────────────────
    fn handle_icmp(&mut self, src_mac: &[u8; 6], src_ip: Ipv4Addr, dst_ip: Ipv4Addr, payload: &[u8]) {
        if payload.len() < 8 || payload[0] != ICMP_ECHO_REQUEST { return; }
        let ident = r16(payload, 4);
        let seq   = r16(payload, 6);

        // If destination is our gateway, reply locally without hitting the network.
        if dst_ip == self.config.gateway_ip {
            dlog_dev!(LogModule::Net, "NAT ICMP {} → {} ident={} seq={} (local reply)", src_ip, dst_ip, ident, seq);
            let mut icmp = payload.to_vec();
            icmp[0] = ICMP_ECHO_REPLY;
            icmp[2] = 0; icmp[3] = 0;
            let c = ip_checksum(&icmp); w16(&mut icmp, 2, c);
            let frame = ip_frame(src_mac, &self.config.gateway_mac,
                                 self.config.gateway_ip, src_ip, IP_PROTO_ICMP, &icmp);
            self.enqueue_rx(frame);
            return;
        }

        // Forward to external host via ICMP socket.
        // Linux supports unprivileged SOCK_DGRAM+ICMPV4 (kernel ≥3.11).
        // Windows requires SOCK_RAW+ICMPV4 (needs admin privileges) and returns
        // replies with a prepended IP header that must be stripped on recv.
        let is_new = !self.icmp_nat.contains_key(&(u32::from(dst_ip), ident));
        dlog_dev!(LogModule::Net, "NAT ICMP {} → {} ident={} seq={}{}", src_ip, dst_ip, ident, seq,
            if is_new { " [new]" } else { "" });
        if self.icmp_unavailable { return; }
        let key = (u32::from(dst_ip), ident);
        let entry = self.icmp_nat.entry(key).or_insert_with(|| {
            #[cfg(not(windows))]
            let sock_type = Type::DGRAM;
            #[cfg(windows)]
            let sock_type = Type::RAW;
            let sock = match Socket::new(Domain::IPV4, sock_type, Some(Protocol::ICMPV4)) {
                Ok(s) => { let _ = s.set_nonblocking(true); Some(s) }
                Err(e) => {
                    eprintln!("iris: ICMP unavailable ({}); ping will time out. \
                        On Windows, run as Administrator to enable raw ICMP.", e);
                    None
                }
            };
            NatIcmpEntry { sock, client_mac: *src_mac, client_ip: src_ip, last_use: Instant::now() }
        });
        if entry.sock.is_none() {
            self.icmp_unavailable = true;
            return;
        }
        entry.last_use = Instant::now();
        let dest = SocketAddr::new(IpAddr::V4(dst_ip), 0);
        let _ = entry.sock.as_ref().unwrap().send_to(payload, &dest.into());
    }

    fn poll_icmp(&mut self) {
        let mut expired = Vec::new();
        // (icmp_reply_bytes, key)
        let mut replies: Vec<(Vec<u8>, (u32, u16))> = Vec::new();
        for (&key, entry) in &mut self.icmp_nat {
            if entry.last_use.elapsed() > Duration::from_secs(30) {
                expired.push(key); continue;
            }
            let Some(sock) = &mut entry.sock else { continue };
            let mut buf = [std::mem::MaybeUninit::<u8>::uninit(); 1500];
            // Linux SOCK_DGRAM ICMP: kernel delivers ICMP payload only (no IP header).
            // Windows SOCK_RAW ICMP: kernel prepends the IP header; strip it.
            while let Ok(n) = sock.recv(&mut buf) {
                let raw: Vec<u8> = buf[..n].iter().map(|b| unsafe { b.assume_init() }).collect();
                #[cfg(windows)]
                let data = {
                    // IP header length is in the low nibble of byte 0, in 32-bit words.
                    let ihl = ((raw.first().copied().unwrap_or(0x45) & 0x0f) as usize) * 4;
                    if raw.len() > ihl { raw[ihl..].to_vec() } else { continue }
                };
                #[cfg(not(windows))]
                let data = raw;
                replies.push((data, key));
            }
        }
        for k in expired { self.icmp_nat.remove(&k); }
        for (mut icmp, key) in replies {
            let (dst_ip_u32, ident) = key;
            if let Some(entry) = self.icmp_nat.get(&key) {
                let remote_ip = Ipv4Addr::from(dst_ip_u32);
                let client_mac = entry.client_mac;
                let client_ip  = entry.client_ip;
                // Restore the original guest identifier (kernel may have rewritten it)
                // and recompute the ICMP checksum.
                if icmp.len() >= 8 {
                    w16(&mut icmp, 4, ident);
                    icmp[2] = 0; icmp[3] = 0;
                    let c = ip_checksum(&icmp);
                    w16(&mut icmp, 2, c);
                }
                {
                    let seq = if icmp.len() >= 8 { r16(&icmp, 6) } else { 0 };
                    dlog_dev!(LogModule::Net, "NAT ICMP reply {} → {} ident={} seq={}", remote_ip, client_ip, ident, seq);
                }
                let frame = ip_frame(&client_mac, &self.config.gateway_mac,
                                     remote_ip, client_ip, IP_PROTO_ICMP, &icmp);
                self.enqueue_rx(frame);
            }
        }
    }

    // ── UDP dispatch ──────────────────────────────────────────────────────────
    fn handle_udp(&mut self, src_mac: &[u8; 6], src_ip: Ipv4Addr, dst_ip: Ipv4Addr, udp: &[u8]) {
        if udp.len() < 8 { return; }
        let sport = r16(udp, 0);
        let dport = r16(udp, 2);
        let payload = &udp[8..];
        dlog_dev!(LogModule::Net, "NAT UDP {}:{} → {}:{}", src_ip, sport, dst_ip, dport);
        match dport {
            UDP_PORT_BOOTP_SERVER => self.handle_bootp(src_mac, sport, payload),
            UDP_PORT_DNS => self.forward_dns(src_mac, src_ip, sport, payload),
            _ => self.nat_udp(src_mac, src_ip, dst_ip, sport, dport, payload),
        }
    }

    // ── BOOTP / DHCP ──────────────────────────────────────────────────────────
    fn handle_bootp(&mut self, client_mac: &[u8; 6], _client_port: u16, payload: &[u8]) {
        if payload.len() < 236 || payload[0] != BOOTP_OP_REQUEST { return; }
        let xid = r32(payload, 4);
        let chaddr: [u8; 6] = payload[28..34].try_into().unwrap();

        let is_dhcp = payload.len() >= 240
            && &payload[236..240] == &[99, 130, 83, 99];

        let dhcp_type = if is_dhcp { parse_dhcp_type(&payload[240..]) } else { None };

        dlog_dev!(LogModule::Net, "NAT BOOTP xid={:#010x} mac={} dhcp_type={:?}",
                      xid, mac_str(&chaddr), dhcp_type);

        let reply_dhcp_type: Option<u8> = match dhcp_type {
            Some(1) => Some(2),
            Some(3) => Some(5),
            None    => None,
            _       => return,
        };

        let mut rep = vec![0u8; 300];
        rep[0] = 2;
        rep[1] = 1; rep[2] = 6;
        w32(&mut rep, 4, xid);
        rep[16..20].copy_from_slice(&self.config.client_ip.octets());
        rep[20..24].copy_from_slice(&self.config.gateway_ip.octets());
        rep[28..34].copy_from_slice(&chaddr);

        if is_dhcp {
            rep[236..240].copy_from_slice(&[99, 130, 83, 99]);
            let mut o = 240usize;
            if let Some(t) = reply_dhcp_type {
                rep[o]=53; rep[o+1]=1; rep[o+2]=t; o+=3;
            }
            rep[o]=1; rep[o+1]=4;
            rep[o+2..o+6].copy_from_slice(&self.config.netmask.octets()); o+=6;
            rep[o]=3; rep[o+1]=4;
            rep[o+2..o+6].copy_from_slice(&self.config.gateway_ip.octets()); o+=6;
            let dns_ip = match self.config.dns_upstream.ip() {
                IpAddr::V4(ip) => ip,
                _              => Ipv4Addr::new(8,8,8,8),
            };
            rep[o]=6; rep[o+1]=4;
            rep[o+2..o+6].copy_from_slice(&dns_ip.octets()); o+=6;
            rep[o]=51; rep[o+1]=4; w32(&mut rep, o+2, 86400); o+=6;
            rep[o]=54; rep[o+1]=4;
            rep[o+2..o+6].copy_from_slice(&self.config.gateway_ip.octets()); o+=6;
            rep[o]=255;
        }

        let udp = udp_packet(self.config.gateway_ip, Ipv4Addr::BROADCAST,
                             UDP_PORT_BOOTP_SERVER, UDP_PORT_BOOTP_CLIENT, &rep);
        let frame = ip_frame(client_mac, &self.config.gateway_mac,
                             self.config.gateway_ip, Ipv4Addr::BROADCAST,
                             IP_PROTO_UDP, &udp);
        self.enqueue_rx(frame);
    }

    // ── DNS forwarding ────────────────────────────────────────────────────────
    fn forward_dns(&mut self, client_mac: &[u8; 6], client_ip: Ipv4Addr, client_port: u16, query: &[u8]) {
        dlog_dev!(LogModule::Net, "NAT DNS forward len={}", query.len());
        let Ok(sock) = UdpSocket::bind("0.0.0.0:0") else { return; };
        let _ = sock.set_read_timeout(Some(Duration::from_secs(2)));
        if sock.send_to(query, self.config.dns_upstream).is_err() { return; }
        let mut buf = [0u8; 512];
        if let Ok((n, _)) = sock.recv_from(&mut buf) {
            let udp = udp_packet(self.config.gateway_ip, client_ip,
                                 UDP_PORT_DNS, client_port, &buf[..n]);
            let frame = ip_frame(client_mac, &self.config.gateway_mac,
                                 self.config.gateway_ip, client_ip, IP_PROTO_UDP, &udp);
            self.enqueue_rx(frame);
        }
    }

    // ── UDP NAT ───────────────────────────────────────────────────────────────
    fn nat_udp(&mut self, client_mac: &[u8; 6], src_ip: Ipv4Addr, dst_ip: Ipv4Addr,
               sport: u16, dport: u16, payload: &[u8]) {
        let key = (u32::from(dst_ip), dport, sport);
        let is_new = !self.udp_nat.contains_key(&key);
        dlog_dev!(LogModule::Net, "NAT UDP {}:{} → {}:{} len={}{}", src_ip, sport, dst_ip, dport, payload.len(),
            if is_new { " [new]" } else { "" });
        let entry = self.udp_nat.entry(key).or_insert_with(|| {
            let sock = UdpSocket::bind("0.0.0.0:0").expect("UDP NAT bind");
            let _ = sock.set_nonblocking(true);
            NatUdpEntry { sock, client_mac: *client_mac, client_ip: src_ip,
                          client_port: sport, last_use: Instant::now() }
        });
        entry.last_use = Instant::now();
        let _ = entry.sock.send_to(payload, SocketAddr::new(IpAddr::V4(dst_ip), dport));
    }

    fn poll_udp(&mut self) {
        let mut expired = Vec::new();
        let mut responses: Vec<(Vec<u8>, (u32, u16, u16))> = Vec::new();
        for (&key, entry) in &mut self.udp_nat {
            if entry.last_use.elapsed() > Duration::from_secs(30) {
                dlog_dev!(LogModule::Net, "NAT UDP {}:{} expired", Ipv4Addr::from(key.0), key.1);
                expired.push(key); continue;
            }
            let mut buf = [0u8; 1500];
            while let Ok((n, from)) = entry.sock.recv_from(&mut buf) {
                dlog_dev!(LogModule::Net, "NAT UDP reply {} → {}:{} len={}", from, entry.client_ip, key.2, n);
                responses.push((buf[..n].to_vec(), key));
            }
        }
        for k in expired { self.udp_nat.remove(&k); }
        for (data, key) in responses {
            let (dst_ip_u32, dst_port, client_port) = key;
            if let Some(entry) = self.udp_nat.get(&key) {
                let remote_ip = Ipv4Addr::from(dst_ip_u32);
                let udp = udp_packet(remote_ip, entry.client_ip, dst_port, client_port, &data);
                let client_mac = entry.client_mac;
                let client_ip = entry.client_ip;
                let frame = ip_frame(&client_mac, &self.config.gateway_mac,
                                     remote_ip, client_ip, IP_PROTO_UDP, &udp);
                self.enqueue_rx(frame);
            }
        }
    }

    // ── TCP NAT ───────────────────────────────────────────────────────────────
    fn handle_tcp(&mut self, client_mac: &[u8; 6], src_ip: Ipv4Addr, dst_ip: Ipv4Addr, tcp: &[u8]) {
        if tcp.len() < 20 { return; }
        let sport    = r16(tcp, 0);
        let dport    = r16(tcp, 2);
        let seq      = r32(tcp, 4);
        let _ack_num = r32(tcp, 8);
        let doff     = ((tcp[12] >> 4) as usize) * 4;
        let flags    = tcp[13];
        let payload  = if doff <= tcp.len() { &tcp[doff..] } else { &[] };
        let syn      = flags & 0x02 != 0;
        let ack      = flags & 0x10 != 0;
        let fin      = flags & 0x01 != 0;
        let rst      = flags & 0x04 != 0;


        let key = (u32::from(dst_ip), dport, sport);

        if syn && !ack {
            dlog_dev!(LogModule::Net, "NAT TCP connect {}:{} → {}:{}", src_ip, sport, dst_ip, dport);
            let dest = SocketAddr::new(IpAddr::V4(dst_ip), dport);
            match TcpStream::connect_timeout(&dest, Duration::from_secs(5)) {
                Ok(stream) => {
                    let _ = stream.set_nonblocking(true);
                    let server_seq = 0x4000_0000u32;
                    dlog_dev!(LogModule::Net, "NAT TCP connected {}:{} → {}:{}", src_ip, sport, dst_ip, dport);
                    self.tcp_nat.insert(key, NatTcpEntry {
                        stream, client_mac: *client_mac, client_ip: src_ip,
                        client_port: sport, server_ip: dst_ip,
                        server_seq: server_seq.wrapping_add(1),
                        server_seq_acked: server_seq.wrapping_add(1),
                        client_win: r16(tcp, 14) as u32,
                        client_seq: seq.wrapping_add(1),
                        last_use: Instant::now(),
                        fin_wait: false,
                        server_fin: false,
                        retransmit: VecDeque::new(),
                    });
                    let seg = tcp_segment(dst_ip, src_ip, dport, sport,
                                         server_seq, seq.wrapping_add(1), 0x12, &[]);
                    let frame = ip_frame(client_mac, &self.config.gateway_mac,
                                        dst_ip, src_ip, IP_PROTO_TCP, &seg);
                    self.enqueue_rx(frame);
                }
                Err(e) => {
                    dlog_dev!(LogModule::Net, "NAT TCP connect {}:{} failed: {}", dst_ip, dport, e);
                    let seg = tcp_segment(dst_ip, src_ip, dport, sport,
                                         0, seq.wrapping_add(1), 0x14, &[]);
                    let frame = ip_frame(client_mac, &self.config.gateway_mac,
                                        dst_ip, src_ip, IP_PROTO_TCP, &seg);
                    self.enqueue_rx(frame);
                }
            }
            return;
        }

        if rst {
            dlog_dev!(LogModule::Net, "NAT TCP RST {}:{} → {}:{}", src_ip, sport, dst_ip, dport);
            self.tcp_nat.remove(&key); return;
        }

        let entry = match self.tcp_nat.get_mut(&key) {
            Some(e) => e,
            None => {
                // If in TIME_WAIT, silently absorb the packet (typically IRIX's final ACK).
                if self.tcp_tw.contains_key(&key) {
                    return;
                }
                dlog_dev!(LogModule::Net, "NAT TCP no entry for {}:{} → {}:{} flags={:#04x} — sending RST",
                          src_ip, sport, dst_ip, dport, flags);
                // Send RST so IRIX closes the connection cleanly instead of retransmitting.
                let ack_num = r32(tcp, 8);
                let rst_seq = if ack { ack_num } else { 0 };
                let seg = tcp_segment(dst_ip, src_ip, dport, sport,
                                      rst_seq, seq.wrapping_add(payload.len() as u32 + if fin { 1 } else { 0 }),
                                      0x14, &[]);
                let frame = ip_frame(client_mac, &self.config.gateway_mac,
                                     dst_ip, src_ip, IP_PROTO_TCP, &seg);
                self.enqueue_rx(frame);
                return;
            }
        };
        entry.last_use = Instant::now();

        // Update IRIX's window and what it has ACKed from us.
        if ack {
            let ack_num = r32(tcp, 8);
            let win = r16(tcp, 14) as u32;
            let in_flight = entry.server_seq.wrapping_sub(entry.server_seq_acked);
            dlog_dev!(LogModule::Net, "NAT TCP ACK from IRIX :{}  ack={:#010x} win={}  srv_seq={:#010x} srv_acked={:#010x} in_flight={}",
                      sport, ack_num, win, entry.server_seq, entry.server_seq_acked, in_flight);
            // Only advance if this ACK is newer (wrapping compare).
            if ack_num.wrapping_sub(entry.server_seq_acked) <= 0x8000_0000 {
                entry.server_seq_acked = ack_num;
            }
            entry.client_win = win;
            // Drain retransmit queue: drop entries fully covered by ack_num.
            while let Some(front) = entry.retransmit.front() {
                let end = front.seq.wrapping_add(front.data.len() as u32);
                if ack_num.wrapping_sub(end) <= 0x8000_0000 {
                    entry.retransmit.pop_front();
                } else {
                    break;
                }
            }
        }

        if !payload.is_empty() {
            // Detect retransmit: if seq is already ACKed, just re-ACK, don't write again.
            let already_acked = entry.client_seq.wrapping_sub(seq) <= 0x8000_0000
                                && seq != entry.client_seq;
            if !already_acked {
                use std::io::Write as _;
                let _ = entry.stream.write_all(payload);
                entry.client_seq = seq.wrapping_add(payload.len() as u32);
            }
            let sip  = entry.server_ip;
            let seg = tcp_segment(sip, entry.client_ip,
                                  dport, entry.client_port,
                                  entry.server_seq, entry.client_seq, 0x10, &[]);
            let cmac = entry.client_mac;
            let cip  = entry.client_ip;
            let frame = ip_frame(&cmac, &self.config.gateway_mac,
                                 sip, cip, IP_PROTO_TCP, &seg);
            self.enqueue_rx(frame);
        }

        if fin {
            let frame = {
                let entry = self.tcp_nat.get_mut(&key).unwrap();
                dlog_dev!(LogModule::Net, "NAT TCP FIN {}:{} → {}:{}", src_ip, sport, entry.server_ip, dport);
                entry.client_seq = entry.client_seq.wrapping_add(1);
                // ACK IRIX's FIN so it stops retransmitting.
                let sip  = entry.server_ip;
                let seg = tcp_segment(sip, entry.client_ip,
                                      dport, entry.client_port,
                                      entry.server_seq, entry.client_seq, 0x10, &[]);
                let cmac = entry.client_mac;
                let cip  = entry.client_ip;
                ip_frame(&cmac, &self.config.gateway_mac, sip, cip, IP_PROTO_TCP, &seg)
            };
            self.enqueue_rx(frame);
            // Shut down our write side so the server sees EOF, then wait for server's FIN.
            let entry = self.tcp_nat.get_mut(&key).unwrap();
            use std::net::Shutdown;
            let _ = entry.stream.shutdown(Shutdown::Write);
            entry.fin_wait = true;
        }
    }

    fn poll_tcp(&mut self) {
        let mut expired  = Vec::new();   // timed out — just remove

        for (&key, entry) in &mut self.tcp_nat {
            let timeout = if entry.fin_wait { Duration::from_secs(10) } else { Duration::from_secs(300) };
            if entry.last_use.elapsed() > timeout {
                dlog_dev!(LogModule::Net, "NAT TCP timeout {}:{} → {}:{}", entry.client_ip, key.2, Ipv4Addr::from(key.0), key.1);
                expired.push(key); continue;
            }
            let (_, dport, sport) = key;
            let sip = entry.server_ip;

            // ── Retransmit timed-out segments ────────────────────────────────
            for rt in &mut entry.retransmit {
                if rt.sent_at.elapsed() < RTO { break; }  // queue is ordered; stop at first fresh entry
                if self.rx_prod.slots() < 5 { break; }
                dlog_dev!(LogModule::Net, "NAT TCP RETRANSMIT {}:{} → {}:{} seq={} len={}",
                          sip, dport, entry.client_ip, sport, rt.seq, rt.data.len());
                let seg = tcp_segment(sip, entry.client_ip, dport, sport,
                                      rt.seq, entry.client_seq, 0x18, &rt.data);
                let frame = ip_frame(&entry.client_mac, &self.config.gateway_mac,
                                     sip, entry.client_ip, IP_PROTO_TCP, &seg);
                let _ = self.rx_prod.push(frame);
                self.rx_wake.1.notify_one();
                rt.sent_at = Instant::now();
            }

            // ── Read new data from server ─────────────────────────────────────
            let mut buf = [0u8; 1460];
            use std::io::Read as _;
            loop {
                // Respect IRIX's receive window: don't send more than it can buffer.
                let in_flight = entry.server_seq.wrapping_sub(entry.server_seq_acked);
                let window_remaining = entry.client_win.saturating_sub(in_flight);
                if window_remaining == 0 { break; }  // zero-window: wait for ACK
                // Stop if the RX ring is nearly full — avoids reading bytes we'd have
                // to drop, which would advance server_seq without IRIX ever seeing the data.
                if self.rx_prod.slots() < 5 { break; }
                let read_max = (window_remaining as usize).min(buf.len());
                match entry.stream.read(&mut buf[..read_max]) {
                    Ok(0) => {
                        dlog_dev!(LogModule::Net, "NAT TCP server closed {}:{} → {}:{} fin_wait={}",
                                  Ipv4Addr::from(key.0), key.1, entry.client_ip, key.2, entry.fin_wait);
                        entry.server_fin = true; break;
                    }
                    Ok(n) => {
                        let seq = entry.server_seq;
                        entry.server_seq = seq.wrapping_add(n as u32);
                        dlog_dev!(LogModule::Net, "NAT TCP poll_tcp PUSH {}:{} → {}:{} seq={} ack={} len={} win_rem={}",
                                  sip, dport, entry.client_ip, sport, seq, entry.client_seq, n, window_remaining);
                        let data = buf[..n].to_vec();
                        let seg = tcp_segment(sip, entry.client_ip, dport, sport,
                                              seq, entry.client_seq, 0x18, &data);
                        let frame = ip_frame(&entry.client_mac, &self.config.gateway_mac,
                                             sip, entry.client_ip, IP_PROTO_TCP, &seg);
                        let _ = self.rx_prod.push(frame);
                        self.rx_wake.1.notify_one();
                        entry.retransmit.push_back(RetransmitEntry { seq, data, sent_at: Instant::now() });
                    }
                    Err(_) => break,
                }
            }
        }
        for k in expired { self.tcp_nat.remove(&k); }
        // Expire old TIME_WAIT entries (4 seconds is plenty for a LAN).
        self.tcp_tw.retain(|_, t| t.elapsed() < Duration::from_secs(4));
        // Send FIN+ACK to IRIX for connections where the server closed.
        // If the ring is full, leave server_fin set and retry next poll cycle.
        let mut fin_sent = Vec::new();
        for (&k, entry) in &mut self.tcp_nat {
            if !entry.server_fin { continue; }
            // Wait until IRIX has ACKed all data before sending FIN — otherwise IRIX
            // may discard buffered data when it receives FIN and closes the connection.
            let in_flight = entry.server_seq.wrapping_sub(entry.server_seq_acked);
            if in_flight > 0 { continue; }
            if self.rx_prod.slots() < 1 { continue; }
            let sip = entry.server_ip;
            let seg = tcp_segment(sip, entry.client_ip,
                                  k.1, entry.client_port,
                                  entry.server_seq, entry.client_seq, 0x11, &[]);
            let frame = ip_frame(&entry.client_mac, &self.config.gateway_mac,
                                 sip, entry.client_ip, IP_PROTO_TCP, &seg);
            let _ = self.rx_prod.push(frame);
            self.rx_wake.1.notify_one();
            fin_sent.push(k);
        }
        for k in fin_sent {
            self.tcp_nat.remove(&k);
            self.tcp_tw.insert(k, Instant::now());
        }
    }
}
