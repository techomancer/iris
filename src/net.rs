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
use crate::config::NfsConfig;
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
const UDP_PORT_PORTMAP:      u16 = 111;
const BOOTP_OP_REQUEST: u8      = 1;

// NFS-visible ports (what IRIX thinks the server is on)
const NFS_VM_PORT:    u16 = 2049;
const MOUNTD_VM_PORT: u16 = 1234;

// RPC program numbers
const RPC_PROG_PORTMAP:  u32 = 100000;
const RPC_PROG_NFS:      u32 = 100003;
const RPC_PROG_MOUNTD:   u32 = 100005;
const RPC_PORTMAP_GETPORT: u32 = 3;

// ── Gateway configuration ─────────────────────────────────────────────────────
#[derive(Clone)]
pub struct GatewayConfig {
    pub gateway_mac: [u8; 6],
    pub gateway_ip:  Ipv4Addr,
    pub client_ip:   Ipv4Addr,
    pub netmask:     Ipv4Addr,
    pub dns_upstream: SocketAddr,
    /// NFS configuration; if Some, portmap and NAT redirection for NFS/mountd are enabled.
    pub nfs: Option<NfsConfig>,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            gateway_mac: [0x02, 0x00, 0xDE, 0xAD, 0xBE, 0xEF],
            gateway_ip:  Ipv4Addr::new(192, 168, 0, 1),
            client_ip:   Ipv4Addr::new(192, 168, 0, 2),
            netmask:     Ipv4Addr::new(255, 255, 255, 0),
            dns_upstream: "8.8.8.8:53".parse().unwrap(),
            nfs: None,
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
        let hex: String = data.iter().map(|b| format!(" {:02x}", b)).collect();
        dlog_dev!(LogModule::Net, "     data:{}", hex);
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

// ── Portmap helpers ───────────────────────────────────────────────────────────

/// Parse an RPC GETPORT request and return the VM-visible port, or 0 if unknown.
fn portmap_lookup(payload: &[u8], nfs: &NfsConfig) -> u32 {
    // Need at least 14 u32s = 56 bytes for a well-formed GETPORT call.
    if payload.len() < 56 { return 0; }
    let msg_type = r32(payload,  4);
    let rpcvers  = r32(payload,  8);
    let prog     = r32(payload, 12);
    let proc_num = r32(payload, 20);
    if msg_type != 0 || rpcvers != 2 { return 0; }
    if prog != RPC_PROG_PORTMAP { return 0; }
    if proc_num != RPC_PORTMAP_GETPORT { return 0; }
    // cred: [24]=flavor [28]=len; verf: [32]=flavor [36]=len  (both len=0)
    let cred_len = r32(payload, 28) as usize;
    let verf_off = 32 + cred_len;
    if payload.len() < verf_off + 8 + 16 { return 0; }
    let verf_len = r32(payload, verf_off + 4) as usize;
    let args_off = verf_off + 8 + verf_len;
    if payload.len() < args_off + 16 { return 0; }
    let req_prog = r32(payload, args_off);
    let _ = nfs;
    match req_prog {
        RPC_PROG_NFS    => NFS_VM_PORT as u32,
        RPC_PROG_MOUNTD => MOUNTD_VM_PORT as u32,
        RPC_PROG_PORTMAP => UDP_PORT_PORTMAP as u32,
        _ => 0,
    }
}

/// Build an RPC PORTMAP GETPORT reply with the given xid and port value.
fn portmap_reply(xid: u32, port: u32) -> Vec<u8> {
    let mut r = vec![0u8; 28];
    w32(&mut r,  0, xid);
    w32(&mut r,  4, 1);    // REPLY
    w32(&mut r,  8, 0);    // MSG_ACCEPTED
    w32(&mut r, 12, 0);    // verf_flavor = AUTH_NULL
    w32(&mut r, 16, 0);    // verf_len = 0
    w32(&mut r, 20, 0);    // accept_stat = SUCCESS
    w32(&mut r, 24, port);
    r
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
        let ttl    = ip[8];
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
            IP_PROTO_ICMP => self.handle_icmp(src_mac, src_ip, dst_ip, ttl, payload),
            IP_PROTO_UDP  => self.handle_udp(src_mac, src_ip, dst_ip, payload),
            IP_PROTO_TCP  => self.handle_tcp(src_mac, src_ip, dst_ip, payload),
            _ => {}
        }
    }

    // ── ICMP echo ─────────────────────────────────────────────────────────────
    fn handle_icmp(&mut self, src_mac: &[u8; 6], src_ip: Ipv4Addr, dst_ip: Ipv4Addr, ttl: u8, payload: &[u8]) {
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
        // Linux: unprivileged SOCK_DGRAM+ICMPV4 works (kernel ≥3.11) but Time Exceeded
        //   replies are not delivered — traceroute sees * * * for intermediate hops.
        // macOS: SOCK_DGRAM+ICMPV4 requires root; falls back gracefully if unavailable.
        // Windows: SOCK_RAW+ICMPV4 requires admin; Time Exceeded IS delivered on recv,
        //   so traceroute works correctly when running as Administrator.
        let is_new = !self.icmp_nat.contains_key(&(u32::from(dst_ip), ident));
        dlog_dev!(LogModule::Net, "NAT ICMP {} → {} ident={} seq={} ttl={}{}", src_ip, dst_ip, ident, seq, ttl,
            if is_new { " [new]" } else { "" });
        if self.icmp_unavailable { return; }
        let key = (u32::from(dst_ip), ident);
        let entry = self.icmp_nat.entry(key).or_insert_with(|| {
            // Linux: unprivileged SOCK_DGRAM+ICMPV4 works (kernel ≥3.11).
            // Windows/macOS: need SOCK_RAW+ICMPV4, which requires admin/root.
            #[cfg(target_os = "linux")]
            let sock_type = Type::DGRAM;
            #[cfg(not(target_os = "linux"))]
            let sock_type = Type::RAW;
            let sock = match Socket::new(Domain::IPV4, sock_type, Some(Protocol::ICMPV4)) {
                Ok(s) => { let _ = s.set_nonblocking(true); Some(s) }
                Err(e) => {
                    #[cfg(windows)]
                    eprintln!("iris: ICMP unavailable ({}); ping will time out. \
                        Run as Administrator to enable raw ICMP.", e);
                    #[cfg(target_os = "macos")]
                    eprintln!("iris: ICMP unavailable ({}); ping will time out. \
                        Run as root (sudo) to enable raw ICMP.", e);
                    #[cfg(target_os = "linux")]
                    eprintln!("iris: ICMP unavailable ({}); ping will time out.", e);
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
        let sock = entry.sock.as_ref().unwrap();
        // Preserve the guest's TTL so intermediate routers respond with Time Exceeded
        // at the right hop count.  On Windows/macOS (SOCK_RAW) those replies arrive back
        // on this socket and we forward them to the guest.  On Linux (SOCK_DGRAM) they
        // are silently dropped by the kernel — traceroute sees * * *.
        let _ = sock.set_ttl(ttl as u32);
        let dest = SocketAddr::new(IpAddr::V4(dst_ip), 0);
        let _ = sock.send_to(payload, &dest.into());
    }

    fn poll_icmp(&mut self) {
        let mut expired = Vec::new();
        // (icmp_payload, outer_src_ip_u32, key)
        let mut replies: Vec<(Vec<u8>, u32, (u32, u16))> = Vec::new();
        for (&key, entry) in &mut self.icmp_nat {
            if entry.last_use.elapsed() > Duration::from_secs(30) {
                expired.push(key); continue;
            }
            let Some(sock) = &mut entry.sock else { continue };
            let mut buf = [std::mem::MaybeUninit::<u8>::uninit(); 1500];
            while let Ok(n) = sock.recv(&mut buf) {
                let raw: Vec<u8> = buf[..n].iter().map(|b| unsafe { b.assume_init() }).collect();
                // On Linux SOCK_DGRAM the kernel delivers only the ICMP payload.
                // On Windows/macOS SOCK_RAW the kernel prepends the outer IP header.
                #[cfg(not(target_os = "linux"))]
                let (outer_src_u32, icmp) = {
                    let ihl = ((raw.first().copied().unwrap_or(0x45) & 0x0f) as usize) * 4;
                    if raw.len() <= ihl { continue }
                    let src = u32::from_be_bytes([raw[12], raw[13], raw[14], raw[15]]);
                    (src, raw[ihl..].to_vec())
                };
                #[cfg(target_os = "linux")]
                let (outer_src_u32, icmp) = (key.0, raw);
                replies.push((icmp, outer_src_u32, key));
            }
        }
        for k in expired { self.icmp_nat.remove(&k); }
        for (mut icmp, outer_src_u32, key) in replies {
            if icmp.len() < 8 { continue; }
            let (dst_ip_u32, ident) = key;
            let icmp_type = icmp[0];

            // On Windows/macOS (SOCK_RAW) we receive Time Exceeded (type 11) for traceroute hops.
            // The payload of a Time Exceeded is: [unused 4B][original IP hdr][orig 8B].
            // We match via the ident embedded in the original probe's first 8 bytes,
            // rewrite the embedded src IP back to the guest IP, and forward to guest.
            #[cfg(not(target_os = "linux"))]
            if icmp_type == 11 {
                // Time Exceeded: find the right NAT entry via ident in embedded probe.
                // Embedded layout: icmp[8..] = original IP header + first 8 probe bytes.
                let emb = &icmp[8..];
                if emb.len() < 28 { continue; } // 20B IP hdr + 8B probe minimum
                let emb_ihl = ((emb[0] & 0x0f) as usize) * 4;
                if emb.len() < emb_ihl + 8 { continue; }
                let emb_dst = u32::from_be_bytes([emb[16], emb[17], emb[18], emb[19]]);
                let emb_ident = u16::from_be_bytes([emb[emb_ihl + 4], emb[emb_ihl + 5]]);
                let te_key = (emb_dst, emb_ident);
                if let Some(entry) = self.icmp_nat.get(&te_key) {
                    let client_mac = entry.client_mac;
                    let client_ip  = entry.client_ip;
                    let router_ip  = Ipv4Addr::from(outer_src_u32);
                    // Rewrite embedded src IP from host's real IP → guest IP.
                    let guest_bytes = client_ip.octets();
                    icmp[8 + 12] = guest_bytes[0]; icmp[8 + 13] = guest_bytes[1];
                    icmp[8 + 14] = guest_bytes[2]; icmp[8 + 15] = guest_bytes[3];
                    // Recompute ICMP checksum over the whole Time Exceeded message.
                    icmp[2] = 0; icmp[3] = 0;
                    let c = ip_checksum(&icmp);
                    w16(&mut icmp, 2, c);
                    dlog_dev!(LogModule::Net, "NAT ICMP TimeExceeded {} → {} ident={}", router_ip, client_ip, emb_ident);
                    let frame = ip_frame(&client_mac, &self.config.gateway_mac,
                                         router_ip, client_ip, IP_PROTO_ICMP, &icmp);
                    self.enqueue_rx(frame);
                }
                continue;
            }

            // Echo Reply (type 0): restore the original guest identifier
            // (kernel/NAT may have rewritten it) and recompute checksum.
            let _ = dst_ip_u32; // used via key above
            if let Some(entry) = self.icmp_nat.get(&key) {
                let remote_ip  = Ipv4Addr::from(outer_src_u32);
                let client_mac = entry.client_mac;
                let client_ip  = entry.client_ip;
                w16(&mut icmp, 4, ident);
                icmp[2] = 0; icmp[3] = 0;
                let c = ip_checksum(&icmp);
                w16(&mut icmp, 2, c);
                let seq = r16(&icmp, 6);
                dlog_dev!(LogModule::Net, "NAT ICMP reply {} → {} ident={} seq={}", remote_ip, client_ip, ident, seq);
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
            UDP_PORT_DNS          => self.forward_dns(src_mac, src_ip, sport, payload),
            UDP_PORT_PORTMAP if self.config.nfs.is_some()
                              => self.handle_portmap_udp(src_mac, src_ip, sport, payload),
            _ => {
                // NFS/mountd: rewrite destination to localhost high port before NAT.
                let real_dst = self.nfs_remap_dst(dst_ip, dport);
                self.nat_udp(src_mac, src_ip, real_dst.0, sport, real_dst.1, payload);
            }
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

    // ── NFS destination remapping ─────────────────────────────────────────────
    //
    // IRIX talks to 192.168.0.1 on VM-visible NFS/mountd ports.  Rewrite the
    // destination to 127.0.0.1 on the high host-side ports where unfsd listens.
    fn nfs_remap_dst(&self, dst_ip: Ipv4Addr, dport: u16) -> (Ipv4Addr, u16) {
        let Some(nfs) = &self.config.nfs else { return (dst_ip, dport); };
        if dst_ip != self.config.gateway_ip { return (dst_ip, dport); }
        match dport {
            NFS_VM_PORT    => (Ipv4Addr::LOCALHOST, nfs.nfs_host_port),
            MOUNTD_VM_PORT => (Ipv4Addr::LOCALHOST, nfs.mountd_host_port),
            _              => (dst_ip, dport),
        }
    }

    // Reverse: translate (127.0.0.1, host_port) back to (192.168.0.1, vm_port)
    // so replies to IRIX appear to come from the gateway on the standard NFS ports.
    fn nfs_unmap_src(&self, src_ip: Ipv4Addr, sport: u16) -> (Ipv4Addr, u16) {
        let Some(nfs) = &self.config.nfs else { return (src_ip, sport); };
        if src_ip != Ipv4Addr::LOCALHOST { return (src_ip, sport); }
        if sport == nfs.nfs_host_port    { return (self.config.gateway_ip, NFS_VM_PORT);    }
        if sport == nfs.mountd_host_port { return (self.config.gateway_ip, MOUNTD_VM_PORT); }
        (src_ip, sport)
    }

    // ── Portmap (port 111) — tiny inline RPC GETPORT responder ───────────────
    //
    // Parses an RPC GETPORT request (XDR) and replies with the VM-visible port
    // for the requested program.  We only answer GETPORT calls; everything else
    // gets a null reply with port=0.
    //
    // XDR layout of a GETPORT call (big-endian u32s):
    //   [0]  xid
    //   [1]  msg_type  = 0 (CALL)
    //   [2]  rpcvers   = 2
    //   [3]  prog      = 100000 (PORTMAP)
    //   [4]  vers      = 2
    //   [5]  proc      = 3 (GETPORT)
    //   [6]  cred_flavor, [7] cred_len=0, [8] verf_flavor, [9] verf_len=0
    //   [10] prog_to_query
    //   [11] vers_to_query
    //   [12] protocol (6=TCP / 17=UDP)
    //   [13] port (ignored in request)
    //
    // Reply layout:
    //   [0] xid  [1] 1 (REPLY)  [2] 0 (MSG_ACCEPTED)
    //   [3] verf_flavor=0  [4] verf_len=0  [5] accept_stat=0 (SUCCESS)
    //   [6] port
    fn handle_portmap_udp(&mut self, client_mac: &[u8; 6], client_ip: Ipv4Addr,
                          client_port: u16, payload: &[u8]) {
        let Some(nfs) = self.config.nfs.clone() else { return };
        let port = portmap_lookup(payload, &nfs);
        let xid = if payload.len() >= 4 { r32(payload, 0) } else { 0 };
        dlog_dev!(LogModule::Net, "NAT portmap UDP from {}:{} xid={:#010x} → port={}", client_ip, client_port, xid, port);
        let reply = portmap_reply(xid, port);
        let udp = udp_packet(self.config.gateway_ip, client_ip, UDP_PORT_PORTMAP, client_port, &reply);
        let frame = ip_frame(client_mac, &self.config.gateway_mac,
                             self.config.gateway_ip, client_ip, IP_PROTO_UDP, &udp);
        self.enqueue_rx(frame);
    }

    // ── Portmap TCP — send a complete RPC record-marked reply then RST ────────
    //
    // TCP RPC wraps each message in a 4-byte record mark: high bit set + 3-byte length.
    // We handle the full exchange in one shot: parse the first record from the SYN payload
    // (or from the first data segment after the SYN), reply, and RST the connection so
    // IRIX doesn't linger.  In practice mount sends a single GETPORT then closes anyway.
    fn handle_portmap_tcp_data(&mut self, client_mac: &[u8; 6], client_ip: Ipv4Addr,
                               client_port: u16, client_seq: u32, payload: &[u8]) {
        let Some(nfs) = self.config.nfs.clone() else { return };
        // Strip the 4-byte record mark if present.
        let rpc = if payload.len() >= 4 && (payload[0] & 0x80) != 0 {
            &payload[4..]
        } else {
            payload
        };
        let port = portmap_lookup(rpc, &nfs);
        let xid = if rpc.len() >= 4 { r32(rpc, 0) } else { 0 };
        dlog_dev!(LogModule::Net, "NAT portmap TCP from {}:{} xid={:#010x} → port={}", client_ip, client_port, xid, port);
        let rpc_reply = portmap_reply(xid, port);
        // Wrap in record mark (last fragment, bit 31 set).
        let rm_len = rpc_reply.len() as u32 | 0x8000_0000;
        let mut body = vec![0u8; 4 + rpc_reply.len()];
        w32(&mut body, 0, rm_len);
        body[4..].copy_from_slice(&rpc_reply);

        let gw   = self.config.gateway_ip;
        let gmac = self.config.gateway_mac;
        // SYN-ACK
        let server_isn = 0x5000_0000u32;
        let seg = tcp_segment(gw, client_ip, UDP_PORT_PORTMAP, client_port,
                              server_isn, client_seq, 0x12, &[]);
        let frame = ip_frame(client_mac, &gmac, gw, client_ip, IP_PROTO_TCP, &seg);
        self.enqueue_rx(frame);
        // Data + PSH+ACK
        let seg = tcp_segment(gw, client_ip, UDP_PORT_PORTMAP, client_port,
                              server_isn.wrapping_add(1), client_seq, 0x18, &body);
        let frame = ip_frame(client_mac, &gmac, gw, client_ip, IP_PROTO_TCP, &seg);
        self.enqueue_rx(frame);
        // FIN+ACK
        let seg = tcp_segment(gw, client_ip, UDP_PORT_PORTMAP, client_port,
                              server_isn.wrapping_add(1 + body.len() as u32),
                              client_seq, 0x11, &[]);
        let frame = ip_frame(client_mac, &gmac, gw, client_ip, IP_PROTO_TCP, &seg);
        self.enqueue_rx(frame);
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
                // Reverse-map localhost high port back to gateway VM-visible port.
                let (reply_ip, reply_port) = self.nfs_unmap_src(Ipv4Addr::from(dst_ip_u32), dst_port);
                dlog_dev!(LogModule::Net, "NAT UDP reply → IRIX: {}:{} → {}:{} len={}", reply_ip, reply_port, entry.client_ip, client_port, data.len());
                let udp = udp_packet(reply_ip, entry.client_ip, reply_port, client_port, &data);
                let client_mac = entry.client_mac;
                let client_ip = entry.client_ip;
                let frame = ip_frame(&client_mac, &self.config.gateway_mac,
                                     reply_ip, client_ip, IP_PROTO_UDP, &udp);
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

        // Intercept portmap TCP (port 111) — handle inline, never hits the NAT table.
        if dport == UDP_PORT_PORTMAP && self.config.nfs.is_some() {
            if syn && !ack {
                // SYN only — IRIX will send data in the next segment; we respond
                // with SYN-ACK and wait for the data segment to arrive.
                let gw   = self.config.gateway_ip;
                let gmac = self.config.gateway_mac;
                let server_isn = 0x5000_0000u32;
                let seg = tcp_segment(gw, src_ip, UDP_PORT_PORTMAP, sport,
                                      server_isn, seq.wrapping_add(1), 0x12, &[]);
                let frame = ip_frame(client_mac, &gmac, gw, src_ip, IP_PROTO_TCP, &seg);
                self.enqueue_rx(frame);
            } else if !payload.is_empty() {
                self.handle_portmap_tcp_data(client_mac, src_ip, sport, seq.wrapping_add(payload.len() as u32), payload);
            }
            return;
        }

        // NFS/mountd: rewrite destination to localhost high port.
        let (dst_ip, dport) = self.nfs_remap_dst(dst_ip, dport);

        let key = (u32::from(dst_ip), dport, sport);

        if syn && !ack {
            dlog_dev!(LogModule::Net, "NAT TCP connect {}:{} → {}:{}", src_ip, sport, dst_ip, dport);
            // visible_ip is the address IRIX sees as the remote end (gateway for NFS, dst otherwise).
            let (visible_ip, _) = self.nfs_unmap_src(dst_ip, dport);
            let dest = SocketAddr::new(IpAddr::V4(dst_ip), dport);
            match TcpStream::connect_timeout(&dest, Duration::from_secs(5)) {
                Ok(stream) => {
                    let _ = stream.set_nonblocking(true);
                    let server_seq = 0x4000_0000u32;
                    dlog_dev!(LogModule::Net, "NAT TCP connected {}:{} → {}:{}", src_ip, sport, dst_ip, dport);
                    self.tcp_nat.insert(key, NatTcpEntry {
                        stream, client_mac: *client_mac, client_ip: src_ip,
                        client_port: sport, server_ip: visible_ip,
                        server_seq: server_seq.wrapping_add(1),
                        server_seq_acked: server_seq.wrapping_add(1),
                        client_win: r16(tcp, 14) as u32,
                        client_seq: seq.wrapping_add(1),
                        last_use: Instant::now(),
                        fin_wait: false,
                        server_fin: false,
                        retransmit: VecDeque::new(),
                    });
                    let seg = tcp_segment(visible_ip, src_ip, dport, sport,
                                         server_seq, seq.wrapping_add(1), 0x12, &[]);
                    let frame = ip_frame(client_mac, &self.config.gateway_mac,
                                        visible_ip, src_ip, IP_PROTO_TCP, &seg);
                    self.enqueue_rx(frame);
                }
                Err(e) => {
                    dlog_dev!(LogModule::Net, "NAT TCP connect {}:{} failed: {}", dst_ip, dport, e);
                    let seg = tcp_segment(visible_ip, src_ip, dport, sport,
                                         0, seq.wrapping_add(1), 0x14, &[]);
                    let frame = ip_frame(client_mac, &self.config.gateway_mac,
                                        visible_ip, src_ip, IP_PROTO_TCP, &seg);
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
