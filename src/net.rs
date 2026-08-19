// NAT gateway / network engine for the SEEQ 8003 emulator.
//
// Runs in its own thread ("seeq-nat"). Receives outbound Ethernet frames from
// the enet thread via an rtrb::Consumer<Vec<u8>>, processes them through a
// software NAT stack, and enqueues inbound frames back via rtrb::Producer<Vec<u8>>.

use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream, UdpSocket};
use socket2::{Domain, Protocol, Socket, Type};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use crate::config::{ForwardBind, ForwardProto, NatSubnet, NetMode, NfsConfig, PortForwardConfig};
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
const UDP_PORT_TIME:         u16 = 37;
const UDP_PORT_NTP:          u16 = 123;
const UDP_PORT_TFTP:         u16 = 69;
const XDMCP_GUEST_PORT:      u16 = 177;  // XDMCP control channel inside the guest
const X11_BASE_PORT:         u16 = 6000; // X11 display N's TCP port = 6000 + N
const TCP_PORT_TIME:         u16 = 37;
const BOOTP_OP_REQUEST: u8      = 1;

// Seconds between 1900-01-01 (NTP/RFC868 epoch) and 1970-01-01 (Unix epoch).
const NTP_EPOCH_OFFSET: u64 = 2_208_988_800;

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
}

impl Default for GatewayConfig {
    fn default() -> Self {
        let subnet = NatSubnet::default();
        Self {
            gateway_mac:  [0x02, 0x00, 0xDE, 0xAD, 0xBE, 0xEF],
            gateway_ip:   subnet.gateway_ip,
            client_ip:    subnet.client_ip,
            netmask:      subnet.netmask,
            dns_upstream: "8.8.8.8:53".parse().unwrap(),
            nfs:          None,
            port_forwards: vec![],
            mode:         NetMode::Nat,
            pcap_interface: None,
            nfs_pcap_ip:  None,
            tftp_dir:     None,
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
    /// the calling thread (the `seeq-nat` thread).
    fn run(&mut self);
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

fn icmp_socket() -> std::io::Result<Socket> {
    let new = |t| Socket::new(Domain::IPV4, t, Some(Protocol::ICMPV4));
    #[cfg(target_os = "linux")]
    { new(Type::DGRAM) }
    #[cfg(target_os = "macos")]
    { new(Type::DGRAM).or_else(|_| new(Type::RAW)) }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    { new(Type::RAW) }
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

/// Build one IP fragment frame. `frag_offset` is in bytes (must be multiple of 8).
fn ip_fragment_frame(dst_mac: &[u8; 6], gw_mac: &[u8; 6],
                     src_ip: Ipv4Addr, dst_ip: Ipv4Addr, proto: u8,
                     id: u16, frag_offset: usize, more_frags: bool,
                     payload_chunk: &[u8]) -> Vec<u8> {
    let mut h = [0u8; 20];
    h[0] = 0x45;
    w16(&mut h, 2, 20 + payload_chunk.len() as u16);
    w16(&mut h, 4, id);
    let offset_field = ((frag_offset / 8) as u16) | if more_frags { 0x2000 } else { 0 };
    w16(&mut h, 6, offset_field);
    h[8]  = 64;
    h[9]  = proto;
    h[12..16].copy_from_slice(&src_ip.octets());
    h[16..20].copy_from_slice(&dst_ip.octets());
    let c = ip_checksum(&h); w16(&mut h, 10, c);
    let mut ip = h.to_vec();
    ip.extend_from_slice(payload_chunk);
    eth_frame(dst_mac, gw_mac, ETHERTYPE_IP, &ip)
}

/// Fragment a UDP datagram into ≤1500-byte Ethernet frames and return them all.
/// `udp_payload` is the already-built UDP header+data (output of `udp_packet`).
fn ip_frames_udp(dst_mac: &[u8; 6], gw_mac: &[u8; 6],
                 src_ip: Ipv4Addr, dst_ip: Ipv4Addr,
                 id: u16, udp_payload: &[u8]) -> Vec<Vec<u8>> {
    // Maximum IP payload per fragment: 1480 bytes (1500 MTU - 20 IP header), must be
    // a multiple of 8 so subsequent fragment offsets are representable.
    const MAX_FRAG: usize = 1480;
    if udp_payload.len() <= MAX_FRAG {
        // Fits in one frame — no fragmentation needed.
        let mut h = [0u8; 20];
        h[0] = 0x45;
        w16(&mut h, 2, 20 + udp_payload.len() as u16);
        w16(&mut h, 4, id);
        h[8]  = 64;
        h[9]  = IP_PROTO_UDP;
        h[12..16].copy_from_slice(&src_ip.octets());
        h[16..20].copy_from_slice(&dst_ip.octets());
        let c = ip_checksum(&h); w16(&mut h, 10, c);
        let mut ip = h.to_vec();
        ip.extend_from_slice(udp_payload);
        return vec![eth_frame(dst_mac, gw_mac, ETHERTYPE_IP, &ip)];
    }
    let mut frames = Vec::new();
    let mut offset = 0;
    while offset < udp_payload.len() {
        let remaining = udp_payload.len() - offset;
        // Chunk size must be a multiple of 8 unless it's the last fragment.
        let chunk_size = if remaining <= MAX_FRAG {
            remaining
        } else {
            MAX_FRAG // already a multiple of 8
        };
        let more = offset + chunk_size < udp_payload.len();
        frames.push(ip_fragment_frame(dst_mac, gw_mac, src_ip, dst_ip, IP_PROTO_UDP,
                                      id, offset, more, &udp_payload[offset..offset + chunk_size]));
        offset += chunk_size;
    }
    frames
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

/// A virtual NFS host on the bridged LAN, used only in PCAP mode. The in-process
/// NFS server normally lives inside the NAT engine and answers at the gateway IP;
/// in PCAP mode the NAT engine isn't running and the guest sits directly on the
/// real LAN, so we present the server as its own L2 host at a configured virtual
/// IP. [`PcapEngine`] feeds each guest TX frame to [`maybe_handle`]: an ARP-for or
/// UDP-to our IP (portmap / NFS / mountd) is answered and the reply frame(s) are
/// injected straight back to the guest — still zero host sockets. Anything else is
/// forwarded to the wire unchanged.
///
/// Inbound IP fragments (a large NFS WRITE whose wsize exceeds the link MTU) are
/// reassembled in [`handle_udp`] before dispatch, keyed by (src, id, proto) and
/// aged out after 5s — mirroring the NAT engine's `handle_ip`. Outbound replies are
/// auto-fragmented (via [`ip_frames_udp`]); MOUNT/READ and small ops still fit one
/// datagram.
///
/// [`maybe_handle`]: NfsVirtualHost::maybe_handle
/// [`handle_udp`]: NfsVirtualHost::handle_udp
#[cfg(feature = "pcap")]
pub(crate) struct NfsVirtualHost {
    ip: Ipv4Addr,
    mac: [u8; 6],
    server: crate::nfsudp::NfsServer,
    nfs_cfg: NfsConfig,
    ip_id: u16,
    /// Inbound IP-fragment reassembly buffer, keyed by (src IP, IP id, proto).
    frag_reasm: HashMap<(u32, u16, u8), FragReasm>,
}

#[cfg(feature = "pcap")]
impl NfsVirtualHost {
    pub(crate) fn new(ip: Ipv4Addr, nfs_cfg: NfsConfig) -> Self {
        // Locally-administered MAC, distinct from the NAT gateway's, so the guest
        // keeps the virtual NFS host as its own ARP entry. (0xBF53 ≈ "BF-NFS".)
        let mac = [0x02, 0x00, 0xDE, 0xAD, 0xBF, 0x53];
        let server = crate::nfsudp::NfsServer::new(nfs_cfg.shared_dir.clone(), nfs_cfg.version);
        Self { ip, mac, server, nfs_cfg, ip_id: 1, frag_reasm: HashMap::new() }
    }

    /// Handle a guest-originated Ethernet `frame` if it's addressed to the virtual
    /// NFS host; returns reply frame(s) to inject back to the guest, or `None` if
    /// the frame isn't for us (the caller bridges it to the wire).
    pub(crate) fn maybe_handle(&mut self, frame: &[u8]) -> Option<Vec<Vec<u8>>> {
        if frame.len() < 14 {
            return None;
        }
        match r16(frame, 12) {
            ETHERTYPE_ARP => self.handle_arp(frame),
            ETHERTYPE_IP => self.handle_udp(frame),
            _ => None,
        }
    }

    fn handle_arp(&self, frame: &[u8]) -> Option<Vec<Vec<u8>>> {
        if frame.len() < 14 + 28 {
            return None;
        }
        let a = &frame[14..];
        if r16(a, 0) != ARP_HW_ETHER
            || r16(a, 2) != ARP_PROTO_IP
            || a[4] != 6
            || a[5] != 4
            || r16(a, 6) != ARP_OP_REQUEST
        {
            return None;
        }
        let sender_mac: [u8; 6] = a[8..14].try_into().unwrap();
        let sender_ip = Ipv4Addr::new(a[14], a[15], a[16], a[17]);
        let target_ip = Ipv4Addr::new(a[24], a[25], a[26], a[27]);
        if target_ip != self.ip {
            return None;
        }
        let mut arp = [0u8; 28];
        w16(&mut arp, 0, ARP_HW_ETHER);
        w16(&mut arp, 2, ARP_PROTO_IP);
        arp[4] = 6;
        arp[5] = 4;
        w16(&mut arp, 6, ARP_OP_REPLY);
        arp[8..14].copy_from_slice(&self.mac);
        arp[14..18].copy_from_slice(&self.ip.octets());
        arp[18..24].copy_from_slice(&sender_mac);
        arp[24..28].copy_from_slice(&sender_ip.octets());
        Some(vec![eth_frame(&sender_mac, &self.mac, ETHERTYPE_ARP, &arp)])
    }

    fn handle_udp(&mut self, frame: &[u8]) -> Option<Vec<Vec<u8>>> {
        if frame.len() < 34 {
            return None;
        }
        let src_mac: [u8; 6] = frame[6..12].try_into().unwrap();
        let ip = &frame[14..];
        if ip[9] != IP_PROTO_UDP {
            return None;
        }
        let dst_ip = Ipv4Addr::new(ip[16], ip[17], ip[18], ip[19]);
        if dst_ip != self.ip {
            return None;
        }
        let src_ip = Ipv4Addr::new(ip[12], ip[13], ip[14], ip[15]);
        let ihl = ((ip[0] & 0x0f) as usize) * 4;
        let ip_total = r16(ip, 2) as usize;
        if ip_total < ihl || frame.len() < 14 + ihl {
            return None;
        }
        let ip_end = ip_total.min(frame.len() - 14);
        let ip_payload = &ip[ihl..ip_end];

        // Inbound IP-fragment reassembly: a large NFS WRITE arrives fragmented when
        // its wsize exceeds the link MTU. Buffer fragments by (src, id, proto) and
        // only dispatch once the whole UDP datagram is contiguous; mirrors the NAT
        // engine's `handle_ip`. Without this, large guest→host writes silently fail.
        // A buffered-but-incomplete fragment returns `Some(vec![])` so the caller
        // treats it as consumed (not bridged onto the wire), since it's addressed to
        // our virtual host.
        let flags_frag = r16(ip, 6);
        let more = flags_frag & 0x2000 != 0;
        let frag_off = ((flags_frag & 0x1fff) as usize) * 8;
        let reassembled: Option<Vec<u8>> = if more || frag_off != 0 {
            let id = r16(ip, 4);
            let key = (u32::from(src_ip), id, IP_PROTO_UDP);
            self.frag_reasm.retain(|_, v| v.last.elapsed() < Duration::from_secs(5));
            match self.frag_reasm.entry(key).or_insert_with(FragReasm::new).add(frag_off, ip_payload, more) {
                Some(full) => { self.frag_reasm.remove(&key); Some(full) }
                None => return Some(Vec::new()), // consumed; still waiting on fragments
            }
        } else {
            None
        };
        // The complete UDP datagram: the reassembled one, or this lone unfragmented one.
        let udp: &[u8] = reassembled.as_deref().unwrap_or(ip_payload);
        if udp.len() < 8 {
            return None;
        }
        let sport = r16(udp, 0);
        let dport = r16(udp, 2);
        let payload = &udp[8..];
        let reply = match dport {
            UDP_PORT_PORTMAP => {
                let xid = if payload.len() >= 4 { r32(payload, 0) } else { 0 };
                portmap_reply(xid, portmap_lookup(payload, &self.nfs_cfg))
            }
            NFS_VM_PORT | MOUNTD_VM_PORT => self.server.handle(payload)?,
            _ => return None,
        };
        // Reply source = our virtual host; reply port = the port the guest hit.
        let dgram = udp_packet(self.ip, src_ip, dport, sport, &reply);
        let id = self.ip_id;
        self.ip_id = self.ip_id.wrapping_add(1);
        Some(ip_frames_udp(&src_mac, &self.mac, self.ip, src_ip, id, &dgram))
    }
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
    pub guest_frames: AtomicU64,
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
        Arc::new(Self {
            debug_tcp:  AtomicBool::new(false),
            debug_udp:  AtomicBool::new(false),
            debug_icmp: AtomicBool::new(false),
            snapshot:   Mutex::new(NatSnapshot::default()),
            reset_nat:  AtomicBool::new(false),
            guest_frames: AtomicU64::new(0),
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
    pub fn guest_frames(&self) -> u64 { self.guest_frames.load(Ordering::Relaxed) }

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

// ── Port-forward listeners ────────────────────────────────────────────────────

/// Active TCP listener for one inbound port-forward rule.
struct TcpFwdListener {
    listener:   TcpListener,
    guest_port: u16,
}

/// Active UDP socket for one inbound port-forward rule.
struct UdpFwdListener {
    sock:        UdpSocket,
    guest_port:  u16,
    host_port:   u16,
    /// Most recent sender — replies from IRIX are forwarded back here.
    last_sender: Option<SocketAddr>,
}

/// A TCP forward connection that has been accepted from the host but whose
/// SYN to the guest is still awaiting a SYN-ACK.
/// key in tcp_fwd_pending: (guest_port, ephemeral_sport)  — same layout as tcp_nat key
/// where "dst" = guest_ip:guest_port and "src" = gateway:ephemeral.
struct TcpFwdPending {
    stream:     TcpStream,  // the host-side accepted connection
    client_isn: u32,        // ISN we put in the synthetic SYN to the guest
}

/// FTP application-layer gateway (passive mode). If `payload` — server→client
/// bytes on an FTP control connection — contains a `227 Entering Passive Mode
/// (h1,h2,h3,h4,p1,p2)` reply, rewrite the address tuple to advertise
/// `host`:`host_port` and return `(rewritten_payload, guest_data_port)`. Returns
/// `None` when there's no PASV reply to rewrite.
///
/// The length-changing rewrite is safe here because the NAT relays the byte
/// stream between two independent TCP connections (a host OS socket and the
/// userspace guest-side TCP), so the host stack re-sequences on its own — no
/// seq/ack surgery. The non-FTP / no-match path is left untouched.
fn ftp_pasv_rewrite(payload: &[u8], host: Ipv4Addr, host_port: u16) -> Option<(Vec<u8>, u16)> {
    // The "227" reply code at the start of a line (start of buffer or after \n).
    let pos = (0..payload.len().saturating_sub(2))
        .find(|&i| &payload[i..i + 3] == b"227" && (i == 0 || payload[i - 1] == b'\n'))?;
    let open  = pos + payload[pos..].iter().position(|&b| b == b'(')?;
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
    let replacement = format!("({},{},{},{},{},{})",
        o[0], o[1], o[2], o[3], host_port >> 8, host_port & 0xff);
    let mut out = Vec::with_capacity(payload.len() + replacement.len());
    out.extend_from_slice(&payload[..open]);
    out.extend_from_slice(replacement.as_bytes());
    out.extend_from_slice(&payload[close + 1..]);
    Some((out, data_port))
}

#[cfg(test)]
mod ftp_alg_tests {
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
        // Malformed tuple (not six bytes) is rejected.
        assert!(ftp_pasv_rewrite(b"227 (1,2,3,4,5)\r\n", Ipv4Addr::LOCALHOST, 1).is_none());
        assert!(ftp_pasv_rewrite(b"227 (1,2,3,4,5,999)\r\n", Ipv4Addr::LOCALHOST, 1).is_none());
    }
}

#[cfg(test)]
mod tftp_nat_tests {
    use super::*;

    fn tmp_root(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let d = std::env::temp_dir().join(format!("iris-nat-tftp-{}-{}", tag, nanos));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A NAT engine serving `root` over TFTP, with the rings a real one uses.
    fn engine(root: &std::path::Path) -> (NatEngine, rtrb::Consumer<Vec<u8>>) {
        let (_tx_prod, tx_cons) = rtrb::RingBuffer::new(8);
        let (rx_prod, rx_cons) = rtrb::RingBuffer::new(8);
        let config = GatewayConfig { tftp_dir: Some(root.to_path_buf()), ..GatewayConfig::default() };
        let e = NatEngine::new(
            config, tx_cons, rx_prod,
            Arc::new((Mutex::new(()), Condvar::new())),
            Arc::new((Mutex::new(()), Condvar::new())),
            Arc::new(AtomicBool::new(true)),
            NatControl::new(),
        );
        (e, rx_cons)
    }

    /// Guest → gateway UDP 69 frame carrying `payload`.
    fn guest_frame(e: &NatEngine, guest_mac: &[u8; 6], sport: u16, payload: &[u8]) -> Vec<u8> {
        let guest_ip = e.config.client_ip;
        let gw_ip = e.config.gateway_ip;
        let udp = udp_packet(guest_ip, gw_ip, sport, UDP_PORT_TFTP, payload);
        ip_frame(&e.config.gateway_mac, guest_mac, guest_ip, gw_ip, IP_PROTO_UDP, &udp)
    }

    /// The reply's (src_port, dst_port, payload).
    fn parse_reply(frame: &[u8]) -> (u16, u16, Vec<u8>) {
        let ihl = ((frame[14] & 0x0F) as usize) * 4;
        let udp = &frame[14 + ihl..];
        (r16(udp, 0), r16(udp, 2), udp[8..].to_vec())
    }

    #[test]
    fn rrq_through_the_nat_returns_data_from_the_gateway() {
        let root = tmp_root("rrq");
        let data: Vec<u8> = (0..700u32).map(|i| (i * 7) as u8).collect();
        std::fs::write(root.join("cputest"), &data).unwrap();
        let (mut e, _rx) = engine(&root);
        let guest_mac = [0x08, 0x00, 0x69, 0x11, 0x22, 0x33];

        let mut rrq = vec![0, 1];
        rrq.extend_from_slice(b"cputest\0octet\0");
        e.process(&guest_frame(&e, &guest_mac, 4242, &rrq));

        assert_eq!(e.deferred_rx.len(), 1, "one DATA datagram back");
        let (sport, dport, payload) = parse_reply(&e.deferred_rx[0]);
        assert_eq!(sport, UDP_PORT_TFTP, "reply comes from the gateway's TFTP port");
        assert_eq!(dport, 4242, "reply goes back to the client's TID");
        assert_eq!(u16::from_be_bytes([payload[0], payload[1]]), 3, "opcode DATA");
        assert_eq!(u16::from_be_bytes([payload[2], payload[3]]), 1, "block 1");
        assert_eq!(&payload[4..], &data[..512]);
        // Frame is addressed to the guest, from the gateway.
        assert_eq!(&e.deferred_rx[0][0..6], &guest_mac);
        assert_eq!(&e.deferred_rx[0][6..12], &e.config.gateway_mac);

        // ACK 1 draws the short final block, which ends the transfer.
        e.deferred_rx.clear();
        e.process(&guest_frame(&e, &guest_mac, 4242, &[0, 4, 0, 1]));
        let (_, _, payload) = parse_reply(&e.deferred_rx[0]);
        assert_eq!(u16::from_be_bytes([payload[2], payload[3]]), 2, "block 2");
        assert_eq!(&payload[4..], &data[512..], "188 remaining bytes");

        e.deferred_rx.clear();
        e.process(&guest_frame(&e, &guest_mac, 4242, &[0, 4, 0, 2]));
        assert!(e.deferred_rx.is_empty(), "final ACK ends the transfer silently");
        // The per-client MAC is dropped by the poll tick that follows, which is
        // also the retransmit path — and it must have nothing left to resend.
        e.poll_tftp();
        assert!(e.tftp_macs.is_empty(), "client bookkeeping is dropped with the transfer");
        assert!(e.deferred_rx.is_empty(), "a completed transfer never retransmits");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn tftp_is_inert_when_no_directory_is_configured() {
        let (_tx_prod, tx_cons) = rtrb::RingBuffer::new(8);
        let (rx_prod, _rx_cons) = rtrb::RingBuffer::new(8);
        let mut e = NatEngine::new(
            GatewayConfig::default(), tx_cons, rx_prod,
            Arc::new((Mutex::new(()), Condvar::new())),
            Arc::new((Mutex::new(()), Condvar::new())),
            Arc::new(AtomicBool::new(true)),
            NatControl::new(),
        );
        let guest_mac = [0x08, 0x00, 0x69, 0x11, 0x22, 0x33];
        let mut rrq = vec![0, 1];
        rrq.extend_from_slice(b"anything\0octet\0");
        e.process(&guest_frame(&e, &guest_mac, 4242, &rrq));
        assert!(e.tftp.is_none());
        // Falls through to the generic NAT path, so no reply is generated here.
        assert!(e.deferred_rx.is_empty());
    }
}

#[cfg(all(test, feature = "pcap"))]
mod nfs_pcap_tests {
    use super::*;

    /// A 56-byte portmap GETPORT call asking for the port of `req_prog` over UDP.
    fn getport_call(xid: u32, req_prog: u32) -> Vec<u8> {
        let mut p = vec![0u8; 56];
        w32(&mut p,  0, xid);
        w32(&mut p,  4, 0);                    // msg_type = CALL
        w32(&mut p,  8, 2);                    // rpcvers
        w32(&mut p, 12, RPC_PROG_PORTMAP);
        w32(&mut p, 16, 2);                    // portmap vers
        w32(&mut p, 20, RPC_PORTMAP_GETPORT);
        // cred (flavor+len) and verf (flavor+len) all zero (AUTH_NULL).
        w32(&mut p, 40, req_prog);             // mapping.prog
        w32(&mut p, 44, 3);                    // mapping.vers
        w32(&mut p, 48, 17);                   // mapping.prot = UDP
        w32(&mut p, 52, 0);                    // mapping.port
        p
    }

    /// A fragmented inbound portmap request (large NFS WRITEs arrive the same way)
    /// must be reassembled, not dropped: intermediate fragments are consumed
    /// (`Some([])`, so the caller doesn't bridge them onto the wire) and the final
    /// fragment yields the reply built from the whole datagram.
    #[test]
    fn reassembles_fragmented_request() {
        let nfs_ip = Ipv4Addr::new(192, 168, 1, 213);
        let guest_ip = Ipv4Addr::new(192, 168, 1, 50);
        let guest_mac = [0x08, 0x00, 0x69, 0x01, 0x02, 0x03];
        let host_mac  = [0x02, 0x00, 0xde, 0xad, 0xbf, 0x53];

        let mut host = NfsVirtualHost::new(
            nfs_ip,
            NfsConfig { shared_dir: "/nonexistent".into(), version: crate::nfsudp::NfsVersion::Auto },
        );

        // Full UDP datagram (header + portmap GETPORT for NFS), then split its IP
        // payload across two fragments at an 8-byte boundary.
        let dgram = udp_packet(guest_ip, nfs_ip, 0x9abc, UDP_PORT_PORTMAP, &getport_call(0x1234, RPC_PROG_NFS));
        let (a, b) = dgram.split_at(8);
        let frag1 = ip_fragment_frame(&host_mac, &guest_mac, guest_ip, nfs_ip, IP_PROTO_UDP, 0x42, 0, true,  a);
        let frag2 = ip_fragment_frame(&host_mac, &guest_mac, guest_ip, nfs_ip, IP_PROTO_UDP, 0x42, 8, false, b);

        // First fragment: consumed, no reply yet (NOT None — None would bridge it).
        assert_eq!(host.maybe_handle(&frag1), Some(Vec::new()));

        // Second fragment completes the datagram → one reply frame.
        let frames = host.maybe_handle(&frag2).expect("reply after reassembly");
        assert_eq!(frames.len(), 1);
        // eth(14) + ip(20) + udp(8) = 42-byte headers, then the 28-byte portmap reply
        // whose final word is the resolved port.
        let reply = &frames[0];
        let port = u32::from_be_bytes(reply[reply.len() - 4..].try_into().unwrap());
        assert_eq!(port, NFS_VM_PORT as u32, "portmap resolved NFS to its VM port");
    }

    /// An unfragmented request still works unchanged (no reassembly path taken).
    #[test]
    fn handles_unfragmented_request() {
        let nfs_ip = Ipv4Addr::new(10, 0, 0, 9);
        let guest_ip = Ipv4Addr::new(10, 0, 0, 2);
        let guest_mac = [0x08, 0x00, 0x69, 0x0a, 0x0b, 0x0c];
        let host_mac  = [0x02, 0x00, 0xde, 0xad, 0xbf, 0x53];

        let mut host = NfsVirtualHost::new(
            nfs_ip,
            NfsConfig { shared_dir: "/nonexistent".into(), version: crate::nfsudp::NfsVersion::Auto },
        );
        let dgram = udp_packet(guest_ip, nfs_ip, 0x9abc, UDP_PORT_PORTMAP, &getport_call(7, RPC_PROG_MOUNTD));
        let frame = ip_frame(&host_mac, &guest_mac, guest_ip, nfs_ip, IP_PROTO_UDP, &dgram);
        let frames = host.maybe_handle(&frame).expect("reply");
        let reply = &frames[0];
        let port = u32::from_be_bytes(reply[reply.len() - 4..].try_into().unwrap());
        assert_eq!(port, MOUNTD_VM_PORT as u32);
    }
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
    icmp_unavailable: bool,  // true after first failed ICMP socket creation
    // Replies generated while draining TX frames are deferred to the next loop iteration
    // so they don't race with the TX completion interrupt in IRIX's interrupt handler.
    deferred_rx: Vec<Vec<u8>>,
    // Port-forward listeners (set up once at construction, never replaced).
    tcp_fwd_listeners: Vec<TcpFwdListener>,
    udp_fwd_listeners: Vec<UdpFwdListener>,
    // TCP forward connections whose SYN to the guest has been sent but SYN-ACK not yet received.
    // key: (u32::from(guest_ip), guest_port, ephemeral_sport)
    tcp_fwd_pending: HashMap<(u32, u16, u16), TcpFwdPending>,
    // Monotonically increasing counter for generating ephemeral ports for inbound forwards.
    fwd_ephemeral_next: u16,
    // Same, for forwards to BSD r-services, which reject any source port outside 512..1023.
    fwd_reserved_next: u16,
    // Number of configured (static) forwards at the front of `tcp_fwd_listeners`;
    // anything past this index is a transient FTP-ALG data forward (bounded, FIFO).
    fwd_static_count: usize,
    // Guest MAC learned from any outbound frame (ARP SHA or Ethernet src).
    guest_mac: Option<[u8; 6]>,
    // Monotonically increasing IP identification counter for fragmented datagrams.
    ip_id: u16,
    // In-core NFS/UDP server (replaces external unfsd). Some when an NFS export
    // is configured; the NAT dispatches guest MOUNT/NFS RPC straight to it.
    nfs: Option<crate::nfsudp::NfsServer>,
    // Inbound IP-fragment reassembly buffers, keyed by (src_ip, ip_id, proto).
    // NFS writes arrive fragmented when wsize > MTU.
    frag_reasm: HashMap<(u32, u16, u8), FragReasm>,
    // XDMCP reverse-proxy ALG: active X11 session targets, keyed by the X11 TCP
    // port (6000 + display). nfs_remap_dst relays guest→gateway:(that port) to the
    // real X server recorded here instead of the generic loopback. Populated by
    // the XDMCP UDP-forward hook in poll_udp_fwd_listeners.
    xdmcp_sessions: HashMap<u16, Ipv4Addr>,
    // Read-only TFTP server for PROM network boot. Some when tftp_dir is set.
    tftp: Option<crate::tftp::TftpServer>,
    // MAC to reply to per TFTP client, so a retransmit can be addressed without
    // relying on the last-learned guest MAC.
    tftp_macs: HashMap<crate::tftp::ClientId, [u8; 6]>,
}

/// Reassembly state for one fragmented inbound IP datagram.
struct FragReasm {
    frags: std::collections::BTreeMap<usize, Vec<u8>>, // offset -> bytes (deduped)
    total: Option<usize>,                              // known once the last fragment arrives
    last: Instant,
}
impl FragReasm {
    fn new() -> Self {
        Self { frags: std::collections::BTreeMap::new(), total: None, last: Instant::now() }
    }
    /// Add a fragment; return the reassembled payload once it's contiguous +
    /// complete.
    fn add(&mut self, offset: usize, data: &[u8], more: bool) -> Option<Vec<u8>> {
        self.frags.insert(offset, data.to_vec());
        self.last = Instant::now();
        if !more {
            self.total = Some(offset + data.len());
        }
        let total = self.total?;
        let mut out = Vec::with_capacity(total);
        for (&off, d) in &self.frags {
            if off != out.len() {
                return None; // gap — still waiting on a fragment
            }
            out.extend_from_slice(d);
        }
        (out.len() == total).then_some(out)
    }
}

/// Bind host listeners for a set of port-forward rules. Used at construction and
/// when forwards are reconfigured live (see `rebind_forwards`).
fn bind_forwards(rules: &[PortForwardConfig]) -> (Vec<TcpFwdListener>, Vec<UdpFwdListener>) {
    let mut tcp = Vec::new();
    let mut udp = Vec::new();
    for rule in rules {
        let bind_addr = match rule.bind {
            ForwardBind::Localhost => Ipv4Addr::LOCALHOST,
            ForwardBind::Any      => Ipv4Addr::UNSPECIFIED,
        };
        let addr = SocketAddr::new(IpAddr::V4(bind_addr), rule.host_port);
        match rule.proto {
            ForwardProto::Tcp => match TcpListener::bind(addr) {
                Ok(listener) => {
                    let _ = listener.set_nonblocking(true);
                    eprintln!("iris: TCP port forward {}:{} → guest:{}",
                              bind_addr, rule.host_port, rule.guest_port);
                    tcp.push(TcpFwdListener { listener, guest_port: rule.guest_port });
                }
                Err(e) => eprintln!("iris: TCP port forward {}:{} failed to bind: {}",
                                    bind_addr, rule.host_port, e),
            },
            ForwardProto::Udp => match UdpSocket::bind(addr) {
                Ok(sock) => {
                    let _ = sock.set_nonblocking(true);
                    eprintln!("iris: UDP port forward {}:{} → guest:{}",
                              bind_addr, rule.host_port, rule.guest_port);
                    udp.push(UdpFwdListener {
                        sock, guest_port: rule.guest_port, host_port: rule.host_port, last_sender: None,
                    });
                }
                Err(e) => eprintln!("iris: UDP port forward {}:{} failed to bind: {}",
                                    bind_addr, rule.host_port, e),
            },
        }
    }
    (tcp, udp)
}

impl NatEngine {
    pub fn new(config: GatewayConfig,
               tx_cons: rtrb::Consumer<Vec<u8>>,
               rx_prod: rtrb::Producer<Vec<u8>>,
               rx_wake: Arc<(Mutex<()>, Condvar)>,
               tx_wake: Arc<(Mutex<()>, Condvar)>,
               running: Arc<AtomicBool>,
               ctl:     Arc<NatControl>) -> Self {
        let (tcp_fwd_listeners, udp_fwd_listeners) = bind_forwards(&config.port_forwards);
        let fwd_static_count = tcp_fwd_listeners.len();
        // Spin up the in-core NFS server if an export is configured.
        let nfs = config.nfs.as_ref().map(|c| {
            eprintln!("iris: in-core NFS server exporting {}", c.shared_dir);
            crate::nfsudp::NfsServer::new(c.shared_dir.clone(), c.version)
        });
        let tftp = config.tftp_dir.as_ref().map(|dir| {
            eprintln!("iris: TFTP server (read-only) serving {}", dir.display());
            crate::tftp::TftpServer::new(dir.clone())
        });
        Self { config, tx_cons, rx_prod, rx_wake, tx_wake, running, ctl,
               udp_nat: HashMap::new(), tcp_nat: HashMap::new(), tcp_tw: HashMap::new(),
               icmp_nat: HashMap::new(), icmp_unavailable: false, deferred_rx: Vec::new(),
               tcp_fwd_listeners, udp_fwd_listeners, fwd_static_count,
               tcp_fwd_pending: HashMap::new(), fwd_ephemeral_next: 49152,
               fwd_reserved_next: 512,
               guest_mac: None, ip_id: 1, nfs, frag_reasm: HashMap::new(),
               xdmcp_sessions: HashMap::new(),
               tftp, tftp_macs: HashMap::new() }
    }

    /// Rebind the static port-forward listeners from a new rule set, live (no
    /// reboot). The old static listeners are dropped (closing their host
    /// sockets); transient FTP-ALG data forwards and already-established
    /// connections (in `tcp_nat`) are preserved.
    fn rebind_forwards(&mut self, rules: &[PortForwardConfig]) {
        let cut = self.fwd_static_count.min(self.tcp_fwd_listeners.len());
        let transient = self.tcp_fwd_listeners.split_off(cut); // FTP-ALG data forwards
        let (mut tcp, udp) = bind_forwards(rules);
        self.fwd_static_count = tcp.len();
        tcp.extend(transient);
        self.tcp_fwd_listeners = tcp; // drops the old static listeners
        self.udp_fwd_listeners = udp;
    }

    pub fn run(&mut self) {
        while self.running.load(Ordering::Relaxed) {
            // Wait for new TX frames from the enet thread, or timeout to poll sockets.
            // Timeout of 10ms is enough for UDP/TCP response polling.
            {
                let (lock, cvar) = &*self.tx_wake;
                let mut guard = lock.lock();
                let _ = cvar.wait_for(&mut guard, Duration::from_millis(1));
            }

            // Machine reset: flush all NAT tables, close all host sockets.
            if self.ctl.reset_nat.swap(false, Ordering::AcqRel) {
                self.tcp_nat.clear();  // drops all TcpStreams, closing connections
                self.tcp_tw.clear();
                self.udp_nat.clear();  // drops all UdpSockets
                self.icmp_nat.clear(); // drops all ICMP raw sockets
                self.tcp_fwd_pending.clear();
                self.xdmcp_sessions.clear();
                self.tcp_fwd_listeners.truncate(self.fwd_static_count); // drop transient FTP data forwards
                self.ctl.routed.store(false, Ordering::Relaxed); // re-arm plug-and-play adoption
            }

            // Live subnet change requested by an embedder: swap the gateway /
            // client / netmask and flush connection state so nothing lingers on
            // the old subnet. The guest then reaches the new gateway as soon as
            // it ARPs for it (or routes to it) — no reboot. Adoption still
            // applies if the guest is on a different subnet again.
            if self.ctl.apply_subnet.swap(false, Ordering::AcqRel) {
                self.config.gateway_ip = Ipv4Addr::from(self.ctl.apply_gateway.load(Ordering::Relaxed));
                self.config.client_ip  = Ipv4Addr::from(self.ctl.apply_client.load(Ordering::Relaxed));
                self.config.netmask    = Ipv4Addr::from(self.ctl.apply_netmask.load(Ordering::Relaxed));
                self.tcp_nat.clear();
                self.tcp_tw.clear();
                self.udp_nat.clear();
                self.icmp_nat.clear();
                self.tcp_fwd_pending.clear();
                self.tcp_fwd_listeners.truncate(self.fwd_static_count); // drop transient FTP data forwards
                self.ctl.routed.store(false, Ordering::Relaxed); // re-arm adoption onto the new subnet
            }

            // Live port-forward reconfigure: rebind the static listeners.
            if self.ctl.apply_forwards.swap(false, Ordering::AcqRel) {
                let rules = self.ctl.pending_forwards.lock().take(); // drop the lock before rebinding
                if let Some(rules) = rules {
                    self.rebind_forwards(&rules);
                }
            }

            // FIXME: investigate interrupt race between TX completion and RX delivery.
            // When a gateway reply (e.g. ICMP echo) is generated synchronously while
            // draining TX frames, it can arrive at IRIX while the TX completion interrupt
            // handler is still running. IRIX writes CLRINT which clears *all* pending
            // interrupts, silently dropping the RX interrupt. Deferring to the next loop
            // iteration (after the tx_wake wait) gives IRIX time to exit the TX handler
            // before we signal RX. This masks the symptom but the root cause — whether
            // IRIX's driver should re-check for new RX after CLRINT, or whether we should
            // hold the RX interrupt line asserted until explicitly cleared — is unclear.
            // Flush replies deferred from the previous iteration; stop if ring is full.
            let pending = std::mem::take(&mut self.deferred_rx);
            let mut iter = pending.into_iter();
            for frame in &mut iter {
                if self.rx_prod.slots() == 0 {
                    // Ring full — put this frame and all remaining back, preserving order.
                    self.deferred_rx.push(frame);
                    self.deferred_rx.extend(iter);
                    break;
                }
                let _ = self.rx_prod.push(frame);
                self.rx_wake.1.notify_one();
            }

            // Drain all pending outbound frames
            while let Ok(frame) = self.tx_cons.pop() {
                self.process(&frame);
            }

            self.poll_udp();
            self.poll_tcp();
            self.poll_icmp();
            self.poll_tcp_fwd_listeners();
            self.poll_udp_fwd_listeners();
            self.poll_tftp();
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
        // Learn guest MAC from any outbound frame.
        if self.guest_mac.is_none() {
            self.guest_mac = Some(src_mac);
        }
        let etype = r16(frame, 12);
        // Record the guest's own source address so the GUI can tell what IP it's
        // using — even when that IP is wrong and nothing routes. ARP carries it
        // (sender protocol address) even with zero IP traffic, since the guest
        // ARPs for its configured gateway.
        let src_ip = match etype {
            ETHERTYPE_ARP if frame.len() >= 14 + 28 =>
                Some(Ipv4Addr::new(frame[28], frame[29], frame[30], frame[31])),
            ETHERTYPE_IP if frame.len() >= 14 + 20 =>
                Some(Ipv4Addr::new(frame[26], frame[27], frame[28], frame[29])),
            _ => None,
        };
        if let Some(ip) = src_ip {
            if !ip.is_unspecified() {
                self.ctl.observed_guest_ip.store(u32::from(ip), Ordering::Relaxed);
            }
        }
        // Candidate default gateway: a guest ARP *request* (op=1) for an
        // in-subnet address it can't resolve (NAT doesn't own it) is almost
        // always the gateway it's trying — and failing — to reach. Record it so
        // the GUI can tell what gateway the guest expects.
        if etype == ETHERTYPE_ARP && frame.len() >= 14 + 28 && r16(frame, 14 + 6) == 1 {
            let spa = [frame[28], frame[29], frame[30], frame[31]];
            let tpa = Ipv4Addr::new(frame[38], frame[39], frame[40], frame[41]);
            let t = tpa.octets();
            dlog_dev!(LogModule::Net, "NAT guest ARP who-has {} tell {}.{}.{}.{} (nat gw={} client={} guest_frames={})",
                tpa, spa[0], spa[1], spa[2], spa[3],
                self.config.gateway_ip, self.config.client_ip,
                self.ctl.guest_frames.load(Ordering::Relaxed));
            if !tpa.is_unspecified() && t != spa && t[0] == spa[0] && t[1] == spa[1] && t[2] == spa[2] {
                self.ctl.observed_gateway.store(u32::from(tpa), Ordering::Relaxed);
                let adopt_net = Ipv4Addr::new(t[0], t[1], t[2], 0);
                let conflicts = self.ctl.host_conflict(adopt_net, 24);
                // Plug-and-play: while nothing is routing yet, adopt the gateway
                // the guest is asking for so NAT answers that ARP and traffic
                // flows — no config change or restart. Self-limiting: once any IP
                // frame routes (guest_frames > 0) we stop, so a working setup is
                // never moved. Refused when the guest's subnet overlaps a host
                // network: adopting it would shadow the host's real LAN, so we
                // leave the guest unrouted and the GUI asks the user to change the
                // guest's ec0 to a non-overlapping subnet instead.
                if self.config.gateway_ip != tpa
                    && !self.ctl.routed.load(Ordering::Relaxed)
                    && !conflicts
                {
                    dlog_dev!(LogModule::Net, "NAT adopting gateway {} (was {})", tpa, self.config.gateway_ip);
                    self.config.gateway_ip = tpa;
                    self.config.client_ip = Ipv4Addr::new(t[0], t[1], t[2], 2);
                    self.config.netmask = Ipv4Addr::new(255, 255, 255, 0);
                } else if conflicts {
                    dlog_dev!(LogModule::Net, "NAT refusing to adopt {}: subnet {}/24 overlaps a host network", tpa, adopt_net);
                }
            }
        }
        dlog_dev!(LogModule::Net, "NAT TX {}", eth_summary(frame));
        if self.ctl.dbg_tcp() && etype == ETHERTYPE_IP {
            dlog_dev!(LogModule::Net, "NAT RX (IRIX→NAT):");
            log_eth_frame(">>", frame);
        }
        match etype {
            ETHERTYPE_ARP => self.handle_arp(frame, &src_mac),
            ETHERTYPE_IP  => {
                // Count only IP traffic — the actual NAT workload — as the
                // network-alive signal. ARP (and other link-layer chatter)
                // happens even when the guest's IP is missing or wrong, so
                // counting it would flash the indicator green misleadingly.
                self.ctl.guest_frames.fetch_add(1, Ordering::Relaxed);
                // A frame addressed to the gateway's MAC is off-subnet traffic
                // the guest is routing through us — that, not local/broadcast IP
                // chatter, is what locks adoption to the current gateway.
                if frame[0..6] == self.config.gateway_mac {
                    self.ctl.routed.store(true, Ordering::Relaxed);
                }
                self.handle_ip(frame, &src_mac);
            }
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

        // Inbound IP-fragment reassembly: a guest NFS WRITE with a large wsize
        // arrives fragmented. Buffer fragments keyed by (src, id, proto) and
        // dispatch the whole datagram once it's contiguous.
        let flags_frag = r16(ip, 6);
        let more_frags = flags_frag & 0x2000 != 0;
        let frag_off = ((flags_frag & 0x1fff) as usize) * 8;
        if more_frags || frag_off != 0 {
            let id = r16(ip, 4);
            let key = (u32::from(src_ip), id, proto);
            self.frag_reasm.retain(|_, v| v.last.elapsed() < Duration::from_secs(5));
            let assembled = self
                .frag_reasm
                .entry(key)
                .or_insert_with(FragReasm::new)
                .add(frag_off, payload, more_frags);
            if let Some(full) = assembled {
                self.frag_reasm.remove(&key);
                if proto == IP_PROTO_UDP {
                    self.handle_udp(src_mac, src_ip, dst_ip, &full);
                }
            }
            return;
        }

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
            self.deferred_rx.push(frame);
            return;
        }

        // Forward to external host via ICMP socket.
        // Linux: unprivileged SOCK_DGRAM+ICMPV4 works (kernel ≥3.11) but Time Exceeded
        //   replies are not delivered — traceroute sees * * * for intermediate hops.
        // macOS: unprivileged SOCK_DGRAM+ICMPV4 works for any user and — unlike Linux —
        //   recv yields the full IP header and delivers Time Exceeded, so it behaves like
        //   SOCK_RAW; falls back to SOCK_RAW (root) only if DGRAM is unavailable.
        // Windows: SOCK_RAW+ICMPV4 requires admin; Time Exceeded IS delivered on recv,
        //   so traceroute works correctly when running as Administrator.
        let is_new = !self.icmp_nat.contains_key(&(u32::from(dst_ip), ident));
        dlog_dev!(LogModule::Net, "NAT ICMP {} → {} ident={} seq={} ttl={}{}", src_ip, dst_ip, ident, seq, ttl,
            if is_new { " [new]" } else { "" });
        if self.icmp_unavailable { return; }
        let key = (u32::from(dst_ip), ident);
        let entry = self.icmp_nat.entry(key).or_insert_with(|| {
            let sock = match icmp_socket() {
                Ok(s) => { let _ = s.set_nonblocking(true); Some(s) }
                Err(e) => {
                    #[cfg(windows)]
                    eprintln!("iris: ICMP unavailable ({}); ping will time out. \
                        Run as Administrator to enable raw ICMP.", e);
                    #[cfg(not(windows))]
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
        // at the right hop count.  On Windows (SOCK_RAW) and macOS (SOCK_DGRAM) those
        // replies arrive back on this socket and we forward them to the guest.  On Linux they
        // are silently dropped by the kernel — traceroute sees * * *.
        let _ = sock.set_ttl_v4(ttl as u32);
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
                // On Linux (SOCK_DGRAM) the kernel delivers only the ICMP payload.
                // On Windows (SOCK_RAW) and macOS (SOCK_DGRAM) it prepends the outer IP header.
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

            // On Windows (SOCK_RAW) and macOS (SOCK_DGRAM) we receive Time Exceeded (type 11) for traceroute hops.
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

        // Intercept replies from guest back to a UDP port-forward listener.
        // IRIX sends to gateway_ip:host_port; we forward to the original external sender.
        if dst_ip == self.config.gateway_ip {
            if let Some(idx) = self.udp_fwd_listeners.iter()
                .position(|f| f.host_port == dport && f.guest_port == sport)
            {
                if let Some(sender) = self.udp_fwd_listeners[idx].last_sender {
                    dlog_dev!(LogModule::Net, "NAT FWD UDP reply guest:{} → host:{} → {}", sport, dport, sender);
                    let _ = self.udp_fwd_listeners[idx].sock.send_to(payload, sender);
                }
                return;
            }
        }

        match dport {
            UDP_PORT_BOOTP_SERVER => self.handle_bootp(src_mac, sport, payload),
            UDP_PORT_DNS          => self.forward_dns(src_mac, src_ip, sport, payload),
            UDP_PORT_PORTMAP if self.config.nfs.is_some()
                              => self.handle_portmap_udp(src_mac, src_ip, sport, payload),
            NFS_VM_PORT | MOUNTD_VM_PORT if self.nfs.is_some() && dst_ip == self.config.gateway_ip
                              => self.handle_nfs_udp(src_mac, src_ip, sport, dport, payload),
            UDP_PORT_TIME if dst_ip == self.config.gateway_ip
                              => self.handle_time_udp(src_mac, src_ip, sport),
            UDP_PORT_NTP  if dst_ip == self.config.gateway_ip
                              => self.handle_ntp_udp(src_mac, src_ip, sport, payload),
            UDP_PORT_TFTP if self.tftp.is_some() && dst_ip == self.config.gateway_ip
                              => self.handle_tftp_udp(src_mac, src_ip, sport, payload),
            _ => {
                // NFS/mountd: rewrite destination to localhost high port before NAT.
                let real_dst = self.nfs_remap_dst(dst_ip, dport);
                self.nat_udp(src_mac, src_ip, real_dst.0, sport, real_dst.1, payload);
            }
        }
    }

    // ── in-core NFS ─────────────────────────────────────────────────────────
    /// Dispatch a guest MOUNT/NFS RPC datagram to the in-core server and inject
    /// the reply (auto-fragmented). `server_port` is the port the guest sent to
    /// (NFS 2049 or mountd 1234), which becomes the reply's source port.
    fn handle_nfs_udp(&mut self, client_mac: &[u8; 6], client_ip: Ipv4Addr,
                      client_port: u16, server_port: u16, payload: &[u8]) {
        let reply = match self.nfs.as_mut() {
            Some(server) => server.handle(payload),
            None => return,
        };
        let Some(reply) = reply else { return };
        let udp = udp_packet(self.config.gateway_ip, client_ip, server_port, client_port, &reply);
        let id = self.ip_id;
        self.ip_id = self.ip_id.wrapping_add(1);
        self.deferred_rx.extend(ip_frames_udp(
            client_mac, &self.config.gateway_mac, self.config.gateway_ip, client_ip, id, &udp));
    }

    // ── TFTP (read-only, PROM network boot) ───────────────────────────────────
    /// Hand a guest TFTP datagram to the in-core server and inject its reply.
    /// A reply always fits one 516-byte datagram, so no fragmentation is needed.
    fn handle_tftp_udp(&mut self, client_mac: &[u8; 6], client_ip: Ipv4Addr,
                       client_port: u16, payload: &[u8]) {
        let client = (client_ip, client_port);
        let Some(reply) = self.tftp.as_mut().and_then(|s| s.handle(client, payload, Instant::now()))
        else { return };
        dlog_dev!(LogModule::Net, "NAT TFTP {}:{} ← {} bytes", client_ip, client_port, reply.len());
        self.tftp_macs.insert(client, *client_mac);
        self.send_tftp(client, *client_mac, &reply);
    }

    /// Retransmit TFTP data the guest hasn't ACKed, and forget clients whose
    /// transfer the server has given up on.
    fn poll_tftp(&mut self) {
        let Some(server) = self.tftp.as_mut() else { return };
        let due = server.tick(Instant::now());
        for (client, packet) in due {
            let Some(mac) = self.tftp_macs.get(&client).copied() else { continue };
            dlog_dev!(LogModule::Net, "NAT TFTP retransmit → {}:{}", client.0, client.1);
            self.send_tftp(client, mac, &packet);
        }
        let server = self.tftp.as_ref().unwrap();
        self.tftp_macs.retain(|client, _| server.has_transfer(client));
    }

    fn send_tftp(&mut self, client: crate::tftp::ClientId, mac: [u8; 6], packet: &[u8]) {
        let udp = udp_packet(self.config.gateway_ip, client.0, UDP_PORT_TFTP, client.1, packet);
        let frame = ip_frame(&mac, &self.config.gateway_mac,
                             self.config.gateway_ip, client.0, IP_PROTO_UDP, &udp);
        self.deferred_rx.push(frame);
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

    // ── Time services (RFC 868 + NTP) ─────────────────────────────────────────

    /// RFC 868: 32-bit big-endian seconds since 1900-01-01 00:00:00 UTC.
    fn rfc868_time() -> [u8; 4] {
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

    fn handle_time_udp(&mut self, client_mac: &[u8; 6], client_ip: Ipv4Addr, client_port: u16) {
        let payload = Self::rfc868_time();
        dlog_dev!(LogModule::Net, "NAT TIME(udp) → {}:{}", client_ip, client_port);
        let udp = udp_packet(self.config.gateway_ip, client_ip,
                             UDP_PORT_TIME, client_port, &payload);
        let frame = ip_frame(client_mac, &self.config.gateway_mac,
                             self.config.gateway_ip, client_ip, IP_PROTO_UDP, &udp);
        self.enqueue_rx(frame);
    }

    fn handle_ntp_udp(&mut self, client_mac: &[u8; 6], client_ip: Ipv4Addr,
                     client_port: u16, query: &[u8]) {
        if query.len() < 48 { return; }
        // Echo client's transmit timestamp (offset 40..48) into originate (offset 24..32).
        let client_tx: [u8; 8] = query[40..48].try_into().unwrap();
        dlog_dev!(LogModule::Net, "NAT NTP → {}:{}", client_ip, client_port);

        let mut pkt = [0u8; 48];
        // LI=0, VN=4, Mode=4 (server).
        pkt[0]  = (0 << 6) | (4 << 3) | 4;
        pkt[1]  = 1;                                // stratum=1 (primary)
        pkt[2]  = 4;                                // poll interval (log2 16s)
        pkt[3]  = 0xEC;                             // precision ~ 2^-20s
        // root delay / dispersion = 0 (already)
        pkt[12..16].copy_from_slice(b"LOCL");       // reference identifier
        let recv = Self::ntp_timestamp();
        pkt[16..24].copy_from_slice(&recv.to_be_bytes()); // reference timestamp
        pkt[24..32].copy_from_slice(&client_tx);          // originate (echoed)
        pkt[32..40].copy_from_slice(&recv.to_be_bytes()); // receive timestamp
        let tx = Self::ntp_timestamp();
        pkt[40..48].copy_from_slice(&tx.to_be_bytes());   // transmit timestamp

        let udp = udp_packet(self.config.gateway_ip, client_ip,
                             UDP_PORT_NTP, client_port, &pkt);
        let frame = ip_frame(client_mac, &self.config.gateway_mac,
                             self.config.gateway_ip, client_ip, IP_PROTO_UDP, &udp);
        self.enqueue_rx(frame);
    }

    /// RFC 868 over TCP: full SYN-ACK / data / FIN sequence in one shot.
    fn handle_time_tcp(&mut self, client_mac: &[u8; 6], client_ip: Ipv4Addr,
                       client_port: u16, client_seq: u32) {
        let gw   = self.config.gateway_ip;
        let gmac = self.config.gateway_mac;
        let server_isn = 0x6000_0000u32;
        let payload = Self::rfc868_time();
        dlog_dev!(LogModule::Net, "NAT TIME(tcp) → {}:{}", client_ip, client_port);
        // SYN-ACK
        let seg = tcp_segment(gw, client_ip, TCP_PORT_TIME, client_port,
                              server_isn, client_seq, 0x12, &[]);
        let frame = ip_frame(client_mac, &gmac, gw, client_ip, IP_PROTO_TCP, &seg);
        self.enqueue_rx(frame);
        // Data + PSH+ACK
        let seg = tcp_segment(gw, client_ip, TCP_PORT_TIME, client_port,
                              server_isn.wrapping_add(1), client_seq, 0x18, &payload);
        let frame = ip_frame(client_mac, &gmac, gw, client_ip, IP_PROTO_TCP, &seg);
        self.enqueue_rx(frame);
        // FIN+ACK
        let seg = tcp_segment(gw, client_ip, TCP_PORT_TIME, client_port,
                              server_isn.wrapping_add(1 + payload.len() as u32),
                              client_seq, 0x11, &[]);
        let frame = ip_frame(client_mac, &gmac, gw, client_ip, IP_PROTO_TCP, &seg);
        self.enqueue_rx(frame);
    }

    // ── NFS destination remapping ─────────────────────────────────────────────
    //
    // Rewrite guest outbound destination to a host-reachable address.
    //
    // IRIX sees the gateway at 192.168.0.1 but that's a virtual address iris
    // doesn't actually bind to, so unmodified TcpStream::connect() fails. We
    // rewrite any gateway-destined packet to 127.0.0.1. (NFS no longer goes
    // through here — it's served in-process by the NAT before this point.)
    fn nfs_remap_dst(&self, dst_ip: Ipv4Addr, dport: u16) -> (Ipv4Addr, u16) {
        if dst_ip != self.config.gateway_ip { return (dst_ip, dport); }
        // XDMCP X11 session: the guest's xdm dials gateway:(6000+display); relay
        // it to the real X server the XDMCP ALG recorded for that display.
        if let Some(&xserver) = self.xdmcp_sessions.get(&dport) {
            return (xserver, dport);
        }
        // Generic outbound: guest→gateway becomes guest→host loopback on
        // the same port. Lets the guest reach any service the host is
        // running on 127.0.0.1:<dport> (pyftpdlib on 2121, python -m
        // http.server, etc.).
        (Ipv4Addr::LOCALHOST, dport)
    }

    // Reverse: translate (127.0.0.1, host_port) back to the address the guest
    // dialed, so replies look like they came from the gateway.
    fn nfs_unmap_src(&self, src_ip: Ipv4Addr, sport: u16) -> (Ipv4Addr, u16) {
        // XDMCP X11 session reply: traffic from the real X server is presented to
        // the guest as coming from the gateway it dialed.
        if let Some(&xserver) = self.xdmcp_sessions.get(&sport) {
            if src_ip == xserver { return (self.config.gateway_ip, sport); }
        }
        if src_ip != Ipv4Addr::LOCALHOST { return (src_ip, sport); }
        // Generic outbound: reply from host-side dport becomes gateway:dport
        // to the guest.
        (self.config.gateway_ip, sport)
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
            let mut buf = [0u8; 65536];
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
                let id = self.ip_id; self.ip_id = self.ip_id.wrapping_add(1);
                self.deferred_rx.extend(ip_frames_udp(&client_mac, &self.config.gateway_mac,
                                                      reply_ip, client_ip, id, &udp));
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

        // Intercept SYN-ACK from guest completing a port-forward TCP handshake.
        // The guest sends: src=guest_ip:guest_port → dst=gateway_ip:ephemeral_sport
        // We match by (guest_ip, guest_port, ephemeral_sport) in tcp_fwd_pending.
        if syn && ack && dst_ip == self.config.gateway_ip {
            let fwd_key = (u32::from(src_ip), sport, dport);
            if let Some(pending) = self.tcp_fwd_pending.remove(&fwd_key) {
                dlog_dev!(LogModule::Net, "NAT FWD TCP SYN-ACK from guest {}:{} ephemeral={}", src_ip, sport, dport);
                // Send ACK to guest to complete 3-way handshake.
                let ack_seq = seq.wrapping_add(1);
                let seg = tcp_segment(self.config.gateway_ip, src_ip, dport, sport,
                                      pending.client_isn.wrapping_add(1), ack_seq, 0x10, &[]);
                let frame = ip_frame(client_mac, &self.config.gateway_mac,
                                     self.config.gateway_ip, src_ip, IP_PROTO_TCP, &seg);
                self.enqueue_rx(frame);
                // Add to tcp_nat: stream is the accepted host connection; guest is the "client".
                // Key: (dst_ip_u32=guest_ip, dport=guest_port, sport=ephemeral)
                // But tcp_nat key is (dst_ip, dst_port, src_port) — for outbound it's
                // (remote_ip, remote_port, local_client_port).
                // For forwards: "remote" = gateway_ip (visible to guest as originator),
                // "client" = guest. We'll use (gateway_ip, ephemeral, guest_port) as key
                // so that when guest sends data to gateway:ephemeral, we can look it up.
                let nat_key = (u32::from(self.config.gateway_ip), dport, sport);
                let _ = pending.stream.set_nonblocking(true);
                // server_seq/server_seq_acked track the sequence numbers WE send to IRIX.
                // Our SYN consumed client_isn, so our first data byte is client_isn+1.
                // ack_seq (= guest ISN+1) is what IRIX expects us to ACK — that goes into client_seq.
                let our_send_seq = pending.client_isn.wrapping_add(1);
                self.tcp_nat.insert(nat_key, NatTcpEntry {
                    stream:           pending.stream,
                    client_mac:       *client_mac,
                    client_ip:        src_ip,
                    client_port:      sport,
                    server_ip:        self.config.gateway_ip,
                    server_seq:       our_send_seq,
                    server_seq_acked: our_send_seq,
                    client_win:       r16(tcp, 14) as u32,
                    client_seq:       ack_seq,
                    last_use:         Instant::now(),
                    fin_wait:         false,
                    server_fin:       false,
                    retransmit:       VecDeque::new(),
                });
                return;
            }
        }

        // Intercept RFC 868 time TCP (port 37) — handle inline, no NAT entry.
        // On SYN we send SYN-ACK + the 4-byte timestamp + FIN in one burst.
        // Subsequent ACK/FIN from guest are absorbed silently.
        if dport == TCP_PORT_TIME && dst_ip == self.config.gateway_ip {
            if syn && !ack {
                self.handle_time_tcp(client_mac, src_ip, sport, seq.wrapping_add(1));
            }
            return;
        }

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
        // Don't remap if this port already has a tcp_nat entry keyed on the
        // original dst — port-forward entries use gateway_ip as the key and
        // the generic loopback remap would cause a key miss on follow-on packets.
        let pre_remap_key = (u32::from(dst_ip), dport, sport);
        let (dst_ip, dport) = if self.tcp_nat.contains_key(&pre_remap_key) {
            (dst_ip, dport)
        } else {
            self.nfs_remap_dst(dst_ip, dport)
        };

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

        // FTP ALG: a rewritten PASV reply binds a host data listener here and
        // defers registering its forward until the `entry` borrow ends below.
        let mut new_data_fwd: Option<(TcpListener, u16)> = None;
        let gateway_ip = self.config.gateway_ip;
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
                // FTP ALG: on an inbound port-forward to the guest's ftpd
                // (server = gateway, guest control port 21), rewrite a passive
                // 227 reply so the host client reaches the data connection via a
                // freshly-bound host forward. client_seq still advances by the
                // *original* length (that's what the guest sent and we ACK).
                let is_ftp_ctrl = entry.server_ip == gateway_ip && entry.client_port == 21;
                let mut handled = false;
                if is_ftp_ctrl && ftp_pasv_rewrite(payload, Ipv4Addr::LOCALHOST, 0).is_some() {
                    if let Ok(listener) = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)) {
                        if let Ok(addr) = listener.local_addr() {
                            let host_port = addr.port();
                            let _ = listener.set_nonblocking(true);
                            if let Some((rewritten, data_port)) =
                                ftp_pasv_rewrite(payload, Ipv4Addr::LOCALHOST, host_port)
                            {
                                dlog_dev!(LogModule::Net,
                                    "NAT FTP ALG: PASV guest data port {} -> host 127.0.0.1:{}",
                                    data_port, host_port);
                                let _ = entry.stream.write_all(&rewritten);
                                new_data_fwd = Some((listener, data_port));
                                handled = true;
                            }
                        }
                    }
                }
                if !handled {
                    let _ = entry.stream.write_all(payload);
                }
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

        // Register the FTP-ALG data forward now that the `entry` borrow is gone.
        // Transient and FIFO-bounded so a long session's PASV transfers don't
        // leak host listeners; the oldest is dropped once the cap is hit.
        if let Some((listener, data_port)) = new_data_fwd {
            const MAX_FTP_DATA_FWD: usize = 16;
            while self.tcp_fwd_listeners.len() >= self.fwd_static_count + MAX_FTP_DATA_FWD
                && self.tcp_fwd_listeners.len() > self.fwd_static_count
            {
                self.tcp_fwd_listeners.remove(self.fwd_static_count); // drop oldest dynamic
            }
            self.tcp_fwd_listeners.push(TcpFwdListener { listener, guest_port: data_port });
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

    // ── Inbound port-forward: TCP listener poll ───────────────────────────────
    //
    // For each TCP forward rule, accept() any pending connections (non-blocking).
    // On accept: allocate an ephemeral port, inject a synthetic SYN frame into the
    // guest's ethernet as if gateway_ip:ephemeral opened a connection to the guest.
    // The accepted TcpStream is stored in tcp_fwd_pending until the guest replies
    // with SYN-ACK (handled in handle_tcp above).
    // Pick a source port for a synthesized inbound-forward connection that isn't
    // already live for this guest port. Forwards to rsh/rlogin (513/514) must use
    // a reserved 512..=1023 port or the guest's rshd/rlogind rejects them; all
    // others use the 49152.. ephemeral range. The chosen port becomes part of the
    // tcp_fwd_pending / tcp_nat / tcp_tw keys, so probe past any still in use — a
    // blindly wrapped counter would otherwise overwrite a live entry and break
    // that connection. Returns None if every port in the range is occupied.
    fn alloc_fwd_sport(&mut self, guest_ip: Ipv4Addr, guest_port: u16) -> Option<u16> {
        let reserved = matches!(guest_port, 513 | 514);
        let (lo, hi) = if reserved { (512u16, 1023u16) } else { (49152u16, 65535u16) };
        let gi = u32::from(guest_ip);
        let gw = u32::from(self.config.gateway_ip);
        let mut cur = if reserved { self.fwd_reserved_next } else { self.fwd_ephemeral_next };
        if cur < lo || cur > hi { cur = lo; }
        let mut chosen = None;
        for _ in 0..=(hi - lo) {
            let p = cur;
            cur = if p >= hi { lo } else { p + 1 };
            if !self.tcp_fwd_pending.contains_key(&(gi, guest_port, p))
                && !self.tcp_nat.contains_key(&(gw, p, guest_port))
                && !self.tcp_tw.contains_key(&(gw, p, guest_port))
            {
                chosen = Some(p);
                break;
            }
        }
        if reserved { self.fwd_reserved_next = cur; } else { self.fwd_ephemeral_next = cur; }
        chosen
    }

    fn poll_tcp_fwd_listeners(&mut self) {
        // Collect accepted streams to avoid mut borrow conflict while iterating listeners.
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
            // Forward to the guest's *actual* IP (learned from its traffic), not
            // the assumed NAT client address — a static guest can be on any host.
            let guest_ip   = self.ctl.observed_guest_ip().unwrap_or(self.config.client_ip);
            let gw_ip      = self.config.gateway_ip;
            let gw_mac     = self.config.gateway_mac;

            // Use the guest's real MAC if we've learned it; the SYN must be unicast or
            // IRIX will drop it (broadcast dst MAC with unicast dst IP is rejected).
            let Some(guest_mac) = self.guest_mac else {
                dlog_dev!(LogModule::Net, "NAT FWD TCP: guest MAC not yet known, deferring SYN for guest:{}", guest_port);
                // Drop this accept without consuming a source port; the host will retry.
                // In practice the guest MAC is always known by the time port-forward traffic arrives.
                drop(stream);
                continue;
            };

            // Synthesize the client's source port (the guest never sees the host
            // client's real one). None means the range is exhausted — drop the accept.
            let Some(ephemeral) = self.alloc_fwd_sport(guest_ip, guest_port) else {
                dlog_dev!(LogModule::Net, "NAT FWD TCP: no free source port for guest:{}, dropping accept", guest_port);
                drop(stream);
                continue;
            };
            let client_isn = 0x6000_0000u32.wrapping_add(ephemeral as u32);

            dlog_dev!(LogModule::Net, "NAT FWD TCP inject SYN {}:{} → {}:{}", gw_ip, ephemeral, guest_ip, guest_port);
            let seg = tcp_segment(gw_ip, guest_ip, ephemeral, guest_port,
                                  client_isn, 0, 0x02, &[]);
            let frame = ip_frame(&guest_mac, &gw_mac, gw_ip, guest_ip, IP_PROTO_TCP, &seg);
            self.enqueue_rx(frame);

            let fwd_key = (u32::from(guest_ip), guest_port, ephemeral);
            self.tcp_fwd_pending.insert(fwd_key, TcpFwdPending { stream, client_isn });
        }

        // Stale pending entries (guest never answered the SYN) are flushed on machine reset.
        // No per-entry expiry here since we don't track a timestamp; the guest will RST
        // or the host side will close if the connection is abandoned.
    }

    // ── Inbound port-forward: UDP listener poll ───────────────────────────────
    //
    // For each UDP forward rule, drain all pending datagrams (non-blocking).
    // Each datagram is injected into the guest as a UDP frame from gateway_ip:host_port
    // to guest_ip:guest_port, so the guest can reply to gateway_ip:host_port.
    // Replies from the guest go through the normal UDP NAT path (nat_udp) and are
    // forwarded back to the host-side sender via the UDP forward socket.
    fn poll_udp_fwd_listeners(&mut self) {
        let mut received: Vec<(Vec<u8>, SocketAddr, u16, u16)> = Vec::new();
        for fwd in &self.udp_fwd_listeners {
            let mut buf = [0u8; 65536];
            loop {
                match fwd.sock.recv_from(&mut buf) {
                    Ok((n, from)) => {
                        dlog_dev!(LogModule::Net, "NAT FWD UDP {} → guest:{} len={}", from, fwd.guest_port, n);
                        received.push((buf[..n].to_vec(), from, fwd.guest_port, fwd.host_port));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(e) => { dlog_dev!(LogModule::Net, "NAT FWD UDP recv error: {}", e); break; }
                }
            }
        }
        for (data, from, guest_port, host_port) in received {
            let Some(guest_mac) = self.guest_mac else {
                dlog_dev!(LogModule::Net, "NAT FWD UDP: guest MAC not yet known, dropping datagram for guest:{}", guest_port);
                continue;
            };
            // Record sender so we can forward IRIX's reply back to them.
            if let Some(fwd) = self.udp_fwd_listeners.iter_mut()
                .find(|f| f.host_port == host_port && f.guest_port == guest_port)
            {
                fwd.last_sender = Some(from);
            }
            let guest_ip = self.ctl.observed_guest_ip().unwrap_or(self.config.client_ip);
            let gw_ip    = self.config.gateway_ip;
            let gw_mac   = self.config.gateway_mac;
            // XDMCP ALG: on the control forward (→ guest:177), rewrite a Request's
            // connection-addresses to the gateway so the guest's xdm opens the X11
            // session at gateway:(6000+display) — which nfs_remap_dst then relays
            // to the real X server. Record display→X-server (the datagram source).
            let data = if guest_port == XDMCP_GUEST_PORT {
                match (crate::xdmcp::rewrite_request_ipv4(&data, gw_ip), from.ip()) {
                    (Some(rw), IpAddr::V4(xserver)) => {
                        let x11_port = X11_BASE_PORT.wrapping_add(rw.display_number);
                        self.xdmcp_sessions.insert(x11_port, xserver);
                        dlog_dev!(LogModule::Net, "XDMCP Request: display {} → relay gateway:{} to X server {}", rw.display_number, x11_port, xserver);
                        rw.packet
                    }
                    _ => data,
                }
            } else {
                data
            };
            // Inject as UDP: src=gateway_ip:host_port dst=guest_ip:guest_port
            let udp = udp_packet(gw_ip, guest_ip, host_port, guest_port, &data);
            let id = self.ip_id; self.ip_id = self.ip_id.wrapping_add(1);
            self.deferred_rx.extend(ip_frames_udp(&guest_mac, &gw_mac, gw_ip, guest_ip, id, &udp));
        }
    }
}

impl NetBackend for NatEngine {
    fn run(&mut self) {
        // Delegate to the inherent run loop.
        NatEngine::run(self)
    }
}
