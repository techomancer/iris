//! Ethernet / ARP / IPv4 / ICMP / UDP / TCP packets as typed views and builders.
//!
//! A frame is one contiguous byte buffer (what the seeq rings carry). Parsing
//! wraps a slice in a view after checking it is long enough for every field
//! the view exposes, so accessors never panic and never need offsets at the
//! call site. Builders write a whole frame into one `Vec` and fill in lengths
//! and checksums; nothing else computes a checksum.
//!
//! Multi-byte fields are network byte order on the wire. That is the Edge, so
//! the `from_be_bytes`/`to_be_bytes` here are the correct place to swap.

use std::net::Ipv4Addr;

pub type MacAddr = [u8; 6];
pub const BROADCAST_MAC: MacAddr = [0xff; 6];

pub const ETHERTYPE_IPV4: u16 = 0x0800;
pub const ETHERTYPE_ARP: u16 = 0x0806;

pub const PROTO_ICMP: u8 = 1;
pub const PROTO_TCP: u8 = 6;
pub const PROTO_UDP: u8 = 17;

pub const ETH_HDR: usize = 14;
pub const IPV4_HDR: usize = 20;
pub const UDP_HDR: usize = 8;
pub const TCP_HDR: usize = 20;
/// Smallest Ethernet frame without the FCS; shorter frames are zero-padded.
pub const ETH_MIN: usize = 60;
pub const MTU: usize = 1500;
/// Largest TCP payload in one unfragmented frame.
pub const TCP_MSS: u16 = (MTU - IPV4_HDR - TCP_HDR) as u16;

#[inline]
fn be16(b: &[u8], o: usize) -> u16 {
    u16::from_be_bytes([b[o], b[o + 1]])
}
#[inline]
fn be32(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn put16(b: &mut [u8], o: usize, v: u16) {
    b[o..o + 2].copy_from_slice(&v.to_be_bytes());
}
#[inline]
fn put32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_be_bytes());
}
#[inline]
fn ip_at(b: &[u8], o: usize) -> Ipv4Addr {
    Ipv4Addr::new(b[o], b[o + 1], b[o + 2], b[o + 3])
}
#[inline]
fn mac_at(b: &[u8], o: usize) -> MacAddr {
    b[o..o + 6].try_into().unwrap()
}

// ── Checksums ────────────────────────────────────────────────────────────────

/// Ones'-complement sum of `data` folded into `acc` (not yet complemented).
fn sum16(mut acc: u32, data: &[u8]) -> u32 {
    let mut chunks = data.chunks_exact(2);
    for c in &mut chunks {
        acc += u16::from_be_bytes([c[0], c[1]]) as u32;
    }
    if let [last] = chunks.remainder() {
        acc += (*last as u32) << 8;
    }
    acc
}

fn fold(mut acc: u32) -> u16 {
    while acc >> 16 != 0 {
        acc = (acc & 0xffff) + (acc >> 16);
    }
    !(acc as u16)
}

/// The Internet checksum (RFC 1071) of `data`.
pub fn checksum(data: &[u8]) -> u16 {
    fold(sum16(0, data))
}

/// TCP/UDP checksum over the IPv4 pseudo-header and `segment` (whose own
/// checksum field must be zero).
pub fn pseudo_checksum(src: Ipv4Addr, dst: Ipv4Addr, proto: u8, segment: &[u8]) -> u16 {
    let mut acc = sum16(0, &src.octets());
    acc = sum16(acc, &dst.octets());
    acc += proto as u32;
    acc += segment.len() as u32;
    fold(sum16(acc, segment))
}

// ── Ethernet ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct Ether<'a>(&'a [u8]);

impl<'a> Ether<'a> {
    pub fn parse(b: &'a [u8]) -> Option<Self> {
        (b.len() >= ETH_HDR).then_some(Self(b))
    }
    pub fn dst(&self) -> MacAddr { mac_at(self.0, 0) }
    pub fn src(&self) -> MacAddr { mac_at(self.0, 6) }
    pub fn ethertype(&self) -> u16 { be16(self.0, 12) }
    pub fn payload(&self) -> &'a [u8] { &self.0[ETH_HDR..] }
    pub fn bytes(&self) -> &'a [u8] { self.0 }

    pub fn arp(&self) -> Option<Arp<'a>> {
        (self.ethertype() == ETHERTYPE_ARP).then(|| Arp::parse(self.payload())).flatten()
    }
    pub fn ipv4(&self) -> Option<Ipv4<'a>> {
        (self.ethertype() == ETHERTYPE_IPV4).then(|| Ipv4::parse(self.payload())).flatten()
    }
}

/// An Ethernet frame around `payload`, padded to the 60-byte minimum.
pub fn ether(dst: MacAddr, src: MacAddr, ethertype: u16, payload: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(ETH_MIN.max(ETH_HDR + payload.len()));
    write_ether(&mut f, dst, src, ethertype);
    f.extend_from_slice(payload);
    pad(&mut f);
    f
}

fn write_ether(f: &mut Vec<u8>, dst: MacAddr, src: MacAddr, ethertype: u16) {
    f.extend_from_slice(&dst);
    f.extend_from_slice(&src);
    f.extend_from_slice(&ethertype.to_be_bytes());
}

fn pad(f: &mut Vec<u8>) {
    if f.len() < ETH_MIN {
        f.resize(ETH_MIN, 0);
    }
}

// ── ARP ──────────────────────────────────────────────────────────────────────

pub const ARP_REQUEST: u16 = 1;
pub const ARP_REPLY: u16 = 2;
const ARP_LEN: usize = 28;

/// An Ethernet/IPv4 ARP packet (other hardware or protocol types don't parse).
#[derive(Clone, Copy)]
pub struct Arp<'a>(&'a [u8]);

impl<'a> Arp<'a> {
    pub fn parse(b: &'a [u8]) -> Option<Self> {
        let ok = b.len() >= ARP_LEN
            && be16(b, 0) == 1 && be16(b, 2) == ETHERTYPE_IPV4 && b[4] == 6 && b[5] == 4;
        ok.then_some(Self(b))
    }
    pub fn op(&self) -> u16 { be16(self.0, 6) }
    pub fn sender_mac(&self) -> MacAddr { mac_at(self.0, 8) }
    pub fn sender_ip(&self) -> Ipv4Addr { ip_at(self.0, 14) }
    pub fn target_mac(&self) -> MacAddr { mac_at(self.0, 18) }
    pub fn target_ip(&self) -> Ipv4Addr { ip_at(self.0, 24) }
}

/// A complete ARP frame. Replies go to `target_mac`; requests are broadcast.
pub fn arp(op: u16, sender_mac: MacAddr, sender_ip: Ipv4Addr,
           target_mac: MacAddr, target_ip: Ipv4Addr) -> Vec<u8> {
    let mut a = [0u8; ARP_LEN];
    put16(&mut a, 0, 1);
    put16(&mut a, 2, ETHERTYPE_IPV4);
    a[4] = 6;
    a[5] = 4;
    put16(&mut a, 6, op);
    a[8..14].copy_from_slice(&sender_mac);
    a[14..18].copy_from_slice(&sender_ip.octets());
    a[18..24].copy_from_slice(&target_mac);
    a[24..28].copy_from_slice(&target_ip.octets());
    let dst = if op == ARP_REQUEST { BROADCAST_MAC } else { target_mac };
    ether(dst, sender_mac, ETHERTYPE_ARP, &a)
}

// ── IPv4 ─────────────────────────────────────────────────────────────────────

const IP_MF: u16 = 0x2000;
const IP_OFFSET_MASK: u16 = 0x1fff;

/// An IPv4 header and payload. `payload()` stops at the header's total
/// length, so Ethernet padding never leaks into the transport layer.
#[derive(Clone, Copy)]
pub struct Ipv4<'a> {
    b: &'a [u8],
    hlen: usize,
    end: usize,
}

impl<'a> Ipv4<'a> {
    /// Needs a version-4 header of at least 20 bytes that fits in `b`. A total
    /// length beyond `b` is clamped to what arrived.
    pub fn parse(b: &'a [u8]) -> Option<Self> {
        if b.len() < IPV4_HDR || b[0] >> 4 != 4 {
            return None;
        }
        let hlen = ((b[0] & 0x0f) as usize) * 4;
        let total = be16(b, 2) as usize;
        if hlen < IPV4_HDR || b.len() < hlen || total < hlen {
            return None;
        }
        Some(Self { b, hlen, end: total.min(b.len()) })
    }
    pub fn header_len(&self) -> usize { self.hlen }
    pub fn total_len(&self) -> u16 { be16(self.b, 2) }
    pub fn id(&self) -> u16 { be16(self.b, 4) }
    pub fn more_fragments(&self) -> bool { be16(self.b, 6) & IP_MF != 0 }
    /// Fragment offset in bytes.
    pub fn frag_offset(&self) -> usize { ((be16(self.b, 6) & IP_OFFSET_MASK) as usize) * 8 }
    pub fn is_fragment(&self) -> bool { self.more_fragments() || self.frag_offset() != 0 }
    pub fn ttl(&self) -> u8 { self.b[8] }
    pub fn proto(&self) -> u8 { self.b[9] }
    pub fn src(&self) -> Ipv4Addr { ip_at(self.b, 12) }
    pub fn dst(&self) -> Ipv4Addr { ip_at(self.b, 16) }
    pub fn header(&self) -> &'a [u8] { &self.b[..self.hlen] }
    pub fn payload(&self) -> &'a [u8] { &self.b[self.hlen..self.end] }
    pub fn checksum_ok(&self) -> bool { checksum(self.header()) == 0 }

    pub fn tcp(&self) -> Option<Tcp<'a>> {
        (self.proto() == PROTO_TCP && !self.is_fragment()).then(|| Tcp::parse(self.payload())).flatten()
    }
    pub fn udp(&self) -> Option<Udp<'a>> {
        (self.proto() == PROTO_UDP && !self.is_fragment()).then(|| Udp::parse(self.payload())).flatten()
    }
    pub fn icmp(&self) -> Option<Icmp<'a>> {
        (self.proto() == PROTO_ICMP && !self.is_fragment()).then(|| Icmp::parse(self.payload())).flatten()
    }
}

/// Addressing for a frame carrying an IPv4 datagram.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ipv4Route {
    pub dst_mac: MacAddr,
    pub src_mac: MacAddr,
    pub src: Ipv4Addr,
    pub dst: Ipv4Addr,
}

impl Ipv4Route {
    pub fn new(dst_mac: MacAddr, src_mac: MacAddr, src: Ipv4Addr, dst: Ipv4Addr) -> Self {
        Self { dst_mac, src_mac, src, dst }
    }

    /// Ethernet + IPv4 headers with `l4_len` bytes reserved after them.
    /// Returns the frame and the offset of the transport header.
    fn start(&self, proto: u8, id: u16, frag: u16, l4_len: usize) -> (Vec<u8>, usize) {
        let mut f = Vec::with_capacity(ETH_MIN.max(ETH_HDR + IPV4_HDR + l4_len));
        write_ether(&mut f, self.dst_mac, self.src_mac, ETHERTYPE_IPV4);
        let ip = f.len();
        f.resize(ip + IPV4_HDR, 0);
        let h = &mut f[ip..];
        h[0] = 0x45;
        put16(h, 2, (IPV4_HDR + l4_len) as u16);
        put16(h, 4, id);
        put16(h, 6, frag);
        h[8] = 64;
        h[9] = proto;
        h[12..16].copy_from_slice(&self.src.octets());
        h[16..20].copy_from_slice(&self.dst.octets());
        let c = checksum(&f[ip..ip + IPV4_HDR]);
        put16(&mut f, ip + 10, c);
        (f, ip + IPV4_HDR)
    }

    /// An unfragmented datagram with a ready-made transport payload.
    pub fn raw(&self, proto: u8, id: u16, payload: &[u8]) -> Vec<u8> {
        let (mut f, _) = self.start(proto, id, 0, payload.len());
        f.extend_from_slice(payload);
        pad(&mut f);
        f
    }

    /// An ICMP message; its checksum is computed here.
    pub fn icmp(&self, id: u16, ty: u8, code: u8, rest: [u8; 4], body: &[u8]) -> Vec<u8> {
        let (mut f, l4) = self.start(PROTO_ICMP, id, 0, 8 + body.len());
        f.extend_from_slice(&[ty, code, 0, 0]);
        f.extend_from_slice(&rest);
        f.extend_from_slice(body);
        let c = checksum(&f[l4..]);
        put16(&mut f, l4 + 2, c);
        pad(&mut f);
        f
    }

    /// One UDP datagram in one frame, however large.
    pub fn udp(&self, id: u16, sport: u16, dport: u16, payload: &[u8]) -> Vec<u8> {
        let (mut f, _) = self.start(PROTO_UDP, id, 0, UDP_HDR + payload.len());
        write_udp(&mut f, self.src, self.dst, sport, dport, payload);
        pad(&mut f);
        f
    }

    /// One ICMP message, IP-fragmented into frames no larger than the MTU.
    pub fn icmp_fragments(&self, id: u16, ty: u8, code: u8, rest: [u8; 4], body: &[u8]) -> Vec<Vec<u8>> {
        let f = self.icmp(id, ty, code, rest, body);
        if f.len() <= ETH_HDR + MTU {
            return vec![f];
        }
        self.fragments(PROTO_ICMP, id, &f[ETH_HDR + IPV4_HDR..])
    }

    /// One UDP datagram, IP-fragmented into frames no larger than the MTU.
    pub fn udp_fragments(&self, id: u16, sport: u16, dport: u16, payload: &[u8]) -> Vec<Vec<u8>> {
        if UDP_HDR + payload.len() <= MTU - IPV4_HDR {
            return vec![self.udp(id, sport, dport, payload)];
        }
        let mut dgram = Vec::with_capacity(UDP_HDR + payload.len());
        write_udp(&mut dgram, self.src, self.dst, sport, dport, payload);
        self.fragments(PROTO_UDP, id, &dgram)
    }

    /// A whole transport message split into IP fragments that fit the MTU.
    fn fragments(&self, proto: u8, id: u16, dgram: &[u8]) -> Vec<Vec<u8>> {
        // Fragment payloads must be multiples of 8 bytes except the last.
        const CHUNK: usize = (MTU - IPV4_HDR) / 8 * 8;
        dgram
            .chunks(CHUNK)
            .enumerate()
            .map(|(i, chunk)| {
                let offset = i * CHUNK;
                let more = offset + chunk.len() < dgram.len();
                let frag = (offset / 8) as u16 | if more { IP_MF } else { 0 };
                let (mut f, _) = self.start(proto, id, frag, chunk.len());
                f.extend_from_slice(chunk);
                pad(&mut f);
                f
            })
            .collect()
    }

    /// One TCP segment.
    pub fn tcp(&self, id: u16, h: &TcpHeader, payload: &[u8]) -> Vec<u8> {
        let opt_len = if h.mss.is_some() { 4 } else { 0 };
        let hdr = TCP_HDR + opt_len;
        let (mut f, l4) = self.start(PROTO_TCP, id, 0, hdr + payload.len());
        f.resize(l4 + hdr, 0);
        {
            let t = &mut f[l4..];
            put16(t, 0, h.sport);
            put16(t, 2, h.dport);
            put32(t, 4, h.seq);
            put32(t, 8, h.ack);
            t[12] = ((hdr / 4) as u8) << 4;
            t[13] = h.flags.0;
            put16(t, 14, h.window);
            if let Some(mss) = h.mss {
                t[20] = 2;
                t[21] = 4;
                put16(t, 22, mss);
            }
        }
        f.extend_from_slice(payload);
        let c = pseudo_checksum(self.src, self.dst, PROTO_TCP, &f[l4..]);
        put16(&mut f, l4 + 16, c);
        pad(&mut f);
        f
    }
}

fn write_udp(f: &mut Vec<u8>, src: Ipv4Addr, dst: Ipv4Addr, sport: u16, dport: u16, payload: &[u8]) {
    let at = f.len();
    f.extend_from_slice(&sport.to_be_bytes());
    f.extend_from_slice(&dport.to_be_bytes());
    f.extend_from_slice(&((UDP_HDR + payload.len()) as u16).to_be_bytes());
    f.extend_from_slice(&[0, 0]);
    f.extend_from_slice(payload);
    let c = match pseudo_checksum(src, dst, PROTO_UDP, &f[at..]) {
        0 => 0xffff, // a computed zero is sent as all ones (RFC 768)
        c => c,
    };
    put16(f, at + 6, c);
}

// ── ICMP ─────────────────────────────────────────────────────────────────────

pub const ICMP_ECHO_REPLY: u8 = 0;
pub const ICMP_ECHO_REQUEST: u8 = 8;
pub const ICMP_TIME_EXCEEDED: u8 = 11;

#[derive(Clone, Copy)]
pub struct Icmp<'a>(&'a [u8]);

impl<'a> Icmp<'a> {
    pub fn parse(b: &'a [u8]) -> Option<Self> {
        (b.len() >= 8).then_some(Self(b))
    }
    pub fn ty(&self) -> u8 { self.0[0] }
    pub fn code(&self) -> u8 { self.0[1] }
    /// Echo identifier (bytes 4..6; "unused" in other types).
    pub fn ident(&self) -> u16 { be16(self.0, 4) }
    pub fn seq(&self) -> u16 { be16(self.0, 6) }
    /// Bytes 4..8: identifier and sequence for echo messages.
    pub fn rest(&self) -> [u8; 4] { self.0[4..8].try_into().unwrap() }
    pub fn body(&self) -> &'a [u8] { &self.0[8..] }
    pub fn bytes(&self) -> &'a [u8] { self.0 }
}

// ── UDP ──────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct Udp<'a>(&'a [u8]);

impl<'a> Udp<'a> {
    /// The payload stops at the header's length field when it is sane.
    pub fn parse(b: &'a [u8]) -> Option<Self> {
        if b.len() < UDP_HDR {
            return None;
        }
        let len = be16(b, 4) as usize;
        let end = if (UDP_HDR..=b.len()).contains(&len) { len } else { b.len() };
        Some(Self(&b[..end]))
    }
    pub fn sport(&self) -> u16 { be16(self.0, 0) }
    pub fn dport(&self) -> u16 { be16(self.0, 2) }
    pub fn payload(&self) -> &'a [u8] { &self.0[UDP_HDR..] }
}

// ── TCP ──────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct TcpFlags(pub u8);

impl TcpFlags {
    pub const FIN: Self = Self(0x01);
    pub const SYN: Self = Self(0x02);
    pub const RST: Self = Self(0x04);
    pub const PSH: Self = Self(0x08);
    pub const ACK: Self = Self(0x10);
    pub const SYN_ACK: Self = Self(0x12);
    pub const FIN_ACK: Self = Self(0x11);
    pub const RST_ACK: Self = Self(0x14);
    pub const PSH_ACK: Self = Self(0x18);

    pub fn has(self, f: Self) -> bool { self.0 & f.0 == f.0 }
    pub fn fin(self) -> bool { self.has(Self::FIN) }
    pub fn syn(self) -> bool { self.has(Self::SYN) }
    pub fn rst(self) -> bool { self.has(Self::RST) }
    pub fn ack(self) -> bool { self.has(Self::ACK) }
}

impl std::ops::BitOr for TcpFlags {
    type Output = Self;
    fn bitor(self, o: Self) -> Self { Self(self.0 | o.0) }
}

impl std::fmt::Debug for TcpFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let names = [(Self::SYN, "SYN"), (Self::ACK, "ACK"), (Self::FIN, "FIN"),
                     (Self::RST, "RST"), (Self::PSH, "PSH")];
        let set: Vec<&str> = names.iter().filter(|(b, _)| self.has(*b)).map(|(_, n)| *n).collect();
        write!(f, "{}", if set.is_empty() { "-".to_string() } else { set.join("|") })
    }
}

#[derive(Clone, Copy)]
pub struct Tcp<'a> {
    b: &'a [u8],
    hlen: usize,
}

impl<'a> Tcp<'a> {
    pub fn parse(b: &'a [u8]) -> Option<Self> {
        if b.len() < TCP_HDR {
            return None;
        }
        let hlen = ((b[12] >> 4) as usize) * 4;
        (hlen >= TCP_HDR && hlen <= b.len()).then_some(Self { b, hlen })
    }
    pub fn sport(&self) -> u16 { be16(self.b, 0) }
    pub fn dport(&self) -> u16 { be16(self.b, 2) }
    pub fn seq(&self) -> u32 { be32(self.b, 4) }
    pub fn ack(&self) -> u32 { be32(self.b, 8) }
    pub fn flags(&self) -> TcpFlags { TcpFlags(self.b[13] & 0x3f) }
    pub fn window(&self) -> u16 { be16(self.b, 14) }
    pub fn payload(&self) -> &'a [u8] { &self.b[self.hlen..] }
    /// Sequence space this segment occupies: payload plus one each for SYN and FIN.
    pub fn seq_len(&self) -> u32 {
        let f = self.flags();
        self.payload().len() as u32 + f.syn() as u32 + f.fin() as u32
    }
    /// The MSS option, if present.
    pub fn mss(&self) -> Option<u16> {
        let opts = &self.b[TCP_HDR..self.hlen];
        let mut i = 0;
        while i < opts.len() {
            match opts[i] {
                0 => break,
                1 => i += 1,
                kind => {
                    let len = *opts.get(i + 1)? as usize;
                    if len < 2 || i + len > opts.len() {
                        return None;
                    }
                    if kind == 2 && len == 4 {
                        return Some(be16(opts, i + 2));
                    }
                    i += len;
                }
            }
        }
        None
    }
}

/// The fields a TCP segment is built from.
#[derive(Clone, Copy, Debug, Default)]
pub struct TcpHeader {
    pub sport: u16,
    pub dport: u16,
    pub seq: u32,
    pub ack: u32,
    pub flags: TcpFlags,
    pub window: u16,
    /// Sent as an MSS option (SYN and SYN-ACK only).
    pub mss: Option<u16>,
}

// ── Debug helpers ────────────────────────────────────────────────────────────

pub fn mac_str(m: &MacAddr) -> String {
    format!("{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}", m[0], m[1], m[2], m[3], m[4], m[5])
}

/// One-line summary of an Ethernet frame for debug logging.
pub fn summary(frame: &[u8]) -> String {
    let Some(eth) = Ether::parse(frame) else {
        return format!("<runt {} bytes>", frame.len());
    };
    let inner = if let Some(a) = eth.arp() {
        let op = if a.op() == ARP_REQUEST { "who-has" } else { "is-at" };
        format!("ARP {} {} tell {}", op, a.target_ip(), a.sender_ip())
    } else if let Some(ip) = eth.ipv4() {
        let l4 = match ip.proto() {
            PROTO_ICMP => "ICMP".to_string(),
            PROTO_TCP => match ip.tcp() {
                Some(t) => format!("TCP :{}->{} {:?}", t.sport(), t.dport(), t.flags()),
                None => "TCP".to_string(),
            },
            PROTO_UDP => match ip.udp() {
                Some(u) => format!("UDP :{}->{}", u.sport(), u.dport()),
                None => "UDP".to_string(),
            },
            n => format!("proto={}", n),
        };
        format!("IPv4 {} > {}  {}", ip.src(), ip.dst(), l4)
    } else {
        format!("etype={:#06x}", eth.ethertype())
    };
    format!("{} > {}  {}  {} bytes", mac_str(&eth.src()), mac_str(&eth.dst()), inner, frame.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: MacAddr = [0x08, 0, 0x69, 1, 2, 3];
    const B: MacAddr = [0x02, 0, 0, 0, 0, 1];
    const GUEST: Ipv4Addr = Ipv4Addr::new(192, 168, 0, 2);
    const GW: Ipv4Addr = Ipv4Addr::new(192, 168, 0, 1);

    fn route() -> Ipv4Route { Ipv4Route::new(A, B, GW, GUEST) }

    #[test]
    fn checksum_matches_rfc1071_example() {
        // RFC 1071 §3 worked example: sum 0xddf2, checksum its complement.
        let data = [0x00, 0x01, 0xf2, 0x03, 0xf4, 0xf5, 0xf6, 0xf7];
        assert_eq!(checksum(&data), !0xddf2);
        // Odd length pads with a zero byte.
        assert_eq!(checksum(&[0xab]), !0xab00);
    }

    #[test]
    fn udp_round_trip_with_valid_checksums() {
        let f = route().udp(7, 1234, 69, b"hello");
        assert_eq!(f.len(), ETH_MIN, "short frames are padded");
        let eth = Ether::parse(&f).unwrap();
        assert_eq!((eth.dst(), eth.src()), (A, B));
        let ip = eth.ipv4().unwrap();
        assert!(ip.checksum_ok());
        assert_eq!((ip.src(), ip.dst(), ip.id(), ip.ttl()), (GW, GUEST, 7, 64));
        assert_eq!(ip.payload().len(), UDP_HDR + 5, "padding is not payload");
        assert_eq!(pseudo_checksum(GW, GUEST, PROTO_UDP, ip.payload()), 0);
        let u = ip.udp().unwrap();
        assert_eq!((u.sport(), u.dport(), u.payload()), (1234, 69, &b"hello"[..]));
    }

    #[test]
    fn tcp_round_trip_with_mss_option() {
        let h = TcpHeader { sport: 80, dport: 4000, seq: 0x1000_0000, ack: 42,
                            flags: TcpFlags::SYN_ACK, window: 8192, mss: Some(TCP_MSS) };
        let f = route().tcp(1, &h, &[]);
        let ip = Ether::parse(&f).unwrap().ipv4().unwrap();
        assert_eq!(pseudo_checksum(GW, GUEST, PROTO_TCP, ip.payload()), 0);
        let t = ip.tcp().unwrap();
        assert_eq!((t.sport(), t.dport(), t.seq(), t.ack(), t.window()), (80, 4000, 0x1000_0000, 42, 8192));
        assert!(t.flags().syn() && t.flags().ack() && !t.flags().fin());
        assert_eq!(t.mss(), Some(1460));
        assert_eq!(t.seq_len(), 1, "SYN takes one sequence number");
        assert!(t.payload().is_empty());

        let data = vec![0x5a; 1000];
        let h = TcpHeader { flags: TcpFlags::PSH_ACK | TcpFlags::FIN, mss: None, ..h };
        let f = route().tcp(2, &h, &data);
        let ip = Ether::parse(&f).unwrap().ipv4().unwrap();
        assert_eq!(pseudo_checksum(GW, GUEST, PROTO_TCP, ip.payload()), 0);
        let t = ip.tcp().unwrap();
        assert_eq!(t.payload(), &data[..]);
        assert_eq!(t.mss(), None);
        assert_eq!(t.seq_len(), 1001);
    }

    #[test]
    fn udp_fragments_reassemble_to_the_datagram() {
        let payload: Vec<u8> = (0..5000u32).map(|i| i as u8).collect();
        let frames = route().udp_fragments(99, 2049, 1023, &payload);
        assert_eq!(frames.len(), 4, "5008 bytes = 3 x 1480 + 568");
        let mut whole = Vec::new();
        for (i, f) in frames.iter().enumerate() {
            assert!(f.len() <= ETH_HDR + MTU);
            let ip = Ether::parse(f).unwrap().ipv4().unwrap();
            assert!(ip.checksum_ok());
            assert_eq!(ip.id(), 99);
            assert_eq!(ip.frag_offset(), whole.len());
            assert_eq!(ip.more_fragments(), i + 1 < frames.len());
            assert!(ip.udp().is_none(), "a fragment is not a whole UDP datagram");
            whole.extend_from_slice(ip.payload());
        }
        assert_eq!(pseudo_checksum(GW, GUEST, PROTO_UDP, &whole), 0);
        let u = Udp::parse(&whole).unwrap();
        assert_eq!(u.payload(), &payload[..]);
    }

    #[test]
    fn small_udp_is_not_fragmented() {
        let frames = route().udp_fragments(1, 1, 2, &[0; 1472]);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].len(), ETH_HDR + MTU);
    }

    #[test]
    fn arp_request_and_reply() {
        let req = arp(ARP_REQUEST, A, GUEST, [0; 6], GW);
        let eth = Ether::parse(&req).unwrap();
        assert_eq!(eth.dst(), BROADCAST_MAC);
        let a = eth.arp().unwrap();
        assert_eq!((a.op(), a.sender_mac(), a.sender_ip(), a.target_ip()), (ARP_REQUEST, A, GUEST, GW));

        let rep = arp(ARP_REPLY, B, GW, A, GUEST);
        let eth = Ether::parse(&rep).unwrap();
        assert_eq!(eth.dst(), A);
        assert_eq!(eth.arp().unwrap().target_mac(), A);
    }

    #[test]
    fn icmp_checksum_and_fields() {
        let f = route().icmp(3, ICMP_ECHO_REPLY, 0, [0x12, 0x34, 0, 9], b"ping");
        let ip = Ether::parse(&f).unwrap().ipv4().unwrap();
        let i = ip.icmp().unwrap();
        assert_eq!(checksum(i.bytes()), 0);
        assert_eq!((i.ty(), i.ident(), i.seq(), i.body()), (0, 0x1234, 9, &b"ping"[..]));
    }

    #[test]
    fn malformed_input_does_not_parse() {
        assert!(Ether::parse(&[0; 13]).is_none());
        assert!(Ipv4::parse(&[0x45; 19]).is_none());
        let mut h = [0u8; 20];
        h[0] = 0x46; // claims 24-byte header in 20 bytes
        assert!(Ipv4::parse(&h).is_none());
        h[0] = 0x65; // IPv6 version nibble
        assert!(Ipv4::parse(&h).is_none());
        let mut t = [0u8; 20];
        t[12] = 0x60; // 24-byte TCP header in 20 bytes
        assert!(Tcp::parse(&t).is_none());
        let mut opts = [0u8; 24];
        opts[12] = 0x60;
        opts[20] = 2;
        opts[21] = 9; // MSS option running past the header
        assert_eq!(Tcp::parse(&opts).unwrap().mss(), None);
    }

    #[test]
    fn ip_total_length_trims_padding_and_is_clamped() {
        let f = route().udp(1, 1, 2, b"x");
        let eth = Ether::parse(&f).unwrap();
        let ip = eth.ipv4().unwrap();
        assert_eq!(ip.payload().len(), 9);
        // A total length larger than what arrived is clamped, not trusted.
        let mut bad = f.clone();
        bad[ETH_HDR + 2] = 0x05;
        let ip = Ipv4::parse(&bad[ETH_HDR..]).unwrap();
        assert_eq!(ip.payload().len(), bad.len() - ETH_HDR - IPV4_HDR);
    }
}
