// XDMCP application-layer gateway (reverse proxy into the guest's `xdm`).
//
// XDMCP (X Display Manager Control Protocol, RFC-less but specified in the X11
// docs) lets an X *server* (the display) ask a remote host's display manager for
// a login session. It runs over UDP/177, version 1. Both IRIX 5.3 and 6.5 speak
// XDMCP 1.0, so there's no version skew to handle.
//
// Why an ALG (like the FTP PASV gateway in net.rs): the `Request` packet the X
// server sends carries the *connection addresses* where the manager should open
// the X11 session back to. Behind the software NAT the X server's real address
// isn't reachable from the guest as-is (e.g. it's `127.0.0.1` for an X server on
// the IRIS host), so we rewrite those addresses to the NAT gateway and proxy the
// guest's resulting X11 TCP connection (`gateway:6000+display`) out to the X
// server. This module is the pure, testable half: parse a packet, rewrite the
// IPv4 connection-addresses in a `Request`, and report the display number + the
// X server's original address(es) so net.rs can wire up the session proxy.
//
// Rewriting IPv4→IPv4 is length-preserving, so (unlike the FTP ASCII rewrite)
// the packet length is unchanged and only the UDP checksum must be recomputed.
//
// Authorization note: `MIT-MAGIC-COOKIE-1` and "no auth" are address-independent,
// so rewriting is safe. `XDM-AUTHORIZATION-1` cryptographically binds the
// addresses; rewriting would break it. Callers can use `request_auth_names()` to
// detect that case and decline rather than silently corrupt the session.

use std::net::Ipv4Addr;

/// XDMCP version this gateway understands (the only deployed version).
const XDMCP_VERSION: u16 = 1;

// Opcodes (subset we care about). See the XDMCP spec.
pub const OP_BROADCAST_QUERY: u16 = 1;
pub const OP_QUERY: u16 = 2;
pub const OP_INDIRECT_QUERY: u16 = 3;
pub const OP_WILLING: u16 = 5;
pub const OP_REQUEST: u16 = 7;
pub const OP_ACCEPT: u16 = 8;
pub const OP_MANAGE: u16 = 10;

/// Connection family for an IPv4 ("internet") address in `connection-types`.
const FAMILY_INTERNET: u16 = 0;

/// Big-endian CARD16 read with bounds checking.
fn rd_u16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*b.get(o)?, *b.get(o + 1)?]))
}

/// Peek the XDMCP opcode of a packet, validating the version header.
/// Returns `None` if the buffer is too short or the version isn't 1.
pub fn opcode(pkt: &[u8]) -> Option<u16> {
    if rd_u16(pkt, 0)? != XDMCP_VERSION {
        return None;
    }
    rd_u16(pkt, 2)
}

/// Result of rewriting a `Request` packet's connection-addresses.
#[derive(Debug, Clone)]
pub struct RequestRewrite {
    /// The packet with every IPv4 connection-address replaced by the new address.
    /// Same length as the input (IPv4→IPv4 is length-preserving).
    pub packet: Vec<u8>,
    /// The X11 display number the session targets (port = 6000 + display_number).
    pub display_number: u16,
    /// The X server's original IPv4 connection-address(es), in order — where the
    /// X11 session must ultimately be proxied to.
    pub client_ipv4: Vec<Ipv4Addr>,
}

/// Walk a `Request` body and, for every IPv4 `connection-address` (family
/// `internet`, 4 bytes), record the original address and overwrite it with
/// `new_addr`. Returns the rewritten packet plus the display number and the
/// original IPv4 addresses, or `None` if the packet isn't a well-formed
/// version-1 `Request`.
///
/// `Request` body layout (all multi-byte fields big-endian):
///   display-number       CARD16
///   connection-types     ARRAY16        (CARD8 count, then count×CARD16)
///   connection-addresses ARRAYofARRAY8  (CARD8 count, then count×ARRAY8)
///   authentication-name  ARRAY8         (CARD16 len, then bytes)
///   authentication-data  ARRAY8
///   authorization-names  ARRAYofARRAY8
///   manufacturer-display-ID ARRAY8
/// where ARRAY8 = CARD16 length + that many bytes.
pub fn rewrite_request_ipv4(pkt: &[u8], new_addr: Ipv4Addr) -> Option<RequestRewrite> {
    if rd_u16(pkt, 0)? != XDMCP_VERSION || rd_u16(pkt, 2)? != OP_REQUEST {
        return None;
    }
    let length = rd_u16(pkt, 4)? as usize;
    if pkt.len() < 6 + length {
        return None; // truncated relative to the declared body length
    }

    // Edit in a full copy; IPv4→IPv4 keeps every offset stable.
    let mut out = pkt.to_vec();
    let mut o = 6usize; // start of the body, right after the 6-byte header

    // display-number
    let display_number = rd_u16(&out, o)?;
    o += 2;

    // connection-types: CARD8 count, then count × CARD16
    let n_types = *out.get(o)? as usize;
    o += 1;
    let mut types = Vec::with_capacity(n_types);
    for _ in 0..n_types {
        types.push(rd_u16(&out, o)?);
        o += 2;
    }

    // connection-addresses: CARD8 count, then count × ARRAY8
    let n_addrs = *out.get(o)? as usize;
    o += 1;
    let mut client_ipv4 = Vec::new();
    for i in 0..n_addrs {
        let alen = rd_u16(&out, o)? as usize; // ARRAY8 length (CARD16)
        o += 2;
        let astart = o;
        if out.len() < astart + alen {
            return None; // address runs past the buffer
        }
        // A 4-byte address whose paired connection-type is `internet` is IPv4.
        // (If the counts disagree, treat a 4-byte address as IPv4 anyway.)
        let is_inet = types.get(i).copied().map(|t| t == FAMILY_INTERNET).unwrap_or(true);
        if is_inet && alen == 4 {
            client_ipv4.push(Ipv4Addr::new(
                out[astart], out[astart + 1], out[astart + 2], out[astart + 3],
            ));
            out[astart..astart + 4].copy_from_slice(&new_addr.octets());
        }
        o += alen;
    }

    Some(RequestRewrite { packet: out, display_number, client_ipv4 })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid XDMCP Request with the given IPv4 connection
    /// address and display number. Returns the full on-wire packet.
    fn build_request(display: u16, addr: Ipv4Addr) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&display.to_be_bytes()); // display-number
        // connection-types: 1 entry = internet(0)
        body.push(1);
        body.extend_from_slice(&FAMILY_INTERNET.to_be_bytes());
        // connection-addresses: 1 entry = ARRAY8 of the 4 address bytes
        body.push(1);
        body.extend_from_slice(&4u16.to_be_bytes());
        body.extend_from_slice(&addr.octets());
        // authentication-name: empty ARRAY8
        body.extend_from_slice(&0u16.to_be_bytes());
        // authentication-data: empty ARRAY8
        body.extend_from_slice(&0u16.to_be_bytes());
        // authorization-names: empty ARRAYofARRAY8 (count 0)
        body.push(0);
        // manufacturer-display-ID: empty ARRAY8
        body.extend_from_slice(&0u16.to_be_bytes());

        let mut pkt = Vec::new();
        pkt.extend_from_slice(&XDMCP_VERSION.to_be_bytes());
        pkt.extend_from_slice(&OP_REQUEST.to_be_bytes());
        pkt.extend_from_slice(&(body.len() as u16).to_be_bytes());
        pkt.extend_from_slice(&body);
        pkt
    }

    #[test]
    fn rewrites_ipv4_connection_address() {
        let orig = Ipv4Addr::new(192, 168, 1, 50);
        let gw = Ipv4Addr::new(10, 0, 2, 1);
        let pkt = build_request(7, orig);
        let before_len = pkt.len();

        let r = rewrite_request_ipv4(&pkt, gw).expect("valid request");
        assert_eq!(r.display_number, 7);
        assert_eq!(r.client_ipv4, vec![orig]);
        // Length preserved (IPv4 → IPv4).
        assert_eq!(r.packet.len(), before_len);
        // The declared body length field is unchanged.
        assert_eq!(rd_u16(&r.packet, 4), rd_u16(&pkt, 4));
        // The address bytes now read as the gateway, and re-parsing confirms it.
        let again = rewrite_request_ipv4(&r.packet, gw).unwrap();
        assert_eq!(again.client_ipv4, vec![gw]);
    }

    #[test]
    fn opcode_peek_and_version_guard() {
        let pkt = build_request(0, Ipv4Addr::LOCALHOST);
        assert_eq!(opcode(&pkt), Some(OP_REQUEST));
        // Wrong version → None.
        let mut bad = pkt.clone();
        bad[1] = 9;
        assert_eq!(opcode(&bad), None);
        assert!(rewrite_request_ipv4(&bad, Ipv4Addr::LOCALHOST).is_none());
    }

    #[test]
    fn non_request_and_truncated_return_none() {
        // A Query packet (opcode 2) isn't a Request.
        let mut q = build_request(0, Ipv4Addr::LOCALHOST);
        q[2..4].copy_from_slice(&OP_QUERY.to_be_bytes());
        assert!(rewrite_request_ipv4(&q, Ipv4Addr::LOCALHOST).is_none());
        // Truncated body (declared length longer than the buffer).
        let pkt = build_request(0, Ipv4Addr::LOCALHOST);
        assert!(rewrite_request_ipv4(&pkt[..pkt.len() - 3], Ipv4Addr::LOCALHOST).is_none());
    }

    #[test]
    fn handles_two_addresses_rewriting_only_ipv4() {
        // Build a Request with two connection types/addresses: one IPv4 (internet),
        // one non-internet 6-byte address that must be left untouched.
        let mut body = Vec::new();
        body.extend_from_slice(&3u16.to_be_bytes()); // display 3
        body.push(2); // 2 connection-types
        body.extend_from_slice(&FAMILY_INTERNET.to_be_bytes()); // internet
        body.extend_from_slice(&6u16.to_be_bytes()); // some other family
        body.push(2); // 2 connection-addresses
        body.extend_from_slice(&4u16.to_be_bytes());
        body.extend_from_slice(&Ipv4Addr::new(172, 16, 0, 9).octets());
        body.extend_from_slice(&6u16.to_be_bytes());
        body.extend_from_slice(&[1, 2, 3, 4, 5, 6]); // 6-byte non-IPv4 addr
        body.extend_from_slice(&0u16.to_be_bytes()); // auth-name
        body.extend_from_slice(&0u16.to_be_bytes()); // auth-data
        body.push(0); // authorization-names
        body.extend_from_slice(&0u16.to_be_bytes()); // mfg id

        let mut pkt = Vec::new();
        pkt.extend_from_slice(&XDMCP_VERSION.to_be_bytes());
        pkt.extend_from_slice(&OP_REQUEST.to_be_bytes());
        pkt.extend_from_slice(&(body.len() as u16).to_be_bytes());
        pkt.extend_from_slice(&body);

        let gw = Ipv4Addr::new(10, 0, 2, 1);
        let r = rewrite_request_ipv4(&pkt, gw).unwrap();
        assert_eq!(r.display_number, 3);
        assert_eq!(r.client_ipv4, vec![Ipv4Addr::new(172, 16, 0, 9)]);
        // The non-IPv4 6-byte address is byte-for-byte unchanged.
        let tail = &r.packet[r.packet.len() - 1 - 2 - 1 - 2 - 2 - 6..];
        assert!(tail.windows(6).any(|w| w == [1, 2, 3, 4, 5, 6]));
    }
}
