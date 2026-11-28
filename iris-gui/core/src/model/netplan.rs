//! Pure planning/validation logic for the Networking tab's subnet controls.
//!
//! The Networking tab lets the user pick a **network address** and a **subnet
//! mask** that recompose into the single `nat_subnet` CIDR string the backend
//! stores (`iris::config::parse_nat_subnet`). This module is the brain behind
//! those controls, kept free of UI and (almost) free of I/O so it's unit
//! testable:
//!
//! - mask/network/broadcast math ([`netmask`], [`network_addr`], …),
//! - [`first_free_24`] — the first unused /24 in an RFC1918 block, for the
//!   "first free 192.168/172.16/10" presets,
//! - [`classify`] — sort a proposed (base, prefix) into ok / off-boundary (snap)
//!   / soft-invalid (override dialog) / hard-invalid (blocked),
//! - [`to_cidr`] — compose the snapped CIDR string to store.
//!
//! The only I/O is [`gather_host_ifaces`], a thin `if-addrs` wrapper that reads
//! the host's own interface addresses for conflict detection; the pure logic
//! takes a `&[HostIface]` slice so tests don't need real interfaces.
//!
//! Backend invariants this respects: `parse_nat_subnet` wants the **network
//! address** (host bits zero), rejects prefix `> /30`, and derives
//! **gateway = network + 1**, **Indy `ec0` = network + 2**.

// Phase 0: the logic lands first; the Networking tab wires it in Phase 1.
#![allow(dead_code)]

use std::net::Ipv4Addr;

/// Subnet-mask prefixes offered in the mask dropdown (default `/24`). `/8 /12
/// /16` are the native sizes of the three RFC1918 blocks. `Custom…` covers the
/// rest down to `/30`.
pub const MASK_PRESETS: &[u8] = &[8, 12, 16, 22, 24, 25, 26];

/// Default subnet when `nat_subnet` is unset — matches `NatSubnet::default`
/// (192.168.0.0/24, gateway .1, Indy .2).
pub const DEFAULT_BASE: Ipv4Addr = Ipv4Addr::new(192, 168, 0, 0);
pub const DEFAULT_PREFIX: u8 = 24;

/// The largest prefix the engine can represent (`parse_nat_subnet` rejects
/// `> /30`; a /30 is the minimum viable subnet — gateway + Indy + broadcast).
pub const MAX_PREFIX: u8 = 30;

// ---------------------------------------------------------------------------
// Bit math
// ---------------------------------------------------------------------------

/// 32-bit mask for a prefix length (`/24` → `0xffffff00`).
pub fn mask_bits(prefix: u8) -> u32 {
    if prefix == 0 { 0 } else { u32::MAX << (32 - prefix.min(32) as u32) }
}

/// Dotted-decimal netmask for a prefix (`/24` → `255.255.255.0`).
pub fn netmask(prefix: u8) -> Ipv4Addr {
    Ipv4Addr::from(mask_bits(prefix))
}

/// Network address: `base` with its host bits cleared.
pub fn network_addr(base: Ipv4Addr, prefix: u8) -> Ipv4Addr {
    Ipv4Addr::from(u32::from(base) & mask_bits(prefix))
}

/// Broadcast address: network with all host bits set.
pub fn broadcast_addr(base: Ipv4Addr, prefix: u8) -> Ipv4Addr {
    Ipv4Addr::from((u32::from(base) & mask_bits(prefix)) | !mask_bits(prefix))
}

/// Whether `base` is already on the subnet boundary (no host bits set).
pub fn is_network_addr(base: Ipv4Addr, prefix: u8) -> bool {
    u32::from(base) & !mask_bits(prefix) == 0
}

/// Usable host count for a prefix (excludes network + broadcast). `0` for the
/// degenerate /31, /32 that the dropdown never offers.
pub fn usable_hosts(prefix: u8) -> u64 {
    if prefix >= 31 { 0 } else { (1u64 << (32 - prefix as u32)) - 2 }
}

/// Prefix length implied by a dotted-decimal netmask (popcount).
pub fn mask_to_prefix(mask: Ipv4Addr) -> u8 {
    u32::from(mask).count_ones() as u8
}

/// Whether two CIDR blocks overlap (one contains the other's network).
pub fn cidr_overlap(a_net: Ipv4Addr, a_prefix: u8, b_net: Ipv4Addr, b_prefix: u8) -> bool {
    let p = a_prefix.min(b_prefix);
    network_addr(a_net, p) == network_addr(b_net, p)
}

// ---------------------------------------------------------------------------
// RFC1918 blocks
// ---------------------------------------------------------------------------

/// One private (RFC1918) address block, used for the network-address presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivateBlock {
    /// `192.168.0.0/16` — default; "class C" sized /24s within it.
    C,
    /// `172.16.0.0/12`.
    B,
    /// `10.0.0.0/8`.
    A,
}

impl PrivateBlock {
    pub fn base(self) -> Ipv4Addr {
        match self {
            PrivateBlock::C => Ipv4Addr::new(192, 168, 0, 0),
            PrivateBlock::B => Ipv4Addr::new(172, 16, 0, 0),
            PrivateBlock::A => Ipv4Addr::new(10, 0, 0, 0),
        }
    }
    pub fn prefix(self) -> u8 {
        match self { PrivateBlock::C => 16, PrivateBlock::B => 12, PrivateBlock::A => 8 }
    }
    /// Short label for the preset dropdown (e.g. `192.168.x`).
    pub fn label(self) -> &'static str {
        match self {
            PrivateBlock::C => "192.168.x",
            PrivateBlock::B => "172.16.x",
            PrivateBlock::A => "10.x",
        }
    }
}

/// Which RFC1918 block a network address falls in, or `None` if it's public.
pub fn block_of(net: Ipv4Addr) -> Option<PrivateBlock> {
    let o = net.octets();
    if o[0] == 10 { Some(PrivateBlock::A) }
    else if o[0] == 172 && (16..=31).contains(&o[1]) { Some(PrivateBlock::B) }
    else if o[0] == 192 && o[1] == 168 { Some(PrivateBlock::C) }
    else { None }
}

/// `true` if `net` is in private (RFC1918) space.
pub fn is_rfc1918(net: Ipv4Addr) -> bool {
    block_of(net).is_some()
}

// ---------------------------------------------------------------------------
// Host interfaces (the only I/O)
// ---------------------------------------------------------------------------

/// An IPv4 network the host already occupies, scraped from `if-addrs` and used
/// for first-free selection and conflict warnings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostIface {
    /// Interface name (e.g. `en0`) — shown in conflict messages.
    pub name: String,
    /// The interface's own address.
    pub addr: Ipv4Addr,
    /// Network address (host bits cleared).
    pub network: Ipv4Addr,
    /// Prefix length implied by the interface's netmask.
    pub prefix: u8,
}

/// Read the host's IPv4 interfaces. Loopback is skipped. Returns an empty Vec on
/// any error (conflict detection then simply finds nothing).
pub fn gather_host_ifaces() -> Vec<HostIface> {
    let Ok(ifaces) = if_addrs::get_if_addrs() else { return Vec::new() };
    ifaces
        .into_iter()
        .filter_map(|i| match i.addr {
            if_addrs::IfAddr::V4(v4) if !v4.ip.is_loopback() => {
                let prefix = mask_to_prefix(v4.netmask);
                Some(HostIface {
                    name: i.name,
                    addr: v4.ip,
                    network: network_addr(v4.ip, prefix),
                    prefix,
                })
            }
            _ => None,
        })
        .collect()
}

/// The first host interface whose network overlaps `network/prefix`, if any.
pub fn conflict<'a>(network: Ipv4Addr, prefix: u8, host: &'a [HostIface]) -> Option<&'a HostIface> {
    host.iter().find(|h| cidr_overlap(network, prefix, h.network, h.prefix))
}

/// The first network of size `prefix` in `block` that doesn't overlap any host
/// interface — the "first free 192.168/172.16/10" preset at the chosen mask. If
/// `prefix` is no longer than the block itself, the block base is the only
/// network. Falls back to the block base if everything conflicts (pathological).
pub fn first_free(block: PrivateBlock, prefix: u8, host: &[HostIface]) -> Ipv4Addr {
    if prefix <= block.prefix() {
        return block.base();
    }
    let base = u32::from(block.base());
    let step = 1u32 << (32 - prefix as u32);          // size of one network of this prefix
    let count = 1u32 << (prefix - block.prefix());     // networks of this size in the block
    let scan = count.min(1 << 16);                     // first candidate is almost always free
    for i in 0..scan {
        let cand = Ipv4Addr::from(base.wrapping_add(i.wrapping_mul(step)));
        if conflict(cand, prefix, host).is_none() {
            return cand;
        }
    }
    block.base()
}

/// First free /24 in `block` — the common case, used for the safe suggestion.
pub fn first_free_24(block: PrivateBlock, host: &[HostIface]) -> Ipv4Addr {
    first_free(block, 24, host)
}

// ---------------------------------------------------------------------------
// Derived addressing + sanity classification
// ---------------------------------------------------------------------------

/// Everything the UI shows for a chosen subnet (computed from the *snapped*
/// network, so `network` is always on-boundary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Derived {
    pub network: Ipv4Addr,
    pub prefix: u8,
    pub netmask: Ipv4Addr,
    /// Gateway / IRIS host = network + 1.
    pub gateway: Ipv4Addr,
    /// Indy `ec0` = network + 2.
    pub client: Ipv4Addr,
    pub broadcast: Ipv4Addr,
    pub first_host: Ipv4Addr,
    pub last_host: Ipv4Addr,
    pub usable_hosts: u64,
}

/// Derive addressing from a (possibly off-boundary) base + prefix. The base is
/// snapped to the network address first.
pub fn derive(base: Ipv4Addr, prefix: u8) -> Derived {
    let net = u32::from(network_addr(base, prefix));
    let bcast = net | !mask_bits(prefix);
    Derived {
        network: Ipv4Addr::from(net),
        prefix,
        netmask: netmask(prefix),
        gateway: Ipv4Addr::from(net + 1),
        client: Ipv4Addr::from(net + 2),
        broadcast: Ipv4Addr::from(bcast),
        first_host: Ipv4Addr::from(net + 1),
        last_host: Ipv4Addr::from(bcast - 1),
        usable_hosts: usable_hosts(prefix),
    }
}

/// A soft (overridable) problem with a chosen subnet, plus a safe suggestion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoftWarn {
    /// Human-readable reason (e.g. "Overlaps en0 (192.168.1.5) on this computer.").
    pub reason: String,
    /// A known-good network to offer via "Use suggested" (always a /24).
    pub suggestion_net: Ipv4Addr,
    pub suggestion_prefix: u8,
}

/// Result of sanity-checking a proposed subnet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assessment {
    /// Derived addressing, or `None` when [`hard_error`](Self::hard_error) is set.
    pub derived: Option<Derived>,
    /// Set when the engine cannot represent the subnet at all — blocks saving.
    pub hard_error: Option<String>,
    /// Set (with the typed base) when the base wasn't on the subnet boundary and
    /// was snapped to the network address — informational, not blocking.
    pub off_boundary: Option<Ipv4Addr>,
    /// Set when the subnet parses but is unwise (non-RFC1918 or host conflict) —
    /// drives the "Override Sanity Checks / Cancel" dialog.
    pub soft: Option<SoftWarn>,
}

/// Sort a proposed `base`/`prefix` into the sanity tiers, using `host` for
/// conflict detection.
pub fn classify(base: Ipv4Addr, prefix: u8, host: &[HostIface]) -> Assessment {
    if prefix == 0 || prefix > MAX_PREFIX {
        return Assessment {
            derived: None,
            hard_error: Some(t!("netplan-prefix-out-of-range", prefix = prefix, max = MAX_PREFIX)),
            off_boundary: None,
            soft: None,
        };
    }

    let derived = derive(base, prefix);
    let net = derived.network;
    let off_boundary = (!is_network_addr(base, prefix)).then_some(base);

    let soft = match block_of(net) {
        None => Some(SoftWarn {
            reason: t!("netplan-not-private", network = net.to_string()),
            suggestion_net: first_free_24(PrivateBlock::C, host),
            suggestion_prefix: 24,
        }),
        Some(block) => conflict(net, prefix, host).map(|h| SoftWarn {
            reason: t!("netplan-overlaps-host", interface = &h.name, address = h.addr.to_string()),
            suggestion_net: first_free_24(block, host),
            suggestion_prefix: 24,
        }),
    };

    Assessment { derived: Some(derived), hard_error: None, off_boundary, soft }
}

// ---------------------------------------------------------------------------
// CIDR string compose / parse
// ---------------------------------------------------------------------------

/// Compose the CIDR string to store in `nat_subnet`, snapping `base` to its
/// network address so `parse_nat_subnet` accepts it.
pub fn to_cidr(base: Ipv4Addr, prefix: u8) -> String {
    format!("{}/{}", network_addr(base, prefix), prefix)
}

/// Best-effort parse of a stored CIDR string back into (base, prefix) for
/// editing. Returns the default subnet on `None`/malformed input.
pub fn parse_cidr(s: Option<&str>) -> (Ipv4Addr, u8) {
    let Some(s) = s else { return (DEFAULT_BASE, DEFAULT_PREFIX) };
    let parsed = s.split_once('/').and_then(|(a, p)| {
        Some((a.trim().parse::<Ipv4Addr>().ok()?, p.trim().parse::<u8>().ok()?))
    });
    parsed.unwrap_or((DEFAULT_BASE, DEFAULT_PREFIX))
}

/// Group a number with thousands separators for host-count labels.
pub fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let bytes = s.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// Label for a mask-dropdown entry, e.g. `/24 = 255.255.255.0 (254 hosts)`.
pub fn mask_label(prefix: u8) -> String {
    let count = usable_hosts(prefix);
    t!("netplan-mask-option", prefix = prefix, mask = netmask(prefix).to_string(),
       count = count, hosts = commas(count))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(a: u8, b: u8, c: u8, d: u8) -> Ipv4Addr { Ipv4Addr::new(a, b, c, d) }

    fn iface(name: &str, a: u8, b: u8, c: u8, d: u8, prefix: u8) -> HostIface {
        let addr = ip(a, b, c, d);
        HostIface { name: name.into(), addr, network: network_addr(addr, prefix), prefix }
    }

    #[test]
    fn masks_and_networks() {
        assert_eq!(netmask(24), ip(255, 255, 255, 0));
        assert_eq!(netmask(25), ip(255, 255, 255, 128));
        assert_eq!(netmask(16), ip(255, 255, 0, 0));
        assert_eq!(netmask(12), ip(255, 240, 0, 0));
        assert_eq!(netmask(8), ip(255, 0, 0, 0));
        assert_eq!(netmask(30), ip(255, 255, 255, 252));
        assert_eq!(network_addr(ip(192, 168, 40, 200), 24), ip(192, 168, 40, 0));
        assert_eq!(broadcast_addr(ip(192, 168, 40, 0), 24), ip(192, 168, 40, 255));
        assert_eq!(mask_to_prefix(ip(255, 255, 255, 128)), 25);
    }

    #[test]
    fn host_counts() {
        assert_eq!(usable_hosts(24), 254);
        assert_eq!(usable_hosts(25), 126);
        assert_eq!(usable_hosts(26), 62);
        assert_eq!(usable_hosts(22), 1022);
        assert_eq!(usable_hosts(16), 65534);
        assert_eq!(usable_hosts(12), 1_048_574);
        assert_eq!(usable_hosts(8), 16_777_214);
        assert_eq!(usable_hosts(30), 2);
    }

    #[test]
    fn boundary_detection() {
        assert!(is_network_addr(ip(192, 168, 0, 0), 24));
        assert!(!is_network_addr(ip(192, 168, 0, 5), 24));
        // /25: .0 and .128 are boundaries; .65 is not (it's a host in .0/25).
        assert!(is_network_addr(ip(192, 168, 99, 0), 25));
        assert!(is_network_addr(ip(192, 168, 99, 128), 25));
        assert!(!is_network_addr(ip(192, 168, 99, 65), 25));
    }

    #[test]
    fn overlap() {
        assert!(cidr_overlap(ip(192, 168, 1, 0), 24, ip(192, 168, 1, 0), 16));
        assert!(cidr_overlap(ip(192, 168, 1, 0), 24, ip(192, 168, 1, 5), 24));
        assert!(!cidr_overlap(ip(192, 168, 1, 0), 24, ip(192, 168, 2, 0), 24));
        // a /16 host net swallows any /24 inside it.
        assert!(cidr_overlap(ip(192, 168, 40, 0), 24, ip(192, 168, 0, 0), 16));
    }

    #[test]
    fn first_free_skips_conflicts() {
        // 192.168.0.0/24 and .1.0/24 taken → first free is .2.0.
        let host = vec![iface("en0", 192, 168, 0, 5, 24), iface("en1", 192, 168, 1, 9, 24)];
        assert_eq!(first_free_24(PrivateBlock::C, &host), ip(192, 168, 2, 0));
        // No hosts → block base.
        assert_eq!(first_free_24(PrivateBlock::C, &[]), ip(192, 168, 0, 0));
        assert_eq!(first_free_24(PrivateBlock::B, &[]), ip(172, 16, 0, 0));
        assert_eq!(first_free_24(PrivateBlock::A, &[]), ip(10, 0, 0, 0));
        // A host's /16 swallows the whole 192.168 block → fall through to first
        // free /24 (still returns base via fallback only if ALL conflict; here a
        // /16 over 192.168 conflicts with every /24, so fallback = block base).
        let wide = vec![iface("vpn0", 192, 168, 0, 1, 16)];
        assert_eq!(first_free_24(PrivateBlock::C, &wide), ip(192, 168, 0, 0));
    }

    #[test]
    fn first_free_honors_prefix() {
        // prefix == block prefix or wider → block base is the only network.
        assert_eq!(first_free(PrivateBlock::C, 16, &[]), ip(192, 168, 0, 0));
        assert_eq!(first_free(PrivateBlock::C, 8, &[]), ip(192, 168, 0, 0));
        // /25: first network is the block base; if it's taken, step to .0.128.
        assert_eq!(first_free(PrivateBlock::C, 25, &[]), ip(192, 168, 0, 0));
        let host = vec![iface("en0", 192, 168, 0, 5, 25)]; // occupies 192.168.0.0/25
        assert_eq!(first_free(PrivateBlock::C, 25, &host), ip(192, 168, 0, 128));
    }

    #[test]
    fn rfc1918_membership() {
        assert_eq!(block_of(ip(192, 168, 0, 0)), Some(PrivateBlock::C));
        assert_eq!(block_of(ip(172, 16, 0, 0)), Some(PrivateBlock::B));
        assert_eq!(block_of(ip(172, 31, 255, 0)), Some(PrivateBlock::B));
        assert_eq!(block_of(ip(172, 32, 0, 0)), None);
        assert_eq!(block_of(ip(10, 1, 2, 0)), Some(PrivateBlock::A));
        assert_eq!(block_of(ip(8, 8, 8, 0)), None);
    }

    #[test]
    fn derive_default() {
        let d = derive(ip(192, 168, 0, 0), 24);
        assert_eq!(d.gateway, ip(192, 168, 0, 1));
        assert_eq!(d.client, ip(192, 168, 0, 2));
        assert_eq!(d.broadcast, ip(192, 168, 0, 255));
        assert_eq!(d.first_host, ip(192, 168, 0, 1));
        assert_eq!(d.last_host, ip(192, 168, 0, 254));
        assert_eq!(d.usable_hosts, 254);
    }

    #[test]
    fn classify_ok() {
        let a = classify(ip(192, 168, 0, 0), 24, &[]);
        assert!(a.hard_error.is_none());
        assert!(a.off_boundary.is_none());
        assert!(a.soft.is_none());
        assert_eq!(a.derived.unwrap().client, ip(192, 168, 0, 2));
    }

    #[test]
    fn classify_user_slash25_example() {
        // The user's "192.168.99.65/25": a host, not a network. Snaps to
        // 192.168.99.0/25, gateway .1, Indy .2, mask 255.255.255.128.
        let a = classify(ip(192, 168, 99, 65), 25, &[]);
        assert_eq!(a.off_boundary, Some(ip(192, 168, 99, 65)));
        assert!(a.soft.is_none());
        let d = a.derived.unwrap();
        assert_eq!(d.network, ip(192, 168, 99, 0));
        assert_eq!(d.gateway, ip(192, 168, 99, 1));
        assert_eq!(d.client, ip(192, 168, 99, 2));
        assert_eq!(d.netmask, ip(255, 255, 255, 128));
    }

    #[test]
    fn classify_widening_mask_snaps() {
        // first-free hands 192.168.40.0; switching to /16 is off-boundary →
        // snaps to 192.168.0.0/16.
        let a = classify(ip(192, 168, 40, 0), 16, &[]);
        assert_eq!(a.off_boundary, Some(ip(192, 168, 40, 0)));
        assert_eq!(a.derived.unwrap().network, ip(192, 168, 0, 0));
    }

    #[test]
    fn classify_non_rfc1918_is_soft() {
        let a = classify(ip(8, 8, 8, 0), 24, &[]);
        let soft = a.soft.expect("public range should warn");
        assert!(soft.reason.contains("RFC1918") || soft.reason.contains("private"));
        assert_eq!(soft.suggestion_net, ip(192, 168, 0, 0));
        assert!(a.derived.is_some()); // still representable, just unwise
    }

    #[test]
    fn classify_conflict_is_soft_with_suggestion() {
        let host = vec![iface("en0", 192, 168, 0, 5, 24)];
        let a = classify(ip(192, 168, 0, 0), 24, &host);
        let soft = a.soft.expect("conflict should warn");
        assert!(soft.reason.contains("en0"));
        assert_eq!(soft.suggestion_net, ip(192, 168, 1, 0)); // first free past the conflict
    }

    #[test]
    fn classify_hard_prefix() {
        assert!(classify(ip(192, 168, 0, 0), 31, &[]).hard_error.is_some());
        assert!(classify(ip(192, 168, 0, 0), 0, &[]).hard_error.is_some());
        assert!(classify(ip(192, 168, 0, 0), 30, &[]).hard_error.is_none());
    }

    #[test]
    fn cidr_roundtrip() {
        assert_eq!(to_cidr(ip(192, 168, 99, 65), 25), "192.168.99.0/25");
        assert_eq!(to_cidr(ip(192, 168, 0, 0), 24), "192.168.0.0/24");
        assert_eq!(parse_cidr(Some("10.0.0.0/8")), (ip(10, 0, 0, 0), 8));
        assert_eq!(parse_cidr(None), (DEFAULT_BASE, DEFAULT_PREFIX));
        assert_eq!(parse_cidr(Some("garbage")), (DEFAULT_BASE, DEFAULT_PREFIX));
    }

    #[test]
    fn commas_format() {
        assert_eq!(commas(254), "254");
        assert_eq!(commas(1022), "1,022");
        assert_eq!(commas(16_777_214), "16,777,214");
    }
}
