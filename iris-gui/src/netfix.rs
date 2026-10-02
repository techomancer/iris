//! Pure logic for the "check / fix guest networking" feature.
//!
//! The emulator's NAT engine expects the IRIX guest at a fixed address derived
//! from the configured NAT subnet (gateway = network+1, guest = network+2; see
//! `src/config.rs` / `src/net/mod.rs`). When the guest's `ec0` is unset or set to a
//! different address — e.g. a static config left over from another subnet — NAT
//! traffic never flows and the GUI's NET light stays red.
//!
//! This module is the brain of the fix, kept free of any I/O so it's unit
//! testable without booting IRIX:
//! - [`ExpectedNet::from_subnet`] — what `ec0` *should* be.
//! - [`parse_ec0_inet`] — what it *is*, scraped from `ifconfig ec0` output.
//! - [`diagnose`] — compare the two.
//! - [`runtime_fix_commands`] — IRIX shell commands to correct it this session.
//!
//! The GUI injects [`PROBE_CMD`] / the fix commands over the serial console
//! (z85c30 channel B) and feeds the console text back into [`parse_ec0_inet`].
//! The live serial round-trip is validated on a real boot; this logic isn't.

// Foundation for the in-progress "check / fix guest networking" UI; the probe
// and Fix-networking actions that call these land in the next increment.
#![allow(dead_code)]

use iris::config::NatSubnet;
use std::net::Ipv4Addr;

/// Command injected at the IRIX shell to dump `ec0`'s current config. Absolute
/// path so it works regardless of the login shell's PATH.
pub const PROBE_CMD: &str = "/usr/etc/ifconfig ec0";

/// The address `ec0` should hold for NAT to work, derived from the NAT subnet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedNet {
    /// NAT-assigned guest address (subnet network + 2).
    pub ip: Ipv4Addr,
    /// NAT gateway/router address (subnet network + 1).
    pub gateway: Ipv4Addr,
    /// Subnet mask.
    pub netmask: Ipv4Addr,
}

impl ExpectedNet {
    pub fn from_subnet(s: &NatSubnet) -> Self {
        Self { ip: s.client_ip, gateway: s.gateway_ip, netmask: s.netmask }
    }

    /// IRIX `ifconfig` wants the mask as a 0x-prefixed 32-bit hex word
    /// (e.g. `0xffffff00` for a /24), not dotted-decimal.
    pub fn netmask_hex(&self) -> String {
        let o = self.netmask.octets();
        format!("0x{:02x}{:02x}{:02x}{:02x}", o[0], o[1], o[2], o[3])
    }
}

/// Result of comparing the guest's detected `ec0` address to the expected one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetDiagnosis {
    /// `ec0` is at the expected NAT address — nothing to do.
    Correct,
    /// `ec0` has an address, but not the one NAT expects (likely a static
    /// config for a different subnet). Carries the wrong address.
    Wrong(Ipv4Addr),
    /// `ec0` has no `inet` address configured at all.
    Unconfigured,
}

/// Pull the `inet` address out of `ifconfig ec0` output. IRIX prints, e.g.:
///
/// ```text
/// ec0: flags=415c43<UP,BROADCAST,RUNNING,FILTMULTI,MULTICAST,CKSUM,DRVMCAST>
///         inet 192.168.0.2 netmask 0xffffff00 broadcast 192.168.0.255
/// ```
///
/// Tolerant of the echoed command and shell prompt mixed into the captured
/// console text — it just scans for the first `inet <addr>` token.
pub fn parse_ec0_inet(console: &str) -> Option<Ipv4Addr> {
    for line in console.lines() {
        // Skip the echoed command line so a path like ".../inet..." can't match.
        let t = line.trim_start();
        if t.contains("ifconfig") {
            continue;
        }
        if let Some(rest) = t.strip_prefix("inet ") {
            if let Some(tok) = rest.split_whitespace().next() {
                if let Ok(ip) = tok.parse::<Ipv4Addr>() {
                    return Some(ip);
                }
            }
        }
    }
    None
}

/// Compare a detected `ec0` address (from [`parse_ec0_inet`]) to the expected
/// NAT address.
pub fn diagnose(detected: Option<Ipv4Addr>, expected: &ExpectedNet) -> NetDiagnosis {
    match detected {
        Some(ip) if ip == expected.ip => NetDiagnosis::Correct,
        Some(ip) => NetDiagnosis::Wrong(ip),
        None => NetDiagnosis::Unconfigured,
    }
}

/// IRIX shell commands (run as root over the serial console) to bring `ec0` up
/// at the expected address and point the default route at the NAT gateway, for
/// the current session only. Persisting across reboots is a separate step
/// (edit `/etc/config/ipaddr` + hostname → IP in `/etc/hosts`).
pub fn runtime_fix_commands(e: &ExpectedNet) -> Vec<String> {
    vec![
        format!("/usr/etc/ifconfig ec0 inet {} netmask {} up", e.ip, e.netmask_hex()),
        format!("/usr/etc/route add default {} 1", e.gateway),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subnet(net: [u8; 4]) -> NatSubnet {
        // network+1 gateway, network+2 client, /24 mask — matching parse_nat_subnet.
        NatSubnet {
            gateway_ip: Ipv4Addr::new(net[0], net[1], net[2], net[3] + 1),
            client_ip: Ipv4Addr::new(net[0], net[1], net[2], net[3] + 2),
            netmask: Ipv4Addr::new(255, 255, 255, 0),
        }
    }

    #[test]
    fn expected_from_subnet() {
        let e = ExpectedNet::from_subnet(&subnet([192, 168, 0, 0]));
        assert_eq!(e.ip, Ipv4Addr::new(192, 168, 0, 2));
        assert_eq!(e.gateway, Ipv4Addr::new(192, 168, 0, 1));
        assert_eq!(e.netmask_hex(), "0xffffff00");
    }

    #[test]
    fn parse_configured() {
        let out = "\
# /usr/etc/ifconfig ec0
ec0: flags=415c43<UP,BROADCAST,RUNNING,FILTMULTI,MULTICAST,CKSUM,DRVMCAST>
\tinet 192.168.0.2 netmask 0xffffff00 broadcast 192.168.0.255
# ";
        assert_eq!(parse_ec0_inet(out), Some(Ipv4Addr::new(192, 168, 0, 2)));
    }

    #[test]
    fn parse_unconfigured() {
        // Interface present but no inet line.
        let out = "ec0: flags=8002<BROADCAST,MULTICAST>\n# ";
        assert_eq!(parse_ec0_inet(out), None);
    }

    #[test]
    fn diagnose_three_ways() {
        let e = ExpectedNet::from_subnet(&subnet([192, 168, 0, 0]));
        assert_eq!(diagnose(Some(e.ip), &e), NetDiagnosis::Correct);
        let wrong = Ipv4Addr::new(10, 0, 0, 9);
        assert_eq!(diagnose(Some(wrong), &e), NetDiagnosis::Wrong(wrong));
        assert_eq!(diagnose(None, &e), NetDiagnosis::Unconfigured);
    }

    #[test]
    fn fix_commands_for_default_subnet() {
        let e = ExpectedNet::from_subnet(&subnet([192, 168, 0, 0]));
        let cmds = runtime_fix_commands(&e);
        assert_eq!(cmds[0], "/usr/etc/ifconfig ec0 inet 192.168.0.2 netmask 0xffffff00 up");
        assert_eq!(cmds[1], "/usr/etc/route add default 192.168.0.1 1");
    }

    #[test]
    fn non_default_subnet() {
        // A user who changed nat_subnet to 192.168.2.0/24 — guest should be .2.
        let e = ExpectedNet::from_subnet(&subnet([192, 168, 2, 0]));
        assert_eq!(e.ip, Ipv4Addr::new(192, 168, 2, 2));
        assert_eq!(parse_ec0_inet("\tinet 192.168.2.2 netmask 0xffffff00"), Some(e.ip));
        assert_eq!(diagnose(Some(e.ip), &e), NetDiagnosis::Correct);
    }
}
