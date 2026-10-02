//! The host's configured DNS server. NAT forwards guest DNS queries here, so the
//! guest resolves names the way the host does — including through a VPN that
//! only permits its own resolver. See rules/irix/nat-dns-host-resolver.md.
use std::net::Ipv4Addr;

/// The host's first usable IPv4 DNS server, if one is configured.
pub fn system_dns_server() -> Option<Ipv4Addr> {
    lookup()
}

// macOS keeps /etc/resolv.conf in sync with the primary resolver (VPN included);
// on Linux it is either the real servers or a local stub such as systemd-resolved.
#[cfg(unix)]
fn lookup() -> Option<Ipv4Addr> {
    parse_resolv_conf(&std::fs::read_to_string("/etc/resolv.conf").ok()?)
}

#[cfg(any(unix, test))]
fn parse_resolv_conf(text: &str) -> Option<Ipv4Addr> {
    text.lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            if words.next()? != "nameserver" { return None; }
            words.next()?.parse::<Ipv4Addr>().ok()
        })
        .find(|ip| !ip.is_unspecified())
}

// Windows has no resolv.conf: take the DNS servers of the connected adapter with
// the lowest IPv4 metric, which is the one the host routes through (a VPN
// adapter when one is up).
#[cfg(windows)]
fn lookup() -> Option<Ipv4Addr> {
    use windows_sys::Win32::Foundation::{ERROR_BUFFER_OVERFLOW, NO_ERROR};
    use windows_sys::Win32::NetworkManagement::IpHelper::{
        GetAdaptersAddresses, GAA_FLAG_SKIP_ANYCAST, GAA_FLAG_SKIP_FRIENDLY_NAME,
        GAA_FLAG_SKIP_MULTICAST, GAA_FLAG_SKIP_UNICAST, IP_ADAPTER_ADDRESSES_LH,
    };
    use windows_sys::Win32::NetworkManagement::Ndis::IfOperStatusUp;
    use windows_sys::Win32::Networking::WinSock::{AF_INET, SOCKADDR_IN};

    let flags = GAA_FLAG_SKIP_UNICAST | GAA_FLAG_SKIP_ANYCAST
              | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_FRIENDLY_NAME;
    let mut len: u32 = 16 * 1024;
    let mut buf: Vec<u64>; // u64 storage keeps the adapter structs aligned
    let mut tries = 0;
    loop {
        buf = vec![0; (len as usize).div_ceil(8)];
        // SAFETY: `buf` holds at least `len` writable bytes.
        let rc = unsafe {
            GetAdaptersAddresses(AF_INET as u32, flags, std::ptr::null(),
                                 buf.as_mut_ptr().cast(), &mut len)
        };
        tries += 1;
        match rc {
            NO_ERROR => break,
            ERROR_BUFFER_OVERFLOW if tries < 3 => continue, // `len` now holds the size needed
            _ => return None,
        }
    }

    let mut best: Option<(u32, Ipv4Addr)> = None;
    let mut adapter = buf.as_ptr() as *const IP_ADAPTER_ADDRESSES_LH;
    // SAFETY: GetAdaptersAddresses filled `buf` with a linked list whose nodes
    // and address pointers all live inside `buf`, which outlives this walk.
    unsafe {
        while !adapter.is_null() {
            let a = &*adapter;
            if a.OperStatus == IfOperStatusUp && best.map_or(true, |(m, _)| a.Ipv4Metric < m) {
                let mut dns = a.FirstDnsServerAddress;
                while !dns.is_null() {
                    let sa = (*dns).Address.lpSockaddr;
                    if !sa.is_null() && (*sa).sa_family == AF_INET {
                        let sin = &*(sa as *const SOCKADDR_IN);
                        // S_addr is in network byte order, so its in-memory bytes are the octets.
                        let ip = Ipv4Addr::from(sin.sin_addr.S_un.S_addr.to_ne_bytes());
                        if !ip.is_unspecified() {
                            best = Some((a.Ipv4Metric, ip));
                            break;
                        }
                    }
                    dns = (*dns).Next;
                }
            }
            adapter = a.Next;
        }
    }
    best.map(|(_, ip)| ip)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolv_conf_first_ipv4_nameserver() {
        let text = "# generated\nsearch lan\nnameserver fe80::1%en0\n\
                    nameserver 10.64.0.1\nnameserver 1.1.1.1\n";
        assert_eq!(parse_resolv_conf(text), Some(Ipv4Addr::new(10, 64, 0, 1)));
    }

    #[test]
    fn resolv_conf_loopback_stub_is_used() {
        assert_eq!(parse_resolv_conf("nameserver 127.0.0.53\noptions edns0\n"),
                   Some(Ipv4Addr::new(127, 0, 0, 53)));
    }

    #[test]
    fn resolv_conf_without_ipv4_nameserver() {
        assert_eq!(parse_resolv_conf(""), None);
        assert_eq!(parse_resolv_conf("nameserver ::1\nnameserver 0.0.0.0\n#nameserver 8.8.4.4\n"), None);
        assert_eq!(parse_resolv_conf("nameserver\n"), None);
    }
}
