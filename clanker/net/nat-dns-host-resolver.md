# NAT DNS follows the host's DNS server

Symptom that motivated it: with a VPN such as Mullvad connected, guest TCP to
public IPs worked but hostnames did not resolve. NAT forwarded every guest UDP
port-53 query to a hard-coded 8.8.8.8, and Mullvad blocks DNS to anything but
its own resolver.

`src/net/host_dns.rs` finds the host's DNS server; `NatEngine::dns_upstream` uses it
unless `GatewayConfig::dns_upstream` is set, re-reading every
`HOST_DNS_REFRESH` (5 s) so VPN changes apply without a restart. Fallback is
8.8.8.8.

- **macOS/Linux:** first IPv4 `nameserver` in `/etc/resolv.conf`. macOS keeps
  that file in sync with the primary resolver, VPN included (check with
  `scutil --dns`). It is often a loopback address (a VPN's local resolver,
  systemd-resolved's 127.0.0.53) — fine, the query is sent from the host.
- **Windows:** `GetAdaptersAddresses`, IPv4 DNS servers of the up adapter with
  the lowest `Ipv4Metric` (a VPN adapter when connected).
- **DHCP must not advertise the host's DNS server.** A loopback address given to
  IRIX points at the guest's own loopback and never reaches the NAT. DHCP keeps
  advertising 8.8.8.8; NAT intercepts port-53 UDP to any destination anyway.
- IPv6-only host DNS is ignored (the forwarding socket is IPv4).
- Split DNS (per-domain resolvers, e.g. corporate VPNs) is not reflected in
  resolv.conf, so those domains won't resolve. Full-tunnel VPNs are fine.
- DNS over TCP is not forwarded this way; it follows the normal TCP NAT path to
  whatever address the guest used.

## The reply must come from the address the guest asked

DHCP advertises `DNS_FALLBACK` (8.8.8.8) and NAT intercepts port-53 UDP to any
destination, so the guest's query never reaches the address on the envelope.
The reply therefore has to be *addressed back from that same address*: a
resolver that checks — NetBSD's `res_send` does, IRIX's does not — discards an
answer arriving from somewhere it never queried.

Sending it from the gateway instead is invisible on IRIX and fatal on NetBSD,
where every lookup fails while ping and TCP to the same host work:

    ftp: Can't LOOKUP `cdn.NetBSD.org:http': No address associated with hostname

ICMP already answered as the remote host rather than as the gateway; DNS was
the outlier.
