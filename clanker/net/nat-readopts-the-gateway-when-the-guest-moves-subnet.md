# The NAT re-adopts the gateway when the guest moves to another subnet

Status: rule, 2026-11-09 (`src/net/engine.rs` `maybe_adopt_gateway`;
`testnet.rs` `a_guest_that_moves_subnet_gets_its_new_gateway`).

**Symptom.** An IP28 disk cloned from a LAN machine (static address
192.168.86.98, gateway .86.1) booted, was re-addressed from its winterm to
192.168.0.2 with default route 192.168.0.1, and worked: telnet, NFS. About
20 minutes later the network died for good: telnet connections got no
SYN-ACK, the guest could not ping the gateway, the NFS mount hung. The
guest was healthy (idle, ec0 counting packets out). Twice in a row, both
times mid-piglit.

**Cause.** At boot the guest ARPed for .86.1; plug-and-play adoption made
that the NAT's gateway, and the first routed frame locked it. After the
re-addressing the guest still reached .86.1 through the ARP entry it had
cached at boot, but ARPs for .0.1 went unanswered (not the gateway any
more). When the cached entry expired, nothing resolved.

**Rule.** Adoption stays locked once traffic routes, unless the ARP comes
from a guest address on a different /24 than the NAT's client address:
the guest has moved, which is a new setup. The NAT then adopts the new
gateway and flushes its flows, as a live subnet change does.

**Debugging that found it.** `net status` (port-forward SYNs with no
reply), `seeq status` (frames delivered, nothing produced), `pdma status`
and `pdma chain` (the guest's TX ring moving), `arp -a` in the guest
(only .86.1 resolved), and a static ARP entry for .0.1 that made the
NAT see the ping but treat .0.1 as an outside host.
