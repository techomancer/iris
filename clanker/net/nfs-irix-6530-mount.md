# IRIX 6.5.30 mounts the share only if portmap has no TCP mappings and MOUNT v3 lists AUTH_UNIX

Status: rule, 2026-11-05.

Symptom: `mount 192.168.0.1:/shared /shared` on an IRIX 6.5.30 guest
(the VPro Octane image) failed with "cannot connect to mountd ...
Connection refused"; with `-o proto=udp` it failed with "client does not
support any of server 192.168.0.1 security modes"; `-o proto=udp,vers=2`
worked. 6.5.22 guests mount with no options.

Cause: 6.5.30's mount asks the portmapper for mountd over TCP first. The
in-core portmapper answered every GETPORT with the UDP port whatever the
protocol, so the guest connected over TCP and was refused. Over UDP it then
tried NFSv3, whose MNT reply listed only AUTH_NULL; IRIX insists on a server
that lists AUTH_UNIX.

Fix (src/net/portmap.rs, src/net/nfsudp.rs): GETPORT answers 0 for a
non-UDP mapping, and the v3 MNT reply lists AUTH_UNIX and AUTH_NULL.
Credentials are still not checked. A plain `mount` now prints "Program not
registered" for its TCP attempts and mounts NFSv2 over UDP; `-o
vers=3,proto=udp` mounts NFSv3.
