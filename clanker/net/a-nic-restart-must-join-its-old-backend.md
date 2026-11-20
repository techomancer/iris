# A NIC restart must join its old backend before the new one binds

**Status: rule (2026-11-12, `8e3c00dc`).**

**Symptom:** after a machine reset the configured port forwards were gone
until iris restarted. A dropped O2 followed by a new O2 in the same process
lost its forwards the same way.

**Cause:** a reset stops every device and starts it again. The NAT engine
sleeps in its poller for up to 100 ms before it looks at its running flag,
so the old backend still held the port-forward listeners when the new one
tried to bind them ("Address already in use"). The old engine then exited
and took the only listeners with it. The MEC never stopped its backend at
all.

**Rule:** every NIC runs its backend through `net::link::Session`
(`src/net/link.rs`). `stop` clears the flag and wakes the backend; the next
`start` drops the previous `Running`, which joins the thread and frees its
sockets, before the fresh link spawns. A NIC with no start/stop of its own
(the MEC) holds the `Running` from `Link::spawn`, so dropping the device
frees the sockets. Do not give a new NIC its own start/stop/join code, and do
not reuse one NAT engine across a stop: each start gets a fresh backend and
fresh rings, so a frame queued while stopped is discarded and the flow tables
are cleared, the same on every NIC.

**Tests:** one restart test per NIC (SEEQ, DaynaPort, IOC3, BCM5701) restarts
a device with a port forward and expects the forward to listen again.
