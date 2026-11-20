# A test that holds a forward's port must hold it on the wildcard address, or macOS refuses the forward

Status: rule (2026-11-20).

**Symptom.** On macOS the restart tests for every NIC
(`a_restart_keeps_the_port_forwards` on bcm5701, daynaport, ioc3 eth and
seeq8003, and mec's `a_dropped_mec_frees_its_port_forwards`) panicked with
"the forward never listened"; the forward logged `failed to bind: Address
already in use`. On Linux they passed.

**Cause.** `Link::forward_on_a_held_port` keeps a SO_REUSEADDR socket bound,
not listening, on the port so no other test is handed it, and the NAT
forward binds the same port with `std::net::TcpListener::bind`, which sets
SO_REUSEADDR only. Linux lets two SO_REUSEADDR sockets share an exact
address and port while neither listens. macOS, like the BSDs, refuses a
second bind of the identical address and port unless both sockets set
SO_REUSEPORT. It does let a specific address share a port with a wildcard
bind.

**Rule.** Hold the port on `0.0.0.0` and let the forward bind `127.0.0.1`.
Both systems accept that, and neither hands the held port out as an
ephemeral port. Do not add SO_REUSEPORT to the forward listener to make
the tests pass: it would let a second emulator bind the same forward port
and share its connections.
