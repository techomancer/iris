# NAT TCP: a slow host reader must not lose the guest's data

**Symptom:** a big transfer *out of* the guest through the NAT (a port-forward
or an outbound connection) corrupts or truncates, but small ones are fine.
`ssh guest 'tar cf - big-dir' > out.tar` died after ~50 MB with
`ssh_dispatch_run_fatal: ... message authentication code incorrect`: bytes
went missing from the TCP stream and ssh's MAC caught it.

**Cause:** `handle_tcp` wrote the guest's payload straight into the host
socket with `write_all` and ACKed the segment regardless. The socket is
non-blocking, so once the host read more slowly than the guest sent (ssh
writing to disk, a slow consumer), the socket buffer filled, `write_all`
failed part-way with `WouldBlock`, the error was dropped -- and the guest had
already been told those bytes arrived, so it never resent them. A segment
arriving beyond a gap was also accepted and skipped the gap.

**Fix:** each NAT TCP entry has a receive buffer, `to_host` (64 KB, the
largest window without window scaling). Only bytes that continue the stream
exactly at `client_seq` are taken, and only as many as fit; the rest (and
anything beyond a gap) is left for the guest to retransmit. What is taken is
ACKed and written to the host socket as fast as it accepts it (`flush_to_host`,
also from `poll_tcp`). Every segment we send advertises the room left as the
window (`tcp_segment_win`), and `poll_tcp` sends a window update when the host
catches up. The guest's FIN is taken only when every byte before it has been,
and shuts the host write side only after `to_host` drains.

**Half-close, found while testing:** when the host closed first, the entry was
deleted as soon as our FIN went to the guest, so whatever the guest still
sent (an echo, the rest of a reply) hit "no entry" and was reset. The entry now
stays (`fin_sent`) until the guest's FIN has come too.

**How to test:** a clone with inetd's TCP `chargen` (19) and `echo` (7)
forwarded, and a host reader slower than the guest sends. Before: 37 corrupt
stretches in 10 MB of chargen read at 0.1 MB/s, and echo returned all but the
last ~68 KB. After: both clean, and 40 MB over ssh to a reader at 0.25 MB/s
matches the guest's `sum -r`. Tests: `net::testnet::slow_host_reader_loses_nothing`
and `tcp_out_of_order_guest_data_is_not_skipped`.

**Since 2026-10-14** the logic lives in `src/net/tcp.rs` (`TcpConn`): the
buffer is `to_backend`, and the same rules hold for every backend (host
socket or in-core service).
