# The NAT network stack: analysis and redesign plan

Status: analysis and steps 1-5 done 2026-10-14 (results in §2, what is
left in §8).
Scope: `src/net/` (NAT backend and its services) and the seeq8003 pump that
feeds it. PCAP bridging (`net_pcap.rs`) is out of scope except where it shares
the backend trait.

## 1. What is there today

```
 guest RAM ⇄ HPC3 DMA ⇄ seeq-enet thread ──tx ring (256)──▶ seeq-nat thread (NatEngine)
                                        ◀─rx ring (256)───        │
                                                                   ├─ host TcpStream per TCP flow
                                                                   ├─ host UdpSocket per UDP flow
                                                                   ├─ ICMP socket per (dst, ident)
                                                                   ├─ port-forward listeners
                                                                   └─ in-core services: ARP, BOOTP/DHCP,
                                                                      DNS relay, NFS+mountd, portmap
                                                                      (UDP+TCP), TFTP, time (UDP+TCP),
                                                                      NTP, FTP ALG, XDMCP ALG
```

`src/net/mod.rs` is 3,100 lines. One `NatEngine` owns everything above. Frames
are `Vec<u8>`; every handler parses with `r16`/`r32` at hand-computed offsets
and builds replies with `udp_packet`/`tcp_segment_win`/`ip_frame`, each
computing its own checksum. Services are wired in by matching ports inside
`handle_udp`/`handle_tcp`, ahead of the generic NAT path.

Both threads loop on a condvar with a **1 ms timeout**. The NAT loop runs every
poll function (`poll_udp`, `poll_tcp`, `poll_icmp`, both forward listeners,
`poll_tftp`) and rebuilds the status snapshot (`update_snapshot`, three fresh
`Vec`s) on every pass, about 1000 times a second whether or not anything moved.

## 2. Measurements (2026-10-14, Indy R5000 + XZ, IRIX 6.5, `test/net`)

Before and after, same guest, same `nettest`. Host clock for the forwarded
rows; guest clock (approximate: lost ticks) for the rest. Every transfer is
checked byte for byte.

| case | before | after |
|---|---|---|
| host → guest, TCP through a port forward | 1.03-1.07 MB/s | 5.4-6.5 MB/s |
| guest → host, TCP through a port forward | 2.4-3.0 MB/s | 6.7-7.4 MB/s |
| TCP echo through a forward (full duplex) | 0.52 MB/s | 3.3 MB/s |
| guest TCP sink / source / echo via the gateway | 1.76 / 1.08 / 0.45 MB/s | 5.5 / 5.2 / 2.1 MB/s |
| FTP put / get to a LAN NAS (EPSV) | 1.19 / 0.99 MB/s | 6.8 / 7.2 MB/s |
| UDP echo 20 × 30000 (fragmented) round trip | 48 ms | 5 ms |
| UDP echo 20 × 8000 through a forward, round trip | 13 ms | 1 ms |
| ping the gateway during a connect to a black hole | 3983 ms | 2 ms |
| idle CPU, seeq-nat + seeq-enet threads | 2.7% + 0.7% | 0.1% + 0.9% |

The first baseline (below) used a `nettest` that checked each byte with a
division, which on the emulated CPU capped everything near 1-2 MB/s; the table
above re-measured the old binary with the fixed tool. The original table is
kept for the record:

| case | result |
|---|---|
| guest → gateway (host loopback), TCP sink 4 MB | 0.90 MB/s |
| gateway → guest, TCP source 4 MB | 0.99 MB/s |
| TCP echo 1 MB (full duplex) | 0.41 MB/s |
| UDP echo 200 × 1400 | 0 lost, rtt avg 2 ms, max 28 ms |
| UDP echo 20 × 30000 (fragmented both ways) | 0 lost, rtt 49 ms |
| host → guest port forward, TCP sink 4 MB | 1.06 MB/s |
| guest → host port forward, TCP source 4 MB | 1.91 MB/s |
| FTP (EPSV) to the LAN NAS, 1 MB put / get | 0.79 / 0.99 MB/s |
| HTTP GET external, telnet to a LAN Octane | work |
| ping gateway while one connect() hangs on a black hole | **max 3975 ms** |

## 3. Findings

Status after this work: every item in 3.1 and 3.2 is fixed except where
marked; 3.3 is the refactor (§5).

### 3.1 Reliability

1. **Outbound TCP connect blocks the NAT thread.** `handle_tcp` calls
   `TcpStream::connect_timeout(.., 5 s)` inline. Every guest connection to a
   slow or black-holed address freezes all guest networking (pings, NFS, other
   TCP flows) for up to 5 s. Measured: a 4 s ping during one such connect.
2. **DNS forwarding blocks the NAT thread** for up to 2 s per query
   (`forward_dns`: bind, `send_to`, blocking `recv_from` with a 2 s timeout).
   Every lookup stalls the stack for the host resolver's round trip.
3. **UDP NAT presents every reply as coming from the original destination.**
   `poll_udp` ignores the real source of each datagram. Protocols that answer
   from another port (TFTP, some RPC, SIP) see the wrong TID. One host socket
   per (dst, dport, sport) also means a guest UDP endpoint talking to N peers
   holds N host sockets with N different host ports.
4. **No MSS option in our SYN/SYN-ACK.** For off-subnet peers IRIX falls back
   to its default MSS (512/536) instead of 1460, so uploads through the NAT
   use small segments.
5. **Two ways to queue a frame toward the guest.** `enqueue_rx` pushes
   immediately and drops on a full ring; `deferred_rx` waits a loop pass and
   preserves order. Which one a reply uses is historical (see the FIXME on the
   TX-completion interrupt race), and drops on the immediate path are silent.
6. **Fake TCP endpoints.** Portmap-over-TCP and RFC 868 time-over-TCP are
   stateless one-shot segment sequences with fixed ISNs. They work for the one
   client each was written against and nothing else.
7. Fixed ISN (`0x4000_0000`) for every outbound connection; pending port
   forward SYNs never expire.
8. Found while porting: an established connection idle for 5 minutes was
   dropped silently (an idle telnet session got reset on its next keystroke);
   portmap was intercepted for *any* destination, so a guest mounting a real
   NFS server on the LAN got our ports back; the forward SYN was never
   retransmitted.

### 3.2 Performance

1. **One RX frame per enet wake.** `seeq-enet` delivers at most one frame per
   loop pass, then waits on `rx_wake` with a 1 ms timeout. The NAT notifies
   per push, but a notify that lands while the enet thread is busy is lost, so
   a burst drains at about one frame per millisecond: about 1.4 MB/s at best,
   which is what host → guest measures.
2. 1 ms polling on both threads plus the per-pass snapshot rebuild: idle CPU
   and latency floor (each reply waits for the next pass).
3. Per-frame allocations: `tcp_segment_win` → `Vec`, `ip_frame` → another
   `Vec`, plus `buf[..n].to_vec()` for the retransmit copy.

### 3.3 Structure

- No packet types: offsets are spelled out at every use (`frame[26..30]`,
  `r16(tcp, 14)`), checksums are recomputed by hand in five places.
- One struct holds NAT state for three protocols, eight services, two ALGs,
  forward listeners, fragment reassembly and GUI telemetry.
- Services are written at packet level, each re-implementing addressing and
  reply construction; TCP services each fake their own handshake.
- The TCP state is a set of booleans (`fin_wait`, `server_fin`, `fin_sent`,
  `shut_pending`) instead of a state machine, and the guest-facing TCP logic
  (sequence, window, retransmit, FIN) is welded to `TcpStream`.

## 4. The irix7 fork

The fork's only network change (`docs/irix7-fork-review.md` §2.5) is the
event-driven NAT loop: one `poll(2)` over every host socket plus a wake pipe
for guest TX, the status snapshot built only when needed, and its companion
note on the same 1 ms loop in seeq-enet. The idea is right and is now in
(`engine.rs` + `wake.rs`), with the review's correction: the timeout comes from
the earliest TCP timer, not a fixed 200 ms tick. Its own effect is idle CPU
(3.7% → 0.1% of a core). It was never the throughput limit: the enet
one-frame-per-wake drain, the missing MSS option and the blocking connect/DNS
were (§3), and the fork touched none of them. Nothing else in the fork
concerns networking.

## 5. Design (as built)

```
src/net/
  packet.rs      Ether/Arp/Ipv4/Icmp/Udp/Tcp views (validated once, accessors
                 for every field) and Ipv4Route builders that write a whole
                 frame and compute lengths and checksums; nothing else does
  engine.rs      NatEngine: the poll(2) loop, ARP, IP dispatch, the queue of
                 frames toward the guest (never drops; replies to guest frames
                 wait one pass, see the FIXME there)
  wake.rs        Wake: NIC <-> backend signal, condvar + self-pipe, never lost
  tcp.rs         TcpConn: guest-facing TCP (states, ISN, MSS, go-back-N with
                 backoff, zero-window probe, in-order receive into a bounded
                 buffer) over a Backend: HostConn (non-blocking connect) or
                 ServiceConn (a TcpService: time, portmap)
  nat_tcp.rs     TCP flows keyed by what the guest sees; FTP passive ALG
  nat_udp.rs     UDP: one host socket per guest endpoint, replies from their
                 real source, aliases for gateway/loopback and the DNS relay
  nat_icmp.rs    echo: local at the gateway, relayed elsewhere
  services.rs    UdpService (time, NTP, portmap, NFS+mountd, TFTP) and BOOTP
  rsh.rs         TcpService: read-only rsh (cat, ls, echo, pwd) on one directory;
                 streams files with `pull`, opens rsh's stderr connection back
                 to the guest with `side_connect`
  fwd.rs         inbound forwards (+ XDMCP ALG)
  frag.rs, portmap.rs, nfs_pcap.rs (the NFS server as a LAN host in PCAP mode)
  testnet.rs     directed tests: a scripted guest against a real engine
```

- **Packets.** A frame stays one contiguous buffer (it is what the ring
  carries), wrapped in typed views. Parsing validates once; builders write
  headers in place and compute checksums in one place.
- **Flows.** `TcpFlow` = guest-facing endpoint + a backend: `HostStream`
  (outbound NAT, inbound forward) or `Internal(Box<dyn TcpService>)`. Portmap
  and time over TCP become ordinary services and get real TCP for free.
- **Services** never see packets: they get (peer, bytes) and return bytes.
- **Non-blocking everything.** Outbound connect becomes a pending flow (SYN
  answered when the socket connects, RST when it fails); DNS becomes a UDP flow.
- **One outbound queue** toward the guest with one policy.
- **Event-driven loop** (after measurement): `poll(2)` on host sockets plus a
  wake fd, timeout from the earliest deadline; enet drains its ring each wake.

## 6. Order of work (done)

Each step kept `cargo test` and `test/net` green and was measured against §2.

1. `packet.rs` and port the existing code onto it (no behaviour change).
2. Split `mod.rs` along §5; directed TCP/UDP tests at engine level (a test
   guest that speaks TCP to the engine against real host sockets).
3. TCP endpoint extracted from `NatTcpEntry`; host-stream and internal
   backends; portmap/time TCP as services.
4. Reliability: non-blocking connect, async DNS, MSS option, UDP mapping by
   guest endpoint, one outbound queue.
5. Performance: enet drain, event-driven NAT loop, lazy snapshot, fewer
   allocations. Measure each.

## 7. Tests

- `cargo test --lib net::` — unit and engine-level tests.
- `test/net/nettest.c` — one portable client/server (guest and host).
- `test/net/run.sh` (guest) — local TCP/UDP through the gateway, external
  HTTP/ping, FTP to a LAN server, telnet to a LAN host, and the stall probe.
- `test/net/host.sh` (host) — the same through inbound port forwards.
- `test/tftp/` — TFTP and PROM network boot.
- `rsh_*` in `testnet.rs` — the rsh server end to end (request, stdout, the
  stderr connection from a reserved port); `net::rsh::tests` — the protocol
  and the sandbox.

## 8. Left to do

- seeq-enet still ticks at 1 ms (0.9% of a core idle): HPC3 kicks it only
  when a DMA channel goes inactive → active, so a descriptor appended to a
  running TX chain is found only by the tick. Kick on append, then let it
  sleep on its `Wake`.
- The TX-completion/RX interrupt race behind the "replies wait one pass"
  rule (engine.rs FIXME) belongs in seeq8003's interrupt model, not the NAT.
- Windows has no poll(2) path; the NAT thread ticks at 1 ms there.
- Per-segment allocations on the TCP send path (a `Vec` per read and per
  frame); not measurable yet next to the guest's own cost.
- IRIX `ftp` has no active-mode toggle, so outbound PORT (which would need an
  ALG of its own) is untested and unsupported.
