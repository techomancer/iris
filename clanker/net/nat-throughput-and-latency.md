# NAT throughput and latency: what actually limited it

Measured 2026-10-14 with `test/net` (Indy R5000 + XZ, IRIX 6.5). Design and
numbers: `clanker/plans/net-stack.md`.

## Measure with a guest program that is not the bottleneck

The first `nettest` checked every byte with `i % 251` and summed with a
modulo per byte. On an emulated R5000 that capped every guest-side and
forwarded TCP number near 1-2 MB/s whatever the NAT did, and hid the real
improvements. Fills and checks are now `memcpy`/`memcmp` against a
precomputed table, and Adler reduces every 5552 bytes. Any new guest
benchmark: check it on the host first, then ask whether the guest CPU spends
more time in your test than in the network stack.

## Four things that were really limiting it

1. **No MSS option in our SYN-ACK.** For off-subnet peers IRIX then uses its
   512-byte default MSS, so every upload through the NAT used small segments.
   FTP put to a LAN NAS went 0.8 → 6 MB/s once the SYN-ACK carried MSS 1460.
2. **seeq-enet lost wake-ups.** It delivered one frame per loop pass and then
   waited on a condvar with a 1 ms timeout. A notify that lands while the
   thread is busy is gone (a condvar has no memory), so bursts toward the
   guest drained at one frame per millisecond. The loop now goes straight
   round while frames move, and `net::Wake` remembers a notify until the next
   wait consumes it.
3. **Blocking calls on the NAT thread.** `TcpStream::connect_timeout(5 s)`
   and the DNS relay's blocking `recv` (2 s) froze every flow, NFS included,
   for as long as they waited: a 4 s ping during one connect to a black hole.
   Connect is now non-blocking (the SYN-ACK goes out when the socket
   connects) and DNS is an ordinary UDP flow.
4. **1 ms polling on the NAT thread** cost 3.7% of a core idle (0.7% of it
   rebuilding the status snapshot every pass). It now sleeps in `poll(2)` on
   every host socket plus the `Wake` self-pipe, with the timeout taken from
   the earliest TCP timer: 0.1% idle.

## Wake ordering gotcha

`Wake::consume` must drain the pipe *before* clearing the pending flag. The
other order lets a notify set the flag and write a byte between the two steps;
the drain eats the byte, the flag stays set, and every later notify skips its
write because the flag is already set. A `poll(2)` sleeper then never wakes
again.

## The reply hold cost a whole tick (2026-10-23)

Replies generated while draining guest frames go to `next_pass` (the
CLRINT FIXME in `step`). `idle_timeout` then returned 1 ms, and nothing
wakes the engine early because the guest is idle waiting for exactly that
reply, so every lock-step exchange paid one full `poll(2)` tick per packet
(Windows: one ~15.6 ms condvar tick). Built-in TFTP: 512 B per tick, about
450 KB/s at best. Now a held reply asks for `REPLY_HOLD` (150 µs), and waits
under 1 ms are a plain `thread::sleep` (`short_wait`), since poll and the
condvar round them up to a tick. Test:
`testnet::tftp_blocks_are_not_paced_by_the_poll_tick`. If the CLRINT race is
real, a hold this short is where it would show: check NetBSD bulk receive
and IRIX flood ping after changing it.

Verified 2026-10-23 on IRIX: TFTP download much faster; flood ping
(`ping -f -q -c 20000 -s 1400` to the gateway) lost 1 packet on the first
run and 0 on five more. No sign of the CLRINT race at 150 µs.

## The hold is gone (2026-10-31)

The CLRINT race the hold guarded against did not show up without it. On an
Indy (IRIX 6.5.22, R4400), 4 MB TFTP gets had no retransmits with or
without the hold. In two pairs of boots they took 8.6-10.2 and 9.0-14.9 s
without it, 12.2-17.7 and 13.1-15.9 s with it (four or five gets each,
man-page indexer running). IRIX's handler reads INTPEND, writes CLRINT,
then reads the receive control register, so a frame finishing after the
CLRINT needs a new interrupt, which the model raises.

Frames are still lost in bulk TCP, with or without the hold, and the one
loss traced (without it) was a full receive ring. The SEEQ thread copies
whatever the NAT has queued into the ring as fast as the host runs, while
the guest takes the interrupt on its own clock. In the first `rsh
192.168.0.1 cat` of 4 MB after boot, with the man-page indexer busy, 26-27
full-size frames went in before the handler ran, the channel stopped at
the end-of-ring descriptor, and IRIX sometimes reset the interface
(CH_RESET, reprogram the SEEQ, NDBP back to the ring's start) and re-armed
the filled descriptors with the same buffers: frames the model counted as
delivered that `netstat -i` never counted. TCP recovers them. The gap
between `seeq status` rx_delivered and Ipkts after the first transfers was
14 and 88 with the hold, 12, 26, 79 and 186 without, and 0 in other boots
of each; later transfers do not add to it. A wire this fast is what makes
it possible: a 10 Mb/s SEEQ cannot take a full frame faster than every 1.2
ms. Pacing delivery to that, or to the guest's clock, is the fix if it
matters.

## Still open

- seeq-enet keeps its 1 ms wait (0.9% of a core idle): HPC3 kicks it only when
  a DMA channel goes inactive → active, so a descriptor appended to a running
  TX chain is found only by the tick. Kick on append, then it can sleep.
- Windows has no `poll` path: the NAT thread still ticks at 1 ms there.
