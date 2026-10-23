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

## Still open

- seeq-enet keeps its 1 ms wait (0.9% of a core idle): HPC3 kicks it only when
  a DMA channel goes inactive → active, so a descriptor appended to a running
  TX chain is found only by the tick. Kick on append, then it can sleep.
- Windows has no `poll` path: the NAT thread still ticks at 1 ms there.
