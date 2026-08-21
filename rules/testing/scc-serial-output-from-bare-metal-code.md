# Getting serial output out of bare-metal guest code

Three things silently swallow SCC output, all hit while bringing up the CPU
test-suite support. In order of how long each one costs you:

1. **IOC serial 1 is SCC channel *B*, not A.** `IOC_SERIAL1_CMD`/`_DATA`
   (`0x1FBD9830`/`0x34`) map to `Z85c30::write` indices 0/1, and that function
   dispatches `0 => channel B, 2 => channel A`. So the first serial port in the
   IOC map is ttyd1 → **TCP 8881**. Channel A (ttyd2, TCP 8880) is at
   `0x1FBD9838`/`0x3C`. Writing to `0x…30` and listening on 8880 gets you
   nothing, and looks exactly like a broken device.

2. **Transmit is gated on WR5.TX_ENABLE.** An uninitialised channel queues the
   byte and the TX thread never latches it. Minimum init for output: write `5`
   to the command register (select WR5), then `0x68` (`TX_8BITS | TX_ENABLE`).
   Baud/BRG can stay unset — `tx_delay` then works out to 4 µs per character.

3. **The TCP serial backend holds exactly one client, and only notices a dead
   one on a failed write.** `TcpSocketBackend::send_byte` writes to
   `conn` if it is `Some`; `conn` is only replaced by an `accept()` inside
   `recv_byte`, which runs when `guard.is_none()`. A client that connected and
   went away leaves `conn` populated, so a *new* connection sits unaccepted in
   the backlog and every transmitted byte is written into the dead socket.
   Symptom: you connect, get no telnet handshake (`ff fb 01 …`), and see no
   output. Treat "no handshake within a second" as "reconnect", or use
   `--serial-log FILE`, which tees ttyd1 to a file and avoids the socket
   entirely — much the better option for a CI harness.

## The same gate costs enormous *time*, not just output

Point 2 above has a consequence nobody noticed for a long time: a bare-metal
image loaded with `--load-elf` never runs the PROM, so WR5.TX_ENABLE is never
set, so the SCC's four-byte holding queue fills and `RR0.TX_BUFFER_EMPTY` goes
low and never comes back. `cpu-tests/harness/console.c`'s `scc_putc` spins on
that bit with a `TX_SPIN_LIMIT` of 100,000 before giving up — **per character**.

At interpreter speed that is roughly 5 ms of emulated spinning per byte. The
benchmark suite prints its table a row at a time and then a full machine-readable
block, several thousand characters in all:

| | full `bench/` run, r4400 lightning |
|---|---|
| before | **117 s** |
| after | **46 s** |

Same 40/40 accuracy, same timed totals (12.5 s vs 12.4 s) — the entire
difference was the guest waiting on a serial port with nothing on the other end.
`cpu-tests` was paying the same tax.

The fix is in `scc_putc`: latch a `scc_dead` flag the first time the spin limit
is exceeded and stop trying. It is deliberately conditional on `have_testdev`,
because that is the question that actually matters — *is anything else reading
this?* With a test device the host reads that instead and serial is redundant;
without one serial is the only sink there is, and a slow port beats no output. A
PROM-booted run has a working SCC, never trips the limit, and is unaffected
either way — which matters, because `run-prom.sh` decides pass or fail by
grepping the serial log.

**Watch for this shape generally.** Any bounded spin on a device bit that a
bare-metal image never enabled is a per-operation cost that looks like slow
emulation. If a workload's wall clock is far larger than the sum of what it
claims to have timed, suspect polling before suspecting the CPU: here the
benchmark's own `#totals ns=` said 12.5 s while the process took 117.
