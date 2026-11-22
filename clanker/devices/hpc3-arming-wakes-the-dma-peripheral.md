# HPC3 arming wakes the DMA peripheral; device threads do not poll for it

**Status:** current (2026-11-22, issue 346). **Keywords:** hpc3, pdma, scsi,
wd33c93a, seeq, enet, wake, poll, dmago, wait_active, armwait, netbsd

## Symptom

A driver can issue the WD33C93A's TRANSFER_INFO before it arms the HPC3 SCSI
channel (NetBSD and OpenBSD do: `SET_SBIC_cmd`, then `dmago`). The chip's
thread reaches its data phase with the channel inactive. It used to retry the
channel every 100 µs for up to 100 ms (before that, a `yield_now()` hot loop).
Each retry took the channel lock and cost one host sleep, and the data phase
started up to one sleep (100 µs or more on macOS) after the arming.

## Who hits it

NetBSD 11.1 (the sgimips install CD on an Indy) waits on almost every
transfer: reading the CD with `dd` gave 1579 waits for 2256 reads. IRIX 6.5.22
on an IP28 arms the channel first: no wait in a boot, a directory walk and a
20 MiB write. `scsi armwait` counts the waits; `tools/scsi-armwait-e2e.py`
reproduces the NetBSD case. Measured 2026-11-22 on a loaded host: every wait
took 100-200 µs (mean 129 µs) with the poll, mean 6 µs (2 KiB reads) and
38 µs (32 KiB reads) with the wake, all but one under 100 µs.

## Rule

Every `PdmaChannel` carries an `ArmSignal` (a generation counter and a
condvar). `set_active(true)` bumps it on the inactive→active edge, under the
channel lock. `PdmaClientImpl::wait_active(timeout)`:

1. reads the generation (signal lock only),
2. takes the channel lock and returns if the channel is active,
3. waits on the signal until the generation moves or the deadline passes.

An arming between steps 1 and 2 shows up in 2; one after 2 moves the
generation, so the wait cannot miss it. The waiter never holds the signal lock
while taking the channel lock, and the arming side takes the signal lock last
(a leaf), so there is no lock-order cycle. The WD33C93A holds its own state
lock across the wait, as it did across the sleeps; the CPU thread's arming
write takes only the channel lock and the signal.

`DmaClient::wait_active` has a default that sleeps 100 µs and returns false,
so a peripheral on another DMA engine keeps polling until that engine signals.

The read path retries a channel that is active yet returns no data (a bus
error on the buffer) every 100 µs, because `wait_active` returns at once then.

## Ethernet

The SEEQ already had this wake: `ENET_RX_CTRL` and `ENET_TX_CTRL` writes that
make the channel active call `kick_rx`/`kick_tx`, and NAT signals the same
`Wake` for received frames. Its remaining timeout (1 ms backing off to 50 ms,
`ENET_POLL_MAX`) is a safety net, about 20 wakeups/s on an idle machine. It
was left as it is: the gain from lengthening it is small and was not measured.
The 1 ms floor while a received frame is held for room also stays: the guest
makes room by writing descriptors in memory, which no register write signals
(apart from the receive-control read that `kick_if_holding` catches).
