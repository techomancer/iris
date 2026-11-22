# Idle poll backoffs: sleeps in milliseconds, not microseconds

**Status: current (2026-10-27).**

**Keywords:** perf,idle,poll,park,wake,backoff,seeq,enet,wd33c93a,scsi,gr2,mgras,o2gfx,ip32,ip30,yield,small-host,2-core

## Symptom

On a 2-core/4-thread host (or a small board), an idle iris still burns
noticeable CPU and never lets the host drop into deep C-states. Every device
thread with nothing to do was waking at 100 Hz - 10 kHz:

| Thread | Old idle wake rate | Cause |
|---|---|---|
| `seeq-enet` | 1000/s | `rx_wake.wait(1ms)` "to poll TX DMA" |
| `GR2-HQ2`/`GR2-RE3`, `MGRAS-HQ3`/RSS, `CRIME-RE` | ~500/s each | `park_timeout(2ms)` backstop in the engine `sleep()` |
| `O2-Service` (`ip32_machine.rs`) | 10 000/s | `sleep(100µs)` service loop |
| `IP30-Service` | 10 000/s | `sleep(100µs)` service loop |

The WD33C93A was worse under load: waiting for the driver to arm the DMA
channel, it retried in a `yield_now()` hot loop for up to 100 ms, hammering
the PdmaChannel lock the CPU thread needed to arm it — on a small host the
SCSI thread competes its own core away from the arming it is waiting for.

## Cause

Every one of these waits was a *safety net around an already-event-driven wake
path*, sized as if it were the wake path:

- `Wake` (`src/net/wake.rs`) keeps a notify pending until the next `wait`, so
  a signal sent while the consumer was busy is never lost. The enet thread's
  real wakes are `seeq.kick_tx()` (HPC3, when the CPU arms the TX channel)
  and `rx_wake.notify()` (NAT, when a frame arrives); the 1 ms timeout only
  covered "driver re-armed a descriptor chain while the channel stayed
  active".
- The GR2/MGRAS/O2 engine parks use the standard two-phase protocol (raise
  `asleep`, `fence(SeqCst)`, re-check for work, then park). A producer push +
  wake cannot slip past that; the timeout is pure paranoia.
- The O2/Octane service loops polled at 10 kHz to move host input (UART
  typing, net frames, audio buffers) into the devices, but nothing they
  service is hurt by a millisecond of latency.

## Fix

- **seeq-enet** (`src/dev/seeq8003.rs`): the poll backs off 1ms -> 2 -> 4 ->
  ... -> 50ms while nothing moves; any notify, frame or interrupt returns it
  to 1ms. Bursts still drain at speed (`busy` already skips the wait), and
  the mid-burst re-arm case always has activity within a few ms, so it never
  sees the grown timeout.
- **gr2 / mgras / o2gfx engine sleep()**: park backstop 2ms -> 50ms. A missed
  wake is impossible by the protocol; if one ever happened it would cost one
  50ms latency, never correctness.
- **ip32 service loop**: adaptive — 100µs while CRIME is interrupting
  (machine busy: net frames, audio, keys all raise interrupts), backing off
  to 2ms while it is quiet.
- **ip30 service loop**: fixed 1ms (bring-up-stage machine; UART typing and
  the compare timer don't care, and there is no activity bool to key on —
  `Ip30Bus::service()` returns `()`).
- **wd33c93a**: both "channel not yet armed" waits sleep 100µs between
  attempts instead of `yield_now()`. Still far below SCSI command timescales;
  the deadline stays 100ms. *Superseded (issue 346):* they now block in
  `DmaClient::wait_active`, woken by HPC3 on the channel's inactive→active
  edge; see `clanker/devices/hpc3-arming-wakes-the-dma-peripheral.md`.

Deliberately **not** changed:

- `hptimer`'s sub-200µs spin — that is the 44.1 kHz audio pacing path; see
  `clanker/perf/hal2-pump-thread.md` before touching it.
- The SCC RX threads' 10ms poll — `SerialBackend::recv_byte` is polled with
  no wake primitive; restructuring the backends for 200 wakeups/s is not
  worth it (and would risk console latency).
- Anything gated on `idle-pause` — that is the maintainer's opt-in call.

## Rule

When a wait is only a safety net around a wake that cannot be lost
(`Wake`'s pending flag, the two-phase park protocol), size it for idle cost,
not worst-case latency. And never spin-yield waiting for another thread on a
small host: yield keeps the waiter hot and contended for the very lock the
other side needs; sleep lets it finish.
