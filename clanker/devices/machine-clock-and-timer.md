# One machine clock and `Timer`; devices take time and deadlines from them

Status: design (read `src/clock.rs` and `src/hptimer.rs` before relying on details).

**What to know.** A machine has one `Clock`, in nanoseconds. A device that
needs guest-visible time reads it; it does not keep its own `Instant`.

- The `TimerManager` owns the clock (`TimerManager::clock()`); clones share
  it. Timers still run on host time, devices read machine time.
- It holds still while the machine is stopped. The machine stops it after its
  devices have stopped, so a device's own `stop` still reads a running clock,
  and starts it before they start.
- `restore(ns)` puts the reading forward or back; a stopped clock stays
  stopped. The Octane's snapshot saves the reading and a load restores it
  with the machine stopped, before the devices load and re-arm.
- Tests use `Clock::stepped()`, which moves only on `step(ns)`. Never the
  wall clock: a test that sleeps for a deadline is a flake.

**Host time.** The source is `platform::get_host_ticks`: nanoseconds since the
first call, read through `quanta` (the TSC where it is invariant, the OS
monotonic clock where not). Each socket has its own TSC and on a 2-socket
host one runs about 390 ns behind the other. Each thread keeps the largest
reading it was given, so a thread never reads below its own earlier reading.
Two threads can still differ by that much, so:

- take a difference of two readings with `saturating_sub` where threads can
  differ;
- a counter the guest reads from several CPUs clamps itself where it is read
  (HEART, the Hub PI and Bedrock counters do);
- `Clock::now` itself uses `wrapping_sub` on purpose: `restore` puts the
  reading ahead of the source by making the shift wrap.

A process-wide maximum would fix the skew but costs a shared cache line on
every read, which is why the clamp is per thread.

## Timer: one deadline a device owns

`hptimer::Timer` holds the timer id, the weak self reference and the
callback plumbing a device used to write by hand. Its deadline is an absolute
machine time.

- Build it in the constructor: `Timer::new(&tm, &me, callback)`, with `me` the
  `Weak` from `Arc::new_cyclic`. The callback is `Fn(&Device, due)`.
- Set it with `arm_at(when)`, `set_due(Option)` or `cancel()`. These are safe
  before `start`; the deadline waits.
- `Device::start` calls `Timer::start`, `Device::stop` calls `Timer::stop`.
  `stop` removes the host timer, keeps the deadline (a save sees it, `start`
  arms it again) and waits out a running callback. Never call it from a
  callback or under a lock the callback takes.

**The callback resyncs, it does not trust `due`.** A fire for a deadline the
device has since replaced can still arrive. Under the device's own lock,
recompute what is due from device state and the clock, act on that, and set
the next deadline.

**Fast paths.** Devices set every deadline after each register write, and most
calls repeat the one already armed.

- An unchanged deadline returns without taking a lock or touching the
  manager (an atomic copy of the deadline).
- A deadline that moves later keeps its pending host timer. The timer fires
  at the old time, finds the clock short of `due`, and re-arms for the rest.
  Re-timing on every store cost the storing CPU microseconds that a guest
  comparing two CPUs' counters can see.
- A deadline that moves earlier gets a new host timer.

Host timers run on host time and the machine clock can lag it (held while
stopped, stepped in tests), so an early fire reschedules instead of calling
back.

## Snapshots

Save a deadline as a machine time, not as time left. The load restores the
clock first, then the device, then `start` arms. A snapshot taken with the
old keys is refused; there are no fallback keys, and the schema is bumped
when a device's keys change.

## Tests

Build the `TimerManager` with `TimerManager::with_clock(Clock::stepped())`,
`step` the clock past the deadline and wait for the callback with the bounded
`test_board::until` helper. The existing tests in `src/hptimer.rs` show an
early fire, a replaced or cancelled deadline staying silent, `stop` keeping
the deadline for `start`, and a callback re-arming itself.
