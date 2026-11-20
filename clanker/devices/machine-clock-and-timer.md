# One machine clock; devices take time and deadlines from it

Status: design (read `src/clock.rs` before relying on details).

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
