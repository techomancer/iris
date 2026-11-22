# Device deadlines arm through `DeadlineTimers`, and accesses read the clock only when one is pending

Status: rule (2026-11-19, issue techomancer/iris#784).

**Symptom.** The devices ported from QEMU (IOC3, IOC4, the 16550 they and
the O2 share, the PHYs) brought their own timer bookkeeping and read the host
clock on every register access. Examples: every 16550 LSR or IIR read, every
THR write, every IOC4 register, and every IOC3 INT_STATUS access. Every
re-arm also woke the `hptimer` worker (an unpark syscall plus a scan of all
slots), even when the new deadline was later than the one the worker already
slept toward.

**Rules.**

- Arm device deadlines through `dev::deadline_timers::DeadlineTimers`. Use
  `arm` when one device lock covers every deadline (IOC4), and `arm_with`
  when the parts publish their own deadlines (IOC3). Do not write another
  table of `(TimerId, Instant)` slots. An access that moves no deadline
  returns early, checked against atomic mirrors without the table's lock.
- On a register path, read the clock lazily: `Ns16550::poll_deadlines`,
  `check_timeout_now`, `SerialDma::catch_up_now`. Keep the `now`-taking forms
  for tests and timer callbacks, where a deadline is already pending.
- Save a deadline in a snapshot with `dev::deadline::{until, due}`, not a
  local copy of the "nanoseconds left plus one" encoding.
- `TimerManager::add_one_shot`, `add_recurring` and `enable` wake the worker
  only if the new expiry is before its planned wake-up (`Sleep` in
  `src/hptimer.rs`). The plan is published under the same lock the adders
  take. If an adder runs before the worker's scan, the scan sees its timer.
  If it runs after, it sees the plan and wakes the worker if it must.

**Not unified (yet).** These remain separate time bases on purpose or for
later:

- `dev::chip_clock::Clock`: Indy DS1386/8254, switches to guest cycles under
  `ci_clock`.
- `dev::heart::Clock` (`Arc<dyn Fn() -> u64>`): HEART, the Hubs, the ELSC and
  the O2000 module. Tests step it.
- `dev::qlogic_ring` `RingThread::now_ns`: ISP1040/ISP2200/ISP12160. It is
  host time minus the time the machine was stopped, with a fake clock for
  tests.
- `bcm5701-rx`: the receive thread doubles as the chip's timer. It waits on
  its own `deadline()` with `Wake::wait`, because a deadline's work is DMA
  into guest memory and does not belong on the shared timer thread. Tests
  freeze `Bcm5701::now`.
- DS1744 and M48T35 timekeeper: calendar closures over `Instant`, persisted
  with `battery_clock`.
- `mace_ust`: host-time or instruction-count UST.

Folding these into one device clock (host `Instant` with a test override and
stopped-time accounting) would let the QLogic and BCM5701 test clocks go. It
touches every Origin-family machine, so it needs guest boots on each.
