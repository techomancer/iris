# On Windows a timed park rounds up to the system tick; the hptimer uses a waitable timer

Status: fixed 2026-11-11 (`src/hptimer.rs`, `Waiter`).

**Symptom.** On the Windows build the Octane's IRIX ran its clock interrupt
at 64-80 Hz instead of 1000 Hz: `cop0` on the monitor showed the IP7 fired
count climbing by about 1,270 in 15.7 s while the guest idled. Everything
the kernel paces by ticks took ten times as long. glquake after a timedemo
sat in `glColor4f` with the VPro idle (command FIFO empty, no interrupt
asserted, HEART pending 0) and moved on after a long while, then exited
normally: a kernel timeout running slow, not a lost wake-up.

**Cause.** The CP0 Compare deadline is an `hptimer` one-shot. The worker
slept to 50 µs short of each deadline with `thread::park_timeout`, which on
Windows is `WaitOnAddress` with a millisecond timeout, and such a wait only
returns on a system timer tick: 15.625 ms unless some process has raised the
resolution. Every ~1 ms sleep returned ~15 ms late. Linux and macOS futex
and condition-variable waits return within tens of microseconds, which is
what the 50 µs spin margin was sized for.

**Measurements** (Windows 11, this machine, 1,500 waits each; late = slept
minus asked, µs):

| wait of 950 µs | mean | p50 | p99 |
|---|---|---|---|
| `park_timeout` | 14,614 | 15,033 | 15,161 |
| `park_timeout` with `timeBeginPeriod(1)` | 613 | 1,045 | 1,065 |
| high-resolution waitable timer | 80 | 57 | 520 |
| same with `timeBeginPeriod(1)` | 80 | 57 | 496 |
| `thread::sleep` (std uses that timer since 1.75) | 82 | 56 | 536 |

The high-resolution waitable timer (`CreateWaitableTimerExW` with
`CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`, Windows 10 1803+) ignores the
system tick but has a grid of its own: it fires at the next multiple of
0.5 ms counted from the request, 25-80 µs after it, whatever the system
tick, and whether the due time is relative, absolute or set with
`SetWaitableTimerEx`. Asked for 100, 300 or 450 µs it sleeps ~525; for 550
to 950 µs ~1,030; for 1,000 to 1,450 µs ~1,530. `timeBeginPeriod(1)` does
not change that grid.

**Fix.** The worker's wait is a `Waiter`: on Linux and macOS still
`park_timeout`/`unpark`; on Windows the high-resolution waitable timer and
an auto-reset event, waited on together, so `TimerManager::wake_up` still
ends a wait early (the event keeps a wake delivered between waits, as the
park token does). The waiter measures its grid at start and `plan`s each
sleep: ask for the grid point under the deadline and spin the rest when
that rest fits `SPIN_MAX`, else ask for the whole wait and take the next
grid point late. The sweep that set `SPIN_MAX` (2 s runs, process CPU):

| budget | 1 kHz recurring: late mean / p99, CPU | 1 ms one-shot re-armed from its callback: late mean / p50 |
|---|---|---|
| none | 253 / 714 µs, 2.3% | 62 / 8 µs |
| 250 µs | 207 / 632 µs, 0.8% | 41 / 6 µs |
| 400 µs (chosen) | 62 / 532 µs, 4.7% | 35 / 6 µs |
| 500 µs (always spin the step) | 1 / 1 µs, 38% | 1 / 1 µs, 45% |

The re-armed one-shot is the guest clock's pattern (a Compare write arms a
fresh deadline from now) and is accurate at any budget; the recurring case
compounds a late wake into a shorter next sleep that rounds up to the same
grid point, so it needs the budget. Exact timing costs a third of a core
and was not taken.

The waiter also calls `timeBeginPeriod(1)` for its lifetime. It does nothing
for the waitable timer but brings every other millisecond wait in the
process (the CPU's idle slices in `idle_park.rs`, `park_timeout` in the
display clocks and receive polls, `Condvar::wait_timeout`) from ~15 ms to
~1 ms. On Windows 10 2004 and later the raised resolution is scoped to the
process.

**Rule.** On Windows never pace anything finer than the system tick with
`park_timeout`, `Condvar::wait_timeout` or `WaitOnAddress`; use
`thread::sleep` (high-resolution timer in std) when no early wake is
needed, or the hptimer. Expect a 0.5 ms floor and grid on either.

**Check.** `cargo test --release --lib hptimer -- --nocapture` prints the
1 kHz recurring rate and mean lateness
(`recurring_timer_at_1khz_keeps_its_rate`, not ignored: it fails below 600
fires a second). On a live guest, `cop0` twice some seconds apart: the IP7
fired count should grow by the guest's tick rate (1000/s on the Octane).
