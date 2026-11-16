# The compile pool parks; the producer wakes it

**Status: implemented 2026-11-16.**

**Keywords:** jitv2,compile-pool,idle,park,unpark,wake,backoff,small-host,2-core

## Symptom

An idle guest (at the PROM screen, nothing to compile) still kept a
`jitv2-compile-N` thread busy. Measured on an IP28+maximpact guest sitting at
the PROM, one worker cost **1.4% of a host core** — the third-largest thread
in the process after `hptimer` and `Display-Refresh`, and pure waste.

## Cause

`CompileQueue::worker_loop` polled the request queue on a timer:
`WORKER_IDLE_POLL_BACKOFF` was 200µs, so an idle worker woke ~5000 times a
second, popped two empty queues, and slept again. The wakeups are what cost
the CPU, and on a small host they also keep the host out of deep C-states.

This is the same failure the device threads had, and the same fix
([`../perf/idle-poll-backoffs-for-small-hosts.md`](../perf/idle-poll-backoffs-for-small-hosts.md)):
*size a wait for idle cost, and let the producer wake the waiter*.

## Fix

The producer now wakes the worker directly.

- `CompileWake` (`src/cpu/jitv2/jitv2.rs`) holds the worker threads' handles.
  Each worker registers its own handle when it starts; `CompileQueue::stop`
  clears the registry, so a restarted pool does not accumulate dead handles.
- `CompileLane` wraps the request queue together with the pool's wake-up
  channel, and derefs to the queue. Both lanes (`queue` and `hot`) share one
  channel — the same workers drain both. The executor's dispatch gate holds
  an `Arc<CompileLane>` (as it held an `Arc<ArrayQueue>` before), so pushing
  a request wakes the pool without taking `Jitv2`'s mutex.
- `push_compile_request`/`push_hot_request` unpark the workers after the
  push. A push is rare — one per *scheduled page*, not one per instruction —
  so the mutex + unpark there is off the hot path.
- The worker's idle path is now `thread::park_timeout(20ms)`, not
  `thread::sleep(200µs)`. `park`/`unpark` tokens are sticky: an `unpark` that
  arrives while the worker is busy is remembered, so the next `park` returns
  immediately. **A wakeup cannot be lost**, which is what lets the timeout be
  long — it is only a backstop for the periodic pending-publish sweep, not
  the request path.
- Two backstops, because the worker has two jobs when the queue is empty: it
  retries a publish that is still outstanding (its own range is gap-blocked
  behind another worker's, so the code is compiled but not sealed — the guest
  is running that page interpreted until it lands), and it sweeps for a
  force-seal. The first keeps the old 200µs interval
  (`WORKER_IDLE_PENDING_BACKOFF`) and the second gets the long one
  (`WORKER_IDLE_PARK_BACKSTOP`). The pending state is bounded by the pending
  threshold and the 100ms force-seal, so the short interval is transient.

Two more places had to wake workers, or a long park would have shown up as
latency:

- The flush/seal **leader** wakes the followers right after it wins the
  election (`try_flush`/`try_force_seal`), so they reach the quiesce barrier
  now instead of after their backstop. Without this, `wait_for_followers`
  would stall for up to the backstop on every arena flush.
- `CompileQueue::stop` wakes the workers after clearing the barrier's run
  flag, so the join does not wait out a backstop.

## Rule

A background worker that waits for work should park, and the producer should
unpark it. `park`/`unpark` tokens are sticky, so the "check the queue, then
park" race cannot lose a wakeup — which means the timeout only has to cover
the periodic housekeeping, not the work itself. Poll intervals on a small
host are a last resort, not a wakeup mechanism.

## Not done

- The pool's *size* is still one thread per guest CPU (up to 4), independent of
  how many host cores there are. On a 2-core host a 2-CPU guest still gets 2
  compile workers, which oversubscribe the host during a compile-heavy phase.
  See [compile-pool-size-on-small-hosts.md](compile-pool-size-on-small-hosts.md).
