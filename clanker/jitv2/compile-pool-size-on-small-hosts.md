# The compile pool is sized for the host, not just the guest

**Status: revised 2026-11-16.** The rule below replaced an earlier one that
also reserved the guest's own CPUs; the revision and its reason are in
[Revision](#revision-2026-11-16).

**Keywords:** jitv2,compile-pool,threads,sizing,small-host,2-core,oversubscribe

## Symptom

The jitv2 compile pool was sized from the *guest* alone: one worker per
guest CPU, up to four. On a small host that oversubscribes it.

The target host is an Intel i3-4130: **2 physical cores, 4 logical CPUs**.
A 2-CPU guest (Octane, Tezro) already owns two host threads, and the
emulator adds its device threads on top (graphics compose, SCSI, ethernet,
audio, the timer). Two compile workers on top of that is four runnable
threads on two physical cores — every compile competes with the guest it is
compiling *for*, and with the device thread that has to compose a frame.

The pool's own idle cost made it worse: before the parking change, each
worker woke every 200µs to find an empty queue (see
[compile-workers-park-instead-of-polling.md](compile-workers-park-instead-of-polling.md)),
so a 2-worker pool burned host time even with nothing to compile.

## Rule

The compile pool is a *guest* tool that runs on the *host*. Size it against
both:

```
want    = guest_cpus.clamp(1, AUTO_JITV2_THREADS_MAX)   // 4
usable  = host_cpus - AUTO_JITV2_HOST_RESERVE             // reserve 1 for the device threads
threads = min(want, max(1, usable))
```

- `host_cpus` is `std::thread::available_parallelism()`: the logical CPUs
  the process may run on. It honours the CPU affinity mask and cgroup
  quotas, so a container or a pinned process sees its own budget rather
  than the machine's.
- The floor is **one**: a pool with no workers would compile nothing and the
  guest would run the interpreter forever.
- The device threads get one host CPU. The guest's own CPU threads do **not**
  come out of the pool's budget — see the revision below for why.

On the i3-4130 (4 logical CPUs):

| guest | before | after |
|---|---|---|
| 1 CPU | 1 | 1 |
| 2 CPUs | 2 | 2 |
| 4 CPUs | 4 | 3 |

On a big host (16 logical CPUs) nothing changes: 2 guest CPUs still get 2
workers, 4 guest CPUs still get 4.

## Fix

- `config::auto_jitv2_threads(guest_cpus, host_cpus)` is the pure function
  (with the host count as an argument, so the tests do not depend on the
  machine they run on). `config::host_parallelism()` supplies the real one.
- `MachineConfig::jitv2_pool_size()` returns `(threads, reason)`: an
  explicit `[jitv2] threads` / `--jitv2-threads` wins and says "set
  explicitly"; otherwise the auto size and its explanation.
- `MachineConfig::settle_jitv2_threads()` writes both into the config, so
  the GUI and the `toml` writer see the same number the pool will use.
- The machine builders ask `cfg.jitv2_pool_size()` instead of reading
  `cfg.jitv2.threads`, so a configuration built in code (never parsed,
  never settled) still gets the auto size.
- `CompileQueue::set_thread_count_with_reason()` carries the reason to
  `j2 threads`, which now prints e.g.

  ```
  j2 threads: 2 (auto: one worker per guest CPU (2); the host's 4 usable CPUs have room)
  ```

## Gotchas

- **An explicit 0 must still reach `validate()`.** `jitv2_pool_size()` does
  *not* clamp an explicitly configured 0 — it returns it as it is, so
  `validate()` can still refuse it ("jitv2.threads must be at least 1").
  The clamp to one happens at the point of use (`CompileQueue::set_thread_count`).
  Clamping in the config would silently turn a refused config into a
  running one.
- **`settle_jitv2_threads()` now always writes `jitv2.threads`.** Before,
  it left an explicitly configured value alone; now it writes the same
  value back (and the reason). The GUI's `machines::to_toml` still leaves
  the key out while it is automatic, so the auto value is never frozen into
  a config file.

## Revision (2026-11-16)

The first version of this rule also took the guest's own CPUs out of the
pool's budget:

```
room    = host_cpus - guest_cpus - AUTO_JITV2_HOST_RESERVE
threads = min(want, max(1, room))
```

On the i3-4130 that gave a 2-CPU guest **one** worker and a 4-CPU guest
**one** worker. That is wrong, for two measured reasons:

1. **The guest does not wait for a compile — it interprets.** A guest CPU
   that reaches a page with no compiled code does not block on the pool:
   the dispatch gate in `src/cpu/mips_exec.rs` pushes the request and falls
   through to `step_int()`, so the guest keeps running (in the interpreter)
   until the pool publishes the page. The guest's CPUs are therefore *not*
   free during a compile burst, but neither are they the ones a pool
   worker is competing with for *progress*: the worker's output is what
   the guest is waiting for, and the guest's own use of the CPU in the
   meantime is the interpreter fallback, which is slower but not idle.
2. **The pool is demand-bound, not CPU-bound.** The measured boot profile
   (`compile-time-profile.md`: 5,000-odd compiles at 50-55 ms over a
   2.5-minute run) left four workers at roughly **half** busy each. The
   host was not saturated; the compile *latency* was the limit. Halving the
   pool doubles the wall-clock of the compile-bound phase for no host-CPU
   saving, because the workers were not the scarce resource — the host's
   spare logical CPUs were.

So the guest's CPUs are no longer subtracted. The pool's width now follows
the guest's *demand* (one worker per guest CPU, the measured want) and is
capped only by the host's usable CPUs, one of which stays with the device
threads.

## Not done

- The pool size is still a *count*. A compile worker is a Cranelift
  compile — CPU-bound, memory-hungry — and on a 2-physical-core host two
  workers still share one core with the guest. A work-stealing or
  time-sliced pool (one worker that borrows a second core only while the
  guest is blocked) would fit a small host better than a fixed count.
  Note that "while the guest is blocked" has no easy trigger today: the
  guest never blocks on a compile, it interprets.
- Physical-core detection (as opposed to logical CPUs) is not done:
  `available_parallelism()` counts hyperthreads. For 1- and 2-CPU guests it
  makes no difference (both give one worker), so it is not worth the
  platform-specific code yet.
- **Selective wake is not done.** `CompileWake::notify_all` unparks every
  worker on every push. With a pool of 2-4 and a request arriving every
  ~30 ms this is a handful of futex calls a second — measured as noise —
  but a per-worker "parked" flag would let the producer wake only as many
  workers as there is backlog for. It needs the usual
  set-flag-then-recheck-the-condition dance to avoid a lost wakeup.
