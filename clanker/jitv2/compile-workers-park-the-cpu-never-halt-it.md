# A compile worker parks the CPU for a flush; it never halts it

Status: rule (issue #306, November 2026).

**Symptom.** A single-CPU machine under jitv2 hangs: the CPU thread sits in
`CompileQueue::stop` (inside `flush_from_cpu_thread`, holding `Mutex<Jitv2>`),
joining a compile worker, and that worker sits in `run_leader_flush`, in
`cpu.stop()` → `MipsCpu::halt`, joining the CPU thread (or, when the monitor
holds the executor, waiting for the executor lock, or for `Mutex<Jitv2>`).

**Cause.** Two flushes, two leaders. The CPU flushes when its page pool runs
out (or for `j2 flush`, a cache-geometry change); a worker flushes when its
code arena fills. On a machine with several CPUs the two already took turns
through `World` (`stop_others` on the CPU side, `try_lead` plus a poll of the
pool's run flag on the worker side). A CPU with its own pool had no world, so
the worker stopped it with `Device::stop`, which joins the very thread that
may be joining the worker.

The start-up half of #306 (a worker taking `Mutex<Jitv2>` to read
`dc_geometry` / `jit_consts`) was fixed earlier, in 95031882: workers get those
values before they spawn.

**Rule.**

- Every `MipsCpu` has a world: `MipsCpu::new` gives it `World::solo(park)`
  and sets it on the executor (`jitv2_world`) and on its own pool.
  `share_jitv2` replaces it, on the pool too.
- A CPU belongs to one world. Two worlds would both hold it with
  `HOLD_WORLD`, and one's release would let it run under the other. Every
  machine now has a world of its own (bank remaps call `layout_changed` on
  it), so every machine passes it through `share_jitv2` and `attach_pool`,
  also to a lone CPU and to a node whose processor module has a pool of its
  own (`share_pool`). The pool and the CPU must hold the same world: with two,
  the CPU's `stop_others` and the worker's `try_lead` do not take turns and
  the deadlock is back. `World::solo` remains for CPUs built without a
  machine, in tests.
- A world finds the caller's slot by the CPU id the slot records
  (`ParkSlot::new(cpu, ..)`), so a world of CPU 3 alone works too.
- A thread that runs no CPU stops the pool with `Jitv2::stop_pool`, which
  joins with `Mutex<Jitv2>` released. `CompileQueue::stop` under the lock is
  for the CPU side, inside `stop_others`.
- The `cpu.stop()`/`cpu.start()` branch of `run_leader_flush` remains only for
  a bare `CompileQueue` in tests.

**Test.** `a_cpu_flush_inside_a_worker_led_flush_takes_its_turn` (jitv2.rs)
forces a worker flush (`PoolBarrier::force_flush`, test-only) against a fake
CPU that flushes the moment it is held. Without the world it deadlocks and the
test fails after 20 s.
