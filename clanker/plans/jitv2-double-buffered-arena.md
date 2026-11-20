# Plan: a double-buffered jitv2 code arena, rotated without stopping a CPU

Status: **plan, nothing built.** Written 2026-11-19 against 1f72e2387
(`pcp-bank-offset`). Line numbers below are from that commit.

## Why

The compile pool's code arena is one 512 MB reservation
(`ARENA_RESERVE_SIZE`). Nothing in it is ever freed: every recompile of a
page leaves the old function behind. When it fills (16 MB short of the end,
or an allocation fails), a worker leads a flush. In practice this is the
flush that fires; the page pool running out of slots is rare.

What that flush does today (`run_leader_flush`, `jitv2.rs:3957`):

1. wins `PoolBarrier::try_lead`, waits for every other worker to park;
2. stops every CPU (`World::try_lead`, retried every 20 ms while a CPU leads
   a flush of its own, or `cpu.stop()` on a uniprocessor);
3. drains both request lanes;
4. `mega_flush`: resets every page slot, the key index and every frame table;
5. reserves a new 512 MB arena, rebuilds its `Codegen` and the memory
   helpers, releases the barrier (the other workers rebuild on wake), and
   lets the CPUs go.

The guest is stopped for all of it. Two threads can lead (a worker for the
arena, a CPU for the page pool), which is what the merged barrier, the 20 ms
`BARRIER_FOLLOWER_POLL_INTERVAL` retry and `stop_for_drop`'s loop exist to
survive.

## The idea

Two arenas, A and B. Workers compile into one; the other is empty and ready.
When the active one fills, the worker that notices switches allocation to
the spare, takes the old arena's code away from every page, asks each CPU to
step out of compiled code once, and when they all have, wipes the old arena
in place. It becomes the next spare. No CPU parks and no worker parks.

## Why a grace period is enough, and a refcount is not needed

- **Dispatch never keeps a code pointer.** `step_jit` loads `page.func()`
  again for every entry (`mips_exec.rs:4251`). There is no per-CPU cache of
  function pointers and no chaining from one page's code into another's.
- **Compiled code reaches its helpers by absolute address**
  (`call_indirect` on a constant, `is_pic` off). Code in A calls helpers in
  A; nothing in B points into A.
- **The kick exists.** `REQUEST_JIT_REMAP` (`smp.rs:44`) sets
  `HOST_PARK_BIT` in the CPU's interrupt word; every compiled preamble and
  loop bails on it; the run loop handles the request at the batch boundary
  (`mips_exec.rs:11380`) and carries on without parking. `ParkSlot::release`
  returning `true` is the CPU's acknowledgement, and `holds()` lets another
  thread see it.

So once no page points into A, a CPU can only be in A's code if it entered
before the sweep. One trip through the batch boundary proves it is out, and
it cannot get back in. A refcount on the page would cost two atomic
read-modify-writes per dispatch on a line every CPU shares, for the same
guarantee.

## Protocol

State: `code_epoch: AtomicU64` on the pool; an arena pair with an `active`
index; per worker `worker_epoch`; a `retiring: Option<(arena, epoch)>` that
the rotating worker owns.

1. **Trigger.** A worker sees the active arena over the threshold or gets an
   out-of-memory compile. It wins a CAS on a `rotating` flag. Losers carry
   on; nothing parks.
2. **Switch.** The rotator makes the spare the active arena and bumps
   `code_epoch`. The spare already holds its memory helpers (step 6).
3. **Workers follow.** Each worker compares `code_epoch` with its own at the
   top of its loop, rebuilds its `Codegen` on the new arena
   (`reset_with_shared_arena`), drops its `pending` count and stores
   `worker_epoch`. A worker in the middle of a compile finishes it into the
   old arena; step 4 throws the result away. `wake.notify_all()` gets idle
   workers there without waiting out their backstop.
4. **Publish refuses old code.** `PublishInfo` and the direct publish path
   carry the epoch the code was compiled in. `PhysicalCodePage::publish`
   and the redundant-compile revalidation check it against `code_epoch`
   under `publish_lock` and refuse a mismatch. That covers in-flight
   compiles and everything still in the old arena's seal queue.
5. **Sweep.** The rotator walks the page pool. For each claimed page, under
   `publish_lock`, if `func` lies in the old arena: clear `compiled`, point
   `func` at the poison stub, invalidate the compile snapshot, clear
   `func_fr1`. `requested`, `denied`, the FR pin, the key and the frame
   tables stay. Because the epoch moved before the sweep, a publish either
   lands before the sweep reaches the page (and is swept) or is refused.
6. **Kick.** Only after the sweep is complete, post `REQUEST_JIT_RETIRE` to
   every CPU's `ParkSlot`. A CPU that is parked, stopped or idle-parked is
   outside compiled code and counts as having answered.
7. **Reclaim.** The rotator does not wait. At the top of its loop it checks
   `retiring`: when every CPU has released the request (or is parked or
   inactive) and every `worker_epoch` has caught up, it recycles the old
   arena in place: back to its initial protection, pages dropped
   (`MADV_DONTNEED`; decommit on Windows), `position`, `sealed_up_to`,
   `collapsed_up_to` and the seal queue reset. Then it builds the memory
   helpers into it, alone, which is what a forced seal needs. The arena is
   now the spare; `rotating` clears.

**If the new arena fills before the old one is reclaimed:** workers stop
compiling (requests are dropped, as on a full queue) until the reclaim. CPUs
run what is published and interpret the rest. Nothing stops.

## What changes, by file

- `paged_memory.rs`: `SharedArena::recycle()`; `position` is no longer
  "never rewound". Arena address range readable without the mutex (the sweep
  tests `func` against it).
- `jitv2.rs`: the arena pair and `code_epoch` on `CompileQueue`; the
  rotation in `worker_loop` in place of `run_leader_flush`/`try_flush`;
  `PhysicalCodePage::retire_code(range)`; the epoch argument to `publish`
  and to the revalidation path.
- `pool_barrier.rs`: keeps the forced-seal cycle only. `was_flush` and the
  arena hand-off go; `arena()` for the monitor moves to the pair.
- `smp.rs`: `REQUEST_JIT_RETIRE`. `ParkSlot::park` and `release` mask only
  `REQUEST_JIT_REMAP` today; both request bits must be masked, or a CPU
  with the new bit posted parks for ever.
- `mips_exec.rs`: release the request in the run loop beside the remap one
  (no work to do: the release is the answer), and in `take_remap_request`
  for steppers.
- `codegen.rs`: nothing in emitted code. `emit_mem_helpers` is called by the
  rotator on the recycled arena.

## Hazards

- **`is_runnable` then `func()` are two loads.** Today `func` goes null only
  with the world stopped (or `j2 clear`). With a live sweep a CPU can pass
  `is_runnable` and then load the swept pointer. The poison must be callable:
  a static `extern "C" fn(*mut MipsCore) -> u32` returning `EXEC_FALLBACK`,
  which `step_jit` already turns into `step_int`. Never null.
- **The compile snapshot.** A swept page with a valid snapshot would have its
  dead code "re-validated" by the redundant-compile skip. Invalidate in the
  sweep; refuse by epoch in the skip.
- **Page slots do not move in a rotation.** A parked CPU keeps `self.pcp`,
  queued requests and seal entries hold raw page pointers. With the slots in
  place all of them stay valid, and each re-reads the page's state.
- **Kick after sweep, never before.** A CPU that answers before the sweep
  finishes can still load an old pointer afterwards.
- **Readers of code bytes off the CPU threads**: `j2 pcp`, the html dump,
  `jitv2_pcp_dump`. They must check the page under `publish_lock`, or hold
  off the reclaim while they read.
- **Inline compile (`j2 inline on`) and lockstep** own their `Codegen` and
  run on the CPU thread. They keep the stop-and-reset path.
- **Per-platform recycle.** Linux: `mprotect` + `MADV_DONTNEED`, and the
  hugepage collapse starts over. Windows: `MEM_DECOMMIT`. macOS: whatever
  the JIT mapping there needs; check before assuming `mprotect` back to RW.
- **A rotation still discards every function.** The stall goes; the
  recompile storm stays. Do not requeue hot pages at first: keeping pages
  across a flush was measured neutral to harmful
  (`jitv2/flush-preserved-entry-point-inflation.md`). The persistent cache
  makes the recompiles cheap where it is on.

## Order of work

1. `SharedArena::recycle()` with a unit test: allocate, seal, recycle,
   allocate again at offset 0; old pages read back as zero.
2. Poison stub, `retire_code`, epoch-checked `publish`. Unit tests on
   `PhysicalCodePage`: a publish from an old epoch is refused; a swept page
   is not runnable, is not revalidated, and compiles again on request.
3. `REQUEST_JIT_RETIRE` in `smp.rs` and the run loop, with the `ParkSlot`
   tests extended (a posted request does not park the CPU; a parked CPU
   counts as answered).
4. A loom model of epoch bump / publish / sweep / kick / reclaim, next to
   the barrier's: no interleaving lets a CPU call into a recycled arena, and
   none leaves a page pointing into it.
5. The rotation in `worker_loop`; delete `run_leader_flush`; trim
   `pool_barrier`.
6. Soak with `IRIS_JIT_ARENA_FLUSH_MB` set low (a few MB) so rotations
   happen every few seconds: IRIX boot to login on IP24 uniprocessor and on
   a two-CPU Octane, then the panther boot test.
7. `HACKING.md` (how jitv2 publishes and invalidates code), `debug.md` if
   `j2 status`/`j2 flush` output changes, a `clanker/jitv2/` note for the
   rules that came out of it.

## Later, same machinery

The page pool running out (`Claim::Full`, CPU-led `flush_from_cpu_thread`)
can use the same epoch: stamp a page with the epoch it was last entered (on
the page-crossing path only), and at a rotation return the pages nobody
entered to the free list after the same CPU answer, with a per-slot tenancy
counter in `CompileRequest` and `PublishInfo` so stale ones are dropped. On
`Claim::Full` the CPU would interpret and ask a worker to rotate early. That
removes the CPU-led flush, the pool stop/join inside it and the two-leader
problem. Not needed for the arena work, and rare today, so it waits.
`j2 flush`, restore and rollback would then become "rotate and wait for the
reclaim".
