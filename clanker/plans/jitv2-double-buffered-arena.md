# Plan: a ring of jitv2 code arenas, rotated without stopping a CPU

Status: **built 2026-11-24**; results under "Results". First written 2026-11-19 against 1f72e2387 as a
two-arena plan; revised 2026-11-24 (three arenas, retire the oldest, page
recycling with a hazard slot). 

## Why

The compile pool's code arena is one 512 MB reservation
(`ARENA_RESERVE_SIZE`). Nothing in it is ever freed: every recompile of a
page leaves the old function behind. When it fills (16 MB short of the end,
or an allocation fails), a worker leads a flush: every other worker parks,
every CPU stops, both request lanes drain, `mega_flush` resets every page
slot, a new arena is reserved, and the CPUs go again. The guest is stopped
for all of it, and afterwards every page compiles again from nothing.

A second flush exists: a CPU that finds the page pool full (`Claim::Full`)
stops the other CPUs and the pool and does the same. Two possible leaders is
what the merged barrier, the 20 ms `BARRIER_FOLLOWER_POLL_INTERVAL` retry and
`stop_for_drop`'s loop exist to survive.

## The design

### Arenas

`ARENA_COUNT` arenas (3) in a ring: one **active** (workers compile into
it), the ones before it **live** (their code still runs), one **spare**
(empty, memory helpers already built).

- When the active arena fills, the worker that notices makes the spare the
  active one. **Nothing is invalidated by the switch.** Code in the older
  arenas keeps running.
- The switch leaves no spare, so the same worker then **retires the oldest**
  live arena: takes its code away from every page, asks each CPU to step out
  of compiled code once, and when all have, wipes that arena in place. It is
  the next spare.
- Only what was still installed from the oldest arena compiles again, on
  demand. By then most of that arena is dead code left behind by recompiles.
  There is no requeue and no evacuation: a page that is entered again asks
  for a compile as any page without code does.

No CPU parks and no worker parks.

### Why a grace period is enough for the arena

- **Dispatch never keeps a code pointer.** `step_jit` loads `page.func()`
  again for every entry. There is no per-CPU cache of function pointers and
  no chaining from one page's code into another's.
- **Compiled code reaches its helpers by absolute address** in its own
  arena. Nothing in one arena points into another.
- **The kick exists.** A request bit on the CPU's `ParkSlot` sets
  `HOST_PARK_BIT` in its interrupt word; every compiled preamble and loop
  bails on it; the run loop handles the request at the batch boundary and
  carries on without parking. `ParkSlot::release` returning `true` is the
  CPU's acknowledgement.

So once no page points into the retiring arena, a CPU can only be in its
code if it entered before the sweep. One trip through the batch boundary
proves it is out, and it cannot get back in.

### Epochs

`code_epoch: AtomicU64` on the pool counts switches; each arena carries the
epoch it became active in. `retired_epoch` is the newest epoch whose arena
has been taken away. A page records the epoch of the arena its `func` lives
in (`code_epoch`, written by `publish` under `publish_lock`). Code is valid
iff its epoch is newer than `retired_epoch`.

The check is **not** on the dispatch or page-acquire path. The retiring
worker sweeps the pool (4096 slots) instead, so after the sweep no page
points into the retiring arena, and those paths stay as they are.

### Rotation protocol

1. **Trigger.** A worker sees the active arena over the threshold or gets an
   out-of-memory compile. It wins a CAS on `rotating`. Losers carry on.
2. **Switch.** The spare becomes active; `code_epoch` moves.
3. **Workers follow.** Each worker compares `code_epoch` with its own at the
   top of its loop, rebuilds its `Codegen` on the active arena, drops its
   `pending` count and stores `worker_epoch`. A compile in flight finishes
   into the arena it started in; that code is live and publishes normally
   unless that arena is the one being retired (step 4).
4. **Publish refuses retired code.** `PublishInfo` and the direct publish
   path carry the epoch the code was compiled in. `publish` and the
   redundant-compile revalidation refuse an epoch at or below
   `retired_epoch`, under `publish_lock`. `retired_epoch` moves before the
   sweep, so a publish either lands before the sweep reaches the page (and
   is swept) or is refused.
5. **Sweep.** For each claimed page, under `publish_lock`, if its
   `code_epoch` is retired: `func = jit_no_code`, clear `compiled`,
   `requested`, the compile snapshot and `func_fr1`. `requested` goes too: a
   page that lives through many rotations must not keep every entry point
   it ever had (`jitv2/flush-preserved-entry-point-inflation.md`). The key,
   the FR pin and the frame tables stay.
6. **Kick.** Only after the sweep, post `REQUEST_JIT_RETIRE` to every CPU.
   A CPU that is parked, stopped or idle-parked is outside compiled code and
   counts as having answered.
7. **Reclaim.** The rotating worker does not wait. At the top of its loop it
   checks: when every CPU has answered and every `worker_epoch` has passed
   the retired arena, it recycles the arena in place (protection back to
   initial, pages dropped, positions and seal queue reset), builds the
   memory helpers into it, alone, and clears `rotating`.

**If the active arena fills before the spare exists:** workers stop
compiling (requests are dropped, as on a full queue) until the reclaim. CPUs
run what is published and interpret the rest. Nothing stops.

### Page recycling

The pool never runs out; a claim for a new key with no free slot takes one
from a page nobody is using.

- **Hazard slot.** Each CPU publishes the slot it is on in its own word
  (`current_slot[cpu]`, one cache line per CPU): store the slot, then check
  the frame-table entry again; if it changed, retry. The recycler, under the
  pool lock, skips every slot a CPU has published. At worst one page per CPU
  is pinned, out of 4096. There is no refcount.
- **Clock, not a list.** A page crossing sets the page's `referenced` flag
  (plain store). The recycler's hand clears flags and takes the first slot
  that is unreferenced, not pinned and not scheduled for a compile. A true
  MRU list would put the pool lock on every page crossing.
- **A request carries the tenancy too.** A page's in-flight flag is
  released when its compile ends, but the hot lane's copy of the request can
  still be queued. The worker sets the flag and then checks the request's
  tenancy (`pin_for_compile`), the mirror of the CPU's publish-then-check.
- **Tenancy.** Each slot counts its tenants. The frame-table entry carries
  the count with the slot index, and the fast path compares it, so a stale
  `frame -> slot` entry in any address space misses instead of running the
  new tenant's code. `CompileRequest` and `PublishInfo` carry it too:
  `page_scheduled` is released before the publish, which waits in the seal
  queue holding a raw page pointer, so `publish` refuses a tenancy mismatch
  as it refuses a retired epoch.
- A recycled slot starts clean: `func = jit_no_code`, every bitmap reset,
  new key and generation pointer.

This removes `Claim::Full`, the CPU-led flush, the pool stop/join inside it
and the two-leader problem. `j2 flush`, restore and rollback become "retire
every live arena and wait for the reclaim".

## Hazards

- **`is_runnable` then `func()` are two loads.** With a live sweep a CPU can
  pass `is_runnable` and then load the swept pointer. It is `jit_no_code`,
  which returns `EXEC_FALLBACK`; never null.
- **The compile snapshot.** A swept page with a valid snapshot would have
  its dead code "re-validated" by the redundant-compile skip. Invalidate in
  the sweep; refuse by epoch in the skip.
- **Kick after sweep, never before.** A CPU that answers before the sweep
  finishes can still load an old pointer afterwards.
- **`ParkSlot::park` and `release` mask only `REQUEST_JIT_REMAP`.** Both
  request bits must be masked, or a CPU with the new bit posted parks for
  ever.
- **Hazard store then re-check needs a full barrier** (`SeqCst` store, or a
  swap) between the store and the reload, and the recycler must unlink the
  frame-table entries before it reads the hazard words.
- **Readers of code bytes off the CPU threads**: `j2 pcp`, the html dump,
  `jitv2_pcp_dump`. They must check the page under `publish_lock`, or hold
  off the reclaim while they read.
- **Inline compile (`j2 inline on`) and lockstep** own their `Codegen` and
  run on the CPU thread. They keep the stop-and-reset path.
- **Per-platform recycle.** Linux: `mprotect` + `MADV_DONTNEED`, and the
  hugepage collapse starts over. Windows: `MEM_DECOMMIT`. macOS: check what
  the JIT mapping needs before assuming `mprotect` back to RW.

## What changes, by file

- `paged_memory.rs`: `SharedArena::recycle()`; `position` is no longer
  "never rewound".
- `jitv2.rs`: the arena ring, `code_epoch`, `retired_epoch` on
  `CompileQueue`; the rotation in `worker_loop` in place of
  `run_leader_flush`; `PhysicalCodePage::retire_code`, `code_epoch` and the
  epoch argument to `publish` and the revalidation path. Phase B: tenancy,
  `referenced`, the clock hand, the hazard words, `claim_page` recycling.
- `pool_barrier.rs`: keeps the forced-seal cycle only.
- `smp.rs`: `REQUEST_JIT_RETIRE`, both request bits masked in `park` and
  `release`.
- `mips_exec.rs`: release the request in the run loop and in
  `take_remap_request` (the CPU drops its tracked page; the release is the
  answer). Phase B: publish the hazard slot in the page-crossing path, drop
  the `Claim::Full` flush.
- `codegen.rs`: nothing in emitted code.

## Order of work

Phase A, arenas:

1. [x] `SharedArena::recycle()` with a unit test: allocate, seal, recycle,
   allocate again at offset 0; old pages read back as zero.
2. [x] `code_epoch` on the page, `retire_code`, epoch-checked `publish` and
   revalidation, with unit tests.
3. [x] `REQUEST_JIT_RETIRE` in `smp.rs` and the run loop, `ParkSlot` tests
   extended.
4. [x] The arena ring and the rotation in `worker_loop`; delete
   `run_leader_flush`; trim `pool_barrier`.
5. [x] A loom model of switch / publish / sweep / kick / reclaim.

Phase B, pages:

6. [x] Tenancy in the slot, the frame-table entry, `CompileRequest` and
   `PublishInfo`.
7. [x] Hazard words and `referenced`; the clock recycler in `claim_page`.
8. [x] The pool-exhaustion flush is gone (`Claim::Full` now means "every
   slot is in use": interpret, ask again). `j2 flush`, snapshot restore and
   inline compile mode keep the stop-the-world `flush_from_cpu_thread`,
   which resets the ring. That is by decision (2026-11-24): an explicit
   flush that stops the world is a debugging tool, and a restore needs it.

Phase C, proof:

9. [x] Soak with `IRIS_JIT_ARENA_FLUSH_MB` set to a few MB so rotations
   happen every few seconds, and a small page pool so recycling happens:
   IP24 uniprocessor, two-CPU Octane, four-CPU Tezro, two-node eight-CPU
   Onyx 3000, then the panther boot test.
10. [x] `HACKING.md` (how jitv2 publishes and invalidates code), `debug.md`
    for `j2 status`/`j2 flush`, `jitv2/jit-v2-design.md` §6.3, a
    `clanker/jitv2/` note for the rules that came out of it.

## Results (2026-11-24)

Release build, `lightning,jitv2,jitv2_opcodefusion,gpujit`, on the 2990WX.
Stress settings: `IRIS_JIT_ARENA_FLUSH_MB=8 IRIS_JIT_PAGES=256`. Every guest
on a copy-on-write overlay, IRIX 6.5, to a telnet or serial login, then
`ls -lR` and two `sum` passes over `/usr/bin` whose outputs must agree.

| Guest | CPUs | Arena switches | Arenas retired | Slots recycled | Whole-pool flushes |
|---|---|---|---|---|---|
| Indy (IP24), boot | 1 | 107 | 106 | 403,484 | 0 |
| Tezro (IP53), boot + workload | 4 | 162 | 161 | 1,002,956 | 0 |
| Onyx 3000 (IP35, two nodes), boot + workload | 8 | 127 | 126 | 1,973,707 | 0 |

- A retirement took the code of about 130 pages on average with 8 MB arenas
  and the full page pool (Tezro, arenas-only run: 11,461 pages over 86
  retirements), and 8 to 18 with the 256-page pool as well, because
  recycling had already dropped most of them.
- Default settings, Indy boot to login: 53 s, against 58 s for a build of
  main without this work (one run each; no regression, not a measured gain).
  That boot filled one 512 MiB arena in under two minutes, so the old flush
  fired about that often.
- The Onyx 3000 image does not autoboot: it stops at the PROM's System
  Maintenance Menu. `ptr move` to the Start System icon, then `ptr down`,
  `ptr up` (`ptr click` refuses when the pointer lands a pixel short).
- Panther (Celeron 2955U, two cores, no AVX), `~/iris-test/run-boot-test.sh`
  on a build with the same features: 10 of 10 checks pass at default
  settings in 299 s (serial login 125 s, graphics login 151 s) and with the
  stress settings in 724 s (320 s, 366 s). The 522 s baseline in the host
  notes was a default build with no jitv2, so the times do not compare.
- Windows and Linux desktop sessions were checked by hand by the user.
- Memory helpers: every worker's `Codegen` now calls the helpers built into
  its arena. Before, only the first worker of a pool had them (a
  process-wide latch), and the others emitted the inline guard.
