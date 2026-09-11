# `SC` succeeds on the LLbit alone — `LLAddr` is not a gate

`exec_sc`/`exec_scd` used to require the store's physical address to match
`LLAddr`, failing the SC on a mismatch. That is not R4000 behaviour.

## What the hardware does

`LLAddr` is a **diagnostic** register: `LL` writes it, software can read it, and
it plays no part in `SC`'s success condition. The reservation is the **LLbit**.

MAME's `r4000.cpp` agrees exactly:

```c
case 0x38: // SC
    if (m_ll_active) { ... m_r[RTREG] = 1 ... m_ll_active = false; }
    else m_r[RTREG] = 0;
```

It writes `m_cp0[CP0_LLAddr]` at LL time purely so the guest can read it back,
and never compares it at SC. IRIS now does the same (still writing `LLAddr` at
LL, for the same reason).

## Why the extra check was wrong

It can only ever make SC fail where hardware succeeds. `LLAddr` is a single
global register while the LLbit is the real reservation, and every
architectural reason for an SC to fail already clears the LLbit:

- any exception, via all three `handle_exception*` wrappers (including
  `handle_exception_at`, the jitv2 path);
- `ERET`;
- a **coherent** (CACHE-op) invalidation of the line — `invalidate_l1d_line`
  calls `check_and_clear_llbit` only when `coherent` is set, deliberately not on
  capacity evictions ("on a uniprocessor R4000 there are no external snoops").

## Why it matters more than it looks

IRIX builds **every** kernel atomic on LL/SC loops (`kern/sys/atomic_ops.h`),
including the `mutex_bitlock` on `k_flags` that `kern/sys/kthread.h:74` says
guards *"all locking in the sync routines"* — i.e. sleep and wakeup. An SC that
spuriously fails there is exactly the shape of "processes sleep forever while
everything else runs".

## LL/SC are interpreter-only

`analyzer.rs` classifies `OP_LL | OP_LLD | OP_SC | OP_SCD` as
`Classify::Excluded`, and nothing in `src/jitv2/` touches `llbit` or `lladdr`
(both greps empty). So the semantics are identical under either engine, and the
JIT cannot corrupt a reservation directly — it can only change *interleaving*,
i.e. how much runs between an LL and its SC and where interrupts land.

## `ll stats`: the instrument

`--features llstats` adds a per-`LLAddr` histogram, queryable with `ll` /
`ll stats` and resettable with `ll clear`. Columns: `ll`, `sc_ok`, `sc_fail`,
`cur_run`, `max_run` — sorted worst `max_run` first.

`max_run` (longest run of consecutive SC failures) is the column that answers
"is a kernel LL/SC loop failing to make progress?". A couple of retries is
normal; hundreds is a stuck lock. Rows are keyed by physical address, so a row
matching a hung thread's WCHAN links the stuck lock to the sleeping thread.

**Its own feature, deliberately not part of `developer`.** `developer` and
`lightning` are mutually exclusive (`src/lib.rs` `compile_error!`), and the
failure this exists to diagnose so far reproduces only at full JIT speed — a
`developer`-gated histogram would have been unusable for its own purpose.
Implemented as a fixed-size direct-mapped table (1024 slots, ~40 KB, no hashing,
no allocation) so it is cheap enough to carry in a `lightning` build. On a slot
collision the row is reset and re-keyed rather than summing two addresses:
collisions lose history but never fabricate it.

## Gotcha: `execute_command` has early returns that shadow the match

`ll` already existed as an `if actual_cmd == "ll" { ... return Ok(()); }` early
return (live `llbit`/`lladdr` state), roughly 700 lines *above* the big
`match cmd { ... }`. A new `"ll" => { ... }` arm added to that match is
**unreachable dead code**, and nothing warns: the arm compiles, the help line
registers, and the binary silently lacks the feature.

That cost real time here, and the misdiagnosis was instructive: the strings were
missing from the binary, cargo kept reporting "Finished in 0.3s", and the
obvious-looking conclusion was a stale-build/caching problem. It was not. The
give-away was dumping the binary's help text and finding **two** `ll`
registrations — one from the early return's table entry, one from the new arm.

So: before adding a monitor command, `grep -n '"<name>"' src/mips_exec.rs` and
check for `if actual_cmd == "<name>"` as well as a match arm. The early return
now gates on `actual_args.is_empty()` so `ll stats`/`ll clear` fall through,
with a comment saying why.

A general form of the same lesson: when a freshly built binary appears not to
contain code you just wrote, check reachability before blaming the build.
