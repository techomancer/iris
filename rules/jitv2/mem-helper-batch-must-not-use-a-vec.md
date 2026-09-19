# The memory-helper batch must not be collected in a `Vec`

`Codegen::emit_mem_helpers` builds exactly `MEM_HELPER_COUNT` (12) helpers and
holds their `FuncId`s until `finalize_definitions()` returns. That collection
**must be a fixed array**, not a `Vec`.

With a `Vec`, `iris --jitv2-threads 16 --cpu r5000` under `developer` panicked
on `start` roughly 6 times in 10:

```
thread 'jitv2-compile-N' panicked at cranelift-jit-0.134.3/src/backend.rs:281:
function must be compiled before it can be finalized
```

With a fixed array: **0 failures in 25 runs**, then 0 in 15 more.

## Symptom

The batch would come back holding ids the module never handed out — twelve
consecutive values in the 32000s when `declare_anonymous_function` had returned
`0..=11`:

```
jitv2 BUG: built[0] = funcid32514, but only 12 helpers were declared
jitv2 BUG: built[1] = funcid32515, ...
```

`finalize_definitions()` then walked ids with no definition behind them.

## How to investigate this (and how not to)

Three separate dead ends cost the most time here, all of them worth avoiding
next time:

**Printing hides it.** An `eprintln!` inside the build loop takes the failure
rate from ~6/10 to **0/10**. Every "the fix works" reading taken with live
logging in that function is worthless. Record into preallocated atomics and dump
from the panic hook instead (`crash_diag`'s hook is the natural place).

**Bit-packed trace records lie.** A first attempt packed `(call, phase, value)`
into one `u64` with overlapping shifts — `call` at bit 40, `phase` at bit 32 —
so a large call id bled straight into the value field. It reported corruption it
had caused itself, and sent the investigation after a nonexistent cross-thread
writer for several rounds. Use parallel arrays; they cannot overlap.

**ASan and a hardware watchpoint both said the memory was clean.** ASan
(instrument only the `iris` crate — a full-graph build breaks `gdbstub` against
uninstrumented std) reported no error at all. A gdb hardware watchpoint on the
buffer fired exactly once, and the writer was our own `Vec::push`. Both were
correct: there was never a foreign write.

The thing that actually settled it was the dumbest possible probe — assert in
code that every id is `< 12`, print and move on. No formatting, no packing, no
timing perturbation.

## Why a fixed array is right here regardless

The size is a compile-time constant, so a `Vec` buys nothing and costs a heap
allocation plus a reallocation sitting in the middle of the one loop whose
contents have to be trusted. Indexing by `helper.index()` also means a helper
that declines to build leaves a `None` hole rather than shifting every entry
after it, which the old `push`-based code silently relied on not happening.

## Related

- `docs/jit-inline-memory.md` §9, §11 — why `finalize_definitions()` may only be
  called before any region is compiled into the arena.
- `rules/jitv2/` — `emit_mem_helpers` is called from three places (startup,
  worker entry, flush leader); all three are guarded by `Codegen::mem_helpers`
  already being populated.
