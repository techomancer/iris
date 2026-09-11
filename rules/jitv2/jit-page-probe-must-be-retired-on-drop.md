# The jitv2 dirty-page probe must be retired when its cache dies

This is the root cause of the long-standing `--features jitv2,j2wp` crash (~2 runs in 3,
alternating SIGABRT and SIGSEGV on the `jitv2-compile-N` threads). **It was not the
"racy as hell" cache probe** that commit `81f1b20` self-describes — that raciness produces
*stale answers*, which is by design and harmless. This was a dangling pointer.

## Symptom

```
thread 'jitv2-compile-2' panicked at src/mips_cache_v2.rs:940:
unsafe precondition(s) violated: slice::get_unchecked requires that the index is within the slice
```

`mips_cache_v2.rs:940` is `CacheLevel::get_tag`. Instrumenting it printed:

```
IRISDBG get_tag OOB idx=0 len=0 TAG=iris::mips_cache_v2::L1DTag
```

`len=0`, not a large index — the *slice itself* was empty. That rules out an index/geometry
bug (all the `set | way << shift` arithmetic checks out for both models) and points at the
`&self` being dereferenced not actually being a live cache. Without debug assertions the
same access is a plain SIGSEGV, which is the other half of the alternating failure.

## Cause

`install_jit_page_probe` publishes `&self.cache` as a bare `*const ()` into a process-wide
global, with no lifetime attached. `clear_jit_page_probe`'s own doc says:

> *"Must be called before the cache handed to `install_jit_page_probe` can go away."*

Nothing enforced it. `MipsExecutor` had no `Drop`, and the only `clear` call sites were in
`comp.rs` / `mips_cache_v2.rs` unit tests. In production this is benign — the executor lives
for the process lifetime — but a **test binary creates and drops executors continuously**
while the probe global stays installed. Each dropped executor left the global pointing at a
freed cache for the next test's compile worker to dereference.

`tcache` builds never crashed because `install_jit_page_probe` is gated
`#[cfg(not(feature = "tcache"))]` — which is also why `jitv2,j2wp,tcache` was green while
bare `jitv2,j2wp` was not.

## Fix

A `JitPageProbeGuard` field on `MipsExecutor` holding the published ctx, with a `Drop` that
retires it.

Two deliberate choices:

1. **A guard field, not `impl Drop for MipsExecutor`.** A `Drop` on the executor itself makes
   it illegal to move fields out of one (E0509), which several tests do
   (`mips_exec_test.rs`, `equiv_test.rs` move `exec.core`). The guard confines the drop
   glue to a zero-cost field.
2. **A conditional clear** — `clear_jit_page_probe_if(ctx)`, a `compare_exchange` that nulls
   the global only if it still holds *this* ctx. A later executor may have installed its own
   probe before this one died; an unconditional clear would silently disable that live
   executor's dirty-page check instead of merely retiring the dead one's. Silently disabling
   the check is worse than the crash, because it degrades to wrong compiles rather than a
   loud abort.

## Result

`jitv2,j2wp` went from crashing ~2 runs in 3 to **5 consecutive clean runs, 832/832**.

## Lesson

A global that stores a raw pointer to something owned elsewhere needs an owner-side `Drop`,
not a doc comment saying callers must remember. The comment was present and correct here,
and was still not enough — the same "every caller must remember" shape as
`rules/testing/cp0-status-writes-must-resync-privilege-state.md`.

Also: when a UB report says the *length* is zero rather than the index being large, stop
auditing index arithmetic and go looking for a dead object.
