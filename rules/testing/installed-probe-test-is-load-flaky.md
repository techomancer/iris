# `installed_probe_round_trips_to_the_compile_side` is load-flaky

Observed 2026-09-19 under `--features jitv2,j2wp`.

`mips_cache_v2::tests::installed_probe_round_trips_to_the_compile_side` fails
intermittently at its third assertion (`mips_cache_v2.rs:4745`, "the dirty line
must be visible through the erased (ctx, fn) pair the worker actually calls").

**It is not caused by whatever you just changed.** Evidence:

- 1 failure in 5 full-suite runs while the machine was saturated by concurrent
  release builds and a 77k-region corpus compile.
- 10/10 clean full-suite runs on the same working tree once the machine was
  idle.
- 10/10 clean when the cache tests are run alone
  (`--lib mips_cache_v2`), in any load condition.

So: intermittent, load-dependent, and only reproducible under the full parallel
suite. It surfaced during unrelated work and cost a stash/bisect cycle to rule
out — hence this note.

## Why the existing lock doesn't cover it

The test takes `probe_global_lock()`, which delegates to
`jitv2::probe_test_lock()` so that `comp.rs`'s probe tests and these share one
mutex. That correctly serialises **probe-pointer installation**. The failing
assertion is about **cache dirty state** — `make_cache` + `full_flush` + a
write, then reading dirtiness back through the probe — which that lock does not
protect. Under enough parallel load the window between the write and the read
is wide enough for something else to perturb the shared state.

Not yet root-caused. If you fix it, the fix is almost certainly about isolating
cache state (or widening what the lock covers), not about the probe indirection
the test is nominally exercising.
