# jitv2_lockstep must disable the inline load/store path

## The gap

`jitv2_lockstep`'s **entire** memory verification lives in the callout
wrappers. `jit_read8/16/32/64` and `jit_write8/16/32/64` in `mips_exec.rs` are
cfg-switched: under lockstep they divert to `MipsExecutor::lockstep_jit_read` /
`lockstep_jit_write`, which never touch the bus — they re-translate the VA
independently and compare address/phys/value against `core.lockstep_mem`, the
triple the interpreter captured when it ran the same instruction.

The inline fast path (`emit_inline_mem_guard` → `emit_mem_read_split` /
`emit_mem_write_split`) **deliberately bypasses those wrappers** — its own
comment says so, and it stores `core.jit_mem_exc` by hand precisely because it
never calls the wrapper that would normally do it.

Nothing connected the two. `jitv2_lockstep = ["jitv2"]`, `INLINE_MEM_ENABLED`
defaulted **on**, and lockstep's own inline-compile path
(`mips_exec.rs`, `codegen.dc_geometry = self.cache.jit_dc_geometry()`) stamps
the *real* cache geometry — so on an R4400 + `nutlb` build `geom.supported` is
genuinely true and the inline path really was emitted.

**Consequence:** every load/store hitting nutlb + the L1-D tag (~82% of loads,
per docs/jit-inline-memory.md) executed completely unverified. Lockstep
silently checked only the slow-path minority while reporting itself as on —
worse than not running it, since address-computation bugs on the hot path are
exactly what these hooks exist to catch.

R5000 builds were unaffected (`jit_dc_geometry` returns `unsupported()` for the
2-way L1-D). (At the time this was written `nutlb` was a feature and builds
without it were also unaffected; it is now unconditional, so only the R5000
geometry exemption remains.)

## The fix

Gated inside `codegen::inline_mem_enabled()` itself:

```rust
if cfg!(feature = "jitv2_lockstep") { return false; }
```

Deliberately *not* at the two `emit_inline_mem_guard` call sites — putting it in
the shared accessor means the env var, the monitor toggle, and any future call
site are all covered by construction. `j2 inline_mem on` now reports that it is
forced off rather than accepting the toggle and silently ignoring it.

Cost is nil: a lockstep build already runs every instruction through the
interpreter twice.

## Verifying a gate like this actually bites

The unit test (`lockstep_forces_inline_mem_off`) only proves the flag. The one
that matters is the **negative control**: compile `lw`/`sw` with
`dc_geometry.supported` forced true and count `INLINE_MEM_EMITTED`.

- without lockstep: **delta = 2** (both accesses inline)
- with lockstep: **delta = 0**

Without that control the lockstep-side assertion is vacuous — it would pass just
as happily if the inline path were globally broken, or if the synthetic test's
`PassthroughCache` had declined for unrelated reasons (which it does:
`jit_dc_geometry` → `unsupported()`, which is why the first probe I wrote showed
`declined=2` and proved nothing). `lockstep_emits_no_inline_mem_even_with_supported_geometry`
forces the geometry so the decline can only come from the lockstep gate.

## Feature interactions

Cargo features are additive — an incompatible feature can only be *rejected*,
never unset. So:

- `jitv2_lockstep = ["jitv2", "developer"]` — pulled in additively. Lockstep's
  divergence path breaks into the monitor (`EXEC_BREAKPOINT`) and its reports
  are only actionable with `dt`/`r`/`d pc`.
- `lightning`, `opcodefusion`, `jitv2_opcodefusion` — refused via
  `compile_error!` in `src/lib.rs`. A fused pair's second instruction is never
  independently fetched/decoded/dispatched, so lockstep's per-instruction
  interpreter reference has nothing to bracket it against: the JIT would run it
  while the comparison silently skipped it. (codegen's fusion sites already
  self-disable under lockstep, so enabling the feature would be a silent no-op
  that misrepresents the build.)

## Pre-existing lockstep failures (NOT caused by this)

Verified against the parent commit in a scratch worktree with
`--features jitv2_lockstep,developer`:

- `full_mutex_lock_cas_call_chain_matches_interpreter` — SIGABRT in
  `jit_hooks_not_installed_write64`. Reproduces on the parent commit; unrelated
  to inline memory (that test uses `PassthroughCache`, which reports the
  geometry unsupported, so it never took the inline path). Its `build(jit)`
  closure only calls `install_jit_hooks()` on the `jit` branch — adding it to
  both does **not** fix the abort, so the real cause is still open.
- 12 CVT/conversion `equiv_test` failures (`jit=0x0` vs a real interp value) —
  the already-known jitv2+jitv2_lockstep conversion bug.

Before this change the whole lockstep suite died with SIGSEGV; it now runs to
completion (801 passed) with the above known failures.
