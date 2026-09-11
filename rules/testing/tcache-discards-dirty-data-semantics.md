# tcache makes L1-D write-through, so Hit_Invalidate cannot discard

`cpu-tests` `cache/hit_inv_discards` **fails under `--features tcache`** (2 extra
failed checks: 63 instead of the baseline 61) and passes in every non-tcache
build. This is a design consequence, not a bug in the invalidate path — don't go
looking for a missing writeback-suppression in `invalidate_l1d_line`.

## What the test asserts

`cpu-tests/tests/cache/cache.c:135` — `Hit_Invalidate` must throw a dirty line
away *without* writing it back:

```c
*k1 = 0xBA5E0000;  dcache_invalidate_range(k0, 64);   /* memory = BA5E */
*k0 = 0xDEAD0000;                                     /* dirty in L1-D only */
dcache_invalidate_range(k0, 64);                      /* discard, no WB */
CHECK_EQ(*k1, 0xBA5E0000);                            /* memory unchanged */
```

Under tcache both checks report `got 0xDEAD0000 / want 0xBA5E0000`.

## Why

For a transparent line the cache holds **no data of its own**. `Cache::write`
(`src/mips_cache_v2.rs`, the `#[cfg(feature = "tcache")]` arm) sends the store
straight into ppmem — the comment there says it outright: *"The cache stores no
data — the write goes to RAM."* `mark_l1d_dirty` still runs, but the dirty bit
now describes a line whose data already landed in memory.

So by the time `C_HINV` reaches `invalidate_l1d_line(eidx, true, cascade)` —
which correctly clears the tag with no writeback — there is nothing left to
discard. `0xDEAD0000` is already in RAM.

`tc_transparent` is a per-64MB-region bitmap over ordinary RAM, so any normal
scratch buffer is transparent. Only non-transparent (device/uncached) regions
keep true write-back behaviour.

## Consequence

tcache trades away the one behaviour that distinguishes a write-back cache from
a write-through one: dirty data held privately until flushed. Guest code that
*relies* on discarding a dirty line — the R4400 `Hit_Invalidate` idiom for
abandoning speculative writes — will see the write persist. IRIX's normal
flush-then-invalidate paths are unaffected, which is why the emulator still
boots fine with tcache on.

Fixing it would mean giving transparent lines private storage, i.e. undoing the
point of tcache. Treat this failure as **expected for tcache builds** and compare
against 63, not 61.

## Baselines (2026-08-26)

Local, `--cpu r4400`:

| build | checks passed | failed |
|---|---|---|
| interpreter | 2101 | 61 |
| `jitv2` | 2101 | 61 |
| `tcache` | 2099 | **63** |
| `tcache,jitv2` | 2099 | **63** |

The four CI cells, built and run exactly as `.github/workflows/suites.yml` does
(r5000 via `--features r5k` plus `--cpu r5000`):

| cell | checks passed | failed |
|---|---|---|
| r4400 / interp | 2101 | 61 |
| r4400 / jitv2 | 2101 | 61 |
| r5000 / interp | 2071 | 61 |
| r5000 / jitv2 | 2071 | 61 |

61 is uniform across every cell — r5000 skips some R4400-only checks (hence the
lower pass count) but fails the identical fpu set, and jitv2 matches the
interpreter check-for-check. That number is `CPUTEST_BASELINE` in the workflow's
`Result` step; the gate fails on *more* than the baseline and prints a notice on
fewer. Note the historical figures in 29f5602 (121 r4400 / 37 r5000) no longer
hold: FPU rounding/CVT and arithmetic-flag fixes since then changed both.

Those 61 are the long-standing FPU exception-model gaps (no Unimplemented
Operation, inert `FCSR.FS`, trapped results still written, `Cause` accumulates)
documented in `cpu-tests/docs/findings.md` §6-§10 — identical in every config.
