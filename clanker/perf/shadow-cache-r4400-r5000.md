# Shadow cache for R4400/R5000: what it is for, what it costs

`ShadowCache` (src/cpu/mips_cache_shadow.rs) keeps no line data: memory is the
only store, and its tag arrays exist only to answer CACHE instructions. It now
covers the R4400 and R5000 as well as the R10000, selected by `MODEL`
(`mips_cache_v2::model`, as for `CpuCache`).

Selected at machine build: `[machine] cache = "auto" | "full" | "shadow"`, or
`--cache` (also `iris-bench run --cache`). `auto` means shadow in a jitv2
build, full otherwise. The R10000 only has the shadow model.

## Why jitv2, and not the interpreter

Measured 2026-10-12, full-length `iris-bench run`, `lightning,jitv2`, shadow vs
full. Accuracy identical on all 46 kernels.

| | geomean | biggest wins | losses |
|---|---|---|---|
| R4400 | ×1.33 | mem/copy ×10.8, stream_copy ×8.0, tlb_hit ×2.8 | llsc ×0.63, cache_flush ×0.63, exception ×0.73, tlb_miss ×0.77 |
| R5000 | ×1.50 | adler32 ×21.6, stream_scale ×13.4, tlb_hit ×2.8 | llsc ×0.51, tlb_miss ×0.71, cache_flush ×0.75 |

Compiled loads and stores against a shadow cache need no tag probe (the
tagless inline path, `JitDcGeometry::tagless`).

The losers are the kernels that spend their time in the interpreter
(exceptions, TLB refills, LL/SC, CACHE ops). `ShadowCache::fetch` decodes
into one scratch slot, so the interpreter re-decodes every instruction; the
full model keeps decoded instructions in its lines. On a pure interpreter
build the shadow is about ×0.55–0.6 (measured 2026-10-11, `--quick`), which is
why `auto` keeps the full cache there.

`mem/latency_dram` and `mem/random` (×0.7) did not obviously fit that pattern;
both recovered once the shadow read RAM through the ppmem window (below).

## Decode cache and ppmem window (always on)

`src/cpu/decode_cache.rs`: decoded instructions by physical page, plugged into
`ShadowCache::fetch` only, never touching the shadow's tag arrays. The shadow
also reads and writes RAM through ppmem's window in every build (the window
hand-over is no longer gated on `tcache`). PROM, MMIO and the low-512KB alias
are not RAM in ppmem's bitmap and go through the bus.

Shadow vs full, measured 2026-10-12 (interpreter: `--quick`, best of 2;
jitv2: full-length):

| step | interp R4400 | interp R5000 |
|---|---|---|
| shadow alone | ×0.71 | ×0.53 |
| + decode cache, bus reads | ×0.77 | ×0.57 |
| + ppmem window | ×1.03 | ×0.72 |
| + inlined fetch fast path | **×1.36** | **×1.03** |

jitv2, shadow + both: **×1.52** on both CPUs; accuracy identical.

Lessons, each measured:
- **Decode was not the main cost.** Bus reads were: every raw-compare fetch
  and every data access through the dynamic `BusDevice` call. Profile first.
- **Validate by compare, not by jitv2's write-generation counters.** Through
  the window a compare is one host load, and it measured equal or faster
  (raw/gen ×1.05 R4400, ×1.00 R5000). It also works without jitv2 and outside
  RAM.
- **Inline the hit path, keep the fill cold.** `ShadowCache::fetch` was a call
  taking 34% of R5000 interpreter time; `#[inline(always)]` on a small hit
  path, with `#[inline(never)] #[cold]` on the rest, closed most of the gap.
- **Each decoded page keeps a pointer to its window page** (null for bus
  pages), taken once at page fill; the cache records the region bitmap it was
  filled under and starts over when it changes (remap).

Still lagging: `sys/tlb_miss` (×0.66–0.76 jitv2, ×0.73 R5000 interp).

## What software sees

The same as the full model, by construction and by test:
- `shadow_matches_the_full_models`: geometry, PRId/FIR, ISA, TLB, name.
- `tags_read_back_as_the_full_model_does`: 20,000 random Index_Store_Tag /
  Index_Load_Tag pairs per CPU must return what the full model returns. Both
  go through `r4k_store_taglo` / `r4k_load_taglo`.
- Index ops pick their line the way `CpuCache` does: R4400 direct-mapped, R5000
  way = VA[14], R10000 way = VA[0] (`tag_slot`).

cpu-tests: shadow loses only `cache/hit_inv_discards` (no line data to
discard), the trade jitv2's tcache already made.

## Snapshots

The manifest records `cpu_cache` ("full" | "shadow"). A mismatch is fatal
(`cache_model_mismatch`): a full cache's state holds line data and dirty lines
a shadow has nowhere to put. Legacy manifests mean the old default: full for
R4400/R5000, shadow for the R10000.

## Cost

Two more monomorphised CPUs: `.text` +323 KB (+4.3%, `lightning` iris-bench).

## Seen along the way, not yet chased

- Plain R4400 with the full cache runs 256 of 262 cpu-tests (6 `cache/` tests
  skipped); with the shadow, all 262.
- `umode/load_miss_third` fails with the full cache on both engines and passes
  with the shadow. That suggests a full-model bug.
