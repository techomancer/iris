# jitv2: the dirty-cache blind spot, and the probe that closes it

## The hole

The jitv2 compile worker builds a region from a 4KB page snapshot it reads
**off the bus** (`comp::handle_request` / `handle_request_deferred`). The bus
shows it RAM. The guest CPU's view is RAM *overlaid with its own dirty cache
lines*.

The per-page generation counter — the mechanism that normally invalidates
compiled code when the guest rewrites it — is bumped by **bus writes**. A store
that retires entirely inside L1-D bumps nothing. So:

1. Guest stores new code into a page. The store lands in L1-D. No gen bump.
2. Compile worker snapshots that page off the bus, sees the **pre-store** bytes.
3. It compiles them and publishes. `publish`'s `gen_snap` re-check passes,
   because the generation genuinely never moved.
4. The guest now executes stale compiled code, indefinitely.

The PROM does not hit this: it flushes caches after relocating, so its writes
reach RAM before the JIT ever looks. IRIX does, because it relies on the
R4400's L1I/L1D coherency and inclusive L2 to make freshly written code visible
to the fetch path without any explicit writeback. That is correct on real
hardware and correct in our CPU emulation — it is only invisible to the
*compiler*, which is the one component that reads memory asynchronously and has
no view into the cache.

## What was tried before

- **Cache flushes** for every page the compiler visits. Correct, and
  catastrophic for guest performance. Not a serious option.
- **`tcache`** (transparent cache: the cache reads/writes RAM through the ppmem
  window, so there is no hidden dirty data by construction). Correct and
  structural, but costs roughly 10–20% across the benchmark suite —
  whetstone 434→416, dhrystone 303k→263k, irisbench 3.12→2.98 under
  `lightning,rex-jit,nutlb,j2wp`.

## What this is

`MipsCache::jit_page_has_dirty_lines(page_base)` — the compile worker asks the
cache directly, once per compile attempt, "do you hold dirty lines for this
page?" If yes, it aborts: publishes nothing, denylists nothing. The offset is
retried on its next arrival, by which point the lines have usually drained to
RAM on their own.

Cost is one short tag scan on the compile thread, per *compile*, not per memory
access. A 4KB page maps to **contiguous** set ranges at both levels because
both are indexed by low physical address bits — on R4400 that is 256 L1-D tags
plus 32 L2 tags. No hashing, no full-cache walk.

## Why two probes make this watertight

The probe runs **twice**: once before the page snapshot (cheap early-out) and
again immediately before `publish`. That pair, combined with the `gen_snap`
check `publish` already performed, closes the window completely.

A store that happens during the compile is in exactly one of two states, and
both are caught:

- **Still in the cache** → its line is dirty → the pre-publish probe sees it
  and the entry is dropped.
- **Reached RAM** → it got there via `writeback_l1d_line` or
  `writeback_l2_line`, and *both* drain through `BusDevice::write_block`.
  `mem.rs`'s `write_block` bumps the page generation counter, so
  `publish`'s existing `gen_snap != current_gen()` check rejects it.

There is no third state — dirty data is either in a cache or in RAM, and the
only transition between them bumps the generation. **Partial writebacks do not
weaken this**: any writeback at all bumps the counter, regardless of how much
of the line it moved. That was the open question when the second probe was
proposed, and `write_block` bumping unconditionally is what resolves it.

The same argument covers the **publish→dispatch** window, so no third check is
needed: a published entry carries `gen_snap` and dispatch validates against it,
and a store arriving after publish must eventually write back — bumping the
generation and invalidating the entry before its data could be observed stale.

Cost is one extra tag scan per *published* region, on the compile thread. The
CPU thread stays untouched, which is the whole reason for doing it this way
rather than tracking dirty state as the guest runs (per-line dirty bits would
need hooks on every fill/writeback/eviction/CACHE-op path, and a missed hook
fails *closed* — the page silently never compiles again).

## The single-probe version was only a mitigation

Before the pre-publish re-check existed, the probe was explicitly *not* a
correctness barrier — a store landing after the scan but before publish would
still be baked in. That earlier framing is preserved below because the
individual scan is still an unsynchronised racy read; what changed is that the
*pair* of checks is now exhaustive, so the race no longer has an escape.

## The individual scan is still unsynchronised

No lock, no atomics on the tags, no ordering. It is a plain racy read of tag
words from the compile thread while the CPU thread mutates them. That is an
accepted trade, not an oversight:

- **False `true`** (saw a dirty tag being cleaned right now): free. One skipped
  compile, retried later. Pure lost work, zero correctness effect.
- **False `false`** (a store lands just after we looked): caught by the
  *other* end of the pair — either the pre-publish probe sees the still-dirty
  line, or the writeback that cleaned it bumped the generation. A single scan
  answering stale is not a correctness problem; the pair is what matters.

## The trap: L2-first short-circuiting is wrong

The obvious implementation — "check L2; if the line is there and clean, we're
done" — **misses the exact case this exists for.**

On R4400 the L2 is inclusive, but a store marks only the **L1-D** tag dirty.
L2 stays `CLEAN_EXCLUSIVE`; it is `writeback_l1d_line` that later promotes it to
`DIRTY_EXCLUSIVE` (`mips_cache_v2.rs`, the `L2_CS_CLEAN_EXCLUSIVE =>
L2_CS_DIRTY_EXCLUSIVE` transition). So a freshly self-modified page looks like:

    RAM:  stale        L2: CLEAN (stale)        L1-D: DIRTY (current)

Both levels must be scanned, and L2 must not gate the L1-D scan. L2 is checked
in addition, to catch data already written back out of L1-D but still sitting
ahead of RAM. Pinned by
`probe_sees_a_store_still_sitting_in_l1d` and
`probe_sees_dirty_data_that_reached_l2_but_not_ram`.

The second trap: compare **full line addresses**, not set indices. Another
page aliasing into the same sets is not this page's data, and a probe that
ignored the tag would veto compiles constantly
(`probe_ignores_dirty_lines_belonging_to_other_pages`).

## Wiring

- `MipsCache::jit_page_has_dirty_lines` — the scan. Default `false`
  (`PassthroughCache` hides nothing from the bus).
- `MipsExecutor::install_jit_page_probe` — publishes a type-erased
  `(ctx, fn)` pair, called from `install_jit_mem_ptrs`, which already re-runs
  on everything that can move the cache (snapshot restore, reconfigure).
  A pair rather than a trait object because `MipsCache` is const-generic over
  geometry and the executor owns its cache by value.
- `jitv2::jit_page_has_dirty_lines` — the read side, a process-global like
  `MIN_CALLS_BEFORE_COMPILE`. Null ctx = no probe = pre-probe behaviour.
- Consumed in **both** compile designs, at two kinds of site:

  **Before the page read** (cheap early-out — no point paying for a snapshot
  about to be discarded). Three sites, all gated: `old_impl`'s
  `handle_request` and `handle_request_deferred`, and `new_impl`'s
  `prepare_multi_entry_compile` (the shared read path behind both of *its*
  entry points).

  **Immediately before `publish`** (the check that makes this watertight —
  see above). Four sites, all gated, one per publish call: `handle_request`
  and `publish_all` in each design. The deferred path's window is the wider
  one — entries sit in the seal queue after compiling.

  Verify coverage by *module*, not by grep count:
  `awk '/^mod old_impl/{m="old"} /^mod new_impl/{m="new"} /\.publish\(/{print m" "NR}' src/jitv2/comp.rs`
  and check every publish line has a probe line just above it.

  **This was originally wrong and is worth remembering.** Both gates first went
  into `old_impl`, because `comp.rs` contains two same-named functions per
  design and a grep for the page-read site returned two hits that looked like
  one per design. They were both in `old_impl`. `j2wp` builds — the fast ones,
  the ones actually used — had no probe at all. Nothing caught it: every unit
  test passed and IRIX booted. The only symptom was `PageDirtyInCache` reading
  zero forever. `a_dirty_page_is_never_compiled` now exists in both designs'
  test modules to fail loudly if either path stops consulting the probe.

  For the *pre-publish* guards, reachability was confirmed by mutation
  instead: forcing them to always veto makes every publishing test in the
  module fail, which proves they sit on the real publish path. A behavioural
  test was tried first and abandoned — it drove the process-global probe while
  `cargo test` runs tests in parallel, so other tests kept clearing the global
  out from under it. That is not fixable by scoping the stub; **do not write
  tests that depend on exclusive ownership of the probe global.** Tests that
  merely install a stub must take `jitv2::probe_test_lock()` (one shared
  mutex — `comp.rs` and `mips_cache_v2.rs` previously had separate ones, which
  guarded nothing).

Under `tcache` the **entire mechanism is compiled out** — trait method, impl,
global, executor install, and both `comp` call sites are all
`#[cfg(not(feature = "tcache"))]`. That feature closes the hole by
construction: a transparent cache writes straight through the ppmem window, so
RAM is current the moment a store retires. Leaving the check in would be a
global load on the compile path that can only ever answer false.

## Note on units

`old_impl` counts one abort per *entry offset*; `new_impl` is whole-page
multi-entry, so one abort rejects every candidate offset on the page and the
counter reads one per *page-compile*. Consistent within each design, not
comparable across them.

## Observability

Aborts are counted as `RejectReason::PageDirtyInCache`, visible in `j2 stats`
on a `developer` build.

Two reporting traps, both already hit:

- `j2 stats` used to iterate a **hardcoded list** of reasons, so a new variant
  was invisible however often it fired — the only tell was `failed` not
  matching the sum of the printed buckets (`3120 failed` vs `3119 entry
  excluded`, one probe abort hiding in the gap). The loop is now driven by
  `RejectReason::ALL`.
- A dirty-page abort is a **deferral, not a failure** — the same offsets
  recompile fine once the lines drain. It is deliberately excluded from
  `failed_compiles`, so it does not make codegen look like it is regressing.
  The breakdown therefore gates on the buckets rather than on `failed`. A steadily climbing count means the guest is writing
code into a page it keeps executing from — expected during module load or
relocation, suspicious if it never settles.
