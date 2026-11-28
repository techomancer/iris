# The dispatch gate must not write shared per-page state on every arrival

**Status: implemented 2026-11-28 (branch `perf/jitv2-smallhostopts`).**

**Keywords:** jitv2,dispatch-gate,cache-line,ping-pong,small-host,2-core,hot-lane,translation

## Symptom

A `jitv2` build was slower than the plain interpreter on a two-physical-core
host (i3-4130, 7.7 GiB) for code the JIT could not compile — the case where
the dispatch gate is on but nothing comes back from the compile pool.

Measured with the bare-metal bench suite (`bench/`), one guest CPU, the
emulator pinned to `taskset -c 0,1` (one logical CPU per physical core), the
suite at `time_pct=30`:

| build | gate | suite wall time |
|---|---|---|
| interpreter | off | 9.9 s |
| `jitv2`, `IRIS_JIT_DISPATCH=off` | off (pure interpreter in a JIT build) | 10.9 s |
| `jitv2`, `j2 min-instrs 100000` | **on, nothing compiles** | **18.9 s** |

The gate cost **~75%** of the interpreter's throughput on that workload. The
same build with the gate on and the JIT working was 2.7 s — so the tax was
not the JIT, it was the gate.

`IRIS_JIT_DISPATCH=off` was the control that isolated it: same binary, same
Cranelift, same compile pool, only the gate off.

## Cause

Every arrival at a page whose code is not published yet ran the whole request
path, and that path wrote **five shared atomic read-modify-writes per
arrival** into the `PhysicalCodePage` — the struct the compile worker is
reading and writing at the same time:

1. `mark_requested` — `fetch_or` on the `requested` bitmap.
2. `take_fr_repin` — `swap` on `fr_repin`, then `swap` on `fr_unpinned`.
3. `try_schedule_page` — `swap` on `page_scheduled`.
4. `note_waiting_arrival` — `fetch_add` on `hot_wait`.

Plus a `promoted.swap` on the promotion path.

None of them changed anything in the steady state. The bit was already set,
the request was already queued, the FR pin was already right, the count was
already over the threshold — and every one of them still wrote. Each write
invalidated the compile worker's copy of that cache line, and the worker
invalidated the CPU's copy when it published (it clears `requested`,
`page_scheduled`, `hot_wait` and `promoted` — `clear_scheduled`). The two
threads ping-ponged the line for every arrival.

Two more costs sat on the same path:

- The gate translated the fetch (`nanotlb_translate`) to find the code page,
  and every interpreter fallback then translated the *same* address again
  through `fetch_instr`. Two translations per interpreted instruction.
- The gate's per-arrival work ran even when the page's request was already
  in flight, which is the whole "waiting for the compile" window.

## Fix

**The repeat-arrival path is now read-only.**

- `mark_requested` (`src/cpu/jitv2/jitv2.rs`) loads the bitmap word and only
  does the `fetch_or` when the bit is clear. The `fetch_or` still does the
  actual set, so two CPUs racing on a fresh offset both end up with it
  requested.
- `try_schedule_page` loads `page_scheduled` first and returns `false`
  without writing when a request is already in flight. The `swap` below it
  is still the test-and-set that picks the one caller that sends.
- `take_fr_repin` returns the page's pinned FR mode without writing when no
  re-pin is outstanding and the page is pinned — the steady state. A re-pin
  requested between the load and the return is not lost: it stays outstanding
  and the next arrival consumes it.
- `note_waiting_arrivals(n)` counts a batch of arrivals. The dispatch gate
  counts arrivals **per CPU** in a field of its own executor
  (`jitv2_arrival_page` / `jitv2_arrival_count`, plain fields — the executor
  belongs to one thread) and folds the total into the page's shared counter
  every `HOT_ARRIVAL_BATCH` (16) arrivals. The total counted is the same and
  the promotion threshold (`HOT_PROMOTE_ARRIVALS`, 256) is unchanged; the
  shared line is written once per batch instead of once per arrival.
- The count belongs to the page it was made at, so the count for the page the
  CPU leaves is folded into that page when the fetch moves on
  (`jitv2_fold_arrivals`). Without that fold a page the CPU visits in short
  bursts — alternating between two pages, say — would never accumulate a
  batch, its counter would stay at zero and it would never be promoted: the
  hot lane would silently stop working.
- The promotion check is separate from the counting
  (`PhysicalCodePage::take_hot_promotion`): only the CPU that is *at* the
  page can put that page's request on the hot lane, so the check is made at
  every arrival at the page (two loads of the page's own cache line) and the
  `promoted` flag is swapped only when the page is actually due.
- The gate's fetch translation is reused by the interpreter fallback:
  `step_int_pretranslated` / `fetch_instr_pretranslated`
  (`src/cpu/mips_exec.rs`). The translation is only reused when `core.pc`
  still names the same address after `step_preamble!` — an interrupt
  delivered by the preamble moves the PC — so an instruction can never run
  against a stale translation. The JIT's own `EXEC_FALLBACK` path does *not*
  reuse it: compiled code may have moved the PC.

Net effect on the hot path: from five shared atomic RMWs per arrival to
**one shared write per 16 arrivals** (plus one when the CPU leaves a page),
plus no second fetch translation.

**One bounded deviation from the old count.** The per-CPU batch belongs to
the wait it was counted in, but a batch can straddle a wait boundary: if the
page's request completes while the CPU still holds part of a batch, those
arrivals are folded into the *next* wait, so the page can be promoted at
most `HOT_ARRIVAL_BATCH - 1` (15) arrivals early out of
`HOT_PROMOTE_ARRIVALS` (256). The threshold is a heuristic — how hard a page
is being executed — so a few arrivals either way do not change what it is
for, and the published code was the same in the coverage run below.

## Rule

The dispatch gate runs on every instruction. A shared per-page word that
does not change must not be written on that path: the compile worker owns
the same cache line, and every write is a round trip between the two
threads. Test-then-set where the value is usually already right, count
per-CPU and fold in batches, and reuse a translation the caller already
paid for.

## Measurements

Host: Intel i3-4130, 2 physical cores / 4 logical, 7.7 GiB RAM, Linux
x86-64, `taskset -c 0,1`. Guest: the bare-metal bench suite, one guest CPU.
Build: `cargo build --release --features jitv2`.

Per-kernel guest MIPS is the metric: the harness calibrates each kernel's
iteration count toward a target *time*, so the retired-instruction count in
the window tracks the speed, while the suite's wall time tracks it only
loosely.

**Measuring this needs a fast monitor client.** The suite starts the guest
the moment the emulator is up, so a monitor command that lands late is
already too late: the kernels it would have caught get compiled and run at
JIT speed instead of interpreter speed, and the two builds then look
*different* when they are not. `tools/mon.py` spends ~40 ms in Python's
start-up, which is a whole group; a small C client that retries every
500 µs lands the command in time, and the numbers become stable.

### The gate's tax, isolated (`j2 min-instrs 100000`: the gate is on, every
compile is declined)

Suite wall time, three runs each, `time_pct=30`:

| build | run 1 | run 2 | run 3 | median |
|---|---|---|---|---|
| before | 18.39 s | 19.17 s | 18.46 s | 18.46 s |
| after | 15.32 s | 15.45 s | 15.35 s | **15.35 s** (−17%) |

Per kernel (guest MIPS, median of three runs), the gate's tax is gone from
every kernel — the geomean over all 46 is **×1.197**:

| kernel | before | after |
|---|---|---|
| int/branch | 21.7 | 28.7 |
| int/bitops | 30.4 | 39.2 |
| mem/latency_l1 | 19.5 | 28.5 |
| mem/latency_l2 | 18.9 | 28.6 |
| mem/latency_dram | 11.2 | 13.7 |
| vid/yuv2rgb | 30.6 | 38.2 |
| codec/adler32 | 25.4 | 33.8 |
| codec/rle | 21.5 | 29.1 |

### The control: the gate off

`IRIS_JIT_DISPATCH=off` (the same binary, the gate never entered) is
unchanged: 10.80 s before, 10.76 s after, per-kernel geomean ×1.002. The
refactor of the interpreter's fetch path (`step_int_finish`,
`fetch_instr_translated`) costs nothing — both halves are
`#[inline(always)]`, because as plain calls they cost the dispatch-off
path ~8%.

### Coverage is unchanged

With the gate on and the JIT working, the published code is the same before
and after: the same 14 pages, the same 14 published entries, the same
instruction counts (5, 55, 417, 699, 845, 954, 1004, 1014 ×3, 1017, 1019,
1020, 1021) and the same 11,098 instructions compiled, on the whole suite
at `time_pct=300`. Only the churn differs — 297 vs 285 functions compiled,
259 vs 249 checked, 241 vs 229 recompiled — which is the order the compiles
happened in, not what was published.

**A short measurement window makes the JIT look like it lost coverage.** At
`time_pct=30` a kernel's window is a few milliseconds, and whether a
compile lands inside it is a race — the old and new builds take opposite
sides of it on different kernels from run to run. That is a measurement
artifact, not a behaviour change: raise the window and the two builds
publish the same code.

## Not done

- The always-on tax is still there: a `jitv2` build with the gate *off* is
  ~10% slower than a `tcache` build with no JIT in it (10.9 s vs 9.9 s on
  the same suite). That is not the gate: `tc_bump_gen`
  (`src/cpu/mips_cache_v2.rs`) bumps a per-page generation counter on
  **every store**, because a data write to a code page invalidates the code
  compiled from it. The bump is `jitv2`-only, but it is paid by every store
  in every `jitv2` build, whether or not anything was ever compiled. Worth
  its own change: the generation only matters for a page that has code or a
  pending request.
- The gate still translates the fetch even when the page has no code at all
  and no request pending. A cheaper "is there anything to dispatch here?"
  check would save that too, but it changes the gate's shape more than this
  change wanted to.
