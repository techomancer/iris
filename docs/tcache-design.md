# tcache — transparent cache

Status: **implemented (R4400 path), measured ~2% below baseline. Off by
default.** Not dead — v3 recovered most of an earlier, much worse result that
came from an implementation bug, so the remaining gap is small. See §0.

Companion to `docs/ppmem-design.md`; depends on ppmem being available.

---

## 0. Result: close to parity, still not a win

`iris-bench run`, same host, back to back:

| Build | MIPS | DMIPS | Accuracy |
|---|---|---|---|
| baseline (real cache) | **40.3** | 48.8 | 100% |
| ppmem only | 39.4 | 45.3 | 100% |
| tcache v1 — gate per access, **L2 not actually bypassed** | 37.7 | 45.7 | 100% |
| tcache v2 — gate hoisted into `tag.backed` | 38.2 | 47.0 | 100% |
| tcache v3 — **L2 genuinely bypassed** | **39.6** | 46.1 | 100% |

Still ~2% short of baseline, so it stays off — but the trajectory says the
remaining gap is small and the earlier "this idea is dead" conclusion was
premature and based on a broken implementation.

### What v1 got wrong (and the analysis that followed)

**v1 did not bypass L2 at all.** The `if !transparent` guard wrapped only the
L2→L1D copy. Everything above it still ran: the L2 tag probe, and
`fill_l2_line` on a miss pulling 128 bytes from memory. L2 writeback to memory
was untouched. So v1 skipped one 16-byte copy and left both 128-byte ones
intact — it bypassed L2 *for L1D reads*, nothing more. Calling that
"transparent" was wrong, and it explains the poor v1 number far better than any
theory about the idea itself.

**A retracted claim.** An earlier revision of this document blamed the residual
on host cache locality — that reading through `tc_base + phys` touches a large
working set while `dc.data` stays hot. That argument does not hold: the guest
RAM being read *is* the same memory the cache array would have been filled
from, so the host sees comparable traffic either way. The real cause was the
un-bypassed L2 copies above.

### What v3 actually does

- **L1D miss on a transparent line skips L2 entirely** — no probe-and-fill, no
  128-byte read. The VCE check still runs against whatever L2 already holds, so
  aliasing is still detected; L2 simply never gets *populated* on a data miss.
- **`fill_l2_line_for(.., fill_instructions)`** records why a line was filled.
  A data-origin fill of a transparent line reads nothing and stores nothing.
- **`L2Tag::has_code`** (bit 25) marks lines whose `l2.instrs` slots are valid.
  An L1I fill treats a tag match as a hit **only if `has_code`** — otherwise it
  does a real instruction fill. This is what lets L2 data be bypassed while
  R4400's physically-indexed decode slots keep working.
- **A transparent write clears `has_code`** for the covering L2 line, so
  self-modifying code cannot dispatch stale decoded words. (L1I still needs the
  guest's own CACHE flush, exactly as on hardware — `has_code` only keeps L2
  honest once L1I does miss.)
- **Writeback is naturally free**: transparent lines never enter L2, and
  `writeback_l1d_line` early-returns on `!backed`.

### Where the remaining ~2% is

Not stale copying — v3 removed that. Most likely the miss-path `tc_transparent`
probe plus the extra branches through `read`/`write`. Untried ideas, cheapest
first:

1. Cache transparency per *L1D set* rather than recomputing on each miss.
2. Skip L1D tag maintenance entirely for transparent lines (changes
   guest-visible CACHE behaviour — bigger commitment, see §6).
3. Check whether R5000's 32-byte lines / no-L2 config behaves differently; only
   the R4400 path is implemented.

## 1. What and why

Today `mips_cache_v2.rs` is a *real* cache: L1I/L1D/L2 each own storage, fills
copy bytes in, dirty lines are written back on eviction. That is faithful, and
it is also a lot of memcpy for something the guest cannot observe directly.

Now that ppmem gives the CPU a direct host pointer for any mapped physical
address (plus the generation counter, plus the mapped-region bitmap), the data
copy is redundant for *cacheable RAM*: the backing store is already addressable
at `window_base + phys`.

**tcache** (`--features tcache`) keeps the whole tag/state machine — line
states, LRU, dirty bits, VCE checks, CACHE-instruction semantics — but stops
storing line data for RAM. Reads and writes go straight through to ppmem's
window. Nothing is written back on eviction, because nothing was ever
diverted.

Goal is **measurable throughput**, verified with `bench/` and `iris-bench
matrix`, not architectural elegance. If it does not measure faster it should
not ship.

### The one exception: decoded instructions

L1I does not store bytes, it stores **`DecodedInstr`** — pre-decoded slots that
the interpreter dispatches from. That is a genuine cache of *work*, not of
memory, and it cannot be reconstructed by a pointer dereference. tcache keeps
it.

Where those slots live differs by model, which the design has to respect:

| Model | Decode slots owned by | Notes |
|---|---|---|
| R4400 (`IS_R5K == false`) | **L2** (`l2.instrs`, indexed by phys) | L1I holds tags only |
| R5000 (`IS_R5K == true`) | **L1I** (`ic_instrs`, indexed by set/way) | non-inclusive L2 |

So on R4400 the L2 instruction array must stay populated even though L2 *data*
becomes transparent — the two are separate arrays and only `l2.data` is
redundant.

---

## 2. What stays and what goes

| Structure | Real cache | tcache |
|---|---|---|
| L1I tags (`ic.tags`) | maintained | **maintained** |
| L1I decode slots (`ic_instrs`, R5K) | filled | **filled** — the exception |
| L1D tags (`dc.tags`) | maintained | **maintained** |
| L1D data (`dc.data`) | filled + written back | **skipped** (see §4) |
| L2 tags (`l2.tags`) | maintained | **maintained** |
| L2 decode slots (`l2.instrs`, R4K) | filled | **filled** — the exception |
| L2 data (`l2.data`) | filled + written back | **skipped** |
| LRU / dirty / cache state | maintained | **maintained** |

Tags stay because the guest observes them: `CACHE` index-load-tag, VCE
detection, and the timing-visible hit/miss behaviour all read tag state. Only
the *payload* becomes a pointer.

---

## 3. The gate: when is an address transparent?

Not every access can bypass storage. An address qualifies only if **all** hold:

1. ppmem is present and the address is in a fully-mapped 64MB region —
   `bitmap & (1 << (phys >> 26))`, the same one-shift test the CPU uses;
2. the access is cacheable (uncached already bypasses the cache entirely);
3. the region is RAM, which (1) already guarantees, since the bitmap never
   claims a 64MB span containing MMIO.

Anything failing the gate — MMIO, the low-512KB alias, PROM, an unmapped bank —
falls back to the **existing real-cache path unchanged**. So tcache is a fast
path layered over the current implementation, not a replacement, and the
fallback keeps working for the regions ppmem's coarse bitmap cannot cover.

The cache therefore needs the bitmap pointer, alongside the window base:

```rust
tc_base:   *mut u8,        // ppmem window base   (null => tcache inactive)
tc_gen:    *mut AtomicU64, // gen window base     (jitv2 only)
tc_bitmap: *const u64,     // live mapped-region bitmap
```

All three are plain pointers set once after `Physical` is at its final address,
exactly like `Physical`'s own cached copies. `tc_bitmap` must be re-synced when
the CPU claims the sink (`resync_ppmem_bitmap`'s counterpart).

---

## 4. L1D: transparent data

`read<SIZE>` / `write<SIZE>` / `write64_masked` currently do:

```
if !tag.matches_phys(phys) { fill_l1d_line(...) }   // copies 16/32 bytes
dc.dc_read::<SIZE>(da)                              // reads the copy
```

Under tcache, for a gated address:

```
if !tag.matches_phys(phys) { tfill_l1d_line(...) }  // tags + LRU only, no copy
// read/write straight through the window at `phys`
```

`tfill_l1d_line` does everything `fill_l1d_line` does **except** the two data
copies and the writeback: victim selection, `writeback_l1d_line` (a no-op for a
transparent line — see below), invalidate, L2 tag check, VCE check, tag
install, LRU update.

### Dirty bits and writeback

A transparent line's data was never diverted, so there is nothing to write
back. But the dirty bit still has to be *maintained*, because `CACHE
Hit_Writeback_Invalidate` and friends are guest-visible and IRIX uses them.

Resolution: keep setting `dirty` on write, and make writeback of a transparent
line a **no-op that clears the bit**. The memory is already coherent by
construction, so "write back" is vacuously satisfied. This is the key
simplification tcache buys — the entire writeback memcpy disappears.

### Line-valid flag for mixed lines

The brief calls for a flag marking L1D lines that hold *real* data. It is
needed because a single L1D index can hold a transparent line now and a real
line (MMIO-adjacent, or ppmem-absent) later, and eviction must know which. Add
to `L1DTag`:

```rust
/// This line's data lives in `dc.data` and must be written back on eviction.
/// Clear for a transparent line, whose data was never diverted from RAM.
pub backed: bool,
```

`L1DTag` is currently `{ ptag: u64, cs: u8, dirty: bool }` — 16 bytes with
padding, so a second `bool` is free. `matches_phys` is untouched (it compares
`ptag` only), so the hot path does not get slower.

---

## 5. L1I / L2 instructions: still filled

The fill path keeps populating decode slots, and this is where ppmem's pointer
already helps: `fill_l2_line`/`fill_l1i_line` call `downstream.mem_ptr()`, which
now returns a direct window pointer (previously `Physical` returned `None` and
every fill went through `read_block`). So instruction fills get faster without
tcache doing anything special.

What tcache changes for them:

- **R4400**: `fill_l2_line` still writes `l2.instrs`, but skips writing
  `l2.data` for a gated line. The instruction array is the point; the byte
  array is redundant.
- **R5000**: `fill_l1i_line` still writes `ic_instrs`. L1I has no data array to
  skip.

Per the brief, an L1D line whose *bytes* were pulled in to back decoded
instructions keeps `backed = true` so it behaves like a real line.

### Self-modifying code

The decode slots are a cache of work keyed by physical address, so they must be
invalidated when the underlying memory changes. That is exactly what the
generation counters are for, and jitv2 already consumes them. With `tc_gen`
available inline, a decode slot can be validated against its page's generation
on fetch, or invalidated on write.

**Deliberately out of scope for the first cut.** The existing design already
handles SMC via the `CACHE` instruction (IRIX flushes I-cache after writing
code) and tcache does not change that contract: writes still go through the
same `write<SIZE>` entry point and still mark the same tags. Generation-based
decode invalidation is a follow-on, listed in §8.

---

## 6. Correctness argument

The claim tcache rests on: **for a gated address, the cache's data array and
the ppmem window always hold the same bytes.**

That holds because the window *is* the bank's storage — the same physical pages
`PpMemory`'s `BusDevice` reads and writes, verified by
`window_and_busdevice_see_the_same_memory` and
`window_block_access_matches_busdevice`. There is no second copy to drift.

The risks are therefore not about data, they are about *state*:

1. **A line transitions between transparent and backed.** Handled by
   `backed`, checked on eviction.
2. **The bitmap changes under a live line.** Only at MEMCFG remap, in PROM POST
   before caches matter — but a remap must still invalidate the caches, since a
   line's physical backing may have moved. `remap_banks` should flush.
3. **DMA writes RAM behind the cache.** Already true today and unchanged: a
   device writing RAM does not invalidate L1D in either scheme. Arguably tcache
   is *more* correct here, since a transparent line reads through to whatever
   DMA wrote instead of returning a stale copy.

Point 3 is worth stating plainly: tcache changes emulated-cache behaviour in
the DMA-coherency corner. Real hardware has the same incoherency, so the guest
is expected to flush — but any latent emulator bug that *depended* on the stale
copy will surface. That is a reason to keep tcache opt-in until it has boot
mileage.

---

## 7. Measuring

The whole point. Before/after on the same host:

- `make -C cpu-tests run` — correctness first, both engines.
- `iris-bench matrix` — sweeps R4400/R5000 × interpreter/jitv2, reports guest
  MIPS and an accuracy score per kernel.
- A desktop boot-to-X11 wall-clock, since the benchmark kernels are small
  enough to live in cache and will understate the win.

Expected shape: the gain scales with miss rate. Kernels that fit in L1 see
little; anything streaming memory should see the fill/writeback memcpy vanish.
`rules/testing/benchmark-suite-gotchas.md` before adding any kernel.

---

## 8. Plan

1. `tcache` feature flag; plumb `tc_base`/`tc_gen`/`tc_bitmap` into `CpuCache`,
   set from `Physical` after final placement.
2. `backed` on `L1DTag`; make `writeback_l1d_line` a bit-clear for
   `!backed`.
3. `tfill_l1d_line` (tags/LRU/VCE, no copy) + gate in `read`/`write`/
   `write64_masked`. **R4400 path first** — it is the shipping model and the
   1-way path is simpler.
4. Equivalence test: same access sequence through tcache and real cache must
   produce identical results, including at region boundaries and on eviction.
5. `cpu-tests` green, then boot.
6. R5000 2-way path.
7. Skip `l2.data` fills for gated lines (keep `l2.instrs`).
8. *(later)* generation-based decode-slot invalidation for SMC.

Steps 1–4 are testable without booting.

---

## Open questions

- **[T1]** Should `remap_banks` flush the caches? §6 point 2 says yes; needs
  confirming that nothing depends on cache surviving a MEMCFG write.
- **[T2]** Does the L2 *tag* array still earn its keep under tcache once L2
  data is gone? On R4400 it gates `l2.instrs`, so probably yes; on R5000 the L2
  is vestigial already.
- **[T3]** How much of the win is already delivered by `Physical::mem_ptr`
  alone (§5)? Measure that separately before attributing gains to tcache — it
  landed in the same change and would otherwise be double-counted.
