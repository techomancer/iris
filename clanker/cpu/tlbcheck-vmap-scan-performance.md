# tlbcheck: naive vmap re-derivation cost ~800us/call, dominated boot time

## Symptom

With `--features tlbcheck,tlbvmap`, IRIX boot crawled at roughly 0.1MHz
effective speed — orders of magnitude slower than normal interpreted/JIT
speed, even accounting for `tlbcheck` being a diagnostic-only feature never
meant for production use.

## Cause

`find_consistency_violations`'s vmap-vs-scan check (case 4) naively verified
every one of the 524288 `vmap` slots (`VMAP_SIZE = 4GB / 8KB`) against a full
scan of all 48 TLB entries on **every single TLBWI/TLBWR**:

```rust
for vpn2 in 0..VMAP_SIZE {              // 524288
    for i in 0..TLB_NUM_ENTRIES {       // 48
        ...
    }
}
```

~25M iterations per call. IRIX boot issues many thousands of TLBWI/TLBWR
(PROM's TLB-clear loop, every TLB refill on a cold process, every ASID/wired
churn on context switch), so this overhead compounds directly into wall-clock
boot time — easily tens of seconds of pure `tlbcheck` bookkeeping.

## Fix (two-stage)

**Stage 1** — invert the loop: derive an `owners: HashMap<vpn2, (entry_idx,
multiple)>` by iterating the 48 *entries* once (mirroring what
`vmap_fill`/`vmap_erase` themselves do — each entry only covers `count =
page_size/8KB` VPN2 slots, 1 for a 4KB page, at most ~2048 for the largest
16MB page) instead of iterating all 524288 vmap slots. This alone doesn't fix
it: measured ~800us/call, barely different from the naive version, because
verifying "every *other* slot is VMAP_MISS" still required touching all
524288 elements of the `vmap` array at least once (a HashMap `.contains_key`
lookup on every element, or even just a plain array read on every element,
still costs ~500-800us at this scale).

**Stage 2** — added `MipsTlb::vmap_touched: HashSet<usize>` (feature-gated
under `tlbcheck`, zero cost when off), maintained by `vmap_fill`/`vmap_erase`
recording the `[vpn2, vpn2+count)` range they touch. Since `vmap` starts as
`[VMAP_MISS; VMAP_SIZE]` and nothing else ever mutates it, any slot **not**
in `vmap_touched` is guaranteed still `VMAP_MISS` without needing to read it
at all. Check 4 now only walks `vmap_touched` (bounded by total pages ever
mapped across the session — realistically dozens to low thousands, not
524288) instead of the full array.

Result: **~9us/call**, down from ~800us — roughly 87x. Combined with stage
1's elimination of the O(VMAP_SIZE × 48) double loop, this is a ~90,000x
reduction from the original naive implementation.

## Lesson

When writing a consistency checker against a sparse fast-path table (vmap
here; anything keyed by a huge index space but populated by a handful of
writes), re-deriving "which slots don't have a claim" by touching the full
table is deceptively expensive even when the per-element work is trivial —
524288 iterations of *anything* non-trivial is not free at the frequency a
per-instruction hook runs. Track the touched-set incrementally alongside the
real writes instead of recomputing it from the sparse population every check.
