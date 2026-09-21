# Flush-preserved pages inherited the previous program's entry points

Changed 2026-09-20. `j2 flushkeep <n>`, default **0**.

## The mechanism

`PhysicalCodePage::reset_for_flush_survivor` folds `compiled` back into
`requested`:

```rust
for (req, comp) in self.requested.iter().zip(self.compiled.iter()) {
    let bits = comp.load(Relaxed);
    if bits != 0 { req.fetch_or(bits, Relaxed); }   // entry points survive
}
```

That is deliberate churn reduction — a survivor should not relearn its own
entry points. But `JITV2_FLUSH_PRESERVED` was **1024, the whole pool**, so
*every* page survived every flush and kept its accumulated entry set forever.
A physical page later reused by a **different program** then inherits the
previous occupant's entry offsets and compiles a function with entries for code
that is no longer there.

## Measured

Two real 1400-1900 page corpora, captured before and after:

| | keep=1024 | keep=0 |
|---|---|---|
| entry points (total) | 77,690 | **56,287** (-27.6%) |
| mean entries/page | 54.4 | **29.0** (-47%) |
| median entries/page | 43 | **18** (-58%) |
| max entries/page | 222 | 164 |
| **walked words per entry** | 7.63 | **9.16** (+20%) |

The worst page observed under the old policy had **98 entry points for 329
walked instructions** — one entry per 3.4 instructions, which is not plausible
control flow.

Two costs per surplus entry point: a larger compiled function, and a deeper
entry-dispatch binary search (`(pc & 0xfff) >> 2` down a ladder, ~3 x86
instructions per level) paid on **every external entry into the page**. Median
43 -> 18 entries takes that ladder from ~6 levels to ~5.

## Caveats on those numbers

- The two corpora are **different workloads**, not a controlled A/B (1414 vs
  1902 pages, 13% fewer walked words). Per-page and per-word *ratios* are the
  meaningful comparison; absolute totals are not.
- Normalized emitted size went *up* slightly (683 -> 719 bytes per walked word,
  +5.2%) — almost certainly workload mix, not a regression, but it is not
  evidence of a size win either. A real size comparison needs the same
  workload captured both ways.

## The trade

`flushkeep 0` frees every page, so survivors relearn entry points and take real
recompiles. Watch `j2 stats`' churn line — under the old policy it read
*"30137 compiles skipped as redundant, 1007 had to recompile (96.8% avoided)"*,
and part of that avoidance comes from preserved bitmaps. Pushing entry-point
counts down may push recompile counts up.

`j2 flushkeep 1024` restores the old behaviour for A/B testing without a
rebuild. Four unit tests that specifically exercise preservation now pin the
value explicitly (serialized by `flush_preserved_test_lock`, since it is a
process-wide global) rather than relying on the default.
