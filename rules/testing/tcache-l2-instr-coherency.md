# tcache: L2 decoded-instruction coherency, and how it was broken

## Status: bug found and fixed, 2026-08-24 — read before touching tcache's write path

Symptom: IRIX boot panicked with `EXC_RMISS` on a nonsense address (0x78) with
almost every register zero — the classic shape of executing garbage. Root cause
was stale **decoded instructions** served out of L2, not a data bug.

## The invariant tcache broke

Pre-tcache, `l2.instrs` (R4400's decoded-instruction slots, physically indexed)
stayed coherent through one mechanism: **`writeback_l1d_line` re-synced them**.
Every dirty L1D line flushed to L2 also refreshed the decoded slots covering
that region, so CPU-written code — kernel relocations, module loads, anything
going through the data path — automatically invalidated the decode cache. No
explicit flag was needed.

tcache removes that mechanism for transparent lines: they never write back to
L2 at all (`writeback_l1d_line` early-returns on `!backed`). The replacement is
`L2Tag::has_code`, set only by an instruction-origin fill, with a
`tc_invalidate_l2_code()` call clearing it whenever the line is written.

## The bug

The invalidation was placed **inside the `if transparent` branch** of
`write<SIZE>` / `write64_masked`. That leaves a hole: a *backed* L1D line in a
transparent region can still be written while L2 holds `has_code` from an
earlier instruction fill. Nothing clears the flag, so the next L1I miss probes
L2, sees `has_code`, and serves decoded slots from **before** the write.

Backed lines in a transparent region are not exotic — `L1DTag::from(u32)`
(CACHE index-store-tag) always creates one, and any line that was filled before
the window was mapped is one too.

```rust
// WRONG — invalidation only runs when the data happened to go to RAM
if transparent {
    self.tc_write::<SIZE>(phys_addr, val);
    self.tc_invalidate_l2_code(phys_addr);   // <-- inside the branch
} else {
    self.dc.dc_write::<SIZE>(virt_addr, val);
}

// RIGHT — the line's decoded slots are stale either way
if transparent {
    self.tc_write::<SIZE>(phys_addr, val);
} else {
    self.dc.dc_write::<SIZE>(virt_addr, val);
}
self.tc_invalidate_l2_code(phys_addr);
```

**Rule: `has_code` invalidation must never be gated on `transparent`.** The
question "did this write make L2's decoded slots stale?" has nothing to do with
where the bytes went.

## The second bug: the writeback early return

The first fix was not enough — IRIX still failed to boot. The deeper problem was
this, in `writeback_l1d_line`:

```rust
// WRONG — skips far more than a memcpy
if !tag.backed {
    let mut t = tag;
    t.dirty = false;
    self.dc.set_tag(l1_idx, t);
    return true;          // <-- bails out of the whole function
}
```

"Nothing to write back" is true of the *data*, but the rest of that function is
not about copying bytes. Returning early skipped:

1. **The L2 CLEAN→DIRTY transition.** L2 kept believing its line was clean, so a
   later `writeback_l2_line` **discarded** it instead of flushing to memory.
2. **The `l2.instrs` re-sync** — the pre-tcache coherency mechanism described
   above.
3. **The `cascade` branch** (`writeback_l2_line_phys`), which callers depend on
   for ordering.

Worse, a transparent L1D line could sit over a *live* L2 line holding the same
address, with the two disagreeing about where truth lived. Once anything marked
that L2 line dirty, evicting it wrote never-populated `l2.data` over correct
RAM — reproduced by `dirty_l2_over_transparent_line_corrupts_ram`.

### The fix: run the whole body, skip only the writes

Do **not** short-circuit. Let every tag transition, cascade, and `l2.instrs`
re-sync happen exactly as before, and gate only the actual data stores:

- the R5K direct-to-memory `write_block` — skipped entirely (RAM already has it);
- the L1D→L2 copy — the line's words are read **from RAM** into a small stack
  buffer, then written into `l2.data`, so L2 stays a faithful mirror.

**The `l2.instrs` re-sync must decode from that RAM-sourced buffer, not from
`l2.data` and not from `dc.data`.** `dc.data` was never populated for a
transparent line, and re-reading `l2.data` would launder whatever was already
there. The buffer is filled before L2 is touched, so it is unambiguously RAM.

With that, a co-resident L2 line is fine — L2 is a mirror, not a rival — and no
eviction hack is needed on the fill path.

## Regression tests

In `mips_cache_v2::tcache_tests`:

- `code_written_through_data_path_is_not_served_stale` — the ELF-relocation
  shape: fetch (L2 gains `has_code`), write new code through the data path,
  flush L1I only, fetch again. Must see the new word.
- `backed_write_also_invalidates_l2_code` — same, but the L1D line is forced
  *backed* first. **This caught the first bug**; the transparent-only test
  passed throughout.
- `dirty_l2_over_transparent_line_corrupts_ram` — forces L2 dirty over a
  transparent line and evicts. **This caught the second bug**, and is the
  direct analogue of the IRIX boot failure.
- `coexisting_l2_line_stays_consistent_with_ram` — a transparent line over a
  live L2 line, written, flushed and evicted; RAM must still hold the written
  value.

## What is still the guest's job

L1I is not invalidated by a write, under tcache or without it. Self-modifying
code must still issue the `CACHE` flush, exactly as on real hardware —
`has_code` only stops L2 from serving stale decoded words once L1I does miss.
`transparent_write_clears_l2_code_flag` pins this down.

## R5000

tcache is **not implemented** on the 2-way paths — they always take the
real-cache route, so no transparent lines exist and the feature is a silent
no-op there. R5000 also has no `l2.instrs` (its decode slots live in
`ic_instrs`). If tcache is ever extended to R5000, the `has_code` invalidation
has to be added to those write paths too.

## Related

- `docs/tcache-design.md` — the feature, and its measurements
- [[project_ppmem]] — the window tcache reads through
