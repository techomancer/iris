# tlbcheck: vmap fast-path can strand a still-valid TLB entry as a phantom Miss

## Status: real bug found via the new `tlbcheck` feature, not yet fixed

`MipsTlb`'s O(1) `vmap[VA[31:13]]` fast path (`mips_tlb.rs`) is last-writer-wins,
keyed only by TLB entry index. `write(index, entry)` always does
`vmap_erase(index)` (using the *old* entry at that index) then
`vmap_fill(index)` (using the *new* one) — unconditionally, independent of the
entry's V bit.

If two TLB entries ever transiently cover the same VPN2 (ordinary TLBWR
replacement can produce this — the victim index is effectively random, nothing
prevents it momentarily colliding with another live entry's VPN2 before that
entry is itself evicted), `vmap_fill` for the later write clobbers the vmap
slot to point at the later index. If that later entry is then rewritten again
— to a new VPN2, **or** invalidated (V=0) — `vmap_erase` sees the slot still
tagged with its own index and wipes it to `VMAP_MISS`. The *earlier* entry,
which was never touched and is still fully valid, never gets its claim
restored: nothing re-derives "does some other entry still cover this VPN2"
before erasing.

`translate()`'s vmap fast path then treats `VMAP_MISS` as a **definite** miss:

```rust
// mips_tlb.rs, translate()
if entry_idx != VMAP_MISS {
    ...
} else {
    return TlbResult::Miss { vpn2: virt_addr >> 13 };   // <-- no fallback
}
// MRU linear scan lives below this — but VMAP_MISS never reaches it
```

Unlike the ASID-mismatch case just above it (which deliberately falls through
to the linear scan because "a different entry ... may exist"), a vmap miss
short-circuits straight to `TlbResult::Miss` and never gives the linear MRU
scan — which *would* find the orphaned entry — a chance to run. The entry is
silently unreachable via `translate()` until something rewrites that specific
index again.

This is a strong candidate for (at least a contributor to) the original
"duplicate TLB entry" IRIX panics that motivated building `tlbcheck`
([[project_tlbcheck]] if that memory exists, else see the tlbcheck feature
itself): a phantom miss on an address IRIX believes is mapped can plausibly
cascade into the kernel's own TLB-refill handler inserting a fresh entry that
collides with the orphaned one.

## Repro (see `src/mips_exec_test.rs`)

`test_tlbcheck_detects_vmap_left_dangling_by_overlap_repair`:
1. TLBWI index 5: VPN2=0x100, ASID=10, valid.
2. TLBWI index 6: same VPN2/ASID (a transient overlap — `tlbcheck` correctly
   flags this immediately as a duplicate).
3. TLBWI index 6 again, now VPN2=0x300 (simulating the replacement algorithm
   picking a fresh victim next cycle) — **or** just invalidating index 6
   (V=0) produces the identical fallout.

After step 3, `tlbcheck` reports:
```
vmap[00100] = 255 but entry 5 covers this VPN2
```
Entry 5 was never touched after step 1, is still fully valid, but is now
unreachable through the vmap fast path.

## Not fixed here

Fixing needs one of:
- In `translate()`, fall back to the linear MRU scan on `VMAP_MISS` too
  (same as the existing ASID-mismatch fallthrough) instead of returning a
  hard `Miss` — safe but gives up the vmap's O(1) guarantee on this specific
  edge case.
- In `write()`, before `vmap_erase(index)` unconditionally blanks a slot,
  re-scan for any *other* entry whose range still covers that slot and
  restore its claim (more invasive, keeps strict O(1) hot path).

Left as a `tlbcheck`-covered regression (see [[project_tlbcheck]] /
`src/mips_exec_test.rs`) rather than fixed in the same pass that added the
checker, since it's a distinct, pre-existing correctness bug the checker
happened to surface.
