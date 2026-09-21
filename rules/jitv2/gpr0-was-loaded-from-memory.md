# `$zero` was being loaded from memory, 79 times in one region

Fixed 2026-09-20 (`emit_read_gpr`, src/jitv2/codegen.rs).

`emit_write_gpr` has always had a `reg == 0` early return — gpr[0] reads as 0
architecturally and is never written, so the store is skipped. **`emit_read_gpr`
had no such case.** Every `addu rd, rs, $zero` (the standard MIPS register
move), every `beq rs, $zero`, every `sll rd, rt, 0` emitted a real
`load.i64 v0+104` of a location hardwired to zero.

Cranelift cannot fix this itself: the load uses `MemFlagsData::trusted()`
(`notrap + aligned`, **no alias region**), so it cannot prove the location is
invariant across a callout and must reload it every time.

Measured on one real corpus region (pfn 0x8004, entry 0x258): `core.gpr[]+0x68`
was **the single hottest address in the whole region — 79 loads, 0 stores**,
33% of all GPR-range loads. Corpus-wide the fix is **-1.4% emitted bytes** at
`intrun=1`, `opt_level=speed`, with no interrupt-latency cost and no knob.

The fix is one line mirroring the write side:

```rust
if reg == 0 { return ctx.builder.ins().iconst(ir::types::I64, 0); }
```

## How it was found, and the lesson

Not by reading the code. It surfaced while classifying *why* GPR loads survive
optimization ([[what-actually-blocks-gpr-forwarding]]): 42% of remaining loads
had **no prior store anywhere in the region**, and the hottest offset had a
load:store ratio of 79:0. A ratio like that is not "a value from an earlier
region" — it is a constant being fetched.

**Per-address load/store ratios are a cheap bug detector.** Any address with
many loads and zero stores in a region is either a genuine read-only input or
something that should not be in memory at all.

## Related: writes to `$zero` with side effects

`lw $zero, x` must still perform the access (it can fault, and is used as a
prefetch/probe idiom) while discarding the result. Two paths, both already
correct before this fix:

- **Inline path**: `emit_write_gpr(ctx, 0, result)` returns early and `result`
  goes unused. Cranelift's DCE keeps the side-effecting load and drops only the
  dead extend/narrow chain. **No dummy store is needed** — an unused `Value`
  costs nothing.
- **Mem-helper path**: the helper is a shared out-of-line function taking the
  destination as a runtime *offset*, so it cannot "not write". It writes to
  `MipsCore::gpr_scratch` instead (`emit_load`, codegen.rs). That dummy store
  is inherent to the shared-helper design, not a Cranelift limitation, and is
  the right trade against a second helper variant or a per-load branch.
