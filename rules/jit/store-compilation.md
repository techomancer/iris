# JIT Store Compilation Rules

## Full-tier blocks must be non-speculative

Set `speculative: tier != BlockTier::Full` in the compiler.

**Why:** Snapshot rollback restores CPU+TLB but NOT memory. If a store block
does read-modify-write (LW, ADDIU, SW) and then hits an exception, rollback
rewinds CPU to pre-block state but memory has the modified value. The
interpreter re-runs from block entry, reads the modified value, modifies it
again. Counters become N+2 instead of N+1. This corrupts kernel data structures.

## Full-tier blocks must terminate at the first store

In trace_block, break after pushing the first store instruction at Full tier:
```rust
if tier == BlockTier::Full && is_compilable_store(&d) {
    break;
}
```

**Why:** Long blocks with multiple load/store helper calls create complex CFG
(ok_block/exc_block diamond patterns per helper). This triggers Cranelift
regalloc2 codegen issues on x86_64 — rare but fatal corruption that manifests
after millions of block executions. Short blocks (~3-10 instructions) work
perfectly. Confirmed empirically: short blocks = stable with 5K+ Full
promotions; long blocks = crash at 780M instructions.

## Write helpers must use status != EXEC_COMPLETE

```rust
if status != EXEC_COMPLETE { ctx.exit_reason = EXIT_EXCEPTION; ... }
```

**NEVER** use `status & EXEC_IS_EXCEPTION != 0`. BUS_BUSY (0x100) does not
have the EXEC_IS_EXCEPTION bit (bit 27) set, so it would be treated as
success. But BUS_BUSY means the write was NOT performed. This silently drops
uncached writes (MMIO stores to device registers), causing slow corruption.

## Verify mode cannot validate stores

Verify mode snapshots CPU/TLB but NOT memory. After a JIT block with stores
modifies memory, the interpreter re-run reads the JIT-modified values.
Read-modify-write sequences get double-applied. Verify mode is only valid
for ALU and Load tiers.

## Delay-slot stores should be excluded from compilation

In trace_block, when checking the delay slot instruction for a branch, exclude
stores:
```rust
if is_compilable_for_tier(&delay_d, tier) && !is_compilable_store(&delay_d) {
    instrs.push((delay_raw, delay_d));
    delay_ok = true;
}
```

**Why:** If a delay-slot store faults, sync_to_executor clears in_delay_slot.
exec.step() re-executes the store as a non-delay-slot instruction.
handle_exception sets cp0_epc to the store PC (not the branch PC) and doesn't
set the BD bit. On ERET, the branch is permanently skipped, corrupting control
flow. This is defensive — the block length fix is the primary fix for stores.
