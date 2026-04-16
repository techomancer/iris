# JIT Store Compilation Rules

## Full-tier blocks must terminate BEFORE the first store

In trace_block, break BEFORE pushing a store instruction at Full tier:
```rust
if tier == BlockTier::Full && is_compilable_store(&d) {
    record_termination(&d, tier);
    break;
}
```

**Why:** This ensures all Full-tier blocks are load-only → always speculative →
self-healing via rollback+demotion. Cranelift regalloc2 produces occasional
codegen errors in blocks with multiple ok_block/exc_block helper-call diamonds.
At Loads tier, speculative rollback catches these errors and demotes bad blocks.
Non-speculative blocks (which store-containing blocks must be) have no safety
net — codegen errors persist permanently, corrupting state silently.

By terminating before stores, all compiled blocks stay load-only, all get the
speculative safety net, and stores execute via the interpreter where they're
always correct.

Confirmed via systematic isolation matrix (2026-04-16):
- Loads tier (speculative): 3/3 clean
- Full tier with stores compiled (non-speculative): 0/3 clean (hang, broken, broken)
- Full tier store-free but non-speculative: 3/3 broken
- Full tier store-free and speculative: 3/3 clean

## Speculative flag must be based on store presence, not tier

```rust
speculative: !block_has_stores(instrs)
```

**NOT** `speculative: tier != BlockTier::Full`. The old rule was overly broad —
it made ALL Full-tier blocks non-speculative, including load-only blocks that
are safe to roll back. The correct rule: only blocks containing stores
(SB/SH/SW/SD) need to be non-speculative, because rollback can't undo memory
writes.

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
for ALU and Load tiers. Running VERIFY with Full-tier store blocks causes
kernel panics (confirmed 3/3 kernel panic in isolation matrix).

## Delay-slot stores should be excluded from compilation

In trace_block, when checking the delay slot instruction for a branch, exclude
stores (and loads — any faulting instruction):
```rust
let delay_can_fault = is_compilable_load(&delay_d) || is_compilable_store(&delay_d);
if is_compilable_for_tier(&delay_d, tier) && !delay_can_fault {
    instrs.push((delay_raw, delay_d));
    delay_ok = true;
}
```

**Why:** If a delay-slot instruction faults, sync_to_executor clears
in_delay_slot. exec.step() re-executes as a non-delay-slot instruction.
handle_exception sets cp0_epc to the instruction PC (not the branch PC) and
doesn't set the BD bit. On ERET, the branch is permanently skipped.
