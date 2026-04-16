# JIT Speculative Execution Is a Safety Net, Not Just an Optimization

## Rule

All compiled blocks that do NOT contain store instructions MUST be speculative.
The speculative flag enables snapshot/rollback/demotion — without it, Cranelift
codegen errors persist permanently and silently corrupt emulator state.

## Why

Cranelift regalloc2 produces occasional miscompilations in blocks with multiple
helper-call diamonds (ok_block/exc_block CFG patterns from load helpers). These
are rare (perhaps 1 in 10,000 blocks) but fatal over billions of block
executions.

The speculative execution path provides three-layer defense:
1. **Snapshot** before block entry captures correct pre-block state
2. **Rollback** on exception restores correct state, preventing propagation
3. **Demotion** after 3 exceptions replaces the bad block with a lower-tier
   version that doesn't compile the problematic instruction

Non-speculative blocks have NONE of these defenses. A codegen error that
produces a wrong GPR value persists in the executor state permanently, with
no demotion trigger and no escape.

## Evidence (isolation matrix, 2026-04-16)

| Configuration | Speculative? | Result |
|---|---|---|
| Loads tier (load-only blocks) | Yes | 3/3 clean |
| Full tier (load+store blocks) | No | 0/3 clean |
| Full tier (load-only, forced non-speculative) | No | 3/3 broken |
| Full tier (load-only, speculative) | Yes | 3/3 clean |

The ONLY variable that correlated with success was the speculative flag.
Instruction mix, tier label, and block length were all irrelevant.

## How to apply

```rust
// In compile_block:
speculative: !block_has_stores(instrs),

// In trace_block: terminate Full-tier blocks before stores
if tier == BlockTier::Full && is_compilable_store(&d) {
    break;
}
```

This ensures ALL compiled blocks are load-only → always speculative →
always self-healing. Stores execute via interpreter.
