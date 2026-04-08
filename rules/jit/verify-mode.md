# JIT Verify Mode Rules

## Timing false positive detection

Verify mode re-runs each block through the interpreter at a different wall-clock
time. The interpreter may see different external interrupt state via the atomic
and take an exception the JIT didn't see (or vice versa).

Detection: if the interpreter PC is in exception vectors (0x80000000-0x80000400
or 0x80000180) but the JIT PC is not, it's a timing false positive. Keep the
block, don't invalidate. Use the interpreter's result as authoritative.

## Verify mode is invalid for store blocks

See store-compilation.md. Memory is not part of the snapshot, so verify mode
double-applies read-modify-write sequences for blocks containing stores.
Only use verify mode for ALU and Loads tiers.
