# `equiv_test`'s reference interpreter must sit at the JIT's ISA level

Addresses `cpucritique2.md` JIT-B.

## Symptom

```
cargo test --lib --features r5k,jitv2,j2wp,tcache jitv2::equiv_test
→ 223 passed; 7 failed
```

All seven are MIPS IV opcodes:
`fmovcf_d`, `fmovcf_s`, `movci`, `movn`, `movz`, `movz_writing_r0_is_a_noop`, `pref`.

## Cause

`equiv_test` compared the JIT against an interpreter instantiated with
`PassthroughCache = PassthroughCacheOf<false>`, where the const bool is `MIPS4`. Under
`--features r5k` the JIT compiles MOVN/MOVZ/MOVCI/PREF correctly, while the reference
interpreter — still MIPS III — rejects them.

The divergence dump makes it unambiguous. Interpreter side:

```
cp0_cause: 40, cp0_status: 4194310, pc: <exception vector>
```

`40 == 10 << 2` — **EXC_RI, Reserved Instruction**. The JIT side executed the instruction
and advanced normally. So every one of these "divergences" was the harness, not the JIT.

## Fix

```rust
#[cfg(feature = "mips4")]
use crate::mips_cache_v2::PassthroughCacheM4 as PassthroughCache;
#[cfg(not(feature = "mips4"))]
use crate::mips_cache_v2::PassthroughCache;
```

One import; every `MipsExecutor<PassthroughTlb, PassthroughCache>` in the file follows.
`PassthroughCacheM4` already existed — no new type.

**Gate on `mips4`, not `r5k`.** `r5k = ["mips4"]`, and `mips4` is the flag that actually
selects ISA level, so gating on `r5k` would miss a plain `--features mips4` build.

Result: `r5k,jitv2,j2wp,tcache` equiv_test goes 223/7 → **230/0**; full suite 873/0.

## The general rule

An equivalence harness is only as good as its reference. When *every* failure in a suite
shares one shape — here, one exception code on one side — suspect the harness before the
thing under test. Chasing seven "JIT bugs" here would have been wasted effort.

## Unrelated flake seen while verifying

`jitv2::jitv2::old_impl::tests::multi_worker_pool_compiles_and_publishes_every_request`
fails occasionally under `--features jitv2,developer` when the machine is loaded. It polls
for all N pages to publish against a hard 30-second deadline
(`assert!(Instant::now() < deadline, "only {published}/{N} entries published")`). Verified
load-sensitive and not a regression: passes 5/5 in isolation both with and without the
TLB/equiv changes, and fails only inside a full back-to-back matrix run. If it becomes
annoying, the deadline is the thing to revisit, not the pool.
