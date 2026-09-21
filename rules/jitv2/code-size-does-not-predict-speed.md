# Emitted-code size did not predict speed — four changes, one marginal win

Session of 2026-09-19/20. Four codegen changes were measured by emitted bytes
and then benchmarked on **pure-CPU guest workloads** (whetstone, dhrystone,
`iris-bench` ssl). Recording the outcome because the pattern was consistent and
cost a lot of measurement to establish.

| change | emitted bytes | benchmark |
|---|---|---|
| `j2 intrun 8` (coalesce interrupt checks) | -6% to -10% | marginal uplift on dhrystone + ssl |
| `gpr[0]` read fix ([[gpr0-was-loaded-from-memory]]) | -1.4% | no measurable change |
| shared absolute-PC exit block | -2.5% | no measurable change |
| `j2 flushkeep 0` (drop inherited entry points) | entry points -27.6% | (see that note) |

## Why removing instructions bought nothing

These benchmarks are CPU-bound with little I/O, so JIT-emitted code *is* the
bottleneck — "the time must be going somewhere else" is **not** the
explanation. The real one is that all three flat changes removed instructions
that were never costing cycles on an out-of-order host:

- **the interrupt check** is a seqcst *load*, which on x86-64 is a plain `mov`
  (no fence, no `lock`). It constrains the compiler, not the CPU. Its `brif` is
  perfectly predicted.
- **the eliminated GPR loads** hit L1 and sat off the dependency critical path.
- **the shared exit block** removed epilogue bytes from paths taken once per
  *region exit* — a region averages hundreds of instructions between exits.

All three removed instruction *count*. None removed a stall: no cache miss, no
mispredict, no dependency chain.

## The rule

**Do not use emitted-code size as a proxy for speed in this JIT.** Quote it as
what it is — arena footprint, which has its own value (less code resident,
fewer exhaustion-driven `mega_flush`es on long runs). If a change is claimed to
be faster, benchmark it.

Corollary for reading `rules/jitv2/`: every byte-count claim in these notes
predates this finding and should be read as a size claim only, unless it was
paired with a wall-clock measurement.

## Two measurement traps hit along the way

1. **Never measure emitted code under `developer`.** It forces
   `opt_level=none` *and* emits `emit_dev_trace_bp`, a 4-argument
   `call_indirect` per instruction which is also an opaque clobber. Measured
   live: ~380 code bytes/guest instruction under `developer` where the same
   pages compile to ~92 without it — a **4x** error. This is the second time
   that trap has produced a wrong conclusion (see
   [[block-fragmentation-blocks-cse]]). The size counters are no longer
   `developer`-gated, precisely so the metric can be read from a build that
   ships.

2. **Scale a percentage against the region before believing it.** "25% of GPR
   loads eliminated" was 22 instructions in a region of ~8,100 — 0.27% of the
   stream. A true percentage can be irrelevant in absolute terms, and was,
   twice.

## What was NOT tried

A **per-region GPR cache** — keep a hot register in a host register across the
whole region, spill at exit. Distinct from store-to-load forwarding: it attacks
the ~42% of loads that have no prior store in the region at all, which
forwarding structurally cannot touch. Cranelift cannot do it (it cannot prove a
callout won't write `gpr[n]`); the JIT could, since it decodes every
instruction and knows which ones call out. Ceiling unmeasured.

Given the table above, measure the ceiling before building it.
