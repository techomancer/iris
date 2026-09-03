# Block fragmentation, not callouts, is what starves `opt_level=speed`

Measured 2026-09-01 on 300 real IRIX corpus pages (`jitv2_corpus/`) via
`zz_corpus_sizes` with `IRIS_JIT_DISASM=1` and `IRIS_OPT_SPEED=1`.

## First: measure without `developer`

`CODEGEN_OPT_LEVEL_SPEED` (codegen.rs) defaults to
`!cfg!(feature = "developer")` — production and `lightning` get
`opt_level=speed`; `developer` gets `none`. Two consequences for anyone
measuring emitted code:

1. A `developer` build measures **unoptimized** codegen unless you set
   `IRIS_OPT_SPEED=1`.
2. Even with that set, `emit_dev_trace_bp` adds a `call_indirect` **per
   instruction**. In the first run of this investigation, **11,214 of 15,723
   callouts (71%) were the dev-trace hook** — and each one is an opaque
   clobber of the whole core struct plus a `brif`, so it wrecks both the
   callout statistics and the block-size distribution.

`zz_corpus_sizes` used to require `developer` (it calls `last_code_size()`,
which is developer-gated) and so silently measured the wrong thing. It now
builds without it, reporting size 0 in that case.

**Rule: any claim about emitted-code shape must come from a non-`developer`
build.** The first pass of this investigation produced a completely wrong
field-traffic ranking (it made `pc`/`in_delay_slot` look like a 5:1 majority
of stores; the real figure is 1.59:1) purely from this contamination.

## The actual numbers (clean build, `speed`, 300 pages)

```
regions=300  total_asm_instrs=495353  bytes=2146233
  mean region 7154 B, median 3200 B
  loads 65908 (13%)  stores 22639 (4%)  calls 4509 (0.9%)
  machine blocks=89906  mean 5.5 instrs  median 3.0
```

Field-level traffic (offsets: `hot.interrupts`=0x0, `hot.cycles`=0x8,
`pc`=0x50, `in_delay_slot`=0x58, `gpr[]`=0x68..0x164, `fpr[]`=0x178):

| field | loads | stores |
|---|---|---|
| `gpr[*]` | 12,027 | 6,790 |
| `hot.interrupts` | 8,720 | — |
| `pc` | 3,959 | 5,884 |
| `in_delay_slot` | — | 4,900 |
| `hot.cycles` | — | 2,509 |

## Finding 1: Cranelift's CSE works *within* a block, and only there

Zero redundant same-address loads inside any machine block. That zero is real,
not broken instrumentation — ignoring block boundaries finds **27,668**
duplicate loads (11,522 in the clean build). Of the clean build's duplicates:

- **58% separated by a block boundary only** — no call in between. Pure
  structural loss: Cranelift would have eliminated these had the instructions
  shared a block.
- 24% separated by call + block.
- 17% within the scan window with nothing between (mostly cross-region
  artifacts of a flat-file scan).

## Finding 2: callouts are *not* the main barrier

Only 4,509 calls across 495,353 instructions (0.9%). The inline L1-D fast path
(`emit_inline_mem_guard`, gated on `dc_geometry.supported`) is doing its job —
most loads/stores never reach `emit_mem_read_callout`. Callout clobbering of
the core struct is real but rare enough not to dominate.

**Corollary for benchmarking**: `Codegen::dc_geometry` defaults to
`unsupported()`, which skips the inline path entirely and makes every access
call out. Any harness measuring emitted code must stamp real geometry (as
`zz_corpus_sizes` does) or it measures callout-only code and is blind to the
whole inline path.

## Finding 3: the interrupt preamble is smaller than it looks, but fragments everything

`emit_pending_interrupt_preamble` emits, per head instruction, an
`atomic_load` of `core.hot.interrupts` + test + `brif` to a cold bail block.
Exactly 8,720 sequences, matching the 8,720 `hot.interrupts` loads (the two
counts cross-validate the detector). That is 26,160 instructions = **5.3%** of
emitted code — not the main cost by volume.

Its real cost is structural: it is a **seqcst** load (deliberately — see the
comment at its definition, `speed` mode would otherwise be free to hoist it),
which is a full barrier for alias analysis, and its `brif` splits the block at
**every instruction boundary**. That is what holds machine blocks at a median
of 3 instructions.

## Ranking (by evidence, not intuition)

1. **Block fragmentation** — 6,745 provably-recoverable redundant loads,
   median block of 3 instructions. Merging straight-line runs into single
   blocks is the real lever.
2. **Hoisting the interrupt check to block granularity** — small by volume,
   but it is the *enabler* for (1): merging pass-1 blocks without hoisting the
   preamble buys little, since the preamble re-splits every instruction.
   Must stay per-instruction under `jitv2_lockstep`.
3. **`pc`/`in_delay_slot` traffic** — partly addressed, see
   [[inlined-slot-pc-bd-bracket-is-dead]].
4. **GPR load/store traffic** — smaller than expected, and partly fixed for
   free by (1), since Cranelift already CSEs these within a block.

## What was *wrong* about the initial hypothesis

The intuition going in was "callouts clobber the core struct, so nothing can
stay in host registers." That is true in principle and near-irrelevant in
practice at 0.9% call density. The measurement inverted the ranking: the
barrier is the block structure the JIT itself emits, not the callouts.

Note also that `emit_read_gpr`/`emit_write_gpr` are plain `load`/`store`
against `core_ptr` with `MemFlagsData::trusted()` (= `notrap + aligned`, **no
alias region**). There is no register cache; promoting GPRs to host registers
is entirely Cranelift's redundant-load-elimination, which is why block scope
determines how much of it happens.
