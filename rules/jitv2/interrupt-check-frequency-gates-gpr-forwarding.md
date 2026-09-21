# The pending-interrupt check, not block structure, is what blocks GPR forwarding

Measured 2026-09-19 with `zz_forwarding::zz_cl_forwarding` (a six-shape
Cranelift probe, `src/jitv2/mod.rs`) plus real emitted code from the corpus.

> **Bottom line first — and the title oversells it.** The mechanism is real in
> isolation, and `j2 intrun` does shrink emitted code 6-10%. But at corpus
> scale that shrink is **~95% deleted preamble, not unlocked forwarding**
> (~32 B per preamble, see below), GPR loads fall only 8.4%, and a live
> `lightning` boot showed **no benchmark difference**. The knob defaults to 1.
>
> The real cap on forwarding turned out to be **callouts, not the interrupt
> check** — see "Why removing the barrier barely moved forwarding". That
> reverses the framing this note was written with, and also reverses
> [[block-fragmentation-blocks-cse]]'s "callouts are not the main barrier"
> (true by corpus-wide call *density*, false in the call-dense regions that
> matter).

## The finding

`emit_read_gpr`/`emit_write_gpr` are plain `load`/`store` against `core_ptr`.
There is no register cache — promoting a GPR to a host register is entirely
Cranelift's store-to-load forwarding. Two adjacent instructions where one
writes `gpr[n]` and the next reads it should need no reload.

At `opt_level=speed`, with nothing between them, that is exactly what happens:

```asm
movq 0x90(%rdi), %rsi
leaq 7(%rsi), %r8
movq %r8, 0x88(%rdi)      ; store gpr[a]
leaq 8(%rsi), %rsi        ; forwarded — gpr[a] never reloaded
movq %rsi, 0x90(%rdi)
```

Put `emit_pending_interrupt_preamble` between them and the reload comes back:

```asm
movq %rsi, 0x88(%rdi)     ; store gpr[a]
movq (%rdi), %r8          ; the seqcst atomic_load
addq 0x88(%rdi), %rsi     ; RELOADED
```

The preamble's `atomic_load` is **seqcst** deliberately (so `speed` can't hoist
a stale snapshot), and seqcst is a full barrier for alias analysis. Emitting
one per instruction therefore suppresses forwarding at *every* instruction
boundary.

## Block merging is NOT the lever — this corrects the earlier ranking

[[block-fragmentation-blocks-cse]] ranked "merge straight-line runs into single
Cranelift blocks" first, inferring fragmentation from the fact that 58% of
duplicate loads were separated by a block boundary and nothing else. That
inference was wrong, and the probe's `split_plain` shape is the disproof:

```
plain        (one block)                      -> forwards
split_plain  (two blocks, unconditional jump) -> forwards IDENTICALLY
```

Cranelift merges a block with a single predecessor reached by an unconditional
jump, so the per-word block structure costs nothing. The duplicate loads that
note measured were separated by a block boundary *and* a preamble; only the
preamble mattered. **No blocks were merged to get the win below.**

`barrier` and `sidexit` also emit identical traffic, so the preamble's `brif`
and cold bail block are not the problem either — only the seqcst ordering is.

## Callouts still invalidate, and must

Memory reads write the destination GPR *directly from Rust* through a `dst`
pointer (`emit_mem_read_callout`, commit de24493 — the MS x64 ABI can't
register-return a 16-byte struct). If forwarding survived such a call, the JIT
would read a stale register the callee had just overwritten.

It doesn't. Probe shape `callout` (store `gpr[a]`, `call_indirect` handed
`&gpr[a]`, load `gpr[a]`):

```asm
movq %r8, 0x88(%rdi)      ; store gpr[a]
leaq 0x88(%rdi), %rdx     ; &gpr[a] as dst
call    *%r10
addq 0x88(%rbx), %r9      ; RELOADED — correct
```

Cranelift treats `call_indirect` as an opaque memory clobber and invalidates
everything. It does not know the callee writes `gpr[a]` specifically; it
assumes any write, which is conservative in the safe direction. So the
direct-to-GPR read scheme stays sound under any `intrun`, and callouts remain
natural run boundaries for forwarding regardless of the knob.

## The knob

`CODEGEN_INTERRUPT_RUN` / `j2 intrun <n>`: how many consecutive head
instructions share one check. Default 1 (historical per-instruction
behaviour). Forced to 1 under `jitv2_lockstep` **only** — that build exists to
verify the unmodified per-instruction emission, so it keeps emitting every
preamble as before.

`developer` is NOT pinned (an earlier version pinned it too, wrongly:
`developer` is the build you use to debug and test things, so compiling the
feature out there meant `j2 intrun 2` reported `1` and exercised nothing).
Nothing in that build needs a pin — `emit_dev_trace_bp` is emitted outside the
preamble skip and still runs per instruction at any `intrun`.

**But `developer` defaults to `opt_level=none`** (`CODEGEN_OPT_LEVEL_SPEED`),
and at `none` Cranelift runs no store-to-load forwarding for this knob to
unblock — so raising `intrun` there changes almost nothing in emitted code and
looks like a broken knob. It is not: run `IRIS_OPT_SPEED=1` at launch (or
`j2 opt speed` + stop + `j2 flush`), which works in `developer` like any other
build. `j2 intrun` now prints the live `opt_level` next to the value, and warns
when it is `none`, so this is visible rather than something you have to know.

Coverage uses the same predicate as `try_emit_fused_lui` — a word may skip its
own check only if nothing else can reach it (`is_branch_target`,
`is_entry_point`, `is_branch_fallback_successor`, `is_fallback` all force a new
run) and the predecessor's edges stay in-region.

The cost is interrupt-sampling **latency**, bounded by `n` guest instructions —
a timing property, not a semantic one, of the same kind `skip_entry_preamble`
already accepts. Not "once per region": a region can be hundreds of
instructions and latency should not scale with region length.

## Measured effect

60-page corpus subset, `opt_level=speed`, emitted bytes:

| intrun | bytes | vs baseline |
|---|---|---|
| 1  | 22,808,725 | — |
| 2  | 21,492,306 | -5.8% |
| 4  | 20,814,590 | -8.7% |
| 8  | 20,511,490 | -10.1% |
| 16 | 20,428,844 | -10.4% |
| 32 | 20,401,812 | -10.6% |

Region count compiled (`ok=4384`) is identical at every setting — nothing is
declined by raising it.

One real region (pfn 0x8004 entry 0x258), intrun 1 -> 8:

- interrupt `atomic_load`s: 204 -> 54
- core-struct loads: 1023 -> 844 (**-17.5%**)
- core-struct stores: 658 -> 656 (flat, as expected — architectural writes
  must still happen)
- emitted asm lines: 9722 -> 8427 (-13.3%)

### Not all of that is just the deleted preamble

Worth checking, because the obvious assumption is that removing N preambles
removes exactly N preambles' worth of code and changes nothing else. It does
not. The preamble is 3 x86 instructions (`movq (%rbx),%rax` / `testq` /
`jnz`) plus its cold-block `jmp`. Removing 150 of them should cost ~600
instructions; the measured delta is **981**, i.e. 6.54 per removed preamble.

By mnemonic, intrun 1 -> 8 on that region:

| mnemonic | 1 | 8 | delta |
|---|---|---|---|
| `movl` | 678 | 364 | **-314** |
| `movq` | 2491 | 2301 | -190 |
| `testq` | 271 | 114 | -157 |
| `jnz` | 492 | 335 | -157 |
| `jmp` | 810 | 653 | -157 |

The whole delta is the preamble, once you count what the preamble actually
costs. **Its cold bail block is not shared** — each one bakes in its own word
offset:

```asm
block134:
  movl $0x9b, %ecx      ; 5 B — this word's offset
  movl $0x200, %eax     ; 5 B — EXEC_FALLBACK
  jmp  label1684        ; 5 B — to the shared exit
```

492 distinct cold targets for 204 preambles in that region. So one preamble is:

| part | bytes |
|---|---|
| `movq (%rbx), %rax` | 3 |
| `testq %rax, %rax` | 3 |
| `jnz` rel32 | 6 |
| `jmp` to the next block | 5 |
| cold block (2 x `movl` + `jmp`) | 15 |
| **total** | **~32** |

At ~32 B each, the live boot's ~20 B/instruction saving is **~0.63 preambles
removed per instruction**, which is what mean run length 4.33 predicts. (At the
3-instruction figure one might assume for the preamble it would come out above
1.0 per instruction, which is impossible — that mismatch is what exposed the
unshared cold block.)

This also kills a tempting misreading of the `movl` row: those 314 removed
`movl $imm` are **mostly the cold blocks' own** `movl $word` / `movl
$EXEC_FALLBACK` pairs — 150 removed preambles x 2 = 300 of the 314 — not
Cranelift rematerialization. Only ~14 `movl` and ~40 `movq` are attributable to
better codegen (forwarded loads, folded constants, as the probe showed in
miniature with `plain`'s `leaq 8` vs `barrier`'s split `+7`/`+1`).

So the shrink is ~95% deleted preamble and ~5% improved codegen. The
forwarding this note is named for is real and visible in the probe, but at
corpus scale it is a rounding error next to simply emitting 32 fewer bytes of
check.

Most of the shrink is at n=2..8; past 8 it is noise. Runs of 32+ are 0.2% of
the corpus and the mean run is 4.33 heads ([[corpus-capture-from-the-pcp-cache]]
for how to reproduce the corpus).

### Live boot (`lightning`, `j2 status` after a flush)

Better evidence than the offline corpus — real workload, real compile mix.
`j2 status`' histogram is a scan of *currently published* entries, not a
running total, so it resets on `mega_flush` and refills; the bytes/instruction
ratio is the robust figure (the absolute totals also reflect how far refill had
progressed when sampled).

| intrun | code bytes/instruction | total bytes | instructions |
|---|---|---|---|
| 1  | 372.6 | 180,796,841 | 485,190 |
| 8  | 353.2 | 165,050,774 | 467,320 |
| 16 | 350.8 | 151,909,753 | 433,015 |

-5.2% at 8, -5.8% at 16. The curve flattens exactly where the run-length
histogram says it should (mean run 4.33 heads; runs of >=16 are a thin tail),
so there is nothing to gain above ~8. Three independent measurements — the
Cranelift probe, the offline corpus and this live boot — now agree on the
mechanism and on where it saturates.

## Why removing the barrier barely moved forwarding

Measured on the same region (pfn 0x8004 entry 0x258), intrun 1 -> 8, counting
only `core.gpr[]`-range (0x68..0x164) traffic:

```
GPR loads : 262 -> 240  (-8.4%)
GPR stores: 216 -> 214  (-0.9%)
```

Only 22 loads were eliminated, against 150 removed preambles. Counting what
kills a pending store-to-load value in the intrun=8 output:

| barrier | pending values killed |
|---|---|
| **calls** (81 of them) | **207** |
| remaining atomic loads (54) | 177 |

**Calls kill more forwarding than the interrupt check does**, even after 74% of
the checks are gone. A call is an opaque clobber that invalidates *every*
pending value at once, and this region has 81 calls across ~200 instructions —
one every ~2.5 instructions, far denser than the 0.9% corpus-wide figure
[[block-fragmentation-blocks-cse]] used to dismiss callouts as a barrier. That
dismissal was right about average density and wrong about the regions where
forwarding would actually pay.

Forwarding needs a *clear window* between a store and the load of the same
address. Removing one of two interleaved barrier kinds does not create one.

### It is not calls either — there is simply little to forward

Restricting the measurement to call-free windows (the straight ALU runs where
hoisting was supposed to pay) shows the barrier removal working exactly as
designed, and still not producing loads:

| inside call-free windows (avg ~60 instrs) | intrun=1 | intrun=8 |
|---|---|---|
| pending values killed by `atomic_load` | 182 | **70** (-61%) |
| GPR loads in those windows | 262 | **240** (-8.4%) |

61% fewer values destroyed, 8.4% fewer loads. The barrier was not the binding
constraint. Breaking down what the 240 remaining loads *are*:

- **102 (42%) are loads of GPRs the region never stores.** The single hottest,
  `core.gpr[]+0x68`, is loaded **79 times and stored 0 times**. There is no
  store to forward from — store-to-load forwarding is definitionally
  inapplicable.
- The rest are registers that are written, but with a call between the store
  and the load, or genuinely needed from memory.

That is the real shape of MIPS code under this JIT: most register reads are of
values written by some *earlier region*, not by the instruction two slots back.
Adjacent store-then-reload of the same GPR — the pattern the probe hand-built —
is comparatively rare.

**What would actually address those 79 loads is a different optimization**: a
per-region register cache (load a hot GPR into a host register at region entry,
keep it there, spill at exit). That is not redundancy elimination — Cranelift
cannot do it, because any callout might write `gpr[n]` and it has no way to
know otherwise. It requires the JIT to assert which GPRs a region's callouts
cannot touch. Strictly more work than this knob, and the only path to the win
this note originally claimed.

### What this says about the probe

`zz_cl_forwarding` proved the mechanism exists: remove the seqcst load and
Cranelift forwards. It could not say how often the precondition holds, because
it contained two instructions, one store, one load and nothing else. Real code
has a callout every few instructions. **A microbenchmark that isolates a
mechanism cannot tell you its frequency** — that needs the corpus, measured
directly (load counts) rather than by a proxy (total bytes) that moves for
other reasons.

## It does NOT make anything measurably faster

**Every number above is static** — emitted bytes and instruction counts, not
cycles. A live `lightning` boot (i.e. `opt_level=speed`, knob genuinely
active) showed **no benchmark difference**. Treat the code-size win as real and
the performance win as unproven-to-absent.

Why a large static win can be worth ~nothing at runtime, in order of
likelihood:

1. **The eliminated loads were already free.** `core.gpr[]` is a hot,
   L1-resident struct. A redundant load that hits L1 and is not on the
   dependency critical path costs approximately nothing on an out-of-order
   core. Removing 17% of load *instructions* can remove 0% of *cycles*.
2. **The `atomic_load` was not a hardware cost either.** seqcst *load* on
   x86-64 is a plain `mov` — no fence, no `lock` prefix. It constrains the
   compiler, not the CPU. So deleting it shrinks code without removing any
   real serialization.
3. **The benchmark may not be JIT-code-bound** — time in memory callouts,
   REX3, device threads or TLB work is untouched by this.

The methodological lesson, which is the durable part: **emitted-code size is
not a proxy for speed here.** Every size claim in `rules/jitv2/` should be read
with that caveat unless it was paired with a wall-clock measurement. This one
was not, until the boot benchmark contradicted it.

Given that, the default stays **1**. The knob is kept because it is cheap, is
the only way to test the hypothesis further, and may matter on an in-order or
memory-bound host — not because it is known to pay.

One secondary effect is real regardless of throughput: at intrun=16 the live
boot held **151.9 MB of code where intrun=1 held 180.8 MB**. That is less
code-arena pressure and so fewer exhaustion-driven `mega_flush`es on a long
run, which is worth something on a memory-constrained host even when it never
shows up in guest MIPS.

## Measuring this yourself

`IRIS_INTRUN=<n>` seeds the knob process-wide, read once via `Once` in
`Codegen::interrupt_run()`. That seeding exists because the equivalence tests
and `jitv2_pcp_dump` build their own `Codegen` and never call the setter — an
earlier version only wired the env var into `zz_corpus_sizes`, so
`IRIS_INTRUN=8 cargo test equiv_test` silently verified the default and
`jitv2_pcp_dump` printed two byte-identical listings for different settings.
Both now report the effective `opt_level`/`intrun` they used.

**Do not run full-corpus sweeps unbounded.** Cranelift parallelizes hard: four
sequential 1427-page passes pinned 61 cores at 6106% CPU for 49 minutes and
made the machine unusable. Use a subset for sweeps.
