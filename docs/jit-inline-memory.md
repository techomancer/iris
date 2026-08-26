# JIT inline loads and stores

Status: **first attempt implemented and REVERTED — 9x slower.** The scaffolding
(pointer plumbing, geometry, `jitstats` measurement) is sound and kept; the
emit strategy is not. See §7 before attempting this again.

Original status line: design, not implemented. Scope: aligned 8/16/32/64-bit loads and
stores only. Unaligned (LWL/LWR/LDL/LDR/SWL/SWR/SDL/SDR) keeps calling out —
rare, and `write64_masked` already handles it.

Why this matters: every guest memory access currently costs a C-ABI call
through `core.read{8,16,32,64}_fn` plus a `jit_mem_exc` check. No JIT reaches
QEMU-TCG / MAME-DRC class performance while that holds. `nutlb` and `tcache`
were built to make this possible; this is the payoff.

---

## 1. Measured coverage (feature = "jitstats")

`iris-bench run`, non-tcache, nutlb on. Each row is a prefix of the next:

| stage | loads (464M) | stores (150M) |
|---|---|---|
| nutlb hit | 97.2% | 96.1% |
| + L1D tag hit | **81.8%** | 75.4% |
| + dirty (stores) | — | **75.4%** |
| **inline-eligible** | **81.8%** | **75.4%** |

**Translation is not the constraint — the L1D tag is.** nutlb delivers 97%,
then the tag check costs another 15–21 points. The residual callouts are
genuine cache misses, which must call out anyway to run the fill, so ~82%/75%
is close to the ceiling for this structure without inlining fills too.

Two caveats before trusting these numbers for design decisions:

- Bare-metal. Live IRIX has consistently differed from this suite (see
  `docs/nutlb-design.md` §9); re-measure there before tuning.
- The store `dirty` row shows *zero* additional drop, i.e. every tag-hit store
  was already dirty. Plausible for a benchmark that rewrites buffers, but if it
  holds on IRIX too then the clean→dirty callout is free and needn't be
  designed around.

---

## 2. The emitted sequence

Both variants share stages 1–2 and differ only in where the data lives.

```
1. nutlb probe          VA -> phys      miss  -> callout
2. L1D tag match        idx + 1 cmp     miss  -> callout   (Rust runs the fill)
   stores: line dirty?                  clean -> callout   (Rust does the RMW)
3. tcache only: transparent(phys)       clear -> callout   (MMIO / unmapped)
4. data access          base + swizzle
   stores: mark dirty (non-tcache) + jitv2 gen bump
```

Non-tcache is the *shorter* path: a resident line is authoritative regardless
of whether the region is mapped RAM, so no bitmap test is needed.

### 2.1 Where the data comes from

| | tcache off | tcache on |
|---|---|---|
| source | `dc.data` (the line holds the bytes) | `tc_base + phys` (ppmem window) |
| store | write `dc.data`, set tag dirty | write window, set tag dirty |
| bitmap test | not needed | required |

Both are correct in their own build. Under tcache the L1D holds tags but no
data, so the tag check is **not** protecting read correctness — RAM is the
right source either way. It is preserving *cache-model fidelity*: fills, LRU,
and dirty state that the guest can observe through CACHE ops and writebacks.
Worth stating explicitly, because it is what determines whether stage 2 could
ever be dropped under tcache (it could, at the cost of that fidelity — do not,
without deciding that deliberately).

### 2.2 Byte swizzles — the easiest thing to get wrong

Guest memory is stored word-swapped ("The Edge", HACKING.md). The two data
paths use **different** swizzles because their base units differ — bytes into
the ppmem window vs `u64` chunks in the cache array:

| access | tcache (`tc_base + phys`, byte-indexed) | non-tcache (`dc.data`, u64-indexed) |
|---|---|---|
| 64-bit | load u64, then `rotate_left(32)` | `data[masked >> 3]`, no rotate |
| 32-bit | plain u32 at `off` | `words[(masked >> 2) ^ 1]` |
| 16-bit | `halves[(off >> 1) ^ 1]` | `halves[(masked >> 1) ^ 3]` |
| 8-bit | `bytes[off ^ 3]` | `bytes[masked ^ 7]` |

(`masked = virt_addr & (DC_SIZE - 1)`; source: `tc_read` and `dc_read` in
`mips_cache_v2.rs`.)

Codegen needs two tables selected by the tcache cfg. A wrong constant produces
byte-scrambled data only on sub-word accesses — it survives whetstone and dies
somewhere unrelated later. **Lockstep-test every (path × size) combination
before trusting any of it.**

---

## 3. What has to change outside codegen

### 3.1 `L1DTag` needs `#[repr(C)]`

Currently `repr(Rust)`: measured as 16 B, `ptag` at 0, `dirty` at 9 — but none
of that is guaranteed. Compiled code will hardcode those offsets, so the type
needs the same treatment `MipsCore` already documents for itself: `#[repr(C)]`
plus `offset_of!` at codegen time, never a literal.

### 3.2 Pointer block in `MipsCore`

Compiled code reaches everything through `core_ptr` at fixed offsets, but the
cache state lives in `C: Cache`. Mirror what the fast path needs into
`MipsCore`, installed alongside the existing jitv2 hooks:

```rust
pub jit_dc_tags:   *mut u8,   // L1D tag array base
pub jit_dc_data:   *mut u8,   // L1D data array base (non-tcache)
pub jit_tc_base:   *mut u8,   // ppmem window base   (tcache)
pub jit_tc_bitmap: *const u64,// mapped-region bitmap (tcache)
pub jit_gen_base:  *mut u8,   // jitv2 generation window
```

These are raw pointers into boxed slices with stable addresses, so caching them
is safe — **provided they are refreshed anywhere the backing allocation can
change**: snapshot restore, cache resize/reconfigure, ppmem remap. Getting that
wrong is a dangling-pointer bug in compiled code, which will not be pleasant to
debug. A single `install_jit_mem_ptrs()` called from the same places as
`install_jit_hooks` is the safest shape.

### 3.3 Geometry as codegen constants

`DC_SIZE`, `DC_LINE`, `LINE_SHIFT`, `NUM_LINES_MASK` are const-generic on the
cache type. Codegen must read them from the instantiated `C` (they differ
R4400 vs R5000), not hardcode R4400's.

---

## 4. Scope limits for the first cut

- **1-way only (`!IS_R5K`).** R5000 is 2-way; the data address folds the way
  into the index (`dc_data_addr`), so inline needs way selection. R5000 keeps
  calling out. `jit_probe` already reports R5K as never-inlinable so the
  measured numbers stay honest.
- **Aligned only.** Misaligned addresses fall back; the address-error path
  stays entirely in Rust.
- **Stores only on already-dirty lines.** The clean→dirty transition is a tag
  read-modify-write; leave it to Rust. First store to a line calls out, the
  rest inline. (§1 suggests this may cost nothing at all.)
- **No inline fills.** A tag miss calls out and lets Rust run the fill.

---

## 5. Correctness obligations

1. **jitv2 generation bump on every inline store.** The fast path must not skip
   the gen bump the Rust path would have done, or self-modifying code silently
   stops invalidating compiled regions. `gen_ptr` is a pure shift
   (`gen_base + (phys >> 9)`, `GEN_RATIO = 512`) so this is cheap — but it is
   not optional, and it is invisible until something breaks weirdly.
2. **Watchpoints / breakpoints.** `check_breakpoint::<PhysRead/PhysWrite>` runs
   on the Rust path. Inline code skips it. Either gate the whole fast path off
   when breakpoints are active (simplest, and consistent with `lightning`
   already trading debuggability for speed), or emit the check.
3. **`developer` undo buffer** likewise records writes on the Rust path.
4. **Lockstep is the acceptance test.** `jitv2_lockstep` compares every JIT
   instruction against the interpreter inline; that is exactly the right net
   for swizzle and dirty-state bugs. Do not declare this working on benchmark
   throughput alone.

---

## 6. Suggested order

1. `#[repr(C)]` on `L1DTag`; pointer block + `install_jit_mem_ptrs`.
2. Loads only, 32-bit only, non-tcache, `!IS_R5K`, behind a runtime `j2` toggle
   so it can be A/B'd live. Validate under lockstep.
3. Remaining load sizes, then stores (with gen bump + dirty rules).
4. tcache variant (bitmap test + window swizzles).
5. Re-measure on live IRIX; only then consider R5000 or inline fills.


---

## 7. Attempt 1: why the obvious emit is 9x SLOWER

Implemented exactly as §2 describes — per-access guard chain, fast block, slow
block, join with a block param — and measured on `iris-bench run`, one binary,
`IRIS_NO_INLINE_MEM` selecting the path at emit time:

| | MIPS | DMIPS | accuracy |
|---|---|---|---|
| inline OFF | **312.3** | 297.2 | 100% |
| inline ON | 34.3 | 46.4 | 100% |

100% accurate, ~9x slower, uniformly across kernels (`codec/lz` 9.69x).

### The bisect that found it

Forcing `proceed = false` — so the fast block is emitted but **never taken**,
and every access runs the byte-identical old callout — still measured
**34.1 MIPS**. The checks, the swizzles, and the data access are therefore all
innocent. The cost is *structural*: creating the blocks at all.

Page-pool stats were identical either way (13/4096 pages, ~3400 emits), so this
is not compile thrash or region rejection. It is the emitted code being slow.

### Mechanism

`rules/jitv2/jit-v2-design.md:25` names this exact failure as the reason the v1
JIT was abandoned:

> Register pressure: 34 live guest values into 15 x86_64 registers with
> exception paths crossing block boundaries -> spill bugs. v2 answer:
> memory-resident registers (§5).

The guard adds 4-5 blocks **per memory access**. A region with 20 loads goes
from a handful of blocks to ~100, and every join is a point where Cranelift
must reconcile live values. v2's whole performance model is that guest
registers live in memory at fixed offsets off `core_ptr` and the surrounding
code stays in one straight-line block; carving that into a control-flow diamond
per access destroys it.

### What this rules out, and what it doesn't

Ruled out: **any design where the fast path is a branch around a callout inside
the same Cranelift function.** The block structure is the cost, so no amount of
tuning the checks recovers it.

Not ruled out — the coverage numbers in §1 (81.8% of loads, 75.4% of stores are
inline-eligible) are still real and still worth chasing. Candidate directions,
none validated:

1. **Branchless, no blocks.** Compute both a fast address and a fallback, select
   with `select` rather than `brif`. Requires the slow path to be safe to
   *always* execute (or a safe dummy address), which the callout is not — so
   this probably means a different fallback shape, not a different branch shape.
2. **Out-of-line stubs.** Emit the guard as a call to a tiny hand-written stub
   rather than inline IR, keeping the main function single-block. Trades the
   call back in, but a leaf stub is far cheaper than the current wrapper.
3. **Hoist the guard out of the access.** Check once per region/loop for a
   known page rather than per access — amortizes the block cost over many
   accesses. Needs the analyzer to prove the page cannot change.
4. **Bypass Cranelift for this.** The measured ~10-14 instruction sequence is
   trivial machine code; the cost is entirely in expressing it as IR that
   Cranelift then has to schedule. Emitting it as raw bytes around the
   Cranelift-generated body is how DRC-style JITs typically do this.

Direction 4 is the most promising and the most work. Direction 3 is the
cheapest to try. Direction 1 is likely a dead end for stores.

### Kept from this attempt

- `L1DTag` is now `#[repr(C)]` (needed by any of the above).
- `Cache::jit_dc_tags_ptr`/`jit_dc_data_ptr`/`jit_dc_geometry`,
  `tcache_base_ptr`, and the `MipsCore` pointer block + `install_jit_mem_ptrs`.
- `JitDcGeometry` plumbed through `Jitv2` -> `Codegen` -> `EmitCtx`.
- `feature = "jitstats"` and the §1 coverage numbers.
- `Cache::jit_probe`, which is the executable spec of the guard predicate.

All of it is prerequisite for every candidate above, and none of it costs
anything when the emit is disabled.


---

## 8. Attempt 2 (planned): one compiled helper, not per-access IR

§7's failure was block count, not instruction count. So keep the guard in
compiled machine code, but emit it **once per module** as its own Cranelift
function and have every access site `call` it. Access sites go back to a single
straight-line `call` — exactly the shape they have today — so the caller's
block structure, and therefore its register allocation, is unchanged from the
current (fast) baseline.

```
per access site:      call helper_load32(core_ptr, vaddr)   <- one instruction
helper_load32:        guard chain (nutlb, tag, [bitmap]) in ITS OWN function
                      hit  -> load + return
                      miss -> tail-call the existing read32_fn wrapper
```

### Why this should beat both the current callout and attempt 1

- vs **attempt 1**: the guard's blocks live in one small function compiled once,
  not multiplied across every access site in the caller. Nothing crosses the
  caller's register allocation.
- vs **today's callout**: the helper is a leaf that returns without entering
  `read_data` — no breakpoint check, no `translate_fn` indirect call, no cache
  trait dispatch, no undo bookkeeping. Only the ~82%/75% of accesses that miss
  the guard pay the full wrapper.

The remaining cost is the call itself. That is what makes this worth measuring
rather than assuming: if the win is entirely eaten by call overhead, the answer
is direction 4 (raw machine code, bypassing Cranelift), not more tuning here.

### What makes this non-trivial

**The helper's address is stable for the module's lifetime, and a flush takes
everything with it.** `mega_flush` discards *every* compiled function, so after
a flush there is no surviving caller that could reach a moved helper — the
dangling-pointer window simply does not exist. (This is unlike the `pcp` case,
where a pointer outlives the pool it indexes; do not import that hazard here.)

So the only requirement is ordering: emit the helpers into each new module
before the first region compiled against it, and rebuild them alongside
`Codegen` on `new_with_shared_arena`. Within a module, regions can call the
helper by its resolved address.

**One helper per (size x load/store)** = 8 functions, since `SIZE` and the
load/store split are both compile-time in the guard (different swizzle, and
stores need the dirty check). They are small; emitting 8 per module is cheap.

**Calling convention.** The helper needs `core_ptr` and `vaddr`, returns the
value plus a hit/miss indication. Cheapest is to have the miss path tail-call
the existing wrapper and return its result, so the caller cannot tell the
difference and no extra branch is needed at the call site — the `jit_mem_exc`
check the caller already emits covers both.

### Order

1. Emit the 8 helpers into the module at construction; verify they appear and
   that regions can call them (a counter, as in attempt 1 — `emitted=0` was
   what caught that the geometry never reached the worker).
2. Loads only, 32-bit, `!IS_R5K`, non-tcache. Measure against the
   `IRIS_NO_INLINE_MEM` baseline on the same binary.
3. **Only if it wins**, extend to the other sizes, then stores, then tcache.

If step 2 does not clearly beat 312 MIPS, stop and go to direction 4 rather
than tuning.


---

## 9. Attempt 2 results — and a retraction of §7's diagnosis

### §7's conclusion was wrong

§7 blamed the 9x regression on per-access block count wrecking register
allocation, citing the forced-slow-path bisect (fast block emitted, never
taken, still 34 MIPS). That bisect was real but **misinterpreted**: the
slowdown was never about blocks at all.

Attempt 2 built the guard as a single once-per-module helper function, so
access sites are one `call_indirect` with no extra blocks — and it was *still*
33.9 MIPS. Then, with the helper call fully disabled (`emitted=0`, every access
back on the original wrapper), still 33.9 MIPS. The regression survived
removing the thing §7 blamed.

### The actual cause

`build_load32_helper` calls `Module::finalize_definitions()` to resolve the
helper's address. Doing that mid-session breaks the arena's deferred-sealing
model (`paged_memory.rs`: `sealed_up_to`, `try_seal_ready`, and the seal queue
that `finalize_batch` drives via `patch_pending_publish`). The observable
symptom is **runaway recompilation**: the `func_id_counter` heartbeat prints
"10000 / 20000 functions compiled into this Codegen's arena", which the
untouched baseline never prints at all. Regions stop staying published and are
recompiled endlessly; the ~9x is that churn, not any emitted instruction.

Proof, one binary, three env toggles:

| helper built? | helper called? | MIPS | recompile heartbeats |
|---|---|---|---|
| no | no | **304.2** | none |
| yes | no | 33.9 | 10000, 20000 |
| yes | yes | 33.9 | 10000, 20000 |

Building it is sufficient to cause the regression. Calling it changes nothing.

### What is still unknown

**Whether the inline path is a win has not been measured yet.** Both attempts
died before the fast path ever executed — attempt 2's runtime counters read
`runtime_calls=0`, so compiled code never reached the helper even when emission
was enabled. Every performance claim in §7 about "the emit strategy" should be
disregarded; only the §1 coverage numbers (81.8% loads / 75.4% stores
inline-eligible) survive, and those came from `jit_probe`, not from codegen.

### Next step

Resolve the helper's address **without** `finalize_definitions()` mid-session.
Options, in order of preference:

1. Emit the helper through the same path regions use — `define_function` +
   the `func_ranges` / seal-queue reservation that `compile_region_uncommitted`
   does, then `finalize_batch` — so it participates in deferred sealing like
   everything else instead of forcing a global finalize.
2. Emit it lazily as *part of the first region's batch*, so its finalize is
   that batch's finalize.
3. Declare it `Linkage::Import` with a fixed symbol and resolve via
   `JITBuilder::symbol`, sidestepping in-module finalization entirely.

Only after one of those holds 300+ MIPS with `runtime_calls > 0` is there a
number worth reporting for the fast path itself.


---

## 10. It works. The bug was a verifier error, not the strategy.

### Root cause of both failed attempts

Three `uextend(ptr_ty, x)` where `x` was **already i64** (and `ptr_ty` is i64 on
x86-64). Cranelift's verifier rejects a widening that isn't one:

```
VerifierError { context: "v14 = uextend.i64 v13",
                message: "arg 0 (v13) with type i64 failed to satisfy type set ..." }
```

Every region containing a 32-bit load therefore failed `define_function`, got
declined, and fell back to the interpreter. That is the entire 9x. It was the
same three lines in attempt 1 and attempt 2 — the helper surfaced it only
because `build_load32_helper` printed the error, while region compiles swallow
it as an ordinary decline.

**Retract §7 and §9's diagnoses.** Block count did not matter. The
once-per-module helper did not matter. The emitted code simply never ran.

Lesson for next time: `emitted=N` is an *emit-time* counter and proves nothing
about execution. Verify with a runtime counter or a compile-decline count
before drawing any performance conclusion.

### Measured, 32-bit loads only (`iris-bench run`, 3 runs each)

| | MIPS | DMIPS | accuracy |
|---|---|---|---|
| baseline | 302.3 / 314.9 / 314.1 | 292.0 / 293.8 / 292.1 | 100% |
| **inline loads** | **338.3 / 329.4 / 333.6** | **371.1 / 370.8 / 375.7** | 100% |

**+7% MIPS, +27% DMIPS**, ranges non-overlapping, `emitted=10396 declined=0`.
This is with stores and the other three sizes still calling out — §1 says loads
alone are 81.8% inline-eligible, so most of the remaining headroom is stores.

---

## 11. Next: a shared helper page, runtime-switchable

The inline emit wins, but it duplicates the guard at every access site. A
shared helper is still worth having — for code size, and to make the inline
path switchable without recompiling. The design that avoids attempt 2's
mistake:

**Emit all helpers first, into their own page(s), and force-seal them
immediately.** At module construction (and after every `mega_flush`, before any
region compiles), emit the full set — 4 load sizes x 4 store sizes, plus any
variants — into a page or two at the very start of the arena, then seal that
prefix with the existing forced path (`patch_pending_publish(.., force=true)` /
`try_seal_ready_forced`, as `finalize_batch` already does).

Why this is safe where attempt 2 was not:

- Attempt 2 called `Module::finalize_definitions()` **mid-session**, which
  breaks the deferred-sealing model and triggered endless region recompiles
  (the `func_id_counter` heartbeat gives it away: baseline prints none, the
  broken build prints "10000 / 20000 functions compiled").
- Helpers emitted *first* occupy the arena's lowest range. `sealed_up_to` is a
  contiguous watermark, so force-sealing that prefix is exactly the operation
  the arena is built for — and every region compiled afterwards sits above it
  and seals normally.
- Their addresses are then fixed for the module's lifetime, and `mega_flush`
  discards helpers and callers together, so no stale reference can survive.

**Runtime switchable.** With helpers always present, choosing inline-vs-callout
becomes a per-compile decision (emit a `call helper` or a `call read32_fn`),
so a `j2` monitor command can flip it and force a flush — no rebuild needed for
A/B on live IRIX, which is where the numbers that matter come from.

Open question worth measuring: whether the shared helper is actually *faster*
than the duplicated inline emit, or merely smaller. §10's win came from fully
inlined guards; adding a call back in may give some of it up. Keep both paths
selectable and measure.
