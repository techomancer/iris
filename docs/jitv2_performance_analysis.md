# jitv2 performance — where to spend effort next

Measured with `bench/` (`iris-bench matrix`) on 2026-08-21, Core i5-9500T, Linux
x86_64. Every number below is guest instructions retired per host second, from
the test device's `ICOUNT` register — the same counter in both engines, so the
ratios are real.

The key table is **jitv2 vs interpreter with build flags held constant** (both
cells `lightning`, so opcodefusion and breakpoint removal cancel out). That
isolates what the JIT itself contributes:

| kernel class | JIT contribution |
|---|---|
| integer ALU (`int/alu`, `muldiv`, `bitops`, `alu64`) | **5.2 – 6.7×** |
| imaging / codec | 2.6 – 5.1× |
| FP arithmetic (`fpu/scalar_d`, `scalar_s`, `divsqrt`) | **0.98 – 1.01× — nothing** |
| memory streaming (`mem/copy`, `stream_*`) | **0.75 – 0.86× — the JIT loses** |
| region-boundary kernels (`sys/cache_flush`, `sys/llsc`) | **0.67 – 0.71× — the JIT loses** |

Headline cells for reference:

| cell | guest MIPS | DMIPS |
|---|---:|---:|
| r4400-interp | 51.0 | 70.9 |
| r4400-lightning | 75.3 | 119.7 |
| r4400-jitv2 | 202.9 | 213.0 |
| r4400-jitv2-lightning | 234.1 | 299.6 |

---

## Recommendations

Confidence = probability the change produces a measurable net win on the kernels
named, given the code as it stands today.

| # | Do this | Confidence |
|---|---|---:|
| **R0** | Add a JIT-coverage counter (retired instructions executed in compiled code / total) before touching anything else | **95%** |
| **R1** | Inline the memory fast path instead of `call_indirect` per access | **85%** |
| **R2a** | Guard regions on `FCSR enables == 0` and hoist the MXCSR round-trip out of per-instruction FP | **75%** |
| **R4** | Check the JIT gate *before* `fetch_instr`/decode in non-lightning builds | **80%** |
| **R3** | Enable interpreter-fallback region admission so `CACHE`/`LL`/`SC` stop ending regions | **70%** |
| **R5** | Coarsen the per-instruction preamble (masked-region elision; batch the cycle counter) | **60%** |
| **R2b** | Replace MXCSR entirely with analytic IEEE flag computation in IR | **55%** |

Ruled out after checking: `opt_level` is already `speed` in non-`developer`
builds (`codegen.rs:277`), so this is not a "turn the optimizer on" problem.

---

## High-level findings

**FP gets exactly nothing from the JIT, and the reason is not dispatch.** Every
FP arithmetic emitter calls `emit_fpu_clear_status` (STMXCSR + LDMXCSR) before
the operation and `emit_fpu_update_fcsr` (STMXCSR, then another clear) after —
both as `call_indirect` through function pointers on `MipsCore`. Two LDMXCSR per
emulated FP instruction, each of which serializes the host FP pipeline. The
interpreter pays the same cost, which is why the ratio is 1.00×. FP kernels run
at 19–25 MIPS against 630 for integer ALU. Not an x86 quirk: the aarch64 path
(`platform.rs:160`, `177`) is the same shape with `mrs`/`msr fpsr`, so an Apple
Silicon host has the same problem.

**On memory-streaming loops the JIT is a net loss.** `emit_mem_read` /
`emit_mem_write` load `jit_ctx`, load a function pointer, `call_indirect`, then
load and branch on `core.jit_mem_exc` — per access. The interpreter inlines the
same work: `read_data`, `write_data` and both `*_impl` bodies all carry
`#[inline]` (`mips_exec.rs:3215`, `3384`), so an interpreted `lw` has no call at
all where a compiled one has an indirect call it cannot inline through. When a
loop is more than half memory operations, that overhead exceeds the dispatch the
JIT saves.
`jit-v2-design.md` §3.3 already sanctions the fix — "compiled loads/stores call
**or inline the fast path of** the shared memory-access core" — the second half
was never built.

**Region boundaries are expensive and more common than they need to be.**
`JR`/`JALR` always exit; `J`/`JAL` are treated as page-leaving and always exit;
`CACHE`, `LL`, `SC`, `SYSCALL`, `BREAK` are `Excluded`, and
`analyzer.rs:34 FALLBACK_ENABLED` defaults **off**, so an excluded word ends the
region rather than being admitted as an interpreter-fallback head — even though
`emit_interp_fallback_head` is written and working. A loop containing one LL/SC
pair therefore pays a full exit + re-entry every iteration.

**Region re-entry is more expensive than it looks in non-lightning builds.** The
JIT gate in `exec_decoded` runs *after* `step()` has already done
`fetch_instr` — nanotlb translate, I-cache probe, and decode. Only `lightning`
builds get `jitv2_try_dispatch_without_decode`, which checks the gate first.
Measured cost of that difference: +15% overall, **+41% on Dhrystone**, up to
+58% on the most exit-heavy kernels.

**The per-instruction preamble is a compiler barrier, not just three µops.**
`emit_pending_interrupt_preamble` emits a **sequentially-consistent
`atomic_load`** before every instruction, plus `emit_increment_cycles`
(load/add/store of `hot.cycles`). §5 of the design doc assumes "within-block
value forwarding comes largely free from Cranelift GVN/alias analysis" for the
memory-resident register file — a SeqCst load between every pair of instructions
is exactly what stops that. The design doc's own §3.2 already prices the
preamble at 30–50% of the per-unit budget and lists the coarsening roadmap;
item 1 (masked-region elision) is described there as unobservable and
strict-validate-safe.

**Caveat on R5's measured upside.** `bench/` runs bare metal with `Status.IE`
clear throughout, so masked-region elision would apply to the *entire* suite.
Real IRIX userland runs unmasked. Expect the benchmark to overstate that
specific change; use `sys/*` and a real IRIX boot to sanity-check it.

**What is not a problem.** R5000 vs R4400 under jitv2 is within a few percent
(202.9 vs 211.6 MIPS), so the 2-way L1 model is not a JIT-side cost. Accuracy is
100% (40/40 checksums) on every cell measured, including both lightning cells —
none of this is a correctness tradeoff being paid for speed.

---

## Resume prompt

Paste the block below into a fresh session to pick this up. It is written to be
self-contained.

---

> You are continuing a performance investigation into `jitv2`, the Cranelift
> region compiler in this repo (`src/jitv2/`). Read `docs/jitv2_performance_analysis.md`
> (this file) and `rules/jitv2/jit-v2-design.md` first — especially §3.2
> (interrupt sampling contract and its coarsening roadmap), §3.3 (exit stubs and
> the memory-helper ABI), §4.4 (excluded instructions) and §5 (memory-resident
> register state).
>
> **What is already established** (measured, not assumed — reproduce with
> `iris-bench matrix --cells r4400-lightning,r4400-jitv2-lightning`):
> with build flags held constant, jitv2 gives 5–6.7× on integer ALU kernels,
> 2.6–5.1× on imaging/codec, **1.00× on FP arithmetic**, **0.75–0.86× on memory
> streaming**, and **0.67–0.71× on `sys/cache_flush` and `sys/llsc`**. The last
> two classes are cases where turning the JIT on makes the guest slower.
>
> **Your job**, in this order:
>
> **Step 0 — instrument (R0).** There is no way today to tell "compiled but
> slow" from "never compiled": `j2 stats` (`src/mips_exec.rs:10459`) reports
> pages, functions compiled, arena bytes and mega-flushes, but no coverage.
> Add a counter of retired instructions executed inside compiled code — the
> natural place is alongside `emit_increment_cycles` (`codegen.rs:2387`),
> incrementing a second `MipsCore` field, with the ratio against `hot.cycles`
> reported by `j2 stats`. Then re-run the suite and record per-kernel JIT
> coverage. Everything below is much cheaper to evaluate once this exists, and
> several of the hypotheses may resolve immediately.
>
> **Step 1 — memory helper ABI (R1, highest expected value).** Read
> `emit_mem_read`/`emit_mem_write`/`emit_check_mem_exc` (`codegen.rs:2552`,
> `2592`, `2676`) and the wrappers they call (`jit_read32`/`jit_write32`,
> `mips_exec.rs:1189`, `1229`). Each access is: load `jit_ctx`, load a fn
> pointer, `call_indirect`, load `core.jit_mem_exc`, compare, branch. The
> interpreter inlines the same `read_data`/`write_data`. Emit the hit path
> inline in IR — the nanotlb probe and the cached-RAM hit — and keep the
> `call_indirect` only as the miss/MMIO/fault tail. §3.3 already sanctions this
> shape. Target kernels: `mem/copy`, `mem/stream_copy`, `mem/stream_scale`,
> `mem/stream_triad`; success is getting them above 1.0× against the
> interpreter, and the imaging/codec kernels should move too since they are
> load/store dense.
>
> **Step 2 — FP status handling (R2).** Read `emit_fpu_clear_status`
> (`codegen.rs:2754`), `emit_fpu_update_fcsr_with_inexact_override`
> (`codegen.rs:2996`), `emit_fbinop_d` (`codegen.rs:4854`), and
> `platform::x86_64::{get_fpu_status, clear_fpu_status}` (`src/platform.rs:82`,
> `99`). Two LDMXCSR per emulated FP instruction is the cost.
> Do **R2a first**: extend the existing region-wide FPU guard
> (`emit_fpu_entry_guard`, `codegen.rs:1712` — it already guards CU1 and FR the
> same way) with an "FCSR enable bits are all zero" condition. In that mode no
> FP instruction can trap, so Cause/Flag only need to be correct at the next
> *observation* point — a CFC1, an FP branch, or a region exit. Clear MXCSR once
> at region entry, accumulate, and write FCSR at the exit stub. R2b (compute
> V/Z/O/U analytically in IR and drop MXCSR entirely) is the bigger win but
> Inexact is genuinely hard; note that the emitters already compute subnormal
> and NaN predicates in IR (`emit_is_subnormal_or_qnan_d`, `codegen.rs:3251`),
> and `emit_round_and_convert` already overrides Inexact analytically, so the
> precedent exists.
>
> **Step 3 — dispatch shortcut (R4).** `jitv2_try_dispatch_without_decode`
> (`mips_exec.rs:2413`) is `cfg(all(jitv2, lightning, …))`. In non-lightning
> builds the gate in `exec_decoded` (`mips_exec.rs:6560`) only runs after
> `fetch_instr` has translated, probed the I-cache and decoded. Work out what
> actually blocks using the shortcut without `lightning` — the PC-breakpoint
> check is the obvious one; the I-cache probe is already sanctioned as skippable
> by §8.1 — and gate it on "no breakpoints armed" rather than on the feature.
> Measure with `--cells r4400-jitv2,r4400-jitv2-lightning`; the remaining gap
> after the change is the part that was really opcodefusion.
>
> **Step 4 — region admission (R3).** `FALLBACK_ENABLED` (`analyzer.rs:34`)
> defaults false, so an `Excluded` word ends the region. `j2 fallback on` flips
> it at runtime and `emit_interp_fallback_head` already implements the admitted
> path. This is nearly free to evaluate: turn it on, re-run, look at
> `sys/cache_flush` and `sys/llsc`. If it holds up under a real IRIX boot,
> consider defaulting it on.
>
> **Step 5 — preamble coarsening (R5, do last).** `emit_pending_interrupt_preamble`
> (`codegen.rs:1631`) emits a SeqCst `atomic_load` per instruction;
> `emit_increment_cycles` (`codegen.rs:2387`) emits a load/add/store per
> instruction. Two separable questions: (a) does the SeqCst load defeat
> Cranelift's alias analysis and block the store-to-load forwarding §5 assumes —
> test by switching to a plain `load` and measuring `int/alu`, which is the
> cleanest ALU-density signal; (b) can `hot.cycles` be batched per straight-line
> run — outside `ci_clock`, Count is virtual and wall-clock anchored, so
> `hot.cycles` is not the timer source, and an exit stub can add a
> statically-known constant. Then attempt roadmap item 1, masked-region elision.
> **Remember the benchmark runs with IE clear throughout**, so it will overstate
> that last one; cross-check against a real IRIX boot.
>
> **Guardrails — run all three before believing any result:**
> - `make -C bench && iris-bench matrix` — accuracy must stay 100% (40/40) on
>   every cell. A checksum mismatch means the change altered guest-visible
>   results.
> - `make -C cpu-tests && make -C cpu-tests run` — expect 2160 checks passed, 2
>   failed. The two failures are the known FPU-flag findings
>   (`fpu/cvt_s_d_rounds`, `fpu/cvt_out_of_range`); anything else is new. This
>   matters most for step 2 — the FP flag model is exactly what those tests
>   cover.
> - `cargo test --release` — 387 lib tests, plus `src/jitv2/equiv_test.rs`, which
>   is the JIT-vs-interpreter differential net.
>
> **Reporting.** Every claim should come with a number from `iris-bench`, not an
> expectation. Run-to-run variation on an idle machine is under 1.1%, so a 5%
> change is real and anything under 2% is noise. Write findings into
> `rules/jitv2/` as short notes, and update the table at the top of
> `docs/jitv2_performance_analysis.md` with what actually happened — including
> the recommendations that turned out to be wrong.

---

## Appendix — full per-kernel JIT contribution

`r4400-lightning` → `r4400-jitv2-lightning`, guest MIPS. Both cells are
`lightning`, so this is the JIT's own contribution with everything else equal.

| kernel | interp | jitv2 | ratio |
|---|---:|---:|---:|
| sys/cache_flush | 73.0 | 48.7 | 0.67× |
| sys/llsc | 95.4 | 67.7 | 0.71× |
| mem/stream_copy | 12.0 | 9.0 | 0.75× |
| mem/stream_scale | 13.5 | 10.4 | 0.77× |
| sys/exception | 74.8 | 60.8 | 0.81× |
| mem/copy | 12.2 | 10.3 | 0.85× |
| mem/stream_triad | 15.5 | 13.2 | 0.86× |
| sys/tlb_miss | 27.5 | 24.2 | 0.88× |
| mem/random | 31.3 | 29.7 | 0.95× |
| fpu/scalar_s | 25.7 | 25.1 | 0.98× |
| fpu/divsqrt | 18.8 | 18.8 | 1.00× |
| fpu/scalar_d | 25.2 | 25.4 | 1.01× |
| mem/latency_dram | 9.2 | 9.4 | 1.02× |
| fpu/transcend | 26.9 | 31.3 | 1.17× |
| fpu/whetstone | 29.3 | 37.7 | 1.29× |
| fpu/linpack | 42.7 | 59.0 | 1.38× |
| mem/fill | 68.6 | 95.8 | 1.40× |
| sys/tlb_hit | 81.1 | 140.2 | 1.73× |
| mem/latency_l2 | 87.5 | 156.4 | 1.79× |
| fpu/matmul | 59.4 | 110.1 | 1.85× |
| int/dhrystone | 89.2 | 223.2 | 2.50× |
| img/convolve3x3 | 85.1 | 220.1 | 2.59× |
| mem/unaligned | 88.8 | 235.2 | 2.65× |
| img/histogram | 88.9 | 237.9 | 2.68× |
| mem/latency_l1 | 92.4 | 262.9 | 2.84× |
| img/composite | 94.8 | 279.1 | 2.94× |
| img/sharpen5x5 | 104.5 | 315.8 | 3.02× |
| img/rotate90 | 102.8 | 318.0 | 3.09× |
| codec/adler32 | 117.6 | 384.2 | 3.27× |
| vid/motion_est | 105.6 | 351.0 | 3.32× |
| codec/crc32 | 105.1 | 353.7 | 3.36× |
| img/resize | 94.6 | 324.0 | 3.42× |
| codec/lz | 96.3 | 336.7 | 3.50× |
| codec/huffman | 104.3 | 371.1 | 3.56× |
| sys/uncached | 102.9 | 370.7 | 3.60× |
| codec/rle | 95.2 | 349.5 | 3.67× |
| img/dither | 90.9 | 348.5 | 3.83× |
| img/rgb2ycbcr | 89.4 | 347.9 | 3.89× |
| int/branch | 99.3 | 460.5 | 4.64× |
| img/dct8x8 | 66.6 | 322.1 | 4.84× |
| vid/yuv2rgb | 92.2 | 471.6 | 5.11× |
| int/bitops | 125.3 | 656.0 | 5.23× |
| int/alu64 | 119.1 | 628.4 | 5.28× |
| int/alu_ilp | 116.2 | 655.3 | 5.64× |
| int/alu | 105.7 | 625.9 | 5.92× |
| int/muldiv | 101.0 | 673.2 | 6.66× |

Relevant tunables, for the record: `MIN_CALLS_BEFORE_COMPILE = 4`
(`jitv2.rs:54`), `MAX_INSTRS_PER_COMPILE = 128` (`comp.rs:57`),
`FALLBACK_ENABLED = false` (`analyzer.rs:34`), `opt_level = speed`
(`codegen.rs:277`).
