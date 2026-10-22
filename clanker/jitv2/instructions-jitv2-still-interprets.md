# What jitv2 still interprets, and which of it is worth compiling

Historical measurements; current builds select MIPS IV through the CPU model,
not a Cargo feature. IP28/R10000 is now a supported runtime profile.

Surveyed 2026-09-22, after enabling `mips4` turned out to be worth ~20% on
integer code and exactly nothing on FP. This is the follow-on question: what
else is left on the table.

## The gap

Of 204 `InstrKind` variants, **31 have no jitv2 emitter** — computed by
diffing the enum against `has_jitv2_emitter()` + `has_jitv2_support()` in
`mips_instr_stats.rs`.

They fall into three groups:

**Privileged / system (17) — correctly interpreted, and they stay that way.**
`Syscall Break Mfc0 Dmfc0 Mtc0 Dmtc0 Tlbr Tlbwi Tlbwr Tlbp Eret Wait Cache`
plus the atomics `Ll Sc Lld Scd`. These need to trap into the emulator by
nature, and none of them will ever get a native emitter.

That was never the expensive part, though. CP0 access *ending* a compiled
region was — a separate problem from emitter coverage, and **fixed 2026-09-22**:
the safe subset now stays in-region as an interpreter-fallback head, worth
~13-15% on syscall-bound work. See
[`cop0-does-not-have-to-end-a-region.md`](cop0-does-not-have-to-end-a-region.md),
including why three different measurements said "no change" before one found
it.

**MIPS IV FP arithmetic (13) — DONE 2026-09-22, ~3x on FP code.**
`Madd_s Madd_d Msub_s Msub_d Nmadd_s Nmadd_d Nmsub_s Nmsub_d`,
`Frecip_s Frecip_d Frsqrt_s Frsqrt_d`, `Prefx`.

**`Bc1` (1) — DONE 2026-09-22, ~4x on a BC1-heavy loop.** It had been worse
than a fallback: `classify` returned `Classify::Excluded` for `RS_BC1`, so
branch-on-FP-condition *terminated* the compiled region rather than merely
bailing for one instruction. It is now classified as the ordinary
PC-relative branch it always was.

**So the remaining 17 are all privileged or atomic**, and belong in the
interpreter. On emitter coverage alone there is nothing left worth taking.

**MIPS III has no gaps at all.** Every MIPS III compute instruction already
has an emitter. There is nothing to win by looking there.

## Measured usage, not guessed

Disassembled the actual MIPSpro `-Ofast -mips4` guest binaries (N32 MIPS-IV)
with `cross-binutils/bin/mips-sgi-irix6.5-objdump`:

| | `movz`/`movn` (mips4 enables) | MADD-family | `bc1*` | measured result |
|---|---|---|---|---|
| `dhry` | 7 + 3 | 0 | 0 | **20% faster with mips4** |
| `whetstone` | 0 | 29 `madd.d` + 18 `nmsub.d` | 1 | **flat** |

That is the whole story of the `mips4` A/B in one table, and it is causal
rather than correlational.

**`/usr/lib32/libm.so` is where it really bites** — every transcendental any
FP program calls:

| instruction | sites |
|---|---|
| `madd.d` | **2554** |
| `nmsub.d` | 528 |
| `bc1t` + `bc1f` | 456 + 396 = **852** |
| `msub.d` / `madd.s` / `msub.s` / `nmsub.s` / `nmadd.d` | 180 |
| `recip.d` / `recip.s` | 17 |

## DONE 2026-09-22: the multiply-add family, RECIP/RSQRT and PREFX

Thirteen emitters landed. Measured on IP28 (R10000), 5 reps, fresh clone and boot,
host wall clock:

| | Dhrystone 50M | Whetstone 1M |
|---|---|---|
| mips4, before | 33 s | 19 s |
| mips4, after | 33 s | **6-7 s** |

**About 3x on FP code.** Dhrystone is unchanged to the second, exactly as the
site counts predicted — it contains no MADD at all. Coverage went
`fpu 64 -> 77`, `loadstore 28 -> 29`.

Correctness was checked on the real workload, not just in unit tests: build
Whetstone with `-DPRINTOUT` so it prints its computed values, run it, then
`cpu stop` / `j2 fpu off` / `j2 flush` / `cpu start` to force the whole FPU
category back to the interpreter and run the identical binary again. **Every
computed value across all twelve modules was identical.** That technique is
worth reusing for any future emitter: it compares JIT against interpreter on
real code in a single boot, no harness required.

What remains is the privileged/atomic set — `Bc1` landed too (below).

## Ranked, with the work each needs

1. ~~**MADD/MSUB/NMADD/NMSUB — 8 emitters. Do this one.**~~ **Done — see above.**
   ~3300 sites in libm alone. Pure arithmetic with no memory or addressing
   complexity; the existing `fadd`/`fmul` emitters are the template.

   **It must NOT be a fused multiply-add** (corrected 2026-09-27). MIPS IV
   rounds the product to the format and then adds, rounding again: the
   R10000 User's Manual describes one pass through the multiplier and a
   second through the adder, and compilers (MIPSpro, LLVM) emit MADD for
   `a*b + c` on that understanding. The emitters were first written as
   Cranelift `fma` to match an interpreter that used Rust's fused
   `mul_add`; both were wrong, and both are now `fmul` then `fadd`/`fsub`
   (`exec_madd_*`, `emit_fternop_*`). A guest test that tells them apart:
   fs*ft = 1 - 2^-60 exactly, fr = -1 gives 0 unfused, -2^-60 fused.

   Watch the operand mapping: COP1X puts `fr` in `rs`, `ft` in `rt`, `fs` in
   `rd` and `fd` in `sa`, and the result is `fd = fs*ft + fr`. NMADD/NMSUB
   negate the result. Flags: Invalid iff any of the *three* sources is a
   signalling NaN (`fpu_arith_flags_snan_only3_d`).

2. ~~**`Bc1` — 852 sites in libm, and the cost compounds.**~~ **Done
   2026-09-22 — worth ~4x on a BC1-heavy loop.** The analyzer's "the target is
   condition-code dependent" reasoning turned out to be wrong: the target is
   the same PC-relative offset every conditional branch uses, and only the
   predicate is CP1. See
   [`bc1-is-an-ordinary-branch.md`](bc1-is-an-ordinary-branch.md), including
   why measuring it required writing the kernel in assembly.

3. **RECIP/RSQRT — done; 4 emitters, 17 sites.** Low value, but trivial next to the
   MADD work and shares its shape.

4. **`Prefx` — done; 1 emitter, emits nothing after the CU1 check.** A prefetch is architecturally a
   hint; `Pref` is already in the compiled set and `Prefx` is its indexed
   form. Near-zero sites in what we run, but it is a one-line arm.

## Gating — fixed 2026-09-22

These are all MIPS IV, and the `mips4` cargo feature that used to gate them
was on the wrong axis: ISA level belongs to the CPU model, which is a runtime
choice. `MipsExecutor::new` passes `C::MIPS4` to its inline analyzer
and compile pool; each worker uses that model's ISA at
`opcode_support::has_emitter`. The `jitv2::isa` global is only a default
for constructors/tools without a CPU. Every MIPS IV emitter is built in. See
[Build and execution settings](../../HACKING.md).

Use host wall-clock timing for these comparisons. The guest-CPU-time
accounting note cited by the original investigation was not committed.
