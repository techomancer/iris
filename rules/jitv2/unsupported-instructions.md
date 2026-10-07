# JIT v2 instruction coverage and interpreter boundaries

Current policy as of October 2026. Native-emitter coverage and eligibility to
stay inside a region are different questions. The monitor's `j2 instrs`
reports emitter coverage; `opcode_support.rs`, `cop0.rs`, and `atomics.rs`
define the active policy. Category/per-instruction toggles can disable a
supported instruction for diagnosis.

## Native coverage

`src/cpu/jitv2/codegen.rs` provides the integer ALU, branches/jumps, traps,
aligned/unaligned memory operations, FP arithmetic/moves/compares, FP memory,
and MIPS IV additions implemented by the interpreter. BC1 is a PC-relative
conditional branch with native codegen.

The earlier emitter-gap inventory is closed for the supported scalar forms:

- MOVCI/MOVZ/MOVN, DMULT/DMULTU/DDIV/DDIVU, SYNC/PREF, trap/register-immediate
  families, DADDI, and unaligned word/doubleword loads/stores.
- LWC1/LDC1/SWC1/SDC1 and indexed LWXC1/LDXC1/SWXC1/SDXC1.
- FP MOVCF/MOVZ/MOVN, scalar MADD/MSUB/NMADD/NMSUB, RECIP/RSQRT, and PREFX
  (a hint with a CP1 usability check).

MIPS IV is selected by the runtime CPU model: R4400 is MIPS III, R5000 and
R10000 are MIPS IV. Every live analyzer/worker receives its CPU's ISA value;
the `jitv2::isa` global is only a default for tools/constructors without a CPU.
There is no `mips4` Cargo feature.

Paired-single (`RS_PS`, `*_PS`, `CVT.PS`) is deliberately absent from both
engines: the emulated SGI MIPS IV processors do not implement it. It is not
a scalar MIPS IV feature-completion item.

## Interpreter handlers inside a region

Safe CP0 operations and LL/LLD/SC/SCD stay in a compiled region through calls
to the interpreter's existing handlers. They have no native emitters. This
preserves privilege, exception, TLB, and reservation behavior in one place.
`cop0.rs` admits only operations whose side effects are safe for the region;
for example, Status writes are rejected when they would invalidate baked FPU
assumptions, and Cause writes can affect pending software interrupts.

General interpreter fallback is separately controlled by `j2 fallback`, off
by default. A deliberately inserted `JIT_REGION_BOUNDARY_SENTINEL` always
terminates a walk and is never executed as a fallback.

## Boundaries and unavailable coprocessors

CACHE, SYSCALL/BREAK, unsafe CP0 operations, and unsupported CP2 operations
are handled through the interpreter/boundary policy. CP2 is not present on
these machines. These are not missing arithmetic emitters to implement by
duplicating interpreter semantics.

Emitter coverage does not establish architectural correctness. Remaining FP,
self-modifying-code, fusion, and lifecycle work is tracked in
[TODO.md](../../TODO.md). Historical performance measurements are in
[what JIT still interprets](instructions-jitv2-still-interprets.md).

## Lessons from the earlier inventory

The analyzer once accepted opcodes absent from codegen's emitter tables,
causing one unsupported word to decline an entire region. Classification now
uses `opcode_support::has_emitter`; coverage tables must continue to mirror
codegen lookup tables.

MOVCF.S and MOVCF.D require different FPR widths. Reusing the full-slot move
for the single form clobbered upper bits under FR=1; equivalence tests caught
it. Masked SWL/SWR/SDL/SDR writes likewise require the variable-byte-mask hook,
not a fixed-width store. COP1X's `rs` is a base/source register, not COP1's
format selector; conflating them misclassified indexed FP memory operations
whose base register number happened to equal RS_BC1.
