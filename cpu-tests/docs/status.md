# Status

*Last full run: 2026-08-19, tests-only branch (no emulator changes).*
*A full run is a few minutes per cell on a quiet machine.*

## Coverage

| group | tests | what it covers |
|---|---:|---|
| `identity` | 5 | PRId, FIR, cache geometry, Config.K0, TLB size |
| `alu` | 29 | sign extension across the 32/64-bit boundary, overflow traps, shifts, logic, SLT |
| `muldiv` | 18 | mult/div in both widths, HI/LO, the unspecified cases |
| `mem` | 18 | load/store widths, the whole unaligned family at every offset, alignment faults, KSEG0/KSEG1 |
| `branch` | 15 | every conditional, likely-nullification, link registers, delay slots, faults in delay slots |
| `excep` | 15 | traps, reserved instructions, coprocessor usability, EXL/ERET, vector selection |
| `cp0` | 21 | read-only registers, reserved-bit masks, 64-bit access, Count/Compare, LL/SC |
| `tlb` | 10 | entry round-trip over all 48, TLBP, every page size, real translation, V/D bits, ASIDs, refill |
| `fpu` | 88 | see below |
| `cache` | 8 | geometry, tag round-trip, cached/uncached views, I-cache coherency |
| `mips4` | 13 | every MIPS IV addition — computes on R5000, must raise RI on R4400 |
| **total** | **240** | |

### Inside `fpu`

Eight files, because the group grew from 26 tests to 88 and one file could no
longer say what it was about:

| file | tests | what it covers |
|---|---:|---|
| `fpu.c` | 26 | both formats, signed zeros, infinities, NaNs, rounding modes, comparisons, FCSR, FR aliasing |
| `fpu_trap.c` | 14 | trapped exceptions: the Cause/Enable/Flag interaction, what a trap does and does not write, EPC and Cause.BD |
| `fpu_denorm.c` | 9 | denormal operands and results, underflow, and FCSR.FS — the Unimplemented Operation path |
| `fpu_compare.c` | 7 | all sixteen `C.cond` predicates in both formats, signalling vs quiet NaNs, the condition bit's ownership |
| `fpu_vectors.c` | 10 | arithmetic, all four rounding modes, and every integer conversion, against generated tables |
| `fpu_double.c` | 8 | the double-precision paths with no vector: NaN results, bit operations, format conversions |
| `fpu_fr0.c` | 8 | the FR=0 paired register file — the mode every o32 IRIX binary runs in |
| `fpu_breadth.c` | 6 | all 32 registers, signalling-NaN operands, out-of-range conversions, unaligned FP access |

Expectations for `fpu_vectors.c` come from `gen/fpvectors.py`, which computes
them with exact rational arithmetic and cross-checks against the host FPU
before writing (`make vectors`). That is what makes it practical to assert the
*flags* as well as the results, exactly, for every vector.

## Results

Measured on this branch, which carries **tests only** — no emulator changes, so
the suite reports every finding rather than hiding any.

| cell | pass | fail | failing tests |
|---|---:|---:|---|
| R4400, interpreter | 2041 | 121 | 29 |
| R5000, interpreter | 2095 | 37 | 13 |
| R4400, PROM boot from a `mkvh` image | 2041 | 121 | 29 |

The PROM row is the same binary reaching the same answer down the path a
bootable disc will use: the PROM reads the volume header, loads the ELF and
jumps to it. The JIT cells `run/matrix.sh` defines are still unrun.


Every failure is a recorded finding — see [findings.md](findings.md):

| failing tests | finding | CPUs |
|---|---|---:|
| the eleven `mips4/` tests | 2 — MIPS IV executes instead of raising RI | R4400 |
| `mips4/multi_fp_cc` | 1 — `c.cond.fmt` writes the wrong condition code | R5000 |
| `cp0/wired_reserved_bits` | 3 — reserved bits do not read back as zero | both |
| `fpu/fcsr_reserved` | 4 — reserved bits do not read back as zero | both |
| the six `fpu/trap_*` tests, and `fpu/cmp_trap_on_signal` | 6 — a trapped exception still writes its result and its Flag bit | both |
| `fpu/cause_per_instruction` | 7 — Cause accumulates instead of being rewritten | both |
| `fpu/denorm_*`, `fpu/qnan_operand`, `fpu/underflow_enable_e` | 8 — no Unimplemented Operation, and FS is inert | R4400 (asserted; reported on R5000) |
| `fpu/cmp_snan_any_pred` | 9 — a signalling NaN raises Invalid only where a quiet one would | both |
| `fpu/snan_operands` | 10 — ABS and NEG never raise Invalid | both |

The R4400 column is dominated by two findings that are not about the FPU's
arithmetic at all: the MIPS IV decode (11 tests) and the missing Unimplemented
Operation path (6 tests). Everything the FPU is asked to *compute* — every
vector in `fpu_vectors.c`, all 128 predicate results in `fpu_compare.c`, both
formats, all four rounding modes, every integer conversion — passes on both
CPUs.

> The total check count drifts by one or two between runs. `cp0/compare_sets_ip7`
> executes two checks when the timer fires and one when it reports a skip, and
> whether it fires depends on the wallclock-anchored counter — see finding 5.

## Not yet done

- **The EFS CD.** The volume-header path is proven — `boot -f dksc(0,2,8)cputest`
  runs the whole suite — but an EFS writer does not exist yet. See PLAN.md §12.
- **The JIT cells.** `run/matrix.sh` and the CI workflow define them; they have
  not been run here.
- **Interrupt delivery.** Everything runs with `Status.IE` clear, so the suite
  tests that interrupts become *pending* but never that one is *taken*. That
  needs a handler that can distinguish an interrupt from a fault, and is the
  most valuable single addition left.
- **Supervisor and user mode.** Everything runs in kernel mode. The privileged-
  instruction and address-space tests that need a mode switch are absent.
