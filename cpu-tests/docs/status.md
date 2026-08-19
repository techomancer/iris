# Status

*Last full run: 2026-08-19, commit `98db5bb`.*

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
| `fpu` | 26 | both formats, signed zeros, infinities, NaNs, all rounding modes, comparisons, FCSR, FR aliasing |
| `cache` | 8 | geometry, tag round-trip, cached/uncached views, I-cache coherency |
| `mips4` | 7 | every MIPS IV addition — computes on R5000, must raise RI on R4400 |
| **total** | **172** | **~804 checks** |

## Results

| cell | pass | fail |
|---|---:|---:|
| R4400, interpreter, `--load-elf` | 783 | 19 |
| R5000, interpreter, `--load-elf` | 801 | 2 |
| R4400, interpreter, PROM boot from a `mkvh` image | 784 | 20 |

Every failure is a recorded finding, not an unknown — see
[findings.md](findings.md):

- **17 of the 19 R4400 failures are a single finding**: MIPS IV instructions
  execute instead of raising Reserved Instruction. The whole `mips4/` group
  fails on R4400 and passes on R5000, which is exactly the asymmetry the
  two-CPU axis was built to measure.
- **The remaining 2, on both CPUs**, are reserved-bit masking in `Wired` and
  `FCSR` — documented requirements with no practical consequence.

The R5000 column is the useful one for judging the emulator: 801 of 803 checks
pass, and both failures are the same cosmetic masking issue.

One bug the suite found is **fixed**: `c.cond.fmt` wrote the wrong FP condition
code (`bdb00c8`), along with the matching off-by-two in IRIS's own test encoder
that had been cancelling it (`4f1d4a1`). `mips4/multi_fp_cc` now passes on
R5000, and `cargo test --lib mips_exec_test` is 95/95.

## Not yet done

- **The EFS CD.** The volume-header path is proven — `boot -f dksc(0,2,8)cputest`
  runs the whole suite — but an EFS writer does not exist yet. See PLAN.md §12.
- **The JIT cells.** `run/matrix.sh` and the CI workflow define them; they have
  not been run here.
- **`gen/fpvectors.py`.** FP expectations are currently hand-written IEEE-754
  bit patterns. A generator would let the `fpu` group grow by an order of
  magnitude without hand arithmetic.
- **Interrupt delivery.** Everything runs with `Status.IE` clear, so the suite
  tests that interrupts become *pending* but never that one is *taken*. That
  needs a handler that can distinguish an interrupt from a fault, and is the
  most valuable single addition left.
- **Supervisor and user mode.** Everything runs in kernel mode. The privileged-
  instruction and address-space tests that need a mode switch are absent.
