# Status

*Last full run: 2026-08-19, tests-only branch (no emulator changes).*

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

Measured on this branch, which carries **tests only** — no emulator changes, so
the suite reports every finding rather than hiding any.

| cell | pass | fail | failing tests |
|---|---:|---:|---|
| R4400, interpreter | 783 | 19 | 7 |
| R5000, interpreter | 802 | 3 | 3 |
| R4400, PROM boot from a `mkvh` image | 784 | 20 | 7 |

Every failure is a recorded finding — see [findings.md](findings.md):

| failing test | finding | CPUs |
|---|---|---|
| `mips4/movn_movz`, `pref`, `recip_rsqrt`, `movci`, `cop1x` | MIPS IV executes instead of raising RI | R4400 only |
| `mips4/multi_fp_cc` | `c.cond.fmt` writes the wrong condition code | R5000 only |
| `cp0/wired_reserved_bits` | reserved bits do not read as zero | both |
| `fpu/fcsr_reserved` | reserved bits do not read as zero | both |

**17 of the R4400's 19 failing checks are the single MIPS IV finding.** The
whole `mips4/` group fails on R4400 and passes on R5000 — apart from
`multi_fp_cc`, which is the mirror image: it passes on R4400 (where the test
only reports, since a non-zero `cc` is not a defined MIPS III encoding) and
fails on R5000, where the instruction actually executes. That inversion is
precisely what the two-CPU axis was built to expose.

A fix for `multi_fp_cc` exists on branch **`fix/fp-condition-code`** (three
commits, `src/` only). With it applied, R5000 goes to **801/2** and
`cargo test --lib mips_exec_test` stays green at 95/95. It is deliberately not
on this branch.

> The total check count drifts by one or two between runs. `cp0/compare_sets_ip7`
> executes two checks when the timer fires and one when it reports a skip, and
> whether it fires depends on the wallclock-anchored counter — see finding 5.

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
