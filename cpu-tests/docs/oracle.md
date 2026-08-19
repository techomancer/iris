# Where the expected values come from

A test whose expected value was recorded from IRIS proves only that IRIS still
does what it used to. This file is the standing rule for how each expectation in
the suite was arrived at, in priority order.

## 1. The manuals

`docs/R4000_um2.pdf` is in-repo and is the primary source. Everything in `alu`,
`muldiv`, `mem`, `branch`, `excep`, `cp0` and `tlb` is derivable from it by
hand, and several tests quote it directly in their comments — for example
Table 4-10 on the Wired register ("Reserved. Must be written as zeroes, and
returns zeroes when read"), and Figure 4-9 on EntryHi's Fill field ("Reserved.
0 on read; ignored on write").

Reading the manual has already been the difference between a real finding and a
wrong test twice:

- **Wired's reserved bits.** The manual is explicit, so the failing test stands
  as a genuine deviation.
- **EntryHi's Fill field.** The same-shaped failure was the *test* being wrong:
  EntryHi is not a flat 64-bit register, and IRIS was right to return zero.

When a test fails, read the manual section before touching either side.

## 2. Nothing, when the manual says nothing

Some behaviour is genuinely unspecified, and asserting it would invent a
requirement the architecture does not make. The suite marks these explicitly and
**reports rather than compares**:

- `muldiv/div_by_zero_no_trap` and the two overflow cases — HI/LO are
  architecturally UNPREDICTABLE after a divide by zero or after
  `0x80000000 / -1`. Only "no trap fires" is asserted; the values are printed so
  a change is visible in the log without being a failure.
- `excep/cop2_unusable` — for a coprocessor that is absent rather than merely
  disabled, both Coprocessor Unusable and Reserved Instruction are defensible.
  The test accepts either and reports which. (IRIS gives RI with CE=1.)
- `mips4/multi_fp_cc` on R4400 — a non-zero `cc` field is not an
  architecturally-defined MIPS III encoding, so R4400 behaviour is not pinned.
- MTC0 into a 64-bit CP0 register — the manual's Operation section is just
  `CPR[0,rd] <- GPR[rt]` and says nothing about the width, so the suite
  deliberately does **not** assert sign-extension.
- `fpu/fr0_odd_undefined` and `fpu/fr_mode_switch` — Appendix B says a format
  operation naming an odd register with FR=0 has an "undefined result", and
  says nothing at all about what the paired view shows after FR=1 code wrote
  the two registers separately. Both are printed.
- `fpu/cvt_out_of_range` and `fpu/cvt_infinity` — see below.

### Which CPU an expectation applies to

`docs/R4000_um2.pdf` is in this repository; NEC's VR5000 manual is not. Where
a rule is one that two implementations of the same architecture may legitimately
differ on, the suite therefore asserts it for the R4400 and **reports** it for
the R5000, rather than assuming the R5000 inherited it.

The whole `fpu/denorm_*` group works this way. Denormal handling is the
canonical example of an implementation-defined escape hatch: the R4400 punts
denormals to software with an Unimplemented Operation, and whether the R5000
does the same is not something this repository can establish. The reports still
carry their weight — they show up in the log next to the R4400 assertions, so
the two cells can be compared by eye.

### Two documents that disagree

`cvt.w.s` of a value with no integer representation is the one case where the
two applicable specifications conflict outright. The R4000 manual's Table 7-2
makes "Overflow on convert" an Unimplemented Operation (E) whether or not the
trap is enabled; the MIPS IV ISA makes it an Invalid Operation whose untrapped
default result is the largest representable integer. Both are defensible for
these parts, so `fpu/cvt_out_of_range` accepts either — and then insists on the
whole of whichever it gets: an E must come with an untouched destination, a V
must come with a saturated result. A silently wrong number matches neither.

## 3. Host-computed values

FP expectations are IEEE-754 bit patterns computed on the host rather than
worked out by hand — `0.25f` is `0x3E800000`, not "whatever came out".

`gen/fpvectors.py` is where that happens for anything larger than a handful of
constants. It computes with **exact rational arithmetic** (`fractions.Fraction`)
and rounds to the destination format by hand, so the expectations follow from
the definition of IEEE 754 rather than from any floating-point unit — not the
host's, and certainly not the emulated one. `make vectors` regenerates
`tests/fpu/fpvectors.{c,h}`; the generated files are checked in so that building
the suite still needs nothing but the cross toolchain.

Two properties are worth keeping if it is ever rewritten:

- **`--check` cross-checks against the host FPU** before writing anything,
  which is §4 of this document applied to the generator itself: a disagreement
  is a reason to go back to the standard, not to copy the host's answer. Every
  vector currently agrees. (Double rounding is not a hazard here — a
  single-precision result computed in double and then rounded to single is
  correctly rounded, since a double has more than the 2p+2 = 50 bits that
  requires.)
- **Operands are rounded through the destination format first.** Computing an
  expectation from a value the register cannot hold produces a table no correct
  implementation can match; see [gotchas.md](gotchas.md).

The tables deliberately contain no NaN results (the payload is
implementation-defined) and no denormal or underflowing results (on an R4400
those are Unimplemented Operation traps rather than arithmetic at all).
`fpu_double.c` and `fpu_denorm.c` cover those by hand, with the manual quoted
next to each expectation.

## 4. Cross-checking against another implementation

Where the spec is ambiguous and the behaviour still matters, MAME's MIPS3 core
and QEMU's MIPS64 target are the tiebreakers. Neither is authoritative; a
disagreement is a reason to go back to the manual, not to copy an answer.

## 5. Real hardware

The gold standard, and the reason the suite is kept free of IRIS-specific
requirements: the test device is *probed* rather than assumed, and everything
essential goes over the SCC, so the same binary can boot on a real Indy and
print the same lines. Any expectation confirmed that way should be annotated as
such in the test.

## 6. Golden recording — last resort

Recording what IRIS does and calling it "expected" is allowed only for
behaviour no spec pins down and where a regression would still be worth
catching. Anything recorded this way must say so in the test comment, so nobody
later mistakes it for an architectural requirement.

---

## Differential testing, which needs no oracle at all

Two axes give real signal without anyone deciding what the right answer is,
because the *same binary* runs on both sides and any disagreement is a bug:

- **R4400 vs R5000.** Every test in `mips4/` runs on both CPUs and branches
  internally: the instruction must compute on R5000 and raise Reserved
  Instruction on R4400. An emulator that implements MIPS IV unconditionally
  passes half of that and fails the other half. `mips4/mips3_control` is the
  negative control — a MIPS III instruction that must fault on neither, so a
  CPU that raised RI for everything cannot pass by accident.
- **Interpreter vs JIT.** A guest-visible ISA suite is the cleanest JIT
  differential available. `rules/jit/verify-mode.md` records that JIT verify
  mode is structurally invalid for blocks containing stores — so this covers
  ground verify mode cannot reach.

`run/matrix.sh` runs all four cells.
