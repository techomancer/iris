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

## 3. Host-computed values

FP expectations are IEEE-754 bit patterns computed on the host rather than
worked out by hand — `0.25f` is `0x3E800000`, not "whatever came out". This is
what `gen/` is for as the `fpu` group grows.

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
