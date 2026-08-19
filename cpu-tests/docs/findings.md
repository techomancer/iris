# Findings

Deviations from the R4000/MIPS IV architecture that the suite has surfaced in
IRIS. Each entry says what the manual requires, what IRIS did, and how much it
matters in practice.

Test-side mistakes — cases where the suite was wrong and IRIS was right — are
in [gotchas.md](gotchas.md) instead. Telling the two apart is most of the work.

---

## 1. `c.cond.fmt` wrote the wrong FP condition code — **fixed**

*Found by `mips4/multi_fp_cc`. Fixed in `src/mips_exec.rs`.*

MIPS IV widened the FP condition from a single bit to eight codes, and
`C.cond.fmt` gained a `cc` field at **bits 10:8** of the instruction word:

```
31-26  25-21  20-16  15-11  10-8   7   6    5-4   3-0
COP1   fmt    ft     fs     cc     0   A    FC    cond
```

`exec_fcc_s` and `exec_fcc_d` extracted it as `d.sa & 0x7` — the low three bits
of the 5-bit `sa` field, which spans bits **10:6**. That is the field shifted
two places too far right:

| written | `sa` | correct cc | IRIS cc |
|---|---|---|---|
| `c.eq.s $fcc3` | `0b01100` | 3 | 4 |
| `c.eq.s $fcc5` | `0b10100` | 5 | 4 |

Both compares landed on cc4, so the second overwrote the first and `FCSR` read
back as exactly `0x00000000` — the symptom that made it findable.

Every *reader* was already correct: `BC1T/BC1F` (`exec_bc1`), `MOVCF`, and
`MOVF/MOVT` all use `(d.raw >> 18) & 0x7` for their own field. Only the two
write sites were wrong, so the fix is to extract from the raw word the same
way: `(d.raw >> 8) & 0x7`.

**Why it never showed up before.** On MIPS III the `cc` field does not exist and
bits 10:6 are always zero, where `sa & 0x7` and `sa >> 2` agree. So the bug is
invisible to R4400 code and to every MIPS III binary — including all of IRIX —
and bites only MIPS IV code that actually uses a condition code other than 0.
**Why the existing tests did not catch it.** Two reasons, and the second is the
interesting one:

- `test_fpu_cc1_7_visible_via_fcsr_readback` calls `set_fpu_cc()` directly
  rather than executing a `c.cond.fmt`, so it never exercised the field being
  decoded.
- `test_fpu_multi_cc_compare_and_branch` *does* execute the instruction — but
  its `make_compare_s` helper encoded `cc` as `(cc << 6)`, putting it at the
  bottom of the 5-bit `sa` field instead of at bits 10:8. That is the **same
  off-by-two**, and it cancelled the executor's. The test passed because both
  halves were wrong in the same direction, and it started failing the moment
  the executor was fixed. Its own comment said "cc in fd field [10:8]" all
  along.

The helper is fixed too. This is the failure mode a hand-written encoder cannot
catch on its own: only running the real assembler's output — where `c.eq.s
$fcc3` is encoded by GAS, not by the test — puts an independent second opinion
on the encoding.

This is also the case the whole R4400-vs-R5000 axis exists for.

---

## 2. MIPS IV instructions execute on the R4400 — **open**

*Found by the whole `mips4/` group. 19 of the 21 R4400 failures.*

Every MIPS IV instruction the suite tries runs to completion on an R4400 build
instead of raising Reserved Instruction:

| instruction | R4400 should | R4400 does |
|---|---|---|
| `MOVN` / `MOVZ` | RI | executes |
| `PREF` | RI | executes |
| `RECIP.fmt` / `RSQRT.fmt` | RI | executes |
| `MOVF` / `MOVT` (MOVCI) | RI | executes |
| COP1X (`LWXC1`, `MADD.fmt`, …) | RI | executes |

IRIS decodes the MIPS IV additions unconditionally: nothing consults `PRId`, and
the `r5k` cargo feature changes cache geometry and CPU identity but not which
opcodes decode. So an emulated R4400 is a strictly more permissive machine than
the real one.

**How much it matters.** Nothing that runs today is affected — code built for
MIPS III never emits these encodings, so IRIX and its applications behave
identically either way. The cost is fidelity in one specific direction: a
binary built for an R5000 will appear to work on an emulated R4400 and fault on
real hardware, so IRIS cannot be used to check that something is actually
MIPS III-clean. That is a real use for an emulator, and it is the thing this
axis exists to measure.

Fixing it means gating the MIPS IV decode paths on the CPU model rather than
compiling them in unconditionally. That is a larger change than the condition-
code fix above and touches the decode hot path, so it is left as a finding
rather than done here.

`mips4/mips3_control` is the negative control for all of this: a MIPS III
instruction (`daddu`) that must fault on neither CPU. It passes, so the RI
expectations above are not being met vacuously by a CPU that traps everything.

---

## 3. `Wired`'s reserved bits do not read back as zero — **open**

*Found by `cp0/wired_reserved_bits`.*

R4000 manual, Table 4-10: bits 31:6 of `Wired` are "Reserved. Must be written as
zeroes, and returns zeroes when read." IRIS stores and returns the full 32-bit
value, so writing `0xFFFFFFFF` reads back `0xFFFFFFFF` instead of `0x0000003F`.

**How much it matters: very little.** The manual also says software must write
zeroes there, and IRIX does — it only ever writes small entry counts. Nothing in
a real workload can observe the difference. It is recorded because it is a
documented architectural requirement and the suite is the place that notices
such things, not because anything depends on it.

---

## 4. FCSR's reserved bits do not read back as zero — **open**

*Found by `fpu/fcsr_reserved`.*

The same class as `Wired`. R4000 manual Figure 6-4 gives FCSR as:

```
31        25 24 23 22    18 17    12 11   7 6   2 1  0
     0       FS  C     0      Cause    Enables Flags  RM
     7        1  1     5        6         5      5    2
```

Bits 22:18 are a reserved zero field. IRIS stores and returns them, so writing
`0xFFFFFFFF` reads back with `0x007C0000` set.

(The test asserts only 22:18, not 31:25 — those are zero on the R4400 but are
FCC7..FCC1 on MIPS IV, so a suite that runs on both cannot require them to be
zero.)

Same practical impact as `Wired`: none for real software, recorded because it
is a documented requirement.

---

## 5. `Count` can skip past `Compare` without IP7 firing — **observation**

*Seen by `cp0/compare_sets_ip7`, which reports rather than asserts.*

IP7 fires when `Count` becomes numerically **equal** to `Compare` — not when it
exceeds it. IRIS models that faithfully and says so
(`schedule_compare_timer` in `src/mips_core.rs`): a `Compare` written "in the
past" correctly does not fire until `Count` wraps through 2^32.

But `Count` is wallclock-anchored by default, so it advances in jumps rather
than one tick at a time, and a jump can step over the match:

```
[timer did not fire: Count 0x023b76fa -> 0x023bc51d, deadline 0x023bc51a,
 2253 iterations]
```

`Count` passed the deadline by 3 and no interrupt became pending. On real
hardware `Count` increments by one per tick, so the equality is never missed.

**Why this is filed as an observation rather than a bug.** It is a consequence
of the timing model, not of the Count/Compare logic, and the `ci_clock` cargo
feature exists precisely to make the clock deterministic for cases that need
it. It is recorded because the consequence is real: a kernel that arms its
timer with `Compare = Count + delta` and relies on the interrupt can, in
principle, have one silently swallowed. Whether that ever happens to IRIX under
a normal workload is not something this suite can answer.

The test now sets `Compare` relative to a freshly-read `Count` — the way a
kernel does — and reports the skip instead of failing on it.

---

## Handled correctly

Worth stating, since these are the same shape as the two findings above and
IRIS gets them right: `PageMask` drops its reserved bits, `EntryHi`'s `Fill`
field reads back as zero, `Cause` is read-only except for the two software
interrupt bits, and `PRId` and `Random` reject writes. All verified by passing
tests.

---

## Behaviour deliberately not asserted

Not everything that varies is a bug. These are reported in the log rather than
compared, because no specification pins them down and asserting one answer
would invent a requirement:

| what | why |
|---|---|
| HI/LO after divide-by-zero, and after `0x80000000 / -1` | architecturally UNPREDICTABLE; only "no trap fires" is required. IRIS leaves HI/LO unchanged. |
| COP2 access | Coprocessor Unusable and Reserved Instruction are both defensible for a coprocessor that is absent rather than disabled. IRIS gives RI with CE=1. |
| `c.eq.s` with `cc != 0` on R4400 | not an architecturally-defined MIPS III encoding, so R4400 behaviour is not specified. |
| MTC0 into a 64-bit CP0 register | the manual's Operation section is `CPR[0,rd] <- GPR[rt]` and says nothing about width, so sign-extension is not asserted. |
| Cache tag bit layout | implementation-specific (R4000 manual Figure 11-4); the suite tests the Index_Store_Tag / Index_Load_Tag round-trip instead of a particular encoding. |

See [oracle.md](oracle.md) for the standing rule on where expected values come
from.
