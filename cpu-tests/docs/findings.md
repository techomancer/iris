# Findings

Deviations from the R4000/MIPS IV architecture that the suite has surfaced in
IRIS. Each entry says what the manual requires, what IRIS did, and how much it
matters in practice.

Test-side mistakes — cases where the suite was wrong and IRIS was right — are
in [gotchas.md](gotchas.md) instead. Telling the two apart is most of the work.

---

## 1. `c.cond.fmt` wrote the wrong FP condition code — **open here; fix on a branch**

*Found by `mips4/multi_fp_cc`.*

> This branch carries tests only. The fix lives on **`fix/fp-condition-code`**
> (two commits, `src/mips_exec.rs` and `src/mips_exec_test.rs`), so
> `mips4/multi_fp_cc` fails here — which is what the suite should be reporting
> until the fix lands.

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

The helper is fixed on the same branch. This is the failure mode a hand-written
encoder cannot catch on its own: only running the real assembler's output —
where `c.eq.s $fcc3` is encoded by GAS, not by the test — puts an independent
second opinion on the encoding.

This is also the case the whole R4400-vs-R5000 axis exists for.

---

## 2. MIPS IV instructions execute on the R4400 — **open**

*Found by the whole `mips4/` group, and by `mips4_fp.c` in more detail.*

Every MIPS IV instruction the suite tries runs to completion on an R4400 build
instead of raising Reserved Instruction:

| instruction | R4400 should | R4400 does |
|---|---|---|
| `MOVN` / `MOVZ` | RI | executes |
| `PREF` | RI | executes |
| `RECIP.fmt` / `RSQRT.fmt` | RI | executes |
| `MOVF` / `MOVT` (MOVCI) | RI | executes |
| COP1X (`LWXC1`, `LDXC1`, `SWXC1`, `SDXC1`, `PREFX`) | RI | executes |
| `MADD`/`MSUB`/`NMADD`/`NMSUB`, both formats | RI | executes |
| `MOVF.fmt`/`MOVT.fmt`/`MOVN.fmt`/`MOVZ.fmt` | RI | executes |
| `RECIP.D` / `RSQRT.D` | RI | executes |

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

## 6. A trapped FP exception still writes its result and its Flag bit — **open**

*Found by the whole `fpu/trap_*` group. Two rules, one root cause.*

The R4000 manual states both halves of the rule outright:

> When a floating-point exception is taken, no results are stored, and the only
> state affected is the Cause bit.

and, for the Flag field specifically:

> When a floating-point exception is taken, the flag bits are not set by the
> hardware; floating-point exception software is responsible for setting these
> bits before invoking a user handler.

IRIS does the arithmetic, writes the destination register, then decides whether
to trap:

```rust
let result = f32::from_bits(...) + f32::from_bits(...);
(self.fpr_write_w)(&mut self.core, fd_reg, result.to_bits());
self.fpu_update_fcsr()          // <- the trap decision happens here
```

and `fpu_update_fcsr` ORs the host's exception flags into the Flag field before
testing them against the Enable field, so both happen unconditionally. Every
trapped case shows it: with Invalid enabled, `div.s $f4, 0.0, 0.0` traps *and*
leaves a quiet NaN in `$f4` *and* sets Flag.V.

**How much it matters.** The destination half is the one with teeth. IEEE 754
trap handlers exist to substitute a result, and a handler that declines to
substitute one — because it only counts the event, or because it decides the
default was fine — leaves the register holding a value the architecture says
was never written. Under IRIS that value is the IEEE default result, which is
usually what the handler would have supplied anyway; on hardware it is the
destination's old contents. The Flag half is more benign: software that sets
the flag itself, as the manual tells it to, simply sets a bit that is already
set.

`c.cond.fmt` is half an exception: `exec_fcc_s` returns before writing the
condition bit when it traps, so the *result* rule holds there — but it sets
Cause.V and Flag.V together on the way out, so the *flag* rule does not.
`fpu/cmp_trap_on_signal` checks both and reports the second.

---

## 7. FCSR Cause bits accumulate instead of being rewritten — **open**

*Found by `fpu/cause_per_instruction`.*

> The Cause bits are written by each floating-point operation... they identify
> the exceptions raised by the last floating-point operation.

and, in the state-saving section:

> The Cause field of the Control/Status register holds the results of only one
> instruction.

That is the entire difference between the Cause field and the Flag field: Cause
is what the *last* operation did, Flags are what *every* operation since the
last clear did. IRIS ORs into both:

```rust
self.core.fpu_fcsr |= causes;
self.core.fpu_fcsr |= flags & FCSR_FM;
```

so Cause is a second, redundant copy of Flags. A divide by zero followed by an
exact addition leaves Cause.Z set, where hardware clears it.

**How much it matters.** This is the field an FP exception handler reads to
find out what it is being asked to fix, and the field a program reads to ask
"did *this* operation raise anything". Under IRIS the answer is always "this
operation, or any operation since the last CTC1". Nothing in IRIX depends on it
today — the kernel's FP assist path is only entered by exceptions IRIS does not
generate (finding 8) — but it is the sort of thing a numerical program that
polls Cause after each step would get wrong, silently.

---

## 8. Denormals never trap, and FCSR.FS is inert — **open**

*Found by the `fpu/denorm_*` group.*

An R4400 does not compute with denormalized numbers. It refuses them, in both
directions, and lets software finish the job — R4000 manual, chapter 7:

| condition | R4400 |
|---|---|
| denormalized operand (except to a Compare) | Unimplemented Operation (Cause.E) |
| quiet NaN operand (except to a Compare) | Unimplemented Operation (Cause.E) |
| denormalized result, FS clear | Unimplemented Operation (Cause.E) |
| denormalized result, FS set, U and I disabled | flushed to a signed zero, Cause.U and Cause.I set |
| denormalized result, FS set, U or I enabled | Unimplemented Operation (Cause.E) |

Cause.E has no Enable bit and no Flag bit: "whenever this exception occurs, an
unimplemented exception trap is taken".

IRIS raises E in exactly one situation — an underflow with the Underflow trap
enabled — and computes everything else on the host FPU, which handles
denormals in hardware. `FCSR.FS` is storable and readable (`fpu/fs_bit_round_trip`
passes) but no execution path consults bit 24 at all, so setting it changes
nothing: `2^-126 * 0.5` yields the denormal `0x00400000` with FS set or clear,
where hardware gives `+0` with Cause.U and Cause.I in the first case and a trap
in the second.

**How much it matters.** For the *numbers*, IRIS is arguably nicer than the
hardware: it delivers the correctly-rounded denormal result that the R4400's
software handler would have had to compute, and does it without the trap. What
is lost is the ability to see the FP-assist path at all — a kernel's
Unimplemented Operation handler is dead code under IRIS, and an IRIX
installation whose libc sets FS for flush-to-zero gets denormals anyway.

The R5000 side of each of these tests is reported rather than asserted: the
R4000 manual is in-repo and pins the R4400, the VR5000's is not, and denormal
handling is exactly the kind of thing two implementations of one architecture
are permitted to differ on. See [oracle.md](oracle.md).

---

## 9. A signalling NaN raises Invalid only where a quiet one would — **open**

*Found by `fpu/cmp_snan_any_pred`.*

Two rules govern Invalid on a compare, and IRIS implements one of them:

1. If either operand is **any** NaN and the predicate's high cond bit is set
   (the eight signalling predicates, `SF NGLE SEQ NGL LT NGE LE NGT`), Invalid
   is raised. IRIS does this correctly — `fpu/cmp_signalling_qnan` passes for
   all sixteen predicates.
2. If either operand is a **signalling** NaN, Invalid is raised whatever the
   predicate is: Table 7-2 lists "Signaling NaN source" as V with the trap
   enabled and V with it disabled, and the Invalid Operation list names
   "Comparison or a Convert From Floating-point Operation on a signaling NaN".
   IRIS does not do this.

```rust
if (funct_val & 0x8) != 0 && (fs_val.is_nan() || ft_val.is_nan()) {
```

`is_nan()` does not distinguish the two kinds, and the predicate test gates the
whole check — so `c.eq.s` against a signalling NaN is silent, in both formats.

**How much it matters.** Little in practice: signalling NaNs only exist where a
program deliberately creates them, which is a debugging technique rather than
something IRIX or its applications do. It is a precise, cheap-to-fix deviation
in code that is otherwise exactly right, which is the main reason to record it.

---

## 10. ABS.fmt and NEG.fmt never raise Invalid — **open**

*Found by `fpu/snan_operands`.*

The manual is unusually explicit that these two are not the bit-twiddling
operations they look like:

> A move (MOV) operation is not considered to be an arithmetic operation, but
> absolute value (ABS) and negate (NEG) are considered to be arithmetic
> operations and cause this exception if one or both operands is a signaling
> NaN.

`exec_fabs_s` and `exec_fneg_s` compute the result and end with
`handle_exec_complete()` — they never reach `fpu_update_fcsr`, so no FCSR field
is touched by either instruction under any circumstances. `ADD.S` with the same
signalling NaN operand *does* set Invalid, so the two instructions disagree
with each other about the same operand.

**How much it matters.** The same as finding 9, and for the same reason. Worth
recording because the fix is one line each and because the manual singles these
two instructions out precisely because they are easy to get wrong.

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
