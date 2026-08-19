# What IRIS's FPU does that an R4400's does not

The arithmetic is right. The *exception model* around it is a simplification,
and these are the five places it shows, all found by `cpu-tests/tests/fpu/` and
written up at length in `cpu-tests/docs/findings.md`. Worth knowing before
debugging anything FP-shaped, because each one makes IRIS look *more*
IEEE-conformant than the hardware, not less.

1. **There is no Unimplemented Operation.** A real R4400 refuses denormals in
   both directions — denormal operand, denormal result, quiet-NaN operand —
   and traps with `FCSR.Cause.E` so software finishes the job. IRIS computes
   them on the host FPU and delivers the correct IEEE answer with no trap. So
   a kernel's FP-assist handler is dead code under IRIS, and the R4400's
   `Cause.E` path is never exercised.

2. **`FCSR.FS` is inert.** The bit stores and reads back, but nothing consults
   it, so flush-to-zero never happens: `2^-126 * 0.5` yields the denormal with
   FS set or clear, where hardware gives `+0` with `Cause.U` and `Cause.I`.

3. **A trapped exception still writes its result.** The manual says "when a
   floating-point exception is taken, no results are stored"; IRIS computes,
   writes the destination register, and *then* decides whether to raise
   `EXC_FPE`. It also sets the Flag bit, which hardware leaves to software.
   `c.cond.fmt` is the one instruction that gets the result half right.

4. **`FCSR.Cause` accumulates.** It should hold "what the last FP operation
   did" and be rewritten — including to zero — by every FP operation. IRIS ORs
   into it, so it behaves as a second copy of the sticky Flag field. Anything
   that reads Cause to find out what the *last* instruction raised gets the
   wrong answer.

5. **Signalling NaNs are not distinguished from quiet ones.** `c.cond.fmt`
   raises Invalid on any NaN when the predicate's high cond bit is set — which
   is correct — but never raises it for a signalling NaN under the other eight
   predicates, which it should. `ABS.fmt` and `NEG.fmt` never touch FCSR at
   all, so they never raise Invalid either, while `ADD.S` on the same operand
   does.

None of this affects IRIX, which is why it went unnoticed: nothing in the
system relies on the FP-assist path, and signalling NaNs only exist where a
program deliberately makes one. It matters when you are asking whether a
number came out of the FPU the way it would have on hardware, or when writing
tests that expect the hardware's exception behaviour — start from
`cpu-tests/docs/findings.md` §6-§10 rather than from the R4000 manual alone.
