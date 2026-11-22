# FPU conversions hand the hard cases to software, on every modelled part

Status: rule, 2026-11-19 (issue #71).

## Rule

Every conversion (ROUND/TRUNC/CEIL/FLOOR/CVT.W/CVT.L from S or D, CVT.S.D,
CVT.D.S, CVT.S/D.W/L) runs in one shared helper in `mips_exec.rs`, used by the
interpreter and called from jitv2's cold path: `cvt_to_int_and_commit`,
`cvt_int_to_float_and_commit`, `cvt_d_to_s_and_commit`,
`cvt_s_to_d_and_commit`. They take Unimplemented Operation (Cause.E alone, fd
and Flags untouched, `EXC_FPE`) for:

| case | parts | source |
|---|---|---|
| denormal source, any conversion | all | R4000 UM "Unimplemented Instruction Exception"; VR5000 UM 9.4.6; R10000 UM p. 309 |
| MIPS quiet-NaN source (fraction MSB clear) | all | same lists ("Quiet Not a Number operand") |
| float to integer result the format cannot hold (NaN, inf, out of range) | all (`CVT_UNREPRESENTABLE_UNIMPLEMENTED = true`) | Indy R4400 oracle (`cvt.w.s` of 2^31 and +inf trap E); VR5000 UM 9.4.6; NEC VR10000 UM B.4 |
| float to long result outside [-2^53, 2^53) | R4400, R5000 (`CVT_L_RANGE_BITS = 53`) | VR5000 UM 9.4.6; R4400 inferred |
| float to long result outside [-2^51, 2^51) | R10000 (51) | R10000 UM p. 312 |
| long source to CVT.S.L/D.L outside [-2^52, 2^52) | R4400, R5000 (`CVT_FROM_L_RANGE_BITS = 52`) | VR5000 UM 9.4.6; Indy R4400 and R5000 measured (2^40+1 converts, 2^53+1 traps) |
| CVT.S.D result that underflows, FS clear or U/I enabled | all | as for arithmetic (`fpu_update_fcsr_full`) |

The kernel's FP emulator then computes the IEEE answer, so IRIX programs see
what they see on hardware. Nothing about this costs ordinary conversions:
the inline jitv2 paths send only the special operands to the callout.

Other rules the helpers implement:

- The untrapped Invalid default for a float-to-integer conversion is
  2^31-1 / 2^63-1 **whatever the sign** (R4000 UM appendix B, every
  conversion page; MIPS64 Vol. II). IRIS used to deliver -2^31 for a
  negative overflow. No modelled part reaches that path any more; the helper
  keeps it for a model with `CVT_UNREPRESENTABLE_UNIMPLEMENTED = false` and
  `cvt_default_result_tests` pins it.
- ROUND is nearest-even whatever FCSR.RM holds; CVT.W/CVT.L follow RM. Both
  round by bits (`round_f32/64_to_int_mode`), not on the host.
- Inexact is "result differs from source", by value. For a long source that
  rounds to 2^63 the comparison must be exact (i128 in the interpreter, an
  extra `>= 2^63` test inline): a saturating cast back to i64 calls
  `i64::MAX -> 2^63` exact.
- CVT.S.D overflow raises O **and** I, and is detected in every rounding
  mode: an infinity from a finite source, or ±MAX from a source of magnitude
  2^128 or more. Between MAX and 2^128 a toward-zero result is MAX with I
  alone.
- A trapped Inexact on CVT.S/D.W/L leaves fd unwritten (it used to write it).
- CVT.D.S of a signalling NaN is Invalid with the default NaN
  0x7FF7FFFF_FFFFFFFF. The host's widening sets the fraction MSB, which under
  the MIPS encoding hands on another signalling NaN.

## jitv2

`JitConsts` carries `cvt_l_range_bits` and `cvt_from_l_range_bits` (63 = no
extra limit). Its `Default` is the R4400's (53 and 52), the same as the
`CpuModel` defaults, because every test harness that builds a bare
`Codegen::new()` compares against an interpreter on the default model: a
default of "no limit" makes the JIT convert what the interpreter traps.

Inline gates added: the to-int fast path also requires a non-denormal source
(`(abs_bits - 1) >= min_normal - 1`); CVT.S/D.L requires the source in range;
CVT.D.S widens inline only a normal or zero operand and calls
`fpu_cvt_s_to_d_fn` for the rest. The cost of the extra compares is not
measured (host was busy).

## Not established

- The R4400's float-to-long limit (inferred from the VR5000).
- What an R5000 does with CVT.D.S of a MIPS signalling NaN (`fpu/cvt_nan_source`
  reports it; `fpu/qnan_operand` leaves the arithmetic case unasserted too).
- The R12000 flushes a denormal conversion source to zero when FCSR.FS is set
  (NEC UM B.4); IRIS traps it like the R10000.
- The new cpu-tests (`fpu/cvt_denormal_source`, `cvt_nan_source`,
  `cvt_s_d_underflow`, `cvt_s_d_overflow_rz`, `cvt_l_limit`) have not run on
  an Indy; their expectations are the manuals'.
