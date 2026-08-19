/* fpu_denorm — denormalized numbers, underflow, and FCSR.FS.
 *
 * The R4400's FPU does not finish these cases in hardware. It punts them to
 * software with an Unimplemented Operation (Cause bit E), which has no Enable
 * bit and no Flag bit and therefore always traps:
 *
 *   "Any attempt to execute an instruction with an operation code or format
 *    code that has been reserved for future definition sets the Unimplemented
 *    bit ... The Unimplemented Instruction exception can also be signaled when
 *    unusual operands or result conditions are detected that the implemented
 *    hardware cannot handle properly. These include:
 *      - Denormalized operand, except for Compare instruction
 *      - Quiet Not a Number operand, except for Compare instruction
 *      - Denormalized result or Underflow, when either Underflow or Inexact
 *        Enable bits are set or the FS bit is not set"
 *                                       — R4000 manual, chapter 7
 *
 * FCSR.FS is the escape hatch: "When the FS bit is set, denormalized results
 * are flushed to 0 instead of causing an unimplemented operation exception."
 * Note what that sentence covers — *results*. The operand rules above have no
 * FS qualifier, and later MIPS revisions widened FS to flush denormal inputs
 * as well, so this file asserts the FS=0 operand case and only reports the
 * FS=1 one rather than inventing a rule the R4000 manual does not state.
 *
 * Table 7-2 is the summary the tests are written from:
 *
 *   Exponent underflow       U  |  E (trap enabled)  |  U,I (trap disabled)‡
 *   Denormalized or QNaN   none |  E                 |  E
 *
 *   ‡ "Exponent underflow sets the U and I Cause bits if both the U and I
 *      Enable bits are not set and the FS bit is set; otherwise exponent
 *      underflow sets the E Cause bit."
 *
 * Which CPU each expectation applies to: the R4000 manual is in-repo
 * (docs/R4000_um2.pdf) and pins the R4400 side of every rule here. The R5000's
 * manual is not, and denormal handling is exactly the kind of thing two
 * implementations of the same architecture are allowed to differ on — so the
 * R5000 side is reported rather than asserted. See docs/oracle.md §2.
 */

#include "fpu_common.h"

FPOP_S(op_add_s,  "add.s $f4, $f0, $f2")
FPOP_S(op_mul_s,  "mul.s $f4, $f0, $f2")
FPOP_S(op_mov_s,  "mov.s $f4, $f0")
FPOP_D(op_add_d,  "add.d $f4, $f0, $f2")
FPOP_D(op_mul_d,  "mul.d $f4, $f0, $f2")

/* The observation, printed instead of compared. */
static void report_obs(const char *what, const struct fp_obs *o, int dbl)
{
    con_printf("\n      [%s: exc=%u code=%u cause=%x flags=%x result=",
               what, o->exceptions, o->excode, o->cause, o->flags);
    if (dbl) con_hex64(o->result_d); else con_hex32(o->result);
    con_puts("]");
    CHECK(1);
}

/* "sets the Unimplemented bit ... and traps": E, ExcCode 15, nothing stored. */
static void check_unimplemented(const struct fp_obs *o, int dbl)
{
    CHECK_EQ(o->exceptions, 1u);
    CHECK_EQ(o->excode, (u32)EXC_FPE);
    CHECK_EQ(o->cause & FP_E, (u32)FP_E);
    if (dbl) CHECK_EQ(o->result_d, SENTINEL_D);
    else     CHECK_EQ(o->result, SENTINEL_S);
}

/* ── FCSR.FS ──────────────────────────────────────────────────────────────── */

/* Bit 24. Writable, and readable back — a control bit the suite is about to
 * depend on, so check it exists before drawing conclusions from it. */
static void t_fs_bit_round_trip(void)
{
    fcsr_set(FCSR_FS);
    CHECK_EQ(fcsr() & FCSR_FS, (u32)FCSR_FS);
    /* FS is not one of the sticky fields: writing zero clears it. */
    fcsr_reset();
    CHECK_EQ(fcsr() & FCSR_FS, 0u);
}

/* ── denormalized operands ────────────────────────────────────────────────── */

/*
 * A denormal operand to a computational operation is an Unimplemented
 * Operation on the R4400 — the hardware never produces an arithmetic result at
 * all, so the "answer" is whatever the software handler computes.
 */
static void t_denorm_operand_is_unimplemented(void)
{
    struct fp_obs o = observe_s(F_DENORM, F_1, 0, op_add_s);

    if (is_r4400()) check_unimplemented(&o, 0);
    else            report_obs("denorm operand, R5000", &o, 0);

    /* FS covers denormalized *results*; the manual says nothing about whether
     * it also flushes a denormal operand, and later MIPS revisions widened it
     * to do exactly that. Reported on both parts rather than asserted. */
    o = observe_s(F_DENORM, F_1, FCSR_FS, op_add_s);
    report_obs("denorm operand with FS=1", &o, 0);
}

/*
 * The same rule for a quiet NaN operand, which is the half of Table 7-2 that
 * surprises people: an R4400 does not propagate a NaN through ADD in hardware,
 * it traps and lets software do it. (MOV is exempt — see below.)
 */
static void t_qnan_operand_is_unimplemented(void)
{
    struct fp_obs o = observe_s(F_QNAN, F_1, 0, op_add_s);

    if (is_r4400()) check_unimplemented(&o, 0);
    else            report_obs("qNaN operand, R5000", &o, 0);
}

/* Compare is explicitly exempt — "Denormalized operand, except for Compare
 * instruction" — and comparisons are exact, so a denormal compares as the tiny
 * positive number it is. */
static void t_denorm_compare_does_not_trap(void)
{
    u32 cc;

    fcsr_reset();
    exc_clear();
    w()[0] = F_0; w()[1] = F_DENORM; w()[2] = F_DENORM;
    SYNC();
    __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                            "lwc1 $f2, 4(%0)\n\t"
                            "lwc1 $f4, 8(%0)\n\t"
                            "c.lt.s $f0, $f2\n\t"      /* 0 < denorm */
                            "nop" Z :: "r"(w()));
    cc = (fcsr() & FCSR_CC0) ? 1u : 0u;
    CHECK_NO_EXC();
    CHECK_EQ(cc, 1u);
    CHECK_EQ(fcsr_cause() & FP_E, 0u);

    exc_clear();
    fcsr_set(0);
    __asm__ __volatile__(AF "lwc1 $f2, 4(%0)\n\t"
                            "lwc1 $f4, 8(%0)\n\t"
                            "c.eq.s $f2, $f4\n\t"      /* denorm == denorm */
                            "nop" Z :: "r"(w()));
    cc = (fcsr() & FCSR_CC0) ? 1u : 0u;
    CHECK_NO_EXC();
    CHECK_EQ(cc, 1u);
    fcsr_reset();
}

/*
 * "Moves do not trap if their operands are either denormalized or NaNs."
 * MOV.S is a bit copy, so it must deliver the denormal unchanged — and the
 * load and store that surround it are not FP operations at all.
 */
static void t_denorm_move_does_not_trap(void)
{
    struct fp_obs o = observe_s(F_DENORM, F_0, 0, op_mov_s);

    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.result, F_DENORM);
    CHECK_EQ(o.cause, 0u);
}

/* ── denormalized results ─────────────────────────────────────────────────── */

/*
 * 2^-126 * 0.5 = 2^-127, which is denormal in single precision. With FS clear
 * that is an Unimplemented Operation; the FPU has no way to deliver it.
 */
static void t_denorm_result_without_fs(void)
{
    struct fp_obs o = observe_s(F_MIN_NORM, F_0P5, 0, op_mul_s);

    if (is_r4400()) check_unimplemented(&o, 0);
    else            report_obs("denorm result FS=0, R5000", &o, 0);
}

/*
 * With FS set and neither U nor I enabled, the same multiply takes the
 * Table 7-1 default action instead: "Modify underflow values to 0 with the
 * sign of the intermediate result", with the U and I Cause bits set.
 */
static void t_denorm_result_flushed_with_fs(void)
{
    struct fp_obs o = observe_s(F_MIN_NORM, F_0P5, FCSR_FS, op_mul_s);

    if (is_r4400()) {
        CHECK_EQ(o.exceptions, 0u);
        CHECK_EQ(o.result, F_0);
        CHECK_EQ(o.cause & (FP_U | FP_I), (u32)(FP_U | FP_I));
        CHECK_EQ(o.flags & (FP_U | FP_I), (u32)(FP_U | FP_I));
    } else {
        report_obs("denorm result FS=1, R5000", &o, 0);
    }

    /* The sign of the intermediate result survives the flush. */
    o = observe_s(0x80800000u, F_0P5, FCSR_FS, op_mul_s);   /* -2^-126 * 0.5 */
    if (is_r4400()) CHECK_EQ(o.result, F_NEG0);
    else            report_obs("negative underflow FS=1, R5000", &o, 0);
}

/*
 * FS is not enough on its own. "If Underflow or Inexact traps are enabled, or
 * if the FS bit is not set, then an Unimplemented exception (E) is generated,
 * and the result register is not modified." So arming the underflow trap turns
 * a flushed result back into an E — and the Cause bit is E, not U: there is no
 * underflow trap as such on this part.
 */
static void t_underflow_enable_forces_unimplemented(void)
{
    struct fp_obs o = observe_s(F_MIN_NORM, F_0P5,
                                FCSR_FS | FCSR_ENABLE(FP_U), op_mul_s);

    if (is_r4400()) {
        check_unimplemented(&o, 0);
        CHECK_EQ(o.cause & FP_U, 0u);
    } else {
        report_obs("FS=1 with U enabled, R5000", &o, 0);
    }
}

/* ── the same rules in double precision ───────────────────────────────────── */

/*
 * Nothing about any of this is single-precision-specific, and the emulator's
 * single and double paths are separate code. 2^-1022 * 0.5 = 2^-1023 is
 * denormal in double.
 */
static void t_denorm_double(void)
{
    struct fp_obs o = observe_d(D_DENORM, D_1, 0, op_add_d);

    if (is_r4400()) check_unimplemented(&o, 1);
    else            report_obs("denorm operand .d, R5000", &o, 1);

    o = observe_d(D_MIN_NORM, D_0P5, 0, op_mul_d);
    if (is_r4400()) check_unimplemented(&o, 1);
    else            report_obs("denorm result .d FS=0, R5000", &o, 1);

    o = observe_d(D_MIN_NORM, D_0P5, FCSR_FS, op_mul_d);
    if (is_r4400()) {
        CHECK_EQ(o.exceptions, 0u);
        CHECK_EQ(o.result_d, D_0);
        CHECK_EQ(o.cause & (FP_U | FP_I), (u32)(FP_U | FP_I));
    } else {
        report_obs("denorm result .d FS=1, R5000", &o, 1);
    }
}

static const struct test tests[] = {
    TEST("fpu/fs_bit_round_trip",   t_fs_bit_round_trip,                 CPU_ALL),
    TEST("fpu/denorm_operand",      t_denorm_operand_is_unimplemented,   CPU_ALL),
    TEST("fpu/qnan_operand",        t_qnan_operand_is_unimplemented,     CPU_ALL),
    TEST("fpu/denorm_compare",      t_denorm_compare_does_not_trap,      CPU_ALL),
    TEST("fpu/denorm_move",         t_denorm_move_does_not_trap,         CPU_ALL),
    TEST("fpu/denorm_result_fs0",   t_denorm_result_without_fs,          CPU_ALL),
    TEST("fpu/denorm_result_fs1",   t_denorm_result_flushed_with_fs,     CPU_ALL),
    TEST("fpu/underflow_enable_e",  t_underflow_enable_forces_unimplemented, CPU_ALL),
    TEST("fpu/denorm_double",       t_denorm_double,                     CPU_ALL),
};

const struct test_group group_fpu_denorm = {
    "fpu_denorm", tests, sizeof(tests) / sizeof(tests[0])
};
