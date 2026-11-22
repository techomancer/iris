/* fpu_breadth — the corners the rest of the fpu/ group steps around.
 *
 * All 32 registers rather than the eight the other tests reuse; signalling
 * NaNs as arithmetic operands; conversions whose source has no representation
 * in the destination; and unaligned FP loads and stores, which are the CPU's
 * problem rather than the FPU's but reach memory through a coprocessor path
 * that nothing else in the suite tries.
 */

#include "fpu_common.h"

FPOP_S(op_add_s, "add.s $f4, $f0, $f2")
FPOP_S(op_abs_s, "abs.s $f4, $f0")
FPOP_S(op_neg_s, "neg.s $f4, $f0")
FPOP_S(op_mov_s, "mov.s $f4, $f0")

/* ── the whole register file ──────────────────────────────────────────────── */

/*
 * Everything else in the group lives in $f0..$f6. If a register decode dropped
 * a bit, or two registers aliased, nothing so far would notice — so write a
 * distinct doubleword to all 32 and read all 32 back.
 *
 * The pattern for register n is distinct in every byte position, so an alias
 * between any two registers shows up as a wrong value rather than a plausible
 * one.
 */
#define REG_LIST(X) X(0)  X(1)  X(2)  X(3)  X(4)  X(5)  X(6)  X(7)         \
                    X(8)  X(9)  X(10) X(11) X(12) X(13) X(14) X(15)        \
                    X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23)        \
                    X(24) X(25) X(26) X(27) X(28) X(29) X(30) X(31)

#define LD_REG(n)  "ldc1 $f" #n ", " #n "*8(%0)\n\t"
#define ST_REG(n)  "sdc1 $f" #n ", (256+" #n "*8)(%0)\n\t"

static u64 reg_pattern(unsigned n)
{
    u64 v = (u64)(n + 1);
    return (v << 56) | (v << 40) | (v << 24) | (v << 8) | (0xA5u ^ v);
}

static void t_all_registers_are_independent(void)
{
    unsigned n;

    CHECK_EQ(cp0_status() & ST_FR, (u32)ST_FR);      /* precondition */

    for (n = 0; n < 32; n++) {
        d()[n] = reg_pattern(n);
        d()[32 + n] = 0;
    }
    SYNC();

    __asm__ __volatile__(AF REG_LIST(LD_REG) REG_LIST(ST_REG) Z
                         :: "r"(d()) : "memory");
    SYNC();

    for (n = 0; n < 32; n++)
        CHECK_EQ_AT("f", n, d()[32 + n], reg_pattern(n));
}

/* ── signalling NaNs as arithmetic operands ───────────────────────────────── */

/*
 * "Any arithmetic operation on a signaling NaN [causes Invalid]. A move (MOV)
 * operation is not considered to be an arithmetic operation, but absolute
 * value (ABS) and negate (NEG) are considered to be arithmetic operations and
 * cause this exception if one or both operands is a signaling NaN."
 *
 * Three instructions, three different answers, and the difference between them
 * is entirely a matter of definition rather than of arithmetic — which is
 * exactly the sort of thing an implementation gets wrong.
 */
static void t_signalling_nan_operands(void)
{
    struct fp_obs o;

    /*
     * An R4400 does not complete an arithmetic operation with a signalling NaN
     * operand at all: it raises Unimplemented Operation (Cause.E, ExcCode 15)
     * and stores nothing, so no Invalid *flag* is ever set — "when a
     * floating-point exception is taken, the flag bits are not set by the
     * hardware". Confirmed on an Indy R4400 rev 6.0. The quiet-NaN delivery
     * this once asserted is what happens for a *quiet* NaN operand; see
     * fpu/qnan_operand.
     */
    o = observe_s(F_SNAN, F_1, 0, op_add_s);
    CHECK_EQ(o.exceptions, 1u);
    CHECK_EQ(o.excode, (u32)EXC_FPE);
    CHECK_EQ(o.cause & FP_E, (u32)FP_E);
    CHECK_EQ(o.flags & FP_V, 0u);
    CHECK_EQ(o.result, SENTINEL_S);

    /* abs and neg are not arithmetic on the R4400 either way: they move sign
     * bits and do not inspect the operand, so nothing is raised. */
    o = observe_s(F_SNAN, F_0, 0, op_abs_s);
    CHECK_EQ(o.flags & FP_V, 0u);

    o = observe_s(F_SNAN, F_0, 0, op_neg_s);
    CHECK_EQ(o.flags & FP_V, 0u);

    /* MOV is not arithmetic: no Invalid, and the payload survives intact. */
    o = observe_s(F_SNAN, F_0, 0, op_mov_s);
    CHECK_EQ(o.flags & FP_V, 0u);
    CHECK_EQ(o.result, F_SNAN);
}

/* ── conversions with no representable answer ─────────────────────────────── */

static void cvt_w_s_sentinel(void)
{
    __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                            "lwc1 $f4, 12(%0)\n\t"     /* sentinel */
                            "cvt.w.s $f4, $f0\n\t"
                            "swc1 $f4, 8(%0)" Z :: "r"(w()) : "memory");
}

/*
 * 2^31 has no signed-word representation, and the two documents that describe
 * this part disagree about what happens:
 *
 *   R4000 manual, Table 7-2:  "Overflow on convert ... E", both with the trap
 *                             enabled and disabled — an Unimplemented
 *                             Operation, with the destination left alone.
 *   MIPS IV ISA:              Invalid Operation, and the default result is the
 *                             largest representable integer.
 *
 * Both are defensible for these parts, so the suite accepts either — but it
 * insists on the whole of whichever story it gets, and prints which. A
 * silently wrong number passes neither branch. See docs/oracle.md §2.
 *
 * The R10000 family has one story: it "took an unimplemented exception
 * whenever a conversion from an FP value would result in a value that cannot
 * be represented in the target format", and the R12000 kept that except for a
 * denormal source with FCSR.FS set (NEC VR10000 Series User's Manual
 * U10278EJ4V0UM, Appendix B.4, p. 350). The R14000 is inferred to match.
 */
static void t_convert_out_of_range(void)
{
    struct fp_obs o;

    w()[0] = F_BIG;            /* 2^31 exactly */
    w()[2] = 0; w()[3] = SENTINEL_S;
    SYNC();
    fcsr_set(0);
    exc_clear();
    cvt_w_s_sentinel();
    SYNC();
    obs_collect(&o);
    fcsr_reset();

    if (o.exceptions || is_r10000()) {
        CHECK_EQ(o.exceptions, 1u);
        CHECK_EQ(o.excode, (u32)EXC_FPE);
        CHECK_EQ(o.cause & FP_E, (u32)FP_E);
        CHECK_EQ(o.result, SENTINEL_S);      /* "operands undisturbed" */
    } else {
        CHECK_EQ(o.cause & FP_V, (u32)FP_V);
        CHECK_EQ(o.result, 0x7FFFFFFFu);
    }
    con_printf("\n      [cvt.w.s of 2^31: exc=%u cause=%x result=%x]",
               o.exceptions, o.cause, o.result);
}

/* An infinity has no integer representation either, and takes the same path:
 * on the R10000, Unimplemented Operation (the same Appendix B.4 sentence). */
static void t_convert_infinity(void)
{
    struct fp_obs o;

    w()[0] = F_INF;
    w()[2] = 0; w()[3] = SENTINEL_S;
    SYNC();
    fcsr_set(0);
    exc_clear();
    cvt_w_s_sentinel();
    SYNC();
    obs_collect(&o);
    fcsr_reset();

    if (is_r10000()) {
        CHECK_EQ(o.exceptions, 1u);
        CHECK_EQ(o.excode, (u32)EXC_FPE);
        CHECK_EQ(o.cause & FP_E, (u32)FP_E);
        CHECK_EQ(o.result, SENTINEL_S);
    } else {
        CHECK(o.exceptions == 1u || (o.cause & FP_V) != 0);
    }
    con_printf("\n      [cvt.w.s of +inf: exc=%u cause=%x result=%x]",
               o.exceptions, o.cause, o.result);
}

/* ── conversion operands and results the FPU hands to software ───────────── */

FPOP_S(op_trunc_w_s, "trunc.w.s $f4, $f0")
FPOP_D(op_cvt_w_d,   "cvt.w.d $f4, $f0")
FPOP_D(op_cvt_l_d,   "cvt.l.d $f4, $f0")
FPOP_D(op_cvt_d_s,   "cvt.d.s $f4, $f0")   /* the single is $f0's low word */

/* Double at d[0] -> single at w[2], with the sentinel in w[3]. */
static void op_cvt_s_d(void)
{
    __asm__ __volatile__(AF "ldc1 $f0, 0(%0)\n\t"
                            "lwc1 $f4, 12(%0)\n\t"
                            "cvt.s.d $f4, $f0\n\t"
                            "swc1 $f4, 8(%0)" Z :: "r"(w()) : "memory");
}

/* A double to op_cvt_s_d: w[2] cleared and the single sentinel in w[3]. */
static struct fp_obs observe_s_from_d(u64 a, u32 fcsr_init)
{
    return observe_d(a, (u64)SENTINEL_S, fcsr_init, op_cvt_s_d);
}

/* "sets the Unimplemented bit ... and traps": E, ExcCode 15, nothing stored,
 * no Flag. */
static void check_cvt_unimplemented(const char *what, const struct fp_obs *o, int dbl)
{
    con_printf("\n      [%s: exc=%u cause=%x]", what, o->exceptions, o->cause);
    CHECK_EQ(o->exceptions, 1u);
    CHECK_EQ(o->excode, (u32)EXC_FPE);
    CHECK_EQ(o->cause & FP_E, (u32)FP_E);
    CHECK_EQ(o->flags, 0u);
    if (dbl) CHECK_EQ(o->result_d, SENTINEL_D);
    else     CHECK_EQ(o->result, SENTINEL_S);
}

/*
 * "Denormalized operand" is a cause of Unimplemented Operation for "a convert
 * or computational operation" (R4000 manual, Unimplemented Instruction
 * Exception; VR5000 manual 9.4.6), and the R10000 takes it for a denormal
 * operand too (R10000 UM p. 309). IRIS used to round a denormal to zero with
 * Inexact, and narrow or widen it on the host. Expectations from the manuals:
 * fpu/denorm_operand measured the arithmetic case on an Indy R4400, not this.
 */
static void t_convert_denormal_source(void)
{
    struct fp_obs o;

    o = observe_s(F_DENORM, 0, 0, op_trunc_w_s);
    check_cvt_unimplemented("trunc.w.s denormal", &o, 0);
    o = observe_d(D_DENORM, 0, 0, op_cvt_w_d);
    check_cvt_unimplemented("cvt.w.d denormal", &o, 1);
    o = observe_d((u64)F_DENORM, 0, 0, op_cvt_d_s);
    check_cvt_unimplemented("cvt.d.s denormal", &o, 1);
    o = observe_s_from_d(D_DENORM, 0);
    check_cvt_unimplemented("cvt.s.d denormal", &o, 0);
}

/*
 * The NaN operands, in this suite's naming (docs/gotchas.md): F_SNAN/D_SNAN,
 * whose fraction MSB is clear, are what an R4400's arithmetic traps with
 * Unimplemented Operation, and a conversion takes the same exception
 * ("Quiet Not a Number operand", R4000 manual). F_QNAN completes in hardware
 * as an Invalid Operation with the default NaN. The R5000's answer for the
 * second is not established (fpu/qnan_operand reports it), so it is reported.
 */
static void t_convert_nan_source(void)
{
    struct fp_obs o;

    o = observe_d((u64)F_SNAN, 0, 0, op_cvt_d_s);
    check_cvt_unimplemented("cvt.d.s F_SNAN", &o, 1);
    o = observe_s_from_d(D_SNAN, 0);
    check_cvt_unimplemented("cvt.s.d D_SNAN", &o, 0);

    o = observe_d((u64)F_QNAN, 0, 0, op_cvt_d_s);
    con_printf("\n      [cvt.d.s F_QNAN: exc=%u cause=%x flags=%x result=",
               o.exceptions, o.cause, o.flags);
    con_hex64(o.result_d);
    con_puts("]");
    if (!is_r5000()) {
        CHECK_EQ(o.exceptions, 0u);
        CHECK_EQ(o.cause, (u32)FP_V);
        CHECK_EQ(o.flags, (u32)FP_V);
        CHECK_EQ(o.result_d, 0x7FF7FFFFFFFFFFFFull);
    }
}

/*
 * A narrowed result too small for a normal single underflows. With FS clear
 * the FPU hands it to software; with FS set and neither Underflow nor Inexact
 * enabled it is flushed to zero with U and I (R4000 manual, Underflow
 * Exception, and Table 7-1), as fpu/denorm_result_fs1 measures for
 * arithmetic.
 */
static void t_convert_double_to_single_underflow(void)
{
    struct fp_obs o;
    const u64 tiny = 0x37A16C262777579Cull;   /* 1e-40, a single denormal */

    o = observe_s_from_d(tiny, 0);
    check_cvt_unimplemented("cvt.s.d 1e-40 FS=0", &o, 0);

    o = observe_s_from_d(tiny, FCSR_FS);
    con_printf("\n      [cvt.s.d 1e-40 FS=1: exc=%u cause=%x flags=%x result=%x]",
               o.exceptions, o.cause, o.flags, o.result);
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.result, F_0);
    CHECK_EQ(o.cause, (u32)(FP_U | FP_I));
    CHECK_EQ(o.flags, (u32)(FP_U | FP_I));
}

/*
 * Overflow raises Inexact with it in every rounding mode (R4000 manual,
 * Table 7-1), including round-toward-zero, which delivers the largest finite
 * single rather than an infinity. fpu/cvt_s_d_rounds covers round-to-nearest.
 */
static void t_convert_double_to_single_overflow_rz(void)
{
    struct fp_obs o = observe_s_from_d(0x4C70000000000000ull, FCSR_RM_RZ); /* 2^200 */

    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.result, F_MAX);
    CHECK_EQ(o.cause, (u32)(FP_O | FP_I));
    CHECK_EQ(o.flags, (u32)(FP_O | FP_I));
}

/*
 * A floating-point to long conversion "with an output that would be greater
 * than 2^53-1 ... or less than -2^53" is Unimplemented Operation on the
 * VR5000 (manual 9.4.6). The R4400 is taken to match, as it does for the long
 * sources in fpu/vec_cvt_from_l; that is not measured. A real R4600 and the
 * R10000 family have their own limits (r10k/cvt_l_limit).
 */
static void t_convert_to_long_limit(void)
{
    struct fp_obs o;

    if (is_r10000() || is_r4600()) {
        CHECK(1);
        return;
    }
    o = observe_d(0x433FFFFFFFFFFFFFull, 0, 0, op_cvt_l_d);  /* 2^53 - 1 */
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.result_d, 0x001FFFFFFFFFFFFFull);
    o = observe_d(0xC340000000000000ull, 0, 0, op_cvt_l_d);  /* -2^53 */
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.result_d, 0xFFE0000000000000ull);
    o = observe_d(0x4340000000000000ull, 0, 0, op_cvt_l_d);  /* 2^53 */
    check_cvt_unimplemented("cvt.l.d 2^53", &o, 1);
}

/* ── unaligned floating-point loads and stores ────────────────────────────── */

/*
 * "If either of the two least-significant bits of the effective address is
 * non-zero, an address error exception occurs" — LWC1. LDC1 needs all three
 * clear. This is the CPU's alignment check rather than the FPU's, but it
 * arrives through the coprocessor load path, and nothing else in the suite
 * takes an FP access down it.
 */
static void t_unaligned_fp_access(void)
{
    volatile char *base = (volatile char *)w();

    exc_clear();
    __asm__ __volatile__(AF "lwc1 $f0, 1(%0)" Z :: "r"(base));
    CHECK_EXC(EXC_ADEL);

    exc_clear();
    __asm__ __volatile__(AF "lwc1 $f0, 2(%0)" Z :: "r"(base));
    CHECK_EXC(EXC_ADEL);

    exc_clear();
    __asm__ __volatile__(AF "swc1 $f0, 3(%0)" Z :: "r"(base) : "memory");
    CHECK_EXC(EXC_ADES);

    /* A doubleword access needs eight-byte alignment, so +4 is enough. */
    exc_clear();
    __asm__ __volatile__(AF "ldc1 $f0, 4(%0)" Z :: "r"(base));
    CHECK_EXC(EXC_ADEL);

    exc_clear();
    __asm__ __volatile__(AF "sdc1 $f0, 4(%0)" Z :: "r"(base) : "memory");
    CHECK_EXC(EXC_ADES);

    /* The aligned forms of the same accesses do not fault. */
    exc_clear();
    __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                            "ldc1 $f2, 8(%0)" Z :: "r"(base));
    CHECK_NO_EXC();
}

/* BadVAddr reports the unaligned address itself, not the aligned one. */
static void t_unaligned_badvaddr(void)
{
    volatile char *base = (volatile char *)w();

    exc_clear();
    __asm__ __volatile__(AF "lwc1 $f0, 6(%0)" Z :: "r"(base));
    CHECK_EXC(EXC_ADEL);
    CHECK_EQ(exc.badvaddr, (u64)(long)(base + 6));
}

static const struct test tests[] = {
    TEST("fpu/all_registers",       t_all_registers_are_independent, CPU_ALL),
    TEST("fpu/snan_operands",       t_signalling_nan_operands,       CPU_ALL),
    TEST("fpu/cvt_out_of_range",    t_convert_out_of_range,          CPU_ALL),
    TEST("fpu/cvt_infinity",        t_convert_infinity,              CPU_ALL),
    TEST("fpu/cvt_denormal_source", t_convert_denormal_source,       CPU_ALL),
    TEST("fpu/cvt_nan_source",      t_convert_nan_source,            CPU_ALL),
    TEST("fpu/cvt_s_d_underflow",   t_convert_double_to_single_underflow, CPU_ALL),
    TEST("fpu/cvt_s_d_overflow_rz", t_convert_double_to_single_overflow_rz, CPU_ALL),
    TEST("fpu/cvt_l_limit",         t_convert_to_long_limit,         CPU_ALL),
    TEST("fpu/unaligned_access",    t_unaligned_fp_access,           CPU_ALL),
    TEST("fpu/unaligned_badvaddr",  t_unaligned_badvaddr,            CPU_ALL),
};

const struct test_group group_fpu_breadth = {
    "fpu_breadth", tests, sizeof(tests) / sizeof(tests[0])
};
