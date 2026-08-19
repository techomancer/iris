/* fpu_double — the double-precision paths fpu.c never exercised.
 *
 * fpu.c tests single precision thoroughly and double precision barely: three
 * arithmetic operations and two conversions. That matters because an emulator
 * implements the two formats in separate code, so single-precision coverage
 * says nothing at all about the double-precision path. fpu_vectors.c now
 * covers add/sub/mul/div/sqrt and every conversion against generated tables;
 * what is left, and what is here, is the behaviour that has no vector because
 * its result is a NaN or its rule is about bits rather than arithmetic.
 */

#include "fpu_common.h"

FPOP_D(op_add_d,  "add.d $f4, $f0, $f2")
FPOP_D(op_sub_d,  "sub.d $f4, $f0, $f2")
FPOP_D(op_mul_d,  "mul.d $f4, $f0, $f2")
FPOP_D(op_div_d,  "div.d $f4, $f0, $f2")
FPOP_D(op_sqrt_d, "sqrt.d $f4, $f0")
FPOP_D(op_abs_d,  "abs.d $f4, $f0")
FPOP_D(op_neg_d,  "neg.d $f4, $f0")
FPOP_D(op_mov_d,  "mov.d $f4, $f0")

/* ── bit operations, not arithmetic ───────────────────────────────────────── */

/*
 * MOV.D copies a doubleword. It is not an arithmetic operation — "A move (MOV)
 * operation is not considered to be an arithmetic operation" — so it neither
 * quiets a NaN nor raises anything, and it is exempt from the Unimplemented
 * Operation rule that catches denormal and NaN operands elsewhere.
 */
static void t_mov_d_is_a_bit_copy(void)
{
    struct fp_obs o;

    o = observe_d(D_QNAN, D_0, 0, op_mov_d);
    CHECK_EQ(o.result_d, D_QNAN);
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.flags, 0u);

    /* A signalling NaN too: MOV does not quiet it and does not signal. */
    o = observe_d(D_SNAN, D_0, 0, op_mov_d);
    CHECK_EQ(o.result_d, D_SNAN);
    CHECK_EQ(o.flags & FP_V, 0u);

    o = observe_d(0x0123456789ABCDEFull, D_0, 0, op_mov_d);
    CHECK_EQ(o.result_d, 0x0123456789ABCDEFull);
}

/* ABS.D and NEG.D are sign-bit operations on finite values. */
static void t_abs_neg_double(void)
{
    struct fp_obs o;

    o = observe_d(D_NEG1, D_0, 0, op_abs_d);
    CHECK_EQ(o.result_d, D_1);
    o = observe_d(D_1, D_0, 0, op_abs_d);
    CHECK_EQ(o.result_d, D_1);
    o = observe_d(D_NEG0, D_0, 0, op_abs_d);
    CHECK_EQ(o.result_d, D_0);              /* |-0| = +0 */

    o = observe_d(D_2, D_0, 0, op_neg_d);
    CHECK_EQ(o.result_d, 0xC000000000000000ull);
    o = observe_d(D_0, D_0, 0, op_neg_d);
    CHECK_EQ(o.result_d, D_NEG0);           /* -(+0) = -0 */
    o = observe_d(D_NEGINF, D_0, 0, op_abs_d);
    CHECK_EQ(o.result_d, D_INF);
    CHECK_EQ(o.flags, 0u);
}

/* ── signed zeros and infinities ──────────────────────────────────────────── */

static void t_double_signed_zero(void)
{
    struct fp_obs o;

    o = observe_d(D_0, D_NEG0, 0, op_add_d);
    CHECK_EQ(o.result_d, D_0);              /* (+0) + (-0) = +0 in RN */
    o = observe_d(D_NEG0, D_NEG0, 0, op_add_d);
    CHECK_EQ(o.result_d, D_NEG0);
    o = observe_d(D_2, D_NEG0, 0, op_mul_d);
    CHECK_EQ(o.result_d, D_NEG0);           /* 2 * -0 = -0 */
    o = observe_d(D_1, D_NEG1, 0, op_add_d);
    CHECK_EQ(o.result_d, D_0);              /* exact cancellation is +0 in RN */
}

static void t_double_infinity(void)
{
    struct fp_obs o;

    o = observe_d(D_INF, D_1, 0, op_add_d);
    CHECK_EQ(o.result_d, D_INF);
    o = observe_d(D_INF, D_INF, 0, op_mul_d);
    CHECK_EQ(o.result_d, D_INF);
    o = observe_d(D_1, D_INF, 0, op_div_d);
    CHECK_EQ(o.result_d, D_0);              /* 1/inf = +0 */
    o = observe_d(D_NEGINF, D_1, 0, op_mul_d);
    CHECK_EQ(o.result_d, D_NEGINF);
    CHECK_EQ(o.flags, 0u);                  /* none of that raises anything */
}

/* ── the exceptional results the vector tables cannot express ─────────────── */

/*
 * Every case here produces a NaN, whose payload is implementation-defined —
 * so the generator refuses to predict it and these are asserted as
 * "some NaN", with the flags checked exactly.
 */
static void t_double_invalid_operations(void)
{
    struct fp_obs o;

    o = observe_d(D_0, D_0, 0, op_div_d);            /* 0/0 */
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.flags & FP_V, (u32)FP_V);
    CHECK(is_nan_d(o.result_d));

    o = observe_d(D_INF, D_INF, 0, op_sub_d);        /* inf - inf */
    CHECK_EQ(o.flags & FP_V, (u32)FP_V);
    CHECK(is_nan_d(o.result_d));

    o = observe_d(D_INF, D_INF, 0, op_div_d);        /* inf/inf */
    CHECK_EQ(o.flags & FP_V, (u32)FP_V);
    CHECK(is_nan_d(o.result_d));

    o = observe_d(D_0, D_INF, 0, op_mul_d);          /* 0 * inf */
    CHECK_EQ(o.flags & FP_V, (u32)FP_V);
    CHECK(is_nan_d(o.result_d));

    o = observe_d(0xC000000000000000ull, D_0, 0, op_sqrt_d);   /* sqrt(-2) */
    CHECK_EQ(o.flags & FP_V, (u32)FP_V);
    CHECK(is_nan_d(o.result_d));
}

/* ── conversions between the two formats ──────────────────────────────────── */

static void cvt_s_d(void)      /* double at d[0] -> single at w[4] */
{
    __asm__ __volatile__(AF "ldc1 $f0, 0(%0)\n\t"
                            "cvt.s.d $f4, $f0\n\t"
                            "swc1 $f4, 16(%0)" Z :: "r"(d()) : "memory");
}

static void cvt_d_s(void)      /* single at w[0] -> double at d[2] */
{
    __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                            "cvt.d.s $f4, $f0\n\t"
                            "sdc1 $f4, 16(%0)" Z :: "r"(w()) : "memory");
}

/*
 * Widening is always exact: every single-precision value is a double. The
 * value below is 1 + 2^-23, the smallest single above 1.0; as a double its
 * significand bit lands at 52-23 = 29, so the pattern is 1<<29 in the
 * mantissa.
 */
static void t_convert_single_to_double_is_exact(void)
{
    w()[0] = 0x3F800001u; d()[2] = 0;
    SYNC();
    fcsr_set(0);
    cvt_d_s();
    SYNC();
    CHECK_EQ(d()[2], 0x3FF0000020000000ull);
    CHECK_EQ(fcsr_flags(), 0u);

    /* Infinities and zeros widen unchanged. */
    w()[0] = F_NEGINF;
    SYNC();
    cvt_d_s();
    SYNC();
    CHECK_EQ(d()[2], D_NEGINF);

    w()[0] = F_NEG0;
    SYNC();
    cvt_d_s();
    SYNC();
    CHECK_EQ(d()[2], D_NEG0);
    CHECK_EQ(fcsr_flags(), 0u);
    fcsr_reset();
}

/*
 * Narrowing rounds, and can overflow.
 *
 *   1 + 2^-30 is representable as a double (mantissa bit at 52-30 = 22) and is
 *   nowhere near representable as a single: half an ulp of 1.0f is 2^-24, and
 *   2^-30 is far below it, so the result is exactly 1.0f with Inexact set.
 *
 *   2^200 is an ordinary double and hopelessly out of range for a single,
 *   whose largest finite value is just under 2^128. Table 7-1 says
 *   round-to-nearest turns an overflow into an infinity, with O and I set.
 */
static void t_convert_double_to_single_rounds(void)
{
    d()[0] = 0x3FF0000000400000ull;         /* 1 + 2^-30 */
    w()[4] = 0;
    SYNC();
    fcsr_set(0);
    cvt_s_d();
    SYNC();
    CHECK_EQ(w()[4], F_1);
    CHECK_EQ(fcsr_flags(), (u32)FP_I);

    d()[0] = 0x4C70000000000000ull;         /* 2^200 */
    w()[4] = 0;
    SYNC();
    fcsr_set(0);
    cvt_s_d();
    SYNC();
    CHECK_EQ(w()[4], F_INF);
    CHECK_EQ(fcsr_flags(), (u32)(FP_O | FP_I));

    /* Narrowing a value that fits is exact. */
    d()[0] = D_3;
    w()[4] = 0;
    SYNC();
    fcsr_set(0);
    cvt_s_d();
    SYNC();
    CHECK_EQ(w()[4], F_3);
    CHECK_EQ(fcsr_flags(), 0u);
    fcsr_reset();
}

/* Rounding mode applies to the narrowing conversion as it does to arithmetic:
 * 1 + 2^-30 rounds up to the next single under round-to-plus-infinity. */
static void t_convert_double_to_single_rounding_mode(void)
{
    d()[0] = 0x3FF0000000400000ull;         /* 1 + 2^-30 */

    w()[4] = 0;
    SYNC();
    fcsr_set(FCSR_RM_RP);
    cvt_s_d();
    SYNC();
    CHECK_EQ(w()[4], 0x3F800001u);          /* next single above 1.0 */

    w()[4] = 0;
    SYNC();
    fcsr_set(FCSR_RM_RZ);
    cvt_s_d();
    SYNC();
    CHECK_EQ(w()[4], F_1);

    /* And the negative value rounds the other way under RM. */
    d()[0] = 0xBFF0000000400000ull;         /* -(1 + 2^-30) */
    w()[4] = 0;
    SYNC();
    fcsr_set(FCSR_RM_RM);
    cvt_s_d();
    SYNC();
    CHECK_EQ(w()[4], 0xBF800001u);
    fcsr_reset();
}

static const struct test tests[] = {
    TEST("fpu/double_mov_bitwise",    t_mov_d_is_a_bit_copy,                CPU_ALL),
    TEST("fpu/double_abs_neg",        t_abs_neg_double,                     CPU_ALL),
    TEST("fpu/double_signed_zero",    t_double_signed_zero,                 CPU_ALL),
    TEST("fpu/double_infinity",       t_double_infinity,                    CPU_ALL),
    TEST("fpu/double_invalid_ops",    t_double_invalid_operations,          CPU_ALL),
    TEST("fpu/cvt_d_s_exact",         t_convert_single_to_double_is_exact,  CPU_ALL),
    TEST("fpu/cvt_s_d_rounds",        t_convert_double_to_single_rounds,    CPU_ALL),
    TEST("fpu/cvt_s_d_rm",            t_convert_double_to_single_rounding_mode, CPU_ALL),
};

const struct test_group group_fpu_double = {
    "fpu_double", tests, sizeof(tests) / sizeof(tests[0])
};
