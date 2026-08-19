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

    o = observe_s(F_SNAN, F_1, 0, op_add_s);
    CHECK_EQ(o.exceptions, 0u);                  /* Invalid is not enabled */
    CHECK_EQ(o.flags & FP_V, (u32)FP_V);
    CHECK(is_nan_s(o.result));
    /* "A quiet NaN is delivered to the destination register." */
    CHECK_EQ(o.result & 0x00400000u, 0x00400000u);

    o = observe_s(F_SNAN, F_0, 0, op_abs_s);
    CHECK_EQ(o.flags & FP_V, (u32)FP_V);

    o = observe_s(F_SNAN, F_0, 0, op_neg_s);
    CHECK_EQ(o.flags & FP_V, (u32)FP_V);

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

    if (o.exceptions) {
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

/* An infinity has no integer representation either, and takes the same path. */
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

    CHECK(o.exceptions == 1u || (o.cause & FP_V) != 0);
    con_printf("\n      [cvt.w.s of +inf: exc=%u cause=%x result=%x]",
               o.exceptions, o.cause, o.result);
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
    TEST("fpu/unaligned_access",    t_unaligned_fp_access,           CPU_ALL),
    TEST("fpu/unaligned_badvaddr",  t_unaligned_badvaddr,            CPU_ALL),
};

const struct test_group group_fpu_breadth = {
    "fpu_breadth", tests, sizeof(tests) / sizeof(tests[0])
};
