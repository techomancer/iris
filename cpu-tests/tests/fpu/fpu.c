/* fpu — the floating-point unit: arithmetic, conversions, rounding modes,
 * comparisons, FCSR bookkeeping, and the FR-bit register aliasing.
 *
 * Values cross the C/asm boundary through memory, never through FP registers.
 * The suite is built -msoft-float precisely so the compiler never touches the
 * FPU on its own — these tests change FR and FCSR underneath it, and any
 * compiler-generated FP code would silently depend on the state being changed.
 * A side effect is that GAS refuses FP mnemonics without `.set hardfloat`,
 * which the AF prologue supplies.
 *
 * Expected bit patterns are IEEE-754 values, not recordings: 0.25f is
 * 0x3E800000 because that is what the format says, and a mismatch is a real
 * disagreement rather than a changed baseline.
 */

#include "testlib.h"
#include "cp0.h"

#define A  ".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"
#define AF A ".set hardfloat\n\t"
#define Z  "\n\t.set pop"

extern char _scratch_start[];

/* A word-addressable and doubleword-addressable view of the scratch area. */
static volatile u32 *w(void) { return (volatile u32 *)_scratch_start; }
static volatile u64 *d(void) { return (volatile u64 *)_scratch_start; }

/* IEEE-754 single-precision bit patterns used throughout. */
#define F_0        0x00000000u
#define F_NEG0     0x80000000u
#define F_1        0x3F800000u
#define F_2        0x40000000u
#define F_3        0x40400000u
#define F_4        0x40800000u
#define F_5        0x40A00000u
#define F_6        0x40C00000u
#define F_0P5      0x3F000000u
#define F_0P25     0x3E800000u
#define F_INF      0x7F800000u
#define F_NEGINF   0xFF800000u
#define F_QNAN     0x7FC00000u
#define F_SNAN     0x7FA00000u
#define F_MAX      0x7F7FFFFFu   /* largest finite single */
#define F_MIN_NORM 0x00800000u   /* smallest normal single */
#define F_DENORM   0x00000001u   /* smallest denormal      */

/* Double-precision. */
#define D_1        0x3FF0000000000000ull
#define D_2        0x4000000000000000ull
#define D_3        0x4008000000000000ull
#define D_0P5      0x3FE0000000000000ull
#define D_INF      0x7FF0000000000000ull
#define D_QNAN     0x7FF8000000000000ull

/* Run one single-precision op on scratch[0] and scratch[1], result to [2]. */
#define FOP_S(insn)                                                        \
    __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"                          \
                            "lwc1 $f2, 4(%0)\n\t"                          \
                            insn "\n\t"                                    \
                            "swc1 $f4, 8(%0)" Z                            \
                         :: "r"(w()) : "memory")

#define FOP_D(insn)                                                        \
    __asm__ __volatile__(AF "ldc1 $f0, 0(%0)\n\t"                          \
                            "ldc1 $f2, 8(%0)\n\t"                          \
                            insn "\n\t"                                    \
                            "sdc1 $f4, 16(%0)" Z                           \
                         :: "r"(d()) : "memory")

static u32 run_s(u32 a, u32 b, const char *unused, void (*op)(void))
{
    (void)unused;
    w()[0] = a; w()[1] = b; w()[2] = 0;
    SYNC();
    op();
    SYNC();
    return w()[2];
}

/* Reset FCSR to a known state: round-to-nearest, no enables, no flags. */
static void fcsr_reset(void)
{
    fcsr_set(0);
}

/* ── data movement ────────────────────────────────────────────────────────── */

static void t_mtc1_mfc1_round_trip(void)
{
    u64 out;
    u64 in = 0x12345678ull;
    __asm__ __volatile__(AF "mtc1 %1, $f0\n\t"
                            "nop\n\t"
                            "mfc1 %0, $f0" Z
                         : "=r"(out) : "r"(in));
    /* MFC1 delivers a sign-extended 32-bit value. */
    CHECK_EQ(out, 0x0000000012345678ull);

    in = 0x87654321ull;
    __asm__ __volatile__(AF "mtc1 %1, $f0\n\t"
                            "nop\n\t"
                            "mfc1 %0, $f0" Z
                         : "=r"(out) : "r"(in));
    CHECK_EQ(out, 0xFFFFFFFF87654321ull);
}

static void t_dmtc1_dmfc1_round_trip(void)
{
    u64 out, in = 0x0123456789ABCDEFull;
    /* DMTC1 needs FR=1 to address a full 64-bit register; start.S sets it. */
    __asm__ __volatile__(AF "dmtc1 %1, $f0\n\t"
                            "nop\n\t"
                            "dmfc1 %0, $f0" Z
                         : "=r"(out) : "r"(in));
    CHECK_EQ(out, 0x0123456789ABCDEFull);
}

static void t_lwc1_swc1_ldc1_sdc1(void)
{
    w()[0] = 0xAABBCCDDu; w()[1] = 0;
    SYNC();
    __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                            "swc1 $f0, 4(%0)" Z :: "r"(w()) : "memory");
    SYNC();
    CHECK_EQ(w()[1], 0xAABBCCDDu);

    d()[0] = 0x0011223344556677ull; d()[1] = 0;
    SYNC();
    __asm__ __volatile__(AF "ldc1 $f0, 0(%0)\n\t"
                            "sdc1 $f0, 8(%0)" Z :: "r"(d()) : "memory");
    SYNC();
    CHECK_EQ(d()[1], 0x0011223344556677ull);
}

/* ── single-precision arithmetic ──────────────────────────────────────────── */

static void op_add_s(void) { FOP_S("add.s $f4, $f0, $f2"); }
static void op_sub_s(void) { FOP_S("sub.s $f4, $f0, $f2"); }
static void op_mul_s(void) { FOP_S("mul.s $f4, $f0, $f2"); }
static void op_div_s(void) { FOP_S("div.s $f4, $f0, $f2"); }
static void op_sqrt_s(void){ FOP_S("sqrt.s $f4, $f0"); }
static void op_abs_s(void) { FOP_S("abs.s $f4, $f0"); }
static void op_neg_s(void) { FOP_S("neg.s $f4, $f0"); }
static void op_mov_s(void) { FOP_S("mov.s $f4, $f0"); }

static void t_arith_single(void)
{
    fcsr_reset();
    CHECK_EQ(run_s(F_2, F_3, 0, op_add_s), F_5);
    CHECK_EQ(run_s(F_5, F_3, 0, op_sub_s), F_2);
    CHECK_EQ(run_s(F_2, F_3, 0, op_mul_s), F_6);
    CHECK_EQ(run_s(F_6, F_3, 0, op_div_s), F_2);
    CHECK_EQ(run_s(F_4, 0,   0, op_sqrt_s), F_2);
    CHECK_EQ(run_s(0xC0000000u, 0, 0, op_abs_s), F_2);   /* |-2| = 2 */
    CHECK_EQ(run_s(F_2, 0, 0, op_neg_s), 0xC0000000u);   /* -(2) = -2 */
    CHECK_EQ(run_s(0xDEADBEEFu, 0, 0, op_mov_s), 0xDEADBEEFu);
}

/* MOV.S, ABS.S and NEG.S are bit operations, not arithmetic: they must not
 * quiet a NaN or raise Invalid. */
static void t_mov_abs_neg_are_bitwise(void)
{
    fcsr_reset();
    CHECK_EQ(run_s(F_QNAN, 0, 0, op_mov_s), F_QNAN);
    CHECK_EQ(run_s(F_NEG0, 0, 0, op_abs_s), F_0);       /* |-0| = +0 */
    CHECK_EQ(run_s(F_0, 0, 0, op_neg_s), F_NEG0);       /* -(+0) = -0 */
}

/* ── signed zeros and infinities ──────────────────────────────────────────── */

static void t_signed_zero(void)
{
    fcsr_reset();
    /* (+0) + (-0) = +0 in round-to-nearest. */
    CHECK_EQ(run_s(F_0, F_NEG0, 0, op_add_s), F_0);
    /* (-0) + (-0) = -0. */
    CHECK_EQ(run_s(F_NEG0, F_NEG0, 0, op_add_s), F_NEG0);
    /* 2 * -0 = -0. */
    CHECK_EQ(run_s(F_2, F_NEG0, 0, op_mul_s), F_NEG0);
}

static void t_infinity(void)
{
    fcsr_reset();
    CHECK_EQ(run_s(F_INF, F_1, 0, op_add_s), F_INF);
    CHECK_EQ(run_s(F_INF, F_INF, 0, op_mul_s), F_INF);
    CHECK_EQ(run_s(F_1, F_INF, 0, op_div_s), F_0);        /* 1/inf = +0 */
    CHECK_EQ(run_s(F_NEGINF, F_1, 0, op_mul_s), F_NEGINF);
}

/* Division by zero sets the Z flag and produces a correctly-signed infinity;
 * with the enable bit clear it must not trap. */
static void t_divide_by_zero_flag(void)
{
    fcsr_reset();
    exc_clear();
    CHECK_EQ(run_s(F_1, F_0, 0, op_div_s), F_INF);
    CHECK_NO_EXC();
    CHECK_EQ((fcsr() >> FCSR_FLAGS_SHIFT) & FP_Z, (u32)FP_Z);

    fcsr_reset();
    CHECK_EQ(run_s(0xBF800000u, F_0, 0, op_div_s), F_NEGINF);  /* -1/0 */
}

/* 0/0 and inf-inf are Invalid, and produce a quiet NaN. */
static void t_invalid_operations(void)
{
    u32 r;
    fcsr_reset();
    exc_clear();
    r = run_s(F_0, F_0, 0, op_div_s);
    CHECK_NO_EXC();
    CHECK_EQ((fcsr() >> FCSR_FLAGS_SHIFT) & FP_V, (u32)FP_V);
    /* The result is a quiet NaN — the exact payload is implementation-defined,
     * so only NaN-ness is asserted. */
    CHECK((r & 0x7F800000u) == 0x7F800000u && (r & 0x007FFFFFu) != 0);

    fcsr_reset();
    r = run_s(F_INF, F_INF, 0, op_sub_s);
    CHECK_EQ((fcsr() >> FCSR_FLAGS_SHIFT) & FP_V, (u32)FP_V);
    CHECK((r & 0x7F800000u) == 0x7F800000u && (r & 0x007FFFFFu) != 0);
}

/* sqrt of a negative is Invalid. */
static void t_sqrt_negative_is_invalid(void)
{
    u32 r;
    fcsr_reset();
    exc_clear();
    r = run_s(0xC0000000u, 0, 0, op_sqrt_s);      /* sqrt(-2) */
    CHECK_NO_EXC();
    CHECK_EQ((fcsr() >> FCSR_FLAGS_SHIFT) & FP_V, (u32)FP_V);
    CHECK((r & 0x7F800000u) == 0x7F800000u && (r & 0x007FFFFFu) != 0);
}

/* ── double precision ─────────────────────────────────────────────────────── */

static void t_arith_double(void)
{
    fcsr_reset();
    d()[0] = D_1; d()[1] = D_2; d()[2] = 0;
    SYNC();
    FOP_D("add.d $f4, $f0, $f2");
    SYNC();
    CHECK_EQ(d()[2], D_3);

    d()[0] = D_1; d()[1] = D_2;
    SYNC();
    FOP_D("div.d $f4, $f0, $f2");
    SYNC();
    CHECK_EQ(d()[2], D_0P5);

    d()[0] = D_2; d()[1] = D_2;
    SYNC();
    FOP_D("mul.d $f4, $f0, $f2");
    SYNC();
    CHECK_EQ(d()[2], 0x4010000000000000ull);   /* 4.0 */
}

/* ── conversions ──────────────────────────────────────────────────────────── */

/* CVT.D.S widens exactly; CVT.S.D narrows with rounding. */
static void t_convert_between_formats(void)
{
    fcsr_reset();
    w()[0] = F_2;
    SYNC();
    __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                            "cvt.d.s $f4, $f0\n\t"
                            "sdc1 $f4, 8(%0)" Z :: "r"(w()) : "memory");
    SYNC();
    CHECK_EQ(d()[1], D_2);

    d()[0] = D_3;
    SYNC();
    __asm__ __volatile__(AF "ldc1 $f0, 0(%0)\n\t"
                            "cvt.s.d $f4, $f0\n\t"
                            "swc1 $f4, 16(%0)" Z :: "r"(d()) : "memory");
    SYNC();
    CHECK_EQ(w()[4], F_3);
}

/* CVT.W.S rounds per the current mode; TRUNC/CEIL/FLOOR/ROUND ignore it. */
static void t_convert_to_word_rounding_modes(void)
{
    static const struct { u32 in; s32 rn, rz, rp, rm; } cases[] = {
        /*  2.5 */ { 0x40200000u,  2,  2,  3,  2 },
        /*  3.5 */ { 0x40600000u,  4,  3,  4,  3 },
        /* -2.5 */ { 0xC0200000u, -2, -2, -2, -3 },
        /*  1.5 */ { 0x3FC00000u,  2,  1,  2,  1 },
    };
    unsigned i;
    static const u32 modes[4] = { FCSR_RM_RN, FCSR_RM_RZ, FCSR_RM_RP, FCSR_RM_RM };

    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        unsigned m;
        s32 want[4];
        want[0] = cases[i].rn; want[1] = cases[i].rz;
        want[2] = cases[i].rp; want[3] = cases[i].rm;

        for (m = 0; m < 4; m++) {
            fcsr_set(modes[m]);
            w()[0] = cases[i].in; w()[2] = 0;
            SYNC();
            __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                                    "cvt.w.s $f4, $f0\n\t"
                                    "swc1 $f4, 8(%0)" Z :: "r"(w()) : "memory");
            SYNC();
            CHECK_EQ_AT("case*4+mode", i * 4 + m, (s32)w()[2], want[m]);
        }
    }
    fcsr_reset();
}

/* ROUND/TRUNC/CEIL/FLOOR use a fixed mode regardless of FCSR.RM — set RM to
 * something contradictory and confirm each still does its own thing. */
static void t_fixed_rounding_conversions(void)
{
    fcsr_set(FCSR_RM_RZ);          /* deliberately not round-to-nearest */
    w()[0] = 0x40200000u;          /* 2.5 */
    SYNC();

    __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                            "round.w.s $f4, $f0\n\tswc1 $f4, 8(%0)\n\t"
                            "trunc.w.s $f4, $f0\n\tswc1 $f4, 12(%0)\n\t"
                            "ceil.w.s  $f4, $f0\n\tswc1 $f4, 16(%0)\n\t"
                            "floor.w.s $f4, $f0\n\tswc1 $f4, 20(%0)" Z
                         :: "r"(w()) : "memory");
    SYNC();
    CHECK_EQ((s32)w()[2], 2);      /* round: 2.5 -> 2, ties to even */
    CHECK_EQ((s32)w()[3], 2);      /* trunc */
    CHECK_EQ((s32)w()[4], 3);      /* ceil  */
    CHECK_EQ((s32)w()[5], 2);      /* floor */
    fcsr_reset();
}

/* CVT.L / conversions to and from 64-bit integers. */
static void t_long_conversions(void)
{
    fcsr_reset();
    d()[0] = D_3;
    SYNC();
    __asm__ __volatile__(AF "ldc1 $f0, 0(%0)\n\t"
                            "cvt.l.d $f4, $f0\n\t"
                            "sdc1 $f4, 8(%0)" Z :: "r"(d()) : "memory");
    SYNC();
    CHECK_EQ(d()[1], 3ull);

    d()[0] = 42ull;
    SYNC();
    __asm__ __volatile__(AF "ldc1 $f0, 0(%0)\n\t"
                            "cvt.d.l $f4, $f0\n\t"
                            "sdc1 $f4, 8(%0)" Z :: "r"(d()) : "memory");
    SYNC();
    CHECK_EQ(d()[1], 0x4045000000000000ull);   /* 42.0 */
}

/* ── comparisons ──────────────────────────────────────────────────────────── */

/* Compare a against b with the given predicate and return the condition bit. */
static u32 compare_s(u32 a, u32 b, void (*op)(void))
{
    w()[0] = a; w()[1] = b;
    SYNC();
    fcsr_set(fcsr() & ~FCSR_CC0);
    op();
    return (fcsr() & FCSR_CC0) ? 1u : 0u;
}

static void cmp_eq(void)  { __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\tlwc1 $f2, 4(%0)\n\tc.eq.s $f0, $f2\n\tnop" Z :: "r"(w())); }
static void cmp_lt(void)  { __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\tlwc1 $f2, 4(%0)\n\tc.lt.s $f0, $f2\n\tnop" Z :: "r"(w())); }
static void cmp_le(void)  { __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\tlwc1 $f2, 4(%0)\n\tc.le.s $f0, $f2\n\tnop" Z :: "r"(w())); }
static void cmp_un(void)  { __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\tlwc1 $f2, 4(%0)\n\tc.un.s $f0, $f2\n\tnop" Z :: "r"(w())); }
static void cmp_ueq(void) { __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\tlwc1 $f2, 4(%0)\n\tc.ueq.s $f0, $f2\n\tnop" Z :: "r"(w())); }
static void cmp_f(void)   { __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\tlwc1 $f2, 4(%0)\n\tc.f.s $f0, $f2\n\tnop" Z :: "r"(w())); }

static void t_comparisons_ordered(void)
{
    fcsr_reset();
    CHECK_EQ(compare_s(F_1, F_1, cmp_eq), 1u);
    CHECK_EQ(compare_s(F_1, F_2, cmp_eq), 0u);
    CHECK_EQ(compare_s(F_1, F_2, cmp_lt), 1u);
    CHECK_EQ(compare_s(F_2, F_1, cmp_lt), 0u);
    CHECK_EQ(compare_s(F_1, F_1, cmp_le), 1u);
    CHECK_EQ(compare_s(F_2, F_1, cmp_le), 0u);
    /* C.F is always false, by definition. */
    CHECK_EQ(compare_s(F_1, F_1, cmp_f), 0u);
    /* +0 == -0 numerically, despite different bit patterns. */
    CHECK_EQ(compare_s(F_0, F_NEG0, cmp_eq), 1u);
}

/*
 * NaN comparisons are the interesting half. Every ordered predicate is false
 * when either operand is NaN; the unordered predicates are true. Comparing a
 * quiet NaN with a non-signalling predicate must NOT set Invalid.
 */
static void t_comparisons_with_nan(void)
{
    fcsr_reset();
    CHECK_EQ(compare_s(F_QNAN, F_1, cmp_eq), 0u);
    CHECK_EQ(compare_s(F_QNAN, F_1, cmp_lt), 0u);
    CHECK_EQ(compare_s(F_QNAN, F_1, cmp_le), 0u);
    CHECK_EQ(compare_s(F_QNAN, F_1, cmp_un), 1u);
    CHECK_EQ(compare_s(F_QNAN, F_1, cmp_ueq), 1u);
    CHECK_EQ(compare_s(F_1, F_1, cmp_un), 0u);

    /* C.EQ (predicate 2) is a non-signalling compare: a quiet NaN operand must
     * leave the Invalid flag alone. */
    fcsr_reset();
    (void)compare_s(F_QNAN, F_1, cmp_eq);
    CHECK_EQ((fcsr() >> FCSR_FLAGS_SHIFT) & FP_V, 0u);
}

/* ── branches on the FP condition ─────────────────────────────────────────── */

static void t_bc1t_bc1f(void)
{
    u64 taken;

    fcsr_reset();
    (void)compare_s(F_1, F_1, cmp_eq);        /* sets the condition */

    __asm__ __volatile__(AF "daddiu %0, $zero, 0\n\t"
                            "bc1t 1f\n\t"
                            "nop\n\t"
                            "b 2f\n\t"
                            "nop\n\t"
                            "1:\n\t"
                            "daddiu %0, $zero, 1\n\t"
                            "2:" Z : "=&r"(taken));
    CHECK_EQ(taken, 1ull);

    __asm__ __volatile__(AF "daddiu %0, $zero, 0\n\t"
                            "bc1f 1f\n\t"
                            "nop\n\t"
                            "b 2f\n\t"
                            "nop\n\t"
                            "1:\n\t"
                            "daddiu %0, $zero, 1\n\t"
                            "2:" Z : "=&r"(taken));
    CHECK_EQ(taken, 0ull);
}

/* BC1TL/BC1FL nullify their delay slot when not taken, like the integer
 * likely branches. */
static void t_bc1tl_nullifies(void)
{
    u64 taken, slot;

    fcsr_reset();
    (void)compare_s(F_1, F_2, cmp_eq);        /* condition FALSE */

    __asm__ __volatile__(AF "daddiu %0, $zero, 0\n\t"
                            "daddiu %1, $zero, 0\n\t"
                            "bc1tl 1f\n\t"
                            "daddiu %1, $zero, 1\n\t"   /* delay slot */
                            "b 2f\n\t"
                            "nop\n\t"
                            "1:\n\t"
                            "daddiu %0, $zero, 1\n\t"
                            "2:" Z : "=&r"(taken), "=&r"(slot));
    CHECK_EQ(taken, 0ull);
    CHECK_EQ(slot, 0ull);        /* nullified */
}

/* ── FCSR bookkeeping ─────────────────────────────────────────────────────── */

/* Flags are sticky: they accumulate until software clears them. */
static void t_flags_are_sticky(void)
{
    fcsr_reset();
    (void)run_s(F_1, F_0, 0, op_div_s);            /* raises Z */
    CHECK_EQ((fcsr() >> FCSR_FLAGS_SHIFT) & FP_Z, (u32)FP_Z);

    (void)run_s(F_2, F_3, 0, op_add_s);            /* exact, raises nothing */
    /* Z must still be set — flags are not cleared by a later clean op. */
    CHECK_EQ((fcsr() >> FCSR_FLAGS_SHIFT) & FP_Z, (u32)FP_Z);

    fcsr_reset();
    CHECK_EQ((fcsr() >> FCSR_FLAGS_SHIFT) & 0x1Fu, 0u);
}

/* An inexact result sets I. 1/3 is not representable. */
static void t_inexact_flag(void)
{
    fcsr_reset();
    (void)run_s(F_1, F_3, 0, op_div_s);
    CHECK_EQ((fcsr() >> FCSR_FLAGS_SHIFT) & FP_I, (u32)FP_I);
}

/* Overflow to infinity sets O and I. */
static void t_overflow_flag(void)
{
    fcsr_reset();
    {
        u32 r = run_s(F_MAX, F_MAX, 0, op_mul_s);
        CHECK_EQ(r, F_INF);
        CHECK_EQ((fcsr() >> FCSR_FLAGS_SHIFT) & FP_O, (u32)FP_O);
        CHECK_EQ((fcsr() >> FCSR_FLAGS_SHIFT) & FP_I, (u32)FP_I);
    }
}

/* FCSR's reserved bits must not stick. Bits 24 and 22:18 are unimplemented on
 * these parts. */
static void t_fcsr_reserved_bits(void)
{
    fcsr_set(0xFFFFFFFFu);
    /* Bit 24 (FO on later parts) and bits 22:18 are not implemented here. */
    CHECK_EQ(fcsr() & 0x007C0000u, 0u);
    fcsr_reset();
    CHECK_EQ(fcsr(), 0u);
}

/* The rounding-mode field round-trips through all four values. */
static void t_fcsr_rounding_mode_round_trip(void)
{
    u32 m;
    for (m = 0; m < 4; m++) {
        fcsr_set(m);
        CHECK_EQ_AT("rm", m, fcsr() & FCSR_RM_MASK, m);
    }
    fcsr_reset();
}

/* ── FR = 0 register aliasing ─────────────────────────────────────────────── */

/*
 * With Status.FR clear the 32 FP registers become 16 doubleword pairs: an even
 * register holds the low word of a double and the following odd register holds
 * the high word. This is the mode IRIX's o32 binaries run in, so it matters.
 *
 * FR is restored before returning, since the rest of the suite assumes FR=1.
 */
static void t_fr0_pair_aliasing(void)
{
    u32 saved = cp0_status();
    cp0_status_set(saved & ~ST_FR);

    /* Write a double into $f0 (which under FR=0 spans $f0/$f1), then read the
     * two halves back through MFC1 on $f0 and $f1. */
    d()[0] = 0x0123456789ABCDEFull;
    SYNC();
    {
        u64 lo, hi;
        __asm__ __volatile__(AF "ldc1 $f0, 0(%2)\n\t"
                                "mfc1 %0, $f0\n\t"
                                "mfc1 %1, $f1" Z
                             : "=r"(lo), "=r"(hi) : "r"(d()));
        /* Big-endian double in an FR=0 pair: $f0 holds the LOW word and $f1
         * the HIGH word. */
        CHECK_EQ(lo & 0xFFFFFFFFull, 0x89ABCDEFull);
        CHECK_EQ(hi & 0xFFFFFFFFull, 0x01234567ull);
    }

    cp0_status_set(saved);
    CHECK_EQ(cp0_status() & ST_FR, ST_FR);
}

/* With FR=1 the registers are 32 independent 64-bit registers, so $f1 is not
 * part of $f0. */
static void t_fr1_registers_are_independent(void)
{
    u64 a, b;
    CHECK_EQ(cp0_status() & ST_FR, ST_FR);      /* precondition */
    __asm__ __volatile__(AF "dmtc1 %2, $f0\n\t"
                            "dmtc1 %3, $f1\n\t"
                            "nop\n\t"
                            "dmfc1 %0, $f0\n\t"
                            "dmfc1 %1, $f1" Z
                         : "=r"(a), "=r"(b)
                         : "r"(0x1111111111111111ull), "r"(0x2222222222222222ull));
    CHECK_EQ(a, 0x1111111111111111ull);
    CHECK_EQ(b, 0x2222222222222222ull);
}

static const struct test tests[] = {
    TEST("fpu/mtc1_mfc1",          t_mtc1_mfc1_round_trip,           CPU_ALL),
    TEST("fpu/dmtc1_dmfc1",        t_dmtc1_dmfc1_round_trip,         CPU_ALL),
    TEST("fpu/load_store",         t_lwc1_swc1_ldc1_sdc1,            CPU_ALL),
    TEST("fpu/arith_single",       t_arith_single,                   CPU_ALL),
    TEST("fpu/mov_abs_neg_bitwise", t_mov_abs_neg_are_bitwise,       CPU_ALL),
    TEST("fpu/signed_zero",        t_signed_zero,                    CPU_ALL),
    TEST("fpu/infinity",           t_infinity,                       CPU_ALL),
    TEST("fpu/divide_by_zero",     t_divide_by_zero_flag,            CPU_ALL),
    TEST("fpu/invalid_operations", t_invalid_operations,             CPU_ALL),
    TEST("fpu/sqrt_negative",      t_sqrt_negative_is_invalid,       CPU_ALL),
    TEST("fpu/arith_double",       t_arith_double,                   CPU_ALL),
    TEST("fpu/convert_formats",    t_convert_between_formats,        CPU_ALL),
    TEST("fpu/cvt_w_rounding",     t_convert_to_word_rounding_modes, CPU_ALL),
    TEST("fpu/fixed_rounding",     t_fixed_rounding_conversions,     CPU_ALL),
    TEST("fpu/long_conversions",   t_long_conversions,               CPU_ALL),
    TEST("fpu/compare_ordered",    t_comparisons_ordered,            CPU_ALL),
    TEST("fpu/compare_nan",        t_comparisons_with_nan,           CPU_ALL),
    TEST("fpu/bc1t_bc1f",          t_bc1t_bc1f,                      CPU_ALL),
    TEST("fpu/bc1tl_nullifies",    t_bc1tl_nullifies,                CPU_ALL),
    TEST("fpu/flags_sticky",       t_flags_are_sticky,               CPU_ALL),
    TEST("fpu/inexact_flag",       t_inexact_flag,                   CPU_ALL),
    TEST("fpu/overflow_flag",      t_overflow_flag,                  CPU_ALL),
    TEST("fpu/fcsr_reserved",      t_fcsr_reserved_bits,             CPU_ALL),
    TEST("fpu/fcsr_rm_round_trip", t_fcsr_rounding_mode_round_trip,  CPU_ALL),
    TEST("fpu/fr0_pair_aliasing",  t_fr0_pair_aliasing,              CPU_ALL),
    TEST("fpu/fr1_independent",    t_fr1_registers_are_independent,  CPU_ALL),
};

const struct test_group group_fpu = {
    "fpu", tests, sizeof(tests) / sizeof(tests[0])
};
