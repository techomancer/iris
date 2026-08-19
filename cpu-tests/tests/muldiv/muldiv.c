/* muldiv — multiply, divide, and the HI/LO pair.
 *
 * The 32-bit forms write sign-extended 32-bit halves into HI and LO even
 * though the registers are 64 bits wide; the 64-bit forms write full 64-bit
 * halves. Division by zero does not trap on MIPS — it leaves HI/LO
 * architecturally undefined, which is worth separating carefully from things
 * the spec actually requires (see t_div_by_zero).
 */

#include "testlib.h"
#include "cp0.h"

/* Strict asm prologue — see the note in testlib.h. */
#define A ".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"
#define Z "\n\t.set pop"

#define OPAQUE(x) ({ __typeof__(x) __v = (x); __asm__ __volatile__("" : "+r"(__v)); __v; })

/* Multiplies: the two-operand spelling is already the raw instruction. */
#define MULDIV2(op, a, b, hi, lo)                                          \
    __asm__ __volatile__(A op " %2, %3\n\t"                                \
                           "mfhi %0\n\t"                                   \
                           "mflo %1" Z                                     \
                         : "=r"(hi), "=r"(lo) : "r"(a), "r"(b))

/*
 * Divides: spelled with an explicit `$0` destination, three-operand form.
 *
 * GAS turns the plain two-operand `div rs, rt` into a MACRO — a branch plus
 * `break 0x7` for a zero divisor, and a second branch plus `break 0x6` for the
 * 0x80000000 / -1 overflow case, wrapped around the real instruction. Written
 * the obvious way, the divide-by-zero and overflow tests below measure the
 * assembler's opinion of division rather than the CPU's, and "fail" against
 * traps the CPU never raised. `.set nomacro` in the prologue now makes any
 * such expansion an assembly error instead of a silent one.
 */
#define DIV2(op, a, b, hi, lo)                                             \
    __asm__ __volatile__(A op " $0, %2, %3\n\t"                            \
                           "mfhi %0\n\t"                                   \
                           "mflo %1" Z                                     \
                         : "=r"(hi), "=r"(lo) : "r"(a), "r"(b))

/* ── 32-bit multiply ──────────────────────────────────────────────────────── */

static void t_mult_signed(void)
{
    u64 hi, lo;
    s32 a = OPAQUE((s32)-3), b = OPAQUE((s32)5);
    MULDIV2("mult", a, b, hi, lo);
    /* -15: LO = 0xfffffff1 sign-extended, HI = the sign, also sign-extended. */
    CHECK_EQ(lo, 0xFFFFFFFFFFFFFFF1ull);
    CHECK_EQ(hi, 0xFFFFFFFFFFFFFFFFull);
}

/* The full 64-bit product splits across HI:LO, and each half is independently
 * sign-extended from bit 31. */
static void t_mult_wide_product(void)
{
    u64 hi, lo;
    s32 a = OPAQUE((s32)0x7FFFFFFF), b = OPAQUE((s32)0x7FFFFFFF);
    MULDIV2("mult", a, b, hi, lo);
    /* 0x7fffffff^2 = 0x3FFFFFFF00000001 */
    CHECK_EQ(lo, 0x0000000000000001ull);
    CHECK_EQ(hi, 0x000000003FFFFFFFull);
}

static void t_mult_high_bit_set_in_lo(void)
{
    u64 hi, lo;
    s32 a = OPAQUE((s32)0x00010000), b = OPAQUE((s32)0x00008000);
    MULDIV2("mult", a, b, hi, lo);
    /* product = 0x80000000: LO's bit 31 is set, so LO sign-extends. */
    CHECK_EQ(lo, 0xFFFFFFFF80000000ull);
    CHECK_EQ(hi, 0ull);
}

static void t_multu_unsigned(void)
{
    u64 hi, lo;
    u32 a = OPAQUE(0xFFFFFFFFu), b = OPAQUE(0xFFFFFFFFu);
    MULDIV2("multu", a, b, hi, lo);
    /* 0xfffffffe00000001 */
    CHECK_EQ(lo, 0x0000000000000001ull);
    CHECK_EQ(hi, 0xFFFFFFFFFFFFFFFEull);   /* 0xfffffffe, sign-extended */
}

/* multu must not treat its operands as signed: 0xffffffff * 2 is a large
 * positive product, not -2. */
static void t_multu_operands_are_unsigned(void)
{
    u64 hi, lo;
    u32 a = OPAQUE(0xFFFFFFFFu), b = OPAQUE(2u);
    MULDIV2("multu", a, b, hi, lo);
    /* 0x1FFFFFFFE */
    CHECK_EQ(lo, 0xFFFFFFFFFFFFFFFEull);   /* 0xfffffffe sign-extended */
    CHECK_EQ(hi, 0x0000000000000001ull);
}

/* ── 64-bit multiply ──────────────────────────────────────────────────────── */

static void t_dmult(void)
{
    u64 hi, lo;
    s64 a = OPAQUE((s64)-3), b = OPAQUE((s64)5);
    MULDIV2("dmult", a, b, hi, lo);
    CHECK_EQ(lo, (u64)(s64)-15);
    CHECK_EQ(hi, 0xFFFFFFFFFFFFFFFFull);
}

static void t_dmultu_wide(void)
{
    u64 hi, lo;
    u64 a = OPAQUE(0xFFFFFFFFFFFFFFFFull), b = OPAQUE(0xFFFFFFFFFFFFFFFFull);
    MULDIV2("dmultu", a, b, hi, lo);
    /* (2^64-1)^2 = 2^128 - 2^65 + 1 → hi = 0xfffffffffffffffe, lo = 1 */
    CHECK_EQ(lo, 0x0000000000000001ull);
    CHECK_EQ(hi, 0xFFFFFFFFFFFFFFFEull);
}

/* Unlike the 32-bit forms, dmult's halves are true 64-bit values with no
 * sign extension applied to either. */
static void t_dmult_no_32bit_extension(void)
{
    u64 hi, lo;
    u64 a = OPAQUE(0x0000000100000000ull), b = OPAQUE(0x0000000000000002ull);
    MULDIV2("dmultu", a, b, hi, lo);
    CHECK_EQ(lo, 0x0000000200000000ull);
    CHECK_EQ(hi, 0ull);
}

/* ── division ─────────────────────────────────────────────────────────────── */

static void t_div_signed(void)
{
    u64 hi, lo;
    s32 a = OPAQUE((s32)-17), b = OPAQUE((s32)5);
    DIV2("div", a, b, hi, lo);
    /* Truncation toward zero: -17/5 = -3 remainder -2. */
    CHECK_EQ(lo, (u64)(s64)-3);
    CHECK_EQ(hi, (u64)(s64)-2);
}

static void t_divu_unsigned(void)
{
    u64 hi, lo;
    u32 a = OPAQUE(0xFFFFFFFFu), b = OPAQUE(2u);
    DIV2("divu", a, b, hi, lo);
    CHECK_EQ(lo, 0x000000007FFFFFFFull);
    CHECK_EQ(hi, 0x0000000000000001ull);
}

/* divu treats 0xffffffff as 4294967295, not -1 — the quotient is large and
 * positive, not zero. */
static void t_divu_is_not_signed(void)
{
    u64 hi, lo;
    u32 a = OPAQUE(0xFFFFFFFFu), b = OPAQUE(0x10u);
    DIV2("divu", a, b, hi, lo);
    CHECK_EQ(lo, 0x000000000FFFFFFFull);
    CHECK_EQ(hi, 0x000000000000000Full);
}

static void t_ddiv(void)
{
    u64 hi, lo;
    s64 a = OPAQUE((s64)-17), b = OPAQUE((s64)5);
    DIV2("ddiv", a, b, hi, lo);
    CHECK_EQ(lo, (u64)(s64)-3);
    CHECK_EQ(hi, (u64)(s64)-2);
}

static void t_ddivu(void)
{
    u64 hi, lo;
    u64 a = OPAQUE(0xFFFFFFFFFFFFFFFFull), b = OPAQUE(2ull);
    DIV2("ddivu", a, b, hi, lo);
    CHECK_EQ(lo, 0x7FFFFFFFFFFFFFFFull);
    CHECK_EQ(hi, 1ull);
}

/*
 * The one signed-division case that overflows: the most negative value divided
 * by -1 has no representable quotient. MIPS does not trap. The R4000 manual
 * leaves the result undefined, but every implementation returns the dividend
 * with a zero remainder, which is what a two's-complement divider naturally
 * produces — so the no-trap requirement is asserted and the values are only
 * reported.
 */
static void t_div_overflow_case(void)
{
    u64 hi, lo;
    s32 a = OPAQUE((s32)0x80000000), b = OPAQUE((s32)-1);
    exc_clear();
    DIV2("div", a, b, hi, lo);
    CHECK_NO_EXC();
    con_printf("\n      [div 0x80000000/-1: lo=%lx hi=%lx]", lo, hi);
}

static void t_ddiv_overflow_case(void)
{
    u64 hi, lo;
    s64 a = OPAQUE((s64)0x8000000000000000ull), b = OPAQUE((s64)-1);
    exc_clear();
    DIV2("ddiv", a, b, hi, lo);
    CHECK_NO_EXC();
    con_printf("\n      [ddiv min/-1: lo=%lx hi=%lx]", lo, hi);
}

/*
 * Division by zero raises no exception, and HI/LO are architecturally
 * UNPREDICTABLE. So there is nothing here to assert about their values: the
 * only requirements the spec imposes are that no trap fires and that the
 * machine keeps running, and that is exactly what is checked.
 *
 * The values are printed rather than compared, so a change in what IRIS
 * produces is visible in the log without being a failure. Asserting them would
 * invent an architectural requirement that does not exist.
 */
static void t_div_by_zero(void)
{
    u64 hi, lo, hi2, lo2;
    s32 a = OPAQUE((s32)42), b = OPAQUE((s32)0);
    u32 c = OPAQUE(42u), d = OPAQUE(0u);

    exc_clear();
    DIV2("div", a, b, hi, lo);
    DIV2("divu", c, d, hi2, lo2);
    CHECK_NO_EXC();
    con_printf("\n      [div/0: lo=%lx hi=%lx  divu/0: lo=%lx hi=%lx]",
               lo, hi, lo2, hi2);
}

/* ── HI/LO as registers ───────────────────────────────────────────────────── */

static void t_mthi_mtlo_round_trip(void)
{
    u64 hi, lo;
    u64 a = OPAQUE(0x0123456789ABCDEFull), b = OPAQUE(0xFEDCBA9876543210ull);
    __asm__ __volatile__(A "mthi %2\n\t"
                           "mtlo %3\n\t"
                           "mfhi %0\n\t"
                           "mflo %1" Z
                         : "=r"(hi), "=r"(lo) : "r"(a), "r"(b));
    /* mthi/mtlo move all 64 bits — no truncation, no sign extension. */
    CHECK_EQ(hi, 0x0123456789ABCDEFull);
    CHECK_EQ(lo, 0xFEDCBA9876543210ull);
}

/* A multiply must overwrite BOTH halves, not just the one whose value
 * changes — a stale HI left over from a previous op is a real emulator bug. */
static void t_multiply_overwrites_both_halves(void)
{
    u64 hi, lo;
    u64 poison = OPAQUE(0xDEADBEEFDEADBEEFull);
    s32 a = OPAQUE((s32)6), b = OPAQUE((s32)7);
    __asm__ __volatile__(A "mthi %2\n\t"
                           "mtlo %2\n\t"
                           "mult %3, %4\n\t"
                           "mfhi %0\n\t"
                           "mflo %1" Z
                         : "=r"(hi), "=r"(lo)
                         : "r"(poison), "r"(a), "r"(b));
    CHECK_EQ(lo, 42ull);
    CHECK_EQ(hi, 0ull);
}

static const struct test tests[] = {
    TEST("muldiv/mult_signed",           t_mult_signed,                     CPU_ALL),
    TEST("muldiv/mult_wide_product",     t_mult_wide_product,               CPU_ALL),
    TEST("muldiv/mult_lo_sign_extend",   t_mult_high_bit_set_in_lo,         CPU_ALL),
    TEST("muldiv/multu",                 t_multu_unsigned,                  CPU_ALL),
    TEST("muldiv/multu_unsigned_ops",    t_multu_operands_are_unsigned,     CPU_ALL),
    TEST("muldiv/dmult",                 t_dmult,                           CPU_ALL),
    TEST("muldiv/dmultu_wide",           t_dmultu_wide,                     CPU_ALL),
    TEST("muldiv/dmult_no_32bit_ext",    t_dmult_no_32bit_extension,        CPU_ALL),
    TEST("muldiv/div_signed",            t_div_signed,                      CPU_ALL),
    TEST("muldiv/divu",                  t_divu_unsigned,                   CPU_ALL),
    TEST("muldiv/divu_not_signed",       t_divu_is_not_signed,              CPU_ALL),
    TEST("muldiv/ddiv",                  t_ddiv,                            CPU_ALL),
    TEST("muldiv/ddivu",                 t_ddivu,                           CPU_ALL),
    TEST("muldiv/div_overflow_no_trap",  t_div_overflow_case,               CPU_ALL),
    TEST("muldiv/ddiv_overflow_no_trap", t_ddiv_overflow_case,              CPU_ALL),
    TEST("muldiv/div_by_zero_no_trap",   t_div_by_zero,                     CPU_ALL),
    TEST("muldiv/mthi_mtlo",             t_mthi_mtlo_round_trip,            CPU_ALL),
    TEST("muldiv/mult_writes_both",      t_multiply_overwrites_both_halves, CPU_ALL),
};

const struct test_group group_muldiv = {
    "muldiv", tests, sizeof(tests) / sizeof(tests[0])
};
