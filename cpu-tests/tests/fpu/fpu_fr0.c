/* fpu_fr0 — the FR=0 register file, which is the one IRIX actually uses.
 *
 * Status.FR chooses between two views of the same 32 physical registers:
 *
 *   FR=1   32 independent 64-bit registers. What start.S sets, what the rest
 *          of this suite assumes, and what n32/n64 code uses.
 *   FR=0   16 usable registers, each an even/odd *pair* of 32-bit halves. The
 *          o32 ABI's view — so every IRIX 5.3 binary, and every 32-bit IRIX
 *          6.5 binary, runs the FPU in this mode.
 *
 * The manual (Appendix B, ValueFPR/StoreFPR) is precise about it. For a valid
 * specifier — fpr0 = 0, with 32-bit wide FGRs — S and W formats read
 * FGR[fpr], and D and L formats read FGR[fpr+1] concatenated with FGR[fpr];
 * for an odd register number the value is, in the manual's word, "undefined".
 *
 * So under FR=0 the odd register holds the *high* word of a double, and any
 * format operation naming an odd register is undefined — which this file
 * reports rather than asserts (docs/oracle.md §2). Loads and stores are a
 * different matter and are defined for both halves: "If FR equals zero, LWC1
 * loads either the high or low half of the 16 even Floating-Point registers."
 *
 * Every test here restores FR=1 before returning. The suite assumes it.
 */

#include "fpu_common.h"

/* Run `body` with Status.FR clear, then put FR back. */
#define WITH_FR0(body)                                                     \
    do {                                                                   \
        u32 __saved = cp0_status();                                        \
        cp0_status_set(__saved & ~ST_FR);                                  \
        body;                                                              \
        cp0_status_set(__saved);                                           \
        CHECK_EQ(cp0_status() & ST_FR, (u32)ST_FR);                        \
    } while (0)

/* ── loads and stores address individual halves ───────────────────────────── */

/*
 * Build a double out of two LWC1s — the low half into $f0, the high half into
 * $f1 — and read it back with SDC1, which sees the pair. This is exactly what
 * an o32 compiler emits to load a double from a pair of words.
 */
static void t_fr0_lwc1_halves_form_a_double(void)
{
    w()[0] = 0x89ABCDEFu;      /* low  word */
    w()[1] = 0x01234567u;      /* high word */
    d()[1] = 0;
    SYNC();

    WITH_FR0({
        __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                                "lwc1 $f1, 4(%0)\n\t"
                                "sdc1 $f0, 8(%0)" Z :: "r"(w()) : "memory");
    });
    SYNC();
    CHECK_EQ(d()[1], 0x0123456789ABCDEFull);
}

/* And the reverse: SDC1 writes the pair, SWC1 reads each half back out. */
static void t_fr0_swc1_stores_each_half(void)
{
    d()[0] = 0x1122334455667788ull;
    w()[4] = 0; w()[5] = 0;
    SYNC();

    WITH_FR0({
        __asm__ __volatile__(AF "ldc1 $f0, 0(%0)\n\t"
                                "swc1 $f0, 16(%0)\n\t"
                                "swc1 $f1, 20(%0)" Z :: "r"(w()) : "memory");
    });
    SYNC();
    CHECK_EQ(w()[4], 0x55667788u);      /* $f0 is the low word  */
    CHECK_EQ(w()[5], 0x11223344u);      /* $f1 is the high word */
}

/* MTC1 and MFC1 address the same halves as LWC1 and SWC1. */
static void t_fr0_mtc1_mfc1_halves(void)
{
    u64 lo = 0, hi = 0;

    d()[1] = 0;
    SYNC();
    WITH_FR0({
        __asm__ __volatile__(AF "mtc1 %0, $f0\n\t"
                                "mtc1 %1, $f1\n\t"
                                "nop\n\t"
                                "sdc1 $f0, 8(%2)" Z
                             :: "r"(0xDEADBEEFull), "r"(0xFEEDFACEull), "r"(w())
                             : "memory");
        SYNC();
        __asm__ __volatile__(AF "ldc1 $f2, 8(%2)\n\t"
                                "mfc1 %0, $f2\n\t"
                                "mfc1 %1, $f3" Z
                             : "=r"(lo), "=r"(hi) : "r"(w()));
    });
    CHECK_EQ(d()[1], 0xFEEDFACEDEADBEEFull);
    CHECK_EQ(lo & 0xFFFFFFFFull, 0xDEADBEEFull);
    CHECK_EQ(hi & 0xFFFFFFFFull, 0xFEEDFACEull);
}

/* ── arithmetic in the paired mode ────────────────────────────────────────── */

/* Double arithmetic on even registers is the ordinary case for o32 code, and
 * must work identically to FR=1. */
static void t_fr0_double_arithmetic(void)
{
    d()[0] = D_1; d()[1] = D_2; d()[2] = 0;
    SYNC();
    fcsr_reset();
    exc_clear();

    WITH_FR0({
        __asm__ __volatile__(AF "ldc1 $f0, 0(%0)\n\t"
                                "ldc1 $f2, 8(%0)\n\t"
                                "add.d $f4, $f0, $f2\n\t"
                                "sdc1 $f4, 16(%0)" Z :: "r"(d()) : "memory");
    });
    SYNC();
    CHECK_EQ(d()[2], D_3);
    CHECK_NO_EXC();
}

/* Single-precision arithmetic on *even* registers is equally well defined:
 * "S, W: value <- FGR[fpr]" for an even specifier. */
static void t_fr0_single_arithmetic_on_even_registers(void)
{
    w()[0] = F_2; w()[1] = F_3; w()[2] = 0;
    SYNC();
    fcsr_reset();
    exc_clear();

    WITH_FR0({
        __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                                "lwc1 $f2, 4(%0)\n\t"
                                "add.s $f4, $f0, $f2\n\t"
                                "swc1 $f4, 8(%0)" Z :: "r"(w()) : "memory");
    });
    SYNC();
    CHECK_EQ(w()[2], F_5);
    CHECK_NO_EXC();
}

/*
 * An odd register in a format operation is "undefined result for odd 32-bit
 * reg #s" — the manual's words. Nothing is asserted about the answer; what is
 * reported is whether the machine produced one at all, and whether it faulted.
 * A test that demanded a particular result here would be inventing a
 * requirement the architecture explicitly declines to make.
 */
static void t_fr0_odd_register_single_is_undefined(void)
{
    u32 got;

    w()[0] = F_2; w()[1] = F_3; w()[2] = 0;
    SYNC();
    fcsr_reset();
    exc_clear();

    WITH_FR0({
        __asm__ __volatile__(AF "lwc1 $f1, 0(%0)\n\t"
                                "lwc1 $f3, 4(%0)\n\t"
                                "add.s $f5, $f1, $f3\n\t"
                                "swc1 $f5, 8(%0)" Z :: "r"(w()) : "memory");
    });
    SYNC();
    got = w()[2];
    con_printf("\n      [FR=0 add.s on odd registers: result=%x exc=%u]",
               got, exc.count);
    CHECK(1);
}

/* ── switching modes ──────────────────────────────────────────────────────── */

/*
 * What the pair view shows after FR=1 code wrote the two registers
 * independently. The architecture does not define this — with FR=0 "the 32
 * General Purpose registers of the FPU are 32-bits wide", so what became of
 * the upper halves is not something software may rely on — but a kernel that
 * switches FR while context-switching cares a great deal about what the
 * hardware actually does, so it is worth reporting.
 */
static void t_fr_mode_switch_aliasing(void)
{
    d()[1] = 0;
    SYNC();

    __asm__ __volatile__(AF "dmtc1 %0, $f0\n\t"
                            "dmtc1 %1, $f1\n\t"
                            "nop" Z
                         :: "r"(0x1111111122222222ull), "r"(0x3333333344444444ull));
    WITH_FR0({
        __asm__ __volatile__(AF "sdc1 $f0, 8(%0)" Z :: "r"(w()) : "memory");
    });
    SYNC();
    con_printf("\n      [FR=1 $f0=1111111122222222 $f1=3333333344444444"
               " -> FR=0 pair reads %lx]", d()[1]);
    CHECK(1);
}

/*
 * The FR bit itself: writable, readable, and it does not disturb the rest of
 * Status on the way through. This is the precondition for everything above,
 * and cheap insurance against a Status write mask that swallows bit 26.
 */
static void t_fr_bit_is_writable(void)
{
    u32 saved = cp0_status();

    cp0_status_set(saved & ~ST_FR);
    CHECK_EQ(cp0_status() & ST_FR, 0u);
    CHECK_EQ(cp0_status() & (ST_CU1 | ST_KX), (u32)(ST_CU1 | ST_KX));

    cp0_status_set(saved | ST_FR);
    CHECK_EQ(cp0_status() & ST_FR, (u32)ST_FR);
    CHECK_EQ(cp0_status(), saved);
}

static const struct test tests[] = {
    TEST("fpu/fr0_lwc1_halves",     t_fr0_lwc1_halves_form_a_double,        CPU_ALL),
    TEST("fpu/fr0_swc1_halves",     t_fr0_swc1_stores_each_half,            CPU_ALL),
    TEST("fpu/fr0_mtc1_mfc1",       t_fr0_mtc1_mfc1_halves,                 CPU_ALL),
    TEST("fpu/fr0_double_arith",    t_fr0_double_arithmetic,                CPU_ALL),
    TEST("fpu/fr0_single_arith",    t_fr0_single_arithmetic_on_even_registers, CPU_ALL),
    TEST("fpu/fr0_odd_undefined",   t_fr0_odd_register_single_is_undefined, CPU_ALL),
    TEST("fpu/fr_mode_switch",      t_fr_mode_switch_aliasing,              CPU_ALL),
    TEST("fpu/fr_bit_writable",     t_fr_bit_is_writable,                   CPU_ALL),
};

const struct test_group group_fpu_fr0 = {
    "fpu_fr0", tests, sizeof(tests) / sizeof(tests[0])
};
