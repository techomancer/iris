/* mips4_fp — the MIPS IV floating-point additions, and the R4400's obligation
 * to refuse them.
 *
 * Same contract as mips4.c, which this file extends: every test runs on BOTH
 * CPUs and branches internally — the instruction must compute on an R5000 and
 * raise Reserved Instruction on an R4400. mips4.c reached one instruction from
 * each family (RECIP.S, LWXC1, MADD.S, the GPR form of MOVF/MOVT); this covers
 * the rest of them, which is where an emulator's decode table is most likely to
 * have a hole:
 *
 *   RECIP.D / RSQRT.D        the double-precision half of the reciprocals
 *   LDXC1 / SWXC1 / SDXC1    the indexed loads and stores other than LWXC1
 *   PREFX                    the indexed prefetch
 *   MSUB / NMADD / NMSUB     the three multiply-adds other than MADD, in both
 *                            formats, plus MADD.D
 *   MOVF/MOVT/MOVN/MOVZ.fmt  the FP conditional moves — a different instruction
 *                            from the GPR MOVCI that mips4.c covers
 *
 * The R4400 encodings below are `.word` literals because GAS will not emit a
 * MIPS IV mnemonic under -march=mips3, and switching the whole file to mips4
 * would let one slip into code that runs on both parts. Every literal was read
 * back out of the assembler rather than derived by hand — findings.md records
 * what happens when a hand-written encoder and a hand-written decoder share the
 * same mistake and cancel each other out.
 */

#include "testlib.h"
#include "cp0.h"

#define A ".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"
#define Z "\n\t.set pop"
#define A4 ".set push; .set mips4; .set noreorder; .set nomacro; .set noat\n\t" \
           ".set hardfloat\n\t"

extern char _scratch_start[];
static volatile u32 *w(void) { return (volatile u32 *)_scratch_start; }
static volatile u64 *dw(void) { return (volatile u64 *)_scratch_start; }

/* Run a literal encoding and assert Reserved Instruction. */
#define CHECK_RI(word_literal)                                             \
    do {                                                                   \
        exc_clear();                                                       \
        __asm__ __volatile__(A ".word " word_literal Z);                   \
        CHECK_EQ(exc.count, 1u);                                           \
        CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_RI);                       \
    } while (0)

/*
 * The same, for the COP1X encodings, which name $13 as base and $14 as index:
 * point them at the scratch area first. Without that the literal runs — on a
 * CPU that wrongly accepts it — against whatever the compiler happened to
 * leave in those registers, and reports an address error instead of executing
 * cleanly. Both outcomes fail the test, but only one of them says plainly
 * what went wrong.
 */
#define CHECK_RI_INDEXED(word_literal)                                     \
    do {                                                                   \
        exc_clear();                                                       \
        __asm__ __volatile__(A "daddu $13, $zero, %0\n\t"                  \
                               "daddu $14, $zero, $zero\n\t"              \
                               ".word " word_literal Z                     \
                             :: "r"(dw()) : "$13", "$14", "memory");       \
        CHECK_EQ(exc.count, 1u);                                           \
        CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_RI);                       \
    } while (0)

/* Single-precision bit patterns used below. */
#define F_1        0x3F800000u
#define F_NEG1     0xBF800000u
#define F_2        0x40000000u
#define F_3        0x40400000u
#define F_5        0x40A00000u
#define F_11       0x41300000u
#define F_NEG11    0xC1300000u
#define F_SENTINEL 0x5A5A5A5Au

/* ── RECIP.D / RSQRT.D ────────────────────────────────────────────────────── */

/*
 * recip.d $f2, $f0   = 0x46200095    (COP1, fmt=D, funct 0x15)
 * rsqrt.d $f2, $f0   = 0x46200096
 *
 * 1/4 and 1/sqrt(4) are exact in binary, which matters: MIPS IV leaves the
 * accuracy of these two instructions implementation-dependent, so a test that
 * used a value needing rounding would be asserting something the architecture
 * does not require.
 */
static void t_recip_rsqrt_double(void)
{
    if (is_r4400()) {
        CHECK_RI("0x46200095");
        CHECK_RI("0x46200096");
        return;
    }

    dw()[0] = 0x4010000000000000ull;        /* 4.0 */
    dw()[1] = 0; dw()[2] = 0;
    SYNC();
    exc_clear();
    __asm__ __volatile__(A4 "ldc1 $f0, 0(%0)\n\t"
                            "recip.d $f2, $f0\n\t"
                            "sdc1 $f2, 8(%0)\n\t"
                            "rsqrt.d $f4, $f0\n\t"
                            "sdc1 $f4, 16(%0)" Z :: "r"(dw()) : "memory");
    SYNC();
    CHECK_NO_EXC();
    CHECK_EQ(dw()[1], 0x3FD0000000000000ull);   /* 0.25 */
    CHECK_EQ(dw()[2], 0x3FE0000000000000ull);   /* 0.5  */
}

/* ── COP1X: the indexed loads and stores ──────────────────────────────────── */

/*
 * With base=$13 and index=$14:
 *   ldxc1 $f0, $14($13)  = 0x4DAE0001
 *   swxc1 $f0, $14($13)  = 0x4DAE0008
 *   sdxc1 $f0, $14($13)  = 0x4DAE0009
 *   prefx  0, $14($13)   = 0x4DAE000F
 *
 * The base goes in as a pointer, never as an integer: under n32 `unsigned long`
 * is 32 bits, so casting a KSEG0 pointer through one zero-extends it into
 * xkuseg and the access takes a TLB miss that reads like a broken instruction.
 * See docs/gotchas.md.
 */
static void t_cop1x_indexed_memory(void)
{
    if (is_r4400()) {
        CHECK_RI_INDEXED("0x4DAE0001");
        CHECK_RI_INDEXED("0x4DAE0008");
        CHECK_RI_INDEXED("0x4DAE0009");
        CHECK_RI_INDEXED("0x4DAE000F");
        return;
    }
    {
        u64 index8 = 8ull, index16 = 16ull;

        dw()[0] = 0x0123456789ABCDEFull;
        dw()[1] = 0x1122334455667788ull;
        dw()[2] = 0; dw()[3] = 0;
        SYNC();
        exc_clear();

        /* LDXC1 loads dw[1] via base+8, SDXC1 stores it to dw[2]. */
        __asm__ __volatile__(A4 "ldxc1 $f0, %1(%0)\n\t"
                                "sdxc1 $f0, %2(%0)" Z
                             :: "r"(dw()), "r"(index8), "r"(index16) : "memory");
        SYNC();
        CHECK_NO_EXC();
        CHECK_EQ(dw()[2], 0x1122334455667788ull);

        /* SWXC1 stores the low word of $f0 — big-endian, so the *second* word
         * of the doubleword just loaded. */
        w()[6] = 0;
        SYNC();
        {
            u64 index24 = 24ull;
            __asm__ __volatile__(A4 "swxc1 $f0, %1(%0)" Z
                                 :: "r"(dw()), "r"(index24) : "memory");
        }
        SYNC();
        CHECK_NO_EXC();
        CHECK_EQ(w()[6], 0x55667788u);

        /* PREFX is a hint: it must not fault and must change nothing. */
        __asm__ __volatile__(A4 "prefx 0, %1(%0)\n\t"
                                "prefx 1, %1(%0)" Z
                             :: "r"(dw()), "r"(index8) : "memory");
        SYNC();
        CHECK_NO_EXC();
        CHECK_EQ(dw()[1], 0x1122334455667788ull);
    }
}

/* ── COP1X: the rest of the multiply-add family ───────────────────────────── */

/*
 * With fd=$f6, fr=$f4, fs=$f0, ft=$f2:
 *   msub.s  = 0x4C8201A8      nmadd.s = 0x4C8201B0     nmsub.s = 0x4C8201B8
 *   madd.d  = 0x4C8201A1      msub.d  = 0x4C8201A9
 *   nmadd.d = 0x4C8201B1      nmsub.d = 0x4C8201B9
 *
 * fs*ft is 2*3 and fr is 5, all exact, so the four results are exact too and
 * the test says nothing about whether the intermediate product is rounded —
 * which is just as well, since that is implementation-dependent.
 *
 *   MADD   fd =  fs*ft + fr =  11      MSUB   fd =  fs*ft - fr =  1
 *   NMADD  fd = -(fs*ft + fr) = -11    NMSUB  fd = -(fs*ft - fr) = -1
 */
static void t_multiply_add_family_single(void)
{
    if (is_r4400()) {
        CHECK_RI("0x4C8201A8");
        CHECK_RI("0x4C8201B0");
        CHECK_RI("0x4C8201B8");
        return;
    }

    w()[0] = F_2; w()[1] = F_3; w()[2] = F_5;
    w()[4] = 0; w()[5] = 0; w()[6] = 0;
    SYNC();
    exc_clear();
    __asm__ __volatile__(A4 "lwc1 $f0, 0(%0)\n\t"
                            "lwc1 $f2, 4(%0)\n\t"
                            "lwc1 $f4, 8(%0)\n\t"
                            "msub.s $f6, $f4, $f0, $f2\n\t"
                            "swc1 $f6, 16(%0)\n\t"
                            "nmadd.s $f6, $f4, $f0, $f2\n\t"
                            "swc1 $f6, 20(%0)\n\t"
                            "nmsub.s $f6, $f4, $f0, $f2\n\t"
                            "swc1 $f6, 24(%0)" Z :: "r"(w()) : "memory");
    SYNC();
    CHECK_NO_EXC();
    CHECK_EQ(w()[4], F_1);          /* 2*3 - 5 */
    CHECK_EQ(w()[5], F_NEG11);      /* -(2*3 + 5) */
    CHECK_EQ(w()[6], F_NEG1);       /* -(2*3 - 5) */
}

static void t_multiply_add_family_double(void)
{
    if (is_r4400()) {
        CHECK_RI("0x4C8201A1");
        CHECK_RI("0x4C8201A9");
        CHECK_RI("0x4C8201B1");
        CHECK_RI("0x4C8201B9");
        return;
    }

    dw()[0] = 0x4000000000000000ull;    /* 2.0 */
    dw()[1] = 0x4008000000000000ull;    /* 3.0 */
    dw()[2] = 0x4014000000000000ull;    /* 5.0 */
    dw()[4] = 0; dw()[5] = 0; dw()[6] = 0; dw()[7] = 0;
    SYNC();
    exc_clear();
    __asm__ __volatile__(A4 "ldc1 $f0, 0(%0)\n\t"
                            "ldc1 $f2, 8(%0)\n\t"
                            "ldc1 $f4, 16(%0)\n\t"
                            "madd.d $f6, $f4, $f0, $f2\n\t"
                            "sdc1 $f6, 32(%0)\n\t"
                            "msub.d $f6, $f4, $f0, $f2\n\t"
                            "sdc1 $f6, 40(%0)\n\t"
                            "nmadd.d $f6, $f4, $f0, $f2\n\t"
                            "sdc1 $f6, 48(%0)\n\t"
                            "nmsub.d $f6, $f4, $f0, $f2\n\t"
                            "sdc1 $f6, 56(%0)" Z :: "r"(dw()) : "memory");
    SYNC();
    CHECK_NO_EXC();
    CHECK_EQ(dw()[4], 0x4026000000000000ull);   /*  11.0 */
    CHECK_EQ(dw()[5], 0x3FF0000000000000ull);   /*   1.0 */
    CHECK_EQ(dw()[6], 0xC026000000000000ull);   /* -11.0 */
    CHECK_EQ(dw()[7], 0xBFF0000000000000ull);   /*  -1.0 */
}

/* ── the FP conditional moves ─────────────────────────────────────────────── */

/*
 * A different instruction from the MOVF/MOVT that mips4.c covers: those move
 * between *GPRs* under FP-condition control (MOVCI, in SPECIAL), these move
 * between FP registers (COP1 funct 0x11/0x12/0x13).
 *
 *   movt.s $f4, $f0, $fcc0 = 0x46010111    movf.s = 0x46000111
 *   movn.s $f4, $f0, $13   = 0x460D0113    movz.s = 0x460D0112
 *
 * "Conditional" means the destination is left ALONE when the condition fails,
 * so each case below preloads $f4 with a sentinel and checks for it.
 */
static void t_fp_conditional_moves_single(void)
{
    if (is_r4400()) {
        CHECK_RI("0x46010111");
        CHECK_RI("0x46000111");
        CHECK_RI("0x460D0113");
        CHECK_RI("0x460D0112");
        return;
    }
    {
        u64 nonzero = 1ull, zero = 0ull;

        w()[0] = F_3; w()[1] = F_SENTINEL;
        SYNC();
        exc_clear();

        /* Condition true: MOVT moves, MOVF does not. */
        fcsr_set(fcsr() | FCSR_CC0);
        __asm__ __volatile__(A4 "lwc1 $f0, 0(%0)\n\t"
                                "lwc1 $f4, 4(%0)\n\t"
                                "movt.s $f4, $f0, $fcc0\n\t"
                                "swc1 $f4, 8(%0)\n\t"
                                "lwc1 $f4, 4(%0)\n\t"
                                "movf.s $f4, $f0, $fcc0\n\t"
                                "swc1 $f4, 12(%0)" Z :: "r"(w()) : "memory");
        SYNC();
        CHECK_EQ(w()[2], F_3);
        CHECK_EQ(w()[3], F_SENTINEL);

        /* Condition false: the senses reverse. */
        fcsr_set(fcsr() & ~FCSR_CC0);
        __asm__ __volatile__(A4 "lwc1 $f0, 0(%0)\n\t"
                                "lwc1 $f4, 4(%0)\n\t"
                                "movt.s $f4, $f0, $fcc0\n\t"
                                "swc1 $f4, 8(%0)\n\t"
                                "lwc1 $f4, 4(%0)\n\t"
                                "movf.s $f4, $f0, $fcc0\n\t"
                                "swc1 $f4, 12(%0)" Z :: "r"(w()) : "memory");
        SYNC();
        CHECK_EQ(w()[2], F_SENTINEL);
        CHECK_EQ(w()[3], F_3);

        /* MOVN moves when the GPR is non-zero, MOVZ when it is zero. */
        __asm__ __volatile__(A4 "lwc1 $f0, 0(%0)\n\t"
                                "lwc1 $f4, 4(%0)\n\t"
                                "movn.s $f4, $f0, %1\n\t"
                                "swc1 $f4, 8(%0)\n\t"
                                "lwc1 $f4, 4(%0)\n\t"
                                "movn.s $f4, $f0, %2\n\t"
                                "swc1 $f4, 12(%0)" Z
                             :: "r"(w()), "r"(nonzero), "r"(zero) : "memory");
        SYNC();
        CHECK_EQ(w()[2], F_3);
        CHECK_EQ(w()[3], F_SENTINEL);

        __asm__ __volatile__(A4 "lwc1 $f0, 0(%0)\n\t"
                                "lwc1 $f4, 4(%0)\n\t"
                                "movz.s $f4, $f0, %1\n\t"
                                "swc1 $f4, 8(%0)\n\t"
                                "lwc1 $f4, 4(%0)\n\t"
                                "movz.s $f4, $f0, %2\n\t"
                                "swc1 $f4, 12(%0)" Z
                             :: "r"(w()), "r"(zero), "r"(nonzero) : "memory");
        SYNC();
        CHECK_EQ(w()[2], F_3);
        CHECK_EQ(w()[3], F_SENTINEL);
        CHECK_NO_EXC();
        fcsr_set(0);
    }
}

/*
 * And in double precision, where a wrong format field would move 32 bits of a
 * 64-bit value and leave the rest of the register behind.
 *
 *   movt.d $f4, $f0, $fcc0 = 0x46210111    movn.d $f4, $f0, $13 = 0x462D0113
 */
static void t_fp_conditional_moves_double(void)
{
    if (is_r4400()) {
        CHECK_RI("0x46210111");
        CHECK_RI("0x462D0113");
        return;
    }
    {
        u64 nonzero = 1ull;

        dw()[0] = 0x0123456789ABCDEFull;
        dw()[1] = 0x5A5A5A5A5A5A5A5Aull;
        dw()[2] = 0;
        SYNC();
        exc_clear();

        fcsr_set(fcsr() | FCSR_CC0);
        __asm__ __volatile__(A4 "ldc1 $f0, 0(%0)\n\t"
                                "ldc1 $f4, 8(%0)\n\t"
                                "movt.d $f4, $f0, $fcc0\n\t"
                                "sdc1 $f4, 16(%0)" Z :: "r"(dw()) : "memory");
        SYNC();
        CHECK_EQ(dw()[2], 0x0123456789ABCDEFull);   /* all 64 bits moved */

        fcsr_set(fcsr() & ~FCSR_CC0);
        __asm__ __volatile__(A4 "ldc1 $f0, 0(%0)\n\t"
                                "ldc1 $f4, 8(%0)\n\t"
                                "movt.d $f4, $f0, $fcc0\n\t"
                                "sdc1 $f4, 16(%0)" Z :: "r"(dw()) : "memory");
        SYNC();
        CHECK_EQ(dw()[2], 0x5A5A5A5A5A5A5A5Aull);   /* untouched */

        __asm__ __volatile__(A4 "ldc1 $f0, 0(%0)\n\t"
                                "ldc1 $f4, 8(%0)\n\t"
                                "movn.d $f4, $f0, %1\n\t"
                                "sdc1 $f4, 16(%0)" Z
                             :: "r"(dw()), "r"(nonzero) : "memory");
        SYNC();
        CHECK_EQ(dw()[2], 0x0123456789ABCDEFull);
        CHECK_NO_EXC();
        fcsr_set(0);
    }
}

static const struct test tests[] = {
    TEST("mips4/recip_rsqrt_d",   t_recip_rsqrt_double,           CPU_ALL),
    TEST("mips4/cop1x_memory",    t_cop1x_indexed_memory,         CPU_ALL),
    TEST("mips4/madd_family_s",   t_multiply_add_family_single,   CPU_ALL),
    TEST("mips4/madd_family_d",   t_multiply_add_family_double,   CPU_ALL),
    TEST("mips4/fp_cond_move_s",  t_fp_conditional_moves_single,  CPU_ALL),
    TEST("mips4/fp_cond_move_d",  t_fp_conditional_moves_double,  CPU_ALL),
};

const struct test_group group_mips4_fp = {
    "mips4_fp", tests, sizeof(tests) / sizeof(tests[0])
};
