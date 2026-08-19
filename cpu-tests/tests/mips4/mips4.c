/* mips4 — the MIPS IV additions, and the R4400's obligation to refuse them.
 *
 * This is the group that justifies running the same binary on both CPUs. Every
 * test here is registered for BOTH, and branches internally:
 *
 *   on R5000 the instruction must compute the right answer;
 *   on R4400 the very same encoding must raise Reserved Instruction.
 *
 * An emulator that implements MIPS IV unconditionally passes the first half
 * and fails the second, and nothing else in the suite would notice.
 *
 * The encodings are written as `.word` in the R4400 direction, because the
 * assembler will not emit a MIPS IV mnemonic under -march=mips3 — and switching
 * the assembler to mips4 for the whole file would let a MIPS IV instruction
 * slip into code that runs on both.
 */

#include "testlib.h"
#include "cp0.h"

#define A ".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"
#define Z "\n\t.set pop"
/* MIPS IV, with the FPU enabled: needed for the R5000 half of each test. */
#define A4 ".set push; .set mips4; .set noreorder; .set nomacro; .set noat\n\t" \
           ".set hardfloat\n\t"

#define OPAQUE(x) ({ __typeof__(x) __v = (x); __asm__ __volatile__("" : "+r"(__v)); __v; })

/*
 * FP registers are never listed in a clobber list here. Under -msoft-float GCC
 * refuses them outright ("the register '$f0' cannot be clobbered in 'asm' for
 * the current target") — and it does not need them: a soft-float compilation
 * never allocates an FP register, so an asm block may use $f0..$f31 freely
 * without telling the compiler. Values still cross the boundary through memory
 * rather than through FP registers, so nothing depends on this.
 */

/* Run a literal encoding and assert Reserved Instruction. The encoding has to
 * be a literal because `.word` needs a compile-time constant — a
 * runtime-selected encoding would take self-modifying code. */
/* Print whatever the last exception was — used when a test that expects no
 * exception gets one, so the log says which. */
static void report_exc(const char *what)
{
    if (exc.count == 0) return;
    con_printf("\n      [%s: %u exception(s), last ExcCode=%u vector=%u"
               " EPC=%lx BadVAddr=%lx]",
               what, exc.count, CAUSE_EXC(exc.cause), exc.vector,
               exc.epc, exc.badvaddr);
}

#define CHECK_RI(word_literal)                                             \
    do {                                                                   \
        exc_clear();                                                       \
        __asm__ __volatile__(A ".word " word_literal Z);                   \
        CHECK_EQ(exc.count, 1u);                                           \
        CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_RI);                       \
    } while (0)

/* ── conditional moves on GPRs: MOVN / MOVZ ───────────────────────────────── */

/*
 * MOVN rd, rs, rt  — SPECIAL function 0x0B: rd <- rs if rt != 0
 * MOVZ rd, rs, rt  — SPECIAL function 0x0A: rd <- rs if rt == 0
 *
 * Encodings below use rd=$12, rs=$13, rt=$14:
 *   SPECIAL(0) rs=13 rt=14 rd=12 shamt=0 funct
 *   = (13 << 21) | (14 << 16) | (12 << 11) | funct
 *   = 0x01AE6000 | funct
 */
static void t_movn_movz(void)
{
    if (is_r4400()) {
        CHECK_RI("0x01AE600B");     /* movn $12, $13, $14 */
        CHECK_RI("0x01AE600A");     /* movz $12, $13, $14 */
        return;
    }

    {
        u64 rd;
        u64 src = OPAQUE(0x1234ull), nz = OPAQUE(1ull), z = OPAQUE(0ull);

        /* MOVN with a non-zero condition moves. */
        __asm__ __volatile__(A4 "daddiu %0, $zero, 0\n\t"
                                "movn %0, %1, %2" Z
                             : "=&r"(rd) : "r"(src), "r"(nz));
        CHECK_EQ(rd, 0x1234ull);

        /* MOVN with a zero condition does not. */
        __asm__ __volatile__(A4 "daddiu %0, $zero, 0\n\t"
                                "movn %0, %1, %2" Z
                             : "=&r"(rd) : "r"(src), "r"(z));
        CHECK_EQ(rd, 0ull);

        /* MOVZ is the mirror image. */
        __asm__ __volatile__(A4 "daddiu %0, $zero, 0\n\t"
                                "movz %0, %1, %2" Z
                             : "=&r"(rd) : "r"(src), "r"(z));
        CHECK_EQ(rd, 0x1234ull);

        __asm__ __volatile__(A4 "daddiu %0, $zero, 0\n\t"
                                "movz %0, %1, %2" Z
                             : "=&r"(rd) : "r"(src), "r"(nz));
        CHECK_EQ(rd, 0ull);
    }
}

/* ── PREF ─────────────────────────────────────────────────────────────────── */

/*
 * PREF hint, offset(base) — primary opcode 0x33. Architecturally a hint: it
 * must not fault and must not change any register. On R4400 the same encoding
 * is a reserved opcode.
 *
 *   0x33 << 26 | base=13 | hint=0 | offset=0 = 0xCDA00000... computed below.
 *   base $13 -> (13 << 21) = 0x01A00000; opcode 0x33 << 26 = 0xCC000000
 *   => 0xCDA00000
 */
static void t_pref(void)
{
    if (is_r4400()) {
        CHECK_RI("0xCDA00000");
        return;
    }
    {
        extern char _scratch_start[];
        volatile u32 *p = (volatile u32 *)_scratch_start;
        exc_clear();
        __asm__ __volatile__(A4 "pref 0, 0(%0)\n\t"
                                "pref 1, 0(%0)\n\t"
                                "pref 4, 0(%0)" Z :: "r"(p) : "memory");
        CHECK_NO_EXC();
    }
}

/* ── floating-point MIPS IV ───────────────────────────────────────────────── */

/*
 * RECIP.S fd, fs — COP1(0x11) fmt=S(16) funct=0x15
 *   = (0x11 << 26) | (16 << 21) | (0 << 16) | (fs << 11) | (fd << 6) | 0x15
 * With fs=$f0, fd=$f2: 0x46000095... computed: 0x44000000 | (16<<21)=0x02000000
 *   -> 0x46000000, fd=2 -> (2 << 6) = 0x80, funct 0x15
 *   => 0x46000095
 */
static void t_recip_rsqrt(void)
{
    if (is_r4400()) {
        CHECK_RI("0x46000095");     /* recip.s $f2, $f0 */
        CHECK_RI("0x46000096");     /* rsqrt.s $f2, $f0 */
        return;
    }
    {
        /* 1/4 = 0.25, and 1/sqrt(4) = 0.5. Values go in and out through
         * memory so no FP calling convention is involved. */
        extern char _scratch_start[];
        volatile float *f = (volatile float *)_scratch_start;
        f[0] = 4.0f;
        SYNC();
        exc_clear();
        __asm__ __volatile__(A4 "lwc1 $f0, 0(%0)\n\t"
                                "recip.s $f2, $f0\n\t"
                                "swc1 $f2, 4(%0)\n\t"
                                "rsqrt.s $f4, $f0\n\t"
                                "swc1 $f4, 8(%0)" Z
                             :: "r"(f) : "memory");
        SYNC();
        CHECK_NO_EXC();
        CHECK_EQ(*(volatile u32 *)&f[1], 0x3E800000u);   /* 0.25f */
        CHECK_EQ(*(volatile u32 *)&f[2], 0x3F000000u);   /* 0.5f  */
    }
}

/*
 * MOVF/MOVT on GPRs (MOVCI, SPECIAL funct 0x01) and the FP conditional moves
 * MOVF.fmt / MOVT.fmt / MOVN.fmt / MOVZ.fmt.
 *
 * MOVCI encoding: SPECIAL(0), rs, cc/tf in rt, rd, funct 0x01.
 *   rs=$13, rd=$12, cc=0, tf=1 (MOVT) -> rt field = (cc << 2) | tf = 1
 *   = (13 << 21) | (1 << 16) | (12 << 11) | 0x01 = 0x01A16001
 *   tf=0 (MOVF) -> rt = 0 -> 0x01A06001
 */
static void t_movci(void)
{
    if (is_r4400()) {
        CHECK_RI("0x01A16001");     /* movt $12, $13, $fcc0 */
        CHECK_RI("0x01A06001");     /* movf $12, $13, $fcc0 */
        return;
    }
    {
        u64 rd;
        u64 src = OPAQUE(0xABCDull);

        /* Set FP condition code 0, then MOVT must move and MOVF must not. */
        fcsr_set(fcsr() | FCSR_CC0);
        __asm__ __volatile__(A4 "daddiu %0, $zero, 0\n\t"
                                "movt %0, %1, $fcc0" Z
                             : "=&r"(rd) : "r"(src));
        CHECK_EQ(rd, 0xABCDull);

        __asm__ __volatile__(A4 "daddiu %0, $zero, 0\n\t"
                                "movf %0, %1, $fcc0" Z
                             : "=&r"(rd) : "r"(src));
        CHECK_EQ(rd, 0ull);

        /* Clear it and the sense reverses. */
        fcsr_set(fcsr() & ~FCSR_CC0);
        __asm__ __volatile__(A4 "daddiu %0, $zero, 0\n\t"
                                "movf %0, %1, $fcc0" Z
                             : "=&r"(rd) : "r"(src));
        CHECK_EQ(rd, 0xABCDull);
    }
}

/*
 * COP1X (opcode 0x13): indexed FP loads/stores and the fused multiply-adds.
 *
 * LWXC1 fd, index(base) — COP1X funct 0x00
 *   = (0x13 << 26) | (base << 21) | (index << 16) | (fd << 6) | 0x00
 * With base=$13, index=$14, fd=$f0: 0x4DAE0000
 *
 * MADD.S fd, fr, fs, ft — COP1X funct 0x20 | fmt
 *   = (0x13 << 26) | (fr << 21) | (ft << 16) | (fs << 11) | (fd << 6) | 0x20
 * With all registers 0: 0x4C000020
 */
static void t_cop1x(void)
{
    if (is_r4400()) {
        CHECK_RI("0x4DAE0000");     /* lwxc1  $f0, $14($13) */
        CHECK_RI("0x4C000020");     /* madd.s $f0, $f0, $f0, $f0 */
        return;
    }
    {
        extern char _scratch_start[];
        volatile float *f = (volatile float *)_scratch_start;
        /*
         * The base goes in as the POINTER, not as `(u64)(unsigned long)f`.
         * Under n32 `unsigned long` is 32 bits, so that cast truncates and
         * then ZERO-extends: 0xffffffff88228000 becomes 0x0000000088228000,
         * which is xkuseg rather than KSEG0. The first version of this test
         * failed with a TLB-miss-on-store at BadVAddr=0x000000008822800c,
         * which reads like a broken LWXC1 and is really a broken cast.
         * Pointers are sign-extended by the ABI, so passing one is correct
         * by construction. See docs/gotchas.md.
         */
        u64 index = OPAQUE(4ull);

        f[0] = 2.0f; f[1] = 3.0f; f[2] = 5.0f;
        SYNC();
        exc_clear();

        /* LWXC1 loads f[1] = 3.0 via base+index. */
        __asm__ __volatile__(A4 "lwxc1 $f0, %1(%0)\n\t"
                                "swc1 $f0, 12(%0)" Z
                             :: "r"(f), "r"(index) : "memory");
        SYNC();
        report_exc("lwxc1");
        CHECK_NO_EXC();
        CHECK_EQ(*(volatile u32 *)&f[3], 0x40400000u);   /* 3.0f */

        /* MADD.S fd = fr + fs*ft = 5.0 + 2.0*3.0 = 11.0 */
        __asm__ __volatile__(A4 "lwc1 $f0, 0(%0)\n\t"     /* 2.0 */
                                "lwc1 $f2, 4(%0)\n\t"     /* 3.0 */
                                "lwc1 $f4, 8(%0)\n\t"     /* 5.0 */
                                "madd.s $f6, $f4, $f0, $f2\n\t"
                                "swc1 $f6, 12(%0)" Z
                             :: "r"(f) : "memory");
        SYNC();
        report_exc("madd.s");
        CHECK_NO_EXC();
        CHECK_EQ(*(volatile u32 *)&f[3], 0x41300000u);   /* 11.0f */
    }
}

/*
 * MIPS IV widens the FP condition to eight codes (CC0..CC7) rather than the
 * single bit MIPS III has. C.cond.fmt gains a cc field, and BC1T/BC1F can
 * select which one to test.
 *
 * C.EQ.S with cc=3: COP1 fmt=S, funct = 0x32 (C.EQ), cc in bits 10:8
 *   = 0x46000000 | (ft << 16) | (fs << 11) | (cc << 8) | 0x32
 * With fs=ft=$f0, cc=3: 0x46000332
 */
static void t_multiple_fp_condition_codes(void)
{
    if (is_r4400()) {
        /* On MIPS III the cc field is reserved-zero, so a non-zero cc is not
         * an architecturally-defined encoding. R4400 behaviour here is not
         * pinned by the manual — it may ignore the field rather than trap —
         * so this is reported, not asserted. */
        exc_clear();
        __asm__ __volatile__(A ".word 0x46000332" Z);
        con_printf("\n      [c.eq.s cc=3 on R4400: exc=%u code=%u]",
                   exc.count, exc.count ? CAUSE_EXC(exc.cause) : 0);
        CHECK(1);
        return;
    }
    {
        extern char _scratch_start[];
        volatile float *f = (volatile float *)_scratch_start;
        u32 fc;
        f[0] = 1.0f; f[1] = 1.0f; f[2] = 2.0f;
        SYNC();
        exc_clear();

        /* Equal into cc3, unequal into cc5; both must be independently
         * visible in FCSR. */
        __asm__ __volatile__(A4 "lwc1 $f0, 0(%0)\n\t"
                                "lwc1 $f2, 4(%0)\n\t"
                                "lwc1 $f4, 8(%0)\n\t"
                                "c.eq.s $fcc3, $f0, $f2\n\t"
                                "c.eq.s $fcc5, $f0, $f4\n\t"
                                "nop" Z
                             :: "r"(f) :);
        fc = fcsr();
        report_exc("c.eq.s with cc");
        con_printf("\n      [FCSR after cc3/cc5 compares = %x]", fc);
        CHECK_NO_EXC();
        /* CC1..CC7 live at FCSR bits 31:25; CC3 is bit 27, CC5 is bit 29. */
        CHECK_EQ((fc >> 27) & 1u, 1u);      /* 1.0 == 1.0 */
        CHECK_EQ((fc >> 29) & 1u, 0u);      /* 1.0 != 2.0 */
    }
}

/* ── the R4400 must also still refuse MIPS IV it never had ────────────────── */

/* A negative control: an instruction that exists on BOTH must not raise RI on
 * either. Without this, a CPU that raised RI for everything would pass every
 * R4400 expectation above. */
static void t_mips3_instruction_never_faults(void)
{
    u64 r;
    exc_clear();
    /* daddu $12, $13, $14 — SPECIAL funct 0x2D, MIPS III, valid on both. */
    __asm__ __volatile__(A "daddiu $13, $zero, 3\n\t"
                           "daddiu $14, $zero, 4\n\t"
                           ".word 0x01AE602D\n\t"
                           "daddu %0, $zero, $12" Z
                         : "=r"(r) :: "$12", "$13", "$14");
    CHECK_NO_EXC();
    CHECK_EQ(r, 7ull);
}

static const struct test tests[] = {
    TEST("mips4/movn_movz",        t_movn_movz,                   CPU_ALL),
    TEST("mips4/pref",             t_pref,                        CPU_ALL),
    TEST("mips4/recip_rsqrt",      t_recip_rsqrt,                 CPU_ALL),
    TEST("mips4/movci",            t_movci,                       CPU_ALL),
    TEST("mips4/cop1x",            t_cop1x,                       CPU_ALL),
    TEST("mips4/multi_fp_cc",      t_multiple_fp_condition_codes, CPU_ALL),
    TEST("mips4/mips3_control",    t_mips3_instruction_never_faults, CPU_ALL),
};

const struct test_group group_mips4 = {
    "mips4", tests, sizeof(tests) / sizeof(tests[0])
};
