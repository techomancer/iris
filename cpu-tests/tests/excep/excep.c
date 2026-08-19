/* excep — explicit traps, reserved instructions, coprocessor usability,
 * vector selection, and the Status/Cause bits around an exception.
 *
 * The harness's own exception plumbing (start.S) is what makes the rest of the
 * suite able to test faults at all, so several tests here are really tests of
 * that plumbing: which vector ran, whether EXL was set on entry and cleared by
 * ERET, and whether the handler left the register file alone.
 */

#include "testlib.h"
#include "cp0.h"
#include "excoff.h"

#define A ".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"
#define Z "\n\t.set pop"

/* The suite is built -msoft-float so the compiler never emits FP code of its
 * own (the FPU tests change FR and FCSR underneath it). That also makes GAS
 * refuse FP mnemonics outright, so any block containing one needs an explicit
 * `.set hardfloat`. */
#define AF A ".set hardfloat\n\t"

#define OPAQUE(x) ({ __typeof__(x) __v = (x); __asm__ __volatile__("" : "+r"(__v)); __v; })

/* ── explicit trap instructions ───────────────────────────────────────────── */

static void t_syscall(void)
{
    exc_clear();
    __asm__ __volatile__(A "syscall" Z);
    CHECK_EXC(EXC_SYS);
    CHECK_EQ(exc.vector, (u32)VECID_GENERAL);
}

static void t_break(void)
{
    exc_clear();
    __asm__ __volatile__(A "break" Z);
    CHECK_EXC(EXC_BP);
    CHECK_EQ(exc.vector, (u32)VECID_GENERAL);
}

/* The conditional traps: each fires only when its condition holds. */
static void t_teq_tne(void)
{
    u64 a = OPAQUE(5ull), b = OPAQUE(5ull), c = OPAQUE(6ull);

    exc_clear();
    __asm__ __volatile__(A "teq %0, %1" Z :: "r"(a), "r"(b));
    CHECK_EXC(EXC_TR);

    exc_clear();
    __asm__ __volatile__(A "teq %0, %1" Z :: "r"(a), "r"(c));
    CHECK_NO_EXC();

    exc_clear();
    __asm__ __volatile__(A "tne %0, %1" Z :: "r"(a), "r"(c));
    CHECK_EXC(EXC_TR);

    exc_clear();
    __asm__ __volatile__(A "tne %0, %1" Z :: "r"(a), "r"(b));
    CHECK_NO_EXC();
}

static void t_tlt_tge(void)
{
    u64 small = OPAQUE((u64)(s64)-1), big = OPAQUE(1ull);

    exc_clear();
    __asm__ __volatile__(A "tlt %0, %1" Z :: "r"(small), "r"(big));
    CHECK_EXC(EXC_TR);            /* -1 < 1 signed */

    exc_clear();
    __asm__ __volatile__(A "tltu %0, %1" Z :: "r"(small), "r"(big));
    CHECK_NO_EXC();               /* 0xffff... > 1 unsigned */

    exc_clear();
    __asm__ __volatile__(A "tge %0, %1" Z :: "r"(big), "r"(small));
    CHECK_EXC(EXC_TR);            /* 1 >= -1 signed */

    exc_clear();
    __asm__ __volatile__(A "tgeu %0, %1" Z :: "r"(big), "r"(small));
    CHECK_NO_EXC();               /* 1 < 0xffff... unsigned */
}

/* The immediate trap forms sign-extend their 16-bit immediate. */
static void t_trap_immediate(void)
{
    u64 v = OPAQUE((u64)(s64)-1);

    exc_clear();
    __asm__ __volatile__(A "teqi %0, -1" Z :: "r"(v));
    CHECK_EXC(EXC_TR);

    exc_clear();
    __asm__ __volatile__(A "tnei %0, -1" Z :: "r"(v));
    CHECK_NO_EXC();

    v = OPAQUE(0ull);
    exc_clear();
    __asm__ __volatile__(A "tlti %0, 1" Z :: "r"(v));
    CHECK_EXC(EXC_TR);            /* 0 < 1 */

    exc_clear();
    __asm__ __volatile__(A "tltiu %0, -1" Z :: "r"(v));
    CHECK_EXC(EXC_TR);            /* 0 < 0xffffffffffffffff unsigned */
}

/* ── reserved instructions ────────────────────────────────────────────────── */

/*
 * An undefined primary opcode raises Reserved Instruction. `.word` is used
 * rather than a mnemonic because there is, by construction, no mnemonic.
 *
 * Picking the encoding matters more than it looks. The obvious-seeming 0x3F is
 * SD, not a hole — `.word 0xFC000000` is `sd $zero, 0($zero)`, which stores to
 * virtual address 0, misses in the TLB, and reports TLBS through the XTLB
 * refill vector. That is a perfectly good TLB test and a useless RI test.
 *
 * Primary opcodes 0x1C..0x1F are the real holes on R4400/R5000. (Later MIPS32
 * revisions claimed 0x1C and 0x1F as SPECIAL2/SPECIAL3, but neither of these
 * parts implements them.) 0x1E is skipped: IRIS's own jitv2 uses it as a
 * region-boundary sentinel (src/mips_isa.rs:64), so testing it would measure
 * the JIT's tooling rather than the CPU.
 */
static void t_reserved_instruction(void)
{
    exc_clear();
    __asm__ __volatile__(A ".word 0x70000000" Z);   /* opcode 0x1C */
    CHECK_EXC(EXC_RI);
    CHECK_EQ(exc.vector, (u32)VECID_GENERAL);

    exc_clear();
    __asm__ __volatile__(A ".word 0x74000000" Z);   /* opcode 0x1D */
    CHECK_EXC(EXC_RI);

    exc_clear();
    __asm__ __volatile__(A ".word 0x7C000000" Z);   /* opcode 0x1F */
    CHECK_EXC(EXC_RI);
}

/* ── coprocessor usability ────────────────────────────────────────────────── */

/*
 * With Status.CU1 clear, any COP1 access raises Coprocessor Unusable with
 * Cause.CE == 1. This is also the test that forced the handler in start.S to
 * guard its own FCSR read: an unguarded `cfc1` in the handler would fault
 * again, inside the handler, and never return.
 */
static void t_cop1_unusable_when_cu1_clear(void)
{
    u32 saved = cp0_status();
    cp0_status_set(saved & ~ST_CU1);

    exc_clear();
    __asm__ __volatile__(AF "mfc1 $12, $f0" Z ::: "$12");

    cp0_status_set(saved);      /* restore before asserting, so a failure
                                 * message can still use the FPU-free path */
    CHECK_EXC(EXC_CPU);
    CHECK_EQ((exc.cause & CAUSE_CE_MASK) >> CAUSE_CE_SHIFT, 1u);
}

static void t_cop1_usable_when_cu1_set(void)
{
    u32 saved = cp0_status();
    cp0_status_set(saved | ST_CU1);
    exc_clear();
    __asm__ __volatile__(AF "mfc1 $12, $f0" Z ::: "$12");
    CHECK_NO_EXC();
    cp0_status_set(saved);
}

/* COP2 does not exist on either part, so it is unusable no matter what CU2
 * says — and Cause.CE reports 2. */
static void t_cop2_always_unusable(void)
{
    u32 saved = cp0_status();
    cp0_status_set(saved | ST_CU2);
    exc_clear();
    /* `mfc2 $12, $0` as a raw word: COP2 (opcode 0x12), rs=0 (MF), rt=12,
     * rd=0. Spelled out because GAS will not encode a COP2 access for a
     * -march=mips3 target that has no COP2. */
    __asm__ __volatile__(A ".word 0x480C0000" Z ::: "$12");
    cp0_status_set(saved);
    /* Either Coprocessor Unusable (CE=2) or Reserved Instruction is defensible
     * for a coprocessor that is not merely disabled but absent. Accept both,
     * and report which. */
    CHECK_EQ(exc.count, 1u);
    CHECK(CAUSE_EXC(exc.cause) == EXC_CPU || CAUSE_EXC(exc.cause) == EXC_RI);
    con_printf("\n      [cop2: ExcCode=%u CE=%u]",
               CAUSE_EXC(exc.cause), (exc.cause & CAUSE_CE_MASK) >> CAUSE_CE_SHIFT);
}

/* ── Status and Cause around an exception ─────────────────────────────────── */

/* EXL must be set on entry to the handler and cleared by ERET. */
static void t_exl_set_in_handler_cleared_by_eret(void)
{
    u32 after;
    exc_clear();
    __asm__ __volatile__(A "syscall" Z);
    after = cp0_status();

    CHECK_EXC(EXC_SYS);
    CHECK_EQ(exc.status & ST_EXL, ST_EXL);   /* set inside the handler */
    CHECK_EQ(after & ST_EXL, 0u);            /* cleared after ERET */
}

/* A handler runs in kernel mode with interrupts effectively masked by EXL,
 * and the suite runs with IE clear anyway — so no interrupt should appear. */
static void t_no_interrupt_pending_during_test(void)
{
    exc_clear();
    __asm__ __volatile__(A "syscall" Z);
    CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_SYS);
    CHECK_NE(CAUSE_EXC(exc.cause), (u32)EXC_INT);
}

/*
 * The handler must not disturb the register file. Fill the callee-saved
 * registers with a known pattern, take an exception, and check every one
 * survived — this is what lets the rest of the suite trust "the destination
 * register was not written" assertions.
 */
static void t_handler_preserves_registers(void)
{
    u64 s0, s1, s2, s3, t0, t1;
    exc_clear();
    __asm__ __volatile__(A
        "dli $16, 0x1111111111111111\n\t"
        "dli $17, 0x2222222222222222\n\t"
        "dli $18, 0x3333333333333333\n\t"
        "dli $19, 0x4444444444444444\n\t"
        "dli $12, 0x5555555555555555\n\t"
        "dli $13, 0x6666666666666666\n\t"
        "syscall\n\t"
        "daddu %0, $zero, $16\n\t"
        "daddu %1, $zero, $17\n\t"
        "daddu %2, $zero, $18\n\t"
        "daddu %3, $zero, $19\n\t"
        "daddu %4, $zero, $12\n\t"
        "daddu %5, $zero, $13" Z
        : "=r"(s0), "=r"(s1), "=r"(s2), "=r"(s3), "=r"(t0), "=r"(t1)
        :: "$16", "$17", "$18", "$19", "$12", "$13");

    CHECK_EXC(EXC_SYS);
    CHECK_EQ(s0, 0x1111111111111111ull);
    CHECK_EQ(s1, 0x2222222222222222ull);
    CHECK_EQ(s2, 0x3333333333333333ull);
    CHECK_EQ(s3, 0x4444444444444444ull);
    CHECK_EQ(t0, 0x5555555555555555ull);
    CHECK_EQ(t1, 0x6666666666666666ull);
}

/* EPC points at the trapping instruction itself. */
static void t_epc_points_at_faulting_instruction(void)
{
    u64 addr;
    exc_clear();
    __asm__ __volatile__(A
        "dla %0, 1f\n\t"
        "1:\n\t"
        "syscall" Z
        : "=r"(addr));
    CHECK_EXC(EXC_SYS);
    CHECK_EQ(exc.epc, addr);
}

/* ── nested exceptions ────────────────────────────────────────────────────── */

/*
 * Taking an exception with EXL already set is legal — it just does not
 * re-write EPC, because the first exception's EPC must survive. Set EXL by
 * hand, trap, and confirm EPC was left alone.
 *
 * Interrupts stay off throughout; the point is the EPC-preservation rule, not
 * re-entrancy of the handler.
 */
static void t_exception_with_exl_set_preserves_epc(void)
{
    u32 saved = cp0_status();
    /* A recognisable value that is nonetheless a plausible KSEG0 address, so
     * nothing downstream is tempted to treat it as a real target. */
    const u64 sentinel = 0xFFFFFFFF80ABCDE0ull;

    exc_clear();
    /* The default handler resumes through EPC, which this test deliberately
     * leaves pointing at the sentinel — so install the resume-at-a-label
     * handler instead, or the ERET would jump into the sentinel and loop. */
    exc_user_handler = (u32)(unsigned long)&exl_resume_handler;

    __asm__ __volatile__(A
        "dla $12, 1f\n\t"
        "sd $12, 0(%0)\n\t"          /* exl_resume_pc = after the syscall */
        "dmtc0 %1, $14\n\t"          /* EPC = sentinel */
        "nop\n\t"
        "mtc0 %2, $12\n\t"           /* Status |= EXL */
        "nop\n\t"
        "nop\n\t"
        "syscall\n\t"
        "1:" Z
        :: "r"(&exl_resume_pc), "r"(sentinel), "r"(saved | ST_EXL)
        : "$12", "memory");

    cp0_status_set(saved);
    exc_user_handler = 0;

    CHECK_EQ(exc.count, 1u);
    CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_SYS);
    /* EPC must still hold the sentinel: an exception taken while EXL is
     * already set does not overwrite it. */
    CHECK_EQ(exc.epc, sentinel);
    /* And EXL must have been set when the handler saw it. */
    CHECK_EQ(exc.status & ST_EXL, ST_EXL);
}

/* ── vector selection ─────────────────────────────────────────────────────── */

/* Everything that is not a TLB refill goes to the general vector at
 * 0x80000180 when BEV is clear. */
static void t_general_vector_used(void)
{
    exc_clear();
    __asm__ __volatile__(A "break" Z);
    CHECK_EQ(exc.vector, (u32)VECID_GENERAL);

    exc_clear();
    __asm__ __volatile__(A "syscall" Z);
    CHECK_EQ(exc.vector, (u32)VECID_GENERAL);

    exc_clear();
    __asm__ __volatile__(A ".word 0x70000000" Z);   /* reserved opcode 0x1C */
    CHECK_EQ(exc.vector, (u32)VECID_GENERAL);
}

static const struct test tests[] = {
    TEST("excep/syscall",              t_syscall,                            CPU_ALL),
    TEST("excep/break",                t_break,                              CPU_ALL),
    TEST("excep/teq_tne",              t_teq_tne,                            CPU_ALL),
    TEST("excep/tlt_tge",              t_tlt_tge,                            CPU_ALL),
    TEST("excep/trap_immediate",       t_trap_immediate,                     CPU_ALL),
    TEST("excep/reserved_instruction", t_reserved_instruction,               CPU_ALL),
    TEST("excep/cop1_unusable",        t_cop1_unusable_when_cu1_clear,       CPU_ALL),
    TEST("excep/cop1_usable",          t_cop1_usable_when_cu1_set,           CPU_ALL),
    TEST("excep/cop2_unusable",        t_cop2_always_unusable,               CPU_ALL),
    TEST("excep/exl_set_and_cleared",  t_exl_set_in_handler_cleared_by_eret, CPU_ALL),
    TEST("excep/no_spurious_int",      t_no_interrupt_pending_during_test,   CPU_ALL),
    TEST("excep/handler_preserves_gpr", t_handler_preserves_registers,       CPU_ALL),
    TEST("excep/epc_is_faulting_insn", t_epc_points_at_faulting_instruction, CPU_ALL),
    TEST("excep/exl_preserves_epc",    t_exception_with_exl_set_preserves_epc, CPU_ALL),
    TEST("excep/general_vector",       t_general_vector_used,                CPU_ALL),
};

const struct test_group group_excep = {
    "excep", tests, sizeof(tests) / sizeof(tests[0])
};
