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
 * region-boundary sentinel (src/cpu/mips_isa.rs:64), so testing it would measure
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
    /* With CU2 *set*, a real R4400 takes no exception at all: nothing tells the
     * CPU that coprocessor 2 is absent rather than merely present-and-enabled,
     * so the access simply executes and returns an undefined value. Confirmed
     * on an Indy R4400 rev 6.0, where this asserted an exception that never
     * came. The absent-coprocessor case is only observable with CU2 clear. */
    CHECK_EQ(exc.count, 0u);
    con_printf("\n      [cop2 with CU2 set: exceptions=%u]", exc.count);
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

/* ── CP0 is usable only in Kernel mode or with Status.CU0 ─────────────────── */

/*
 * "The CP0 instructions ... are usable in Kernel mode, or in User and
 * Supervisor mode when the CU0 bit of the Status register is set; otherwise a
 * Coprocessor Unusable exception is taken" (R4000 manual, chapter 5, and the
 * Coprocessor Unusable exception's cause list). Everything else in the suite
 * runs in Kernel mode, so nothing else can see whether a CPU enforces it.
 *
 * To test it the suite has to leave Kernel mode, which it otherwise never
 * does: KSEG0, where the suite lives, is not addressable from User mode. So a
 * few words of code go into a scratch page mapped at a kuseg address, ERET
 * enters them with KSU set, and a `syscall` at the end brings control back.
 * um_handler (below) is installed as the exception hook for the duration: a
 * syscall returns to Kernel mode at um_state.return_pc; any other exception
 * is counted, its Cause kept if it is the first, and stepped over in the mode
 * it came from. Eight of them and it goes home regardless, so a CPU that
 * mishandles this cannot hang the suite.
 *
 * Derived from the manual, not yet measured on silicon.
 */
#define UM_VA        0x00400000ull
#define UM_TLB_INDEX 12u
#define ST_KSU_SUPER 0x00000008u
#define ST_KSU_USER  0x00000010u

extern char _scratch_start[];
extern void um_handler(void);
extern struct { u32 faults; u32 first_cause; u64 return_pc; } um_state;

__asm__(
    "    .text\n"
    "    .set push; .set mips3; .set noreorder; .set nomacro; .set noat\n"
    "    .globl um_handler\n"
    "    .ent um_handler\n"
    "um_handler:\n"
    "    lui     $k1, %hi(um_state)\n"
    "    addiu   $k1, $k1, %lo(um_state)\n"
    "    mfc0    $k0, $13\n"
    "    nop\n"
    "    andi    $k0, $k0, 0x7c\n"
    "    xori    $k0, $k0, 0x20\n"          /* zero iff ExcCode 8, Sys */
    "    beqz    $k0, um_to_kernel\n"
    "    nop\n"
    "    lw      $k0, 0($k1)\n"
    "    bnez    $k0, 1f\n"
    "    nop\n"
    "    mfc0    $k0, $13\n"
    "    nop\n"
    "    sw      $k0, 4($k1)\n"             /* the first fault's Cause */
    "    lw      $k0, 0($k1)\n"
    "1:  addiu   $k0, $k0, 1\n"
    "    sw      $k0, 0($k1)\n"
    "    sltiu   $k0, $k0, 8\n"
    "    beqz    $k0, um_to_kernel\n"       /* runaway: go home anyway */
    "    nop\n"
    "    dmfc0   $k0, $14\n"
    "    nop\n"
    "    daddiu  $k0, $k0, 4\n"             /* step over it, same mode */
    "    dmtc0   $k0, $14\n"
    "    nop\n"
    "    b       um_eret\n"
    "    nop\n"
    "um_to_kernel:\n"
    "    mfc0    $k0, $12\n"
    "    nop\n"
    "    ori     $k0, $k0, 0x18\n"
    "    xori    $k0, $k0, 0x18\n"          /* KSU = Kernel; EXL still set */
    "    mtc0    $k0, $12\n"
    "    nop\n"
    "    ld      $k0, 8($k1)\n"
    "    dmtc0   $k0, $14\n"
    "    nop\n"
    "um_eret:\n"
    "    lui     $k0, %hi(exc_save)\n"
    "    addiu   $k0, $k0, %lo(exc_save)\n"
    "    ld      $at, 0($k0)\n"
    "    ld      $v0, 8($k0)\n"
    "    ld      $v1, 16($k0)\n"
    "    ssnop\n"
    "    ssnop\n"
    "    ssnop\n"
    "    ssnop\n"
    "    eret\n"
    "    nop\n"
    "    .end um_handler\n"
    "    .set pop\n"
    "    .section .bss\n"
    "    .align 3\n"
    "    .globl um_state\n"
    "um_state:\n"
    "    .space 16\n"
    "    .text\n");

/*
 * Run `n` words of code at UM_VA with KSU = `ksu` and CU0 as given; the code
 * must end in a syscall. Returns with Kernel mode, Status, the TLB entry and
 * the exception hook all put back.
 */
static void run_unprivileged(const u32 *code, unsigned n, u32 ksu, u32 cu0)
{
    volatile u32 *page = (volatile u32 *)_scratch_start;
    u64 phys = (u64)((u32)(unsigned long)_scratch_start & 0x1FFFFFFFu);
    u64 saved_hi = cp0_entryhi();
    u32 saved_pm = cp0_pagemask();
    u32 saved_status = cp0_status();
    u32 status;
    unsigned i;

    for (i = 0; i < n; i++) page[i] = code[i];
    SYNC();
    dcache_wb_invalidate_range(page, n * 4);

    cp0_index_set(UM_TLB_INDEX);
    cp0_pagemask_set(PM_4K);
    cp0_entryhi_set(UM_VA);
    cp0_entrylo0_set(((phys >> 12) << ELO_PFN_SHIFT) |
                     ((u64)CA_CACHEABLE_NC << ELO_C_SHIFT) | ELO_V | ELO_D | ELO_G);
    cp0_entrylo1_set(0);
    tlb_write_indexed();
    /* The same physical line through the mapping, so no instruction cached
     * from an earlier use of the scratch page survives into this one. */
    icache_invalidate_range(page, n * 4);
    icache_invalidate_range(SEXT_PTR((u32)UM_VA), n * 4);

    um_state.faults = 0;
    um_state.first_cause = 0;
    exc_clear();
    exc_user_handler = (u32)(unsigned long)&um_handler;

    status = (saved_status & ~(ST_CU0 | 0x18u | ST_IE)) | ST_EXL | ksu | (cu0 ? ST_CU0 : 0);
    __asm__ __volatile__(A
        "dla    $8, 1f\n\t"
        "sd     $8, 8(%0)\n\t"             /* um_state.return_pc */
        "mfc0   $9, $12\n\t"
        "nop\n\t"
        "ori    $9, $9, 0x2\n\t"           /* EXL first: still Kernel */
        "mtc0   $9, $12\n\t"
        "nop; nop; nop\n\t"
        "mtc0   %1, $12\n\t"               /* KSU and CU0, EXL kept */
        "nop; nop; nop\n\t"
        "daddiu $10, $zero, 0x40\n\t"
        "dsll   $10, $10, 16\n\t"          /* UM_VA = 0x00400000 */
        "dmtc0  $10, $14\n\t"
        "nop; nop; nop\n\t"
        "eret\n\t"
        "nop\n\t"
        "1:" Z
        :: "r"(&um_state), "r"(status)
        : "$2", "$8", "$9", "$10", "memory");

    exc_user_handler = 0;
    cp0_status_set(saved_status);

    cp0_index_set(UM_TLB_INDEX);
    cp0_pagemask_set(PM_4K);
    cp0_entryhi_set(0x1FFFE000ull);
    cp0_entrylo0_set(0);
    cp0_entrylo1_set(0);
    tlb_write_indexed();
    cp0_entryhi_set(saved_hi);
    cp0_pagemask_set(saved_pm);
}

#define UM_SYSCALL  0x0000000Cu            /* syscall                */
#define UM_MFC0_SR  0x40026000u            /* mfc0 $2, $12 (Status)  */
#define UM_TLBP     0x42000008u            /* tlbp                   */
#define UM_NOP      0x00000000u

static void t_cp0_unusable_outside_kernel(void)
{
    static const u32 control[] = { UM_SYSCALL, UM_NOP };
    static const u32 mfc0[]    = { UM_MFC0_SR, UM_SYSCALL, UM_NOP };
    static const u32 tlbp[]    = { UM_TLBP, UM_SYSCALL, UM_NOP };

    /* The control: into User mode and straight back, no fault on the way.
     * Without it a CPU that cannot enter User mode at all would pass the
     * rest by faulting for the wrong reason. */
    run_unprivileged(control, 2, ST_KSU_USER, 0);
    CHECK_EQ(um_state.faults, 0u);
    CHECK_EQ(exc.count, 1u);
    CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_SYS);
    CHECK_EQ(exc.status & 0x18u, ST_KSU_USER);

    /* MFC0 from User mode, CU0 clear: Coprocessor Unusable, CE = 0. */
    run_unprivileged(mfc0, 3, ST_KSU_USER, 0);
    CHECK_EQ(um_state.faults, 1u);
    CHECK_EQ(CAUSE_EXC(um_state.first_cause), (u32)EXC_CPU);
    CHECK_EQ((um_state.first_cause & CAUSE_CE_MASK) >> CAUSE_CE_SHIFT, 0u);

    /* A TLB instruction is a CP0 instruction too. */
    run_unprivileged(tlbp, 3, ST_KSU_USER, 0);
    CHECK_EQ(um_state.faults, 1u);
    CHECK_EQ(CAUSE_EXC(um_state.first_cause), (u32)EXC_CPU);

    /* Supervisor mode gets no implicit access either. */
    run_unprivileged(mfc0, 3, ST_KSU_SUPER, 0);
    CHECK_EQ(um_state.faults, 1u);
    CHECK_EQ(CAUSE_EXC(um_state.first_cause), (u32)EXC_CPU);
}

static void t_cp0_usable_with_cu0(void)
{
    static const u32 mfc0[] = { UM_MFC0_SR, UM_SYSCALL, UM_NOP };

    run_unprivileged(mfc0, 3, ST_KSU_USER, 1);
    CHECK_EQ(um_state.faults, 0u);
    CHECK_EQ(exc.count, 1u);
    CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_SYS);
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
    TEST("excep/cp0_unusable_user",    t_cp0_unusable_outside_kernel,        CPU_ALL),
    TEST("excep/cp0_usable_cu0",       t_cp0_usable_with_cu0,                CPU_ALL),
};

const struct test_group group_excep = {
    "excep", tests, sizeof(tests) / sizeof(tests[0])
};
