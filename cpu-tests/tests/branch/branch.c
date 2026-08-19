/* branch — conditional branches, jumps, delay slots, and nullification.
 *
 * Two things dominate here. First, every branch has a delay slot whose
 * instruction executes regardless of whether the branch is taken — except for
 * the "likely" variants, where it is nullified when the branch is NOT taken.
 * Second, the link register gets PC+8 (past the delay slot), not PC+4.
 *
 * Almost every test sets a flag in the delay slot and inspects it afterwards,
 * so a delay slot that is skipped, executed twice, or executed in the wrong
 * order shows up as a wrong flag rather than as a crash.
 */

#include "testlib.h"
#include "cp0.h"

#define A ".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"
#define Z "\n\t.set pop"

#define OPAQUE(x) ({ __typeof__(x) __v = (x); __asm__ __volatile__("" : "+r"(__v)); __v; })

/*
 * Run one conditional branch and report both whether it was taken and whether
 * the delay slot ran. `cond` is the mnemonic plus its operands; the branch
 * targets a local label.
 *
 *   taken = 1 if control reached the target
 *   slot  = 1 if the delay-slot instruction executed
 */
#define BRANCH_PROBE(insn, taken, slot, ...)                               \
    __asm__ __volatile__(A                                                 \
        "daddu %0, $zero, $zero\n\t"      /* taken = 0 */                  \
        "daddu %1, $zero, $zero\n\t"      /* slot  = 0 */                  \
        insn " 1f\n\t"                                                     \
        "daddiu %1, $zero, 1\n\t"         /* DELAY SLOT: slot = 1 */       \
        "b 2f\n\t"                                                         \
        "nop\n\t"                                                          \
        "1:\n\t"                                                           \
        "daddiu %0, $zero, 1\n\t"         /* taken = 1 */                  \
        "2:" Z                                                             \
        : "=&r"(taken), "=&r"(slot) : __VA_ARGS__)

/* ── conditional branches, taken and not taken ────────────────────────────── */

static void t_beq_bne(void)
{
    u64 taken, slot;
    u64 a = OPAQUE(5ull), b = OPAQUE(5ull), c = OPAQUE(6ull);

    BRANCH_PROBE("beq %2, %3,", taken, slot, "r"(a), "r"(b));
    CHECK_EQ(taken, 1ull); CHECK_EQ(slot, 1ull);

    BRANCH_PROBE("beq %2, %3,", taken, slot, "r"(a), "r"(c));
    CHECK_EQ(taken, 0ull); CHECK_EQ(slot, 1ull);   /* slot runs either way */

    BRANCH_PROBE("bne %2, %3,", taken, slot, "r"(a), "r"(c));
    CHECK_EQ(taken, 1ull); CHECK_EQ(slot, 1ull);

    BRANCH_PROBE("bne %2, %3,", taken, slot, "r"(a), "r"(b));
    CHECK_EQ(taken, 0ull); CHECK_EQ(slot, 1ull);
}

/* beq/bne compare all 64 bits, so two values that agree in their low words but
 * differ above bit 31 are NOT equal. */
static void t_beq_compares_64_bits(void)
{
    u64 taken, slot;
    u64 a = OPAQUE(0x0000000000000001ull);
    u64 b = OPAQUE(0x0000000100000001ull);
    BRANCH_PROBE("beq %2, %3,", taken, slot, "r"(a), "r"(b));
    CHECK_EQ(taken, 0ull);
}

static void t_bgez_bltz(void)
{
    u64 taken, slot;
    u64 pos = OPAQUE(1ull), neg = OPAQUE((u64)(s64)-1), zero = OPAQUE(0ull);

    BRANCH_PROBE("bgez %2,", taken, slot, "r"(pos));  CHECK_EQ(taken, 1ull);
    BRANCH_PROBE("bgez %2,", taken, slot, "r"(zero)); CHECK_EQ(taken, 1ull);
    BRANCH_PROBE("bgez %2,", taken, slot, "r"(neg));  CHECK_EQ(taken, 0ull);

    BRANCH_PROBE("bltz %2,", taken, slot, "r"(neg));  CHECK_EQ(taken, 1ull);
    BRANCH_PROBE("bltz %2,", taken, slot, "r"(zero)); CHECK_EQ(taken, 0ull);
    BRANCH_PROBE("bltz %2,", taken, slot, "r"(pos));  CHECK_EQ(taken, 0ull);
}

static void t_bgtz_blez(void)
{
    u64 taken, slot;
    u64 pos = OPAQUE(1ull), neg = OPAQUE((u64)(s64)-1), zero = OPAQUE(0ull);

    BRANCH_PROBE("bgtz %2,", taken, slot, "r"(pos));  CHECK_EQ(taken, 1ull);
    BRANCH_PROBE("bgtz %2,", taken, slot, "r"(zero)); CHECK_EQ(taken, 0ull);
    BRANCH_PROBE("bgtz %2,", taken, slot, "r"(neg));  CHECK_EQ(taken, 0ull);

    BRANCH_PROBE("blez %2,", taken, slot, "r"(zero)); CHECK_EQ(taken, 1ull);
    BRANCH_PROBE("blez %2,", taken, slot, "r"(neg));  CHECK_EQ(taken, 1ull);
    BRANCH_PROBE("blez %2,", taken, slot, "r"(pos));  CHECK_EQ(taken, 0ull);
}

/* The sign test is on the full 64-bit value: 0x80000000 is a positive 64-bit
 * number even though it is a negative 32-bit one. */
static void t_branch_sign_test_is_64bit(void)
{
    u64 taken, slot;
    u64 v = OPAQUE(0x0000000080000000ull);
    BRANCH_PROBE("bltz %2,", taken, slot, "r"(v));
    CHECK_EQ(taken, 0ull);

    v = OPAQUE(0xFFFFFFFF80000000ull);      /* properly sign-extended: negative */
    BRANCH_PROBE("bltz %2,", taken, slot, "r"(v));
    CHECK_EQ(taken, 1ull);
}

/* ── branch-likely: the delay slot is nullified when NOT taken ────────────── */

static void t_branch_likely_nullification(void)
{
    u64 taken, slot;
    u64 a = OPAQUE(5ull), b = OPAQUE(5ull), c = OPAQUE(6ull);

    BRANCH_PROBE("beql %2, %3,", taken, slot, "r"(a), "r"(b));
    CHECK_EQ(taken, 1ull);
    CHECK_EQ(slot, 1ull);           /* taken: the slot DOES run */

    BRANCH_PROBE("beql %2, %3,", taken, slot, "r"(a), "r"(c));
    CHECK_EQ(taken, 0ull);
    CHECK_EQ(slot, 0ull);           /* not taken: the slot is NULLIFIED */

    BRANCH_PROBE("bnel %2, %3,", taken, slot, "r"(a), "r"(c));
    CHECK_EQ(taken, 1ull); CHECK_EQ(slot, 1ull);

    BRANCH_PROBE("bnel %2, %3,", taken, slot, "r"(a), "r"(b));
    CHECK_EQ(taken, 0ull); CHECK_EQ(slot, 0ull);
}

static void t_regimm_likely_nullification(void)
{
    u64 taken, slot;
    u64 pos = OPAQUE(1ull), neg = OPAQUE((u64)(s64)-1);

    BRANCH_PROBE("bgezl %2,", taken, slot, "r"(pos));
    CHECK_EQ(taken, 1ull); CHECK_EQ(slot, 1ull);
    BRANCH_PROBE("bgezl %2,", taken, slot, "r"(neg));
    CHECK_EQ(taken, 0ull); CHECK_EQ(slot, 0ull);

    BRANCH_PROBE("bltzl %2,", taken, slot, "r"(neg));
    CHECK_EQ(taken, 1ull); CHECK_EQ(slot, 1ull);
    BRANCH_PROBE("bltzl %2,", taken, slot, "r"(pos));
    CHECK_EQ(taken, 0ull); CHECK_EQ(slot, 0ull);

    BRANCH_PROBE("bgtzl %2,", taken, slot, "r"(pos));
    CHECK_EQ(taken, 1ull); CHECK_EQ(slot, 1ull);
    BRANCH_PROBE("blezl %2,", taken, slot, "r"(pos));
    CHECK_EQ(taken, 0ull); CHECK_EQ(slot, 0ull);
}

/* ── link registers ───────────────────────────────────────────────────────── */

/*
 * BAL/JAL/JALR link PC+8 — the instruction AFTER the delay slot. Checked by
 * having the linked address point at a known label and comparing.
 */
static void t_bal_links_past_delay_slot(void)
{
    u64 ra_val, here;
    __asm__ __volatile__(A
        "bal 1f\n\t"
        "nop\n\t"                 /* delay slot */
        "3:\n\t"                  /* PC+8: where $ra must point */
        "b 2f\n\t"
        "nop\n\t"
        "1:\n\t"
        "daddu %0, $zero, $ra\n\t"
        "dla %1, 3b\n\t"
        "2:" Z
        : "=r"(ra_val), "=r"(here) :: "$31");
    CHECK_EQ(ra_val, here);
}

/* JALR with an explicit destination register other than $ra. */
static void t_jalr_explicit_rd(void)
{
    u64 link, here;
    __asm__ __volatile__(A
        "dla $12, 1f\n\t"
        "jalr $13, $12\n\t"       /* link into $13, not $31 */
        "nop\n\t"
        "3:\n\t"
        "b 2f\n\t"
        "nop\n\t"
        "1:\n\t"
        "daddu %0, $zero, $13\n\t"
        "dla %1, 3b\n\t"
        "2:" Z
        : "=r"(link), "=r"(here) :: "$12", "$13");
    CHECK_EQ(link, here);
}

/* The delay slot of a JAL may itself write the link register; the link write
 * happens first, so the delay slot's value is what survives. */
static void t_delay_slot_may_overwrite_ra(void)
{
    u64 ra_val;
    __asm__ __volatile__(A
        "bal 1f\n\t"
        "daddiu $31, $zero, 0x123\n\t"   /* delay slot clobbers $ra */
        "b 2f\n\t"
        "nop\n\t"
        "1:\n\t"
        "daddu %0, $zero, $ra\n\t"
        "2:" Z
        : "=r"(ra_val) :: "$31");
    CHECK_EQ(ra_val, 0x123ull);
}

/* ── jumps ────────────────────────────────────────────────────────────────── */

static void t_jr_and_delay_slot(void)
{
    u64 slot = OPAQUE(0ull), reached = OPAQUE(0ull);
    __asm__ __volatile__(A
        "dla $12, 1f\n\t"
        "daddu %0, $zero, $zero\n\t"
        "daddu %1, $zero, $zero\n\t"
        "jr $12\n\t"
        "daddiu %0, $zero, 1\n\t"        /* delay slot runs */
        "daddiu %1, $zero, 99\n\t"       /* skipped */
        "1:\n\t"
        "daddiu %1, %1, 1" Z
        : "=&r"(slot), "=&r"(reached) :: "$12");
    CHECK_EQ(slot, 1ull);
    CHECK_EQ(reached, 1ull);
}

/* A load in the delay slot must complete, and its result must be visible at
 * the branch target — the classic load-delay interaction. */
static void t_load_in_delay_slot(void)
{
    extern char _scratch_start[];
    volatile u32 *p = (volatile u32 *)_scratch_start;
    u64 v;
    *p = 0xABCD1234u;
    SYNC();
    __asm__ __volatile__(A
        "daddu %0, $zero, $zero\n\t"
        "b 1f\n\t"
        "lw %0, 0(%1)\n\t"               /* delay slot */
        "daddiu %0, $zero, 0\n\t"        /* skipped */
        "1:" Z
        : "=&r"(v) : "r"(p));
    CHECK_EQ(v, 0xFFFFFFFFABCD1234ull);
}

/* ── exceptions in a delay slot ───────────────────────────────────────────── */

/*
 * When an instruction in a branch delay slot faults, EPC points at the BRANCH,
 * not at the faulting instruction, and Cause.BD is set. Resuming therefore has
 * to skip both — which is what the harness's EXC_RESUME_SKIP does (EPC += 8
 * when BD is set).
 */
static void t_exception_in_delay_slot_sets_bd(void)
{
    u64 branch_addr, taken;
    u64 a = OPAQUE(0x7FFFFFFFull), b = OPAQUE(1ull);

    exc_clear();
    __asm__ __volatile__(A
        "daddu %1, $zero, $zero\n\t"
        "dla %0, 3f\n\t"
        "3:\n\t"
        "b 1f\n\t"
        "add $12, %2, %3\n\t"            /* delay slot: overflows, traps */
        "1:\n\t"
        "daddiu %1, $zero, 1" Z
        : "=&r"(branch_addr), "=&r"(taken)
        : "r"(a), "r"(b) : "$12");

    CHECK_EXC(EXC_OV);
    CHECK_EQ(exc.cause & CAUSE_BD, CAUSE_BD);   /* BD must be set */
    CHECK_EQ(exc.epc, branch_addr);             /* EPC points at the branch */
    CHECK_EQ(taken, 1ull);                      /* and the branch still happened */
}

/* Without a delay slot involved, BD is clear and EPC is the faulting
 * instruction itself. */
static void t_exception_not_in_delay_slot(void)
{
    u64 faulting_addr;
    u64 a = OPAQUE(0x7FFFFFFFull), b = OPAQUE(1ull);

    exc_clear();
    __asm__ __volatile__(A
        "dla %0, 3f\n\t"
        "3:\n\t"
        "add $12, %1, %2" Z
        : "=&r"(faulting_addr) : "r"(a), "r"(b) : "$12");

    CHECK_EXC(EXC_OV);
    CHECK_EQ(exc.cause & CAUSE_BD, 0u);
    CHECK_EQ(exc.epc, faulting_addr);
}

/* ── branch displacement ──────────────────────────────────────────────────── */

/* Backward branches use a negative displacement; a loop that counts down
 * exercises the sign extension of the 16-bit offset. */
static void t_backward_branch(void)
{
    u64 count;
    __asm__ __volatile__(A
        "daddiu %0, $zero, 0\n\t"
        "daddiu $12, $zero, 10\n\t"
        "1:\n\t"
        "daddiu %0, %0, 1\n\t"
        "daddiu $12, $12, -1\n\t"
        "bnez $12, 1b\n\t"
        "nop" Z
        : "=&r"(count) :: "$12");
    CHECK_EQ(count, 10ull);
}

static const struct test tests[] = {
    TEST("branch/beq_bne",             t_beq_bne,                        CPU_ALL),
    TEST("branch/beq_64bit_compare",   t_beq_compares_64_bits,           CPU_ALL),
    TEST("branch/bgez_bltz",           t_bgez_bltz,                      CPU_ALL),
    TEST("branch/bgtz_blez",           t_bgtz_blez,                      CPU_ALL),
    TEST("branch/sign_test_64bit",     t_branch_sign_test_is_64bit,      CPU_ALL),
    TEST("branch/likely_nullifies",    t_branch_likely_nullification,    CPU_ALL),
    TEST("branch/regimm_likely",       t_regimm_likely_nullification,    CPU_ALL),
    TEST("branch/bal_links_pc_plus_8", t_bal_links_past_delay_slot,      CPU_ALL),
    TEST("branch/jalr_explicit_rd",    t_jalr_explicit_rd,               CPU_ALL),
    TEST("branch/delay_slot_sets_ra",  t_delay_slot_may_overwrite_ra,    CPU_ALL),
    TEST("branch/jr_delay_slot",       t_jr_and_delay_slot,              CPU_ALL),
    TEST("branch/load_in_delay_slot",  t_load_in_delay_slot,             CPU_ALL),
    TEST("branch/exc_in_delay_slot",   t_exception_in_delay_slot_sets_bd, CPU_ALL),
    TEST("branch/exc_no_delay_slot",   t_exception_not_in_delay_slot,    CPU_ALL),
    TEST("branch/backward_branch",     t_backward_branch,                CPU_ALL),
};

const struct test_group group_branch = {
    "branch", tests, sizeof(tests) / sizeof(tests[0])
};
