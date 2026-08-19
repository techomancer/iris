/* fpu_trap — trapped floating-point exceptions.
 *
 * Everything in fpu.c runs with the FCSR Enable bits clear, so only the
 * *untrapped* path is exercised there: the Flag bit is set, a default result
 * is delivered, and execution continues. This file is the other half — what
 * happens when an Enable bit is set and the FPU actually raises Floating-Point
 * Exception (ExcCode 15) through the CPU.
 *
 * The three-field interaction is the whole subject, and the R4000 manual is
 * unusually explicit about it (chapter 6, "Control/Status Register Cause,
 * Flag, and Enable Fields", and chapter 7):
 *
 *   Cause    written by every FP *operation* — and only operations, not loads,
 *            stores or moves. It "reflects the results of the most recently
 *            executed instruction", so a clean operation clears it.
 *   Enable   "A floating-point exception is generated any time a Cause bit and
 *            the corresponding Enable bit are set", including when CTC1 is
 *            what sets them.
 *   Flag     cumulative, and set only for exceptions that were NOT trapped:
 *            "When a floating-point exception is taken, the flag bits are not
 *            set by the hardware."
 *
 * And, for every trapped case: "When a floating-point exception is taken, no
 * results are stored, and the only state affected is the Cause bit." Each
 * operation below therefore preloads its destination register with a sentinel,
 * so "the result register was not modified" is a thing the test can check
 * rather than assume.
 *
 * The harness makes this survivable: exc_dispatch clears the FP Cause field
 * before returning, or an enabled exception would re-fire the moment the
 * handler ERETs.
 */

#include "fpu_common.h"

FPOP_S(op_add_s,  "add.s $f4, $f0, $f2")
FPOP_S(op_sub_s,  "sub.s $f4, $f0, $f2")
FPOP_S(op_mul_s,  "mul.s $f4, $f0, $f2")
FPOP_S(op_div_s,  "div.s $f4, $f0, $f2")
FPOP_S(op_sqrt_s, "sqrt.s $f4, $f0")

/* Shared assertions for "this operation trapped". */
static void check_trapped(struct fp_obs *o, u32 cause_bit)
{
    CHECK_EQ(o->exceptions, 1u);
    CHECK_EQ(o->excode, (u32)EXC_FPE);
    CHECK_EQ(o->cause & cause_bit, cause_bit);
    /* "When a floating-point exception is taken, the flag bits are not set by
     * the hardware; floating-point exception software is responsible for
     * setting these bits." */
    CHECK_EQ(o->flags & cause_bit, 0u);
    /* "When a floating-point exception is taken, no results are stored." */
    CHECK_EQ(o->result, SENTINEL_S);
}

/* ── one enabled exception at a time ──────────────────────────────────────── */

/* 0/0 is Invalid. With V enabled it must trap instead of delivering a NaN. */
static void t_trap_invalid(void)
{
    struct fp_obs o = observe_s(F_0, F_0, FCSR_ENABLE(FP_V), op_div_s);
    check_trapped(&o, FP_V);

    /* Magnitude subtraction of infinities is the other classic Invalid. */
    o = observe_s(F_INF, F_INF, FCSR_ENABLE(FP_V), op_sub_s);
    check_trapped(&o, FP_V);
}

static void t_trap_divide_by_zero(void)
{
    struct fp_obs o = observe_s(F_1, F_0, FCSR_ENABLE(FP_Z), op_div_s);
    check_trapped(&o, FP_Z);
}

/* Overflow sets Inexact as well — "This exception also sets the Inexact
 * exception and Flag bits" — so enabling either one is enough to trap. */
static void t_trap_overflow(void)
{
    struct fp_obs o = observe_s(F_MAX, F_MAX, FCSR_ENABLE(FP_O), op_mul_s);
    check_trapped(&o, FP_O);
    CHECK_EQ(o.cause & FP_I, (u32)FP_I);
}

static void t_trap_inexact(void)
{
    struct fp_obs o = observe_s(F_1, F_3, FCSR_ENABLE(FP_I), op_div_s);
    check_trapped(&o, FP_I);
}

/* An overflow with only Inexact enabled still traps: the Cause bit that
 * matches an enabled bit is what generates the exception, whichever it is. */
static void t_trap_overflow_via_inexact_enable(void)
{
    struct fp_obs o = observe_s(F_MAX, F_MAX, FCSR_ENABLE(FP_I), op_mul_s);
    check_trapped(&o, FP_I);
    CHECK_EQ(o.cause & FP_O, (u32)FP_O);
}

/* sqrt of a negative is Invalid, and traps like any other Invalid. */
static void t_trap_sqrt_negative(void)
{
    struct fp_obs o = observe_s(F_NEG2, F_0, FCSR_ENABLE(FP_V), op_sqrt_s);
    check_trapped(&o, FP_V);
}

/* ── the negative half: unenabled exceptions must not trap ────────────────── */

/*
 * The control for every test above. "For a floating-point operation that sets
 * only unenabled Cause bits, no exception occurs and the default result
 * defined by IEEE 754 is stored." Without this, a CPU that never traps at all
 * would look correct here, and one that always trapped would fail visibly.
 */
static void t_no_trap_when_disabled(void)
{
    struct fp_obs o;

    o = observe_s(F_0, F_0, 0, op_div_s);            /* Invalid */
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.flags & FP_V, (u32)FP_V);
    CHECK(is_nan_s(o.result));

    o = observe_s(F_1, F_0, 0, op_div_s);            /* Divide by zero */
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.flags & FP_Z, (u32)FP_Z);
    CHECK_EQ(o.result, F_INF);

    o = observe_s(F_MAX, F_MAX, 0, op_mul_s);        /* Overflow */
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.flags & (FP_O | FP_I), (u32)(FP_O | FP_I));
    CHECK_EQ(o.result, F_INF);                       /* Table 7-1, RN */

    o = observe_s(F_1, F_3, 0, op_div_s);            /* Inexact */
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.flags & FP_I, (u32)FP_I);
}

/* An Enable bit arms exactly one exception, not the others. */
static void t_enable_is_selective(void)
{
    struct fp_obs o;

    /* Invalid enabled, but this operation raises Divide-by-zero. */
    o = observe_s(F_1, F_0, FCSR_ENABLE(FP_V), op_div_s);
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.flags & FP_Z, (u32)FP_Z);
    CHECK_EQ(o.result, F_INF);

    /* Divide-by-zero enabled, but this operation raises Invalid. */
    o = observe_s(F_0, F_0, FCSR_ENABLE(FP_Z), op_div_s);
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.flags & FP_V, (u32)FP_V);
    CHECK(is_nan_s(o.result));

    /* Overflow enabled, but an exact operation raises nothing at all. */
    o = observe_s(F_2, F_3, FCSR_ENABLE(FP_O), op_add_s);
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.flags, 0u);
    CHECK_EQ(o.result, F_5);
}

/* ── CTC1 as an exception source ──────────────────────────────────────────── */

/*
 * "A floating-point operation that sets an enabled Cause bit forces an
 * immediate exception, as does setting both Cause and Enable bits with CTC1."
 *
 * This is the path a real kernel takes when it restores FP state, and it is
 * why software must clear the enabled Cause bits before returning from an FP
 * exception — which is exactly what exc_dispatch does for this suite.
 */
static void t_ctc1_with_cause_and_enable_traps(void)
{
    u32 arm = FCSR_ENABLE(FP_V) | ((u32)FP_V << FCSR_CAUSE_SHIFT);

    fcsr_reset();
    exc_clear();
    fcsr_set(arm);
    fcsr_reset();

    CHECK_EQ(exc.count, 1u);
    CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_FPE);
    CHECK_EQ(FCSR_CAUSE_OF(exc.fcsr) & FP_V, (u32)FP_V);
}

/* A Cause bit with no matching Enable bit is just a record, not a trap. */
static void t_ctc1_cause_without_enable_is_quiet(void)
{
    fcsr_reset();
    exc_clear();
    fcsr_set((u32)FP_V << FCSR_CAUSE_SHIFT);
    CHECK_NO_EXC();
    CHECK_EQ(fcsr_cause() & FP_V, (u32)FP_V);
    fcsr_reset();
}

/* ── Cause is per-instruction; Flags accumulate ───────────────────────────── */

/*
 * The distinction that makes Cause and Flag two fields rather than one:
 *
 *   "The Cause bits ... reflect the results of the most recently executed
 *    instruction."
 *   "The Flag bits are cumulative ... never cleared as a side effect of
 *    floating-point operations."
 *
 * So a clean operation after a dirty one leaves Cause zero and the Flag set.
 */
static void t_cause_is_per_instruction(void)
{
    fcsr_reset();
    w()[0] = F_1; w()[1] = F_0; w()[3] = SENTINEL_S;
    SYNC();
    op_div_s();                      /* 1/0 — raises Divide-by-zero */
    SYNC();
    CHECK_EQ(fcsr_cause() & FP_Z, (u32)FP_Z);
    CHECK_EQ(fcsr_flags() & FP_Z, (u32)FP_Z);

    w()[0] = F_2; w()[1] = F_3;
    SYNC();
    op_add_s();                      /* 2+3 — exact, raises nothing */
    SYNC();
    CHECK_EQ(fcsr_cause() & FP_Z, 0u);          /* rewritten by the add */
    CHECK_EQ(fcsr_flags() & FP_Z, (u32)FP_Z);   /* still cumulative     */
    fcsr_reset();
}

/*
 * "The Cause bits are written by each floating-point operation (but not by
 * load, store, or move operations)." A load or a move between a dirty
 * operation and the read of FCSR must therefore leave Cause alone.
 */
static void t_loads_and_moves_do_not_write_cause(void)
{
    u64 tmp;

    fcsr_reset();
    w()[0] = F_1; w()[1] = F_0; w()[3] = SENTINEL_S;
    SYNC();
    op_div_s();                      /* leaves Cause.Z set */
    SYNC();
    CHECK_EQ(fcsr_cause() & FP_Z, (u32)FP_Z);

    /* A load, a store, and a move — none of them an FP operation. */
    __asm__ __volatile__(AF "lwc1 $f6, 0(%1)\n\t"
                            "swc1 $f6, 16(%1)\n\t"
                            "mfc1 %0, $f6\n\t"
                            "mtc1 %0, $f8\n\t"
                            "mov.s $f10, $f6" Z
                         : "=r"(tmp) : "r"(w()) : "memory");
    CHECK_EQ(fcsr_cause() & FP_Z, (u32)FP_Z);
    fcsr_reset();
}

/* ── where the exception is reported ──────────────────────────────────────── */

/* EPC points at the FP instruction itself, like any other precise exception. */
static void t_epc_points_at_the_fp_instruction(void)
{
    u64 addr;

    w()[0] = F_1; w()[1] = F_0;
    SYNC();
    fcsr_set(FCSR_ENABLE(FP_Z));
    exc_clear();
    __asm__ __volatile__(AF "lwc1 $f0, 0(%1)\n\t"
                            "lwc1 $f2, 4(%1)\n\t"
                            DLA("%0", "1f")
                            "1:\n\t"
                            "div.s $f4, $f0, $f2" Z
                         : "=&r"(addr) : "r"(w()));
    fcsr_reset();

    CHECK_EQ(exc.count, 1u);
    CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_FPE);
    CHECK_EQ(exc.epc, addr);
    CHECK_EQ(exc.cause & CAUSE_BD, 0u);
}

/*
 * An FP trap in a branch delay slot reports the *branch* in EPC and sets
 * Cause.BD, so a handler can re-execute the pair. The harness's resume policy
 * already depends on this rule (it steps EPC by 8 when BD is set); this is the
 * test that says the rule holds for FP exceptions too.
 */
static void t_trap_in_a_branch_delay_slot(void)
{
    u64 branch;

    w()[0] = F_1; w()[1] = F_0;
    SYNC();
    fcsr_set(FCSR_ENABLE(FP_Z));
    exc_clear();
    __asm__ __volatile__(AF "lwc1 $f0, 0(%1)\n\t"
                            "lwc1 $f2, 4(%1)\n\t"
                            DLA("%0", "1f")
                            "1:\n\t"
                            "b 2f\n\t"
                            "div.s $f4, $f0, $f2\n\t"   /* delay slot */
                            "2:" Z
                         : "=&r"(branch) : "r"(w()));
    fcsr_reset();

    CHECK_EQ(exc.count, 1u);
    CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_FPE);
    CHECK_EQ(exc.cause & CAUSE_BD, (u32)CAUSE_BD);
    CHECK_EQ(exc.epc, branch);
}

static const struct test tests[] = {
    TEST("fpu/trap_invalid",          t_trap_invalid,                     CPU_ALL),
    TEST("fpu/trap_divide_by_zero",   t_trap_divide_by_zero,              CPU_ALL),
    TEST("fpu/trap_overflow",         t_trap_overflow,                    CPU_ALL),
    TEST("fpu/trap_inexact",          t_trap_inexact,                     CPU_ALL),
    TEST("fpu/trap_overflow_via_i",   t_trap_overflow_via_inexact_enable, CPU_ALL),
    TEST("fpu/trap_sqrt_negative",    t_trap_sqrt_negative,               CPU_ALL),
    TEST("fpu/trap_disabled_control", t_no_trap_when_disabled,            CPU_ALL),
    TEST("fpu/trap_enable_selective", t_enable_is_selective,              CPU_ALL),
    TEST("fpu/trap_ctc1_arms",        t_ctc1_with_cause_and_enable_traps, CPU_ALL),
    TEST("fpu/trap_ctc1_quiet",       t_ctc1_cause_without_enable_is_quiet, CPU_ALL),
    TEST("fpu/cause_per_instruction", t_cause_is_per_instruction,         CPU_ALL),
    TEST("fpu/cause_not_by_moves",    t_loads_and_moves_do_not_write_cause, CPU_ALL),
    TEST("fpu/trap_epc",              t_epc_points_at_the_fp_instruction, CPU_ALL),
    TEST("fpu/trap_delay_slot",       t_trap_in_a_branch_delay_slot,      CPU_ALL),
};

const struct test_group group_fpu_trap = {
    "fpu_trap", tests, sizeof(tests) / sizeof(tests[0])
};
