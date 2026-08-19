/* fpu_compare — all sixteen C.cond predicates, in both formats.
 *
 * fpu.c covers six of the sixteen, all from the non-signalling half. The half
 * that is missing is the interesting one: predicates 8..15 raise Invalid on
 * *any* NaN operand, quiet ones included, where 0..7 raise it only for a
 * signalling NaN. That distinction is a single bit of the instruction — the
 * high bit of the cond field — and it is live code with no coverage.
 *
 * The expectations are not a table of 64 hand-written answers. They come from
 * the manual's own definition of the instruction (Appendix B, C.cond.fmt):
 *
 *     if NaN(fs) or NaN(ft) then
 *         less <- false; equal <- false; unordered <- true
 *         if cond3 then signal InvalidOperationException
 *     else
 *         less <- fs < ft; equal <- fs = ft; unordered <- false
 *     condition <- (cond2 and less) or (cond1 and equal) or (cond0 and unordered)
 *
 * so `predicted()` below is that last line, and the table of predicates is
 * just the sixteen mnemonics in cond order. Table B-2 in the manual is the
 * same thing written out, and agrees with it — including the column showing
 * that "greater than" is always false, which is why sixteen comparisons cover
 * all thirty-two relations (BC1F negates the other half).
 */

#include "fpu_common.h"

#define CMP_FN_S(name, mnem)                                               \
    static void name(void)                                                 \
    {                                                                      \
        __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"                      \
                                "lwc1 $f2, 4(%0)\n\t"                      \
                                mnem " $f0, $f2\n\t"                       \
                                "nop" Z :: "r"(w()));                      \
    }

#define CMP_FN_D(name, mnem)                                               \
    static void name(void)                                                 \
    {                                                                      \
        __asm__ __volatile__(AF "ldc1 $f0, 0(%0)\n\t"                      \
                                "ldc1 $f2, 8(%0)\n\t"                      \
                                mnem " $f0, $f2\n\t"                       \
                                "nop" Z :: "r"(d()));                      \
    }

CMP_FN_S(cs_f,    "c.f.s")     CMP_FN_D(cd_f,    "c.f.d")
CMP_FN_S(cs_un,   "c.un.s")    CMP_FN_D(cd_un,   "c.un.d")
CMP_FN_S(cs_eq,   "c.eq.s")    CMP_FN_D(cd_eq,   "c.eq.d")
CMP_FN_S(cs_ueq,  "c.ueq.s")   CMP_FN_D(cd_ueq,  "c.ueq.d")
CMP_FN_S(cs_olt,  "c.olt.s")   CMP_FN_D(cd_olt,  "c.olt.d")
CMP_FN_S(cs_ult,  "c.ult.s")   CMP_FN_D(cd_ult,  "c.ult.d")
CMP_FN_S(cs_ole,  "c.ole.s")   CMP_FN_D(cd_ole,  "c.ole.d")
CMP_FN_S(cs_ule,  "c.ule.s")   CMP_FN_D(cd_ule,  "c.ule.d")
CMP_FN_S(cs_sf,   "c.sf.s")    CMP_FN_D(cd_sf,   "c.sf.d")
CMP_FN_S(cs_ngle, "c.ngle.s")  CMP_FN_D(cd_ngle, "c.ngle.d")
CMP_FN_S(cs_seq,  "c.seq.s")   CMP_FN_D(cd_seq,  "c.seq.d")
CMP_FN_S(cs_ngl,  "c.ngl.s")   CMP_FN_D(cd_ngl,  "c.ngl.d")
CMP_FN_S(cs_lt,   "c.lt.s")    CMP_FN_D(cd_lt,   "c.lt.d")
CMP_FN_S(cs_nge,  "c.nge.s")   CMP_FN_D(cd_nge,  "c.nge.d")
CMP_FN_S(cs_le,   "c.le.s")    CMP_FN_D(cd_le,   "c.le.d")
CMP_FN_S(cs_ngt,  "c.ngt.s")   CMP_FN_D(cd_ngt,  "c.ngt.d")

/* In cond order, so the index *is* the cond field. */
static const struct {
    const char *name;
    void (*s)(void);
    void (*d)(void);
} preds[16] = {
    { "f",    cs_f,    cd_f    }, { "un",   cs_un,   cd_un   },
    { "eq",   cs_eq,   cd_eq   }, { "ueq",  cs_ueq,  cd_ueq  },
    { "olt",  cs_olt,  cd_olt  }, { "ult",  cs_ult,  cd_ult  },
    { "ole",  cs_ole,  cd_ole  }, { "ule",  cs_ule,  cd_ule  },
    { "sf",   cs_sf,   cd_sf   }, { "ngle", cs_ngle, cd_ngle },
    { "seq",  cs_seq,  cd_seq  }, { "ngl",  cs_ngl,  cd_ngl  },
    { "lt",   cs_lt,   cd_lt   }, { "nge",  cs_nge,  cd_nge  },
    { "le",   cs_le,   cd_le   }, { "ngt",  cs_ngt,  cd_ngt  },
};

/* The four relations a pair of operands can be in. "Greater than" is a
 * relation the condition never selects — no cond bit corresponds to it. */
#define REL_LESS       0
#define REL_EQUAL      1
#define REL_GREATER    2
#define REL_UNORDERED  3

static u32 predicted(u32 cond, int rel)
{
    switch (rel) {
    case REL_LESS:      return (cond >> 2) & 1u;
    case REL_EQUAL:     return (cond >> 1) & 1u;
    case REL_GREATER:   return 0u;
    default:            return cond & 1u;
    }
}

/* Run one compare and return the condition bit. FCSR is cleared first, so the
 * bit read back is this comparison's own answer. */
static u32 compare_s(u32 a, u32 b, void (*op)(void))
{
    w()[0] = a; w()[1] = b;
    SYNC();
    fcsr_set(0);
    op();
    return (fcsr() & FCSR_CC0) ? 1u : 0u;
}

static u32 compare_d(u64 a, u64 b, void (*op)(void))
{
    d()[0] = a; d()[1] = b;
    SYNC();
    fcsr_set(0);
    op();
    return (fcsr() & FCSR_CC0) ? 1u : 0u;
}

/* ── the truth table ──────────────────────────────────────────────────────── */

static void t_all_predicates_single(void)
{
    static const u32 lhs[4] = { F_1,   F_1, F_2, F_QNAN };
    static const u32 rhs[4] = { F_2,   F_1, F_1, F_1    };
    /*                       less  equal greater unordered */
    unsigned c, rel;

    exc_clear();
    for (c = 0; c < 16; c++) {
        for (rel = 0; rel < 4; rel++) {
            u32 got = compare_s(lhs[rel], rhs[rel], preds[c].s);
            CHECK_EQ_AT("cond*4+rel", c * 4 + rel, got, predicted(c, rel));
        }
    }
    fcsr_reset();
    /* A signalling predicate on a NaN raises Invalid, but with the enable bit
     * clear that must not have trapped anywhere above. */
    CHECK_NO_EXC();
}

static void t_all_predicates_double(void)
{
    static const u64 lhs[4] = { D_1, D_1, D_2, D_QNAN };
    static const u64 rhs[4] = { D_2, D_1, D_1, D_1    };
    unsigned c, rel;

    exc_clear();
    for (c = 0; c < 16; c++) {
        for (rel = 0; rel < 4; rel++) {
            u32 got = compare_d(lhs[rel], rhs[rel], preds[c].d);
            CHECK_EQ_AT("cond*4+rel", c * 4 + rel, got, predicted(c, rel));
        }
    }
    fcsr_reset();
    CHECK_NO_EXC();
}

/* ── which predicates signal ──────────────────────────────────────────────── */

/*
 * "If one of the values is a Not a Number (NaN), and the high-order bit of the
 * cond field is set, an invalid operation exception is taken." A *quiet* NaN,
 * which is the whole point: the eight non-signalling predicates exist so that
 * NaN-safe code can compare without raising anything.
 */
static void t_signalling_predicates_and_quiet_nan(void)
{
    unsigned c;

    exc_clear();
    for (c = 0; c < 16; c++) {
        u32 want = (c & 8u) ? (u32)FP_V : 0u;
        (void)compare_s(F_QNAN, F_1, preds[c].s);
        CHECK_EQ_AT("cond", c, fcsr_flags() & FP_V, want);
    }
    /* Nothing may trap: the Invalid Enable bit is clear throughout. */
    CHECK_NO_EXC();
    fcsr_reset();
}

/*
 * A *signalling* NaN raises Invalid on every predicate, signalling or not —
 * Table 7-2, "Signaling NaN source". This is the case that distinguishes the
 * two NaN kinds, and it is the reason c.eq.s is not simply "safe".
 */
static void t_signalling_nan_raises_invalid_on_any_predicate(void)
{
    unsigned c;

    exc_clear();
    for (c = 0; c < 16; c++) {
        (void)compare_s(F_SNAN, F_1, preds[c].s);
        CHECK_EQ_AT("cond", c, fcsr_flags() & FP_V, (u32)FP_V);
    }
    CHECK_NO_EXC();

    /* And in double precision. */
    for (c = 0; c < 16; c++) {
        (void)compare_d(D_SNAN, D_1, preds[c].d);
        CHECK_EQ_AT("cond", c, fcsr_flags() & FP_V, (u32)FP_V);
    }
    CHECK_NO_EXC();
    fcsr_reset();
}

/*
 * With Invalid enabled, a signalling predicate on a NaN traps — and because
 * "the only state affected is the Cause bit", the condition bit must survive
 * the attempt. It is preloaded here through CTC1, which the manual names as
 * the one non-compare instruction allowed to write it.
 */
static void t_signalling_compare_traps(void)
{
    exc_clear();
    w()[0] = F_QNAN; w()[1] = F_1;
    SYNC();
    fcsr_set(FCSR_ENABLE(FP_V) | FCSR_CC0);      /* armed, condition preset */
    cs_lt();
    SYNC();

    CHECK_EQ(exc.count, 1u);
    CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_FPE);
    CHECK_EQ(FCSR_CAUSE_OF(exc.fcsr) & FP_V, (u32)FP_V);
    CHECK_EQ(fcsr() & FCSR_CC0, (u32)FCSR_CC0);  /* the result was not stored */
    /* "When a floating-point exception is taken, the flag bits are not set by
     * the hardware" — the same rule the fpu/trap_* tests check for arithmetic,
     * asked here of a compare. */
    CHECK_EQ(fcsr_flags() & FP_V, 0u);

    /* The same comparison with a non-signalling predicate does not trap, and
     * does write its answer. */
    exc_clear();
    fcsr_set(FCSR_ENABLE(FP_V) | FCSR_CC0);
    cs_olt();
    CHECK_NO_EXC();
    CHECK_EQ(fcsr() & FCSR_CC0, 0u);             /* unordered: olt is false */
    fcsr_reset();
}

/* ── odds and ends the double-precision path has never been asked ─────────── */

/* "Comparisons ignore the sign of zero, so +0 = -0." */
static void t_double_zero_signs(void)
{
    fcsr_reset();
    CHECK_EQ(compare_d(D_0, D_NEG0, cd_eq), 1u);
    CHECK_EQ(compare_d(D_NEG0, D_0, cd_eq), 1u);
    CHECK_EQ(compare_d(D_NEG0, D_0, cd_olt), 0u);
    CHECK_EQ(compare_d(D_0, D_NEG0, cd_ole), 1u);
    /* Infinities compare as ordinary ordered values. */
    CHECK_EQ(compare_d(D_NEGINF, D_INF, cd_olt), 1u);
    CHECK_EQ(compare_d(D_INF, D_INF, cd_eq), 1u);
}

/*
 * "Bit 23 is affected only by compare and Move Control To FPU instructions."
 * An arithmetic operation between two compares must leave the condition
 * alone — otherwise the one-instruction delay slot between C.cond and BC1T
 * could not be filled with anything useful.
 */
static void t_condition_bit_only_written_by_compare(void)
{
    CHECK_EQ(compare_s(F_1, F_1, cs_eq), 1u);    /* condition now true */

    w()[0] = F_2; w()[1] = F_3; w()[3] = SENTINEL_S;
    SYNC();
    __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                            "lwc1 $f2, 4(%0)\n\t"
                            "add.s $f4, $f0, $f2\n\t"
                            "sub.s $f6, $f0, $f2\n\t"
                            "mul.s $f8, $f0, $f2\n\t"
                            "swc1 $f4, 8(%0)" Z :: "r"(w()) : "memory");
    SYNC();
    CHECK_EQ(w()[2], F_5);
    CHECK_EQ(fcsr() & FCSR_CC0, (u32)FCSR_CC0);  /* still true */

    CHECK_EQ(compare_s(F_1, F_2, cs_eq), 0u);    /* and a compare can clear it */
    fcsr_reset();
}

static const struct test tests[] = {
    TEST("fpu/cmp_all_predicates_s", t_all_predicates_single,             CPU_ALL),
    TEST("fpu/cmp_all_predicates_d", t_all_predicates_double,             CPU_ALL),
    TEST("fpu/cmp_signalling_qnan",  t_signalling_predicates_and_quiet_nan, CPU_ALL),
    TEST("fpu/cmp_snan_any_pred",    t_signalling_nan_raises_invalid_on_any_predicate, CPU_ALL),
    TEST("fpu/cmp_trap_on_signal",   t_signalling_compare_traps,          CPU_ALL),
    TEST("fpu/cmp_double_zeros",     t_double_zero_signs,                 CPU_ALL),
    TEST("fpu/cmp_c_bit_ownership",  t_condition_bit_only_written_by_compare, CPU_ALL),
};

const struct test_group group_fpu_compare = {
    "fpu_compare", tests, sizeof(tests) / sizeof(tests[0])
};
