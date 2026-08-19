/* fpu_vectors — arithmetic and conversions checked against generated tables.
 *
 * The expectations come from tests/fpu/fpvectors.h, which gen/fpvectors.py
 * computes with exact rational arithmetic and cross-checks against the host's
 * own IEEE implementation before writing (`make vectors`). Nothing here was
 * recorded from IRIS, and nothing was worked out by hand — which is what lets
 * this file assert the *flags* as well as the results, exactly, for every
 * vector: a table of a hundred hand-computed Inexact bits would be a second
 * implementation of floating-point arithmetic, and the wrong one.
 *
 * The tables deliberately contain no denormal or NaN results. Those are not
 * arithmetic on an R4400 — they are Unimplemented Operation traps — and
 * fpu_denorm.c covers them with the manual quoted alongside.
 *
 * Conversions are the other half. One table per (format, integer width) gives
 * the result under all four rounding modes, which covers both CVT.W/CVT.L
 * driven by FCSR.RM *and* the four fixed-mode instructions: ROUND is RN,
 * TRUNC is RZ, CEIL is RP and FLOOR is RM, so the same column serves both and
 * a disagreement between them is visible.
 */

#include "fpu_common.h"
#include "fpvectors.h"

FPOP_S(op_add_s,  "add.s $f4, $f0, $f2")
FPOP_S(op_sub_s,  "sub.s $f4, $f0, $f2")
FPOP_S(op_mul_s,  "mul.s $f4, $f0, $f2")
FPOP_S(op_div_s,  "div.s $f4, $f0, $f2")
FPOP_S(op_sqrt_s, "sqrt.s $f4, $f0")

FPOP_D(op_add_d,  "add.d $f4, $f0, $f2")
FPOP_D(op_sub_d,  "sub.d $f4, $f0, $f2")
FPOP_D(op_mul_d,  "mul.d $f4, $f0, $f2")
FPOP_D(op_div_d,  "div.d $f4, $f0, $f2")
FPOP_D(op_sqrt_d, "sqrt.d $f4, $f0")

/* ── arithmetic ───────────────────────────────────────────────────────────── */

static void run_s2(const char *label, const struct fpvs2 *v, unsigned n,
                   void (*op)(void))
{
    unsigned i;
    for (i = 0; i < n; i++) {
        struct fp_obs o = observe_s(v[i].a, v[i].b, 0, op);
        CHECK_EQ_AT(label, i, o.result, v[i].r);
        CHECK_EQ_AT(label, i, o.flags, v[i].flags);
    }
}

static void run_d2(const char *label, const struct fpvd2 *v, unsigned n,
                   void (*op)(void))
{
    unsigned i;
    for (i = 0; i < n; i++) {
        struct fp_obs o = observe_d(v[i].a, v[i].b, 0, op);
        CHECK_EQ_AT(label, i, o.result_d, v[i].r);
        CHECK_EQ_AT(label, i, o.flags, v[i].flags);
    }
}

static void t_arith_vectors_single(void)
{
    run_s2("add.s", fpv_add_s, FPV_ADD_S_N, op_add_s);
    run_s2("sub.s", fpv_sub_s, FPV_SUB_S_N, op_sub_s);
    run_s2("mul.s", fpv_mul_s, FPV_MUL_S_N, op_mul_s);
    run_s2("div.s", fpv_div_s, FPV_DIV_S_N, op_div_s);
}

static void t_arith_vectors_double(void)
{
    run_d2("add.d", fpv_add_d, FPV_ADD_D_N, op_add_d);
    run_d2("sub.d", fpv_sub_d, FPV_SUB_D_N, op_sub_d);
    run_d2("mul.d", fpv_mul_d, FPV_MUL_D_N, op_mul_d);
    run_d2("div.d", fpv_div_d, FPV_DIV_D_N, op_div_d);
}

static void t_sqrt_vectors(void)
{
    unsigned i;
    for (i = 0; i < FPV_SQRT_S_N; i++) {
        struct fp_obs o = observe_s(fpv_sqrt_s[i].a, F_0, 0, op_sqrt_s);
        CHECK_EQ_AT("sqrt.s", i, o.result, fpv_sqrt_s[i].r);
        CHECK_EQ_AT("sqrt.s", i, o.flags, fpv_sqrt_s[i].flags);
    }
    for (i = 0; i < FPV_SQRT_D_N; i++) {
        struct fp_obs o = observe_d(fpv_sqrt_d[i].a, D_0, 0, op_sqrt_d);
        CHECK_EQ_AT("sqrt.d", i, o.result_d, fpv_sqrt_d[i].r);
        CHECK_EQ_AT("sqrt.d", i, o.flags, fpv_sqrt_d[i].flags);
    }
}

/* The four rounding modes, in FCSR.RM order. ROUND/TRUNC/CEIL/FLOOR hardwire
 * one each, in the same order. */
static const u32 rm_modes[4] = { FCSR_RM_RN, FCSR_RM_RZ, FCSR_RM_RP, FCSR_RM_RM };

/* ── rounding modes applied to arithmetic ─────────────────────────────────── */

/*
 * FCSR.RM is tested on conversions elsewhere; it governs arithmetic too. The
 * overflow rows are Table 7-1's default actions: round-to-nearest delivers an
 * infinity, round-to-zero the largest finite number, and the directed modes
 * one or the other according to the sign.
 */
static void run_rm_s(const char *label, const struct fpvrms *v, unsigned n,
                     void (*op)(void))
{
    unsigned i, m;
    for (i = 0; i < n; i++) {
        u32 want[4];
        want[0] = v[i].rn; want[1] = v[i].rz; want[2] = v[i].rp; want[3] = v[i].rm;
        for (m = 0; m < 4; m++) {
            struct fp_obs o = observe_s(v[i].a, v[i].b, rm_modes[m], op);
            CHECK_EQ_AT(label, i * 4 + m, o.result, want[m]);
        }
    }
}

static void run_rm_d(const char *label, const struct fpvrmd *v, unsigned n,
                     void (*op)(void))
{
    unsigned i, m;
    for (i = 0; i < n; i++) {
        u64 want[4];
        want[0] = v[i].rn; want[1] = v[i].rz; want[2] = v[i].rp; want[3] = v[i].rm;
        for (m = 0; m < 4; m++) {
            struct fp_obs o = observe_d(v[i].a, v[i].b, rm_modes[m], op);
            CHECK_EQ_AT(label, i * 4 + m, o.result_d, want[m]);
        }
    }
}

static void t_rounding_modes_arithmetic(void)
{
    run_rm_s("div.s rm", fpv_rm_div_s, FPV_RM_DIV_S_N, op_div_s);
    run_rm_s("mul.s rm", fpv_rm_mul_s, FPV_RM_MUL_S_N, op_mul_s);
    run_rm_d("div.d rm", fpv_rm_div_d, FPV_RM_DIV_D_N, op_div_d);
    run_rm_d("mul.d rm", fpv_rm_mul_d, FPV_RM_MUL_D_N, op_mul_d);
    fcsr_reset();
}

/* ── conversions to integer ───────────────────────────────────────────────── */

/* Source at scratch[0]; a word result lands at scratch+16, a doubleword result
 * at the same address read as a doubleword. */
#define CVT_TO_W(name, load, mnem, ptr)                                    \
    static void name(void)                                                 \
    {                                                                      \
        __asm__ __volatile__(AF load " $f0, 0(%0)\n\t"                     \
                                mnem " $f4, $f0\n\t"                       \
                                "swc1 $f4, 16(%0)" Z                       \
                             :: "r"(ptr) : "memory");                      \
    }

#define CVT_TO_L(name, load, mnem, ptr)                                    \
    static void name(void)                                                 \
    {                                                                      \
        __asm__ __volatile__(AF load " $f0, 0(%0)\n\t"                     \
                                mnem " $f4, $f0\n\t"                       \
                                "sdc1 $f4, 16(%0)" Z                       \
                             :: "r"(ptr) : "memory");                      \
    }

CVT_TO_W(cvt_w_s,   "lwc1", "cvt.w.s",   w())
CVT_TO_W(round_w_s, "lwc1", "round.w.s", w())
CVT_TO_W(trunc_w_s, "lwc1", "trunc.w.s", w())
CVT_TO_W(ceil_w_s,  "lwc1", "ceil.w.s",  w())
CVT_TO_W(floor_w_s, "lwc1", "floor.w.s", w())

CVT_TO_W(cvt_w_d,   "ldc1", "cvt.w.d",   d())
CVT_TO_W(round_w_d, "ldc1", "round.w.d", d())
CVT_TO_W(trunc_w_d, "ldc1", "trunc.w.d", d())
CVT_TO_W(ceil_w_d,  "ldc1", "ceil.w.d",  d())
CVT_TO_W(floor_w_d, "ldc1", "floor.w.d", d())

CVT_TO_L(cvt_l_s,   "lwc1", "cvt.l.s",   w())
CVT_TO_L(round_l_s, "lwc1", "round.l.s", w())
CVT_TO_L(trunc_l_s, "lwc1", "trunc.l.s", w())
CVT_TO_L(ceil_l_s,  "lwc1", "ceil.l.s",  w())
CVT_TO_L(floor_l_s, "lwc1", "floor.l.s", w())

CVT_TO_L(cvt_l_d,   "ldc1", "cvt.l.d",   d())
CVT_TO_L(round_l_d, "ldc1", "round.l.d", d())
CVT_TO_L(trunc_l_d, "ldc1", "trunc.l.d", d())
CVT_TO_L(ceil_l_d,  "ldc1", "ceil.l.d",  d())
CVT_TO_L(floor_l_d, "ldc1", "floor.l.d", d())

static s32 run_cvt_w(u32 rm, void (*op)(void))
{
    fcsr_set(rm);
    SYNC();
    op();
    SYNC();
    return (s32)w()[4];
}

static s64 run_cvt_l(u32 rm, void (*op)(void))
{
    fcsr_set(rm);
    SYNC();
    op();
    SYNC();
    return (s64)d()[2];
}

/*
 * CVT.W.S follows FCSR.RM; ROUND/TRUNC/CEIL/FLOOR ignore it. Both are checked
 * against the same generated column, and the fixed-mode instructions are run
 * with FCSR.RM set to something contradictory so that an implementation which
 * quietly consults it fails here.
 */
static void t_convert_to_word_single(void)
{
    unsigned i, m;
    void (*const fixed[4])(void) = { round_w_s, trunc_w_s, ceil_w_s, floor_w_s };

    for (i = 0; i < FPV_CVT_S32_N; i++) {
        const struct fpvcvt_s32 *v = &fpv_cvt_s32[i];
        s32 want[4];
        want[0] = v->rn; want[1] = v->rz; want[2] = v->rp; want[3] = v->rm;

        w()[0] = v->in;
        SYNC();
        for (m = 0; m < 4; m++) {
            CHECK_EQ_AT("cvt.w.s", i * 4 + m,
                        (s64)run_cvt_w(rm_modes[m], cvt_w_s), (s64)want[m]);
            /* The fixed-mode form, with RM deliberately wrong. */
            CHECK_EQ_AT("fixed.w.s", i * 4 + m,
                        (s64)run_cvt_w(rm_modes[(m + 1) & 3], fixed[m]),
                        (s64)want[m]);
        }
    }
    fcsr_reset();
}

static void t_convert_to_word_double(void)
{
    unsigned i, m;
    void (*const fixed[4])(void) = { round_w_d, trunc_w_d, ceil_w_d, floor_w_d };

    for (i = 0; i < FPV_CVT_D32_N; i++) {
        const struct fpvcvt_d32 *v = &fpv_cvt_d32[i];
        s32 want[4];
        want[0] = v->rn; want[1] = v->rz; want[2] = v->rp; want[3] = v->rm;

        d()[0] = v->in;
        SYNC();
        for (m = 0; m < 4; m++) {
            CHECK_EQ_AT("cvt.w.d", i * 4 + m,
                        (s64)run_cvt_w(rm_modes[m], cvt_w_d), (s64)want[m]);
            CHECK_EQ_AT("fixed.w.d", i * 4 + m,
                        (s64)run_cvt_w(rm_modes[(m + 1) & 3], fixed[m]),
                        (s64)want[m]);
        }
    }
    fcsr_reset();
}

/* The 64-bit forms — CVT.L and friends, which nothing in the suite reached. */
static void t_convert_to_long_single(void)
{
    unsigned i, m;
    void (*const fixed[4])(void) = { round_l_s, trunc_l_s, ceil_l_s, floor_l_s };

    for (i = 0; i < FPV_CVT_S64_N; i++) {
        const struct fpvcvt_s64 *v = &fpv_cvt_s64[i];
        s64 want[4];
        want[0] = v->rn; want[1] = v->rz; want[2] = v->rp; want[3] = v->rm;

        w()[0] = v->in;
        SYNC();
        for (m = 0; m < 4; m++) {
            CHECK_EQ_AT("cvt.l.s", i * 4 + m, run_cvt_l(rm_modes[m], cvt_l_s),
                        want[m]);
            CHECK_EQ_AT("fixed.l.s", i * 4 + m,
                        run_cvt_l(rm_modes[(m + 1) & 3], fixed[m]), want[m]);
        }
    }
    fcsr_reset();
}

static void t_convert_to_long_double(void)
{
    unsigned i, m;
    void (*const fixed[4])(void) = { round_l_d, trunc_l_d, ceil_l_d, floor_l_d };

    for (i = 0; i < FPV_CVT_D64_N; i++) {
        const struct fpvcvt_d64 *v = &fpv_cvt_d64[i];
        s64 want[4];
        want[0] = v->rn; want[1] = v->rz; want[2] = v->rp; want[3] = v->rm;

        d()[0] = v->in;
        SYNC();
        for (m = 0; m < 4; m++) {
            CHECK_EQ_AT("cvt.l.d", i * 4 + m, run_cvt_l(rm_modes[m], cvt_l_d),
                        want[m]);
            CHECK_EQ_AT("fixed.l.d", i * 4 + m,
                        run_cvt_l(rm_modes[(m + 1) & 3], fixed[m]), want[m]);
        }
    }
    fcsr_reset();
}

/* ── conversions from integer ─────────────────────────────────────────────── */

/* Integer source at scratch[0] (word) or scratch as a doubleword; results to
 * scratch+16. */
static void cvt_s_w(void)
{
    __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                            "cvt.s.w $f4, $f0\n\t"
                            "swc1 $f4, 16(%0)" Z :: "r"(w()) : "memory");
}
static void cvt_d_w(void)
{
    __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                            "cvt.d.w $f4, $f0\n\t"
                            "sdc1 $f4, 16(%0)" Z :: "r"(w()) : "memory");
}
static void cvt_s_l(void)
{
    __asm__ __volatile__(AF "ldc1 $f0, 0(%0)\n\t"
                            "cvt.s.l $f4, $f0\n\t"
                            "swc1 $f4, 16(%0)" Z :: "r"(d()) : "memory");
}
static void cvt_d_l(void)
{
    __asm__ __volatile__(AF "ldc1 $f0, 0(%0)\n\t"
                            "cvt.d.l $f4, $f0\n\t"
                            "sdc1 $f4, 16(%0)" Z :: "r"(d()) : "memory");
}

/*
 * A 32-bit integer always fits a double exactly, so cvt.d.w never rounds; a
 * single has 24 bits of significand, so cvt.s.w does — 0x01000001 is the
 * smallest positive case, and it must set Inexact and nothing else.
 */
static void t_convert_from_word(void)
{
    unsigned i;

    for (i = 0; i < FPV_FROM_I32_N; i++) {
        const struct fpvfromi32 *v = &fpv_from_i32[i];

        w()[0] = (u32)v->in; w()[4] = 0;
        SYNC();
        fcsr_set(0);
        cvt_s_w();
        SYNC();
        CHECK_EQ_AT("cvt.s.w", i, w()[4], v->s);
        CHECK_EQ_AT("cvt.s.w flags", i, fcsr_flags(), v->sflags);

        w()[0] = (u32)v->in; d()[2] = 0;
        SYNC();
        fcsr_set(0);
        cvt_d_w();
        SYNC();
        CHECK_EQ_AT("cvt.d.w", i, d()[2], v->d);
        CHECK_EQ_AT("cvt.d.w flags", i, fcsr_flags(), v->dflags);
    }
    fcsr_reset();
}

static void t_convert_from_long(void)
{
    unsigned i;

    for (i = 0; i < FPV_FROM_I64_N; i++) {
        const struct fpvfromi64 *v = &fpv_from_i64[i];

        d()[0] = (u64)v->in; w()[4] = 0;
        SYNC();
        fcsr_set(0);
        cvt_s_l();
        SYNC();
        CHECK_EQ_AT("cvt.s.l", i, w()[4], v->s);
        CHECK_EQ_AT("cvt.s.l flags", i, fcsr_flags(), v->sflags);

        d()[0] = (u64)v->in; d()[2] = 0;
        SYNC();
        fcsr_set(0);
        cvt_d_l();
        SYNC();
        CHECK_EQ_AT("cvt.d.l", i, d()[2], v->d);
        CHECK_EQ_AT("cvt.d.l flags", i, fcsr_flags(), v->dflags);
    }
    fcsr_reset();
}

static const struct test tests[] = {
    TEST("fpu/vec_arith_single",   t_arith_vectors_single,   CPU_ALL),
    TEST("fpu/vec_arith_double",   t_arith_vectors_double,   CPU_ALL),
    TEST("fpu/vec_sqrt",           t_sqrt_vectors,           CPU_ALL),
    TEST("fpu/vec_rounding_modes", t_rounding_modes_arithmetic, CPU_ALL),
    TEST("fpu/vec_cvt_w_s",        t_convert_to_word_single, CPU_ALL),
    TEST("fpu/vec_cvt_w_d",        t_convert_to_word_double, CPU_ALL),
    TEST("fpu/vec_cvt_l_s",        t_convert_to_long_single, CPU_ALL),
    TEST("fpu/vec_cvt_l_d",        t_convert_to_long_double, CPU_ALL),
    TEST("fpu/vec_cvt_from_w",     t_convert_from_word,      CPU_ALL),
    TEST("fpu/vec_cvt_from_l",     t_convert_from_long,      CPU_ALL),
};

const struct test_group group_fpu_vectors = {
    "fpu_vectors", tests, sizeof(tests) / sizeof(tests[0])
};
