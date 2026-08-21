/*
 * fpu.c — floating-point kernels.
 *
 * All checksums fold in the raw bit patterns of the results, so a one-ulp
 * disagreement with the host reference fails rather than passing quietly. That
 * is the point: rules/testing/fpu-exception-model-vs-r4400.md records that the
 * FPU flag model is the part of this emulator known to be imperfect, and an
 * instruction-level test suite exercises operations one at a time, in
 * isolation, with clean state. These kernels run millions of operations with
 * whatever state the previous million left behind, which is where a rounding
 * or a denormal path that is subtly wrong actually shows up.
 */

#include "benchlib.h"
#include "bmath.h"

/* ── fpu/scalar_d — dependent double-precision add/mul chain ──────────────── */

static double dchain(u32 rounds, double seed)
{
    double a = seed, b = seed * 0.5 + 1.0, c = 1.0 / 3.0, d = 0.7071067811865476;
    u32 i;
    for (i = 0; i < rounds; i++) {
        a = a * c + d;
        b = b * d - c;
        a = a - b * 0.25;
        b = b + a * 0.125;
        c = c * 1.0000001 - 1.0e-9;
        d = d * 0.9999999 + 1.0e-9;
        if (a > 1.0e12 || a < -1.0e12) a *= 1.0e-12;
        if (b > 1.0e12 || b < -1.0e12) b *= 1.0e-12;
    }
    return a + b + c + d;
}

static u64 v_scalar_d(void) { return cksum_f64(CKSUM_INIT, dchain(4096, 1.0)); }
static u64 r_scalar_d(u32 n) { double r = dchain(n, 1.0); SINK((int)(r != 0.0)); return (u64)n * 8; }

/* ── fpu/scalar_s — the same shape in single precision ────────────────────── */

static float fchain(u32 rounds, float seed)
{
    float a = seed, b = seed * 0.5f + 1.0f, c = 1.0f / 3.0f, d = 0.70710678f;
    u32 i;
    for (i = 0; i < rounds; i++) {
        a = a * c + d;
        b = b * d - c;
        a = a - b * 0.25f;
        b = b + a * 0.125f;
        c = c * 1.0000001f - 1.0e-9f;
        d = d * 0.9999999f + 1.0e-9f;
        if (a > 1.0e12f || a < -1.0e12f) a *= 1.0e-12f;
        if (b > 1.0e12f || b < -1.0e12f) b *= 1.0e-12f;
    }
    return a + b + c + d;
}

static u64 v_scalar_s(void) { return cksum_f32(CKSUM_INIT, fchain(4096, 1.0f)); }
static u64 r_scalar_s(u32 n) { float r = fchain(n, 1.0f); SINK((int)(r != 0.0f)); return (u64)n * 8; }

/* ── fpu/divsqrt — the long-latency FP units ──────────────────────────────── */

static double divsqrt(u32 rounds, double seed)
{
    double a = seed + 1.0, acc = 0.0;
    u32 i;
    for (i = 0; i < rounds; i++) {
        double r = b_sqrt(a);
        acc += 1.0 / r;
        a = a * 1.0000003 + 0.5;
        acc += r / (a + 1.0);
        if (a > 1.0e18) a = seed + 1.0;
    }
    return acc + a;
}

static u64 v_divsqrt(void) { return cksum_f64(CKSUM_INIT, divsqrt(2048, 2.0)); }
static u64 r_divsqrt(u32 n) { double r = divsqrt(n, 2.0); SINK((int)(r != 0.0)); return (u64)n * 5; }

/* ── fpu/transcend — sin/cos/exp/log/atan over the suite's own libm ───────── */

static double transcend(u32 rounds, double seed)
{
    double x = seed, acc = 0.0;
    u32 i;
    for (i = 0; i < rounds; i++) {
        acc += b_sin(x) * b_cos(x * 0.5);
        acc += b_exp(x * 0.001);
        acc += b_log(x + 2.0);
        acc += b_atan(x * 0.25);
        x += 0.0009765625;                  /* 2^-10: exact, so no drift */
        if (x > 100.0) x -= 100.0;
    }
    return acc;
}

static u64 v_transcend(void) { return cksum_f64(CKSUM_INIT, transcend(512, 0.5)); }
static u64 r_transcend(u32 n) { double r = transcend(n, 0.5); SINK((int)(r != 0.0)); return (u64)n * 5; }

/* ══ fpu/whetstone ═══════════════════════════════════════════════════════════
 *
 * The Curnow/Wichmann module structure with the conventional weights: modules
 * 1-4, 6-9 and 11 in order, and the same balance of scalar arithmetic, array
 * traffic, integer work, procedure calls and transcendentals that has made it
 * a useful floating-point mix for four decades.
 *
 * Reported in **loops per second, not MWIPS**. Converting needs the "Whetstone
 * instructions per loop" constant from a reference implementation, and that
 * constant is not something this suite can verify — a derived MWIPS resting on
 * an unchecked factor of a thousand would look authoritative without being so.
 * Loops per second is exact, and it is what comparisons between cells use.
 *
 * Two departures from a reference implementation either way. The
 * transcendentals in modules 7 and 11 are bmath.h's rather than a libm's,
 * because a libm difference between host and guest would be scored as an
 * emulator fault. And there is no self-timing loop; the harness times it.
 */

static double whet_t  = 0.499975;
static double whet_t1 = 0.50025;
static double whet_t2 = 2.0;

static void whet_pa(double e[4])
{
    int j = 0;
    do {
        e[0] = (e[0] + e[1] + e[2] - e[3]) * whet_t;
        e[1] = (e[0] + e[1] - e[2] + e[3]) * whet_t;
        e[2] = (e[0] - e[1] + e[2] + e[3]) * whet_t;
        e[3] = (-e[0] + e[1] + e[2] + e[3]) / whet_t2;
        j += 1;
    } while (j < 6);
}

static void whet_p3(double *x, double *y, double *z)
{
    double x1 = *x, y1 = *y;
    x1 = whet_t * (x1 + y1);
    y1 = whet_t * (x1 + y1);
    *z = (x1 + y1) / whet_t2;
}

static void whet_p0(double e1[5], int j, int k, int l)
{
    e1[j] = e1[k];
    e1[k] = e1[l];
    e1[l] = e1[j];
}

/* The classic weights, per loop. */
#define WN1   0
#define WN2  12
#define WN3  14
#define WN4 345
#define WN6 210
#define WN7  32
#define WN8 899
#define WN9 616
#define WN11 93

static double whetstone(u32 loops)
{
    double x1, x2, x3, x4, x, y, z;
    double e1[5];
    int j, k, l, n;
    u32 loop;
    double acc = 0.0;

    for (loop = 0; loop < loops; loop++) {
        /* Module 1: simple identifiers */
        x1 = 1.0; x2 = -1.0; x3 = -1.0; x4 = -1.0;
        for (n = 0; n < WN1; n++) {
            x1 = (x1 + x2 + x3 - x4) * whet_t;
            x2 = (x1 + x2 - x3 + x4) * whet_t;
            x3 = (x1 - x2 + x3 + x4) * whet_t;
            x4 = (-x1 + x2 + x3 + x4) * whet_t;
        }

        /* Module 2: array elements */
        e1[0] = 1.0; e1[1] = -1.0; e1[2] = -1.0; e1[3] = -1.0;
        for (n = 0; n < WN2; n++) {
            e1[0] = (e1[0] + e1[1] + e1[2] - e1[3]) * whet_t;
            e1[1] = (e1[0] + e1[1] - e1[2] + e1[3]) * whet_t;
            e1[2] = (e1[0] - e1[1] + e1[2] + e1[3]) * whet_t;
            e1[3] = (-e1[0] + e1[1] + e1[2] + e1[3]) * whet_t;
        }

        /* Module 3: array as parameter */
        for (n = 0; n < WN3; n++) whet_pa(e1);

        /* Module 4: conditional jumps */
        j = 1;
        for (n = 0; n < WN4; n++) {
            j = (j == 1) ? 2 : 3;
            j = (j > 2) ? 0 : 1;
            j = (j < 1) ? 1 : 0;
        }

        /* Module 6: integer arithmetic */
        j = 1; k = 2; l = 3;
        for (n = 0; n < WN6; n++) {
            j = j * (k - j) * (l - k);
            k = l * k - (l - j) * k;
            l = (l - k) * (k + j);
            e1[l - 2] = (double)(j + k + l);
            e1[k - 2] = (j * k * l) < 0 ? -(double)(j * k * l) : (double)(j * k * l);
        }

        /* Module 7: trigonometric functions */
        x = 0.5; y = 0.5;
        for (n = 0; n < WN7; n++) {
            x = whet_t * b_atan(whet_t2 * b_sin(x) * b_cos(x) /
                                (b_cos(x + y) + b_cos(x - y) - 1.0));
            y = whet_t * b_atan(whet_t2 * b_sin(y) * b_cos(y) /
                                (b_cos(x + y) + b_cos(x - y) - 1.0));
        }

        /* Module 8: procedure calls */
        x = 1.0; y = 1.0; z = 1.0;
        for (n = 0; n < WN8; n++) whet_p3(&x, &y, &z);

        /* Module 9: array references */
        j = 0; k = 1; l = 2;
        e1[0] = 1.0; e1[1] = 2.0; e1[2] = 3.0;
        for (n = 0; n < WN9; n++) whet_p0(e1, j, k, l);

        /* Module 11: standard functions */
        x = 0.75;
        for (n = 0; n < WN11; n++) x = b_sqrt(b_exp(b_log(x) / whet_t1));

        acc += x1 + x2 + x3 + x4 + e1[0] + e1[1] + e1[2] + e1[3] + x + y + z;
    }
    return acc;
}

static u64 v_whetstone(void) { return cksum_f64(CKSUM_INIT, whetstone(8)); }
/* One work unit is one pass over the whole module set — see the note above. */
static u64 r_whetstone(u32 n) { double r = whetstone(n); SINK((int)(r != 0.0)); return (u64)n; }

/* ══ fpu/linpack — LINPACK 100x100, dgefa + dgesl ═════════════════════════════
 *
 * The netlib benchmark: LU factorisation with partial pivoting, then a solve,
 * on a 100x100 double matrix. Reported as flops, so rate/s is MFLOPS x 1e6 in
 * the same units every published LINPACK figure uses.
 *
 * Column-major with lda = 101, exactly as the original, because that odd
 * leading dimension is doing real work — it staggers each column across cache
 * sets instead of aliasing them all onto the same one, and a benchmark that
 * "tidied" it to 100 would be measuring conflict misses.
 */

#define LP_N    100
#define LP_LDA  101

static double *lp_a, *lp_b;
static double *lp_a0, *lp_b0;      /* pristine system, generated once */
static int    *lp_ipvt;
static int     lp_ready;

static void lp_matgen(void);

/*
 * The reference LINPACK generates the system outside the timed region and
 * times only dgefa+dgesl. The harness times whatever run() does, so the
 * pristine system is built once and restored with a copy — 80 KB against
 * ~690k flops of solve, and identical on every cell being compared. work_reset
 * hands back the same addresses on every call, so "already built" is a safe
 * thing to remember.
 */
static void lp_alloc(void)
{
    double *a  = (double *)work_alloc(LP_LDA * LP_N * (u32)sizeof(double), 64);
    double *b  = (double *)work_alloc(LP_N * (u32)sizeof(double), 64);
    double *a0 = (double *)work_alloc(LP_LDA * LP_N * (u32)sizeof(double), 64);
    double *b0 = (double *)work_alloc(LP_N * (u32)sizeof(double), 64);
    int    *pv = (int *)work_alloc(LP_N * (u32)sizeof(int), 64);

    if (lp_ready && a == lp_a && a0 == lp_a0) { lp_b = b; lp_ipvt = pv; return; }

    lp_a = a; lp_b = b; lp_a0 = a0; lp_b0 = b0; lp_ipvt = pv;
    lp_matgen();
    memcpy(lp_a0, lp_a, LP_LDA * LP_N * sizeof(double));
    memcpy(lp_b0, lp_b, LP_N * sizeof(double));
    lp_ready = 1;
}

static void lp_restore(void)
{
    memcpy(lp_a, lp_a0, LP_LDA * LP_N * sizeof(double));
    memcpy(lp_b, lp_b0, LP_N * sizeof(double));
}

/* The original matgen: a 3125-multiplier LCG mod 65536, mapped to [-2, 2). */
static void lp_matgen(void)
{
    int i, j, init = 1325;
    for (j = 0; j < LP_N; j++) {
        for (i = 0; i < LP_N; i++) {
            init = 3125 * init % 65536;
            lp_a[i + j * LP_LDA] = ((double)(init - 32768)) / 16384.0;
        }
    }
    for (i = 0; i < LP_N; i++) lp_b[i] = 0.0;
    for (j = 0; j < LP_N; j++)
        for (i = 0; i < LP_N; i++) lp_b[i] += lp_a[i + j * LP_LDA];
}

static void lp_daxpy(int n, double da, const double *dx, double *dy)
{
    int i;
    if (da == 0.0) return;
    for (i = 0; i < n; i++) dy[i] += da * dx[i];
}

static void lp_dscal(int n, double da, double *dx)
{
    int i;
    for (i = 0; i < n; i++) dx[i] *= da;
}

static double lp_ddot(int n, const double *dx, const double *dy)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++) s += dx[i] * dy[i];
    return s;
}

static int lp_idamax(int n, const double *dx)
{
    double dmax;
    int i, best = 0;
    if (n < 1) return -1;
    dmax = dx[0] < 0.0 ? -dx[0] : dx[0];
    for (i = 1; i < n; i++) {
        double v = dx[i] < 0.0 ? -dx[i] : dx[i];
        if (v > dmax) { dmax = v; best = i; }
    }
    return best;
}

static void lp_dgefa(void)
{
    int j, k, l;
    for (k = 0; k < LP_N - 1; k++) {
        double *ak = &lp_a[k + k * LP_LDA];
        double t;
        l = lp_idamax(LP_N - k, ak) + k;
        lp_ipvt[k] = l;
        if (lp_a[l + k * LP_LDA] == 0.0) continue;
        if (l != k) {
            t = lp_a[l + k * LP_LDA];
            lp_a[l + k * LP_LDA] = lp_a[k + k * LP_LDA];
            lp_a[k + k * LP_LDA] = t;
        }
        t = -1.0 / lp_a[k + k * LP_LDA];
        lp_dscal(LP_N - k - 1, t, &lp_a[k + 1 + k * LP_LDA]);
        for (j = k + 1; j < LP_N; j++) {
            t = lp_a[l + j * LP_LDA];
            if (l != k) {
                lp_a[l + j * LP_LDA] = lp_a[k + j * LP_LDA];
                lp_a[k + j * LP_LDA] = t;
            }
            lp_daxpy(LP_N - k - 1, t, &lp_a[k + 1 + k * LP_LDA], &lp_a[k + 1 + j * LP_LDA]);
        }
    }
    lp_ipvt[LP_N - 1] = LP_N - 1;
}

static void lp_dgesl(void)
{
    int k, l;
    double t;
    for (k = 0; k < LP_N - 1; k++) {
        l = lp_ipvt[k];
        t = lp_b[l];
        if (l != k) { lp_b[l] = lp_b[k]; lp_b[k] = t; }
        lp_daxpy(LP_N - k - 1, t, &lp_a[k + 1 + k * LP_LDA], &lp_b[k + 1]);
    }
    for (k = LP_N - 1; k >= 0; k--) {
        lp_b[k] /= lp_a[k + k * LP_LDA];
        t = -lp_b[k];
        lp_daxpy(k, t, &lp_a[k * LP_LDA], lp_b);
    }
}

static double lp_solve_once(void)
{
    lp_restore();
    lp_dgefa();
    lp_dgesl();
    return lp_ddot(LP_N, lp_b, lp_b);
}

static u64 v_linpack(void)
{
    u64 h = CKSUM_INIT;
    int i;
    lp_alloc();
    (void)lp_solve_once();
    /* The solution should be all ones; fold in every element so a single wrong
     * pivot cannot hide behind a norm. */
    for (i = 0; i < LP_N; i++) h = cksum_f64(h, lp_b[i]);
    return h;
}

/* 2/3 n^3 + 2 n^2 flops per factor-and-solve, the standard LINPACK count. */
#define LP_FLOPS  ((2ull * LP_N * LP_N * LP_N) / 3ull + 2ull * LP_N * LP_N)

static u64 r_linpack(u32 n)
{
    u32 i;
    lp_alloc();
    for (i = 0; i < n; i++) SINK((int)(lp_solve_once() != 0.0));
    return (u64)n * LP_FLOPS;
}

/* ── fpu/matmul — 64x64 double matrix multiply ────────────────────────────── */

#define MM_N 64

static u64 mm_go(u32 iters, u64 *out_sum)
{
    double *a = (double *)work_alloc(MM_N * MM_N * (u32)sizeof(double), 64);
    double *b = (double *)work_alloc(MM_N * MM_N * (u32)sizeof(double), 64);
    double *c = (double *)work_alloc(MM_N * MM_N * (u32)sizeof(double), 64);
    u64 s = 0x1234567ull;
    u32 it;
    int i, j, k;

    for (i = 0; i < MM_N * MM_N; i++) {
        a[i] = (double)(s32)(rng_next(&s) >> 40) * (1.0 / 8388608.0);
        b[i] = (double)(s32)(rng_next(&s) >> 40) * (1.0 / 8388608.0);
    }

    for (it = 0; it < iters; it++) {
        for (i = 0; i < MM_N; i++) {
            for (j = 0; j < MM_N; j++) c[i * MM_N + j] = 0.0;
            for (k = 0; k < MM_N; k++) {
                double aik = a[i * MM_N + k];
                const double *brow = &b[k * MM_N];
                double *crow = &c[i * MM_N];
                for (j = 0; j < MM_N; j++) crow[j] += aik * brow[j];
            }
        }
    }
    if (out_sum) {
        u64 h = CKSUM_INIT;
        for (i = 0; i < MM_N * MM_N; i++) h = cksum_f64(h, c[i]);
        *out_sum = h;
    }
    return (u64)iters * 2ull * MM_N * MM_N * MM_N;
}

static u64 v_matmul(void) { u64 h = 0; (void)mm_go(1, &h); return h; }
static u64 r_matmul(u32 n) { return mm_go(n, 0); }

/* ── registration ─────────────────────────────────────────────────────────── */

static const struct bench benches[] = {
    BENCH("fpu/scalar_d",  "ops",  v_scalar_d,  r_scalar_d,  1u << 13, BG_FPU),
    BENCH("fpu/scalar_s",  "ops",  v_scalar_s,  r_scalar_s,  1u << 13, BG_FPU),
    BENCH("fpu/divsqrt",   "ops",  v_divsqrt,   r_divsqrt,   1u << 12, BG_FPU),
    BENCH("fpu/transcend", "ops",  v_transcend, r_transcend, 1u << 10, BG_FPU),
    BENCH("fpu/whetstone", "loop", v_whetstone, r_whetstone, 1u <<  6, BG_FPU),
    BENCH("fpu/linpack",   "flop", v_linpack,   r_linpack,   4,        BG_FPU),
    BENCH("fpu/matmul",    "flop", v_matmul,    r_matmul,    4,        BG_FPU),
};

const struct bench_group group_fpu = {
    "fpu", benches, sizeof(benches) / sizeof(benches[0])
};
