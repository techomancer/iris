/* bmath.h — the libm the benchmark suite carries with it.
 *
 * Every transcendental here is a polynomial in +, -, * and / (plus the
 * hardware square root), evaluated in a fixed order. That is not a stylistic
 * choice: the suite's accuracy score compares the guest's results against
 * golden checksums computed by building these same sources for the host, so
 * every operation on the path has to be one IEEE-754 defines exactly. A real
 * libm call would not be — two libms disagree in the last ulp, and the
 * checksum would then report an emulator fault that is really a libm
 * difference.
 *
 * Accuracy of the approximations themselves does not matter for the same
 * reason. They are good to roughly single-precision over the reduced range,
 * which is plenty for a workload, and whatever they compute they compute
 * identically everywhere.
 */
#ifndef BMATH_H
#define BMATH_H

#define B_PI      3.14159265358979323846
#define B_TWO_PI  6.28318530717958647692
#define B_PI_2    1.57079632679489661923
#define B_LN2     0.69314718055994530942

/* Hardware square root. MIPS III has sqrt.d/sqrt.s and x86-64 has sqrtsd/
 * sqrtss; IEEE-754 requires both to be correctly rounded, so they agree bit
 * for bit. Anything else here would not. */
#if defined(BENCH_HOST)
static inline double b_sqrt(double x) { return __builtin_sqrt(x); }
static inline float  b_sqrtf(float x) { return __builtin_sqrtf(x); }
#else
static inline double b_sqrt(double x)
{
    double r;
    __asm__(".set push; .set mips3; .set hardfloat\n\t"
            "sqrt.d %0, %1\n\t"
            ".set pop" : "=f"(r) : "f"(x));
    return r;
}
static inline float b_sqrtf(float x)
{
    float r;
    __asm__(".set push; .set mips3; .set hardfloat\n\t"
            "sqrt.s %0, %1\n\t"
            ".set pop" : "=f"(r) : "f"(x));
    return r;
}
#endif

static inline double b_fabs(double x) { return x < 0.0 ? -x : x; }

/* Round-to-nearest via the "add and subtract a big constant" trick — exact in
 * round-to-nearest-even for |x| < 2^52, and free of any conversion
 * instruction, so it cannot pick up a flush-to-zero or an invalid-operation
 * difference on the way through an integer register. */
static inline double b_round(double x)
{
    const double big = 6755399441055744.0;      /* 3 * 2^51 */
    if (b_fabs(x) >= 4503599627370496.0) return x;
    return x >= 0.0 ? (x + big) - big : (x - big) + big;
}

/* sin over the full line: reduce by 2*pi, fold into [-pi/2, pi/2], then a
 * degree-13 odd minimax polynomial. */
static inline double b_sin(double x)
{
    double y, y2, r;
    int neg = 0;
    x = x - B_TWO_PI * b_round(x / B_TWO_PI);    /* -> [-pi, pi] */
    if (x > B_PI_2)       { x = B_PI - x; }
    else if (x < -B_PI_2) { x = -B_PI - x; }
    y = x;
    if (y < 0.0) { y = -y; neg = 1; }
    y2 = y * y;
    r = -2.5052108385441718e-8;
    r = r * y2 + 2.7557319223985893e-6;
    r = r * y2 - 1.9841269841269841e-4;
    r = r * y2 + 8.3333333333333333e-3;
    r = r * y2 - 1.6666666666666666e-1;
    r = r * y2 * y + y;
    return neg ? -r : r;
}

static inline double b_cos(double x) { return b_sin(x + B_PI_2); }

/* exp: split into 2^k * exp(f) with |f| <= ln2/2, then a degree-7 series.
 * The 2^k scaling is done by repeated multiplication rather than by building
 * an exponent field through a union, so it stays pure arithmetic. */
static inline double b_exp(double x)
{
    double k, f, r, p;
    int i, n;
    if (x > 700.0)  x = 700.0;
    if (x < -700.0) x = -700.0;
    k = b_round(x / B_LN2);
    f = x - k * B_LN2;
    r = 1.0 / 5040.0;
    r = r * f + 1.0 / 720.0;
    r = r * f + 1.0 / 120.0;
    r = r * f + 1.0 / 24.0;
    r = r * f + 1.0 / 6.0;
    r = r * f + 0.5;
    r = r * f + 1.0;
    r = r * f + 1.0;
    n = (int)k;
    p = 1.0;
    if (n >= 0) { for (i = 0; i < n; i++) p *= 2.0; }
    else        { for (i = 0; i < -n; i++) p *= 0.5; }
    return r * p;
}

/* log for x > 0: scale into [2/3, 4/3] by halving/doubling, then atanh series
 * on (x-1)/(x+1). */
static inline double b_log(double x)
{
    double s, s2, r;
    int k = 0;
    if (x <= 0.0) return -1.0e300;
    while (x > 1.3333333333333333) { x *= 0.5; k++; }
    while (x < 0.6666666666666666) { x *= 2.0; k--; }
    s = (x - 1.0) / (x + 1.0);
    s2 = s * s;
    r = 2.0 / 13.0;
    r = r * s2 + 2.0 / 11.0;
    r = r * s2 + 2.0 / 9.0;
    r = r * s2 + 2.0 / 7.0;
    r = r * s2 + 2.0 / 5.0;
    r = r * s2 + 2.0 / 3.0;
    r = r * s2 + 2.0;
    return r * s + (double)k * B_LN2;
}

/* atan over the full line: fold |x| > 1 through the reciprocal identity, then
 * a degree-19 odd polynomial on [-1, 1]. */
static inline double b_atan(double x)
{
    double y, y2, r;
    int neg = 0, inv = 0;
    if (x < 0.0) { x = -x; neg = 1; }
    if (x > 1.0) { x = 1.0 / x; inv = 1; }
    y = x; y2 = y * y;
    r =  1.0 / 19.0;
    r = -1.0 / 17.0 + r * y2;
    r =  1.0 / 15.0 + r * y2;
    r = -1.0 / 13.0 + r * y2;
    r =  1.0 / 11.0 + r * y2;
    r = -1.0 /  9.0 + r * y2;
    r =  1.0 /  7.0 + r * y2;
    r = -1.0 /  5.0 + r * y2;
    r =  1.0 /  3.0 + r * y2;
    r = -1.0        + r * y2;
    r = r * y2 * y + y;
    if (inv) r = B_PI_2 - r;
    return neg ? -r : r;
}

static inline double b_pow_i(double x, int n)
{
    double r = 1.0;
    int i;
    if (n < 0) { x = 1.0 / x; n = -n; }
    for (i = 0; i < n; i++) r *= x;
    return r;
}

#endif /* BMATH_H */
