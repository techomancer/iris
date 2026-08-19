/* fpu_common.h — conventions shared by the fpu/ test files.
 *
 * The rules these encode, all of which cost time to rediscover (see
 * docs/gotchas.md):
 *
 *   - `AF`, not `A`, for any asm block containing an FP instruction. The suite
 *     is built -msoft-float so the compiler never touches the FPU on its own,
 *     and that also makes GAS refuse FP mnemonics without `.set hardfloat`.
 *   - Never list `$f*` in a clobber list. GCC rejects it under -msoft-float,
 *     and it is unnecessary: a soft-float build never allocates an FP
 *     register, so asm may use $f0..$f31 freely.
 *   - Values cross the C/asm boundary through memory, never through FP
 *     registers. `w()` and `d()` are the word and doubleword views of the
 *     scratch area.
 *   - start.S sets FR=1. A test that clears it must put it back.
 */
#ifndef FPU_COMMON_H
#define FPU_COMMON_H

#include "testlib.h"
#include "cp0.h"

#define A  ".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"
#define AF A ".set hardfloat\n\t"
#define Z  "\n\t.set pop"

/* `dla` is a macro by construction — it expands to lui/daddiu — so it cannot
 * assemble under the `.set nomacro` the prologue installs. Enabling macros for
 * exactly one instruction says that is deliberate, and keeps the assembler
 * quiet about the rest. */
#define DLA(dst, label)  ".set macro\n\tdla " dst ", " label "\n\t.set nomacro\n\t"

extern char _scratch_start[];

static inline volatile u32 *w(void) { return (volatile u32 *)_scratch_start; }
static inline volatile u64 *d(void) { return (volatile u64 *)_scratch_start; }

/* ── IEEE-754 single-precision bit patterns ───────────────────────────────── */
#define F_0        0x00000000u
#define F_NEG0     0x80000000u
#define F_1        0x3F800000u
#define F_NEG1     0xBF800000u
#define F_2        0x40000000u
#define F_NEG2     0xC0000000u
#define F_3        0x40400000u
#define F_4        0x40800000u
#define F_5        0x40A00000u
#define F_6        0x40C00000u
#define F_0P5      0x3F000000u
#define F_0P25     0x3E800000u
#define F_INF      0x7F800000u
#define F_NEGINF   0xFF800000u
#define F_QNAN     0x7FC00000u
#define F_SNAN     0x7FA00000u
#define F_MAX      0x7F7FFFFFu   /* largest finite single            */
#define F_MIN_NORM 0x00800000u   /* smallest normal single, 2^-126   */
#define F_DENORM   0x00000001u   /* smallest denormal single, 2^-149 */
#define F_BIG      0x4F000000u   /* 2^31 — out of range for a signed word */

/* Double-precision. */
#define D_0        0x0000000000000000ull
#define D_NEG0     0x8000000000000000ull
#define D_1        0x3FF0000000000000ull
#define D_NEG1     0xBFF0000000000000ull
#define D_2        0x4000000000000000ull
#define D_3        0x4008000000000000ull
#define D_4        0x4010000000000000ull
#define D_0P5      0x3FE0000000000000ull
#define D_0P25     0x3FD0000000000000ull
#define D_INF      0x7FF0000000000000ull
#define D_NEGINF   0xFFF0000000000000ull
#define D_QNAN     0x7FF8000000000000ull
#define D_SNAN     0x7FF4000000000000ull
#define D_MAX      0x7FEFFFFFFFFFFFFFull
#define D_MIN_NORM 0x0010000000000000ull   /* 2^-1022 */
#define D_DENORM   0x0000000000000001ull   /* 2^-1074 */

/* ── FCSR field helpers ───────────────────────────────────────────────────── */
/* The five IEEE bits (V,Z,O,U,I) appear three times in FCSR — as Cause at
 * 17:12 (with E at 17), as Enables at 11:7, and as Flags at 6:2. */
#define FCSR_ENABLE(bits)  ((u32)(bits) << FCSR_ENABLE_SHIFT)
#define FCSR_CAUSE_OF(v)   (((v) >> FCSR_CAUSE_SHIFT) & 0x3Fu)
#define FCSR_FLAGS_OF(v)   (((v) >> FCSR_FLAGS_SHIFT) & 0x1Fu)

static inline u32 fcsr_flags(void)  { return FCSR_FLAGS_OF(fcsr()); }
static inline u32 fcsr_cause(void)  { return FCSR_CAUSE_OF(fcsr()); }

/* Round-to-nearest, no enables, no flags, no cause. */
static inline void fcsr_reset(void) { fcsr_set(0); }

/* A NaN in single/double format — the payload is implementation-defined, so
 * tests assert NaN-ness rather than a particular pattern. */
static inline int is_nan_s(u32 v)
{
    return (v & 0x7F800000u) == 0x7F800000u && (v & 0x007FFFFFu) != 0;
}
static inline int is_nan_d(u64 v)
{
    return (v & 0x7FF0000000000000ull) == 0x7FF0000000000000ull
        && (v & 0x000FFFFFFFFFFFFFull) != 0;
}

/* ── observing one operation ──────────────────────────────────────────────── */

/*
 * The destination register is preloaded with a sentinel before every operation
 * below, because "when a floating-point exception is taken, no results are
 * stored" (R4000 manual, chapter 6) is a rule the suite can only check if it
 * knows what was in the register beforehand. Neither value is a plausible
 * result of anything tested.
 */
#define SENTINEL_S 0x5A5A5A5Au
#define SENTINEL_D 0x5A5A5A5A5A5A5A5Aull

/* scratch[0]=a  scratch[1]=b  scratch[2]=result  scratch[3]=sentinel */
#define FPOP_S(name, insn)                                                 \
    static void name(void)                                                 \
    {                                                                      \
        __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"                      \
                                "lwc1 $f2, 4(%0)\n\t"                      \
                                "lwc1 $f4, 12(%0)\n\t"                     \
                                insn "\n\t"                                \
                                "swc1 $f4, 8(%0)" Z                        \
                             :: "r"(w()) : "memory");                      \
    }

/* Same, in doublewords. */
#define FPOP_D(name, insn)                                                 \
    static void name(void)                                                 \
    {                                                                      \
        __asm__ __volatile__(AF "ldc1 $f0, 0(%0)\n\t"                      \
                                "ldc1 $f2, 8(%0)\n\t"                      \
                                "ldc1 $f4, 24(%0)\n\t"                     \
                                insn "\n\t"                                \
                                "sdc1 $f4, 16(%0)" Z                       \
                             :: "r"(d()) : "memory");                      \
    }

/* One observation of a single FP operation: what the FPU recorded, and what
 * the CPU did about it. */
struct fp_obs {
    u32 exceptions;   /* CPU exceptions taken during the operation */
    u32 excode;       /* ExcCode of the last one                   */
    u32 cause;        /* FCSR Cause field (E,V,Z,O,U,I)            */
    u32 flags;        /* FCSR Flag field after the operation       */
    u32 result;       /* single-precision destination              */
    u64 result_d;     /* double-precision destination              */
};

static inline void obs_collect(struct fp_obs *o)
{
    o->exceptions = exc.count;
    o->excode = exc.count ? CAUSE_EXC(exc.cause) : 0u;
    /* On a trap the handler clears the Cause field before returning, so the
     * value it captured on entry is the only surviving copy. */
    o->cause = exc.count ? FCSR_CAUSE_OF(exc.fcsr) : fcsr_cause();
    o->flags = fcsr_flags();
    o->result = w()[2];
    o->result_d = d()[2];
}

/* Run `op` on (a, b) with `fcsr` loaded first (enables, FS, rounding mode). */
static inline struct fp_obs observe_s(u32 a, u32 b, u32 fcsr_init, void (*op)(void))
{
    struct fp_obs o;
    w()[0] = a; w()[1] = b; w()[2] = 0; w()[3] = SENTINEL_S;
    SYNC();
    fcsr_set(fcsr_init);
    exc_clear();
    op();
    SYNC();
    obs_collect(&o);
    fcsr_reset();
    return o;
}

static inline struct fp_obs observe_d(u64 a, u64 b, u32 fcsr_init, void (*op)(void))
{
    struct fp_obs o;
    d()[0] = a; d()[1] = b; d()[2] = 0; d()[3] = SENTINEL_D;
    SYNC();
    fcsr_set(fcsr_init);
    exc_clear();
    op();
    SYNC();
    obs_collect(&o);
    fcsr_reset();
    return o;
}

#endif /* FPU_COMMON_H */
