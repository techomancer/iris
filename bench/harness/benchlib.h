/* benchlib.h — the benchmark harness API.
 *
 * A benchmark is a pair of functions and a table entry. `verify()` runs one
 * fixed reference workload and returns a checksum, which the runner compares
 * against a golden value computed by building the same source for the host —
 * that is the accuracy score. `run(n)` performs n iterations of the same kind
 * of work and returns how many work units that was, and the runner calls it
 * with whatever n makes the measurement long enough to mean something — that
 * is the performance score. The two are separate on purpose: an
 * autoscaled iteration count would make the checksum depend on how fast the
 * host is.
 */
#ifndef BENCHLIB_H
#define BENCHLIB_H

#if defined(BENCH_HOST)
#  include "hostshim.h"
#else
#  include "console.h"
#endif

/* ── time base ────────────────────────────────────────────────────────────── */

/* True when the IRIS test device is present and advertises TESTDEV_CAP_TIMEBASE.
 * Without it (real hardware, or an older emulator build) the suite falls back
 * to CP0 Count and says so in its output, because everything derived from a
 * guessed Count frequency is a guess. */
extern int have_timebase;

/* Host monotonic nanoseconds. 0 when !have_timebase. */
u64 bench_host_ns(void);
/* Guest instructions retired. 0 when !have_timebase. */
u64 bench_icount(void);
/* CP0 Count, which on real hardware is the CPU clock / 2 and under IRIS is a
 * virtual clock derived from `count_hz` — hence measured, not assumed. */
u32 bench_cp0_count(void);

/* CP0 Count ticks per real second, measured against the host clock at startup.
 * BENCH_COUNT_HZ_ASSUMED when there is no host clock to measure against. */
extern u64 count_hz_measured;
#define BENCH_COUNT_HZ_ASSUMED  100000000ull   /* a 200 MHz R4x00: Count = clk/2 */

/* ── working memory ───────────────────────────────────────────────────────── */

/* Scratch RAM immediately above the image, sized by probing at startup. Every
 * kernel that needs a buffer carves it out of here rather than declaring its
 * own, so the total footprint is known and the cache-hierarchy sweeps can rely
 * on it being physically contiguous KSEG0. */
extern unsigned char *work;
extern u32 work_bytes;
#define WORK_WANT_BYTES  (24u * 1024u * 1024u)

/*
 * Carve `n` bytes off the work area, aligned to `align`. Panics if the area is
 * exhausted — a kernel silently getting a smaller buffer than it asked for
 * would make its score meaningless rather than obviously wrong.
 *
 * This is a bump allocator with no free: call it once per run(), OUTSIDE the
 * iteration loop. A kernel that allocates per iteration exhausts 24 MB in a
 * handful of passes, and because the runner resets between benchmarks it does
 * so only at whatever iteration count the autoscaler happened to pick — which
 * makes it look like a data-dependent failure rather than the plain bug it is.
 */
void *work_alloc(u32 n, u32 align);
/* Reset the bump allocator. The runner does this before each benchmark, so
 * every kernel sees the same addresses and the same cache alignment. */
void work_reset(void);

/* ── checksums ────────────────────────────────────────────────────────────── */

/* FNV-1a over a stream of 64-bit values. Order-sensitive, which is what we
 * want: a kernel that produces the right values in the wrong order is broken. */
#define CKSUM_INIT 0xcbf29ce484222325ull
static inline u64 cksum_u64(u64 h, u64 v)
{
    int i;
    for (i = 0; i < 8; i++) {
        h ^= (v >> (i * 8)) & 0xFF;
        h *= 0x100000001b3ull;
    }
    return h;
}
static inline u64 cksum_bytes(u64 h, const void *p, u32 n)
{
    const unsigned char *b = (const unsigned char *)p;
    u32 i;
    for (i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}
/* Fold a double in by its bit pattern, so the comparison is exact rather than
 * "close enough" — a one-ulp difference is exactly the kind of FPU fault this
 * suite exists to find. */
static inline u64 cksum_f64(u64 h, double d)
{
    union { double d; u64 u; } cv;
    cv.d = d;
    return cksum_u64(h, cv.u);
}
static inline u64 cksum_f32(u64 h, float f)
{
    union { float f; u32 u; } cv;
    cv.f = f;
    return cksum_u64(h, (u64)cv.u);
}

/* ── deterministic input data ─────────────────────────────────────────────── */

/* xorshift64*: every kernel seeds its own copy, so kernels never depend on the
 * order they ran in. */
static inline u64 rng_next(u64 *s)
{
    u64 x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1Dull;
}

/* Keep the optimiser from folding a value away or hoisting the work that
 * produced it out of a loop. */
#define OPAQUE(x)  ({ __typeof__(x) __o = (x); __asm__ __volatile__("" : "+r"(__o)); __o; })
#define SINK(x)    __asm__ __volatile__("" :: "r"(x) : "memory")

/* ── the three libc functions the compiler emits calls to ─────────────────── */
/* Defined in harness/string.c for the MIPS build; hostshim.h pulls the real
 * ones in for the golden generator. */
#if !defined(BENCH_HOST)
void *memset(void *dst, int c, unsigned long n);
void *memcpy(void *dst, const void *src, unsigned long n);
void *memmove(void *dst, const void *src, unsigned long n);
int   memcmp(const void *a, const void *b, unsigned long n);
#endif

/* ── machine inventory ────────────────────────────────────────────────────── */

/*
 * What the machine actually is, at the moment the suite runs.
 *
 * Every field comes from the hardware itself — CP0 Config for the cache
 * geometry, the memory controller's MEMCFG registers for the bank layout — so
 * this is as true under `--load-elf` (no PROM, no POST) as it is on a
 * PROM-booted or real machine. Nothing here is passed in by the host or taken
 * from a build-time constant, which is the point: a result that recorded what
 * the *runner* believed rather than what the guest found would be worth
 * nothing when the two disagreed, and the disagreements are the interesting
 * cases.
 *
 * It matters for comparison, not just for display. The `mem/` kernels are a
 * direct readout of the cache hierarchy, so two results from machines with
 * different L1 sizes are not measuring the same thing — and until this existed,
 * nothing in a saved result said so.
 */
struct hwinv {
    u32 prid, fir, config, sysid;
    u32 cpu_rev_major, cpu_rev_minor;      /* PRId 7:4 / 3:0 */
    u32 fpu_imp, fpu_rev_major, fpu_rev_minor;
    u32 l1i_bytes, l1i_line;
    u32 l1d_bytes, l1d_line;
    int l2_present;
    u32 l2_line;
    /* 0 when the architecture does not report it — true on the R4400 and on a
     * non-Triton R5000, where the PROM reads the size out of the EEPROM
     * instead. Reported as unknown rather than guessed. */
    u32 l2_bytes;
    u32 ram_mb;                            /* total across valid banks */
    u32 bank_mb[4], bank_base[4];
    unsigned banks;                        /* count of valid banks */
};

extern struct hwinv hw;

/* Fill `hw`. Called by bench_init(); safe to call before the work area exists. */
void bench_probe_hw(void);

/* ── run configuration ────────────────────────────────────────────────────── */

/*
 * What the host asked for, read once at startup from TESTDEV_RUN_CONFIG — the
 * only channel there is, since a bare-metal image loaded with --load-elf has no
 * argv and no environment. All three are set to their defaults when the host
 * asked for nothing, which is also what an emulator without the register gives
 * us, so nothing downstream needs to know whether it was there.
 *
 * Note what is *not* configurable: whether a kernel verifies itself. Accuracy
 * is scored against golden checksums compiled into this binary, and a shorter
 * run that quietly checked less would report the same 100% while covering less
 * ground. Only the timed measurement gets cheaper.
 */
extern u32 bench_groups;     /* BG_* mask of groups to run; BG_ALL by default */
extern u32 bench_time_pct;   /* per-kernel target time, percent of default */
extern u32 bench_repeats;    /* timed passes per kernel */

/* Best of two. The slow sample is host scheduling noise, not the emulator: the
 * guest performs a fixed amount of work either way. */
#define BENCH_REPEATS_DEFAULT  2
/* Below this the ~30 ns Count granularity and the two uncached device reads on
 * each side of a timed region stop being noise and start being the measurement. */
#define BENCH_TIME_PCT_MIN     10

/* ── benchmark registration ───────────────────────────────────────────────── */

#define BG_INT    0x01
#define BG_FPU    0x02
#define BG_MEM    0x04
#define BG_IMG    0x08
#define BG_CODEC  0x10
#define BG_SYS    0x20
#define BG_ALL    0x3F

/*
 * CPU applicability. Same first two bits as cpu-tests, plus one that suite
 * deliberately does not have.
 *
 * BCPU_OTHER covers every MIPS III-or-later CPU that is not one of the two the
 * emulator models — an R4000, R4600, R8000, R10000, a real machine's RM7000.
 * Those can run this suite and get a *meaningful* score, because unlike
 * cpu-tests nothing here is CPU-specific: the kernels are ordinary compiled
 * MIPS III, and golden.h is one flat table computed natively rather than a
 * per-CPU one. So an unrecognised CPU is a labelling problem, not a
 * correctness problem, and the suite runs and says what it found.
 *
 * cpu-tests is the opposite and keeps its own two-value CPU_* — its tests
 * check R4400-versus-R5000 behaviour by construction, so "some other CPU"
 * genuinely has no expected answer there.
 */
#define BCPU_R4400  0x1
#define BCPU_R5000  0x2
#define BCPU_OTHER  0x4
#define BCPU_ALL    (BCPU_R4400 | BCPU_R5000 | BCPU_OTHER)

/* A kernel that is *supposed* to take exceptions. Everything else taking one
 * is a bug: the shared dispatcher records and steps over the faulting
 * instruction, so a kernel that faults does not crash — it quietly produces a
 * number for doing something other than what it claims. mem/unaligned scored a
 * plausible-looking 871 k accesses/s that way, taking an address error three
 * loads in four, and only the checksum gave it away. */
#define BF_TAKES_EXC  0x1

struct bench {
    const char *name;        /* "img/dct8x8" — group/kernel */
    const char *unit;        /* work unit: "ops", "B", "px", "blk", "flop" */
    u64  (*verify)(void);    /* fixed reference run -> checksum; 0 if unchecked */
    u64  (*run)(u32 iters);  /* timed loop -> work units performed */
    u32   base_iters;        /* first guess for the autoscaler */
    u32   group;             /* BG_* */
    u32   cpus;              /* BCPU_* */
    u32   flags;             /* BF_* */
};

struct bench_group {
    const char *name;
    const struct bench *benches;
    unsigned count;
};

#define BENCH(n, u, v, r, it, g)         { n, u, v, r, it, g, BCPU_ALL, 0 }
#define BENCH_EXC(n, u, v, r, it, g)     { n, u, v, r, it, g, BCPU_ALL, BF_TAKES_EXC }
#define BENCH_CPU(n, u, v, r, it, g, c)  { n, u, v, r, it, g, c, 0 }

#define DECLARE_BGROUP(g)  extern const struct bench_group g

/* ── CPU identity ─────────────────────────────────────────────────────────── */
extern u32 cpu_kind, cpu_prid, cpu_fir, cpu_config;
extern int have_l2;

/* Bring up the time base, the work area and the machine identity. benchlib.c
 * on MIPS, gen/hostplat.c on the host. */
void bench_init(void);

/* Exceptions taken since the last bench_exc_reset(). Always 0 on the host,
 * which has no exception vector to count through. */
void bench_exc_reset(void);
u32  bench_exc_count(void);

#if !defined(BENCH_HOST)

/* ── machine plumbing (MIPS build only) ───────────────────────────────────── */

void dcache_wb_range(volatile void *addr, u32 len);
void dcache_inv_range(volatile void *addr, u32 len);
void icache_inv_range(volatile void *addr, u32 len);

/* Field order and padding are fixed by excoff.h, which start.S's shared
 * dispatcher stores through; benchlib.c static-asserts every offset. */
struct exc_record {
    u32 count;
    u32 status;
    u32 cause;
    u32 vector;
    u32 fcsr;
    u32 pad;
    u64 epc;
    u64 badvaddr;
    u64 errorepc;
    u64 entryhi;
    u64 context;
    u64 xcontext;
};
extern volatile struct exc_record exc;

#define EXC_RESUME_SKIP    0
#define EXC_RESUME_RETRY   1
extern volatile u32 exc_resume_mode;
extern volatile u32 exc_user_handler;

void exc_clear(void);
void exc_install(void);

#endif /* !BENCH_HOST */

#endif /* BENCHLIB_H */
