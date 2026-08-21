/* benchlib.c — time base, working memory, and machine identity. */

#include "benchlib.h"
#include "cp0.h"
#include "excoff.h"

u32 cpu_kind, cpu_prid, cpu_fir, cpu_config;
int have_l2;
int have_timebase;
u64 count_hz_measured = BENCH_COUNT_HZ_ASSUMED;

unsigned char *work;
u32 work_bytes;
static u32 work_used;

/* start.S's exception dispatcher stores through these. The benchmark suite
 * only needs the TLB-refill path (sys/tlb_miss) and a trap round trip
 * (sys/exception), but the dispatcher is shared verbatim with cpu-tests, so
 * the whole record has to exist. */
volatile struct exc_record exc;
volatile u32 exc_resume_mode = EXC_RESUME_SKIP;
volatile u32 exc_user_handler = 0;

_Static_assert(__builtin_offsetof(struct exc_record, count)    == EXC_O_COUNT,    "exc.count");
_Static_assert(__builtin_offsetof(struct exc_record, status)   == EXC_O_STATUS,   "exc.status");
_Static_assert(__builtin_offsetof(struct exc_record, cause)    == EXC_O_CAUSE,    "exc.cause");
_Static_assert(__builtin_offsetof(struct exc_record, vector)   == EXC_O_VECTOR,   "exc.vector");
_Static_assert(__builtin_offsetof(struct exc_record, fcsr)     == EXC_O_FCSR,     "exc.fcsr");
_Static_assert(__builtin_offsetof(struct exc_record, epc)      == EXC_O_EPC,      "exc.epc");
_Static_assert(__builtin_offsetof(struct exc_record, badvaddr) == EXC_O_BADVADDR, "exc.badvaddr");
_Static_assert(__builtin_offsetof(struct exc_record, errorepc) == EXC_O_ERROREPC, "exc.errorepc");
_Static_assert(__builtin_offsetof(struct exc_record, entryhi)  == EXC_O_ENTRYHI,  "exc.entryhi");
_Static_assert(__builtin_offsetof(struct exc_record, context)  == EXC_O_CONTEXT,  "exc.context");
_Static_assert(__builtin_offsetof(struct exc_record, xcontext) == EXC_O_XCONTEXT, "exc.xcontext");
_Static_assert(sizeof(struct exc_record) == EXC_SIZEOF, "exc_record size");

extern u32 tramp_tlb[], tramp_tlb_end[];
extern u32 tramp_xtlb[], tramp_xtlb_end[];
extern u32 tramp_general[], tramp_general_end[];

#define RD32(a)  (*(volatile u32 *)(unsigned long)(a))

/* ── time base ────────────────────────────────────────────────────────────── */

u64 bench_host_ns(void)
{
    u32 lo, hi;
    if (!have_timebase) return 0;
    /* LO first: that read latches the whole 64-bit sample, and HI then returns
     * the high half of the same one. The other order reads two samples. */
    lo = RD32(TESTDEV_HOST_NS_LO);
    hi = RD32(TESTDEV_HOST_NS_HI);
    return ((u64)hi << 32) | lo;
}

u64 bench_icount(void)
{
    u32 lo, hi;
    if (!have_timebase) return 0;
    lo = RD32(TESTDEV_ICOUNT_LO);
    hi = RD32(TESTDEV_ICOUNT_HI);
    return ((u64)hi << 32) | lo;
}

u32 bench_cp0_count(void) { return cp0_count(); }

/*
 * Decide whether the host time base is really there.
 *
 * Reading TESTDEV_CAPS and believing the answer is not enough. An emulator
 * build from before these registers existed decodes only 16 bytes and repeats,
 * so offset 0x20 aliases back onto SIGNATURE and 0x10 onto SIGNATURE as well —
 * and SIGNATURE ('IRIS') has bit 0 set, so a naive `caps & CAP_TIMEBASE` test
 * says yes, then every timing comes back as a frozen clock. Which is exactly
 * what happened the first time this ran.
 *
 * So: reject a CAPS word that is the signature, and then require the clock to
 * actually move. Nothing else about the suite is safe if this is wrong, and
 * falling back to CP0 Count is a perfectly good second choice.
 */
static int probe_timebase(void)
{
    u32 caps = RD32(TESTDEV_CAPS);
    u64 a, b;
    volatile u32 spin;

    if (caps == TESTDEV_MAGIC) return 0;              /* aliased SIGNATURE */
    if (!(caps & TESTDEV_CAP_TIMEBASE)) return 0;

    have_timebase = 1;                                /* bench_host_ns gates on it */
    a = bench_host_ns();
    for (spin = 0; spin < 100000u; spin++) { }
    b = bench_host_ns();
    have_timebase = 0;

    return b > a;
}

/*
 * Measure the CP0 Count rate against the host clock.
 *
 * Under IRIS, Count is virtual: mips_core.rs materializes it from a wall-clock
 * anchor at a `count_hz` that is *inferred* from the guest's own Compare
 * writes, and a bare-metal binary never writes a plausible one — so it sits at
 * the 33 MHz default rather than at the ~100 MHz a real 200 MHz R4400 would
 * show. Assuming either number would silently scale every Count-derived
 * figure. Measuring it turns that into data: the ratio between what Count
 * claims and what the host clock says is itself a report on the emulator's
 * timer model, and it is what makes the fallback path (real hardware, no test
 * device) honest about being an assumption.
 */
static void calibrate_count(void)
{
    u64 t0, t1, dns;
    u32 c0, c1, dc;
    volatile u32 spin;

    if (!have_timebase) { count_hz_measured = BENCH_COUNT_HZ_ASSUMED; return; }

    /* Long enough that Count's ~30 ns granularity and the device-read overhead
     * are both noise, short enough not to matter to total suite runtime. */
    t0 = bench_host_ns();
    c0 = bench_cp0_count();
    for (spin = 0; spin < 2000000u; spin++) { }
    c1 = bench_cp0_count();
    t1 = bench_host_ns();

    dns = t1 - t0;
    dc  = c1 - c0;                       /* 32-bit, wraps correctly */
    count_hz_measured = dns ? ((u64)dc * 1000000000ull) / dns : BENCH_COUNT_HZ_ASSUMED;
}

/* ── working memory ───────────────────────────────────────────────────────── */

extern unsigned char _work_start[];

/*
 * Find how much RAM there is above the image by writing a signature to the top
 * of each candidate size and reading it back through KSEG1, so a cached write
 * that never reached DRAM cannot be mistaken for real memory. Banks that are
 * not populated alias or swallow, and both show up as a mismatch.
 */
static void probe_work(void)
{
    u32 want = WORK_WANT_BYTES;
    work = _work_start;
    while (want >= 1024u * 1024u) {
        volatile u32 *k0 = (volatile u32 *)SEXT_PTR((u32)(unsigned long)work + want - 4);
        volatile u32 *k1 = (volatile u32 *)K1_PTR((u32)(unsigned long)work + want - 4);
        *k0 = 0xA5A5F00Du;
        dcache_wb_range((volatile void *)k0, 4);
        if (*k1 == 0xA5A5F00Du) break;
        want >>= 1;
    }
    work_bytes = want >= 1024u * 1024u ? want : 0;
    work_used = 0;
    if (work_bytes == 0) panic("no usable work RAM above the image");
}

void work_reset(void) { work_used = 0; }

void *work_alloc(u32 n, u32 align)
{
    u32 off = (work_used + (align - 1)) & ~(align - 1);
    if (off + n > work_bytes) {
        con_printf("\nwork_alloc(%u, %u) at offset %u of %u\n", n, align, off, work_bytes);
        panic("work area exhausted");
    }
    work_used = off + n;
    return work + off;
}

/* ── cache maintenance ────────────────────────────────────────────────────── */

/* Step 16 — the R4400 line size, and correct-if-redundant for the R5000's 32. */
#define RANGE_LOOP(op, addr, len)                                          \
    do {                                                                   \
        char *__p = (char *)((unsigned long)(addr) & ~15ul);                \
        char *__e = (char *)(((unsigned long)(addr) + (len) + 15) & ~15ul); \
        for (; __p != __e; __p += 16) CACHE_OP(op, __p);                    \
    } while (0)

void dcache_wb_range(volatile void *addr, u32 len)
{
    RANGE_LOOP(CACHE_D | CACHE_OP_HIT_WB_INV, addr, len);
    if (have_l2) RANGE_LOOP(CACHE_SD | CACHE_OP_HIT_WB_INV, addr, len);
}

void dcache_inv_range(volatile void *addr, u32 len)
{
    RANGE_LOOP(CACHE_D | CACHE_OP_HIT_INV, addr, len);
    if (have_l2) RANGE_LOOP(CACHE_SD | CACHE_OP_HIT_INV, addr, len);
}

void icache_inv_range(volatile void *addr, u32 len)
{
    RANGE_LOOP(CACHE_I | CACHE_OP_HIT_INV, addr, len);
    if (have_l2) RANGE_LOOP(CACHE_SD | CACHE_OP_HIT_WB_INV, addr, len);
}

/* ── exception vectors ────────────────────────────────────────────────────── */

static void install_one(u32 vec, const u32 *src, const u32 *end)
{
    volatile u32 *dst = (volatile u32 *)SEXT_PTR(vec);
    unsigned n = (unsigned)(end - src), i;
    if (n > 32) panic("vector trampoline too long");
    for (i = 0; i < n; i++) dst[i] = src[i];
}

void exc_install(void)
{
    install_one(VEC_TLB_REFILL,  tramp_tlb,     tramp_tlb_end);
    install_one(VEC_XTLB_REFILL, tramp_xtlb,    tramp_xtlb_end);
    install_one(VEC_GENERAL,     tramp_general, tramp_general_end);
    dcache_wb_range(SEXT_PTR(VEC_TLB_REFILL), 0x200);
    icache_inv_range(SEXT_PTR(VEC_TLB_REFILL), 0x200);
    SYNC();
}

void bench_exc_reset(void) { exc.count = 0; }
u32  bench_exc_count(void) { return exc.count; }

void exc_clear(void)
{
    exc.count = 0; exc.status = 0; exc.cause = 0; exc.vector = 0; exc.fcsr = 0;
    exc.epc = 0; exc.badvaddr = 0; exc.errorepc = 0; exc.entryhi = 0;
    exc.context = 0; exc.xcontext = 0;
    exc_resume_mode = EXC_RESUME_SKIP;
    exc_user_handler = 0;
}

/* ── startup ──────────────────────────────────────────────────────────────── */

void bench_init(void)
{
    cpu_prid   = cp0_prid();
    cpu_config = cp0_config();
    cpu_fir    = fir();
    switch (PRID_IMP(cpu_prid)) {
    case IMP_R4400: cpu_kind = BCPU_R4400; break;
    case IMP_R5000: cpu_kind = BCPU_R5000; break;
    default:        cpu_kind = 0; break;
    }
    have_l2 = (cpu_config & CFG_SC) == 0;

    testdev_probe();
    have_timebase = have_testdev && probe_timebase();

    exc_clear();
    exc_install();
    probe_work();
    calibrate_count();
}
