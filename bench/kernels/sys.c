/*
 * sys.c — the machine underneath the machine.
 *
 * These four have no meaning on a host and no golden checksum: they measure
 * paths that exist only because this is a MIPS running under an emulator —
 * address translation, exception entry and exit, cache maintenance, and
 * uncached device access. They are the kernels most likely to move when the
 * emulator's own internals change, and the ones an ALU benchmark can never
 * see. rules/perf and rules/jitv2 are full of work whose payoff shows up here
 * and nowhere else.
 */

#include "benchlib.h"

#if !defined(BENCH_HOST)

#include "cp0.h"

/* ── sys/tlb_hit — translation on the fast path ───────────────────────────── */

/*
 * Fill all 48 entries, then walk exactly those 48 pages. Every access
 * translates and every translation hits, so this is the cost of the lookup
 * itself — which is precisely what the tlbvmap fast path in mips_tlb.rs
 * exists to make cheap, and the only kernel here that can show it working.
 */
#define TLB_ENTRIES_USED  48
#define PAGE_BYTES        4096u
#define TLB_HIT_VA        0x00800000u

static unsigned char *tlb_phys;
static int tlb_mapped;

static u32 va_to_pfn(unsigned char *p) { return ((u32)(unsigned long)p & 0x1FFFFFFFu) >> 12; }

/* Park every entry on a distinct invalid VPN before writing real ones. Leaving
 * the power-on state in place risks two entries claiming the same VPN2, which
 * on real silicon is undefined and in IRIS trips the duplicate-entry checks
 * that rules/testing exists because of. */
static void tlb_reset_all(void)
{
    int i;
    cp0_pagemask_set(PM_4K);
    for (i = 0; i < 48; i++) {
        cp0_entryhi_set((u64)(s64)(s32)(0x40000000u + (u32)i * 0x2000u));
        cp0_entrylo0_set(0);
        cp0_entrylo1_set(0);
        cp0_index_set((u32)i);
        __asm__ __volatile__(".set push; .set mips3; .set noreorder\n\t"
                             "tlbwi\n\tnop\n\tnop\n\t.set pop" ::: "memory");
    }
}

static void tlb_hit_setup(void)
{
    int i;
    unsigned char *phys = (unsigned char *)work_alloc(TLB_ENTRIES_USED * 2u * PAGE_BYTES, 16384);
    u32 pfn0 = va_to_pfn(phys);

    tlb_phys = phys;
    tlb_reset_all();
    cp0_pagemask_set(PM_4K);
    /* One entry maps a pair of pages, so 48 entries cover 96 pages; the walk
     * below touches every one of them. */
    for (i = 0; i < TLB_ENTRIES_USED; i++) {
        u32 va = TLB_HIT_VA + (u32)i * 2u * PAGE_BYTES;
        u32 pfn = pfn0 + (u32)i * 2u;
        cp0_entryhi_set((u64)(s64)(s32)va);
        cp0_entrylo0_set(((u64)pfn << ELO_PFN_SHIFT) | ((u64)CA_CACHEABLE_NC << ELO_C_SHIFT) | ELO_D | ELO_V | ELO_G);
        cp0_entrylo1_set(((u64)(pfn + 1) << ELO_PFN_SHIFT) | ((u64)CA_CACHEABLE_NC << ELO_C_SHIFT) | ELO_D | ELO_V | ELO_G);
        cp0_index_set((u32)i);
        __asm__ __volatile__(".set push; .set mips3; .set noreorder\n\t"
                             "tlbwi\n\tnop\n\tnop\n\t.set pop" ::: "memory");
    }
    tlb_mapped = 1;

    /* Seed through the mapping so the pages are real and the first timed pass
     * is not paying for cold cache lines on top of the translation. */
    for (i = 0; i < TLB_ENTRIES_USED * 2; i++) {
        volatile u32 *p = (volatile u32 *)SEXT_PTR(TLB_HIT_VA + (u32)i * PAGE_BYTES);
        *p = (u32)i;
    }
}

static u64 tlb_hit_walk(u32 iters)
{
    u32 it, i, acc = 0;
    for (it = 0; it < iters; it++) {
        for (i = 0; i < TLB_ENTRIES_USED * 2; i++) {
            volatile u32 *p = (volatile u32 *)SEXT_PTR(TLB_HIT_VA + i * PAGE_BYTES);
            acc += *p;
        }
    }
    SINK(acc);
    return (u64)iters * TLB_ENTRIES_USED * 2u;
}

static u64 r_tlb_hit(u32 n)
{
    u64 w;
    tlb_hit_setup();
    w = tlb_hit_walk(n);
    tlb_reset_all();
    return w;
}

/* ── sys/tlb_miss — translation on the slow path ──────────────────────────── */

#define MISS_VA     0x02000000u
#define MISS_PAGES  2048u                    /* 8 MB: 42x the TLB, so every
                                              * touch refills */

extern u32 bench_tlb_refill[], bench_tlb_refill_end[];
extern u32 bench_pfn_delta;

static void install_refill(void)
{
    volatile u32 *dst;
    unsigned n = (unsigned)(bench_tlb_refill_end - bench_tlb_refill), i;
    if (n > 32) panic("refill handler too long for a vector slot");
    /* Both vectors: start.S runs with KX/SX/UX set, so a kuseg address refills
     * through the 64-bit XTLB vector, not the 32-bit one — and which of the
     * two fires is exactly the sort of thing worth being immune to. */
    dst = (volatile u32 *)SEXT_PTR(VEC_TLB_REFILL);
    for (i = 0; i < n; i++) dst[i] = bench_tlb_refill[i];
    dst = (volatile u32 *)SEXT_PTR(VEC_XTLB_REFILL);
    for (i = 0; i < n; i++) dst[i] = bench_tlb_refill[i];
    dcache_wb_range(SEXT_PTR(VEC_TLB_REFILL), 0x200);
    icache_inv_range(SEXT_PTR(VEC_TLB_REFILL), 0x200);
    SYNC();
}

static u64 r_tlb_miss(u32 n)
{
    unsigned char *phys = (unsigned char *)work_alloc(MISS_PAGES * PAGE_BYTES, 16384);
    u32 it, i, acc = 0;

    bench_pfn_delta = va_to_pfn(phys) - (MISS_VA >> 12);
    dcache_wb_range((volatile void *)&bench_pfn_delta, 4);

    tlb_reset_all();
    cp0_pagemask_set(PM_4K);
    install_refill();

    /* Prove the mapping before measuring it. A refill handler that computes
     * the wrong PFN does not fail — it faults on unmapped physical memory,
     * the shared dispatcher records and skips, and the kernel happily reports
     * a throughput figure for taking bus errors. Check that a word written
     * through KSEG0 reads back through the mapping, and stop if it does not. */
    {
        volatile u32 *k0 = (volatile u32 *)SEXT_PTR((u32)(unsigned long)phys);
        volatile u32 *va = (volatile u32 *)SEXT_PTR(MISS_VA);
        *k0 = 0x7EB1AB1Eu;
        dcache_wb_range((volatile void *)k0, 4);
        if (*va != 0x7EB1AB1Eu) {
            tlb_reset_all();
            exc_install();
            panic("sys/tlb_miss: refill handler does not map the region");
        }
    }

    /* Stride a whole page so no two consecutive touches share an entry, and
     * walk far more pages than the TLB holds so nothing survives to be reused. */
    for (it = 0; it < n; it++) {
        for (i = 0; i < MISS_PAGES; i++) {
            volatile u32 *p = (volatile u32 *)SEXT_PTR(MISS_VA + i * PAGE_BYTES);
            acc += *p;
        }
    }
    SINK(acc);

    tlb_reset_all();
    exc_install();                  /* hand the vectors back to the harness */
    return (u64)n * MISS_PAGES;
}

/* ── sys/exception — a full trap round trip ───────────────────────────────── */

/*
 * A `break` per iteration through the shared dispatcher: vector entry, eleven
 * CP0 reads, the EPC fixup and an eret. That is a heavier handler than a
 * syscall would meet in practice, but it is the same handler on every cell,
 * and an exception is the one operation where an emulator can be
 * catastrophically slower than the hardware it stands in for.
 */
static u64 r_exception(u32 n)
{
    u32 i;
    exc_clear();
    exc_resume_mode = EXC_RESUME_SKIP;
    for (i = 0; i < n; i++)
        __asm__ __volatile__(".set push; .set mips3; .set noreorder\n\t"
                             "break 0\n\tnop\n\t.set pop" ::: "memory");
    SINK((u32)exc.count);
    exc_clear();
    return (u64)n;
}

/* ── sys/cache_flush — cache maintenance over a range ─────────────────────── */

#define FLUSH_BYTES (256u * 1024u)

static u64 r_cache_flush(u32 n)
{
    unsigned char *p = (unsigned char *)work_alloc(FLUSH_BYTES, 4096);
    u32 it, i;
    for (it = 0; it < n; it++) {
        /* Dirty it first, or the writeback has nothing to do and the number is
         * a measurement of the "already clean" early-out instead. */
        for (i = 0; i < FLUSH_BYTES; i += 64) p[i] = (unsigned char)(it + i);
        dcache_wb_range(p, FLUSH_BYTES);
    }
    SINK(p[0]);
    return (u64)n * (FLUSH_BYTES / 16u);       /* cache ops issued */
}

/* ── sys/uncached — KSEG1, straight down the MC bus ───────────────────────── */

#define UNCACHED_BYTES (64u * 1024u)

static u64 r_uncached(u32 n)
{
    unsigned char *p = (unsigned char *)work_alloc(UNCACHED_BYTES, 4096);
    volatile u32 *k1 = (volatile u32 *)K1_PTR(p);
    u32 it, i, acc = 0;
    dcache_wb_range(p, UNCACHED_BYTES);
    for (it = 0; it < n; it++)
        for (i = 0; i < UNCACHED_BYTES / 4u; i++) acc += k1[i];
    SINK(acc);
    return (u64)n * UNCACHED_BYTES;
}

/* ── sys/llsc — load-linked / store-conditional, IRIX's lock primitive ────── */

static u64 r_llsc(u32 n)
{
    volatile u32 *p = (volatile u32 *)work_alloc(64, 64);
    u32 i, ok = 0;
    *p = 0;
    for (i = 0; i < n; i++) {
        u32 v, res;
        __asm__ __volatile__(".set push; .set mips3; .set noreorder\n\t"
                             "ll   %0, 0(%2)\n\t"
                             "addiu %0, %0, 1\n\t"
                             "move %1, %0\n\t"
                             "sc   %1, 0(%2)\n\t"
                             ".set pop"
                             : "=&r"(v), "=&r"(res) : "r"(p) : "memory");
        ok += res;
    }
    SINK(ok);
    return (u64)n;
}

/* ── registration ─────────────────────────────────────────────────────────── */

static const struct bench benches[] = {
    BENCH("sys/tlb_hit",     "xlat", 0, r_tlb_hit,     1u << 8,  BG_SYS),
    BENCH_EXC("sys/tlb_miss", "miss", 0, r_tlb_miss,   1u << 2,  BG_SYS),
    BENCH_EXC("sys/exception", "exc", 0, r_exception,  1u << 12, BG_SYS),
    BENCH("sys/cache_flush", "op",   0, r_cache_flush, 1u << 3,  BG_SYS),
    BENCH("sys/uncached",    "B",    0, r_uncached,    1u << 2,  BG_SYS),
    BENCH("sys/llsc",        "op",   0, r_llsc,        1u << 14, BG_SYS),
};

const struct bench_group group_sys = {
    "sys", benches, sizeof(benches) / sizeof(benches[0])
};

#endif /* !BENCH_HOST */
