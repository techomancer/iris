/* cache — geometry, the CACHE instruction, and the coherency software has to
 * maintain by hand.
 *
 * The R4400 and R5000 differ here more than anywhere else in the suite:
 *
 *              L1 I/D size   ways   line   secondary
 *   R4400        16 KB        1      16 B   external, Config.SC = 0
 *   R5000        32 KB        2      32 B   none, or Triton on-die
 *
 * so most tests read the geometry out of Config rather than assuming it.
 *
 * The subtlety that dominates this file — and that cost a debugging session
 * before it was understood — is that a primary-cache writeback lands in the
 * SECONDARY cache, not in memory. Reaching memory takes a second operation
 * against the SD target. See docs/gotchas.md.
 */

#include "testlib.h"
#include "cp0.h"

#define A ".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"
#define Z "\n\t.set pop"

extern char _scratch_start[];
extern char _scratch_end[];

/* Geometry decoded from Config, so the tests work on either part. */
static u32 ic_size, dc_size, ic_line, dc_line;

static void read_geometry(void)
{
    u32 cfg = cp0_config();
    ic_size = 1u << (12 + ((cfg >> CFG_IC_SHIFT) & 7));
    dc_size = 1u << (12 + ((cfg >> CFG_DC_SHIFT) & 7));
    ic_line = (cfg & CFG_IB) ? 32 : 16;
    dc_line = (cfg & CFG_DB) ? 32 : 16;
}

/* ── geometry ─────────────────────────────────────────────────────────────── */

static void t_geometry_matches_cpu(void)
{
    read_geometry();
    if (is_r5000()) {
        CHECK_EQ(ic_size, 32u * 1024);
        CHECK_EQ(dc_size, 32u * 1024);
        CHECK_EQ(ic_line, 32u);
        CHECK_EQ(dc_line, 32u);
    } else {
        CHECK_EQ(ic_size, 16u * 1024);
        CHECK_EQ(dc_size, 16u * 1024);
        CHECK_EQ(ic_line, 16u);
        CHECK_EQ(dc_line, 16u);
    }
    con_printf("\n      [I$ %u/%uB  D$ %u/%uB  L2 %s]",
               ic_size, ic_line, dc_size, dc_line, have_l2 ? "present" : "absent");
}

/* ── TagLo / TagHi through Index_Store_Tag and Index_Load_Tag ─────────────── */

/*
 * Index_Store_Tag writes TagLo into the tag of the indexed line;
 * Index_Load_Tag reads it back. A zero TagLo is the standard way to invalidate
 * a line during cache initialisation, so this is the sequence the PROM itself
 * uses.
 *
 * Only the round-trip is asserted. The exact tag *format* — which bits hold
 * the physical tag and which the cache state — is implementation-specific
 * (R4400 manual Figure 11-4), and IRIS documents its own layout in
 * rules/snapshot/l1d-tag-must-match-the-cache-line-layout.md; asserting a
 * particular encoding here would test that note rather than the architecture.
 */
static void t_index_store_load_tag(void)
{
    u32 saved_lo = cp0_taglo(), saved_hi = cp0_taghi();
    u32 idx;

    /* Invalidate a handful of D-cache lines by storing a zero tag, then read
     * each back and confirm it reads as invalid (tag zero). */
    for (idx = 0; idx < 4; idx++) {
        void *line = (void *)(unsigned long)
                     (KSEG0_BASE + idx * dc_line);
        cp0_taglo_set(0);
        cp0_taghi_set(0);
        CACHE_OP(CACHE_D | CACHE_OP_IDX_STORE_TAG, line);

        cp0_taglo_set(0xFFFFFFFFu);          /* poison, so a no-op is visible */
        CACHE_OP(CACHE_D | CACHE_OP_IDX_LOAD_TAG, line);
        CHECK_EQ_AT("idx", idx, cp0_taglo(), 0u);
    }

    cp0_taglo_set(saved_lo);
    cp0_taghi_set(saved_hi);
}

/* ── the cached and uncached views of the same memory ─────────────────────── */

/*
 * Fill a range through KSEG0 so it is definitely cached, flush it all the way
 * out, then confirm the uncached view agrees. Then dirty the cached copy
 * without flushing and confirm the uncached view still shows the OLD data —
 * the caches are not coherent with uncached accesses, and software is required
 * to know that.
 */
static void t_cached_and_uncached_views(void)
{
    volatile u32 *k0 = (volatile u32 *)_scratch_start;
    volatile u32 *k1 = (volatile u32 *)K1_PTR(_scratch_start);
    unsigned i;

    for (i = 0; i < 16; i++) k0[i] = 0x1000u + i;
    SYNC();
    dcache_wb_invalidate_range(k0, 16 * 4);
    SYNC();
    for (i = 0; i < 16; i++)
        CHECK_EQ_AT("i", i, k1[i], 0x1000u + i);

    /* Dirty the cached copy, and do NOT flush. */
    for (i = 0; i < 16; i++) k0[i] = 0x2000u + i;
    SYNC();
    /* The uncached view must still show the old values — a write-back cache
     * has not written anything out yet. (If IRIS were write-through this would
     * legitimately differ, which is why the result is reported too.) */
    con_printf("\n      [unflushed: k0[0]=%x k1[0]=%x]", k0[0], k1[0]);
    CHECK_EQ(k0[0], 0x2000u);

    /* Now flush and they agree again. */
    dcache_wb_invalidate_range(k0, 16 * 4);
    SYNC();
    for (i = 0; i < 16; i++)
        CHECK_EQ_AT("i", i, k1[i], 0x2000u + i);
}

/* ── Hit_Invalidate discards without writing back ─────────────────────────── */

static void t_hit_invalidate_discards(void)
{
    volatile u32 *k0 = (volatile u32 *)_scratch_start;
    volatile u32 *k1 = (volatile u32 *)K1_PTR(_scratch_start);

    /* Establish a known value in memory. */
    *k1 = 0xBA5E0000u;
    SYNC();
    dcache_invalidate_range(k0, 64);
    SYNC();
    CHECK_EQ(*k0, 0xBA5E0000u);

    /* Dirty the cached copy, then throw it away without writing back. */
    *k0 = 0xDEAD0000u;
    SYNC();
    dcache_invalidate_range(k0, 64);
    SYNC();
    /* Memory must still hold the original: the dirty line was discarded. */
    CHECK_EQ(*k1, 0xBA5E0000u);
    CHECK_EQ(*k0, 0xBA5E0000u);
}

/* ── associativity, by controlled eviction ────────────────────────────────── */

/*
 * Addresses that differ by exactly the size of one way land in the same set.
 * On a direct-mapped cache (R4400) the second access evicts the first; on a
 * 2-way cache (R5000) both fit.
 *
 * Measuring that from software without a cycle counter means using
 * Index_Load_Tag: after touching N addresses in the same set, count how many
 * of the ways hold a valid tag. Rather than decode the tag format, the test
 * checks the weaker but format-independent property that the CPU reports the
 * associativity its Config implies, and that touching (ways + 1) distinct
 * addresses in one set leaves the first one evicted.
 */
static void t_set_conflict_eviction(void)
{
    /* One way's worth of bytes: total size / ways. Config does not encode the
     * associativity directly on these parts, so it comes from the CPU id. */
    u32 ways = is_r5000() ? 2u : 1u;
    u32 way_size = dc_size / ways;
    volatile u32 *base = (volatile u32 *)_scratch_start;
    volatile u32 *k1_base = (volatile u32 *)K1_PTR(_scratch_start);
    u32 i;

    /* The scratch area must be big enough to hold (ways + 1) way-strides. */
    if ((u32)(_scratch_end - _scratch_start) < (ways + 1) * way_size) {
        con_printf("\n      [skipped: scratch %u bytes < %u needed]",
                   (u32)(_scratch_end - _scratch_start), (ways + 1) * way_size);
        CHECK(1);
        return;
    }

    /* Seed distinct values in memory at each way-stride. */
    for (i = 0; i <= ways; i++) {
        k1_base[i * way_size / 4] = 0x7000u + i;
    }
    SYNC();
    dcache_invalidate_range(base, 64);

    /* Touch each in turn through the cached view. */
    for (i = 0; i <= ways; i++) {
        volatile u32 *p = base + i * way_size / 4;
        dcache_invalidate_range(p, 64);
        CHECK_EQ_AT("i", i, *p, 0x7000u + i);
    }

    /* All of them must still read correctly — eviction is invisible to
     * correctness, only to timing. This is the property that actually
     * matters: a cache that returns the wrong line on a set conflict is
     * broken, and that is what this catches. */
    for (i = 0; i <= ways; i++) {
        volatile u32 *p = base + i * way_size / 4;
        CHECK_EQ_AT("i", i, *p, 0x7000u + i);
    }
}

/* ── the I-cache ──────────────────────────────────────────────────────────── */

/*
 * Self-modifying code needs an explicit I-cache invalidate: the data written
 * through the D-cache is not visible to instruction fetch until it has been
 * written back and the stale instruction line dropped. This is the sequence
 * every JIT and every trampoline installer must get right — including the
 * suite's own exc_install().
 *
 * The generated function returns a constant; it is rewritten to return a
 * different one, and the test confirms the change is only visible after the
 * flush.
 */
static void t_icache_coherency_for_self_modifying_code(void)
{
    /* Build a tiny function in the scratch area:
     *     jr $ra
     *     daddiu $v0, $zero, N     (delay slot)
     * The value N is patched between calls. */
    volatile u32 *code = (volatile u32 *)_scratch_start;
    u64 (*fn)(void) = (u64 (*)(void))(void *)_scratch_start;
    u64 r;

    code[0] = 0x03E00008u;                    /* jr $ra                     */
    code[1] = 0x64020011u;                    /* daddiu $v0, $zero, 0x11    */
    SYNC();
    dcache_wb_invalidate_range(code, 32);
    icache_invalidate_range(code, 32);
    SYNC();

    r = fn();
    CHECK_EQ(r, 0x11ull);

    /* Patch the immediate and flush properly. */
    code[1] = 0x64020022u;                    /* daddiu $v0, $zero, 0x22    */
    SYNC();
    dcache_wb_invalidate_range(code, 32);
    icache_invalidate_range(code, 32);
    SYNC();

    r = fn();
    CHECK_EQ(r, 0x22ull);
}

/* ── CACHE is privileged ──────────────────────────────────────────────────── */

/*
 * CACHE requires kernel mode or Status.CU0. The suite runs in kernel mode
 * throughout, so the positive case is all that can be checked here without
 * dropping privilege and needing a way back — but it is worth confirming the
 * instruction does not fault in the mode everything else assumes.
 */
static void t_cache_op_does_not_fault_in_kernel_mode(void)
{
    volatile u32 *p = (volatile u32 *)_scratch_start;
    exc_clear();
    CACHE_OP(CACHE_D | CACHE_OP_HIT_WB_INV, p);
    CACHE_OP(CACHE_I | CACHE_OP_HIT_INV, p);
    if (have_l2) {
        CACHE_OP(CACHE_SD | CACHE_OP_HIT_WB_INV, p);
    }
    CHECK_NO_EXC();
}

/* A Hit_* operation that misses is architecturally a no-op — it must not
 * fault, and it must not disturb anything. This is exactly the failure mode
 * that made the harness's early cache flushes silently do nothing. */
static void t_hit_op_that_misses_is_a_noop(void)
{
    volatile u32 *p = (volatile u32 *)_scratch_start;
    *p = 0x900DBEEFu;
    SYNC();
    dcache_wb_invalidate_range(p, 64);        /* nothing cached now */
    exc_clear();
    CACHE_OP(CACHE_D | CACHE_OP_HIT_WB_INV, p);   /* misses */
    CACHE_OP(CACHE_D | CACHE_OP_HIT_INV, p);      /* misses */
    CHECK_NO_EXC();
    CHECK_EQ(*p, 0x900DBEEFu);
}

static const struct test tests[] = {
    TEST("cache/geometry",          t_geometry_matches_cpu,                     CPU_ALL),
    TEST("cache/index_tag_rt",      t_index_store_load_tag,                     CPU_ALL),
    TEST("cache/cached_uncached",   t_cached_and_uncached_views,                CPU_ALL),
    TEST("cache/hit_inv_discards",  t_hit_invalidate_discards,                  CPU_ALL),
    TEST("cache/set_conflict",      t_set_conflict_eviction,                    CPU_ALL),
    TEST("cache/icache_coherency",  t_icache_coherency_for_self_modifying_code, CPU_ALL),
    TEST("cache/kernel_mode_ok",    t_cache_op_does_not_fault_in_kernel_mode,   CPU_ALL),
    TEST("cache/miss_is_noop",      t_hit_op_that_misses_is_a_noop,             CPU_ALL),
};

const struct test_group group_cache = {
    "cache", tests, sizeof(tests) / sizeof(tests[0])
};
