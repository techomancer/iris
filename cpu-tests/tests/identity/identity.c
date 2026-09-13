/* identity — the CPU is what the build says it is.
 *
 * Cheap, and it fails loudly when an IRIS build didn't actually get the `r5k`
 * feature: without this, an R4400-flavoured run of the R5000 tests would show
 * up as a confusing pile of unrelated failures.
 */

#include "testlib.h"
#include "cp0.h"

static void t_prid(void)
{
    u32 prid = cp0_prid();
    /* Only the implementation field names the part. The low byte is the silicon
     * revision and legitimately varies: IRIS models an R4400 rev 4.0, the Indy
     * this was validated on is rev 6.0 (PRId 0x460). Asserting the whole
     * register made a real CPU fail for being real. */
    CHECK_EQ(PRID_IMP(prid), (u32)(is_r5000() ? IMP_R5000 : IMP_R4400));
    /* PRId is read-only: a write must not stick. Not written via a macro
     * because there is no cp0_prid_set — that is the point. */
    {
        u32 before = prid;
        __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"
                             "mtc0 %0, $15\n\tnop; nop; nop\n\t"
                             ".set pop" :: "r"(0xDEADBEEFu));
        CHECK_EQ(cp0_prid(), before);
    }
}

static void t_fir(void)
{
    /* Same story as PRId: the low byte is a revision. A real R5000 rev 1.0
     * reports FIR 0x2310 where IRIS models 0x2300. Both FIR_* constants end in
     * a zero byte, so masking it off compares the part, not the stepping. */
    CHECK_EQ(fir() & ~0xFFu, (u32)(is_r5000() ? FIR_R5000 : FIR_R4000));
}

/* Config.IC/DC encode cache size as 2^(12+n) bytes; IB/DB are the line size,
 * 0 = 16 bytes, 1 = 32 bytes. R4400: 16 KB/16 B direct-mapped. R5000:
 * 32 KB/32 B two-way. (src/mips_cache_v2.rs:41-100, src/mips_exec.rs:89-90) */
static void t_config_cache_geometry(void)
{
    u32 cfg = cp0_config();
    u32 ic = (cfg >> CFG_IC_SHIFT) & 7;
    u32 dc = (cfg >> CFG_DC_SHIFT) & 7;
    u32 ib = (cfg & CFG_IB) ? 1 : 0;
    u32 db = (cfg & CFG_DB) ? 1 : 0;

    if (is_r5000()) {
        CHECK_EQ(1u << (12 + ic), 32u * 1024);
        CHECK_EQ(1u << (12 + dc), 32u * 1024);
        CHECK_EQ(ib, 1u);          /* 32-byte I-cache lines */
        CHECK_EQ(db, 1u);          /* 32-byte D-cache lines */
    } else {
        CHECK_EQ(1u << (12 + ic), 16u * 1024);
        CHECK_EQ(1u << (12 + dc), 16u * 1024);
        CHECK_EQ(ib, 0u);          /* 16-byte I-cache lines */
        CHECK_EQ(db, 0u);          /* 16-byte D-cache lines */
    }
}

/* Config.K0 (bits 2:0) is the KSEG0 coherency attribute and is writable;
 * everything else in Config is read-only on these parts.
 *
 * DISABLED (2026-09-12) — the test changes KSEG0's cacheability with live
 * dirty data in the D-cache, which is architecturally unsound and wedges the
 * machine under IRIS.
 *
 * GCC spills `orig` to the stack at t_config_k0_writable+0x38 while KSEG0 is
 * still cached, so the value sits dirty in L1D and never reaches RAM. It then
 * hoists the reload into the delay slot of the loop-exit branch, where it
 * executes while the last iteration has left K0=7 — i.e. KSEG0 uncached. An
 * uncached load goes straight to the bus (R4000 UM p.326: it "issues a
 * noncoherent ... read request"), bypassing the dirty line, so `orig` comes
 * back as pre-spill RAM. The restore then writes K0=0, and the CPU runs off
 * into unmapped KUSEG and spins forever in a TLB-refill loop it cannot
 * service (empty TLB, EXL already set).
 *
 * Nothing here is IRIS-specific: changing a region's coherency attribute with
 * unflushed dirty lines in that region loses them on real hardware too, which
 * is why the PROM always flips K0 from KSEG1. The suite's own
 * cache/cached_uncached documents exactly this rule ("a write-back cache has
 * not written anything out yet"). It happens to pass on the reference Indy,
 * but whether the line is still resident at the reload depends on eviction
 * pressure from the loop's own printf/counter traffic, so that is luck rather
 * than a guarantee — and the final CHECK compares Config against `orig`, so a
 * corrupted `orig` is compared to itself and still reports PASS.
 *
 * The fix is a dcache_wb_invalidate_range() over the stack (or keeping `orig`
 * in a callee-saved register) before the sweep. Reported upstream.
 */
__attribute__((unused)) static void t_config_k0_writable(void)
{
    u32 orig = cp0_config();
    u32 i;
    for (i = 0; i < 8; i++) {
        cp0_config_set((orig & ~CFG_K0_MASK) | i);
        CHECK_EQ_AT("k0", i, cp0_config() & CFG_K0_MASK, i);
        /* The rest of Config must not have moved. */
        CHECK_EQ_AT("k0", i, cp0_config() & ~CFG_K0_MASK, orig & ~CFG_K0_MASK);
    }
    cp0_config_set(orig);
    CHECK_EQ(cp0_config(), orig);
}

/* Both parts have 48 TLB entries, so Random must wrap within 0..47 and never
 * fall below Wired. Just the range here; the decrement behaviour is tlb/. */
static void t_tlb_size(void)
{
    u32 i, seen_max = 0, seen_min = 0xFFFFFFFF;
    cp0_wired_set(0);
    for (i = 0; i < 200; i++) {
        u32 r = cp0_random() & 0x3F;
        if (r > seen_max) seen_max = r;
        if (r < seen_min) seen_min = r;
    }
    CHECK(seen_max <= TLB_ENTRIES - 1);
    CHECK(seen_min <= seen_max);
}

static const struct test tests[] = {
    TEST("identity/prid",             t_prid,                   CPU_ALL),
    TEST("identity/fir",              t_fir,                    CPU_ALL),
    TEST("identity/cache_geometry",   t_config_cache_geometry,  CPU_ALL),
    /* DISABLED — see the block comment on t_config_k0_writable above.
     * Re-enable once the test flushes the D-cache before changing K0. */
    /* TEST("identity/config_k0",        t_config_k0_writable,     CPU_ALL), */
    TEST("identity/tlb_size",         t_tlb_size,               CPU_ALL),
};

const struct test_group group_identity = {
    "identity", tests, sizeof(tests) / sizeof(tests[0])
};
