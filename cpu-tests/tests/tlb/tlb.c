/* tlb — TLB entry management, page sizes, ASIDs, and the miss/invalid/modified
 * exceptions.
 *
 * These tests are only safe because the suite runs from KSEG0, which is
 * unmapped: rewriting all 48 entries cannot unmap the code doing the
 * rewriting. See docs/memory-map.md.
 *
 * Everything is done through a scratch virtual region well away from anything
 * else, and every test puts the TLB back the way it found it.
 */

#include "testlib.h"
#include "cp0.h"
#include "excoff.h"

#define A ".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"
#define Z "\n\t.set pop"

extern char _scratch_start[];

/*
 * A virtual address in kuseg that nothing else uses. 0x00400000 is 4 MB in:
 * clear of the low 512 KB physical alias and of anything the PROM leaves
 * behind, and 4 MB-aligned so a large page can map it.
 */
#define TEST_VA   0x00400000ull
#define TEST_VA2  0x00800000ull

/* Physical page to map: taken from the suite's own scratch area, so writing
 * through the mapping is writing to memory we already own. */
static u64 scratch_phys(void)
{
    return (u64)((u32)(unsigned long)_scratch_start & 0x1FFFFFFFu);
}

/* ── raw entry round-trip ─────────────────────────────────────────────────── */

/* Write an entry with TLBWI, read it back with TLBR, and compare. */
static void write_entry(u32 index, u64 vpn2, u64 lo0, u64 lo1, u32 pagemask)
{
    cp0_index_set(index);
    cp0_pagemask_set(pagemask);
    cp0_entryhi_set(vpn2);
    cp0_entrylo0_set(lo0);
    cp0_entrylo1_set(lo1);
    tlb_write_indexed();
}

/* Build an EntryLo for a physical address with the given flags. */
static u64 entrylo(u64 phys, u64 flags)
{
    return ((phys >> 12) << ELO_PFN_SHIFT) |
           ((u64)CA_CACHEABLE_NC << ELO_C_SHIFT) | flags;
}

static void t_tlbwi_tlbr_round_trip(void)
{
    u64 saved_hi = cp0_entryhi();
    u32 saved_pm = cp0_pagemask();
    u64 phys = scratch_phys();
    u64 lo0 = entrylo(phys, ELO_V | ELO_D | ELO_G);
    u64 lo1 = entrylo(phys + 0x1000, ELO_V | ELO_G);

    write_entry(10, TEST_VA, lo0, lo1, PM_4K);

    /* Read it back from a different starting state so a no-op TLBR would be
     * visible. */
    cp0_entryhi_set(0);
    cp0_entrylo0_set(0);
    cp0_entrylo1_set(0);
    cp0_pagemask_set(PM_16M);
    cp0_index_set(10);
    tlb_read();

    CHECK_EQ(cp0_pagemask(), PM_4K);
    CHECK_EQ(cp0_entryhi() & ~0x1FFFull, TEST_VA);
    CHECK_EQ(cp0_entrylo0() & 0x3FFFFFFFull, lo0 & 0x3FFFFFFFull);
    CHECK_EQ(cp0_entrylo1() & 0x3FFFFFFFull, lo1 & 0x3FFFFFFFull);

    /* Retire the entry. */
    write_entry(10, 0x1FFFE000ull, 0, 0, PM_4K);
    cp0_entryhi_set(saved_hi);
    cp0_pagemask_set(saved_pm);
}

/* Every one of the 48 entries must be independently addressable. */
static void t_all_entries_round_trip(void)
{
    u64 saved_hi = cp0_entryhi();
    u32 i;
    u64 phys = scratch_phys();

    for (i = 0; i < TLB_ENTRIES; i++) {
        /* A distinct VPN2 per entry, far from anything real. */
        u64 va = 0x10000000ull + ((u64)i << 13);
        write_entry(i, va, entrylo(phys, ELO_V | ELO_G), 0, PM_4K);
    }
    for (i = 0; i < TLB_ENTRIES; i++) {
        u64 va = 0x10000000ull + ((u64)i << 13);
        cp0_entryhi_set(0);
        cp0_index_set(i);
        tlb_read();
        CHECK_EQ_AT("entry", i, cp0_entryhi() & ~0x1FFFull, va);
    }
    /* Retire all of them to distinct unmapped VPNs so nothing matches later. */
    for (i = 0; i < TLB_ENTRIES; i++)
        write_entry(i, 0x1FFF0000ull + ((u64)i << 13), 0, 0, PM_4K);
    cp0_entryhi_set(saved_hi);
}

/* ── TLBP ─────────────────────────────────────────────────────────────────── */

static void t_tlbp_hit_and_miss(void)
{
    u64 saved_hi = cp0_entryhi();
    u64 phys = scratch_phys();

    write_entry(7, TEST_VA, entrylo(phys, ELO_V | ELO_G), 0, PM_4K);

    /* Probe for it: Index must come back as 7 with bit 31 clear. */
    cp0_entryhi_set(TEST_VA);
    tlb_probe();
    CHECK_EQ(cp0_index() & 0x80000000u, 0u);
    CHECK_EQ(cp0_index() & 0x3Fu, 7u);

    /* Probe for something not mapped: Index bit 31 set. */
    cp0_entryhi_set(0x1E000000ull);
    tlb_probe();
    CHECK_EQ(cp0_index() & 0x80000000u, 0x80000000u);

    write_entry(7, 0x1FFFE000ull, 0, 0, PM_4K);
    cp0_entryhi_set(saved_hi);
}

/* ── page sizes ───────────────────────────────────────────────────────────── */

/*
 * Each PageMask value must round-trip through a TLB entry, and the VPN2 stored
 * must be masked to that page size — the low bits covered by the mask are not
 * part of the tag.
 */
static void t_page_sizes_round_trip(void)
{
    static const u32 masks[] = { PM_4K, PM_16K, PM_64K, PM_256K,
                                 PM_1M, PM_4M, PM_16M };
    u64 saved_hi = cp0_entryhi();
    u32 saved_pm = cp0_pagemask();
    u64 phys = scratch_phys();
    unsigned i;

    for (i = 0; i < sizeof(masks) / sizeof(masks[0]); i++) {
        /* A VA aligned to twice the page size (an entry covers an even/odd
         * pair), so no VPN2 bits are lost. */
        u64 va = 0x20000000ull;
        write_entry(5, va, entrylo(phys, ELO_V | ELO_G), 0, masks[i]);

        cp0_pagemask_set(PM_4K);
        cp0_entryhi_set(0);
        cp0_index_set(5);
        tlb_read();
        CHECK_EQ_AT("mask", i, cp0_pagemask(), masks[i]);
        CHECK_EQ_AT("mask", i, cp0_entryhi() & ~(u64)(masks[i] | 0x1FFF), va);
    }

    write_entry(5, 0x1FFFE000ull, 0, 0, PM_4K);
    cp0_entryhi_set(saved_hi);
    cp0_pagemask_set(saved_pm);
}

/* ── translation through a mapped page ────────────────────────────────────── */

/*
 * The real thing: map a scratch physical page at TEST_VA, write through the
 * mapping, and read the same bytes back through KSEG0. If translation works,
 * the two views agree.
 */
static void t_translation_actually_works(void)
{
    u64 saved_hi = cp0_entryhi();
    u64 phys = scratch_phys();
    volatile u32 *via_k0 = (volatile u32 *)_scratch_start;
    volatile u32 *via_tlb = (volatile u32 *)(unsigned long)TEST_VA;

    /* Map TEST_VA -> scratch page, cacheable, valid, dirty (writable), global
     * so the current ASID does not matter. Even page only; the odd half is
     * left invalid. */
    write_entry(3, TEST_VA, entrylo(phys, ELO_V | ELO_D | ELO_G), 0, PM_4K);

    exc_clear();
    *via_k0 = 0x5A5A0001u;
    SYNC();
    CHECK_EQ(*via_tlb, 0x5A5A0001u);
    CHECK_NO_EXC();

    /* And the other direction. */
    *via_tlb = 0xA5A50002u;
    SYNC();
    CHECK_EQ(*via_k0, 0xA5A50002u);
    CHECK_NO_EXC();

    write_entry(3, 0x1FFFE000ull, 0, 0, PM_4K);
    cp0_entryhi_set(saved_hi);
}

/* ── V and D bits ─────────────────────────────────────────────────────────── */

/* A page with V clear raises TLB Invalid (TLBL on a load), not a refill. */
static void t_invalid_page_raises_tlbl(void)
{
    u64 saved_hi = cp0_entryhi();
    u64 phys = scratch_phys();
    volatile u32 *p = (volatile u32 *)(unsigned long)TEST_VA;
    u64 v = 0;

    write_entry(3, TEST_VA, entrylo(phys, ELO_G), 0, PM_4K);   /* V clear */

    exc_clear();
    __asm__ __volatile__(A "lw %0, 0(%1)" Z : "+r"(v) : "r"(p));

    CHECK_EQ(exc.count, 1u);
    CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_TLBL);
    /* Invalid goes to the general vector, not the refill vector — the entry
     * matched, it just is not usable. */
    CHECK_EQ(exc.vector, (u32)VECID_GENERAL);
    CHECK_EQ(exc.badvaddr, TEST_VA);

    write_entry(3, 0x1FFFE000ull, 0, 0, PM_4K);
    cp0_entryhi_set(saved_hi);
}

/* A page with V set but D clear is readable but raises TLB Modified on a
 * store. */
static void t_readonly_page_raises_mod(void)
{
    u64 saved_hi = cp0_entryhi();
    u64 phys = scratch_phys();
    volatile u32 *p = (volatile u32 *)(unsigned long)TEST_VA;
    volatile u32 *k0 = (volatile u32 *)_scratch_start;
    u64 v = 0, val = 0xDEADull;

    *k0 = 0x11223344u;
    SYNC();
    write_entry(3, TEST_VA, entrylo(phys, ELO_V | ELO_G), 0, PM_4K); /* no D */

    /* Reading is fine. */
    exc_clear();
    __asm__ __volatile__(A "lw %0, 0(%1)" Z : "+r"(v) : "r"(p));
    CHECK_NO_EXC();
    CHECK_EQ(v, 0x11223344ull);

    /* Writing raises TLB Modified. */
    exc_clear();
    __asm__ __volatile__(A "sw %0, 0(%1)" Z :: "r"(val), "r"(p) : "memory");
    CHECK_EQ(exc.count, 1u);
    CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_MOD);
    CHECK_EQ(exc.vector, (u32)VECID_GENERAL);
    CHECK_EQ(exc.badvaddr, TEST_VA);
    /* And memory must be unchanged. */
    SYNC();
    CHECK_EQ(*k0, 0x11223344u);

    write_entry(3, 0x1FFFE000ull, 0, 0, PM_4K);
    cp0_entryhi_set(saved_hi);
}

/* ── ASIDs ────────────────────────────────────────────────────────────────── */

/*
 * A non-global entry matches only when EntryHi's ASID equals the entry's. The
 * same VA under a different ASID must miss.
 */
static void t_asid_match_and_mismatch(void)
{
    u64 saved_hi = cp0_entryhi();
    u64 phys = scratch_phys();

    /* Entry tagged with ASID 0x42, not global. */
    write_entry(3, TEST_VA | 0x42ull, entrylo(phys, ELO_V | ELO_D), 0, PM_4K);

    /* Probe with the matching ASID: hit. */
    cp0_entryhi_set(TEST_VA | 0x42ull);
    tlb_probe();
    CHECK_EQ(cp0_index() & 0x80000000u, 0u);

    /* Probe with a different ASID: miss. */
    cp0_entryhi_set(TEST_VA | 0x43ull);
    tlb_probe();
    CHECK_EQ(cp0_index() & 0x80000000u, 0x80000000u);

    write_entry(3, 0x1FFFE000ull, 0, 0, PM_4K);
    cp0_entryhi_set(saved_hi);
}

/* A global entry matches regardless of ASID. G is the AND of both EntryLo G
 * bits, so both halves must set it. */
static void t_global_entry_ignores_asid(void)
{
    u64 saved_hi = cp0_entryhi();
    u64 phys = scratch_phys();

    write_entry(3, TEST_VA | 0x42ull,
                entrylo(phys, ELO_V | ELO_D | ELO_G),
                entrylo(phys + 0x1000, ELO_V | ELO_G), PM_4K);

    cp0_entryhi_set(TEST_VA | 0x99ull);
    tlb_probe();
    CHECK_EQ(cp0_index() & 0x80000000u, 0u);

    write_entry(3, 0x1FFFE000ull, 0, 0, PM_4K);
    cp0_entryhi_set(saved_hi);
}

/* ── refill ───────────────────────────────────────────────────────────────── */

/*
 * An access to an unmapped kuseg address takes a TLB refill. Which vector it
 * uses depends on the addressing mode: 32-bit addresses use the 0x80000000
 * refill vector, 64-bit ones the 0x80000080 XTLB vector. The suite runs with
 * Status.KX set, so a kuseg (xkuseg, really) address uses the XTLB vector.
 *
 * EntryHi, Context and XContext must all be loaded with the faulting VPN2.
 */
static void t_refill_vector_and_context(void)
{
    u64 saved_hi = cp0_entryhi();
    volatile u32 *p = (volatile u32 *)(unsigned long)TEST_VA2;
    u64 v = 0;

    /* Make sure nothing maps TEST_VA2. */
    cp0_entryhi_set(TEST_VA2);
    tlb_probe();
    if ((cp0_index() & 0x80000000u) == 0) {
        u32 idx = cp0_index() & 0x3F;
        write_entry(idx, 0x1FFFE000ull, 0, 0, PM_4K);
    }

    exc_clear();
    __asm__ __volatile__(A "lw %0, 0(%1)" Z : "+r"(v) : "r"(p));

    CHECK_EQ(exc.count, 1u);
    CHECK_EQ(CAUSE_EXC(exc.cause), (u32)EXC_TLBL);
    CHECK_EQ(exc.badvaddr, TEST_VA2);
    /* EntryHi is loaded with the VPN2 that missed. */
    CHECK_EQ(exc.entryhi & ~0x1FFFull, TEST_VA2);
    /* Context.BadVPN2 (bits 22:4) is VA[31:13]. */
    CHECK_EQ((exc.context >> 4) & 0x7FFFFull, TEST_VA2 >> 13);
    /* With KX set, the refill goes to the XTLB vector. */
    con_printf("\n      [refill vector = %u (1=TLB 2=XTLB), Status.KX=%u]",
               exc.vector, (exc.status & ST_KX) ? 1u : 0u);
    CHECK(exc.vector == (u32)VECID_XTLB || exc.vector == (u32)VECID_TLB);

    cp0_entryhi_set(saved_hi);
}

static const struct test tests[] = {
    TEST("tlb/tlbwi_tlbr",         t_tlbwi_tlbr_round_trip,       CPU_ALL),
    TEST("tlb/all_entries",        t_all_entries_round_trip,      CPU_ALL),
    TEST("tlb/tlbp_hit_miss",      t_tlbp_hit_and_miss,           CPU_ALL),
    TEST("tlb/page_sizes",         t_page_sizes_round_trip,       CPU_ALL),
    TEST("tlb/translation_works",  t_translation_actually_works,  CPU_ALL),
    TEST("tlb/invalid_page_tlbl",  t_invalid_page_raises_tlbl,    CPU_ALL),
    TEST("tlb/readonly_page_mod",  t_readonly_page_raises_mod,    CPU_ALL),
    TEST("tlb/asid_match",         t_asid_match_and_mismatch,     CPU_ALL),
    TEST("tlb/global_ignores_asid", t_global_entry_ignores_asid,  CPU_ALL),
    TEST("tlb/refill_context",     t_refill_vector_and_context,   CPU_ALL),
};

const struct test_group group_tlb = {
    "tlb", tests, sizeof(tests) / sizeof(tests[0])
};
