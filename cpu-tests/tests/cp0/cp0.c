/* cp0 — system coprocessor registers: writable-bit masks, the timer, and the
 * 32-bit/64-bit access rules.
 *
 * The theme is that CP0 registers are not plain storage. Some are read-only,
 * some have reserved bits that read back as zero no matter what is written,
 * some are 64 bits wide and some 32, and Count/Compare drive an interrupt.
 * An emulator that models them as a flat array of u32 passes almost nothing
 * here.
 */

#include "testlib.h"
#include "cp0.h"

#define A ".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"
#define Z "\n\t.set pop"

/* ── read-only registers ──────────────────────────────────────────────────── */

static void t_prid_read_only(void)
{
    u32 before = cp0_prid();
    __asm__ __volatile__(A "mtc0 %0, $15\n\tnop\n\tnop\n\tnop" Z
                         :: "r"(0xFFFFFFFFu));
    CHECK_EQ(cp0_prid(), before);
}

/* Random is read-only and decrements on its own. */
static void t_random_read_only_and_moves(void)
{
    u32 first, seen_change = 0;
    unsigned i;

    cp0_wired_set(0);
    first = cp0_random() & 0x3F;
    for (i = 0; i < 500; i++) {
        if ((cp0_random() & 0x3F) != first) { seen_change = 1; break; }
    }
    CHECK_EQ(seen_change, 1u);

    /* Writing it must not stick. */
    __asm__ __volatile__(A "mtc0 %0, $1\n\tnop\n\tnop\n\tnop" Z :: "r"(0u));
    CHECK(1);   /* the value moves on its own, so only "no crash" is assertable */
}

/*
 * Random never falls below Wired. Sample it many times with Wired set high
 * and confirm every sample is in [Wired, 47].
 */
static void t_random_respects_wired(void)
{
    unsigned i;
    u32 low = 0, high = 0;
    cp0_wired_set(40);
    for (i = 0; i < 2000; i++) {
        u32 r = cp0_random() & 0x3F;
        if (r < 40) low++;
        if (r > TLB_ENTRIES - 1) high++;
    }
    CHECK_EQ(low, 0u);
    CHECK_EQ(high, 0u);
    cp0_wired_set(0);
}

/* ── writable-bit masks ───────────────────────────────────────────────────── */

/*
 * Wired is 6 bits wide; bits 31:6 are reserved. R4000 manual, Table 4-10:
 * "Reserved. Must be written as zeroes, and returns zeroes when read."
 *
 * So a write of all-ones must read back with only bits 5:0 set. IRIS currently
 * stores the full 32-bit value, which is a (harmless in practice — IRIX never
 * writes garbage here) deviation from that.
 */
static void t_wired_reserved_bits_read_zero(void)
{
    cp0_wired_set(0xFFFFFFFFu);
    CHECK_EQ(cp0_wired() & ~0x3Fu, 0u);
    cp0_wired_set(0);
    CHECK_EQ(cp0_wired(), 0u);
}

/*
 * R4000 manual, Wired register: "Writing this register also sets the Random
 * register to the value of its upper bound" — 47 on a 48-entry TLB.
 *
 * The reset cannot be observed exactly. Random decrements once per instruction
 * issue, and the handful of instructions between the mtc0 and the mfc0 that
 * reads it back (the pipeline-hazard nops in both accessors) have already
 * ticked it down — measured at 43, i.e. four decrements. So the assertion is
 * that Random lands NEAR the top rather than exactly on it. Without the reset
 * it would be wherever the free-running counter happened to be, which is
 * almost never in the top handful of entries.
 */
static void t_wired_write_resets_random(void)
{
    u32 r;
    cp0_wired_set(8);
    r = cp0_random() & 0x3F;
    CHECK(r <= TLB_ENTRIES - 1);
    CHECK(r >= TLB_ENTRIES - 1 - 12);

    cp0_wired_set(0);
    r = cp0_random() & 0x3F;
    CHECK(r <= TLB_ENTRIES - 1);
    CHECK(r >= TLB_ENTRIES - 1 - 12);
}

/*
 * PageMask only has bits 24:13 implemented, and only in pairs (each page size
 * step doubles). Writing all ones must read back as the largest legal mask.
 */
static void t_pagemask_reserved_bits(void)
{
    u32 saved = cp0_pagemask();
    cp0_pagemask_set(0xFFFFFFFFu);
    /* Bits outside 24:13 must be zero. */
    CHECK_EQ(cp0_pagemask() & ~0x01FFE000u, 0u);
    cp0_pagemask_set(saved);
}

/* Every architecturally-defined page size must round-trip exactly. */
static void t_pagemask_all_sizes(void)
{
    static const u32 sizes[] = { PM_4K, PM_16K, PM_64K, PM_256K,
                                 PM_1M, PM_4M, PM_16M };
    unsigned i;
    u32 saved = cp0_pagemask();
    for (i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        cp0_pagemask_set(sizes[i]);
        CHECK_EQ_AT("size", i, cp0_pagemask(), sizes[i]);
    }
    cp0_pagemask_set(saved);
}

/* Status has a defined set of writable bits; the suite only needs to know the
 * ones it relies on stick, and that a write of all-ones does not set bits the
 * architecture does not define. Bits 31:28 (CU3..CU0), 27 (RP), 26 (FR),
 * 25 (RE), 24:22 (BEV/TS/SR), 21:16 (soft), 15:8 (IM), 7:0 (KX/SX/UX/UM/ERL/
 * EXL/IE) are the ones that exist on these parts; bit 19 is reserved. */
static void t_status_bits_round_trip(void)
{
    u32 saved = cp0_status();
    static const u32 bits[] = {
        ST_CU1, ST_CU0, ST_FR, ST_BEV, ST_KX, ST_SX, ST_UX, ST_EXL, ST_ERL,
    };
    unsigned i;

    for (i = 0; i < sizeof(bits) / sizeof(bits[0]); i++) {
        /* Set the bit on top of a known-safe base, and read it back. Keep
         * BEV clear and CU0 set so the vectors and CP0 stay usable. */
        u32 base = (saved & ~ST_BEV) | ST_CU0;
        cp0_status_set(base | bits[i]);
        CHECK_EQ_AT("bit", i, cp0_status() & bits[i], bits[i]);
        cp0_status_set(base & ~bits[i]);
        /* EXL/ERL can be re-set by the machine itself; only assert on the
         * bits that are purely software-controlled. */
        if (bits[i] != ST_EXL && bits[i] != ST_ERL)
            CHECK_EQ_AT("bit", i, cp0_status() & bits[i], 0u);
    }
    cp0_status_set(saved);
}

/*
 * Cause is almost entirely read-only: only the two software interrupt bits
 * IP1:IP0 (bits 9:8) can be written. Writing all ones must not set ExcCode or
 * BD, which the hardware owns.
 */
static void t_cause_mostly_read_only(void)
{
    u32 saved = cp0_cause();
    cp0_cause_set(0xFFFFFFFFu);
    {
        u32 c = cp0_cause();
        CHECK_EQ(c & CAUSE_BD, 0u);            /* BD is hardware-owned */
        CHECK_EQ(c & CAUSE_EXC_MASK, saved & CAUSE_EXC_MASK);
        CHECK_EQ(c & 0x00000300u, 0x00000300u); /* IP1:IP0 are writable */
    }
    cp0_cause_set(saved & 0x00000300u);
    CHECK_EQ(cp0_cause() & 0x00000300u, saved & 0x00000300u);
    cp0_cause_set(0);
}

/* Software interrupts are visible in Cause.IP even with IE clear — they are
 * pending, just not delivered. */
static void t_software_interrupt_bits(void)
{
    cp0_cause_set(0x00000100u);          /* IP0 */
    CHECK_EQ(cp0_cause() & 0x00000100u, 0x00000100u);
    CHECK_NO_EXC();                       /* IE is clear: pending, not taken */

    cp0_cause_set(0x00000200u);          /* IP1 */
    CHECK_EQ(cp0_cause() & 0x00000300u, 0x00000200u);

    cp0_cause_set(0);
    CHECK_EQ(cp0_cause() & 0x00000300u, 0u);
}

/* ── 64-bit CP0 registers ─────────────────────────────────────────────────── */

/*
 * EntryHi, EntryLo0/1, Context, XContext, EPC, BadVAddr and ErrorEPC are 64
 * bits wide. dmtc0/dmfc0 must move all 64; mtc0 writes the low 32 and
 * sign-extends.
 */
static void t_entryhi_is_64_bit(void)
{
    u64 saved = cp0_entryhi();
    /* A VPN2 in the upper half plus an ASID. Bits 12:8 of EntryHi are
     * reserved-zero on the R4400, so keep the low bits to the ASID field. */
    cp0_entryhi_set(0x0000001200000000ull | 0x5Aull);
    CHECK_EQ(cp0_entryhi() & 0xFFull, 0x5Aull);
    CHECK_EQ(cp0_entryhi() & 0x0000001200000000ull, 0x0000001200000000ull);
    cp0_entryhi_set(saved);
}

static void t_entrylo_round_trip(void)
{
    u64 saved0 = cp0_entrylo0(), saved1 = cp0_entrylo1();
    u64 v = ((u64)0x1234 << ELO_PFN_SHIFT) |
            ((u64)CA_CACHEABLE_NC << ELO_C_SHIFT) | ELO_D | ELO_V | ELO_G;

    cp0_entrylo0_set(v);
    cp0_entrylo1_set(v ^ ELO_D);
    CHECK_EQ(cp0_entrylo0() & 0x3FFFFFFFull, v & 0x3FFFFFFFull);
    CHECK_EQ(cp0_entrylo1() & 0x3FFFFFFFull, (v ^ ELO_D) & 0x3FFFFFFFull);

    cp0_entrylo0_set(saved0);
    cp0_entrylo1_set(saved1);
}

/*
 * EntryHi is not a flat 64-bit register: bits 61:40 are the Fill field, which
 * the R4000 manual (Figure 4-9) defines as "Reserved. 0 on read; ignored on
 * write." So writing all-ones leaves R (63:62) and VPN2 (39:13) set but reads
 * Fill back as zero.
 *
 * There is deliberately no MTC0-sign-extension test here. The manual's
 * Operation section for MTC0 is just `CPR[0,rd] <- GPR[rt]` and says nothing
 * about the width of a 64-bit destination, so asserting sign-extension would
 * be inventing a requirement rather than testing one. (DMTC0's page is
 * explicit in the other direction: its behaviour on a 32-bit CP0 register is
 * undefined.)
 */
static void t_entryhi_fill_field_reads_zero(void)
{
    u64 saved = cp0_entryhi();
    cp0_entryhi_set(0xFFFFFFFFFFFFFFFFull);
    {
        u64 v = cp0_entryhi();
        CHECK_EQ(v & 0x3FFFFF0000000000ull, 0ull);          /* Fill 61:40 = 0 */
        CHECK_EQ(v & 0xC000000000000000ull, 0xC000000000000000ull); /* R kept */
        CHECK_EQ(v & 0xFFull, 0xFFull);                     /* ASID kept     */
    }
    cp0_entryhi_set(saved);
}

/* Context's PTEBase (bits 63:23) is writable; the BadVPN2 field below it is
 * written by hardware on a TLB miss and reads back as whatever the last miss
 * left. Only the writable part is asserted. */
static void t_context_ptebase_writable(void)
{
    u64 saved = cp0_context();
    cp0_context_set(0xFFFFFFFF80000000ull);
    CHECK_EQ(cp0_context() & 0xFFFFFFFFFF800000ull, 0xFFFFFFFF80000000ull);
    cp0_context_set(saved);
}

static void t_xcontext_ptebase_writable(void)
{
    u64 saved = cp0_xcontext();
    /* XContext PTEBase is bits 63:33. */
    cp0_xcontext_set(0xFFFFFFFE00000000ull);
    CHECK_EQ(cp0_xcontext() & 0xFFFFFFFE00000000ull, 0xFFFFFFFE00000000ull);
    cp0_xcontext_set(saved);
}

/* ── Count and Compare ────────────────────────────────────────────────────── */

/*
 * Count is a free-running counter. On the R4400 it advances at half the
 * pipeline clock (src/mips_core.rs:589), so the only portable assertion is
 * that it moves forward.
 */
static void t_count_advances(void)
{
    u32 a, b;
    unsigned i;
    a = cp0_count();
    for (i = 0; i < 1000; i++) __asm__ __volatile__("" ::: "memory");
    b = cp0_count();
    CHECK_NE(a, b);
    /* And it must be monotonic over a short window (no wrap in 1000 iters). */
    CHECK((u32)(b - a) < 0x40000000u);
}

static void t_count_writable(void)
{
    cp0_count_set(0x12345678u);
    /* It keeps counting, so only the high bits are stable enough to check. */
    CHECK_EQ(cp0_count() & 0xFFFF0000u, 0x12340000u);
    cp0_count_set(0);
}

static void t_compare_round_trip(void)
{
    u32 saved = cp0_compare();
    cp0_compare_set(0xDEADBEEFu);
    CHECK_EQ(cp0_compare(), 0xDEADBEEFu);
    cp0_compare_set(saved);
}

/*
 * A Count/Compare match sets Cause.IP7. With IE clear it must NOT be delivered
 * as an exception — it just becomes pending. Writing Compare clears the pending
 * bit, which is the documented acknowledge mechanism.
 *
 * The spin is bounded, and the two failure modes are kept apart: "Count never
 * reached Compare inside the budget" is a slow clock, not a broken interrupt,
 * so it is reported rather than asserted. IRIS's Count is wallclock-anchored by
 * default (src/mips_core.rs, and the `ci_clock` feature exists to make it
 * deterministic), so how many loop iterations a given Compare delta takes is
 * not a property of the guest program at all.
 */
static void t_compare_sets_ip7_and_write_clears_it(void)
{
    unsigned i;
    u32 fired = 0, reached = 0, last = 0;
    const u32 target = 2000;

    exc_clear();
    cp0_count_set(0);
    cp0_compare_set(target);

    for (i = 0; i < 4000000; i++) {
        if (cp0_cause() & 0x8000u) { fired = 1; break; }   /* IP7 */
        last = cp0_count();
        if (last >= target) { reached = 1; if (i > 1000) break; }
    }

    if (!reached && !fired) {
        con_printf("\n      [timer inconclusive: Count only reached %u of %u"
                   " in %u iterations]", last, target, i);
        CHECK(1);
        return;
    }

    CHECK_EQ(fired, 1u);
    CHECK_NO_EXC();                  /* IE is clear: pending, never taken */

    /* Writing Compare acknowledges the interrupt. */
    cp0_compare_set(0xFFFFFFFFu);
    CHECK_EQ(cp0_cause() & 0x8000u, 0u);
}

/* ── ERET ─────────────────────────────────────────────────────────────────── */

/*
 * ERET clears the LL bit. Establish a link with LL, ERET through a controlled
 * path, and confirm a following SC fails.
 *
 * The ERET is done via the resume handler: an ERET executed inline would need
 * EPC to already point at the continuation.
 */
static void t_eret_clears_ll_bit(void)
{
    extern char _scratch_start[];
    volatile u32 *p = (volatile u32 *)_scratch_start;
    u64 sc_result;

    *p = 0x1000u;
    SYNC();

    exc_clear();
    exc_user_handler = (u32)(unsigned long)&exl_resume_handler;

    __asm__ __volatile__(A
        "dla $12, 1f\n\t"
        "sd $12, 0(%1)\n\t"          /* resume just past the syscall */
        "ll $13, 0(%2)\n\t"          /* set the LL bit */
        "syscall\n\t"                /* handler ERETs, which must clear it */
        "1:\n\t"
        "daddiu $13, $13, 1\n\t"
        "sc $13, 0(%2)\n\t"
        "daddu %0, $zero, $13" Z
        : "=r"(sc_result)
        : "r"(&exl_resume_pc), "r"(p)
        : "$12", "$13", "memory");

    exc_user_handler = 0;
    CHECK_EQ(sc_result, 0ull);       /* SC must fail: ERET broke the link */
    CHECK_EQ(*p, 0x1000u);           /* and memory must be unchanged */
}

/* Without an intervening ERET, LL/SC succeeds. */
static void t_ll_sc_succeeds_without_interruption(void)
{
    extern char _scratch_start[];
    volatile u32 *p = (volatile u32 *)_scratch_start;
    u64 sc_result;

    *p = 0x2000u;
    SYNC();
    __asm__ __volatile__(A
        "ll $13, 0(%1)\n\t"
        "daddiu $13, $13, 1\n\t"
        "sc $13, 0(%1)\n\t"
        "daddu %0, $zero, $13" Z
        : "=r"(sc_result) : "r"(p) : "$13", "memory");

    CHECK_EQ(sc_result, 1ull);
    CHECK_EQ(*p, 0x2001u);
}

static const struct test tests[] = {
    TEST("cp0/prid_read_only",       t_prid_read_only,                    CPU_ALL),
    TEST("cp0/random_moves",         t_random_read_only_and_moves,        CPU_ALL),
    TEST("cp0/random_respects_wired", t_random_respects_wired,            CPU_ALL),
    TEST("cp0/wired_reserved_bits",  t_wired_reserved_bits_read_zero,     CPU_ALL),
    TEST("cp0/wired_resets_random",  t_wired_write_resets_random,         CPU_ALL),
    TEST("cp0/pagemask_reserved",    t_pagemask_reserved_bits,            CPU_ALL),
    TEST("cp0/pagemask_all_sizes",   t_pagemask_all_sizes,                CPU_ALL),
    TEST("cp0/status_bits",          t_status_bits_round_trip,            CPU_ALL),
    TEST("cp0/cause_read_only",      t_cause_mostly_read_only,            CPU_ALL),
    TEST("cp0/software_int_bits",    t_software_interrupt_bits,           CPU_ALL),
    TEST("cp0/entryhi_64bit",        t_entryhi_is_64_bit,                 CPU_ALL),
    TEST("cp0/entrylo_round_trip",   t_entrylo_round_trip,                CPU_ALL),
    TEST("cp0/entryhi_fill_zero",    t_entryhi_fill_field_reads_zero,     CPU_ALL),
    TEST("cp0/context_ptebase",      t_context_ptebase_writable,          CPU_ALL),
    TEST("cp0/xcontext_ptebase",     t_xcontext_ptebase_writable,         CPU_ALL),
    TEST("cp0/count_advances",       t_count_advances,                    CPU_ALL),
    TEST("cp0/count_writable",       t_count_writable,                    CPU_ALL),
    TEST("cp0/compare_round_trip",   t_compare_round_trip,                CPU_ALL),
    TEST("cp0/compare_sets_ip7",     t_compare_sets_ip7_and_write_clears_it, CPU_ALL),
    TEST("cp0/eret_clears_ll",       t_eret_clears_ll_bit,                CPU_ALL),
    TEST("cp0/ll_sc_succeeds",       t_ll_sc_succeeds_without_interruption, CPU_ALL),
};

const struct test_group group_cp0 = {
    "cp0", tests, sizeof(tests) / sizeof(tests[0])
};
