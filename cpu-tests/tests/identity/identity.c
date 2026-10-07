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
    u32 want = is_r5000() ? IMP_R5000 : is_r4600() ? IMP_R4600 : IMP_R4400;
    /* Only the implementation field names the part. The low byte is the silicon
     * revision and legitimately varies: IRIS models an R4400 rev 4.0, the Indy
     * this was validated on is rev 6.0 (PRId 0x460). Asserting the whole
     * register made a real CPU fail for being real. */
    CHECK_EQ(PRID_IMP(prid), want);
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
     * a zero byte, so masking it off compares the part, not the stepping.
     *
     * The R4600's FPU is on the same die and reports the CPU's implementation
     * number, 0x20: IRIX's hinv names an Indy R4600's FPU "MIPS R4600 Floating
     * Point Coprocessor" from exactly this field. Documented, not measured —
     * see docs/r4600.md. */
    u32 want = is_r5000() ? FIR_R5000 : is_r4600() ? FIR_R4600 : FIR_R4000;
    CHECK_EQ(fir() & ~0xFFu, want);
}

/* Config.IC/DC encode cache size as 2^(12+n) bytes; IB/DB are the line size,
 * 0 = 16 bytes, 1 = 32 bytes. R4400: 16 KB/16 B direct-mapped. R5000:
 * 32 KB/32 B two-way. (src/cpu/mips_cache_v2.rs:41-100, src/cpu/mips_exec.rs:89-90)
 * R4600: 16 KB/32 B two-way, from the IDT79R4600 data sheet — no R4600 has
 * run this suite yet. */
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
    } else if (is_r4600()) {
        CHECK_EQ(1u << (12 + ic), 16u * 1024);
        CHECK_EQ(1u << (12 + dc), 16u * 1024);
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
 * WHY THE SWEEP IS IN REGISTERS. The first version of this test was disabled
 * (2026-09-12) because it changed KSEG0's cacheability with live dirty data in
 * the D-cache: GCC spilled `orig` to the stack while KSEG0 was cached, so the
 * value sat dirty in L1D and never reached RAM, then hoisted the reload into a
 * delay slot that ran while the loop had left K0 on an uncached value. An
 * uncached load goes straight to the bus (R4000 UM p.326: it "issues a
 * noncoherent ... read request"), bypassing the dirty line, so `orig` came back
 * as pre-spill RAM, the restore wrote K0=0, and the CPU ran off into unmapped
 * KUSEG. Nothing about that is emulator-specific - changing a region's
 * coherency attribute with unflushed dirty lines in that region loses them on
 * hardware too, which is why the PROM always flips K0 from KSEG1 - and it
 * passed on the reference Indy only because the line happened to still be
 * resident at the reload.
 *
 * So from the first MTC0 to the restore there is no load, no store and no
 * call: `orig`, the loop counter and the two failure bitmasks all live in
 * registers, and nothing reaches memory until Config is back as it was. The
 * one thing KSEG0's attribute can still affect is instruction fetch, and the
 * text was written back to memory when the suite relocated itself.
 */
static void t_config_k0_writable(void)
{
    u64 bad_k0 = 0, bad_rest = 0, orig = 0, final = 0;

    __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"
        "mfc0   $8, $16\n\t"               /* $8  = orig                        */
        "nop; nop\n\t"
        "daddu  $9, $zero, $zero\n\t"      /* $9  = K0 value under test         */
        "daddu  $10, $zero, $zero\n\t"     /* $10 = bitmask: K0 did not stick   */
        "daddu  $11, $zero, $zero\n\t"     /* $11 = bitmask: other bits moved   */
        "addiu  $12, $zero, -8\n\t"        /* $12 = ~7                          */
        "and    $13, $8, $12\n\t"          /* $13 = orig & ~K0                  */
        "1:\n\t"
        "or     $14, $13, $9\n\t"
        "mtc0   $14, $16\n\t"
        "nop; nop; nop\n\t"
        "mfc0   $15, $16\n\t"
        "nop; nop\n\t"
        "addiu  $25, $zero, 1\n\t"
        "sllv   $25, $25, $9\n\t"          /* $25 = 1 << K0                     */
        "andi   $24, $15, 7\n\t"
        "beq    $24, $9, 2f\n\t"
        "nop\n\t"
        "or     $10, $10, $25\n\t"
        "2:\n\t"
        "and    $24, $15, $12\n\t"
        "beq    $24, $13, 3f\n\t"
        "nop\n\t"
        "or     $11, $11, $25\n\t"
        "3:\n\t"
        "addiu  $9, $9, 1\n\t"
        "sltiu  $24, $9, 8\n\t"
        "bnez   $24, 1b\n\t"
        "nop\n\t"
        "mtc0   $8, $16\n\t"               /* restore, before any memory access */
        "nop; nop; nop\n\t"
        "mfc0   $24, $16\n\t"
        "nop; nop\n\t"
        "daddu  %0, $10, $zero\n\t"
        "daddu  %1, $11, $zero\n\t"
        "daddu  %2, $8, $zero\n\t"
        "daddu  %3, $24, $zero\n\t"
        ".set pop"
        : "=r"(bad_k0), "=r"(bad_rest), "=r"(orig), "=r"(final)
        :: "$8", "$9", "$10", "$11", "$12", "$13", "$14", "$15", "$24", "$25");

    /* One bit per K0 value 0..7. */
    CHECK_EQ(bad_k0, 0u);
    CHECK_EQ(bad_rest, 0u);
    CHECK_EQ(final, orig);
}

/* All three parts have 48 TLB entries, so Random must wrap within 0..47 and
 * never fall below Wired. Just the range here; the decrement behaviour is tlb/. */
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
    TEST("identity/config_k0",        t_config_k0_writable,     CPU_ALL),
    TEST("identity/tlb_size",         t_tlb_size,               CPU_ALL),
};

const struct test_group group_identity = {
    "identity", tests, sizeof(tests) / sizeof(tests[0])
};
