/* r10k — what the R10000 family does that the R4400 and R5000 do not.
 *
 * Every test here runs only on an R10000, R12000 or R14000 and asserts
 * something one of their manuals states, cited by page:
 *
 *   R10000 UM  MIPS R10000 Microprocessor User's Manual, Version 2.0
 *              (SGI 007-2490-001), printed page numbers.
 *   NEC UM     NEC VR10000 Series User's Manual, U10278EJ4V0UM (fourth
 *              edition), which covers the R12000 and lists what changed from
 *              the R10000 in its Appendix B.
 *
 * Where a manual leaves a case open the test reports what happened rather
 * than asserting it, as the rest of the suite does for undefined results.
 * No R10000 has run this file; docs/r10000.md says where every R10000
 * expectation in the suite comes from.
 */

#include "../fpu/fpu_common.h"
#include "excoff.h"

/* MIPS IV with hard float, for the conditional moves and RECIP/MADD. */
#define A4F ".set push; .set mips4; .set noreorder; .set nomacro; .set noat\n\t" \
            ".set hardfloat\n\t"

/* A value that no result in this file can produce, in both halves. */
#define JUNK 0xDEADBEEFCAFEF00Dull

/* ── single-precision results in FR=1 mode ────────────────────────────────── */

/*
 * "Single-precision results are written into the low half of the register.
 * The high half of the result register is architecturally undefined; in the
 * R10000 implementation, it is set to zero" (R10000 UM p. 306; NEC UM p. 279
 * says the same of the R12000). p. 313 widens it: the R10000 "clears the
 * upper 32 bits, including MOV.S", for S and W results, and for MTC1 and LWC1
 * in FR=1 mode — where the R4400 keeps them.
 */
static void t_single_result_clears_upper(void)
{
    u64 add, mov, lw, mt, cvts, cvtw;

    w()[0] = F_1; w()[1] = F_2;
    d()[1] = D_2;
    SYNC();
    fcsr_reset();
    exc_clear();
    __asm__ __volatile__(AF "lwc1  $f0, 0(%6)\n\t"
                            "lwc1  $f2, 4(%6)\n\t"
                            "ldc1  $f12, 8(%6)\n\t"
                            "dmtc1 %7, $f4\n\t"
                            "dmtc1 %7, $f6\n\t"
                            "dmtc1 %7, $f8\n\t"
                            "dmtc1 %7, $f10\n\t"
                            "dmtc1 %7, $f14\n\t"
                            "dmtc1 %7, $f16\n\t"
                            "nop\n\t"
                            "add.s   $f4, $f0, $f2\n\t"
                            "mov.s   $f6, $f0\n\t"
                            "lwc1    $f8, 0(%6)\n\t"
                            "mtc1    %8, $f10\n\t"
                            "cvt.s.d $f14, $f12\n\t"
                            "cvt.w.s $f16, $f2\n\t"
                            "nop\n\t"
                            "dmfc1 %0, $f4\n\t"
                            "dmfc1 %1, $f6\n\t"
                            "dmfc1 %2, $f8\n\t"
                            "dmfc1 %3, $f10\n\t"
                            "dmfc1 %4, $f14\n\t"
                            "dmfc1 %5, $f16" Z
                         : "=r"(add), "=r"(mov), "=r"(lw), "=r"(mt), "=r"(cvts),
                           "=r"(cvtw)
                         : "r"(w()), "r"(JUNK), "r"(0x80000001u) : "memory");
    CHECK_NO_EXC();
    CHECK_EQ(add,  (u64)F_3);
    CHECK_EQ(mov,  (u64)F_1);
    CHECK_EQ(lw,   (u64)F_1);
    CHECK_EQ(mt,   0x0000000080000001ull);
    CHECK_EQ(cvts, (u64)F_2);
    CHECK_EQ(cvtw, 2ull);
}

/*
 * "For the floating-point conditional move instructions, MOVT.S, MOVF.S,
 * MOVZ.S, and MOVN.S, the R10000 processor always clears the upper 32 bits of
 * the destination register even though the condition is false" (R10000 UM
 * p. 313). The low word is the old one when nothing moves. The NEC UM (p. 279)
 * calls the upper half "undefined ..., even if no move occurs"; holding the
 * R12000 and R14000 to the R10000's rule is a model choice.
 */
static void t_cond_move_s_clears_upper(void)
{
    u64 n_no, z_no, t_no, f_yes, n_yes;

    w()[0] = F_1;
    SYNC();
    fcsr_reset();                              /* $fcc0 false */
    exc_clear();
    __asm__ __volatile__(A4F "lwc1  $f0, 0(%5)\n\t"
                             "dmtc1 %6, $f4\n\t"
                             "dmtc1 %6, $f6\n\t"
                             "dmtc1 %6, $f8\n\t"
                             "dmtc1 %6, $f10\n\t"
                             "dmtc1 %6, $f12\n\t"
                             "nop\n\t"
                             "movn.s $f4, $f0, $zero\n\t"   /* rt == 0: no move */
                             "movz.s $f6, $f0, %7\n\t"      /* rt != 0: no move */
                             "movt.s $f8, $f0, $fcc0\n\t"   /* cc false: no move */
                             "movf.s $f10, $f0, $fcc0\n\t"  /* cc false: moves */
                             "movn.s $f12, $f0, %7\n\t"     /* rt != 0: moves */
                             "nop\n\t"
                             "dmfc1 %0, $f4\n\t"
                             "dmfc1 %1, $f6\n\t"
                             "dmfc1 %2, $f8\n\t"
                             "dmfc1 %3, $f10\n\t"
                             "dmfc1 %4, $f12" Z
                          : "=r"(n_no), "=r"(z_no), "=r"(t_no), "=r"(f_yes), "=r"(n_yes)
                          : "r"(w()), "r"(JUNK), "r"(1u) : "memory");
    CHECK_NO_EXC();
    CHECK_EQ(n_no,  JUNK & 0xFFFFFFFFull);
    CHECK_EQ(z_no,  JUNK & 0xFFFFFFFFull);
    CHECK_EQ(t_no,  JUNK & 0xFFFFFFFFull);
    CHECK_EQ(f_yes, (u64)F_1);
    CHECK_EQ(n_yes, (u64)F_1);
}

/*
 * With FR=0 a single-precision result lands in the low half of an even/odd
 * pair, and whether the R10000 then zeroes the odd register is not said: the
 * manuals call the high half of a single result undefined (R10000 UM p. 307,
 * Figure 15-5) without saying which mode p. 313's zeroing covers. Reported.
 */
static void t_fr0_single_result_odd_half(void)
{
    u32 saved = cp0_status();
    u32 odd = 0;

    w()[0] = F_1; w()[1] = F_2; w()[2] = 0x13579BDFu; w()[3] = 0;
    SYNC();
    fcsr_reset();
    exc_clear();
    cp0_status_set(saved & ~ST_FR);
    __asm__ __volatile__(AF "lwc1  $f0, 0(%0)\n\t"
                            "lwc1  $f2, 4(%0)\n\t"
                            "lwc1  $f5, 8(%0)\n\t"
                            "add.s $f4, $f0, $f2\n\t"
                            "nop\n\t"
                            "swc1  $f5, 12(%0)" Z :: "r"(w()) : "memory");
    cp0_status_set(saved);
    SYNC();
    odd = w()[3];
    con_printf("\n      [FR=0 add.s $f4: $f5 %x -> %x]", 0x13579BDFu, odd);
    CHECK_NO_EXC();
}

/* ── FCSR.FS flushes every denormal result ────────────────────────────────── */

/* scratch[0]=fs  scratch[1]=ft  scratch[2]=result  scratch[3]=sentinel  scratch[4]=fr */
static void op_recip_s(void)
{
    __asm__ __volatile__(A4F "lwc1 $f0, 0(%0)\n\t"
                             "lwc1 $f4, 12(%0)\n\t"
                             "recip.s $f4, $f0\n\t"
                             "swc1 $f4, 8(%0)" Z :: "r"(w()) : "memory");
}

static void op_madd_s(void)
{
    __asm__ __volatile__(A4F "lwc1 $f0, 0(%0)\n\t"
                             "lwc1 $f2, 4(%0)\n\t"
                             "lwc1 $f6, 16(%0)\n\t"
                             "lwc1 $f4, 12(%0)\n\t"
                             "madd.s $f4, $f6, $f0, $f2\n\t"
                             "swc1 $f4, 8(%0)" Z :: "r"(w()) : "memory");
}

/*
 * FS: "0: A denormalized result causes an Unimplemented Operation exception.
 * 1: A denormalized result is replaced with zero. No exception is flagged"
 * (R10000 UM p. 309, Figure 15-7). The rule names no instruction, so it holds
 * for MIPS IV's RECIP and multiply-add as for ADD: 1/2^127 is 2^-127, and
 * 2^-126 * 1.5 - 2^-126 is 2^-127, both denormal from normal operands. What
 * "no exception is flagged" means for the U and I bits is not settled, so
 * only the result and the absence of a trap are asserted with FS set.
 */
static void t_fs_flushes_recip_and_madd(void)
{
    struct fp_obs o;

    o = observe_s(0x7F000000u, 0, FCSR_FS, op_recip_s);        /* 1 / 2^127 */
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.result, F_0);
    o = observe_s(0x7F000000u, 0, 0, op_recip_s);
    CHECK_EQ(o.exceptions, 1u);
    CHECK_EQ(o.excode, (u32)EXC_FPE);
    CHECK_EQ(o.cause & FP_E, (u32)FP_E);
    CHECK_EQ(o.result, SENTINEL_S);

    w()[4] = 0x80800000u;                                        /* fr = -2^-126 */
    o = observe_s(F_MIN_NORM, 0x3FC00000u, FCSR_FS, op_madd_s);  /* 2^-126 * 1.5 */
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.result, F_0);
    w()[4] = 0x80800000u;
    o = observe_s(F_MIN_NORM, 0x3FC00000u, 0, op_madd_s);
    CHECK_EQ(o.exceptions, 1u);
    CHECK_EQ(o.excode, (u32)EXC_FPE);
    CHECK_EQ(o.cause & FP_E, (u32)FP_E);
    CHECK_EQ(o.result, SENTINEL_S);
}

/* ── CVT.L stops at 51 bits ───────────────────────────────────────────────── */

FPOP_D(op_cvt_l_d, "cvt.l.d $f4, $f0")

/* The single source is the high word of scratch doubleword 0. */
static void op_cvt_l_s(void)
{
    __asm__ __volatile__(AF "lwc1 $f0, 0(%0)\n\t"
                            "ldc1 $f4, 24(%0)\n\t"
                            "cvt.l.s $f4, $f0\n\t"
                            "sdc1 $f4, 16(%0)" Z :: "r"(d()) : "memory");
}

/*
 * The R4400 converts to a long integer of up to 53 bits; "In the R10000, the
 * conversion allows only up to 51 bits; otherwise an Unimplemented Operation
 * exception is taken" (R10000 UM p. 312). 2^52 is 53 bits wide, which the
 * R4400 converts and the R10000 must not; 2^49 is well inside both. The exact
 * edge, 2^51, is left alone: "up to 51 bits" does not say on which side of it
 * the limit falls.
 *
 * The sentence is in the R10000's manual alone; the NEC UM has no CVT.L
 * section, so on an R12000 or R14000 the 2^52 case is reported, not asserted.
 */
static void t_cvt_l_limit(void)
{
    struct fp_obs o;

    if (is_r12000_up()) {
        o = observe_d(0x4330000000000000ull, 0, 0, op_cvt_l_d);
        con_printf("\n      [cvt.l.d of 2^52: exc=%u cause=%x result=%lx]",
                   o.exceptions, o.cause, o.result_d);
        o = observe_d(0x4300000000000000ull, 0, 0, op_cvt_l_d);
        CHECK_EQ(o.exceptions, 0u);
        CHECK_EQ(o.result_d, 1ull << 49);
        return;
    }

    o = observe_d(0x4330000000000000ull, 0, 0, op_cvt_l_d);      /* 2^52 */
    CHECK_EQ(o.exceptions, 1u);
    CHECK_EQ(o.excode, (u32)EXC_FPE);
    CHECK_EQ(o.cause & FP_E, (u32)FP_E);
    CHECK_EQ(o.result_d, SENTINEL_D);
    o = observe_d(0xC330000000000000ull, 0, 0, op_cvt_l_d);      /* -2^52 */
    CHECK_EQ(o.exceptions, 1u);
    CHECK_EQ(o.result_d, SENTINEL_D);
    o = observe_d(0x4300000000000000ull, 0, 0, op_cvt_l_d);      /* 2^49 */
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.result_d, 1ull << 49);

    o = observe_d((u64)0x59800000u << 32, 0, 0, op_cvt_l_s);     /* 2^52 */
    CHECK_EQ(o.exceptions, 1u);
    CHECK_EQ(o.cause & FP_E, (u32)FP_E);
    CHECK_EQ(o.result_d, SENTINEL_D);
    o = observe_d((u64)0x58000000u << 32, 0, 0, op_cvt_l_s);     /* 2^49 */
    CHECK_EQ(o.exceptions, 0u);
    CHECK_EQ(o.result_d, 1ull << 49);
}

/* ── undefined encodings ──────────────────────────────────────────────────── */

/*
 * "In the R10000 processor, undefined opcodes (bits 25:24 are 0 or 1) take
 * Reserved Instruction exceptions and undefined formats (bits 25:24 are 2 or
 * 3) take Unimplemented Operation exceptions" (R10000 UM p. 357), where the
 * R4400 takes Unimplemented Operation for both. COP1 rs = 7 is an undefined
 * opcode; fmt = 18 with ADD's function code is an undefined format.
 */
static void t_cop1_undefined_encodings(void)
{
    fcsr_reset();
    exc_clear();
    __asm__ __volatile__(AF ".word 0x44E00000" Z);     /* COP1 rs=7 */
    CHECK_EXC(EXC_RI);

    fcsr_reset();
    exc_clear();
    __asm__ __volatile__(AF ".word 0x46400000" Z);     /* COP1 fmt=18 ADD */
    CHECK_EXC(EXC_FPE);
    CHECK_EQ(FCSR_CAUSE_OF(exc.fcsr) & FP_E, (u32)FP_E);
    fcsr_reset();
}

/*
 * The R4400 lets BC0T/BC0F test the CH bit. "The R10000 processor also
 * implements the CH bit but it is not associated with a Coprocessor 0
 * condition. Instead, execution of a branch on Coprocessor 0 instruction
 * takes a Reserved Instruction exception" (R10000 UM p. 285). And "Execution
 * of an RFE instruction causes a Reserved Instruction exception in the R10000
 * processor" (p. 357).
 */
static void t_bc0_and_rfe_are_reserved(void)
{
    exc_clear();
    __asm__ __volatile__(A ".word 0x41000001\n\tnop\n\tnop" Z);  /* bc0f +1 */
    CHECK_EXC(EXC_RI);

    exc_clear();
    __asm__ __volatile__(A ".word 0x41010001\n\tnop\n\tnop" Z);  /* bc0t +1 */
    CHECK_EXC(EXC_RI);

    exc_clear();
    __asm__ __volatile__(A ".word 0x42000010" Z);                /* rfe */
    CHECK_EXC(EXC_RI);
}

/* ── TLB ──────────────────────────────────────────────────────────────────── */

extern char _scratch_start[];

static u64 scratch_phys(void)
{
    return (u64)((u32)(unsigned long)_scratch_start & 0x1FFFFFFFu);
}

static u64 entrylo(u64 phys, u64 flags)
{
    return ((phys >> 12) << ELO_PFN_SHIFT) |
           ((u64)CA_CACHEABLE_NC << ELO_C_SHIFT) | flags;
}

static void write_entry(u32 index, u64 vpn2, u64 lo0, u64 lo1, u32 pagemask)
{
    cp0_index_set(index);
    cp0_pagemask_set(pagemask);
    cp0_entryhi_set(vpn2);
    cp0_entrylo0_set(lo0);
    cp0_entrylo1_set(lo1);
    tlb_write_indexed();
}

/* Kuseg pages this file maps, and the places it parks its entries after:
 * neither is used by tlb/ or by each other. */
#define R10K_VA       0x00C00000ull
#define R10K_VA_16K   0x01000000ull
#define R10K_PARK(n)  (0x3FF00000ull + ((u64)(n) << 13))

/* Retire whatever maps `va`, so a load from it misses. */
static void unmap(u64 va)
{
    cp0_entryhi_set(va);
    tlb_probe();
    if ((cp0_index() & 0x80000000u) == 0)
        write_entry(cp0_index() & 0x3F, R10K_PARK(9), 0, 0, PM_4K);
}

/*
 * "The TLB refill vector selection is based on the address space of the
 * address ... that caused the TLB miss, and the value of the corresponding
 * extended addressing bit in the Status register (UX, SX, or KX). The current
 * operating mode of the processor is not important" (R10000 UM p. 333), where
 * the R4400 goes by the current mode. So a kernel-mode miss on a kuseg address
 * takes the 32-bit vector with UX clear even though KX is set, and the XTLB
 * vector with UX set even though KX is clear (Table 17-2, p. 334).
 */
static void t_refill_vector_follows_the_address(void)
{
    u32 saved = cp0_status();
    u64 saved_hi = cp0_entryhi();
    volatile u32 *p = (volatile u32 *)(unsigned long)R10K_VA;
    u32 v = 0, vec_ux0, vec_kx0;

    unmap(R10K_VA);

    cp0_status_set((saved | ST_KX) & ~ST_UX);
    exc_clear();
    __asm__ __volatile__(A "lw %0, 0(%1)" Z : "+r"(v) : "r"(p));
    cp0_status_set(saved);
    vec_ux0 = exc.vector;
    CHECK_EXC(EXC_TLBL);
    CHECK_EQ(exc.badvaddr, R10K_VA);

    cp0_status_set((saved | ST_UX) & ~ST_KX);
    exc_clear();
    __asm__ __volatile__(A "lw %0, 0(%1)" Z : "+r"(v) : "r"(p));
    cp0_status_set(saved);
    vec_kx0 = exc.vector;
    CHECK_EXC(EXC_TLBL);

    CHECK_EQ(vec_ux0, (u32)VECID_TLB);
    CHECK_EQ(vec_kx0, (u32)VECID_XTLB);
    cp0_entryhi_set(saved_hi);
}

/*
 * The R10000's virtual addresses are 44 bits, so a refill on one above 2^40
 * fills XContext's 31-bit BadVPN2 (bits 34:4, VA[43:13]) under a two-bit R
 * at 36:35 (R10000 UM p. 259, Figure 14-19), and EntryHi's VPN2 is VA[43:13]
 * (p. 245, Figure 14-10).
 */
static void t_xtlb_refill_44_bit_address(void)
{
    u64 saved_hi = cp0_entryhi();
    const u64 va = 0x0000012345678000ull;
    u64 v = 0;

    exc_clear();
    __asm__ __volatile__(A ".set macro\n\tdli $9, 0x0000012345678000\n\t.set nomacro\n\t"
                           "ld %0, 0($9)" Z : "+r"(v) :: "$9");
    CHECK_EXC(EXC_TLBL);
    CHECK_EQ(exc.vector, (u32)VECID_XTLB);
    CHECK_EQ(exc.badvaddr, va);
    CHECK_EQ((exc.xcontext >> 4) & 0x7FFFFFFFull, va >> 13);
    CHECK_EQ((exc.xcontext >> 35) & 3ull, 0ull);
    CHECK_EQ(exc.entryhi & 0xC0000FFFFFFFE000ull, va);
    cp0_entryhi_set(saved_hi);
}

/*
 * A TLB write whose entry would match another "sets the TS bit ... the TLB
 * entries that allow the multiple matches, even in the Wired area, are
 * invalidated before the new TLB entry is written", and software clears TS by
 * writing a zero to it (R10000 UM p. 249). The R4400 instead shuts the TLB
 * down. Two entries for one page, each to a different frame: after the second
 * write, translation must use it, and TLBP must find it.
 */
static void t_duplicate_entry_is_invalidated(void)
{
    u32 saved = cp0_status() & ~ST_TS;
    u64 saved_hi = cp0_entryhi();
    u64 phys = scratch_phys();
    volatile u32 *k0 = (volatile u32 *)_scratch_start;
    volatile u32 *va = (volatile u32 *)(unsigned long)R10K_VA;
    u32 ts_first, ts_second, ts_cleared, idx, got;

    /* 32 KB apart, so each frame's KSEG0 view and R10K_VA agree in VA[14:12]
     * and share a primary cache index. */
    k0[0x8000 / 4] = 0x11110000u;
    k0[0x10000 / 4] = 0x22220000u;
    SYNC();
    dcache_wb_invalidate_range(&k0[0x8000 / 4], 4);
    dcache_wb_invalidate_range(&k0[0x10000 / 4], 4);
    unmap(R10K_VA);

    cp0_status_set(saved);
    write_entry(20, R10K_VA, entrylo(phys + 0x8000, ELO_V | ELO_D | ELO_G), 0, PM_4K);
    ts_first = cp0_status() & ST_TS;
    write_entry(21, R10K_VA, entrylo(phys + 0x10000, ELO_V | ELO_D | ELO_G), 0, PM_4K);
    ts_second = cp0_status() & ST_TS;
    cp0_status_set(saved);
    ts_cleared = cp0_status() & ST_TS;

    cp0_entryhi_set(R10K_VA);
    tlb_probe();
    idx = cp0_index();
    exc_clear();
    got = *va;

    write_entry(20, R10K_PARK(0), 0, 0, PM_4K);
    write_entry(21, R10K_PARK(1), 0, 0, PM_4K);
    cp0_status_set(saved);
    cp0_entryhi_set(saved_hi);

    CHECK_EQ(ts_first, 0u);
    CHECK_EQ(ts_second, (u32)ST_TS);
    CHECK_EQ(ts_cleared, 0u);
    CHECK_EQ(idx, 21u);
    CHECK_NO_EXC();
    CHECK_EQ(got, 0x22220000u);
}

/*
 * "In the R10000 processor, the lower bits not used for address translation
 * are forced to zeroes during a TLB write ... a TLB read may not retrieve what
 * was originally written" (R10000 UM p. 299). A 16 KB page does not use the
 * low bit of the PFN.
 */
static void t_tlb_write_clears_unused_pfn_bits(void)
{
    u64 saved_hi = cp0_entryhi();
    u32 saved_pm = cp0_pagemask();
    u64 pfn = (scratch_phys() >> 12) | 1;
    u64 lo = (pfn << ELO_PFN_SHIFT) | ((u64)CA_CACHEABLE_NC << ELO_C_SHIFT) | ELO_V | ELO_G;
    u64 got;

    write_entry(22, R10K_VA_16K, lo, lo, PM_16K);
    cp0_entrylo0_set(0);
    cp0_index_set(22);
    tlb_read();
    got = cp0_entrylo0();
    write_entry(22, R10K_PARK(2), 0, 0, PM_4K);
    cp0_entryhi_set(saved_hi);
    cp0_pagemask_set(saved_pm);

    CHECK_EQ((got >> ELO_PFN_SHIFT) & 0x0FFFFFFFull, pfn & ~1ull);
    CHECK_EQ(got & 0x3Full, lo & 0x3Full);
}

/*
 * FrameMask (CP0 21) is new with the R10000: "Bits 15:0 of the FrameMask
 * register control bits 33:18 of the EntryLo registers", a one passing a zero
 * to the TLB on a write, and "The remaining bits of this register are ignored
 * on write and read as zeroes" (R10000 UM p. 260).
 */
static void t_framemask(void)
{
    u64 saved_hi = cp0_entryhi();
    u32 all;
    u64 lo = entrylo(scratch_phys() | 0x01000000ull, ELO_V | ELO_G);   /* EntryLo bit 18 */
    u64 got;

    unmap(R10K_VA);
    CP0_W32(21, 0xFFFFFFFFu);
    all = CP0_R32(21);
    CP0_W32(21, 0x0001u);
    write_entry(23, R10K_VA, lo, ELO_G, PM_4K);       /* G is the AND of both */
    CP0_W32(21, 0);
    cp0_index_set(23);
    tlb_read();
    got = cp0_entrylo0();
    write_entry(23, R10K_PARK(3), 0, 0, PM_4K);
    cp0_entryhi_set(saved_hi);

    CHECK_EQ(all, 0x0000FFFFu);
    CHECK_EQ(lo & (1ull << 18), 1ull << 18);
    CHECK_EQ(got & 0x3FFFFFFFull, lo & 0x3FFFFFFFull & ~(1ull << 18));
}

/* ── CACHE ────────────────────────────────────────────────────────────────── */

/*
 * The R10000's primary caches are two-way, and the Index operations take the
 * way from bit 0 of the address, which for that reason is never an alignment
 * error (R10000 UM p. 188, p. 189). Index Store Tag writes TagLo and TagHi
 * into the tag fields directly and Index Load Tag reads them back, D:
 * TagLo[31:8] tag, [7:6] state, [3] LRU, TagHi[3:0] tag, [31:29] StateMod; I:
 * TagLo[31:8] tag, [6] state (pp. 194-196). Two distinct invalid tags, one
 * per way, must each come back from their own way.
 *
 * The lines are written back and invalidated first, in the same asm with
 * nothing between, as cache/index_tag_rt does.
 */
static void t_index_tag_ways(void)
{
    u32 saved_lo = cp0_taglo(), saved_hi = cp0_taghi();
    const u32 tag0 = 0x12345600u, tag1 = 0x65432100u, hi = 0x20000005u;
    u32 d0, d1, dh0, dh1, i0, i1;
    char *line = (char *)(unsigned long)(KSEG0_BASE + 0x2A0);   /* a D set */
    char *iline = (char *)(unsigned long)(KSEG0_BASE + 0x3C0);  /* an I set */

    exc_clear();
    __asm__ __volatile__(A
        "cache  0x01, 0(%6)\n\t"           /* Index WB Inv D, way 0 */
        "cache  0x01, 1(%6)\n\t"           /* way 1 */
        "mtc0   %8, $28\n\t"
        "mtc0   %10, $29\n\t"
        "cache  0x09, 0(%6)\n\t"           /* Index Store Tag D, way 0 */
        "mtc0   %9, $28\n\t"
        "cache  0x09, 1(%6)\n\t"           /* way 1 */
        "mtc0   $0, $28\n\t"
        "mtc0   $0, $29\n\t"
        "cache  0x05, 0(%6)\n\t"           /* Index Load Tag D, way 0 */
        "mfc0   %0, $28\n\t"
        "mfc0   %2, $29\n\t"
        "cache  0x05, 1(%6)\n\t"           /* way 1 */
        "mfc0   %1, $28\n\t"
        "mfc0   %3, $29\n\t"
        "mtc0   %8, $28\n\t"
        "mtc0   $0, $29\n\t"
        "cache  0x08, 0(%7)\n\t"           /* Index Store Tag I, way 0 */
        "mtc0   %9, $28\n\t"
        "cache  0x08, 1(%7)\n\t"           /* way 1 */
        "mtc0   $0, $28\n\t"
        "cache  0x04, 0(%7)\n\t"           /* Index Load Tag I, way 0 */
        "mfc0   %4, $28\n\t"
        "cache  0x04, 1(%7)\n\t"           /* way 1 */
        "mfc0   %5, $28" Z
        : "=&r"(d0), "=&r"(d1), "=&r"(dh0), "=&r"(dh1), "=&r"(i0), "=&r"(i1)
        : "r"(line), "r"(iline), "r"(tag0), "r"(tag1), "r"(hi)
        : "memory");
    cp0_taglo_set(saved_lo);
    cp0_taghi_set(saved_hi);

    CHECK_NO_EXC();
    CHECK_EQ(d0 & 0xFFFFFFC8u, tag0);
    CHECK_EQ(d1 & 0xFFFFFFC8u, tag1);
    CHECK_EQ(dh0 & 0xE000000Fu, hi);
    CHECK_EQ(dh1 & 0xE000000Fu, hi);
    CHECK_EQ(i0 & 0xFFFFFF48u, tag0);
    CHECK_EQ(i1 & 0xFFFFFF48u, tag1);
}

/*
 * Cache Barrier (op 0x14) "does not change any cache fields" (R10000 UM
 * p. 198): no exception, the data where it was.
 */
static void t_cache_barrier(void)
{
    volatile u32 *p = (volatile u32 *)(_scratch_start + 0x3000);

    *p = 0xB0A4B0A4u;
    SYNC();
    exc_clear();
    CACHE_OP(0x14, p);
    CHECK_NO_EXC();
    CHECK_EQ(*p, 0xB0A4B0A4u);
}

/*
 * Status.CH: "CH is set if there is a hit in the secondary cache, and cleared
 * if there is a miss" by Hit Invalidate (S) and Hit WriteBack Invalidate (S),
 * "The CH bit can also be modified by a MTC0 instruction" (R10000 UM p. 190).
 */
static void t_ch_bit(void)
{
    u32 saved = cp0_status() & ~ST_CH;
    volatile u32 *p = (volatile u32 *)(_scratch_start + 0x3400);
    u32 hit, miss, set;

    *p = 0x0C4B17u;                     /* the line is now in both levels */
    SYNC();
    cp0_status_set(saved);
    CACHE_OP(CACHE_SD | CACHE_OP_HIT_WB_INV, p);
    hit = cp0_status() & ST_CH;
    CACHE_OP(CACHE_SD | CACHE_OP_HIT_WB_INV, p);
    miss = cp0_status() & ST_CH;
    cp0_status_set(saved | ST_CH);
    set = cp0_status() & ST_CH;
    cp0_status_set(saved);

    CHECK_EQ(hit, (u32)ST_CH);
    CHECK_EQ(miss, 0u);
    CHECK_EQ(set, (u32)ST_CH);
    CHECK_EQ(*p, 0x0C4B17u);
}

static const struct test tests[] = {
    TEST("r10k/single_clears_upper",   t_single_result_clears_upper,        CPU_R10000),
    TEST("r10k/cond_move_s_upper",     t_cond_move_s_clears_upper,          CPU_R10000),
    TEST("r10k/fr0_single_odd_half",   t_fr0_single_result_odd_half,        CPU_R10000),
    TEST("r10k/fs_flush_recip_madd",   t_fs_flushes_recip_and_madd,         CPU_R10000),
    TEST("r10k/cvt_l_limit",           t_cvt_l_limit,                       CPU_R10000),
    TEST("r10k/cop1_undefined",        t_cop1_undefined_encodings,          CPU_R10000),
    TEST("r10k/bc0_rfe_reserved",      t_bc0_and_rfe_are_reserved,          CPU_R10000),
    TEST("r10k/refill_vector_by_addr", t_refill_vector_follows_the_address, CPU_R10000),
    TEST("r10k/xtlb_refill_44bit",     t_xtlb_refill_44_bit_address,        CPU_R10000),
    TEST("r10k/tlb_duplicate",         t_duplicate_entry_is_invalidated,    CPU_R10000),
    TEST("r10k/tlb_unused_pfn",        t_tlb_write_clears_unused_pfn_bits,  CPU_R10000),
    TEST("r10k/framemask",             t_framemask,                         CPU_R10000),
    TEST("r10k/index_tag_ways",        t_index_tag_ways,                    CPU_R10000),
    TEST("r10k/cache_barrier",         t_cache_barrier,                     CPU_R10000),
    TEST("r10k/ch_bit",                t_ch_bit,                            CPU_R10000),
};

const struct test_group group_r10k = {
    "r10k", tests, sizeof(tests) / sizeof(tests[0])
};
