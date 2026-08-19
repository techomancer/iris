/* cp0.h — inline accessors for CP0, CP1, and the CACHE instruction.
 *
 * Every one is a statement expression with explicit .set directives, so it can
 * be used from any C context without the surrounding code needing to know we
 * are reaching into MIPS III territory.
 */
#ifndef CP0_H
#define CP0_H

#include "testlib.h"

/* 32-bit CP0 read/write (mfc0/mtc0). `sel` is not used on R4400/R5000 — the
 * select field arrived with MIPS32r1 — so these take the register number
 * alone. */
#define CP0_R32(reg) ({                                                    \
    u32 __v;                                                               \
    __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"                       \
                         "mfc0 %0, $" #reg "\n\t"                          \
                         "nop; nop\n\t"                                    \
                         ".set pop" : "=r"(__v));                          \
    __v; })

#define CP0_W32(reg, val) do {                                             \
    u32 __v = (u32)(val);                                                  \
    __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"                       \
                         "mtc0 %0, $" #reg "\n\t"                          \
                         "nop; nop; nop\n\t"                               \
                         ".set pop" :: "r"(__v));                          \
} while (0)

/* 64-bit CP0 read/write (dmfc0/dmtc0). Needs Status.KX for the wide
 * registers to be meaningful, which start.S sets. */
#define CP0_R64(reg) ({                                                    \
    u64 __v;                                                               \
    __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat; .set noat\n\t"            \
                         "dmfc0 %0, $" #reg "\n\t"                         \
                         "nop; nop\n\t"                                    \
                         ".set pop" : "=r"(__v));                          \
    __v; })

#define CP0_W64(reg, val) do {                                             \
    u64 __v = (u64)(val);                                                  \
    __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat; .set noat\n\t"            \
                         "dmtc0 %0, $" #reg "\n\t"                         \
                         "nop; nop; nop\n\t"                               \
                         ".set pop" :: "r"(__v));                          \
} while (0)

/* Named CP0 registers. */
#define cp0_index()          CP0_R32(0)
#define cp0_index_set(v)     CP0_W32(0, v)
#define cp0_random()         CP0_R32(1)
#define cp0_entrylo0()       CP0_R64(2)
#define cp0_entrylo0_set(v)  CP0_W64(2, v)
#define cp0_entrylo1()       CP0_R64(3)
#define cp0_entrylo1_set(v)  CP0_W64(3, v)
#define cp0_context()        CP0_R64(4)
#define cp0_context_set(v)   CP0_W64(4, v)
#define cp0_pagemask()       CP0_R32(5)
#define cp0_pagemask_set(v)  CP0_W32(5, v)
#define cp0_wired()          CP0_R32(6)
#define cp0_wired_set(v)     CP0_W32(6, v)
#define cp0_badvaddr()       CP0_R64(8)
#define cp0_count()          CP0_R32(9)
#define cp0_count_set(v)     CP0_W32(9, v)
#define cp0_entryhi()        CP0_R64(10)
#define cp0_entryhi_set(v)   CP0_W64(10, v)
#define cp0_compare()        CP0_R32(11)
#define cp0_compare_set(v)   CP0_W32(11, v)
#define cp0_status()         CP0_R32(12)
#define cp0_status_set(v)    CP0_W32(12, v)
#define cp0_cause()          CP0_R32(13)
#define cp0_cause_set(v)     CP0_W32(13, v)
#define cp0_epc()            CP0_R64(14)
#define cp0_epc_set(v)       CP0_W64(14, v)
#define cp0_prid()           CP0_R32(15)
#define cp0_config()         CP0_R32(16)
#define cp0_config_set(v)    CP0_W32(16, v)
#define cp0_lladdr()         CP0_R32(17)
#define cp0_watchlo()        CP0_R32(18)
#define cp0_watchlo_set(v)   CP0_W32(18, v)
#define cp0_watchhi()        CP0_R32(19)
#define cp0_watchhi_set(v)   CP0_W32(19, v)
#define cp0_xcontext()       CP0_R64(20)
#define cp0_xcontext_set(v)  CP0_W64(20, v)
#define cp0_ecc()            CP0_R32(26)
#define cp0_cacheerr()       CP0_R32(27)
#define cp0_taglo()          CP0_R32(28)
#define cp0_taglo_set(v)     CP0_W32(28, v)
#define cp0_taghi()          CP0_R32(29)
#define cp0_taghi_set(v)     CP0_W32(29, v)
#define cp0_errorepc()       CP0_R64(30)

/* TLB instructions. */
#define tlb_probe()   __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"     \
                        "tlbp\n\tnop; nop; nop; nop\n\t.set pop" ::: "memory")
#define tlb_read()    __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"     \
                        "tlbr\n\tnop; nop; nop; nop\n\t.set pop" ::: "memory")
#define tlb_write_indexed() __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t" \
                        "tlbwi\n\tnop; nop; nop; nop\n\t.set pop" ::: "memory")
#define tlb_write_random()  __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t" \
                        "tlbwr\n\tnop; nop; nop; nop\n\t.set pop" ::: "memory")

/* CP1 control registers. */
#define fcr_read(n) ({                                                     \
    u32 __v;                                                               \
    __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"                       \
                         "cfc1 %0, $" #n "\n\tnop\n\t.set pop"             \
                         : "=r"(__v));                                     \
    __v; })

#define fcr_write(n, val) do {                                             \
    u32 __v = (u32)(val);                                                  \
    __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"                       \
                         "ctc1 %0, $" #n "\n\tnop\n\t.set pop"             \
                         :: "r"(__v));                                     \
} while (0)

#define fcsr()          fcr_read(31)
#define fcsr_set(v)     fcr_write(31, v)
#define fir()           fcr_read(0)

/* CACHE instruction with a compile-time op and a runtime address. */
#define CACHE_OP(op, addr) do {                                            \
    void *__a = (void *)(unsigned long)(addr);                             \
    __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"                       \
                         "cache %0, 0(%1)\n\t"                             \
                         ".set pop" :: "i"(op), "r"(__a) : "memory");      \
} while (0)

/* Ordering barrier for the caches / write buffer. */
#define SYNC()  __asm__ __volatile__(".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\tsync\n\t.set pop" \
                                     ::: "memory")

#endif /* CP0_H */
