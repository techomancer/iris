/* iris.h — machine definitions for the IRIS bare-metal CPU test suite.
 *
 * Physical addresses are quoted as KSEG1 (uncached, 0xA0000000 + phys) because
 * every device access in the suite must bypass the caches.
 */
#ifndef IRIS_H
#define IRIS_H

/* ── Z85C30 SCC serial, via IOC2 (src/ioc.rs: IOC_BASE 0x1FBD9800) ────────── */
#define IOC_BASE          0xBFBD9800u
#define SCC_CHB_CMD       (IOC_BASE + 0x30)   /* IOC_SERIAL1_CMD  — tty1 */
#define SCC_CHB_DATA      (IOC_BASE + 0x34)   /* IOC_SERIAL1_DATA */
#define SCC_CHA_CMD       (IOC_BASE + 0x38)   /* IOC_SERIAL2_CMD  — tty2 */
#define SCC_CHA_DATA      (IOC_BASE + 0x3C)   /* IOC_SERIAL2_DATA */
#define SCC_RR0_TX_EMPTY  0x04u               /* RR0 bit 2: Tx buffer empty */

/* ── Memory controller (src/mc.rs: MC_BASE 0x1FA00000) ────────────────────── */
#define MC_BASE           0xBFA00000u
#define MC_SYSID          (MC_BASE + 0x0018)  /* board revision / system id */
#define MC_MEMCFG0        (MC_BASE + 0x00C0)  /* bank 0 in 31:16, bank 1 in 15:0 */
#define MC_MEMCFG1        (MC_BASE + 0x00C8)  /* bank 2 in 31:16, bank 3 in 15:0 */

/*
 * One MEMCFG half-word. Valid whether or not the PROM ran: POST programs these
 * on a real boot, and `--load-elf` gets the same values from
 * MemoryController::post_map_banks, which maps the banks exactly as POST would
 * before the image is loaded.
 *
 *   15   14    13   12         8 7            0
 *   +----+-----+----+-----------+-------------+
 *   |    |rank | VLD|   size    |  base >> 22 |
 *   +----+-----+----+-----------+-------------+
 *
 * A dual-rank SIMM stores the size of *one* rank, so the installed total is
 * doubled — see MemoryController::encode_memcfg_half for the table this mirrors.
 */
#define MEMCFG_VALID(h)   (((h) >> 13) & 1u)
#define MEMCFG_BASE(h)    (((h) & 0xFFu) << 22)
#define MEMCFG_MB(h)      (((((h) >> 8) & 0x1Fu) + 1u) * 4u << (((h) >> 14) & 1u))

/* ── IRIS test device, GIO64 expansion slot 0 (src/testdev.rs) ────────────── */
#define TESTDEV_BASE      0xBF400000u
#define TESTDEV_SIGNATURE (TESTDEV_BASE + 0x00)  /* reads "IRIS" */
#define TESTDEV_PUTC      (TESTDEV_BASE + 0x04)
#define TESTDEV_DUMP      (TESTDEV_BASE + 0x08)
#define TESTDEV_EXIT      (TESTDEV_BASE + 0x0C)
/* Host monotonic nanoseconds and retired-guest-instruction count. Reading the
 * LO half latches the whole 64-bit value; the HI half then reads that same
 * latch, so LO-then-HI cannot tear. Present only when TESTDEV_CAPS has
 * TESTDEV_CAP_TIMEBASE — older emulator builds decode only 16 bytes here and
 * alias these back onto SIGNATURE/PUTC/DUMP/EXIT, so probe before you use it,
 * and never *write* an unprobed offset (0x0C aliases EXIT). */
#define TESTDEV_HOST_NS_LO (TESTDEV_BASE + 0x10)
#define TESTDEV_HOST_NS_HI (TESTDEV_BASE + 0x14)
#define TESTDEV_ICOUNT_LO  (TESTDEV_BASE + 0x18)
#define TESTDEV_ICOUNT_HI  (TESTDEV_BASE + 0x1C)
#define TESTDEV_CAPS       (TESTDEV_BASE + 0x20)
#define TESTDEV_CAP_TIMEBASE 0x00000001u
/* Run configuration, set by the host before the guest starts. A bare-metal
 * image loaded with --load-elf has no argv and no environment, so this register
 * is the only way to ask it for a shorter run. Every field is "unrestricted"
 * when zero, which is what an emulator without TESTDEV_CAP_RUN_CONFIG returns,
 * so it can be read unconditionally. See src/testdev.rs's RunConfig.
 *
 *   31            16 15   12 11             0
 *   +---------------+-------+---------------+
 *   |     groups    |repeats|    time_pct   |
 *   +---------------+-------+---------------+
 */
#define TESTDEV_RUN_CONFIG (TESTDEV_BASE + 0x24)
#define TESTDEV_CAP_RUN_CONFIG 0x00000002u
#define TESTDEV_RC_GROUPS(w)   (((w) >> 16) & 0xFFFFu)
#define TESTDEV_RC_REPEATS(w)  (((w) >> 12) & 0xFu)
#define TESTDEV_RC_TIME_PCT(w) ((w) & 0xFFFu)
#define TESTDEV_MAGIC     0x49524953u            /* 'I','R','I','S' */

/* ── CPU identity (src/mips_core.rs:348-364) ──────────────────────────────── */
#define PRID_R4400        0x00000440u
#define PRID_R5000        0x00002321u
#define FIR_R4000         0x00000500u
#define FIR_R5000         0x00002300u
#define PRID_IMP(p)       (((p) >> 8) & 0xFF)
#define PRID_REV_MAJOR(p) (((p) >> 4) & 0xF)
#define PRID_REV_MINOR(p) ((p) & 0xF)

/*
 * PRId implementation numbers for the MIPS CPUs SGI shipped. The architectural
 * ones (imp) are stable across vendors; which machine took which is the SGI
 * part:
 *
 *   IP20 Indigo          R4000
 *   IP22/IP24 Indy       R4000, R4400, R4600, R5000
 *   IP22 Indigo2         R4400, R4600, R8000, R10000
 *   IP32 O2              R5000, RM5200, RM7000, R10000, R12000
 *   IP30 Octane          R10000, R12000, R14000
 *   IP27/IP35 Origin     R10000, R12000, R14000
 *
 * R4000 and R4400 share imp 0x04 and are told apart by revision — major >= 4
 * is an R4400, which is the same rule IRIX and Linux use, and is why IRIS
 * reports PRId 0x0440.
 */
#define IMP_R4000         0x04   /* R4400 too — see PRID_REV_MAJOR */
#define IMP_R4400         0x04
#define IMP_R10000        0x09
#define IMP_R4300         0x0B
#define IMP_R12000        0x0E
#define IMP_R14000        0x0F
#define IMP_R8000         0x10
#define IMP_R4600         0x20
#define IMP_R4700         0x21
#define IMP_R4650         0x22
#define IMP_R5000         0x23
#define IMP_RM7000        0x27
#define IMP_RM5200        0x28

/* ── CP0 Status ───────────────────────────────────────────────────────────── */
#define ST_IE             0x00000001u
#define ST_EXL            0x00000002u
#define ST_ERL            0x00000004u
#define ST_UX             0x00000020u
#define ST_SX             0x00000040u
#define ST_KX             0x00000080u
#define ST_IM_SHIFT       8
#define ST_IM_MASK        0x0000FF00u
#define ST_DE             0x00010000u
#define ST_CE             0x00020000u
#define ST_CH             0x00040000u
#define ST_SR             0x00100000u
#define ST_TS             0x00200000u
#define ST_BEV            0x00400000u
#define ST_RE             0x02000000u
#define ST_FR             0x04000000u
#define ST_RP             0x08000000u
#define ST_CU0            0x10000000u
#define ST_CU1            0x20000000u
#define ST_CU2            0x40000000u
#define ST_CU3            0x80000000u

/* ── CP0 Cause ────────────────────────────────────────────────────────────── */
#define CAUSE_EXC_SHIFT   2
#define CAUSE_EXC_MASK    0x0000007Cu
#define CAUSE_IP_SHIFT    8
#define CAUSE_IP_MASK     0x0000FF00u
#define CAUSE_CE_SHIFT    28
#define CAUSE_CE_MASK     0x30000000u
#define CAUSE_BD          0x80000000u
#define CAUSE_EXC(c)      (((c) & CAUSE_EXC_MASK) >> CAUSE_EXC_SHIFT)

/* ExcCode values (MIPS III/IV) */
#define EXC_INT           0    /* interrupt                    */
#define EXC_MOD           1    /* TLB modified                 */
#define EXC_TLBL          2    /* TLB miss, load/fetch         */
#define EXC_TLBS          3    /* TLB miss, store              */
#define EXC_ADEL          4    /* address error, load/fetch    */
#define EXC_ADES          5    /* address error, store         */
#define EXC_IBE           6    /* bus error, instruction fetch */
#define EXC_DBE           7    /* bus error, data              */
#define EXC_SYS           8    /* syscall                      */
#define EXC_BP            9    /* breakpoint                   */
#define EXC_RI            10   /* reserved instruction         */
#define EXC_CPU           11   /* coprocessor unusable         */
#define EXC_OV            12   /* arithmetic overflow          */
#define EXC_TR            13   /* trap                         */
#define EXC_VCEI          14   /* virtual coherency, instr     */
#define EXC_FPE           15   /* floating point               */
#define EXC_WATCH         23   /* watchpoint                   */
#define EXC_VCED          31   /* virtual coherency, data      */

/* ── CP0 Config (src/mips_exec.rs:89-90 and around 777) ───────────────────── */
#define CFG_K0_MASK       0x00000007u
#define CFG_CU            0x00000008u
#define CFG_DB            0x00000010u   /* dcache line: 0=16B 1=32B */
#define CFG_IB            0x00000020u   /* icache line: 0=16B 1=32B */
#define CFG_DC_SHIFT      6
#define CFG_IC_SHIFT      9
#define CFG_SE            0x00001000u   /* R5K/Triton: L2 enable    */
#define CFG_SC            0x00020000u   /* 1 = no secondary cache   */
#define CFG_SB_SHIFT      22            /* L2 line: 4<<SB words     */
#define CFG_TR_SS_SHIFT   20            /* Triton only: L2 size     */
#define CFG_BE            0x00008000u   /* 1 = big endian           */
#define CFG_EC_SHIFT      28

/* ── FPU FCSR ─────────────────────────────────────────────────────────────── */
#define FCSR_RM_MASK      0x00000003u
#define FCSR_RM_RN        0
#define FCSR_RM_RZ        1
#define FCSR_RM_RP        2
#define FCSR_RM_RM        3
#define FCSR_FLAGS_SHIFT  2
#define FCSR_ENABLE_SHIFT 7
#define FCSR_CAUSE_SHIFT  12
#define FCSR_FS           0x01000000u
#define FCSR_CC0          0x00800000u   /* condition bit 0 (bit 23) */
#define FCSR_CC_SHIFT     25            /* CC1..CC7 at bits 25..31  */
/* Exception bits within each of the flag/enable/cause fields */
#define FP_I              0x01u         /* inexact       */
#define FP_U              0x02u         /* underflow     */
#define FP_O              0x04u         /* overflow      */
#define FP_Z              0x08u         /* divide by 0   */
#define FP_V              0x10u         /* invalid       */
#define FP_E              0x20u         /* unimplemented (cause only) */

/* ── Address spaces ───────────────────────────────────────────────────────── */
#define KUSEG_BASE        0x00000000u
#define KSEG0_BASE        0x80000000u
#define KSEG1_BASE        0xA0000000u
#define KSSEG_BASE        0xC0000000u
#define KSEG3_BASE        0xE0000000u
#define K0_TO_K1(a)       (((a) & 0x1FFFFFFFu) | KSEG1_BASE)
#define K1_TO_K0(a)       (((a) & 0x1FFFFFFFu) | KSEG0_BASE)
#define PHYS(a)           ((a) & 0x1FFFFFFFu)

/*
 * Turn a 32-bit compatibility-segment address into a pointer.
 *
 * In 64-bit mode KSEG0/KSEG1 are the *sign-extended* ranges
 * 0xffffffff80000000..0xffffffffbfffffff. A `u32` such as 0x88218000 held in a
 * 64-bit register is zero-extended to 0x0000000088218000, which is xkuseg —
 * TLB-mapped, and nothing like the address intended. Passing one of those to
 * `cache` makes the operation a silent no-op (or a TLB refill), which is
 * exactly the kind of failure that looks like a cache bug in the emulator.
 * Casting through s32 forces the sign extension.
 */
#define SEXT_PTR(a)       ((void *)(long)(s32)(a))
#define K1_PTR(p)         ((volatile void *)(long)(s32)K0_TO_K1((u32)(unsigned long)(p)))
#define K0_PTR(p)         ((volatile void *)(long)(s32)K1_TO_K0((u32)(unsigned long)(p)))

/* Physical RAM base on IP22/IP24 (src/physical.rs:136). Only the bottom 512 KB
 * of the physical map is RAM, as an alias of 0x08000000..0x0807ffff; the span
 * from 0x00080000 to 0x08000000 is unmapped and swallows writes silently. */
#define LOMEM_PHYS_BASE   0x08000000u
#define LOMEM_ALIAS_SIZE  0x00080000u

/* Where the suite relocates itself to and runs from (see PLAN.md §4.4).
 * KSEG0 view of physical 0x08200000 — 2 MB into real RAM. */
#define SUITE_KSEG0_BASE  0x88200000u

/* ── Exception vectors ────────────────────────────────────────────────────── */
#define VEC_TLB_REFILL    0x80000000u   /* BEV=0, 32-bit TLB refill */
#define VEC_XTLB_REFILL   0x80000080u   /* BEV=0, 64-bit TLB refill */
#define VEC_CACHE_ERR     0x80000100u
#define VEC_GENERAL       0x80000180u   /* BEV=0, all other exceptions */
#define VEC_BEV_TLB       0xBFC00200u
#define VEC_BEV_XTLB      0xBFC00280u
#define VEC_BEV_GENERAL   0xBFC00380u

/* ── TLB ──────────────────────────────────────────────────────────────────── */
#define TLB_ENTRIES       48            /* R4400 and R5000 both: 48 */
#define PM_4K             0x00000000u
#define PM_16K            0x00006000u
#define PM_64K            0x0001E000u
#define PM_256K           0x0007E000u
#define PM_1M             0x001FE000u
#define PM_4M             0x007FE000u
#define PM_16M            0x01FFE000u
/* EntryLo coherency attribute (bits 5:3) */
#define CA_UNCACHED       2
#define CA_CACHEABLE_NC   3             /* noncoherent = normal cached */
#define ELO_G             0x00000001u
#define ELO_V             0x00000002u
#define ELO_D             0x00000004u
#define ELO_C_SHIFT       3
#define ELO_PFN_SHIFT     6

/* ── CACHE instruction encodings (op = (target) | (operation << 2)) ───────── */
#define CACHE_I           0             /* primary instruction */
#define CACHE_D           1             /* primary data        */
#define CACHE_SI          2             /* secondary instruction */
#define CACHE_SD          3             /* secondary data      */
#define CACHE_OP_IDX_INV        (0 << 2)
#define CACHE_OP_IDX_LOAD_TAG   (1 << 2)
#define CACHE_OP_IDX_STORE_TAG  (2 << 2)
#define CACHE_OP_CREATE_DIRTY   (3 << 2)
#define CACHE_OP_HIT_INV        (4 << 2)
#define CACHE_OP_HIT_WB_INV     (5 << 2)
#define CACHE_OP_FILL           (5 << 2)   /* I-cache: Fill */
#define CACHE_OP_HIT_WB         (6 << 2)

#endif /* IRIS_H */
