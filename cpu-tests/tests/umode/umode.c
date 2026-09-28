/* umode — User mode the way IRIX 6 runs it.
 *
 * IRIX 6 runs n32 processes with Status.UX = 1 (64-bit operations and the
 * 64-bit user address space) under a kernel that runs with KX = 0 on the
 * 32-bit kernels (IP22 among them). So every exception taken from such a
 * process switches the CPU between 64-bit and 32-bit addressing, and every
 * ERET back switches it again. The rest of the suite never does that: it runs
 * with KX = SX = UX = 1 throughout, and IRIX 5's o32 processes run with all
 * three clear, so neither exercises the switch.
 *
 * Each test runs a few words of code at a kuseg address in User mode, in
 * three Status settings:
 *
 *   all64   KX = SX = UX = 1   what the rest of the suite runs
 *   irix5   KX = SX = UX = 0   o32 under a 32-bit kernel
 *   irix6   KX = 0, UX = 1     n32 under a 32-bit kernel
 *
 * and records every exception the code takes: which vector, Cause, EPC,
 * BadVAddr, Context, XContext and EntryHi. The code ends in a syscall, which
 * returns to Kernel mode. Any other exception is resumed in User mode by the
 * test's policy: stepped over, or (TLB misses) fixed up by writing a TLB
 * entry and retried, or (interrupts) acknowledged and retried.
 *
 * The case that matters most is an exception taken by the FIRST instruction
 * executed after an ERET into User mode - a syscall right after an interrupt
 * returns, or a load that misses the TLB right after a refill returns. IRIX 6
 * does that all the time.
 *
 * Derived from the R4000 manual (exception vectors, Status.UX, the 64-bit
 * address spaces), not yet measured on silicon.
 */

#include "testlib.h"
#include "cp0.h"
#include "excoff.h"

#define A ".set push; .set mips3; .set noreorder; .set nomacro; .set noat\n\t"
#define Z "\n\t.set pop"

extern char _scratch_start[];

#define ST_KSU_USER   0x00000010u

/* Virtual layout. The code page and data page share one TLB entry (even and
 * odd page of one VPN2); the other two are mapped on demand by the fix-up
 * policy, i.e. the first touch of them is a TLB refill. */
#define UMX_CODE_VA   0x00400000ull
#define UMX_DATA_VA   0x00401000ull
#define UMX_DATA2_VA  0x00800000ull
#define UMX_CODE2_VA  0x00810000ull

/* Physical pages, offsets into the suite's scratch area (16 KB aligned). Each
 * one has the same address bits 13:12 as the virtual page it is mapped at, so
 * no virtually indexed cache sees two colours of it. */
#define UMX_P_CODE    0x0000u
#define UMX_P_DATA    0x1000u
#define UMX_P_DATA2   0x4000u
#define UMX_P_CODE2   0x8000u

#define UMX_TLB_INDEX 20u      /* code + data */
#define UMX_FIX_INDEX 21u      /* written by the fix-up policy */

#define UMX_NREC      8
#define UMX_RUNAWAY   12       /* this many exceptions and it goes home */

#define UMX_POLICY_SKIP  0     /* step over anything but Sys and Int */
#define UMX_POLICY_MAP   1     /* TLB miss: map it (map_lo0/1) and retry */

struct umx_rec {
    u32 vector;                /* VECID_* */
    u32 cause;
    u32 status;
    u32 pad;
    u64 epc;
    u64 badvaddr;
    u64 context;
    u64 xcontext;
    u64 entryhi;
};

struct umx_state {
    u32 n;                     /*  0: exceptions taken (may exceed UMX_NREC) */
    u32 policy;                /*  4 */
    u64 return_pc;             /*  8: where a syscall goes home to */
    u64 map_lo0;               /* 16: EntryLo0/1 the MAP policy writes */
    u64 map_lo1;               /* 24 */
    struct umx_rec rec[UMX_NREC];   /* 32, 56 bytes each */
};

_Static_assert(sizeof(struct umx_rec) == 56, "umx_rec layout is fixed by umx_handler");
_Static_assert(__builtin_offsetof(struct umx_state, rec) == 32, "umx.rec");

extern volatile struct umx_state umx;
extern void umx_handler(void);

/*
 * The exception hook. exc_dispatch has already filled `exc` with this
 * exception's registers and saved $at/$v0/$v1; $k0/$k1 are free.
 */
__asm__(
    "    .text\n"
    "    .set push; .set mips3; .set noreorder; .set nomacro; .set noat\n"
    "    .globl umx_handler\n"
    "    .ent umx_handler\n"
    "umx_handler:\n"
    "    lui     $k1, %hi(umx)\n"
    "    addiu   $k1, $k1, %lo(umx)\n"
    "    lui     $at, %hi(exc)\n"
    "    addiu   $at, $at, %lo(exc)\n"
    "    lw      $v0, 0($k1)\n"
    "    sltiu   $v1, $v0, 8\n"                 /* UMX_NREC */
    "    beqz    $v1, 2f\n"
    "    nop\n"
    "    sll     $v1, $v0, 6\n"
    "    sll     $k0, $v0, 3\n"
    "    subu    $v1, $v1, $k0\n"               /* n * 56 */
    "    addu    $v1, $v1, $k1\n"
    "    lw      $k0, 12($at)\n"                /* EXC_O_VECTOR */
    "    sw      $k0, 32($v1)\n"
    "    lw      $k0, 8($at)\n"                 /* EXC_O_CAUSE */
    "    sw      $k0, 36($v1)\n"
    "    lw      $k0, 4($at)\n"                 /* EXC_O_STATUS */
    "    sw      $k0, 40($v1)\n"
    "    ld      $k0, 24($at)\n"                /* EXC_O_EPC */
    "    sd      $k0, 48($v1)\n"
    "    ld      $k0, 32($at)\n"                /* EXC_O_BADVADDR */
    "    sd      $k0, 56($v1)\n"
    "    ld      $k0, 56($at)\n"                /* EXC_O_CONTEXT */
    "    sd      $k0, 64($v1)\n"
    "    ld      $k0, 64($at)\n"                /* EXC_O_XCONTEXT */
    "    sd      $k0, 72($v1)\n"
    "    ld      $k0, 48($at)\n"                /* EXC_O_ENTRYHI */
    "    sd      $k0, 80($v1)\n"
    "2:  addiu   $v0, $v0, 1\n"
    "    sw      $v0, 0($k1)\n"
    "    sltiu   $v1, $v0, 12\n"                /* UMX_RUNAWAY */
    "    beqz    $v1, umx_home\n"
    "    nop\n"
    "    lw      $v0, 8($at)\n"                 /* Cause */
    "    andi    $v0, $v0, 0x7c\n"
    "    xori    $v1, $v0, 0x20\n"              /* ExcCode 8, Sys */
    "    beqz    $v1, umx_home\n"
    "    nop\n"
    "    beqz    $v0, umx_int\n"                /* ExcCode 0, Int */
    "    nop\n"
    "    lw      $v1, 4($k1)\n"                 /* policy */
    "    beqz    $v1, umx_skip\n"
    "    nop\n"
    "    xori    $v1, $v0, 0x08\n"              /* ExcCode 2, TLBL */
    "    beqz    $v1, umx_map\n"
    "    nop\n"
    "    xori    $v1, $v0, 0x0c\n"              /* ExcCode 3, TLBS */
    "    bnez    $v1, umx_skip\n"
    "    nop\n"
    "umx_map:\n"                                /* EntryHi: loaded by the miss */
    "    li      $v1, 21\n"                     /* UMX_FIX_INDEX */
    "    mtc0    $v1, $0\n"
    "    mtc0    $zero, $5\n"
    "    ld      $v1, 16($k1)\n"
    "    dmtc0   $v1, $2\n"
    "    ld      $v1, 24($k1)\n"
    "    dmtc0   $v1, $3\n"
    "    nop\n"
    "    nop\n"
    "    tlbwi\n"
    "    nop\n"
    "    nop\n"
    "    nop\n"
    "    nop\n"
    "    b       umx_eret\n"                    /* retry: EPC unchanged */
    "    nop\n"
    "umx_int:\n"
    "    mtc0    $zero, $13\n"                  /* drop the software interrupt */
    "    nop\n"
    "    nop\n"
    "    b       umx_eret\n"                    /* retry */
    "    nop\n"
    "umx_skip:\n"
    "    dmfc0   $k0, $14\n"
    "    nop\n"
    "    daddiu  $k0, $k0, 4\n"
    "    dmtc0   $k0, $14\n"
    "    nop\n"
    "    b       umx_eret\n"
    "    nop\n"
    "umx_home:\n"
    "    mfc0    $k0, $12\n"
    "    nop\n"
    "    ori     $k0, $k0, 0x18\n"
    "    xori    $k0, $k0, 0x18\n"              /* KSU = Kernel; EXL still set */
    "    mtc0    $k0, $12\n"
    "    nop\n"
    "    ld      $k0, 8($k1)\n"
    "    dmtc0   $k0, $14\n"
    "    nop\n"
    "umx_eret:\n"
    "    lui     $k0, %hi(exc_save)\n"
    "    addiu   $k0, $k0, %lo(exc_save)\n"
    "    ld      $at, 0($k0)\n"
    "    ld      $v0, 8($k0)\n"
    "    ld      $v1, 16($k0)\n"
    "    ssnop\n"
    "    ssnop\n"
    "    ssnop\n"
    "    ssnop\n"
    "    eret\n"
    "    nop\n"
    "    .end umx_handler\n"
    "    .set pop\n"
    "    .section .bss\n"
    "    .align 3\n"
    "    .globl umx\n"
    "umx:\n"
    "    .space 480\n"                          /* sizeof(struct umx_state) */
    "    .text\n");

_Static_assert(sizeof(struct umx_state) == 480, "umx storage in umx_handler's .space");

/* ── the User-mode code ───────────────────────────────────────────────────── */
/*
 * Assembled here, copied to the code pages at run time. On entry $4, $5 and
 * $6 hold the runner's a0/a1/a2; $5 is always UMX_DATA_VA.
 */
#define UCODE(name, body)                                                  \
    __asm__("    .section .rodata\n"                                       \
            "    .set push; .set mips3; .set noreorder; .set nomacro; .set noat\n" \
            "    .align 2\n"                                               \
            "    .globl " #name "\n" #name ":\n" body                      \
            "    .globl " #name "_end\n" #name "_end:\n"                   \
            "    .set pop\n"                                               \
            "    .text\n");                                                \
    extern const u32 name[], name##_end[]

/* A syscall as the first instruction. */
UCODE(uc_sys,
    "    syscall\n"
    "    nop\n");

/* libc's _getuid: li v0, 1024; syscall. */
UCODE(uc_getuid,
    "    li      $2, 1024\n"
    "    syscall\n"
    "    nop\n");

/* Six nops for a pending interrupt to land in, then _getuid. */
UCODE(uc_int_sled,
    "    nop\n"
    "    nop\n"
    "    nop\n"
    "    nop\n"
    "    nop\n"
    "    nop\n"
    "    li      $2, 1024\n"
    "    syscall\n"
    "    nop\n");

UCODE(uc_break,
    "    break\n"
    "    syscall\n"
    "    nop\n");

/* A load through $4 as the first instruction; the value goes to data+16. */
UCODE(uc_load_first,
    "    lw      $8, 0($4)\n"
    "    sw      $8, 16($5)\n"
    "    syscall\n"
    "    nop\n");

UCODE(uc_load_third,
    "    sd      $4, 24($5)\n"                /* the address, for the log */
    "    nop\n"
    "    lw      $8, 0($4)\n"
    "    sw      $8, 16($5)\n"
    "    syscall\n"
    "    nop\n");

/* Jump through $6. */
UCODE(uc_jump,
    "    jr      $6\n"
    "    nop\n");

/* 64-bit operations, stored to data+0 and data+8. */
UCODE(uc_dword,
    "    daddiu  $8, $0, 1\n"
    "    dsll32  $8, $8, 4\n"
    "    sd      $8, 0($5)\n"
    "    ld      $9, 0($5)\n"
    "    daddu   $10, $8, $9\n"
    "    sd      $10, 8($5)\n"
    "    syscall\n"
    "    nop\n");

/* ── the runner ───────────────────────────────────────────────────────────── */

#define N_MODES 3
static const u32 modes[N_MODES] = { ST_KX | ST_SX | ST_UX, 0, ST_UX };
static const char *const mode_names[N_MODES] = { "all64", "irix5", "irix6" };

static u64 scratch_phys(void)
{
    return (u64)((u32)(unsigned long)_scratch_start & 0x1FFFFFFFu);
}

static u64 umx_lo(u32 page_off)
{
    return (((scratch_phys() + page_off) >> 12) << ELO_PFN_SHIFT) |
           ((u64)CA_CACHEABLE_NC << ELO_C_SHIFT) | ELO_V | ELO_D | ELO_G;
}

static void tlb_write(u32 index, u64 hi, u64 lo0, u64 lo1)
{
    cp0_index_set(index);
    cp0_pagemask_set(PM_4K);
    cp0_entryhi_set(hi);
    cp0_entrylo0_set(lo0);
    cp0_entrylo1_set(lo1);
    tlb_write_indexed();
}

/* Park an entry on a VPN2 of its own that nothing uses, invalid. */
static void tlb_retire(u32 index)
{
    tlb_write(index, 0x1FF00000ull + ((u64)index << 13), 0, 0);
}

/* Make sure no entry maps `va`. */
static void tlb_unmap(u64 va)
{
    cp0_entryhi_set(va);
    tlb_probe();
    if ((cp0_index() & 0x80000000u) == 0) tlb_retire(cp0_index() & 0x3F);
}

static void put_code(u32 page_off, const u32 *code, const u32 *end)
{
    volatile u32 *page = (volatile u32 *)(_scratch_start + page_off);
    unsigned n = (unsigned)(end - code), i;
    for (i = 0; i < n; i++) page[i] = code[i];
    SYNC();
    dcache_wb_invalidate_range(page, n * 4);
    icache_invalidate_range(page, n * 4);
}

static volatile u32 *data_page(void)
{
    return (volatile u32 *)(_scratch_start + UMX_P_DATA);
}

/*
 * Run `code` in User mode at `entry` with Status = the suite's own plus
 * `mode` (KX/SX/UX, IE/IM) and Cause = `cause` (software interrupt bits).
 */
static void umx_run(const u32 *code, const u32 *code_end, u64 entry, u32 mode,
                    u32 cause, u32 policy, u64 a0, u64 a2)
{
    u64 saved_hi = cp0_entryhi();
    u32 saved_pm = cp0_pagemask();
    u32 saved_status = cp0_status();
    u64 a1 = UMX_DATA_VA;
    u32 status;
    unsigned i;

    put_code(UMX_P_CODE, code, code_end);

    tlb_unmap(UMX_CODE_VA);
    tlb_unmap(UMX_DATA2_VA);
    tlb_unmap(UMX_CODE2_VA);
    tlb_retire(UMX_FIX_INDEX);
    tlb_write(UMX_TLB_INDEX, UMX_CODE_VA, umx_lo(UMX_P_CODE), umx_lo(UMX_P_DATA));
    /* The same physical lines through the mapping, so nothing cached from an
     * earlier run of the code page survives into this one. */
    icache_invalidate_range(SEXT_PTR((u32)UMX_CODE_VA), (u32)(code_end - code) * 4);

    umx.n = 0;
    umx.policy = policy;
    for (i = 0; i < UMX_NREC; i++) {
        umx.rec[i].vector = 0;
        umx.rec[i].cause = 0;
        umx.rec[i].status = 0;
        umx.rec[i].epc = 0;
        umx.rec[i].badvaddr = 0;
        umx.rec[i].context = 0;
        umx.rec[i].xcontext = 0;
        umx.rec[i].entryhi = 0;
    }
    exc_clear();
    exc_user_handler = (u32)(unsigned long)&umx_handler;

    status = (saved_status & ~(ST_CU0 | ST_KX | ST_SX | ST_UX | 0x18u | ST_IE | ST_IM_MASK))
             | ST_EXL | ST_KSU_USER | mode;
    __asm__ __volatile__(A
        "dla    $8, 1f\n\t"
        "sd     $8, 8(%0)\n\t"             /* umx.return_pc */
        "mfc0   $9, $12\n\t"
        "nop\n\t"
        "ori    $9, $9, 0x2\n\t"           /* EXL first: still Kernel */
        "mtc0   $9, $12\n\t"
        "nop; nop; nop\n\t"
        "mtc0   %2, $13\n\t"               /* software interrupts, if any */
        "mtc0   %1, $12\n\t"               /* KSU, KX/SX/UX, IE/IM; EXL kept */
        "nop; nop; nop\n\t"
        "move   $4, %4\n\t"
        "move   $5, %5\n\t"
        "move   $6, %6\n\t"
        "dmtc0  %3, $14\n\t"
        "nop; nop; nop\n\t"
        "eret\n\t"
        "nop\n\t"
        "1:" Z
        :: "r"(&umx), "r"(status), "r"(cause), "r"(entry), "r"(a0), "r"(a1), "r"(a2)
        : "$2", "$3", "$4", "$5", "$6", "$7", "$8", "$9", "$10", "$11",
          "$12", "$13", "$14", "$15", "$24", "$25", "memory");

    exc_user_handler = 0;
    cp0_status_set(saved_status);
    cp0_cause_set(0);

    tlb_retire(UMX_TLB_INDEX);
    tlb_retire(UMX_FIX_INDEX);
    cp0_entryhi_set(saved_hi);
    cp0_pagemask_set(saved_pm);
}

/* What the run recorded, for a failing test's log. */
static void umx_dump(int m)
{
    u32 i, n = umx.n < UMX_NREC ? umx.n : UMX_NREC;
    con_printf("      [%s: %u exception(s)]\n", mode_names[m], umx.n);
    for (i = 0; i < n; i++) {
        con_printf("        #%u vec=%u exc=%u epc=", i, umx.rec[i].vector,
                   CAUSE_EXC(umx.rec[i].cause));
        con_hex64(umx.rec[i].epc);
        con_puts(" badva=");
        con_hex64(umx.rec[i].badvaddr);
        con_printf(" sr=%x\n", umx.rec[i].status);
    }
}

/* Check record `i`: its ExcCode, vector and EPC. */
static void expect(int m, u32 i, u32 code, u32 vecid, u64 epc)
{
    CHECK_EQ_AT(mode_names[m], (int)i, CAUSE_EXC(umx.rec[i].cause), code);
    CHECK_EQ_AT(mode_names[m], (int)i, umx.rec[i].vector, vecid);
    CHECK_EQ_AT(mode_names[m], (int)i, umx.rec[i].epc, epc);
    /* Taken from User mode: KSU still says User inside the handler. */
    CHECK_EQ_AT(mode_names[m], (int)i, umx.rec[i].status & 0x18u, ST_KSU_USER);
}

#define DUMP_IF_FAILED(m, before) do { if (cur_test_fails != (before)) umx_dump(m); } while (0)

/* An EPC somewhere in uc_int_sled's nops or its li. */
#define CHECK_AT_SLED(m, epc)                                              \
    CHECK_EQ_AT(mode_names[m], 0,                                          \
                (epc) >= UMX_CODE_VA && (epc) <= UMX_CODE_VA + 24 && ((epc) & 3) == 0, 1)

static u32 refill_vecid(int m)
{
    return (modes[m] & ST_UX) ? (u32)VECID_XTLB : (u32)VECID_TLB;
}

/* ── tests ────────────────────────────────────────────────────────────────── */

/* A syscall as the first instruction after ERET into User mode. */
static void t_syscall_first(void)
{
    int m;
    for (m = 0; m < N_MODES; m++) {
        u32 before = cur_test_fails;
        umx_run(uc_sys, uc_sys_end, UMX_CODE_VA, modes[m], 0, UMX_POLICY_SKIP, 0, 0);
        CHECK_EQ_AT(mode_names[m], 0, umx.n, 1u);
        expect(m, 0, EXC_SYS, VECID_GENERAL, UMX_CODE_VA);
        DUMP_IF_FAILED(m, before);
    }
}

/* libc's _getuid: the syscall is the second instruction. */
static void t_syscall_second(void)
{
    int m;
    for (m = 0; m < N_MODES; m++) {
        u32 before = cur_test_fails;
        umx_run(uc_getuid, uc_getuid_end, UMX_CODE_VA, modes[m], 0, UMX_POLICY_SKIP, 0, 0);
        CHECK_EQ_AT(mode_names[m], 0, umx.n, 1u);
        expect(m, 0, EXC_SYS, VECID_GENERAL, UMX_CODE_VA + 4);
        DUMP_IF_FAILED(m, before);
    }
}

/* A break first, stepped over; then the syscall is the first instruction
 * after the handler's ERET. */
static void t_break_then_syscall(void)
{
    int m;
    for (m = 0; m < N_MODES; m++) {
        u32 before = cur_test_fails;
        umx_run(uc_break, uc_break_end, UMX_CODE_VA, modes[m], 0, UMX_POLICY_SKIP, 0, 0);
        CHECK_EQ_AT(mode_names[m], 0, umx.n, 2u);
        expect(m, 0, EXC_BP, VECID_GENERAL, UMX_CODE_VA);
        expect(m, 1, EXC_SYS, VECID_GENERAL, UMX_CODE_VA + 4);
        DUMP_IF_FAILED(m, before);
    }
}

/*
 * A software interrupt is pending when ERET enters User mode. It is taken in
 * User mode, from one of the first instructions - which one is a matter of
 * pipeline latency the architecture does not pin down, hence the nop sled -
 * and the handler acknowledges it and returns there. Then _getuid's syscall.
 */
static void t_interrupt_then_syscall(void)
{
    int m;
    for (m = 0; m < N_MODES; m++) {
        u32 before = cur_test_fails;
        umx_run(uc_int_sled, uc_int_sled_end, UMX_CODE_VA, modes[m] | ST_IE | (1u << ST_IM_SHIFT),
                1u << CAUSE_IP_SHIFT, UMX_POLICY_SKIP, 0, 0);
        CHECK_EQ_AT(mode_names[m], 0, umx.n, 2u);
        CHECK_EQ_AT(mode_names[m], 0, CAUSE_EXC(umx.rec[0].cause), (u32)EXC_INT);
        CHECK_EQ_AT(mode_names[m], 0, umx.rec[0].vector, (u32)VECID_GENERAL);
        CHECK_EQ_AT(mode_names[m], 0, umx.rec[0].status & 0x18u, ST_KSU_USER);
        CHECK_AT_SLED(m, umx.rec[0].epc);
        expect(m, 1, EXC_SYS, VECID_GENERAL, UMX_CODE_VA + 28);
        DUMP_IF_FAILED(m, before);
    }
}

/*
 * A load that misses the TLB. The refill goes to the XTLB vector when UX is
 * set (the address is in the 64-bit user space) and to the 32-bit one when it
 * is not; the fix-up maps the page and retries, so the load then runs as the
 * first instruction after the refill's ERET, and the syscall after it.
 */
static void load_miss(const u32 *code, const u32 *end, u64 load_pc)
{
    int m;
    volatile u32 *data2 = (volatile u32 *)(_scratch_start + UMX_P_DATA2);

    for (m = 0; m < N_MODES; m++) {
        u32 before = cur_test_fails;

        data2[0] = 0x5EC0DE00u + (u32)m;
        data_page()[4] = 0;
        SYNC();
        dcache_wb_invalidate_range(data2, 16);
        dcache_wb_invalidate_range(data_page(), 32);

        umx.map_lo0 = umx_lo(UMX_P_DATA2);
        umx.map_lo1 = 0;
        umx_run(code, end, UMX_CODE_VA, modes[m], 0, UMX_POLICY_MAP, UMX_DATA2_VA, 0);

        CHECK_EQ_AT(mode_names[m], 0, umx.n, 2u);
        expect(m, 0, EXC_TLBL, refill_vecid(m), load_pc);
        CHECK_EQ_AT(mode_names[m], 0, umx.rec[0].badvaddr, UMX_DATA2_VA);
        CHECK_EQ_AT(mode_names[m], 0, umx.rec[0].entryhi & ~0x1FFFull, UMX_DATA2_VA);
        CHECK_EQ_AT(mode_names[m], 0, (umx.rec[0].context >> 4) & 0x7FFFFull, UMX_DATA2_VA >> 13);
        if (modes[m] & ST_UX) {
            /* XContext: R (32:31) = 0 for xuseg, BadVPN2 (30:4) = VA[39:13]. */
            CHECK_EQ_AT(mode_names[m], 0, (umx.rec[0].xcontext >> 4) & 0x7FFFFFFull, UMX_DATA2_VA >> 13);
            CHECK_EQ_AT(mode_names[m], 0, (umx.rec[0].xcontext >> 31) & 3ull, 0u);
        }
        expect(m, 1, EXC_SYS, VECID_GENERAL, load_pc + 8);
        CHECK_EQ_AT(mode_names[m], 0, data_page()[4], 0x5EC0DE00u + (u32)m);
        DUMP_IF_FAILED(m, before);
    }
}

static void t_load_miss_first(void)
{
    load_miss(uc_load_first, uc_load_first_end, UMX_CODE_VA);
}

static void t_load_miss_third(void)
{
    load_miss(uc_load_third, uc_load_third_end, UMX_CODE_VA + 8);
}

/*
 * An instruction fetch that misses the TLB: a jump to an unmapped page
 * (`jump`), or ERET straight to one (`entry`, so the miss comes before any
 * User instruction has executed). The fix-up maps the second code page,
 * whose first instruction is a syscall.
 */
static void fetch_miss(int via_jump)
{
    int m;

    put_code(UMX_P_CODE2, uc_sys, uc_sys_end);
    for (m = 0; m < N_MODES; m++) {
        u32 before = cur_test_fails;

        umx.map_lo0 = umx_lo(UMX_P_CODE2);
        umx.map_lo1 = 0;
        if (via_jump)
            umx_run(uc_jump, uc_jump_end, UMX_CODE_VA, modes[m], 0, UMX_POLICY_MAP, 0, UMX_CODE2_VA);
        else
            umx_run(uc_sys, uc_sys_end, UMX_CODE2_VA, modes[m], 0, UMX_POLICY_MAP, 0, 0);

        CHECK_EQ_AT(mode_names[m], 0, umx.n, 2u);
        expect(m, 0, EXC_TLBL, refill_vecid(m), UMX_CODE2_VA);
        CHECK_EQ_AT(mode_names[m], 0, umx.rec[0].badvaddr, UMX_CODE2_VA);
        CHECK_EQ_AT(mode_names[m], 0, umx.rec[0].entryhi & ~0x1FFFull, UMX_CODE2_VA);
        expect(m, 1, EXC_SYS, VECID_GENERAL, UMX_CODE2_VA);
        DUMP_IF_FAILED(m, before);
    }
}

static void t_fetch_miss_jump(void)  { fetch_miss(1); }
static void t_fetch_miss_entry(void) { fetch_miss(0); }

/* A load from KSEG0 in User mode, as the first instruction: Address Error. */
static void t_kseg0_load_first(void)
{
    int m;
    for (m = 0; m < N_MODES; m++) {
        u32 before = cur_test_fails;
        umx_run(uc_load_first, uc_load_first_end, UMX_CODE_VA, modes[m], 0, UMX_POLICY_SKIP,
                0xFFFFFFFF80000000ull, 0);
        CHECK_EQ_AT(mode_names[m], 0, umx.n, 2u);
        expect(m, 0, EXC_ADEL, VECID_GENERAL, UMX_CODE_VA);
        CHECK_EQ_AT(mode_names[m], 0, umx.rec[0].badvaddr, 0xFFFFFFFF80000000ull);
        expect(m, 1, EXC_SYS, VECID_GENERAL, UMX_CODE_VA + 8);
        DUMP_IF_FAILED(m, before);
    }
}

/* The same, third instruction: the first two are not the ones after ERET. */
static void t_kseg0_load_third(void)
{
    int m;
    for (m = 0; m < N_MODES; m++) {
        u32 before = cur_test_fails;
        umx_run(uc_load_third, uc_load_third_end, UMX_CODE_VA, modes[m], 0, UMX_POLICY_SKIP,
                0xFFFFFFFF80000000ull, 0);
        CHECK_EQ_AT(mode_names[m], 0, umx.n, 2u);
        expect(m, 0, EXC_ADEL, VECID_GENERAL, UMX_CODE_VA + 8);
        CHECK_EQ_AT(mode_names[m], 0, umx.rec[0].badvaddr, 0xFFFFFFFF80000000ull);
        expect(m, 1, EXC_SYS, VECID_GENERAL, UMX_CODE_VA + 16);
        if (cur_test_fails != before) {
            con_printf("      [%s: address ", mode_names[m]);
            con_hex64(*(volatile u64 *)((volatile char *)data_page() + 24));
            con_printf(" loaded %x]\n", data_page()[4]);
        }
        DUMP_IF_FAILED(m, before);
    }
}

/* A misaligned word load as the first instruction: Address Error, in every
 * mode (nothing about it depends on the address space). */
static void t_unaligned_load_first(void)
{
    int m;
    for (m = 0; m < N_MODES; m++) {
        u32 before = cur_test_fails;
        umx_run(uc_load_first, uc_load_first_end, UMX_CODE_VA, modes[m], 0, UMX_POLICY_SKIP,
                UMX_DATA_VA + 2, 0);
        CHECK_EQ_AT(mode_names[m], 0, umx.n, 2u);
        expect(m, 0, EXC_ADEL, VECID_GENERAL, UMX_CODE_VA);
        CHECK_EQ_AT(mode_names[m], 0, umx.rec[0].badvaddr, UMX_DATA_VA + 2);
        expect(m, 1, EXC_SYS, VECID_GENERAL, UMX_CODE_VA + 8);
        DUMP_IF_FAILED(m, before);
    }
}

/* With UX set, 64-bit operations are legal in User mode. */
static void t_dword_ops_with_ux(void)
{
    int m;
    for (m = 0; m < N_MODES; m++) {
        u32 before = cur_test_fails;
        volatile u64 *d = (volatile u64 *)data_page();
        if (!(modes[m] & ST_UX)) continue;
        d[0] = 0;
        d[1] = 0;
        SYNC();
        dcache_wb_invalidate_range(d, 16);
        umx_run(uc_dword, uc_dword_end, UMX_CODE_VA, modes[m], 0, UMX_POLICY_SKIP, 0, 0);
        CHECK_EQ_AT(mode_names[m], 0, umx.n, 1u);
        expect(m, 0, EXC_SYS, VECID_GENERAL, UMX_CODE_VA + 24);
        CHECK_EQ_AT(mode_names[m], 0, d[0], 1ull << 36);
        CHECK_EQ_AT(mode_names[m], 0, d[1], 2ull << 36);
        DUMP_IF_FAILED(m, before);
    }
}

static const struct test tests[] = {
    TEST("umode/syscall_first",        t_syscall_first,          CPU_ALL),
    TEST("umode/syscall_second",       t_syscall_second,         CPU_ALL),
    TEST("umode/break_then_syscall",   t_break_then_syscall,     CPU_ALL),
    TEST("umode/interrupt_then_sys",   t_interrupt_then_syscall, CPU_ALL),
    TEST("umode/load_miss_first",      t_load_miss_first,        CPU_ALL),
    TEST("umode/load_miss_third",      t_load_miss_third,        CPU_ALL),
    TEST("umode/fetch_miss_jump",      t_fetch_miss_jump,        CPU_ALL),
    TEST("umode/fetch_miss_entry",     t_fetch_miss_entry,       CPU_ALL),
    TEST("umode/kseg0_load_first",     t_kseg0_load_first,       CPU_ALL),
    TEST("umode/kseg0_load_third",     t_kseg0_load_third,       CPU_ALL),
    TEST("umode/unaligned_load_first", t_unaligned_load_first,   CPU_ALL),
    TEST("umode/dword_ops_with_ux",    t_dword_ops_with_ux,      CPU_ALL),
};

const struct test_group group_umode = {
    "umode", tests, sizeof(tests) / sizeof(tests[0])
};
