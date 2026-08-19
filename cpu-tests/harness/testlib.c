/* testlib.c — result accounting, exception plumbing, and the runner. */

#include "testlib.h"
#include "cp0.h"
#include "excoff.h"

u32 cpu_kind, cpu_prid, cpu_fir, cpu_config;
u32 n_pass, n_fail, n_skip, n_tests_run;
u32 cur_test_fails;

volatile struct exc_record exc;
volatile u32 exc_resume_mode = EXC_RESUME_SKIP;
volatile u32 exc_user_handler = 0;

/* The handler in start.S stores through the offsets in excoff.h and cannot see
 * this declaration — so pin them together here. */
_Static_assert(__builtin_offsetof(struct exc_record, count)    == EXC_O_COUNT,    "exc.count");
_Static_assert(__builtin_offsetof(struct exc_record, status)   == EXC_O_STATUS,   "exc.status");
_Static_assert(__builtin_offsetof(struct exc_record, cause)    == EXC_O_CAUSE,    "exc.cause");
_Static_assert(__builtin_offsetof(struct exc_record, vector)   == EXC_O_VECTOR,   "exc.vector");
_Static_assert(__builtin_offsetof(struct exc_record, fcsr)     == EXC_O_FCSR,     "exc.fcsr");
_Static_assert(__builtin_offsetof(struct exc_record, epc)      == EXC_O_EPC,      "exc.epc");
_Static_assert(__builtin_offsetof(struct exc_record, badvaddr) == EXC_O_BADVADDR, "exc.badvaddr");
_Static_assert(__builtin_offsetof(struct exc_record, errorepc) == EXC_O_ERROREPC, "exc.errorepc");
_Static_assert(__builtin_offsetof(struct exc_record, entryhi)  == EXC_O_ENTRYHI,  "exc.entryhi");
_Static_assert(__builtin_offsetof(struct exc_record, context)  == EXC_O_CONTEXT,  "exc.context");
_Static_assert(__builtin_offsetof(struct exc_record, xcontext) == EXC_O_XCONTEXT, "exc.xcontext");
_Static_assert(sizeof(struct exc_record) == EXC_SIZEOF, "exc_record size");

void exc_clear(void)
{
    exc.count = 0;
    exc.status = 0;
    exc.cause = 0;
    exc.vector = 0;
    exc.fcsr = 0;
    exc.epc = 0;
    exc.badvaddr = 0;
    exc.errorepc = 0;
    exc.entryhi = 0;
    exc.context = 0;
    exc.xcontext = 0;
    exc_resume_mode = EXC_RESUME_SKIP;
    exc_user_handler = 0;
}

/* Trampolines from start.S. */
extern u32 tramp_tlb[], tramp_tlb_end[];
extern u32 tramp_xtlb[], tramp_xtlb_end[];
extern u32 tramp_general[], tramp_general_end[];

static void install_one(u32 vec, const u32 *src, const u32 *end)
{
    volatile u32 *dst = (volatile u32 *)SEXT_PTR(vec);
    unsigned n = (unsigned)(end - src);
    unsigned i;
    if (n > 32) panic("vector trampoline too long");
    for (i = 0; i < n; i++) dst[i] = src[i];
}

void exc_install(void)
{
    install_one(VEC_TLB_REFILL,  tramp_tlb,     tramp_tlb_end);
    install_one(VEC_XTLB_REFILL, tramp_xtlb,    tramp_xtlb_end);
    install_one(VEC_GENERAL,     tramp_general, tramp_general_end);

    /* The stores above went through the D-cache; instruction fetch at the
     * vectors must see them. */
    dcache_wb_invalidate_range(SEXT_PTR(VEC_TLB_REFILL), 0x200);
    icache_invalidate_range(SEXT_PTR(VEC_TLB_REFILL), 0x200);
    SYNC();
}

/* ── cache maintenance ────────────────────────────────────────────────────── */
/*
 * Step by 16: the R4400 line size, and a correct (if redundant) step for the
 * R5000's 32-byte lines.
 *
 * Everything here works on `char *`, never on a u32. A `cache` instruction
 * given a zero-extended 0x0000000088218000 instead of the sign-extended
 * 0xffffffff88218000 addresses xkuseg, misses, and does nothing at all —
 * silently, since a missing Hit_* operation is architecturally a no-op.
 */
/* Distinct names: CACHE_OP has its own __a, and shadowing it here would
 * silently apply the operation to an uninitialised address. */
#define RANGE_LOOP(op, addr, len)                                          \
    do {                                                                   \
        char *__rl_p = (char *)((unsigned long)(addr) & ~15ul);            \
        char *__rl_e = (char *)(((unsigned long)(addr) + (len) + 15)       \
                                & ~15ul);                                  \
        for (; __rl_p != __rl_e; __rl_p += 16) CACHE_OP(op, __rl_p);       \
    } while (0)

/*
 * Whether a secondary cache is present, from Config.SC (0 = present).
 *
 * This is not cosmetic. On a machine with an L2, a primary-cache writeback
 * lands in the SECONDARY cache, not in memory. Reaching memory — which is what
 * an uncached KSEG1 read sees — takes a second operation against the SD
 * target. Flushing only the primary and then reading through KSEG1 returns
 * stale data, and looks exactly like a broken writeback in the emulator.
 */
int have_l2;

void cache_detect(void)
{
    have_l2 = (cp0_config() & CFG_SC) == 0;
}

void icache_invalidate_range(volatile void *addr, u32 len)
{
    RANGE_LOOP(CACHE_I | CACHE_OP_HIT_INV, addr, len);
    if (have_l2) RANGE_LOOP(CACHE_SD | CACHE_OP_HIT_WB_INV, addr, len);
}

/* Push dirty lines all the way out to memory, then drop them. Use before
 * reading the same bytes through KSEG1. */
void dcache_wb_invalidate_range(volatile void *addr, u32 len)
{
    RANGE_LOOP(CACHE_D | CACHE_OP_HIT_WB_INV, addr, len);
    if (have_l2) RANGE_LOOP(CACHE_SD | CACHE_OP_HIT_WB_INV, addr, len);
}

/*
 * Drop lines WITHOUT writing them back, at every level. Use after writing
 * through KSEG1: a writeback here would push the stale cached copy over the
 * value that just went straight to memory.
 */
void dcache_invalidate_range(volatile void *addr, u32 len)
{
    RANGE_LOOP(CACHE_D | CACHE_OP_HIT_INV, addr, len);
    if (have_l2) RANGE_LOOP(CACHE_SD | CACHE_OP_HIT_INV, addr, len);
}

/* ── failure reporting ────────────────────────────────────────────────────── */
/* Failures print under the test's own name, which the runner has already
 * emitted without a newline — so the first failure closes that line. */
static int line_open;   /* runner set: the "name ...." line is still open */

void fail_begin(const char *file, int line, const char *expr)
{
    const char *base = file, *p = file;
    while (*p) { if (*p == '/') base = p + 1; p++; }
    if (line_open) { con_puts("FAIL\n"); line_open = 0; }
    con_printf("    %s:%d: %s", base, line, expr);
}

void fail_u64(const char *what, u64 got, u64 want)
{
    if (what && what[0]) con_printf(" %s", what);
    con_puts("\n      got  ");
    con_hex64(got);
    con_puts("\n      want ");
    con_hex64(want);
}

void fail_end(void)
{
    con_putc('\n');
}

/* ── the runner ───────────────────────────────────────────────────────────── */
extern const struct test_group *const all_groups[];
extern const unsigned n_groups;

static void identify(void)
{
    cpu_prid   = cp0_prid();
    cpu_config = cp0_config();
    cpu_fir    = fir();

    switch (PRID_IMP(cpu_prid)) {
    case IMP_R4400: cpu_kind = CPU_R4400; break;
    case IMP_R5000: cpu_kind = CPU_R5000; break;
    default:        cpu_kind = 0; break;
    }
}

static const char *cpu_name(void)
{
    if (cpu_kind == CPU_R4400) return "R4400";
    if (cpu_kind == CPU_R5000) return "R5000";
    return "unknown";
}

/* Pad the test name out so the PASS/FAIL column lines up. */
static void print_name(const char *name)
{
    int n = 0;
    const char *p = name;
    while (*p) { con_putc(*p++); n++; }
    con_putc(' ');
    for (n++; n < 44; n++) con_putc('.');
    con_putc(' ');
}

int main(void)
{
    unsigned gi, ti;

    con_init();
    testdev_probe();
    identify();
    cache_detect();
    exc_clear();
    exc_install();

    con_puts("\n");
    con_puts("========================================================\n");
    con_printf(" IRIS CPU test suite   cpu=%s\n", cpu_name());
    con_printf("   PRId   %x    FIR    %x\n", cpu_prid, cpu_fir);
    con_printf("   Config %x    testdev %s   L2 %s\n", cpu_config,
               have_testdev ? "yes" : "no", have_l2 ? "yes" : "no");
    con_puts("========================================================\n");

    if (cpu_kind == 0) {
        con_puts("\nUNKNOWN CPU — refusing to run: every expectation in this\n"
                 "suite is selected by PRId, so results would be meaningless.\n");
        con_puts("\nIRIS-CPUTEST-DONE rc=127\n");
        testdev_exit(127);
    }

    for (gi = 0; gi < n_groups; gi++) {
        const struct test_group *g = all_groups[gi];
        for (ti = 0; ti < g->count; ti++) {
            const struct test *t = &g->tests[ti];

            if (!(t->cpus & cpu_kind)) {
                n_skip++;
                print_name(t->name);
                con_printf("skip (%s only)\n",
                           t->cpus == CPU_R5000 ? "R5000" : "R4400");
                continue;
            }

            print_name(t->name);
            line_open = 1;

            cur_test_fails = 0;
            exc_clear();
            t->fn();
            /* Leave no state behind for the next test. */
            exc_clear();

            n_tests_run++;
            if (cur_test_fails == 0 && line_open) con_puts("PASS\n");
            line_open = 0;
        }
    }

    con_puts("========================================================\n");
    con_printf(" RESULT: %u checks passed, %u failed", n_pass, n_fail);
    if (n_skip) con_printf(", %u tests skipped", n_skip);
    con_printf("  (%u tests)\n", n_tests_run);
    con_puts("========================================================\n");

    /* The token CI matches on, and the exit code when the test device is
     * present. Exit code saturates at 100 so it can never collide with the
     * harness's own 127. */
    {
        u32 rc = n_fail > 100 ? 100 : n_fail;
        con_printf("\nIRIS-CPUTEST-DONE rc=%u\n", rc);
        testdev_exit(rc);
    }
    return 0;
}
