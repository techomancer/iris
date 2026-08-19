/* testlib.h — the test-suite harness API.
 *
 * A test is a function registered in a table with the CPUs it applies to. It
 * calls CHECK_* macros; each records a pass or a failure line. The runner
 * prints one line per test and a summary, then reports the failure count
 * through the test device (exit code) and the serial console (DONE token).
 */
#ifndef TESTLIB_H
#define TESTLIB_H

#include "iris.h"

typedef unsigned char       u8;
typedef unsigned short      u16;
typedef unsigned int        u32;
typedef unsigned long long  u64;
typedef signed char         s8;
typedef short               s16;
typedef int                 s32;
typedef long long           s64;

/* ── CPU selection ────────────────────────────────────────────────────────── */
/* Bitmask: a test declares which CPUs it is valid on. */
#define CPU_R4400   0x1
#define CPU_R5000   0x2
#define CPU_ALL     (CPU_R4400 | CPU_R5000)

extern u32 cpu_kind;          /* CPU_R4400 or CPU_R5000, set at startup */
extern u32 cpu_prid;
extern u32 cpu_fir;
extern u32 cpu_config;
static inline int is_r5000(void) { return cpu_kind == CPU_R5000; }
static inline int is_r4400(void) { return cpu_kind == CPU_R4400; }

/* ── Console ──────────────────────────────────────────────────────────────── */
void con_init(void);
void con_putc(int c);
void con_puts(const char *s);
void con_hex32(u32 v);
void con_hex64(u64 v);
void con_dec(long long v);
void con_udec(unsigned long long v);
/* Minimal printf: %s %c %d %u %x (32-bit), %X/%lx (64-bit), %% */
void con_printf(const char *fmt, ...);
/* Wait for the serial transmitter to drain. Call before anything that stops
 * the machine, or the tail of the last line is lost. */
void con_flush(void);

/* ── Test device (absent on real hardware; probed at startup) ─────────────── */
extern int have_testdev;
void testdev_probe(void);
void testdev_dump(u32 tag);
void testdev_exit(u32 code);   /* no return when present */

/* ── Result accounting ────────────────────────────────────────────────────── */
extern u32 n_pass, n_fail, n_skip, n_tests_run;
extern u32 cur_test_fails;

void fail_begin(const char *file, int line, const char *expr);
void fail_u64(const char *what, u64 got, u64 want);
void fail_end(void);

/* ── CHECK macros ─────────────────────────────────────────────────────────── */
/* Every check counts; a failure prints one indented line under the test name
 * and lets the test continue, so one broken instruction doesn't hide the rest. */
#define CHECK(expr)                                                        \
    do {                                                                   \
        if (expr) { n_pass++; }                                            \
        else { n_fail++; cur_test_fails++;                                 \
               fail_begin(__FILE__, __LINE__, #expr); fail_end(); }        \
    } while (0)

#define CHECK_EQ(got, want)                                                \
    do {                                                                   \
        u64 _g = (u64)(got), _w = (u64)(want);                             \
        if (_g == _w) { n_pass++; }                                        \
        else { n_fail++; cur_test_fails++;                                 \
               fail_begin(__FILE__, __LINE__, #got);                       \
               fail_u64("", _g, _w); fail_end(); }                         \
    } while (0)

/* Same, with a caller-supplied label — use inside loops so the failing
 * iteration identifies itself. */
#define CHECK_EQ_AT(label, idx, got, want)                                 \
    do {                                                                   \
        u64 _g = (u64)(got), _w = (u64)(want);                             \
        if (_g == _w) { n_pass++; }                                        \
        else { n_fail++; cur_test_fails++;                                 \
               fail_begin(__FILE__, __LINE__, #got);                       \
               con_printf(" [%s=%d]", (label), (int)(idx));                \
               fail_u64("", _g, _w); fail_end(); }                         \
    } while (0)

#define CHECK_NE(got, notwant)                                             \
    do {                                                                   \
        u64 _g = (u64)(got), _w = (u64)(notwant);                          \
        if (_g != _w) { n_pass++; }                                        \
        else { n_fail++; cur_test_fails++;                                 \
               fail_begin(__FILE__, __LINE__, #got);                       \
               con_printf(" unexpectedly == "); con_hex64(_g);             \
               fail_end(); }                                               \
    } while (0)

/* ── Exception capture ────────────────────────────────────────────────────── */
/* The general handler records here. `count` is cumulative until cleared, so a
 * test can assert "exactly one exception happened". */
/* Field order and padding are fixed by excoff.h, which the assembly handler
 * stores through; testlib.c static-asserts every offset. Do not reorder. */
struct exc_record {
    u32 count;
    u32 status;
    u32 cause;
    u32 vector;     /* VECID_TLB / VECID_XTLB / VECID_GENERAL */
    u32 fcsr;       /* FCSR as seen in the handler; 0 if CU1 was clear */
    u32 pad;
    u64 epc;
    u64 badvaddr;
    u64 errorepc;
    u64 entryhi;
    u64 context;
    u64 xcontext;
};
extern volatile struct exc_record exc;

void exc_clear(void);
void exc_install(void);          /* write the vectors and flush I-cache */

/* How the handler resumes. Default EXC_RESUME_SKIP: EPC += 4 (or, when
 * Cause.BD is set, EPC += 8) so the faulting instruction is stepped over. */
#define EXC_RESUME_SKIP    0
#define EXC_RESUME_RETRY   1     /* return to EPC unchanged (for TLB fixups)  */
extern volatile u32 exc_resume_mode;

/* A handler hook: when non-zero, the general handler jumps here (in kernel
 * mode, with $k0/$k1 free) instead of its default record-and-skip. It owns the
 * ERET and the restore of $at/$v0/$v1 from exc_save. */
extern volatile u32 exc_user_handler;

/*
 * A ready-made hook for tests that break the normal resume path: it resumes at
 * `exl_resume_pc` rather than at EPC. Needed when an exception is taken with
 * Status.EXL already set, since EPC is then not updated and returning through
 * it would loop forever.
 */
void exl_resume_handler(void);
extern volatile u64 exl_resume_pc;

/* ── Assertions about a captured exception ────────────────────────────────── */
#define CHECK_EXC(code)                                                    \
    do {                                                                   \
        CHECK_EQ(exc.count, 1u);                                           \
        CHECK_EQ(CAUSE_EXC(exc.cause), (u32)(code));                       \
    } while (0)

#define CHECK_NO_EXC()  CHECK_EQ(exc.count, 0u)

/* ── Test registration ────────────────────────────────────────────────────── */
struct test {
    const char *name;      /* "alu/dadd_overflow" */
    void (*fn)(void);
    u32 cpus;              /* CPU_ALL, CPU_R4400, CPU_R5000 */
};

/* Every test file defines its tests then exposes a table via TEST_GROUP. */
#define TEST(name_, fn_, cpus_)  { name_, fn_, cpus_ }

/* Group registration: each area exports one of these. */
struct test_group {
    const char *name;
    const struct test *tests;
    unsigned count;
};

/* Declared by each area's .c file, collected in tests.c */
#define DECLARE_GROUP(g)  extern const struct test_group g

/* ── Small helpers available to tests ─────────────────────────────────────── */
u32  cp0_status_read(void);
void cp0_status_write(u32 v);
u32  cp0_cause_read(void);
u32  cp0_config_read(void);
u32  cp0_prid_read(void);
u32  cp0_count_read(void);
void cp0_count_write(u32 v);
u32  cp0_compare_read(void);
void cp0_compare_write(u32 v);
u32  fpu_fir_read(void);
u32  fpu_fcsr_read(void);
void fpu_fcsr_write(u32 v);

/* Set by cache_detect(): a secondary cache is present (Config.SC == 0). */
extern int have_l2;
void cache_detect(void);

/* Cache maintenance over a range. These take a pointer rather than a u32 so
 * the address reaches the `cache` instruction sign-extended — see SEXT_PTR in
 * iris.h for why that matters. */
void icache_invalidate_range(volatile void *addr, u32 len);
void dcache_wb_invalidate_range(volatile void *addr, u32 len);
void dcache_invalidate_range(volatile void *addr, u32 len);

/* Halt with a message — unrecoverable harness failure. */
void panic(const char *msg) __attribute__((noreturn));

#endif /* TESTLIB_H */
