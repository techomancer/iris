/*
 * hostplat.c — the platform layer for the host build.
 *
 * Everything harness/benchlib.c and cpu-tests' console.c provide on the
 * machine, provided here by a libc: a bump allocator over malloc, a monotonic
 * clock, and a console. Deliberately the same shapes, so harness/main.c
 * compiles against either without a single #if of its own beyond the two that
 * name genuinely machine-only things.
 *
 * There is no instruction counter. The host's retired-instruction count would
 * need perf counters and root, and the number it is compared against — guest
 * instructions per host second — has no host analogue anyway. It reports 0,
 * and the runner prints "n/a" rather than a fiction.
 */

#include "benchlib.h"

#include <time.h>
#include <stdarg.h>

unsigned char *work;
u32 work_bytes;
static unsigned char *pool;
static u32 pool_used;

u32 cpu_kind, cpu_prid, cpu_fir, cpu_config;
int have_l2;
int have_timebase = 1;
int have_testdev;
/*
 * No CP0 and no memory controller to read an inventory out of, and no portable
 * way to get one: /proc/cpuinfo, sysctl and GetSystemInfo are three different
 * answers on three platforms and none of them is what this suite is for. Left
 * zeroed, and print_inventory says so rather than showing a machine of zeroes.
 */
struct hwinv hw;
void bench_probe_hw(void) { }
/* No test device to read a configuration from: the golden generator and the
 * native baseline always run everything, full length. */
u32 bench_groups   = BG_ALL;
u32 bench_time_pct = 100;
u32 bench_repeats  = BENCH_REPEATS_DEFAULT;
u64 count_hz_measured = 1000000000ull;   /* the host clock is the time base */

void work_reset(void) { pool_used = 0; }

void *work_alloc(u32 n, u32 align)
{
    u32 off = (pool_used + (align - 1)) & ~(align - 1);
    if (off + n > work_bytes) {
        con_printf("\nwork_alloc(%u, %u) at offset %u of %u\n", n, align, off, work_bytes);
        panic("work area exhausted");
    }
    pool_used = off + n;
    return pool + off;
}

u64 bench_host_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
}

u64 bench_icount(void) { return 0; }
u32 bench_cp0_count(void) { return (u32)bench_host_ns(); }

void testdev_probe(void) { have_testdev = 0; }

void bench_exc_reset(void) { }
u32 bench_exc_count(void) { return 0; }

void testdev_exit(u32 code) { fflush(stdout); exit((int)code); }

void panic(const char *msg)
{
    fflush(stdout);
    fprintf(stderr, "\nPANIC: %s\n", msg);
    exit(127);
}

void con_hex32(u32 v) { printf("0x%08x", v); }
void con_hex64(u64 v) { printf("0x%016llx", (unsigned long long)v); }
void con_udec(unsigned long long v) { printf("%llu", v); }
void con_dec(long long v) { printf("%lld", v); }

/* Only the conversions harness/main.c actually uses. */
void con_printf(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    for (; *fmt; fmt++) {
        if (*fmt != '%') { con_putc(*fmt); continue; }
        fmt++;
        switch (*fmt) {
        case 's': con_puts(va_arg(ap, const char *)); break;
        case 'c': con_putc(va_arg(ap, int)); break;
        case 'd': con_dec(va_arg(ap, int)); break;
        case 'u': con_udec((unsigned)va_arg(ap, unsigned int)); break;
        case 'x': con_hex32(va_arg(ap, u32)); break;
        case 'X': con_hex64(va_arg(ap, u64)); break;
        case '%': con_putc('%'); break;
        case '\0': con_putc('%'); va_end(ap); return;
        default:  con_putc('%'); con_putc(*fmt); break;
        }
    }
    va_end(ap);
}

/*
 * Stand-in for benchlib.c's bench_init. Claims to be both CPUs so nothing is
 * skipped, and takes the whole working set up front so a kernel never measures
 * a page fault that the guest — running in physical RAM with no demand paging
 * — would never take.
 */
void bench_init(void)
{
    u32 i;
    cpu_kind = BCPU_R4400 | BCPU_R5000;
    work_bytes = WORK_WANT_BYTES;
    pool = (unsigned char *)malloc(work_bytes + 8192);
    if (!pool) panic("host: out of memory for the work area");
    pool = (unsigned char *)(((unsigned long)pool + 4095) & ~4095ul);
    for (i = 0; i < work_bytes; i += 4096) pool[i] = 0;
    work = pool;
    pool_used = 0;
}
