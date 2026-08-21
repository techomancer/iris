/* console.c — serial output via the Z85C30 SCC, plus the IRIS test device.
 *
 * Two independent sinks, both optional:
 *   - SCC channel B (tty1, the SGI serial console). Always attempted; the
 *     PROM has already programmed the baud rate and WR registers before it
 *     hands control over, so we only poll TX-empty and write.
 *   - The IRIS test device's PUTC, when present. Absent on real hardware (an
 *     empty GIO slot times out), which is why we probe rather than assume.
 */

#include "console.h"

#define RD8(a)      (*(volatile u8  *)(unsigned long)(a))
#define WR8(a, v)   (*(volatile u8  *)(unsigned long)(a) = (u8)(v))
#define RD32(a)     (*(volatile u32 *)(unsigned long)(a))
#define WR32(a, v)  (*(volatile u32 *)(unsigned long)(a) = (u32)(v))

int have_testdev = 0;

/* Bounded so a wedged SCC can't hang the whole suite. */
#define TX_SPIN_LIMIT 100000

/* Latched once the SCC has proved it is not transmitting — see scc_putc. */
static int scc_dead = 0;

void con_init(void)
{
    /* The PROM leaves the console configured. Nothing to do; kept as a hook
     * for a future "run before POST" mode that would have to program WR4/9/11
     * /12/13/14/3/5 itself. */
}

/*
 * Write one byte to the serial console, giving up on the port for good once it
 * has demonstrated that it is not going to transmit.
 *
 * The transmitter is enabled by WR5, which the PROM programs — so booting an
 * image with `--load-elf` (no PROM, straight into RAM) leaves it disabled. The
 * SCC's four-byte holding queue then fills, TX_BUFFER_EMPTY goes low and never
 * comes back, and every subsequent character burns the whole spin limit. That
 * is not a small cost: it was about 78 seconds of a 117-second bare-metal
 * benchmark run, spent waiting on a port with nothing on the other end.
 *
 * Latching only when `have_testdev` is what keeps this from silently dropping
 * output: with a test device the host is reading that instead and serial is
 * redundant, and without one serial is the only sink there is, so a slow port
 * still beats no output. A PROM-booted run has a working SCC, never trips the
 * limit, and is unaffected either way — which matters, because run-prom.sh
 * decides pass or fail by grepping the serial log.
 */
static void scc_putc(int c)
{
    int spins = 0;

    if (scc_dead) return;
    while (!(RD8(SCC_CHB_CMD) & SCC_RR0_TX_EMPTY)) {
        if (++spins > TX_SPIN_LIMIT) {
            if (have_testdev) scc_dead = 1;
            return;
        }
    }
    WR8(SCC_CHB_DATA, (u8)c);
}

void testdev_probe(void)
{
    /* If the slot is empty this read takes a bus error on real hardware. On
     * IRIS with --test-device it returns the signature; without it, the GIO
     * timeout handler returns something that isn't our magic. */
    have_testdev = (RD32(TESTDEV_SIGNATURE) == TESTDEV_MAGIC);
}

void testdev_dump(u32 tag)
{
    if (have_testdev) WR32(TESTDEV_DUMP, tag);
}

/*
 * Let the SCC finish shifting out what has been handed to it.
 *
 * scc_putc waits for the transmit *buffer* to empty before handing over each
 * byte, which says nothing about the byte already in the shift register. The
 * test device then terminates the machine the instant it is written, so the
 * tail of the last line is simply lost: booted through the PROM, where serial
 * output actually works, "IRIS-CPUTEST-DONE rc=100" arrived as
 * "IRIS-CPUTEST-DONE rc=" — and run/run-prom.sh decides pass or fail by
 * matching on that line.
 *
 * The spin is bounded and runs exactly once, at the end of a run that already
 * took minutes.
 */
void con_flush(void)
{
    int spins = 0;

    /* Nothing was ever handed to the SCC, so there is nothing in flight. */
    if (scc_dead) return;

    while (!(RD8(SCC_CHB_CMD) & SCC_RR0_TX_EMPTY)) {
        if (++spins > TX_SPIN_LIMIT) break;
    }
    for (spins = 0; spins < 20000000; spins++)
        __asm__ __volatile__("" ::: "memory");
}

void testdev_exit(u32 code)
{
    if (have_testdev) WR32(TESTDEV_EXIT, code);
    /* No test device: the host is reading serial, so the DONE line is the
     * result. Spin forever rather than falling off the end of the world. */
    for (;;) { }
}

void con_putc(int c)
{
    if (c == '\n') {
        scc_putc('\r');
        if (have_testdev) WR32(TESTDEV_PUTC, '\r');
    }
    scc_putc(c);
    if (have_testdev) WR32(TESTDEV_PUTC, (u32)(u8)c);
}

void con_puts(const char *s)
{
    while (*s) con_putc(*s++);
}

static const char hexd[] = "0123456789abcdef";

void con_hex32(u32 v)
{
    int i;
    con_puts("0x");
    for (i = 28; i >= 0; i -= 4) con_putc(hexd[(v >> i) & 0xF]);
}

void con_hex64(u64 v)
{
    int i;
    con_puts("0x");
    for (i = 60; i >= 0; i -= 4) con_putc(hexd[(int)((v >> i) & 0xF)]);
}

void con_udec(unsigned long long v)
{
    char buf[24];
    int n = 0;
    if (v == 0) { con_putc('0'); return; }
    while (v) { buf[n++] = (char)('0' + (int)(v % 10)); v /= 10; }
    while (n) con_putc(buf[--n]);
}

void con_dec(long long v)
{
    if (v < 0) { con_putc('-'); con_udec((unsigned long long)(-v)); }
    else con_udec((unsigned long long)v);
}

/* Freestanding varargs: we have no stdarg.h from a libc, but GCC's builtins
 * are always available. */
#define va_list   __builtin_va_list
#define va_start  __builtin_va_start
#define va_arg    __builtin_va_arg
#define va_end    __builtin_va_end

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
        case 'l':
            /* %lx and %ld both mean 64-bit here. */
            fmt++;
            if (*fmt == 'x') con_hex64(va_arg(ap, u64));
            else if (*fmt == 'd') con_dec(va_arg(ap, long long));
            else if (*fmt == 'u') con_udec(va_arg(ap, unsigned long long));
            else { con_putc('%'); con_putc('l'); con_putc(*fmt); }
            break;
        case '%': con_putc('%'); break;
        case '\0': con_putc('%'); return;
        default:  con_putc('%'); con_putc(*fmt); break;
        }
    }
    va_end(ap);
}

void panic(const char *msg)
{
    con_puts("\nPANIC: ");
    con_puts(msg);
    con_puts("\n");
    testdev_exit(127);
    for (;;) { }
}
