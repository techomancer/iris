/* hostshim.h — what the host build substitutes for the machine.
 *
 * The kernels and the runner are compiled a second time, natively, and produce
 * the same output block from the same code. Two things come out of that:
 *
 *   - the golden checksums the guest is scored against, computed by an
 *     independent IEEE-754 implementation rather than by a previous run of the
 *     emulator (a checksum recorded from IRIS would agree with IRIS by
 *     construction, including everywhere IRIS is wrong);
 *   - a baseline for the machine the emulator is running on, measured by
 *     literally the same kernels with the same autoscaler and the same
 *     best-of-two, so "IRIS delivers 1/85th of native on this kernel" is a
 *     real ratio and not two benchmarks pretending to be comparable.
 *
 * Only the `sys` group is left out: a TLB refill has no host analogue.
 */
#ifndef HOSTSHIM_H
#define HOSTSHIM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned char       u8;
typedef unsigned short      u16;
typedef unsigned int        u32;
typedef unsigned long long  u64;
typedef signed char         s8;
typedef short               s16;
typedef int                 s32;
typedef long long           s64;

/* The guest console API, on stdout. Buffered rather than per-character: the
 * SCC costs the guest a bus transaction per byte and the host should not
 * pretend to pay it. */
static inline void con_putc(int c) { putchar(c); if (c == '\n') fflush(stdout); }
static inline void con_puts(const char *s) { fputs(s, stdout); }
void con_printf(const char *fmt, ...);
void con_hex32(u32 v);
void con_hex64(u64 v);
void con_udec(unsigned long long v);
void con_dec(long long v);
static inline void con_init(void) { }
static inline void con_flush(void) { fflush(stdout); }

extern int have_testdev;
void testdev_probe(void);
__attribute__((noreturn)) void testdev_exit(u32 code);
__attribute__((noreturn)) void panic(const char *msg);

#endif /* HOSTSHIM_H */
