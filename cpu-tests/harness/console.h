/* console.h — base types, serial console, and the IRIS test device.
 *
 * Split out of testlib.h so that code which is not a *test* can use the
 * console without dragging in the CHECK macros and the exception-record
 * plumbing: bench/ compiles this same console.c against its own harness.
 * testlib.h includes this, so every existing test file is unaffected.
 */
#ifndef CONSOLE_H
#define CONSOLE_H

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


/* Halt with a message — unrecoverable harness failure. */
void panic(const char *msg) __attribute__((noreturn));

#endif /* CONSOLE_H */
