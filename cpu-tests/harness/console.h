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
/* The R4600 is the one part here that no oracle run has covered: its
 * expectations are inferred, and docs/r4600.md says from what. */
#define CPU_R4600   0x4
/* The R10000 family: R10000, R12000 and R14000, one kind because they share
 * the R10000 User's Manual's programming model. Nothing in this suite has run
 * on one: no Indy ever carried an R10000, so the kind exists only under IRIS,
 * and its expectations come from the manuals. docs/r10000.md says which. */
#define CPU_R10000  0x8
#define CPU_ALL     (CPU_R4400 | CPU_R5000 | CPU_R4600 | CPU_R10000)

extern u32 cpu_kind;          /* one CPU_* bit, set at startup */
extern u32 cpu_prid;
extern u32 cpu_fir;
extern u32 cpu_config;
static inline int is_r5000(void) { return cpu_kind == CPU_R5000; }
static inline int is_r4400(void) { return cpu_kind == CPU_R4400; }
static inline int is_r4600(void) { return cpu_kind == CPU_R4600; }
/* Any of the R10000 family. */
static inline int is_r10000(void) { return cpu_kind == CPU_R10000; }
/* An R12000 or R14000 rather than an R10000: PRId implementation 0x0E or 0x0F.
 * Only CVT.L's 51-bit limit, which the R10000's manual alone states, tells
 * them apart here. */
static inline int is_r12000_up(void)
{
    return is_r10000() && ((cpu_prid >> 8) & 0xFF) != 0x09;
}
/* MIPS IV is the R5000's and the R10000's: the R4400 and the R4600 are both
 * MIPS III parts and must refuse every MIPS IV encoding. */
static inline int has_mips4(void) { return cpu_kind & (CPU_R5000 | CPU_R10000); }

/* ── Console ──────────────────────────────────────────────────────────────── */
void con_init(void);
void con_putc(int c);
/* Optional second sink for every console byte; null unless a harness sets it. */
extern void (*con_tap)(int c);
void con_disable_scc(void);
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
