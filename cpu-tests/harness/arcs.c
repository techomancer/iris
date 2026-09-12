/* arcs.c — write the console through the PROM's ARCS firmware vector.
 *
 * Addresses here were read out of a live PROM-booted machine with the IRIS
 * monitor rather than taken from a specification:
 *
 *   physical 0x1000   SPB, signature 'ARCS' 0x53435241, length 0x48
 *   SPB + 0x1c        FirmwareVectorLength, 0x8c = 35 entries
 *   SPB + 0x20        FirmwareVector       (observed 0xa0001800)
 *   vector entry 27   Write(FileId, Buffer, Length, &Count)
 *
 * The 35-entry length is itself the cross-check that the layout is the ARC
 * one, and SPB + 0x10 (DebugBlock) reads 0 exactly as the IRIX source in
 * src/debug.md expects when symmon is absent.
 *
 * exc_install() writes three trampolines and flushes 0x200 bytes from
 * 0x80000000, so it never reaches the SPB at 0x1000 or the vector table at
 * 0x1800 — the firmware stays callable for the whole run, after the TLB and
 * cache tests have done their worst. Everything here is read through KSEG1 and
 * the vectors live in KSEG0, so none of it depends on TLB state.
 */

#include "arcs.h"
#include "cp0.h"

#define SPB_PHYS          0x1000u
#define SPB_SIGNATURE     0x53435241u
#define SPB_O_VECLEN      0x1c
#define SPB_O_VECTOR      0x20
#define ARCS_VEC_ENTRIES  35
#define ARCS_OPEN         23
#define ARCS_CLOSE        24
#define ARCS_WRITE        27
#define ARCS_SEEK         28
#define ARCS_STDOUT       1
#define ARCS_OPEN_RW      2      /* OpenReadWrite */
#define ARCS_SEEK_ABS     0

/* Sign-extend before dereferencing: a bare u32 KSEG1 address zero-extends into
 * TLB-mapped xkuseg in 64-bit mode — see iris.h. */
#define RD32_K1(a)        (*(volatile u32 *)SEXT_PTR(K0_TO_K1((u32)(a))))

typedef long (*arcs_fn1)(long);
typedef long (*arcs_fn3)(long, long, long);
typedef long (*arcs_fn4)(long, long, long, long);

static u32 vec_base;
static u32 write_vector;
static char line[256];
static unsigned line_len;

int arcs_probe(void)
{
    u32 spb = K0_TO_K1(SPB_PHYS);
    u32 veclen, vector;

    write_vector = 0;
    if (RD32_K1(spb) != SPB_SIGNATURE) return 0;

    veclen = RD32_K1(spb + SPB_O_VECLEN);
    vector = RD32_K1(spb + SPB_O_VECTOR);
    if (veclen < ARCS_VEC_ENTRIES * 4u || vector == 0) return 0;

    vec_base = vector;
    write_vector = RD32_K1(vector + ARCS_WRITE * 4u);
    /* Every real entry points into PROM space; anything else means the layout
     * is not what we think it is, and calling it would be a jump into nothing. */
    if ((write_vector & 0xFFF00000u) != 0x9FC00000u) {
        write_vector = 0;
        vec_base = 0;
        return 0;
    }
    return 1;
}

/* One vector entry, range-checked: a bad layout would be a jump into nothing on
 * a machine with no console to report it. */
static u32 arcs_vec(unsigned n)
{
    u32 v;

    if (!vec_base) return 0;
    v = RD32_K1(vec_base + n * 4u);
    return (v & 0xFFF00000u) == 0x9FC00000u ? v : 0;
}

#define SEXT(p)  ((long)(s32)(u32)(unsigned long)(p))

/*
 * Write through the firmware's own disk driver rather than harness/scsilog.c.
 *
 * The PROM negotiated synchronous SCSI with the drive when it read the boot
 * file; IRIS models no such thing (`SYNC_TRANSFER` is a bare constant and the
 * SDTR bytes are discarded), so a hand-written driver proven under the emulator
 * has only ever run asynchronously. This path inherits whatever the firmware
 * actually does, which is by construction what the machine supports.
 *
 * Seek takes its offset as a 64-bit value *by pointer*, not as an immediate.
 */
int arcs_disk_write(const char *path, u64 offset, const void *buf, u32 len)
{
    arcs_fn3 f_open;
    arcs_fn3 f_seek;
    arcs_fn4 f_write;
    arcs_fn1 f_close;
    u32 status;
    long fid = 0, count = 0, r;
    u64 off = offset;

    f_open  = (arcs_fn3)(long)(s32)arcs_vec(ARCS_OPEN);
    f_seek  = (arcs_fn3)(long)(s32)arcs_vec(ARCS_SEEK);
    f_write = (arcs_fn4)(long)(s32)arcs_vec(ARCS_WRITE);
    f_close = (arcs_fn1)(long)(s32)arcs_vec(ARCS_CLOSE);
    if (!f_open || !f_seek || !f_write || !f_close) return -1;

    status = cp0_status();

    r = f_open(SEXT(path), ARCS_OPEN_RW, SEXT(&fid));
    if (r != 0) { cp0_status_set(status); return -2; }

    r = f_seek(fid, SEXT(&off), ARCS_SEEK_ABS);
    if (r != 0) { f_close(fid); cp0_status_set(status); return -3; }

    r = f_write(fid, SEXT(buf), (long)len, SEXT(&count));
    f_close(fid);
    cp0_status_set(status);

    if (r != 0) return -4;
    return (u32)count == len ? 0 : -5;
}

void arcs_flush(void)
{
    arcs_fn4 fn;
    long count = 0;
    u32 status, fc = 0;

    if (!write_vector || line_len == 0) return;

    /* The firmware clobbers FP state, and the FP tests print diagnostics while
     * a test is still in flight — without this the suite reports five extra
     * failed checks that are the console's fault, not the CPU's. */
    status = cp0_status();
    if (status & ST_CU1) fc = fcsr();

    fn = (arcs_fn4)(long)(s32)write_vector;
    fn(ARCS_STDOUT, (long)(s32)(u32)(unsigned long)line, (long)line_len,
       (long)(s32)(u32)(unsigned long)&count);

    cp0_status_set(status);
    if (status & ST_CU1) fcsr_set(fc);
    line_len = 0;
}

/* Buffered per line: one firmware call per character would make a 19,000-byte
 * run painfully slow, and the PROM expects a counted buffer anyway. */
void arcs_putc(int c)
{
    if (!write_vector) return;

    if (c == '\n' && line_len < sizeof(line) - 1)
        line[line_len++] = '\r';
    line[line_len++] = (char)c;

    if (c == '\n' || line_len >= sizeof(line) - 2)
        arcs_flush();
}
