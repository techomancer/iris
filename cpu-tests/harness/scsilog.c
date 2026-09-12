/* scsilog.c — console mirror to the boot disk, via HPC3 DMA + WD33C93A.
 *
 * Register values here are not invented: they are what the Indy PROM's own
 * driver leaves in the chip after it loads the boot file, read back with the
 * IRIS monitor's `scsi regs` while a PROM-booted run was live. Copying the
 * firmware verbatim is the closest thing to a hardware guarantee available
 * without the hardware — in particular CONTROL, whose DMA-mode field is 001
 * (burst) and not the 100 a datasheet reading would suggest.
 *
 * The buffer and the DMA descriptor are touched only through their KSEG1
 * aliases, so no cache writeback is needed before the engine reads them. That
 * also makes the flush safe after the TLB and cache tests have run: KSEG0 and
 * KSEG1 are unmapped, so nothing here depends on TLB state.
 */

#include "scsilog.h"
#include "testlib.h"   /* dcache_wb_invalidate_range */

#define RD8(a)      (*(volatile u8  *)(unsigned long)(a))
#define WR8(a, v)   (*(volatile u8  *)(unsigned long)(a) = (u8)(v))
#define WR32(a, v)  (*(volatile u32 *)(unsigned long)(a) = (u32)(v))

/* WD33C93A via HPC3 (hpc3.rs: HPC3_BASE + SCSI_REG_BASE, idx = offset >> 2).
 * Byte lane 3 is the low byte of the big-endian word — the lane real hardware
 * decodes, and one IRIS accepts since it masks the low two bits away. */
#define WD_ADDR           0xBFBC0003u
#define WD_DATA           0xBFBC0007u

#define WD_OWN_ID         0x00
#define WD_CONTROL        0x01
#define WD_TIMEOUT        0x02
#define WD_CDB_1          0x03
#define WD_TARGET_LUN     0x0F
#define WD_XFER_MSB       0x12
#define WD_DEST_ID        0x15
#define WD_SCSI_STATUS    0x17
#define WD_COMMAND        0x18

#define ASR_CIP           0x10
#define ASR_INT           0x80

#define CMD_SELECT_ATN_XFER  0x08
#define ST_SELECT_XFER_OK    0x16

/* Exactly what the PROM leaves behind. */
#define OWN_ID_VALUE      0x0Au
#define CONTROL_VALUE     0x2Du
#define TIMEOUT_VALUE     0x40u

/* HPC3 SCSI channel 0 DMA (hpc3.rs: SCSI0_BASE, SCSI_NBDP, SCSI_CTRL). */
#define HPC_NBDP          0xBFB90004u
#define HPC_CTRL          0xBFB91004u
#define HPC_CTRL_DIR      0x04u
#define HPC_CTRL_FLUSH    0x08u
#define HPC_CTRL_ACTIVE   0x10u
#define HPC_CTRL_RESET    0x40u

/* Descriptor word 1 carries the count in bits 0-13 plus these flags. */
#define DESC_EOX          0x80000000u
#define DESC_XIE          0x20000000u

/* K1_PTR forces the sign extension 64-bit mode needs — see iris.h. */
#define K1_U8(p)          (*(volatile u8  *)K1_PTR(p))
#define K1_U32(p)         (*(volatile u32 *)K1_PTR(p))
#define PHYS_OF(p)        PHYS((u32)(unsigned long)(p))

#define SPIN_LIMIT        4000000
#define CHUNK             8192u
#define SECTOR            512u

static u8  logbuf[SCSILOG_MAX] __attribute__((aligned(16)));
static u32 desc[4] __attribute__((aligned(16)));
static u32 loglen = 16;   /* byte 0..15 is the header, stamped at flush */
static int overflow;

/* Called from con_putc for every byte. Written cached and written back before
 * the buffer is handed to a reader — the PROM's Write() reads it through a
 * cached mapping, so uncached stores alone leave stale lines and punch holes in
 * the log. */
void scsilog_tap(int c)
{
    if (loglen >= SCSILOG_MAX - SECTOR) { overflow = 1; return; }
    logbuf[loglen] = (u8)c;
    loglen++;
}

static void wd_put(u8 reg, u8 v)
{
    WR8(WD_ADDR, reg);
    WR8(WD_DATA, v);
}

static u8 wd_get(u8 reg)
{
    WR8(WD_ADDR, reg);
    return RD8(WD_DATA);
}

/* Wait for the command to retire. Bounded: a wedged bus must not hang a run
 * whose whole purpose is to report a result. */
static int wd_wait_int(void)
{
    int spins = 0;
    while ((RD8(WD_ADDR) & ASR_INT) == 0) {
        if (++spins > SPIN_LIMIT) return -1;
    }
    return 0;
}

static int wd_idle(void)
{
    int spins = 0;
    while (RD8(WD_ADDR) & ASR_CIP) {
        if (++spins > SPIN_LIMIT) return -1;
    }
    return 0;
}

/* One WRITE(10) of `bytes` from `phys` to `lba`, DMA pre-armed as the PROM does. */
static int write_chunk(unsigned target, u32 lba, u32 phys, u32 bytes)
{
    u32 blocks = (bytes + SECTOR - 1) / SECTOR;
    u8 st;

    K1_U32(&desc[0]) = phys;
    K1_U32(&desc[1]) = (bytes & 0x3FFFu) | DESC_EOX | DESC_XIE;
    K1_U32(&desc[2]) = 0;

    WR32(HPC_CTRL, HPC_CTRL_RESET);
    WR32(HPC_CTRL, 0);
    WR32(HPC_NBDP, PHYS_OF(&desc[0]));
    WR32(HPC_CTRL, HPC_CTRL_DIR | HPC_CTRL_ACTIVE);

    if (wd_idle() < 0) return -1;

    wd_put(WD_OWN_ID, OWN_ID_VALUE);
    wd_put(WD_CONTROL, CONTROL_VALUE);
    wd_put(WD_TIMEOUT, TIMEOUT_VALUE);
    wd_put(WD_TARGET_LUN, 0);
    /* Bit 6 is the data-phase direction: set for the PROM's READ, clear here. */
    wd_put(WD_DEST_ID, (u8)(target & 0x07u));

    wd_put(WD_CDB_1 + 0, 0x2A);
    wd_put(WD_CDB_1 + 1, 0x00);
    wd_put(WD_CDB_1 + 2, (u8)(lba >> 24));
    wd_put(WD_CDB_1 + 3, (u8)(lba >> 16));
    wd_put(WD_CDB_1 + 4, (u8)(lba >> 8));
    wd_put(WD_CDB_1 + 5, (u8)lba);
    wd_put(WD_CDB_1 + 6, 0x00);
    wd_put(WD_CDB_1 + 7, (u8)(blocks >> 8));
    wd_put(WD_CDB_1 + 8, (u8)blocks);
    wd_put(WD_CDB_1 + 9, 0x00);

    wd_put(WD_XFER_MSB + 0, (u8)(bytes >> 16));
    wd_put(WD_XFER_MSB + 1, (u8)(bytes >> 8));
    wd_put(WD_XFER_MSB + 2, (u8)bytes);

    wd_put(WD_COMMAND, CMD_SELECT_ATN_XFER);

    if (wd_wait_int() < 0) return -2;
    st = wd_get(WD_SCSI_STATUS);

    WR32(HPC_CTRL, HPC_CTRL_FLUSH);
    WR32(HPC_CTRL, 0);

    return st == ST_SELECT_XFER_OK ? 0 : -3;
}

/* Stamp a locatable header so an extractor can find the text and its length. */
static void write_header(void)
{
    static const char magic[8] = { 'I','R','I','S','L','O','G','1' };
    unsigned i;

    for (i = 0; i < 8; i++)
        logbuf[i] = (u8)magic[i];
    logbuf[8]  = (u8)(loglen >> 24);
    logbuf[9]  = (u8)(loglen >> 16);
    logbuf[10] = (u8)(loglen >> 8);
    logbuf[11] = (u8)loglen;
    logbuf[12] = (u8)overflow;
}

/* Stamp the header and hand back the buffer for someone else to write. Length
 * is rounded to a whole sector for a raw device. */
const void *scsilog_prepare(unsigned *len)
{
    write_header();
    *len = (loglen + SECTOR - 1) & ~(SECTOR - 1);
    dcache_wb_invalidate_range(logbuf, *len);
    return (const void *)logbuf;
}

int scsilog_flush(unsigned target)
{
    u32 off, total;
    int rc;

    write_header();
    dcache_wb_invalidate_range(logbuf, (loglen + SECTOR - 1) & ~(SECTOR - 1));

    total = loglen;
    for (off = 0; off < total; off += CHUNK) {
        u32 n = (total - off) > CHUNK ? CHUNK : (total - off);
        /* DMA length must equal the WRITE(10) length, so round to whole
         * sectors: a short descriptor leaves the transfer never completing. */
        n = (n + SECTOR - 1) & ~(SECTOR - 1);
        rc = write_chunk(target, SCSILOG_LBA + off / SECTOR,
                         PHYS_OF(&logbuf[off]), n);
        if (rc != 0) return rc;
    }
    return 0;
}
