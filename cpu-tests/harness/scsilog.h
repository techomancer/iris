/* scsilog.h — mirror the console to a reserved region of the boot disk.
 *
 * Real hardware with a graphics console has no reader for SCC channel B, so a
 * PROM-booted run on an Indy produces no visible output at all. This tees every
 * console byte into RAM and writes it to fixed LBAs past the ELF, where it can
 * be recovered from the BlueSCSI image file with dd.
 */
#ifndef SCSILOG_H
#define SCSILOG_H

#include "console.h"

/* First LBA of the log region. The ELF occupies blocks 0..1252 of a 16 MB
 * image (mkvh dump), so this is clear of it with room to spare. */
#define SCSILOG_LBA     8192u
#define SCSILOG_MAX     (64u * 1024u)

/* The image is attached at SCSI ID 2 (run/boot.toml), matching boot -f dksc(0,2,8). */
#define SCSILOG_TARGET  2u

/* Partition 10 is PT_VOLUME, the whole disk; partition 8 (PT_VOLHDR) is only
 * 1265 blocks and does not reach SCSILOG_LBA. */
#define SCSILOG_ARCS_PATH "dksc(0,2,10)"

void scsilog_tap(int c);
int  scsilog_flush(unsigned target);
const void *scsilog_prepare(unsigned *len);

#endif /* SCSILOG_H */
