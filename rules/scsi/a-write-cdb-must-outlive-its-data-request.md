# A write CDB must outlive the chip's request for its data

NetBSD/sgimips could not install: `disklabel -w` failed every time with

    sd1(wdsc0:0:2:0): illegal request, data = 00 00 00 00 25 00 00 00
    disklabel: ioctl DIOCWDINFO: Label magic number or checksum is wrong!

Fixed 2026-09-19 in `wd33c93a.rs`. The bug was ours, not NetBSD's.

## Reading that sense line

`scsipi_base.c` prints `sense->csi[n]` — sense bytes 8 onward — so the eight
bytes are 8..15 and **byte 12 is the ASC**: `0x25`, LOGICAL UNIT NOT SUPPORTED.
Not a disk-geometry or label-format complaint at all. `scsi.rs` produces that
one code in exactly one place: a CDB whose `cdb[1] >> 5` is non-zero.

## What was happening

A driver that steps the bus phase by phase issues a write as *two* Transfer
Info commands:

1. the six CDB bytes, after which the chip answers `TRANSFER_DATA_OUT` to say
   "arm your DMA, I want the data";
2. the data — same command, same registers, only the transfer count differs.

We read both as CDBs. The first pass threw its CDB away, and the second
decoded the sector as a command. The sector was an SGI volume header, so:

| byte | value | read as |
|---|---|---|
| 0 | `0x0b` | opcode — unimplemented |
| 1 | `0xe5` | `>> 5` = LUN 7 |

LUN 7 is checked before the opcode, so the disk answered LUN NOT SUPPORTED,
nothing was written, and the label NetBSD read back afterwards was zeroes.

The fix is to hold the CDB in `pending_cdb` across the data-out request and
execute it when the data arrives; `process_scsi_command` pulls the bytes
itself, using the count the driver has by then programmed. A fresh selection
drops any CDB whose data never came.

## Why this went unnoticed

IRIX never takes this path. It writes through `SELECT_ATN_XFER`, which carries
the CDB in registers 0x03–0x0E and runs the whole command without the chip
handing control back between phases — so there is nothing to remember.

## Things that looked like the cause and were not

Hours went into these; none of them is where the bug lived.

- **A missing SGI volume header.** `disklabel_bsd_to_sgimips` always returns 0
  and builds a fresh header, so an unlabelled disk is fine.
- **SCSI writes in general.** `dd` to the raw device appeared to work.
- **`disklabel` itself.** `disklabel -R` from the install shell succeeded.
- **A CDB phase gate.** Tracing showed NetBSD's `XFER_INFO` arrives at phases
  0x10 and 0x20, both of which we accept.

The thing that actually found it was one trace line:

    CMD phase CDB 0x0a count=6 dma=true
    CMD phase CDB 0x0b count=512 dma=true

Two command phases back to back, the second 512 bytes long. A CDB is never
512 bytes. Read the counts, not just the opcodes.

## Regression test

`wd33c93a::data_out_phase_tests::the_transfer_after_a_write_cdb_is_its_data_not_another_command`
replays that exact pair of Transfer Info commands against a file-backed disk
and asserts the sector reaches it. Reverted, it fails with status 0x02 —
CHECK CONDITION, the real symptom.

Related: [SCSI PIO and LUN findings](linux-lun-scan-and-mode-sense-gaps.md) is the
PIO half of the same seam.
