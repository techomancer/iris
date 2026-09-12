#!/usr/bin/env python3
"""extract-log.py — recover a hardware run's console log from a BlueSCSI image.

The suite mirrors its console into fixed LBAs past the ELF (harness/scsilog.c),
because an Indy on a graphics console has no reader for SCC channel B. After a
hardware run, put the SD card in a host and point this at HD2_512.img.

    run/extract-log.py /media/you/BLUESCSI/HD2_512.img -o cputest-hw-r4400.log
"""
import argparse
import struct
import sys

LBA = 8192
SECTOR = 512
MAGIC = b"IRISLOG1"
HEADER = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o", "--output")
    ap.add_argument("--lba", type=int, default=LBA)
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        f.seek(args.lba * SECTOR)
        head = f.read(HEADER)
        if head[:8] != MAGIC:
            sys.exit(
                "no %s at LBA %d — the run never reached scsilog_flush(), or the "
                "image is not the one that was booted" % (MAGIC.decode(), args.lba)
            )
        total = struct.unpack(">I", head[8:12])[0]
        overflow = head[12]
        f.seek(args.lba * SECTOR)
        text = f.read(total)[HEADER:]

    if overflow:
        print("warning: log overflowed SCSILOG_MAX; output is truncated", file=sys.stderr)

    if args.output:
        with open(args.output, "wb") as out:
            out.write(text)
        print("wrote %s (%d bytes)" % (args.output, len(text)))
    else:
        sys.stdout.write(text.decode("ascii", "replace"))


if __name__ == "__main__":
    main()
