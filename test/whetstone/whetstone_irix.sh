#!/bin/sh
# Build the C Whetstone (netlib whetstone.c, unmodified) on IRIX 6.5 with
# MIPSpro cc. Copy this file and whetstone.c onto the IRIX guest, then run it
# there. -s keeps the guest C library's build paths out of the binary. The
# source's header permits copying when the whole comment block is kept.
set -e
cc -O2 -n32 -mips3 -s whetstone.c -lm -o whetstone.n32
echo "built: ./whetstone.n32"
