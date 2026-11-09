#!/bin/sh
# Build Dhrystone 2.1 (netlib dhry-c, unmodified) on IRIX 6.5 with MIPSpro cc.
# Copy this file, dhry.h, dhry_1.c and dhry_2.c onto the IRIX guest, then run
# it there. -DTIME selects the time(2) clock (the default, -DTIMES, needs an
# edit on IRIX); -s keeps the guest C library's build paths out of the binary.
set -e
cc -O2 -n32 -mips3 -s -DTIME dhry_1.c dhry_2.c -o dhrystone.n32
echo "built: ./dhrystone.n32"
