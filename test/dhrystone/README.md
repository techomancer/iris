# Dhrystone 2.1

`dhry.h`, `dhry_1.c` and `dhry_2.c` are Reinhold P. Weicker's version 2.1 of
25 May 1988, unmodified from <https://www.netlib.org/benchmark/dhry-c>. They
carry no licence text and netlib states no redistribution terms, so their status
here is unconfirmed.

`dhrystone.n32` is an IRIX 6.5 N32 MIPS-III build made by `dhrystone_irix.sh`
(MIPSpro cc 7.4.4m). `dhrystone_linux.sh` builds a host `dhrystone` to compare
with. The program times itself with `time(2)` in whole seconds and refuses to
report under about 2 s, so give it at least 10M runs.

## Build flags

The build is `cc -O2 -n32 -mips3 -s -DTIME`. `-DTIME` selects the `time(2)`
clock. The source's default, `-DTIMES`, calls `times()` and would need an edit
on IRIX: `dhry_1.c` declares `times()` itself and `dhry.h` needs `HZ` defined.
The compiler's message for it was not kept. `-s` keeps the guest C library's
build paths out of the binary.

`-O2` matches a prebuilt N32 MIPS-III Dhrystone built with MIPSpro 7.44 that
Dominik Behr shared. The `-O2` link without `-s` has the same `.text` bytes as
that binary, and every other section the same size except `.debug_line` and
`.comment`, which record the build directory. `-O2 -r10000` gives the same
`.text` too; `-O2 -g3` gives one of the same size with different bytes, and
`-O0`, `-O1`, `-O3`, `-Ofast` and `-mips4` give different code.

Timed against the prebuilt binary in an emulated Indy (R4400, IRIX 6.5.22m),
10M runs, the two alternated over seven rounds with the host wall time around
each run, the medians were 25.49 s for this binary and 25.43 s for the prebuilt
one on a busy host, and 20.59 s against 20.33 s for the unstripped build on a
quieter one: within 1.3%, and inside each binary's own spread.
