#!/bin/sh
# Lines and points, one case per run, each read back from the front
# buffer into OUT/<case>_front.ppm. Run the same sweep on the emulator
# and on a real board, then compare the images (tools/ppmdiff.py).
#
#   sh linesweep.sh OUTDIR
#
# Window 64x48, black clear, flat colours unless the case says --smooth;
# the default triangle's three edges give a horizontal, a y-major and an
# x-major segment.

out=${1:-/tmp/linesweep}
mkdir -p $out
G="./glprim --size 64x48 --clear 0,0,0 --hold 50 --read front --visual 8,8,8,8"

run() {
    name=$1; shift
    $G --readout $out/$name "$@" > $out/$name.log 2>&1 || echo "$name: failed"
}

for w in 1 1.5 2 2.5 3 4 5 7 10; do
    run lloop_w$w -p lloop --linewidth $w
done
run lloop_smoothshade -p lloop --smooth
run lloop_stipple_1_0f0f -p lloop --linestipple 1,0x0f0f
run lloop_stipple_3_3333 -p lloop --linestipple 3,0x3333
run lloop_stipple_2_0f0f_w3 -p lloop --linestipple 2,0x0f0f --linewidth 3
for w in 1 2 3.5; do
    run lloop_aa_w$w -p lloop --linesmooth --linewidth $w --blendfunc src_alpha,one_minus_src_alpha
done
for s in 1 1.5 2 2.5 3 4 5 8 10; do
    run points_s$s -p points --pointsize $s
done
for s in 1 3 6.5; do
    run points_aa_s$s -p points --pointsmooth --pointsize $s --blendfunc src_alpha,one_minus_src_alpha
done
run polymode_line -p triangles --polymode line
run polymode_point -p triangles --polymode point --pointsize 3
echo done $out
