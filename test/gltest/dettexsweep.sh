#!/bin/sh
# dettexsweep.sh: measure how a board combines detail textures, with
# dettex's uniform images (--fill) and custom detail functions (--func).
# Run in the guest from the directory holding dettex; the same script on a
# real board and on the emulator gives comparable dumps.
#
#   sh dettexsweep.sh OUTDIR [fill weight lod cutoff ...]
#
#   fill    base 64/128/192 against detail 0..255, F = 1 and F = 0.5
#   weight  table weight 1..31 (a constant function w/31), detail 0..255
#   lod     F against LOD: ramps over -6..0 and a constant, quads of
#           240..12 pixels on an 8-texel base
#   cutoff  a step at LOD -1 / -0.9 around table index 16 (base 16), and a
#           constant function near index 31 (base 32)
#
# Read the centre pixel of each dump (PPM, rows bottom up): out - base,
# over (4095 - 2185) / 512 levels per unit of table weight at detail 255.
out=${1:?outdir}; shift
modes=${*:-"fill weight lod cutoff"}
mkdir -p $out
D="./dettex --mipmap --detail 256 --impact"
has() { case " $modes " in *" $1 "*) return 0 ;; esac; return 1; }
one() { name=$1; shift; $D "$@" --dump $out/$name > $out/$name.txt 2>&1; }

if has fill; then
    for b in 64 128 192; do
        for d in 0 32 64 96 112 120 124 126 127 128 129 130 132 136 144 160 192 224 255; do
            one fill_b${b}_d$d --fill $b,$d --only add
        done
    done
fi
if has weight; then
    for w in 1 2 3 4 5 6 7 8 10 12 14 15 16 17 20 24 28 30 31; do
        x=`echo "$w / 31" | bc -l`
        for d in 0 64 128 192 255; do
            one weight_w${w}_d$d --fill 128,$d --only func --func -10,$x,0,$x
        done
    done
fi
if has lod; then
    for s in 240 200 160 128 112 96 80 64 56 48 40 32 28 24 20 16 12; do
        one lod_s${s}_down --fill 128,255 --only func --func -6,1,0,0 --size $s
        one lod_s${s}_up --fill 128,255 --only func --func -6,0,0,1 --size $s
        one lod_s${s}_const --fill 128,255 --only func --func -10,1,0,1 --size $s
    done
fi
if has cutoff; then
    for s in 40 36 35 34 33 32; do
        one cut_b16_s${s}_step --base 16 --fill 128,255 --only func --func -1,1,-0.9,0 --size $s
        one cut_b16_s${s}_ramp --base 16 --fill 128,255 --only func --func -2,1,-1,0 --size $s
    done
    for s in 36 34 33 32; do
        one cut_b32_s${s}_const --base 32 --fill 128,255 --only func --func -10,1,0,1 --size $s
    done
fi
echo done > $out/done
