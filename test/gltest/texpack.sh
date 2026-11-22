#!/bin/sh
# texpack.sh: run in the guest (cd to glprim's directory first). Every way
# glprim makes the library upload a texture: each sized internal format and
# each unsized source format, small (FIFO image) and 256x256 (DMA), with
# and without mipmap levels, plus ABGR, 16-bit sources, a sub-image and
# the gluBuild2DMipmaps scene. On VPro (iris built with --features
# vprodiag), compare the board model's packing
# check before and after (`vpro status`, "texture packing"): each level
# descriptor is checked against the upload that wrote its memory.
H=${HOLD:-200}
MIP=linear_mipmap_linear,linear
run() { echo "$*"; ./glprim -p none --scene tex --hold $H "$@" > /dev/null 2>&1; }
for f in alpha4 alpha8 alpha12 alpha16 luminance4 luminance8 luminance12 luminance16 \
         luminance4_alpha4 luminance6_alpha2 luminance8_alpha8 luminance12_alpha4 \
         luminance12_alpha12 luminance16_alpha16 intensity4 intensity8 intensity12 \
         intensity16 r3_g3_b2 rgb4 rgb5 rgb8 rgb10 rgb12 rgb16 rgba2 rgba4 rgb5_a1 \
         rgba8 rgb10_a2 rgba12 rgba16; do
    for n in 8 256; do
        run --texifmt $f --texsize $n
        run --texifmt $f --texsize $n --texfilter $MIP
    done
done
for f in rgba rgb luminance luminance_alpha alpha intensity; do
    for n in 8 256; do
        run --texfmt $f --texsize $n
        run --texfmt $f --texsize $n --texfilter $MIP
        run --texfmt $f --texsize $n --texushort
    done
done
for n in 8 256; do
    run --texabgr --texsize $n
    run --texsize $n --texsubrect 2,2,4,4
done
for f in rgba rgb luminance luminance_alpha alpha intensity; do
    echo "texmod $f"; ./glprim -p none --scene texmod --texfmt $f --hold $H > /dev/null 2>&1
done
