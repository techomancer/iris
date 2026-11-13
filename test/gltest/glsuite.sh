#!/bin/sh
# Every GL check suite in the guest, for comparing two runs (two raster
# units, two builds). Expects test/gltest copied to /shared/gltest and the
# piglit irix tree to /shared/piglit (TESTING.md).
#   sh glsuite.sh TAG [suites...]   suites: state pix vis tex lines scenes piglit
# Results in /usr/tmp/res-TAG, copied to /shared/results/TAG as each
# suite finishes; progress in its "progress" file.
TAG=${1:?tag}; shift
SUITES=${*:-"state pix vis tex lines scenes piglit"}
R=/usr/tmp/res-$TAG
S=/shared/results/$TAG
DISPLAY=:0; export DISPLAY
mkdir -p $R $S
rm -rf /usr/tmp/gltest; cp -r /shared/gltest /usr/tmp/gltest
cd /usr/tmp/gltest
xset s off 2>/dev/null; xset s noblank 2>/dev/null
out() { cp -r $R/* $S/ 2>/dev/null; }
stamp() { echo "`date '+%H:%M:%S'` $*" >> $R/progress; out; }
has() { case " $SUITES " in *" $1 "*) return 0 ;; esac; return 1; }

stamp start $SUITES
if has state; then
    ./glstate > $R/glstate.txt 2>&1
    ./blendtest > $R/blendtest.txt 2>&1
    ./seltest > $R/seltest.txt 2>&1
    ./cicopy > $R/cicopy.txt 2>&1
    stamp state
fi
if has pix; then
    ./pixtest > $R/pixtest.txt 2>&1
    ./pixtest --rgba8888 > $R/pixtest-rgba8888.txt 2>&1
    ./pixtest --ci 8 > $R/pixtest-ci8.txt 2>&1
    ./pixtest --ci 12 > $R/pixtest-ci12.txt 2>&1
    stamp pix
fi
if has vis; then
    sh vissweep.sh > $R/vissweep.txt 2>&1
    stamp vis
fi
if has tex; then
    HOLD=300 sh texsweep.sh > $R/texsweep.txt 2>&1
    stamp tex
fi
if has lines; then
    sh linesweep.sh $R/lines > $R/linesweep.txt 2>&1
    stamp lines
fi
if has scenes; then
    # Each scene read back through GL (--readout) and, while held, the
    # whole screen through X (scrsave): two readback paths of one image.
    mkdir -p $R/scenes
    for p in points lines lloop lstrip triangles tstrip tfan quads qstrip polygon; do
        ./glprim -p $p --smooth --size 128x128 --read front --readout $R/scenes/prim_$p --hold 0 > $R/scenes/prim_$p.log 2>&1
    done
    for sc in depth stencil alphatest blend lit litlocal litrgb twoside fog litcmat litnorm dlist clipplane fogexp fogexp2 blendsmooth quadrants copycolor copydepth tex texmod; do
        ./glprim -p none --scene $sc --size 256x256 --read front --readout $R/scenes/$sc --hold 4000 > $R/scenes/$sc.log 2>&1 &
        sleep 3
        /usr/sbin/scrsave $R/scenes/$sc.rgb
        wait
    done
    stamp scenes
fi
if has piglit; then
    rm -rf /usr/tmp/piglit; cp -r /shared/piglit /usr/tmp/piglit
    cd /usr/tmp/piglit/irix && ./piglit-run.sh -t 300 -o $R/piglit lists/vpro.list > $R/piglit.log 2>&1
    stamp piglit
fi
stamp done
