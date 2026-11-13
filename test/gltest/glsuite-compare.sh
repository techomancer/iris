#!/bin/bash
# Compare two result sets from glsuite.sh / xsweep.sh (host side):
#   test/gltest/glsuite-compare.sh shared/results/cpu shared/results/gpu
A=${1:?reference}; B=${2:?candidate}
sum() { tail -1 "$1" 2>/dev/null; }
echo "== state and pixel suites (summary lines; failing cases only in one side listed)"
for f in glstate blendtest seltest cicopy pixtest pixtest-rgba8888 pixtest-ci8 pixtest-ci12; do
    a=$(sum $A/$f.txt); b=$(sum $B/$f.txt)
    [ "$a" = "$b" ] && echo "  same   $f: $a" || echo "  DIFFER $f: [$a] vs [$b]"
    diff <(grep -E "^FAIL" $A/$f.txt | cut -d: -f1) <(grep -E "^FAIL" $B/$f.txt | cut -d: -f1) | grep "^[<>]" | sed 's/^</    only in reference:/; s/^>/    only in candidate:/'
done
echo "== visual sweep"
a=$(sum $A/vissweep.txt); b=$(sum $B/vissweep.txt)
[ "$a" = "$b" ] && echo "  same: $a" || echo "  DIFFER: [$a] vs [$b]"
diff <(grep -E "^[0-9,]+ +[sd]b" $A/vissweep.txt) <(grep -E "^[0-9,]+ +[sd]b" $B/vissweep.txt) | grep "^[<>]" | sed 's/^/    /'
echo "== texture sweep"
diff $A/texsweep.txt $B/texsweep.txt > /dev/null && echo "  same" || diff $A/texsweep.txt $B/texsweep.txt | grep "^[<>]" | sed 's/^/    /'
echo "== lines and points (GL readbacks)"
n=0; d=0; for f in $A/lines/*.ppm; do n=$((n+1)); cmp -s $f $B/lines/$(basename $f) || { d=$((d+1)); echo "    differs: $(basename $f)"; }; done; echo "  $d of $n differ"
echo "== scenes (GL readbacks, and the screen through scrsave cropped to the window)"
for f in $A/scenes/*.ppm; do n=$(basename $f); cmp -s $f $B/scenes/$n || echo "    readback differs: $n $(compare -metric AE $f $B/scenes/$n null: 2>&1)"; done
for f in $A/scenes/*.rgb; do
    n=$(basename $f .rgb); o=$(grep -o "origin [0-9]*,[0-9]*" $A/scenes/$n.log | head -1 | tr -dc '0-9,')
    [ -n "$o" ] || continue; x=${o%,*}; y=${o#*,}
    c=$(compare -metric AE <(magick $f -crop 256x256+$x+$y +repage ppm:-) <(magick $B/scenes/$n.rgb -crop 256x256+$x+$y +repage ppm:-) null: 2>&1)
    [ "${c%% *}" = "0" ] || echo "    screen differs: $n $c"
done
echo "== X server modes"
for d in $A/x-*; do m=$(basename $d)
    q="$(grep -c FAIL $d/quadrants.txt 2>/dev/null)/$(grep -c FAIL $B/$m/quadrants.txt 2>/dev/null) quadrant failures"
    t=$(cmp -s $d/tri_front.ppm $B/$m/tri_front.ppm && echo "tri readback same" || echo "TRI READBACK DIFFERS")
    s=$(compare -metric AE $d/desktop.rgb $B/$m/desktop.rgb null: 2>&1)
    echo "  $m: $q, $t, desktop pixels differing $s"
done
echo "== piglit"
if [ -f $A/piglit/results ] && [ -f $B/piglit/results ]; then
    echo "  reference: $(cut -f1 $A/piglit/results | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
    echo "  candidate: $(cut -f1 $B/piglit/results | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
    join -t $'\t' -1 2 -2 2 <(sort -t $'\t' -k2 $A/piglit/results) <(sort -t $'\t' -k2 $B/piglit/results) | awk -F'\t' '$2 != $3 {print "    " $1 ": " $2 " -> " $3}'
fi
