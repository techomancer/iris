#!/bin/sh
# Restart the X server in each root visual class and depth, log in again
# (autologin), and capture: the desktop through scrsave, xdpyinfo, and a GL
# quadrants check and a smooth triangle read back through GL and scrsave.
#   sh xsweep.sh TAG [modes...]   modes: tc24 pc8 pc12 tc12 tc15 tc30 dc24
# Results in /usr/tmp/res-TAG/x-MODE, copied to /shared/results/TAG.
TAG=${1:?tag}; shift
MODES=${*:-"pc8 pc12 tc12 tc15 tc30 dc24 tc24"}
R=/usr/tmp/res-$TAG; S=/shared/results/$TAG
X=/var/X11/xdm/Xservers
DISPLAY=:0; export DISPLAY
mkdir -p $R $S
[ -f $X.orig ] || cp $X $X.orig
G=/usr/tmp/gltest/glprim
[ -x $G ] || { rm -rf /usr/tmp/gltest; cp -r /shared/gltest /usr/tmp/gltest; }
stamp() { echo "`date '+%H:%M:%S'` $*" >> $R/progress; cp -r $R/* $S/ 2>/dev/null; }
base=":0 secure /usr/bin/X11/X -bs -nobitscale -c -solidroot sgilightblue -cursorFG red -cursorBG white"
for m in $MODES; do
    case $m in
        tc24) o="-depth 24 -class TrueColor" ;;
        pc8)  o="-depth 8 -class PseudoColor" ;;
        pc12) o="-depth 12 -class PseudoColor" ;;
        tc12) o="-depth 12 -class TrueColor" ;;
        tc15) o="-depth 15 -class TrueColor" ;;
        tc30) o="-depth 30 -class TrueColor" ;;
        dc24) o="-depth 24 -class DirectColor" ;;
        *) echo "unknown mode $m"; continue ;;
    esac
    D=$R/x-$m; mkdir -p $D
    grep -v "^:0" $X.orig > $X; echo "$base $o" >> $X
    touch /etc/autologin.on
    killall Xsgi; sleep 10
    n=0
    while [ $n -lt 60 ] && ! ps -ef | grep -v grep | grep -q toolchest; do sleep 5; n=`expr $n + 1`; done
    if [ $n -ge 60 ]; then stamp "$m: no desktop"; continue; fi
    sleep 25
    xset s off; xset s noblank
    /usr/bin/X11/xdpyinfo > $D/xdpyinfo.txt 2>&1
    /usr/sbin/scrsave $D/desktop.rgb
    $G -p none --scene quadrants --check --read front --readout $D/quadrants --hold 0 > $D/quadrants.txt 2>&1
    $G -p triangles --smooth --size 256x256 --read front --readout $D/tri --hold 4000 > $D/tri.txt 2>&1 &
    sleep 3; /usr/sbin/scrsave $D/tri.rgb; wait
    stamp "$m: `grep -c FAIL $D/quadrants.txt` quadrant failures, root `grep 'depth of root' $D/xdpyinfo.txt`"
done
# Back to the server as it was.
cp $X.orig $X; touch /etc/autologin.on; killall Xsgi
stamp done
