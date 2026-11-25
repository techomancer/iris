#!/bin/sh
# The XZ / Extreme comparison run: every GL suite (piglit on express.list,
# vissweep over XZ's visuals), then the X server sweep through XZ's root
# visual classes. Same script on the emulated and the real Indy.
#   nohup sh run-xz.sh TAG > /usr/tmp/run-TAG.log 2>&1 &
TAG=${1:?tag}
cd /shared/gltest || exit 1
PIGLIT_LIST=express.list
COLORS="1,2,1 3,3,2 4,4,4 8,8,8"
ZS="none z24 s4z20"
export PIGLIT_LIST COLORS ZS
sh glsuite.sh $TAG
sh xsweep.sh $TAG pc8 pc12 tc12 tc24
