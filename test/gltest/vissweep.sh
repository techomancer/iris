#!/bin/sh
# Sweep glprim over the colour formats IMPACT is sold with, single and
# double buffered, window and pbuffer, and the depth/stencil layouts.
# Each run draws --scene quadrants and checks the readbacks (--check).
#
# Usage: sh vissweep.sh [glprim options...]
#   GLPRIM=path    glprim binary (default ./glprim)
#   COLORS="..."   colour formats (default below)
#   ZS="..."       depth/stencil layouts (default below)
#   VERBOSE=1      print every run's full output
#
# Results: PASS, FAIL (failing checks follow), none (no such
# visual/fbconfig), or ERR (anything else; output follows).

G=${GLPRIM:-./glprim}
COLORS=${COLORS:-"3,3,2 4,4,4 4,4,4,4 8,8,8 8,8,8,8 10,10,10,2 12,12,12 12,12,12,12"}
ZS=${ZS:-"none s8z24 z32 z24 s4z20 z16 s8z16"}

$G --listvisuals
echo

pass=0; fail=0; none=0; err=0
for c in $COLORS; do
    for db in sb db; do
        for tgt in window pbuffer; do
            for zs in $ZS; do
                reads=front
                [ $db = db ] && reads=front,back
                case $zs in none) ;; *z*) reads=$reads,depth ;; esac
                case $zs in *s*) reads=$reads,stencil ;; esac
                opts="-p none --scene quadrants --visual $c --zs $zs --read $reads --check --hold 0"
                [ $db = db ] && opts="$opts --db"
                [ $tgt = pbuffer ] && opts="$opts --pbuffer"
                out=`$G $opts "$@" 2>&1`
                case "$out" in
                    *"check PASS"*) r=PASS; pass=`expr $pass + 1` ;;
                    *"check FAIL"*) r=FAIL; fail=`expr $fail + 1` ;;
                    *"no matching"*|*"pbuffers need"*) r=none; none=`expr $none + 1` ;;
                    *) r=ERR; err=`expr $err + 1` ;;
                esac
                printf "%-12s %-3s %-8s %-6s %s\n" $c $db $tgt $zs $r
                if [ -n "$VERBOSE" ]; then
                    echo "$out" | sed 's/^/    /'
                elif [ $r = FAIL ]; then
                    echo "$out" | egrep "config|FAIL" | sed 's/^/    /'
                elif [ $r = ERR ]; then
                    echo "$out" | sed 's/^/    /'
                fi
            done
        done
    done
done
echo
echo "pass $pass fail $fail none $none err $err"
