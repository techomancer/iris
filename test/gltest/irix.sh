#!/bin/sh
# Build the GL tests on IRIX with MIPSpro cc (stock IRIX has no bash).
# Usage: sh irix.sh [target...]   targets: gltest glprim pixtest blendtest glstate xgetpix xresize xclick cicopy (default: all)

targets="$*"
[ -z "$targets" ] && targets="gltest glprim pixtest blendtest glstate xgetpix xresize xclick cicopy"

status=0
for t in $targets; do
    case $t in
        gltest) src=main.c; libs="-lX11 -lGL -lm" ;;
        glprim) src=glprim.c; libs="-lX11 -lGL -lm" ;;
        pixtest) src=pixtest.c; libs="-lX11 -lGL -lm" ;;
        blendtest) src=blendtest.c; libs="-lX11 -lGL" ;;
        glstate) src=glstate.c; libs="-lX11 -lGL -lm" ;;
        xgetpix) src=xgetpix.c; libs="-lX11" ;;
        xresize) src=xresize.c; libs="-lX11" ;;
        xclick) src=xclick.c; libs="-lXtst -lXext -lX11" ;;
        cicopy) src=cicopy.c; libs="-lX11 -lGL" ;;
        *) echo "unknown target $t"; exit 2 ;;
    esac
    echo "cc -o $t $src"
    cc -o $t $src -mips3 -n32 $libs || status=1
done
[ $status -eq 0 ] && echo "Done. Run ./gltest or ./glprim --help"
exit $status
