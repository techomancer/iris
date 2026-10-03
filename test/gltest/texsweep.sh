#!/bin/sh
# texsweep.sh: run in the guest (cd to glprim's directory first). Every
# sized internal texture format through glprim --scene tex --texread (load,
# draw, glGetTexImage), then the wrap-mode / border cases, the sub-image
# and screen-copy cases. Prints glprim's texread / error lines; watch the
# window, or take screenshots during --hold, for the images.
H=${HOLD:-1500}
for f in alpha4 alpha8 alpha12 alpha16 luminance4 luminance8 luminance12 luminance16 \
         luminance4_alpha4 luminance6_alpha2 luminance8_alpha8 luminance12_alpha4 \
         luminance12_alpha12 luminance16_alpha16 intensity4 intensity8 intensity12 \
         intensity16 r3_g3_b2 rgb4 rgb5 rgb8 rgb10 rgb12 rgb16 rgba2 rgba4 rgb5_a1 \
         rgba8 rgb10_a2 rgba12 rgba16; do
    echo "$f: `./glprim -p none --scene tex --texifmt $f --texread --hold $H 2>&1 | egrep 'texread|error'`"
done
for w in repeat,repeat clamp,repeat repeat,clamp clamp,clamp border,border clamp,border; do
    echo "wrap $w"; ./glprim -p none --scene tex --texwrap $w --texfilter linear,linear --hold $H >/dev/null 2>&1
    echo "wrap $w, border"; ./glprim -p none --scene tex --texwrap $w --texborder --texfilter linear,linear --hold $H >/dev/null 2>&1
done
echo "border colour"; ./glprim -p none --scene tex --texwrap clamp --texbcolor 1,1,0,1 --texfilter linear,linear --hold $H >/dev/null 2>&1
echo "sub-image: `./glprim -p none --scene tex --texsize 64 --texsubrect 32,8,32,8 --texread --hold $H 2>&1 | egrep 'texsub|texread'`"
echo "copy full: `./glprim -p none --scene tex --texcopy full --hold $H 2>&1 | egrep 'texcopy'`"
echo "copy sub: `./glprim -p none --scene tex --texsize 64 --texcopy sub --hold $H 2>&1 | egrep 'texcopy'`"
echo "copy pixels"; ./glprim -p none --scene copycolor --smooth --hold $H >/dev/null 2>&1
echo "copy depth"; ./glprim -p none --scene copydepth --smooth --hold $H >/dev/null 2>&1
