#!/bin/sh
# host.sh SOCKET CAPTURE OUT [PATTERN]: run the audio sweep in a booted guest.
#
# The guest is an Octane or an O2 with a MIPSpro compiler, logged in or
# loginable, with iris started with the control socket SOCKET and the capture
# file in its environment: IRIS_AUDIO_DUMP=CAPTURE. This copies the test in,
# builds it, plays each case from sweep.sh (only those matching the egrep
# PATTERN, if given), and writes OUT/marks.txt. Then run
# check.py CAPTURE OUT/marks.txt (check.py --o2 for the O2's dump).
# SKIP_BUILD=1 uses an audiotest already in /usr/tmp/at, for a guest without a
# compiler.
#
# iris-ci put reads the file in iris's own filesystem. When iris runs somewhere
# whose paths differ from this shell's, set PUT to a command run as
# "$PUT HOSTFILE GUESTPATH".
set -e
SOCK=$1 CAP=$2 OUT=$3 PATTERN=${4:-.}
D=$(cd "$(dirname "$0")" && pwd)
ci() { iris-ci --socket "$SOCK" "$@"; }
put() { if [ -n "$PUT" ]; then $PUT "$1" "$2"; else ci put -q --to "$2" "$1"; fi; }
mkdir -p "$OUT"
tar cf "$OUT/audiotest.tar" -C "$D" audiotest.c irix.sh sweep.sh
ci login >/dev/null
put "$OUT/audiotest.tar" /usr/tmp/audiotest.tar
if [ -n "$SKIP_BUILD" ]; then
    ci run 'mkdir -p /usr/tmp/at; cd /usr/tmp/at; tar xf ../audiotest.tar' >/dev/null
else
    ci run --timeout 300 'rm -rf /usr/tmp/at; mkdir /usr/tmp/at; cd /usr/tmp/at; tar xf ../audiotest.tar; sh irix.sh' >/dev/null
fi
# iris creates the capture when the card first plays, so it may not exist yet.
# A WAV's marks skip its 44-byte header; a dump has none.
size() {
    if [ ! -f "$CAP" ]; then echo 0
    elif [ "$(head -c 4 "$CAP" | tr -d '\000')" = RIFF ]; then echo $(($(stat -c %s "$CAP") - 44))
    else stat -c %s "$CAP"; fi
}
: > "$OUT/marks.txt"
for c in $(ci run 'cd /usr/tmp/at; sh sweep.sh cases' | grep -E '^[a-z]+_' | tr -d '\r' | grep -E "$PATTERN"); do
    ci run --timeout 60 "cd /usr/tmp/at; sh sweep.sh prep $c" >/dev/null
    a=$(size)
    ci run --timeout 60 "cd /usr/tmp/at; sh sweep.sh play $c" >/dev/null
    echo "$c $a $(size)" >> "$OUT/marks.txt"
done
