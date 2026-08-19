#!/usr/bin/env bash
# run-prom.sh — boot the suite the way real hardware would: through the PROM,
# from a disk image whose volume header names the binary.
#
# This is the path the bootable CD will use. `--load-elf` is faster for
# day-to-day work, but only this proves the image is genuinely bootable —
# the volume-header layout, the ELF the PROM will accept, and the load address.
#
# Drives the PROM over the CI serial socket:
#   1. interrupt the power-on countdown
#   2. enter the command monitor (maintenance menu option 5)
#   3. boot -f dksc(0,<id>,8)<name>
#
# usage: run/run-prom.sh [scsi-id] [bootfile]
set -uo pipefail

cd "$(dirname "$0")/.."

ID="${1:-2}"
NAME="${2:-cputest}"
# The per-cell binaries matrix.sh caches are named <cpu>-<engine>; this default
# tracks that naming. Override IRIS= to boot any other cell through the PROM.
IRIS="${IRIS:-build/iris-r4400-interp}"
CI="${CI:-../target/release/iris-ci}"
SOCK="/tmp/iris-cputest-prom.sock"
LOG="build/prom.log"

[[ -x "$IRIS" ]] || { echo "run-prom: no iris at $IRIS" >&2; exit 2; }
[[ -x "$CI" ]]   || { echo "run-prom: no iris-ci at $CI" >&2; exit 2; }
[[ -f build/cputest.img ]] || { echo "run-prom: make image first" >&2; exit 2; }

rm -f "$SOCK" "$LOG"
"$IRIS" --config run/boot.toml --ci --ci-socket "$SOCK" \
        --test-device --headless --noaudio --serial-log "$LOG" \
        > build/prom-stdout.log 2>&1 &
IRIS_PID=$!
trap 'kill $IRIS_PID 2>/dev/null' EXIT

# Wait for the control socket.
for _ in $(seq 1 60); do [[ -S "$SOCK" ]] && break; sleep 1; done
[[ -S "$SOCK" ]] || { echo "run-prom: socket never appeared" >&2; exit 2; }

ci() { "$CI" --socket "$SOCK" "$@"; }

ci start >/dev/null 2>&1 || true

# The PROM counts down before auto-booting; ESC interrupts it and lands in the
# maintenance menu, where 5 is "Enter Command Monitor".
echo "run-prom: waiting for the PROM banner"
ci serial-wait "System Maintenance" --timeout 120 >/dev/null 2>&1 \
  || ci serial-send --no-cr $'\e' >/dev/null 2>&1
sleep 2
ci serial-send --no-cr $'\e' >/dev/null 2>&1
ci serial-wait "Option" --timeout 60 >/dev/null 2>&1 || true
ci serial-send "5" >/dev/null 2>&1
ci serial-wait ">>" --timeout 60 >/dev/null 2>&1 || true

echo "run-prom: boot -f dksc(0,$ID,8)$NAME"
ci serial-send "boot -f dksc(0,$ID,8)$NAME" >/dev/null 2>&1

if ci serial-wait "IRIS-CPUTEST-DONE" --timeout 300 >/dev/null 2>&1; then
    # serial-wait returns on the token, which is several characters short of the
    # end of the line — at the PROM's baud rate the rc digits are still in
    # flight. Deciding anything from the log at this instant reads
    # "IRIS-CPUTEST-DONE rc=" and calls a green run a failure. Wait for the
    # whole line. (The guest's side of the same race is con_flush(), which stops
    # it from halting the machine before the SCC has drained at all.)
    for _ in $(seq 1 50); do
        grep -qE "IRIS-CPUTEST-DONE rc=[0-9]+" "$LOG" && break
        sleep 0.2
    done
    grep -E "RESULT:|IRIS-CPUTEST-DONE" "$LOG" | tail -3
    grep -q "IRIS-CPUTEST-DONE rc=0" "$LOG" && { echo "run-prom: PASS"; exit 0; }
    echo "run-prom: suite reported failures"; exit 1
fi

echo "run-prom: never reached the DONE token; last serial output:" >&2
tail -40 "$LOG" >&2
exit 1
