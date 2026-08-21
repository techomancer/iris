#!/usr/bin/env bash
# run-prom.sh — boot the benchmark the way real hardware would: through the
# PROM, off a disk image whose volume header names the binary.
#
# --load-elf is faster and is what day-to-day work uses, but only this proves
# the image is genuinely bootable — the volume-header layout, the ELF the PROM
# will accept, and the load address. It is also the path a real Indy takes, so
# an image built this way can be burned to a CD and run on the hardware the
# emulator is imitating, which is the only way to get a reference number that
# is not itself an emulator's opinion.
#
# usage: run/run-prom.sh [scsi-id] [bootfile]
set -uo pipefail

cd "$(dirname "$0")/.."

ID="${1:-2}"
NAME="${2:-irisbench}"
IRIS="${IRIS:-../target/release/iris}"
CI="${CI:-../target/release/iris-ci}"
SOCK="/tmp/iris-bench-prom.sock"
LOG="build/prom.log"

[[ -x "$IRIS" ]] || { echo "run-prom: no iris at $IRIS" >&2; exit 2; }
[[ -x "$CI" ]]   || { echo "run-prom: no iris-ci at $CI" >&2; exit 2; }
[[ -f build/irisbench.img ]] || { echo "run-prom: make image first" >&2; exit 2; }

cat > build/prom.toml <<'TOML'
# The image goes on SCSI ID 2 so nothing mistakes it for a system disk and so
# `boot -f dksc(0,2,8)irisbench` reads naturally.
banks = [128, 128, 0, 0]
headless = true
no_audio = true

[machine]
profile = "indy_ip24"

[scsi]

[scsi.2]
path  = "build/irisbench.img"
cdrom = false
TOML

rm -f "$SOCK" "$LOG"
"$IRIS" --config build/prom.toml --ci --ci-socket "$SOCK" \
        --test-device --headless --noaudio --serial-log "$LOG" \
        > build/prom-stdout.log 2>&1 &
IRIS_PID=$!
trap 'kill $IRIS_PID 2>/dev/null' EXIT

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

if ci serial-wait "IRIS-BENCH-DONE" --timeout "${TIMEOUT:-2400}" >/dev/null 2>&1; then
    # serial-wait returns on the token, several characters short of the end of
    # the line — at this baud rate the rc digits are still in flight. Wait for
    # the whole line before deciding anything, the same race cpu-tests'
    # run-prom.sh documents.
    for _ in $(seq 1 50); do
        grep -qE "IRIS-BENCH-DONE rc=[0-9]+" "$LOG" && break
        sleep 0.2
    done
    grep -E "accuracy|emulator speed|IRIS-BENCH-DONE" "$LOG" | tail -4
    grep -q "IRIS-BENCH-DONE rc=0" "$LOG" && { echo "run-prom: PASS"; exit 0; }
    echo "run-prom: suite reported checksum mismatches"; exit 1
fi

echo "run-prom: never reached the DONE token; last serial output:" >&2
tail -40 "$LOG" >&2
exit 1
