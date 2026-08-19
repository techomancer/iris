#!/usr/bin/env bash
# run-local.sh — run a suite binary under IRIS and report the result.
#
# Loads the ELF directly (--load-elf), so no PROM, no disk image, no IRIX. The
# suite's own exit code comes back through the test device; the serial log is
# kept for inspection either way.
#
# usage: run-local.sh [build/cputest.elf] [extra iris args...]
set -uo pipefail

ELF="${1:-build/cputest.elf}"; shift || true
IRIS="${IRIS:-../target/release/iris}"
TIMEOUT="${TIMEOUT:-120}"
LOG="${LOG:-build/serial.log}"
DUMP="${DUMP:-build/dump.json}"

[[ -f "$ELF" ]]  || { echo "run-local: no such binary: $ELF" >&2; exit 2; }
[[ -x "$IRIS" ]] || { echo "run-local: no iris at $IRIS (cargo build --release)" >&2; exit 2; }

mkdir -p "$(dirname "$LOG")"
rm -f "$LOG" "$DUMP"

# --headless: no window. --noaudio: no cpal device in CI.
# --test-device: PUTC/DUMP/EXIT, and the exit code we propagate.
# A bare-metal binary touches no disk, so an empty --config keeps SCSI out.
timeout "$TIMEOUT" "$IRIS" \
    --load-elf "$ELF" \
    --test-device --test-device-dump "$DUMP" \
    --headless --noaudio \
    --serial-log "$LOG" \
    "$@"
rc=$?

if [[ $rc -eq 124 ]]; then
    echo "run-local: TIMED OUT after ${TIMEOUT}s" >&2
    [[ -f "$LOG" ]] && { echo "--- serial tail ---" >&2; tail -30 "$LOG" >&2; }
    exit 124
fi

echo "run-local: iris exited rc=$rc   (serial: $LOG)"
exit $rc
