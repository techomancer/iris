#!/usr/bin/env bash
# run-local.sh — run the benchmark under IRIS and keep the result.
#
# Loads the ELF straight into RAM (--load-elf): no PROM, no disk, no IRIX, so
# nothing between the kernels and the emulator. The machine block the guest
# prints is the actual output; the human table above it is for watching.
#
# usage: run-local.sh [build/irisbench.elf] [extra iris args...]
set -uo pipefail

ELF="${1:-build/irisbench.elf}"; shift || true
IRIS="${IRIS:-../target/release/iris}"
# A hang detector, not a performance budget. The suite targets ~250 ms per
# timed run over ~45 kernels with 2 repeats plus calibration, so a couple of
# minutes is normal and ten is not.
TIMEOUT="${TIMEOUT:-900}"
LOG="${LOG:-build/bench.log}"

[[ -f "$ELF" ]]  || { echo "run-local: no such binary: $ELF" >&2; exit 2; }
[[ -x "$IRIS" ]] || { echo "run-local: no iris at $IRIS (cargo build --release)" >&2; exit 2; }

mkdir -p "$(dirname "$LOG")"
rm -f "$LOG"

# --test-device is not optional here: without it there is no host clock and no
# retired-instruction count, and every score falls back to CP0 Count at an
# assumed frequency. The suite says so in its header when that happens.
timeout "$TIMEOUT" "$IRIS" \
    --config run/bare.toml \
    --load-elf "$ELF" \
    --test-device \
    --headless --noaudio \
    "$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

if [[ $rc -eq 124 ]]; then
    echo "run-local: TIMED OUT after ${TIMEOUT}s" >&2
    tail -30 "$LOG" >&2
    exit 124
fi

echo "run-local: iris exited rc=$rc   (log: $LOG)"
exit $rc
