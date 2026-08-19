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
# The whole suite is a few minutes of emulated time on a quiet machine and
# rather more on a busy one; the FP trap tests each take a real exception. This
# is a hang detector, not a performance budget, so keep it generous.
TIMEOUT="${TIMEOUT:-600}"
LOG="${LOG:-build/serial.log}"
DUMP="${DUMP:-build/dump.json}"

[[ -f "$ELF" ]]  || { echo "run-local: no such binary: $ELF" >&2; exit 2; }
[[ -x "$IRIS" ]] || { echo "run-local: no iris at $IRIS (cargo build --release)" >&2; exit 2; }

mkdir -p "$(dirname "$LOG")"
rm -f "$LOG" "$DUMP"

# --headless: no window. --noaudio: no cpal device in CI.
# --test-device: PUTC/DUMP/EXIT, and the exit code we propagate.
# --config run/bare.toml: a bare-metal binary touches no disk, and bare.toml's
# present-but-empty [scsi] keeps SCSI out. Without it the default config
# attaches scsi1.raw and startup is fatal when that file is absent, which is
# always — nothing in this suite creates one.
#
# The log is a tee of stdout, not --serial-log: headless IRIS already writes
# the guest's console to its own stdout, and --serial-log tees the channel-B
# *backend*, which in non-CI mode is the TCP listener nothing is attached to —
# so it produced an empty file and the diagnostics below had nothing to show.
# matrix.sh and the CI workflow capture stdout for the same reason.
timeout "$TIMEOUT" "$IRIS" \
    --config run/bare.toml \
    --load-elf "$ELF" \
    --test-device --test-device-dump "$DUMP" \
    --headless --noaudio \
    "$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

if [[ $rc -eq 124 ]]; then
    echo "run-local: TIMED OUT after ${TIMEOUT}s" >&2
    [[ -f "$LOG" ]] && { echo "--- serial tail ---" >&2; tail -30 "$LOG" >&2; }
    exit 124
fi

echo "run-local: iris exited rc=$rc   (serial: $LOG)"
exit $rc
