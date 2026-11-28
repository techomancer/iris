#!/usr/bin/env bash
# compare-after-debugger-read.sh — issue 302, end to end.
#
# Boots build/e2e/compare_after_debugger_read.elf bare-metal, waits for it
# to read Count and spin, runs the monitor's `cop0` dump (mode `cop0`) or
# nothing (mode `none`, the control), releases the guest, and reports
# whether its already-missed Compare deadline raised IP7. Exit 0 = IP7
# raised (correct), 1 = not raised (the bug), anything else = harness error.
#
# usage: run/compare-after-debugger-read.sh [cop0|none] [extra iris args...]
#   IRIS=../target/release/iris  MON_PORT=8888  TIMEOUT=60
set -uo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-cop0}"; shift || true
IRIS="${IRIS:-../target/release/iris}"
ELF="${ELF:-build/e2e/compare_after_debugger_read.elf}"
MON_PORT="${MON_PORT:-8888}"
TIMEOUT="${TIMEOUT:-60}"
LOG="${LOG:-build/e2e/compare-after-debugger-read-$MODE.log}"
MON="python3 ../tools/mon.py"

case "$MODE" in cop0|none) ;; *) echo "mode must be cop0 or none" >&2; exit 2;; esac
[[ -f "$ELF" ]]  || { echo "no such binary: $ELF (make e2e)" >&2; exit 2; }
[[ -x "$IRIS" ]] || { echo "no iris at $IRIS" >&2; exit 2; }
mkdir -p "$(dirname "$LOG")"; rm -f "$LOG"

"$IRIS" --config run/bare.toml --load-elf "$ELF" --test-device \
    --headless --noaudio --monitor-port "$MON_PORT" "$@" >"$LOG" 2>&1 &
pid=$!

fail() { echo "harness: $*" >&2; tail -20 "$LOG" >&2; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; exit 3; }

go=""
for _ in $(seq 1 $((TIMEOUT * 10))); do
    go=$(sed -n 's/.*READY go=0x\([0-9a-fA-F]*\).*/\1/p' "$LOG" | head -1)
    [[ -n "$go" ]] && break
    kill -0 "$pid" 2>/dev/null || fail "iris exited before READY"
    sleep 0.1
done
[[ -n "$go" ]] || fail "no READY within ${TIMEOUT}s"

export IRIS_MON_PORT="$MON_PORT"
if [[ "$MODE" == cop0 ]]; then
    $MON "cop0" | grep -i count || fail "monitor cop0 failed"
fi
# mon.py exits 0 whatever the command said, so check the reply.
# The guest may exit at once after this store, so read nothing back.
out=$($MON "mw 0x$go 1 w") || fail "monitor unreachable"
grep -qi "error\|could not" <<<"$out" && fail "monitor: $out"

for _ in $(seq 1 $((TIMEOUT * 10))); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
done
kill -0 "$pid" 2>/dev/null && fail "guest did not finish within ${TIMEOUT}s"
wait "$pid"; rc=$?

grep -E '^(read=|PASS|FAIL)' "$LOG"
echo "mode=$MODE rc=$rc (log: $LOG)"
exit "$rc"
