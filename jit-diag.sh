#!/bin/bash
# JIT diagnostic launcher — runs emulator and captures output for analysis
# Usage: ./jit-diag.sh [mode]
#   mode: "jit"      — JIT enabled (default)
#         "verify"   — JIT with verification
#         "nojit"    — interpreter only through JIT dispatch
#         "interp"   — pure interpreter (no JIT feature, baseline)
#         "perf"     — perf profile, interpreter only (text report for analysis)
#         "perf-jit" — perf profile with JIT enabled
#
# All IRIS_JIT_* env vars are passed through automatically:
#   IRIS_JIT_MAX_TIER=0 ./jit-diag.sh jit
#   IRIS_JIT_PROBE=500 IRIS_JIT_PROBE_MIN=100 ./jit-diag.sh jit

MODE="${1:-jit}"
OUTFILE="jit-diag-$(date +%Y%m%d-%H%M%S)-${MODE}.log"

# Collect all IRIS_JIT_* env vars for display and passthrough
JIT_VARS=$(env | grep '^IRIS_JIT_' | tr '\n' ' ')

echo "=== IRIS JIT Diagnostic ===" | tee "$OUTFILE"
echo "Mode: $MODE" | tee -a "$OUTFILE"
echo "Date: $(date)" | tee -a "$OUTFILE"
echo "Host: $(uname -m) $(uname -s) $(uname -r)" | tee -a "$OUTFILE"
echo "Rust: $(rustc --version)" | tee -a "$OUTFILE"
[ -n "$JIT_VARS" ] && echo "Env: $JIT_VARS" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

case "$MODE" in
  jit)
    echo "Running: IRIS_JIT=1 ${JIT_VARS}cargo run --release --features jit,lightning" | tee -a "$OUTFILE"
    IRIS_JIT=1 cargo run --release --features jit,lightning 2>&1 | tee -a "$OUTFILE"
    ;;
  verify)
    echo "Running: IRIS_JIT=1 IRIS_JIT_VERIFY=1 ${JIT_VARS}cargo run --release --features jit,lightning" | tee -a "$OUTFILE"
    IRIS_JIT=1 IRIS_JIT_VERIFY=1 cargo run --release --features jit,lightning 2>&1 | tee -a "$OUTFILE"
    ;;
  nojit)
    echo "Running: cargo run --release --features jit,lightning (no IRIS_JIT)" | tee -a "$OUTFILE"
    cargo run --release --features jit,lightning 2>&1 | tee -a "$OUTFILE"
    ;;
  interp)
    echo "Running: cargo run --release --features lightning (no jit feature)" | tee -a "$OUTFILE"
    cargo run --release --features lightning 2>&1 | tee -a "$OUTFILE"
    ;;
  perf)
    PERFREPORT="perf-report-$(date +%Y%m%d-%H%M%S).txt"
    echo "Building (profiling profile, no jit feature)..." | tee -a "$OUTFILE"
    cargo build --profile profiling --features lightning 2>&1 | tee -a "$OUTFILE"
    echo "--- Press Ctrl-C when you have enough samples ---"
    perf record -F 99 --call-graph dwarf -o perf.data -- ./target/profiling/iris
    echo "Processing perf data..." | tee -a "$OUTFILE"
    perf report --stdio --no-children -i perf.data > "$PERFREPORT" 2>&1
    echo "Perf report saved to: $PERFREPORT"
    ;;
  perf-jit)
    PERFREPORT="perf-report-jit-$(date +%Y%m%d-%H%M%S).txt"
    echo "Building (profiling profile, jit feature)..." | tee -a "$OUTFILE"
    cargo build --profile profiling --features jit,lightning 2>&1 | tee -a "$OUTFILE"
    echo "--- Press Ctrl-C when you have enough samples ---"
    IRIS_JIT=1 perf record -F 99 --call-graph dwarf -o perf.data -- ./target/profiling/iris
    echo "Processing perf data..." | tee -a "$OUTFILE"
    perf report --stdio --no-children -i perf.data > "$PERFREPORT" 2>&1
    echo "Perf report saved to: $PERFREPORT"
    ;;
  smoke)
    # Headless boot smoke test: boots IRIX with JIT, checks milestones, exits.
    # Uses COW overlay to protect disk image. Exits 0 if all milestones pass.
    TIMEOUT="${IRIS_SMOKE_TIMEOUT:-120}"
    echo "Running: headless smoke test (timeout=${TIMEOUT}s)" | tee -a "$OUTFILE"

    # Clean up stale overlays
    rm -f scsi1.raw.overlay scsi2.raw.overlay

    # Build
    cargo build --release --features jit,lightning 2>&1 | tee -a "$OUTFILE"

    # Run headless with JIT, capture output, kill after timeout
    IRIS_JIT=1 timeout "$TIMEOUT" cargo run --release --features jit,lightning -- --headless 2>&1 | tee -a "$OUTFILE" &
    EMUPID=$!

    # Wait for emulator to finish or timeout
    wait $EMUPID 2>/dev/null
    EXIT=$?

    echo "" | tee -a "$OUTFILE"
    echo "=== Smoke Test Results ===" | tee -a "$OUTFILE"

    PASS=0
    FAIL=0

    check_milestone() {
      local name="$1"
      local pattern="$2"
      if grep -q "$pattern" "$OUTFILE"; then
        echo "  PASS: $name" | tee -a "$OUTFILE"
        PASS=$((PASS + 1))
      else
        echo "  FAIL: $name (pattern '$pattern' not found)" | tee -a "$OUTFILE"
        FAIL=$((FAIL + 1))
      fi
    }

    check_milestone "JIT initialized"    "JIT: adaptive mode"
    check_milestone "First compilation"  "JIT: compiled #1"
    check_milestone "Blocks compiled"    "JIT:.*blocks,"

    # Check for crashes
    if grep -qiE "KERNEL FAULT|PANIC|TLBMISS.*KERNEL|SEGV" "$OUTFILE"; then
      echo "  FAIL: kernel panic detected" | tee -a "$OUTFILE"
      FAIL=$((FAIL + 1))
    else
      echo "  PASS: no kernel panic" | tee -a "$OUTFILE"
      PASS=$((PASS + 1))
    fi

    # Check instruction count reached a reasonable level
    LAST_TOTAL=$(grep -oP 'JIT: \K[0-9]+(?= total)' "$OUTFILE" | tail -1)
    if [ -n "$LAST_TOTAL" ] && [ "$LAST_TOTAL" -gt 100000000 ] 2>/dev/null; then
      echo "  PASS: reached ${LAST_TOTAL} instructions" | tee -a "$OUTFILE"
      PASS=$((PASS + 1))
    else
      echo "  FAIL: instruction count too low (${LAST_TOTAL:-0})" | tee -a "$OUTFILE"
      FAIL=$((FAIL + 1))
    fi

    echo "" | tee -a "$OUTFILE"
    echo "Score: $PASS passed, $FAIL failed" | tee -a "$OUTFILE"

    # Clean up overlay
    rm -f scsi1.raw.overlay scsi2.raw.overlay

    if [ "$FAIL" -gt 0 ]; then
      exit 1
    fi
    exit 0
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [jit|verify|nojit|interp|perf|perf-jit|smoke]"
    exit 1
    ;;
esac

echo "" >> "$OUTFILE"
echo "=== Exit code: $? ===" >> "$OUTFILE"
echo "Output saved to: $OUTFILE"
