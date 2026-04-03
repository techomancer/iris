#!/bin/bash
# JIT diagnostic launcher — runs emulator and captures output for analysis
# Usage: ./jit-diag.sh [mode]
#   mode: "jit"     — JIT enabled (default)
#         "verify"  — JIT with verification
#         "nojit"   — interpreter only through JIT dispatch
#         "interp"  — pure interpreter (no JIT feature, baseline)

MODE="${1:-jit}"
OUTFILE="jit-diag-$(date +%Y%m%d-%H%M%S)-${MODE}.log"

echo "=== IRIS JIT Diagnostic ===" | tee "$OUTFILE"
echo "Mode: $MODE" | tee -a "$OUTFILE"
echo "Date: $(date)" | tee -a "$OUTFILE"
echo "Host: $(uname -m) $(uname -s) $(uname -r)" | tee -a "$OUTFILE"
echo "Rust: $(rustc --version)" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

case "$MODE" in
  jit)
    echo "Running: IRIS_JIT=1 cargo run --release --features jit,lightning" | tee -a "$OUTFILE"
    IRIS_JIT=1 cargo run --release --features jit,lightning 2>&1 | tee -a "$OUTFILE"
    ;;
  verify)
    echo "Running: IRIS_JIT=1 IRIS_JIT_VERIFY=1 cargo run --release --features jit,lightning" | tee -a "$OUTFILE"
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
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [jit|verify|nojit|interp]"
    exit 1
    ;;
esac

echo "" >> "$OUTFILE"
echo "=== Exit code: $? ===" >> "$OUTFILE"
echo "Output saved to: $OUTFILE"
