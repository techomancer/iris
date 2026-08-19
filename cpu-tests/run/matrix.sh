#!/usr/bin/env bash
# matrix.sh — run the suite across CPU and execution-engine combinations.
#
# The point of the matrix is differential testing. The same guest binary runs
# on every cell, so any disagreement between cells is an emulator bug rather
# than a test bug:
#
#   R4400 vs R5000   the MIPS IV opcodes must compute on one and raise
#                    Reserved Instruction on the other
#   interp vs jitv2  a guest-visible ISA suite is the cleanest possible JIT
#                    differential test. rules/jit/verify-mode.md notes that
#                    verify mode is structurally invalid for blocks containing
#                    stores, so this covers ground it cannot.
#
# Each cell needs its own IRIS build, because the CPU is a compile-time cargo
# feature (see rules/perf/hardware-profiles.md). Builds are cached in
# build/iris-<cpu>/ so a re-run only relinks what changed.
#
# usage: run/matrix.sh [cell ...]      (default: all cells)
#        CELLS="r4400-interp r5000-interp" run/matrix.sh
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=..
OUT=build/matrix
mkdir -p "$OUT"

ALL_CELLS="r4400-interp r5000-interp r4400-jitv2 r5000-jitv2"
CELLS="${*:-${CELLS:-$ALL_CELLS}}"

feature_for() {
    case "$1" in
        r4400) echo "" ;;
        # r5ksc_triton gives the R5000 its on-die secondary cache, which is
        # what an Indy R5000 actually has. Without it Config.SC reports no L2
        # and the cache tests would be exercising a machine that never shipped.
        r5000) echo "r5k,r5ksc_triton" ;;
    esac
}

build_iris() {
    local cpu="$1" engine="$2" feats target
    feats="$(feature_for "$cpu")"
    # jitv2 needs no runtime switch — it is active as soon as it is compiled in.
    [[ "$engine" == "jitv2" ]] && feats="${feats:+$feats,}jitv2"
    target="build/iris-$cpu-$engine"
    if [[ -x "$target" ]] && [[ -z "${FORCE_BUILD:-}" ]]; then
        echo "  (reusing $target)"
        return 0
    fi
    echo "  building iris for $cpu ${feats:+--features $feats}"
    if [[ -n "$feats" ]]; then
        (cd "$ROOT" && cargo build --release --features "$feats") || return 1
    else
        (cd "$ROOT" && cargo build --release) || return 1
    fi
    cp "$ROOT/target/release/iris" "$target"
}

run_cell() {
    local cell="$1" cpu engine iris log rc
    cpu="${cell%%-*}"
    engine="${cell##*-}"
    iris="build/iris-$cpu-$engine"
    log="$OUT/$cell.log"

    build_iris "$cpu" "$engine" || { echo "FAIL  $cell (build)"; return 1; }

    timeout "${TIMEOUT:-1200}" "$iris" \
        --config run/bare.toml \
        --load-elf build/cputest.elf \
        --test-device --test-device-dump "$OUT/$cell.dump.json" \
        --headless --noaudio > "$log" 2>&1
    rc=$?

    if [[ $rc -eq 124 ]]; then
        echo "FAIL  $cell — TIMED OUT"
        return 1
    fi

    # Confirm the binary that ran is actually the CPU this cell names. Earlier
    # in this suite's history an --features r5k build overwrote
    # target/release/iris between the copy and the run, and an "R4400" cell
    # silently exercised an R5000 — every mips4 expectation inverted, with no
    # indication anything was wrong. The banner is authoritative because the
    # guest reads it out of PRId.
    local want_cpu banner
    want_cpu="$(echo "$cpu" | tr 'a-z' 'A-Z')"
    banner="$(grep -o 'cpu=[A-Za-z0-9]*' "$log" | head -1)"
    if [[ "$banner" != "cpu=$want_cpu" ]]; then
        echo "FAIL  $cell — binary reports $banner, expected cpu=$want_cpu"
        return 1
    fi

    local summary
    summary="$(grep -E '^ RESULT:' "$log" | head -1)"
    if [[ $rc -eq 0 ]]; then
        echo "PASS  $cell   ${summary:-(no summary line)}"
    else
        echo "FAIL  $cell   rc=$rc   ${summary:-(no summary line)}"
        grep -E '^\S+ \.+ FAIL' "$log" | sed 's/^/        /' | head -20
    fi
    return $rc
}

echo "== cpu-tests matrix =="
make --no-print-directory >/dev/null || { echo "guest build failed"; exit 2; }

fails=0
for cell in $CELLS; do
    echo "-- $cell"
    run_cell "$cell" || fails=$((fails + 1))
done

echo
if [[ $fails -eq 0 ]]; then
    echo "matrix: all cells passed"
else
    echo "matrix: $fails cell(s) failed — logs in $OUT/"
fi
exit $((fails > 0))
