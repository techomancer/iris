#!/usr/bin/env bash
# Run every CPU x engine cell (or the cells named) through `make check-run`.
# usage: run/matrix.sh [cell ...]       FORCE_BUILD=1 rebuilds cached binaries
#
# The same guest runs on every cell, so a disagreement between cells is an
# emulator bug: R4400 vs R5000 for the MIPS IV opcodes, interp vs jitv2 as a
# JIT differential. The CPU is a runtime flag; the engine is a cargo feature,
# so each cell's iris is built once and cached as build/iris-<cpu>-<engine>.
set -u
cd "$(dirname "$0")/.."
make --no-print-directory >/dev/null || { echo "guest build failed"; exit 2; }

fails=0
for cell in ${*:-r4400-interp r5000-interp r4400-jitv2 r5000-jitv2}; do
    cpu=${cell%%-*} engine=${cell##*-} bin=build/iris-${cell}
    if [[ ! -x $bin || -n ${FORCE_BUILD:-} ]]; then
        features=; [[ $engine == jitv2 ]] && features="--features jitv2"
        (cd .. && cargo build --release $features) && cp ../target/release/iris "$bin" \
            || { echo "FAIL  $cell (build)"; fails=$((fails + 1)); continue; }
    fi
    if make --no-print-directory check-run CPU="$cpu" ENGINE="$engine" IRIS="$PWD/$bin" \
            > "build/$cell.out" 2>&1; then
        echo "PASS  $cell"
    else
        echo "FAIL  $cell"
        grep -E '^(check|run-local):' "build/$cell.out" | sed 's/^/        /'
        fails=$((fails + 1))
    fi
done
exit $((fails > 0))
