#!/bin/bash
# scripts/vpro-goldens.sh: the VPro (Odyssey) regression nets.
#
# Run from the repo root:  scripts/vpro-goldens.sh COMMAND
#
#   check            draw-level frames and vertex numbers, bit-exact (cargo test)
#   tolerance        the same, passing a frame that moved inside its stored limits
#   bless            re-take the frames, hashes and vertex numbers
#   replay           replay the recorded guest sessions to their checkpoints
#                    (fails when there are no recordings)
#   replay-bless     re-take the recordings' checkpoints
#
# VPRO_RUN is a prefix for the build container when cargo is not on the host
# (for example VPRO_RUN="/path/to/icargo"); it is put before `env`, so the
# variables below reach the tests inside it. Paths given to the tests must be
# visible there.
#
# The README beside the data, src/dev/odyssey/testdata/README.md, says when
# to re-take each net and how to judge a change by its tolerance.

set -euo pipefail
cd "$(dirname "$0")/.."

run=${VPRO_RUN:-}
cargo_test() { # ENV... -- FILTER [TEST-HARNESS-ARGS...]
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    local filter=$1
    shift
    # shellcheck disable=SC2086
    $run env "${envs[@]}" cargo test --release --lib "$filter" -- "$@"
}

cmd=${1:-check}
case $cmd in
check)
    cargo_test -- odyssey::golden
    ;;
tolerance)
    cargo_test VPRO_GOLDEN=tolerance -- odyssey::golden
    ;;
bless)
    cargo_test VPRO_GOLDEN_BLESS=1 -- odyssey::golden::bless --ignored --nocapture
    ;;
replay)
    cargo_test VPRO_REPLAY_REQUIRE=1 -- odyssey::golden::replay::recorded_sessions --nocapture
    ;;
replay-bless)
    cargo_test VPRO_REPLAY_BLESS=1 -- odyssey::golden::replay::replay_bless --ignored --nocapture
    ;;
*)
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
