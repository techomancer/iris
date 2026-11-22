#!/bin/sh
# End-to-end check of the offline (distribution) build, CHD support included.
#
# Vendors the dependencies (the only step that uses the network), then builds
# iris and chd-extract with the network denied, a fresh CARGO_HOME that knows
# only the vendor directory, an empty download cache and a fresh target
# directory, as a distribution build system would. Finally the offline-built
# chd-extract unpacks a CHD made by MAME's chdman and the result is compared
# with the original image.
#
# Usage: tools/offline-build-test.sh [--expect-download-failure] [--gui] [--keep] [WORKDIR]
#
#   --expect-download-failure  first build without LIBCHDMAN_FORCE_SOURCE and
#                              require it to fail at libchdman-rs's prebuilt
#                              download, then build as documented
#   --gui                      also check iris-gui (lightning,bundled) offline
#   --keep                     keep WORKDIR (vendor tree, logs, target)
#
# Environment:
#   JOBS            cargo -j (default 4)
#   BUILD_WRAPPER   command prefix for each cargo build, e.g. "nice -n 10"
#
# The network is denied with sandbox-exec on macOS and an unshared network
# namespace (unshare -rn) on Linux. Without either the script stops, because
# CARGO_NET_OFFLINE alone does not stop a build script from downloading.
# The CHD round trip needs chdman (MAME tools) and is skipped without it.
# See FEATURES.md, "Offline and distribution builds".

set -eu

expect_fail=0 gui=0 keep=0 work=""
for a in "$@"; do
    case "$a" in
        --expect-download-failure) expect_fail=1 ;;
        --gui) gui=1 ;;
        --keep) keep=1 ;;
        -h|--help) sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown option $a" >&2; exit 2 ;;
        *) work="$a" ;;
    esac
done

repo=$(cd "$(dirname "$0")/.." && pwd)
jobs=${JOBS:-4}
wrap=${BUILD_WRAPPER:-}
[ -n "$work" ] || work=$(mktemp -d "${TMPDIR:-/tmp}/iris-offline.XXXXXX")
mkdir -p "$work"
work=$(cd "$work" && pwd)
cleanup() { [ "$keep" = 1 ] || rm -rf "$work"; }
trap cleanup EXIT

say() { printf '== %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; keep=1; exit 1; }

# A runner that denies the network to everything it starts.
case "$(uname -s)" in
    Darwin)
        cat > "$work/nonet.sb" <<'EOF'
(version 1)
(allow default)
(deny network-outbound (remote ip))
(deny network-outbound (remote unix-socket (path-literal "/private/var/run/mDNSResponder")))
EOF
        nonet() { sandbox-exec -f "$work/nonet.sb" "$@"; } ;;
    Linux)
        unshare -rn true 2>/dev/null || fail "unshare -rn is not permitted here; cannot deny the network"
        nonet() { unshare -rn "$@"; } ;;
    *) fail "no way to deny the network on $(uname -s)" ;;
esac
if command -v curl >/dev/null 2>&1; then
    curl -s -m 10 -o /dev/null https://index.crates.io/config.json \
        || fail "index.crates.io is unreachable even outside the sandbox; vendoring needs the network"
    if nonet curl -s -m 10 -o /dev/null https://index.crates.io/config.json; then
        fail "the network is still reachable inside the sandbox"
    fi
    say "network denied for builds ($(uname -s)): index.crates.io unreachable inside, reachable outside"
else
    say "network denied for builds ($(uname -s)); curl missing, not probed"
fi

say "vendoring into $work/vendor (online)"
mkdir -p "$work/cargo-home"
(cd "$repo" && cargo vendor --versioned-dirs "$work/vendor" > "$work/cargo-home/config.toml" 2> "$work/vendor.log") \
    || fail "cargo vendor (see $work/vendor.log)"

# A fresh, offline environment for one build: no registry cache, no
# libchdman-rs prebuilt cache, no earlier build output.
offline_build() { # name, extra env..., then -- cargo args
    name=$1; shift
    rm -rf "${work:?}/${name:?}"; mkdir -p "$work/$name/cache"
    envs="CARGO_HOME=$work/cargo-home CARGO_NET_OFFLINE=true XDG_CACHE_HOME=$work/$name/cache CARGO_TARGET_DIR=$work/$name/target"
    while [ "$1" != "--" ]; do envs="$envs $1"; shift; done
    shift
    say "$name: env $envs cargo $*"
    # shellcheck disable=SC2086
    (cd "$repo" && nonet env -u LIBCHDMAN_FORCE_SOURCE -u LIBCHDMAN_PREBUILT_FALLBACK $envs $wrap cargo "$@") \
        > "$work/$name.log" 2>&1
}

bins="--bin iris --bin chd-extract"
if [ "$expect_fail" = 1 ]; then
    # shellcheck disable=SC2086
    if offline_build without-force-source -- build --offline --locked --release -j "$jobs" $bins; then
        fail "the offline build succeeded without LIBCHDMAN_FORCE_SOURCE; expected the prebuilt download to fail"
    fi
    grep -q 'Prebuilt archive fetch failed' "$work/without-force-source.log" \
        || fail "the build without LIBCHDMAN_FORCE_SOURCE failed for another reason (see $work/without-force-source.log)"
    say "without LIBCHDMAN_FORCE_SOURCE: failed as expected:"
    grep -m1 'Prebuilt archive fetch failed' "$work/without-force-source.log" | cut -c1-200
    rm -rf "$work/without-force-source"
fi

# shellcheck disable=SC2086
offline_build with-force-source LIBCHDMAN_FORCE_SOURCE=1 -- build --offline --locked --release -j "$jobs" $bins \
    || fail "the documented offline build failed (see $work/with-force-source.log)"
grep -q 'Compiling libchdman-rs' "$work/with-force-source.log" || fail "libchdman-rs was not compiled"
say "with LIBCHDMAN_FORCE_SOURCE=1: built ($(grep Finished "$work/with-force-source.log" | tail -1 | sed 's/^ *//'))"
bin="$work/with-force-source/target/release"

if [ "$gui" = 1 ]; then
    # Reuse the same isolated cache and target so iris is not rebuilt from scratch.
    say "iris-gui: cargo check (lightning,bundled)"
    # shellcheck disable=SC2086
    (cd "$repo" && nonet env CARGO_HOME="$work/cargo-home" CARGO_NET_OFFLINE=true \
        XDG_CACHE_HOME="$work/with-force-source/cache" CARGO_TARGET_DIR="$work/with-force-source/target" \
        LIBCHDMAN_FORCE_SOURCE=1 $wrap cargo check --offline --locked --release -j "$jobs" \
        -p iris-gui --features iris/lightning,bundled) > "$work/gui.log" 2>&1 \
        || fail "iris-gui offline check failed (see $work/gui.log)"
    say "iris-gui: checked"
fi

if command -v chdman >/dev/null 2>&1; then
    say "CHD round trip through the offline-built chd-extract"
    d="$work/chd"; mkdir -p "$d"
    # Compressible text followed by incompressible bytes, 8 MiB in all.
    i=0; : > "$d/disk.raw"
    while [ "$i" -lt 4096 ]; do printf 'iris offline build sector pattern %08d\n' "$i"; i=$((i + 1)); done > "$d/text"
    while [ "$(wc -c < "$d/disk.raw")" -lt 4194304 ]; do cat "$d/text" >> "$d/disk.raw"; done
    head -c 4194304 "$d/disk.raw" > "$d/half"; mv "$d/half" "$d/disk.raw"
    head -c 4194304 /dev/urandom >> "$d/disk.raw"
    chdman createhd -i "$d/disk.raw" -o "$d/disk.chd" > "$d/chdman.log" 2>&1 || fail "chdman createhd (see $d/chdman.log)"
    "$bin/chd-extract" "$d/disk.chd" "$d/out.raw" > "$d/extract.log" 2>&1 || fail "chd-extract (see $d/extract.log)"
    cmp "$d/disk.raw" "$d/out.raw" || fail "chd-extract output differs from the original image"
    say "chd-extract: $(wc -c < "$d/out.raw" | tr -d ' ') bytes, identical to the image chdman compressed"
else
    say "SKIP CHD round trip: chdman not installed"
fi

say "PASS"
