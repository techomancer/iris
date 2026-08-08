#!/bin/bash
# Emit a build manifest: every crate version that went into a specific build,
# plus the toolchain and git state that produced it.
#
# Why: Cargo.lock is gitignored (see rules/build/dependency-upgrade-gotchas.md),
# so every CI run re-resolves the whole graph from scratch. A build that was
# green yesterday can go red today because a transitive crate five levels down
# shipped a new minor — and nothing in the diff shows it. Shipping this file
# with each release turns that class of failure into a `diff` of two manifests.
#
# Usage — pass the SAME cargo selection flags the build used, after `--`:
#
#   ./scripts/build-manifest.sh -o dist/manifest.txt -- -p iris-gui --features premiere,pcap
#   ./scripts/build-manifest.sh -o dist/manifest.txt -- --features lightning,rex-jit,chd
#   ./scripts/build-manifest.sh                        # defaults: root pkg, default features, stdout
#
# Run it AFTER the build, in the same job, on the same machine. It reads the
# Cargo.lock the build just produced; run it before, or with different feature
# flags, and it describes a resolve that never shipped.
#
# The fingerprint at the bottom is a sha256 of the crate list alone. Two
# releases with the same fingerprint had byte-identical dependency graphs, so a
# behaviour difference between them is NOT a dependency change — look
# elsewhere. Different fingerprints: `diff` the two manifests and the culprit
# is on the changed line.

set -euo pipefail

OUT=""
CARGO_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        -o|--output) OUT="$2"; shift 2 ;;
        --) shift; CARGO_ARGS=("$@"); break ;;
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (cargo flags go after --)" >&2; exit 2 ;;
    esac
done

# cargo has to run from the workspace, but a relative -o should still land where
# the caller expects, not silently under the repo root.
case "${OUT:-}" in
    ""|/*) ;;
    *) OUT="$PWD/$OUT" ;;
esac
cd "$(dirname "$0")/.."

# --- git state ---------------------------------------------------------------
if git rev-parse --git-dir >/dev/null 2>&1; then
    GIT_COMMIT=$(git rev-parse --short HEAD)
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if [ -n "$(git status --porcelain)" ]; then
        GIT_STATE="DIRTY (uncommitted changes — this build is not reproducible from git)"
    else
        GIT_STATE="clean"
    fi
else
    GIT_COMMIT="(not a git checkout)"; GIT_BRANCH="-"; GIT_STATE="-"
fi

# --- what was actually selected ----------------------------------------------
# cargo tree resolves for the host target unless --target is passed through, and
# honours -p / --features exactly as cargo build does. Keeping the args verbatim
# is what makes this manifest describe THIS build rather than some other one.
HOST=$(rustc -vV | awk '/^host:/ {print $2}')
SELECTION="${CARGO_ARGS[*]:-(none — root package, default features)}"

# {p} renders "name vX.Y.Z" for registry crates and appends the path/git source
# for everything else, so [patch.crates-io] entries (the vendored winit) are
# visible rather than masquerading as the crates.io release.
# The ${a[@]+"${a[@]}"} dance keeps `set -u` happy on bash 3.2 (what macOS
# ships) when no cargo args were passed — a plain "${a[@]}" is an unbound-
# variable error there, and "${a[@]:-}" would smuggle in an empty argument.
TREE=$(cargo tree --edges normal --prefix none --format '{p}' \
           ${CARGO_ARGS[@]+"${CARGO_ARGS[@]}"} 2>/dev/null \
       | sed 's/ (\*)$//' | grep -v '^[[:space:]]*$' | sort -u)

CRATES=$(printf '%s\n' "$TREE" | sed -E 's/^([^ ]+) v([^ ]+).*/\1 \2/' | sort -u)

# Non-registry = a path or URL source in the trailing parens. Excludes the
# "(proc-macro)" marker cargo also renders there, and excludes this workspace's
# own members (always path deps, never interesting) so the section shows only
# genuine overrides — the [patch.crates-io] winit, or any git dependency.
# --no-deps limits `packages` to workspace members. The "name" -> "version" key
# pair is unique to a package entry; dependency entries pair "name" with
# "source"/"req", so this can't accidentally match one.
MEMBERS=$(cargo metadata --no-deps --format-version 1 2>/dev/null \
          | grep -o '"name":"[^"]*","version":"' | sed -E 's/"name":"([^"]*)".*/\1/' || true)
NONREG=$(printf '%s\n' "$TREE" | grep -E ' \((/|[a-z+]+://)' || true)
for m in $MEMBERS; do
    NONREG=$(printf '%s\n' "$NONREG" | grep -v "^${m} v" || true)
done
NONREG=$(printf '%s\n' "$NONREG" | grep -v '^[[:space:]]*$' || true)

COUNT=$(printf '%s\n' "$CRATES" | grep -c '' || true)
FINGERPRINT=$(printf '%s\n' "$CRATES" | { shasum -a 256 2>/dev/null || sha256sum; } | cut -d' ' -f1)

# --- emit --------------------------------------------------------------------
emit() {
cat <<EOF
IRIS build manifest
===================
git commit   : $GIT_COMMIT ($GIT_BRANCH)
git state    : $GIT_STATE
built (UTC)  : $(date -u +%Y-%m-%dT%H:%M:%SZ)
host         : $HOST
cargo select : $SELECTION

toolchain
---------
$(rustc --version)
$(cargo --version)

crates ($COUNT resolved)
$(printf '%.0s-' $(seq 1 $((${#COUNT} + 18))))
$CRATES
EOF

if [ -n "$NONREG" ]; then
cat <<EOF

non-registry sources
--------------------
These did NOT come from crates.io. A [patch.crates-io] path override or git
dependency means the published version number alone does not identify the code
that shipped — check the patch table in Cargo.toml before trusting a version
match here.

$NONREG
EOF
fi

cat <<EOF

dependency fingerprint
----------------------
sha256(crate list) = $FINGERPRINT

Same fingerprint as a previous release => identical dependency graph, so any
behaviour change between them came from iris's own code or the toolchain, not
from a crate update. Different => diff the two manifests to find which crate
moved.
EOF
}

if [ -n "$OUT" ]; then
    mkdir -p "$(dirname "$OUT")"
    emit > "$OUT"
    echo "wrote $OUT ($COUNT crates, fingerprint ${FINGERPRINT:0:12})"
else
    emit
fi
