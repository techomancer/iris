# shellcheck shell=bash disable=SC2034
# Shared helpers for the scripts in scripts/dist/. Sourced, never run.
#
# Every distribution build (nightly, tagged release, a local build on your own
# machine) goes through these scripts. The GitHub workflows only install
# toolchains, call a script, and upload whatever it left in dist/, so anything
# CI can build you can build the same way locally.

set -euo pipefail

DIST_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$DIST_ROOT"

# The feature sets distribution builds ship with. The workflows pass these
# explicitly; these defaults are what you get when you run a script by hand.
DEFAULT_GUI_FEATURES="iris/lightning,iris/jitv2,iris/gpujit,iris/idle-pause,hostgl"
DEFAULT_CLI_FEATURES="lightning,jitv2,gpujit,idle-pause,hostgl"

# Bundle ID of the Developer ID (nightly/release) macOS app. The App Store
# build keeps its own ID, set in .github/workflows/appstore.yml.
DEFAULT_BUNDLE_ID="io.techomancer.iris"

log()  { printf '==> %s\n' "$*" >&2; }
warn() { if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::warning::$*" >&2; else printf 'warning: %s\n' "$*" >&2; fi; }
die()  { if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::error::$*" >&2; else printf 'error: %s\n' "$*" >&2; fi; exit 1; }

group_start() { if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::group::$*"; else log "$*"; fi; }
group_end()   { if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::endgroup::"; fi; }

need() {
    local cmd
    for cmd in "$@"; do
        command -v "$cmd" >/dev/null 2>&1 || die "'$cmd' is required but not installed"
    done
}

host_target() { rustc -vV | awk '/^host:/ {print $2}'; }

# Make sure the toolchain cargo will use (rust-toolchain.toml pins nightly)
# has the std for TARGET. A no-op without rustup (e.g. Arch's system rust).
ensure_target() {
    local target="$1"
    command -v rustup >/dev/null 2>&1 || return 0
    rustup target list --installed 2>/dev/null | grep -qx "$target" && return 0
    log "adding rust target $target"
    rustup target add "$target"
}

# The packaged GUI is always a `bundled` build: it hides the tools that only
# make sense in a source checkout. Appended here so the feature lists above stay
# the engine features the user asked for.
with_bundled() {
    case ",$1," in
        *,bundled,*|*,appstore,*) echo "$1" ;;
        *) echo "${1:+$1,}bundled" ;;
    esac
}

# Version string for package managers that need a leading digit (deb, rpm):
# 1.2.3 stays as it is, nightly.2026.10.17 becomes 0~nightly.2026.10.17 so a
# real release always sorts above it.
pkg_version() {
    case "$1" in
        [0-9]*) echo "$1" ;;
        *) echo "0~$1" ;;
    esac
}

# Common option parsing. Sets VERSION, TARGET, GUI_FEATURES, CLI_FEATURES,
# OUT_DIR, SKIP_BUILD and leaves anything it does not recognise in REST.
VERSION=""
TARGET=""
GUI_FEATURES="$DEFAULT_GUI_FEATURES"
CLI_FEATURES="$DEFAULT_CLI_FEATURES"
OUT_DIR="$DIST_ROOT/dist"
SKIP_BUILD=0
REST=()

parse_common_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --version)      VERSION="$2"; shift 2 ;;
            --target)       TARGET="$2"; shift 2 ;;
            --gui-features) GUI_FEATURES="$2"; shift 2 ;;
            --cli-features) CLI_FEATURES="$2"; shift 2 ;;
            --out)          OUT_DIR="$2"; shift 2 ;;
            --skip-build)   SKIP_BUILD=1; shift ;;
            *)              REST+=("$1"); shift ;;
        esac
    done
    [ -n "$VERSION" ] || VERSION=$("$DIST_ROOT/scripts/dist/version.sh" dev)
    [ -n "$TARGET" ]  || TARGET=$(host_target)
    case "$OUT_DIR" in /*) ;; *) OUT_DIR="$DIST_ROOT/$OUT_DIR" ;; esac
    mkdir -p "$OUT_DIR"
    # iris-gui/build.rs bakes this into the About window, src/lib.rs into the
    # CLI's --version.
    export RELEASE_VERSION="$VERSION"
}

# Print the common options for a script's --help.
common_usage() {
    cat <<EOF
Common options:
  --version V          version to stamp (default: dev.<commit>)
  --target TRIPLE      rust target (default: the host)
  --gui-features LIST  iris-gui features (default: $DEFAULT_GUI_FEATURES)
  --cli-features LIST  iris/iris-ci features (default: $DEFAULT_CLI_FEATURES)
  --out DIR            where finished files go (default: dist/)
  --skip-build         package binaries already in target/ instead of building
EOF
}

release_dir() { echo "$DIST_ROOT/target/$1/release"; }

build_gui() {  # <target> <features>
    [ "$SKIP_BUILD" = 1 ] && return 0
    ensure_target "$1"
    group_start "cargo build iris-gui ($1) features=$2"
    cargo build --release --target "$1" -p iris-gui --features "$2"
    group_end
}

build_cli() {  # <target> <features>
    [ "$SKIP_BUILD" = 1 ] && return 0
    ensure_target "$1"
    group_start "cargo build iris + iris-ci ($1) features=$2"
    cargo build --release --target "$1" -p iris --bin iris --bin iris-ci --features "$2"
    group_end
}

# Stage files into a temporary directory and return its path. Callers archive
# from inside it so archives hold flat members, never target/... paths.
stage_dir() {
    local d
    d=$(mktemp -d "${TMPDIR:-/tmp}/iris-dist.XXXXXX")
    echo "$d"
}

# Fail if an archive is missing any of the given top-level members.
verify_members() {  # <archive> <member>...
    local archive="$1"; shift
    local listing want
    case "$archive" in
        *.zip) listing=$(unzip -Z1 "$archive" 2>/dev/null || 7z l -ba -slt "$archive" | sed -n 's/^Path = //p') ;;
        *)     listing=$(tar tzf "$archive") ;;
    esac
    for want in "$@"; do
        printf '%s\n' "$listing" | grep -qx "$want" \
            || die "$(basename "$archive") is missing member '$want'"
    done
}
