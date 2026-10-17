#!/usr/bin/env bash
# Write the GitHub release body (with the downloads table) for a version, or
# check that a directory holds every file that table links to.
#
#   scripts/dist/release-notes.sh --version V --tag TAG [--channel nightly|release]
#                                 [--repo owner/name] [--commit SHA] [-o FILE]
#                                 [--gui-features LIST] [--cli-features LIST]
#   scripts/dist/release-notes.sh --version V --check DIR
#
# The asset list below is the single source of truth for what a release ships;
# build.yml checks the built files against it before anything is published.
source "$(dirname "$0")/lib.sh"

CHANNEL=release
TAG=""
REPO="${GITHUB_REPOSITORY:-techomancer/iris}"
COMMIT=""
OUT=""
CHECK=""
while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --tag)     TAG="$2"; shift 2 ;;
        --channel) CHANNEL="$2"; shift 2 ;;
        --repo)    REPO="$2"; shift 2 ;;
        --commit)  COMMIT="$2"; shift 2 ;;
        --gui-features) GUI_FEATURES="$2"; shift 2 ;;
        --cli-features) CLI_FEATURES="$2"; shift 2 ;;
        -o)        OUT="$2"; shift 2 ;;
        --check)   CHECK="$2"; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done
[ -n "$VERSION" ] || die "--version is required"
V="$VERSION"

assets() {
    local a
    for a in x64 arm64; do
        echo "IRIS-Setup-windows-${a}-${V}.exe"
        echo "IRIS-gui-windows-${a}-${V}.zip"
        echo "IRIS-cli-windows-${a}-${V}.zip"
        echo "IRIS-gui-linux-${a}-${V}.AppImage"
        echo "IRIS-gui-linux-${a}-${V}.deb"
        echo "IRIS-gui-linux-${a}-${V}.rpm"
        echo "IRIS-gui-linux-${a}-${V}.pkg.tar.zst"
        echo "IRIS-cli-linux-${a}-${V}.tar.gz"
    done
    echo "IRIS-gui-macos-universal-${V}.dmg"
    echo "IRIS-gui-macos-universal-${V}.zip"
    echo "IRIS-cli-macos-universal-${V}.tar.gz"
    echo "IRIS-gui-linux-riscv64-${V}.deb"
    echo "IRIS-gui-linux-riscv64-${V}.rpm"
    echo "IRIS-gui-linux-riscv64-${V}.tar.gz"
    echo "IRIS-cli-linux-riscv64-${V}.tar.gz"
}

if [ -n "$CHECK" ]; then
    missing=0
    while read -r f; do
        if [ ! -s "$CHECK/$f" ]; then
            echo "missing: $f" >&2
            missing=$((missing + 1))
        fi
    done < <(assets)
    [ "$missing" = 0 ] || die "$missing expected release file(s) missing from $CHECK"
    log "all $(assets | wc -l | tr -d ' ') release files present"
    exit 0
fi

[ -n "$TAG" ] || die "--tag is required"
D="https://github.com/${REPO}/releases/download/${TAG}"
dl() { echo "[$1](${D}/$2)"; }

{
if [ "$CHANNEL" = nightly ]; then
    echo "**Version \`${V}\`**${COMMIT:+, built from [\`${COMMIT:0:8}\`](https://github.com/${REPO}/commit/${COMMIT})}"
else
    echo "**Version \`${V}\`**"
fi
cat <<EOF

## Downloads

| Platform | Emulator with GUI | Command line only |
|---|---|---|
| Windows x64 | $(dl Installer "IRIS-Setup-windows-x64-${V}.exe") · $(dl "Portable zip" "IRIS-gui-windows-x64-${V}.zip") | $(dl zip "IRIS-cli-windows-x64-${V}.zip") |
| Windows arm64 | $(dl Installer "IRIS-Setup-windows-arm64-${V}.exe") · $(dl "Portable zip" "IRIS-gui-windows-arm64-${V}.zip") | $(dl zip "IRIS-cli-windows-arm64-${V}.zip") |
| macOS, Apple silicon and Intel | $(dl dmg "IRIS-gui-macos-universal-${V}.dmg") · $(dl zip "IRIS-gui-macos-universal-${V}.zip") | $(dl tar.gz "IRIS-cli-macos-universal-${V}.tar.gz") |
| Linux x64 | $(dl AppImage "IRIS-gui-linux-x64-${V}.AppImage") · $(dl deb "IRIS-gui-linux-x64-${V}.deb") · $(dl rpm "IRIS-gui-linux-x64-${V}.rpm") · $(dl "Arch pkg" "IRIS-gui-linux-x64-${V}.pkg.tar.zst") | $(dl tar.gz "IRIS-cli-linux-x64-${V}.tar.gz") |
| Linux arm64 | $(dl AppImage "IRIS-gui-linux-arm64-${V}.AppImage") · $(dl deb "IRIS-gui-linux-arm64-${V}.deb") · $(dl rpm "IRIS-gui-linux-arm64-${V}.rpm") · $(dl "Arch pkg" "IRIS-gui-linux-arm64-${V}.pkg.tar.zst") | $(dl tar.gz "IRIS-cli-linux-arm64-${V}.tar.gz") |
| Linux riscv64 | $(dl deb "IRIS-gui-linux-riscv64-${V}.deb") · $(dl rpm "IRIS-gui-linux-riscv64-${V}.rpm") · $(dl tar.gz "IRIS-gui-linux-riscv64-${V}.tar.gz") | $(dl tar.gz "IRIS-cli-linux-riscv64-${V}.tar.gz") |

Not sure which to pick? On Windows take the installer, on macOS the dmg, on Linux the AppImage.

The command-line archives hold \`iris\`, the emulator itself, and \`iris-ci\`, the client for its automation socket.

<sub>Features: \`${GUI_FEATURES}\` (iris-gui), \`${CLI_FEATURES}\` (iris, iris-ci)</sub>
EOF
} > "${OUT:-/dev/stdout}"
