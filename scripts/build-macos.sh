#!/bin/bash
# Build iris-gui as a local IRIS.app for testing on this Mac.
#
# Running the binary directly (cargo run / ./iris-gui) always opens inside the
# current Terminal window; an .app launched with `open IRIS.app` does not.
# This is a shortcut for scripts/dist/macos-bundle.sh, building for this Mac's
# architecture only. Use that script directly for a universal build or other
# features.
#
# Usage:
#   ./scripts/build-macos.sh            # the nightly/release feature set
#   ./scripts/build-macos.sh lightning  # iris/lightning only
#   ./scripts/build-macos.sh appstore   # SANDBOXED build (App Store parity)
#
# The `appstore` variant compiles `--features appstore` (so the real
# security-scoped bookmark code + IRIS_CHD_DIFF_DIR are active, not the
# off-sandbox stubs) and signs with installer/iris-gui-sandbox-local.entitlements
# (app-sandbox = true). The result is a genuinely sandboxed IRIS.app you can run
# locally — no App Store round-trip — to test the folder-grant / CHD-fold flow.
# Ad-hoc signing by default; set CODESIGN_IDENTITY="Developer ID Application: …"
# to get persistent app-scoped bookmarks (see the entitlements file's caveat).
#
# The app lands in the repository root: open IRIS.app

set -e
cd "$(dirname "$0")/.."

VARIANT="${1:-standard}"
case "$(uname -m)" in
    arm64)  ARCH=arm64 ;;
    x86_64) ARCH=x86_64 ;;
    *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

ARGS=(--arch "$ARCH" --out . --sign "${CODESIGN_IDENTITY:--}")
case "$VARIANT" in
    standard) ;;
    lightning) ARGS+=(--features iris/lightning) ;;
    appstore|sandbox)
        # lightning gives a usable interpreter (the appstore feature forces
        # IRIS_NO_JIT, so the MIPS/REX JITs are off regardless).
        ARGS+=(--features "appstore,iris/lightning"
               --entitlements installer/iris-gui-sandbox-local.entitlements) ;;
    *) echo "Unknown variant '$VARIANT' (standard, lightning, appstore)" >&2; exit 2 ;;
esac

scripts/dist/macos-bundle.sh "${ARGS[@]}"

echo ""
echo "Launch without Terminal:  open IRIS.app"
if [ "$VARIANT" = "appstore" ] || [ "$VARIANT" = "sandbox" ]; then
    echo ""
    echo "This is a SANDBOXED build. Verify the sandbox is actually on:"
    echo "  codesign -d --entitlements - IRIS.app 2>/dev/null | grep -A1 app-sandbox"
    echo "  ls ~/Library/Containers/io.techomancer.iris/   # created on first launch"
    echo ""
    echo "Test the CHD folder-grant / fold: attach a compressed .chd from a"
    echo "folder you have NOT granted -> the grant modal should appear; grant the"
    echo "folder, boot, then quit -> the .diff.chd should fold away and the disk"
    echo "shrink. (Ad-hoc bookmarks may not persist across relaunches; the"
    echo "within-session fold does not depend on that.)"
fi
