#!/usr/bin/env bash
# Build iris-gui and wrap it in IRIS.app: Info.plist, icon, code signature.
#
#   scripts/dist/macos-bundle.sh [options]
#
# This is the one place the macOS app bundle is assembled. The nightly and
# release builds call it through package-macos.sh, the App Store workflow calls
# it with --appstore, and you can run it by hand to get the same .app locally.
#
# Options:
#   --features LIST       iris-gui features
#                         (default: iris/lightning,iris/jitv2,iris/gpujit,iris/idle-pause,hostgl)
#   --version V           version shown in About and written to Info.plist
#                         (default: dev.<commit>)
#   --arch A              universal (default), arm64, x86_64
#   --binary PATH         bundle this already-built iris-gui instead of building
#   --out DIR             where IRIS.app goes (default: dist/)
#   --bundle-id ID        CFBundleIdentifier (default: io.techomancer.iris)
#   --short-version V     CFBundleShortVersionString (default: --version)
#   --build-version V     CFBundleVersion (default: --version)
#   --sign IDENTITY       codesign identity; "-" (the default) signs ad hoc.
#                         Also read from CODESIGN_IDENTITY.
#   --entitlements FILE   entitlements to sign with. Defaults to
#                         installer/iris-gui-notarized.entitlements for a real
#                         identity and none for ad-hoc signing.
#   --no-sign             leave the bundle unsigned
#   --appstore            App Store bundle: adds the export-compliance key and
#                         signs without a secure timestamp
#   --provisioning-profile FILE   embed this profile (App Store)
#
# Then: open dist/IRIS.app
source "$(dirname "$0")/lib.sh"

FEATURES="$DEFAULT_GUI_FEATURES"
ARCH_SEL=universal
BINARY=""
BUNDLE_ID="$DEFAULT_BUNDLE_ID"
SHORT_VERSION=""
BUILD_VERSION=""
IDENTITY="${CODESIGN_IDENTITY:--}"
ENTITLEMENTS=""
SIGN=1
APPSTORE=0
PROFILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --features)      FEATURES="$2"; shift 2 ;;
        --version)       VERSION="$2"; shift 2 ;;
        --arch)          ARCH_SEL="$2"; shift 2 ;;
        --binary)        BINARY="$2"; shift 2 ;;
        --out)           OUT_DIR="$2"; shift 2 ;;
        --bundle-id)     BUNDLE_ID="$2"; shift 2 ;;
        --short-version) SHORT_VERSION="$2"; shift 2 ;;
        --build-version) BUILD_VERSION="$2"; shift 2 ;;
        --sign)          IDENTITY="$2"; shift 2 ;;
        --entitlements)  ENTITLEMENTS="$2"; shift 2 ;;
        --no-sign)       SIGN=0; shift ;;
        --appstore)      APPSTORE=1; shift ;;
        --provisioning-profile) PROFILE="$2"; shift 2 ;;
        -h|--help)       sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[ "$(uname -s)" = Darwin ] || die "macos-bundle.sh runs on macOS only"
[ -n "$VERSION" ] || VERSION=$("$DIST_ROOT/scripts/dist/version.sh" dev)
[ -n "$SHORT_VERSION" ] || SHORT_VERSION="$VERSION"
[ -n "$BUILD_VERSION" ] || BUILD_VERSION="$VERSION"
case "$OUT_DIR" in /*) ;; *) OUT_DIR="$DIST_ROOT/$OUT_DIR" ;; esac
mkdir -p "$OUT_DIR"
export RELEASE_VERSION="$VERSION"

case "$ARCH_SEL" in
    universal) TARGETS=(aarch64-apple-darwin x86_64-apple-darwin) ;;
    arm64)     TARGETS=(aarch64-apple-darwin) ;;
    x86_64|x64) TARGETS=(x86_64-apple-darwin) ;;
    *) die "--arch must be universal, arm64 or x86_64" ;;
esac

# ── Build ────────────────────────────────────────────────────────────────────
WORK=$(stage_dir)
trap 'rm -rf "$WORK"' EXIT
if [ -z "$BINARY" ]; then
    slices=()
    for t in "${TARGETS[@]}"; do
        ensure_target "$t"
        group_start "cargo build iris-gui ($t) features=$FEATURES"
        cargo build --release --target "$t" -p iris-gui --features "$FEATURES"
        group_end
        slices+=("$(release_dir "$t")/iris-gui")
    done
    BINARY="$WORK/iris-gui"
    if [ "${#slices[@]}" -gt 1 ]; then
        lipo -create -output "$BINARY" "${slices[@]}"
    else
        cp "${slices[0]}" "$BINARY"
    fi
fi
[ -f "$BINARY" ] || die "$BINARY does not exist"
lipo -info "$BINARY" >&2

# ── Assemble ─────────────────────────────────────────────────────────────────
APP="$OUT_DIR/IRIS.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BINARY" "$APP/Contents/MacOS/iris-gui"
chmod 755 "$APP/Contents/MacOS/iris-gui"
cp iris-gui/assets/icons/icon.icns "$APP/Contents/Resources/AppIcon.icns"
if [ "$APPSTORE" = 0 ]; then
    cp LICENSE LICENSE-libchdman-rs.txt "$APP/Contents/Resources/"
fi
if [ -n "$PROFILE" ]; then
    cp "$PROFILE" "$APP/Contents/embedded.provisionprofile"
fi

APPSTORE_KEYS=""
if [ "$APPSTORE" = 1 ]; then
    # Export compliance: the app uses no non-exempt encryption (its network
    # path is plaintext; rustls/ring are build-time deps of libchdman-rs's
    # downloader only). Saying so here skips the per-build question in App
    # Store Connect.
    APPSTORE_KEYS="    <key>ITSAppUsesNonExemptEncryption</key><false/>"
fi

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>IRIS</string>
    <key>CFBundleDisplayName</key><string>IRIS</string>
    <key>CFBundleIdentifier</key><string>${BUNDLE_ID}</string>
    <key>CFBundleVersion</key><string>${BUILD_VERSION}</string>
    <key>CFBundleShortVersionString</key><string>${SHORT_VERSION}</string>
    <key>CFBundleExecutable</key><string>iris-gui</string>
    <key>CFBundleIconFile</key><string>AppIcon.icns</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSMinimumSystemVersion</key><string>10.13</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
    <key>NSCameraUsageDescription</key><string>Provides the IndyCam video input for SGI Indy emulation (VINO device).</string>
${APPSTORE_KEYS}
</dict>
</plist>
EOF
plutil -lint "$APP/Contents/Info.plist" >/dev/null

# ── Sign ─────────────────────────────────────────────────────────────────────
if [ "$SIGN" = 1 ]; then
    if [ "$IDENTITY" = "-" ]; then
        log "signing ad hoc"
        if [ -n "$ENTITLEMENTS" ]; then
            plutil -lint "$ENTITLEMENTS" >/dev/null
            codesign --force --deep --sign - --entitlements "$ENTITLEMENTS" "$APP"
        else
            codesign --force --deep --sign - "$APP"
        fi
    else
        [ -n "$ENTITLEMENTS" ] || ENTITLEMENTS="installer/iris-gui-notarized.entitlements"
        # codesign's entitlements parser is strict, and a parse failure would
        # otherwise surface as an unsigned or unsandboxed app.
        plutil -lint "$ENTITLEMENTS" >/dev/null
        log "signing as '$IDENTITY' with $ENTITLEMENTS"
        ts=(--timestamp)
        [ "$APPSTORE" = 1 ] && ts=()
        codesign --force --options runtime ${ts[@]+"${ts[@]}"} \
            --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$APP"
        codesign --verify --deep --strict "$APP"
    fi
fi

log "built $APP ($VERSION, $BUNDLE_ID)"
