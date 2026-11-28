#!/usr/bin/env bash
# Build iris-appkit and wrap it in "IRIS.app", the native macOS shell
# (bundle id io.techomancer.iris.appkit). Optionally sign with a Developer ID,
# notarize, and make a DMG. The release packager, scripts/dist/package-macos.sh,
# calls this with --binary to wrap the universal binary CI built, and the App
# Store workflow (.github/workflows/appstore.yml) with --appstore.
#
#   iris-gui/frontends/appkit/scripts/bundle.sh [options]
#
# Options:
#   --features LIST    iris-gui-appkit features
#                      (default: bundled,lightning,jitv2,idle-pause,hostgl)
#   --version V        the version (default: dev.<commit>)
#   --short-version V  CFBundleShortVersionString (default: --version)
#   --build-version V  CFBundleVersion (default: --version)
#   --arch A           universal (default), arm64, x86_64
#   --binary PATH      wrap this iris-appkit instead of building one
#   --out DIR          where the .app (and .dmg) go (default: dist/)
#   --bundle-id ID     CFBundleIdentifier (default: io.techomancer.iris.appkit)
#   --sign IDENTITY    codesign identity; "-" (the default) signs ad hoc.
#                      Also read from CODESIGN_IDENTITY.
#   --no-sign          leave the bundle unsigned (the caller signs it)
#   --entitlements F   sign with these entitlements
#   --appstore         Mac App Store bundle: the sandboxed build (default
#                      features appstore,lightning), the export-compliance
#                      key, signed with installer/iris-gui.entitlements
#   --provisioning-profile F   embed this profile (App Store)
#   --dmg              also make "IRIS-<version>-macos.dmg"
#   --notarize         submit the DMG with notarytool and staple it. Uses
#                      NOTARY_KEY_ID/NOTARY_ISSUER_ID/NOTARY_KEY_PATH, or
#                      NOTARY_APPLE_ID/NOTARY_PASSWORD/NOTARY_TEAM_ID, as
#                      scripts/dist/package-macos.sh does.
#
# Then: open dist/IRIS.app
source "$(dirname "$0")/../../../../scripts/dist/lib.sh"

FEATURES="bundled,lightning,jitv2,idle-pause,hostgl"
ARCH_SEL=universal
BUNDLE_ID="io.techomancer.iris.appkit"
IDENTITY="${CODESIGN_IDENTITY:--}"
DMG=0
NOTARIZE=0
NAME="IRIS"
PREBUILT=""
SHORT_VERSION=""
BUILD_VERSION=""
SIGN=1
ENTITLEMENTS=""
APPSTORE=0
PROFILE=""
FEATURES_SET=0

while [ $# -gt 0 ]; do
    case "$1" in
        --features)  FEATURES="$2"; FEATURES_SET=1; shift 2 ;;
        --short-version) SHORT_VERSION="$2"; shift 2 ;;
        --build-version) BUILD_VERSION="$2"; shift 2 ;;
        --no-sign)   SIGN=0; shift ;;
        --entitlements) ENTITLEMENTS="$2"; shift 2 ;;
        --appstore)  APPSTORE=1; shift ;;
        --provisioning-profile) PROFILE="$2"; shift 2 ;;
        --version)   VERSION="$2"; shift 2 ;;
        --arch)      ARCH_SEL="$2"; shift 2 ;;
        --binary)    PREBUILT="$2"; shift 2 ;;
        --out)       OUT_DIR="$2"; shift 2 ;;
        --bundle-id) BUNDLE_ID="$2"; shift 2 ;;
        --sign)      IDENTITY="$2"; shift 2 ;;
        --dmg)       DMG=1; shift ;;
        --notarize)  NOTARIZE=1; DMG=1; shift ;;
        -h|--help)   sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[ "$(uname -s)" = Darwin ] || die "bundle.sh runs on macOS only"
cd "$DIST_ROOT"
[ -n "$VERSION" ] || VERSION=$("$DIST_ROOT/scripts/dist/version.sh" dev)
[ -n "$SHORT_VERSION" ] || SHORT_VERSION="$VERSION"
[ -n "$BUILD_VERSION" ] || BUILD_VERSION="$VERSION"
if [ "$APPSTORE" = 1 ]; then
    # The sandbox has no use for the JIT features (the core turns the JITs
    # off there) nor for host GL; egui's App Store build was the same.
    [ "$FEATURES_SET" = 1 ] || FEATURES="appstore,lightning"
    [ -n "$ENTITLEMENTS" ] || ENTITLEMENTS="installer/iris-gui.entitlements"
fi
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
BINARY="$WORK/iris-appkit"
if [ -n "$PREBUILT" ]; then
    [ -x "$PREBUILT" ] || die "--binary $PREBUILT is not an executable"
    cp "$PREBUILT" "$BINARY"
else
    slices=()
    for t in "${TARGETS[@]}"; do
        ensure_target "$t"
        group_start "cargo build iris-gui-appkit ($t) features=$FEATURES"
        # macOS 11 is the floor (LSMinimumSystemVersion below).
        MACOSX_DEPLOYMENT_TARGET=11.0 cargo build --release --target "$t" -p iris-gui-appkit --features "$FEATURES"
        group_end
        slices+=("$(release_dir "$t")/iris-appkit")
    done
    if [ "${#slices[@]}" -gt 1 ]; then
        lipo -create -output "$BINARY" "${slices[@]}"
    else
        cp "${slices[0]}" "$BINARY"
    fi
fi
lipo -info "$BINARY" >&2

# ── Assemble ─────────────────────────────────────────────────────────────────
APP="$OUT_DIR/$NAME.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BINARY" "$APP/Contents/MacOS/iris-appkit"
chmod 755 "$APP/Contents/MacOS/iris-appkit"
cp iris-gui/assets/icons/icon.icns "$APP/Contents/Resources/AppIcon.icns"
cp LICENSE LICENSE-libchdman-rs.txt LICENSE-rusty-backup-crates.txt "$APP/Contents/Resources/"
[ -z "$PROFILE" ] || cp "$PROFILE" "$APP/Contents/embedded.provisionprofile"

APPSTORE_KEYS=""
if [ "$APPSTORE" = 1 ]; then
    # Export compliance: the app uses no non-exempt encryption (its network
    # path is plaintext). Saying so here skips the per-build question in App
    # Store Connect.
    APPSTORE_KEYS="    <key>ITSAppUsesNonExemptEncryption</key><false/>"
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>${NAME}</string>
    <key>CFBundleDisplayName</key><string>${NAME}</string>
    <key>CFBundleIdentifier</key><string>${BUNDLE_ID}</string>
    <key>CFBundleVersion</key><string>${BUILD_VERSION}</string>
    <key>CFBundleShortVersionString</key><string>${SHORT_VERSION}</string>
    <key>CFBundleExecutable</key><string>iris-appkit</string>
    <key>CFBundleIconFile</key><string>AppIcon.icns</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSPrincipalClass</key><string>NSApplication</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
    <key>NSCameraUsageDescription</key><string>Provides the IndyCam video input for SGI Indy emulation (VINO device).</string>
${APPSTORE_KEYS}
</dict>
</plist>
PLIST
plutil -lint "$APP/Contents/Info.plist" >/dev/null

# ── Sign ─────────────────────────────────────────────────────────────────────
if [ "$SIGN" = 0 ]; then
    log "left unsigned"
elif [ "$IDENTITY" = "-" ]; then
    log "signing ad hoc"
    if [ -n "$ENTITLEMENTS" ]; then
        plutil -lint "$ENTITLEMENTS" >/dev/null
        codesign --force --deep --sign - --entitlements "$ENTITLEMENTS" "$APP"
    else
        codesign --force --deep --sign - "$APP"
    fi
else
    ENT="${ENTITLEMENTS:-installer/iris-gui-notarized.entitlements}"
    # codesign's entitlements parser is strict, and a parse failure would
    # otherwise surface as an unsigned or unsandboxed app.
    plutil -lint "$ENT" >/dev/null
    log "signing as '$IDENTITY' with $ENT"
    ts=(--timestamp)
    [ "$APPSTORE" = 1 ] && ts=()
    codesign --force --options runtime ${ts[@]+"${ts[@]}"} --entitlements "$ENT" --sign "$IDENTITY" "$APP"
    codesign --verify --deep --strict "$APP"
fi

# ── DMG and notarization ─────────────────────────────────────────────────────
if [ "$DMG" = 1 ]; then
    DMG_PATH="$OUT_DIR/IRIS-${VERSION}-macos.dmg"
    rm -f "$DMG_PATH"
    STAGE="$WORK/dmg"
    mkdir -p "$STAGE"
    cp -R "$APP" "$STAGE/"
    ln -s /Applications "$STAGE/Applications"
    # An explicit size: hdiutil's own estimate from -srcfolder can come out
    # too small ("No space left on device"), as package-macos.sh found.
    DMG_KB=$(du -sk "$STAGE" | cut -f1)
    hdiutil create -volname "$NAME" -srcfolder "$STAGE" -fs HFS+ \
        -size "$(( DMG_KB + DMG_KB / 5 + 20480 ))k" -ov -format UDZO "$DMG_PATH" >/dev/null
    [ "$IDENTITY" = "-" ] || codesign --force --timestamp --sign "$IDENTITY" "$DMG_PATH"
    if [ "$NOTARIZE" = 1 ]; then
        need="--notarize needs NOTARY_KEY_ID/NOTARY_KEY_PATH/NOTARY_ISSUER_ID or NOTARY_APPLE_ID/NOTARY_PASSWORD/NOTARY_TEAM_ID"
        if [ -n "${NOTARY_KEY_ID:-}" ] && [ -n "${NOTARY_KEY_PATH:-}" ] && [ -n "${NOTARY_ISSUER_ID:-}" ]; then
            auth=(--key "$NOTARY_KEY_PATH" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER_ID")
        elif [ -n "${NOTARY_APPLE_ID:-}" ] && [ -n "${NOTARY_PASSWORD:-}" ] && [ -n "${NOTARY_TEAM_ID:-}" ]; then
            auth=(--apple-id "$NOTARY_APPLE_ID" --password "$NOTARY_PASSWORD" --team-id "$NOTARY_TEAM_ID")
        else
            die "$need"
        fi
        # notarytool exits 0 on a rejection too: read the verdict.
        verdict=$(xcrun notarytool submit "$DMG_PATH" "${auth[@]}" --wait --output-format json)
        status=$(printf '%s' "$verdict" | plutil -extract status raw -o - - 2>/dev/null || true)
        if [ "$status" != "Accepted" ]; then
            id=$(printf '%s' "$verdict" | plutil -extract id raw -o - - 2>/dev/null || true)
            [ -n "$id" ] && xcrun notarytool log "$id" "${auth[@]}" || true
            die "notarization finished as '${status:-unknown}'"
        fi
        xcrun stapler staple "$DMG_PATH"
    fi
    log "built $DMG_PATH"
fi

log "built $APP ($SHORT_VERSION build $BUILD_VERSION, $BUNDLE_ID)"
