#!/usr/bin/env bash
# Build and package IRIS for macOS as universal (arm64 + x86_64) binaries:
# IRIS.app as a .dmg and a .zip, plus the CLI tarball holding iris and iris-ci.
#
#   scripts/dist/package-macos.sh [options]
#
# Signing and notarization run when their credentials are present and are
# skipped (ad-hoc signature, no notarization) when they are not:
#
#   MACOS_SIGN_IDENTITY   a "Developer ID Application: …" identity in your
#                         keychain. Without it everything is signed ad hoc.
#   Notarization, either an App Store Connect API key
#     NOTARY_KEY_ID, NOTARY_ISSUER_ID, NOTARY_KEY_PATH (the .p8 file)
#   or an Apple ID
#     NOTARY_APPLE_ID, NOTARY_PASSWORD (app-specific), NOTARY_TEAM_ID
#
# Set REQUIRE_NOTARIZATION=1 to fail instead of shipping unnotarized files.
source "$(dirname "$0")/lib.sh"

parse_common_args "$@"
set -- ${REST[@]+"${REST[@]}"}
case "${1:-}" in
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; common_usage; exit 0 ;;
    "") ;;
    *) die "unknown option: $1" ;;
esac
[ "$(uname -s)" = Darwin ] || die "package-macos.sh runs on macOS only"
need lipo codesign hdiutil ditto

IDENTITY="${MACOS_SIGN_IDENTITY:--}"
TARGETS=(aarch64-apple-darwin x86_64-apple-darwin)

NOTARY=()
if [ -n "${NOTARY_KEY_ID:-}" ] && [ -n "${NOTARY_ISSUER_ID:-}" ] && [ -n "${NOTARY_KEY_PATH:-}" ]; then
    NOTARY=(--key "$NOTARY_KEY_PATH" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER_ID")
elif [ -n "${NOTARY_APPLE_ID:-}" ] && [ -n "${NOTARY_PASSWORD:-}" ] && [ -n "${NOTARY_TEAM_ID:-}" ]; then
    NOTARY=(--apple-id "$NOTARY_APPLE_ID" --password "$NOTARY_PASSWORD" --team-id "$NOTARY_TEAM_ID")
fi
if [ "$IDENTITY" = "-" ]; then
    NOTARY=()   # Apple refuses to notarize ad-hoc signatures
    msg="no Developer ID identity: signing ad hoc, not notarizing"
    [ "${REQUIRE_NOTARIZATION:-0}" = 1 ] && die "$msg"
    warn "$msg"
elif [ "${#NOTARY[@]}" = 0 ]; then
    msg="signing as '$IDENTITY' but no notarization credentials are set: not notarizing"
    [ "${REQUIRE_NOTARIZATION:-0}" = 1 ] && die "$msg"
    warn "$msg"
fi

# Submit a file and wait. notarytool exits 0 even for a rejected submission, so
# read the status and print Apple's log when it is not Accepted.
notarize() {  # <file>
    [ "${#NOTARY[@]}" -gt 0 ] || return 0
    local out id status
    group_start "notarize $(basename "$1")"
    out=$(xcrun notarytool submit "$1" "${NOTARY[@]}" --wait --output-format json)
    echo "$out"
    id=$(printf '%s' "$out" | plutil -extract id raw -o - - 2>/dev/null || true)
    status=$(printf '%s' "$out" | plutil -extract status raw -o - - 2>/dev/null || true)
    if [ "$status" != "Accepted" ]; then
        [ -n "$id" ] && xcrun notarytool log "$id" "${NOTARY[@]}" || true
        die "notarization of $(basename "$1") finished as '${status:-unknown}'"
    fi
    group_end
}

sign_binary() {  # <file>
    if [ "$IDENTITY" = "-" ]; then
        codesign --force --sign - "$1"
    else
        # The JIT entitlements keep the hardened runtime from killing the
        # first Cranelift-compiled block (REX3 shaders, jitv2).
        codesign --force --options runtime --timestamp \
            --entitlements installer/iris-gui-notarized.entitlements \
            --sign "$IDENTITY" "$1"
    fi
}

WORK=$(stage_dir)
trap 'rm -rf "$WORK"' EXIT

# ── GUI ──────────────────────────────────────────────────────────────────────
BUNDLE_ARGS=(--features "$(with_bundled "$GUI_FEATURES")" --version "$VERSION" --out "$WORK" --sign "$IDENTITY")
if [ "$SKIP_BUILD" = 1 ]; then
    lipo -create -output "$WORK/iris-gui" \
        "$(release_dir aarch64-apple-darwin)/iris-gui" "$(release_dir x86_64-apple-darwin)/iris-gui"
    BUNDLE_ARGS+=(--binary "$WORK/iris-gui")
fi
"$DIST_ROOT/scripts/dist/macos-bundle.sh" "${BUNDLE_ARGS[@]}"
APP="$WORK/IRIS.app"

if [ "${#NOTARY[@]}" -gt 0 ]; then
    ditto -c -k --keepParent "$APP" "$WORK/IRIS-notarize.zip"
    notarize "$WORK/IRIS-notarize.zip"
    xcrun stapler staple "$APP"
    rm -f "$WORK/IRIS-notarize.zip"
fi

group_start "zip + dmg"
ZIP="$OUT_DIR/IRIS-gui-macos-universal-${VERSION}.zip"
DMG="$OUT_DIR/IRIS-gui-macos-universal-${VERSION}.dmg"
rm -f "$ZIP" "$DMG"
ditto -c -k --keepParent "$APP" "$ZIP"
mkdir -p "$WORK/dmg"
cp -R "$APP" "$WORK/dmg/"
ln -s /Applications "$WORK/dmg/Applications"
hdiutil create -volname "IRIS" -srcfolder "$WORK/dmg" -ov -format UDZO "$DMG"
[ "$IDENTITY" = "-" ] || codesign --force --timestamp --sign "$IDENTITY" "$DMG"
group_end
notarize "$DMG"
[ "${#NOTARY[@]}" -gt 0 ] && xcrun stapler staple "$DMG"

# ── CLI ──────────────────────────────────────────────────────────────────────
for t in "${TARGETS[@]}"; do
    build_cli "$t" "$CLI_FEATURES"
done
mkdir -p "$WORK/cli"
for b in iris iris-ci; do
    lipo -create -output "$WORK/cli/$b" \
        "$(release_dir aarch64-apple-darwin)/$b" "$(release_dir x86_64-apple-darwin)/$b"
    sign_binary "$WORK/cli/$b"
done
# Bare executables cannot carry a stapled ticket; notarizing still registers
# them, so Gatekeeper's online check passes.
ditto -c -k "$WORK/cli" "$WORK/cli-notarize.zip"
notarize "$WORK/cli-notarize.zip"
cp LICENSE LICENSE-libchdman-rs.txt "$WORK/cli/"
CLI="$OUT_DIR/IRIS-cli-macos-universal-${VERSION}.tar.gz"
tar -C "$WORK/cli" -czf "$CLI" iris iris-ci LICENSE LICENSE-libchdman-rs.txt
verify_members "$CLI" iris iris-ci LICENSE LICENSE-libchdman-rs.txt

log "done:"; ls -lh "$OUT_DIR" >&2
