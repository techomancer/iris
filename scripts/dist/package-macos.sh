#!/usr/bin/env bash
# Build and package IRIS for macOS as universal (arm64 + x86_64) binaries:
# the native AppKit app, "IRIS.app" (iris-gui/frontends/appkit), as a
# .dmg and a .zip, plus the CLI tarball holding iris and iris-ci. The egui
# IRIS.app is no longer packaged for macOS (scripts/dist/macos-bundle.sh still
# builds it by hand).
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
#
# CI builds each architecture on its own runner: `--build-only --target T`
# builds T's iris-appkit, iris and iris-ci into target/T/release and stops; the
# packaging run then gets both slices there and runs with --skip-build.
source "$(dirname "$0")/lib.sh"

parse_common_args "$@"
set -- ${REST[@]+"${REST[@]}"}
BUILD_ONLY=0
case "${1:-}" in
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; common_usage
               echo "  --build-only         build --target's binaries into target/ and stop"; exit 0 ;;
    --build-only) BUILD_ONLY=1 ;;
    "") ;;
    *) die "unknown option: $1" ;;
esac
[ "$(uname -s)" = Darwin ] || die "package-macos.sh runs on macOS only"

if [ "$BUILD_ONLY" = 1 ]; then
    [ "$SKIP_BUILD" = 1 ] && die "--build-only and --skip-build are opposites"
    # The same features the bundle and the CLI steps below build with.
    build_appkit "$TARGET" "$(with_bundled "$GUI_FEATURES")"
    build_cli "$TARGET" "$CLI_FEATURES"
    ls -l "$(release_dir "$TARGET")"/{iris-appkit,iris,iris-ci} >&2
    exit 0
fi
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

# Submit a file and wait for Apple's verdict, printing Apple's log when it is
# not Accepted. The notary service sometimes drops a connection (a TLS error
# while `notarytool submit --wait` polls), which used to fail the build with
# the submission still in progress: submit, then poll, retrying both.
# Only notarytool's standard output is read as JSON (its warnings go to the
# log), and a status query that keeps failing (a revoked key, a wrong team)
# stops after a few minutes instead of spending the whole deadline.
notarize() {  # <file>
    [ "${#NOTARY[@]}" -gt 0 ] || return 0
    local out="" id="" status="" try deadline name err fails=0
    name=$(basename "$1")
    err=$(mktemp "${TMPDIR:-/tmp}/notarytool.XXXXXX")
    group_start "notarize $name"
    for try in 1 2 3; do
        if out=$(xcrun notarytool submit "$1" "${NOTARY[@]}" --output-format json 2>"$err"); then
            id=$(printf '%s' "$out" | plutil -extract id raw -o - - 2>/dev/null || true)
            [ -n "$id" ] && break
        fi
        warn "submitting $name for notarization failed (try $try): $out $(cat "$err")"
        [ "$try" -lt 3 ] && sleep 30
    done
    [ -n "$id" ] || { rm -f "$err"; die "could not submit $name for notarization"; }
    echo "submitted $name: $id"
    deadline=$(( $(date +%s) + 40 * 60 ))
    while :; do
        if out=$(xcrun notarytool info "$id" "${NOTARY[@]}" --output-format json 2>"$err"); then
            fails=0
            status=$(printf '%s' "$out" | plutil -extract status raw -o - - 2>/dev/null || true)
            [ -n "$status" ] && [ "$status" != "In Progress" ] && break
        else
            fails=$((fails + 1))
            warn "asking for the notarization status of $name failed ($fails in a row): $(cat "$err")"
            if [ "$fails" -ge 8 ]; then
                out=$(cat "$err"); rm -f "$err"
                die "the notarization status of $name ($id) could not be read: $out"
            fi
        fi
        [ "$(date +%s)" -lt "$deadline" ] || { rm -f "$err"; die "notarization of $name ($id) did not finish within 40 minutes"; }
        sleep 20
    done
    rm -f "$err"
    echo "$out"
    if [ "$status" != "Accepted" ]; then
        xcrun notarytool log "$id" "${NOTARY[@]}" || true
        die "notarization of $name finished as '${status:-unknown}'"
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
    lipo -create -output "$WORK/iris-appkit" \
        "$(release_dir aarch64-apple-darwin)/iris-appkit" "$(release_dir x86_64-apple-darwin)/iris-appkit"
    BUNDLE_ARGS+=(--binary "$WORK/iris-appkit")
fi
"$DIST_ROOT/iris-gui/frontends/appkit/scripts/bundle.sh" "${BUNDLE_ARGS[@]}"
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
# Size the image ourselves: hdiutil's own estimate from -srcfolder came out
# too small on the macos-26 runners ("No space left on device" copying
# iris-gui into /Volumes/IRIS with 95 GB free on the host). Content plus a
# fifth plus 20 MB; UDZO compresses the slack away.
DMG_KB=$(du -sk "$WORK/dmg" | cut -f1)
IMAGE_KB=$(( DMG_KB + DMG_KB / 5 + 20480 ))
echo "dmg contents: $(( DMG_KB / 1024 )) MB, image: $(( IMAGE_KB / 1024 )) MB"
hdiutil create -volname "IRIS" -srcfolder "$WORK/dmg" -fs HFS+ \
    -size "${IMAGE_KB}k" -ov -format UDZO "$DMG"
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
