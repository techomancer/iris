#!/usr/bin/env bash
# Build and package IRIS for Windows: the iris-gui installer (Inno Setup) and
# portable zip, plus the CLI zip holding iris.exe and iris-ci.exe.
#
#   scripts/dist/package-windows.sh [options]
#
# Run it from Git Bash (what GitHub's Windows runners use for `shell: bash`).
# Needs 7-Zip (7z) and Inno Setup 6; set ISCC to the compiler's path if it is
# not in the default install location.
source "$(dirname "$0")/lib.sh"

parse_common_args "$@"
set -- ${REST[@]+"${REST[@]}"}
case "${1:-}" in
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; common_usage; exit 0 ;;
    "") ;;
    *) die "unknown option: $1" ;;
esac

case "$TARGET" in
    x86_64-pc-windows-*)  ARCH=x64 ;;
    aarch64-pc-windows-*) ARCH=arm64 ;;
    *) die "unsupported Windows target $TARGET" ;;
esac

need 7z cygpath
ISCC="${ISCC:-/c/Program Files (x86)/Inno Setup 6/ISCC.exe}"
[ -f "$ISCC" ] || command -v "$ISCC" >/dev/null 2>&1 \
    || die "Inno Setup compiler not found at '$ISCC' (choco install innosetup, or set ISCC)"

GUI_FEATURES=$(with_bundled "$GUI_FEATURES")
build_gui "$TARGET" "$GUI_FEATURES"
build_cli "$TARGET" "$CLI_FEATURES"
REL=$(release_dir "$TARGET")

winpath() { cygpath -w "$1"; }

group_start "installer"
WORK=$(stage_dir)
printf '%s\n\n\n---\n\n\n' "$(cat LICENSE)" > "$WORK/COMBINED-LICENSE.txt"
cat LICENSE-libchdman-rs.txt >> "$WORK/COMBINED-LICENSE.txt"
"$ISCC" "/DMyAppVersion=$VERSION" "/DTargetArch=$ARCH" \
    "/DSourceDir=$(winpath "$REL")" \
    "/DAssetsDir=$(winpath "$DIST_ROOT/iris-gui/assets/icons")" \
    "/DLicenseFile=$(winpath "$WORK/COMBINED-LICENSE.txt")" \
    "/O$(winpath "$OUT_DIR")" \
    "/FIRIS-Setup-windows-${ARCH}-${VERSION}" \
    "$(winpath "$DIST_ROOT/installer/iris-gui.iss")"
rm -rf "$WORK"
[ -f "$OUT_DIR/IRIS-Setup-windows-${ARCH}-${VERSION}.exe" ] || die "the installer was not produced"
group_end

group_start "portable zip"
S=$(stage_dir)
cp "$REL/iris-gui.exe" iris-gui/assets/icons/icon.ico LICENSE LICENSE-libchdman-rs.txt "$S/"
(cd "$S" && 7z a -bso0 "$(winpath "$OUT_DIR")\\IRIS-gui-windows-${ARCH}-${VERSION}.zip" *)
rm -rf "$S"
group_end

group_start "CLI zip"
CLI="$OUT_DIR/IRIS-cli-windows-${ARCH}-${VERSION}.zip"
S=$(stage_dir)
cp "$REL/iris.exe" "$REL/iris-ci.exe" LICENSE LICENSE-libchdman-rs.txt "$S/"
(cd "$S" && 7z a -bso0 "$(winpath "$CLI")" *)
rm -rf "$S"
verify_members "$CLI" iris.exe iris-ci.exe LICENSE LICENSE-libchdman-rs.txt
group_end

log "done:"; ls -lh "$OUT_DIR" >&2
