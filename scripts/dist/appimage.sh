#!/usr/bin/env bash
# Build iris-gui as a self-contained AppImage with quick-sharun.
#
#   scripts/dist/appimage.sh [options] [--install-deps]
#
# Meant for the Arch Linux container CI uses (ghcr.io/pkgforge-dev/archlinux):
# quick-sharun collects libraries from the running system, and it writes into
# /usr, so run it as root in a throwaway container, not on your desktop.
# --install-deps installs the pacman packages it needs first.
#
# ANYLINUX_REF picks the Anylinux-AppImages revision the tools come from
# (default refs/heads/main; quick-sharun sha256-pins what it downloads itself).
source "$(dirname "$0")/lib.sh"

INSTALL_DEPS=0
parse_common_args "$@"
set -- ${REST[@]+"${REST[@]}"}
while [ $# -gt 0 ]; do
    case "$1" in
        --install-deps) INSTALL_DEPS=1; shift ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; common_usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$TARGET" in
    x86_64-*)  ARCH=x64 ;;
    aarch64-*) ARCH=arm64 ;;
    *) die "no AppImage for $TARGET" ;;
esac

ANYLINUX_REF="${ANYLINUX_REF:-refs/heads/main}"
TOOLS="https://raw.githubusercontent.com/pkgforge-dev/Anylinux-AppImages/${ANYLINUX_REF}/useful-tools"

if [ "$INSTALL_DEPS" = 1 ]; then
    group_start "pacman dependencies"
    pacman -Syu --noconfirm \
        base-devel cmake rust clang alsa-lib gtk3 openssl pkgconf v4l-utils \
        wget curl git \
        libxcb libxcursor libxi libxkbcommon libxkbcommon-x11 libxrandr libxtst \
        wayland wayland-protocols mesa vulkan-icd-loader \
        xorg-server-xvfb strace dbus file patchelf \
        desktop-file-utils shared-mime-info zsync
    group_end
fi
need wget

GUI_FEATURES=$(with_bundled "$GUI_FEATURES")
build_gui "$TARGET" "$GUI_FEATURES"

WORK=$(stage_dir)
group_start "debloated runtime packages + quick-sharun"
wget --retry-connrefused --tries=30 -q "$TOOLS/get-debloated-pkgs.sh" -O "$WORK/get-debloated-pkgs.sh"
wget --retry-connrefused --tries=30 -q "$TOOLS/quick-sharun.sh" -O "$WORK/quick-sharun"
chmod +x "$WORK/get-debloated-pkgs.sh" "$WORK/quick-sharun"
(cd "$WORK" && ./get-debloated-pkgs.sh --add-common --add-mesa --prefer-nano)
group_end

install -Dm755 "$(release_dir "$TARGET")/iris-gui" /usr/bin/iris-gui
install -Dm644 iris-gui/assets/icons/icon-256.png /usr/share/icons/hicolor/256x256/apps/iris-gui.png
install -Dm644 iris-gui/iris-gui.desktop /usr/share/applications/iris-gui.desktop

group_start "AppImage"
# appimagetool reads OUTPATH/OUTNAME/VERSION, and emits the .zsync from
# GITHUB_REPOSITORY when that is set. Do not override APPIMAGETOOL_LINK:
# quick-sharun sha256-checks the appimagetool it ships with.
export OUTPATH="$OUT_DIR"
export OUTNAME="IRIS-gui-linux-${ARCH}-${VERSION}.AppImage"
export VERSION
export ICON=/usr/share/icons/hicolor/256x256/apps/iris-gui.png
export DESKTOP=/usr/share/applications/iris-gui.desktop
(cd "$WORK" && ./quick-sharun /usr/bin/iris-gui && ./quick-sharun --make-appimage)
group_end
rm -rf "$WORK"

[ -f "$OUT_DIR/$OUTNAME" ] || die "quick-sharun did not produce $OUTNAME"
log "done:"; ls -lh "$OUT_DIR" >&2
