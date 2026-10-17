#!/usr/bin/env bash
# Build and package IRIS for Linux: iris-gui as deb/rpm/Arch pkg (and an
# optional plain tarball), plus the CLI tarball holding iris and iris-ci.
#
#   scripts/dist/package-linux.sh [options] [--formats deb,rpm,pkg,tar]
#
# Cross builds (any --target whose arch differs from the host) strip with
# <triple>-strip and read the rpm's library requirements with <triple>-objdump,
# since the host tools cannot read a foreign ELF. They expect the cross linker
# and pkg-config environment to be set up already (see build.yml's riscv64 job).
#
# Needs: cargo-deb, cargo-generate-rpm, dpkg-deb, rpm, zstd.
source "$(dirname "$0")/lib.sh"

FORMATS="deb,rpm,pkg"
parse_common_args "$@"
set -- ${REST[@]+"${REST[@]}"}
while [ $# -gt 0 ]; do
    case "$1" in
        --formats) FORMATS="$2"; shift 2 ;;
        -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; common_usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$TARGET" in
    x86_64-*)  ARCH=x64;     DEB_ARCH=amd64;   PKG_ARCH=x86_64 ;;
    aarch64-*) ARCH=arm64;   DEB_ARCH=arm64;   PKG_ARCH=aarch64 ;;
    riscv64*)  ARCH=riscv64; DEB_ARCH=riscv64; PKG_ARCH=riscv64 ;;
    *) die "unsupported Linux target $TARGET" ;;
esac

HOST=$(host_target)
CROSS=0
[ "${HOST%%-*}" = "${TARGET%%-*}" ] || CROSS=1
if [ "$CROSS" = 1 ]; then
    TRIPLE_PREFIX="${TARGET%%-*}"; TRIPLE_PREFIX="${TRIPLE_PREFIX%gc}-linux-gnu"
    STRIP="${TRIPLE_PREFIX}-strip"
    OBJDUMP="${TRIPLE_PREFIX}-objdump"
else
    STRIP="strip"
    OBJDUMP="objdump"
fi

GUI_FEATURES=$(with_bundled "$GUI_FEATURES")
build_gui "$TARGET" "$GUI_FEATURES"
build_cli "$TARGET" "$CLI_FEATURES"

REL=$(release_dir "$TARGET")
GUI_BIN="$REL/iris-gui"
[ -x "$GUI_BIN" ] || die "$GUI_BIN was not built"

# The release profile keeps debug info. Strip here, with a strip that can read
# this ELF, and tell the packagers not to strip again.
"$STRIP" "$GUI_BIN" "$REL/iris" "$REL/iris-ci"

PKGVER=$(pkg_version "$VERSION")
BASE="IRIS-gui-linux-${ARCH}-${VERSION}"

has_format() { case ",$FORMATS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

if has_format deb; then
    group_start "deb"
    need cargo-deb dpkg-deb
    cargo deb -p iris-gui --target "$TARGET" --no-build --no-strip \
        --deb-version "$PKGVER" -o "$OUT_DIR/$BASE.deb"
    dpkg-deb -f "$OUT_DIR/$BASE.deb" Package Version Architecture Depends
    [ "$(dpkg-deb -f "$OUT_DIR/$BASE.deb" Architecture)" = "$DEB_ARCH" ] \
        || die "$BASE.deb is not Architecture: $DEB_ARCH"
    # cargo-deb only warns when $auto cannot resolve the shared-library deps,
    # and then ships an empty Depends. Catch that here.
    DEPENDS=$(dpkg-deb -f "$OUT_DIR/$BASE.deb" Depends)
    case "$DEPENDS" in *libc6*) ;; *) die "deb Depends did not resolve (no libc6): $DEPENDS" ;; esac
    case "$DEPENDS" in *-cross*) die "deb Depends names a cross-toolchain package: $DEPENDS" ;; esac
    group_end
fi

if has_format rpm; then
    group_start "rpm"
    need cargo-generate-rpm rpm
    META="version = \"$PKGVER\""
    RPM_ARGS=()
    if [ "$CROSS" = 1 ]; then
        # Auto-requires runs the host ldd, which cannot load a foreign ELF, so
        # list the soname()(64bit) requirements from the cross objdump instead.
        REQS=$("$OBJDUMP" -p "$GUI_BIN" \
            | awk '$1 == "NEEDED" { printf "%s\"%s()(64bit)\" = \"*\"", sep, $2; sep = ", " }')
        [ -n "$REQS" ] || die "no NEEDED entries in $GUI_BIN"
        META="$META
requires = { $REQS }"
        RPM_ARGS+=(--auto-req disabled)
    fi
    rm -rf "target/$TARGET/generate-rpm"
    cargo generate-rpm -p iris-gui --target "$TARGET" --arch "$PKG_ARCH" \
        ${RPM_ARGS[@]+"${RPM_ARGS[@]}"} --set-metadata "$META"
    mv "target/$TARGET/generate-rpm/"*.rpm "$OUT_DIR/$BASE.rpm"
    rpm -qp --qf '%{NAME} %{VERSION} %{ARCH}\n' "$OUT_DIR/$BASE.rpm"
    [ "$(rpm -qp --qf '%{ARCH}' "$OUT_DIR/$BASE.rpm")" = "$PKG_ARCH" ] \
        || die "$BASE.rpm is not arch $PKG_ARCH"
    group_end
fi

if has_format pkg; then
    group_start "Arch pkg"
    need zstd
    S=$(stage_dir)
    install -Dm755 "$GUI_BIN" "$S/usr/bin/iris-gui"
    for size in 16 32 48 64 128 256; do
        install -Dm644 "iris-gui/assets/icons/icon-${size}.png" \
            "$S/usr/share/icons/hicolor/${size}x${size}/apps/iris-gui.png"
    done
    install -Dm644 iris-gui/iris-gui.desktop "$S/usr/share/applications/iris-gui.desktop"
    install -Dm644 LICENSE "$S/usr/share/licenses/iris-gui/LICENSE"
    install -Dm644 LICENSE-libchdman-rs.txt "$S/usr/share/licenses/iris-gui/LICENSE-libchdman-rs.txt"
    # pacman's pkgver may not contain '-' or ':'; ours never do.
    cat > "$S/.PKGINFO" <<EOF
pkgname = iris-gui
pkgbase = iris-gui
pkgver = ${VERSION}-1
pkgdesc = SGI Indy and Indigo2 emulator
url = https://github.com/techomancer/iris
builddate = $(date -u +%s)
packager = IRIS release builds
size = $(du -sb "$S/usr" | cut -f1)
arch = ${PKG_ARCH}
license = BSD-3-Clause
depend = gtk3
depend = alsa-lib
depend = libxkbcommon
EOF
    (cd "$S" && tar --zstd -cf "$OUT_DIR/$BASE.pkg.tar.zst" .PKGINFO usr)
    rm -rf "$S"
    group_end
fi

if has_format tar; then
    group_start "GUI tarball"
    S=$(stage_dir)
    cp "$GUI_BIN" "$S/iris-gui"
    cp iris-gui/iris-gui.desktop iris-gui/assets/icons/icon-256.png LICENSE LICENSE-libchdman-rs.txt "$S/"
    tar -C "$S" -czf "$OUT_DIR/$BASE.tar.gz" iris-gui iris-gui.desktop icon-256.png LICENSE LICENSE-libchdman-rs.txt
    rm -rf "$S"
    group_end
fi

group_start "CLI tarball"
CLI="$OUT_DIR/IRIS-cli-linux-${ARCH}-${VERSION}.tar.gz"
S=$(stage_dir)
cp "$REL/iris" "$REL/iris-ci" LICENSE LICENSE-libchdman-rs.txt "$S/"
tar -C "$S" -czf "$CLI" iris iris-ci LICENSE LICENSE-libchdman-rs.txt
rm -rf "$S"
verify_members "$CLI" iris iris-ci LICENSE LICENSE-libchdman-rs.txt
group_end

log "done:"; ls -lh "$OUT_DIR" >&2
