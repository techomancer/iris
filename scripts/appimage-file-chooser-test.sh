#!/bin/bash
# End-to-end check of the iris-gui AppImage's file choosers on Linux (see
# TESTING.md, "Linux file choosers"). For a throwaway container or session:
# it starts its own X server and D-Bus session bus.
#
#   appimage-file-chooser-test.sh <AppImage> portal|none
#
# Starts the AppImage on a fresh profile under Xvfb, clicks a 📁 button and
# checks what happens:
#
#   portal  The session bus can start xdg-desktop-portal with its GTK
#           backend. A chooser window titled "Choose CD-ROM image" must
#           appear.
#   none    The session bus offers no portal. No chooser may appear, and
#           iris-gui must say it is showing its "No file chooser" notice.
#
# zenity must not be installed: rfd falls back to it, and it would stand in
# for the portal. Exits 0 when the check passes, 1 when it fails. Leaves
# screenshots, the app's log and the portal's log in $OUT.
#
# Needs Xvfb, dbus-daemon, xdotool and ImageMagick's `import`; for `portal`
# also xdg-desktop-portal and xdg-desktop-portal-gtk.
#
# HIDE_HOST_LIBDBUS=1 moves the host's libdbus-1.so.3 aside (and back on
# exit) once the bus and the portal are running, so the AppImage can only use
# a libdbus it carries. That is the reported host: a 32-bit userland whose
# libdbus a 64-bit AppImage cannot load, with a portal running. Needs root,
# so use it in a throwaway container only.
#
# CLICK_X/CLICK_Y: the button, in window coordinates; CHOOSER_TITLE: the
# title its chooser gets. The defaults are the CD-ROM row's 📁 button in the
# "New machine" dialog a fresh profile opens with, at the window size this
# script sets: picking a CD image is what techomancer/iris#457 reported.

set -euo pipefail

APPIMAGE=$(realpath "${1:?usage: $0 <AppImage> portal|none}")
MODE=${2:?usage: $0 <AppImage> portal|none}
OUT=$(realpath -m "${OUT:-appimage-file-chooser-out}")
CLICK_X=${CLICK_X:-847}
CLICK_Y=${CLICK_Y:-616}
CHOOSER_TITLE=${CHOOSER_TITLE:-Choose CD-ROM image}
NOTICE_LOG="no file chooser could be opened; showing the notice"
HIDE_HOST_LIBDBUS=${HIDE_HOST_LIBDBUS:-0}

case "$MODE" in portal|none) ;; *) echo "mode must be portal or none" >&2; exit 2 ;; esac
for cmd in Xvfb dbus-daemon xdotool import; do
    command -v "$cmd" >/dev/null || { echo "missing: $cmd" >&2; exit 2; }
done
if command -v zenity >/dev/null; then
    echo "zenity is installed; it would stand in for the portal" >&2
    exit 2
fi

mkdir -p "$OUT"
WORK=$(mktemp -d)
PIDS=()
HIDDEN=()
cleanup() {
    # No machine is ever started, so stopping the app cannot hurt a guest. It
    # may be blocked inside a dialog call, hence the KILL after TERM.
    for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
    sleep 1
    for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null || true; done
    wait 2>/dev/null || true
    for f in "${HIDDEN[@]}"; do mv -f "$f.hidden-by-test" "$f"; done
    rm -rf "$WORK"
}
trap cleanup EXIT

export HOME="$WORK/home" XDG_RUNTIME_DIR="$WORK/run" XDG_CONFIG_HOME="$WORK/home/.config"
mkdir -p "$HOME" "$XDG_RUNTIME_DIR" "$XDG_CONFIG_HOME/xdg-desktop-portal"
chmod 700 "$XDG_RUNTIME_DIR"
# Pick the GTK backend whatever desktop the portal thinks this is.
printf '[preferred]\ndefault=gtk\n' > "$XDG_CONFIG_HOME/xdg-desktop-portal/portals.conf"
export XDG_CURRENT_DESKTOP=GNOME GDK_BACKEND=x11 APPIMAGE_EXTRACT_AND_RUN=1

export DISPLAY=:57
Xvfb "$DISPLAY" -screen 0 1600x1200x24 -nolisten tcp >"$OUT/xvfb.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 50); do xdotool getdisplaygeometry >/dev/null 2>&1 && break; sleep 0.1; done

# A session bus of our own. Its service directory decides whether a portal
# exists: the system's for `portal`, an empty one for `none`.
SERVICES="$WORK/no-services"
mkdir -p "$SERVICES"
[ "$MODE" = portal ] && SERVICES=/usr/share/dbus-1/services
cat > "$WORK/bus.conf" <<EOF
<!DOCTYPE busconfig PUBLIC "-//freedesktop//DTD D-BUS Bus Configuration 1.0//EN"
 "http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">
<busconfig>
  <type>session</type>
  <listen>unix:path=$XDG_RUNTIME_DIR/bus</listen>
  <servicedir>$SERVICES</servicedir>
  <policy context="default">
    <allow send_destination="*" eavesdrop="true"/>
    <allow eavesdrop="true"/>
    <allow own="*"/>
  </policy>
</busconfig>
EOF
# Services the bus starts inherit its environment, DISPLAY included.
dbus-daemon --config-file="$WORK/bus.conf" --nofork >"$OUT/portal.log" 2>&1 &
PIDS+=($!)
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
for _ in $(seq 50); do [ -S "$XDG_RUNTIME_DIR/bus" ] && break; sleep 0.1; done

if [ "$HIDE_HOST_LIBDBUS" = 1 ]; then
    # The bus and the GTK backend link libdbus, so start the portal first: a
    # running process keeps the copy it mapped.
    if [ "$MODE" = portal ]; then
        dbus-send --session --print-reply --dest=org.freedesktop.portal.Desktop \
            /org/freedesktop/portal/desktop org.freedesktop.DBus.Peer.Ping >/dev/null
        for _ in $(seq 100); do
            dbus-send --session --print-reply --dest=org.freedesktop.DBus / \
                org.freedesktop.DBus.NameHasOwner string:org.freedesktop.impl.portal.desktop.gtk \
                | grep -q 'boolean true' && break
            sleep 0.1
        done
    fi
    for f in $(ldconfig -p | awk '/libdbus-1\.so/ {print $NF}' | sort -u); do
        f=$(realpath "$f")
        mv "$f" "$f.hidden-by-test"
        HIDDEN+=("$f")
    done
    echo "hid the host's libdbus: ${HIDDEN[*]:-(none found)}"
fi

"$APPIMAGE" >"$OUT/app.log" 2>&1 &
APP=$!
PIDS+=("$APP")

WID=$(timeout 120 xdotool search --sync --onlyvisible --name '^IRIS$' | head -1) || {
    echo "FAIL: the IRIS window never appeared" >&2
    tail -20 "$OUT/app.log" >&2
    exit 1
}
xdotool windowsize "$WID" 1280 900 windowmove "$WID" 0 0
sleep 5    # let egui lay out a few frames at the new size
import -window root "$OUT/$MODE-1-start.png"

xdotool mousemove --window "$WID" "$CLICK_X" "$CLICK_Y" sleep 0.3 click 1

result=1
if [ "$MODE" = portal ]; then
    if timeout 30 xdotool search --sync --onlyvisible --name "^$CHOOSER_TITLE\$" >/dev/null; then
        echo "PASS: the portal opened the file chooser"
        result=0
    else
        echo "FAIL: no file chooser appeared within 30 s" >&2
    fi
    sleep 2
    import -window root "$OUT/$MODE-2-after-click.png"
    xdotool key Escape || true
else
    for _ in $(seq 300); do
        grep -qF "$NOTICE_LOG" "$OUT/app.log" && { result=0; break; }
        sleep 0.1
    done
    sleep 2
    import -window root "$OUT/$MODE-2-after-click.png"
    if xdotool search --onlyvisible --name "^$CHOOSER_TITLE\$" >/dev/null 2>&1; then
        echo "FAIL: a file chooser appeared, but the bus has no portal" >&2
        result=1
    elif [ "$result" = 0 ]; then
        echo "PASS: no chooser, and iris-gui showed its notice"
    else
        echo "FAIL: no chooser, and no notice within 30 s" >&2
    fi
fi

echo "--- rfd and iris-gui lines from the app's log:"
grep -E 'rfd|file chooser' "$OUT/app.log" || echo "(none)"
exit "$result"
