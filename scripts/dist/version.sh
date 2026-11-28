#!/usr/bin/env bash
# Print the version a distribution build is stamped with.
#
#   version.sh nightly [YYYY-MM-DD]   nightly.YYYY.MM.DD (UTC today by default)
#   version.sh release vX.Y.Z         X.Y.Z  (accepts the tag or the bare version)
#   version.sh appstore [V]           V (vX.Y.Z, X.Y.Z or YYYYMMDD.HHMM), or the
#                                     UTC time as YYYYMMDD.HHMM, as every App
#                                     Store build has been numbered
#   version.sh dev                    dev.<short commit>
#
# The same string goes into the About window, iris/iris-ci --version, the macOS
# Info.plist, the Windows installer and every file name. deb/rpm get a leading
# "0~" when the version does not start with a digit (see pkg_version in lib.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."

RELEASE_RE='^[0-9]+\.[0-9]+\.[0-9]+$'

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

channel="${1:-}"
[ -n "$channel" ] || usage 2
shift

case "$channel" in
    nightly)
        if [ $# -gt 0 ]; then
            d="$1"
            [[ "$d" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "bad date '$d' (want YYYY-MM-DD)" >&2; exit 2; }
            echo "nightly.${d//-/.}"
        else
            date -u +nightly.%Y.%m.%d
        fi
        ;;
    release)
        [ $# -gt 0 ] || { echo "release needs a tag (vX.Y.Z)" >&2; exit 2; }
        v="${1#refs/tags/}"; v="${v#v}"
        [[ "$v" =~ $RELEASE_RE ]] || { echo "'$1' is not a vX.Y.Z release tag" >&2; exit 2; }
        echo "$v"
        ;;
    appstore)
        # App Store Connect only takes a version above the last one it
        # approved, and the App Store builds have always been numbered by
        # date (20260831.2042): a vX.Y.Z tag would now be lower.
        if [ $# -gt 0 ] && [ -n "$1" ]; then
            v="${1#v}"
            [[ "$v" =~ ^[0-9]{8}\.[0-9]{4}$ ]] && { echo "$v"; exit 0; }
            exec "$0" release "$1"
        fi
        date -u +%Y%m%d.%H%M
        ;;
    dev)
        sha=$(git rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
        echo "dev.${sha}"
        ;;
    -h|--help) usage ;;
    *) usage 2 ;;
esac
