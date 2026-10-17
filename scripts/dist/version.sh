#!/usr/bin/env bash
# Print the version a distribution build is stamped with.
#
#   version.sh nightly [YYYY-MM-DD]   nightly.YYYY.MM.DD (UTC today by default)
#   version.sh release vX.Y.Z         X.Y.Z  (accepts the tag or the bare version)
#   version.sh appstore [vX.Y.Z]      X.Y.Z of the given tag, or of the newest
#                                     vX.Y.Z tag reachable from HEAD
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
        if [ $# -gt 0 ] && [ -n "$1" ]; then
            exec "$0" release "$1"
        fi
        tag=$(git tag --merged HEAD --list 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname \
              | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || true)
        [ -n "$tag" ] || { echo "no vX.Y.Z tag is reachable from HEAD; tag a release first or pass a version" >&2; exit 1; }
        exec "$0" release "$tag"
        ;;
    dev)
        sha=$(git rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
        echo "dev.${sha}"
        ;;
    -h|--help) usage ;;
    *) usage 2 ;;
esac
