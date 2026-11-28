#!/usr/bin/env bash
# Print the version a distribution build is stamped with.
#
#   version.sh nightly [YYYY-MM-DD]   nightly.YYYY.MM.DD (UTC today by default)
#   version.sh release vX.Y.Z         X.Y.Z  (accepts the tag or the bare version)
#   version.sh appstore [V]           V (vX.Y, X.Y, vX.Y.Z or X.Y.Z), or the
#                                     one in scripts/dist/appstore-version
#   version.sh appstore-build RUN     the App Store build number for
#                                     pipeline run RUN: 20261129.RUN
#   version.sh dev                    dev.<short commit>
#
# The same string goes into the About window, iris/iris-ci --version, the macOS
# Info.plist, the Windows installer and every file name. deb/rpm get a leading
# "0~" when the version does not start with a digit (see pkg_version in lib.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."

RELEASE_RE='^[0-9]+\.[0-9]+\.[0-9]+$'
APPSTORE_RE='^[0-9]+\.[0-9]+(\.[0-9]+)?$'
# The App Store build number's first part. A Mac build number must be above
# every build ever uploaded for the app, whatever its version; the last ones
# were dated (20261128.156), so the run number goes after this.
APPSTORE_BUILD_FLOOR=20261129

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

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
        # The version App Store Connect shows (CFBundleShortVersionString).
        # It stays the same for every TestFlight build of a release, so only
        # the first build of each version waits for Beta App Review; change
        # it when the next release starts.
        if [ $# -gt 0 ] && [ -n "$1" ]; then
            v="${1#v}"
        else
            v=$(tr -d '[:space:]' < scripts/dist/appstore-version)
        fi
        [[ "$v" =~ $APPSTORE_RE ]] || { echo "'$v' is not an App Store version (X.Y or X.Y.Z)" >&2; exit 2; }
        echo "$v"
        ;;
    appstore-build)
        # CFBundleVersion: new on every upload, higher than every earlier one.
        [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]] || { echo "appstore-build needs the pipeline's run number" >&2; exit 2; }
        echo "${APPSTORE_BUILD_FLOOR}.$1"
        ;;
    dev)
        sha=$(git rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
        echo "dev.${sha}"
        ;;
    -h|--help) usage ;;
    *) usage 2 ;;
esac
