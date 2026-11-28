#!/usr/bin/env bash
# o2-ci-rollback.sh — end-to-end check of iris-ci restore, rollback and CD
# changes on the O2 (issue #305), on the built-in PROM; no IRIX needed.
#
# Boots a headless O2 into the PROM's command monitor over the CI console,
# then checks, from what the PROM prints:
#   1. restore: a PROM variable set after `save` is gone after `restore`;
#   2. rollback: a variable set after that `restore` is gone after `rollback`;
#   3. CD change: `ls` of the CD's volume header names the file on disc A,
#      and after `cdrom-eject` (the changer moves to disc B) the file on B;
#   4. scratch volume: the PROM's `hinv` lists it, and bytes written with
#      `scratch write` read back unchanged with `scratch read`.
# The two CD images are SGI volume headers this script makes with mkvh.
#
# Usage: tools/tests/o2-ci-rollback.sh [--bin DIR] [--port BASE] [--keep]
#   --bin DIR    directory holding iris, iris-ci and mkvh (default target/release)
#   --port BASE  monitor on BASE, serial ports on BASE+1 and BASE+2 (default 23050)
#   --keep       leave the work directory behind
# Exit status: 0 when every check passes, 1 when one fails.

set -u
here="$(cd "$(dirname "$0")/../.." && pwd)"
bin="$here/target/release"
base=23050
keep=0
while [ $# -gt 0 ]; do
    case "$1" in
        --bin) bin="$(cd "$2" && pwd)"; shift 2 ;;
        --port) base="$2"; shift 2 ;;
        --keep) keep=1; shift ;;
        *) echo "usage: $0 [--bin DIR] [--port BASE] [--keep]" >&2; exit 2 ;;
    esac
done
for b in iris iris-ci mkvh; do
    [ -x "$bin/$b" ] || { echo "missing $bin/$b (cargo build --release --bin $b)" >&2; exit 2; }
done

work="$(mktemp -d "${TMPDIR:-/tmp}/o2-ci-rollback.XXXXXX")"
sock="$work/ci.sock"
ci() { "$bin/iris-ci" --socket "$sock" "$@"; }
pid=
fails=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; fails=$((fails + 1)); }

finish() {
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        ci quit >/dev/null 2>&1
        for _ in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
        # Only the PROM runs here, so there is no guest file system to hurt.
        kill -0 "$pid" 2>/dev/null && kill "$pid"
    fi
    if [ "$keep" = 1 ] || [ "$fails" -gt 0 ]; then echo "work directory: $work"; else rm -rf "$work"; fi
}
trap finish EXIT

# Send a PROM command and return what it printed up to the next prompt
# (the O2's command monitor prompts with "> ").
prom() {
    ci serial-read >/dev/null 2>&1
    ci serial-send "$1" >/dev/null || return 1
    ci serial-wait --timeout "${2:-60}" "> " | tr -d '\r'
}

cd "$work" || exit 2
printf 'disc a\n' > a.txt
printf 'disc b\n' > b.txt
"$bin/mkvh" build disca.img --size 4M DISCA=a.txt >/dev/null || exit 2
"$bin/mkvh" build discb.img --size 4M DISCB=b.txt >/dev/null || exit 2

cat > iris.toml <<EOF
headless = true
ci = true
ci_socket = "$sock"
monitor_port = $base
devices = [ { type = "serial", at = "serial1", backend = "tcp:$((base + 1))" }, { type = "serial", at = "serial2", backend = "tcp:$((base + 2))" } ]
nvram = "flash.bin"
serial_log = "serial.log"

[machine]
profile = "o2_ip32"

[scsi.2]
path = "scratch.img"
scratch = true
size_mb = 8

[scsi.4]
path = "disca.img"
discs = ["discb.img"]
cdrom = true
overlay = true
EOF

"$bin/iris" --config iris.toml > iris.log 2>&1 &
pid=$!
for _ in $(seq 300); do [ -S "$sock" ] && break; kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
ci status --wait-ready 60 >/dev/null || { fail "iris did not come up (see $work/iris.log)"; exit 1; }
ci start >/dev/null || { fail "iris-ci start"; exit 1; }

# Reach the command monitor from the maintenance menu.
if ! ci serial-wait --timeout 120 "Option?" >/dev/null; then
    fail "the PROM never showed its maintenance menu"; exit 1
fi
ci serial-send "5" >/dev/null
ci serial-wait --timeout 60 "> " >/dev/null || { fail "no command monitor prompt"; exit 1; }

prom "setenv E2EMARK base" >/dev/null
ci save e2e/base >/dev/null || { fail "save"; exit 1; }

# 1. restore
prom "setenv E2EMARK after-save" >/dev/null
if ci restore e2e/base; then
    out="$(prom "printenv E2EMARK")"
    case "$out" in
        *E2EMARK=base*) pass "restore brings back the saved PROM variable" ;;
        *) fail "restore: printenv shows: $out" ;;
    esac
else
    fail "iris-ci restore returned an error"
fi

# 2. rollback
prom "setenv E2EMARK after-restore" >/dev/null
if ci rollback; then
    out="$(prom "printenv E2EMARK")"
    case "$out" in
        *E2EMARK=base*) pass "rollback returns to the last restore" ;;
        *) fail "rollback: printenv shows: $out" ;;
    esac
else
    fail "iris-ci rollback returned an error"
fi

# 3. CD change
cd_ls() { prom "ls pci(0)scsi(0)cdrom(4)partition(8)" 60; }
out="$(cd_ls)"
case "$out" in
    *DISCA*) pass "the PROM lists disc A's volume header" ;;
    *) fail "disc A: ls shows: $out" ;;
esac
if ci cdrom-eject 4; then
    out="$(cd_ls)"
    case "$out" in
        *DISCB*) pass "after cdrom-eject the PROM lists disc B" ;;
        *) fail "disc B: ls shows: $out" ;;
    esac
else
    fail "iris-ci cdrom-eject returned an error"
fi
if ci cdrom-load 4 disca.img; then
    out="$(cd_ls)"
    case "$out" in
        *DISCA*) pass "after cdrom-load the PROM lists disc A again" ;;
        *) fail "disc A again: ls shows: $out" ;;
    esac
else
    fail "iris-ci cdrom-load returned an error"
fi

# 4. scratch volume (a file this script made, so it needs no overlay)
out="$(prom "hinv")"
case "$out" in
    *"disk(2)"*) pass "the PROM's hinv lists the scratch disk" ;;
    *) fail "scratch: hinv shows: $out" ;;
esac
head -c 65536 /dev/urandom > payload.bin
if ci scratch write payload.bin >/dev/null && ci scratch read back.bin --length 65536 >/dev/null \
        && cmp -s payload.bin back.bin; then
    pass "scratch write and read round-trip 64 KiB"
else
    fail "scratch write/read did not round-trip"
fi

[ "$fails" -eq 0 ] && echo "all checks passed" || echo "$fails check(s) failed"
[ "$fails" -eq 0 ]
