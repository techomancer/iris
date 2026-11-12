#!/bin/bash
# Make a fixture `tests/snapshot_schemas.rs` loads.
#
#   generate.sh MACHINE-vN IRIS_BINARY
#
# IRIS_BINARY is a debug build of a revision whose src/snapshot.rs has
# SCHEMA_VERSION = N (find it with `git log -S"SCHEMA_VERSION: u32 = N"`). The machine
# runs an all-zero PROM, headless, with the smallest memory it takes, and
# saves once as `fixture`; nothing in it comes from a real PROM or guest.
# The result is written to tests/snapshots/MACHINE-vN.
set -euo pipefail
name=$1 bin=$(readlink -f "$2")
machine=${name%-v*}
case $machine in
  indy)   profile=indy_ip24;   prom=$((512 << 10)); bank=8 ;;
  o2)     profile=o2_ip32;     prom=$((512 << 10)); bank=32 ;;
  octane) profile=octane_ip30; prom=$((1 << 20));   bank=64 ;;
  *) echo "unknown machine $machine" >&2; exit 2 ;;
esac
out=$(readlink -f "$(dirname "$0")")/$name
work=$(mktemp -d)
cd "$work"
head -c $prom /dev/zero > prom.bin
# The old build prints the port it was asked for, not the one it bound, so
# this picks one.
port=$((20000 + RANDOM % 20000))
cat > iris.toml <<EOT
headless = true
no_audio = true
prom = "prom.bin"
banks = [$bank, 0, 0, 0]
monitor_port = $port
serial_port_a = $((port + 1))
serial_port_b = $((port + 2))
scsi = {}
[machine]
profile = "$profile"
[graphics]
board = "none"
EOT
"$bin" --config iris.toml > out.log 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT
sleep 5
exec 3<>/dev/tcp/127.0.0.1/$port
sleep 1
echo "save fixture" >&3
sleep 5
echo quit >&3
sleep 1
test -f saves/fixture/snapshot.toml
cp -r saves "${out:?}"
echo "wrote $out"
