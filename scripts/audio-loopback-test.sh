#!/bin/bash
# Run the real-device audio tests through a PulseAudio null sink: cards play
# into the host's default output and the tests record it back from the
# sink's monitor (see TESTING.md). For a machine with no sound card, such as
# a CI runner or a container.
#
# Starts a PulseAudio server if none answers, makes a null sink the default
# output and its monitor the default input, and checks that noise played
# into the sink comes back from the monitor before running anything: a
# broken loopback fails here, not as an iris bug. Then runs the command
# once per output host in IRIS_AUDIO_HOSTS (`default`, which is PulseAudio
# on Linux, unless set), with IRIS_AUDIO_LOOPBACK naming it.
# A server the script started keeps running.
#
# Needs pulseaudio and pulseaudio-utils (pactl, pacat, parec).
# Usage: audio-loopback-test.sh <test command...>
#   audio-loopback-test.sh cargo nextest run -p iris --lib -E 'test(/on_the_real_device$/)'

set -euo pipefail

SINK=${IRIS_TEST_SINK:-iris_sink}

if ! pactl info >/dev/null 2>&1; then
    pulseaudio --start --exit-idle-time=-1
    for _ in $(seq 50); do
        pactl info >/dev/null 2>&1 && break
        sleep 0.1
    done
fi
pactl info | grep -E '^Server (Name|Version|String)'
# Name the server's socket outright: pactl finds it in more places than
# iris's PulseAudio client may look, and a container may set neither
# PULSE_SERVER nor XDG_RUNTIME_DIR.
server=$(pactl info | sed -n 's/^Server String: //p')
case $server in
    /*) export PULSE_SERVER="unix:$server" ;;
    *) export PULSE_SERVER="$server" ;;
esac

if ! pactl list short sinks | cut -f2 | grep -qxF "$SINK"; then
    pactl load-module module-null-sink "sink_name=$SINK" "sink_properties=device.description=$SINK" >/dev/null
fi
pactl set-default-sink "$SINK"
pactl set-default-source "$SINK.monitor"
pactl list short sources | cut -f2 | grep -qxF "$SINK.monitor" || {
    echo "audio-loopback-test: no source $SINK.monitor" >&2
    exit 1
}

# One second of noise into the sink must come back from the monitor.
check=$(mktemp)
trap 'rm -f "$check"' EXIT
# 1.5 s of the monitor, cut by size: parec loses what it buffered when a
# signal stops it.
timeout 5 sh -c "parec -d '$SINK.monitor' --raw --format=s16le --rate=48000 --channels=2 --latency-msec=20 | head -c 288000" >"$check" &
rec=$!
sleep 0.3
head -c 192000 /dev/urandom | pacat -d "$SINK" --raw --format=s16le --rate=48000 --channels=2 --latency-msec=20
wait "$rec" || true
heard=$(tr -d '\0' <"$check" | wc -c)
echo "audio-loopback-test: the monitor gave $(wc -c <"$check") bytes, $heard of them not zero"
if [ "$heard" -lt 48000 ]; then
    echo "audio-loopback-test: the null sink's monitor does not hear what plays into it" >&2
    exit 1
fi

status=0
for host in ${IRIS_AUDIO_HOSTS:-default}; do
    echo "== IRIS_AUDIO_LOOPBACK=$host:$SINK.monitor"
    IRIS_AUDIO_LOOPBACK="$host:$SINK.monitor" "$@" || status=1
done
exit $status
