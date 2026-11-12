#!/bin/bash
# Give a Mac with no audio hardware a virtual loopback device, so the
# real-device audio tests have a default output to play into and an input
# to record it from (TESTING.md, Host audio). For a CI runner: it installs
# a system audio driver with sudo and changes the default devices, so don't
# run it on a Mac you use.
#
# Installs BlackHole 2ch and SwitchAudioSource from Homebrew, restarts
# coreaudiod so it loads the driver, waits for the device, and makes it the
# default output and input. BlackHole's source is GPL-3.0; the Homebrew cask
# installs the vendor's own binary build, which is copyright Existential
# Audio. This only runs it as a tool: nothing of it is linked into iris or
# shipped. Then run the tests with
# IRIS_AUDIO_LOOPBACK="default:BlackHole 2ch".
#
# Usage: audio-loopback-macos.sh

set -euo pipefail

DEVICE="BlackHole 2ch"

export HOMEBREW_NO_AUTO_UPDATE=1
brew install switchaudio-osx blackhole-2ch

# coreaudiod reads its plug-ins at start, and launchd starts it again.
sudo killall -9 coreaudiod || true
# The list goes through a variable: grep -q ends a pipe early, and under
# pipefail the writer's SIGPIPE would then fail the test.
devices() { SwitchAudioSource -a -t output 2>/dev/null || true; }
for _ in $(seq 60); do
    grep -qF "$DEVICE" <<<"$(devices)" && break
    sleep 1
done
grep -qF "$DEVICE" <<<"$(devices)" || {
    echo "audio-loopback-macos: no output device $DEVICE after 60 s; devices:" >&2
    SwitchAudioSource -a >&2 || true
    exit 1
}

SwitchAudioSource -s "$DEVICE" -t output
SwitchAudioSource -s "$DEVICE" -t input
echo "audio-loopback-macos: default output: $(SwitchAudioSource -c -t output)"
echo "audio-loopback-macos: default input: $(SwitchAudioSource -c -t input)"
if [ "$(SwitchAudioSource -c -t output)" != "$DEVICE" ] || [ "$(SwitchAudioSource -c -t input)" != "$DEVICE" ]; then
    echo "audio-loopback-macos: $DEVICE is not the default output and input" >&2
    exit 1
fi
