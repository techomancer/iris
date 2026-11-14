# Agent instructions: iris-audio

This file is the handoff between agent sessions working on this repository
from different machines (Linux, macOS, Windows). Read it first, then
`DESIGN.md` (why the crate is shaped as it is, and the backend contract),
then `README.md` (use, status, measurements).

## What this is

A Rust crate that plays an emulator's sound: a guest device pushes frames at
its own rate and pace, the crate gets them to the host's speakers with low
latency and no clicks. It was started for IRIS, an SGI workstation emulator,
whose current audio (cpal plus a per-device resampler) crackles on Linux and
Windows and does not follow a change of the default output device on
Windows. It is meant to be reusable by other emulators.

It began on an `iris-audio` branch of the IRIS repository and was copied here
to be developed on its own. IRIS uses it since 2026-11-13: `src/host_audio.rs`
there opens the streams for HAL2, RAD1, MACE audio and the ICE1724, as a path
dependency on this directory.

## Decisions already made (by the project owner; do not reopen)

- **Build from the bottom.** Own backends on each host's native API, not a
  wrapper over cpal, SDL, miniaudio or cubeb.
- **IRIS is not touched** until this crate works on Linux, Windows and macOS.
- **The priority is low-latency output fed asynchronously by the guest.** It
  must serve a device that trickles a few frames per interrupt and one that
  writes large blocks. Prebuffering is configurable per stream.
- **Input is accommodated, not implemented.** Keep the input interface
  compiling and coherent; do not build it until asked.
- **Licence: BSD 3-Clause.** New dependencies must be compatible with it.
- **Linux needs a pure ALSA backend eventually**, for a machine that boots
  straight into the emulator with no sound server.
- **A native PipeWire backend is wanted** beside the Pulse-protocol one. The
  owner had native PipeWire in mind when asking for the prototype.
- **The crate has its own rate converter** (`src/resample.rs`). A block-based
  library resampler was set aside because it needs a buffer in front of it,
  which is latency. The owner has seen this choice and not objected; do not
  swap it without asking.

## State

Done and tested on Linux (see the README's status table and measurements):
the stream engine, the file backend, the tee, the PulseAudio protocol
backend, the `tone` example.

Done on Windows: the `wasapi` backend (`src/backend/wasapi.rs`) and the
`play` example, which plays until Enter is pressed and reports device
changes. Checklist steps 1 to 5 were run silently and briefly (5 to 10 s
each); see the README's measurements.

Done everywhere: the `null` backend (`src/backend/null.rs`), which discards
in real time. Its clock, `Pacer`, is what a device backend uses to keep a
stream consumed while it has no device; `coreaudio` and `wasapi` do so.
`pulse` does the same while it has no server.
`OutputConfig::null_fallback` (`OutputRequest::null_fallback`) asks a
backend to open a stream even with no device and move it to one later, so
the program above never reopens; `coreaudio`, `wasapi` and `pulse` honour
it. `PulseBackend::new` no longer fails when there is no server: the first
`open_output` does, unless the fallback was asked for.
`PulseBackend::available()` tells beforehand, for choosing between backends,
and `.server(path)` names a socket.

`pulse` is built on every Unix and is `BackendChoice::Default` on all of
them except macOS, so FreeBSD has it. It has never been compiled or run
there: this host has no cross toolchain.

Done on macOS: the `coreaudio` backend (`src/backend/coreaudio.rs`).
Checklist steps 1 to 5 and 8 were run silently; see the README's
measurements. Steps 6 and 7 have not been done with a real device: closing,
discarding and reopening have only run with the loss of the device
simulated in the code.

Not started: `pipewire`, `alsa`. Input is interface only.

Watched by the owner on Windows: an audible tone from `play` following
switches of the default output device between active outputs.

Never tested anywhere: a default-device switch while playing on the Pulse
backend; a real sound server restart on Pulse (only a relay socket was
taken away, see the notes); anything on a BSD; a device unplugged or disabled on
WASAPI; on CoreAudio, a default-device switch, a device unplugged, and a
change of the device's rate (Audio MIDI Setup) while playing. The code is
written for all of them; nobody has watched it happen.
`examples/play` exists for watching. Audible output has not been listened to either: every
run on a real device so far played silence (`--amp 0`).

## Build and test

```sh
cargo test
cargo run --release --example tone -- --backend file:out.wav
cargo run --release --example tone -- --amp 0          # default backend, silent
```

Plain stable Rust with edition 2024; nothing needs nightly. The crate builds
on every platform today: where no native backend exists yet,
`BackendChoice::Default` returns `Error::Unsupported` and the file backend
still works. Run `cargo test` first on a new machine: the stream tests are
simulated and must pass everywhere; the file backend test uses the real
clock.

## Writing a backend

The contract is in `DESIGN.md` ("The backend contract") and `src/backend.rs`.
`src/backend/pulse.rs` is the model for a backend with its own thread. In
short: one module `src/backend/<name>.rs`, gated by target and by a Cargo
feature if it links a system library; one type implementing `Backend`; wire
it into `BackendChoice` and `Audio::named` in `src/lib.rs` and make it the
platform's `Default`.

Rules that are easy to break:

- The thread that serves the device calls `Render::render` and nothing that
  blocks, locks or allocates. Keep logging, device queries and reopening off
  it, or do them only when the stream is already broken.
- Call `configure` before the first `render` and after every format change.
- The first open happens inside `open_output`, so its failure is returned to
  the caller.
- Dropping the handle stops the stream and joins the thread, also while the
  backend is waiting to reconnect.
- The library prints nothing. Report through `Error`, `RenderInfo` and
  `StreamHandle::device`.

### Windows (WASAPI): what is there, and what to keep

The backend is written; its module documentation says how it works. Things
to keep when changing it:

- The IRIS project had a rule from a past crash: window and device calls
  belong on the thread that owns them. Every WASAPI call for a stream is
  made on that stream's thread. The `IMMNotificationClient` callback runs on
  a system thread and only compares device names, sets a flag and an event.
- The thread fills its queue to a target (two periods, or the requested
  latency if longer), not to `buffer size - padding`: filling the whole
  buffer would make the buffer's size the latency.
- `IAudioClient3` is tried first for a short period and only with the mix
  format; any failure falls back to `IAudioClient::Initialize` on a freshly
  activated client.
- The `windows` crate's `implement` macro generates code that names the
  `windows_core` crate, which is why `windows-core` is a direct dependency.
- While there is no device the stream's thread calls `render` itself
  (`idle`, with `null::Pacer`), in the last configured format, so nothing
  is reconfigured if the same kind of device comes back. No stream is open
  then, which is what makes it safe. It waits on the signal event, not on
  the `changed` flag: the flag can stay set while opening keeps failing.
  The first open still fails with no device unless the request has
  `null_fallback`; then the stream starts in `idle` at the guest's rate and
  channels. Any failure of the first open is treated that way, not only a
  missing endpoint, as on CoreAudio.

### macOS (CoreAudio): what is there, and what to keep

The backend is written; its module documentation says how it works. Things
to keep when changing it:

- It is a HAL output unit (`kAudioUnitSubType_HALOutput`) on purpose, not
  the default-output unit: that one follows the default device by itself
  but converts behind the client's back when the new device has another
  rate. The backend listens and reopens instead, so `configure` is called.
- CoreAudio is declared by hand in the module's `sys`; no crate. Only APIs
  from Mac OS X 10.6 or earlier are used (the test machine is from 2015 and
  Rust's deployment target is older than its macOS). Check a new call's
  availability before adding it.
- Every call that changes the unit is made on the stream's thread. The
  property listeners run on a CoreAudio thread and only set a flag and wake
  it; the thread then checks (`Session::current`) whether anything that
  matters changed, because most notifications are of no consequence.
- Listeners are registered with a number, not a pointer, and look their
  stream up in `STREAMS`: CoreAudio does not promise that a listener is not
  still running when its removal returns.
- The render callback owns the `Feed` from `AudioOutputUnitStart` to
  `AudioOutputUnitStop`; the thread touches it only in between. The
  callback is the device's real-time thread: nothing but `render` there.
- While there is no device the stream's thread calls `render` itself
  (`idle`, with `null::Pacer`), in the last configured format, so nothing
  is reconfigured if the same kind of device comes back. This is safe only
  because no unit is running then; never call it with a session alive. The
  first open still fails with no device, as the contract says, unless the
  request has `null_fallback`; then the stream starts in `idle` at the
  guest's rate and channels.
- The buffer's time stamp already includes the device's safety offset
  (measured: host time minus now is buffer plus safety offset), so only the
  device's and the stream's latency are added to it for `device_delay`.
- `CoreAudioBackend::new` sets `kAudioHardwarePropertyRunLoop` to null so
  that notifications arrive without a run loop on the main thread. It is a
  setting of the whole process.

### Linux: what is left

- `pipewire`: native client behind a feature (it links the PipeWire client
  library). Goal: set the node's quantum directly and run on PipeWire's
  real-time loop. On the reference host pipewire-pulse never asks for fewer
  than 256 frames.
- `alsa`: `default` or a named PCM, a thread sleeping in `poll`, own sample
  format conversion. A previous ALSA client on this project spun a core
  because the PCM's descriptor stayed readable below the wake-up threshold;
  check the revents through the ALSA call meant for it instead of trusting
  `poll` alone.

## Checklist for a new backend or a new host

Run each with `examples/tone` and record the numbers in the README's
"Measured" section (host, OS version, sound system, device rate):

1. `--burst 4` for 30 s: underruns, settled queue, reported device delay.
2. `--burst 1024 --rate 48000 --prebuffer 30`.
3. `--drift 2000` and `--drift -2000`: the adjustment settles near ±0.2%.
4. `--jitter 30`: one underrun per stall and a clean recovery.
5. `--rate 11025`: converted; listen for a metallic edge (there should be
   none).
6. Switch the default output device while it plays.
7. Unplug or disable the device, or restart the sound server, and bring it
   back.
8. The lowest `--prebuffer` and `--latency` that run clean for a minute.

## Working rules

- **Sound on someone's machine.** Automated runs use `--amp 0`. Play an
  audible tone only when the person at the machine has asked for it or agreed
  to it, and keep it quiet (`--amp 0.1`). Do not change the user's default
  device or restart their sound server without asking; steps 6 and 7 of the
  checklist need the user.
- **Documents for people** (`README.md`, `DESIGN.md`, module documentation)
  describe the crate as it is, in prose a newcomer can follow: no session
  logs, no "I tried X". Update them in the same change as the behaviour.
- **This file** is for agents. When a session ends with something the next
  one needs (a decision, a trap, an unfinished step), put it here under
  "Notes between sessions", dated, and remove notes that stop being true.
- **Tests** are named as sentences stating the behaviour
  (`a_sound_shorter_than_the_prebuffer_still_plays`). Stream behaviour is
  tested in simulated time through `Output::detached`; add a case there
  before reaching for a real device.
- **Comments** say why, in full sentences. Constants that tune behaviour are
  named and explained at the top of their module.
- Commit only when asked. Do not push anywhere unless asked.

## Notes between sessions

- 2026-11-13, Linux: `Audio::open_outputs` and `src/mix.rs` (`Mixer`) added
  for IRIS's O2 audio, which has two DACs at different rates. The voice set
  is fixed at open on purpose; adding a voice to a running stream would need
  a lock-free handover to the device's thread and nobody needs it yet.
  `Output` now holds its stream handle in an `Arc<Mutex<..>>` shared by the
  voices. Tested in simulated time and through the file backend from IRIS;
  not run on WASAPI or CoreAudio, though no backend code changed.

- 2026-11-13, Linux: repository created from the IRIS branch. On the
  reference host (PipeWire 1.6.8, 48 kHz) the Pulse backend runs clean at the
  default 10 ms prebuffer; total delay is about 40 ms, of which 21 ms is what
  the server reports for the sink. The owner's next wish on Linux is the
  native PipeWire backend; macOS and Windows backends can proceed in
  parallel, one per machine.
- 2026-11-13, Linux: Pulse got the null clock and `null_fallback`. To lose
  the server without touching the owner's, put a relay in front of it and
  take the relay away:
  `socat UNIX-LISTEN:/some/px,fork UNIX-CONNECT:$XDG_RUNTIME_DIR/pulse/native`
  under `setsid`, run `tone` with `PULSE_SERVER=unix:/some/px --amp 0
  --fallback on`, then kill socat's process group and delete the socket,
  and start it again. Seen: clean both ways. Each connection adds about
  15 ms to the queue (the guest keeps writing during the handshake and the
  server's first request), followed back at the 0.5% limit. The
  permissions of the Linux working tree on the share were found write-only
  after the Windows sessions; `chmod -R u+rwX` put them right.
- 2026-11-13, Windows: WASAPI backend and `examples/play` written. On the
  test machine (Windows 11 build 26200, AMD HDMI audio, 48 kHz) the driver
  offers no period under 10 ms, so the `IAudioClient3` short-period path and
  the Windows-converts path (`AUTOCONVERTPCM`) have compiled but never run
  with effect; try a machine with an inbox HD Audio or USB device. Still to
  do with the owner at the machine: checklist step 7 with `play`, and step
  8. Step 6 was done by the owner: the tone followed the default device. Building on the network share is
  slow; set `CARGO_TARGET_DIR` to a local disk. Cargo has also missed an
  edit made on the share (it reported the build as fresh), presumably the
  file server's clock; `touch` the edited file if a change seems to have had
  no effect.
- 2026-11-13, Windows: the WASAPI backend stands in for a missing device
  and honours `null_fallback`. Both were tried silently with temporary
  edits that made `open` fail for a stretch of seconds and `serve` return
  an error once; no device was unplugged or disabled. A real test (step 7)
  still needs the owner: disable every output in the sound settings while
  `play` runs, and start `play --fallback on` with all of them disabled.
- 2026-11-13, macOS: CoreAudio backend written and run silently on a
  MacBookPro11,5 with macOS 13.7 (built-in output, 44.1 kHz). Still to do
  with the owner at the machine, using `play`: listen to a tone at all,
  checklist step 6 (switch the default output, e.g. plug in headphones or
  pick another device in the menu bar), step 7 (unplug it), and a change of
  the device's rate in Audio MIDI Setup. The fallback to discarding was
  tried by making `default_output` fail for six seconds in a temporary
  edit (and `null_fallback` by making it fail for the first five); a Mac with built-in output never really has no device, so a true
  test needs a machine whose only output can be unplugged. Whether the processor-overload
  listener ever fires is not known: no overload happened. This machine
  stalls ordinary threads for up to 50 ms a few times a minute (measured
  with no audio running), so guest-side underruns at a 10 ms prebuffer here
  are the host, not the backend; 40 ms runs clean. In zsh, a variable
  holding several options needs `${=var}` to be split.
