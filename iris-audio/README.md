# iris-audio

Low-latency audio output for emulators. An emulated sound device pushes
frames at whatever rate the guest programmed, on the guest's schedule and by
the guest's clock; this crate gets them to the host's speakers without
blocking the device, without clicks, and without the device knowing what the
host's sound system is.

It was started for the IRIS emulator, which plays all its sound devices through it. `DESIGN.md`
explains the design and what a backend has to provide.

```rust
use iris_audio::{Audio, BackendChoice, OutputConfig};
use std::time::Duration;

let audio = Audio::new(BackendChoice::Default)?;
let mut out = audio.open_output(OutputConfig {
    name: "HAL2".into(),
    rate: 44100,
    prebuffer: Duration::from_millis(20),
    ..Default::default()
})?;

out.write(&[0i16; 8]);   // four stereo frames; never blocks
out.set_rate(48000)?;    // the guest reprogrammed its codec
out.idle();              // the guest stopped its DAC
println!("{:?}", out.stats());
```

## Status

| Part | State |
|---|---|
| Stream engine: ring, prebuffer, clock follower, rate converter, channel map | working, unit-tested in simulated time |
| File backend (a WAV played in real time) | working |
| Null backend (frames taken in real time and discarded) | working |
| Tee (`OutputConfig::tee`: record what the device was given) | working |
| Mixer (`Audio::open_outputs`: several outputs, each at its own rate, in one host stream) | working, unit-tested in simulated time |
| PulseAudio protocol backend (PulseAudio and PipeWire's pipewire-pulse) | working on Linux; built on every Unix and the default on all but macOS, not yet run on a BSD; discards like the null backend while there is no server |
| WASAPI backend (Windows 10 and later) | working, follows a switch of the default output device; discards like the null backend while there is no device |
| CoreAudio backend (macOS) | working; written to follow the default output device, which has not been watched yet; discards like the null backend while there is no device |
| ALSA, native PipeWire backends | not started |
| Input | interface only (`Capture`, `Backend::open_input`, `Input`) |

## How it works

```text
 guest thread                        backend's audio thread
 Output::write ──► ring (guest rate) ──► Renderer ──► device buffer
                                         prebuffer gate
                                         clock follower
                                         rate converter
                                         channel map
```

The ring carries frames at the guest's rate. Everything that depends on the
device is on the device's side of it, so a guest changing its rate and a host
changing its output device are both invisible to the other end.

**Prebuffer.** Nothing plays until the ring holds `prebuffer` plus one device
period. If it runs dry the renderer plays silence and waits for that level
again. A sound shorter than the prebuffer still plays: once the guest has
added nothing for a prebuffer's time, or has called `idle()`, what is queued
goes out.

**Clock follower.** The guest's timers and the sound card's crystal never
agree. The renderer watches the ring's level and bends the conversion ratio by
up to `max_adjust` (half a percent by default, under a tenth of a semitone) to
hold it at the prebuffer level, so a small mismatch never becomes an underrun
or a growing delay. A guest that runs ahead by more than that is cut back to
the target in one step (`Stats::overruns`).

**Rate converter.** A 48-tap windowed-sinc interpolator, about 80 dB down in
the stop band. It works a frame at a time, so it adds only its own delay: 24
guest frames, half a millisecond at 48 kHz and two at 11025 Hz.

**Backends** pull. Each owns its device and thread, calls `Render::render`
for exactly the frames the device wants, and is responsible for following the
default device and reconnecting. See the `backend` module.

## Several outputs in one stream

A device with more than one DAC can open them together:

```rust
let mut dacs = audio.open_outputs(&[
    OutputConfig { name: "O2 audio".into(), rate: 44100, ..Default::default() },
    OutputConfig { rate: 8000, ..Default::default() },
])?;
dacs[1].set_rate(22050)?;
```

Each `Output` behaves as a stream of its own: its own rate, prebuffer, clock
follower and counters. On the host they are one connection and one entry in
the mixer, added together on the device's thread, and a tee records the sum.
The stream stops when the last of them is dropped. The voices are not kept
aligned to the sample with one another; a device that needs that sums its
streams itself and writes one output.

## Choosing the prebuffer

`prebuffer` is the latency the library adds and its only margin, so it is the
setting that matters. It has to cover two things:

- **The guest's largest burst.** A device that writes a few frames per timer
  interrupt needs nothing for this; one that writes 1024-frame blocks needs
  more than 21 ms at 48 kHz.
- **How late the guest's thread can run.** Every stall longer than the
  prebuffer is an underrun. This is what a trickling device depends on.

The device period is added automatically; do not include it.

Total delay from `write` to the speaker is `prebuffer` + one device period +
the device's own latency (`Stats::queued` and `Stats::device_delay` report
both).

## Trying it

```sh
cargo test
cargo run --release --example tone -- --burst 4 --drift 1500
cargo run --release --example tone -- --burst 1024 --rate 48000 --prebuffer 30
cargo run --release --example tone -- --backend file:out.wav --rate 11025
cargo run --release --example play
```

`tone` plays a 440 Hz tone as an emulated device would and prints the
stream's counters once a second. `--burst` sets how many frames it writes at a
time, `--drift` puts its clock off by so many parts per million, `--jitter`
stalls its thread now and then, `--amp 0` plays silence, `--tee FILE` records
what the device was given. `--backend` picks `pulse`, `wasapi`, `coreaudio`,
`null` or `file:PATH` instead of the platform's own. `--fallback on` sets
`OutputConfig::null_fallback`, so the stream opens with no device present.

`play` plays a steady tone until Enter is pressed, sleeping between writes
instead of spinning. It prints a line whenever the output device or its
format changes and keeps one line of counters up to date, so it is the thing
to leave running while switching the default output device, unplugging it or
restarting the sound server. `--amp`, `--freq`, `--rate`, `--prebuffer` and
`--latency` work as in `tone`.

## Measured

PipeWire 1.6.8 through pipewire-pulse, 48 kHz sink, default settings:

- The server asks for 256 frames every 5.3 ms, evenly, whether 20 ms or 5 ms
  of device latency is requested: 256 frames is pipewire-pulse's default
  minimum.
- It reports 21.3 ms of device delay (one 1024-frame graph period).
- With the default 10 ms prebuffer the ring settles at 15 to 18 ms. A guest
  clock 0.15% fast is followed with no underruns or overruns, trickling 4
  frames at a time or writing 1024-frame blocks (with a 30 ms prebuffer).
- A guest thread that stalls for 30 ms against a 10 ms prebuffer underruns
  once per stall, as it must.
- With the server reached through a relay socket that was taken away and
  put back (the server itself was left alone): a stream opened with
  `null_fallback` and no server ran on the null clock, moved to the server
  when it appeared, went back to the null clock when it went and returned
  again, with no underruns and nothing dropped. Each connection leaves
  about 15 ms more in the queue, which the clock follower takes back over
  the next three seconds.

Windows 11 (build 26200), WASAPI, AMD High Definition Audio over HDMI,
48 kHz stereo mix format, default settings, runs of 5 to 10 seconds:

- The engine runs the stream at its default 10 ms period; this driver offers
  nothing shorter. The stream's queue is held at two periods and the backend
  reports 10 ms of device delay.
- With the default 10 ms prebuffer the ring settles at 20 ms (prebuffer plus
  one period). Trickling 4 frames at a time, writing 1024-frame blocks (with
  a 30 ms prebuffer) and a guest at 11025 Hz all run with no underruns.
- A guest clock 0.2% fast or slow is followed at +0.2% or -0.2% with the ring
  4 ms above or below its target.
- A guest thread that stalls for 30 ms against a 10 ms prebuffer underruns
  once per stall.
- With the stream made to fail and the default device made to vanish for
  six seconds (simulated inside the backend; no device was unplugged), the
  stream goes on being consumed with nothing dropped, the ring 5 to 10 ms
  above its usual level, and plays again when the device is back, with one
  underrun at the change.
- Opened with `null_fallback` while the device was hidden in the same way,
  the stream starts at once, is consumed at the guest's rate, and moves to
  the device when it appears, with no underrun.

macOS 13.7 on a MacBook Pro of 2015 (MacBookPro11,5), CoreAudio, built-in
output, 44.1 kHz stereo, runs of 10 to 60 seconds:

- The device accepts buffers of 14 to 4096 frames. At the default 5 ms it
  asks for 221 frames at a time and the backend reports 16.1 ms of device
  delay: the buffer, a safety offset of 93 frames, and 399 frames (9 ms) of
  latency in the device and its stream. With `--latency 1` (44 frames) the
  delay is 12.1 ms; buffers of 14 frames are served too.
- The ring settles at 15 ms with the default 10 ms prebuffer (prebuffer plus
  one period). A guest at 11025 Hz and one writing 1024-frame blocks at
  48 kHz (with a 30 ms prebuffer) are converted and followed.
- A guest clock 0.2% fast or slow is followed at +0.2% or -0.2% with the ring
  4 ms above or below its target.
- This machine stalls ordinary threads for 10 to 50 ms several times a
  minute, whatever they are doing, so a guest on such a thread underruns a
  10 ms prebuffer two or three times a minute, on the file backend as well.
  The device's own thread is not affected: no overload was reported at any
  buffer size. A minute runs clean at a 40 ms prebuffer, also with 1 ms of
  device latency.
- With the default device made to vanish for six seconds (simulated inside
  the backend; no device was unplugged), the stream goes on being consumed
  with nothing dropped and no underrun, and plays again when the device is
  back. The queue rises by about 20 ms at the change, the time it takes to
  close the stream, and the clock follower brings it back.
- Opened with `null_fallback` while the device was hidden in the same way,
  the stream starts at once, is consumed at the guest's rate, and moves to
  the device when it appears. The first open of a device in a process takes
  over 100 ms, during which nothing is consumed: at the default
  `max_latency` the renderer skipped 120 ms of sound once at that moment.
- A guest thread that stalls for 30 ms every 0.7 s against a 10 ms prebuffer
  underruns on about two stalls in three: `tone` catches up after a stall,
  which leaves the ring far enough above its target to ride out some of the
  next.

## Known limits

- The Pulse and file backends' threads run at normal priority; real-time
  scheduling through rtkit is not done. The WASAPI backend's thread joins
  the "Pro Audio" class of the multimedia scheduler. On macOS the renderer
  runs on CoreAudio's own real-time thread.
- The CoreAudio backend sets the device's buffer size, which macOS keeps per
  process and device, not per stream: streams of one process on one device
  share the size the last one asked for.
- The CoreAudio backend plays to the device's first channels as they are. A
  multichannel interface whose preferred stereo pair is not its first two
  channels is not handled.
- A switch of the default device on macOS is a short gap, as on Windows.
- With `OutputConfig::null_fallback` the Pulse backend opens a stream when
  no server is running. A server that is running but refuses the client or
  the stream is still an error: it would refuse again.
- On FreeBSD and the other BSDs the only backend is Pulse, which needs the
  user to run PulseAudio or pipewire-pulse. There is no backend for OSS or
  sndio, so a machine with neither server has the null and file backends
  only. The Pulse backend has not been built or run on a BSD yet.
- The WASAPI backend cannot see the device run dry. It counts a device
  underrun when it wakes to find its queue empty, which means it was at
  least a period late.
- A switch of the default device on Windows is a short gap: the stream is
  closed and opened again on the new device.
- When a guest changes rate, the few milliseconds already queued play at the
  new rate.
- Fewer device channels than guest channels are folded crudely (left-side
  channels to left, right-side to right). The Pulse backend avoids this by
  opening the guest's layout and letting the server mix.
- The Pulse backend treats every lost connection the same way: it discards
  the guest's frames in real time and retries twice a second until the
  server is back.

## Licence

BSD 3-Clause; see `LICENSE`.
