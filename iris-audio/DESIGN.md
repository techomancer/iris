# iris-audio design

This document explains why the crate is shaped the way it is and what each
backend has to do. `README.md` says how to use it; the module documentation
in `src/` says how each part works in detail.

## The problem

An emulated sound device is an awkward audio client:

- **It runs on the guest's clock.** Frames appear when the guest's timer
  interrupts fire, at a rate set by emulated time. That rate is never exactly
  the host sound card's, and it wobbles when the emulator is busy.
- **It delivers unevenly.** One device hands over a few frames per interrupt
  (HAL2 on the SGI Indy); another hands over a DMA buffer of a thousand
  frames at once.
- **It picks its own rate, and changes it.** 11025 Hz for a game, 44100 Hz
  for a CD player, 48000 Hz for the desktop, switched while running.
- **It must never wait.** The device model runs on an emulator thread; a
  blocked write stalls the guest.

A host sound API expects the opposite: a client that fills exactly the buffer
it is asked for, at the device's rate, on the device's schedule, from a
real-time thread.

General-purpose audio libraries leave the gap between the two to the
application. Left unfilled, it produces the usual emulator symptoms: crackle
from small periodic underruns, latency that creeps up, sound that stops when
the user switches output device.

## Goals

1. Low latency from the guest's write to the speaker, with the amount of
   buffering chosen per stream.
2. No clicks from clock mismatch, at any guest rate.
3. The guest side never blocks, locks or allocates.
4. The host's default device can change, or its server restart, without the
   guest side noticing.
5. One small interface per host audio system, so that a backend is a few
   hundred lines and can be written and tested on its own platform.
6. Input later, through the same shapes.

Not goals: device enumeration and selection UI, exclusive-mode output,
mixing several streams (each stream is its own host stream; the host mixes),
effects, file decoding.

## Shape

```text
 guest thread                        backend's audio thread
 Output::write ──► ring (guest rate) ──► Renderer ──► device buffer
                                         prebuffer gate
                                         clock follower
                                         rate converter
                                         channel map
```

Three decisions carry the design.

**The ring holds guest-rate frames.** Rate conversion happens on the device's
side of the ring, not the guest's. A device change that brings a new rate is
then a matter for the renderer alone, and a guest rate change is one atomic
store. Had the ring carried device-rate frames, every device model would own
a resampler aimed at a rate that can change under it.

**Backends pull.** A backend owns its device and its thread and calls
`Render::render` for exactly the frames the device wants. The renderer always
fills the whole buffer, with silence if it has to. This is the model every
host API already has (WASAPI event-driven render, CoreAudio's render
callback, PipeWire's process callback, PulseAudio's write requests), so a
backend is a thin adapter and adds no buffer of its own.

**Following the device is the backend's job.** When the default device
changes, the backend reopens and calls `Render::configure` with the new
format before its next buffer. Nothing above the backend takes part.

## The stream

### Prebuffer gate

The renderer is either waiting or playing.

*Waiting*, it renders silence until the ring holds `prebuffer` plus one device
period, then starts. Two exceptions let short sounds through: if the ring
holds something and the guest has added nothing for a prebuffer's time, or
the guest has called `idle()`, it starts with what there is. Without them a
sound shorter than the prebuffer (a boot chime's last fragment, a beep) would
never play.

*Playing*, it converts frames until the ring runs dry. Then it fills the rest
of the buffer with silence, counts an underrun (unless the guest is idle) and
goes back to waiting. Restarting on the first frame to arrive would underrun
again immediately; waiting for the prebuffer turns a stall into one gap
instead of a burst of crackle.

The device period is added to the target because the device takes a period at
a time: with only the prebuffer in the ring, a device whose period exceeds it
would drain the ring on every call. The period is measured, not configured:
it is the largest recent request, in guest frames.

### Clock follower

The level of the ring is the integral of the difference between the two
clocks, so holding the level steady is the same as matching them. Each render
call:

```text
level  += (1 - exp(-dt / 0.5 s)) * (fill - level)      smoothed fill
adjust  = clamp((level - target) / rate * 0.5, ±max_adjust)
step    = (guest rate / device rate) * (1 + adjust)
```

`fill` is sampled at the start of a render call. The smoothing sees through
both sawtooths: the guest's bursts and the device's periods. The gain and the
smoothing time are chosen together so that the loop is critically damped: it
settles in a few seconds without overshoot.

It is a proportional controller, so a constant clock error leaves a constant
offset in the level: 0.1% of error holds the ring 2 ms above target. That is
accepted; an integral term would remove it at the cost of overshoot after
every underrun.

`max_adjust` defaults to 0.5%, about a twelfth of a semitone, which is not
audible as a pitch change. A guest that runs further off than that (an
emulator in fast-forward, or one that cannot keep up) is beyond what bending
the rate can hide. Running fast, the ring passes `max_latency` and the
renderer skips back to the target in one step. Running slow, it underruns.

### Rate converter

A polyphase windowed-sinc interpolator: 48 taps, 128 phases with linear
interpolation between them, Kaiser window with beta 8. The stop band is about
80 dB down. Going down in rate the cutoff follows the output's Nyquist
frequency, so content above it is removed instead of aliasing.

It pulls one input frame at a time as its position advances, so it adds no
block buffering: its delay is 24 guest frames. The ratio is a parameter of
each call, which is what the clock follower needs. Changing rates recomputes
the filter in place, so nothing allocates after construction.

A library resampler was considered and set aside: block-based interfaces want
fixed-size chunks of input or output, which means a buffer between the ring
and the converter, and that buffer is latency.

With the follower switched off (`max_adjust = 0`) and equal rates, frames are
copied through untouched, which keeps recordings bit-exact for tests.

### Channels

The guest's frame width is fixed when the stream opens. A backend opens that
many channels if the host can take them and reports what it got; the renderer
maps a guest frame into a device frame (mono to both front speakers, extra
device channels silent, a crude fold when the device has fewer). Backends
whose host mixes properly should open the guest's layout and let the host do
it.

### Several outputs in one stream

`Audio::open_outputs` gives a device with several DACs one connection to the
host. Nothing below the `Render` interface knows: the backend is handed a
`Mixer`, which is a `Render` made of the renderers above. It asks each for the
buffer the device wants and adds the results, clamped to full scale. A voice is
therefore a whole stream, ring, gate, follower and converter, and the voices
differ freely in rate, width and prebuffer. The set is fixed when the stream
opens, so nothing is ever handed to the device's thread while it runs. Voices
after the first are rendered into a scratch buffer of fixed size, a chunk at a
time, because the thread may not allocate. What belongs to the stream (its
name, the device latency, the tee) comes from the first configuration.

Each voice follows the guest's clock from its own ring, so two voices driven
by one guest clock are bent by slightly different amounts and are not aligned
to the sample. That is inaudible for independent sounds and wrong for two
halves of one signal; those a device adds together itself.

## The backend contract

```rust
trait Backend {
    fn name(&self) -> &'static str;
    fn open_output(&self, request: &OutputRequest, render: Box<dyn Render>)
        -> Result<Box<dyn StreamHandle>, Error>;
    fn open_input(&self, request: &InputRequest, capture: Box<dyn Capture>)
        -> Result<Box<dyn StreamHandle>, Error>;   // default: Unsupported
}

trait Render: Send + 'static {
    fn configure(&mut self, format: DeviceFormat);
    fn render(&mut self, out: &mut [f32], info: &RenderInfo);
}
```

What a backend must do:

1. **Open the default device** in shared mode at the device's own rate, with
   32-bit float samples if the host offers them. Convert on the backend's
   side if it does not.
2. **Fail synchronously.** If the first open fails, `open_output` returns the
   error; it does not start a thread that fails later. The exception is asked
   for by the caller: with `OutputRequest::null_fallback`, a backend that can
   stand in for a missing device (point 5) starts the stream that way, at the
   guest's rate and channels, and moves it to a device when there is one, so
   that the program above never has to open the stream again.
3. **Call `configure` before the first `render`**, and again before the first
   `render` after any change of rate or channel count.
4. **Call `render` for exactly what the device asks**, from the thread that
   serves the device, and use everything it returns.
5. **Follow the default device and survive its loss.** Reopen without help.
   While there is no device, either stand in for it (call `render` at the
   lost device's pace and format from the backend's own thread and discard
   the frames, which `backend::null::Pacer` does), or do not call `render`
   at all: then the ring fills, the guest's writes are dropped and counted,
   and the renderer skips to its target when the device returns. Standing
   in is better for the guest, which sees a device that never went away.
6. **Report what it can** in `RenderInfo`: the delay from the buffer to the
   speaker, and how often the device itself ran dry.
7. **Stop and join on drop** of the handle, promptly.
8. **Ask for a small period.** `OutputRequest::latency` is the device-side
   budget; with none given, aim for 10 to 20 ms.
9. **Raise the thread's priority** where the host has a supported way.

`Render` implementations in the crate do not block, lock or allocate, so they
are safe on a real-time thread. A backend must not add anything to that
thread that does.

### Backends

| Backend | Host | State | How the device is followed |
|---|---|---|---|
| `file` | any | done | no device |
| `null` | any | done | no device |
| `pulse` | PulseAudio, pipewire-pulse, on any Unix | done | the server moves the stream; reconnects if the server restarts; discards while there is none |
| `pipewire` | PipeWire, native | planned | the session manager moves the node |
| `alsa` | bare Linux | planned | no default to follow; reopen on error |
| `wasapi` | Windows | done | device-change notification, reopen; discards while there is none |
| `coreaudio` | macOS | done | property listeners, reopen; discards while there is none |

**file.** A thread paced by the host's clock appends to a WAV. It behaves
like a card at a fixed rate and period, underruns included, so tests can
listen to a stream on machines with no sound.

**null.** The same thread and clock with nowhere to write: the frames that
are due are rendered and thrown away. For an emulator run with its sound
off, and for a host with no output, where a guest that waits for its samples
to be played must still see them go. Its clock is a small type of its own so
that a backend for a real device can keep a stream running through a time
without one.

**pulse.** A blocking client of the native protocol on the server's Unix
socket, one thread per stream. The server clocks the stream with write
requests. The stream is not tied to a sink, so the server moves it and
converts when the default changes; the format holds for the life of a
connection. When the connection is lost the thread takes the frames itself
with the null backend's clock, in the format the server had, and tries for a
new connection twice a second; a stream opened with `null_fallback` can
begin in that state, at 48 kHz. Only the absence of a server is covered that
way: a server that answers and refuses is an error. The setting-up of a
connection is under a timeout, because it happens on the thread that keeps
the guest consumed. The backend uses nothing of Linux, so it is the default
on every Unix that has no backend for its own sound system, which today is
all of them but macOS.

**pipewire (native).** Wanted for control that the Pulse protocol does not
give: the node's quantum directly, and PipeWire's own real-time thread. It
links the PipeWire client library, so it sits behind a feature. PipeWire and
its client library are also packaged for FreeBSD, so the backend should not
be tied to Linux in its `cfg`; whether it is worth having there is doubtful,
since PipeWire on FreeBSD reaches the sound card through OSS and few people
run it for audio.

**oss, sndio (not planned yet).** What FreeBSD and OpenBSD themselves
provide. Either would be the BSD counterpart of the ALSA backend: a device
file or a small library, a thread, no default to follow.

**alsa.** For a machine with no sound server: an emulator booted straight
from a console. Opens `default` (or a named PCM), negotiates period and
buffer sizes, and serves the PCM from a thread that sleeps in `poll`. Needs
its own format conversion, since hardware PCMs are rarely float.

**wasapi.** Shared mode, event-driven, in the engine's mix format, one
thread per stream that owns every COM object the stream uses. On each wake
the thread tops its queue up to two periods, not to the whole buffer, so the
buffer's size does not become latency. It asks for a period shorter than the
engine's default 10 ms where the driver has one (`IAudioClient3`). Windows
does not move a stream to a new default device: the backend registers for
device notifications, and the notification, which arrives on a thread of the
system's, only compares the new default with the device in use and wakes the
stream's thread, which closes its stream and opens one on the new device. A
device that disappears or changes format invalidates the stream, which is
handled the same way. With no device left, the thread takes the frames
itself with the null backend's clock, in the format of the device that went,
asking for a device twice a second and on every notification; a stream
opened with `null_fallback` can begin in that state, at the guest's rate and
channels. Windows converts in two cases only: a mix format the renderer cannot
fill (not 32-bit float, or more than eight channels), and a guest with more
channels than the device, which Windows folds down better than the renderer.

**coreaudio.** A HAL output unit on the default device, fed 32-bit float at
the device's own rate. Its render callback, on the device's real-time
thread, is the `render` call, so nothing is queued on the device's side but
the buffer being filled; the backend sets that buffer's size from the
requested latency (5 ms by default). The other kind of unit, the
default-output unit, follows the default device by itself, but it does so
behind the client's back, converting to whatever the new device runs at; the
renderer would go on converting to the old rate, and the sound would pass
through two converters. So the backend follows the device itself, the way the
WASAPI backend does: one thread per stream owns the unit, property listeners
for the default device and for the device's life, rate and channels only
wake it, and it reopens on the new device and reconfigures the renderer if
the stream is no longer where it should be. With no device left, the thread
takes the frames itself with the null backend's clock, in the format of the
device that went, until one appears; a stream opened with `null_fallback`
can begin in that state. CoreAudio is declared by hand in
the module; the backend adds no dependency.

## Input

Input is output turned round. The backend's thread hands device-rate frames
to a `Capture`; that end converts to the guest's rate, following the guest's
clock from the other direction (the guest now consumes), and queues them in a
ring; `Input::read` takes them and returns silence when the ring is empty.
`InputConfig`, `Input`, `Capture` and `Backend::open_input` fix that shape.
None of it is implemented, and `Input` cannot be constructed.

## Testing

- **In simulated time.** `Output::detached` returns both ends with no backend.
  The stream tests step a guest (rate, clock error, burst size) and a device
  (period) through a minute of pretend time in milliseconds of real time and
  check the counters and the rendered tone. Everything about the stream that
  can be tested this way is.
- **In real time, without sound.** The file backend runs the whole path
  against the host's clock.
- **On a real device.** `examples/tone` plays as a device model would and
  prints the stream's counters; `--amp 0` plays silence. Results per host go
  in the README's "Measured" section.

A backend cannot be unit-tested without its host. Each one is checked with
`tone` against the same list: steady trickle, large blocks, a guest clock
that is off, a stalling guest, a default-device switch while playing, the
device or server removed and brought back.

## Open questions

- **Guest rate changes are not in-band.** The few milliseconds queued at the
  old rate play at the new one. Marking the change in the ring would fix it;
  it has not been audible enough to justify the cost.
- **An underrun cuts to silence.** A short fade at each end would soften the
  click, at the price of touching samples on the way out.
- **The follower steers by level alone.** Backends that report device time
  could let it compare clocks directly and settle faster.
- **Several streams from one guest** (a card with independent outputs) are
  independent host streams, each with its own follower. Whether they need to
  stay sample-aligned with each other is not known yet.
