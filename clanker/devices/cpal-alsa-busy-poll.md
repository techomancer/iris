# cpal's ALSA backend can spin a core: `[audio] host = "pulseaudio"`

**Keywords:** hal2,audio,alsa,cpal,pulseaudio,pipewire,busy-poll,xrun,underrun,linux,red led

**Symptom.** On Linux, HAL2 logs a steady stream of xruns / underruns and the
guest runs slowly; `top -H` shows cpal's ALSA worker thread at 100% of a core.
Once (Debian 12 arm64, Parallels VM, 4 vCPUs) the IP28 sat in its power-on
diagnostics with the red LED lit because the emulator was starved.

**Cause (cpal 0.18).** The PCM's poll descriptor stays readable while `avail`
is below `avail_min`, so cpal's worker returns from `poll()` immediately and
loops, about five million times a second. Seen through PipeWire's ALSA plugin
(`default`) and through the card's own device (`hw`/`plughw`) alike. `aplay`
with the same period and buffer sizes sleeps normally, so it is cpal's loop,
not the driver. Not reproduced yet on bare metal; it may need the VM's timing.

**Workaround.** `[audio] host = "pulseaudio"` (or `--audio-host pulseaudio`):
cpal's own PulseAudio client (pure Rust, the `pulseaudio` feature) talks to a
PulseAudio server or PipeWire's pipewire-pulse over the socket. The same
stream used 0.01 s of CPU in 5 s. The default stays ALSA until the bug is
confirmed on real hardware and reported upstream; then ALSA could become the
fallback only.

**Buffer size.** Left to itself, PulseAudio (and pipewire-pulse) targets about
two seconds of buffering, and IRIX's sound came that late. HAL2 asks for a
fixed period of 512 frames there (cpal makes the target two periods, ~21 ms
at 48 kHz, with adjust-latency) unless `cpal_buffer_frames` says otherwise.

**Picking the host.** With the `pulseaudio` feature compiled in, cpal's own
`default_host()` prefers PulseAudio whenever its socket exists, so HAL2 must
not call it on Linux: `output_host()` in `src/dev/hal2.rs` asks for ALSA by id
unless PulseAudio was requested.
