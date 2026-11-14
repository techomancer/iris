# Host audio is the `iris-audio` crate: devices write guest-rate frames, nothing else

**Keywords:** audio,hal2,rad1,mace,ice1724,iris-audio,cpal,resampler,prebuffer,underrun,pulse,wasapi,coreaudio

**Status: rule (2026-11-13).** cpal and the per-device resamplers are gone.

`src/host_audio.rs` is one function, `open_output(cfg, who, channels, rate,
capture_var) -> Option<iris_audio::Output>`. A device then calls
`out.set_rate()` when the guest reprograms its clock and `out.write(&[i16])`
with whole frames. Everything else is the library's, on the far side of its
ring: rate conversion, following the guest's clock (it bends the ratio by up
to 0.5%), the prebuffer (`[audio] prebuf_ms`), following the host's default
device, and running with no device (`null_fallback` is always set, so an
open never needs a retry).

Rules that follow:

- **Do not prebuffer, resample or retry in a device.** HAL2's own 20 ms
  prebuffer and the cards' once-a-second reopen were removed for this.
- **What reaches a sound card or the null host is never sample-exact**, even
  at equal rates: the clock follower always runs the frames through the
  rate converter. **`[audio] file` turns the follower off** (`max_adjust`
  0): a file has no crystal to drift from, so a card playing at the file's
  48 kHz is recorded exactly and one at another rate as a fixed-ratio
  conversion, and a capture variable on the same stream records the same.
  The price: a guest whose clock runs off the host's drains or fills the
  ring until it underruns or skips, and a host too loaded to keep the guest
  at full speed leaves gaps in the file that the follower used to hide. At
  a hypothetical 250 ppm that would be one underrun per about 100 s of
  unbroken sound with the guest behind (the ring starts about 25 ms full)
  or one skip per about 300 s with it ahead (the ceiling is 100 ms); the
  real mismatch has not been measured. A file is for exact captures, so
  that is the right trade. A capture variable on a sound card or the null
  host still follows and is not exact. Tests that need the card's
  exact frames read a tap (`Sink::tap`, `PcmOut::tap`), a detached stream
  (`host_audio::test_signal::detached`) or an `[audio] file`.
- **Underruns are counted in times, not samples** (`Output::stats()`), and
  only while the device has not called `idle()`. Only HAL2 calls `idle()`
  (codec A disarmed).
- **A burst larger than the stream queues is skipped by the renderer**
  (`Stats::overruns`), not dropped at the write. `max_latency` is 100 ms or
  three prebuffers. A card that catches up after a stall should keep its
  burst under that.
- **The O2 opens its two outputs as voices of one host stream**
  (`host_audio::open_outputs`, the library's `Mixer`): each has its own rate,
  prebuffer and clock follower, and a file or capture hears the mix. Voices
  are not sample-aligned with one another, which is why the ICE1724 still
  sums its analog and S/PDIF engines itself.
- **The ICE1724 folds to `[audio] channels`** (default 2) itself with its
  BS.775 matrix. The library cannot report how many speakers the host has
  (the Pulse backend opens whatever width is asked and lets the server mix),
  and its own fold of 8 slots to stereo divides the level by four.
- **No capture on Windows or macOS in tests.** The real-device loopback
  tests record with `parec`, Linux only; the cpal recorder went with cpal.
- The `*_CAPTURE` variables now write a WAV at the device's rate (the
  library's tee), not raw samples.

`iris-audio/` is a path dependency with a workspace of its own (excluded
from the root workspace); its own notes are in `iris-audio/CLAUDE.md`.
