//! Several output streams mixed into one on the host.
//!
//! A [`Mixer`] is a [`Render`] made of [`Renderer`]s. The backend asks it
//! for a buffer; it asks each voice for the same buffer and adds them up.
//! Every voice keeps everything a stream of its own has (its ring, its
//! guest rate, its prebuffer gate and its clock follower), so a device
//! with two DACs at different rates gets one connection to the host and
//! one entry in its mixer. [`Audio::open_outputs`] builds one.
//!
//! The voices are independent: each follows the guest's clock from the
//! level of its own ring, so two of them fed from one guest clock are
//! bent by amounts that differ a little and do not stay aligned to the
//! sample. A device that needs that adds its streams together itself and
//! writes the sum to one output.
//!
//! [`Audio::open_outputs`]: crate::Audio::open_outputs

use crate::backend::{Render, RenderInfo};
use crate::format::{DeviceFormat, MAX_CHANNELS};
use crate::stream::Renderer;

/// Frames a voice after the first is rendered at a time, into a scratch
/// buffer allocated once: the device's thread must not allocate, and a
/// backend may ask for any amount.
const CHUNK_FRAMES: usize = 1024;

pub struct Mixer {
    voices: Vec<Renderer>,
    scratch: Box<[f32]>,
    /// Samples per device frame.
    channels: usize,
}

impl Mixer {
    /// Mix `voices`, which must not be empty.
    pub fn new(voices: Vec<Renderer>) -> Self {
        assert!(!voices.is_empty(), "a mixer needs a voice");
        let channels = voices[0].format().channels;
        Self { voices, scratch: vec![0.0; CHUNK_FRAMES * MAX_CHANNELS].into_boxed_slice(), channels }
    }
}

impl Render for Mixer {
    fn configure(&mut self, format: DeviceFormat) {
        self.channels = format.channels;
        for v in &mut self.voices {
            v.configure(format);
        }
    }

    fn render(&mut self, out: &mut [f32], info: &RenderInfo) {
        let Self { voices, scratch, channels } = self;
        let (first, rest) = voices.split_first_mut().expect("a mixer has a voice");
        first.render(out, info);
        if rest.is_empty() {
            return;
        }
        // Whole frames to a chunk, so every voice keeps its channels in
        // place; a tail that is not a whole frame is already silence.
        for part in out.chunks_mut(CHUNK_FRAMES * *channels) {
            for v in rest.iter_mut() {
                let buf = &mut scratch[..part.len()];
                v.render(buf, info);
                for (o, s) in part.iter_mut().zip(buf.iter()) {
                    *o += *s;
                }
            }
            // The sum of full-scale voices is past full scale.
            for o in part.iter_mut() {
                *o = o.clamp(-1.0, 1.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Output, OutputConfig};
    use std::time::Duration;

    const DEVICE: DeviceFormat = DeviceFormat { rate: 48000, channels: 2 };

    /// Two voices at `rates` with no prebuffer and the follower off, and
    /// their mixer on a 48 kHz stereo device.
    fn open(rates: [u32; 2]) -> ([Output; 2], Mixer) {
        let voice = |rate| {
            let cfg = OutputConfig { rate, prebuffer: Duration::ZERO, max_latency: Duration::from_secs(1), max_adjust: 0.0, ..Default::default() };
            Output::detached(&cfg).unwrap()
        };
        let ((a, ra), (b, rb)) = (voice(rates[0]), voice(rates[1]));
        let mut mixer = Mixer::new(vec![ra, rb]);
        mixer.configure(DEVICE);
        ([a, b], mixer)
    }

    fn render(mixer: &mut Mixer, frames: usize) -> Vec<f32> {
        let mut out = vec![9.0; frames * DEVICE.channels];
        mixer.render(&mut out, &RenderInfo::default());
        out
    }

    #[test]
    fn two_voices_at_different_rates_play_added_together() {
        let ([mut a, mut b], mut mixer) = open([48000, 24000]);
        // A steady level on each: a on the left, b on the right, and both
        // a little on the other side.
        a.write(&[[0.25f32, 0.1]; 4800].concat());
        b.write(&[[0.05f32, 0.5]; 2400].concat());
        // More than a chunk, so the scratch buffer is used in parts.
        let out = render(&mut mixer, 3000);
        // Past the rate converter's settling at the start.
        for f in out.chunks_exact(2).skip(200) {
            assert!((f[0] - 0.30).abs() < 0.005 && (f[1] - 0.60).abs() < 0.005, "{f:?}");
        }
        assert_eq!(a.stats().played, 3000);
        assert_eq!(b.stats().played, 3000);
    }

    #[test]
    fn a_voice_with_nothing_to_play_leaves_the_other_untouched() {
        let ([mut a, b], mut mixer) = open([48000, 48000]);
        let sent: Vec<f32> = (0..2000).map(|n| (n % 200) as f32 / 400.0 - 0.25).collect();
        a.write(&sent);
        assert_eq!(render(&mut mixer, 1000), sent);
        // Nor does it count as an underrun of the silent one.
        assert_eq!(b.stats().underruns, 0);
        assert!(a.stats().playing && !b.stats().playing);
    }

    #[test]
    fn a_dropped_voice_falls_silent_and_the_rest_play_on() {
        let ([a, mut b], mut mixer) = open([48000, 48000]);
        drop(a);
        b.write(&[0.5f32; 960]);
        assert_eq!(render(&mut mixer, 480), [0.5; 960]);
    }

    #[test]
    fn full_scale_voices_clip_at_full_scale() {
        let ([mut a, mut b], mut mixer) = open([48000, 48000]);
        a.write(&[0.75f32, -0.75].repeat(100));
        b.write(&[0.75f32, -0.75].repeat(100));
        assert_eq!(render(&mut mixer, 100), [1.0, -1.0].repeat(100));
    }

    #[test]
    fn each_voice_keeps_its_own_rate_when_one_changes() {
        let ([mut a, mut b], mut mixer) = open([48000, 48000]);
        b.set_rate(12000).unwrap();
        a.write(&[0.2f32; 2 * 2400]);
        b.write(&[0.4f32; 2 * 600]);
        let out = render(&mut mixer, 2000);
        for f in out.chunks_exact(2).skip(400) {
            assert!((f[0] - 0.6).abs() < 0.005, "{f:?}");
        }
        assert_eq!(a.rate(), 48000);
    }
}
