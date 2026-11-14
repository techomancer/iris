//! The rate converter: a polyphase windowed-sinc interpolator whose ratio
//! can be changed on every call.
//!
//! It works a frame at a time and pulls its input when it needs it, so it
//! adds no buffering of its own beyond the filter's history ([`DELAY`]
//! input frames). It allocates only when it is built: changing the rates
//! recomputes the filter in place, which keeps it usable on an audio
//! thread.

use crate::format::MAX_CHANNELS;

/// Input frames the filter spans.
const TAPS: usize = 48;
/// Filter rows between two input frames; rows in between are interpolated.
const PHASES: usize = 128;
/// Kaiser window shape: about 80 dB of stop-band attenuation.
const BETA: f64 = 8.0;
/// The filter's -6 dB point as a fraction of the lower of the two Nyquist
/// frequencies. With [`TAPS`] and [`BETA`] the pass band reaches about 0.8
/// and the stop band starts at about 1.0.
const CUTOFF: f64 = 0.90;

/// The filter's delay in input frames.
pub const DELAY: usize = TAPS / 2;

pub struct Resampler {
    channels: usize,
    in_rate: u32,
    out_rate: u32,
    /// The Kaiser window for every row and tap; does not depend on the rates.
    window: Box<[f32]>,
    /// `(PHASES + 1) * TAPS` coefficients for the current rates.
    kernel: Box<[f32]>,
    /// The cutoff `kernel` was computed for.
    cutoff: f64,
    /// Each channel's last TAPS input samples, stored twice so the window
    /// oldest-to-newest is always the contiguous `hist[c][pos..pos + TAPS]`.
    hist: [[f32; 2 * TAPS]; MAX_CHANNELS],
    pos: usize,
    /// Where the next output lies, in input frames past the centre of the
    /// window. At 1.0 or more the window has to advance first.
    frac: f64,
}

/// Distance in input frames of tap `k` from the output point of row `p`.
#[inline]
fn tap_distance(p: usize, k: usize) -> f64 {
    k as f64 - (TAPS / 2 - 1) as f64 - p as f64 / PHASES as f64
}

/// The zeroth-order modified Bessel function of the first kind.
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    let q = x * x / 4.0;
    for n in 1..64 {
        term *= q / (n * n) as f64;
        sum += term;
        if term < sum * 1e-12 {
            break;
        }
    }
    sum
}

impl Resampler {
    pub fn new(channels: usize, in_rate: u32, out_rate: u32) -> Self {
        assert!((1..=MAX_CHANNELS).contains(&channels));
        let i0_beta = bessel_i0(BETA);
        let mut window = vec![0.0f32; (PHASES + 1) * TAPS].into_boxed_slice();
        for p in 0..=PHASES {
            for k in 0..TAPS {
                let x = tap_distance(p, k) / (TAPS / 2) as f64;
                let w = if x.abs() >= 1.0 { 0.0 } else { bessel_i0(BETA * (1.0 - x * x).sqrt()) / i0_beta };
                window[p * TAPS + k] = w as f32;
            }
        }
        let mut r = Self {
            channels,
            in_rate: 0,
            out_rate: 0,
            window,
            kernel: vec![0.0f32; (PHASES + 1) * TAPS].into_boxed_slice(),
            cutoff: 0.0,
            hist: [[0.0; 2 * TAPS]; MAX_CHANNELS],
            pos: 0,
            frac: 0.0,
        };
        r.set_rates(in_rate, out_rate);
        r
    }

    pub fn in_rate(&self) -> u32 { self.in_rate }
    pub fn out_rate(&self) -> u32 { self.out_rate }

    /// Input frames consumed per output frame at the nominal ratio.
    pub fn step(&self) -> f64 { f64::from(self.in_rate) / f64::from(self.out_rate) }

    /// Change the rates. The history is kept, so sound already in the
    /// filter plays on at the new rate without a gap. Does not allocate.
    pub fn set_rates(&mut self, in_rate: u32, out_rate: u32) {
        assert!(in_rate > 0 && out_rate > 0);
        self.in_rate = in_rate;
        self.out_rate = out_rate;
        // Going down, the filter has to cut at the output's Nyquist
        // frequency instead of the input's, or what lies above it aliases.
        let cutoff = CUTOFF * (f64::from(out_rate) / f64::from(in_rate)).min(1.0);
        if cutoff == self.cutoff {
            return;
        }
        self.cutoff = cutoff;
        for p in 0..=PHASES {
            let row = &mut self.kernel[p * TAPS..(p + 1) * TAPS];
            let mut sum = 0.0f64;
            for (k, c) in row.iter_mut().enumerate() {
                let x = std::f64::consts::PI * cutoff * tap_distance(p, k);
                let sinc = if x.abs() < 1e-9 { 1.0 } else { x.sin() / x };
                let v = cutoff * sinc * f64::from(self.window[p * TAPS + k]);
                *c = v as f32;
                sum += v;
            }
            // Unity gain at DC on every row, so a steady level has no ripple.
            let norm = (1.0 / sum) as f32;
            row.iter_mut().for_each(|c| *c *= norm);
        }
    }

    /// Forget the sound in the filter.
    pub fn reset(&mut self) {
        self.hist = [[0.0; 2 * TAPS]; MAX_CHANNELS];
        self.pos = 0;
        self.frac = 0.0;
    }

    #[inline]
    fn push(&mut self, frame: &[f32; MAX_CHANNELS]) {
        for c in 0..self.channels {
            self.hist[c][self.pos] = frame[c];
            self.hist[c][self.pos + TAPS] = frame[c];
        }
        self.pos = (self.pos + 1) % TAPS;
    }

    /// Produce up to `frames` output frames, each handed to `emit` with its
    /// index. `step` is the input frames to consume per output frame: the
    /// nominal [`Resampler::step`], or a little off it to follow a clock.
    /// `next` supplies one input frame, or returns false when there is
    /// none; then this returns early with the number of frames produced,
    /// and a later call carries on where it stopped.
    #[inline]
    pub fn process(
        &mut self,
        step: f64,
        frames: usize,
        mut next: impl FnMut(&mut [f32; MAX_CHANNELS]) -> bool,
        mut emit: impl FnMut(usize, &[f32; MAX_CHANNELS]),
    ) -> usize {
        let mut input = [0.0f32; MAX_CHANNELS];
        let mut out = [0.0f32; MAX_CHANNELS];
        let mut coef = [0.0f32; TAPS];
        for n in 0..frames {
            while self.frac >= 1.0 {
                if !next(&mut input) {
                    return n;
                }
                self.push(&input);
                self.frac -= 1.0;
            }
            let p = self.frac * PHASES as f64;
            let row = (p as usize).min(PHASES - 1);
            let t = (p - row as f64) as f32;
            let a = &self.kernel[row * TAPS..(row + 1) * TAPS];
            let b = &self.kernel[(row + 1) * TAPS..(row + 2) * TAPS];
            for k in 0..TAPS {
                coef[k] = a[k] + (b[k] - a[k]) * t;
            }
            for c in 0..self.channels {
                let x = &self.hist[c][self.pos..self.pos + TAPS];
                let mut acc = 0.0f32;
                for k in 0..TAPS {
                    acc += x[k] * coef[k];
                }
                out[c] = acc;
            }
            emit(n, &out);
            self.frac += step;
        }
        frames
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The amplitude of the `freq` component of `s` (Goertzel).
    pub(crate) fn amplitude(s: &[f32], rate: u32, freq: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI * freq / f64::from(rate);
        let (mut re, mut im) = (0.0, 0.0);
        for (n, &v) in s.iter().enumerate() {
            re += f64::from(v) * (w * n as f64).cos();
            im -= f64::from(v) * (w * n as f64).sin();
        }
        2.0 * (re * re + im * im).sqrt() / s.len() as f64
    }

    /// The root-mean-square level of `s`: 0.707 of a sine's peak.
    pub(crate) fn rms(s: &[f32]) -> f64 {
        (s.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / s.len() as f64).sqrt()
    }

    /// Resample a mono sine of `freq` and return the output.
    fn convert(in_rate: u32, out_rate: u32, freq: f64, out_frames: usize) -> Vec<f32> {
        let mut r = Resampler::new(1, in_rate, out_rate);
        let mut n = 0u64;
        let mut out = vec![0.0; out_frames];
        let step = r.step();
        let made = r.process(
            step,
            out_frames,
            |f| {
                f[0] = (2.0 * std::f64::consts::PI * freq * n as f64 / f64::from(in_rate)).sin() as f32;
                n += 1;
                true
            },
            |i, f| out[i] = f[0],
        );
        assert_eq!(made, out_frames);
        out
    }

    #[test]
    fn a_tone_keeps_its_pitch_and_level_going_up() {
        // Quake's 11025 Hz to a 48 kHz card.
        let out = convert(11025, 48000, 1000.0, 48000);
        let s = &out[4800..];
        assert!((amplitude(s, 48000, 1000.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn going_up_leaves_no_image_of_the_tone() {
        // A 1 kHz tone sampled at 11025 Hz has its first image at 10025 Hz.
        // A zero-order hold leaves it at about -20 dB; this must do far better.
        let out = convert(11025, 48000, 1000.0, 48000);
        let image = amplitude(&out[4800..], 48000, 10025.0);
        assert!(image < 0.001, "image at {image}");
    }

    #[test]
    fn going_down_removes_what_would_alias() {
        // 30 kHz at 96 kHz cannot exist at 44.1 kHz; unfiltered it would
        // come out at 14.1 kHz.
        let out = convert(96000, 44100, 30000.0, 44100);
        let alias = amplitude(&out[4410..], 44100, 14100.0);
        assert!(alias < 0.001, "alias at {alias}");
        // And a tone well inside the band survives.
        let out = convert(96000, 44100, 5000.0, 44100);
        assert!((amplitude(&out[4410..], 44100, 5000.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn a_steady_level_stays_steady() {
        let mut r = Resampler::new(2, 44100, 48000);
        let mut out = Vec::new();
        let step = r.step();
        r.process(step, 2000, |f| { f[0] = 0.5; f[1] = -0.25; true }, |_, f| out.push([f[0], f[1]]));
        for f in &out[200..] {
            assert!((f[0] - 0.5).abs() < 1e-4 && (f[1] + 0.25).abs() < 1e-4, "{f:?}");
        }
    }

    #[test]
    fn it_stops_when_the_input_runs_out_and_carries_on_later() {
        let mut r = Resampler::new(1, 48000, 48000);
        let mut left = 10;
        let made = r.process(1.0, 100, |f| { f[0] = 1.0; left -= 1; left >= 0 }, |_, _| {});
        // One frame comes from the empty filter before any input is needed.
        assert_eq!(made, 11);
        let made = r.process(1.0, 100, |f| { f[0] = 1.0; true }, |_, _| {});
        assert_eq!(made, 100);
    }

    #[test]
    fn the_ratio_can_move_between_calls() {
        // 0.5% fast for a second, then 0.5% slow: the input consumed
        // follows the step exactly.
        let mut r = Resampler::new(1, 48000, 48000);
        let mut pulled = 0u32;
        r.process(1.005, 48000, |_| { pulled += 1; true }, |_, _| {});
        assert!((48238..=48242).contains(&pulled), "{pulled}");
        pulled = 0;
        r.process(0.995, 48000, |_| { pulled += 1; true }, |_, _| {});
        assert!((47758..=47762).contains(&pulled), "{pulled}");
    }
}
