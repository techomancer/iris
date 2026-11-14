//! An output stream: the guest's end ([`Output`]) and the device's end
//! ([`Renderer`]), joined by a lock-free ring of guest-rate frames.
//!
//! ```text
//!  guest thread                       backend's audio thread
//!  Output::write ──► ring (guest rate) ──► Renderer ──► device buffer
//!                                          prebuffer gate
//!                                          clock follower
//!                                          rate converter
//!                                          channel map
//! ```
//!
//! The ring holds frames at the guest's rate, so everything that depends
//! on the device lives on the device's side of it. A guest that changes
//! its rate, or a device that is replaced by another, never involves the
//! other end.
//!
//! # Two clocks
//!
//! The guest produces frames by its own timers and the device consumes
//! them by its crystal, and the two never agree. Left alone the ring
//! slowly drains or fills until it underruns or overflows, which is heard
//! as a click every few seconds or minutes. The renderer instead watches
//! the ring's level and bends the conversion ratio by a fraction of a
//! percent to hold it at a target: inaudible, and the ring never runs out.
//!
//! # Latency
//!
//! The renderer holds the ring at [`OutputConfig::prebuffer`] plus one
//! device period. That, plus what the device buffers
//! ([`OutputConfig::device_latency`]), is the delay from
//! [`Output::write`] to the speaker.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::time::Duration;

use rtrb::{Consumer, Producer, RingBuffer};

use crate::Error;
use crate::backend::{Render, RenderInfo, StreamHandle};
use crate::format::{DeviceFormat, MAX_CHANNELS, Sample};
use crate::resample::Resampler;

/// How quickly the level the clock follower steers by tracks the ring's
/// real level, in seconds. Long enough to see through a guest that writes
/// in bursts and a device that reads in periods.
const LEVEL_SMOOTHING: f64 = 0.5;
/// How hard the clock follower pulls: the ratio moves by this much per
/// second of error in the level. With [`LEVEL_SMOOTHING`] this settles
/// without overshoot.
const FOLLOW_GAIN: f64 = 0.5;
/// The lowest guest rate accepted.
const MIN_RATE: u32 = 1000;

/// How to open an output stream.
#[derive(Clone, Debug)]
pub struct OutputConfig {
    /// The stream's name in the host's mixer.
    pub name: String,
    /// Samples per guest frame, 1 to [`MAX_CHANNELS`]. The order beyond
    /// stereo is front left, front right, then (4) rear left, rear right,
    /// or (6, 8) centre, LFE, rear left, rear right, side left, side right.
    pub channels: usize,
    /// The guest's rate now. [`Output::set_rate`] changes it.
    pub rate: u32,
    /// The highest rate the guest will ever set; sizes the ring.
    pub max_rate: u32,
    /// How much sound the ring should hold. Playback starts when the ring
    /// has this much (plus one device period) and the clock follower holds
    /// it there afterwards, so this is both the start-up delay and the
    /// steady latency the library adds.
    ///
    /// It has to cover the guest's unevenness: at least the largest burst
    /// the guest writes at once, plus however late the guest's thread can
    /// run. A guest that trickles a few frames per interrupt needs little
    /// for the bursts but depends entirely on its timer being on time; one
    /// that writes 20 ms blocks needs more than 20 ms.
    pub prebuffer: Duration,
    /// The most sound the ring may hold. When a guest runs ahead by more
    /// than the clock follower can absorb, the renderer drops back to the
    /// target and counts the frames in [`Stats::overruns`].
    pub max_latency: Duration,
    /// How much the device should buffer, or `None` for the backend's
    /// low-latency default.
    pub device_latency: Option<Duration>,
    /// What to do when there is no device to open. `false`: opening fails,
    /// as it does for any other reason. `true`: the stream opens anyway and
    /// its frames are taken in real time and discarded, as the null backend
    /// does, until a device appears; then it plays there. The guest and the
    /// program need not know which is the case, though
    /// [`Output::device`] is empty meanwhile. Honoured by the Pulse,
    /// CoreAudio and WASAPI backends; the file and null backends have no
    /// device to miss.
    pub null_fallback: bool,
    /// How far the clock follower may bend the rate, as a fraction: 0.005
    /// is half a percent, under a tenth of a semitone. Zero turns the
    /// follower off, and then equal guest and device rates are copied
    /// through untouched.
    pub max_adjust: f64,
    /// Also write what the device is given to this WAV file.
    pub tee: Option<PathBuf>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            name: "iris-audio".into(),
            channels: 2,
            rate: 48000,
            max_rate: 96000,
            prebuffer: Duration::from_millis(10),
            max_latency: Duration::from_millis(100),
            device_latency: None,
            null_fallback: false,
            max_adjust: 0.005,
            tee: None,
        }
    }
}

impl OutputConfig {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        let bad = |m: String| Err(Error::Config(m));
        if !(1..=MAX_CHANNELS).contains(&self.channels) {
            return bad(format!("{} channels; 1 to {MAX_CHANNELS} are supported", self.channels));
        }
        if self.max_rate < MIN_RATE || !(MIN_RATE..=self.max_rate).contains(&self.rate) {
            return bad(format!("rate {} Hz is outside {MIN_RATE}..={} Hz", self.rate, self.max_rate));
        }
        if self.max_latency <= self.prebuffer {
            return bad(format!("max_latency {:?} must exceed prebuffer {:?}", self.max_latency, self.prebuffer));
        }
        if !(0.0..=0.05).contains(&self.max_adjust) {
            return bad(format!("max_adjust {} is outside 0..=0.05", self.max_adjust));
        }
        Ok(())
    }
}

/// What the two ends share.
#[derive(Default)]
struct Shared {
    /// The guest's rate.
    rate: AtomicU32,
    /// The guest said it has stopped; running dry is not an underrun.
    idle: AtomicBool,
    written: AtomicU64,
    dropped: AtomicU64,
    played: AtomicU64,
    underruns: AtomicU64,
    overruns: AtomicU64,
    device_underruns: AtomicU64,
    /// Guest frames in the ring at the last render.
    level: AtomicU32,
    /// The clock follower's adjustment, an `f32`'s bits.
    adjust: AtomicU32,
    playing: AtomicBool,
    /// Microseconds, or `u64::MAX` if the backend cannot tell.
    device_delay: AtomicU64,
    device_rate: AtomicU32,
    device_channels: AtomicU32,
}

/// A snapshot of a stream's counters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    /// Guest frames accepted by [`Output::write`].
    pub written: u64,
    /// Guest frames [`Output::write`] had no room for.
    pub dropped: u64,
    /// Device frames of sound (not silence) rendered.
    pub played: u64,
    /// Times the ring ran dry while the guest was playing.
    pub underruns: u64,
    /// Guest frames skipped because the ring exceeded
    /// [`OutputConfig::max_latency`].
    pub overruns: u64,
    /// Times the device itself ran dry; see
    /// [`RenderInfo::device_underruns`].
    pub device_underruns: u64,
    /// Sound waiting in the ring at the last render.
    pub queued: Duration,
    /// The device's own delay, if the backend can tell.
    pub device_delay: Option<Duration>,
    /// The clock follower's current adjustment: +0.001 plays 0.1% fast.
    pub adjust: f64,
    /// Whether sound is flowing (false while prebuffering or silent).
    pub playing: bool,
    /// The device's format, once the backend has reported it.
    pub device: Option<DeviceFormat>,
}

/// The guest's end of an output stream. One thread owns it; writes never
/// block and never allocate.
pub struct Output {
    producer: Producer<f32>,
    shared: Arc<Shared>,
    channels: usize,
    max_rate: u32,
    idle: bool,
    // Dropped with the Output, which stops the backend's stream; with
    // the last Output of the voices mixed into one stream (see `mix`).
    handle: Option<Arc<Mutex<Box<dyn StreamHandle>>>>,
}

impl Output {
    /// The two ends of a stream with no backend between them: the caller
    /// drives the [`Renderer`] itself. For tests, and for hosts that have
    /// their own audio callback.
    pub fn detached(config: &OutputConfig) -> Result<(Output, Renderer), Error> {
        config.validate()?;
        let ch = config.channels;
        // Twice the ceiling, so the renderer and not the ring decides what
        // is too much. A multiple of the frame size keeps the ring's wrap
        // on a frame boundary.
        let frames = (config.max_latency.as_secs_f64() * 2.0 * f64::from(config.max_rate)).ceil() as usize + 64;
        let (producer, consumer) = RingBuffer::new(frames * ch);
        let shared = Arc::new(Shared::default());
        shared.rate.store(config.rate, Relaxed);
        shared.device_delay.store(u64::MAX, Relaxed);
        let output = Output { producer, shared: shared.clone(), channels: ch, max_rate: config.max_rate, idle: false, handle: None };
        let renderer = Renderer {
            consumer,
            shared,
            channels: ch,
            prebuffer: config.prebuffer.as_secs_f64(),
            max_latency: config.max_latency.as_secs_f64(),
            max_adjust: config.max_adjust,
            format: DeviceFormat { rate: config.rate, channels: ch },
            resampler: Resampler::new(ch, config.rate, config.rate),
            playing: false,
            stalled: 0,
            last_level: 0,
            level: 0.0,
            period: 0.0,
        };
        Ok((output, renderer))
    }

    pub(crate) fn attach(&mut self, handle: Arc<Mutex<Box<dyn StreamHandle>>>) {
        self.handle = Some(handle);
    }

    /// Queue interleaved frames at the current rate. Returns the frames
    /// taken: all of them unless the ring is full, and then the rest are
    /// dropped whole and counted in [`Stats::dropped`]. A guest should not
    /// wait and retry; the renderer is already skipping ahead.
    pub fn write<S: Sample>(&mut self, samples: &[S]) -> usize {
        let ch = self.channels;
        debug_assert_eq!(samples.len() % ch, 0, "a write must be whole frames");
        let want = samples.len() / ch;
        let take = want.min(self.producer.slots() / ch);
        if take > 0 {
            if let Ok(chunk) = self.producer.write_chunk_uninit(take * ch) {
                chunk.fill_from_iter(samples[..take * ch].iter().map(|s| s.to_f32()));
            }
            self.shared.written.fetch_add(take as u64, Relaxed);
        }
        if take < want {
            self.shared.dropped.fetch_add((want - take) as u64, Relaxed);
        }
        if self.idle {
            self.idle = false;
            self.shared.idle.store(false, Relaxed);
        }
        take
    }

    /// The guest reprogrammed its codec. Frames written from now on are at
    /// `rate`; the few milliseconds still queued play at it too.
    pub fn set_rate(&mut self, rate: u32) -> Result<(), Error> {
        if !(MIN_RATE..=self.max_rate).contains(&rate) {
            return Err(Error::Config(format!("rate {rate} Hz is outside {MIN_RATE}..={} Hz", self.max_rate)));
        }
        self.shared.rate.store(rate, Relaxed);
        Ok(())
    }

    /// The guest stopped its DAC. What is queued plays out at once, even
    /// if it is less than the prebuffer, and the silence that follows is
    /// not an underrun. The next [`Output::write`] ends the idle state.
    pub fn idle(&mut self) {
        self.idle = true;
        self.shared.idle.store(true, Relaxed);
    }

    pub fn rate(&self) -> u32 { self.shared.rate.load(Relaxed) }
    pub fn channels(&self) -> usize { self.channels }

    /// Frames a write would take right now.
    pub fn space(&self) -> usize { self.producer.slots() / self.channels }

    /// What the stream plays to, if a backend is attached.
    pub fn device(&self) -> Option<String> { self.handle.as_ref().map(|h| h.lock().unwrap_or_else(|e| e.into_inner()).device()) }

    pub fn stats(&self) -> Stats {
        let s = &self.shared;
        let rate = f64::from(s.rate.load(Relaxed).max(1));
        let delay = s.device_delay.load(Relaxed);
        let (dev_rate, dev_ch) = (s.device_rate.load(Relaxed), s.device_channels.load(Relaxed));
        Stats {
            written: s.written.load(Relaxed),
            dropped: s.dropped.load(Relaxed),
            played: s.played.load(Relaxed),
            underruns: s.underruns.load(Relaxed),
            overruns: s.overruns.load(Relaxed),
            device_underruns: s.device_underruns.load(Relaxed),
            queued: Duration::from_secs_f64(f64::from(s.level.load(Relaxed)) / rate),
            device_delay: (delay != u64::MAX).then(|| Duration::from_micros(delay)),
            adjust: f64::from(f32::from_bits(s.adjust.load(Relaxed))),
            playing: s.playing.load(Relaxed),
            device: (dev_rate != 0).then_some(DeviceFormat { rate: dev_rate, channels: dev_ch as usize }),
        }
    }
}

/// The device's end of an output stream: the [`Render`] a backend calls.
pub struct Renderer {
    consumer: Consumer<f32>,
    shared: Arc<Shared>,
    /// Samples per guest frame.
    channels: usize,
    /// Seconds.
    prebuffer: f64,
    max_latency: f64,
    max_adjust: f64,
    format: DeviceFormat,
    resampler: Resampler,
    /// Sound is flowing. False while waiting for the prebuffer.
    playing: bool,
    /// Device frames rendered while waiting with the ring's level unchanged.
    stalled: usize,
    last_level: usize,
    /// The smoothed ring level the clock follower steers by, guest frames.
    level: f64,
    /// The device's period in guest frames: the largest recent request.
    period: f64,
}

/// Put a guest frame of `src_ch` channels into a device frame.
#[inline]
fn map_frame(src: &[f32; MAX_CHANNELS], src_ch: usize, dst: &mut [f32]) {
    let dst_ch = dst.len();
    if dst_ch == src_ch {
        dst.copy_from_slice(&src[..src_ch]);
    } else if src_ch == 1 {
        // Mono to both front speakers.
        dst.fill(0.0);
        dst[0] = src[0];
        if dst_ch > 1 {
            dst[1] = src[0];
        }
    } else if dst_ch > src_ch {
        dst[..src_ch].copy_from_slice(&src[..src_ch]);
        dst[src_ch..].fill(0.0);
    } else if dst_ch == 1 {
        dst[0] = src[..src_ch].iter().sum::<f32>() / src_ch as f32;
    } else if dst_ch == 2 {
        // Fold every left-side channel into left and every right-side one
        // into right. Good enough until a backend cannot do better itself.
        let (mut l, mut r) = (0.0, 0.0);
        for pair in src[..src_ch].chunks(2) {
            l += pair[0];
            r += *pair.get(1).unwrap_or(&pair[0]);
        }
        let n = src_ch.div_ceil(2) as f32;
        dst[0] = l / n;
        dst[1] = r / n;
    } else {
        dst.copy_from_slice(&src[..dst_ch]);
    }
}

impl Renderer {
    /// The format last configured.
    pub fn format(&self) -> DeviceFormat { self.format }

    fn publish(&self, level: usize, adjust: f64) {
        self.shared.level.store(level as u32, Relaxed);
        self.shared.adjust.store((adjust as f32).to_bits(), Relaxed);
        self.shared.playing.store(self.playing, Relaxed);
    }
}

impl Render for Renderer {
    fn configure(&mut self, format: DeviceFormat) {
        assert!(format.rate > 0 && (1..=MAX_CHANNELS).contains(&format.channels));
        self.format = format;
        self.resampler.set_rates(self.resampler.in_rate(), format.rate);
        self.period = 0.0;
        self.shared.device_rate.store(format.rate, Relaxed);
        self.shared.device_channels.store(format.channels as u32, Relaxed);
    }

    fn render(&mut self, out: &mut [f32], info: &RenderInfo) {
        let (gch, och) = (self.channels, self.format.channels);
        let frames = out.len() / och;
        let shared = &*self.shared;
        shared.device_delay.store(info.device_delay.map_or(u64::MAX, |d| d.as_micros() as u64), Relaxed);
        shared.device_underruns.store(info.device_underruns, Relaxed);

        let in_rate = shared.rate.load(Relaxed);
        if in_rate != self.resampler.in_rate() {
            // What the ring holds is now so many seconds of the new rate;
            // let the level settle again from where it is.
            self.resampler.set_rates(in_rate, self.format.rate);
            self.level = (self.consumer.slots() / gch) as f64;
        }
        let (in_hz, out_hz) = (f64::from(in_rate), f64::from(self.format.rate));
        let nominal = in_hz / out_hz;

        // The device takes a period at a time, so the ring has to hold a
        // period more than the margin asked for. Periods vary (a server
        // may ask for odd amounts); follow the largest recent one.
        // While waiting to start only this request counts: a server's
        // first, oversized fill would otherwise set the starting level
        // far above where the stream then settles.
        let asked = frames as f64 * nominal;
        self.period = if self.playing { asked.max(self.period * 0.99) } else { asked };
        let target = self.prebuffer * in_hz + self.period;
        let fill = self.consumer.slots() / gch;
        let idle = shared.idle.load(Relaxed);

        if !self.playing {
            // A sound shorter than the prebuffer never reaches the target.
            // Once the guest has stopped adding to it for a prebuffer's
            // time, or has said it is done, play what there is.
            if fill == self.last_level {
                self.stalled += frames;
            } else {
                self.stalled = 0;
            }
            self.last_level = fill;
            let patience = (self.prebuffer * out_hz) as usize;
            let start = fill as f64 >= target || (fill > 0 && (idle || self.stalled > patience));
            if !start {
                out.fill(0.0);
                self.publish(fill, 0.0);
                return;
            }
            self.playing = true;
            self.level = fill as f64;
        }

        // Far too much queued: the guest ran ahead (or the device stood
        // still) by more than a rate adjustment can make up. Skip to the
        // target once instead of playing everything late.
        let mut fill = fill;
        let ceiling = (self.max_latency * in_hz).max(2.0 * target);
        if fill as f64 > ceiling {
            let skip = fill - target.round() as usize;
            if let Ok(chunk) = self.consumer.read_chunk(skip * gch) {
                chunk.commit_all();
            }
            shared.overruns.fetch_add(skip as u64, Relaxed);
            fill -= skip;
            self.level = fill as f64;
        }

        // Follow the guest's clock: play a little fast when the ring is
        // above the target, a little slow when it is below.
        let follow = self.max_adjust > 0.0;
        let adjust = if follow {
            let dt = frames as f64 / out_hz;
            self.level += (1.0 - (-dt / LEVEL_SMOOTHING).exp()) * (fill as f64 - self.level);
            ((self.level - target) / in_hz * FOLLOW_GAIN).clamp(-self.max_adjust, self.max_adjust)
        } else {
            0.0
        };

        let avail = fill * gch;
        let produced = match self.consumer.read_chunk(avail) {
            Ok(chunk) => {
                let (a, b) = chunk.as_slices();
                debug_assert_eq!(a.len() % gch, 0, "the ring wraps on a frame boundary");
                let mut taken = 0;
                let next = |f: &mut [f32; MAX_CHANNELS]| {
                    if taken >= avail {
                        return false;
                    }
                    let s = if taken < a.len() { &a[taken..taken + gch] } else { &b[taken - a.len()..][..gch] };
                    f[..gch].copy_from_slice(s);
                    taken += gch;
                    true
                };
                let emit = |n: usize, f: &[f32; MAX_CHANNELS]| map_frame(f, gch, &mut out[n * och..(n + 1) * och]);
                let produced = if !follow && in_rate == self.format.rate {
                    copy_frames(frames, next, emit)
                } else {
                    self.resampler.process(nominal * (1.0 + adjust), frames, next, emit)
                };
                chunk.commit(taken);
                produced
            }
            Err(_) => 0,
        };
        shared.played.fetch_add(produced as u64, Relaxed);

        if produced < frames {
            // Ran dry. Silence for the rest, and wait for the prebuffer
            // again: starting on the first frame that arrives would only
            // run dry again at once.
            out[produced * och..].fill(0.0);
            self.playing = false;
            self.stalled = 0;
            self.last_level = 0;
            if !idle {
                shared.underruns.fetch_add(1, Relaxed);
            }
        } else {
            // A buffer that is not whole frames: the tail is silence.
            out[frames * och..].fill(0.0);
        }
        self.publish(fill, adjust);
    }
}

/// Equal rates and no clock follower: the guest's frames, untouched.
#[inline]
fn copy_frames(
    frames: usize,
    mut next: impl FnMut(&mut [f32; MAX_CHANNELS]) -> bool,
    mut emit: impl FnMut(usize, &[f32; MAX_CHANNELS]),
) -> usize {
    let mut f = [0.0; MAX_CHANNELS];
    for n in 0..frames {
        if !next(&mut f) {
            return n;
        }
        emit(n, &f);
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resample::tests::{amplitude, rms};

    const DEVICE: DeviceFormat = DeviceFormat { rate: 48000, channels: 2 };

    fn open(config: OutputConfig) -> (Output, Renderer) {
        let (o, mut r) = Output::detached(&config).unwrap();
        r.configure(DEVICE);
        (o, r)
    }

    /// A guest and a device stepped through simulated time: the guest
    /// writes `burst` frames whenever that many are due by its clock
    /// (`guest_hz` real frames a second, whatever rate it claims), the
    /// device renders `period` frames at a time.
    struct Sim {
        out: Output,
        r: Renderer,
        guest_hz: f64,
        burst: usize,
        period: usize,
        written: u64,
        phase: f64,
        heard: Vec<f32>,
    }

    impl Sim {
        fn new(config: OutputConfig, guest_hz: f64, burst: usize, period: usize) -> Self {
            let (out, r) = open(config);
            Self { out, r, guest_hz, burst, period, written: 0, phase: 0.0, heard: Vec::new() }
        }

        /// Run for `secs`, the guest playing a 440 Hz tone.
        fn run(&mut self, secs: f64) {
            let ch = self.out.channels();
            let rate = f64::from(self.out.rate());
            let periods = (secs * f64::from(DEVICE.rate) / self.period as f64) as usize;
            let mut buf = vec![0.0f32; self.period * DEVICE.channels];
            let mut frame = vec![0.0f32; self.burst * ch];
            let start = self.heard.len() / DEVICE.channels;
            for p in 0..periods {
                let now = (start + (p + 1) * self.period) as f64 / f64::from(DEVICE.rate);
                while (self.written + self.burst as u64) as f64 <= now * self.guest_hz {
                    for f in frame.chunks_mut(ch) {
                        f.fill((self.phase.sin() * 0.5) as f32);
                        self.phase += 2.0 * std::f64::consts::PI * 440.0 / rate;
                    }
                    self.out.write(&frame);
                    self.written += self.burst as u64;
                }
                self.r.render(&mut buf, &RenderInfo::default());
                self.heard.extend_from_slice(&buf);
            }
        }
    }

    #[test]
    fn nothing_plays_until_the_prebuffer_is_reached() {
        let cfg = OutputConfig { prebuffer: Duration::from_millis(20), ..Default::default() };
        let (mut o, mut r) = open(cfg);
        let mut buf = [1.0f32; 480 * 2];
        // 10 ms queued, 20 ms plus a 10 ms period wanted.
        o.write(&[0.5f32; 480 * 2]);
        r.render(&mut buf, &RenderInfo::default());
        assert!(buf.iter().all(|&v| v == 0.0));
        assert!(!o.stats().playing);
        o.write(&[0.5f32; 960 * 2]);
        r.render(&mut buf, &RenderInfo::default());
        assert!(o.stats().playing);
        r.render(&mut buf, &RenderInfo::default());
        assert!(buf.iter().all(|&v| (v - 0.5).abs() < 0.01), "{:?}", &buf[..4]);
    }

    #[test]
    fn a_sound_shorter_than_the_prebuffer_still_plays() {
        // A 5 ms blip against a 20 ms prebuffer: it plays once the guest
        // has added nothing for 20 ms, and its end is an underrun only
        // because the guest never said it stopped.
        let cfg = OutputConfig { prebuffer: Duration::from_millis(20), ..Default::default() };
        let (mut o, mut r) = open(cfg);
        o.write(&[0.5f32; 240 * 2]);
        let mut heard = Vec::new();
        let mut buf = [0.0f32; 240 * 2];
        for _ in 0..12 {
            r.render(&mut buf, &RenderInfo::default());
            heard.extend_from_slice(&buf);
        }
        let loud = heard.iter().filter(|v| v.abs() > 0.25).count() / 2;
        assert!((200..=260).contains(&loud), "{loud} loud frames");
        assert_eq!(o.stats().underruns, 1);
    }

    #[test]
    fn an_idle_guest_plays_out_at_once_and_is_not_an_underrun() {
        let cfg = OutputConfig { prebuffer: Duration::from_millis(20), ..Default::default() };
        let (mut o, mut r) = open(cfg);
        o.write(&[0.5f32; 240 * 2]);
        o.idle();
        let mut buf = [0.0f32; 480 * 2];
        r.render(&mut buf, &RenderInfo::default());
        assert!(buf[..200].iter().any(|v| v.abs() > 0.25));
        assert_eq!(o.stats().underruns, 0);
        assert!(!o.stats().playing);
    }

    #[test]
    fn an_underrun_is_counted_and_playback_waits_for_the_prebuffer_again() {
        let mut sim = Sim::new(OutputConfig::default(), 48000.0, 64, 240);
        sim.run(0.5);
        assert_eq!(sim.out.stats().underruns, 0);
        // The guest stalls for 100 ms.
        let mut buf = [0.0f32; 240 * 2];
        for _ in 0..20 {
            sim.r.render(&mut buf, &RenderInfo::default());
        }
        assert_eq!(sim.out.stats().underruns, 1);
        assert!(!sim.out.stats().playing);
        // One frame is not enough to start again.
        sim.out.write(&[0.1f32; 2]);
        sim.r.render(&mut buf, &RenderInfo::default());
        assert!(buf.iter().all(|&v| v == 0.0));
        assert_eq!(sim.out.stats().underruns, 1);
    }

    #[test]
    fn a_full_ring_drops_whole_frames() {
        let (mut o, mut r) = open(OutputConfig { channels: 2, max_adjust: 0.0, ..Default::default() });
        let space = o.space();
        let frames: Vec<f32> = (0..space + 10).flat_map(|_| [0.25f32, -0.25]).collect();
        assert_eq!(o.write(&frames), space);
        assert_eq!(o.write(&[0.25f32, -0.25]), 0);
        assert_eq!(o.stats().dropped, 11);
        // Left is still left after the renderer skips down to its target.
        let mut buf = [0.0f32; 480 * 2];
        r.render(&mut buf, &RenderInfo::default());
        assert!(o.stats().overruns > 0);
        for f in buf.chunks(2) {
            assert_eq!(f, [0.25, -0.25]);
        }
    }

    #[test]
    fn equal_rates_without_the_follower_are_copied_exactly() {
        let (mut o, mut r) = open(OutputConfig { max_adjust: 0.0, prebuffer: Duration::ZERO, ..Default::default() });
        let pcm: Vec<i16> = (0..960).map(|i| (i * 31 % 2000 - 1000) as i16).collect();
        o.write(&pcm);
        let mut buf = [0.0f32; 480 * 2];
        r.render(&mut buf, &RenderInfo::default());
        let back: Vec<i16> = buf.iter().map(|&v| i16::from_f32(v)).collect();
        assert_eq!(back, pcm);
    }

    /// From `skip` frames in, `heard` (stereo) is a half-scale tone with
    /// nothing missing: the level of a whole sine, which a gap would lower.
    /// The follower moves the pitch by a hair, so this measures level and
    /// not one exact frequency.
    fn assert_whole_tone(heard: &[f32], skip: usize) {
        let left: Vec<f32> = heard.chunks(2).skip(skip).map(|f| f[0]).collect();
        let level = rms(&left) * std::f64::consts::SQRT_2;
        assert!((level - 0.5).abs() < 0.005, "tone level {level}");
    }

    /// The clock follower's case: whatever the guest's pace, no underruns
    /// and no overruns after start-up, and the level stays near the target.
    fn follows(guest_hz: f64, burst: usize, period: usize, prebuffer_ms: u64) -> Stats {
        let cfg = OutputConfig { prebuffer: Duration::from_millis(prebuffer_ms), ..Default::default() };
        let mut sim = Sim::new(cfg, guest_hz, burst, period);
        sim.run(60.0);
        let s = sim.out.stats();
        assert_eq!((s.underruns, s.overruns, s.dropped), (0, 0, 0), "{s:?}");
        // The tone is whole: no gaps after the start.
        assert_whole_tone(&sim.heard, 48000);
        s
    }

    #[test]
    fn a_trickling_guest_with_a_fast_clock_is_followed() {
        // HAL2's way: four frames at a time, and a clock 0.2% fast.
        let s = follows(48000.0 * 1.002, 4, 256, 10);
        assert!((s.adjust - 0.002).abs() < 0.0005, "{s:?}");
        assert!(s.queued < Duration::from_millis(25), "{s:?}");
    }

    #[test]
    fn a_trickling_guest_with_a_slow_clock_is_followed() {
        let s = follows(48000.0 * 0.998, 4, 256, 10);
        assert!((s.adjust + 0.002).abs() < 0.0005, "{s:?}");
    }

    #[test]
    fn a_guest_writing_large_blocks_is_followed() {
        // 1024 frames at a time (21 ms) into a device taking 5 ms periods.
        follows(48000.0 * 1.001, 1024, 240, 30);
    }

    #[test]
    fn a_large_device_period_is_covered_without_asking() {
        // A 20 ms device period against a 5 ms prebuffer: the target grows
        // by the period, so it still never runs dry.
        follows(48000.0, 8, 960, 5);
    }

    #[test]
    fn a_guest_at_another_rate_is_converted_and_followed() {
        let cfg = OutputConfig { rate: 11025, ..Default::default() };
        let mut sim = Sim::new(cfg, 11025.0 * 1.001, 16, 256);
        sim.run(20.0);
        let s = sim.out.stats();
        assert_eq!((s.underruns, s.overruns), (0, 0), "{s:?}");
        assert_whole_tone(&sim.heard, 48000);
        // And it is the tone the guest played, a little sharp by its clock.
        let left: Vec<f32> = sim.heard.chunks(2).skip(15 * 48000).take(4800).map(|f| f[0]).collect();
        assert!(amplitude(&left, 48000, 440.0 * 1.001) > 0.45);
    }

    #[test]
    fn the_guest_can_change_its_rate_while_playing() {
        let mut sim = Sim::new(OutputConfig::default(), 48000.0, 16, 256);
        sim.run(2.0);
        sim.out.set_rate(22050).unwrap();
        sim.guest_hz = 22050.0;
        sim.written = (sim.heard.len() / 2) as u64 * 22050 / 48000;
        sim.run(5.0);
        let s = sim.out.stats();
        assert_eq!((s.underruns, s.overruns), (0, 0), "{s:?}");
        assert_whole_tone(&sim.heard, 5 * 48000);
        assert!(sim.out.set_rate(500_000).is_err());
    }

    #[test]
    fn a_guest_running_far_ahead_is_cut_back_to_the_target() {
        // 5% fast: ten times what the follower may absorb.
        let mut sim = Sim::new(OutputConfig::default(), 48000.0 * 1.05, 16, 256);
        sim.run(20.0);
        let s = sim.out.stats();
        assert!(s.overruns > 0 && s.underruns == 0 && s.dropped == 0, "{s:?}");
        assert!(s.queued <= Duration::from_millis(100), "{s:?}");
    }

    #[test]
    fn channels_are_mapped_to_the_device() {
        let f = |src: &[f32], dst_ch: usize| {
            let mut frame = [0.0; MAX_CHANNELS];
            frame[..src.len()].copy_from_slice(src);
            let mut dst = vec![9.0; dst_ch];
            map_frame(&frame, src.len(), &mut dst);
            dst
        };
        assert_eq!(f(&[0.5], 2), [0.5, 0.5]);
        assert_eq!(f(&[0.5], 6), [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(f(&[0.2, 0.4], 4), [0.2, 0.4, 0.0, 0.0]);
        assert_eq!(f(&[0.2, 0.4], 1), [0.3]);
        assert_eq!(f(&[0.2, 0.4, 0.6, 0.8], 2), [0.4, 0.6]);
    }

    #[test]
    fn a_bad_config_is_refused() {
        let bad = |c: OutputConfig| Output::detached(&c).is_err();
        assert!(bad(OutputConfig { channels: 0, ..Default::default() }));
        assert!(bad(OutputConfig { channels: 9, ..Default::default() }));
        assert!(bad(OutputConfig { rate: 200_000, ..Default::default() }));
        assert!(bad(OutputConfig { prebuffer: Duration::from_millis(200), ..Default::default() }));
    }
}
