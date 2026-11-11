//! What a PCI audio card with a host-timed sample clock shares with the
//! others: the host output (a stereo [`Sink`] with a WAV capture of it, and
//! the multichannel [`PcmOut`] with its [`Downmix`]), and the clock that
//! stands still while the machine is stopped. The host streams themselves
//! are in [`crate::host_audio`].
//!
//! RAD1 was the first such card; a second one can use the same parts so the
//! two do not drift apart.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::host_audio::{FrameResampler, HostOut, Resampler, WavWriter};

/// Words a host frame carries: the mixer takes two channels.
pub const PCM_CHANNELS: usize = 2;
/// The WAV capture's rate.
const WAV_RATE: u32 = 44100;
/// Most of the card's clock held back for hold-ups, and the share of an
/// interval won back per on-time completion (see [`CardClock::absorb_stall`]).
const STALL_MAX_NS: i64 = 200_000_000;
const STALL_REPAY_DIV: i64 = 32;

/// Where decoded output goes: the host stream (through the shared resampler),
/// an optional 44.1 kHz WAV file of what the card played, and for tests a
/// plain buffer of the card's own frames.
pub struct Sink {
    pub host: Option<(HostOut, Resampler)>,
    /// When the host stream was last tried and failed to open.
    pub host_failed_at: Option<Instant>,
    pub wav: Option<Wav>,
    pub tap: Option<Arc<Mutex<Vec<i16>>>>,
}

/// How long to wait before trying to open the host stream again.
pub const HOST_RETRY: Duration = Duration::from_secs(1);

/// Whether a host stream wanted but not open (`closed`) is due another try.
fn retry_due(closed: bool, failed_at: Option<Instant>, now: Instant) -> bool {
    closed && failed_at.is_none_or(|t| now.duration_since(t) >= HOST_RETRY)
}

impl Sink {
    /// Whether the host stream, wanted but not open, is due another try.
    pub fn host_retry_due(&self, now: Instant) -> bool {
        retry_due(self.host.is_none(), self.host_failed_at, now)
    }

    /// Play stereo frames the card produced at `rate` Hz. A muted output
    /// still moves the ring; the host and the WAV get silence. Returns the
    /// host ring's free and total samples.
    pub fn play(&mut self, pcm: &[i16], rate: u32, muted: bool) -> Option<(usize, usize)> {
        let silence;
        let host_pcm = if muted {
            silence = vec![0i16; pcm.len()];
            &silence[..]
        } else {
            pcm
        };
        if let Some((out, rs)) = self.host.as_mut() {
            if rs.in_rate != rate {
                *rs = Resampler::new(rate, out.stream_rate);
            }
            for p in host_pcm.chunks_exact(PCM_CHANNELS) {
                rs.push(p[0], p[1], &mut out.producer);
            }
        }
        if let Some(w) = self.wav.as_mut() {
            w.play(host_pcm, rate);
        }
        if let Some(t) = self.tap.as_ref() {
            t.lock().extend_from_slice(pcm);
        }
        self.host.as_ref().map(|(out, _)| (out.producer.slots(), out.producer.buffer().capacity()))
    }
}

/// A 44.1 kHz stereo 16-bit WAV of the output, converted from the card's
/// rate, its sizes rewritten after every append so a file from a killed run
/// still plays. A debugging aid, set by the card that owns the sink.
pub struct Wav {
    out: WavWriter,
    rs: Resampler,
}

impl Wav {
    pub fn create(path: &std::ffi::OsStr) -> Option<Self> {
        let out = WavWriter::create(std::path::Path::new(path), WAV_RATE, PCM_CHANNELS as u16).ok()?;
        Some(Self { out, rs: Resampler::new(WAV_RATE, WAV_RATE) })
    }

    /// Append frames the card produced at `rate` Hz.
    fn play(&mut self, pcm: &[i16], rate: u32) {
        if self.rs.in_rate != rate {
            self.rs = Resampler::new(rate, WAV_RATE);
        }
        if self.rs.passthrough() {
            self.append(pcm);
            return;
        }
        // The resampler writes into a ring; take its output back out.
        let frames = pcm.len() / PCM_CHANNELS;
        let room = (frames * WAV_RATE as usize / rate.max(1) as usize + 2) * PCM_CHANNELS;
        let (mut prod, mut cons) = rtrb::RingBuffer::<i16>::new(room);
        for p in pcm.chunks_exact(PCM_CHANNELS) {
            self.rs.push(p[0], p[1], &mut prod);
        }
        let out: Vec<i16> = std::iter::from_fn(|| cons.pop().ok()).collect();
        self.append(&out);
    }

    pub fn append(&mut self, pcm: &[i16]) {
        let _ = self.out.append(pcm);
    }
}

// ---- multichannel output ----

// Speaker positions, numbered in the order of the usual host layouts.
const FL: u8 = 0;
const FR: u8 = 1;
const FC: u8 = 2;
const LFE: u8 = 3;
const BL: u8 = 4;
const BR: u8 = 5;
const SL: u8 = 6;
const SR: u8 = 7;
/// Gain of 1.0 and of -3 dB in Q15.
const UNITY: i32 = 1 << 15;
const MINUS_3DB: i32 = 23170;

/// The speakers fed by the channels of a frame of `n` channels, in channel
/// order: stereo, quad, 5.1 and 7.1, the usual host layouts.
fn speakers(n: usize) -> &'static [u8] {
    match n {
        4 => &[FL, FR, BL, BR],
        6 => &[FL, FR, FC, LFE, BL, BR],
        8 => &[FL, FR, FC, LFE, BL, BR, SL, SR],
        _ => &[FL, FR],
    }
}

/// How a frame of `src` channels becomes a frame of `dst` channels: a
/// speaker the destination has plays there, and one it lacks folds into the
/// ones it has by the ITU-R BS.775 stereo matrix (centre and surrounds at
/// -3 dB, the low-frequency channel dropped). A destination wider than the
/// source leaves its extra speakers silent. Nothing is scaled down: a
/// speaker plays its own channel at unity, so stereo carried in the first
/// two of eight slots keeps its level, and a fold that exceeds full scale
/// saturates.
pub struct Downmix {
    src: usize,
    dst: usize,
    /// (destination channel, source channel, Q15 gain).
    terms: Vec<(u8, u8, i32)>,
}

impl Downmix {
    pub fn new(src: usize, dst: usize) -> Self {
        let (sr, dr) = (speakers(src), speakers(dst));
        let has = |r: u8| dr.contains(&r);
        let mut terms = Vec::new();
        for (slot, &r) in sr.iter().enumerate() {
            let to: Vec<(u8, i32)> = match r {
                _ if has(r) => vec![(r, UNITY)],
                FC => vec![(FL, MINUS_3DB), (FR, MINUS_3DB)],
                BL => vec![(FL, MINUS_3DB)],
                BR => vec![(FR, MINUS_3DB)],
                SL if has(BL) => vec![(BL, UNITY)],
                SR if has(BR) => vec![(BR, UNITY)],
                SL => vec![(FL, MINUS_3DB)],
                SR => vec![(FR, MINUS_3DB)],
                _ => vec![],
            };
            for (role, gain) in to {
                let d = dr.iter().position(|&x| x == role).unwrap_or(0);
                terms.push((d as u8, slot as u8, gain));
            }
        }
        Self { src, dst, terms }
    }

    /// Mix one source frame into one destination frame.
    #[inline]
    pub fn mix(&self, src: &[i16], dst: &mut [i16]) {
        // Unscaled folds can sum past what 32 bits hold in Q15.
        let mut acc = [1i64 << 14; FrameResampler::MAX_CHANNELS];
        for &(d, s, g) in &self.terms {
            acc[d as usize] += i64::from(src[s as usize]) * i64::from(g);
        }
        for (o, a) in dst.iter_mut().zip(acc) {
            *o = (a >> 15).clamp(i16::MIN as i64, i16::MAX as i64) as i16;
        }
    }
}

/// Where the ICE1724's output goes: the multichannel stream and the S/PDIF
/// stream the card plays, mixed into one host stream.
///
/// The engines stage the frames they move here; [`PcmOut::flush`] mixes what
/// both have made due, so the two streams, whose engines start at different
/// moments, still sum sample for sample. The analog frames fold into the
/// host's channel count ([`Downmix`]) and the S/PDIF pair adds to the front
/// pair. Nothing allocates once the buffers have grown to a tick's worth.
pub struct PcmOut {
    pub host: Option<(HostOut, FrameResampler)>,
    /// When the host stream was last tried and failed to open.
    pub host_failed_at: Option<Instant>,
    /// For tests: the mixed frames at the card's rate, as the host stream
    /// would get them, with `tap_channels` to a frame when there is no
    /// host stream.
    #[cfg(test)]
    pub tap: Option<Arc<Mutex<Vec<i16>>>>,
    #[cfg(test)]
    pub tap_channels: usize,
    analog: Vec<i16>,
    analog_slots: usize,
    spdif: Vec<i16>,
    mix: Downmix,
    mixed: Vec<i16>,
    resampled: Vec<i16>,
}

/// Most a stream may wait in staging for the other, in frames: the engines
/// start a little apart, never a half second.
const STAGE_MAX_FRAMES: usize = 1 << 14;

impl PcmOut {
    pub fn new(host: Option<(HostOut, FrameResampler)>, host_failed_at: Option<Instant>) -> Self {
        Self {
            host,
            host_failed_at,
            #[cfg(test)]
            tap: None,
            #[cfg(test)]
            tap_channels: PCM_CHANNELS,
            analog: Vec::new(),
            analog_slots: 0,
            spdif: Vec::new(),
            mix: Downmix::new(2, 2),
            mixed: Vec::new(),
            resampled: Vec::new(),
        }
    }

    /// Whether the host stream, wanted but not open, is due another try.
    pub fn host_retry_due(&self, now: Instant) -> bool {
        retry_due(self.host.is_none(), self.host_failed_at, now)
    }

    /// Where the interleaved engine puts its samples, `slots` to a frame.
    /// Frames staged at another width are dropped.
    pub fn analog_stage(&mut self, slots: usize) -> &mut Vec<i16> {
        if slots != self.analog_slots {
            self.analog.clear();
            self.analog_slots = slots;
        }
        &mut self.analog
    }

    /// Where the S/PDIF engine puts its left and right samples.
    pub fn spdif_stage(&mut self) -> &mut Vec<i16> {
        &mut self.spdif
    }

    /// Mix what is staged and play it at `rate` Hz. A stream whose engine is
    /// not running contributes nothing and loses what it staged. Returns
    /// the frames the host ring had no room for, or `None` without a host
    /// stream.
    pub fn flush(&mut self, rate: u32, analog_on: bool, spdif_on: bool) -> Option<u64> {
        if !analog_on {
            self.analog.clear();
        }
        if !spdif_on {
            self.spdif.clear();
        }
        let slots = self.analog_slots.max(1);
        let (a, s) = (self.analog.len() / slots, self.spdif.len() / 2);
        let n = match (analog_on, spdif_on) {
            (true, true) => a.min(s),
            (true, false) => a,
            (false, true) => s,
            _ => 0,
        };
        #[cfg(test)]
        let no_host_channels = self.tap_channels;
        #[cfg(not(test))]
        let no_host_channels = PCM_CHANNELS;
        let chans = self.host.as_ref().map_or(no_host_channels, |(out, _)| out.channels);
        let mut dropped = 0;
        if n > 0 {
            if self.mix.src != slots || self.mix.dst != chans {
                self.mix = Downmix::new(slots, chans);
            }
            self.mixed.clear();
            self.mixed.resize(n * chans, 0);
            if analog_on {
                for (f, o) in self.analog.chunks_exact(slots).zip(self.mixed.chunks_exact_mut(chans)) {
                    self.mix.mix(f, o);
                }
            }
            if spdif_on {
                for (f, o) in self.spdif.as_chunks::<2>().0.iter().zip(self.mixed.chunks_exact_mut(chans)) {
                    for (o, &v) in o.iter_mut().zip(f) {
                        *o = o.saturating_add(v);
                    }
                }
            }
            if analog_on {
                self.analog.drain(..n * slots);
            }
            if spdif_on {
                self.spdif.drain(..n * 2);
            }
            #[cfg(test)]
            if let Some(t) = self.tap.as_ref() {
                t.lock().extend_from_slice(&self.mixed);
            }
            if let Some((out, rs)) = self.host.as_mut() {
                if rs.in_rate != rate {
                    *rs = FrameResampler::new(rate, out.stream_rate, chans);
                }
                self.resampled.clear();
                for f in self.mixed.chunks_exact(chans) {
                    rs.push(f, &mut self.resampled);
                }
                // A frame the ring has no room for is dropped whole, or
                // the channels of every later frame would shift; one that
                // fits is published whole.
                for f in self.resampled.chunks_exact(chans) {
                    if !crate::host_audio::push_frame(&mut out.producer, f) {
                        dropped += 1;
                    }
                }
            }
        }
        // Whatever one stream got ahead by stays for the other, but not
        // without limit.
        for (v, w) in [(&mut self.analog, slots), (&mut self.spdif, 2)] {
            if v.len() > STAGE_MAX_FRAMES * w {
                v.drain(..v.len() - STAGE_MAX_FRAMES * w);
            }
        }
        self.host.is_some().then_some(dropped)
    }
}

/// A card's sample clock: host time less the time the machine was stopped
/// and less the time held back after a late timer. Lock order: `tick_pass`,
/// then `stopped_at` (a leaf, taken by `now`).
pub struct CardClock {
    pub start: Instant,
    /// In tests, the clock in nanoseconds once a test sets it; `i64::MIN`
    /// follows the host.
    pub(crate) fake_ns: AtomicI64,
    /// Nanoseconds the clock stood still while the machine was stopped, and
    /// when the current stop began: the sample clock and whatever is
    /// scheduled from it do not run on while the guest cannot see them.
    pub stopped_ns: AtomicI64,
    pub stopped_at: Mutex<Option<Instant>>,
    /// Nanoseconds taken off the clock for timer hold-ups (see
    /// [`CardClock::absorb_stall`]), and the latest time it has handed out.
    pub stalled_ns: AtomicI64,
    pub seen_ns: AtomicI64,
    /// Held across each timer callback's work, so a stop can wait one out.
    pub tick_pass: Mutex<()>,
    /// Between `Device::start` and `Device::stop`: timers are armed only
    /// then.
    pub armed: AtomicBool,
}

impl Default for CardClock {
    fn default() -> Self {
        Self::new()
    }
}

impl CardClock {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            fake_ns: AtomicI64::new(i64::MIN),
            stopped_ns: AtomicI64::new(0),
            stalled_ns: AtomicI64::new(0),
            seen_ns: AtomicI64::new(i64::MIN),
            stopped_at: Mutex::new(None),
            tick_pass: Mutex::new(()),
            armed: AtomicBool::new(false),
        }
    }

    pub fn now(&self) -> i64 {
        let at = self.stopped_at.lock();
        let t = self.clock_locked(&at);
        self.seen_ns.fetch_max(t, Ordering::SeqCst);
        t
    }

    /// The card's clock, with `stopped_at` held; not recorded as seen.
    pub fn clock_locked(&self, at: &Option<Instant>) -> i64 {
        let at = at.unwrap_or_else(Instant::now);
        let host = at.duration_since(self.start).as_nanos() as i64 - self.stopped_ns.load(Ordering::SeqCst);
        let host = match self.fake_ns.load(Ordering::Relaxed) {
            i64::MIN => host,
            t => t,
        };
        host - self.stalled_ns.load(Ordering::SeqCst)
    }

    /// Whether time passes by itself, so timers may be armed.
    pub fn runs_by_itself(&self) -> bool {
        self.fake_ns.load(Ordering::Relaxed) == i64::MIN
    }

    /// Set the clock to read `now` with `stalled` held back, as a restored
    /// card's clock does.
    pub fn restore(&self, now: i64, stalled: i64) {
        self.stalled_ns.store(stalled, Ordering::SeqCst);
        if self.fake_ns.load(Ordering::Relaxed) != i64::MIN {
            self.fake_ns.store(now + stalled, Ordering::Relaxed);
            self.seen_ns.store(now, Ordering::SeqCst);
            return;
        }
        let at = self.stopped_at.lock().unwrap_or_else(Instant::now);
        let since_start = at.duration_since(self.start).as_nanos() as i64;
        self.stopped_ns.store(since_start - now - stalled, Ordering::SeqCst);
        self.seen_ns.store(now, Ordering::SeqCst);
    }

    /// The card's clock as a completion due at `due` sees it. One more than
    /// `interval` late means the host held up the timer thread (and the
    /// guest's clock interrupt with it); crediting that would replay the
    /// time in one burst. So the clock goes back to `due`, at most
    /// [`STALL_MAX_NS`] in all and never behind a time already handed out.
    /// An on-time one wins back `interval / STALL_REPAY_DIV` of what is held
    /// back. Emulator approximation; see the cards' module notes.
    pub fn absorb_stall(&self, due: i64, interval: i64) -> i64 {
        let at = self.stopped_at.lock();
        let t = self.clock_locked(&at);
        let held = self.stalled_ns.load(Ordering::SeqCst);
        let late = t - due;
        if late > interval {
            crate::pf_instant!(crate::perfetto::events::RAD_LATE, late / 1000);
        }
        let shift = if late > interval {
            let floor = self.seen_ns.load(Ordering::SeqCst);
            -late.min(STALL_MAX_NS - held).min(t.saturating_sub(floor)).max(0)
        } else {
            held.min(interval / STALL_REPAY_DIV)
        };
        self.stalled_ns.fetch_sub(shift, Ordering::SeqCst);
        let t = t + shift;
        self.seen_ns.fetch_max(t, Ordering::SeqCst);
        t
    }

    /// Stop the clock where it is.
    pub fn stop_clock(&self) {
        let mut at = self.stopped_at.lock();
        if at.is_none() {
            *at = Some(Instant::now());
        }
    }

    /// Run the clock again from where it stopped.
    pub fn start_clock(&self) {
        let mut at = self.stopped_at.lock();
        if let Some(t) = at.take() {
            self.stopped_ns.fetch_add(t.elapsed().as_nanos() as i64, Ordering::SeqCst);
        }
    }

    pub fn stopped(&self) -> bool {
        self.stopped_at.lock().is_some()
    }
}

/// The test clock's value, to read and set by hand.
#[cfg(test)]
pub(crate) mod test_time {
    use super::*;

    pub(crate) struct TestTime;

    impl TestTime {
        /// A clock that follows the host until a test sets `fake_ns`.
        pub(crate) fn clock() -> CardClock {
            CardClock::new()
        }
    }

    pub(crate) trait FakeNs {
        fn fake_ns(&self) -> &AtomicI64;
    }

    impl FakeNs for CardClock {
        fn fake_ns(&self) -> &AtomicI64 {
            &self.fake_ns
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One frame through the matrix, as unit gains times `v` on `slot`.
    fn mix_one(src: usize, dst: usize, slot: usize, v: i16) -> Vec<i16> {
        let mut f = vec![0i16; src];
        f[slot] = v;
        let mut o = vec![0i16; dst];
        Downmix::new(src, dst).mix(&f, &mut o);
        o
    }

    #[test]
    fn equal_layouts_pass_through_unchanged() {
        for n in [2, 4, 6, 8] {
            for slot in 0..n {
                let o = mix_one(n, n, slot, -12345);
                assert!(o.iter().enumerate().all(|(i, &v)| v == if i == slot { -12345 } else { 0 }), "{n} {slot} {o:?}");
            }
        }
    }

    /// IRIX plays stereo in the first two of eight slots and leaves the
    /// rest silent; a stereo host plays it at the level it was written.
    #[test]
    fn stereo_in_eight_slots_reaches_a_stereo_host_at_unity() {
        assert_eq!(mix_one(8, 2, 0, 12345), [12345, 0]);
        assert_eq!(mix_one(8, 2, 1, -12345), [0, -12345]);
    }

    #[test]
    fn five_one_folds_to_stereo_with_the_usual_matrix() {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        let near = |a: i16, b: f64| (f64::from(a) - b).abs() <= 1.0;
        // Centre feeds both sides at -3 dB.
        let o = mix_one(6, 2, 2, 10000);
        assert!(near(o[0], 10000.0 * k) && o[0] == o[1], "{o:?}");
        // The low-frequency channel is dropped.
        assert_eq!(mix_one(6, 2, 3, 10000), [0, 0]);
        // Back left folds to the left only.
        let o = mix_one(6, 2, 4, 10000);
        assert!(near(o[0], 10000.0 * k) && o[1] == 0, "{o:?}");
        // Front left plays at unity.
        assert_eq!(mix_one(6, 2, 0, 10000), [10000, 0]);
    }

    #[test]
    fn seven_one_folds_sides_into_the_back_of_a_five_one_host() {
        // Each back speaker takes its side's channel as well as its own,
        // both at unity; the front is untouched.
        assert_eq!(mix_one(8, 6, 6, 8000), [0, 0, 0, 0, 8000, 0]);
        assert_eq!(mix_one(8, 6, 7, 8000), [0, 0, 0, 0, 0, 8000]);
        assert_eq!(mix_one(8, 6, 0, 8000), [8000, 0, 0, 0, 0, 0]);
        // A quad host has no centre: it folds into the front pair.
        let o = mix_one(8, 4, 2, 10000);
        assert!(o[0] > 0 && o[0] == o[1] && o[2] == 0 && o[3] == 0, "{o:?}");
    }

    /// A fold past full scale saturates at the rail; it does not wrap.
    #[test]
    fn a_full_scale_fold_saturates() {
        for (src, dst) in [(8, 2), (6, 2), (8, 4), (4, 2)] {
            let mut o = vec![0i16; dst];
            Downmix::new(src, dst).mix(&vec![i16::MAX; src], &mut o);
            assert!(o.iter().all(|&v| v == i16::MAX), "{src}->{dst} {o:?}");
            Downmix::new(src, dst).mix(&vec![i16::MIN; src], &mut o);
            assert!(o.iter().all(|&v| v == i16::MIN), "{src}->{dst} {o:?}");
        }
    }

    #[test]
    fn spdif_on_top_of_the_analog_pair_saturates() {
        let tap = Arc::new(Mutex::new(Vec::new()));
        let mut out = PcmOut::new(None, None);
        out.tap = Some(tap.clone());
        out.analog_stage(2).extend([0x6000, -0x6000]);
        out.spdif_stage().extend([0x6000, -0x6000]);
        out.flush(48000, true, true);
        // 0xc000 does not fit in 16 bits: it clips, it does not wrap.
        assert_eq!(*tap.lock(), [0x7fff, i16::MIN]);
    }
}
